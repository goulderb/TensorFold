//! Exact model.visual.* layouts used by Mia b5808ac6 GlmVisionTower.
//! File-backed tensors are borrowed; only the validated tensor table and conversion outputs are owned.
const std = @import("std");
const st = @import("core").safetensors;
const Config = @import("cuda_vision_config.zig").Config;
const core = @import("core");
const source = @import("cuda_weight_source.zig");

const Converted = union(enum) { fp8: source.Source, fp4: source.Fp4 };

pub const Root = enum { patch_weight, patch_bias, post_norm, downsample_weight, downsample_bias, projection, projection_norm, projection_bias, gate, up, down };
pub const Block = enum { norm1, qkv_weight, qkv_bias, q_norm, k_norm, attention_weight, attention_bias, norm2, gate_weight, gate_bias, up_weight, up_bias, down_weight, down_bias };
const roots = [_][]const u8{ "patch_embed.proj.weight", "patch_embed.proj.bias", "post_layernorm.weight", "downsample.weight", "downsample.bias", "merger.proj.weight", "merger.post_projection_norm.weight", "merger.post_projection_norm.bias", "merger.gate_proj.weight", "merger.up_proj.weight", "merger.down_proj.weight" };
const blocks = [_][]const u8{ "norm1.weight", "attn.qkv.weight", "attn.qkv.bias", "attn.q_norm.weight", "attn.k_norm.weight", "attn.proj.weight", "attn.proj.bias", "norm2.weight", "mlp.gate_proj.weight", "mlp.gate_proj.bias", "mlp.up_proj.weight", "mlp.up_proj.bias", "mlp.down_proj.weight", "mlp.down_proj.bias" };

pub const Spec = struct {
    name: []const u8,
    shape: [5]usize = @splat(1),
    rank: u8,
    pub fn dims(self: *const Spec) []const usize {
        return self.shape[0..self.rank];
    }
};
pub const Specs = struct {
    c: Config,
    at: usize = 0,
    pub fn init(c: Config) !Specs {
        try c.check();
        return .{ .c = c };
    }
    pub fn count(self: Specs) usize {
        return roots.len + self.c.depth * blocks.len;
    }
    /// `name` borrows buf until the next call; shape is carried by value.
    pub fn next(self: *Specs, buf: []u8) !?Spec {
        if (self.at == self.count()) return null;
        const c = self.c;
        const i = self.at;
        self.at += 1;
        var spec: Spec = .{ .name = "", .rank = 0 };
        var dims: [5]usize = @splat(1);
        if (i < roots.len) {
            spec.name = try std.fmt.bufPrint(buf, "model.visual.{s}", .{roots[i]});
            switch (@as(Root, @fromBackingInt(@intCast(i)))) {
                .patch_weight => {
                    dims = .{ c.hidden, c.channels, c.temporal, c.patch, c.patch };
                    spec.rank = 5;
                },
                .patch_bias, .post_norm => {
                    dims[0] = c.hidden;
                    spec.rank = 1;
                },
                .downsample_weight => {
                    dims = .{ c.output, c.hidden, c.merge, c.merge, 1 };
                    spec.rank = 4;
                },
                .downsample_bias, .projection_norm, .projection_bias => {
                    dims[0] = c.output;
                    spec.rank = 1;
                },
                .projection => {
                    dims[0] = c.output;
                    dims[1] = c.output;
                    spec.rank = 2;
                },
                .gate, .up => {
                    dims[0] = c.projection_intermediate;
                    dims[1] = c.output;
                    spec.rank = 2;
                },
                .down => {
                    dims[0] = c.output;
                    dims[1] = c.projection_intermediate;
                    spec.rank = 2;
                },
            }
        } else {
            const local = i - roots.len;
            const layer = local / blocks.len;
            const kind = local % blocks.len;
            spec.name = try std.fmt.bufPrint(buf, "model.visual.blocks.{d}.{s}", .{ layer, blocks[kind] });
            switch (@as(Block, @fromBackingInt(@intCast(kind)))) {
                .norm1, .norm2, .attention_bias, .down_bias => {
                    dims[0] = c.hidden;
                    spec.rank = 1;
                },
                .q_norm, .k_norm => {
                    dims[0] = c.headDim();
                    spec.rank = 1;
                },
                .qkv_weight => {
                    dims[0] = 3 * c.hidden;
                    dims[1] = c.hidden;
                    spec.rank = 2;
                },
                .qkv_bias => {
                    dims[0] = 3 * c.hidden;
                    spec.rank = 1;
                },
                .attention_weight => {
                    dims[0] = c.hidden;
                    dims[1] = c.hidden;
                    spec.rank = 2;
                },
                .gate_weight, .up_weight => {
                    dims[0] = c.intermediate;
                    dims[1] = c.hidden;
                    spec.rank = 2;
                },
                .gate_bias, .up_bias => {
                    dims[0] = c.intermediate;
                    spec.rank = 1;
                },
                .down_weight => {
                    dims[0] = c.hidden;
                    dims[1] = c.intermediate;
                    spec.rank = 2;
                },
            }
        }
        spec.shape = dims;
        return spec;
    }
};

fn checkTensor(t: st.Tensor, shape: []const usize) !void {
    if (t.dtype != .bf16 and t.dtype != .f16 and t.dtype != .f32) return error.UnsupportedVisionDType;
    if (t.rank != shape.len or !std.mem.eql(usize, t.shape[0..t.rank], shape)) return error.InvalidVisionWeightShape;
    var bytes = t.dtype.size();
    for (shape) |d| bytes = std.math.mul(usize, bytes, d) catch return error.InvalidVisionWeightShape;
    if (t.bytes.len != bytes) return error.InvalidVisionWeightShape;
}

pub const Weights = struct {
    c: Config,
    tensors: []st.Tensor,
    quantized: []?Converted = &.{},

    pub fn loadMixed(a: std.mem.Allocator, c: Config, ck: *core.Checkpoint, table: *const core.modelopt.Table) !Weights {
        var specs = try Specs.init(c);
        const tensors = try a.alloc(st.Tensor, specs.count());
        errdefer a.free(tensors);
        const quantized = try a.alloc(?Converted, specs.count());
        @memset(quantized, null);
        const w: Weights = .{ .c = c, .tensors = tensors, .quantized = quantized };
        errdefer {
            for (quantized) |*q| if (q.*) |*v| if (v.* == .fp4) v.fp4.deinit(a);
            a.free(quantized);
        }
        var buf: [256]u8 = undefined;
        var i: usize = 0;
        while (try specs.next(&buf)) |spec| : (i += 1) {
            const rec = if (std.mem.endsWith(u8, spec.name, ".weight")) source.recipe(table, spec.name[0 .. spec.name.len - 7]) else null;
            if (rec) |r| {
                if (spec.rank != 2) return error.InvalidVisionWeightShape;
                if (r.algo == .nvfp4 or r.algo == .w4a16_nvfp4) {
                    quantized[i] = .{ .fp4 = try source.Fp4.read(a, ck, table, spec.name[0 .. spec.name.len - 7]) };
                    const l = quantized[i].?.fp4.linear;
                    if (l.n != spec.shape[0] or l.k != spec.shape[1]) return error.InvalidVisionWeightShape;
                } else {
                    const src = try source.Source.read(ck, table, spec.name);
                    if (!std.mem.eql(usize, src.weight.shape[0..src.weight.rank], spec.dims())) return error.InvalidVisionWeightShape;
                    quantized[i] = .{ .fp8 = src };
                }
                tensors[i] = .{ .dtype = .bf16, .rank = spec.rank, .shape = spec.shape, .bytes = &.{} };
            } else {
                const t = try ck.get(spec.name);
                try checkTensor(t, spec.dims());
                tensors[i] = t;
            }
        }
        if (ck.unused() != 0) return error.UnusedVisionTensors;
        return w;
    }

    pub fn converted(self: *const Weights, a: std.mem.Allocator, i: usize) ![]u16 {
        if (self.quantized.len != 0) if (self.quantized[i]) |q| {
            const t = self.tensors[i];
            const out = try a.alloc(u16, t.numel());
            errdefer a.free(out);
            const rows: source.Span = .{ .lo = 0, .hi = t.dim(0) };
            const cols: source.Span = .{ .lo = 0, .hi = t.dim(1) };
            switch (q) {
                .fp8 => |s| try s.bf16(rows, cols, out),
                .fp4 => |s| try s.bf16(rows, cols, out),
            }
            for (out) |v| if (v & 0x7f80 == 0x7f80) return error.NonFiniteVisionWeight;
            return out;
        };
        return toBf16(a, self.tensors[i]);
    }
    /// Maps must outlive this table. Missing/wrong tensors never produce a partly loaded table.
    pub fn load(a: std.mem.Allocator, c: Config, files: []const st.File) !Weights {
        var specs = try Specs.init(c);
        const tensors = try a.alloc(st.Tensor, specs.count());
        errdefer a.free(tensors);
        var buf: [256]u8 = undefined;
        var i: usize = 0;
        while (try specs.next(&buf)) |spec| {
            var found: ?st.Tensor = null;
            for (files) |*file| if (file.get(spec.name)) |t| {
                found = t;
            };
            const t = found orelse return error.MissingVisionWeight;
            try checkTensor(t, spec.dims());
            tensors[i] = t;
            i += 1;
        }
        return .{ .c = c, .tensors = tensors };
    }
    pub fn root(self: *const Weights, kind: Root) st.Tensor {
        return self.tensors[@backingInt(kind)];
    }
    pub fn block(self: *const Weights, layer: u32, kind: Block) st.Tensor {
        std.debug.assert(layer < self.c.depth);
        return self.tensors[roots.len + @as(usize, layer) * blocks.len + @backingInt(kind)];
    }
    pub fn deinit(self: *Weights, a: std.mem.Allocator) void {
        for (self.quantized) |*q| if (q.*) |*v| if (v.* == .fp4) v.fp4.deinit(a);
        a.free(self.quantized);
        a.free(self.tensors);
        self.* = undefined;
    }
};

/// Matches the oracle's .to(torch.bfloat16), without a runtime Torch dependency.
pub fn toBf16(a: std.mem.Allocator, t: st.Tensor) ![]u16 {
    if (t.rank > st.max_rank) return error.InvalidVisionWeightShape;
    try checkTensor(t, t.shape[0..t.rank]);
    const out = try a.alloc(u16, t.bytes.len / t.dtype.size());
    errdefer a.free(out);
    for (out, 0..) |*v, i| {
        const x: f32 = switch (t.dtype) {
            .bf16 => @bitCast(@as(u32, std.mem.readInt(u16, t.bytes[2 * i ..][0..2], .little)) << 16),
            .f16 => @as(f16, @bitCast(std.mem.readInt(u16, t.bytes[2 * i ..][0..2], .little))),
            .f32 => @bitCast(std.mem.readInt(u32, t.bytes[4 * i ..][0..4], .little)),
            else => unreachable,
        };
        if (!std.math.isFinite(x)) return error.NonFiniteVisionWeight;
        const bits: u32 = @bitCast(x);
        const rounded = bits +% (0x7fff + ((bits >> 16) & 1));
        v.* = @truncate(rounded >> 16);
        if (v.* & 0x7f80 == 0x7f80) return error.NonFiniteVisionWeight;
    }
    return out;
}

test "vision specs preserve Conv3d, Conv2d and head-normalization layouts" {
    const c = try @import("cuda_vision_config.zig").parse(std.testing.allocator, @embedFile("fixtures_cuda_config_exl3.json"));
    var specs = try Specs.init(c);
    try std.testing.expectEqual(@as(usize, 347), specs.count());
    var buf: [256]u8 = undefined;
    const patch = (try specs.next(&buf)).?;
    try std.testing.expectEqualStrings("model.visual.patch_embed.proj.weight", patch.name);
    try std.testing.expectEqualSlices(usize, &.{ 1024, 3, 2, 14, 14 }, patch.dims());
    _ = try specs.next(&buf);
    _ = try specs.next(&buf);
    const downsample = (try specs.next(&buf)).?;
    try std.testing.expectEqualStrings("model.visual.downsample.weight", downsample.name);
    try std.testing.expectEqualSlices(usize, &.{ 4096, 1024, 2, 2 }, downsample.dims());
}
