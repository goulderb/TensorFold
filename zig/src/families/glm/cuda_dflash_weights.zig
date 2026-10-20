//! DFlash numerical loader adapted from Mia f013e1ae. Ash Hart and TensorFold contributors.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const quant = @import("cuda_dflash_quantize.zig");
const mm = @import("cuda_dflash_matmuls.zig");
const dc = @import("cuda_dflash_config.zig");
const Source = enum { model, dflash2 };
pub const Slice = struct { axis: usize, start: usize, len: usize };
fn sliceFor(name: []const u8, rank: dc.Rank) !?Slice {
    var q0: usize = 0;
    var k0: usize = 0;
    var m0: usize = 0;
    for (0..rank.rank) |r| {
        const prev = try dc.Rank.init(rank.world, r);
        q0 += prev.heads * 128;
        k0 += prev.kv * 128;
        m0 += prev.inter;
    }
    if (std.mem.eql(u8, name, "lm_head.weight")) return .{ .axis = 0, .start = rank.vocab_start, .len = rank.vocab };
    if (std.mem.endsWith(u8, name, ".self_attn.q_proj.weight")) return .{ .axis = 0, .start = q0, .len = rank.heads * 128 };
    if (std.mem.endsWith(u8, name, ".self_attn.k_proj.weight") or std.mem.endsWith(u8, name, ".self_attn.v_proj.weight")) return .{ .axis = 0, .start = k0, .len = rank.kv * 128 };
    if (std.mem.endsWith(u8, name, ".self_attn.o_proj.weight")) return .{ .axis = 1, .start = q0, .len = rank.heads * 128 };
    if (std.mem.endsWith(u8, name, ".mlp.gate_proj.weight") or std.mem.endsWith(u8, name, ".mlp.up_proj.weight")) return .{ .axis = 0, .start = m0, .len = rank.inter };
    if (std.mem.endsWith(u8, name, ".mlp.down_proj.weight")) return .{ .axis = 1, .start = m0, .len = rank.inter };
    return null;
}
fn slicedShape(t: core.checkpoint.Tensor, cut: ?Slice) ![3]usize {
    if (t.rank < 1 or t.rank > 3) return error.BadDraftShape;
    var shape: [3]usize = @splat(1);
    for (0..t.rank) |i| shape[i] = t.dim(i);
    if (cut) |c| {
        if (c.axis >= t.rank or c.start > shape[c.axis] or c.len > shape[c.axis] - c.start) return error.BadDraftShape;
        shape[c.axis] = c.len;
    }
    return shape;
}
fn value(t: core.checkpoint.Tensor, i: usize) !f32 {
    return switch (t.dtype) {
        .bf16 => quant.bf16ToF32(std.mem.readInt(u16, t.bytes[i * 2 ..][0..2], .little)),
        .f32 => @bitCast(std.mem.readInt(u32, t.bytes[i * 4 ..][0..4], .little)),
        .f16 => @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, t.bytes[i * 2 ..][0..2], .little)))),
        else => error.UnsupportedDraftDtype,
    };
}
fn decode(gpa: std.mem.Allocator, t: core.checkpoint.Tensor, cut: ?Slice) ![]u16 {
    const shape = try slicedShape(t, cut);
    const out = try gpa.alloc(u16, shape[0] * shape[1] * shape[2]);
    errdefer gpa.free(out);
    var inner: usize = 1;
    if (cut) |c| for (c.axis + 1..t.rank) |j| {
        inner *= t.dim(j);
    };
    for (out, 0..) |*v, i| {
        const index = if (cut) |c| i / (c.len * inner) * t.dim(c.axis) * inner + c.start * inner + i % (c.len * inner) else i;
        v.* = if (t.dtype == .bf16) std.mem.readInt(u16, t.bytes[index * 2 ..][0..2], .little) else quant.bf16Rne(try value(t, index));
    }
    return out;
}
fn decodeMxfp8(gpa: std.mem.Allocator, ck: *core.Checkpoint, t: core.checkpoint.Tensor, cut: ?Slice) ![]u16 {
    if (!ck.has("lm_head.weight_scale")) return error.UnsupportedDraftDtype;
    const scale = try ck.get("lm_head.weight_scale");
    if (scale.dtype != .u8 and scale.dtype != .f8_e8m0) return error.UnsupportedDraftDtype;
    if (cut) |c| if (c.axis != 0) return error.BadDraftShape;
    const shape = try slicedShape(t, cut);
    const out = try gpa.alloc(u16, shape[0] * shape[1] * shape[2]);
    errdefer gpa.free(out);
    const rows: core.fp8_source.Span = if (cut) |c| .{ .lo = c.start, .hi = c.start + c.len } else core.fp8_source.all(t.dim(0));
    try core.fp8_source.mxfp8Bf16(t, scale, rows, core.fp8_source.all(t.dim(1)), out);
    return out;
}
fn decodeHead(gpa: std.mem.Allocator, ck: *core.Checkpoint, t: core.checkpoint.Tensor, cut: ?Slice) ![]u16 {
    return switch (t.dtype) {
        .bf16 => decode(gpa, t, cut),
        .f8_e4m3 => decodeMxfp8(gpa, ck, t, cut),
        else => error.UnsupportedDraftDtype,
    };
}
pub fn decodeTargetHead(gpa: std.mem.Allocator, ck: *core.Checkpoint, cut: ?Slice) ![]u16 {
    return decodeHead(gpa, ck, try ck.get("lm_head.weight"), cut);
}
pub fn readCodebook(gpa: std.mem.Allocator, ck: *core.Checkpoint, name: []const u8) ![]f32 {
    const t = try ck.get(name);
    if (t.rank != 2 or t.dim(0) != 154880 or t.dim(1) != 256 or (t.dtype != .f32 and t.dtype != .bf16)) return error.BadSelector;
    const out = try gpa.alloc(f32, t.numel());
    errdefer gpa.free(out);
    for (out, 0..) |*v, i| v.* = try value(t, i);
    return out;
}
pub const Layer = struct {
    in_norm: u64,
    post_norm: u64,
    a_base: u64,
    m_base: u64,
    q_norm: u64,
    k_norm: u64,
    a_kp: mm.Q4,
    m_kp: mm.Q4,
    qkv: mm.Q4,
    kv: mm.Q4,
    o: mm.Q4,
    gu: mm.Q4,
    down: mm.Q4,
};
pub const Loader = struct {
    gpa: std.mem.Allocator,
    driver: *const cuda.Driver,
    target: *core.Checkpoint,
    draft: *core.Checkpoint,
    rank: dc.Rank,
    owned: std.ArrayList(cuda.DeviceBuffer) = .empty,
    pub fn deinit(l: *Loader) void {
        for (l.owned.items) |*b| b.free();
        l.owned.deinit(l.gpa);
    }
    fn tensor(l: *Loader, source: Source, name: []const u8, n: usize, k: usize) ![]u16 {
        const ck = if (source == .model) l.target else l.draft;
        const t = try ck.get(name);
        const cut = try sliceFor(name, l.rank);
        const got = if (source == .model and std.mem.eql(u8, name, "lm_head.weight")) try decodeHead(l.gpa, ck, t, cut) else try decode(l.gpa, t, cut);
        errdefer l.gpa.free(got);
        const shape = try slicedShape(t, cut);
        if (got.len != n * k or shape[0] != n or (t.rank >= 2 and shape[1] != k)) return error.BadDraftShape;
        return got;
    }
    pub fn upload(l: *Loader, bytes: []const u8) !u64 {
        var b = try cuda.DeviceBuffer.fromHost(l.driver, bytes);
        errdefer b.free();
        try l.owned.append(l.gpa, b);
        return b.ptr;
    }
    fn raw(l: *Loader, name: []const u8, n: usize, k: usize) !u64 {
        const t = try l.tensor(.dflash2, name, n, k);
        defer l.gpa.free(t);
        return l.upload(std.mem.sliceAsBytes(t));
    }
    pub fn q4(l: *Loader, source: Source, names: []const []const u8, ns: []const usize, k: usize) !mm.Q4 {
        if (names.len != ns.len or k % 64 != 0) return error.BadDraftShape;
        var n: usize = 0;
        for (ns) |rows| n += rows;
        const rows = try l.gpa.alloc(u16, n * k);
        defer l.gpa.free(rows);
        var at: usize = 0;
        for (names, ns) |name, width| {
            const t = try l.tensor(source, name, width, k);
            defer l.gpa.free(t);
            @memcpy(rows[at..][0..t.len], t);
            at += t.len;
        }
        const words = try l.gpa.alloc(u32, n * k / 8);
        defer l.gpa.free(words);
        const scales = try l.gpa.alloc(u16, n * k / 64);
        defer l.gpa.free(scales);
        const biases = try l.gpa.alloc(u16, scales.len);
        defer l.gpa.free(biases);
        quant.quantize4(rows, n, k, words, scales, biases, false);
        const pad = std.mem.alignForward(usize, n, 128);
        const pack_buf = try l.gpa.alloc(u32, pad * k / 8);
        defer l.gpa.free(pack_buf);
        const ps = try l.gpa.alloc(u16, pad * k / 64);
        defer l.gpa.free(ps);
        const pb = try l.gpa.alloc(u16, ps.len);
        defer l.gpa.free(pb);
        quant.pack(words, scales, biases, n, k, pack_buf, ps, pb);
        return .{ .n = n, .k = k, .w = try l.upload(std.mem.sliceAsBytes(pack_buf)), .s = try l.upload(std.mem.sliceAsBytes(ps)), .b = try l.upload(std.mem.sliceAsBytes(pb)) };
    }
    fn one(l: *Loader, name: []const u8, n: usize, k: usize) !mm.Q4 {
        return l.q4(.dflash2, &.{name}, &.{n}, k);
    }
    /// The isolated tap-commit probe never reads the target head or draft decoder weights.
    pub fn loadTaps(l: *Loader) !TapWeights {
        var w: TapWeights = undefined;
        w.fc = try l.one("fc.weight", 4096, 5 * 4096);
        w.hidden_norm = try l.raw("hidden_norm.weight", 4096, 1);
        var buf: [192]u8 = undefined;
        for (&w.layers, 0..) |*layer, i| {
            const k = try std.fmt.allocPrint(l.gpa, "layers.{d}.self_attn.k_proj.weight", .{i});
            defer l.gpa.free(k);
            const v = try std.fmt.allocPrint(l.gpa, "layers.{d}.self_attn.v_proj.weight", .{i});
            defer l.gpa.free(v);
            layer.kv = try l.q4(.dflash2, &.{ k, v }, &.{ l.rank.kv * 128, l.rank.kv * 128 }, 4096);
            layer.q_norm = try l.raw(try std.fmt.bufPrint(&buf, "layers.{d}.self_attn.q_norm.weight", .{i}), 128, 1);
            layer.k_norm = try l.raw(try std.fmt.bufPrint(&buf, "layers.{d}.self_attn.k_norm.weight", .{i}), 128, 1);
        }
        return w;
    }

    pub fn load(l: *Loader, rank: dc.Rank) !Weights {
        if (l.rank.rank != rank.rank or l.rank.world != rank.world) return error.BadDraftRank;
        var w: Weights = undefined;
        w.fc = try l.one("fc.weight", 4096, 5 * 4096);
        w.hidden_norm = try l.raw("hidden_norm.weight", 4096, 1);
        w.norm = try l.raw("norm.weight", 4096, 1);
        w.hproj = try l.raw("candidate_selector.hidden_projection.weight", 256, 4096);
        w.head = try l.q4(.model, &.{"lm_head.weight"}, &.{rank.vocab}, 4096);
        var buf: [512]u8 = undefined;
        for (&w.layers, 0..) |*layer, i| {
            const prefix = try std.fmt.allocPrint(l.gpa, "layers.{d}.", .{i});
            defer l.gpa.free(prefix);
            inline for (.{ "in_norm", "post_norm", "a_base", "m_base", "q_norm", "k_norm" }, .{ "input_layernorm.weight", "post_attention_layernorm.weight", "attention_conv.base_kernel", "mlp_conv.base_kernel", "self_attn.q_norm.weight", "self_attn.k_norm.weight" }, .{ 4096, 4096, 4, 4, 128, 128 }, .{ 1, 1, 4096, 4096, 1, 1 }) |field, suffix, n, k| {
                const name = try std.fmt.bufPrint(&buf, "{s}{s}", .{ prefix, suffix });
                if (n == 4) {
                    const t0 = try l.draft.get(name);
                    if (t0.rank != 3 or t0.dim(0) != 2 or t0.dim(1) != 2 or t0.dim(2) != 4096) return error.BadDraftShape;
                    const t = try decode(l.gpa, t0, null);
                    defer l.gpa.free(t);
                    @field(layer, field) = try l.upload(std.mem.sliceAsBytes(t));
                } else @field(layer, field) = try l.raw(name, n, k);
            }
            inline for (.{ "a_kp", "m_kp", "o", "down" }, .{ "attention_conv.kernel_projection.weight", "mlp_conv.kernel_projection.weight", "self_attn.o_proj.weight", "mlp.down_proj.weight" }, .{ 1024, 1024, 4096, 4096 }, .{ @as(usize, 4096), 4096, rank.heads * 128, rank.inter }) |field, suffix, n, k| {
                @field(layer, field) = try l.one(try std.fmt.bufPrint(&buf, "{s}{s}", .{ prefix, suffix }), n, k);
            }
            const q = try std.fmt.allocPrint(l.gpa, "{s}self_attn.q_proj.weight", .{prefix});
            defer l.gpa.free(q);
            const k = try std.fmt.allocPrint(l.gpa, "{s}self_attn.k_proj.weight", .{prefix});
            defer l.gpa.free(k);
            const v = try std.fmt.allocPrint(l.gpa, "{s}self_attn.v_proj.weight", .{prefix});
            defer l.gpa.free(v);
            layer.qkv = try l.q4(.dflash2, &.{ q, k, v }, &.{ rank.heads * 128, rank.kv * 128, rank.kv * 128 }, 4096);
            layer.kv = try l.q4(.dflash2, &.{ k, v }, &.{ rank.kv * 128, rank.kv * 128 }, 4096);
            const gate = try std.fmt.allocPrint(l.gpa, "{s}mlp.gate_proj.weight", .{prefix});
            defer l.gpa.free(gate);
            const up = try std.fmt.allocPrint(l.gpa, "{s}mlp.up_proj.weight", .{prefix});
            defer l.gpa.free(up);
            layer.gu = try l.q4(.dflash2, &.{ gate, up }, &.{ rank.inter, rank.inter }, 4096);
        }
        return w;
    }
};
pub const TapWeights = struct { fc: mm.Q4, hidden_norm: u64, layers: [5]struct { kv: mm.Q4, q_norm: u64, k_norm: u64 } };
pub const Weights = struct { fc: mm.Q4, head: mm.Q4, hidden_norm: u64, norm: u64, hproj: u64, layers: [5]Layer };

test "checkpoint column slices retain row order and BF16 bits" {
    const bytes = [_]u8{ 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0, 8, 0 };
    const tensor_: core.checkpoint.Tensor = .{ .dtype = .bf16, .rank = 2, .shape = .{ 2, 4, 1, 1, 1 }, .bytes = &bytes };
    const got = try decode(std.testing.allocator, tensor_, .{ .axis = 1, .start = 1, .len = 2 });
    defer std.testing.allocator.free(got);
    try std.testing.expectEqualSlices(u16, &.{ 2, 3, 6, 7 }, got);
    try std.testing.expectError(error.BadDraftShape, decode(std.testing.allocator, tensor_, .{ .axis = 1, .start = 3, .len = 2 }));
}
test "rank slicing matches the distinct projection axes" {
    const rank = try dc.Rank.init(2, 1);
    const q = (try sliceFor("layers.0.self_attn.q_proj.weight", rank)).?;
    const o = (try sliceFor("layers.0.self_attn.o_proj.weight", rank)).?;
    const down = (try sliceFor("layers.4.mlp.down_proj.weight", rank)).?;
    const head = (try sliceFor("lm_head.weight", rank)).?;
    try std.testing.expectEqual(@as(usize, 0), q.axis);
    try std.testing.expectEqual(@as(usize, 1), o.axis);
    try std.testing.expectEqual(@as(usize, 2048), q.start);
    try std.testing.expectEqual(q.start, o.start);
    try std.testing.expectEqual(@as(usize, 6144), down.start);
    try std.testing.expectEqual(@as(usize, 77440), head.start);
    try std.testing.expect((try sliceFor("fc.weight", rank)) == null);
}
test "selector float reads retain precision that BF16 would remove" {
    const values = [_]f32{ 1.0001, -0.0001234567 };
    const tensor_: core.checkpoint.Tensor = .{ .dtype = .f32, .rank = 1, .shape = .{ 2, 1, 1, 1, 1 }, .bytes = std.mem.sliceAsBytes(&values) };
    try std.testing.expectEqual(values[0], try value(tensor_, 0));
    try std.testing.expectEqual(values[1], try value(tensor_, 1));
    try std.testing.expect(values[0] != quant.bf16ToF32(quant.bf16Rne(values[0])));
}
