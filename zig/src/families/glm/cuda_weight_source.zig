const std = @import("std");
const core = @import("core");
const X = @import("cuda").experts.Nvfp4Experts;
const Tensor = core.checkpoint.Tensor;
const DType = core.checkpoint.DType;
pub const Span = core.fp8_source.Span;

pub fn recipe(table: ?*const core.modelopt.Recipes, module: []const u8) ?core.modelopt.Recipe {
    const t = table orelse return null;
    if (t.exact(module)) |r| return r;
    if (std.mem.indexOf(u8, module, ".mlp.experts.")) |at| return t.exact(module[0 .. at + ".mlp.experts".len]);
    return null;
}

fn tensor(ck: *core.Checkpoint, module: []const u8, suffix: []const u8) !Tensor {
    var buf: [256]u8 = undefined;
    return ck.get(try std.fmt.bufPrint(&buf, "{s}.{s}", .{ module, suffix }));
}

pub const Source = struct {
    weight: Tensor,
    format: ?core.modelopt.Algo = null,
    scale: ?Tensor = null,

    pub fn read(ck: *core.Checkpoint, table: ?*const core.modelopt.Recipes, name: []const u8) !Source {
        const w = try ck.get(name);
        if (std.mem.endsWith(u8, name, ".weight")) {
            const module = name[0 .. name.len - 7];
            if (recipe(table, module)) |r| {
                if (r.algo() != .mxfp8 and r.algo() != .fp8_pb_wo) return error.WeightRecipeMismatch;
                const s = try tensor(ck, module, "weight_scale");
                const source: Source = .{ .weight = w, .format = r.algo(), .scale = s };
                var empty: [0]u16 = .{};
                try source.bf16(.{ .lo = 0, .hi = 0 }, .{ .lo = 0, .hi = 0 }, &empty);
                return source;
            }
        }
        if (w.dtype != .bf16 and w.dtype != .f32 and w.dtype != .f16) return error.WeightRecipeMismatch;
        return .{ .weight = w };
    }

    pub fn bf16(s: Source, rows: Span, cols: Span, out: []u16) !void {
        if (s.format) |f| return switch (f) {
            .mxfp8 => core.fp8_source.mxfp8Bf16(s.weight, s.scale.?, rows, cols, out),
            .fp8_pb_wo => core.fp8_source.pbWoBf16(s.weight, s.scale.?, rows, cols, out),
            else => error.WeightRecipeMismatch,
        };
        const t = s.weight;
        if (t.rank != 2 or rows.lo > rows.hi or rows.hi > t.dim(0) or cols.lo > cols.hi or cols.hi > t.dim(1) or out.len != (rows.hi - rows.lo) * (cols.hi - cols.lo)) return error.UnexpectedTensor;
        for (rows.lo..rows.hi) |r| for (cols.lo..cols.hi) |c| {
            out[(r - rows.lo) * (cols.hi - cols.lo) + c - cols.lo] = core.quant.convert.bf16(core.quant.convert.load(t.dtype, t.bytes, r * t.dim(1) + c));
        };
    }

    pub fn append(s: Source, a: std.mem.Allocator, host: *std.ArrayList(u8), lo: usize, hi: usize, to: DType) !void {
        const t = s.weight;
        if (lo > hi or hi > t.numel()) return error.UnexpectedTensor;
        if (s.format != null) {
            if (to != .bf16) return error.WeightRecipeMismatch;
            const k = t.dim(1);
            var decoded: [4096]u16 = undefined;
            var at = lo;
            while (at < hi) {
                const end = @min(hi, (at / k + 1) * k, at + decoded.len);
                try s.bf16(.{ .lo = at / k, .hi = at / k + 1 }, .{ .lo = at % k, .hi = (end - 1) % k + 1 }, decoded[0 .. end - at]);
                try host.appendSlice(a, std.mem.sliceAsBytes(decoded[0 .. end - at]));
                at = end;
            }
            return;
        }
        if (t.dtype == to) return host.appendSlice(a, t.bytes[lo * to.size() .. hi * to.size()]);
        if (t.dtype != .bf16 and t.dtype != .f32 and t.dtype != .f16) return error.UnexpectedTensor;
        for (lo..hi) |i| {
            const v = core.quant.convert.load(t.dtype, t.bytes, i);
            switch (to) {
                .f32 => try host.appendSlice(a, std.mem.asBytes(&v)),
                .bf16 => try host.appendSlice(a, std.mem.asBytes(&core.quant.convert.bf16(v))),
                else => return error.UnexpectedTensor,
            }
        }
    }
};

pub const Fp4 = struct {
    linear: core.nvfp4.Linear,
    scratch: ?[]u8 = null,

    pub fn deinit(s: *Fp4, a: std.mem.Allocator) void {
        if (s.scratch) |b| a.free(b);
        s.* = undefined;
    }

    pub fn read(a: std.mem.Allocator, ck: *core.Checkpoint, table: ?*const core.modelopt.Recipes, module: []const u8) !Fp4 {
        const r = recipe(table, module);
        if (table != null and (r == null or (r.?.algo() != .nvfp4 and r.?.algo() != .w4a16_nvfp4))) return error.WeightRecipeMismatch;
        if (r == null or r.?.scaleEncoding() == .none) return .{ .linear = try core.nvfp4.Linear.read(ck, module) };
        const w = try tensor(ck, module, "weight");
        const g = try tensor(ck, module, "weight_scale_2");
        if (w.dtype != .u8 or w.rank != 2 or w.dim(0) == 0 or w.dim(1) == 0 or w.dim(1) % 8 != 0) return error.WeightRecipeMismatch;
        const n = w.dim(0);
        const k = 2 * w.dim(1);
        var buf: [256]u8 = undefined;
        const input_name = try std.fmt.bufPrint(&buf, "{s}.input_scale", .{module});
        const input_tensor: ?Tensor = if (ck.has(input_name)) try ck.get(input_name) else null;
        const input: ?core.nvfp4.Meta = if (input_tensor) |*t| meta(t) else null;
        const scales = try a.alloc(u8, n * k / 16);
        errdefer a.free(scales);
        const s: Tensor = .{ .dtype = .f8_e4m3, .rank = 2, .shape = .{ n, k / 16, 1, 1, 1 }, .bytes = scales };
        _ = try core.nvfp4.dims(meta(&w), meta(&s), meta(&g), input);
        var parts: [5]Tensor = undefined;
        inline for (.{ "bases", "codes", "page_bases", "offsets", "exceptions" }, .{ DType.u8, .u8, .u32, .u16, .u16 }, 0..) |suffix, dtype, i| {
            parts[i] = try tensor(ck, module, "weight_scale.csf32_" ++ suffix);
            if (parts[i].dtype != dtype or parts[i].rank != 1) return error.WeightRecipeMismatch;
        }
        try core.csf.decode(parts[0].bytes, parts[1].bytes, parts[2].bytes, parts[3].bytes, parts[4].bytes, n, k / 16, scales);
        return .{ .linear = .{ .n = n, .k = k, .codes = w.bytes, .scales = scales, .global = std.mem.bytesToValue(f32, g.bytes[0..4]) }, .scratch = scales };
    }

    pub fn bf16(s: Fp4, rows: Span, cols: Span, out: []u16) !void {
        const l = s.linear;
        if (rows.lo > rows.hi or rows.hi > l.n or cols.lo > cols.hi or cols.hi > l.k or out.len != (rows.hi - rows.lo) * (cols.hi - cols.lo)) return error.UnexpectedTensor;
        for (rows.lo..rows.hi) |r| for (cols.lo..cols.hi) |c| {
            const byte = l.codes[r * l.k / 2 + c / 2];
            const nibble = if (c % 2 == 0) byte & 15 else byte >> 4;
            const v = core.quant.nvfp4.e2m1(nibble) * core.quant.fp8.e4m3(l.scales[r * l.k / 16 + c / 16]) * l.global;
            out[(r - rows.lo) * (cols.hi - cols.lo) + c - cols.lo] = core.quant.convert.bf16(@floatCast(v));
        };
    }
};

fn meta(t: *const Tensor) core.nvfp4.Meta {
    return .{ .dtype = t.dtype, .shape = t.shape[0..t.rank] };
}

pub fn packExpert(a: std.mem.Allocator, ck: *core.Checkpoint, table: ?*const core.modelopt.Recipes, layer: u32, expert: usize, cut: Span, d: usize, up: []u32, down: []u32, codes: []u8, scales: []u8) ![3]f32 {
    if (cut.lo > cut.hi or cut.lo % 32 != 0 or cut.hi % 32 != 0 or cut.lo == cut.hi or d == 0 or d % 32 != 0) return error.UnexpectedTensor;
    const ni = cut.hi - cut.lo;
    if (up.len != X.bytes(1, ni, d, 2) / 4 or down.len != X.bytes(1, d, ni, 1) / 4 or codes.len != d * ni / 2 or scales.len != d * ni / 16) return error.UnexpectedTensor;
    var stored: [3]Fp4 = undefined;
    var made: usize = 0;
    defer for (stored[0..made]) |*v| v.deinit(a);
    var ls: [3]core.nvfp4.Linear = undefined;
    inline for (.{ "gate_proj", "up_proj", "down_proj" }, 0..) |proj, m| {
        var buf: [256]u8 = undefined;
        const name = try std.fmt.bufPrint(&buf, "model.language_model.layers.{d}.mlp.experts.{d}." ++ proj, .{ layer, expert });
        stored[m] = try Fp4.read(a, ck, table, name);
        made += 1;
        ls[m] = stored[m].linear;
        if (m < 2) {
            if (ls[m].k != d or cut.hi > ls[m].n) return error.UnexpectedTensor;
        } else if (ls[m].n != d or cut.hi > ls[m].k) return error.UnexpectedTensor;
    }
    const gate = .{ ls[0].codes[cut.lo * d / 2 ..][0 .. ni * d / 2], ls[0].scales[cut.lo * d / 16 ..][0 .. ni * d / 16] };
    const upm = .{ ls[1].codes[cut.lo * d / 2 ..][0 .. ni * d / 2], ls[1].scales[cut.lo * d / 16 ..][0 .. ni * d / 16] };
    X.packGateUp(up, gate, upm, ni, d);
    const l = ls[2];
    for (0..d) |r| {
        @memcpy(codes[r * ni / 2 ..][0 .. ni / 2], l.codes[r * l.k / 2 + cut.lo / 2 ..][0 .. ni / 2]);
        @memcpy(scales[r * ni / 16 ..][0 .. ni / 16], l.scales[r * l.k / 16 + cut.lo / 16 ..][0 .. ni / 16]);
    }
    X.packOne(down, codes, scales, d, ni);
    return .{ ls[0].global, ls[1].global, l.global };
}
