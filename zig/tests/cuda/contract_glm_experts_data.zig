//! Independent logical NVFP4 expert inputs, float64 projections and declared activation-rounding references.
const std = @import("std");
const core = @import("core");
const cuda = @import("cuda");
const c = core.contract.spec;
const g = core.contract.gates;
const rows = core.contract.rows;
const X = cuda.experts.Nvfp4Experts;

pub const Epi = enum { swiglu, down_f32, down_bf16, relu2 };
pub const slots = 9;
pub const experts = 10;

fn hash(v: u64) u64 {
    var x = v;
    x = (x ^ (x >> 30)) *% 0xbf58476d1ce4e5b9;
    x = (x ^ (x >> 27)) *% 0x94d049bb133111eb;
    return x ^ (x >> 31);
}

fn bf(v: f64) f64 {
    return c.fromBf16(c.toBf16(@floatCast(v)));
}

fn maxNumber(a: f64, b: f64) f64 {
    if (std.math.isNan(a)) return b;
    if (std.math.isNan(b)) return a;
    return @max(a, b);
}

/// Shapes name token rows, output columns and input columns; outputs contain nine routed pairs per token.
pub const Host = struct {
    a: std.mem.Allocator,
    shape: c.Shape,
    epi: Epi,
    fill: g.Fill,
    seed: u64,
    x: []u16,
    limit: f32 = 0,

    pub fn init(a: std.mem.Allocator, shape: c.Shape, epi: Epi, fill: g.Fill, seed: u64) !Host {
        const input_rows = shape.rows * (if (epi == .swiglu or epi == .relu2) @as(usize, 1) else slots);
        const h: Host = .{ .a = a, .shape = shape, .epi = epi, .fill = fill, .seed = seed, .x = try a.alloc(u16, input_rows * (shape.k + 8)) };
        @memset(h.x, c.toBf16(19));
        const step: f32 = if (shape.k <= 1024) 64 else 32;
        for (0..input_rows) |r| for (0..shape.k) |i| {
            const mirror = if (fill == .stress and i >= shape.k / 2) (i + 1) % (shape.k / 2) else i % (shape.k / 2);
            const v = hash(seed ^ (r *% 8191 +% mirror));
            const f: f32 = if (fill == .stress)
                step * @as(f32, @floatFromInt(1 + v % 32)) + (if (i >= shape.k / 2 and mirror < 8) step else 0)
            else
                @as(f32, @floatFromInt(@as(i32, @intCast(hash(v +% i) % 128)) - 64)) / 128;
            h.x[r * h.stride() + i] = c.toBf16(f);
        };
        if (fill == .nonfinite) {
            const at = [_][2]usize{ .{ 0, shape.k / 3 }, .{ 1, shape.k / 2 }, .{ 2, shape.k - 1 }, .{ 3, 0 }, .{ 3, shape.k - 1 } };
            const vals = [_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32), std.math.inf(f32), -std.math.inf(f32) };
            for (at, vals) |p, v| if (p[0] < shape.rows) {
                h.x[p[0] * h.stride() + p[1]] = c.toBf16(v);
            };
        }
        return h;
    }

    pub fn deinit(h: Host) void {
        h.a.free(h.x);
    }

    /// The served SwiGLU limit, with inputs scaled 64x so the sums cross it on both sides.
    pub fn clamp(h: *Host, limit: f32) void {
        h.limit = limit;
        for (h.x) |*v| v.* = c.toBf16(c.fromBf16(v.*) * 64);
    }

    /// Outputs whose bf16 gate exceeds the limit, whose up exceeds it, falls below its negative, and neither clamps.
    pub fn clampRegions(h: Host) ![4]usize {
        const t = try h.a.alloc(f64, h.shape.k);
        defer h.a.free(t);
        var out: [4]usize = @splat(0);
        for (0..h.shape.rows * slots) |p| for (0..h.shape.n) |j| {
            var sums: [2]f64 = @splat(0);
            for (&sums, 0..) |*sum, m| {
                h.terms(p, m, j, t);
                for (t) |v| sum.* += v;
                sum.* = bf(sum.* * h.scale(h.expert(p), m));
            }
            const gate = sums[0] > h.limit;
            const high = sums[1] > h.limit;
            const low = sums[1] < -h.limit;
            for ([_]bool{ gate, high, low, !gate and !high and !low }, &out) |hit, *n| n.* += @intFromBool(hit);
        };
        return out;
    }

    pub fn stride(h: Host) usize {
        return h.shape.k + 8;
    }

    pub fn up(h: Host) bool {
        return h.epi == .swiglu or h.epi == .relu2;
    }

    pub fn parts(h: Host) usize {
        return if (h.epi == .swiglu) 2 else 1;
    }

    /// Rotated nine unique routes leave one empty expert and cause cross-token expert collisions.
    pub fn expert(_: Host, pair: usize) usize {
        const token = pair / slots;
        return (token / 3 + pair % slots) % (experts - 1);
    }

    pub fn scale(h: Host, e: usize, m: usize) f32 {
        return if (h.fill == .stress) 1 else @as(f32, @floatFromInt(1 + (e + m) % 3)) / 16;
    }

    fn key(h: Host, e: usize, m: usize, j: usize, i: usize) u64 {
        return hash(h.seed ^ (e *% 65537 +% m *% 262147 +% j *% 8191 +% i));
    }

    fn code(h: Host, e: usize, m: usize, j: usize, i: usize) u8 {
        const mirror = if (i >= h.shape.k / 2) (i + 1) % (h.shape.k / 2) else i;
        const v = h.key(e, m, j, if (h.fill == .stress) mirror else i);
        if (h.fill == .stress) {
            if (j % 16 == 0) return 5;
            const positive = if (mirror < 8) @as(u8, 2) else ([5]u8{ 2, 4, 5, 6, 7 })[v % 5];
            return positive | (if (i < h.shape.k / 2) @as(u8, 8) else 0);
        }
        return if (h.fill == .nonfinite) @intCast(1 + v % 7) else @truncate(v & 15);
    }

    fn blockScale(h: Host, e: usize, m: usize, j: usize, block: usize) u8 {
        return if (h.fill == .stress) 0x38 else @intCast(0x20 + h.key(e, m, j, block + 17) % 24);
    }

    fn weight(h: Host, e: usize, m: usize, j: usize, i: usize) f64 {
        return @as(f64, core.quant.nvfp4.e2m1(h.code(e, m, j, i))) * core.quant.fp8.e4m3(h.blockScale(e, m, j, i / 16));
    }

    /// Pack independent logical codes with the serving ABI; references never read these packed words.
    pub fn packedWords(h: Host) ![]u32 {
        const n = h.shape.n;
        const k = h.shape.k;
        const all = try h.a.alloc(u32, X.bytes(experts, n, k, h.parts()) / 4);
        errdefer h.a.free(all);
        const codes = try h.a.alloc(u8, h.parts() * n * k / 2);
        defer h.a.free(codes);
        const scales = try h.a.alloc(u8, h.parts() * n * k / 16);
        defer h.a.free(scales);
        const one = all.len / experts;
        for (0..experts) |e| {
            for (0..h.parts()) |m| {
                const co = codes[m * n * k / 2 ..][0 .. n * k / 2];
                const sc = scales[m * n * k / 16 ..][0 .. n * k / 16];
                for (0..n) |j| {
                    for (0..k / 2) |i| co[j * (k / 2) + i] = h.code(e, m, j, 2 * i) | (h.code(e, m, j, 2 * i + 1) << 4);
                    for (0..k / 16) |i| sc[j * (k / 16) + i] = h.blockScale(e, m, j, i);
                }
            }
            const dst = all[e * one ..][0..one];
            if (h.parts() == 1) X.packOne(dst, codes, scales, n, k) else X.packGateUp(dst, .{ codes[0 .. n * k / 2], scales[0 .. n * k / 16] }, .{ codes[n * k / 2 ..], scales[n * k / 16 ..] }, n, k);
        }
        return all;
    }

    fn activation(h: Host, gate: f64, up_value: f64) f64 {
        return switch (h.epi) {
            .down_f32, .down_bf16 => gate,
            .relu2 => blk: {
                const v = maxNumber(bf(gate), 0);
                break :blk v * v;
            },
            .swiglu => blk: {
                const gv, const uv = h.clamped(bf(gate), bf(up_value));
                break :blk bf(gv / (1 + @exp(-gv))) * uv;
            },
        };
    }

    /// minNumber(gate, limit) and up clamped to [-limit, limit] by maxNumber/minNumber; unchanged without a limit.
    fn clamped(h: Host, gate: f64, up_value: f64) struct { f64, f64 } {
        if (h.limit <= 0) return .{ gate, up_value };
        return .{ -maxNumber(-gate, -h.limit), -maxNumber(-maxNumber(up_value, -h.limit), -h.limit) };
    }

    fn terms(h: Host, pair: usize, m: usize, j: usize, out: []f64) void {
        const row = if (h.up()) pair / slots else pair;
        const e = h.expert(pair);
        for (out, 0..) |*v, i| v.* = @as(f64, c.fromBf16(h.x[row * h.stride() + i])) * h.weight(e, m, j, i);
    }

    pub fn calculate(h: Host, acc: ?c.Accum, order: g.Order, want: []f64, mag: ?[]f64) !void {
        const terms_buf = try h.a.alloc(f64, h.shape.k);
        defer h.a.free(terms_buf);
        for (0..h.shape.rows * slots) |p| for (0..h.shape.n) |j| {
            var sums: [2]f64 = @splat(0);
            var abs: [2]f64 = @splat(0);
            for (0..h.parts()) |m| {
                h.terms(p, m, j, terms_buf);
                if (acc) |a| {
                    sums[m] = rows.sumIn(a, order, terms_buf);
                } else {
                    for (terms_buf) |v| sums[m] += v;
                }
                for (terms_buf) |v| abs[m] += @abs(v);
                sums[m] *= h.scale(h.expert(p), m);
                abs[m] *= h.scale(h.expert(p), m);
            }
            const at = p * h.shape.n + j;
            want[at] = h.activation(sums[0], sums[1]);
            if (mag) |out| out[at] = switch (h.epi) {
                .down_f32, .down_bf16 => abs[0],
                .relu2 => 2 * @max(@abs(sums[0]), 1) * abs[0] + c.gamma(@floatFromInt(h.shape.k / 16 + 4), c.unit(.f32)) * abs[0] * abs[0],
                .swiglu => blk: {
                    const gv, const uv = h.clamped(bf(sums[0]), bf(sums[1]));
                    break :blk 1.1 * @abs(uv) * abs[0] + @abs(gv) * abs[1];
                },
            };
        };
    }
};

test "clamped SwiGLU reference follows minNumber/maxNumber for finite values and planted nonfinites" {
    var h = try Host.init(std.testing.allocator, .{ .rows = 1, .n = 32, .k = 32 }, .swiglu, .random, 1);
    defer h.deinit();
    h.limit = 10;
    try std.testing.expectEqual(@as(f64, 100), h.activation(1000, 1000));
    try std.testing.expectEqual(@as(f64, -100), h.activation(1000, -1000));
    try std.testing.expectEqual(@as(f64, -100), h.activation(std.math.nan(f64), std.math.nan(f64)));
    try std.testing.expectEqual(@as(f64, 100), h.activation(std.math.inf(f64), std.math.inf(f64)));
    try std.testing.expect(std.math.isNan(h.activation(-std.math.inf(f64), -std.math.inf(f64))));
    h.limit = 0;
    try std.testing.expect(std.math.isNan(h.activation(std.math.nan(f64), std.math.nan(f64))));
}
