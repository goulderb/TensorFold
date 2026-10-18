//! MXFP8 [1,32] and FP8_PB_WO [128,128] source weights decoded to fp64 and bf16 (torch nearest-even).

const std = @import("std");
const st = @import("safetensors.zig");
const fp8 = @import("quant/fp8.zig");
const convert = @import("quant/convert.zig");

const Tensor = st.Tensor;
const DType = st.DType;

pub const Error = error{BadFp8Source};

/// Half-open [lo, hi) of rows or columns a TP shard decodes.
pub const Span = struct { lo: usize, hi: usize };

pub fn all(n: usize) Span {
    return .{ .lo = 0, .hi = n };
}

/// OCP MX E8M0: 2^(byte-127); 255 is NaN.
pub fn e8m0(b: u8) f64 {
    if (b == 255) return std.math.nan(f64);
    return std.math.ldexp(@as(f64, 1.0), @as(i32, b) - 127);
}

pub fn mxfp8(weight: Tensor, scale: Tensor, rows: Span, cols: Span, out: []f64) Error!void {
    try decode(.mxfp8, weight, scale, rows, cols, f64, out);
}

pub fn mxfp8Bf16(weight: Tensor, scale: Tensor, rows: Span, cols: Span, out: []u16) Error!void {
    try decode(.mxfp8, weight, scale, rows, cols, u16, out);
}

pub fn pbWo(weight: Tensor, scale: Tensor, rows: Span, cols: Span, out: []f64) Error!void {
    try decode(.fp8_pb_wo, weight, scale, rows, cols, f64, out);
}

pub fn pbWoBf16(weight: Tensor, scale: Tensor, rows: Span, cols: Span, out: []u16) Error!void {
    try decode(.fp8_pb_wo, weight, scale, rows, cols, u16, out);
}

const Kind = enum { mxfp8, fp8_pb_wo };

fn decode(kind: Kind, weight: Tensor, scale: Tensor, rows: Span, cols: Span, comptime T: type, out: []T) Error!void {
    const nk = try weightNk(weight);
    const n = nk[0];
    const k = nk[1];
    try checkSpan(rows, n);
    try checkSpan(cols, k);
    const rn = rows.hi - rows.lo;
    const cn = cols.hi - cols.lo;
    if (out.len != rn * cn) return error.BadFp8Source;
    const layout = try scaleLayout(kind, scale, n, k);
    for (rows.lo..rows.hi) |r| {
        for (cols.lo..cols.hi) |c| {
            const code = weight.bytes[r * k + c];
            const si = (r / layout.row_block) * layout.kb + (c / layout.col_block);
            const s = switch (kind) {
                .mxfp8 => e8m0(scale.bytes[si]),
                .fp8_pb_wo => f32le(scale.bytes, si),
            };
            store(T, out, (r - rows.lo) * cn + (c - cols.lo), code, s);
        }
    }
}

fn store(comptime T: type, out: []T, i: usize, code: u8, scale: f64) void {
    if (T == f64) {
        out[i] = fp8.e4m3(code) * scale;
    } else if (T == u16) {
        out[i] = convert.bf16(@as(f32, @floatCast(fp8.e4m3(code))) * @as(f32, @floatCast(scale)));
    } else unreachable;
}

fn weightNk(weight: Tensor) Error!struct { usize, usize } {
    if (weight.dtype != .f8_e4m3 or weight.rank != 2) return error.BadFp8Source;
    const n = weight.shape[0];
    const k = weight.shape[1];
    if (n == 0 or k == 0 or weight.bytes.len != n * k) return error.BadFp8Source;
    return .{ n, k };
}

const Layout = struct { kb: usize, row_block: usize, col_block: usize };

fn scaleLayout(kind: Kind, scale: Tensor, n: usize, k: usize) Error!Layout {
    switch (kind) {
        .mxfp8 => {
            if (scale.dtype != .u8 and scale.dtype != .f8_e8m0) return error.BadFp8Source;
            if (k % 32 != 0 or scale.rank != 2 or scale.shape[0] != n or scale.shape[1] != k / 32) return error.BadFp8Source;
            if (scale.bytes.len != n * (k / 32)) return error.BadFp8Source;
            return .{ .kb = k / 32, .row_block = 1, .col_block = 32 };
        },
        .fp8_pb_wo => {
            if (scale.dtype != .f32) return error.BadFp8Source;
            const nb = std.math.divCeil(usize, n, fp8.block) catch unreachable;
            const kb = std.math.divCeil(usize, k, fp8.block) catch unreachable;
            if (!pbShape(scale, nb, kb) or scale.bytes.len != nb * kb * 4) return error.BadFp8Source;
            return .{ .kb = kb, .row_block = fp8.block, .col_block = fp8.block };
        },
    }
}

fn pbShape(scale: Tensor, nb: usize, kb: usize) bool {
    if (scale.rank == 2) return scale.shape[0] == nb and scale.shape[1] == kb;
    if (scale.rank == 4) return scale.shape[0] == nb and scale.shape[1] == 1 and scale.shape[2] == kb and scale.shape[3] == 1;
    return false;
}

fn checkSpan(s: Span, dim: usize) Error!void {
    if (s.lo > s.hi or s.hi > dim) return error.BadFp8Source;
}

fn f32le(bytes: []const u8, i: usize) f64 {
    return @floatCast(@as(f32, @bitCast(std.mem.readInt(u32, bytes[4 * i ..][0..4], .little))));
}

fn tensor2(dtype: DType, n: usize, k: usize, bytes: []const u8) Tensor {
    var t: Tensor = .{ .dtype = dtype, .rank = 2, .shape = @splat(1), .bytes = bytes };
    t.shape[0] = n;
    t.shape[1] = k;
    return t;
}

fn tensor4(dtype: DType, a: usize, b: usize, c: usize, d: usize, bytes: []const u8) Tensor {
    var t: Tensor = .{ .dtype = dtype, .rank = 4, .shape = @splat(1), .bytes = bytes };
    t.shape[0] = a;
    t.shape[1] = b;
    t.shape[2] = c;
    t.shape[3] = d;
    return t;
}

fn fromBf16(h: u16) f32 {
    return @bitCast(@as(u32, h) << 16);
}

test "e8m0 follows OCP MX (255 is NaN)" {
    try std.testing.expectEqual(std.math.ldexp(@as(f64, 1.0), -127), e8m0(0));
    try std.testing.expectEqual(@as(f64, 1), e8m0(127));
    try std.testing.expectEqual(std.math.ldexp(@as(f64, 1.0), 127), e8m0(254));
    try std.testing.expect(std.math.isNan(e8m0(255)));
}

test "MXFP8 hand-decoded values, zeros, subnormals, NaN, and E8M0 extremes" {
    var codes: [4 * 32]u8 = @splat(0);
    codes[0] = 0x38; // 1
    codes[1] = 0xc0; // -2
    codes[2] = 0x80; // -0
    codes[4] = 0x01; // 2^-9
    codes[5] = 0x7f; // e4m3 NaN
    codes[32] = 0x38;
    codes[64] = 0x38;
    codes[96] = 0x38;
    const scales = [_]u8{ 127, 0, 254, 255 };
    const w = tensor2(.f8_e4m3, 4, 32, &codes);
    const s = tensor2(.u8, 4, 1, &scales);
    var out: [4 * 32]f64 = undefined;
    try mxfp8(w, s, all(4), all(32), &out);
    try std.testing.expectEqual(@as(f64, 1), out[0]);
    try std.testing.expectEqual(@as(f64, -2), out[1]);
    try std.testing.expect(std.math.signbit(out[2]) and out[2] == 0);
    try std.testing.expect(!std.math.signbit(out[3]) and out[3] == 0);
    try std.testing.expectEqual(std.math.ldexp(@as(f64, 1.0), -9), out[4]);
    try std.testing.expect(std.math.isNan(out[5]));
    try std.testing.expectEqual(e8m0(0), out[32]);
    try std.testing.expectEqual(e8m0(254), out[64]);
    try std.testing.expect(std.math.isNan(out[96]));
    var bf: [4 * 32]u16 = undefined;
    try mxfp8Bf16(w, s, all(4), all(32), &bf);
    try std.testing.expectEqual(@as(u16, 0x3f80), bf[0]);
    try std.testing.expectEqual(convert.bf16(-2), bf[1]);
    try std.testing.expectEqual(@as(u16, 0x8000), bf[2]);
    try std.testing.expectEqual(@as(u16, 0), bf[3]);
    try std.testing.expect(std.math.isNan(fromBf16(bf[96])));
}

test "MXFP8 block boundary and a column slice matches the whole matrix" {
    var codes: [2 * 64]u8 = @splat(0x38);
    codes[31] = 0xc0;
    codes[32] = 0x40;
    const scales = [_]u8{ 127, 128, 126, 129 }; // 1, 2, 0.5, 4
    const w = tensor2(.f8_e4m3, 2, 64, &codes);
    const s = tensor2(.u8, 2, 2, &scales);
    var full: [2 * 64]f64 = undefined;
    try mxfp8(w, s, all(2), all(64), &full);
    try std.testing.expectEqual(@as(f64, 1), full[0]);
    try std.testing.expectEqual(@as(f64, -2), full[31]);
    try std.testing.expectEqual(@as(f64, 4), full[32]);
    try std.testing.expectEqual(@as(f64, 0.5), full[64]);
    var part: [1 * 40]f64 = undefined;
    const rows: Span = .{ .lo = 1, .hi = 2 };
    const cols: Span = .{ .lo = 16, .hi = 56 };
    try mxfp8(w, s, rows, cols, &part);
    for (0..40) |i| try std.testing.expectEqual(full[64 + 16 + i], part[i]);
    var part_bf: [40]u16 = undefined;
    var full_bf: [2 * 64]u16 = undefined;
    try mxfp8Bf16(w, s, all(2), all(64), &full_bf);
    try mxfp8Bf16(w, s, rows, cols, &part_bf);
    for (0..40) |i| try std.testing.expectEqual(full_bf[64 + 16 + i], part_bf[i]);
}

test "FP8_PB_WO non-multiple-of-128 shapes, 4D scales, and slices" {
    var codes: [130 * 130]u8 = @splat(0x38);
    codes[128] = 0xc0;
    codes[128 * 130 + 128] = 0x40;
    const inv = [_]f32{ 0.5, 2, 3, 4 };
    const w = tensor2(.f8_e4m3, 130, 130, &codes);
    const s4 = tensor4(.f32, 2, 1, 2, 1, std.mem.sliceAsBytes(&inv));
    var full: [130 * 130]f64 = undefined;
    try pbWo(w, s4, all(130), all(130), &full);
    try std.testing.expectEqual(@as(f64, 0.5), full[0]);
    try std.testing.expectEqual(@as(f64, -4), full[128]);
    try std.testing.expectEqual(@as(f64, 3), full[128 * 130]);
    try std.testing.expectEqual(@as(f64, 8), full[128 * 130 + 128]);
    const s2 = tensor2(.f32, 2, 2, std.mem.sliceAsBytes(&inv));
    var again: [130 * 130]f64 = undefined;
    try pbWo(w, s2, all(130), all(130), &again);
    for (full, again) |a, b| try std.testing.expectEqual(a, b);
    var part: [10 * 10]f64 = undefined;
    const rows: Span = .{ .lo = 120, .hi = 130 };
    const cols: Span = .{ .lo = 120, .hi = 130 };
    try pbWo(w, s4, rows, cols, &part);
    for (0..10) |rr| for (0..10) |cc| {
        try std.testing.expectEqual(full[(120 + rr) * 130 + (120 + cc)], part[rr * 10 + cc]);
    };
    var full_bf: [130 * 130]u16 = undefined;
    var part_bf: [100]u16 = undefined;
    try pbWoBf16(w, s4, all(130), all(130), &full_bf);
    try pbWoBf16(w, s4, rows, cols, &part_bf);
    for (0..10) |rr| for (0..10) |cc| {
        try std.testing.expectEqual(full_bf[(120 + rr) * 130 + (120 + cc)], part_bf[rr * 10 + cc]);
    };
    try std.testing.expectEqual(convert.bf16(0.5), full_bf[0]);
    const halfway: f32 = @bitCast(@as(u32, 0x3F808000));
    var one: [1]u8 = .{0x38};
    var sc: [1]f32 = .{halfway};
    const w1 = tensor2(.f8_e4m3, 1, 1, &one);
    const s1 = tensor2(.f32, 1, 1, std.mem.sliceAsBytes(&sc));
    var bf1: [1]u16 = undefined;
    try pbWoBf16(w1, s1, all(1), all(1), &bf1);
    try std.testing.expectEqual(convert.bf16(halfway), bf1[0]);
    try std.testing.expectEqual(@as(u16, 0x3f80), bf1[0]);
}

test "source decode refuses dtype and shape mismatches" {
    var codes: [32]u8 = @splat(0x38);
    var u8s: [1]u8 = .{127};
    var f32s: [1]f32 = .{1};
    const w = tensor2(.f8_e4m3, 1, 32, &codes);
    const smx = tensor2(.u8, 1, 1, &u8s);
    var out: [32]f64 = undefined;
    try std.testing.expectError(error.BadFp8Source, mxfp8(tensor2(.u8, 1, 32, &codes), smx, all(1), all(32), &out));
    try std.testing.expectError(error.BadFp8Source, mxfp8(w, tensor2(.f32, 1, 1, std.mem.sliceAsBytes(&f32s)), all(1), all(32), &out));
    try std.testing.expectError(error.BadFp8Source, mxfp8(w, tensor2(.u8, 1, 2, &u8s), all(1), all(32), &out));
    try std.testing.expectError(error.BadFp8Source, mxfp8(w, smx, .{ .lo = 0, .hi = 2 }, all(32), &out));
    var tiny: [1]f64 = undefined;
    try std.testing.expectError(error.BadFp8Source, mxfp8(w, smx, all(1), all(32), &tiny));
    const wpb = tensor2(.f8_e4m3, 1, 1, codes[0..1]);
    try std.testing.expectError(error.BadFp8Source, pbWo(wpb, tensor2(.u8, 1, 1, &u8s), all(1), all(1), tiny[0..1]));
    try std.testing.expectError(error.BadFp8Source, pbWo(wpb, tensor4(.f32, 1, 1, 1, 2, std.mem.sliceAsBytes(&f32s)), all(1), all(1), tiny[0..1]));
}
