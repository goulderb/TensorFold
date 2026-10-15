const std = @import("std");
const ref = @import("cuda_vision_forward_ops.zig");
const t = std.testing;
// Offline CPU Torch 2.9.0+cu128, git 0fabc3ba44823f257e70ce397d989c8de5e362c1.
// CUDA_VISIBLE_DEVICES='', CPU tensors, one thread. These finite equation fixtures
// qualify this CPU reference only, not target CUDA reduction/rsqrt/exp/erf bits.
const x = [_]u16{ 16288, 49184, 16192, 16456, 48896, 15872, 16592, 49304, 16128, 48960, 16320, 16400, 49224, 16000, 15744, 49024 };
const weight = [_]u16{ 16272, 16192, 49088, 16128, 16400, 16256, 48992, 16352 };
const bias = [_]u16{ 15872, 48768, 16128, 16192, 49024, 0, 16064, 48896 };

test "vision equations BF16 nearest-even ties, signed zero and cast" {
    try t.expectEqual(@as(u16, 0x3f80), ref.bf16(@bitCast(@as(u32, 0x3f808000))));
    try t.expectEqual(@as(u16, 0x3f82), ref.bf16(@bitCast(@as(u32, 0x3f818000))));
    try t.expectEqual(@as(u16, 0xbf80), ref.bf16(@bitCast(@as(u32, 0xbf808000))));
    try t.expectEqual(@as(u16, 0x8000), ref.bf16(-@as(f32, 0)));
    try t.expectEqual(@as(u16, 0x7fc0), ref.bf16(std.math.nan(f32)));
    var out: [4]u16 = undefined;
    try ref.cast(&.{ 0, -@as(f32, 0), 1, -2 }, &out);
    try t.expectEqualSlices(u16, &.{ 0, 0x8000, 0x3f80, 0xc000 }, &out);
}
test "vision equations RMS preserves two BF16 rounding boundaries, unlike single-round multiplication" {
    var out: [16]u16 = undefined;
    try ref.rms(.{ .rows = 2, .cols = 8 }, 1e-6, &x, &weight, &out);
    const golden = [_]u16{ 16095, 48916, 48818, 16120, 48819, 15647, 49122, 49189, 16059, 48827, 49083, 16187, 49298, 15910, 48401, 49041 };
    try t.expectEqualSlices(u16, &golden, &out);
    // Torch single-round controls differ at these coordinates.
    try t.expect(out[1] != 48917 and out[2] != 48819 and out[14] != 48402 and out[15] != 49042);
    var inplace = x;
    try ref.rms(.{ .rows = 2, .cols = 8 }, 1e-6, &inplace, &weight, &inplace);
    try t.expectEqualSlices(u16, &golden, &inplace);
}
test "vision equations QKV exact unpack, head and group indexing, half rotation and V bit preservation" {
    const shape: ref.native.Qkv = .{ .groups = 2, .sequence = 4, .heads = 2, .dim = 4 };
    var dense: [192]u16 = undefined;
    for (0..8) |row| for (0..2) |head| {
        for (0..4) |j| {
            dense[row * 24 + head * 4 + j] = ref.bf16(if (j % 2 == 0) @as(f32, 1) else -1);
            dense[row * 24 + 8 + head * 4 + j] = ref.bf16(if (j < 2) @as(f32, 1) else -1);
            dense[row * 24 + 16 + head * 4 + j] = @intCast(0x3f00 + row * 8 + head * 4 + j);
        }
    };
    var co: [16]f32 = undefined;
    var si: [16]f32 = undefined;
    for (0..4) |row| for (0..4) |j| {
        co[row * 4 + j] = if (row % 2 == 0) 1 else 0;
        si[row * 4 + j] = if (row % 2 == 0) 0 else 1;
    };
    const w = [_]u16{ 0x3f80, 0x3f80, 0x3f80, 0x3f80 };
    var q: [64]u16 = undefined;
    var k: [64]u16 = undefined;
    var v: [64]u16 = undefined;
    try ref.qkvRope(shape, 1e-6, &dense, &w, &w, &co, &si, &q, &k, &v);
    for (0..8) |row| for (0..2) |head| for (0..4) |j| {
        const other = if (row % 2 == 0) j else (j + 2) % 4;
        const sign: f32 = if (row % 2 == 1 and j < 2) -1 else 1;
        try t.expectEqual(ref.bf16(sign * ref.float(dense[row * 24 + head * 4 + other])), q[row * 8 + head * 4 + j]);
        try t.expectEqual(ref.bf16(sign * ref.float(dense[row * 24 + 8 + head * 4 + other])), k[row * 8 + head * 4 + j]);
        try t.expectEqual(dense[row * 24 + 16 + head * 4 + j], v[row * 8 + head * 4 + j]);
    };
}
test "vision equations residual and clamp SiLU product BF16 stages match offline Torch" {
    var residual: [3]u16 = undefined;
    try ref.residual(&.{ 0x3f80, 0x3f81, 0xbf80 }, &.{ 0x3b80, 0x3b80, 0x3f80 }, &residual);
    try t.expectEqualSlices(u16, &.{ 0x3f80, 0x3f82, 0 }, &residual);
    const gate = [_]u16{ 49864, 49472, 49440, 49280, 49152, 49024, 48640, 0, 15872, 16256, 16384, 16512, 16672, 16704 };
    const up = [_]u16{ 16704, 49472, 16416, 16192, 49088, 16616, 15744, 48896, 16272, 49184, 16496, 16704, 49472, 15872 };
    var out: [14]u16 = undefined;
    try ref.swiglu(&gate, &up, &out, 10);
    try t.expectEqualSlices(u16, &.{ 32768, 14914, 47765, 48476, 16055, 49146, 47984, 32768, 15769, 49130, 16595, 16925, 49864, 16288 }, &out);
}
test "vision equations projector population LayerNorm then BF16 then exact-erf GELU matches offline Torch" {
    var out: [16]u16 = undefined;
    try ref.projectorNormGelu(.{ .rows = 2, .cols = 8 }, &x, &weight, &bias, &out);
    const golden = [_]u16{ 16002, 48678, 15998, 16259, 48538, 48474, 48644, 47771, 16058, 48680, 48674, 16306, 46128, 15837, 15946, 48565 };
    // CPU Torch's vector erf and Windows libc erff differ near negative-tail
    // cancellation: coordinate 12 is -1.639e-7 in Torch and -0 in libc.
    // Every other coordinate agrees in BF16 bits. This is explicitly not a
    // bitwise GELU/LayerNorm guarantee for libdevice or another reduction tree.
    for (golden, out, 0..) |expected, actual, i| {
        if (i == 12) try t.expectApproxEqAbs(ref.float(expected), ref.float(actual), 2e-7) else try t.expectEqual(expected, actual);
    }
    // Removing the BF16 LayerNorm output cast gives Torch control values
    // 16258/47775/15947 at these coordinates, rather than the staged result.
    try t.expect(out[3] != 16258 and out[7] != 47775 and out[14] != 15947);
    var shifted: [4]u16 = undefined;
    try ref.projectorNormGelu(.{ .rows = 1, .cols = 4 }, &.{ 18304, 18305, 18303, 18306 }, &.{ 0x3f80, 0x3f80, 0x3f80, 0x3f80 }, &.{ 0, 0, 0, 0 }, &shifted);
    try t.expectEqualSlices(u16, &.{ 48684, 16002, 48657, 16297 }, &shifted);
    var tiny: [4]u16 = undefined;
    try ref.projectorNormGelu(.{ .rows = 1, .cols = 4 }, &.{ 0, 14976, 47744, 15104 }, &.{ 0x3f80, 0x3f80, 0x3f80, 0x3f80 }, &.{ 0, 0, 0, 0 }, &tiny);
    try t.expectEqualSlices(u16, &.{ 48516, 15782, 48660, 16022 }, &tiny);
    var constant: [4]u16 = undefined;
    try ref.projectorNormGelu(.{ .rows = 1, .cols = 4 }, &.{ 0x4780, 0x4780, 0x4780, 0x4780 }, &.{ 0x3f80, 0x3f80, 0x3f80, 0x3f80 }, &.{ 0, 0, 0, 0 }, &constant);
    try t.expectEqualSlices(u16, &.{ 0, 0, 0, 0 }, &constant);
}
test "vision equations production-width reduction fixtures are CPU-only bit measurements" {
    var input: [1024]u16 = undefined;
    var w: [1024]u16 = undefined;
    var out: [1024]u16 = undefined;
    for (0..1024) |j| {
        input[j] = ref.bf16(@as(f32, @floatFromInt(@as(i32, @intCast((j * 37) % 113)) - 56)) / 16);
        w[j] = ref.bf16(1 + @as(f32, @floatFromInt(@as(i32, @intCast((j * 13) % 17)) - 8)) / 32);
    }
    try ref.rms(.{ .rows = 1, .cols = 1024 }, 1e-6, &input, &w, &out);
    var hash: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(std.mem.sliceAsBytes(&out), &hash, .{});
    var rms_bytes: [32]u8 = undefined;
    const rms_hash = try std.fmt.hexToBytes(&rms_bytes, "afcad92520dc7d0889a1b776e168ed2388029cb96cbee3e5577c2388a75e0259");
    try t.expectEqualSlices(u8, rms_hash, &hash);
    const zero: [1024]u16 = @splat(0);
    try ref.projectorNormGelu(.{ .rows = 1, .cols = 1024 }, &input, &w, &zero, &out);
    std.crypto.hash.sha2.Sha256.hash(std.mem.sliceAsBytes(&out), &hash, .{});
    var ln_bytes: [32]u8 = undefined;
    const ln_hash = try std.fmt.hexToBytes(&ln_bytes, "4ec3c026fccdcfd34da95be59fd2c57d507fde2bc73db6e137f9b4b983776f0d");
    try t.expectEqualSlices(u8, ln_hash, &hash);
}
test "vision equations nontrivial FP32 RoPE and weighted QK RMS match offline Torch CPU hashes" {
    var dense: [192]u16 = undefined;
    for (0..192) |j| dense[j] = ref.bf16(@as(f32, @floatFromInt(@as(i32, @intCast((j * 17) % 37)) - 18)) / 8);
    const qw = [_]u16{ 0x3f90, 0xbf40, 0x3fc0, 0x3f00 };
    const kw = [_]u16{ 0x3f40, 0x3fa0, 0xbf00, 0x3fe0 };
    const co = [_]f32{ 1, 0.8, 1, 0.8, 0.8, 0.6, 0.8, 0.6, 0.6, 0, 0.6, 0, 0, -0.6, 0, -0.6 };
    const si = [_]f32{ 0, 0.6, 0, 0.6, 0.6, 0.8, 0.6, 0.8, 0.8, 1, 0.8, 1, 1, 0.8, 1, 0.8 };
    var q: [64]u16 = undefined;
    var k: [64]u16 = undefined;
    var v: [64]u16 = undefined;
    try ref.qkvRope(.{ .groups = 2, .sequence = 4, .heads = 2, .dim = 4 }, 1e-6, &dense, &qw, &kw, &co, &si, &q, &k, &v);
    var expected: [32]u8 = undefined;
    var actual: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(std.mem.sliceAsBytes(&q), &actual, .{});
    _ = try std.fmt.hexToBytes(&expected, "ec9ca6a6f67fcfeba360a96bd066fadd185c7bfa5bbc8377c78a87ddf737760b");
    try t.expectEqualSlices(u8, &expected, &actual);
    std.crypto.hash.sha2.Sha256.hash(std.mem.sliceAsBytes(&k), &actual, .{});
    _ = try std.fmt.hexToBytes(&expected, "e4416b5b07e1884d0b823799ab17e413ef4792fa4e3fe97619e1c2dc91371094");
    try t.expectEqualSlices(u8, &expected, &actual);
}
test "vision equations invalid shapes scalars and nonfinite reference inputs refuse" {
    var out: [4]u16 = undefined;
    try t.expectError(error.InvalidOperatorShape, ref.cast(&.{}, &out));
    try t.expectError(error.NonFiniteVisionReferenceInput, ref.cast(&.{ std.math.inf(f32), 0, 0, 0 }, &out));
    try t.expectError(error.InvalidOperatorScalar, ref.rms(.{ .rows = 1, .cols = 4 }, 0, &.{ 0, 0, 0, 0 }, &.{ 0, 0, 0, 0 }, &out));
    try t.expectError(error.NonFiniteVisionReferenceInput, ref.rms(.{ .rows = 1, .cols = 4 }, 1e-6, &.{ 0x7f80, 0, 0, 0 }, &.{ 0, 0, 0, 0 }, &out));
    try t.expectError(error.InvalidOperatorScalar, ref.swiglu(&.{ 0, 0, 0, 0 }, &.{ 0, 0, 0, 0 }, &out, std.math.nan(f32)));
    try t.expectError(error.NonFiniteVisionReferenceInput, ref.projectorNormGelu(.{ .rows = 1, .cols = 4 }, &.{ 0, 0, 0, 0 }, &.{ 0, 0, 0x7fc0, 0 }, &.{ 0, 0, 0, 0 }, &out));
}
