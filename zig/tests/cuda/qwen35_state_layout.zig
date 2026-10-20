const std = @import("std");

pub fn canonicalize(bytes: []u8, scratch: []u8, deltas: usize, heads: usize, dim: usize) !void {
    const recurrence = deltas * heads * dim * dim * 4;
    if (bytes.len < recurrence or scratch.len < recurrence) return error.InvalidStateShape;
    @memcpy(scratch[0..recurrence], bytes[0..recurrence]);
    for (0..deltas * heads) |head| for (0..dim) |v| for (0..dim) |k| {
        const dst = ((head * dim + v) * dim + k) * 4;
        const src = ((head * dim + k) * dim + v) * 4;
        @memcpy(bytes[dst..][0..4], scratch[src..][0..4]);
    };
}

test "canonical recurrence transpose preserves all convolution tails" {
    var bytes: [80]u8 = undefined;
    for (&bytes, 0..) |*b, i| b.* = @intCast(i);
    const original = bytes;
    var scratch: [64]u8 = undefined;
    try canonicalize(&bytes, &scratch, 2, 2, 2);
    try std.testing.expectEqualSlices(u8, original[64..], bytes[64..]);
    try std.testing.expectEqualSlices(u8, original[8..12], bytes[4..8]);
    try std.testing.expect(!std.mem.eql(u8, &original, &bytes));
    try canonicalize(&bytes, &scratch, 2, 2, 2);
    try std.testing.expectEqualSlices(u8, &original, &bytes);
    bytes[79] ^= 1;
    try std.testing.expect(!std.mem.eql(u8, &original, &bytes));
}

test "canonical recurrence transpose rejects truncated state and scratch" {
    var bytes: [64]u8 = @splat(0);
    var scratch: [64]u8 = undefined;
    try std.testing.expectError(error.InvalidStateShape, canonicalize(bytes[0..63], &scratch, 2, 2, 2));
    try std.testing.expectError(error.InvalidStateShape, canonicalize(&bytes, scratch[0..63], 2, 2, 2));
}
