//! GLM latent and pooled-index cache rows. FP8 uses the deployed kv8.py byte layout.
const std = @import("std");

pub const Format = enum(u32) {
    bf16,
    fp8,

    pub fn rowBytes(format: Format, width: usize) usize {
        return switch (format) {
            .bf16 => width * 2,
            .fp8 => width + 16,
        };
    }
};

pub const Cache = struct { ptr: u64, format: Format = .bf16 };

test "production cache row strides include scales and alignment" {
    try std.testing.expectEqual(@as(usize, 1024), Format.bf16.rowBytes(512));
    try std.testing.expectEqual(@as(usize, 528), Format.fp8.rowBytes(512));
    try std.testing.expectEqual(@as(usize, 256), Format.bf16.rowBytes(128));
    try std.testing.expectEqual(@as(usize, 144), Format.fp8.rowBytes(128));
    for ([_]usize{ 128, 512 }) |width| try std.testing.expectEqual(@as(usize, 0), Format.fp8.rowBytes(width) % 16);
}
