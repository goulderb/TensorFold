const std = @import("std");

pub const Dense = enum { bf16, q4, fp8 };
pub const Head = enum { bf16, fp8 };
pub const Settings = struct {
    dense: Dense = .bf16,
    head: Head = .bf16,
};

pub fn parseDense(value: ?[]const u8) !Dense {
    const v = value orelse return .bf16;
    if (v.len == 0) return .bf16;
    return std.meta.stringToEnum(Dense, v) orelse error.InvalidDensePrecision;
}

pub fn parseHead(value: ?[]const u8) !Head {
    const v = value orelse return .bf16;
    if (v.len == 0) return .bf16;
    return std.meta.stringToEnum(Head, v) orelse error.InvalidHeadPrecision;
}

test "precision is explicit and defaults to checkpoint BF16" {
    try std.testing.expectEqual(Dense.bf16, try parseDense(null));
    try std.testing.expectEqual(Head.bf16, try parseHead(""));
    try std.testing.expectEqual(Dense.q4, try parseDense("q4"));
    try std.testing.expectEqual(Head.fp8, try parseHead("fp8"));
    try std.testing.expectEqual(Dense.fp8, try parseDense("fp8"));
    try std.testing.expectError(error.InvalidDensePrecision, parseDense("int8"));
    try std.testing.expectError(error.InvalidHeadPrecision, parseHead("q4"));
}
