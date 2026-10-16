//! A token list as Python's json.dumps writes it, "[1, 2, 3]": the text every engine's token SHA is taken over.
const std = @import("std");

/// The caller owns the text.
pub fn write(gpa: std.mem.Allocator, ids: []const u32) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(gpa);
    try out.append(gpa, '[');
    for (ids, 0..) |t, i| {
        if (i > 0) try out.appendSlice(gpa, ", ");
        try out.print(gpa, "{d}", .{t});
    }
    try out.append(gpa, ']');
    return out.toOwnedSlice(gpa);
}

test "matches json.dumps" {
    const text = try write(std.testing.allocator, &.{ 1, 22, 333 });
    defer std.testing.allocator.free(text);
    try std.testing.expectEqualStrings("[1, 22, 333]", text);
}

// The same vectors pin server/qual_ids.zig's copy of this writer (the server's module cannot share this file): an edit
// to either side that changes any of these fails that side's test.
test "the shared canonical vectors: empty, zero, the largest id, and the reference list with its SHA-256" {
    const vectors = [_]struct { ids: []const u32, text: []const u8 }{
        .{ .ids = &.{}, .text = "[]" },
        .{ .ids = &.{0}, .text = "[0]" },
        .{ .ids = &.{4294967295}, .text = "[4294967295]" },
        .{ .ids = &.{ 1, 22, 333 }, .text = "[1, 22, 333]" },
    };
    for (vectors) |v| {
        const text = try write(std.testing.allocator, v.ids);
        defer std.testing.allocator.free(text);
        try std.testing.expectEqualStrings(v.text, text);
    }
    var d: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash("[1, 22, 333]", &d, .{});
    try std.testing.expectEqualStrings("276d8737d9d5a80805e0731e8d5ae18972a8e3ba721f4988f67151c6e9159068", &std.fmt.bytesToHex(d, .lower));
}
