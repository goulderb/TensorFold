const std = @import("std");
pub const Pending = struct { start: usize, rows: usize };
pub const Lane = struct {
    end: usize = 0,
    pending: ?Pending = null,
    healthy: bool = true,
    pub fn validateStage(l: Lane, start: usize, rows: usize, capacity: usize) !void {
        if (!l.healthy) return error.DraftFailed;
        if (l.pending != null or start != l.end or rows == 0 or rows > 32 or start > capacity or rows > capacity - start) return error.BadDraftWindow;
    }
    pub fn validateAccept(l: Lane, kept: usize, target_end: usize) !void {
        if (!l.healthy) return error.DraftFailed;
        const p = l.pending orelse return error.NoDraftWindow;
        if (p.start != l.end or kept > p.rows or target_end != l.end + kept) return error.BadDraftCommit;
    }
    pub fn ready(l: Lane, end: usize) bool {
        return l.healthy and l.pending == null and l.end == end;
    }
};
pub fn tapOffset(plane_rows: usize, row0: usize, plane: usize, row: usize) usize {
    return (plane * plane_rows + row0 + row) * 8192;
}
pub fn ringOffset(streams: usize, slot: usize, head: usize) usize {
    return (head * (streams * 2176 + 64) + slot * 2176) * 256;
}
test "a consumed accepted prefix cannot be committed twice or publish rejected rows" {
    var l: Lane = .{ .end = 20 };
    try l.validateStage(20, 4, 64);
    l.pending = .{ .start = 20, .rows = 4 };
    try std.testing.expectError(error.BadDraftCommit, l.validateAccept(2, 24));
    try std.testing.expectError(error.BadDraftCommit, l.validateAccept(5, 25));
    try l.validateAccept(2, 22);
    l.end = 22;
    l.pending = null;
    try std.testing.expectError(error.NoDraftWindow, l.validateAccept(2, 24));
    try std.testing.expect(l.ready(22));
}
test "lane pending state and tap plane strides are independent" {
    var lanes: [8]Lane = @splat(.{});
    lanes[3].pending = .{ .start = 0, .rows = 4 };
    try std.testing.expect(lanes[2].ready(0));
    try std.testing.expect(!lanes[3].ready(0));
    try std.testing.expectEqual(@as(usize, 67 * 8192), tapOffset(32, 2, 2, 1));
    try std.testing.expectEqual(@as(usize, (8 * 2176 + 64 + 3 * 2176) * 256), ringOffset(8, 3, 1));
}
test "failed lanes and mismatched pending windows refuse continuation" {
    const l: Lane = .{ .end = 8, .healthy = false };
    try std.testing.expectError(error.DraftFailed, l.validateStage(8, 4, 64));
    try std.testing.expect(!l.ready(8));
    try std.testing.expectError(error.BadDraftWindow, (Lane{ .end = 9 }).validateStage(8, 4, 64));
}
