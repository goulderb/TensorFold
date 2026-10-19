//! GLM learned halves use a versioned checksum envelope; legacy, corrupt and partial files never enter a cache.
const std = @import("std");
const checked = @import("../../core/snapshot_file.zig");
const fsys = @import("lanes").learned_fs;
const magic: u32 = 0x474c4d54;
pub const Head = extern struct { magic: u32 = magic, version: u32 = 1, at: u32, reserved: u32 = 0, bytes: u64 };
pub fn write(file: [:0]const u8, at: u32, payload: []const u8) !void {
    var tmp_buf: [1200]u8 = undefined;
    const tmp = try std.fmt.bufPrintSentinel(&tmp_buf, "{s}.part", .{file}, 0);
    const fd = fsys.open(tmp, .{ .access = .write, .create = true, .trunc = true }) catch return error.SnapshotWrite;
    errdefer fsys.unlink(tmp) catch {};
    {
        defer fsys.close(fd);
        const head: Head = .{ .at = at, .bytes = payload.len };
        try checked.write(fd, std.mem.asBytes(&head), payload);
        if (!fsys.sync(fd)) return error.SnapshotWrite;
    }
    fsys.rename(tmp, file) catch return error.SnapshotWrite;
}
pub fn read(file: [:0]const u8, at: u32, payload: []u8) !void {
    const fd = fsys.open(file, .{}) catch return error.SnapshotRead;
    defer fsys.close(fd);
    const expected: Head = .{ .at = at, .bytes = payload.len };
    var head: Head = undefined;
    if (!fsys.readAt(fd, std.mem.asBytes(&head), 0) or !std.mem.eql(u8, std.mem.asBytes(&head), std.mem.asBytes(&expected))) return error.SnapshotRead;
    try checked.read(fd, std.mem.asBytes(&head), payload);
}
test "both GLM half files refuse a changed body, legacy header, truncation and another position" {
    var path: [160]u8 = undefined;
    const file = try fsys.tempRoot(&path, "glm-file");
    defer fsys.unlink(file) catch {};
    try write(file, 128, "body");
    var out: [4]u8 = undefined;
    try read(file, 128, &out);
    try std.testing.expectEqualStrings("body", &out);
    try std.testing.expectError(error.SnapshotRead, read(file, 129, &out));
    const fd = fsys.open(file, .{ .access = .read_write }) catch return error.TestFile;
    defer fsys.close(fd);
    try std.testing.expect(fsys.writeAt(fd, "X", @sizeOf(Head) + 1));
    try std.testing.expectError(error.SnapshotRead, read(file, 128, &out));
    try std.testing.expect(fsys.truncate(fd, 16));
    try std.testing.expectError(error.SnapshotRead, read(file, 128, &out));
}
