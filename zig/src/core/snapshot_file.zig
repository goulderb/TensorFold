//! Learned snapshots carry a SHA-256 of their entire header and payload; exact length is required on admission.
const std = @import("std");
const fsys = @import("lanes").learned_fs;

const Hash = std.crypto.hash.sha2.Sha256;

pub fn write(fd: fsys.Fd, header: []const u8, payload: []const u8) !void {
    var hash = Hash.init(.{});
    hash.update(header);
    hash.update(payload);
    try put(fd, header);
    try put(fd, payload);
    try put(fd, &hash.finalResult());
}

pub fn read(fd: fsys.Fd, header: []const u8, payload: []u8) !void {
    const length = std.math.add(usize, header.len, payload.len) catch return error.SnapshotRead;
    const total = std.math.add(usize, length, 32) catch return error.SnapshotRead;
    const size = fsys.size(fd) catch return error.SnapshotRead;
    if (size != total) return error.SnapshotRead;
    if (!get(fd, payload, header.len)) return error.SnapshotRead;
    var stored: [32]u8 = undefined;
    if (!get(fd, &stored, length)) return error.SnapshotRead;
    var hash = Hash.init(.{});
    hash.update(header);
    hash.update(payload);
    if (!std.mem.eql(u8, &stored, &hash.finalResult())) return error.SnapshotRead;
}

test "payload bit flips, header changes, truncation, legacy files and extra bytes are refused" {
    var dir_buf: [256]u8 = undefined;
    const name = try fsys.tempRoot(&dir_buf, "snapshot-test");
    defer fsys.unlink(name) catch {};
    var output: [4]u8 = undefined;
    {
        const fd = try fsys.open(name, .{ .access = .read_write, .create = true, .trunc = true });
        defer fsys.close(fd);
        try write(fd, "head", "body");
        try read(fd, "head", &output);
        try std.testing.expectEqualStrings("body", &output);
        try std.testing.expectError(error.SnapshotRead, read(fd, "dead", &output));
        try std.testing.expect(fsys.writeAt(fd, "X", 5));
        try std.testing.expectError(error.SnapshotRead, read(fd, "head", &output));
        try std.testing.expect(fsys.truncate(fd, 8));
        try std.testing.expectError(error.SnapshotRead, read(fd, "head", &output));
    }
    {
        const fd = try fsys.open(name, .{ .access = .read_write, .create = true, .trunc = true });
        defer fsys.close(fd);
        try write(fd, "head", "body");
        try put(fd, "x");
        try std.testing.expectError(error.SnapshotRead, read(fd, "head", &output));
    }
}

fn put(fd: fsys.Fd, bytes: []const u8) !void {
    fsys.write(fd, bytes) catch return error.SnapshotWrite;
}
fn get(fd: fsys.Fd, bytes: []u8, at: usize) bool {
    return fsys.readAt(fd, bytes, at);
}
