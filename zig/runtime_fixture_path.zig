//! Generated host-test fixtures must be fully qualified before the Windows confined loader.
const std = @import("std");

pub fn absolute(io: std.Io, allocator: std.mem.Allocator, path: []const u8) ![:0]u8 {
    if (path.len == 0 or std.mem.indexOfScalar(u8, path, 0) != null) return error.InvalidPath;
    if (std.fs.path.isAbsolute(path)) return try allocator.dupeSentinel(u8, path, 0);
    const real = try std.Io.Dir.cwd().realPathFileAlloc(io, path, allocator);
    defer allocator.free(real);
    return try allocator.dupeSentinel(u8, real, 0);
}
