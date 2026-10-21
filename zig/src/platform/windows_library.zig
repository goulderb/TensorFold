//! Bare DLL names search System32 only. A file path must be fully qualified and resolves imports beside it or in System32.
const std = @import("std");
pub const Error = error{ FileNotFound, NameTooLong, InvalidWtf8, InvalidPath };
pub const search_system32: u32 = 0x00000800;
pub const search_dll_load_dir: u32 = 0x00000100;

pub fn open(comptime Api: type, path: []const u8) Error!Api.Handle {
    if (path.len == 0 or std.mem.indexOfScalar(u8, path, 0) != null) return error.InvalidPath;
    const absolute = (path.len >= 3 and std.ascii.isAlphabetic(path[0]) and path[1] == ':' and slash(path[2])) or
        (path.len >= 3 and slash(path[0]) and slash(path[1]));
    if (!absolute and std.mem.indexOfAny(u8, path, "/\\:") != null) return error.InvalidPath;
    var buf: [1024:0]u16 = undefined;
    const need = std.unicode.calcWtf16LeLen(path) catch return error.InvalidWtf8;
    if (need >= buf.len) return error.NameTooLong;
    const n = std.unicode.wtf8ToWtf16Le(buf[0 .. buf.len - 1], path) catch return error.InvalidWtf8;
    for (buf[0..n]) |*ch| if (ch.* == '/') {
        ch.* = '\\';
    };
    buf[n] = 0;
    const flags = search_system32 | if (absolute) search_dll_load_dir else @as(u32, 0);
    return Api.load(buf[0..n :0], flags) orelse error.FileNotFound;
}

fn slash(ch: u8) bool {
    return ch == '/' or ch == '\\';
}
