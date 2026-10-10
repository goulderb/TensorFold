//! The OS seams the server, the Flash Next link and the cache's memory check share, each with a Windows arm and a POSIX arm.
const std = @import("std");
const builtin = @import("builtin");
const Allocator = std.mem.Allocator;

pub const sock = @import("platform/sock.zig");

extern "c" fn read(fd: c_int, buf: [*]u8, n: usize) isize;
extern "c" fn open(path: [*:0]const u8, flags: c_int, ...) c_int;
extern "c" fn close(fd: c_int) c_int;

const MemoryStatusEx = extern struct {
    length: u32,
    load: u32,
    total_phys: u64,
    avail_phys: u64,
    total_page_file: u64,
    avail_page_file: u64,
    total_virtual: u64,
    avail_virtual: u64,
    avail_extended_virtual: u64,
};
extern "kernel32" fn GlobalMemoryStatusEx(buf: *MemoryStatusEx) callconv(.winapi) i32;

extern "kernel32" fn CreateFileW(name: [*:0]const u16, access: u32, share: u32, security: ?*anyopaque, disposition: u32, flags: u32, template: ?*anyopaque) callconv(.winapi) ?*anyopaque;
extern "kernel32" fn WriteFile(h: *anyopaque, buf: [*]const u8, n: u32, written: *u32, overlapped: ?*anyopaque) callconv(.winapi) i32;
extern "kernel32" fn CloseHandle(h: *anyopaque) callconv(.winapi) i32;
extern "kernel32" fn GetStdHandle(which: u32) callconv(.winapi) ?*anyopaque;

const win_invalid_handle = std.math.maxInt(usize);

/// Appends `bytes` to the file at `path`, created 0644 if absent, in one write. A failure is silent: the caller logs
/// nothing it could not write anyway.
pub fn appendFile(a: Allocator, path: []const u8, bytes: []const u8) void {
    if (builtin.os.tag == .windows) {
        const wide = std.unicode.utf8ToUtf16LeAllocZ(a, path) catch return;
        // FILE_APPEND_DATA alone makes every write land at the end, as O_APPEND does; OPEN_ALWAYS creates the file
        const h = CreateFileW(wide.ptr, 0x0004, 0x7, null, 4, 0x80, null) orelse return;
        defer _ = CloseHandle(h);
        if (@intFromPtr(h) == win_invalid_handle) return;
        var written: u32 = 0;
        _ = WriteFile(h, bytes.ptr, @intCast(bytes.len), &written, null);
        return;
    }
    const pathz = a.dupeSentinel(u8, path, 0) catch return;
    const fd = std.posix.openatZ(std.posix.AT.FDCWD, pathz, .{ .ACCMODE = .WRONLY, .APPEND = true, .CREAT = true }, 0o644) catch return;
    defer _ = std.posix.system.close(fd);
    _ = std.posix.system.write(fd, bytes.ptr, bytes.len);
}

/// Writes all of `bytes` to standard output, stopping quietly at the first failure (a closed pipe, say).
pub fn stdoutWrite(bytes: []const u8) void {
    if (builtin.os.tag == .windows) {
        const STD_OUTPUT_HANDLE: u32 = @bitCast(@as(i32, -11));
        const h = GetStdHandle(STD_OUTPUT_HANDLE) orelse return;
        if (@intFromPtr(h) == win_invalid_handle) return;
        var off: usize = 0;
        while (off < bytes.len) {
            var written: u32 = 0;
            const chunk: u32 = @intCast(@min(bytes.len - off, std.math.maxInt(u32)));
            if (WriteFile(h, bytes[off..].ptr, chunk, &written, null) == 0 or written == 0) return;
            off += written;
        }
        return;
    }
    var off: usize = 0;
    while (off < bytes.len) {
        const rc = std.posix.system.write(1, bytes[off..].ptr, bytes.len - off);
        if (std.posix.errno(rc) != .SUCCESS or rc <= 0) return;
        off += @intCast(rc);
    }
}

/// The memory a new allocation can use, in bytes, or null: MemAvailable from /proc/meminfo on Linux, the available
/// physical memory from GlobalMemoryStatusEx on Windows.
pub fn memAvailable() ?u64 {
    if (builtin.os.tag == .windows) {
        var st: MemoryStatusEx = undefined;
        st.length = @sizeOf(MemoryStatusEx);
        if (GlobalMemoryStatusEx(&st) == 0) return null;
        return st.avail_phys;
    }
    const fd = open("/proc/meminfo", 0);
    if (fd < 0) return null;
    defer _ = close(fd);
    var buf: [4096]u8 = undefined;
    const got = read(fd, &buf, buf.len);
    if (got <= 0) return null;
    var it = std.mem.tokenizeScalar(u8, buf[0..@intCast(got)], '\n');
    while (it.next()) |line| if (std.mem.startsWith(u8, line, "MemAvailable:")) {
        var f = std.mem.tokenizeAny(u8, line["MemAvailable:".len..], " \t");
        const kb = std.fmt.parseInt(u64, f.next() orelse return null, 10) catch return null;
        return kb * 1024;
    };
    return null;
}

test "the available memory is read on this system" {
    const avail = memAvailable() orelse return error.MemoryUnavailable;
    try std.testing.expect(avail > 0);
}
