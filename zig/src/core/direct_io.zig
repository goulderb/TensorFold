//! Positional reads that skip the page cache (O_DIRECT on Linux, FILE_FLAG_NO_BUFFERING on Windows): 4 KiB-aligned spans into aligned buffers.

const std = @import("std");
const builtin = @import("builtin");

/// Offset, length and buffer alignment a direct read needs (the logical block size of every NVMe we run on).
pub const alignment = 4096;

pub const Buffer = []align(alignment) u8;

/// Bytes a read of `len` bytes at `offset` occupies once widened to aligned ends.
pub fn span(offset: u64, len: usize) usize {
    const lo = std.mem.alignBackward(u64, offset, alignment);
    return @intCast(std.mem.alignForward(u64, offset + len, alignment) - lo);
}

/// The largest read whose span always fits a buffer of `bytes` (whatever its offset).
pub fn fits(bytes: usize) usize {
    return bytes - 2 * alignment;
}

/// A file's handle: a descriptor on POSIX, a HANDLE on Windows.
pub const Fd = if (builtin.os.tag == .windows) win.HANDLE else std.c.fd_t;

const win = struct {
    pub const HANDLE = *anyopaque;
    pub const OVERLAPPED = extern struct { Internal: usize, InternalHigh: usize, Offset: u32, OffsetHigh: u32, hEvent: ?*anyopaque };
    extern "kernel32" fn CreateFileW(name: [*:0]const u16, access: u32, share: u32, sa: ?*anyopaque, disposition: u32, flags: u32, template: ?*anyopaque) callconv(.winapi) ?HANDLE;
    extern "kernel32" fn ReadFile(h: HANDLE, buf: [*]u8, n: u32, read: *u32, ov: *OVERLAPPED) callconv(.winapi) i32;
    extern "kernel32" fn CloseHandle(h: HANDLE) callconv(.winapi) i32;
    extern "kernel32" fn GetLastError() callconv(.winapi) u32;
    const handle_eof: u32 = 38;
    const invalid: usize = std.math.maxInt(usize);

    fn isInvalid(h: ?HANDLE) bool {
        return h == null or @intFromPtr(h.?) == invalid;
    }
};

pub const File = struct {
    fd: Fd,
    direct: bool,

    /// `path` read-only, direct when its file system allows it, else through the page cache.
    pub fn open(path: [:0]const u8) !File {
        if (builtin.os.tag == .windows) return openWindows(path);
        if (builtin.os.tag == .linux) {
            const fd = std.c.open(path, .{ .ACCMODE = .RDONLY, .CLOEXEC = true, .DIRECT = true });
            if (fd >= 0) return .{ .fd = fd, .direct = true };
        }
        const fd = std.c.open(path, .{ .ACCMODE = .RDONLY, .CLOEXEC = true });
        if (fd < 0) return error.FileNotFound;
        return .{ .fd = fd, .direct = false };
    }

    fn openWindows(path: [:0]const u8) !File {
        var buf: [1024:0]u16 = undefined;
        const n = std.unicode.utf8ToUtf16Le(&buf, path) catch return error.FileNotFound;
        if (n >= buf.len) return error.FileNotFound;
        buf[n] = 0;
        const read_only: u32 = 0x80000000;
        const share_read: u32 = 0x1;
        const open_existing: u32 = 3;
        const normal: u32 = 0x80;
        const no_buffering: u32 = 0x20000000;
        const direct_h = win.CreateFileW(buf[0..n :0], read_only, share_read, null, open_existing, normal | no_buffering, null);
        if (!win.isInvalid(direct_h)) return .{ .fd = direct_h.?, .direct = true };
        const h = win.CreateFileW(buf[0..n :0], read_only, share_read, null, open_existing, normal, null);
        if (win.isInvalid(h)) return error.FileNotFound;
        return .{ .fd = h.?, .direct = false };
    }

    pub fn close(f: *File) void {
        if (builtin.os.tag == .windows) _ = win.CloseHandle(f.fd) else _ = std.c.close(f.fd);
        f.* = undefined;
    }

    /// `len` bytes at `offset`, read as their aligned span into `buf`; the result is the requested bytes within it.
    pub fn read(f: File, buf: Buffer, offset: u64, len: usize) ![]u8 {
        var ignored: u8 = 0;
        return readAt(f, buf, offset, len, &ignored, sysPread);
    }
};

fn sysPread(_: *anyopaque, fd: Fd, buf: [*]u8, len: usize, offset: u64) isize {
    if (builtin.os.tag == .windows) return winPread(fd, buf, len, offset);
    return std.c.pread(fd, buf, len, @intCast(offset));
}

/// A positional read: the bytes read, 0 at the end of the file, -1 on failure.
fn winPread(h: Fd, buf: [*]u8, len: usize, offset: u64) isize {
    var ov: win.OVERLAPPED = .{ .Internal = 0, .InternalHigh = 0, .Offset = @truncate(offset), .OffsetHigh = @truncate(offset >> 32), .hEvent = null };
    var got: u32 = 0;
    if (win.ReadFile(h, buf, @intCast(len), &got, &ov) != 0) return got;
    return if (win.GetLastError() == win.handle_eof) 0 else -1;
}

fn readAt(f: File, buf: Buffer, offset: u64, len: usize, ctx: *anyopaque, pread: *const fn (*anyopaque, Fd, [*]u8, usize, u64) isize) ![]u8 {
    const lo = std.mem.alignBackward(u64, offset, alignment);
    const want: usize = @intCast(offset + len - lo);
    const whole = span(offset, len);
    if (whole > buf.len) return error.BufferTooSmall;
    var got: usize = 0;
    while (got < want) {
        // A short O_DIRECT read is retried from the last 4 KiB boundary so the next pread stays aligned.
        const at: usize = if (f.direct) std.mem.alignBackward(usize, got, alignment) else got;
        const ask = whole - at;
        const n = pread(ctx, f.fd, buf.ptr + at, ask, lo + @as(u64, @intCast(at)));
        if (n < 0) {
            if (builtin.os.tag != .windows and std.c.errno(n) == .INTR) continue;
            std.log.err("read of {d} bytes at {d} failed", .{ ask, lo + at });
            return error.ReadFailed;
        }
        if (n == 0) return error.EndOfFile;
        if (n > @as(isize, @intCast(ask))) return error.ReadFailed;
        const next = at + @as(usize, @intCast(n));
        if (next <= got) return error.ReadFailed;
        got = next;
    }
    return buf[@intCast(offset - lo)..][0..len];
}

test "span and fits" {
    try std.testing.expectEqual(@as(usize, 4096), span(0, 1));
    try std.testing.expectEqual(@as(usize, 8192), span(4095, 2));
    try std.testing.expectEqual(@as(usize, 4096), span(4096, 4096));
    try std.testing.expectEqual(@as(usize, 3 * 4096), span(100, 2 * 4096));
    try std.testing.expect(span(4095, fits(1 << 20)) <= 1 << 20);
}

test "direct reads return the same bytes as buffered reads" {
    if (builtin.os.tag != .linux) return error.SkipZigTest;
    const gpa = std.testing.allocator;
    var f = try File.open("/proc/self/exe");
    defer f.close();
    // The reference uses the page cache, so its unaligned reads do not inherit O_DIRECT.
    var plain: File = .{ .fd = std.c.open("/proc/self/exe", .{ .ACCMODE = .RDONLY, .CLOEXEC = true }), .direct = false };
    try std.testing.expect(plain.fd >= 0);
    defer plain.close();
    const buf = try gpa.alignedAlloc(u8, .fromByteUnits(alignment), 1 << 20);
    defer gpa.free(buf);
    const want = try gpa.alloc(u8, 1 << 19);
    defer gpa.free(want);
    for ([_][2]usize{ .{ 0, 1 }, .{ 1, 4095 }, .{ 4095, 2 }, .{ 12345, 100000 }, .{ 8192, 1 << 19 } }) |c| {
        const n = std.c.pread(plain.fd, want.ptr, c[1], @intCast(c[0]));
        try std.testing.expectEqual(@as(isize, @intCast(c[1])), n);
        try std.testing.expectEqualSlices(u8, want[0..c[1]], try f.read(buf, c[0], c[1]));
    }
    try std.testing.expectError(error.BufferTooSmall, f.read(buf, 1, 1 << 20));
}

const Script = struct {
    plan: []const usize,
    offs: [8]u64 = undefined,
    i: usize = 0,
    calls: usize = 0,
    bad: bool = false,

    fn pread(ctx: *anyopaque, _: Fd, buf: [*]u8, len: usize, offset: u64) isize {
        const s: *Script = @ptrCast(@alignCast(ctx));
        if (s.calls < s.offs.len) s.offs[s.calls] = offset;
        s.calls += 1;
        if (offset % alignment != 0 or len % alignment != 0 or @intFromPtr(buf) % alignment != 0) s.bad = true;
        const step = if (s.i < s.plan.len) s.plan[s.i] else len;
        if (s.i < s.plan.len) s.i += 1;
        const n = @min(step, len);
        var k: usize = 0;
        while (k < n) : (k += 1) buf[k] = @intCast((offset + k) % 251);
        return @intCast(n);
    }
};

fn direct(script: *Script, out: []u8, offset: u64) !void {
    var storage: [8192]u8 align(alignment) = undefined;
    const f: File = .{ .fd = undefined, .direct = true };
    const got = try readAt(f, &storage, offset, out.len, script, Script.pread);
    @memcpy(out, got);
}

test "a short direct read is retried from the aligned boundary" {
    var script: Script = .{ .plan = &.{ 100, 4096 } };
    var got: [4096]u8 = undefined;
    try direct(&script, &got, 0);
    try std.testing.expect(!script.bad);
    try std.testing.expectEqual(@as(usize, 2), script.calls);
    try std.testing.expectEqual(@as(u64, 0), script.offs[0]);
    try std.testing.expectEqual(@as(u64, 0), script.offs[1]);
    for (got, 0..) |b, i| try std.testing.expectEqual(@as(u8, @intCast(i % 251)), b);
    var boundary: Script = .{ .plan = &.{ 4096, 4096 } };
    var wide: [8192]u8 = undefined;
    try direct(&boundary, &wide, 0);
    try std.testing.expect(!boundary.bad);
    try std.testing.expectEqual(@as(u64, 0), boundary.offs[0]);
    try std.testing.expectEqual(@as(u64, 4096), boundary.offs[1]);
    try std.testing.expectEqual(@as(u8, @intCast(4096 % 251)), wide[4096]);
}
