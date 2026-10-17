//! The file work the snapshot and the rollback share: clones, hashed copies, whole writes and synced folders.
const std = @import("std");
const builtin = @import("builtin");
const Allocator = std.mem.Allocator;
const Sha256 = std.crypto.hash.sha2.Sha256;

const chunk = 4 << 20; // bytes a read or write moves at a time
const windows = builtin.os.tag == .windows;
// std.c.O is void on Windows, so the POSIX flag value is not compiled there.
const new_flags = if (windows) {} else std.c.O{ .ACCMODE = .WRONLY, .CREAT = true, .EXCL = true, .CLOEXEC = true };
const file_mode: std.c.mode_t = 0o644;

/// The folder at `path` opened to read (a descriptor to lock or sync), negative when it cannot be.
pub fn openFolder(path: [:0]const u8) std.c.fd_t {
    if (comptime windows) return winOpen(path, 0x80000000, 3, 0x02000000);
    return std.c.open(path, .{ .ACCMODE = .RDONLY, .DIRECTORY = true, .CLOEXEC = true }, @as(std.c.mode_t, 0));
}

extern "c" fn clonefile(src: [*:0]const u8, dst: [*:0]const u8, flags: u32) c_int;
const clone_nofollow: u32 = 1;
const ficlone: c_int = 0x40049409; // Linux's _IOW(0x94, 9, int)

/// A clone of `src` at the new path `dst` (true), or false where this file system makes no clones.
pub fn clone(src: [:0]const u8, dst: [:0]const u8) !bool {
    switch (builtin.os.tag) {
        .macos => {
            const rc = clonefile(src, dst, clone_nofollow);
            if (rc == 0) return true;
            return switch (std.c.errno(rc)) {
                .OPNOTSUPP, .XDEV, .INVAL => false,
                else => error.CloneFailed,
            };
        },
        .linux => {
            const in = std.c.open(src, .{ .ACCMODE = .RDONLY, .CLOEXEC = true }, @as(std.c.mode_t, 0));
            if (in < 0) return error.ReadFailed;
            defer _ = std.c.close(in);
            const out = std.c.open(dst, new_flags, file_mode);
            if (out < 0) return error.WriteFailed;
            const rc = std.c.ioctl(out, ficlone, in);
            const e = std.c.errno(rc);
            _ = std.c.close(out);
            if (rc == 0) return true;
            _ = std.c.unlink(dst);
            return switch (e) {
                .OPNOTSUPP, .XDEV, .INVAL, .NOTTY, .NOSYS, .BADF => false,
                else => error.CloneFailed,
            };
        },
        else => return false,
    }
}

/// SHA-256 of the file at `path`.
pub fn hashFile(path: [:0]const u8) ![32]u8 {
    if (comptime windows) return winHash(path);
    const fd = std.c.open(path, .{ .ACCMODE = .RDONLY, .CLOEXEC = true }, @as(std.c.mode_t, 0));
    if (fd < 0) return error.ReadFailed;
    defer _ = std.c.close(fd);
    const buf = try std.heap.page_allocator.alloc(u8, chunk);
    defer std.heap.page_allocator.free(buf);
    var h = Sha256.init(.{});
    while (try readSome(fd, buf)) |got| h.update(got);
    return h.finalResult();
}

/// `src` copied to the new file `dst`, synced; the SHA-256 of the bytes read from `src`.
pub fn copyHashed(src: [:0]const u8, dst: [:0]const u8) ![32]u8 {
    if (comptime windows) return winCopyHashed(src, dst);
    const in = std.c.open(src, .{ .ACCMODE = .RDONLY, .CLOEXEC = true }, @as(std.c.mode_t, 0));
    if (in < 0) return error.ReadFailed;
    defer _ = std.c.close(in);
    const out = std.c.open(dst, new_flags, file_mode);
    if (out < 0) return error.WriteFailed;
    defer _ = std.c.close(out);
    const buf = try std.heap.page_allocator.alloc(u8, chunk);
    defer std.heap.page_allocator.free(buf);
    var h = Sha256.init(.{});
    while (try readSome(in, buf)) |got| {
        h.update(got);
        try writeAll(out, got);
    }
    if (std.c.fsync(out) != 0) return error.WriteFailed;
    return h.finalResult();
}

/// The next bytes of `fd` in `buf`, or null at its end.
fn readSome(fd: std.c.fd_t, buf: []u8) !?[]u8 {
    while (true) {
        const n = std.c.read(fd, buf.ptr, buf.len);
        if (n < 0 and std.c.errno(n) == .INTR) continue;
        if (n < 0) return error.ReadFailed;
        return if (n == 0) null else buf[0..@intCast(n)];
    }
}

fn writeAll(fd: std.c.fd_t, bytes: []const u8) !void {
    var done: usize = 0;
    while (done < bytes.len) {
        const n = std.c.write(fd, bytes.ptr + done, bytes.len - done);
        if (n < 0 and std.c.errno(n) == .INTR) continue;
        if (n <= 0) return error.WriteFailed;
        done += @intCast(n);
    }
}

/// `bytes` to `path` through `path`.tmp, synced, then renamed over it.
pub fn writeWhole(a: Allocator, path: []const u8, bytes: []const u8) !void {
    if (comptime windows) return winWriteWhole(a, path, bytes);
    const tmp = try std.fmt.allocPrintSentinel(a, "{s}.tmp", .{path}, 0);
    const fd = std.c.open(tmp, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true, .CLOEXEC = true }, file_mode);
    if (fd < 0) return error.WriteFailed;
    {
        defer _ = std.c.close(fd);
        try writeAll(fd, bytes);
        if (std.c.fsync(fd) != 0) return error.WriteFailed;
    }
    try rename(tmp, try a.dupeSentinel(u8, path, 0));
}

pub fn rename(from: [:0]const u8, to: [:0]const u8) !void {
    if (comptime windows) return winRename(from, to);
    if (std.c.rename(from, to) != 0) return error.RenameFailed;
}

/// The folder's entries made durable (best effort: a file system that cannot sync a folder still has them).
pub fn syncDir(a: Allocator, dir: []const u8) void {
    const fd = openFolder(a.dupeSentinel(u8, dir, 0) catch return);
    if (comptime windows) {
        Win.sync(fd);
        return;
    }
    if (fd < 0) return;
    _ = std.c.fsync(fd);
    _ = std.c.close(fd);
}

/// Win32 copies of the snapshot file work. Empty on POSIX, so a HANDLE is never checked against an int descriptor.
const Win = if (!windows) struct {} else struct {
    // Local Win32 declarations. lanes/win_fs.zig already belongs to the lanes module, and a file can be in only one.
    const api = struct {
        const GENERIC_READ: u32 = 0x80000000;
        const GENERIC_WRITE: u32 = 0x40000000;
        const FILE_SHARE_ALL: u32 = 0x7;
        const OPEN_EXISTING: u32 = 3;
        const CREATE_ALWAYS: u32 = 2;
        const MOVEFILE_REPLACE_EXISTING: u32 = 0x1;
        const MOVEFILE_WRITE_THROUGH: u32 = 0x8;
        const Wide = [1024:0]u16;
        const Overlapped = extern struct { internal: usize, internal_high: usize, offset: u32, offset_high: u32, event: ?*anyopaque };

        extern "kernel32" fn CreateFileW(name: [*:0]const u16, access: u32, share: u32, security: ?*anyopaque, disposition: u32, flags: u32, template: ?*anyopaque) callconv(.winapi) ?*anyopaque;
        extern "kernel32" fn ReadFile(h: *anyopaque, buf: [*]u8, n: u32, read: *u32, overlapped: ?*Overlapped) callconv(.winapi) i32;
        extern "kernel32" fn WriteFile(h: *anyopaque, buf: [*]const u8, n: u32, written: *u32, overlapped: ?*Overlapped) callconv(.winapi) i32;
        extern "kernel32" fn CloseHandle(h: *anyopaque) callconv(.winapi) i32;
        extern "kernel32" fn FlushFileBuffers(h: *anyopaque) callconv(.winapi) i32;
        extern "kernel32" fn MoveFileExW(from: [*:0]const u16, to: [*:0]const u16, flags: u32) callconv(.winapi) i32;

        fn wide(path: []const u8, buf: *Wide) ?[:0]const u16 {
            const n = std.unicode.utf8ToUtf16Le(buf[0..1023], path) catch return null;
            buf[n] = 0;
            return buf[0..n :0];
        }
    };
    const invalid = std.math.maxInt(usize);

    fn sync(fd: std.c.fd_t) void {
        if (@intFromPtr(fd) == invalid) return;
        _ = api.FlushFileBuffers(fd);
        _ = api.CloseHandle(fd);
    }

    fn close(fd: std.c.fd_t) void {
        _ = api.CloseHandle(fd);
    }

    fn open(path: [:0]const u8, access: u32, disposition: u32, flags: u32) std.c.fd_t {
        var buf: api.Wide = undefined;
        const name = api.wide(path, &buf) orelse return @ptrFromInt(invalid);
        const h = api.CreateFileW(name.ptr, access, api.FILE_SHARE_ALL, null, disposition, flags, null) orelse return @ptrFromInt(invalid);
        return if (@intFromPtr(h) == invalid) @ptrFromInt(invalid) else h;
    }

    fn bad(h: std.c.fd_t) bool {
        return @intFromPtr(h) == invalid;
    }

    fn digest(h: std.c.fd_t) ![32]u8 {
        const buf = try std.heap.page_allocator.alloc(u8, chunk);
        defer std.heap.page_allocator.free(buf);
        var hasher = Sha256.init(.{});
        while (true) {
            var got: u32 = 0;
            if (api.ReadFile(h, buf.ptr, @intCast(buf.len), &got, null) == 0) return error.ReadFailed;
            if (got == 0) break;
            hasher.update(buf[0..got]);
        }
        return hasher.finalResult();
    }

    fn hash(path: [:0]const u8) ![32]u8 {
        const h = open(path, api.GENERIC_READ, api.OPEN_EXISTING, 0);
        if (bad(h)) return error.ReadFailed;
        defer _ = api.CloseHandle(h);
        return digest(h);
    }

    fn copyHashed(src: [:0]const u8, dst: [:0]const u8) ![32]u8 {
        const in = open(src, api.GENERIC_READ, api.OPEN_EXISTING, 0);
        if (bad(in)) return error.ReadFailed;
        defer _ = api.CloseHandle(in);
        const create_new: u32 = 1; // exclusive, as O_EXCL
        const out = open(dst, api.GENERIC_WRITE, create_new, 0x80);
        if (bad(out)) return error.WriteFailed;
        defer _ = api.CloseHandle(out);
        const buf = try std.heap.page_allocator.alloc(u8, chunk);
        defer std.heap.page_allocator.free(buf);
        var hasher = Sha256.init(.{});
        while (true) {
            var got: u32 = 0;
            if (api.ReadFile(in, buf.ptr, @intCast(buf.len), &got, null) == 0) return error.ReadFailed;
            if (got == 0) break;
            hasher.update(buf[0..got]);
            var done: u32 = 0;
            while (done < got) {
                var wrote: u32 = 0;
                if (api.WriteFile(out, buf[done..].ptr, got - done, &wrote, null) == 0 or wrote == 0) return error.WriteFailed;
                done += wrote;
            }
        }
        if (api.FlushFileBuffers(out) == 0) return error.WriteFailed;
        return hasher.finalResult();
    }

    fn writeWhole(a: Allocator, path: []const u8, bytes: []const u8) !void {
        const tmp = try std.fmt.allocPrintSentinel(a, "{s}.tmp", .{path}, 0);
        const out = open(tmp, api.GENERIC_WRITE, api.CREATE_ALWAYS, 0x80);
        if (bad(out)) return error.WriteFailed;
        {
            defer _ = api.CloseHandle(out);
            var done: usize = 0;
            while (done < bytes.len) {
                var wrote: u32 = 0;
                const n: u32 = @intCast(@min(bytes.len - done, std.math.maxInt(u32)));
                if (api.WriteFile(out, bytes[done..].ptr, n, &wrote, null) == 0 or wrote == 0) return error.WriteFailed;
                done += wrote;
            }
            if (api.FlushFileBuffers(out) == 0) return error.WriteFailed;
        }
        try move(tmp, try a.dupeSentinel(u8, path, 0));
    }

    fn move(from: [:0]const u8, to: [:0]const u8) !void {
        var from_buf: api.Wide = undefined;
        var to_buf: api.Wide = undefined;
        const src = api.wide(from, &from_buf) orelse return error.RenameFailed;
        const dst = api.wide(to, &to_buf) orelse return error.RenameFailed;
        if (api.MoveFileExW(src.ptr, dst.ptr, api.MOVEFILE_REPLACE_EXISTING | api.MOVEFILE_WRITE_THROUGH) == 0) return error.RenameFailed;
    }
};

fn winOpen(path: [:0]const u8, access: u32, disposition: u32, flags: u32) std.c.fd_t {
    if (comptime windows) return Win.open(path, access, disposition, flags);
    unreachable;
}

fn winHash(path: [:0]const u8) ![32]u8 {
    if (comptime windows) return Win.hash(path);
    unreachable;
}

fn winCopyHashed(src: [:0]const u8, dst: [:0]const u8) ![32]u8 {
    if (comptime windows) return Win.copyHashed(src, dst);
    unreachable;
}

fn winWriteWhole(a: Allocator, path: []const u8, bytes: []const u8) !void {
    if (comptime windows) return Win.writeWhole(a, path, bytes);
    unreachable;
}

fn winRename(from: [:0]const u8, to: [:0]const u8) !void {
    if (comptime windows) return Win.move(from, to);
    unreachable;
}

pub fn winClose(h: std.c.fd_t) void {
    if (comptime windows) Win.close(h);
}
