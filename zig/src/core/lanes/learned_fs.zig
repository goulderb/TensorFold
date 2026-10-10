//! The learned-state file layer: POSIX calls, or Win32 with owner-only DACLs where POSIX uses modes 0600 and 0700.
const std = @import("std");
const builtin = @import("builtin");
const w = @import("win_fs.zig");
const win = builtin.os.tag == .windows;

pub const Error = error{ Missing, Exists, Failed };
pub const Fd = if (win) w.HANDLE else c_int;
pub const Access = enum { read, write, read_write };
pub const Open = struct { access: Access = .read, create: bool = false, trunc: bool = false, append: bool = false, nofollow: bool = false, mode: u16 = 0o600 };
pub const Kind = enum { file, dir, other };
pub const Entry = struct { name: []const u8, kind: Kind };
pub const Dir = if (win) struct { h: w.HANDLE, data: w.FindData, first: bool, name: [780]u8 = undefined } else struct { d: *anyopaque };

pub fn pid() u64 {
    return if (win) w.GetCurrentProcessId() else @intCast(std.c.getpid());
}

/// A scratch path `tf-<tag>-<pid>` under the temp directory (/tmp, or %TEMP% on Windows).
pub fn tempRoot(buf: []u8, tag: []const u8) ![:0]const u8 {
    if (!win) return std.fmt.bufPrintSentinel(buf, "/tmp/tf-{s}-{d}", .{ tag, pid() }, 0);
    var wide_dir: [260]u16 = undefined;
    const n = w.GetTempPathW(wide_dir.len, &wide_dir);
    if (n == 0 or n >= wide_dir.len) return error.Failed;
    var dir: [780]u8 = undefined;
    const len = try std.unicode.utf16LeToUtf8(&dir, wide_dir[0..n]);
    return std.fmt.bufPrintSentinel(buf, "{s}tf-{s}-{d}", .{ dir[0..len], tag, pid() }, 0);
}

pub fn open(path: [:0]const u8, o: Open) Error!Fd {
    return if (win) openWin(path, o) else openPosix(path, o);
}

fn openPosix(path: [:0]const u8, o: Open) Error!c_int {
    const flags: std.c.O = .{
        .ACCMODE = switch (o.access) {
            .read => .RDONLY,
            .write => .WRONLY,
            .read_write => .RDWR,
        },
        .CREAT = o.create,
        .TRUNC = o.trunc,
        .APPEND = o.append,
        .NOFOLLOW = o.nofollow,
    };
    const fd = std.c.open(path, flags, @as(std.c.mode_t, o.mode));
    if (fd < 0) return if (std.c.errno(fd) == .NOENT) error.Missing else error.Failed;
    return fd;
}

fn openWin(path: [:0]const u8, o: Open) Error!w.HANDLE {
    var buf: w.Wide = undefined;
    const p = w.wide(path, &buf) orelse return error.Failed;
    const access: u32 = switch (o.access) {
        .read => w.GENERIC_READ,
        .write => w.GENERIC_WRITE,
        .read_write => w.GENERIC_READ | w.GENERIC_WRITE,
    };
    const disposition: u32 = if (o.create and o.trunc) w.CREATE_ALWAYS else if (o.create) w.OPEN_ALWAYS else if (o.trunc) w.TRUNCATE_EXISTING else w.OPEN_EXISTING;
    const flags: u32 = if (o.nofollow) w.FILE_FLAG_OPEN_REPARSE_POINT else 0;
    const h = w.valid(w.CreateFileW(p.ptr, access, w.FILE_SHARE_ALL, null, disposition, flags, null)) orelse return mapErr(w.lastError());
    errdefer _ = w.CloseHandle(h);
    if (o.nofollow and w.isLink(h)) return error.Failed;
    if (o.create and !w.privatize(p, false)) return error.Failed;
    if (o.append and w.SetFilePointerEx(h, 0, null, w.FILE_END) == 0) return error.Failed;
    return h;
}

fn mapErr(code: u32) Error {
    return switch (code) {
        w.ERROR_FILE_NOT_FOUND, w.ERROR_PATH_NOT_FOUND => error.Missing,
        w.ERROR_FILE_EXISTS, w.ERROR_ALREADY_EXISTS => error.Exists,
        else => error.Failed,
    };
}

pub fn close(fd: Fd) void {
    if (win) _ = w.CloseHandle(fd) else _ = std.c.close(fd);
}

/// The file's length in bytes.
pub fn size(fd: Fd) Error!u64 {
    if (win) {
        var n: i64 = 0;
        if (w.GetFileSizeEx(fd, &n) == 0 or n < 0) return error.Failed;
        return @intCast(n);
    }
    const end = std.c.lseek(fd, 0, std.c.SEEK.END);
    if (end < 0) return error.Failed;
    return @intCast(end);
}

/// Cuts or extends the file to `n` bytes.
pub fn truncate(fd: Fd, n: u64) bool {
    if (win) return w.SetFilePointerEx(fd, @intCast(n), null, w.FILE_BEGIN) != 0 and w.SetEndOfFile(fd) != 0;
    return std.c.ftruncate(fd, @intCast(n)) == 0;
}

/// Flushes the file to stable storage (fsync, FlushFileBuffers).
pub fn sync(fd: Fd) bool {
    if (win) return w.FlushFileBuffers(fd) != 0;
    return std.c.fsync(fd) == 0;
}

/// Reads up to `dest.len` bytes at the current position; 0 at the end.
pub fn read(fd: Fd, dest: []u8) Error!usize {
    if (win) {
        var got: u32 = 0;
        const n: u32 = @intCast(@min(dest.len, std.math.maxInt(u32)));
        if (w.ReadFile(fd, dest.ptr, n, &got, null) == 0) return error.Failed;
        return got;
    }
    while (true) {
        const n = std.c.read(fd, dest.ptr, dest.len);
        if (n >= 0) return @intCast(n);
        if (std.c.errno(n) != .INTR) return error.Failed;
    }
}

/// Fills `dest` from offset `at`; false at the end or on an I/O error.
pub fn readAt(fd: Fd, dest: []u8, at: u64) bool {
    var done: usize = 0;
    while (done < dest.len) {
        if (win) {
            const pos = at + done;
            var ov: w.Overlapped = .{ .internal = 0, .internal_high = 0, .offset = @truncate(pos), .offset_high = @truncate(pos >> 32), .event = null };
            var got: u32 = 0;
            const n: u32 = @intCast(@min(dest.len - done, std.math.maxInt(u32)));
            if (w.ReadFile(fd, dest.ptr + done, n, &got, &ov) == 0 or got == 0) return false;
            done += got;
        } else {
            const n = std.c.pread(fd, dest.ptr + done, dest.len - done, @intCast(at + done));
            if (n <= 0) return false;
            done += @intCast(n);
        }
    }
    return true;
}

/// Writes all of `bytes` at the current position (the end, for an append open).
pub fn write(fd: Fd, bytes: []const u8) Error!void {
    var done: usize = 0;
    while (done < bytes.len) {
        if (win) {
            var n: u32 = 0;
            const want: u32 = @intCast(@min(bytes.len - done, std.math.maxInt(u32)));
            if (w.WriteFile(fd, bytes.ptr + done, want, &n, null) == 0 or n == 0) return error.Failed;
            done += n;
        } else {
            const n = std.c.write(fd, bytes.ptr + done, bytes.len - done);
            if (n <= 0) return error.Failed;
            done += @intCast(n);
        }
    }
}

/// Writes all of `bytes` at offset `at`; false on an I/O error.
pub fn writeAt(fd: Fd, bytes: []const u8, at: u64) bool {
    var done: usize = 0;
    while (done < bytes.len) {
        if (win) {
            const pos = at + done;
            var ov: w.Overlapped = .{ .internal = 0, .internal_high = 0, .offset = @truncate(pos), .offset_high = @truncate(pos >> 32), .event = null };
            var n: u32 = 0;
            const want: u32 = @intCast(@min(bytes.len - done, std.math.maxInt(u32)));
            if (w.WriteFile(fd, bytes.ptr + done, want, &n, &ov) == 0 or n == 0) return false;
            done += n;
        } else {
            const n = std.c.pwrite(fd, bytes.ptr + done, bytes.len - done, @intCast(at + done));
            if (n <= 0) return false;
            done += @intCast(n);
        }
    }
    return true;
}

pub fn unlink(path: [:0]const u8) Error!void {
    if (win) {
        var buf: w.Wide = undefined;
        const p = w.wide(path, &buf) orelse return error.Failed;
        if (w.DeleteFileW(p.ptr) != 0) return;
        return mapErr(w.lastError());
    }
    const rc = std.c.unlink(path);
    if (rc == 0) return;
    return if (std.c.errno(rc) == .NOENT) error.Missing else error.Failed;
}

pub fn rmdir(path: [:0]const u8) Error!void {
    if (win) {
        var buf: w.Wide = undefined;
        const p = w.wide(path, &buf) orelse return error.Failed;
        if (w.RemoveDirectoryW(p.ptr) != 0) return;
        return mapErr(w.lastError());
    }
    const rc = std.c.rmdir(path);
    if (rc == 0) return;
    return if (std.c.errno(rc) == .NOENT) error.Missing else error.Failed;
}

/// One directory, owner-only (mode 0700 on POSIX).
pub fn mkdir(path: [:0]const u8) Error!void {
    if (win) {
        var buf: w.Wide = undefined;
        const p = w.wide(path, &buf) orelse return error.Failed;
        if (w.CreateDirectoryW(p.ptr, null) == 0) return mapErr(w.lastError());
        if (!w.privatize(p, true)) return error.Failed;
        return;
    }
    const rc = std.c.mkdir(path, 0o700);
    if (rc == 0) return;
    return if (std.c.errno(rc) == .EXIST) error.Exists else error.Failed;
}

/// Makes `to` name the file `from` and drops the old `to` in one step: rename, or MoveFileExW with write-through.
pub fn rename(from: [:0]const u8, to: [:0]const u8) Error!void {
    if (win) {
        var a: w.Wide = undefined;
        var b: w.Wide = undefined;
        const pa = w.wide(from, &a) orelse return error.Failed;
        const pb = w.wide(to, &b) orelse return error.Failed;
        if (w.MoveFileExW(pa.ptr, pb.ptr, w.MOVEFILE_REPLACE_EXISTING | w.MOVEFILE_WRITE_THROUGH) == 0) return mapErr(w.lastError());
        return;
    }
    if (std.c.rename(from, to) != 0) return error.Failed;
}

pub fn openDir(path: []const u8) Error!Dir {
    if (win) {
        var pat: [1100]u8 = undefined;
        const s = std.fmt.bufPrint(&pat, "{s}\\*", .{path}) catch return error.Failed;
        var buf: w.Wide = undefined;
        const p = w.wide(s, &buf) orelse return error.Failed;
        var d: Dir = .{ .h = undefined, .data = undefined, .first = true };
        d.h = w.valid(w.FindFirstFileW(p.ptr, &d.data)) orelse return mapErr(w.lastError());
        return d;
    }
    var z: [1100]u8 = undefined;
    const p = std.fmt.bufPrintSentinel(&z, "{s}", .{path}, 0) catch return error.Failed;
    const d = std.c.opendir(p) orelse return if (std.c.errno(-1) == .NOENT) error.Missing else error.Failed;
    return .{ .d = d };
}

/// The next entry, `.` and `..` included; its name is valid until the next call.
pub fn next(d: *Dir) ?Entry {
    if (win) {
        if (d.first) {
            d.first = false;
        } else if (w.FindNextFileW(d.h, &d.data) == 0) return null;
        const utf16 = std.mem.sliceTo(&d.data.name, 0);
        const n = std.unicode.utf16LeToUtf8(&d.name, utf16) catch return null;
        const kind: Kind = if (d.data.attributes & w.FILE_ATTRIBUTE_REPARSE_POINT != 0) .other else if (d.data.attributes & w.FILE_ATTRIBUTE_DIRECTORY != 0) .dir else .file;
        return .{ .name = d.name[0..n], .kind = kind };
    }
    const ent = std.c.readdir(@ptrCast(@alignCast(d.d))) orelse return null;
    const kind: Kind = if (ent.type == std.c.DT.REG) .file else if (ent.type == std.c.DT.DIR) .dir else .other;
    return .{ .name = std.mem.sliceTo(&ent.name, 0), .kind = kind };
}

pub fn closeDir(d: *Dir) void {
    if (win) _ = w.FindClose(d.h) else _ = std.c.closedir(@ptrCast(@alignCast(d.d)));
}

/// Replaces `dest` with `body` through `dest.part`; `cut` stops the write after that many bytes, as a crash would.
pub fn publish(dest: [:0]const u8, body: []const u8, cut: ?usize) Error!void {
    var part_buf: [1100]u8 = undefined;
    const part = std.fmt.bufPrintSentinel(&part_buf, "{s}.part", .{dest}, 0) catch return error.Failed;
    {
        const fd = try open(part, .{ .access = .write, .create = true, .trunc = true });
        defer close(fd);
        errdefer unlink(part) catch {};
        const n = @min(cut orelse body.len, body.len);
        try write(fd, body[0..n]);
        if (cut != null) return error.Failed;
        if (!sync(fd)) return error.Failed;
    }
    try rename(part, dest);
}

test "learned files and directories are owner-only, where POSIX uses modes 0600 and 0700" {
    if (!win) return error.SkipZigTest; // POSIX checks the mode bits themselves; this test reads a Windows DACL
    var root_buf: [600]u8 = undefined;
    const root = try tempRoot(&root_buf, "acl");
    try mkdir(root);
    defer rmdir(root) catch {};
    var file_buf: [700]u8 = undefined;
    const dest = try std.fmt.bufPrintSentinel(&file_buf, "{s}/state", .{root}, 0);
    defer unlink(dest) catch {};
    try publish(dest, "owned", null);
    var wide_dir: w.Wide = undefined;
    var wide_file: w.Wide = undefined;
    try std.testing.expectEqual(@as(?bool, false), w.othersCanRead(try winPath(&wide_dir, root)));
    try std.testing.expectEqual(@as(?bool, false), w.othersCanRead(try winPath(&wide_file, dest)));
}

fn winPath(buf: *w.Wide, path: []const u8) ![:0]const u16 {
    return w.wide(path, buf) orelse error.Failed;
}

fn readWhole(path: [:0]const u8, dest: []u8) Error![]u8 {
    const fd = try open(path, .{});
    defer close(fd);
    const n = try read(fd, dest);
    return dest[0..n];
}

test "a publish that fails mid-write leaves the previous file whole" {
    var root_buf: [600]u8 = undefined;
    const root = try tempRoot(&root_buf, "publish");
    try mkdir(root);
    defer rmdir(root) catch {};
    var file_buf: [700]u8 = undefined;
    const dest = try std.fmt.bufPrintSentinel(&file_buf, "{s}/state", .{root}, 0);
    defer unlink(dest) catch {};
    try publish(dest, "old state", null);
    try std.testing.expectError(error.Failed, publish(dest, "new state, longer", 4));
    var got: [64]u8 = undefined;
    try std.testing.expectEqualStrings("old state", try readWhole(dest, &got));
}

/// %LOCALAPPDATA% as UTF-8 in `buf`; null when it is unset.
pub fn localAppData(buf: []u8) ?[]const u8 {
    var wbuf: [1024]u16 = undefined;
    const n = w.GetEnvironmentVariableW(std.unicode.utf8ToUtf16LeStringLiteral("LOCALAPPDATA"), &wbuf, wbuf.len);
    if (n == 0 or n >= wbuf.len) return null;
    const len = std.unicode.utf16LeToUtf8(buf, wbuf[0..n]) catch return null;
    return buf[0..len];
}
