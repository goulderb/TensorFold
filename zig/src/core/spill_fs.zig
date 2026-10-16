//! The spill tier's files on each OS (spill.zig). POSIX works relative to a directory descriptor opened without
//! following a link and checked private (this user's or root's, not group or world writable). Windows works on the
//! checked directory's path: a directory is private when no one but its owner, SYSTEM and Administrators may change
//! it, the directories the tier makes get that DACL (lanes/win_fs.zig's), and no link is followed (a reparse point is
//! refused). Every Windows handle shares delete, so a file is unlinked or replaced while a reader holds it, as on
//! POSIX. NTFS journals a rename, so a directory needs no flush after one (MoveFileExW with write-through).
//! Local Win32 declarations: lanes/win_fs.zig belongs to the lanes module, and a file can be in only one.
const std = @import("std");
const builtin = @import("builtin");

pub const windows = builtin.os.tag == .windows;

/// An open file.
pub const File = if (windows) *anyopaque else c_int;
/// A directory the tier works in: its checked descriptor (POSIX) or its checked path (Windows, borrowed).
pub const Dir = if (windows) [:0]const u8 else c_int;

// -- POSIX ---------------------------------------------------------------------------------------------------------

const posix = struct {
    fn createExcl(d: c_int, name: [:0]const u8) ?c_int {
        const fd = std.c.openat(d, name, .{ .ACCMODE = .WRONLY, .CREAT = true, .EXCL = true, .NOFOLLOW = true, .CLOEXEC = true }, @as(std.c.mode_t, 0o600));
        return if (fd < 0) null else fd;
    }

    fn createTrunc(d: c_int, name: [:0]const u8) ?c_int {
        const fd = std.c.openat(d, name, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true, .NOFOLLOW = true, .CLOEXEC = true }, @as(std.c.mode_t, 0o600));
        return if (fd < 0) null else fd;
    }

    fn openRead(d: c_int, name: [:0]const u8) ?c_int {
        const fd = std.c.openat(d, name, .{ .ACCMODE = .RDONLY, .NOFOLLOW = true, .CLOEXEC = true }, @as(std.c.mode_t, 0));
        return if (fd < 0) null else fd;
    }

    fn openPath(path: [:0]const u8) ?c_int {
        const fd = std.c.open(path, .{ .ACCMODE = .RDONLY }, @as(std.c.mode_t, 0));
        return if (fd < 0) null else fd;
    }

    /// Directory `name` in directory `at`, opened (not through a link) when only this process's user or root can
    /// change it; else another user could put a link where the tier writes. The descriptor is the caller's.
    fn openPrivate(at: c_int, name: [:0]const u8) !c_int {
        const fd = std.c.openat(at, name, .{ .ACCMODE = .RDONLY, .DIRECTORY = true, .NOFOLLOW = true, .CLOEXEC = true }, @as(std.c.mode_t, 0));
        if (fd < 0) return error.SpillDirUnsafe;
        errdefer _ = std.c.close(fd);
        const st = ownerOf(fd) orelse return error.SpillDir;
        if ((st.uid != std.c.geteuid() and st.uid != 0) or st.mode & 0o022 != 0) return error.SpillDirUnsafe;
        return fd;
    }

    fn ownerOf(fd: c_int) ?struct { uid: u32, mode: u32 } {
        if (builtin.os.tag == .linux) {
            var sx: std.os.linux.Statx = undefined;
            if (std.c.statx(fd, "", 0x1000, .{ .MODE = true, .UID = true }, &sx) != 0) return null; // AT_EMPTY_PATH
            return .{ .uid = sx.uid, .mode = sx.mode };
        }
        var st: std.c.Stat = undefined;
        if (std.c.fstat(fd, &st) != 0) return null;
        return .{ .uid = st.uid, .mode = @intCast(st.mode) };
    }

    const Statvfs = extern struct {
        bsize: c_ulong,
        frsize: c_ulong,
        blocks: u64,
        bfree: u64,
        bavail: u64,
        files: u64,
        ffree: u64,
        favail: u64,
        fsid: c_ulong,
        flag: c_ulong,
        namemax: c_ulong,
        spare: [6]c_int,
    };
    extern "c" fn statvfs(path: [*:0]const u8, buf: *Statvfs) c_int;
};

// -- Windows -------------------------------------------------------------------------------------------------------

const win = struct {
    const HANDLE = *anyopaque;
    const Wide = [1024:0]u16;
    const GENERIC_READ: u32 = 0x80000000;
    const GENERIC_WRITE: u32 = 0x40000000;
    const FILE_SHARE_ALL: u32 = 0x7; // read, write, delete
    const CREATE_NEW: u32 = 1;
    const CREATE_ALWAYS: u32 = 2;
    const OPEN_EXISTING: u32 = 3;
    const FILE_BEGIN: u32 = 0;
    const FILE_FLAG_BACKUP_SEMANTICS: u32 = 0x02000000;
    const FILE_FLAG_OPEN_REPARSE_POINT: u32 = 0x00200000;
    const FILE_ATTRIBUTE_REPARSE_POINT: u32 = 0x400;
    const FILE_ATTRIBUTE_DIRECTORY: u32 = 0x10;
    const MOVEFILE_REPLACE_EXISTING: u32 = 0x1;
    const MOVEFILE_WRITE_THROUGH: u32 = 0x8;
    const ERROR_ALREADY_EXISTS: u32 = 183;
    const SE_FILE_OBJECT: u32 = 1;
    const OWNER_SECURITY_INFORMATION: u32 = 0x1;
    const DACL_SECURITY_INFORMATION: u32 = 0x4;
    const PROTECTED_DACL_SECURITY_INFORMATION: u32 = 0x80000000;
    const ACCESS_ALLOWED_ACE_TYPE: u8 = 0;
    const FULL_CONTROL: u32 = 0x1F01FF;
    // write data / add file, append / add subdirectory, delete child, delete, write DAC, write owner, generic
    // write, generic all, maximum allowed: any of them lets the holder put a link or a file where the tier writes
    const CHANGE_BITS: u32 = 0x2 | 0x4 | 0x40 | 0x10000 | 0x40000 | 0x80000 | 0x40000000 | 0x10000000 | 0x02000000;
    const WinLocalSystemSid: c_int = 22;
    const WinBuiltinAdministratorsSid: c_int = 26;
    const AclSizeInformation: u32 = 2;

    const Overlapped = extern struct { internal: usize = 0, internal_high: usize = 0, offset: u32, offset_high: u32, event: ?HANDLE = null };
    const FindData = extern struct {
        attributes: u32,
        created: [2]u32,
        accessed: [2]u32,
        written: [2]u32,
        size_high: u32,
        size_low: u32,
        reserved0: u32,
        reserved1: u32,
        name: [260]u16,
        alt: [14]u16,
    };
    const ByHandleInfo = extern struct {
        attributes: u32,
        creation: [2]u32,
        access: [2]u32,
        write: [2]u32,
        volume_serial: u32,
        size_high: u32,
        size_low: u32,
        links: u32,
        index_high: u32,
        index_low: u32,
    };
    const AclSize = extern struct { count: u32, bytes_in_use: u32, bytes_free: u32 };

    extern "kernel32" fn CreateFileW(name: [*:0]const u16, access: u32, share: u32, security: ?*anyopaque, disposition: u32, flags: u32, template: ?*anyopaque) callconv(.winapi) ?HANDLE;
    extern "kernel32" fn ReadFile(h: HANDLE, buf: [*]u8, n: u32, read: *u32, overlapped: ?*Overlapped) callconv(.winapi) i32;
    extern "kernel32" fn WriteFile(h: HANDLE, buf: [*]const u8, n: u32, written: *u32, overlapped: ?*Overlapped) callconv(.winapi) i32;
    extern "kernel32" fn CloseHandle(h: HANDLE) callconv(.winapi) i32;
    extern "kernel32" fn FlushFileBuffers(h: HANDLE) callconv(.winapi) i32;
    extern "kernel32" fn GetFileSizeEx(h: HANDLE, size: *i64) callconv(.winapi) i32;
    extern "kernel32" fn SetFilePointerEx(h: HANDLE, dist: i64, new_pos: ?*i64, method: u32) callconv(.winapi) i32;
    extern "kernel32" fn SetEndOfFile(h: HANDLE) callconv(.winapi) i32;
    extern "kernel32" fn GetFileInformationByHandle(h: HANDLE, info: *ByHandleInfo) callconv(.winapi) i32;
    extern "kernel32" fn CreateDirectoryW(name: [*:0]const u16, security: ?*anyopaque) callconv(.winapi) i32;
    extern "kernel32" fn RemoveDirectoryW(name: [*:0]const u16) callconv(.winapi) i32;
    extern "kernel32" fn DeleteFileW(name: [*:0]const u16) callconv(.winapi) i32;
    extern "kernel32" fn MoveFileExW(from: [*:0]const u16, to: [*:0]const u16, flags: u32) callconv(.winapi) i32;
    extern "kernel32" fn FindFirstFileW(pattern: [*:0]const u16, data: *FindData) callconv(.winapi) ?HANDLE;
    extern "kernel32" fn FindNextFileW(h: HANDLE, data: *FindData) callconv(.winapi) i32;
    extern "kernel32" fn FindClose(h: HANDLE) callconv(.winapi) i32;
    extern "kernel32" fn GetLastError() callconv(.winapi) u32;
    extern "kernel32" fn GetDiskFreeSpaceExW(dir: [*:0]const u16, avail: *u64, total: ?*u64, free: ?*u64) callconv(.winapi) i32;
    extern "kernel32" fn LocalFree(mem: ?*anyopaque) callconv(.winapi) ?*anyopaque;
    extern "advapi32" fn GetSecurityInfo(h: HANDLE, object: u32, info: u32, owner: *?*anyopaque, group: ?*?*anyopaque, dacl: *?*anyopaque, sacl: ?*?*anyopaque, sd: *?*anyopaque) callconv(.winapi) u32;
    extern "advapi32" fn SetNamedSecurityInfoW(name: [*:0]const u16, object: u32, info: u32, owner: ?*anyopaque, group: ?*anyopaque, dacl: ?*anyopaque, sacl: ?*anyopaque) callconv(.winapi) u32;
    extern "advapi32" fn GetAclInformation(acl: *anyopaque, info: *AclSize, len: u32, class: u32) callconv(.winapi) i32;
    extern "advapi32" fn GetAce(acl: *anyopaque, index: u32, ace: *?*anyopaque) callconv(.winapi) i32;
    extern "advapi32" fn EqualSid(a: *anyopaque, b: *anyopaque) callconv(.winapi) i32;
    extern "advapi32" fn IsWellKnownSid(sid: *anyopaque, kind: c_int) callconv(.winapi) i32;
    extern "advapi32" fn InitializeAcl(acl: *anyopaque, len: u32, revision: u32) callconv(.winapi) i32;
    extern "advapi32" fn AddAccessAllowedAce(acl: *anyopaque, revision: u32, mask: u32, sid: *anyopaque) callconv(.winapi) i32;
    extern "advapi32" fn ConvertStringSidToSidW(text: [*:0]const u16, sid: *?*anyopaque) callconv(.winapi) i32;

    fn wide(path: []const u8, buf: *Wide) ?[:0]const u16 {
        const n = std.unicode.utf8ToUtf16Le(buf[0..1023], path) catch return null;
        buf[n] = 0;
        return buf[0..n :0];
    }

    /// `dir`/`name` as NUL-terminated UTF-16; null when it does not fit.
    fn join(d: []const u8, name: []const u8, buf: *Wide) ?[:0]const u16 {
        var p: [1600]u8 = undefined;
        const path = std.fmt.bufPrint(&p, "{s}/{s}", .{ d, name }) catch return null;
        return wide(path, buf);
    }

    fn valid(h: ?HANDLE) ?HANDLE {
        const x = h orelse return null;
        return if (@intFromPtr(x) == std.math.maxInt(usize)) null else x;
    }

    fn isLink(h: HANDLE) bool {
        var info: ByHandleInfo = undefined;
        if (GetFileInformationByHandle(h, &info) == 0) return true;
        return info.attributes & FILE_ATTRIBUTE_REPARSE_POINT != 0;
    }

    /// A file at `path` (no link followed; a link found is closed and refused).
    fn open(path: [:0]const u16, access: u32, disposition: u32) ?HANDLE {
        const h = valid(CreateFileW(path.ptr, access, FILE_SHARE_ALL, null, disposition, FILE_FLAG_OPEN_REPARSE_POINT, null)) orelse return null;
        if (isLink(h)) {
            _ = CloseHandle(h);
            return null;
        }
        return h;
    }

    /// The owner's, SYSTEM's and Administrators' full control only (win_fs.privatize): POSIX 0700 for a directory.
    fn privatize(path: [:0]const u16) bool {
        const h = valid(CreateFileW(path.ptr, GENERIC_READ, FILE_SHARE_ALL, null, OPEN_EXISTING, FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OPEN_REPARSE_POINT, null)) orelse return false;
        defer _ = CloseHandle(h);
        if (isLink(h)) return false;
        var owner: ?*anyopaque = null;
        var sd: ?*anyopaque = null;
        var dacl: ?*anyopaque = null;
        if (GetSecurityInfo(h, SE_FILE_OBJECT, OWNER_SECURITY_INFORMATION, &owner, null, &dacl, null, &sd) != 0) return false;
        defer _ = LocalFree(sd);
        const me = owner orelse return false;
        var system: ?*anyopaque = null;
        var admins: ?*anyopaque = null;
        if (ConvertStringSidToSidW(std.unicode.utf8ToUtf16LeStringLiteral("S-1-5-18"), &system) == 0) return false;
        defer _ = LocalFree(system);
        if (ConvertStringSidToSidW(std.unicode.utf8ToUtf16LeStringLiteral("S-1-5-32-544"), &admins) == 0) return false;
        defer _ = LocalFree(admins);
        var buf: [512]u8 align(8) = undefined;
        if (InitializeAcl(&buf, buf.len, 2) == 0) return false;
        // object and container inherit: the files and directories the tier makes below hold the same three
        if (AddAccessAllowedAceEx(&buf, 2, 0x3, FULL_CONTROL, me) == 0) return false;
        if (AddAccessAllowedAceEx(&buf, 2, 0x3, FULL_CONTROL, system.?) == 0) return false;
        if (AddAccessAllowedAceEx(&buf, 2, 0x3, FULL_CONTROL, admins.?) == 0) return false;
        return SetNamedSecurityInfoW(path.ptr, SE_FILE_OBJECT, PROTECTED_DACL_SECURITY_INFORMATION | DACL_SECURITY_INFORMATION, null, null, &buf, null) == 0;
    }
    extern "advapi32" fn AddAccessAllowedAceEx(acl: *anyopaque, revision: u32, flags: u32, mask: u32, sid: *anyopaque) callconv(.winapi) i32;

    /// Whether anyone but the directory's owner, SYSTEM and Administrators may change it (null: unreadable security).
    fn othersCanChange(h: HANDLE) ?bool {
        var owner: ?*anyopaque = null;
        var dacl: ?*anyopaque = null;
        var sd: ?*anyopaque = null;
        if (GetSecurityInfo(h, SE_FILE_OBJECT, OWNER_SECURITY_INFORMATION | DACL_SECURITY_INFORMATION, &owner, null, &dacl, null, &sd) != 0) return null;
        defer _ = LocalFree(sd);
        const acl = dacl orelse return true; // a null DACL grants everyone everything
        var acl_size: AclSize = undefined;
        if (GetAclInformation(acl, &acl_size, @sizeOf(AclSize), AclSizeInformation) == 0) return null;
        var i: u32 = 0;
        while (i < acl_size.count) : (i += 1) {
            var ace: ?*anyopaque = null;
            if (GetAce(acl, i, &ace) == 0) return null;
            const bytes: [*]u8 = @ptrCast(ace.?);
            if (bytes[0] != ACCESS_ALLOWED_ACE_TYPE) continue;
            if (bytes[1] & 0x8 != 0) continue; // inherit-only: it governs the children, not this directory
            if (std.mem.readInt(u32, bytes[4..8], .little) & CHANGE_BITS == 0) continue;
            const sid: *anyopaque = @ptrCast(bytes + 8);
            if (owner) |o| if (EqualSid(sid, o) != 0) continue;
            if (IsWellKnownSid(sid, WinLocalSystemSid) != 0) continue;
            if (IsWellKnownSid(sid, WinBuiltinAdministratorsSid) != 0) continue;
            return true;
        }
        return false;
    }
};

// -- the interface -------------------------------------------------------------------------------------------------

/// Whether `path` is absolute here: `/...` on POSIX; a drive (`C:/` or `C:\`) on Windows.
pub fn absolute(path: []const u8) bool {
    if (!windows) return path.len > 0 and path[0] == '/';
    return path.len >= 3 and std.ascii.isAlphabetic(path[0]) and path[1] == ':' and (path[2] == '/' or path[2] == '\\');
}

/// A new file `name` in `d`, made only when nothing is there (a link included); null when it cannot be made.
pub fn createExcl(d: Dir, name: [:0]const u8) ?File {
    if (!windows) return posix.createExcl(d, name);
    var b: win.Wide = undefined;
    return win.valid(win.CreateFileW((win.join(d, name, &b) orelse return null).ptr, win.GENERIC_READ | win.GENERIC_WRITE, win.FILE_SHARE_ALL, null, win.CREATE_NEW, win.FILE_FLAG_OPEN_REPARSE_POINT, null));
}

/// File `name` in `d` made or emptied (not through a link).
pub fn createTrunc(d: Dir, name: [:0]const u8) ?File {
    if (!windows) return posix.createTrunc(d, name);
    var b: win.Wide = undefined;
    return win.open(win.join(d, name, &b) orelse return null, win.GENERIC_READ | win.GENERIC_WRITE, win.CREATE_ALWAYS);
}

/// File `name` in `d` for reading (not through a link).
pub fn openRead(d: Dir, name: [:0]const u8) ?File {
    if (!windows) return posix.openRead(d, name);
    var b: win.Wide = undefined;
    return win.open(win.join(d, name, &b) orelse return null, win.GENERIC_READ, win.OPEN_EXISTING);
}

/// The file at `path` for reading.
pub fn openPath(path: [:0]const u8) ?File {
    if (!windows) return posix.openPath(path);
    var b: win.Wide = undefined;
    return win.open(win.wide(path, &b) orelse return null, win.GENERIC_READ, win.OPEN_EXISTING);
}

pub fn close(f: File) void {
    if (windows) _ = win.CloseHandle(f) else _ = std.c.close(f);
}

pub fn unlink(d: Dir, name: [:0]const u8) bool {
    if (!windows) return std.c.unlinkat(d, name, 0) == 0;
    var b: win.Wide = undefined;
    return win.DeleteFileW((win.join(d, name, &b) orelse return false).ptr) != 0;
}

/// `from` renamed `to` in `d`, replacing it.
pub fn rename(d: Dir, from: [:0]const u8, to: [:0]const u8) bool {
    if (!windows) return std.c.renameat(d, from, d, to) == 0;
    var a: win.Wide = undefined;
    var b: win.Wide = undefined;
    const x = win.join(d, from, &a) orelse return false;
    const y = win.join(d, to, &b) orelse return false;
    return win.MoveFileExW(x.ptr, y.ptr, win.MOVEFILE_REPLACE_EXISTING | win.MOVEFILE_WRITE_THROUGH) != 0;
}

/// The file's bytes on the device.
pub fn sync(f: File) bool {
    return if (windows) win.FlushFileBuffers(f) != 0 else std.c.fsync(f) == 0;
}

/// A rename in `d` made durable: fsync on POSIX; NTFS journals it (the rename was write-through).
pub fn syncDir(d: Dir) void {
    if (!windows) _ = std.c.fsync(d);
}

/// The file cut or grown to `size` bytes.
pub fn truncate(f: File, bytes: u64) bool {
    if (!windows) return std.c.ftruncate(f, @intCast(bytes)) == 0;
    if (win.SetFilePointerEx(f, @intCast(bytes), null, win.FILE_BEGIN) == 0) return false;
    return win.SetEndOfFile(f) != 0;
}

pub fn pwriteAll(f: File, bytes: []const u8, at: u64) !void {
    var done: usize = 0;
    while (done < bytes.len) {
        if (windows) {
            const want: u32 = @intCast(@min(bytes.len - done, 1 << 30));
            const off = at + done;
            var ov: win.Overlapped = .{ .offset = @truncate(off), .offset_high = @truncate(off >> 32) };
            var n: u32 = 0;
            if (win.WriteFile(f, bytes.ptr + done, want, &n, &ov) == 0 or n == 0) return error.SpillWrite;
            done += n;
        } else {
            const n = std.c.pwrite(f, bytes.ptr + done, bytes.len - done, @intCast(at + done));
            if (n <= 0) return error.SpillWrite;
            done += @intCast(n);
        }
    }
}

pub fn preadAll(f: File, dest: []u8, at: u64) bool {
    var done: usize = 0;
    while (done < dest.len) {
        if (windows) {
            const want: u32 = @intCast(@min(dest.len - done, 1 << 30));
            const off = at + done;
            var ov: win.Overlapped = .{ .offset = @truncate(off), .offset_high = @truncate(off >> 32) };
            var n: u32 = 0;
            if (win.ReadFile(f, dest.ptr + done, want, &n, &ov) == 0 or n == 0) return false;
            done += n;
        } else {
            const n = std.c.pread(f, dest.ptr + done, dest.len - done, @intCast(at + done));
            if (n <= 0) return false;
            done += @intCast(n);
        }
    }
    return true;
}

pub fn size(f: File) u64 {
    if (windows) {
        var n: i64 = 0;
        return if (win.GetFileSizeEx(f, &n) != 0 and n > 0) @intCast(n) else 0;
    }
    const end = std.c.lseek(f, 0, std.c.SEEK.END);
    return if (end > 0) @intCast(end) else 0;
}

/// `dir` and its parents; the ones made here private (the files hold prompt token ids).
pub fn makePath(gpa: std.mem.Allocator, dir: [:0]const u8) !void {
    const tmp = try gpa.dupeSentinel(u8, dir, 0);
    defer gpa.free(tmp);
    const first: usize = if (windows and absolute(dir)) 3 else 1; // past `C:/` or `/`
    var i: usize = first;
    while (i <= tmp.len) : (i += 1) {
        const end = i == tmp.len;
        if (!end and tmp[i] != '/' and tmp[i] != '\\') continue;
        if (end) {
            if (!mkdir(tmp)) return error.SpillDir;
            continue;
        }
        if (i == 0) continue;
        const ch = tmp[i];
        tmp[i] = 0;
        defer tmp[i] = ch;
        _ = mkdir(tmp[0..i :0]);
    }
}

/// One directory made private, or there already; false when it is neither.
fn mkdir(path: [:0]const u8) bool {
    if (!windows) {
        const rc = std.c.mkdir(path.ptr, 0o700);
        return rc == 0 or std.c.errno(rc) == .EXIST;
    }
    var b: win.Wide = undefined;
    const w = win.wide(path, &b) orelse return false;
    if (win.CreateDirectoryW(w.ptr, null) != 0) return win.privatize(w);
    return win.GetLastError() == win.ERROR_ALREADY_EXISTS;
}

pub fn rmdir(path: [:0]const u8) void {
    if (!windows) {
        _ = std.c.rmdir(path);
        return;
    }
    var b: win.Wide = undefined;
    _ = win.RemoveDirectoryW((win.wide(path, &b) orelse return).ptr);
}

pub fn unlinkPath(path: [:0]const u8) bool {
    if (!windows) return std.c.unlink(path) == 0;
    var b: win.Wide = undefined;
    return win.DeleteFileW((win.wide(path, &b) orelse return false).ptr) != 0;
}

/// Directory `name` under the checked directory `at` (whose path is `path`'s parent), checked private: not a link,
/// changeable by no one but its owner and the system (POSIX: this user or root, not group or world writable).
/// `at` null: `path` is the root. POSIX returns a new descriptor (the caller's); Windows returns `path`.
pub fn openPrivate(at: ?Dir, name: [:0]const u8, path: [:0]const u8) !Dir {
    if (!windows) return posix.openPrivate(at orelse std.c.AT.FDCWD, if (at == null) path else name);
    var b: win.Wide = undefined;
    const w = win.wide(path, &b) orelse return error.SpillDir;
    const h = win.valid(win.CreateFileW(w.ptr, win.GENERIC_READ, win.FILE_SHARE_ALL, null, win.OPEN_EXISTING, win.FILE_FLAG_BACKUP_SEMANTICS | win.FILE_FLAG_OPEN_REPARSE_POINT, null)) orelse return error.SpillDirUnsafe;
    defer _ = win.CloseHandle(h);
    if (win.isLink(h)) return error.SpillDirUnsafe;
    if (win.othersCanChange(h) orelse return error.SpillDir) return error.SpillDirUnsafe;
    return path;
}

pub fn closeDir(d: Dir) void {
    if (!windows) _ = std.c.close(d);
}

/// Bytes an unprivileged writer may still use on `dir`'s file system (no limit known: the most).
pub fn diskFree(dir: [:0]const u8) u64 {
    if (windows) {
        var b: win.Wide = undefined;
        const w = win.wide(dir, &b) orelse return 0;
        var avail: u64 = 0;
        return if (win.GetDiskFreeSpaceExW(w.ptr, &avail, null, null) != 0) avail else 0;
    }
    if (builtin.os.tag != .linux) return std.math.maxInt(u64);
    var st: posix.Statvfs = undefined;
    if (posix.statvfs(dir.ptr, &st) != 0) return 0;
    return st.bavail * st.frsize;
}

/// The names in a directory, files and directories told apart.
pub const Lister = struct {
    state: if (windows) struct { h: ?win.HANDLE, data: win.FindData, first: bool } else *std.c.DIR,
    name: [1040]u8 = undefined,

    pub const Entry = struct { name: []const u8, file: bool };

    pub fn open(dir: [:0]const u8) ?Lister {
        if (!windows) return .{ .state = std.c.opendir(dir) orelse return null };
        var p: [1600]u8 = undefined;
        const pattern = std.fmt.bufPrint(&p, "{s}/*", .{dir}) catch return null;
        var b: win.Wide = undefined;
        var l: Lister = .{ .state = .{ .h = null, .data = undefined, .first = true } };
        l.state.h = win.valid(win.FindFirstFileW((win.wide(pattern, &b) orelse return null).ptr, &l.state.data)) orelse return null;
        return l;
    }

    pub fn next(l: *Lister) ?Entry {
        if (!windows) {
            while (std.c.readdir(l.state)) |ent| {
                const n = std.mem.sliceTo(@as([*:0]const u8, @ptrCast(&ent.name)), 0);
                if (std.mem.eql(u8, n, ".") or std.mem.eql(u8, n, "..")) continue;
                @memcpy(l.name[0..n.len], n);
                return .{ .name = l.name[0..n.len], .file = ent.type == std.c.DT.REG };
            }
            return null;
        }
        while (true) {
            if (!l.state.first and win.FindNextFileW(l.state.h.?, &l.state.data) == 0) return null;
            l.state.first = false;
            const w = std.mem.sliceTo(&l.state.data.name, 0);
            const n = std.unicode.utf16LeToUtf8(&l.name, w) catch continue;
            const name = l.name[0..n];
            if (std.mem.eql(u8, name, ".") or std.mem.eql(u8, name, "..")) continue;
            const attrs = l.state.data.attributes;
            return .{ .name = name, .file = attrs & (win.FILE_ATTRIBUTE_DIRECTORY | win.FILE_ATTRIBUTE_REPARSE_POINT) == 0 };
        }
    }

    pub fn close(l: *Lister) void {
        if (windows) {
            if (l.state.h) |h| _ = win.FindClose(h);
        } else _ = std.c.closedir(l.state);
    }
};

test "absolute paths on this OS" {
    if (windows) {
        try std.testing.expect(absolute("C:/x"));
        try std.testing.expect(absolute("d:\\x"));
        try std.testing.expect(!absolute("/x"));
        try std.testing.expect(!absolute("x"));
    } else {
        try std.testing.expect(absolute("/x"));
        try std.testing.expect(!absolute("C:/x"));
    }
}
