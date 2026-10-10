//! Win32 bindings for the learned-state file layer: handles, directories, atomic replace, free space and owner-only DACLs.
const std = @import("std");

pub const HANDLE = *anyopaque;
pub const Wide = [1024:0]u16;
pub const GENERIC_READ: u32 = 0x80000000;
pub const GENERIC_WRITE: u32 = 0x40000000;
pub const FILE_SHARE_ALL: u32 = 0x7;
pub const OPEN_EXISTING: u32 = 3;
pub const OPEN_ALWAYS: u32 = 4;
pub const CREATE_ALWAYS: u32 = 2;
pub const TRUNCATE_EXISTING: u32 = 5;
pub const FILE_BEGIN: u32 = 0;
pub const FILE_END: u32 = 2;
pub const FILE_FLAG_BACKUP_SEMANTICS: u32 = 0x02000000;
pub const FILE_FLAG_OPEN_REPARSE_POINT: u32 = 0x00200000;
pub const FILE_ATTRIBUTE_REPARSE_POINT: u32 = 0x400;
pub const FILE_ATTRIBUTE_DIRECTORY: u32 = 0x10;
pub const MOVEFILE_REPLACE_EXISTING: u32 = 0x1;
pub const MOVEFILE_WRITE_THROUGH: u32 = 0x8;
pub const ERROR_FILE_NOT_FOUND: u32 = 2;
pub const ERROR_PATH_NOT_FOUND: u32 = 3;
pub const ERROR_FILE_EXISTS: u32 = 80;
pub const ERROR_ALREADY_EXISTS: u32 = 183;
pub const SE_FILE_OBJECT: u32 = 1;
pub const OWNER_SECURITY_INFORMATION: u32 = 0x1;
pub const DACL_SECURITY_INFORMATION: u32 = 0x4;
pub const PROTECTED_DACL_SECURITY_INFORMATION: u32 = 0x80000000;
const ACCESS_ALLOWED_ACE_TYPE: u8 = 0;
// FILE_READ_DATA, GENERIC_READ, GENERIC_ALL, MAXIMUM_ALLOWED: any of them lets the holder read the file
const READ_BITS: u32 = 0x1 | 0x80000000 | 0x10000000 | 0x02000000;
const FULL_CONTROL: u32 = 0x1F01FF;
const WinLocalSystemSid: c_int = 22;
const WinBuiltinAdministratorsSid: c_int = 26;
const AclSizeInformation: u32 = 2;

pub const Overlapped = extern struct { internal: usize, internal_high: usize, offset: u32, offset_high: u32, event: ?HANDLE };
pub const FindData = extern struct {
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

pub extern "kernel32" fn CreateFileW(name: [*:0]const u16, access: u32, share: u32, security: ?*anyopaque, disposition: u32, flags: u32, template: ?*anyopaque) callconv(.winapi) ?HANDLE;
pub extern "kernel32" fn ReadFile(h: HANDLE, buf: [*]u8, n: u32, read: *u32, overlapped: ?*Overlapped) callconv(.winapi) i32;
pub extern "kernel32" fn WriteFile(h: HANDLE, buf: [*]const u8, n: u32, written: *u32, overlapped: ?*Overlapped) callconv(.winapi) i32;
pub extern "kernel32" fn CloseHandle(h: HANDLE) callconv(.winapi) i32;
pub extern "kernel32" fn FlushFileBuffers(h: HANDLE) callconv(.winapi) i32;
pub extern "kernel32" fn GetFileSizeEx(h: HANDLE, size: *i64) callconv(.winapi) i32;
pub extern "kernel32" fn SetFilePointerEx(h: HANDLE, dist: i64, new_pos: ?*i64, method: u32) callconv(.winapi) i32;
pub extern "kernel32" fn SetEndOfFile(h: HANDLE) callconv(.winapi) i32;
pub extern "kernel32" fn GetFileInformationByHandle(h: HANDLE, info: *ByHandleInfo) callconv(.winapi) i32;
pub extern "kernel32" fn CreateDirectoryW(name: [*:0]const u16, security: ?*anyopaque) callconv(.winapi) i32;
pub extern "kernel32" fn RemoveDirectoryW(name: [*:0]const u16) callconv(.winapi) i32;
pub extern "kernel32" fn DeleteFileW(name: [*:0]const u16) callconv(.winapi) i32;
pub extern "kernel32" fn MoveFileExW(from: [*:0]const u16, to: [*:0]const u16, flags: u32) callconv(.winapi) i32;
pub extern "kernel32" fn FindFirstFileW(pattern: [*:0]const u16, data: *FindData) callconv(.winapi) ?HANDLE;
pub extern "kernel32" fn FindNextFileW(h: HANDLE, data: *FindData) callconv(.winapi) i32;
pub extern "kernel32" fn FindClose(h: HANDLE) callconv(.winapi) i32;
pub extern "kernel32" fn GetLastError() callconv(.winapi) u32;
pub extern "kernel32" fn GetCurrentProcessId() callconv(.winapi) u32;
pub extern "kernel32" fn GetTempPathW(n: u32, buf: [*]u16) callconv(.winapi) u32;
pub extern "kernel32" fn GetEnvironmentVariableW(name: [*:0]const u16, buf: [*]u16, n: u32) callconv(.winapi) u32;
pub extern "kernel32" fn GetDiskFreeSpaceExW(dir: [*:0]const u16, avail: *u64, total: ?*u64, free: ?*u64) callconv(.winapi) i32;
pub extern "advapi32" fn GetSecurityInfo(h: HANDLE, object: u32, info: u32, owner: *?*anyopaque, group: ?*?*anyopaque, dacl: *?*anyopaque, sacl: ?*?*anyopaque, sd: *?*anyopaque) callconv(.winapi) u32;
pub extern "advapi32" fn SetNamedSecurityInfoW(name: [*:0]const u16, object: u32, info: u32, owner: ?*anyopaque, group: ?*anyopaque, dacl: ?*anyopaque, sacl: ?*anyopaque) callconv(.winapi) u32;
extern "advapi32" fn GetAclInformation(acl: *anyopaque, info: *AclSize, len: u32, class: u32) callconv(.winapi) i32;
extern "advapi32" fn GetAce(acl: *anyopaque, index: u32, ace: *?*anyopaque) callconv(.winapi) i32;
extern "advapi32" fn EqualSid(a: *anyopaque, b: *anyopaque) callconv(.winapi) i32;
extern "advapi32" fn IsWellKnownSid(sid: *anyopaque, kind: c_int) callconv(.winapi) i32;
extern "advapi32" fn InitializeAcl(acl: *anyopaque, len: u32, revision: u32) callconv(.winapi) i32;
extern "advapi32" fn AddAccessAllowedAce(acl: *anyopaque, revision: u32, mask: u32, sid: *anyopaque) callconv(.winapi) i32;
extern "advapi32" fn ConvertStringSidToSidW(text: [*:0]const u16, sid: *?*anyopaque) callconv(.winapi) i32;
extern "kernel32" fn LocalFree(mem: ?*anyopaque) callconv(.winapi) ?*anyopaque;

/// A UTF-8 path as NUL-terminated UTF-16 in `buf`; null when it does not fit.
pub fn wide(path: []const u8, buf: *Wide) ?[:0]const u16 {
    const n = std.unicode.utf8ToUtf16Le(buf[0..1023], path) catch return null;
    buf[n] = 0;
    return buf[0..n :0];
}

/// A handle, or null for INVALID_HANDLE_VALUE and a failed call.
pub fn valid(h: ?HANDLE) ?HANDLE {
    const x = h orelse return null;
    return if (@intFromPtr(x) == std.math.maxInt(usize)) null else x;
}

pub fn lastError() u32 {
    return GetLastError();
}

/// Whether the handle is a reparse point (a link or junction), which a no-follow open refuses.
pub fn isLink(h: HANDLE) bool {
    var info: ByHandleInfo = undefined;
    if (GetFileInformationByHandle(h, &info) == 0) return true;
    return info.attributes & FILE_ATTRIBUTE_REPARSE_POINT != 0;
}

/// Bytes available to this user on the volume holding `dir`; null when the query fails.
pub fn freeBytes(dir: [:0]const u8) ?u64 {
    var buf: Wide = undefined;
    const p = wide(dir, &buf) orelse return null;
    var avail: u64 = 0;
    if (GetDiskFreeSpaceExW(p.ptr, &avail, null, null) == 0) return null;
    return avail;
}

/// The owner's, SYSTEM's and Administrators' full control only: the Windows form of POSIX 0600 (files) and 0700 (dirs).
pub fn privatize(path: [:0]const u16, dir: bool) bool {
    const flags: u32 = if (dir) FILE_FLAG_BACKUP_SEMANTICS else 0;
    const h = valid(CreateFileW(path.ptr, GENERIC_READ, FILE_SHARE_ALL, null, OPEN_EXISTING, flags, null)) orelse return false;
    defer _ = CloseHandle(h);
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
    if (AddAccessAllowedAce(&buf, 2, FULL_CONTROL, me) == 0) return false;
    if (AddAccessAllowedAce(&buf, 2, FULL_CONTROL, system.?) == 0) return false;
    if (AddAccessAllowedAce(&buf, 2, FULL_CONTROL, admins.?) == 0) return false;
    return SetNamedSecurityInfoW(path.ptr, SE_FILE_OBJECT, PROTECTED_DACL_SECURITY_INFORMATION | DACL_SECURITY_INFORMATION, null, null, &buf, null) == 0;
}

/// False when only the owner, SYSTEM or Administrators can read the file; null when its security cannot be read.
pub fn othersCanRead(path: [:0]const u16) ?bool {
    const h = valid(CreateFileW(path.ptr, GENERIC_READ, FILE_SHARE_ALL, null, OPEN_EXISTING, FILE_FLAG_BACKUP_SEMANTICS, null)) orelse return null;
    defer _ = CloseHandle(h);
    var owner: ?*anyopaque = null;
    var dacl: ?*anyopaque = null;
    var sd: ?*anyopaque = null;
    if (GetSecurityInfo(h, SE_FILE_OBJECT, OWNER_SECURITY_INFORMATION | DACL_SECURITY_INFORMATION, &owner, null, &dacl, null, &sd) != 0) return null;
    defer _ = LocalFree(sd);
    const acl = dacl orelse return true;
    var size: AclSize = undefined;
    if (GetAclInformation(acl, &size, @sizeOf(AclSize), AclSizeInformation) == 0) return null;
    var i: u32 = 0;
    while (i < size.count) : (i += 1) {
        var ace: ?*anyopaque = null;
        if (GetAce(acl, i, &ace) == 0) return null;
        const bytes: [*]u8 = @ptrCast(ace.?);
        if (bytes[0] != ACCESS_ALLOWED_ACE_TYPE) continue;
        if (std.mem.readInt(u32, bytes[4..8], .little) & READ_BITS == 0) continue;
        const sid: *anyopaque = @ptrCast(bytes + 8);
        if (owner) |o| if (EqualSid(sid, o) != 0) continue;
        if (IsWellKnownSid(sid, WinLocalSystemSid) != 0) continue;
        if (IsWellKnownSid(sid, WinBuiltinAdministratorsSid) != 0) continue;
        return true;
    }
    return false;
}
