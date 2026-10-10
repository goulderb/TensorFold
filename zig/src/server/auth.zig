//! API keys held as SHA-256 digests with labels; a restricted key file reread on change or SIGHUP (on Windows, the
//! process's reload event).
const std = @import("std");
const posix = std.posix;
const builtin = @import("builtin");
const log = @import("log.zig");
const win32 = @import("win32.zig");
const Allocator = std.mem.Allocator;
const Sha256 = std.crypto.hash.sha2.Sha256;

const Key = struct { digest: [32]u8, label: []const u8 };

pub var reload_requested: std.atomic.Value(bool) = .init(false);
var reload_event: ?win32.HANDLE = null;

fn onHangup(_: posix.SIG) callconv(.c) void {
    reload_requested.store(true, .release);
}

/// Hears SIGHUP as "reread the key file". On Windows the reload event does the same job.
pub fn installHangup() void {
    if (comptime builtin.os.tag == .windows) {
        reload_event = win32.createReloadEvent(win32.GetCurrentProcessId());
        if (reload_event == null) log.line("reload event refused: the key file reloads on change only", .{});
        return;
    }
    const action: posix.Sigaction = .{ .handler = .{ .handler = onHangup }, .mask = posix.sigemptyset(), .flags = 0 };
    posix.sigaction(.HUP, &action, null);
}

/// True when a reload was asked since the last call: SIGHUP's flag, or a set reload event (which it resets).
fn takeReloadRequest() bool {
    if (comptime builtin.os.tag == .windows) {
        if (reload_event) |h| if (win32.takeReloadEvent(h)) reload_requested.store(true, .release);
    }
    return reload_requested.swap(false, .acq_rel);
}

pub const Store = struct {
    gpa: Allocator,
    static: std.ArrayList(Key) = .empty,
    path: ?[]const u8,
    metrics_open: bool,
    mutex: std.Io.Mutex = .init,
    file: []Key = &.{},
    file_arena: ?std.heap.ArenaAllocator = null,
    valid: bool = true,
    stamp: [5]i128 = .{ 0, 0, 0, 0, 0 },
    checked_ns: ?i96 = null,

    /// ``keys`` from --api-key, ``environment`` from TENSORFOLD_API_KEY; errors carry Python's message.
    pub fn init(gpa: Allocator, keys: []const []const u8, environment: []const u8, path: ?[]const u8, metrics_open: bool, problem: *[]const u8) !Store {
        var store: Store = .{ .gpa = gpa, .path = path, .metrics_open = metrics_open };
        errdefer store.deinit();
        for (keys, 1..) |key, i| {
            const d = digest(key) orelse return fail(problem, "API keys must be nonempty text without whitespace");
            try store.static.ensureUnusedCapacity(gpa, 1);
            store.static.appendAssumeCapacity(.{ .digest = d, .label = try std.fmt.allocPrint(gpa, "cli-{d}", .{i}) });
        }
        var it = std.mem.splitScalar(u8, environment, ',');
        var i: usize = 0;
        while (it.next()) |raw| {
            i += 1;
            const key = std.mem.trim(u8, raw, " \t\r\n\x0b\x0c");
            if (key.len == 0) continue;
            const d = digest(key) orelse return fail(problem, "API keys must be nonempty text without whitespace");
            try store.static.ensureUnusedCapacity(gpa, 1);
            store.static.appendAssumeCapacity(.{ .digest = d, .label = try std.fmt.allocPrint(gpa, "env-{d}", .{i}) });
        }
        if (path != null) {
            store.readFile(problem) catch |e| switch (e) {
                error.Unsafe => {
                    if (problem.len == 0) problem.* = "API key file cannot be opened safely";
                    return error.KeyFile;
                },
                error.Invalid => return error.KeyFile,
                error.OutOfMemory => return error.OutOfMemory,
            };
        }
        return store;
    }

    pub fn deinit(s: *Store) void {
        for (s.static.items) |key| s.gpa.free(key.label);
        s.static.deinit(s.gpa);
        if (s.file_arena) |*arena| arena.deinit();
    }

    fn fail(problem: *[]const u8, message: []const u8) error{KeyFile} {
        problem.* = message;
        return error.KeyFile;
    }

    pub fn enabled(s: *const Store) bool {
        return s.static.items.len > 0 or s.path != null;
    }

    /// Reads the key file, replacing the file's keys; Unsafe for an OS error, Invalid with ``problem`` set.
    fn readFile(s: *Store, problem: *[]const u8) error{ Unsafe, Invalid, OutOfMemory }!void {
        const path = s.path.?;
        const pathz = try s.gpa.dupeSentinel(u8, path, 0);
        defer s.gpa.free(pathz);
        const fd = openKey(pathz) orelse return error.Unsafe;
        defer closeKey(fd);
        const st = meta(fd, null) orelse return error.Unsafe;
        if ((st.mode & S_IFMT) != S_IFREG or st.mode & 0o044 != 0) {
            problem.* = "API key file must be a regular file unreadable by other users; use chmod 600";
            return error.Invalid;
        }
        if (st.size > 1 << 20) {
            problem.* = "API key file exceeds 1 MiB";
            return error.Invalid;
        }
        var arena: std.heap.ArenaAllocator = .init(s.gpa);
        errdefer arena.deinit();
        const a = arena.allocator();
        const text = try a.alloc(u8, (1 << 20) + 1);
        var got: usize = 0;
        while (got < text.len) {
            const n = readKey(fd, text[got..]) catch return error.Unsafe;
            if (n == 0) break;
            got += n;
        }
        if (got > 1 << 20) {
            problem.* = "API key file exceeds 1 MiB";
            return error.Invalid;
        }
        if (!std.unicode.utf8ValidateSlice(text[0..got])) return error.Unsafe;
        var keys: std.ArrayList(Key) = .empty;
        var lines = std.mem.splitAny(u8, text[0..got], "\n\r");
        while (lines.next()) |raw| {
            const line = std.mem.trim(u8, raw, " \t\x0b\x0c\x1c\x1d\x1e\x1f");
            if (line.len == 0 or line[0] == '#') continue;
            var label: []const u8 = undefined;
            var key: []const u8 = line;
            if (std.mem.indexOfScalar(u8, line, ':')) |colon| {
                label = std.mem.trim(u8, line[0..colon], " \t");
                key = std.mem.trim(u8, line[colon + 1 ..], " \t");
                if (!validLabel(label)) {
                    problem.* = "API key labels need 1-64 letters, digits, dots, underscores or hyphens";
                    return error.Invalid;
                }
            } else label = try std.fmt.allocPrint(a, "file-{d}", .{keys.items.len + 1});
            const d = digest(key) orelse {
                problem.* = "API keys must be nonempty text without whitespace";
                return error.Invalid;
            };
            try keys.append(a, .{ .digest = d, .label = try a.dupe(u8, label) });
        }
        if (keys.items.len == 0) {
            problem.* = "API key file contains no keys";
            return error.Invalid;
        }
        if (s.file_arena) |*old| old.deinit();
        s.file_arena = arena;
        s.file = keys.items;
        s.stamp = .{ st.dev, st.ino, st.mtime_ns, st.size, st.mode };
    }

    fn statStamp(s: *Store) ?[5]i128 {
        const pathz = s.gpa.dupeSentinel(u8, s.path.?, 0) catch return null;
        defer s.gpa.free(pathz);
        const st = meta(null, pathz) orelse return null;
        return .{ st.dev, st.ino, st.mtime_ns, st.size, st.mode };
    }

    /// Rereads the key file when SIGHUP asked or, at most once a second, when its stamp changed.
    fn refresh(s: *Store, now_ns: i96) void {
        if (s.path == null) return;
        const forced = takeReloadRequest();
        if (!forced) if (s.checked_ns) |last| if (now_ns - last < std.time.ns_per_s) return;
        s.checked_ns = now_ns;
        var problem: []const u8 = "";
        const stamp = s.statStamp();
        if (stamp == null or forced or !s.valid or !std.mem.eql(i128, &stamp.?, &s.stamp)) {
            const ok = if (stamp == null) false else if (s.readFile(&problem)) |_| true else |_| false;
            if (ok) {
                s.valid = true;
            } else {
                if (s.valid) log.line("API key file reload refused; authenticated routes remain closed", .{});
                if (s.file_arena) |*old| old.deinit();
                s.file_arena = null;
                s.file = &.{};
                s.valid = false;
            }
        }
    }

    /// The label of the key a request carries (Authorization: Bearer, or x-api-key), or null.
    pub fn match(s: *Store, a: Allocator, authorization: ?[]const u8, x_api_key: ?[]const u8) ?[]const u8 {
        const m = log.io();
        s.mutex.lockUncancelable(m);
        defer s.mutex.unlock(m);
        s.refresh(std.Io.Clock.awake.now(m).toNanoseconds());
        var candidates: [2][32]u8 = undefined;
        candidates[0] = emptyDigest();
        if (authorization) |value| {
            if (std.mem.indexOfScalar(u8, value, ' ')) |space| {
                if (std.ascii.eqlIgnoreCase(value[0..space], "bearer")) candidates[0] = headerDigest(a, std.mem.trim(u8, value[space + 1 ..], latin1_space));
            }
        }
        candidates[1] = if (x_api_key) |value| headerDigest(a, value) else emptyDigest();
        var label: ?[]const u8 = null;
        for ([_][]Key{ s.static.items, s.file }) |keys| for (keys) |key| {
            var matched = false;
            for (candidates) |c| matched = std.crypto.timing_safe.eql([32]u8, key.digest, c) or matched;
            if (matched and label == null) label = key.label;
        };
        if (!s.valid) return null;
        return a.dupe(u8, label orelse return null) catch null; // the request's own copy: a reload frees the file's
    }
};

const latin1_space = " \t\r\n\x0b\x0c\x1c\x1d\x1e\x1f\x85\xa0";

/// A header value is latin-1 text whose UTF-8 is hashed, as Python decodes and encodes it.
fn headerDigest(a: Allocator, value: []const u8) [32]u8 {
    const text = @import("http_conn.zig").latin1(a, value) catch return emptyDigest();
    return digest(text) orelse emptyDigest();
}

fn validLabel(label: []const u8) bool {
    if (label.len == 0 or label.len > 64) return false;
    for (label) |ch| if (!(std.ascii.isAlphanumeric(ch) or ch == '_' or ch == '.' or ch == '-')) return false;
    return true;
}

fn emptyDigest() [32]u8 {
    var out: [32]u8 = undefined;
    Sha256.hash("", &out, .{});
    return out;
}

/// SHA-256 of a key, or null for empty text or text with whitespace or control characters.
fn digest(key: []const u8) ?[32]u8 {
    if (key.len == 0) return null;
    var view = std.unicode.Utf8View.init(key) catch return null;
    var it = view.iterator();
    while (it.nextCodepoint()) |cp| if (cp < 32 or isSpace(cp)) return null;
    var out: [32]u8 = undefined;
    Sha256.hash(key, &out, .{});
    return out;
}

fn isSpace(cp: u21) bool {
    return switch (cp) {
        ' ', 0x85, 0xa0, 0x1680, 0x2000...0x200a, 0x2028, 0x2029, 0x202f, 0x205f, 0x3000 => true,
        else => false,
    };
}

/// The paths only a key opens (``auth_http.gated``).
pub fn gated(raw_path: []const u8, metrics_open: bool) bool {
    const path = routePath(raw_path);
    if (metrics_open and (std.mem.eql(u8, path, "/metrics") or std.mem.eql(u8, path, "/v1/metrics"))) return false;
    if (std.mem.eql(u8, path, "/v1") or std.mem.startsWith(u8, path, "/v1/")) return true;
    for ([_][]const u8{ "/metrics", "/dashboard", "/stats", "/memory", "/tokenize", "/detokenize", "/models", "/messages", "/messages/count_tokens", "/chat/completions", "/completions", "/decisions", "/responses" }) |p| if (std.mem.eql(u8, path, p)) return true;
    if (std.mem.startsWith(u8, path, "/responses/")) return true;
    for ([_][]const u8{ "/chat/completions", "/completions", "/decisions", "/models" }) |end| if (std.mem.endsWith(u8, path, end)) return true;
    return false;
}

/// A request path without its query and trailing slashes.
pub fn routePath(raw: []const u8) []const u8 {
    const q = std.mem.indexOfScalar(u8, raw, '?') orelse raw.len;
    return std.mem.trimEnd(u8, raw[0..q], "/");
}

/// Whether ``host`` is a loopback address (a server without keys warns otherwise).
pub fn loopback(host: []const u8) bool {
    if (std.ascii.eqlIgnoreCase(host, "localhost")) return true;
    const ip = std.Io.net.IpAddress.parse(host, 0) catch return false;
    return switch (ip) {
        .ip4 => |v4| v4.bytes[0] == 127,
        .ip6 => |v6| std.mem.eql(u8, &v6.bytes, &[16]u8{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 }),
    };
}

test "gates and digests" {
    try std.testing.expect(gated("/v1/models", false));
    try std.testing.expect(!gated("/health", false));
    try std.testing.expect(gated("/memory?reset_peak=1", true));
    try std.testing.expect(gated("/v1/memory/", true));
    try std.testing.expect(!gated("/metrics/", true));
    try std.testing.expect(gated("/alternative/chat/completions", false));
    try std.testing.expect(digest("has space") == null);
    try std.testing.expect(loopback("127.0.0.1") and loopback("::1") and !loopback("0.0.0.0"));
}

test "auth cleanup after invalid initialization" {
    var problem: []const u8 = "";
    const gpa = std.testing.allocator;
    try std.testing.expectError(error.KeyFile, Store.init(gpa, &.{ "sk-one", "has space" }, "", null, false, &problem));
    try std.testing.expectError(error.KeyFile, Store.init(gpa, &.{"sk-one"}, "sk-two,has space", null, false, &problem));
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const missing = try std.fmt.allocPrint(gpa, ".zig-cache/tmp/{s}/missing", .{tmp.sub_path});
    defer gpa.free(missing);
    try std.testing.expectError(error.KeyFile, Store.init(gpa, &.{"sk-one"}, "sk-two", missing, false, &problem));
}

test "auth cleanup on every initialization allocation failure" {
    const gpa = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "keys", .data = "ops: sk-file\nsk-unlabelled\n" });
    const path = try std.fmt.allocPrintSentinel(gpa, ".zig-cache/tmp/{s}/keys", .{tmp.sub_path}, 0);
    defer gpa.free(path);
    try std.testing.expectEqual(@as(c_int, 0), std.c.fchmodat(posix.AT.FDCWD, path, 0o600, 0));
    const Case = struct {
        fn run(a: Allocator, file: ?[]const u8) !void {
            var problem: []const u8 = "";
            var s = try Store.init(a, &.{ "sk-one", "sk-two", "sk-three", "sk-four" }, "sk-env", file, false, &problem);
            defer s.deinit();
            try std.testing.expectEqual(@as(usize, 5), s.static.items.len);
            try std.testing.expectEqual(@as(usize, if (file == null) 0 else 2), s.file.len);
        }
    };
    try std.testing.checkAllAllocationFailures(gpa, Case.run, .{null});
    try std.testing.checkAllAllocationFailures(gpa, Case.run, .{@as(?[]const u8, path)});
}
const S_IFMT: u32 = 0o170000;
const S_IFREG: u32 = 0o100000;
const S_IFDIR: u32 = 0o040000;

/// A file's type and permission bits, size, identity and mtime: statx on Linux (its libc has no stat), stat elsewhere.
/// On Windows the mode is synthesized: a regular file, 0600, plus 0044 when the DACL lets someone else read it.
const Meta = struct { mode: u32, size: u64, dev: u64, ino: u64, mtime_ns: i128 };

/// The key file, opened without following a link. Windows has no O_NOFOLLOW: a reparse point is refused by `meta`.
fn openKey(pathz: [:0]const u8) ?posix.fd_t {
    if (builtin.os.tag == .windows) return winOpen(pathz);
    return posix.openatZ(posix.AT.FDCWD, pathz, .{ .ACCMODE = .RDONLY, .NOFOLLOW = true, .NONBLOCK = true }, 0) catch null;
}

fn closeKey(fd: posix.fd_t) void {
    if (builtin.os.tag == .windows) return winClose(fd);
    _ = posix.system.close(fd);
}

fn readKey(fd: posix.fd_t, buf: []u8) error{Unsafe}!usize {
    if (builtin.os.tag == .windows) return winRead(fd, buf);
    return posix.read(fd, buf) catch error.Unsafe;
}

/// `fd`, or the file at `path` when `path` is given (opened for the stat and closed again).
fn meta(fd: ?posix.fd_t, path: ?[*:0]const u8) ?Meta {
    if (builtin.os.tag == .windows) return winMeta(fd, path);
    if (builtin.os.tag == .linux) {
        const linux = std.os.linux;
        var sx: linux.Statx = undefined;
        const want: linux.STATX = .{ .TYPE = true, .MODE = true, .INO = true, .SIZE = true, .MTIME = true };
        const rc = if (path) |p| std.c.statx(posix.AT.FDCWD, p, 0, want, &sx) else std.c.statx(fd.?, "", 0x1000, want, &sx); // AT_EMPTY_PATH
        if (rc != 0) return null;
        const dev = (@as(u64, sx.dev_major) << 32) | sx.dev_minor;
        return .{ .mode = sx.mode, .size = sx.size, .dev = dev, .ino = sx.ino, .mtime_ns = @as(i128, sx.mtime.sec) * std.time.ns_per_s + sx.mtime.nsec };
    }
    var st: posix.Stat = undefined;
    const rc = if (path) |p| posix.system.stat(p, &st) else posix.system.fstat(fd.?, &st);
    if (posix.errno(rc) != .SUCCESS) return null;
    const t = st.mtime();
    return .{ .mode = @intCast(st.mode), .size = @intCast(st.size), .dev = @intCast(st.dev), .ino = @intCast(st.ino), .mtime_ns = std.math.lossyCast(i128, t.nsec) + @as(i128, t.sec) * std.time.ns_per_s };
}

test "the key file: refused while others can read it, read at 0600, reread after an edit" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "keys", .data = "ops: sk-one\n" });
    const path = try std.fmt.allocPrintSentinel(a, ".zig-cache/tmp/{s}/keys", .{tmp.sub_path}, 0);
    try std.testing.expectEqual(@as(c_int, 0), std.c.fchmodat(posix.AT.FDCWD, path, 0o644, 0));
    var problem: []const u8 = "";
    try std.testing.expectError(error.KeyFile, Store.init(a, &.{}, "", path, false, &problem));
    try std.testing.expect(std.mem.indexOf(u8, problem, "chmod 600") != null);
    try std.testing.expectEqual(@as(c_int, 0), std.c.fchmodat(posix.AT.FDCWD, path, 0o600, 0));
    var s = try Store.init(std.testing.allocator, &.{"sk-cli"}, "sk-env", path, false, &problem);
    defer s.deinit();
    try std.testing.expectEqualStrings("cli-1", s.match(a, "Bearer sk-cli", null).?);
    try std.testing.expectEqualStrings("env-1", s.match(a, null, "sk-env").?);
    try std.testing.expectEqualStrings("ops", s.match(a, "Bearer sk-one", null).?);
    const seen = meta(-1, path).?;
    try std.testing.expectEqual(@as(u64, 12), seen.size);
    try std.testing.expectEqual(@as(u32, 0o600), seen.mode & 0o777);
    try tmp.dir.writeFile(io, .{ .sub_path = "keys", .data = "ops: sk-one\nci: sk-two\n" });
    try std.testing.expect(!std.mem.eql(i128, &s.stamp, &s.statStamp().?));
    s.checked_ns = null;
    try std.testing.expectEqualStrings("ci", s.match(a, null, "sk-two").?);
}

test "a request's key label outlives a reload of the key file" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "keys", .data = "ops: sk-one\n" });
    const path = try std.fmt.allocPrintSentinel(a, ".zig-cache/tmp/{s}/keys", .{tmp.sub_path}, 0);
    try std.testing.expectEqual(@as(c_int, 0), std.c.fchmodat(posix.AT.FDCWD, path, 0o600, 0));
    var problem: []const u8 = "";
    var s = try Store.init(gpa, &.{}, "", path, false, &problem);
    defer if (s.file_arena) |*keys| keys.deinit();
    const label = s.match(a, "Bearer sk-one", null).?; // a request holds its label until it answers
    try tmp.dir.writeFile(io, .{ .sub_path = "keys", .data = "ops: sk-one\nci: sk-two\n" });
    s.checked_ns = null;
    try std.testing.expectEqualStrings("ci", s.match(a, null, "sk-two").?); // another request rereads the file
    try std.testing.expectEqualStrings("ops", label);
    const kept = s.match(a, null, "sk-two").?;
    try std.testing.expectEqual(@as(c_int, 0), std.c.fchmodat(posix.AT.FDCWD, path, 0o644, 0));
    s.checked_ns = null;
    try std.testing.expect(s.match(a, "Bearer sk-one", null) == null); // a refused reload frees the file's keys too
    try std.testing.expectEqualStrings("ci", kept);
}

// Windows: the key file's handle, its identity, and an access check on its DACL.

const win = struct {
    const GENERIC_READ: u32 = 0x80000000;
    const FILE_SHARE_ALL: u32 = 0x7;
    const OPEN_EXISTING: u32 = 3;
    const FILE_FLAG_OPEN_REPARSE_POINT: u32 = 0x00200000;
    const FILE_ATTRIBUTE_REPARSE_POINT: u32 = 0x400;
    const FILE_ATTRIBUTE_DIRECTORY: u32 = 0x10;
    const SE_FILE_OBJECT: u32 = 1;
    const OWNER_SECURITY_INFORMATION: u32 = 0x1;
    const DACL_SECURITY_INFORMATION: u32 = 0x4;
    const ACCESS_ALLOWED_ACE_TYPE: u8 = 0;
    // FILE_READ_DATA, GENERIC_READ, GENERIC_ALL, MAXIMUM_ALLOWED: any of them lets the holder read the file
    const READ_BITS: u32 = 0x1 | 0x80000000 | 0x10000000 | 0x02000000;
    const WinLocalSystemSid: c_int = 22;
    const WinBuiltinAdministratorsSid: c_int = 26;
    const AclSizeInformation: u32 = 2;
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
};

extern "kernel32" fn CreateFileW(name: [*:0]const u16, access: u32, share: u32, security: ?*anyopaque, disposition: u32, flags: u32, template: ?*anyopaque) callconv(.winapi) ?*anyopaque;
extern "kernel32" fn ReadFile(h: *anyopaque, buf: [*]u8, n: u32, read: *u32, overlapped: ?*anyopaque) callconv(.winapi) i32;
extern "kernel32" fn CloseHandle(h: *anyopaque) callconv(.winapi) i32;
extern "kernel32" fn GetFileInformationByHandle(h: *anyopaque, info: *win.ByHandleInfo) callconv(.winapi) i32;
extern "kernel32" fn LocalFree(mem: ?*anyopaque) callconv(.winapi) ?*anyopaque;
extern "advapi32" fn GetSecurityInfo(h: *anyopaque, object: u32, info: u32, owner: *?*anyopaque, group: ?*?*anyopaque, dacl: *?*anyopaque, sacl: ?*?*anyopaque, sd: *?*anyopaque) callconv(.winapi) u32;
extern "advapi32" fn GetAclInformation(acl: *anyopaque, info: *win.AclSize, len: u32, class: u32) callconv(.winapi) i32;
extern "advapi32" fn GetAce(acl: *anyopaque, index: u32, ace: *?*anyopaque) callconv(.winapi) i32;
extern "advapi32" fn EqualSid(a: *anyopaque, b: *anyopaque) callconv(.winapi) i32;
extern "advapi32" fn IsWellKnownSid(sid: *anyopaque, kind: c_int) callconv(.winapi) i32;

fn winOpen(pathz: [:0]const u8) ?posix.fd_t {
    var wide: [1024:0]u16 = undefined;
    const n = std.unicode.utf8ToUtf16Le(wide[0..1023], pathz) catch return null;
    wide[n] = 0;
    const h = CreateFileW(wide[0..n :0].ptr, win.GENERIC_READ, win.FILE_SHARE_ALL, null, win.OPEN_EXISTING, win.FILE_FLAG_OPEN_REPARSE_POINT, null) orelse return null;
    if (@intFromPtr(h) == std.math.maxInt(usize)) return null;
    return h;
}

fn winClose(h: posix.fd_t) void {
    _ = CloseHandle(h);
}

fn winRead(h: posix.fd_t, buf: []u8) error{Unsafe}!usize {
    var got: u32 = 0;
    const n: u32 = @intCast(@min(buf.len, std.math.maxInt(u32)));
    if (ReadFile(h, buf.ptr, n, &got, null) == 0) return error.Unsafe;
    return got;
}

/// Windows' FILETIME (100 ns ticks since 1601) as Unix nanoseconds.
fn fileTimeNs(low: u32, high: u32) i128 {
    const ticks = (@as(i128, high) << 32) | low;
    return (ticks - 116444736000000000) * 100;
}

fn winMeta(fd: ?posix.fd_t, path: ?[*:0]const u8) ?Meta {
    var opened: ?posix.fd_t = null;
    defer if (opened) |h| winClose(h);
    const h = if (fd) |f| f else blk: {
        opened = winOpen(std.mem.span(path.?)) orelse return null;
        break :blk opened.?;
    };
    var info: win.ByHandleInfo = undefined;
    if (GetFileInformationByHandle(h, &info) == 0) return null;
    // a link is unsafe to read, as O_NOFOLLOW makes it on POSIX
    if (info.attributes & win.FILE_ATTRIBUTE_REPARSE_POINT != 0) return null;
    const others = othersCanRead(h) orelse return null;
    const kind: u32 = if (info.attributes & win.FILE_ATTRIBUTE_DIRECTORY != 0) S_IFDIR else S_IFREG;
    return .{
        .mode = kind | 0o600 | (if (others) @as(u32, 0o044) else 0),
        .size = (@as(u64, info.size_high) << 32) | info.size_low,
        .dev = info.volume_serial,
        .ino = (@as(u64, info.index_high) << 32) | info.index_low,
        .mtime_ns = fileTimeNs(info.write[0], info.write[1]),
    };
}

/// True when an allowed ACE other than the owner's, SYSTEM's or Administrators' grants read, or when the DACL is absent
/// (which grants everyone everything). Null when the security information cannot be read.
fn othersCanRead(h: *anyopaque) ?bool {
    var owner: ?*anyopaque = null;
    var dacl: ?*anyopaque = null;
    var sd: ?*anyopaque = null;
    if (GetSecurityInfo(h, win.SE_FILE_OBJECT, win.OWNER_SECURITY_INFORMATION | win.DACL_SECURITY_INFORMATION, &owner, null, &dacl, null, &sd) != 0) return null;
    defer _ = LocalFree(sd);
    const acl = dacl orelse return true;
    var size: win.AclSize = undefined;
    if (GetAclInformation(acl, &size, @sizeOf(win.AclSize), win.AclSizeInformation) == 0) return null;
    var i: u32 = 0;
    while (i < size.count) : (i += 1) {
        var ace: ?*anyopaque = null;
        if (GetAce(acl, i, &ace) == 0) return null;
        const bytes: [*]u8 = @ptrCast(ace.?);
        if (bytes[0] != win.ACCESS_ALLOWED_ACE_TYPE) continue;
        if (std.mem.readInt(u32, bytes[4..8], .little) & win.READ_BITS == 0) continue;
        const sid: *anyopaque = @ptrCast(bytes + 8);
        if (owner) |o| if (EqualSid(sid, o) != 0) continue;
        if (IsWellKnownSid(sid, win.WinLocalSystemSid) != 0) continue;
        if (IsWellKnownSid(sid, win.WinBuiltinAdministratorsSid) != 0) continue;
        return true;
    }
    return false;
}

extern "kernel32" fn WriteFile(h: *anyopaque, buf: [*]const u8, n: u32, written: *u32, overlapped: ?*anyopaque) callconv(.winapi) i32;
extern "kernel32" fn DeleteFileW(name: [*:0]const u16) callconv(.winapi) i32;
extern "advapi32" fn SetNamedSecurityInfoW(name: [*:0]const u16, object: u32, info: u32, owner: ?*anyopaque, group: ?*anyopaque, dacl: ?*anyopaque, sacl: ?*anyopaque) callconv(.winapi) u32;
extern "advapi32" fn InitializeAcl(acl: *anyopaque, len: u32, revision: u32) callconv(.winapi) i32;
extern "advapi32" fn AddAccessAllowedAce(acl: *anyopaque, revision: u32, mask: u32, sid: *anyopaque) callconv(.winapi) i32;
extern "advapi32" fn ConvertStringSidToSidW(text: [*:0]const u16, sid: *?*anyopaque) callconv(.winapi) i32;

/// Gives the key file a DACL of its owner, SYSTEM and Administrators, plus Everyone's read when `everyone_reads`.
fn setKeyAcl(wide_path: [*:0]const u16, everyone_reads: bool) !void {
    const h = CreateFileW(wide_path, win.GENERIC_READ, win.FILE_SHARE_ALL, null, win.OPEN_EXISTING, 0, null) orelse return error.KeyFileSetup;
    defer _ = CloseHandle(h);
    var owner: ?*anyopaque = null;
    var sd: ?*anyopaque = null;
    var dacl: ?*anyopaque = null;
    if (GetSecurityInfo(h, win.SE_FILE_OBJECT, win.OWNER_SECURITY_INFORMATION, &owner, null, &dacl, null, &sd) != 0) return error.KeyFileSetup;
    defer _ = LocalFree(sd);
    var system: ?*anyopaque = null;
    var admins: ?*anyopaque = null;
    var everyone: ?*anyopaque = null;
    if (ConvertStringSidToSidW(std.unicode.utf8ToUtf16LeStringLiteral("S-1-5-18"), &system) == 0) return error.KeyFileSetup;
    if (ConvertStringSidToSidW(std.unicode.utf8ToUtf16LeStringLiteral("S-1-5-32-544"), &admins) == 0) return error.KeyFileSetup;
    if (ConvertStringSidToSidW(std.unicode.utf8ToUtf16LeStringLiteral("S-1-1-0"), &everyone) == 0) return error.KeyFileSetup;
    var buf: [512]u8 align(8) = undefined;
    if (InitializeAcl(&buf, buf.len, 2) == 0) return error.KeyFileSetup;
    const full: u32 = 0x1F01FF;
    if (AddAccessAllowedAce(&buf, 2, full, owner.?) == 0) return error.KeyFileSetup;
    if (AddAccessAllowedAce(&buf, 2, full, system.?) == 0) return error.KeyFileSetup;
    if (AddAccessAllowedAce(&buf, 2, full, admins.?) == 0) return error.KeyFileSetup;
    if (everyone_reads and AddAccessAllowedAce(&buf, 2, 0x120089, everyone.?) == 0) return error.KeyFileSetup;
    const protected_dacl: u32 = 0x80000000 | win.DACL_SECURITY_INFORMATION;
    if (SetNamedSecurityInfoW(wide_path, win.SE_FILE_OBJECT, protected_dacl, null, null, &buf, null) != 0) return error.KeyFileSetup;
}

test "a key file that another account can read is refused, and an owner-only one is read" {
    if (builtin.os.tag != .windows) return error.SkipZigTest;
    const name = "tf-auth-acl-test.key";
    var wide: [64:0]u16 = undefined;
    const n = try std.unicode.utf8ToUtf16Le(&wide, name);
    wide[n] = 0;
    const path_w: [*:0]const u16 = wide[0..n :0];
    const created = CreateFileW(path_w, 0xC0000000, 0, null, 2, 0x80, null) orelse return error.KeyFileSetup;
    var written: u32 = 0;
    const body = "tester-key\n";
    _ = WriteFile(created, body.ptr, body.len, &written, null);
    _ = CloseHandle(created);
    defer _ = DeleteFileW(path_w);

    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    var problem: []const u8 = "";

    try setKeyAcl(path_w, false);
    const private = try Store.init(arena.allocator(), &.{}, "", name, false, &problem);
    try std.testing.expectEqual(@as(usize, 1), private.file.len);

    try setKeyAcl(path_w, true);
    try std.testing.expectError(error.KeyFile, Store.init(arena.allocator(), &.{}, "", name, false, &problem));
    try std.testing.expectEqualStrings("API key file must be a regular file unreadable by other users; use chmod 600", problem);
}

extern "kernel32" fn SetEvent(handle: win32.HANDLE) callconv(.winapi) i32;
extern "kernel32" fn GetFileTime(h: *anyopaque, created: ?*anyopaque, accessed: ?*anyopaque, written: *anyopaque) callconv(.winapi) i32;
extern "kernel32" fn SetFileTime(h: *anyopaque, created: ?*anyopaque, accessed: ?*anyopaque, written: *anyopaque) callconv(.winapi) i32;

/// Overwrites a key file in place with a body of the same length and the same write time, so its stamp cannot show
/// the change: only a reload request can make the store reread it.
fn rewriteKeepingStamp(path_w: [*:0]const u16, body: []const u8) !void {
    const h = CreateFileW(path_w, 0xC0000000, 0, null, 3, 0x80, null) orelse return error.KeyFileSetup;
    defer _ = CloseHandle(h);
    var mtime: [2]u32 = undefined;
    if (GetFileTime(h, null, null, &mtime) == 0) return error.KeyFileSetup;
    var written: u32 = 0;
    if (WriteFile(h, body.ptr, @intCast(body.len), &written, null) == 0 or written != body.len) return error.KeyFileSetup;
    if (SetFileTime(h, null, null, &mtime) == 0) return error.KeyFileSetup;
}

test "a reload event rereads the key file, so the new key replaces the old one" {
    if (builtin.os.tag != .windows) return error.SkipZigTest;
    const name = "tf-auth-reload-test.key";
    var wide: [64:0]u16 = undefined;
    const n = try std.unicode.utf8ToUtf16Le(&wide, name);
    wide[n] = 0;
    const path_w: [*:0]const u16 = wide[0..n :0];
    const created = CreateFileW(path_w, 0xC0000000, 0, null, 2, 0x80, null) orelse return error.KeyFileSetup;
    var written: u32 = 0;
    _ = WriteFile(created, "old-key\n", 8, &written, null);
    _ = CloseHandle(created);
    defer _ = DeleteFileW(path_w);
    try setKeyAcl(path_w, false);

    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var problem: []const u8 = "";
    var store = try Store.init(a, &.{}, "", name, false, &problem);
    installHangup();
    try std.testing.expect(store.match(a, null, "old-key") != null);

    try rewriteKeepingStamp(path_w, "new-key\n");
    try std.testing.expect(store.match(a, null, "new-key") == null); // the stamp is unchanged: no reread yet

    const event = win32.createReloadEvent(win32.GetCurrentProcessId()) orelse return error.EventRefused;
    try std.testing.expect(SetEvent(event) != 0);
    try std.testing.expect(store.match(a, null, "new-key") != null);
    try std.testing.expect(store.match(a, null, "old-key") == null);
}
