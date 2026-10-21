//! The OS as TensorFold calls it, imported as `sys`: `c` is std.c or win_c.zig, `DynLib` is std.DynLib or LoadLibraryW.

const std = @import("std");
const builtin = @import("builtin");

pub const windows = builtin.os.tag == .windows;

pub const c = if (windows) @import("win_c.zig") else std.c;

/// std.posix, or its sockets, signals and limits over Winsock and console events.
pub const posix = if (windows) @import("win_posix.zig") else std.posix;

pub const DynLib = if (windows) WinDynLib else std.DynLib;
pub const experimental = @import("experimental.zig");
pub const optional_libraries = @import("optional_libraries.zig");

/// An environment variable's value, through libc (std.process.Environ.getPosix does not build on Windows).
pub fn getenv(name: [*:0]const u8) ?[:0]const u8 {
    return std.mem.span(c.getenv(name) orelse return null);
}

/// The CUDA driver's file name: nvcuda.dll comes with the display driver on Windows.
pub const cuda_driver = if (windows) "nvcuda.dll" else "libcuda.so.1";

/// The file a loaded library's `symbol` comes from, as dladdr's dli_fname, in `buf` on Windows.
pub fn libraryOf(symbol: *const anyopaque, buf: []u8) ?[]const u8 {
    if (!windows) @compileError("libraryOf: dladdr gives the path elsewhere");
    var module: ?*anyopaque = null;
    // GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | UNCHANGED_REFCOUNT: the module holding the address, not pinned
    if (GetModuleHandleExA(0x4 | 0x2, @ptrCast(symbol), &module) == 0) return null;
    const n = GetModuleFileNameA(module, buf.ptr, @intCast(@min(buf.len, std.math.maxInt(u32))));
    return if (n == 0 or n >= buf.len) null else buf[0..n];
}

extern "kernel32" fn GetModuleHandleExA(flags: u32, name: ?[*]const u8, module: *?*anyopaque) callconv(.winapi) c_int;
extern "kernel32" fn GetModuleFileNameA(module: ?*anyopaque, name: [*]u8, size: u32) callconv(.winapi) u32;
extern "kernel32" fn LoadLibraryExW(name: [*:0]const u16, file: ?*anyopaque, flags: u32) callconv(.winapi) ?*anyopaque;
extern "kernel32" fn GetProcAddress(module: *anyopaque, name: [*:0]const u8) callconv(.winapi) ?*anyopaque;
extern "kernel32" fn FreeLibrary(module: *anyopaque) callconv(.winapi) c_int;

const WinDynLib = struct {
    module: *anyopaque,

    pub const Error = @import("windows_library.zig").Error;

    pub fn open(path: []const u8) Error!WinDynLib {
        return .{ .module = try @import("windows_library.zig").open(Kernel, path) };
    }
    const Kernel = struct {
        pub const Handle = *anyopaque;
        pub fn load(path: [:0]const u16, flags: u32) ?Handle {
            return LoadLibraryExW(path.ptr, null, flags);
        }
    };

    pub fn openZ(path: [*:0]const u8) Error!WinDynLib {
        return WinDynLib.open(std.mem.span(path));
    }

    pub fn close(self: *WinDynLib) void {
        _ = FreeLibrary(self.module);
    }

    pub fn lookup(self: *WinDynLib, comptime T: type, name: [:0]const u8) ?T {
        const p = GetProcAddress(self.module, name.ptr) orelse return null;
        return @ptrCast(@alignCast(p));
    }
};

pub const sock = @import("socket.zig");

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

/// Appends `bytes` to the file at `path`, created 0600 if absent, in one write. A failure is silent: the caller logs
/// nothing it could not write anyway.
pub fn appendFile(a: std.mem.Allocator, path: []const u8, bytes: []const u8) void {
    if (builtin.os.tag == .windows) {
        const wide = std.unicode.utf8ToUtf16LeAllocZ(a, path) catch return;
        defer a.free(wide);
        // FILE_APPEND_DATA alone makes every write land at the end, as O_APPEND does; OPEN_ALWAYS creates the file
        const h = CreateFileW(wide.ptr, 0x0004, 0x7, null, 4, 0x80, null) orelse return;
        defer _ = CloseHandle(h);
        if (@intFromPtr(h) == win_invalid_handle) return;
        var written: u32 = 0;
        _ = WriteFile(h, bytes.ptr, @intCast(bytes.len), &written, null);
        return;
    }
    const pathz = a.dupeSentinel(u8, path, 0) catch return;
    defer a.free(pathz);
    const fd = std.posix.openatZ(std.posix.AT.FDCWD, pathz, .{ .ACCMODE = .WRONLY, .APPEND = true, .CREAT = true }, 0o600) catch return;
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

/// Loads an explicitly configured file without a relative-path fallback.
pub fn openLibraryFile(path: []const u8) !DynLib {
    if (!std.fs.path.isAbsolute(path)) return error.InvalidPath;
    return DynLib.open(path);
}

/// A missing symbol refuses the complete foreign API table.
pub fn resolveSymbols(lib: *DynLib, comptime Table: type, missing: *[]const u8) error{MissingSymbol}!Table {
    var table: Table = undefined;
    const info = @typeInfo(Table).@"struct";
    inline for (info.field_names, info.field_types) |name, T| {
        @field(table, name) = lib.lookup(T, name) orelse {
            missing.* = name;
            return error.MissingSymbol;
        };
    }
    return table;
}

test "platform DLL table: missing symbols refuse the complete table" {
    var lib = try DynLib.open(if (windows) "kernel32.dll" else "libc.so.6");
    defer lib.close();
    var missing: []const u8 = "";
    if (windows) {
        _ = try resolveSymbols(&lib, struct { GetTickCount64: *const fn () callconv(.winapi) u64 }, &missing);
    } else {
        _ = try resolveSymbols(&lib, struct { getpid: *const fn () callconv(.c) c_int }, &missing);
    }
    try std.testing.expectError(error.MissingSymbol, resolveSymbols(&lib, struct { tf_no_such_symbol: *const fn () callconv(.c) void }, &missing));
    try std.testing.expectEqualStrings("tf_no_such_symbol", missing);
    try std.testing.expectError(error.InvalidPath, openLibraryFile("relative/library.dll"));
}
