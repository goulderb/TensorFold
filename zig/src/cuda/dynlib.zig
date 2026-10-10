//! Shared libraries opened at run time on Linux (std.DynLib) and Windows (LoadLibraryExW), where Zig 0.17's
//! std.DynLib is a compile error.
//!
//! `openSystem` takes a bare name. On Windows it searches System32 alone (LOAD_LIBRARY_SEARCH_SYSTEM32), so a DLL of
//! the same name beside the executable, in the working directory or on PATH is never loaded in place of the driver.
//! `openFile` takes an absolute path and loads exactly that file.

const std = @import("std");
const builtin = @import("builtin");

pub const Error = error{ LibraryUnavailable, MissingSymbol };

const is_windows = builtin.os.tag == .windows;

const win = struct {
    const HMODULE = std.os.windows.HMODULE;
    const load_library_search_system32: u32 = 0x00000800;
    const load_library_search_dll_load_dir: u32 = 0x00000100;
    extern "kernel32" fn LoadLibraryExW(name: [*:0]const u16, file: ?*anyopaque, flags: u32) callconv(.winapi) ?HMODULE;
    extern "kernel32" fn GetProcAddress(module: HMODULE, name: [*:0]const u8) callconv(.winapi) ?*anyopaque;
    extern "kernel32" fn FreeLibrary(module: HMODULE) callconv(.winapi) c_int;
};

pub const Library = struct {
    inner: if (is_windows) win.HMODULE else std.DynLib,

    /// A library by bare name: System32 only on Windows, the dynamic linker's search on Linux.
    pub fn openSystem(name: []const u8) Error!Library {
        if (std.mem.indexOfAny(u8, name, "/\\") != null) return error.LibraryUnavailable;
        if (is_windows) return loadW(name, win.load_library_search_system32);
        return .{ .inner = std.DynLib.open(name) catch return error.LibraryUnavailable };
    }

    /// Exactly the file at absolute `path`; its own directory and System32 resolve its imports on Windows.
    pub fn openFile(path: []const u8) Error!Library {
        if (!std.fs.path.isAbsolute(path)) return error.LibraryUnavailable;
        if (is_windows) return loadW(path, win.load_library_search_dll_load_dir | win.load_library_search_system32);
        return .{ .inner = std.DynLib.open(path) catch return error.LibraryUnavailable };
    }

    /// `openFile` for an absolute path, `openSystem` for a bare name.
    pub fn open(path: []const u8) Error!Library {
        return if (std.fs.path.isAbsolute(path)) openFile(path) else openSystem(path);
    }

    fn loadW(name: []const u8, flags: u32) Error!Library {
        var buf: [512:0]u16 = undefined;
        const n = std.unicode.utf8ToUtf16Le(&buf, name) catch return error.LibraryUnavailable;
        if (n >= buf.len) return error.LibraryUnavailable;
        buf[n] = 0;
        const h = win.LoadLibraryExW(buf[0..n :0], null, flags) orelse return error.LibraryUnavailable;
        return .{ .inner = h };
    }

    pub fn lookup(self: *Library, comptime T: type, name: [:0]const u8) ?T {
        if (is_windows) {
            const p = win.GetProcAddress(self.inner, name.ptr) orelse return null;
            return @ptrCast(@alignCast(p));
        }
        return self.inner.lookup(T, name);
    }

    pub fn close(self: *Library) void {
        if (is_windows) _ = win.FreeLibrary(self.inner) else self.inner.close();
        self.* = undefined;
    }

    /// Every field of the struct `Table` resolved by its exact name; the first missing one refuses the whole table.
    pub fn resolve(self: *Library, comptime Table: type, missing: *[]const u8) Error!Table {
        var t: Table = undefined;
        const info = @typeInfo(Table).@"struct";
        inline for (info.field_names, info.field_types) |name, T| {
            @field(t, name) = self.lookup(T, name) orelse {
                missing.* = name;
                return error.MissingSymbol;
            };
        }
        return t;
    }
};

/// The CUDA driver's file name on this OS.
pub const cuda_driver = if (is_windows) "nvcuda.dll" else "libcuda.so.1";

test "a bare name with a path separator is refused before any search" {
    try std.testing.expectError(error.LibraryUnavailable, Library.openSystem("./libc.so.6"));
    try std.testing.expectError(error.LibraryUnavailable, Library.openSystem("..\\nvcuda.dll"));
    try std.testing.expectError(error.LibraryUnavailable, Library.openFile("relative/lib.so"));
}

test "a library that does not exist is LibraryUnavailable, not a crash" {
    try std.testing.expectError(error.LibraryUnavailable, Library.openSystem("tf-no-such-library-4e1c.so"));
}

test "the C library resolves a present symbol and a table with one wrong name is MissingSymbol naming it" {
    if (is_windows) return error.SkipZigTest; // tf-dynlib-test covers Windows natively
    var lib = try Library.openSystem("libc.so.6");
    defer lib.close();
    try std.testing.expect(lib.lookup(*const fn () callconv(.c) c_int, "getpid") != null);
    var missing: []const u8 = "";
    const Good = struct { getpid: *const fn () callconv(.c) c_int };
    _ = try lib.resolve(Good, &missing);
    const Bad = struct { getpid: *const fn () callconv(.c) c_int, tf_no_such_symbol: *const fn () callconv(.c) c_int };
    try std.testing.expectError(error.MissingSymbol, lib.resolve(Bad, &missing));
    try std.testing.expectEqualStrings("tf_no_such_symbol", missing);
}

test "kernel32 resolves a present symbol and a table with one wrong name is MissingSymbol naming it (Windows)" {
    if (!is_windows) return error.SkipZigTest;
    var lib = try Library.openSystem("kernel32.dll");
    defer lib.close();
    var missing: []const u8 = "";
    const Good = struct { GetTickCount64: *const fn () callconv(.winapi) u64 };
    _ = try lib.resolve(Good, &missing);
    const Bad = struct { GetTickCount64: *const fn () callconv(.winapi) u64, tf_no_such_symbol: *const fn () callconv(.winapi) u64 };
    try std.testing.expectError(error.MissingSymbol, lib.resolve(Bad, &missing));
    try std.testing.expectEqualStrings("tf_no_such_symbol", missing);
}
