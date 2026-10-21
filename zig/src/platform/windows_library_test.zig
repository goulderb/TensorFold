const std = @import("std");
const library = @import("windows_library.zig");
const t = std.testing;
const u = std.unicode.utf8ToUtf16LeStringLiteral;
const Fake = struct {
    pub const Handle = usize;
    var fail_load = false;
    var loads: usize = 0;
    var flags: u32 = 0;
    var loaded: [1024]u16 = undefined;
    var loaded_len: usize = 0;
    fn reset() void {
        fail_load = false;
        loads = 0;
        flags = 0;
    }
    pub fn load(path: [:0]const u16, search: u32) ?Handle {
        loads += 1;
        flags = search;
        loaded_len = path.len;
        @memcpy(loaded[0..loaded_len], path);
        std.debug.assert(path[path.len] == 0);
        return if (fail_load) null else 7;
    }
};

test "Windows DLL: absolute and Unicode files search only their directory and System32" {
    const cases = .{
        .{ "C:/CUDA/cublas.dll", u("C:\\CUDA\\cublas.dll") },
        .{ "C:\\CUDA\\音𝄞.dll", u("C:\\CUDA\\音𝄞.dll") },
        .{ "\\\\server\\runtime\\cublas.dll", u("\\\\server\\runtime\\cublas.dll") },
    };
    inline for (cases) |c| {
        Fake.reset();
        try t.expectEqual(@as(usize, 7), try library.open(Fake, c[0]));
        try t.expectEqual(@as(usize, 1), Fake.loads);
        try t.expectEqual(library.search_system32 | library.search_dll_load_dir, Fake.flags);
        try t.expectEqualSlices(u16, c[1], Fake.loaded[0..Fake.loaded_len]);
    }
}

test "Windows DLL: bare names never search PATH, the executable directory or the current directory" {
    Fake.reset();
    _ = try library.open(Fake, "nvcuda.dll");
    try t.expectEqual(library.search_system32, Fake.flags);
    try t.expectEqualSlices(u16, u("nvcuda.dll"), Fake.loaded[0..Fake.loaded_len]);
    for ([_][]const u8{ "runtime/cublas.dll", "..\\nvcuda.dll", "C:nvcuda.dll", "\\runtime\\cublas.dll", "/runtime/cublas.dll", "", "nvcuda.dll\x00other.dll" }) |name| {
        Fake.reset();
        try t.expectError(error.InvalidPath, library.open(Fake, name));
        try t.expectEqual(@as(usize, 0), Fake.loads);
    }
}

test "Windows DLL: missing files, encoding and length failures do not fall back to another search" {
    Fake.reset();
    Fake.fail_load = true;
    try t.expectError(error.FileNotFound, library.open(Fake, "C:/runtime/missing.dll"));
    try t.expectEqual(@as(usize, 1), Fake.loads);
    Fake.reset();
    try t.expectError(error.NameTooLong, library.open(Fake, &@as([1024]u8, @splat('a'))));
    try t.expectError(error.InvalidWtf8, library.open(Fake, &.{0xff}));
    try t.expectEqual(@as(usize, 0), Fake.loads);
}
