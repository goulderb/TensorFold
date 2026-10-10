//! Environment variables on both hosts: POSIX reads and sets the C environment, Windows the process block in UTF-16.
//! Names and values are bounded (255 and 4095 bytes), so no call allocates.
const std = @import("std");
const builtin = @import("builtin");

const is_windows = builtin.os.tag == .windows;
const ERROR_ENVVAR_NOT_FOUND: u32 = 203;
const max_name = 255;
const max_value = 4095;

extern "kernel32" fn GetEnvironmentVariableW(name: [*:0]const u16, buf: ?[*]u16, size: u32) callconv(.winapi) u32;
extern "kernel32" fn SetEnvironmentVariableW(name: [*:0]const u16, value: ?[*:0]const u16) callconv(.winapi) i32;
extern "kernel32" fn GetLastError() callconv(.winapi) u32;
extern "kernel32" fn SetLastError(code: u32) callconv(.winapi) void;

const posix = if (is_windows) struct {} else struct {
    extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
    extern "c" fn unsetenv(name: [*:0]const u8) c_int;
};

/// The value of `name` copied into `buf` as UTF-8, or null when it is unset. Empty is present and empty.
pub fn getInto(buf: []u8, name: []const u8) !?[]const u8 {
    if (comptime !is_windows) {
        var z: [max_name:0]u8 = undefined;
        if (name.len > max_name) return error.EnvironmentName;
        @memcpy(z[0..name.len], name);
        z[name.len] = 0;
        const p = std.c.getenv(&z) orelse return null;
        const v = std.mem.span(p);
        if (v.len > buf.len) return error.EnvironmentTooLong;
        @memcpy(buf[0..v.len], v);
        return buf[0..v.len];
    }
    var wname: [max_name:0]u16 = undefined;
    const n = try std.unicode.utf8ToUtf16Le(&wname, name);
    wname[n] = 0;
    var wbuf: [max_value + 1]u16 = undefined;
    SetLastError(0);
    const got = GetEnvironmentVariableW(&wname, &wbuf, wbuf.len);
    if (got == 0) {
        const code = GetLastError();
        if (code == 0) return buf[0..0];
        return if (code == ERROR_ENVVAR_NOT_FOUND) null else error.EnvironmentRead;
    }
    if (got > max_value) return error.EnvironmentTooLong;
    var utf8: [3 * max_value]u8 = undefined; // a UTF-16 unit makes at most 3 UTF-8 bytes
    const len = try std.unicode.utf16LeToUtf8(&utf8, wbuf[0..got]);
    if (len > buf.len) return error.EnvironmentTooLong;
    @memcpy(buf[0..len], utf8[0..len]);
    return buf[0..len];
}

/// Sets `name` to `value`, or removes it when `value` is null.
pub fn set(name: []const u8, value: ?[]const u8) !void {
    if (comptime !is_windows) {
        if (name.len > max_name) return error.EnvironmentName;
        var z: [max_name:0]u8 = undefined;
        @memcpy(z[0..name.len], name);
        z[name.len] = 0;
        if (value) |v| {
            if (v.len > max_value) return error.EnvironmentTooLong;
            var vz: [max_value:0]u8 = undefined;
            @memcpy(vz[0..v.len], v);
            vz[v.len] = 0;
            if (posix.setenv(&z, &vz, 1) != 0) return error.EnvironmentWrite;
        } else if (posix.unsetenv(&z) != 0) return error.EnvironmentWrite;
        return;
    }
    var wname: [max_name:0]u16 = undefined;
    const n = try std.unicode.utf8ToUtf16Le(&wname, name);
    wname[n] = 0;
    var wvalue: [max_value:0]u16 = undefined;
    var wv: ?[*:0]const u16 = null;
    if (value) |v| {
        const m = try std.unicode.utf8ToUtf16Le(&wvalue, v);
        wvalue[m] = 0;
        wv = &wvalue;
    }
    if (SetEnvironmentVariableW(&wname, wv) == 0) return error.EnvironmentWrite;
}

test "a value round-trips through set and getInto, unicode included, and unsetting makes it absent" {
    var buf: [256]u8 = undefined;
    try set("TF_ENV_TEST_UNICODE", "caf\u{e9} \u{2603}");
    const got = (try getInto(&buf, "TF_ENV_TEST_UNICODE")) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqualStrings("caf\u{e9} \u{2603}", got);
    try set("TF_ENV_TEST_UNICODE", null);
    try std.testing.expect((try getInto(&buf, "TF_ENV_TEST_UNICODE")) == null);
}

test "an empty value is present and empty, not absent" {
    var buf: [16]u8 = undefined;
    try set("TF_ENV_TEST_EMPTY", "");
    defer set("TF_ENV_TEST_EMPTY", null) catch {};
    const got = (try getInto(&buf, "TF_ENV_TEST_EMPTY")) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(@as(usize, 0), got.len);
}

test "a value longer than the caller's buffer is refused, not truncated" {
    var small: [2]u8 = undefined;
    try set("TF_ENV_TEST_LONG", "abcdef");
    defer set("TF_ENV_TEST_LONG", null) catch {};
    try std.testing.expectError(error.EnvironmentTooLong, getInto(&small, "TF_ENV_TEST_LONG"));
}

test "a name that was never set is absent" {
    var buf: [16]u8 = undefined;
    try std.testing.expect((try getInto(&buf, "TF_ENV_NEVER_SET_4e1c_9b07")) == null);
}
