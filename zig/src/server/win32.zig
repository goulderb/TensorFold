//! The Win32 calls the server needs that Zig 0.17's std does not bind: console control, the named reload event, and
//! the memory status. Only the Windows build reaches these; the POSIX build never analyses this file's bodies.
const std = @import("std");
const builtin = @import("builtin");

pub const BOOL = i32;
pub const DWORD = u32;
pub const HANDLE = *anyopaque;

pub const WAIT_OBJECT_0: DWORD = 0;
pub const CTRL_C_EVENT: DWORD = 0;
pub const CTRL_BREAK_EVENT: DWORD = 1;
pub const CTRL_CLOSE_EVENT: DWORD = 2;

pub extern "kernel32" fn SetConsoleCtrlHandler(handler: ?*const fn (DWORD) callconv(.winapi) BOOL, add: BOOL) callconv(.winapi) BOOL;
pub extern "kernel32" fn CreateEventW(attrs: ?*anyopaque, manual_reset: BOOL, initial: BOOL, name: ?[*:0]const u16) callconv(.winapi) ?HANDLE;
pub extern "kernel32" fn WaitForSingleObject(handle: HANDLE, ms: DWORD) callconv(.winapi) DWORD;
pub extern "kernel32" fn ResetEvent(handle: HANDLE) callconv(.winapi) BOOL;
pub extern "kernel32" fn GetCurrentProcessId() callconv(.winapi) DWORD;

/// The stop function the console handler calls. Runs on a thread the system creates, so it must only set flags.
var on_console_stop: ?*const fn () void = null;

fn consoleHandler(ctrl: DWORD) callconv(.winapi) BOOL {
    switch (ctrl) {
        CTRL_C_EVENT, CTRL_BREAK_EVENT, CTRL_CLOSE_EVENT => {
            if (on_console_stop) |f| f();
            return 1;
        },
        else => return 0,
    }
}

/// Routes Ctrl+C, Ctrl+Break and console close to ``stop``. Returns false if the system refused the handler.
pub fn installConsoleStop(stop: *const fn () void) bool {
    comptime std.debug.assert(builtin.os.tag == .windows);
    on_console_stop = stop;
    return SetConsoleCtrlHandler(consoleHandler, 1) != 0;
}

/// The name of the event that asks a server to reread its key file: the POSIX SIGHUP. It is per process, so a
/// fleet can signal one server without touching another.
pub fn reloadEventName(buf: *[64:0]u16, pid: DWORD) [*:0]const u16 {
    var i: usize = 0;
    for ("Local\\tensorfold-reload-") |c| {
        buf[i] = c;
        i += 1;
    }
    var digits: [10]u8 = undefined;
    const text = std.fmt.bufPrint(&digits, "{d}", .{pid}) catch unreachable;
    for (text) |c| {
        buf[i] = c;
        i += 1;
    }
    buf[i] = 0;
    return buf[0..i :0];
}

/// Creates (or opens, if one exists) the auto-reset reload event for this process. Null when the system refuses.
pub fn createReloadEvent(pid: DWORD) ?HANDLE {
    var name: [64:0]u16 = undefined;
    return CreateEventW(null, 0, 0, reloadEventName(&name, pid));
}

/// True once per signal: a set event is reset as it is seen.
pub fn takeReloadEvent(handle: HANDLE) bool {
    if (WaitForSingleObject(handle, 0) != WAIT_OBJECT_0) return false;
    _ = ResetEvent(handle);
    return true;
}

extern "kernel32" fn SetEvent(handle: HANDLE) callconv(.winapi) BOOL;

test "a set reload event is taken once, and the name is per process" {
    if (comptime builtin.os.tag != .windows) return error.SkipZigTest;
    const h = createReloadEvent(GetCurrentProcessId()) orelse return error.EventRefused;
    defer _ = std.os.windows.CloseHandle(h);
    try std.testing.expect(!takeReloadEvent(h));
    try std.testing.expect(SetEvent(h) != 0);
    try std.testing.expect(takeReloadEvent(h));
    try std.testing.expect(!takeReloadEvent(h));
    var mine: [64:0]u16 = undefined;
    var theirs: [64:0]u16 = undefined;
    const own_name = std.mem.span(reloadEventName(&mine, 4242));
    const other_name = std.mem.span(reloadEventName(&theirs, 4243));
    try std.testing.expect(!std.mem.eql(u16, own_name, other_name));
    try std.testing.expectEqual(@as(usize, "Local\\tensorfold-reload-4242".len), own_name.len);
}
