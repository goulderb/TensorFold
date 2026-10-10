//! W2's console control test, run in a console of its own (the `test-server-console` step, not `zig build test`).
//!
//!   tf-console-test                  the driver: starts a child in a new process group of this console, sends it
//!                                    CTRL_BREAK after a moment, and checks how it ended
//!   tf-console-test handler          the child installs the server's console handler and waits for it
//!   tf-console-test bare             the child installs nothing: the red control, which must not end cleanly
//!
//! The driver passes when the handler child exits 0 after printing STOPPED, and the bare child does not exit 0.
const std = @import("std");
const win32 = @import("win32");

const CREATE_NEW_PROCESS_GROUP: u32 = 0x200;
const INFINITE: u32 = 0xFFFF_FFFF;
const CTRL_BREAK_EVENT: u32 = 1;

const StartupInfo = extern struct {
    cb: u32,
    reserved: ?*anyopaque = null,
    desktop: ?*anyopaque = null,
    title: ?*anyopaque = null,
    x: u32 = 0,
    y: u32 = 0,
    x_size: u32 = 0,
    y_size: u32 = 0,
    x_chars: u32 = 0,
    y_chars: u32 = 0,
    fill: u32 = 0,
    flags: u32 = 0,
    show: u16 = 0,
    reserved2_len: u16 = 0,
    reserved2: ?*anyopaque = null,
    stdin: ?*anyopaque = null,
    stdout: ?*anyopaque = null,
    stderr: ?*anyopaque = null,
};

const ProcessInfo = extern struct { process: ?*anyopaque, thread: ?*anyopaque, pid: u32, tid: u32 };

extern "kernel32" fn CreateProcessW(app: ?[*:0]const u16, cmd: [*:0]u16, pa: ?*anyopaque, ta: ?*anyopaque, inherit: i32, flags: u32, env: ?*anyopaque, cwd: ?[*:0]const u16, si: *StartupInfo, pi: *ProcessInfo) callconv(.winapi) i32;
extern "kernel32" fn WaitForSingleObject(h: *anyopaque, ms: u32) callconv(.winapi) u32;
extern "kernel32" fn GetExitCodeProcess(h: *anyopaque, code: *u32) callconv(.winapi) i32;
extern "kernel32" fn GenerateConsoleCtrlEvent(event: u32, group: u32) callconv(.winapi) i32;
extern "kernel32" fn CloseHandle(h: *anyopaque) callconv(.winapi) i32;
extern "kernel32" fn Sleep(ms: u32) callconv(.winapi) void;

var stop_seen: std.atomic.Value(bool) = .init(false);

fn onStop() void {
    stop_seen.store(true, .release);
}

fn child(mode: []const u8) u8 {
    if (std.mem.eql(u8, mode, "handler")) {
        if (!win32.installConsoleStop(onStop)) {
            std.debug.print("handler refused\n", .{});
            return 3;
        }
        while (!stop_seen.load(.acquire)) Sleep(10);
        std.debug.print("STOPPED\n", .{});
        return 0;
    }
    // bare: no handler; the default handler ends the process, so this line is never reached
    Sleep(20_000);
    std.debug.print("SURVIVED\n", .{});
    return 0;
}

fn utf16(a: std.mem.Allocator, s: []const u8) ![:0]u16 {
    return std.unicode.utf8ToUtf16LeAllocZ(a, s);
}

/// Starts this program as ``mode`` in a new process group of this console, signals it, and returns its exit code.
fn runChild(a: std.mem.Allocator, self: []const u8, mode: []const u8) !u32 {
    const cmd = try std.fmt.allocPrint(a, "\"{s}\" {s}", .{ self, mode });
    const cmdw = try utf16(a, cmd);
    var si: StartupInfo = .{ .cb = @sizeOf(StartupInfo) };
    var pi: ProcessInfo = undefined;
    if (CreateProcessW(null, cmdw.ptr, null, null, 0, CREATE_NEW_PROCESS_GROUP, null, null, &si, &pi) == 0) return error.SpawnFailed;
    defer _ = CloseHandle(pi.process.?);
    defer _ = CloseHandle(pi.thread.?);
    Sleep(1500); // the child has installed its handler by now (the bare child has no handler to install)
    if (GenerateConsoleCtrlEvent(CTRL_BREAK_EVENT, pi.pid) == 0) return error.SignalFailed;
    _ = WaitForSingleObject(pi.process.?, INFINITE);
    var code: u32 = 0;
    if (GetExitCodeProcess(pi.process.?, &code) == 0) return error.ExitCodeUnavailable;
    return code;
}

pub fn main(init: std.process.Init) !u8 {
    const a = init.arena.allocator();
    const argv = try init.minimal.args.toSlice(a);
    if (argv.len >= 2) return child(argv[1]);
    const self = argv[0];

    const handler_code = try runChild(a, self, "handler");
    const bare_code = try runChild(a, self, "bare");
    std.debug.print("handler child exit {d} (want 0); bare child exit 0x{X} (want nonzero)\n", .{ handler_code, bare_code });
    if (handler_code != 0) return 1;
    if (bare_code == 0) return 1;
    std.debug.print("PASS console control: CTRL_BREAK ends the handler child cleanly; the bare child does not\n", .{});
    return 0;
}
