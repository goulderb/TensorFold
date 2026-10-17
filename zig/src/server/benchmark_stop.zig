//! Ctrl-C, SIGTERM and SIGHUP during a benchmark: one flag the waits read; a second Ctrl-C exits at once.
//! Windows has no sigaction or termios. The flag still works; install, the child and the terminal are no-ops there.
const std = @import("std");
const builtin = @import("builtin");
const posix = std.posix;

const windows = builtin.os.tag == .windows;

var stored: std.atomic.Value(u32) = .init(0);

/// POSIX process state. Empty on Windows, so sigaction, termios and the HUP signal are not compiled there.
const Host = if (windows) struct {} else struct {
    var saved: [3]posix.Sigaction = undefined;
    var installed = false;
    const watched = [3]posix.SIG{ .INT, .TERM, .HUP };
    var child: std.atomic.Value(posix.pid_t) = .init(0);
    var tty_fd: std.atomic.Value(posix.fd_t) = .init(-1);
    var tty_saved: posix.termios = undefined;

    fn onSignal(sig: posix.SIG) callconv(.c) void {
        const n: u32 = @intCast(@backingInt(sig));
        if (stored.cmpxchgStrong(0, n, .acq_rel, .acquire)) |_| if (sig == .INT) {
            abandonNow();
            const default: posix.Sigaction = .{ .handler = .{ .handler = posix.SIG.DFL }, .mask = posix.sigemptyset(), .flags = 0 };
            posix.sigaction(.INT, &default, null);
            _ = std.c.raise(.INT);
        };
    }

    fn install() void {
        const act: posix.Sigaction = .{ .handler = .{ .handler = onSignal }, .mask = posix.sigemptyset(), .flags = 0 };
        for (watched, &saved) |sig, *old| posix.sigaction(sig, &act, old);
        installed = true;
    }

    fn restore() void {
        if (!installed) return;
        for (watched, &saved) |sig, *old| posix.sigaction(sig, old, null);
        installed = false;
    }

    fn guardChild(pid: posix.pid_t) void {
        child.store(pid, .release);
    }

    fn releaseChild(pid: posix.pid_t) void {
        _ = child.cmpxchgStrong(pid, 0, .acq_rel, .acquire);
    }

    fn guardedChild() posix.pid_t {
        return child.load(.acquire);
    }

    fn guardTerminal(fd: posix.fd_t, settings: posix.termios) void {
        tty_fd.store(-1, .release);
        tty_saved = settings;
        tty_fd.store(fd, .release);
    }

    fn releaseTerminal() void {
        tty_fd.store(-1, .release);
    }

    fn guardedTerminal() posix.fd_t {
        return tty_fd.load(.acquire);
    }

    fn abandonNow() void {
        const pid = child.swap(0, .acq_rel);
        if (pid > 0) _ = std.c.kill(pid, .KILL);
        const fd = tty_fd.swap(-1, .acq_rel);
        if (fd >= 0) _ = std.c.tcsetattr(fd, .NOW, &tty_saved);
    }
};

/// Routes INT, TERM and HUP to the flag; the old actions come back with ``restore``.
pub fn install() void {
    if (comptime !windows) Host.install();
}

pub fn restore() void {
    if (comptime !windows) Host.restore();
}

/// The first stop signal received, or null.
pub fn requested() ?posix.SIG {
    const n = stored.load(.acquire);
    return if (n == 0) null else @fromBackingInt(@intCast(n));
}

/// Records ``sig`` as if it arrived, for tests.
pub fn request(sig: posix.SIG) void {
    _ = stored.cmpxchgStrong(0, @intCast(@backingInt(sig)), .acq_rel, .acquire);
}

pub fn reset() void {
    stored.store(0, .release);
}

/// The shell's code for a run ended by ``sig``: 130 for Ctrl-C, 143 for SIGTERM, 129 for SIGHUP.
pub fn exitCode(sig: posix.SIG) u8 {
    return 128 + @as(u8, @intCast(@backingInt(sig)));
}

/// The serve child a second Ctrl-C must kill; ``releaseChild`` once it is reaped.
pub fn guardChild(pid: posix.pid_t) void {
    if (comptime !windows) Host.guardChild(pid);
}

pub fn releaseChild(pid: posix.pid_t) void {
    if (comptime !windows) Host.releaseChild(pid);
}

/// The guarded serve child, or 0.
pub fn guardedChild() posix.pid_t {
    if (comptime windows) return @ptrFromInt(0);
    return Host.guardedChild();
}

/// The terminal settings a second Ctrl-C must put back; they are whole before the release store publishes ``fd``.
pub fn guardTerminal(fd: posix.fd_t, settings: posix.termios) void {
    if (comptime !windows) Host.guardTerminal(fd, settings);
}

pub fn releaseTerminal() void {
    if (comptime !windows) Host.releaseTerminal();
}

/// The guarded terminal, or -1.
pub fn guardedTerminal() posix.fd_t {
    if (comptime windows) return @ptrFromInt(std.math.maxInt(usize));
    return Host.guardedTerminal();
}

/// What a second Ctrl-C does before the process dies: kill the guarded child and restore the guarded terminal.
pub fn abandon() void {
    if (comptime !windows) Host.abandonNow();
}
