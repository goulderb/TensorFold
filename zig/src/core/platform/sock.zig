//! The one TCP socket layer the server and the Flash Next control link share. Winsock on Windows (WSAPoll, recv, send,
//! closesocket) and the POSIX calls elsewhere, so Linux makes the same syscalls it made before. Handles are created
//! non-inheritable on Windows, so a spawned rank cannot inherit them. Addresses use std's sockaddr union on both systems.
const std = @import("std");
const builtin = @import("builtin");
const posix = std.posix;
const Threaded = std.Io.Threaded;

const windows = builtin.os.tag == .windows;

pub const Handle = posix.socket_t;
/// The handle of a connection with no socket (tests drive the handlers with one).
pub const none: Handle = if (windows) @ptrFromInt(std.math.maxInt(usize)) else -1;
pub const Address = Threaded.PosixAddress;

/// poll events, with the values of the system's own poll interface (WSAPOLLFD's on Windows).
pub const IN: i16 = if (windows) 0x0300 else posix.POLL.IN;
pub const OUT: i16 = if (windows) 0x0010 else posix.POLL.OUT;
// WSAPoll does not take POLLPRI (it fails the call), so Windows asks for nothing out of band
pub const PRI: i16 = if (windows) 0 else posix.POLL.PRI;
/// poll's error bits, and the peer's write-side close (Linux's POLLRDHUP, which Windows has no bit for)
pub const ERR: i16 = if (windows) 0x0001 else posix.POLL.ERR;
pub const HUP: i16 = if (windows) 0x0002 else posix.POLL.HUP;
pub const NVAL: i16 = if (windows) 0x0004 else posix.POLL.NVAL;
pub const RDHUP: i16 = if (windows) 0 else 0x2000;

const WSAPOLLFD = extern struct { fd: usize, events: i16, revents: i16 };
const SOCKET_ERROR: c_int = -1;
const WSAEINTR: c_int = 10004;
const WSAEWOULDBLOCK: c_int = 10035;
const WSAEADDRINUSE: c_int = 10048;
const WSAEACCES: c_int = 10013;
const WSAEMFILE: c_int = 10024;
const WSAENOBUFS: c_int = 10055;
const WSAECONNABORTED: c_int = 10053;
const WSAECONNRESET: c_int = 10054;
const WSA_FLAG_OVERLAPPED: u32 = 0x01;
const WSA_FLAG_NO_HANDLE_INHERIT: u32 = 0x80;
const SD_SEND: c_int = 1;
const SD_BOTH: c_int = 2;
const SOCK_STREAM: c_int = 1;
const MSG_PEEK: c_int = 0x2;

extern "ws2_32" fn WSAStartup(version: u16, data: *anyopaque) callconv(.winapi) c_int;
extern "ws2_32" fn WSASocketW(af: c_int, kind: c_int, protocol: c_int, info: ?*anyopaque, group: u32, flags: u32) callconv(.winapi) usize;
extern "ws2_32" fn WSAPoll(fds: *WSAPOLLFD, count: u32, timeout: c_int) callconv(.winapi) c_int;
extern "ws2_32" fn WSAGetLastError() callconv(.winapi) c_int;
extern "ws2_32" fn closesocket(s: usize) callconv(.winapi) c_int;
extern "ws2_32" fn recv(s: usize, buf: [*]u8, len: c_int, flags: c_int) callconv(.winapi) c_int;
extern "ws2_32" fn send(s: usize, buf: [*]const u8, len: c_int, flags: c_int) callconv(.winapi) c_int;
extern "ws2_32" fn shutdown(s: usize, how: c_int) callconv(.winapi) c_int;
extern "ws2_32" fn setsockopt(s: usize, level: c_int, name: c_int, val: *const anyopaque, len: c_int) callconv(.winapi) c_int;
extern "ws2_32" fn getsockopt(s: usize, level: c_int, name: c_int, val: *anyopaque, len: *c_int) callconv(.winapi) c_int;
extern "ws2_32" fn WSAIoctl(s: usize, code: u32, in: *const anyopaque, in_len: u32, out: ?*anyopaque, out_len: u32, returned: *u32, overlapped: ?*anyopaque, routine: ?*anyopaque) callconv(.winapi) c_int;
extern "ws2_32" fn bind(s: usize, name: *const anyopaque, len: c_int) callconv(.winapi) c_int;
extern "ws2_32" fn listen(s: usize, backlog: c_int) callconv(.winapi) c_int;
extern "ws2_32" fn connect(s: usize, name: *const anyopaque, len: c_int) callconv(.winapi) c_int;
extern "ws2_32" fn accept(s: usize, addr: ?*anyopaque, len: ?*c_int) callconv(.winapi) usize;
extern "ws2_32" fn getsockname(s: usize, name: *anyopaque, len: *c_int) callconv(.winapi) c_int;

inline fn sock(h: Handle) usize {
    return @intFromPtr(h);
}

inline fn fromSock(s: usize) Handle {
    return @ptrFromInt(s);
}

/// Winsock starts once per process; a second start would only add a reference, so the flag is the whole guard.
var started = std.atomic.Value(bool).init(false);

fn startWinsock() error{SocketFailed}!void {
    if (started.load(.acquire)) return;
    var data: [512]u8 align(8) = undefined;
    if (WSAStartup(0x0202, &data) != 0) return error.SocketFailed;
    started.store(true, .release);
}

/// A new TCP socket of `family`. `cloexec` only matters on POSIX (Windows handles are always non-inheritable here).
pub fn create(family: u32, cloexec: bool) error{SocketFailed}!Handle {
    if (windows) {
        try startWinsock();
        const s = WSASocketW(@intCast(family), SOCK_STREAM, 0, null, 0, WSA_FLAG_OVERLAPPED | WSA_FLAG_NO_HANDLE_INHERIT);
        if (s == ~@as(usize, 0)) return error.SocketFailed;
        return fromSock(s);
    }
    const kind: u32 = posix.SOCK.STREAM | if (cloexec) @as(u32, posix.SOCK.CLOEXEC) else 0;
    const rc = posix.system.socket(family, kind, 0);
    if (posix.errno(rc) != .SUCCESS) return error.SocketFailed;
    return @intCast(rc);
}

pub fn close(h: Handle) void {
    if (windows) {
        _ = closesocket(sock(h));
    } else {
        _ = posix.system.close(h);
    }
}

/// The write side off: the peer reads end of file, and this side can still read.
pub fn shutdownWrite(h: Handle) void {
    if (windows) {
        _ = shutdown(sock(h), SD_SEND);
    } else {
        _ = posix.system.shutdown(h, posix.SHUT.WR);
    }
}

/// Both directions off: a blocked peer call on this socket returns.
pub fn shutdownBoth(h: Handle) void {
    if (windows) {
        _ = shutdown(sock(h), SD_BOTH);
    } else {
        _ = posix.system.shutdown(h, posix.SHUT.RDWR);
    }
}

pub const Option = enum { reuse_addr, exclusive_addr, nodelay };

/// Sets one integer option. `exclusive_addr` is Windows' listener rule (SO_EXCLUSIVEADDRUSE): a second bind to the port
/// fails there, where SO_REUSEADDR would let it succeed. POSIX has no such option and never asks for it.
pub fn setOption(h: Handle, which: Option, on: bool) error{OptionFailed}!void {
    var v: c_int = @intFromBool(on);
    if (windows) {
        const level: c_int = switch (which) {
            .reuse_addr, .exclusive_addr => 0xFFFF,
            .nodelay => 6,
        };
        const name: c_int = switch (which) {
            .reuse_addr => 0x0004,
            .exclusive_addr => ~@as(c_int, 0x0004),
            .nodelay => 1,
        };
        if (setsockopt(sock(h), level, name, &v, @sizeOf(c_int)) == SOCKET_ERROR) return error.OptionFailed;
        return;
    }
    switch (which) {
        .reuse_addr => posix.setsockopt(h, posix.SOL.SOCKET, posix.SO.REUSEADDR, std.mem.asBytes(&v)) catch return error.OptionFailed,
        .nodelay => posix.setsockopt(h, posix.IPPROTO.TCP, posix.TCP.NODELAY, std.mem.asBytes(&v)) catch return error.OptionFailed,
        .exclusive_addr => return error.OptionFailed,
    }
}

/// The listener's address-reuse option: exclusive on Windows, SO_REUSEADDR elsewhere.
pub fn setListenReuse(h: Handle) error{OptionFailed}!void {
    return setOption(h, if (windows) .exclusive_addr else .reuse_addr, true);
}

pub const BindError = error{ AddressInUse, AccessDenied, BindFailed };

pub fn bindAddr(h: Handle, addr: *const Address, len: u32) BindError!void {
    if (windows) {
        if (bind(sock(h), addr, @intCast(len)) == SOCKET_ERROR) return switch (WSAGetLastError()) {
            WSAEADDRINUSE => error.AddressInUse,
            WSAEACCES => error.AccessDenied,
            else => error.BindFailed,
        };
        return;
    }
    switch (posix.errno(posix.system.bind(h, &addr.any, len))) {
        .SUCCESS => {},
        .ADDRINUSE => return error.AddressInUse,
        .ACCES => return error.AccessDenied,
        else => return error.BindFailed,
    }
}

pub fn listenOn(h: Handle, backlog: u31) error{ListenFailed}!void {
    if (windows) {
        if (listen(sock(h), @intCast(backlog)) == SOCKET_ERROR) return error.ListenFailed;
        return;
    }
    if (posix.errno(posix.system.listen(h, backlog)) != .SUCCESS) return error.ListenFailed;
}

pub fn connectAddr(h: Handle, addr: *const Address, len: u32) error{ConnectFailed}!void {
    if (windows) {
        if (connect(sock(h), addr, @intCast(len)) == SOCKET_ERROR) return error.ConnectFailed;
        return;
    }
    if (posix.errno(posix.system.connect(h, &addr.any, len)) != .SUCCESS) return error.ConnectFailed;
}

/// The socket's bound local port.
pub fn localPort(h: Handle) u16 {
    var storage: Address = undefined;
    if (windows) {
        var len: c_int = @sizeOf(Address);
        if (getsockname(sock(h), &storage, &len) == SOCKET_ERROR) return 0;
    } else {
        var len: posix.socklen_t = @sizeOf(Address);
        if (posix.errno(posix.system.getsockname(h, &storage.any, &len)) != .SUCCESS) return 0;
    }
    return Threaded.addressFromPosix(&storage).getPort();
}

pub const AcceptError = error{
    /// Out of descriptors (EMFILE, ENFILE): the caller logs once per burst and retries.
    Exhausted,
    /// The connection went away or the call was interrupted: retry.
    Transient,
    Fatal,
};

/// The next connection; the peer's address lands in `storage`.
pub fn acceptConn(h: Handle, storage: *Address) AcceptError!Handle {
    if (windows) {
        var len: c_int = @sizeOf(Address);
        const s = accept(sock(h), storage, &len);
        if (s != ~@as(usize, 0)) return fromSock(s);
        return switch (WSAGetLastError()) {
            WSAEMFILE => error.Exhausted,
            WSAENOBUFS, WSAEINTR, WSAEWOULDBLOCK, WSAECONNABORTED, WSAECONNRESET => error.Transient,
            else => error.Fatal,
        };
    }
    var len: posix.socklen_t = @sizeOf(Address);
    const rc = posix.system.accept(h, &storage.any, &len);
    switch (posix.errno(rc)) {
        .SUCCESS => return @intCast(rc),
        .INTR, .AGAIN, .CONNABORTED, .NOBUFS, .NOMEM => return error.Transient,
        .MFILE, .NFILE => return error.Exhausted,
        else => return error.Fatal,
    }
}

/// Waits up to `timeout_ms` for `events` and returns the revents (0 on timeout). A poll that fails is an error.
pub fn poll(h: Handle, events: i16, timeout_ms: i32) error{PollFailed}!i16 {
    // the no-socket value polls as nothing ready, as a negative descriptor does on POSIX
    if (h == none) return 0;
    if (windows) {
        var fd: WSAPOLLFD = .{ .fd = sock(h), .events = events, .revents = 0 };
        const rc = WSAPoll(&fd, 1, timeout_ms);
        if (rc == SOCKET_ERROR) return error.PollFailed;
        return if (rc == 0) 0 else fd.revents;
    }
    var fds = [_]posix.pollfd{.{ .fd = h, .events = events, .revents = 0 }};
    const n = posix.poll(&fds, timeout_ms) catch return error.PollFailed;
    return if (n == 0) 0 else fds[0].revents;
}

/// Reads what is there (at least one byte, unless `buf` is empty). 0 means the peer closed its side.
pub fn read(h: Handle, buf: []u8) error{ReadFailed}!usize {
    if (windows) {
        if (buf.len == 0) return 0;
        const n = recv(sock(h), buf.ptr, @intCast(@min(buf.len, std.math.maxInt(c_int))), 0);
        if (n < 0) return error.ReadFailed;
        return @intCast(n);
    }
    return posix.read(h, buf) catch error.ReadFailed;
}

pub const WriteError = error{ WouldBlock, WriteFailed };

/// Writes some of `bytes` and returns how many went. A full buffer on a blocking socket is WouldBlock; the caller polls.
pub fn write(h: Handle, bytes: []const u8) WriteError!usize {
    if (windows) {
        const n = send(sock(h), bytes.ptr, @intCast(@min(bytes.len, std.math.maxInt(c_int))), 0);
        if (n >= 0) return @intCast(n);
        return if (WSAGetLastError() == WSAEWOULDBLOCK) error.WouldBlock else error.WriteFailed;
    }
    while (true) {
        const rc = posix.system.write(h, bytes.ptr, bytes.len);
        switch (posix.errno(rc)) {
            .SUCCESS => return @intCast(rc),
            .INTR => continue,
            .AGAIN => return error.WouldBlock,
            else => return error.WriteFailed,
        }
    }
}

/// The next bytes without consuming them. Call it only after `poll` reported readable: it then does not wait.
/// WouldBlock means nothing was there.
pub fn peek(h: Handle, buf: []u8) error{ ReadFailed, WouldBlock }!usize {
    if (windows) {
        const n = recv(sock(h), buf.ptr, @intCast(@min(buf.len, std.math.maxInt(c_int))), MSG_PEEK);
        if (n >= 0) return @intCast(n);
        return if (WSAGetLastError() == WSAEWOULDBLOCK) error.WouldBlock else error.ReadFailed;
    }
    const rc = posix.system.recvfrom(h, buf.ptr, buf.len, posix.MSG.PEEK | posix.MSG.DONTWAIT, null, null);
    return switch (posix.errno(rc)) {
        .SUCCESS => @intCast(rc),
        .AGAIN, .INTR => error.WouldBlock,
        else => error.ReadFailed,
    };
}

/// Closes with a reset instead of an end of file: the linger is on with a zero timeout, so the close drops what the
/// peer has not read and sends RST. The caller still closes the handle.
pub fn abort(h: Handle) void {
    if (windows) {
        const linger = extern struct { onoff: u16, linger: u16 }{ .onoff = 1, .linger = 0 };
        _ = setsockopt(sock(h), SOL_SOCKET_WIN, SO_LINGER_WIN, &linger, @sizeOf(@TypeOf(linger)));
        return;
    }
    const linger = extern struct { onoff: c_int, linger: c_int }{ .onoff = 1, .linger = 0 };
    posix.setsockopt(h, posix.SOL.SOCKET, posix.SO.LINGER, std.mem.asBytes(&linger)) catch {};
}

const SOL_SOCKET_WIN: c_int = 0xFFFF;
const SO_KEEPALIVE_WIN: c_int = 0x0008;
const SO_LINGER_WIN: c_int = 0x0080;
const IPPROTO_TCP_WIN: c_int = 6;
const TCP_KEEPIDLE_WIN: c_int = 3;
const TCP_KEEPCNT_WIN: c_int = 16;
const TCP_KEEPINTVL_WIN: c_int = 17;
const SIO_KEEPALIVE_VALS: u32 = 0x98000004;

/// The keepalive that judges a silent peer: SO_KEEPALIVE on, the first probe after `idle_s` seconds of silence, one
/// every `interval_s`, and the link dead after `probes` unanswered ones. Windows takes the first two in one call
/// (SIO_KEEPALIVE_VALS, milliseconds) and the probe count with TCP_KEEPCNT (Windows 10 1703 and later).
pub const Keepalive = struct { on: bool, idle_s: c_int, interval_s: c_int, probes: c_int };

pub fn setKeepalive(h: Handle, idle_s: c_int, interval_s: c_int, probes: c_int) error{OptionFailed}!void {
    if (windows) {
        const vals = [3]c_ulong{ 1, @intCast(idle_s * 1000), @intCast(interval_s * 1000) };
        var returned: u32 = 0;
        if (WSAIoctl(sock(h), SIO_KEEPALIVE_VALS, &vals, @sizeOf(@TypeOf(vals)), null, 0, &returned, null, null) == SOCKET_ERROR) return error.OptionFailed;
        if (setsockopt(sock(h), IPPROTO_TCP_WIN, TCP_KEEPCNT_WIN, &probes, @sizeOf(c_int)) == SOCKET_ERROR) return error.OptionFailed;
        return;
    }
    const on: c_int = 1;
    posix.setsockopt(h, posix.SOL.SOCKET, posix.SO.KEEPALIVE, std.mem.asBytes(&on)) catch return error.OptionFailed;
    posix.setsockopt(h, posix.IPPROTO.TCP, posix.TCP.KEEPIDLE, std.mem.asBytes(&idle_s)) catch return error.OptionFailed;
    posix.setsockopt(h, posix.IPPROTO.TCP, posix.TCP.KEEPINTVL, std.mem.asBytes(&interval_s)) catch return error.OptionFailed;
    posix.setsockopt(h, posix.IPPROTO.TCP, posix.TCP.KEEPCNT, std.mem.asBytes(&probes)) catch return error.OptionFailed;
}

/// What the socket reads back for its keepalive; null when a read fails. The test checks the bound this way.
pub fn keepaliveSettings(h: Handle) ?Keepalive {
    if (windows) return .{
        .on = (winOpt(h, SOL_SOCKET_WIN, SO_KEEPALIVE_WIN) orelse return null) != 0,
        .idle_s = winOpt(h, IPPROTO_TCP_WIN, TCP_KEEPIDLE_WIN) orelse return null,
        .interval_s = winOpt(h, IPPROTO_TCP_WIN, TCP_KEEPINTVL_WIN) orelse return null,
        .probes = winOpt(h, IPPROTO_TCP_WIN, TCP_KEEPCNT_WIN) orelse return null,
    };
    return .{
        .on = (posixOpt(h, posix.SOL.SOCKET, posix.SO.KEEPALIVE) orelse return null) != 0,
        .idle_s = posixOpt(h, posix.IPPROTO.TCP, posix.TCP.KEEPIDLE) orelse return null,
        .interval_s = posixOpt(h, posix.IPPROTO.TCP, posix.TCP.KEEPINTVL) orelse return null,
        .probes = posixOpt(h, posix.IPPROTO.TCP, posix.TCP.KEEPCNT) orelse return null,
    };
}

fn winOpt(h: Handle, level: c_int, name: c_int) ?c_int {
    var v: c_int = 0;
    var len: c_int = @sizeOf(c_int);
    if (getsockopt(sock(h), level, name, &v, &len) == SOCKET_ERROR) return null;
    return v;
}

fn posixOpt(h: Handle, level: i32, name: u32) ?c_int {
    var v: c_int = 0;
    var len: posix.socklen_t = @sizeOf(c_int);
    if (posix.errno(posix.system.getsockopt(h, level, name, @ptrCast(&v), &len)) != .SUCCESS) return null;
    return v;
}

test "a listener, a client, the handshake, bytes and a half-close" {
    const listener = try create(posix.AF.INET, true);
    defer close(listener);
    try setListenReuse(listener);
    var bound: Address = undefined;
    const ip = try std.Io.net.IpAddress.parse("127.0.0.1", 0);
    const len = Threaded.addressToPosix(&ip, &bound);
    try bindAddr(listener, &bound, len);
    try listenOn(listener, 4);
    const port = localPort(listener);
    try std.testing.expect(port != 0);

    const client = try create(posix.AF.INET, true);
    defer close(client);
    var target: Address = undefined;
    const to = try std.Io.net.IpAddress.parse("127.0.0.1", port);
    const tlen = Threaded.addressToPosix(&to, &target);
    try connectAddr(client, &target, tlen);
    try setOption(client, .nodelay, true);

    try std.testing.expect(try poll(listener, IN, 1000) != 0);
    var peer: Address = undefined;
    const accepted = try acceptConn(listener, &peer);
    defer close(accepted);

    try std.testing.expectEqual(@as(usize, 5), try write(client, "hello"));
    try std.testing.expect(try poll(accepted, IN, 1000) != 0);
    var got: [16]u8 = undefined;
    try std.testing.expectEqualStrings("hello", got[0..try read(accepted, &got)]);
    try std.testing.expectEqual(@as(i16, 0), try poll(accepted, IN, 0));

    shutdownBoth(client);
    try std.testing.expect(try poll(accepted, IN, 1000) != 0);
    try std.testing.expectEqual(@as(usize, 0), try peek(accepted, &got));
}
