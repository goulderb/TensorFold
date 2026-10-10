//! The listening socket: SO_REUSEADDR only (never REUSEPORT, so a second server on the port fails), backlog 128. On
//! Windows the port is held exclusively (SO_EXCLUSIVEADDRUSE) instead, which gives the same refusal.
const std = @import("std");
const net = std.Io.net;
const log = @import("log.zig");
const sock = @import("platform").sock;
const Threaded = std.Io.Threaded;

/// The run has said once that accepts ran out of descriptors; a successful accept arms it again.
var nofile_logged = std.atomic.Value(bool).init(false);

pub const Listener = struct {
    fd: sock.Handle,

    pub fn open(address: net.IpAddress) !Listener {
        var storage: sock.Address = undefined;
        const len = Threaded.addressToPosix(&address, &storage);
        const family: u32 = if (address == .ip4) std.posix.AF.INET else std.posix.AF.INET6;
        const fd = sock.create(family, true) catch return error.SocketFailed;
        errdefer sock.close(fd);
        try sock.setListenReuse(fd);
        try sock.bindAddr(fd, &storage, len);
        try sock.listenOn(fd, 128);
        return .{ .fd = fd };
    }

    pub fn port(l: Listener) u16 {
        return sock.localPort(l.fd);
    }

    pub const Accepted = struct { fd: sock.Handle, peer: net.IpAddress };

    /// The next connection; null once ``stop`` is set or the socket fails.
    pub fn accept(l: Listener, stop: *const std.atomic.Value(bool)) ?Accepted {
        while (true) {
            if (stop.load(.acquire)) return null;
            const ready = sock.poll(l.fd, sock.IN, 200) catch return null;
            if (ready == 0) continue;
            var storage: sock.Address = undefined;
            const fd = sock.acceptConn(l.fd, &storage) catch |e| switch (e) {
                error.Exhausted => {
                    if (firstExhausted()) log.line("out of file descriptors: new connections wait until open ones close; raise the open-file limit (ulimit -n)", .{});
                    std.Io.sleep(log.io(), .fromMilliseconds(10), .awake) catch {};
                    continue;
                },
                error.Transient => {
                    std.Io.sleep(log.io(), .fromMilliseconds(10), .awake) catch {};
                    continue;
                },
                error.Fatal => return null,
            };
            nofile_logged.store(false, .release);
            return .{ .fd = fd, .peer = Threaded.addressFromPosix(&storage) };
        }
    }

    pub fn close(l: Listener) void {
        sock.shutdownBoth(l.fd);
        sock.close(l.fd);
    }
};

/// Whether this failure is the burst's first descriptor exhaustion: one log line per burst.
fn firstExhausted() bool {
    return !nofile_logged.swap(true, .acq_rel);
}

test "the descriptor log fires once per burst" {
    defer nofile_logged.store(false, .release);
    try std.testing.expect(firstExhausted());
    try std.testing.expect(!firstExhausted());
    nofile_logged.store(false, .release); // what a successful accept does
    try std.testing.expect(firstExhausted());
}

/// The peer's address as Python's ``client_address[0]`` prints it.
pub fn peerText(buf: []u8, peer: net.IpAddress) []const u8 {
    switch (peer) {
        .ip4 => |a| return std.fmt.bufPrint(buf, "{d}.{d}.{d}.{d}", .{ a.bytes[0], a.bytes[1], a.bytes[2], a.bytes[3] }) catch "",
        .ip6 => |a| {
            var groups: [8]u16 = undefined;
            for (0..8) |i| groups[i] = std.mem.readInt(u16, a.bytes[2 * i ..][0..2], .big);
            var best_start: usize = 8;
            var best_len: usize = 0;
            var i: usize = 0;
            while (i < 8) {
                if (groups[i] != 0) {
                    i += 1;
                    continue;
                }
                const start = i;
                while (i < 8 and groups[i] == 0) i += 1;
                if (i - start > best_len and i - start > 1) {
                    best_start = start;
                    best_len = i - start;
                }
            }
            var w: std.Io.Writer = .fixed(buf);
            i = 0;
            while (i < 8) {
                if (i == best_start) {
                    w.writeAll(if (i == 0) "::" else ":") catch {};
                    i += best_len;
                    continue;
                }
                w.print("{x}", .{groups[i]}) catch {};
                if (i < 7) w.writeAll(":") catch {};
                i += 1;
            }
            return w.buffered();
        },
    }
}

test "peer text" {
    var buf: [64]u8 = undefined;
    try std.testing.expectEqualStrings("127.0.0.1", peerText(&buf, .{ .ip4 = .loopback(0) }));
    try std.testing.expectEqualStrings("::1", peerText(&buf, .{ .ip6 = .loopback(0) }));
}
