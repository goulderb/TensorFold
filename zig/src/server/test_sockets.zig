const std = @import("std");
const sock = @import("sys").sock;

pub fn pair() ![2]sock.Handle {
    const listener = try sock.create(std.posix.AF.INET, true);
    defer sock.close(listener);
    try sock.setListenReuse(listener);
    var address: sock.Address = undefined;
    const local: std.Io.net.IpAddress = .{ .ip4 = .loopback(0) };
    const len = std.Io.Threaded.addressToPosix(&local, &address);
    try sock.bindAddr(listener, &address, len);
    try sock.listenOn(listener, 1);
    const client = try sock.create(std.posix.AF.INET, true);
    errdefer sock.close(client);
    const target: std.Io.net.IpAddress = .{ .ip4 = .loopback(sock.localPort(listener)) };
    const target_len = std.Io.Threaded.addressToPosix(&target, &address);
    try sock.connectAddr(client, &address, target_len);
    _ = try sock.poll(listener, sock.IN, 1000);
    const accepted = try sock.acceptConn(listener, &address);
    errdefer sock.close(accepted);
    try sock.setOption(client, .nodelay, true);
    try sock.setOption(accepted, .nodelay, true);
    return .{ client, accepted };
}
