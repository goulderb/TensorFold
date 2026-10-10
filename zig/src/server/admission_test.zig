//! A full admission answers 429 with Retry-After: 5 on the wire; a busy engine keeps its 503 without it.
const std = @import("std");
const api = @import("engine_api");
const server = @import("root.zig");
const json = server.json;
const model_text = @import("model_text.zig");
const routes = @import("routes.zig");
const Conn = @import("http_conn.zig").Conn;

/// Bytes as tokens; every prompt renders as "prompt".
const Text = struct {
    fn text(t: *@This()) model_text.Text {
        return .{ .ctx = t, .vtable = &.{ .encode = encode, .decode = decode, .token_id = tokenId, .token_string = tokenString, .vocab_size = vocabSize, .eos_ids = eosIds, .render = render, .template_source = templateSource } };
    }

    fn encode(_: *anyopaque, a: std.mem.Allocator, input: []const u8, _: bool) model_text.Error![]u32 {
        const ids = try a.alloc(u32, input.len);
        for (input, ids) |byte, *id| id.* = byte;
        return ids;
    }

    fn decode(_: *anyopaque, a: std.mem.Allocator, ids: []const u32) std.mem.Allocator.Error![]u8 {
        const decoded = try a.alloc(u8, ids.len);
        for (ids, decoded) |id, *byte| byte.* = @intCast(id);
        return decoded;
    }

    fn tokenId(_: *anyopaque, _: []const u8) ?u32 {
        return null;
    }

    fn tokenString(_: *anyopaque, a: std.mem.Allocator, id: u32) std.mem.Allocator.Error![]u8 {
        return std.fmt.allocPrint(a, "{d}", .{id});
    }

    fn vocabSize(_: *anyopaque) u32 {
        return 256;
    }

    fn eosIds(_: *anyopaque) []const u32 {
        return &.{};
    }

    fn render(_: *anyopaque, a: std.mem.Allocator, _: json.Value, _: model_text.RenderOptions, _: *[]const u8) model_text.Error![]u8 {
        return a.dupe(u8, "prompt");
    }

    fn templateSource(_: *anyopaque) []const u8 {
        return "";
    }
};

/// An engine that refuses every submit with ``refusal`` and never serves.
const Refusing = struct {
    refusal: api.SubmitError,

    fn engine(e: *@This()) api.Engine {
        return .{ .ctx = e, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }

    fn info(_: *anyopaque) api.Info {
        return .{ .context_window = 4096 };
    }

    fn submit(ctx: *anyopaque, _: api.Id, _: *const api.Request, _: api.Sink) api.SubmitError!void {
        const e: *@This() = @ptrCast(@alignCast(ctx));
        return e.refusal;
    }

    fn cancel(_: *anyopaque, _: api.Id) void {}

    fn status(_: *anyopaque, out: *api.Status, _: []u32) void {
        out.* = .{};
    }

    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }
};

const Reply = struct { status: u16, retry_after: bool };

/// One request over a socket pair through the routes, as the listener runs it: the status and whether it sent Retry-After.
fn exchange(srv: *server.Server, path: []const u8, body: []const u8) !Reply {
    var pair: [2]std.c.fd_t = undefined;
    if (std.c.socketpair(std.posix.AF.UNIX, std.posix.SOCK.STREAM, 0, &pair) != 0) return error.SocketPair;
    defer _ = std.c.close(pair[0]);
    defer _ = std.c.close(pair[1]);
    const head = try std.fmt.allocPrint(std.testing.allocator, "POST {s} HTTP/1.1\r\nHost: t\r\nContent-Length: {d}\r\n\r\n", .{ path, body.len });
    defer std.testing.allocator.free(head);
    try writeAll(pair[0], head);
    try writeAll(pair[0], body);
    var conn = try Conn.init(std.testing.allocator, pair[1], "test");
    defer conn.deinit();
    conn.timeouts = .{ .idle_ms = 2000, .read_ms = 2000, .write_ms = 2000 };
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    if (try conn.readRequest(a, false) != .ready) return error.BadRequest;
    routes.dispatch(srv, &conn, a);
    const raw = try readReady(std.testing.allocator, pair[0]);
    defer std.testing.allocator.free(raw);
    const split = std.mem.indexOf(u8, raw, "\r\n\r\n") orelse return error.BadReply;
    return .{
        .status = std.fmt.parseInt(u16, raw["HTTP/1.1 ".len..][0..3], 10) catch return error.BadReply,
        .retry_after = std.mem.indexOf(u8, raw[0..split], "\r\nRetry-After: 5") != null,
    };
}

fn writeAll(fd: std.c.fd_t, data: []const u8) !void {
    var sent: usize = 0;
    while (sent < data.len) {
        const n = std.c.write(fd, data[sent..].ptr, data.len - sent);
        if (n <= 0) return error.Closed;
        sent += @intCast(n);
    }
}

fn readReady(a: std.mem.Allocator, fd: std.c.fd_t) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(a);
    var tmp: [4096]u8 = undefined;
    while (true) {
        var pollfd = [_]std.posix.pollfd{.{ .fd = fd, .events = std.posix.POLL.IN, .revents = 0 }};
        if (try std.posix.poll(&pollfd, 200) == 0) break;
        const n = std.posix.read(fd, &tmp) catch |e| return e;
        if (n == 0) break;
        try out.appendSlice(a, tmp[0..n]);
    }
    return out.toOwnedSlice(a);
}

const chat_body = "{\"model\":\"m\",\"max_tokens\":4,\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}";

/// A chat request against an engine that refuses its submit with ``refusal``.
fn chatRefused(refusal: api.SubmitError) !Reply {
    var backend: Refusing = .{ .refusal = refusal };
    var text: Text = .{};
    var srv = try server.Server.init(std.testing.allocator, std.testing.io, backend.engine(), text.text(), .{
        .served_name = "m",
        .model_ids = &.{"m"},
        .enable_thinking = false,
        .use_drafts = false,
    }, null);
    defer srv.deinit();
    return exchange(srv, "/v1/chat/completions", chat_body);
}

test "a full admission answers 429 with Retry-After 5 on chat" {
    const full = try chatRefused(error.Full);
    try std.testing.expectEqual(@as(u16, 429), full.status);
    try std.testing.expect(full.retry_after);
}

test "a busy engine keeps its 503 and sends no Retry-After" {
    const busy = try chatRefused(error.Busy);
    try std.testing.expectEqual(@as(u16, 503), busy.status);
    try std.testing.expect(!busy.retry_after);
}
