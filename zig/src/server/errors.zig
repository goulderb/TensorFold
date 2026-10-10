//! A request's refusal: Python's RequestError family, or another exception's bare message.
const std = @import("std");
const Allocator = std.mem.Allocator;

pub const Kind = enum {
    /// RequestError: 400, ``{"message", "type": "invalid_request_error"}``.
    request,
    /// ContextLengthError: as request, plus ``param`` and ``code: context_length_exceeded``.
    context_length,
    /// CapacityError: 503, retry shortly.
    capacity,
    /// Every lane and waiting place is taken (TENSORFOLD_MAX_QUEUED): 429 with Retry-After on every route.
    admission,
    /// Any other exception (ValueError, KeyError...): 400 with ``{"message"}`` only.
    other,
    /// An exception while the reply runs: 500 with ``{"message"}``, or a stream's server_error event.
    server,
};

pub const context_limit = "This server's maximum context length is"; // OpenAI's wording, which clients match to compact

pub const Refused = error{ Refused, OutOfMemory };

/// The request's allocator and the refusal it ended with.
pub const Cx = struct {
    a: Allocator,
    kind: Kind = .request,
    message: []const u8 = "",

    pub fn fail(cx: *Cx, kind: Kind, comptime fmt: []const u8, args: anytype) Refused {
        cx.kind = kind;
        cx.message = try std.fmt.allocPrint(cx.a, fmt, args);
        return error.Refused;
    }

    pub fn refuse(cx: *Cx, message: []const u8) Refused {
        cx.kind = .request;
        cx.message = message;
        return error.Refused;
    }

    pub fn other(cx: *Cx, message: []const u8) Refused {
        cx.kind = .other;
        cx.message = message;
        return error.Refused;
    }

    pub fn status(cx: *const Cx) u16 {
        return switch (cx.kind) {
            .capacity => 503,
            .admission => 429,
            .server => 500,
            else => 400,
        };
    }
};

test "an admission refusal answers 429; a busy engine keeps its 503" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    var full: Cx = .{ .a = arena.allocator() };
    const refused: anyerror = full.fail(.admission, "every lane and waiting place is taken; retry shortly", .{});
    try std.testing.expectEqual(@as(anyerror, error.Refused), refused);
    try std.testing.expectEqual(@as(u16, 429), full.status());
    var busy: Cx = .{ .a = arena.allocator() };
    const busy_refused: anyerror = busy.fail(.capacity, "the engine is busy; retry shortly", .{});
    try std.testing.expectEqual(@as(anyerror, error.Refused), busy_refused);
    try std.testing.expectEqual(@as(u16, 503), busy.status());
}
