//! Upstream custom proposers are request-owned across completion, cancellation and open failures.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const a = std.testing.allocator;

const Box = struct {
    reason: api.Reason = .failed,
    done: std.atomic.Value(bool) = .init(false),
    fn event(ctx: *anyopaque, _: api.Id, ev: *const api.Event) void {
        const b: *Box = @ptrCast(@alignCast(ctx));
        if (ev.* == .finished) {
            b.reason = ev.finished.reason;
            b.done.store(true, .release);
        }
    }
    fn wait(b: *Box) !api.Reason {
        const began = std.Io.Clock.awake.now(std.testing.io).toNanoseconds();
        while (!b.done.load(.acquire)) {
            if (std.Io.Clock.awake.now(std.testing.io).toNanoseconds() - began > 5 * std.time.ns_per_s) return error.RequestTimedOut;
            std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
        }
        return b.reason;
    }
};
const Factory = struct {
    host: *api.LaneHost,
    made: usize = 0,
    freed: usize = 0,
    fail: bool = false,
    const Copy = struct {
        owner: *Factory,
        gpa: std.mem.Allocator,
        fn propose(_: *anyopaque, _: []const u32, _: i64) ![]const u32 {
            return &.{};
        }
        fn observe(_: *anyopaque, _: i64, _: i64) void {}
        fn lastMatch(_: *anyopaque) i64 {
            return 0;
        }
    };
    fn make(ctx: *anyopaque, gpa: std.mem.Allocator, _: []const u32, _: []const u32) !lanes.proposer.Proposer {
        const f: *Factory = @ptrCast(@alignCast(ctx));
        if (f.fail) return error.OutOfMemory;
        const c = try gpa.create(Copy);
        c.* = .{ .owner = f, .gpa = gpa };
        f.made += 1;
        return .{ .ptr = c, .vtable = &.{ .propose = Copy.propose, .observe = Copy.observe, .last_match = Copy.lastMatch, .priced = true } };
    }
    fn free(_: *anyopaque, p: lanes.proposer.Proposer) void {
        const c: *Copy = @ptrCast(@alignCast(p.ptr));
        c.owner.freed += 1;
        c.gpa.destroy(c);
    }
    fn cancel(ctx: *anyopaque, _: *lanes.Stream, chunk: usize) void {
        const f: *Factory = @ptrCast(@alignCast(ctx));
        if (chunk == 2) f.host.engine().cancel(2);
    }
};

test "custom proposer cleanup covers normal reply, cancelled prefill, grammar failure and factory failure" {
    const Failure = enum { none, cancel, grammar, factory };
    for ([_]Failure{ .none, .cancel, .grammar, .factory }) |failure| {
        var cfg = try lanes.Config.init(a, .{ .exact_width = 16 }, 16, 7);
        defer cfg.deinit(a);
        var target: lanes.fake.Fake = .{ .gpa = a, .prefill_chunks = 4 };
        defer target.deinit();
        var clock: lanes.fake.FixedClock = .{};
        var core = lanes.Engine.init(a, &cfg, target.stepped(), clock.clock());
        defer core.deinit();
        var host = api.LaneHost.init(a, std.testing.io, &core, .{ .lanes = 1 });
        var factory: Factory = .{ .host = &host };
        host.proposers = .{ .ctx = &factory, .make = Factory.make, .free = Factory.free };
        try host.start();
        var running = true;
        defer if (running) host.stop();
        const first: api.Request = .{ .prompt = &.{ 1, 2, 3, 4, 5 }, .max_tokens = 4, .drafts = false };
        var first_box: Box = .{};
        try host.engine().submit(1, &first, .{ .ctx = &first_box, .event = Box.event });
        try std.testing.expectEqual(api.Reason.length, try first_box.wait());
        factory.fail = failure == .factory;
        if (failure == .cancel) {
            target.prefill_hook = Factory.cancel;
            target.prefill_hook_ctx = &factory;
        }
        var second = first;
        if (failure == .grammar) second.structure = .{ .kind = .json }; // no compiler: fails after make
        var second_box: Box = .{};
        try host.engine().submit(2, &second, .{ .ctx = &second_box, .event = Box.event });
        const expected: api.Reason = switch (failure) {
            .none => .length,
            .cancel => .cancelled,
            .grammar, .factory => .failed,
        };
        try std.testing.expectEqual(expected, try second_box.wait());
        host.stop();
        running = false;
        try std.testing.expectEqual(@as(usize, if (failure == .factory) 1 else 2), factory.made);
        try std.testing.expectEqual(factory.made, factory.freed);
        try std.testing.expectEqual(@as(usize, 0), target.lanes.count());
    }
}
