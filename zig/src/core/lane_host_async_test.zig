const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const LaneHost = @import("lane_host.zig").LaneHost;
const fixture = @import("prompt_cache_async_test.zig");

const gpa = std.testing.allocator;
const io = std.testing.io;

const Box = struct {
    mutex: std.Io.Mutex = .init,
    done: ?api.Reason = null,
    tokens: usize = 0,

    fn event(ptr: *anyopaque, _: api.Id, e: *const api.Event) void {
        const b: *Box = @ptrCast(@alignCast(ptr));
        b.mutex.lockUncancelable(io);
        defer b.mutex.unlock(io);
        switch (e.*) {
            .tokens => |t| b.tokens += t.len,
            .finished => |f| b.done = f.reason,
            else => {},
        }
    }

    fn result(b: *Box) ?api.Reason {
        b.mutex.lockUncancelable(io);
        defer b.mutex.unlock(io);
        return b.done;
    }

    fn wait(b: *Box) !api.Reason {
        for (0..5000) |_| {
            if (b.result()) |r| return r;
            try std.Io.sleep(io, .fromMilliseconds(1), .awake);
        }
        return error.TestTimedOut;
    }
};

test "a paused disk restore leaves another lane decoding and an unstarted cancellation releases the reader" {
    const x = try fixture.Fixture.init();
    defer x.deinit();
    x.family.hold.store(true, .release);
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, io, &core, .{ .lanes = 2 });
    host.cache = &x.store;
    try host.start();
    defer host.stop();
    const engine = host.engine();
    var cached: Box = .{};
    var independent: Box = .{};
    const waiting: api.Request = .{ .prompt = &fixture.tokens, .max_tokens = 8, .history_len = 450 };
    const plain: api.Request = .{ .prompt = &.{ 2, 3, 4 }, .max_tokens = 8, .untaught = true };
    try engine.submit(1, &waiting, .{ .ctx = &cached, .event = Box.event });
    try engine.submit(2, &plain, .{ .ctx = &independent, .event = Box.event });
    try std.testing.expectEqual(api.Reason.length, try independent.wait());
    try std.testing.expectEqual(@as(usize, 8), independent.tokens);
    try std.testing.expect(cached.result() == null);
    engine.cancel(1);
    try std.testing.expectEqual(api.Reason.cancelled, try cached.wait());
    try std.testing.expectEqual(@as(u64, 0), x.store.pending_bytes);
    try std.testing.expectEqual(@as(u32, 1), x.family.cancelled);
}
