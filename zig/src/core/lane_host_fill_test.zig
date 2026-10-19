//! The order a lane host fills waiting prompts in: foreground first, then the shortest (lane_host_fill.zig).
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const LaneHost = @import("lane_host.zig").LaneHost;
const Reason = api.Reason;
const Id = api.Id;
const Event = api.Event;
const Request = api.Request;
const Engine = api.Engine;

test "batched host prefill preserves priority, delivery and per-job memory requeue" {
    for ([_]?usize{ null, 1 }) |limit| {
        const gpa = std.testing.allocator;
        var cfg = try lanes.Config.init(gpa, .{ .exact_width = 1, .gpu_tokens = false }, 1, 0);
        defer cfg.deinit(gpa);
        var target: lanes.fake.Fake = .{ .gpa = gpa, .memory_lanes = limit };
        defer target.deinit();
        var clock: lanes.fake.FixedClock = .{};
        var core = lanes.Engine.init(gpa, &cfg, target.coldBatched(), clock.clock());
        defer core.deinit();
        var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
        const Results = struct {
            mutex: std.Io.Mutex = .init,
            finished: usize = 0,
            tokens: usize = 0,
            errors: usize = 0,
            order: [2]Id = .{ 0, 0 },
            filled: usize = 0,
            fn event(ctx: *anyopaque, id: Id, ev: *const Event) void {
                const r: *@This() = @ptrCast(@alignCast(ctx));
                r.mutex.lockUncancelable(std.testing.io);
                defer r.mutex.unlock(std.testing.io);
                switch (ev.*) {
                    .tokens => |ids| r.tokens += ids.len,
                    .prefilled => {
                        r.order[r.filled] = id;
                        r.filled += 1;
                    },
                    .finished => |end| {
                        r.finished += 1;
                        r.errors += @intFromBool(end.reason != .length);
                    },
                    else => {},
                }
            }
        };
        var results: Results = .{};
        const first: Request = .{ .prompt = &.{ 1, 2, 3, 4 }, .max_tokens = 4, .drafts = false };
        const second: Request = .{ .prompt = &.{ 7, 8 }, .max_tokens = 4, .drafts = false };
        const e = host.engine();
        try e.submit(1, &first, .{ .ctx = &results, .event = Results.event });
        try e.submit(2, &second, .{ .ctx = &results, .event = Results.event });
        try host.start();
        defer host.stop();
        var waits: usize = 0;
        while (true) {
            results.mutex.lockUncancelable(std.testing.io);
            const done = results.finished;
            results.mutex.unlock(std.testing.io);
            if (done == 2) break;
            if (waits == 5000) return error.TestTimedOut;
            waits += 1;
            std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
        }
        try std.testing.expectEqual(@as(usize, 1), target.prefill_batches);
        try std.testing.expectEqual(@as(usize, 8), results.tokens);
        try std.testing.expectEqual(@as(usize, 0), results.errors);
        try std.testing.expectEqualSlices(Id, &.{ 2, 1 }, &results.order);
    }
}

test "fresh prompt cohorts cannot starve an aged long or background fill" {
    const fill = @import("lane_host_fill.zig");
    for ([_]bool{ false, true }) |background| {
        const gpa = std.testing.allocator;
        var cfg = try lanes.Config.init(gpa, .{ .exact_width = 1, .gpu_tokens = false }, 1, 0);
        defer cfg.deinit(gpa);
        var target: lanes.fake.Fake = .{ .gpa = gpa, .batch_max_prompt = 2 };
        defer target.deinit();
        var clock: lanes.fake.FixedClock = .{};
        var core = lanes.Engine.init(gpa, &cfg, target.coldBatched(), clock.clock());
        defer core.deinit();
        var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 3 });
        const Results = struct {
            const short_limit = 2 * (fill.preference_rounds + 2);
            engine: Engine,
            request: *const Request,
            mutex: std.Io.Mutex = .init,
            submitted: usize = 2,
            short_done: usize = 0,
            long_after: ?usize = null,
            done: usize = 0,
            errors: usize = 0,
            fn event(ctx: *anyopaque, id: Id, ev: *const Event) void {
                const r: *@This() = @ptrCast(@alignCast(ctx));
                r.mutex.lockUncancelable(std.testing.io);
                defer r.mutex.unlock(std.testing.io);
                switch (ev.*) {
                    .prefilled => if (id == 0) {
                        r.long_after = r.short_done;
                    },
                    .finished => |end| {
                        r.done += 1;
                        r.errors += @intFromBool(end.reason != .length);
                        if (id != 0) {
                            r.short_done += 1;
                            if (r.submitted < short_limit) {
                                r.submitted += 1;
                                r.engine.submit(@intCast(r.submitted), r.request, .{ .ctx = r, .event = event }) catch {
                                    r.errors += 1;
                                };
                            }
                        }
                    },
                    else => {},
                }
            }
        };
        const short: Request = .{ .prompt = &.{ 1, 2 }, .max_tokens = 1, .drafts = false };
        const long: Request = .{ .prompt = &@as([128]u32, @splat(3)), .max_tokens = 1, .drafts = false, .background = background };
        const e = host.engine();
        var results: Results = .{ .engine = e, .request = &short };
        try e.submit(0, &long, .{ .ctx = &results, .event = Results.event });
        try e.submit(1, &short, .{ .ctx = &results, .event = Results.event });
        try e.submit(2, &short, .{ .ctx = &results, .event = Results.event });
        try host.start();
        defer host.stop();
        var waits: usize = 0;
        while (true) {
            results.mutex.lockUncancelable(std.testing.io);
            const done = results.done;
            results.mutex.unlock(std.testing.io);
            if (done == Results.short_limit + 1) break;
            if (waits == 5000) return error.TestTimedOut;
            waits += 1;
            std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
        }
        try std.testing.expectEqual(@as(usize, 0), results.errors);
        try std.testing.expect(target.prefill_batches >= fill.preference_rounds);
        try std.testing.expect(results.long_after != null);
        try std.testing.expect(results.long_after.? <= 2 * (fill.preference_rounds + 1));
        try std.testing.expect(results.long_after.? < Results.short_limit);
    }
}

test "a lane host fills the shortest waiting prompt first, so a short request waits on no long one" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa, .prefill_chunks = 3 };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.stepped(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
    const Order = struct {
        mutex: std.Io.Mutex = .init,
        prefilled: [2]Id = .{ 0, 0 },
        count: usize = 0,
        done: usize = 0,
        fn event(ctx: *anyopaque, id: Id, e: *const Event) void {
            const o: *@This() = @ptrCast(@alignCast(ctx));
            o.mutex.lockUncancelable(std.testing.io);
            defer o.mutex.unlock(std.testing.io);
            switch (e.*) {
                .prefilled => {
                    o.prefilled[o.count] = id;
                    o.count += 1;
                },
                .finished => o.done += 1,
                else => {},
            }
        }
    };
    var order: Order = .{};
    const long = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3 };
    const short = [_]u32{ 2, 7, 1, 8 };
    const first: Request = .{ .prompt = &long, .max_tokens = 4 };
    const second: Request = .{ .prompt = &short, .max_tokens = 4 };
    const e = host.engine();
    // both wait before the engine thread starts: it admits them together and picks between them
    try e.submit(1, &first, .{ .ctx = &order, .event = Order.event });
    try e.submit(2, &second, .{ .ctx = &order, .event = Order.event });
    try host.start();
    defer host.stop();
    while (true) {
        order.mutex.lockUncancelable(std.testing.io);
        const d = order.done;
        order.mutex.unlock(std.testing.io);
        if (d == 2) break;
        std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
    }
    try std.testing.expectEqualSlices(Id, &.{ 2, 1 }, &order.prefilled);
}

test "with fill_arrival a lane host fills prompts in admission order, so a long one is never starved" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa, .prefill_chunks = 3 };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.stepped(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2, .fill_arrival = true });
    const Order = struct {
        mutex: std.Io.Mutex = .init,
        prefilled: [2]Id = .{ 0, 0 },
        count: usize = 0,
        done: usize = 0,
        fn event(ctx: *anyopaque, id: Id, e: *const Event) void {
            const o: *@This() = @ptrCast(@alignCast(ctx));
            o.mutex.lockUncancelable(std.testing.io);
            defer o.mutex.unlock(std.testing.io);
            switch (e.*) {
                .prefilled => {
                    o.prefilled[o.count] = id;
                    o.count += 1;
                },
                .finished => o.done += 1,
                else => {},
            }
        }
    };
    var order: Order = .{};
    const long = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3 };
    const short = [_]u32{ 2, 7, 1, 8 };
    const first: Request = .{ .prompt = &long, .max_tokens = 4 };
    const second: Request = .{ .prompt = &short, .max_tokens = 4 };
    const e = host.engine();
    try e.submit(1, &first, .{ .ctx = &order, .event = Order.event });
    try e.submit(2, &second, .{ .ctx = &order, .event = Order.event });
    try host.start();
    defer host.stop();
    while (true) {
        order.mutex.lockUncancelable(std.testing.io);
        const d = order.done;
        order.mutex.unlock(std.testing.io);
        if (d == 2) break;
        std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
    }
    try std.testing.expectEqualSlices(Id, &.{ 1, 2 }, &order.prefilled);
}

test "a foreground prompt arriving while a background one fills takes the next chunk" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa, .prefill_chunks = 4 };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.stepped(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
    const Order = struct {
        mutex: std.Io.Mutex = .init,
        prefilled: [2]Id = .{ 0, 0 },
        count: usize = 0,
        done: usize = 0,
        fn event(ctx: *anyopaque, id: Id, e: *const Event) void {
            const o: *@This() = @ptrCast(@alignCast(ctx));
            o.mutex.lockUncancelable(std.testing.io);
            defer o.mutex.unlock(std.testing.io);
            switch (e.*) {
                .prefilled => {
                    o.prefilled[o.count] = id;
                    o.count += 1;
                },
                .finished => o.done += 1,
                else => {},
            }
        }
    };
    var order: Order = .{};
    const short = [_]u32{ 2, 7, 1, 8 };
    const long = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3 };
    const background: Request = .{ .prompt = &short, .max_tokens = 4, .background = true };
    const foreground: Request = .{ .prompt = &long, .max_tokens = 4 };
    const e = host.engine();
    // the background prompt's first chunk sends the foreground request: it is admitted while the other fills
    const Arrive = struct {
        engine: api.Engine,
        request: *const Request,
        order: *Order,
        sent: bool = false,
        fn call(ctx: *anyopaque, _: *lanes.Stream, _: usize) void {
            const a: *@This() = @ptrCast(@alignCast(ctx));
            if (a.sent) return;
            a.sent = true;
            a.engine.submit(2, a.request, .{ .ctx = a.order, .event = Order.event }) catch {};
        }
    };
    var arrive: Arrive = .{ .engine = e, .request = &foreground, .order = &order };
    target.prefill_hook = Arrive.call;
    target.prefill_hook_ctx = &arrive;
    try e.submit(1, &background, .{ .ctx = &order, .event = Order.event });
    try host.start();
    defer host.stop();
    while (true) {
        order.mutex.lockUncancelable(std.testing.io);
        const d = order.done;
        order.mutex.unlock(std.testing.io);
        if (d == 2) break;
        std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
    }
    try std.testing.expectEqualSlices(Id, &.{ 2, 1 }, &order.prefilled);
}

test "status reads a filling prompt's tokens while its pass runs, and none once it ends" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa, .prefill_chunks = 4 };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.stepped(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 1 });
    try host.start();
    defer host.stop();
    const Seen = struct {
        engine: Engine = undefined,
        lefts: std.ArrayList(u32) = .empty,
        done: std.atomic.Value(bool) = .init(false),

        fn hook(ctx: *anyopaque, _: *lanes.Stream, _: usize) void {
            const seen: *@This() = @ptrCast(@alignCast(ctx));
            var status: api.Status = .{};
            seen.engine.status(&status, &.{}); // the pass runs without the host's lock: this can read
            if (status.fill_tokens_left) |left| seen.lefts.append(gpa, left) catch {};
        }
        fn event(ctx: *anyopaque, _: Id, e: *const Event) void {
            const seen: *@This() = @ptrCast(@alignCast(ctx));
            if (e.* == .finished) seen.done.store(true, .release);
        }
    };
    var seen: Seen = .{};
    seen.engine = host.engine();
    target.prefill_hook = Seen.hook;
    target.prefill_hook_ctx = &seen;
    const prompt = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6 };
    const request: Request = .{ .prompt = &prompt, .max_tokens = 4 };
    try seen.engine.submit(1, &request, .{ .ctx = &seen, .event = Seen.event });
    while (!seen.done.load(.acquire)) std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
    defer seen.lefts.deinit(gpa);
    try std.testing.expect(seen.lefts.items.len >= 2); // each chunk of the pass sampled the status
    for (seen.lefts.items) |left| try std.testing.expect(left > 0 and left <= prompt.len);
    try std.testing.expect(seen.lefts.items[0] > seen.lefts.items[seen.lefts.items.len - 1]); // it counted down
    var status: api.Status = .{};
    seen.engine.status(&status, &.{});
    try std.testing.expect(status.fill_prompt_tokens == null and status.fill_tokens_left == null); // the pass is over
}

test {
    _ = @import("lane_host_fill_aging_test.zig");
}
