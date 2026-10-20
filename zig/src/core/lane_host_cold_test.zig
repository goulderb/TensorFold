const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const LaneHost = @import("lane_host.zig").LaneHost;
const a = std.testing.allocator;
const io = std.testing.io;
const Mode = enum { cold, bounded, cancelled, blocked, capacity, busy, solo };

const Gate = struct {
    mutex: std.Io.Mutex = .init,
    cond: std.Io.Condition = .init,
    entered: usize = 0,
    released: usize = 0,
    finished: usize = 0,
    tokens: [8][16]u32 = @splat(@splat(0)),
    counts: [8]usize = @splat(0),
    reasons: [8]?api.Reason = @splat(null),

    fn block(g: *Gate, stage: usize) void {
        g.mutex.lockUncancelable(io);
        defer g.mutex.unlock(io);
        g.entered = stage;
        g.cond.broadcast(io);
        while (g.released < stage) g.cond.waitUncancelable(io, &g.mutex);
    }
    fn wait(g: *Gate, stage: usize, done: bool) !void {
        g.mutex.lockUncancelable(io);
        defer g.mutex.unlock(io);
        const deadline = std.Io.Clock.awake.now(io).toNanoseconds() + 5 * std.time.ns_per_s;
        while ((if (done) g.finished else g.entered) < stage) {
            if (std.Io.Clock.awake.now(io).toNanoseconds() >= deadline) return error.GateTimedOut;
            g.cond.waitTimeout(io, &g.mutex, .{ .duration = .{ .raw = .fromMilliseconds(10), .clock = .awake } }) catch {};
        }
    }
    fn release(g: *Gate, stage: usize) void {
        g.mutex.lockUncancelable(io);
        defer g.mutex.unlock(io);
        g.released = stage;
        g.cond.broadcast(io);
    }
    fn event(ptr: *anyopaque, id: api.Id, ev: *const api.Event) void {
        const g: *Gate = @ptrCast(@alignCast(ptr));
        g.mutex.lockUncancelable(io);
        defer g.mutex.unlock(io);
        const index: usize = @intCast(id - 1);
        switch (ev.*) {
            .tokens => |tokens| {
                @memcpy(g.tokens[index][g.counts[index]..][0..tokens.len], tokens);
                g.counts[index] += tokens.len;
            },
            .finished => |end| {
                std.debug.assert(g.reasons[index] == null);
                g.reasons[index] = end.reason;
                g.finished += 1;
                g.cond.broadcast(io);
            },
            else => {},
        }
    }
};

const Target = struct {
    fake: lanes.fake.Fake = .{ .gpa = a },
    base: *const lanes.backend.Backend.VTable = undefined,
    table: lanes.backend.Backend.VTable = undefined,
    gate: *Gate,
    mode: Mode,
    fills: usize = 0,
    verifies: usize = 0,
    widths: [2]usize = @splat(0),
    positions: [8]u64 = @splat(0),
    releases: usize = 0,

    fn backend(t: *Target) lanes.backend.Backend {
        var backend_ = t.fake.coldBatched();
        t.base = backend_.vtable;
        t.table = t.base.*;
        t.table.prefill_batch = prefill;
        t.table.verify = verify;
        t.table.release = release;
        backend_.vtable = &t.table;
        return backend_;
    }
    fn self(ptr: *anyopaque) *Target {
        const fake: *lanes.fake.Fake = @ptrCast(@alignCast(ptr));
        return @fieldParentPtr("fake", fake);
    }
    fn prefill(ptr: *anyopaque, streams: []const *lanes.Stream, out: []lanes.backend.PrefillOutcome) !void {
        const t = self(ptr);
        t.fills += 1;
        if (t.mode == .busy) {
            if (t.fills == 1) t.gate.block(2);
        } else if (t.fills == 1 or (t.mode == .bounded and t.fills == 2)) t.gate.block(t.fills);
        try t.base.prefill_batch.?(ptr, streams, out);
    }
    fn verify(ptr: *anyopaque, windows: []const lanes.backend.Window, out: []lanes.backend.Verified) !void {
        const t = self(ptr);
        if (t.verifies < 2) t.widths[t.verifies] = windows.len;
        if (t.verifies == 0) {
            for (windows, 0..) |window, index| t.positions[index] = window.positions[0];
            if (t.mode == .busy) t.gate.block(1);
        }
        t.verifies += 1;
        try t.base.verify(ptr, windows, out);
    }
    fn release(ptr: *anyopaque, stream: *lanes.Stream) void {
        const t = self(ptr);
        t.releases += 1;
        t.base.release(ptr, stream);
    }
    fn fits(_: *anyopaque, _: u32, _: u32) bool {
        return false;
    }
};

const Result = struct { width: usize, second: usize, tokens: [8][16]u32 };

fn run(enabled: bool, mode: Mode) !Result {
    var gate: Gate = .{};
    var target: Target = .{ .gate = &gate, .mode = mode };
    defer target.fake.deinit();
    var cfg = try lanes.Config.init(a, .{ .exact_width = 8, .hidden_rows = true, .mtp = true, .speculate = true, .drafts = 7 }, 8, 7);
    defer cfg.deinit(a);
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(a, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(a, io, &core, .{ .lanes = if (mode == .capacity) 3 else 8, .admit_after_cold_fill = enabled, .admit = if (mode == .blocked) .{ .ctx = &target, .fits = Target.fits } else null });
    const request: api.Request = .{ .prompt = &.{ 2, 3, 5, 7 }, .max_tokens = 16 };
    const e = host.engine();
    const count: usize = if (mode == .solo) 1 else 8;
    const initial: usize = if (mode == .busy or mode == .solo) 1 else 3;
    for (0..initial) |index| try e.submit(index + 1, &request, .{ .ctx = &gate, .event = Gate.event });
    try host.start();
    var stopped = false;
    defer if (!stopped) {
        gate.release(std.math.maxInt(usize));
        host.stop();
    };
    if (mode != .solo) {
        try gate.wait(1, false);
        const middle: usize = if (mode == .bounded) 6 else if (mode == .busy) 4 else count;
        for (initial..middle) |index| try e.submit(index + 1, &request, .{ .ctx = &gate, .event = Gate.event });
        if (mode == .cancelled) e.cancel(8);
        gate.release(1);
        if (mode == .bounded or mode == .busy) {
            try gate.wait(2, false);
            for (middle..count) |index| try e.submit(index + 1, &request, .{ .ctx = &gate, .event = Gate.event });
            gate.release(2);
        }
    }
    try gate.wait(count, true);
    host.stop();
    stopped = true;
    for (0..count) |index| {
        const cancelled = mode == .cancelled and index == 7;
        try std.testing.expectEqual(if (cancelled) api.Reason.cancelled else api.Reason.length, gate.reasons[index].?);
        try std.testing.expectEqual(@as(usize, if (cancelled) 0 else 16), gate.counts[index]);
    }
    const expected_release = count - @intFromBool(mode == .cancelled);
    try std.testing.expectEqual(expected_release, target.releases);
    for (target.positions[0..target.widths[0]]) |position| try std.testing.expectEqual(@as(u64, request.prompt.len + 1), position);
    return .{ .width = target.widths[0], .second = target.widths[1], .tokens = gate.tokens };
}

test "cold admission default verifies three; opt-in fills queued five before first verify" {
    try std.testing.expect(!(api.Info{}).admit_after_cold_fill);
    const before = try run(false, .cold);
    try std.testing.expectEqual(@as(usize, 3), before.width);
    const after = try run(true, .cold);
    try std.testing.expectEqual(@as(usize, 8), after.width);
    try std.testing.expectEqualDeep(before.tokens, after.tokens);
}

test "cold admission extension is bounded and leaves busy decode ordering intact" {
    try std.testing.expectEqual(@as(usize, 6), (try run(true, .bounded)).width);
    const busy = try run(true, .busy);
    try std.testing.expectEqual(@as(usize, 1), busy.width);
    try std.testing.expectEqual(@as(usize, 4), busy.second);
}

test "cold admission handles cancellation, capacity, blocked head and lone request" {
    try std.testing.expectEqual(@as(usize, 7), (try run(true, .cancelled)).width);
    try std.testing.expectEqual(@as(usize, 3), (try run(true, .capacity)).width);
    try std.testing.expectEqual(@as(usize, 3), (try run(true, .blocked)).width);
    try std.testing.expectEqual(@as(usize, 1), (try run(true, .solo)).width);
}
