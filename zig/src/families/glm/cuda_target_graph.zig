//! Bounded exact-width graphs: FFN sites, or one complete target computation.
//! Full-target callers stage device metadata before dispatch; keys never own sequence addresses.
const std = @import("std");
const cuda = @import("cuda");

pub const max_rows = 32;
pub const Buffer = enum { lone, shared };
// Resource addresses alone cannot identify a new owner allocated at a recycled address.
var next_lifetime: std.atomic.Value(u64) = .init(1);
pub fn newLifetime() u64 {
    const id = next_lifetime.fetchAdd(1, .monotonic);
    if (id == 0 or id == std.math.maxInt(u64)) @panic("capture lifetime exhausted");
    return id;
}

pub const Sink = struct { lifetime: u64 = 0, address: u64 = 0, callback: usize = 0 };
pub const Context = struct {
    lifetime: u64 = 0,
    modules: usize = 0,
    scratch: usize = 0,
    weights: usize = 0,
    transport: usize = 0,
    prefetch: usize = 0,
    replicas: usize = 0,
    packs: usize = 0,
    local_taps: u64 = 0,
    tap_rows: usize = 0,
    policy: u64 = 0,
    tap: Sink = .{},
};

pub const max_marks = 512;
/// Host metadata for event nodes captured in a graph. Replay executes those nodes, not Prof.mark's host work.
pub const Trace = struct {
    count: usize = 0,
    sections: [max_marks]u8 = undefined,
};
pub const Profile = struct {
    ctx: *anyopaque,
    lifetime: u64,
    start: usize,
    rewind: *const fn (*anyopaque, usize) void,
    save: *const fn (*anyopaque, usize) Trace,
    replay: *const fn (*anyopaque, usize, *const Trace) void,
};
pub const Key = struct {
    buffer: Buffer,
    layer: usize,
    rows: usize,
    context: Context = .{},
    profile: ?Profile = null,
};
const Identity = struct {
    context: Context,
    compute: usize,
    stream: usize,
    driver: usize,
    profile: usize,
    profile_lifetime: u64,
    profile_start: usize,
    profile_callbacks: [3]usize,
};
pub const Stats = struct { site_eager: u64 = 0, site_captures: u64 = 0, site_replays: u64 = 0 };

pub const Bank = struct {
    gpa: std.mem.Allocator,
    layers: usize,
    slots: []?cuda.graph.Exec,
    records: []Record,
    stats: Stats = .{},
    const Record = struct { identity: ?Identity = null, stream: ?cuda.Stream = null, trace: Trace = .{} };

    pub fn init(gpa: std.mem.Allocator, layers: usize) !Bank {
        const count = try std.math.mul(usize, layers, 2 * max_rows);
        const slots = try gpa.alloc(?cuda.graph.Exec, count);
        errdefer gpa.free(slots);
        @memset(slots, null);
        const records = try gpa.alloc(Record, count);
        @memset(records, .{});
        return .{ .gpa = gpa, .layers = layers, .slots = slots, .records = records };
    }

    /// Call before releasing/replacing any captured resource. A failed drain leaves the executable and owner alive.
    pub fn retire(b: *Bank, stream: cuda.Stream) !void {
        try stream.synchronize();
        // Settle every captured stream before releasing even the first reference.
        for (b.records) |record| if (record.stream) |s| try s.synchronize();
        for (b.slots, b.records) |*slot, *record| {
            if (slot.*) |*exec| exec.deinit();
            slot.* = null;
            record.* = .{};
        }
    }

    /// The owner completes the stream first, and keeps scratch, modules and communicator alive until this returns.
    pub fn deinit(b: *Bank) void {
        for (b.slots) |*slot| if (slot.*) |*exec| exec.deinit();
        b.gpa.free(b.slots);
        b.gpa.free(b.records);
        b.* = undefined;
    }

    fn index(b: *const Bank, key: Key) !usize {
        if (key.layer >= b.layers or key.rows == 0 or key.rows > max_rows) return error.InvalidTargetGraphKey;
        return (@backingInt(key.buffer) * b.layers + key.layer) * max_rows + key.rows - 1;
    }

    /// The first real site runs eager once; capture records the warmed site but never executes it again that round.
    /// Errors propagate to the paired-rank owner: no capture or replay failure silently falls back to eager work.
    pub fn run(b: *Bank, stream: cuda.Stream, key: Key, context: anytype, comptime compute: anytype) !void {
        const at = try b.index(key);
        const slot = &b.slots[at];
        const record = &b.records[at];
        const identity: Identity = .{
            .context = key.context,
            .compute = @intFromPtr(&compute),
            .stream = @intFromPtr(stream.handle),
            .driver = @intFromPtr(stream.d),
            .profile = if (key.profile) |p| @intFromPtr(p.ctx) else 0,
            .profile_lifetime = if (key.profile) |p| p.lifetime else 0,
            .profile_start = if (key.profile) |p| p.start else 0,
            .profile_callbacks = if (key.profile) |p| .{ @intFromPtr(p.rewind), @intFromPtr(p.save), @intFromPtr(p.replay) } else .{ 0, 0, 0 },
        };
        if (key.profile) |p| if (p.lifetime == 0) return error.InvalidCaptureLifetime;
        if (slot.*) |*exec| {
            if (record.identity != null and std.meta.eql(record.identity.?, identity)) {
                try exec.launchOn(stream);
                if (key.profile) |p| p.replay(p.ctx, p.start, &record.trace);
                b.stats.site_replays += 1;
                return;
            }
            // A replay can still be in flight. Do not destroy its executable or release its references before this.
            try record.stream.?.synchronize();
            exec.deinit();
            slot.* = null;
            record.* = .{};
        }
        try compute(context, key.layer, key.rows);
        b.stats.site_eager += 1;
        try stream.synchronize();
        if (key.profile) |p| p.rewind(p.ctx, p.start);
        try cuda.graph.beginCapture(stream, .thread_local);
        compute(context, key.layer, key.rows) catch |err| {
            var discarded = cuda.graph.endCapture(stream) catch return err;
            discarded.deinit();
            return err;
        };
        var graph = try cuda.graph.endCapture(stream);
        defer graph.deinit();
        slot.* = try graph.instantiate();
        record.identity = identity;
        record.stream = stream;
        record.trace = if (key.profile) |p| p.save(p.ctx, p.start) else .{};
        b.stats.site_captures += 1;
    }
};

const Fake = struct {
    const abi = cuda.abi;
    var capturing: bool = false;
    var begins: usize = 0;
    var ends: usize = 0;
    var syncs: usize = 0;
    var graphs: usize = 0;
    var graph_frees: usize = 0;
    var execs: usize = 0;
    var exec_frees: usize = 0;
    var launches: usize = 0;
    var begin_code: abi.Result = 0;
    var end_code: abi.Result = 0;
    var sync_code: abi.Result = 0;
    var instantiate_code: abi.Result = 0;
    var launch_code: abi.Result = 0;

    const Site = struct {
        calls: usize = 0,
        writes: usize = 0,
        layer: usize = 0,
        rows: usize = 0,
        fail_eager: bool = false,
        fail_capture: bool = false,

        fn compute(s: *Site, layer: usize, rows: usize) !void {
            s.calls += 1;
            s.layer = layer;
            s.rows = rows;
            if (if (capturing) s.fail_capture else s.fail_eager) return error.InjectedCompute;
            if (!capturing) s.writes += 1;
        }
    };

    fn driver() cuda.Driver {
        capturing = false;
        begins = 0;
        ends = 0;
        syncs = 0;
        graphs = 0;
        graph_frees = 0;
        execs = 0;
        exec_frees = 0;
        launches = 0;
        begin_code = 0;
        end_code = 0;
        sync_code = 0;
        instantiate_code = 0;
        launch_code = 0;
        var d: cuda.Driver = .{ .lib = undefined, .api = undefined };
        d.api.cuStreamSynchronize = synchronize;
        d.api.cuStreamBeginCapture_v2 = begin;
        d.api.cuStreamEndCapture = end;
        d.api.cuGraphDestroy = destroyGraph;
        d.api.cuGraphInstantiateWithFlags = instantiate;
        d.api.cuGraphExecDestroy = destroyExec;
        d.api.cuGraphLaunch = launch;
        return d;
    }

    fn synchronize(_: abi.Stream) callconv(.c) abi.Result {
        syncs += 1;
        return sync_code;
    }
    fn begin(_: abi.Stream, mode: abi.CaptureMode) callconv(.c) abi.Result {
        std.debug.assert(mode == .thread_local and !capturing);
        begins += 1;
        if (begin_code == 0) capturing = true;
        return begin_code;
    }
    fn end(_: abi.Stream, out: *abi.Graph) callconv(.c) abi.Result {
        std.debug.assert(capturing);
        ends += 1;
        if (end_code != 0) return end_code;
        capturing = false;
        graphs += 1;
        out.* = @ptrFromInt(graphs);
        return 0;
    }
    fn destroyGraph(_: abi.Graph) callconv(.c) abi.Result {
        graph_frees += 1;
        return 0;
    }
    fn instantiate(out: *abi.GraphExec, _: abi.Graph, _: u64) callconv(.c) abi.Result {
        std.debug.assert(!capturing);
        if (instantiate_code != 0) return instantiate_code;
        execs += 1;
        out.* = @ptrFromInt(execs);
        return 0;
    }
    fn destroyExec(_: abi.GraphExec) callconv(.c) abi.Result {
        exec_frees += 1;
        return 0;
    }
    fn launch(_: abi.GraphExec, _: abi.Stream) callconv(.c) abi.Result {
        std.debug.assert(!capturing);
        launches += 1;
        return launch_code;
    }
};

test "a missed site executes eager once and subsequent rounds replay without the host callback" {
    var d = Fake.driver();
    const stream: cuda.Stream = .{ .d = &d, .handle = null };
    var bank = try Bank.init(std.testing.allocator, 2);
    var alive = true;
    defer if (alive) bank.deinit();
    var site: Fake.Site = .{};
    const key: Key = .{ .buffer = .shared, .layer = 1, .rows = 32 };
    try bank.run(stream, key, &site, Fake.Site.compute);
    try std.testing.expectEqual(@as(usize, 2), site.calls);
    try std.testing.expectEqual(@as(usize, 1), site.writes);
    try std.testing.expectEqual(@as(usize, 1), site.layer);
    try std.testing.expectEqual(@as(usize, 32), site.rows);
    try std.testing.expectEqual(@as(usize, 0), Fake.launches);
    try std.testing.expectEqual(@as(usize, 1), Fake.syncs);
    try std.testing.expectEqualDeep(Stats{ .site_eager = 1, .site_captures = 1 }, bank.stats);
    var other: Fake.Site = .{ .fail_eager = true, .fail_capture = true };
    try bank.run(stream, key, &other, Fake.Site.compute);
    try std.testing.expectEqual(@as(usize, 0), other.calls);
    try std.testing.expectEqual(@as(usize, 1), Fake.launches);
    try std.testing.expectEqualDeep(Stats{ .site_eager = 1, .site_captures = 1, .site_replays = 1 }, bank.stats);
    bank.deinit();
    alive = false;
    try std.testing.expectEqual(Fake.graphs, Fake.graph_frees);
    try std.testing.expectEqual(Fake.execs, Fake.exec_frees);
}

test "buffer layer and exact rows independently name bounded sites and invalid keys execute nothing" {
    var d = Fake.driver();
    const stream: cuda.Stream = .{ .d = &d, .handle = null };
    var bank = try Bank.init(std.testing.allocator, 2);
    defer bank.deinit();
    try std.testing.expectEqual(@as(usize, 128), bank.slots.len);
    var site: Fake.Site = .{};
    for ([_]Key{
        .{ .buffer = .lone, .layer = 0, .rows = 1 },
        .{ .buffer = .shared, .layer = 0, .rows = 1 },
        .{ .buffer = .shared, .layer = 1, .rows = 1 },
        .{ .buffer = .shared, .layer = 1, .rows = 2 },
    }) |key| try bank.run(stream, key, &site, Fake.Site.compute);
    try std.testing.expectEqual(@as(u64, 4), bank.stats.site_captures);
    for ([_]Key{
        .{ .buffer = .lone, .layer = 2, .rows = 1 },
        .{ .buffer = .shared, .layer = 0, .rows = 0 },
        .{ .buffer = .shared, .layer = 0, .rows = 33 },
    }) |key| try std.testing.expectError(error.InvalidTargetGraphKey, bank.run(stream, key, &site, Fake.Site.compute));
    try std.testing.expectEqual(@as(usize, 8), site.calls);
    try std.testing.expectEqual(@as(usize, 0), Fake.launches);
    try std.testing.expectError(error.Overflow, Bank.init(std.testing.allocator, std.math.maxInt(usize)));
}

test "eager sync begin capture and instantiate failures never replay or publish an executable" {
    const Failure = enum { eager, sync, begin, compute, end, instantiate };
    for (std.enums.values(Failure)) |failure| {
        var d = Fake.driver();
        const stream: cuda.Stream = .{ .d = &d, .handle = null };
        var bank = try Bank.init(std.testing.allocator, 1);
        defer bank.deinit();
        var site: Fake.Site = .{};
        switch (failure) {
            .eager => site.fail_eager = true,
            .sync => Fake.sync_code = cuda.abi.error_not_ready,
            .begin => Fake.begin_code = cuda.abi.error_not_ready,
            .compute => site.fail_capture = true,
            .end => Fake.end_code = cuda.abi.error_not_ready,
            .instantiate => Fake.instantiate_code = cuda.abi.error_not_ready,
        }
        const expected = if (failure == .eager or failure == .compute) error.InjectedCompute else error.NotReady;
        try std.testing.expectError(expected, bank.run(stream, .{ .buffer = .lone, .layer = 0, .rows = 1 }, &site, Fake.Site.compute));
        try std.testing.expectEqual(@as(usize, 0), Fake.launches);
        try std.testing.expectEqual(@as(u64, 0), bank.stats.site_captures);
        for (bank.slots) |slot| try std.testing.expect(slot == null);
        if (failure == .end) {
            try std.testing.expectError(error.UnownedCaptureSource, cuda.copy_source.synchronousCopy(&d, 0, &.{}, std.testing.allocator));
            Fake.end_code = 0;
            var abandoned = try cuda.graph.endCapture(stream);
            abandoned.deinit();
        }
        try cuda.copy_source.synchronousCopy(&d, 0, &.{}, std.testing.allocator);
        try std.testing.expectEqual(Fake.graphs, Fake.graph_frees);
        try std.testing.expectEqual(Fake.execs, Fake.exec_frees);
    }
}

test "a failed capture-body cleanup keeps source custody until a successful capture end" {
    var d = Fake.driver();
    const stream: cuda.Stream = .{ .d = &d, .handle = null };
    var bank = try Bank.init(std.testing.allocator, 1);
    defer bank.deinit();
    var site: Fake.Site = .{ .fail_capture = true };
    Fake.end_code = cuda.abi.error_not_ready;
    try std.testing.expectError(error.InjectedCompute, bank.run(stream, .{ .buffer = .lone, .layer = 0, .rows = 1 }, &site, Fake.Site.compute));
    try std.testing.expectEqual(@as(usize, 1), site.writes);
    try std.testing.expectEqual(@as(usize, 0), Fake.launches);
    try std.testing.expectEqual(@as(u64, 0), bank.stats.site_captures);
    try std.testing.expectError(error.UnownedCaptureSource, cuda.copy_source.synchronousCopy(&d, 0, &.{}, std.testing.allocator));
    Fake.end_code = 0;
    var abandoned = try cuda.graph.endCapture(stream);
    abandoned.deinit();
    try cuda.copy_source.synchronousCopy(&d, 0, &.{}, std.testing.allocator);
    try std.testing.expectEqual(Fake.graphs, Fake.graph_frees);
}

test "one full-target site has distinct exact-width graphs for both stable buffers" {
    var d = Fake.driver();
    const stream: cuda.Stream = .{ .d = &d, .handle = null };
    var bank = try Bank.init(std.testing.allocator, 1);
    defer bank.deinit();
    var first: Fake.Site = .{};
    for ([_]Buffer{ .lone, .shared }) |buffer| {
        for (1..33) |rows| try bank.run(stream, .{ .buffer = buffer, .layer = 0, .rows = rows }, &first, Fake.Site.compute);
    }
    try std.testing.expectEqual(@as(usize, 64), bank.slots.len);
    try std.testing.expectEqual(@as(usize, 64), first.writes);
    try std.testing.expectEqualDeep(Stats{ .site_eager = 64, .site_captures = 64 }, bank.stats);
    var changed_metadata: Fake.Site = .{ .fail_eager = true, .fail_capture = true };
    for ([_]Buffer{ .shared, .lone }) |buffer| {
        for (1..33) |rows| try bank.run(stream, .{ .buffer = buffer, .layer = 0, .rows = rows }, &changed_metadata, Fake.Site.compute);
    }
    try std.testing.expectEqual(@as(usize, 0), changed_metadata.calls);
    try std.testing.expectEqual(@as(usize, 64), Fake.launches);
    try std.testing.expectEqualDeep(Stats{ .site_eager = 64, .site_captures = 64, .site_replays = 64 }, bank.stats);
}

test "failed replay propagates without calling eager compute or counting a successful replay" {
    var d = Fake.driver();
    const stream: cuda.Stream = .{ .d = &d, .handle = null };
    var bank = try Bank.init(std.testing.allocator, 1);
    defer bank.deinit();
    var site: Fake.Site = .{};
    const key: Key = .{ .buffer = .lone, .layer = 0, .rows = 1 };
    try bank.run(stream, key, &site, Fake.Site.compute);
    Fake.launch_code = cuda.abi.error_not_ready;
    try std.testing.expectError(error.NotReady, bank.run(stream, key, &site, Fake.Site.compute));
    try std.testing.expectEqual(@as(usize, 2), site.calls);
    try std.testing.expectEqual(@as(usize, 1), site.writes);
    try std.testing.expectEqual(@as(usize, 1), Fake.launches);
    try std.testing.expectEqual(@as(u64, 0), bank.stats.site_replays);
}

test "context retirement failures retain the old owner and failed recapture never publishes the new identity" {
    var d = Fake.driver();
    const stream: cuda.Stream = .{ .d = &d, .handle = null };
    var bank = try Bank.init(std.testing.allocator, 1);
    defer bank.deinit();
    var site: Fake.Site = .{};
    var key: Key = .{ .buffer = .lone, .layer = 0, .rows = 1, .context = .{ .tap = .{ .lifetime = newLifetime(), .address = 0x1000 } } };
    try bank.run(stream, key, &site, Fake.Site.compute);
    const old = bank.records[0].identity.?;
    key.context.tap.lifetime = newLifetime();
    Fake.sync_code = cuda.abi.error_not_ready;
    try std.testing.expectError(error.NotReady, bank.run(stream, key, &site, Fake.Site.compute));
    try std.testing.expectEqualDeep(old, bank.records[0].identity.?);
    try std.testing.expectEqual(@as(usize, 0), Fake.exec_frees);
    try std.testing.expectEqual(@as(usize, 2), site.calls);
    try std.testing.expectError(error.NotReady, bank.retire(stream));
    try std.testing.expectEqual(@as(usize, 0), Fake.exec_frees);
    Fake.sync_code = 0;
    Fake.instantiate_code = cuda.abi.error_not_ready;
    try std.testing.expectError(error.NotReady, bank.run(stream, key, &site, Fake.Site.compute));
    try std.testing.expectEqual(@as(usize, 1), Fake.exec_frees);
    try std.testing.expect(bank.slots[0] == null and bank.records[0].identity == null);
    Fake.instantiate_code = 0;
    try bank.run(stream, key, &site, Fake.Site.compute);
    const calls = site.calls;
    try bank.run(stream, key, &site, Fake.Site.compute);
    try std.testing.expectEqual(calls, site.calls);
    try bank.retire(stream);
    try std.testing.expectEqual(Fake.execs, Fake.exec_frees);
}

test "capture callback identity participates independently of resource addresses" {
    const Other = struct {
        fn compute(s: *Fake.Site, layer: usize, rows: usize) !void {
            try Fake.Site.compute(s, layer, rows);
        }
    };
    var d = Fake.driver();
    const stream: cuda.Stream = .{ .d = &d, .handle = null };
    var bank = try Bank.init(std.testing.allocator, 1);
    defer bank.deinit();
    var site: Fake.Site = .{};
    const key: Key = .{ .buffer = .lone, .layer = 0, .rows = 1 };
    try bank.run(stream, key, &site, Fake.Site.compute);
    try bank.run(stream, key, &site, Other.compute);
    try std.testing.expectEqual(@as(u64, 2), bank.stats.site_captures);
    try std.testing.expectEqual(@as(usize, 1), Fake.exec_frees);
}

test "bounded graph slot and metadata allocation failures release earlier ownership" {
    const Check = struct {
        fn run(gpa: std.mem.Allocator) !void {
            var bank = try Bank.init(gpa, 2);
            defer bank.deinit();
        }
    };
    try std.testing.checkAllAllocationFailures(std.testing.allocator, Check.run, .{});
}
