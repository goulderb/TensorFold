//! Mia f013e1ae numerical DFlash package in the current GLM engine.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const config = @import("cuda_dflash_config.zig");
const weights = @import("cuda_dflash_weights.zig");
const Forward = @import("cuda_dflash_forward.zig").Forward;
const Ops = @import("cuda_dflash_ops.zig").Ops;
const Entries = @import("cuda_dflash_entries.zig").Entries;
const policy = @import("cuda_draft_policy.zig");
const state = @import("cuda_dflash_state.zig");
const batch = @import("cuda_dflash_batch.zig");
const pinned_aot = @import("cuda_dflash_aot.zig");
const verify = @import("cuda_verify_plan.zig");
const Sampling = @import("cuda_sampler.zig").Sampling;
pub const BatchRequest = batch.Request;
pub const BatchResult = batch.Result;

const BlockKey = struct {
    const Item = struct { slot: usize, end: usize, pending: u32, generation: u64 };
    items: [batch.max_lanes]Item = undefined,
    count: usize = 0,

    fn matches(a: BlockKey, b: BlockKey) bool {
        if (a.count != b.count) return false;
        for (a.items[0..a.count], b.items[0..b.count]) |left, right| {
            if (left.slot != right.slot or left.end != right.end or left.pending != right.pending or left.generation != right.generation) return false;
        }
        return true;
    }
};

const PendingBlock = union(enum) {
    idle,
    queued: struct { key: BlockKey, plan: batch.Plan },
};

const WidthHistogram = struct {
    bins: [batch.max_lanes]u64 = @splat(0),
    total: u64 = 0,
    emitted: u64 = 0,

    fn record(h: *WidthHistogram, width: usize) void {
        std.debug.assert(width > 0 and width <= batch.max_lanes);
        h.bins[width - 1] += 1;
        h.total += 1;
    }

    fn due(h: *const WidthHistogram) bool {
        return h.total == 1 or (h.total != 0 and h.total % 64 == 0);
    }

    const Snapshot = struct { total: u64, bins: [batch.max_lanes]u64 };

    fn snapshot(h: *WidthHistogram) ?Snapshot {
        if (h.total == h.emitted) return null;
        h.emitted = h.total;
        return .{ .total = h.total, .bins = h.bins };
    }
};

pub const Dependencies = struct {
    driver: *const cuda.Driver,
    context: *const cuda.Context,
    stream: cuda.Stream,
    embed: u64,
    lib: ?*const cuda.nccl.Library,
    comm: cuda.nccl.Comm,
    rank: usize,
    world: usize,
    kernels_dir: ?[]const u8 = null,
    verify_plan: verify.Plan = .baseline,
    tp_group: ?*const cuda.tp_group.Group = null,
};

test "deferred block identity preserves cohort order and slot generations" {
    const key: BlockKey = .{
        .count = 2,
        .items = .{
            .{ .slot = 5, .end = 40, .pending = 11, .generation = 7 },
            .{ .slot = 1, .end = 400, .pending = 22, .generation = 3 },
            undefined,
            undefined,
            undefined,
            undefined,
            undefined,
            undefined,
        },
    };
    var same = key;
    try std.testing.expect(key.matches(same));
    same.items[1].generation += 1;
    try std.testing.expect(!key.matches(same));
    same = key;
    std.mem.swap(BlockKey.Item, &same.items[0], &same.items[1]);
    try std.testing.expect(!key.matches(same));
}

pub const DFlash = struct {
    gpa: std.mem.Allocator,
    loader: weights.Loader,
    forward: Forward,
    ops: Ops,
    entries: Entries,
    lt: cuda.cublaslt.Library,
    aot: cuda.aot.Set,
    pred: []f32,
    succ: []f32,
    capacity: usize,
    verify_plan: verify.Plan = .baseline,
    lanes: [8]state.Lane = @splat(.{}),
    generations: [8]u64 = @splat(0),
    pending_block: PendingBlock = .idle,
    prelaunch_stats: struct { launched: u64 = 0, used: u64 = 0, discarded: u64 = 0 } = .{},
    pending: [8]cuda.DeviceBuffer,
    max_packed_width: usize = 0,
    width_histogram: WidthHistogram = .{},

    pub fn open(gpa: std.mem.Allocator, io: std.Io, target_dir: []const u8, draft_dir: []const u8, streams: usize, capacity: usize, deps: Dependencies) !*DFlash {
        if (streams != 8 or capacity == 0 or capacity > std.math.maxInt(i64) - 8) return error.BadDraftCapacity;
        if (deps.world != 2 or deps.lib == null or deps.comm == null) return error.UnsupportedDraftTp;
        if (try deps.context.capability() != 121) return error.UnsupportedDraftDevice;
        const cfg = try config.Config.read(gpa, io, draft_dir);
        const rank = try config.Rank.init(deps.world, deps.rank);
        var target = try core.Checkpoint.openModelPrefix(gpa, io, target_dir, "lm_head.");
        defer target.close();
        if (!target.has("lm_head.weight")) return error.DFlashNeedsDenseTargetHead;
        var draft = try core.Checkpoint.openModel(gpa, io, draft_dir);
        defer draft.close();
        const p = try gpa.create(DFlash);
        errdefer gpa.destroy(p);
        p.gpa = gpa;
        p.capacity = capacity;
        p.verify_plan = deps.verify_plan;
        p.lanes = @splat(.{});
        p.generations = @splat(0);
        p.pending_block = .idle;
        p.prelaunch_stats = .{};
        p.max_packed_width = 0;
        p.width_histogram = .{};
        p.loader = .{ .gpa = gpa, .driver = deps.driver, .target = &target, .draft = &draft, .rank = rank };
        errdefer p.loader.deinit();
        const w = try p.loader.load(rank);
        p.pred = try weights.readCodebook(gpa, &draft, "candidate_selector.predecessor_codebook");
        errdefer gpa.free(p.pred);
        p.succ = try weights.readCodebook(gpa, &draft, "candidate_selector.successor_codebook");
        errdefer gpa.free(p.succ);
        try (policy.Selector{ .pred = p.pred, .succ = p.succ }).validate(154880);
        p.lt = try cuda.cublaslt.Library.open();
        errdefer p.lt.close();
        p.ops = try Ops.load(deps.context);
        errdefer p.ops.deinit();
        p.entries = try Entries.load(deps.driver);
        errdefer p.entries.deinit();
        const kernels_dir = try pinned_aot.directory(gpa, io, deps.kernels_dir, draft_dir);
        defer gpa.free(kernels_dir);
        try pinned_aot.validate(gpa, io, kernels_dir);
        p.aot = try cuda.aot.Set.load(gpa, io, deps.driver, deps.context.device, kernels_dir);
        errdefer p.aot.deinit();
        p.forward = try Forward.init(gpa, deps.driver, deps.stream, cfg, rank, streams, w, deps.embed, .{ .set = &p.aot, .s = deps.stream }, p.ops.on(deps.stream), &p.entries, .{ .lib = deps.lib.?, .comm = deps.comm, .rank = deps.rank, .world = deps.world, .tp_group = deps.tp_group }, &p.lt);
        errdefer p.forward.deinit();
        var allocated: usize = 0;
        errdefer for (p.pending[0..allocated]) |*b| b.free();
        for (&p.pending) |*b| {
            b.* = try cuda.DeviceBuffer.alloc(deps.driver, 32 * 5 * 8192);
            allocated += 1;
        }
        try deps.stream.synchronize();
        return p;
    }
    pub fn prepare(d: *DFlash) !void {
        try d.forward.capture();
    }
    pub fn asyncAvailable(d: *const DFlash) bool {
        return d.forward.asyncOk() and d.forward.stager.ready();
    }
    pub fn deinit(d: *DFlash) void {
        d.settlePending() catch {};
        d.synchronize() catch {};
        d.flushWidths();
        for (&d.pending) |*b| b.free();
        d.forward.deinit();
        d.aot.deinit();
        d.entries.deinit();
        d.ops.deinit();
        d.lt.close();
        d.loader.deinit();
        d.gpa.free(d.pred);
        d.gpa.free(d.succ);
        d.gpa.destroy(d);
    }
    fn lane(d: *DFlash, slot: usize) !*state.Lane {
        if (slot >= 8) return error.BadDraftSlot;
        return &d.lanes[slot];
    }
    fn blockKey(d: *const DFlash, plan: batch.Plan) BlockKey {
        var key: BlockKey = .{ .count = plan.count };
        for (plan.requests[0..plan.count], 0..) |r, i| key.items[i] = .{ .slot = r.slot, .end = r.end, .pending = r.pending, .generation = d.generations[r.slot] };
        return key;
    }
    fn synchronize(d: *const DFlash) !void {
        try d.forward.stream.synchronize();
        if (d.forward.comm) |comm| if (comm.tp_group) |group| try group.check();
    }
    fn settlePending(d: *DFlash) !void {
        switch (d.pending_block) {
            .idle => return,
            .queued => |queued| {
                d.prelaunch_stats.discarded += 1;
                d.synchronize() catch |err| {
                    for (queued.plan.requests[0..queued.plan.count]) |r| d.lanes[r.slot].healthy = false;
                    d.pending_block = .idle;
                    return err;
                };
                d.forward.settle(true) catch |err| {
                    for (queued.plan.requests[0..queued.plan.count]) |r| d.lanes[r.slot].healthy = false;
                    d.pending_block = .idle;
                    return err;
                };
            },
        }
        d.pending_block = .idle;
    }
    fn invalidateContext(d: *DFlash, slot: usize) !void {
        if (slot >= d.generations.len) return error.BadDraftSlot;
        if (d.generations[slot] == std.math.maxInt(u64)) return error.DraftGenerationExhausted;
        try d.settlePending();
        d.generations[slot] += 1;
    }
    pub fn reset(d: *DFlash, slot: usize) !void {
        const l = try d.lane(slot);
        try d.invalidateContext(slot);
        l.healthy = false;
        const bytes = config.Config.ring * 256;
        for (0..5) |layer| for (0..d.forward.rank.kv) |head| {
            const offset = state.ringOffset(8, slot, head);
            try d.forward.th.base.fill32(d.forward.kc[layer] + offset, 0, bytes / 4);
            try d.forward.th.base.fill32(d.forward.vc[layer] + offset, 0, bytes / 4);
        };
        try d.synchronize();
        l.* = .{};
    }
    pub fn release(d: *DFlash, slot: usize) void {
        if (slot < 8) {
            d.settlePending() catch |err| std.log.err("draining DFlash prelaunch on release of slot {d}: {s}", .{ slot, @errorName(err) });
            if (d.generations[slot] != std.math.maxInt(u64)) d.generations[slot] += 1;
            d.lanes[slot] = .{ .healthy = false };
            d.flushWidths();
        }
    }
    fn flushWidths(d: *DFlash) void {
        if (d.forward.rank.rank != 0) return;
        const snapshot = d.width_histogram.snapshot() orelse return;
        std.log.info("DFlash width histogram: version=1 rank=0 total={d} widths={d},{d},{d},{d},{d},{d},{d},{d}", .{
            snapshot.total,   snapshot.bins[0], snapshot.bins[1], snapshot.bins[2], snapshot.bins[3],
            snapshot.bins[4], snapshot.bins[5], snapshot.bins[6], snapshot.bins[7],
        });
    }
    fn validateView(plane_rows: usize, row0: usize, rows: usize) !void {
        if (row0 > plane_rows or rows > plane_rows - row0) return error.BadDraftWindow;
    }
    fn pack(d: *DFlash, destination: u64, source: u64, plane_rows: usize, row0: usize, rows: usize) !void {
        if (rows == 0) return;
        for (0..5) |plane| {
            const copy: cuda.abi.Memcpy2D = .{
                .src = source + state.tapOffset(plane_rows, row0, plane, 0),
                .src_pitch = 8192,
                .dst = destination + plane * 8192,
                .dst_pitch = 5 * 8192,
                .width = 8192,
                .height = rows,
            };
            try d.forward.d.check(d.forward.d.api.cuMemcpy2DAsync_v2(&copy, d.forward.stream.handle), "DFlash pack taps");
        }
    }
    pub fn stageWindow(d: *DFlash, slot: usize, source: u64, plane_rows: usize, row0: usize, rows: usize, start: usize) !void {
        const l = try d.lane(slot);
        try l.validateStage(start, rows, d.capacity);
        try validateView(plane_rows, row0, rows);
        errdefer l.healthy = false;
        try d.pack(d.pending[slot].ptr, source, plane_rows, row0, rows);
        l.pending = .{ .start = start, .rows = rows };
    }
    fn absorb(d: *DFlash, slot: usize, source: u64, rows: usize, start: usize) !void {
        if (rows == 0) return;
        const bucket = std.mem.alignForward(usize, rows, 8);
        var positions: [64]i64 = undefined;
        var slots: [64]i64 = undefined;
        for (0..rows) |i| {
            positions[i] = @intCast(start + i);
            slots[i] = @intCast(slot);
        }
        for (rows..bucket) |i| {
            positions[i] = @intCast(i - rows);
            slots[i] = 8;
        }
        if (source != d.forward.tap_in) try d.forward.th.base.copy(d.forward.tap_in, source, rows * 5 * 8192);
        if (bucket > rows) try d.forward.th.base.fill32(d.forward.tap_in + rows * 5 * 8192, 0, (bucket - rows) * 5 * 4096 / 2);
        try d.forward.stage(d.forward.rowpos, std.mem.sliceAsBytes(positions[0..bucket]));
        try d.forward.stage(d.forward.rowslots, std.mem.sliceAsBytes(slots[0..bucket]));
        try d.forward.tapsPass(bucket);
        try d.forward.check();
    }
    pub fn accept(d: *DFlash, slot: usize, kept: usize, target_end: usize) !void {
        const l = try d.lane(slot);
        try l.validateAccept(kept, target_end);
        try d.invalidateContext(slot);
        errdefer l.healthy = false;
        try d.absorb(slot, d.pending[slot].ptr, kept, l.end);
        l.end = target_end;
        l.pending = null;
    }
    pub fn acceptPrompt(d: *DFlash, slot: usize, source: u64, plane_rows: usize, rows: usize, start: usize) !void {
        const l = try d.lane(slot);
        if (!l.healthy) return error.DraftFailed;
        if (l.pending != null or l.end != start or start > d.capacity or rows > d.capacity - start) return error.BadDraftCommit;
        try d.invalidateContext(slot);
        try validateView(plane_rows, 0, rows);
        errdefer l.healthy = false;
        var at: usize = 0;
        while (at < rows) {
            const take = @min(rows - at, 64);
            try d.pack(d.forward.tap_in, source, plane_rows, at, take);
            try d.absorb(slot, d.forward.tap_in, take, start + at);
            at += take;
        }
        l.end = start + rows;
    }
    pub fn validateBatch(d: *const DFlash, requests: []const BatchRequest) !void {
        _ = try batch.Plan.buildWithPlan(&d.lanes, d.capacity, requests, d.verify_plan);
    }
    pub fn proposeBatch(d: *DFlash, requests: []const BatchRequest, out: []BatchResult) !void {
        return d.finishBatch(requests, out);
    }
    fn account(d: *DFlash, plan: batch.Plan) void {
        d.width_histogram.record(plan.count);
        if (d.width_histogram.due()) d.flushWidths();
        if (plan.count > d.max_packed_width) {
            d.max_packed_width = plan.count;
            if (d.forward.rank.rank == 0) std.log.info("DFlash packed proposal width: active_count={d} requested_count={d}", .{ plan.count, plan.total });
        }
    }
    fn readAndSelect(d: *DFlash, plan: batch.Plan, out: []BatchResult) !void {
        var candidates: [4 * batch.max_lanes * 7 * 32]f32 = undefined;
        var projection: [batch.max_lanes * 7 * 256]f32 = undefined;
        const candidate_slice = candidates[0 .. d.forward.rank.world * plan.count * 7 * 32];
        const projection_slice = projection[0 .. plan.count * 7 * 256];
        if (d.forward.asyncOk()) {
            try d.forward.queueProposal(plan.count);
            try d.synchronize();
            try d.forward.settle(true);
            @memcpy(candidate_slice, d.forward.candidatesHost(plan.count));
            @memcpy(projection_slice, d.forward.projectionHost(plan.count));
        } else {
            try d.forward.check();
            try d.forward.read(d.forward.candidates, std.mem.sliceAsBytes(candidate_slice));
            try d.forward.read(d.forward.projection, std.mem.sliceAsBytes(projection_slice));
        }
        d.account(plan);
        try batch.select(plan, candidate_slice, projection_slice, d.forward.rank.world, .{ .pred = d.pred, .succ = d.succ }, out);
    }
    fn computeBatch(d: *DFlash, plan: batch.Plan, out: []BatchResult) !void {
        if (plan.count == 0) {
            @memset(out, .{});
            return;
        }
        errdefer {
            for (plan.requests[0..plan.count]) |r| d.lanes[r.slot].healthy = false;
        }
        const meta = plan.metadata(d.forward.cfg.mask);
        try d.forward.stage(d.forward.ids, std.mem.sliceAsBytes(meta.ids[0 .. plan.count * 8]));
        try d.forward.stage(d.forward.rowpos, std.mem.sliceAsBytes(meta.positions[0 .. plan.count * 8]));
        try d.forward.stage(d.forward.rowslots, std.mem.sliceAsBytes(meta.row_slots[0 .. plan.count * 8]));
        try d.forward.stage(d.forward.bpos, std.mem.sliceAsBytes(meta.begins[0..plan.count]));
        try d.forward.stage(d.forward.bslots, std.mem.sliceAsBytes(meta.block_slots[0..plan.count]));
        try d.forward.block(plan.count);
        try d.readAndSelect(plan, out);
    }
    pub fn prelaunchBatch(d: *DFlash, requests: []const BatchRequest) !void {
        const plan = try batch.Plan.buildWithPlan(&d.lanes, d.capacity, requests, d.verify_plan);
        if (plan.count == 0) {
            try d.settlePending();
            return;
        }
        const key = d.blockKey(plan);
        switch (d.pending_block) {
            .queued => |pending| if (pending.key.matches(key)) return,
            .idle => {},
        }
        try d.settlePending();
        if (!d.asyncAvailable()) return error.AsyncReadbackUnavailable;
        errdefer {
            for (plan.requests[0..plan.count]) |r| d.lanes[r.slot].healthy = false;
        }
        const meta = plan.metadata(d.forward.cfg.mask);
        try d.forward.stage(d.forward.ids, std.mem.sliceAsBytes(meta.ids[0 .. plan.count * 8]));
        try d.forward.stage(d.forward.rowpos, std.mem.sliceAsBytes(meta.positions[0 .. plan.count * 8]));
        try d.forward.stage(d.forward.rowslots, std.mem.sliceAsBytes(meta.row_slots[0 .. plan.count * 8]));
        try d.forward.stage(d.forward.bpos, std.mem.sliceAsBytes(meta.begins[0..plan.count]));
        try d.forward.stage(d.forward.bslots, std.mem.sliceAsBytes(meta.block_slots[0..plan.count]));
        try d.forward.block(plan.count);
        try d.forward.queueProposal(plan.count);
        d.pending_block = .{ .queued = .{ .key = key, .plan = plan } };
        d.prelaunch_stats.launched += 1;
        if (d.forward.rank.rank == 0 and (d.prelaunch_stats.launched == 1 or d.prelaunch_stats.launched % 64 == 0)) {
            std.log.info("GLM DFlash prelaunch launched={d} used={d} discarded={d}", .{ d.prelaunch_stats.launched, d.prelaunch_stats.used, d.prelaunch_stats.discarded });
        }
    }
    pub fn finishBatch(d: *DFlash, requests: []const BatchRequest, out: []BatchResult) !void {
        if (out.len != requests.len) return error.BadDraftBatch;
        var results: [batch.max_lanes]BatchResult = @splat(.{});
        try d.finishBatchInto(requests, results[0..requests.len]);
        @memcpy(out, results[0..requests.len]);
    }
    fn finishBatchInto(d: *DFlash, requests: []const BatchRequest, out: []BatchResult) !void {
        const plan = try batch.Plan.buildWithPlan(&d.lanes, d.capacity, requests, d.verify_plan);
        if (plan.count == 0) {
            try d.settlePending();
            @memset(out, .{});
            return;
        }
        const key = d.blockKey(plan);
        switch (d.pending_block) {
            .queued => |pending| if (pending.key.matches(key)) {
                d.prelaunch_stats.used += 1;
                d.pending_block = .idle;
                errdefer {
                    for (plan.requests[0..plan.count]) |r| d.lanes[r.slot].healthy = false;
                }
                try d.synchronize();
                try d.forward.settle(true);
                const candidate_slice = d.forward.candidatesHost(plan.count);
                const projection_slice = d.forward.projectionHost(plan.count);
                d.account(plan);
                try batch.select(plan, candidate_slice, projection_slice, d.forward.rank.world, .{ .pred = d.pred, .succ = d.succ }, out);
                return;
            },
            .idle => {},
        }
        try d.settlePending();
        try d.computeBatch(plan, out);
    }
    pub fn propose(d: *DFlash, slot: usize, pending: u32, sampling: ?Sampling, out: []u32) !usize {
        if (out.len > batch.max_depth) return error.BadDraftRequest;
        const l = try d.lane(slot);
        var result: [1]BatchResult = undefined;
        try d.proposeBatch(&.{.{ .slot = slot, .end = l.end, .pending = pending, .depth = out.len, .sampling = sampling }}, &result);
        @memcpy(out[0..result[0].count], result[0].tokens[0..result[0].count]);
        return result[0].count;
    }
    pub fn snapshotBytes(d: *const DFlash) usize {
        return 5 * 2 * d.forward.rank.kv * config.Config.ring * 256;
    }
    pub fn snapshotReady(d: *const DFlash, slot: usize, end: usize) bool {
        return slot < 8 and d.lanes[slot].ready(end);
    }
    pub fn residentBytes(d: *const DFlash) usize {
        var bytes: usize = 0;
        for (d.loader.owned.items) |b| bytes += b.len;
        for (d.forward.owned.items) |b| bytes += b.len;
        for (d.pending) |b| bytes += b.len;
        return bytes;
    }
    fn transfer(d: *DFlash, slot: usize, buffer: u64, restore_: bool) !void {
        const bytes = config.Config.ring * 256;
        var at: usize = 0;
        for (0..5) |layer| for ([_]u64{ d.forward.kc[layer], d.forward.vc[layer] }) |ring| for (0..d.forward.rank.kv) |head| {
            const ptr = ring + state.ringOffset(8, slot, head);
            try d.forward.th.base.copy(if (restore_) ptr else buffer + at, if (restore_) buffer + at else ptr, bytes);
            at += bytes;
        };
        try d.synchronize();
    }
    pub fn save(d: *DFlash, slot: usize, destination: u64) !void {
        const l = try d.lane(slot);
        if (!l.ready(l.end)) return error.DraftSnapshotNotReady;
        try d.settlePending();
        try d.transfer(slot, destination, false);
    }
    pub fn restore(d: *DFlash, slot: usize, source: u64, end: usize) !void {
        const l = try d.lane(slot);
        if (end > d.capacity) return error.BadDraftCapacity;
        try d.invalidateContext(slot);
        l.healthy = false;
        l.pending = null;
        try d.transfer(slot, source, true);
        l.* = .{ .end = end };
    }
};

test {
    _ = state;
    _ = batch;
    _ = pinned_aot;
    _ = config;
    _ = weights;
    _ = policy;
    _ = @import("cuda_dflash_quantize.zig");
    _ = @import("cuda_dflash_stage.zig");
    std.testing.refAllDecls(DFlash);
    std.testing.refAllDecls(Forward);
}

fn testPackage(driver: *const cuda.Driver) DFlash {
    var d: DFlash = undefined;
    const stream: cuda.Stream = .{ .d = driver, .handle = null };
    d.capacity = 8192;
    d.verify_plan = .baseline;
    d.lanes = @splat(.{});
    d.generations = @splat(0);
    d.pending_block = .idle;
    d.prelaunch_stats = .{};
    d.max_packed_width = 0;
    d.width_histogram = .{};
    d.forward.d = driver;
    d.forward.comm = null;
    d.forward.stream = stream;
    d.forward.th = .{ .ops = undefined, .s = stream, .base = .{ .f = undefined, .s = stream } };
    d.forward.rank = config.Rank.init(2, 0) catch unreachable;
    return d;
}

test "two and eight packed lanes launch one graph and read one invalid flag" {
    const Memory = @import("cuda_test_driver.zig").Memory;
    var driver = Memory.driver();
    const Mock = struct {
        var launches: usize = 0;
        var reads: usize = 0;
        var invalid: u64 = 0;
        var download: @TypeOf(driver.api.cuMemcpyDtoH_v2) = undefined;
        fn graph(_: cuda.abi.GraphExec, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
            launches += 1;
            @memset(Memory.slice(invalid, 4), 1);
            return 0;
        }
        fn read(dst: ?*anyopaque, src: u64, n: usize) callconv(.c) cuda.abi.Result {
            reads += 1;
            return download(dst, src, n);
        }
        fn upload(dst: u64, src: ?*const anyopaque, n: usize) callconv(.c) cuda.abi.Result {
            @memcpy(Memory.slice(dst, n), @as([*]const u8, @ptrCast(src.?))[0..n]);
            return 0;
        }
    };
    Mock.download = driver.api.cuMemcpyDtoH_v2;
    driver.api.cuGraphLaunch = Mock.graph;
    driver.api.cuMemcpyDtoH_v2 = Mock.read;
    driver.api.cuMemcpyHtoD_v2 = Mock.upload;
    for ([_]usize{ 2, 8 }) |count| {
        var d = testPackage(&driver);
        d.forward.cfg = .{};
        d.forward.streams = 8;
        d.forward.stager = .{};
        d.forward.pin = null;
        d.forward.blocks = @splat(null);
        d.forward.blocks[count - 1] = .{ .d = &driver, .handle = @ptrFromInt(count) };
        var scratch = try cuda.DeviceBuffer.alloc(&driver, 4096);
        defer scratch.free();
        d.forward.ids = scratch.ptr;
        d.forward.rowpos = scratch.ptr + 256;
        d.forward.rowslots = scratch.ptr + 768;
        d.forward.bpos = scratch.ptr + 1280;
        d.forward.bslots = scratch.ptr + 1408;
        d.forward.invalid = scratch.ptr + 1536;
        Mock.invalid = d.forward.invalid;
        Mock.launches = 0;
        Mock.reads = 0;
        var requests: [8]BatchRequest = undefined;
        for (requests[0..count], 0..) |*r, i| {
            const slot = (i * 3) % 8;
            d.lanes[slot].end = i + 10;
            r.* = .{ .slot = slot, .end = i + 10, .pending = @intCast(i + 1), .depth = 2, .sampling = null };
        }
        var results: [8]BatchResult = undefined;
        try std.testing.expectError(error.BadDraftIndex, d.proposeBatch(requests[0..count], results[0..count]));
        try std.testing.expectEqual(@as(usize, 1), Mock.launches);
        try std.testing.expectEqual(@as(usize, 1), Mock.reads);
        try std.testing.expectEqual(@as(u64, 0), d.width_histogram.total);
        try std.testing.expectEqual(@as(usize, 0), d.max_packed_width);
        for (requests[0..count]) |r| try std.testing.expect(!d.lanes[r.slot].healthy);
        Mock.launches = 0;
        Mock.reads = 0;
        for (requests[0..count]) |r| d.lanes[r.slot].healthy = true;
        requests[count - 1].slot = requests[0].slot;
        try std.testing.expectError(error.BadDraftSlot, d.proposeBatch(requests[0..count], results[0..count]));
        try std.testing.expectEqual(@as(usize, 0), Mock.launches);
        try std.testing.expectEqual(@as(usize, 0), Mock.reads);
        try std.testing.expectEqual(@as(u64, 0), d.width_histogram.total);
    }
}

test "width histogram snapshots first and every 64 completions and flushes only dirty state" {
    var h: WidthHistogram = .{};
    try std.testing.expect(h.snapshot() == null);
    h.record(2);
    try std.testing.expect(h.due());
    const first = h.snapshot().?;
    try std.testing.expectEqual(@as(u64, 1), first.total);
    try std.testing.expectEqualSlices(u64, &.{ 0, 1, 0, 0, 0, 0, 0, 0 }, &first.bins);
    try std.testing.expect(h.snapshot() == null);
    for (2..65) |total| {
        h.record(8);
        try std.testing.expectEqual(total == 64, h.due());
    }
    const interval = h.snapshot().?;
    try std.testing.expectEqual(@as(u64, 64), interval.total);
    try std.testing.expectEqualSlices(u64, &.{ 0, 1, 0, 0, 0, 0, 0, 63 }, &interval.bins);
    h.record(1);
    try std.testing.expect(!h.due());
    const tail = h.snapshot().?;
    try std.testing.expectEqual(@as(u64, 65), tail.total);
    try std.testing.expectEqualSlices(u64, &.{ 1, 1, 0, 0, 0, 0, 0, 63 }, &tail.bins);
    try std.testing.expect(h.snapshot() == null);
}

test "release flushes dirty rank zero widths without resetting lifetime counts" {
    const Memory = @import("cuda_test_driver.zig").Memory;
    var driver = Memory.driver();
    var d = testPackage(&driver);
    d.width_histogram.record(3);
    d.release(2);
    try std.testing.expectEqual(@as(u64, 1), d.width_histogram.emitted);
    try std.testing.expect(!d.lanes[2].healthy);
    d.release(2);
    try std.testing.expect(d.width_histogram.snapshot() == null);
    d.width_histogram.record(8);
    d.release(7);
    try std.testing.expectEqual(@as(u64, 2), d.width_histogram.total);
    try std.testing.expectEqual(@as(u64, 2), d.width_histogram.emitted);
    try std.testing.expectEqualSlices(u64, &.{ 0, 0, 1, 0, 0, 0, 0, 1 }, &d.width_histogram.bins);
    d.forward.rank = try config.Rank.init(2, 1);
    d.width_histogram.record(1);
    d.release(1);
    try std.testing.expectEqual(@as(u64, 2), d.width_histogram.emitted);
}

test "completed readbacks count once before CPU selection and failed GPU work never counts" {
    const Memory = @import("cuda_test_driver.zig").Memory;
    var driver = Memory.driver();
    const Mock = struct {
        const Failure = enum { none, launch, sync, flag, invalid, candidates, projection, selection };
        var failure: Failure = .none;
        var invalid: u64 = 0;
        var candidates: u64 = 0;
        var projection: u64 = 0;
        var histogram: *const WidthHistogram = undefined;
        var baseline: u64 = 0;
        var premature: bool = false;
        var launches: usize = 0;
        var reads: usize = 0;
        var syncs: usize = 0;
        var download: @TypeOf(driver.api.cuMemcpyDtoH_v2) = undefined;
        fn observe() void {
            premature = premature or histogram.total != baseline;
        }
        fn graph(_: cuda.abi.GraphExec, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
            observe();
            launches += 1;
            if (failure == .launch) return cuda.abi.error_not_ready;
            @memset(Memory.slice(invalid, 4), if (failure == .invalid) 1 else 0);
            @memset(Memory.slice(candidates, 2 * 2 * 7 * 32 * 4), 0);
            @memset(Memory.slice(projection, 2 * 7 * 256 * 4), 0);
            if (failure == .selection) @memset(Memory.slice(candidates, 4), 0xff);
            return 0;
        }
        fn sync(_: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
            observe();
            syncs += 1;
            return if (failure == .sync and launches != 0) cuda.abi.error_not_ready else 0;
        }
        fn read(dst: ?*anyopaque, src: u64, n: usize) callconv(.c) cuda.abi.Result {
            observe();
            reads += 1;
            if ((failure == .flag and src == invalid) or (failure == .candidates and src == candidates) or (failure == .projection and src == projection)) return cuda.abi.error_not_ready;
            return download(dst, src, n);
        }
        fn readAsync(dst: ?*anyopaque, src: u64, n: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
            return read(dst, src, n);
        }
        fn upload(dst: u64, src: ?*const anyopaque, n: usize) callconv(.c) cuda.abi.Result {
            @memcpy(Memory.slice(dst, n), @as([*]const u8, @ptrCast(src.?))[0..n]);
            return 0;
        }
    };
    Mock.download = driver.api.cuMemcpyDtoH_v2;
    driver.api.cuGraphLaunch = Mock.graph;
    driver.api.cuStreamSynchronize = Mock.sync;
    driver.api.cuMemcpyDtoH_v2 = Mock.read;
    driver.api.cuMemcpyDtoHAsync_v2 = Mock.readAsync;
    driver.api.cuMemcpyHtoD_v2 = Mock.upload;
    var scratch = try cuda.DeviceBuffer.alloc(&driver, 96 * 1024);
    defer scratch.free();
    const host = try std.testing.allocator.alignedAlloc(u8, .fromByteUnits(16), 96 * 1024);
    defer std.testing.allocator.free(host);
    const staging = try std.testing.allocator.alignedAlloc(u8, .fromByteUnits(16), 16 * 1024);
    defer std.testing.allocator.free(staging);
    const codes: [256]f32 = @splat(0);
    for ([_]bool{ false, true }) |async_read| {
        for (std.enums.values(Mock.Failure)) |failure| {
            var d = testPackage(&driver);
            d.pred = @constCast(&codes);
            d.succ = @constCast(&codes);
            d.forward.cfg = .{};
            d.forward.streams = 8;
            d.forward.stager = .{};
            if (async_read) d.forward.stager = .{ .stream = d.forward.stream, .host = .{ .d = &driver, .bytes = staging } };
            d.forward.pin = if (async_read) .{ .d = &driver, .bytes = host } else null;
            d.forward.tap_pending = false;
            d.forward.blocks = @splat(null);
            d.forward.blocks[1] = .{ .d = &driver, .handle = @ptrFromInt(2) };
            d.forward.ids = scratch.ptr;
            d.forward.rowpos = scratch.ptr + 256;
            d.forward.rowslots = scratch.ptr + 768;
            d.forward.bpos = scratch.ptr + 1280;
            d.forward.bslots = scratch.ptr + 1408;
            d.forward.invalid = scratch.ptr + 1536;
            d.forward.candidates = scratch.ptr + 2048;
            d.forward.projection = scratch.ptr + 16384;
            Mock.invalid = d.forward.invalid;
            Mock.candidates = d.forward.candidates;
            Mock.projection = d.forward.projection;
            Mock.histogram = &d.width_histogram;
            Mock.baseline = 0;
            Mock.premature = false;
            Mock.launches = 0;
            Mock.reads = 0;
            Mock.syncs = 0;
            Mock.failure = failure;
            d.lanes[0].end = 10;
            d.lanes[3].end = 20;
            const requests = [_]BatchRequest{
                .{ .slot = 0, .end = 10, .pending = 0, .depth = 1, .sampling = null },
                .{ .slot = 3, .end = 20, .pending = 0, .depth = 1, .sampling = null },
                .{ .slot = 5, .end = 0, .pending = 0, .depth = 0, .sampling = null },
            };
            var results: [3]BatchResult = @splat(.{ .count = 99 });
            switch (failure) {
                .none => {
                    try d.proposeBatch(&requests, &results);
                    try std.testing.expectEqual(@as(usize, 1), results[0].count);
                    try std.testing.expectEqual(@as(usize, 1), results[1].count);
                    try std.testing.expectEqual(@as(usize, 0), results[2].count);
                },
                .selection => try std.testing.expectError(error.BadCandidates, d.proposeBatch(&requests, &results)),
                .invalid => try std.testing.expectError(error.BadDraftIndex, d.proposeBatch(&requests, &results)),
                else => try std.testing.expectError(error.NotReady, d.proposeBatch(&requests, &results)),
            }
            const completed = failure == .none or failure == .selection;
            try std.testing.expectEqual(@as(u64, if (completed) 1 else 0), d.width_histogram.total);
            try std.testing.expectEqual(@as(u64, if (completed) 1 else 0), d.width_histogram.bins[1]);
            try std.testing.expectEqual(@as(usize, if (completed) 2 else 0), d.max_packed_width);
            try std.testing.expect(!Mock.premature);
            if (failure != .none) for (results) |r| try std.testing.expectEqual(@as(usize, 99), r.count);
            const launches = Mock.launches;
            const reads = Mock.reads;
            Mock.baseline = d.width_histogram.total;
            try d.proposeBatch(&.{requests[2]}, results[0..1]);
            try std.testing.expectEqual(Mock.baseline, d.width_histogram.total);
            try std.testing.expectEqual(launches, Mock.launches);
            try std.testing.expectEqual(reads, Mock.reads);
            if (async_read and failure == .none) {
                const before_syncs = Mock.syncs;
                try d.prelaunchBatch(&requests);
                try std.testing.expectEqual(before_syncs, Mock.syncs);
                try std.testing.expectEqual(launches + 1, Mock.launches);
                try d.prelaunchBatch(&requests);
                try std.testing.expectEqual(launches + 1, Mock.launches);
                try d.finishBatch(&requests, &results);
                try std.testing.expectEqual(before_syncs + 1, Mock.syncs);
                try std.testing.expectEqual(launches + 1, Mock.launches);
                try std.testing.expectEqual(@as(usize, 1), results[0].count);
                try std.testing.expectEqual(@as(usize, 0), results[2].count);
                try std.testing.expectEqual(@as(u64, 1), d.prelaunch_stats.used);
                Mock.baseline = d.width_histogram.total;
                try d.prelaunchBatch(&requests);
                d.release(0);
                try std.testing.expectEqual(@as(u64, 1), d.prelaunch_stats.discarded);
                try std.testing.expectEqual(PendingBlock.idle, d.pending_block);
                try std.testing.expect(!d.lanes[0].healthy);
                try std.testing.expectEqual(@as(u64, 1), d.generations[0]);
            }
        }
    }
}

test "staging copies strided planes into independent owned lane windows" {
    const Memory = @import("cuda_test_driver.zig").Memory;
    var driver = Memory.driver();
    const Copies = struct {
        var calls: usize = 0;
        var delegate: @TypeOf(driver.api.cuMemcpy2DAsync_v2) = undefined;
        fn copy(transfer: *const cuda.abi.Memcpy2D, stream: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
            calls += 1;
            return delegate(transfer, stream);
        }
    };
    Copies.calls = 0;
    Copies.delegate = driver.api.cuMemcpy2DAsync_v2;
    driver.api.cuMemcpy2DAsync_v2 = Copies.copy;
    var d = testPackage(&driver);
    var source = try cuda.DeviceBuffer.alloc(&driver, 5 * 32 * 8192);
    defer source.free();
    var a = try cuda.DeviceBuffer.alloc(&driver, 32 * 5 * 8192);
    defer a.free();
    var b = try cuda.DeviceBuffer.alloc(&driver, 32 * 5 * 8192);
    defer b.free();
    d.pending[3] = a;
    d.pending[5] = b;
    for (0..5) |plane| for (0..32) |row| {
        @memset(Memory.slice(source.ptr + (plane * 32 + row) * 8192, 8192), @intCast(plane * 32 + row));
    };
    try d.stageWindow(3, source.ptr, 32, 2, 4, 0);
    try d.stageWindow(5, source.ptr, 32, 8, 3, 0);
    try std.testing.expectEqual(@as(usize, 10), Copies.calls);
    @memset(Memory.slice(source.ptr, source.len), 0xff);
    for (0..4) |row| for (0..5) |plane| {
        try std.testing.expect(std.mem.allEqual(u8, Memory.slice(a.ptr + (row * 5 + plane) * 8192, 8192), @intCast(plane * 32 + row + 2)));
    };
    for (0..3) |row| for (0..5) |plane| {
        try std.testing.expect(std.mem.allEqual(u8, Memory.slice(b.ptr + (row * 5 + plane) * 8192, 8192), @intCast(plane * 32 + row + 8)));
    };
    try std.testing.expectError(error.BadDraftWindow, d.stageWindow(3, source.ptr, 32, 0, 1, 0));
    try d.accept(3, 0, 0);
    try std.testing.expect(d.snapshotReady(3, 0));
    try std.testing.expect(!d.snapshotReady(5, 0));
    try std.testing.expectError(error.NoDraftWindow, d.accept(3, 0, 0));
}

test "snapshots restore every KV ring head and wrapped tail into a different slot" {
    const Memory = @import("cuda_test_driver.zig").Memory;
    var driver = Memory.driver();
    var d = testPackage(&driver);
    const head_bytes = 2176 * 256;
    const layer_bytes = 4 * (8 * 2176 + 64) * 256;
    var rings = try cuda.DeviceBuffer.alloc(&driver, 10 * layer_bytes);
    defer rings.free();
    @memset(Memory.slice(rings.ptr, rings.len), 0x3c);
    for (0..5) |layer| {
        d.forward.kc[layer] = rings.ptr + layer * 2 * layer_bytes;
        d.forward.vc[layer] = rings.ptr + (layer * 2 + 1) * layer_bytes;
        for ([_]u64{ d.forward.kc[layer], d.forward.vc[layer] }, 0..) |ring, kind| for (0..4) |head| {
            const region = Memory.slice(ring + state.ringOffset(8, 3, head), head_bytes);
            @memset(region, @intCast(1 + layer * 8 + kind * 4 + head));
            @memset(region[head_bytes - 256 ..], @intCast(128 + layer * 8 + kind * 4 + head));
        };
    }
    var snapshot = try cuda.DeviceBuffer.alloc(&driver, d.snapshotBytes());
    defer snapshot.free();
    try std.testing.expectEqual(@as(usize, 22_282_240), snapshot.len);
    d.lanes[3].end = 4000;
    try d.save(3, snapshot.ptr);
    d.lanes[5].pending = .{ .start = 0, .rows = 4 };
    try d.restore(5, snapshot.ptr, 4000);
    try std.testing.expect(d.snapshotReady(5, 4000));
    for (0..5) |layer| for ([_]u64{ d.forward.kc[layer], d.forward.vc[layer] }) |ring| for (0..4) |head| {
        try std.testing.expectEqualSlices(u8, Memory.slice(ring + state.ringOffset(8, 3, head), head_bytes), Memory.slice(ring + state.ringOffset(8, 5, head), head_bytes));
        try std.testing.expect(std.mem.allEqual(u8, Memory.slice(ring + state.ringOffset(8, 1, head), head_bytes), 0x3c));
    };
    const Failure = struct {
        var calls: usize = 0;
        fn copy(dst: u64, src: u64, n: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
            calls += 1;
            if (calls == 2) return cuda.abi.error_not_ready;
            @memcpy(Memory.slice(dst, n), Memory.slice(src, n));
            return 0;
        }
    };
    Failure.calls = 0;
    driver.api.cuMemcpyDtoDAsync_v2 = Failure.copy;
    try std.testing.expectError(error.NotReady, d.restore(5, snapshot.ptr, 5000));
    try std.testing.expectEqual(@as(usize, 4000), d.lanes[5].end);
    try std.testing.expect(!d.snapshotReady(5, 4000));
    d.width_histogram.record(4);
    try d.reset(5);
    try std.testing.expect(d.snapshotReady(5, 0));
    try std.testing.expectEqual(@as(u64, 1), d.width_histogram.total);
    try std.testing.expectEqualSlices(u64, &.{ 0, 0, 0, 1, 0, 0, 0, 0 }, &d.width_histogram.bins);
}
