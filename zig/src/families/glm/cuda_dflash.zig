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
const pinned_aot = @import("cuda_dflash_aot.zig");
const Sampling = @import("cuda_sampler.zig").Sampling;

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
};

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
    lanes: [8]state.Lane = @splat(.{}),
    pending: [8]cuda.DeviceBuffer,

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
        p.lanes = @splat(.{});
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
        p.forward = try Forward.init(gpa, deps.driver, deps.stream, cfg, rank, streams, w, deps.embed, .{ .set = &p.aot, .s = deps.stream }, p.ops.on(deps.stream), &p.entries, .{ .lib = deps.lib.?, .comm = deps.comm, .rank = deps.rank, .world = deps.world }, &p.lt);
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
    pub fn deinit(d: *DFlash) void {
        d.forward.stream.synchronize() catch {};
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
    pub fn reset(d: *DFlash, slot: usize) !void {
        const l = try d.lane(slot);
        l.healthy = false;
        const bytes = config.Config.ring * 256;
        for (0..5) |layer| for (0..d.forward.rank.kv) |head| {
            const offset = state.ringOffset(8, slot, head);
            try d.forward.th.base.fill32(d.forward.kc[layer] + offset, 0, bytes / 4);
            try d.forward.th.base.fill32(d.forward.vc[layer] + offset, 0, bytes / 4);
        };
        try d.forward.stream.synchronize();
        l.* = .{};
    }
    pub fn release(d: *DFlash, slot: usize) void {
        if (slot < 8) d.lanes[slot] = .{ .healthy = false };
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
        errdefer l.healthy = false;
        try d.absorb(slot, d.pending[slot].ptr, kept, l.end);
        l.end = target_end;
        l.pending = null;
    }
    pub fn acceptPrompt(d: *DFlash, slot: usize, source: u64, plane_rows: usize, rows: usize, start: usize) !void {
        const l = try d.lane(slot);
        if (!l.healthy) return error.DraftFailed;
        if (l.pending != null or l.end != start or start > d.capacity or rows > d.capacity - start) return error.BadDraftCommit;
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
    pub fn propose(d: *DFlash, slot: usize, pending: u32, sampling: ?Sampling, out: []u32) !usize {
        const l = try d.lane(slot);
        if (!l.healthy) return error.DraftFailed;
        if (l.pending != null or pending >= 154880) return error.BadDraftRequest;
        if (l.end == 0 or l.end >= d.capacity or out.len == 0) return 0;
        const count = @min(@min(out.len, 3), d.capacity - l.end - 1);
        if (count == 0) return 0;
        errdefer l.healthy = false;
        var ids: [8]u32 = @splat(d.forward.cfg.mask);
        ids[0] = pending;
        var pos: [8]i64 = undefined;
        const slots: [8]i64 = @splat(@intCast(slot));
        const begin: [1]i64 = .{@intCast(l.end)};
        const slot_ids: [1]i64 = .{@intCast(slot)};
        for (&pos, 0..) |*p, i| p.* = @intCast(l.end + i);
        try d.forward.stage(d.forward.ids, std.mem.sliceAsBytes(&ids));
        try d.forward.stage(d.forward.rowpos, std.mem.sliceAsBytes(&pos));
        try d.forward.stage(d.forward.rowslots, std.mem.sliceAsBytes(&slots));
        try d.forward.stage(d.forward.bpos, std.mem.sliceAsBytes(&begin));
        try d.forward.stage(d.forward.bslots, std.mem.sliceAsBytes(&slot_ids));
        try d.forward.block(1);
        var candidates: [4 * 7 * 32]f32 = undefined;
        var projection: [7 * 256]f32 = undefined;
        const candidate_slice = candidates[0 .. d.forward.rank.world * 7 * 32];
        if (d.forward.asyncOk()) {
            try d.forward.queueProposal(1);
            try d.forward.stream.synchronize();
            try d.forward.settle(true);
            @memcpy(candidate_slice, d.forward.candidatesHost(1));
            @memcpy(&projection, d.forward.projectionHost(1));
        } else {
            try d.forward.check();
            try d.forward.read(d.forward.candidates, std.mem.sliceAsBytes(candidate_slice));
            try d.forward.read(d.forward.projection, std.mem.sliceAsBytes(&projection));
        }
        var merged: [7 * 16]policy.Candidate = undefined;
        try policy.merge(candidate_slice, d.forward.rank.world, 7, 16, 154880, &merged);
        return policy.chain(&merged, &projection, .{ .pred = d.pred, .succ = d.succ }, pending, l.end + 1, sampling, .{}, out[0..count], null);
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
        try d.forward.stream.synchronize();
    }
    pub fn save(d: *DFlash, slot: usize, destination: u64) !void {
        const l = try d.lane(slot);
        if (!l.ready(l.end)) return error.DraftSnapshotNotReady;
        try d.transfer(slot, destination, false);
    }
    pub fn restore(d: *DFlash, slot: usize, source: u64, end: usize) !void {
        const l = try d.lane(slot);
        if (end > d.capacity) return error.BadDraftCapacity;
        l.healthy = false;
        l.pending = null;
        try d.transfer(slot, source, true);
        l.* = .{ .end = end };
    }
};

test {
    _ = state;
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
    d.lanes = @splat(.{});
    d.forward.d = driver;
    d.forward.stream = stream;
    d.forward.th = .{ .ops = undefined, .s = stream, .base = .{ .f = undefined, .s = stream } };
    d.forward.rank = config.Rank.init(2, 0) catch unreachable;
    return d;
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
}
