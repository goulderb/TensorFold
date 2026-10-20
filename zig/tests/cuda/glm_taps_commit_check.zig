//! Isolated real-weight DFlash tap commits. No target head or draft decoder is loaded.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const parts = @import("glm_cuda").taps_fixture;
const check = @import("check.zig");
const Forward = parts.forward.Forward;
const rows_max = 64;
const width = 5 * 4096;
const streams = 8;
const ring = parts.config.Config.ring;
const cap = streams * ring + 64;
const guard = 32;
const marker = 0xa5;
const repetitions = 5;
const calls = 10;
const Arm = enum { per_lane_sync, packed_sync, packed_deferred };
const Frame = struct { taps: u64, pos: u64, slots: u64, bucket: usize, real: usize, graph: ?cuda.graph.Exec = null };
const Cache = struct { base: u64, ptr: u64, bytes: usize };
const Memory = struct {
    gpu: check.Gpu,
    bufs: std.ArrayList(cuda.DeviceBuffer) = .empty,
    fn alloc(m: *Memory, bytes: usize) !u64 {
        var b = try cuda.DeviceBuffer.alloc(m.gpu.d, bytes);
        errdefer b.free();
        try b.fill8(0, null);
        try m.bufs.append(m.gpu.gpa, b);
        return b.ptr;
    }
    fn upload(m: *Memory, bytes: []const u8) !u64 {
        var b = try cuda.DeviceBuffer.fromHost(m.gpu.d, bytes);
        errdefer b.free();
        try m.bufs.append(m.gpu.gpa, b);
        return b.ptr;
    }
    fn deinit(m: *Memory) void {
        for (m.bufs.items) |*b| b.free();
        m.bufs.deinit(m.gpu.gpa);
    }
};

fn fill(d: *const cuda.Driver, stream: cuda.Stream, ptr: u64, bytes: usize, value: u8) !void {
    try d.check(d.api.cuMemsetD8Async(ptr, value, bytes, stream.handle), "tap probe memset");
}

fn makeFrame(m: *Memory, counts: []const usize, first_lane: usize) !Frame {
    var total: usize = 0;
    for (counts) |n| total += n;
    const bucket = std.mem.alignForward(usize, total, 8);
    const taps = try m.gpu.gpa.alloc(u16, bucket * width);
    defer m.gpu.gpa.free(taps);
    @memset(taps, 0);
    var positions: [64]i64 = @splat(0);
    var slots: [64]i64 = @splat(streams);
    var at: usize = 0;
    for (counts, 0..) |n, lane| {
        for (0..n) |row| {
            positions[at] = @intCast(ring - 4 + row);
            slots[at] = @intCast(first_lane + lane);
            for (taps[at * width ..][0..width], 0..) |*v, c| {
                const x: f32 = @as(f32, @floatFromInt(@as(i32, @intCast((c * 7 + row * 13 + (first_lane + lane) * 17) % 101)) - 50)) / 64;
                const bits: u32 = @bitCast(x);
                v.* = @truncate((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
            }
            at += 1;
        }
    }
    for (total..bucket) |row| positions[row] = @intCast(row - total);
    return .{ .taps = try m.upload(std.mem.sliceAsBytes(taps)), .pos = try m.upload(std.mem.sliceAsBytes(positions[0..bucket])), .slots = try m.upload(std.mem.sliceAsBytes(slots[0..bucket])), .bucket = bucket, .real = total };
}

fn bind(f: *Forward, frame: Frame) void {
    f.tap_in = frame.taps;
    f.rowpos = frame.pos;
    f.rowslots = frame.slots;
}

fn capture(f: *Forward, frame: *Frame) !void {
    bind(f, frame.*);
    try f.tapsCompute(frame.bucket);
    try f.check();
    try cuda.graph.beginCapture(f.stream, .global);
    f.tapsCompute(frame.bucket) catch |err| {
        var aborted = cuda.graph.endCapture(f.stream) catch return err;
        aborted.deinit();
        return err;
    };
    var graph = try cuda.graph.endCapture(f.stream);
    defer graph.deinit();
    frame.graph = try graph.instantiate();
}

fn launch(f: *Forward, frame: Frame) !void {
    bind(f, frame);
    f.taps[frame.bucket / 8 - 1] = frame.graph;
    try f.tapsPass(frame.bucket);
}

fn runArm(f: *Forward, lanes: []const Frame, packed_frame: Frame, arm: Arm) !void {
    switch (arm) {
        .per_lane_sync => for (lanes) |lane| {
            try launch(f, lane);
            try f.check();
        },
        .packed_sync => {
            try launch(f, packed_frame);
            try f.check();
        },
        .packed_deferred => {
            try launch(f, packed_frame);
            try f.tapsDone();
            try f.synchronize();
            try f.settle(false);
        },
    }
}

fn download(gpu: check.Gpu, ptr: u64, out: []u8) !void {
    const b: cuda.DeviceBuffer = .{ .d = gpu.d, .ptr = ptr, .len = out.len };
    try b.download(0, out);
}

fn validDest(row: usize, counts: []const usize) bool {
    const slot = row / ring;
    if (slot >= counts.len) return false;
    const r = row % ring;
    for (0..counts[slot]) |i| if (r == (ring - 4 + i) % ring) return true;
    return false;
}

fn verify(gpu: check.Gpu, caches: []const Cache, kv: usize, counts: []const usize, expected: []u8, save: bool) !void {
    const one = kv * streams * ring * 128 * 2;
    const raw = try gpu.gpa.alloc(u8, kv * cap * 128 * 2);
    defer gpu.gpa.free(raw);
    for (caches, 0..) |cache, i| {
        var guards: [guard]u8 = undefined;
        try download(gpu, cache.base, &guards);
        for (guards) |v| try check.expect(v == marker, "tap commit cache leading sentinel", .{});
        try download(gpu, cache.ptr + cache.bytes, &guards);
        for (guards) |v| try check.expect(v == marker, "tap commit cache trailing sentinel", .{});
        try download(gpu, cache.ptr, raw);
        for (0..kv) |head| {
            const real = raw[head * cap * 256 ..][0 .. streams * ring * 256];
            const destination = expected[i * one + head * streams * ring * 256 ..][0..real.len];
            if (save) @memcpy(destination, real) else try check.sameBytes("tap commit all real context rows", real, destination);
            for (0..streams * ring) |row| {
                const bytes = real[row * 256 ..][0..256];
                if (validDest(row, counts)) {
                    var changed = false;
                    for (bytes) |v| changed = changed or v != marker;
                    try check.expect(changed, "tap commit every real destination written", .{});
                } else {
                    for (bytes) |v| try check.expect(v == marker, "tap commit non-target context untouched", .{});
                }
            }
        }
    }
}

fn reset(f: *Forward, caches: []const Cache) !void {
    for (caches) |cache| try fill(f.d, f.stream, cache.base, cache.bytes + 2 * guard, marker);
    f.tap_pending = false;
    try f.synchronize();
}

pub fn run(gpu: check.Gpu, draft_dir: []const u8, aot_dir: []const u8, rank_id: usize) !void {
    const rank = try parts.config.Rank.init(2, rank_id);
    const cfg = try parts.config.Config.read(gpu.gpa, gpu.io, draft_dir);
    var draft = try core.Checkpoint.openModel(gpu.gpa, gpu.io, draft_dir);
    defer draft.close();
    var loader: parts.weights.Loader = .{ .gpa = gpu.gpa, .driver = gpu.d, .target = &draft, .draft = &draft, .rank = rank };
    defer loader.deinit();
    const weights = try loader.loadTaps();
    try parts.aot.validate(gpu.gpa, gpu.io, aot_dir);
    var aot = try cuda.aot.Set.load(gpu.gpa, gpu.io, gpu.d, gpu.ctx.device, aot_dir);
    defer aot.deinit();
    var entries = try parts.entries.Entries.load(gpu.d);
    defer entries.deinit();
    var ops = try parts.ops.Ops.load(gpu.ctx);
    defer ops.deinit();
    var stream = try cuda.Stream.init(gpu.d, true);
    defer stream.deinit();
    var memory: Memory = .{ .gpu = gpu };
    defer memory.deinit();
    var pin = try cuda.HostBuffer.alloc(gpu.d, 4);
    defer pin.free();
    defer stream.synchronize() catch |err| @import("glm_cuda").kv_owner.failPair(err);
    var f: Forward = undefined;
    f.gpa = gpu.gpa;
    f.d = gpu.d;
    f.stream = stream;
    f.cfg = cfg;
    f.rank = rank;
    f.streams = streams;
    f.comm = null;
    f.entries = &entries;
    f.tri = .{ .set = &aot, .s = stream };
    f.th = ops.on(stream);
    f.w = undefined;
    f.w.fc = weights.fc;
    f.w.hidden_norm = weights.hidden_norm;
    for (&f.w.layers, weights.layers) |*to, from| {
        to.* = undefined;
        to.kv = from.kv;
        to.q_norm = from.q_norm;
        to.k_norm = from.k_norm;
    }
    f.taps = @splat(null);
    f.stager = .{};
    f.pin = pin;
    f.tap_pending = false;
    try check.expect(f.asyncOk(), "tap deferred proof requires TF_GLM_SYNC_STAGE unset", .{});
    var inverse: [64]f32 = undefined;
    for (&inverse, 0..) |*v, i| v.* = 1 / std.math.pow(f32, cfg.theta, @as(f32, @floatFromInt(2 * i)) / 128);
    f.inv = try memory.upload(std.mem.sliceAsBytes(&inverse));
    f.indices = try memory.alloc(64 * 8);
    f.sums = try memory.alloc(64 * width / 64 * 4);
    f.prod = try memory.alloc(64 * 4096 * 2);
    f.normed = try memory.alloc(64 * 4096 * 2);
    f.cos = try memory.alloc(64 * 64 * 4);
    f.sin = try memory.alloc(64 * 64 * 4);
    f.q = try memory.alloc(64 * rank.heads * 128 * 2);
    f.k = try memory.alloc(64 * rank.kv * 128 * 2);
    f.v = try memory.alloc(64 * rank.kv * 128 * 2);
    f.invalid = try memory.alloc(4);
    f.part = try memory.alloc(64 * 8 * 4096 * 4);
    var caches: [10]Cache = undefined;
    for (&caches, 0..) |*cache, i| {
        cache.bytes = rank.kv * cap * 128 * 2;
        cache.base = try memory.alloc(cache.bytes + 2 * guard);
        cache.ptr = cache.base + guard;
        if (i % 2 == 0) f.kc[i / 2] = cache.ptr else f.vc[i / 2] = cache.ptr;
    }
    const cases = [_][]const usize{ &.{ 1, 1, 1, 1, 1, 1, 1, 1 }, &.{ 2, 2, 2, 2, 2, 2, 2, 2 }, &.{ 4, 4, 4, 4, 4, 4, 4, 4 }, &.{ 1, 3, 5, 7, 2, 4, 1, 1 } };
    const expected = try gpu.gpa.alloc(u8, 10 * rank.kv * streams * ring * 128 * 2);
    defer gpu.gpa.free(expected);
    for (cases, 0..) |counts, case_id| {
        var lanes: [8]Frame = undefined;
        var made: usize = 0;
        var packed_frame = try makeFrame(&memory, counts, 0);
        defer if (packed_frame.graph) |*graph| graph.deinit();
        defer for (lanes[0..made]) |*lane| if (lane.graph) |*graph| graph.deinit();
        defer stream.synchronize() catch |err| @import("glm_cuda").kv_owner.failPair(err);
        for (counts, 0..) |_, i| {
            lanes[i] = try makeFrame(&memory, counts[i..][0..1], i);
            made += 1;
            try capture(&f, &lanes[i]);
        }
        try capture(&f, &packed_frame);
        for ([_]Arm{ .per_lane_sync, .packed_sync, .packed_deferred }, 0..) |arm, i| {
            try reset(&f, &caches);
            try runArm(&f, lanes[0..counts.len], packed_frame, arm);
            try verify(gpu, &caches, rank.kv, counts, expected, i == 0);
        }
        check.pass("tap commit rank={d} case={d} counts={any}: real contexts exact, every non-target real row and outer sentinel untouched", .{ rank_id, case_id, counts });
        for (0..repetitions) |sample| {
            for (0..3) |offset| {
                const arm: Arm = @fromBackingInt(@intCast((sample + offset) % 3));
                const start = std.Io.Clock.awake.now(gpu.io).toNanoseconds();
                for (0..calls) |_| try runArm(&f, lanes[0..counts.len], packed_frame, arm);
                const end = std.Io.Clock.awake.now(gpu.io).toNanoseconds();
                std.debug.print("RESULT tap_commit rank={d} case={d} counts={any} arm={s} sample={d} calls={d} ns_per_call={d}\n", .{ rank_id, case_id, counts, @tagName(arm), sample, calls, @divTrunc(end - start, calls) });
            }
        }
        if (case_id == cases.len - 1) {
            check.pass("tap commit correctness and timing completed; deferred accumulation diagnostic follows", .{});
            try invalidProof(&f, packed_frame);
        }
    }
}

fn invalidProof(f: *Forward, frame: Frame) !void {
    var slots: [64]i64 = @splat(streams);
    var positions: [64]i64 = undefined;
    for (&positions, 0..) |*pos, i| pos.* = @intCast(i);
    const pos_buffer: cuda.DeviceBuffer = .{ .d = f.d, .ptr = frame.pos, .len = frame.bucket * 8 };
    try pos_buffer.upload(0, std.mem.sliceAsBytes(positions[0..frame.bucket]));
    slots[0] = -1;
    const b: cuda.DeviceBuffer = .{ .d = f.d, .ptr = frame.slots, .len = frame.bucket * 8 };
    try b.upload(0, std.mem.sliceAsBytes(slots[0..frame.bucket]));
    try launch(f, frame);
    try f.tapsDone();
    try f.synchronize();
    var detected = false;
    f.settle(false) catch |err| {
        if (err != error.BadDraftIndex) return err;
        detected = true;
    };
    try check.expect(detected, "tap deferred settle detects an invalid destination", .{});
    try launch(f, frame);
    try f.tapsDone();
    try f.synchronize();
    @memset(&slots, streams);
    try b.upload(0, std.mem.sliceAsBytes(slots[0..frame.bucket]));
    try launch(f, frame);
    try f.tapsDone();
    try f.synchronize();
    detected = false;
    f.settle(false) catch |err| {
        if (err != error.BadDraftIndex) return err;
        detected = true;
    };
    std.debug.print("RESULT tap_commit deferred_bad_then_valid_preserved={}\n", .{detected});
    if (!detected) std.debug.print("EXPECTED_KNOWN_NEGATIVE tap_deferred_invalid_overwritten_after_later_valid_pass\n", .{});
    try check.expect(detected, "tap deferred error survives a later valid pass before settle", .{});
}
