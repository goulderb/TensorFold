const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");

const dims = 2048;
const experts = 256;
const score_bytes = (experts + 1) * 4;
const launches = 128;
const floor_bytes = 1536 * 1024 * 1024;
pub const scope = "fixed-input L2-warm router arithmetic diagnostic; not model speed";
const FilePin = struct { bytes: usize, sha256: []const u8 };
const Manifest = struct { scope: []const u8, files: std.json.ArrayHashMap(FilePin) };
const Row = struct { input: []const u8, picks: []const u8, weights: []const u8 };
const Fixture = struct { router: []const u8, shared_gate: []const u8, rows: [6]Row, manifest_sha256: [64]u8 };
const Stage = enum { combined, products, select };
const stages = std.meta.tags(Stage);

fn hash(bytes: []const u8) [64]u8 {
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &digest, .{});
    return std.fmt.bytesToHex(digest, .lower);
}

fn verifyHash(bytes: []const u8, expected: []const u8) ![64]u8 {
    const actual = hash(bytes);
    if (!std.mem.eql(u8, &actual, expected)) return error.RouterFixtureHashMismatch;
    return actual;
}

fn finite(bytes: []const u8, fp32: bool) !void {
    const stride: usize = if (fp32) 4 else 2;
    if (bytes.len % stride != 0) return error.InvalidRouterValueSize;
    var offset: usize = 0;
    while (offset < bytes.len) : (offset += stride) {
        const bits: u32 = if (fp32)
            std.mem.readInt(u32, bytes[offset..][0..4], .little)
        else
            @as(u32, std.mem.readInt(u16, bytes[offset..][0..2], .little)) << 16;
        if (!std.math.isFinite(@as(f32, @bitCast(bits)))) return error.NonfiniteRouterValue;
    }
}

fn read(a: std.mem.Allocator, io: std.Io, dir: []const u8, manifest: Manifest, name: []const u8, size: usize) ![]const u8 {
    const pin = manifest.files.map.get(name) orelse return error.MissingRouterFixturePin;
    if (pin.bytes != size) return error.InvalidRouterFixtureSize;
    const path = try std.fs.path.join(a, &.{ dir, name });
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(size + 1));
    if (bytes.len != size) return error.InvalidRouterFixtureSize;
    _ = try verifyHash(bytes, pin.sha256);
    return bytes;
}

fn loadFixture(a: std.mem.Allocator, io: std.Io, dir: []const u8, expected: []const u8) !Fixture {
    const path = try std.fs.path.join(a, &.{ dir, "manifest.json" });
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 20));
    const digest = try verifyHash(bytes, expected);
    const manifest = try std.json.parseFromSliceLeaky(Manifest, a, bytes, .{ .ignore_unknown_fields = true });
    var fixture: Fixture = .{
        .router = try read(a, io, dir, manifest, "router.bin", experts * dims * 2),
        .shared_gate = try read(a, io, dir, manifest, "shared-gate.bin", dims * 2),
        .rows = undefined,
        .manifest_sha256 = digest,
    };
    try finite(fixture.router, false);
    try finite(fixture.shared_gate, false);
    for (&fixture.rows, 0..) |*row, position| {
        row.* = .{
            .input = try read(a, io, dir, manifest, try std.fmt.allocPrint(a, "route-{d:0>3}-input.bin", .{position}), dims * 2),
            .picks = try read(a, io, dir, manifest, try std.fmt.allocPrint(a, "route-{d:0>3}-picks.bin", .{position}), 8 * 4),
            .weights = try read(a, io, dir, manifest, try std.fmt.allocPrint(a, "route-{d:0>3}-weights.bin", .{position}), 9 * 4),
        };
        try finite(row.input, false);
        try finite(row.weights, true);
        var seen: [experts]bool = @splat(false);
        for (0..8) |slot| {
            const id = std.mem.readInt(u32, row.picks[slot * 4 ..][0..4], .little);
            if (id >= experts or seen[id]) return error.InvalidRouterFixturePicks;
            seen[id] = true;
        }
    }
    return fixture;
}

pub fn checkInput(a: std.mem.Allocator, io: std.Io, dir: []const u8, expected: []const u8) !void {
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    _ = try loadFixture(arena.allocator(), io, dir, expected);
    var poisoned: [score_bytes]u8 = @splat(0);
    @memset(poisoned[128 * 4 ..][0..4], 0xff);
    if (finite(&poisoned, true)) |_| return error.RouterNegativeControlInsensitive else |err| if (err != error.NonfiniteRouterValue) return err;
}

fn floor(gpu: check.Gpu) !void {
    if ((try gpu.ctx.memInfo()).free < floor_bytes) return error.GpuMemoryFloor;
}

const Buffers = struct { input: cuda.DeviceBuffer, router: cuda.DeviceBuffer, shared_gate: cuda.DeviceBuffer, scores: cuda.DeviceBuffer, picks: cuda.DeviceBuffer, weights: cuda.DeviceBuffer };
const Functions = struct {
    products: cuda.Function,
    select: cuda.Function,

    fn launch(self: Functions, stage: Stage, s: cuda.Stream, b: Buffers) !void {
        if (stage != .select) {
            var args: cuda.Args = .{};
            for ([_]u64{ b.input.ptr, b.router.ptr, b.shared_gate.ptr, b.scores.ptr }) |address| args.add(address);
            try cuda.launch.launch(self.products, .{ .grid = .{ .x = 257 }, .block = .{ .x = 256 } }, s, &args);
        }
        if (stage != .products) {
            var args: cuda.Args = .{};
            for ([_]u64{ b.scores.ptr, b.picks.ptr, b.weights.ptr }) |address| args.add(address);
            try cuda.launch.launch(self.select, .{ .grid = .{ .x = 1 }, .block = .{ .x = 256 } }, s, &args);
        }
    }
};

fn poison(b: Buffers, s: cuda.Stream, stage: Stage) !void {
    if (stage != .select) try b.scores.fill8(0xff, s.handle);
    if (stage != .products) {
        try b.picks.fill8(0xff, s.handle);
        try b.weights.fill8(0xff, s.handle);
    }
}

fn verify(b: Buffers, row: Row, expected_scores: ?[]const u8) ![score_bytes]u8 {
    var scores: [score_bytes]u8 = undefined;
    try b.scores.download(0, &scores);
    try finite(&scores, true);
    if (expected_scores) |expected| try check.sameBytes("257 router scores", &scores, expected);
    var picks: [32]u8 = undefined;
    var weights: [36]u8 = undefined;
    try b.picks.download(0, &picks);
    try b.weights.download(0, &weights);
    try finite(&weights, true);
    try check.sameBytes("saved native router picks", &picks, row.picks);
    try check.sameBytes("saved native router FP32 weights", &weights, row.weights);
    return scores;
}

fn capture(f: Functions, stage: Stage, s: cuda.Stream, b: Buffers) !cuda.graph.Exec {
    try cuda.graph.beginCapture(s, .thread_local);
    for (0..launches) |_| f.launch(stage, s, b) catch |err| {
        if (cuda.graph.endCapture(s)) |graph| {
            var dead = graph;
            dead.deinit();
        } else |_| {}
        return err;
    };
    var graph = try cuda.graph.endCapture(s);
    defer graph.deinit();
    if (try graph.nodeCount() != nodeCount(stage)) return error.RouterGraphWorkCountMismatch;
    var exec = try graph.instantiate();
    errdefer exec.deinit();
    try exec.upload(s);
    try s.synchronize();
    return exec;
}

fn nodeCount(stage: Stage) usize {
    return launches * @as(usize, if (stage == .combined) 2 else 1);
}

const Arm = enum { old, new };
const candidate_stages = [_]Stage{ .select, .combined };
const Control = enum { all_zero, cutoff_ties, signed_zero, all_negative, pos_inf, neg_inf, nan_first, nan_middle, nan_last, nan_shared_gate };
const Outputs = struct {
    picks: [32]u8,
    weights: [36]u8,

    fn download(b: Buffers) !Outputs {
        var output: Outputs = undefined;
        try b.picks.download(0, &output.picks);
        try b.weights.download(0, &output.weights);
        return output;
    }

    fn same(actual: Outputs, expected: Outputs) !void {
        try check.sameBytes("old/new router picks", &actual.picks, &expected.picks);
        try check.sameBytes("old/new router weights including exceptional bits", &actual.weights, &expected.weights);
    }
};

fn plantedScores(control: Control) [experts + 1]f32 {
    var values: [experts + 1]f32 = @splat(0);
    switch (control) {
        .all_zero => {},
        .cutoff_ties => {
            for (values[0..7]) |*value| value.* = 2;
            for (values[7..24]) |*value| value.* = 1;
        },
        .signed_zero => for (&values, 0..) |*value, index| {
            value.* = @bitCast(@as(u32, if (index % 2 == 0) 0x80000000 else 0));
        },
        .all_negative => for (&values, 0..) |*value, index| {
            value.* = -@as(f32, @floatFromInt(index + 1)) / 256;
        },
        .pos_inf => values[0] = std.math.inf(f32),
        .neg_inf => values[0] = -std.math.inf(f32),
        .nan_first => values[0] = @bitCast(@as(u32, 0x7fc00001)),
        .nan_middle => values[128] = @bitCast(@as(u32, 0x7fc00002)),
        .nan_last => values[255] = @bitCast(@as(u32, 0x7fc00003)),
        .nan_shared_gate => values[256] = @bitCast(@as(u32, 0x7fc00004)),
    }
    return values;
}

fn finiteControl(control: Control) bool {
    return switch (control) {
        .all_zero, .cutoff_ties, .signed_zero, .all_negative => true,
        else => false,
    };
}

fn runCandidate(gpu: check.Gpu, functions: [2]Functions, s: cuda.Stream, b: Buffers, fixture: Fixture, reference: *const [6][score_bytes]u8, journal: anytype) !void {
    for (fixture.rows, 0..) |row, position| {
        try b.scores.upload(0, &reference[position]);
        for (std.meta.tags(Arm), functions) |arm, f| {
            try poison(b, s, .select);
            try f.launch(.select, s, b);
            try s.synchronize();
            const actual = try verify(b, row, &reference[position]);
            const digest = hash(&actual);
            try floor(gpu);
            try journal.write(.{ .kind = "router_candidate_row_parity", .ordinal = gpu.ctx.device_ordinal, .position = position, .arm = @tagName(arm), .finite_scores = 257, .scores_sha256 = &digest, .scores_bit_exact = true, .saved_native_picks_and_weights_bit_exact = true });
        }
    }
    for (std.meta.tags(Control)) |control| {
        var values = plantedScores(control);
        const bytes = std.mem.asBytes(&values);
        if (finiteControl(control)) try finite(bytes, true);
        try b.scores.upload(0, bytes);
        var outputs: [2]Outputs = undefined;
        for (functions, 0..) |f, arm| {
            try poison(b, s, .select);
            try f.launch(.select, s, b);
            try s.synchronize();
            outputs[arm] = try Outputs.download(b);
            if (finiteControl(control)) try finite(&outputs[arm].weights, true);
            var actual: [score_bytes]u8 = undefined;
            try b.scores.download(0, &actual);
            try check.sameBytes("planted router scores unchanged", &actual, bytes);
        }
        try outputs[1].same(outputs[0]);
        const digest = hash(bytes);
        try floor(gpu);
        try journal.write(.{ .kind = "router_candidate_control", .ordinal = gpu.ctx.device_ordinal, .case_id = @tagName(control), .finite_case = finiteControl(control), .finite_outputs = if (finiteControl(control)) @as(?bool, true) else null, .scores_sha256 = &digest, .scores_bit_exact = true, .old_new_picks_bit_exact = true, .old_new_weights_bit_exact = true, .control_pass = true });
    }
    try b.input.upload(0, fixture.rows[0].input);
    try b.scores.upload(0, &reference[0]);
    var graphs: [4]cuda.graph.Exec = undefined;
    var made: usize = 0;
    defer for (graphs[0..made]) |*graph| graph.deinit();
    for (candidate_stages, 0..) |stage, stage_index| {
        for (std.meta.tags(Arm), functions, 0..) |arm, f, arm_index| {
            graphs[stage_index * 2 + arm_index] = try capture(f, stage, s, b);
            made += 1;
            try floor(gpu);
            try journal.write(.{ .kind = "router_candidate_graph", .ordinal = gpu.ctx.device_ordinal, .stage = @tagName(stage), .arm = @tagName(arm), .node_count = nodeCount(stage), .kernel_launches = nodeCount(stage), .route_repetitions = launches });
        }
    }
    var start = try cuda.Event.init(gpu.d, true);
    defer start.deinit();
    var end = try cuda.Event.init(gpu.d, true);
    defer end.deinit();
    for (0..6) |pair| {
        for (candidate_stages, 0..) |stage, stage_index| {
            for (0..2) |order| {
                const arm_index = if (pair % 2 == 0) order else 1 - order;
                const arm: Arm = @fromBackingInt(@intCast(arm_index));
                try floor(gpu);
                if (stage == .select) try b.scores.upload(0, &reference[0]);
                try poison(b, s, stage);
                try start.record(s);
                try graphs[stage_index * 2 + arm_index].launchOn(s);
                try end.record(s);
                try end.synchronize();
                const elapsed_ms = try cuda.Event.elapsedMs(start, end);
                const actual = try verify(b, fixture.rows[0], &reference[0]);
                const digest = hash(&actual);
                try floor(gpu);
                try journal.write(.{ .kind = "router_candidate_sample", .ordinal = gpu.ctx.device_ordinal, .stage = @tagName(stage), .pair = pair, .warmup = pair == 0, .order = order, .arm = @tagName(arm), .position = 0, .elapsed_ms = elapsed_ms, .node_count = nodeCount(stage), .kernel_launches = nodeCount(stage), .graph_launches = 1, .route_repetitions = launches, .finite_scores = 257, .scores_sha256 = &digest, .scores_bit_exact = true, .saved_native_picks_and_weights_bit_exact = true, .scope = scope });
            }
        }
    }
    try journal.write(.{ .kind = "router_candidate_complete", .ordinal = gpu.ctx.device_ordinal, .correctness_pass = true, .saved_rows_exact = 6, .planted_controls_exact = std.meta.tags(Control).len, .measured_pairs_per_stage = 5, .warmup_pairs_per_stage = 1, .scope = scope, .production_qualification = false });
}

pub fn run(gpu: check.Gpu, fixture_dir: []const u8, expected: []const u8, candidate: bool, journal: anytype) !void {
    try floor(gpu);
    var arena = std.heap.ArenaAllocator.init(gpu.gpa);
    defer arena.deinit();
    const fixture = try loadFixture(arena.allocator(), gpu.io, fixture_dir, expected);
    var image = try cuda.Module.load(gpu.d, cuda.kernels.qwen35_ops);
    defer image.unload();
    const f: Functions = .{ .products = try image.function("tf_qwen35_route_products"), .select = try image.function("tf_qwen35_route_select") };
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var router = try cuda.DeviceBuffer.fromHost(gpu.d, fixture.router);
    defer router.free();
    var shared_gate = try cuda.DeviceBuffer.fromHost(gpu.d, fixture.shared_gate);
    defer shared_gate.free();
    var input = try cuda.DeviceBuffer.alloc(gpu.d, dims * 2);
    defer input.free();
    var scores = try cuda.DeviceBuffer.alloc(gpu.d, score_bytes);
    defer scores.free();
    var picks = try cuda.DeviceBuffer.alloc(gpu.d, 32);
    defer picks.free();
    var weights = try cuda.DeviceBuffer.alloc(gpu.d, 36);
    defer weights.free();
    const b: Buffers = .{ .input = input, .router = router, .shared_gate = shared_gate, .scores = scores, .picks = picks, .weights = weights };
    try floor(gpu);
    try journal.write(.{ .kind = "router_fixture", .ordinal = gpu.ctx.device_ordinal, .manifest_sha256 = &fixture.manifest_sha256, .scope = scope, .rows = fixture.rows.len, .router_bytes = fixture.router.len, .shared_gate_bytes = fixture.shared_gate.len });
    try scores.fill8(0xff, s.handle);
    try s.synchronize();
    var planted: [score_bytes]u8 = undefined;
    try scores.download(0, &planted);
    if (finite(&planted, true)) |_| return error.RouterNegativeControlInsensitive else |err| if (err != error.NonfiniteRouterValue) return err;
    try journal.write(.{ .kind = "router_negative_control", .ordinal = gpu.ctx.device_ordinal, .poisoned_scores_rejected = true });
    var reference: [6][score_bytes]u8 = undefined;
    for (fixture.rows, 0..) |row, position| {
        try input.upload(0, row.input);
        try poison(b, s, .combined);
        try f.launch(.combined, s, b);
        try s.synchronize();
        reference[position] = try verify(b, row, null);
        try poison(b, s, .combined);
        try f.launch(.combined, s, b);
        try s.synchronize();
        _ = try verify(b, row, &reference[position]);
        const digest = hash(&reference[position]);
        try floor(gpu);
        try journal.write(.{ .kind = "router_row_parity", .ordinal = gpu.ctx.device_ordinal, .position = position, .finite_scores = 257, .scores_sha256 = &digest, .scores_repeat_bit_exact = true, .saved_native_picks_and_weights_bit_exact = true });
    }
    if (candidate) {
        const parallel: Functions = .{ .products = f.products, .select = try image.function("tf_qwen35_route_select_parallel") };
        return runCandidate(gpu, .{ f, parallel }, s, b, fixture, &reference, journal);
    }
    try input.upload(0, fixture.rows[0].input);
    try scores.upload(0, &reference[0]);
    var graphs: [stages.len]cuda.graph.Exec = undefined;
    var made: usize = 0;
    defer for (graphs[0..made]) |*graph| graph.deinit();
    for (stages, 0..) |stage, index| {
        graphs[index] = try capture(f, stage, s, b);
        made += 1;
        try floor(gpu);
        try journal.write(.{ .kind = "router_graph", .ordinal = gpu.ctx.device_ordinal, .stage = @tagName(stage), .node_count = nodeCount(stage), .kernel_launches = nodeCount(stage), .route_repetitions = launches });
    }
    var start = try cuda.Event.init(gpu.d, true);
    defer start.deinit();
    var end = try cuda.Event.init(gpu.d, true);
    defer end.deinit();
    for (0..6) |round| for (0..stages.len) |order| {
        const index = if (round % 2 == 0) order else stages.len - 1 - order;
        const stage = stages[index];
        try floor(gpu);
        if (stage == .select) try scores.upload(0, &reference[0]);
        try poison(b, s, stage);
        try start.record(s);
        try graphs[index].launchOn(s);
        try end.record(s);
        try end.synchronize();
        const elapsed_ms = try cuda.Event.elapsedMs(start, end);
        if (stage == .products) {
            var got: [score_bytes]u8 = undefined;
            try scores.download(0, &got);
            try finite(&got, true);
            try check.sameBytes("timed products scores", &got, &reference[0]);
            try poison(b, s, .select);
            try f.launch(.select, s, b);
            try s.synchronize();
        }
        const actual = try verify(b, fixture.rows[0], &reference[0]);
        const digest = hash(&actual);
        try floor(gpu);
        try journal.write(.{ .kind = "router_sample", .ordinal = gpu.ctx.device_ordinal, .stage = @tagName(stage), .round = round, .warmup = round == 0, .order = order, .position = 0, .elapsed_ms = elapsed_ms, .node_count = nodeCount(stage), .kernel_launches = nodeCount(stage), .graph_launches = 1, .route_repetitions = launches, .finite_scores = 257, .scores_sha256 = &digest, .scores_bit_exact = true, .saved_native_picks_and_weights_bit_exact = true, .scope = scope });
    };
    try journal.write(.{ .kind = "router_screen_complete", .ordinal = gpu.ctx.device_ordinal, .correctness_pass = true, .saved_rows_exact = 6, .timed_repetitions_per_stage = 5, .warmup_repetitions_per_stage = 1, .scope = scope, .production_qualification = false });
}

test "router screen rejects changed fixture bytes and any nonfinite score" {
    _ = try verifyHash("abc", "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    try std.testing.expectError(error.RouterFixtureHashMismatch, verifyHash("abd", "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"));
    var scores: [score_bytes]u8 = @splat(0);
    try finite(&scores, true);
    @memset(scores[256 * 4 ..][0..4], 0xff);
    try std.testing.expectError(error.NonfiniteRouterValue, finite(&scores, true));
    try std.testing.expectEqual(@as(usize, 256), nodeCount(.combined));
    try std.testing.expectEqual(@as(usize, 128), nodeCount(.products));
    try std.testing.expectEqual(@as(usize, 128), nodeCount(.select));
}
