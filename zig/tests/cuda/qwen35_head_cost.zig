const std = @import("std");
const qwen = @import("qwen35_cuda");
const cuda = @import("cuda");
const bench_options = @import("qwen_bench_options");
comptime {
    if (bench_options.engine_optimization != .fast) @compileError("Qwen timing probes require ReleaseFast engine modules");
}
const admission_module = @import("qwen35_probe_device.zig");

const floor_bytes = 1536 << 20;
const width = 2048;
const vocab = 248320;
const calls_per_block = 32;
const physical = [_][]const u8{
    "GPU-f2c0ea1e-a2be-f0b4-6c23-ffe33b537a06",
    "GPU-dd2a988c-fe38-24ab-233a-9e9bb3b2645a",
};
const Row = struct {
    name: []const u8,
    zero: bool,
    input: []const u8,
    input_sha256: []const u8,
    normalized: []const u8,
    normalized_sha256: []const u8,
    expected: []const u8,
    expected_sha256: []const u8,
};
const Input = struct {
    devices: []const c_int,
    expected_device_uuids: []const []const u8,
    config_sha256: []const u8,
    index_sha256: []const u8,
    absolute_tolerance: f32,
    relative_tolerance: f32,
    rows: []const Row,
};
const Fixture = struct { row: Row, residual: []const u8, normalized: []const u8, expected: []const u8 };
const Fixtures = struct { manifest: Input, rows: []Fixture, manifest_hash: [64]u8 };
const Arm = enum { full_boundary, projection_only };
const ErrorStats = struct {
    max_absolute: f32 = 0,
    mean_absolute: f64 = 0,
    top_actual: usize = 0,
    top_expected: usize = 0,
    mismatches: usize = 0,
    first_mismatch: ?usize = null,
    first_actual: ?f32 = null,
    first_expected: ?f32 = null,
};

fn hash(bytes: []const u8) [64]u8 {
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &digest, .{});
    return std.fmt.bytesToHex(digest, .lower);
}

fn readPinned(a: std.mem.Allocator, io: std.Io, dir: []const u8, name: []const u8, expected_hash: []const u8, count: ?usize) ![]const u8 {
    const path = try std.fs.path.join(a, &.{ dir, name });
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(64 << 20));
    if (!std.mem.eql(u8, &hash(bytes), expected_hash)) return error.FixtureHashMismatch;
    if (count) |n| {
        if (bytes.len != n * 2) return error.FixtureLengthMismatch;
        for (0..n) |i| if (!std.math.isFinite(value(bytes, i))) return error.NonfiniteFixture;
    }
    return bytes;
}

fn value(bytes: []const u8, i: usize) f32 {
    const bits = std.mem.readInt(u16, bytes[i * 2 ..][0..2], .little);
    return @bitCast(@as(u32, bits) << 16);
}

fn loadFixtures(a: std.mem.Allocator, io: std.Io, path: []const u8) !Fixtures {
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(4 << 20));
    const input = try std.json.parseFromSliceLeaky(Input, a, bytes, .{ .ignore_unknown_fields = true });
    if (input.devices.len != 1 or input.devices[0] != 1 or input.expected_device_uuids.len != 1 or
        !std.mem.eql(u8, input.expected_device_uuids[0], physical[1])) return error.InvalidHeadDevice;
    if (input.absolute_tolerance != 1e-6 or input.relative_tolerance != 1.0 / 128.0) return error.InvalidTolerance;
    if (input.rows.len != 4) return error.InvalidFixtureCount;
    const dir = std.fs.path.dirname(path) orelse ".";
    const rows = try a.alloc(Fixture, input.rows.len);
    var zeros: usize = 0;
    for (input.rows, rows, 0..) |row, *fixture, idx| {
        if (row.name.len == 0) return error.InvalidFixtureName;
        for (input.rows[0..idx]) |prior| if (std.mem.eql(u8, prior.name, row.name)) return error.DuplicateFixtureName;
        fixture.* = .{
            .row = row,
            .residual = try readPinned(a, io, dir, row.input, row.input_sha256, width),
            .normalized = try readPinned(a, io, dir, row.normalized, row.normalized_sha256, width),
            .expected = try readPinned(a, io, dir, row.expected, row.expected_sha256, vocab),
        };
        if (row.zero) {
            zeros += 1;
            for ([_][]const u8{ fixture.residual, fixture.normalized, fixture.expected }) |array| {
                for (0..array.len / 2) |i| if (value(array, i) != 0) return error.InvalidZeroFixture;
            }
        } else {
            var nonzero = false;
            for (0..width) |i| nonzero = nonzero or value(fixture.residual, i) != 0;
            if (!nonzero) return error.InvalidNonzeroFixture;
        }
    }
    if (zeros != 1) return error.InvalidZeroFixture;
    return .{ .manifest = input, .rows = rows, .manifest_hash = hash(bytes) };
}

const Sink = struct {
    a: std.mem.Allocator,
    io: std.Io,
    file: std.Io.File,
    fn emit(s: Sink, record: anytype) !void {
        var output: std.Io.Writer.Allocating = .init(s.a);
        defer output.deinit();
        try std.json.Stringify.value(record, .{}, &output.writer);
        try output.writer.writeByte('\n');
        try s.file.writeStreamingAll(s.io, output.written());
    }
};

fn compare(actual: []const u8, expected: []const u8, zero: bool) !ErrorStats {
    if (actual.len != expected.len) return error.ResultShapeMismatch;
    var stats: ErrorStats = .{};
    for (0..actual.len / 2) |i| {
        const x = value(actual, i);
        const y = value(expected, i);
        if (!std.math.isFinite(x)) return error.NonfiniteOutput;
        const delta = @abs(x - y);
        stats.max_absolute = @max(stats.max_absolute, delta);
        stats.mean_absolute += delta;
        if (x > value(actual, stats.top_actual)) stats.top_actual = i;
        if (y > value(expected, stats.top_expected)) stats.top_expected = i;
        if (delta > (if (zero) @as(f32, 0) else 1e-6 + @abs(y) / 128.0)) {
            stats.mismatches += 1;
            if (stats.first_mismatch == null) {
                stats.first_mismatch = i;
                stats.first_actual = x;
                stats.first_expected = y;
            }
        }
    }
    stats.mean_absolute /= @floatFromInt(actual.len / 2);
    return stats;
}

fn alloc(ctx: *const cuda.Context, bytes: usize) !cuda.DeviceBuffer {
    if ((try ctx.memInfo()).free < floor_bytes + bytes) return error.GpuMemoryFloor;
    var buffer = try cuda.DeviceBuffer.alloc(ctx.d, bytes);
    errdefer buffer.free();
    if ((try ctx.memInfo()).free < floor_bytes) return error.GpuMemoryFloor;
    return buffer;
}

fn freePair(ctx: *const cuda.Context, peer: *const cuda.Context) ![2]usize {
    const first = (try peer.memInfo()).free;
    const second = (try ctx.memInfo()).free;
    if (first < floor_bytes or second < floor_bytes) return error.GpuMemoryFloor;
    return .{ first, second };
}

const Rig = struct {
    stream: cuda.Stream,
    norms: *const cuda.norms.Norms,
    linear: cuda.qlinear.Linear,
    head: cuda.qlinear.Weight,
    norm: u64,
    residual: cuda.DeviceBuffer,
    normalized: cuda.DeviceBuffer,
    output: cuda.DeviceBuffer,
    scratch: cuda.DeviceBuffer,
    host: []u8,

    fn full(r: Rig) !void {
        try r.norms.rms(r.stream, r.residual.ptr, .bf16, r.norm, r.normalized.ptr, .bf16, 1, width, 1e-6);
        try r.project();
        try r.stream.synchronize();
        try r.output.download(0, r.host);
    }

    fn project(r: Rig) !void {
        try r.linear.decode(r.stream, r.head, .{ .x = r.normalized.ptr, .part = r.scratch.ptr }, r.output.ptr, 1);
    }
};

fn run(init: std.process.Init, model: []const u8, fixtures: Fixtures, sink: Sink) !void {
    const a = init.arena.allocator();
    const input = fixtures.manifest;
    _ = try readPinned(a, init.io, model, "config.json", input.config_sha256, null);
    _ = try readPinned(a, init.io, model, "model.safetensors.index.json", input.index_sha256, null);
    const admission = try admission_module.Admission.parse(init.environ_map.get("CUDA_DEVICE_ORDER"), init.environ_map.get("CUDA_VISIBLE_DEVICES"), &.{ 0, 1 }, &physical);
    const cfg = try qwen.Config.read(init.gpa, init.io, model);
    var driver = try cuda.Driver.open();
    defer driver.close();
    const identities = try admission.verify(a, &driver);
    var peer = try cuda.Context.init(&driver, 0);
    defer peer.deinit();
    var ctx = try cuda.Context.init(&driver, 1);
    defer ctx.deinit();
    const initial_free = try freePair(&ctx, &peer);
    try sink.emit(.{ .kind = "admission", .identities = identities, .selected_ordinal = 1, .free_bytes = initial_free, .fixture_sha256 = fixtures.manifest_hash[0..], .source_commit = bench_options.source_commit, .root_optimization = @tagName(@import("builtin").mode), .engine_optimization = @tagName(bench_options.engine_optimization), .external_observer_required = true, .workload = "synthetic BF16 residuals; no transformer", .primary_boundary = "resident residual -> final RMS -> NVFP4 head -> stream synchronize -> 496640-byte BF16 download" });
    var weights = try qwen.weights.Weights.load(init.gpa, init.io, &ctx, model, cfg, 40, 40, false, true);
    defer weights.deinit();
    const head = weights.head orelse return error.MissingHead;
    if (head != .nvfp4 or head.inputs() != width or head.outputs() != vocab) return error.UnexpectedHead;
    var stream = try cuda.Stream.init(&driver, true);
    defer stream.deinit();
    var norms = try cuda.norms.Norms.load(&driver);
    defer norms.unload();
    var lane = try cuda.qmmf.Lane.load(&driver, (try ctx.capability()) / 10);
    defer lane.unload();
    var residual = try alloc(&ctx, width * 2);
    defer residual.free();
    var normalized = try alloc(&ctx, width * 2);
    defer normalized.free();
    var output = try alloc(&ctx, vocab * 2);
    defer output.free();
    const scratch_bytes = @max(1, lane.partBytes(1, head.nvfp4));
    var scratch = try alloc(&ctx, scratch_bytes);
    defer scratch.free();
    const host = try a.alloc(u8, vocab * 2);
    const norm_host = try a.alloc(u8, width * 2);
    const norm_repeat = try a.alloc(u8, width * 2);
    const repeat = try a.alloc(u8, vocab * 2);
    const rig: Rig = .{ .stream = stream, .norms = &norms, .linear = .{ .lane = &lane }, .head = head, .norm = weights.norm orelse return error.MissingNorm, .residual = residual, .normalized = normalized, .output = output, .scratch = scratch, .host = host };
    var start_event = try cuda.Event.init(&driver, true);
    defer start_event.deinit();
    var stop_event = try cuda.Event.init(&driver, true);
    defer stop_event.deinit();
    try sink.emit(.{ .kind = "loaded", .weight_format = "nvfp4", .weight_bytes = weights.totalDeviceBytes(), .scratch_bytes = scratch_bytes, .free_bytes = try freePair(&ctx, &peer), .absolute_tolerance = input.absolute_tolerance, .relative_tolerance = input.relative_tolerance });
    for (fixtures.rows) |fixture| {
        try residual.upload(0, fixture.residual);
        try rig.full();
        try normalized.download(0, norm_host);
        const norm_stats = try compare(norm_host, fixture.normalized, fixture.row.zero);
        const head_stats = try compare(host, fixture.expected, fixture.row.zero);
        try sink.emit(.{ .kind = "reference", .row = fixture.row.name, .norm = norm_stats, .head = head_stats, .output_sha256 = hash(host)[0..] });
        if (norm_stats.mismatches != 0 or head_stats.mismatches != 0) return error.ReferenceMismatch;
        if (head_stats.top_actual != head_stats.top_expected) return error.ReferenceTopIdMismatch;
        @memcpy(repeat, host);
        @memcpy(norm_repeat, norm_host);
        try rig.full();
        if (!std.mem.eql(u8, repeat, host)) return error.NondeterministicHead;
        try normalized.download(0, norm_host);
        if (!std.mem.eql(u8, norm_repeat, norm_host)) return error.NondeterministicNorm;
        const repeat_norm = try compare(norm_host, fixture.normalized, fixture.row.zero);
        if (repeat_norm.mismatches != 0) return error.ReferenceMismatch;
        try sink.emit(.{ .kind = "determinism", .row = fixture.row.name, .repeat_exact = true });
        for (0..6) |pair| {
            for (0..2) |order| {
                const arm: Arm = if ((pair + order) % 2 == 0) .full_boundary else .projection_only;
                const free_before = try freePair(&ctx, &peer);
                const begin = std.Io.Clock.awake.now(init.io).toNanoseconds();
                if (arm == .projection_only) try start_event.record(stream);
                for (0..calls_per_block) |_| {
                    if (arm == .full_boundary) try rig.full() else try rig.project();
                }
                if (arm == .projection_only) try stop_event.record(stream);
                if (arm == .projection_only) try stream.synchronize();
                const elapsed = std.Io.Clock.awake.now(init.io).toNanoseconds() - begin;
                const gpu_ms: ?f32 = if (arm == .projection_only) try cuda.Event.elapsedMs(start_event, stop_event) else null;
                if (arm == .projection_only) try output.download(0, host);
                if (!std.mem.eql(u8, repeat, host)) return error.NondeterministicHead;
                const checked = try compare(host, fixture.expected, fixture.row.zero);
                if (checked.mismatches != 0) return error.ReferenceMismatch;
                try sink.emit(.{ .kind = "sample", .row = fixture.row.name, .pair = pair, .warmup = pair == 0, .order = order, .arm = @tagName(arm), .calls = calls_per_block, .rms_calls = if (arm == .full_boundary) calls_per_block else @as(usize, 0), .download_bytes = if (arm == .full_boundary) calls_per_block * vocab * 2 else @as(usize, 0), .wall_ns = elapsed, .cuda_ms = gpu_ms, .free_before = free_before, .free_after = try freePair(&ctx, &peer), .diagnostic_omits_rms_and_download = arm == .projection_only });
            }
        }
    }
    const baseline_start = std.Io.Clock.awake.now(init.io).toNanoseconds();
    try start_event.record(stream);
    try stop_event.record(stream);
    try stream.synchronize();
    const baseline_ns = std.Io.Clock.awake.now(init.io).toNanoseconds() - baseline_start;
    try sink.emit(.{ .kind = "empty_event_sync_baseline", .wall_ns = baseline_ns, .cuda_ms = try cuda.Event.elapsedMs(start_event, stop_event), .subtracted = false });
    try sink.emit(.{ .kind = "complete", .status = "pass", .rows = fixtures.rows.len, .measured_pairs_per_row = 5, .warmup_pairs_per_row = 1, .calls_per_block = calls_per_block, .validation_full_calls_per_row = 2, .measured_projection_calls = fixtures.rows.len * 5 * 2 * calls_per_block, .measured_rms_calls = fixtures.rows.len * 5 * calls_per_block, .measured_download_bytes = fixtures.rows.len * 5 * calls_per_block * vocab * 2, .full_model_speedup_measured = false, .runtime_elision_implemented = false, .free_bytes = try freePair(&ctx, &peer) });
}

pub fn main(init: std.process.Init) !u8 {
    const a = init.arena.allocator();
    const args = try init.minimal.args.toSlice(a);
    if (args.len == 2 and std.mem.eql(u8, args[1], "--identity")) {
        var output: std.Io.Writer.Allocating = .init(a);
        try std.json.Stringify.value(.{ .source_commit = bench_options.source_commit, .engine_optimization = @tagName(bench_options.engine_optimization), .root_optimization = @tagName(@import("builtin").mode) }, .{}, &output.writer);
        try output.writer.writeByte('\n');
        try std.Io.File.stdout().writeStreamingAll(init.io, output.written());
        return 0;
    }
    if (args.len == 3 and std.mem.eql(u8, args[1], "--check")) {
        const fixtures = try loadFixtures(a, init.io, args[2]);
        try std.Io.File.stdout().writeStreamingAll(init.io, try std.fmt.allocPrint(a, "Validated {d} pinned BF16 residual/norm/head fixture rows on CPU; no CUDA driver opened.\n", .{fixtures.rows.len}));
        return 0;
    }
    if (args.len != 4) {
        try std.Io.File.stderr().writeStreamingAll(init.io, "usage: tf-qwen35-head-cost --check INPUT_JSON\n       tf-qwen35-head-cost MODEL_DIR INPUT_JSON OUTPUT_DIR\nDevice execution requires a separately approved window and external physical-GPU observation. OUTPUT_DIR must not exist.\n");
        return 2;
    }
    try std.Io.Dir.cwd().createDir(init.io, args[3], .default_dir);
    const path = try std.fs.path.join(a, &.{ args[3], "results.jsonl" });
    const file = try std.Io.Dir.cwd().createFile(init.io, path, .{ .exclusive = true });
    defer file.close(init.io);
    const sink: Sink = .{ .a = a, .io = init.io, .file = file };
    const fixtures = loadFixtures(a, init.io, args[2]) catch |err| {
        try sink.emit(.{ .kind = "failure", .stage = "fixtures", .err = @errorName(err) });
        return err;
    };
    run(init, args[1], fixtures, sink) catch |err| {
        try sink.emit(.{ .kind = "failure", .stage = "device_experiment", .err = @errorName(err), .partial_results = true });
        return err;
    };
    return 0;
}
