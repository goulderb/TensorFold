//! Alternating prefill A/B through the serving lane host, with one checkpoint load.
const std = @import("std");
const cuda = @import("cuda");
const api = @import("engine_api");
const lanes = @import("lanes");
const glm = @import("glm_cuda");
const bound = @import("cuda_bound");
const check = @import("check.zig");

const streams = 8;
const generated = 128;

const Results = struct {
    io: std.Io,
    began: i96,
    mutex: std.Io.Mutex = .init,
    done: usize = 0,
    failed: usize = 0,
    tokens: [streams][generated]u32 = @splat(@splat(0)),
    counts: [streams]usize = @splat(0),
    rounds: [streams]u64 = @splat(0),
    ttft_ms: [streams]f64 = @splat(0),
    prefilled_ms: [streams]f64 = @splat(0),

    fn event(ptr: *anyopaque, id: api.Id, ev: *const api.Event) void {
        const r: *Results = @ptrCast(@alignCast(ptr));
        r.mutex.lockUncancelable(r.io);
        defer r.mutex.unlock(r.io);
        const i: usize = @intCast(id - 1);
        switch (ev.*) {
            .tokens => |ids| {
                if (r.counts[i] == 0) r.ttft_ms[i] = r.elapsed();
                if (ids.len > generated - r.counts[i]) {
                    r.failed += 1;
                    return;
                }
                @memcpy(r.tokens[i][r.counts[i]..][0..ids.len], ids);
                r.counts[i] += ids.len;
            },
            .prefilled => r.prefilled_ms[i] = r.elapsed(),
            .finished => |end| {
                r.failed += @intFromBool(end.reason != .length);
                r.rounds[i] = end.stats.rounds;
                r.done += 1;
            },
            else => {},
        }
    }

    fn elapsed(r: *const Results) f64 {
        return @as(f64, @floatFromInt(std.Io.Clock.awake.now(r.io).toNanoseconds() - r.began)) / 1e6;
    }
};

const Bound = struct {
    ctx: *const cuda.Context,
    inner: lanes.backend.Backend,
    threadlocal var current: ?*const cuda.Context = null;
    pub fn bind(ptr: *anyopaque) *Bound {
        const b: *Bound = @ptrCast(@alignCast(ptr));
        if (current != b.ctx) {
            b.ctx.makeCurrent() catch @panic("benchmark CUDA context bind failed");
            current = b.ctx;
        }
        return b;
    }
};

const Sample = struct {
    pattern: []const u8,
    pair: usize,
    warmup: bool,
    batch: bool,
    wall_ms: f64,
    completion_tokens: usize,
    ttft_ms: [streams]f64,
    prefilled_ms: [streams]f64,
    batches: u64,
    steps: u64,
    shared_rounds: u64,
    stream_rounds: [streams]u64,
    drafted: u64,
    accepted: u64,
    draft_widths: [streams]u64,
    output_sha256: [64]u8,
    verify_drafts: usize = 0,
    verify_widths: [glm.engine.shared_rows + 1]u64 = @splat(0),
};

pub fn run(gpu: check.Gpu, checkpoint: []const u8, input: []const u8, output: []const u8) !void {
    return runMode(gpu, checkpoint, input, output, false);
}

pub fn verifyRun(gpu: check.Gpu, checkpoint: []const u8, input: []const u8, output: []const u8) !void {
    return runMode(gpu, checkpoint, input, output, true);
}

fn runMode(gpu: check.Gpu, checkpoint: []const u8, input: []const u8, output: []const u8, verify: bool) !void {
    var arena = std.heap.ArenaAllocator.init(gpu.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const prompts = try std.json.parseFromSliceLeaky([]const []const u32, a, input, .{});
    if (prompts.len != streams) return error.BadBenchmarkInputs;
    var total_rows: usize = 0;
    for (prompts) |ids| total_rows += ids.len;
    if (total_rows > glm.engine.prompt_rows) return error.BadBenchmarkInputs;
    const loaded = try glm.native.open(gpu.gpa, gpu.io, gpu.ctx, checkpoint, null, .{ .context = 262144, .drafts = true });
    // This CLI owns the process. Followers exit after end; NCCL destroy can wait forever for them.
    const gl: *glm.lanes.Lanes = @ptrCast(@alignCast(loaded.backend.ptr));
    if (verify and gl.x.e.verifyPlan() != .dflash7) return error.BadVerifyBenchmarkProfile;
    defer gl.x.end() catch {};
    var samples: std.ArrayList(Sample) = .empty;
    var reference: ?[streams][generated]u32 = null;
    for ([_][]const u8{ "burst", "staggered" }) |pattern| {
        for (0..6) |pair| {
            const order = if (pair % 2 == 0) [_]bool{ false, true } else [_]bool{ true, false };
            for (order) |batch| {
                gl.batch_prefill = if (verify) true else batch;
                if (verify) gl.x.e.settings.dflash_verify = if (batch) .dflash7 else .dflash6;
                var cfg = try lanes.Config.init(gpu.gpa, gl.facts(), loaded.rows, loaded.rows - 1);
                defer cfg.deinit(gpu.gpa);
                var clock: lanes.backend.WallClock = .{ .io = gpu.io };
                var wrapper: Bound = .{ .ctx = gpu.ctx, .inner = gl.backend() };
                const vt = bound.vtable(Bound, wrapper.inner.vtable.*);
                var core = lanes.Engine.init(gpu.gpa, &cfg, .{ .ptr = &wrapper, .vtable = &vt }, clock.clock());
                defer core.deinit();
                var host = api.LaneHost.init(gpu.gpa, gpu.io, &core, .{ .lanes = streams });
                var results: Results = .{ .io = gpu.io, .began = std.Io.Clock.awake.now(gpu.io).toNanoseconds() };
                const before = gl.prefill_batches;
                const before_verify = gl.x.e.verify_widths;
                const before_widths: [streams]u64 = if (gl.x.e.dflash) |d| d.width_histogram.bins else @splat(0);
                const staggered = std.mem.eql(u8, pattern, "staggered");
                var requests: [streams]api.Request = undefined;
                const endpoint = host.engine();
                for (prompts, &requests) |ids, *r| r.* = .{ .prompt = ids, .max_tokens = generated };
                var started = false;
                defer if (started) host.stop();
                if (staggered) {
                    try host.start();
                    started = true;
                }
                for (&requests, 0..) |*request, i| {
                    try endpoint.submit(@intCast(i + 1), request, .{ .ctx = &results, .event = Results.event });
                    if (staggered and i + 1 < streams) try std.Io.sleep(gpu.io, .fromMilliseconds(25), .awake);
                }
                if (!started) {
                    try host.start();
                    started = true;
                }
                while (true) {
                    results.mutex.lockUncancelable(gpu.io);
                    const done = results.done;
                    results.mutex.unlock(gpu.io);
                    if (done == streams) break;
                    if (results.elapsed() > 120000) return error.BenchmarkTimedOut;
                    try std.Io.sleep(gpu.io, .fromMilliseconds(1), .awake);
                }
                const wall_ms = results.elapsed();
                host.stop();
                started = false;
                try gpu.ctx.makeCurrent();
                var count: usize = 0;
                for (results.counts) |n| count += n;
                if (results.failed != 0 or count != streams * generated) return error.BenchmarkOutputFailed;
                const batches = gl.prefill_batches - before;
                if (!verify and !batch and batches != 0) return error.UnexpectedBenchmarkBatch;
                if ((verify or batch) and !staggered and batches == 0) return error.BenchmarkBatchNotExercised;
                if (reference) |want| {
                    if (!std.mem.eql(u8, std.mem.asBytes(&want), std.mem.asBytes(&results.tokens))) {
                        const failure = try std.json.Stringify.valueAlloc(a, .{ .complete = false, .failure = "BenchmarkOutputChanged", .input_rows = total_rows, .samples = samples.items, .rejected = .{ .pattern = pattern, .pair = pair, .batch = batch, .batches = batches, .counts = results.counts, .failed = results.failed, .reference = want, .tokens = results.tokens } }, .{ .whitespace = .indent_2 });
                        try std.Io.Dir.cwd().writeFile(gpu.io, .{ .sub_path = output, .data = failure });
                        std.log.err("benchmark output identity failed; rejected tokens saved to {s}", .{output});
                        return error.BenchmarkOutputChanged;
                    }
                } else reference = results.tokens;
                var digest: [32]u8 = undefined;
                std.crypto.hash.sha2.Sha256.hash(std.mem.asBytes(&results.tokens), &digest, .{});
                var widths: [streams]u64 = @splat(0);
                if (gl.x.e.dflash) |d| for (&widths, d.width_histogram.bins, before_widths) |*delta, current, prior| {
                    delta.* = current - prior;
                };
                var verify_widths: [glm.engine.shared_rows + 1]u64 = undefined;
                for (&verify_widths, gl.x.e.verify_widths, before_verify) |*delta, now, prior| delta.* = now - prior;
                try samples.append(a, .{ .pattern = pattern, .pair = pair, .warmup = pair == 0, .batch = batch, .wall_ms = wall_ms, .completion_tokens = count, .ttft_ms = results.ttft_ms, .prefilled_ms = results.prefilled_ms, .batches = batches, .steps = core.steps, .shared_rounds = core.shared_rounds, .stream_rounds = results.rounds, .drafted = core.drafted, .accepted = core.accepted, .draft_widths = widths, .output_sha256 = std.fmt.bytesToHex(digest, .lower), .verify_drafts = gl.x.e.verifyPlan().drafts(), .verify_widths = verify_widths });
                const report = try std.json.Stringify.valueAlloc(a, .{ .complete = false, .input_rows = total_rows, .samples = samples.items }, .{ .whitespace = .indent_2 });
                try std.Io.Dir.cwd().writeFile(gpu.io, .{ .sub_path = output, .data = report });
                std.debug.print("RESULT prefill-batch pattern={s} pair={d} batch={} wall_ms={d:.3} tokens={d} batches={d}\n", .{ pattern, pair, batch, wall_ms, count, gl.prefill_batches - before });
            }
        }
    }
    const report = try std.json.Stringify.valueAlloc(a, .{ .complete = true, .input_rows = total_rows, .samples = samples.items }, .{ .whitespace = .indent_2 });
    try std.Io.Dir.cwd().writeFile(gpu.io, .{ .sub_path = output, .data = report });
    if (verify) std.debug.print("VERIFY_REPORT {s}\n", .{report});
}
