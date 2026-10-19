const std = @import("std");
const builtin = @import("builtin");
const options = @import("qwen_bench_options");
const qwen = @import("qwen35_cuda");
const cuda = @import("cuda");
const admission = @import("qwen35_probe_device.zig");
const turn = @import("qwen35_sampler_turn.zig");
const generation_graph = @import("qwen35_generation_graph.zig");
const expert_screen = @import("qwen35_expert_screen.zig");
const router_screen = @import("qwen35_router_screen.zig");

comptime {
    if (options.engine_optimization != .fast) @compileError("diagnostics require ReleaseFast engine");
}

const Input = struct { prompt: []const u32, token: u32, repetitions: usize = 16, expert_fixture: ?[]const u8 = null, expert_fixture_sha256: ?[]const u8 = null, router_screen: bool = false, router_candidate: bool = false };
const Journal = struct {
    a: std.mem.Allocator,
    io: std.Io,
    file: std.Io.File,
    pub fn write(j: Journal, value: anytype) !void {
        var out: std.Io.Writer.Allocating = .init(j.a);
        defer out.deinit();
        try std.json.Stringify.value(value, .{}, &out.writer);
        try out.writer.writeByte('\n');
        try j.file.writeStreamingAll(j.io, out.written());
    }
};

fn now(io: std.Io) i96 {
    return std.Io.Clock.awake.now(io).toNanoseconds();
}
fn hashFinish(h: *std.crypto.hash.sha2.Sha256) [64]u8 {
    var digest: [32]u8 = undefined;
    h.final(&digest);
    return std.fmt.bytesToHex(digest, .lower);
}
fn floor(e: *qwen.Engine) !void {
    for (e.partitions) |*p| if ((try p.context.memInfo()).free < qwen.floor_bytes) return error.GpuMemoryFloor;
}

const Saved = struct { rank: usize, dev: cuda.DeviceBuffer, host: []u8 };
const Snapshot = struct {
    entries: std.ArrayList(Saved) = .empty,
    transfer: [4096]u8,
    fn add(self: *Snapshot, a: std.mem.Allocator, rank: usize, dev: cuda.DeviceBuffer) !void {
        const bytes = try a.alloc(u8, dev.len);
        try dev.download(0, bytes);
        try self.entries.append(a, .{ .rank = rank, .dev = dev, .host = bytes });
    }
    fn take(a: std.mem.Allocator, s: *qwen.Session) !Snapshot {
        var result: Snapshot = .{ .transfer = undefined };
        const e = s.engine;
        @memcpy(&result.transfer, e.transfer.slice(u8));
        for (e.partitions, 0..) |*p, rank| {
            try p.context.makeCurrent();
            try p.stream.synchronize();
            try result.add(a, rank, p.scratch);
            for (s.layers[p.first..p.end]) |state| switch (state) {
                .gdn => |g| {
                    try result.add(a, rank, g.history);
                    try result.add(a, rank, g.recurrent);
                },
                .full => |kv| {
                    try result.add(a, rank, kv.keys);
                    try result.add(a, rank, kv.values);
                },
            };
        }
        return result;
    }
    fn restore(self: Snapshot, s: *qwen.Session) !void {
        for (self.entries.items) |entry| {
            try s.engine.partitions[entry.rank].context.makeCurrent();
            try entry.dev.upload(0, entry.host);
        }
        @memcpy(s.engine.transfer.slice(u8), &self.transfer);
        try floor(s.engine);
    }
};

fn stateHash(s: *qwen.Session, buffer: []u8) ![64]u8 {
    for (s.logits) |bits| if ((bits & 0x7f80) == 0x7f80) return error.NonfiniteLogits;
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    for (s.engine.partitions, 0..) |*p, rank| {
        try p.context.makeCurrent();
        try p.stream.synchronize();
        for (s.layers[p.first..p.end]) |state| {
            const buffers: [2]cuda.DeviceBuffer = switch (state) {
                .gdn => |g| .{ g.history, g.recurrent },
                .full => |kv| .{ kv.keys, kv.values },
            };
            for (buffers) |dev| {
                try dev.download(0, buffer[0..dev.len]);
                h.update(buffer[0..dev.len]);
            }
        }
        try p.scratch.download(@intCast(p.b.residual - p.scratch.ptr), buffer[0..4096]);
        h.update(buffer[0..4096]);
        if (rank + 1 == s.engine.partitions.len) {
            try p.scratch.download(@intCast(p.b.logits - p.scratch.ptr), buffer[0 .. 248320 * 2]);
            h.update(buffer[0 .. 248320 * 2]);
        }
    }
    return hashFinish(&h);
}

const Stamp = struct { event: cuda.Event, phase: qwen.DiagnosticPhase = .partition_begin, layer: usize = 99 };
const Tap = struct {
    stamps: []Stamp,
    used: usize = 0,
    current_layer: usize = 99,
    stream: cuda.Stream,
    record_external: *const fn (cuda.abi.Event, cuda.abi.Stream, c_uint) callconv(.c) cuda.abi.Result,
    fn init(a: std.mem.Allocator, d: *cuda.Driver, stream: cuda.Stream) !Tap {
        const stamps = try a.alloc(Stamp, 400);
        for (stamps) |*stamp| stamp.* = .{ .event = try cuda.Event.init(d, true) };
        const record = d.lib.lookup(*const fn (cuda.abi.Event, cuda.abi.Stream, c_uint) callconv(.c) cuda.abi.Result, "cuEventRecordWithFlags") orelse return error.MissingExternalEventSymbol;
        return .{ .stamps = stamps, .stream = stream, .record_external = record };
    }
    pub fn layer(self: *Tap, id: usize) void {
        self.current_layer = id;
    }
    pub fn mark(self: *Tap, phase: qwen.DiagnosticPhase) !void {
        if (self.used >= self.stamps.len) return error.TraceOverflow;
        const stamp = &self.stamps[self.used];
        stamp.phase = phase;
        stamp.layer = self.current_layer;
        try self.stream.d.check(self.record_external(stamp.event.handle, self.stream.handle, 1), "cuEventRecordWithFlags(EXTERNAL)");
        self.used += 1;
    }
    fn deinit(self: *Tap) void {
        for (self.stamps) |*stamp| stamp.event.deinit();
    }
};

const Captured = struct { exec: cuda.graph.Exec, nodes: usize };
fn capture(s: *qwen.Session, rank: usize, token: u32, position: usize, tap: anytype) !Captured {
    const p = &s.engine.partitions[rank];
    try p.context.makeCurrent();
    try cuda.graph.beginCapture(p.stream, .global);
    s.diagnosticPartition(rank, token, position, tap) catch |err| {
        if (cuda.graph.endCapture(p.stream)) |g| {
            var dead = g;
            dead.deinit();
        } else |_| {}
        return err;
    };
    var graph = try cuda.graph.endCapture(p.stream);
    defer graph.deinit();
    var exec = try graph.instantiate();
    errdefer exec.deinit();
    try exec.upload(p.stream);
    try p.stream.synchronize();
    try floor(s.engine);
    try p.context.makeCurrent();
    return .{ .exec = exec, .nodes = try graph.nodeCount() };
}

fn step(s: *qwen.Session, token: u32, position: usize, graphs: ?[]const Captured) !void {
    const e = s.engine;
    for (e.partitions, 0..) |*p, rank| {
        try p.context.makeCurrent();
        if (rank != 0) try p.scratch.upload(@intCast(p.b.residual - p.scratch.ptr), e.transfer.slice(u8));
        if (graphs) |captured| try captured[rank].exec.launchOn(p.stream) else try s.diagnosticPartition(rank, token, position, qwen.DiagnosticNoTap{});
        try p.stream.synchronize();
        if (rank + 1 < e.partitions.len) try p.scratch.download(@intCast(p.b.residual - p.scratch.ptr), e.transfer.slice(u8));
        if ((try p.context.memInfo()).free < qwen.floor_bytes) return error.GpuMemoryFloor;
    }
    const last = &e.partitions[e.partitions.len - 1];
    try last.scratch.download(@intCast(last.b.logits - last.scratch.ptr), std.mem.sliceAsBytes(s.logits));
}

fn parse(a: std.mem.Allocator, io: std.Io, path: []const u8) !Input {
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 20));
    const input = try std.json.parseFromSliceLeaky(Input, a, bytes, .{});
    if ((input.expert_fixture == null) != (input.expert_fixture_sha256 == null)) return error.InvalidExpertFixturePin;
    if (input.router_screen and input.expert_fixture == null) return error.RouterScreenRequiresPinnedFixture;
    if (input.router_candidate and !input.router_screen) return error.RouterCandidateRequiresRouterScreen;
    if (input.expert_fixture_sha256) |hash| {
        if (hash.len != 64 or input.expert_fixture.?.len == 0) return error.InvalidExpertFixturePin;
        for (hash) |char| if (!std.ascii.isDigit(char) and !(char >= 'a' and char <= 'f')) return error.InvalidExpertFixturePin;
    }
    if (input.prompt.len != 64 or input.token >= 248320 or input.repetitions != 16) return error.InvalidInput;
    for (input.prompt) |id| if (id >= 248320) return error.InvalidInput;
    return input;
}

fn routeMode(s: *qwen.Session, cooperative: bool) void {
    for (s.engine.partitions) |*p| p.ops.cooperative_route = cooperative;
}

fn routeParity(s: *qwen.Session, j: Journal) !void {
    for (s.engine.partitions, 0..) |*p, rank| {
        try p.context.makeCurrent();
        for (0..2) |fixture| {
            if (fixture == 1) try p.scratch.fill8(0, p.stream.handle);
            for (p.weights.layers, p.first..) |layer, index| {
                var picks: [2][8]i32 = undefined;
                var weights: [2][9]u32 = undefined;
                for (0..2) |arm| {
                    p.ops.cooperative_route = arm == 1;
                    try p.ops.route(p.stream, p.b.x, layer.moe.router, layer.moe.shared_gate, p.b.picks, p.b.route_weights, p.b.route_scores);
                    try p.stream.synchronize();
                    try p.scratch.download(@intCast(p.b.picks - p.scratch.ptr), std.mem.asBytes(&picks[arm]));
                    try p.scratch.download(@intCast(p.b.route_weights - p.scratch.ptr), std.mem.asBytes(&weights[arm]));
                }
                if (!std.mem.eql(i32, &picks[0], &picks[1]) or !std.mem.eql(u32, &weights[0], &weights[1])) return error.RouterParityMismatch;
                if (fixture == 1) for (picks[1], 0..) |id, expected| {
                    if (id != expected) return error.RouterTieMismatch;
                };
                try j.write(.{ .kind = "router_parity", .rank = rank, .layer = index, .fixture = if (fixture == 1) "zero-ties" else "saved-real-row", .exact_picks_and_fp32_weights = true });
            }
        }
    }
    routeMode(s, false);
}

fn generation(init: std.process.Init, s: *qwen.Session, input: Input, j: Journal) !void {
    const a = init.arena.allocator();
    var sampler = try turn.Sampler.init(init.gpa, s.logits.len);
    defer sampler.deinit();
    var output: [128]u32 = undefined;
    var reference: [128]u32 = undefined;
    const reference_rows = try a.alloc(u16, 128 * s.logits.len);
    const candidate_rows = try a.alloc(u16, reference_rows.len);
    routeMode(s, false);
    _ = try turn.run(true, init.io, s, &sampler, .current, input.prompt, &reference, reference_rows);
    routeMode(s, true);
    _ = try turn.run(true, init.io, s, &sampler, .current, input.prompt, &output, candidate_rows);
    if (!std.mem.eql(u32, &reference, &output) or !std.mem.eql(u16, reference_rows, candidate_rows)) return error.GenerationParityMismatch;
    try j.write(.{ .kind = "generation_parity", .rows = 128, .exact_consumed_bf16_rows_and_tokens = true, .reference_tokens = reference });
    for (0..6) |pair| for (0..2) |order| {
        const candidate = (pair + order) % 2 == 1;
        routeMode(s, candidate);
        const timing = try turn.run(false, init.io, s, &sampler, .current, input.prompt, &output, &.{});
        if (!std.mem.eql(u32, &reference, &output)) return error.GenerationRepeatMismatch;
        try j.write(.{ .kind = "generation_turn", .pair = pair, .order = order, .warmup = pair == 0, .cooperative = candidate, .times = timing, .tokens = output });
    };
}

fn profilePhases(init: std.process.Init, s: *qwen.Session, baseline: Snapshot, input: Input, position: usize, graphs: []const Captured, hash_buffer: []u8, j: Journal, comptime candidate: bool) !void {
    const a = init.arena.allocator();
    const e = s.engine;
    var taps: [2]Tap = undefined;
    var phased: [2]Captured = undefined;
    for (&taps, &phased, 0..) |*tap, *graph, rank| {
        try e.partitions[rank].context.makeCurrent();
        tap.* = try Tap.init(a, &e.driver, e.partitions[rank].stream);
        graph.* = try capture(s, rank, input.token, position, tap);
    }
    defer for (&taps, &phased, 0..) |*tap, *g, rank| {
        e.partitions[rank].context.makeCurrent() catch {};
        g.exec.deinit();
        tap.deinit();
    };
    for (0..5) |sample| {
        try baseline.restore(s);
        const phased_began = now(init.io);
        for (0..input.repetitions) |_| try step(s, input.token, position, &phased);
        const phased_elapsed = now(init.io) - phased_began;
        const digest = try stateHash(s, hash_buffer);
        try baseline.restore(s);
        const control_began = now(init.io);
        for (0..input.repetitions) |_| try step(s, input.token, position, graphs);
        const control_elapsed = now(init.io) - control_began;
        const control = try stateHash(s, hash_buffer);
        if (!std.mem.eql(u8, &digest, &control)) return error.ProfileStateMismatch;
        try j.write(.{ .kind = if (candidate) "router_profile_turn" else "profile_turn", .sample = sample, .phased_elapsed_ns = phased_elapsed, .control_elapsed_ns = control_elapsed, .steps = input.repetitions });
        for (taps, 0..) |tap, rank| {
            try e.partitions[rank].context.makeCurrent();
            for (1..tap.used) |i| {
                const stamp = tap.stamps[i];
                try j.write(.{ .kind = if (candidate) "router_phase" else "phase", .sample = sample, .rank = rank, .layer = stamp.layer, .phase = @tagName(stamp.phase), .ms = try cuda.Event.elapsedMs(tap.stamps[i - 1].event, stamp.event) });
            }
        }
    }
}

fn graphGeneration(init: std.process.Init, s: *qwen.Session, input: Input, hash_buffer: []u8, j: Journal) !void {
    const a = init.arena.allocator();
    routeMode(s, true);
    const setup_start = now(init.io);
    var replay = try generation_graph.Replay.init(a, s);
    defer replay.deinit();
    try j.write(.{ .kind = "generation_graph_setup", .elapsed_ns = now(init.io) - setup_start, .nodes = replay.partitions[0].nodes + replay.partitions[1].nodes, .dynamic_nodes = replay.partitions[0].patches.len + replay.partitions[1].patches.len });
    try s.reset();
    if (replay.append(&.{ 17, 248320 })) |_| return error.InvalidAdmissionAccepted else |err| if (err != error.InvalidToken) return err;
    s.position = s.engine.capacity;
    if (replay.append(&.{17})) |_| return error.InvalidAdmissionAccepted else |err| if (err != error.ContextFull) return err;
    s.position = 0;
    if (replay.appends != 0 or replay.updates != 0 or replay.launches != 0 or s.poisoned) return error.AdmissionMutatedState;
    try s.append(&.{ 17, 18 });
    const proper = try stateHash(s, hash_buffer);
    try s.reset();
    try replay.appendFrozenPosition(&.{ 17, 18 });
    const frozen = try stateHash(s, hash_buffer);
    if (std.mem.eql(u8, &proper, &frozen)) return error.NegativeControlInsensitive;
    try j.write(.{ .kind = "generation_graph_controls", .admission_before_mutation = true, .stale_position_changes_state = true });
    var sampler = try turn.Sampler.init(init.gpa, s.logits.len);
    defer sampler.deinit();
    var reference: [128]u32 = undefined;
    var output: [128]u32 = undefined;
    const reference_rows = try a.alloc(u16, 128 * s.logits.len);
    const actual_rows = try a.alloc(u16, reference_rows.len);
    _ = try turn.run(true, init.io, s, &sampler, .current, input.prompt, &reference, reference_rows);
    const reference_state = try stateHash(s, hash_buffer);
    replay.clearCounts();
    _ = try turn.runWithAppender(true, init.io, s, &replay, &sampler, .current, input.prompt, &output, actual_rows);
    const actual_state = try stateHash(s, hash_buffer);
    if (!std.mem.eql(u32, &reference, &output) or !std.mem.eql(u16, reference_rows, actual_rows) or !std.mem.eql(u8, &reference_state, &actual_state) or s.position != 191) return error.GraphGenerationParityMismatch;
    try j.write(.{ .kind = "generation_graph_parity", .rows = 128, .exact_rows_tokens_and_final_state = true, .state_and_logits_sha256 = &actual_state, .reference_tokens = reference });
    for (0..6) |pair| for (0..2) |order| {
        const graph = (pair + order) % 2 == 1;
        replay.clearCounts();
        const timing = if (graph)
            try turn.runWithAppender(false, init.io, s, &replay, &sampler, .current, input.prompt, &output, &.{})
        else
            try turn.run(false, init.io, s, &sampler, .current, input.prompt, &output, &.{});
        const state = try stateHash(s, hash_buffer);
        if (!std.mem.eql(u32, &reference, &output) or !std.mem.eql(u8, &reference_state, &state) or s.position != 191) return error.GraphGenerationRepeatMismatch;
        if (graph and (replay.appends != 191 or replay.launches != 382 or replay.updates != 4011)) return error.GraphWorkCountMismatch;
        try j.write(.{ .kind = "generation_graph_turn", .pair = pair, .order = order, .warmup = pair == 0, .graph = graph, .times = timing, .observed_graph_appends = replay.appends, .observed_graph_launches = replay.launches, .observed_parameter_updates = replay.updates, .position = s.position, .exact_tokens_and_final_state = true, .tokens = output });
    };
}

pub fn main(init: std.process.Init) !u8 {
    const a = init.arena.allocator();
    const args = try init.minimal.args.toSlice(a);
    if (args.len == 2 and std.mem.eql(u8, args[1], "--identity")) {
        var out: std.Io.Writer.Allocating = .init(a);
        try std.json.Stringify.value(.{ .source_commit = options.source_commit, .engine_optimization = @tagName(options.engine_optimization), .root_optimization = @tagName(builtin.mode) }, .{}, &out.writer);
        try out.writer.writeByte('\n');
        try std.Io.File.stdout().writeStreamingAll(init.io, out.written());
        return 0;
    }
    if (args.len == 3 and std.mem.eql(u8, args[1], "--check")) {
        _ = try parse(a, init.io, args[2]);
        return 0;
    }
    if (args.len == 3 and std.mem.eql(u8, args[1], "--check-router-input")) {
        const input = try parse(a, init.io, args[2]);
        if (!input.router_screen) return error.RouterScreenRequired;
        try router_screen.checkInput(init.gpa, init.io, input.expert_fixture.?, input.expert_fixture_sha256.?);
        return 0;
    }
    if (args.len != 4) return 2;
    try std.Io.Dir.cwd().createDir(init.io, args[3], .default_dir);
    const file = try std.Io.Dir.cwd().createFile(init.io, try std.fs.path.join(a, &.{ args[3], "results.jsonl" }), .{ .exclusive = true });
    defer file.close(init.io);
    const journal: Journal = .{ .a = a, .io = init.io, .file = file };
    execute(init, args[1], try parse(a, init.io, args[2]), journal) catch |err| {
        try journal.write(.{ .kind = "failure", .error_name = @errorName(err) });
        return 1;
    };
    return 0;
}

fn execute(init: std.process.Init, model_dir: []const u8, input: Input, j: Journal) !void {
    const a = init.arena.allocator();
    const uuids = [_][]const u8{ "GPU-f2c0ea1e-a2be-f0b4-6c23-ffe33b537a06", "GPU-dd2a988c-fe38-24ab-233a-9e9bb3b2645a" };
    const admitted = try admission.Admission.parse(init.environ_map.get("CUDA_DEVICE_ORDER"), init.environ_map.get("CUDA_VISIBLE_DEVICES"), &.{ 0, 1 }, &uuids);
    var driver = try cuda.Driver.open();
    var driver_open = true;
    defer if (driver_open) driver.close();
    const identities = try admitted.verify(a, &driver);
    try j.write(.{ .kind = "start", .source_commit = options.source_commit, .identities = identities, .scope = if (input.router_screen) router_screen.scope else if (input.expert_fixture != null) "real layer0 expert seam with rotating banks" else "fixed-token fixed-position evolving-state forward diagnostic; not generation throughput", .position = input.prompt.len, .token = input.token, .repetitions = input.repetitions });
    if (input.expert_fixture) |fixture_dir| {
        for (identities) |identity| {
            var context = try cuda.Context.init(&driver, identity.ordinal);
            defer context.deinit();
            if ((try context.memInfo()).free < qwen.floor_bytes) return error.GpuMemoryFloor;
            const gpu: @import("check.zig").Gpu = .{ .d = &driver, .ctx = &context, .gpa = init.gpa, .io = init.io };
            if (input.router_screen)
                try router_screen.run(gpu, fixture_dir, input.expert_fixture_sha256.?, input.router_candidate, j)
            else
                try expert_screen.run(gpu, fixture_dir, input.expert_fixture_sha256.?, j);
            if ((try context.memInfo()).free < qwen.floor_bytes) return error.GpuMemoryFloor;
        }
        try j.write(.{ .kind = "complete", .correctness_pass = true, .scope = if (input.router_screen) router_screen.scope else "real layer0 expert seam with 32 rotating banks; saved bank0 and variant self-parity", .production_qualification = false });
        return;
    }
    driver.close();
    driver_open = false;
    const e = try qwen.Engine.open(init.gpa, init.io, model_dir, .{});
    defer e.deinit();
    var s = try e.createSession();
    defer s.deinit();
    for (input.prompt) |id| try s.append(&.{id});
    const position = s.position;
    const baseline = try Snapshot.take(a, &s);
    const hash_buffer = try a.alloc(u8, 8 << 20);
    try s.append(&.{input.token});
    const ordinary_hash = try stateHash(&s, hash_buffer);
    try baseline.restore(&s);
    try step(&s, input.token, position, null);
    const diagnostic_hash = try stateHash(&s, hash_buffer);
    if (!std.mem.eql(u8, &ordinary_hash, &diagnostic_hash)) return error.DiagnosticPathMismatch;
    s.position = position;
    try baseline.restore(&s);
    var graphs: [2]Captured = undefined;
    for (&graphs, 0..) |*graph, rank| graph.* = try capture(&s, rank, input.token, position, qwen.DiagnosticNoTap{});
    defer for (&graphs, 0..) |*g, rank| {
        e.partitions[rank].context.makeCurrent() catch {};
        g.exec.deinit();
    };
    try j.write(.{ .kind = "graph_nodes", .nodes = .{ graphs[0].nodes, graphs[1].nodes }, .total = graphs[0].nodes + graphs[1].nodes, .ordinary_append_state_and_logits_equal = true });
    var references: [16][64]u8 = undefined;
    try baseline.restore(&s);
    for (&references) |*digest| {
        try step(&s, input.token, position, null);
        digest.* = try stateHash(&s, hash_buffer);
    }
    try baseline.restore(&s);
    for (references, 0..) |expected, i| {
        try step(&s, input.token, position, &graphs);
        const actual = try stateHash(&s, hash_buffer);
        if (!std.mem.eql(u8, &expected, &actual)) return error.IntermediateGraphStateMismatch;
        try j.write(.{ .kind = "step_parity", .step = i, .state_and_logits_sha256 = &actual });
    }
    for (0..6) |pair| {
        var hashes: [2][64]u8 = undefined;
        for (0..2) |order| {
            const arm = (pair + order) % 2;
            try baseline.restore(&s);
            const began = now(init.io);
            for (0..input.repetitions) |_| try step(&s, input.token, position, if (arm == 1) &graphs else null);
            const elapsed = now(init.io) - began;
            hashes[arm] = try stateHash(&s, hash_buffer);
            try j.write(.{ .kind = "turn", .pair = pair, .warmup = pair == 0, .order = order, .arm = if (arm == 1) "graph" else "eager", .elapsed_ns = elapsed, .steps = input.repetitions, .persistent_state_and_logits_sha256 = &hashes[arm] });
        }
        if (!std.mem.eql(u8, &hashes[0], &hashes[1])) return error.GraphStateMismatch;
        try j.write(.{ .kind = "pair_verified", .pair = pair, .exact_state_and_logits = true });
    }
    try profilePhases(init, &s, baseline, input, position, &graphs, hash_buffer, j, false);
    try baseline.restore(&s);
    try routeParity(&s, j);
    try baseline.restore(&s);
    routeMode(&s, true);
    var cooperative: [2]Captured = undefined;
    for (&cooperative, 0..) |*graph, rank| graph.* = try capture(&s, rank, input.token, position, qwen.DiagnosticNoTap{});
    defer for (&cooperative, 0..) |*g, rank| {
        e.partitions[rank].context.makeCurrent() catch {};
        g.exec.deinit();
    };
    try j.write(.{ .kind = "router_graph_nodes", .total = cooperative[0].nodes + cooperative[1].nodes });
    try baseline.restore(&s);
    for (references, 0..) |expected, i| {
        try step(&s, input.token, position, &cooperative);
        const actual = try stateHash(&s, hash_buffer);
        if (!std.mem.eql(u8, &expected, &actual)) return error.CooperativeIntermediateMismatch;
        try j.write(.{ .kind = "cooperative_step_parity", .step = i, .state_and_logits_sha256 = &actual });
    }
    for (0..2) |graph_mode| for (0..6) |pair| for (0..2) |order| {
        const candidate = (pair + order) % 2 == 1;
        routeMode(&s, candidate);
        try baseline.restore(&s);
        const selected: ?[]const Captured = if (graph_mode == 0) null else if (candidate) &cooperative else &graphs;
        const began = now(init.io);
        for (0..input.repetitions) |_| try step(&s, input.token, position, selected);
        const elapsed = now(init.io) - began;
        const digest = try stateHash(&s, hash_buffer);
        if (!std.mem.eql(u8, &digest, &references[15])) return error.CooperativeFinalMismatch;
        try j.write(.{ .kind = "router_turn", .graph = graph_mode == 1, .pair = pair, .order = order, .warmup = pair == 0, .cooperative = candidate, .elapsed_ns = elapsed, .steps = input.repetitions, .exact_state_and_logits = true });
    };
    routeMode(&s, true);
    try profilePhases(init, &s, baseline, input, position, &cooperative, hash_buffer, j, true);
    try generation(init, &s, input, j);
    try graphGeneration(init, &s, input, hash_buffer, j);
    try floor(e);
    try j.write(.{ .kind = "complete", .correctness_pass = true, .scope = "fixed-step and synthetic 64/128 greedy router self-parity; independent numerical gate remains open", .production_qualification = false });
}
