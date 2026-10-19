const std = @import("std");
const builtin = @import("builtin");
const options = @import("qwen_bench_options");
const qwen = @import("qwen35_cuda");
const cuda = @import("cuda");
const admission = @import("qwen35_probe_device.zig");

comptime {
    if (options.engine_optimization != .fast) @compileError("diagnostics require ReleaseFast engine");
}

const Input = struct { prompt: []const u32, token: u32, repetitions: usize = 16 };
const Journal = struct {
    a: std.mem.Allocator,
    io: std.Io,
    file: std.Io.File,
    fn write(j: Journal, value: anytype) !void {
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
    fn init(a: std.mem.Allocator, d: *cuda.Driver, stream: cuda.Stream) !Tap {
        const stamps = try a.alloc(Stamp, 400);
        for (stamps) |*stamp| stamp.* = .{ .event = try cuda.Event.init(d, true) };
        return .{ .stamps = stamps, .stream = stream };
    }
    pub fn layer(self: *Tap, id: usize) void {
        self.current_layer = id;
    }
    pub fn mark(self: *Tap, phase: qwen.DiagnosticPhase) !void {
        if (self.used >= self.stamps.len) return error.TraceOverflow;
        const stamp = &self.stamps[self.used];
        stamp.phase = phase;
        stamp.layer = self.current_layer;
        try stamp.event.record(self.stream);
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
    if (input.prompt.len != 64 or input.token >= 248320 or input.repetitions != 16) return error.InvalidInput;
    for (input.prompt) |id| if (id >= 248320) return error.InvalidInput;
    return input;
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
    const identities = try admitted.verify(a, &driver);
    driver.close();
    try j.write(.{ .kind = "start", .source_commit = options.source_commit, .identities = identities, .scope = "fixed-token fixed-position evolving-state forward diagnostic; not generation throughput", .position = input.prompt.len, .token = input.token, .repetitions = input.repetitions });
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
    var taps: [2]Tap = undefined;
    var phased: [2]Captured = undefined;
    for (&taps, &phased, 0..) |*tap, *graph, rank| {
        try e.partitions[rank].context.makeCurrent();
        tap.* = try Tap.init(a, &e.driver, e.partitions[rank].stream);
        graph.* = try capture(&s, rank, input.token, position, tap);
    }
    defer for (&taps, &phased, 0..) |*tap, *g, rank| {
        e.partitions[rank].context.makeCurrent() catch {};
        g.exec.deinit();
        tap.deinit();
    };
    for (0..5) |sample| {
        try baseline.restore(&s);
        const phased_began = now(init.io);
        for (0..input.repetitions) |_| try step(&s, input.token, position, &phased);
        const phased_elapsed = now(init.io) - phased_began;
        const digest = try stateHash(&s, hash_buffer);
        try baseline.restore(&s);
        const control_began = now(init.io);
        for (0..input.repetitions) |_| try step(&s, input.token, position, &graphs);
        const control_elapsed = now(init.io) - control_began;
        const control = try stateHash(&s, hash_buffer);
        if (!std.mem.eql(u8, &digest, &control)) return error.ProfileStateMismatch;
        try j.write(.{ .kind = "profile_turn", .sample = sample, .phased_elapsed_ns = phased_elapsed, .control_elapsed_ns = control_elapsed, .steps = input.repetitions });
        for (taps, 0..) |tap, rank| {
            try e.partitions[rank].context.makeCurrent();
            for (1..tap.used) |i| {
                const stamp = tap.stamps[i];
                try j.write(.{ .kind = "phase", .sample = sample, .rank = rank, .layer = stamp.layer, .phase = @tagName(stamp.phase), .ms = try cuda.Event.elapsedMs(tap.stamps[i - 1].event, stamp.event) });
            }
        }
    }
    try floor(e);
    try j.write(.{ .kind = "complete", .correctness_pass = true, .scope = "fixed-step greedy self-parity only; independent numerical gate remains open", .new_generation_speed_claim = false });
}
