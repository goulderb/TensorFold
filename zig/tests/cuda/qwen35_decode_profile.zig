const std = @import("std");
const builtin = @import("builtin");
const options = @import("qwen_bench_options");
const qwen = @import("qwen35_cuda");
const cuda = @import("cuda");
const admission = @import("qwen35_probe_device.zig");

comptime {
    if (options.engine_optimization != .fast) @compileError("diagnostics require ReleaseFast engine");
}

const Input = struct {
    prompt: []const u32,
    token: u32,
    repetitions: usize = 16,
    output_tokens: usize = 128,
    context: usize = 4096,
    cut: usize = 20,
    forced: []const u32 = &.{},
    router_turns: bool = false,
    router_screen: bool = false,
    router_candidate: bool = false,
    expert_fixture: ?[]const u8 = null,
    expert_fixture_sha256: ?[]const u8 = null,
};
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
fn finishHash(h: *std.crypto.hash.sha2.Sha256) [64]u8 {
    var digest: [32]u8 = undefined;
    h.final(&digest);
    return std.fmt.bytesToHex(digest, .lower);
}
fn hash(bytes: []const u8) [64]u8 {
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    h.update(bytes);
    return finishHash(&h);
}
fn parse(a: std.mem.Allocator, io: std.Io, path: []const u8) !Input {
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 20));
    const input = try std.json.parseFromSliceLeaky(Input, a, bytes, .{});
    if (input.prompt.len == 0 or input.output_tokens == 0 or input.output_tokens > 1024 or input.context > 262144 or input.context < input.output_tokens or input.prompt.len > input.context - input.output_tokens or input.forced.len > input.output_tokens or input.repetitions == 0) return error.InvalidInput;
    if (input.router_screen or input.router_candidate or input.expert_fixture != null or input.expert_fixture_sha256 != null) return error.ArchivedDiagnosticMode;
    return input;
}

fn stateHash(a: std.mem.Allocator, s: *qwen.Session) ![64]u8 {
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    h.update(std.mem.asBytes(&s.position));
    for (s.chain.partitions, 0..) |*p, rank| {
        try p.context.makeCurrent();
        try p.engine.stream.synchronize();
        const cache = &s.caches[rank];
        const buffer = try a.alloc(u8, @max(cache.state.len, s.position * 256 * 2));
        defer a.free(buffer);
        try cache.state.download(0, buffer[0..cache.state.len]);
        h.update(buffer[0..cache.state.len]);
        const g = p.engine.g;
        const used = s.position * 256 * 2;
        for (0..g.attentions() * 2 * g.kv_heads) |row| {
            try cache.kv.download(row * cache.cap * 256 * 2, buffer[0..used]);
            h.update(buffer[0..used]);
        }
    }
    return finishHash(&h);
}

fn captured(c: *qwen.Engine, enabled: bool) void {
    for (c.partitions) |*p| {
        p.engine.captured = enabled;
        p.engine.graph_launches = 0;
    }
}
fn argmax(row: []const u16) !u32 {
    var best: usize = 0;
    var best_value: f32 = -std.math.inf(f32);
    for (row, 0..) |bits, i| {
        const value: f32 = @bitCast(@as(u32, bits) << 16);
        if (!std.math.isFinite(value)) return error.NonfiniteLogits;
        if (value > best_value) {
            best = i;
            best_value = value;
        }
    }
    return @intCast(best);
}
const Timing = struct { prefill_ns: i96, decode_ns: i96, elapsed_ns: i96, appends: usize };
fn run(io: std.Io, s: *qwen.Session, input: Input, tokens: []u32, rows: []u16) !Timing {
    try s.reset();
    const start = now(io);
    try s.append(input.prompt);
    const prefilled = now(io);
    for (tokens, 0..) |*token, i| {
        @memcpy(rows[i * s.logits.len ..][0..s.logits.len], s.logits);
        const greedy = try argmax(s.logits);
        token.* = if (i < input.forced.len) input.forced[i] else greedy;
        if (i + 1 < tokens.len) try s.append(&.{token.*});
    }
    const end = now(io);
    return .{ .prefill_ns = prefilled - start, .decode_ns = end - prefilled, .elapsed_ns = end - start, .appends = input.prompt.len + tokens.len - 1 };
}

const Phase = qwen.shared_engine.Phase;
const Stamp = struct { event: cuda.Event, phase: Phase = .begin, layer: usize = 0 };
const Trace = struct {
    events: []Stamp,
    used: usize = 0,
    stream: cuda.Stream,
    fn init(a: std.mem.Allocator, e: *qwen.shared_engine.Engine) !Trace {
        const events = try a.alloc(Stamp, e.g.layers * 20 + 4);
        var made: usize = 0;
        errdefer for (events[0..made]) |*stamp| stamp.event.deinit();
        for (events) |*stamp| {
            stamp.* = .{ .event = try cuda.Event.init(e.d, true) };
            made += 1;
        }
        return .{ .events = events, .stream = e.stream };
    }
    fn mark(ptr: *anyopaque, phase: Phase, layer: usize) !void {
        const t: *Trace = @ptrCast(@alignCast(ptr));
        if (t.used >= t.events.len) return error.TraceFull;
        const stamp = &t.events[t.used];
        stamp.phase = phase;
        stamp.layer = layer;
        try stamp.event.record(t.stream);
        t.used += 1;
    }
    fn deinit(t: *Trace, a: std.mem.Allocator) void {
        for (t.events) |*stamp| stamp.event.deinit();
        a.free(t.events);
    }
};
fn trace(init: std.process.Init, s: *qwen.Session, input: Input, j: Journal) !void {
    const c = s.chain;
    captured(c, false);
    try s.reset();
    try s.append(input.prompt);
    var traces: [2]Trace = undefined;
    var count: usize = 0;
    defer for (c.partitions[0..count], traces[0..count]) |*p, *t| {
        p.context.makeCurrent() catch {};
        p.engine.observer = null;
        t.deinit(init.gpa);
    };
    for (c.partitions, 0..) |*p, rank| {
        try p.context.makeCurrent();
        traces[rank] = try Trace.init(init.gpa, p.engine);
        count += 1;
        p.engine.observer = .{ .ptr = &traces[rank], .mark = Trace.mark };
    }
    try s.append(&.{input.token});
    for (c.partitions, traces[0..count], 0..) |*p, t, rank| {
        try p.context.makeCurrent();
        for (1..t.used) |i| {
            const ms = try cuda.Event.elapsedMs(t.events[i - 1].event, t.events[i].event);
            try j.write(.{ .kind = "phase", .rank = rank, .layer = t.events[i].layer, .phase = @tagName(t.events[i].phase), .elapsed_ms = ms, .captured = false, .instrumented = true });
        }
    }
}
fn failStage(_: *anyopaque, _: Phase, _: usize) !void {
    return error.InjectedStageFailure;
}
fn poisonControl(s: *qwen.Session, input: Input) !void {
    captured(s.chain, false);
    try s.reset();
    try s.append(input.prompt);
    const position = s.position;
    const second = s.chain.partitions[1].engine;
    second.observer = .{ .ptr = s, .mark = failStage };
    s.append(&.{input.token}) catch |err| {
        second.observer = null;
        if (err != error.InjectedStageFailure) return err;
        if (!s.poisoned or s.position != position or s.caches[0].len != position or s.caches[1].len != position) return error.PartialCommit;
        s.append(&.{input.token}) catch |poison| {
            if (poison != error.PoisonedSession) return poison;
            try s.reset();
            if (s.poisoned or s.position != 0 or s.caches[0].len != 0 or s.caches[1].len != 0) return error.ResetFailed;
            return;
        };
        return error.MissingPoisonRefusal;
    };
    second.observer = null;
    return error.MissingInjectedFailure;
}

pub fn main(init: std.process.Init) !u8 {
    const a = init.arena.allocator();
    const args = try init.minimal.args.toSlice(a);
    if (args.len == 2 and std.mem.eql(u8, args[1], "--identity")) {
        var out: std.Io.Writer.Allocating = .init(a);
        try std.json.Stringify.value(.{ .source_commit = options.source_commit, .engine_optimization = @tagName(options.engine_optimization), .root_optimization = @tagName(builtin.mode), .implementation = "shared-qwen-engine" }, .{}, &out.writer);
        try out.writer.writeByte('\n');
        try std.Io.File.stdout().writeStreamingAll(init.io, out.written());
        return 0;
    }
    if (args.len == 3 and std.mem.eql(u8, args[1], "--check")) {
        _ = try parse(a, init.io, args[2]);
        return 0;
    }
    if (args.len != 4) return 2;
    const input = try parse(a, init.io, args[2]);
    try std.Io.Dir.cwd().createDir(init.io, args[3], .default_dir);
    const file = try std.Io.Dir.cwd().createFile(init.io, try std.fs.path.join(a, &.{ args[3], "results.jsonl" }), .{ .exclusive = true });
    defer file.close(init.io);
    const j: Journal = .{ .a = a, .io = init.io, .file = file };
    execute(init, args[1], args[3], input, j) catch |err| {
        try j.write(.{ .kind = "failure", .error_name = @errorName(err) });
        return 1;
    };
    return 0;
}

fn execute(init: std.process.Init, dir: []const u8, output: []const u8, input: Input, j: Journal) !void {
    const a = init.arena.allocator();
    const uuids = [_][]const u8{ "GPU-f2c0ea1e-a2be-f0b4-6c23-ffe33b537a06", "GPU-dd2a988c-fe38-24ab-233a-9e9bb3b2645a" };
    const admitted = try admission.Admission.parse(init.environ_map.get("CUDA_DEVICE_ORDER"), init.environ_map.get("CUDA_VISIBLE_DEVICES"), &.{ 0, 1 }, &uuids);
    const identities = blk: {
        var driver = try cuda.Driver.open();
        defer driver.close();
        break :blk try admitted.verify(a, &driver);
    };
    try j.write(.{ .kind = "start", .source_commit = options.source_commit, .implementation = "shared-qwen-engine", .identities = identities, .scope = "changing-token changing-position eager and captured shared Qwen forwards", .prompt_tokens = input.prompt.len, .output_tokens = input.output_tokens, .context = input.context, .cut = input.cut });
    const c = try qwen.Engine.open(init.gpa, init.io, dir, .{ .context = input.context, .cut = input.cut });
    defer c.deinit();
    var s = try c.createSession();
    defer s.deinit();
    const tokens = try a.alloc(u32, input.output_tokens);
    const expected_tokens = try a.alloc(u32, input.output_tokens);
    const rows = try a.alloc(u16, input.output_tokens * s.logits.len);
    const expected_rows = try a.alloc(u16, rows.len);
    captured(c, false);
    _ = try run(init.io, &s, input, expected_tokens, expected_rows);
    const expected_state = try stateHash(init.gpa, &s);
    const expected_hash = hash(std.mem.sliceAsBytes(expected_rows));
    try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = try std.fs.path.join(a, &.{ output, "consumed-logits.bf16" }), .data = std.mem.sliceAsBytes(expected_rows) });
    for (0..6) |pair| for (0..2) |order| {
        const graph = (pair + order) % 2 == 1;
        captured(c, graph);
        const timing = try run(init.io, &s, input, tokens, rows);
        const state = try stateHash(init.gpa, &s);
        if (!std.mem.eql(u16, rows, expected_rows) or !std.mem.eql(u32, tokens, expected_tokens) or !std.mem.eql(u8, &state, &expected_state)) return error.EagerCapturedMismatch;
        var launches: usize = 0;
        var nodes: usize = 0;
        for (c.partitions, 0..) |*p, rank| {
            launches += p.engine.graph_launches;
            nodes += p.engine.graph_nodes;
            try j.write(.{ .kind = "memory", .pair = pair, .order = order, .rank = rank, .free_bytes = (try p.context.memInfo()).free, .floor_bytes = qwen.floor_bytes });
        }
        if (graph and launches != 2 * timing.appends) return error.GraphWorkCountMismatch;
        try j.write(.{ .kind = "turn", .pair = pair, .order = order, .warmup = pair == 0, .arm = if (graph) "graph" else "eager", .graph = graph, .times = timing, .elapsed_ns = timing.elapsed_ns, .steps = timing.appends, .graph_launches = launches, .graph_nodes = nodes, .position = s.position, .exact_rows_tokens_and_final_state = true, .consumed_logits_sha256 = &expected_hash, .persistent_state_sha256 = &state, .tokens = tokens });
    };
    var peer = try c.createSession();
    defer peer.deinit();
    captured(c, true);
    try s.reset();
    try peer.reset();
    for (input.prompt) |token| {
        try s.append(&.{token});
        const saved = try a.dupe(u16, s.logits);
        defer a.free(saved);
        try peer.append(&.{token});
        if (!std.mem.eql(u16, saved, s.logits) or !std.mem.eql(u16, saved, peer.logits)) return error.SessionIsolationMismatch;
    }
    const first_state = try stateHash(init.gpa, &s);
    const peer_state = try stateHash(init.gpa, &peer);
    if (!std.mem.eql(u8, &first_state, &peer_state)) return error.SessionStateMismatch;
    try poisonControl(&s, input);
    _ = try run(init.io, &s, input, tokens, rows);
    if (!std.mem.eql(u16, rows, expected_rows)) return error.PoisonResetMismatch;
    try j.write(.{ .kind = "controls", .all_consumed_rows_equal = true, .changing_tokens_and_positions = true, .interleaved_session_isolation = true, .reset = true, .partial_stage_failure_poison = true, .uncommitted_lengths = true });
    try trace(init, &s, input, j);
    try j.write(.{ .kind = "complete", .correctness_pass = true, .scope = "shared engine eager/captured parity and session controls", .production_qualification = false, .independent_reference_parity = "not checked by this executable" });
}
