const std = @import("std");
const builtin = @import("builtin");
const options = @import("qwen_bench_options");
const qwen = @import("qwen35_cuda");
const cuda = @import("cuda");
const admission = @import("qwen35_probe_device.zig");
const expert_screen = @import("qwen35_expert_screen.zig");
const expert_latency_screen = @import("qwen35_expert_latency_screen.zig");
const gdn_layout_screen = @import("qwen35_gdn_layout_screen.zig");
const state_layout = @import("qwen35_state_layout.zig");
const quant_reuse_screen = @import("qwen35_quant_reuse_screen.zig");
const transport_screen = @import("qwen35_transport_screen.zig");
const prefill_screen = @import("qwen35_prefill_screen.zig");
const operator_suite = @import("qwen35_operator_suite.zig");
const mma_capture_screen = @import("qwen35_mma_capture_screen.zig");
const ck_stage_screen = @import("qwen35_ck_stage_screen.zig");
const slice_major_input = @import("qwen35_slice_major_input.zig");
const slice_major_screen = @import("qwen35_slice_major_screen.zig");
const boundary_input = @import("qwen35_boundary_input.zig");
const terminal_greedy_screen = @import("qwen35_terminal_greedy_screen.zig");
const shared_activation_screen = @import("qwen35_shared_activation_screen.zig");

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
    expert_n8_screen: bool = false,
    expert_shard_screen: bool = false,
    expert_latency_screen: bool = false,
    gdn_layout_screen: bool = false,
    gdn_layout_ab: bool = false,
    expert_n8_ab: bool = false,
    quant_reuse_screen: bool = false,
    transport_screen: bool = false,
    verify_default_policy: bool = false,
    prefill_operator_screen: bool = false,
    prefill_ab: bool = false,
    combined_ab: bool = false,
    operator_suite: ?operator_suite.Suite = null,
    slice_major_screen: ?slice_major_input.Screen = null,
    boundary_screen: ?boundary_input.Screen = null,
};
const Journal = struct {
    a: std.mem.Allocator,
    io: std.Io,
    file: std.Io.File,
    directory: []const u8,
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
fn kernelName(f: cuda.Function) ![]const u8 {
    var lib = f.d.lib;
    const get_name = lib.lookup(*const fn (*?[*:0]const u8, cuda.abi.Function) callconv(.c) cuda.abi.Result, "cuFuncGetName") orelse return error.MissingSymbol;
    var name: ?[*:0]const u8 = null;
    try f.d.check(get_name(&name, f.handle), "cuFuncGetName");
    return std.mem.span(name orelse return error.MissingSymbol);
}
fn bundleResources(j: Journal, rank: usize, arm: []const u8, f: cuda.Function, resident: usize, threads: u32, grid_cap: ?usize) !void {
    try j.write(.{ .kind = "bundle_resources", .rank = rank, .arm = arm, .symbol = try kernelName(f), .resident_blocks = resident, .threads = threads, .grid_cap = grid_cap, .regs = try f.attribute(.num_regs), .local_bytes = try f.attribute(.local_size_bytes), .static_shared_bytes = try f.attribute(.shared_size_bytes), .dynamic_shared_bytes = 0, .resident_blocks_per_sm = try f.occupancy(threads, 0) });
}
fn parse(a: std.mem.Allocator, io: std.Io, path: []const u8) !Input {
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 20));
    const input = try std.json.parseFromSliceLeaky(Input, a, bytes, .{});
    if (input.prompt.len == 0 or input.output_tokens == 0 or input.output_tokens > 1024 or input.context > 262144 or input.context < input.output_tokens or input.prompt.len > input.context - input.output_tokens or input.forced.len > input.output_tokens or input.repetitions == 0) return error.InvalidInput;
    try expert_screen.validateShardInput(input);
    try expert_latency_screen.validateInput(input);
    try prefill_screen.validateInput(input);
    try operator_suite.validateInput(input);
    try slice_major_input.validateInput(input);
    try boundary_input.validateInput(input);
    if ((input.prefill_ab or input.combined_ab) and (input.prompt.len != 64 or input.output_tokens != 128 or input.context != 4096 or input.cut != 20)) return error.InvalidInput;
    if (input.router_screen or input.router_candidate) return error.ArchivedDiagnosticMode;
    if (input.gdn_layout_ab and input.expert_n8_ab) return error.InvalidInput;
    if ((input.gdn_layout_ab or input.expert_n8_ab) and (input.expert_fixture != null or input.gdn_layout_screen or input.quant_reuse_screen or input.transport_screen or input.forced.len != 0)) return error.InvalidInput;
    if (input.expert_n8_screen and (input.expert_fixture == null or input.gdn_layout_screen)) return error.InvalidInput;
    if (input.verify_default_policy and (input.gdn_layout_ab or input.expert_n8_ab or input.expert_fixture != null or input.gdn_layout_screen or input.quant_reuse_screen or input.transport_screen or input.forced.len != 0)) return error.InvalidInput;
    if ((input.expert_fixture == null) != (input.expert_fixture_sha256 == null)) return error.InvalidExpertFixture;
    if (input.expert_fixture_sha256) |sha| {
        if (sha.len != 64) return error.InvalidExpertFixture;
        for (sha) |c| if (!(c >= '0' and c <= '9') and !(c >= 'a' and c <= 'f')) return error.InvalidExpertFixture;
    }
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

fn canonicalState(a: std.mem.Allocator, s: *qwen.Session) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(a);
    try out.appendSlice(a, "qwen35-canonical-state-v2");
    try out.appendSlice(a, std.mem.asBytes(&s.position));
    for (s.chain.partitions, 0..) |*p, rank| {
        try p.context.makeCurrent();
        try p.engine.stream.synchronize();
        const cache = &s.caches[rank];
        const g = p.engine.g;
        const used = s.position * 256 * 2;
        const bytes = try a.alloc(u8, @max(cache.state.len, used));
        defer a.free(bytes);
        try cache.state.download(0, bytes[0..cache.state.len]);
        if (cache.gdn_layout == .key_value) {
            const scratch = try a.alloc(u8, g.deltas() * g.recBytes());
            defer a.free(scratch);
            try state_layout.canonicalize(bytes[0..cache.state.len], scratch, g.deltas(), g.v_heads, 128);
        }
        try out.appendSlice(a, std.mem.asBytes(&cache.len));
        try out.appendSlice(a, std.mem.asBytes(&cache.written));
        try out.appendSlice(a, std.mem.asBytes(&cache.delta));
        try out.appendSlice(a, bytes[0..cache.state.len]);
        for (0..g.attentions() * 2 * g.kv_heads) |row| {
            try cache.kv.download(row * cache.cap * 256 * 2, bytes[0..used]);
            try out.appendSlice(a, bytes[0..used]);
        }
    }
    try out.appendSlice(a, std.mem.sliceAsBytes(s.logits));
    return out.toOwnedSlice(a);
}

fn layoutCheck(a: std.mem.Allocator, s: *qwen.Session, expected: []const u8, rows: []const u16, expected_rows: []const u16, tokens: []const u32, expected_tokens: []const u32) !void {
    const state = try canonicalState(a, s);
    defer a.free(state);
    if (!std.mem.eql(u8, state, expected) or !std.mem.eql(u16, rows, expected_rows) or !std.mem.eql(u32, tokens, expected_tokens)) return error.GdnLayoutFullModelMismatch;
}

const Trial = enum {
    gdn_layout,
    expert_n8,

    fn kind(self: Trial, comptime suffix: []const u8) []const u8 {
        return switch (self) {
            .gdn_layout => "layout_" ++ suffix,
            .expert_n8 => "expert_" ++ suffix,
        };
    }

    fn select(self: Trial, c: *qwen.Engine, arm: usize, graph: bool) void {
        captured(c, graph);
        for (c.partitions) |*p| p.engine.expert_up_n8 = self == .expert_n8 and arm == 1;
    }
};

fn trialControls(init: std.process.Init, s: *qwen.Session, peer: *qwen.Session, input: Input, expected: []const u8, expected_rows: []const u16, expected_tokens: []const u32, rows: []u16, tokens: []u32, j: Journal, arm: []const u8, trial: Trial, index: usize) !void {
    const a = init.gpa;
    trial.select(s.chain, index, true);
    try s.reset();
    try peer.reset();
    try s.append(input.prompt[0..1]);
    const saved = try canonicalState(a, s);
    defer a.free(saved);
    const different = [_]u32{ (input.prompt[0] + 1) % @as(u32, @intCast(s.logits.len)), (input.prompt[0] + 2) % @as(u32, @intCast(s.logits.len)) };
    try peer.append(&different);
    const after_peer = try canonicalState(a, s);
    defer a.free(after_peer);
    if (!std.mem.eql(u8, saved, after_peer)) return error.SessionIsolationMismatch;
    const peer_saved = try canonicalState(a, peer);
    defer a.free(peer_saved);
    try s.append(input.prompt[1..]);
    const peer_after = try canonicalState(a, peer);
    defer a.free(peer_after);
    if (!std.mem.eql(u8, peer_saved, peer_after)) return error.SessionIsolationMismatch;
    try poisonControl(s, input);
    _ = try run(init.io, s, input, tokens, rows);
    try layoutCheck(a, s, expected, rows, expected_rows, tokens, expected_tokens);
    try j.write(.{ .kind = trial.kind("controls"), .arm = arm, .distinct_prompts_and_positions = true, .inactive_complete_state_unchanged = true, .reset_exact = true, .partial_stage_failure_poison = true, .uncommitted_lengths = true, .canonical_state_version = 2 });
}

fn trialAB(init: std.process.Init, c: *qwen.Engine, input: Input, output: []const u8, j: Journal, trial: Trial) !void {
    const a = init.gpa;
    for (c.partitions) |p| {
        const g = p.engine.g;
        if (p.engine.kv8 or g.recBytes() != g.v_heads * 128 * 128 * 4 or g.kvBytes(c.context, false) != g.attentions() * 2 * g.kv_heads * c.context * 256 * 2) return error.UnsupportedCanonicalStateGeometry;
    }
    var baseline = try c.createSessionWithLayout(if (trial == .gdn_layout) .value_key else .key_value);
    defer baseline.deinit();
    var candidate = try c.createSessionWithLayout(.key_value);
    defer candidate.deinit();
    const tokens = try a.alloc(u32, input.output_tokens);
    defer a.free(tokens);
    const expected_tokens = try a.alloc(u32, input.output_tokens);
    defer a.free(expected_tokens);
    const rows = try a.alloc(u16, input.output_tokens * baseline.logits.len);
    defer a.free(rows);
    const expected_rows = try a.alloc(u16, rows.len);
    defer a.free(expected_rows);
    trial.select(c, 0, false);
    _ = try run(init.io, &baseline, input, expected_tokens, expected_rows);
    const expected = try canonicalState(a, &baseline);
    defer a.free(expected);
    const state_hash = hash(expected);
    const logit_hash = hash(std.mem.sliceAsBytes(expected_rows));
    try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = try std.fs.path.join(init.arena.allocator(), &.{ output, "consumed-logits.bf16" }), .data = std.mem.sliceAsBytes(expected_rows) });
    try j.write(.{ .kind = trial.kind("reference"), .arm = "baseline", .graph = false, .tokens = expected_tokens, .consumed_logits_sha256 = &logit_hash, .canonical_state_sha256 = &state_hash, .canonical_state_bytes = expected.len, .canonical_state_version = 2 });
    const sessions = [_]*qwen.Session{ &baseline, &candidate };
    const names = [_][]const u8{ "baseline", "candidate" };
    for (c.partitions) |*p| {
        p.engine.verify_gdn_graphs = trial == .gdn_layout;
        p.engine.verify_expert_graphs = trial == .expert_n8;
    }
    for (0..2) |mode| {
        for (sessions, names, 0..) |session, name, index| {
            trial.select(c, index, mode == 1);
            _ = try run(init.io, session, input, tokens, rows);
            try layoutCheck(a, session, expected, rows, expected_rows, tokens, expected_tokens);
            try j.write(.{ .kind = trial.kind("qualification"), .arm = name, .graph = mode == 1, .all_rows_tokens_and_canonical_state_exact = true });
        }
    }
    for (c.partitions, 0..) |*p, rank| {
        if (trial == .gdn_layout) {
            try j.write(.{ .kind = "layout_graph_bindings", .rank = rank, .expected_gdn_nodes = p.engine.g.deltas(), .baseline_verified_nodes = p.engine.gdn_graph_nodes_baseline, .candidate_verified_nodes = p.engine.gdn_graph_nodes_candidate });
        } else {
            try j.write(.{ .kind = "expert_graph_bindings", .rank = rank, .expected_expert_nodes = p.engine.g.layers, .baseline_verified_nodes = p.engine.expert_graph_nodes_baseline, .candidate_verified_nodes = p.engine.expert_graph_nodes_n8 });
        }
    }
    for (0..6) |pair| for (0..2) |order| {
        const index = (pair + order) % 2;
        const session = sessions[index];
        trial.select(c, index, true);
        const timing = try run(init.io, session, input, tokens, rows);
        try layoutCheck(a, session, expected, rows, expected_rows, tokens, expected_tokens);
        var launches: usize = 0;
        var nodes: usize = 0;
        for (c.partitions, 0..) |*p, rank| {
            launches += p.engine.graph_launches;
            nodes += p.engine.graph_nodes;
            try j.write(.{ .kind = trial.kind("memory"), .pair = pair, .order = order, .rank = rank, .free_bytes = (try p.context.memInfo()).free, .floor_bytes = qwen.floor_bytes });
        }
        if (launches != c.partitions.len * timing.appends) return error.GraphWorkCountMismatch;
        try j.write(.{ .kind = trial.kind("turn"), .pair = pair, .order = order, .warmup = pair == 0, .arm = names[index], .graph = true, .times = timing, .graph_launches = launches, .graph_nodes = nodes, .position = session.position, .all_rows_tokens_and_canonical_state_exact = true, .consumed_logits_sha256 = &logit_hash, .canonical_state_sha256 = &state_hash, .canonical_state_version = 2, .tokens = tokens });
    };
    for (sessions, names, 0..) |session, name, index| try trialControls(init, session, sessions[1 - index], input, expected, expected_rows, expected_tokens, rows, tokens, j, name, trial, index);
    {
        var recreated = try c.createSessionWithLayout(.key_value);
        defer recreated.deinit();
        trial.select(c, 1, true);
        _ = try run(init.io, &recreated, input, tokens, rows);
        try layoutCheck(a, &recreated, expected, rows, expected_rows, tokens, expected_tokens);
        trial.select(c, 0, true);
        _ = try run(init.io, &baseline, input, tokens, rows);
        try layoutCheck(a, &baseline, expected, rows, expected_rows, tokens, expected_tokens);
        try j.write(.{ .kind = trial.kind("recreated_session"), .candidate_exact = true, .baseline_after_candidate_exact = true });
    }
    try j.write(.{ .kind = "complete", .correctness_pass = true, .scope = if (trial == .gdn_layout) "single-load captured GDN layout A/B with complete canonical state and logit parity" else "single-load captured expert N8 A/B with complete canonical state and logit parity", .production_qualification = false, .independent_reference_parity = "not checked by this executable" });
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

fn prefillMode(c: *qwen.Engine, rows: usize, graph: bool, bundle: bool) void {
    c.prefill_rows = rows;
    captured(c, graph);
    for (c.partitions) |*p| {
        p.engine.bounded_hf_prefill = rows > 1;
        p.engine.expert_up_split_m = bundle;
        p.engine.reuse_fp8_quant = bundle;
    }
}

fn prefillLaunches(c: *qwen.Engine) usize {
    var count: usize = 0;
    for (c.partitions) |p| count += p.engine.graph_launches;
    return count;
}

fn prefillFile(j: Journal, name: []const u8, bytes: []const u8) !void {
    const path = try std.fs.path.join(j.a, &.{ j.directory, name });
    const file = try std.Io.Dir.cwd().createFile(j.io, path, .{ .exclusive = true });
    defer file.close(j.io);
    try file.writeStreamingAll(j.io, bytes);
    try j.write(.{ .kind = "prefill_file", .name = name, .bytes = bytes.len, .sha256 = &hash(bytes) });
}

fn prefillMetadata(s: *qwen.Session) !void {
    if (s.poisoned) return error.PoisonedSession;
    for (s.caches[0..s.chain.partitions.len]) |cache| {
        if (cache.len != s.position or cache.written != s.position or cache.last != null) return error.PrefillMetadataMismatch;
    }
}

fn prefillState(a: std.mem.Allocator, serial: *qwen.Session, batch: *qwen.Session, j: Journal, label: []const u8, save: bool) ![64]u8 {
    try prefillMetadata(serial);
    try prefillMetadata(batch);
    const expected = try canonicalState(a, serial);
    defer a.free(expected);
    const actual = try canonicalState(a, batch);
    defer a.free(actual);
    const exact = std.mem.eql(u8, expected, actual);
    if (save or !exact) {
        try prefillFile(j, try std.fmt.allocPrint(j.a, "{s}-serial-state.bin", .{label}), expected);
        try prefillFile(j, try std.fmt.allocPrint(j.a, "{s}-batch-state.bin", .{label}), actual);
    }
    try j.write(.{ .kind = "prefill_state", .label = label, .bytes = expected.len, .serial_sha256 = &hash(expected), .batch_sha256 = &hash(actual), .exact = exact, .position = serial.position, .batch_position = batch.position, .last_null = true, .written_equals_position = true, .canonical_version = 2, .raw_saved = save or !exact });
    if (!exact) return error.PrefillStateMismatch;
    return hash(expected);
}

fn prefillPrefix(init: std.process.Init, serial: *qwen.Session, batch: *qwen.Session, input: Input, rows: usize, start: usize, graph: bool, serial_rows: []u16, batch_rows: []u16, j: Journal) !void {
    const c = serial.chain;
    const vocab = serial.logits.len;
    const count = rows * vocab;
    const reference = serial_rows[0..count];
    const candidate = batch_rows[0..count];
    prefillMode(c, 1, graph, false);
    try serial.reset();
    if (start != 0) try serial.append(input.prompt[0..start]);
    try serial.appendLogitRows(input.prompt[0..rows], reference);
    const serial_launches = prefillLaunches(c);
    prefillMode(c, 16, graph, input.combined_ab);
    try batch.reset();
    if (start != 0) try batch.append(input.prompt[0..start]);
    try batch.appendLogitRows(input.prompt[0..rows], candidate);
    const batch_launches = prefillLaunches(c);
    if (serial_launches != (if (graph) (start + rows) * c.partitions.len else 0) or
        batch_launches != (if (graph) ((start + 15) / 16 + (rows + 15) / 16) * c.partitions.len else 0)) return error.PrefillGraphWorkCount;
    const label = try std.fmt.allocPrint(j.a, "prefix-{s}-start{d}-rows{d}", .{ if (graph) "captured" else "eager", start, rows });
    try prefillFile(j, try std.fmt.allocPrint(j.a, "{s}-serial-logits.bf16", .{label}), std.mem.sliceAsBytes(reference));
    try prefillFile(j, try std.fmt.allocPrint(j.a, "{s}-batch-logits.bf16", .{label}), std.mem.sliceAsBytes(candidate));
    if (!std.mem.eql(u16, reference, candidate)) return error.PrefillPromptLogitsMismatch;
    for (0..rows) |row| _ = try argmax(reference[row * vocab ..][0..vocab]);
    const state = try prefillState(init.gpa, serial, batch, j, label, graph and rows == 64 and start == 7);
    const continuation = [_]u32{ input.token, (input.token + 1) % @as(u32, @intCast(vocab)), (input.token + 2) % @as(u32, @intCast(vocab)) };
    for (continuation, 0..) |token, r| {
        prefillMode(c, 1, graph, false);
        try serial.append(&.{token});
        @memcpy(serial_rows[r * vocab ..][0..vocab], serial.logits);
        prefillMode(c, 16, graph, input.combined_ab);
        try batch.append(&.{token});
        @memcpy(batch_rows[r * vocab ..][0..vocab], batch.logits);
        if (!std.mem.eql(u16, serial.logits, batch.logits)) {
            try prefillFile(j, try std.fmt.allocPrint(j.a, "{s}-continuation-row{d}-serial-failure.bf16", .{ label, r }), std.mem.sliceAsBytes(serial.logits));
            try prefillFile(j, try std.fmt.allocPrint(j.a, "{s}-continuation-row{d}-batch-failure.bf16", .{ label, r }), std.mem.sliceAsBytes(batch.logits));
            return error.PrefillContinuationMismatch;
        }
        _ = try argmax(serial.logits);
    }
    const after = try std.fmt.allocPrint(j.a, "{s}-continuation", .{label});
    try prefillFile(j, try std.fmt.allocPrint(j.a, "{s}-serial-logits.bf16", .{after}), std.mem.sliceAsBytes(serial_rows[0 .. 3 * vocab]));
    try prefillFile(j, try std.fmt.allocPrint(j.a, "{s}-batch-logits.bf16", .{after}), std.mem.sliceAsBytes(batch_rows[0 .. 3 * vocab]));
    const after_state = try prefillState(init.gpa, serial, batch, j, after, graph and rows == 64 and start == 7);
    try j.write(.{ .kind = "prefill_prefix", .graph = graph, .start = start, .rows = rows, .position = batch.position, .serial_graph_launches = serial_launches, .batch_graph_launches = batch_launches, .all_row_logits_exact = true, .state_sha256 = &state, .continuation_state_sha256 = &after_state, .continuation_tokens = continuation, .continuation_rows = 3, .canonical_state_exact = true });
}

fn prefillSessionControls(init: std.process.Init, serial: *qwen.Session, batch: *qwen.Session, input: Input, j: Journal) !void {
    const a = init.gpa;
    const c = serial.chain;
    prefillMode(c, 16, true, input.combined_ab);
    try serial.reset();
    try batch.reset();
    try serial.append(input.prompt[0..16]);
    const saved_serial = try canonicalState(a, serial);
    defer a.free(saved_serial);
    var different: [17]u32 = undefined;
    for (&different, input.prompt[0..17]) |*dst, token| dst.* = (token + 7) % @as(u32, @intCast(serial.logits.len));
    try batch.append(&different);
    const after_peer = try canonicalState(a, serial);
    defer a.free(after_peer);
    if (!std.mem.eql(u8, saved_serial, after_peer)) return error.PrefillSessionIsolation;
    const saved_batch = try canonicalState(a, batch);
    defer a.free(saved_batch);
    try serial.append(input.prompt[16..32]);
    const after_serial = try canonicalState(a, batch);
    defer a.free(after_serial);
    if (!std.mem.eql(u8, saved_batch, after_serial)) return error.PrefillSessionIsolation;
    prefillMode(c, 16, false, input.combined_ab);
    try batch.reset();
    const empty = try canonicalState(a, batch);
    defer a.free(empty);
    try batch.append(input.prompt[0..16]);
    const position = batch.position;
    const second = c.partitions[1].engine;
    second.observer = .{ .ptr = batch, .mark = failStage };
    batch.append(input.prompt[16..32]) catch |err| {
        second.observer = null;
        if (err != error.InjectedStageFailure) return err;
        if (!batch.poisoned or batch.position != position or batch.caches[0].len != position or batch.caches[1].len != position or batch.caches[0].written != position or batch.caches[1].written != position or batch.caches[0].last != null or batch.caches[1].last != null) return error.PartialCommit;
        batch.append(&.{input.token}) catch |poison| {
            if (poison != error.PoisonedSession) return poison;
            try batch.reset();
            const reset = try canonicalState(a, batch);
            defer a.free(reset);
            if (!std.mem.eql(u8, reset, empty)) return error.PrefillResetMismatch;
            try j.write(.{ .kind = "prefill_controls", .distinct_prompts_and_positions = true, .inactive_complete_state_unchanged = true, .serial_before_peer_sha256 = &hash(saved_serial), .serial_after_peer_sha256 = &hash(after_peer), .peer_before_serial_sha256 = &hash(saved_batch), .peer_after_serial_sha256 = &hash(after_serial), .partial_stage_failure_rows = 16, .partial_stage_failure_poison = true, .uncommitted_lengths = true, .uncommitted_written = true, .last_null = true, .reset_complete_state_exact = true, .reset_reference_sha256 = &hash(empty), .reset_sha256 = &hash(reset) });
            return;
        };
        return error.MissingPoisonRefusal;
    };
    second.observer = null;
    return error.MissingInjectedFailure;
}

fn prefillAB(init: std.process.Init, c: *qwen.Engine, input: Input, output: []const u8, j: Journal) !void {
    _ = output;
    if (c.partitions.len != 2) return error.PrefillRequiresTwoRanks;
    for (c.partitions) |*p| {
        p.engine.verify_gdn_graphs = true;
        p.engine.verify_expert_graphs = true;
    }
    var serial = try c.createSession();
    defer serial.deinit();
    var batch = try c.createSession();
    defer batch.deinit();
    const a = init.gpa;
    const vocab = serial.logits.len;
    const serial_rows = try a.alloc(u16, input.output_tokens * vocab);
    defer a.free(serial_rows);
    const batch_rows = try a.alloc(u16, serial_rows.len);
    defer a.free(batch_rows);
    for ([_]bool{ false, true }) |graph| for ([_]usize{ 0, 7 }) |start| {
        for ([_]usize{ 1, 2, 3, 15, 16, 17, 31, 32, 64 }) |rows| try prefillPrefix(init, &serial, &batch, input, rows, start, graph, serial_rows, batch_rows, j);
    };
    prefillMode(c, 1, false, false);
    const expected_tokens = try a.alloc(u32, input.output_tokens);
    defer a.free(expected_tokens);
    const tokens = try a.alloc(u32, input.output_tokens);
    defer a.free(tokens);
    _ = try run(init.io, &serial, input, expected_tokens, serial_rows);
    try prefillFile(j, "turn-reference-logits.bf16", std.mem.sliceAsBytes(serial_rows));
    const expected_state = try canonicalState(a, &serial);
    defer a.free(expected_state);
    try prefillFile(j, "turn-reference-state.bin", expected_state);
    for (0..2) |arm| {
        const rows: usize = if (input.combined_ab or arm == 1) 16 else 1;
        const bundle = input.combined_ab and arm == 1;
        prefillMode(c, rows, true, bundle);
        _ = try run(init.io, &batch, input, tokens, batch_rows);
        const state = try canonicalState(a, &batch);
        defer a.free(state);
        if (!std.mem.eql(u16, batch_rows, serial_rows) or !std.mem.eql(u32, tokens, expected_tokens) or !std.mem.eql(u8, state, expected_state)) {
            try prefillFile(j, try std.fmt.allocPrint(j.a, "normal-gate-rows{d}-failure-logits.bf16", .{rows}), std.mem.sliceAsBytes(batch_rows));
            try prefillFile(j, try std.fmt.allocPrint(j.a, "normal-gate-rows{d}-failure-state.bin", .{rows}), state);
            return error.PrefillTurnMismatch;
        }
        try j.write(.{ .kind = "prefill_normal_gate", .prefill_rows = rows, .bundle_enabled = bundle, .graph_launches = prefillLaunches(c), .logits_sha256 = &hash(std.mem.sliceAsBytes(batch_rows)), .state_sha256 = &hash(state), .validators_enabled = true, .timing_reported = false });
    }
    var qualified_shapes: [2]usize = undefined;
    var qualified_nodes: [2]usize = undefined;
    for (c.partitions, 0..) |*p, rank| {
        const e = p.engine;
        if (!e.verify_gdn_graphs or !e.verify_expert_graphs or e.gdn_graph_nodes_candidate == 0 or e.expert_graph_nodes_n8 == 0 or e.gdn_graph_nodes_baseline != 0 or e.expert_graph_nodes_baseline != 0) return error.PrefillBindingProof;
        if (input.combined_ab and (e.expert_graph_nodes_split_m == 0 or e.quant_graph_nodes_reuse == 0 or e.quant_graphs_verified == 0)) return error.PrefillBindingProof;
        try j.write(.{ .kind = "prefill_bindings", .rank = rank, .gdn_candidate_nodes = e.gdn_graph_nodes_candidate, .gdn_baseline_nodes = e.gdn_graph_nodes_baseline, .expert_n8_nodes = e.expert_graph_nodes_n8, .expert_split_m_nodes = e.expert_graph_nodes_split_m, .expert_baseline_nodes = e.expert_graph_nodes_baseline, .quant_baseline_nodes = e.quant_graph_nodes_baseline, .quant_reuse_nodes = e.quant_graph_nodes_reuse, .quant_graphs_verified = e.quant_graphs_verified, .graph_nodes = e.graph_nodes, .cached_shapes = e.graphs.count(), .validators_disabled_after_qualification = true });
        if (input.combined_ab) {
            try p.context.makeCurrent();
            const experts = e.modelopt.?.nvfp4_experts;
            const n8 = experts.n8_up orelse return error.PrefillBindingProof;
            const split = experts.split_m_up orelse return error.PrefillBindingProof;
            try bundleResources(j, rank, "n8", n8.function, n8.resident, cuda.experts.Nvfp4Experts.N8Up.threads, null);
            try bundleResources(j, rank, "split_m", split.function, split.resident, cuda.experts.Nvfp4Experts.SplitMUp.threads, 128);
            for (e.expert_graph_roster.items) |node| try j.write(.{ .kind = "bundle_geometry", .rank = rank, .node = node });
        }
        e.verify_gdn_graphs = false;
        e.verify_expert_graphs = false;
        qualified_shapes[rank] = e.graphs.count();
        qualified_nodes[rank] = e.graph_nodes;
    }
    for (0..6) |pair| for (0..2) |order| {
        const candidate = (pair + order) % 2 == 1;
        prefillMode(c, if (candidate or input.combined_ab) 16 else 1, true, input.combined_ab and candidate);
        for (c.partitions) |p| if (p.engine.verify_gdn_graphs or p.engine.verify_expert_graphs) return error.PrefillTimingValidators;
        const timing = try run(init.io, &batch, input, tokens, batch_rows);
        for (c.partitions, 0..) |p, rank| if (p.engine.graphs.count() != qualified_shapes[rank] or p.engine.graph_nodes != qualified_nodes[rank]) return error.PrefillUnqualifiedGraph;
        const state = try canonicalState(a, &batch);
        defer a.free(state);
        const exact = std.mem.eql(u16, batch_rows, serial_rows) and std.mem.eql(u32, tokens, expected_tokens) and std.mem.eql(u8, state, expected_state);
        if (pair == 0 or !exact) {
            const label = try std.fmt.allocPrint(j.a, "turn-pair{d}-order{d}", .{ pair, order });
            try prefillFile(j, try std.fmt.allocPrint(j.a, "{s}-logits.bf16", .{label}), std.mem.sliceAsBytes(batch_rows));
            try prefillFile(j, try std.fmt.allocPrint(j.a, "{s}-state.bin", .{label}), state);
        }
        if (!exact) return error.PrefillTurnMismatch;
        try prefillMetadata(&batch);
        const launches = prefillLaunches(c);
        const body_calls = (if (candidate or input.combined_ab) @as(usize, 4) else 64) + input.output_tokens - 1;
        if (launches != body_calls * c.partitions.len) return error.PrefillGraphWorkCount;
        try j.write(.{ .kind = "prefill_turn", .pair = pair, .order = order, .warmup = pair == 0, .arm = if (input.combined_ab) (if (candidate) @as([]const u8, "bundle_on") else "bundle_off") else (if (candidate) @as([]const u8, "batch16") else "serial1"), .bundle_enabled = input.combined_ab and candidate, .prefill_rows = c.prefill_rows, .times = timing, .body_calls_per_rank = body_calls, .graph_launches = launches, .validators_enabled = false, .qualified_graph_cache_unchanged = true, .all_consumed_logits_tokens_and_state_exact = true, .logits_sha256 = &hash(std.mem.sliceAsBytes(batch_rows)), .state_sha256 = &hash(state), .tokens = tokens });
    };
    try prefillSessionControls(init, &serial, &batch, input, j);
    try j.write(.{ .kind = "complete", .correctness_pass = true, .scope = if (input.combined_ab) "shared Engine retained-candidate bundle parity and matched C=1 turns" else "shared Engine bounded prefill parity and matched C=1 turns", .production_qualification = false, .independent_hf_quality_gate = "unchanged FAIL" });
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
    const j: Journal = .{ .a = a, .io = init.io, .file = file, .directory = args[3] };
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
    const single_gpu_micro = input.expert_fixture != null or input.gdn_layout_screen or input.quant_reuse_screen or input.prefill_operator_screen or input.operator_suite != null or input.slice_major_screen != null or input.boundary_screen != null;
    const micro = single_gpu_micro or input.transport_screen;
    if (micro) {
        try j.write(.{ .kind = "start", .source_commit = options.source_commit, .implementation = "shared-qwen-engine", .identities = identities, .scope = if (input.boundary_screen) |screen| boundary_input.scope(screen) else if (input.slice_major_screen != null) "isolated slice-major expert-up screen; no model throughput claim" else if (input.operator_suite != null) "isolated CK pipeline and FP8 intermediate screens; no complete model load or model throughput claim" else if (input.prefill_operator_screen) "isolated partial-checkpoint prefill operators; no complete model load or model throughput claim" else "isolated Qwen operator screens; no model load or model throughput claim", .driver_kind = "qwen35-decode-profile", .expert_shard_screen = input.expert_shard_screen, .expert_latency_screen = input.expert_latency_screen, .prefill_operator_screen = input.prefill_operator_screen, .operator_suite = input.operator_suite, .slice_major_screen = input.slice_major_screen, .boundary_screen = input.boundary_screen, .prompt_tokens = input.prompt.len, .output_tokens = input.output_tokens, .context = input.context, .cut = input.cut });
        if (input.transport_screen) try transport_screen.run(init.gpa, init.io, .{ identities[0].ordinal, identities[1].ordinal }, j);
        if (single_gpu_micro) for (identities) |identity| {
            var driver = try cuda.Driver.open();
            defer driver.close();
            var context = try cuda.Context.init(&driver, identity.ordinal);
            defer context.deinit();
            const gpu: @import("check.zig").Gpu = .{ .d = &driver, .ctx = &context, .gpa = init.gpa, .io = init.io };
            if (input.expert_fixture) |fixture| {
                if (input.expert_latency_screen) try expert_latency_screen.run(gpu, fixture, input.expert_fixture_sha256.?, j) else if (input.expert_shard_screen) try expert_screen.runShards(gpu, fixture, input.expert_fixture_sha256.?, j) else if (input.expert_n8_screen) try expert_screen.runN8(gpu, fixture, input.expert_fixture_sha256.?, j) else try expert_screen.run(gpu, fixture, input.expert_fixture_sha256.?, j);
            }
            if (input.gdn_layout_screen) try gdn_layout_screen.run(gpu, j);
            if (input.quant_reuse_screen) try quant_reuse_screen.run(gpu, j);
            if (input.prefill_operator_screen) try prefill_screen.run(gpu, dir, j);
            if (input.operator_suite) |suite| {
                try ck_stage_screen.run(gpu, dir, suite.stage3.path, suite.stage3.sha256, j);
                try mma_capture_screen.run(gpu, suite.mma.fixture, suite.mma.fixture_sha256, suite.mma.image, suite.mma.image_sha256, j);
            }
            if (input.slice_major_screen) |screen| try slice_major_screen.run(gpu, screen.fixture, screen.fixture_sha256, screen.image, screen.image_sha256, j);
            if (input.boundary_screen) |screen| switch (screen) {
                .terminal_greedy => |f| try terminal_greedy_screen.run(gpu, f.fixture, f.fixture_sha256, j),
                .shared_activation => |f| try shared_activation_screen.run(gpu, f.fixture, f.fixture_sha256, f.image, f.image_sha256, j),
            };
        };
        try j.write(.{ .kind = "complete", .correctness_pass = true, .scope = "isolated operator parity and timing screens", .production_qualification = false });
        return;
    }
    const c = try qwen.Engine.open(init.gpa, init.io, dir, .{ .context = input.context, .cut = input.cut, .captured = !input.gdn_layout_ab and !input.expert_n8_ab and !input.verify_default_policy and !input.prefill_ab and !input.combined_ab, .prefill_rows = if (input.prefill_ab or input.combined_ab) 16 else 1 });
    defer c.deinit();
    if (input.prefill_ab or input.combined_ab) {
        try j.write(.{ .kind = "start", .source_commit = options.source_commit, .implementation = "shared-qwen-engine", .identities = identities, .scope = if (input.combined_ab) "shared Engine retained-candidate bundle parity and matched C=1 turns" else "shared Engine bounded prefill parity and matched C=1 turns", .prefill_rows = 16, .combined_ab = input.combined_ab, .prompt_tokens = input.prompt.len, .output_tokens = input.output_tokens, .context = input.context, .cut = input.cut });
        return prefillAB(init, c, input, output, j);
    }
    if (input.gdn_layout_ab or input.expert_n8_ab) {
        try j.write(.{ .kind = "start", .source_commit = options.source_commit, .implementation = "shared-qwen-engine", .identities = identities, .scope = "changing-token changing-position eager and captured shared Qwen forwards", .prompt_tokens = input.prompt.len, .output_tokens = input.output_tokens, .context = input.context, .cut = input.cut });
        return trialAB(init, c, input, output, j, if (input.expert_n8_ab) .expert_n8 else .gdn_layout);
    }
    if (input.verify_default_policy) for (c.partitions) |*p| {
        if (!p.engine.expert_up_n8) return error.DefaultExpertN8PolicyMismatch;
        p.engine.verify_expert_graphs = true;
        p.engine.verify_gdn_graphs = true;
    };
    var s = try c.createSession();
    defer s.deinit();
    if (input.verify_default_policy) for (s.caches) |cache| {
        if (cache.gdn_layout != .key_value) return error.DefaultGdnLayoutPolicyMismatch;
    };
    try j.write(.{ .kind = "start", .source_commit = options.source_commit, .implementation = "shared-qwen-engine", .identities = identities, .scope = "changing-token changing-position eager and captured shared Qwen forwards", .prompt_tokens = input.prompt.len, .output_tokens = input.output_tokens, .context = input.context, .cut = input.cut, .default_policies = if (input.verify_default_policy) .{ .enabled = true, .expert_up_n8 = c.partitions[0].engine.expert_up_n8, .gdn_layout = @tagName(s.caches[0].gdn_layout), .verify_expert_graphs = c.partitions[0].engine.verify_expert_graphs, .verify_gdn_graphs = c.partitions[0].engine.verify_gdn_graphs } else null });
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
        try j.write(.{ .kind = "turn", .pair = pair, .order = order, .warmup = pair == 0, .arm = if (graph) "graph" else "eager", .graph = graph, .times = timing, .elapsed_ns = timing.elapsed_ns, .steps = timing.appends, .graph_launches = launches, .graph_nodes = nodes, .position = s.position, .exact_rows_tokens_and_final_state = true, .consumed_logits_sha256 = &expected_hash, .persistent_state_sha256 = &state, .tokens = tokens, .default_policies = if (input.verify_default_policy) .{ .expert_up_n8 = c.partitions[0].engine.expert_up_n8, .gdn_layout = @tagName(s.caches[0].gdn_layout), .verify_expert_graphs = c.partitions[0].engine.verify_expert_graphs, .verify_gdn_graphs = c.partitions[0].engine.verify_gdn_graphs, .graph_verifiers = .{ .{ .rank = 0, .expert_baseline_nodes = c.partitions[0].engine.expert_graph_nodes_baseline, .expert_n8_nodes = c.partitions[0].engine.expert_graph_nodes_n8, .gdn_baseline_nodes = c.partitions[0].engine.gdn_graph_nodes_baseline, .gdn_candidate_nodes = c.partitions[0].engine.gdn_graph_nodes_candidate }, .{ .rank = 1, .expert_baseline_nodes = c.partitions[1].engine.expert_graph_nodes_baseline, .expert_n8_nodes = c.partitions[1].engine.expert_graph_nodes_n8, .gdn_baseline_nodes = c.partitions[1].engine.gdn_graph_nodes_baseline, .gdn_candidate_nodes = c.partitions[1].engine.gdn_graph_nodes_candidate } } } else null });
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
    try j.write(.{ .kind = "complete", .correctness_pass = true, .scope = "shared engine eager/captured parity and session controls", .production_qualification = false, .independent_reference_parity = "not checked by this executable", .default_policies = if (input.verify_default_policy) .{ .expert_up_n8 = c.partitions[0].engine.expert_up_n8, .gdn_layout = @tagName(s.caches[0].gdn_layout), .verify_expert_graphs = c.partitions[0].engine.verify_expert_graphs, .verify_gdn_graphs = c.partitions[0].engine.verify_gdn_graphs, .graph_verifiers = .{ .{ .rank = 0, .expert_baseline_nodes = c.partitions[0].engine.expert_graph_nodes_baseline, .expert_n8_nodes = c.partitions[0].engine.expert_graph_nodes_n8, .gdn_baseline_nodes = c.partitions[0].engine.gdn_graph_nodes_baseline, .gdn_candidate_nodes = c.partitions[0].engine.gdn_graph_nodes_candidate }, .{ .rank = 1, .expert_baseline_nodes = c.partitions[1].engine.expert_graph_nodes_baseline, .expert_n8_nodes = c.partitions[1].engine.expert_graph_nodes_n8, .gdn_baseline_nodes = c.partitions[1].engine.gdn_graph_nodes_baseline, .gdn_candidate_nodes = c.partitions[1].engine.gdn_graph_nodes_candidate } } } else null });
}
