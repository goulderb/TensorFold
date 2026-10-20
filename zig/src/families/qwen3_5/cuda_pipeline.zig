const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const checkpoint = @import("cuda_checkpoint.zig");
const weights = @import("cuda_weights.zig");
const engine = @import("cuda_engine.zig");
const reuse = @import("cuda_reuse.zig");

pub const floor_bytes = 1536 << 20;
pub const MoeMath = engine.MoeMath;
pub const AttentionMode = engine.AttentionMode;
pub const Options = struct {
    devices: []const c_int = &.{ 0, 1 },
    context: usize = 4096,
    cut: usize = 20,
    captured: bool = true,
    prefill_rows: usize = 1,
    expert_up_layout: cuda.experts.Nvfp4UpLayout = .lane_major,
    terminal_greedy: bool = false,
    moe_math: engine.MoeMath = .legacy,
    attention_mode: engine.AttentionMode = .chunked,
    kv8: bool = false,
};

fn validatePrefillRows(rows: usize) !void {
    if (rows == 0 or rows > engine.hf_prefill_rows) return error.InvalidPrefillRows;
}

fn logitElements(rows: usize, vocab: usize) !usize {
    return std.math.mul(usize, rows, vocab);
}

fn seamStride(hidden: usize) usize {
    return engine.hf_prefill_rows * hidden * 2;
}
pub const MemoryBudget = struct {
    weights_bytes: usize,
    staging_bytes: usize,
    scratch_bytes: usize,
    stream_bytes: usize,
    streams: usize,
    state_bytes: usize,
    steady_bytes: usize,
    required_bytes: usize,
};

pub fn memoryBudget(plan: weights.Plan, admitted: checkpoint.Admitted, placement: checkpoint.Placement, context: usize, streams: usize) !MemoryBudget {
    if (context == 0 or context > admitted.geometry.context or streams == 0) return error.InvalidContext;
    const g = try placement.localGeometry(admitted.geometry);
    const scratch = try engine.Engine.scratchBytes(admitted, placement);
    const per_stream = g.stateBytes() + g.kvBytes(context, false) + engine.aligned(g.logBytes(if (admitted.numerics() == .hf_modelopt) 1 else engine.most_window));
    const states = try std.math.mul(usize, per_stream, streams);
    const fixed = try std.math.add(usize, plan.weights_bytes, scratch);
    const steady = try std.math.add(usize, fixed, states);
    const peak = try std.math.add(usize, fixed, @max(plan.staging_bytes, states));
    return .{ .weights_bytes = plan.weights_bytes, .staging_bytes = plan.staging_bytes, .scratch_bytes = scratch, .stream_bytes = per_stream, .streams = streams, .state_bytes = states, .steady_bytes = steady, .required_bytes = try std.math.add(usize, peak, floor_bytes) };
}

pub fn checkCapacity(available: usize, required: usize) !void {
    if (available < required) return error.InsufficientGpuMemory;
}

pub const Stage = struct { context: cuda.Context, engine: *engine.Engine };

pub const Chain = struct {
    gpa: std.mem.Allocator,
    driver: cuda.Driver,
    config: checkpoint.Config,
    context: usize,
    prefill_rows: usize = 1,
    partitions: []Stage,

    pub fn open(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, options: Options) !*Chain {
        try validatePrefillRows(options.prefill_rows);
        const config = try checkpoint.Config.read(gpa, io, dir);
        if (options.context == 0 or options.context > config.context) return error.InvalidContext;
        if (options.devices.len != 1 and options.devices.len != 2) return error.InvalidDevices;
        for (options.devices) |device| if (device < 0) return error.InvalidDevices;
        if (options.devices.len == 2 and (options.devices[0] == options.devices[1] or options.cut == 0 or options.cut >= config.geometry.layers)) return error.InvalidPartition;
        const admitted: checkpoint.Admitted = .{ .geometry = config.geometry, .encoding = .{ .modelopt = config } };
        try engine.validateMoeMath(admitted, options.moe_math);
        try engine.validateAttentionMode(admitted, options.attention_mode, options.kv8);
        var placements: [2]checkpoint.Placement = undefined;
        var budgets: [2]MemoryBudget = undefined;
        var ck = try core.Checkpoint.openModel(gpa, io, dir);
        defer ck.close();
        for (options.devices, 0..) |_, i| {
            const p: checkpoint.Placement = .{ .first = if (i == 0) 0 else options.cut, .end = if (options.devices.len == 1 or i == 1) config.geometry.layers else options.cut, .total = config.geometry.layers };
            placements[i] = p;
            budgets[i] = try memoryBudget(try weights.plan(&ck, admitted, p), admitted, p, options.context, 1);
        }
        const c = try gpa.create(Chain);
        errdefer gpa.destroy(c);
        c.* = .{ .gpa = gpa, .driver = try cuda.Driver.open(), .config = config, .context = options.context, .prefill_rows = options.prefill_rows, .partitions = undefined };
        errdefer c.driver.close();
        c.partitions = try gpa.alloc(Stage, options.devices.len);
        errdefer gpa.free(c.partitions);
        var contexts: usize = 0;
        errdefer for (c.partitions[0..contexts]) |*p| p.context.deinit();
        for (options.devices, c.partitions, budgets[0..options.devices.len]) |device, *p, budget| {
            p.context = try cuda.Context.init(&c.driver, device);
            contexts += 1;
            try checkCapacity((try p.context.memInfo()).free, budget.required_bytes);
        }
        var engines: usize = 0;
        errdefer for (c.partitions[0..engines]) |*p| {
            p.context.makeCurrent() catch {};
            p.engine.deinit();
        };
        for (c.partitions, placements[0..c.partitions.len]) |*p, placement| {
            try p.context.makeCurrent();
            p.engine = try engine.Engine.initPlacedWithOptions(gpa, io, &p.context, dir, admitted, placement, options.kv8, .{ .expert_up_layout = options.expert_up_layout, .terminal_greedy = options.terminal_greedy, .moe_math = options.moe_math, .attention_mode = options.attention_mode });
            engines += 1;
            p.engine.captured = options.captured;
            p.engine.bounded_hf_prefill = options.prefill_rows > 1;
            try checkCapacity((try p.context.memInfo()).free, floor_bytes);
        }
        var warm = try c.createSession();
        defer warm.deinit();
        try warm.append(&.{0});
        return c;
    }

    pub fn deinit(c: *Chain) void {
        c.partitions[c.partitions.len - 1].context.makeCurrent() catch {};
        for (c.partitions) |*p| {
            p.context.makeCurrent() catch {};
            p.engine.deinit();
            p.context.deinit();
        }
        c.gpa.free(c.partitions);
        c.driver.close();
        c.gpa.destroy(c);
    }

    pub fn keptBytes(c: *const Chain, at: u32) u64 {
        var n: u64 = 0;
        for (c.partitions) |p| n += reuse.hostBytes(p.engine, at);
        return n;
    }

    pub fn createSession(c: *Chain) !Session {
        return c.createSessionWithLayout(.key_value);
    }

    pub fn createSessionWithLayout(c: *Chain, layout: engine.Cache.GdnLayout) !Session {
        for (c.partitions) |p| try layout.validate(p.engine.numerics);
        var s: Session = .{ .chain = c, .caches = undefined, .logits = try c.gpa.alloc(u16, c.config.geometry.vocab) };
        errdefer c.gpa.free(s.logits);
        var made: usize = 0;
        errdefer for (c.partitions[0..made], s.caches[0..made]) |*p, *cache| {
            p.context.makeCurrent() catch {};
            cache.deinit(c.gpa);
        };
        for (c.partitions, 0..) |*p, i| {
            try p.context.makeCurrent();
            try checkCapacity((try p.context.memInfo()).free, p.engine.streamBytes(c.context) + floor_bytes);
            s.caches[i] = try engine.Cache.initWithLayout(c.gpa, p.engine, c.context, layout);
            made += 1;
            try checkCapacity((try p.context.memInfo()).free, floor_bytes);
        }
        if (c.partitions.len > 1) {
            s.seam = try cuda.HostBuffer.alloc(&c.driver, 2 * seamStride(c.config.geometry.hidden));
        }
        @memset(s.logits, 0);
        return s;
    }
};

pub const Session = struct {
    chain: *Chain,
    caches: [2]engine.Cache,
    position: usize = 0,
    logits: []u16,
    poisoned: bool = false,
    seam: ?cuda.HostBuffer = null,
    held_ids: [engine.hf_prefill_rows]u32 = undefined,
    held_n: usize = 0,
    tail_shape: ?engine.Engine.Shape = null,
    tail_keep: engine.Keep = .last,
    tail_open: bool = false,

    pub fn deinit(s: *Session) void {
        for (s.chain.partitions, 0..) |*p, i| {
            p.context.makeCurrent() catch {};
            s.caches[i].deinit(s.chain.gpa);
        }
        if (s.seam) |*seam| seam.free();
        s.chain.gpa.free(s.logits);
    }

    pub fn reset(s: *Session) !void {
        s.poisoned = true;
        s.invalidateGreedy();
        for (s.chain.partitions, 0..) |*p, i| {
            try p.context.makeCurrent();
            try s.caches[i].reset(p.engine);
            try p.engine.stream.synchronize();
        }
        s.position = 0;
        @memset(s.logits, 0);
        s.poisoned = false;
    }

    pub fn append(s: *Session, tokens: []const u32) !void {
        try s.validateAppend(tokens);
        var at: usize = 0;
        while (at < tokens.len) {
            const end = at + @min(s.chain.prefill_rows, tokens.len - at);
            try s.stepRows(tokens[at..end], .last);
            at = end;
        }
    }

    pub fn appendLogitRows(s: *Session, tokens: []const u32, out: []u16) !void {
        if (s.poisoned) return error.PoisonedSession;
        const expected = try logitElements(tokens.len, s.logits.len);
        if (out.len != expected) return error.InvalidLogits;
        try s.validateAppend(tokens);
        var at: usize = 0;
        while (at < tokens.len) {
            const end = at + @min(s.chain.prefill_rows, tokens.len - at);
            const keep: engine.Keep = if (s.chain.prefill_rows > 1) .prompt_logits else .last;
            try s.stepRows(tokens[at..end], keep);
            const e = s.chain.partitions[s.chain.partitions.len - 1].engine;
            for (0..end - at) |row| @memcpy(out[(at + row) * s.logits.len ..][0..s.logits.len], e.rowLogits(row));
            at = end;
        }
    }

    fn validateAppend(s: *const Session, tokens: []const u32) !void {
        if (s.poisoned) return error.PoisonedSession;
        if (s.position > s.chain.context or tokens.len > s.chain.context - s.position) return error.ContextFull;
        for (tokens) |token| if (token >= s.logits.len) return error.InvalidToken;
    }

    fn stepRows(s: *Session, tokens: []const u32, keep: engine.Keep) !void {
        try s.runHead(tokens, keep);
        if (s.chain.partitions.len > 1) {
            try s.launchTail();
            try s.waitTail();
        }
    }

    /// GPU0 (or the only partition) for `tokens`. Two-GPU sessions leave GPU1 to launchTail.
    pub fn runHead(s: *Session, tokens: []const u32, keep: engine.Keep) !void {
        s.invalidateGreedy();
        errdefer {
            s.invalidateGreedy();
            s.poisoned = true;
        }
        if (s.tail_open) return error.TailPending;
        if (tokens.len == 0 or tokens.len > engine.hf_prefill_rows) return error.InvalidPrefillRows;
        try s.runPartition(0, tokens, keep, if (s.chain.partitions.len > 1) .download else .none, true);
        if (s.chain.partitions.len == 1) {
            s.commitRows(tokens.len);
            return;
        }
        @memcpy(s.held_ids[0..tokens.len], tokens);
        s.held_n = tokens.len;
        s.tail_keep = keep;
    }

    /// Start GPU1 from the residual left by runHead. Does not wait for logits.
    pub fn launchTail(s: *Session) !void {
        errdefer {
            s.invalidateGreedy();
            s.poisoned = true;
        }
        if (s.chain.partitions.len < 2 or s.held_n == 0 or s.tail_open) return error.TailPending;
        try s.runPartition(1, s.held_ids[0..s.held_n], s.tail_keep, .upload, false);
        s.tail_open = true;
    }

    /// Finish GPU1, commit both caches, and copy logits.
    pub fn waitTail(s: *Session) !void {
        errdefer {
            s.invalidateGreedy();
            s.poisoned = true;
            s.tail_open = false;
            s.tail_shape = null;
        }
        if (!s.tail_open) return error.TailPending;
        const p = &s.chain.partitions[1];
        try p.context.makeCurrent();
        const parts = [_]engine.Part{.{ .cache = &s.caches[1], .rows = s.held_n }};
        try p.engine.finish(&parts, s.tail_keep, s.tail_shape orelse return error.TailPending);
        try checkCapacity((try p.context.memInfo()).free, floor_bytes);
        s.tail_open = false;
        s.tail_shape = null;
        s.commitRows(s.held_n);
        s.held_n = 0;
    }

    const Copy = enum { none, download, upload };

    fn runPartition(s: *Session, i: usize, tokens: []const u32, keep: engine.Keep, copy: Copy, wait: bool) !void {
        const c = s.chain;
        const p = &c.partitions[i];
        try p.context.makeCurrent();
        const e = p.engine;
        const parts = [_]engine.Part{.{ .cache = &s.caches[i], .rows = tokens.len }};
        const shape = try e.prepare(&parts, tokens, keep);
        const bytes = tokens.len * c.config.geometry.hidden * 2;
        const residual = seamStride(c.config.geometry.hidden);
        if (copy != .none) {
            const seam = &(s.seam orelse return error.InvalidPartition);
            if (copy == .upload) {
                try c.driver.check(c.driver.api.cuMemcpyHtoDAsync_v2(e.h, seam.bytes.ptr, bytes, e.stream.handle), "pipeline h upload");
                try c.driver.check(c.driver.api.cuMemcpyHtoDAsync_v2(e.r, seam.bytes.ptr + residual, bytes, e.stream.handle), "pipeline r upload");
            }
            try e.execute(shape);
            if (copy == .download) {
                try c.driver.check(c.driver.api.cuMemcpyDtoHAsync_v2(seam.bytes.ptr, e.h, bytes, e.stream.handle), "pipeline h download");
                try c.driver.check(c.driver.api.cuMemcpyDtoHAsync_v2(seam.bytes.ptr + residual, e.r, bytes, e.stream.handle), "pipeline r download");
            }
        } else try e.execute(shape);
        if (wait) {
            try e.finish(&parts, keep, shape);
            try checkCapacity((try p.context.memInfo()).free, floor_bytes);
        } else s.tail_shape = shape;
    }

    fn commitRows(s: *Session, n: usize) void {
        const c = s.chain;
        for (s.caches[0..c.partitions.len]) |*cache| {
            const parts = [_]engine.Part{.{ .cache = cache, .rows = n }};
            engine.Engine.commit(&parts, false);
        }
        @memcpy(s.logits, s.caches[c.partitions.len - 1].logits);
        s.position += n;
    }

    /// Overlap GPU1 of `a` with GPU0 of `b` on a two-partition chain.
    pub fn pipelineRows(a: *Session, a_tokens: []const u32, b: *Session, b_tokens: []const u32, keep: engine.Keep) !void {
        var sessions = [_]*Session{ a, b };
        const tokens = [_][]const u32{ a_tokens, b_tokens };
        return pipelineMany(&sessions, &tokens, keep);
    }

    /// GPU0 of each later session overlaps GPU1 of the previous one.
    pub fn pipelineMany(sessions: []const *Session, tokens: []const []const u32, keep: engine.Keep) !void {
        if (sessions.len == 0 or sessions.len != tokens.len) return error.InvalidPartition;
        const chain = sessions[0].chain;
        if (chain.partitions.len < 2) {
            for (sessions, tokens) |s, t| try s.stepRows(t, keep);
            return;
        }
        for (sessions) |s| if (s.chain != chain) return error.InvalidPartition;
        try sessions[0].runHead(tokens[0], keep);
        var i: usize = 1;
        while (i < sessions.len) : (i += 1) {
            try sessions[i - 1].launchTail();
            try sessions[i].runHead(tokens[i], keep);
            try sessions[i - 1].waitTail();
        }
        try sessions[sessions.len - 1].launchTail();
        try sessions[sessions.len - 1].waitTail();
    }

    pub fn copyLogits(s: *const Session, out: []f32) !void {
        if (s.poisoned) return error.PoisonedSession;
        if (out.len != s.logits.len) return error.InvalidLogits;
        for (out, s.logits) |*value, bits| value.* = @bitCast(@as(u32, bits) << 16);
    }

    fn invalidateGreedy(s: *Session) void {
        for (s.caches[0..s.chain.partitions.len]) |*cache| cache.terminal_greedy = null;
    }

    pub const Kept = struct {
        parts: [2]?*reuse.Copy = .{ null, null },
        n: usize = 0,
        at: usize = 0,
    };

    pub fn saveKept(s: *Session, at: u32) !*anyopaque {
        if (s.poisoned or s.position != at) return error.NotAtMark;
        const kept = try s.chain.gpa.create(Kept);
        errdefer s.chain.gpa.destroy(kept);
        kept.* = .{ .n = s.chain.partitions.len, .at = at };
        var made: usize = 0;
        errdefer {
            for (kept.parts[0..made], s.chain.partitions[0..made]) |copy, *p| {
                p.context.makeCurrent() catch {};
                if (copy) |c| reuse.dropCopy(p.engine, c);
            }
        }
        for (s.chain.partitions, 0..) |*p, i| {
            try p.context.makeCurrent();
            kept.parts[i] = try reuse.saveCache(p.engine, &s.caches[i], at, &.{});
            made += 1;
        }
        return kept;
    }

    pub fn restoreKept(s: *Session, saved: *anyopaque) !void {
        const kept: *Kept = @ptrCast(@alignCast(saved));
        if (kept.n != s.chain.partitions.len or kept.at == 0) return error.BadReuse;
        for (s.chain.partitions, 0..) |*p, i| {
            try p.context.makeCurrent();
            try reuse.restoreCache(p.engine, &s.caches[i], @ptrCast(kept.parts[i] orelse return error.BadReuse), &.{});
        }
        s.position = kept.at;
        @memcpy(s.logits, s.caches[s.chain.partitions.len - 1].logits);
    }

    pub fn dropKept(s: *Session, saved: *anyopaque) void {
        const kept: *Kept = @ptrCast(@alignCast(saved));
        for (s.chain.partitions[0..kept.n], 0..) |*p, i| {
            p.context.makeCurrent() catch {};
            if (kept.parts[i]) |copy| reuse.dropCopy(p.engine, copy);
        }
        s.chain.gpa.destroy(kept);
    }

    pub fn cachedGreedy(s: *const Session, position: u64) !?u32 {
        if (s.poisoned) return error.PoisonedSession;
        if (position != s.position) return error.PositionMismatch;
        if (position == 0) return null;
        const cache = &s.caches[s.chain.partitions.len - 1];
        if (cache.len != position) return null;
        const pick = cache.terminal_greedy orelse return null;
        return if (pick.position == position) pick.token else null;
    }
};

test "prefill bounds reject before checkpoint or CUDA access" {
    try std.testing.expectError(error.InvalidPrefillRows, Chain.open(std.testing.allocator, std.testing.io, "not-a-checkpoint", .{ .prefill_rows = 0 }));
    try std.testing.expectError(error.InvalidPrefillRows, Chain.open(std.testing.allocator, std.testing.io, "not-a-checkpoint", .{ .prefill_rows = 17 }));
    for (1..17) |rows| try validatePrefillRows(rows);
    const defaults: Options = .{};
    try std.testing.expectEqual(@as(usize, 1), defaults.prefill_rows);
    try std.testing.expect(!defaults.terminal_greedy);
    try std.testing.expectEqual(AttentionMode.chunked, defaults.attention_mode);
}

test "prefill seam keeps residual payload disjoint for every admitted chunk" {
    const stride = seamStride(2048);
    try std.testing.expectEqual(@as(usize, 65536), stride);
    for (1..17) |rows| {
        const bytes = rows * 2048 * 2;
        try std.testing.expect(bytes <= stride);
        try std.testing.expect(stride + bytes <= 2 * stride);
    }
}

test "diagnostic validates output shape and every token without mutation" {
    var chain: Chain = undefined;
    chain.context = 4;
    var logits: [8]u16 = @splat(0x3f80);
    var out: [16]u16 = @splat(0xa5a5);
    var full: [32]u16 = @splat(0xa5a5);
    var s: Session = .{ .chain = &chain, .caches = undefined, .position = 1, .logits = &logits };
    try std.testing.expectError(error.InvalidLogits, s.appendLogitRows(&.{ 1, 2 }, out[0..15]));
    try std.testing.expectError(error.InvalidToken, s.appendLogitRows(&.{ 1, 8 }, &out));
    try std.testing.expectError(error.ContextFull, s.appendLogitRows(&.{ 1, 2, 3, 4 }, &full));
    try std.testing.expectEqual(@as(usize, 1), s.position);
    try std.testing.expect(!s.poisoned);
    for (out) |value| try std.testing.expectEqual(@as(u16, 0xa5a5), value);
    for (logits) |value| try std.testing.expectEqual(@as(u16, 0x3f80), value);
    try s.appendLogitRows(&.{}, &.{});
    s.poisoned = true;
    try std.testing.expectError(error.PoisonedSession, s.appendLogitRows(&.{}, &.{}));
    try std.testing.expectError(error.Overflow, logitElements(std.math.maxInt(usize), 2));
}

test "a stage bind failure poisons the chunk without committing host state" {
    const Fake = struct {
        var pending: *engine.Cache = undefined;
        var inject: bool = false;
        fn bind(_: cuda.abi.Context) callconv(.c) cuda.abi.Result {
            if (inject) pending.terminal_greedy = .{ .position = 5, .token = 1 };
            return cuda.abi.error_not_ready;
        }
    };
    var d: cuda.Driver = undefined;
    d.api.cuCtxSetCurrent = Fake.bind;
    var stages: [2]Stage = undefined;
    for (&stages) |*p| p.context = .{ .d = &d, .device = 0, .handle = null };
    var chain: Chain = undefined;
    chain.context = 64;
    chain.prefill_rows = 2;
    chain.config.geometry.hidden = 2048;
    chain.partitions = &stages;
    var logits: [8]u16 = @splat(0x3f80);
    var out: [16]u16 = @splat(0xa5a5);
    var s: Session = .{ .chain = &chain, .caches = undefined, .position = 3, .logits = &logits };
    for (&s.caches) |*cache| {
        cache.len = 3;
        cache.written = 3;
        cache.last = null;
        cache.terminal_greedy = .{ .position = 3, .token = 1 };
    }
    Fake.pending = &s.caches[1];
    Fake.inject = true;
    try std.testing.expectError(error.NotReady, s.appendLogitRows(&.{ 1, 2 }, &out));
    try std.testing.expect(s.poisoned);
    try std.testing.expectEqual(@as(usize, 3), s.position);
    for (s.caches) |cache| {
        try std.testing.expectEqual(@as(usize, 3), cache.len);
        try std.testing.expectEqual(@as(usize, 3), cache.written);
        try std.testing.expect(cache.last == null);
        try std.testing.expect(cache.terminal_greedy == null);
    }
    for (logits) |value| try std.testing.expectEqual(@as(u16, 0x3f80), value);
    for (out) |value| try std.testing.expectEqual(@as(u16, 0xa5a5), value);
    try std.testing.expectError(error.PoisonedSession, s.append(&.{1}));
    try std.testing.expectError(error.PoisonedSession, s.cachedGreedy(3));
    for (&s.caches) |*cache| cache.terminal_greedy = .{ .position = 3, .token = 1 };
    Fake.inject = false;
    try std.testing.expectError(error.NotReady, s.reset());
    try std.testing.expect(s.poisoned);
    for (s.caches) |cache| try std.testing.expect(cache.terminal_greedy == null);
}

test "terminal picks require committed session positions and stay isolated between sessions" {
    var stages: [2]Stage = undefined;
    var chain: Chain = undefined;
    chain.partitions = &stages;
    chain.context = 64;
    var logits_a = [_]u16{ 0x3f80, 0x4000 };
    var logits_b = [_]u16{ 0x4000, 0x3f80 };
    var a: Session = .{ .chain = &chain, .caches = undefined, .logits = &logits_a, .position = 7 };
    var b: Session = .{ .chain = &chain, .caches = undefined, .logits = &logits_b, .position = 7 };
    for ([_]*Session{ &a, &b }, 0..) |s, i| {
        for (&s.caches) |*cache| {
            cache.len = 7;
            cache.terminal_greedy = null;
        }
        s.caches[1].terminal_greedy = .{ .position = 8, .token = @intCast(1 - i) };
        try std.testing.expect(try s.cachedGreedy(7) == null);
        for (&s.caches) |*cache| {
            const parts = [_]engine.Part{.{ .cache = cache, .rows = 1 }};
            engine.Engine.commit(&parts, false);
        }
        try std.testing.expect(try s.cachedGreedy(7) == null);
        s.position = 8;
    }
    try std.testing.expectEqual(@as(?u32, 1), try a.cachedGreedy(8));
    try std.testing.expectEqual(@as(?u32, 0), try b.cachedGreedy(8));
    try a.append(&.{});
    try std.testing.expectEqual(@as(?u32, 1), try a.cachedGreedy(8));
    try std.testing.expectError(error.PositionMismatch, a.cachedGreedy(7));
    try std.testing.expectError(error.InvalidToken, a.append(&.{2}));
    try std.testing.expectEqual(@as(?u32, 1), try a.cachedGreedy(8));
    b.invalidateGreedy();
    try std.testing.expect(try b.cachedGreedy(8) == null);
    try std.testing.expectEqual(@as(?u32, 1), try a.cachedGreedy(8));
    var out: [2]f32 = undefined;
    try a.copyLogits(&out);
    try std.testing.expectEqualSlices(f32, &.{ 1, 2 }, &out);
    a.poisoned = true;
    try std.testing.expectError(error.PoisonedSession, a.cachedGreedy(8));
}

test "partial and recorded commits discard terminal picks" {
    var cache: engine.Cache = undefined;
    cache.len = 7;
    cache.terminal_greedy = .{ .position = 23, .token = 1 };
    const parts = [_]engine.Part{.{ .cache = &cache, .rows = 1 }};
    engine.Engine.commit(&parts, false);
    try std.testing.expect(cache.terminal_greedy == null);
    cache.terminal_greedy = .{ .position = 9, .token = 1 };
    engine.Engine.commit(&parts, true);
    try std.testing.expect(cache.terminal_greedy == null);
}
