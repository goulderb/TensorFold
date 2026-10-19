const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const checkpoint = @import("cuda_checkpoint.zig");
const weights = @import("cuda_weights.zig");
const engine = @import("cuda_engine.zig");

pub const floor_bytes = 1536 << 20;
pub const Options = struct {
    devices: []const c_int = &.{ 0, 1 },
    context: usize = 4096,
    cut: usize = 20,
    captured: bool = true,
    prefill_rows: usize = 1,
    expert_up_layout: cuda.experts.Nvfp4UpLayout = .lane_major,
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
    seam: cuda.HostBuffer,

    pub fn open(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, options: Options) !*Chain {
        try validatePrefillRows(options.prefill_rows);
        const config = try checkpoint.Config.read(gpa, io, dir);
        if (options.context == 0 or options.context > config.context) return error.InvalidContext;
        if (options.devices.len != 1 and options.devices.len != 2) return error.InvalidDevices;
        for (options.devices) |device| if (device < 0) return error.InvalidDevices;
        if (options.devices.len == 2 and (options.devices[0] == options.devices[1] or options.cut == 0 or options.cut >= config.geometry.layers)) return error.InvalidPartition;
        const admitted: checkpoint.Admitted = .{ .geometry = config.geometry, .encoding = .{ .modelopt = config } };
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
        c.* = .{ .gpa = gpa, .driver = try cuda.Driver.open(), .config = config, .context = options.context, .prefill_rows = options.prefill_rows, .partitions = undefined, .seam = undefined };
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
            p.engine = try engine.Engine.initPlacedWithOptions(gpa, io, &p.context, dir, admitted, placement, false, .{ .expert_up_layout = options.expert_up_layout });
            engines += 1;
            p.engine.captured = options.captured;
            p.engine.bounded_hf_prefill = options.prefill_rows > 1;
            try checkCapacity((try p.context.memInfo()).free, floor_bytes);
        }
        c.seam = try cuda.HostBuffer.alloc(&c.driver, 2 * seamStride(config.geometry.hidden));
        errdefer c.seam.free();
        var warm = try c.createSession();
        defer warm.deinit();
        try warm.append(&.{0});
        return c;
    }

    pub fn deinit(c: *Chain) void {
        c.partitions[c.partitions.len - 1].context.makeCurrent() catch {};
        c.seam.free();
        for (c.partitions) |*p| {
            p.context.makeCurrent() catch {};
            p.engine.deinit();
            p.context.deinit();
        }
        c.gpa.free(c.partitions);
        c.driver.close();
        c.gpa.destroy(c);
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

    pub fn deinit(s: *Session) void {
        for (s.chain.partitions, 0..) |*p, i| {
            p.context.makeCurrent() catch {};
            s.caches[i].deinit(s.chain.gpa);
        }
        s.chain.gpa.free(s.logits);
    }

    pub fn reset(s: *Session) !void {
        s.poisoned = true;
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
        errdefer s.poisoned = true;
        const c = s.chain;
        const bytes = tokens.len * c.config.geometry.hidden * 2;
        // The residual payload starts at the fixed maximum-row stride for every chunk.
        const residual = seamStride(c.config.geometry.hidden);
        for (c.partitions, 0..) |*p, i| {
            try p.context.makeCurrent();
            const e = p.engine;
            const parts = [_]engine.Part{.{ .cache = &s.caches[i], .rows = tokens.len }};
            const shape = try e.prepare(&parts, tokens, keep);
            if (i > 0) {
                try c.driver.check(c.driver.api.cuMemcpyHtoDAsync_v2(e.h, c.seam.bytes.ptr, bytes, e.stream.handle), "pipeline h upload");
                try c.driver.check(c.driver.api.cuMemcpyHtoDAsync_v2(e.r, c.seam.bytes.ptr + residual, bytes, e.stream.handle), "pipeline r upload");
            }
            try e.execute(shape);
            if (i + 1 < c.partitions.len) {
                try c.driver.check(c.driver.api.cuMemcpyDtoHAsync_v2(c.seam.bytes.ptr, e.h, bytes, e.stream.handle), "pipeline h download");
                try c.driver.check(c.driver.api.cuMemcpyDtoHAsync_v2(c.seam.bytes.ptr + residual, e.r, bytes, e.stream.handle), "pipeline r download");
            }
            try e.finish(&parts, keep, shape);
            try checkCapacity((try p.context.memInfo()).free, floor_bytes);
        }
        for (s.caches[0..c.partitions.len]) |*cache| {
            const parts = [_]engine.Part{.{ .cache = cache, .rows = tokens.len }};
            engine.Engine.commit(&parts, false);
        }
        @memcpy(s.logits, s.caches[c.partitions.len - 1].logits);
        s.position += tokens.len;
    }

    pub fn copyLogits(s: *const Session, out: []f32) !void {
        if (s.poisoned) return error.PoisonedSession;
        if (out.len != s.logits.len) return error.InvalidLogits;
        for (out, s.logits) |*value, bits| value.* = @bitCast(@as(u32, bits) << 16);
    }
};

test "prefill bounds reject before checkpoint or CUDA access" {
    try std.testing.expectError(error.InvalidPrefillRows, Chain.open(std.testing.allocator, std.testing.io, "not-a-checkpoint", .{ .prefill_rows = 0 }));
    try std.testing.expectError(error.InvalidPrefillRows, Chain.open(std.testing.allocator, std.testing.io, "not-a-checkpoint", .{ .prefill_rows = 17 }));
    for (1..17) |rows| try validatePrefillRows(rows);
    const defaults: Options = .{};
    try std.testing.expectEqual(@as(usize, 1), defaults.prefill_rows);
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
        fn bind(_: cuda.abi.Context) callconv(.c) cuda.abi.Result {
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
    }
    try std.testing.expectError(error.NotReady, s.appendLogitRows(&.{ 1, 2 }, &out));
    try std.testing.expect(s.poisoned);
    try std.testing.expectEqual(@as(usize, 3), s.position);
    for (s.caches) |cache| {
        try std.testing.expectEqual(@as(usize, 3), cache.len);
        try std.testing.expectEqual(@as(usize, 3), cache.written);
        try std.testing.expect(cache.last == null);
    }
    for (logits) |value| try std.testing.expectEqual(@as(u16, 0x3f80), value);
    for (out) |value| try std.testing.expectEqual(@as(u16, 0xa5a5), value);
    try std.testing.expectError(error.PoisonedSession, s.append(&.{1}));
    try std.testing.expectError(error.NotReady, s.reset());
    try std.testing.expect(s.poisoned);
}
