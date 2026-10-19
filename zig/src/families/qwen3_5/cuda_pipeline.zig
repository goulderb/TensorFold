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
};
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
    partitions: []Stage,
    seam: cuda.HostBuffer,

    pub fn open(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, options: Options) !*Chain {
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
        c.* = .{ .gpa = gpa, .driver = try cuda.Driver.open(), .config = config, .context = options.context, .partitions = undefined, .seam = undefined };
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
            p.engine = try engine.Engine.initPlaced(gpa, io, &p.context, dir, admitted, placement, false);
            engines += 1;
            p.engine.captured = options.captured;
            try checkCapacity((try p.context.memInfo()).free, floor_bytes);
        }
        c.seam = try cuda.HostBuffer.alloc(&c.driver, config.geometry.hidden * 2 * 2);
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
            s.caches[i] = try engine.Cache.init(c.gpa, p.engine, c.context);
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
        if (s.poisoned) return error.PoisonedSession;
        if (tokens.len > s.chain.context - s.position) return error.ContextFull;
        for (tokens) |token| if (token >= s.logits.len) return error.InvalidToken;
        for (tokens) |token| try s.step(token);
    }

    fn step(s: *Session, token: u32) !void {
        errdefer s.poisoned = true;
        const c = s.chain;
        const bytes = c.config.geometry.hidden * 2;
        for (c.partitions, 0..) |*p, i| {
            try p.context.makeCurrent();
            const e = p.engine;
            const parts = [_]engine.Part{.{ .cache = &s.caches[i], .rows = 1 }};
            const shape = try e.prepare(&parts, &.{token}, .last);
            if (i > 0) {
                try c.driver.check(c.driver.api.cuMemcpyHtoDAsync_v2(e.h, c.seam.bytes.ptr, bytes, e.stream.handle), "pipeline h upload");
                try c.driver.check(c.driver.api.cuMemcpyHtoDAsync_v2(e.r, c.seam.bytes.ptr + bytes, bytes, e.stream.handle), "pipeline r upload");
            }
            try e.execute(shape);
            if (i + 1 < c.partitions.len) {
                try c.driver.check(c.driver.api.cuMemcpyDtoHAsync_v2(c.seam.bytes.ptr, e.h, bytes, e.stream.handle), "pipeline h download");
                try c.driver.check(c.driver.api.cuMemcpyDtoHAsync_v2(c.seam.bytes.ptr + bytes, e.r, bytes, e.stream.handle), "pipeline r download");
            }
            try e.finish(&parts, .last, shape);
            try checkCapacity((try p.context.memInfo()).free, floor_bytes);
        }
        for (s.caches[0..c.partitions.len]) |*cache| {
            cache.len += 1;
            cache.written = cache.len;
        }
        @memcpy(s.logits, s.caches[c.partitions.len - 1].logits);
        s.position += 1;
    }

    pub fn copyLogits(s: *const Session, out: []f32) !void {
        if (s.poisoned) return error.PoisonedSession;
        if (out.len != s.logits.len) return error.InvalidLogits;
        for (out, s.logits) |*value, bits| value.* = @bitCast(@as(u32, bits) << 16);
    }
};
