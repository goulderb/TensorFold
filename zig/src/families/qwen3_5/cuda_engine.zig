const std = @import("std");
const cuda = @import("cuda");
const checkpoint = @import("cuda_checkpoint.zig");
const weights = @import("cuda_weights.zig");
const Ops = @import("cuda_kernels.zig").Ops;
const Allocator = std.mem.Allocator;
pub const floor_bytes = 1536 << 20;

pub const Options = struct {
    devices: []const c_int = &.{ 0, 1 },
    context: usize = 4096,
    cut: usize = 20,
};

const Buffers = struct {
    residual: u64,
    x: u64,
    update: u64,
    qkv: u64,
    z: u64,
    a: u64,
    b: u64,
    q: u64,
    k: u64,
    v: u64,
    decay: u64,
    beta: u64,
    recurrent_out: u64,
    mixed: u64,
    full_k: u64,
    full_v: u64,
    picks: u64,
    route_weights: u64,
    members: u64,
    items: u64,
    counts: u64,
    expert_hidden: u64,
    pairs: u64,
    shared_gate: u64,
    shared_up: u64,
    shared_act: u64,
    shared_out: u64,
    ck_rows: u64,
    ck_part: u64,
    logits: u64,
};

const scratch_sizes = [_]usize{
    2048 * 2,    2048 * 2,     2048 * 2, 8192 * 2, 4096 * 2, 32 * 2,    32 * 2,
    2048 * 4,    2048 * 4,     4096 * 2, 32 * 4,   32 * 4,   4096 * 4,  8192 * 2,
    512 * 2,     512 * 2,      8 * 4,    9 * 4,    8 * 4,    8 * 3 * 4, 2 * 4,
    8 * 512 * 2, 8 * 2048 * 4, 512 * 2,  512 * 2,  512 * 2,  2048 * 2,  8192,
    8 << 20,     248320 * 2,
};

fn scratchBytes() usize {
    var size: usize = 0;
    for (scratch_sizes) |n| size += std.mem.alignForward(usize, n, 256);
    return size;
}

fn buffers(memory: cuda.DeviceBuffer) Buffers {
    var out: Buffers = undefined;
    var offset: usize = 0;
    inline for (@typeInfo(Buffers).@"struct".field_names, scratch_sizes) |name, size| {
        @field(out, name) = memory.ptr + offset;
        offset += std.mem.alignForward(usize, size, 256);
    }
    return out;
}

pub fn stateBytes(full: bool, context: usize) !usize {
    return if (full) std.math.mul(usize, context, 2048) else 32 * 128 * 128 * 4 + 3 * 8192 * 2;
}

fn checkFloor(ctx: *const cuda.Context) !void {
    if ((try ctx.memInfo()).free < floor_bytes) return error.GpuMemoryFloor;
}

pub const MemoryBudget = struct {
    weights_bytes: usize,
    staging_bytes: usize,
    scratch_bytes: usize,
    streams: usize,
    stream_bytes: usize,
    state_bytes: usize,
    steady_bytes: usize,
    required_bytes: usize,
};

pub fn memoryBudget(plan: weights.MemoryPlan, cfg: checkpoint.Config, first: usize, end: usize, context: usize, streams: usize) !MemoryBudget {
    if (streams == 0) return error.InvalidStreamCount;
    if (context == 0 or context > cfg.context) return error.InvalidContext;
    if (first > end or end > cfg.spec.n_layers) return error.InvalidLayerRange;
    var state: usize = 0;
    for (first..end) |i| state = try std.math.add(usize, state, try stateBytes(cfg.spec.full(i), context));
    const states = try std.math.mul(usize, state, streams);
    const live = try std.math.add(usize, scratchBytes(), states);
    const steady = try std.math.add(usize, plan.weights, live);
    const peak = try std.math.add(usize, plan.weights, @max(plan.staging, live));
    return .{
        .weights_bytes = plan.weights,
        .staging_bytes = plan.staging,
        .scratch_bytes = scratchBytes(),
        .streams = streams,
        .stream_bytes = state,
        .state_bytes = states,
        .steady_bytes = steady,
        .required_bytes = try std.math.add(usize, peak, floor_bytes),
    };
}

pub fn requiredMemory(plan: weights.MemoryPlan, cfg: checkpoint.Config, first: usize, end: usize, context: usize) !usize {
    return (try memoryBudget(plan, cfg, first, end, context, 1)).required_bytes;
}

pub fn checkCapacity(available: usize, required: usize) !void {
    if (available < required) return error.InsufficientGpuMemory;
}

fn releaseContext(ctx: *cuda.Context) void {
    _ = ctx.d.api.cuDevicePrimaryCtxRelease_v2(ctx.device);
    ctx.* = undefined;
}

const Partition = struct {
    context: cuda.Context,
    stream: cuda.Stream,
    ops: Ops,
    weights: weights.Weights,
    scratch: cuda.DeviceBuffer,
    b: Buffers,
    first: usize,
    end: usize,

    fn init(p: *Partition, e: *Engine, ordinal: c_int, first: usize, end: usize) !void {
        p.first = first;
        p.end = end;
        p.context = try cuda.Context.init(&e.driver, ordinal);
        errdefer releaseContext(&p.context);
        try checkFloor(&p.context);
        p.stream = try cuda.Stream.init(&e.driver, true);
        errdefer p.stream.deinit();
        p.ops = try Ops.load(&p.context);
        errdefer p.ops.unload();
        p.weights = try weights.Weights.load(e.gpa, e.io, &p.context, e.dir, e.config, first, end, first == 0, end == 40);
        errdefer p.weights.deinit();
        if ((try p.context.memInfo()).free < floor_bytes + scratchBytes()) return error.GpuMemoryFloor;
        p.scratch = try cuda.DeviceBuffer.alloc(&e.driver, scratchBytes());
        errdefer p.scratch.free();
        p.b = buffers(p.scratch);
        try checkFloor(&p.context);
    }

    fn deinit(p: *Partition) void {
        defer releaseContext(&p.context);
        defer p.weights.deinit();
        p.context.makeCurrent() catch return;
        p.stream.synchronize() catch {};
        p.scratch.free();
        p.ops.unload();
        p.stream.deinit();
    }

    fn projection(p: *Partition, w: weights.Projection, x: u64, out: u64) !void {
        return p.ops.linear(p.b.ck_rows, p.b.ck_part).decode(p.stream, w, .{ .x = x, .part = p.b.ck_part }, out, 1);
    }

    fn block(p: *Partition, layer: weights.Layer, state: *LayerState, position: usize, capacity: usize) !void {
        const b = p.b;
        const o = &p.ops;
        const s = p.stream;
        try o.rms(s, b.residual, layer.input_norm, b.x, 2048, 1e-6);
        switch (layer.mixer) {
            .gdn => |g| {
                try p.projection(g.qkv, b.x, b.qkv);
                try p.projection(g.z, b.x, b.z);
                try p.projection(g.a, b.x, b.a);
                try p.projection(g.b, b.x, b.b);
                try o.gdnPre(s, b.qkv, g.conv, state.gdn.history.ptr, b.a, b.b, g.a_log, g.dt_bias, b.q, b.k, b.v, b.decay, b.beta);
                try o.gdnStep(s, b.q, b.k, b.v, b.decay, b.beta, state.gdn.recurrent.ptr, b.recurrent_out);
                try o.gdnOut(s, b.recurrent_out, b.z, g.norm, b.mixed, 1e-6);
                try p.projection(g.out, b.mixed, b.update);
            },
            .full => |f| {
                try p.projection(f.q, b.x, b.mixed);
                try p.projection(f.k, b.x, b.full_k);
                try p.projection(f.v, b.x, b.full_v);
                try o.attention(s, b.mixed, b.full_k, b.full_v, f.q_norm, f.k_norm, state.full.keys.ptr, state.full.values.ptr, position, capacity, b.qkv, 1e-6);
                try p.projection(f.out, b.qkv, b.update);
            },
        }
        try o.add(s, b.residual, b.update, 2048);
        try o.rms(s, b.residual, layer.post_norm, b.x, 2048, 1e-6);
        const m = layer.moe;
        try o.route(s, b.x, m.router, m.shared_gate, b.picks, b.route_weights);
        const plan: cuda.grouped.Plan = .{ .members = b.members, .items = b.items, .counts = b.counts, .rank = 0, .hist = 0 };
        try o.router.route(s, b.picks, 8, 256, 16, plan);
        try o.grouped().decode(s, .up, .{ .x = b.x, .stride = 2048, .slots = 8 }, m.experts, plan, b.expert_hidden, 8, -1);
        try o.grouped().decode(s, .down_f32, .{ .x = b.expert_hidden, .stride = 512 }, m.experts, plan, b.pairs, 8, -1);
        try p.projection(m.shared.gate, b.x, b.shared_gate);
        try p.projection(m.shared.up, b.x, b.shared_up);
        try o.swiglu(s, b.shared_gate, b.shared_up, b.shared_act, 512);
        try p.projection(m.shared.down, b.shared_act, b.shared_out);
        try o.combine(s, b.pairs, b.route_weights, b.shared_out, b.update);
        try o.add(s, b.residual, b.update, 2048);
    }
};

pub const Engine = struct {
    gpa: Allocator,
    io: std.Io,
    dir: []const u8,
    config: checkpoint.Config,
    capacity: usize,
    driver: cuda.Driver,
    partitions: []Partition,
    transfer: cuda.HostBuffer,

    pub fn open(gpa: Allocator, io: std.Io, dir: []const u8, options: Options) !*Engine {
        const cfg = try checkpoint.Config.read(gpa, io, dir);
        try validateOptions(options, cfg.context);
        const e = try gpa.create(Engine);
        errdefer gpa.destroy(e);
        e.gpa = gpa;
        e.io = io;
        e.config = cfg;
        e.capacity = options.context;
        e.dir = try gpa.dupe(u8, dir);
        errdefer gpa.free(e.dir);
        e.driver = try cuda.Driver.open();
        errdefer e.driver.close();
        for (options.devices, 0..) |device, i| {
            const first = if (i == 0) 0 else options.cut;
            const end = if (i + 1 == options.devices.len) 40 else options.cut;
            const plan = try weights.planPartition(gpa, io, dir, cfg, first, end);
            const required = try requiredMemory(plan, cfg, first, end, options.context);
            var context = try cuda.Context.init(&e.driver, device);
            defer releaseContext(&context);
            const memory = try context.memInfo();
            checkCapacity(memory.free, required) catch |err| {
                std.log.warn("Qwen CUDA device {d} needs at least {d} MiB free for layers {d}..{d}, including the {d} MiB floor; available {d} MiB", .{ device, (required + (1 << 20) - 1) >> 20, first, end, floor_bytes >> 20, memory.free >> 20 });
                return err;
            };
        }
        const parts = try gpa.alloc(Partition, options.devices.len);
        errdefer gpa.free(parts);
        var initialized: usize = 0;
        errdefer for (parts[0..initialized]) |*p| p.deinit();
        for (parts, options.devices, 0..) |*p, device, i| {
            const first = if (i == 0) 0 else options.cut;
            const end = if (i + 1 == parts.len) 40 else options.cut;
            try p.init(e, device, first, end);
            initialized += 1;
        }
        e.partitions = parts;
        e.transfer = try cuda.HostBuffer.alloc(&e.driver, 2048 * 2);
        return e;
    }

    pub fn deinit(e: *Engine) void {
        e.transfer.free();
        for (e.partitions) |*p| p.deinit();
        e.gpa.free(e.partitions);
        e.driver.close();
        const a = e.gpa;
        a.free(e.dir);
        a.destroy(e);
    }

    pub fn createSession(e: *Engine) !Session {
        const states = try e.gpa.alloc(LayerState, 40);
        errdefer e.gpa.free(states);
        var initialized: usize = 0;
        errdefer {
            for (e.partitions) |*p| {
                if (initialized <= p.first) continue;
                p.context.makeCurrent() catch continue;
                for (p.first..@min(p.end, initialized)) |i| states[i].deinit();
            }
        }
        for (e.partitions) |*p| {
            try p.context.makeCurrent();
            var needed: usize = 0;
            for (p.first..p.end) |i| needed += try stateBytes(e.config.spec.full(i), e.capacity);
            if ((try p.context.memInfo()).free < floor_bytes + needed) return error.GpuMemoryFloor;
            for (p.first..p.end) |i| {
                states[i] = try LayerState.init(&e.driver, e.config.spec.full(i), e.capacity);
                initialized += 1;
            }
            try checkFloor(&p.context);
        }
        const logits = try e.gpa.alloc(u16, 248320);
        errdefer e.gpa.free(logits);
        var session: Session = .{ .engine = e, .layers = states, .logits = logits };
        try session.reset();
        return session;
    }
};

pub fn validateOptions(options: Options, limit: usize) !void {
    if (options.context == 0 or options.context > limit) return error.InvalidContext;
    if (options.devices.len < 1 or options.devices.len > 2) return error.InvalidDevices;
    for (options.devices, 0..) |d, i| {
        if (d < 0) return error.InvalidDevices;
        for (options.devices[0..i]) |previous| if (previous == d) return error.InvalidDevices;
    }
    if (options.devices.len == 2 and (options.cut == 0 or options.cut >= 40)) return error.InvalidLayerCut;
}

const LayerState = union(enum) {
    full: struct { keys: cuda.DeviceBuffer, values: cuda.DeviceBuffer },
    gdn: struct { history: cuda.DeviceBuffer, recurrent: cuda.DeviceBuffer },

    fn init(d: *const cuda.Driver, full: bool, capacity: usize) !LayerState {
        if (full) {
            var k = try cuda.DeviceBuffer.alloc(d, capacity * 512 * 2);
            errdefer k.free();
            return .{ .full = .{ .keys = k, .values = try cuda.DeviceBuffer.alloc(d, capacity * 512 * 2) } };
        }
        var history = try cuda.DeviceBuffer.alloc(d, 3 * 8192 * 2);
        errdefer history.free();
        return .{ .gdn = .{ .history = history, .recurrent = try cuda.DeviceBuffer.alloc(d, 32 * 128 * 128 * 4) } };
    }

    fn clear(state: *LayerState, stream: cuda.Stream) !void {
        switch (state.*) {
            .full => |kv| {
                try kv.keys.fill8(0, stream.handle);
                try kv.values.fill8(0, stream.handle);
            },
            .gdn => |g| {
                try g.history.fill8(0, stream.handle);
                try g.recurrent.fill8(0, stream.handle);
            },
        }
    }

    fn deinit(state: *LayerState) void {
        switch (state.*) {
            .full => |*kv| {
                kv.keys.free();
                kv.values.free();
            },
            .gdn => |*g| {
                g.history.free();
                g.recurrent.free();
            },
        }
    }
};

pub const Session = struct {
    engine: *Engine,
    layers: []LayerState,
    logits: []u16,
    position: usize = 0,
    poisoned: bool = true,

    pub fn deinit(session: *Session) void {
        const e = session.engine;
        for (e.partitions) |*p| {
            p.context.makeCurrent() catch continue;
            p.stream.synchronize() catch {};
            for (session.layers[p.first..p.end]) |*state| state.deinit();
        }
        e.gpa.free(session.layers);
        e.gpa.free(session.logits);
        session.* = undefined;
    }

    pub fn reset(session: *Session) !void {
        session.poisoned = true;
        for (session.engine.partitions) |*p| {
            try p.context.makeCurrent();
            for (session.layers[p.first..p.end]) |*state| try state.clear(p.stream);
            try p.stream.synchronize();
        }
        session.position = 0;
        session.poisoned = false;
    }

    pub fn append(session: *Session, tokens: []const u32) !void {
        const e = session.engine;
        if (session.poisoned) return error.PoisonedSession;
        if (tokens.len > e.capacity - session.position) return error.ContextFull;
        for (tokens) |id| if (id >= 248320) return error.InvalidToken;
        if (tokens.len == 0) return;
        errdefer session.poisoned = true;
        for (tokens) |id| {
            for (e.partitions, 0..) |*p, rank| {
                try p.context.makeCurrent();
                if (rank == 0) {
                    try p.ops.embed(p.stream, p.weights.embed.?, id, p.b.residual);
                } else {
                    try p.scratch.upload(@intCast(p.b.residual - p.scratch.ptr), e.transfer.slice(u8));
                }
                for (p.weights.layers, p.first..) |layer, i| try p.block(layer, &session.layers[i], session.position, e.capacity);
                if (rank + 1 < e.partitions.len) {
                    try p.stream.synchronize();
                    try p.scratch.download(@intCast(p.b.residual - p.scratch.ptr), e.transfer.slice(u8));
                } else {
                    try p.ops.rms(p.stream, p.b.residual, p.weights.norm.?, p.b.x, 2048, 1e-6);
                    try p.projection(p.weights.head.?, p.b.x, p.b.logits);
                    try p.stream.synchronize();
                }
                try checkFloor(&p.context);
            }
            session.position += 1;
        }
        const last = &e.partitions[e.partitions.len - 1];
        try last.scratch.download(@intCast(last.b.logits - last.scratch.ptr), std.mem.sliceAsBytes(session.logits));
    }

    pub fn copyLogits(session: *Session, out: []f32) !void {
        if (session.poisoned or session.position == 0) return error.NoLogits;
        if (out.len != 248320) return error.InvalidLogitsLength;
        for (out, session.logits) |*value, bits| value.* = @bitCast(@as(u32, bits) << 16);
    }
};

test "context and device selection are validated before CUDA loading" {
    try validateOptions(.{}, 262144);
    try std.testing.expectError(error.InvalidContext, validateOptions(.{ .context = 0 }, 262144));
    try std.testing.expectError(error.InvalidContext, validateOptions(.{ .context = 262145 }, 262144));
    try std.testing.expectError(error.InvalidDevices, validateOptions(.{ .devices = &.{ 0, 0 } }, 262144));
    try std.testing.expectError(error.InvalidDevices, validateOptions(.{ .devices = &.{-1} }, 262144));
    try std.testing.expectError(error.InvalidLayerCut, validateOptions(.{ .cut = 40 }, 262144));
}

test "session cache budgets include both KV planes and complete GDN state" {
    try std.testing.expectEqual(@as(usize, 2048 * 4096), try stateBytes(true, 4096));
    try std.testing.expectEqual(@as(usize, 2 * 1024 * 1024 + 49152), try stateBytes(false, 4096));
    try std.testing.expect(scratchBytes() > 8 << 20);
}

test "multi-stream memory budget includes full KV and GDN state for every stream" {
    const cfg = try checkpoint.Config.parse(std.testing.allocator, @embedFile("fixtures/ornith_nvfp4/config.json"));
    const budget = try memoryBudget(.{ .weights = 1024 }, cfg, 0, 40, 131072, 4);
    const state = 10 * 131072 * 2048 + 30 * (32 * 128 * 128 * 4 + 3 * 8192 * 2);
    try std.testing.expectEqual(@as(usize, state), budget.stream_bytes);
    try std.testing.expectEqual(@as(usize, 4 * state), budget.state_bytes);
    try std.testing.expectEqual(@as(usize, 1024 + scratchBytes() + 4 * state), budget.steady_bytes);
    try std.testing.expectEqual(budget.steady_bytes + floor_bytes, budget.required_bytes);
    try std.testing.expectEqual(try requiredMemory(.{ .weights = 1024 }, cfg, 0, 40, 131072), (try memoryBudget(.{ .weights = 1024 }, cfg, 0, 40, 131072, 1)).required_bytes);
}

test "memory budget accounts for load staging and rejects invalid or overflowing requests" {
    const cfg = try checkpoint.Config.parse(std.testing.allocator, @embedFile("fixtures/ornith_nvfp4/config.json"));
    const budget = try memoryBudget(.{ .weights = 1024, .staging = 4 << 30 }, cfg, 0, 20, 4096, 1);
    try std.testing.expectEqual(@as(usize, 1024 + (4 << 30) + floor_bytes), budget.required_bytes);
    try std.testing.expectError(error.InvalidStreamCount, memoryBudget(.{}, cfg, 0, 20, 4096, 0));
    try std.testing.expectError(error.InvalidContext, memoryBudget(.{}, cfg, 0, 20, 0, 1));
    try std.testing.expectError(error.InvalidLayerRange, memoryBudget(.{}, cfg, 20, 41, 4096, 1));
    try std.testing.expectError(error.Overflow, memoryBudget(.{}, cfg, 0, 40, 131072, std.math.maxInt(usize)));
}

test "partition teardown releases its primary retain when context selection fails" {
    const Mock = struct {
        var releases: usize = 0;

        fn select(_: cuda.abi.Context) callconv(.c) cuda.abi.Result {
            return cuda.abi.error_not_ready;
        }

        fn release(device: cuda.abi.Device) callconv(.c) cuda.abi.Result {
            std.debug.assert(device == 7);
            releases += 1;
            return cuda.abi.success;
        }
    };
    var driver: cuda.Driver = .{ .lib = undefined, .api = undefined };
    driver.api.cuCtxSetCurrent = Mock.select;
    driver.api.cuDevicePrimaryCtxRelease_v2 = Mock.release;
    var partition: Partition = undefined;
    partition.context = .{ .d = &driver, .device = 7, .handle = @ptrFromInt(32) };
    partition.weights = .{ .layers = try std.testing.allocator.alloc(weights.Layer, 1), .gpa = std.testing.allocator, .ctx = &partition.context };
    Mock.releases = 0;
    partition.deinit();
    try std.testing.expectEqual(@as(usize, 1), Mock.releases);
}
