//! The native Clef decision engine: the shared BF16 Qwen backbone's hidden rows, the host lexical gather and the
//! joint head on one CUDA context, behind the tfclef C ABI.
const std = @import("std");
const qwen = @import("qwen35");
const record = @import("record.zig");
const cuda = @import("cuda");
const abi = @import("abi.zig");
const head_mod = @import("head.zig");
const build_options = @import("clef_build");
const checkpoint = @import("head_checkpoint.zig");

/// The backbone's share: the full BF16 engine, or only its mapped vocabulary tables for head-only parity runs.
pub const Backbone = union(enum) {
    engine: qwen.engine.Engine.Bf16Budget,
    tables_only: usize,
};

/// Logical device and host allocations at open. Every request reuses them; none allocates.
pub const Budget = struct {
    backbone_steady: usize,
    backbone_load_peak: usize,
    head_weights: usize,
    head_load_stage: usize,
    head_scratch: usize,
    lexical_device: usize,
    host_lexical: usize,
    host_staging: usize,
    host_tables: usize,
    /// Pinned host bytes: the lexical stage, the request tables and logits, the backbone's input staging.
    host_pinned: usize,
    steady: usize,
    load_peak: usize,
    /// What the runtime's allocation limit counts at the load peak: device bytes plus pinned host bytes.
    accounted_peak: usize,

    pub fn of(limits: record.Limits, head_bytes: checkpoint.Bytes, backbone: Backbone) !Budget {
        const add = std.math.add;
        const d = head_mod.Dims.of(limits);
        const head_only = backbone == .tables_only;
        const scratch = try head_mod.Scratch.layout(d, head_only);
        const rows = try std.math.mul(usize, d.tokens, head_mod.hidden * 2);
        const staging = try add(usize, scratch.tables.bytes, try std.math.mul(usize, d.options, 2));
        const steady_b, const peak_b, const tables, const input = switch (backbone) {
            .engine => |e| .{ e.steady_device_bytes, e.load_device_peak_bytes, e.host_tables_bytes, e.host_input_bytes },
            .tables_only => |t| .{ 0, 0, t, 0 },
        };
        const pinned = try add(usize, try add(usize, rows, staging), input);
        const steady = try add(usize, try add(usize, steady_b, head_bytes.device_weights), try add(usize, scratch.bytes, rows));
        const head_load = try add(usize, try add(usize, steady_b, head_bytes.device_weights), head_bytes.load_stage);
        const load_peak = @max(steady, @max(peak_b, head_load));
        return .{
            .backbone_steady = steady_b,
            .backbone_load_peak = peak_b,
            .head_weights = head_bytes.device_weights,
            .head_load_stage = head_bytes.load_stage,
            .head_scratch = scratch.bytes,
            .lexical_device = rows,
            .host_lexical = rows,
            .host_staging = staging,
            .host_tables = tables,
            .host_pinned = pinned,
            .steady = steady,
            .load_peak = load_peak,
            .accounted_peak = try add(usize, load_peak, pinned),
        };
    }
};

/// Where a request's hidden rows and lexical rows come from.
pub const Source = union(enum) {
    /// Head-only parity runs: caller hidden rows, and the output table mapped from the checkpoint.
    tables: qwen.bf16.HostTables,
};

const GetUuid = *const fn (*[16]u8, cuda.abi.Device) callconv(.c) cuda.abi.Result;

/// One CUDA context, its stream and every buffer a request uses, all allocated at open.
pub const Engine = struct {
    gpa: std.mem.Allocator,
    threaded: std.Io.Threaded,
    driver: cuda.Driver,
    ctx: cuda.Context,
    stream: cuda.Stream,
    bounds: record.Limits,
    budget: Budget,
    uuid: [16]u8,
    kernels: head_mod.Kernels,
    weights: cuda.DeviceBuffer,
    scratch: cuda.DeviceBuffer,
    lexical: cuda.DeviceBuffer,
    /// The request tables, then the logits read back.
    staging: cuda.HostBuffer,
    lexical_host: cuda.HostBuffer,
    joint: head_mod.Head,
    source: Source,

    /// Opens the device whose UUID matches, refuses a budget above `cap_bytes` or the device's free memory, caps the
    /// runtime's allocations there, loads the head (its file hashed against the pin) and allocates every buffer.
    pub fn open(gpa: std.mem.Allocator, o: abi.Open, m: *abi.Message) !*Engine {
        if (!o.head_only) {
            m.set("the backbone decision path is not built yet; open with TF_CLEF_OPEN_HEAD_ONLY", .{});
            return error.BackboneUnavailable;
        }
        const e = try gpa.create(Engine);
        errdefer gpa.destroy(e);
        e.gpa = gpa;
        e.uuid = o.device_uuid;
        e.threaded = .init(gpa, .{});
        errdefer e.threaded.deinit();
        const io = e.threaded.io();
        e.driver = cuda.Driver.open() catch |err| {
            m.set("CUDA driver: {s}", .{@errorName(err)});
            return err;
        };
        errdefer e.driver.close();
        const ordinal = findDevice(&e.driver, o.device_uuid) catch |err| {
            m.set("no CUDA device has the requested UUID ({s})", .{@errorName(err)});
            return err;
        };
        e.ctx = try cuda.Context.init(&e.driver, ordinal);
        errdefer e.ctx.deinit();

        var backbone = try qwen.bf16.Plan.inspect(gpa, io, o.model_dir);
        defer backbone.deinit();
        var plan = try checkpoint.Plan.inspect(gpa, io, o.model_dir);
        defer plan.deinit();
        e.bounds = try record.Limits.init(o.max_tokens, std.math.cast(u32, backbone.geometry.vocab) orelse return error.Overflow);
        e.budget = try Budget.of(e.bounds, plan.bytes, .{ .tables_only = backbone.bytes().host_tables });
        const memory = try e.ctx.memInfo();
        if (e.budget.accounted_peak > o.cap_bytes or e.budget.load_peak > memory.free) {
            m.set("budget {d} bytes (peak {d} with pinned host) exceeds the cap {d} or the free {d}", .{ e.budget.load_peak, e.budget.accounted_peak, o.cap_bytes, memory.free });
            return error.BudgetExceedsCapacity;
        }
        _ = try cuda.setAllocationLimit(o.cap_bytes);
        errdefer _ = cuda.setAllocationLimit(null) catch {};

        e.source = .{ .tables = try backbone.upload(gpa, io, o.model_dir, &NoDevice{}) };
        errdefer e.source.tables.deinit();
        e.stream = try cuda.Stream.init(&e.driver, true);
        errdefer e.stream.deinit();
        e.kernels = head_mod.Kernels.load(&e.ctx) catch |err| {
            m.set("kernel images: {s}", .{@errorName(err)});
            return err;
        };
        errdefer e.kernels.unload();
        e.weights = try cuda.DeviceBuffer.alloc(&e.driver, plan.bytes.device_weights);
        errdefer e.weights.free();
        var weights: head_mod.Weights = undefined;
        {
            var pack = try cuda.modelopt.Packer.load(&e.driver);
            defer pack.unload();
            var stage = try cuda.DeviceBuffer.alloc(&e.driver, plan.bytes.load_stage);
            defer stage.free();
            const path = try std.fs.path.join(gpa, &.{ o.model_dir, "joint_head.safetensors" });
            defer gpa.free(path);
            weights = head_mod.loadWeights(&plan, io, path, o.head_sha256, e.weights, stage, e.stream, &pack) catch |err| {
                m.set("joint head weights: {s}", .{@errorName(err)});
                return err;
            };
        }
        const dims = head_mod.Dims.of(e.bounds);
        const layout = try head_mod.Scratch.layout(dims, true);
        e.scratch = try cuda.DeviceBuffer.alloc(&e.driver, layout.bytes);
        errdefer e.scratch.free();
        e.lexical = try cuda.DeviceBuffer.alloc(&e.driver, e.budget.lexical_device);
        errdefer e.lexical.free();
        e.staging = try cuda.HostBuffer.alloc(&e.driver, e.budget.host_staging);
        errdefer e.staging.free();
        e.lexical_host = try cuda.HostBuffer.alloc(&e.driver, e.budget.host_lexical);
        errdefer e.lexical_host.free();
        e.joint = .{ .k = &e.kernels, .w = weights, .scratch = layout, .base = e.scratch.ptr, .dims = dims };
        return e;
    }

    pub fn limits(e: *Engine) record.Limits {
        return e.bounds;
    }

    pub fn decide(e: *Engine, rec: record.Record, out: []f32, m: *abi.Message) abi.Error!void {
        _ = .{ e, rec, out };
        m.set("this engine has no backbone", .{});
        return error.Invalid;
    }

    /// The head over caller rows: [ids.len, 4096] BF16 bits after the backbone's final norm, in host memory.
    pub fn head(e: *Engine, rec: record.Record, rows: []const u16, out: []f32, m: *abi.Message) abi.Error!void {
        const tables = switch (e.source) {
            .tables => |*t| t,
        };
        try e.bind(m);
        e.gather(tables, rec) catch |err| {
            m.set("lexical rows refused: {s}", .{@errorName(err)});
            return error.Invalid;
        };
        e.submit(rec, std.mem.sliceAsBytes(rows)) catch |err| return e.fault(err, m);
        return e.read(rec, out, m);
    }

    /// Binds this engine's context on the calling thread; a context that cannot be made current is unusable.
    fn bind(e: *Engine, m: *abi.Message) abi.Error!void {
        e.ctx.makeCurrent() catch |err| {
            m.set("the engine's CUDA context cannot be made current: {s}", .{@errorName(err)});
            return error.Fatal;
        };
    }

    /// Every option span's output-table rows, packed in option order into the pinned lexical stage. Host only.
    fn gather(e: *Engine, tables: *const qwen.bf16.HostTables, rec: record.Record) !void {
        const row = head_mod.hidden * 2;
        var at: usize = 0;
        for (rec.options) |o| {
            const n = (o.end - o.start) * row;
            try tables.gather(.output, rec.ids[o.start..o.end], e.lexical_host.bytes[at..][0..n]);
            at += n;
        }
    }

    /// Tables, lexical rows and hidden rows in; the head; logits out; one synchronization.
    fn submit(e: *Engine, rec: record.Record, hidden_bytes: []const u8) !void {
        const t = e.joint.scratch.tables;
        head_mod.stageTables(t, rec, e.staging.bytes[0..t.bytes]);
        try e.copyIn(e.joint.at(.tables), e.staging.bytes[0..t.bytes]);
        try e.copyIn(e.lexical.ptr, e.lexical_host.bytes[0 .. @as(usize, rec.lexical_rows) * head_mod.hidden * 2]);
        try e.copyIn(e.joint.at(.input), hidden_bytes);
        try e.joint.run(e.stream, .{ .hidden = e.joint.at(.input), .lexical = e.lexical.ptr, .rec = &rec });
        try e.copyOut(e.staging.bytes[t.bytes..][0 .. rec.options.len * 2], e.joint.at(.logits));
        try e.stream.synchronize();
    }

    /// The BF16 logits widened to f32 in option order; a NaN or infinite logit refuses the answer.
    fn read(e: *Engine, rec: record.Record, out: []f32, m: *abi.Message) abi.Error!void {
        const bits = e.staging.bytes[e.joint.scratch.tables.bytes..][0 .. rec.options.len * 2];
        var bad: ?usize = null;
        for (out, 0..) |*x, i| {
            x.* = @bitCast(@as(u32, std.mem.readInt(u16, bits[i * 2 ..][0..2], .little)) << 16);
            if (!std.math.isFinite(x.*) and bad == null) bad = i;
        }
        if (bad) |i| {
            m.set("logit {d} is not finite", .{i});
            return error.NonFinite;
        }
    }

    /// A failed device step: drained (stream, else context) is a recoverable fault; undrained poisons the engine.
    /// This never calls the shared drain-or-exit helper.
    fn fault(e: *Engine, err: anyerror, m: *abi.Message) abi.Error {
        if (e.stream.synchronize()) |_| {
            m.set("device fault ({s}); the stream drained", .{@errorName(err)});
            return error.Device;
        } else |_| {}
        if (e.driver.synchronize()) |_| {
            m.set("device fault ({s}); the context drained", .{@errorName(err)});
            return error.Device;
        } else |_| {}
        m.set("device fault ({s}); the context did not drain", .{@errorName(err)});
        return error.Fatal;
    }

    fn copyIn(e: *Engine, dst: u64, src: []const u8) !void {
        if (src.len == 0) return;
        try e.driver.check(e.driver.api.cuMemcpyHtoDAsync_v2(dst, src.ptr, src.len, e.stream.handle), "cuMemcpyHtoDAsync");
    }

    fn copyOut(e: *Engine, dst: []u8, src: u64) !void {
        if (dst.len == 0) return;
        try e.driver.check(e.driver.api.cuMemcpyDtoHAsync_v2(dst.ptr, src, dst.len, e.stream.handle), "cuMemcpyDtoHAsync");
    }

    pub fn info(e: *Engine, out: *abi.Info) void {
        out.budget_bytes = e.budget.steady;
        out.device_uuid = e.uuid;
        const commit = build_options.source_commit;
        @memcpy(out.source_commit[0..@min(commit.len, 40)], commit[0..@min(commit.len, 40)]);
        if (e.ctx.memInfo()) |mem| {
            out.device_free_bytes = mem.free;
            out.device_total_bytes = mem.total;
        } else |_| {}
    }

    /// Device teardown runs only while the context drains; a poisoned engine releases host state only and leaves
    /// its device objects to process exit, so no shared drain can end the process here.
    pub fn close(e: *Engine, poisoned: bool) void {
        const drained = !poisoned and if (e.ctx.makeCurrent()) |_| (if (e.stream.synchronize()) |_| true else |_| false) else |_| false;
        if (drained) {
            e.lexical_host.free();
            e.staging.free();
            e.lexical.free();
            e.scratch.free();
            e.weights.free();
            e.kernels.unload();
            e.stream.deinit();
        }
        switch (e.source) {
            .tables => |*t| t.deinit(),
        }
        _ = cuda.setAllocationLimit(null) catch {};
        if (drained) {
            e.ctx.deinit();
            e.driver.close();
        }
        e.threaded.deinit();
        e.gpa.destroy(e);
    }
};

/// The upload target that keeps only the host tables: head-only engines load no backbone weights.
const NoDevice = struct {
    pub fn tensor(_: *const NoDevice, _: anytype, _: anytype) !void {}
};

fn findDevice(d: *cuda.Driver, uuid: [16]u8) !c_int {
    const get = d.lib.lookup(GetUuid, "cuDeviceGetUuid_v2") orelse d.lib.lookup(GetUuid, "cuDeviceGetUuid") orelse return error.MissingSymbol;
    const count = try d.deviceCount();
    var i: c_int = 0;
    while (i < count) : (i += 1) {
        var dev: cuda.abi.Device = 0;
        try d.check(d.api.cuDeviceGet(&dev, i), "cuDeviceGet");
        var got: [16]u8 = undefined;
        try d.check(get(&got, dev), "cuDeviceGetUuid");
        if (std.mem.eql(u8, &got, &uuid)) return i;
    }
    return error.DeviceNotFound;
}

fn pinnedBackbone() qwen.engine.Engine.Bf16Budget {
    // Engine.bf16Budget at 4096 tokens over the pinned checkpoint metadata (docs/clef-bf16.md, its host test).
    return .{
        .weights_bytes = 13_879_009_280,
        .load_stage_bytes = 100_663_296,
        .scratch_bytes = 28_567_552,
        .cache_bytes = 187_403_264,
        .hidden_bytes = 33_554_432,
        .host_input_bytes = 131_072,
        .host_tables_bytes = 4_068_474_880,
        .host_conversion_peak = 201_326_592,
        .steady_device_bytes = 14_128_534_528,
        .load_device_peak_bytes = 14_128_534_528,
    };
}

fn pinnedHead() !checkpoint.Bytes {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const capture = try std.json.parseFromSliceLeaky(std.json.Value, a, @embedFile("fixtures/capture.json"), .{});
    const plan = try checkpoint.build(a, @embedFile("fixtures/joint_head_config.json"), @embedFile("fixtures/joint_head.safetensors.header.json"), @intCast(capture.object.get("file_bytes").?.integer));
    return plan.bytes;
}

test "the 4096-token budget sums its parts with the pinned backbone and head" {
    const limits = try record.Limits.init(4096, 248320);
    const h = try pinnedHead();
    const b = try Budget.of(limits, h, .{ .engine = pinnedBackbone() });
    try std.testing.expectEqual(b.backbone_steady + b.head_weights + b.head_scratch + b.lexical_device, b.steady);
    try std.testing.expectEqual(@as(usize, 4096 * 4096 * 2), b.lexical_device);
    try std.testing.expectEqual(b.lexical_device, b.host_lexical);
    try std.testing.expectEqual(@max(b.steady, b.backbone_steady + b.head_weights + b.head_load_stage), b.load_peak);
    try std.testing.expectEqual(@as(usize, 4_068_474_880), b.host_tables);
    try std.testing.expectEqual((try head_mod.Scratch.layout(head_mod.Dims.of(limits), false)).bytes, b.head_scratch);
    try std.testing.expect(b.steady < 14_700 << 20);
    try std.testing.expectEqual(@as(usize, 14_632_435_712), b.steady);
    try std.testing.expectEqual(b.host_lexical + b.host_staging + 131_072, b.host_pinned);
    try std.testing.expectEqual(b.load_peak + b.host_pinned, b.accounted_peak);
}

test "a head-only budget carries no backbone and owns its hidden upload rows" {
    const limits = try record.Limits.init(4096, 248320);
    const h = try pinnedHead();
    const full = try Budget.of(limits, h, .{ .engine = pinnedBackbone() });
    const only = try Budget.of(limits, h, .{ .tables_only = 4_068_474_880 });
    try std.testing.expectEqual(@as(usize, 0), only.backbone_steady);
    try std.testing.expectEqual(full.head_scratch + 4096 * 4096 * 2, only.head_scratch);
    try std.testing.expectEqual(only.head_weights + only.head_scratch + only.lexical_device, only.steady);
}

test "budget arithmetic is checked and its maxima follow the token limit" {
    const h = try pinnedHead();
    const small = try Budget.of(try record.Limits.init(64, 248320), h, .{ .engine = pinnedBackbone() });
    const large = try Budget.of(try record.Limits.init(4096, 248320), h, .{ .engine = pinnedBackbone() });
    try std.testing.expect(small.head_scratch < large.head_scratch and small.host_staging < large.host_staging);
    var huge = pinnedBackbone();
    huge.steady_device_bytes = std.math.maxInt(usize);
    try std.testing.expectError(error.Overflow, Budget.of(try record.Limits.init(64, 1), h, .{ .engine = huge }));
}
