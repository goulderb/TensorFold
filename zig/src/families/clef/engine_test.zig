//! Host observations of the engine's head-only request over a recording CUDA driver: binding, copies, one
//! synchronization, no allocation, and how launch and drain failures become statuses.
const std = @import("std");
const cuda = @import("cuda");
const cabi = cuda.abi;
const qwen = @import("qwen35");
const abi = @import("abi.zig");
const record = @import("record.zig");
const head = @import("head.zig");
const checkpoint = @import("head_checkpoint.zig");
const engine = @import("engine.zig");

const Copy = struct { dst: u64, len: usize };

const Fake = struct {
    var launches: usize = 0;
    var fail_launch_at: ?usize = null;
    var uploads: [8]Copy = undefined;
    var upload_count: usize = 0;
    var downloads: usize = 0;
    var syncs: usize = 0;
    var context_syncs: usize = 0;
    var fail_stream_sync: bool = false;
    var fail_context_sync: bool = false;
    var fail_bind: bool = false;
    var bound: cabi.Context = null;
    var allocations: usize = 0;
    var calls: usize = 0;
    var logits: [4]u16 = .{ 0x3f80, 0xc020, 0x3f40, 0 };

    fn reset() void {
        launches = 0;
        fail_launch_at = null;
        upload_count = 0;
        downloads = 0;
        syncs = 0;
        context_syncs = 0;
        fail_stream_sync = false;
        fail_context_sync = false;
        fail_bind = false;
        bound = null;
        allocations = 0;
        calls = 0;
        logits = .{ 0x3f80, 0xc020, 0x3f40, 0 };
    }
    fn launch(_: cabi.Function, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: cabi.Stream, _: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cabi.Result {
        calls += 1;
        launches += 1;
        if (fail_launch_at == launches) return cabi.error_not_ready;
        return 0;
    }
    fn upload(dst: u64, _: ?*const anyopaque, n: usize, _: cabi.Stream) callconv(.c) cabi.Result {
        calls += 1;
        uploads[upload_count] = .{ .dst = dst, .len = n };
        upload_count += 1;
        return 0;
    }
    fn download(dst: ?*anyopaque, _: u64, n: usize, _: cabi.Stream) callconv(.c) cabi.Result {
        calls += 1;
        downloads += 1;
        @memcpy(@as([*]u8, @ptrCast(dst.?))[0..n], std.mem.sliceAsBytes(&logits)[0..n]);
        return 0;
    }
    fn sync(_: cabi.Stream) callconv(.c) cabi.Result {
        calls += 1;
        syncs += 1;
        return if (fail_stream_sync) cabi.error_not_ready else 0;
    }
    fn contextSync() callconv(.c) cabi.Result {
        calls += 1;
        context_syncs += 1;
        return if (fail_context_sync) cabi.error_not_ready else 0;
    }
    fn bind(ctx: cabi.Context) callconv(.c) cabi.Result {
        calls += 1;
        if (fail_bind) return cabi.error_not_ready;
        bound = ctx;
        return 0;
    }
    fn alloc(_: *cabi.DevicePtr, _: usize) callconv(.c) cabi.Result {
        calls += 1;
        allocations += 1;
        return cabi.error_not_ready;
    }
    fn hostAlloc(_: *?*anyopaque, _: usize, _: c_uint) callconv(.c) cabi.Result {
        calls += 1;
        allocations += 1;
        return cabi.error_not_ready;
    }

    fn driver() cuda.Driver {
        var d: cuda.Driver = undefined;
        d.api.cuLaunchKernel = launch;
        d.api.cuMemcpyHtoDAsync_v2 = upload;
        d.api.cuMemcpyDtoHAsync_v2 = download;
        d.api.cuStreamSynchronize = sync;
        d.api.cuCtxSynchronize = contextSync;
        d.api.cuCtxSetCurrent = bind;
        d.api.cuMemAlloc_v2 = alloc;
        d.api.cuMemHostAlloc = hostAlloc;
        return d;
    }
};

const vocab = 100;
const row = head.hidden * 2;

const Fixture = struct {
    e: *engine.Engine,
    plan: checkpoint.Plan,
    output: []u8,
    staging: []align(16) u8,
    lexical: []align(16) u8,
    ids: [30]u32,
    questions: [2]record.C.Question,
    options: [3]record.Span,
    c: record.C.Record,
    scratch: record.Scratch,
    rec: record.Record,
    hidden: []u16,

    fn init(f: *Fixture) !void {
        const a = std.testing.allocator;
        Fake.reset();
        var arena: std.heap.ArenaAllocator = .init(a);
        const capture = try std.json.parseFromSliceLeaky(std.json.Value, arena.allocator(), @embedFile("fixtures/capture.json"), .{});
        f.plan = try checkpoint.build(arena.allocator(), @embedFile("fixtures/joint_head_config.json"), @embedFile("fixtures/joint_head.safetensors.header.json"), @intCast(capture.object.get("file_bytes").?.integer));
        f.plan.arena = arena;
        const limits = try record.Limits.init(64, vocab);
        const budget = try engine.Budget.of(limits, f.plan.bytes, .{ .tables_only = vocab * row });
        f.output = try a.alloc(u8, vocab * row);
        for (0..vocab) |i| @memset(f.output[i * row ..][0..row], @intCast(i));
        f.staging = try a.alignedAlloc(u8, .@"16", budget.host_staging);
        f.lexical = try a.alignedAlloc(u8, .@"16", budget.host_lexical);
        f.hidden = try a.alloc(u16, 30 * head.hidden);
        const e = try a.create(engine.Engine);
        f.e = e;
        e.gpa = a;
        e.threaded = .init(a, .{});
        e.driver = Fake.driver();
        e.ctx = .{ .d = &e.driver, .device = 0, .handle = @ptrFromInt(0x77) };
        e.stream = .{ .d = &e.driver, .handle = @ptrFromInt(0x99) };
        e.bounds = limits;
        e.budget = budget;
        e.uuid = @splat(4);
        const fn_: cuda.Function = .{ .d = &e.driver, .handle = @ptrFromInt(5) };
        inline for (@typeInfo(head.Fns).@"struct".field_names) |name| @field(e.kernels.fns, name) = fn_;
        e.kernels.vit = .{ .mod = undefined, .layer_norm_fn = fn_, .pointwise_fn = fn_, .bias_fn = fn_, .unshuffle_fn = fn_, .cast_fn = fn_, .position_fn = fn_ };
        e.kernels.lane = .{ .mod = undefined, .fns = @splat(@splat(fn_)), .f32_fns = @splat(@splat(fn_)), .reduce = @splat(fn_), .clusters = false };
        e.weights = .{ .d = &e.driver, .ptr = 0x1_0000_0000, .len = f.plan.bytes.device_weights, .borrowed = true };
        const dims = head.Dims.of(limits);
        const layout = try head.Scratch.layout(dims, true);
        e.scratch = .{ .d = &e.driver, .ptr = 0x2_0000_0000, .len = layout.bytes, .borrowed = true };
        e.lexical = .{ .d = &e.driver, .ptr = 0x3_0000_0000, .len = budget.lexical_device, .borrowed = true };
        e.staging = .{ .d = &e.driver, .bytes = f.staging };
        e.lexical_host = .{ .d = &e.driver, .bytes = f.lexical };
        e.joint = .{ .k = &e.kernels, .w = .{ .base = e.weights.ptr, .layout = f.plan.layout }, .scratch = layout, .base = e.scratch.ptr, .dims = dims };
        e.source = .{ .tables = .{ .gpa = a, .io = undefined, .maps = &.{}, .tables = .{ null, .{ .bytes = f.output, .rows = vocab, .row_bytes = row, .shard = 0 } } } };
        for (&f.ids, 0..) |*id, i| id.* = @intCast(i);
        f.questions = .{
            .{ .kind = 0, .question = .{ .start = 1, .end = 3 }, .first_option = 0, .option_count = 2 },
            .{ .kind = 1, .question = .{ .start = 10, .end = 12 }, .first_option = 2, .option_count = 1 },
        };
        f.options = .{ .{ .start = 4, .end = 6 }, .{ .start = 7, .end = 8 }, .{ .start = 13, .end = 16 } };
        f.c = .{ .ids = &f.ids, .n_ids = 30, .questions = &f.questions, .n_questions = 2, .options = &f.options, .n_options = 3 };
        f.scratch = try record.Scratch.alloc(a, limits);
        f.rec = try record.Record.parse(&f.c, limits, f.scratch);
    }

    fn deinit(f: *Fixture) void {
        const a = std.testing.allocator;
        f.e.threaded.deinit();
        a.destroy(f.e);
        f.scratch.free(a);
        a.free(f.hidden);
        a.free(f.lexical);
        a.free(f.staging);
        a.free(f.output);
        f.plan.deinit();
    }

    fn run(f: *Fixture, out: []f32, m: *abi.Message) abi.Error!void {
        return f.e.head(f.rec, f.hidden, out, m);
    }
};

test "a head request binds its context, copies tables, lexical and hidden rows, and reads logits after one sync" {
    var f: Fixture = undefined;
    try f.init();
    defer f.deinit();
    var out: [3]f32 = undefined;
    var m: abi.Message = .{};
    try f.run(&out, &m);
    try std.testing.expectEqualSlices(f32, &.{ 1.0, -2.5, 0.75 }, &out);
    try std.testing.expectEqual(f.e.ctx.handle, Fake.bound);
    try std.testing.expectEqual(@as(usize, 0), Fake.allocations);
    try std.testing.expectEqual(@as(usize, 1), Fake.syncs);
    try std.testing.expectEqual(@as(usize, 1), Fake.downloads);
    try std.testing.expectEqual(@as(usize, 3), Fake.upload_count);
    try std.testing.expectEqual(Copy{ .dst = f.e.joint.at(.tables), .len = f.e.joint.scratch.tables.bytes }, Fake.uploads[0]);
    try std.testing.expectEqual(Copy{ .dst = f.e.lexical.ptr, .len = 6 * row }, Fake.uploads[1]);
    try std.testing.expectEqual(Copy{ .dst = f.e.joint.at(.input), .len = 30 * row }, Fake.uploads[2]);
    try std.testing.expect(Fake.launches > 0);
    for ([_]u8{ 4, 5, 7, 13, 14, 15 }, 0..) |id, i| {
        try std.testing.expectEqual(id, f.lexical[i * row]);
        try std.testing.expectEqual(id, f.lexical[i * row + row - 1]);
    }
}

test "a non-finite logit is refused and the engine answers the next request" {
    var f: Fixture = undefined;
    try f.init();
    defer f.deinit();
    var out: [3]f32 = undefined;
    var m: abi.Message = .{};
    Fake.logits[1] = 0x7fc0;
    try std.testing.expectError(error.NonFinite, f.run(&out, &m));
    try std.testing.expectEqualStrings("logit 1 is not finite", m.text());
    Fake.logits[1] = 0xc020;
    try f.run(&out, &m);
    try std.testing.expectEqual(@as(f32, -2.5), out[1]);
}

test "a failed launch that drains is a device fault; one whose stream and context do not drain is fatal" {
    var f: Fixture = undefined;
    try f.init();
    defer f.deinit();
    var out: [3]f32 = undefined;
    var m: abi.Message = .{};
    Fake.fail_launch_at = 5;
    try std.testing.expectError(error.Device, f.run(&out, &m));
    try std.testing.expectEqualStrings("device fault (NotReady); the stream drained", m.text());
    try std.testing.expectEqual(@as(usize, 5), Fake.launches);
    try std.testing.expectEqual(@as(usize, 0), Fake.downloads);
    Fake.reset();
    Fake.fail_launch_at = 3;
    Fake.fail_stream_sync = true;
    try std.testing.expectError(error.Device, f.run(&out, &m));
    try std.testing.expectEqualStrings("device fault (NotReady); the context drained", m.text());
    Fake.reset();
    Fake.fail_launch_at = 3;
    Fake.fail_stream_sync = true;
    Fake.fail_context_sync = true;
    try std.testing.expectError(error.Fatal, f.run(&out, &m));
    try std.testing.expectEqualStrings("device fault (NotReady); the context did not drain", m.text());
}

test "a context that cannot be bound is fatal before any copy or launch" {
    var f: Fixture = undefined;
    try f.init();
    defer f.deinit();
    var out: [3]f32 = undefined;
    var m: abi.Message = .{};
    Fake.fail_bind = true;
    try std.testing.expectError(error.Fatal, f.run(&out, &m));
    try std.testing.expectEqual(@as(usize, 0), Fake.upload_count);
    try std.testing.expectEqual(@as(usize, 0), Fake.launches);
}

test "a head-only engine has no decision path" {
    var f: Fixture = undefined;
    try f.init();
    defer f.deinit();
    var out: [3]f32 = undefined;
    var m: abi.Message = .{};
    try std.testing.expectError(error.Invalid, f.e.decide(f.rec, &out, &m));
    try std.testing.expectEqual(@as(usize, 0), Fake.calls);
}

test "closing a poisoned engine releases host state without touching the device" {
    var f: Fixture = undefined;
    try f.init();
    const a = std.testing.allocator;
    defer {
        f.scratch.free(a);
        a.free(f.hidden);
        a.free(f.lexical);
        a.free(f.staging);
        a.free(f.output);
        f.plan.deinit();
    }
    f.e.close(true);
    try std.testing.expectEqual(@as(usize, 0), Fake.calls);
}

test "the device entry points compile on the host" {
    std.mem.doNotOptimizeAway(&engine.Engine.open);
    std.mem.doNotOptimizeAway(&engine.Engine.info);
    std.mem.doNotOptimizeAway(&engine.Engine.close);
    std.mem.doNotOptimizeAway(&head.Kernels.load);
    std.mem.doNotOptimizeAway(&head.loadWeights);
}
