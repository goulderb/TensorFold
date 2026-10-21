//! Host-only observations of the real hidden forward transaction, using a recording CUDA driver.
const std = @import("std");
const cuda = @import("cuda");
const engine = @import("cuda_engine.zig");
const hidden = @import("cuda_hidden.zig");
const Cache = @import("cuda_cache.zig").Cache;
const abi = cuda.abi;

const Fake = struct {
    var device: [16][32768]u8 align(256) = undefined;
    var host: [8][65536]u8 align(16) = undefined;
    var allocations: usize = 0;
    var host_allocations: usize = 0;
    var frees: usize = 0;
    var captures: usize = 0;
    var replays: usize = 0;
    var kernels: usize = 0;
    var norms: usize = 0;
    var copies: usize = 0;
    var uploads: usize = 0;
    var syncs: usize = 0;
    var context_syncs: usize = 0;
    var copy_bytes: usize = 0;
    var copy_destination: u64 = 0;
    var capturing: bool = false;
    var fail_copy: bool = false;
    var fail_upload: bool = false;
    var fail_launch: bool = false;
    var fail_fill: bool = false;
    var fail_bind: bool = false;
    var consumer_pending: bool = false;
    var freed_with_consumer: bool = false;
    var bound: abi.Context = null;
    var fail_syncs: usize = 0;
    var scratch: []u8 = undefined;
    var pattern: u8 = 1;

    fn init() void {
        allocations = 0;
        host_allocations = 0;
        frees = 0;
        captures = 0;
        replays = 0;
        kernels = 0;
        norms = 0;
        copies = 0;
        uploads = 0;
        syncs = 0;
        context_syncs = 0;
        capturing = false;
        fail_copy = false;
        fail_upload = false;
        fail_launch = false;
        fail_fill = false;
        fail_bind = false;
        consumer_pending = false;
        freed_with_consumer = false;
        bound = null;
        fail_syncs = 0;
        pattern = 1;
        @memset(std.mem.asBytes(&device), 0xa5);
    }
    fn alloc(out: *abi.DevicePtr, bytes: usize) callconv(.c) abi.Result {
        if (bytes > device[0].len or allocations == device.len) return 2;
        out.* = @intFromPtr(&device[allocations]);
        allocations += 1;
        return 0;
    }
    fn free(_: abi.DevicePtr) callconv(.c) abi.Result {
        frees += 1;
        if (consumer_pending) freed_with_consumer = true;
        return 0;
    }
    fn hostAlloc(out: *?*anyopaque, bytes: usize, _: c_uint) callconv(.c) abi.Result {
        if (bytes > 65536 or host_allocations == host.len) return 2;
        out.* = &host[host_allocations];
        host_allocations += 1;
        return 0;
    }
    fn hostFree(_: ?*anyopaque) callconv(.c) abi.Result {
        return 0;
    }
    fn eventCreate(out: *abi.Event, _: c_uint) callconv(.c) abi.Result {
        out.* = @ptrFromInt(100);
        return 0;
    }
    fn eventDone(_: abi.Event) callconv(.c) abi.Result {
        return 0;
    }
    fn eventRecord(_: abi.Event, _: abi.Stream) callconv(.c) abi.Result {
        return 0;
    }
    fn upload(dst: u64, src: ?*const anyopaque, bytes: usize, _: abi.Stream) callconv(.c) abi.Result {
        uploads += 1;
        if (fail_upload) return abi.error_not_ready;
        @memcpy(@as([*]u8, @ptrFromInt(dst))[0..bytes], @as([*]const u8, @ptrCast(src.?))[0..bytes]);
        return 0;
    }
    fn copy(dst: u64, src: u64, bytes: usize, _: abi.Stream) callconv(.c) abi.Result {
        if (capturing) return 1;
        copies += 1;
        copy_bytes = bytes;
        copy_destination = dst;
        @memcpy(@as([*]u8, @ptrFromInt(dst))[0..bytes], @as([*]const u8, @ptrFromInt(src))[0..bytes]);
        return if (fail_copy) abi.error_not_ready else 0;
    }
    fn sync(_: abi.Stream) callconv(.c) abi.Result {
        syncs += 1;
        if (fail_syncs != 0) {
            fail_syncs -= 1;
            return abi.error_not_ready;
        }
        consumer_pending = false;
        return 0;
    }
    fn contextSync() callconv(.c) abi.Result {
        context_syncs += 1;
        consumer_pending = false;
        return 0;
    }
    fn bind(ctx: abi.Context) callconv(.c) abi.Result {
        if (fail_bind) return abi.error_not_ready;
        bound = ctx;
        return 0;
    }
    fn fill(dst: u64, value: u8, bytes: usize, _: abi.Stream) callconv(.c) abi.Result {
        if (fail_fill) return abi.error_not_ready;
        @memset(@as([*]u8, @ptrFromInt(dst))[0..bytes], value);
        return 0;
    }
    fn begin(_: abi.Stream, _: abi.CaptureMode) callconv(.c) abi.Result {
        captures += 1;
        capturing = true;
        return 0;
    }
    fn end(_: abi.Stream, out: *abi.Graph) callconv(.c) abi.Result {
        capturing = false;
        out.* = @ptrFromInt(200);
        return 0;
    }
    fn nodes(_: abi.Graph, _: ?[*]abi.GraphNode, count: *usize) callconv(.c) abi.Result {
        count.* = 2;
        return 0;
    }
    fn instantiate(out: *abi.GraphExec, _: abi.Graph, _: c_ulonglong) callconv(.c) abi.Result {
        out.* = @ptrFromInt(300);
        return 0;
    }
    fn destroy(_: abi.Graph) callconv(.c) abi.Result {
        return 0;
    }
    fn destroyExec(_: abi.GraphExec) callconv(.c) abi.Result {
        return 0;
    }
    fn replay(_: abi.GraphExec, _: abi.Stream) callconv(.c) abi.Result {
        replays += 1;
        for (scratch, 0..) |*byte, i| byte.* = pattern + @as(u8, @intCast(i));
        return if (fail_launch) abi.error_not_ready else 0;
    }
    fn launch(f: abi.Function, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: abi.Stream, args: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) abi.Result {
        kernels += 1;
        if (f == @as(abi.Function, @ptrFromInt(2))) {
            norms += 1;
            const dst: *const u64 = @ptrCast(@alignCast(args.?[3].?));
            if (dst.* != @intFromPtr(scratch.ptr)) return 1;
            if (!capturing) for (scratch, 0..) |*byte, i| {
                byte.* = pattern + @as(u8, @intCast(i));
            };
        } else if (f != @as(abi.Function, @ptrFromInt(1))) return 1;
        return 0;
    }
    fn driver() cuda.Driver {
        var d: cuda.Driver = undefined;
        d.api.cuMemAlloc_v2 = alloc;
        d.api.cuMemFree_v2 = free;
        d.api.cuMemHostAlloc = hostAlloc;
        d.api.cuMemFreeHost = hostFree;
        d.api.cuEventCreate = eventCreate;
        d.api.cuEventDestroy_v2 = eventDone;
        d.api.cuEventQuery = eventDone;
        d.api.cuEventRecord = eventRecord;
        d.api.cuMemcpyHtoDAsync_v2 = upload;
        d.api.cuMemcpyDtoDAsync_v2 = copy;
        d.api.cuStreamSynchronize = sync;
        d.api.cuCtxSynchronize = contextSync;
        d.api.cuCtxSetCurrent = bind;
        d.api.cuMemsetD8Async = fill;
        d.api.cuStreamBeginCapture_v2 = begin;
        d.api.cuStreamEndCapture = end;
        d.api.cuGraphGetNodes = nodes;
        d.api.cuGraphInstantiateWithFlags = instantiate;
        d.api.cuGraphDestroy = destroy;
        d.api.cuGraphExecDestroy = destroyExec;
        d.api.cuGraphLaunch = replay;
        d.api.cuLaunchKernel = launch;
        return d;
    }
};

const Fixture = struct {
    d: cuda.Driver,
    e: engine.Engine,
    cache: Cache,
    tables: [16384]u8 align(16),
    staged: [16384]u8 align(16),
    x: [128]u8 align(16),
    state: [16]u8 align(16),
    logits: [4]u16,

    fn init(f: *Fixture) void {
        Fake.init();
        f.d = Fake.driver();
        f.e = undefined;
        const e = &f.e;
        e.gpa = std.testing.allocator;
        e.d = &f.d;
        e.context = cuda.copy_source.current();
        e.g = .{ .hidden = 4, .intermediate = 4, .layers = 0, .vocab = 4, .k_heads = 1, .v_heads = 1, .q_heads = 1, .kv_heads = 1, .eps = 1e-6, .tied = false, .context = 4096, .linear = @splat(false) };
        e.placement = .{ .first = 0, .end = 1, .total = 1 };
        e.numerics = .hf_modelopt;
        e.modelopt = null;
        e.kv8 = false;
        e.n_taps = 0;
        e.verify_terminal_graphs = false;
        e.verify_expert_graphs = false;
        e.verify_gdn_graphs = false;
        e.observer = null;
        e.captured = true;
        e.expert_up_n8 = false;
        e.expert_up_split_m = false;
        e.reuse_fp8_quant = false;
        e.expert_up_layout = .lane_major;
        e.attention_mode = .reference;
        e.bounded_hf_prefill = true;
        e.terminal = null;
        e.verify_rows = engine.most_verify;
        e.stream = .{ .d = &f.d, .handle = @ptrFromInt(1) };
        e.layers = &.{};
        e.staged = .{ .d = &f.d, .bytes = &f.staged };
        e.tables = .{ .d = &f.d, .ptr = @intFromPtr(&f.tables), .len = f.tables.len, .borrowed = true };
        e.x = @intFromPtr(&f.x);
        e.h = 4000;
        e.r = 5000;
        e.final_norm = 6000;
        e.embed_bits = 4;
        e.embed_w = 7000;
        e.embed_s = 8000;
        e.embed_b = 9000;
        e.geo = .{ .hid = 4, .nk = 1, .nv = 1, .qh = 1, .kvh = 1, .nl = 0 };
        e.fns.embed = .{ .d = &f.d, .handle = @ptrFromInt(1) };
        e.fns.norm = .{ .d = &f.d, .handle = @ptrFromInt(2) };
        e.fns.gdn_chain = .{ .d = &f.d, .handle = @ptrFromInt(3) };
        e.graphs = .empty;
        e.graph_nodes = 0;
        e.graph_launches = 0;
        e.expert_graph_nodes_baseline = 0;
        e.expert_graph_nodes_n8 = 0;
        e.expert_graph_nodes_split_m = 0;
        e.gdn_graph_nodes_baseline = 0;
        e.gdn_graph_nodes_candidate = 0;
        f.cache = .{ .owner = e, .state = .{ .d = &f.d, .ptr = @intFromPtr(&f.state), .len = f.state.len, .borrowed = true }, .kv = undefined, .log = undefined, .cap = 4096, .logits = &f.logits };
        Fake.scratch = &f.x;
    }
    fn deinit(f: *Fixture) void {
        var it = f.e.graphs.valueIterator();
        while (it.next()) |exec| exec.deinit();
        f.e.graphs.deinit(std.testing.allocator);
        cuda.copy_source.dropSpares(&f.d, cuda.copy_source.current());
    }
    fn forward(f: *Fixture, w: hidden.Window, ids: []const u32) !void {
        try f.e.forward(&.{.{ .cache = &f.cache, .rows = ids.len }}, ids, .{ .hidden = w });
    }
};

test "hidden forward copies all rows in multiple chunks and replay uses the current request destination" {
    var f: Fixture = undefined;
    f.init();
    defer f.deinit();
    var output = try f.e.hiddenRows(5);
    defer output.deinit();
    try std.testing.expectError(error.QwenHiddenIncomplete, output.completed());
    try f.forward(try output.window(0, 2), &.{ 0, 1 });
    try std.testing.expectEqual(@as(usize, 16), Fake.copy_bytes);
    try std.testing.expectEqual(@as(usize, 2), f.cache.len);
    try std.testing.expectError(error.QwenHiddenIncomplete, output.completed());
    Fake.pattern = 40;
    try f.forward(try output.window(2, 3), &.{ 1, 2, 3 });
    const record = try output.completed();
    const address = try record.device();
    const got: [*]const u8 = @ptrFromInt(address);
    for (got[0..16], 0..) |byte, i| try std.testing.expectEqual(@as(u8, @intCast(1 + i)), byte);
    for (got[16..40], 0..) |byte, i| try std.testing.expectEqual(@as(u8, @intCast(40 + i)), byte);
    for (got[40..256]) |byte| try std.testing.expectEqual(@as(u8, 0xa5), byte);
    try std.testing.expectEqual(@as(usize, 24), Fake.copy_bytes);
    try std.testing.expectEqual(@as(usize, 5), f.cache.len);
    try std.testing.expectEqual(@as(usize, 2), Fake.captures);
    try std.testing.expectEqual(@as(usize, 2), Fake.norms);
    try std.testing.expectEqual(@as(usize, 4), Fake.kernels);
    try std.testing.expectEqual(@as(usize, 2), f.e.graphs.count());
    try output.reset();
    try std.testing.expectError(error.StaleQwenHiddenView, record.device());
    var second = try f.e.hiddenRows(5);
    defer second.deinit();
    Fake.pattern = 70;
    try f.forward(try second.window(0, 2), &.{ 0, 1 });
    try std.testing.expectEqual(second.buffer.ptr, Fake.copy_destination);
    Fake.pattern = 90;
    try f.forward(try second.window(2, 3), &.{ 1, 2, 3 });
    try std.testing.expectEqual(second.buffer.ptr + 16, Fake.copy_destination);
    const next = try second.completed();
    try std.testing.expect(address != try next.device());
    const next_bytes: [*]const u8 = @ptrFromInt(try next.device());
    for (next_bytes[0..16], 0..) |byte, i| try std.testing.expectEqual(@as(u8, @intCast(70 + i)), byte);
    for (next_bytes[16..40], 0..) |byte, i| try std.testing.expectEqual(@as(u8, @intCast(90 + i)), byte);
    for (got[0..16], 0..) |byte, i| try std.testing.expectEqual(@as(u8, @intCast(1 + i)), byte);
    try std.testing.expectEqual(@as(usize, 2), Fake.captures);
    try std.testing.expectEqual(@as(usize, 4), Fake.replays);
    try std.testing.expectEqual(@as(usize, 2), Fake.norms);
}

test "hidden teardown drains completed-row consumers even after cancellation" {
    for ([_]bool{ false, true }) |cancel_before| for ([_]bool{ false, true }) |context_fallback| {
        var f: Fixture = undefined;
        f.init();
        defer f.deinit();
        var output = try f.e.hiddenRows(2);
        try f.forward(try output.window(0, 2), &.{ 0, 1 });
        _ = try (try output.completed()).device();
        Fake.consumer_pending = true;
        if (cancel_before) try output.cancel();
        const syncs_before = Fake.syncs;
        Fake.fail_syncs = if (context_fallback) 1 else 0;
        output.deinit();
        try std.testing.expect(!Fake.consumer_pending);
        try std.testing.expect(!Fake.freed_with_consumer);
        try std.testing.expectEqual(@as(usize, 1), Fake.frees);
        try std.testing.expectEqual(syncs_before + 1, Fake.syncs);
        try std.testing.expectEqual(@as(usize, @intFromBool(context_fallback)), Fake.context_syncs);
    };
}

test "stale window transaction helpers cannot advance a reset output" {
    var f: Fixture = undefined;
    f.init();
    defer f.deinit();
    var output = try f.e.hiddenRows(4);
    defer output.deinit();
    const stale = try output.window(0, 2);
    try f.forward(stale, &.{ 0, 1 });
    try output.reset();
    stale.begin(&f.cache);
    stale.commit();
    try std.testing.expect(output.state == .collecting);
    try std.testing.expectEqual(@as(usize, 0), output.filled);
    try std.testing.expectEqual(@as(usize, 0), f.cache.len);
    const current = try output.window(0, 2);
    current.commit();
    try std.testing.expectEqual(@as(usize, 0), output.filled);
    try f.forward(current, &.{ 0, 1 });
    stale.begin(&f.cache);
    stale.commit();
    try std.testing.expect(output.state == .collecting);
    try std.testing.expectEqual(@as(usize, 2), output.filled);
    try std.testing.expectError(error.QwenHiddenIncomplete, output.completed());
    try f.forward(try output.window(2, 2), &.{ 2, 3 });
    _ = try output.completed();
}

test "hidden bounds shapes profiles and devices refuse before CUDA or cache mutation" {
    var f: Fixture = undefined;
    f.init();
    defer f.deinit();
    try std.testing.expectError(error.InvalidQwenHiddenShape, f.e.hiddenRows(0));
    try std.testing.expectError(error.InvalidQwenHiddenShape, f.e.hiddenRows(4097));
    var largest = try f.e.hiddenRows(4096);
    try std.testing.expectEqual(@as(usize, 32768), largest.buffer.len);
    largest.deinit();
    const syncs_before = Fake.syncs;
    f.e.g.hidden = std.math.maxInt(usize);
    try std.testing.expectError(error.Overflow, f.e.hiddenRows(2));
    f.e.g.hidden = 4;
    var output = try f.e.hiddenRows(4);
    defer output.deinit();
    try std.testing.expectError(error.InvalidQwenHiddenShape, output.window(0, 0));
    try std.testing.expectError(error.InvalidQwenHiddenShape, output.window(std.math.maxInt(usize), 2));
    try std.testing.expectError(error.InvalidQwenHiddenShape, output.window(0, 5));
    try std.testing.expectError(error.QwenHiddenWriteOrder, output.window(1, 1));
    const w = try output.window(0, 2);
    try std.testing.expectError(error.InvalidQwenHiddenShape, f.forward(w, &.{0}));
    var wrong_rows = w;
    wrong_rows.rows = 5;
    try std.testing.expectError(error.InvalidQwenHiddenShape, f.forward(wrong_rows, &.{ 0, 1 }));
    output.width = 5;
    try std.testing.expectError(error.InvalidQwenHiddenShape, f.forward(w, &.{ 0, 1 }));
    output.width = 4;
    var other: engine.Engine = undefined;
    output.engine = &other;
    try std.testing.expectError(error.QwenHiddenDeviceMismatch, f.forward(w, &.{ 0, 1 }));
    output.engine = &f.e;
    const context = f.e.context;
    f.e.context = @ptrFromInt(9999);
    try std.testing.expectError(error.QwenHiddenDeviceMismatch, f.e.hiddenRows(4));
    try std.testing.expectError(error.QwenHiddenDeviceMismatch, f.forward(w, &.{ 0, 1 }));
    f.e.context = context;
    f.cache.owner = &other;
    try std.testing.expectError(error.QwenHiddenDeviceMismatch, f.forward(w, &.{ 0, 1 }));
    f.cache.owner = &f.e;
    f.cache.cap = 3;
    try std.testing.expectError(error.InvalidQwenHiddenCache, f.forward(w, &.{ 0, 1 }));
    f.cache.cap = 4096;
    f.cache.written = 1;
    try std.testing.expectError(error.InvalidQwenHiddenCache, f.forward(w, &.{ 0, 1 }));
    f.cache.written = 0;
    f.cache.last = .{ .start = 0, .rows = 1 };
    try std.testing.expectError(error.InvalidQwenHiddenCache, f.forward(w, &.{ 0, 1 }));
    f.cache.last = null;
    f.cache.delta = 1;
    try std.testing.expectError(error.InvalidQwenHiddenCache, f.forward(w, &.{ 0, 1 }));
    f.cache.delta = 0;
    try std.testing.expectError(error.InvalidQwenToken, f.forward(w, &.{ 0, 4 }));
    try std.testing.expectError(error.InvalidQwenHiddenShape, f.e.forward(&.{.{ .cache = &f.cache, .rows = 2, .parents = &.{ -1, 0 } }}, &.{ 0, 1 }, .{ .hidden = w }));
    const image: engine.Images = undefined;
    try std.testing.expectError(error.InvalidQwenHiddenShape, f.e.forward(&.{.{ .cache = &f.cache, .rows = 2, .images = &image }}, &.{ 0, 1 }, .{ .hidden = w }));
    try std.testing.expectError(error.InvalidQwenHiddenShape, f.e.forward(&.{ .{ .cache = &f.cache, .rows = 1 }, .{ .cache = &f.cache, .rows = 1 } }, &.{ 0, 1 }, .{ .hidden = w }));
    f.e.kv8 = true;
    try std.testing.expectError(error.UnsupportedQwenHiddenProfile, f.forward(w, &.{ 0, 1 }));
    f.e.kv8 = false;
    f.e.n_taps = 1;
    try std.testing.expectError(error.UnsupportedQwenHiddenProfile, f.forward(w, &.{ 0, 1 }));
    f.e.n_taps = 0;
    f.e.verify_terminal_graphs = true;
    try std.testing.expectError(error.UnsupportedQwenHiddenProfile, f.forward(w, &.{ 0, 1 }));
    f.e.verify_terminal_graphs = false;
    f.e.numerics = .hf_dense_bf16;
    try std.testing.expectError(error.UnsupportedQwenPlainExecution, f.forward(w, &.{ 0, 1 }));
    f.e.numerics = .hf_modelopt;
    f.e.bounded_hf_prefill = false;
    try std.testing.expectError(error.UnsupportedQwenWindow, f.forward(w, &.{ 0, 1 }));
    f.e.bounded_hf_prefill = true;
    try std.testing.expectError(error.QwenHiddenRequiresForward, f.e.prepare(&.{.{ .cache = &f.cache, .rows = 2 }}, &.{ 0, 1 }, .{ .hidden = w }));
    try std.testing.expectEqual(@as(usize, 0), Fake.uploads);
    try std.testing.expectEqual(@as(usize, 0), Fake.kernels);
    try std.testing.expectEqual(@as(usize, 0), Fake.replays);
    try std.testing.expectEqual(@as(usize, 0), Fake.copies);
    try std.testing.expectEqual(syncs_before, Fake.syncs);
    try std.testing.expectEqual(@as(usize, 0), f.cache.len);
    try std.testing.expect(!f.cache.hidden_invalid);
    try std.testing.expect(f.cache.last == null);
}

test "hidden failures invalidate partial records drain before release and require cache and output reset" {
    var f: Fixture = undefined;
    f.init();
    defer f.deinit();
    var output = try f.e.hiddenRows(4);
    defer output.deinit();
    const stale = try output.window(0, 2);
    try f.forward(stale, &.{ 0, 1 });
    try std.testing.expectError(error.QwenHiddenWriteOrder, f.forward(stale, &.{ 0, 1 }));
    try std.testing.expectError(error.QwenHiddenWriteOrder, output.window(0, 2));
    Fake.fail_copy = true;
    Fake.fail_syncs = 1;
    try std.testing.expectError(error.NotReady, f.forward(try output.window(2, 2), &.{ 2, 3 }));
    try std.testing.expectEqual(@as(usize, 1), Fake.context_syncs);
    try std.testing.expectEqual(@as(usize, 2), f.cache.len);
    try std.testing.expect(f.cache.hidden_invalid);
    try std.testing.expectError(error.QwenHiddenIncomplete, output.completed());
    try std.testing.expectError(error.QwenHiddenNotCollecting, output.window(2, 2));
    try std.testing.expectError(error.QwenHiddenNeedsReset, f.e.forward(&.{.{ .cache = &f.cache, .rows = 1 }}, &.{0}, .last));
    try std.testing.expectError(error.QwenHiddenNeedsReset, f.cache.settle(&f.e));
    try std.testing.expectError(error.QwenHiddenNeedsReset, f.cache.keep(&f.e, &.{0}));
    Fake.fail_copy = false;
    Fake.fail_fill = true;
    try std.testing.expectError(error.NotReady, output.reset());
    try std.testing.expect(f.cache.hidden_invalid);
    try std.testing.expectError(error.QwenHiddenNotCollecting, output.window(0, 2));
    Fake.fail_fill = false;
    try output.reset();
    try std.testing.expect(!f.cache.hidden_invalid);
    try std.testing.expectEqual(@as(usize, 0), f.cache.len);
    try std.testing.expectError(error.StaleQwenHiddenView, f.forward(stale, &.{ 0, 1 }));
    Fake.fail_syncs = 1;
    try std.testing.expectError(error.NotReady, f.forward(try output.window(0, 2), &.{ 0, 1 }));
    try std.testing.expect(f.cache.hidden_invalid);
    try std.testing.expectError(error.QwenHiddenIncomplete, output.completed());
    try output.reset();
    try f.forward(try output.window(0, 2), &.{ 0, 1 });
    try f.forward(try output.window(2, 2), &.{ 2, 3 });
    const record = try output.completed();
    _ = try record.device();
    try output.cancel();
    try std.testing.expectError(error.QwenHiddenIncomplete, record.device());
    try std.testing.expect(f.cache.hidden_invalid);
}

test "MLX hidden prompts use the same final norm without vocabulary weights" {
    var f: Fixture = undefined;
    f.init();
    defer f.deinit();
    f.e.numerics = .mlx_sanitized;
    var output = try f.e.hiddenRows(5);
    defer output.deinit();
    try f.forward(try output.window(0, 5), &.{ 0, 1, 2, 3, 0 });
    const bytes: [*]const u8 = @ptrFromInt(try (try output.completed()).device());
    for (bytes[0..40], 0..) |byte, i| try std.testing.expectEqual(@as(u8, @intCast(1 + i)), byte);
    try std.testing.expectEqual(@as(usize, 40), Fake.copy_bytes);
    try std.testing.expectEqual(@as(usize, 1), Fake.norms);
    try std.testing.expectEqual(@as(usize, 2), Fake.kernels);
    try std.testing.expectEqual(@as(usize, 0), Fake.captures);
    try std.testing.expectEqual(@as(usize, 0), Fake.replays);
}

test "unbounded HF hidden output keeps single-row admission and disables installed greedy terminal" {
    var f: Fixture = undefined;
    f.init();
    defer f.deinit();
    f.e.bounded_hf_prefill = false;
    f.e.terminal = .{ .module = undefined, .kernels = undefined };
    var output = try f.e.hiddenRows(2);
    defer output.deinit();
    try f.forward(try output.window(0, 1), &.{0});
    try f.forward(try output.window(1, 1), &.{1});
    _ = try (try output.completed()).device();
    try std.testing.expectEqual(@as(usize, 1), Fake.norms);
    try std.testing.expectEqual(@as(usize, 2), Fake.kernels);
    try std.testing.expectEqual(@as(usize, 1), Fake.captures);
    try std.testing.expectEqual(@as(usize, 2), Fake.replays);
    try std.testing.expect(f.cache.terminal_greedy == null);
}

test "hidden upload and replay failures do not commit rows" {
    var f: Fixture = undefined;
    f.init();
    defer f.deinit();
    var output = try f.e.hiddenRows(2);
    defer output.deinit();
    Fake.fail_upload = true;
    try std.testing.expectError(error.NotReady, f.forward(try output.window(0, 2), &.{ 0, 1 }));
    try std.testing.expect(f.cache.hidden_invalid);
    try std.testing.expectEqual(@as(usize, 0), f.cache.len);
    try std.testing.expectEqual(@as(usize, 0), Fake.replays);
    Fake.fail_upload = false;
    try output.reset();
    Fake.fail_launch = true;
    try std.testing.expectError(error.NotReady, f.forward(try output.window(0, 2), &.{ 0, 1 }));
    try std.testing.expect(f.cache.hidden_invalid);
    try std.testing.expectEqual(@as(usize, 0), f.cache.len);
    try std.testing.expectEqual(@as(usize, 0), Fake.copies);
    try std.testing.expectError(error.QwenHiddenIncomplete, output.completed());
    Fake.fail_launch = false;
}

test "hidden teardown binds its allocation context and retains storage if binding fails" {
    var f: Fixture = undefined;
    f.init();
    defer f.deinit();
    var output = try f.e.hiddenRows(2);
    cuda.copy_source.setCurrent(@ptrFromInt(9999));
    Fake.fail_bind = true;
    output.deinit();
    try std.testing.expectEqual(@as(usize, 0), Fake.frees);
    Fake.fail_bind = false;
    output.deinit();
    try std.testing.expectEqual(@as(usize, 1), Fake.frees);
    try std.testing.expectEqual(f.e.context, Fake.bound);
    try std.testing.expectEqual(f.e.context, cuda.copy_source.current());
}

test "resetting or releasing an old output cannot poison a newer cache epoch" {
    var f: Fixture = undefined;
    f.init();
    defer f.deinit();
    var first = try f.e.hiddenRows(2);
    try f.forward(try first.window(0, 2), &.{ 0, 1 });
    const old = try first.completed();
    try f.cache.reset(&f.e);
    var next = try f.e.hiddenRows(2);
    defer next.deinit();
    try f.forward(try next.window(0, 2), &.{ 0, 1 });
    try std.testing.expectError(error.QwenHiddenNeedsReset, old.device());
    try first.reset();
    first.deinit();
    try std.testing.expect(!f.cache.hidden_invalid);
    _ = try (try next.completed()).device();
}
