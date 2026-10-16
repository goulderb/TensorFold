//! Records actual native API calls. No library, context, GPU or model is opened;
//! no numerical embeddings are fabricated or asserted by these mocks.
const std = @import("std");
const e = @import("cuda_vision_engine.zig");
const t = std.testing;
const cuda = e.cuda;
const lt = cuda.cublaslt;
const C = e.conv.cudnn.c;
const CM = e.conv.cudnn.test_support.Mock;
const SM = e.attention.sdpa.test_support.Mock;
const Kind = enum { cast, rms, qkv, residual, swiglu, ln, conv, bias, sdpa, linear };
const Trace = struct {
    const Record = struct { kind: Kind, x: u64 = 0, w: u64 = 0, out: u64 = 0 };
    var records: [1024]Record = undefined;
    var count: usize = 0;
    fn push(r: Record) void {
        records[count] = r;
        count += 1;
    }
};
const Driver = struct {
    var modules: usize = 0;
    var allocations: usize = 0;
    var lookups: usize = 0;
    var launches: usize = 0;
    var syncs: usize = 0;
    var copies: usize = 0;
    var init_calls: usize = 0;
    var sync_trace: usize = 0;
    var fail_launch: usize = 0;
    var fail_sync: usize = 0;
    var fail_copy: usize = 0;
    fn init(_: c_uint) callconv(.c) cuda.abi.Result {
        init_calls += 1;
        return cuda.abi.error_not_ready;
    }
    fn load(out: *cuda.abi.Module, _: ?*const anyopaque) callconv(.c) cuda.abi.Result {
        modules += 1;
        out.* = @ptrFromInt(0x1000);
        return 0;
    }
    fn unload(_: cuda.abi.Module) callconv(.c) cuda.abi.Result {
        modules -= 1;
        return 0;
    }
    fn function(out: *cuda.abi.Function, _: cuda.abi.Module, name: [*:0]const u8) callconv(.c) cuda.abi.Result {
        const i = lookups % 6;
        std.debug.assert(std.mem.eql(u8, std.mem.span(name), e.ops.symbols[i]));
        lookups += 1;
        out.* = @ptrFromInt(0x8000 + i * 16);
        return 0;
    }
    fn alloc(out: *u64, _: usize) callconv(.c) cuda.abi.Result {
        allocations += 1;
        out.* = 0x400000000;
        return 0;
    }
    fn free(_: u64) callconv(.c) cuda.abi.Result {
        allocations -= 1;
        return 0;
    }
    fn copy(_: u64, _: ?*const anyopaque, _: usize) callconv(.c) cuda.abi.Result {
        copies += 1;
        return if (copies == fail_copy) cuda.abi.error_not_ready else 0;
    }
    fn sync(_: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        syncs += 1;
        sync_trace = Trace.count;
        return if (syncs == fail_sync) cuda.abi.error_not_ready else 0;
    }
    fn launch(f: cuda.abi.Function, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: cuda.abi.Stream, args: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
        launches += 1;
        const op = (@intFromPtr(f) - 0x8000) / 16;
        const kind = [_]Kind{ .cast, .rms, .qkv, .residual, .swiglu, .ln };
        const out_index = [_]usize{ 1, 2, 5, 2, 2, 3 };
        Trace.push(.{ .kind = kind[op], .x = @as(*align(1) const u64, @ptrCast(args.?[0].?)).*, .out = @as(*align(1) const u64, @ptrCast(args.?[out_index[op]].?)).* });
        return if (launches == fail_launch) cuda.abi.error_not_ready else 0;
    }
    fn value() cuda.Driver {
        var api: cuda.abi.Api = undefined;
        api.cuInit = init;
        api.cuModuleLoadData = load;
        api.cuModuleUnload = unload;
        api.cuModuleGetFunction = function;
        api.cuMemAlloc_v2 = alloc;
        api.cuMemFree_v2 = free;
        api.cuMemcpyHtoD_v2 = copy;
        api.cuStreamSynchronize = sync;
        api.cuLaunchKernel = launch;
        return .{ .lib = undefined, .api = api };
    }
};
const LT = struct {
    var next: usize = 1;
    var live: usize = 0;
    var heuristics: usize = 0;
    var matmuls: usize = 0;
    var fail_init: usize = 0;
    var fail_matmul: usize = 0;
    var biased: usize = 0;
    var owned: [4096]bool = @splat(false);
    var rows: [4096]u64 = undefined;
    var cols: [4096]u64 = undefined;
    fn create(comptime T: type, out: *T) c_int {
        const id = next;
        next += 1;
        owned[id] = true;
        live += 1;
        out.* = @ptrFromInt(id);
        return 0;
    }
    fn destroy(ptr: anytype) c_int {
        const id = @intFromPtr(ptr);
        std.debug.assert(owned[id]);
        owned[id] = false;
        live -= 1;
        return 0;
    }
    fn handle(out: *lt.Handle) callconv(.c) c_int {
        return create(lt.Handle, out);
    }
    fn unhandle(ptr: lt.Handle) callconv(.c) c_int {
        return destroy(ptr);
    }
    fn version() callconv(.c) usize {
        return 133073;
    }
    fn name(_: c_int) callconv(.c) ?[*:0]const u8 {
        return "mock failure";
    }
    fn desc(out: *lt.MatmulDesc, compute: c_int, dtype: lt.DataType) callconv(.c) c_int {
        std.debug.assert(compute == 68 and dtype == .f32);
        return create(lt.MatmulDesc, out);
    }
    fn undesc(ptr: lt.MatmulDesc) callconv(.c) c_int {
        return destroy(ptr);
    }
    fn descAttr(_: lt.MatmulDesc, attr: c_int, _: ?*const anyopaque, _: usize) callconv(.c) c_int {
        if (attr == lt.desc_epilogue) biased += 1;
        return 0;
    }
    fn layout(out: *lt.Layout, dtype: lt.DataType, r: u64, c: u64, ld: i64) callconv(.c) c_int {
        std.debug.assert(dtype == .bf16 and ld == r);
        const rc = create(lt.Layout, out);
        rows[@intFromPtr(out.*)] = r;
        cols[@intFromPtr(out.*)] = c;
        return rc;
    }
    fn unlayout(ptr: lt.Layout) callconv(.c) c_int {
        return destroy(ptr);
    }
    fn layoutAttr(_: lt.Layout, _: c_int, _: ?*const anyopaque, _: usize) callconv(.c) c_int {
        return 0;
    }
    fn pref(out: *lt.Preference) callconv(.c) c_int {
        return create(lt.Preference, out);
    }
    fn unpref(ptr: lt.Preference) callconv(.c) c_int {
        return destroy(ptr);
    }
    fn prefAttr(_: lt.Preference, _: c_int, _: ?*const anyopaque, _: usize) callconv(.c) c_int {
        return 0;
    }
    fn heuristic(_: lt.Handle, _: lt.MatmulDesc, _: lt.Layout, _: lt.Layout, _: lt.Layout, _: lt.Layout, _: lt.Preference, _: c_int, out: [*]lt.Heuristic, count: *c_int) callconv(.c) c_int {
        heuristics += 1;
        if (heuristics == fail_init) return 7;
        count.* = 1;
        out[0] = .{ .algo = .{ .data = @splat(0) }, .workspace_size = 512, .state = 0, .waves_count = 1, .reserved = @splat(0) };
        return 0;
    }
    fn algoCheck(_: lt.Handle, _: lt.MatmulDesc, _: lt.Layout, _: lt.Layout, _: lt.Layout, _: lt.Layout, _: *const lt.Algo, _: *lt.Heuristic) callconv(.c) c_int {
        unreachable;
    }
    fn matmul(_: lt.Handle, _: lt.MatmulDesc, _: ?*const anyopaque, w: u64, _: lt.Layout, x: u64, _: lt.Layout, _: ?*const anyopaque, _: u64, _: lt.Layout, out: u64, _: lt.Layout, _: ?*const lt.Algo, _: u64, n: usize, _: cuda.abi.Stream) callconv(.c) c_int {
        std.debug.assert(n == 512);
        matmuls += 1;
        Trace.push(.{ .kind = .linear, .x = x, .w = w, .out = out });
        return if (matmuls == fail_matmul) 7 else 0;
    }
    const api: lt.Api = .{ .cublasLtCreate = handle, .cublasLtDestroy = unhandle, .cublasLtGetVersion = version, .cublasLtGetStatusName = name, .cublasLtMatmulDescCreate = desc, .cublasLtMatmulDescDestroy = undesc, .cublasLtMatmulDescSetAttribute = descAttr, .cublasLtMatrixLayoutCreate = layout, .cublasLtMatrixLayoutDestroy = unlayout, .cublasLtMatrixLayoutSetAttribute = layoutAttr, .cublasLtMatmulPreferenceCreate = pref, .cublasLtMatmulPreferenceDestroy = unpref, .cublasLtMatmulPreferenceSetAttribute = prefAttr, .cublasLtMatmulAlgoGetHeuristic = heuristic, .cublasLtMatmulAlgoCheck = algoCheck, .cublasLtMatmul = matmul };
};
fn convolution(h: C.cudnnHandle_t, a: ?*const anyopaque, xd: C.cudnnTensorDescriptor_t, x: ?*const anyopaque, wd: C.cudnnFilterDescriptor_t, w: ?*const anyopaque, cd: C.cudnnConvolutionDescriptor_t, algo: C.cudnnConvolutionFwdAlgo_t, s: ?*anyopaque, n: usize, b: ?*const anyopaque, yd: C.cudnnTensorDescriptor_t, y: ?*anyopaque) callconv(.c) C.cudnnStatus_t {
    Trace.push(.{ .kind = .conv, .x = @intFromPtr(x), .w = @intFromPtr(w), .out = @intFromPtr(y) });
    return CM.api.cudnnConvolutionForward(h, a, xd, x, wd, w, cd, algo, s, n, b, yd, y);
}
fn bias(h: C.cudnnHandle_t, a: ?*const anyopaque, bd: C.cudnnTensorDescriptor_t, b: ?*const anyopaque, beta: ?*const anyopaque, yd: C.cudnnTensorDescriptor_t, y: ?*anyopaque) callconv(.c) C.cudnnStatus_t {
    Trace.push(.{ .kind = .bias });
    return CM.api.cudnnAddTensor(h, a, bd, b, beta, yd, y);
}
fn sdpa(h: C.cudnnHandle_t, p: C.cudnnBackendDescriptor_t, v: C.cudnnBackendDescriptor_t) callconv(.c) C.cudnnStatus_t {
    Trace.push(.{ .kind = .sdpa });
    return SM.api.cudnnBackendExecute(h, p, v);
}
var conv_api: e.conv.cudnn.Api = undefined;
var sdpa_api: e.attention.sdpa.Api = undefined;
var library: lt.Library = .{ .lib = undefined, .api = LT.api };
const image: [8]u8 align(8) = @splat(0);
fn reset() void {
    std.debug.assert(Driver.modules == 0 and Driver.allocations == 0 and LT.live == 0);
    CM.reset();
    SM.reset();
    conv_api = CM.api;
    conv_api.cudnnConvolutionForward = convolution;
    conv_api.cudnnAddTensor = bias;
    sdpa_api = SM.api;
    sdpa_api.cudnnBackendExecute = sdpa;
    Driver.lookups = 0;
    Driver.launches = 0;
    Driver.syncs = 0;
    Driver.copies = 0;
    Driver.init_calls = 0;
    Driver.fail_launch = 0;
    Driver.fail_sync = 0;
    Driver.fail_copy = 0;
    LT.next = 1;
    LT.owned = @splat(false);
    LT.heuristics = 0;
    LT.matmuls = 0;
    LT.fail_init = 0;
    LT.fail_matmul = 0;
    LT.biased = 0;
    Trace.count = 0;
}
fn runtime() e.Runtime {
    return .{ .convolution = &conv_api, .sdpa = &sdpa_api, .linear = &library, .image = &image, .package = .target_qualification };
}
fn weights(d: *const cuda.Driver) !e.memory.DeviceWeights {
    const c = try @import("cuda_vision_config.zig").parse(t.allocator, @embedFile("fixtures_cuda_config_exl3.json"));
    const layout = try e.memory.WeightLayout.init(t.allocator, c, 2 << 30);
    // Synthetic borrowed device address, never read as host/model data.
    return .{ .layout = layout, .pool = .{ .d = d, .ptr = 0x100000000, .len = layout.bytes, .borrowed = true } };
}
const geometry: e.memory.Geometry = .{ .groups = 2, .gh = 2, .gw = 2 };
const options: e.Options = .{ .workspace_limit = 1024, .max_activation_bytes = 1 << 24 };
fn clean() !void {
    try t.expectEqual(@as(usize, 0), LT.live);
    try t.expectEqual(@as(usize, 0), CM.live);
    try t.expectEqual(@as(usize, 0), SM.live);
    try t.expectEqual(@as(usize, 0), Driver.modules);
    try t.expectEqual(@as(usize, 0), Driver.allocations);
    try t.expectEqual(@as(usize, 0), Driver.init_calls);
}

test "vision encoder actual constructor and full native source-order dispatch publish only after synchronization" {
    reset();
    var d = Driver.value();
    var w = try weights(&d);
    defer w.deinit();
    const stream: cuda.Stream = .{ .d = &d, .handle = @ptrFromInt(0x44440) };
    var engine = try e.Engine.init(t.allocator, &w, stream, geometry, runtime(), options);
    try t.expectEqual(@as(usize, 124), LT.heuristics);
    try t.expectEqual(@as(usize, 120), LT.biased);
    try t.expect(!engine.published);
    const pixels = try t.allocator.alloc(f32, 8 * 1176);
    defer t.allocator.free(pixels);
    @memset(pixels, 0.5);
    Trace.count = 0;
    const output = try engine.encodePixels(pixels);
    try t.expect(output.borrowed and engine.published);
    try t.expectEqual(@as(usize, 2 * 4096 * 2), output.len);
    try t.expectEqual(@as(usize, 300), Trace.count);
    try t.expectEqual(Trace.count, Driver.sync_trace);
    const block = [_]Kind{ .rms, .linear, .qkv, .sdpa, .linear, .residual, .rms, .linear, .linear, .swiglu, .linear, .residual };
    try t.expectEqual(Kind.cast, Trace.records[0].kind);
    try t.expectEqual(Kind.conv, Trace.records[1].kind);
    try t.expectEqual(Kind.bias, Trace.records[2].kind);
    for (0..24) |l| {
        for (block, 0..) |kind, j| try t.expectEqual(kind, Trace.records[3 + l * 12 + j].kind);
        try t.expectEqual((try w.block(@intCast(l), .qkv_weight)).ptr, Trace.records[4 + l * 12].w);
        try t.expectEqual((try w.block(@intCast(l), .down_weight)).ptr, Trace.records[13 + l * 12].w);
    }
    const tail = [_]Kind{ .rms, .conv, .bias, .linear, .ln, .linear, .linear, .swiglu, .linear };
    for (tail, 0..) |kind, j| try t.expectEqual(kind, Trace.records[291 + j].kind);
    try t.expectEqual((try w.root(.projection)).ptr, Trace.records[294].w);
    try t.expectEqual((try w.root(.down)).ptr, Trace.records[299].w);
    try t.expectEqual((try engine.activations.buffer(.q)).ptr, @as(u64, @intCast(SM.last_pointers[0])));
    try t.expectEqual(stream.handle, @as(cuda.abi.Stream, @ptrCast(SM.last_stream)));
    try t.expectEqual(@as(u64, 1024), LT.rows[@intFromPtr(engine.linears[0].a)]);
    try t.expectEqual(@as(u64, 3072), LT.cols[@intFromPtr(engine.linears[0].a)]);
    try t.expectEqual(@as(u64, 8), LT.cols[@intFromPtr(engine.linears[0].c)]);
    try t.expectEqual(@as(u64, 4096), LT.rows[@intFromPtr(engine.linears[121].a)]);
    try t.expectEqual(@as(u64, 10240), LT.cols[@intFromPtr(engine.linears[121].a)]);
    try t.expectEqual(@as(u64, 2), LT.cols[@intFromPtr(engine.linears[121].c)]);
    const Prepared = @import("cuda_vision_preprocess.zig").Prepared;
    const first: Prepared = .{ .canvas = .{ .height = 28, .width = 28 }, .content = .{ .height = 28, .width = 28 }, .grid = .{ 1, 2, 2 }, .patch_width = 1176, .pixels = pixels[0 .. 4 * 1176] };
    var second = first;
    second.pixels = pixels[4 * 1176 ..];
    Trace.count = 0;
    const next = try engine.encodePrepared(&.{ first, second });
    try t.expectEqual(output.ptr, next.ptr);
    try t.expect(engine.published);
    try t.expectEqual(@as(usize, 300), Trace.count);
    try engine.deinit();
    try clean();
}
test "vision encoder missing unqualified or mismatched runtime refuses before native construction" {
    reset();
    var d = Driver.value();
    var w = try weights(&d);
    defer w.deinit();
    const s: cuda.Stream = .{ .d = &d, .handle = null };
    try t.expectError(error.MissingVisionRuntime, e.Engine.init(t.allocator, &w, s, geometry, null, options));
    var r = runtime();
    r.package = .unqualified;
    try t.expectError(error.UnqualifiedVisionRuntime, e.Engine.init(t.allocator, &w, s, geometry, r, options));
    var other = Driver.value();
    try t.expectError(error.InvalidOperatorStream, e.Engine.init(t.allocator, &w, .{ .d = &other, .handle = null }, geometry, runtime(), options));
    CM.version = 92300;
    try t.expectError(error.VersionMismatch, e.Engine.init(t.allocator, &w, s, geometry, runtime(), options));
    try clean();
}
test "vision encoder every linear construction failure unwinds all initialized owners" {
    var d = Driver.value();
    var w = try weights(&d);
    defer w.deinit();
    for (1..125) |i| {
        reset();
        LT.fail_init = i;
        try t.expectError(error.CublasFailed, e.Engine.init(t.allocator, &w, .{ .d = &d, .handle = null }, geometry, runtime(), options));
        try clean();
    }
}
test "vision encoder errors poison publication and failed synchronization retains all resources" {
    reset();
    var d = Driver.value();
    var w = try weights(&d);
    defer w.deinit();
    var engine = try e.Engine.init(t.allocator, &w, .{ .d = &d, .handle = @ptrFromInt(0x44440) }, geometry, runtime(), options);
    const pixels = try t.allocator.alloc(f32, 8 * 1176);
    defer t.allocator.free(pixels);
    @memset(pixels, 0);
    Driver.fail_launch = 5;
    try t.expectError(error.NotReady, engine.encodePixels(pixels));
    try t.expect(engine.poisoned and !engine.published);
    try t.expectError(error.VisionEncoderPoisoned, engine.encodePixels(pixels));
    Driver.fail_sync = Driver.syncs + 1;
    const held = LT.live;
    try t.expectError(error.NotReady, engine.deinit());
    try t.expectEqual(held, LT.live);
    try t.expectEqual(@as(usize, 1), Driver.allocations);
    Driver.fail_sync = 0;
    try engine.deinit();
    try clean();
}
test "vision encoder late synchronization and native linear or SDPA failures publish nothing" {
    var d = Driver.value();
    var w = try weights(&d);
    defer w.deinit();
    const pixels = try t.allocator.alloc(f32, 8 * 1176);
    defer t.allocator.free(pixels);
    @memset(pixels, 0.5);
    for (0..3) |fault| {
        reset();
        var engine = try e.Engine.init(t.allocator, &w, .{ .d = &d, .handle = null }, geometry, runtime(), options);
        switch (fault) {
            0 => {
                Driver.fail_sync = Driver.syncs + 3;
                try t.expectError(error.NotReady, engine.encodePixels(pixels));
                try t.expectEqual(@as(usize, 300), Trace.count);
            },
            1 => {
                LT.fail_matmul = 124;
                try t.expectError(error.CublasFailed, engine.encodePixels(pixels));
            },
            2 => {
                SM.runtime_need = 512;
                try t.expectError(error.WorkspaceRequirementChanged, engine.encodePixels(pixels));
            },
            else => unreachable,
        }
        try t.expect(engine.poisoned and !engine.published);
        Driver.fail_sync = 0;
        try engine.deinit();
        try clean();
    }
}
test "vision encoder allocation failures unwind the complete native constructor" {
    var d = Driver.value();
    var w = try weights(&d);
    defer w.deinit();
    const Trial = struct {
        fn run(a: std.mem.Allocator, weight: *const e.memory.DeviceWeights, driver: *const cuda.Driver) !void {
            reset();
            defer {
                std.debug.assert(LT.live == 0 and CM.live == 0 and SM.live == 0 and Driver.modules == 0 and Driver.allocations == 0);
            }
            var engine = try e.Engine.init(a, weight, .{ .d = driver, .handle = null }, geometry, runtime(), options);
            try engine.deinit();
        }
    };
    try t.checkAllAllocationFailures(t.allocator, Trial.run, .{ &w, &d });
    try clean();
}
test "vision encoder validation and busy refusal publish nothing and valid input remains retryable" {
    reset();
    var d = Driver.value();
    var w = try weights(&d);
    defer w.deinit();
    var engine = try e.Engine.init(t.allocator, &w, .{ .d = &d, .handle = null }, geometry, runtime(), options);
    try t.expectError(error.InvalidVisionGrid, engine.encodePixels(&.{0}));
    try t.expect(!engine.published and !engine.poisoned);
    try t.expect(engine.mutex.tryLock());
    try t.expectError(error.VisionBusy, engine.encodePixels(&.{0}));
    engine.mutex.unlock();
    try engine.deinit();
    try clean();
}
