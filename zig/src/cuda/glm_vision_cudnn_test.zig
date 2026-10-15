const std = @import("std");
const d = @import("glm_vision_cudnn.zig");
const c = d.c;
const testing = std.testing;
const cc = @typeInfo(@TypeOf(c.cudnnCreate)).@"fn".attrs.@"callconv";
const S = d.Status;

pub const Mock = struct {
    pub var calls: usize = 0;
    pub var fail_at: usize = 0;
    pub var null_at: usize = 0;
    pub var live: usize = 0;
    var next: usize = 1;
    var owned: [64]bool = @splat(false);
    pub var version: usize = c.CUDNN_VERSION;
    pub var need: usize = 256;
    pub var count: c_int = 1;
    pub var candidates: [8]c.cudnnConvolutionFwdAlgoPerf_t = undefined;
    pub var forwarded: bool = false;
    pub var biased: bool = false;
    pub var last_stream: c.cudaStream_t = null;
    pub var tensor_dims: [64][5]c_int = undefined;
    pub var tensor_strides: [64][5]c_int = undefined;
    pub var filter_dims: [5]c_int = undefined;
    pub var conv_stride: [3]c_int = undefined;
    pub var current_math: c.cudnnMathType_t = 0;
    pub var destroy_order: [64]usize = undefined;
    pub var destroyed: usize = 0;
    pub fn reset() void {
        calls = 0;
        fail_at = 0;
        null_at = 0;
        live = 0;
        next = 1;
        owned = @splat(false);
        version = c.CUDNN_VERSION;
        need = 256;
        count = 1;
        forwarded = false;
        biased = false;
        last_stream = null;
        current_math = 0;
        destroyed = 0;
        candidates = @splat(.{ .algo = c.CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM, .status = c.CUDNN_STATUS_SUCCESS, .time = 1, .memory = 256, .determinism = c.CUDNN_NON_DETERMINISTIC, .mathType = c.CUDNN_TENSOR_OP_MATH, .reserved = @splat(0) });
    }
    fn tick() S {
        calls += 1;
        return if (calls == fail_at) c.CUDNN_STATUS_INTERNAL_ERROR else c.CUDNN_STATUS_SUCCESS;
    }
    fn make(comptime T: type, out: [*c]T) S {
        const status = tick();
        if (status != 0) return status;
        if (calls == null_at) {
            out[0] = null;
            return 0;
        }
        const id = next;
        next += 1;
        std.debug.assert(id < owned.len and !owned[id]);
        owned[id] = true;
        live += 1;
        out[0] = @ptrFromInt(id * 16);
        return 0;
    }
    fn drop(value: anytype) S {
        const id = @intFromPtr(value) / 16;
        std.debug.assert(id > 0 and id < owned.len and owned[id]);
        owned[id] = false;
        live -= 1;
        destroy_order[destroyed] = id;
        destroyed += 1;
        return 0;
    }
    fn getVersion() callconv(cc) usize {
        return version;
    }
    fn create(out: [*c]c.cudnnHandle_t) callconv(cc) S {
        return make(c.cudnnHandle_t, out);
    }
    fn destroy(h: c.cudnnHandle_t) callconv(cc) S {
        return drop(h);
    }
    fn setStream(_: c.cudnnHandle_t, stream: c.cudaStream_t) callconv(cc) S {
        last_stream = stream;
        return tick();
    }
    fn createTensor(out: [*c]c.cudnnTensorDescriptor_t) callconv(cc) S {
        return make(c.cudnnTensorDescriptor_t, out);
    }
    fn destroyTensor(t: c.cudnnTensorDescriptor_t) callconv(cc) S {
        return drop(t);
    }
    fn setTensor(t: c.cudnnTensorDescriptor_t, dtype: c.cudnnDataType_t, rank: c_int, dims: [*c]const c_int, strides: [*c]const c_int) callconv(cc) S {
        std.debug.assert(dtype == c.CUDNN_DATA_BFLOAT16 and rank >= 4 and rank <= 5);
        const id = @intFromPtr(t) / 16;
        @memcpy(tensor_dims[id][0..@intCast(rank)], dims[0..@intCast(rank)]);
        @memcpy(tensor_strides[id][0..@intCast(rank)], strides[0..@intCast(rank)]);
        return tick();
    }
    fn createFilter(out: [*c]c.cudnnFilterDescriptor_t) callconv(cc) S {
        return make(c.cudnnFilterDescriptor_t, out);
    }
    fn destroyFilter(t: c.cudnnFilterDescriptor_t) callconv(cc) S {
        return drop(t);
    }
    fn setFilter(_: c.cudnnFilterDescriptor_t, dtype: c.cudnnDataType_t, format: c.cudnnTensorFormat_t, rank: c_int, dims: [*c]const c_int) callconv(cc) S {
        std.debug.assert(dtype == c.CUDNN_DATA_BFLOAT16 and format == c.CUDNN_TENSOR_NCHW);
        @memcpy(filter_dims[0..@intCast(rank)], dims[0..@intCast(rank)]);
        return tick();
    }
    fn createConv(out: [*c]c.cudnnConvolutionDescriptor_t) callconv(cc) S {
        return make(c.cudnnConvolutionDescriptor_t, out);
    }
    fn destroyConv(t: c.cudnnConvolutionDescriptor_t) callconv(cc) S {
        return drop(t);
    }
    fn setConv(_: c.cudnnConvolutionDescriptor_t, rank: c_int, pad: [*c]const c_int, stride: [*c]const c_int, dilation: [*c]const c_int, mode: c.cudnnConvolutionMode_t, compute: c.cudnnDataType_t) callconv(cc) S {
        std.debug.assert(mode == c.CUDNN_CROSS_CORRELATION and compute == c.CUDNN_DATA_FLOAT);
        for (0..@intCast(rank)) |i| std.debug.assert(pad[i] == 0 and dilation[i] == 1);
        @memcpy(conv_stride[0..@intCast(rank)], stride[0..@intCast(rank)]);
        return tick();
    }
    fn setGroups(_: c.cudnnConvolutionDescriptor_t, groups: c_int) callconv(cc) S {
        std.debug.assert(groups == 1);
        return tick();
    }
    fn setMath(_: c.cudnnConvolutionDescriptor_t, math: c.cudnnMathType_t) callconv(cc) S {
        current_math = math;
        return tick();
    }
    fn algorithms(_: c.cudnnHandle_t, _: c.cudnnTensorDescriptor_t, _: c.cudnnFilterDescriptor_t, _: c.cudnnConvolutionDescriptor_t, _: c.cudnnTensorDescriptor_t, requested: c_int, returned: [*c]c_int, results: [*c]c.cudnnConvolutionFwdAlgoPerf_t) callconv(cc) S {
        std.debug.assert(requested == 8);
        returned[0] = count;
        if (count >= 0 and count <= 8) @memcpy(results[0..@intCast(count)], candidates[0..@intCast(count)]);
        return tick();
    }
    fn workspace(_: c.cudnnHandle_t, _: c.cudnnTensorDescriptor_t, _: c.cudnnFilterDescriptor_t, _: c.cudnnConvolutionDescriptor_t, _: c.cudnnTensorDescriptor_t, _: c.cudnnConvolutionFwdAlgo_t, out: [*c]usize) callconv(cc) S {
        out[0] = need;
        return tick();
    }
    fn forward(_: c.cudnnHandle_t, alpha: ?*const anyopaque, _: c.cudnnTensorDescriptor_t, x: ?*const anyopaque, _: c.cudnnFilterDescriptor_t, w: ?*const anyopaque, _: c.cudnnConvolutionDescriptor_t, _: c.cudnnConvolutionFwdAlgo_t, scratch: ?*anyopaque, length: usize, beta: ?*const anyopaque, _: c.cudnnTensorDescriptor_t, y: ?*anyopaque) callconv(cc) S {
        std.debug.assert(@as(*const f32, @ptrCast(@alignCast(alpha))).* == 1 and @as(*const f32, @ptrCast(@alignCast(beta))).* == 0);
        std.debug.assert(x != null and w != null and y != null and length == need and (need == 0 or scratch != null));
        const status = tick();
        if (status == 0) forwarded = true;
        return status;
    }
    fn add(_: c.cudnnHandle_t, alpha: ?*const anyopaque, _: c.cudnnTensorDescriptor_t, bias: ?*const anyopaque, beta: ?*const anyopaque, _: c.cudnnTensorDescriptor_t, y: ?*anyopaque) callconv(cc) S {
        std.debug.assert(forwarded and bias != null and y != null);
        std.debug.assert(@as(*const f32, @ptrCast(@alignCast(alpha))).* == 1 and @as(*const f32, @ptrCast(@alignCast(beta))).* == 1);
        const status = tick();
        if (status == 0) biased = true;
        return status;
    }
    pub const api: d.Api = .{
        .cudnnGetVersion = getVersion,
        .cudnnCreate = create,
        .cudnnDestroy = destroy,
        .cudnnSetStream = setStream,
        .cudnnCreateTensorDescriptor = createTensor,
        .cudnnSetTensorNdDescriptor = setTensor,
        .cudnnDestroyTensorDescriptor = destroyTensor,
        .cudnnCreateFilterDescriptor = createFilter,
        .cudnnSetFilterNdDescriptor = setFilter,
        .cudnnDestroyFilterDescriptor = destroyFilter,
        .cudnnCreateConvolutionDescriptor = createConv,
        .cudnnSetConvolutionNdDescriptor = setConv,
        .cudnnSetConvolutionMathType = setMath,
        .cudnnSetConvolutionGroupCount = setGroups,
        .cudnnDestroyConvolutionDescriptor = destroyConv,
        .cudnnGetConvolutionForwardAlgorithm_v7 = algorithms,
        .cudnnGetConvolutionForwardWorkspaceSize = workspace,
        .cudnnConvolutionForward = forward,
        .cudnnAddTensor = add,
    };
};

pub fn spec() !d.Spec {
    return .{ .input = try d.Tensor.contiguous(&.{ 4, 3, 2, 14, 14 }), .filter = try d.Tensor.contiguous(&.{ 4, 3, 2, 14, 14 }), .output = try d.Tensor.contiguous(&.{ 4, 4, 1, 1, 1 }), .bias = try d.Tensor.contiguous(&.{ 1, 4, 1, 1, 1 }), .stride = .{ 2, 14, 14 } };
}
pub fn buffers(s: d.Spec, workspace_need: usize) !d.Buffers {
    return .{ .input = .{ .ptr = 0x10000000, .len = try s.input.bytes() }, .filter = .{ .ptr = 0x20000000, .len = try s.filter.bytes() }, .output = .{ .ptr = 0x30000000, .len = try s.output.bytes() }, .bias = .{ .ptr = 0x40000000, .len = try s.bias.bytes() }, .workspace = .{ .ptr = if (workspace_need == 0) 0 else 0x50000000, .len = workspace_need } };
}

test "cuDNN SDK types, version and heuristic struct ABI come from pinned headers" {
    try testing.expectEqual(@as(usize, 92400), c.CUDNN_VERSION);
    try testing.expectEqual(@as(usize, 48), @sizeOf(c.cudnnConvolutionFwdAlgoPerf_t));
    try testing.expectEqual(@as(usize, 16), @offsetOf(c.cudnnConvolutionFwdAlgoPerf_t, "memory"));
    try testing.expectEqual(@as(c.cudnnDataType_t, 9), c.CUDNN_DATA_BFLOAT16);
}
test "cuDNN operator owns all six objects, dispatches conv then bias, and closes in reverse order" {
    Mock.reset();
    const s = try spec();
    var conv = try d.Convolution.init(&Mock.api, s, .{ .workspace_limit = 1024, .deterministic = false });
    try testing.expectEqual(@as(usize, 6), Mock.live);
    try testing.expectEqual(@as(usize, 256), conv.workspace_need);
    const stream: @import("cuda").abi.Stream = @ptrFromInt(0x1000);
    try conv.run(try buffers(s, conv.workspace_need), stream);
    try testing.expect(Mock.forwarded and Mock.biased);
    try testing.expectEqual(@as(usize, 0x1000), @intFromPtr(Mock.last_stream));
    conv.deinit();
    try testing.expectEqual(@as(usize, 0), Mock.live);
    try testing.expectEqualSlices(usize, &.{ 6, 5, 4, 3, 2, 1 }, Mock.destroy_order[0..Mock.destroyed]);
}
test "every descriptor, heuristic and workspace initialization failure releases acquired objects" {
    Mock.reset();
    const s = try spec();
    var baseline = try d.Convolution.init(&Mock.api, s, .{ .workspace_limit = 1024 });
    const calls = Mock.calls;
    baseline.deinit();
    for (1..calls + 1) |fail| {
        Mock.reset();
        Mock.fail_at = fail;
        try testing.expectError(error.CudnnFailed, d.Convolution.init(&Mock.api, s, .{ .workspace_limit = 1024 }));
        try testing.expectEqual(@as(usize, 0), Mock.live);
    }
}
test "success with a null handle or descriptor is an error and unwinds prior objects" {
    for ([_]usize{ 1, 2, 4, 6, 10, 12 }) |step| {
        Mock.reset();
        Mock.null_at = step;
        try testing.expectError(error.BadApiReturn, d.Convolution.init(&Mock.api, try spec(), .{ .workspace_limit = 1024 }));
        try testing.expectEqual(@as(usize, 0), Mock.live);
    }
}
test "exact SDK runtime policy is default; major-nine compatibility requires explicit opt-in" {
    try testing.expect(d.acceptsVersion(92400, .exact_header));
    try testing.expect(!d.acceptsVersion(91002, .exact_header));
    try testing.expect(d.acceptsVersion(91002, .major9));
    try testing.expect(!d.acceptsVersion(100000, .major9));
    Mock.reset();
    Mock.version = 91002;
    try testing.expectError(error.VersionMismatch, d.Convolution.init(&Mock.api, try spec(), .{ .workspace_limit = 1024 }));
    try testing.expectEqual(@as(usize, 0), Mock.calls);
    var opt_in = try d.Convolution.init(&Mock.api, try spec(), .{ .workspace_limit = 1024, .version_policy = .major9 });
    opt_in.deinit();
    Mock.reset();
    Mock.version = 100000;
    try testing.expectError(error.VersionMismatch, d.Convolution.init(&Mock.api, try spec(), .{ .workspace_limit = 1024, .version_policy = .major9 }));
    try testing.expectEqual(@as(usize, 0), Mock.live);
}
test "invalid shape and dtype refuse before any API call; SDK version mismatch refuses before handle creation" {
    Mock.reset();
    var s = try spec();
    s.input.dtype = .f32;
    try testing.expectError(error.UnsupportedDType, d.Convolution.init(&Mock.api, s, .{ .workspace_limit = 1024 }));
    s = try spec();
    s.output.dims[0] = 3;
    try testing.expectError(error.InvalidShape, d.Convolution.init(&Mock.api, s, .{ .workspace_limit = 1024 }));
    try testing.expectEqual(@as(usize, 0), Mock.calls);
    Mock.version = 90000;
    try testing.expectError(error.VersionMismatch, d.Convolution.init(&Mock.api, try spec(), .{ .workspace_limit = 1024 }));
    try testing.expectEqual(@as(usize, 0), Mock.live);
}
test "algorithm selection respects determinism, advertised and actual workspace limits, and malformed returns" {
    const s = try spec();
    Mock.reset();
    try testing.expectError(error.NoAlgorithm, d.Convolution.init(&Mock.api, s, .{ .workspace_limit = 1024, .deterministic = true }));
    try testing.expectEqual(@as(usize, 0), Mock.live);
    Mock.reset();
    Mock.count = 2;
    Mock.candidates[1].determinism = c.CUDNN_DETERMINISTIC;
    var good = try d.Convolution.init(&Mock.api, s, .{ .workspace_limit = 1024, .deterministic = true });
    good.deinit();
    Mock.reset();
    try testing.expectError(error.NoAlgorithm, d.Convolution.init(&Mock.api, s, .{ .workspace_limit = 128 }));
    Mock.reset();
    Mock.need = 2048;
    try testing.expectError(error.NoAlgorithm, d.Convolution.init(&Mock.api, s, .{ .workspace_limit = 1024 }));
    for ([_]c_int{ -1, 9 }) |count| {
        Mock.reset();
        Mock.count = count;
        try testing.expectError(error.BadApiReturn, d.Convolution.init(&Mock.api, s, .{ .workspace_limit = 1024 }));
        try testing.expectEqual(@as(usize, 0), Mock.live);
    }
}
test "span dtype, bounds, pointer overflow, workspace alignment and aliases refuse before stream or dispatch" {
    Mock.reset();
    const s = try spec();
    var conv = try d.Convolution.init(&Mock.api, s, .{ .workspace_limit = 1024 });
    defer conv.deinit();
    const calls = Mock.calls;
    const good = try buffers(s, 256);
    var b = good;
    b.input.len -= 1;
    try testing.expectError(error.InvalidSpan, conv.run(b, null));
    b = good;
    b.filter.ptr = 0;
    try testing.expectError(error.InvalidSpan, conv.run(b, null));
    b = good;
    b.bias.dtype = .f16;
    try testing.expectError(error.UnsupportedDType, conv.run(b, null));
    b = good;
    b.input.ptr = std.math.maxInt(u64) - 1;
    try testing.expectError(error.InvalidSpan, conv.run(b, null));
    b = good;
    b.input.ptr += 1;
    try testing.expectError(error.InvalidSpan, conv.run(b, null));
    b = good;
    b.workspace.ptr += 2;
    try testing.expectError(error.InvalidSpan, conv.run(b, null));
    b = good;
    b.output.ptr = b.input.ptr;
    try testing.expectError(error.Aliasing, conv.run(b, null));
    b = good;
    b.workspace.ptr = b.filter.ptr;
    try testing.expectError(error.Aliasing, conv.run(b, null));
    try testing.expectEqual(calls, Mock.calls);
}
test "zero workspace is legal; dispatch failures propagate without claiming successful bias" {
    const s = try spec();
    Mock.reset();
    Mock.need = 0;
    var zero = try d.Convolution.init(&Mock.api, s, .{ .workspace_limit = 1024 });
    try zero.run(try buffers(s, 0), null);
    zero.deinit();
    for (1..4) |step| {
        Mock.reset();
        var conv = try d.Convolution.init(&Mock.api, s, .{ .workspace_limit = 1024 });
        Mock.fail_at = Mock.calls + step;
        try testing.expectError(error.CudnnFailed, conv.run(try buffers(s, 256), null));
        try testing.expect(!Mock.biased);
        conv.deinit();
        try testing.expectEqual(@as(usize, 0), Mock.live);
    }
}
test "missing libraries and missing symbols refuse; real cuDNN bodies are compiled but never entered" {
    try testing.expectError(error.LibraryUnavailable, d.Library.openPath("tf-no-cudnn-sdk-library-6a90.so"));
    if (@import("builtin").os.tag == .windows) try testing.expectError(error.MissingSymbol, d.Library.openPath("kernel32.dll"));
    var enabled = false;
    std.mem.doNotOptimizeAway(&enabled);
    if (enabled) {
        var library = try d.Library.openPathWithPolicy("libcudnn.so.9", .major9);
        defer library.close();
        var conv = try d.Convolution.init(&library.api, try spec(), .{ .workspace_limit = 1 << 20, .deterministic = false });
        defer conv.deinit();
        try conv.run(try buffers(try spec(), conv.workspace_need), null);
    }
}
