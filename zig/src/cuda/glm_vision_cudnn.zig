//! Owning BF16 convolution adapter. Target cuDNN 9.24.0 declarations come from the SDK,
//! not hand-written ABI. Handles require the caller's CUDA context to be current.
//! No context/driver creation, no matmul fallback, no numerical-parity claim.
const std = @import("std");
const builtin = @import("builtin");
const cuda = @import("cuda");
const dynlib = cuda.dynlib;
const abi = cuda.abi;
pub const c = @import("cudnn_headers");
pub const Status = c.cudnnStatus_t;
pub const Error = error{ LibraryUnavailable, MissingSymbol, VersionMismatch, CudnnFailed, UnsupportedOperation, UnsupportedDType, InvalidShape, InvalidSpan, Aliasing, NoAlgorithm, BadApiReturn };
pub const DataType = enum { bf16, f16, f32 };
pub const header_version: usize = c.CUDNN_VERSION;
/// major9 is an explicit ABI compatibility opt-in, not a numerical qualification.
pub const VersionPolicy = enum { exact_header, major9 };
pub fn acceptsVersion(version: usize, policy: VersionPolicy) bool {
    return switch (policy) {
        .exact_header => version == header_version,
        .major9 => version / 10000 == 9,
    };
}

pub const Api = struct {
    cudnnGetVersion: @TypeOf(&c.cudnnGetVersion),
    cudnnCreate: @TypeOf(&c.cudnnCreate),
    cudnnDestroy: @TypeOf(&c.cudnnDestroy),
    cudnnSetStream: @TypeOf(&c.cudnnSetStream),
    cudnnCreateTensorDescriptor: @TypeOf(&c.cudnnCreateTensorDescriptor),
    cudnnSetTensorNdDescriptor: @TypeOf(&c.cudnnSetTensorNdDescriptor),
    cudnnDestroyTensorDescriptor: @TypeOf(&c.cudnnDestroyTensorDescriptor),
    cudnnCreateFilterDescriptor: @TypeOf(&c.cudnnCreateFilterDescriptor),
    cudnnSetFilterNdDescriptor: @TypeOf(&c.cudnnSetFilterNdDescriptor),
    cudnnDestroyFilterDescriptor: @TypeOf(&c.cudnnDestroyFilterDescriptor),
    cudnnCreateConvolutionDescriptor: @TypeOf(&c.cudnnCreateConvolutionDescriptor),
    cudnnSetConvolutionNdDescriptor: @TypeOf(&c.cudnnSetConvolutionNdDescriptor),
    cudnnSetConvolutionMathType: @TypeOf(&c.cudnnSetConvolutionMathType),
    cudnnSetConvolutionGroupCount: @TypeOf(&c.cudnnSetConvolutionGroupCount),
    cudnnDestroyConvolutionDescriptor: @TypeOf(&c.cudnnDestroyConvolutionDescriptor),
    cudnnGetConvolutionForwardAlgorithm_v7: @TypeOf(&c.cudnnGetConvolutionForwardAlgorithm_v7),
    cudnnGetConvolutionForwardWorkspaceSize: @TypeOf(&c.cudnnGetConvolutionForwardWorkspaceSize),
    cudnnConvolutionForward: @TypeOf(&c.cudnnConvolutionForward),
    cudnnAddTensor: @TypeOf(&c.cudnnAddTensor),
};
fn check(status: Status) Error!void {
    if (status == c.CUDNN_STATUS_SUCCESS) return;
    if (status >= 3000 and status < 4000) return error.UnsupportedOperation;
    return error.CudnnFailed;
}

pub const Library = struct {
    lib: dynlib.Library,
    api: Api,
    /// Exactly the compiled SDK version; package the full cuDNN sublibrary set, not just this soname.
    pub fn open() Error!Library {
        if (builtin.os.tag == .windows) return error.LibraryUnavailable;
        return openPath("libcudnn.so.9");
    }
    pub fn openPath(path: []const u8) Error!Library {
        return openPathWithPolicy(path, .exact_header);
    }
    pub fn openPathWithPolicy(path: []const u8, policy: VersionPolicy) Error!Library {
        var lib = dynlib.Library.open(path) catch return error.LibraryUnavailable;
        errdefer lib.close();
        var missing: []const u8 = "";
        const api = lib.resolve(Api, &missing) catch return error.MissingSymbol;
        if (!acceptsVersion(api.cudnnGetVersion(), policy)) return error.VersionMismatch;
        return .{ .lib = lib, .api = api };
    }
    /// All Convolution instances borrowing api must have been destroyed first.
    pub fn close(self: *Library) void {
        self.lib.close();
        self.* = undefined;
    }
};

pub const Tensor = struct {
    dtype: DataType = .bf16,
    rank: u8,
    dims: [5]c_int = @splat(1),
    strides: [5]c_int = @splat(1),
    pub fn contiguous(dims: []const c_int) Error!Tensor {
        if (dims.len < 4 or dims.len > 5) return error.InvalidShape;
        var t: Tensor = .{ .rank = @intCast(dims.len) };
        @memcpy(t.dims[0..dims.len], dims);
        var stride: c_int = 1;
        var i = dims.len;
        while (i > 0) {
            i -= 1;
            if (dims[i] <= 0) return error.InvalidShape;
            t.strides[i] = stride;
            stride = std.math.mul(c_int, stride, dims[i]) catch return error.InvalidShape;
        }
        _ = try t.bytes();
        return t;
    }
    /// Required storage for nonoverlapping positive-stride views, including any gaps.
    pub fn bytes(self: Tensor) Error!usize {
        if (self.dtype != .bf16) return error.UnsupportedDType;
        if (self.rank < 4 or self.rank > 5) return error.InvalidShape;
        var indices: [5]usize = .{ 0, 1, 2, 3, 4 };
        const order = indices[0..self.rank];
        for (order) |i| if (self.dims[i] <= 0 or self.strides[i] <= 0) return error.InvalidShape;
        for (0..order.len) |i| for (i + 1..order.len) |j| {
            if (self.strides[order[j]] < self.strides[order[i]]) std.mem.swap(usize, &order[i], &order[j]);
        };
        var extent: usize = 1;
        for (order) |i| {
            if (self.dims[i] == 1) continue;
            const stride: usize = @intCast(self.strides[i]);
            if (stride < extent) return error.InvalidShape;
            const extra = std.math.mul(usize, @intCast(self.dims[i] - 1), stride) catch return error.InvalidShape;
            extent = std.math.add(usize, extent, extra) catch return error.InvalidShape;
        }
        if (extent > std.math.maxInt(c_int)) return error.InvalidShape;
        return std.math.mul(usize, extent, 2) catch error.InvalidShape;
    }
};

pub const Spec = struct {
    input: Tensor,
    filter: Tensor,
    output: Tensor,
    bias: Tensor,
    stride: [3]c_int,
    pub fn check(self: Spec) Error!void {
        _ = try self.input.bytes();
        _ = try self.filter.bytes();
        _ = try self.output.bytes();
        _ = try self.bias.bytes();
        const rank = self.input.rank;
        if (self.filter.rank != rank or self.output.rank != rank or self.bias.rank != rank or
            self.input.dims[0] != self.output.dims[0] or self.input.dims[1] != self.filter.dims[1] or
            self.output.dims[1] != self.filter.dims[0] or self.bias.dims[0] != 1 or self.bias.dims[1] != self.output.dims[1]) return error.InvalidShape;
        const dense = try Tensor.contiguous(self.filter.dims[0..rank]);
        if (!std.mem.eql(c_int, dense.strides[0..rank], self.filter.strides[0..rank])) return error.InvalidShape;
        for (2..rank) |i| {
            if (self.stride[i - 2] <= 0 or self.input.dims[i] < self.filter.dims[i] or self.bias.dims[i] != 1 or
                self.output.dims[i] != @divTrunc(self.input.dims[i] - self.filter.dims[i], self.stride[i - 2]) + 1) return error.InvalidShape;
        }
    }
};

pub const Span = struct {
    ptr: abi.DevicePtr,
    len: usize,
    dtype: DataType = .bf16,
    fn checked(self: Span, need: usize, alignment: usize) Error!void {
        if (need == 0) return;
        if (self.ptr == 0 or self.len < need or self.ptr % alignment != 0) return error.InvalidSpan;
        _ = std.math.add(u64, self.ptr, need) catch return error.InvalidSpan;
    }
    fn constant(self: Span) ?*const anyopaque {
        return if (self.ptr == 0) null else @ptrFromInt(@as(usize, @intCast(self.ptr)));
    }
    fn mutable(self: Span) ?*anyopaque {
        return if (self.ptr == 0) null else @ptrFromInt(@as(usize, @intCast(self.ptr)));
    }
};
pub const Buffers = struct { input: Span, filter: Span, output: Span, bias: Span, workspace: Span };
pub const Options = struct { workspace_limit: usize, deterministic: bool = false, version_policy: VersionPolicy = .exact_header };
fn overlap(a: Span, an: usize, b: Span, bn: usize) bool {
    return an != 0 and bn != 0 and a.ptr < b.ptr + bn and b.ptr < a.ptr + an;
}

pub const Convolution = struct {
    api: *const Api,
    spec: Spec,
    handle: c.cudnnHandle_t = null,
    x: c.cudnnTensorDescriptor_t = null,
    w: c.cudnnFilterDescriptor_t = null,
    conv: c.cudnnConvolutionDescriptor_t = null,
    y: c.cudnnTensorDescriptor_t = null,
    b: c.cudnnTensorDescriptor_t = null,
    algo: c.cudnnConvolutionFwdAlgo_t = 0,
    workspace_need: usize = 0,
    /// Caller owns context, stream and GPU buffers. API/library outlives this object.
    pub fn init(api: *const Api, spec: Spec, options: Options) Error!Convolution {
        try spec.check();
        if (!acceptsVersion(api.cudnnGetVersion(), options.version_policy)) return error.VersionMismatch;
        var self: Convolution = .{ .api = api, .spec = spec };
        errdefer self.deinit();
        try check(api.cudnnCreate(&self.handle));
        if (self.handle == null) return error.BadApiReturn;
        try check(api.cudnnCreateTensorDescriptor(&self.x));
        if (self.x == null) return error.BadApiReturn;
        try setTensor(api, self.x, spec.input);
        try check(api.cudnnCreateFilterDescriptor(&self.w));
        if (self.w == null) return error.BadApiReturn;
        try check(api.cudnnSetFilterNdDescriptor(self.w, c.CUDNN_DATA_BFLOAT16, c.CUDNN_TENSOR_NCHW, spec.filter.rank, &spec.filter.dims));
        try check(api.cudnnCreateConvolutionDescriptor(&self.conv));
        if (self.conv == null) return error.BadApiReturn;
        const zero = [_]c_int{ 0, 0, 0 };
        const one = [_]c_int{ 1, 1, 1 };
        try check(api.cudnnSetConvolutionNdDescriptor(self.conv, spec.input.rank - 2, &zero, &spec.stride, &one, c.CUDNN_CROSS_CORRELATION, c.CUDNN_DATA_FLOAT));
        try check(api.cudnnSetConvolutionGroupCount(self.conv, 1));
        try check(api.cudnnSetConvolutionMathType(self.conv, c.CUDNN_TENSOR_OP_MATH));
        try check(api.cudnnCreateTensorDescriptor(&self.y));
        if (self.y == null) return error.BadApiReturn;
        try setTensor(api, self.y, spec.output);
        try check(api.cudnnCreateTensorDescriptor(&self.b));
        if (self.b == null) return error.BadApiReturn;
        try setTensor(api, self.b, spec.bias);
        var candidates: [8]c.cudnnConvolutionFwdAlgoPerf_t = undefined;
        var count: c_int = 0;
        try check(api.cudnnGetConvolutionForwardAlgorithm_v7(self.handle, self.x, self.w, self.conv, self.y, candidates.len, &count, &candidates));
        if (count < 0 or count > candidates.len) return error.BadApiReturn;
        for (candidates[0..@intCast(count)]) |candidate| {
            if (candidate.status != c.CUDNN_STATUS_SUCCESS or candidate.memory > options.workspace_limit or
                candidate.algo >= c.CUDNN_CONVOLUTION_FWD_ALGO_COUNT or
                (options.deterministic and candidate.determinism != c.CUDNN_DETERMINISTIC) or
                (candidate.mathType != c.CUDNN_DEFAULT_MATH and candidate.mathType != c.CUDNN_TENSOR_OP_MATH)) continue;
            try check(api.cudnnSetConvolutionMathType(self.conv, candidate.mathType));
            var need: usize = 0;
            check(api.cudnnGetConvolutionForwardWorkspaceSize(self.handle, self.x, self.w, self.conv, self.y, candidate.algo, &need)) catch |err| switch (err) {
                error.UnsupportedOperation => continue,
                else => return err,
            };
            if (need > options.workspace_limit) continue;
            self.algo = candidate.algo;
            self.workspace_need = need;
            return self;
        }
        return error.NoAlgorithm;
    }
    fn setTensor(api: *const Api, desc: c.cudnnTensorDescriptor_t, t: Tensor) Error!void {
        try check(api.cudnnSetTensorNdDescriptor(desc, c.CUDNN_DATA_BFLOAT16, t.rank, &t.dims, &t.strides));
    }
    pub fn deinit(self: *Convolution) void {
        const api = self.api;
        if (self.b != null) _ = api.cudnnDestroyTensorDescriptor(self.b);
        if (self.y != null) _ = api.cudnnDestroyTensorDescriptor(self.y);
        if (self.conv != null) _ = api.cudnnDestroyConvolutionDescriptor(self.conv);
        if (self.w != null) _ = api.cudnnDestroyFilterDescriptor(self.w);
        if (self.x != null) _ = api.cudnnDestroyTensorDescriptor(self.x);
        if (self.handle != null) _ = api.cudnnDestroy(self.handle);
        self.* = undefined;
    }
    /// Conv rounds to BF16 before a separate BF16 bias add. Algorithm and rounding parity
    /// against the target Torch/cuDNN backend remain GPU qualification work.
    /// Enqueues on stream; caller retains all buffers until GPU completion. Not thread-safe.
    pub fn run(self: *const Convolution, buffers: Buffers, stream: abi.Stream) Error!void {
        const spans = [_]Span{ buffers.input, buffers.filter, buffers.output, buffers.bias, buffers.workspace };
        const sizes = [_]usize{ try self.spec.input.bytes(), try self.spec.filter.bytes(), try self.spec.output.bytes(), try self.spec.bias.bytes(), self.workspace_need };
        for (spans[0..4]) |span| if (span.dtype != .bf16) return error.UnsupportedDType;
        for (spans, sizes, 0..) |span, need, i| try span.checked(need, if (i == 4) 256 else 2);
        for ([_]usize{ 0, 1, 3, 4 }) |i| if (overlap(spans[2], sizes[2], spans[i], sizes[i])) return error.Aliasing;
        for (0..4) |i| if (overlap(spans[4], sizes[4], spans[i], sizes[i])) return error.Aliasing;
        const one: f32 = 1;
        const zero: f32 = 0;
        try check(self.api.cudnnSetStream(self.handle, @ptrCast(stream)));
        try check(self.api.cudnnConvolutionForward(self.handle, &one, self.x, buffers.input.constant(), self.w, buffers.filter.constant(), self.conv, self.algo, if (self.workspace_need == 0) null else buffers.workspace.mutable(), self.workspace_need, &zero, self.y, buffers.output.mutable()));
        try check(self.api.cudnnAddTensor(self.handle, &one, self.b, buffers.bias.constant(), &one, self.y, buffers.output.mutable()));
    }
};

comptime {
    if (@sizeOf(usize) != 8) @compileError("cuDNN vision adapter requires a 64-bit CUDA target");
    if (c.CUDNN_MAJOR != 9) @compileError("cuDNN vision adapter requires the v9 SDK ABI");
}

pub const test_support = if (builtin.is_test) @import("glm_vision_cudnn_test.zig") else void;
test {
    _ = test_support;
}
