//! Owning direct-SDPA backend operator, not a tower or a numerical-parity qualification.
//! All cuDNN declarations/constants are translated from the actual 9.24 SDK.
//! Unmasked noncausal BF16 self-attention only; no paging, padding masks, dropout or training stats.
//! Caller supplies normalized/rotated Q,K and unchanged V. CUDA context must be current.
const std = @import("std");
const builtin = @import("builtin");
const cuda = @import("cuda");
const platform = @import("sys");
const abi = cuda.abi;
pub const c = @import("cudnn_headers");
pub const Status = c.cudnnStatus_t;
pub const Descriptor = c.cudnnBackendDescriptor_t;
pub const target_cudnn: usize = 92400;
/// CUDART_VERSION encoding is major*1000 + minor*10 (12.8 is 12080).
pub const target_cudart: usize = 13 * 1000 + 3 * 10;
pub const max_engine_configs = 32;
pub const tensor_ids = [5]i64{ 1, 2, 3, 4, 5 }; // Q,K,V,O,scale
pub const Error = error{ LibraryUnavailable, MissingSymbol, VersionMismatch, CudaVersionMismatch, CudnnFailed, UnsupportedOperation, UnsupportedDType, InvalidShape, InvalidOptions, InvalidSpan, Aliasing, DriverMismatch, NoPlan, BadApiReturn, WorkspaceLimitExceeded, WorkspaceRequirementChanged };

pub const Api = struct {
    cudnnGetVersion: @TypeOf(&c.cudnnGetVersion),
    cudnnGetCudartVersion: @TypeOf(&c.cudnnGetCudartVersion),
    cudnnCreate: @TypeOf(&c.cudnnCreate),
    cudnnDestroy: @TypeOf(&c.cudnnDestroy),
    cudnnSetStream: @TypeOf(&c.cudnnSetStream),
    cudnnBackendCreateDescriptor: @TypeOf(&c.cudnnBackendCreateDescriptor),
    cudnnBackendDestroyDescriptor: @TypeOf(&c.cudnnBackendDestroyDescriptor),
    cudnnBackendSetAttribute: @TypeOf(&c.cudnnBackendSetAttribute),
    cudnnBackendGetAttribute: @TypeOf(&c.cudnnBackendGetAttribute),
    cudnnBackendFinalize: @TypeOf(&c.cudnnBackendFinalize),
    cudnnGetExecutionPlanWorkspaceSize: @TypeOf(&c.cudnnGetExecutionPlanWorkspaceSize),
    cudnnBackendExecute: @TypeOf(&c.cudnnBackendExecute),
};
fn check(status: Status) Error!void {
    if (status == c.CUDNN_STATUS_SUCCESS) return;
    if (status >= c.CUDNN_STATUS_NOT_SUPPORTED and status < c.CUDNN_STATUS_INTERNAL_ERROR) return error.UnsupportedOperation;
    return error.CudnnFailed;
}
fn version(api: *const Api) Error!void {
    if (api.cudnnGetVersion() != target_cudnn) return error.VersionMismatch;
    // Statically linked cuDNN CUDA runtime version, not proof of the current driver/context.
    if (api.cudnnGetCudartVersion() != target_cudart) return error.CudaVersionMismatch;
}

pub const Library = struct {
    lib: platform.DynLib,
    api: Api,
    pub fn open() Error!Library {
        if (builtin.os.tag == .windows) return error.LibraryUnavailable;
        return openPath("libcudnn.so.9");
    }
    /// Package the full matching sublibrary set. No older-version or graph-composition fallback.
    pub fn openPath(path: []const u8) Error!Library {
        var lib = platform.DynLib.open(path) catch return error.LibraryUnavailable;
        errdefer lib.close();
        var missing: []const u8 = "";
        const api = platform.resolveSymbols(&lib, Api, &missing) catch return error.MissingSymbol;
        try version(&api);
        return .{ .lib = lib, .api = api };
    }
    /// All operators borrowing api must be destroyed first.
    pub fn close(self: *Library) void {
        self.lib.close();
        self.* = undefined;
    }
};

pub const Tensor = struct {
    dims: [4]i64,
    strides: [4]i64,
    dtype: c.cudnnDataType_t = c.CUDNN_DATA_BFLOAT16,
    alignment: i64 = 16,
    /// Conservative positive nonoverlapping view extent; includes inter-row gaps.
    pub fn bytes(self: Tensor) Error!usize {
        const unit: usize = switch (self.dtype) {
            c.CUDNN_DATA_BFLOAT16 => 2,
            c.CUDNN_DATA_FLOAT => 4,
            else => return error.UnsupportedDType,
        };
        if (self.alignment < unit or self.alignment > 256 or !std.math.isPowerOfTwo(self.alignment)) return error.InvalidShape;
        var order = [4]usize{ 0, 1, 2, 3 };
        for (self.dims, self.strides) |dim, stride| if (dim <= 0 or stride <= 0) return error.InvalidShape;
        for (0..4) |i| for (i + 1..4) |j| {
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
        if (extent > std.math.maxInt(i64)) return error.InvalidShape;
        return std.math.mul(usize, extent, unit) catch error.InvalidShape;
    }
};
pub const Spec = struct {
    q: Tensor,
    k: Tensor,
    v: Tensor,
    output: Tensor,
    scale: f32,
    pub fn check(self: Spec) Error!void {
        if (!std.math.isFinite(self.scale) or self.scale <= 0) return error.InvalidShape;
        for ([_]Tensor{ self.q, self.k, self.v, self.output }) |tensor| {
            if (tensor.dtype != c.CUDNN_DATA_BFLOAT16) return error.UnsupportedDType;
            _ = try tensor.bytes();
            if (!std.mem.eql(i64, &self.q.dims, &tensor.dims)) return error.InvalidShape;
        }
    }
};
pub const Span = struct {
    ptr: abi.DevicePtr,
    len: usize,
    pub fn checked(self: Span, need: usize, alignment: usize) Error!void {
        if (need == 0) return;
        if (self.ptr == 0 or self.len < need or self.ptr % alignment != 0) return error.InvalidSpan;
        _ = std.math.add(u64, self.ptr, need) catch return error.InvalidSpan;
    }
    fn pointer(self: Span) ?*anyopaque {
        return if (self.ptr == 0) null else @ptrFromInt(@as(usize, @intCast(self.ptr)));
    }
};
pub const Buffers = struct { q: Span, k: Span, v: Span, output: Span, workspace: Span };
pub const Options = struct {
    workspace_limit: usize,
    /// A bounded prefix of heuristic results, not an exhaustive engine search/autotune.
    engine_limit: u8 = 8,
    heuristic: c.cudnnBackendHeurMode_t = c.CUDNN_HEUR_MODE_A,
    fn check(self: Options) Error!void {
        if (self.engine_limit == 0 or self.engine_limit > max_engine_configs) return error.InvalidOptions;
        switch (self.heuristic) {
            c.CUDNN_HEUR_MODE_A, c.CUDNN_HEUR_MODE_B, c.CUDNN_HEUR_MODE_FALLBACK, c.CUDNN_HEUR_MODE_INSTANT => {},
            else => return error.InvalidOptions,
        }
    }
};
fn overlap(a: Span, an: usize, b: Span, bn: usize) bool {
    return an != 0 and bn != 0 and a.ptr < b.ptr + bn and b.ptr < a.ptr + an;
}

pub const Attention = struct {
    api: *const Api,
    spec: Spec,
    handle: c.cudnnHandle_t = null,
    plan: Descriptor = null,
    descriptors: [8 + max_engine_configs + 1]Descriptor = @splat(null),
    count: usize = 0,
    workspace_need: usize = 0,
    workspace_limit: usize,

    /// Owns handle/descriptors only. API/library/context outlive this object; no buffer ownership transfer.
    pub fn init(api: *const Api, spec: Spec, options: Options) Error!Attention {
        try spec.check();
        try options.check();
        try version(api);
        var self: Attention = .{ .api = api, .spec = spec, .workspace_limit = options.workspace_limit };
        errdefer self.deinit();
        try check(api.cudnnCreate(&self.handle));
        if (self.handle == null) return error.BadApiReturn;
        const q = try self.tensor(spec.q, tensor_ids[0], false);
        const k = try self.tensor(spec.k, tensor_ids[1], false);
        const v = try self.tensor(spec.v, tensor_ids[2], false);
        const o = try self.tensor(spec.output, tensor_ids[3], false);
        const scale = try self.tensor(.{ .dims = .{ 1, 1, 1, 1 }, .strides = .{ 1, 1, 1, 1 }, .dtype = c.CUDNN_DATA_FLOAT, .alignment = 4 }, tensor_ids[4], true);
        const op = try self.create(c.CUDNN_BACKEND_OPERATION_SDPA_FWD_DESCRIPTOR);
        try self.set(op, c.CUDNN_ATTR_OPERATION_SDPA_FWD_QDESC, c.CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &q);
        try self.set(op, c.CUDNN_ATTR_OPERATION_SDPA_FWD_KDESC, c.CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &k);
        try self.set(op, c.CUDNN_ATTR_OPERATION_SDPA_FWD_VDESC, c.CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &v);
        try self.set(op, c.CUDNN_ATTR_OPERATION_SDPA_FWD_ODESC, c.CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &o);
        try self.set(op, c.CUDNN_ATTR_OPERATION_SDPA_FWD_SCALEDESC, c.CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &scale);
        // No mask/stats/page/sequence-length/softmax/dropout attributes: the basic unmasked formula.
        try check(api.cudnnBackendFinalize(op));
        const graph = try self.create(c.CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR);
        try self.set(graph, c.CUDNN_ATTR_OPERATIONGRAPH_HANDLE, c.CUDNN_TYPE_HANDLE, 1, &self.handle);
        try self.set(graph, c.CUDNN_ATTR_OPERATIONGRAPH_OPS, c.CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &op);
        // Leave the graph mode AUTO as the SDK requires for hand-constructed graphs.
        try check(api.cudnnBackendFinalize(graph));
        const heur = try self.create(c.CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR);
        try self.set(heur, c.CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH, c.CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &graph);
        try self.set(heur, c.CUDNN_ATTR_ENGINEHEUR_MODE, c.CUDNN_TYPE_HEUR_MODE, 1, &options.heuristic);
        try check(api.cudnnBackendFinalize(heur));
        var available: i64 = 0;
        try check(api.cudnnBackendGetAttribute(heur, c.CUDNN_ATTR_ENGINEHEUR_RESULTS, c.CUDNN_TYPE_BACKEND_DESCRIPTOR, 0, &available, null));
        if (available < 0) return error.BadApiReturn;
        if (available == 0) return error.NoPlan;
        const n: usize = @intCast(@min(available, options.engine_limit));
        var configs: [max_engine_configs]Descriptor = @splat(null);
        for (configs[0..n]) |*config| config.* = try self.create(c.CUDNN_BACKEND_ENGINECFG_DESCRIPTOR);
        const original = configs;
        var returned: i64 = 0;
        try check(api.cudnnBackendGetAttribute(heur, c.CUDNN_ATTR_ENGINEHEUR_RESULTS, c.CUDNN_TYPE_BACKEND_DESCRIPTOR, @intCast(n), &returned, @ptrCast(&configs)));
        if (returned < 0 or returned > n) return error.BadApiReturn;
        // cuDNN populates caller-created engine configs; it must not substitute opaque handles.
        for (configs[0..n], original[0..n]) |actual, expected| if (actual != expected) return error.BadApiReturn;
        for (configs[0..@intCast(returned)]) |config| {
            const plan = try self.create(c.CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR);
            try self.set(plan, c.CUDNN_ATTR_EXECUTION_PLAN_HANDLE, c.CUDNN_TYPE_HANDLE, 1, &self.handle);
            try self.set(plan, c.CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG, c.CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &config);
            check(api.cudnnBackendFinalize(plan)) catch |err| switch (err) {
                error.UnsupportedOperation => {
                    self.pop();
                    continue;
                },
                else => return err,
            };
            var count: i64 = 0;
            var workspace: i64 = 0;
            try check(api.cudnnBackendGetAttribute(plan, c.CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE, c.CUDNN_TYPE_INT64, 1, &count, &workspace));
            if (count != 1 or workspace < 0) return error.BadApiReturn;
            const need: usize = @intCast(workspace);
            if (need > options.workspace_limit) {
                self.pop();
                continue;
            }
            self.plan = plan;
            self.workspace_need = need;
            return self;
        }
        return error.NoPlan;
    }
    fn create(self: *Attention, kind: c.cudnnBackendDescriptorType_t) Error!Descriptor {
        if (self.count == self.descriptors.len) return error.BadApiReturn;
        var desc: Descriptor = null;
        const status = self.api.cudnnBackendCreateDescriptor(kind, &desc);
        // Retain even a non-null result accompanying failure, so unwind attempts cleanup.
        if (desc != null) {
            self.descriptors[self.count] = desc;
            self.count += 1;
        }
        try check(status);
        if (desc == null) return error.BadApiReturn;
        return desc;
    }
    fn pop(self: *Attention) void {
        self.count -= 1;
        _ = self.api.cudnnBackendDestroyDescriptor(self.descriptors[self.count]);
        self.descriptors[self.count] = null;
    }
    fn set(self: *const Attention, desc: Descriptor, name: c.cudnnBackendAttributeName_t, kind: c.cudnnBackendAttributeType_t, n: i64, ptr: anytype) Error!void {
        try check(self.api.cudnnBackendSetAttribute(desc, name, kind, n, @ptrCast(ptr)));
    }
    fn tensor(self: *Attention, t: Tensor, id: i64, by_value: bool) Error!Descriptor {
        _ = try t.bytes();
        const desc = try self.create(c.CUDNN_BACKEND_TENSOR_DESCRIPTOR);
        const nonvirtual = false;
        try self.set(desc, c.CUDNN_ATTR_TENSOR_UNIQUE_ID, c.CUDNN_TYPE_INT64, 1, &id);
        try self.set(desc, c.CUDNN_ATTR_TENSOR_DATA_TYPE, c.CUDNN_TYPE_DATA_TYPE, 1, &t.dtype);
        try self.set(desc, c.CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT, c.CUDNN_TYPE_INT64, 1, &t.alignment);
        try self.set(desc, c.CUDNN_ATTR_TENSOR_DIMENSIONS, c.CUDNN_TYPE_INT64, 4, &t.dims);
        try self.set(desc, c.CUDNN_ATTR_TENSOR_STRIDES, c.CUDNN_TYPE_INT64, 4, &t.strides);
        try self.set(desc, c.CUDNN_ATTR_TENSOR_IS_VIRTUAL, c.CUDNN_TYPE_BOOLEAN, 1, &nonvirtual);
        try self.set(desc, c.CUDNN_ATTR_TENSOR_IS_BY_VALUE, c.CUDNN_TYPE_BOOLEAN, 1, &by_value);
        try check(self.api.cudnnBackendFinalize(desc));
        return desc;
    }
    pub fn deinit(self: *Attention) void {
        while (self.count > 0) self.pop();
        if (self.handle != null) _ = self.api.cudnnDestroy(self.handle);
        self.* = undefined;
    }
    /// Not thread-safe. Retain operator, API/library, stream and all GPU buffers until GPU completion.
    /// A temporary variant pack holds this object's stable host scale, not a stack-local scalar.
    pub fn run(self: *const Attention, buffers: Buffers, stream: abi.Stream) Error!void {
        const spans = [_]Span{ buffers.q, buffers.k, buffers.v, buffers.output, buffers.workspace };
        const tensors = [_]Tensor{ self.spec.q, self.spec.k, self.spec.v, self.spec.output };
        var sizes: [5]usize = undefined;
        for (tensors, 0..) |view, i| {
            sizes[i] = try view.bytes();
            try spans[i].checked(sizes[i], @intCast(view.alignment));
        }
        sizes[4] = self.workspace_need;
        try spans[4].checked(sizes[4], 256);
        for (0..3) |i| if (overlap(spans[3], sizes[3], spans[i], sizes[i])) return error.Aliasing;
        for (0..4) |i| if (overlap(spans[4], sizes[4], spans[i], sizes[i])) return error.Aliasing;
        var pack: Descriptor = null;
        const status = self.api.cudnnBackendCreateDescriptor(c.CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR, &pack);
        defer {
            if (pack != null) _ = self.api.cudnnBackendDestroyDescriptor(pack);
        }
        try check(status);
        if (pack == null) return error.BadApiReturn;
        const pointers = [5]?*const anyopaque{ buffers.q.pointer(), buffers.k.pointer(), buffers.v.pointer(), buffers.output.pointer(), &self.spec.scale };
        const workspace = if (self.workspace_need == 0) null else buffers.workspace.pointer();
        try self.set(pack, c.CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS, c.CUDNN_TYPE_INT64, tensor_ids.len, &tensor_ids);
        try self.set(pack, c.CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS, c.CUDNN_TYPE_VOID_PTR, pointers.len, &pointers);
        try self.set(pack, c.CUDNN_ATTR_VARIANT_PACK_WORKSPACE, c.CUDNN_TYPE_VOID_PTR, 1, &workspace);
        try check(self.api.cudnnBackendFinalize(pack));
        try check(self.api.cudnnSetStream(self.handle, @ptrCast(stream)));
        var runtime_need: usize = 0;
        try check(self.api.cudnnGetExecutionPlanWorkspaceSize(self.handle, self.plan, pack, &runtime_need));
        if (runtime_need > self.workspace_limit) return error.WorkspaceLimitExceeded;
        // Static tensor geometry: refuse an unexpected larger requirement rather than execute unsafely.
        if (runtime_need > self.workspace_need) return error.WorkspaceRequirementChanged;
        try check(self.api.cudnnBackendExecute(self.handle, self.plan, pack));
    }
};

comptime {
    if (@sizeOf(usize) != 8) @compileError("direct SDPA requires a 64-bit CUDA target");
    if (c.CUDNN_VERSION != target_cudnn) @compileError("direct SDPA requires translated cuDNN 9.24.0 declarations");
}
pub const test_support = if (builtin.is_test) @import("glm_vision_sdpa_test.zig") else void;
test {
    _ = test_support;
}
