const std = @import("std");
const d = @import("glm_vision_sdpa.zig");
const c = d.c;
const t = std.testing;
const cc = @typeInfo(@TypeOf(c.cudnnCreate)).@"fn".attrs.@"callconv";
const S = d.Status;
const Desc = d.Descriptor;

/// Typed SDK callbacks only. No CUDA context/library/device access and no fake numerical output.
pub const Mock = struct {
    const Record = struct {
        kind: c.cudnnBackendDescriptorType_t = 0,
        live: bool = false,
        finalized: bool = false,
        dims: [4]i64 = @splat(0),
        strides: [4]i64 = @splat(0),
        uid: i64 = 0,
        dtype: c.cudnnDataType_t = 0,
        alignment: i64 = 0,
        virtual: bool = true,
        by_value: bool = false,
        refs: [5]usize = @splat(0),
        config_index: usize = 0,
        heuristic: c.cudnnBackendHeurMode_t = 0,
        pointers: [5]usize = @splat(0),
        workspace: usize = 0,
        attrs: usize = 0,
    };
    pub var calls: usize = 0;
    pub var fail_at: usize = 0;
    pub var null_at: usize = 0;
    pub var nonnull_failure: bool = false;
    pub var live: usize = 0;
    var next: usize = 1;
    pub var records: [128]Record = @splat(.{});
    pub var creates: [128]usize = @splat(0);
    pub var create_count: usize = 0;
    pub var destroyed: [128]usize = @splat(0);
    pub var destroy_count: usize = 0;
    pub var cudnn_version: usize = d.target_cudnn;
    pub var cuda_version: usize = d.target_cudart;
    pub var available: i64 = 1;
    pub var returned: ?i64 = null;
    pub var substitute_result: bool = false;
    pub var workspace_count: i64 = 1;
    pub var needs: [32]i64 = @splat(256);
    pub var supported: [32]bool = @splat(true);
    pub var runtime_need: usize = 256;
    pub var runtime_queried: bool = false;
    pub var executed: bool = false;
    pub var last_stream: c.cudaStream_t = null;
    pub var last_pointers: [5]usize = @splat(0);
    pub var expected_scale: f32 = 0.125;
    pub var plans_tried: usize = 0;
    pub var configs_requested: i64 = 0;
    pub var direct_supported: bool = true;
    pub fn reset() void {
        calls = 0;
        fail_at = 0;
        null_at = 0;
        nonnull_failure = false;
        live = 0;
        next = 1;
        records = @splat(.{});
        creates = @splat(0);
        create_count = 0;
        destroy_count = 0;
        cudnn_version = d.target_cudnn;
        cuda_version = d.target_cudart;
        available = 1;
        returned = null;
        substitute_result = false;
        workspace_count = 1;
        needs = @splat(256);
        supported = @splat(true);
        runtime_need = 256;
        runtime_queried = false;
        executed = false;
        last_stream = null;
        last_pointers = @splat(0);
        expected_scale = 0.125;
        plans_tried = 0;
        configs_requested = 0;
        direct_supported = true;
    }
    fn tick() S {
        calls += 1;
        return if (calls == fail_at) c.CUDNN_STATUS_INTERNAL_ERROR else c.CUDNN_STATUS_SUCCESS;
    }
    fn id(value: anytype) usize {
        const n = @intFromPtr(value) / 16;
        std.debug.assert(n > 0 and n < records.len and records[n].live);
        return n;
    }
    fn read(comptime T: type, ptr: ?*const anyopaque) T {
        return @as(*const T, @ptrCast(@alignCast(ptr))).*;
    }
    fn make(comptime T: type, out: [*c]T, kind: c.cudnnBackendDescriptorType_t) S {
        const status = tick();
        creates[create_count] = calls;
        create_count += 1;
        if (status != c.CUDNN_STATUS_SUCCESS and !nonnull_failure) return status;
        if (calls == null_at) {
            out[0] = null;
            return c.CUDNN_STATUS_SUCCESS;
        }
        const n = next;
        next += 1;
        std.debug.assert(n < records.len);
        records[n] = .{ .kind = kind, .live = true };
        live += 1;
        out[0] = @ptrFromInt(n * 16);
        return status;
    }
    fn drop(value: anytype) S {
        const n = id(value);
        records[n].live = false;
        live -= 1;
        destroyed[destroy_count] = n;
        destroy_count += 1;
        return c.CUDNN_STATUS_SUCCESS;
    }
    fn getVersion() callconv(cc) usize {
        return cudnn_version;
    }
    fn getCudartVersion() callconv(cc) usize {
        return cuda_version;
    }
    fn createHandle(out: [*c]c.cudnnHandle_t) callconv(cc) S {
        return make(c.cudnnHandle_t, out, 0xffffffff);
    }
    fn destroyHandle(handle: c.cudnnHandle_t) callconv(cc) S {
        return drop(handle);
    }
    fn setStream(handle: c.cudnnHandle_t, value: c.cudaStream_t) callconv(cc) S {
        _ = id(handle);
        last_stream = value;
        return tick();
    }
    fn createDescriptor(kind: c.cudnnBackendDescriptorType_t, out: [*c]Desc) callconv(cc) S {
        return make(Desc, out, kind);
    }
    fn destroyDescriptor(desc: Desc) callconv(cc) S {
        return drop(desc);
    }
    fn set(desc: Desc, name: c.cudnnBackendAttributeName_t, kind: c.cudnnBackendAttributeType_t, n: i64, ptr: ?*const anyopaque) callconv(cc) S {
        const status = tick();
        if (status != c.CUDNN_STATUS_SUCCESS) return status;
        const r = &records[id(desc)];
        std.debug.assert(!r.finalized and ptr != null);
        r.attrs += 1;
        switch (name) {
            c.CUDNN_ATTR_TENSOR_UNIQUE_ID => {
                std.debug.assert(kind == c.CUDNN_TYPE_INT64 and n == 1);
                r.uid = read(i64, ptr);
            },
            c.CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT => {
                std.debug.assert(kind == c.CUDNN_TYPE_INT64 and n == 1);
                r.alignment = read(i64, ptr);
            },
            c.CUDNN_ATTR_TENSOR_DATA_TYPE => {
                std.debug.assert(kind == c.CUDNN_TYPE_DATA_TYPE and n == 1);
                r.dtype = read(c.cudnnDataType_t, ptr);
            },
            c.CUDNN_ATTR_TENSOR_DIMENSIONS, c.CUDNN_ATTR_TENSOR_STRIDES => {
                std.debug.assert(kind == c.CUDNN_TYPE_INT64 and n == 4);
                const values = read([4]i64, ptr);
                if (name == c.CUDNN_ATTR_TENSOR_DIMENSIONS) r.dims = values else r.strides = values;
            },
            c.CUDNN_ATTR_TENSOR_IS_VIRTUAL, c.CUDNN_ATTR_TENSOR_IS_BY_VALUE => {
                std.debug.assert(kind == c.CUDNN_TYPE_BOOLEAN and n == 1);
                if (name == c.CUDNN_ATTR_TENSOR_IS_VIRTUAL) r.virtual = read(bool, ptr) else r.by_value = read(bool, ptr);
            },
            c.CUDNN_ATTR_OPERATION_SDPA_FWD_QDESC, c.CUDNN_ATTR_OPERATION_SDPA_FWD_KDESC, c.CUDNN_ATTR_OPERATION_SDPA_FWD_VDESC, c.CUDNN_ATTR_OPERATION_SDPA_FWD_ODESC, c.CUDNN_ATTR_OPERATION_SDPA_FWD_SCALEDESC => {
                std.debug.assert(kind == c.CUDNN_TYPE_BACKEND_DESCRIPTOR and n == 1);
                const index: usize = @intCast(name - c.CUDNN_ATTR_OPERATION_SDPA_FWD_QDESC);
                const slot = if (index == 5) 4 else index;
                r.refs[slot] = id(read(Desc, ptr));
                std.debug.assert(records[r.refs[slot]].finalized);
            },
            c.CUDNN_ATTR_OPERATIONGRAPH_HANDLE, c.CUDNN_ATTR_EXECUTION_PLAN_HANDLE => {
                std.debug.assert(kind == c.CUDNN_TYPE_HANDLE and n == 1);
                _ = id(read(c.cudnnHandle_t, ptr));
            },
            c.CUDNN_ATTR_OPERATIONGRAPH_OPS, c.CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH, c.CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG => {
                std.debug.assert(kind == c.CUDNN_TYPE_BACKEND_DESCRIPTOR and n == 1);
                r.refs[0] = id(read(Desc, ptr));
                std.debug.assert(records[r.refs[0]].finalized);
            },
            c.CUDNN_ATTR_ENGINEHEUR_MODE => {
                std.debug.assert(kind == c.CUDNN_TYPE_HEUR_MODE and n == 1);
                r.heuristic = read(c.cudnnBackendHeurMode_t, ptr);
            },
            c.CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS => {
                std.debug.assert(kind == c.CUDNN_TYPE_INT64 and n == 5);
                std.debug.assert(std.mem.eql(i64, &read([5]i64, ptr), &d.tensor_ids));
            },
            c.CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS => {
                std.debug.assert(kind == c.CUDNN_TYPE_VOID_PTR and n == 5);
                const pointers = read([5]?*const anyopaque, ptr);
                for (pointers, 0..) |value, i| r.pointers[i] = @intFromPtr(value);
            },
            c.CUDNN_ATTR_VARIANT_PACK_WORKSPACE => {
                std.debug.assert(kind == c.CUDNN_TYPE_VOID_PTR and n == 1);
                r.workspace = @intFromPtr(read(?*anyopaque, ptr));
            },
            // A mask/causal/dropout attribute must never be set by this unmasked inference operator.
            else => unreachable,
        }
        return c.CUDNN_STATUS_SUCCESS;
    }
    fn finalize(desc: Desc) callconv(cc) S {
        const status = tick();
        if (status != c.CUDNN_STATUS_SUCCESS) return status;
        const r = &records[id(desc)];
        std.debug.assert(!r.finalized);
        if (r.kind == c.CUDNN_BACKEND_OPERATION_SDPA_FWD_DESCRIPTOR) {
            std.debug.assert(r.attrs == 5);
            for (r.refs) |ref| std.debug.assert(ref != 0);
            if (!direct_supported) return c.CUDNN_STATUS_NOT_SUPPORTED_GRAPH_PATTERN;
        }
        if (r.kind == c.CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR) {
            plans_tried += 1;
            if (!supported[records[r.refs[0]].config_index]) return c.CUDNN_STATUS_NOT_SUPPORTED_SHAPE;
        }
        r.finalized = true;
        return c.CUDNN_STATUS_SUCCESS;
    }
    fn get(desc: Desc, name: c.cudnnBackendAttributeName_t, kind: c.cudnnBackendAttributeType_t, requested: i64, count: [*c]i64, ptr: ?*anyopaque) callconv(cc) S {
        const status = tick();
        if (status != c.CUDNN_STATUS_SUCCESS) return status;
        const r = &records[id(desc)];
        std.debug.assert(r.finalized);
        switch (name) {
            c.CUDNN_ATTR_ENGINEHEUR_RESULTS => {
                std.debug.assert(kind == c.CUDNN_TYPE_BACKEND_DESCRIPTOR);
                if (requested == 0) {
                    std.debug.assert(ptr == null);
                    count[0] = available;
                } else {
                    configs_requested = requested;
                    const actual = returned orelse @min(available, requested);
                    count[0] = actual;
                    const configs: [*]Desc = @ptrCast(@alignCast(ptr));
                    if (actual >= 0 and actual <= requested) for (0..@intCast(actual)) |i| {
                        const record = &records[id(configs[i])];
                        std.debug.assert(record.kind == c.CUDNN_BACKEND_ENGINECFG_DESCRIPTOR and !record.finalized);
                        record.config_index = i;
                        record.finalized = true;
                    };
                    if (substitute_result) configs[0] = null;
                }
            },
            c.CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE => {
                std.debug.assert(kind == c.CUDNN_TYPE_INT64 and requested == 1);
                count[0] = workspace_count;
                @as(*i64, @ptrCast(@alignCast(ptr))).* = needs[records[r.refs[0]].config_index];
            },
            else => unreachable,
        }
        return c.CUDNN_STATUS_SUCCESS;
    }
    fn runtimeWorkspace(handle: c.cudnnHandle_t, plan: Desc, pack: Desc, size: [*c]usize) callconv(cc) S {
        _ = id(handle);
        std.debug.assert(records[id(plan)].finalized and records[id(pack)].finalized);
        runtime_queried = true;
        size[0] = runtime_need;
        return tick();
    }
    fn execute(handle: c.cudnnHandle_t, plan: Desc, pack: Desc) callconv(cc) S {
        _ = id(handle);
        const r = records[id(pack)];
        std.debug.assert(runtime_queried and records[id(plan)].finalized and r.finalized);
        std.debug.assert(@as(*const f32, @ptrFromInt(r.pointers[4])).* == expected_scale);
        const status = tick();
        last_pointers = r.pointers;
        if (status == c.CUDNN_STATUS_SUCCESS) executed = true;
        return status;
    }
    pub const api: d.Api = .{
        .cudnnGetVersion = getVersion,
        .cudnnGetCudartVersion = getCudartVersion,
        .cudnnCreate = createHandle,
        .cudnnDestroy = destroyHandle,
        .cudnnSetStream = setStream,
        .cudnnBackendCreateDescriptor = createDescriptor,
        .cudnnBackendDestroyDescriptor = destroyDescriptor,
        .cudnnBackendSetAttribute = set,
        .cudnnBackendGetAttribute = get,
        .cudnnBackendFinalize = finalize,
        .cudnnGetExecutionPlanWorkspaceSize = runtimeWorkspace,
        .cudnnBackendExecute = execute,
    };
};

pub fn spec() !d.Spec {
    const q: d.Tensor = .{ .dims = .{ 2, 16, 4, 64 }, .strides = .{ 12288, 64, 3072, 1 } };
    const out: d.Tensor = .{ .dims = q.dims, .strides = .{ 4096, 64, 1024, 1 } };
    return .{ .q = q, .k = q, .v = q, .output = out, .scale = 0.125 };
}
pub fn buffers(s: d.Spec, need: usize) !d.Buffers {
    return .{
        .q = .{ .ptr = 0x10000000, .len = try s.q.bytes() },
        .k = .{ .ptr = 0x10000800, .len = try s.k.bytes() },
        .v = .{ .ptr = 0x10001000, .len = try s.v.bytes() },
        .output = .{ .ptr = 0x20000000, .len = try s.output.bytes() },
        .workspace = .{ .ptr = if (need == 0) 0 else 0x30000000, .len = need },
    };
}
const stream: @import("cuda").abi.Stream = @ptrFromInt(0x5000);

test "direct SDPA SDK ABI, required scale and target versions are authoritative" {
    try t.expectEqual(@as(usize, 92400), c.CUDNN_VERSION);
    try t.expectEqual(@as(usize, 13030), d.target_cudart);
    try t.expectEqual(@as(c.cudnnBackendDescriptorType_t, 41), c.CUDNN_BACKEND_OPERATION_SDPA_FWD_DESCRIPTOR);
    try t.expectEqual(@as(c.cudnnBackendAttributeName_t, 2805), c.CUDNN_ATTR_OPERATION_SDPA_FWD_SCALEDESC);
    // Force loader body compilation without calling/opening any cuDNN library.
    std.mem.doNotOptimizeAway(&d.Library.openPath);
    std.mem.doNotOptimizeAway(&d.Library.open);
    std.mem.doNotOptimizeAway(&d.Library.close);
}

test "owning plan submits exact SDK tensors scale variant pack and stream then destroys dependencies in reverse" {
    Mock.reset();
    const s = try spec();
    var op = try d.Attention.init(&Mock.api, s, .{ .workspace_limit = 1024 });
    try t.expectEqual(@as(usize, 11), Mock.live);
    try t.expectEqual(@as(usize, 256), op.workspace_need);
    for (2..6) |n| {
        try t.expectEqual(@as(c.cudnnDataType_t, c.CUDNN_DATA_BFLOAT16), Mock.records[n].dtype);
        try t.expectEqualDeep(s.q.dims, Mock.records[n].dims);
        try t.expect(!Mock.records[n].virtual and !Mock.records[n].by_value);
    }
    try t.expectEqualDeep(s.q.strides, Mock.records[2].strides);
    try t.expectEqualDeep(s.output.strides, Mock.records[5].strides);
    try t.expectEqual(@as(c.cudnnDataType_t, c.CUDNN_DATA_FLOAT), Mock.records[6].dtype);
    try t.expectEqualDeep([4]i64{ 1, 1, 1, 1 }, Mock.records[6].dims);
    try t.expect(Mock.records[6].by_value and !Mock.records[6].virtual);
    try t.expectEqual(@as(c.cudnnBackendHeurMode_t, c.CUDNN_HEUR_MODE_A), Mock.records[9].heuristic);
    const b = try buffers(s, op.workspace_need);
    try op.run(b, stream);
    try t.expect(Mock.executed);
    try t.expectEqual(@as(usize, 0x5000), @intFromPtr(Mock.last_stream));
    try t.expectEqualDeep([5]usize{ b.q.ptr, b.k.ptr, b.v.ptr, b.output.ptr, @intFromPtr(&op.spec.scale) }, Mock.last_pointers);
    try t.expectEqual(@as(usize, 11), Mock.live); // variant pack already released
    op.deinit();
    try t.expectEqual(@as(usize, 0), Mock.live);
    try t.expectEqualSlices(usize, &.{ 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1 }, Mock.destroyed[0..Mock.destroy_count]);
}

test "every initialization API failure and null creation unwinds complete ownership" {
    Mock.reset();
    var baseline = try d.Attention.init(&Mock.api, try spec(), .{ .workspace_limit = 1024 });
    const calls = Mock.calls;
    const creates = Mock.creates;
    const count = Mock.create_count;
    baseline.deinit();
    for (1..calls + 1) |failure| {
        Mock.reset();
        Mock.fail_at = failure;
        try t.expectError(error.CudnnFailed, d.Attention.init(&Mock.api, try spec(), .{ .workspace_limit = 1024 }));
        try t.expectEqual(@as(usize, 0), Mock.live);
    }
    for (creates[0..count]) |step| {
        Mock.reset();
        Mock.null_at = step;
        try t.expectError(error.BadApiReturn, d.Attention.init(&Mock.api, try spec(), .{ .workspace_limit = 1024 }));
        try t.expectEqual(@as(usize, 0), Mock.live);
        Mock.reset();
        Mock.fail_at = step;
        Mock.nonnull_failure = true;
        try t.expectError(error.CudnnFailed, d.Attention.init(&Mock.api, try spec(), .{ .workspace_limit = 1024 }));
        try t.expectEqual(@as(usize, 0), Mock.live);
    }
}

test "unsupported candidate plans and over-budget workspace are skipped within the bounded search" {
    Mock.reset();
    Mock.available = 3;
    Mock.supported[0] = false;
    Mock.needs[1] = 2048;
    Mock.needs[2] = 512;
    Mock.runtime_need = 512;
    var op = try d.Attention.init(&Mock.api, try spec(), .{ .workspace_limit = 1024 });
    try t.expectEqual(@as(usize, 3), Mock.plans_tried);
    try t.expectEqual(@as(usize, 512), op.workspace_need);
    try op.run(try buffers(op.spec, 512), stream);
    op.deinit();
    try t.expectEqual(@as(usize, 0), Mock.live);
    Mock.reset();
    Mock.available = 100000;
    Mock.supported = @splat(false);
    try t.expectError(error.NoPlan, d.Attention.init(&Mock.api, try spec(), .{ .workspace_limit = 1024, .engine_limit = 2 }));
    try t.expectEqual(@as(i64, 2), Mock.configs_requested);
    try t.expectEqual(@as(usize, 2), Mock.plans_tried);
    try t.expectEqual(@as(usize, 0), Mock.live);
    Mock.reset();
    Mock.available = std.math.maxInt(i64);
    Mock.supported = @splat(false);
    try t.expectError(error.NoPlan, d.Attention.init(&Mock.api, try spec(), .{ .workspace_limit = 1024, .engine_limit = 32 }));
    try t.expectEqual(@as(i64, 32), Mock.configs_requested);
    try t.expectEqual(@as(usize, 32), Mock.plans_tried);
    try t.expectEqual(@as(usize, 0), Mock.live);
}

test "direct SDK unsupported operation refuses without a composition fallback or execution" {
    Mock.reset();
    Mock.direct_supported = false;
    try t.expectError(error.UnsupportedOperation, d.Attention.init(&Mock.api, try spec(), .{ .workspace_limit = 1024 }));
    try t.expectEqual(@as(usize, 0), Mock.live);
    try t.expect(!Mock.executed and !Mock.runtime_queried);
}

test "bad heuristic counts substituted handles and workspace values explicitly refuse" {
    for ([_]i64{ -1, 0 }) |count| {
        Mock.reset();
        Mock.available = count;
        const expected = if (count < 0) error.BadApiReturn else error.NoPlan;
        try t.expectError(expected, d.Attention.init(&Mock.api, try spec(), .{ .workspace_limit = 1024 }));
        try t.expectEqual(@as(usize, 0), Mock.live);
    }
    for ([_]i64{ -1, 0, 2 }) |count| {
        Mock.reset();
        Mock.returned = count;
        const expected = if (count == 0) error.NoPlan else error.BadApiReturn;
        try t.expectError(expected, d.Attention.init(&Mock.api, try spec(), .{ .workspace_limit = 1024 }));
        try t.expectEqual(@as(usize, 0), Mock.live);
    }
    Mock.reset();
    Mock.substitute_result = true;
    try t.expectError(error.BadApiReturn, d.Attention.init(&Mock.api, try spec(), .{ .workspace_limit = 1024 }));
    try t.expectEqual(@as(usize, 0), Mock.live);
    Mock.reset();
    Mock.workspace_count = 0;
    try t.expectError(error.BadApiReturn, d.Attention.init(&Mock.api, try spec(), .{ .workspace_limit = 1024 }));
    try t.expectEqual(@as(usize, 0), Mock.live);
    Mock.reset();
    Mock.needs[0] = -1;
    try t.expectError(error.BadApiReturn, d.Attention.init(&Mock.api, try spec(), .{ .workspace_limit = 1024 }));
    try t.expectEqual(@as(usize, 0), Mock.live);
    Mock.reset();
    Mock.needs[0] = 1025;
    try t.expectError(error.NoPlan, d.Attention.init(&Mock.api, try spec(), .{ .workspace_limit = 1024 }));
    try t.expectEqual(@as(usize, 0), Mock.live);
}

test "exact cuDNN and statically-linked CUDA policy rejects older or newer libraries before creation" {
    for ([_]usize{ 92300, 92500, 100000 }) |value| {
        Mock.reset();
        Mock.cudnn_version = value;
        try t.expectError(error.VersionMismatch, d.Attention.init(&Mock.api, try spec(), .{ .workspace_limit = 1024 }));
        try t.expectEqual(@as(usize, 0), Mock.calls);
    }
    for ([_]usize{ 12080, 13020, 13300, 14000 }) |value| {
        Mock.reset();
        Mock.cuda_version = value;
        try t.expectError(error.CudaVersionMismatch, d.Attention.init(&Mock.api, try spec(), .{ .workspace_limit = 1024 }));
        try t.expectEqual(@as(usize, 0), Mock.calls);
    }
}

test "invalid shapes strides precision scales and options refuse before API calls" {
    const valid = try spec();
    var s = valid;
    s.q.dims[0] = 0;
    Mock.reset();
    try t.expectError(error.InvalidShape, d.Attention.init(&Mock.api, s, .{ .workspace_limit = 0 }));
    s = valid;
    s.k.dims[2] = 8;
    try t.expectError(error.InvalidShape, s.check());
    s = valid;
    s.v.strides[2] = -1;
    try t.expectError(error.InvalidShape, s.check());
    s = valid;
    s.q.strides[1] = 1;
    try t.expectError(error.InvalidShape, s.check());
    s = valid;
    s.output.dtype = c.CUDNN_DATA_HALF;
    try t.expectError(error.UnsupportedDType, s.check());
    for ([_]f32{ 0, -1, std.math.inf(f32), std.math.nan(f32) }) |value| {
        s = valid;
        s.scale = value;
        try t.expectError(error.InvalidShape, s.check());
    }
    s = valid;
    s.q.dims = .{ std.math.maxInt(i64), 16, 4, 64 };
    s.q.strides[0] = std.math.maxInt(i64);
    try t.expectError(error.InvalidShape, s.check());
    for ([_]i64{ 0, 1, 3, 512 }) |alignment| {
        s = valid;
        s.q.alignment = alignment;
        try t.expectError(error.InvalidShape, s.check());
    }
    for ([_]u8{ 0, 33 }) |limit| try t.expectError(error.InvalidOptions, d.Attention.init(&Mock.api, valid, .{ .workspace_limit = 0, .engine_limit = limit }));
    try t.expectError(error.InvalidOptions, d.Attention.init(&Mock.api, valid, .{ .workspace_limit = 0, .heuristic = 99 }));
    try t.expectEqual(@as(usize, 0), Mock.calls);
}

test "all active spans alignment wrapping and output or workspace alias fail before variant creation" {
    Mock.reset();
    const s = try spec();
    var op = try d.Attention.init(&Mock.api, s, .{ .workspace_limit = 1024 });
    defer op.deinit();
    const good = try buffers(s, 256);
    const before = Mock.calls;
    inline for (.{ "q", "k", "v", "output", "workspace" }) |field| {
        var b = good;
        @field(b, field).len -= 1;
        try t.expectError(error.InvalidSpan, op.run(b, stream));
        b = good;
        @field(b, field).ptr += 1;
        try t.expectError(error.InvalidSpan, op.run(b, stream));
        b = good;
        @field(b, field).ptr = 0;
        try t.expectError(error.InvalidSpan, op.run(b, stream));
        b = good;
        @field(b, field).ptr = std.math.maxInt(u64) - 255;
        try t.expectError(error.InvalidSpan, op.run(b, stream));
    }
    for ([_]d.Span{ good.q, good.k, good.v, good.workspace }) |other| {
        var b = good;
        b.output.ptr = other.ptr;
        try t.expectError(error.Aliasing, op.run(b, stream));
    }
    for ([_]d.Span{ good.q, good.k, good.v, good.output }) |other| {
        var b = good;
        b.workspace.ptr = other.ptr;
        try t.expectError(error.Aliasing, op.run(b, stream));
    }
    try t.expectEqual(before, Mock.calls);
}

test "every variant pack stream runtime query and execute failure releases the temporary pack" {
    Mock.reset();
    var op = try d.Attention.init(&Mock.api, try spec(), .{ .workspace_limit = 1024 });
    const b = try buffers(op.spec, 256);
    Mock.calls = 0;
    try op.run(b, stream);
    const calls = Mock.calls;
    op.deinit();
    for (1..calls + 1) |failure| {
        Mock.reset();
        op = try d.Attention.init(&Mock.api, try spec(), .{ .workspace_limit = 1024 });
        const owned = Mock.live;
        Mock.calls = 0;
        Mock.fail_at = failure;
        try t.expectError(error.CudnnFailed, op.run(b, stream));
        try t.expectEqual(owned, Mock.live);
        op.deinit();
        try t.expectEqual(@as(usize, 0), Mock.live);
    }
    Mock.reset();
    op = try d.Attention.init(&Mock.api, try spec(), .{ .workspace_limit = 1024 });
    Mock.calls = 0;
    Mock.null_at = 1;
    try t.expectError(error.BadApiReturn, op.run(b, stream));
    op.deinit();
    try t.expectEqual(@as(usize, 0), Mock.live);
    Mock.reset();
    op = try d.Attention.init(&Mock.api, try spec(), .{ .workspace_limit = 1024 });
    Mock.calls = 0;
    Mock.fail_at = 1;
    Mock.nonnull_failure = true;
    try t.expectError(error.CudnnFailed, op.run(b, stream));
    op.deinit();
    try t.expectEqual(@as(usize, 0), Mock.live);
}

test "runtime workspace changes explicitly refuse without submitting attention" {
    Mock.reset();
    var op = try d.Attention.init(&Mock.api, try spec(), .{ .workspace_limit = 1024 });
    defer op.deinit();
    Mock.runtime_need = 257;
    try t.expectError(error.WorkspaceRequirementChanged, op.run(try buffers(op.spec, 256), stream));
    try t.expect(!Mock.executed);
    Mock.runtime_need = 1025;
    try t.expectError(error.WorkspaceLimitExceeded, op.run(try buffers(op.spec, 256), stream));
    try t.expect(!Mock.executed);
}

test "zero workspace is represented by a null variant pointer and repeat calls borrow different buffers" {
    Mock.reset();
    Mock.needs[0] = 0;
    Mock.runtime_need = 0;
    var op = try d.Attention.init(&Mock.api, try spec(), .{ .workspace_limit = 0 });
    defer op.deinit();
    var b = try buffers(op.spec, 0);
    try op.run(b, null);
    try t.expect(Mock.last_stream == null);
    b.output.ptr += 0x10000;
    try op.run(b, stream);
    try t.expectEqual(b.output.ptr, Mock.last_pointers[3]);
}
