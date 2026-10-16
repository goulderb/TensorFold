//! CPU-only ABI mocks. No cuBLASLt library, CUDA driver, context, or device is opened.
//! Enum provenance: CUDA 13.3.73 headers supplied by the coordinator and hash-checked locally.
//! cublasLt.h SHA256 36e8db9e46c95af62bfcd796aa6091d6922adb592a219f6a30957e3cd5ead495
//! library_types.h SHA256 7ccccd9cf42214386170704865517bbbf83702d6bbb5e0c0cba3fc3a48cbd29d

const std = @import("std");
const builtin = @import("builtin");
const lt = @import("cublaslt.zig");
const abi = @import("abi.zig");
const testing = std.testing;
const CPtr = ?*const anyopaque;

const Attribute = struct {
    kind: enum { desc, layout, pref },
    handle: usize,
    id: c_int,
    len: usize,
    bytes: [8]u8 = @splat(0),
    operation: usize,

    fn value(self: Attribute, comptime T: type) T {
        return @bitCast(self.bytes[0..@sizeOf(T)].*);
    }
};
const Matrix = struct { handle: usize, dtype: lt.DataType, rows: u64, cols: u64, ld: i64 };
const Dispatch = struct {
    handle: usize,
    desc: usize,
    alpha: f32,
    a: u64,
    a_layout: usize,
    b: u64,
    b_layout: usize,
    beta: f32,
    c: u64,
    c_layout: usize,
    d: u64,
    d_layout: usize,
    algo: lt.Algo,
    workspace: u64,
    workspace_len: usize,
    stream: usize,
};
const State = struct {
    operations: usize = 0,
    fail_at: usize = 0,
    null_at: usize = 0,
    fail_create_sets_handle: bool = true,
    created: [6]bool = @splat(false),
    destroyed: [6]usize = @splat(0),
    destroy_count: usize = 0,
    compute: c_int = -1,
    scale: lt.DataType = .f16,
    attrs: [32]Attribute = undefined,
    attr_count: usize = 0,
    matrices: [3]Matrix = undefined,
    matrix_count: usize = 0,
    heuristic_operation: usize = 0,
    heuristic_attr_count: usize = 0,
    heuristic_layouts: [4]usize = @splat(0),
    heuristic_request: c_int = 0,
    heuristic_count: c_int = 1,
    heuristic_state: c_int = 0,
    heuristic_workspace: usize = 256,
    check_workspace: usize = 256,
    check_state: c_int = 0,
    check_status: c_int = 0,
    matmul_count: usize = 0,
    dispatch: Dispatch = undefined,
};
var state: State = .{};

fn step() c_int {
    state.operations += 1;
    return if (state.operations == state.fail_at) 7 else 0;
}
fn id(pointer: anytype) usize {
    return if (pointer) |p| @intFromPtr(p) else 0;
}
fn create(comptime T: type, out: *T, resource: usize) c_int {
    const status = step();
    out.* = null;
    if (state.operations != state.null_at and (status == 0 or state.fail_create_sets_handle)) {
        out.* = @ptrFromInt(resource);
        state.created[resource - 1] = true;
    }
    return status;
}
fn destroy(pointer: anytype) c_int {
    const resource = id(pointer);
    std.debug.assert(resource >= 1 and resource <= 6);
    std.debug.assert(state.created[resource - 1]);
    std.debug.assert(state.destroy_count < 6);
    state.destroyed[state.destroy_count] = resource;
    state.destroy_count += 1;
    return 0;
}
fn setAttr(kind: @FieldType(Attribute, "kind"), pointer: anytype, attr: c_int, buffer: CPtr, len: usize) c_int {
    const status = step();
    std.debug.assert(buffer != null and (len == 4 or len == 8));
    var record: Attribute = .{ .kind = kind, .handle = id(pointer), .id = attr, .len = len, .operation = state.operations };
    const bytes: [*]const u8 = @ptrCast(buffer.?);
    @memcpy(record.bytes[0..len], bytes[0..len]);
    state.attrs[state.attr_count] = record;
    state.attr_count += 1;
    return status;
}
fn createHandle(out: *lt.Handle) callconv(.c) c_int {
    return create(lt.Handle, out, 1);
}
fn destroyHandle(handle: lt.Handle) callconv(.c) c_int {
    return destroy(handle);
}
fn version() callconv(.c) usize {
    return 133073;
}
fn statusName(_: c_int) callconv(.c) ?[*:0]const u8 {
    return "expected mock failure";
}
fn createDesc(out: *lt.MatmulDesc, compute: c_int, scale: lt.DataType) callconv(.c) c_int {
    state.compute = compute;
    state.scale = scale;
    return create(lt.MatmulDesc, out, 2);
}
fn destroyDesc(desc: lt.MatmulDesc) callconv(.c) c_int {
    return destroy(desc);
}
fn descAttribute(desc: lt.MatmulDesc, attr: c_int, buffer: CPtr, len: usize) callconv(.c) c_int {
    return setAttr(.desc, desc, attr, buffer, len);
}
fn createLayout(out: *lt.Layout, dtype: lt.DataType, rows: u64, cols: u64, ld: i64) callconv(.c) c_int {
    const resource = state.matrix_count + 3;
    const status = create(lt.Layout, out, resource);
    state.matrices[state.matrix_count] = .{ .handle = resource, .dtype = dtype, .rows = rows, .cols = cols, .ld = ld };
    state.matrix_count += 1;
    return status;
}
fn destroyLayout(layout: lt.Layout) callconv(.c) c_int {
    return destroy(layout);
}
fn layoutAttribute(layout: lt.Layout, attr: c_int, buffer: CPtr, len: usize) callconv(.c) c_int {
    return setAttr(.layout, layout, attr, buffer, len);
}
fn createPref(out: *lt.Preference) callconv(.c) c_int {
    return create(lt.Preference, out, 6);
}
fn destroyPref(pref: lt.Preference) callconv(.c) c_int {
    return destroy(pref);
}
fn prefAttribute(pref: lt.Preference, attr: c_int, buffer: CPtr, len: usize) callconv(.c) c_int {
    return setAttr(.pref, pref, attr, buffer, len);
}
fn heuristic(handle: lt.Handle, desc: lt.MatmulDesc, a: lt.Layout, b: lt.Layout, c: lt.Layout, result_layout: lt.Layout, pref: lt.Preference, requested: c_int, found: [*]lt.Heuristic, count: *c_int) callconv(.c) c_int {
    std.debug.assert(id(handle) == 1 and id(desc) == 2 and id(pref) == 6);
    const status = step();
    state.heuristic_operation = state.operations;
    state.heuristic_attr_count = state.attr_count;
    state.heuristic_layouts = .{ id(a), id(b), id(c), id(result_layout) };
    state.heuristic_request = requested;
    count.* = state.heuristic_count;
    // Deliberately do not initialize a result when count is zero or invalid.
    if (state.heuristic_count == 1) {
        found[0] = .{
            .algo = .{ .data = @splat(0x1122334455667788) },
            .workspace_size = state.heuristic_workspace,
            .state = state.heuristic_state,
            .waves_count = 1,
            .reserved = @splat(0),
        };
    }
    return status;
}
fn algoCheck(_: lt.Handle, _: lt.MatmulDesc, _: lt.Layout, _: lt.Layout, _: lt.Layout, _: lt.Layout, algo: *const lt.Algo, result: *lt.Heuristic) callconv(.c) c_int {
    const status = step();
    result.* = .{ .algo = algo.*, .workspace_size = state.check_workspace, .state = state.check_state, .waves_count = 1, .reserved = @splat(0) };
    return if (status != 0) status else state.check_status;
}
fn matmul(handle: lt.Handle, desc: lt.MatmulDesc, alpha: CPtr, a: u64, a_layout: lt.Layout, b: u64, b_layout: lt.Layout, beta: CPtr, c: u64, c_layout: lt.Layout, result_ptr: u64, d_layout: lt.Layout, algo: ?*const lt.Algo, scratch: u64, workspace_len: usize, stream: abi.Stream) callconv(.c) c_int {
    state.matmul_count += 1;
    state.dispatch = .{
        .handle = id(handle),
        .desc = id(desc),
        .alpha = @as(*const f32, @ptrCast(@alignCast(alpha.?))).*,
        .a = a,
        .a_layout = id(a_layout),
        .b = b,
        .b_layout = id(b_layout),
        .beta = @as(*const f32, @ptrCast(@alignCast(beta.?))).*,
        .c = c,
        .c_layout = id(c_layout),
        .d = result_ptr,
        .d_layout = id(d_layout),
        .algo = algo.?.*,
        .workspace = scratch,
        .workspace_len = workspace_len,
        .stream = id(stream),
    };
    return step();
}
fn library() lt.Library {
    return .{ .lib = undefined, .api = .{
        .cublasLtCreate = createHandle,
        .cublasLtDestroy = destroyHandle,
        .cublasLtGetVersion = version,
        .cublasLtGetStatusName = statusName,
        .cublasLtMatmulDescCreate = createDesc,
        .cublasLtMatmulDescDestroy = destroyDesc,
        .cublasLtMatmulDescSetAttribute = descAttribute,
        .cublasLtMatrixLayoutCreate = createLayout,
        .cublasLtMatrixLayoutDestroy = destroyLayout,
        .cublasLtMatrixLayoutSetAttribute = layoutAttribute,
        .cublasLtMatmulPreferenceCreate = createPref,
        .cublasLtMatmulPreferenceDestroy = destroyPref,
        .cublasLtMatmulPreferenceSetAttribute = prefAttribute,
        .cublasLtMatmulAlgoGetHeuristic = heuristic,
        .cublasLtMatmulAlgoCheck = algoCheck,
        .cublasLtMatmul = matmul,
    } };
}
fn span(ptr: u64, len: usize) lt.DeviceSpan {
    return .{ .ptr = ptr, .len = len };
}
const bias = lt.DeviceSpan{ .ptr = 0x100002, .len = 10 }; // n=5 BF16, intentionally not 256-aligned
const x = lt.DeviceSpan{ .ptr = 0x200002, .len = 42 }; // m=3,k=7
const w = lt.DeviceSpan{ .ptr = 0x300002, .len = 70 }; // n=5,k=7
const d = lt.DeviceSpan{ .ptr = 0x400002, .len = 30 }; // m=3,n=5
const workspace = lt.DeviceSpan{ .ptr = 0x500000, .len = 1024 };

fn expectAttribute(index: usize, kind: @FieldType(Attribute, "kind"), handle: usize, attr: c_int, comptime T: type, value: T) !void {
    const record = state.attrs[index];
    try testing.expectEqual(kind, record.kind);
    try testing.expectEqual(handle, record.handle);
    try testing.expectEqual(attr, record.id);
    try testing.expectEqual(@sizeOf(T), record.len);
    try testing.expectEqual(value, record.value(T));
    try testing.expect(record.operation < state.heuristic_operation);
}
fn expectCleaned() !void {
    var expected: [6]usize = undefined;
    var count: usize = 0;
    var resource: usize = 6;
    while (resource > 0) : (resource -= 1) {
        if (state.created[resource - 1]) {
            expected[count] = resource;
            count += 1;
        }
    }
    try testing.expectEqualSlices(usize, expected[0..count], state.destroyed[0..state.destroy_count]);
}

test "cublaslt target CUDA13.3 enums and ABI sizes use independent header literals" {
    try testing.expectEqual(@as(c_int, 7), lt.desc_epilogue);
    try testing.expectEqual(@as(u32, 4), @backingInt(lt.Epilogue.bias));
    try testing.expectEqual(@as(c_int, 8), lt.desc_bias_pointer);
    try testing.expectEqual(@as(c_int, 26), lt.desc_bias_data_type);
    try testing.expectEqual(@as(c_int, 14), @backingInt(lt.DataType.bf16));
    try testing.expectEqual(@as(c_int, 68), lt.compute_32f);
    try testing.expectEqual(@as(c_int, 5), lt.pref_min_alignment_a_bytes);
    try testing.expectEqual(@as(c_int, 6), lt.pref_min_alignment_b_bytes);
    try testing.expectEqual(@as(c_int, 7), lt.pref_min_alignment_c_bytes);
    try testing.expectEqual(@as(c_int, 8), lt.pref_min_alignment_d_bytes);
    try testing.expectEqual(@as(usize, 64), @sizeOf(lt.Algo));
    try testing.expectEqual(@as(usize, 96), @sizeOf(lt.Heuristic));
    try testing.expectEqual(@as(usize, 8), @sizeOf(abi.DevicePtr));
    try testing.expectEqual(@sizeOf(?*const anyopaque), @sizeOf(abi.DevicePtr));
    try testing.expectEqual(@as(usize, 64), @offsetOf(lt.Heuristic, "workspace_size"));
    try testing.expectEqual(@as(usize, 72), @offsetOf(lt.Heuristic, "state"));
    try testing.expectEqual(@as(usize, 76), @offsetOf(lt.Heuristic, "waves_count"));
    try testing.expectEqual(@as(usize, 80), @offsetOf(lt.Heuristic, "reserved"));
    try testing.expectEqual(@as(usize, 16), @typeInfo(lt.Api).@"struct".field_names.len);
}

test "cublaslt bias dtype pointer payload and column-major layouts precede heuristic" {
    state = .{};
    const lib = library();
    var linear = try lt.Linear.initEpilogueChecked(&lib, 3, 5, 7, .bf16, 512, .bias, bias);
    defer linear.deinit();
    try testing.expectEqual(@as(c_int, 68), state.compute);
    try testing.expectEqual(@as(c_int, 0), @backingInt(state.scale));
    try testing.expectEqual(@as(usize, 13), state.attr_count);
    try testing.expectEqual(state.attr_count, state.heuristic_attr_count);
    try expectAttribute(0, .desc, 2, 3, c_int, 1); // transa=T
    try expectAttribute(1, .desc, 2, 4, c_int, 0); // transb=N
    try expectAttribute(2, .desc, 2, 7, u32, 4);
    try expectAttribute(3, .desc, 2, 26, c_int, 14);
    try expectAttribute(4, .desc, 2, 8, u64, bias.ptr); // pointer value, not address of host storage
    for (0..3) |i| try expectAttribute(5 + i, .layout, 3 + i, 1, c_int, 0);
    try expectAttribute(8, .pref, 6, 1, u64, 512);
    for (0..4) |i| try expectAttribute(9 + i, .pref, 6, @intCast(5 + i), u32, 2);
    try testing.expectEqual(@as(usize, 3), state.matrix_count);
    try testing.expectEqual(Matrix{ .handle = 3, .dtype = .bf16, .rows = 7, .cols = 5, .ld = 7 }, state.matrices[0]);
    try testing.expectEqual(Matrix{ .handle = 4, .dtype = .bf16, .rows = 7, .cols = 3, .ld = 7 }, state.matrices[1]);
    try testing.expectEqual(Matrix{ .handle = 5, .dtype = .bf16, .rows = 5, .cols = 3, .ld = 5 }, state.matrices[2]);
    for (state.matrices) |matrix| try testing.expectEqual(@as(c_int, 14), @backingInt(matrix.dtype));
    try testing.expectEqualSlices(usize, &.{ 3, 4, 5, 5 }, &state.heuristic_layouts);
    try testing.expectEqual(@as(c_int, 1), state.heuristic_request);
    try testing.expectEqual(@as(usize, 256), linear.workspace_need);
    try testing.expectEqual(bias, linear.bias.?);
}

test "cublaslt checked bias dispatch is one fused matmul with fp32 alpha beta and BF16 D" {
    state = .{};
    const lib = library();
    var linear = try lt.Linear.initEpilogueChecked(&lib, 3, 5, 7, .bf16, 512, .bias, bias);
    const attrs_before = state.attr_count;
    const stream: abi.Stream = @ptrFromInt(0x600000);
    try linear.runChecked(x, w, d, workspace, stream);
    try testing.expectEqual(@as(usize, 1), state.matmul_count);
    try testing.expectEqual(attrs_before, state.attr_count); // no post-init mutation
    try testing.expectEqual(Dispatch{
        .handle = 1,
        .desc = 2,
        .alpha = 1,
        .a = w.ptr,
        .a_layout = 3,
        .b = x.ptr,
        .b_layout = 4,
        .beta = 0,
        .c = d.ptr,
        .c_layout = 5,
        .d = d.ptr,
        .d_layout = 5,
        .algo = .{ .data = @splat(0x1122334455667788) },
        .workspace = workspace.ptr,
        .workspace_len = 256,
        .stream = 0x600000,
    }, state.dispatch);
    linear.deinit();
    try expectCleaned();
    try testing.expectEqual(@as(usize, 6), state.destroy_count);
}

test "cublaslt bias pointer-only legacy dispatch cannot bypass span checks" {
    state = .{};
    const lib = library();
    var linear = try lt.Linear.initEpilogueChecked(&lib, 3, 5, 7, .bf16, 512, .bias, bias);
    defer linear.deinit();
    try testing.expectError(error.Invalid, linear.run(x.ptr, w.ptr, d.ptr, workspace.ptr, workspace.len, null));
    try testing.expectEqual(@as(usize, 0), state.matmul_count);
}

test "cublaslt dimensions and byte products refuse zero signed overflow and wrap before any API" {
    const shapes = [_][3]u64{
        .{ 0, 5, 7 },                                 .{ 3, 0, 7 },                    .{ 3, 5, 0 },
        .{ @as(u64, 1) << 63, 1, 1 },                 .{ 1, @as(u64, 1) << 63, 1 },    .{ 1, 1, @as(u64, 1) << 63 },
        .{ std.math.maxInt(i64), 1, 2 },              .{ 1, std.math.maxInt(i64), 2 }, .{ std.math.maxInt(i64), 2, 1 },
        .{ @as(u64, 1) << 32, 1, @as(u64, 1) << 32 },
    };
    const lib = library();
    for (shapes) |shape| {
        state = .{};
        try testing.expectError(error.Invalid, lt.Linear.initEpilogueChecked(&lib, shape[0], shape[1], shape[2], .bf16, 512, .bias, bias));
        try testing.expectEqual(@as(usize, 0), state.operations);
        try testing.expectError(error.Invalid, lt.Linear.init(&lib, shape[0], shape[1], shape[2], .f32, 512));
        try testing.expectEqual(@as(usize, 0), state.operations);
    }
}

test "cublaslt borrowed BF16 bias requires nonzero aligned full span and nonwrapping capacity" {
    const invalid = [_]lt.DeviceSpan{
        span(0, 10),       span(bias.ptr + 1, 10),             span(bias.ptr, 9),
        span(bias.ptr, 0), span(std.math.maxInt(u64) - 1, 10), span(bias.ptr, std.math.maxInt(usize)),
    };
    const lib = library();
    for (invalid) |bad| {
        state = .{};
        try testing.expectError(error.Invalid, lt.Linear.initEpilogueChecked(&lib, 3, 5, 7, .bf16, 512, .bias, bad));
        try testing.expectEqual(@as(usize, 0), state.operations);
    }
    state = .{};
    // A larger allocation is valid; only its first n BF16 elements are read.
    var linear = try lt.Linear.initEpilogueChecked(&lib, 3, 5, 7, .bf16, 512, .bias, span(bias.ptr, 128));
    linear.deinit();
    try expectCleaned();
}

test "cublaslt every biased initializer failure unwinds exactly its created owners in reverse" {
    const lib = library();
    state = .{};
    var success = try lt.Linear.initEpilogueChecked(&lib, 3, 5, 7, .bf16, 512, .bias, bias);
    const total = state.operations;
    success.deinit();
    try testing.expectEqual(@as(usize, 20), total);
    for (1..total + 1) |fault| {
        for ([_]bool{ false, true }) |sets_handle| {
            state = .{ .fail_at = fault, .fail_create_sets_handle = sets_handle };
            try testing.expectError(error.CublasFailed, lt.Linear.initEpilogueChecked(&lib, 3, 5, 7, .bf16, 512, .bias, bias));
            try testing.expectEqual(fault, state.operations);
            try expectCleaned();
        }
    }
}

test "cublaslt successful-null handle descriptor layouts and preference refuse and unwind" {
    const lib = library();
    const create_operations = [_]usize{ 1, 2, 8, 9, 10, 14 };
    for (create_operations) |fault| {
        state = .{ .null_at = fault };
        try testing.expectError(error.CublasFailed, lt.Linear.initEpilogueChecked(&lib, 3, 5, 7, .bf16, 512, .bias, bias));
        try testing.expectEqual(fault, state.operations);
        try expectCleaned();
    }
}

test "cublaslt heuristic count status and workspace above limit refuse without reading absent result" {
    const lib = library();
    const variants = [_]State{
        .{ .heuristic_count = 0 }, .{ .heuristic_count = -1 },      .{ .heuristic_count = 2 },
        .{ .heuristic_state = 7 }, .{ .heuristic_workspace = 513 },
    };
    for (variants) |variant| {
        state = variant;
        try testing.expectError(error.NoAlgorithm, lt.Linear.initEpilogueChecked(&lib, 3, 5, 7, .bf16, 512, .bias, bias));
        try expectCleaned();
        try testing.expectEqual(@as(usize, 6), state.destroy_count);
        try testing.expectEqual(@as(usize, 0), state.matmul_count);
    }
}

test "cublaslt checked X W D and workspace reject short null misaligned and wrapped spans before dispatch" {
    state = .{};
    const lib = library();
    var linear = try lt.Linear.initEpilogueChecked(&lib, 3, 5, 7, .bf16, 512, .bias, bias);
    defer linear.deinit();
    const valid = [_]lt.DeviceSpan{ x, w, d, workspace };
    for (0..valid.len) |index| {
        const need = [_]usize{ 42, 70, 30, 256 };
        const alignment: u64 = if (index == 3) 256 else 2;
        const invalid = [_]lt.DeviceSpan{
            span(valid[index].ptr, need[index] - 1),        span(0, valid[index].len),
            span(valid[index].ptr + 1, valid[index].len),   span(std.math.maxInt(u64) - (alignment - 1), valid[index].len),
            span(valid[index].ptr, std.math.maxInt(usize)),
        };
        for (invalid) |bad| {
            var buffers = valid;
            buffers[index] = bad;
            const ops = state.operations;
            try testing.expectError(error.Invalid, linear.runChecked(buffers[0], buffers[1], buffers[2], buffers[3], null));
            try testing.expectEqual(ops, state.operations);
            try testing.expectEqual(@as(usize, 0), state.matmul_count);
        }
    }
}

test "cublaslt checked output and workspace may not alias X W bias or each other" {
    state = .{};
    const lib = library();
    var linear = try lt.Linear.initEpilogueChecked(&lib, 3, 5, 7, .bf16, 512, .bias, bias);
    defer linear.deinit();
    for ([_]u64{ x.ptr, w.ptr, bias.ptr, workspace.ptr, x.ptr + x.len - 2 }) |ptr| {
        try testing.expectError(error.Invalid, linear.runChecked(x, w, span(ptr, d.len), workspace, null));
    }
    for ([_]u64{ x.ptr & ~@as(u64, 255), w.ptr & ~@as(u64, 255), bias.ptr & ~@as(u64, 255), d.ptr & ~@as(u64, 255) }) |ptr| {
        try testing.expectError(error.Invalid, linear.runChecked(x, w, d, span(ptr, 256), null));
    }
    try testing.expectEqual(@as(usize, 0), state.matmul_count);
}

test "cublaslt exact workspace limit and zero-workspace algorithms are supported" {
    const lib = library();
    for ([_]usize{ 0, 256 }) |need| {
        state = .{ .heuristic_workspace = need };
        var linear = try lt.Linear.initEpilogueChecked(&lib, 3, 5, 7, .bf16, need, .bias, bias);
        const ws = if (need == 0) span(0, 0) else span(workspace.ptr, need);
        try linear.runChecked(x, w, d, ws, null);
        try testing.expectEqual(need, state.dispatch.workspace_len);
        try testing.expectEqual(ws.ptr, state.dispatch.workspace);
        linear.deinit();
        try expectCleaned();
    }
    state = .{ .heuristic_workspace = 0 };
    var linear = try lt.Linear.initEpilogueChecked(&lib, 3, 5, 7, .bf16, 0, .bias, bias);
    defer linear.deinit();
    try linear.runChecked(x, w, d, workspace, null);
    try testing.expectEqual(@as(u64, 0), state.dispatch.workspace);
    try testing.expectEqual(@as(usize, 0), state.dispatch.workspace_len);
}

test "cublaslt unbiased caller signature output layouts and legacy dispatch remain intact" {
    const lib = library();
    for ([_]lt.DataType{ .bf16, .f32, .f16 }) |out| {
        state = .{};
        var linear = try lt.Linear.init(&lib, 3, 5, 7, out, 512);
        try testing.expectEqual(@as(usize, 3), state.attr_count);
        try testing.expectEqual(@as(usize, 10), state.operations);
        try expectAttribute(0, .desc, 2, 3, c_int, 1);
        try expectAttribute(1, .desc, 2, 4, c_int, 0);
        try expectAttribute(2, .pref, 6, 1, u64, 512);
        try testing.expectEqual(out, state.matrices[2].dtype);
        try testing.expect(linear.bias == null);
        try linear.run(0x200000, 0x300000, 0x400000, workspace.ptr, workspace.len, null);
        try testing.expectEqual(@as(usize, 1), state.matmul_count);
        try testing.expectEqual(workspace.len, state.dispatch.workspace_len);
        try testing.expectEqual(@as(f32, 1), state.dispatch.alpha);
        try testing.expectEqual(@as(f32, 0), state.dispatch.beta);
        linear.deinit();
        try expectCleaned();
    }
}

test "cublaslt unbiased create failures and successful-null results also unwind" {
    const lib = library();
    for (1..11) |fault| {
        state = .{ .fail_at = fault };
        try testing.expectError(error.CublasFailed, lt.Linear.init(&lib, 3, 5, 7, .bf16, 512));
        try expectCleaned();
    }
    for ([_]usize{ 1, 2, 5, 6, 7, 8 }) |fault| {
        state = .{ .null_at = fault };
        try testing.expectError(error.CublasFailed, lt.Linear.init(&lib, 3, 5, 7, .bf16, 512));
        try expectCleaned();
    }
    state = .{ .heuristic_workspace = 513 };
    try testing.expectError(error.NoAlgorithm, lt.Linear.init(&lib, 3, 5, 7, .bf16, 512));
    try expectCleaned();
}

test "cublaslt unbiased checked path honors default heuristic matrix alignment and fp32 span size" {
    state = .{};
    const lib = library();
    var linear = try lt.Linear.init(&lib, 3, 5, 7, .f32, 512);
    defer linear.deinit();
    try testing.expectError(error.Invalid, linear.runChecked(x, w, d, workspace, null));
    try testing.expectError(error.Invalid, linear.runChecked(span(0x200000, 42), span(0x300000, 70), span(0x400000, 59), workspace, null));
    try testing.expectEqual(@as(usize, 0), state.matmul_count);
    try linear.runChecked(span(0x200000, 42), span(0x300000, 70), span(0x400000, 60), workspace, null);
    try testing.expectEqual(@as(usize, 1), state.matmul_count);
}

test "cublaslt matmul errors propagate with owner resources still live until deinit" {
    state = .{};
    const lib = library();
    var linear = try lt.Linear.initEpilogueChecked(&lib, 3, 5, 7, .bf16, 512, .bias, bias);
    state.fail_at = state.operations + 1;
    try testing.expectError(error.CublasFailed, linear.runChecked(x, w, d, workspace, null));
    try testing.expectEqual(@as(usize, 0), state.destroy_count);
    try testing.expectEqual(@as(usize, 1), state.matmul_count);
    linear.deinit();
    try expectCleaned();
}

test "cublaslt legacy unbiased dispatch checks heuristic alignment null wrap and workspace capacity" {
    state = .{};
    const lib = library();
    var linear = try lt.Linear.init(&lib, 129, 5, 7, .f32, 512);
    defer linear.deinit();
    const valid = [_]u64{ 0x200000, 0x300000, 0x400000, workspace.ptr };
    for (0..valid.len) |index| {
        for ([_]u64{ 0, valid[index] + 2 }) |bad| {
            var pointers = valid;
            pointers[index] = bad;
            try testing.expectError(error.Invalid, linear.run(pointers[0], pointers[1], pointers[2], pointers[3], 256, null));
        }
    }
    const last_aligned = std.math.maxInt(u64) - 255;
    try testing.expectError(error.Invalid, linear.run(last_aligned, valid[1], valid[2], valid[3], 256, null));
    try testing.expectError(error.Invalid, linear.run(valid[0], valid[1], last_aligned, valid[3], 256, null));
    try testing.expectError(error.Invalid, linear.run(valid[0], valid[1], valid[2], last_aligned, 256, null));
    try testing.expectError(error.Invalid, linear.run(valid[0], valid[1], valid[2], valid[3], 255, null));
    try testing.expectError(error.Invalid, linear.run(valid[0], valid[1], valid[2], valid[3], std.math.maxInt(usize), null));
    try testing.expectEqual(@as(usize, 0), state.matmul_count);
    try linear.run(valid[0], valid[1], valid[2], valid[3], 256, null);
    try testing.expectEqual(@as(usize, 1), state.matmul_count);
}

test "cublaslt unit dimensions and adjacent disjoint spans dispatch safely" {
    state = .{ .heuristic_workspace = 0 };
    const lib = library();
    // Two-byte alignment and half-open overlap checks permit adjacent BF16 vectors.
    var linear = try lt.Linear.initEpilogueChecked(&lib, 1, 1, 1, .bf16, 0, .bias, span(0x100002, 2));
    defer linear.deinit();
    try linear.runChecked(span(0x100004, 2), span(0x100006, 2), span(0x100008, 2), span(0, 0), null);
    try testing.expectEqual(@as(usize, 1), state.matmul_count);
    try testing.expectEqual(@as(u64, 1), state.matrices[2].rows);
    try testing.expectEqual(@as(u64, 1), state.matrices[2].cols);
}

test "cublaslt no-GPU Windows availability and public body compilation" {
    // Windows open() has an empty name list, so this cannot load a library or touch CUDA.
    if (builtin.os.tag == .windows) try testing.expectError(error.LibraryUnavailable, lt.Library.open());
    comptime {
        testing.refAllDecls(lt);
        testing.refAllDecls(lt.Linear);
        testing.refAllDecls(lt.Library);
    }
    // Escape function addresses so bodies are compiled even with ReleaseFast dead-code elimination.
    // This does not invoke any of these functions.
    std.mem.doNotOptimizeAway(&lt.Library.open);
    std.mem.doNotOptimizeAway(&lt.Library.openPath);
    std.mem.doNotOptimizeAway(&lt.Library.close);
}

test "cublaslt optional upstream epilogue keeps fusion and fallback contracts" {
    const lib = library();
    const variants = [_]State{
        .{}, .{ .check_workspace = 513 }, .{ .check_status = 7 }, .{ .check_state = 7 },
    };
    for (variants, 0..) |variant, index| {
        state = variant;
        var linear = try lt.Linear.initEpilogue(&lib, 3, 5, 7, .bf16, 512, .bias);
        try testing.expectEqual(if (index == 0) lt.Epilogue.bias else lt.Epilogue.none, linear.epilogue);
        if (index == 0) {
            try linear.runBias(0x200000, 0x300000, bias.ptr, 0x400000, workspace.ptr, workspace.len, null);
            const attr = state.attrs[state.attr_count - 1];
            try testing.expectEqual(lt.desc_bias_pointer, attr.id);
            try testing.expectEqual(bias.ptr, attr.value(u64));
        } else {
            try testing.expectError(error.Invalid, linear.runBias(0x200000, 0x300000, bias.ptr, 0x400000, workspace.ptr, workspace.len, null));
            const attr = state.attrs[state.attr_count - 1];
            try testing.expectEqual(lt.desc_epilogue, attr.id);
            try testing.expectEqual(@as(u32, 1), attr.value(u32));
            try linear.run(0x200000, 0x300000, 0x400000, workspace.ptr, workspace.len, null);
        }
        try testing.expectEqual(@as(usize, 1), state.matmul_count);
        linear.deinit();
        try expectCleaned();
    }
}

test "cublaslt bound epilogue rejects mismatched dtype and cannot mutate its bias through runBias" {
    state = .{};
    const lib = library();
    try testing.expectError(error.Invalid, lt.Linear.initEpilogueChecked(&lib, 3, 5, 7, .f32, 512, .bias, bias));
    try testing.expectError(error.Invalid, lt.Linear.initEpilogueChecked(&lib, 3, 5, 7, .bf16, 512, .none, bias));
    try testing.expectEqual(@as(usize, 0), state.operations);
    var linear = try lt.Linear.initEpilogueChecked(&lib, 3, 5, 7, .bf16, 512, .bias, bias);
    defer linear.deinit();
    const operations = state.operations;
    try testing.expectError(error.Invalid, linear.runBias(x.ptr, w.ptr, bias.ptr, d.ptr, workspace.ptr, workspace.len, null));
    try testing.expectEqual(operations, state.operations);
    try testing.expectEqual(lt.Epilogue.bias, linear.epilogue);
}

test "cublaslt optional fusion raises workspace need and attribute failures unwind" {
    const lib = library();
    state = .{ .check_workspace = 512 };
    var linear = try lt.Linear.initEpilogue(&lib, 3, 5, 7, .bf16, 512, .bias);
    try testing.expectEqual(@as(usize, 512), linear.workspace_need);
    linear.deinit();
    try expectCleaned();
    for ([_]usize{ 11, 13 }) |fault| {
        state = .{ .fail_at = fault, .check_state = 7 };
        try testing.expectError(error.CublasFailed, lt.Linear.initEpilogue(&lib, 3, 5, 7, .bf16, 512, .bias));
        try expectCleaned();
    }
}
