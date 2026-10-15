const std = @import("std");
const cuda = @import("cuda");
const ops = @import("glm_vision_ops.zig");
const testing = std.testing;
const Mock = struct {
    var live: bool = false;
    var loads: usize = 0;
    var unloads: usize = 0;
    var lookups: usize = 0;
    var launches: usize = 0;
    var missing: usize = 0;
    var null_symbol: usize = 0;
    var null_module: bool = false;
    var fail_load: bool = false;
    var fail_launch: bool = false;
    var init_calls: usize = 0;
    var last_op: usize = 0;
    var grid: u32 = 0;
    var threads: u32 = 0;
    var count: usize = 0;
    var args: [13]u64 = undefined;
    fn reset() void {
        live = false;
        loads = 0;
        unloads = 0;
        lookups = 0;
        launches = 0;
        missing = 0;
        null_symbol = 0;
        null_module = false;
        fail_load = false;
        fail_launch = false;
        init_calls = 0;
    }
    fn init(_: c_uint) callconv(.c) cuda.abi.Result {
        init_calls += 1;
        return cuda.abi.error_not_ready;
    }
    fn load(out: *cuda.abi.Module, _: ?*const anyopaque) callconv(.c) cuda.abi.Result {
        loads += 1;
        if (fail_load) return cuda.abi.error_not_ready;
        out.* = if (null_module) null else @ptrFromInt(0x1000);
        live = !null_module;
        return 0;
    }
    fn unload(m: cuda.abi.Module) callconv(.c) cuda.abi.Result {
        std.debug.assert(live and m != null);
        live = false;
        unloads += 1;
        return 0;
    }
    fn function(out: *cuda.abi.Function, m: cuda.abi.Module, name: [*:0]const u8) callconv(.c) cuda.abi.Result {
        std.debug.assert(live and m != null);
        lookups += 1;
        std.debug.assert(std.mem.eql(u8, std.mem.span(name), ops.symbols[lookups - 1]));
        if (lookups == missing) return cuda.abi.error_not_found;
        out.* = if (lookups == null_symbol) null else @ptrFromInt(0x8000 + (lookups - 1) * 16);
        return 0;
    }
    fn launch(f: cuda.abi.Function, gx: c_uint, gy: c_uint, gz: c_uint, bx: c_uint, by: c_uint, bz: c_uint, shared: c_uint, _: cuda.abi.Stream, params: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
        std.debug.assert(live and gy == 1 and gz == 1 and by == 1 and bz == 1 and shared == 0);
        launches += 1;
        last_op = (@intFromPtr(f) - 0x8000) / 16;
        grid = gx;
        threads = bx;
        const pointers = [_]usize{ 2, 3, 8, 3, 3, 4 };
        const counts = [_]usize{ 3, 6, 13, 4, 5, 7 };
        count = counts[last_op];
        for (0..count) |i| args[i] = if (i < pointers[last_op]) @as(*align(1) const u64, @ptrCast(params.?[i].?)).* else @as(*align(1) const u32, @ptrCast(params.?[i].?)).*;
        return if (fail_launch) cuda.abi.error_not_ready else 0;
    }
    fn driver() cuda.Driver {
        var api: cuda.abi.Api = undefined;
        api.cuInit = init;
        api.cuModuleLoadData = load;
        api.cuModuleUnload = unload;
        api.cuModuleGetFunction = function;
        api.cuLaunchKernel = launch;
        return .{ .lib = undefined, .api = api };
    }
};
const image: [8]u8 align(8) = @splat(0);
fn bf(ptr: u64, n: usize) ops.Span {
    return .{ .ptr = ptr, .len = n * 2 };
}
fn fp(ptr: u64, n: usize) ops.Span {
    return .{ .ptr = ptr, .len = n * 4, .dtype = .f32 };
}

test "vision operators native vision image owns its module and resolves every operator atomically" {
    Mock.reset();
    var d = Mock.driver();
    var k = try ops.Kernels.load(&d, &image);
    try testing.expect(Mock.live);
    try testing.expectEqual(@as(usize, 6), Mock.lookups);
    k.deinit();
    try testing.expect(!Mock.live);
    try testing.expectEqual(@as(usize, 1), Mock.unloads);
    try testing.expectEqual(@as(usize, 0), Mock.init_calls);
    for (1..7) |missing| {
        Mock.reset();
        Mock.missing = missing;
        try testing.expectError(error.MissingVisionKernel, ops.Kernels.load(&d, &image));
        try testing.expect(!Mock.live);
        try testing.expectEqual(@as(usize, 1), Mock.unloads);
    }
    for (1..7) |null_symbol| {
        Mock.reset();
        Mock.null_symbol = null_symbol;
        try testing.expectError(error.MissingVisionKernel, ops.Kernels.load(&d, &image));
        try testing.expectEqual(@as(usize, 1), Mock.unloads);
        try testing.expectEqual(null_symbol, Mock.lookups);
    }
    Mock.reset();
    Mock.null_module = true;
    try testing.expectError(error.InvalidVisionModule, ops.Kernels.load(&d, &image));
    try testing.expectEqual(@as(usize, 0), Mock.unloads);
    Mock.reset();
    Mock.fail_load = true;
    try testing.expectError(error.NotReady, ops.Kernels.load(&d, &image));
    try testing.expectEqual(@as(usize, 0), Mock.unloads);
    Mock.reset();
    try testing.expectError(error.MissingVisionKernelImage, ops.Kernels.load(&d, &.{}));
    try testing.expectError(error.MissingVisionKernelImage, ops.Kernels.load(&d, image[1..]));
    try testing.expectEqual(@as(usize, 0), Mock.loads);
}
test "vision operators all six native launch bodies marshal exact ABI shapes, floats and group sequence geometry" {
    Mock.reset();
    var d = Mock.driver();
    var k = try ops.Kernels.load(&d, &image);
    defer k.deinit();
    const s: cuda.Stream = .{ .d = &d, .handle = null };
    try k.cast(s, 17, fp(0x100000, 17), bf(0x200000, 17));
    try testing.expectEqual(@as(usize, 0), Mock.last_op);
    try testing.expectEqualSlices(u64, &.{ 0x100000, 0x200000, 17 }, Mock.args[0..Mock.count]);
    const matrix: ops.Matrix = .{ .rows = 4, .cols = 8 };
    try k.rms(s, matrix, 1e-6, bf(0x100000, 32), bf(0x200000, 8), bf(0x300000, 32));
    try testing.expectEqual(@as(usize, 1), Mock.last_op);
    try testing.expectEqualSlices(u64, &.{ 0x100000, 0x200000, 0x300000, 4, 8, @as(u32, @bitCast(@as(f32, 1e-6))) }, Mock.args[0..Mock.count]);
    try testing.expectEqual(@as(u32, 4), Mock.grid);
    try testing.expectEqual(@as(u32, 256), Mock.threads);
    try testing.expectEqual(@as(u64, @as(u32, @bitCast(@as(f32, 1e-6)))), Mock.args[5]);
    const qkv: ops.Qkv = .{ .groups = 2, .sequence = 4, .heads = 2, .dim = 4 };
    try k.qkvRope(s, qkv, 1e-6, bf(0x100000, 192), bf(0x200000, 4), bf(0x300000, 4), fp(0x400000, 16), fp(0x500000, 16), bf(0x600000, 64), bf(0x700000, 64), bf(0x800000, 64));
    try testing.expectEqual(@as(usize, 2), Mock.last_op);
    try testing.expectEqualSlices(u64, &.{ 0x100000, 0x200000, 0x300000, 0x400000, 0x500000, 0x600000, 0x700000, 0x800000, 8, 4, 2, 4, @as(u32, @bitCast(@as(f32, 1e-6))) }, Mock.args[0..Mock.count]);
    try testing.expectEqual(@as(u32, 16), Mock.grid);
    try testing.expectEqual(@as(u32, 128), Mock.threads);
    try k.residual(s, 32, bf(0x100000, 32), bf(0x200000, 32), bf(0x100000, 32));
    try testing.expectEqual(@as(usize, 3), Mock.last_op);
    try testing.expectEqualSlices(u64, &.{ 0x100000, 0x200000, 0x100000, 32 }, Mock.args[0..Mock.count]);
    try k.swiglu(s, 32, 10, bf(0x100000, 32), bf(0x200000, 32), bf(0x300000, 32));
    try testing.expectEqual(@as(usize, 4), Mock.last_op);
    try testing.expectEqualSlices(u64, &.{ 0x100000, 0x200000, 0x300000, 32, @as(u32, @bitCast(@as(f32, 10))) }, Mock.args[0..Mock.count]);
    try testing.expectEqual(@as(u64, @as(u32, @bitCast(@as(f32, 10)))), Mock.args[4]);
    try k.projectorNormGelu(s, matrix, bf(0x100000, 32), bf(0x200000, 8), bf(0x300000, 8), bf(0x400000, 32));
    try testing.expectEqual(@as(usize, 5), Mock.last_op);
    try testing.expectEqualSlices(u64, &.{ 0x100000, 0x200000, 0x300000, 0x400000, 4, 8, @as(u32, @bitCast(@as(f32, 1e-5))) }, Mock.args[0..Mock.count]);
    try testing.expectEqual(@as(u64, @as(u32, @bitCast(@as(f32, 1e-5)))), Mock.args[6]);
    try testing.expectEqual(@as(usize, 6), Mock.launches);
    try testing.expectEqual(@as(usize, 0), Mock.init_calls);
}
test "vision operators unsafe geometry, scalar, dtype, pointer span, alias or stream refuses before launch" {
    Mock.reset();
    var d = Mock.driver();
    var k = try ops.Kernels.load(&d, &image);
    defer k.deinit();
    const s: cuda.Stream = .{ .d = &d, .handle = null };
    try testing.expectError(error.InvalidOperatorShape, k.cast(s, 0, fp(0x100000, 1), bf(0x200000, 1)));
    try testing.expectError(error.InvalidOperatorShape, k.cast(s, std.math.maxInt(u32), fp(0x100000, 1), bf(0x200000, 1)));
    try testing.expectError(error.InvalidOperatorSpan, k.cast(s, 4, fp(0x100000, 3), bf(0x200000, 4)));
    try testing.expectError(error.UnsupportedOperatorDType, k.cast(s, 4, bf(0x100000, 8), bf(0x200000, 4)));
    try testing.expectError(error.OperatorAliasing, k.cast(s, 4, fp(0x100000, 4), bf(0x100000, 4)));
    try testing.expectError(error.InvalidOperatorSpan, k.cast(s, 4, fp(std.math.maxInt(u64) - 3, 4), bf(0x200000, 4)));
    const m: ops.Matrix = .{ .rows = 4, .cols = 8 };
    try testing.expectError(error.InvalidOperatorScalar, k.rms(s, m, 0, bf(0x100000, 32), bf(0x200000, 8), bf(0x300000, 32)));
    try testing.expectError(error.InvalidOperatorScalar, k.swiglu(s, 32, std.math.nan(f32), bf(0x100000, 32), bf(0x200000, 32), bf(0x300000, 32)));
    try testing.expectError(error.OperatorAliasing, k.rms(s, m, 1e-6, bf(0x100000, 32), bf(0x200000, 8), bf(0x200000, 32)));
    try testing.expectError(error.OperatorAliasing, k.residual(s, 32, bf(0x100000, 32), bf(0x200000, 32), bf(0x100002, 32)));
    const qkv: ops.Qkv = .{ .groups = 2, .sequence = 4, .heads = 2, .dim = 4 };
    try testing.expectError(error.OperatorAliasing, k.qkvRope(s, qkv, 1e-6, bf(0x100000, 192), bf(0x200000, 4), bf(0x300000, 4), fp(0x400000, 16), fp(0x500000, 16), bf(0x600000, 64), bf(0x600000, 64), bf(0x800000, 64)));
    var other = Mock.driver();
    try testing.expectError(error.InvalidOperatorStream, k.cast(.{ .d = &other, .handle = null }, 4, fp(0x100000, 4), bf(0x200000, 4)));
    try testing.expectEqual(@as(usize, 0), Mock.launches);
    try testing.expectEqual(@as(usize, 0), Mock.init_calls);
    try testing.expectError(error.InvalidOperatorShape, (ops.Matrix{ .rows = 32001, .cols = 1024 }).elements());
    try testing.expectError(error.InvalidOperatorShape, (ops.Qkv{ .groups = 1, .sequence = 3, .heads = 16, .dim = 64 }).elements());
    try testing.expectError(error.InvalidOperatorShape, (ops.Qkv{ .groups = std.math.maxInt(u32), .sequence = 4, .heads = 16, .dim = 64 }).elements());
}
test "vision operators native launch failures propagate and the caller still owns a releasable image" {
    Mock.reset();
    var d = Mock.driver();
    var k = try ops.Kernels.load(&d, &image);
    const s: cuda.Stream = .{ .d = &d, .handle = null };
    Mock.fail_launch = true;
    try testing.expectError(error.NotReady, k.cast(s, 4, fp(0x100000, 4), bf(0x200000, 4)));
    const m: ops.Matrix = .{ .rows = 4, .cols = 8 };
    try testing.expectError(error.NotReady, k.rms(s, m, 1e-6, bf(0x100000, 32), bf(0x200000, 8), bf(0x300000, 32)));
    try testing.expectError(error.NotReady, k.residual(s, 32, bf(0x100000, 32), bf(0x200000, 32), bf(0x300000, 32)));
    try testing.expectError(error.NotReady, k.swiglu(s, 32, 10, bf(0x100000, 32), bf(0x200000, 32), bf(0x300000, 32)));
    try testing.expectError(error.NotReady, k.projectorNormGelu(s, m, bf(0x100000, 32), bf(0x200000, 8), bf(0x300000, 8), bf(0x400000, 32)));
    try testing.expectError(error.NotReady, k.qkvRope(s, .{ .groups = 2, .sequence = 4, .heads = 2, .dim = 4 }, 1e-6, bf(0x100000, 192), bf(0x200000, 4), bf(0x300000, 4), fp(0x400000, 16), fp(0x500000, 16), bf(0x600000, 64), bf(0x700000, 64), bf(0x800000, 64)));
    try testing.expectEqual(@as(usize, 6), Mock.launches);
    try testing.expect(Mock.live);
    k.deinit();
    try testing.expect(!Mock.live);
    try testing.expectEqual(@as(usize, 0), Mock.init_calls);
}

test "vision operators additional boundary and inplace policies are checked without device reads" {
    Mock.reset();
    var d = Mock.driver();
    var k = try ops.Kernels.load(&d, &image);
    defer k.deinit();
    const s: cuda.Stream = .{ .d = &d, .handle = null };
    const m: ops.Matrix = .{ .rows = 2, .cols = 8 };
    try testing.expectError(error.InvalidOperatorSpan, k.cast(s, 4, fp(0, 4), bf(0x200000, 4)));
    try testing.expectError(error.InvalidOperatorSpan, k.cast(s, 4, fp(0x100002, 4), bf(0x200000, 4)));
    try testing.expectError(error.InvalidOperatorSpan, k.residual(s, 4, bf(0x100001, 4), bf(0x200000, 4), bf(0x300000, 4)));
    try testing.expectError(error.OperatorAliasing, k.projectorNormGelu(s, m, bf(0x100000, 16), bf(0x200000, 8), bf(0x300000, 8), bf(0x100002, 16)));
    try testing.expectError(error.OperatorAliasing, k.projectorNormGelu(s, m, bf(0x100000, 16), bf(0x200000, 8), bf(0x300000, 8), bf(0x300000, 16)));
    try testing.expectError(error.OperatorAliasing, k.swiglu(s, 16, 10, bf(0x100000, 16), bf(0x200000, 16), bf(0x200002, 16)));
    const qkv: ops.Qkv = .{ .groups = 2, .sequence = 4, .heads = 2, .dim = 4 };
    try testing.expectError(error.OperatorAliasing, k.qkvRope(s, qkv, 1e-6, bf(0x100000, 192), bf(0x200000, 4), bf(0x300000, 4), fp(0x400000, 16), fp(0x500000, 16), bf(0x100000, 64), bf(0x700000, 64), bf(0x800000, 64)));
    try testing.expectEqual(@as(usize, 0), Mock.launches);
    try k.rms(s, m, 1e-6, bf(0x100000, 16), bf(0x200000, 8), bf(0x100000, 16));
    try k.projectorNormGelu(s, m, bf(0x100000, 16), bf(0x200000, 8), bf(0x300000, 8), bf(0x100000, 16));
    try k.swiglu(s, 16, 10, bf(0x100000, 16), bf(0x200000, 16), bf(0x200000, 16));
    try k.residual(s, 257, bf(0x100000, 257), bf(0x200000, 257), bf(0x300000, 257));
    try testing.expectEqual(@as(u32, 2), Mock.grid);
    try testing.expectEqual(@as(u32, 256), Mock.threads);
    try testing.expectEqual(@as(usize, 0), Mock.init_calls);
    for ([_]ops.Matrix{ .{ .rows = 0, .cols = 4 }, .{ .rows = 1, .cols = 0 }, .{ .rows = 1, .cols = 4097 } }) |bad| try testing.expectError(error.InvalidOperatorShape, bad.elements());
    for ([_]ops.Qkv{ .{ .groups = 0, .sequence = 4, .heads = 2, .dim = 4 }, .{ .groups = 1, .sequence = 4, .heads = 129, .dim = 4 }, .{ .groups = 1, .sequence = 4, .heads = 2, .dim = 258 } }) |bad| try testing.expectError(error.InvalidOperatorShape, bad.elements());
}
