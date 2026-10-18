const std = @import("std");
const cuda = @import("cuda");
const W = @import("cuda_weights.zig");
const Config = @import("cuda_checkpoint.zig").Config;

const Mock = struct {
    var available: usize = 0;
    var current: cuda.abi.Context = null;
    var creates: usize = 0;
    var destroys: usize = 0;
    var frees: usize = 0;
    var next: u64 = 4096;
    var can_select: bool = true;
    var can_sync: bool = true;

    fn setCurrent(ctx: cuda.abi.Context) callconv(.c) cuda.abi.Result {
        if (!can_select) return cuda.abi.error_not_ready;
        current = ctx;
        return cuda.abi.success;
    }
    fn memory(free_bytes: *usize, total: *usize) callconv(.c) cuda.abi.Result {
        free_bytes.* = available;
        total.* = 24 * 1024 * 1024 * 1024;
        return cuda.abi.success;
    }
    fn synchronize() callconv(.c) cuda.abi.Result {
        return if (can_sync) cuda.abi.success else cuda.abi.error_not_ready;
    }
    fn create(stream: *cuda.abi.Stream, _: c_uint) callconv(.c) cuda.abi.Result {
        creates += 1;
        stream.* = @ptrFromInt(16);
        return cuda.abi.success;
    }
    fn destroy(_: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        destroys += 1;
        return cuda.abi.success;
    }
    fn alloc(ptr: *u64, bytes: usize) callconv(.c) cuda.abi.Result {
        ptr.* = next;
        next += bytes;
        return cuda.abi.success;
    }
    fn free(_: u64) callconv(.c) cuda.abi.Result {
        frees += 1;
        return cuda.abi.success;
    }
    fn driver() cuda.Driver {
        can_select = true;
        can_sync = true;
        var d: cuda.Driver = .{ .lib = undefined, .api = undefined };
        d.api.cuCtxSetCurrent = setCurrent;
        d.api.cuMemGetInfo_v2 = memory;
        d.api.cuCtxSynchronize = synchronize;
        d.api.cuStreamCreate = create;
        d.api.cuStreamDestroy_v2 = destroy;
        d.api.cuMemAlloc_v2 = alloc;
        d.api.cuMemFree_v2 = free;
        return d;
    }
};

test "cuda Qwen weights refuse a GPU below the floor before creating upload resources" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    const config = try Config.parse(a, @embedFile("fixtures/ornith_nvfp4/config.json"));
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const image = [_]u8{ 2, 0, 0, 0, 0, 0, 0, 0, '{', '}' };
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors", .data = &image });
    var path: [96]u8 = undefined;
    const dir = try std.fmt.bufPrint(&path, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    var d = Mock.driver();
    const ctx: cuda.Context = .{ .d = &d, .device = 1, .handle = @ptrFromInt(32) };
    Mock.available = W.gpu_floor_bytes - 1;
    Mock.creates = 0;
    Mock.destroys = 0;
    try std.testing.expectError(error.GpuMemoryFloor, W.Weights.load(a, io, &ctx, dir, config, 0, 0, false, false));
    try std.testing.expectEqual(@as(usize, 0), Mock.creates);
    try std.testing.expectEqual(ctx.handle, Mock.current);
    if (!cuda.kernels.available) {
        Mock.available = 24 * 1024 * 1024 * 1024;
        try std.testing.expectError(error.BuiltWithoutKernels, W.Weights.load(a, io, &ctx, dir, config, 0, 0, false, false));
        try std.testing.expectEqual(@as(usize, 1), Mock.creates);
        try std.testing.expectEqual(Mock.creates, Mock.destroys);
    }
}

test "cuda Qwen weights release every owned buffer from a partially constructed partition" {
    const a = std.testing.allocator;
    var d = Mock.driver();
    const ctx: cuda.Context = .{ .d = &d, .device = 1, .handle = @ptrFromInt(32) };
    const before = cuda.usage(false).device;
    var w: W.Weights = .{ .layers = try a.alloc(W.Layer, 2), .gpa = a, .ctx = &ctx };
    try w.buffers.ensureTotalCapacity(a, 2);
    w.buffers.appendAssumeCapacity(try cuda.DeviceBuffer.alloc(&d, 1024));
    w.buffers.appendAssumeCapacity(try cuda.DeviceBuffer.alloc(&d, 2048));
    w.device_bytes = 3072;
    Mock.frees = 0;
    try std.testing.expectEqual(@as(usize, 3072), w.totalDeviceBytes());
    try std.testing.expectEqual(before + 3072, cuda.usage(false).device);
    Mock.can_select = false;
    w.deinit();
    try std.testing.expectEqual(@as(usize, 0), Mock.frees);
    Mock.can_select = true;
    Mock.can_sync = false;
    w.deinit();
    try std.testing.expectEqual(@as(usize, 0), Mock.frees);
    Mock.can_sync = true;
    w.deinit();
    try std.testing.expectEqual(@as(usize, 2), Mock.frees);
    try std.testing.expectEqual(ctx.handle, Mock.current);
    try std.testing.expectEqual(before, cuda.usage(false).device);
}
