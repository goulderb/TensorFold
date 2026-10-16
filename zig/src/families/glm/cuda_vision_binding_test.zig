const std = @import("std");
const cuda = @import("cuda");
const binding = @import("cuda_vision_binding.zig");
const t = std.testing;

const image_token: u32 = 154854;
const prompt = [_]u32{ 1, image_token, image_token, 2, 3, image_token, 4, 5, image_token, 6, 7, 8 };
const positions = [_]u32{ 1, 2, 5, 8 };
fn table(d: *const cuda.Driver) binding.Table {
    return .{ .driver = d, .context = @ptrFromInt(0x1000), .stream = .{ .d = d, .handle = null }, .features = .{ .d = d, .ptr = 0x100000, .len = 4 * 4096 * 2 }, .positions = &positions, .ids = &prompt, .image_token = image_token, .rows = 4, .ready = true, .generation = 1, .replica = 9 };
}
fn target(d: *const cuda.Driver, start: usize) binding.Target {
    return .{ .stream = .{ .d = d, .handle = null }, .context = @ptrFromInt(0x1000), .embedding = .{ .d = d, .ptr = 0x200000, .len = 8 * 4 * 4096 * 2, .borrowed = true }, .start = start, .capacity = 32, .window = 8, .vocab = 154880 };
}
test "vision binding maps absolute positions to packed BF16 rows in each bounded window" {
    var driver: cuda.Driver = undefined;
    var owner = table(&driver);
    const b = try owner.binding();
    const p = try binding.plan(b, target(&driver, 0), &.{ 1, @intCast(image_token), @intCast(image_token), 2 });
    try t.expectEqual(@as(usize, 2), p.count);
    try t.expectEqual(@as(usize, 1), p.rows[0].window);
    try t.expectEqual(@as(usize, 0), p.rows[0].table);
    try t.expectEqual(@as(usize, 2), p.rows[1].window);
    try t.expectEqual(@as(usize, 1), p.rows[1].table);
    const next = try binding.plan(b, target(&driver, 4), &.{ 3, @intCast(image_token), 4, 5, @intCast(image_token), 6, 7, 8 });
    try t.expectEqual(@as(usize, 2), next.count);
    try t.expectEqual(@as(usize, 1), next.rows[0].window);
    try t.expectEqual(@as(usize, 2), next.rows[0].table);
    try t.expectEqual(@as(usize, 4), next.rows[1].window);
    try t.expectEqual(@as(usize, 3), next.rows[1].table);
    try t.expectEqual(@as(usize, 0), (try binding.plan(b, target(&driver, 3), &.{ 2, 3 })).count);
    const one = try binding.plan(b, target(&driver, 8), &.{@intCast(image_token)});
    try t.expectEqual(@as(usize, 1), one.count);
    try t.expectEqual(@as(usize, 0), one.rows[0].window);
    try t.expectEqual(@as(usize, 3), one.rows[0].table);
}

test "vision binding rejects stale readiness, ownership, dtype, geometry and every malformed placeholder map" {
    var driver: cuda.Driver = undefined;
    var other: cuda.Driver = undefined;
    var owner = table(&driver);
    const b = try owner.binding();
    owner.ready = false;
    try t.expectError(error.VisionNotReady, binding.plan(b, target(&driver, 0), &.{1}));
    owner.ready = true;
    owner.generation += 1;
    try t.expectError(error.StaleVisionBinding, binding.plan(b, target(&driver, 0), &.{1}));
    owner = table(&driver);
    owner.features.borrowed = true;
    try t.expectError(error.VisionOwnership, owner.binding());
    owner = table(&driver);
    owner.dtype = .f32;
    try t.expectError(error.BadVisionShape, owner.binding());
    owner = table(&driver);
    owner.width = 2048;
    try t.expectError(error.BadVisionShape, owner.binding());
    owner = table(&driver);
    owner.features.len -= 2;
    try t.expectError(error.BadVisionShape, owner.binding());
    owner = table(&driver);
    owner.rows = 0;
    try t.expectError(error.BadVisionShape, owner.binding());
    owner = table(&driver);
    owner.features.d = &other;
    try t.expectError(error.DriverMismatch, owner.binding());
    owner = table(&driver);
    owner.context = null;
    try t.expectError(error.VisionContextMismatch, owner.binding());
    owner = table(&driver);
    for ([_][4]u32{ .{ 1, 1, 5, 8 }, .{ 2, 1, 5, 8 }, .{ 1, 2, 5, 12 }, .{ 1, 2, 3, 8 } }) |bad| {
        owner.positions = &bad;
        try t.expectError(error.BadVisionPositions, owner.binding());
    }
    owner = table(&driver);
    var original = prompt;
    original[10] = image_token;
    owner.ids = &original;
    try t.expectError(error.BadVisionPositions, owner.binding());
    original = prompt;
    original[2] = 3;
    try t.expectError(error.BadVisionPositions, owner.binding());
    original = prompt;
    original[10] = 154880;
    try t.expectError(error.BadToken, owner.binding());
    owner = table(&driver);
    owner.features.ptr = std.math.maxInt(u64) - 1;
    try t.expectError(error.Overflow, owner.binding());
    owner = table(&driver);
    owner.features.ptr += 1;
    try t.expectError(error.BadVisionShape, owner.binding());
}

test "vision window requires original language IDs and bounded context, row count, stream and packed destination" {
    var driver: cuda.Driver = undefined;
    var other: cuda.Driver = undefined;
    var owner = table(&driver);
    const b = try owner.binding();
    var dst = target(&driver, 0);
    try t.expectError(error.BadRows, binding.plan(b, dst, &.{}));
    try t.expectError(error.BadRows, binding.plan(b, dst, &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9 }));
    try t.expectError(error.BadToken, binding.plan(b, dst, &.{-1}));
    try t.expectError(error.BadToken, binding.plan(b, dst, &.{154880}));
    try t.expectError(error.VisionTokenMismatch, binding.plan(b, dst, &.{0}));
    dst.start = 1;
    try t.expectError(error.VisionTokenMismatch, binding.plan(b, dst, &.{0}));
    dst.start = std.math.maxInt(usize);
    try t.expectError(error.ContextFull, binding.plan(b, dst, &.{1}));
    dst = target(&driver, 32);
    try t.expectError(error.ContextFull, binding.plan(b, dst, &.{1}));
    dst = target(&driver, 12);
    try t.expectError(error.VisionWindowMismatch, binding.plan(b, dst, &.{1}));
    dst = target(&driver, 0);
    dst.capacity = 0;
    try t.expectError(error.BadCapacity, binding.plan(b, dst, &.{1}));
    dst.capacity = 256 * 1024 + 1;
    try t.expectError(error.BadCapacity, binding.plan(b, dst, &.{1}));
    dst = target(&driver, 0);
    dst.capacity = 8;
    try t.expectError(error.ContextFull, binding.plan(b, dst, &.{1}));
    dst = target(&driver, 0);
    dst.width = 2048;
    try t.expectError(error.BadVisionShape, binding.plan(b, dst, &.{1}));
    dst = target(&driver, 0);
    dst.streams = 1;
    try t.expectError(error.BadVisionShape, binding.plan(b, dst, &.{1}));
    dst = target(&driver, 0);
    dst.embedding.len -= 2;
    try t.expectError(error.BadVisionShape, binding.plan(b, dst, &.{1}));
    dst = target(&driver, 0);
    dst.embedding.ptr = owner.features.ptr;
    try t.expectError(error.VisionAliasedBuffers, binding.plan(b, dst, &.{1}));
    dst = target(&driver, 0);
    dst.embedding.ptr = std.math.maxInt(u64) - 1;
    try t.expectError(error.Overflow, binding.plan(b, dst, &.{1}));
    dst = target(&driver, 0);
    dst.stream.d = &other;
    try t.expectError(error.DriverMismatch, binding.plan(b, dst, &.{1}));
    dst = target(&driver, 0);
    dst.stream.handle = @ptrFromInt(0x2000);
    try t.expectError(error.VisionStreamMismatch, binding.plan(b, dst, &.{1}));
    dst = target(&driver, 0);
    dst.context = @ptrFromInt(0x3000);
    try t.expectError(error.VisionContextMismatch, binding.plan(b, dst, &.{1}));
    dst = target(&driver, 0);
    dst.rank = 1;
    try t.expectError(error.VisionRankMismatch, binding.plan(b, dst, &.{1}));
}

test "padded native sequence capacity admits a short image prompt inside logical context" {
    var driver: cuda.Driver = undefined;
    var owner = table(&driver);
    const b = try owner.binding();
    var dst = target(&driver, 0);
    dst.capacity = binding.logicalCapacity(262208);
    try t.expectEqual(@as(usize, 262144), dst.capacity);
    const plan = try binding.plan(b, dst, &.{ 1, @intCast(image_token), @intCast(image_token), 2 });
    try t.expectEqual(@as(usize, 2), plan.count);
    dst.capacity = 262208;
    try t.expectError(error.BadCapacity, binding.plan(b, dst, &.{ 1, @intCast(image_token), @intCast(image_token), 2 }));
}

// ABI mocks exercise the REAL native copy/launch path; no driver/library/context/module is opened or loaded.
const Native = struct {
    const Copy = struct { dst: u64, src: u64, bytes: usize };
    var copies: [32]Copy = undefined;
    var count: usize = 0;
    var queries: usize = 0;
    var uploads: usize = 0;
    var launches: usize = 0;
    var synchronizes: usize = 0;
    var fail_copy: ?usize = null;
    var context: cuda.abi.Context = @ptrFromInt(0x1000);
    var input: ?[]const u16 = null;
    var output: ?[]u16 = null;
    var source: u64 = 0;
    var destination: u64 = 0;
    fn reset() void {
        count = 0;
        queries = 0;
        uploads = 0;
        launches = 0;
        synchronizes = 0;
        fail_copy = null;
        context = @ptrFromInt(0x1000);
        input = null;
        output = null;
    }
    fn current(out: *cuda.abi.Context) callconv(.c) cuda.abi.Result {
        queries += 1;
        out.* = context;
        return cuda.abi.success;
    }
    fn copy(dst: u64, src: u64, bytes: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        copies[count] = .{ .dst = dst, .src = src, .bytes = bytes };
        count += 1;
        if (fail_copy == count) return cuda.abi.error_not_ready;
        if (input) |x| if (output) |y| @memcpy(y[(dst - destination) / 2 ..][0 .. bytes / 2], x[(src - source) / 2 ..][0 .. bytes / 2]);
        return cuda.abi.success;
    }
    fn upload(_: u64, _: ?*const anyopaque, _: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        uploads += 1;
        return cuda.abi.success;
    }
    fn synchronize(_: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        synchronizes += 1;
        return cuda.abi.success;
    }
    fn launch(_: cuda.abi.Function, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: cuda.abi.Stream, _: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
        launches += 1;
        return cuda.abi.success;
    }
    fn driver() cuda.Driver {
        var d: cuda.Driver = undefined;
        d.api.cuCtxGetCurrent = current;
        d.api.cuMemcpyDtoDAsync_v2 = copy;
        d.api.cuMemcpyHtoDAsync_v2 = upload;
        d.api.cuStreamSynchronize = synchronize;
        d.api.cuLaunchKernel = launch;
        return d;
    }
};

test "actual native injection copies BF16 bits to every HC stream while leaving language rows intact" {
    Native.reset();
    var driver = Native.driver();
    var owner = table(&driver);
    const b = try owner.binding();
    const dst = target(&driver, 4);
    var input: [4 * 4096]u16 = undefined;
    for (&input, 0..) |*x, i| x.* = @intCast(0x8000 + i / 4096);
    var output: [8 * 4 * 4096]u16 = @splat(0x1234);
    Native.input = &input;
    Native.output = &output;
    Native.source = owner.features.ptr;
    Native.destination = dst.embedding.ptr;
    try binding.inject(b, dst, &.{ 3, @intCast(image_token), 4, 5, @intCast(image_token), 6, 7, 8 });
    try t.expectEqual(@as(usize, 8), Native.count);
    for (0..8) |r| for (0..4) |s| {
        const bits: u16 = if (r == 1) 0x8002 else if (r == 4) 0x8003 else 0x1234;
        for (output[(r * 4 + s) * 4096 ..][0..4096]) |x| try t.expectEqual(bits, x);
    };
    for (Native.copies[0..Native.count], 0..) |c, i| {
        try t.expectEqual(@as(usize, 8192), c.bytes);
        try t.expectEqual(owner.features.ptr + (2 + i / 4) * 8192, c.src);
        try t.expectEqual(dst.embedding.ptr + ((if (i < 4) @as(usize, 1) else 4) * 4 + i % 4) * 8192, c.dst);
    }
}

test "bad live binding dispatches nothing and context admission precedes native copies" {
    Native.reset();
    var driver = Native.driver();
    var owner = table(&driver);
    const b = try owner.binding();
    owner.ready = false;
    try t.expectError(error.VisionNotReady, binding.inject(b, target(&driver, 0), &.{1}));
    owner.ready = true;
    try t.expectError(error.VisionTokenMismatch, binding.inject(b, target(&driver, 0), &.{0}));
    try t.expectEqual(@as(usize, 0), Native.queries);
    try t.expectEqual(@as(usize, 0), Native.count);
    Native.context = @ptrFromInt(0x5000);
    try t.expectError(error.VisionContextMismatch, binding.inject(b, target(&driver, 0), &.{ 1, @intCast(image_token) }));
    try t.expectEqual(@as(usize, 0), Native.count);
}
