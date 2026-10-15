const std = @import("std");
const family = @import("cuda_vision_sdpa.zig");
const sdpa = family.sdpa;
const cuda = @import("cuda");
const Config = @import("cuda_vision_config.zig").Config;
const Mock = sdpa.test_support.Mock;
const t = std.testing;

fn config() !Config {
    return @import("cuda_vision_config.zig").parse(t.allocator, @embedFile("fixtures_cuda_config_exl3.json"));
}
fn buffer(driver: *const cuda.Driver, ptr: u64, len: usize) cuda.DeviceBuffer {
    return .{ .d = driver, .ptr = ptr, .len = len, .borrowed = true };
}
fn packedBuffers(driver: *const cuda.Driver, geometry: family.Shape) !family.Packed {
    return .{ .qkv = buffer(driver, 0x10000000, try geometry.bytes(.packed_qkv)), .output = buffer(driver, 0x40000000, try geometry.bytes(.separate_qkv)), .workspace = buffer(driver, 0x50000000, 256) };
}
fn separate(driver: *const cuda.Driver, geometry: family.Shape) !family.Separate {
    const bytes = try geometry.bytes(.separate_qkv);
    return .{ .q = buffer(driver, 0x10000000, bytes), .k = buffer(driver, 0x20000000, bytes), .v = buffer(driver, 0x30000000, bytes), .output = buffer(driver, 0x40000000, bytes), .workspace = buffer(driver, 0x50000000, 256) };
}
const shape: family.Shape = .{ .batch = 2, .sequence = 4 };

test "vision packed and separate descriptors exactly preserve pinned BHSD views and BSHD output" {
    const c = try config();
    const interleaved = try family.packedSpec(c, shape);
    const independent = try family.separateSpec(c, shape);
    try t.expectEqualDeep([4]i64{ 2, 16, 4, 64 }, interleaved.q.dims);
    try t.expectEqualDeep([4]i64{ 12288, 64, 3072, 1 }, interleaved.q.strides);
    try t.expectEqualDeep([4]i64{ 4096, 64, 1024, 1 }, independent.q.strides);
    try t.expectEqualDeep(independent.q.strides, independent.k.strides);
    try t.expectEqualDeep(independent.q.strides, independent.v.strides);
    try t.expectEqualDeep(independent.q.strides, interleaved.output.strides);
    try t.expectEqualDeep(independent.q.strides, independent.output.strides);
    try t.expectEqual(@as(f32, 0.125), interleaved.scale);
    try t.expectEqual(@as(f32, 0.125), independent.scale);
    try t.expectEqual(@as(usize, 45056), try interleaved.q.bytes());
    try t.expectEqual(@as(usize, 16384), try independent.q.bytes());
    try t.expectEqual(@as(usize, 49152), try shape.bytes(.packed_qkv));
}

test "every coordinate maps to its image frame pair and correct QKV component without mixing batches" {
    const c = try config();
    const geometry: family.Shape = .{ .batch = 3, .sequence = 8 };
    const ps = try family.packedSpec(c, geometry);
    const ss = try family.separateSpec(c, geometry);
    for (0..geometry.batch) |b| for (0..geometry.sequence) |s| for (0..16) |h| for (0..64) |d| {
        const index = [4]usize{ b, h, s, d };
        var packed_at: usize = 0;
        var separate_at: usize = 0;
        var output_at: usize = 0;
        for (index, ps.q.strides, ss.q.strides, ps.output.strides) |i, a, z, o| {
            packed_at += i * @as(usize, @intCast(a));
            separate_at += i * @as(usize, @intCast(z));
            output_at += i * @as(usize, @intCast(o));
        }
        const expected = ((b * geometry.sequence + s) * 16 + h) * 64 + d;
        try t.expectEqual(expected, separate_at);
        try t.expectEqual(expected, output_at);
        for (0..3) |component| {
            const dense = (((b * geometry.sequence + s) * 3 + component) * 16 + h) * 64 + d;
            try t.expectEqual(dense, packed_at + component * 1024);
        }
    };
}

test "family refuses unsupported model geometry and image canvas admission before backend calls" {
    Mock.reset();
    var c = try config();
    c.hidden = 512;
    c.heads = 8;
    try t.expectError(error.InvalidShape, family.separateSpec(c, shape));
    c = try config();
    c.heads = 32;
    try t.expectError(error.InvalidShape, family.packedSpec(c, shape));
    c = try config();
    for ([_]family.Shape{
        .{ .batch = 0, .sequence = 4 },     .{ .batch = 257, .sequence = 4 },
        .{ .batch = 1, .sequence = 0 },     .{ .batch = 1, .sequence = 3 },
        .{ .batch = 1, .sequence = 32004 },
    }) |bad| try t.expectError(error.InvalidShape, family.packedSpec(c, bad));
    const maximum: family.Shape = .{ .batch = 1, .sequence = 32000 };
    _ = try family.packedSpec(c, maximum);
    _ = try family.separateSpec(c, maximum);
    _ = try family.separateSpec(c, .{ .batch = 256, .sequence = 124 });
    try t.expectError(error.InvalidShape, family.separateSpec(c, .{ .batch = 256, .sequence = 128 }));
    try t.expectEqual(@as(usize, 65536000), try maximum.bytes(.separate_qkv));
    try t.expectEqual(@as(usize, 0), Mock.calls);
}

test "checked packed subviews include last V row and never transfer device-buffer ownership" {
    var driver: cuda.Driver = undefined;
    const original = try packedBuffers(&driver, shape);
    const views = try family.packedViews(shape, original);
    try t.expectEqual(original.qkv.ptr, views.q.ptr);
    try t.expectEqual(original.qkv.ptr + 2048, views.k.ptr);
    try t.expectEqual(original.qkv.ptr + 4096, views.v.ptr);
    try t.expectEqual(original.qkv.len - 2048, views.k.len);
    try t.expectEqual(original.qkv.len - 4096, views.v.len);
    try t.expectEqual(views.v.len, try (try family.packedSpec(try config(), shape)).v.bytes());
    var bad = original;
    bad.qkv.len -= 1;
    try t.expectError(error.InvalidSpan, family.packedViews(shape, bad));
    bad = original;
    bad.qkv.ptr = std.math.maxInt(u64) - 255;
    try t.expectError(error.InvalidSpan, family.packedViews(shape, bad));
    bad = original;
    bad.qkv.ptr += 2;
    try t.expectError(error.InvalidSpan, family.packedViews(shape, bad));
    try t.expect(original.qkv.borrowed and original.output.borrowed);
}

test "producer separate Q K V dispatches zero-copy with common driver stream and workspace" {
    var driver: cuda.Driver = undefined;
    Mock.reset();
    var op = try family.Attention.init(&Mock.api, &driver, try config(), shape, .separate_qkv, .{ .workspace_limit = 1024 });
    const b = try separate(&driver, shape);
    try op.run(.{ .separate_qkv = b }, .{ .d = &driver, .handle = @ptrFromInt(0x6000) });
    try t.expect(Mock.executed);
    try t.expectEqualDeep([5]usize{ b.q.ptr, b.k.ptr, b.v.ptr, b.output.ptr, @intFromPtr(&op.operator.spec.scale) }, Mock.last_pointers);
    try t.expectEqual(@as(usize, 0x6000), @intFromPtr(Mock.last_stream));
    try t.expectEqual(@as(usize, 256), op.operator.workspace_need);
    op.deinit();
    try t.expectEqual(@as(usize, 0), Mock.live);
    try t.expectEqual(@as(u64, 0x10000000), b.q.ptr);
}

test "packed QKV dispatches only under its explicit layout and output is never silently reinterpreted" {
    var driver: cuda.Driver = undefined;
    Mock.reset();
    var op = try family.Attention.init(&Mock.api, &driver, try config(), shape, .packed_qkv, .{ .workspace_limit = 1024 });
    defer op.deinit();
    const b = try packedBuffers(&driver, shape);
    const stream: cuda.Stream = .{ .d = &driver, .handle = @ptrFromInt(0x6000) };
    const before = Mock.calls;
    try t.expectError(error.InvalidSdpaLayout, op.run(.{ .separate_qkv = try separate(&driver, shape) }, stream));
    try t.expectEqual(before, Mock.calls);
    try op.run(.{ .packed_qkv = b }, stream);
    try t.expectEqualDeep([5]usize{ b.qkv.ptr, b.qkv.ptr + 2048, b.qkv.ptr + 4096, b.output.ptr, @intFromPtr(&op.operator.spec.scale) }, Mock.last_pointers);
}

test "every family driver mismatch span failure and alias refuses without API calls" {
    var first: cuda.Driver = undefined;
    var second: cuda.Driver = undefined;
    Mock.reset();
    var op = try family.Attention.init(&Mock.api, &first, try config(), shape, .separate_qkv, .{ .workspace_limit = 1024 });
    defer op.deinit();
    const good = try separate(&first, shape);
    const stream: cuda.Stream = .{ .d = &first, .handle = null };
    const before = Mock.calls;
    try t.expectError(error.DriverMismatch, op.run(.{ .separate_qkv = good }, .{ .d = &second, .handle = null }));
    inline for (.{ "q", "k", "v", "output", "workspace" }) |field| {
        var bad = good;
        @field(bad, field).d = &second;
        try t.expectError(error.DriverMismatch, op.run(.{ .separate_qkv = bad }, stream));
        bad = good;
        @field(bad, field).len -= 1;
        try t.expectError(error.InvalidSpan, op.run(.{ .separate_qkv = bad }, stream));
    }
    var alias = good;
    alias.output.ptr = alias.q.ptr;
    try t.expectError(error.Aliasing, op.run(.{ .separate_qkv = alias }, stream));
    alias = good;
    alias.workspace.ptr = alias.k.ptr;
    try t.expectError(error.Aliasing, op.run(.{ .separate_qkv = alias }, stream));
    try t.expectEqual(before, Mock.calls);
}

test "family initialization preserves backend failure cleanup and exact runtime refusal" {
    var driver: cuda.Driver = undefined;
    Mock.reset();
    Mock.fail_at = 10;
    try t.expectError(error.CudnnFailed, family.Attention.init(&Mock.api, &driver, try config(), shape, .separate_qkv, .{ .workspace_limit = 1024 }));
    try t.expectEqual(@as(usize, 0), Mock.live);
    Mock.reset();
    Mock.cuda_version = 12080;
    try t.expectError(error.CudaVersionMismatch, family.Attention.init(&Mock.api, &driver, try config(), shape, .separate_qkv, .{ .workspace_limit = 1024 }));
    try t.expectEqual(@as(usize, 0), Mock.calls);
}
