const std = @import("std");
const projection = @import("cuda_vision_cudnn.zig");
const d = projection.cudnn;
const Mock = d.test_support.Mock;
const config = @import("cuda_vision_config.zig");
const testing = std.testing;
fn production() !config.Config {
    return config.parse(testing.allocator, @embedFile("fixtures_cuda_config_exl3.json"));
}

test "patch Conv3d has the oracle's contiguous BF16 tensor and filter layouts" {
    const c = try production();
    const p = try projection.patch(c, 80);
    try testing.expectEqual(@as(u8, 5), p.input.rank);
    try testing.expectEqualSlices(c_int, &.{ 80, 3, 2, 14, 14 }, &p.input.dims);
    try testing.expectEqualSlices(c_int, &.{ 1176, 392, 196, 14, 1 }, &p.input.strides);
    try testing.expectEqualSlices(c_int, &.{ 1024, 3, 2, 14, 14 }, &p.filter.dims);
    try testing.expectEqualSlices(c_int, &.{ 80, 1024, 1, 1, 1 }, &p.output.dims);
    try testing.expectEqualSlices(c_int, &.{ 2, 14, 14 }, &p.stride);
    try testing.expectEqual(@as(usize, 80 * 1176 * 2), try p.input.bytes());
    try testing.expectEqual(@as(usize, 80 * 1024 * 2), try p.output.bytes());
}
test "downsample Conv2d consumes the actual NHWC-backed permuted view without repacking" {
    const c = try production();
    const p = try projection.downsample(c, 20);
    try testing.expectEqual(@as(u8, 4), p.input.rank);
    try testing.expectEqualSlices(c_int, &.{ 20, 1024, 2, 2 }, p.input.dims[0..4]);
    try testing.expectEqualSlices(c_int, &.{ 4096, 1, 2048, 1024 }, p.input.strides[0..4]);
    try testing.expectEqualSlices(c_int, &.{ 4096, 1024, 2, 2 }, p.filter.dims[0..4]);
    try testing.expectEqualSlices(c_int, &.{ 20, 4096, 1, 1 }, p.output.dims[0..4]);
    try testing.expectEqual(@as(usize, 20 * 4 * 1024 * 2), try p.input.bytes());
    try testing.expectEqual(@as(usize, 20 * 4096 * 2), try p.output.bytes());
    // Index (n,c,h,w) maps into the existing [n,h,w,c] merge-block storage.
    for (0..2) |n| for (0..1024) |channel| for (0..2) |h| for (0..2) |w| {
        const offset = n * 4096 + channel + h * 2048 + w * 1024;
        try testing.expectEqual(((n * 2 + h) * 2 + w) * 1024 + channel, offset);
    };
}
test "both projection lifetimes unwind if any second-operator initialization step fails" {
    const c = try production();
    Mock.reset();
    var first = try d.Convolution.init(&Mock.api, try projection.patch(c, 80), .{ .workspace_limit = 1024 });
    const first_calls = Mock.calls;
    first.deinit();
    Mock.reset();
    var good = try projection.Projections.init(&Mock.api, c, 80, .{ .workspace_limit = 1024, .deterministic = false });
    const total_calls = Mock.calls;
    try testing.expectEqual(@as(usize, 12), Mock.live);
    good.deinit();
    try testing.expectEqual(@as(usize, 0), Mock.live);
    for (first_calls + 1..total_calls + 1) |fail| {
        Mock.reset();
        Mock.fail_at = fail;
        try testing.expectError(error.CudnnFailed, projection.Projections.init(&Mock.api, c, 80, .{ .workspace_limit = 1024 }));
        try testing.expectEqual(@as(usize, 0), Mock.live);
    }
}
test "invalid rows, unsupported geometry, excessive descriptor size and overlapping tensor views refuse" {
    const c = try production();
    for ([_]u32{ 0, 3, 32004, std.math.maxInt(u32) }) |rows| try testing.expectError(error.InvalidShape, projection.patch(c, rows));
    for ([_]u32{ 0, 8001 }) |rows| try testing.expectError(error.InvalidShape, projection.downsample(c, rows));
    var invalid = c;
    invalid.merge = 3;
    try testing.expectError(error.UnsupportedVisionGeometry, projection.patch(invalid, 4));
    invalid = c;
    invalid.output = std.math.maxInt(u32);
    try testing.expectError(error.InvalidShape, projection.downsample(invalid, 4));
    var spec = try projection.downsample(c, 4);
    spec.input.strides[2] = 1;
    try testing.expectError(error.InvalidShape, spec.check());
    spec = try projection.downsample(c, 4);
    spec.filter.strides[1] += 1;
    try testing.expectError(error.InvalidShape, spec.check());
    Mock.reset();
    try testing.expectError(error.InvalidShape, projection.Projections.init(&Mock.api, c, 3, .{ .workspace_limit = 1024 }));
    try testing.expectEqual(@as(usize, 0), Mock.calls);
}
test "both actual SDK dispatch bodies compile behind a deterministic false runtime gate" {
    var enabled = false;
    std.mem.doNotOptimizeAway(&enabled);
    if (enabled) {
        var library = try d.Library.open();
        defer library.close();
        var operators = try projection.Projections.init(&library.api, try production(), 80, .{ .workspace_limit = 1 << 26, .deterministic = false });
        defer operators.deinit();
        const patch_buffers = try d.test_support.buffers(operators.patch.spec, operators.patch.workspace_need);
        const down_buffers = try d.test_support.buffers(operators.downsample.spec, operators.downsample.workspace_need);
        try operators.patch.run(patch_buffers, null);
        try operators.downsample.run(down_buffers, null);
    }
}
