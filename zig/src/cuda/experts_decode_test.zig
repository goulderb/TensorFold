//! NVFP4 decode variants retain the baseline plan, packed-weight and launch ABI.
const std = @import("std");
const experts = @import("experts.zig");
const Nvfp4Experts = experts.Nvfp4Experts;
const Driver = @import("driver.zig").Driver;
const Stream = @import("stream.zig").Stream;
const Plan = @import("grouped.zig").Plan;
const Layer = experts.Layer;
const Epi = experts.Epi;

test "nvfp4 decode grids cover GLM rank column slices and respect occupancy" {
    const X = Nvfp4Experts;
    try std.testing.expectEqual(@as(usize, 64), X.decodeGrid(.baseline, 8, 1024 / 32, 192));
    try std.testing.expectEqual(@as(usize, 64), X.decodeGrid(.pipe4, 8, 1024 / 32, 192));
    try std.testing.expectEqual(@as(usize, 128), X.decodeGrid(.n16, 8, 1024 / 32, 192));
    try std.testing.expectEqual(@as(usize, 240), X.decodeGrid(.n16_pipe4, 64, 4096 / 32, 240));
    inline for (std.meta.tags(X.Variant)) |v| {
        try std.testing.expectEqual(@as(usize, 0), X.decodeGrid(v, 0, 32, 192));
        try std.testing.expectEqual(@as(usize, 0), X.decodeGrid(v, 8, 32, 0));
    }
    try std.testing.expectEqual(@as(usize, 1), X.decodeGrid(.n16, 1, 1, 192));
    try std.testing.expectEqualStrings("tf_nvfp4_n16_swiglu", X.candidateSymbol(.n16, 0));
    try std.testing.expectEqualStrings("tf_nvfp4_pipe4_down_f32", X.candidateSymbol(.pipe4, 1));
    try std.testing.expectEqualStrings("tf_nvfp4_n16_pipe4_down_bf16", X.candidateSymbol(.n16_pipe4, 2));
    try std.testing.expectEqualStrings("tf_nvfp4_n16_relu2", X.candidateSymbol(.n16, 3));
}

test "nvfp4 A/B dispatch preserves packed weight and plan ABI and baseline default" {
    const abi = @import("abi.zig");
    const Mock = struct {
        var function: usize = 0;
        var grid: u32 = 0;
        var block: u32 = 0;
        var values: [14]u64 = undefined;
        fn launch(f: abi.Function, gx: c_uint, _: c_uint, _: c_uint, bx: c_uint, _: c_uint, _: c_uint, _: c_uint, _: abi.Stream, params: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) abi.Result {
            function = @intFromPtr(f);
            grid = gx;
            block = bx;
            for (0..14) |i| values[i] = switch (i) {
                1, 2, 5, 6, 11, 12, 13 => @as(*align(1) const u32, @ptrCast(params.?[i].?)).*,
                else => @as(*align(1) const u64, @ptrCast(params.?[i].?)).*,
            };
            return abi.success;
        }
    };
    var d: Driver = undefined;
    d.api.cuLaunchKernel = Mock.launch;
    var ex: Nvfp4Experts = undefined;
    ex.warps = 4;
    ex.candidates_loaded = true;
    for (&ex.fns, 0..) |*f, i| f.* = .{ .d = &d, .handle = @ptrFromInt(1 + i) };
    ex.resident = @splat(192);
    for (&ex.candidate_fns, 0..) |*row, vi| {
        for (row, 0..) |*f, ei| f.* = .{ .d = &d, .handle = @ptrFromInt(10 + 4 * vi + ei) };
        ex.candidate_resident[vi] = @splat(192);
    }
    const stream: Stream = .{ .d = &d, .handle = null };
    const p: Plan = .{ .items = 101, .counts = 102, .members = 103, .rank = 104, .hist = 105 };
    var l: Layer = .{ .format = .nvfp4, .up = 201, .down = 202, .up_scale = 203, .down_scale = 204, .dims = 4096, .width = 1024, .experts = 288, .limit = 8 };
    inline for (std.meta.tags(Nvfp4Experts.Variant), 0..) |v, vi| {
        for ([_]Epi{ .up, .down_f32, .down_bf16 }) |epi| {
            const up = epi == .up;
            const ei: usize = @backingInt(epi);
            try ex.runVariant(v, stream, epi, .{ .x = 301, .stride = if (up) 4096 else 1024, .slots = 9 }, l, p, 302, 9, 288);
            try std.testing.expectEqual(@as(usize, if (vi == 0) 1 + ei else 10 + 4 * (vi - 1) + ei), Mock.function);
            try std.testing.expectEqual(@as(u32, if (!up) 192 else if (v == .n16 or v == .n16_pipe4) 144 else 72), Mock.grid);
            try std.testing.expectEqual(@as(u32, 128), Mock.block);
            try std.testing.expectEqualSlices(u64, &.{ 101, 102, 103, 302 }, Mock.values[7..11]);
            try std.testing.expectEqual(@as(u64, if (up) 201 else 202), Mock.values[3]);
            try std.testing.expectEqual(@as(u64, if (up) 203 else 204), Mock.values[4]);
            try std.testing.expectEqual(@as(u64, if (up) 128 else 32), Mock.values[5]);
            try std.testing.expectEqual(@as(u64, if (up) 32 else 128), Mock.values[6]);
            try std.testing.expectEqual(@as(u64, if (up) 9 else 0), Mock.values[2]);
            try std.testing.expectEqual(@as(u64, if (up) 1024 else 4096), Mock.values[11]);
            try std.testing.expectEqual(@as(u64, if (up) @as(u32, @bitCast(@as(f32, 8))) else 0), Mock.values[12]);
            try std.testing.expectEqual(@as(u64, 288), Mock.values[13]);
        }
    }
    try ex.run(stream, .up, .{ .x = 301, .stride = 4096, .slots = 9 }, l, p, 302, 9, 288);
    try std.testing.expectEqual(@as(usize, 1), Mock.function);
    l.relu2 = true;
    try ex.runVariant(.n16, stream, .up, .{ .x = 301, .stride = 4096, .slots = 9 }, l, p, 302, 9, 288);
    try std.testing.expectEqual(@as(usize, 17), Mock.function);
    ex.candidate_resident[1][3] = 13;
    try ex.runVariant(.n16, stream, .up, .{ .x = 301, .stride = 4096, .slots = 9 }, l, p, 302, 9, 288);
    try std.testing.expectEqual(@as(u32, 13), Mock.grid);
    Mock.function = 0;
    try ex.runVariant(.n16, stream, .up, .{ .x = 301, .stride = 4096 }, l, p, 302, 0, 288);
    try std.testing.expectEqual(@as(usize, 0), Mock.function);
    ex.candidates_loaded = false;
    try std.testing.expectError(error.CandidateNotLoaded, ex.runVariant(.n16, stream, .up, .{ .x = 301, .stride = 4096 }, l, p, 302, 9, 288));
    try std.testing.expectEqual(@as(usize, 0), Mock.function);
}
