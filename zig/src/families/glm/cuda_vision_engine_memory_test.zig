const std = @import("std");
const m = @import("cuda_vision_engine_memory.zig");
const st = @import("core").safetensors;
const t = std.testing;
const Mock = struct {
    var allocs: usize = 0;
    var frees: usize = 0;
    var copies: usize = 0;
    var syncs: usize = 0;
    var context_syncs: usize = 0;
    var retained_at_context: usize = 0;
    const Pending = struct { source: []const u8, head: [8]u8 };
    var pending: ?Pending = null;
    var live: usize = 0;
    var init_calls: usize = 0;
    var fail_alloc: bool = false;
    var null_alloc: bool = false;
    var odd_alloc: bool = false;
    var overflow_alloc: bool = false;
    var fail_copy: usize = 0;
    var fail_sync: usize = 0;
    var ptrs: [8]u64 = undefined;
    var sizes: [8]usize = undefined;
    var used: [8]bool = @splat(false);
    const Copy = struct { dst: u64, len: usize, head: [8]u8 };
    var records: [1024]Copy = undefined;
    fn reset() void {
        std.debug.assert(live == 0 and pending == null);
        allocs = 0;
        frees = 0;
        copies = 0;
        syncs = 0;
        context_syncs = 0;
        retained_at_context = 0;
        init_calls = 0;
        fail_alloc = false;
        null_alloc = false;
        odd_alloc = false;
        overflow_alloc = false;
        fail_copy = 0;
        fail_sync = 0;
        used = @splat(false);
    }
    fn init(_: c_uint) callconv(.c) m.cuda.abi.Result {
        init_calls += 1;
        return m.cuda.abi.error_not_ready;
    }
    fn alloc(p: *u64, n: usize) callconv(.c) m.cuda.abi.Result {
        allocs += 1;
        if (fail_alloc) return m.cuda.abi.error_not_ready;
        p.* = if (null_alloc) 0 else if (odd_alloc) 0x10000002 else if (overflow_alloc) std.math.maxInt(u64) - 255 else 0x100000000 + @as(u64, allocs) * 0x100000000;
        if (p.* != 0) {
            const i = allocs - 1;
            std.debug.assert(i < 8);
            ptrs[i] = p.*;
            sizes[i] = n;
            used[i] = true;
            live += 1;
        }
        return 0;
    }
    fn free(p: u64) callconv(.c) m.cuda.abi.Result {
        std.debug.assert(pending == null);
        for (0..8) |i| if (used[i] and ptrs[i] == p) {
            used[i] = false;
            live -= 1;
            frees += 1;
            return 0;
        };
        unreachable;
    }
    fn copy(dst: u64, src: ?*const anyopaque, n: usize) callconv(.c) m.cuda.abi.Result {
        std.debug.assert(pending == null);
        copies += 1;
        if (copies == fail_copy) return m.cuda.abi.error_not_ready;
        var found = false;
        for (0..8) |i| if (used[i] and dst >= ptrs[i] and dst - ptrs[i] <= sizes[i] and n <= sizes[i] - (dst - ptrs[i])) {
            found = true;
        };
        std.debug.assert(found and copies <= records.len and n > 0);
        var head: [8]u8 = @splat(0);
        const bytes: [*]const u8 = @ptrCast(src.?);
        @memcpy(head[0..@min(8, n)], bytes[0..@min(8, n)]);
        records[copies - 1] = .{ .dst = dst, .len = n, .head = head };
        pending = .{ .source = bytes[0..n], .head = head };
        return 0;
    }
    fn finishCopy() void {
        if (pending) |p| {
            std.debug.assert(live > 0);
            const n = @min(p.source.len, p.head.len);
            std.debug.assert(std.mem.eql(u8, p.head[0..n], p.source[0..n]));
            pending = null;
        }
    }
    fn sync(stream: m.cuda.abi.Stream) callconv(.c) m.cuda.abi.Result {
        std.debug.assert(stream == null);
        syncs += 1;
        if (fail_sync != 0 and (syncs == fail_sync or syncs == fail_sync + 1)) return m.cuda.abi.error_not_ready;
        finishCopy();
        return 0;
    }
    fn contextSync() callconv(.c) m.cuda.abi.Result {
        context_syncs += 1;
        retained_at_context = m.cuda.copy_source.retained();
        std.debug.assert(pending != null and retained_at_context > 0);
        finishCopy();
        return 0;
    }
    fn driver() m.cuda.Driver {
        var api: m.cuda.abi.Api = undefined;
        api.cuInit = init;
        api.cuMemAlloc_v2 = alloc;
        api.cuMemFree_v2 = free;
        api.cuMemcpyHtoD_v2 = copy;
        api.cuStreamSynchronize = sync;
        api.cuCtxSynchronize = contextSync;
        return .{ .lib = undefined, .api = api };
    }
};
fn production() !m.Config {
    return @import("cuda_vision_config.zig").parse(t.allocator, @embedFile("fixtures_cuda_config_exl3.json"));
}
fn tiny() !m.Config {
    var c = try production();
    c.hidden = 4;
    c.heads = 1;
    c.depth = 2;
    c.intermediate = 8;
    c.output = 4;
    c.projection_intermediate = 8;
    return c;
}
const Fixture = struct {
    a: std.mem.Allocator,
    w: m.host.Weights,
    fn init(a: std.mem.Allocator, c: m.Config) !Fixture {
        var specs = try m.host.Specs.init(c);
        const tensors = try a.alloc(st.Tensor, specs.count());
        errdefer a.free(tensors);
        var initialized: usize = 0;
        errdefer for (tensors[0..initialized]) |tensor| a.free(tensor.bytes);
        var name: [256]u8 = undefined;
        while (try specs.next(&name)) |spec| {
            const dtype: st.DType = switch (initialized % 3) {
                0 => .bf16,
                1 => .f16,
                else => .f32,
            };
            var n: usize = 1;
            for (spec.dims()) |dim| n *= dim;
            const bytes = try a.alloc(u8, n * dtype.size());
            const value = @as(f32, @floatFromInt(initialized % 7 + 1)) / 8;
            for (0..n) |i| switch (dtype) {
                .bf16 => std.mem.writeInt(u16, bytes[i * 2 ..][0..2], @truncate(@as(u32, @bitCast(value)) >> 16), .little),
                .f16 => std.mem.writeInt(u16, bytes[i * 2 ..][0..2], @bitCast(@as(f16, @floatCast(value))), .little),
                .f32 => std.mem.writeInt(u32, bytes[i * 4 ..][0..4], @bitCast(value), .little),
                else => unreachable,
            };
            tensors[initialized] = .{ .dtype = dtype, .rank = spec.rank, .shape = spec.shape, .bytes = bytes };
            initialized += 1;
        }
        return .{ .a = a, .w = .{ .c = c, .tensors = tensors } };
    }
    fn deinit(self: *Fixture) void {
        for (self.w.tensors) |tensor| self.a.free(tensor.bytes);
        self.a.free(self.w.tensors);
        self.* = undefined;
    }
};

test "vision memory production layouts cover all 347 weights and exact B,S activation geometry" {
    const c = try production();
    var weights = try m.WeightLayout.init(t.allocator, c, 2 << 30);
    defer weights.deinit();
    try t.expectEqual(@as(usize, 347), weights.regions.len);
    try t.expectEqual(@as(usize, 1127260160), weights.bytes);
    try t.expectEqual(@as(usize, 1024 * 1176 * 2), weights.regions[0].len);
    try t.expectEqualSlices(usize, &.{ 4096, 1024, 2, 2 }, weights.regions[3].shape[0..4]);
    try t.expectEqualSlices(usize, &.{ 3072, 1024 }, weights.regions[12].shape[0..2]);
    for (weights.regions, 0..) |r, i| {
        try t.expectEqual(@as(usize, 0), r.offset % 256);
        if (i > 0) try t.expect(r.offset >= weights.regions[i - 1].offset + weights.regions[i - 1].len);
    }
    const a = try m.ActivationLayout.init(c, .{ .groups = 2, .gh = 2, .gw = 4 }, .{ .convolution = 256, .attention = 1024, .linear = 512 }, 1 << 30);
    try t.expectEqualSlices(usize, &.{ 16, 1176 }, a.region(.pixels).shape[0..2]);
    try t.expectEqualSlices(usize, &.{ 16, 3, 16, 64 }, a.region(.qkv).shape[0..4]);
    try t.expectEqualSlices(usize, &.{ 8, 64 }, a.region(.cosine).shape[0..2]);
    try t.expectEqualSlices(usize, &.{ 4, 4096 }, a.region(.output).shape[0..2]);
    try t.expectEqualSlices(usize, &.{ 4, 10240 }, a.region(.projector_gate).shape[0..2]);
    try t.expectEqual(@as(usize, 1024), a.region(.workspace).len);
    for (a.regions, 0..) |r, i| {
        try t.expectEqual(@as(usize, 0), r.offset % 256);
        if (i > 0) try t.expect(r.offset >= a.regions[i - 1].offset + a.regions[i - 1].len);
    }
    try t.expectError(error.InvalidVisionGrid, (m.Geometry{ .groups = 257, .gh = 2, .gw = 2 }).rows());
    try t.expectError(error.InvalidVisionGrid, (m.Geometry{ .groups = 2, .gh = 2, .gw = 16000 }).rows());
    try t.expectError(error.InvalidVisionGrid, (m.Geometry{ .groups = 1, .gh = 3, .gw = 2 }).rows());
    try t.expectError(error.InvalidVisionGrid, (m.Geometry{ .groups = 1, .gh = std.math.maxInt(u32) - 1, .gw = 4 }).rows());
    var huge = c;
    huge.projection_intermediate = std.math.maxInt(c_int);
    try t.expectError(error.UnsupportedVisionMemoryShape, m.ActivationLayout.init(huge, .{ .groups = 2, .gh = 2, .gw = 2 }, .{}, std.math.maxInt(usize)));
    try t.expectError(error.VisionMemoryBudgetExceeded, m.WeightLayout.init(t.allocator, c, 1));
    try t.expectError(error.VisionMemoryBudgetExceeded, m.ActivationLayout.init(c, .{ .groups = 1, .gh = 2, .gw = 2 }, .{}, 1));
    try t.expectError(error.InvalidVisionMemoryShape, m.ActivationLayout.init(c, .{ .groups = 1, .gh = 2, .gw = 2 }, .{ .attention = std.math.maxInt(usize) }, std.math.maxInt(usize)));
}
test "vision memory genuine native checkpoint upload owns one pool with BF16 converted payloads" {
    var fixture = try Fixture.init(t.allocator, try tiny());
    defer fixture.deinit();
    Mock.reset();
    var driver = Mock.driver();
    const before = m.cuda.usage(false).device;
    var owned = try m.DeviceWeights.init(t.allocator, &driver, &fixture.w, 1 << 20);
    try t.expectEqual(@as(usize, 1), Mock.allocs);
    try t.expectEqual(fixture.w.tensors.len, Mock.copies);
    try t.expectEqual(Mock.copies, Mock.syncs);
    for (owned.layout.regions, 0..) |r, i| {
        try t.expectEqual(owned.pool.ptr + r.offset, Mock.records[i].dst);
        try t.expectEqual(r.len, Mock.records[i].len);
        const value = @as(f32, @floatFromInt(i % 7 + 1)) / 8;
        const bits: u16 = @truncate(@as(u32, @bitCast(value)) >> 16);
        try t.expectEqual(bits, std.mem.readInt(u16, Mock.records[i].head[0..2], .little));
    }
    var root = try owned.root(.projection);
    try t.expect(root.borrowed);
    root.free();
    try t.expectEqual(@as(usize, 0), Mock.frees);
    const block = try owned.block(1, .down_bias);
    try t.expect(block.borrowed);
    try t.expectEqual(@as(usize, 8), block.len);
    try t.expectError(error.InvalidVisionWeightIndex, owned.block(2, .norm1));
    owned.deinit();
    try t.expectEqual(@as(usize, 1), Mock.frees);
    try t.expectEqual(@as(usize, 0), Mock.live);
    try t.expectEqual(before, m.cuda.usage(false).device);
    try t.expectEqual(@as(usize, 0), Mock.init_calls);
}
test "vision memory bad checkpoint tables and late nonfinite tensors refuse before allocation" {
    var f = try Fixture.init(t.allocator, try tiny());
    defer f.deinit();
    Mock.reset();
    var d = Mock.driver();
    const old = f.w.tensors;
    f.w.tensors = old[0 .. old.len - 1];
    try t.expectError(error.InvalidVisionWeightTable, m.DeviceWeights.init(t.allocator, &d, &f.w, 1 << 20));
    f.w.tensors = old;
    const shape = f.w.tensors[0].shape;
    f.w.tensors[0].shape[0] += 1;
    try t.expectError(error.InvalidVisionWeightShape, m.DeviceWeights.init(t.allocator, &d, &f.w, 1 << 20));
    f.w.tensors[0].shape = shape;
    const rank = f.w.tensors[0].rank;
    f.w.tensors[0].rank = 255;
    try t.expectError(error.InvalidVisionWeightShape, m.DeviceWeights.init(t.allocator, &d, &f.w, 1 << 20));
    f.w.tensors[0].rank = rank;
    const original_bytes = f.w.tensors[0].bytes;
    f.w.tensors[0].bytes = original_bytes[0 .. original_bytes.len - 1];
    try t.expectError(error.InvalidVisionWeightShape, m.DeviceWeights.init(t.allocator, &d, &f.w, 1 << 20));
    f.w.tensors[0].bytes = original_bytes;
    const dtype = f.w.tensors[0].dtype;
    f.w.tensors[0].dtype = .u8;
    try t.expectError(error.UnsupportedVisionDType, m.DeviceWeights.init(t.allocator, &d, &f.w, 1 << 20));
    f.w.tensors[0].dtype = dtype;
    const last = f.w.tensors.len - 1;
    const bytes = @constCast(f.w.tensors[last].bytes);
    const old_bytes = bytes[0..4].*;
    // Last tensor is F32 in this two-block fixture. All earlier conversion
    // scratch is freed, and no device allocation occurs for this late failure.
    try t.expectEqual(st.DType.f32, f.w.tensors[last].dtype);
    std.mem.writeInt(u32, bytes[0..4], 0x7f7fffff, .little);
    try t.expectError(error.NonFiniteVisionWeight, m.DeviceWeights.init(t.allocator, &d, &f.w, 1 << 20));
    @memcpy(bytes[0..4], &old_bytes);
    try t.expectEqual(@as(usize, 0), Mock.allocs);
    try t.expectEqual(@as(usize, 0), Mock.copies);
}
test "vision memory every upload or wait failure and malformed driver allocation releases ownership" {
    var f = try Fixture.init(t.allocator, try tiny());
    defer f.deinit();
    var d = Mock.driver();
    const before = m.cuda.usage(false).device;
    for (1..f.w.tensors.len + 1) |i| {
        Mock.reset();
        Mock.fail_copy = i;
        try t.expectError(error.NotReady, m.DeviceWeights.init(t.allocator, &d, &f.w, 1 << 20));
        try t.expectEqual(@as(usize, 0), Mock.live);
        try t.expectEqual(before, m.cuda.usage(false).device);
        Mock.reset();
        Mock.fail_sync = i;
        const retained = m.cuda.copy_source.retained();
        try t.expectError(error.NotReady, m.DeviceWeights.init(t.allocator, &d, &f.w, 1 << 20));
        try t.expectEqual(@as(usize, 0), Mock.live);
        try t.expectEqual(before, m.cuda.usage(false).device);
        try t.expectEqual(@as(usize, 1), Mock.context_syncs);
        try t.expectEqual(retained + 1, Mock.retained_at_context);
        try t.expectEqual(retained, m.cuda.copy_source.retained());
    }
    Mock.reset();
    Mock.fail_alloc = true;
    try t.expectError(error.NotReady, m.DeviceWeights.init(t.allocator, &d, &f.w, 1 << 20));
    try t.expectEqual(@as(usize, 0), Mock.live);
    Mock.reset();
    Mock.null_alloc = true;
    try t.expectError(error.InvalidVisionDeviceAllocation, m.DeviceWeights.init(t.allocator, &d, &f.w, 1 << 20));
    try t.expectEqual(before, m.cuda.usage(false).device);
    Mock.reset();
    Mock.odd_alloc = true;
    try t.expectError(error.InvalidVisionDeviceAllocation, m.DeviceWeights.init(t.allocator, &d, &f.w, 1 << 20));
    try t.expectEqual(@as(usize, 0), Mock.live);
    Mock.reset();
    Mock.overflow_alloc = true;
    try t.expectError(error.InvalidVisionDeviceAllocation, m.DeviceWeights.init(t.allocator, &d, &f.w, 1 << 20));
    try t.expectEqual(@as(usize, 0), Mock.live);
    try t.expectEqual(@as(usize, 0), Mock.init_calls);
}
test "vision memory activation owner uploads shared RoPE and stages independent same-grid groups" {
    Mock.reset();
    var d = Mock.driver();
    const c = try tiny();
    const before = m.cuda.usage(false).device;
    var owned = try m.Activations.init(t.allocator, &d, c, .{ .groups = 2, .gh = 2, .gw = 2 }, .{}, 1 << 20);
    try t.expectEqual(@as(usize, 2), Mock.copies);
    try t.expect(!owned.pixels_ready);
    try t.expectEqual(@as(u32, 0x3f800000), std.mem.readInt(u32, Mock.records[0].head[0..4], .little));
    try t.expectEqual(@as(u32, 0), std.mem.readInt(u32, Mock.records[1].head[0..4], .little));
    const empty = try owned.buffer(.workspace);
    try t.expectEqual(@as(u64, 0), empty.ptr);
    try t.expectEqual(@as(usize, 0), empty.len);
    const pixels = try t.allocator.alloc(f32, 8 * 1176);
    defer t.allocator.free(pixels);
    @memset(pixels, 0.5);
    try owned.stagePixels(pixels);
    try t.expect(owned.pixels_ready);
    pixels[pixels.len - 1] = std.math.nan(f32);
    const calls = Mock.copies;
    try t.expectError(error.NonFiniteVisionPixels, owned.stagePixels(pixels));
    try t.expect(!owned.pixels_ready);
    try t.expectEqual(calls, Mock.copies);
    pixels[pixels.len - 1] = 0.5;
    const image: @import("cuda_vision_preprocess.zig").Prepared = .{ .canvas = .{ .height = 28, .width = 28 }, .content = .{ .height = 28, .width = 28 }, .grid = .{ 1, 2, 2 }, .patch_width = 1176, .pixels = pixels[0 .. 4 * 1176] };
    var second = image;
    second.pixels = pixels[4 * 1176 ..];
    try owned.stagePrepared(&.{ image, second });
    try t.expect(owned.pixels_ready);
    const before_late = Mock.copies;
    second.pixels[second.pixels.len - 1] = std.math.inf(f32);
    try t.expectError(error.NonFiniteVisionPixels, owned.stagePrepared(&.{ image, second }));
    try t.expectEqual(before_late, Mock.copies);
    second.pixels[second.pixels.len - 1] = 0.5;
    var wrong = image;
    wrong.grid = .{ 1, 2, 4 };
    const before_bad = Mock.copies;
    try t.expectError(error.InvalidVisionGrid, owned.stagePrepared(&.{ image, wrong }));
    try t.expectEqual(before_bad, Mock.copies);
    try t.expect(!owned.pixels_ready);
    Mock.fail_copy = Mock.copies + 1;
    try t.expectError(error.NotReady, owned.stagePixels(pixels));
    try t.expect(!owned.pixels_ready);
    owned.deinit();
    try t.expectEqual(@as(usize, 0), Mock.live);
    try t.expectEqual(before, m.cuda.usage(false).device);
    try t.expectEqual(@as(usize, 0), Mock.init_calls);
}
test "vision staging drains failed pixel and prepared uploads before destination reuse or release" {
    const c = try tiny();
    const pixels = try t.allocator.alloc(f32, 8 * 1176);
    defer t.allocator.free(pixels);
    @memset(pixels, 0.5);
    const first: @import("cuda_vision_preprocess.zig").Prepared = .{ .canvas = .{ .height = 28, .width = 28 }, .content = .{ .height = 28, .width = 28 }, .grid = .{ 1, 2, 2 }, .patch_width = 1176, .pixels = pixels[0 .. 4 * 1176] };
    var second = first;
    second.pixels = pixels[4 * 1176 ..];
    for (0..3) |case| for ([_]bool{ false, true }) |reuse| {
        Mock.reset();
        var d = Mock.driver();
        const retained = m.cuda.copy_source.retained();
        const before = m.cuda.usage(false).device;
        var owned = try m.Activations.init(t.allocator, &d, c, .{ .groups = 2, .gh = 2, .gw = 2 }, .{}, 1 << 20);
        Mock.fail_sync = Mock.syncs + (if (case == 2) @as(usize, 2) else 1);
        if (case == 0) {
            try t.expectError(error.NotReady, owned.stagePixels(pixels));
        } else {
            try t.expectError(error.NotReady, owned.stagePrepared(&.{ first, second }));
        }
        try t.expect(!owned.pixels_ready);
        try t.expectEqual(@as(usize, 1), Mock.context_syncs);
        try t.expectEqual(retained + 1, Mock.retained_at_context);
        try t.expectEqual(retained, m.cuda.copy_source.retained());
        try t.expectEqual(@as(usize, 1), Mock.live);
        try t.expect(Mock.pending == null);
        if (reuse) {
            try owned.stagePrepared(&.{ first, second });
            try t.expect(owned.pixels_ready);
        }
        owned.deinit();
        try t.expectEqual(@as(usize, 0), Mock.live);
        try t.expectEqual(@as(usize, 1), Mock.frees);
        try t.expectEqual(before, m.cuda.usage(false).device);
    };
}

test "vision memory allocation failure sweeps balance CPU and native ownership" {
    var fixture = try Fixture.init(t.allocator, try tiny());
    defer fixture.deinit();
    const Trial = struct {
        fn weights(a: std.mem.Allocator, w: *const m.host.Weights) !void {
            Mock.reset();
            var d = Mock.driver();
            const before = m.cuda.usage(false).device;
            defer {
                std.debug.assert(Mock.live == 0);
                std.debug.assert(m.cuda.usage(false).device == before);
            }
            var owner = try m.DeviceWeights.init(a, &d, w, 1 << 20);
            defer owner.deinit();
        }
        fn activations(a: std.mem.Allocator, c: m.Config) !void {
            Mock.reset();
            var d = Mock.driver();
            const before = m.cuda.usage(false).device;
            defer {
                std.debug.assert(Mock.live == 0);
                std.debug.assert(m.cuda.usage(false).device == before);
            }
            var owner = try m.Activations.init(a, &d, c, .{ .groups = 2, .gh = 2, .gw = 2 }, .{ .attention = 512 }, 1 << 20);
            defer owner.deinit();
        }
    };
    try t.checkAllAllocationFailures(t.allocator, Trial.weights, .{&fixture.w});
    try t.checkAllAllocationFailures(t.allocator, Trial.activations, .{fixture.w.c});
    for (1..3) |i| {
        Mock.reset();
        var d = Mock.driver();
        Mock.fail_copy = i;
        try t.expectError(error.NotReady, m.Activations.init(t.allocator, &d, fixture.w.c, .{ .groups = 2, .gh = 2, .gw = 2 }, .{}, 1 << 20));
        try t.expectEqual(@as(usize, 0), Mock.live);
        Mock.reset();
        Mock.fail_sync = i;
        const retained = m.cuda.copy_source.retained();
        try t.expectError(error.NotReady, m.Activations.init(t.allocator, &d, fixture.w.c, .{ .groups = 2, .gh = 2, .gw = 2 }, .{}, 1 << 20));
        try t.expectEqual(@as(usize, 0), Mock.live);
        try t.expectEqual(@as(usize, 1), Mock.context_syncs);
        try t.expectEqual(retained + 1, Mock.retained_at_context);
        try t.expectEqual(retained, m.cuda.copy_source.retained());
    }
}
