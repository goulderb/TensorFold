const std = @import("std");
const cuda = @import("cuda");
const W = @import("cuda_weights.zig");
const b16 = @import("cuda_b16.zig");
const quant = @import("cuda_quant.zig");
const precision = @import("cuda_weight_precision.zig");
const dense = @import("cuda_weight_source_dense.zig");

fn owned(L: *W.Loader, name: []const u8, suffix: []const u8, bytes: usize) !u64 {
    const label = try std.fmt.allocPrint(L.gpa, "{s}.{s}", .{ name, suffix });
    errdefer L.gpa.free(label);
    var buffer = try cuda.DeviceBuffer.alloc(L.d, bytes);
    errdefer buffer.free();
    try L.w.buffers.ensureUnusedCapacity(L.gpa, 1);
    try L.w.named.append(L.gpa, .{ .name = label, .ptr = buffer.ptr, .len = bytes });
    L.w.buffers.appendAssumeCapacity(buffer);
    return buffer.ptr;
}

fn fp8Projection(L: *W.Loader, name: []const u8, n: usize, k: usize, source: u64) !b16.Proj {
    if (n == 0 or k == 0 or k % 128 != 0) return error.InvalidFp8Shape;
    const qk = L.tools.quant orelse return error.NoQuantizer;
    const npad = cuda.fp8.padded(n);
    const codes = try owned(L, name, "codes", npad * k);
    const scales = try owned(L, name, "scales", npad * (k / 64) * 4);
    try quant.fp8Rows(qk, L.stream, source, n, k, codes, scales);
    try L.stream.synchronize();
    return .{ .q = .{ .fp8g = cuda.fp8.weight(codes, scales, n, k) } };
}

fn nvfp4Projection(L: *W.Loader, name: []const u8, n: usize, k: usize, source: u64) !b16.Proj {
    const qk = L.tools.quant orelse return error.NoQuantizer;
    const pack = L.tools.modelopt orelse return error.NoModeloptPacker;
    var codes = try cuda.DeviceBuffer.alloc(L.d, n * k / 2);
    defer codes.free();
    var scales = try cuda.DeviceBuffer.alloc(L.d, n * k / 16);
    defer scales.free();
    var global = try cuda.DeviceBuffer.alloc(L.d, @sizeOf(f32));
    defer global.free();
    var launched = false;
    defer if (launched) L.stream.synchronize() catch {};
    launched = true;
    try quant.nvfp4(qk, L.stream, source, 1, n, k, codes.ptr, scales.ptr, global.ptr);
    const npad = cuda.nvfp4.padded(n);
    const words = try owned(L, name, "words", npad * k / 2);
    const tiles = try owned(L, name, "scales", npad * k / 16);
    try pack.fp4Lane(L.stream, codes.ptr, scales.ptr, words, tiles, n, k);
    try L.stream.synchronize();
    launched = false;
    var scale: f32 = undefined;
    try global.download(0, std.mem.asBytes(&scale));
    if (!std.math.isFinite(scale) or scale <= 0) return error.InvalidNvfp4Scale;
    return .{ .q = .{ .nvfp4 = cuda.nvfp4.weight(words, tiles, scale, n, k) } };
}

fn nativeProjection(L: *W.Loader, name: []const u8, n: usize, k: usize, mode: cuda.ck.NativeMode) !b16.Proj {
    const npad = try cuda.ck.shape(n, k);
    const ck = L.tools.ck orelse return error.NoNativeDenseKernels;
    var amax: f32 = 0;
    for (0..n * k) |i| {
        const bits = @as(u16, L.host.items[2 * i]) | (@as(u16, L.host.items[2 * i + 1]) << 8);
        const value: f32 = @bitCast(@as(u32, bits) << 16);
        if (!std.math.isFinite(value)) return error.NonfiniteDenseSource;
        amax = @max(amax, @abs(value));
    }
    const scale = try cuda.ck.tensorScale(amax, mode);
    var source = try cuda.DeviceBuffer.fromHost(L.d, L.host.items);
    defer source.free();
    L.host.clearRetainingCapacity();
    var raw = try cuda.DeviceBuffer.alloc(L.d, npad * k / (if (mode == .fp8) @as(usize, 1) else 2));
    defer raw.free();
    var launched = false;
    defer if (launched) L.stream.synchronize() catch {};
    launched = true;
    try raw.fill8(0, L.stream.handle);
    if (mode == .fp8) {
        const pack = L.tools.modelopt orelse return error.NoModeloptPacker;
        const codes = try owned(L, name, "ck8.codes", npad * k);
        try ck.quantizeWeight8(L.stream, source.ptr, n, k, scale, raw.ptr);
        try pack.fp8Lane(L.stream, raw.ptr, codes, n, k);
        try L.stream.synchronize();
        launched = false;
        return .{ .q = .{ .ck8 = .{ .codes = codes, .scale = scale, .n = @intCast(n), .k = @intCast(k), .npad = @intCast(npad) } } };
    }
    var scales = try cuda.DeviceBuffer.alloc(L.d, npad * k / 16);
    defer scales.free();
    defer if (launched) L.stream.synchronize() catch {};
    try scales.fill8(0, L.stream.handle);
    const words = try owned(L, name, "ck4.words", npad * k / 2);
    const tiles = try owned(L, name, "ck4.scales", npad * k / 16);
    try ck.quantizeWeight4(L.stream, source.ptr, n, k, scale, raw.ptr, scales.ptr);
    try ck.pack(L.stream, raw.ptr, scales.ptr, 1, npad, k, 0, k, words, tiles);
    try L.stream.synchronize();
    launched = false;
    return .{ .q = .{ .ck4 = .{ .words = words, .scales = tiles, .scale = scale, .n = @intCast(n), .k = @intCast(k), .npad = @intCast(npad) } } };
}

pub fn sourceFinish(L: *W.Loader, group: precision.Group, comptime fmt: []const u8, args: anytype, p: dense.Packed) !b16.Proj {
    if (L.tools.precision.dense.get(group) != .source) return error.SourceDenseNeedsPack;
    const j = p.job();
    try L.host.appendSlice(L.gpa, p.codes);
    try L.host.appendSlice(L.gpa, p.scales);
    const weights = try L.put(fmt, args);
    try L.host.appendSlice(L.gpa, std.mem.asBytes(&j));
    const jobs = try L.put(fmt ++ ".job", args);
    const groups = try @import("core").exlr8.job.denseGroups(L.gpa, &.{j});
    defer L.gpa.free(groups);
    try L.host.appendSlice(L.gpa, std.mem.sliceAsBytes(groups));
    const gptr = try L.put(fmt ++ ".groups", args);
    return .{ .source = .{ .weights = weights, .jobs = jobs, .groups = gptr, .groups_n = @intCast(groups.len), .n = p.outs, .k = p.ins } };
}

pub fn finish(L: *W.Loader, group: precision.Group, comptime fmt: []const u8, args: anytype, n: usize, k: usize) !b16.Proj {
    const selected = L.tools.precision.dense.get(group);
    if (selected == .source) return error.SourceDenseNeedsPack;
    if (selected == .bf16) return .{ .b16 = try L.put(fmt, args) };
    const elements = std.math.mul(usize, n, k) catch return error.InvalidDenseShape;
    const source_bytes = std.math.mul(usize, elements, 2) catch return error.InvalidDenseShape;
    if (n == 0 or k == 0 or n > std.math.maxInt(i32) or k > std.math.maxInt(i32) or k % 64 != 0 or (selected == .fp8 and k % 128 != 0) or L.host.items.len != source_bytes) return error.InvalidDenseShape;
    if (selected == .fp8_native or selected == .nvfp4_native) {
        const name = try std.fmt.allocPrint(L.gpa, fmt, args);
        defer L.gpa.free(name);
        const result = try nativeProjection(L, name, n, k, if (selected == .fp8_native) .fp8 else .nvfp4);
        L.w.native_conversions[@backingInt(group)] += 1;
        return result;
    }
    if (selected == .nvfp4) {
        if (L.tools.modelopt == null) return error.NoModeloptPacker;
        for (0..elements) |i| {
            const bits = @as(u16, L.host.items[i * 2]) | (@as(u16, L.host.items[i * 2 + 1]) << 8);
            if ((bits & 0x7f80) == 0x7f80) return error.NonfiniteDenseSource;
        }
        const npad = cuda.nvfp4.padded(n);
        _ = std.math.mul(usize, npad, k) catch return error.InvalidDenseShape;
    }
    const qk = L.tools.quant orelse return error.NoQuantizer;
    const name = try std.fmt.allocPrint(L.gpa, fmt, args);
    defer L.gpa.free(name);
    var source = try cuda.DeviceBuffer.fromHost(L.d, L.host.items);
    defer source.free();
    L.host.clearRetainingCapacity();
    if (selected == .fp8) return fp8Projection(L, name, n, k, source.ptr);
    if (selected == .nvfp4) return nvfp4Projection(L, name, n, k, source.ptr);
    const pack = L.tools.affine4 orelse return error.NoQuantizer;
    var raw: [3]cuda.DeviceBuffer = undefined;
    var made: usize = 0;
    defer for (raw[0..made]) |*buffer| buffer.free();
    for (&raw, [_]usize{ n * k / 2, n * k / 32, n * k / 32 }) |*buffer, bytes| {
        buffer.* = try cuda.DeviceBuffer.alloc(L.d, bytes);
        made += 1;
    }
    try quant.q4Mse(qk, L.stream, source.ptr, n, k, raw[0].ptr, raw[1].ptr, raw[2].ptr);
    const layout = cuda.qlinear.Affine4.layout(n, k);
    const view: cuda.qlinear.Affine4 = .{
        .w = try owned(L, name, "words", layout.words),
        .s = try owned(L, name, "scales", layout.scales),
        .b = try owned(L, name, "biases", layout.scales),
        .n = n,
        .k = k,
        .npad = layout.npad,
    };
    try pack.pack(L.stream, raw[0].ptr, raw[1].ptr, raw[2].ptr, view);
    try L.stream.synchronize();
    return .{ .q = .{ .affine4 = view } };
}

pub fn head(L: *W.Loader) !void {
    if (L.tools.precision.head == .bf16) return;
    const source = switch (L.w.head) {
        .b16 => |ptr| ptr,
        .q, .source => return error.HeadPrecisionRequiresBf16Source,
    };
    const n = L.w.share.vocab.len();
    const k = L.w.config.hidden;
    L.w.head = try fp8Projection(L, "head", n, k, source);
    for (L.w.buffers.items, 0..) |buffer, i| if (buffer.ptr == source) {
        var old = L.w.buffers.swapRemove(i);
        old.free();
        break;
    };
    for (L.w.named.items, 0..) |named, i| if (named.ptr == source) {
        L.gpa.free(L.w.named.swapRemove(i).name);
        break;
    };
}

test "default precision uploads BF16 bytes without a quantizer" {
    const M = @import("cuda_test_driver.zig").Memory;
    const Upload = struct {
        fn copy(dst: u64, src: ?*const anyopaque, len: usize) callconv(.c) cuda.abi.Result {
            @memcpy(M.slice(dst, len), @as([*]const u8, @ptrCast(src.?))[0..len]);
            return 0;
        }
    };
    var d = M.driver();
    d.api.cuMemcpyHtoD_v2 = Upload.copy;
    var model: W.Weights = .{ .gpa = std.testing.allocator, .config = undefined, .share = undefined };
    defer model.deinit();
    var loader: W.Loader = .{ .gpa = std.testing.allocator, .d = &d, .stream = .{ .d = &d, .handle = null }, .pack_dense = undefined, .tools = .{ .pack_dense = undefined }, .w = &model, .ck = undefined };
    defer loader.host.deinit(std.testing.allocator);
    const original = [_]u16{ 0x3f80, 0xbf00, 0x0000, 0x4040 };
    try loader.host.appendSlice(std.testing.allocator, std.mem.sliceAsBytes(&original));
    const projection = try finish(&loader, .kda, "plain", .{}, 2, 2);
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(&original), M.slice(projection.b16, 8));
    try std.testing.expectEqual(@as(usize, 1), model.buffers.items.len);
    try std.testing.expectEqual(@as(usize, 8), model.buffers.items[0].len);
    try std.testing.expectEqual(@as(usize, 0), loader.host.items.len);
}

test "source dense finish refuses a BF16 host gather" {
    var loader: W.Loader = undefined;
    loader.tools = .{ .pack_dense = undefined, .precision = .{ .dense = .uniform(.source) } };
    try std.testing.expectError(error.SourceDenseNeedsPack, finish(&loader, .kda, "source", .{}, 2, 2));
}

test "Q4 rejects an unsupported shape before uploading" {
    var loader: W.Loader = undefined;
    loader.tools = .{ .pack_dense = undefined, .precision = .{ .dense = .uniform(.q4) } };
    try std.testing.expectError(error.InvalidDenseShape, finish(&loader, .kda, "invalid", .{}, 4, 63));
}

test "FP8 dense rejects an unsupported input tile before driver access" {
    var loader: W.Loader = undefined;
    loader.tools = .{ .pack_dense = undefined, .precision = .{ .dense = .uniform(.fp8) } };
    try @import("std").testing.expectError(error.InvalidDenseShape, finish(&loader, .kda, "invalid", .{}, 4, 64));
}

test "NVFP4 rejects unsupported shape and nonfinite source before driver access" {
    var loader: W.Loader = undefined;
    loader.tools = .{ .pack_dense = undefined, .precision = .{ .dense = .uniform(.nvfp4) } };
    loader.host = .empty;
    defer loader.host.deinit(std.testing.allocator);
    try std.testing.expectError(error.InvalidDenseShape, finish(&loader, .shared, "invalid", .{}, 1, 32));
    try loader.host.appendNTimes(std.testing.allocator, 0, 128);
    try std.testing.expectError(error.NoModeloptPacker, finish(&loader, .shared, "missing", .{}, 1, 64));
    var pack: cuda.modelopt.Packer = undefined;
    loader.tools.modelopt = &pack;
    loader.host.items[1] = 0x7f;
    loader.host.items[0] = 0x80;
    try std.testing.expectError(error.NonfiniteDenseSource, finish(&loader, .shared, "nonfinite", .{}, 1, 64));
}
