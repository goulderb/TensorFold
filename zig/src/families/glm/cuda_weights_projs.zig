const std = @import("std");
const cuda = @import("cuda");
const W = @import("cuda_weights.zig");
const b16 = @import("cuda_b16.zig");
const quant = @import("cuda_quant.zig");

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

pub fn finish(L: *W.Loader, comptime fmt: []const u8, args: anytype, n: usize, k: usize) !b16.Proj {
    if (L.tools.precision.dense == .bf16) return .{ .b16 = try L.put(fmt, args) };
    if (n == 0 or k == 0 or k % 64 != 0 or L.host.items.len != n * k * 2) return error.InvalidDenseShape;
    const qk = L.tools.quant orelse return error.NoQuantizer;
    const pack = L.tools.affine4 orelse return error.NoQuantizer;
    const name = try std.fmt.allocPrint(L.gpa, fmt, args);
    defer L.gpa.free(name);
    var source = try cuda.DeviceBuffer.fromHost(L.d, L.host.items);
    defer source.free();
    L.host.clearRetainingCapacity();
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
        .q => return error.HeadPrecisionRequiresBf16Source,
    };
    const n = L.w.share.vocab.len();
    const k = L.w.config.hidden;
    const npad = cuda.fp8.padded(n);
    const codes = try owned(L, "head", "codes", npad * k);
    const scales = try owned(L, "head", "scales", npad * (k / 64) * 4);
    try quant.fp8Rows(L.tools.quant orelse return error.NoQuantizer, L.stream, source, n, k, codes, scales);
    try L.stream.synchronize();
    L.w.head = .{ .q = .{ .fp8g = cuda.fp8.weight(codes, scales, n, k) } };
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
    const projection = try finish(&loader, "plain", .{}, 2, 2);
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(&original), M.slice(projection.b16, 8));
    try std.testing.expectEqual(@as(usize, 1), model.buffers.items.len);
    try std.testing.expectEqual(@as(usize, 8), model.buffers.items[0].len);
    try std.testing.expectEqual(@as(usize, 0), loader.host.items.len);
}

test "Q4 rejects an unsupported shape before uploading" {
    var loader: W.Loader = undefined;
    loader.tools = .{ .pack_dense = undefined, .precision = .{ .dense = .q4 } };
    try std.testing.expectError(error.InvalidDenseShape, finish(&loader, "invalid", .{}, 4, 63));
}
