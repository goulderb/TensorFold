const std = @import("std");
const core = @import("core");
const dflash = @import("cuda_dflash_weights.zig");
const a = std.testing.allocator;
const io = std.testing.io;

const Item = struct { name: []const u8, dtype: []const u8, shape: []const usize, bytes: []const u8 };

fn write(dir: std.Io.Dir, items: []const Item) !void {
    var h: std.Io.Writer.Allocating = .init(a);
    defer h.deinit();
    const w = &h.writer;
    try w.writeAll("{");
    var offset: usize = 0;
    for (items, 0..) |item, i| {
        if (i != 0) try w.writeAll(",");
        try w.print("\"{s}\":{{\"dtype\":\"{s}\",\"shape\":[", .{ item.name, item.dtype });
        for (item.shape, 0..) |dim, j| {
            if (j != 0) try w.writeAll(",");
            try w.print("{d}", .{dim});
        }
        try w.print("],\"data_offsets\":[{d},{d}]}}", .{ offset, offset + item.bytes.len });
        offset += item.bytes.len;
    }
    try w.writeAll("}");
    var image: std.ArrayList(u8) = .empty;
    defer image.deinit(a);
    var size: [8]u8 = undefined;
    std.mem.writeInt(u64, &size, h.written().len, .little);
    try image.appendSlice(a, &size);
    try image.appendSlice(a, h.written());
    for (items) |item| try image.appendSlice(a, item.bytes);
    try dir.writeFile(io, .{ .sub_path = "model.safetensors", .data = image.items });
}

fn open(tmp: *std.testing.TmpDir) !core.Checkpoint {
    var path: [256]u8 = undefined;
    return core.Checkpoint.openModel(a, io, try std.fmt.bufPrint(&path, ".zig-cache/tmp/{s}", .{tmp.sub_path}));
}

test "MXFP8 target head decodes rank rows exactly as fp8_source" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var codes: [4 * 64]u8 = @splat(0x38);
    codes[64 + 31] = 0xc0;
    codes[64 + 32] = 0x40;
    const scales = [_]u8{ 127, 128, 126, 129, 127, 125, 130, 127 };
    try write(tmp.dir, &.{
        .{ .name = "lm_head.weight", .dtype = "F8_E4M3", .shape = &.{ 4, 64 }, .bytes = &codes },
        .{ .name = "lm_head.weight_scale", .dtype = "U8", .shape = &.{ 4, 2 }, .bytes = &scales },
    });
    var ck = try open(&tmp);
    defer ck.close();
    const cut: dflash.Slice = .{ .axis = 0, .start = 1, .len = 2 };
    const got = try dflash.decodeTargetHead(a, &ck, cut);
    defer a.free(got);
    var want: [2 * 64]u16 = undefined;
    try core.fp8_source.mxfp8Bf16(try ck.get("lm_head.weight"), try ck.get("lm_head.weight_scale"), .{ .lo = 1, .hi = 3 }, core.fp8_source.all(64), &want);
    try std.testing.expectEqualSlices(u16, &want, got);
}

test "BF16 target head keeps stored bits" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const bits = [_]u16{ 1, 2, 3, 4, 5, 6, 7, 8 };
    try write(tmp.dir, &.{
        .{ .name = "lm_head.weight", .dtype = "BF16", .shape = &.{ 2, 4 }, .bytes = std.mem.sliceAsBytes(&bits) },
    });
    var ck = try open(&tmp);
    defer ck.close();
    const got = try dflash.decodeTargetHead(a, &ck, .{ .axis = 0, .start = 1, .len = 1 });
    defer a.free(got);
    try std.testing.expectEqualSlices(u16, bits[4..], got);
    const full = try dflash.decodeTargetHead(a, &ck, null);
    defer a.free(full);
    try std.testing.expectEqualSlices(u16, &bits, full);
}

test "F8 head without scale or with a non-MXFP8 recipe is refused" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const codes: [4 * 64]u8 = @splat(0x38);
    try write(tmp.dir, &.{
        .{ .name = "lm_head.weight", .dtype = "F8_E4M3", .shape = &.{ 4, 64 }, .bytes = &codes },
    });
    var missing = try open(&tmp);
    defer missing.close();
    try std.testing.expectError(error.UnsupportedDraftDtype, dflash.decodeTargetHead(a, &missing, null));

    var wrong = std.testing.tmpDir(.{});
    defer wrong.cleanup();
    const f32_scale = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    try write(wrong.dir, &.{
        .{ .name = "lm_head.weight", .dtype = "F8_E4M3", .shape = &.{ 4, 64 }, .bytes = &codes },
        .{ .name = "lm_head.weight_scale", .dtype = "F32", .shape = &.{ 4, 2 }, .bytes = std.mem.sliceAsBytes(&f32_scale) },
    });
    var pb = try open(&wrong);
    defer pb.close();
    try std.testing.expectError(error.UnsupportedDraftDtype, dflash.decodeTargetHead(a, &pb, null));

    var block = std.testing.tmpDir(.{});
    defer block.cleanup();
    const u8_scale = [_]u8{ 127, 127, 127, 127 };
    try write(block.dir, &.{
        .{ .name = "lm_head.weight", .dtype = "F8_E4M3", .shape = &.{ 4, 64 }, .bytes = &codes },
        .{ .name = "lm_head.weight_scale", .dtype = "U8", .shape = &.{ 4, 1 }, .bytes = &u8_scale },
    });
    var bad = try open(&block);
    defer bad.close();
    try std.testing.expectError(error.BadFp8Source, dflash.decodeTargetHead(a, &bad, null));

    var f16_dir = std.testing.tmpDir(.{});
    defer f16_dir.cleanup();
    const f16_bits = [_]u16{ 0, 0, 0, 0 };
    try write(f16_dir.dir, &.{
        .{ .name = "lm_head.weight", .dtype = "F16", .shape = &.{ 1, 4 }, .bytes = std.mem.sliceAsBytes(&f16_bits) },
    });
    var f16_ck = try open(&f16_dir);
    defer f16_ck.close();
    try std.testing.expectError(error.UnsupportedDraftDtype, dflash.decodeTargetHead(a, &f16_ck, null));
}
