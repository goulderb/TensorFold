const std = @import("std");
const config = @import("cuda_vision_config.zig");
const pre = @import("cuda_vision_preprocess.zig");
const weights = @import("cuda_vision_weights.zig");
const rope = @import("cuda_vision_rope.zig");
const st = @import("core").safetensors;
const testing = std.testing;
const Allocator = std.mem.Allocator;

fn production() !config.Config {
    return config.parse(testing.allocator, @embedFile("fixtures_cuda_config_exl3.json"));
}
fn pattern(a: Allocator, shape: pre.Shape) ![]u8 {
    const bytes = try a.alloc(u8, @as(usize, shape.height) * shape.width * 3);
    for (0..shape.height) |y| for (0..shape.width) |x| {
        const i = (y * shape.width + x) * 3;
        bytes[i] = @intCast((y * 7 + x * 3) % 256);
        bytes[i + 1] = @intCast((y * 2 + x * 11) % 256);
        bytes[i + 2] = @intCast((y + x) % 256);
    };
    return bytes;
}

test "geometry equals pinned Mia smart_resize and content_size, including extreme and tiny images" {
    const Case = struct { input: [4]u32, canvas: pre.Shape, content: pre.Shape };
    const cases = [_]Case{
        .{ .input = .{ 112, 112, 2048, 16 }, .canvas = .{ .height = 112, .width = 112 }, .content = .{ .height = 112, .width = 112 } },
        .{ .input = .{ 1080, 1920, 2048, 16 }, .canvas = .{ .height = 952, .width = 1680 }, .content = .{ .height = 945, .width = 1680 } },
        .{ .input = .{ 1920, 1080, 2048, 16 }, .canvas = .{ .height = 1680, .width = 952 }, .content = .{ .height = 1680, .width = 945 } },
        .{ .input = .{ 1, 8192, 2048, 16 }, .canvas = .{ .height = 28, .width = 8204 }, .content = .{ .height = 1, .width = 8204 } },
        .{ .input = .{ 8192, 1, 2048, 16 }, .canvas = .{ .height = 8204, .width = 28 }, .content = .{ .height = 8204, .width = 1 } },
        .{ .input = .{ 2, 3, 16, 16 }, .canvas = .{ .height = 28, .width = 28 }, .content = .{ .height = 18, .width = 28 } },
        .{ .input = .{ 30, 31, 16, 1 }, .canvas = .{ .height = 56, .width = 56 }, .content = .{ .height = 30, .width = 31 } },
        .{ .input = .{ 8192, 8192, 2048, 16 }, .canvas = .{ .height = 1260, .width = 1260 }, .content = .{ .height = 1260, .width = 1260 } },
        .{ .input = .{ 13, 17, 2048, 16 }, .canvas = .{ .height = 112, .width = 140 }, .content = .{ .height = 107, .width = 140 } },
    };
    for (cases) |cs| {
        const shape: pre.Shape = .{ .height = cs.input[0], .width = cs.input[1] };
        const budget: pre.Budget = .{ .max_tokens = cs.input[2], .min_tokens = cs.input[3] };
        const fitted = try pre.canvas(shape, budget);
        try testing.expectEqualDeep(cs.canvas, fitted);
        try testing.expectEqualDeep(cs.content, try pre.content(shape, fitted, budget));
        try testing.expect(@as(u64, fitted.height) * fitted.width / (28 * 28) <= budget.max_tokens);
    }
    try testing.expectError(error.InvalidVisionBudget, pre.canvas(.{ .height = 10, .width = 10 }, .{ .max_tokens = 0 }));
    try testing.expectError(error.InvalidImageShape, pre.canvas(.{ .height = 0, .width = 10 }, .{}));
}

// Hashes are complete float32 outputs of the pinned patch permutation and normalization,
// executed offline with Torch 2.9.0 on CPU, CUDA_VISIBLE_DEVICES empty, no checkpoint/model load.
test "RGB normalization, temporal duplication, merge order and padding match the CPU Torch oracle bitwise" {
    const a = testing.allocator;
    const c = try production();
    const cases = [_]struct { shape: pre.Shape, budget: pre.Budget, hash: []const u8 }{
        .{ .shape = .{ .height = 112, .width = 112 }, .budget = .{}, .hash = "f0576a48c949bd41b93fa3458e1f5a655bf4119bae0e05a38cb851fec5a6bc71" },
        .{ .shape = .{ .height = 28, .width = 56 }, .budget = .{ .min_tokens = 1, .max_tokens = 16 }, .hash = "da5120797c34c4ac332168f10f915d59f1c2bce596653e182e1e6fe214ae2a75" },
        .{ .shape = .{ .height = 30, .width = 31 }, .budget = .{ .min_tokens = 1, .max_tokens = 16 }, .hash = "5b461b617d5288df0f1cfaaeb91173c17416a98a7af8ec083da2e54c2d3729c5" },
    };
    for (cases) |cs| {
        const bytes = try pattern(a, cs.shape);
        defer a.free(bytes);
        var prepared = try pre.prepareRgb(a, c, .{ .shape = cs.shape, .bytes = bytes }, cs.budget, .{}, null);
        defer prepared.deinit(a);
        var digest: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(std.mem.sliceAsBytes(prepared.pixels), &digest, .{});
        try testing.expectEqualStrings(cs.hash, &std.fmt.bytesToHex(digest, .lower));
        try testing.expectEqual(@as(u32, 1176), prepared.patch_width);
        for (0..prepared.pixels.len / 1176) |r| for (0..3) |channel| {
            const start = r * 1176 + channel * 392;
            try testing.expectEqualSlices(f32, prepared.pixels[start..][0..196], prepared.pixels[start + 196 ..][0..196]);
        };
    }
}

test "missing native image/resize helpers and malformed RGB are explicit errors" {
    const a = testing.allocator;
    const c = try production();
    try testing.expectError(error.InvalidImageShape, pre.prepareRgb(a, c, .{ .shape = .{ .height = 112, .width = 112 }, .bytes = &.{} }, .{}, .{}, null));
    const bytes = try pattern(a, .{ .height = 13, .width = 17 });
    defer a.free(bytes);
    try testing.expectError(error.NativeBicubicResizeUnavailable, pre.prepareRgb(a, c, .{ .shape = .{ .height = 13, .width = 17 }, .bytes = bytes }, .{}, .{}, null));
    try testing.expectError(error.NativeImageDecoderUnavailable, pre.decodeInline(a, "data:image/png;base64,AQID", 16, .{}, null));
    try testing.expectError(error.ImageUrlsDisabled, pre.inlineBytes(a, "https://example.test/image.png", 16));
    try testing.expectError(error.InvalidImageDataUrl, pre.inlineBytes(a, "data:image/png;base64,AA?=", 16));
    try testing.expectError(error.ImageEncodedByteLimit, pre.inlineBytes(a, "data:image/png;base64,AQID", 2));
    const plain = try pre.inlineBytes(a, "data:image/png,%01%02%FF", 16);
    defer a.free(plain);
    try testing.expectEqualSlices(u8, &.{ 1, 2, 255 }, plain);
    const encoded = try pre.inlineBytes(a, "data:image/png;base64,AQID", 16);
    defer a.free(encoded);
    try testing.expectEqualSlices(u8, &.{ 1, 2, 3 }, encoded);
}

test "image positions bind exactly one native output row per image token" {
    const a = testing.allocator;
    const c = try production();
    const bytes = try pattern(a, .{ .height = 112, .width = 112 });
    defer a.free(bytes);
    var image = try pre.prepareRgb(a, c, .{ .shape = .{ .height = 112, .width = 112 }, .bytes = bytes }, .{}, .{}, null);
    defer image.deinit(a);
    var ids: [20]u32 = @splat(1);
    @memset(ids[2..18], c.image_token);
    const positions = try pre.imagePositions(a, c, &ids, &.{image}, 32);
    defer a.free(positions);
    try testing.expectEqual(@as(usize, 16), positions.len);
    for (positions, 2..) |p, want| try testing.expectEqual(want, p);
    try testing.expectError(error.ImagePromptTooLong, pre.imagePositions(a, c, &ids, &.{image}, 20));
    var corrupt = image;
    corrupt.grid[1] = 0;
    try testing.expectError(error.InvalidVisionGrid, pre.imagePositions(a, c, &ids, &.{corrupt}, 32));
    ids[2] = 1;
    try testing.expectError(error.ImageTokenCountMismatch, pre.imagePositions(a, c, &ids, &.{image}, 32));
}

fn tiny() !config.Config {
    var c = try production();
    c.hidden = 4;
    c.heads = 1;
    c.depth = 1;
    c.intermediate = 8;
    c.output = 4;
    c.projection_intermediate = 8;
    try c.check();
    return c;
}
const Variant = enum { good, missing, patch_rank, projection_shape, dtype };
fn weightImage(a: Allocator, c: config.Config, variant: Variant) ![]u8 {
    var writer: std.Io.Writer.Allocating = .init(a);
    defer writer.deinit();
    const w = &writer.writer;
    try w.writeByte('{');
    var specs = try weights.Specs.init(c);
    var buf: [256]u8 = undefined;
    var end: usize = 0;
    var first = true;
    while (try specs.next(&buf)) |s| {
        var spec = s;
        if (variant == .missing and std.mem.endsWith(u8, spec.name, "patch_embed.proj.bias")) continue;
        if (variant == .patch_rank and std.mem.endsWith(u8, spec.name, "patch_embed.proj.weight")) {
            spec.rank = 4;
            spec.shape = .{ c.hidden, 3, 2, 196, 1 };
        }
        if (variant == .projection_shape and std.mem.endsWith(u8, spec.name, "merger.proj.weight")) spec.shape[0] -= 1;
        const dtype: []const u8 = if (variant == .dtype and std.mem.endsWith(u8, spec.name, "patch_embed.proj.weight")) "U16" else "BF16";
        try w.print("{s}\"{s}\":{{\"dtype\":\"{s}\",\"shape\":[", .{ if (first) "" else ",", spec.name, dtype });
        first = false;
        var n: usize = 2;
        for (spec.dims(), 0..) |d, i| {
            try w.print("{s}{d}", .{ if (i == 0) "" else ",", d });
            n *= d;
        }
        try w.print("],\"data_offsets\":[{d},{d}]}}", .{ end, end + n });
        end += n;
    }
    try w.writeByte('}');
    const header = writer.written();
    const bytes = try a.alloc(u8, 8 + header.len + end);
    std.mem.writeInt(u64, bytes[0..8], header.len, .little);
    @memcpy(bytes[8..][0..header.len], header);
    @memset(bytes[8 + header.len ..], 0);
    return bytes;
}

test "vision loading validates every tensor, exact convolution ranks and later-file precedence" {
    const a = testing.allocator;
    const c = try tiny();
    const io = testing.io;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const good_bytes = try weightImage(a, c, .good);
    defer a.free(good_bytes);
    try tmp.dir.writeFile(io, .{ .sub_path = "good.safetensors", .data = good_bytes });
    var buf: [256]u8 = undefined;
    var good = try st.File.open(a, io, try std.fmt.bufPrint(&buf, ".zig-cache/tmp/{s}/good.safetensors", .{tmp.sub_path}));
    defer good.close(io);
    var loaded = try weights.Weights.load(a, c, &.{good});
    defer loaded.deinit(a);
    try testing.expectEqual(@as(usize, 25), loaded.tensors.len);
    try testing.expectEqual(@as(u8, 5), loaded.root(.patch_weight).rank);
    try testing.expectEqual(@as(usize, 4), loaded.block(0, .q_norm).dim(0));
    const converted = try weights.toBf16(a, loaded.root(.patch_bias));
    defer a.free(converted);
    try testing.expectEqualSlices(u16, &.{ 0, 0, 0, 0 }, converted);
    const cases = [_]struct { variant: Variant, err: anyerror }{
        .{ .variant = .missing, .err = error.MissingVisionWeight },
        .{ .variant = .patch_rank, .err = error.InvalidVisionWeightShape },
        .{ .variant = .projection_shape, .err = error.InvalidVisionWeightShape },
        .{ .variant = .dtype, .err = error.UnsupportedVisionDType },
    };
    for (cases) |cs| {
        const bytes = try weightImage(a, c, cs.variant);
        defer a.free(bytes);
        try tmp.dir.writeFile(io, .{ .sub_path = "bad.safetensors", .data = bytes });
        var bad = try st.File.open(a, io, try std.fmt.bufPrint(&buf, ".zig-cache/tmp/{s}/bad.safetensors", .{tmp.sub_path}));
        defer bad.close(io);
        try testing.expectError(cs.err, weights.Weights.load(a, c, &.{bad}));
        if (cs.variant != .missing) try testing.expectError(cs.err, weights.Weights.load(a, c, &.{ good, bad }));
        var preferred = try weights.Weights.load(a, c, &.{ bad, good });
        preferred.deinit(a);
    }
    const Trial = struct {
        fn run(alloc: Allocator, cfg: config.Config, f: st.File) !void {
            var w = try weights.Weights.load(alloc, cfg, &.{f});
            defer w.deinit(alloc);
        }
    };
    try testing.checkAllAllocationFailures(a, Trial.run, .{ c, good });
}

test "preprocessing, inline bytes and rotary tables balance allocation failures" {
    const a = testing.allocator;
    const c = try production();
    const bytes = try pattern(a, .{ .height = 112, .width = 112 });
    defer a.free(bytes);
    const PrepTrial = struct {
        fn run(alloc: Allocator, cfg: config.Config, rgb: []const u8) !void {
            var image = try pre.prepareRgb(alloc, cfg, .{ .shape = .{ .height = 112, .width = 112 }, .bytes = rgb }, .{}, .{}, null);
            defer image.deinit(alloc);
        }
    };
    const InlineTrial = struct {
        fn run(alloc: Allocator) !void {
            const data = try pre.inlineBytes(alloc, "data:image/png,%01%02%FF", 16);
            defer alloc.free(data);
        }
    };
    const RopeTrial = struct {
        fn run(alloc: Allocator, cfg: config.Config) !void {
            var table = try rope.Table.init(alloc, cfg, .{ 1, 2, 4 });
            defer table.deinit(alloc);
        }
    };
    try testing.checkAllAllocationFailures(a, PrepTrial.run, .{ c, bytes });
    try testing.checkAllAllocationFailures(a, InlineTrial.run, .{});
    try testing.checkAllAllocationFailures(a, RopeTrial.run, .{c});
}

test "public geometry and prepared metadata reject malformed or overflowing shapes" {
    const a = testing.allocator;
    const c = try production();
    try testing.expectError(error.InvalidImageShape, pre.content(.{ .height = std.math.maxInt(u32), .width = std.math.maxInt(u32) }, .{ .height = 28, .width = 28 }, .{}));
    try testing.expectError(error.InvalidImageShape, pre.content(.{ .height = 28, .width = 28 }, .{ .height = 27, .width = 28 }, .{}));
    try testing.expectError(error.InvalidImageShape, pre.content(.{ .height = 28, .width = 28 }, .{ .height = 2800, .width = 2800 }, .{}));
    var malformed: pre.Prepared = .{ .canvas = .{ .height = 28, .width = 28 }, .content = .{ .height = 28, .width = 28 }, .grid = .{ 1, 2, 2 }, .patch_width = 1176, .pixels = &.{} };
    try testing.expectError(error.InvalidVisionGrid, malformed.check(c));
    malformed.content.width = 29;
    try testing.expectError(error.InvalidVisionGrid, malformed.check(c));
    var bad_config = c;
    bad_config.merge = 0;
    try testing.expectError(error.UnsupportedVisionGeometry, malformed.check(bad_config));
    try testing.expectError(error.InvalidVisionGrid, rope.Table.init(a, c, .{ 1, 200, 200 }));
    bad_config = c;
    bad_config.theta = @bitCast(@as(u32, 1));
    try testing.expectError(error.InvalidVisionConfig, rope.Table.init(a, bad_config, .{ 1, 2, 4 }));
}

test "invalid owned decoder and resizer outputs are rejected and freed" {
    const a = testing.allocator;
    const c = try production();
    const Helper = struct {
        fn resize(_: ?*anyopaque, alloc: Allocator, _: pre.Rgb, _: pre.Shape) !pre.OwnedRgb {
            return .{ .shape = .{ .height = 1, .width = 1 }, .bytes = try alloc.alloc(u8, 3) };
        }
        fn decode(_: ?*anyopaque, alloc: Allocator, _: []const u8, _: pre.Limits) !pre.OwnedRgb {
            return .{ .shape = .{ .height = 0, .width = 1 }, .bytes = try alloc.alloc(u8, 3) };
        }
    };
    const bytes = try pattern(a, .{ .height = 13, .width = 17 });
    defer a.free(bytes);
    try testing.expectError(error.InvalidResizeOutput, pre.prepareRgb(a, c, .{ .shape = .{ .height = 13, .width = 17 }, .bytes = bytes }, .{}, .{}, .{ .run = Helper.resize }));
    try testing.expectError(error.InvalidImageShape, pre.decodeInline(a, "data:image/png;base64,AQID", 16, .{}, .{ .run = Helper.decode }));
}

test "BF16 conversion rounds ties to even and refuses malformed or nonfinite output" {
    const a = testing.allocator;
    var bytes: [16]u8 = undefined;
    const bits = [_]u32{ 0x3f808000, 0x3f818000, 0xbf808000, 0xbf818000 };
    for (bits, 0..) |v, i| std.mem.writeInt(u32, bytes[i * 4 ..][0..4], v, .little);
    var t: st.Tensor = .{ .dtype = .f32, .rank = 1, .shape = .{ 4, 1, 1, 1, 1 }, .bytes = &bytes };
    const out = try weights.toBf16(a, t);
    defer a.free(out);
    try testing.expectEqualSlices(u16, &.{ 0x3f80, 0x3f82, 0xbf80, 0xbf82 }, out);
    const Trial = struct {
        fn run(alloc: Allocator, tensor: st.Tensor) !void {
            const converted = try weights.toBf16(alloc, tensor);
            defer alloc.free(converted);
        }
    };
    try testing.checkAllAllocationFailures(a, Trial.run, .{t});
    for ([_]u32{ 0x7f800000, 0x7fc00000, 0x7f7fffff, 0xff7fffff }) |v| {
        std.mem.writeInt(u32, bytes[0..4], v, .little);
        try testing.expectError(error.NonFiniteVisionWeight, weights.toBf16(a, t));
    }
    t.bytes = bytes[0..15];
    try testing.expectError(error.InvalidVisionWeightShape, weights.toBf16(a, t));
    t.rank = 6;
    try testing.expectError(error.InvalidVisionWeightShape, weights.toBf16(a, t));
}

test "vision JSON parsing preserves allocation failure rather than masking it" {
    const Trial = struct {
        fn run(a: Allocator) !void {
            _ = try config.parse(a, @embedFile("fixtures_cuda_config_exl3.json"));
        }
    };
    try testing.checkAllAllocationFailures(testing.allocator, Trial.run, .{});
}
