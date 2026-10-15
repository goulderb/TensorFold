const std = @import("std");
const image = @import("cuda_vision_image.zig");
const exif = @import("cuda_vision_image_exif.zig");
const fixtures = @import("cuda_vision_image_fixtures.zig");
const t = std.testing;
const Allocator = std.mem.Allocator;

fn hex(a: Allocator, text: []const u8) ![]u8 {
    const bytes = try a.alloc(u8, text.len / 2);
    errdefer a.free(bytes);
    _ = try std.fmt.hexToBytes(bytes, text);
    return bytes;
}
fn hashEqual(expected: []const u8, bytes: []const u8) !void {
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &digest, .{});
    try t.expectEqualStrings(expected, &std.fmt.bytesToHex(digest, .lower));
}
fn orientation(value: usize) !image.Orientation {
    return image.Orientation.fromExif(@intCast(value));
}
fn pattern(a: Allocator, shape: image.Shape, channels: usize) ![]u8 {
    const bytes = try a.alloc(u8, @as(usize, shape.width) * shape.height * channels);
    const alpha = [_]u8{ 0, 1, 2, 63, 64, 127, 128, 129, 191, 254, 255, 17, 33, 65, 99 };
    for (0..bytes.len / channels) |i| {
        bytes[i * channels] = @intCast((37 * i + 11) % 256);
        bytes[i * channels + 1] = @intCast((73 * i + 19) % 256);
        bytes[i * channels + 2] = @intCast((109 * i + 23) % 256);
        if (channels == 4) bytes[i * channels + 3] = alpha[i % alpha.len];
    }
    return bytes;
}

test "all eight EXIF dimensions and complete RGBA colors match offline Pillow bytes" {
    const a = t.allocator;
    const rgba = try hex(a, fixtures.rgba_3x5);
    defer a.free(rgba);
    const generated = try pattern(a, .{ .height = 3, .width = 5 }, 4);
    defer a.free(generated);
    try t.expectEqualSlices(u8, rgba, generated);
    for (fixtures.rgba_oriented_rgb, 1..) |expected_hex, o| {
        const expected = try hex(a, expected_hex);
        defer a.free(expected);
        var out = try image.fromRgba(a, .{ .shape = .{ .height = 3, .width = 5 }, .bytes = rgba }, try orientation(o), .{});
        defer out.deinit(a);
        try t.expectEqualDeep(image.Shape{ .height = if (o >= 5) 5 else 3, .width = if (o >= 5) 3 else 5 }, out.shape);
        try t.expectEqualSlices(u8, expected, out.bytes);
        try out.borrowed().check(.{});
    }
    // The borrowed input never changes (including its transparent RGB values).
    try t.expectEqualSlices(u8, generated, rgba);
}

test "all eight opaque RGB transformations match offline Pillow complete-output hashes" {
    const a = t.allocator;
    const bytes = try pattern(a, .{ .height = 3, .width = 5 }, 3);
    defer a.free(bytes);
    for (fixtures.opaque_3x5, 1..) |expected, o| {
        var out = try image.orientRgb(a, .{ .shape = .{ .height = 3, .width = 5 }, .bytes = bytes }, try orientation(o), .{});
        defer out.deinit(a);
        try hashEqual(expected, out.bytes);
        try t.expectEqualDeep(image.Shape{ .height = if (o >= 5) 5 else 3, .width = if (o >= 5) 3 else 5 }, out.shape);
        try t.expect(out.bytes.ptr != bytes.ptr);
    }
}

test "single-pixel and degenerate axes match Pillow for all orientations" {
    const a = t.allocator;
    const shapes = [_]image.Shape{ .{ .height = 1, .width = 7 }, .{ .height = 7, .width = 1 }, .{ .height = 1, .width = 1 } };
    const reverse = [2][8]bool{
        .{ false, true, true, false, false, false, true, true },
        .{ false, false, true, true, false, true, true, false },
    };
    for (shapes, 0..) |shape, index| {
        const bytes = try pattern(a, shape, 4);
        defer a.free(bytes);
        for (1..9) |o| {
            var out = try image.fromRgba(a, .{ .shape = shape, .bytes = bytes }, try orientation(o), .{});
            defer out.deinit(a);
            const expected = if (index == 2) fixtures.one_pixel else if (reverse[index][o - 1]) fixtures.degenerate_reversed else fixtures.degenerate_normal;
            try hashEqual(expected, out.bytes);
            try t.expectEqualDeep(image.Shape{ .height = if (o >= 5) shape.width else shape.height, .width = if (o >= 5) shape.height else shape.width }, out.shape);
        }
    }
}

test "white compositing is exact for all 65536 channel-alpha pairs" {
    const a = t.allocator;
    const bytes = try a.alloc(u8, 256 * 256 * 4);
    defer a.free(bytes);
    for (0..256) |alpha| for (0..256) |channel| {
        const at = (alpha * 256 + channel) * 4;
        @memset(bytes[at..][0..3], @intCast(channel));
        bytes[at + 3] = @intCast(alpha);
    };
    var out = try image.fromRgba(a, .{ .shape = .{ .height = 256, .width = 256 }, .bytes = bytes }, .normal, .{});
    defer out.deinit(a);
    try hashEqual(fixtures.exhaustive_white, out.bytes);
    try t.expectEqualSlices(u8, &.{ 255, 255, 255 }, out.bytes[0..3]);
    // Negative control: truncating 0 at alpha=1 would give 254 only at this endpoint;
    // channel=1,alpha=128 rounds to 128, not floor(127.50)=127.
    const rounded = (128 * 256 + 1) * 3;
    try t.expectEqual(@as(u8, 128), out.bytes[rounded]);
}

test "invalid orientation, shape, limits and byte lengths fail before allocation" {
    try t.expectError(error.InvalidExifOrientation, image.Orientation.fromExif(0));
    try t.expectError(error.InvalidExifOrientation, image.Orientation.fromExif(9));
    try t.expectError(error.InvalidExifOrientation, image.Orientation.fromExif(65535));
    var failing = t.FailingAllocator.init(t.allocator, .{ .fail_index = 0 });
    const a = failing.allocator();
    const bytes = [_]u8{ 1, 2, 3, 4 };
    const rgba: image.Rgba = .{ .shape = .{ .height = 1, .width = 1 }, .bytes = &bytes };
    try t.expectError(error.InvalidImageShape, image.fromRgba(a, .{ .shape = .{ .height = 0, .width = 1 }, .bytes = &bytes }, .normal, .{}));
    try t.expectError(error.InvalidImageShape, image.fromRgba(a, rgba, .normal, .{ .max_dimension = 0 }));
    try t.expectError(error.InvalidImageShape, image.fromRgba(a, rgba, .normal, .{ .max_pixels = 0 }));
    try t.expectError(error.InvalidImageShape, image.fromRgba(a, .{ .shape = .{ .height = 2, .width = 1 }, .bytes = &bytes }, .normal, .{ .max_pixels = 1 }));
    try t.expectError(error.InvalidImageShape, image.fromRgba(a, .{ .shape = .{ .height = 1, .width = 8193 }, .bytes = &bytes }, .normal, .{}));
    try t.expectError(error.InvalidImageShape, image.fromRgba(a, .{ .shape = rgba.shape, .bytes = bytes[0..3] }, .normal, .{}));
    try t.expectError(error.InvalidImageShape, image.fromRgba(a, .{ .shape = rgba.shape, .bytes = &.{ 1, 2, 3, 4, 5 } }, .normal, .{}));
    try t.expectError(error.InvalidImageShape, image.orientRgb(a, .{ .shape = rgba.shape, .bytes = &bytes }, .normal, .{}));
    const huge: image.Shape = .{ .height = std.math.maxInt(u32), .width = std.math.maxInt(u32) };
    const wide: image.Limits = .{ .max_dimension = std.math.maxInt(u32), .max_pixels = std.math.maxInt(usize) };
    try t.expectError(error.InvalidImageShape, image.fromRgba(a, .{ .shape = huge, .bytes = &bytes }, .normal, wide));
    try t.expectError(error.InvalidImageShape, image.orientRgb(a, .{ .shape = huge, .bytes = &bytes }, .normal, wide));
    // Exact budget accepted, so the first allocation's OOM is visible and not masked as invalid shape.
    try t.expectError(error.OutOfMemory, image.fromRgba(a, rgba, .rotate_90_cw, .{ .max_dimension = 1, .max_pixels = 1 }));
}

test "every conversion allocation failure is leak-free for all orientations" {
    const Trial = struct {
        fn run(a: Allocator) !void {
            const rgba = [_]u8{ 1, 2, 3, 127, 4, 5, 6, 255 };
            const rgb = [_]u8{ 1, 2, 3, 4, 5, 6 };
            for (1..9) |o| {
                var transparent = try image.fromRgba(a, .{ .shape = .{ .height = 1, .width = 2 }, .bytes = &rgba }, try orientation(o), .{});
                defer transparent.deinit(a);
                var rgb_out = try image.orientRgb(a, .{ .shape = .{ .height = 1, .width = 2 }, .bytes = &rgb }, try orientation(o), .{});
                defer rgb_out.deinit(a);
            }
        }
    };
    try t.checkAllAllocationFailures(t.allocator, Trial.run, .{});
}

fn put16(bytes: []u8, at: usize, value: u16, endian: std.builtin.Endian) void {
    std.mem.writeInt(u16, bytes[at..][0..2], value, endian);
}
fn put32(bytes: []u8, at: usize, value: u32, endian: std.builtin.Endian) void {
    std.mem.writeInt(u32, bytes[at..][0..4], value, endian);
}
fn tiff(endian: std.builtin.Endian, value: u16) [26]u8 {
    var bytes: [26]u8 = @splat(0);
    @memcpy(bytes[0..2], if (endian == .little) "II" else "MM");
    put16(&bytes, 2, 42, endian);
    put32(&bytes, 4, 8, endian);
    put16(&bytes, 8, 1, endian);
    put16(&bytes, 10, 0x112, endian);
    put16(&bytes, 12, 3, endian);
    put32(&bytes, 14, 1, endian);
    put16(&bytes, 18, value, endian);
    return bytes;
}
fn reject(result: anyerror!?image.Orientation) !void {
    _ = result catch return;
    return error.TestExpectedError;
}

fn repairPngCrc(bytes: []u8, at: usize) void {
    const n: usize = std.mem.readInt(u32, bytes[at..][0..4], .big);
    var crc = std.hash.Crc32.init();
    crc.update(bytes[at + 4 ..][0 .. 4 + n]);
    put32(bytes, at + 8 + n, crc.final(), .big);
}
fn patchPillowOrientation(bytes: []u8, format: usize, value: u16) void {
    const at = std.mem.indexOf(u8, bytes, "MM\x00\x2a").?;
    put16(bytes, at + 18, value, .big);
    if (format == 1) repairPngCrc(bytes, at - 8);
}

test "24 JPEG PNG WebP metadata outputs are byte-identical to offline Pillow exports" {
    for (fixtures.encoded_hex, 0..) |text, format| {
        const bytes = try hex(t.allocator, text);
        defer t.allocator.free(bytes);
        for (1..9) |o| {
            patchPillowOrientation(bytes, format, @intCast(o));
            try hashEqual(fixtures.encoded_hashes[format][o - 1], bytes);
            try t.expectEqual(try orientation(o), (try exif.encodedOrientation(bytes, .{})).?);
        }
    }
}

test "TIFF IFD0 orientation is exact for both endians and bounded unrelated entries" {
    for ([_]std.builtin.Endian{ .little, .big }) |endian| {
        for (1..9) |o| {
            const bytes = tiff(endian, @intCast(o));
            try t.expectEqual(try orientation(o), (try exif.tiffOrientation(&bytes, .{})).?);
        }
        var bytes = tiff(endian, 1);
        put16(&bytes, 10, 0x110, endian);
        try t.expectEqual(@as(?image.Orientation, null), try exif.tiffOrientation(&bytes, .{}));
        put16(&bytes, 8, 0, endian);
        put32(&bytes, 10, 0, endian);
        try t.expectEqual(@as(?image.Orientation, null), try exif.tiffOrientation(bytes[0..14], .{}));
        put32(&bytes, 4, 0, endian);
        try t.expectEqual(@as(?image.Orientation, null), try exif.tiffOrientation(bytes[0..8], .{}));
    }
}

test "TIFF malformed headers, offsets, counts, next pointers and tags are refused" {
    for ([_]std.builtin.Endian{ .little, .big }) |endian| {
        const valid = tiff(endian, 6);
        for (0..valid.len) |n| try reject(exif.tiffOrientation(valid[0..n], .{}));
        var bytes = valid;
        bytes[0] = 'X';
        try t.expectError(error.InvalidImageMetadata, exif.tiffOrientation(&bytes, .{}));
        bytes = valid;
        put16(&bytes, 2, 43, endian); // BigTIFF is intentionally unsupported.
        try t.expectError(error.InvalidImageMetadata, exif.tiffOrientation(&bytes, .{}));
        for ([_]u32{ 2, 7, 9, 26, 0xfffffffe, 0xffffffff }) |offset| {
            bytes = valid;
            put32(&bytes, 4, offset, endian);
            try reject(exif.tiffOrientation(&bytes, .{}));
        }
        bytes = valid;
        put16(&bytes, 8, 2, endian);
        try reject(exif.tiffOrientation(&bytes, .{}));
        for ([_]u32{ 2, 8, 10, 25, 26, 0xfffffffe, 0xffffffff }) |next| {
            bytes = valid;
            put32(&bytes, 22, next, endian);
            try reject(exif.tiffOrientation(&bytes, .{}));
        }
        for ([_]u16{ 0, 9, 65535 }) |value| {
            bytes = tiff(endian, value);
            try t.expectError(error.InvalidExifOrientation, exif.tiffOrientation(&bytes, .{}));
        }
        bytes = valid;
        put16(&bytes, 12, 4, endian);
        try t.expectError(error.InvalidImageMetadata, exif.tiffOrientation(&bytes, .{}));
        bytes = valid;
        put32(&bytes, 14, 2, endian);
        try t.expectError(error.InvalidImageMetadata, exif.tiffOrientation(&bytes, .{}));
        // Unknown entry's out-of-line data is checked, including count*unit overflow on 32-bit.
        bytes = valid;
        put16(&bytes, 10, 0x110, endian);
        put16(&bytes, 12, 12, endian);
        put32(&bytes, 14, 0xffffffff, endian);
        put32(&bytes, 18, 8, endian);
        try reject(exif.tiffOrientation(&bytes, .{}));
        bytes = valid;
        put16(&bytes, 10, 0x110, endian);
        put16(&bytes, 12, 7, endian);
        put32(&bytes, 14, 5, endian);
        put32(&bytes, 18, 10, endian);
        try t.expectError(error.InvalidImageMetadata, exif.tiffOrientation(&bytes, .{}));
    }
}

test "duplicate TIFF orientation is refused even when both values agree" {
    for ([_]std.builtin.Endian{ .little, .big }) |endian| {
        var bytes: [38]u8 = @splat(0);
        const valid = tiff(endian, 6);
        @memcpy(bytes[0..22], valid[0..22]);
        @memcpy(bytes[22..34], valid[10..22]);
        put16(&bytes, 8, 2, endian);
        try t.expectError(error.DuplicateExifMetadata, exif.tiffOrientation(&bytes, .{}));
    }
}

test "every encoded fixture prefix is refused, including truncation after successful EXIF" {
    for (fixtures.encoded_hex, 0..) |text, format| {
        const bytes = try hex(t.allocator, text);
        defer t.allocator.free(bytes);
        for (0..bytes.len) |n| try reject(exif.encodedOrientation(bytes[0..n], .{}));
        for ([_]u16{ 0, 9, 65535 }) |value| {
            patchPillowOrientation(bytes, format, value);
            try t.expectError(error.InvalidExifOrientation, exif.encodedOrientation(bytes, .{}));
        }
    }
}

fn jpegCarrier(a: Allocator, metadata: []const u8, repeat: usize) ![]u8 {
    const size = 4 + repeat * (metadata.len + 10);
    const out = try a.alloc(u8, size);
    @memcpy(out[0..2], "\xff\xd8");
    var at: usize = 2;
    for (0..repeat) |_| {
        @memcpy(out[at..][0..2], "\xff\xe1");
        put16(out, at + 2, @intCast(metadata.len + 8), .big);
        @memcpy(out[at + 4 ..][0..6], "Exif\x00\x00");
        @memcpy(out[at + 10 ..][0..metadata.len], metadata);
        at += metadata.len + 10;
    }
    @memcpy(out[at..][0..2], "\xff\xd9");
    return out;
}
fn pngChunk(bytes: []u8, at: usize, kind: *const [4]u8, payload: []const u8) usize {
    put32(bytes, at, @intCast(payload.len), .big);
    @memcpy(bytes[at + 4 ..][0..4], kind);
    @memcpy(bytes[at + 8 ..][0..payload.len], payload);
    repairPngCrc(bytes, at);
    return at + 12 + payload.len;
}
fn pngCarrier(a: Allocator, metadata: []const u8, repeat: usize) ![]u8 {
    const size = 8 + 25 + repeat * (metadata.len + 12) + 12;
    const bytes = try a.alloc(u8, size);
    @memcpy(bytes[0..8], "\x89PNG\r\n\x1a\n");
    var at = pngChunk(bytes, 8, "IHDR", &[_]u8{ 0, 0, 0, 5, 0, 0, 0, 3, 8, 2, 0, 0, 0 });
    for (0..repeat) |_| at = pngChunk(bytes, at, "eXIf", metadata);
    _ = pngChunk(bytes, at, "IEND", &.{});
    return bytes;
}
fn webpCarrier(a: Allocator, metadata: []const u8, repeat: usize, prefix: bool) ![]u8 {
    const n = metadata.len + @as(usize, if (prefix) 6 else 0);
    const bytes = try a.alloc(u8, 12 + repeat * (8 + n + n % 2));
    @memcpy(bytes[0..4], "RIFF");
    put32(bytes, 4, @intCast(bytes.len - 8), .little);
    @memcpy(bytes[8..12], "WEBP");
    var at: usize = 12;
    for (0..repeat) |_| {
        @memcpy(bytes[at..][0..4], "EXIF");
        put32(bytes, at + 4, @intCast(n), .little);
        at += 8;
        if (prefix) {
            @memcpy(bytes[at..][0..6], "Exif\x00\x00");
            at += 6;
        }
        @memcpy(bytes[at..][0..metadata.len], metadata);
        at += metadata.len;
        if (n % 2 != 0) {
            bytes[at] = 0;
            at += 1;
        }
    }
    return bytes;
}

test "both TIFF endians and WebP optional Exif prefix work in metadata-only carriers" {
    for ([_]std.builtin.Endian{ .little, .big }) |endian| for (1..9) |o| {
        const metadata = tiff(endian, @intCast(o));
        const jpeg_bytes = try jpegCarrier(t.allocator, &metadata, 1);
        defer t.allocator.free(jpeg_bytes);
        const png_bytes = try pngCarrier(t.allocator, &metadata, 1);
        defer t.allocator.free(png_bytes);
        const webp_bytes = try webpCarrier(t.allocator, &metadata, 1, false);
        defer t.allocator.free(webp_bytes);
        const prefixed = try webpCarrier(t.allocator, &metadata, 1, true);
        defer t.allocator.free(prefixed);
        for ([_][]const u8{ jpeg_bytes, png_bytes, webp_bytes, prefixed }) |bytes|
            try t.expectEqual(try orientation(o), (try exif.encodedOrientation(bytes, .{})).?);
        // These carriers have NO compressed pixels: metadata success is not decode success.
    };
}

test "duplicate EXIF blocks in all containers are refused" {
    const metadata = tiff(.little, 6);
    const jpeg_bytes = try jpegCarrier(t.allocator, &metadata, 2);
    defer t.allocator.free(jpeg_bytes);
    const png_bytes = try pngCarrier(t.allocator, &metadata, 2);
    defer t.allocator.free(png_bytes);
    const webp_bytes = try webpCarrier(t.allocator, &metadata, 2, false);
    defer t.allocator.free(webp_bytes);
    for ([_][]const u8{ jpeg_bytes, png_bytes, webp_bytes }) |bytes|
        try t.expectError(error.DuplicateExifMetadata, exif.encodedOrientation(bytes, .{}));
}

test "JPEG marker lengths, stuffing, restart, SOS and Exif signature are bounded" {
    for ([_][]const u8{
        "\xff\xd8\xff\xe1\x00\x00\xff\xd9", // segment length < 2
        "\xff\xd8\xff\xe1\xff\xff\xff\xd9", // segment end outside input
        "\xff\xd8\x00\xff\xd9", // nonmarker in headers
        "\xff\xd8\xff\x00\xff\xd9", // stuffing outside scan
        "\xff\xd8\xff\xd0\xff\xd9", // restart outside scan
        "\xff\xd8\xff\xd8\xff\xd9", // repeated SOI
        "\xff\xd8\xff\xe1\x00\x06Exif\xff\xd9", // missing signature NULs
        "\xff\xd8\xff\xda\x00\x08\x00\x00\x00\x00\x00\x00\xff\xd9", // zero scan components
        "\xff\xd8\xff\xda\x00\x08\x01\x01\x00\x00\x3f\x00\xff", // scan missing EOI
    }) |bytes| try reject(exif.encodedOrientation(bytes, .{}));
    const scanned = "\xff\xd8\xff\xff\xda\x00\x08\x01\x01\x00\x00\x3f\x00\x23\xff\x00\x45\xff\xd0\x67\xff\xd9";
    try t.expectEqual(@as(?image.Orientation, null), try exif.encodedOrientation(scanned, .{}));
    try t.expectError(error.ImageMetadataLimitExceeded, exif.encodedOrientation(scanned, .{ .max_chunks = 2 }));
    // Non-EXIF APP1 (XMP) is not misinterpreted as a TIFF or orientation source.
    try t.expectEqual(@as(?image.Orientation, null), try exif.encodedOrientation("\xff\xd8\xff\xe1\x00\x05XMP\xff\xd9", .{}));
}

test "PNG CRC, chunk length, chunk type, IEND and IHDR framing are checked" {
    const bytes = try hex(t.allocator, fixtures.png);
    defer t.allocator.free(bytes);
    const original = try t.allocator.dupe(u8, bytes);
    defer t.allocator.free(original);
    bytes[bytes.len - 1] ^= 1;
    try t.expectError(error.InvalidImageMetadata, exif.encodedOrientation(bytes, .{}));
    @memcpy(bytes, original);
    put32(bytes, 33, 0xffffffff, .big);
    try t.expectError(error.InvalidImageMetadata, exif.encodedOrientation(bytes, .{}));
    @memcpy(bytes, original);
    bytes[37] = '1';
    try t.expectError(error.InvalidImageMetadata, exif.encodedOrientation(bytes, .{}));
    @memcpy(bytes, original);
    bytes[39] = 'i'; // PNG's reserved third chunk letter must be uppercase.
    try t.expectError(error.InvalidImageMetadata, exif.encodedOrientation(bytes, .{}));
    @memcpy(bytes, original);
    @memcpy(bytes[12..16], "IDAT");
    repairPngCrc(bytes, 8);
    try t.expectError(error.InvalidImageMetadata, exif.encodedOrientation(bytes, .{}));
    const duplicate_header = try pngCarrier(t.allocator, &@as([13]u8, @splat(0)), 1);
    defer t.allocator.free(duplicate_header);
    @memcpy(duplicate_header[37..41], "IHDR");
    repairPngCrc(duplicate_header, 33);
    try t.expectError(error.InvalidImageMetadata, exif.encodedOrientation(duplicate_header, .{}));
}

test "WebP RIFF/chunk overflow, odd padding and malformed EXIF are refused" {
    const metadata = tiff(.little, 8);
    var odd: [27]u8 = undefined;
    @memcpy(odd[0..26], &metadata);
    odd[26] = 0;
    const bytes = try webpCarrier(t.allocator, &odd, 1, false);
    defer t.allocator.free(bytes);
    try t.expectEqual(image.Orientation.rotate_90_ccw, (try exif.encodedOrientation(bytes, .{})).?);
    bytes[bytes.len - 1] = 1;
    try t.expectError(error.InvalidImageMetadata, exif.encodedOrientation(bytes, .{}));
    bytes[bytes.len - 1] = 0;
    try t.expectError(error.InvalidImageMetadata, exif.encodedOrientation(bytes[0 .. bytes.len - 1], .{}));
    put32(bytes, 4, @intCast(bytes.len - 9), .little);
    try t.expectError(error.InvalidImageMetadata, exif.encodedOrientation(bytes[0 .. bytes.len - 1], .{}));
    put32(bytes, 4, @intCast(bytes.len - 8), .little);
    put32(bytes, 16, 0xffffffff, .little);
    try t.expectError(error.InvalidImageMetadata, exif.encodedOrientation(bytes, .{}));
    put32(bytes, 4, 0xffffffff, .little);
    try t.expectError(error.InvalidImageMetadata, exif.encodedOrientation(bytes, .{}));
}

test "encoded input, EXIF size, chunk counts and IFD entries enforce exact budgets" {
    const bytes = try hex(t.allocator, fixtures.png);
    defer t.allocator.free(bytes);
    const exact: exif.MetadataLimits = .{ .max_encoded_bytes = bytes.len, .max_exif_bytes = 26, .max_chunks = 4, .max_ifd_entries = 1 };
    try t.expectEqual(image.Orientation.rotate_90_cw, (try exif.encodedOrientation(bytes, exact)).?);
    try t.expectError(error.ImageEncodedByteLimit, exif.encodedOrientation(bytes, .{ .max_encoded_bytes = bytes.len - 1 }));
    try t.expectError(error.ImageMetadataLimitExceeded, exif.encodedOrientation(bytes, .{ .max_exif_bytes = 25 }));
    try t.expectError(error.ImageMetadataLimitExceeded, exif.encodedOrientation(bytes, .{ .max_chunks = 3 }));
    for ([_]exif.MetadataLimits{
        .{ .max_encoded_bytes = 0 }, .{ .max_exif_bytes = 0 }, .{ .max_chunks = 0 }, .{ .max_ifd_entries = 0 },
    }) |limits| try t.expectError(error.InvalidImageMetadataBudget, exif.encodedOrientation(bytes, limits));
    var metadata = tiff(.little, 6);
    put16(&metadata, 8, 65535, .little);
    try t.expectError(error.ImageMetadataLimitExceeded, exif.tiffOrientation(&metadata, .{}));
    const two: [38]u8 = @splat(0);
    var two_tags = two;
    metadata = tiff(.little, 6);
    @memcpy(two_tags[0..22], metadata[0..22]);
    @memcpy(two_tags[22..34], metadata[10..22]);
    put16(&two_tags, 8, 2, .little);
    try t.expectError(error.ImageMetadataLimitExceeded, exif.tiffOrientation(&two_tags, .{ .max_ifd_entries = 1 }));
    try t.expectError(error.UnsupportedImageFormat, exif.encodedOrientation("GIF89a", .{}));
    try t.expectError(error.UnsupportedImageFormat, exif.encodedOrientation("RIFF\x04\x00\x00\x00WAVE", .{}));
}

test "bounded byte mutations of metadata never panic or allocate" {
    // Exhaustive one-byte substitution of TIFF catches parser arithmetic and enum traps.
    var metadata = tiff(.little, 6);
    const valid = metadata;
    for (0..metadata.len) |at| for (0..256) |value| {
        metadata = valid;
        metadata[at] = @intCast(value);
        const result = exif.tiffOrientation(&metadata, .{}) catch continue;
        if (result) |o| try t.expect(@backingInt(o) >= 1 and @backingInt(o) <= 8);
    };
    for (fixtures.encoded_hex) |text| {
        const bytes = try hex(t.allocator, text);
        defer t.allocator.free(bytes);
        for (0..bytes.len) |at| {
            const old = bytes[at];
            bytes[at] ^= 0xff;
            _ = exif.encodedOrientation(bytes, .{}) catch {};
            bytes[at] = old;
        }
    }
}
