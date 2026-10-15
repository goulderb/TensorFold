const std = @import("std");
const codec = @import("cuda_vision_codec.zig");
const fixtures = @import("cuda_vision_codec_fixtures.zig");
const pre = @import("cuda_vision_preprocess.zig");
const c = codec.c;
const a = std.testing.allocator;

const Mock = struct {
    probes: usize = 0,
    decodes: usize = 0,
    probe_status: c_int = c.TF_GLM_CODEC_OK,
    decode_status: c_int = c.TF_GLM_CODEC_OK,
    width: u32 = 3,
    height: u32 = 2,
    fn of(p: ?*anyopaque) *Mock {
        return @ptrCast(@alignCast(p.?));
    }
    fn probe(p: ?*anyopaque, _: u32, _: [*c]const u8, _: usize, _: u32, _: usize, width: [*c]u32, height: [*c]u32) callconv(.c) c_int {
        const self = of(p);
        self.probes += 1;
        width.* = self.width;
        height.* = self.height;
        return self.probe_status;
    }
    fn decode(p: ?*anyopaque, _: u32, _: [*c]const u8, _: usize, _: u32, _: u32, out: [*c]u8, size: usize) callconv(.c) c_int {
        const self = of(p);
        self.decodes += 1;
        if (self.decode_status != c.TF_GLM_CODEC_OK) return self.decode_status;
        std.debug.assert(size == fixtures.all[0].rgba.len);
        @memcpy(out[0..size], fixtures.all[0].rgba);
        return c.TF_GLM_CODEC_OK;
    }
    const api: codec.Api = .{
        .tf_glm_codec_abi = undefined,
        .tf_glm_codec_open = undefined,
        .tf_glm_codec_close = undefined,
        .tf_glm_codec_features = undefined,
        .tf_glm_codec_probe = probe,
        .tf_glm_codec_decode = decode,
    };
    fn backend(self: *Mock, features: u32) codec.Backend {
        return .{ .context = self, .api = &api, .features = features };
    }
};

test "format sniffing is bounded and covers every independent target fixture" {
    for (fixtures.all) |f| try std.testing.expectEqual(f.format, try codec.formatOf(f.encoded));
    try std.testing.expectError(error.ImageEncodedByteLimit, codec.formatOf(""));
    try std.testing.expectError(error.UnsupportedImageFormat, codec.formatOf("GIF89a"));
    const oversized = try a.alloc(u8, codec.max_encoded_bytes + 1);
    defer a.free(oversized);
    try std.testing.expectError(error.ImageEncodedByteLimit, codec.formatOf(oversized));
}
test "native paths are explicit absolute paths, not a library search or glob" {
    try std.testing.expectError(error.InvalidNativeCodecPath, codec.Library.open(a, .{ .shim = "relative.so" }, .{}));
    const absolute = if (@import("builtin").os.tag == .windows) "C:/tf-no-library.dll" else "/tf-no-library.so";
    try std.testing.expectError(error.InvalidNativeCodecPath, codec.Library.open(a, .{ .shim = absolute, .png = "libpng.so" }, .{}));
    try std.testing.expectError(error.NativeImageLibraryUnavailable, codec.Library.open(a, .{ .shim = absolute }, .{}));
}
test "missing codec features refuse rather than invent pixels" {
    var m: Mock = .{};
    for (fixtures.all) |f| {
        const expected = switch (f.format) {
            .png => error.NativePngDecoderUnavailable,
            .jpeg => error.NativeJpegDecoderUnavailable,
            .webp => error.NativeWebpDecoderUnavailable,
        };
        try std.testing.expectError(expected, m.backend(0).decode(a, f.encoded, .{}));
    }
    try std.testing.expectEqual(@as(usize, 0), m.probes + m.decodes);
}
test "fake backend tests ownership and composition only, not encoded decoding" {
    var m: Mock = .{};
    var rgba = try m.backend(c.TF_GLM_CODEC_PNG).rgba(a, fixtures.all[0].encoded, .{});
    defer rgba.deinit(a);
    try std.testing.expectEqualSlices(u8, fixtures.all[0].rgba, rgba.bytes);
    var rgb = try m.backend(c.TF_GLM_CODEC_PNG).decode(a, fixtures.all[0].encoded, .{});
    defer rgb.deinit(a);
    try std.testing.expectEqualSlices(u8, fixtures.all[0].rgb, rgb.bytes);
    try std.testing.expectEqual(@as(usize, 2), m.probes);
    try std.testing.expectEqual(@as(usize, 2), m.decodes);
}
test "invalid limits and dimensions fail before decode or pixel allocation" {
    var m: Mock = .{};
    const backend = m.backend(c.TF_GLM_CODEC_PNG);
    try std.testing.expectError(error.InvalidImageBudget, backend.rgba(a, fixtures.all[0].encoded, .{ .max_pixels = 0 }));
    try std.testing.expectError(error.InvalidImageBudget, backend.rgba(a, fixtures.all[0].encoded, .{ .max_dimension = 8193 }));
    try std.testing.expectError(error.InvalidImageShape, backend.rgba(a, fixtures.all[0].encoded, .{ .max_pixels = 5 }));
    try std.testing.expectEqual(@as(usize, 0), m.probes);
    m.width = 0;
    try std.testing.expectError(error.InvalidImageShape, backend.rgba(a, fixtures.all[0].encoded, .{}));
    m.width = std.math.maxInt(u32);
    try std.testing.expectError(error.InvalidImageShape, backend.rgba(a, fixtures.all[0].encoded, .{}));
    try std.testing.expectEqual(@as(usize, 0), m.decodes);
}
test "native errors and allocation failures release the intermediate RGBA" {
    var m: Mock = .{};
    const backend = m.backend(c.TF_GLM_CODEC_PNG);
    m.probe_status = c.TF_GLM_CODEC_COLOR;
    try std.testing.expectError(error.UnsupportedNativeImageColorMode, backend.decode(a, fixtures.all[0].encoded, .{}));
    m.probe_status = c.TF_GLM_CODEC_OK;
    m.decode_status = c.TF_GLM_CODEC_INVALID;
    try std.testing.expectError(error.InvalidEncodedImage, backend.decode(a, fixtures.all[0].encoded, .{}));
    m.decode_status = 999;
    try std.testing.expectError(error.InvalidNativeCodecResult, backend.rgba(a, fixtures.all[0].encoded, .{}));
    m.decode_status = c.TF_GLM_CODEC_OK;
    var failing = std.testing.FailingAllocator.init(a, .{ .fail_index = 0 });
    const before = m.decodes;
    try std.testing.expectError(error.OutOfMemory, backend.rgba(failing.allocator(), fixtures.all[0].encoded, .{}));
    try std.testing.expectEqual(before, m.decodes);
    var second = std.testing.FailingAllocator.init(a, .{ .fail_index = 1 });
    try std.testing.expectError(error.OutOfMemory, backend.decode(second.allocator(), fixtures.all[0].encoded, .{}));
}
test "PNG CRC and truncated container refuse before calling native code" {
    var m: Mock = .{};
    const bytes = try a.dupe(u8, fixtures.all[0].encoded);
    defer a.free(bytes);
    bytes[29] ^= 1;
    try std.testing.expectError(error.InvalidImageMetadata, m.backend(c.TF_GLM_CODEC_PNG).decode(a, bytes, .{}));
    try std.testing.expectError(error.InvalidImageMetadata, m.backend(c.TF_GLM_CODEC_PNG).decode(a, bytes[0..24], .{}));
    try std.testing.expectEqual(@as(usize, 0), m.probes);
}
test "APNG animation admission precedes native pixel decode" {
    const source = fixtures.all[0].encoded;
    const bytes = try a.alloc(u8, source.len + 20);
    defer a.free(bytes);
    @memcpy(bytes[0..33], source[0..33]);
    std.mem.writeInt(u32, bytes[33..37], 8, .big);
    @memcpy(bytes[37..41], "acTL");
    std.mem.writeInt(u32, bytes[41..45], 2, .big);
    std.mem.writeInt(u32, bytes[45..49], 0, .big);
    std.mem.writeInt(u32, bytes[49..53], std.hash.Crc32.hash(bytes[37..49]), .big);
    @memcpy(bytes[53..], source[33..]);
    var m: Mock = .{};
    try std.testing.expectError(error.AnimatedImagesUnsupported, m.backend(c.TF_GLM_CODEC_PNG).decode(a, bytes, .{}));
    try std.testing.expectEqual(@as(usize, 0), m.probes);
}
test "WebP animation and JPEG MPF refuse before native code" {
    const webp = "RIFF\x16\x00\x00\x00WEBPVP8X\x0a\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00";
    var m: Mock = .{};
    try std.testing.expectError(error.AnimatedImagesUnsupported, m.backend(c.TF_GLM_CODEC_WEBP).decode(a, webp, .{}));
    const source = fixtures.all[17].encoded;
    const jpeg = try a.alloc(u8, source.len + 8);
    defer a.free(jpeg);
    @memcpy(jpeg[0..2], source[0..2]);
    @memcpy(jpeg[2..10], "\xff\xe2\x00\x06MPF\x00");
    @memcpy(jpeg[10..], source[2..]);
    try std.testing.expectError(error.AnimatedImagesUnsupported, m.backend(c.TF_GLM_CODEC_JPEG).decode(a, jpeg, .{}));
    try std.testing.expectEqual(@as(usize, 0), m.probes);
}

fn env(name: [:0]const u8) ?[]const u8 {
    return std.mem.span(std.c.getenv(name) orelse return null);
}
fn native(format: codec.Format) !codec.Library {
    const shim = env("TF_GLM_CODEC_SHIM") orelse return error.SkipZigTest;
    const path = env(switch (format) {
        .png => "TF_GLM_CODEC_PNG",
        .jpeg => "TF_GLM_CODEC_JPEG",
        .webp => "TF_GLM_CODEC_WEBP",
    }) orelse return error.SkipZigTest;
    var paths: codec.Paths = .{ .shim = shim };
    switch (format) {
        .png => paths.png = path,
        .jpeg => paths.jpeg = path,
        .webp => paths.webp = path,
    }
    return codec.Library.open(a, paths, .{});
}
fn golden(format: codec.Format) !void {
    var lib = try native(format);
    defer lib.close();
    for (fixtures.all) |f| {
        if (f.format != format) continue;
        var rgba = try lib.backend().rgba(a, f.encoded, .{});
        defer rgba.deinit(a);
        try std.testing.expectEqual(f.shape, rgba.shape);
        try std.testing.expectEqualSlices(u8, f.rgba, rgba.bytes);
        const decoder = lib.decoder();
        var rgb = try decoder.run(decoder.context, a, f.encoded, .{});
        defer rgb.deinit(a);
        try std.testing.expectEqual(f.oriented, rgb.shape);
        try std.testing.expectEqualSlices(u8, f.rgb, rgb.bytes);
    }
}
test "actual native PNG equals complete Pillow goldens including raw gamma and orientation" {
    try golden(.png);
}
test "actual native JPEG equals complete Pillow goldens including progressive and Adobe CMYK" {
    try golden(.jpeg);
}
test "actual native WebP equals complete Pillow goldens including lossy alpha" {
    try golden(.webp);
}
test "actual shim refuses missing dependency paths and incompatible version" {
    const shim = env("TF_GLM_CODEC_SHIM") orelse return error.SkipZigTest;
    const missing = if (@import("builtin").os.tag == .windows) "C:/tf-no-png.dll" else "/tf-no-png.so";
    try std.testing.expectError(error.NativeImageLibraryUnavailable, codec.Library.open(a, .{ .shim = shim, .png = missing }, .{}));
    if (env("TF_GLM_CODEC_PNG")) |png| {
        try std.testing.expectError(error.NativeImageVersionMismatch, codec.Library.open(a, .{ .shim = shim, .png = png }, .{ .png = 1 }));
    }
}
test "actual PNG shim refuses a short output span without modifying canaries" {
    var lib = try native(.png);
    defer lib.close();
    var bytes: [28]u8 = @splat(0xa5);
    const f = fixtures.all[0];
    try std.testing.expectEqual(@as(c_int, c.TF_GLM_CODEC_LIMIT), lib.api.tf_glm_codec_decode(lib.context, c.TF_GLM_CODEC_PNG, f.encoded.ptr, f.encoded.len, 3, 2, bytes[2..].ptr, 23));
    for (bytes) |byte| try std.testing.expectEqual(@as(u8, 0xa5), byte);
}
