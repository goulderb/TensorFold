//! Native encoded pixels through a header-compiled C error-boundary shim.
//! The library owns codec handles; requests own allocations through their Zig allocator.
//! EXIF and alpha arithmetic stay in cuda_vision_image*.zig. No Python, subprocess,
//! URL fetch, gamma correction, resize, or alternate inference backend is used.
const std = @import("std");
const dynlib = @import("cuda").dynlib;
const pre = @import("cuda_vision_preprocess.zig");
const image = @import("cuda_vision_image.zig");
const exif = @import("cuda_vision_image_exif.zig");
pub const c = @import("image_codec_headers");
const Allocator = std.mem.Allocator;
pub const Limits = pre.Limits;
pub const max_encoded_bytes = 10 * 1024 * 1024;
pub const target_png_header_version: u32 = 10643;
pub const target_webp_version: u32 = 0x010302;
pub const target_jpeg_abi: u32 = 80;
pub const target_png_library_sha256 = "5aeb2bcd1675ffe64f50ba3bc7ae5cfd03997d57f6e16ae2a1d31feb28fed128";
pub const target_jpeg_library_sha256 = "d45014b67ae6df2f15c8fb3e86d37b78375deede47ef73b7c61e8b6b28c8223f";
pub const target_webp_library_sha256 = "d921459ab6508b3239218b0e4c2e44dbd2e702aeff4273190bf25d2036373930";

pub const Format = enum(u32) { png = c.TF_GLM_CODEC_PNG, jpeg = c.TF_GLM_CODEC_JPEG, webp = c.TF_GLM_CODEC_WEBP };
pub const Paths = struct {
    shim: []const u8,
    png: ?[]const u8 = null,
    jpeg: ?[]const u8 = null,
    webp: ?[]const u8 = null,
};
pub const Pins = struct {
    png: u32 = target_png_header_version,
    jpeg_abi: u32 = target_jpeg_abi,
    webp: u32 = target_webp_version,
};
pub const Api = struct {
    tf_glm_codec_abi: @TypeOf(&c.tf_glm_codec_abi),
    tf_glm_codec_open: @TypeOf(&c.tf_glm_codec_open),
    tf_glm_codec_close: @TypeOf(&c.tf_glm_codec_close),
    tf_glm_codec_features: @TypeOf(&c.tf_glm_codec_features),
    tf_glm_codec_probe: @TypeOf(&c.tf_glm_codec_probe),
    tf_glm_codec_decode: @TypeOf(&c.tf_glm_codec_decode),
};

fn checkStatus(status: c_int) !void {
    return switch (status) {
        c.TF_GLM_CODEC_OK => {},
        c.TF_GLM_CODEC_LIBRARY => error.NativeImageLibraryUnavailable,
        c.TF_GLM_CODEC_SYMBOL => error.NativeImageSymbolMissing,
        c.TF_GLM_CODEC_VERSION => error.NativeImageVersionMismatch,
        c.TF_GLM_CODEC_UNAVAILABLE => error.NativeImageDecoderUnavailable,
        c.TF_GLM_CODEC_INVALID => error.InvalidEncodedImage,
        c.TF_GLM_CODEC_LIMIT => error.InvalidImageShape,
        c.TF_GLM_CODEC_COLOR => error.UnsupportedNativeImageColorMode,
        c.TF_GLM_CODEC_MEMORY => error.OutOfMemory,
        c.TF_GLM_CODEC_ANIMATED => error.AnimatedImagesUnsupported,
        else => error.InvalidNativeCodecResult,
    };
}
fn pathZ(a: Allocator, path: ?[]const u8) !?[:0]u8 {
    const p = path orelse return null;
    if (p.len == 0 or !std.fs.path.isAbsolute(p) or std.mem.indexOfScalar(u8, p, 0) != null)
        return error.InvalidNativeCodecPath;
    return try a.dupeSentinel(u8, p, 0);
}
fn checkedSize(shape: pre.Shape, limits: Limits) !usize {
    if (limits.max_dimension == 0 or limits.max_pixels == 0 or
        limits.max_dimension > 8192 or limits.max_pixels > 16 * 1024 * 1024)
        return error.InvalidImageBudget;
    if (shape.width == 0 or shape.height == 0 or shape.width > limits.max_dimension or shape.height > limits.max_dimension)
        return error.InvalidImageShape;
    const pixels = std.math.mul(usize, shape.width, shape.height) catch return error.InvalidImageShape;
    if (pixels > limits.max_pixels) return error.InvalidImageShape;
    return std.math.mul(usize, pixels, 4) catch error.InvalidImageShape;
}
pub fn formatOf(encoded: []const u8) !Format {
    if (encoded.len == 0 or encoded.len > max_encoded_bytes) return error.ImageEncodedByteLimit;
    if (std.mem.startsWith(u8, encoded, "\x89PNG\r\n\x1a\n")) return .png;
    if (std.mem.startsWith(u8, encoded, "\xff\xd8")) return .jpeg;
    if (encoded.len >= 12 and std.mem.eql(u8, encoded[0..4], "RIFF") and std.mem.eql(u8, encoded[8..12], "WEBP")) return .webp;
    return error.UnsupportedImageFormat;
}

// Framing/CRC validation belongs to the EXIF helper. This pass adds only
// single-frame admission, before a native codec can ignore animation metadata.
fn singleFrame(encoded: []const u8, format: Format) !void {
    switch (format) {
        .png => {
            var at: usize = 8;
            while (at < encoded.len) {
                const n: usize = std.mem.readInt(u32, encoded[at..][0..4], .big);
                const kind = encoded[at + 4 ..][0..4];
                if (std.mem.eql(u8, kind, "acTL") or std.mem.eql(u8, kind, "fcTL") or std.mem.eql(u8, kind, "fdAT"))
                    return error.AnimatedImagesUnsupported;
                at += 12 + n;
            }
        },
        .webp => {
            var at: usize = 12;
            while (at < encoded.len) {
                const kind = encoded[at..][0..4];
                const n: usize = std.mem.readInt(u32, encoded[at + 4 ..][0..4], .little);
                if (std.mem.eql(u8, kind, "ANIM") or std.mem.eql(u8, kind, "ANMF") or
                    (std.mem.eql(u8, kind, "VP8X") and n >= 1 and encoded[at + 8] & 2 != 0))
                    return error.AnimatedImagesUnsupported;
                at += 8 + n + n % 2;
            }
        },
        .jpeg => {
            var at: usize = 2;
            while (at < encoded.len) {
                while (at < encoded.len and encoded[at] == 0xff) : (at += 1) {}
                if (at == encoded.len) return error.InvalidEncodedImage;
                const marker = encoded[at];
                at += 1;
                if (marker == 0xda or marker == 0xd9) return;
                if (marker == 0x01 or (marker >= 0xd0 and marker <= 0xd7)) continue;
                const n: usize = std.mem.readInt(u16, encoded[at..][0..2], .big);
                if (marker == 0xe2 and n >= 6 and std.mem.eql(u8, encoded[at + 2 ..][0..4], "MPF\x00"))
                    return error.AnimatedImagesUnsupported;
                at += n;
            }
        },
    }
}
const Admission = struct { format: Format, orientation: image.Orientation };
fn admit(encoded: []const u8, limits: Limits) !Admission {
    // Validate the budget even for malformed bytes and before touching a codec.
    _ = try checkedSize(.{ .width = 1, .height = 1 }, limits);
    const format = try formatOf(encoded);
    const orientation = (try exif.encodedOrientation(encoded, .{})) orelse .normal;
    try singleFrame(encoded, format);
    if (format == .png) {
        const shape: pre.Shape = .{
            .width = std.mem.readInt(u32, encoded[16..20], .big),
            .height = std.mem.readInt(u32, encoded[20..24], .big),
        };
        _ = try checkedSize(shape, limits);
    }
    return .{ .format = format, .orientation = orientation };
}

pub const OwnedRgba = struct {
    shape: pre.Shape,
    bytes: []u8,
    pub fn borrowed(self: OwnedRgba) image.Rgba {
        return .{ .shape = self.shape, .bytes = self.bytes };
    }
    pub fn deinit(self: *OwnedRgba, a: Allocator) void {
        a.free(self.bytes);
        self.* = undefined;
    }
};

/// Borrowed API/context; the Library and its stable address outlive each call.
/// Public for CPU fault injection, which is not evidence of real pixel decoding.
pub const Backend = struct {
    api: *const Api,
    context: *anyopaque,
    features: u32,
    fn require(self: Backend, format: Format) !void {
        if (self.features & @backingInt(format) != 0) return;
        return switch (format) {
            .png => error.NativePngDecoderUnavailable,
            .jpeg => error.NativeJpegDecoderUnavailable,
            .webp => error.NativeWebpDecoderUnavailable,
        };
    }
    fn rgbaAdmitted(self: Backend, a: Allocator, encoded: []const u8, limits: Limits, admission: Admission) !OwnedRgba {
        try self.require(admission.format);
        var width: u32 = 0;
        var height: u32 = 0;
        try checkStatus(self.api.tf_glm_codec_probe(self.context, @backingInt(admission.format), encoded.ptr, encoded.len, limits.max_dimension, limits.max_pixels, &width, &height));
        const shape: pre.Shape = .{ .width = width, .height = height };
        const size = try checkedSize(shape, limits);
        const pixels = try a.alloc(u8, size);
        errdefer a.free(pixels);
        try checkStatus(self.api.tf_glm_codec_decode(self.context, @backingInt(admission.format), encoded.ptr, encoded.len, width, height, pixels.ptr, pixels.len));
        return .{ .shape = shape, .bytes = pixels };
    }
    /// Owned, straight RGBA before EXIF orientation and white composition.
    pub fn rgba(self: Backend, a: Allocator, encoded: []const u8, limits: Limits) !OwnedRgba {
        return self.rgbaAdmitted(a, encoded, limits, try admit(encoded, limits));
    }
    pub fn decode(self: Backend, a: Allocator, encoded: []const u8, limits: Limits) !pre.OwnedRgb {
        const admission = try admit(encoded, limits);
        var pixels = try self.rgbaAdmitted(a, encoded, limits, admission);
        defer pixels.deinit(a);
        return image.fromRgba(a, pixels.borrowed(), admission.orientation, limits);
    }
};

pub const Library = struct {
    lib: dynlib.Library,
    api: Api,
    context: *anyopaque,
    /// Deployment validates exact library file hashes before loading. Version
    /// checks here are ABI checks, not a substitute for that provenance gate.
    pub fn open(a: Allocator, paths: Paths, pins: Pins) !Library {
        const shim = (try pathZ(a, paths.shim)).?;
        defer a.free(shim);
        const png = try pathZ(a, paths.png);
        defer if (png) |p| a.free(p);
        const jpeg = try pathZ(a, paths.jpeg);
        defer if (jpeg) |p| a.free(p);
        const webp = try pathZ(a, paths.webp);
        defer if (webp) |p| a.free(p);
        var lib = dynlib.Library.openFile(shim) catch return error.NativeImageLibraryUnavailable;
        errdefer lib.close();
        var missing: []const u8 = "";
        const api = lib.resolve(Api, &missing) catch return error.NativeImageSymbolMissing;
        if (api.tf_glm_codec_abi() != c.TF_GLM_CODEC_ABI) return error.NativeImageVersionMismatch;
        var status: c_int = c.TF_GLM_CODEC_OK;
        const context = api.tf_glm_codec_open(if (png) |p| p.ptr else null, if (jpeg) |p| p.ptr else null, if (webp) |p| p.ptr else null, pins.png, pins.jpeg_abi, pins.webp, &status);
        if (status != c.TF_GLM_CODEC_OK) {
            if (context) |p| api.tf_glm_codec_close(p);
            try checkStatus(status);
        }
        return .{ .lib = lib, .api = api, .context = context orelse return error.InvalidNativeCodecResult };
    }
    pub fn backend(self: *Library) Backend {
        return .{ .api = &self.api, .context = self.context, .features = self.api.tf_glm_codec_features(self.context) };
    }
    /// No request or Decoder may use this Library after close; no async work is started.
    pub fn close(self: *Library) void {
        self.api.tf_glm_codec_close(self.context);
        self.lib.close();
        self.* = undefined;
    }
    pub fn decoder(self: *Library) pre.Decoder {
        return .{ .context = self, .run = run };
    }
    fn run(context: ?*anyopaque, a: Allocator, encoded: []const u8, limits: Limits) anyerror!pre.OwnedRgb {
        const self: *Library = @ptrCast(@alignCast(context orelse return error.NativeImageDecoderUnavailable));
        return self.backend().decode(a, encoded, limits);
    }
};
