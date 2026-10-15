//! Canonical host pixels, not an encoded-image codec or a Decoder implementation.
//! Source: da6282f9c9db1980cade733753330fe8d2354751, vision/glm.py load_images/_picture
//! -> vision/images.py _decode: ImageOps.exif_transpose, then RGB white.paste(RGBA, mask=A).
//! Input RGBA is straight (not premultiplied), interleaved, tightly packed, 8-bit sRGB.
//! No color-space conversion, resizing, animation admission or Python runtime is supplied here.
const std = @import("std");
const pre = @import("cuda_vision_preprocess.zig");
const Allocator = std.mem.Allocator;
pub const Rgb = pre.Rgb;
pub const OwnedRgb = pre.OwnedRgb;
pub const Limits = pre.Limits;
pub const Shape = pre.Shape;
pub const source_commit = "da6282f9c9db1980cade733753330fe8d2354751";
pub const glm_sha256 = "07da2a80bd37a95c5728026024ccdddbc36725d1d84066aee116860987b93e01";
pub const images_sha256 = "33521ea50f592c6aba9640dfda362a2049f12fa7bc717cc7b19fada0232e4bd3";

pub const Orientation = enum(u8) {
    normal = 1,
    flip_horizontal = 2,
    rotate_180 = 3,
    flip_vertical = 4,
    transpose = 5,
    rotate_90_cw = 6,
    transverse = 7,
    rotate_90_ccw = 8,

    /// Missing orientation is handled by the caller as normal. Invalid values are not missing.
    pub fn fromExif(value: u16) !Orientation {
        return std.enums.fromInt(Orientation, value) orelse error.InvalidExifOrientation;
    }
};

pub const Rgba = struct {
    shape: Shape,
    bytes: []const u8,
    pub fn check(self: Rgba, limits: Limits) !void {
        const pixels = try checkedPixels(self.shape, limits);
        const bytes = std.math.mul(usize, pixels, 4) catch return error.InvalidImageShape;
        if (self.bytes.len != bytes) return error.InvalidImageShape;
    }
};

fn checkedPixels(shape: Shape, limits: Limits) !usize {
    if (shape.height == 0 or shape.width == 0 or limits.max_dimension == 0 or limits.max_pixels == 0 or
        shape.height > limits.max_dimension or shape.width > limits.max_dimension) return error.InvalidImageShape;
    const pixels = std.math.mul(usize, shape.height, shape.width) catch return error.InvalidImageShape;
    if (pixels > limits.max_pixels) return error.InvalidImageShape;
    return pixels;
}

pub fn orientedShape(shape: Shape, orientation: Orientation, limits: Limits) !Shape {
    _ = try checkedPixels(shape, limits);
    return switch (orientation) {
        .normal, .flip_horizontal, .rotate_180, .flip_vertical => shape,
        .transpose, .rotate_90_cw, .transverse, .rotate_90_ccw => .{ .height = shape.width, .width = shape.height },
    };
}

// Source -> destination mapping; validated shapes bound every product and coordinate.
fn destination(x: usize, y: usize, width: usize, height: usize, orientation: Orientation) [2]usize {
    return switch (orientation) {
        .normal => .{ x, y },
        .flip_horizontal => .{ width - 1 - x, y },
        .rotate_180 => .{ width - 1 - x, height - 1 - y },
        .flip_vertical => .{ x, height - 1 - y },
        .transpose => .{ y, x },
        .rotate_90_cw => .{ height - 1 - y, x },
        .transverse => .{ height - 1 - y, width - 1 - x },
        .rotate_90_ccw => .{ y, width - 1 - x },
    };
}

/// Pillow's paste(RGBA, mask=A) onto white, rounded to nearest integer, not truncation.
/// Exhaustively verified offline for all 65,536 (channel, alpha) pairs with Pillow 12.3.0.
fn white(channel: u8, alpha: u8) u8 {
    const a: u32 = alpha;
    return @intCast((@as(u32, channel) * a + 255 * (255 - a) + 127) / 255);
}

/// A fresh owned RGB allocation. Input is borrowed and unchanged; even identity is copied.
/// One allocation after all validation, so allocation failure cannot leak intermediate buffers.
pub fn fromRgba(a: Allocator, rgba: Rgba, orientation: Orientation, limits: Limits) !OwnedRgb {
    try rgba.check(limits);
    const shape = try orientedShape(rgba.shape, orientation, limits);
    const pixels = try checkedPixels(shape, limits);
    const size = std.math.mul(usize, pixels, 3) catch return error.InvalidImageShape;
    const bytes = try a.alloc(u8, size);
    for (0..rgba.shape.height) |y| for (0..rgba.shape.width) |x| {
        const src = (y * rgba.shape.width + x) * 4;
        const xy = destination(x, y, rgba.shape.width, rgba.shape.height, orientation);
        const dst = (xy[1] * shape.width + xy[0]) * 3;
        for (0..3) |c| bytes[dst + c] = white(rgba.bytes[src + c], rgba.bytes[src + 3]);
    };
    return .{ .shape = shape, .bytes = bytes };
}

/// Opaque RGB follows the same orientation path, without an unnecessary RGBA intermediate.
pub fn orientRgb(a: Allocator, rgb: Rgb, orientation: Orientation, limits: Limits) !OwnedRgb {
    try rgb.check(limits);
    const shape = try orientedShape(rgb.shape, orientation, limits);
    const bytes = try a.alloc(u8, rgb.bytes.len);
    for (0..rgb.shape.height) |y| for (0..rgb.shape.width) |x| {
        const src = (y * rgb.shape.width + x) * 3;
        const xy = destination(x, y, rgb.shape.width, rgb.shape.height, orientation);
        const dst = (xy[1] * shape.width + xy[0]) * 3;
        @memcpy(bytes[dst..][0..3], rgb.bytes[src..][0..3]);
    };
    return .{ .shape = shape, .bytes = bytes };
}
