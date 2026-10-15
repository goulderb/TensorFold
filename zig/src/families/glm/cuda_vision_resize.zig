//! CPU uint8 BICUBIC + antialias, align_corners=false, explicit H/W, RGB only.
//! Coefficient and two-pass integer arithmetic follows PyTorch 0fabc3ba
//! aten/src/ATen/native/cpu/UpSampleKernel.cpp; no codec, padding or runtime Torch.
const std = @import("std");
const pre = @import("cuda_vision_preprocess.zig");
const Allocator = std.mem.Allocator;

pub const resizer: pre.Resizer = .{ .run = run };
fn run(_: ?*anyopaque, a: Allocator, rgb: pre.Rgb, target: pre.Shape) !pre.OwnedRgb {
    return resize(a, rgb, target);
}

const Range = struct { start: usize, count: usize };
const Axis = struct {
    ranges: []Range,
    weights: []i16,
    stride: usize,
    precision: u5,

    fn init(a: Allocator, input: u32, output: u32) !Axis {
        @setFloatMode(.strict);
        const scale = @as(f64, @floatFromInt(input)) / @as(f64, @floatFromInt(output));
        const support = if (scale >= 1) 2 * scale else 2;
        const stride: usize = @as(usize, @intFromFloat(@ceil(support))) * 2 + 1;
        const n = try std.math.mul(usize, output, stride);
        const ranges = try a.alloc(Range, output);
        errdefer a.free(ranges);
        const floating = try a.alloc(f64, n);
        defer a.free(floating);
        @memset(floating, 0);
        const invscale = if (scale >= 1) 1 / scale else 1;
        var maximum: f64 = 0;
        for (ranges, 0..) |*range, i| {
            const center = scale * (@as(f64, @floatFromInt(i)) + 0.5);
            const start = @max(@as(i64, @intFromFloat(center - support + 0.5)), 0);
            const end = @min(@as(i64, @intFromFloat(center + support + 0.5)), input);
            const count: usize = @intCast(@min(@max(end - start, 0), stride));
            if (count == 0 or start >= input) return error.UnsupportedResizeShape;
            range.* = .{ .start = @intCast(start), .count = count };
            const values = floating[i * stride ..][0..count];
            var total: f64 = 0;
            for (values, 0..) |*v, j| {
                v.* = cubic((@as(f64, @floatFromInt(j + range.start)) - center + 0.5) * invscale);
                total += v.*;
            }
            if (!std.math.isFinite(total) or total == 0) return error.UnsupportedResizeShape;
            for (values) |*v| {
                v.* /= total;
                maximum = @max(maximum, v.*);
            }
        }
        // The same precision applies to every output coordinate of this axis.
        var precision: u5 = 0;
        while (precision < 22) : (precision += 1) {
            const next: i64 = @intFromFloat(0.5 + maximum * @as(f64, @floatFromInt(@as(u32, 1) << (precision + 1))));
            if (next >= 1 << 15) break;
        }
        if (precision == 0) return error.UnsupportedResizeShape;
        const weights = try a.alloc(i16, n);
        errdefer a.free(weights);
        const factor: f64 = @floatFromInt(@as(u32, 1) << precision);
        for (floating, weights) |v, *w| {
            const scaled = v * factor;
            const integer: i64 = @intFromFloat(if (scaled < 0) -0.5 + scaled else 0.5 + scaled);
            w.* = std.math.cast(i16, integer) orelse return error.UnsupportedResizeShape;
        }
        // Upstream accumulates in int32, including on AVX. Refuse shapes that could overflow it.
        for (ranges, 0..) |range, i| {
            var bound: i64 = @as(i64, 1) << (precision - 1);
            for (weights[i * stride ..][0..range.count]) |w| bound += @as(i64, @intCast(@abs(@as(i32, w)))) * 255;
            if (bound > std.math.maxInt(i32)) return error.UnsupportedResizeShape;
        }
        return .{ .ranges = ranges, .weights = weights, .stride = stride, .precision = precision };
    }
    fn deinit(self: *Axis, a: Allocator) void {
        a.free(self.ranges);
        a.free(self.weights);
        self.* = undefined;
    }
};

fn cubic(value: f64) f64 {
    @setFloatMode(.strict);
    const x = @abs(value);
    const k: f64 = -0.5;
    if (x < 1) return ((k + 2) * x - (k + 3)) * x * x + 1;
    if (x < 2) return ((k * x - 5 * k) * x + 8 * k) * x - 4 * k;
    return 0;
}
fn rounded(sum: i64, precision: u5) u8 {
    return @intCast(std.math.clamp(sum >> precision, 0, 255));
}
fn horizontal(dst: []u8, rgb: pre.Rgb, target_width: u32, axis: Axis) void {
    for (0..rgb.shape.height) |y| for (axis.ranges, 0..) |range, x| {
        const weights = axis.weights[x * axis.stride ..][0..range.count];
        for (0..3) |channel| {
            var sum: i64 = @as(i64, 1) << (axis.precision - 1);
            for (weights, range.start..) |w, ix| sum += @as(i64, rgb.bytes[(y * rgb.shape.width + ix) * 3 + channel]) * w;
            dst[(y * target_width + x) * 3 + channel] = rounded(sum, axis.precision);
        }
    };
}
fn vertical(dst: []u8, src: []const u8, width: u32, axis: Axis) void {
    const row_bytes = @as(usize, width) * 3;
    for (axis.ranges, 0..) |range, y| {
        const weights = axis.weights[y * axis.stride ..][0..range.count];
        for (0..row_bytes) |x| {
            var sum: i64 = @as(i64, 1) << (axis.precision - 1);
            for (weights, range.start..) |w, iy| sum += @as(i64, src[iy * row_bytes + x]) * w;
            dst[y * row_bytes + x] = rounded(sum, axis.precision);
        }
    }
}

/// Owns the returned RGB; input is borrowed. Supports canonical source limits and
/// <=8000 visual tokens' content pixels. Excessive horizontal intermediate size is refused.
pub fn resize(a: Allocator, rgb: pre.Rgb, target: pre.Shape) !pre.OwnedRgb {
    try rgb.check(.{});
    if (target.height == 0 or target.width == 0 or target.height > 8000 * 28 or target.width > 8000 * 28)
        return error.UnsupportedResizeShape;
    const output_pixels = std.math.mul(usize, target.height, target.width) catch return error.UnsupportedResizeShape;
    if (output_pixels > 8000 * 28 * 28) return error.UnsupportedResizeShape;
    const change_x = rgb.shape.width != target.width;
    const change_y = rgb.shape.height != target.height;
    const intermediate_pixels = std.math.mul(usize, rgb.shape.height, target.width) catch return error.UnsupportedResizeShape;
    if (change_x and change_y and intermediate_pixels > 16 * 1024 * 1024) return error.UnsupportedResizeShape;
    if (!change_x and !change_y) return .{ .shape = target, .bytes = try a.dupe(u8, rgb.bytes) };
    var x: ?Axis = null;
    defer if (x) |*axis| axis.deinit(a);
    var y: ?Axis = null;
    defer if (y) |*axis| axis.deinit(a);
    if (change_x) x = try Axis.init(a, rgb.shape.width, target.width);
    if (change_y) y = try Axis.init(a, rgb.shape.height, target.height);
    const out = try a.alloc(u8, output_pixels * 3);
    errdefer a.free(out);
    var intermediate: ?[]u8 = null;
    defer if (intermediate) |bytes| a.free(bytes);
    if (change_x and change_y) intermediate = try a.alloc(u8, intermediate_pixels * 3);
    if (x) |axis| horizontal(intermediate orelse out, rgb, target.width, axis);
    if (y) |axis| vertical(out, intermediate orelse rgb.bytes, target.width, axis);
    return .{ .shape = target, .bytes = out };
}

test {
    _ = @import("cuda_vision_resize_test.zig");
}
