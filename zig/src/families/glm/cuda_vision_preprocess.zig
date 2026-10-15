//! Native CPU geometry and patch packing for Mia b5808ac6 vision/glm.py.
//! RGB inputs are already EXIF-oriented, with transparency composited onto white.
//! A resizing helper must implement torchvision's uint8 bicubic+antialias path; none is silently substituted.
const std = @import("std");
const Config = @import("cuda_vision_config.zig").Config;
const Allocator = std.mem.Allocator;

pub const Shape = struct { height: u32, width: u32 };
pub const Limits = struct { max_dimension: u32 = 8192, max_pixels: usize = 16 * 1024 * 1024 };
pub const Budget = struct {
    max_tokens: u32 = 2048,
    min_tokens: u32 = 16,
    pub fn check(self: Budget) !void {
        if (self.min_tokens == 0 or self.max_tokens < self.min_tokens or self.max_tokens > 8000)
            return error.InvalidVisionBudget;
    }
};
pub const Rgb = struct {
    shape: Shape,
    bytes: []const u8,
    pub fn check(self: Rgb, limits: Limits) !void {
        if (self.shape.height == 0 or self.shape.width == 0 or limits.max_dimension == 0 or limits.max_pixels == 0 or
            self.shape.height > limits.max_dimension or self.shape.width > limits.max_dimension)
            return error.InvalidImageShape;
        const pixels = std.math.mul(usize, self.shape.height, self.shape.width) catch return error.InvalidImageShape;
        if (pixels > limits.max_pixels or pixels > std.math.maxInt(usize) / 3 or self.bytes.len != pixels * 3)
            return error.InvalidImageShape;
    }
};
pub const OwnedRgb = struct {
    shape: Shape,
    bytes: []u8,
    pub fn borrowed(self: OwnedRgb) Rgb {
        return .{ .shape = self.shape, .bytes = self.bytes };
    }
    pub fn deinit(self: *OwnedRgb, a: Allocator) void {
        a.free(self.bytes);
        self.* = undefined;
    }
};
pub const Resizer = struct {
    context: ?*anyopaque = null,
    /// Returns owned, interleaved RGB bytes at exactly the requested size.
    run: *const fn (?*anyopaque, Allocator, Rgb, Shape) anyerror!OwnedRgb,
};
pub const Decoder = struct {
    context: ?*anyopaque = null,
    /// Single-frame JPEG/PNG/WebP only; EXIF orientation and white-alpha composition must precede RGB output.
    run: *const fn (?*anyopaque, Allocator, []const u8, Limits) anyerror!OwnedRgb,
};

fn aligned28(value: f64) u32 {
    return @intFromFloat(@ceil(value / 28) * 28);
}
pub fn canvas(shape: Shape, budget: Budget) !Shape {
    try budget.check();
    if (shape.height == 0 or shape.width == 0 or shape.height > 8192 or shape.width > 8192) return error.InvalidImageShape;
    const low = @as(u64, budget.min_tokens) * 2 * 28 * 28;
    const high = @as(u64, budget.max_tokens) * 2 * 28 * 28;
    const h: f64 = @floatFromInt(shape.height);
    const w: f64 = @floatFromInt(shape.width);
    var out: Shape = .{ .height = aligned28(h), .width = aligned28(w) };
    if (@as(u64, out.height) * out.width * 2 < low) {
        const scale = @sqrt(@as(f64, @floatFromInt(low)) / (2 * h * w));
        out = .{ .height = aligned28(@max(1, @ceil(h * scale))), .width = aligned28(@max(1, @ceil(w * scale))) };
    }
    if (@as(u64, out.height) * out.width * 2 > high) {
        var lo: u32 = 1;
        var hi: u32 = shape.height;
        out = .{ .height = 28, .width = 28 };
        while (lo <= hi) {
            const mid = lo + (hi - lo) / 2;
            const ch = aligned28(@floatFromInt(mid));
            const cw = aligned28(@max(1, @floor(w * @as(f64, @floatFromInt(mid)) / h)));
            if (@as(u64, ch) * cw * 2 <= high) {
                out = .{ .height = ch, .width = cw };
                lo = mid + 1;
            } else hi = mid - 1;
        }
    }
    return out;
}
pub fn content(shape: Shape, fitted: Shape, budget: Budget) !Shape {
    try budget.check();
    if (shape.height == 0 or shape.width == 0 or shape.height > 8192 or shape.width > 8192 or
        fitted.height == 0 or fitted.width == 0 or fitted.height % 28 != 0 or fitted.width % 28 != 0 or
        @as(u64, fitted.height) * fitted.width > @as(u64, budget.max_tokens) * 28 * 28) return error.InvalidImageShape;
    const h: f64 = @floatFromInt(shape.height);
    const w: f64 = @floatFromInt(shape.width);
    var scale = @min(@as(f64, @floatFromInt(fitted.height)) / h, @as(f64, @floatFromInt(fitted.width)) / w);
    if (@as(u64, shape.height) * shape.width * 2 >= @as(u64, budget.min_tokens) * 2 * 28 * 28) scale = @min(1, scale);
    return .{
        .height = @intFromFloat(@max(1, @min(@as(f64, @floatFromInt(fitted.height)), @floor(h * scale)))),
        .width = @intFromFloat(@max(1, @min(@as(f64, @floatFromInt(fitted.width)), @floor(w * scale)))),
    };
}

pub const Prepared = struct {
    canvas: Shape,
    content: Shape,
    /// [time-pairs=1, height/14, width/14], with rows in 2x2 merge-block order.
    grid: [3]u32,
    patch_width: u32,
    pixels: []f32,
    pub fn check(self: Prepared, c: Config) !void {
        try c.check();
        if (self.grid[0] != 1 or self.grid[1] == 0 or self.grid[2] == 0 or
            self.grid[1] % c.merge != 0 or self.grid[2] % c.merge != 0 or self.patch_width != c.patchWidth() or
            self.canvas.height % c.patch != 0 or self.canvas.width % c.patch != 0 or
            self.canvas.height / c.patch != self.grid[1] or self.canvas.width / c.patch != self.grid[2] or
            self.content.height == 0 or self.content.width == 0 or
            self.content.height > self.canvas.height or self.content.width > self.canvas.width) return error.InvalidVisionGrid;
        const rows = std.math.mul(usize, self.grid[1], self.grid[2]) catch return error.InvalidVisionGrid;
        if (rows > 4 * 8000) return error.InvalidVisionGrid;
        const n = std.math.mul(usize, rows, self.patch_width) catch return error.InvalidVisionGrid;
        if (self.pixels.len != n) return error.InvalidVisionGrid;
    }
    /// Valid after prepareRgb or check; one image contributes grid_h * grid_w / 4 feature rows.
    pub fn visualTokens(self: Prepared) usize {
        return @as(usize, self.grid[1]) * self.grid[2] / 4;
    }
    pub fn deinit(self: *Prepared, a: Allocator) void {
        a.free(self.pixels);
        self.* = undefined;
    }
};
pub const mean = [3]f32{ 0.48145466, 0.4578275, 0.40821073 };
pub const deviation = [3]f32{ 0.26862954, 0.26130258, 0.27577711 };
pub fn normalized(value: u8, channel: usize) f32 {
    const m: f32 = mean[channel] * 255;
    const s: f32 = deviation[channel] * 255;
    const x: f32 = @floatFromInt(value);
    return (x - m) / s;
}

pub fn prepareRgb(a: Allocator, c: Config, rgb: Rgb, budget: Budget, limits: Limits, resizer: ?Resizer) !Prepared {
    try c.check();
    try rgb.check(limits);
    const fitted = try canvas(rgb.shape, budget);
    const inside = try content(rgb.shape, fitted, budget);
    var resized: ?OwnedRgb = null;
    defer if (resized) |*r| r.deinit(a);
    if (inside.height != rgb.shape.height or inside.width != rgb.shape.width) {
        const helper = resizer orelse return error.NativeBicubicResizeUnavailable;
        resized = try helper.run(helper.context, a, rgb, inside);
        const r = resized.?.borrowed();
        if (r.shape.height != inside.height or r.shape.width != inside.width) return error.InvalidResizeOutput;
        try r.check(.{ .max_dimension = @max(inside.height, inside.width), .max_pixels = std.math.maxInt(usize) });
    }
    const source = if (resized) |r| r.borrowed() else rgb;
    const gh = fitted.height / c.patch;
    const gw = fitted.width / c.patch;
    const rows = try std.math.mul(usize, gh, gw);
    const n = try std.math.mul(usize, rows, c.patchWidth());
    const pixels = try a.alloc(f32, n);
    errdefer a.free(pixels);
    var at: usize = 0;
    for (0..gh / c.merge) |bh| for (0..gw / c.merge) |bw| for (0..c.merge) |mh| for (0..c.merge) |mw| {
        for (0..c.channels) |channel| for (0..c.temporal) |_| for (0..c.patch) |y| for (0..c.patch) |x| {
            const iy = (bh * c.merge + mh) * c.patch + y;
            const ix = (bw * c.merge + mw) * c.patch + x;
            const value: u8 = if (iy < inside.height and ix < inside.width) source.bytes[(iy * inside.width + ix) * 3 + channel] else 0;
            pixels[at] = normalized(value, channel);
            at += 1;
        };
    };
    std.debug.assert(at == pixels.len);
    return .{ .canvas = fitted, .content = inside, .grid = .{ 1, gh, gw }, .patch_width = c.patchWidth(), .pixels = pixels };
}

/// Tokenization stays with the native caller. Every image-token position must have exactly one feature row.
pub fn imagePositions(a: Allocator, c: Config, ids: []const u32, images: []const Prepared, prompt_limit: usize) ![]u32 {
    try c.check();
    if (prompt_limit == 0 or ids.len >= prompt_limit or ids.len > std.math.maxInt(u32)) return error.ImagePromptTooLong;
    var rows: usize = 0;
    for (images) |image| {
        try image.check(c);
        rows = std.math.add(usize, rows, image.visualTokens()) catch return error.InvalidImageShape;
    }
    var positions: usize = 0;
    for (ids) |id| positions += @intFromBool(id == c.image_token);
    if (positions != rows) return error.ImageTokenCountMismatch;
    const out = try a.alloc(u32, rows);
    var at: usize = 0;
    for (ids, 0..) |id, i| if (id == c.image_token) {
        out[at] = @intCast(i);
        at += 1;
    };
    return out;
}

pub fn inlineBytes(a: Allocator, url: []const u8, max_bytes: usize) ![]u8 {
    if (max_bytes == 0 or max_bytes > 10 * 1024 * 1024) return error.InvalidImageBudget;
    if (!std.mem.startsWith(u8, url, "data:")) return error.ImageUrlsDisabled;
    const comma = std.mem.indexOfScalar(u8, url, ',') orelse return error.InvalidImageDataUrl;
    if (comma > 256) return error.InvalidImageDataUrl;
    const payload = url[comma + 1 ..];
    for (payload) |ch| if (ch > 127) return error.InvalidImageDataUrl;
    var media = std.mem.splitScalar(u8, url[5..comma], ';');
    _ = media.next();
    const encoding = media.next();
    if (media.next() != null) return error.InvalidImageDataUrl;
    if (encoding) |e| {
        if (!std.ascii.eqlIgnoreCase(e, "base64")) return error.InvalidImageDataUrl;
        if (payload.len > 4 * ((max_bytes + 2) / 3)) return error.ImageEncodedByteLimit;
        const decoder = std.base64.standard.Decoder;
        const size = decoder.calcSizeForSlice(payload) catch return error.InvalidImageDataUrl;
        if (size == 0 or size > max_bytes) return error.ImageEncodedByteLimit;
        const out = try a.alloc(u8, size);
        errdefer a.free(out);
        decoder.decode(out, payload) catch return error.InvalidImageDataUrl;
        return out;
    }
    if (payload.len > max_bytes * 3) return error.ImageEncodedByteLimit;
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(a);
    var i: usize = 0;
    while (i < payload.len) {
        var ch = payload[i];
        if (ch == '%' and i + 2 < payload.len) {
            ch = std.fmt.parseInt(u8, payload[i + 1 .. i + 3], 16) catch return error.InvalidImageDataUrl;
            i += 2;
        } else if (ch == '%') return error.InvalidImageDataUrl;
        if (out.items.len == max_bytes) return error.ImageEncodedByteLimit;
        try out.append(a, ch);
        i += 1;
    }
    if (out.items.len == 0) return error.ImageEncodedByteLimit;
    return out.toOwnedSlice(a);
}

pub fn decodeInline(a: Allocator, url: []const u8, max_bytes: usize, limits: Limits, decoder: ?Decoder) !OwnedRgb {
    const bytes = try inlineBytes(a, url, max_bytes);
    defer a.free(bytes);
    const helper = decoder orelse return error.NativeImageDecoderUnavailable;
    var rgb = try helper.run(helper.context, a, bytes, limits);
    errdefer rgb.deinit(a);
    try rgb.borrowed().check(limits);
    return rgb;
}
