//! Rank-zero CPU admission for still images. The exact native decoder and
//! geometry are shared with the tower; no CUDA work runs on an HTTP thread.
const std = @import("std");
const codec = @import("cuda_vision_codec.zig");
const pre = @import("cuda_vision_preprocess.zig");
const Config = @import("cuda_vision_config.zig").Config;

pub const Probe = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    library: codec.Library,
    config: Config,
    mutex: std.Io.Mutex = .init,

    pub fn open(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, shim: []const u8) !*Probe {
        const config = try Config.read(gpa, io, dir);
        const png = std.c.getenv("TF_GLM_VISION_PNG") orelse return error.MissingVisionPngPath;
        const jpeg = std.c.getenv("TF_GLM_VISION_JPEG");
        const webp = std.c.getenv("TF_GLM_VISION_WEBP");
        const self = try gpa.create(Probe);
        errdefer gpa.destroy(self);
        self.* = .{ .gpa = gpa, .io = io, .library = try codec.Library.open(gpa, .{ .shim = shim, .png = std.mem.span(png), .jpeg = if (jpeg) |p| std.mem.span(p) else null, .webp = if (webp) |p| std.mem.span(p) else null }, .{}), .config = config };
        return self;
    }

    pub fn close(self: *Probe) void {
        self.library.close();
        const gpa = self.gpa;
        gpa.destroy(self);
    }

    pub fn rows(ctx: *anyopaque, encoded: []const u8, images: usize) anyerror!u32 {
        const self: *Probe = @ptrCast(@alignCast(ctx));
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        var rgb = try self.library.backend().decode(self.gpa, encoded, .{});
        defer rgb.deinit(self.gpa);
        return rowsForShape(self.config, rgb.shape, images);
    }
};

pub fn rowsForShape(c: Config, shape: pre.Shape, images: usize) !u32 {
    if (images == 0 or images > 50) return error.InvalidVisionImageCount;
    try c.check();
    const budget: pre.Budget = .{ .min_tokens = 16, .max_tokens = @max(16, @min(2048, @as(u32, @intCast(16384 / images)))) };
    const fitted = try pre.canvas(shape, budget);
    if (fitted.height % c.patch != 0 or fitted.width % c.patch != 0) return error.InvalidVisionGrid;
    const patches = try std.math.mul(u32, fitted.height / c.patch, fitted.width / c.patch);
    if (patches % (c.merge * c.merge) != 0) return error.InvalidVisionGrid;
    return patches / (c.merge * c.merge);
}

test "probe geometry matches the native prepared patch grid" {
    const c = try @import("cuda_vision_config.zig").parse(std.testing.allocator, @embedFile("fixtures_cuda_config_exl3.json"));
    const shape: pre.Shape = .{ .height = 333, .width = 451 };
    const rows = try rowsForShape(c, shape, 2);
    const canvas = try pre.canvas(shape, .{ .min_tokens = 16, .max_tokens = 2048 });
    try std.testing.expectEqual(canvas.height * canvas.width / (c.patch * c.patch * c.merge * c.merge), rows);
}
