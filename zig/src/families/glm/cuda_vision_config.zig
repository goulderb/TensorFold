//! GLM5-Next vision dimensions from the pinned Mia/archived CUDA oracle, not the Metal tower.
const std = @import("std");

pub const Config = struct {
    hidden: u32,
    heads: u32,
    depth: u32,
    intermediate: u32,
    output: u32,
    projection_intermediate: u32,
    channels: u32,
    patch: u32,
    temporal: u32,
    merge: u32,
    eps: f32,
    limit: f32,
    theta: f32,
    image_token: u32,
    image_start: u32,
    image_end: u32,

    pub fn headDim(self: Config) u32 {
        return self.hidden / self.heads;
    }
    pub fn patchWidth(self: Config) u32 {
        return self.channels * self.temporal * self.patch * self.patch;
    }
    pub fn check(self: Config) !void {
        if (self.hidden == 0 or self.heads == 0 or self.hidden % self.heads != 0 or
            self.headDim() % 4 != 0 or self.depth == 0 or self.depth > 64 or
            self.intermediate == 0 or self.output == 0 or self.projection_intermediate == 0)
            return error.InvalidVisionConfig;
        if (self.channels != 3 or self.patch != 14 or self.temporal != 2 or self.merge != 2)
            return error.UnsupportedVisionGeometry;
        if (!std.math.isFinite(self.eps) or self.eps <= 0 or !std.math.isFinite(self.limit) or self.limit <= 0 or
            !std.math.isFinite(self.theta) or self.theta <= 0 or
            self.image_token == self.image_start or self.image_token == self.image_end or self.image_start == self.image_end)
            return error.InvalidVisionConfig;
        _ = std.math.mul(u32, self.hidden, 3) catch return error.InvalidVisionConfig;
        _ = std.math.mul(usize, self.output, self.projection_intermediate) catch return error.InvalidVisionConfig;
    }
    pub fn read(a: std.mem.Allocator, io: std.Io, dir: []const u8) !Config {
        const path = try std.fs.path.join(a, &.{ dir, "config.json" });
        defer a.free(path);
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 24));
        defer a.free(bytes);
        return parse(a, bytes);
    }
};

fn object(value: std.json.Value) !std.json.ObjectMap {
    if (value != .object) return error.InvalidVisionConfig;
    return value.object;
}
fn field(o: std.json.ObjectMap, name: []const u8) !std.json.Value {
    return o.get(name) orelse error.InvalidVisionConfig;
}
fn uint(o: std.json.ObjectMap, name: []const u8) !u32 {
    const v = try field(o, name);
    if (v != .integer) return error.InvalidVisionConfig;
    return std.math.cast(u32, v.integer) orelse error.InvalidVisionConfig;
}
fn number(v: std.json.Value) !f32 {
    const f: f64 = switch (v) {
        .float => |x| x,
        .integer => |x| @floatFromInt(x),
        else => return error.InvalidVisionConfig,
    };
    if (!std.math.isFinite(f) or @abs(f) > std.math.floatMax(f32)) return error.InvalidVisionConfig;
    return @floatCast(f);
}
fn eq(o: std.json.ObjectMap, name: []const u8, expected: []const u8) !void {
    const v = try field(o, name);
    if (v != .string or !std.mem.eql(u8, v.string, expected)) return error.UnsupportedVisionConfig;
}

pub fn parse(a: std.mem.Allocator, bytes: []const u8) !Config {
    const parsed = std.json.parseFromSlice(std.json.Value, a, bytes, .{}) catch |err| return switch (err) {
        error.OutOfMemory => error.OutOfMemory,
        else => error.InvalidVisionConfig,
    };
    defer parsed.deinit();
    const root = try object(parsed.value);
    try eq(root, "model_type", "glm5_next");
    const v = try object(try field(root, "vision_config"));
    try eq(v, "model_type", "glm5_next_vision");
    try eq(v, "hidden_act", "silu");
    const text = try object(try field(root, "text_config"));
    const bias = try field(v, "attention_bias");
    if (bias != .bool or !bias.bool) return error.UnsupportedVisionConfig;
    if (try number(try field(v, "attention_dropout")) != 0) return error.UnsupportedVisionConfig;
    var theta: f32 = 10000;
    if (v.get("rope_parameters")) |r| {
        if (r != .null) {
            const ro = try object(r);
            if (ro.get("rope_theta")) |t| theta = try number(t);
        }
    }
    const c: Config = .{
        .hidden = try uint(v, "hidden_size"),
        .heads = try uint(v, "num_heads"),
        .depth = try uint(v, "depth"),
        .intermediate = try uint(v, "intermediate_size"),
        .output = try uint(v, "out_hidden_size"),
        .projection_intermediate = try uint(v, "projection_intermediate_size"),
        .channels = try uint(v, "in_channels"),
        .patch = try uint(v, "patch_size"),
        .temporal = try uint(v, "temporal_patch_size"),
        .merge = try uint(v, "spatial_merge_size"),
        .eps = try number(try field(v, "rms_norm_eps")),
        .limit = if (v.get("swiglu_limit")) |x| try number(x) else 10,
        .theta = theta,
        .image_token = try uint(root, "image_token_id"),
        .image_start = try uint(root, "image_start_token_id"),
        .image_end = try uint(root, "image_end_token_id"),
    };
    if (c.output != try uint(text, "hidden_size")) return error.VisionTextWidthMismatch;
    try c.check();
    return c;
}

test "the production config describes the CUDA vision tower" {
    const c = try parse(std.testing.allocator, @embedFile("fixtures_cuda_config_exl3.json"));
    try std.testing.expectEqual(@as(u32, 24), c.depth);
    try std.testing.expectEqual(@as(u32, 64), c.headDim());
    try std.testing.expectEqual(@as(u32, 1176), c.patchWidth());
    try std.testing.expectEqual(@as(u32, 4096), c.output);
    try std.testing.expectEqual(@as(u32, 10240), c.projection_intermediate);
    try std.testing.expectEqual(@as(u32, 154854), c.image_token);
}

test "vision config rejects invalid geometry and text projection width" {
    const a = std.testing.allocator;
    const fixture = @embedFile("fixtures_cuda_config_exl3.json");
    const bad_width = try std.mem.replaceOwned(u8, a, fixture, "\"out_hidden_size\": 4096", "\"out_hidden_size\": 4095");
    defer a.free(bad_width);
    try std.testing.expectError(error.VisionTextWidthMismatch, parse(a, bad_width));
    const bad_patch = try std.mem.replaceOwned(u8, a, fixture, "\"patch_size\": 14", "\"patch_size\": 16");
    defer a.free(bad_patch);
    try std.testing.expectError(error.UnsupportedVisionGeometry, parse(a, bad_patch));
    var c = try parse(a, fixture);
    c.heads = 0;
    try std.testing.expectError(error.InvalidVisionConfig, c.check());
    c = try parse(a, fixture);
    c.theta = std.math.nan(f32);
    try std.testing.expectError(error.InvalidVisionConfig, c.check());
}
