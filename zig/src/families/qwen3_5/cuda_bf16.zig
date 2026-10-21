const cuda = @import("cuda");
const hf = @import("cuda_hf.zig");
const layers = @import("cuda_layers.zig");

pub const Bindings = struct {
    image: cuda.Module,
    glue: cuda.Module,
    lane: cuda.qmmf.Lane,
    pack: cuda.modelopt.Packer,
    part: u64 = 0,

    pub fn load(ctx: *const cuda.Context, fns: *layers.Fns) !Bindings {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        const major: u32 = @intCast(try ctx.attribute(.compute_capability_major));
        if (major < 8) return error.Bf16NeedsAmpere;
        var image = try cuda.Module.load(ctx.d, cuda.kernels.qwen35_hf);
        errdefer image.unload();
        var lane = try cuda.qmmf.Lane.load(ctx.d, major);
        errdefer lane.unload();
        var pack = try cuda.modelopt.Packer.load(ctx.d);
        errdefer pack.unload();
        var glue = try cuda.Module.load(ctx.d, cuda.kernels.qwen35);
        errdefer glue.unload();
        var bound: layers.Fns = undefined;
        inline for (@typeInfo(layers.Fns).@"struct".field_names) |name|
            @field(bound, name) = try glue.function("tf_q35_" ++ name);
        try hf.bind(image, &bound);
        fns.* = bound;
        return .{ .image = image, .glue = glue, .lane = lane, .pack = pack };
    }

    pub fn unload(b: *Bindings) void {
        b.pack.unload();
        b.lane.unload();
        b.image.unload();
        b.glue.unload();
    }

    pub fn linear(b: *const Bindings) cuda.qlinear.Linear {
        return .{ .lane = &b.lane };
    }
};

test "plain BF16 binds only the shared lane projection format" {
    const std = @import("std");
    var b: Bindings = undefined;
    const linear = b.linear();
    try std.testing.expect(linear.lane.? == &b.lane);
    try std.testing.expect(linear.nvfp4_prompt == null and linear.ck == null and linear.affine4 == null);
    try std.testing.expect(linear.exl3_trellis == null and linear.exl3 == null and linear.ggml == null);
    try std.testing.expectEqual(@as(u64, 0), linear.ck_rows);
    try std.testing.expectEqual(@as(u64, 0), linear.ck_part);
}

test "a host-only build refuses BF16 bindings before accessing a device" {
    if (!cuda.kernels.available) {
        const std = @import("std");
        try std.testing.expectError(error.BuiltWithoutKernels, Bindings.load(undefined, undefined));
    }
}
