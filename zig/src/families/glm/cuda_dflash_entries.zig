const std = @import("std");
const cuda = @import("cuda");
const Spec = struct { name: [:0]const u8, shared: u32 };
const specs = [_]Spec{
    .{ .name = "glm_qmm_16_0_0", .shared = 17664 },
    .{ .name = "glm_qmm_16_1_0", .shared = 17664 },
    .{ .name = "glm_qmm_32_0_0", .shared = 26112 },
    .{ .name = "glm_qmm_32_1_0", .shared = 26112 },
    .{ .name = "glm_qmm_64_0_0", .shared = 43008 },
    .{ .name = "glm_qmm_64_1_0", .shared = 43008 },
    .{ .name = "glm_qmm_cfg2_0_0", .shared = 26880 },
    .{ .name = "glm_qmm_cfg2_1_0", .shared = 26880 },
    .{ .name = "glm_qmm_cfg4_0_0", .shared = 40320 },
    .{ .name = "glm_qmm_cfg4_1_0", .shared = 40320 },
    .{ .name = "glm_qmm_reduce_0", .shared = 0 },
    .{ .name = "glm_qmm_reduce_1", .shared = 0 },
};
pub const Entries = struct {
    module: cuda.Module,
    functions: [specs.len]cuda.Function,
    pub fn load(d: *const cuda.Driver) !Entries {
        var module = try cuda.Module.load(d, cuda.kernels.dflash_qmm);
        errdefer module.unload();
        var functions: [specs.len]cuda.Function = undefined;
        for (specs, &functions) |spec, *f| {
            f.* = try module.function(spec.name);
            if (spec.shared != 0) try f.allowDynamicShared(spec.shared);
        }
        return .{ .module = module, .functions = functions };
    }
    pub fn deinit(e: *Entries) void {
        e.module.unload();
    }
    pub fn launch(e: *const Entries, image: usize, name: [:0]const u8, config: cuda.Config, stream: cuda.Stream, args: *cuda.Args) !void {
        if (image != 0) return error.BadDraftImage;
        for (specs, e.functions) |spec, f| if (std.mem.eql(u8, spec.name, name)) {
            if (config.shared != spec.shared) return error.BadDraftSharedMemory;
            return cuda.launch.launch(f, config, stream, args);
        };
        return error.BadDraftSymbol;
    }
};
test "private entry table matches donor tile shared memory sizes" {
    const mm = @import("cuda_dflash_matmuls.zig");
    try std.testing.expectEqual(specs[0].shared, mm.tile(8, 77440, 4096, false).shared(false));
    try std.testing.expectEqual(specs[2].shared, mm.tile(32, 4096, 4096, false).shared(false));
    try std.testing.expectEqual(specs[4].shared, mm.tile(64, 4096, 4096, false).shared(false));
    try std.testing.expectEqual(specs[6].shared, mm.tile(8, 4096, 4096, false).shared(false));
    try std.testing.expectEqual(specs[8].shared, mm.tile(8, 4096, 8192, false).shared(false));
}
