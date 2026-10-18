//! GLM CUDA launch declarations, scoped to the measured deployment target rather than unmeasured architectures.
const limits = @import("limits.zig");
const U = limits.Usage;

const missing: U = .{ .regs = 0, .static = 0 };
const expert_prefix = "_ZN16tf_nvfp4_experts19nvfp4_expert_kernelILi";
const expert_suffix = "ELi4EEEvPK13__nv_bfloat16iiPK5uint4PKfiiPKiSA_SA_Pvifi";

fn launch(comptime R: type, comptime image: []const u8, comptime symbol: []const u8, threads: u32, dynamic: u32, usage: U) R {
    var recorded: [limits.sms.len]U = @splat(missing);
    for (limits.sms, 0..) |t, i| if (t.sm == 121) {
        recorded[i] = usage;
    };
    return .{
        .launch = .{ .name = "GLM " ++ symbol, .symbol = symbol, .image = image, .threads = threads, .dynamic = dynamic, .opt_in = dynamic > 48 * 1024 },
        .usage = recorded,
        .target_sm = 121,
    };
}

/// The sm_121 measurements and exact image hashes are retained in fixtures/glm-sm121-960ed42.json.
pub fn records(comptime R: type) [34]R {
    return .{
        launch(R, "glm_glue", "tf_glm_chunks_16", 256, 83328, .{ .regs = 121, .static = 0, .bound = 256 }),
        launch(R, "glm_glue", "tf_glm_chunks_32", 256, 101120, .{ .regs = 128, .static = 0, .bound = 256 }),
        launch(R, "glm_glue", "tf_glm_chunks_16_fp8", 256, 83328, .{ .regs = 67, .static = 0, .bound = 256 }),
        launch(R, "glm_glue", "tf_glm_chunks_32_fp8", 256, 101120, .{ .regs = 130, .static = 0, .bound = 256 }),
        launch(R, "glm_glue", "tf_glm_chunks_32_fp8load", 256, 101376, .{ .regs = 128, .static = 0, .bound = 256 }),
        launch(R, "glm_glue", "tf_glm_merge", 128, 0, .{ .regs = 34, .static = 0, .bound = 128 }),
        launch(R, "nvfp4_experts", expert_prefix ++ "2ELi2" ++ expert_suffix, 128, 0, .{ .regs = 128, .static = 0, .bound = 128 }),
        launch(R, "nvfp4_experts", expert_prefix ++ "1ELi0" ++ expert_suffix, 128, 0, .{ .regs = 96, .static = 0, .bound = 128 }),
        launch(R, "nvfp4_experts", expert_prefix ++ "1ELi3" ++ expert_suffix, 128, 0, .{ .regs = 96, .static = 0, .bound = 128 }),
        launch(R, "nvfp4_experts", expert_prefix ++ "1ELi1" ++ expert_suffix, 128, 0, .{ .regs = 96, .static = 0, .bound = 128 }),
        launch(R, "nvfp4_experts", "tf_nvfp4_pipe4_swiglu", 128, 0, .{ .regs = 166, .static = 0, .bound = 128 }),
        launch(R, "nvfp4_experts", "tf_nvfp4_pipe4_down_f32", 128, 0, .{ .regs = 128, .static = 0, .bound = 128 }),
        launch(R, "nvfp4_experts", "tf_nvfp4_pipe4_down_bf16", 128, 0, .{ .regs = 128, .static = 0, .bound = 128 }),
        launch(R, "nvfp4_experts", "tf_nvfp4_pipe4_relu2", 128, 0, .{ .regs = 128, .static = 0, .bound = 128 }),
        launch(R, "nvfp4_experts", "tf_nvfp4_n16_swiglu", 128, 0, .{ .regs = 120, .static = 0, .bound = 128 }),
        launch(R, "nvfp4_experts", "tf_nvfp4_n16_down_f32", 128, 0, .{ .regs = 80, .static = 0, .bound = 128 }),
        launch(R, "nvfp4_experts", "tf_nvfp4_n16_down_bf16", 128, 0, .{ .regs = 80, .static = 0, .bound = 128 }),
        launch(R, "nvfp4_experts", "tf_nvfp4_n16_relu2", 128, 0, .{ .regs = 80, .static = 0, .bound = 128 }),
        launch(R, "nvfp4_experts", "tf_nvfp4_n16_pipe4_swiglu", 128, 0, .{ .regs = 158, .static = 0, .bound = 128 }),
        launch(R, "nvfp4_experts", "tf_nvfp4_n16_pipe4_down_f32", 128, 0, .{ .regs = 117, .static = 0, .bound = 128 }),
        launch(R, "nvfp4_experts", "tf_nvfp4_n16_pipe4_down_bf16", 128, 0, .{ .regs = 117, .static = 0, .bound = 128 }),
        launch(R, "nvfp4_experts", "tf_nvfp4_n16_pipe4_relu2", 128, 0, .{ .regs = 117, .static = 0, .bound = 128 }),
        launch(R, "nvfp4_experts", "tf_nvfp4_prompt_swiglu", 128, 34816, .{ .regs = 208, .static = 0, .bound = 128 }),
        launch(R, "nvfp4_experts", "tf_nvfp4_prompt_down", 256, 34816, .{ .regs = 126, .static = 0, .bound = 256 }),
        launch(R, "nvfp4_experts", "tf_nvfp4_prompt_relu2", 256, 34816, .{ .regs = 126, .static = 0, .bound = 256 }),
        launch(R, "exlr8_dense", "tf_exlr8_dense_1", 256, 0, .{ .regs = 76, .static = 512, .bound = 256 }),
        launch(R, "exlr8_dense", "tf_exlr8_dense_2", 256, 0, .{ .regs = 64, .static = 1024, .bound = 256 }),
        launch(R, "exlr8_dense", "tf_exlr8_dense_4", 256, 0, .{ .regs = 125, .static = 2048, .bound = 256 }),
        launch(R, "exlr8_dense", "tf_exlr8_dense_8", 256, 0, .{ .regs = 128, .static = 4096, .bound = 256 }),
        launch(R, "exlr8_dense", "tf_exlr8_dense_16", 256, 0, .{ .regs = 252, .static = 8192, .bound = 256 }),
        launch(R, "torch_pointwise", "tf_bf16_to_f32_kernel", 256, 0, .{ .regs = 14, .static = 0, .bound = 0 }),
        launch(R, "dflash_qmm", "glm_qmm_16_0_0", 128, 17664, .{ .regs = 64, .static = 0, .bound = 128 }),
        launch(R, "dflash_qmm", "glm_qmm_32_0_0", 128, 26112, .{ .regs = 78, .static = 0, .bound = 128 }),
        launch(R, "dflash_qmm", "glm_qmm_64_0_0", 128, 43008, .{ .regs = 124, .static = 0, .bound = 128 }),
    };
}

test "the GLM records equal the retained sm_121 measurements, one record per measured symbol" {
    const std = @import("std");
    const Record = struct { symbol: []const u8, regs: u32, static: u32, bound: u32, threads: u32, dynamic: u32 };
    const Image = struct { image: []const u8, sha256: []const u8, sm: u32, records: []const Record };
    const Fixture = struct { images: []const Image };
    const a = std.testing.allocator;
    const parsed = try std.json.parseFromSlice(Fixture, a, @embedFile("fixtures/glm-sm121-960ed42.json"), .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    const R = @import("registry.zig").CudaRecord;
    const table = records(R);
    const at = for (limits.sms, 0..) |t, i| {
        if (t.sm == 121) break i;
    } else unreachable;
    var measured: usize = 0;
    for (parsed.value.images) |image| for (image.records) |m| {
        try std.testing.expectEqual(@as(u32, 121), image.sm);
        var found: usize = 0;
        for (table) |r| if (std.mem.eql(u8, r.launch.symbol, m.symbol)) {
            found += 1;
            try std.testing.expect(std.mem.startsWith(u8, image.image, r.launch.image));
            try std.testing.expectEqualStrings(".fatbin", image.image[r.launch.image.len..]);
            try std.testing.expectEqual(U{ .regs = m.regs, .static = m.static, .bound = m.bound }, r.usage[at]);
            try std.testing.expectEqual(m.threads, r.launch.threads);
            try std.testing.expectEqual(m.dynamic, r.launch.dynamic);
        };
        try std.testing.expectEqual(@as(usize, 1), found);
        measured += 1;
    };
    try std.testing.expectEqual(table.len, measured);
}
