//! The GLM registry's declared launches against the production launchers' shared bytes and block widths (CPU only).
const std = @import("std");
const core = @import("core");
const cuda = @import("cuda");
const glm = @import("glm_cuda");
const registry = core.contract.registry;
const X = cuda.experts.Nvfp4Experts;

const Want = struct { symbol: []const u8, threads: u32, dynamic: u32 };

const wants = [_]Want{
    .{ .symbol = "tf_glm_chunks_16", .threads = 256, .dynamic = glm.glue.chunksSmem(16) },
    .{ .symbol = "tf_glm_chunks_32", .threads = 256, .dynamic = glm.glue.chunksSmem(32) },
    .{ .symbol = "tf_glm_chunks_16_fp8", .threads = 256, .dynamic = glm.glue.chunksSmem(16) },
    .{ .symbol = "tf_glm_chunks_32_fp8", .threads = 256, .dynamic = glm.glue.chunksSmem(32) },
    .{ .symbol = "tf_glm_chunks_32_fp8load", .threads = 256, .dynamic = glm.glue.chunksFP8LoadSmem() },
    .{ .symbol = "tf_glm_merge", .threads = 128, .dynamic = 0 },
    .{ .symbol = "tf_nvfp4_prompt_relu2", .threads = X.prompt_wn[0] * 32, .dynamic = X.promptSmem(1, X.prompt_wn[0]) },
    .{ .symbol = "tf_nvfp4_prompt_swiglu", .threads = X.prompt_wn[1] * 32, .dynamic = X.promptSmem(2, X.prompt_wn[1]) },
    .{ .symbol = "tf_nvfp4_prompt_down", .threads = X.prompt_wn[2] * 32, .dynamic = X.promptSmem(1, X.prompt_wn[2]) },
};

fn record(symbol: []const u8) ?registry.CudaRecord {
    for (registry.all_cuda) |r| if (std.mem.eql(u8, r.launch.symbol, symbol)) return r;
    return null;
}

test "GLM attention and prompt expert records declare the production launch's threads and shared bytes" {
    for (wants) |w| {
        const r = record(w.symbol) orelse return error.UnregisteredLaunch;
        try std.testing.expectEqual(w.threads, r.launch.threads);
        try std.testing.expectEqual(w.dynamic, r.launch.dynamic);
        try std.testing.expectEqual(w.dynamic > 48 * 1024, r.launch.opt_in);
    }
}
