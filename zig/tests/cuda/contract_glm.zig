//! GLM FP4-Local arithmetic contracts; all objects and modules outlive their registered hooks.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");
const attn = @import("contract_glm_attention.zig");
const merge = @import("contract_glm_attention_merge.zig");
const dense = @import("contract_glm_dense.zig");
const cast = @import("contract_glm_dense_cast.zig");
const head = @import("contract_glm_dense_head.zig");
const expert = @import("contract_glm_experts.zig");
const prompt = @import("contract_glm_experts_prompt.zig");
const g = core.contract.gates;

pub fn run(gpu: check.Gpu, s: cuda.Stream, affine4: *const cuda.qlinear.Affine4Kernels, tally: *g.Tally) !void {
    const a = gpu.gpa;
    var glue = try cuda.Module.load(gpu.d, cuda.kernels.glm_glue);
    defer glue.unload();
    var pointwise = try cuda.Module.load(gpu.d, cuda.kernels.torch_pointwise);
    defer pointwise.unload();
    var qmm = try cuda.Module.load(gpu.d, cuda.kernels.dflash_qmm);
    defer qmm.unload();
    var source_mod = try cuda.Module.load(gpu.d, cuda.kernels.exlr8_dense);
    defer source_mod.unload();
    const source = try cuda.exlr8.Dense.resolve(source_mod);
    const sms: usize = @intCast(try gpu.ctx.attribute(.multiprocessor_count));
    var experts = try cuda.experts.Nvfp4Experts.load(gpu.d, sms);
    defer experts.unload();
    try experts.loadCandidates(sms);
    const upcast = try pointwise.function("tf_bf16_to_f32_kernel");
    var kernels: std.ArrayList(g.Kernel) = .empty;
    defer kernels.deinit(a);
    const kinds = std.enums.values(attn.Kind);
    var attention: [kinds.len]attn.Attention = undefined;
    var na: usize = 0;
    defer for (attention[0..na]) |*v| v.deinit();
    for (kinds, &attention) |kind, *v| {
        v.* = try attn.Attention.init(gpu, s, glue, kind);
        na += 1;
        try kernels.append(a, v.kernel());
    }
    var merges: [2]merge.Merge = undefined;
    var nm: usize = 0;
    defer for (merges[0..nm]) |*v| v.deinit();
    for (&merges, 0..) |*v, i| {
        v.* = try merge.Merge.init(gpu, s, glue, i != 0);
        nm += 1;
        try kernels.append(a, v.kernel());
    }
    var projections: [8]dense.Source = undefined;
    for (std.enums.values(dense.Kind), 0..) |kind, i| for (0..2) |j| {
        projections[2 * i + j] = .{ .rig = .{ .gpu = gpu, .s = s }, .dense = &source, .cast = upcast, .kind = kind, .sliced = j != 0 };
    };
    defer for (&projections) |*v| v.deinit();
    for (&projections) |*v| try kernels.append(a, v.kernel());
    var conversion: cast.Cast = .{ .rig = .{ .gpu = gpu, .s = s }, .f = upcast };
    defer conversion.deinit();
    try kernels.append(a, conversion.kernel());
    var heads: [3]head.Head = undefined;
    var nh: usize = 0;
    defer for (heads[0..nh]) |*v| v.deinit();
    inline for (.{ 16, 32, 64 }, &heads) |bm, *v| {
        v.* = .{ .rig = .{ .gpu = gpu, .s = s }, .affine4 = affine4, .f = try qmm.function(std.fmt.comptimePrint("glm_qmm_{d}_0_0", .{bm})), .bm = bm };
        nh += 1;
        try kernels.append(a, v.kernel());
    }
    var expert_cases: [expert.kernel_count]expert.Expert = undefined;
    for (expert.variants, 0..) |variant, vi| for (expert.epilogues, 0..) |epi, ei| {
        expert_cases[vi * expert.epilogues.len + ei] = .{ .rig = .{ .gpu = gpu, .s = s }, .experts = &experts, .variant = variant, .epi = epi };
    };
    defer for (&expert_cases) |*v| v.deinit();
    for (&expert_cases) |*v| try kernels.append(a, v.kernel());
    var clamped: [expert.variants.len]expert.Expert = undefined;
    for (expert.variants, &clamped) |variant, *v| {
        v.* = .{ .rig = .{ .gpu = gpu, .s = s }, .experts = &experts, .variant = variant, .epi = .swiglu, .clamped = true };
    }
    defer for (&clamped) |*v| v.deinit();
    for (&clamped) |*v| try kernels.append(a, v.kernel());
    var prompt_cases: [prompt.kernel_count + prompt.clamped_count]prompt.Prompt = undefined;
    for (prompt.epilogues, 0..) |epi, i| prompt_cases[i] = prompt.init(.{ .gpu = gpu, .s = s }, &experts, epi, false);
    prompt_cases[prompt.kernel_count] = prompt.init(.{ .gpu = gpu, .s = s }, &experts, .swiglu, true);
    defer for (&prompt_cases) |*v| v.deinit();
    for (&prompt_cases) |*v| try kernels.append(a, v.kernel());
    try g.runAll(a, kernels.items, 0x5388, tally);
}
