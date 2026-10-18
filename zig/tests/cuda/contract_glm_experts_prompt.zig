//! Existing NVFP4 prompt epilogues over the GLM FP4-Local served envelope and independent 64-pair plans.
const cuda = @import("cuda");
const decode = @import("contract_glm_experts.zig");
const rig = @import("contract_rig.zig");

pub const Prompt = decode.Expert;
pub const epilogues = [_]decode.Epi{ .relu2, .swiglu, .down_bf16 };
pub const kernel_count = epilogues.len;
pub const clamped_count = 1;

pub fn init(r: rig.Rig, experts: *const cuda.experts.Nvfp4Experts, epi: decode.Epi, clamped: bool) Prompt {
    return .{ .rig = r, .experts = experts, .variant = .baseline, .epi = epi, .prompt = true, .clamped = clamped };
}
