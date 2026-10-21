//! GLM-5.3-Flash on CUDA: the family's parts, ported from the Python family's CUDA path.

pub const experts = @import("cuda_experts.zig");
pub const exl3 = @import("cuda_exl3.zig");
pub const mm = @import("cuda_mm.zig");
pub const relay = @import("cuda_relay.zig");
pub const vision_config = @import("vision_config.zig");
pub const image_core = @import("image_core.zig");
pub const image = image_core.For(@import("core").picture);
pub const glue = @import("cuda_glue.zig");
pub const kda = @import("cuda_kda.zig");
pub const b16 = @import("cuda_b16.zig");
pub const source_dense = @import("cuda_weight_source_dense.zig");
pub const kv_owner = @import("cuda_kv_owner.zig");
pub const kv_pool = @import("cuda_kv_pool.zig");
pub const layout = @import("cuda_layout.zig");
pub const cache = @import("cuda_cache.zig");
pub const latent = @import("cuda_latent.zig");
pub const sparse = @import("cuda_sparse.zig");
pub const moe = @import("cuda_moe.zig");
pub const kda_block = @import("cuda_kda_block.zig");
pub const dsa_block = @import("cuda_dsa_block.zig");
pub const mlp = @import("cuda_mlp.zig");
pub const shard = @import("cuda_shard.zig");
pub const weights = @import("cuda_weights.zig");
pub const forward = @import("cuda_forward.zig");
pub const config = @import("config.zig");
pub const quant = @import("cuda_quant.zig");
pub const q4 = @import("cuda_q4.zig");
pub const ranks = @import("cuda_ranks.zig");
pub const draft_profile = @import("cuda_draft_profile.zig");
pub const copy_drafts = @import("cuda_copy_drafts.zig");
pub const target_graph = @import("cuda_target_graph.zig");
pub const target_meta = @import("cuda_target_meta.zig");
pub const target_forward = @import("cuda_target_forward.zig");
pub const sampler = @import("cuda_sampler.zig");
pub const greedy = @import("cuda_greedy.zig");
pub const prompt_sink = @import("cuda_prompt_sink.zig");
pub const prompt_logprobs = @import("cuda_prompt_logprobs.zig");
pub const engine = @import("cuda_engine.zig");
pub const follow = @import("cuda_follow.zig");
pub const lanes = @import("cuda_lanes.zig");
pub const native = @import("cuda_native.zig");
pub const vision = @import("cuda_vision.zig");
pub const vision_codec = @import("cuda_vision_codec.zig");
pub const vision_engine = @import("cuda_vision_engine.zig");
pub const vision_pipeline = @import("cuda_vision_pipeline.zig");
pub const vision_binding = @import("cuda_vision_binding.zig");
pub const vision_replica = @import("cuda_vision_replica.zig");
pub const vision_probe = @import("cuda_vision_probe.zig");
pub const vision_runtime = @import("cuda_vision_runtime.zig");

test {
    _ = @import("cuda_load_owner.zig");
    _ = @import("cuda_images_interleave_test.zig");
    _ = exl3;
    _ = mm;
    _ = relay;
    _ = image_core;
    _ = @import("cuda_qpack.zig");
    _ = @import("cuda_tile.zig");
    _ = @import("cuda_l2pf_plan.zig");
    _ = @import("cuda_l2pf.zig");
    _ = @import("cuda_attn_fp8load_test.zig");
    _ = @import("cuda_prompt_latent_policy_test.zig");
    _ = @import("cuda_prompt_latent_aot.zig");
    _ = @import("cuda_verify_plan_test.zig");
    _ = @import("cuda_checkpoint.zig");
    _ = @import("cuda_weight_source_test.zig");
    _ = experts;
    _ = glue;
    _ = kda;
    _ = b16;
    _ = cache;
    _ = layout;
    _ = kv_pool;
    _ = kv_owner;
    _ = kv_pool.extent;
    _ = latent;
    _ = sparse;
    _ = moe;
    _ = kda_block;
    _ = dsa_block;
    _ = @import("cuda_prefill_plan.zig");
    _ = mlp;
    _ = shard;
    _ = weights;
    _ = forward;
    _ = quant;
    _ = q4;
    _ = ranks;
    _ = draft_profile;
    _ = copy_drafts;
    _ = target_graph;
    _ = target_meta;
    _ = target_forward;
    _ = sampler;
    _ = prompt_logprobs;
    _ = engine;
    _ = follow;
    _ = lanes;
    _ = native;
    _ = vision;
    _ = vision_codec;
    _ = vision_engine;
    _ = vision_pipeline;
    _ = vision_binding;
    _ = vision_replica;
    _ = vision_probe;
    _ = vision_runtime;
    _ = @import("heat"); // upstream's heat bands, run where the GLM ranks use them
}

// Native entry coverage belongs to the family boundary, not the independently runnable source-format gather tests.
test "mixed loader native entry points compile" {
    _ = &@import("cuda_weights.zig").load;
    _ = &@import("cuda_engine.zig").Engine.init;
    _ = &@import("cuda_vision_runtime.zig").Service.open;
}

/// Components used by the model-free tap-commit proof, without constructing a DFlash engine.
pub const taps_fixture = struct {
    pub const forward = @import("cuda_dflash_forward.zig");
    pub const weights = @import("cuda_dflash_weights.zig");
    pub const config = @import("cuda_dflash_config.zig");
    pub const entries = @import("cuda_dflash_entries.zig");
    pub const ops = @import("cuda_dflash_ops.zig");
    pub const aot = @import("cuda_dflash_aot.zig");
};
pub const dflash = @import("dflash.zig");
pub const dflash_candidates = @import("dflash_candidates.zig");
pub const dflash_load_owner = @import("dflash_load_owner.zig");

pub const prompts = @import("cuda_prompts.zig");
pub const prompt_hold = @import("cuda_prompt_hold.zig");
pub const prompt_wire = @import("cuda_prompt_wire.zig");
pub const structured = @import("cuda_structured.zig");
