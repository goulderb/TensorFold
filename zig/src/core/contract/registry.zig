//! The registered launches gate 5 checks: each one's threads and memory, and the limits its compiles recorded.
const std = @import("std");
const limits = @import("limits.zig");

const M = limits.MetalLaunch;
const C = limits.CudaLaunch;
const U = limits.Usage;

/// A CUDA launch and what nvcc made of it for sm_80, sm_86, sm_89, sm_120 and sm_121 (limits.sms' order; regs 0: none).
pub const CudaRecord = struct {
    launch: C,
    usage: [limits.sms.len]U,
    /// Zero declares every registry target; a scoped compile is not evidence for the other targets.
    target_sm: u32 = 0,

    pub fn applies(self: CudaRecord, sm: u32) bool {
        return self.target_sm == 0 or self.target_sm == sm;
    }
};

const none: [limits.sms.len]U = @splat(.{ .regs = 0, .static = 0 });
const group_symbol = "_ZN12tf_qmm_group12group_kernelILi64ELi16ELi64ELi1ELi4ELi8ELb0ELb0ELb0ELb0EEEvPK13__nv_bfloat16PKfNS_5PartsEiiiii";
const gemv_symbol = "_ZN12tf_lane_gemv11gemv_kernelILi64ELi64ELi8EEEvPK13__nv_bfloat16PKfNS_4PartEiii";
const prompt_symbol = "_ZN11tf_prompt1613prompt_kernelILi{d}ELi128ELi128ELi2ELi2ELi2ELb{d}EEEvPK13__nv_bfloat16PKhS5_fPviiiiii";
const qmmf_f32_symbol = "_ZN7tf_qmmf11qmmf_kernelILi0ELi{d}ELi64ELi1ELi4ELi4ELb1ELb{d}ELb{d}EEEvPK13__nv_bfloat16PKhS5_fPvPfiiiiiii";
const vision80_symbol = "_ZN20tf_prefill_attention13vision_kernelILi80ELi8EEEvPK13__nv_bfloat16PS1_iif";
const full72_symbol = "_ZN20tf_prefill_attention11full_kernelILi72ELi8ELi8EEEvPK13__nv_bfloat16PS1_iif";

/// CUDA kernels as launched; tf-cuda-test kernel-contract-limits re-derives the usage records.
pub const cuda = [_]CudaRecord{
    .{ .launch = .{ .name = "flashnext_gdn flashnext_gdn_chain", .symbol = "flashnext_gdn_chain", .image = "flashnext_gdn", .threads = 1024 }, .usage = .{ .{ .regs = 51, .static = 2060, .bound = 1024 }, .{ .regs = 56, .static = 2060, .bound = 1024 }, .{ .regs = 56, .static = 2060, .bound = 1024 }, .{ .regs = 62, .static = 2060, .bound = 1024 }, .{ .regs = 62, .static = 2060, .bound = 1024 } } },
    .{ .launch = .{ .name = "flashnext_hc fn_add_streams", .symbol = "fn_add_streams", .image = "flashnext_hc", .threads = 256 }, .usage = .{ .{ .regs = 32, .static = 0, .bound = 0 }, .{ .regs = 40, .static = 0, .bound = 0 }, .{ .regs = 40, .static = 0, .bound = 0 }, .{ .regs = 35, .static = 0, .bound = 0 }, .{ .regs = 35, .static = 0, .bound = 0 } } },
    .{ .launch = .{ .name = "flashnext_attn fn_attn_chunks", .symbol = "fn_attn_chunks", .image = "flashnext_attn", .threads = 128, .dynamic = 43520, .opt_in = true }, .usage = .{ .{ .regs = 71, .static = 640, .bound = 128 }, .{ .regs = 71, .static = 640, .bound = 128 }, .{ .regs = 71, .static = 640, .bound = 128 }, .{ .regs = 72, .static = 640, .bound = 128 }, .{ .regs = 72, .static = 640, .bound = 128 } } },
    .{ .launch = .{ .name = "flashnext_attn fn_attn_chunks_ahead", .symbol = "fn_attn_chunks_ahead", .image = "flashnext_attn", .threads = 128, .dynamic = 77312, .opt_in = true }, .usage = .{ .{ .regs = 72, .static = 768, .bound = 128 }, .{ .regs = 72, .static = 768, .bound = 128 }, .{ .regs = 72, .static = 768, .bound = 128 }, .{ .regs = 71, .static = 768, .bound = 128 }, .{ .regs = 71, .static = 768, .bound = 128 } } },
    .{ .launch = .{ .name = "flashnext_attn fn_attn_gate", .symbol = "fn_attn_gate", .image = "flashnext_attn", .threads = 256 }, .usage = .{ .{ .regs = 22, .static = 0, .bound = 256 }, .{ .regs = 23, .static = 0, .bound = 256 }, .{ .regs = 23, .static = 0, .bound = 256 }, .{ .regs = 22, .static = 0, .bound = 256 }, .{ .regs = 22, .static = 0, .bound = 256 } } },
    .{ .launch = .{ .name = "flashnext_attn fn_attn_merge", .symbol = "fn_attn_merge", .image = "flashnext_attn", .threads = 256 }, .usage = .{ .{ .regs = 26, .static = 0, .bound = 256 }, .{ .regs = 28, .static = 0, .bound = 256 }, .{ .regs = 28, .static = 0, .bound = 256 }, .{ .regs = 40, .static = 0, .bound = 256 }, .{ .regs = 26, .static = 0, .bound = 256 } } },
    .{ .launch = .{ .name = "flashnext_attn fn_attn_prep", .symbol = "fn_attn_prep", .image = "flashnext_attn", .threads = 256 }, .usage = .{ .{ .regs = 32, .static = 128, .bound = 256 }, .{ .regs = 34, .static = 128, .bound = 256 }, .{ .regs = 34, .static = 128, .bound = 256 }, .{ .regs = 30, .static = 128, .bound = 256 }, .{ .regs = 30, .static = 128, .bound = 256 } } },
    .{ .launch = .{ .name = "flashnext_batch fn_gdn_front", .symbol = "fn_gdn_front", .image = "flashnext_batch", .threads = 256 }, .usage = .{ .{ .regs = 26, .static = 1024, .bound = 256 }, .{ .regs = 26, .static = 1024, .bound = 256 }, .{ .regs = 26, .static = 1024, .bound = 256 }, .{ .regs = 32, .static = 1024, .bound = 256 }, .{ .regs = 32, .static = 1024, .bound = 256 } } },
    .{ .launch = .{ .name = "flashnext_batch fn_gdn_norm", .symbol = "fn_gdn_norm", .image = "flashnext_batch", .threads = 128 }, .usage = .{ .{ .regs = 24, .static = 516, .bound = 128 }, .{ .regs = 26, .static = 516, .bound = 128 }, .{ .regs = 26, .static = 516, .bound = 128 }, .{ .regs = 24, .static = 516, .bound = 128 }, .{ .regs = 24, .static = 516, .bound = 128 } } },
    .{ .launch = .{ .name = "flashnext_batch fn_gdn_rows", .symbol = "fn_gdn_rows", .image = "flashnext_batch", .threads = 128 }, .usage = .{ .{ .regs = 64, .static = 0, .bound = 128 }, .{ .regs = 64, .static = 0, .bound = 128 }, .{ .regs = 64, .static = 0, .bound = 128 }, .{ .regs = 64, .static = 0, .bound = 128 }, .{ .regs = 64, .static = 0, .bound = 128 } } },
    .{ .launch = .{ .name = "flashnext_batch fn_gdn_rows_to", .symbol = "fn_gdn_rows_to", .image = "flashnext_batch", .threads = 128 }, .usage = .{ .{ .regs = 64, .static = 0, .bound = 128 }, .{ .regs = 64, .static = 0, .bound = 128 }, .{ .regs = 64, .static = 0, .bound = 128 }, .{ .regs = 64, .static = 0, .bound = 128 }, .{ .regs = 64, .static = 0, .bound = 128 } } },
    .{ .launch = .{ .name = "flashnext_hc fn_hc_dense_mix", .symbol = "fn_hc_dense_mix", .image = "flashnext_hc", .threads = 32 }, .usage = .{ .{ .regs = 22, .static = 128, .bound = 0 }, .{ .regs = 36, .static = 128, .bound = 0 }, .{ .regs = 36, .static = 128, .bound = 0 }, .{ .regs = 36, .static = 128, .bound = 0 }, .{ .regs = 36, .static = 128, .bound = 0 } } },
    .{ .launch = .{ .name = "flashnext_hc fn_hc_down", .symbol = "fn_hc_down", .image = "flashnext_hc", .threads = 128, .dynamic = 11136 }, .usage = .{ .{ .regs = 56, .static = 64, .bound = 128 }, .{ .regs = 54, .static = 64, .bound = 128 }, .{ .regs = 54, .static = 64, .bound = 128 }, .{ .regs = 60, .static = 64, .bound = 128 }, .{ .regs = 60, .static = 64, .bound = 128 } } },
    .{ .launch = .{ .name = "flashnext_hc fn_hc_normalize", .symbol = "fn_hc_normalize", .image = "flashnext_hc", .threads = 256 }, .usage = .{ .{ .regs = 32, .static = 16, .bound = 0 }, .{ .regs = 40, .static = 16, .bound = 0 }, .{ .regs = 40, .static = 16, .bound = 0 }, .{ .regs = 40, .static = 16, .bound = 0 }, .{ .regs = 38, .static = 16, .bound = 0 } } },
    .{ .launch = .{ .name = "flashnext_hc fn_hc_reduce_act", .symbol = "fn_hc_reduce_act", .image = "flashnext_hc", .threads = 352 }, .usage = .{ .{ .regs = 32, .static = 4096, .bound = 1024 }, .{ .regs = 63, .static = 4096, .bound = 1024 }, .{ .regs = 63, .static = 4096, .bound = 1024 }, .{ .regs = 58, .static = 4096, .bound = 1024 }, .{ .regs = 58, .static = 4096, .bound = 1024 } } },
    .{ .launch = .{ .name = "flashnext_hc fn_hc_upmix", .symbol = "fn_hc_upmix", .image = "flashnext_hc", .threads = 256, .dynamic = 18688 }, .usage = .{ .{ .regs = 69, .static = 2048, .bound = 256 }, .{ .regs = 64, .static = 2048, .bound = 256 }, .{ .regs = 64, .static = 2048, .bound = 256 }, .{ .regs = 78, .static = 2048, .bound = 256 }, .{ .regs = 78, .static = 2048, .bound = 256 } } },
    .{ .launch = .{ .name = "flashnext_hc fn_hc_writeback0", .symbol = "fn_hc_writeback0", .image = "flashnext_hc", .threads = 256 }, .usage = .{ .{ .regs = 30, .static = 256, .bound = 256 }, .{ .regs = 30, .static = 256, .bound = 256 }, .{ .regs = 30, .static = 256, .bound = 256 }, .{ .regs = 33, .static = 256, .bound = 256 }, .{ .regs = 37, .static = 256, .bound = 256 } } },
    .{ .launch = .{ .name = "flashnext_hc fn_hc_writeback_moe", .symbol = "fn_hc_writeback_moe", .image = "flashnext_hc", .threads = 256 }, .usage = .{ .{ .regs = 32, .static = 256, .bound = 256 }, .{ .regs = 40, .static = 256, .bound = 256 }, .{ .regs = 40, .static = 256, .bound = 256 }, .{ .regs = 40, .static = 256, .bound = 256 }, .{ .regs = 40, .static = 256, .bound = 256 } } },
    .{ .launch = .{ .name = "flashnext_hc fn_hc_writeback_slices", .symbol = "fn_hc_writeback_slices", .image = "flashnext_hc", .threads = 256 }, .usage = .{ .{ .regs = 32, .static = 256, .bound = 256 }, .{ .regs = 38, .static = 256, .bound = 256 }, .{ .regs = 38, .static = 256, .bound = 256 }, .{ .regs = 40, .static = 256, .bound = 256 }, .{ .regs = 40, .static = 256, .bound = 256 } } },
    .{ .launch = .{ .name = "flashnext_hc fn_ple_conv", .symbol = "fn_ple_conv", .image = "flashnext_hc", .threads = 512 }, .usage = .{ .{ .regs = 23, .static = 0, .bound = 512 }, .{ .regs = 35, .static = 0, .bound = 512 }, .{ .regs = 35, .static = 0, .bound = 512 }, .{ .regs = 35, .static = 0, .bound = 512 }, .{ .regs = 35, .static = 0, .bound = 512 } } },
    .{ .launch = .{ .name = "flashnext_hc fn_ple_gate", .symbol = "fn_ple_gate", .image = "flashnext_hc", .threads = 512 }, .usage = .{ .{ .regs = 32, .static = 128, .bound = 512 }, .{ .regs = 34, .static = 128, .bound = 512 }, .{ .regs = 34, .static = 128, .bound = 512 }, .{ .regs = 35, .static = 128, .bound = 512 }, .{ .regs = 35, .static = 128, .bound = 512 } } },
    .{ .launch = .{ .name = "flashnext_attn fn_qsa_pool", .symbol = "fn_qsa_pool", .image = "flashnext_attn", .threads = 128 }, .usage = .{ .{ .regs = 32, .static = 1152, .bound = 128 }, .{ .regs = 40, .static = 1152, .bound = 128 }, .{ .regs = 40, .static = 1152, .bound = 128 }, .{ .regs = 40, .static = 1152, .bound = 128 }, .{ .regs = 39, .static = 1152, .bound = 128 } } },
    .{ .launch = .{ .name = "flashnext_attn fn_qsa_scores", .symbol = "fn_qsa_scores", .image = "flashnext_attn", .threads = 256 }, .usage = .{ .{ .regs = 40, .static = 4096, .bound = 256 }, .{ .regs = 46, .static = 4096, .bound = 256 }, .{ .regs = 46, .static = 4096, .bound = 256 }, .{ .regs = 48, .static = 4096, .bound = 256 }, .{ .regs = 48, .static = 4096, .bound = 256 } } },
    .{ .launch = .{ .name = "flashnext_batch fn_qsa_select", .symbol = "fn_qsa_select", .image = "flashnext_batch", .threads = 1024 }, .usage = .{ .{ .regs = 32, .static = 1160, .bound = 1024 }, .{ .regs = 50, .static = 1160, .bound = 1024 }, .{ .regs = 50, .static = 1160, .bound = 1024 }, .{ .regs = 44, .static = 1160, .bound = 1024 }, .{ .regs = 40, .static = 1160, .bound = 1024 } } },
    .{ .launch = .{ .name = "flashnext_mtp fn_row_best", .symbol = "fn_row_best", .image = "flashnext_mtp", .threads = 1024 }, .usage = .{ .{ .regs = 17, .static = 392, .bound = 1024 }, .{ .regs = 17, .static = 392, .bound = 1024 }, .{ .regs = 17, .static = 392, .bound = 1024 }, .{ .regs = 21, .static = 392, .bound = 1024 }, .{ .regs = 21, .static = 392, .bound = 1024 } } },
    .{ .launch = .{ .name = "flashnext_hc fn_shared_scatter", .symbol = "fn_shared_scatter", .image = "flashnext_hc", .threads = 256 }, .usage = .{ .{ .regs = 12, .static = 0, .bound = 0 }, .{ .regs = 12, .static = 0, .bound = 0 }, .{ .regs = 12, .static = 0, .bound = 0 }, .{ .regs = 14, .static = 0, .bound = 0 }, .{ .regs = 14, .static = 0, .bound = 0 } } },
    .{ .launch = .{ .name = "flashnext_hc fn_shared_swiglu", .symbol = "fn_shared_swiglu", .image = "flashnext_hc", .threads = 256 }, .usage = .{ .{ .regs = 17, .static = 0, .bound = 0 }, .{ .regs = 18, .static = 0, .bound = 0 }, .{ .regs = 18, .static = 0, .bound = 0 }, .{ .regs = 18, .static = 0, .bound = 0 }, .{ .regs = 18, .static = 0, .bound = 0 } } },
    .{ .launch = .{ .name = "flashnext_batch fn_shift_window", .symbol = "fn_shift_window", .image = "flashnext_batch", .threads = 256 }, .usage = .{ .{ .regs = 32, .static = 0, .bound = 0 }, .{ .regs = 40, .static = 0, .bound = 0 }, .{ .regs = 40, .static = 0, .bound = 0 }, .{ .regs = 40, .static = 0, .bound = 0 }, .{ .regs = 40, .static = 0, .bound = 0 } } },
    .{ .launch = .{ .name = "flashnext_batch fn_sum_slices", .symbol = "fn_sum_slices", .image = "flashnext_batch", .threads = 256 }, .usage = .{ .{ .regs = 32, .static = 0, .bound = 0 }, .{ .regs = 38, .static = 0, .bound = 0 }, .{ .regs = 38, .static = 0, .bound = 0 }, .{ .regs = 40, .static = 0, .bound = 0 }, .{ .regs = 40, .static = 0, .bound = 0 } } },
    .{ .launch = .{ .name = "embed_rows tf_embed_rows_bf16_copies", .symbol = "tf_embed_rows_bf16_copies", .image = "embed_rows", .threads = 256 }, .usage = .{ .{ .regs = 32, .static = 0, .bound = 256 }, .{ .regs = 40, .static = 0, .bound = 256 }, .{ .regs = 40, .static = 0, .bound = 256 }, .{ .regs = 40, .static = 0, .bound = 256 }, .{ .regs = 40, .static = 0, .bound = 256 } } },
    .{ .launch = .{ .name = "norms tf_rms_rows_sums", .symbol = "tf_rms_rows_sums", .image = "norms", .threads = 1024 }, .usage = .{ .{ .regs = 31, .static = 132, .bound = 1024 }, .{ .regs = 37, .static = 132, .bound = 1024 }, .{ .regs = 37, .static = 132, .bound = 1024 }, .{ .regs = 34, .static = 132, .bound = 1024 }, .{ .regs = 34, .static = 132, .bound = 1024 } } },
    .{ .launch = .{ .name = "route_rows tf_route_logits", .symbol = "tf_route_logits", .image = "route_rows", .threads = 128, .dynamic = 26112 }, .usage = .{ .{ .regs = 31, .static = 0, .bound = 128 }, .{ .regs = 39, .static = 0, .bound = 128 }, .{ .regs = 39, .static = 0, .bound = 128 }, .{ .regs = 35, .static = 0, .bound = 128 }, .{ .regs = 34, .static = 0, .bound = 128 } } },
    .{ .launch = .{ .name = "route_rows tf_route_logits_few", .symbol = "tf_route_logits_few", .image = "route_rows", .threads = 128, .dynamic = 76032, .opt_in = true }, .usage = .{ .{ .regs = 31, .static = 0, .bound = 128 }, .{ .regs = 40, .static = 0, .bound = 128 }, .{ .regs = 40, .static = 0, .bound = 128 }, .{ .regs = 35, .static = 0, .bound = 128 }, .{ .regs = 37, .static = 0, .bound = 128 } } },
    .{ .launch = .{ .name = "route_rows tf_route_topk", .symbol = "tf_route_topk", .image = "route_rows", .threads = 128 }, .usage = .{ .{ .regs = 39, .static = 64, .bound = 128 }, .{ .regs = 39, .static = 64, .bound = 128 }, .{ .regs = 39, .static = 64, .bound = 128 }, .{ .regs = 36, .static = 64, .bound = 128 }, .{ .regs = 36, .static = 64, .bound = 128 } } },
    .{ .launch = .{ .name = "norms tf_rms_rows_bf16", .symbol = "tf_rms_rows_bf16", .image = "norms", .threads = 256 }, .usage = .{ .{ .regs = 79, .static = 32, .bound = 256 }, .{ .regs = 79, .static = 32, .bound = 256 }, .{ .regs = 79, .static = 32, .bound = 256 }, .{ .regs = 80, .static = 32, .bound = 256 }, .{ .regs = 80, .static = 32, .bound = 256 } } },
    .{ .launch = .{ .name = "norms tf_rms_rows", .symbol = "tf_rms_rows", .image = "norms", .threads = 1024 }, .usage = .{ .{ .regs = 32, .static = 132, .bound = 1024 }, .{ .regs = 46, .static = 132, .bound = 1024 }, .{ .regs = 46, .static = 132, .bound = 1024 }, .{ .regs = 43, .static = 132, .bound = 1024 }, .{ .regs = 43, .static = 132, .bound = 1024 } } },
    .{ .launch = .{ .name = "router tf_nemo_router", .symbol = "tf_nemo_router", .image = "nemotron_route", .threads = 256, .dynamic = 28800 }, .usage = .{ .{ .regs = 32, .static = 0, .bound = 256 }, .{ .regs = 40, .static = 0, .bound = 256 }, .{ .regs = 40, .static = 0, .bound = 256 }, .{ .regs = 40, .static = 0, .bound = 256 }, .{ .regs = 40, .static = 0, .bound = 256 } } },
    .{ .launch = .{ .name = "router tf_nemo_topk", .symbol = "tf_nemo_topk", .image = "nemotron_route", .threads = 32 }, .usage = .{ .{ .regs = 94, .static = 0, .bound = 32 }, .{ .regs = 95, .static = 0, .bound = 32 }, .{ .regs = 95, .static = 0, .bound = 32 }, .{ .regs = 91, .static = 0, .bound = 32 }, .{ .regs = 91, .static = 0, .bound = 32 } } },
    .{ .launch = .{ .name = "router tf_nemo_topk_wide", .symbol = "tf_nemo_topk_wide", .image = "nemotron_route", .threads = 32 }, .usage = .{ .{ .regs = 168, .static = 0, .bound = 32 }, .{ .regs = 168, .static = 0, .bound = 32 }, .{ .regs = 168, .static = 0, .bound = 32 }, .{ .regs = 168, .static = 0, .bound = 32 }, .{ .regs = 168, .static = 0, .bound = 32 } } },
    .{ .launch = .{ .name = "latent tf_super_topk_few", .symbol = "tf_super_topk_few", .image = "nemotron_latent", .threads = 1024 }, .usage = .{ .{ .regs = 32, .static = 4352, .bound = 1024 }, .{ .regs = 38, .static = 4352, .bound = 1024 }, .{ .regs = 38, .static = 4352, .bound = 1024 }, .{ .regs = 30, .static = 4352, .bound = 1024 }, .{ .regs = 30, .static = 4352, .bound = 1024 } } },
    .{ .launch = .{ .name = "qmm_group cluster of 8", .symbol = group_symbol, .image = "qmm_group", .threads = 128, .dynamic = 35328, .cluster = 8, .min_sm = 90 }, .usage = .{ .{ .regs = 54, .static = 0, .bound = 128 }, .{ .regs = 54, .static = 0, .bound = 128 }, .{ .regs = 54, .static = 0, .bound = 128 }, .{ .regs = 48, .static = 0, .bound = 128 }, .{ .regs = 48, .static = 0, .bound = 128 } } },
    .{ .launch = .{ .name = "qmm_group one CTA a tile", .symbol = group_symbol, .image = "qmm_group", .threads = 128, .dynamic = 35328 }, .usage = .{ .{ .regs = 54, .static = 0, .bound = 128 }, .{ .regs = 54, .static = 0, .bound = 128 }, .{ .regs = 54, .static = 0, .bound = 128 }, .{ .regs = 48, .static = 0, .bound = 128 }, .{ .regs = 48, .static = 0, .bound = 128 } } },
    .{ .launch = .{ .name = "lane_gemv", .symbol = gemv_symbol, .image = "lane_gemv", .threads = 128, .dynamic = 35328 }, .usage = .{ .{ .regs = 64, .static = 0, .bound = 128 }, .{ .regs = 58, .static = 0, .bound = 128 }, .{ .regs = 58, .static = 0, .bound = 128 }, .{ .regs = 66, .static = 0, .bound = 128 }, .{ .regs = 66, .static = 0, .bound = 128 } } },
    .{ .launch = .{ .name = "lane_gemv split-K", .symbol = "tf_lane_gemv_split", .image = "lane_gemv", .threads = 128, .dynamic = 17664 }, .usage = .{ .{ .regs = 62, .static = 128, .bound = 128 }, .{ .regs = 56, .static = 128, .bound = 128 }, .{ .regs = 56, .static = 128, .bound = 128 }, .{ .regs = 72, .static = 128, .bound = 128 }, .{ .regs = 72, .static = 128, .bound = 128 } } },
    .{ .launch = .{ .name = "prompt16 FP4", .symbol = std.fmt.comptimePrint(prompt_symbol, .{ 0, 0 }), .image = "prompt16", .threads = 128, .dynamic = 41984 }, .usage = .{ .{ .regs = 255, .static = 0, .bound = 128 }, .{ .regs = 255, .static = 0, .bound = 128 }, .{ .regs = 255, .static = 0, .bound = 128 }, .{ .regs = 212, .static = 0, .bound = 128 }, .{ .regs = 212, .static = 0, .bound = 128 } } },
    .{ .launch = .{ .name = "prompt16 FP8", .symbol = std.fmt.comptimePrint(prompt_symbol, .{ 1, 0 }), .image = "prompt16", .threads = 128, .dynamic = 49152 }, .usage = .{ .{ .regs = 243, .static = 0, .bound = 128 }, .{ .regs = 243, .static = 0, .bound = 128 }, .{ .regs = 243, .static = 0, .bound = 128 }, .{ .regs = 212, .static = 0, .bound = 128 }, .{ .regs = 212, .static = 0, .bound = 128 } } },
    .{ .launch = .{ .name = "prompt16 FP4 fp32 out", .symbol = std.fmt.comptimePrint(prompt_symbol, .{ 0, 1 }), .image = "prompt16", .threads = 128, .dynamic = 41984 }, .usage = .{ .{ .regs = 255, .static = 0, .bound = 128 }, .{ .regs = 255, .static = 0, .bound = 128 }, .{ .regs = 255, .static = 0, .bound = 128 }, .{ .regs = 212, .static = 0, .bound = 128 }, .{ .regs = 212, .static = 0, .bound = 128 } } },
    .{ .launch = .{ .name = "qmmf FP4 fp32 out 16-row tile", .symbol = std.fmt.comptimePrint(qmmf_f32_symbol, .{ 16, 0, 0 }), .image = "qmmf", .threads = 128, .dynamic = 17408 }, .usage = .{ .{ .regs = 88, .static = 0, .bound = 128 }, .{ .regs = 88, .static = 0, .bound = 128 }, .{ .regs = 96, .static = 0, .bound = 128 }, .{ .regs = 56, .static = 0, .bound = 128 }, .{ .regs = 56, .static = 0, .bound = 128 } } },
    .{ .launch = .{ .name = "qmmf FP4 fp32 out 16-row tile, cluster of 8", .symbol = std.fmt.comptimePrint(qmmf_f32_symbol, .{ 16, 1, 0 }), .image = "qmmf", .threads = 128, .dynamic = 17408, .cluster = 8, .min_sm = 90 }, .usage = .{ .{ .regs = 64, .static = 0, .bound = 128 }, .{ .regs = 64, .static = 0, .bound = 128 }, .{ .regs = 78, .static = 0, .bound = 128 }, .{ .regs = 48, .static = 0, .bound = 128 }, .{ .regs = 48, .static = 0, .bound = 128 } } },
    .{ .launch = .{ .name = "qmmf FP4 fp32 out 32-row tile", .symbol = std.fmt.comptimePrint(qmmf_f32_symbol, .{ 32, 0, 0 }), .image = "qmmf", .threads = 128, .dynamic = 25600 }, .usage = .{ .{ .regs = 106, .static = 0, .bound = 128 }, .{ .regs = 106, .static = 0, .bound = 128 }, .{ .regs = 128, .static = 0, .bound = 128 }, .{ .regs = 64, .static = 0, .bound = 128 }, .{ .regs = 64, .static = 0, .bound = 128 } } },
    .{ .launch = .{ .name = "qmmf FP4 fp32 out 32-row tile, cluster of 8", .symbol = std.fmt.comptimePrint(qmmf_f32_symbol, .{ 32, 1, 0 }), .image = "qmmf", .threads = 128, .dynamic = 25600, .cluster = 8, .min_sm = 90 }, .usage = .{ .{ .regs = 64, .static = 0, .bound = 128 }, .{ .regs = 64, .static = 0, .bound = 128 }, .{ .regs = 80, .static = 0, .bound = 128 }, .{ .regs = 61, .static = 0, .bound = 128 }, .{ .regs = 61, .static = 0, .bound = 128 } } },
    .{ .launch = .{ .name = "qmmf FP4 fp32 out 64-row tile", .symbol = std.fmt.comptimePrint(qmmf_f32_symbol, .{ 64, 0, 0 }), .image = "qmmf", .threads = 128, .dynamic = 41984 }, .usage = .{ .{ .regs = 148, .static = 0, .bound = 128 }, .{ .regs = 148, .static = 0, .bound = 128 }, .{ .regs = 180, .static = 0, .bound = 128 }, .{ .regs = 78, .static = 0, .bound = 128 }, .{ .regs = 78, .static = 0, .bound = 128 } } },
    .{ .launch = .{ .name = "qmmf FP4 fp32 out 64-row tile, cluster of 8", .symbol = std.fmt.comptimePrint(qmmf_f32_symbol, .{ 64, 1, 0 }), .image = "qmmf", .threads = 128, .dynamic = 41984, .cluster = 8, .min_sm = 90 }, .usage = .{ .{ .regs = 64, .static = 0, .bound = 128 }, .{ .regs = 64, .static = 0, .bound = 128 }, .{ .regs = 72, .static = 0, .bound = 128 }, .{ .regs = 78, .static = 0, .bound = 128 }, .{ .regs = 78, .static = 0, .bound = 128 } } },
    .{ .launch = .{ .name = "qmmf FP4 fp32 out fused slices", .symbol = std.fmt.comptimePrint(qmmf_f32_symbol, .{ 64, 0, 1 }), .image = "qmmf", .threads = 128, .dynamic = 41984 }, .usage = .{ .{ .regs = 182, .static = 0, .bound = 128 }, .{ .regs = 182, .static = 0, .bound = 128 }, .{ .regs = 228, .static = 0, .bound = 128 }, .{ .regs = 168, .static = 0, .bound = 128 }, .{ .regs = 168, .static = 0, .bound = 128 } } },
    .{ .launch = .{ .name = "qmmf MXFP8 16-row tile", .symbol = "_ZN7tf_qmmf11qmmf_kernelILi2ELi16ELi64ELi1ELi4ELi4ELb0ELb0ELb0EEEvPK13__nv_bfloat16PKhS5_fPvPfiiiiiii", .image = "qmmf", .threads = 128, .dynamic = 25088 }, .usage = .{ .{ .regs = 94, .static = 0, .bound = 128 }, .{ .regs = 94, .static = 0, .bound = 128 }, .{ .regs = 94, .static = 0, .bound = 128 }, .{ .regs = 60, .static = 0, .bound = 128 }, .{ .regs = 60, .static = 0, .bound = 128 } } },
    .{ .launch = .{ .name = "qmmf MXFP8 16-row tile, cluster of 8", .symbol = "_ZN7tf_qmmf11qmmf_kernelILi2ELi16ELi64ELi1ELi4ELi4ELb0ELb1ELb0EEEvPK13__nv_bfloat16PKhS5_fPvPfiiiiiii", .image = "qmmf", .threads = 128, .dynamic = 25088, .cluster = 8, .min_sm = 90 }, .usage = .{ .{ .regs = 69, .static = 0, .bound = 128 }, .{ .regs = 69, .static = 0, .bound = 128 }, .{ .regs = 69, .static = 0, .bound = 128 }, .{ .regs = 56, .static = 0, .bound = 128 }, .{ .regs = 56, .static = 0, .bound = 128 } } },
    .{ .launch = .{ .name = "qmmf MXFP8 32-row tile", .symbol = "_ZN7tf_qmmf11qmmf_kernelILi2ELi32ELi64ELi1ELi4ELi4ELb0ELb0ELb0EEEvPK13__nv_bfloat16PKhS5_fPvPfiiiiiii", .image = "qmmf", .threads = 128, .dynamic = 33280 }, .usage = .{ .{ .regs = 128, .static = 0, .bound = 128 }, .{ .regs = 128, .static = 0, .bound = 128 }, .{ .regs = 128, .static = 0, .bound = 128 }, .{ .regs = 57, .static = 0, .bound = 128 }, .{ .regs = 57, .static = 0, .bound = 128 } } },
    .{ .launch = .{ .name = "qmmf MXFP8 32-row tile, cluster of 8", .symbol = "_ZN7tf_qmmf11qmmf_kernelILi2ELi32ELi64ELi1ELi4ELi4ELb0ELb1ELb0EEEvPK13__nv_bfloat16PKhS5_fPvPfiiiiiii", .image = "qmmf", .threads = 128, .dynamic = 33280, .cluster = 8, .min_sm = 90 }, .usage = .{ .{ .regs = 80, .static = 0, .bound = 128 }, .{ .regs = 80, .static = 0, .bound = 128 }, .{ .regs = 80, .static = 0, .bound = 128 }, .{ .regs = 66, .static = 0, .bound = 128 }, .{ .regs = 66, .static = 0, .bound = 128 } } },
    .{ .launch = .{ .name = "qmmf MXFP8 64-row tile", .symbol = "_ZN7tf_qmmf11qmmf_kernelILi2ELi64ELi64ELi1ELi4ELi4ELb0ELb0ELb0EEEvPK13__nv_bfloat16PKhS5_fPvPfiiiiiii", .image = "qmmf", .threads = 128, .dynamic = 49664, .opt_in = true }, .usage = .{ .{ .regs = 168, .static = 0, .bound = 128 }, .{ .regs = 168, .static = 0, .bound = 128 }, .{ .regs = 168, .static = 0, .bound = 128 }, .{ .regs = 108, .static = 0, .bound = 128 }, .{ .regs = 108, .static = 0, .bound = 128 } } },
    .{ .launch = .{ .name = "qmmf MXFP8 64-row tile, cluster of 8", .symbol = "_ZN7tf_qmmf11qmmf_kernelILi2ELi64ELi64ELi1ELi4ELi4ELb0ELb1ELb0EEEvPK13__nv_bfloat16PKhS5_fPvPfiiiiiii", .image = "qmmf", .threads = 128, .dynamic = 49664, .opt_in = true, .cluster = 8, .min_sm = 90 }, .usage = .{ .{ .regs = 72, .static = 0, .bound = 128 }, .{ .regs = 72, .static = 0, .bound = 128 }, .{ .regs = 72, .static = 0, .bound = 128 }, .{ .regs = 103, .static = 0, .bound = 128 }, .{ .regs = 103, .static = 0, .bound = 128 } } },
    .{ .launch = .{ .name = "qmmf MXFP8 64-row tile, fused slices", .symbol = "_ZN7tf_qmmf11qmmf_kernelILi2ELi64ELi64ELi1ELi4ELi4ELb0ELb0ELb1EEEvPK13__nv_bfloat16PKhS5_fPvPfiiiiiii", .image = "qmmf", .threads = 128, .dynamic = 49664, .opt_in = true }, .usage = .{ .{ .regs = 229, .static = 0, .bound = 128 }, .{ .regs = 229, .static = 0, .bound = 128 }, .{ .regs = 229, .static = 0, .bound = 128 }, .{ .regs = 159, .static = 0, .bound = 128 }, .{ .regs = 159, .static = 0, .bound = 128 } } },
    .{ .launch = .{ .name = "qmmf reduce_kernel<false>", .symbol = "_ZN7tf_qmmf13reduce_kernelILb0EEEvPKfPvxif", .image = "qmmf", .threads = 256 }, .usage = .{ .{ .regs = 32, .static = 0, .bound = 0 }, .{ .regs = 36, .static = 0, .bound = 0 }, .{ .regs = 36, .static = 0, .bound = 0 }, .{ .regs = 40, .static = 0, .bound = 0 }, .{ .regs = 40, .static = 0, .bound = 0 } } },
    .{ .launch = .{ .name = "qmmf FP8 fp32 out 16-row tile", .symbol = "_ZN7tf_qmmf11qmmf_kernelILi1ELi16ELi64ELi1ELi4ELi4ELb1ELb0ELb0EEEvPK13__nv_bfloat16PKhS5_fPvPfiiiiiii", .image = "qmmf", .threads = 128, .dynamic = 24576 }, .usage = .{ .{ .regs = 92, .static = 0, .bound = 128 }, .{ .regs = 92, .static = 0, .bound = 128 }, .{ .regs = 92, .static = 0, .bound = 128 }, .{ .regs = 50, .static = 0, .bound = 128 }, .{ .regs = 50, .static = 0, .bound = 128 } } },
    .{ .launch = .{ .name = "qmmf FP8 fp32 out 16-row tile, cluster of 8", .symbol = "_ZN7tf_qmmf11qmmf_kernelILi1ELi16ELi64ELi1ELi4ELi4ELb1ELb1ELb0EEEvPK13__nv_bfloat16PKhS5_fPvPfiiiiiii", .image = "qmmf", .threads = 128, .dynamic = 24576, .cluster = 8, .min_sm = 90 }, .usage = .{ .{ .regs = 64, .static = 0, .bound = 128 }, .{ .regs = 64, .static = 0, .bound = 128 }, .{ .regs = 64, .static = 0, .bound = 128 }, .{ .regs = 40, .static = 0, .bound = 128 }, .{ .regs = 40, .static = 0, .bound = 128 } } },
    .{ .launch = .{ .name = "qmmf FP8 fp32 out 32-row tile", .symbol = "_ZN7tf_qmmf11qmmf_kernelILi1ELi32ELi64ELi1ELi4ELi4ELb1ELb0ELb0EEEvPK13__nv_bfloat16PKhS5_fPvPfiiiiiii", .image = "qmmf", .threads = 128, .dynamic = 32768 }, .usage = .{ .{ .regs = 92, .static = 0, .bound = 128 }, .{ .regs = 92, .static = 0, .bound = 128 }, .{ .regs = 92, .static = 0, .bound = 128 }, .{ .regs = 56, .static = 0, .bound = 128 }, .{ .regs = 56, .static = 0, .bound = 128 } } },
    .{ .launch = .{ .name = "qmmf FP8 fp32 out 32-row tile, cluster of 8", .symbol = "_ZN7tf_qmmf11qmmf_kernelILi1ELi32ELi64ELi1ELi4ELi4ELb1ELb1ELb0EEEvPK13__nv_bfloat16PKhS5_fPvPfiiiiiii", .image = "qmmf", .threads = 128, .dynamic = 32768, .cluster = 8, .min_sm = 90 }, .usage = .{ .{ .regs = 64, .static = 0, .bound = 128 }, .{ .regs = 64, .static = 0, .bound = 128 }, .{ .regs = 64, .static = 0, .bound = 128 }, .{ .regs = 48, .static = 0, .bound = 128 }, .{ .regs = 48, .static = 0, .bound = 128 } } },
    .{ .launch = .{ .name = "qmmf FP8 fp32 out 64-row tile", .symbol = "_ZN7tf_qmmf11qmmf_kernelILi1ELi64ELi64ELi1ELi4ELi4ELb1ELb0ELb0EEEvPK13__nv_bfloat16PKhS5_fPvPfiiiiiii", .image = "qmmf", .threads = 128, .dynamic = 49152 }, .usage = .{ .{ .regs = 126, .static = 0, .bound = 128 }, .{ .regs = 126, .static = 0, .bound = 128 }, .{ .regs = 126, .static = 0, .bound = 128 }, .{ .regs = 72, .static = 0, .bound = 128 }, .{ .regs = 72, .static = 0, .bound = 128 } } },
    .{ .launch = .{ .name = "qmmf FP8 fp32 out 64-row tile, cluster of 8", .symbol = "_ZN7tf_qmmf11qmmf_kernelILi1ELi64ELi64ELi1ELi4ELi4ELb1ELb1ELb0EEEvPK13__nv_bfloat16PKhS5_fPvPfiiiiiii", .image = "qmmf", .threads = 128, .dynamic = 49152, .cluster = 8, .min_sm = 90 }, .usage = .{ .{ .regs = 64, .static = 0, .bound = 128 }, .{ .regs = 64, .static = 0, .bound = 128 }, .{ .regs = 64, .static = 0, .bound = 128 }, .{ .regs = 64, .static = 0, .bound = 128 }, .{ .regs = 64, .static = 0, .bound = 128 } } },
    .{ .launch = .{ .name = "qmmf FP8 fp32 out 64-row tile, fused slices", .symbol = "_ZN7tf_qmmf11qmmf_kernelILi1ELi64ELi64ELi1ELi4ELi4ELb1ELb0ELb1EEEvPK13__nv_bfloat16PKhS5_fPvPfiiiiiii", .image = "qmmf", .threads = 128, .dynamic = 49152 }, .usage = .{ .{ .regs = 179, .static = 0, .bound = 128 }, .{ .regs = 179, .static = 0, .bound = 128 }, .{ .regs = 179, .static = 0, .bound = 128 }, .{ .regs = 141, .static = 0, .bound = 128 }, .{ .regs = 141, .static = 0, .bound = 128 } } },
    .{ .launch = .{ .name = "qmmf MXFP8 fp32 out 16-row tile", .symbol = "_ZN7tf_qmmf11qmmf_kernelILi2ELi16ELi64ELi1ELi4ELi4ELb1ELb0ELb0EEEvPK13__nv_bfloat16PKhS5_fPvPfiiiiiii", .image = "qmmf", .threads = 128, .dynamic = 25088 }, .usage = .{ .{ .regs = 92, .static = 0, .bound = 128 }, .{ .regs = 92, .static = 0, .bound = 128 }, .{ .regs = 92, .static = 0, .bound = 128 }, .{ .regs = 60, .static = 0, .bound = 128 }, .{ .regs = 60, .static = 0, .bound = 128 } } },
    .{ .launch = .{ .name = "qmmf MXFP8 fp32 out 16-row tile, cluster of 8", .symbol = "_ZN7tf_qmmf11qmmf_kernelILi2ELi16ELi64ELi1ELi4ELi4ELb1ELb1ELb0EEEvPK13__nv_bfloat16PKhS5_fPvPfiiiiiii", .image = "qmmf", .threads = 128, .dynamic = 25088, .cluster = 8, .min_sm = 90 }, .usage = .{ .{ .regs = 69, .static = 0, .bound = 128 }, .{ .regs = 69, .static = 0, .bound = 128 }, .{ .regs = 69, .static = 0, .bound = 128 }, .{ .regs = 56, .static = 0, .bound = 128 }, .{ .regs = 56, .static = 0, .bound = 128 } } },
    .{ .launch = .{ .name = "qmmf MXFP8 fp32 out 32-row tile", .symbol = "_ZN7tf_qmmf11qmmf_kernelILi2ELi32ELi64ELi1ELi4ELi4ELb1ELb0ELb0EEEvPK13__nv_bfloat16PKhS5_fPvPfiiiiiii", .image = "qmmf", .threads = 128, .dynamic = 33280 }, .usage = .{ .{ .regs = 128, .static = 0, .bound = 128 }, .{ .regs = 128, .static = 0, .bound = 128 }, .{ .regs = 128, .static = 0, .bound = 128 }, .{ .regs = 57, .static = 0, .bound = 128 }, .{ .regs = 57, .static = 0, .bound = 128 } } },
    .{ .launch = .{ .name = "qmmf MXFP8 fp32 out 32-row tile, cluster of 8", .symbol = "_ZN7tf_qmmf11qmmf_kernelILi2ELi32ELi64ELi1ELi4ELi4ELb1ELb1ELb0EEEvPK13__nv_bfloat16PKhS5_fPvPfiiiiiii", .image = "qmmf", .threads = 128, .dynamic = 33280, .cluster = 8, .min_sm = 90 }, .usage = .{ .{ .regs = 78, .static = 0, .bound = 128 }, .{ .regs = 78, .static = 0, .bound = 128 }, .{ .regs = 78, .static = 0, .bound = 128 }, .{ .regs = 66, .static = 0, .bound = 128 }, .{ .regs = 66, .static = 0, .bound = 128 } } },
    .{ .launch = .{ .name = "qmmf MXFP8 fp32 out 64-row tile", .symbol = "_ZN7tf_qmmf11qmmf_kernelILi2ELi64ELi64ELi1ELi4ELi4ELb1ELb0ELb0EEEvPK13__nv_bfloat16PKhS5_fPvPfiiiiiii", .image = "qmmf", .threads = 128, .dynamic = 49664, .opt_in = true }, .usage = .{ .{ .regs = 168, .static = 0, .bound = 128 }, .{ .regs = 168, .static = 0, .bound = 128 }, .{ .regs = 168, .static = 0, .bound = 128 }, .{ .regs = 108, .static = 0, .bound = 128 }, .{ .regs = 108, .static = 0, .bound = 128 } } },
    .{ .launch = .{ .name = "qmmf MXFP8 fp32 out 64-row tile, cluster of 8", .symbol = "_ZN7tf_qmmf11qmmf_kernelILi2ELi64ELi64ELi1ELi4ELi4ELb1ELb1ELb0EEEvPK13__nv_bfloat16PKhS5_fPvPfiiiiiii", .image = "qmmf", .threads = 128, .dynamic = 49664, .opt_in = true, .cluster = 8, .min_sm = 90 }, .usage = .{ .{ .regs = 72, .static = 0, .bound = 128 }, .{ .regs = 72, .static = 0, .bound = 128 }, .{ .regs = 72, .static = 0, .bound = 128 }, .{ .regs = 103, .static = 0, .bound = 128 }, .{ .regs = 103, .static = 0, .bound = 128 } } },
    .{ .launch = .{ .name = "qmmf MXFP8 fp32 out 64-row tile, fused slices", .symbol = "_ZN7tf_qmmf11qmmf_kernelILi2ELi64ELi64ELi1ELi4ELi4ELb1ELb0ELb1EEEvPK13__nv_bfloat16PKhS5_fPvPfiiiiiii", .image = "qmmf", .threads = 128, .dynamic = 49664, .opt_in = true }, .usage = .{ .{ .regs = 229, .static = 0, .bound = 128 }, .{ .regs = 229, .static = 0, .bound = 128 }, .{ .regs = 229, .static = 0, .bound = 128 }, .{ .regs = 159, .static = 0, .bound = 128 }, .{ .regs = 159, .static = 0, .bound = 128 } } },
    .{ .launch = .{ .name = "qmmf FP8G fp32 out 16-row tile", .symbol = "_ZN7tf_qmmf11qmmf_kernelILi3ELi16ELi64ELi1ELi4ELi4ELb1ELb0ELb0EEEvPK13__nv_bfloat16PKhS5_fPvPfiiiiiii", .image = "qmmf", .threads = 128, .dynamic = 25600 }, .usage = .{ .{ .regs = 92, .static = 0, .bound = 128 }, .{ .regs = 92, .static = 0, .bound = 128 }, .{ .regs = 92, .static = 0, .bound = 128 }, .{ .regs = 58, .static = 0, .bound = 128 }, .{ .regs = 58, .static = 0, .bound = 128 } } },
    .{ .launch = .{ .name = "qmmf FP8G fp32 out 16-row tile, cluster of 8", .symbol = "_ZN7tf_qmmf11qmmf_kernelILi3ELi16ELi64ELi1ELi4ELi4ELb1ELb1ELb0EEEvPK13__nv_bfloat16PKhS5_fPvPfiiiiiii", .image = "qmmf", .threads = 128, .dynamic = 25600, .cluster = 8, .min_sm = 90 }, .usage = .{ .{ .regs = 64, .static = 0, .bound = 128 }, .{ .regs = 64, .static = 0, .bound = 128 }, .{ .regs = 64, .static = 0, .bound = 128 }, .{ .regs = 56, .static = 0, .bound = 128 }, .{ .regs = 56, .static = 0, .bound = 128 } } },
    .{ .launch = .{ .name = "qmmf FP8G fp32 out 32-row tile", .symbol = "_ZN7tf_qmmf11qmmf_kernelILi3ELi32ELi64ELi1ELi4ELi4ELb1ELb0ELb0EEEvPK13__nv_bfloat16PKhS5_fPvPfiiiiiii", .image = "qmmf", .threads = 128, .dynamic = 33792 }, .usage = .{ .{ .regs = 116, .static = 0, .bound = 128 }, .{ .regs = 116, .static = 0, .bound = 128 }, .{ .regs = 116, .static = 0, .bound = 128 }, .{ .regs = 56, .static = 0, .bound = 128 }, .{ .regs = 56, .static = 0, .bound = 128 } } },
    .{ .launch = .{ .name = "qmmf FP8G fp32 out 32-row tile, cluster of 8", .symbol = "_ZN7tf_qmmf11qmmf_kernelILi3ELi32ELi64ELi1ELi4ELi4ELb1ELb1ELb0EEEvPK13__nv_bfloat16PKhS5_fPvPfiiiiiii", .image = "qmmf", .threads = 128, .dynamic = 33792, .cluster = 8, .min_sm = 90 }, .usage = .{ .{ .regs = 72, .static = 0, .bound = 128 }, .{ .regs = 72, .static = 0, .bound = 128 }, .{ .regs = 72, .static = 0, .bound = 128 }, .{ .regs = 64, .static = 0, .bound = 128 }, .{ .regs = 64, .static = 0, .bound = 128 } } },
    .{ .launch = .{ .name = "qmmf FP8G fp32 out 64-row tile", .symbol = "_ZN7tf_qmmf11qmmf_kernelILi3ELi64ELi64ELi1ELi4ELi4ELb1ELb0ELb0EEEvPK13__nv_bfloat16PKhS5_fPvPfiiiiiii", .image = "qmmf", .threads = 128, .dynamic = 50176, .opt_in = true }, .usage = .{ .{ .regs = 164, .static = 0, .bound = 128 }, .{ .regs = 164, .static = 0, .bound = 128 }, .{ .regs = 164, .static = 0, .bound = 128 }, .{ .regs = 108, .static = 0, .bound = 128 }, .{ .regs = 108, .static = 0, .bound = 128 } } },
    .{ .launch = .{ .name = "qmmf FP8G fp32 out 64-row tile, cluster of 8", .symbol = "_ZN7tf_qmmf11qmmf_kernelILi3ELi64ELi64ELi1ELi4ELi4ELb1ELb1ELb0EEEvPK13__nv_bfloat16PKhS5_fPvPfiiiiiii", .image = "qmmf", .threads = 128, .dynamic = 50176, .opt_in = true, .cluster = 8, .min_sm = 90 }, .usage = .{ .{ .regs = 78, .static = 0, .bound = 128 }, .{ .regs = 78, .static = 0, .bound = 128 }, .{ .regs = 78, .static = 0, .bound = 128 }, .{ .regs = 122, .static = 0, .bound = 128 }, .{ .regs = 122, .static = 0, .bound = 128 } } },
    .{ .launch = .{ .name = "qmmf FP8G fp32 out 64-row tile, fused slices", .symbol = "_ZN7tf_qmmf11qmmf_kernelILi3ELi64ELi64ELi1ELi4ELi4ELb1ELb0ELb1EEEvPK13__nv_bfloat16PKhS5_fPvPfiiiiiii", .image = "qmmf", .threads = 128, .dynamic = 50176, .opt_in = true }, .usage = .{ .{ .regs = 196, .static = 0, .bound = 128 }, .{ .regs = 196, .static = 0, .bound = 128 }, .{ .regs = 196, .static = 0, .bound = 128 }, .{ .regs = 157, .static = 0, .bound = 128 }, .{ .regs = 157, .static = 0, .bound = 128 } } },
    .{ .launch = .{ .name = "qmmf BF16 fp32 out 16-row tile", .symbol = "_ZN7tf_qmmf11qmmf_kernelILi4ELi16ELi64ELi1ELi4ELi4ELb1ELb0ELb0EEEvPK13__nv_bfloat16PKhS5_fPvPfiiiiiii", .image = "qmmf", .threads = 128, .dynamic = 40960 }, .usage = .{ .{ .regs = 80, .static = 0, .bound = 128 }, .{ .regs = 80, .static = 0, .bound = 128 }, .{ .regs = 80, .static = 0, .bound = 128 }, .{ .regs = 48, .static = 0, .bound = 128 }, .{ .regs = 48, .static = 0, .bound = 128 } } },
    .{ .launch = .{ .name = "qmmf BF16 fp32 out 16-row tile, cluster of 8", .symbol = "_ZN7tf_qmmf11qmmf_kernelILi4ELi16ELi64ELi1ELi4ELi4ELb1ELb1ELb0EEEvPK13__nv_bfloat16PKhS5_fPvPfiiiiiii", .image = "qmmf", .threads = 128, .dynamic = 40960, .cluster = 8, .min_sm = 90 }, .usage = .{ .{ .regs = 64, .static = 0, .bound = 128 }, .{ .regs = 64, .static = 0, .bound = 128 }, .{ .regs = 64, .static = 0, .bound = 128 }, .{ .regs = 40, .static = 0, .bound = 128 }, .{ .regs = 40, .static = 0, .bound = 128 } } },
    .{ .launch = .{ .name = "qmmf BF16 fp32 out 32-row tile", .symbol = "_ZN7tf_qmmf11qmmf_kernelILi4ELi32ELi64ELi1ELi4ELi4ELb1ELb0ELb0EEEvPK13__nv_bfloat16PKhS5_fPvPfiiiiiii", .image = "qmmf", .threads = 128, .dynamic = 49152 }, .usage = .{ .{ .regs = 93, .static = 0, .bound = 128 }, .{ .regs = 93, .static = 0, .bound = 128 }, .{ .regs = 93, .static = 0, .bound = 128 }, .{ .regs = 55, .static = 0, .bound = 128 }, .{ .regs = 55, .static = 0, .bound = 128 } } },
    .{ .launch = .{ .name = "qmmf BF16 fp32 out 32-row tile, cluster of 8", .symbol = "_ZN7tf_qmmf11qmmf_kernelILi4ELi32ELi64ELi1ELi4ELi4ELb1ELb1ELb0EEEvPK13__nv_bfloat16PKhS5_fPvPfiiiiiii", .image = "qmmf", .threads = 128, .dynamic = 49152, .cluster = 8, .min_sm = 90 }, .usage = .{ .{ .regs = 64, .static = 0, .bound = 128 }, .{ .regs = 64, .static = 0, .bound = 128 }, .{ .regs = 64, .static = 0, .bound = 128 }, .{ .regs = 48, .static = 0, .bound = 128 }, .{ .regs = 48, .static = 0, .bound = 128 } } },
    .{ .launch = .{ .name = "qmmf BF16 fp32 out 64-row tile", .symbol = "_ZN7tf_qmmf11qmmf_kernelILi4ELi64ELi64ELi1ELi4ELi4ELb1ELb0ELb0EEEvPK13__nv_bfloat16PKhS5_fPvPfiiiiiii", .image = "qmmf", .threads = 128, .dynamic = 65536, .opt_in = true }, .usage = .{ .{ .regs = 124, .static = 0, .bound = 128 }, .{ .regs = 124, .static = 0, .bound = 128 }, .{ .regs = 124, .static = 0, .bound = 128 }, .{ .regs = 72, .static = 0, .bound = 128 }, .{ .regs = 72, .static = 0, .bound = 128 } } },
    .{ .launch = .{ .name = "qmmf BF16 fp32 out 64-row tile, cluster of 8", .symbol = "_ZN7tf_qmmf11qmmf_kernelILi4ELi64ELi64ELi1ELi4ELi4ELb1ELb1ELb0EEEvPK13__nv_bfloat16PKhS5_fPvPfiiiiiii", .image = "qmmf", .threads = 128, .dynamic = 65536, .opt_in = true, .cluster = 8, .min_sm = 90 }, .usage = .{ .{ .regs = 64, .static = 0, .bound = 128 }, .{ .regs = 64, .static = 0, .bound = 128 }, .{ .regs = 64, .static = 0, .bound = 128 }, .{ .regs = 64, .static = 0, .bound = 128 }, .{ .regs = 64, .static = 0, .bound = 128 } } },
    .{ .launch = .{ .name = "qmmf BF16 fp32 out 64-row tile, fused slices", .symbol = "_ZN7tf_qmmf11qmmf_kernelILi4ELi64ELi64ELi1ELi4ELi4ELb1ELb0ELb1EEEvPK13__nv_bfloat16PKhS5_fPvPfiiiiiii", .image = "qmmf", .threads = 128, .dynamic = 65536, .opt_in = true }, .usage = .{ .{ .regs = 180, .static = 0, .bound = 128 }, .{ .regs = 180, .static = 0, .bound = 128 }, .{ .regs = 180, .static = 0, .bound = 128 }, .{ .regs = 143, .static = 0, .bound = 128 }, .{ .regs = 143, .static = 0, .bound = 128 } } },
    .{ .launch = .{ .name = "qmmf reduce_kernel<true>", .symbol = "_ZN7tf_qmmf13reduce_kernelILb1EEEvPKfPvxif", .image = "qmmf", .threads = 256 }, .usage = .{ .{ .regs = 32, .static = 0, .bound = 0 }, .{ .regs = 36, .static = 0, .bound = 0 }, .{ .regs = 36, .static = 0, .bound = 0 }, .{ .regs = 40, .static = 0, .bound = 0 }, .{ .regs = 40, .static = 0, .bound = 0 } } },
    .{ .launch = .{ .name = "ck tf_ck4_up_relu2_few", .symbol = "tf_ck4_up_relu2_few", .image = "ck_experts", .threads = 128, .min_sm = 120 }, .usage = .{ .{ .regs = 19, .static = 0, .bound = 128 }, .{ .regs = 19, .static = 0, .bound = 128 }, .{ .regs = 19, .static = 0, .bound = 128 }, .{ .regs = 125, .static = 0, .bound = 128 }, .{ .regs = 125, .static = 0, .bound = 128 } } },
    .{ .launch = .{ .name = "ck tf_ck4_down_f32_few", .symbol = "tf_ck4_down_f32_few", .image = "ck_experts", .threads = 128, .min_sm = 120 }, .usage = .{ .{ .regs = 24, .static = 0, .bound = 128 }, .{ .regs = 24, .static = 0, .bound = 128 }, .{ .regs = 24, .static = 0, .bound = 128 }, .{ .regs = 128, .static = 0, .bound = 128 }, .{ .regs = 128, .static = 0, .bound = 128 } } },
    .{ .launch = .{ .name = "mamba tf_nemo_scan", .symbol = "tf_nemo_scan", .image = "nemotron_mamba", .threads = 128 }, .usage = .{ .{ .regs = 138, .static = 41088, .bound = 128 }, .{ .regs = 144, .static = 41088, .bound = 128 }, .{ .regs = 144, .static = 41088, .bound = 128 }, .{ .regs = 116, .static = 41088, .bound = 128 }, .{ .regs = 116, .static = 41088, .bound = 128 } } },
    .{ .launch = .{ .name = "rank_sum tf_rank_sum", .symbol = "tf_rank_sum", .image = "rank_sum", .threads = 256 }, .usage = .{ .{ .regs = 32, .static = 0, .bound = 256 }, .{ .regs = 40, .static = 0, .bound = 256 }, .{ .regs = 40, .static = 0, .bound = 256 }, .{ .regs = 40, .static = 0, .bound = 256 }, .{ .regs = 40, .static = 0, .bound = 256 } } },
    .{ .launch = .{ .name = "vit vision_kernel<80, 8> (RADIO)", .symbol = vision80_symbol, .image = "prefill_attention", .threads = 256, .dynamic = 45056 }, .usage = .{ .{ .regs = 143, .static = 0, .bound = 256 }, .{ .regs = 143, .static = 0, .bound = 256 }, .{ .regs = 143, .static = 0, .bound = 256 }, .{ .regs = 122, .static = 0, .bound = 256 }, .{ .regs = 122, .static = 0, .bound = 256 } } },
    .{ .launch = .{ .name = "vit full_kernel<72> (SigLIP2)", .symbol = full72_symbol, .image = "prefill_attention", .threads = 256, .dynamic = 65536, .opt_in = true }, .usage = .{ .{ .regs = 158, .static = 0, .bound = 256 }, .{ .regs = 158, .static = 0, .bound = 256 }, .{ .regs = 158, .static = 0, .bound = 256 }, .{ .regs = 169, .static = 0, .bound = 256 }, .{ .regs = 169, .static = 0, .bound = 256 } } },
    .{ .launch = .{ .name = "vit tf_vit_layer_norm", .symbol = "tf_vit_layer_norm", .image = "vit", .threads = 256 }, .usage = .{ .{ .regs = 38, .static = 32, .bound = 256 }, .{ .regs = 38, .static = 32, .bound = 256 }, .{ .regs = 38, .static = 32, .bound = 256 }, .{ .regs = 40, .static = 32, .bound = 256 }, .{ .regs = 40, .static = 32, .bound = 256 } } },
    .{ .launch = .{ .name = "vit tf_vit_pointwise", .symbol = "tf_vit_pointwise", .image = "vit", .threads = 256 }, .usage = .{ .{ .regs = 14, .static = 0, .bound = 256 }, .{ .regs = 14, .static = 0, .bound = 256 }, .{ .regs = 14, .static = 0, .bound = 256 }, .{ .regs = 14, .static = 0, .bound = 256 }, .{ .regs = 14, .static = 0, .bound = 256 } } },
    .{ .launch = .{ .name = "vit tf_vit_bias", .symbol = "tf_vit_bias", .image = "vit", .threads = 256 }, .usage = .{ .{ .regs = 12, .static = 0, .bound = 256 }, .{ .regs = 12, .static = 0, .bound = 256 }, .{ .regs = 12, .static = 0, .bound = 256 }, .{ .regs = 12, .static = 0, .bound = 256 }, .{ .regs = 12, .static = 0, .bound = 256 } } },
    .{ .launch = .{ .name = "vit tf_vit_unshuffle", .symbol = "tf_vit_unshuffle", .image = "vit", .threads = 256 }, .usage = .{ .{ .regs = 29, .static = 0, .bound = 256 }, .{ .regs = 32, .static = 0, .bound = 256 }, .{ .regs = 32, .static = 0, .bound = 256 }, .{ .regs = 34, .static = 0, .bound = 256 }, .{ .regs = 34, .static = 0, .bound = 256 } } },

pub fn newer(x: limits.Nvcc, y: limits.Nvcc) bool {
    return x.major > y.major or (x.major == y.major and x.minor > y.minor);
}

/// The release whose records a compiler's images are checked against: the newest set at or below it, else cuda's.
pub fn setFor(release: ?limits.Nvcc) limits.Nvcc {
    var at = cuda_release;
    const rel = release orelse return at;
    for (cuda_sets) |s| {
        if (newer(s.release, rel)) break;
        at = s.release;
    }
    return at;
}

/// A record's usage on every SM for images built by `release` (null: an image that names no compiler, cuda's).
pub fn usageFor(r: CudaRecord, release: ?limits.Nvcc) [limits.sms.len]U {
    var u = r.usage;
    if (for (u) |v| {
        if (v.regs > 0) break false;
    } else true) return u;
    const rel = release orelse return u;
    for (cuda_sets) |s| {
        if (newer(s.release, rel)) break;
        for (s.usage) |o| if (std.mem.eql(u8, o.name, r.launch.name)) {
            u = o.usage;
        };
    }
    return u;
}

/// Gate 5's planted CUDA controls: a launch past its __launch_bounds__, one needing more registers than a block has.
pub const cuda_controls = [_]CudaRecord{
    .{ .launch = .{ .name = "control: past its bounds", .symbol = "tf_ctl_bounded", .image = "contract_controls", .threads = 512 }, .usage = .{ .{ .regs = 10, .static = 0, .bound = 256 }, .{ .regs = 10, .static = 0, .bound = 256 }, .{ .regs = 10, .static = 0, .bound = 256 }, .{ .regs = 10, .static = 0, .bound = 256 }, .{ .regs = 10, .static = 0, .bound = 256 } } },
    .{ .launch = .{ .name = "control: heavy at 1,024 threads", .symbol = "tf_ctl_heavy", .image = "contract_controls", .threads = 1024 }, .usage = .{ .{ .regs = 168, .static = 0, .bound = 0 }, .{ .regs = 166, .static = 0, .bound = 0 }, .{ .regs = 166, .static = 0, .bound = 0 }, .{ .regs = 166, .static = 0, .bound = 0 }, .{ .regs = 166, .static = 0, .bound = 0 } } },
};

/// Every CUDA record's problems on every SM we build for, printed; returns how many (a missing record is one).
pub fn checkCuda(a: std.mem.Allocator, list: []const CudaRecord, quiet: bool) !usize {
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(a);
    var n: usize = 0;
    for (list) |r| if (r.target_sm != 0 and limits.target(r.target_sm) == null) {
        try out.print(a, "{s}: unknown declared sm_{d}\n", .{ r.launch.name, r.target_sm });
        n += 1;
    };
    for (list) |r| for (limits.sms, r.usage) |t, u| {
        if (!r.applies(t.sm)) continue;
        if (u.regs == 0) {
            try out.print(a, "{s} on sm_{d}: no compiled usage recorded\n", .{ r.launch.name, t.sm });
            n += 1;
        } else n += try limits.cuda(r.launch, u, t, &out, a);
    };
    if (!quiet and out.items.len > 0) std.debug.print("{s}", .{out.items});
    return n;
}

/// Metal pipelines as the engine launches them; tg_memory is the most any chip reflects, m1 and m2 the compiled limits.
pub const metal = [_]M{
    .{ .name = "softmax tf_softmax_bf16", .threads = 1024, .tg_memory = 256, .m1 = 1024, .m2 = 1024 },
    .{ .name = "rms tf_rms_norm_bf16", .threads = 1024, .tg_memory = 144, .m1 = 1024, .m2 = 1024 },
    .{ .name = "top-k tf_bf16_topk", .threads = 256, .tg_memory = 4096, .m1 = 768, .m2 = 768 },
    .{ .name = "top-k tf_bf16_topk_max", .threads = 256, .tg_memory = 4112, .m1 = 1024, .m2 = 1024 },
    .{ .name = "top-k tf_bf16_topk_span", .threads = 256, .tg_memory = 4096, .m1 = 704, .m2 = 704 },
    .{ .name = "top-k tf_bf16_topk_merge", .threads = 256, .tg_memory = 4096, .m1 = 1024, .m2 = 1024 },
    .{ .name = "row projection q4 g64 tf_row_projection_q4_f32", .threads = 64, .tg_memory = 0, .m1 = 512, .m2 = 512 },
    .{ .name = "row projection q6 g32 tf_row_projection_q6_f32", .threads = 64, .tg_memory = 0, .m1 = 512, .m2 = 512 },
    .{ .name = "row projection q8 g64 tf_row_projection_q8_f32", .threads = 64, .tg_memory = 0, .m1 = 512, .m2 = 512 },
    .{ .name = "row projection q3 g32 tf_row_projection_q3_f32", .threads = 64, .tg_memory = 0, .m1 = 512, .m2 = 512 },
    .{ .name = "affine_mm q4 g64 tf_affine_row_sums", .threads = 64, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    .{ .name = "affine_mm q4 g64 tf_affine_mm", .threads = 128, .tg_memory = 9728, .offline = false },
    .{ .name = "affine_mm q8 g32 tf_affine_mm", .threads = 128, .tg_memory = 10240, .offline = false },
    .{ .name = "simd_qmm tf_sq_mma_16_4_1", .threads = 512, .tg_memory = 16384, .m1 = 768, .m2 = 768, .adapts = true },
    .{ .name = "simd_qmm tf_sq_mma_16_2_2", .threads = 512, .tg_memory = 16384, .m1 = 704, .m2 = 704, .adapts = true },
    .{ .name = "simd_qmm tf_sq_mma_8_4_1", .threads = 256, .tg_memory = 8192, .m1 = 768, .m2 = 768, .adapts = true },
    .{ .name = "simd_qmm tf_sq_mma_8_4_2", .threads = 256, .tg_memory = 16384, .m1 = 576, .m2 = 576, .adapts = true },
    .{ .name = "simd_qmm_bits tf_sqw_mma_q3_16_4_1", .threads = 512, .tg_memory = 16384, .adapts = true, .m1 = 512, .m2 = 512 },
    .{ .name = "simd_qmm_bits tf_sqw_mma_q6_16_4_1", .threads = 512, .tg_memory = 16384, .adapts = true, .m1 = 512, .m2 = 512 },
    .{ .name = "simd_qmm_bits tf_sqw_mma_q8_16_4_1", .threads = 512, .tg_memory = 16384, .adapts = true, .m1 = 224, .m2 = 224 },
    .{ .name = "lane projection q6 g32 sk1 tf_lane", .threads = 32, .tg_memory = 3072, .offline = false },
    .{ .name = "lane projection q6 g32 sk4 tf_lane", .threads = 128, .tg_memory = 10240, .offline = false },
    .{ .name = "tile_qmm q4 tf_tile_qmm_q4", .threads = 128, .tg_memory = 11008, .reserved = 128 },
    .{ .name = "tile_qmm grouped q4 tf_tile_qmm_grouped_q4", .threads = 128, .tg_memory = 11008, .reserved = 128 },
    .{ .name = "tile_qmm grouped glu q4 tf_tile_qmm_grouped_glu_q4", .threads = 128, .tg_memory = 11008, .reserved = 128 },
    .{ .name = "tile_qmm twin q4 tf_tile_qmv_q4", .threads = 64, .tg_memory = 0, .dynamic = 10240, .m1 = 448, .m2 = 448 },
    .{ .name = "tile_qmm twin indexed q4 tf_tile_qmv_indexed_q4", .threads = 64, .tg_memory = 0, .dynamic = 10240, .m1 = 576, .m2 = 576 },
    .{ .name = "tile_qmm twin glu q4 tf_tile_qmv_glu_q4", .threads = 64, .tg_memory = 0, .dynamic = 10240, .m1 = 576, .m2 = 576 },
    .{ .name = "tile_qmm twin few q4 tf_tile_qmv_few_q4", .threads = 64, .tg_memory = 16, .dynamic = 10240, .m1 = 576, .m2 = 576 },
    .{ .name = "tile_qmm window rows q4 tf_tile_qmv_rows_q4", .threads = 64, .tg_memory = 4096, .m1 = 640, .m2 = 640 },
    .{ .name = "tile_qmm twin by expert q4 tf_tile_qmv_grouped_q4", .threads = 64, .tg_memory = 32, .dynamic = 5120, .m1 = 512, .m2 = 512 },
    .{ .name = "tile_qmm twin by expert glu q4 tf_tile_qmv_grouped_glu_q4", .threads = 64, .tg_memory = 32, .dynamic = 5120, .m1 = 512, .m2 = 512 },
    .{ .name = "tile_qmm q6 tf_tile_qmm_q6", .threads = 128, .tg_memory = 11008, .reserved = 128 },
    .{ .name = "tile_qmm grouped q6 tf_tile_qmm_grouped_q6", .threads = 128, .tg_memory = 11008, .reserved = 128 },
    .{ .name = "tile_qmm grouped glu q6 tf_tile_qmm_grouped_glu_q6", .threads = 128, .tg_memory = 11008, .reserved = 128 },
    .{ .name = "tile_qmm twin q6 tf_tile_qmv_q6", .threads = 64, .tg_memory = 0, .dynamic = 10240, .m1 = 640, .m2 = 640 },
    .{ .name = "tile_qmm twin indexed q6 tf_tile_qmv_indexed_q6", .threads = 64, .tg_memory = 0, .dynamic = 10240, .m1 = 640, .m2 = 640 },
    .{ .name = "tile_qmm twin glu q6 tf_tile_qmv_glu_q6", .threads = 64, .tg_memory = 0, .dynamic = 10240, .m1 = 640, .m2 = 640 },
    .{ .name = "tile_qmm twin few q6 tf_tile_qmv_few_q6", .threads = 64, .tg_memory = 16, .dynamic = 10240, .m1 = 512, .m2 = 512 },
    .{ .name = "tile_qmm window rows q6 tf_tile_qmv_rows_q6", .threads = 64, .tg_memory = 4096, .m1 = 576, .m2 = 576 },
    .{ .name = "tile_qmm twin by expert q6 tf_tile_qmv_grouped_q6", .threads = 64, .tg_memory = 32, .dynamic = 5120, .m1 = 448, .m2 = 448 },
    .{ .name = "tile_qmm twin by expert glu q6 tf_tile_qmv_grouped_glu_q6", .threads = 64, .tg_memory = 32, .dynamic = 5120, .m1 = 384, .m2 = 384 },
    .{ .name = "tile_qmm q8 tf_tile_qmm_q8", .threads = 128, .tg_memory = 11008, .reserved = 128 },
    .{ .name = "tile_qmm grouped q8 tf_tile_qmm_grouped_q8", .threads = 128, .tg_memory = 11008, .reserved = 128 },
    .{ .name = "tile_qmm grouped glu q8 tf_tile_qmm_grouped_glu_q8", .threads = 128, .tg_memory = 11008, .reserved = 128 },
    .{ .name = "tile_qmm twin q8 tf_tile_qmv_q8", .threads = 64, .tg_memory = 0, .dynamic = 10240, .m1 = 896, .m2 = 896 },
    .{ .name = "tile_qmm twin indexed q8 tf_tile_qmv_indexed_q8", .threads = 64, .tg_memory = 0, .dynamic = 10240, .m1 = 896, .m2 = 896 },
    .{ .name = "tile_qmm twin glu q8 tf_tile_qmv_glu_q8", .threads = 64, .tg_memory = 0, .dynamic = 10240, .m1 = 832, .m2 = 832 },
    .{ .name = "tile_qmm twin few q8 tf_tile_qmv_few_q8", .threads = 64, .tg_memory = 16, .dynamic = 10240, .m1 = 512, .m2 = 512 },
    .{ .name = "tile_qmm window rows q8 tf_tile_qmv_rows_q8", .threads = 64, .tg_memory = 4096, .m1 = 576, .m2 = 576 },
    .{ .name = "tile_qmm twin by expert q8 tf_tile_qmv_grouped_q8", .threads = 64, .tg_memory = 32, .dynamic = 5120, .m1 = 448, .m2 = 448 },
    .{ .name = "tile_qmm twin by expert glu q8 tf_tile_qmv_grouped_glu_q8", .threads = 64, .tg_memory = 32, .dynamic = 5120, .m1 = 384, .m2 = 384 },
    .{ .name = "tile_qmm work list tf_tile_qmm_work", .threads = 1, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    .{ .name = "tile_qmm twin picks by expert tf_tile_qmv_group", .threads = 128, .tg_memory = 1024, .m1 = 1024, .m2 = 1024 },
    .{ .name = "tile_qmm group sums tf_tile_xsum", .threads = 256, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    .{ .name = "glm hc_expand_10 custom_kernel_tf_glm5_fused_hc_expand_cc77af98f2__4096_1_0_0_0_bfloat16_t_bfloat16_t_float_float_floatc_bfloat16_t_float_float", .threads = 1024, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    .{ .name = "glm gather glm_gather_glu", .threads = 64, .tg_memory = 0, .m1 = 512, .m2 = 512 },
    .{ .name = "glm gather glm_gather_down", .threads = 64, .tg_memory = 0, .m1 = 576, .m2 = 576 },
    .{ .name = "h3 rows in h3_rows_in", .threads = 1024, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    .{ .name = "h3 modulated norm h3_norm", .threads = 32, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    .{ .name = "h3 gated add h3_gate_add", .threads = 32, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    .{ .name = "h3 pooled branch mix h3_gate_mix", .threads = 256, .tg_memory = 0, .m1 = 928, .m2 = 928 },
    .{ .name = "h3 SwiGLU h3_swiglu", .threads = 256, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    .{ .name = "h3 final norm h3_final_norm", .threads = 32, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    .{ .name = "h3 rows out h3_rows_out", .threads = 32, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    .{ .name = "h3 head layout h3_heads", .threads = 32, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    .{ .name = "h3 tile means h3_pool", .threads = 32, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    .{ .name = "h3 tile scores h3_tile_scores", .threads = 256, .tg_memory = 0, .m1 = 928, .m2 = 928 },
    .{ .name = "h3 tile top-k h3_topk", .threads = 32, .tg_memory = 288, .m1 = 1024, .m2 = 1024 },
    .{ .name = "h3 pooled branch h3_coarse", .threads = 32, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    .{ .name = "h3 AdaLN tables h3_modulation", .threads = 256, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    .{ .name = "h3 routed attention h3_attention", .threads = 256, .tg_memory = 25856, .offline = false },
    .{ .name = "h3 gemm whole custom_kernel_tf_gemm_nax_bf16_n_t_t_t_n_64_128_256_2_4_bfloat16_t_bfloat16_t_int32_t_bfloat16_t", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "h3 gemm ragged custom_kernel_tf_gemm_nax_bf16_n_t_n_t_n_64_128_256_2_4_bfloat16_t_bfloat16_t_int32_t_bfloat16_t", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "qwen vision qwv_gemm_bf16_nt_00", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "qwen vision qwv_gemm_bf16_nt_01", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "qwen vision qwv_gemm_bf16_nt_10", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "qwen vision qwv_gemm_bf16_nt_11", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "qwen vision qwv_gemm_f32_nt_00", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "qwen vision qwv_gemm_f32_nt_01", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "qwen vision qwv_gemm_f32_nt_10", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "qwen vision qwv_gemm_f32_nt_11", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "qwen vision qwv_gemm_f32_nn_00", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "qwen vision qwv_gemm_f32_nn_01", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "qwen vision qwv_gemm_f32_nn_10", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "qwen vision qwv_gemm_f32_nn_11", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "qwen vision qwv_embed", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "qwen vision qwv_layer_norm", .threads = 256, .tg_memory = 144, .reserved = 256 },
    .{ .name = "qwen vision qwv_qkv", .threads = 216, .tg_memory = 0, .reserved = 256 },
    .{ .name = "qwen vision qwv_softmax", .threads = 256, .tg_memory = 144, .reserved = 256 },
    .{ .name = "qwen vision qwv_heads", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "qwen vision qwv_bias_gelu", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "qwen vision qwv_residual", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "qwen vision qwv_bias", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "qwen image qi_mod_norm", .threads = 256, .tg_memory = 144, .reserved = 256 },
    .{ .name = "qwen image qi_text_norm", .threads = 256, .tg_memory = 144, .reserved = 256 },
    .{ .name = "qwen image qi_gelu", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "qwen image qi_qk", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "qwen image qi_prefix", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "qwen image qi_gate_add", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "qwen image qi_swiglu", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "qwen image qi_attention_full_00", .threads = 128, .tg_memory = 0, .reserved = 128 },
    .{ .name = "qwen image qi_attention_full_01", .threads = 128, .tg_memory = 0, .reserved = 128 },
    .{ .name = "qwen image qi_attention_full_10", .threads = 128, .tg_memory = 0, .reserved = 128 },
    .{ .name = "qwen image qi_attention_full_11", .threads = 128, .tg_memory = 0, .reserved = 128 },
    .{ .name = "qwen image qi_attention_causal_00", .threads = 128, .tg_memory = 0, .reserved = 128 },
    .{ .name = "qwen image qi_attention_causal_01", .threads = 128, .tg_memory = 0, .reserved = 128 },
    .{ .name = "qwen image qi_attention_causal_10", .threads = 128, .tg_memory = 0, .reserved = 128 },
    .{ .name = "qwen image qi_attention_causal_11", .threads = 128, .tg_memory = 0, .reserved = 128 },
    .{ .name = "qwen27 copy q27_take", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "drama3 dense d3_rows_r1", .threads = 128, .tg_memory = 0, .reserved = 128 },
    .{ .name = "drama3 dense d3_rows_r8", .threads = 128, .tg_memory = 0, .reserved = 128 },
    .{ .name = "drama3 dense d3_rmsnorm", .threads = 256, .tg_memory = 128, .reserved = 256 },
    .{ .name = "drama3 dense d3_add_rmsnorm", .threads = 256, .tg_memory = 128, .reserved = 256 },
    .{ .name = "drama3 dense d3_embed_backbone", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "drama3 dense d3_swiglu", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "drama3 dense d3_moe_combine", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "drama3 dense d3_tower_mix", .threads = 256, .tg_memory = 8320, .reserved = 256 },
    .{ .name = "drama3 gdn d3_conv_silu", .threads = 128, .tg_memory = 0, .reserved = 128 },
    .{ .name = "drama3 gdn d3_gdn", .threads = 128, .tg_memory = 1536, .reserved = 128 },
    .{ .name = "drama3 gdn d3_gated_norm", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "drama3 attn d3_qk_rope", .threads = 256, .tg_memory = 1152, .reserved = 256 },
    .{ .name = "drama3 attn d3_attn", .threads = 256, .tg_memory = 8256, .reserved = 256 },
    .{ .name = "drama3 moe d3_router", .threads = 256, .tg_memory = 256, .reserved = 256 },
    .{ .name = "drama3 moe d3_act_quant", .threads = 128, .tg_memory = 128, .reserved = 128 },
    .{ .name = "drama3 moe d3_plan", .threads = 1024, .tg_memory = 1024, .reserved = 1024 },
    .{ .name = "drama3 moe d3_expert_up", .threads = 128, .tg_memory = 0, .reserved = 128 },
    .{ .name = "drama3 moe d3_expert_down", .threads = 128, .tg_memory = 0, .reserved = 128 },
    .{ .name = "drama3 sample d3_argmax", .threads = 1024, .tg_memory = 256, .reserved = 1024 },
    .{ .name = "drama3 sample d3_sample_main", .threads = 256, .tg_memory = 3632, .reserved = 256 },
    .{ .name = "drama3 sample d3_sample_code", .threads = 256, .tg_memory = 3632, .reserved = 256 },
    .{ .name = "drama3 sample d3_finish_frame", .threads = 32, .tg_memory = 0, .reserved = 32 },
    .{ .name = "drama3 sample d3_next_input", .threads = 32, .tg_memory = 0, .reserved = 32 },
    .{ .name = "drama3 sample d3_copy_i32", .threads = 64, .tg_memory = 0, .reserved = 64 },
    .{ .name = "drama3 codec c_gemm", .threads = 256, .tg_memory = 8192, .reserved = 256 },
    .{ .name = "drama3 codec c_vq_gather", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "drama3 codec c_vq_encode", .threads = 256, .tg_memory = 400, .reserved = 256 },
    .{ .name = "drama3 codec c_rmsnorm", .threads = 256, .tg_memory = 128, .reserved = 256 },
    .{ .name = "drama3 codec c_layernorm", .threads = 256, .tg_memory = 128, .reserved = 256 },
    .{ .name = "drama3 codec c_rope_kv", .threads = 512, .tg_memory = 0, .reserved = 512 },
    .{ .name = "drama3 codec c_attn", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "drama3 codec c_add_gamma", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "drama3 codec c_silu_mul", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "drama3 codec c_gelu", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "drama3 codec c_bias", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "drama3 codec c_snake", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "drama3 codec c_dwconv7", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "drama3 codec c_copy", .threads = 256, .tg_memory = 0, .reserved = 256 },
};

/// Gate 5's planted Metal controls: a launch past M1/M2's compiled limit (#708's class), 36 KiB of threadgroup memory.
pub const metal_controls = [_]M{
    .{ .name = "control: row projection q4 g64 at 1024 threads tf_row_projection_q4_f32", .threads = 1024, .tg_memory = 0, .m1 = 512, .m2 = 512 },
    .{ .name = "control: 36 KiB tf_ctl_big_tg", .threads = 256, .tg_memory = 36 * 1024 },
};

/// Every Metal launch's problems on every family we serve, printed; returns how many.
pub fn checkMetal(a: std.mem.Allocator, list: []const M, quiet: bool) !usize {
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(a);
    var n: usize = 0;
    for (list) |l| for (limits.apple) |f| {
        n += try limits.metal(l, f, &out, a);
    };
    if (!quiet and out.items.len > 0) std.debug.print("{s}", .{out.items});
    return n;
}

test "the registered launches fit every Apple family and every SM, and each planted control breaks one" {
    const a = std.testing.allocator;
    try std.testing.expectEqual(@as(usize, 0), try checkMetal(a, &metal, false));
    for (metal_controls) |l| try std.testing.expect(try checkMetal(a, &.{l}, true) > 0);
    try std.testing.expectEqual(@as(usize, 0), try checkCuda(a, &cuda, false));
    for (cuda_controls) |r| try std.testing.expect(try checkCuda(a, &.{r}, true) > 0);
}

test "scoped records still require every declared compile and an unknown target cannot skip all checks" {
    const a = std.testing.allocator;
    var r = CudaRecord{ .launch = .{ .name = "scoped", .symbol = "scoped", .image = "scoped", .threads = 128 }, .usage = @splat(.{ .regs = 0, .static = 0 }), .target_sm = 121 };
    try std.testing.expectEqual(@as(usize, 1), try checkCuda(a, &.{r}, true));
    r.usage[4] = .{ .regs = 32, .static = 0, .bound = 128 };
    try std.testing.expectEqual(@as(usize, 0), try checkCuda(a, &.{r}, true));
    r.target_sm = 0;
    try std.testing.expectEqual(@as(usize, 4), try checkCuda(a, &.{r}, true));
    r.target_sm = 75;
    try std.testing.expectEqual(@as(usize, 1), try checkCuda(a, &.{r}, true));
}

test {
    _ = @import("registry_sets_test.zig");
}

const paging = @import("paging_pending.zig");
/// What the live audit checks: the serving records, then the paging launches and the Qwen engine's MoE and MTP ones.
pub const all_cuda = cuda ++ paging.extra ++ @import("qwen_served_records.zig").records ++ @import("registry_glm.zig").records(CudaRecord);
test {
    _ = paging;
}

/// The live Metal collector's list: the serving declarations, then the 27B's register-slot DeltaNet pipelines.
pub const all_metal = metal ++ @import("delta_slots_records.zig").records;
