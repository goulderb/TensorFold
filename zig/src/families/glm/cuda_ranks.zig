//! A TP rank's settings fingerprint, all-gathered before weights load: a mismatch stops every rank.

const std = @import("std");
const cuda = @import("cuda");
const heat = @import("heat");
const cache = @import("cuda_cache.zig");
const precision = @import("cuda_weight_precision.zig");
const mia = @import("cuda_experts_mia.zig");
const verify = @import("cuda_verify_plan.zig");
const profile = @import("cuda_draft_profile.zig");
const route_trace = @import("cuda_route_trace.zig");
const l2pf = @import("cuda_l2pf_plan.zig");
const kda = @import("cuda_kda.zig");
const prompt_latent = @import("cuda_prompt_latent_aot.zig");
const Sha256 = std.crypto.hash.sha2.Sha256;
pub const Drafter = enum(u8) { off, mtp, dflash2 };
pub const TargetGraphs = enum(u8) { off, ffn, full };
pub const Communication = enum(u8) { nccl, roce };

pub fn effectiveDraft(mode: Drafter, enabled: bool) !Drafter {
    if (mode == .dflash2 and !enabled) return error.DFlashNeedsDrafts;
    return if (enabled) mode else .off;
}

/// Canonical --drafter loads the draft head without enabling target MTP execution.
pub fn loadDraftHead(canonical: bool, draft: Drafter) bool {
    return canonical or draft == .mtp;
}

pub fn targetMtp(canonical: bool, draft: Drafter) bool {
    return draft == .mtp and !canonical;
}

/// Leftover TF_GLM_DRAFT_PROFILE is not the selected owner when --drafter is set.
pub fn selectedDraftProfile(canonical: bool, settings: Settings) bool {
    return settings.draft_profile.enabled and !canonical;
}

/// Engine.init's profile check: leftover production must not refuse canonical MTP, and selected production still requires DFlash2/7.
pub fn validateDraftOwner(settings: Settings, canonical: bool) ?anyerror {
    if (!selectedDraftProfile(canonical, settings)) return null;
    if (settings.draft != .dflash2 or settings.dflash_verify != .dflash7) return error.BadDraftProfile;
    settings.draft_profile.copy.validate() catch |err| return err;
    return null;
}

/// The forward's knobs a rank reads from its environment (the Python family's names and defaults).
pub const Settings = struct {
    kv: cache.Format = .bf16,
    attn_fp8load: bool = false,
    prompt_latent_aot: bool = false,
    weight_dense: precision.DensePolicy = .{},
    weight_head: precision.Head = .bf16,
    exl3_decode: cuda.exl3_trellis_experts.Decode = .scalar,
    exl3_routed: mia.Mode = .generic,
    draft: Drafter = .mtp,
    dflash_verify: verify.Plan = .baseline,
    draft_profile: profile.Settings = .{},
    draft_dir: ?[]const u8 = null,
    draft_checkpoint: [32]u8 = @splat(0),
    vision: bool = false,
    kv_pool_tokens: u32 = 0,
    kv_shard: bool = false,
    split: bool = true, // TF_GLM_PREFILL_REDUCE: split (each rank glues its own rows) or gather
    kda_prompt: kda.PromptAlgorithm = .serial,
    affine_prompt_policy: cuda.qlinear.AffinePromptPolicy = .stock,
    prompt_fixed_k: bool = false, // TF_GLM_PROMPT_K: one (default) or shape (experimental fixed K slices)
    tiled: bool = true, // TF_GLM_MM: 1 (the bf16 linears on glm_mm, tiled at load) or 0
    ring: bool = false, // TF_GLM_PREFILL_REDUCE=ring: split, its exchanges passed along the ring (cuda_relay)
    kda_wide: bool = true, // TF_GLM_KDA_DECODE_WIDE: 1 (decode windows on KDA's three-kernel chain) or 0
    target_graphs: TargetGraphs = .off,
    communication: Communication = .nccl,
    roce_max_bytes: u32 = 512 * 1024,
    prefetch: l2pf.Settings = .{},
    draft_prelaunch: bool = false,
    prefill_overlap: u8 = 0,
    prefill_pieces: u32 = 2,
    pieces: u32 = 3, // TF_GLM_OVERLAP_PIECES: a prompt chunk's exchanges in this many row pieces (1: whole).
    heat: ?heat.Bands = null, // TF_HEAT_HIGH and TF_HEAT_LOW (upstream's names): a prompt chunk waits on every rank's hottest zone
    layers: u32 = 0, // TF_GLM_LAYERS: the first this many layers only, a thin model for one-GPU checks (0: all)
    spill: [32]u8 = @splat(0),
    dense_bits: u8 = 16,
    head_bits: u8 = 16,

    pub fn hashNumerics(s: Settings, digest: *Sha256) void {
        digest.update("glm-mixed1050-default-stored-v1\x00");
        digest.update(&.{ @intFromBool(s.prompt_fixed_k), @intFromBool(s.tiled), @intFromBool(s.ring), s.dense_bits, s.head_bits });
        digest.update("glm-numerics-mia-f1ee79a6-v1\x00");
        digest.update("glm-affine4-prefill-explicit-v1\x00");
        if (s.prompt_latent_aot) digest.update(&prompt_latent.numerical_identity);
        if (s.kda_prompt == .wy32) digest.update("glm-kda-wy32-production-v1\x00");
        if (s.affine_prompt_policy == .glm_production) digest.update("glm-affine-prompt-tiles-production-v1\x00");
        if (s.prefill_overlap != 0) {
            digest.update("glm-prompt-global-half-v1\x00");
            digest.update(&.{s.prefill_overlap});
            var pieces_bytes: [4]u8 = undefined;
            std.mem.writeInt(u32, &pieces_bytes, s.prefill_pieces, .little);
            digest.update(&pieces_bytes);
        }
        s.weight_dense.hash(digest);
        s.draft_profile.hash(digest);
        if (s.prefetch.mode != null) digest.update(&s.prefetch.digest());
        digest.update(@tagName(s.draft));
        digest.update("\x00");
        digest.update(&s.draft_checkpoint);
        switch (s.dflash_verify) {
            .baseline => {},
            .dflash6 => digest.update("dflash-verify-six-v1\x00"),
            .dflash7 => digest.update("dflash-verify-seven-v1\x00"),
        }
        for ([_][]const u8{ @tagName(s.weight_head), @tagName(s.exl3_decode), @tagName(s.exl3_routed) }) |value| {
            digest.update(value);
            digest.update("\x00");
        }
        // Execution identity, not a claim of qualified graph arithmetic.
        switch (s.target_graphs) {
            .off => {},
            .ffn => digest.update("glm-target-ffn-graphs-research-v1\x00"),
            .full => digest.update("glm-target-full-graphs-v1\x00"),
        }
    }

    /// The settings `get` (an environment lookup) gives; `problem` names a value that is not one of the choices.
    pub fn read(get: *const fn ([:0]const u8) ?[]const u8, problem: *[]const u8) !Settings {
        return readForOwner(get, problem, false);
    }

    /// Canonical `--drafter` is selected before leftover TF_GLM_DRAFTER is validated or applied.
    pub fn readForOwner(get: *const fn ([:0]const u8) ?[]const u8, problem: *[]const u8, canonical: bool) !Settings {
        var s: Settings = .{};
        if (get("TF_GLM_PROMPT_LATENT_AOT")) |v| {
            if (std.mem.eql(u8, v, "1")) s.prompt_latent_aot = true else if (!std.mem.eql(u8, v, "0")) {
                problem.* = "TF_GLM_PROMPT_LATENT_AOT: 0 or 1";
                return error.BadSetting;
            }
        }
        if (get("TF_GLM_PROMPT_TILE")) |v| {
            if (std.mem.eql(u8, v, "1")) s.affine_prompt_policy = .glm_production else if (!std.mem.eql(u8, v, "0")) {
                problem.* = "TF_GLM_PROMPT_TILE: 0 or 1";
                return error.BadSetting;
            }
        }
        if (get("TF_GLM_PREFILL_OVERLAP")) |v| {
            if (std.mem.eql(u8, v, "0")) s.prefill_overlap = 0 else if (std.mem.eql(u8, v, "1")) s.prefill_overlap = 1 else if (std.mem.eql(u8, v, "2")) s.prefill_overlap = 2 else {
                problem.* = "TF_GLM_PREFILL_OVERLAP: 0, 1 or 2";
                return error.BadSetting;
            }
        }
        if (get("TF_GLM_PREFILL_OVERLAP_PIECES")) |v| {
            s.prefill_pieces = std.fmt.parseInt(u32, v, 10) catch 0;
            if (s.prefill_pieces < 1 or s.prefill_pieces > 8) {
                problem.* = "TF_GLM_PREFILL_OVERLAP_PIECES: 1 to 8";
                return error.BadSetting;
            }
        }
        if (get("TF_GLM_KDA_CHUNKED")) |v| {
            if (std.mem.eql(u8, v, "1")) s.kda_prompt = .wy32 else if (!std.mem.eql(u8, v, "0")) {
                problem.* = "TF_GLM_KDA_CHUNKED: 0 or 1";
                return error.BadSetting;
            }
        }
        if (get("TF_GLM_DRAFT_PRELAUNCH")) |v| {
            if (std.mem.eql(u8, v, "1")) s.draft_prelaunch = true else if (!std.mem.eql(u8, v, "0")) {
                problem.* = "TF_GLM_DRAFT_PRELAUNCH: 0 or 1";
                return error.BadSetting;
            }
        }
        s.prefetch = l2pf.Settings.parse(get) catch {
            problem.* = "invalid TF_GLM_L2PF settings";
            return error.BadSetting;
        };
        if (get("TF_GLM_COMM")) |v| {
            s.communication = std.meta.stringToEnum(Communication, v) orelse {
                problem.* = "TF_GLM_COMM: nccl or roce";
                return error.BadSetting;
            };
        }
        if (get("TF_ROCE_MAX_KB")) |v| {
            const kb = std.fmt.parseInt(u32, v, 10) catch 0;
            if (kb == 0 or kb > 16384) {
                problem.* = "TF_ROCE_MAX_KB: 1 to 16384";
                return error.BadSetting;
            }
            s.roce_max_bytes = kb * 1024;
        }
        if (get("TF_GLM_TARGET_GRAPHS")) |v| {
            s.target_graphs = std.meta.stringToEnum(TargetGraphs, v) orelse {
                problem.* = "TF_GLM_TARGET_GRAPHS: off, ffn or full";
                return error.BadSetting;
            };
        }
        if (s.target_graphs != .off) {
            const samples = route_trace.samples(get("TF_GLM_ROUTE_SAMPLES")) catch {
                problem.* = "TF_GLM_ROUTE_SAMPLES: 0 to 32";
                return error.BadSetting;
            };
            if (samples != 0) {
                problem.* = "TF_GLM_TARGET_GRAPHS requires TF_GLM_ROUTE_SAMPLES=0: diagnostic trace copies cannot be captured";
                return error.BadSetting;
            }
        }
        s.draft_profile = try profile.Settings.read(get, problem);
        if (s.draft_profile.enabled and !canonical) {
            s.draft = .dflash2;
            s.dflash_verify = .dflash7;
        }
        if (get("TF_GLM_EXL3_ROUTED")) |v| {
            s.exl3_routed = if (std.mem.eql(u8, v, "generic")) .generic else if (std.mem.eql(u8, v, "mia-mcg4")) .mia_mcg4 else {
                problem.* = "TF_GLM_EXL3_ROUTED: generic or mia-mcg4";
                return error.BadSetting;
            };
        }
        if (get("TF_GLM_DRAFTER")) |v| {
            if (!canonical) {
                s.draft = std.meta.stringToEnum(Drafter, v) orelse {
                    problem.* = "TF_GLM_DRAFTER: off, mtp or dflash2";
                    return error.BadSetting;
                };
            }
        }
        if (s.draft_profile.enabled and s.draft != .dflash2 and !canonical) {
            problem.* = "TF_GLM_DRAFT_PROFILE=production requires TF_GLM_DRAFTER=dflash2";
            return error.BadSetting;
        }
        s.draft_dir = get("TF_GLM_DFLASH_DIR");
        if (get("TF_GLM_DFLASH_VERIFY_DRAFTS")) |v| {
            s.dflash_verify = verify.Plan.parse(v) catch {
                problem.* = "TF_GLM_DFLASH_VERIFY_DRAFTS: 3, 6 or 7";
                return error.BadSetting;
            };
        }
        if (!canonical and s.draft == .dflash2 and (s.draft_dir == null or s.draft_dir.?.len == 0)) {
            problem.* = "TF_GLM_DFLASH_DIR is required for dflash2";
            return error.BadSetting;
        }
        if (get("TF_GLM_EXL3_DECODE")) |v| {
            s.exl3_decode = std.meta.stringToEnum(cuda.exl3_trellis_experts.Decode, v) orelse {
                problem.* = "TF_GLM_EXL3_DECODE: scalar or tensor";
                return error.BadSetting;
            };
        }
        s.weight_dense = try precision.DensePolicy.read(get, problem);
        s.weight_head = precision.parseHead(get("TF_GLM_HEAD")) catch {
            problem.* = "TF_GLM_HEAD: bf16 or fp8";
            return error.BadSetting;
        };
        if (get("TF_GLM_VISION")) |v| {
            if (std.mem.eql(u8, v, "1")) s.vision = true else if (!std.mem.eql(u8, v, "0")) {
                problem.* = "TF_GLM_VISION: 0 or 1";
                return error.BadSetting;
            }
        }
        var spill_hash = Sha256.init(.{});
        inline for (.{ "TENSORFOLD_SPILL_GIB", "TENSORFOLD_SPILL_HOST_GIB", "TENSORFOLD_SPILL_HIGHWATER", "TENSORFOLD_SPILL_MIN_TOKENS", "TENSORFOLD_SPILL_MIN_FREE_GIB", "TENSORFOLD_SPILL_FLUSH_S", "TENSORFOLD_SPILL_KEEP_BUILDS", "TF_GLM_SPILL_EAGER" }) |name| {
            spill_hash.update(name ++ "=");
            spill_hash.update(get(name) orelse "");
            spill_hash.update("\x00");
        }
        s.spill = spill_hash.finalResult();
        if (get("TF_GLM_KV_POOL_TOKENS")) |raw| {
            const v = std.mem.trim(u8, raw, " \t\r\n");
            s.kv_pool_tokens = std.fmt.parseInt(u32, v, 10) catch {
                problem.* = "TF_GLM_KV_POOL_TOKENS: an unsigned token count";
                return error.BadSetting;
            };
        }
        if (get("TF_GLM_KV_SHARD")) |v| {
            if (std.mem.eql(u8, v, "1")) s.kv_shard = true else if (!std.mem.eql(u8, v, "0")) {
                problem.* = "TF_GLM_KV_SHARD: 0 or 1";
                return error.BadSetting;
            }
        }
        if (s.kv_shard and s.kv_pool_tokens == 0) {
            problem.* = "TF_GLM_KV_SHARD requires TF_GLM_KV_POOL_TOKENS";
            return error.BadSetting;
        }
        if (get("TF_GLM_KV")) |raw| {
            const v = std.mem.trim(u8, raw, " \t\r\n");
            if (v.len == 0 or std.mem.eql(u8, v, "bf16")) s.kv = .bf16 else if (std.mem.eql(u8, v, "fp8")) s.kv = .fp8 else {
                problem.* = "TF_GLM_KV: bf16 or fp8";
                return error.BadSetting;
            }
        }
        if (get("TF_GLM_ATTN_FP8LOAD")) |v| {
            if (std.mem.eql(u8, v, "1")) s.attn_fp8load = true else if (!std.mem.eql(u8, v, "0")) {
                problem.* = "TF_GLM_ATTN_FP8LOAD: 0 or 1";
                return error.BadSetting;
            }
        }
        if (get("TF_GLM_PROMPT_K")) |v| {
            if (std.mem.eql(u8, v, "shape")) s.prompt_fixed_k = true else if (!std.mem.eql(u8, v, "one")) {
                problem.* = "TF_GLM_PROMPT_K: one or shape";
                return error.BadSetting;
            }
        }
        if (get("TF_GLM_MM")) |v| {
            if (std.mem.eql(u8, v, "0")) s.tiled = false else if (!std.mem.eql(u8, v, "1")) {
                problem.* = "TF_GLM_MM: 0 or 1";
                return error.BadSetting;
            }
        }
        if (get("TF_GLM_PREFILL_REDUCE")) |v| {
            if (std.mem.eql(u8, v, "split")) s.split = true else if (std.mem.eql(u8, v, "ring")) {
                s.split = true;
                s.ring = true;
            } else if (std.mem.eql(u8, v, "gather")) s.split = false else {
                problem.* = "TF_GLM_PREFILL_REDUCE: split, ring or gather";
                return error.BadSetting;
            }
        }
        if (get("TF_GLM_KDA_DECODE_WIDE")) |v| {
            if (std.mem.eql(u8, v, "1")) s.kda_wide = true else if (std.mem.eql(u8, v, "0")) s.kda_wide = false else {
                problem.* = "TF_GLM_KDA_DECODE_WIDE: 0 or 1";
                return error.BadSetting;
            }
        }
        if (get("TF_GLM_OVERLAP_PIECES")) |v| {
            s.pieces = std.fmt.parseInt(u32, v, 10) catch 0;
            if (s.pieces < 1 or s.pieces > 8) {
                problem.* = "TF_GLM_OVERLAP_PIECES: 1 to 8";
                return error.BadSetting;
            }
        }
        if (get("TF_GLM_LAYERS")) |v| {
            s.layers = std.fmt.parseInt(u32, v, 10) catch 0;
            if (s.layers == 0) {
                problem.* = "TF_GLM_LAYERS: the first how many layers, 1 or more";
                return error.BadSetting;
            }
        }
        inline for (.{ .{ "TF_GLM_DENSE_BITS", "dense_bits" }, .{ "TF_GLM_HEAD_BITS", "head_bits" } }) |e| if (get(e[0])) |v| {
            const b = std.fmt.parseInt(u8, v, 10) catch {
                problem.* = e[0] ++ ": 4, 6, 8 or 16";
                return error.BadSetting;
            };
            @field(s, e[1]) = switch (b) {
                4, 6, 8, 16 => b,
                0 => 16, // the old family's spelling of bf16
                else => {
                    problem.* = e[0] ++ ": 4, 6, 8 or 16";
                    return error.BadSetting;
                },
            };
        };
        s.heat = heat.bandsFrom(get("TF_HEAT_HIGH"), get("TF_HEAT_LOW")) catch {
            problem.* = "TF_HEAT_HIGH and TF_HEAT_LOW: both or neither, in degrees, with low at or under high";
            return error.BadSetting;
        };
        if (s.prefill_overlap != 0 and (!s.split or s.ring)) {
            problem.* = "TF_GLM_PREFILL_OVERLAP requires TF_GLM_PREFILL_REDUCE=split";
            return error.BadSetting;
        }
        if (s.prompt_fixed_k and !s.tiled) {
            problem.* = "TF_GLM_PROMPT_K=shape requires TF_GLM_MM=1";
            return error.BadSetting;
        }
        return s;
    }
};

/// The most ranks a world holds: rank 0's greedy pick gathers every rank's top-1 into one host array of this size.
pub const max_world = 16;

/// The process environment's value of `name` (null when unset).
pub fn environment(name: [:0]const u8) ?[]const u8 {
    return @import("sys").getenv(name);
}

/// What every rank must agree on, as bytes the communicator carries.
pub const Fingerprint = extern struct {
    layout: u32 = 24, // canonical drafter content/precision and absorb-only block protocol
    world: u32,
    rank: u32,
    prompt_fixed_k: u32,
    tiled: u32,
    split: u32, // 0 gather, 1 direct split, 2 ring
    kda_wide: u32,
    kda_prompt: u32,
    affine_prompt_policy: u32 = 0,
    target_graphs: u32 = 0,
    communication: u32 = 0,
    roce_max_bytes: u32 = 0,
    prefetch: [32]u8 = @splat(0),
    draft_prelaunch: u32 = 0,
    pieces: u32,
    prefill_overlap: u32,
    prefill_pieces: u32,
    layers: u32, // TF_GLM_LAYERS (0: all)
    kv: u32,
    attn_fp8load: u32 = 0,
    prompt_latent_aot: u32 = 0,
    weight_dense: [32]u8 = @splat(0),
    weight_head: u32 = 0,
    exl3_decode: u32 = 0,
    exl3_routed: u32 = 0,
    draft: u32 = 0,
    dflash_verify: u32 = 0,
    draft_profile: [32]u8 = @splat(0),
    draft_checkpoint: [32]u8 = @splat(0),
    canonical_checkpoint: [32]u8 = @splat(0),
    canonical_bits: u32 = 0, // 0 disabled, 4 Q4, 16 BF16
    vision_setting: u32 = 0, // local explicit tower policy, separate from media geometry
    vision: u32 = 0, // upstream image-token limit (0: media off)
    video_tokens: u32 = 0,
    max_images: u32 = 0,
    max_videos: u32 = 0,
    max_frames: u32 = 0,
    mtp: u32,
    bits: u32 = 16 << 8 | 16,
    kv_pool_tokens: u32,
    kv_shard: u32,
    spill: [32]u8 = @splat(0),
    capacity: u64, // tokens a sequence holds
    heat_high: f64, // the heat bands in degrees (0 and 0: off): every rank must wait alike
    heat_low: f64,
    config: [32]u8, // sha256 of config.json
    program: [32]u8, // sha256 of this executable
    checkpoint: [64]u8, // hex digest of checkpoint metadata and every shard byte

    pub const Field = enum { layout, world, rank, prompt_fixed_k, tiled, split, kda_wide, kda_prompt, affine_prompt_policy, target_graphs, communication, roce_max_bytes, prefetch, draft_prelaunch, pieces, prefill_overlap, prefill_pieces, layers, kv, attn_fp8load, prompt_latent_aot, weight_dense, weight_head, exl3_decode, exl3_routed, draft, dflash_verify, draft_profile, draft_checkpoint, canonical_checkpoint, canonical_bits, vision_setting, vision, video_tokens, max_images, max_videos, max_frames, mtp, bits, kv_pool_tokens, kv_shard, spill, capacity, heat_high, heat_low, config, program, checkpoint };

    pub fn of(world: usize, rank: usize, s: Settings, capacity: usize, mtp: bool, config: [32]u8, program: [32]u8, checkpoint_digest: [32]u8) Fingerprint {
        var f: Fingerprint = .{ .world = @intCast(world), .rank = @intCast(rank), .prompt_fixed_k = @intFromBool(s.prompt_fixed_k), .tiled = @intFromBool(s.tiled), .split = if (!s.split) 0 else if (s.ring) 2 else 1, .kda_wide = @intFromBool(s.kda_wide), .kda_prompt = @backingInt(s.kda_prompt), .pieces = s.pieces, .prefill_overlap = s.prefill_overlap, .prefill_pieces = s.prefill_pieces, .layers = s.layers, .kv = @backingInt(s.kv), .kv_pool_tokens = s.kv_pool_tokens, .kv_shard = @intFromBool(s.kv_shard), .capacity = capacity, .heat_high = if (s.heat) |b| b.high_c else 0, .heat_low = if (s.heat) |b| b.low_c else 0, .config = config, .program = program, .checkpoint = std.fmt.bytesToHex(checkpoint_digest, .lower), .mtp = @intFromBool(mtp), .bits = @as(u32, s.dense_bits) << 8 | s.head_bits };
        f.attn_fp8load = @intFromBool(s.attn_fp8load);
        f.prompt_latent_aot = @intFromBool(s.prompt_latent_aot);
        f.affine_prompt_policy = @backingInt(s.affine_prompt_policy);
        f.target_graphs = @backingInt(s.target_graphs);
        f.communication = @backingInt(s.communication);
        f.roce_max_bytes = s.roce_max_bytes;
        f.prefetch = s.prefetch.digest();
        f.draft_prelaunch = @intFromBool(s.draft_prelaunch);
        f.spill = s.spill;
        f.weight_dense = s.weight_dense.digest();
        f.weight_head = @backingInt(s.weight_head);
        f.exl3_decode = @backingInt(s.exl3_decode);
        f.exl3_routed = @backingInt(s.exl3_routed);
        f.draft = @backingInt(s.draft);
        f.dflash_verify = @backingInt(s.dflash_verify);
        f.draft_profile = s.draft_profile.digest();
        f.draft_checkpoint = s.draft_checkpoint;
        f.vision_setting = @intFromBool(s.vision);
        return f;
    }
};

/// Canonical command-line drafter identity, not the legacy Settings.draft_checkpoint identity.
/// The precision decoder is shared with the real loader: zero means unquantized BF16, not disabled.
pub fn canonicalDraftBits(bits: u8) !u32 {
    return switch (bits) {
        0 => 16,
        4 => 4,
        else => error.BadDrafterBits,
    };
}

pub const CanonicalDraft = struct {
    checkpoint: [32]u8 = @splat(0),
    bits: u32 = 0,

    pub fn read(gpa: std.mem.Allocator, io: std.Io, dir: ?[]const u8, bits: u8) !CanonicalDraft {
        const path = dir orelse return .{};
        const effective = try canonicalDraftBits(bits);
        return .{ .checkpoint = try hashDraftCheckpoint(gpa, io, path), .bits = effective };
    }

    pub fn fingerprint(self: CanonicalDraft, fp: *Fingerprint) void {
        fp.canonical_checkpoint = self.checkpoint;
        fp.canonical_bits = self.bits;
    }

    /// Saved-state identity for the chosen canonical owner. Distinct from Fingerprint.layout.
    pub fn snapshotIdentity(self: CanonicalDraft) [32]u8 {
        var h = Sha256.init(.{});
        h.update("glm-dflash2-saved-v2\x00");
        h.update(&self.checkpoint);
        var bits: [4]u8 = undefined;
        std.mem.writeInt(u32, &bits, self.bits, .little);
        h.update(&bits);
        return h.finalResult();
    }
};

pub const DraftSnapshot = struct { bytes: u64, identity: [32]u8 };

/// Persist the chosen owner's content/precision identity. Legacy env DFlash keeps its checkpoint hash.
pub fn persistedDraftSnapshot(canonical_owner: bool, canonical: CanonicalDraft, canonical_bytes: u64, legacy_bytes: u64, legacy_identity: [32]u8) DraftSnapshot {
    if (canonical_owner) return .{ .bytes = canonical_bytes, .identity = canonical.snapshotIdentity() };
    return .{ .bytes = legacy_bytes, .identity = legacy_identity };
}

/// Engine.init's canonical CLI options. Tests drive this same record, not a reconstructed vote.
pub const CanonicalOptions = struct {
    dir: ?[]const u8 = null,
    bits: u8 = 4,
    drafts: bool = false,
};

/// Production identity load: a content/precision failure or drafts mismatch becomes the vote problem.
pub fn loadCanonical(gpa: std.mem.Allocator, io: std.Io, o: CanonicalOptions, problem: *?anyerror) CanonicalDraft {
    const identity = CanonicalDraft.read(gpa, io, o.dir, o.bits) catch |err| blk: {
        problem.* = err;
        break :blk CanonicalDraft{};
    };
    if (o.dir != null and !o.drafts) problem.* = error.DrafterNeedsDrafts;
    return identity;
}

/// Production fingerprint stamp after Fingerprint.of and optional media fields.
pub fn stampCanonical(fp: *Fingerprint, identity: CanonicalDraft, present: bool) void {
    fp.mtp += @intFromBool(present);
    identity.fingerprint(fp);
}

/// Engine.init's canonical startup: load options, vote, stamp the live fingerprint, then agree.
pub fn applyCanonicalStartup(
    gpa: std.mem.Allocator,
    io: std.Io,
    d: *const cuda.Driver,
    lib: *const cuda.nccl.Library,
    comm: cuda.nccl.Comm,
    stream: cuda.Stream,
    rank: usize,
    o: CanonicalOptions,
    present: bool,
    problem: *?anyerror,
    fp: *Fingerprint,
    vote: []u32,
    all: []Fingerprint,
) !CanonicalDraft {
    const identity = loadCanonical(gpa, io, o, problem);
    try startupVote(d, lib, comm, stream, rank, problem.*, vote);
    stampCanonical(fp, identity, present);
    try agree(d, lib, comm, stream, fp.*, all);
    return identity;
}

pub const Mismatch = struct { field: Fingerprint.Field, rank: usize };

/// The first field where rank `r`'s fingerprint differs from rank 0's (or holds the wrong rank), null when all agree.
pub fn mismatch(all: []const Fingerprint) ?Mismatch {
    for (all, 0..) |f, r| {
        if (f.layout != all[0].layout) return .{ .field = .layout, .rank = r };
        if (f.rank != r) return .{ .field = .rank, .rank = r };
        inline for (@typeInfo(Fingerprint).@"struct".field_names) |name| {
            if (comptime std.mem.eql(u8, name, "layout") or std.mem.eql(u8, name, "rank")) continue;
            if (!std.mem.eql(u8, std.mem.asBytes(&@field(f, name)), std.mem.asBytes(&@field(all[0], name))))
                return .{ .field = @field(Fingerprint.Field, name), .rank = r };
        }
    }
    return null;
}

/// sha256 of a file's bytes (read whole: config.json, the executable).
pub fn hashFile(gpa: std.mem.Allocator, io: std.Io, path: []const u8) ![32]u8 {
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 30));
    defer gpa.free(bytes);
    return hashBytes(bytes);
}

pub fn hashBytes(bytes: []const u8) [32]u8 {
    var out: [32]u8 = undefined;
    Sha256.hash(bytes, &out, .{});
    return out;
}

pub fn hashDraftCheckpoint(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) ![32]u8 {
    var h = Sha256.init(.{});
    h.update("glm-dflash2-checkpoint-v1\x00");
    const scratch = try gpa.alloc(u8, 1 << 20);
    defer gpa.free(scratch);
    for ([_][]const u8{ "config.json", "model.safetensors" }) |name| {
        h.update(name);
        h.update("\x00");
        const path = try std.fs.path.join(gpa, &.{ dir, name });
        defer gpa.free(path);
        var file = try std.Io.Dir.cwd().openFile(io, path, .{});
        defer file.close(io);
        var offset: u64 = 0;
        while (true) {
            const n = try file.readPositionalAll(io, scratch, offset);
            if (n == 0) break;
            h.update(scratch[0..n]);
            offset = try std.math.add(u64, offset, n);
        }
        var size: [8]u8 = undefined;
        std.mem.writeInt(u64, &size, offset, .little);
        h.update(&size);
    }
    return h.finalResult();
}

test "communication and cap are validated before ranks choose a transport" {
    const Env = struct {
        fn selected(name: [:0]const u8) ?[]const u8 {
            if (std.mem.eql(u8, name, "TF_GLM_COMM")) return "roce";
            if (std.mem.eql(u8, name, "TF_ROCE_MAX_KB")) return "512";
            return null;
        }
        fn badMode(name: [:0]const u8) ?[]const u8 {
            return if (std.mem.eql(u8, name, "TF_GLM_COMM")) "auto" else null;
        }
        fn badCap(name: [:0]const u8) ?[]const u8 {
            return if (std.mem.eql(u8, name, "TF_ROCE_MAX_KB")) "0" else null;
        }
    };
    var problem: []const u8 = "";
    const s = try Settings.read(Env.selected, &problem);
    try std.testing.expectEqual(Communication.roce, s.communication);
    try std.testing.expectEqual(@as(u32, 512 * 1024), s.roce_max_bytes);
    try std.testing.expectError(error.BadSetting, Settings.read(Env.badMode, &problem));
    try std.testing.expectError(error.BadSetting, Settings.read(Env.badCap, &problem));
    const h: [32]u8 = @splat(0);
    var all = [_]Fingerprint{ Fingerprint.of(2, 0, .{}, 8192, true, h, h, hashBytes("target")), Fingerprint.of(2, 1, s, 8192, true, h, h, hashBytes("target")) };
    try std.testing.expectEqual(Fingerprint.Field.communication, mismatch(&all).?.field);
    all[0] = Fingerprint.of(2, 0, s, 8192, true, h, h, hashBytes("target"));
    all[1].roce_max_bytes /= 2;
    try std.testing.expectEqual(Fingerprint.Field.roce_max_bytes, mismatch(&all).?.field);
}

test "DFlash requires an explicit checkpoint and speculative backend" {
    const Env = struct {
        fn get(name: [:0]const u8) ?[]const u8 {
            if (std.mem.eql(u8, name, "TF_GLM_DRAFTER")) return "dflash2";
            if (std.mem.eql(u8, name, "TF_GLM_DFLASH_DIR")) return "/draft";
            return null;
        }
        fn missing(name: [:0]const u8) ?[]const u8 {
            return if (std.mem.eql(u8, name, "TF_GLM_DRAFTER")) "dflash2" else null;
        }
    };
    var problem: []const u8 = "";
    const selected = try Settings.read(Env.get, &problem);
    try std.testing.expectEqual(Drafter.dflash2, selected.draft);
    try std.testing.expectEqualStrings("/draft", selected.draft_dir.?);
    try std.testing.expectError(error.BadSetting, Settings.read(Env.missing, &problem));
    try std.testing.expectError(error.DFlashNeedsDrafts, effectiveDraft(.dflash2, false));
    try std.testing.expectEqual(Drafter.off, try effectiveDraft(.mtp, false));
    try std.testing.expectEqual(Drafter.mtp, try effectiveDraft(.mtp, true));
    const h: [32]u8 = @splat(0);
    var all = [_]Fingerprint{ Fingerprint.of(2, 0, selected, 8192, true, h, h, hashBytes("target")), Fingerprint.of(2, 1, selected, 8192, true, h, h, hashBytes("target")) };
    all[1].draft_checkpoint[0] = 1;
    try std.testing.expectEqual(Fingerprint.Field.draft_checkpoint, mismatch(&all).?.field);
    all[1].draft_checkpoint = h;
    all[1].draft = @backingInt(Drafter.mtp);
    try std.testing.expectEqual(Fingerprint.Field.draft, mismatch(&all).?.field);
}

test "Mia expert selection parses once and separates rank and spill identities" {
    const Env = struct {
        fn miaChoice(name: [:0]const u8) ?[]const u8 {
            return if (std.mem.eql(u8, name, "TF_GLM_EXL3_ROUTED")) "mia-mcg4" else null;
        }
        fn unknown(name: [:0]const u8) ?[]const u8 {
            return if (std.mem.eql(u8, name, "TF_GLM_EXL3_ROUTED")) "mcg" else null;
        }
    };
    var problem: []const u8 = "";
    const selected = try Settings.read(Env.miaChoice, &problem);
    try std.testing.expectEqual(mia.Mode.mia_mcg4, selected.exl3_routed);
    try std.testing.expectError(error.BadSetting, Settings.read(Env.unknown, &problem));
    try std.testing.expectEqualStrings("TF_GLM_EXL3_ROUTED: generic or mia-mcg4", problem);
    const h: [32]u8 = @splat(0);
    const all = [_]Fingerprint{ Fingerprint.of(2, 0, .{}, 8192, true, h, h, hashBytes("target")), Fingerprint.of(2, 1, selected, 8192, true, h, h, hashBytes("target")) };
    try std.testing.expectEqual(Fingerprint.Field.exl3_routed, mismatch(&all).?.field);
    var a = Sha256.init(.{});
    var b = Sha256.init(.{});
    (Settings{}).hashNumerics(&a);
    selected.hashNumerics(&b);
    try std.testing.expect(!std.mem.eql(u8, &a.finalResult(), &b.finalResult()));
}

fn targetGraphSetting(comptime mode: ?[]const u8, comptime samples: ?[]const u8) *const fn ([:0]const u8) ?[]const u8 {
    return &struct {
        fn get(name: [:0]const u8) ?[]const u8 {
            if (std.mem.eql(u8, name, "TF_GLM_TARGET_GRAPHS")) return mode;
            if (std.mem.eql(u8, name, "TF_GLM_ROUTE_SAMPLES")) return samples;
            return null;
        }
    }.get;
}

test "target graph modes are opt in and reject unsupported values" {
    var problem: []const u8 = "";
    try std.testing.expectEqual(TargetGraphs.off, (Settings{}).target_graphs);
    try std.testing.expectEqual(TargetGraphs.off, (try Settings.read(targetGraphSetting(null, null), &problem)).target_graphs);
    try std.testing.expectEqual(TargetGraphs.off, (try Settings.read(targetGraphSetting("off", null), &problem)).target_graphs);
    try std.testing.expectEqual(TargetGraphs.ffn, (try Settings.read(targetGraphSetting("ffn", null), &problem)).target_graphs);
    try std.testing.expectEqual(TargetGraphs.full, (try Settings.read(targetGraphSetting("full", null), &problem)).target_graphs);
    inline for ([_][]const u8{ "1", "on", "", "FFN", "FULL", " ffn " }) |mode| {
        try std.testing.expectError(error.BadSetting, Settings.read(targetGraphSetting(mode, null), &problem));
        try std.testing.expectEqualStrings("TF_GLM_TARGET_GRAPHS: off, ffn or full", problem);
    }
}

test "target graphs refuse diagnostic route copies without changing the off path" {
    var problem: []const u8 = "";
    try std.testing.expectEqual(TargetGraphs.off, (try Settings.read(targetGraphSetting("off", "1"), &problem)).target_graphs);
    inline for ([_][]const u8{ "ffn", "full" }) |mode| {
        try std.testing.expect((try Settings.read(targetGraphSetting(mode, "0"), &problem)).target_graphs != .off);
        inline for ([_][]const u8{ "1", "32" }) |samples| {
            try std.testing.expectError(error.BadSetting, Settings.read(targetGraphSetting(mode, samples), &problem));
            try std.testing.expectEqualStrings("TF_GLM_TARGET_GRAPHS requires TF_GLM_ROUTE_SAMPLES=0: diagnostic trace copies cannot be captured", problem);
        }
        inline for ([_][]const u8{ "33", "invalid" }) |samples| {
            try std.testing.expectError(error.BadSetting, Settings.read(targetGraphSetting(mode, samples), &problem));
            try std.testing.expectEqualStrings("TF_GLM_ROUTE_SAMPLES: 0 to 32", problem);
        }
    }
}

test "target graph modes separate rank and numerical identities only when enabled" {
    var problem: []const u8 = "";
    const unset = try Settings.read(targetGraphSetting(null, null), &problem);
    const off = try Settings.read(targetGraphSetting("off", null), &problem);
    const ffn = try Settings.read(targetGraphSetting("ffn", null), &problem);
    const full = try Settings.read(targetGraphSetting("full", null), &problem);
    var full_hash = Sha256.init(.{});
    full.hashNumerics(&full_hash);
    var default_hash = Sha256.init(.{});
    var unset_hash = Sha256.init(.{});
    var off_hash = Sha256.init(.{});
    var ffn_hash = Sha256.init(.{});
    (Settings{}).hashNumerics(&default_hash);
    unset.hashNumerics(&unset_hash);
    off.hashNumerics(&off_hash);
    ffn.hashNumerics(&ffn_hash);
    const original = default_hash.finalResult();
    // The mixed stored-format epoch prefixes the retained conversion identity; off/unset must still agree.
    try std.testing.expectEqualStrings("29d6d946f5672dad28fbf8e090c0fada8236cffbf28c17fbcdd703aaab3589ec", &std.fmt.bytesToHex(original, .lower));
    try std.testing.expectEqualSlices(u8, &original, &unset_hash.finalResult());
    try std.testing.expectEqualSlices(u8, &original, &off_hash.finalResult());
    const ffn_digest = ffn_hash.finalResult();
    const full_digest = full_hash.finalResult();
    try std.testing.expect(!std.mem.eql(u8, &original, &ffn_digest));
    try std.testing.expect(!std.mem.eql(u8, &original, &full_digest));
    try std.testing.expect(!std.mem.eql(u8, &ffn_digest, &full_digest));
    const h: [32]u8 = @splat(0);
    var all = [_]Fingerprint{ Fingerprint.of(2, 0, unset, 8192, true, h, h, hashBytes("target")), Fingerprint.of(2, 1, off, 8192, true, h, h, hashBytes("target")) };
    try std.testing.expectEqual(@as(u32, 24), all[0].layout);
    try std.testing.expectEqual(@as(u32, @backingInt(TargetGraphs.off)), all[0].target_graphs);
    try std.testing.expect(mismatch(&all) == null);
    all[1] = Fingerprint.of(2, 1, ffn, 8192, true, h, h, hashBytes("target"));
    try std.testing.expectEqual(@as(u32, @backingInt(TargetGraphs.ffn)), all[1].target_graphs);
    try std.testing.expectEqual(Mismatch{ .field = .target_graphs, .rank = 1 }, mismatch(&all).?);
    all[1] = Fingerprint.of(2, 1, full, 8192, true, h, h, hashBytes("target"));
    try std.testing.expectEqual(@as(u32, @backingInt(TargetGraphs.full)), all[1].target_graphs);
    try std.testing.expectEqual(Mismatch{ .field = .target_graphs, .rank = 1 }, mismatch(&all).?);
    all[0] = Fingerprint.of(2, 0, full, 8192, true, h, h, hashBytes("target"));
    try std.testing.expect(mismatch(&all) == null);
    try std.testing.expectEqualStrings("TF_GLM_TARGET_GRAPHS (off, partial FFN or full target)", explain(.target_graphs));
}

/// Every rank's `mine` in rank order, over one ncclAllGather.
pub fn gather(
    comptime T: type,
    d: *const cuda.Driver,
    lib: *const cuda.nccl.Library,
    comm: cuda.nccl.Comm,
    stream: cuda.Stream,
    rank: usize,
    mine: T,
    out: []T,
) !void {
    const size = @sizeOf(T);
    var dev = try cuda.DeviceBuffer.alloc(d, size * out.len);
    defer dev.free();
    try dev.uploadAsync(rank * size, std.mem.asBytes(&mine), stream.handle);
    const gathered = lib.api.ncclAllGather(dev.ptr + rank * size, dev.ptr, size, .u8, comm, stream.handle);
    try lib.check(gathered, "ncclAllGather");
    try stream.synchronize();
    try dev.download(0, std.mem.sliceAsBytes(out));
}

/// A local content/precision read failure is voted before rank identity or any execution.
/// Even a healthy peer refuses, rather than entering model-load collectives alone.
pub fn startupVote(d: *const cuda.Driver, lib: *const cuda.nccl.Library, comm: cuda.nccl.Comm, stream: cuda.Stream, rank: usize, problem: ?anyerror, out: []u32) !void {
    try gather(u32, d, lib, comm, stream, rank, @intFromBool(problem == null), out);
    for (out) |ready| if (ready == 0) return problem orelse error.PeerStartupRefused;
}

/// All-gathers `mine` over `comm`; any disagreement is error.RankMismatch on every rank, printing what differs.
pub fn agree(
    d: *const cuda.Driver,
    lib: *const cuda.nccl.Library,
    comm: cuda.nccl.Comm,
    stream: cuda.Stream,
    mine: Fingerprint,
    out: []Fingerprint,
) !void {
    try gather(Fingerprint, d, lib, comm, stream, mine.rank, mine, out);
    if (mismatch(out)) |m| {
        std.debug.print("REFUSE rank settings: rank {d}'s {s} differs from rank 0's ({s})\n", .{ m.rank, @tagName(m.field), explain(m.field) });
        return error.RankMismatch;
    }
}

fn explain(f: Fingerprint.Field) []const u8 {
    return switch (f) {
        .layout => "a different build of this check",
        .rank => "a rank number given twice or out of order",
        .world => "the ranks were started with different world sizes",
        .prompt_fixed_k => "TF_GLM_PROMPT_K",
        .tiled => "TF_GLM_MM",
        .split => "TF_GLM_PREFILL_REDUCE",
        .kda_wide => "TF_GLM_KDA_DECODE_WIDE",
        .kda_prompt => "TF_GLM_KDA_CHUNKED",
        .affine_prompt_policy => "TF_GLM_PROMPT_TILE",
        .target_graphs => "TF_GLM_TARGET_GRAPHS (off, partial FFN or full target)",
        .communication => "TF_GLM_COMM",
        .roce_max_bytes => "TF_ROCE_MAX_KB",
        .prefetch => "TF_GLM_L2PF settings",
        .draft_prelaunch => "TF_GLM_DRAFT_PRELAUNCH",
        .pieces => "TF_GLM_OVERLAP_PIECES",
        .prefill_overlap => "TF_GLM_PREFILL_OVERLAP",
        .prefill_pieces => "TF_GLM_PREFILL_OVERLAP_PIECES",
        .layers => "TF_GLM_LAYERS",
        .kv => "TF_GLM_KV",
        .attn_fp8load => "TF_GLM_ATTN_FP8LOAD",
        .prompt_latent_aot => "TF_GLM_PROMPT_LATENT_AOT",
        .weight_dense => "TF_GLM_DENSE and TF_GLM_DENSE_{KDA,DSA,MLP,SHARED,MTP_EH}",
        .weight_head => "TF_GLM_HEAD",
        .exl3_decode => "TF_GLM_EXL3_DECODE",
        .exl3_routed => "TF_GLM_EXL3_ROUTED",
        .draft => "TF_GLM_DRAFTER",
        .draft_checkpoint => "DFlash checkpoint contents",
        .canonical_checkpoint => "--drafter checkpoint contents",
        .canonical_bits => "--drafter effective precision (disabled/Q4/BF16)",
        .dflash_verify => "TF_GLM_DFLASH_VERIFY_DRAFTS",
        .draft_profile => "TF_GLM_DRAFT_PROFILE and production copy settings",
        .vision_setting => "TF_GLM_VISION",
        .vision => "--vision image-token limit",
        .video_tokens => "--vision-video-tokens",
        .max_images => "the image count limit",
        .max_videos => "the video count limit",
        .max_frames => "the frame limit",
        .mtp => "--no-drafts or --drafter given to only some ranks",
        .bits => "TF_GLM_DENSE_BITS or TF_GLM_HEAD_BITS",
        .kv_pool_tokens => "TF_GLM_KV_POOL_TOKENS",
        .kv_shard => "TF_GLM_KV_SHARD",
        .spill => "TENSORFOLD_SPILL settings",
        .capacity => "the sequence capacity (the context)",
        .heat_high => "TF_HEAT_HIGH",
        .heat_low => "TF_HEAT_LOW",
        .config => "config.json",
        .program => "the executable",
        .checkpoint => "checkpoint metadata or shard bytes",
    };
}

test {
    _ = @import("cuda_ranks_test.zig");
}

test "WY prompt selection validates the environment and separates ranks and persisted numerics" {
    const Env = struct {
        fn off(name: [:0]const u8) ?[]const u8 {
            return if (std.mem.eql(u8, name, "TF_GLM_KDA_CHUNKED")) "0" else null;
        }
        fn on(name: [:0]const u8) ?[]const u8 {
            return if (std.mem.eql(u8, name, "TF_GLM_KDA_CHUNKED")) "1" else null;
        }
        fn invalid(name: [:0]const u8) ?[]const u8 {
            return if (std.mem.eql(u8, name, "TF_GLM_KDA_CHUNKED")) "true" else null;
        }
    };
    var problem: []const u8 = "";
    const off = try Settings.read(Env.off, &problem);
    const on = try Settings.read(Env.on, &problem);
    try std.testing.expectEqual(kda.PromptAlgorithm.serial, (Settings{}).kda_prompt);
    try std.testing.expectEqual(kda.PromptAlgorithm.serial, off.kda_prompt);
    try std.testing.expectEqual(kda.PromptAlgorithm.wy32, on.kda_prompt);
    try std.testing.expectError(error.BadSetting, Settings.read(Env.invalid, &problem));
    try std.testing.expectEqualStrings("TF_GLM_KDA_CHUNKED: 0 or 1", problem);
    const h: [32]u8 = @splat(0);
    const all = [_]Fingerprint{ Fingerprint.of(2, 0, off, 8192, true, h, h, hashBytes("target")), Fingerprint.of(2, 1, on, 8192, true, h, h, hashBytes("target")) };
    try std.testing.expectEqual(Fingerprint.Field.kda_prompt, mismatch(&all).?.field);
    var a = Sha256.init(.{});
    var b = Sha256.init(.{});
    var c = Sha256.init(.{});
    off.hashNumerics(&a);
    on.hashNumerics(&b);
    (Settings{}).hashNumerics(&c);
    const serial = a.finalResult();
    const wy = b.finalResult();
    const defaults = c.finalResult();
    try std.testing.expectEqualSlices(u8, &serial, &defaults);
    try std.testing.expect(!std.mem.eql(u8, &serial, &wy));
}

test "Q4 prompt tiles are opt in, strict, rank agreed and cache isolated" {
    const Env = struct {
        fn off(name: [:0]const u8) ?[]const u8 {
            return if (std.mem.eql(u8, name, "TF_GLM_PROMPT_TILE")) "0" else null;
        }
        fn on(name: [:0]const u8) ?[]const u8 {
            return if (std.mem.eql(u8, name, "TF_GLM_PROMPT_TILE")) "1" else null;
        }
        fn invalid(name: [:0]const u8) ?[]const u8 {
            return if (std.mem.eql(u8, name, "TF_GLM_PROMPT_TILE")) "true" else null;
        }
    };
    var problem: []const u8 = "";
    const off = try Settings.read(Env.off, &problem);
    const on = try Settings.read(Env.on, &problem);
    try std.testing.expectEqual(cuda.qlinear.AffinePromptPolicy.stock, (Settings{}).affine_prompt_policy);
    try std.testing.expectEqual(cuda.qlinear.AffinePromptPolicy.stock, off.affine_prompt_policy);
    try std.testing.expectEqual(cuda.qlinear.AffinePromptPolicy.glm_production, on.affine_prompt_policy);
    try std.testing.expectError(error.BadSetting, Settings.read(Env.invalid, &problem));
    try std.testing.expectEqualStrings("TF_GLM_PROMPT_TILE: 0 or 1", problem);
    const h: [32]u8 = @splat(0);
    const all = [_]Fingerprint{ Fingerprint.of(2, 0, off, 8192, true, h, h, hashBytes("target")), Fingerprint.of(2, 1, on, 8192, true, h, h, hashBytes("target")) };
    try std.testing.expectEqual(Fingerprint.Field.affine_prompt_policy, mismatch(&all).?.field);
    var a = Sha256.init(.{});
    var b = Sha256.init(.{});
    var c = Sha256.init(.{});
    off.hashNumerics(&a);
    on.hashNumerics(&b);
    (Settings{}).hashNumerics(&c);
    const stock = a.finalResult();
    const production = b.finalResult();
    const defaults = c.finalResult();
    try std.testing.expectEqualSlices(u8, &stock, &defaults);
    try std.testing.expect(!std.mem.eql(u8, &stock, &production));
}

test "prompt overlap levels and their separate piece count are strict and rank agreed" {
    const Env = struct {
        fn on(name: [:0]const u8) ?[]const u8 {
            return if (std.mem.eql(u8, name, "TF_GLM_PREFILL_OVERLAP")) "2" else null;
        }
        fn bad(name: [:0]const u8) ?[]const u8 {
            return if (std.mem.eql(u8, name, "TF_GLM_PREFILL_OVERLAP")) "3" else null;
        }
        fn badPieces(name: [:0]const u8) ?[]const u8 {
            return if (std.mem.eql(u8, name, "TF_GLM_PREFILL_OVERLAP_PIECES")) "0" else null;
        }
    };
    var problem: []const u8 = "";
    const on = try Settings.read(Env.on, &problem);
    try std.testing.expectEqual(@as(u8, 0), (Settings{}).prefill_overlap);
    try std.testing.expectEqual(@as(u8, 2), on.prefill_overlap);
    try std.testing.expectEqual(@as(u32, 2), on.prefill_pieces);
    try std.testing.expectEqual(@as(u32, 3), on.pieces);
    try std.testing.expectError(error.BadSetting, Settings.read(Env.bad, &problem));
    try std.testing.expectError(error.BadSetting, Settings.read(Env.badPieces, &problem));
    const h: [32]u8 = @splat(0);
    var all = [_]Fingerprint{ Fingerprint.of(2, 0, .{}, 8192, true, h, h, hashBytes("target")), Fingerprint.of(2, 1, on, 8192, true, h, h, hashBytes("target")) };
    try std.testing.expectEqual(Fingerprint.Field.prefill_overlap, mismatch(&all).?.field);
    all[0] = Fingerprint.of(2, 0, on, 8192, true, h, h, hashBytes("target"));
    all[1].prefill_pieces = 3;
    try std.testing.expectEqual(Fingerprint.Field.prefill_pieces, mismatch(&all).?.field);
}

test "prompt K experiment is explicit and all ranks must agree" {
    const Reader = struct {
        fn absent(_: [:0]const u8) ?[]const u8 {
            return null;
        }
        fn one(key: [:0]const u8) ?[]const u8 {
            return if (std.mem.eql(u8, key, "TF_GLM_PROMPT_K")) "one" else null;
        }
        fn untiled(key: [:0]const u8) ?[]const u8 {
            return if (std.mem.eql(u8, key, "TF_GLM_PROMPT_K")) "shape" else if (std.mem.eql(u8, key, "TF_GLM_MM")) "0" else null;
        }

        fn shape(key: [:0]const u8) ?[]const u8 {
            return if (std.mem.eql(u8, key, "TF_GLM_PROMPT_K")) "shape" else null;
        }
        fn bad(key: [:0]const u8) ?[]const u8 {
            return if (std.mem.eql(u8, key, "TF_GLM_PROMPT_K")) "auto" else null;
        }
    };
    var problem: []const u8 = "";
    const selected = try Settings.read(Reader.shape, &problem);
    try std.testing.expect(selected.prompt_fixed_k);
    try std.testing.expect(!(try Settings.read(Reader.absent, &problem)).prompt_fixed_k);
    try std.testing.expect(!(try Settings.read(Reader.one, &problem)).prompt_fixed_k);
    try std.testing.expectError(error.BadSetting, Settings.read(Reader.untiled, &problem));
    try std.testing.expectError(error.BadSetting, Settings.read(Reader.bad, &problem));
    // Compare two parsed settings: read also fingerprints the absent spill environment.
    const absent = try Settings.read(Reader.absent, &problem);
    var all = [_]Fingerprint{ Fingerprint.of(2, 0, absent, 8192, true, @splat(0), @splat(0), @splat(0)), Fingerprint.of(2, 1, selected, 8192, true, @splat(0), @splat(0), @splat(0)) };
    try std.testing.expectEqual(Fingerprint.Field.prompt_fixed_k, mismatch(&all).?.field);
    all[0].prompt_fixed_k = 1;
    try std.testing.expectEqual(@as(?Mismatch, null), mismatch(&all));
}

/// Before loading: direct split reaches every rank, so each pair is probed; the ring only passes rows to neighbours.
pub fn probeStart(d: *const cuda.Driver, lib: *const cuda.nccl.Library, comm: cuda.nccl.Comm, stream: cuda.Stream, s: Settings, rank: usize, world: usize) !void {
    if (s.split and !s.ring) try probeSplit(d, lib, comm, stream, rank, world);
}

/// Before loading, split's peers: one value each way with each, in rank order (no pair waits on a later one).
fn probeSplit(d: *const cuda.Driver, lib: *const cuda.nccl.Library, comm: cuda.nccl.Comm, stream: cuda.Stream, rank: usize, world: usize) !void {
    var dev = try cuda.DeviceBuffer.alloc(d, 8);
    defer dev.free();
    for (0..world) |peer| if (peer != rank) {
        try lib.check(lib.api.ncclGroupStart(), "ncclGroupStart");
        try lib.check(lib.api.ncclSend(dev.ptr, 1, .f32, @intCast(peer), comm, stream.handle), "ncclSend");
        try lib.check(lib.api.ncclRecv(dev.ptr + 4, 1, .f32, @intCast(peer), comm, stream.handle), "ncclRecv");
        lib.check(lib.api.ncclGroupEnd(), "ncclGroupEnd") catch return unreachablePeer(rank, peer);
        stream.synchronize() catch return unreachablePeer(rank, peer);
    };
}

fn unreachablePeer(rank: usize, peer: usize) error{SplitUnreachable} {
    std.debug.print("REFUSE TF_GLM_PREFILL_REDUCE=split: rank {d} cannot exchange with rank {d}; split sends every rank's rows to every other, and a fabric where some ranks reach only their neighbours (a ring of four without a switch) needs TF_GLM_PREFILL_REDUCE=ring or gather\n", .{ rank, peer });
    return error.SplitUnreachable;
}

fn fake(name: [:0]const u8) ?[]const u8 {
    if (std.mem.eql(u8, name, "TF_GLM_PREFILL_REDUCE")) return "gather";
    if (std.mem.eql(u8, name, "TF_GLM_OVERLAP_PIECES")) return "2";
    if (std.mem.eql(u8, name, "TF_HEAT_HIGH")) return "92";
    if (std.mem.eql(u8, name, "TF_HEAT_LOW")) return "88";
    if (std.mem.eql(u8, name, "TF_GLM_DENSE_BITS")) return "4";
    if (std.mem.eql(u8, name, "TF_GLM_HEAD_BITS")) return "8";
    return null;
}

fn ringed(name: [:0]const u8) ?[]const u8 {
    if (std.mem.eql(u8, name, "TF_GLM_PREFILL_REDUCE")) return "ring";
    return null;
}

fn bad(name: [:0]const u8) ?[]const u8 {
    if (std.mem.eql(u8, name, "TF_GLM_PREFILL_REDUCE")) return "sum";
    return null;
}

fn typo(name: [:0]const u8) ?[]const u8 {
    if (std.mem.eql(u8, name, "TF_GLM_DENSE_BITS")) return "abc";
    return null;
}

test "settings from the environment: the defaults, the choices, and a refusal naming a bad value" {
    var problem: []const u8 = "";
    const s = try Settings.read(&fake, &problem);
    try std.testing.expect(!s.split and s.kda_wide and s.pieces == 2 and s.heat.?.high_c == 92 and s.heat.?.low_c == 88);
    try std.testing.expect(s.dense_bits == 4 and s.head_bits == 8);
    try std.testing.expectError(error.BadSetting, Settings.read(&bad, &problem));
    try std.testing.expectEqualStrings("TF_GLM_PREFILL_REDUCE: split, ring or gather", problem);
    const r = try Settings.read(&ringed, &problem);
    try std.testing.expect(r.split and r.ring);
    try std.testing.expectEqual(@as(u32, 2), Fingerprint.of(2, 0, r, 8192, true, @splat(0), @splat(0), hashBytes("/c")).split);

    try std.testing.expectError(error.BadSetting, Settings.read(&typo, &problem)); // a typo is refused, not read as bf16
    try std.testing.expectEqualStrings("TF_GLM_DENSE_BITS: 4, 6, 8 or 16", problem);
}

test "fingerprints: agreeing ranks pass; the first differing field and rank are named" {
    const h0: [32]u8 = @splat(1);
    const h1: [32]u8 = @splat(2);
    var all = [_]Fingerprint{
        Fingerprint.of(2, 0, .{}, 8192, true, h0, h0, hashBytes("/hub/snapshots/423acf37")),
        Fingerprint.of(2, 1, .{}, 8192, true, h0, h0, hashBytes("/hub/snapshots/423acf37")),
    };
    try std.testing.expect(mismatch(&all) == null);
    all[1].program = h1;
    try std.testing.expectEqual(Fingerprint.Field.program, mismatch(&all).?.field);
    all[1] = Fingerprint.of(2, 1, .{ .split = false }, 8192, true, h0, h0, hashBytes("/hub/snapshots/423acf37"));
    try std.testing.expectEqual(Mismatch{ .field = .split, .rank = 1 }, mismatch(&all).?);
    all[1] = Fingerprint.of(2, 0, .{}, 8192, true, h0, h0, hashBytes("/hub/snapshots/423acf37"));
    try std.testing.expectEqual(Fingerprint.Field.rank, mismatch(&all).?.field);
    const heat_on: Settings = .{ .heat = .{ .high_c = 92, .low_c = 88 } };
    all[1] = Fingerprint.of(2, 1, heat_on, 8192, true, h0, h0, hashBytes("/hub/snapshots/423acf37"));
    try std.testing.expectEqual(Fingerprint.Field.heat_high, mismatch(&all).?.field);
    all[1] = Fingerprint.of(2, 1, .{}, 8192, true, h0, h0, hashBytes("/hub/snapshots/other"));
    try std.testing.expectEqual(Fingerprint.Field.checkpoint, mismatch(&all).?.field);
    all[1] = Fingerprint.of(2, 1, .{}, 8192, false, h0, h0, hashBytes("/hub/snapshots/423acf37")); // --no-drafts on rank 1 only
    try std.testing.expectEqual(Mismatch{ .field = .mtp, .rank = 1 }, mismatch(&all).?);
    all[1] = Fingerprint.of(2, 1, .{}, 8192, true, h0, h0, hashBytes("/hub/snapshots/423acf37"));
    all[1].vision = 4096; // --vision on rank 1 only
    try std.testing.expectEqual(Mismatch{ .field = .vision, .rank = 1 }, mismatch(&all).?);
    all[1] = Fingerprint.of(2, 1, .{ .dense_bits = 4 }, 8192, true, h0, h0, hashBytes("/hub/snapshots/423acf37")); // packed on rank 1 only
    try std.testing.expectEqual(Mismatch{ .field = .bits, .rank = 1 }, mismatch(&all).?);
}

/// A world seen from `rank`: a group that sends to or receives from a rank it cannot reach fails at its end.
const Fabric = struct {
    var world: usize = 4;
    var rank: usize = 0;
    var switched = false; // every pair reaches the other; else a ring without a switch, each rank its two neighbours
    var failing = false;
    var posted: usize = 0;
    var allocated: usize = 0;

    fn reaches(peer: c_int) bool {
        const p: usize = @intCast(peer);
        return switched or (p + 1) % world == rank or (rank + 1) % world == p;
    }
    fn post(peer: c_int) cuda.nccl.Result {
        posted += 1;
        if (!reaches(peer)) failing = true;
        return 0;
    }
    fn start() callconv(.c) cuda.nccl.Result {
        failing = false;
        return 0;
    }
    fn end() callconv(.c) cuda.nccl.Result {
        return @intFromBool(failing);
    }
    fn send(_: cuda.abi.DevicePtr, _: usize, _: cuda.nccl.DataType, peer: c_int, _: cuda.nccl.Comm, _: cuda.abi.Stream) callconv(.c) cuda.nccl.Result {
        return post(peer);
    }
    fn recv(_: cuda.abi.DevicePtr, _: usize, _: cuda.nccl.DataType, peer: c_int, _: cuda.nccl.Comm, _: cuda.abi.Stream) callconv(.c) cuda.nccl.Result {
        return post(peer);
    }
    fn alloc(p: *cuda.abi.DevicePtr, _: usize) callconv(.c) cuda.abi.Result {
        allocated += 1;
        p.* = 0x10000;
        return cuda.abi.success;
    }
    fn free(_: cuda.abi.DevicePtr) callconv(.c) cuda.abi.Result {
        allocated -= 1;
        return cuda.abi.success;
    }
    fn sync(_: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        return cuda.abi.success;
    }
    /// One rank's start probes on this fabric, with `s`; the sends and receives it posted.
    fn probe(s: Settings, r: usize, w: usize, all: bool) !usize {
        var api: cuda.abi.Api = undefined;
        api.cuMemAlloc_v2 = alloc;
        api.cuMemFree_v2 = free;
        api.cuStreamSynchronize = sync;
        const d: cuda.Driver = .{ .lib = undefined, .api = api };
        var nccl_api: cuda.nccl.Api = undefined;
        nccl_api.ncclGroupStart = &start;
        nccl_api.ncclGroupEnd = &end;
        nccl_api.ncclSend = &send;
        nccl_api.ncclRecv = &recv;
        const lib: cuda.nccl.Library = .{ .lib = undefined, .api = nccl_api };
        world = w;
        rank = r;
        switched = all;
        posted = 0;
        const got = probeStart(&d, &lib, null, .{ .d = &d, .handle = null }, s, r, w);
        try std.testing.expectEqual(@as(usize, 0), allocated); // the probe's buffer is freed, refused or not
        try got;
        return posted;
    }
};

test "a direct split start on a ring of four without a switch is refused before loading, at every rank" {
    for (0..4) |r| try std.testing.expectError(error.SplitUnreachable, Fabric.probe(.{}, r, 4, false));
    // the same start on four ranks behind a switch probes each pair once each way
    for (0..4) |r| try std.testing.expectEqual(@as(usize, 6), try Fabric.probe(.{}, r, 4, true));
    // two ranks are each other's neighbours: a direct split starts on a ring of two
    for (0..2) |r| try std.testing.expectEqual(@as(usize, 2), try Fabric.probe(.{}, r, 2, false));
}

test "a ring start on four ranks that reach only their neighbours loads without the all-peer probe" {
    var problem: []const u8 = "";
    const ring = try Settings.read(&ringed, &problem);
    for (0..4) |r| try std.testing.expectEqual(@as(usize, 0), try Fabric.probe(ring, r, 4, false));
    // gather needs no probe either
    for (0..4) |r| try std.testing.expectEqual(@as(usize, 0), try Fabric.probe(.{ .split = false }, r, 4, false));
}

pub fn mediaFingerprint(f: *Fingerprint, limits: @import("cuda_images.zig").Limits) void {
    f.vision = limits.image_tokens;
    f.video_tokens = limits.video_tokens;
    f.max_images = limits.max_images;
    f.max_videos = limits.max_videos;
    f.max_frames = limits.max_frames;
}
test "every media limit is part of the ranks' agreement" {
    const L = @import("cuda_images.zig").Limits;
    const base: L = .{ .image_tokens = 2048, .max_images = 8, .video_tokens = 4096, .max_videos = 2, .max_frames = 256 };
    inline for (.{ "video_tokens", "max_images", "max_videos", "max_frames" }) |field| {
        var all = [_]Fingerprint{
            Fingerprint.of(2, 0, .{}, 8192, true, @splat(0), @splat(0), hashBytes("/same")),
            Fingerprint.of(2, 1, .{}, 8192, true, @splat(0), @splat(0), hashBytes("/same")),
        };
        mediaFingerprint(&all[0], base);
        mediaFingerprint(&all[1], base);
        try std.testing.expect(mismatch(&all) == null);
        var changed = base;
        @field(changed, field) += 1;
        mediaFingerprint(&all[1], changed);
        try std.testing.expect(mismatch(&all) != null);
        try std.testing.expectEqual(@as(usize, 1), mismatch(&all).?.rank);
    }
}

test "every scheduling and identity field that differs stops the ranks" {
    const zero = Fingerprint.of(2, 0, .{}, 8192, true, @splat(0), @splat(0), hashBytes("/same"));
    const one = Fingerprint.of(2, 1, .{}, 8192, true, @splat(0), @splat(0), hashBytes("/same"));
    inline for (.{ "layout", "world", "rank", "split", "kda_wide", "pieces", "layers", "mtp", "vision", "video_tokens", "max_images", "max_videos", "max_frames", "capacity", "heat_high", "heat_low" }) |name| {
        var all = [_]Fingerprint{ zero, one };
        @field(all[1], name) += 1;
        try std.testing.expectEqual(@field(Fingerprint.Field, name), mismatch(&all).?.field);
        std.mem.swap(Fingerprint, &all[0], &all[1]);
        all[0].rank = 0;
        all[1].rank = if (std.mem.eql(u8, name, "rank")) 0 else 1;
        try std.testing.expect(mismatch(&all) != null);
    }
    inline for (.{ "config", "program", "checkpoint" }) |name| {
        var all = [_]Fingerprint{ zero, one };
        @field(all[1], name)[0] = 1;
        try std.testing.expectEqual(@field(Fingerprint.Field, name), mismatch(&all).?.field);
    }
}

/// The checkpoint digest a rank sends: its metadata and every shard byte, read again only for a file that changed.
pub fn artifact(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) ![32]u8 {
    return @import("core").checkpoint_identity.fullIdentity(gpa, io, dir);
}

test "ranks with different TF_GLM_MM choices disagree" {
    const h: [32]u8 = @splat(0);
    const all = [_]Fingerprint{ Fingerprint.of(2, 0, .{}, 8192, true, h, h, hashBytes("/model")), Fingerprint.of(2, 1, .{ .tiled = false }, 8192, true, h, h, hashBytes("/model")) };
    try std.testing.expect(mismatch(&all) != null);
}
fn digestFixture(io: std.Io, dir: std.Io.Dir, sub: []const u8, byte: u8) !void {
    try dir.createDirPath(io, sub);
    var d = try dir.openDir(io, sub, .{});
    defer d.close(io);
    try d.writeFile(io, .{ .sub_path = "config.json", .data = "{}" });
    const header = "{\"dense\":{\"dtype\":\"U8\",\"shape\":[4],\"data_offsets\":[0,4]}}";
    var bytes: [8 + header.len + 4]u8 = undefined;
    std.mem.writeInt(u64, bytes[0..8], header.len, .little);
    @memcpy(bytes[8..][0..header.len], header);
    @memset(bytes[8 + header.len ..], byte);
    try d.writeFile(io, .{ .sub_path = "model.safetensors", .data = &bytes });
}

test "ranks disagree on tensor bytes under identical headers, and agree on a copy in a folder of the same name" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try digestFixture(io, tmp.dir, "a/model", 1);
    try digestFixture(io, tmp.dir, "b/model", 2);
    try digestFixture(io, tmp.dir, "c/model", 1);
    var paths: [3][:0]u8 = undefined;
    for ([_][]const u8{ "a/model", "b/model", "c/model" }, &paths) |sub, *p| p.* = try tmp.dir.realPathFileAlloc(io, sub, a);
    defer for (paths) |p| a.free(p);
    const h: [32]u8 = @splat(7);
    var all = [_]Fingerprint{ Fingerprint.of(2, 0, .{}, 8192, true, h, h, try artifact(a, io, paths[0])), Fingerprint.of(2, 1, .{}, 8192, true, h, h, try artifact(a, io, paths[1])) };
    try std.testing.expectEqual(Mismatch{ .field = .checkpoint, .rank = 1 }, mismatch(&all).?);
    all[1] = Fingerprint.of(2, 1, .{}, 8192, true, h, h, try artifact(a, io, paths[2]));
    try std.testing.expect(mismatch(&all) == null);
}

test "canonical option identity reads checkpoint content and effective loader precision independently of legacy settings" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try digestFixture(io, tmp.dir, "a/drafter", 1);
    try digestFixture(io, tmp.dir, "b/drafter", 2);
    try digestFixture(io, tmp.dir, "c/drafter", 1);
    var paths: [3][:0]u8 = undefined;
    for ([_][]const u8{ "a/drafter", "b/drafter", "c/drafter" }, &paths) |sub, *p| p.* = try tmp.dir.realPathFileAlloc(io, sub, a);
    defer for (paths) |p| a.free(p);
    const identity = try CanonicalDraft.read(a, io, paths[0], 4);
    const different = try CanonicalDraft.read(a, io, paths[1], 4);
    const copied = try CanonicalDraft.read(a, io, paths[2], 4);
    var all = [_]Fingerprint{
        Fingerprint.of(2, 0, .{}, 8192, true, @splat(0), @splat(0), hashBytes("target")),
        Fingerprint.of(2, 1, .{}, 8192, true, @splat(0), @splat(0), hashBytes("target")),
    };
    identity.fingerprint(&all[0]);
    copied.fingerprint(&all[1]);
    try std.testing.expect(mismatch(&all) == null);
    different.fingerprint(&all[1]);
    try std.testing.expectEqual(Fingerprint.Field.canonical_checkpoint, mismatch(&all).?.field);
    const bf16 = try CanonicalDraft.read(a, io, paths[0], 0);
    try std.testing.expectEqual(@as(u32, 16), bf16.bits);
    bf16.fingerprint(&all[1]);
    try std.testing.expectEqual(Fingerprint.Field.canonical_bits, mismatch(&all).?.field);
    try std.testing.expectEqual(@as(u32, 4), identity.bits);
    // CLI absence is different from the BF16 option, even though both accept raw bits zero.
    const off = try CanonicalDraft.read(a, io, null, 0);
    try std.testing.expectEqual(@as(u32, 0), off.bits);
    try std.testing.expectError(error.BadDrafterBits, CanonicalDraft.read(a, io, paths[0], 8));
    try std.testing.expectEqualSlices(u8, &(@as([32]u8, @splat(0))), &all[0].draft_checkpoint);
}

test "canonical digest read failure enters the actual startup vote and refuses the healthy peer before execution" {
    const M = @import("cuda_test_driver.zig").Memory;
    const Recording = struct {
        var votes: [2]u32 = .{ 1, 1 };
        var rank: usize = 0;
        var calls: usize = 0;
        fn allGather(src: u64, dst: u64, n: usize, dtype: cuda.nccl.DataType, _: cuda.nccl.Comm, _: cuda.abi.Stream) callconv(.c) cuda.nccl.Result {
            std.debug.assert(n == 4 and dtype == .u8);
            const mine: *const u32 = @ptrFromInt(src);
            std.debug.assert(mine.* == votes[rank]);
            @memcpy(M.slice(dst, 8), std.mem.sliceAsBytes(&votes));
            calls += 1;
            return 0;
        }
    };
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try tmp.dir.realPathFileAlloc(std.testing.io, ".", std.testing.allocator);
    defer std.testing.allocator.free(path);
    var problem: ?anyerror = null;
    _ = CanonicalDraft.read(std.testing.allocator, std.testing.io, path, 4) catch |err| blk: {
        problem = err;
        break :blk CanonicalDraft{};
    };
    try std.testing.expect(problem != null); // no config/checkpoint, not an injected success flag
    var d = M.driver();
    var lib: cuda.nccl.Library = .{ .lib = undefined, .api = undefined };
    lib.api.ncclAllGather = Recording.allGather;
    const stream: cuda.Stream = .{ .d = &d, .handle = null };
    Recording.calls = 0;
    for (0..2) |failed_rank| {
        Recording.votes = .{ 1, 1 };
        Recording.votes[failed_rank] = 0;
        for (0..2) |rank| {
            Recording.rank = rank;
            var all: [2]u32 = undefined;
            const local = if (rank == failed_rank) problem else null;
            try std.testing.expectError(local orelse error.PeerStartupRefused, startupVote(&d, &lib, null, stream, rank, local, &all));
            try std.testing.expectEqualSlices(u32, &Recording.votes, &all);
        }
    }
    Recording.votes = .{ 1, 1 };
    for (0..2) |rank| {
        Recording.rank = rank;
        var all: [2]u32 = undefined;
        try startupVote(&d, &lib, null, stream, rank, null, &all);
    }
    try std.testing.expectEqual(@as(usize, 6), Recording.calls);
    try std.testing.expect(M.empty());
}

test "production canonical option orchestration votes digest failure stamps submitted identity and refuses mismatched peers before execution" {
    const M = @import("cuda_test_driver.zig").Memory;
    const Recording = struct {
        var votes: [2]u32 = .{ 1, 1 };
        var fps: [2]Fingerprint = undefined;
        var rank: usize = 0;
        var votes_seen: usize = 0;
        var agrees: usize = 0;
        var submitted: Fingerprint = undefined;
        fn allGather(src: u64, dst: u64, n: usize, dtype: cuda.nccl.DataType, _: cuda.nccl.Comm, _: cuda.abi.Stream) callconv(.c) cuda.nccl.Result {
            std.debug.assert(dtype == .u8);
            if (n == 4) {
                const mine: *const u32 = @ptrFromInt(src);
                std.debug.assert(mine.* == votes[rank]);
                @memcpy(M.slice(dst, 8), std.mem.sliceAsBytes(&votes));
                votes_seen += 1;
                return 0;
            }
            std.debug.assert(n == @sizeOf(Fingerprint));
            submitted = std.mem.bytesToValue(Fingerprint, M.slice(src, n)[0..@sizeOf(Fingerprint)]);
            std.debug.assert(std.mem.eql(u8, std.mem.asBytes(&submitted), std.mem.asBytes(&fps[rank])));
            @memcpy(M.slice(dst, n * 2), std.mem.sliceAsBytes(&fps));
            agrees += 1;
            return 0;
        }
    };
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try digestFixture(io, tmp.dir, "a/drafter", 1);
    try digestFixture(io, tmp.dir, "b/drafter", 2);
    var paths: [2][:0]u8 = undefined;
    for ([_][]const u8{ "a/drafter", "b/drafter" }, &paths) |sub, *p| p.* = try tmp.dir.realPathFileAlloc(io, sub, a);
    defer for (paths) |p| a.free(p);
    var d = M.driver();
    var lib: cuda.nccl.Library = .{ .lib = undefined, .api = undefined };
    lib.api.ncclAllGather = Recording.allGather;
    const stream: cuda.Stream = .{ .d = &d, .handle = null };
    const zero: [32]u8 = @splat(0);

    const settings: Settings = .{};
    const healthy = try CanonicalDraft.read(a, io, paths[0], 4);
    const other = try CanonicalDraft.read(a, io, paths[1], 4);
    const bf16 = try CanonicalDraft.read(a, io, paths[0], 0);

    // Digest failure: Engine.init's applyCanonicalStartup votes before agree.
    {
        const missing = try tmp.dir.realPathFileAlloc(io, ".", a);
        defer a.free(missing);
        Recording.votes = .{ 0, 1 };
        Recording.votes_seen = 0;
        Recording.agrees = 0;
        for (0..2) |rank| {
            Recording.rank = rank;
            var problem: ?anyerror = null;
            var fp = Fingerprint.of(2, rank, settings, 8192, true, zero, zero, hashBytes("target"));
            var vote: [2]u32 = undefined;
            var all: [2]Fingerprint = undefined;
            const opts: CanonicalOptions = .{ .dir = if (rank == 0) missing else paths[0], .bits = 4, .drafts = true };
            const err = applyCanonicalStartup(a, io, &d, &lib, null, stream, rank, opts, true, &problem, &fp, &vote, &all);
            if (rank == 0) try std.testing.expect(problem != null) else try std.testing.expect(problem == null);
            try std.testing.expectError(if (rank == 0) problem.? else error.PeerStartupRefused, err);
        }
        try std.testing.expectEqual(@as(usize, 2), Recording.votes_seen);
        try std.testing.expectEqual(@as(usize, 0), Recording.agrees);
    }

    // Drafts mismatch is voted before identity exchange.
    {
        Recording.votes = .{ 0, 0 };
        Recording.votes_seen = 0;
        Recording.agrees = 0;
        Recording.rank = 0;
        var problem: ?anyerror = null;
        var fp = Fingerprint.of(2, 0, settings, 8192, true, zero, zero, hashBytes("target"));
        var vote: [2]u32 = undefined;
        var all: [2]Fingerprint = undefined;
        try std.testing.expectError(error.DrafterNeedsDrafts, applyCanonicalStartup(a, io, &d, &lib, null, stream, 0, .{ .dir = paths[0], .bits = 4, .drafts = false }, true, &problem, &fp, &vote, &all));
        try std.testing.expectEqual(@as(?anyerror, error.DrafterNeedsDrafts), problem);
        try std.testing.expectEqual(@as(usize, 1), Recording.votes_seen);
        try std.testing.expectEqual(@as(usize, 0), Recording.agrees);
    }

    // Content mismatch: votes pass, submitted fingerprints disagree, agree refuses before execution.
    {
        Recording.votes = .{ 1, 1 };
        Recording.votes_seen = 0;
        Recording.agrees = 0;
        const identities: [2]CanonicalDraft = .{ healthy, other };
        for (0..2) |rank| {
            Recording.fps[rank] = Fingerprint.of(2, rank, settings, 8192, true, zero, zero, hashBytes("target"));
            stampCanonical(&Recording.fps[rank], identities[rank], true);
        }
        try std.testing.expectEqual(Fingerprint.Field.canonical_checkpoint, mismatch(&Recording.fps).?.field);
        for (0..2) |rank| {
            Recording.rank = rank;
            var problem: ?anyerror = null;
            var fp = Fingerprint.of(2, rank, settings, 8192, true, zero, zero, hashBytes("target"));
            var vote: [2]u32 = undefined;
            var all: [2]Fingerprint = undefined;
            try std.testing.expectError(error.RankMismatch, applyCanonicalStartup(a, io, &d, &lib, null, stream, rank, .{ .dir = paths[rank], .bits = 4, .drafts = true }, true, &problem, &fp, &vote, &all));
            try std.testing.expect(problem == null);
            try std.testing.expectEqualSlices(u8, std.mem.asBytes(&Recording.fps[rank]), std.mem.asBytes(&fp));
        }
        try std.testing.expectEqual(@as(usize, 2), Recording.votes_seen);
        try std.testing.expectEqual(@as(usize, 2), Recording.agrees);
    }

    // Precision mismatch uses the same production stamp.
    {
        Recording.fps[0] = Fingerprint.of(2, 0, settings, 8192, true, zero, zero, hashBytes("target"));
        Recording.fps[1] = Fingerprint.of(2, 1, settings, 8192, true, zero, zero, hashBytes("target"));
        stampCanonical(&Recording.fps[0], healthy, true);
        stampCanonical(&Recording.fps[1], bf16, true);
        try std.testing.expectEqual(Fingerprint.Field.canonical_bits, mismatch(&Recording.fps).?.field);
        Recording.rank = 1;
        Recording.agrees = 0;
        var problem: ?anyerror = null;
        var fp = Fingerprint.of(2, 1, settings, 8192, true, zero, zero, hashBytes("target"));
        var vote: [2]u32 = undefined;
        var all: [2]Fingerprint = undefined;
        try std.testing.expectError(error.RankMismatch, applyCanonicalStartup(a, io, &d, &lib, null, stream, 1, .{ .dir = paths[0], .bits = 0, .drafts = true }, true, &problem, &fp, &vote, &all));
        try std.testing.expectEqual(@as(u32, 16), fp.canonical_bits);
        try std.testing.expectEqual(@as(usize, 1), Recording.agrees);
    }

    // Matching content and precision: submitted fingerprint is the agreed identity.
    {
        Recording.votes_seen = 0;
        Recording.agrees = 0;
        for (0..2) |rank| {
            Recording.fps[rank] = Fingerprint.of(2, rank, settings, 8192, true, zero, zero, hashBytes("target"));
            stampCanonical(&Recording.fps[rank], healthy, true);
        }
        try std.testing.expect(mismatch(&Recording.fps) == null);
        for (0..2) |rank| {
            Recording.rank = rank;
            var problem: ?anyerror = null;
            var fp = Fingerprint.of(2, rank, settings, 8192, true, zero, zero, hashBytes("target"));
            var vote: [2]u32 = undefined;
            var all: [2]Fingerprint = undefined;
            _ = try applyCanonicalStartup(a, io, &d, &lib, null, stream, rank, .{ .dir = paths[0], .bits = 4, .drafts = true }, true, &problem, &fp, &vote, &all);
            try std.testing.expectEqual(@as(u32, 4), all[rank].canonical_bits);
            try std.testing.expectEqualSlices(u8, &healthy.checkpoint, &all[rank].canonical_checkpoint);
        }
        try std.testing.expectEqual(@as(usize, 2), Recording.votes_seen);
        try std.testing.expectEqual(@as(usize, 2), Recording.agrees);
    }
    try std.testing.expect(M.empty());
}

test "saved-state identity uses canonical content and precision, not the wire layout epoch" {
    const a = CanonicalDraft{ .checkpoint = hashBytes("drafter-a"), .bits = 4 };
    const b = CanonicalDraft{ .checkpoint = hashBytes("drafter-b"), .bits = 4 };
    const bf16_bits = CanonicalDraft{ .checkpoint = hashBytes("drafter-a"), .bits = 16 };
    const empty = CanonicalDraft{};
    try std.testing.expect(!std.mem.eql(u8, &a.snapshotIdentity(), &b.snapshotIdentity()));
    try std.testing.expect(!std.mem.eql(u8, &a.snapshotIdentity(), &bf16_bits.snapshotIdentity()));
    try std.testing.expect(!std.mem.eql(u8, &a.snapshotIdentity(), &empty.snapshotIdentity()));
    const legacy: [32]u8 = @splat(7);
    const owned = persistedDraftSnapshot(true, a, 4096, 22_282_240, legacy);
    try std.testing.expectEqual(@as(u64, 4096), owned.bytes);
    try std.testing.expectEqualSlices(u8, &a.snapshotIdentity(), &owned.identity);
    const env = persistedDraftSnapshot(false, a, 4096, 22_282_240, legacy);
    try std.testing.expectEqual(@as(u64, 22_282_240), env.bytes);
    try std.testing.expectEqualSlices(u8, &legacy, &env.identity);
    var old = Sha256.init(.{});
    old.update("glm-dflash2-saved-v1\x00");
    old.update(&a.checkpoint);
    var bits: [4]u8 = undefined;
    std.mem.writeInt(u32, &bits, a.bits, .little);
    old.update(&bits);
    try std.testing.expect(!std.mem.eql(u8, &a.snapshotIdentity(), &old.finalResult()));
}

test "canonical leftover production profile votes success through startupVote" {
    const M = @import("cuda_test_driver.zig").Memory;
    const Env = struct {
        fn get(name: [:0]const u8) ?[]const u8 {
            if (std.mem.eql(u8, name, "TF_GLM_DRAFT_PROFILE")) return "production";
            if (std.mem.eql(u8, name, "TF_GLM_DRAFTER")) return "dflash2";
            if (std.mem.eql(u8, name, "TF_GLM_DFLASH_DIR")) return "/draft";
            return null;
        }
    };
    var problem_text: []const u8 = "";
    const settings = try Settings.readForOwner(Env.get, &problem_text, true);
    try std.testing.expectEqual(Drafter.mtp, settings.draft);
    const local = validateDraftOwner(settings, true);
    try std.testing.expect(local == null);
    const Recording = struct {
        fn allGather(src: u64, dst: u64, n: usize, dtype: cuda.nccl.DataType, _: cuda.nccl.Comm, _: cuda.abi.Stream) callconv(.c) cuda.nccl.Result {
            std.debug.assert(n == 4 and dtype == .u8);
            const mine: *const u32 = @ptrFromInt(src);
            const votes = [_]u32{ mine.*, mine.* };
            @memcpy(M.slice(dst, 8), std.mem.sliceAsBytes(&votes));
            return 0;
        }
    };
    var d = M.driver();
    var lib: cuda.nccl.Library = .{ .lib = undefined, .api = undefined };
    lib.api.ncclAllGather = Recording.allGather;
    const stream: cuda.Stream = .{ .d = &d, .handle = null };
    var all: [2]u32 = undefined;
    try startupVote(&d, &lib, null, stream, 0, local, &all);
    try std.testing.expectEqual(@as(u32, 1), all[0]);
    try std.testing.expect(M.empty());
}
