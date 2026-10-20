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

/// The forward's knobs a rank reads from its environment (the Python family's names and defaults).
pub const Settings = struct {
    kv: cache.Format = .bf16,
    attn_fp8load: bool = false,
    prompt_latent_aot: bool = false,
    weight_dense: precision.DensePolicy = .{},
    weight_head: precision.Head = .bf16,
    exl3_decode: cuda.exl3_experts.Decode = .scalar,
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

    pub fn hashNumerics(s: Settings, digest: *Sha256) void {
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
        if (s.draft_profile.enabled) {
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
            s.draft = std.meta.stringToEnum(Drafter, v) orelse {
                problem.* = "TF_GLM_DRAFTER: off, mtp or dflash2";
                return error.BadSetting;
            };
        }
        if (s.draft_profile.enabled and s.draft != .dflash2) {
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
        if (s.draft == .dflash2 and (s.draft_dir == null or s.draft_dir.?.len == 0)) {
            problem.* = "TF_GLM_DFLASH_DIR is required for dflash2";
            return error.BadSetting;
        }
        if (get("TF_GLM_EXL3_DECODE")) |v| {
            s.exl3_decode = std.meta.stringToEnum(cuda.exl3_experts.Decode, v) orelse {
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
        if (get("TF_GLM_PREFILL_REDUCE")) |v| {
            if (std.mem.eql(u8, v, "split")) s.split = true else if (std.mem.eql(u8, v, "gather")) s.split = false else {
                problem.* = "TF_GLM_PREFILL_REDUCE: split or gather";
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
        s.heat = heat.bandsFrom(get("TF_HEAT_HIGH"), get("TF_HEAT_LOW")) catch {
            problem.* = "TF_HEAT_HIGH and TF_HEAT_LOW: both or neither, in degrees, with low at or under high";
            return error.BadSetting;
        };
        if (s.prefill_overlap != 0 and !s.split) {
            problem.* = "TF_GLM_PREFILL_OVERLAP requires TF_GLM_PREFILL_REDUCE=split";
            return error.BadSetting;
        }
        return s;
    }
};

/// The most ranks a world holds: rank 0's greedy pick gathers every rank's top-1 into one host array of this size.
pub const max_world = 16;

/// The process environment's value of `name` (null when unset).
pub fn environment(name: [:0]const u8) ?[]const u8 {
    return std.mem.span(std.c.getenv(name) orelse return null);
}

/// What every rank must agree on, as bytes the communicator carries.
pub const Fingerprint = extern struct {
    layout: u32 = 22,
    world: u32,
    rank: u32,
    split: u32,
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
    vision: u32 = 0,
    kv_pool_tokens: u32,
    kv_shard: u32,
    spill: [32]u8 = @splat(0),
    capacity: u64, // tokens a sequence holds
    heat_high: f64, // the heat bands in degrees (0 and 0: off): every rank must wait alike
    heat_low: f64,
    config: [32]u8, // sha256 of config.json
    program: [32]u8, // sha256 of this executable
    checkpoint: [64]u8, // the checkpoint folder's last name (the snapshot's revision), zero padded

    pub const Field = enum { layout, rank, world, split, kda_wide, kda_prompt, affine_prompt_policy, target_graphs, communication, roce_max_bytes, prefetch, draft_prelaunch, pieces, prefill_overlap, prefill_pieces, layers, kv, attn_fp8load, prompt_latent_aot, weight_dense, weight_head, exl3_decode, exl3_routed, draft, dflash_verify, draft_profile, draft_checkpoint, vision, kv_pool_tokens, kv_shard, spill, capacity, heat_high, heat_low, config, program, checkpoint };

    pub fn of(world: usize, rank: usize, s: Settings, capacity: usize, config: [32]u8, program: [32]u8, checkpoint_dir: []const u8) Fingerprint {
        var f: Fingerprint = .{ .world = @intCast(world), .rank = @intCast(rank), .split = @intFromBool(s.split), .kda_wide = @intFromBool(s.kda_wide), .kda_prompt = @backingInt(s.kda_prompt), .pieces = s.pieces, .prefill_overlap = s.prefill_overlap, .prefill_pieces = s.prefill_pieces, .layers = s.layers, .kv = @backingInt(s.kv), .kv_pool_tokens = s.kv_pool_tokens, .kv_shard = @intFromBool(s.kv_shard), .capacity = capacity, .heat_high = if (s.heat) |b| b.high_c else 0, .heat_low = if (s.heat) |b| b.low_c else 0, .config = config, .program = program, .checkpoint = @splat(0) };
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
        f.vision = @intFromBool(s.vision);
        const name = std.fs.path.basename(std.mem.trimEnd(u8, checkpoint_dir, "/"));
        const n = @min(name.len, f.checkpoint.len);
        @memcpy(f.checkpoint[0..n], name[0..n]);
        return f;
    }
};

pub const Mismatch = struct { field: Fingerprint.Field, rank: usize };

/// The first field where rank `r`'s fingerprint differs from rank 0's (or holds the wrong rank), null when all agree.
pub fn mismatch(all: []const Fingerprint) ?Mismatch {
    for (all, 0..) |f, r| {
        if (f.layout != all[0].layout) return .{ .field = .layout, .rank = r };
        if (f.rank != r) return .{ .field = .rank, .rank = r };
        inline for (.{ "world", "split", "kda_wide", "kda_prompt", "affine_prompt_policy", "target_graphs", "communication", "roce_max_bytes", "prefetch", "draft_prelaunch", "pieces", "prefill_overlap", "prefill_pieces", "layers", "kv", "attn_fp8load", "prompt_latent_aot", "weight_dense", "weight_head", "exl3_decode", "exl3_routed", "draft", "dflash_verify", "draft_profile", "draft_checkpoint", "vision", "kv_pool_tokens", "kv_shard", "spill", "capacity", "heat_high", "heat_low", "config", "program", "checkpoint" }) |name| {
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
    var all = [_]Fingerprint{ Fingerprint.of(2, 0, .{}, 8192, h, h, "target"), Fingerprint.of(2, 1, s, 8192, h, h, "target") };
    try std.testing.expectEqual(Fingerprint.Field.communication, mismatch(&all).?.field);
    all[0] = Fingerprint.of(2, 0, s, 8192, h, h, "target");
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
    var all = [_]Fingerprint{ Fingerprint.of(2, 0, selected, 8192, h, h, "target"), Fingerprint.of(2, 1, selected, 8192, h, h, "target") };
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
    const all = [_]Fingerprint{ Fingerprint.of(2, 0, .{}, 8192, h, h, "target"), Fingerprint.of(2, 1, selected, 8192, h, h, "target") };
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
    try std.testing.expectEqualStrings("1a54fef2db1a686d756960222c37f77cf252435d19f96faae70394ff44993767", &std.fmt.bytesToHex(original, .lower));
    try std.testing.expectEqualSlices(u8, &original, &unset_hash.finalResult());
    try std.testing.expectEqualSlices(u8, &original, &off_hash.finalResult());
    const ffn_digest = ffn_hash.finalResult();
    const full_digest = full_hash.finalResult();
    try std.testing.expect(!std.mem.eql(u8, &original, &ffn_digest));
    try std.testing.expect(!std.mem.eql(u8, &original, &full_digest));
    try std.testing.expect(!std.mem.eql(u8, &ffn_digest, &full_digest));
    const h: [32]u8 = @splat(0);
    var all = [_]Fingerprint{ Fingerprint.of(2, 0, unset, 8192, h, h, "target"), Fingerprint.of(2, 1, off, 8192, h, h, "target") };
    try std.testing.expectEqual(@as(u32, 22), all[0].layout);
    try std.testing.expectEqual(@as(u32, @backingInt(TargetGraphs.off)), all[0].target_graphs);
    try std.testing.expect(mismatch(&all) == null);
    all[1] = Fingerprint.of(2, 1, ffn, 8192, h, h, "target");
    try std.testing.expectEqual(@as(u32, @backingInt(TargetGraphs.ffn)), all[1].target_graphs);
    try std.testing.expectEqual(Mismatch{ .field = .target_graphs, .rank = 1 }, mismatch(&all).?);
    all[1] = Fingerprint.of(2, 1, full, 8192, h, h, "target");
    try std.testing.expectEqual(@as(u32, @backingInt(TargetGraphs.full)), all[1].target_graphs);
    try std.testing.expectEqual(Mismatch{ .field = .target_graphs, .rank = 1 }, mismatch(&all).?);
    all[0] = Fingerprint.of(2, 0, full, 8192, h, h, "target");
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
        .dflash_verify => "TF_GLM_DFLASH_VERIFY_DRAFTS",
        .draft_profile => "TF_GLM_DRAFT_PROFILE and production copy settings",
        .vision => "TF_GLM_VISION",
        .kv_pool_tokens => "TF_GLM_KV_POOL_TOKENS",
        .kv_shard => "TF_GLM_KV_SHARD",
        .spill => "TENSORFOLD_SPILL settings",
        .capacity => "the sequence capacity (the context)",
        .heat_high => "TF_HEAT_HIGH",
        .heat_low => "TF_HEAT_LOW",
        .config => "config.json",
        .program => "the executable",
        .checkpoint => "the checkpoint's snapshot folder",
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
    const all = [_]Fingerprint{ Fingerprint.of(2, 0, off, 8192, h, h, "target"), Fingerprint.of(2, 1, on, 8192, h, h, "target") };
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
    const all = [_]Fingerprint{ Fingerprint.of(2, 0, off, 8192, h, h, "target"), Fingerprint.of(2, 1, on, 8192, h, h, "target") };
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
    var all = [_]Fingerprint{ Fingerprint.of(2, 0, .{}, 8192, h, h, "target"), Fingerprint.of(2, 1, on, 8192, h, h, "target") };
    try std.testing.expectEqual(Fingerprint.Field.prefill_overlap, mismatch(&all).?.field);
    all[0] = Fingerprint.of(2, 0, on, 8192, h, h, "target");
    all[1].prefill_pieces = 3;
    try std.testing.expectEqual(Fingerprint.Field.prefill_pieces, mismatch(&all).?.field);
}
