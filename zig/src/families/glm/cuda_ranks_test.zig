//! GLM rank settings, numerical identity and startup agreement tests.
const std = @import("std");
const cuda = @import("cuda");
const ranks = @import("cuda_ranks.zig");
const Settings = ranks.Settings;
const Fingerprint = ranks.Fingerprint;
const Mismatch = ranks.Mismatch;
const mismatch = ranks.mismatch;
const cache = @import("cuda_cache.zig");
const precision = @import("cuda_weight_precision.zig");
const Sha256 = std.crypto.hash.sha2.Sha256;

fn fake(name: [:0]const u8) ?[]const u8 {
    if (std.mem.eql(u8, name, "TF_GLM_PREFILL_REDUCE")) return "gather";
    if (std.mem.eql(u8, name, "TF_GLM_OVERLAP_PIECES")) return "2";
    if (std.mem.eql(u8, name, "TF_HEAT_HIGH")) return "92";
    if (std.mem.eql(u8, name, "TF_HEAT_LOW")) return "88";
    return null;
}

fn bad(name: [:0]const u8) ?[]const u8 {
    if (std.mem.eql(u8, name, "TF_GLM_PREFILL_REDUCE")) return "sum";
    return null;
}

test "EXL3 decode defaults to scalar, refuses unknown policy and separates ranks" {
    const Values = struct {
        fn empty(_: [:0]const u8) ?[]const u8 {
            return null;
        }
        fn tensor(name: [:0]const u8) ?[]const u8 {
            return if (std.mem.eql(u8, name, "TF_GLM_EXL3_DECODE")) "tensor" else null;
        }
        fn invalid(name: [:0]const u8) ?[]const u8 {
            return if (std.mem.eql(u8, name, "TF_GLM_EXL3_DECODE")) "1" else null;
        }
    };
    var problem: []const u8 = "";
    const scalar = try Settings.read(Values.empty, &problem);
    const tensor = try Settings.read(Values.tensor, &problem);
    try std.testing.expectEqual(cuda.exl3_trellis_experts.Decode.scalar, scalar.exl3_decode);
    try std.testing.expectEqual(cuda.exl3_trellis_experts.Decode.tensor, tensor.exl3_decode);
    try std.testing.expectError(error.BadSetting, Settings.read(Values.invalid, &problem));
    try std.testing.expectEqualStrings("TF_GLM_EXL3_DECODE: scalar or tensor", problem);
    const h: [32]u8 = @splat(0);
    const all = [_]Fingerprint{ Fingerprint.of(2, 0, scalar, 256, true, h, h, ranks.hashBytes("model")), Fingerprint.of(2, 1, tensor, 256, true, h, h, ranks.hashBytes("model")) };
    try std.testing.expectEqual(Fingerprint.Field.exl3_decode, mismatch(&all).?.field);
    var scalar_hash = Sha256.init(.{});
    var tensor_hash = Sha256.init(.{});
    scalar.hashNumerics(&scalar_hash);
    tensor.hashNumerics(&tensor_hash);
    try std.testing.expect(!std.mem.eql(u8, &scalar_hash.finalResult(), &tensor_hash.finalResult()));
}

test "settings from the environment: the defaults, the choices, and a refusal naming a bad value" {
    var problem: []const u8 = "";
    const s = try Settings.read(&fake, &problem);
    try std.testing.expect(!s.split and s.kda_wide and s.pieces == 2 and s.heat.?.high_c == 92 and s.heat.?.low_c == 88);
    try std.testing.expectError(error.BadSetting, Settings.read(&bad, &problem));
    try std.testing.expectEqualStrings("TF_GLM_PREFILL_REDUCE: split, ring or gather", problem);
}

test "fingerprints: agreeing ranks pass; the first differing field and rank are named" {
    const h0: [32]u8 = @splat(1);
    const h1: [32]u8 = @splat(2);
    var all = [_]Fingerprint{
        Fingerprint.of(2, 0, .{}, 8192, true, h0, h0, ranks.hashBytes("/hub/snapshots/423acf37")),
        Fingerprint.of(2, 1, .{}, 8192, true, h0, h0, ranks.hashBytes("/hub/snapshots/423acf37")),
    };
    try std.testing.expect(mismatch(&all) == null);
    all[1].program = h1;
    try std.testing.expectEqual(Fingerprint.Field.program, mismatch(&all).?.field);
    all[1] = Fingerprint.of(2, 1, .{ .split = false }, 8192, true, h0, h0, ranks.hashBytes("/hub/snapshots/423acf37"));
    try std.testing.expectEqual(Mismatch{ .field = .split, .rank = 1 }, mismatch(&all).?);
    all[1] = Fingerprint.of(2, 0, .{}, 8192, true, h0, h0, ranks.hashBytes("/hub/snapshots/423acf37"));
    try std.testing.expectEqual(Fingerprint.Field.rank, mismatch(&all).?.field);
    const heat_on: Settings = .{ .heat = .{ .high_c = 92, .low_c = 88 } };
    all[1] = Fingerprint.of(2, 1, heat_on, 8192, true, h0, h0, ranks.hashBytes("/hub/snapshots/423acf37"));
    try std.testing.expectEqual(Fingerprint.Field.heat_high, mismatch(&all).?.field);
    all[1] = Fingerprint.of(2, 1, .{}, 8192, true, h0, h0, ranks.hashBytes("/hub/snapshots/other"));
    try std.testing.expectEqual(Fingerprint.Field.checkpoint, mismatch(&all).?.field);
    all[1] = Fingerprint.of(2, 1, .{ .draft = .off }, 8192, true, h0, h0, ranks.hashBytes("/hub/snapshots/423acf37"));
    try std.testing.expectEqual(Mismatch{ .field = .draft, .rank = 1 }, mismatch(&all).?);
}

fn cacheSetting(comptime value: []const u8) *const fn ([:0]const u8) ?[]const u8 {
    return &struct {
        fn get(name: [:0]const u8) ?[]const u8 {
            return if (std.mem.eql(u8, name, "TF_GLM_KV")) value else null;
        }
    }.get;
}

test "cache format is parsed once and rank disagreement is refused" {
    var problem: []const u8 = "";
    try std.testing.expectEqual(cache.Format.bf16, (try Settings.read(cacheSetting(""), &problem)).kv);
    try std.testing.expectEqual(cache.Format.bf16, (try Settings.read(cacheSetting(" bf16 "), &problem)).kv);
    try std.testing.expectEqual(cache.Format.fp8, (try Settings.read(cacheSetting(" fp8\n"), &problem)).kv);
    try std.testing.expectError(error.BadSetting, Settings.read(cacheSetting("fp16"), &problem));
    try std.testing.expectEqualStrings("TF_GLM_KV: bf16 or fp8", problem);
    const hash: [32]u8 = @splat(1);
    var all = [_]Fingerprint{
        Fingerprint.of(2, 0, .{}, 8192, true, hash, hash, ranks.hashBytes("/checkpoint")),
        Fingerprint.of(2, 1, .{ .kv = .fp8 }, 8192, true, hash, hash, ranks.hashBytes("/checkpoint")),
    };
    try std.testing.expectEqual(Mismatch{ .field = .kv, .rank = 1 }, mismatch(&all).?);
    all[0] = Fingerprint.of(2, 0, .{ .kv = .fp8 }, 8192, true, hash, hash, ranks.hashBytes("/checkpoint"));
    try std.testing.expect(mismatch(&all) == null);
}

test "shared pool and ownership options are explicit and ranks agree on their geometry" {
    const Values = struct {
        fn shared(name: [:0]const u8) ?[]const u8 {
            if (std.mem.eql(u8, name, "TF_GLM_KV_POOL_TOKENS")) return "32768";
            if (std.mem.eql(u8, name, "TF_GLM_KV_SHARD")) return "1";
            return null;
        }
        fn missing(name: [:0]const u8) ?[]const u8 {
            if (std.mem.eql(u8, name, "TF_GLM_KV_SHARD")) return "1";
            return null;
        }
    };
    var problem: []const u8 = "";
    const settings = try Settings.read(Values.shared, &problem);
    try std.testing.expectEqual(@as(u32, 32768), settings.kv_pool_tokens);
    try std.testing.expect(settings.kv_shard);
    try std.testing.expectError(error.BadSetting, Settings.read(Values.missing, &problem));
    const hash: [32]u8 = @splat(0);
    var all = [_]Fingerprint{ Fingerprint.of(2, 0, settings, 8192, true, hash, hash, ranks.hashBytes("/checkpoint")), Fingerprint.of(2, 1, settings, 8192, true, hash, hash, ranks.hashBytes("/checkpoint")) };
    try std.testing.expectEqual(@as(?Mismatch, null), mismatch(&all));
    all[1].kv_pool_tokens = 16384;
    try std.testing.expectEqual(Fingerprint.Field.kv_pool_tokens, mismatch(&all).?.field);
    all[1].kv_pool_tokens = 32768;
    all[1].kv_shard = 0;
    try std.testing.expectEqual(Fingerprint.Field.kv_shard, mismatch(&all).?.field);
}

test "weight precision and spill enablement must agree before rank startup" {
    const Values = struct {
        fn get(name: [:0]const u8) ?[]const u8 {
            if (std.mem.eql(u8, name, "TF_GLM_DENSE")) return "q4";
            if (std.mem.eql(u8, name, "TF_GLM_HEAD")) return "fp8";
            if (std.mem.eql(u8, name, "TENSORFOLD_SPILL_GIB")) return "64";
            return null;
        }
        fn empty(_: [:0]const u8) ?[]const u8 {
            return null;
        }
        fn badDense(name: [:0]const u8) ?[]const u8 {
            return if (std.mem.eql(u8, name, "TF_GLM_DENSE")) "int8" else null;
        }
        fn badHead(name: [:0]const u8) ?[]const u8 {
            return if (std.mem.eql(u8, name, "TF_GLM_HEAD")) "q4" else null;
        }
    };
    var problem: []const u8 = "";
    const settings = try Settings.read(Values.get, &problem);
    const defaults = try Settings.read(Values.empty, &problem);
    for (precision.groups) |group| try std.testing.expectEqual(precision.Dense.q4, settings.weight_dense.get(group));
    try std.testing.expectEqual(precision.Head.fp8, settings.weight_head);
    try std.testing.expectError(error.BadSetting, Settings.read(Values.badDense, &problem));
    try std.testing.expectEqualStrings("TF_GLM_DENSE: bf16, q4, fp8, nvfp4, fp8_native, nvfp4_native or source", problem);
    try std.testing.expectError(error.BadSetting, Settings.read(Values.badHead, &problem));
    try std.testing.expectEqualStrings("TF_GLM_HEAD: bf16 or fp8", problem);
    const hash: [32]u8 = @splat(0);
    var all = [_]Fingerprint{ Fingerprint.of(2, 0, settings, 8192, true, hash, hash, ranks.hashBytes("/checkpoint")), Fingerprint.of(2, 1, settings, 8192, true, hash, hash, ranks.hashBytes("/checkpoint")) };
    try std.testing.expect(mismatch(&all) == null);
    all[1].weight_dense = @splat(0);
    try std.testing.expectEqual(Fingerprint.Field.weight_dense, mismatch(&all).?.field);
    all[1].weight_dense = all[0].weight_dense;
    all[1].weight_head = 0;
    try std.testing.expectEqual(Fingerprint.Field.weight_head, mismatch(&all).?.field);
    all[1].weight_head = all[0].weight_head;
    all[1].spill = defaults.spill;
    try std.testing.expectEqual(Fingerprint.Field.spill, mismatch(&all).?.field);
}

test "effective family policy governs rank and persistent numerical identity" {
    const Env = struct {
        fn empty(_: [:0]const u8) ?[]const u8 {
            return null;
        }
        fn equivalent(name: [:0]const u8) ?[]const u8 {
            if (std.mem.eql(u8, name, "TF_GLM_DENSE")) return "fp8";
            inline for (.{ "TF_GLM_DENSE_KDA", "TF_GLM_DENSE_DSA", "TF_GLM_DENSE_MLP", "TF_GLM_DENSE_SHARED", "TF_GLM_DENSE_MTP_EH" }) |key| {
                if (std.mem.eql(u8, name, key)) return "bf16";
            }
            return null;
        }
        fn dsa(name: [:0]const u8) ?[]const u8 {
            return if (std.mem.eql(u8, name, "TF_GLM_DENSE_DSA")) "nvfp4" else null;
        }
    };
    var problem: []const u8 = "";
    const base = try Settings.read(&Env.empty, &problem);
    const same = try Settings.read(&Env.equivalent, &problem);
    const changed = try Settings.read(&Env.dsa, &problem);
    const file_hash: [32]u8 = @splat(7);
    const pair = [_]Fingerprint{
        Fingerprint.of(2, 0, base, 8192, true, file_hash, file_hash, ranks.hashBytes("checkpoint")),
        Fingerprint.of(2, 1, same, 8192, true, file_hash, file_hash, ranks.hashBytes("checkpoint")),
    };
    try std.testing.expect(mismatch(&pair) == null);
    const different = [_]Fingerprint{ pair[0], Fingerprint.of(2, 1, changed, 8192, true, file_hash, file_hash, ranks.hashBytes("checkpoint")) };
    try std.testing.expectEqual(Fingerprint.Field.weight_dense, mismatch(&different).?.field);
    var a = Sha256.init(.{});
    var b = Sha256.init(.{});
    var c = Sha256.init(.{});
    base.hashNumerics(&a);
    same.hashNumerics(&b);
    changed.hashNumerics(&c);
    try std.testing.expectEqualSlices(u8, &a.finalResult(), &b.finalResult());
    try std.testing.expect(!std.mem.eql(u8, &a.finalResult(), &c.finalResult()));
}

test "vision startup is explicit and requires rank agreement" {
    const Values = struct {
        fn enabled(name: [:0]const u8) ?[]const u8 {
            return if (std.mem.eql(u8, name, "TF_GLM_VISION")) "1" else null;
        }
        fn invalid(name: [:0]const u8) ?[]const u8 {
            return if (std.mem.eql(u8, name, "TF_GLM_VISION")) "true" else null;
        }
    };
    var problem: []const u8 = "";
    try std.testing.expect((try Settings.read(Values.enabled, &problem)).vision);
    try std.testing.expectError(error.BadSetting, Settings.read(Values.invalid, &problem));
    try std.testing.expectEqualStrings("TF_GLM_VISION: 0 or 1", problem);
    const hash: [32]u8 = @splat(0);
    const all = [_]Fingerprint{ Fingerprint.of(2, 0, .{ .vision = true }, 8192, true, hash, hash, ranks.hashBytes("/checkpoint")), Fingerprint.of(2, 1, .{}, 8192, true, hash, hash, ranks.hashBytes("/checkpoint")) };
    try std.testing.expectEqual(Fingerprint.Field.vision_setting, mismatch(&all).?.field);
}

test "canonical owner ignores leftover DFlash env and still refuses active bad settings" {
    const Values = struct {
        fn leftover(name: [:0]const u8) ?[]const u8 {
            return if (std.mem.eql(u8, name, "TF_GLM_DRAFTER")) "dflash2" else null;
        }
        fn leftover_and_bad_tile(name: [:0]const u8) ?[]const u8 {
            if (std.mem.eql(u8, name, "TF_GLM_DRAFTER")) return "dflash2";
            if (std.mem.eql(u8, name, "TF_GLM_PROMPT_TILE")) return "nope";
            return null;
        }
        fn leftover_with_dir(name: [:0]const u8) ?[]const u8 {
            if (std.mem.eql(u8, name, "TF_GLM_DRAFTER")) return "dflash2";
            if (std.mem.eql(u8, name, "TF_GLM_DFLASH_DIR")) return "/draft";
            return null;
        }
    };
    var problem: []const u8 = "";
    try std.testing.expectError(error.BadSetting, Settings.read(Values.leftover, &problem));
    try std.testing.expectEqualStrings("TF_GLM_DFLASH_DIR is required for dflash2", problem);
    const canonical = try Settings.readForOwner(Values.leftover, &problem, true);
    try std.testing.expectEqual(ranks.Drafter.mtp, canonical.draft);
    try std.testing.expectError(error.BadSetting, Settings.readForOwner(Values.leftover_and_bad_tile, &problem, true));
    try std.testing.expectEqualStrings("TF_GLM_PROMPT_TILE: 0 or 1", problem);
    const ignored = try Settings.readForOwner(Values.leftover_with_dir, &problem, true);
    try std.testing.expectEqual(ranks.Drafter.mtp, ignored.draft);
    const legacy = try Settings.read(Values.leftover_with_dir, &problem);
    try std.testing.expectEqual(ranks.Drafter.dflash2, legacy.draft);
}
