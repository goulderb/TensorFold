//! A TP rank's settings fingerprint, all-gathered before weights load: a mismatch stops every rank.

const std = @import("std");
const cuda = @import("cuda");
const heat = @import("heat");
const cache = @import("cuda_cache.zig");
const precision = @import("cuda_weight_precision.zig");
const mia = @import("cuda_experts_mia.zig");
const Sha256 = std.crypto.hash.sha2.Sha256;
pub const Drafter = enum(u8) { off, mtp, dflash2 };

pub fn effectiveDraft(mode: Drafter, enabled: bool) !Drafter {
    if (mode == .dflash2 and !enabled) return error.DFlashNeedsDrafts;
    return if (enabled) mode else .off;
}

/// The forward's knobs a rank reads from its environment (the Python family's names and defaults).
pub const Settings = struct {
    kv: cache.Format = .bf16,
    weight_dense: precision.DensePolicy = .{},
    weight_head: precision.Head = .bf16,
    exl3_decode: cuda.exl3_experts.Decode = .scalar,
    exl3_routed: mia.Mode = .generic,
    draft: Drafter = .mtp,
    draft_dir: ?[]const u8 = null,
    draft_checkpoint: [32]u8 = @splat(0),
    vision: bool = false,
    kv_pool_tokens: u32 = 0,
    kv_shard: bool = false,
    split: bool = true, // TF_GLM_PREFILL_REDUCE: split (each rank glues its own rows) or gather
    kda_wide: bool = true, // TF_GLM_KDA_DECODE_WIDE: 1 (decode windows on KDA's three-kernel chain) or 0
    pieces: u32 = 3, // TF_GLM_OVERLAP_PIECES: a prompt chunk's exchanges in this many row pieces (1: whole).
    heat: ?heat.Bands = null, // TF_HEAT_HIGH and TF_HEAT_LOW (upstream's names): a prompt chunk waits on every rank's hottest zone
    layers: u32 = 0, // TF_GLM_LAYERS: the first this many layers only, a thin model for one-GPU checks (0: all)
    spill: [32]u8 = @splat(0),

    pub fn hashNumerics(s: Settings, digest: *Sha256) void {
        digest.update("glm-numerics-mia-f1ee79a6-v1\x00");
        s.weight_dense.hash(digest);
        digest.update(@tagName(s.draft));
        digest.update("\x00");
        digest.update(&s.draft_checkpoint);
        for ([_][]const u8{ @tagName(s.weight_head), @tagName(s.exl3_decode), @tagName(s.exl3_routed) }) |value| {
            digest.update(value);
            digest.update("\x00");
        }
    }

    /// The settings `get` (an environment lookup) gives; `problem` names a value that is not one of the choices.
    pub fn read(get: *const fn ([:0]const u8) ?[]const u8, problem: *[]const u8) !Settings {
        var s: Settings = .{};
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
        s.draft_dir = get("TF_GLM_DFLASH_DIR");
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
    layout: u32 = 13,
    world: u32,
    rank: u32,
    split: u32,
    kda_wide: u32,
    pieces: u32,
    layers: u32, // TF_GLM_LAYERS (0: all)
    kv: u32,
    weight_dense: [32]u8 = @splat(0),
    weight_head: u32 = 0,
    exl3_decode: u32 = 0,
    exl3_routed: u32 = 0,
    draft: u32 = 0,
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

    pub const Field = enum { layout, rank, world, split, kda_wide, pieces, layers, kv, weight_dense, weight_head, exl3_decode, exl3_routed, draft, draft_checkpoint, vision, kv_pool_tokens, kv_shard, spill, capacity, heat_high, heat_low, config, program, checkpoint };

    pub fn of(world: usize, rank: usize, s: Settings, capacity: usize, config: [32]u8, program: [32]u8, checkpoint_dir: []const u8) Fingerprint {
        var f: Fingerprint = .{ .world = @intCast(world), .rank = @intCast(rank), .split = @intFromBool(s.split), .kda_wide = @intFromBool(s.kda_wide), .pieces = s.pieces, .layers = s.layers, .kv = @backingInt(s.kv), .kv_pool_tokens = s.kv_pool_tokens, .kv_shard = @intFromBool(s.kv_shard), .capacity = capacity, .heat_high = if (s.heat) |b| b.high_c else 0, .heat_low = if (s.heat) |b| b.low_c else 0, .config = config, .program = program, .checkpoint = @splat(0) };
        f.spill = s.spill;
        f.weight_dense = s.weight_dense.digest();
        f.weight_head = @backingInt(s.weight_head);
        f.exl3_decode = @backingInt(s.exl3_decode);
        f.exl3_routed = @backingInt(s.exl3_routed);
        f.draft = @backingInt(s.draft);
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
        inline for (.{ "world", "split", "kda_wide", "pieces", "layers", "kv", "weight_dense", "weight_head", "exl3_decode", "exl3_routed", "draft", "draft_checkpoint", "vision", "kv_pool_tokens", "kv_shard", "spill", "capacity", "heat_high", "heat_low", "config", "program", "checkpoint" }) |name| {
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
        .pieces => "TF_GLM_OVERLAP_PIECES",
        .layers => "TF_GLM_LAYERS",
        .kv => "TF_GLM_KV",
        .weight_dense => "TF_GLM_DENSE and TF_GLM_DENSE_{KDA,DSA,MLP,SHARED,MTP_EH}",
        .weight_head => "TF_GLM_HEAD",
        .exl3_decode => "TF_GLM_EXL3_DECODE",
        .exl3_routed => "TF_GLM_EXL3_ROUTED",
        .draft => "TF_GLM_DRAFTER",
        .draft_checkpoint => "DFlash checkpoint contents",
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
    try std.testing.expectEqual(cuda.exl3_experts.Decode.scalar, scalar.exl3_decode);
    try std.testing.expectEqual(cuda.exl3_experts.Decode.tensor, tensor.exl3_decode);
    try std.testing.expectError(error.BadSetting, Settings.read(Values.invalid, &problem));
    try std.testing.expectEqualStrings("TF_GLM_EXL3_DECODE: scalar or tensor", problem);
    const h: [32]u8 = @splat(0);
    const all = [_]Fingerprint{ Fingerprint.of(2, 0, scalar, 256, h, h, "model"), Fingerprint.of(2, 1, tensor, 256, h, h, "model") };
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
    try std.testing.expectEqualStrings("TF_GLM_PREFILL_REDUCE: split or gather", problem);
}

test "fingerprints: agreeing ranks pass; the first differing field and rank are named" {
    const h0: [32]u8 = @splat(1);
    const h1: [32]u8 = @splat(2);
    var all = [_]Fingerprint{
        Fingerprint.of(2, 0, .{}, 8192, h0, h0, "/hub/snapshots/423acf37/"),
        Fingerprint.of(2, 1, .{}, 8192, h0, h0, "/hub/snapshots/423acf37"),
    };
    try std.testing.expect(mismatch(&all) == null);
    all[1].program = h1;
    try std.testing.expectEqual(Fingerprint.Field.program, mismatch(&all).?.field);
    all[1] = Fingerprint.of(2, 1, .{ .split = false }, 8192, h0, h0, "/hub/snapshots/423acf37");
    try std.testing.expectEqual(Mismatch{ .field = .split, .rank = 1 }, mismatch(&all).?);
    all[1] = Fingerprint.of(2, 0, .{}, 8192, h0, h0, "/hub/snapshots/423acf37");
    try std.testing.expectEqual(Fingerprint.Field.rank, mismatch(&all).?.field);
    const heat_on: Settings = .{ .heat = .{ .high_c = 92, .low_c = 88 } };
    all[1] = Fingerprint.of(2, 1, heat_on, 8192, h0, h0, "/hub/snapshots/423acf37");
    try std.testing.expectEqual(Fingerprint.Field.heat_high, mismatch(&all).?.field);
    all[1] = Fingerprint.of(2, 1, .{}, 8192, h0, h0, "/hub/snapshots/other");
    try std.testing.expectEqual(Fingerprint.Field.checkpoint, mismatch(&all).?.field);
    all[1] = Fingerprint.of(2, 1, .{ .draft = .off }, 8192, h0, h0, "/hub/snapshots/423acf37");
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
        Fingerprint.of(2, 0, .{}, 8192, hash, hash, "/checkpoint"),
        Fingerprint.of(2, 1, .{ .kv = .fp8 }, 8192, hash, hash, "/checkpoint"),
    };
    try std.testing.expectEqual(Mismatch{ .field = .kv, .rank = 1 }, mismatch(&all).?);
    all[0] = Fingerprint.of(2, 0, .{ .kv = .fp8 }, 8192, hash, hash, "/checkpoint");
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
    var all = [_]Fingerprint{ Fingerprint.of(2, 0, settings, 8192, hash, hash, "/checkpoint"), Fingerprint.of(2, 1, settings, 8192, hash, hash, "/checkpoint") };
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
    var all = [_]Fingerprint{ Fingerprint.of(2, 0, settings, 8192, hash, hash, "/checkpoint"), Fingerprint.of(2, 1, settings, 8192, hash, hash, "/checkpoint") };
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
        Fingerprint.of(2, 0, base, 8192, file_hash, file_hash, "checkpoint"),
        Fingerprint.of(2, 1, same, 8192, file_hash, file_hash, "checkpoint"),
    };
    try std.testing.expect(mismatch(&pair) == null);
    const different = [_]Fingerprint{ pair[0], Fingerprint.of(2, 1, changed, 8192, file_hash, file_hash, "checkpoint") };
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
    const all = [_]Fingerprint{ Fingerprint.of(2, 0, .{ .vision = true }, 8192, hash, hash, "/checkpoint"), Fingerprint.of(2, 1, .{}, 8192, hash, hash, "/checkpoint") };
    try std.testing.expectEqual(Fingerprint.Field.vision, mismatch(&all).?.field);
}
