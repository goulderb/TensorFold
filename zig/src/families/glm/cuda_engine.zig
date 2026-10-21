//! One TP rank's GLM-5.3-Flash CUDA engine (the Python family's decode.Engine); all ranks draw the same tokens.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const config = @import("config.zig");
const shard = @import("cuda_shard.zig");
const weights = @import("cuda_weights.zig");
const fw = @import("cuda_forward.zig");
const ranks = @import("cuda_ranks.zig");
const sampler = @import("cuda_sampler.zig");
const greedy_mod = @import("cuda_greedy.zig");
const prompt_lp = @import("cuda_prompt_logprobs.zig");
const heat = @import("heat");
const owner_kv = @import("cuda_kv_owner.zig");
const cache_pool = @import("cuda_kv_pool.zig");
const modelopt = @import("core").modelopt;
const vision_binding = @import("cuda_vision_binding.zig");
const vision_runtime = @import("cuda_vision_runtime.zig");
const dflash_mod = @import("cuda_dflash.zig");
const mia = @import("cuda_experts_mia.zig");
const verify_plan = @import("cuda_verify_plan.zig");
const route_trace = @import("cuda_route_trace.zig");
const target_graph = @import("cuda_target_graph.zig");
const target_forward = @import("cuda_target_forward.zig");
const l2pf = @import("cuda_l2pf.zig");
const l2pf_plan = @import("cuda_l2pf_plan.zig");
pub const persistence = @import("cuda_persistence.zig");
pub const spill_payload = @import("cuda_spill_payload.zig");

pub const Room = types.Room;
pub const Options = types.Options;
// Historical equivalence bound for the partial chunk adapter (not prefillMany).
pub const prompt_batch_rows = 127;

fn validateKdaProjection(c: *const config.Config, world: usize, converted: bool) !void {
    if (!converted) return;
    for (0..world) |rank| {
        const heads = (try shard.Share.of(c, rank, world)).kda.len();
        const n = heads * c.kda_dim;
        const width = 3 * n + 2 * c.kda_dim + heads;
        if (width % 8 != 0 or (3 * n) % 8 != 0 or (3 * n + c.kda_dim) % 8 != 0) return error.UnalignedKdaProjection;
    }
}

test "converted KDA validates every TP rank from shared geometry" {
    const c = config.Config{};
    try validateKdaProjection(&c, 2, true);
    try std.testing.expectEqual(@as(usize, 32), (try shard.Share.of(&c, 0, 2)).kda.len());
    try std.testing.expectEqual(@as(usize, 32), (try shard.Share.of(&c, 1, 2)).kda.len());

    try std.testing.expectEqual(@as(usize, 8), (try shard.Share.of(&c, 0, 9)).kda.len());
    try std.testing.expectEqual(@as(usize, 7), (try shard.Share.of(&c, 1, 9)).kda.len());
    try std.testing.expectError(error.UnalignedKdaProjection, validateKdaProjection(&c, 9, true));
    try validateKdaProjection(&c, 9, false);
}
const exl3 = @import("cuda_exl3.zig");
const images_mod = @import("cuda_images.zig");
const tile = @import("cuda_tile.zig");
const mm_mod = @import("cuda_mm.zig");
const dflash = @import("dflash.zig");
const prof_mod = @import("cuda_prof.zig");
const types = @import("cuda_engine_types.zig");

/// Rank 0's own word on its request (cancelled or not); the other ranks have none and hear rank 0's.
pub const Cancel = struct { ptr: ?*anyopaque, check: *const fn (?*anyopaque) bool };

/// The most drafts a chain holds (decode.DepthPolicy's top depth).
pub const max_drafts = verify_plan.max_drafts;

/// A prompt pass's chunk rows (decode.prefill's).
pub const prompt_rows = fw.prefill_rows;
pub const max_together = @import("cuda_state.zig").prompt_heads;
pub const Prompt = struct { slot: u32, ids: []const u32 };

/// Storage ceiling; verifyPlan selects the active row limit below hc_pre's 64-row change.
pub const shared_rows = verify_plan.max_rows;
pub const max_shared = verify_plan.max_streams;

fn targetWarmWindow(rows: usize) !struct { rows: [2]usize, count: usize } {
    const profile = @import("cuda_draft_profile.zig");
    if (rows == 0 or rows > profile.shared_rows) return error.BadTargetGraphWarmupShape;
    const first = @min(rows, profile.target_rows);
    return .{ .rows = .{ first, rows - first }, .count = if (rows > first) 2 else 1 };
}

test "production target warmup covers every shared width with at most two sixteen-row slots" {
    try std.testing.expectError(error.BadTargetGraphWarmupShape, targetWarmWindow(0));
    try std.testing.expectError(error.BadTargetGraphWarmupShape, targetWarmWindow(33));
    for (1..33) |rows| {
        const w = try targetWarmWindow(rows);
        var total: usize = 0;
        for (w.rows[0..w.count]) |n| {
            try std.testing.expect(n > 0 and n <= 16);
            total += n;
        }
        try std.testing.expectEqual(rows, total);
        try std.testing.expect(w.count <= 2);
    }
}

test "canonical Engine.verifyShared stages five 8-row windows and refuses a reduced allocation" {
    const Memory = @import("cuda_test_driver.zig").Memory;
    const block = @import("dflash_block.zig");
    var d = Memory.driver();
    const c = config.Config{};
    const w: weights.Weights = .{ .gpa = std.testing.allocator, .config = c, .share = try shard.Share.of(&c, 0, 2) };
    var dr: dflash.Drafter = .{ .gpa = std.testing.allocator, .d = &d, .module = undefined, .k = undefined, .weights = undefined, .split = false, .ff = 0, .rank = 0, .world = 1 };
    defer {
        for (dr.slots.items) |*s| if (s.ring) |*r| r.free();
        dr.slots.deinit(std.testing.allocator);
    }
    const dummy = Memory.dummyFunction(&d);
    inline for (@typeInfo(dflash.Kernels).@"struct".field_names) |name| @field(dr.k, name) = dummy;
    var tap_store = try cuda.DeviceBuffer.alloc(&d, dflash.STEP_ROWS * dflash.TAP_ROW);
    defer tap_store.free();
    dr.taps_step = tap_store.ptr;
    var consumed = try cuda.DeviceBuffer.alloc(&d, 64 * dflash.TAP_ROW);
    defer consumed.free();
    var ids: [8][8]u32 = undefined;
    var outs: [8][8]u32 = undefined;
    var parts: [8]Part = undefined;
    for (&parts, &ids, &outs, 0..) |*p, *id, *out, i| {
        for (id, 0..) |*tok, r| tok.* = @intCast(i * 8 + r + 1);
        p.* = .{ .slot = @intCast(i), .ids = id, .out = out };
    }
    var e: Engine = undefined;
    e.full_graphs = @as(target_forward.Runtime, undefined);
    e.ffn_graphs = null;
    e.route_trace = null;
    e.prefetch = null;
    e.tp_group = null;
    e.replicas = .{};
    e.target_graphs_active = true;
    e.capture_lifetime = 0;
    e.structured = false;
    e.pair_failure = struct {
        fn ignore(_: anyerror) void {}
    }.ignore;
    e.settings = .{ .kv = .bf16 };
    e.drafter = &dr;
    e.canonical_owner = true;
    e.dflash = null;
    e.gpa = std.testing.allocator;
    e.d = &d;
    e.w = w;
    e.c = c;
    e.share = w.share;
    e.capacity = 64;
    e.mtp = false;
    e.pool = null;
    e.verify_widths = @splat(0);
    e.seqs = .empty;
    e.stream = .{ .d = &d, .handle = null };
    e.m = undefined;
    e.m.glue = Memory.dummyGlue(&d);
    e.m.tiled = false;
    e.m.packs = null;
    e.m.ck = null;
    e.m.prompt_latent = null;
    e.m.source_dense = null;
    e.m.source_upcast = null;
    e.m.copy_rows = .{ .d = &d, .handle = @ptrFromInt(7) };
    const TapCopy = struct {
        var copies: usize = 0;
        fn launch(f: cuda.abi.Function, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: cuda.abi.Stream, args: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
            if (f == @as(cuda.abi.Function, @ptrFromInt(7))) {
                const a = args.?;
                const src = @as(*const u64, @ptrCast(@alignCast(a[0].?))).*;
                const src_stride = @as(*const i64, @ptrCast(@alignCast(a[1].?))).*;
                const dst = @as(*const u64, @ptrCast(@alignCast(a[2].?))).*;
                const dst_stride = @as(*const i64, @ptrCast(@alignCast(a[3].?))).*;
                const nrows: usize = @intCast(@as(*const i32, @ptrCast(@alignCast(a[4].?))).*);
                const bytes: usize = @intCast(@as(*const i32, @ptrCast(@alignCast(a[5].?))).*);
                for (0..nrows) |r| @memcpy(Memory.slice(dst + r * @as(u64, @intCast(dst_stride)), bytes), Memory.slice(src + r * @as(u64, @intCast(src_stride)), bytes));
                copies += 1;
            }
            _ = Memory.launches.fetchAdd(1, .monotonic);
            return 0;
        }
    };
    TapCopy.copies = 0;
    d.api.cuLaunchKernel = TapCopy.launch;
    e.greedy_rows = try greedy_mod.Greedy.init(std.testing.allocator, &d, 1, 64);
    defer e.greedy_rows.deinit();
    defer {
        for (e.seqs.items) |q| if (q) |s| {
            s.st.deinit();
            e.gpa.destroy(s);
        };
        e.seqs.deinit(e.gpa);
    }
    try std.testing.expectEqual(@as(usize, 8), e.targetRows());
    try std.testing.expectEqual(@as(usize, 64), dflash.STEP_ROWS);
    {
        var small = try fw.Buffers.initShared(&d, &w, 32, 64);
        defer small.deinit();
        try small.enableTaps(&d);
        e.bs = small;
        try std.testing.expectError(error.WindowTooWide, e.verifyShared(parts[0..5]));
    }
    {
        var big = try fw.Buffers.initShared(&d, &w, 64, 64);
        defer big.deinit();
        try big.enableTaps(&d);
        e.bs = big;
        try std.testing.expectEqual(@as(usize, 64), e.sharedCapacity());
        var wide: [9]u32 = @splat(1);
        var wide_out: [9]u32 = undefined;
        const overflow = [_]Part{.{ .slot = 0, .ids = &wide, .out = &wide_out }};
        try std.testing.expectError(error.WindowTooWide, e.verifyShared(&overflow));
        const tap_src = e.bs.?.taps.?;
        for (0..40) |row| std.mem.writeInt(u32, Memory.slice(tap_src.ptr + row * @import("cuda_taps.zig").hidden * 2, 4)[0..4], @intCast(0xC0DE0000 + row), .little);
        Memory.launches.store(0, .monotonic);
        TapCopy.copies = 0;
        try e.verifyShared(parts[0..5]);
        try std.testing.expect(TapCopy.copies >= @import("cuda_taps.zig").layers.len);
        try std.testing.expect(Memory.launches.load(.monotonic) >= 44);
        try std.testing.expectEqual(@as(u64, 1), dr.verifies);
        try std.testing.expectEqual(@as(u64, 1), e.verify_widths[40]);
        for (0..5) |slot| {
            const s = dr.slots.items[slot];
            try std.testing.expectEqual(slot * 8, s.v_row0);
            try std.testing.expectEqual(@as(usize, 8), s.v_rows);
            try std.testing.expectEqual(@as(i64, 0), s.v_pos0);
            try std.testing.expectEqual(dr.verifies, s.v_step);
            try block.copy(&e, consumed.ptr + slot * 8 * dflash.TAP_ROW, dr.taps_step + s.v_row0 * dflash.TAP_ROW, 8 * dflash.TAP_ROW);
            for (0..8) |r| {
                const got = std.mem.readInt(u32, Memory.slice(consumed.ptr + (slot * 8 + r) * dflash.TAP_ROW, 4)[0..4], .little);
                try std.testing.expectEqual(@as(u32, @intCast(0xC0DE0000 + slot * 8 + r)), got);
            }
        }
        {
            const D = dflash.D;
            const dense: @import("cuda_mm.zig").Tiled = .{ .w = 0, .n = D, .k = D };
            dr.fc = .{ .bf16 = dense };
            dr.hproj = dense;
            for (&dr.layers) |*layer| layer.* = .{ .in_norm = 0, .post_norm = 0, .qn = 0, .kn = 0, .a_base = 0, .m_base = 0, .qkv = .{ .bf16 = dense }, .kv = .{ .bf16 = dense }, .o = .{ .bf16 = dense }, .gu = .{ .bf16 = dense }, .down = dense, .a_proj = .{ .bf16 = dense }, .m_proj = .{ .bf16 = dense } };
            e.m.mm = .{ .by_rows = @splat(dummy), .packed_rows = @splat(@splat(dummy)), .prompt = dummy, .reduce = dummy, .tile = dummy, .tile_packed = @splat(dummy), .pdl = false };
            e.m.q4 = .{ .group = dummy, .pack_dense = dummy, .transpose = dummy, .gb10 = false };
            var cin = try cuda.DeviceBuffer.alloc(&d, 40 * dflash.TAP_ROW);
            defer cin.free();
            dr.cin = cin.ptr;
            var c1 = try cuda.DeviceBuffer.alloc(&d, 40 * D * 2);
            defer c1.free();
            dr.c1 = c1.ptr;
            var ctx_h = try cuda.DeviceBuffer.alloc(&d, 40 * D * 2);
            defer ctx_h.free();
            dr.ctx = ctx_h.ptr;
            var ckv = try cuda.DeviceBuffer.alloc(&d, 40 * D * 2);
            defer ckv.free();
            dr.ckv = ckv.ptr;
            var cpos = try cuda.DeviceBuffer.alloc(&d, dflash.MAX_CTX * 8);
            defer cpos.free();
            dr.cpos = cpos.ptr;
            var crings = try cuda.DeviceBuffer.alloc(&d, dflash.MAX_CTX * 8);
            defer crings.free();
            dr.crings = crings.ptr;
            var positions: [dflash.MAX_CTX]i64 = undefined;
            var rings: [dflash.MAX_CTX]u64 = undefined;
            dr.h_cpos = &positions;
            dr.h_crings = &rings;
            var matching: [5]dflash.Req = undefined;
            for (&matching, 0..) |*q, i| q.* = .{ .slot = @intCast(i), .pos = 8, .n = 8, .anchor = 1, .propose = false };
            dr.lost = 0;
            try dr.draft(&e, &matching);
            try std.testing.expectEqual(@as(u64, 0), dr.lost);
            for (0..5) |slot| {
                try std.testing.expectEqual(@as(i64, 0), dr.slots.items[slot].from);
                try std.testing.expectEqual(@as(i64, 8), dr.slots.items[slot].upto);
            }
            try std.testing.expectEqualSlices(i64, &.{ 0, 1, 2, 3, 4, 5, 6, 7 }, dr.h_cpos[0..8]);
        }
        var stale: [5]dflash.Req = undefined;
        for (&stale, 0..) |*q, i| q.* = .{ .slot = @intCast(i), .pos = 99, .n = 8, .anchor = 1, .propose = false };
        dr.lost = 0;
        try dr.draft(&e, &stale);
        try std.testing.expectEqual(@as(u64, 5), dr.lost);
        for (0..64) |row| std.mem.writeInt(u32, Memory.slice(tap_src.ptr + row * @import("cuda_taps.zig").hidden * 2, 4)[0..4], @intCast(0xDA7A0000 + row), .little);
        Memory.launches.store(0, .monotonic);
        try e.verifyShared(&parts);
        try std.testing.expect(Memory.launches.load(.monotonic) >= 68);
        try std.testing.expectEqual(@as(u64, 2), dr.verifies);
        try std.testing.expectEqual(@as(u64, 1), e.verify_widths[64]);
        for (0..8) |slot| {
            const s = dr.slots.items[slot];
            try std.testing.expectEqual(slot * 8, s.v_row0);
            try std.testing.expectEqual(@as(usize, 8), s.v_rows);
            try block.copy(&e, consumed.ptr + slot * 8 * dflash.TAP_ROW, dr.taps_step + s.v_row0 * dflash.TAP_ROW, 8 * dflash.TAP_ROW);
            for (0..8) |r| {
                const got = std.mem.readInt(u32, Memory.slice(consumed.ptr + (slot * 8 + r) * dflash.TAP_ROW, 4)[0..4], .little);
                try std.testing.expectEqual(@as(u32, @intCast(0xDA7A0000 + slot * 8 + r)), got);
            }
        }
        var extra: [9]Part = undefined;
        @memcpy(extra[0..8], &parts);
        extra[8] = parts[0];
        extra[8].slot = 8;
        try std.testing.expectError(error.WindowTooWide, e.verifyShared(&extra));
    }
}

pub const Part = types.Part;
pub const DraftPart = struct { slot: u32, position: u64, pending: u32, depth: u32 };
pub const DraftResult = dflash_mod.BatchResult;
pub const Kept = types.Kept;
const Seq = types.Seq;
pub const VisionOwned = types.VisionOwned;
const Snap = types.Snap;
pub const Resume = types.Resume;
pub const Marks = types.Marks;

pub const Engine = struct {
    capture_lifetime: u64 = 0,
    gpa: std.mem.Allocator,
    io: std.Io,
    d: *const cuda.Driver,
    c: config.Config,
    share: shard.Share,
    spans: []shard.Span, // every rank's vocabulary span
    lib: *const cuda.nccl.Library,
    comm: cuda.nccl.Comm,
    tp_group: ?cuda.tp_group.Group = null,
    prefetch: ?l2pf.Prefetch = null,
    replicas: @import("cuda_dsa_block.zig").Replicas = .{},
    images: ?*images_mod.Images = null, // borrowed media service; caller closes after engine drains
    packs: mm_mod.Packed = .{},
    drafter: ?*dflash.Drafter = null,
    canonical_owner: bool = false,
    canonical: ranks.CanonicalDraft = .{},
    profs: [fw.window_rows]?*prof_mod.Prof = @splat(null),
    prompt_profs: [4]?*prof_mod.Prof = @splat(null),
    checkpoint_digest: [32]u8,
    settings: ranks.Settings,
    m: fw.Modules,
    stream: cuda.Stream,
    pieces: fw.Pieces,
    w: weights.Weights,
    bp: fw.Buffers, // prompt chunks
    bd: fw.Buffers, // verify windows
    bm: fw.Buffers, // the MTP head's rows
    bs: ?fw.Buffers = null, // shared verify windows (with the MTP head, as the lane core shares only then)
    verify_widths: [shared_rows + 1]u64 = @splat(0),
    verify_lone_widths: [fw.window_rows + 1]u64 = @splat(0),
    prompt_wy_logged: bool = false,
    prompt_overlap_logged: bool = false,
    route_trace: ?route_trace.Trace = null,
    ffn_graphs: ?target_graph.Bank = null,
    full_graphs: ?target_forward.Runtime = null,
    /// Proof tools may compare eager/replay at drained boundaries; serving leaves this enabled when configured.
    target_graphs_active: bool = true,
    greedy_rows: greedy_mod.Greedy,
    smp: sampler.Sampler,
    seqs: std.ArrayList(?*Seq) = .empty, // every stream's sequence by slot (the lane backend's), slot 0 made at init
    cur: *Seq = undefined, // the bound slot's: every step below runs on it
    slot: u32 = 0,
    capacity: usize,
    mtp: bool,
    dflash: ?*dflash_mod.DFlash = null,
    pool: ?cache_pool.Pool = null,
    pair_failure: *const fn (anyerror) void = owner_kv.failPair,
    heat_gate: heat.Gate, // upstream's bands (off unless set), read before each prompt chunk on the gathered maximum
    shared_buf: cuda.DeviceBuffer, // one f64 every rank all-reduces to its maximum
    snaps: std.AutoHashMapUnmanaged(u64, Snap) = .empty, // kept prompt states by id (the same ids on every rank)
    next_snap: u64 = 1, // rank 0: the next id
    disk: ?persistence.Port = null,
    announce: u64 = 0, // rank 0: a state kept at this chunk's mark, for the other ranks to copy at its gate
    dropped: std.ArrayList(u32) = .empty, // rank 0: ids dropped since the other ranks last heard (cuda_follow)
    prompt_logprobs: ?prompt_lp.Want = null, // a prompt pass's rows asked for (every rank runs the head on them)
    prompt_bufs: ?prompt_lp.Buffers = null, // their head passes' buffers, made at the first pass that asks
    prompt_tried: bool = false, // that pass came: without buffers after it, the passes take the window's
    prompt_out: prompt_lp.Output, // a head pass's logprob words, then its picks, host-mapped (rank 0 reads them)
    structured: bool = false, // a structured output step: every rank's draws gather whole rows, rank 0 masks them

    /// Opens rank `rank` of `world` over `comm`; the ranks agree on settings (cuda_ranks) before the weights load.
    pub fn init(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, ctx: *const cuda.Context, checkpoint: []const u8, lib: *const cuda.nccl.Library, comm: cuda.nccl.Comm, rank: usize, world: usize, o: Options) !*Engine {
        if (world > ranks.max_world) return error.TooManyRanks; // before anything is allocated
        const e = try gpa.create(Engine);
        errdefer gpa.destroy(e);
        var arena = std.heap.ArenaAllocator.init(gpa);
        defer arena.deinit();
        const a = arena.allocator();
        const cfg_path = try std.fs.path.join(a, &.{ checkpoint, "config.json" });
        const text = try std.Io.Dir.cwd().readFileAlloc(io, cfg_path, a, .limited(1 << 24));
        var why: modelopt.Why = .{};
        var admission = @import("cuda_checkpoint.zig").Admission.read(gpa, text, &why) catch |err| {
            std.log.err("{s}: {s}", .{ cfg_path, why.text() });
            return err;
        };
        defer admission.deinit();
        const c = try config.parse(gpa, text);
        var stream = try cuda.Stream.init(d, false);
        errdefer stream.deinit();
        var settings = o.settings;
        var startup_problem: ?anyerror = null;
        settings.draft = ranks.effectiveDraft(settings.draft, o.mtp) catch |err| blk: {
            startup_problem = err;
            break :blk settings.draft;
        };
        if (startup_problem == null) {
            if (ranks.validateDraftOwner(settings, o.drafter != null)) |err| startup_problem = err;
        }
        if (settings.draft == .dflash2 and startup_problem == null) {
            if (world != 2 or c.hidden != 4096 or c.hc != 4 or c.vocab != 154880 or (!o.partial_dflash_check and settings.layers != 0 and settings.layers != c.layers)) startup_problem = error.UnsupportedDFlashTarget;
            if (startup_problem == null) {
                if (settings.draft_dir) |draft_dir| {
                    settings.draft_checkpoint = ranks.hashDraftCheckpoint(gpa, io, draft_dir) catch |err| blk: {
                        startup_problem = err;
                        break :blk @splat(0);
                    };
                } else startup_problem = error.NoDraftCheckpoint;
            }
        }
        if (settings.weight_dense.native() and startup_problem == null) {
            if (ctx.capability()) |capability| {
                if (capability != 121) startup_problem = error.UnsupportedNativeDenseDevice;
            } else |err| startup_problem = err;
        }
        if ((settings.dense_bits != 16 or settings.head_bits != 16) and !settings.tiled)
            startup_problem = error.PackedNeedsGlmMm;
        // Canonical --drafter is the default owner. Env-selected DFlash stays only when --drafter is absent.
        // Read content and decode effective precision before the paired startup vote and before any warmup.
        const startup = try a.alloc(u32, world);
        const with_mtp = ranks.targetMtp(o.drafter != null, settings.draft);
        const load_draft_head = ranks.loadDraftHead(o.drafter != null, settings.draft);
        const checkpoint_digest = try ranks.artifact(a, io, checkpoint);
        var fp = ranks.Fingerprint.of(world, rank, settings, o.capacity, with_mtp, try ranks.hashFile(a, io, cfg_path), try ranks.hashFile(a, io, "/proc/self/exe"), checkpoint_digest);
        if (o.vision) |lim| ranks.mediaFingerprint(&fp, lim);
        const canonical = try ranks.applyCanonicalStartup(gpa, io, d, lib, comm, stream, rank, .{ .dir = o.drafter, .bits = o.drafter_bits, .drafts = o.mtp }, o.drafter != null, &startup_problem, &fp, startup, try a.alloc(ranks.Fingerprint, world));
        try ranks.probeStart(d, lib, comm, stream, settings, rank, world);
        if (o.settings.kv_pool_tokens > 0) {
            if (o.settings.kv_shard and (world != 2 or c.mla_heads % 2 != 0)) return error.KvShardNeedsTp2;
            if (c.kv_lora != 512 or c.i_dim != 128) return error.KvPoolShape;
            _ = try cache_pool.extent.Pool.init(o.settings.kv_pool_tokens, if (o.settings.kv_shard) 2 else 1);
        }
        try validateKdaProjection(&c, world, o.settings.weight_dense.get(.kda) != .bf16);
        const share = try shard.Share.of(&c, rank, world);
        if (settings.exl3_routed == .mia_mcg4) try mia.validate(c.hidden, share.moe.len(), c.experts, c.topk + 1);
        const spans = try gpa.alloc(shard.Span, world);
        errdefer gpa.free(spans);
        for (spans, 0..) |*sp, q| sp.* = (try shard.Share.of(&c, q, world)).vocab;
        var m = try fw.Modules.load(d, ctx);
        errdefer m.deinit();
        if (settings.prompt_latent_aot) {
            var prompt_problem: ?anyerror = null;
            if (@import("cuda_prompt_latent_aot.zig").Kernels.load(gpa, d, ctx)) |kernels| m.prompt_latent = kernels else |err| {
                prompt_problem = err;
            }
            try ranks.gather(u32, d, lib, comm, stream, rank, @intFromBool(prompt_problem == null), startup);
            for (startup) |ready| if (ready == 0) return prompt_problem orelse error.PeerPromptLatentRefused;
        }
        if (settings.affine_prompt_policy == .glm_production) try m.affine4.enablePromptTiles(m.mods[7]);
        if (settings.weight_dense.source()) {
            var source_problem: ?anyerror = null;
            m.enableSourceDense(d) catch |err| {
                source_problem = err;
            };
            try ranks.gather(u32, d, lib, comm, stream, rank, @intFromBool(source_problem == null), startup);
            for (startup) |ready| if (ready == 0) return source_problem orelse error.PeerSourceDenseRefused;
        }
        if (settings.weight_dense.native()) {
            var native_problem: ?anyerror = null;
            m.enableNativeDense(d, ctx) catch |err| {
                native_problem = err;
            };
            try ranks.gather(u32, d, lib, comm, stream, rank, @intFromBool(native_problem == null), startup);
            for (startup) |ready| if (ready == 0) return native_problem orelse error.PeerNativeDenseRefused;
        }
        if (settings.exl3_routed == .mia_mcg4) m.mia = try mia.Kernels.load(d, cuda.kernels.glm_exl3);
        m.experts.exl3.decode = o.settings.exl3_decode;
        var pieces = try fw.Pieces.init(d, o.settings.pieces);
        errdefer pieces.deinit();
        if (o.settings.layers > c.layers) return error.BadLayers;
        const layers = try a.alloc(u32, if (o.settings.layers > 0) o.settings.layers else c.layers);
        for (layers, 0..) |*l, i| l.* = @intCast(i);
        e.* = .{ .capture_lifetime = target_graph.newLifetime(), .gpa = gpa, .io = io, .heat_gate = .{ .bands = settings.heat, .root = @import("cuda_engine_prof.zig").heatRoot() }, .shared_buf = undefined, .d = d, .c = c, .share = share, .spans = spans, .lib = lib, .comm = comm, .checkpoint_digest = checkpoint_digest, .settings = settings, .canonical_owner = o.drafter != null, .canonical = canonical, .m = m, .stream = stream, .pieces = pieces, .w = undefined, .capacity = o.capacity, .bp = undefined, .bd = undefined, .bm = undefined, .greedy_rows = undefined, .smp = undefined, .prompt_out = undefined, .mtp = with_mtp };
        errdefer e.replicas.deinit();
        e.tp_group = try cuda.tp_group.Group.borrow(lib, comm, rank, world);
        errdefer if (e.tp_group) |*group| group.closeFast();
        if (settings.communication == .roce) {
            try e.tp_group.?.openRoce(gpa, io, d, stream, settings.roce_max_bytes);
            if (e.tp_group.?.fast == null) return error.RequiredRoceUnavailable;
        }
        std.log.info("GLM collective rank={d} requested={s} effective={s} cap_bytes={d} prompt_pieces=nccl control=nccl", .{ rank, @tagName(settings.communication), if (e.tp_group.?.fast != null) "roce" else "nccl", if (e.tp_group.?.fast) |fast| fast.max_bytes else 0 });
        var wt = e.m.tools();
        wt.precision = .{ .dense = o.settings.weight_dense, .head = o.settings.weight_head };
        wt.exl3_routed = settings.exl3_routed;
        wt.recipes = if (admission.recipes) |*t| t else null;
        if (rank == 0) {
            var policy_text: [256]u8 = undefined;
            const policy_hash = o.settings.weight_dense.digest();
            std.log.info("GLM dense conversion policy v1: {s}; head={s}; scope=bf16-source-only; digest={s}", .{ try o.settings.weight_dense.format(&policy_text), @tagName(o.settings.weight_head), std.fmt.bytesToHex(policy_hash, .lower) });
        }
        const Tile = struct {
            gpa: std.mem.Allocator,
            mk: *const mm_mod.Kernels,
            s: cuda.Stream,
            d: *const cuda.Driver,
            enabled: bool,
            dense: u8,
            head: u8,
            packs: *mm_mod.Packed,
            fn apply(t: @This(), w: *weights.Weights) !void {
                if (t.enabled) t.packs.* = try tile.tileLinears(t.gpa, w, t.mk, t.s, t.d, t.dense, t.head);
            }
        };
        const tiler: Tile = .{ .gpa = gpa, .mk = &e.m.mm, .s = stream, .d = d, .enabled = settings.tiled, .dense = settings.dense_bits, .head = settings.head_bits, .packs = &e.packs };
        errdefer {
            gpa.free(e.packs.ptrs);
            gpa.free(e.packs.bits);
        }
        var loaded: ?weights.Weights = null;
        errdefer if (loaded) |*w| w.deinit();
        var load_problem: ?anyerror = null;
        loaded = load: {
            const w = weights.load(gpa, io, d, stream, wt, checkpoint, c, share, layers, load_draft_head) catch |err| {
                load_problem = err;
                break :load null;
            };
            break :load @import("cuda_load_owner.zig").finish(w, tiler, Tile.apply) catch |err| {
                load_problem = err;
                break :load null;
            };
        };
        try ranks.gather(u32, d, lib, comm, stream, rank, @intFromBool(load_problem == null), startup);
        for (startup) |ready| if (ready == 0) return load_problem orelse error.PeerWeightsRefused;
        e.w = loaded.?;
        loaded = null;
        errdefer e.w.deinit();
        errdefer if (e.prefetch) |*p| p.deinit() catch {};
        if (settings.prefetch.mode != null) {
            var prefetch_problem: ?anyerror = null;
            var builder: ?l2pf_plan.Builder = l2pf_plan.build(gpa, settings.prefetch, &e.w) catch |err| blk: {
                prefetch_problem = err;
                break :blk null;
            };
            defer if (builder) |*b| b.deinit();
            if (builder) |*b| {
                e.prefetch = l2pf.Prefetch.init(gpa, d, b) catch |err| blk: {
                    prefetch_problem = err;
                    break :blk null;
                };
                for (b.sites.items, 0..) |site, index| if (site) |s| {
                    std.log.info("GLM L2 plan rank={d} layer={d} site={s} bytes={d} pieces={d} grid={d} mode={s} row_limit={d}", .{ rank, index / 3, @tagName(@as(l2pf_plan.Site, @fromBackingInt(@intCast(index % 3)))), s.bytes, s.count, s.grid, @tagName(settings.prefetch.mode.?), settings.prefetch.max_rows });
                };
            }
            try ranks.gather(u32, d, lib, comm, stream, rank, @intFromBool(prefetch_problem == null), startup);
            for (startup) |ready| if (ready == 0) return prefetch_problem orelse error.PeerPrefetchRefused;
        }
        for (e.w.layers) |layer| switch (layer.mixer) {
            .kda => {},
            .mla => |l| std.log.info("GLM loaded DSA rank={d} layer={d} input_layout={s} input={s} qa={d} kva={d} hidden={d} qb={s} out={s} index_kw={s} index_qb={s} absorb=bf16 norms=bf16", .{
                rank,                         layer.index,
                @tagName(l.input),
                switch (l.input) {
                    .combined_bf16 => "combined/bf16",
                    .combined => |p| projectionFormat(p),
                    .separate => "separate",
                },
                l.q_lora,                     l.lw,
                l.dims,                       projectionFormat(l.q_b),
                projectionFormat(l.o),        projectionFormat(l.index.kw),
                projectionFormat(l.index.qb),
            }),
        };
        for (e.w.layers) |layer| switch (layer.ffn) {
            .dense => |l| std.log.info("GLM loaded MLP rank={d} layer={d} input_layout={s} width={d} hidden={d} down={s}", .{ rank, layer.index, @tagName(l.input), l.width, l.dims, projectionFormat(l.down) }),
            .moe => {},
        };
        std.log.info("GLM loaded head rank={d} format={s} kv={s} prelaunch_requested={}", .{ rank, projectionFormat(e.w.head), @tagName(settings.kv), settings.draft_prelaunch });
        if (settings.weight_dense.native()) {
            inline for (@import("cuda_weight_precision.zig").groups) |group| {
                std.log.info("GLM native dense rank={d} family={s} mode={s} matrices={d}", .{ rank, @tagName(group), @tagName(settings.weight_dense.get(group)), e.w.native_conversions[@backingInt(group)] });
            }
        }
        e.m.mm.prompt_fixed_k = settings.prompt_fixed_k;
        e.m.tiled = settings.tiled;
        if (settings.tiled) e.m.packs = &e.packs;
        try e.bind(0);
        errdefer {
            for (0..e.seqs.items.len) |slot| e.free(slot);
            e.seqs.deinit(gpa);
        }
        e.bp = if (o.settings.kv_shard) try fw.Buffers.initOwned(d, &e.w, fw.prefill_rows, true, o.capacity) else try fw.Buffers.init(d, &e.w, fw.prefill_rows, true, o.capacity);
        errdefer e.bp.deinit();
        e.bd = if (o.settings.kv_shard) try fw.Buffers.initOwned(d, &e.w, fw.window_rows, false, o.capacity) else try fw.Buffers.init(d, &e.w, fw.window_rows, false, o.capacity);
        errdefer e.bd.deinit();
        e.bm = if (o.settings.kv_shard) try fw.Buffers.initOwned(d, &e.w, fw.window_rows, false, o.capacity) else try fw.Buffers.init(d, &e.w, fw.window_rows, false, o.capacity);
        errdefer e.bm.deinit();
        const verify_rows = sharedBufferRows(settings, o.drafter != null);
        if (rank == 0 and (settings.draft == .dflash2 or o.drafter != null)) {
            const plan = verify_plan.Plan.effective(settings.dflash_verify, settings.draft == .dflash2, o.drafter != null);
            std.log.info("GLM DFlash verify plan: drafts={d} lane_rows={d} shared_rows={d}", .{ plan.drafts(), 1 + plan.drafts(), verify_rows });
        }
        if (with_mtp or settings.draft == .dflash2 or o.drafter != null) e.bs = if (o.settings.kv_shard) try fw.Buffers.initSharedOwned(d, &e.w, verify_rows, o.capacity) else try fw.Buffers.initShared(d, &e.w, verify_rows, o.capacity);
        errdefer if (e.bs) |*b| b.deinit();
        const route_samples = try route_trace.samples(ranks.environment("TF_GLM_ROUTE_SAMPLES"));
        const route_participants = try route_trace.participants(ranks.environment("TF_GLM_ROUTE_PARTICIPANTS"));
        if (route_samples > 0) {
            const b = e.bs orelse return error.RouteTraceNeedsSharedVerify;
            e.route_trace = try route_trace.Trace.init(gpa, d, rank, e.w.layers.len, b.rows, c.experts, c.topk, route_samples);
            e.route_trace.?.min_participants = route_participants;
            std.log.warn("GLM routing trace rank={d} samples={d}: diagnostic copies perturb timing", .{ rank, route_samples });
        }
        errdefer if (e.route_trace) |*trace| trace.deinit();
        errdefer if (e.dflash) |drafter| drafter.deinit();
        if (settings.draft == .dflash2 or o.drafter != null) {
            try e.bp.enableTaps(d);
            try e.bd.enableTaps(d);
            if (e.bs) |*b| try b.enableTaps(d);
        }
        if (settings.exl3_routed == .mia_mcg4) {
            try e.bp.enableMia(d, &e.w);
            try e.bd.enableMia(d, &e.w);
            try e.bm.enableMia(d, &e.w);
            if (e.bs) |*b| try b.enableMia(d, &e.w);
        }
        e.greedy_rows = try greedy_mod.Greedy.init(gpa, d, world, @max(fw.window_rows, shared_rows));
        errdefer e.greedy_rows.deinit();
        e.smp = try sampler.Sampler.init(d, c.vocab, fw.window_rows);
        errdefer e.smp.deinit();
        errdefer e.freeProfs();
        if (prof_mod.Prof.wanted()) for (&e.profs) |*p| {
            p.* = try prof_mod.Prof.init(d, io);
        };
        if (prof_mod.Prof.wanted()) for (&e.prompt_profs) |*p| {
            p.* = try prof_mod.Prof.init(d, io);
            p.*.?.every = 4;
        };
        e.prompt_out = try prompt_lp.Output.init(d);
        errdefer e.prompt_out.free();
        e.shared_buf = try cuda.DeviceBuffer.alloc(d, 8);
        errdefer e.shared_buf.free();
        if (settings.draft == .dflash2 and o.drafter == null) {
            var draft_problem: ?anyerror = null;
            e.dflash = dflash_mod.DFlash.open(gpa, io, checkpoint, settings.draft_dir.?, max_shared, o.capacity, .{ .driver = d, .context = ctx, .stream = stream, .embed = e.w.embed, .lib = lib, .comm = comm, .rank = rank, .world = world, .kernels_dir = o.draft_kernels, .verify_plan = settings.dflash_verify, .tp_group = e.collectiveGroup() }) catch |err| blk: {
                draft_problem = err;
                break :blk null;
            };
            if (!try e.pairedReady(draft_problem == null)) return draft_problem orelse error.PeerDFlashOpen;
            if (settings.draft_prelaunch and !try e.pairedReady(e.dflash.?.asyncAvailable())) return error.RequiredDraftPrelaunchUnavailable;
            e.dflash.?.prepare() catch |err| {
                e.pair_failure(err);
                return err;
            };
        }
        if (settings.draft_prelaunch and settings.draft != .dflash2 and o.drafter == null) return error.RequiredDraftPrelaunchUnavailable;
        std.log.info("GLM DFlash prelaunch rank={d} requested={} effective={}", .{ rank, settings.draft_prelaunch, settings.draft_prelaunch and e.dflash != null and e.dflash.?.asyncAvailable() });
        errdefer if (e.pool) |*p| p.deinit();
        if (o.settings.kv_pool_tokens > 0) {
            const shape = try cache_pool.Shape.init(o.settings.kv_pool_tokens, if (o.settings.kv_shard) 2 else 1, e.cur.st.caches.len + 1, o.settings.kv);
            const rooms = try a.alloc(u64, world);
            const mine = if (o.room) |r| r.read(r.ctx) orelse 0 else std.math.maxInt(u64);
            try ranks.gather(u64, d, lib, comm, stream, rank, mine, rooms);
            for (rooms) |room| if (shape.bytes > room) return error.NoKvPoolRoom;
            var failed: ?anyerror = null;
            e.pool = cache_pool.Pool.init(d, shape, rank, o.settings.kv) catch |err| blk: {
                failed = err;
                break :blk null;
            };
            try ranks.gather(u64, d, lib, comm, stream, rank, @intFromBool(failed == null), rooms);
            for (rooms) |ready| if (ready == 0) return failed orelse error.PeerKvPoolFailed;
        }
        errdefer if (e.ffn_graphs) |*bank| bank.deinit();
        if (settings.target_graphs == .ffn) {
            var graph_problem: ?anyerror = null;
            if (route_samples > 0) graph_problem = error.TargetGraphsWithRouteTrace else {
                e.ffn_graphs = target_graph.Bank.init(gpa, e.w.layers.len) catch |err| blk: {
                    graph_problem = err;
                    break :blk null;
                };
            }
            if (!try e.pairedReady(graph_problem == null)) return graph_problem orelse error.PeerTargetGraphsRefused;
            std.log.info("GLM partial target graphs rank={d}: FFN sites only; exact rows 1..{d}; attention remains eager", .{ rank, target_graph.max_rows });
        }
        errdefer if (e.full_graphs) |*runtime| runtime.deinit();
        if (settings.target_graphs == .full) {
            var graph_problem: ?anyerror = null;
            if (route_samples > 0) graph_problem = error.TargetGraphsWithRouteTrace else {
                e.full_graphs = target_forward.Runtime.init(gpa, d, &e.w, e.m.mods[0], settings.kv, o.capacity, settings.kv_shard) catch |err| blk: {
                    graph_problem = err;
                    break :blk null;
                };
            }
            if (!try e.pairedReady(graph_problem == null)) return graph_problem orelse error.PeerTargetGraphsRefused;
            std.log.info("GLM complete target graphs rank={d}: embedding, all target layers, taps and head; exact rows 1..{d}", .{ rank, target_graph.max_rows });
            e.full_graphs.?.tp_group = e.collectiveGroup();
            if (ranks.selectedDraftProfile(o.drafter != null, settings)) {
                e.prepareTargetGraphs() catch |err| {
                    e.pair_failure(err);
                    return err;
                };
                if (ctx.memInfo()) |room| {
                    std.log.info("GLM complete target graph settled memory rank={d}: free_bytes={d} total_bytes={d}", .{ rank, room.free, room.total });
                } else |err| std.log.warn("GLM complete target graph settled memory rank={d}: unavailable ({s})", .{ rank, @errorName(err) });
            } else std.log.info("GLM complete target graphs rank={d}: nonproduction profile, lazy first-use capture", .{rank});
        }
        e.synchronize() catch |err| {
            if (e.ffn_graphs != null or e.full_graphs != null) e.pair_failure(err);
            return err;
        };
        return e;
    }

    /// Production admissions see a complete exact-width bank, without allocating eight capacity-sized sequences.
    fn prepareTargetGraphs(e: *Engine) !void {
        const profile = @import("cuda_draft_profile.zig");
        const runtime = if (e.full_graphs) |*r| r else return error.NoFullTargetGraphs;
        const b = if (e.bs) |*scratch| scratch else return error.NoSharedWindows;
        if (e.capacity < profile.target_rows or b.rows < profile.shared_rows) return error.BadTargetGraphWarmupShape;
        const started = std.Io.Clock.awake.now(e.io).toNanoseconds();
        const before = runtime.bank.stats.site_captures;
        try e.reserveTargetWarmSlot(0);
        try e.reserveTargetWarmSlot(1);
        const ids: [profile.shared_rows]i32 = @splat(0);
        for (1..profile.shared_rows + 1) |rows| {
            try e.resetTargetWarmSlots();
            const window = try targetWarmWindow(rows);
            var segs: [2]fw.Seg = undefined;
            var row0: usize = 0;
            for (window.rows[0..window.count], 0..) |n, slot| {
                segs[slot] = .{ .st = &e.seqs.items[slot].?.st, .row0 = row0, .rows = n };
                row0 += n;
            }
            var one: [1]fw.Rank = .{.{ .w = &e.w, .st = segs[0].st, .b = b, .segs = segs[0..window.count] }};
            const f: fw.Forward = .{ .d = e.d, .m = &e.m, .stream = e.stream, .ranks = &one, .exchange = e.exchange(), .tp_group = e.collectiveGroup(), .prefetch = e.prefetchRuntime(), .replicas = &e.replicas, .pieces = null, .split = e.settings.split, .ring = e.settings.ring, .prefill_overlap = e.settings.prefill_overlap, .prefill_pieces = e.settings.prefill_pieces, .kda_prompt = e.settings.kda_prompt, .affine_prompt_policy = e.settings.affine_prompt_policy, .kda_wide = e.settings.kda_wide, .attn_fp8load = e.settings.attn_fp8load, .full_target = runtime, .target_buffer = .shared, .target_failure = e.pair_failure, .capture_lifetime = e.capture_lifetime };
            try f.run(ids[0..rows]);
        }
        for (1..profile.target_rows + 1) |rows| {
            try e.resetTargetWarmSlots();
            var one: [1]fw.Rank = undefined;
            const f = e.view(&e.bd, &one);
            try f.run(ids[0..rows]);
        }
        try e.resetTargetWarmSlots();
        try e.synchronize();
        e.release(1);
        e.release(0);
        const captures = runtime.bank.stats.site_captures - before;
        if (captures != profile.shared_rows + profile.target_rows) return error.IncompleteTargetGraphWarmup;
        const elapsed = @max(0, std.Io.Clock.awake.now(e.io).toNanoseconds() - started);
        std.log.info("GLM complete target graph warmup rank={d}: elapsed_ns={d} captures={d} shared=1..{d} lone=1..{d} temporary_slots=2", .{ e.share.rank, elapsed, captures, profile.shared_rows, profile.target_rows });
    }

    fn reserveTargetWarmSlot(e: *Engine, slot: u32) !void {
        var problem: ?anyerror = null;
        var region: ?cache_pool.extent.Region = null;
        e.bind(slot) catch |err| {
            problem = err;
        };
        if (problem == null) region = e.placement(@import("cuda_draft_profile.zig").shared_rows) catch |err| blk: {
            problem = err;
            break :blk null;
        };
        if (!try e.pairedReady(problem == null)) return problem orelse error.PeerTargetGraphWarmupRefused;
        if (region) |r| try e.reservePaired(slot, r);
    }

    fn resetTargetWarmSlots(e: *Engine) !void {
        for (0..2) |slot| {
            var problem: ?anyerror = null;
            e.bind(@intCast(slot)) catch |err| {
                problem = err;
            };
            if (problem == null) e.begin(null) catch |err| {
                problem = err;
            };
            if (!try e.pairedReady(problem == null)) return problem orelse error.PeerTargetGraphWarmupRefused;
        }
        try e.bind(0);
    }

    pub fn deinit(e: *Engine) void {
        if (e.disk) |p| p.close(p.ctx);
        e.synchronize() catch |err| {
            if (e.ffn_graphs != null or e.full_graphs != null) {
                e.pair_failure(err);
                return;
            }
        };
        e.pieces.side.synchronize() catch |err| {
            e.pair_failure(err);
            return;
        };
        if (e.ffn_graphs) |*bank| {
            const s = bank.stats;
            std.log.info("GLM partial target graph sites rank={d}: eager={d} captures={d} replays={d}", .{ e.share.rank, s.site_eager, s.site_captures, s.site_replays });
            bank.deinit();
        }
        if (e.full_graphs) |*runtime| {
            const s = runtime.bank.stats;
            std.log.info("GLM complete target graphs rank={d}: eager={d} captures={d} replays={d}", .{ e.share.rank, s.site_eager, s.site_captures, s.site_replays });
            runtime.deinit();
        }
        e.replicas.deinit();
        if (e.route_trace) |*trace| trace.deinit();
        var it = e.snaps.valueIterator();
        while (it.next()) |s| e.freeSnap(s);
        e.snaps.deinit(e.gpa);
        e.dropped.deinit(e.gpa);
        if (e.dflash) |drafter| drafter.deinit();
        if (e.drafter) |dr| dr.deinit();
        if (e.prefetch) |*p| p.deinit() catch |err| {
            e.pair_failure(err);
            return;
        };
        e.shared_buf.free();
        e.freeProfs();
        if (e.prompt_bufs) |*b| b.free();
        e.prompt_out.free();
        e.smp.deinit();
        e.greedy_rows.deinit();
        if (e.bs) |*b| b.deinit();
        e.bm.deinit();
        e.bd.deinit();
        e.bp.deinit();
        for (0..e.seqs.items.len) |i| e.free(i);
        e.seqs.deinit(e.gpa);
        if (e.pool) |*p| p.deinit();
        e.gpa.free(e.packs.ptrs);
        e.gpa.free(e.packs.bits);
        e.w.deinit();
        e.pieces.deinit();
        e.m.deinit();
        if (e.tp_group) |*group| group.closeFast();
        e.stream.deinit();
        e.gpa.free(e.spans);
        const gpa = e.gpa;
        gpa.destroy(e);
    }

    const freeProfs = @import("cuda_engine_prof.zig").freeProfs;

    pub fn exchange(e: *const Engine) fw.Exchange {
        return .{ .nccl = .{ .lib = e.lib, .comm = e.comm } };
    }

    fn collectiveGroup(e: *const Engine) ?*const cuda.tp_group.Group {
        return if (e.tp_group) |*group| group else null;
    }

    fn prefetchRuntime(e: *Engine) ?*l2pf.Prefetch {
        return if (e.prefetch) |*p| p else null;
    }

    pub fn drain(e: *const Engine) !void {
        try e.synchronize();
        try e.pieces.side.synchronize();
    }

    /// Before replacing scratch, modules, transport, taps or profilers: settle both streams and retire their graphs.
    pub fn invalidateCaptures(e: *Engine) !void {
        try e.drain();
        if (e.ffn_graphs) |*bank| try bank.retire(e.stream);
        if (e.full_graphs) |*r| try r.bank.retire(e.stream);
        e.capture_lifetime = target_graph.newLifetime();
    }

    /// A replacement owns a fresh generation even when its host/device addresses were recycled.
    /// On failure ownership of both the current drafter and `next` stays with their respective callers.
    pub fn replaceDrafter(e: *Engine, next: ?*dflash.Drafter) !void {
        if (e.drafter == next) return;
        try e.invalidateCaptures();
        if (e.drafter) |old| old.deinit();
        e.drafter = next;
    }

    fn synchronize(e: *const Engine) !void {
        try e.stream.synchronize();
        if (e.tp_group) |*group| try group.check();
    }

    fn graphBank(e: *Engine) ?*target_graph.Bank {
        if (!e.target_graphs_active) return null;
        return if (e.ffn_graphs) |*bank| bank else null;
    }

    fn fullGraph(e: *Engine) ?*target_forward.Runtime {
        if (!e.target_graphs_active) return null;
        return if (e.full_graphs) |*runtime| runtime else null;
    }

    pub fn targetGraphStats(e: *const Engine) target_graph.Stats {
        if (e.full_graphs) |runtime| return runtime.bank.stats;
        return if (e.ffn_graphs) |bank| bank.stats else .{};
    }

    fn view(e: *Engine, b: *fw.Buffers, out: *[1]fw.Rank) fw.Forward {
        out.* = .{.{ .w = &e.w, .st = &e.cur.st, .b = b }};
        return .{ .d = e.d, .m = &e.m, .stream = e.stream, .ranks = out, .exchange = e.exchange(), .tp_group = e.collectiveGroup(), .prefetch = e.prefetchRuntime(), .replicas = &e.replicas, .pieces = if (b == &e.bp) &e.pieces else null, .split = e.settings.split, .ring = e.settings.ring, .prefill_overlap = e.settings.prefill_overlap, .prefill_pieces = e.settings.prefill_pieces, .kda_prompt = e.settings.kda_prompt, .affine_prompt_policy = e.settings.affine_prompt_policy, .kda_wide = e.settings.kda_wide, .attn_fp8load = e.settings.attn_fp8load, .target_graphs = if (b == &e.bd) e.graphBank() else null, .full_target = if (b == &e.bd) e.fullGraph() else null, .target_buffer = .lone, .target_failure = e.pair_failure, .capture_lifetime = e.capture_lifetime, .images = if (e.images) |v| .{ .v = v, .slot = e.slot } else null, .taps = if (e.drafter) |dr| dr.taps(b == &e.bp) else null, .vision = if (b == &e.bp) if (e.cur.vision) |*vision| .{ .owner = vision.table(), .generation = vision.table().generation } else null else null };
    }

    /// Binds slot `slot`'s sequence for the steps that follow, making it (a zeroed state) on first use.
    pub fn bind(e: *Engine, slot: u32) !void {
        if ((e.drafter != null and slot >= dflash.MAX_S) or (e.settings.draft == .dflash2 and slot >= max_shared)) return error.BadDraftSlot;
        if (e.settings.kv_pool_tokens > 0 and slot >= cache_pool.extent.max_slots) return error.BadKvRequest;
        if (slot >= e.seqs.items.len) try e.seqs.appendNTimes(e.gpa, null, slot + 1 - e.seqs.items.len);
        if (e.seqs.items[slot] == null) {
            const q = try e.gpa.create(Seq);
            errdefer e.gpa.destroy(q);
            q.* = .{ .st = if (e.settings.kv_pool_tokens > 0) try fw.State.initShared(e.gpa, e.d, &e.w, e.capacity, e.settings.kv) else try fw.State.init(e.gpa, e.d, &e.w, e.capacity, e.settings.kv) };
            errdefer q.st.deinit();
            if (e.mtp) q.retained = try cuda.DeviceBuffer.alloc(e.d, fw.window_rows * e.c.hidden * 2);
            e.seqs.items[slot] = q;
        }
        e.cur = e.seqs.items[slot].?;
        e.slot = slot;
    }

    pub fn placement(e: *const Engine, rows: usize) !?cache_pool.extent.Region {
        if (e.pool) |*p| return try p.regions.plan(e.slot, rows);
        return null;
    }

    pub fn reserveAt(e: *Engine, region: cache_pool.extent.Region) !void {
        const p = if (e.pool) |*p| p else return error.NoKvPool;
        if (e.slot >= cache_pool.extent.max_slots) return error.BadKvRequest;
        if (p.regions.slots[e.slot]) |old| if (!std.meta.eql(old, region)) return error.KvAlreadyReserved;
        try p.regions.apply(e.slot, region);
        p.bind(&e.cur.st, region);
    }

    pub fn pairedReady(e: *Engine, valid: bool) !bool {
        errdefer e.pair_failure(error.PairedOperationFailed);
        if (e.share.world == 1) return valid;
        return try e.shared(if (valid) 0 else 1) == 0;
    }

    pub fn reservePaired(e: *Engine, slot: u32, region: cache_pool.extent.Region) !void {
        var problem: ?anyerror = null;
        e.bind(slot) catch |err| {
            problem = err;
        };
        const bound = problem == null;
        if (bound) e.reserveAt(region) catch |err| {
            problem = err;
        };
        if (!try e.pairedReady(problem == null)) {
            if (bound) e.release(slot);
            return error.KvReservationRefused;
        }
    }

    pub fn fits(e: *const Engine, prompt: usize, output: usize) bool {
        const p = if (e.pool) |*p| p else return true;
        const rows = cache_pool.extent.requestRows(prompt, output, fw.window_rows, e.capacity) catch return true;
        for (p.regions.slots, 0..) |maybe, slot| if (maybe == null) {
            _ = p.regions.plan(slot, rows) catch |err| return err != error.LaneMemory;
            return true;
        };
        return false;
    }

    /// Frees slot `slot`'s sequence (slot 0 stays: the first stream's, made at init) and binds slot 0.
    pub fn release(e: *Engine, slot: u32) void {
        if (e.share.rank == 0 and ranks.environment("TF_GLM_VERIFY_WORK") != null) {
            const graphs = e.targetGraphStats();
            var forwards: u64 = 0;
            var actual_rows: u64 = 0;
            for (e.verify_widths, 0..) |count, width| {
                forwards += count;
                actual_rows += count * width;
            }
            for (e.verify_lone_widths, 0..) |count, width| {
                forwards += count;
                actual_rows += count * width;
            }
            std.log.info("GLM verify work scope=engine cumulative=true slot={d} successful_verify_forwards={d} successful_verify_rows={d} graph_mode={s} graph_includes_startup=true graph_eager={d} graph_captures={d} graph_replays={d} shared_widths={any} lone_widths={any}", .{
                slot, forwards, actual_rows, @tagName(e.settings.target_graphs), graphs.site_eager, graphs.site_captures, graphs.site_replays, e.verify_widths, e.verify_lone_widths,
            });
        }
        e.replicas.invalidate();
        if (e.dflash) |drafter| drafter.release(slot);
        if (e.drafter) |dr| if (slot < dr.slots.items.len) {
            dr.slots.items[slot].from = 0;
            dr.slots.items[slot].v_rows = 0;
            dr.slots.items[slot].v_step = 0;
            dr.slots.items[slot].v_row0 = 0;
            dr.slots.items[slot].v_pos0 = 0;
        };
        if (e.pool) |*p| p.regions.release(slot);
        if (e.settings.vision and slot < e.seqs.items.len) {
            e.synchronize() catch {};
            e.clearVision(slot);
        }
        if (slot != 0 and slot < e.seqs.items.len) e.free(slot);
        e.cur = e.seqs.items[0].?;
        e.slot = 0;
    }

    fn free(e: *Engine, slot: usize) void {
        const q = e.seqs.items[slot] orelse return;
        e.synchronize() catch {};
        if (e.settings.vision) e.clearVision(slot);
        q.st.deinit();
        if (q.retained) |*rows| rows.free();
        e.gpa.destroy(q);
        e.seqs.items[slot] = null;
    }

    fn clearVision(e: *Engine, slot: usize) void {
        const q = e.seqs.items[slot] orelse return;
        if (q.vision) |*vision| switch (vision.*) {
            .leader => |*state| state.close() catch |err| std.log.err("vision cleanup: {s}", .{@errorName(err)}),
            .follower => |*state| {
                state.table.ready = false;
                state.table.features.free();
                e.gpa.free(state.ids);
                e.gpa.free(state.positions);
            },
        };
        q.vision = null;
    }

    /// Bytes one more stream's sequence takes on this rank, including retained MTP rows.
    pub fn seqBytes(e: *const Engine) u64 {
        const q = e.seqs.items[0].?;
        return q.st.arena.buf.len + if (q.retained) |rows| rows.len else @as(usize, 0);
    }

    /// The committed length: the pending token sits at this position, a window's row r draws position pos + 1 + r.
    pub fn pos(e: *const Engine) usize {
        return e.cur.st.pos;
    }

    pub fn slotPos(e: *const Engine, slot: u32) !usize {
        if (slot >= e.seqs.items.len) return error.BadDraftSlot;
        const seq = e.seqs.items[slot] orelse return error.BadDraftSlot;
        return seq.st.pos;
    }

    /// A fresh sequence with `sampling` (null or temperature 0: greedy).
    pub fn beginFollowing(e: *Engine, sampling: ?sampler.Sampling) !void {
        if (e.images) |v| v.clear();
        return e.begin(sampling);
    }

    pub fn begin(e: *Engine, sampling: ?sampler.Sampling) !void {
        if (e.pool != null and e.slot >= cache_pool.extent.max_slots) return error.BadKvRequest;
        if (e.pool) |*p| if (p.regions.slots[e.slot] == null) {
            const r = try p.regions.plan(e.slot, e.capacity);
            try e.reserveAt(r);
        };
        e.cur.sampling = sampler.check(sampling);
        e.cur.in_head = 0;
        try e.cur.st.reset(e.stream);
        e.replicas.invalidate();
        if (e.dflash) |drafter| try drafter.reset(e.slot);
        if (e.drafter) |dr| try dr.restart(e.slot, 0);
    }

    /// decode.prefill: `ids` in 2,048-row chunks, then the first token from the last row; `from` restores a kept state.
    pub fn prefill(e: *Engine, ids: []const u32, from: ?Resume, cancel: ?Cancel, marks: ?Marks) !u32 {
        var start = try e.prefillStart(ids.len, from);
        while (true) : (start += fw.prefill_rows) {
            if (try e.prefillChunk(ids[start..@min(ids.len, start + fw.prefill_rows + 1)], start, ids.len, null, cancel, marks)) |first| return first;
        }
    }

    /// Where a prompt pass of `len` ids starts: 0, or `from`'s position with its kept state restored.
    fn prefillStart(e: *Engine, len: usize, from: ?Resume) !usize {
        if (len == 0 or len + fw.window_rows >= e.cur.st.capacity) return error.PromptTooLong;
        const r = from orelse return 0;
        if (r.at == 0 or r.at >= len or r.at % fw.prefill_rows != 0) return error.BadResume;
        try e.restoreAt(r.id, r.at);
        if (e.drafter) |dr| {
            if (!dr.snapshotReady(e.slot, r.at)) try dr.restart(e.slot, r.at);
        }
        return r.at;
    }

    /// One chunk of a `len`-id prompt pass: `part` from `start` plus the next id; first token at the end, else null.
    pub fn prefillChunk(e: *Engine, part: []const u32, start: usize, len: usize, from: ?Resume, cancel: ?Cancel, marks: ?Marks) !?u32 {
        if ((from != null or start == 0) and try e.prefillStart(len, from) != start) return error.BadChunk;
        if (start >= len or start % fw.prefill_rows != 0) return error.BadChunk;
        const rows = @min(len - start, fw.prefill_rows);
        if (part.len != @min(len - start, rows + 1)) return error.BadChunk;
        var one: [1]fw.Rank = undefined;
        const f = e.view(&e.bp, &one);
        const as_i32: []const i32 = @ptrCast(part);
        if (marks) |m| if (start > 0 and std.mem.indexOfScalar(u32, m.at, @intCast(start)) != null) m.call(m.ctx, @intCast(start));
        try e.gate(cancel);
        try f.run(as_i32[0..rows]);
        if (e.prompt_logprobs) |p| if (part.len > 1) try prompt_lp.run(e, p, part[1..]);
        if (e.drafter) |dr| try dr.prompt(e, e.slot, start, rows, len);
        const last = start + rows == len;
        var first: ?u32 = null;
        if (last) {
            var t: [1]u32 = undefined;
            try e.draw(e.bp.logits, 1, len, 0, &t);
            first = t[0];
        }
        if (f.prof) |p| p.mark(e.stream, .draw);
        if (last) try e.holdPrompt(e.bp.fnormed + (rows - 1) * e.c.hidden * 2);
        if (e.mtp) {
            const nxt = as_i32[1..];
            if (nxt.len > 0) { // decode._absorb_rows
                try f.mtp(nxt, &.{e.bp.fnormed});
                try e.cur.st.setMtpLen(e.stream, e.cur.st.mtp_len + nxt.len);
            }
        }
        try f.commit(rows);
        if (e.dflash) |drafter| {
            errdefer e.pair_failure(error.DFlashCommitFailed);
            try drafter.acceptPrompt(e.slot, e.bp.taps.?.ptr, e.bp.rows, rows, start);
        }
        if (last) e.logPromptDispatch(rows, false);
        return first;
    }

    fn logPromptDispatch(e: *Engine, rows: usize, shared_prompt: bool) void {
        const actual = fw.promptDispatch();
        const new_wy = actual.wy[0] > 0 and !e.prompt_wy_logged;
        const new_overlap = actual.overlap[0] + actual.overlap[1] > 0 and !e.prompt_overlap_logged;
        if (!new_wy and !new_overlap and !e.settings.prompt_latent_aot) return;
        std.log.info("GLM prompt dispatch scope=process rank={d} slot={d} rows={d} shared={} target_readback=passed wy_launches={d} wy_layer_rows={d} overlap1_chunks={d} overlap2_chunks={d} interleaved_front_rows_kda_dsa_dense_shared={any} latent_aot_absorb_expand_sparse={any}", .{
            e.share.rank, e.slot, rows, shared_prompt, actual.wy[0], actual.wy[1], actual.overlap[0], actual.overlap[1], actual.fronts, actual.latent_aot,
        });
        e.prompt_wy_logged = e.prompt_wy_logged or new_wy;
        e.prompt_overlap_logged = e.prompt_overlap_logged or new_overlap;
    }

    /// The chunk gate on every rank: a kept state at the mark, the heat wait, rank 0's cancel; the id exchanged first.
    pub fn gate(e: *Engine, cancel: ?Cancel) !void {
        if (e.pool != null) {
            while (true) {
                const kept = e.shared(0) catch |err| {
                    e.pair_failure(err);
                    return err;
                };
                if (kept == 0) break;
                e.capturePaired(@intFromFloat(kept)) catch |err| switch (err) {
                    error.SnapshotRefused => {},
                    else => return err,
                };
            }
        } else {
            const kept = try e.shared(@floatFromInt(e.announce));
            e.announce = 0;
            if (kept > 0 and e.share.rank != 0) try e.keepAs(@intFromFloat(kept));
        }
        if (e.heat_gate.bands != null) {
            var w: Wait = .{ .e = e, .cancel = cancel };
            const before = e.heat_gate.waited_s;
            e.heat_gate.waitUntil(Wait.read, Wait.sleep, &w, .{ .ptr = &w, .check = Wait.stop }) catch |err| return w.failed orelse err;
            if (e.heat_gate.waited_s > before) heat.note(e.heat_gate.waited_s - before);
        }
        if (try e.shared(if (cancel) |c| @floatFromInt(@intFromBool(c.check(c.ptr))) else 0) > 0) return error.Cancelled;
    }

    pub const PromptPart = struct { slot: u32, ids: []const u32 };

    /// Complete cold prompts share row work; every sequence keeps its own recurrence, attention, head and taps.
    pub fn prefillShared(e: *Engine, parts: []const PromptPart, out: []u32) !void {
        if (parts.len < 2 or parts.len > max_shared or out.len != parts.len or e.mtp) return error.BadPromptBatch;
        var segs: [max_shared]fw.Seg = undefined;
        var ids: [prompt_rows]u32 = undefined;
        var rows: usize = 0;
        for (parts, 0..) |p, i| {
            if (p.ids.len == 0 or p.ids.len > prompt_batch_rows or p.ids.len > prompt_rows - rows) return error.BadPromptBatch;
            for (parts[0..i]) |prior| if (p.slot == prior.slot) return error.SlotTwice;
            try e.bind(p.slot);
            if (e.cur.st.pos != 0 or e.cur.st.pending != 0 or e.cur.vision != null) return error.BadPromptBatch;
            if (p.ids.len + fw.window_rows >= e.cur.st.capacity) return error.PromptTooLong;
            segs[i] = .{ .st = &e.cur.st, .row0 = rows, .rows = p.ids.len };
            @memcpy(ids[rows..][0..p.ids.len], p.ids);
            rows += p.ids.len;
        }
        try e.gate(null);
        var one: [1]fw.Rank = .{.{ .w = &e.w, .st = segs[0].st, .b = &e.bp, .segs = segs[0..parts.len] }};
        const f: fw.Forward = .{ .d = e.d, .m = &e.m, .stream = e.stream, .ranks = &one, .exchange = e.exchange(), .tp_group = e.collectiveGroup(), .prefetch = e.prefetchRuntime(), .replicas = &e.replicas, .prefill_overlap = e.settings.prefill_overlap, .prefill_pieces = e.settings.prefill_pieces, .kda_prompt = e.settings.kda_prompt, .affine_prompt_policy = e.settings.affine_prompt_policy, .kda_wide = e.settings.kda_wide, .attn_fp8load = e.settings.attn_fp8load, .taps = if (e.drafter) |dr| dr.taps(true) else null };
        try f.run(@ptrCast(ids[0..rows]));
        for (parts, segs[0..parts.len], 0..) |p, g, i| {
            try e.bind(p.slot);
            const last = g.row0 + g.rows - 1;
            try @import("cuda_proj.zig").matmul(f.set(), e.stream, e.w.head, e.bp.fnormed + last * e.c.hidden * 2, e.c.hidden, e.bp.logits, e.bp.kda.mm_part, 1, e.share.vocab.len(), e.c.hidden, false);
            try e.draw(e.bp.logits, 1, p.ids.len, 0, out[i..][0..1]);
            try e.holdPrompt(e.bp.fnormed + last * e.c.hidden * 2);
            if (e.drafter) |dr| {
                errdefer e.pair_failure(error.DFlashCommitFailed);
                try dr.promptSegment(e, p.slot, 0, g.rows, p.ids.len, g.row0);
            }
            if (e.dflash) |drafter| {
                errdefer e.pair_failure(error.DFlashCommitFailed);
                try drafter.acceptPrompt(p.slot, e.bp.taps.?.ptr + g.row0 * e.c.hidden * 2, e.bp.rows, g.rows, 0);
            }
            const own = e.view(&e.bp, &one);
            try own.commit(g.rows);
        }
        e.logPromptDispatch(rows, true);
    }

    /// Device bytes retained at a prompt mark; MTP has absorbed the same prefix at these marks.
    pub fn snapshotBytes(e: *const Engine, at: usize) u64 {
        const plan = e.cur.st.layout.snapshot(.{ .main = at, .mtp = if (e.mtp) at else 0 }) catch return std.math.maxInt(u64);
        return e.payloadBytes(plan.byteCount()) catch return std.math.maxInt(u64);
    }

    fn spillDraft(e: *const Engine) ranks.DraftSnapshot {
        return ranks.persistedDraftSnapshot(e.canonical_owner, e.canonical, dflash.snapshot_bytes, if (e.dflash) |drafter| drafter.snapshotBytes() else 0, e.settings.draft_checkpoint);
    }

    fn payloadBytes(e: *const Engine, target_bytes: u64) !u64 {
        return spill_payload.withDraftBytes(target_bytes, e.spillDraft().bytes);
    }

    pub fn spillShape(e: *const Engine) spill_payload.Shape {
        const st = &e.seqs.items[0].?.st;
        const saved = e.spillDraft();
        return .{ .layout = st.layout, .format = e.settings.kv, .mtp = e.mtp, .sharded = e.settings.kv_shard, .rank = @intCast(e.share.rank), .world = @intCast(e.share.world), .grid = prompt_rows, .parities = st.cur.len, .draft_bytes = saved.bytes, .draft_identity = saved.identity };
    }

    pub const HostSnapshot = struct { bytes: []u8, meta: [spill_payload.max_meta]u8, meta_len: usize };

    pub fn exportState(e: *Engine, id: u64, limit: u64) !HostSnapshot {
        const saved = e.snaps.getPtr(id) orelse return error.NoKeptState;
        if (saved.buf.len > limit) return error.SpillHostLimit;
        var host: HostSnapshot = .{ .bytes = undefined, .meta = undefined, .meta_len = 0 };
        const meta = try spill_payload.encode(e.spillShape(), saved.pos, saved.mtp_len, saved.owner, saved.cur, &host.meta);
        host.meta_len = meta.len;
        host.bytes = try e.gpa.alloc(u8, saved.buf.len);
        errdefer e.gpa.free(host.bytes);
        try e.synchronize();
        try saved.buf.download(0, host.bytes);
        return host;
    }

    pub fn importState(e: *Engine, id: u64, at: u32, meta: []const u8, bytes: u64, ctx: *anyopaque, pump: *const fn (*anyopaque, *cuda.DeviceBuffer) anyerror!void) !void {
        if (e.snaps.contains(id)) return error.StateKept;
        const state = try spill_payload.decode(e.spillShape(), meta, at, bytes);
        var buf = try cuda.DeviceBuffer.alloc(e.d, @intCast(bytes));
        errdefer buf.free();
        const cur = try e.gpa.alloc(u1, state.cur.len);
        errdefer e.gpa.free(cur);
        for (cur, state.cur) |*to, from| to.* = @intCast(from);
        try pump(ctx, &buf);
        try e.snaps.put(e.gpa, id, .{ .buf = buf, .pos = @intCast(state.pos), .mtp_len = @intCast(state.mtp_len), .cur = cur, .owner = state.owner });
    }

    pub fn persist(e: *Engine, command: persistence.Command) !void {
        const port = e.disk orelse return error.SpillDisabled;
        var status: u32 = 0;
        port.apply(port.ctx, command) catch |err| {
            status = if (err == error.NotStored) 1 else 2;
            if (status == 2) std.log.warn("GLM spill {s} at {d}: {s}", .{ @tagName(command.operation), command.at, @errorName(err) });
        };
        if (e.share.world > 1) status = @intFromFloat(e.shared(@floatFromInt(status)) catch |err| {
            e.pair_failure(err);
            return err;
        });
        if (status == 0 and command.operation == .read and e.share.world > 1) {
            const saved = e.snaps.get(command.id).?;
            const owner: u32 = if (saved.owner) |r| r + 1 else 0;
            const largest = e.shared(@floatFromInt(owner)) catch |err| {
                e.pair_failure(err);
                return err;
            };
            if (!try e.pairedReady(largest == @as(f64, @floatFromInt(owner)))) status = 2;
        }
        if (status == 0) return;
        if (command.operation == .read) e.discardState(command.id);
        if (status == 2 or command.operation == .write) port.apply(port.ctx, .{ .operation = .forget, .key = command.key, .at = command.at }) catch {};
        return if (status == 1) error.NotStored else error.SpillRefused;
    }

    pub fn discardState(e: *Engine, id: u64) void {
        var kept = e.snaps.fetchRemove(id) orelse return;
        e.freeSnap(&kept.value);
    }

    /// Rank 0, at a mark in a prompt pass: a copy of the state under a new id, named to the other ranks at the gate.
    pub fn keepState(e: *Engine) !u64 {
        const id = e.next_snap;
        e.next_snap += 1;
        if (e.pool != null) {
            // The follower is already at the chunk gate: announce before capture's status exchange.
            _ = e.shared(@floatFromInt(id)) catch |err| {
                e.pair_failure(err);
                return err;
            };
            try e.capturePaired(id);
        } else {
            try e.keepAs(id);
            e.announce = id;
        }
        return id;
    }

    fn capturePaired(e: *Engine, id: u64) !void {
        var problem: ?anyerror = null;
        e.keepAs(id) catch |err| {
            problem = err;
        };
        if (!try e.pairedReady(problem == null)) {
            if (problem == null) {
                var kept = e.snaps.fetchRemove(id).?;
                e.freeSnap(&kept.value);
            }
            return error.SnapshotRefused;
        }
    }

    fn keepAs(e: *Engine, id: u64) !void {
        if (e.snaps.contains(id)) return error.StateKept;
        const st = &e.cur.st;
        if (st.pending != 0 or e.cur.in_head != 0 or st.mtp_len != (if (e.mtp) st.pos else 0)) return error.NotAtMark;
        if (e.canonical_owner) {
            const dr = e.drafter orelse return error.NoDrafter;
            if (!dr.snapshotReady(e.slot, st.pos)) return error.NotAtMark;
        } else if (e.dflash) |drafter| if (!drafter.snapshotReady(e.slot, st.pos)) return error.NotAtMark;
        const plan = try st.layout.snapshot(.{ .main = st.pos, .mtp = st.mtp_len });
        const owner = if (st.layout.external_planes) st.mtp_caches.owner else null;
        const resident = owner == null or owner.? == e.share.rank;
        const target_bytes = plan.bytesFor(resident);
        var buf = try cuda.DeviceBuffer.alloc(e.d, @intCast(try e.payloadBytes(target_bytes)));
        errdefer buf.free();
        try buf.fill8(0, e.stream.handle);
        var it = plan.rangesFor(resident);
        while (it.next()) |r| try e.copy(buf.ptr + r.saved, st.address(r), r.bytes);
        const draft_at = buf.ptr + std.mem.alignForward(usize, target_bytes, 256);
        if (e.canonical_owner) {
            try e.drafter.?.save(e.slot, draft_at, st.pos, e.stream);
        } else if (e.dflash) |drafter| try drafter.save(e.slot, draft_at);
        try e.synchronize();
        const cur = try e.gpa.dupe(u1, e.cur.st.cur);
        errdefer e.gpa.free(cur);
        try e.snaps.put(e.gpa, id, .{ .buf = buf, .pos = e.cur.st.pos, .mtp_len = e.cur.st.mtp_len, .cur = cur, .owner = owner });
    }

    /// The live state made kept state `id`'s; a rank without it refuses rather than prefill from another position.
    fn restoreState(e: *Engine, id: u64) !void {
        return e.restoreAt(id, null);
    }

    fn restoreAt(e: *Engine, id: u64, expected: ?usize) !void {
        const st = &e.cur.st;
        const destination_owner = if (st.layout.external_planes) st.mtp_caches.owner else null;
        const s = e.snaps.getPtr(id);
        var problem: ?anyerror = null;
        var plan: ?@import("cuda_layout.zig").Plan = null;
        var source_resident = true;
        if (s) |saved| {
            if (expected != null and saved.pos != expected.?) problem = error.BadResume;
            if (problem == null) {
                plan = st.layout.snapshot(.{ .main = saved.pos, .mtp = saved.mtp_len }) catch |err| blk: {
                    problem = err;
                    break :blk null;
                };
                source_resident = saved.owner == null or saved.owner.? == e.share.rank;
                if (plan != null and ((e.payloadBytes(plan.?.bytesFor(source_resident)) catch 0) != saved.buf.len or saved.pos > st.capacity or saved.mtp_len > st.capacity or (saved.owner == null) != (destination_owner == null))) problem = error.BadKeptState;
            }
        } else problem = error.NoKeptState;
        if (destination_owner != null) {
            if (!try e.ownerTransport().prepared(e.stream, e.shared_buf, problem == null)) return error.KeptStateRefused;
        } else if (st.layout.external_planes) {
            if (!try e.pairedReady(problem == null)) return error.KeptStateRefused;
        }
        if (problem) |err| return err;
        errdefer if (st.layout.external_planes or e.dflash != null or e.drafter != null) e.pair_failure(error.PairedRestoreFailed);
        const saved = s.?;
        var full = plan.?.ranges();
        var local = plan.?.rangesFor(source_resident);
        while (full.next()) |r| {
            const from = if (r.plane == .latent and !source_resident) null else local.next();
            if (r.plane == .latent and destination_owner != null) {
                try e.ownerTransport().carry(e.stream, saved.owner.?, destination_owner.?, if (from) |f| saved.buf.ptr + f.saved else 0, if (destination_owner.? == e.share.rank) st.address(r) else 0, r.bytes);
            } else {
                try e.copy(st.address(r), saved.buf.ptr + from.?.saved, r.bytes);
            }
        }
        const draft_at = saved.buf.ptr + std.mem.alignForward(usize, plan.?.bytesFor(source_resident), 256);
        if (e.canonical_owner) {
            const dr = e.drafter orelse return error.NoDrafter;
            try dr.restore(e.slot, draft_at, saved.pos, e.stream);
        } else if (e.dflash) |drafter| try drafter.restore(e.slot, draft_at, saved.pos);
        try e.synchronize();
        try st.restoreHost(e.stream, saved.pos, saved.mtp_len, saved.cur);
        e.replicas.invalidate();
        e.cur.in_head = 0;
        try e.synchronize();
    }

    fn ownerTransport(e: *const Engine) owner_kv.Transport {
        return .{ .d = e.d, .lib = e.lib, .comm = e.comm, .rank = e.share.rank, .tp_group = e.collectiveGroup(), .failure = e.pair_failure };
    }

    pub fn holds(e: *const Engine, id: u64) bool {
        return e.snaps.contains(id);
    }

    /// Frees kept state `id`; rank 0 also notes it for the other ranks (cuda_follow sends it before its next step).
    pub fn dropState(e: *Engine, id: u64) void {
        var kv = e.snaps.fetchRemove(id) orelse return;
        e.freeSnap(&kv.value);
        if (e.share.rank == 0 and e.share.world > 1) e.dropped.append(e.gpa, @intCast(id)) catch
            std.log.warn("kept state {d}: the other ranks keep their copy until they exit (no memory for the note)", .{id});
    }

    fn freeSnap(e: *Engine, s: *Snap) void {
        s.buf.free();
        e.gpa.free(s.cur);
    }

    /// The ranks' maximum of `v` (an all-reduce on the engine's stream).
    pub fn shared(e: *Engine, v: f64) !f64 {
        var x = v;
        try e.shared_buf.uploadAsync(0, std.mem.asBytes(&x), e.stream.handle);
        try e.lib.check(e.lib.api.ncclAllReduce(e.shared_buf.ptr, e.shared_buf.ptr, 1, .f64, .max, e.comm, e.stream.handle), "ncclAllReduce");
        try e.synchronize();
        try e.shared_buf.download(0, std.mem.asBytes(&x));
        return x;
    }

    /// heat.Gate.waitUntil's hooks over the ranks: gathered max reading, rank 0's stop: all pause together.
    const Wait = struct {
        e: *Engine,
        cancel: ?Cancel,
        failed: ?anyerror = null,

        fn read(ptr: *anyopaque) anyerror!f64 {
            const w: *Wait = @ptrCast(@alignCast(ptr));
            const local: f64 = blk: {
                var dir = std.Io.Dir.cwd().openDir(w.e.io, w.e.heat_gate.root, .{ .iterate = true }) catch break :blk -std.math.inf(f64);
                defer dir.close(w.e.io);
                break :blk heat.hottest(w.e.io, dir) catch -std.math.inf(f64);
            };
            return w.e.shared(local);
        }

        fn sleep(ptr: *anyopaque) void {
            const w: *Wait = @ptrCast(@alignCast(ptr));
            std.Io.sleep(w.e.io, .fromMilliseconds(2000), .awake) catch {};
        }

        fn stop(ptr: *anyopaque) bool {
            const w: *Wait = @ptrCast(@alignCast(ptr));
            const mine: f64 = if (w.cancel) |c| @floatFromInt(@intFromBool(c.check(c.ptr))) else 0;
            const any = w.e.shared(mine) catch |err| {
                w.failed = err;
                return true;
            };
            return any > 0;
        }
    };

    /// A verify window: `ids` (pending token, then drafts) at the committed position; `out[r]` drawn at pos + 1 + r.
    pub fn verify(e: *Engine, ids: []const u32, out: []u32, bits: u64) !void {
        return e.verifyWithMask(ids, out, bits, false);
    }

    pub fn verifyMasked(e: *Engine, ids: []const u32, out: []u32) !void {
        return e.verifyWithMask(ids, out, 0, true);
    }

    fn verifyWithMask(e: *Engine, ids: []const u32, out: []u32, bits: u64, masked: bool) !void {
        errdefer if (e.full_graphs != null) e.pair_failure(error.FullTargetVerificationFailed);
        if (ids.len == 0 or ids.len > fw.window_rows) return error.WindowTooWide;
        var one: [1]fw.Rank = undefined;
        var f = e.view(&e.bd, &one);
        const at = e.cur.st.pos;
        f.prof = e.profs[ids.len - 1];
        if (f.prof) |p| try p.begin(e.stream, verifyLabel(ids.len));
        try f.run(@ptrCast(ids));
        try e.retainRows(e.bd.fnormed, ids.len);
        if (e.drafter) |dr| try dr.noteVerify(e.slot, 0, ids.len, at, true);
        if (e.dflash) |drafter| try drafter.stageWindow(e.slot, e.bd.taps.?.ptr, e.bd.rows, 0, ids.len, at);
        if (masked) try e.drawMasked(e.bd.logits, ids.len, at + 1, out) else try e.draw(e.bd.logits, ids.len, at + 1, bits, out);
        if (f.prof) |p| p.end();
        e.verify_lone_widths[ids.len] += 1;
    }

    /// Several streams' windows in one forward: shared row ops, per-part attention and KDA.
    pub fn verifyShared(e: *Engine, parts: []const Part) !void {
        errdefer if (e.full_graphs != null) e.pair_failure(error.FullTargetVerificationFailed);
        const b = if (e.bs) |*x| x else return error.NoSharedWindows;
        if (parts.len == 0 or parts.len > max_shared) return error.WindowTooWide;
        var segs: [max_shared]fw.Seg = undefined;
        var ids: [shared_rows]u32 = undefined;
        var rows: usize = 0;
        var all_greedy = true;
        for (parts, 0..) |p, i| {
            if (p.ids.len == 0 or p.ids.len > e.targetRows() or rows + p.ids.len > e.sharedCapacity()) return error.WindowTooWide;
            for (parts[0..i]) |q| if (q.slot == p.slot) return error.SlotTwice;
            try e.bind(p.slot);
            all_greedy = all_greedy and e.cur.sampling == null and p.bits == 0 and !e.structured;
            if (p.out.len < p.ids.len) return error.BadVerifyOutput;
            segs[i] = .{ .st = &e.cur.st, .row0 = rows, .rows = p.ids.len };
            @memcpy(ids[rows..][0..p.ids.len], p.ids);
            rows += p.ids.len;
        }
        var one: [1]fw.Rank = .{.{ .w = &e.w, .st = segs[0].st, .b = b, .segs = segs[0..parts.len] }};
        const trace = if (e.route_trace) |*t| if (parts.len >= t.min_participants) t.begin(rows) else null else null;
        errdefer if (trace) |t| t.abort(e.stream);
        if (trace) |t| {
            const Segment = struct { slot: u32, row0: usize, rows: usize, position: usize };
            var metadata: [max_shared]Segment = undefined;
            for (parts, segs[0..parts.len], metadata[0..parts.len]) |p, g, *segment| {
                segment.* = .{ .slot = p.slot, .row0 = g.row0, .rows = g.rows, .position = g.st.pos };
            }
            const line = try std.json.Stringify.valueAlloc(e.gpa, .{
                .kind = "verify",
                .rank = e.share.rank,
                .sample = t.sample,
                .rows = rows,
                .participants = parts.len,
                .tokens = ids[0..rows],
                .segments = metadata[0..parts.len],
                .expected_routed_layers = e.w.layers.len - @min(e.c.dense_layers, e.w.layers.len),
            }, .{});
            defer e.gpa.free(line);
            std.log.info("GLM_ROUTE {s}", .{line});
        }
        const f: fw.Forward = .{ .d = e.d, .m = &e.m, .stream = e.stream, .ranks = &one, .exchange = e.exchange(), .tp_group = e.collectiveGroup(), .prefetch = e.prefetchRuntime(), .replicas = &e.replicas, .pieces = null, .split = e.settings.split, .ring = e.settings.ring, .prefill_overlap = e.settings.prefill_overlap, .prefill_pieces = e.settings.prefill_pieces, .kda_prompt = e.settings.kda_prompt, .affine_prompt_policy = e.settings.affine_prompt_policy, .kda_wide = e.settings.kda_wide, .attn_fp8load = e.settings.attn_fp8load, .route_trace = trace, .target_graphs = e.graphBank(), .full_target = e.fullGraph(), .target_buffer = .shared, .target_failure = e.pair_failure, .capture_lifetime = e.capture_lifetime, .taps = if (e.drafter) |dr| dr.taps(false) else null };
        try f.run(@ptrCast(ids[0..rows]));
        if (trace) |t| try t.finish(e.stream);
        for (parts, segs[0..parts.len]) |p, g| {
            try e.bind(p.slot);
            e.cur.rows_at = b.fnormed + g.row0 * e.c.hidden * 2;
            if (e.dflash) |drafter| try drafter.stageWindow(p.slot, b.taps.?.ptr, b.rows, g.row0, g.rows, g.st.pos);
            if (e.drafter) |dr| try dr.noteVerify(p.slot, g.row0, g.rows, g.st.pos, g.row0 == 0);
            if (!all_greedy) try e.draw(b.logits + g.row0 * e.share.vocab.len() * 2, g.rows, g.st.pos + 1, p.bits, p.out);
        }
        if (all_greedy) {
            var tokens: [shared_rows]u32 = undefined;
            try e.drawGreedy(b.logits, rows, tokens[0..rows]);
            for (parts, segs[0..parts.len]) |p, g| @memcpy(p.out[0..g.rows], tokens[g.row0..][0..g.rows]);
        }
        e.verify_widths[rows] += 1;
    }

    pub fn validateDraftBlock(e: *const Engine, reqs: []const dflash.Req) !void {
        if (e.drafter == null) return error.NoDrafter;
        for (reqs) |q| {
            if (q.slot >= e.seqs.items.len or e.seqs.items[q.slot] == null) return error.UnknownSlot;
        }
    }

    pub fn draftBlock(e: *Engine, reqs: []dflash.Req) !void {
        try e.validateDraftBlock(reqs);
        const dr = e.drafter.?;
        for (reqs) |*q| q.pos = @intCast(e.seqs.items[q.slot].?.st.pos);
        try dr.draft(e, reqs);
    }

    pub const holdPrompt = @import("cuda_prompt_hold.zig").save;
    pub const prefillMany = @import("cuda_prompts.zig").prefillMany;

    /// forward.commit of the last window's first `n` rows (the KDA replay when fewer than it ran).
    pub fn keep(e: *Engine, n: usize) !void {
        var one: [1]fw.Rank = undefined;
        const f = e.view(&e.bd, &one);
        try f.commit(n);
        if (e.dflash) |drafter| {
            errdefer e.pair_failure(error.DFlashCommitFailed);
            try drafter.accept(e.slot, n, e.cur.st.pos);
        }
    }

    fn retainRows(e: *Engine, source: u64, rows: usize) !void {
        if (rows == 0 or rows > fw.window_rows) return error.WindowTooWide;
        if (e.cur.retained) |kept| {
            try e.copy(kept.ptr, source, rows * e.c.hidden * 2);
            e.cur.rows_at = kept.ptr;
        } else e.cur.rows_at = source;
    }

    fn draftRows(e: *const Engine, from_prompt: bool) u64 {
        return if (from_prompt) e.cur.retained.?.ptr else e.cur.rows_at;
    }

    /// decode.draft: the MTP head absorbs the kept rows (committed first) with `follow`, then chains `depth` drafts.
    pub fn drafting(e: *const Engine) bool {
        return e.mtp or e.dflash != null or e.drafter != null;
    }

    pub fn verifyPlan(e: *const Engine) verify_plan.Plan {
        return if (e.drafter != null) .dflash7 else e.settings.dflash_verify.active(e.dflash != null);
    }

    pub fn targetRows(e: *const Engine) usize {
        return if (e.settings.draft_profile.enabled) verify_plan.target_rows else 1 + e.verifyPlan().drafts();
    }

    pub fn sharedLimit(e: *const Engine) usize {
        return if (e.settings.draft_profile.enabled) @import("cuda_draft_profile.zig").shared_rows else e.verifyPlan().rows();
    }

    /// One geometry for shared buffers, admission and tap staging. Canonical --drafter is dflash7 (64 rows).
    pub fn sharedBufferRows(settings: ranks.Settings, canonical: bool) usize {
        return verify_plan.Plan.effective(settings.dflash_verify, settings.draft == .dflash2, canonical).rows();
    }

    pub fn sharedCapacity(e: *const Engine) usize {
        const advertised = e.sharedLimit();
        return if (e.bs) |*b| @min(b.rows, advertised) else advertised;
    }

    fn resolveDraftBatch(e: *const Engine, parts: []const DraftPart, out: *[max_shared]dflash_mod.BatchRequest) !void {
        const d = e.dflash orelse return error.NoDraftHead;
        if (parts.len == 0 or parts.len > max_shared) return error.BadDraftBatch;
        for (parts, 0..) |p, i| {
            if (p.depth > e.verifyPlan().drafts() or p.position == 0) return error.BadDraftRequest;
            const end = try e.slotPos(p.slot);
            if (p.position != end + 1) return error.PositionMismatch;
            out[i] = .{ .slot = p.slot, .end = end, .pending = p.pending, .depth = p.depth, .sampling = e.seqs.items[p.slot].?.sampling };
        }
        try d.validateBatch(out[0..parts.len]);
    }

    pub fn validateDraftBatch(e: *const Engine, parts: []const DraftPart) !void {
        var resolved: [max_shared]dflash_mod.BatchRequest = undefined;
        try e.resolveDraftBatch(parts, &resolved);
    }

    pub fn prelaunchDraftBatch(e: *Engine, parts: []const DraftPart) !void {
        var resolved: [max_shared]dflash_mod.BatchRequest = undefined;
        var problem: ?anyerror = null;
        e.resolveDraftBatch(parts, &resolved) catch |err| {
            problem = err;
        };
        if (!try e.pairedReady(problem == null)) return problem orelse error.PeerDraftBatchRefused;
        try e.dflash.?.prelaunchBatch(resolved[0..parts.len]);
    }

    pub fn draftBatch(e: *Engine, parts: []const DraftPart, out: []DraftResult) !void {
        if (out.len != parts.len) return error.BadDraftBatch;
        var resolved: [max_shared]dflash_mod.BatchRequest = undefined;
        var problem: ?anyerror = null;
        e.resolveDraftBatch(parts, &resolved) catch |err| {
            problem = err;
        };
        if (!try e.pairedReady(problem == null)) return problem orelse error.PeerDraftBatchRefused;
        try e.dflash.?.proposeBatch(resolved[0..parts.len], out);
    }

    pub fn draft(e: *Engine, from_prompt: bool, follow: []const u32, depth: usize, out: []u32) !usize {
        if (depth > e.verifyPlan().drafts() or depth > out.len or follow.len == 0) return error.BadDraftRequest;
        if (e.dflash) |drafter| return drafter.propose(e.slot, follow[follow.len - 1], e.cur.sampling, out[0..depth]);
        if (!e.mtp) return error.NoDraftHead;
        if (depth > verify_plan.Plan.baseline.drafts()) return error.TooDeep;
        var one: [1]fw.Rank = undefined;
        const f = e.view(&e.bm, &one);
        const hidden = if (from_prompt) @import("cuda_prompt_hold.zig").input(e) else e.cur.rows_at;
        return e.draftMtpWith(follow, depth, out, hidden, f, draw);
    }

    /// One shared absorption/proposal loop: real Forward/draw in serving, command recorders in CPU lifetime tests.
    /// State length writes remain the actual State.setMtpLen implementation in either case.
    pub fn draftMtpWith(e: *Engine, follow: []const u32, depth: usize, out: []u32, hidden: u64, f: anytype, draw_rows: anytype) !usize {
        if (depth > e.verifyPlan().drafts() or depth > out.len or follow.len == 0 or e.cur.in_head > e.cur.st.mtp_len) return error.BadDraftRequest;
        for (follow) |token| if (token >= e.c.vocab) return error.BadDraftRequest;
        const D = e.c.hidden;
        try e.cur.st.setMtpLen(e.stream, e.cur.st.mtp_len - e.cur.in_head);
        e.cur.in_head = 0;
        const next: []const i32 = @ptrCast(follow);
        var s0: usize = 0;
        while (s0 < next.len) : (s0 += fw.window_rows) {
            const part = next[s0..@min(next.len, s0 + fw.window_rows)];
            try f.mtp(part, &.{hidden + s0 * D * 2});
            try e.cur.st.setMtpLen(e.stream, e.cur.st.mtp_len + part.len);
        }
        for (0..depth) |j| {
            try draw_rows(e, e.bm.logits, 1, e.cur.st.pos + 1 + j, 0, out[j..][0..1]);
            if (j + 1 < depth) {
                try f.mtp(@ptrCast(out[j..][0..1]), &.{e.bm.fnormed});
                try e.cur.st.setMtpLen(e.stream, e.cur.st.mtp_len + 1);
                e.cur.in_head += 1;
            }
        }
        return depth;
    }

    /// `rows` logits rows of this rank's vocabulary -> the tokens at positions first, first + 1, ...
    pub fn draw(e: *Engine, logits: u64, rows: usize, first: u64, bits: u64, out: []u32) !void {
        if (e.structured) return e.drawWhole(logits, rows, first, bits, out);
        if (e.cur.sampling) |s| {
            try e.smp.draw(e.stream, e.exchange(), e.spans, &.{.{ .rank = e.share.rank, .logits = logits }}, rows, first, sampler.rule(s), out);
            if (e.tp_group) |*group| try group.check();
            return;
        }
        try e.drawGreedy(logits, rows, out);
    }

    /// The prompt's target logits survive MTP absorption; redraw them once the lane supplies its first mask.
    pub fn redrawQueued(e: *Engine, position: u64) !u32 {
        var out: [1]u32 = undefined;
        try e.drawMasked(e.bp.logits, 1, position, &out);
        return out[0];
    }

    fn drawMasked(e: *Engine, logits: u64, rows: usize, first: u64, out: []u32) !void {
        const rule: ?sampler.Rule = if (e.cur.sampling) |s| sampler.rule(s) else null;
        try e.smp.masked(e.stream, e.exchange(), e.spans, &.{.{ .rank = e.share.rank, .logits = logits }}, rows, first, rule, e.m.glue.top1, out);
        if (e.tp_group) |*group| try group.check();
    }

    pub const promptHead = @import("cuda_structured.zig").promptHead;
    pub const drawWhole = @import("cuda_structured.zig").drawWhole;
    pub const redrawFirst = @import("cuda_structured.zig").redrawFirst;
    /// decode.sample_rows' greedy pick: per-rank top-1 (tf_glm_top1), all-gathered, largest value then lowest id.
    fn greedy(e: *Engine, logits: u64) !u32 {
        var token: [1]u32 = undefined;
        try e.drawGreedy(logits, 1, &token);
        return token[0];
    }

    fn drawGreedy(e: *Engine, logits: u64, rows: usize, out: []u32) !void {
        errdefer e.pair_failure(error.PairedGreedyDrawFailed);
        try e.greedy_rows.draw(e.stream, .{ .rank = e.share.rank, .vocab = e.share.vocab, .top1 = e.m.glue.top1, .exchange = e.exchange(), .group = e.collectiveGroup() }, logits, rows, out);
    }

    pub fn checkGathers(e: *const Engine) !void {
        if (e.tp_group) |*group| try group.check();
    }

    pub fn copy(e: *const Engine, dst: u64, src: u64, bytes: usize) !void {
        try e.d.check(e.d.api.cuMemcpyDtoDAsync_v2(dst, src, bytes, e.stream.handle), "cuMemcpyDtoDAsync");
    }
};

fn projectionFormat(p: @import("cuda_proj.zig").Proj) []const u8 {
    return switch (p) {
        .bf16 => "bf16",
        .fp8g => "fp8g",
        .mxfp8 => "mxfp8",
        .source => "source",
        .q => |q| @tagName(q),
    };
}

test "prompt snapshot estimates follow the requested mark and MTP setting" {
    var seq: Seq = undefined;
    seq.st.layout = try @import("cuda_layout.zig").Layout.init(.{ 4, 512, 512, 0, 0, 0, 0, 0, 0 }, 8192, fw.prefill_rows, 2, .fp8, 512, 128);
    seq.st.pos = 0;
    seq.st.mtp_len = 0;
    var e: Engine = undefined;
    e.tp_group = null;
    e.dflash = null;
    e.drafter = null;
    e.canonical_owner = false;
    e.cur = &seq;
    e.mtp = true;
    const at = 2048;
    const plan = try seq.st.layout.snapshot(.{ .main = at, .mtp = at });
    try std.testing.expectEqual(plan.byteCount(), e.snapshotBytes(at));
    try std.testing.expect(e.snapshotBytes(4096) > e.snapshotBytes(at));
    try std.testing.expect(e.snapshotBytes(at) < seq.st.layout.total_bytes);
    e.mtp = false;
    const no_mtp = try seq.st.layout.snapshot(.{ .main = at, .mtp = 0 });
    try std.testing.expectEqual(no_mtp.byteCount(), e.snapshotBytes(at));
    try std.testing.expectEqual(std.math.maxInt(u64), e.snapshotBytes(8193));
}

test "snapshot capture refuses uncommitted windows and mismatched MTP marks before allocation" {
    var seq: Seq = undefined;
    seq.st.pos = 2048;
    seq.st.pending = 1;
    seq.st.mtp_len = 2048;
    seq.in_head = 0;
    var e: Engine = undefined;
    e.tp_group = null;
    e.dflash = null;
    e.drafter = null;
    e.canonical_owner = false;
    e.snaps = .empty;
    e.cur = &seq;
    e.mtp = true;
    try std.testing.expectError(error.NotAtMark, e.keepAs(1));
    seq.st.pending = 0;
    seq.in_head = 1;
    try std.testing.expectError(error.NotAtMark, e.keepAs(1));
    seq.in_head = 0;
    seq.st.mtp_len = 2047;
    try std.testing.expectError(error.NotAtMark, e.keepAs(1));
    e.mtp = false;
    try std.testing.expectError(error.NotAtMark, e.keepAs(1));
}

test "prompt resume refuses a saved position that differs from its requested mark" {
    var seq: Seq = undefined;
    seq.st.capacity = 8192;
    seq.st.layout = try @import("cuda_layout.zig").Layout.init(.{ 4, 0, 0, 0, 0, 0, 0, 0, 0 }, 8192, fw.prefill_rows, 0, .bf16, 512, 128);
    var e: Engine = undefined;
    e.tp_group = null;
    e.dflash = null;
    e.cur = &seq;
    e.snaps = .empty;
    defer e.snaps.deinit(std.testing.allocator);
    var snap: Snap = undefined;
    snap.pos = 4096;
    try e.snaps.put(std.testing.allocator, 1, snap);
    try std.testing.expectError(error.BadResume, e.prefillStart(6144, .{ .id = 1, .at = 2048 }));
    try std.testing.expectError(error.NoKeptState, e.prefillStart(6144, .{ .id = 2, .at = 2048 }));
}

test "engine snapshot save and restore preserve visible state through reused storage without a GPU" {
    const Mock = struct {
        var bytes: ?[]align(256) u8 = null;
        fn alloc(ptr: *u64, n: usize) callconv(.c) cuda.abi.Result {
            std.debug.assert(bytes == null);
            bytes = std.testing.allocator.alignedAlloc(u8, .fromByteUnits(256), n) catch return 2;
            ptr.* = @intFromPtr(bytes.?.ptr);
            return 0;
        }
        fn free(ptr: u64) callconv(.c) cuda.abi.Result {
            std.debug.assert(ptr == @intFromPtr(bytes.?.ptr));
            std.testing.allocator.free(bytes.?);
            bytes = null;
            return 0;
        }
        fn copy(dst: u64, src: u64, n: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
            const to: [*]u8 = @ptrFromInt(dst);
            const from: [*]const u8 = @ptrFromInt(src);
            @memcpy(to[0..n], from[0..n]);
            return 0;
        }
        fn fill(dst: u64, value: c_uint, n: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
            const to: [*]u32 = @ptrFromInt(dst);
            @memset(to[0..n], value);
            return 0;
        }
        fn fill8(dst: u64, value: u8, n: usize) callconv(.c) cuda.abi.Result {
            const to: [*]u8 = @ptrFromInt(dst);
            @memset(to[0..n], value);
            return 0;
        }
        fn fill8Async(dst: u64, value: u8, n: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
            return fill8(dst, value, n);
        }
        fn sync(_: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
            return 0;
        }
    };
    var d: cuda.Driver = .{ .lib = undefined, .api = undefined };
    d.api.cuMemAlloc_v2 = Mock.alloc;
    d.api.cuMemFree_v2 = Mock.free;
    d.api.cuMemcpyDtoDAsync_v2 = Mock.copy;
    d.api.cuMemsetD32Async = Mock.fill;
    d.api.cuMemsetD8_v2 = Mock.fill8;
    d.api.cuMemsetD8Async = Mock.fill8Async;
    d.api.cuStreamSynchronize = Mock.sync;
    const stream: cuda.Stream = .{ .d = &d, .handle = null };
    for ([_]@import("cuda_cache.zig").Format{ .bf16, .fp8 }) |format| for ([_]bool{ false, true }) |mtp| {
        const layout = try @import("cuda_layout.zig").Layout.init(.{ 4, 256, 256, 256, 256, 256, 256, 256, 0 }, 4096, fw.prefill_rows, 2, format, 512, 128);
        const live = try std.testing.allocator.alignedAlloc(u8, .fromByteUnits(256), layout.total_bytes);
        defer std.testing.allocator.free(live);
        for (live, 0..) |*v, i| v.* = @truncate(i *% 37);
        var parity = [_]u1{ 1, 0 };
        var seq: Seq = undefined;
        seq.in_head = 0;
        seq.st.arena = .{ .buf = .{ .d = &d, .ptr = @intFromPtr(live.ptr), .len = live.len, .borrowed = true } };
        seq.st.layout = layout;
        seq.st.capacity = layout.capacity;
        var caches: [2]@import("cuda_dsa_block.zig").Caches = undefined;
        for (&caches, 0..) |*c, layer| {
            const offsets = layout.cacheOffsets(layer);
            c.* = .{ .lc = .{ .ptr = seq.st.arena.buf.ptr + offsets[0], .format = format }, .index = .{ .keys = seq.st.arena.buf.ptr + offsets[1], .gates = seq.st.arena.buf.ptr + offsets[2], .rows = layout.ring_rows }, .pk = .{ .ptr = seq.st.arena.buf.ptr + offsets[3], .format = format }, .pk_rows = 1026 };
        }
        seq.st.caches = &caches;
        const mo = layout.cacheOffsets(2);
        seq.st.mtp_caches = .{ .lc = .{ .ptr = seq.st.arena.buf.ptr + mo[0], .format = format }, .index = .{ .keys = seq.st.arena.buf.ptr + mo[1], .gates = seq.st.arena.buf.ptr + mo[2], .rows = layout.ring_rows }, .pk = .{ .ptr = seq.st.arena.buf.ptr + mo[3], .format = format }, .pk_rows = 1026 };
        seq.st.cur = &parity;
        seq.st.pos_dev = seq.st.arena.buf.ptr + layout.fixed[0];
        seq.st.mtp_pos_dev = seq.st.arena.buf.ptr + layout.mtp_pos;
        seq.st.pending = 0;
        try seq.st.setPos(stream, 2115);
        try seq.st.setMtpLen(stream, if (mtp) 2115 else 0);
        var e: Engine = undefined;
        e.tp_group = null;
        e.dflash = null;
        e.drafter = null;
        e.canonical_owner = false;
        e.d = &d;
        e.gpa = std.testing.allocator;
        e.stream = stream;
        e.cur = &seq;
        e.mtp = mtp;
        e.snaps = .empty;
        defer e.snaps.deinit(e.gpa);
        try e.keepAs(1);
        defer e.freeSnap(e.snaps.getPtr(1).?);
        const saved = e.snaps.get(1).?;
        try std.testing.expectEqual(e.snapshotBytes(2115), saved.buf.len);
        try std.testing.expect(saved.buf.len < live.len);
        @memset(live, 0xa5);
        parity = .{ 0, 1 };
        seq.st.pending = 7;
        seq.in_head = 2;
        try e.restoreState(1);
        try std.testing.expectEqual(@as(usize, 2115), seq.st.pos);
        try std.testing.expectEqual(@as(usize, if (mtp) 2115 else 0), seq.st.mtp_len);
        try std.testing.expectEqualSlices(u1, &.{ 1, 0 }, &parity);
        try std.testing.expectEqual(@as(usize, 0), seq.st.pending);
        try std.testing.expectEqual(@as(usize, 0), seq.in_head);
        const plan = try layout.snapshot(.{ .main = seq.st.pos, .mtp = seq.st.mtp_len });
        const kept: [*]const u8 = @ptrFromInt(saved.buf.ptr);
        var it = plan.ranges();
        while (it.next()) |range| try std.testing.expectEqualSlices(u8, kept[range.saved..][0..range.bytes], live[layout.address(range)..][0..range.bytes]);
        try std.testing.expectEqual(@as(u32, if (mtp) 2115 else 0), std.mem.bytesToValue(u32, live[layout.mtp_pos..][0..4]));
        try std.testing.expectEqual(@as(u8, 0xa5), live[layout.cacheOffsets(0)[0] + 2115 * layout.latent_stride]);
    };
}

fn canonicalRingSlot(base: u64, layer: usize, kv: usize, pos: i64) u64 {
    const slot: u64 = @intCast(@mod(pos, dflash.WIN));
    const plane = @as(u64, @intCast(dflash.WIN)) * dflash.KVW * 2;
    return base + (layer * 2 + kv) * plane + slot * dflash.KVW * 2;
}

fn canonicalRingMark(pos: i64) u8 {
    return 0x40 | @as(u8, @intCast(@mod(pos, 32)));
}

fn canonicalRingFill(base: u64, from: i64, pos: i64, unused: u8) void {
    const M = @import("cuda_test_driver.zig").Memory;
    @memset(M.slice(base, dflash.ring_bytes), unused);
    var p = from;
    while (p < pos) : (p += 1) {
        const mark = canonicalRingMark(p);
        for (0..dflash.LAYERS) |layer| {
            for (0..2) |kv| {
                @memset(M.slice(canonicalRingSlot(base, layer, kv, p), dflash.KVW * 2), mark);
            }
        }
    }
}

fn canonicalRingExpect(base: u64, from: i64, pos: i64, unused: u8) !void {
    const M = @import("cuda_test_driver.zig").Memory;
    var occupied: [2048]bool = @splat(false);
    var p = from;
    while (p < pos) : (p += 1) {
        occupied[@intCast(@mod(p, dflash.WIN))] = true;
        const mark = canonicalRingMark(p);
        for (0..dflash.LAYERS) |layer| {
            for (0..2) |kv| {
                try std.testing.expect(std.mem.allEqual(u8, M.slice(canonicalRingSlot(base, layer, kv, p), dflash.KVW * 2), mark));
            }
        }
    }
    for (0..@as(usize, @intCast(dflash.WIN))) |slot| {
        if (occupied[slot]) continue;
        for (0..dflash.LAYERS) |layer| {
            for (0..2) |kv| {
                const ptr = canonicalRingSlot(base, layer, kv, @intCast(slot));
                try std.testing.expect(std.mem.allEqual(u8, M.slice(ptr, dflash.KVW * 2), unused));
            }
        }
    }
}

test "canonical owner snapshot export import restore preserves initialized drafter context" {
    const M = @import("cuda_test_driver.zig").Memory;
    var d = M.driver();
    const stream: cuda.Stream = .{ .d = &d, .handle = null };
    const layout = try @import("cuda_layout.zig").Layout.init(.{ 4, 256, 256, 256, 256, 256, 256, 256, 0 }, 4096, fw.prefill_rows, 2, .bf16, 512, 128);
    const live = try std.testing.allocator.alignedAlloc(u8, .fromByteUnits(256), layout.total_bytes);
    defer std.testing.allocator.free(live);
    for (live, 0..) |*v, i| v.* = @truncate(i *% 37);
    var parity = [_]u1{ 1, 0 };
    var seq: Seq = undefined;
    seq.in_head = 0;
    seq.st.arena = .{ .buf = .{ .d = &d, .ptr = @intFromPtr(live.ptr), .len = live.len, .borrowed = true } };
    seq.st.layout = layout;
    seq.st.capacity = layout.capacity;
    var caches: [2]@import("cuda_dsa_block.zig").Caches = undefined;
    for (&caches, 0..) |*c, layer| {
        const offsets = layout.cacheOffsets(layer);
        c.* = .{ .lc = .{ .ptr = seq.st.arena.buf.ptr + offsets[0], .format = .bf16 }, .index = .{ .keys = seq.st.arena.buf.ptr + offsets[1], .gates = seq.st.arena.buf.ptr + offsets[2], .rows = layout.ring_rows }, .pk = .{ .ptr = seq.st.arena.buf.ptr + offsets[3], .format = .bf16 }, .pk_rows = 1026 };
    }
    seq.st.caches = &caches;
    const mo = layout.cacheOffsets(2);
    seq.st.mtp_caches = .{ .lc = .{ .ptr = seq.st.arena.buf.ptr + mo[0], .format = .bf16 }, .index = .{ .keys = seq.st.arena.buf.ptr + mo[1], .gates = seq.st.arena.buf.ptr + mo[2], .rows = layout.ring_rows }, .pk = .{ .ptr = seq.st.arena.buf.ptr + mo[3], .format = .bf16 }, .pk_rows = 1026 };
    seq.st.cur = &parity;
    seq.st.pos_dev = seq.st.arena.buf.ptr + layout.fixed[0];
    seq.st.mtp_pos_dev = seq.st.arena.buf.ptr + layout.mtp_pos;
    seq.st.pending = 0;
    try seq.st.setPos(stream, 2048);
    try seq.st.setMtpLen(stream, 0);
    var ring0 = try cuda.DeviceBuffer.alloc(&d, dflash.ring_bytes);
    defer ring0.free();
    var ring1 = try cuda.DeviceBuffer.alloc(&d, dflash.ring_bytes);
    defer ring1.free();
    const from: i64 = 1920;
    canonicalRingFill(ring0.ptr, from, 2048, 0xa5);
    @memset(M.slice(ring1.ptr, ring1.len), 0x3c);
    var dr: dflash.Drafter = .{ .gpa = std.testing.allocator, .d = &d, .module = undefined, .k = undefined, .weights = undefined, .split = false, .ff = 0, .rank = 0, .world = 1 };
    try dr.slots.append(std.testing.allocator, .{ .ring = ring0, .from = from, .upto = 2048 });
    try dr.slots.append(std.testing.allocator, .{ .ring = ring1, .from = 7, .v_rows = 3 });
    defer dr.slots.deinit(std.testing.allocator);
    var e: Engine = undefined;
    e.tp_group = null;
    e.dflash = null;
    e.drafter = &dr;
    e.canonical_owner = true;
    e.canonical = .{ .checkpoint = ranks.hashBytes("canonical-snap"), .bits = 4 };
    e.d = &d;
    e.gpa = std.testing.allocator;
    e.stream = stream;
    e.cur = &seq;
    e.slot = 0;
    e.mtp = false;
    e.replicas = .{};
    e.pair_failure = owner_kv.failPair;
    e.settings = .{ .kv = .bf16 };
    e.share.rank = 0;
    e.share.world = 1;
    e.seqs = .empty;
    try e.seqs.append(e.gpa, &seq);
    defer e.seqs.deinit(e.gpa);
    e.snaps = .empty;
    defer e.snaps.deinit(e.gpa);
    try e.keepAs(1);
    defer e.freeSnap(e.snaps.getPtr(1).?);
    const saved = e.snaps.get(1).?;
    try std.testing.expectEqual(e.snapshotBytes(2048), saved.buf.len);
    var host = try e.exportState(1, saved.buf.len);
    defer e.gpa.free(host.bytes);
    try std.testing.expectEqual(saved.buf.len, host.bytes.len);
    const plan = try layout.snapshot(.{ .main = 2048, .mtp = 0 });
    const draft = host.bytes[std.mem.alignForward(usize, plan.byteCount(), 256)..];
    try std.testing.expect(draft.len >= dflash.ring_bytes);
    try std.testing.expect(std.mem.indexOfScalar(u8, draft, 0xaa) == null);
    try std.testing.expect(std.mem.indexOfScalar(u8, draft, 0xa5) == null);
    const Pump = struct {
        fn copy(ptr: *anyopaque, buffer: *cuda.DeviceBuffer) !void {
            const source: *[]u8 = @ptrCast(@alignCast(ptr));
            @memcpy(M.slice(buffer.ptr, buffer.len), source.*);
        }
    };
    var other = e.canonical;
    other.bits = 16;
    const previous = e.canonical;
    e.canonical = other;
    try std.testing.expectError(error.SpillShape, e.importState(2, 2048, host.meta[0..host.meta_len], host.bytes.len, @ptrCast(&host.bytes), Pump.copy));
    try std.testing.expect(!e.holds(2));
    e.canonical = previous;
    {
        var shape = e.spillShape();
        var epoch = std.crypto.hash.sha2.Sha256.init(.{});
        epoch.update("glm-dflash2-saved-v1\x00");
        epoch.update(&e.canonical.checkpoint);
        var bits: [4]u8 = undefined;
        std.mem.writeInt(u32, &bits, e.canonical.bits, .little);
        epoch.update(&bits);
        shape.draft_identity = epoch.finalResult();
        var meta: [spill_payload.max_meta]u8 = undefined;
        const encoded = try spill_payload.encode(shape, 2048, 0, null, saved.cur, &meta);
        defer if (e.holds(3)) e.discardState(3);
        try std.testing.expectError(error.SpillShape, e.importState(3, 2048, encoded, host.bytes.len, @ptrCast(&host.bytes), Pump.copy));
        try std.testing.expect(!e.holds(3));
    }
    try e.importState(2, 2048, host.meta[0..host.meta_len], host.bytes.len, @ptrCast(&host.bytes), Pump.copy);
    defer e.discardState(2);
    @memset(M.slice(ring0.ptr, ring0.len), 0x11);
    dr.slots.items[0].from = 99;
    dr.slots.items[0].v_rows = 4;
    try e.restoreAt(2, 2048);
    try std.testing.expectEqual(from, dr.slots.items[0].from);
    try std.testing.expectEqual(@as(i64, 2048), dr.slots.items[0].upto);
    try std.testing.expectEqual(@as(usize, 0), dr.slots.items[0].v_rows);
    try canonicalRingExpect(ring0.ptr, from, 2048, 0);
    try std.testing.expect(std.mem.allEqual(u8, M.slice(ring1.ptr, ring1.len), 0x3c));
    try std.testing.expectEqual(@as(i64, 7), dr.slots.items[1].from);
    try std.testing.expectEqual(@as(usize, 3), dr.slots.items[1].v_rows);
    const Failed = struct {
        var err: ?anyerror = null;
        fn fail(cause: anyerror) void {
            err = cause;
        }
    };
    Failed.err = null;
    e.pair_failure = Failed.fail;
    {
        const draft_at = std.mem.alignForward(usize, plan.byteCount(), 256);
        std.mem.writeInt(u32, M.slice(e.snaps.get(2).?.buf.ptr + draft_at, 12)[8..12], dflash.snapshot_version_unmaterialized, .little);
        dr.slots.items[0].from = 77;
        dr.slots.items[0].upto = 77;
        try std.testing.expectError(error.BadDraftSnapshot, e.restoreAt(2, 2048));
        try std.testing.expectEqual(@as(i64, 77), dr.slots.items[0].from);
        try std.testing.expectEqual(@as(i64, 77), dr.slots.items[0].upto);
        try std.testing.expectEqual(error.PairedRestoreFailed, Failed.err.?);
    }
    Failed.err = null;
    M.slice(e.snaps.get(2).?.buf.ptr + std.mem.alignForward(usize, plan.byteCount(), 256), 8)[0] = 'X';
    try std.testing.expectError(error.BadDraftSnapshot, e.restoreAt(2, 2048));
    try std.testing.expectEqual(error.PairedRestoreFailed, Failed.err.?);
    seq.st.conv = seq.st.arena.buf.ptr;
    seq.st.rec = seq.st.arena.buf.ptr;
    seq.st.kda_layers = 1;
    seq.st.conv_c = 1;
    seq.st.rec_layer = 8;
    e.pool = null;
    e.images = null;
    try e.begin(null);
    try std.testing.expectEqual(@as(i64, 0), dr.slots.items[0].from);
    try std.testing.expectEqual(@as(i64, 0), dr.slots.items[0].upto);
    try std.testing.expectEqual(@as(usize, 0), dr.slots.items[0].v_rows);
    try std.testing.expect(std.mem.allEqual(u8, M.slice(ring0.ptr, ring0.len), 0));
}

test "canonical keepAs at an early mark refuses until prompt context is materialized" {
    const M = @import("cuda_test_driver.zig").Memory;
    var d = M.driver();
    const stream: cuda.Stream = .{ .d = &d, .handle = null };
    const layout = try @import("cuda_layout.zig").Layout.init(.{ 4, 256, 256, 256, 256, 256, 256, 256, 0 }, 4096, fw.prefill_rows, 2, .bf16, 512, 128);
    const live = try std.testing.allocator.alignedAlloc(u8, .fromByteUnits(256), layout.total_bytes);
    defer std.testing.allocator.free(live);
    var parity = [_]u1{ 1, 0 };
    var seq: Seq = undefined;
    seq.in_head = 0;
    seq.st.arena = .{ .buf = .{ .d = &d, .ptr = @intFromPtr(live.ptr), .len = live.len, .borrowed = true } };
    seq.st.layout = layout;
    seq.st.capacity = layout.capacity;
    var caches: [2]@import("cuda_dsa_block.zig").Caches = undefined;
    for (&caches, 0..) |*c, layer| {
        const offsets = layout.cacheOffsets(layer);
        c.* = .{ .lc = .{ .ptr = seq.st.arena.buf.ptr + offsets[0], .format = .bf16 }, .index = .{ .keys = seq.st.arena.buf.ptr + offsets[1], .gates = seq.st.arena.buf.ptr + offsets[2], .rows = layout.ring_rows }, .pk = .{ .ptr = seq.st.arena.buf.ptr + offsets[3], .format = .bf16 }, .pk_rows = 1026 };
    }
    seq.st.caches = &caches;
    const mo = layout.cacheOffsets(2);
    seq.st.mtp_caches = .{ .lc = .{ .ptr = seq.st.arena.buf.ptr + mo[0], .format = .bf16 }, .index = .{ .keys = seq.st.arena.buf.ptr + mo[1], .gates = seq.st.arena.buf.ptr + mo[2], .rows = layout.ring_rows }, .pk = .{ .ptr = seq.st.arena.buf.ptr + mo[3], .format = .bf16 }, .pk_rows = 1026 };
    seq.st.cur = &parity;
    seq.st.pos_dev = seq.st.arena.buf.ptr + layout.fixed[0];
    seq.st.mtp_pos_dev = seq.st.arena.buf.ptr + layout.mtp_pos;
    seq.st.pending = 0;
    try seq.st.setPos(stream, 2048);
    try seq.st.setMtpLen(stream, 0);
    var dr: dflash.Drafter = .{ .gpa = std.testing.allocator, .d = &d, .module = undefined, .k = undefined, .weights = undefined, .split = false, .ff = 0, .rank = 0, .world = 1 };
    defer dr.slots.deinit(std.testing.allocator);
    var e: Engine = undefined;
    e.tp_group = null;
    e.dflash = null;
    e.drafter = &dr;
    e.canonical_owner = true;
    e.d = &d;
    e.gpa = std.testing.allocator;
    e.stream = stream;
    e.cur = &seq;
    e.slot = 0;
    e.mtp = false;
    e.replicas = .{};
    e.pair_failure = owner_kv.failPair;
    e.settings = .{ .kv = .bf16 };
    e.share.rank = 0;
    e.share.world = 1;
    e.seqs = .empty;
    try e.seqs.append(e.gpa, &seq);
    defer e.seqs.deinit(e.gpa);
    e.snaps = .empty;
    defer e.snaps.deinit(e.gpa);
    try dr.restart(0, 0);
    defer if (dr.slots.items.len > 0) if (dr.slots.items[0].ring) |*r| r.free();
    try std.testing.expectError(error.NotAtMark, e.keepAs(1));
    dr.slots.items[0].upto = 2048;
    try e.keepAs(1);
    defer e.freeSnap(e.snaps.getPtr(1).?);
}

test "spill snapshot roundtrip checks metadata and discards failed imports without publishing an id" {
    const M = @import("cuda_test_driver.zig").Memory;
    var d = M.driver();
    for ([_]@import("cuda_cache.zig").Format{ .bf16, .fp8 }) |format| {
        var seq: Seq = undefined;
        seq.st.layout = try (try @import("cuda_layout.zig").Layout.init(.{ 256, 0, 0, 0, 0, 0, 0, 0, 0 }, 4096, prompt_rows, 1, format, 512, 128)).external();
        var cur = [_]u1{1};
        seq.st.cur = &cur;
        var e: Engine = undefined;
        e.tp_group = null;
        e.dflash = null;
        e.drafter = null;
        e.canonical_owner = false;
        e.gpa = std.testing.allocator;
        e.d = &d;
        e.stream = .{ .d = &d, .handle = null };
        e.settings = .{ .kv = format, .kv_shard = true };
        e.share.rank = 1;
        e.share.world = 2;
        e.mtp = true;
        e.seqs = .empty;
        try e.seqs.append(e.gpa, &seq);
        defer e.seqs.deinit(e.gpa);
        e.snaps = .empty;
        defer e.snaps.deinit(e.gpa);
        const bytes = (try seq.st.layout.snapshot(.{ .main = 2048, .mtp = 2048 })).byteCount();
        const buf = try cuda.DeviceBuffer.alloc(&d, bytes);
        @memset(M.slice(buf.ptr, bytes), 0x5a);
        try e.snaps.put(e.gpa, 1, .{ .buf = buf, .pos = 2048, .mtp_len = 2048, .owner = 1, .cur = try e.gpa.dupe(u1, &cur) });
        defer e.discardState(1);
        try std.testing.expectError(error.SpillHostLimit, e.exportState(1, bytes - 1));
        var host = try e.exportState(1, bytes);
        defer e.gpa.free(host.bytes);
        try std.testing.expect(std.mem.allEqual(u8, host.bytes, 0x5a));
        const Pump = struct {
            fn copy(ptr: *anyopaque, buffer: *cuda.DeviceBuffer) !void {
                const source: *[]u8 = @ptrCast(@alignCast(ptr));
                @memcpy(M.slice(buffer.ptr, buffer.len), source.*);
            }
            fn fail(_: *anyopaque, buffer: *cuda.DeviceBuffer) !void {
                @memset(M.slice(buffer.ptr, buffer.len), 0x11);
                return error.PayloadHash;
            }
        };
        try std.testing.expectError(error.PayloadHash, e.importState(2, 2048, host.meta[0..host.meta_len], bytes, @ptrCast(&host.bytes), Pump.fail));
        try std.testing.expect(!e.holds(2));
        try e.importState(2, 2048, host.meta[0..host.meta_len], bytes, @ptrCast(&host.bytes), Pump.copy);
        defer e.discardState(2);
        const restored = e.snaps.get(2).?;
        try std.testing.expectEqual(@as(usize, 2048), restored.pos);
        try std.testing.expectEqual(@as(?u32, 1), restored.owner);
        try std.testing.expectEqualSlices(u1, &.{1}, restored.cur);
        try std.testing.expect(std.mem.allEqual(u8, M.slice(restored.buf.ptr, restored.buf.len), 0x5a));
        host.meta[0] = 0;
        try std.testing.expectError(error.SpillMetaVersion, e.importState(3, 2048, host.meta[0..host.meta_len], bytes, @ptrCast(&host.bytes), Pump.copy));
        try std.testing.expect(!e.holds(3));
    }
    try std.testing.expect(M.empty());
}

test "paired spill imports refuse missing corrupt and mismatched owner halves before publishing reuse" {
    const M = @import("cuda_test_driver.zig").Memory;
    const C = @import("cuda_test_collectives.zig").Collectives;
    const Fixture = struct {
        e: Engine = undefined,
        outcome: enum { valid, missing, corrupt } = .valid,
        owner: u32 = 0,
        forgotten: usize = 0,
        failed: ?anyerror = null,

        fn apply(ptr: *anyopaque, command: persistence.Command) anyerror!void {
            const f: *@This() = @ptrCast(@alignCast(ptr));
            if (command.operation == .forget) {
                f.forgotten += 1;
                return;
            }
            if (f.outcome == .missing) return error.NotStored;
            if (f.outcome == .corrupt) return error.PayloadHash;
            const buf = try cuda.DeviceBuffer.alloc(f.e.d, 1);
            const cur = try f.e.gpa.alloc(u1, 0);
            try f.e.snaps.put(f.e.gpa, command.id, .{ .buf = buf, .pos = command.at, .mtp_len = command.at, .owner = f.owner, .cur = cur });
        }
        fn unused(_: *anyopaque) anyerror!void {}
        fn close(_: *anyopaque) void {}
        fn forget(_: *anyopaque, _: u64, _: u32) void {}
        fn init(f: *@This(), d: *const cuda.Driver, lib: *const cuda.nccl.Library, endpoint: *C.End) !void {
            f.e.d = d;
            f.e.gpa = std.testing.allocator;
            f.e.lib = lib;
            f.e.comm = @ptrCast(endpoint);
            f.e.share.rank = endpoint.rank;
            f.e.share.world = 2;
            f.e.stream = .{ .d = d, .handle = null };
            f.e.shared_buf = try cuda.DeviceBuffer.alloc(d, 8);
            f.e.snaps = .empty;
            f.e.pair_failure = owner_kv.failPair;
            f.e.disk = .{ .ctx = f, .tier = f, .identity = @splat(0), .apply = apply, .request = apply, .flush = unused, .forget = forget, .close = close };
        }
        fn deinit(f: *@This()) void {
            f.e.discardState(7);
            f.e.snaps.deinit(f.e.gpa);
            f.e.shared_buf.free();
        }
        fn run(f: *@This()) void {
            f.e.persist(.{ .operation = .read, .id = 7, .key = 1, .at = 2048 }) catch |err| {
                f.failed = err;
            };
        }
    };
    var d = M.driver();
    var lib: cuda.nccl.Library = .{ .lib = undefined, .api = undefined };
    lib.api.ncclAllReduce = C.allReduce;
    for (0..4) |case| {
        var channel: C.Channel = .{};
        var endpoints = [_]C.End{ .{ .rank = 0, .channel = &channel }, .{ .rank = 1, .channel = &channel } };
        var a: Fixture = .{};
        var b: Fixture = .{};
        try a.init(&d, &lib, &endpoints[0]);
        defer a.deinit();
        try b.init(&d, &lib, &endpoints[1]);
        defer b.deinit();
        if (case == 0) b.outcome = .missing;
        if (case == 1) b.outcome = .corrupt;
        if (case == 2) b.owner = 1;
        const first = try std.Thread.spawn(.{}, Fixture.run, .{&a});
        const second = try std.Thread.spawn(.{}, Fixture.run, .{&b});
        first.join();
        second.join();
        if (case == 3) {
            try std.testing.expect(a.failed == null and b.failed == null);
            try std.testing.expect(a.e.holds(7) and b.e.holds(7));
        } else {
            const expected: anyerror = if (case == 0) error.NotStored else error.SpillRefused;
            try std.testing.expectEqual(expected, a.failed.?);
            try std.testing.expectEqual(expected, b.failed.?);
            try std.testing.expect(!a.e.holds(7) and !b.e.holds(7));
            try std.testing.expectEqual(@as(usize, if (case == 0) 0 else 1), a.forgotten);
            try std.testing.expectEqual(a.forgotten, b.forgotten);
        }
    }
    try std.testing.expect(M.empty());
}

test "owner-only snapshots carry main and MTP prefixes to a new owner and refuse an invalid peer before transfer" {
    const M = @import("cuda_test_driver.zig").Memory;
    const C = @import("cuda_test_collectives.zig").Collectives;
    const Layout = @import("cuda_layout.zig").Layout;
    const Fixture = struct {
        e: Engine,
        seq: Seq,
        caches: [1]@import("cuda_dsa_block.zig").Caches,
        parity: [1]u1,
        failed: ?anyerror = null,
        action: enum { restore, capture, reserve } = .restore,

        fn init(f: *@This(), d: *const cuda.Driver, lib: *const cuda.nccl.Library, endpoint: *C.End, old_owner: usize, format: @import("cuda_cache.zig").Format) !void {
            const layout = try (try Layout.init(.{ 4, 256, 256, 256, 256, 256, 256, 256, 0 }, 4096, fw.prefill_rows, 1, format, 512, 128)).external();
            f.parity = .{@intCast(endpoint.rank)};
            f.failed = null;
            f.seq = undefined;
            f.seq.in_head = 0;
            f.seq.vision = null;
            f.seq.st = undefined;
            f.seq.st.arena = .{ .buf = try cuda.DeviceBuffer.alloc(d, layout.total_bytes) };
            f.seq.st.layout = layout;
            f.seq.st.capacity = layout.capacity;
            f.seq.st.cur = &f.parity;
            f.seq.st.caches = &f.caches;
            for (0..2) |layer| {
                const offsets = layout.cacheOffsets(layer);
                const raw: @import("cuda_dsa_block.zig").Caches = .{ .lc = .{ .ptr = 0, .format = format }, .index = .{ .keys = f.seq.st.arena.buf.ptr + offsets[1], .gates = f.seq.st.arena.buf.ptr + offsets[2], .rows = layout.ring_rows }, .pk = .{ .ptr = 0, .format = format }, .pk_rows = 0 };
                if (layer == 0) f.caches[0] = raw else f.seq.st.mtp_caches = raw;
            }
            f.seq.st.pos_dev = f.seq.st.arena.buf.ptr;
            f.seq.st.mtp_pos_dev = f.seq.st.arena.buf.ptr + layout.mtp_pos;
            f.seq.st.pending = 0;
            f.e = undefined;
            f.e.tp_group = null;
            f.e.dflash = null;
            f.e.drafter = null;
            f.e.canonical_owner = false;
            f.e.d = d;
            f.e.lib = lib;
            f.e.comm = @ptrCast(endpoint);
            f.e.share.rank = endpoint.rank;
            f.e.share.world = 2;
            f.e.pair_failure = owner_kv.failPair;
            f.e.settings = .{ .kv_pool_tokens = 16384 };
            f.e.seqs = .empty;
            try f.e.seqs.append(std.testing.allocator, &f.seq);
            f.action = .restore;
            f.e.gpa = std.testing.allocator;
            f.e.cur = &f.seq;
            f.e.slot = 0;
            f.e.stream = .{ .d = d, .handle = null };
            f.e.mtp = true;
            f.e.snaps = .empty;
            f.e.shared_buf = try cuda.DeviceBuffer.alloc(d, 8);
            f.e.pool = try cache_pool.Pool.init(d, try cache_pool.Shape.init(16384, 2, 2, format), endpoint.rank, format);
            try f.e.reserveAt(.{ .base = old_owner * 8192, .rows = 4096 });
            const plan = try layout.snapshot(.{ .main = 2115, .mtp = 2115 });
            var it = plan.rangesFor(endpoint.rank == old_owner);
            while (it.next()) |r| @memset(M.slice(f.seq.st.address(r), r.bytes), @intCast(40 + endpoint.rank + @backingInt(r.plane)));
            try f.seq.st.setPos(f.e.stream, 2115);
            try f.seq.st.setMtpLen(f.e.stream, 2115);
            try f.e.keepAs(1);
            @memset(M.slice(f.e.pool.?.buf.ptr, f.e.pool.?.buf.len), 0xa5);
            @memset(M.slice(f.seq.st.arena.buf.ptr, f.seq.st.arena.buf.len), 0xa5);
            f.e.pool.?.regions.release(0);
            try f.e.reserveAt(.{ .base = (1 - old_owner) * 8192, .rows = 4096 });
        }
        fn deinit(f: *@This()) void {
            var it = f.e.snaps.valueIterator();
            while (it.next()) |s| f.e.freeSnap(s);
            f.e.snaps.deinit(f.e.gpa);
            f.e.seqs.deinit(f.e.gpa);
            f.e.pool.?.deinit();
            f.e.shared_buf.free();
            f.seq.st.arena.buf.free();
        }
        fn run(f: *@This()) void {
            f.failed = null;
            const result = switch (f.action) {
                .restore => f.e.restoreAt(1, 2115),
                .capture => f.e.capturePaired(2),
                .reserve => f.e.reservePaired(0, .{ .base = 16384, .rows = 4096 }),
            };
            result catch |err| {
                f.failed = err;
            };
        }
        fn pair(a: *@This(), b: *@This()) !void {
            const first = try std.Thread.spawn(.{}, run, .{a});
            const second = try std.Thread.spawn(.{}, run, .{b});
            first.join();
            second.join();
        }
    };
    var d = M.driver();
    var lib: cuda.nccl.Library = .{ .lib = undefined, .api = undefined };
    lib.api.ncclBroadcast = C.broadcast;
    lib.api.ncclAllReduce = C.allReduce;
    for ([_]@import("cuda_cache.zig").Format{ .bf16, .fp8 }) |format| for (0..2) |old_owner| {
        var channel: C.Channel = .{};
        var endpoints = [_]C.End{ .{ .rank = 0, .channel = &channel }, .{ .rank = 1, .channel = &channel } };
        var fixtures: [2]Fixture = undefined;
        for (&fixtures, &endpoints) |*f, *endpoint| try f.init(&d, &lib, endpoint, old_owner, format);
        defer for (&fixtures) |*f| f.deinit();
        try std.testing.expect(fixtures[old_owner].e.snaps.get(1).?.buf.len > fixtures[1 - old_owner].e.snaps.get(1).?.buf.len);
        try std.testing.expectEqual(@as(usize, 0), channel.epoch.load(.acquire));
        try Fixture.pair(&fixtures[0], &fixtures[1]);
        for (&fixtures, 0..) |*f, rank| {
            try std.testing.expectEqual(@as(?anyerror, null), f.failed);
            try std.testing.expectEqual(@as(usize, 2115), f.seq.st.pos);
            try std.testing.expectEqual(@as(usize, 2115), f.seq.st.mtp_len);
            try std.testing.expectEqual(@as(u1, @intCast(rank)), f.parity[0]);
            const plan = try f.seq.st.layout.snapshot(.{ .main = 2115, .mtp = 2115 });
            var it = plan.rangesFor(rank == 1 - old_owner);
            while (it.next()) |r| {
                if (r.plane == .fixed) continue;
                const expected: u8 = @intCast(40 + (if (r.plane == .latent) old_owner else rank) + @backingInt(r.plane));
                for (M.slice(f.seq.st.address(r), r.bytes)) |value| try std.testing.expectEqual(expected, value);
            }
        }
        try std.testing.expectEqual(@as(usize, 4), channel.epoch.load(.acquire));
        fixtures[1].seq.st.capacity = 2048;
        try Fixture.pair(&fixtures[0], &fixtures[1]);
        try std.testing.expectEqual(error.KeptStateRefused, fixtures[0].failed.?);
        try std.testing.expectEqual(error.KeptStateRefused, fixtures[1].failed.?);
        try std.testing.expectEqual(@as(usize, 6), channel.epoch.load(.acquire));
        fixtures[1].seq.st.capacity = 4096;
        for (&fixtures) |*f| f.action = .capture;
        fixtures[1].seq.in_head = 1;
        try Fixture.pair(&fixtures[0], &fixtures[1]);
        for (&fixtures) |*f| {
            try std.testing.expectEqual(error.SnapshotRefused, f.failed.?);
            try std.testing.expect(!f.e.holds(2));
        }
        fixtures[1].seq.in_head = 0;
        try Fixture.pair(&fixtures[0], &fixtures[1]);
        for (&fixtures) |*f| {
            try std.testing.expectEqual(@as(?anyerror, null), f.failed);
            try std.testing.expect(f.e.holds(2));
            f.action = .reserve;
        }
        try Fixture.pair(&fixtures[0], &fixtures[1]);
        for (&fixtures) |*f| {
            try std.testing.expectEqual(error.KvReservationRefused, f.failed.?);
            try std.testing.expect(f.e.pool.?.regions.slots[0] == null);
        }
        try std.testing.expectEqual(@as(usize, 9), channel.epoch.load(.acquire));
    };
    try std.testing.expect(M.empty());
}

test "pooled engine refuses out-of-range slots before reading state or metadata" {
    var e: Engine = undefined;
    e.tp_group = null;
    e.dflash = null;
    e.drafter = null;
    e.canonical_owner = false;
    e.settings = .{ .kv_pool_tokens = 8192 };
    e.pool = @as(cache_pool.Pool, undefined);
    e.slot = cache_pool.extent.max_slots;
    try std.testing.expectError(error.BadKvRequest, e.bind(e.slot));
    try std.testing.expectError(error.BadKvRequest, e.reserveAt(.{ .base = 0, .rows = 2048 }));
    try std.testing.expectError(error.BadKvRequest, e.begin(null));
}

test "MTP hidden rows survive other sequences reusing prompt and verify scratch" {
    const M = @import("cuda_test_driver.zig").Memory;
    var d = M.driver();
    var e: Engine = undefined;
    e.tp_group = null;
    e.dflash = null;
    e.d = &d;
    e.stream = .{ .d = &d, .handle = null };
    e.c.hidden = 2;
    var a: Seq = .{ .st = undefined, .retained = try cuda.DeviceBuffer.alloc(&d, fw.window_rows * 4) };
    defer a.retained.?.free();
    var b: Seq = .{ .st = undefined, .retained = try cuda.DeviceBuffer.alloc(&d, fw.window_rows * 4) };
    defer b.retained.?.free();
    var scratch = [_]u16{ 0x3f80, 0x4000, 0x4040, 0x4080, 0x40a0, 0x40c0 };
    e.cur = &a;
    try e.retainRows(@intFromPtr(&scratch), 1);
    scratch[0..2].* = .{ 0x4100, 0x4110 };
    e.cur = &b;
    try e.retainRows(@intFromPtr(&scratch), 1);
    @memset(&scratch, 0);
    e.cur = &a;
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(&[_]u16{ 0x3f80, 0x4000 }), M.slice(e.draftRows(true), 4));
    e.cur = &b;
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(&[_]u16{ 0x4100, 0x4110 }), M.slice(e.draftRows(true), 4));

    scratch = .{ 0x3f80, 0x4000, 0x4040, 0x4080, 0x40a0, 0x40c0 };
    e.cur = &a;
    try e.retainRows(@intFromPtr(&scratch), 3);
    scratch = .{ 0x4180, 0x4190, 0x41a0, 0x41b0, 0x41c0, 0x41d0 };
    e.cur = &b;
    try e.retainRows(@intFromPtr(&scratch), 2);
    @memset(&scratch, 0);
    e.cur = &a;
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(&[_]u16{ 0x3f80, 0x4000, 0x4040, 0x4080, 0x40a0, 0x40c0 }), M.slice(e.draftRows(false), 12));
    e.cur = &b;
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(&[_]u16{ 0x4180, 0x4190, 0x41a0, 0x41b0 }), M.slice(e.draftRows(false), 8));
}

fn verifyLabel(rows: usize) []const u8 {
    return switch (rows) {
        1 => "verify1",
        2 => "verify2",
        4 => "verify4",
        8 => "verify8",
        else => "verify",
    };
}
