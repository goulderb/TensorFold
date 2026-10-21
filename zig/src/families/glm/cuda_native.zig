//! GLM-5.3-Flash NVFP4 native CUDA at TP over processes (TF_GLM_RANK, NCCL_COMM_ID): rank 0 serves, others follow.

const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const engine = @import("cuda_engine.zig");
const follow = @import("cuda_follow.zig");
const glm_lanes = @import("cuda_lanes.zig");
const ranks = @import("cuda_ranks.zig");
const fw = @import("cuda_forward.zig");
const config = @import("config.zig");
const shard = @import("cuda_shard.zig");
const vision_probe = @import("cuda_vision_probe.zig");
const vision_runtime = @import("cuda_vision_runtime.zig");
const vision_startup = @import("cuda_vision_startup.zig");
const images_mod = @import("cuda_images.zig");
const dflash = @import("dflash.zig");
const dflash_load = @import("dflash_load.zig");

pub const model_type = "glm5_next";
pub const formats: []const []const u8 = &.{ "modelopt-nvfp4", "modelopt-mixed", "exl3" };
// cc-defer: a fixed window; the Python family sizes it from free memory (capacity.admit), port that when serving needs
pub const default_context: i64 = 8192;
pub const max_segments: u32 = 1;
pub const prompt_rows: u32 = fw.prefill_rows;
/// Prompt kernels change arithmetic with a chunk's rows: states keep and resume only on chunk starts.
pub const snap_grid: u32 = prompt_rows;
pub fn snap_shared_wait() bool {
    return if (@import("sys").getenv("TF_GLM_SHARED_PREFIX_WAIT")) |v| std.mem.eql(u8, v, "1") else false;
}
/// `explain` words this family's open refusals too (where to set the rank and the id).
pub const explains_open = true;

/// The server's reading of this rank's room after its load (the budget streams are admitted by); null if unreadable.
pub const Room = engine.Room;
/// `slide` stays false: the server refuses --slide for a family without a learner (no `slide` here).
pub const PersistenceOpen = struct { ctx: *anyopaque, attach: *const fn (*anyopaque, *engine.Engine, *follow.Link) anyerror!engine.persistence.Port };
pub const Options = struct { context: usize, drafts: bool, segments: usize = 1, slide: bool = false, room: ?Room = null, persistence: ?PersistenceOpen = null, vision: ?images_mod.Limits = null, drafter: ?[]const u8 = null, drafter_bits: u8 = 4 };
/// The smallest room among the other ranks and whose it is.
pub const Peer = struct { bytes: u64, rank: usize };

pub const LoneRun = *const fn (ctx: *anyopaque, s: *lanes.Stream, hooks: *anyopaque, committed: *const fn (*anyopaque) void, yield: *const fn (*anyopaque) bool, resumed: bool) anyerror!bool;

pub const Loaded = struct {
    backend: lanes.backend.Backend,
    facts: lanes.Model,
    rows: u32,
    stream_bytes: usize,
    free_streams: u32 = 1,
    lane_limit: u32 = std.math.maxInt(u32),
    learn_bytes: usize = 0, // no learner: the streams keep all the room
    ctx: *anyopaque,
    deinit: *const fn (*anyopaque) void,
    lone: ?LoneRun = null,
    trim: ?lanes.Trim = null,
    proposers: ?lanes.proposer.Factory = null,
    target: *engine.Engine, // the prompt cache keeps this engine's states (snap_*)
    admit: ?struct { ctx: *anyopaque, fits: *const fn (*anyopaque, u32, u32) bool } = null,
    peer_room: ?Peer = null, // rank 0: the server admits streams and kept states by this when it is smaller
    vision_probe: ?struct {
        ctx: *anyopaque,
        rows: *const fn (*anyopaque, []const u8, usize) anyerror!u32,
    } = null,
};

/// This rank's place: TF_GLM_RANK of TF_GLM_WORLD (both, or neither for a refusal that names them).
pub const Place = struct { rank: usize, world: usize };

pub fn place() !Place {
    const r = @import("sys").getenv("TF_GLM_RANK") orelse return error.NoRank;
    const w = @import("sys").getenv("TF_GLM_WORLD") orelse return error.NoRank;
    const p: Place = .{ .rank = std.fmt.parseInt(usize, r, 10) catch return error.NoRank, .world = std.fmt.parseInt(usize, w, 10) catch return error.NoRank };
    return checked(p);
}

/// A place the family serves: a rank inside its world, and at most `ranks.max_world` ranks.
fn checked(p: Place) !Place {
    if (p.world < 1 or p.rank >= p.world) return error.NoRank;
    if (p.world > ranks.max_world) return error.TooManyRanks;
    return p;
}

/// Bytes this rank's weights take at most, for the server's admission: the checkpoint times its largest axis share.
pub fn weightShare(a: std.mem.Allocator, io: std.Io, dir: []const u8, total: u64) u64 {
    const p = place() catch return total;
    const path = std.fs.path.join(a, &.{ dir, "config.json" }) catch return total;
    const text = std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 24)) catch return total;
    const c = config.parse(a, text) catch return total;
    const f = (shard.Share.of(&c, p.rank, p.world) catch return total).part(&c);
    return @intCast(@as(u128, total) * f[0] / f[1]);
}

/// Slots for `context` plus a verify window, a multiple of 64 so the sparse pool keeps the AOT set.
pub fn capacity(context: usize) usize {
    return std.mem.alignForward(usize, context + fw.window_rows, 64);
}

const Owned = struct {
    gpa: std.mem.Allocator,
    lib: cuda.nccl.Library,
    comm: cuda.nccl.Comm,
    e: *engine.Engine,
    link: follow.Link,
    lanes: glm_lanes.Lanes,
    probe: ?*vision_probe.Probe = null,
    vision: ?*vision_runtime.Service = null,
    images: ?*images_mod.Images = null,
};

fn visionReady(ptr: *anyopaque, valid: bool) !bool {
    const e: *engine.Engine = @ptrCast(@alignCast(ptr));
    return e.pairedReady(valid);
}

pub fn open(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, _: ?[]const u8, o: Options) !Loaded {
    const p = try place();
    if (p.world > 1 and @import("sys").getenv("NCCL_COMM_ID") == null) return error.NoCommId;
    var problem: []const u8 = "";
    const set = ranks.Settings.readForOwner(&ranks.environment, &problem, o.drafter != null) catch |err| {
        std.log.err("{s}", .{problem});
        return err;
    };
    // the ranks' lines before NCCL (cuda/tp_watch.zig): a rank that fails, refuses or dies stops every rank
    if (p.world > 1) try cuda.tp_watch.start(io, p.rank, p.world);
    return openRank(gpa, io, ctx, dir, o, p, set) catch |err| {
        cuda.tp_watch.fail(@errorName(err)); // the other ranks stop with this rank's error
        return err;
    };
}

/// `open` once the ranks' lines are up: NCCL, the engine and the link, then a follower's loop or rank 0's lanes.
fn openRank(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, o: Options, p: Place, set: ranks.Settings) !Loaded {
    cuda.memory_receipt.emit(@intCast(p.rank), .load_begin, true); // the four phases tools/memory_receipts.py reads
    const own = try gpa.create(Owned);
    errdefer gpa.destroy(own);
    own.gpa = gpa;
    own.probe = null;
    own.vision = null;
    own.images = null;
    errdefer if (own.probe) |probe| probe.close();
    own.lib = try cuda.nccl.Library.open();
    errdefer own.lib.close();
    var id: cuda.nccl.UniqueId = undefined;
    try own.lib.check(own.lib.api.ncclGetUniqueId(&id), "ncclGetUniqueId"); // NCCL_COMM_ID: the same id on every rank
    own.comm = null;
    try own.lib.check(own.lib.api.ncclCommInitRank(&own.comm, @intCast(p.world), id, @intCast(p.rank)), "ncclCommInitRank");
    errdefer _ = own.lib.api.ncclCommDestroy(own.comm);
    const cap = capacity(o.context);
    own.e = try engine.Engine.init(gpa, io, ctx.d, ctx, dir, &own.lib, own.comm, p.rank, p.world, .{ .capacity = cap, .mtp = o.drafts, .settings = set, .vision = o.vision, .drafter = o.drafter, .drafter_bits = o.drafter_bits, .room = o.room, .draft_kernels = ranks.environment("TF_GLM_DFLASH_KERNELS") });
    errdefer own.e.deinit();
    errdefer if (own.images) |v| v.deinit();
    // The adapter may borrow Images' Tower and the engine stream: close it before either owner.
    errdefer if (own.vision) |v| v.close();
    if (o.vision) |lim| { // before the rooms: the tower's bytes count against rank 0's
        var problem: []const u8 = "";
        var image_problem: ?anyerror = null;
        const ql: cuda.qlinear.Linear = .{ .lane = &own.e.m.lane, .nvfp4_prompt = &own.e.m.prompt }; // the tower's MXFP8, NVFP4
        own.images = images_mod.Images.init(gpa, io, ctx.d, own.e.stream, dir, lim, fw.prefill_rows + 1, p.rank == 0, ql, &problem) catch |err| blk: {
            image_problem = err;
            break :blk null;
        };
        if (own.images == null and image_problem == null) {
            std.log.err("{s}", .{problem});
            image_problem = error.NoVisionTower;
        }
        // A failed rank-zero tower cannot return while followers enter later collectives.
        if (!try own.e.pairedReady(image_problem == null)) return image_problem orelse error.PeerVisionUnavailable;
    }
    own.e.images = own.images;
    if (o.drafter) |dd| { // before the rooms: its weights and scratch count against every rank's
        if (!o.drafts) return error.DrafterNeedsDrafts;
        // Warmup ran without this sink. Retire its captures before attaching a new resource owner.
        try own.e.invalidateCaptures();
        var draft_problem: ?anyerror = null;
        own.e.drafter = dflash_load.load(gpa, io, own.e, dd, p.rank, p.world, o.drafter_bits) catch |err| blk: {
            draft_problem = err;
            break :blk null;
        };
        if (!try own.e.pairedReady(draft_problem == null)) return draft_problem orelse error.PeerDrafterOpen;
        if (own.e.drafter) |dr| try dr.restart(0, 0);
    }
    own.link = try follow.Link.init(gpa, ctx.d, &own.lib, own.comm, own.e.stream, try linkCapacity(cap, o.vision));
    errdefer own.link.deinit();
    if (o.persistence) |persistence| {
        var problem_: ?anyerror = null;
        own.e.disk = persistence.attach(persistence.ctx, own.e, &own.link) catch |err| blk: {
            problem_ = err;
            break :blk null;
        };
        if (!try own.e.pairedReady(problem_ == null)) return problem_ orelse error.PeerSpillOpen;
        const hashes = try gpa.alloc([32]u8, p.world);
        defer gpa.free(hashes);
        const mine = own.e.disk.?.identity;
        try ranks.gather([32]u8, ctx.d, &own.lib, own.comm, own.e.stream, p.rank, mine, hashes);
        for (hashes) |hash| if (!std.mem.eql(u8, &mine, &hash)) return error.SpillIdentityMismatch;
    }
    if (set.vision) {
        var vision_problem: ?anyerror = null;
        if (p.rank == 0) {
            if (@import("sys").getenv("TF_GLM_VISION_CODEC")) |path| {
                own.probe = vision_probe.Probe.open(gpa, io, dir, path) catch |err| blk: {
                    vision_problem = err;
                    break :blk null;
                };
                if (own.probe) |probe| {
                    own.vision = vision_runtime.Service.open(gpa, io, ctx.d, ctx.handle, own.e.stream, dir, probe, .{ .lane = &own.e.m.lane, .nvfp4_prompt = &own.e.m.prompt }, own.images) catch |err| blk: {
                        vision_problem = err;
                        break :blk null;
                    };
                }
            } else vision_problem = error.MissingVisionCodecPath;
        }
        try vision_startup.paired(vision_problem, .{ .ctx = own.e, .ready = visionReady });
    }
    // Every rank checks TF_GLM_LONE_COSTS so a bad value stops both before they part; rank 0 alone plans with it.
    const lone_costs: ?glm_lanes.LoneCosts = if (@import("sys").getenv("TF_GLM_LONE_COSTS")) |raw| glm_lanes.LoneCosts.parse(raw) catch {
        std.log.err("rank {d}: TF_GLM_LONE_COSTS: ONE,ROW (positive ms, e.g. 38,7.9), not {s}", .{ p.rank, raw });
        return error.BadSetting;
    } else null;
    // Rank 0 admits by the smallest room of the ranks; its sizes are the largest (the first ranks take the remainder).
    const rooms = try gpa.alloc(u64, p.world);
    defer gpa.free(rooms);
    const mine: u64 = if (o.room) |r| r.read(r.ctx) orelse 0 else std.math.maxInt(u64);
    try ranks.gather(u64, ctx.d, &own.lib, own.comm, own.e.stream, p.rank, mine, rooms);
    var peer: ?Peer = null;
    for (rooms[1..], 1..) |r, q| if (peer == null or r < peer.?.bytes) {
        peer = .{ .bytes = r, .rank = q };
    };
    // every rank loaded and agreed (cuda_ranks) before the first step; the digest names what they agreed on
    if (p.world > 1) try cuda.tp_watch.ready(io, own.e.checkpoint_digest);
    cuda.memory_receipt.emit(@intCast(p.rank), .load_end, false);
    if (p.rank != 0) { // a follower never serves: it runs rank 0's steps, frees what it owns, then leaves the process
        follow.follow(own.e, &own.link) catch |err| {
            std.log.err("rank {d}: following rank 0 stopped ({s})", .{ p.rank, @errorName(err) });
            std.process.exit(1);
        };
        std.log.info("rank {d}: rank 0 ended; leaving", .{p.rank});
        cuda.memory_receipt.emit(@intCast(p.rank), .cleanup_begin, false);
        release(own);
        cuda.memory_receipt.emit(@intCast(p.rank), .cleanup_end, false);
        std.process.exit(0);
    }
    // Rank 0 alone reads these: TF_GLM_PROMPT_STEPS=0 passes prompts whole, TF_GLM_SHARED=0 runs windows alone.
    const off = struct {
        fn get(name: [*:0]const u8) bool {
            return if (@import("sys").getenv(name)) |v| std.mem.eql(u8, v, "0") else false;
        }
    }.get;
    own.lanes = .{ .x = .{ .e = own.e, .link = &own.link }, .vision = own.vision, .whole = off("TF_GLM_PROMPT_STEPS"), .alone = off("TF_GLM_SHARED"), .lone_costs = lone_costs };
    own.lanes.batch_prefill = if (@import("sys").getenv("TF_GLM_MULTI_PREFILL")) |v| std.mem.eql(u8, v, "1") else false;
    if (lone_costs != null) std.log.info("GLM lone-stream round costs: {s} ms (one row, each further row)", .{@import("sys").getenv("TF_GLM_LONE_COSTS").?});
    return .{
        .backend = own.lanes.backend(),
        .facts = own.lanes.facts(),
        .rows = if (set.draft_profile.enabled) glm_lanes.max_rows else 1 + engine.max_drafts,
        .trim = if (set.draft_profile.enabled) own.e.settings.draft_profile.rowPolicy() else null,
        .proposers = if (set.draft_profile.enabled) own.e.settings.draft_profile.proposers() else null,
        .stream_bytes = extraStreamBytes(@intCast(own.e.seqBytes()), o.drafter != null, if (own.images) |v| v.streamBytes() else 0), // each stream past the first makes its own sequence on every rank; first canonical ring is allocated at load
        .lane_limit = laneLimit(own.e.drafter != null, own.e.dflash != null, own.e.pool != null),
        .free_streams = 1, // the first is the sequence the engine made
        .ctx = own,
        .deinit = close,
        .target = own.e,
        .peer_room = peer,
        .vision_probe = if (own.probe) |probe| .{ .ctx = probe, .rows = vision_probe.Probe.rows } else null,
        .admit = if (own.e.pool != null) .{ .ctx = own.e, .fits = fits } else null,
    };
}

fn fits(ptr: *anyopaque, prompt: u32, output: u32) bool {
    const e: *const engine.Engine = @ptrCast(@alignCast(ptr));
    return e.fits(prompt, output);
}

/// Canonical DFlash owns 16 slots; leftover env DFlash uses the shared window; a KV pool is tighter still.
pub fn laneLimit(canonical: bool, legacy: bool, pool: bool) u32 {
    var limit: u32 = std.math.maxInt(u32);
    if (canonical) limit = @min(limit, @as(u32, @intCast(dflash.MAX_S)));
    if (legacy) limit = @min(limit, @as(u32, @intCast(engine.max_shared)));
    if (pool) limit = @min(limit, @as(u32, @intCast(@import("cuda_kv_extent.zig").max_slots)));
    return limit;
}

pub fn extraStreamBytes(seq_bytes: usize, canonical: bool, images: usize) usize {
    return seq_bytes + (if (canonical) dflash.ring_bytes else 0) + images;
}

pub fn unchargedFirstRing(canonical: bool, allocated: bool) usize {
    if (!canonical or allocated) return 0;
    return dflash.ring_bytes;
}

test "canonical first ring is reserved before extra-stream admission" {
    const Memory = @import("cuda_test_driver.zig").Memory;
    var d = Memory.driver();
    var dr: dflash.Drafter = .{ .gpa = std.testing.allocator, .d = &d, .module = undefined, .k = undefined, .weights = undefined, .split = false, .ff = 0, .rank = 0, .world = 1 };
    defer dr.slots.deinit(std.testing.allocator);
    try std.testing.expectEqual(dflash.ring_bytes, unchargedFirstRing(true, false));
    try std.testing.expectEqual(@as(usize, 0), unchargedFirstRing(true, true));
    try std.testing.expectEqual(@as(usize, 0), unchargedFirstRing(false, false));
    try dr.restart(0, 0);
    defer if (dr.slots.items.len > 0) if (dr.slots.items[0].ring) |*r| r.free();
    try std.testing.expect(dr.slots.items[0].ring != null);
    try std.testing.expectEqual(@as(usize, 0), unchargedFirstRing(true, dr.slots.items[0].ring != null));
    const seq: usize = 4096;
    try std.testing.expectEqual(seq + dflash.ring_bytes, extraStreamBytes(seq, true, 0));
    try std.testing.expectEqual(seq, extraStreamBytes(seq, false, 0));
}

test "canonical admission includes MAX_S and pool bounds" {
    try std.testing.expectEqual(@as(u32, 16), laneLimit(true, false, false));
    try std.testing.expect(laneLimit(true, false, false) < 17);
    try std.testing.expectEqual(@as(u32, @intCast(engine.max_shared)), laneLimit(false, true, false));
    try std.testing.expectEqual(@as(u32, std.math.maxInt(u32)), laneLimit(false, false, false));
    const pooled = laneLimit(true, false, true);
    try std.testing.expect(pooled <= 16);
    try std.testing.expect(pooled <= @as(u32, @intCast(@import("cuda_kv_extent.zig").max_slots)));
}

// Each rank keeps a copy under rank 0's id, in the room rank 0 admitted by (the smallest over the ranks).

/// Device bytes retained for the visible prefix at this prompt mark.
pub fn snap_bytes(ptr: *anyopaque, at: u32) u64 {
    const e: *engine.Engine = @ptrCast(@alignCast(ptr));
    return e.snapshotBytes(at);
}

/// The prompt pass stands at `at` (a mark at a chunk start): copy the state there under a new id.
pub fn snap_save(ptr: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!*anyopaque {
    _ = owner;
    const e: *engine.Engine = @ptrCast(@alignCast(ptr));
    if (e.pos() != at) return error.NotAtMark;
    const k = try e.gpa.create(engine.Kept);
    errdefer e.gpa.destroy(k);
    k.* = .{ .id = try e.keepState(), .at = at };
    return k;
}

/// A restore outside a prompt pass (the store's `begin`) would leave the other ranks behind, so it refuses.
pub fn snap_restore(ptr: *anyopaque, owner: ?*anyopaque, saved: *anyopaque) anyerror!void {
    _ = ptr;
    _ = owner;
    _ = saved;
    return error.RestoredInPromptPass;
}

pub fn snap_drop(ptr: *anyopaque, saved: *anyopaque) void {
    const e: *engine.Engine = @ptrCast(@alignCast(ptr));
    const k: *engine.Kept = @ptrCast(@alignCast(saved));
    e.dropState(k.id);
    e.gpa.destroy(k);
}

pub fn snap_tier(ptr: *anyopaque) ?*anyopaque {
    const e: *engine.Engine = @ptrCast(@alignCast(ptr));
    return if (e.disk) |p| p.tier else null;
}

pub fn snap_spill_write(ptr: *anyopaque, saved: *anyopaque, _: anytype, name: anytype, tokens: []const u32) !void {
    const e: *engine.Engine = @ptrCast(@alignCast(ptr));
    const k: *engine.Kept = @ptrCast(@alignCast(saved));
    const p = e.disk orelse return error.SpillDisabled;
    try p.request(p.ctx, .{ .operation = .write, .id = k.id, .key = name.key, .at = name.at, .tokens = tokens });
}

pub fn snap_spill_read(ptr: *anyopaque, _: anytype, name: anytype) !*anyopaque {
    const e: *engine.Engine = @ptrCast(@alignCast(ptr));
    const p = e.disk orelse return error.SpillDisabled;
    const k = try e.gpa.create(engine.Kept);
    errdefer e.gpa.destroy(k);
    k.* = .{ .id = e.next_snap, .at = name.at };
    e.next_snap += 1;
    try p.request(p.ctx, .{ .operation = .read, .id = k.id, .key = name.key, .at = name.at });
    return k;
}

pub fn snap_spill_forget(ptr: *anyopaque, name: anytype) void {
    const e: *engine.Engine = @ptrCast(@alignCast(ptr));
    if (e.disk) |p| p.forget(p.ctx, name.key, name.at);
}

fn close(ptr: *anyopaque) void {
    const own: *Owned = @ptrCast(@alignCast(ptr));
    const rank: u32 = @intCast(own.e.share.rank);
    cuda.memory_receipt.emit(rank, .cleanup_begin, false);
    cuda.tp_watch.ended(); // the other ranks leave on the end step, so their lines close on purpose from here
    own.lanes.x.end() catch {};
    own.lanes.deinit();
    release(own);
    cuda.memory_receipt.emit(rank, .cleanup_end, false);
}

/// What a rank owns, freed by rank 0's close and by a follower before it leaves.
fn release(own: *Owned) void {
    own.e.drain() catch |err| {
        own.e.pair_failure(err);
        return;
    };
    if (own.vision) |v| v.close();
    if (own.images) |v| v.deinit();
    own.link.deinit();
    own.e.deinit();
    if (own.probe) |probe| probe.close();
    _ = own.lib.api.ncclCommDestroy(own.comm);
    own.lib.close();
    own.gpa.destroy(own);
}

/// The server's words for this family's refusals.
pub fn explain(_: ?*anyopaque, err: anyerror) ?[]const u8 {
    return switch (err) {
        error.NoRank => "GLM-5.3-Flash runs at TP over hosts: set TF_GLM_RANK and TF_GLM_WORLD on every host (WORLD 1: one GPU, with TF_GLM_LAYERS for a thin model)",
        error.NoCommId => "GLM-5.3-Flash's ranks meet over NCCL: with TF_GLM_WORLD 2 or more, set NCCL_COMM_ID to rank 0's address:port on every host",
        error.RankMismatch => "the ranks' settings differ (each rank's log names the one): start them alike",
        error.TooManyRanks => std.fmt.comptimePrint(
            "GLM-5.3-Flash on CUDA runs on at most {d} ranks: set TF_GLM_WORLD to {d} or fewer",
            .{ ranks.max_world, ranks.max_world },
        ),
        error.WeightRecipeMismatch, error.BadFp8Source => "GLM checkpoint tensors disagree with the ModelOpt module recipe",
        error.NotModelOpt, error.UnsupportedQuant => "GLM-5.3-Flash on CUDA reads ModelOpt NVFP4, mixed FP4/FP8 and EXL3 checkpoints (the log names what config.json has)",
        error.UnreadableFormat => "a projection is stored in a format GLM-5.3-Flash on CUDA does not read (the log names its module, recipe and tensors)",
        else => null,
    };
}

test "a place inside its world of at most 16 ranks serves; a larger world is refused by name" {
    try std.testing.expectEqual(Place{ .rank = 15, .world = 16 }, try checked(.{ .rank = 15, .world = 16 }));
    try std.testing.expectError(error.TooManyRanks, checked(.{ .rank = 0, .world = 17 }));
    try std.testing.expectError(error.NoRank, checked(.{ .rank = 2, .world = 2 }));
    try std.testing.expect(explain(null, error.TooManyRanks) != null);
}

test "a checkpoint refused for its format, at the entry or by the loader, is explained" {
    for ([_]anyerror{ error.NotModelOpt, error.UnsupportedQuant, error.UnreadableFormat }) |e| try std.testing.expect(explain(null, e) != null);
}

test "capacity keeps a verify window past the context and a multiple of 64" {
    try std.testing.expectEqual(@as(usize, 8256), capacity(8192));
    try std.testing.expectEqual(@as(usize, 330560), capacity(330529));
    try std.testing.expect(capacity(330529) % 64 == 0);
}

pub fn linkCapacity(cap: usize, media: ?images_mod.Limits) !usize {
    const lim = media orelse return cap;
    const frames: usize = (@as(usize, lim.max_frames) + 1) / 2;
    const pairs = std.math.mul(usize, frames, lim.max_videos) catch return error.TooManyImages;
    const spans = std.math.add(usize, pairs, lim.max_images) catch return error.TooManyImages;
    const words = std.math.mul(usize, spans, 3) catch return error.TooManyImages;
    return @max(cap, words);
}
test "a small text context still sizes the link for every offered media span" {
    try std.testing.expectEqual(@as(usize, 792), try linkCapacity(64, .{ .image_tokens = 2048, .max_images = 8, .video_tokens = 4096, .max_videos = 2, .max_frames = 256 }));
}
