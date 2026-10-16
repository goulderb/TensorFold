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
const heat = @import("heat");
const owner_kv = @import("cuda_kv_owner.zig");
const cache_pool = @import("cuda_kv_pool.zig");
const modelopt = @import("core").modelopt;
const vision_binding = @import("cuda_vision_binding.zig");
const vision_runtime = @import("cuda_vision_runtime.zig");
const dflash_mod = @import("cuda_dflash.zig");
const mia = @import("cuda_experts_mia.zig");
pub const persistence = @import("cuda_persistence.zig");
pub const spill_payload = @import("cuda_spill_payload.zig");

pub const Room = struct { ctx: *anyopaque, read: *const fn (*anyopaque) ?u64 };

pub const Options = struct {
    room: ?Room = null,
    capacity: usize, // tokens a sequence holds
    mtp: bool, // load the MTP head (drafts)
    settings: ranks.Settings,
    draft_kernels: ?[]const u8 = null,
};

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

/// Rank 0's own word on its request (cancelled or not); the other ranks have none and hear rank 0's.
pub const Cancel = struct { ptr: ?*anyopaque, check: *const fn (?*anyopaque) bool };

/// The most drafts a chain holds (decode.DepthPolicy's top depth).
pub const max_drafts = 3;

/// A prompt pass's chunk rows (decode.prefill's).
pub const prompt_rows = fw.prefill_rows;

/// A shared verify's max rows and streams: under hc_pre's 64-row block change, 8 streams of 4 rows.
pub const shared_rows = 32;
pub const max_shared = 8;

/// One stream's window of a shared verify: its slot, its ids, and its drawn tokens.
pub const Part = struct { slot: u32, ids: []const u32, out: []u32 };

/// A kept prompt state as the prompt cache holds it: the id every rank keeps its own copy under, and its position.
pub const Kept = struct { id: u64, at: u32 };

/// A stream's own sequence: state, sampling rule, MTP drafts; buffers are shared, one stream's round at a time.
const Seq = struct {
    st: fw.State,
    sampling: ?sampler.Sampling = null, // null: greedy
    in_head: usize = 0, // drafted rows the MTP caches hold past the kept ones (st.mtp_drafted)
    retained: ?cuda.DeviceBuffer = null, // MTP rows that must survive another sequence using the forward scratch
    rows_at: u64 = 0, // its last verify's final-norm rows (bf16 [rows][hidden]), which the next draft reads
    vision: ?VisionOwned = null,
};

pub const VisionOwned = union(enum) {
    leader: vision_runtime.Service.Encoded,
    follower: struct { table: vision_binding.Table, ids: []u32, positions: []u32 },

    pub fn table(self: *VisionOwned) *vision_binding.Table {
        return switch (self.*) {
            .leader => |*state| &state.request.table.?,
            .follower => |*state| &state.table,
        };
    }
};

/// One rank's copy of a kept prompt state: visible cache prefixes, raw index tails and fixed recurrent state.
const Snap = struct { buf: cuda.DeviceBuffer, pos: usize, mtp_len: usize, cur: []u1, owner: ?u32 = null };

/// Where a prompt pass resumes: a kept state (every rank holds `id`) at `at`, a multiple of the chunk rows.
pub const Resume = struct { id: u64, at: usize };

/// Rank 0's marks in a prompt pass: `call` may keep the state at a chunk start; other ranks copy at that chunk's gate.
pub const Marks = struct { at: []const u32, ctx: *anyopaque, call: *const fn (ctx: *anyopaque, at: u32) void };

pub const Engine = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    d: *const cuda.Driver,
    c: config.Config,
    share: shard.Share,
    spans: []shard.Span, // every rank's vocabulary span
    lib: *const cuda.nccl.Library,
    comm: cuda.nccl.Comm,
    settings: ranks.Settings,
    m: fw.Modules,
    stream: cuda.Stream,
    pieces: fw.Pieces,
    w: weights.Weights,
    bp: fw.Buffers, // prompt chunks
    bd: fw.Buffers, // verify windows
    bm: fw.Buffers, // the MTP head's rows
    bs: ?fw.Buffers = null, // shared verify windows (with the MTP head, as the lane core shares only then)
    pairs: cuda.DeviceBuffer, // every rank's (value, id) of a greedy draw
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

    /// Opens rank `rank` of `world` over `comm`; the ranks agree on settings (cuda_ranks) before the weights load.
    pub fn init(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, ctx: *const cuda.Context, checkpoint: []const u8, lib: *const cuda.nccl.Library, comm: cuda.nccl.Comm, rank: usize, world: usize, o: Options) !*Engine {
        const e = try gpa.create(Engine);
        errdefer gpa.destroy(e);
        var arena = std.heap.ArenaAllocator.init(gpa);
        defer arena.deinit();
        const a = arena.allocator();
        const cfg_path = try std.fs.path.join(a, &.{ checkpoint, "config.json" });
        const text = try std.Io.Dir.cwd().readFileAlloc(io, cfg_path, a, .limited(1 << 24));
        var why: modelopt.Why = .{};
        _ = @import("cuda_checkpoint.zig").admit(gpa, text, &why) catch |err| {
            std.log.err("{s}: {s}", .{ cfg_path, why.text() });
            return err;
        };
        const c = try config.parse(gpa, text);
        var stream = try cuda.Stream.init(d, false);
        errdefer stream.deinit();
        var settings = o.settings;
        var startup_problem: ?anyerror = null;
        settings.draft = ranks.effectiveDraft(settings.draft, o.mtp) catch |err| blk: {
            startup_problem = err;
            break :blk settings.draft;
        };
        if (settings.draft == .dflash2 and startup_problem == null) {
            if (world != 2 or c.hidden != 4096 or c.hc != 4 or c.vocab != 154880 or (settings.layers != 0 and settings.layers != c.layers)) startup_problem = error.UnsupportedDFlashTarget;
            if (startup_problem == null) settings.draft_checkpoint = ranks.hashDraftCheckpoint(gpa, io, settings.draft_dir orelse return error.NoDraftCheckpoint) catch |err| blk: {
                startup_problem = err;
                break :blk @splat(0);
            };
        }
        const startup = try a.alloc(u32, world);
        try ranks.gather(u32, d, lib, comm, stream, rank, @intFromBool(startup_problem == null), startup);
        for (startup) |ready| if (ready == 0) return startup_problem orelse error.PeerStartupRefused;
        const with_mtp = settings.draft == .mtp;
        const fp = ranks.Fingerprint.of(world, rank, settings, o.capacity, try ranks.hashFile(a, io, cfg_path), try ranks.hashFile(a, io, "/proc/self/exe"), checkpoint);
        try ranks.agree(d, lib, comm, stream, fp, try a.alloc(ranks.Fingerprint, world));
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
        if (settings.exl3_routed == .mia_mcg4) m.mia = try mia.Kernels.load(d, cuda.kernels.glm_exl3);
        m.experts.exl3.decode = o.settings.exl3_decode;
        var pieces = try fw.Pieces.init(d, o.settings.pieces);
        errdefer pieces.deinit();
        if (o.settings.layers > c.layers) return error.BadLayers;
        const layers = try a.alloc(u32, if (o.settings.layers > 0) o.settings.layers else c.layers);
        for (layers, 0..) |*l, i| l.* = @intCast(i);
        e.* = .{ .gpa = gpa, .io = io, .heat_gate = .{ .bands = settings.heat, .root = heatRoot() }, .shared_buf = undefined, .d = d, .c = c, .share = share, .spans = spans, .lib = lib, .comm = comm, .settings = settings, .m = m, .stream = stream, .pieces = pieces, .w = undefined, .capacity = o.capacity, .bp = undefined, .bd = undefined, .bm = undefined, .pairs = undefined, .smp = undefined, .mtp = with_mtp };
        var wt = e.m.tools();
        wt.precision = .{ .dense = o.settings.weight_dense, .head = o.settings.weight_head };
        wt.exl3_routed = settings.exl3_routed;
        if (rank == 0) {
            var policy_text: [128]u8 = undefined;
            const policy_hash = o.settings.weight_dense.digest();
            std.log.info("GLM dense conversion policy v1: {s}; head={s}; scope=bf16-source-only; digest={s}", .{ try o.settings.weight_dense.format(&policy_text), @tagName(o.settings.weight_head), std.fmt.bytesToHex(policy_hash, .lower) });
        }
        e.w = try weights.load(gpa, io, d, stream, wt, checkpoint, c, share, layers, with_mtp);
        errdefer e.w.deinit();
        try e.bind(0);
        errdefer {
            e.free(0);
            e.seqs.deinit(gpa);
        }
        e.bp = if (o.settings.kv_shard) try fw.Buffers.initOwned(d, &e.w, fw.prefill_rows, true, o.capacity) else try fw.Buffers.init(d, &e.w, fw.prefill_rows, true, o.capacity);
        errdefer e.bp.deinit();
        e.bd = if (o.settings.kv_shard) try fw.Buffers.initOwned(d, &e.w, fw.window_rows, false, o.capacity) else try fw.Buffers.init(d, &e.w, fw.window_rows, false, o.capacity);
        errdefer e.bd.deinit();
        e.bm = if (o.settings.kv_shard) try fw.Buffers.initOwned(d, &e.w, fw.window_rows, false, o.capacity) else try fw.Buffers.init(d, &e.w, fw.window_rows, false, o.capacity);
        errdefer e.bm.deinit();
        if (with_mtp or settings.draft == .dflash2) e.bs = if (o.settings.kv_shard) try fw.Buffers.initSharedOwned(d, &e.w, shared_rows, o.capacity) else try fw.Buffers.initShared(d, &e.w, shared_rows, o.capacity);
        errdefer if (e.bs) |*b| b.deinit();
        if (settings.draft == .dflash2) {
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
        e.pairs = try cuda.DeviceBuffer.alloc(d, 8 * world);
        errdefer e.pairs.free();
        e.smp = try sampler.Sampler.init(d, c.vocab, fw.window_rows);
        errdefer e.smp.deinit();
        e.shared_buf = try cuda.DeviceBuffer.alloc(d, 8);
        errdefer e.shared_buf.free();
        if (settings.draft == .dflash2) {
            var draft_problem: ?anyerror = null;
            e.dflash = dflash_mod.DFlash.open(gpa, io, checkpoint, settings.draft_dir.?, max_shared, o.capacity, .{ .driver = d, .context = ctx, .stream = stream, .embed = e.w.embed, .lib = lib, .comm = comm, .rank = rank, .world = world, .kernels_dir = o.draft_kernels }) catch |err| blk: {
                draft_problem = err;
                break :blk null;
            };
            errdefer if (e.dflash) |drafter| drafter.deinit();
            if (!try e.pairedReady(draft_problem == null)) return draft_problem orelse error.PeerDFlashOpen;
            e.dflash.?.prepare() catch |err| {
                e.pair_failure(err);
                return err;
            };
        }
        errdefer if (e.dflash) |drafter| drafter.deinit();
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
        try e.stream.synchronize();
        return e;
    }

    pub fn deinit(e: *Engine) void {
        if (e.disk) |p| p.close(p.ctx);
        e.stream.synchronize() catch {};
        var it = e.snaps.valueIterator();
        while (it.next()) |s| e.freeSnap(s);
        e.snaps.deinit(e.gpa);
        e.dropped.deinit(e.gpa);
        if (e.dflash) |drafter| drafter.deinit();
        e.shared_buf.free();
        e.smp.deinit();
        e.pairs.free();
        if (e.bs) |*b| b.deinit();
        e.bm.deinit();
        e.bd.deinit();
        e.bp.deinit();
        for (0..e.seqs.items.len) |i| e.free(i);
        e.seqs.deinit(e.gpa);
        if (e.pool) |*p| p.deinit();
        e.w.deinit();
        e.pieces.deinit();
        e.m.deinit();
        e.stream.deinit();
        e.gpa.free(e.spans);
        const gpa = e.gpa;
        gpa.destroy(e);
    }

    fn exchange(e: *const Engine) fw.Exchange {
        return .{ .nccl = .{ .lib = e.lib, .comm = e.comm } };
    }

    fn view(e: *Engine, b: *fw.Buffers, out: *[1]fw.Rank) fw.Forward {
        out.* = .{.{ .w = &e.w, .st = &e.cur.st, .b = b }};
        return .{ .d = e.d, .m = &e.m, .stream = e.stream, .ranks = out, .exchange = e.exchange(), .pieces = if (b == &e.bp) &e.pieces else null, .split = e.settings.split, .kda_wide = e.settings.kda_wide, .vision = if (b == &e.bp) if (e.cur.vision) |*vision| .{ .owner = vision.table(), .generation = vision.table().generation } else null else null };
    }

    /// Binds slot `slot`'s sequence for the steps that follow, making it (a zeroed state) on first use.
    pub fn bind(e: *Engine, slot: u32) !void {
        if (e.settings.draft == .dflash2 and slot >= max_shared) return error.BadDraftSlot;
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
        if (e.dflash) |drafter| drafter.release(slot);
        if (e.pool) |*p| p.regions.release(slot);
        if (e.settings.vision and slot < e.seqs.items.len) {
            e.stream.synchronize() catch {};
            e.clearVision(slot);
        }
        if (slot != 0 and slot < e.seqs.items.len) e.free(slot);
        e.cur = e.seqs.items[0].?;
        e.slot = 0;
    }

    fn free(e: *Engine, slot: usize) void {
        const q = e.seqs.items[slot] orelse return;
        e.stream.synchronize() catch {};
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

    /// A fresh sequence with `sampling` (null or temperature 0: greedy).
    pub fn begin(e: *Engine, sampling: ?sampler.Sampling) !void {
        if (e.pool != null and e.slot >= cache_pool.extent.max_slots) return error.BadKvRequest;
        if (e.pool) |*p| if (p.regions.slots[e.slot] == null) {
            const r = try p.regions.plan(e.slot, e.capacity);
            try e.reserveAt(r);
        };
        e.cur.sampling = sampler.check(sampling);
        e.cur.in_head = 0;
        try e.cur.st.reset(e.stream);
        if (e.dflash) |drafter| try drafter.reset(e.slot);
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
        const last = start + rows == len;
        var first: ?u32 = null;
        if (last) {
            var t: [1]u32 = undefined;
            try e.draw(e.bp.logits, 1, len, &t);
            first = t[0];
        }
        if (e.mtp) {
            if (last) try e.retainRows(e.bp.fnormed + (rows - 1) * e.c.hidden * 2, 1);
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
        return first;
    }

    /// The chunk gate on every rank: a kept state at the mark, the heat wait, rank 0's cancel; the id exchanged first.
    fn gate(e: *Engine, cancel: ?Cancel) !void {
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

    /// Device bytes retained at a prompt mark; MTP has absorbed the same prefix at these marks.
    pub fn snapshotBytes(e: *const Engine, at: usize) u64 {
        const plan = e.cur.st.layout.snapshot(.{ .main = at, .mtp = if (e.mtp) at else 0 }) catch return std.math.maxInt(u64);
        return e.payloadBytes(plan.byteCount()) catch return std.math.maxInt(u64);
    }

    fn payloadBytes(e: *const Engine, target_bytes: u64) !u64 {
        return spill_payload.withDraftBytes(target_bytes, if (e.dflash) |drafter| drafter.snapshotBytes() else 0);
    }

    pub fn spillShape(e: *const Engine) spill_payload.Shape {
        const st = &e.seqs.items[0].?.st;
        return .{ .layout = st.layout, .format = e.settings.kv, .mtp = e.mtp, .sharded = e.settings.kv_shard, .rank = @intCast(e.share.rank), .world = @intCast(e.share.world), .grid = prompt_rows, .parities = st.cur.len, .draft_bytes = if (e.dflash) |drafter| drafter.snapshotBytes() else 0, .draft_identity = e.settings.draft_checkpoint };
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
        try e.stream.synchronize();
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
        if (e.dflash) |drafter| if (!drafter.snapshotReady(e.slot, st.pos)) return error.NotAtMark;
        const plan = try st.layout.snapshot(.{ .main = st.pos, .mtp = st.mtp_len });
        const owner = if (st.layout.external_planes) st.mtp_caches.owner else null;
        const resident = owner == null or owner.? == e.share.rank;
        const target_bytes = plan.bytesFor(resident);
        var buf = try cuda.DeviceBuffer.alloc(e.d, @intCast(try e.payloadBytes(target_bytes)));
        errdefer buf.free();
        var it = plan.rangesFor(resident);
        while (it.next()) |r| try e.copy(buf.ptr + r.saved, st.address(r), r.bytes);
        if (e.dflash) |drafter| try drafter.save(e.slot, buf.ptr + std.mem.alignForward(usize, target_bytes, 256));
        try e.stream.synchronize();
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
        errdefer if (st.layout.external_planes or e.dflash != null) e.pair_failure(error.PairedRestoreFailed);
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
        if (e.dflash) |drafter| try drafter.restore(e.slot, saved.buf.ptr + std.mem.alignForward(usize, plan.?.bytesFor(source_resident), 256), saved.pos);
        try e.stream.synchronize();
        try st.restoreHost(e.stream, saved.pos, saved.mtp_len, saved.cur);
        e.cur.in_head = 0;
        try e.stream.synchronize();
    }

    fn ownerTransport(e: *const Engine) owner_kv.Transport {
        return .{ .d = e.d, .lib = e.lib, .comm = e.comm, .rank = e.share.rank, .failure = e.pair_failure };
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
    fn shared(e: *Engine, v: f64) !f64 {
        var x = v;
        try e.shared_buf.uploadAsync(0, std.mem.asBytes(&x), e.stream.handle);
        try e.lib.check(e.lib.api.ncclAllReduce(e.shared_buf.ptr, e.shared_buf.ptr, 1, .f64, .max, e.comm, e.stream.handle), "ncclAllReduce");
        try e.stream.synchronize();
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
    pub fn verify(e: *Engine, ids: []const u32, out: []u32) !void {
        return e.verifyWithMask(ids, out, false);
    }

    pub fn verifyMasked(e: *Engine, ids: []const u32, out: []u32) !void {
        return e.verifyWithMask(ids, out, true);
    }

    fn verifyWithMask(e: *Engine, ids: []const u32, out: []u32, masked: bool) !void {
        if (ids.len == 0 or ids.len > fw.window_rows) return error.WindowTooWide;
        var one: [1]fw.Rank = undefined;
        const f = e.view(&e.bd, &one);
        const at = e.cur.st.pos;
        try f.run(@ptrCast(ids));
        try e.retainRows(e.bd.fnormed, ids.len);
        if (e.dflash) |drafter| try drafter.stageWindow(e.slot, e.bd.taps.?.ptr, e.bd.rows, 0, ids.len, at);
        if (masked) try e.drawMasked(e.bd.logits, ids.len, at + 1, out) else try e.draw(e.bd.logits, ids.len, at + 1, out);
    }

    /// Several streams' windows in one forward: shared row ops, per-part attention and KDA.
    pub fn verifyShared(e: *Engine, parts: []const Part) !void {
        const b = if (e.bs) |*x| x else return error.NoSharedWindows;
        if (parts.len == 0 or parts.len > max_shared) return error.WindowTooWide;
        var segs: [max_shared]fw.Seg = undefined;
        var ids: [shared_rows]u32 = undefined;
        var rows: usize = 0;
        for (parts, 0..) |p, i| {
            if (p.ids.len == 0 or p.ids.len > fw.window_rows or rows + p.ids.len > shared_rows) return error.WindowTooWide;
            for (parts[0..i]) |q| if (q.slot == p.slot) return error.SlotTwice;
            try e.bind(p.slot);
            segs[i] = .{ .st = &e.cur.st, .row0 = rows, .rows = p.ids.len };
            @memcpy(ids[rows..][0..p.ids.len], p.ids);
            rows += p.ids.len;
        }
        var one: [1]fw.Rank = .{.{ .w = &e.w, .st = segs[0].st, .b = b, .segs = segs[0..parts.len] }};
        const f: fw.Forward = .{ .d = e.d, .m = &e.m, .stream = e.stream, .ranks = &one, .exchange = e.exchange(), .pieces = null, .split = e.settings.split, .kda_wide = e.settings.kda_wide };
        try f.run(@ptrCast(ids[0..rows]));
        for (parts, segs[0..parts.len]) |p, g| {
            try e.bind(p.slot);
            e.cur.rows_at = b.fnormed + g.row0 * e.c.hidden * 2;
            if (e.dflash) |drafter| try drafter.stageWindow(p.slot, b.taps.?.ptr, b.rows, g.row0, g.rows, g.st.pos);
            try e.draw(b.logits + g.row0 * e.share.vocab.len() * 2, g.rows, g.st.pos + 1, p.out);
        }
    }

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
        return e.mtp or e.dflash != null;
    }

    pub fn draft(e: *Engine, from_prompt: bool, follow: []const u32, depth: usize, out: []u32) !usize {
        if (depth > max_drafts or depth > out.len or follow.len == 0) return error.BadDraftRequest;
        if (e.dflash) |drafter| return drafter.propose(e.slot, follow[follow.len - 1], e.cur.sampling, out[0..depth]);
        if (!e.mtp) return error.NoDraftHead;
        if (depth > max_drafts) return error.TooDeep;
        var one: [1]fw.Rank = undefined;
        const f = e.view(&e.bm, &one);
        const D = e.c.hidden;
        const hidden = e.draftRows(from_prompt);
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
            try e.draw(e.bm.logits, 1, e.cur.st.pos + 1 + j, out[j..][0..1]);
            if (j + 1 < depth) {
                try f.mtp(@ptrCast(out[j..][0..1]), &.{e.bm.fnormed});
                try e.cur.st.setMtpLen(e.stream, e.cur.st.mtp_len + 1);
                e.cur.in_head += 1;
            }
        }
        return depth;
    }

    /// `rows` logits rows of this rank's vocabulary -> the tokens at positions first, first + 1, ...
    fn draw(e: *Engine, logits: u64, rows: usize, first: u64, out: []u32) !void {
        if (e.cur.sampling) |s| return e.smp.draw(e.stream, e.exchange(), e.spans, &.{.{ .rank = e.share.rank, .logits = logits }}, rows, first, sampler.rule(s), out);
        for (0..rows) |r| out[r] = try e.greedy(logits + r * e.share.vocab.len() * 2);
    }

    /// The prompt's target logits survive MTP absorption; redraw them once the lane supplies its first mask.
    pub fn redrawFirst(e: *Engine, position: u64) !u32 {
        var out: [1]u32 = undefined;
        try e.drawMasked(e.bp.logits, 1, position, &out);
        return out[0];
    }

    fn drawMasked(e: *Engine, logits: u64, rows: usize, first: u64, out: []u32) !void {
        const rule: ?sampler.Rule = if (e.cur.sampling) |s| sampler.rule(s) else null;
        try e.smp.masked(e.stream, e.exchange(), e.spans, &.{.{ .rank = e.share.rank, .logits = logits }}, rows, first, rule, e.m.glue.top1, out);
    }

    /// decode.sample_rows' greedy pick: per-rank top-1 (tf_glm_top1), all-gathered, largest value then lowest id.
    fn greedy(e: *Engine, logits: u64) !u32 {
        const world = e.share.world;
        var a: cuda.Args = .{};
        a.add(logits);
        a.add(@as(i64, @intCast(e.share.vocab.len())));
        a.add(@as(u32, @intCast(e.share.vocab.lo)));
        a.add(e.pairs.ptr + e.share.rank * 8);
        try cuda.launch.launch(e.m.glue.top1, .{ .grid = .{ .x = 1 }, .block = .{ .x = 1024 } }, e.stream, &a);
        try e.lib.check(e.lib.api.ncclAllGather(e.pairs.ptr + e.share.rank * 8, e.pairs.ptr, 2, .u32, e.comm, e.stream.handle), "ncclAllGather");
        try e.stream.synchronize();
        var all: [16][2]u32 = undefined;
        try e.d.check(e.d.api.cuMemcpyDtoH_v2(&all, e.pairs.ptr, world * 8), "cuMemcpyDtoH");
        var pick = all[0];
        for (all[1..world]) |p| {
            const v: f32 = @bitCast(p[0]);
            const b: f32 = @bitCast(pick[0]);
            if (v > b or (v == b and p[1] < pick[1])) pick = p;
        }
        return pick[1];
    }

    fn heatRoot() []const u8 {
        const p = std.c.getenv("TF_HEAT_ROOT") orelse return heat.default_root;
        const s = std.mem.span(p);
        return if (s.len == 0) heat.default_root else s;
    }

    fn copy(e: *const Engine, dst: u64, src: u64, bytes: usize) !void {
        try e.d.check(e.d.api.cuMemcpyDtoDAsync_v2(dst, src, bytes, e.stream.handle), "cuMemcpyDtoDAsync");
    }
};

test "prompt snapshot estimates follow the requested mark and MTP setting" {
    var seq: Seq = undefined;
    seq.st.layout = try @import("cuda_layout.zig").Layout.init(.{ 4, 512, 512, 0, 0, 0, 0, 0 }, 8192, fw.prefill_rows, 2, .fp8, 512, 128);
    seq.st.pos = 0;
    seq.st.mtp_len = 0;
    var e: Engine = undefined;
    e.dflash = null;
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
    e.dflash = null;
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
    seq.st.layout = try @import("cuda_layout.zig").Layout.init(.{ 4, 0, 0, 0, 0, 0, 0, 0 }, 8192, fw.prefill_rows, 0, .bf16, 512, 128);
    var e: Engine = undefined;
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
        fn sync(_: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
            return 0;
        }
    };
    var d: cuda.Driver = .{ .lib = undefined, .api = undefined };
    d.api.cuMemAlloc_v2 = Mock.alloc;
    d.api.cuMemFree_v2 = Mock.free;
    d.api.cuMemcpyDtoDAsync_v2 = Mock.copy;
    d.api.cuMemsetD32Async = Mock.fill;
    d.api.cuStreamSynchronize = Mock.sync;
    const stream: cuda.Stream = .{ .d = &d, .handle = null };
    for ([_]@import("cuda_cache.zig").Format{ .bf16, .fp8 }) |format| for ([_]bool{ false, true }) |mtp| {
        const layout = try @import("cuda_layout.zig").Layout.init(.{ 4, 256, 256, 256, 256, 256, 256, 256 }, 4096, fw.prefill_rows, 2, format, 512, 128);
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
        e.dflash = null;
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

test "spill snapshot roundtrip checks metadata and discards failed imports without publishing an id" {
    const M = @import("cuda_test_driver.zig").Memory;
    var d = M.driver();
    for ([_]@import("cuda_cache.zig").Format{ .bf16, .fp8 }) |format| {
        var seq: Seq = undefined;
        seq.st.layout = try (try @import("cuda_layout.zig").Layout.init(.{ 256, 0, 0, 0, 0, 0, 0, 0 }, 4096, prompt_rows, 1, format, 512, 128)).external();
        var cur = [_]u1{1};
        seq.st.cur = &cur;
        var e: Engine = undefined;
        e.dflash = null;
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
            const layout = try (try Layout.init(.{ 4, 256, 256, 256, 256, 256, 256, 256 }, 4096, fw.prefill_rows, 1, format, 512, 128)).external();
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
            f.e.dflash = null;
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
    e.dflash = null;
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
