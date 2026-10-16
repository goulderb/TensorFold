//! Nemotron's CUDA engine, MTP head and lane backend behind the native family interface.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const engine = @import("cuda_engine.zig");
const state = @import("cuda_state.zig");
const Head = @import("cuda_mtp.zig").Head;
const Lanes = @import("cuda_lanes.zig").Cuda;
const lone = @import("cuda_lone.zig");
const reuse = @import("cuda_reuse.zig");

pub const snap_bytes = reuse.bytes;
pub const snap_save = reuse.save;
pub const snap_restore = reuse.restore;
pub const snap_drop = reuse.drop;
pub const snap_ready = reuse.ready; // the spare Mamba buffers it sets aside come out of the cache's budget
pub const kept_states = reuse; // their shape, a blank one and copies out and in, for the disk (--learn-conversations)
/// The kept-state types and facts the native spill adapter serializes (native/nemotron_spill.zig).
pub const Copy = reuse.Copy;
pub const Target = reuse.Target;
/// A verify window's most rows (Copy.rows never exceeds it).
pub const max_rows: u32 = state.max_rows;
pub const SpillShape = struct { kv_row: u64, mamba: u64, head: bool, max_len: u64 };
pub fn planes(t: *const Target) SpillShape {
    const sh = reuse.shape(t, 1);
    return .{ .kv_row = sh.kv, .mamba = sh.mamba, .head = sh.head, .max_len = t.e.max_len };
}

/// The files `open` reads from `dir` for its arithmetic, in load order (gpa-owned paths): config.json, the shard
/// index when there is one, every shard, and the MLX MTP head's file when `drafts` and it is there.
pub fn artifacts(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, drafts: bool) ![][:0]u8 {
    var out: std.ArrayList([:0]u8) = .empty;
    errdefer {
        for (out.items) |p| gpa.free(p);
        out.deinit(gpa);
    }
    try out.append(gpa, try std.fmt.allocPrintSentinel(gpa, "{s}/config.json", .{dir}, 0));
    const index = try std.fmt.allocPrintSentinel(gpa, "{s}/model.safetensors.index.json", .{dir}, 0);
    if (std.Io.Dir.cwd().access(io, index, .{})) |_| try out.append(gpa, index) else |_| gpa.free(index);
    const shards = try core.checkpoint.shardFiles(gpa, io, dir);
    defer gpa.free(shards);
    for (shards, 0..) |s, i| out.append(gpa, s) catch |err| {
        for (shards[i..]) |rest| gpa.free(rest);
        return err;
    };
    if (drafts) {
        const mtp = try std.fmt.allocPrintSentinel(gpa, "{s}/{s}", .{ dir, @import("cuda_weights.zig").mtp_file }, 0);
        if (std.Io.Dir.cwd().access(io, mtp, .{})) |_| try out.append(gpa, mtp) else |_| gpa.free(mtp);
    }
    return out.toOwnedSlice(gpa);
}

/// Prompt rows the loaded engine computed (owner thread only).
pub fn prefillCounts(t: *const Target) engine.PrefillCounts {
    return t.e.prefill_counts;
}

/// The loaded engine's weight format (`mlx` or `modelopt`).
pub fn formatName(t: *const Target) []const u8 {
    return if (t.e.c.ck) "modelopt/checkpoint" else if (t.e.c.format == .modelopt) "modelopt/full" else "mlx/full";
}
const core = @import("core");

/// Sliding Weights' learner (--slide): lessons change the weights this engine's lanes decode with.
pub const slide = @import("cuda_slide.zig");

pub const model_type = "nemotron_h";
pub const explains_open = true; // explain() words open refusals too (--vision on a checkpoint without a tower)
pub const formats: []const []const u8 = &.{ "mlx-q4g64", "modelopt-nvfp4" };
pub const default_context: i64 = engine.default_context;
pub const max_segments: u32 = @import("cuda_segments.zig").MAX;
pub const prompt_rows: u32 = state.pass_rows;

pub const Options = struct { context: usize, drafts: bool, segments: usize = 1, slide: bool = false, vision: ?vision.Limits = null, precision: ?@import("config.zig").Precision = null, split: bool = false, tp: bool = false };
pub const vision = @import("cuda_vision.zig");

/// Thinking budgets the lone driver takes (cuda_lone.zig): above it, the stream goes to the lane core before its cut.
pub const lone_min_budget = @import("cuda_lone.zig").min_budget;

/// Nanoseconds a reply's background pass waits for the next request first: its first chunk holds the GPU.
pub const warm_quiet: u64 = 150 * std.time.ns_per_ms;

/// A lone drafted stream's own driver: decodes it until it finishes (false) or `yield` hands it over (true).
pub const LoneRun = *const fn (ctx: *anyopaque, s: *lanes.Stream, hooks: *anyopaque, committed: *const fn (*anyopaque) void, yield: *const fn (*anyopaque) bool, resumed: bool) anyerror!bool;

/// What the native server drives: the lane backend, the facts its round loop reads, and how to free it.
pub const Loaded = struct {
    backend: lanes.backend.Backend,
    facts: lanes.Model,
    rows: u32,
    /// Device bytes each admitted stream allocates for its own sequence (caches, state, head caches).
    stream_bytes: usize,
    /// Streams that allocate nothing: the first takes the engine's own sequence, whose buffers the graphs hold.
    free_streams: u32 = 1,
    ctx: *anyopaque,
    deinit: *const fn (*anyopaque) void,
    lone: ?LoneRun = null, // called with `ctx`; null: every stream in the lane core
    target: *reuse.Target, // the prompt cache copies this engine (cuda_reuse.zig)
    learns: ?*engine.Engine = null, // --slide: the engine its learner trains, whose sites it attached at load
    learn_bytes: usize = 0, // --slide: what its trainer allocates at the first lesson, kept out of the streams' room
    vision: ?*vision.Vision = null, // --vision: the tower the server's image requests reach
    taps: []const u32 = &.{}, // the rows a recorder may ask `backend` for (empty: none)
    living: bool = false, // Living Weights loaded from the model folder: decoding is not the checkpoint's alone
    /// Two-GPU TP: the peer's room (free memory less the 1536 MiB floor) after its load; the host admits streams and
    /// kept states by the smaller of the two, since every one takes the same bytes on both.
    peer_room: ?struct { rank: usize, bytes: u64 } = null,
};

/// Two-GPU tensor parallel (TENSORFOLD_TP=2): rank 0 on this device and rank 1 on the other internal one, each with
/// its share of every block (cuda_tp2.zig). Rank 0 holds the head and samples; the lane backend sees one engine.
fn openTp(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, kernels: ?[]const u8, o: Options) !Loaded {
    const pair = (try splitPair(ctx.d)) orelse return error.TpNeedsTwoDevices;
    if (pair[0] != ctx.device_ordinal) return error.TpRankZeroNotFirst;
    const e = try engine.Engine.init(gpa, io, ctx, dir, kernels, .{ .context = o.context, .mtp = false, .graphs = false, .sampling = null, .segments = 1, .precision = o.precision, .tp = 0, .range = .{ .embed = true, .head = true, .norm_f = true } });
    errdefer e.deinit();
    const own = try gpa.create(Owned);
    errdefer gpa.destroy(own);
    own.* = .{ .gpa = gpa, .e = e, .head = null, .lanes = try Lanes.init(gpa, e, null), .target = .{ .e = e, .head = null }, .carve = null };
    errdefer own.lanes.deinit();
    const p = try gpa.create(Peer);
    errdefer gpa.destroy(p);
    p.gpa = gpa;
    p.ctx = try cuda.Context.init(ctx.d, pair[1]);
    errdefer p.ctx.deinit();
    try p.ctx.makeCurrent();
    p.e = try engine.Engine.init(gpa, io, &p.ctx, dir, kernels, .{ .context = o.context, .mtp = false, .graphs = false, .sampling = null, .segments = 1, .precision = o.precision, .tp = 1, .range = .{ .embed = true, .head = false, .norm_f = false } });
    errdefer {
        p.ctx.makeCurrent() catch {};
        p.e.deinit();
    }
    p.stage[0] = try cuda.HostBuffer.alloc(ctx.d, 64); // unused by TP; Peer.deinit frees it
    errdefer p.stage[0].free();
    const link = try tp2.Link.init(gpa, .{ ctx, &p.ctx }, state.pass_rows, e.c.hidden);
    errdefer link.deinit();
    e.peer = p.e;
    e.tp_link = link;
    own.peer = p;
    own.link = link;
    const floor: usize = 1536 << 20;
    const peer_free = (try p.ctx.memInfo()).free;
    if (peer_free < floor or (try ctx.memInfo()).free < floor) return error.TpMemoryFloor;
    try ctx.makeCurrent();
    try own.lanes.measure(io, dir);
    own.taps = .{@import("cuda_lanes.zig").finalTap(e.c)};
    e.prefill_counts = .{};
    return .{
        .backend = own.lanes.backend(),
        .facts = own.lanes.facts(),
        .rows = state.max_rows, // exact windows up to 16 rows, several streams' sharing one forward
        .stream_bytes = e.seqBytes(),
        .ctx = own,
        .deinit = release,
        .target = &own.target,
        .lone = null,
        .learns = null,
        .learn_bytes = 0,
        .vision = null,
        .taps = &own.taps,
        .living = false,
        .peer_room = .{ .rank = 1, .bytes = peer_free - floor },
    };
}

/// The first shard, index or config.json in `dir` that a write in place would reach through a link, or null.
pub fn linked(arena: std.mem.Allocator, io: std.Io, dir: []const u8) !?[]const u8 {
    return core.shard_edit.linked(arena, io, dir);
}

/// A shared expert's projection config.json keeps unquantized, as an earlier release's lessons left it, or null.
pub fn plainShared(arena: std.mem.Allocator, io: std.Io, dir: []const u8) !?[]const u8 {
    return core.shard_edit.plainShared(arena, io, dir);
}

const Owned = struct { gpa: std.mem.Allocator, e: *engine.Engine, head: ?*Head, lanes: Lanes, target: reuse.Target, carve: ?*cuda.Carveout, living: ?@import("cuda_sites.zig").Sites = null, taps: [1]u32 = undefined, peer: ?*Peer = null, link: ?*tp2.Link = null };
const tp2 = @import("cuda_tp2.zig");

/// The second rank of a layer split: its own device, context, engine and seam staging. The lane backend drives rank
/// 0 only, so the peer is invisible above the seam.
const Peer = struct {
    gpa: std.mem.Allocator,
    ctx: cuda.Context,
    e: *engine.Engine,
    stage: [1]cuda.HostBuffer,

    fn deinit(p: *Peer) void {
        p.e.deinit();
        p.stage[0].free();
        p.ctx.deinit();
        p.gpa.destroy(p);
    }
};

/// The two internal devices, by shared compute capability: the eGPU enumerates first and is a different class.
pub fn splitPair(d: *const cuda.Driver) !?[2]c_int {
    const count = try d.deviceCount();
    var caps: [8]u32 = undefined;
    var n: usize = 0;
    var i: c_int = 0;
    while (i < count and n < caps.len) : (i += 1) {
        var dev: cuda.abi.Device = 0;
        try d.check(d.api.cuDeviceGet(&dev, i), "cuDeviceGet");
        var major: c_int = 0;
        var minor: c_int = 0;
        try d.check(d.api.cuDeviceGetAttribute(&major, .compute_capability_major, dev), "cuDeviceGetAttribute");
        try d.check(d.api.cuDeviceGetAttribute(&minor, .compute_capability_minor, dev), "cuDeviceGetAttribute");
        caps[n] = @intCast(10 * major + minor);
        n += 1;
    }
    for (0..n) |a| for (a + 1..n) |b| if (caps[a] == caps[b]) return .{ @intCast(a), @intCast(b) };
    return null;
}

/// The layer each rank owns, by the model's own layer count. Not the midpoint: rank 0 also carries the embedding,
/// the final norm, the head and the MTP head, so the balance point sits past the middle.
pub fn splitAt(layers: usize) usize {
    return layers / 2 + 2;
}

/// Own-sequence graphs support both draw modes; drafting also captures the head.
pub fn open(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, kernels: ?[]const u8, o: Options) !Loaded {
    if (std.c.getenv("TF_NEMOTRON_RANK")) |r| if (!std.mem.eql(u8, std.mem.span(r), "0")) { // expert ranks never serve
        @import("cuda_worker.zig").serve(gpa, io, ctx, dir, o.precision) catch |err| {
            cuda.tp_watch.fail(@errorName(err)); // rank 0 stops with this rank's error
            std.log.err("expert rank stopped: {s}", .{@errorName(err)});
            std.c._exit(1); // a stream may still wait on rank 0: a normal exit would join it
        };
        std.process.exit(0);
    };
    if (o.split and o.tp) return error.SplitOrTp;
    if ((o.split or o.tp) and (o.drafts or o.slide or o.segments != 1 or o.vision != null)) return error.SplitUnsupportedOptions;
    if (o.tp) return openTp(gpa, io, ctx, dir, kernels, o);
    const carve = try openCarveout(gpa, ctx);
    errdefer if (carve) |c| closeCarveout(gpa, c);
    // A split's range must be known before the load, or rank 0 would read the whole checkpoint. The layer count
    // comes from the config, read here rather than inside the engine.
    const layers = if (o.split) (try @import("config.zig").Config.read(gpa, io, dir)).layers else 0;
    const mid = splitAt(layers);
    const range: @import("cuda_weights.zig").Range = if (o.split) .{ .first = 0, .last = mid, .embed = true, .head = true, .norm_f = false, .seam_norm = mid } else .all;
    const e = try engine.Engine.init(gpa, io, ctx, dir, kernels, .{ .context = o.context, .mtp = o.drafts, .graphs = !o.split, .sampling = null, .segments = o.segments, .carveout = carve, .slide = o.slide, .vision = o.vision, .precision = o.precision, .range = range });
    errdefer e.deinit();
    const head: ?*Head = if (o.drafts) try Head.init(e) else null;
    errdefer if (head) |h| h.deinit();
    // A layer split runs eagerly: a graph captures one device's kernels and cannot span the seam, and the seam's
    // synchronize is not permitted inside a capture.
    if (!o.split) {
        if (head) |h| try h.capture();
        if (!o.drafts) try e.captureWindows(); // the one-row window: Engine.init captures windows only when drafting
        // the sampled mode's sets: a placeholder rule now, each stream's own uploaded at its prefill
        try e.setSampling(.{ .seed = 0, .temperature = 1.0 });
        try e.captureWindows();
        if (head) |h| try h.capture();
        try e.setSampling(null);
    }
    const own = try gpa.create(Owned);
    errdefer gpa.destroy(own);
    own.* = .{ .gpa = gpa, .e = e, .head = head, .lanes = try Lanes.init(gpa, e, head), .target = .{ .e = e, .head = head }, .carve = carve };
    errdefer own.lanes.deinit();
    own.living = if (o.split) null else try slide.loadLiving(gpa, io, e, dir); // before the lanes' costs are measured: they decode through it
    errdefer if (own.living) |*s| s.deinit();
    // A layer split: rank 1 on the other internal device, with its own context and engine. Rank 0 keeps the
    // embedding, the head and the sample, so the lane backend above the seam is unchanged.
    if (o.split) {
        const pair = (try splitPair(ctx.d)) orelse return error.SplitNeedsTwoDevices;
        if (pair[0] != ctx.device_ordinal) return error.SplitRankZeroNotFirst;
        const p = try gpa.create(Peer);
        errdefer gpa.destroy(p);
        p.gpa = gpa;
        p.ctx = try cuda.Context.init(ctx.d, pair[1]);
        errdefer p.ctx.deinit();
        try p.ctx.makeCurrent();
        p.e = try engine.Engine.init(gpa, io, &p.ctx, dir, kernels, .{
            .context = o.context,
            .mtp = false, // the head stays on rank 0
            .graphs = false, // a graph captures one device's kernels and cannot span the seam
            .sampling = null,
            .segments = 1,
            .precision = o.precision,
            .range = .{ .first = mid, .last = layers, .embed = false, .head = false, .norm_f = true },
        });
        errdefer {
            p.ctx.makeCurrent() catch {};
            p.e.deinit();
        }
        try ctx.makeCurrent();
        // The seam's staging: h and y are rows*hidden*2 bf16, xs is rows*(hidden/64)*4 fp32. Sized for the widest
        // prompt pass, the widest thing that crosses: a window is at most state.max_rows rows.
        const rows = state.pass_rows;
        const need = 2 * rows * e.c.hidden * 2 + rows * (e.c.hidden / 64) * 4;
        p.stage[0] = try cuda.HostBuffer.allocMapped(ctx.d, need);
        errdefer p.stage[0].free();
        e.peer = p.e;
        e.stage = p.stage;
        own.peer = p;
        const floor: usize = 1536 << 20;
        if ((try p.ctx.memInfo()).free < floor or (try ctx.memInfo()).free < floor) return error.SplitMemoryFloor;
    }
    try ctx.makeCurrent();
    try own.lanes.measure(io, dir);
    own.taps = .{@import("cuda_lanes.zig").finalTap(e.c)};
    e.prefill_counts = .{}; // requests' prompt rows only: the cost measurement's passes are not counted
    return .{
        .backend = own.lanes.backend(),
        .facts = own.lanes.facts(),
        .rows = if (head != null) state.max_rows else 1,
        .stream_bytes = e.seqBytes(),
        .ctx = own,
        .deinit = release,
        .target = &own.target,
        .lone = if (head != null) loneRun else null,
        .learns = if (o.slide) e else null,
        .learn_bytes = if (o.slide) @import("cuda_train.zig").Trainer.memory(e.c) else 0,
        .vision = e.vision,
        .taps = &own.taps,
        .living = own.living != null,
    };
}

/// What a learner over the loaded engine `p` learns into; `dir` must outlive it.
pub fn learnModel(p: *anyopaque, dir: []const u8) slide.Model {
    const own: *Owned = @ptrCast(@alignCast(p));
    return .{ .e = own.e, .dir = dir, .living = &own.living };
}

fn loneRun(p: *anyopaque, s: *lanes.Stream, hooks: *anyopaque, committed: *const fn (*anyopaque) void, yield: *const fn (*anyopaque) bool, resumed: bool) anyerror!bool {
    const own: *Owned = @ptrCast(@alignCast(p));
    return lone.run(own.gpa, &own.lanes, s, .{ .ctx = hooks, .committed = committed, .yield = yield, .resumed = resumed });
}

/// A request this engine refuses, in words; null: none of its own.
pub fn explain(_: ?*anyopaque, err: anyerror) ?[]const u8 {
    return switch (err) {
        error.PromptTooLong => "the prompt and its reply exceed this server's context window: shorten it or lower max_tokens",
        error.OutOfDeviceMemory => "the GPU had no memory left for this request's caches: retry once another request ends",
        error.VisionRefused => "--vision: this checkpoint has no image tower the CUDA engine reads (Nemotron Omni's RADIO); serve it without --vision",
        error.NoVision => "this server reads no images: start it with --vision",
        error.PrecisionNotBuilt => "--precision checkpoint: NVIDIA's FP4 and FP8 math is built for Nemotron 3.5 Super VL's NVFP4 checkpoint so far; serve this one with --precision full",
        error.CheckpointMathNeedsBlackwell => "--precision checkpoint: NVIDIA's FP4 math runs on Blackwell (compute capability 12.x); serve with --precision full",
        else => null,
    };
}

fn release(p: *anyopaque) void {
    const own: *Owned = @ptrCast(@alignCast(p));
    if (own.peer != null) {
        own.e.ctx.makeCurrent() catch {};
        own.e.stream.synchronize() catch {};
    }
    own.lanes.deinit();
    if (own.living) |*s| s.deinit();
    if (own.head) |h| h.deinit();
    // The peer first: rank 0's engine holds a pointer to it and its staging.
    own.e.peer = null;
    own.e.tp_link = null;
    if (own.link) |l| l.deinit();
    if (own.peer) |peer| {
        peer.ctx.makeCurrent() catch {};
        peer.deinit();
        own.e.ctx.makeCurrent() catch {};
    }
    own.e.deinit();
    if (own.carve) |c| closeCarveout(own.gpa, c);
    own.gpa.destroy(own);
}

/// TF_CUDA_CARVEOUT=1: the display carveout KV planes go to first (cuda/carveout.zig); null when it is off.
fn openCarveout(gpa: std.mem.Allocator, ctx: *const cuda.Context) !?*cuda.Carveout {
    const bytes = (try cuda.carveout.requested(false, env("TF_CUDA_CARVEOUT"), env("TF_CUDA_CARVEOUT_MIB"))) orelse return null;
    const card: [*:0]const u8 = std.c.getenv("TF_DRM_CARD") orelse cuda.carveout.default_card;
    const c = try gpa.create(cuda.Carveout);
    errdefer gpa.destroy(c);
    c.* = cuda.Carveout.open(ctx.d, card, bytes) catch |err| {
        std.log.err("display carveout of {d} MiB from {s}: {t} (needs nvidia_drm modeset=1, the card's device node and no display in use)", .{ bytes >> 20, card, err });
        return err;
    };
    std.log.info("display carveout: {d} MiB from {s}; KV planes go there first", .{ bytes >> 20, card });
    return c;
}

fn closeCarveout(gpa: std.mem.Allocator, c: *cuda.Carveout) void {
    c.close();
    gpa.destroy(c);
}

fn env(name: [:0]const u8) ?[]const u8 {
    return std.mem.span(std.c.getenv(name) orelse return null);
}

test "spill precision names describe resolved checkpoint math" {
    var e: engine.Engine = undefined;
    var target: Target = .{ .e = &e, .head = null };
    e.c.format = .modelopt;
    e.c.ck = true;
    try std.testing.expectEqualStrings("modelopt/checkpoint", formatName(&target));
    e.c.ck = false;
    try std.testing.expectEqualStrings("modelopt/full", formatName(&target));
    e.c.format = .mlx;
    try std.testing.expectEqualStrings("mlx/full", formatName(&target));
}
