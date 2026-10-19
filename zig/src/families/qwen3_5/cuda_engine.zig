//! Qwen3.5's hybrids on CUDA: qlinear projections, qwen35.cu glue, one captured forward over every stream's window.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const geometry = @import("cuda_geometry.zig");
const weights = @import("cuda_weights.zig");
const caches = @import("cuda_cache.zig");
const routed = @import("cuda_routed.zig");
const checkpoint = @import("cuda_checkpoint.zig");
const modelopt = @import("cuda_modelopt.zig");
const hf = @import("cuda_hf.zig");

pub const Geometry = geometry.Geometry;
pub const Cache = caches.Cache;
pub const tap_rows = caches.tap_rows;
const Seg = caches.Seg;
const DeviceBuffer = cuda.DeviceBuffer;
const head_dim = geometry.head_dim;

/// Rows of one forward at most: a prompt chunk, or a shared round's windows.
pub const max_rows = 128;
/// Streams' windows in one forward at most.
pub const max_segs = 32;
/// Rows of a verify forward at most: each row's logits are kept and drawn.
pub const most_verify = 32;
/// A window's rows at most (exact: one-row steps' bits; qwen35.cuh's MAXW, the rows a stream's log holds).
pub const most_window = 16;
/// Experimental committed HF prefixes retain the decode operators' row arithmetic.
pub const hf_prefill_rows = 16;
const hf_part_bytes = 8 << 20;
pub const ExpertGraphNode = struct {
    shape_key: u32,
    total: u8,
    heads_all: bool,
    layer_local: usize,
    layer_global: usize,
    arm: enum { baseline, n8, split_m },
    grid: [3]u32,
    block: [3]u32,
    shared_bytes: u32,
    abi_count: usize,
    argument_bytes: usize,
    args_sha256: [64]u8,
};
/// Partial attention chunks a row at most (qwen35.cuh's MAXCH): each row's chunk width grows with its own keys.
const max_chunks = 64;

/// Device allocations are 256-byte aligned, as the driver's are.
pub fn aligned(n: usize) usize {
    return std.mem.alignForward(usize, n, 256);
}

/// qwen35.cuh's Geo.
pub const Geo = extern struct { hid: c_int, nk: c_int, nv: c_int, qh: c_int, kvh: c_int, nl: c_int };

const layers = @import("cuda_layers.zig");
pub const Proj = layers.Proj;
pub const Delta = layers.Delta;
pub const Attention = layers.Attention;
pub const Layer = layers.Layer;
pub const Mlp = layers.Mlp;
const Fns = layers.Fns;
pub const Keeps = layers.Keeps;
const Kv8 = layers.Kv8;

/// A window of one stream in a forward: `parents`, for a tree, names each row's parent row (-1: the first row).
pub const Part = struct { cache: *Cache, rows: usize, parents: ?[]const i32 = null, images: ?*const Images = null };

/// An image prompt's rows for its prompt chunks: device feature rows, each prompt row's feature row, its positions.
pub const Images = struct {
    features: u64, // bf16 [image rows, hidden]
    src: []const i32, // [prompt]: a feature row, or -1 for text
    pos: []const i32, // [3][prompt]: t, h and w rotary positions (get_rope_index)
};

test "GDN layouts have distinct graph keys for identical forward geometry" {
    const baseline: Engine.Shape = .{ .total = 1, .nseg = 1, .record = false, .prompt = true, .taps = false };
    var candidate = baseline;
    candidate.gdn_layout = .key_value;
    try std.testing.expectEqual(@as(u32, 1 << 19), baseline.key() ^ candidate.key());
}

test "expert up choices have distinct graph keys independent of GDN layout" {
    const baseline: Engine.Shape = .{ .total = 1, .nseg = 1, .record = false, .prompt = true, .taps = false, .gdn_layout = .key_value };
    var candidate = baseline;
    candidate.expert_up_n8 = true;
    try std.testing.expectEqual(@as(u32, 1 << 20), baseline.key() ^ candidate.key());
    candidate.gdn_layout = .value_key;
    try std.testing.expectEqual(@as(u32, (1 << 20) | (1 << 19)), baseline.key() ^ candidate.key());
}

test "N8 up rejects MLX before CUDA or graph lookup" {
    var e: Engine = undefined;
    e.numerics = .mlx_sanitized;
    const shape: Engine.Shape = .{ .total = 1, .nseg = 1, .record = false, .prompt = true, .taps = false, .expert_up_n8 = true };
    try std.testing.expectError(error.UnsupportedQwenExpertUp, e.execute(shape));
    var cache: Cache = undefined;
    cache.gdn_layout = .value_key;
    e.expert_up_n8 = true;
    try std.testing.expectError(error.UnsupportedQwenExpertUp, e.prepare(&.{.{ .cache = &cache, .rows = 1 }}, &.{0}, .last));
}

test "mixed GDN layouts fail before settling caches or touching CUDA" {
    var e: Engine = undefined;
    var baseline: Cache = undefined;
    baseline.gdn_layout = .value_key;
    var candidate: Cache = undefined;
    candidate.gdn_layout = .key_value;
    const parts = [_]Part{ .{ .cache = &baseline, .rows = 1 }, .{ .cache = &candidate, .rows = 1 } };
    try std.testing.expectError(error.MixedQwenGdnLayouts, e.prepare(&parts, &.{ 0, 0 }, .last));
}

/// What a forward keeps: a prompt chunk its last row's logits, a verify window every row's draw or logits and log.
pub const Keep = union(enum) { last, draws: []const Draw, logits, prompt_logits };

fn validateCombined(numerics: checkpoint.Numerics, n8: bool, split: bool, reuse: bool) !void {
    if (split and (numerics != .hf_modelopt or !n8)) return error.UnsupportedQwenExpertUp;
    if (reuse and numerics != .hf_modelopt) return error.UnsupportedQwenQuantReuse;
}

fn records(keep: Keep) bool {
    return keep == .draws or keep == .logits;
}

fn validateHfWindow(numerics: checkpoint.Numerics, bounded: bool, parts: []const Part, ids: usize, keep: Keep) !void {
    if (keep == .prompt_logits and numerics != .hf_modelopt) return error.UnsupportedQwenWindow;
    if (numerics != .hf_modelopt) return;
    const limit: usize = if (bounded) hf_prefill_rows else 1;
    if (parts.len != 1 or ids == 0 or ids > limit or parts[0].rows != ids or
        (keep != .last and !(bounded and keep == .prompt_logits))) return error.UnsupportedQwenWindow;
    if (parts[0].parents != null or parts[0].images != null) return error.UnsupportedQwenWindow;
}

fn hfInputWidth(g: Geometry) usize {
    return @max(g.convWidth(), g.valueWidth(), g.attnWidth(), g.hidden, g.intermediate);
}

fn chainView(base: Seg, row: usize) Seg {
    var view = base;
    view.row0 = base.row0 + @as(c_int, @intCast(row));
    view.len = base.len + @as(c_int, @intCast(row));
    view.rows = 1;
    return view;
}

fn validateChainViews(base: Seg, views: []const Seg) !void {
    if (base.row0 != 0 or base.rows != views.len or views.len == 0 or views.len > hf_prefill_rows) return error.QwenGdnTableMismatch;
    for (views, 0..) |view, row| {
        const want = chainView(base, row);
        inline for (@typeInfo(Seg).@"struct".field_names) |name| {
            if (@field(view, name) != @field(want, name)) return error.QwenGdnTableMismatch;
        }
    }
}

const GdnRoster = struct {
    layers: usize,
    rows: usize,
    count: usize = 0,

    fn accept(r: *GdnRoster, layer: usize, row: usize) !void {
        if (r.rows == 0 or r.count >= r.layers * r.rows or layer != r.count / r.rows or row != r.count % r.rows) return error.QwenGdnGraphLayerMismatch;
        r.count += 1;
    }

    fn finish(r: GdnRoster) !void {
        if (r.count != r.layers * r.rows) return error.QwenGdnGraphWorkCountMismatch;
    }
};

/// A kept row's draw on the device: sample.cu's keyed rule at `position`, or greedy without one.
pub const Draw = struct { rule: ?cuda.sample.Rule, position: u64 };

pub const Phase = enum { begin, embedding, input_norm, mixer_inputs, gdn_pre, conv_commit, gdn_chain, gdn_post, attention, mixer_output, post_norm, route, expert_plan, expert_up, expert_down, shared, combine, dense, head };
pub const Observer = struct { ptr: *anyopaque, mark: *const fn (*anyopaque, Phase, usize) anyerror!void };

pub const Engine = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    g: Geometry,
    placement: checkpoint.Placement,
    numerics: checkpoint.Numerics,
    modelopt: ?modelopt.Bindings = null,
    captured: bool = true,
    graph_nodes: usize = 0,
    graph_launches: usize = 0,
    expert_up_n8: bool = false,
    expert_up_split_m: bool = false,
    reuse_fp8_quant: bool = false,
    bounded_hf_prefill: bool = false,
    verify_expert_graphs: bool = false,
    expert_graph_nodes_baseline: usize = 0,
    expert_graph_nodes_n8: usize = 0,
    expert_graph_nodes_split_m: usize = 0,
    expert_graph_roster: std.ArrayList(ExpertGraphNode) = .empty,
    quant_graph_nodes_baseline: usize = 0,
    quant_graph_nodes_reuse: usize = 0,
    quant_graphs_verified: usize = 0,
    verify_gdn_graphs: bool = false,
    gdn_graph_nodes_baseline: usize = 0,
    gdn_graph_nodes_candidate: usize = 0,
    observer: ?Observer = null,
    observed_layer: usize = 0,
    geo: Geo,
    stream: cuda.Stream,
    mods: [16]cuda.Module,
    kv8: bool, // attention caches as e4m3 rows with a scale each
    fp8: Kv8,
    draws: cuda.sample.Draws,
    affine: cuda.qlinear.Affine4Kernels,
    wide: cuda.qlinear.AffineNKernels, // the tiles of every width but 4, and their packs
    experts: cuda.experts.Affine4Experts,
    experts3: cuda.experts3.Affine3Experts,
    plan: cuda.grouped.Router,
    moe: cuda.moe.Fns,
    fns: Fns,
    keeps: Keeps,
    /// Every weight and scratch buffer in one allocation: WDDM's launch cost grows with the allocation count.
    arena: DeviceBuffer,
    arena_at: usize = 0,
    /// Rows a verify forward takes at most, and a window's rows at most.
    verify_rows: usize,
    window_rows: usize,
    // weights
    embed_w: u64,
    embed_s: u64,
    embed_b: u64,
    embed_bits: u8, // the embedding's rows' width: 4 or 8
    head: Proj,
    final_norm: u64,
    layers: []Layer,
    // scratch, max_rows rows each
    h: u64,
    x: u64,
    r: u64,
    xs: u64,
    qkv: u64,
    z: u64,
    a: u64,
    b: u64,
    q: u64,
    k: u64,
    v: u64,
    g_: u64,
    beta: u64,
    y: u64,
    mix: u64,
    qraw: u64,
    kraw: u64,
    vraw: u64,
    qn: u64,
    kn: u64,
    gate: u64,
    up: u64,
    act: u64,
    pm: u64,
    pl: u64,
    po: u64,
    logits: u64,
    routed: routed.Scratch = .{}, // qwen3_5_moe's routed rows
    // per-forward tables (ids, segments, rows' segment, parent and ancestors, the draws' rules) staged in pinned memory
    tables: DeviceBuffer,
    staged: cuda.HostBuffer,
    host_logits: cuda.HostBuffer,
    /// The kept rows' device draws, when forward was given them.
    tokens: [most_verify]u32 = undefined,
    /// The captured forwards by shape (`Shape.key`): the launches from the embedding to the head.
    graphs: std.AutoHashMapUnmanaged(u32, cuda.graph.Exec) = .empty,
    /// A drafter's tapped layers (`setTaps`): each row's residual after them lands in its stream's tap ring.
    taps: [8]u32 = undefined,
    n_taps: usize = 0,
    /// During an image prompt chunk's forward: its feature rows (0: none); the chunk's src and positions are staged.
    features: u64 = 0,

    const ids_at = 0;
    const segs_at = max_rows * 4;
    const rowseg_at = segs_at + max_segs * @sizeOf(Seg);
    const rowpar_at = rowseg_at + max_rows * 4;
    const rowanc_at = rowpar_at + max_rows * 4;
    const rules_at = rowanc_at + max_rows * 4;
    const meta_at = rules_at + most_verify * @sizeOf(cuda.sample.Rule);
    const tokens_at = meta_at + most_verify * 4;
    pub const params_at = tokens_at + most_verify * 4; // two unused words, the rows, the windows, record
    const chain_segs_at = aligned(params_at + 8 * 4);
    const src_at = chain_segs_at + hf_prefill_rows * @sizeOf(Seg); // image feature rows follow the immutable chain views
    const mpos_at = src_at + max_rows * 4; // and their t, h and w positions, max_rows apart (qwen35.cuh's MAXR)
    const tables_len = mpos_at + 3 * max_rows * 4;

    pub fn init(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, kv8: bool) !*Engine {
        const g = try Geometry.read(gpa, io, dir);
        return initPlaced(gpa, io, ctx, dir, .{ .geometry = g, .encoding = .mlx }, .{ .first = 0, .end = g.layers, .total = g.layers }, kv8);
    }

    pub fn initPlaced(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, admitted: checkpoint.Admitted, placement: checkpoint.Placement, kv8: bool) !*Engine {
        const g = try placement.localGeometry(admitted.geometry);
        if (admitted.numerics() == .hf_modelopt and kv8) return error.UnsupportedQwenKv;
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var ck = try core.Checkpoint.openModel(gpa, io, dir);
        defer ck.close();
        const allocation = try weights.plan(&ck, admitted, placement);
        const d = ctx.d;
        const e = try gpa.create(Engine);
        errdefer gpa.destroy(e);
        e.* = undefined;
        e.gpa = gpa;
        e.d = d;
        e.g = g;
        e.placement = placement;
        e.numerics = admitted.numerics();
        e.modelopt = null;
        e.captured = true;
        e.graph_nodes = 0;
        e.graph_launches = 0;
        e.expert_up_n8 = e.numerics == .hf_modelopt;
        e.expert_up_split_m = false;
        e.reuse_fp8_quant = false;
        e.bounded_hf_prefill = false;
        e.verify_expert_graphs = false;
        e.expert_graph_nodes_baseline = 0;
        e.expert_graph_nodes_n8 = 0;
        e.expert_graph_nodes_split_m = 0;
        e.expert_graph_roster = .empty;
        e.quant_graph_nodes_baseline = 0;
        e.quant_graph_nodes_reuse = 0;
        e.quant_graphs_verified = 0;
        e.verify_gdn_graphs = false;
        e.gdn_graph_nodes_baseline = 0;
        e.gdn_graph_nodes_candidate = 0;
        e.observer = null;
        e.observed_layer = 0;
        e.geo = .{ .hid = @intCast(g.hidden), .nk = @intCast(g.k_heads), .nv = @intCast(g.v_heads), .qh = @intCast(g.q_heads), .kvh = @intCast(g.kv_heads), .nl = @intCast(g.deltas()) };
        e.verify_rows = most_verify;
        e.kv8 = kv8;
        e.window_rows = most_window;
        e.arena_at = 0;
        e.graphs = .empty;
        e.n_taps = 0;
        e.features = 0;
        e.stream = try cuda.Stream.init(d, true);
        errdefer e.stream.deinit();
        e.arena = try DeviceBuffer.alloc(d, allocation.weights_bytes + try e.scratch(.measure));
        errdefer e.arena.free();
        try e.arena.fill8(0, null); // touched once now: the driver commits memory on first touch, slowly
        const kk = cuda.kernels;
        const images = [_][]const u8{ kk.qmm_group, kk.qmm_prefill, kk.lane_gemv, kk.affine4_pack, kk.qwen35, kk.sample, kk.qwen35_keep, kk.affine8, kk.affine_prefill, kk.qwen35_kv8, kk.experts, kk.experts_prefill, kk.experts_pack, kk.moe, kk.experts3, kk.experts3_prefill };
        var loaded: usize = 0;
        errdefer for (e.mods[0..loaded]) |*m| m.unload();
        for (images, 0..) |img, i| {
            e.mods[i] = try cuda.Module.load(d, img);
            loaded += 1;
        }
        const sms: usize = @intCast(try ctx.attribute(.multiprocessor_count));
        const gb10 = try ctx.attribute(.compute_capability_major) == 12 and try ctx.attribute(.compute_capability_minor) == 1;
        e.affine = try cuda.qlinear.Affine4Kernels.resolve(d, e.mods[0], e.mods[1], e.mods[2], e.mods[3], sms, gb10);
        errdefer e.affine.deinit();
        e.wide = try cuda.qlinear.AffineNKernels.resolve(d, e.mods[7], e.mods[8], e.mods[3], sms, gb10);
        errdefer e.wide.deinit();
        e.draws = try cuda.sample.Draws.resolve(e.mods[5]);
        e.experts = try cuda.experts.Affine4Experts.resolve(e.mods[10], e.mods[11], e.mods[12], sms);
        e.experts3 = try cuda.experts3.Affine3Experts.resolve(e.mods[14], e.mods[15], sms);
        e.plan = try cuda.grouped.Router.resolve(e.mods[10]);
        e.moe = try cuda.moe.Fns.resolve(e.mods[13]);
        inline for (@typeInfo(Fns).@"struct".field_names) |name| @field(e.fns, name) = try e.mods[4].function("tf_q35_" ++ name);
        inline for (@typeInfo(Keeps).@"struct".field_names) |name| @field(e.keeps, name) = try e.mods[6].function("tf_q35k_" ++ name);
        e.fp8 = .{ .rope = try e.mods[9].function("tf_q35_rope8"), .attn_partial = try e.mods[9].function("tf_q35_attn_partial8"), .compact = try e.mods[9].function("tf_q35k_compact8") };
        if (e.numerics == .hf_modelopt) e.modelopt = try modelopt.Bindings.load(ctx, &e.fns);
        errdefer if (e.modelopt) |*m| m.unload();
        e.layers = try gpa.alloc(Layer, g.layers);
        errdefer gpa.free(e.layers);
        {
            // each projection as stored, packed from here; freed before the server counts what streams can use
            var stage = try DeviceBuffer.alloc(d, allocation.staging_bytes);
            defer stage.free();
            try weights.loadPlaced(e, &ck, &stage, admitted, placement);
        }
        if (e.numerics == .hf_modelopt) try e.validateHfWorkspaces();
        _ = try e.scratch(.allocate);
        e.tables = try DeviceBuffer.alloc(d, tables_len);
        errdefer e.tables.free();
        e.staged = try cuda.HostBuffer.alloc(d, tables_len);
        errdefer e.staged.free();
        e.host_logits = try cuda.HostBuffer.alloc(d, most_verify * g.vocab * 2);
        errdefer e.host_logits.free();
        if (e.numerics == .mlx_sanitized) try e.warm(gpa);
        return e;
    }

    /// Every path a request takes, run once now: WDDM stalls a process's first forwards for seconds (a 20 s launch).
    fn warm(e: *Engine, gpa: std.mem.Allocator) !void {
        var warm_cache = try Cache.init(gpa, e, 64);
        defer warm_cache.deinit(gpa);
        const ids: [32]u32 = @splat(0);
        const greedy: Draw = .{ .rule = null, .position = 33 };
        try e.forward(&.{.{ .cache = &warm_cache, .rows = ids.len }}, &ids, .last); // a prompt chunk: the prompt GEMMs
        try e.forward(&.{.{ .cache = &warm_cache, .rows = 1 }}, ids[0..1], .{ .draws = &.{greedy} });
        try e.forward(&.{.{ .cache = &warm_cache, .rows = 1 }}, ids[0..1], .{ .draws = &.{.{ .rule = .{ .seed = 1, .inv_t = 1.0 / 0.7, .top_p = 0.95, .near = 20.0, .min_log = -std.math.inf(f32), .top_k = 20 }, .position = 34 }} });
        // a tree window and a branch of it kept: the keep kernels, and the forward's branch path
        try e.forward(&.{.{ .cache = &warm_cache, .rows = 3, .parents = &.{ -1, 0, 0 } }}, ids[0..3], .{ .draws = &.{ greedy, greedy, greedy } });
        try warm_cache.keep(e, &.{ 0, 2 });
        try e.stream.synchronize();
    }

    pub fn deinit(e: *Engine) void {
        e.expert_graph_roster.deinit(e.gpa);
        var it = e.graphs.valueIterator();
        while (it.next()) |x| x.deinit();
        e.graphs.deinit(e.gpa);
        e.host_logits.free();
        e.staged.free();
        e.tables.free();
        e.arena.free();
        e.gpa.free(e.layers);
        if (e.modelopt) |*m| m.unload();
        e.affine.deinit();
        e.wide.deinit();
        for (&e.mods) |*m| m.unload();
        e.stream.deinit();
        e.gpa.destroy(e);
    }

    /// A stream's device bytes at a `cap`-token window: recurrences, convolution tails, keys and values, taps, log.
    pub fn streamBytes(e: *const Engine, cap: usize) usize {
        return e.g.stateBytes() + e.g.kvBytes(cap, e.kv8) + e.tapBytes() + e.g.logBytes(if (e.numerics == .hf_modelopt) 1 else most_window);
    }

    /// A stream's tap ring: tap_rows rows of the tapped layers' residuals.
    pub fn tapBytes(e: *const Engine) usize {
        return tap_rows * e.n_taps * e.g.hidden * 2;
    }

    /// The layers a drafter taps, told before the streams' caches exist; each forward then fills their rings.
    pub fn setTaps(e: *Engine, taps: []const u32) !void {
        if (e.numerics == .hf_modelopt) return error.UnsupportedQwenTaps;
        if (taps.len > e.taps.len) return error.TooManyTaps;
        for (taps) |t| if (t >= e.g.layers) return error.BadTap;
        @memcpy(e.taps[0..taps.len], taps);
        e.n_taps = taps.len;
    }

    /// The next `bytes` of the arena, 256-byte aligned.
    pub fn alloc(e: *Engine, bytes: usize) !u64 {
        const at = aligned(e.arena_at);
        if (at + bytes > e.arena.len) return error.ArenaExhausted;
        e.arena_at = at + bytes;
        return e.arena.ptr + at;
    }

    pub fn scratchBytes(admitted: checkpoint.Admitted, placement: checkpoint.Placement) !usize {
        var e: Engine = undefined;
        e.g = try placement.localGeometry(admitted.geometry);
        e.numerics = admitted.numerics();
        return aligned(tables_len) + try e.scratch(.measure);
    }

    /// The scratch rows' bytes (`measure`: summed, nothing taken), or the rows themselves from the arena.
    fn scratch(e: *Engine, mode: enum { measure, allocate }) !usize {
        var total: usize = 0;
        const Take = struct {
            e: *Engine,
            mode: @TypeOf(mode),
            total: *usize,
            pub fn alloc(t: @This(), bytes: usize) !u64 {
                t.total.* += aligned(bytes);
                return if (t.mode == .measure) 0 else try t.e.alloc(bytes);
            }
        };
        const t = Take{ .e = e, .mode = mode, .total = &total };
        const g = e.g;
        const R: usize = if (e.numerics == .hf_modelopt) hf_prefill_rows else max_rows;
        const fp: usize = if (e.numerics == .hf_modelopt) 4 else 2;
        const wide = @max(g.valueWidth(), g.attnWidth(), g.hidden); // y and mix: either mixer's rows
        inline for (.{ "h", "x", "r" }) |name| @field(e, name) = try t.alloc(R * g.hidden * 2);
        e.y = try t.alloc(R * wide * fp);
        inline for (.{ "mix", "z", "v" }) |name| @field(e, name) = try t.alloc(R * wide * 2);
        inline for (.{ "q", "k" }) |name| @field(e, name) = try t.alloc(R * g.k_heads * geometry.dk * fp);
        e.qkv = try t.alloc(R * g.convWidth() * 2);
        inline for (.{ "gate", "up", "act" }) |name| @field(e, name) = try t.alloc(R * g.intermediate * 2);
        e.xs = try t.alloc(R * (@max(g.intermediate, wide) / 64) * 4);
        inline for (.{ "a", "b", "beta" }) |name| @field(e, name) = try t.alloc(R * g.v_heads * 2);
        e.g_ = try t.alloc(R * g.v_heads * 4);
        e.qraw = try t.alloc(R * 2 * g.attnWidth() * 2);
        e.qn = try t.alloc(R * g.attnWidth() * 2);
        inline for (.{ "kraw", "vraw", "kn" }) |name| @field(e, name) = try t.alloc(R * g.kv_heads * head_dim * 2);
        e.pm = try t.alloc(R * g.q_heads * max_chunks * 4);
        e.pl = try t.alloc(R * g.q_heads * max_chunks * 4);
        e.po = try t.alloc(R * g.q_heads * max_chunks * head_dim * 4);
        e.logits = try t.alloc(most_verify * g.vocab * 2);
        if (e.numerics == .hf_modelopt) {
            const rows = try t.alloc(R * hfInputWidth(g));
            const part = try t.alloc(hf_part_bytes);
            if (mode == .allocate) {
                e.modelopt.?.rows = rows;
                e.modelopt.?.part = part;
            }
        }
        if (g.moe()) e.routed = try routed.Scratch.carve(t, g, R);
        return total;
    }

    fn validateHfProjection(e: *const Engine, p: Proj) !void {
        if (p != .linear or p.inputs() > hfInputWidth(e.g)) return error.QwenLinearWorkspace;
        const w = switch (p.linear) {
            .nvfp4, .fp8, .bf16 => |w| w,
            else => return error.QwenLinearWorkspace,
        };
        const m = e.modelopt.?;
        var bytes = m.lane.partBytes(hf_prefill_rows, w);
        if (p.linear == .fp8 and w.act != 0) {
            const sk = cuda.qmmf.splitK(w.n, w.k);
            if (sk > 1 and !m.ck.clusters) bytes = @max(bytes, hf_prefill_rows * @as(usize, w.n) * sk * 4);
        }
        if (bytes > hf_part_bytes) return error.QwenLinearWorkspace;
    }

    fn validateHfWorkspaces(e: *const Engine) !void {
        for (e.layers) |layer| {
            switch (layer.mixer) {
                .delta => |m| inline for (.{ "qkv", "z", "a", "b", "out" }) |name| try e.validateHfProjection(@field(m, name)),
                .attention => |m| inline for (.{ "q", "k", "v", "o" }) |name| try e.validateHfProjection(@field(m, name)),
            }
            switch (layer.mlp) {
                .dense => |m| inline for (.{ "gate", "up", "down" }) |name| try e.validateHfProjection(@field(m, name)),
                .moe => |m| if (m.shared == .projected) {
                    inline for (.{ "gate", "up", "down" }) |name| try e.validateHfProjection(@field(m.shared.projected, name));
                },
            }
        }
        if (e.placement.ownsHead()) try e.validateHfProjection(e.head);
    }

    pub fn run(e: *Engine, f: cuda.Function, grid: [3]usize, block: [3]usize, args: anytype) !void {
        var a: cuda.Args = .{};
        inline for (args) |v| a.add(v);
        const cfg: cuda.Config = .{ .grid = .{ .x = @intCast(grid[0]), .y = @intCast(grid[1]), .z = @intCast(grid[2]) }, .block = .{ .x = @intCast(block[0]), .y = @intCast(block[1]), .z = @intCast(block[2]) } };
        try cuda.launch.launch(f, cfg, e.stream, &a);
    }

    /// out (rows, n) = x (rows, k) through `l`: 16-row decode tiles (a row's bits batch-independent) or a prompt GEMM.
    pub fn proj(e: *Engine, l: Proj, x: u64, out: u64, rows: usize, prompt: bool) !void {
        return e.projSums(l, x, e.xs, out, rows, prompt, e.tables.ptr + params_at);
    }

    /// `proj` with group sums `xs` and device counts `params` (0: all real); prompts past 16 rows take a prompt GEMM.
    pub fn projSums(e: *Engine, l: Proj, x: u64, xs: u64, out: u64, rows: usize, prompt: bool, params: u64) !void {
        if (l == .linear) {
            const m = &(e.modelopt orelse return error.QwenFormatNotLoaded);
            return m.linear().decode(e.stream, l.linear, .{ .x = x, .part = m.part }, out, rows);
        }
        const n_out = l.outputs();
        const k = l.inputs();
        if (prompt and rows > 16) return switch (l) {
            .a4 => |q| e.affine.prompt(e.stream, x, q, out, rows),
            .an => |q| e.wide.prompt(e.stream, x, q, out, rows),
            .linear => unreachable,
        };
        try e.run(e.fns.sums, .{ rows, 1, 1 }, .{ 32, 1, 1 }, .{ x, xs, @as(c_int, @intCast(k)), params });
        var at: usize = 0;
        while (at < rows) : (at += 16) {
            const n = @min(16, rows - at);
            const xa = x + at * k * 2;
            const sa = xs + at * (k / 64) * 4;
            const oa = out + at * n_out * 2;
            switch (l) {
                .a4 => |q| try e.affine.decode(e.stream, xa, sa, q, oa, n),
                .an => |q| try e.wide.decode(e.stream, xa, sa, q, oa, n),
                .linear => unreachable,
            }
        }
    }

    /// Token rows of the embedding into `out` (rows, hidden) bf16, at the width its words are served.
    pub fn embed(e: *Engine, ids: u64, out: u64, rows: usize, params: u64) !void {
        const f = if (e.embed_bits == 8) e.fns.embed8 else e.fns.embed;
        try e.run(f, .{ rows, 1, 1 }, .{ 256, 1, 1 }, .{ ids, e.embed_w, e.embed_s, e.embed_b, out, @as(c_int, @intCast(e.g.hidden)), params });
    }

    fn norm(e: *Engine, residual: bool, w: u64, rows: usize) !void {
        try e.run(e.fns.norm, .{ rows, 1, 1 }, .{ if (e.numerics == .hf_modelopt) (hf.Launches{}).norm else 256, 1, 1 }, .{ e.h, if (residual) e.r else @as(u64, 0), w, e.x, e.g.eps, @as(c_int, @intCast(e.g.hidden)), e.tables.ptr + params_at });
    }

    /// A chain row's rowanc: its index in its window, as a prompt chunk's rows outnumber the bits.
    const chain_row: u32 = 1 << 31;

    /// The row tables of `parts`: each row's segment, parent row (-1: a window's first) and ancestors-and-self bits.
    fn rowTables(st: []u8, parts: []const Part) !void {
        const rowseg = std.mem.bytesAsSlice(c_int, st[rowseg_at..][0 .. max_rows * 4]);
        const rowpar = std.mem.bytesAsSlice(c_int, st[rowpar_at..][0 .. max_rows * 4]);
        const rowanc = std.mem.bytesAsSlice(u32, st[rowanc_at..][0 .. max_rows * 4]);
        var row0: usize = 0;
        for (parts, 0..) |p, i| {
            if (p.parents) |ps| if (ps.len != p.rows or p.rows > most_window or ps[0] != -1) return error.InvalidQwenParents;
            for (0..p.rows) |r| {
                const parent: i32 = if (p.parents) |ps| ps[r] else if (r == 0) -1 else @intCast(r - 1);
                if (parent >= @as(i32, @intCast(r)) or parent < -1) return error.InvalidQwenParents;
                rowseg[row0 + r] = @intCast(i);
                rowpar[row0 + r] = if (parent < 0) -1 else @intCast(row0 + @as(usize, @intCast(parent)));
                rowanc[row0 + r] = if (p.parents == null) chain_row | @as(u32, @intCast(r)) else (@as(u32, 1) << @intCast(r)) | (if (parent < 0) 0 else rowanc[row0 + @as(usize, @intCast(parent))]);
            }
            row0 += p.rows;
        }
    }

    /// One forward over `parts`' windows, `ids` in order; `keep` names the kept rows; an unkept verify is kept first.
    pub fn forward(e: *Engine, parts: []const Part, ids: []const u32, keep: Keep) !void {
        if (!e.placement.ownsEmbedding() or !e.placement.ownsHead()) return error.QwenRequiresPipeline;
        const shape = try e.prepare(parts, ids, keep);
        try e.execute(shape);
        try e.finish(parts, keep, shape);
        commit(parts, shape.record);
    }

    pub fn prepare(e: *Engine, parts: []const Part, ids: []const u32, keep: Keep) !Shape {
        if (parts.len == 0 or parts.len > max_segs) return error.InvalidQwenSegments;
        const gdn_layout = parts[0].cache.gdn_layout;
        for (parts[1..]) |p| if (p.cache.gdn_layout != gdn_layout) return error.MixedQwenGdnLayouts;
        try gdn_layout.validate(e.numerics);
        if (e.expert_up_n8 and e.numerics != .hf_modelopt) return error.UnsupportedQwenExpertUp;
        try validateCombined(e.numerics, e.expert_up_n8, e.expert_up_split_m, e.reuse_fp8_quant);
        const g = e.g;
        try validateHfWindow(e.numerics, e.bounded_hf_prefill, parts, ids.len, keep);
        var total: usize = 0;
        for (parts) |p| {
            try p.cache.settle(e);
            if (p.rows == 0 or p.cache.len + p.rows > p.cache.cap) return error.QwenContextFull;
            total += p.rows;
        }
        if (parts.len == 0 or parts.len > max_segs or total != ids.len or total > max_rows) return error.InvalidQwenSegments;
        const record = records(keep);
        if (record and total > e.verify_rows) return error.InvalidQwenSegments;
        if (!record and parts.len != 1) return error.InvalidQwenSegments;
        if (keep == .draws and keep.draws.len != total) return error.InvalidQwenSegments;
        for (ids) |id| if (id >= g.vocab) return error.InvalidQwenToken;
        for (parts) |p| if (p.parents != null and !record) return error.InvalidQwenParents;
        for (parts) |p| if (p.images != null and record) return error.InvalidQwenImages;
        // the prompt GEMM only for prompt chunks: a window's rows keep the bits of one-row steps
        const prompt = !record;
        // the tables
        const st = e.staged.bytes;
        @memcpy(std.mem.bytesAsSlice(u32, st[ids_at..][0 .. total * 4]), ids);
        const segs = std.mem.bytesAsSlice(Seg, st[segs_at..][0 .. parts.len * @sizeOf(Seg)]);
        try rowTables(st, parts);
        var row0: usize = 0;
        for (parts, 0..) |p, i| {
            segs[i] = p.cache.seg(e, p.cache.len, row0, p.rows);
            row0 += p.rows;
        }
        if (e.numerics == .hf_modelopt and total > 1) {
            const views = std.mem.bytesAsSlice(Seg, st[chain_segs_at..][0 .. total * @sizeOf(Seg)]);
            for (views, 0..) |*view, row| view.* = chainView(segs[0], row);
            try validateChainViews(segs[0], views);
        }
        if (keep == .draws) {
            const rules = std.mem.bytesAsSlice(cuda.sample.Rule, st[rules_at..][0 .. total * @sizeOf(cuda.sample.Rule)]);
            const meta = std.mem.bytesAsSlice(c_int, st[meta_at..][0 .. total * 4]);
            for (keep.draws, rules, meta) |p, *r, *m| {
                if (p.rule) |x| r.* = x;
                m.* = @intCast(p.position -% 1); // sample.cu keys row r at meta + r + 1
            }
        }
        // words 0 and 1 are reserved (each row's attention chunks follow its own keys: qwen35.cu's chunk_keys)
        const params = std.mem.bytesAsSlice(c_int, st[params_at..][0..32]);
        params[0] = 0;
        params[1] = 0;
        params[2] = @intCast(total);
        params[3] = @intCast(parts.len);
        params[4] = @intFromBool(record);
        // an image prompt's chunk (one part, its rows at prompt positions len..): feature rows and t, h, w positions
        const images: ?*const Images = if (record) null else parts[0].images;
        if (images) |im| {
            const p0 = parts[0].cache.len;
            if (p0 + total > im.src.len or im.pos.len != 3 * im.src.len) return error.InvalidQwenImages;
            const n = im.src.len;
            @memcpy(std.mem.bytesAsSlice(i32, st[src_at..][0 .. total * 4]), im.src[p0..][0..total]);
            const mpos = std.mem.bytesAsSlice(i32, st[mpos_at..][0 .. 3 * max_rows * 4]);
            for (0..3) |a| @memcpy(mpos[a * max_rows ..][0..total], im.pos[a * n + p0 ..][0..total]);
        }
        try e.tables.uploadAsync(0, st[0..if (images != null) tables_len else src_at], e.stream.handle); // text: as before
        // a captured forward covers a row bucket and every window; kernels read the true counts; a lone step is exact
        const lone = total == 1 and parts.len == 1;
        const bucket: usize = if (lone) 1 else if (total <= 16) 16 else most_verify;
        const windows: usize = if (lone) 1 else max_segs;
        const shape: Shape = .{ .total = @intCast(if (prompt) total else bucket), .nseg = @intCast(if (prompt) parts.len else windows), .record = record, .prompt = prompt, .taps = e.n_taps > 0, .gdn_layout = gdn_layout, .expert_up_n8 = e.expert_up_n8, .heads_all = keep == .prompt_logits, .expert_up_split_m = e.expert_up_split_m, .reuse_fp8_quant = e.reuse_fp8_quant };
        e.features = if (images) |im| im.features else 0;
        return shape;
    }

    pub fn execute(e: *Engine, shape: Shape) !void {
        try validateCombined(e.numerics, shape.expert_up_n8, shape.expert_up_split_m, shape.reuse_fp8_quant);
        if (shape.expert_up_n8 and (e.numerics != .hf_modelopt or (shape.prompt and shape.total > most_window))) return error.UnsupportedQwenExpertUp;
        if (shape.heads_all and e.numerics != .hf_modelopt) return error.UnsupportedQwenWindow;
        if (e.numerics == .hf_modelopt and (shape.total == 0 or shape.total > hf_prefill_rows or shape.nseg != 1 or shape.record or !shape.prompt or shape.taps or
            (!e.bounded_hf_prefill and (shape.total != 1 or shape.heads_all)))) return error.UnsupportedQwenWindow;
        if (e.observer != null and e.captured) return error.QwenObserverRequiresEager;
        try shape.gdn_layout.validate(e.numerics);
        if (e.verify_gdn_graphs and e.numerics == .hf_modelopt and shape.total > 1) try e.verifyHfChainTables(shape);
        if (shape.prompt and e.numerics == .mlx_sanitized or !e.captured) {
            // uncaptured: a prompt chunk's GEMMs dwarf its launches, and each length's graph would pay the stall
            try e.body(shape);
        } else {
            const gop = try e.graphs.getOrPut(e.gpa, shape.key());
            if (!gop.found_existing) {
                errdefer _ = e.graphs.remove(shape.key());
                // captured, nothing runs: the replay below is this shape's first run
                try cuda.graph.beginCapture(e.stream, .thread_local);
                e.body(shape) catch |err| {
                    var broken = cuda.graph.endCapture(e.stream) catch return err;
                    broken.deinit();
                    return err;
                };
                var captured = try cuda.graph.endCapture(e.stream);
                defer captured.deinit();
                const nodes = try captured.nodeCount();
                const gdn_nodes = if (e.verify_gdn_graphs) try e.verifyGdnGraph(captured, shape, nodes) else 0;
                const expert_nodes = if (e.verify_expert_graphs) try e.verifyExpertGraph(captured, shape, nodes) else 0;
                const quant_nodes = if (e.verify_expert_graphs) try e.verifyQuantGraph(captured, shape, nodes) else 0;
                gop.value_ptr.* = try captured.instantiate();
                if (e.verify_expert_graphs) {
                    e.quant_graphs_verified += 1;
                    if (shape.reuse_fp8_quant) e.quant_graph_nodes_reuse += quant_nodes else e.quant_graph_nodes_baseline += quant_nodes;
                }
                if (shape.expert_up_split_m) {
                    e.expert_graph_nodes_split_m += expert_nodes;
                } else if (shape.expert_up_n8) {
                    e.expert_graph_nodes_n8 += expert_nodes;
                } else e.expert_graph_nodes_baseline += expert_nodes;
                e.graph_nodes += nodes;
                switch (shape.gdn_layout) {
                    .value_key => e.gdn_graph_nodes_baseline += gdn_nodes,
                    .key_value => e.gdn_graph_nodes_candidate += gdn_nodes,
                }
            }
            try gop.value_ptr.launchOn(e.stream);
            e.graph_launches += 1;
        }
    }

    fn verifyHfChainTables(e: *const Engine, shape: Shape) !void {
        try e.stream.synchronize();
        var base: Seg = undefined;
        try e.tables.download(segs_at, std.mem.asBytes(&base));
        const staged = std.mem.bytesAsSlice(Seg, e.staged.bytes[segs_at..][0..@sizeOf(Seg)])[0];
        inline for (@typeInfo(Seg).@"struct".field_names) |name| {
            if (@field(base, name) != @field(staged, name)) return error.QwenGdnTableMismatch;
        }
        var views: [hf_prefill_rows]Seg = undefined;
        try e.tables.download(chain_segs_at, std.mem.sliceAsBytes(views[0..shape.total]));
        try validateChainViews(base, views[0..shape.total]);
        var params: [8]c_int = undefined;
        try e.tables.download(params_at, std.mem.asBytes(&params));
        if (params[2] != shape.total or params[3] != 1 or params[4] != 0) return error.QwenGdnTableMismatch;
        var parents: [hf_prefill_rows]c_int = undefined;
        var segments: [hf_prefill_rows]c_int = undefined;
        var ancestors: [hf_prefill_rows]u32 = undefined;
        try e.tables.download(rowpar_at, std.mem.sliceAsBytes(parents[0..shape.total]));
        try e.tables.download(rowseg_at, std.mem.sliceAsBytes(segments[0..shape.total]));
        try e.tables.download(rowanc_at, std.mem.sliceAsBytes(ancestors[0..shape.total]));
        for (0..shape.total) |row| {
            if (segments[row] != 0 or parents[row] != @as(c_int, @intCast(row)) - 1 or ancestors[row] != (chain_row | @as(u32, @intCast(row)))) return error.QwenGdnTableMismatch;
        }
    }

    fn gdnArguments(e: *const Engine, shape: Shape, layer: c_int, row: usize) cuda.Args {
        var args: cuda.Args = .{};
        args.add(e.geo);
        for ([_]u64{ e.q, e.k, e.v, e.g_, e.beta }) |pointer| args.add(pointer);
        args.add(e.tables.ptr + if (shape.total > 1) chain_segs_at + row * @sizeOf(Seg) else segs_at);
        args.add(e.tables.ptr + rowpar_at);
        args.add(layer);
        args.add(e.y);
        args.add(e.tables.ptr + params_at);
        return args;
    }

    fn checkGdnNode(e: *const Engine, p: cuda.abi.KernelNodeParams, shape: Shape, function: cuda.abi.Function) !struct { layer: usize, row: usize } {
        const launch: hf.Launches = .{};
        if (p.func != function or p.grid_x != 1 or p.grid_y != launch.gdn_chain_grid_y or
            p.grid_z != e.g.v_heads * shape.nseg or p.block_x != launch.gdn_chain[0] or
            p.block_y != launch.gdn_chain[1] or p.block_z != launch.gdn_chain[2] or p.shared_bytes != 0) return error.QwenGdnGraphKernelMismatch;
        const actual = p.params orelse return error.QwenGdnGraphMissingArguments;
        const layer: *const c_int = @ptrCast(@alignCast(actual[8] orelse return error.QwenGdnGraphMissingArguments));
        const segment: *const u64 = @ptrCast(@alignCast(actual[6] orelse return error.QwenGdnGraphMissingArguments));
        if (layer.* < 0 or layer.* >= e.g.deltas()) return error.QwenGdnGraphLayerMismatch;
        var row: usize = 0;
        if (shape.total > 1) {
            const base = e.tables.ptr + chain_segs_at;
            if (segment.* < base or (segment.* - base) % @sizeOf(Seg) != 0) return error.QwenGdnGraphArgumentMismatch;
            row = @intCast((segment.* - base) / @sizeOf(Seg));
            if (row >= shape.total) return error.QwenGdnGraphArgumentMismatch;
        }
        const want = e.gdnArguments(shape, layer.*, row);
        for (0..want.count) |i| {
            const got: [*]const u8 = @ptrCast(actual[i] orelse return error.QwenGdnGraphMissingArguments);
            const bytes = want.storage[want.offsets[i]..][0..want.sizes[i]];
            if (!std.mem.eql(u8, got[0..bytes.len], bytes)) return error.QwenGdnGraphArgumentMismatch;
        }
        return .{ .layer = @intCast(layer.*), .row = row };
    }

    fn verifyGdnGraph(e: *const Engine, graph: cuda.graph.Graph, shape: Shape, node_count: usize) !usize {
        const bindings = e.modelopt orelse return error.UnsupportedQwenGdnLayout;
        const baseline = e.fns.gdn_chain.handle;
        const candidate = bindings.gdn_chain_transposed.handle;
        if (baseline == candidate) return error.QwenGdnGraphKernelMismatch;
        const expected = switch (shape.gdn_layout) {
            .value_key => baseline,
            .key_value => candidate,
        };
        const storage = try e.gpa.alloc(cuda.graph.Node, node_count);
        defer e.gpa.free(storage);
        const nodes = try graph.nodes(storage);
        if (nodes.len != node_count) return error.QwenGdnGraphWorkCountMismatch;
        const predecessors = try e.gpa.alloc(cuda.graph.Node, node_count);
        defer e.gpa.free(predecessors);
        var lib = e.d.lib;
        const get_deps = lib.lookup(*const fn (cuda.abi.GraphNode, ?[*]cuda.abi.GraphNode, *usize) callconv(.c) cuda.abi.Result, "cuGraphNodeGetDependencies") orelse return error.MissingSymbol;
        for (nodes, predecessors) |node, *previous| {
            var deps: [2]cuda.graph.Node = .{ null, null };
            var count: usize = deps.len;
            try e.d.check(get_deps(node, &deps, &count), "cuGraphNodeGetDependencies");
            if (count > 1) return error.QwenGdnGraphDependencyMismatch;
            previous.* = if (count == 0) null else deps[0];
        }
        var roster: GdnRoster = .{ .layers = e.g.deltas(), .rows = if (shape.total > 1) shape.total else 1 };
        var prior: cuda.graph.Node = null;
        for (0..node_count) |_| {
            var found: ?usize = null;
            for (predecessors, 0..) |previous, i| if (previous == prior) {
                if (found != null) return error.QwenGdnGraphDependencyMismatch;
                found = i;
            };
            const index = found orelse return error.QwenGdnGraphDependencyMismatch;
            var params: cuda.abi.KernelNodeParams = undefined;
            try e.d.check(e.d.api.cuGraphKernelNodeGetParams_v2(nodes[index], &params), "cuGraphKernelNodeGetParams");
            prior = nodes[index];
            if (params.func != baseline and params.func != candidate) continue;
            const item = try e.checkGdnNode(params, shape, expected);
            try roster.accept(item.layer, item.row);
        }
        try roster.finish();
        return roster.count;
    }

    const CkProjection = struct { weight: cuda.qmmf.Weight, x: u64, out: u64, rows: usize, quant: bool };

    fn addCkProjection(e: *const Engine, list: *std.ArrayList(CkProjection), p: Proj, x: u64, out: u64, rows: usize, quant: bool) !void {
        if (p == .linear and p.linear == .fp8 and p.linear.fp8.act != 0) {
            try list.append(e.gpa, .{ .weight = p.linear.fp8, .x = x, .out = out, .rows = rows, .quant = quant });
        }
    }

    fn addCkGroup(e: *const Engine, list: *std.ArrayList(CkProjection), projections: []const Proj, outputs: []const u64, shape: Shape) !void {
        var targets: [3]cuda.qlinear.Target = undefined;
        for (projections, outputs, 0..) |p, out, i| {
            if (p != .linear) return error.QwenFormatNotLoaded;
            targets[i] = .{ .weight = p.linear, .output = out };
        }
        const together = shape.reuse_fp8_quant and e.modelopt.?.linear().canDecodeTogether(targets[0..projections.len], .{ .x = e.x, .part = e.modelopt.?.part }, shape.total);
        for (projections, outputs, 0..) |p, out, i| try e.addCkProjection(list, p, e.x, out, shape.total, !together or i == 0);
    }

    fn ckPlan(e: *const Engine, shape: Shape) !std.ArrayList(CkProjection) {
        var list: std.ArrayList(CkProjection) = .empty;
        errdefer list.deinit(e.gpa);
        for (e.layers) |layer| {
            switch (layer.mixer) {
                .delta => |m| {
                    try e.addCkGroup(&list, &.{ m.qkv, m.z }, &.{ e.qkv, e.z }, shape);
                    try e.addCkProjection(&list, m.a, e.x, e.a, shape.total, true);
                    try e.addCkProjection(&list, m.b, e.x, e.b, shape.total, true);
                    try e.addCkProjection(&list, m.out, e.mix, e.r, shape.total, true);
                },
                .attention => |m| {
                    try e.addCkGroup(&list, &.{ m.q, m.k, m.v }, &.{ e.qraw, e.kraw, e.vraw }, shape);
                    try e.addCkProjection(&list, m.o, e.mix, e.r, shape.total, true);
                },
            }
            const dense: ?layers.Dense = switch (layer.mlp) {
                .dense => |m| m,
                .moe => |m| if (m.shared == .projected) m.shared.projected else null,
            };
            if (dense) |m| {
                try e.addCkProjection(&list, m.gate, e.x, e.gate, shape.total, true);
                try e.addCkProjection(&list, m.up, e.x, e.up, shape.total, true);
                try e.addCkProjection(&list, m.down, e.act, if (layer.mlp == .dense) e.r else e.routed.shared, shape.total, true);
            }
        }
        if (e.placement.ownsHead()) {
            const heads: usize = if (shape.record or shape.heads_all) shape.total else 1;
            try e.addCkProjection(&list, e.head, e.x + (shape.total - heads) * e.g.hidden * 2, e.logits, heads, true);
        }
        return list;
    }

    const CkNode = struct {
        function: cuda.abi.Function,
        grid: [3]usize,
        shared: usize = 0,
        threads: usize = 128,
        args: cuda.Args = .{},

        fn check(want: *const CkNode, got: cuda.abi.KernelNodeParams) !void {
            if (got.func != want.function or got.grid_x != want.grid[0] or got.grid_y != want.grid[1] or got.grid_z != want.grid[2] or
                got.block_x != want.threads or got.block_y != 1 or got.block_z != 1 or got.shared_bytes != want.shared) return error.QwenQuantGraphKernelMismatch;
            const args = got.params orelse return error.QwenQuantGraphArgumentMismatch;
            for (0..want.args.count) |i| {
                const actual: [*]const u8 = @ptrCast(args[i] orelse return error.QwenQuantGraphArgumentMismatch);
                const bytes = want.args.storage[want.args.offsets[i]..][0..want.args.sizes[i]];
                if (!std.mem.eql(u8, actual[0..bytes.len], bytes)) return error.QwenQuantGraphArgumentMismatch;
            }
        }
    };

    fn ckNode(e: *const Engine, p: CkProjection, kind: enum { quant, lane, reduce }) CkNode {
        const m = e.modelopt.?;
        const q = p.weight;
        const sk = cuda.qmmf.splitK(q.n, q.k);
        const cluster = sk > 1 and sk <= 8 and m.ck.clusters;
        var node: CkNode = .{ .function = null, .grid = .{ 1, 1, 1 } };
        switch (kind) {
            .quant => {
                node.function = m.ck.quant8_fn.handle;
                node.grid[0] = (p.rows * (q.k / 16) + 127) / 128;
                node.args.add(p.x);
                for ([_]usize{ q.k, p.rows, q.k }) |v| node.args.add(@as(i32, @intCast(v)));
                node.args.add(cuda.ck.inverse(q.act));
                node.args.add(m.rows);
            },
            .lane => {
                node.function = m.ck.lane[@intFromBool(cluster)].handle;
                node.grid = .{ (q.n + 63) / 64, 1, sk };
                node.shared = 4 * (16 * 64 + 64 * 64);
                node.args.add(m.rows);
                node.args.add(q.codes);
                node.args.add(cuda.ck.alpha(q.act, q.scale));
                node.args.add(p.out);
                node.args.add(m.part);
                for ([_]usize{ p.rows, q.n, q.k, sk, 1 }) |v| node.args.add(@as(i32, @intCast(v)));
            },
            .reduce => {
                node.function = m.ck.reduce.handle;
                node.grid[0] = (p.rows * q.n + 255) / 256;
                node.threads = 256;
                node.args.add(m.part);
                node.args.add(p.out);
                node.args.add(@as(i64, @intCast(p.rows * q.n)));
                node.args.add(@as(i32, @intCast(sk)));
                node.args.add(cuda.ck.alpha(q.act, q.scale));
            },
        }
        return node;
    }

    fn orderedGraphNodes(e: *const Engine, graph: cuda.graph.Graph, node_count: usize) ![]cuda.graph.Node {
        const nodes = try e.gpa.alloc(cuda.graph.Node, node_count);
        defer e.gpa.free(nodes);
        if ((try graph.nodes(nodes)).len != node_count) return error.QwenQuantGraphWorkCountMismatch;
        const previous = try e.gpa.alloc(cuda.graph.Node, node_count);
        defer e.gpa.free(previous);
        var lib = e.d.lib;
        const get = lib.lookup(*const fn (cuda.abi.GraphNode, ?[*]cuda.abi.GraphNode, *usize) callconv(.c) cuda.abi.Result, "cuGraphNodeGetDependencies") orelse return error.MissingSymbol;
        for (nodes, previous) |node, *pred| {
            var deps: [2]cuda.graph.Node = .{ null, null };
            var count: usize = deps.len;
            try e.d.check(get(node, &deps, &count), "cuGraphNodeGetDependencies");
            if (count > 1) return error.QwenQuantGraphDependencyMismatch;
            pred.* = if (count == 0) null else deps[0];
        }
        const ordered = try e.gpa.alloc(cuda.graph.Node, node_count);
        errdefer e.gpa.free(ordered);
        var prior: cuda.graph.Node = null;
        for (ordered) |*out| {
            var found: ?usize = null;
            for (previous, 0..) |pred, i| if (pred == prior) {
                if (found != null) return error.QwenQuantGraphDependencyMismatch;
                found = i;
            };
            out.* = nodes[found orelse return error.QwenQuantGraphDependencyMismatch];
            prior = out.*;
        }
        return ordered;
    }

    fn verifyQuantGraph(e: *const Engine, graph: cuda.graph.Graph, shape: Shape, node_count: usize) !usize {
        if (e.numerics != .hf_modelopt) return error.UnsupportedQwenQuantReuse;
        var plan = try e.ckPlan(shape);
        defer plan.deinit(e.gpa);
        var expected: std.ArrayList(CkNode) = .empty;
        defer expected.deinit(e.gpa);
        var quants: usize = 0;
        for (plan.items) |p| {
            if (p.quant) {
                try expected.append(e.gpa, e.ckNode(p, .quant));
                quants += 1;
            }
            try expected.append(e.gpa, e.ckNode(p, .lane));
            const sk = cuda.qmmf.splitK(p.weight.n, p.weight.k);
            if (sk > 1 and !(sk <= 8 and e.modelopt.?.ck.clusters)) try expected.append(e.gpa, e.ckNode(p, .reduce));
        }
        const nodes = try e.orderedGraphNodes(graph, node_count);
        defer e.gpa.free(nodes);
        const c = e.modelopt.?.ck;
        var count: usize = 0;
        for (nodes) |node| {
            var p: cuda.abi.KernelNodeParams = undefined;
            try e.d.check(e.d.api.cuGraphKernelNodeGetParams_v2(node, &p), "cuGraphKernelNodeGetParams");
            var relevant = p.func == c.quant8_fn.handle or p.func == c.reduce.handle;
            for (c.lane) |f| relevant = relevant or p.func == f.handle;
            if (!relevant) continue;
            if (count >= expected.items.len) return error.QwenQuantGraphWorkCountMismatch;
            try expected.items[count].check(p);
            count += 1;
        }
        if (count != expected.items.len) return error.QwenQuantGraphWorkCountMismatch;
        return quants;
    }

    fn verifyExpertGraph(e: *Engine, graph: cuda.graph.Graph, shape: Shape, node_count: usize) !usize {
        const roster_start = e.expert_graph_roster.items.len;
        errdefer e.expert_graph_roster.items.len = roster_start;
        const bindings = e.modelopt orelse return error.UnsupportedQwenExpertUp;
        const experts = bindings.nvfp4_experts;
        const n8 = experts.n8_up orelse return error.CandidateNotLoaded;
        const baseline = experts.fns[0].handle;
        const candidate = n8.function.handle;
        if (baseline == candidate) return error.QwenExpertGraphKernelMismatch;
        const split = experts.split_m_up;
        const split_function = if (split) |v| v.function.handle else null;
        const expected = if (shape.expert_up_split_m) (split orelse return error.UnsupportedQwenExpertUp).function.handle else if (shape.expert_up_n8) candidate else baseline;
        const storage = try e.gpa.alloc(cuda.graph.Node, node_count);
        defer e.gpa.free(storage);
        const nodes = try graph.nodes(storage);
        if (nodes.len != node_count) return error.QwenExpertGraphWorkCountMismatch;
        var layers_seen: [geometry.max_layers]bool = @splat(false);
        var count: usize = 0;
        for (nodes) |node| {
            var params: cuda.abi.KernelNodeParams = undefined;
            try e.d.check(e.d.api.cuGraphKernelNodeGetParams_v2(node, &params), "cuGraphKernelNodeGetParams");
            if (params.func != baseline and params.func != candidate and params.func != split_function) continue;
            if (params.func != expected) return error.QwenExpertGraphKernelMismatch;
            const args = params.params orelse return error.QwenExpertGraphMissingArguments;
            const up: *const u64 = @ptrCast(@alignCast(args[3] orelse return error.QwenExpertGraphMissingArguments));
            var matched = false;
            for (e.layers, 0..) |layer, index| {
                if (layer.mlp != .moe or layer.mlp.moe.experts.up != up.*) continue;
                if (layers_seen[index]) return error.QwenExpertGraphLayerMismatch;
                const m = layer.mlp.moe;
                try cuda.experts.Nvfp4Experts.validateN8Up(m.experts);
                const slots = if (m.shared == .projected) e.g.top_k else e.g.slots();
                const expert_count = if (m.shared == .projected) e.g.experts else e.g.experts + 1;
                const items = cuda.grouped.maxItems(@as(usize, shape.total) * slots, expert_count, 16);
                const nb = m.experts.width / cuda.experts.Nvfp4Experts.cols;
                const grid = if (shape.expert_up_split_m) split.?.grid(items, nb) else if (shape.expert_up_n8) n8.grid(items, nb) else cuda.experts.Nvfp4Experts.decodeGridWarps(.baseline, items, nb, experts.resident[0], experts.warps);
                const threads = if (shape.expert_up_split_m) cuda.experts.Nvfp4Experts.SplitMUp.threads else if (shape.expert_up_n8) cuda.experts.Nvfp4Experts.N8Up.threads else experts.warps * 32;
                if (params.grid_x != grid or params.grid_y != 1 or params.grid_z != 1 or params.block_x != threads or
                    params.block_y != 1 or params.block_z != 1 or params.shared_bytes != 0) return error.QwenExpertGraphKernelMismatch;
                const plan: cuda.grouped.Plan = .{ .members = e.routed.members, .items = e.routed.items, .counts = e.routed.counts, .rank = e.routed.rank, .hist = e.routed.hist };
                const expected_args = cuda.experts.Nvfp4Experts.decodeArgs(.up, .{ .x = e.x, .stride = e.g.hidden, .slots = slots }, m.experts, plan, e.routed.act, -1);
                var digest = std.crypto.hash.sha2.Sha256.init(.{});
                var argument_bytes: usize = 0;
                for (0..expected_args.count) |a| {
                    const actual: [*]const u8 = @ptrCast(args[a] orelse return error.QwenExpertGraphMissingArguments);
                    const bytes = expected_args.storage[expected_args.offsets[a]..][0..expected_args.sizes[a]];
                    if (!std.mem.eql(u8, actual[0..bytes.len], bytes)) return error.QwenExpertGraphArgumentMismatch;
                    digest.update(actual[0..bytes.len]);
                    argument_bytes += bytes.len;
                }
                var hash: [32]u8 = undefined;
                digest.final(&hash);
                try e.expert_graph_roster.append(e.gpa, .{
                    .shape_key = shape.key(),
                    .total = shape.total,
                    .heads_all = shape.heads_all,
                    .layer_local = index,
                    .layer_global = e.placement.first + index,
                    .arm = if (params.func == split_function) .split_m else if (params.func == candidate) .n8 else .baseline,
                    .grid = .{ params.grid_x, params.grid_y, params.grid_z },
                    .block = .{ params.block_x, params.block_y, params.block_z },
                    .shared_bytes = params.shared_bytes,
                    .abi_count = expected_args.count,
                    .argument_bytes = argument_bytes,
                    .args_sha256 = std.fmt.bytesToHex(hash, .lower),
                });
                layers_seen[index] = true;
                matched = true;
                count += 1;
                break;
            }
            if (!matched) return error.QwenExpertGraphLayerMismatch;
        }
        if (count != e.g.layers) return error.QwenExpertGraphWorkCountMismatch;
        return count;
    }

    pub fn commit(parts: []const Part, record: bool) void {
        for (parts) |p| {
            p.cache.last = if (record and p.rows > 1) .{ .start = p.cache.len, .rows = p.rows } else null;
            p.cache.len += p.rows;
            p.cache.written = p.cache.len;
        }
    }

    pub fn finish(e: *Engine, parts: []const Part, keep: Keep, shape: Shape) !void {
        if (!e.placement.ownsHead()) {
            try e.stream.synchronize();
            return;
        }
        const g = e.g;
        var total: usize = 0;
        for (parts) |p| total += p.rows;
        const heads: usize = if (shape.record or shape.heads_all) total else 1;
        switch (keep) {
            .draws => |pk| {
                const toks = e.tables.ptr + tokens_at;
                for (pk, 0..) |p, i| {
                    const row = e.logits + i * g.vocab * 2;
                    if (p.rule != null) {
                        try e.draws.draw(e.stream, row, g.vocab, e.tables.ptr + rules_at + i * @sizeOf(cuda.sample.Rule), e.tables.ptr + meta_at + i * 4, 0, toks + i * 4, 1, null, null);
                    } else try e.run(e.fns.argmax, .{ 1, 1, 1 }, .{ 1024, 1, 1 }, .{ row, @as(c_int, @intCast(g.vocab)), toks + i * 4 });
                }
                try e.d.check(e.d.api.cuMemcpyDtoHAsync_v2(e.host_logits.bytes.ptr, toks, total * 4, e.stream.handle), "cuMemcpyDtoHAsync");
                try e.stream.synchronize();
                @memcpy(e.tokens[0..total], e.host_logits.slice(u32)[0..total]);
            },
            .logits, .last, .prompt_logits => {
                const out = e.host_logits.bytes[0 .. heads * g.vocab * 2];
                try e.d.check(e.d.api.cuMemcpyDtoHAsync_v2(out.ptr, e.logits, out.len, e.stream.handle), "cuMemcpyDtoHAsync");
                try e.stream.synchronize();
                if (keep == .last or keep == .prompt_logits) @memcpy(parts[0].cache.logits, e.rowLogits(heads - 1));
            },
        }
    }

    /// A forward's shape, which fixes its launches: its tables and the caches' lengths are read on the device.
    pub const Shape = struct {
        total: u8,
        nseg: u8,
        record: bool,
        prompt: bool,
        taps: bool,
        gdn_layout: Cache.GdnLayout = .value_key,
        expert_up_n8: bool = false,
        expert_up_split_m: bool = false,
        reuse_fp8_quant: bool = false,
        heads_all: bool = false,

        fn key(s: Shape) u32 {
            return @as(u32, s.total) | @as(u32, s.nseg) << 8 | @as(u32, @intFromBool(s.record)) << 16 | @as(u32, @intFromBool(s.prompt)) << 17 | @as(u32, @intFromBool(s.taps)) << 18 | @as(u32, @backingInt(s.gdn_layout)) << 19 | @as(u32, @intFromBool(s.expert_up_n8)) << 20 | @as(u32, @intFromBool(s.heads_all)) << 21 | @as(u32, @intFromBool(s.expert_up_split_m)) << 22 | @as(u32, @intFromBool(s.reuse_fp8_quant)) << 23;
        }
    };

    pub fn mark(e: *Engine, phase: Phase) !void {
        if (e.observer) |o| try o.mark(o.ptr, phase, e.observed_layer);
    }

    /// The launches of a forward of `shape`, from the embedding to the kept rows' logits (captured once a shape).
    fn body(e: *Engine, shape: Shape) !void {
        const g = e.g;
        const total: usize = shape.total;
        const nseg: usize = shape.nseg;
        const record = shape.record;
        const prompt = shape.prompt;
        const heads: usize = if (record or shape.heads_all) total else 1;
        const t_ids = e.tables.ptr + ids_at;
        const t_segs = e.tables.ptr + segs_at;
        const t_rows = e.tables.ptr + rowseg_at;
        const t_par = e.tables.ptr + rowpar_at;
        const t_anc = e.tables.ptr + rowanc_at;
        const t_params = e.tables.ptr + params_at;
        const geo = e.geo;
        const group = g.q_heads / g.kv_heads;
        const is_hf = e.numerics == .hf_modelopt;
        const launch: hf.Launches = .{};
        const gdn_chain = switch (shape.gdn_layout) {
            .value_key => e.fns.gdn_chain,
            .key_value => e.modelopt.?.gdn_chain_transposed,
        };

        e.observed_layer = e.placement.first;
        try e.mark(.begin);
        if (e.placement.ownsEmbedding()) try e.embed(t_ids, e.h, total, t_params);
        // an image chunk's rows take the tower's features, and rotate at their staged t, h, w positions
        const mpos: u64 = if (e.features != 0) e.tables.ptr + mpos_at else 0;
        if (e.features != 0) try e.run(e.fns.splice, .{ total, 1, 1 }, .{ 256, 1, 1 }, .{ e.h, e.features, e.tables.ptr + src_at, @as(c_int, @intCast(g.hidden)), t_params });
        try e.mark(.embedding);
        var di: usize = 0;
        var ai: c_int = 0;
        for (e.layers, 0..) |l, i| {
            e.observed_layer = e.placement.first + i;
            try e.norm(e.placement.first + i != 0, l.input_norm, total);
            try e.mark(.input_norm);
            switch (l.mixer) {
                .delta => |m| {
                    const layer: c_int = @intCast(di);
                    if (shape.reuse_fp8_quant) {
                        _ = try e.modelopt.?.linear().decodeTogether(e.stream, &.{ .{ .weight = m.qkv.linear, .output = e.qkv }, .{ .weight = m.z.linear, .output = e.z } }, .{ .x = e.x, .part = e.modelopt.?.part }, total);
                    } else {
                        try e.proj(m.qkv, e.x, e.qkv, total, prompt);
                        try e.proj(m.z, e.x, e.z, total, prompt);
                    }
                    try e.proj(m.a, e.x, e.a, total, prompt);
                    try e.proj(m.b, e.x, e.b, total, prompt);
                    try e.mark(.mixer_inputs);
                    try e.run(e.fns.gdn_pre, .{ 1, 2 * g.k_heads + g.v_heads, total }, .{ if (is_hf) launch.gdn_pre else 32, 1, 1 }, .{ geo, e.qkv, t_segs, t_rows, t_par, layer, m.conv, e.a, e.b, m.a_log, m.dt_bias, e.q, e.k, e.v, e.g_, e.beta, t_params });
                    try e.mark(.gdn_pre);
                    try e.run(e.fns.conv_commit, .{ (g.convWidth() + 255) / 256, nseg, 1 }, .{ 256, 1, 1 }, .{ geo, e.qkv, t_segs, t_par, layer, t_params });
                    try e.mark(.conv_commit);
                    if (is_hf and total > 1) {
                        for (0..total) |row| try e.run(gdn_chain, .{ 1, launch.gdn_chain_grid_y, g.v_heads }, launch.gdn_chain, .{ geo, e.q, e.k, e.v, e.g_, e.beta, e.tables.ptr + chain_segs_at + row * @sizeOf(Seg), t_par, layer, e.y, t_params });
                    } else try e.run(gdn_chain, .{ 1, if (is_hf) launch.gdn_chain_grid_y else geometry.dk / 4, g.v_heads * nseg }, if (is_hf) launch.gdn_chain else .{ 32, 4, 1 }, .{ geo, e.q, e.k, e.v, e.g_, e.beta, t_segs, t_par, layer, e.y, t_params });
                    try e.mark(.gdn_chain);
                    try e.run(e.fns.gdn_post, .{ 1, g.v_heads, total }, .{ if (is_hf) launch.gdn_post else 32, 1, 1 }, .{ geo, e.y, e.z, m.norm, g.eps, e.mix, t_params });
                    try e.mark(.gdn_post);
                    try e.proj(m.out, e.mix, e.r, total, prompt);
                    di += 1;
                },
                .attention => |m| {
                    const aw = g.attnWidth();
                    if (shape.reuse_fp8_quant) {
                        _ = try e.modelopt.?.linear().decodeTogether(e.stream, &.{ .{ .weight = m.q.linear, .output = e.qraw }, .{ .weight = m.k.linear, .output = e.kraw }, .{ .weight = m.v.linear, .output = e.vraw } }, .{ .x = e.x, .part = e.modelopt.?.part }, total);
                    } else {
                        try e.proj(m.q, e.x, e.qraw, total, prompt);
                        try e.proj(m.k, e.x, e.kraw, total, prompt);
                        try e.proj(m.v, e.x, e.vraw, total, prompt);
                    }
                    try e.mark(.mixer_inputs);
                    try e.run(e.fns.head_norm, .{ total * g.q_heads, 1, 1 }, .{ if (is_hf) launch.head_norm else 64, 1, 1 }, .{ e.qraw, m.q_norm, e.qn, @as(c_int, @intCast(g.q_heads)), @as(c_int, @intCast(2 * aw)), @as(c_int, 2 * head_dim), g.eps, t_params });
                    try e.run(e.fns.head_norm, .{ total * g.kv_heads, 1, 1 }, .{ if (is_hf) launch.head_norm else 64, 1, 1 }, .{ e.kraw, m.k_norm, e.kn, @as(c_int, @intCast(g.kv_heads)), @as(c_int, @intCast(g.kv_heads * head_dim)), @as(c_int, head_dim), g.eps, t_params });
                    try e.run(if (e.kv8) e.fp8.rope else e.fns.rope, .{ total, 1, 1 }, .{ head_dim, 1, 1 }, .{ geo, e.qn, e.kn, e.vraw, t_segs, t_rows, t_anc, ai, t_params, mpos });
                    try e.run(if (e.kv8) e.fp8.attn_partial else e.fns.attn_partial, .{ if (is_hf) 1 else max_chunks, if (is_hf) g.q_heads else g.kv_heads, total }, if (is_hf) launch.attn_partial else .{ 32 * group, 1, 1 }, .{ geo, e.qn, t_segs, t_rows, t_anc, ai, t_params, @as(f32, 1.0 / 16.0), e.pm, e.pl, e.po });
                    try e.run(e.fns.attn_merge, .{ g.q_heads, total, 1 }, .{ if (is_hf) launch.attn_merge else 32, 1, 1 }, .{ geo, e.pm, e.pl, e.po, t_segs, t_rows, t_anc, t_params, e.y });
                    try e.run(e.fns.attn_gate, .{ (total * aw + 255) / 256, 1, 1 }, .{ 256, 1, 1 }, .{ geo, e.y, e.qraw, e.mix, @as(c_int, @intCast(total)), t_params });
                    try e.mark(.attention);
                    try e.proj(m.o, e.mix, e.r, total, prompt);
                    ai += 1;
                },
            }
            try e.mark(.mixer_output);
            try e.norm(true, l.post_norm, total);
            try e.mark(.post_norm);
            switch (l.mlp) {
                .dense => |m| {
                    try e.proj(m.gate, e.x, e.gate, total, prompt);
                    try e.proj(m.up, e.x, e.up, total, prompt);
                    const n: c_longlong = @intCast(total * g.intermediate);
                    try e.run(e.fns.mlp_act, .{ (total * g.intermediate + 255) / 256, 1, 1 }, .{ 256, 1, 1 }, .{ e.gate, e.up, e.act, n, t_params, @as(c_int, @intCast(g.intermediate)) });
                    try e.proj(m.down, e.act, e.r, total, prompt);
                    try e.mark(.dense);
                },
                .moe => |m| try routed.rows(e, m.router, m.experts, m.shared, e.x, e.r, total, prompt and total > most_window, shape.expert_up_n8, shape.expert_up_split_m),
            }
            if (shape.taps) for (e.taps[0..e.n_taps], 0..) |t, ti| if (t == e.placement.first + i) {
                try e.run(e.fns.tap, .{ total, 1, 1 }, .{ 256, 1, 1 }, .{ e.h, e.r, t_segs, t_rows, @as(c_int, @intCast(ti)), @as(c_int, @intCast(g.hidden)), t_params });
            };
        }
        if (!e.placement.ownsHead()) return;
        try e.norm(true, e.final_norm, total);
        // the head over the kept rows
        try e.proj(e.head, e.x + (total - heads) * g.hidden * 2, e.logits, heads, false);
        try e.mark(.head);
    }

    pub fn routedRows(e: *Engine, router: u64, layer: cuda.experts.Layer, shared: layers.SharedExpert, x: u64, out: u64, total: usize, wide_rows: bool) !void {
        return routed.rows(e, router, layer, shared, x, out, total, wide_rows, false, false);
    }

    /// Row `i`'s logits from the newest forward that kept logits.
    pub fn rowLogits(e: *const Engine, i: usize) []const u16 {
        return e.host_logits.slice(u16)[i * e.g.vocab ..][0..e.g.vocab];
    }
};

test "expert graph verification rejects opposite arms, wrong arguments, duplicates and missing work" {
    const Fake = struct {
        var params: cuda.abi.KernelNodeParams = undefined;
        var count: usize = 1;
        fn nodes(_: cuda.abi.Graph, out: ?[*]cuda.graph.Node, n: *usize) callconv(.c) cuda.abi.Result {
            if (out) |dst| for (0..@min(n.*, count)) |i| {
                dst[i] = @ptrFromInt(i + 1);
            };
            n.* = count;
            return cuda.abi.success;
        }
        fn get(_: cuda.graph.Node, out: *cuda.abi.KernelNodeParams) callconv(.c) cuda.abi.Result {
            out.* = params;
            return cuda.abi.success;
        }
    };
    var driver: cuda.Driver = undefined;
    driver.api.cuGraphGetNodes = Fake.nodes;
    driver.api.cuGraphKernelNodeGetParams_v2 = Fake.get;
    var bindings: modelopt.Bindings = undefined;
    bindings.nvfp4_experts.fns[0] = .{ .d = &driver, .handle = @ptrFromInt(1) };
    bindings.nvfp4_experts.resident[0] = 192;
    bindings.nvfp4_experts.warps = 4;
    bindings.nvfp4_experts.n8_up = .{ .function = .{ .d = &driver, .handle = @ptrFromInt(2) }, .resident = 192 };
    bindings.nvfp4_experts.split_m_up = .{ .function = .{ .d = &driver, .handle = @ptrFromInt(3) }, .resident = 168 };
    const expert_layer: cuda.experts.Layer = .{ .format = .nvfp4, .up = 101, .down = 102, .up_scale = 103, .down_scale = 104, .width = 512, .dims = 2048, .experts = 256 };
    var local_layers: [1]Layer = undefined;
    local_layers[0].mlp = .{ .moe = .{ .router = 0, .experts = expert_layer, .shared = .{ .projected = undefined } } };
    var e: Engine = undefined;
    e.gpa = std.testing.allocator;
    e.expert_graph_roster = .empty;
    defer e.expert_graph_roster.deinit(e.gpa);
    e.placement = .{ .first = 5, .end = 6, .total = 40 };
    e.d = &driver;
    e.modelopt = bindings;
    e.g = undefined;
    e.g.layers = 1;
    e.g.top_k = 8;
    e.g.experts = 256;
    e.g.hidden = 2048;
    e.layers = &local_layers;
    e.x = 301;
    e.routed = .{ .items = 201, .counts = 202, .members = 203, .rank = 204, .hist = 205, .act = 302 };
    const plan: cuda.grouped.Plan = .{ .items = 201, .counts = 202, .members = 203, .rank = 204, .hist = 205 };
    var args = cuda.experts.Nvfp4Experts.decodeArgs(.up, .{ .x = 301, .stride = 2048, .slots = 8 }, expert_layer, plan, 302, -1);
    Fake.params = .{ .func = @ptrFromInt(1), .grid_x = 32, .grid_y = 1, .grid_z = 1, .block_x = 128, .block_y = 1, .block_z = 1, .shared_bytes = 0, .params = args.pointers(), .extra = null };
    Fake.count = 1;
    const graph: cuda.graph.Graph = .{ .d = &driver, .handle = null };
    var shape: Engine.Shape = .{ .total = 1, .nseg = 1, .record = false, .prompt = true, .taps = false };
    try std.testing.expectEqual(@as(usize, 1), try e.verifyExpertGraph(graph, shape, 1));
    shape.expert_up_n8 = true;
    try std.testing.expectError(error.QwenExpertGraphKernelMismatch, e.verifyExpertGraph(graph, shape, 1));
    Fake.params.func = @ptrFromInt(2);
    Fake.params.grid_x = 128;
    try std.testing.expectEqual(@as(usize, 1), try e.verifyExpertGraph(graph, shape, 1));
    Fake.params.block_x = 32;
    try std.testing.expectError(error.QwenExpertGraphKernelMismatch, e.verifyExpertGraph(graph, shape, 1));
    Fake.params.block_x = 128;
    args.storage[args.offsets[10]] ^= 1;
    try std.testing.expectError(error.QwenExpertGraphArgumentMismatch, e.verifyExpertGraph(graph, shape, 1));
    args.storage[args.offsets[10]] ^= 1;
    shape.expert_up_split_m = true;
    try std.testing.expectError(error.QwenExpertGraphKernelMismatch, e.verifyExpertGraph(graph, shape, 1));
    Fake.params.func = @ptrFromInt(3);
    Fake.params.block_x = 256;
    try std.testing.expectEqual(@as(usize, 1), try e.verifyExpertGraph(graph, shape, 1));
    for (0..args.count) |i| {
        args.storage[args.offsets[i]] ^= 1;
        if (i == 3) {
            try std.testing.expectError(error.QwenExpertGraphLayerMismatch, e.verifyExpertGraph(graph, shape, 1));
        } else try std.testing.expectError(error.QwenExpertGraphArgumentMismatch, e.verifyExpertGraph(graph, shape, 1));
        args.storage[args.offsets[i]] ^= 1;
    }
    Fake.count = 2;
    try std.testing.expectError(error.QwenExpertGraphLayerMismatch, e.verifyExpertGraph(graph, shape, 2));
    Fake.count = 0;
    try std.testing.expectError(error.QwenExpertGraphWorkCountMismatch, e.verifyExpertGraph(graph, shape, 0));
    try std.testing.expectEqual(@as(usize, 3), e.expert_graph_roster.items.len);
    const roster = e.expert_graph_roster.items;
    try std.testing.expectEqual(.baseline, roster[0].arm);
    try std.testing.expectEqual(.n8, roster[1].arm);
    try std.testing.expectEqual(.split_m, roster[2].arm);
    try std.testing.expectEqual(@as(usize, 5), roster[2].layer_global);
    try std.testing.expectEqual(@as(usize, 14), roster[2].abi_count);
    try std.testing.expectEqual([3]u32{ 128, 1, 1 }, roster[2].grid);
    try std.testing.expectEqual([3]u32{ 256, 1, 1 }, roster[2].block);
    try std.testing.expectEqualSlices(u8, &roster[0].args_sha256, &roster[2].args_sha256);
}

test "combined candidates require HF and split-M requires the N8 contract" {
    try validateCombined(.hf_modelopt, true, true, true);
    try validateCombined(.hf_modelopt, false, false, true);
    try validateCombined(.mlx_sanitized, false, false, false);
    try std.testing.expectError(error.UnsupportedQwenExpertUp, validateCombined(.hf_modelopt, false, true, false));
    try std.testing.expectError(error.UnsupportedQwenExpertUp, validateCombined(.mlx_sanitized, true, true, false));
    try std.testing.expectError(error.UnsupportedQwenQuantReuse, validateCombined(.mlx_sanitized, false, false, true));
}

test "quant graph plans derive compatible groups and attest all quant lane reduction arguments" {
    var e: Engine = undefined;
    e.gpa = std.testing.allocator;
    e.x = 1024;
    var m: modelopt.Bindings = undefined;
    m.rows = 1048576;
    m.part = 2097152;
    m.ck.clusters = false;
    m.ck.quant8_fn.handle = @ptrFromInt(1);
    m.ck.lane[0].handle = @ptrFromInt(2);
    m.ck.reduce.handle = @ptrFromInt(3);
    e.modelopt = m;
    const w: cuda.qlinear.Weight = .{ .fp8 = .{ .mode = .fp8, .codes = 256, .scales = 0, .n = 64, .k = 4096, .npad = 128, .act = 0.5, .scale = 0.25 } };
    var projections = [_]Proj{ .{ .linear = w }, .{ .linear = w }, .{ .linear = w } };
    var list: std.ArrayList(Engine.CkProjection) = .empty;
    defer list.deinit(e.gpa);
    var shape: Engine.Shape = .{ .total = 16, .nseg = 1, .record = false, .prompt = true, .taps = false, .reuse_fp8_quant = true };
    try e.addCkGroup(&list, &projections, &.{ 3145728, 4194304, 5242880 }, shape);
    try std.testing.expectEqual(@as(usize, 3), list.items.len);
    try std.testing.expect(list.items[0].quant and !list.items[1].quant and !list.items[2].quant);
    inline for (.{ .quant, .lane, .reduce }) |kind| {
        var want = e.ckNode(list.items[1], kind);
        const pristine = want;
        var actual: cuda.abi.KernelNodeParams = .{ .func = want.function, .grid_x = @intCast(want.grid[0]), .grid_y = @intCast(want.grid[1]), .grid_z = @intCast(want.grid[2]), .block_x = @intCast(want.threads), .block_y = 1, .block_z = 1, .shared_bytes = @intCast(want.shared), .params = want.args.pointers(), .extra = null };
        try pristine.check(actual);
        for (0..want.args.count) |i| {
            want.args.storage[want.args.offsets[i]] ^= 1;
            try std.testing.expectError(error.QwenQuantGraphArgumentMismatch, pristine.check(actual));
            want.args.storage[want.args.offsets[i]] ^= 1;
        }
        actual.grid_x += 1;
        try std.testing.expectError(error.QwenQuantGraphKernelMismatch, pristine.check(actual));
    }
    list.clearRetainingCapacity();
    projections[2].linear.fp8.act = 0.25;
    try e.addCkGroup(&list, &projections, &.{ 3145728, 4194304, 5242880 }, shape);
    for (list.items) |p| try std.testing.expect(p.quant);
    list.clearRetainingCapacity();
    projections[2].linear = .{ .bf16 = w.fp8 };
    try e.addCkGroup(&list, &projections, &.{ 3145728, 4194304, 5242880 }, shape);
    try std.testing.expectEqual(@as(usize, 2), list.items.len);
    for (list.items) |p| try std.testing.expect(p.quant);
    list.clearRetainingCapacity();
    projections[2].linear = w;
    shape.reuse_fp8_quant = false;
    try e.addCkGroup(&list, &projections, &.{ 3145728, 4194304, 5242880 }, shape);
    for (list.items) |p| try std.testing.expect(p.quant);
}

test "bounded HF admission is opt in and does not imply recorded windows" {
    var cache: Cache = undefined;
    var parts = [_]Part{.{ .cache = &cache, .rows = 1 }};
    try validateHfWindow(.hf_modelopt, false, &parts, 1, .last);
    try std.testing.expectError(error.UnsupportedQwenWindow, validateHfWindow(.hf_modelopt, false, &parts, 1, .prompt_logits));
    for (1..17) |rows| {
        parts[0].rows = rows;
        try validateHfWindow(.hf_modelopt, true, &parts, rows, .last);
        try validateHfWindow(.hf_modelopt, true, &parts, rows, .prompt_logits);
        if (rows > 1) try std.testing.expectError(error.UnsupportedQwenWindow, validateHfWindow(.hf_modelopt, false, &parts, rows, .last));
        try std.testing.expectError(error.UnsupportedQwenWindow, validateHfWindow(.hf_modelopt, true, &parts, rows, .logits));
    }
    for ([_]usize{ 0, 17 }) |rows| {
        parts[0].rows = rows;
        try std.testing.expectError(error.UnsupportedQwenWindow, validateHfWindow(.hf_modelopt, true, &parts, rows, .last));
    }
    parts[0].rows = 2;
    try std.testing.expectError(error.UnsupportedQwenWindow, validateHfWindow(.hf_modelopt, true, &parts, 1, .last));
    parts[0].parents = &.{ -1, 0 };
    try std.testing.expectError(error.UnsupportedQwenWindow, validateHfWindow(.hf_modelopt, true, &parts, 2, .last));
    parts[0].parents = null;
    parts[0].images = @ptrFromInt(16);
    try std.testing.expectError(error.UnsupportedQwenWindow, validateHfWindow(.hf_modelopt, true, &parts, 2, .last));
    parts[0].images = null;
    try std.testing.expectError(error.UnsupportedQwenWindow, validateHfWindow(.mlx_sanitized, true, &parts, 2, .prompt_logits));
    try validateHfWindow(.mlx_sanitized, false, &parts, 2, .logits);
    try std.testing.expect(!records(.last) and !records(.prompt_logits));
    try std.testing.expect(records(.logits) and records(.{ .draws = &.{} }));
}

test "bounded HF graph keys separate every row count layout expert arm and head mode" {
    var keys: [512]u32 = undefined;
    var count: usize = 0;
    for (1..17) |rows| for ([_]Cache.GdnLayout{ .value_key, .key_value }) |layout| for ([_]bool{ false, true }) |n8| for ([_]bool{ false, true }) |heads| for ([_]bool{ false, true }) |split| for ([_]bool{ false, true }) |reuse| {
        const shape: Engine.Shape = .{ .total = @intCast(rows), .nseg = 1, .record = false, .prompt = true, .taps = false, .gdn_layout = layout, .expert_up_n8 = n8, .heads_all = heads, .expert_up_split_m = split, .reuse_fp8_quant = reuse };
        const key = shape.key();
        try std.testing.expect(std.mem.indexOfScalar(u32, keys[0..count], key) == null);
        keys[count] = key;
        count += 1;
    };
    try std.testing.expectEqual(@as(usize, 512), count);
}

test "bounded HF chain views retain every state binding and reject field corruption" {
    const base: Seg = .{ .rec = 101, .conv = 102, .kv = 103, .tap = 0, .log = 104, .cap = 4096, .len = 127, .row0 = 0, .rows = 16, .tap_rows = 0, .taps = 0, .delta = -3 };
    var views: [hf_prefill_rows]Seg = undefined;
    for (&views, 0..) |*view, row| view.* = chainView(base, row);
    try validateChainViews(base, &views);
    for (&views) |*view| {
        inline for (@typeInfo(Seg).@"struct".field_names) |name| {
            @field(view, name) += 1;
            try std.testing.expectError(error.QwenGdnTableMismatch, validateChainViews(base, &views));
            @field(view, name) -= 1;
        }
    }
    try std.testing.expectError(error.QwenGdnTableMismatch, validateChainViews(base, views[0..15]));
    try std.testing.expect(Engine.chain_segs_at % 256 == 0);
    try std.testing.expect(Engine.src_at >= Engine.chain_segs_at + @sizeOf(@TypeOf(views)));
}

test "bounded HF verifies uploaded row views parents ancestors and counts" {
    const Fake = struct {
        fn sync(_: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
            return cuda.abi.success;
        }
        fn copy(out: ?*anyopaque, source: cuda.abi.DevicePtr, bytes: usize) callconv(.c) cuda.abi.Result {
            const dst: [*]u8 = @ptrCast(out.?);
            const src: [*]const u8 = @ptrFromInt(source);
            @memcpy(dst[0..bytes], src[0..bytes]);
            return cuda.abi.success;
        }
    };
    var driver: cuda.Driver = undefined;
    driver.api.cuStreamSynchronize = Fake.sync;
    driver.api.cuMemcpyDtoH_v2 = Fake.copy;
    var host: [Engine.tables_len]u8 align(16) = @splat(0);
    var device: [Engine.tables_len]u8 align(16) = @splat(0);
    const base: Seg = .{ .rec = 101, .conv = 102, .kv = 103, .tap = 0, .log = 104, .cap = 4096, .len = 127, .row0 = 0, .rows = 2, .tap_rows = 0, .taps = 0, .delta = 0 };
    @memcpy(host[Engine.segs_at..][0..@sizeOf(Seg)], std.mem.asBytes(&base));
    const views = std.mem.bytesAsSlice(Seg, host[Engine.chain_segs_at..][0 .. 2 * @sizeOf(Seg)]);
    for (views, 0..) |*view, row| view.* = chainView(base, row);
    var cache: Cache = undefined;
    try Engine.rowTables(&host, &.{.{ .cache = &cache, .rows = 2 }});
    const params = std.mem.bytesAsSlice(c_int, host[Engine.params_at..][0..32]);
    params[2] = 2;
    params[3] = 1;
    device = host;
    var e: Engine = undefined;
    e.stream = .{ .d = &driver, .handle = null };
    e.tables = .{ .d = &driver, .ptr = @intFromPtr(&device), .len = device.len };
    e.staged = .{ .d = &driver, .bytes = &host };
    const shape: Engine.Shape = .{ .total = 2, .nseg = 1, .record = false, .prompt = true, .taps = false };
    try e.verifyHfChainTables(shape);
    for ([_]usize{ Engine.segs_at, Engine.chain_segs_at, Engine.chain_segs_at + @sizeOf(Seg), Engine.params_at + 8, Engine.params_at + 12, Engine.params_at + 16, Engine.rowseg_at, Engine.rowpar_at + 4, Engine.rowanc_at + 4 }) |offset| {
        device[offset] ^= 1;
        try std.testing.expectError(error.QwenGdnTableMismatch, e.verifyHfChainTables(shape));
        device[offset] ^= 1;
    }
}

test "GDN prefix roster rejects duplicates missing rows and wrong order" {
    var roster: GdnRoster = .{ .layers = 2, .rows = 16 };
    try std.testing.expectError(error.QwenGdnGraphLayerMismatch, roster.accept(0, 1));
    try roster.accept(0, 0);
    try std.testing.expectError(error.QwenGdnGraphLayerMismatch, roster.accept(0, 0));
    try std.testing.expectError(error.QwenGdnGraphLayerMismatch, roster.accept(1, 0));
    try std.testing.expectError(error.QwenGdnGraphWorkCountMismatch, roster.finish());
    for (1..16) |row| try roster.accept(0, row);
    for (0..16) |row| try roster.accept(1, row);
    try roster.finish();
    try std.testing.expectError(error.QwenGdnGraphLayerMismatch, roster.accept(2, 0));
}

test "GDN prefix node verifier checks all eleven ABI arguments" {
    var e: Engine = undefined;
    e.geo = .{ .hid = 2048, .nk = 16, .nv = 32, .qh = 16, .kvh = 2, .nl = 1 };
    e.g = undefined;
    e.g.layers = 1;
    e.g.linear = @splat(true);
    e.g.v_heads = 32;
    e.q = 101;
    e.k = 102;
    e.v = 103;
    e.g_ = 104;
    e.beta = 105;
    e.y = 106;
    e.tables.ptr = 4096;
    const shape: Engine.Shape = .{ .total = 2, .nseg = 1, .record = false, .prompt = true, .taps = false };
    var args = e.gdnArguments(shape, 0, 1);
    try std.testing.expectEqual(@as(usize, 11), args.count);
    var p: cuda.abi.KernelNodeParams = .{ .func = @ptrFromInt(1), .grid_x = 1, .grid_y = 1, .grid_z = 32, .block_x = 128, .block_y = 1, .block_z = 1, .shared_bytes = 0, .params = args.pointers(), .extra = null };
    const item = try e.checkGdnNode(p, shape, @ptrFromInt(1));
    try std.testing.expectEqual(@as(usize, 1), item.row);
    for (0..args.count) |i| {
        args.storage[args.offsets[i]] ^= 1;
        if (i == 8) {
            try std.testing.expectError(error.QwenGdnGraphLayerMismatch, e.checkGdnNode(p, shape, @ptrFromInt(1)));
        } else try std.testing.expectError(error.QwenGdnGraphArgumentMismatch, e.checkGdnNode(p, shape, @ptrFromInt(1)));
        args.storage[args.offsets[i]] ^= 1;
    }
    p.grid_z = 64;
    try std.testing.expectError(error.QwenGdnGraphKernelMismatch, e.checkGdnNode(p, shape, @ptrFromInt(1)));
}

test "prompt logits finish copies every row and commits terminal cache logits only" {
    const Fake = struct {
        fn sync(_: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
            return cuda.abi.success;
        }
        fn copy(out: ?*anyopaque, source: cuda.abi.DevicePtr, bytes: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
            const dst: [*]u8 = @ptrCast(out.?);
            const src: [*]const u8 = @ptrFromInt(source);
            @memcpy(dst[0..bytes], src[0..bytes]);
            return cuda.abi.success;
        }
    };
    var driver: cuda.Driver = undefined;
    driver.api.cuStreamSynchronize = Fake.sync;
    driver.api.cuMemcpyDtoHAsync_v2 = Fake.copy;
    var device = [_]u16{ 1, 2, 3, 4, 5, 6 };
    var host: [12]u8 align(16) = @splat(0);
    var terminal: [2]u16 = @splat(0);
    var cache: Cache = undefined;
    cache.logits = &terminal;
    var e: Engine = undefined;
    e.d = &driver;
    e.stream = .{ .d = &driver, .handle = null };
    e.placement = .{ .first = 0, .end = 1, .total = 1 };
    e.g.vocab = 2;
    e.logits = @intFromPtr(&device);
    e.host_logits = .{ .d = &driver, .bytes = &host };
    const parts = [_]Part{.{ .cache = &cache, .rows = 3 }};
    var shape: Engine.Shape = .{ .total = 3, .nseg = 1, .record = false, .prompt = true, .taps = false, .heads_all = true };
    try e.finish(&parts, .prompt_logits, shape);
    try std.testing.expectEqualSlices(u16, device[4..6], &terminal);
    for (0..3) |row| try std.testing.expectEqualSlices(u16, device[row * 2 ..][0..2], e.rowLogits(row));
    terminal = .{ 99, 98 };
    shape.record = true;
    shape.heads_all = false;
    try e.finish(&parts, .logits, shape);
    try std.testing.expectEqualSlices(u16, &.{ 99, 98 }, &terminal);
}
