//! GLM-5.3-Flash's TP forward on CUDA (the Python family's forward.compute and commit), partials over NCCL or copied.

const std = @import("std");
const cuda = @import("cuda");
const cfg = @import("config.zig");
const weights = @import("cuda_weights.zig");
const glue = @import("cuda_glue.zig");
const kda = @import("cuda_kda.zig");
const b16 = @import("cuda_b16.zig");
const projection = @import("cuda_proj.zig");
const latent = @import("cuda_latent.zig");
const sparse = @import("cuda_sparse.zig");
const experts = @import("cuda_experts.zig");
const kda_block = @import("cuda_kda_block.zig");
const dsa_block = @import("cuda_dsa_block.zig");
const mlp = @import("cuda_mlp.zig");
const moe = @import("cuda_moe.zig");
const quant = @import("cuda_quant.zig");
const q4 = @import("cuda_q4.zig");
const state = @import("cuda_state.zig");
const shared = @import("cuda_shared.zig");
const prompt_split = @import("cuda_prefill_split.zig");
const vision_binding = @import("cuda_vision_binding.zig");
const taps_mod = @import("cuda_taps.zig");
const mia = @import("cuda_experts_mia.zig");
const route_trace = @import("cuda_route_trace.zig");
const target_graph = @import("cuda_target_graph.zig");
const target_forward = @import("cuda_target_forward.zig");
const l2pf = @import("cuda_l2pf.zig");
const l2pf_plan = @import("cuda_l2pf_plan.zig");
const prompt_latent = @import("cuda_prompt_latent_aot.zig");
const mm_mod = @import("cuda_mm.zig");
const routed_exl3 = @import("cuda_exl3.zig");

test {
    _ = taps_mod;
}

pub const prefill_rows = state.prefill_rows;
pub const window_rows = state.window_rows;
pub const State = state.State;
pub const Buffers = state.Buffers;

pub const PromptDispatch = struct {
    wy: [2]u64,
    overlap: [2]u64,
    fronts: [4]u64,
    latent_aot: [3]u64,
};
var overlap_dispatch: [2]std.atomic.Value(u64) = .{ .init(0), .init(0) };
var front_dispatch: [4]std.atomic.Value(u64) = .{ .init(0), .init(0), .init(0), .init(0) };
pub fn promptDispatch() PromptDispatch {
    var value: PromptDispatch = .{ .wy = kda.wyDispatch(), .overlap = undefined, .fronts = undefined, .latent_aot = prompt_latent.dispatchCounts() };
    for (&overlap_dispatch, &value.overlap) |*counter, *count| count.* = counter.load(.monotonic);
    for (&front_dispatch, &value.fronts) |*counter, *count| count.* = counter.load(.monotonic);
    return value;
}

/// The modules a forward launches from: our fatbins.
pub const Modules = struct {
    glue: glue.Kernels,
    mods: [10]cuda.Module,
    kda: kda.Kernels,
    chunk_mod: cuda.Module,
    lane: cuda.qmmf.Lane, // the NVFP4 linears' decode rows (qlinear)
    prompt: cuda.nvfp4.Prompt, // and their prompt rows
    modelopt: cuda.modelopt.Packer,
    exl3_mod: cuda.Module,
    exl3: cuda.exl3_linear.Kernels, // EXL3 checkpoints' projections (qlinear)
    experts: experts.Kernels,
    routed_exl3: routed_exl3.Kernels,
    mm: mm_mod.Kernels,
    tiled: bool = false,
    packs: ?*const mm_mod.Packed = null,
    copy_rows: cuda.Function,
    put_slot: cuda.Function,
    pack_dense: cuda.Function,
    quant: quant.Kernels,
    q4: q4.Kernels,
    affine4: cuda.qlinear.Affine4Kernels,
    mia: ?mia.Kernels = null,
    ck: ?cuda.ck.Ck = null,
    source_mods: ?[2]cuda.Module = null,
    source_dense: ?cuda.exlr8.Dense = null,
    source_upcast: ?cuda.Function = null,
    prompt_latent: ?prompt_latent.Kernels = null,

    pub fn load(d: *const cuda.Driver, ctx: *const cuda.Context) !Modules {
        const k = cuda.kernels;
        var m: Modules = undefined;
        m.mia = null;
        m.ck = null;
        m.source_mods = null;
        m.source_dense = null;
        m.source_upcast = null;
        m.prompt_latent = null;
        m.tiled = false;
        m.packs = null;
        var n: usize = 0;
        errdefer for (m.mods[0..n]) |*x| x.unload();
        for (&m.mods, [_][]const u8{ k.glm_kda, k.glm_ops, k.experts, k.affine4_pack, k.glm_quant, k.qmm_group, k.glm_glue, k.qmm_prefill, k.lane_gemv, k.glm_mm }) |*x, image| {
            x.* = try cuda.Module.load(d, image);
            n += 1;
        }
        m.lane = try cuda.qmmf.Lane.load(d, @intCast(try ctx.attribute(.compute_capability_major)));
        errdefer m.lane.unload();
        m.prompt = try cuda.nvfp4.Prompt.load(d);
        errdefer m.prompt.unload();
        m.modelopt = try cuda.modelopt.Packer.load(d);
        errdefer m.modelopt.unload();
        m.exl3_mod = try cuda.Module.load(d, k.exl3);
        errdefer m.exl3_mod.unload();
        m.exl3 = try cuda.exl3_linear.Kernels.resolve(d, m.exl3_mod, 16384);
        errdefer m.exl3.deinit();
        m.kda = try kda.Kernels.load(m.mods[0]);
        m.chunk_mod = try cuda.Module.load(d, k.glm_kda_chunk);
        errdefer m.chunk_mod.unload();
        try m.kda.loadChunk(m.chunk_mod);
        m.experts = try experts.Kernels.load(d, m.mods[2], @intCast(try ctx.attribute(.multiprocessor_count)));
        errdefer m.experts.unload();
        m.copy_rows = try m.mods[1].function("tf_glm_copy_rows");
        m.put_slot = try m.mods[1].function("tf_glm_put_slot");
        m.pack_dense = try m.mods[3].function("tf_pack_dense");
        m.quant = try quant.Kernels.load(m.mods[4]);
        m.q4 = try q4.Kernels.load(m.mods[5], m.mods[3], ctx);
        m.glue = try glue.Kernels.load(m.mods[6]);
        m.affine4 = try cuda.qlinear.Affine4Kernels.resolve(d, m.mods[5], m.mods[7], m.mods[8], m.mods[3], @intCast(try ctx.attribute(.multiprocessor_count)), try ctx.attribute(.compute_capability_major) == 12 and try ctx.attribute(.compute_capability_minor) == 1);
        m.routed_exl3 = try routed_exl3.Kernels.load(d);
        errdefer m.routed_exl3.unload();
        m.mm = try mm_mod.Kernels.load(m.mods[9]);
        return m;
    }

    /// What weights.load runs: the dense packing, the FP8 lane, and the MTP head's quantizers and 4-bit tiling.
    pub fn tools(m: *const Modules) weights.Tools {
        return .{ .pack_dense = m.pack_dense, .lane = &m.lane, .quant = &m.quant, .q4 = &m.q4, .affine4 = &m.affine4, .modelopt = &m.modelopt, .ck = if (m.ck) |*c| c else null };
    }

    pub fn enableNativeDense(m: *Modules, d: *const cuda.Driver, ctx: *const cuda.Context) !void {
        if (m.ck != null) return;
        if (try ctx.capability() != 121) return error.UnsupportedNativeDenseDevice;
        m.ck = try cuda.ck.Ck.load(d, 12, @intCast(try ctx.attribute(.multiprocessor_count)));
    }

    pub fn enableSourceDense(m: *Modules, d: *const cuda.Driver) !void {
        if (m.source_mods != null) return;
        var dm = try cuda.Module.load(d, cuda.kernels.exlr8_dense);
        errdefer dm.unload();
        var pm = try cuda.Module.load(d, cuda.kernels.torch_pointwise);
        errdefer pm.unload();
        const dense = try cuda.exlr8.Dense.resolve(dm);
        const cast = try pm.function("tf_bf16_to_f32_kernel");
        m.source_mods = .{ dm, pm };
        m.source_dense = dense;
        m.source_upcast = cast;
    }

    pub fn ops(m: *const Modules) cuda.Module {
        return m.mods[1];
    }

    pub fn deinit(m: *Modules) void {
        m.routed_exl3.unload();
        if (m.prompt_latent) |*kernels| kernels.unload();
        if (m.source_mods) |*mods| for (mods) |*x| x.unload();
        if (m.ck) |*c| c.unload();
        if (m.mia) |*k| k.unload();
        m.chunk_mod.unload();
        for (&m.mods) |*x| x.unload();
        m.affine4.deinit();
        m.lane.unload();
        m.prompt.unload();
        m.modelopt.unload();
        m.exl3.deinit();
        m.exl3_mod.unload();
        m.experts.unload();
        m.* = undefined;
    }
};

/// overlap.ROW_STEP and PIECES_MAX: piece boundaries on the BF16 prompt matmul's 128-row blocks, at most 16 pieces.
pub const row_step = 128;
pub const max_pieces = 16;

/// overlap.ranges: [lo, hi) row pieces of a chunk of `rows`, about equal on row_step boundaries (fewer when short).
pub fn ranges(rows: usize, pieces: usize, out: *[max_pieces][2]usize) usize {
    const per = (rows + pieces - 1) / pieces;
    const size = (per + row_step - 1) / row_step * row_step;
    var n: usize = 0;
    var lo: usize = 0;
    while (lo < rows) : (lo += size) {
        out[n] = .{ lo, @min(rows, lo + size) };
        n += 1;
    }
    return n;
}

/// A prompt chunk's pieced exchanges (overlap.Overlap): second stream and events; `count`: TF_GLM_OVERLAP_PIECES (3).
pub const Pieces = struct {
    count: usize = 4,
    side: cuda.Stream,
    filled: [max_pieces]cuda.Event,
    gathered: [max_pieces]cuda.Event,
    glues: [max_pieces]cuda.Event, // split: a piece's own rows glued
    shared: [max_pieces]cuda.Event, // split: every rank's glued rows of a piece in

    pub fn init(d: *const cuda.Driver, count: usize) !Pieces {
        if (count == 0 or count > max_pieces) return error.BadPieces;
        var p: Pieces = .{ .count = count, .side = try cuda.Stream.init(d, false), .filled = undefined, .gathered = undefined, .glues = undefined, .shared = undefined };
        for (&p.filled, &p.gathered, &p.glues, &p.shared) |*a, *b, *g, *s| {
            a.* = try cuda.Event.init(d, false);
            b.* = try cuda.Event.init(d, false);
            g.* = try cuda.Event.init(d, false);
            s.* = try cuda.Event.init(d, false);
        }
        return p;
    }

    pub fn deinit(p: *Pieces) void {
        for (&p.filled, &p.gathered, &p.glues, &p.shared) |*a, *b, *g, *s| {
            a.deinit();
            b.deinit();
            g.deinit();
            s.deinit();
        }
        p.side.deinit();
        p.* = undefined;
    }
};

/// How the ranks' partials meet: every rank in-process copying into the others' slots, or one rank over NCCL.
const relay = @import("cuda_relay.zig");
const prof_mod = @import("cuda_prof.zig");
const images_mod = @import("cuda_images.zig");
pub const ImageRows = struct { v: *images_mod.Images, slot: u32 };

pub const Exchange = union(enum) {
    local,
    nccl: struct { lib: *const cuda.nccl.Library, comm: cuda.nccl.Comm },
};

/// One rank of the forward: its weights, its sequence's caches, the window's buffers.
pub const Rank = struct {
    w: *const weights.Weights,
    st: *State,
    b: *Buffers,
    /// A shared window (Buffers.initShared): each stream's rows in row order; empty: every row on `st`.
    segs: []const Seg = &.{},
};

/// One stream's rows of a shared window: rows row0..row0 + rows, on sequence `st` from its position.
pub const Seg = struct { st: *State, row0: usize, rows: usize };

pub const Forward = struct {
    d: *const cuda.Driver,
    m: *const Modules,
    stream: cuda.Stream,
    ranks: []const Rank, // every rank of a local check, or this process's one
    exchange: Exchange,
    tp_group: ?*const cuda.tp_group.Group = null,
    prefetch: ?*l2pf.Prefetch = null,
    replicas: ?*dsa_block.Replicas = null,
    attn_fp8load: bool = false,
    kda_prompt: kda.PromptAlgorithm = .serial,
    affine_prompt_policy: cuda.qlinear.AffinePromptPolicy = .stock,
    kda_wide: bool = true, // TF_GLM_KDA_DECODE_WIDE (the Python family's default): decode windows on KDA's three-kernel chain
    prefill_overlap: u8 = 0,
    prefill_pieces: u32 = 2,
    pieces: ?*const Pieces = null, // a prompt chunk's exchanges in row pieces on a second stream (null: whole)
    ring: bool = false,
    images: ?ImageRows = null,
    prof: ?*prof_mod.Prof = null,
    taps: ?taps_mod.Taps = null,
    split: bool = true, // TF_GLM_PREFILL_REDUCE=split (the default with send/recv): each rank glues its own rows
    vision: ?@import("cuda_vision_binding.zig").Binding = null,
    route_trace: ?*route_trace.Trace = null,
    target_graphs: ?*target_graph.Bank = null,
    full_target: ?*target_forward.Runtime = null,
    target_buffer: target_graph.Buffer = .lone,
    capture_lifetime: u64 = 0, // renewed only after the engine drains and retires captured resources
    target_failure: *const fn (anyerror) void = @import("cuda_kv_owner.zig").failPair,

    pub fn set(f: *const Forward) glue.Set {
        return .{ .glue = &f.m.glue, .mm = if (f.m.tiled) &f.m.mm else null, .packs = f.m.packs, .linear = f.linear(), .copy_rows = f.m.copy_rows, .exlr8_dense = if (f.m.source_dense) |*d| d else null, .source_upcast = f.m.source_upcast, .attn_fp8load = f.attn_fp8load, .prompt = f.ranks[0].b.prefill, .prompt_aot = if (f.ranks[0].b.prefill) if (f.m.prompt_latent) |*kernels| kernels else null else null };
    }

    /// Captured sinks and fixed owners only. Request addresses/positions live in staged device metadata.
    pub fn graphKey(f: *const Forward, layer: usize, rows: usize) !target_graph.Key {
        if (f.capture_lifetime == 0) return error.InvalidCaptureLifetime;
        const r = f.ranks[0];
        if (f.taps) |t| if (t.lifetime == 0) return error.InvalidCaptureLifetime;
        return .{
            .buffer = f.target_buffer,
            .layer = layer,
            .rows = rows,
            .context = .{
                .lifetime = f.capture_lifetime,
                .modules = @intFromPtr(f.m),
                .scratch = @intFromPtr(r.b),
                .weights = @intFromPtr(r.w),
                .transport = @intFromPtr(f.tp_group),
                .prefetch = @intFromPtr(f.prefetch),
                .replicas = @intFromPtr(f.replicas),
                .packs = @intFromPtr(f.m.packs),
                .local_taps = if (r.b.taps) |t| t.ptr else 0,
                .tap_rows = r.b.rows,
                .policy = @as(u64, @intFromBool(f.m.tiled)) | (@as(u64, @intFromBool(f.attn_fp8load)) << 1) |
                    (@as(u64, @intFromBool(f.kda_wide)) << 2) | (@as(u64, @intFromBool(f.split)) << 3) |
                    (@as(u64, @intFromBool(f.ring)) << 4) | (@as(u64, @backingInt(f.affine_prompt_policy)) << 8) |
                    (@as(u64, @backingInt(f.kda_prompt)) << 16),
                .tap = if (f.taps) |t| .{ .lifetime = t.lifetime, .address = t.out, .callback = @intFromPtr(t.f.handle) } else .{},
            },
            .profile = if (f.prof) |p| p.graphProfile() else null,
        };
    }

    /// Every supported projection format, with explicit conversions kept separate from stored formats.
    pub fn linear(f: *const Forward) cuda.qlinear.Linear {
        return .{ .affine4 = &f.m.affine4, .affine_prompt_policy = f.affine_prompt_policy, .lane = &f.m.lane, .nvfp4_prompt = &f.m.prompt, .exl3_trellis = &f.m.exl3, .ck = if (f.m.ck) |*c| c else null };
    }

    /// reduce.sums' gather: every rank's partial of `rows` rows into every rank's gathered buffer, rank order.
    fn gather(f: *const Forward, rows: usize) !void {
        const c = &f.ranks[0].w.config;
        const bytes = rows * c.hidden * 4;
        switch (f.exchange) {
            .local => for (f.ranks) |to| for (f.ranks) |from| {
                if (to.w.share.rank == from.w.share.rank) continue;
                const off = from.w.share.rank * bytes;
                try f.d.check(f.d.api.cuMemcpyDtoDAsync_v2(to.b.gathered + off, from.b.gathered + off, bytes, f.stream.handle), "cuMemcpyDtoDAsync");
            },
            .nccl => |n| {
                const r = f.ranks[0];
                const count = rows * c.hidden;
                if (f.tp_group) |group| {
                    if (group.world != r.w.share.world or group.rank != r.w.share.rank) return error.TransportIdentityMismatch;
                    try group.allGather(f.stream, r.b.gathered + r.w.share.rank * bytes, r.b.gathered, bytes);
                } else try n.lib.check(n.lib.api.ncclAllGather(r.b.gathered + r.w.share.rank * bytes, r.b.gathered, count, .f32, n.comm, f.stream.handle), "ncclAllGather");
            },
        }
    }

    fn slot(r: Rank, rows: usize) u64 {
        return r.b.gathered + r.w.share.rank * rows * r.w.config.hidden * 4;
    }

    /// forward._pre: hc_pre of every row at a site into b.normed, xs, post and comb.
    fn pre(f: *const Forward, r: Rank, hc: weights.Hc, norm: glue.Norm, rows: usize) !void {
        const c = &r.w.config;
        const b = r.b;
        try glue.hcPre(f.set(), f.stream, b.x, hc.fn_w, hc.base, hc.scale, norm, b.normed, b.xs, b.post, b.comb, b.hcpart, c.eps, c.hc_eps, c.sinkhorn, rows, c.hidden, c.hc);
    }

    fn post(f: *const Forward, r: Rank, rows: usize) !void {
        const c = &r.w.config;
        try glue.hcPost(f.set(), f.stream, r.b.x, r.b.x, r.b.gathered, r.b.post, r.b.comb, rows, c.hidden, c.hc, r.w.share.world);
    }

    pub fn dsaKernels(f: *const Forward, r: Rank) dsa_block.Kernels {
        return .{ .set = f.set(), .ops = f.m.ops(), .replicas = f.replicas, .prefill = r.b.prefill, .owner = switch (f.exchange) {
            .local => null,
            .nccl => |n| .{ .d = f.d, .lib = n.lib, .comm = n.comm, .rank = r.w.share.rank, .tp_group = f.tp_group },
        } };
    }

    /// forward._mixer: the layer's KDA or DSA block on b.normed; `li`/`di`: its index among loaded KDA / DSA layers.
    fn mixer(f: *const Forward, r: Rank, l: weights.Layer, li: usize, di: usize, rows: usize, nch: usize, out: ?u64) !void {
        return f.mixerPrepared(r, l, li, di, rows, nch, out, null);
    }

    fn mixerPrepared(f: *const Forward, r: Rank, l: weights.Layer, li: usize, di: usize, rows: usize, nch: usize, out: ?u64, done: ?prompt_split.Range) !void {
        const c = &r.w.config;
        const b = r.b;
        const st = r.st;
        switch (l.mixer) {
            .kda => |k| {
                const cur = st.cur[li];
                var s = b.kda;
                s.p = if (b.prefill) b.kproj else st.proj + li * st.proj_layer * 2;
                s.saves.state_out = st.recAt(1 - cur, li);
                if (!b.prefill) { // a window's saved rows stay with the layer until its commit
                    const kv = li * window_rows * st.heads * kda.dk;
                    s.saves.k = st.save_k + kv * 4;
                    s.saves.v = st.save_v + kv * 2;
                    s.saves.g = st.save_g + kv * 4;
                    s.saves.b = st.save_b + li * window_rows * st.heads * 4;
                }
                const conv = st.conv + li * 3 * st.conv_c * 2;
                if (b.prefill and done != null) {
                    try f.frontRemainder(r, l, .mixer, rows, done);
                } else try kda_block.project(f.set(), f.stream, k, s, b.normed, c.hidden, rows);
                try f.prefetchSite(l.index, .o, rows);
                if (b.prefill) {
                    try kda_block.promptRows(&f.m.kda, f.stream, k, s, rows, conv, st.recAt(cur, li), st.pos, f.kda_prompt);
                } else try kda_block.chainRows(&f.m.kda, f.stream, k, s, rows, conv, st.recAt(cur, li), f.kda_wide);
                if (b.prefill) try glue.convShift(f.set(), f.stream, conv, s.p, rows, rows * k.width(), k.width(), 1, 3 * k.heads * kda.dk);
                if (out) |o| try kda_block.fill(f.set(), f.stream, k, s, 0, rows, o);
                if (b.prefill) st.cur[li] = 1 - cur; // a prompt chunk keeps every row: the layer commits now
            },
            .mla => |a| {
                var ks = f.dsaKernels(r);
                ks.slot = di;
                if (b.prefill and done != null) {
                    try f.frontRemainder(r, l, .mixer, rows, done);
                    try dsa_block.prepareRest(f.set(), f.stream, a, b.dsa, b.normed, c.hidden, rows, dsa_block.sparseRows(a, st.pos, rows));
                } else try dsa_block.prepare(f.set(), f.stream, a, b.dsa, b.normed, c.hidden, rows, dsa_block.sparseRows(a, st.pos, rows));
                try f.prefetchSite(l.index, .o, rows);
                try dsa_block.attend(ks, f.stream, a, st.caches[di], b.dsa, 0, rows, st.pos_dev, st.pos, nch, b.part_rows);
                try latent.expandV(f.set(), f.stream, b.dsa.ol, a.wv, b.dsa.vn, rows, a.heads, a.dv, a.lw);
                if (out) |o| try dsa_block.fill(f.set(), f.stream, a, b.dsa, 0, rows, o);
            },
        }
    }

    /// forward._ffn: the dense MLP or MoE block on b.normed into this rank's slot.
    fn ffn(f: *const Forward, r: Rank, l: weights.Layer, rows: usize, out: ?u64) !void {
        const c = &r.w.config;
        const b = r.b;
        if (b.prefill and r.segs.len > 0) {
            switch (l.ffn) {
                .dense => |m| for (r.segs) |g| {
                    try mlp.body(f.mlpKernels(), f.stream, m, b.dense, b.normed + g.row0 * c.hidden * 2, c.hidden, g.rows, true);
                    if (out) |o| try mlp.fill(f.mlpKernels(), f.stream, m, b.dense, 0, g.rows, true, o + g.row0 * c.hidden * 4);
                },
                .moe => |m| {
                    try moe.body(f.moeKernels(), f.stream, m, b.moe, b.normed, c.hidden, rows, true);
                    if (out) |o| for (r.segs) |g| try moe.fill(f.moeKernels(), f.stream, m, b.moe, b.normed, c.hidden, g.row0, g.row0 + g.rows, true, o + g.row0 * c.hidden * 4);
                },
            }
            return;
        }
        switch (l.ffn) {
            .dense => |m| {
                try mlp.body(f.mlpKernels(), f.stream, m, b.dense, b.normed, c.hidden, rows, b.prefill);
                if (out) |o| try mlp.fill(f.mlpKernels(), f.stream, m, b.dense, 0, rows, b.prefill, o);
            },
            .moe => |m| {
                const shared_first = if (f.prefetch) |p| !b.prefill and rows <= p.max_rows else false;
                if (shared_first) try moe.shared(f.moeKernels(), f.stream, m, b.moe, b.normed, c.hidden, 0, rows);
                try moe.body(f.moeKernels(), f.stream, m, b.moe, b.normed, c.hidden, rows, b.prefill);
                if (out) |o| {
                    if (shared_first) try moe.finish(f.moeKernels(), f.stream, m, b.moe, 0, rows, false, o) else try moe.fill(f.moeKernels(), f.stream, m, b.moe, b.normed, c.hidden, 0, rows, b.prefill, o);
                }
            },
        }
    }

    fn mlpKernels(f: *const Forward) mlp.Kernels {
        return .{ .set = f.set(), .linear = f.linear(), .copy_rows = f.m.copy_rows };
    }

    fn moeKernels(f: *const Forward) moe.Kernels {
        return .{ .set = f.set(), .experts = &f.m.experts, .put_slot = f.m.put_slot, .mia_kernels = if (f.m.mia) |*k| k else null, .exl3 = &f.m.routed_exl3, .prof = f.prof };
    }

    const Site = enum { mixer, ffn };

    /// forward.out_proj's fill of rows lo..hi at a site of layer `l` into `out` (this rank's rows of a piece).
    fn fillRows(f: *const Forward, r: Rank, l: weights.Layer, site: Site, lo: usize, hi: usize, out: u64) !void {
        const c = &r.w.config;
        const b = r.b;
        switch (site) {
            .mixer => switch (l.mixer) {
                .kda => |k| try kda_block.fill(f.set(), f.stream, k, b.kda, lo, hi, out),
                .mla => |a| try dsa_block.fill(f.set(), f.stream, a, b.dsa, lo, hi, out),
            },
            .ffn => switch (l.ffn) {
                .dense => |m| try mlp.fill(f.mlpKernels(), f.stream, m, b.dense, lo, hi, b.prefill, out),
                .moe => |m| try moe.fill(f.moeKernels(), f.stream, m, b.moe, b.normed, c.hidden, lo, hi, b.prefill, out),
            },
        }
    }

    /// Piece [lo, hi)'s room in a rank's gathered buffer (pieces never share bytes): [world][hi - lo][dims] fp32.
    fn room(r: Rank, lo: usize) u64 {
        return r.b.gathered + r.w.share.world * lo * r.w.config.hidden * 4;
    }

    /// reduce.shares: [lo, hi) of the rows of a piece of `n` each rank owns: about equal, the first n % world one more.
    fn share(n: usize, world: usize, rank: usize) [2]usize {
        const base = n / world;
        const extra = n % world;
        const lo = rank * base + @min(rank, extra);
        return .{ lo, lo + base + @intFromBool(rank < extra) };
    }

    /// Whether piece [lo, lo + n) splits (TF_GLM_PREFILL_REDUCE=split): at least one row for every rank.
    fn splits(f: *const Forward, n: usize) bool {
        return f.split and n >= f.ranks[0].w.share.world;
    }

    fn copy(f: *const Forward, stream: cuda.Stream, dst: u64, src: u64, bytes: usize) !void {
        try f.d.check(f.d.api.cuMemcpyDtoDAsync_v2(dst, src, bytes, stream.handle), "cuMemcpyDtoDAsync");
    }

    /// overlap.Overlap.partials: each piece filled on the main stream, exchanged on the second (gathered or split).
    fn exchangePieces(f: *const Forward, p: *const Pieces, j: usize, site: Site, cut: []const [2]usize) !void {
        const D = f.ranks[0].w.config.hidden;
        const world = f.ranks[0].w.share.world;
        for (cut, 0..) |piece, q| {
            const lo = piece[0];
            const n = piece[1] - lo;
            const split = f.splits(n);
            for (f.ranks) |r| try f.fillRows(r, r.w.layers[j], site, lo, piece[1], if (split) r.b.part + lo * D * 4 else room(r, lo) + r.w.share.rank * n * D * 4);
            try p.filled[q].record(f.stream);
            try p.side.wait(p.filled[q]);
            const bytes = n * D * 4;
            if (!split) switch (f.exchange) {
                .local => for (f.ranks) |to| for (f.ranks) |from| {
                    if (to.w.share.rank == from.w.share.rank) continue;
                    const off = from.w.share.rank * bytes;
                    try f.copy(p.side, room(to, lo) + off, room(from, lo) + off, bytes);
                },
                .nccl => |x| {
                    const r = f.ranks[0];
                    try x.lib.check(x.lib.api.ncclAllGather(room(r, lo) + r.w.share.rank * bytes, room(r, lo), n * D, .f32, x.comm, p.side.handle), "ncclAllGather");
                },
            } else switch (f.exchange) {
                .local => for (f.ranks) |to| {
                    const mine = share(n, world, to.w.share.rank);
                    const own = mine[1] - mine[0];
                    for (f.ranks) |from| try f.copy(p.side, room(to, lo) + from.w.share.rank * own * D * 4, from.b.part + (lo + mine[0]) * D * 4, own * D * 4);
                },
                .nccl => |x| {
                    const r = f.ranks[0];
                    const me = r.w.share.rank;
                    const mine = share(n, world, me);
                    const own = mine[1] - mine[0];
                    const plan: relay.Relay = .{ .world = world, .rank = me, .n = n };
                    if (f.ring) try relay.partials(x.lib, x.comm, p.side, plan, room(r, lo), r.b.part, lo, D) else try relay.splitPartials(x.lib, x.comm, p.side, plan, room(r, lo), r.b.part, lo, D);
                    try f.copy(p.side, room(r, lo) + me * own * D * 4, r.b.part + (lo + mine[0]) * D * 4, own * D * 4);
                },
            }
            try p.gathered[q].record(p.side);
        }
    }

    /// hc_post of rows lo..hi from partials `g` [world][hi - lo][D] (a gathered room or a split one's own rows).
    fn postRows(f: *const Forward, r: Rank, lo: usize, hi: usize, g: u64) !void {
        const c = &r.w.config;
        const b = r.b;
        const S: usize = c.hc;
        const D: usize = c.hidden;
        const x = b.x + lo * S * D * 2;
        try glue.hcPost(f.set(), f.stream, x, x, g, b.post + lo * S * 4, b.comb + lo * S * S * 4, hi - lo, D, S, r.w.share.world);
    }

    /// forward._pre of rows lo..hi.
    fn preRows(f: *const Forward, r: Rank, hc: weights.Hc, norm: glue.Norm, lo: usize, hi: usize) !void {
        const c = &r.w.config;
        const b = r.b;
        const S: usize = c.hc;
        const D: usize = c.hidden;
        try glue.hcPre(f.set(), f.stream, b.x + lo * S * D * 2, hc.fn_w, hc.base, hc.scale, norm, b.normed + lo * D * 2, b.xs + lo * (D / 64) * 4, b.post + lo * S * 4, b.comb + lo * S * S * 4, b.hcpart, c.eps, c.hc_eps, c.sinkhorn, hi - lo, D, S);
    }

    /// One piece's glue after a site: hc_post, then the next layer's hc_pre or (after the last layer) the stream mean.
    fn glueRows(f: *const Forward, r: Rank, j: usize, site: Site, lo: usize, hi: usize, g: u64) !void {
        const c = &r.w.config;
        const layers = r.w.layers;
        try f.postRows(r, lo, hi, g);
        if (site == .mixer) return f.preRows(r, layers[j].ffn_hc, layers[j].post_norm, lo, hi);
        try f.captureTap(r, j, lo, hi);
        if (j + 1 < layers.len) return f.preRows(r, layers[j + 1].attn_hc, layers[j + 1].in_norm, lo, hi);
        try glue.streamMean(f.set(), f.stream, r.b.x + lo * c.hc * c.hidden * 2, r.b.hidden + lo * c.hidden * 2, hi - lo, c.hidden, c.hc);
    }

    /// overlap.Overlap.glue: each piece's glue once its exchange is in, its rows then shared on the second stream.
    fn gluePieces(f: *const Forward, p: *const Pieces, j: usize, site: Site, cut: []const [2]usize) !void {
        const c = &f.ranks[0].w.config;
        const D: usize = c.hidden;
        const world = f.ranks[0].w.share.world;
        const last = site == .ffn and j + 1 == f.ranks[0].w.layers.len;
        var sent: usize = 0;
        for (cut, 0..) |piece, q| {
            const lo = piece[0];
            const n = piece[1] - lo;
            try f.stream.wait(p.gathered[q]);
            if (!f.splits(n)) {
                for (f.ranks) |r| try f.glueRows(r, j, site, lo, piece[1], room(r, lo));
                continue;
            }
            for (f.ranks) |r| {
                const mine = share(n, world, r.w.share.rank);
                try f.glueRows(r, j, site, lo + mine[0], lo + mine[1], room(r, lo));
            }
            try p.glues[q].record(f.stream);
            try p.side.wait(p.glues[q]);
            switch (f.exchange) {
                .local => for (f.ranks) |from| {
                    const mine = share(n, world, from.w.share.rank);
                    const bytes = (mine[1] - mine[0]) * D * 2;
                    for (f.ranks) |to| if (to.w.share.rank != from.w.share.rank) {
                        const at = (lo + mine[0]) * D * 2;
                        try f.copy(p.side, (if (last) to.b.hidden else to.b.normed) + at, (if (last) from.b.hidden else from.b.normed) + at, bytes);
                        if (site == .ffn) if (taps_mod.index(j)) |plane| if (from.b.taps) |source| {
                            const destination = to.b.taps orelse return error.BadTapView;
                            try f.copy(p.side, destination.ptr + plane * to.b.rows * D * 2 + at, source.ptr + plane * from.b.rows * D * 2 + at, bytes);
                        };
                    };
                },
                .nccl => |x| {
                    const r = f.ranks[0];
                    const me = r.w.share.rank;
                    const rows_of = if (last) r.b.hidden else r.b.normed;
                    const plan: relay.Relay = .{ .world = world, .rank = me, .n = n };
                    if (f.ring) try relay.glued(x.lib, x.comm, p.side, plan, rows_of, lo, D) else try relay.splitGlued(x.lib, x.comm, p.side, plan, rows_of, lo, D);
                    if (site == .ffn) if (taps_mod.index(j)) |plane| if (r.b.taps) |storage| {
                        const base = storage.ptr + plane * r.b.rows * D * 2;
                        if (f.ring) try relay.glued(x.lib, x.comm, p.side, plan, base, lo, D) else try relay.splitGlued(x.lib, x.comm, p.side, plan, base, lo, D);
                    };
                },
            }
            try p.shared[q].record(p.side);
            sent |= @as(usize, 1) << @intCast(q);
        }
        for (0..cut.len) |q| if (sent >> @intCast(q) & 1 == 1) try f.stream.wait(p.shared[q]);
    }

    /// forward.pieced_layers: a prompt chunk's layers with exchanges in row pieces; same kernels and bits as unpieced.
    fn pieced(f: *const Forward, p: *const Pieces, R: usize, cut: []const [2]usize) !void {
        const layers = f.ranks[0].w.layers;
        for (f.ranks) |r| try f.pre(r, r.w.layers[0].attn_hc, r.w.layers[0].in_norm, R);
        var li: usize = 0;
        var di: usize = 0;
        for (layers, 0..) |_, j| {
            for (f.ranks) |r| if (r.segs.len > 0) try shared.mixer(f, r, r.w.layers[j], li, di, R, null) else try f.mixer(r, r.w.layers[j], li, di, R, @min(latent.chunksFor(r.st.pos + R), r.b.nch), null);
            try f.exchangePieces(p, j, .mixer, cut);
            try f.gluePieces(p, j, .mixer, cut);
            for (f.ranks) |r| try f.ffn(r, r.w.layers[j], R, null);
            try f.exchangePieces(p, j, .ffn, cut);
            try f.gluePieces(p, j, .ffn, cut);
            try f.publishTaps(j, R);
            switch (layers[j].mixer) {
                .kda => li += 1,
                .mla => di += 1,
            }
        }
    }

    fn frontRows(f: *const Forward, r: Rank, l: weights.Layer, site: Site, lo: usize, hi: usize) !void {
        if (hi <= lo) return;
        const b = r.b;
        const D = r.w.config.hidden;
        const n = hi - lo;
        switch (site) {
            .mixer => switch (l.mixer) {
                .kda => |k| {
                    var v = b.kda;
                    v.p = b.kproj + lo * k.width() * 2;
                    v.ka += lo * k.heads * kda.dk * 2;
                    v.kg += lo * k.heads * kda.dk * 2;
                    try kda_block.project(f.set(), f.stream, k, v, b.normed + lo * D * 2, D, n);
                },
                .mla => |a| {
                    var v = b.dsa;
                    v.qr += lo * a.q_lora * 2;
                    v.xs_qr += lo * (a.q_lora / 64) * 4;
                    v.lat += lo * a.lw * 2;
                    v.xs_lat += lo * (a.lw / 64) * 4;
                    v.q += lo * a.heads * a.qk * 2;
                    try dsa_block.front(f.set(), f.stream, a, v, b.normed + lo * D * 2, D, n);
                },
            },
            .ffn => switch (l.ffn) {
                .dense => |m| {
                    var v = b.dense;
                    v.gu += lo * 2 * m.width * 2;
                    v.act += lo * m.width * 2;
                    v.xs_act += lo * (m.width / 64) * 4;
                    try mlp.body(f.mlpKernels(), f.stream, m, v, b.normed + lo * D * 2, D, n, true);
                },
                .moe => |m| try moe.shared(f.moeKernels(), f.stream, m, b.moe, b.normed, D, lo, hi),
            },
        }
    }

    fn frontRemainder(f: *const Forward, r: Rank, l: weights.Layer, site: Site, rows: usize, done: ?prompt_split.Range) !void {
        if (done) |range| {
            try f.frontRows(r, l, site, 0, @min(range.lo, rows));
            try f.frontRows(r, l, site, @min(range.hi, rows), rows);
        } else try f.frontRows(r, l, site, 0, rows);
    }

    fn promptFfn(f: *const Forward, r: Rank, l: weights.Layer, rows: usize, done: ?prompt_split.Range) !void {
        if (l.ffn == .moe) try moe.body(f.moeKernels(), f.stream, l.ffn.moe, r.b.moe, r.b.normed, r.w.config.hidden, rows, true);
        try f.frontRemainder(r, l, .ffn, rows, done);
    }

    fn promptFill(f: *const Forward, r: Rank, l: weights.Layer, site: Site, lo: usize, hi: usize) !void {
        if (hi <= lo) return;
        const out = r.b.part + lo * r.w.config.hidden * 4;
        if (site == .ffn and l.ffn == .moe) {
            try moe.finish(f.moeKernels(), f.stream, l.ffn.moe, r.b.moe, lo, hi, true, out);
        } else try f.fillRows(r, l, site, lo, hi, out);
    }

    fn promptPlan(f: *const Forward, r: Rank, rows: usize) !prompt_split.Plan {
        return prompt_split.Plan.init(rows, r.b.rows, r.w.share.rank, f.prefill_pieces);
    }

    fn exchangePrompt(f: *const Forward, p: *const Pieces, j: usize, site: Site, rows: usize) !void {
        const first = try f.promptPlan(f.ranks[0], rows);
        const D = f.ranks[0].w.config.hidden;
        for (first.pieces[0..first.count], 0..) |piece, q| {
            for (f.ranks) |r| {
                const plan = try f.promptPlan(r, rows);
                const lo = plan.peer + piece.off;
                try f.promptFill(r, r.w.layers[j], site, @min(lo, rows), @min(lo + piece.rows, rows));
            }
            try p.filled[q].record(f.stream);
            try p.side.wait(p.filled[q]);
            const bytes = piece.rows * D * 4;
            switch (f.exchange) {
                .local => for (f.ranks) |from| {
                    const plan = try f.promptPlan(from, rows);
                    for (f.ranks) |to| if (to.w.share.rank != from.w.share.rank) {
                        try f.copy(p.side, room(to, piece.off) + from.w.share.rank * bytes, from.b.part + (plan.peer + piece.off) * D * 4, bytes);
                    };
                },
                .nccl => |x| {
                    const r = f.ranks[0];
                    const plan = try f.promptPlan(r, rows);
                    const peer = 1 - r.w.share.rank;
                    try x.lib.check(x.lib.api.ncclGroupStart(), "ncclGroupStart");
                    try x.lib.check(x.lib.api.ncclSend(r.b.part + (plan.peer + piece.off) * D * 4, piece.rows * D, .f32, @intCast(peer), x.comm, p.side.handle), "ncclSend");
                    try x.lib.check(x.lib.api.ncclRecv(room(r, piece.off) + peer * bytes, piece.rows * D, .f32, @intCast(peer), x.comm, p.side.handle), "ncclRecv");
                    try x.lib.check(x.lib.api.ncclGroupEnd(), "ncclGroupEnd");
                },
            }
            try p.gathered[q].record(p.side);
        }
        for (f.ranks) |r| {
            const own = (try f.promptPlan(r, rows)).owned();
            try f.promptFill(r, r.w.layers[j], site, own.lo, own.hi);
        }
    }

    fn gluePrompt(f: *const Forward, p: *const Pieces, j: usize, site: Site, rows: usize) !void {
        const first = try f.promptPlan(f.ranks[0], rows);
        const D = f.ranks[0].w.config.hidden;
        const last = site == .ffn and j + 1 == f.ranks[0].w.layers.len;
        for (first.pieces[0..first.count], 0..) |piece, q| {
            try f.stream.wait(p.gathered[q]);
            for (f.ranks) |r| {
                const plan = try f.promptPlan(r, rows);
                const lo = plan.mine + piece.off;
                const bytes = piece.rows * D * 4;
                const base = room(r, piece.off);
                try f.copy(f.stream, base + r.w.share.rank * bytes, r.b.part + lo * D * 4, bytes);
                try f.glueRows(r, j, site, lo, lo + piece.rows, base);
            }
            try p.glues[q].record(f.stream);
            try p.side.wait(p.glues[q]);
            switch (f.exchange) {
                .local => for (f.ranks) |from| {
                    const plan = try f.promptPlan(from, rows);
                    const at = (plan.mine + piece.off) * D * 2;
                    const bytes = piece.rows * D * 2;
                    for (f.ranks) |to| if (to.w.share.rank != from.w.share.rank) {
                        try f.copy(p.side, (if (last) to.b.hidden else to.b.normed) + at, (if (last) from.b.hidden else from.b.normed) + at, bytes);
                        if (site == .ffn) if (taps_mod.index(j)) |plane| if (from.b.taps) |src| {
                            const dst = to.b.taps orelse return error.BadTapView;
                            try f.copy(p.side, dst.ptr + plane * to.b.rows * D * 2 + at, src.ptr + plane * from.b.rows * D * 2 + at, bytes);
                        };
                    };
                },
                .nccl => |x| {
                    const r = f.ranks[0];
                    const plan = try f.promptPlan(r, rows);
                    const mine = plan.mine + piece.off;
                    const peer_at = plan.peer + piece.off;
                    const peer = 1 - r.w.share.rank;
                    const buffer = if (last) r.b.hidden else r.b.normed;
                    try x.lib.check(x.lib.api.ncclGroupStart(), "ncclGroupStart");
                    try x.lib.check(x.lib.api.ncclSend(buffer + mine * D * 2, piece.rows * D, .bf16, @intCast(peer), x.comm, p.side.handle), "ncclSend");
                    try x.lib.check(x.lib.api.ncclRecv(buffer + peer_at * D * 2, piece.rows * D, .bf16, @intCast(peer), x.comm, p.side.handle), "ncclRecv");
                    if (site == .ffn) if (taps_mod.index(j)) |plane| if (r.b.taps) |storage| {
                        const base = storage.ptr + plane * r.b.rows * D * 2;
                        try x.lib.check(x.lib.api.ncclSend(base + mine * D * 2, piece.rows * D, .bf16, @intCast(peer), x.comm, p.side.handle), "ncclSend tap");
                        try x.lib.check(x.lib.api.ncclRecv(base + peer_at * D * 2, piece.rows * D, .bf16, @intCast(peer), x.comm, p.side.handle), "ncclRecv tap");
                    };
                    try x.lib.check(x.lib.api.ncclGroupEnd(), "ncclGroupEnd");
                },
            }
            try p.shared[q].record(p.side);
            if (f.prefill_overlap == 2 and !last) for (f.ranks) |r| {
                const plan = try f.promptPlan(r, rows);
                const lo = plan.mine + piece.off;
                const next = if (site == .mixer) j else j + 1;
                try f.frontRows(r, r.w.layers[next], if (site == .mixer) .ffn else .mixer, @min(lo, rows), @min(lo + piece.rows, rows));
                const kind: usize = if (site == .mixer) switch (r.w.layers[next].ffn) {
                    .dense => 2,
                    .moe => 3,
                } else switch (r.w.layers[next].mixer) {
                    .kda => 0,
                    .mla => 1,
                };
                _ = front_dispatch[kind].fetchAdd(@min(lo + piece.rows, rows) - @min(lo, rows), .monotonic);
            };
        }
        for (0..first.count) |q| try f.stream.wait(p.shared[q]);
    }

    fn productionPrompt(f: *const Forward, p: *const Pieces, rows: usize) !void {
        errdefer f.target_failure(error.PromptOverlapFailed);
        const layers = f.ranks[0].w.layers;
        for (f.ranks) |r| {
            const plan = try f.promptPlan(r, rows);
            if (plan.padded != rows) {
                try f.d.check(f.d.api.cuMemsetD8Async(r.b.x + rows * r.w.config.hc * r.w.config.hidden * 2, 0, r.w.config.hc * r.w.config.hidden * 2, f.stream.handle), "cuMemsetD8Async");
                try f.d.check(f.d.api.cuMemsetD8Async(r.b.part + rows * r.w.config.hidden * 4, 0, r.w.config.hidden * 4, f.stream.handle), "cuMemsetD8Async");
            }
            try f.pre(r, r.w.layers[0].attn_hc, r.w.layers[0].in_norm, plan.padded);
        }
        var li: usize = 0;
        var di: usize = 0;
        for (layers, 0..) |_, j| {
            for (f.ranks) |r| {
                const done: ?prompt_split.Range = if (f.prefill_overlap == 2 and j > 0) (try f.promptPlan(r, rows)).owned() else null;
                try f.mixerPrepared(r, r.w.layers[j], li, di, rows, @min(latent.chunksFor(r.st.pos + rows), r.b.nch), null, done);
            }
            try f.exchangePrompt(p, j, .mixer, rows);
            try f.gluePrompt(p, j, .mixer, rows);
            for (f.ranks) |r| try f.promptFfn(r, r.w.layers[j], rows, if (f.prefill_overlap == 2) (try f.promptPlan(r, rows)).owned() else null);
            try f.exchangePrompt(p, j, .ffn, rows);
            try f.gluePrompt(p, j, .ffn, rows);
            try f.publishTaps(j, rows);
            switch (layers[j].mixer) {
                .kda => li += 1,
                .mla => di += 1,
            }
        }
    }

    fn beginRows(f: *const Forward, rows: usize) !void {
        for (f.ranks) |r| {
            if (rows == 0 or rows > r.b.rows) return error.BadWindow;
            if (r.segs.len == 0) {
                if (r.st.pos > r.st.capacity or rows > r.st.capacity - r.st.pos) return error.PastCapacity;
                r.st.pending = rows;
                r.st.prompt = r.b.prefill;
            } else try shared.begin(r, rows);
        }
    }

    /// All request-dependent host work stays outside the complete target graph.
    pub fn prepareTarget(f: *const Forward, ids: []const i32) !void {
        try f.beginRows(ids.len);
        for (f.ranks) |r| {
            try cuda.copy_source.enqueue(f.d, r.b.ids, std.mem.sliceAsBytes(ids), f.stream);
            if (f.vision) |binding| {
                const target: vision_binding.Target = .{
                    .stream = f.stream,
                    .context = binding.owner.context,
                    .embedding = .{ .d = f.d, .ptr = r.b.x, .len = r.b.rows * 4 * r.w.config.hidden * 2, .borrowed = true },
                    .start = r.st.pos,
                    .capacity = vision_binding.logicalCapacity(r.st.capacity),
                    .window = r.b.rows,
                    .vocab = r.w.config.vocab,
                    .world = @intCast(r.w.share.world),
                    .rank = @intCast(r.w.share.rank),
                };
                const plan = try vision_binding.plan(binding, target, ids);
                try vision_binding.checkContext(f.stream, target.context);
                // Feature injection belongs to eager prompt admission, not a target verification window.
                if (plan.count != 0) return error.TargetGraphContainsVisionPrompt;
            }
        }
    }

    /// Embedding, every target layer and the head, with no sequence-dependent host selection.
    pub fn computeTarget(f: *const Forward, runtime: *target_forward.Runtime, rows: usize) !void {
        const c = &f.ranks[0].w.config;
        for (f.ranks) |r| try glue.embedB16(f.set(), f.stream, r.b.ids, r.w.embed, r.b.x, rows, c.hidden, c.hc);
        var ki: usize = 0;
        var di: usize = 0;
        for (f.ranks[0].w.layers, 0..) |layer, index| {
            for (f.ranks) |r| {
                try f.pre(r, layer.attn_hc, layer.in_norm, rows);
                switch (layer.mixer) {
                    .kda => |k| try runtime.kda(f.set(), f.stream, r.b, k, ki, rows, slot(r, rows), f.kda_wide, f.outputPrefetch(index, rows)),
                    .mla => |a| try runtime.dsa(f.set(), f.stream, f.exchange, r.b, a, di, rows, slot(r, rows), f.outputPrefetch(index, rows)),
                }
            }
            try f.prefetchSite(index, .a, rows);
            try f.gather(rows);
            try f.ffnSite(index, rows);
            switch (layer.mixer) {
                .kda => ki += 1,
                .mla => di += 1,
            }
        }
        for (f.ranks) |r| try glue.streamMean(f.set(), f.stream, r.b.x, r.b.hidden, rows, c.hidden, c.hc);
        try f.finish(rows);
        try f.joinPrefetch();
    }

    /// forward.forward of `ids`: logits in b.logits; a prompt chunk's KDA layers committed, a window's at `commit`.
    pub fn run(f: *const Forward, ids: []const i32) !void {
        if (f.full_target) |runtime| {
            runtime.run(f, ids) catch |err| {
                f.target_failure(err);
                return err;
            };
            return;
        }
        const R = ids.len;
        try f.beginRows(R);
        for (f.ranks) |r| try cuda.copy_source.enqueue(f.d, r.b.ids, std.mem.sliceAsBytes(ids), f.stream);
        const c = &f.ranks[0].w.config;
        for (f.ranks) |r| try glue.embedB16(f.set(), f.stream, r.b.ids, r.w.embed, r.b.x, R, c.hidden, c.hc);
        if (f.vision) |binding| for (f.ranks) |r| try vision_binding.inject(binding, .{
            .stream = f.stream,
            .context = binding.owner.context,
            .embedding = .{ .d = f.d, .ptr = r.b.x, .len = r.b.rows * 4 * c.hidden * 2, .borrowed = true },
            .start = r.st.pos,
            .capacity = vision_binding.logicalCapacity(r.st.capacity),
            .window = r.b.rows,
            .vocab = c.vocab,
            .world = @intCast(r.w.share.world),
            .rank = @intCast(r.w.share.rank),
        }, ids);
        if (f.images) |im| for (f.ranks) |r| try im.v.splice(f.stream, im.slot, r.st.pos, r.st.pos + R, r.b.x, c.hc);
        if (f.prof) |p| p.mark(f.stream, .embed);
        var cut: [max_pieces][2]usize = undefined;
        const pieces = if (f.pieces) |p| (if (f.ranks[0].b.prefill and f.ranks[0].segs.len == 0) ranges(R, p.count, &cut) else 0) else 0;
        // overlap.Overlap.begin: pieced with two pieces or more, and split takes a chunk of one piece too
        const exact_prompt = f.ranks[0].b.prefill and f.ranks[0].segs.len == 0 and f.pieces != null and !f.ring and prompt_split.applies(f.prefill_overlap, f.ranks[0].w.share.world, R, f.ranks[0].b.rows);
        if (exact_prompt) {
            try f.productionPrompt(f.pieces.?, R);
            _ = overlap_dispatch[f.prefill_overlap - 1].fetchAdd(1, .monotonic);
        } else if (pieces > 1 or (pieces == 1 and f.splits(R))) try f.pieced(f.pieces.?, R, cut[0..pieces]) else try f.whole(R);
        try f.finish(R);
        if (f.prof) |p| p.mark(f.stream, .head);
        try f.joinPrefetch();
    }

    /// The unpieced loop over the layers (and the stream mean): each site's partials whole, then the glue.
    fn whole(f: *const Forward, R: usize) !void {
        const c = &f.ranks[0].w.config;
        var li: usize = 0;
        var di: usize = 0;
        for (f.ranks[0].w.layers, 0..) |_, j| {
            for (f.ranks) |r| {
                const l = r.w.layers[j];
                try f.pre(r, l.attn_hc, l.in_norm, R);
                if (r.segs.len > 0) try shared.mixer(f, r, l, li, di, R, slot(r, R)) else try f.mixer(r, l, li, di, R, @min(latent.chunksFor(r.st.pos + R), r.b.nch), slot(r, R));
            }
            if (f.prof) |p| p.mark(f.stream, if (f.ranks[0].w.layers[j].mixer == .kda) .kda else .mla);
            try f.prefetchSite(j, .a, R);
            try f.gather(R);
            if (f.prof) |p| p.mark(f.stream, .gather_a);
            try f.ffnLayer(j, R);
            if (f.prof) |p| p.mark(f.stream, .gather_b);
            switch (f.ranks[0].w.layers[j].mixer) {
                .kda => li += 1,
                .mla => di += 1,
            }
        }
        for (f.ranks) |r| try glue.streamMean(f.set(), f.stream, r.b.x, r.b.hidden, R, c.hidden, c.hc);
    }

    fn ffnLayer(f: *const Forward, layer: usize, rows: usize) !void {
        if (f.target_graphs) |bank| {
            if (!f.ranks[0].b.prefill and f.route_trace == null and rows <= target_graph.max_rows) {
                bank.run(f.stream, try f.graphKey(layer, rows), f, ffnSite) catch |err| {
                    // A peer may already be capturing or replaying NCCL. Never fall back on just this rank.
                    f.target_failure(err);
                    return err;
                };
                return;
            }
        }
        try f.ffnSite(layer, rows);
    }

    /// Only engine-owned scratch and fixed weights: no sequence pointers, positions, parities or KV extents.
    fn ffnSite(f: *const Forward, layer: usize, rows: usize) !void {
        for (f.ranks) |r| {
            const l = r.w.layers[layer];
            try f.post(r, rows);
            try f.pre(r, l.ffn_hc, l.post_norm, rows);
            try f.ffn(r, l, rows, slot(r, rows));
            if (f.route_trace) |trace| switch (l.ffn) {
                .moe => |m| try trace.capture(f.stream, layer, m.count, m.top_k, r.b.moe.plan),
                .dense => {},
            };
        }
        if (f.prof) |p| p.mark(f.stream, if (f.ranks[0].w.layers[layer].ffn == .dense) .dense else .shared);
        try f.prefetchSite(layer, .f, rows);
        try f.gather(rows);
        for (f.ranks) |r| {
            try f.post(r, rows);
            try f.captureTap(r, layer, 0, rows);
        }
        try f.publishTaps(layer, rows);
        if (f.full_target == null) try f.joinPrefetch();
    }

    pub fn prefetchSite(f: *const Forward, layer: usize, which: l2pf_plan.Site, rows: usize) !void {
        if (f.prefetch) |p| try p.site(f.stream, layer, which, rows, f.ranks[0].b.prefill);
    }

    fn outputPrefetch(f: *const Forward, layer: usize, rows: usize) ?l2pf.Hook {
        return if (f.prefetch) |p| .{ .runtime = p, .layer = layer, .rows = rows, .prompt = f.ranks[0].b.prefill } else null;
    }

    fn joinPrefetch(f: *const Forward) !void {
        if (f.prefetch) |p| try p.join(f.stream);
    }

    fn captureTap(f: *const Forward, r: Rank, layer: usize, lo: usize, hi: usize) !void {
        const storage = r.b.taps orelse return;
        const plane = taps_mod.index(layer) orelse return;
        const c = &r.w.config;
        if (c.hidden != taps_mod.hidden or hi > r.b.rows or lo > hi) return error.BadTapView;
        try glue.streamMean(f.set(), f.stream, r.b.x + lo * c.hc * c.hidden * 2, storage.ptr + (plane * r.b.rows + lo) * c.hidden * 2, hi - lo, c.hidden, c.hc);
    }

    fn publishTaps(f: *const Forward, layer: usize, rows: usize) !void {
        const target = f.taps orelse return;
        const plane = taps_mod.index(layer) orelse return;
        const r = f.ranks[0];
        const source = r.b.taps orelse return error.BadTapView;
        const D = r.w.config.hidden;
        if (D != taps_mod.hidden or rows > r.b.rows) return error.BadTapView;
        try projection.copyRows(f.set(), f.stream, source.ptr + plane * r.b.rows * D * 2, D * 2, target.out + plane * D * 2, taps_mod.layers.len * D * 2, rows, D * 2);
    }

    /// The final norm and this rank's rows of the head (a prompt buffer's last row only).
    fn finish(f: *const Forward, R: usize) !void {
        const c = &f.ranks[0].w.config;
        for (f.ranks) |r| {
            const b = r.b;
            try glue.rmsnormBy(f.set(), f.stream, b.hidden, c.hidden, r.w.norm, b.fnormed, c.hidden, b.fxs, c.eps, R, c.hidden);
            if (b.prefill and r.segs.len > 0) {
                try shared.heads(f, r);
                continue;
            }
            const head_rows: usize = if (b.prefill) 1 else R;
            const first = if (b.prefill) R - 1 else 0;
            try projection.matmul(f.set(), f.stream, r.w.head, b.fnormed + first * c.hidden * 2, c.hidden, b.logits, b.kda.mm_part, head_rows, r.w.share.vocab.len(), c.hidden, false);
        }
    }

    /// mtp.mtp_forward: caches from mtp_len, stream in b.mx, a window's draft logits in b.logits (prompt: no head).
    pub fn mtp(f: *const Forward, ids: []const i32, hidden: []const u64) !void {
        const n = ids.len;
        const c = &f.ranks[0].w.config;
        const D: usize = c.hidden;
        for (f.ranks, hidden) |r, h| {
            if (r.w.mtp == null) return error.NoMtpHead;
            if (n == 0 or n > r.b.rows) return error.BadWindow;
            if (r.st.mtp_len + n > r.st.capacity) return error.PastCapacity;
            try cuda.copy_source.enqueue(f.d, r.b.ids, std.mem.sliceAsBytes(ids), f.stream);
            if (h != r.b.hin) try f.d.check(f.d.api.cuMemcpyDtoDAsync_v2(r.b.hin, h, n * D * 2, f.stream.handle), "cuMemcpyDtoDAsync");
            const m = r.w.mtp.?;
            const b = r.b;
            try glue.embedB16(f.set(), f.stream, b.ids, r.w.embed, b.me, n, D, 1);
            if (r.st.mtp_len == 0) try f.d.check(f.d.api.cuMemsetD8Async(b.me, 0, D * 2, f.stream.handle), "cuMemsetD8Async"); // position 0's embedding is zeroed
            if (f.images) |im| try im.v.splice(f.stream, im.slot, r.st.mtp_len + 1, r.st.mtp_len + 1 + n, b.me, 1);
            try glue.rmsnorm(f.set(), f.stream, b.me, D, m.enorm, b.mcat, 2 * D, null, c.eps, n, D);
            try glue.rmsnorm(f.set(), f.stream, b.hin, D, m.hnorm, b.mcat + D * 2, 2 * D, null, c.eps, n, D);
            try projection.matmul(f.set(), f.stream, m.eh, b.mcat, 2 * D, b.mx, b.kda.mm_part, n, D, 2 * D, false);
            try glue.rmsnormBy(f.set(), f.stream, b.mx, D, m.layer.in_norm, b.normed, D, b.xs, c.eps, n, D);
            const nch = @min(latent.chunksFor(r.st.mtp_len + n), b.nch);
            var ks = f.dsaKernels(r);
            ks.slot = 12;
            ks.set.prompt_aot = null;
            try dsa_block.block(ks, f.stream, m.layer.mixer.mla, r.st.mtp_caches, b.dsa, b.normed, D, n, r.st.mtp_pos_dev, r.st.mtp_len, nch, b.part_rows, slot(r, n));
        }
        try f.gather(n);
        for (f.ranks) |r| {
            const m = r.w.mtp.?;
            const b = r.b;
            try glue.residualAdd(f.set(), f.stream, b.mx, b.mx, b.gathered, n, D, r.w.share.world);
            try glue.rmsnormBy(f.set(), f.stream, b.mx, D, m.layer.post_norm, b.normed, D, b.xs, c.eps, n, D);
            try f.ffn(r, m.layer, n, slot(r, n));
        }
        try f.gather(n);
        for (f.ranks) |r| {
            const m = r.w.mtp.?;
            const b = r.b;
            try glue.residualAdd(f.set(), f.stream, b.mx, b.mx, b.gathered, n, D, r.w.share.world);
            try glue.rmsnorm(f.set(), f.stream, b.mx + (n - 1) * D * 2, D, m.norm, b.fnormed, D, b.fxs, c.eps, 1, D);
            if (b.prefill) continue;
            if (m.draft) |dq| try q4.matmul(&f.m.q4, f.stream, b.fnormed, D, b.fxs, dq, b.logits, 1) else try @import("cuda_proj.zig").matmul(f.set(), f.stream, r.w.head, b.fnormed, D, b.logits, b.kda.mm_part, 1, r.w.share.vocab.len(), D, false);
        }
    }

    /// forward.commit: keep the first `keep` rows; window KDA layers take new states (replayed if fewer), conv shifted.
    pub fn commit(f: *const Forward, keep: usize) !void {
        for (f.ranks) |r| {
            const st = r.st;
            if (keep == 0 or keep > st.pending or (st.prompt and keep != st.pending)) return error.BadCommit;
            if (!st.prompt and st.kda_layers > 0) {
                const cur = st.cur[0];
                for (st.cur) |x| if (x != cur) return error.KdaLayersOutOfStep;
                if (keep < st.pending) {
                    const h = st.heads;
                    var a: cuda.Args = .{};
                    a.add(@as(c_int, @intCast(h)));
                    a.add(st.recAt(cur, 0));
                    a.add(@as(usize, st.rec_layer / 4));
                    for ([_]u64{ st.save_k, st.save_v, st.save_g, st.save_b }) |x| a.add(x);
                    a.add(@as(usize, window_rows * h * kda.dk));
                    a.add(@as(usize, window_rows * h));
                    a.add(@as(c_int, @intCast(keep)));
                    a.add(st.recAt(1 - cur, 0));
                    try cuda.launch.launch(f.m.kda.replay_layers, .{ .grid = .{ .x = @intCast(st.kda_layers * h) }, .block = .{ .x = 1024 } }, f.stream, &a);
                }
                for (st.cur) |*x| x.* = 1 - x.*;
                if (st.kda_layers > 0) try glue.convShift(f.set(), f.stream, st.conv, st.proj, keep, st.proj_layer, st.proj_row, st.kda_layers, st.conv_c);
            }
            st.pending = 0;
            try st.setPos(f.stream, st.pos + keep);
        }
    }
};

test "actual forward and MTP bodies refuse invalid input before launches or token upload" {
    const Trace = struct {
        var submissions: usize = 0;
        fn launch(_: cuda.abi.Function, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: cuda.abi.Stream, _: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
            submissions += 1;
            return 2;
        }
        fn upload(_: u64, _: ?*const anyopaque, _: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
            submissions += 1;
            return 2;
        }
    };
    const Memory = @import("cuda_test_driver.zig").Memory;
    var d = Memory.driver();
    Trace.submissions = 0;
    d.api.cuLaunchKernel = Trace.launch;
    d.api.cuMemcpyHtoDAsync_v2 = Trace.upload;
    var b: Buffers = undefined;
    b.rows = 8;
    var st: State = undefined;
    var w: weights.Weights = .{ .gpa = std.testing.allocator, .config = .{}, .share = undefined };
    var modules: Modules = undefined;
    const ranks = [_]Rank{.{ .w = &w, .st = &st, .b = &b }};
    const f: Forward = .{ .d = &d, .m = &modules, .stream = .{ .d = &d, .handle = null }, .ranks = &ranks, .exchange = .local };
    try std.testing.expectError(error.BadWindow, f.run(&.{}));
    // The MTP refusal checks the actual optional head before accessing sequence or device memory.
    try std.testing.expectError(error.NoMtpHead, f.mtp(&.{}, &.{0}));
    try std.testing.expectEqual(@as(usize, 0), Trace.submissions);
    try std.testing.expect(Memory.empty());
}

test {
    _ = prompt_split;
}

// Records the actual copyRows submissions used by publishTaps and their captured replay destinations.
const CaptureRecording = struct {
    const Command = struct { src: u64, src_stride: i64, dst: u64, dst_stride: i64, rows: i32, bytes: i32 };
    var capturing: bool = false;
    var pending: ?Command = null;
    var graphs: [64]?Command = @splat(null);
    var execs: [64]?Command = @splat(null);
    var graph_count: usize = 0;
    var exec_count: usize = 0;
    var freed: usize = 0;
    var launches: usize = 0;
    var writes: [128]Command = undefined;
    var written: usize = 0;
    var inflight: bool = false;
    var syncs: usize = 0;
    var events: usize = 0;

    fn driver() cuda.Driver {
        capturing = false;
        pending = null;
        @memset(&graphs, null);
        @memset(&execs, null);
        graph_count = 0;
        exec_count = 0;
        freed = 0;
        launches = 0;
        written = 0;
        inflight = false;
        syncs = 0;
        events = 0;
        var d: cuda.Driver = .{ .lib = undefined, .api = undefined };
        d.api.cuStreamSynchronize = sync;
        d.api.cuStreamBeginCapture_v2 = begin;
        d.api.cuStreamEndCapture = end;
        d.api.cuGraphInstantiateWithFlags = instantiate;
        d.api.cuGraphDestroy = destroyGraph;
        d.api.cuGraphExecDestroy = destroyExec;
        d.api.cuGraphLaunch = replay;
        d.api.cuLaunchKernel = kernel;
        d.api.cuEventRecord = event;
        return d;
    }
    fn execute(c: Command) void {
        writes[written] = c;
        written += 1;
        inflight = true;
    }
    fn sync(_: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        syncs += 1;
        inflight = false;
        return 0;
    }
    fn begin(_: cuda.abi.Stream, _: cuda.abi.CaptureMode) callconv(.c) cuda.abi.Result {
        std.debug.assert(!capturing and !inflight);
        capturing = true;
        pending = null;
        return 0;
    }
    fn end(_: cuda.abi.Stream, out: *cuda.abi.Graph) callconv(.c) cuda.abi.Result {
        std.debug.assert(capturing);
        capturing = false;
        graph_count += 1;
        graphs[graph_count] = pending;
        out.* = @ptrFromInt(graph_count);
        return 0;
    }
    fn instantiate(out: *cuda.abi.GraphExec, graph: cuda.abi.Graph, _: u64) callconv(.c) cuda.abi.Result {
        exec_count += 1;
        execs[exec_count] = graphs[@intFromPtr(graph)];
        out.* = @ptrFromInt(exec_count);
        return 0;
    }
    fn destroyGraph(_: cuda.abi.Graph) callconv(.c) cuda.abi.Result {
        return 0;
    }
    fn destroyExec(_: cuda.abi.GraphExec) callconv(.c) cuda.abi.Result {
        // A replacement cannot release the old executable while its tap write is in flight.
        std.debug.assert(!inflight);
        freed += 1;
        return 0;
    }
    fn replay(exec: cuda.abi.GraphExec, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        launches += 1;
        if (execs[@intFromPtr(exec)]) |c| execute(c);
        return 0;
    }
    fn event(_: cuda.abi.Event, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        if (!capturing) {
            events += 1;
            inflight = true;
        }
        return 0;
    }
    fn kernel(_: cuda.abi.Function, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: cuda.abi.Stream, args: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
        const a = args.?;
        const c: Command = .{
            .src = @as(*const u64, @ptrCast(@alignCast(a[0].?))).*,
            .src_stride = @as(*const i64, @ptrCast(@alignCast(a[1].?))).*,
            .dst = @as(*const u64, @ptrCast(@alignCast(a[2].?))).*,
            .dst_stride = @as(*const i64, @ptrCast(@alignCast(a[3].?))).*,
            .rows = @as(*const i32, @ptrCast(@alignCast(a[4].?))).*,
            .bytes = @as(*const i32, @ptrCast(@alignCast(a[5].?))).*,
        };
        if (capturing) pending = c else execute(c);
        return 0;
    }
    fn compute(f: *const Forward, layer: usize, rows: usize) !void {
        try f.publishTaps(layer, rows);
        if (f.prof) |p| p.mark(f.stream, .shared);
    }
};

test "warmup without taps is a negative control; attached replaced detached and recycled sinks recapture real tap publication" {
    const R = CaptureRecording;
    var d = R.driver();
    const stream: cuda.Stream = .{ .d = &d, .handle = null };
    var m: Modules = undefined;
    m.tiled = false;
    m.packs = null;
    m.copy_rows = .{ .d = &d, .handle = @ptrFromInt(7) };
    var b: Buffers = undefined;
    b.rows = 8;
    b.taps = .{ .d = &d, .ptr = 0x1000, .len = 8 * taps_mod.hidden * 2 * taps_mod.layers.len };
    var w: weights.Weights = .{ .gpa = std.testing.allocator, .config = .{}, .share = undefined };
    w.config.hidden = taps_mod.hidden;
    var st: State = undefined;
    var st2: State = undefined;
    var ranks = [_]Rank{.{ .w = &w, .st = &st, .b = &b }};
    var f: Forward = .{ .d = &d, .m = &m, .stream = stream, .ranks = &ranks, .exchange = .local, .capture_lifetime = target_graph.newLifetime() };
    var bank = try target_graph.Bank.init(std.testing.allocator, 6);
    defer bank.deinit();
    // This frozen key reproduces the old defect at the actual Bank.run boundary, not merely key equality.
    const frozen = try f.graphKey(5, 3);
    try bank.run(stream, frozen, &f, R.compute);
    f.taps = .{ .lifetime = target_graph.newLifetime(), .f = m.copy_rows, .out = 0x200000 };
    try bank.run(stream, frozen, &f, R.compute);
    try std.testing.expectEqual(@as(usize, 0), R.written); // stale warmup really omits the sink
    // The real forward caller supplies the current context. It must not replay that warmup executable.
    try bank.run(stream, try f.graphKey(5, 3), &f, R.compute);
    try std.testing.expectEqual(@as(usize, 1), R.written);
    try bank.retire(stream);
    R.written = 0;
    // Repeat the real owner transitions in a separate bounded bank.
    var tapped = try target_graph.Bank.init(std.testing.allocator, 6);
    defer tapped.deinit();
    f.taps = null;
    try tapped.run(stream, try f.graphKey(5, 3), &f, R.compute);
    f.taps = .{ .lifetime = target_graph.newLifetime(), .f = m.copy_rows, .out = 0x200000 };
    try tapped.run(stream, try f.graphKey(5, 3), &f, R.compute);
    try std.testing.expectEqual(@as(usize, 1), R.written);
    try std.testing.expectEqual(@as(u64, 0x1000), R.writes[0].src);
    try std.testing.expectEqual(@as(i64, 8192), R.writes[0].src_stride);
    try std.testing.expectEqual(@as(i64, 40960), R.writes[0].dst_stride);
    try std.testing.expectEqual(@as(i32, 3), R.writes[0].rows);
    try std.testing.expectEqual(@as(i32, 8192), R.writes[0].bytes);
    const captures = tapped.stats.site_captures;
    ranks[0].st = &st2; // request ownership changes are metadata, not capture context
    try tapped.run(stream, try f.graphKey(5, 3), &f, R.compute);
    try std.testing.expectEqual(captures, tapped.stats.site_captures);
    try std.testing.expectEqual(@as(usize, 2), R.written);
    f.taps.?.out = 0x300000; // replacement, not just absent/present
    try tapped.run(stream, try f.graphKey(5, 3), &f, R.compute);
    try std.testing.expectEqual(@as(u64, 0x300000), R.writes[2].dst);
    f.taps.?.lifetime = target_graph.newLifetime(); // same address, new allocation/module owner
    try tapped.run(stream, try f.graphKey(5, 3), &f, R.compute);
    try std.testing.expectEqual(captures + 2, tapped.stats.site_captures);
    f.taps = null;
    const written = R.written;
    try tapped.run(stream, try f.graphKey(5, 3), &f, R.compute);
    try tapped.run(stream, try f.graphKey(5, 3), &f, R.compute);
    try std.testing.expectEqual(written, R.written);
    try std.testing.expectEqual(captures + 3, tapped.stats.site_captures);
    try tapped.retire(stream);
    try std.testing.expectEqual(R.exec_count, R.freed);
}

test "graph event profile trace rewinds eager marks and restores host metadata on unchanged replay" {
    const R = CaptureRecording;
    var d = R.driver();
    const stream: cuda.Stream = .{ .d = &d, .handle = null };
    var m: Modules = undefined;
    m.tiled = false;
    m.packs = null;
    var b: Buffers = undefined;
    b.rows = 1;
    b.taps = null;
    var w: weights.Weights = .{ .gpa = std.testing.allocator, .config = .{}, .share = undefined };
    var st: State = undefined;
    const ranks = [_]Rank{.{ .w = &w, .st = &st, .b = &b }};
    var f: Forward = .{ .d = &d, .m = &m, .stream = stream, .ranks = &ranks, .exchange = .local, .capture_lifetime = target_graph.newLifetime() };
    var p: prof_mod.Prof = undefined;
    p.capture_lifetime = target_graph.newLifetime();
    p.open = true;
    p.n = 0;
    for (&p.events, 0..) |*event, i| event.* = .{ .d = &d, .handle = @ptrFromInt(i + 1) };
    var other = p;
    other.capture_lifetime = target_graph.newLifetime();
    var bank = try target_graph.Bank.init(std.testing.allocator, 1);
    defer bank.deinit();
    try bank.run(stream, try f.graphKey(0, 1), &f, R.compute); // warmup, no profiler
    f.prof = &p;
    try bank.run(stream, try f.graphKey(0, 1), &f, R.compute);
    try std.testing.expectEqual(@as(usize, 1), p.n); // not eager + capture twice
    try std.testing.expectEqual(prof_mod.Section.shared, p.ends[0]);
    p.n = 0;
    p.ends[0] = .head; // a different call may overwrite this host array
    try bank.run(stream, try f.graphKey(0, 1), &f, R.compute);
    try std.testing.expectEqual(@as(u64, 2), bank.stats.site_captures);
    try std.testing.expectEqual(@as(usize, 1), p.n);
    try std.testing.expectEqual(prof_mod.Section.shared, p.ends[0]);
    f.prof = &other;
    try bank.run(stream, try f.graphKey(0, 1), &f, R.compute);
    try std.testing.expectEqual(@as(u64, 3), bank.stats.site_captures);
    other.n = 0;
    other.capture_lifetime = target_graph.newLifetime(); // recycled same host/event addresses
    try bank.run(stream, try f.graphKey(0, 1), &f, R.compute);
    try std.testing.expectEqual(@as(u64, 4), bank.stats.site_captures);
    other.n = 1; // event cursor is part of capture context too
    try bank.run(stream, try f.graphKey(0, 1), &f, R.compute);
    try std.testing.expectEqual(@as(u64, 5), bank.stats.site_captures);
    f.prof = null;
    try bank.run(stream, try f.graphKey(0, 1), &f, R.compute);
    try bank.run(stream, try f.graphKey(0, 1), &f, R.compute);
    try std.testing.expectEqual(@as(u64, 6), bank.stats.site_captures);
    try bank.retire(stream);
    try std.testing.expectEqual(R.exec_count, R.freed);
}
