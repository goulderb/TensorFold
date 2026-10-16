//! GLM-5.3-Flash's per-rank CUDA memory: forward Buffers (shared by sequences) and a sequence's State, an arena each.

const std = @import("std");
const cuda = @import("cuda");
const weights = @import("cuda_weights.zig");
const layout_mod = @import("cuda_layout.zig");
const cache_rows = @import("cuda_cache.zig");
const kda = @import("cuda_kda.zig");
const b16 = @import("cuda_b16.zig");
const latent = @import("cuda_latent.zig");
const sparse = @import("cuda_sparse.zig");
const experts = @import("cuda_experts.zig");
const dsa_block = @import("cuda_dsa_block.zig");
const moe = @import("cuda_moe.zig");
const glue = @import("cuda_glue.zig");
const mlp = @import("cuda_mlp.zig");
const kda_block = @import("cuda_kda_block.zig");
const precision = @import("cuda_weight_precision.zig");

/// A prompt chunk's rows (decode.PREFILL_ROWS).
pub const prefill_rows = 2048;
/// A decode window's rows (Engine's max_rows).
pub const window_rows = 8;
/// The context the buffers' dense attention partials reach (Buffers' min(capacity, 2560)).
pub const dense_reach = 2560;

/// Sub-allocations of one device allocation, 256-byte aligned like the driver's own.
const Arena = struct {
    buf: cuda.DeviceBuffer,
    used: usize = 0,

    fn take(a: *Arena, bytes: usize) u64 {
        const at = std.mem.alignForward(usize, a.used, 256);
        a.used = at + bytes;
        std.debug.assert(a.used <= a.buf.len);
        return a.buf.ptr + at;
    }
};

/// Allocates `sizes` (bytes) in one zeroed device buffer and hands out each field's address in order.
fn arena(d: *const cuda.Driver, sizes: []const usize, out: []u64) !Arena {
    var total: usize = 0;
    for (sizes) |n| total += std.mem.alignForward(usize, n, 256);
    var a: Arena = .{ .buf = try cuda.DeviceBuffer.alloc(d, @max(total, 256)) };
    errdefer a.buf.free();
    try a.buf.fill8(0, null);
    for (sizes, out) |n, *p| p.* = a.take(n);
    return a;
}

/// A rank's B16 matmul shapes ([n, k]) whose split-K partials the buffers hold.
fn lanePartBytes(rows: usize, n: usize, k: usize) usize {
    const m = @min(rows, cuda.qmmf.fused_rows - 1);
    const plan = cuda.qmmf.plan(m, n, k, false);
    return if (plan.reduce) @as(usize, plan.sk) * m * n * 4 else 0;
}

pub const Shape = struct { group: precision.Group, n: usize, k: usize };

pub fn projectionShapes(w: *const weights.Weights) [16]Shape {
    const c = &w.config;
    const s = &w.share;
    const n = s.kda.len() * c.kda_dim;
    const kw = 3 * n + 2 * c.kda_dim + s.kda.len();
    return .{
        .{ .group = .kda, .n = kw, .k = c.hidden },
        .{ .group = .kda, .n = n, .k = c.kda_dim },
        .{ .group = .kda, .n = c.hidden, .k = n },
        .{ .group = .dsa, .n = c.q_lora + c.kv_lora, .k = c.hidden },
        .{ .group = .dsa, .n = c.q_lora, .k = c.hidden },
        .{ .group = .dsa, .n = c.kv_lora, .k = c.hidden },
        .{ .group = .dsa, .n = s.mla.len() * c.nope, .k = c.q_lora },
        .{ .group = .dsa, .n = c.i_dim + c.i_heads, .k = c.hidden },
        .{ .group = .dsa, .n = c.i_heads * c.i_dim, .k = c.q_lora },
        .{ .group = .dsa, .n = c.hidden, .k = s.mla.len() * c.v_dim },
        .{ .group = .shared, .n = 2 * s.shared.len(), .k = c.hidden },
        .{ .group = .shared, .n = c.hidden, .k = s.shared.len() },
        .{ .group = .mtp_eh, .n = c.hidden, .k = 2 * c.hidden },
        .{ .group = .kda, .n = 2 * c.kda_dim + s.kda.len(), .k = c.hidden },
        .{ .group = .mlp, .n = s.dense.len(), .k = c.hidden },
        .{ .group = .mlp, .n = c.hidden, .k = s.dense.len() },
    };
}

pub fn mixedPartBytes(w: *const weights.Weights, rows: usize) usize {
    var mm: usize = 4;
    for (projectionShapes(w)) |shape| {
        mm = @max(mm, b16.partBytes(@min(rows, b16.rows_from - 1), shape.n, shape.k));
        mm = @max(mm, rows * (shape.k / 64) * 4);
        mm = @max(mm, lanePartBytes(rows, shape.n, shape.k));
    }
    return mm;
}

/// Scratch for windows of up to `rows` rows (the Python family's Buffers): a prompt buffer (`prefill`) or a decode's.
pub const Buffers = struct {
    arena: Arena,
    rows: usize,
    prefill: bool,
    part_rows: usize, // rows one dense attention call takes
    nch: usize, // the dense pass's most 512-key chunks
    ids: u64,
    x: u64,
    normed: u64,
    xs: u64,
    post: u64,
    comb: u64,
    hcpart: u64,
    gathered: u64, // fp32 [world][rows][dims]: every rank's partial in rank order
    kproj: u64, // a prompt chunk's or a shared window's KDA projection rows (a window's stay in the state, for its commit)
    kda: kda_block.Scratch,
    dsa: dsa_block.Scratch,
    dense: mlp.Scratch,
    moe: moe.Scratch,
    hidden: u64,
    fnormed: u64,
    fxs: u64,
    logits: u64, // bf16 [head rows][this rank's vocabulary]
    // the MTP head (every Buffers holds them, as the Python family's)
    hin: u64, // bf16 [rows][dims]: the hidden rows it reads
    me: u64, // bf16 [rows][dims]: the next tokens' embedding
    mcat: u64, // bf16 [rows][2 dims]: [enorm(me) | hnorm(hin)]
    mx: u64, // bf16 [rows][dims]: its stream
    part: u64, // fp32 [rows][dims]: this rank's partial rows a split exchange sends from

    pub fn init(d: *const cuda.Driver, w: *const weights.Weights, rows: usize, prefill: bool, capacity: usize) !Buffers {
        return make(d, w, rows, prefill, capacity, rows, false);
    }

    /// A shared window's scratch: `rows` rows of several streams, each one's attention and KDA chain on `window_rows`.
    pub fn initShared(d: *const cuda.Driver, w: *const weights.Weights, rows: usize, capacity: usize) !Buffers {
        return make(d, w, rows, false, capacity, window_rows, false);
    }

    pub fn initOwned(d: *const cuda.Driver, w: *const weights.Weights, rows: usize, prefill: bool, capacity: usize) !Buffers {
        return make(d, w, rows, prefill, capacity, if (prefill) rows else window_rows, true);
    }

    pub fn initSharedOwned(d: *const cuda.Driver, w: *const weights.Weights, rows: usize, capacity: usize) !Buffers {
        return make(d, w, rows, false, capacity, window_rows, true);
    }

    /// `seg`: the most rows one stream's attention takes (a decode window's rows, or a shared window's segment).
    fn make(d: *const cuda.Driver, w: *const weights.Weights, rows: usize, prefill: bool, capacity: usize, seg: usize, owned: bool) !Buffers {
        const c = &w.config;
        const s = &w.share;
        const D: usize = c.hidden;
        const S: usize = c.hc;
        const LL = s.kda.len();
        const HL = s.mla.len();
        const n = LL * c.kda_dim;
        const kw = 3 * n + 2 * c.kda_dim + LL;
        const mw = s.dense.len();
        const ew = s.moe.len();
        const sh = s.shared.len();
        const slots: usize = c.topk + 1;
        const pairs = rows * slots;
        const count: usize = c.experts;
        const lw: usize = c.kv_lora;
        const part_rows = if (prefill) @min(rows, dsa_block.prompt_att_rows) else seg;
        const nch = latent.chunksFor(@min(capacity, dense_reach) + seg);
        const partial_rows = @max(nch * part_rows, latent.chunksFor(sparse.width) * seg); // dense calls, or one sparse
        const np_max = capacity / sparse.pool;
        const mm = mixedPartBytes(w, rows);
        const hist_len = if (pairs > cuda.grouped.small) (pairs + cuda.grouped.small - 1) / cuda.grouped.small * (count + 1) else 1;
        const sizes = [_]usize{
            rows * 4,     rows * S * D * 2, rows * D * 2,           rows * (D / 64) * 4,
            rows * S * 4, rows * S * S * 4, glue.hcPartBytes(rows), s.world * rows * D * 4,
            if (prefill or seg < rows) rows * kw * 2 else 4, rows * n * 2, rows * n * 2, // kproj, ka, kg
            rows * n * 2, rows * n * 4, rows * n * 2, rows * n * 4, // saves out, k, v, g
            rows * LL * 4,                    rows * n * 4,               rows * n * 2, // saves b, wide q, y
            rows * (c.q_lora + lw) * 2,       rows * c.q_lora * 2,        rows * (c.q_lora / 64) * 4,
            rows * lw * 2,                    rows * (lw / 64) * 4,       rows * HL * c.nope * 2,
            rows * (c.i_dim + c.i_heads) * 2, rows * c.i_dim * 4,         glue.router_ks * rows * @max(count, c.i_dim) * 4,
            rows * HL * lw * 2,               rows * HL * lw * 2,         rows * HL * c.v_dim * 2,
            rows * c.i_heads * c.i_dim * 2,   partial_rows * HL * lw * 4, partial_rows * HL * 4,
            partial_rows * HL * 4,
            4, // dummy
            @min(seg, sparse.select_rows) * np_max * 4,
            rows * sparse.select_list * 4,
            rows * sparse.select_list * 4,
            rows * sparse.topk_pools * 8,
            4,
            rows * sparse.width * 4,
            rows * 4,
            mm,
            4, // a prompt block's position
            rows * mw * 2,
            rows * 2 * mw * 2,
            rows * mw * 2,
            rows * (mw / 64) * 4,
            rows * count * 4, pairs * 4, pairs * 4, pairs * 4, // mlog, pick, wts, members
            experts.maxItems(pairs, count + 1, experts.tile) * 3 * 4, // items (the decode tile's count bounds the prompt's)
            8,              @max(1, pairs) * 4, hist_len * 4, // counts, rank, hist
            pairs * ew * 2, pairs * D * 4,      rows * 2 * sh * 2,
            rows * sh * 2,
            rows * (sh / 64) * 4, rows * D * 4, // sxs, sy
            rows * D * 2,         rows * D * 2,
            rows * (D / 64) * 4,  (if (prefill) 1 else rows) * s.vocab.len() * 2,
            rows * D * 2, rows * D * 2, rows * 2 * D * 2, rows * D * 2, // hin, me, mcat, mx
            rows * D * 4, // part: this rank's fp32 partial rows a split exchange sends from
            if (w.quantizedAttention()) rows * 3 * n * 2 else 4, // the KDA projection's staging (EXL3)
            if (owned) part_rows * HL * lw * 2 else 0,
            if (owned) part_rows * HL * lw * 2 else 0,
        };
        var p: [sizes.len]u64 = undefined;
        const a = try arena(d, &sizes, &p);
        return .{
            .arena = a,
            .rows = rows,
            .prefill = prefill,
            .part_rows = part_rows,
            .nch = nch,
            .ids = p[0],
            .x = p[1],
            .normed = p[2],
            .xs = p[3],
            .post = p[4],
            .comb = p[5],
            .hcpart = p[6],
            .gathered = p[7],
            .kproj = p[8],
            .kda = .{ .p = 0, .ka = p[9], .kg = p[10], .saves = .{ .out = p[11], .state_out = 0, .k = p[12], .v = p[13], .g = p[14], .b = p[15] }, .wide = .{ .q = p[16], .y = p[17] }, .mm_part = p[42], .tmp = p[71] },
            .dsa = .{
                .dp = p[18],
                .qr = p[19],
                .xs_qr = p[20],
                .lat = p[21],
                .xs_lat = p[22],
                .q = p[23],
                .ikr = p[24],
                .igr = p[25],
                .router_part = p[26],
                .qa = p[27],
                .ol = p[28],
                .vn = p[29],
                .qi = p[30],
                .partials = .{ .po = p[31], .pm = p[32], .pl = p[33] },
                .dummy = p[34],
                .select = .{ .scores = p[35], .ls = p[36], .li = p[37], .pools = p[38], .at = p[39] },
                .tokens = p[40],
                .counts = p[41],
                .mm_part = p[42],
                .at = p[43],
                .owner_q = p[72],
                .owner_out = p[73],
            },
            .dense = .{ .tmp = p[44], .gu = p[45], .act = p[46], .xs_act = p[47], .part = p[42] },
            .moe = .{ .mlog = p[48], .router_part = p[26], .pick = p[49], .wts = p[50], .plan = .{ .members = p[51], .items = p[52], .counts = p[53], .rank = p[54], .hist = p[55] }, .eact = p[56], .ey = p[57], .sgu = p[58], .sact = p[59], .sxs = p[60], .sy = p[61], .mm_part = p[42] },
            .hidden = p[62],
            .fnormed = p[63],
            .fxs = p[64],
            .logits = p[65],
            .hin = p[66],
            .me = p[67],
            .mcat = p[68],
            .mx = p[69],
            .part = p[70],
        };
    }

    pub fn deinit(b: *Buffers) void {
        b.arena.buf.free();
        b.* = undefined;
    }
};

/// A sequence's committed caches on a rank (the Python family's State): KDA conv and states, saved rows, DSA caches.
pub const State = struct {
    arena: Arena,
    layout: layout_mod.Layout,
    gpa: std.mem.Allocator,
    capacity: usize,
    pos: usize = 0,
    pending: usize = 0, // rows of the last forward, until its commit
    prompt: bool = false, // whether that forward was a prompt chunk
    pos_dev: u64, // i32
    conv: u64, // bf16 [kda layers][3][3 heads 128]
    rec: u64, // fp32 [2][kda layers][heads][128][128]
    proj: u64, // bf16 [kda layers][window rows][kda width]
    save_k: u64, // fp32 [kda layers][window rows][heads][128]
    save_v: u64, // bf16, as save_k
    save_g: u64, // fp32, as save_k
    save_b: u64, // fp32 [kda layers][window rows][heads]
    cur: []u1,
    kda_layers: usize,
    heads: usize, // this rank's KDA heads
    caches: []dsa_block.Caches,
    mtp_len: usize = 0, // rows in the MTP layer's caches
    mtp_pos_dev: u64, // i32
    mtp_caches: dsa_block.Caches, // the MTP layer's latent rows, index keys, gates and pooled keys (the Python family's index[-1])
    conv_c: usize,
    rec_layer: usize,
    proj_layer: usize,
    proj_row: usize,

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, w: *const weights.Weights, capacity: usize, format: cache_rows.Format) !State {
        return make(gpa, d, w, capacity, format, false);
    }

    pub fn initShared(gpa: std.mem.Allocator, d: *const cuda.Driver, w: *const weights.Weights, capacity: usize, format: cache_rows.Format) !State {
        return make(gpa, d, w, capacity, format, true);
    }

    fn make(gpa: std.mem.Allocator, d: *const cuda.Driver, w: *const weights.Weights, capacity: usize, format: cache_rows.Format, external_planes: bool) !State {
        const c = &w.config;
        const s = &w.share;
        var nk: usize = 0;
        var nd: usize = 0;
        for (w.layers) |l| switch (l.mixer) {
            .kda => nk += 1,
            .mla => nd += 1,
        };
        const n = s.kda.len() * c.kda_dim;
        const kw = 3 * n + 2 * c.kda_dim + s.kda.len();
        const pk_rows = capacity / sparse.pool + 2;
        var layout = try layout_mod.Layout.init(.{
            4,                        nk * 3 * 3 * n * 2,       2 * nk * s.kda.len() * kda.dk * kda.dv * 4, nk * window_rows * kw * 2,
            nk * window_rows * n * 4, nk * window_rows * n * 2, nk * window_rows * n * 4,                   nk * window_rows * s.kda.len() * 4,
        }, capacity, @max(prefill_rows, window_rows), nd, format, c.kv_lora, c.i_dim);
        if (external_planes) layout = try layout.external();
        var a: Arena = .{ .buf = try cuda.DeviceBuffer.alloc(d, layout.total_bytes), .used = layout.total_bytes };
        errdefer a.buf.free();
        try a.buf.fill8(0, null);
        var p: [8]u64 = undefined;
        for (layout.fixed, &p) |offset, *ptr| ptr.* = a.buf.ptr + offset;
        const cur = try gpa.alloc(u1, nk);
        errdefer gpa.free(cur);
        @memset(cur, 0);
        const caches = try gpa.alloc(dsa_block.Caches, nd);
        for (caches, 0..) |*x, j| x.* = bindCaches(layout, a.buf.ptr, j, format, pk_rows);
        return .{ .arena = a, .layout = layout, .gpa = gpa, .capacity = capacity, .pos_dev = p[0], .conv = p[1], .rec = p[2], .proj = p[3], .save_k = p[4], .save_v = p[5], .save_g = p[6], .save_b = p[7], .cur = cur, .heads = s.kda.len(), .mtp_caches = bindCaches(layout, a.buf.ptr, nd, format, pk_rows), .mtp_pos_dev = a.buf.ptr + layout.mtp_pos, .kda_layers = nk, .caches = caches, .conv_c = 3 * n, .rec_layer = s.kda.len() * kda.dk * kda.dv * 4, .proj_layer = window_rows * kw, .proj_row = kw };
    }

    fn bindCaches(layout: layout_mod.Layout, base: u64, layer: usize, format: cache_rows.Format, pk_rows: usize) dsa_block.Caches {
        const p = layout.cacheOffsets(layer);
        return .{ .lc = .{ .ptr = if (layout.external_planes) 0 else base + p[0], .format = format }, .index = .{ .keys = base + p[1], .gates = base + p[2], .rows = layout.ring_rows }, .pk = .{ .ptr = if (layout.external_planes) 0 else base + p[3], .format = format }, .pk_rows = pk_rows };
    }

    pub fn address(st: *const State, r: layout_mod.Range) u64 {
        if (r.plane == .fixed) return st.arena.buf.ptr + r.offset;
        const c = if (r.layer == st.caches.len) st.mtp_caches else st.caches[r.layer];
        return r.offset + switch (r.plane) {
            .fixed => unreachable,
            .latent => c.lc.ptr,
            .keys => c.index.keys,
            .gates => c.index.gates,
            .pooled => c.pk.ptr,
        };
    }

    pub fn deinit(st: *State) void {
        st.arena.buf.free();
        st.gpa.free(st.cur);
        st.gpa.free(st.caches);
        st.* = undefined;
    }

    /// State.reset: a fresh sequence (zero conv windows and states, position 0); the caches past it are never read.
    pub fn reset(st: *State, stream: cuda.Stream) !void {
        const d = st.arena.buf.d;
        try d.check(d.api.cuMemsetD8Async(st.conv, 0, st.kda_layers * 3 * st.conv_c * 2, stream.handle), "cuMemsetD8Async");
        try d.check(d.api.cuMemsetD8Async(st.rec, 0, 2 * st.kda_layers * st.rec_layer, stream.handle), "cuMemsetD8Async");
        @memset(st.cur, 0);
        st.pending = 0;
        try st.setPos(stream, 0);
        try st.setMtpLen(stream, 0);
    }

    /// Host side of a state just copied back (a kept prompt state): position, MTP rows, KDA parity, nothing pending.
    pub fn restoreHost(st: *State, stream: cuda.Stream, pos: usize, mtp_len: usize, cur: []const u1) !void {
        @memcpy(st.cur, cur);
        st.pending = 0;
        st.prompt = false;
        try st.setPos(stream, pos);
        try st.setMtpLen(stream, mtp_len);
    }

    /// State.set_mtp_len: rows in the MTP layer's caches (the next MTP step writes from here).
    pub fn setMtpLen(st: *State, stream: cuda.Stream, n: usize) !void {
        st.mtp_len = n;
        const v: i32 = @intCast(n);
        try st.arena.buf.d.check(st.arena.buf.d.api.cuMemsetD32Async(st.mtp_pos_dev, @bitCast(v), 1, stream.handle), "cuMemsetD32Async");
    }

    pub fn setPos(st: *State, stream: cuda.Stream, pos: usize) !void {
        st.pos = pos;
        const v: i32 = @intCast(pos);
        try st.arena.buf.d.check(st.arena.buf.d.api.cuMemsetD32Async(st.pos_dev, @bitCast(v), 1, stream.handle), "cuMemsetD32Async");
    }

    pub fn recAt(st: *const State, copy: u1, li: usize) u64 {
        return st.rec + (@as(usize, copy) * st.kda_layers + li) * st.rec_layer;
    }
};

test "FP8 projection scratch covers decode and unfused prompt tails without clusters" {
    try std.testing.expectEqual(@as(usize, 4096), lanePartBytes(1, 128, 4096));
    try std.testing.expectEqual(@as(usize, 2097152), lanePartBytes(32, 4096, 4096));
    try std.testing.expectEqual(@as(usize, 1044480), lanePartBytes(2048, 128, 4096));
}

test "TP2 mixed scratch covers every projection and subrow count" {
    const c: @import("config.zig").Config = .{};
    const share = try @import("cuda_shard.zig").Share.of(&c, 0, 2);
    const w: weights.Weights = .{ .gpa = std.testing.allocator, .config = c, .share = share };
    const shapes = projectionShapes(&w);
    try std.testing.expectEqual(@as(usize, 16), shapes.len);
    try std.testing.expectEqual(Shape{ .group = .dsa, .n = 1536, .k = 4096 }, shapes[4]);
    try std.testing.expectEqual(Shape{ .group = .dsa, .n = 512, .k = 4096 }, shapes[5]);
    try std.testing.expectEqual(Shape{ .group = .shared, .n = 4096, .k = 1024 }, shapes[11]);
    try std.testing.expectEqual(Shape{ .group = .mtp_eh, .n = 4096, .k = 8192 }, shapes[12]);
    for ([_]struct { rows: usize, bytes: usize }{
        .{ .rows = 8, .bytes = 524288 },
        .{ .rows = 32, .bytes = 2097152 },
        .{ .rows = 2048, .bytes = 16711680 },
    }) |sample| {
        const budget = mixedPartBytes(&w, sample.rows);
        try std.testing.expectEqual(sample.bytes, budget);
        for (1..sample.rows + 1) |m| for (shapes) |shape| {
            const bf_need = b16.partBytes(m, shape.n, shape.k);
            const q4_need = m * (shape.k / 64) * 4;
            const plan = cuda.qmmf.plan(m, shape.n, shape.k, false);
            const lane_need = if (plan.reduce) @as(usize, plan.sk) * m * shape.n * 4 else 0;
            try std.testing.expect(budget >= bf_need);
            try std.testing.expect(budget >= q4_need);
            try std.testing.expect(budget >= lane_need);
        };
    }
}
