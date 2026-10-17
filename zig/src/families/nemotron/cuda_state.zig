//! Device buffers of one Nemotron sequence: its caches and recurrent state, and the scratch a window or chunk uses.

const std = @import("std");
const lesson_route = @import("lesson_route.zig");
const cuda = @import("cuda");
const Config = @import("config.zig").Config;
const kern = @import("cuda_kernels.zig");
const sampler = @import("cuda_sampler.zig");
const seq_graphs = @import("cuda_seq_graphs.zig");
const latent = @import("cuda_latent.zig");
const worker = @import("cuda_worker.zig");

/// The blocks one rank of a layer split owns: the weight loader's Range, so the loader and the state agree on one
/// type. `all` is one device.
pub const Range = @import("cuda_weights.zig").Range;

pub const max_rows = 16; // a verify window's rows (the row tile of the row-parallel kernels)
pub const prefill_rows = 2048; // rows of a staggered segment's chunk, and of a pass under dumps or lessons
pub const pass_rows = 2 * prefill_rows; // rows a serial prompt pass runs at most (a row's bits are the same at any cut)
pub const yield_rows = 256; // rows of a background pass's chunk: a request that arrives waits for one at most
pub const chunk_keys = 512; // keys an attention chunk holds, at fixed absolute positions
const shared_words = 2 * max_rows + 8 * max_rows; // the shared expert's members, then (items, counts) a row count

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

/// A sequence's KV planes in the display carveout, given back when the sequence goes.
pub const Carved = struct {
    carve: *cuda.Carveout,
    kv: [2]cuda.DeviceBuffer,

    /// Both KV planes of `n` bytes each from `carve`, or null when it is off or cannot hold both.
    fn take(carve: ?*cuda.Carveout, n: usize) !?Carved {
        const cv = carve orelse return null;
        const k = (try cv.take(n)) orelse return null;
        errdefer cv.give(k);
        const v = (try cv.take(n)) orelse {
            cv.give(k);
            return null;
        };
        return .{ .carve = cv, .kv = .{ k, v } };
    }

    fn give(x: Carved) void {
        x.carve.give(x.kv[1]);
        x.carve.give(x.kv[0]);
    }

    /// Arena bytes for `sizes` with the carved planes left out.
    fn arenaBytes(x: ?Carved, sizes: []const usize) usize {
        var total: usize = 0;
        for (sizes, 0..) |n, i| total += if (x != null and i < 2) 0 else n + 256;
        return total;
    }
};

/// A sequence's KV planes in a shared page pool (TENSORFOLD_KV_POOL_TOKENS): address space for its whole context,
/// pages only for the rows it has written (cuda/kvpool.zig). The kernels address the planes as before.
pub const Paged = struct {
    kv: [2]*cuda.kvpool.Range,

    /// Both planes of `n` bytes each reserved in `pool`, or null when there is no pool.
    fn take(pool: ?*cuda.KvPool, n: usize) !?Paged {
        const p = pool orelse return null;
        const k = try p.reserve(n);
        errdefer k.free();
        const v = try p.reserve(n);
        return .{ .kv = .{ k, v } };
    }

    fn give(x: Paged) void {
        x.kv[1].free();
        x.kv[0].free();
    }

    /// Pages under the first `rows` rows of every attention layer this rank owns, in both planes.
    pub fn ensure(x: Paged, c: Config, max_len: usize, range: Range, rows: usize) !void {
        const row = kvRowBytes(c);
        var j: usize = 0;
        for (c.kinds[range.first..@min(range.last, c.layers)]) |k| if (k == .attention) {
            for (x.kv) |r| try r.ensure(j * max_len * row, rows * row);
            j += 1;
        };
    }

    /// Every page back to the pool; the GPU's work on the planes must have finished.
    pub fn trim(x: Paged) void {
        for (x.kv) |r| r.trim();
    }
};

/// A sequence's own Buffers fields: cache and state planes (the first seven), window io and its sampled rule.
pub const seq_fields = [_][]const u8{ "k_cache", "v_cache", "ssm", "conv_base", "raw", "xc", "dt", "meta", "ids", "hidden", "sampled", "rule" };

/// Bytes of one token's key (or value) row in one attention layer: every KV head's dims in bf16.
pub fn kvRowBytes(c: Config) usize {
    return c.kv_heads * c.head_dim * @as(usize, if (c.kv_fp8) 1 else 2);
}

/// Bytes of every Mamba layer's SSM and conv states (seq_fields' ssm, conv_base): a prompt mark keeps both whole.
pub fn mambaPlanes(c: Config) [2]usize {
    const nm: usize = c.count(.mamba);
    return .{ nm * c.mamba_heads * c.mamba_head_dim * c.state * 4, nm * 3 * c.convDim() * 2 };
}

fn seqSizes(c: Config, max_len: usize) [seq_fields.len]usize {
    return seqSizesRange(c, max_len, .all);
}

/// `seqSizes` for one rank of a layer split: the KV planes and the Mamba state cover only the layers this rank
/// owns, so each rank's sequence state is sized for its own half. The layer indices stay global, so a rank's planes
/// are indexed from its first owned layer.
fn seqSizesRange(c: Config, max_len: usize, range: Range) [seq_fields.len]usize {
    const W: usize = max_rows;
    const last = @min(range.last, c.layers);
    var nm: usize = 0;
    var na: usize = 0;
    for (c.kinds[range.first..last]) |k| switch (k) {
        .mamba => nm += 1,
        .attention => na += 1,
        .moe => {},
    };
    const kv = na * max_len * kvRowBytes(c);
    const cd: usize = c.convDim();
    const m = mambaPlanes(c);
    return .{ kv, kv, m[0], m[1], nm * 2 * W * cd * 2, nm * 2 * W * cd * 2, nm * 2 * W * c.mamba_heads * 4, 16, W * 4, W * c.hidden * 2, W * 4, @sizeOf(sampler.Rule) };
}

const scratch_count = 43;

/// Scratch windows and chunks share on one stream: window logits, chunk io, activations, the expert plan.
fn scratchSizes(c: Config, nch: usize, pass: usize) [scratch_count]usize {
    const R: usize = pass;
    const W: usize = max_rows;
    const D: usize = c.hidden;
    const ns: usize = c.slots();
    const qd: usize = c.heads * c.head_dim;
    const cd: usize = c.convDim();
    const xd: usize = c.inner();
    const pairs = R * ns;
    // a latent MoE plans pairs (row, pick, share held) over its share tables; its experts write fp32 to the exchange
    const lat = c.latent > 0;
    const own: usize = if (lat) c.ownShares() else 1;
    const plan_experts = if (lat) c.experts * own else c.experts + 2;
    const items = kern.maxItems(@intCast(pairs * own), plan_experts, 16);
    const ymoe = if (lat) c.world * latent.partFloats(c, R) * 4 else pairs * D * 4; // a region a rank
    const job = if (lat and c.world > 1) worker.jobBytes(c) else 0;
    const sw = if (lat) c.shared_width / c.shares else 0;
    // the checkpoint's math: e4m3 rows of the widest FP8 projection input, and a latent MoE's NVFP4 rows
    const ck8 = if (c.ck) R * @max(D, @max(xd, qd)) else 0;
    const ck4 = if (c.ck and lat) @import("cuda_latent_ck.zig").rowsBytes(c, R) else 0;
    return .{
        W * c.vocab * 2,                                              16,                                             R * 4,
        R * D * 2,                                                    c.vocab * 2,                                    16,
        R * D * 2,                                                    R * D * 2,                                      R * D * 2,
        R * D * 2,                                                    R * D / 64 * 4,                                 R * D * 2,
        R * c.projDim() * 2,                                          R * cd * 2,                                     R * xd * 2,
        R * xd * 2,                                                   R * xd / 64 * 4,                                R * c.qkvDim() * 2,
        R * qd * 2,                                                   R * qd * 2,                                     R * qd / 64 * 4,
        W * nch * qd * 4,                                             W * nch * c.heads * 4,                          W * nch * c.heads * 4,
        (if (lat) latent.routerSlices(D) else 6) * R * c.experts * 4, pairs * own * 4,                                R * ns * 4,
        pairs * own * 4,                                              items * 12,                                     8,
        pairs * own * 4,                                              (pairs * own + 1023) / 1024 * plan_experts * 4, pairs * c.expert_width * 2,
        ymoe,                                                         shared_words * 4,                               R * kern.pocket_width * 4,
        R * c.latent * 2,                                             R * c.latent * 2,                               R * own * sw * 2,
        R * own * 4,                                                  job,                                            ck8,
        ck4,
    };
}

/// One sequence: its own buffers (seq_fields, then its MTP head's) and where it stands while another is bound.
pub const Seq = struct {
    arena: ?Arena, // null: the engine's own buffers
    carved: ?Carved = null,
    ptr: [seq_fields.len]u64,
    head: [3]u64 = @splat(0), // the MTP head's seq_fields (cuda_mtp.zig)
    graphs: seq_graphs.Graphs = .{}, // what this sequence captured for itself (the own one: masked windows alone)
    epoch: u32 = 0, // the engine's graph epoch its graphs were captured in
    pos: usize = 0,
    parity: usize = 0,
    prev_keep: usize = 0,
    rows: usize = 0,
    head_pos: usize = 0,
    sampling: ?sampler.Sampling = null,
    slot: u32 = 0, // its lesson latch slot (lesson_route)
    decide: i64 = lesson_route.none, // the prompt row where its lessons decide whether they act on the reply
    untaught: bool = false, // reads the model as shipped: no learned neuron acts on its rows (a lesson writer's reads)
    peer: ?*Seq = null, // two-GPU TP: this sequence's share on the other rank (null: the own one's, or no TP)
    paged: ?Paged = null, // its KV planes in the shared page pool (null: in its arena or the carveout)

    /// The device bytes one sequence takes: its buffers for `max_len` cache rows and a head's `head` buffers.
    pub fn bytes(c: Config, max_len: usize, head: []const usize) usize {
        return bytesRange(c, max_len, head, .all);
    }

    /// `bytes` for one rank of a layer split: only `range`'s layers' planes are counted.
    pub fn bytesRange(c: Config, max_len: usize, head: []const usize, range: Range) usize {
        var total: usize = 0;
        for (seqSizesRange(c, max_len, range)) |n| total += n + 256;
        for (head) |n| total += n + 256;
        return total;
    }

    /// Cache and head buffers are zeroed on their owning stream; the KV pair goes to `carve` while it holds both.
    pub fn init(d: *const cuda.Driver, c: Config, max_len: usize, head: []const usize, stream: cuda.Stream, carve: ?*cuda.Carveout) !Seq {
        return initRange(d, c, max_len, head, stream, carve, .all, null);
    }

    /// `init` for one rank of a layer split: the sequence planes cover only `range`'s layers.
    pub fn initRange(d: *const cuda.Driver, c: Config, max_len: usize, head: []const usize, stream: cuda.Stream, carve: ?*cuda.Carveout, range: Range, pool: ?*cuda.KvPool) !Seq {
        const sizes = seqSizesRange(c, max_len, range);
        const paged = try Paged.take(pool, sizes[0]);
        errdefer if (paged) |x| x.give();
        const carved = if (paged == null) try Carved.take(carve, sizes[0]) else null;
        errdefer if (carved) |x| x.give();
        var total = if (paged != null) kvlessBytes(&sizes) else Carved.arenaBytes(carved, &sizes);
        for (head) |n| total += n + 256;
        var a: Arena = .{ .buf = try cuda.DeviceBuffer.alloc(d, total) };
        errdefer a.buf.free();
        try a.buf.fill8(0, stream.handle);
        var s: Seq = .{ .arena = null, .ptr = undefined, .carved = carved, .paged = paged };
        for (&s.ptr, sizes, 0..) |*p, n, i| p.* = if (paged != null and i < 2) paged.?.kv[i].va else if (carved != null and i < 2) carved.?.kv[i].ptr else a.take(n);
        for (s.head[0..head.len], head) |*p, n| p.* = a.take(n);
        s.arena = a;
        return s;
    }

    /// The engine's own buffers as a sequence.
    pub fn view(b: *const Buffers) Seq {
        var s: Seq = .{ .arena = null, .ptr = undefined, .paged = b.paged };
        inline for (seq_fields, &s.ptr) |name, *p| p.* = @field(b, name);
        return s;
    }

    pub fn deinit(s: *Seq) void {
        if (s.arena) |*a| a.buf.free();
        if (s.carved) |x| x.give();
        if (s.arena != null) if (s.paged) |x| x.give(); // a view's planes are its Buffers' to give back
        s.* = undefined;
    }
};

/// Arena bytes for a sequence's `sizes` with both KV planes left out (they are in the page pool).
fn kvlessBytes(sizes: []const usize) usize {
    var total: usize = 0;
    for (sizes[2..]) |n| total += n + 256;
    return total;
}

pub const Buffers = struct {
    arena: Arena,
    carved: ?Carved, // the own sequence's KV planes, when the display carveout holds them
    paged: ?Paged = null, // the own sequence's KV planes, when the page pool holds them
    // sequence state (Engine.STATE in the Python engine)
    k_cache: u64,
    v_cache: u64,
    ssm: u64,
    conv_base: u64,
    raw: u64,
    xc: u64,
    dt: u64,
    state_bytes: [7]usize,
    // a window's inputs and outputs
    meta: u64,
    ids: u64,
    hidden: u64,
    logits: u64,
    sampled: u64,
    // a prompt chunk's own inputs and outputs
    p_meta: u64,
    p_ids: u64,
    p_hidden: u64,
    p_logits: u64,
    p_sampled: u64,
    // scratch shared by windows and chunks (they never overlap on the stream)
    emb: u64,
    h: [2]u64,
    y: u64,
    xs: u64,
    delta: u64,
    proj: u64,
    p_xc: u64,
    sy: u64,
    g: u64,
    gxs: u64,
    qkv: u64,
    q: u64,
    att: u64,
    axs: u64,
    po: u64,
    pm: u64,
    pl: u64,
    part: u64,
    pick: u64,
    wts: u64,
    plan: kern.Plan,
    act: u64,
    ymoe: u64,
    rule: u64, // the sequence's sampled rule, as sample.cu reads it
    shared: u64, // the shared expert's fixed plans: 32 members, then (items, counts) for 1..16 rows
    pocket: u64, // the pocket's activations, fp32 [rows, pocket_width]
    lat: u64, // a latent MoE's rows in the latent width (bf16)
    comb: u64, // its routed pairs' latent sum a row (bf16)
    sact: u64, // its shared expert's activations (bf16)
    sids: u64, // its shared expert's pair ids, share i % own (cuda_latent.initShareIds)
    job: u64, // rank 0's job to the other ranks: a header and the rows their shares read (cuda_worker.zig)
    ck8: u64, // the checkpoint's math: e4m3 rows an FP8 projection reads
    ck4: u64, // and a latent MoE's NVFP4 rows (cuda_latent_ck.zig)
    k16: u64 = 0, // an FP8 cache's rows as bf16 for a prompt chunk's attention (max_len rows of one layer's KV heads)
    v16: u64 = 0,

    /// Sizes every buffer for `max_len` cache rows and `nch` attention chunk partials a row; the own sequence's first.
    pub fn init(d: *const cuda.Driver, c: Config, max_len: usize, nch: usize, carve: ?*cuda.Carveout) !Buffers {
        return initRange(d, c, max_len, nch, carve, .all, null, pass_rows);
    }

    /// `init` for one rank of a layer split: the sequence planes cover only `range`'s layers.
    /// `pass`: the widest prompt pass the scratch holds (pass_rows; a two-GPU TP rank's is prefill_rows).
    pub fn initRange(d: *const cuda.Driver, c: Config, max_len: usize, nch: usize, carve: ?*cuda.Carveout, range: Range, pool: ?*cuda.KvPool, pass: usize) !Buffers {
        const own = seqSizesRange(c, max_len, range);
        const scratch = scratchSizes(c, nch, pass);
        const paged = try Paged.take(pool, own[0]);
        errdefer if (paged) |x| x.give();
        const carved = if (paged == null) try Carved.take(carve, own[0]) else null;
        errdefer if (carved) |x| x.give();
        var total = if (paged != null) kvlessBytes(&own) else Carved.arenaBytes(carved, &own);
        for (scratch) |n| total += n + 256;
        const plane16: usize = if (c.kv_fp8) max_len * c.kv_heads * c.head_dim * 2 else 0;
        total += 2 * (plane16 + 256);
        var a: Arena = .{ .buf = try cuda.DeviceBuffer.alloc(d, total) };
        errdefer a.buf.free();
        // Zeroed: the Mamba convolution and recurrent planes are read before they are first written, so a fresh
        // allocation's contents would otherwise change the first forward's output.
        try a.buf.fill8(0, null);
        var b: Buffers = undefined;
        b.carved = carved;
        b.paged = paged;
        b.state_bytes = own[0..7].*;
        inline for (seq_fields, own, 0..) |name, n, i| @field(b, name) = if (paged != null and i < 2) paged.?.kv[i].va else if (carved != null and i < 2) carved.?.kv[i].ptr else a.take(n);
        for (b.scratchPtrs(), scratch) |f, n| f.* = a.take(n);
        b.k16 = if (plane16 > 0) a.take(plane16) else 0;
        b.v16 = if (plane16 > 0) a.take(plane16) else 0;
        b.arena = a;
        return b;
    }

    /// Its own scratch for another prompt segment, sharing `b`'s caches and state (follow); deinit frees the scratch.
    pub fn sibling(b: *const Buffers, d: *const cuda.Driver, c: Config, nch: usize) !Buffers {
        const scratch = scratchSizes(c, nch, pass_rows);
        var total: usize = 0;
        for (scratch) |n| total += n + 256;
        var s: Buffers = b.*;
        s.carved = null; // the owner gives its planes back
        s.paged = null;
        s.arena = .{ .buf = try cuda.DeviceBuffer.alloc(d, total) };
        for (s.scratchPtrs(), scratch) |f, n| f.* = s.arena.take(n);
        return s;
    }

    /// A sibling takes `b`'s sequence buffers (they move when the engine binds another sequence).
    pub fn follow(s: *Buffers, b: *const Buffers) void {
        inline for (seq_fields) |name| @field(s, name) = @field(b, name);
        s.state_bytes = b.state_bytes;
    }

    /// Every scratch field, in scratchSizes' order.
    fn scratchPtrs(b: *Buffers) [scratch_count]*u64 {
        return .{
            &b.logits,    &b.p_meta,    &b.p_ids, &b.p_hidden, &b.p_logits, &b.p_sampled, &b.emb, &b.h[0],         &b.h[1],       &b.y,
            &b.xs,        &b.delta,     &b.proj,  &b.p_xc,     &b.sy,       &b.g,         &b.gxs, &b.qkv,          &b.q,          &b.att,
            &b.axs,       &b.po,        &b.pm,    &b.pl,       &b.part,     &b.pick,      &b.wts, &b.plan.members, &b.plan.items, &b.plan.counts,
            &b.plan.rank, &b.plan.hist, &b.act,   &b.ymoe,     &b.shared,   &b.pocket,    &b.lat, &b.comb,         &b.sact,       &b.sids,
            &b.job,       &b.ck8,       &b.ck4,
        };
    }

    /// The shared expert's plan at `rows` rows: halves `experts` and `experts + 1` over slots top_k and top_k + 1.
    pub fn sharedPlan(b: *const Buffers, rows: usize) kern.Plan {
        const items = b.shared + (2 * max_rows + 8 * @as(u64, rows - 1)) * 4;
        return .{ .members = b.shared, .items = items, .counts = items + 6 * 4, .rank = 0, .hist = 0 };
    }

    /// Writes every row count's shared plan (pairs in row order, one item a half since rows <= 16).
    pub fn initShared(b: *const Buffers, ops: kern.Ops, c: Config) !void {
        if (c.latent > 0) {
            try latent.initShareIds(ops.k.latent.?, ops.s, b.sids, pass_rows, c.ownShares());
            return ops.s.synchronize();
        }
        var w: [shared_words]u32 = undefined;
        for (0..max_rows) |r| {
            w[r] = @intCast(r * c.slots() + c.top_k);
            w[max_rows + r] = @intCast(r * c.slots() + c.top_k + 1);
        }
        for (1..max_rows + 1) |rows| {
            const at = 2 * max_rows + 8 * (rows - 1);
            const n: u32 = @intCast(rows);
            w[at..][0..8].* = .{ @intCast(c.experts), 0, n, @intCast(c.experts + 1), max_rows, n, 2, 2 };
        }
        try ops.upload(b.shared, std.mem.sliceAsBytes(&w));
        try ops.s.synchronize();
    }

    pub fn deinit(b: *Buffers) void {
        b.arena.buf.free();
        if (b.carved) |x| x.give();
        if (b.paged) |x| x.give();
        b.* = undefined;
    }

    /// The cache and state planes' addresses, in seq_fields' order (the KV pair may live in the carveout).
    pub fn planes(b: *const Buffers) [7]u64 {
        return .{ b.k_cache, b.v_cache, b.ssm, b.conv_base, b.raw, b.xc, b.dt };
    }

    /// Bytes of every cache and state plane together.
    pub fn stateBytes(b: *const Buffers) usize {
        var total: usize = 0;
        for (b.state_bytes) |n| total += n;
        return total;
    }

    /// Engine.snapshot: a device copy of every cache and state plane, packed.
    pub fn snapshot(b: *const Buffers, ops: kern.Ops) !cuda.DeviceBuffer {
        var copy = try cuda.DeviceBuffer.alloc(ops.k.d, b.stateBytes());
        errdefer copy.free();
        var at: usize = 0;
        for (b.planes(), b.state_bytes) |p, n| {
            try ops.copy(copy.ptr + at, p, n);
            at += n;
        }
        return copy;
    }

    pub fn restore(b: *const Buffers, ops: kern.Ops, saved: cuda.DeviceBuffer) !void {
        var at: usize = 0;
        for (b.planes(), b.state_bytes) |p, n| {
            try ops.copy(p, saved.ptr + at, n);
            at += n;
        }
    }

    /// Engine.reset: every cache and state buffer zeroed, as a fresh request starts.
    pub fn reset(b: *const Buffers, ops: kern.Ops) !void {
        // KV rows are written before they are read; pooled planes are not all backed, so they are not zeroed
        const from: usize = if (b.paged != null) 2 else 0;
        for (b.planes()[from..], b.state_bytes[from..]) |p, n| try ops.fill32(p, 0, n / 4);
    }

    /// Bytes of the own sequence's planes in the display carveout.
    pub fn carvedBytes(b: *const Buffers) usize {
        return if (b.carved) |x| x.kv[0].len + x.kv[1].len else 0;
    }
};
