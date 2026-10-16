//! Two-GPU tensor parallel in one process (TENSORFOLD_TP=2): both ranks run every block on their share of it, and
//! the block's two partial outputs meet once before the residual moves on.
//!
//! The shares: a Mamba block's heads and state groups, an attention block's query heads and KV head (each with the
//! out projection's matching columns), and a MoE block's routed experts in contiguous halves with one half of the
//! shared expert (expert parallel). The embedding and the router are whole on both ranks, so both hold the same
//! residual; rank 0 alone runs the final norm, the head and the sample.
//!
//! The exchange stages each rank's partial through portable pinned host memory (the two cards have no peer access
//! under WDDM). Each rank's stream copies its partial out and records an event; the other rank's stream waits on that
//! event, copies it in, and sums its own and the peer's in fp32, rounding to bf16 once. a + b is commutative in IEEE
//! fp32, so both ranks write the same bits. Nothing waits on the host: two parities of staging, each released by the
//! peer's event after its copy-in, keep a later exchange from overwriting one the peer has not read.

const std = @import("std");
const cuda = @import("cuda");
const fwd = @import("cuda_forward.zig");
const state = @import("cuda_state.zig");
const Forward = fwd.Forward;
const Seg = fwd.Seg;
const Walk = fwd.Walk;
const Delta = fwd.Delta;

pub const world = 2;

/// The pair's staging and events, and each rank's partial and received-partial buffers.
pub const Link = struct {
    gpa: std.mem.Allocator,
    rank: [world]Rank,
    parity: u1 = 0,
    /// Rows the widest exchange carries (a prompt pass).
    rows: usize,
    hidden: usize,
    exchanges: u64 = 0,

    pub const Rank = struct {
        ctx: *const cuda.Context,
        own32: cuda.DeviceBuffer, // a MoE block's fp32 partial
        peer: cuda.DeviceBuffer, // the other rank's partial, copied in
        slot: [2]cuda.HostBuffer, // this rank's partial, staged; one a parity
        ready: [2]cuda.Event, // this rank's partial is in its slot
        done: [2]cuda.Event, // this rank has copied the peer's slot in
    };

    /// Each rank's buffers and events in its own context; `rows` the widest exchange.
    pub fn init(gpa: std.mem.Allocator, ctxs: [world]*const cuda.Context, rows: usize, hidden: usize) !*Link {
        const l = try gpa.create(Link);
        errdefer gpa.destroy(l);
        l.* = .{ .gpa = gpa, .rank = undefined, .rows = rows, .hidden = hidden };
        const bytes = rows * hidden * 4;
        var made: usize = 0;
        errdefer for (l.rank[0..made]) |*r| freeRank(r);
        for (&l.rank, ctxs) |*r, ctx| {
            try ctx.makeCurrent();
            r.ctx = ctx;
            r.own32 = try cuda.DeviceBuffer.alloc(ctx.d, bytes);
            errdefer r.own32.free();
            r.peer = try cuda.DeviceBuffer.alloc(ctx.d, bytes);
            errdefer r.peer.free();
            r.slot[0] = try cuda.HostBuffer.alloc(ctx.d, bytes);
            errdefer r.slot[0].free();
            r.slot[1] = try cuda.HostBuffer.alloc(ctx.d, bytes);
            errdefer r.slot[1].free();
            for (&r.ready, &r.done) |*a, *b| {
                a.* = try cuda.Event.init(ctx.d, false);
                b.* = try cuda.Event.init(ctx.d, false);
            }
            made += 1;
        }
        try ctxs[0].makeCurrent();
        return l;
    }

    fn freeRank(r: *Rank) void {
        r.ctx.makeCurrent() catch {};
        for (&r.ready, &r.done) |*a, *b| {
            a.deinit();
            b.deinit();
        }
        r.slot[0].free();
        r.slot[1].free();
        r.peer.free();
        r.own32.free();
    }

    pub fn deinit(l: *Link) void {
        for (&l.rank) |*r| freeRank(r);
        l.rank[0].ctx.makeCurrent() catch {};
        l.gpa.destroy(l);
    }

    /// Every rank's out[r] = src[0] + src[1] over `n` values (each partial bf16, or fp32 when `wide`), queued on the
    /// ranks' streams with no host wait. Leaves rank 0's context current.
    pub fn exchange(l: *Link, fs: [world]*const Forward, src: [world]u64, wide: bool, out: [world]u64, n: usize) !void {
        const bytes = n * @as(usize, if (wide) 4 else 2);
        if (n > l.rows * l.hidden) return error.ExchangeTooWide;
        const p = l.parity;
        l.parity ^= 1;
        for (fs, 0..) |f, r| {
            const me = &l.rank[r];
            const other = &l.rank[1 - r];
            try me.ctx.makeCurrent();
            try f.ops.s.wait(other.done[p]); // the peer has read this parity's slot from two exchanges ago
            try f.ops.download(me.slot[p].bytes[0..bytes], src[r]);
            try me.ready[p].record(f.ops.s);
        }
        for (fs, 0..) |f, r| {
            const me = &l.rank[r];
            const other = &l.rank[1 - r];
            try me.ctx.makeCurrent();
            try f.ops.s.wait(other.ready[p]);
            try f.ops.upload(me.peer.ptr, other.slot[p].bytes[0..bytes]);
            try me.done[p].record(f.ops.s);
            try f.ops.tpSum2(src[r], wide, me.peer.ptr, wide, out[r], n);
        }
        l.exchanges += 1;
        try l.rank[0].ctx.makeCurrent();
    }

    /// One block's partials met: a MoE block's from each rank's fp32 partial, any other's from its b.delta; the sum
    /// lands in each rank's b.delta, so the next norm adds it as a dense delta.
    fn meet(l: *Link, fs: [world]*const Forward, moe: bool, rows: usize) !void {
        const n = rows * fs[0].c.hidden;
        var src: [world]u64 = undefined;
        var out: [world]u64 = undefined;
        for (fs, 0..) |f, r| {
            src[r] = if (moe) l.rank[r].own32.ptr else f.b.delta;
            out[r] = f.b.delta;
        }
        try l.exchange(fs, src, moe, out, n);
    }
};

/// One window over both ranks (one sequence on each, the same rows): every block on both, its partials met, then
/// rank 0's final norm, head and sample. `segs[r]` are rank r's own buffers; their ids and meta are already there.
pub fn window(l: *Link, fs: [world]*const Forward, segs: [world][]const Seg) !void {
    var rows: usize = 0;
    for (segs[0]) |s| rows += s.rows;
    if (rows == 0 or rows > state.max_rows) return error.WindowTooWide;
    var x: [world]u64 = undefined;
    var delta: [world]Delta = .{ .none, .none };
    var mj: [world]usize = .{ 0, 0 };
    var aj: [world]usize = .{ 0, 0 };
    for (fs, 0..) |f, r| {
        try f.ctx.makeCurrent();
        for (segs[r]) |s| try f.embed(s.b.ids, f.b.emb + s.row0 * @as(u64, f.c.hidden) * 2, s.rows);
        x[r] = f.b.emb;
    }
    const c = fs[0].c;
    for (0..c.layers) |li| {
        const moe = c.kinds[li] == .moe;
        for (fs, 0..) |f, r| {
            try f.ctx.makeCurrent();
            const blk = f.w.blocks[li];
            try f.norm(&x[r], delta[r], blk.norm, rows, true);
            if (moe) {
                try f.moeTp(blk.moe, rows, false, l.rank[r].own32.ptr);
            } else try f.blockMixer(segs[r], rows, blk, &delta[r], &mj[r], &aj[r]);
        }
        try l.meet(fs, moe, rows);
        delta = .{ .dense, .dense };
    }
    const f0 = fs[0];
    try f0.ctx.makeCurrent();
    try f0.norm(&x[0], delta[0], f0.w.norm_f, rows, true);
    try f0.tailOnly(segs[0], rows);
}

/// One prompt chunk of `rows` tokens at `pos` over both ranks: every block on both, its partials met, then rank 0's
/// final norm, head and sample. Each rank's p_ids already hold the chunk.
pub fn chunk(l: *Link, fs: [world]*const Forward, rows: usize, pos: usize) !void {
    if (rows > l.rows) return error.ChunkTooWide;
    var w: [world]Walk = .{ .{ .rows = rows, .pos = pos }, .{ .rows = rows, .pos = pos } };
    for (fs, 0..) |f, r| {
        try f.ctx.makeCurrent();
        try f.chunkBegin(&w[r]);
    }
    const c = fs[0].c;
    for (0..c.layers) |li| {
        const moe = c.kinds[li] == .moe;
        for (fs, 0..) |f, r| {
            try f.ctx.makeCurrent();
            try f.chunkPre(&w[r], li);
            try f.chunkMixer(&w[r], li);
            if (moe) {
                try f.moeTp(f.w.blocks[li].moe, rows, true, l.rank[r].own32.ptr);
            } else try f.chunkPost(&w[r], li);
        }
        try l.meet(fs, moe, rows);
        w[0].delta = .dense;
        w[1].delta = .dense;
    }
    try fs[0].ctx.makeCurrent();
    try fs[0].chunkFinish(&w[0]);
}
