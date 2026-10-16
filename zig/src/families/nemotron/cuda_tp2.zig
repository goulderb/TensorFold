//! Two-GPU tensor parallel in one process (TENSORFOLD_TP=2): both ranks run every block on their share of it, and
//! the block's two partial outputs meet once before the residual moves on.
//!
//! The shares: a Mamba block's heads and state groups, an attention block's query heads and KV head (each with the
//! out projection's matching columns), and a MoE block's routed experts in contiguous halves with one half of the
//! shared expert (expert parallel). The embedding and the router are whole on both ranks, so both hold the same
//! residual; rank 0 alone runs the final norm, the head and the sample.
//!
//! Each rank's kernels are enqueued by its own host thread (the caller's for rank 0, the link's worker for rank 1,
//! which keeps rank 1's context current for its life): launches on Windows cost microseconds each, and one thread
//! enqueuing both ranks left the GPUs waiting on it.
//!
//! The exchange stages each rank's partial through portable pinned host memory (the two cards have no peer access
//! under WDDM). A rank's stream copies its partial out and records `ready`; the other rank's stream waits on that
//! event, copies it in, records `done`, and sums its own and the peer's in fp32, rounding to bf16 once. a + b is
//! commutative in IEEE fp32, so both ranks write the same bits. Nothing waits on the GPU from the host: two parities
//! of staging, each released by the peer's `done` after its copy-in, keep an exchange from overwriting a slot the
//! peer has not read. A stream waits on an event's latest enqueued record, so the threads count what each has
//! enqueued (`posted`, `consumed`) and a rank never enqueues a wait before the peer has enqueued the record it means.

const std = @import("std");
const cuda = @import("cuda");
const fwd = @import("cuda_forward.zig");
const state = @import("cuda_state.zig");
const Forward = fwd.Forward;
const Seg = fwd.Seg;
const Walk = fwd.Walk;
const Delta = fwd.Delta;

pub const world = 2;

const Job = union(enum) {
    idle,
    window: struct { fs: [world]*const Forward, segs: [world][]const Seg, base: u64 },
    chunk: struct { fs: [world]*const Forward, rows: usize, pos: usize, base: u64 },
    quit,
};

/// The pair's staging, events and enqueue counters, each rank's partial and received-partial buffers, and rank 1's
/// enqueue thread.
pub const Link = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    rank: [world]Rank,
    /// Rows the widest exchange carries (a prompt pass).
    rows: usize,
    hidden: usize,
    /// Exchanges enqueued so far; a window or chunk numbers its own from here.
    exchanges: u64 = 0,
    /// Per rank: the exchanges whose `ready` record it has enqueued, and whose `done` record.
    posted: [world]std.atomic.Value(u64) = .{ .init(0), .init(0) },
    consumed: [world]std.atomic.Value(u64) = .{ .init(0), .init(0) },
    /// Set by a rank whose enqueue failed: the other stops waiting for it.
    abort: std.atomic.Value(bool) = .init(false),
    worker: ?std.Thread = null,
    job: Job = .idle,
    job_seq: std.atomic.Value(u32) = .init(0), // bumped to post a job; the worker waits on it
    done_seq: std.atomic.Value(u32) = .init(0), // bumped when the worker finishes one; the poster waits on it
    job_err: ?anyerror = null,
    /// TF_TP_PROFILE=1: each window synchronized after its enqueue, the host's and the whole step's time logged.
    profile: bool = false,
    prof: Profile = .{},
    no_exchange: bool = false,
    /// TF_TP_EVENTS=1: every exchange through events and copies, the fused kernel unused (a check of the two paths).
    events_only: bool = false,

    /// A fused exchange's poll gives up after this many clock ticks (about two seconds): the peer never arrived.
    const xchg_limit: i64 = 4_000_000_000;

    /// A fused exchange that timed out since the last check: the pair's state is no longer trusted.
    pub fn failedExchange(l: *Link) bool {
        var any = false;
        for (&l.rank) |*r| {
            const w = std.mem.bytesAsValue(u32, r.failed.bytes[0..4]);
            if (@atomicLoad(u32, w, .acquire) != 0) any = true;
        }
        return any;
    }

    pub const Profile = struct {
        n: u64 = 0,
        streams: u64 = 0,
        enqueue_ns: u64 = 0,
        step_ns: u64 = 0,

        pub fn note(p: *Profile, streams: usize, enqueue: u64, step: u64) void {
            p.n += 1;
            p.streams += streams;
            p.enqueue_ns += enqueue;
            p.step_ns += step;
            if (p.n % 200 != 0) return;
            const n: f64 = @floatFromInt(p.n);
            std.log.info("tp profile: {d} windows, {d:.2} streams, enqueue {d:.3} ms, step {d:.3} ms", .{ p.n, @as(f64, @floatFromInt(p.streams)) / n, @as(f64, @floatFromInt(p.enqueue_ns)) / n / 1e6, @as(f64, @floatFromInt(p.step_ns)) / n / 1e6 });
            p.* = .{};
        }
    };

    pub const Rank = struct {
        ctx: *const cuda.Context,
        /// The fused exchange's mapped host memory: this rank's slot (two parities) and flag, and the device
        /// addresses this rank's kernel reads its own and the peer's at.
        small: [2]cuda.HostBuffer,
        flag: cuda.HostBuffer,
        failed: cuda.HostBuffer,
        small_dev: [2]u64 = .{ 0, 0 },
        flag_dev: u64 = 0,
        failed_dev: u64 = 0,
        peer_small_dev: [2]u64 = .{ 0, 0 },
        peer_flag_dev: u64 = 0,
        own32: cuda.DeviceBuffer, // a MoE block's fp32 partial
        peer: cuda.DeviceBuffer, // the other rank's partial, copied in
        slot: [2]cuda.HostBuffer, // this rank's partial, staged; one a parity
        ready: [2]cuda.Event, // this rank's partial is in its slot
        done: [2]cuda.Event, // this rank has copied the peer's slot in
    };

    /// Each rank's buffers and events in its own context, then rank 1's thread; `rows` the widest exchange.
    pub fn init(gpa: std.mem.Allocator, io: std.Io, ctxs: [world]*const cuda.Context, rows: usize, hidden: usize) !*Link {
        const l = try gpa.create(Link);
        errdefer gpa.destroy(l);
        l.* = .{ .gpa = gpa, .io = io, .rank = undefined, .rows = rows, .hidden = hidden, .profile = std.c.getenv("TF_TP_PROFILE") != null, .no_exchange = std.c.getenv("TF_TP_NOXCHG") != null, .events_only = std.c.getenv("TF_TP_EVENTS") != null };
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
            const small = state.max_rows * hidden * 4;
            r.small[0] = try cuda.HostBuffer.allocMapped(ctx.d, small);
            r.small[1] = try cuda.HostBuffer.allocMapped(ctx.d, small);
            r.flag = try cuda.HostBuffer.allocMapped(ctx.d, 64);
            r.failed = try cuda.HostBuffer.allocMapped(ctx.d, 64);
            @memset(r.flag.bytes, 0);
            @memset(r.failed.bytes, 0);
            made += 1;
        }
        // each kernel addresses mapped memory in its own context: its own slot and flag, and the peer's
        for (&l.rank, 0..) |*r, i| {
            try r.ctx.makeCurrent();
            const o = &l.rank[1 - i];
            for (0..2) |p| {
                r.small_dev[p] = try r.small[p].device();
                r.peer_small_dev[p] = try o.small[p].device();
            }
            r.flag_dev = try r.flag.device();
            r.peer_flag_dev = try o.flag.device();
            r.failed_dev = try r.failed.device();
        }
        try ctxs[0].makeCurrent();
        l.worker = try std.Thread.spawn(.{ .stack_size = 8 << 20 }, workerMain, .{l});
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
        r.small[0].free();
        r.small[1].free();
        r.flag.free();
        r.failed.free();
        r.peer.free();
        r.own32.free();
    }

    pub fn deinit(l: *Link) void {
        if (l.worker) |t| {
            l.post(.quit);
            t.join();
        }
        for (&l.rank) |*r| freeRank(r);
        l.rank[0].ctx.makeCurrent() catch {};
        l.gpa.destroy(l);
    }

    fn post(l: *Link, job: Job) void {
        l.job = job;
        l.job_err = null;
        _ = l.job_seq.fetchAdd(1, .release);
        l.io.futexWake(u32, &l.job_seq.raw, 1);
    }

    /// Waits for the worker to finish the posted job, then its error if it had one.
    fn joinJob(l: *Link, seq: u32) !void {
        while (true) {
            const now = l.done_seq.load(.acquire);
            if (now == seq) break;
            l.io.futexWaitUncancelable(u32, &l.done_seq.raw, now);
        }
        if (l.job_err) |e| return e;
    }

    fn workerMain(l: *Link) void {
        l.rank[1].ctx.makeCurrent() catch |e| {
            std.log.err("tp rank 1 thread: cuCtxSetCurrent: {s}", .{@errorName(e)});
            l.abort.store(true, .release);
        };
        var seen: u32 = 0;
        while (true) {
            while (l.job_seq.load(.acquire) == seen) l.io.futexWaitUncancelable(u32, &l.job_seq.raw, seen);
            seen = l.job_seq.load(.acquire);
            const job = l.job;
            const result: anyerror!void = switch (job) {
                .idle => {},
                .quit => {
                    l.done_seq.store(seen, .release);
                    l.io.futexWake(u32, &l.done_seq.raw, 1);
                    return;
                },
                .window => |w| rankWindow(l, 1, w.fs[1], w.segs[1], w.base),
                .chunk => |c| rankChunk(l, 1, c.fs[1], c.rows, c.pos, c.base),
            };
            if (result) |_| {} else |e| {
                l.job_err = e;
                l.abort.store(true, .release);
            }
            l.done_seq.store(seen, .release);
            l.io.futexWake(u32, &l.done_seq.raw, 1);
        }
    }

    /// Spins until `v` reaches `want` (the peer's enqueue), or the peer gave up.
    fn await(l: *Link, v: *const std.atomic.Value(u64), want: u64) !void {
        var spins: usize = 0;
        while (v.load(.acquire) < want) {
            if (l.abort.load(.acquire)) return error.TpPeerFailed;
            spins += 1;
            if (spins < 4096) std.atomic.spinLoopHint() else std.Thread.yield() catch {};
        }
    }

    /// Rank `r`'s side of exchange `k`: its partial out to its slot, then the peer's in, summed into `out`.
    fn exchange(l: *Link, r: usize, f: *const Forward, k: u64, src: u64, wide: bool, out: u64, n: usize) !void {
        if (n <= state.max_rows * l.hidden and !l.events_only) {
            // a window's rows: the one-kernel exchange through mapped host memory, no host or event between ranks
            const me = &l.rank[r];
            const p: usize = @intCast(k % 2);
            try f.ops.tpXchg(src, wide, me.small_dev[p], me.peer_small_dev[p], me.flag_dev, me.peer_flag_dev, k + 1, out, n, xchg_limit, me.failed_dev);
            // the event path's host counters stay current: its waits are already met by stream order
            l.posted[r].store(k + 1, .release);
            l.consumed[r].store(k + 1, .release);
            return;
        }
        const bytes = n * @as(usize, if (wide) 4 else 2);
        if (n > l.rows * l.hidden) return error.ExchangeTooWide;
        const p: usize = @intCast(k % 2);
        const me = &l.rank[r];
        const other = &l.rank[1 - r];
        // the peer's copy-in of this parity's slot from exchange k - 2 is enqueued before this wait means it
        if (k >= 2) try l.await(&l.consumed[1 - r], k - 1);
        try f.ops.s.wait(other.done[p]);
        try f.ops.download(me.slot[p].bytes[0..bytes], src);
        try me.ready[p].record(f.ops.s);
        l.posted[r].store(k + 1, .release);
        try l.await(&l.posted[1 - r], k + 1);
        try f.ops.s.wait(other.ready[p]);
        try f.ops.upload(me.peer.ptr, other.slot[p].bytes[0..bytes]);
        try me.done[p].record(f.ops.s);
        l.consumed[r].store(k + 1, .release);
        try f.ops.tpSum2(src, wide, me.peer.ptr, wide, out, n);
    }

    /// One block's partials met on rank `r`: a MoE block's from its fp32 partial, any other's from its b.delta; the
    /// sum lands in b.delta, so the next norm adds it as a dense delta.
    fn meet(l: *Link, r: usize, f: *const Forward, k: u64, moe: bool, rows: usize) !void {
        if (l.no_exchange) return; // TF_TP_NOXCHG (profiling only): each rank keeps its own partial; output is wrong
        const src = if (moe) l.rank[r].own32.ptr else f.b.delta;
        try l.exchange(r, f, k, src, moe, f.b.delta, rows * f.c.hidden);
    }

    /// Posts rank 1's part to the worker; rank 0's part runs on the caller's thread next, then `finish`.
    fn start(l: *Link, job: Job) u32 {
        l.abort.store(false, .release);
        const seq = l.job_seq.load(.acquire) + 1;
        l.post(job);
        return seq;
    }

    /// Rank 0's result beside the worker's, once both enqueues are done (not the GPUs).
    fn finish(l: *Link, seq: u32, rank0: anyerror!void) !void {
        if (rank0) |_| {} else |_| l.abort.store(true, .release);
        const rank1 = l.joinJob(seq);
        try rank0;
        try rank1;
    }
};

/// Rank `r`'s window: its embedding, every block's share with the exchange after it, and (rank 0) the tail.
fn rankWindow(l: *Link, r: usize, f: *const Forward, segs: []const Seg, base: u64) !void {
    var rows: usize = 0;
    for (segs) |s| rows += s.rows;
    if (rows == 0 or rows > state.max_rows) return error.WindowTooWide;
    for (segs) |s| try f.embed(s.b.ids, f.b.emb + s.row0 * @as(u64, f.c.hidden) * 2, s.rows);
    var x: u64 = f.b.emb;
    var delta: Delta = .none;
    var mj: usize = 0;
    var aj: usize = 0;
    const c = f.c;
    for (0..c.layers) |li| {
        const moe = c.kinds[li] == .moe;
        const blk = f.w.blocks[li];
        try f.norm(&x, delta, blk.norm, rows, true);
        if (moe) {
            try f.moeTp(blk.moe, rows, false, l.rank[r].own32.ptr);
        } else try f.blockMixer(segs, rows, blk, &delta, &mj, &aj);
        try l.meet(r, f, base + li, moe, rows);
        delta = .dense;
    }
    if (r != 0) return;
    try f.norm(&x, delta, f.w.norm_f, rows, true);
    try f.tailOnly(segs, rows);
}

/// Rank `r`'s prompt chunk of `rows` tokens at `pos`: every block's share with the exchange after it, and (rank 0)
/// the final norm, head and sample. Its p_ids already hold the chunk.
fn rankChunk(l: *Link, r: usize, f: *const Forward, rows: usize, pos: usize, base: u64) !void {
    if (rows > l.rows) return error.ChunkTooWide;
    var w: Walk = .{ .rows = rows, .pos = pos };
    try f.chunkBegin(&w);
    const c = f.c;
    for (0..c.layers) |li| {
        const moe = c.kinds[li] == .moe;
        try f.chunkPre(&w, li);
        try f.chunkMixer(&w, li);
        if (moe) {
            try f.moeTp(f.w.blocks[li].moe, rows, true, l.rank[r].own32.ptr);
        } else try f.chunkPost(&w, li);
        try l.meet(r, f, base + li, moe, rows);
        w.delta = .dense;
    }
    if (r == 0) try f.chunkFinish(&w);
}

/// One window over both ranks (each rank's own buffers in `segs[r]`, the same rows; their ids and meta already
/// there): every block on both, its partials met, then rank 0's final norm, head and sample. Rank 0's context is
/// current on entry and on return.
pub fn window(l: *Link, fs: [world]*const Forward, segs: [world][]const Seg) !void {
    if (l.failedExchange()) return error.TpExchangeTimedOut;
    const base = l.exchanges;
    l.exchanges += fs[0].c.layers;
    const seq = l.start(.{ .window = .{ .fs = fs, .segs = segs, .base = base } });
    try l.finish(seq, rankWindow(l, 0, fs[0], segs[0], base));
}

/// One prompt chunk of `rows` tokens at `pos` over both ranks: every block on both, its partials met, then rank 0's
/// final norm, head and sample. Each rank's p_ids already hold the chunk. Rank 0's context is current.
pub fn chunk(l: *Link, fs: [world]*const Forward, rows: usize, pos: usize) !void {
    const base = l.exchanges;
    l.exchanges += fs[0].c.layers;
    const seq = l.start(.{ .chunk = .{ .fs = fs, .rows = rows, .pos = pos, .base = base } });
    try l.finish(seq, rankChunk(l, 0, fs[0], rows, pos, base));
}
