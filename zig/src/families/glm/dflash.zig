//! DFlash2 drafts (--drafter): incoai/GLM-5.3-Flash-DFlash2 (CC BY-NC-ND 4.0, read unmodified), after z-lab (MIT).
const std = @import("std");
const cuda = @import("cuda");
const mm = @import("cuda_mm.zig");
const q4 = @import("cuda_q4.zig");
const taps_mod = @import("cuda_taps.zig");
const select = @import("dflash_select.zig");
const block = @import("dflash_block.zig");

pub const BLOCK: usize = 8;
pub const MASK: u32 = 154856;
pub const WIN: i64 = 2048; // the context window (and a ring's slots)
pub const LAYERS: usize = 5;
pub const D: usize = 4096;
pub const KVW: usize = 1024;
pub const QKV: usize = 4096 + 2 * KVW;
pub const FF: usize = 12288;
pub const RANK: usize = 256;
pub const TOPK: usize = 16;
pub const THETA: f32 = 10000;
pub const EPS: f32 = 1e-5;
pub const SCALE: f32 = 0.08838834764831845; // 128^-0.5
pub const VOCAB: usize = 154880;
pub const QH: usize = 32;
pub const HD: usize = 128;
pub const CH: usize = 16; // key chunks of the chunked attention (df_attn_part)
pub const CW: usize = 2 * TOPK; // a row's gathered ints: 16 ids, then 16 values' bits
pub const MAX_S: usize = 16; // streams one block pass drafts for
pub const MAX_M: usize = MAX_S * BLOCK; // block rows at most
pub const MAX_CTX: usize = 2048; // context rows one ingest takes (a prompt chunk's)
pub const STEP_ROWS: usize = @import("cuda_verify_plan.zig").Plan.dflash7.rows(); // canonical shared verify tap capacity
pub const TAP_ROW: usize = taps_mod.layers.len * D * 2; // bytes of one row's taps

/// Device bytes a stream's context ring takes (K and V of every layer).
pub const ring_bytes: u64 = LAYERS * 2 * @as(u64, @intCast(WIN)) * KVW * 2;
pub const snapshot_magic = "DF2S0001";
pub const snapshot_version: u32 = 1;
pub const snapshot_header_bytes: usize = 256;
pub const snapshot_bytes: u64 = snapshot_header_bytes + ring_bytes;

pub const SnapshotHeader = extern struct {
    magic: [8]u8,
    version: u32,
    reserved: u32 = 0,
    from: i64,
    pos: i64,
    filled: u64,
    v_step: u64,
    v_row0: u64,
    v_rows: u64,
    v_pos0: i64,
};

comptime {
    std.debug.assert(@sizeOf(SnapshotHeader) <= snapshot_header_bytes);
}

pub fn ringSlotOffset(layer: usize, kv: usize, pos: i64) u64 {
    const slot: u64 = @intCast(@mod(pos, WIN));
    const plane = @as(u64, @intCast(WIN)) * KVW * 2;
    return (layer * 2 + kv) * plane + slot * KVW * 2;
}

fn filledCount(from: i64, end: usize) u64 {
    const last: i64 = @intCast(end);
    if (last <= from) return 0;
    return @intCast(@min(last - from, WIN));
}

fn validStart(from: i64, end: usize) i64 {
    const last: i64 = @intCast(end);
    if (last <= from) return from;
    const span = last - from;
    return if (span > WIN) last - WIN else from;
}

/// Version 1 snapshots store unused tap fields and header padding as zero. Refuse anything else before a slot is touched.
fn parseSnapshotHeader(raw: *const [snapshot_header_bytes]u8, end: usize) !SnapshotHeader {
    const header = std.mem.bytesToValue(SnapshotHeader, raw[0..@sizeOf(SnapshotHeader)]);
    if (!std.mem.eql(u8, &header.magic, snapshot_magic) or header.version != snapshot_version or header.reserved != 0)
        return error.BadDraftSnapshot;
    if (header.v_step != 0 or header.v_row0 != 0 or header.v_rows != 0 or header.v_pos0 != 0)
        return error.BadDraftSnapshot;
    if (!std.mem.allEqual(u8, raw[@sizeOf(SnapshotHeader)..], 0))
        return error.BadDraftSnapshot;
    if (header.from < 0)
        return error.BadDraftSnapshot;
    if (header.pos != @as(i64, @intCast(end)) or header.from > header.pos or header.filled != filledCount(header.from, end))
        return error.BadDraftSnapshot;
    return header;
}

/// glm_dflash's functions.
pub const Kernels = struct {
    tap: cuda.Function,
    block_qkv: cuda.Function,
    ctx_kv: cuda.Function,
    conv: cuda.Function,
    swiglu: cuda.Function,
    attn_part: cuda.Function,
    attn_merge: cuda.Function,
    sum2: cuda.Function,
    top16_part: cuda.Function,
    top16_merge: cuda.Function,
    gsum: cuda.Function,

    pub fn load(m: cuda.Module) !Kernels {
        var k: Kernels = undefined;
        inline for (@typeInfo(Kernels).@"struct".field_names) |n| @field(k, n) = try m.function("df_" ++ n);
        return k;
    }
};

/// A drafter linear: 4 bits (--drafter-bits 4, the default; they only propose) or bf16 tiled for glm_mm.
pub const Lin = union(enum) { q4: q4.Q4, bf16: mm.Tiled };

/// down and the selector's projection stay bf16 (their fp32 outputs are gathered or read on the host).
pub const LayerW = struct { in_norm: u64, post_norm: u64, qn: u64, kn: u64, a_base: u64, m_base: u64, qkv: Lin, kv: Lin, o: Lin, gu: Lin, down: mm.Tiled, a_proj: Lin, m_proj: Lin };

/// A stream's drafter state by slot: its ring, where its context starts, and its last verify's tap rows.
pub const Slot = struct {
    ring: ?cuda.DeviceBuffer = null,
    from: i64 = 0, // the first position the ring holds for this sequence
    v_step: u64 = 0, // the verify that wrote its rows (Drafter.verifies then)
    v_row0: usize = 0,
    v_rows: usize = 0,
    v_pos0: i64 = 0,
};

/// One stream's request: `n` rows of its last verify into its context first (0: none), the anchor, rank 0's `out`.
pub const Req = struct {
    slot: u32,
    pos: i64,
    n: usize,
    anchor: u32,
    /// Absorb-only requests retain accepted tap context without launching a proposal block.
    propose: bool = true,
    out: ?[]u32 = null,
    probs: ?[]f64 = null,
};

/// Context runs: `n` tap rows at `taps` (consecutive), positions pos0.., into `ring`.
pub const Run = struct { ring: u64, taps: u64, n: usize, pos0: i64 };

/// Every rank runs the ingest and the block pass on its half of the MLP and head; rank 0 selects the drafts.
pub const Drafter = struct {
    capture_lifetime: u64 = 0,
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    module: cuda.Module,
    k: Kernels,
    weights: cuda.DeviceBuffer,
    layers: [LAYERS]LayerW = undefined,
    fc: Lin = undefined,
    hproj: mm.Tiled = undefined,
    hnorm: u64 = 0,
    norm: u64 = 0,
    pred: []u16 = &.{}, // the selector's codebooks, bf16 [VOCAB][RANK] (host: it reads 16 rows a draft)
    succ: []u16 = &.{},
    split: bool, // the MLP split across two ranks (gate and up by rows, down by K)
    ff: usize, // the MLP width this rank holds
    rank: usize,
    world: usize,
    temp: f64 = 1.0, // the selector's softmax temperature for the drafts' chances
    scratch: cuda.DeviceBuffer = undefined,
    slots: std.ArrayList(Slot) = .empty,
    verifies: u64 = 0, // verifies so far: a slot's rows are still in taps_step while its v_step is the latest
    lost: u64 = 0, // drafts whose verify rows were gone (their context restarted)
    calls: u64 = 0,
    log: u64 = 0, // TF_GLM_DFLASH2_LOG=n: the first n rounds' drafts logged (rank 0)
    prof: ?std.Io = null, // TF_GLM_DFLASH2_PROF=1: each phase synchronized and timed, logged every 64 rounds
    phase_ms: [4]f64 = @splat(0), // ingest, tables, block pass, selector
    // device scratch (carved by dflash_load)
    x: u64 = 0,
    y: u64 = 0,
    y2: u64 = 0,
    dyn: u64 = 0,
    qkv: u64 = 0,
    q: u64 = 0,
    kb: u64 = 0,
    vb: u64 = 0,
    att: u64 = 0,
    a: u64 = 0,
    gu: u64 = 0,
    act: u64 = 0,
    hn: u64 = 0,
    xs: u64 = 0,
    g: u64 = 0,
    logits: u64 = 0,
    part: u64 = 0, // the linears' K-slice partials
    gs: u64 = 0, // a 4-bit linear's input group sums [MAX_CTX][5 * D / 64]
    sums: u64 = 0, // the split down's gathered fp32 partials [2][MAX_M][D]
    cin: u64 = 0,
    c1: u64 = 0,
    ctx: u64 = 0,
    ckv: u64 = 0,
    taps_step: u64 = 0,
    taps_prefill: u64 = 0,
    ids: u64 = 0,
    bpos: u64 = 0,
    rings: u64 = 0,
    starts: u64 = 0,
    froms: u64 = 0,
    cpos: u64 = 0,
    crings: u64 = 0,
    tk: u64 = 0,
    gtk: u64 = 0,
    pv: u64 = 0,
    pi: u64 = 0,
    pm: u64 = 0,
    pl: u64 = 0,
    pacc: u64 = 0,
    // host tables
    h_ids: [MAX_M]i32 = undefined,
    h_bpos: [MAX_M]i32 = undefined,
    h_rings: [MAX_S]u64 = undefined,
    h_starts: [MAX_S]i64 = undefined,
    h_froms: [MAX_S]i64 = undefined,
    h_cpos: []i64 = &.{},
    h_crings: []u64 = &.{},
    cand: []i32 = &.{}, // all ranks' gathered candidates, [world][MAX_M][CW]
    gh: []f32 = &.{}, // the selector's hidden projections [MAX_M][RANK]

    pub fn deinit(dr: *Drafter) void {
        for (dr.slots.items) |*s| if (s.ring) |*r| r.free();
        dr.slots.deinit(dr.gpa);
        dr.scratch.free();
        dr.weights.free();
        dr.module.unload();
        const g = dr.gpa;
        for ([_][]u16{ dr.pred, dr.succ }) |b| g.free(b);
        g.free(dr.h_cpos);
        g.free(dr.h_crings);
        g.free(dr.cand);
        g.free(dr.gh);
        g.destroy(dr);
    }

    /// TF_GLM_DFLASH2_PROF: the stream idle, the time since `since` added to its phase; the new mark.
    fn mark(dr: *Drafter, e: anytype, since: ?struct { i96, usize }) !i96 {
        const io = dr.prof orelse return 0;
        try e.stream.synchronize();
        const now = std.Io.Clock.awake.now(io).toNanoseconds();
        if (since) |s| dr.phase_ms[s[1]] += @as(f64, @floatFromInt(now - s[0])) / 1e6;
        return now;
    }

    /// Where a forward's taps go: a prompt chunk's rows or a verify's.
    pub fn taps(dr: *const Drafter, chunk: bool) taps_mod.Taps {
        return .{ .lifetime = dr.capture_lifetime, .f = dr.k.tap, .out = if (chunk) dr.taps_prefill else dr.taps_step };
    }

    fn slot(dr: *Drafter, i: u32) !*Slot {
        if (i >= dr.slots.items.len) try dr.slots.appendNTimes(dr.gpa, .{}, i + 1 - dr.slots.items.len);
        const s = &dr.slots.items[i];
        if (s.ring == null) s.ring = try cuda.DeviceBuffer.alloc(dr.d, ring_bytes);
        return s;
    }

    /// A fresh sequence on slot `i` whose rows are tapped from position `from` (0, or a kept prompt state's).
    pub fn restart(dr: *Drafter, i: u32, from: usize) !void {
        const s = try dr.slot(i);
        s.from = @intCast(from);
        s.v_rows = 0;
        s.v_step = 0;
        s.v_row0 = 0;
        s.v_pos0 = 0;
    }

    pub fn snapshotBytes(_: *const Drafter) u64 {
        return snapshot_bytes;
    }

    pub fn snapshotReady(dr: *const Drafter, i: u32, end: usize) bool {
        if (i >= dr.slots.items.len) return end == 0;
        const s = dr.slots.items[i];
        if (s.from > @as(i64, @intCast(end))) return false;
        if (s.v_rows != 0) return false;
        if (s.ring == null) return end == @as(usize, @intCast(s.from));
        return true;
    }

    pub fn save(dr: *Drafter, i: u32, destination: u64, end: usize, stream: cuda.Stream) !void {
        if (!dr.snapshotReady(i, end)) return error.DraftSnapshotNotReady;
        const s: Slot = if (i < dr.slots.items.len) dr.slots.items[i] else .{};
        try dr.d.check(dr.d.api.cuMemsetD8Async(destination, 0, snapshot_bytes, stream.handle), "cuMemsetD8Async");
        var raw: [snapshot_header_bytes]u8 = @splat(0);
        var header: SnapshotHeader = .{
            .magic = snapshot_magic[0..8].*,
            .version = snapshot_version,
            .from = s.from,
            .pos = @intCast(end),
            .filled = filledCount(s.from, end),
            .v_step = 0,
            .v_row0 = 0,
            .v_rows = 0,
            .v_pos0 = 0,
        };
        @memcpy(raw[0..@sizeOf(SnapshotHeader)], std.mem.asBytes(&header));
        try dr.d.check(dr.d.api.cuMemcpyHtoDAsync_v2(destination, &raw, snapshot_header_bytes, stream.handle), "cuMemcpyHtoDAsync");
        if (s.ring) |ring| {
            var p = validStart(s.from, end);
            const last: i64 = @intCast(end);
            while (p < last) : (p += 1) {
                for (0..LAYERS) |layer| {
                    for (0..2) |kv| {
                        const off = ringSlotOffset(layer, kv, p);
                        try dr.d.check(dr.d.api.cuMemcpyDtoDAsync_v2(destination + snapshot_header_bytes + off, ring.ptr + off, KVW * 2, stream.handle), "cuMemcpyDtoDAsync");
                    }
                }
            }
        }
        try dr.d.check(dr.d.api.cuStreamSynchronize(stream.handle), "cuStreamSynchronize");
    }

    pub fn restore(dr: *Drafter, i: u32, source: u64, end: usize, stream: cuda.Stream) !void {
        var raw: [snapshot_header_bytes]u8 = undefined;
        try dr.d.check(dr.d.api.cuMemcpyDtoH_v2(&raw, source, snapshot_header_bytes), "cuMemcpyDtoH");
        const header = try parseSnapshotHeader(&raw, end);
        const s = try dr.slot(i);
        s.from = std.math.maxInt(i64);
        s.v_rows = 0;
        s.v_step = 0;
        s.v_row0 = 0;
        s.v_pos0 = 0;
        errdefer {
            s.from = std.math.maxInt(i64);
            s.v_rows = 0;
        }
        try dr.d.check(dr.d.api.cuMemcpyDtoDAsync_v2(s.ring.?.ptr, source + snapshot_header_bytes, ring_bytes, stream.handle), "cuMemcpyDtoDAsync");
        try dr.d.check(dr.d.api.cuStreamSynchronize(stream.handle), "cuStreamSynchronize");
        s.from = header.from;
    }

    /// A verify on slot `i` wrote `rows` tap rows at row `row0` of taps_step, for positions pos0...
    pub fn noteVerify(dr: *Drafter, i: u32, row0: usize, rows: usize, pos0: usize, first: bool) !void {
        if (rows == 0 or row0 > STEP_ROWS or rows > STEP_ROWS - row0) return error.DrafterContextTooWide;
        if (first) dr.verifies += 1; // a shared verify's parts share one
        const s = try dr.slot(i);
        s.* = .{ .ring = s.ring, .from = s.from, .v_step = dr.verifies, .v_row0 = row0, .v_rows = rows, .v_pos0 = @intCast(pos0) };
    }

    /// A prompt chunk on slot `i` (rows start.. of a `len`-id prompt): its rows within the last window into the ring.
    pub fn prompt(dr: anytype, e: anytype, i: u32, start: usize, rows: usize, len: usize) !void {
        return dr.promptSegment(e, i, start, rows, len, 0);
    }

    /// The same ordinary context ingest, with this prompt's rows at row0 in the batched tap buffer.
    pub const promptSegment = ingestPrompt;

    /// The runs' rows through fc, the hidden norm and each layer's k, v into their rings.
    pub fn ingest(dr: *Drafter, e: anytype, runs: []const Run) !void {
        var N: usize = 0;
        for (runs) |r| N += r.n;
        if (N == 0) return;
        if (N > MAX_CTX) return error.DrafterContextTooWide;
        var row: usize = 0;
        for (runs) |r| {
            try block.copy(e, dr.cin + row * TAP_ROW, r.taps, r.n * TAP_ROW);
            for (0..r.n) |j| {
                dr.h_cpos[row + j] = r.pos0 + @as(i64, @intCast(j));
                dr.h_crings[row + j] = r.ring;
            }
            row += r.n;
        }
        try block.upload(dr, e, dr.cpos, std.mem.sliceAsBytes(dr.h_cpos[0..N]));
        try block.upload(dr, e, dr.crings, std.mem.sliceAsBytes(dr.h_crings[0..N]));
        try block.ingestPass(dr, e, N);
    }

    /// Kept rows into each context, then the block pass for every stream (every rank); rank 0 selects 7 drafts each.
    pub fn draft(dr: *Drafter, e: anytype, reqs: []const Req) !void {
        if (reqs.len == 0) return;
        if (reqs.len > MAX_S) return error.TooManyStreams;
        var tm = try dr.mark(e, null);
        var runs: [MAX_S]Run = undefined;
        var nr: usize = 0;
        for (reqs) |q| {
            const s = try dr.slot(q.slot);
            if (q.n == 0) continue;
            if (s.v_step != dr.verifies or q.n > s.v_rows or s.v_pos0 + @as(i64, @intCast(q.n)) != q.pos) {
                s.from = q.pos; // its verify rows were overwritten (or never written): its context restarts here
                dr.lost += 1;
                continue;
            }
            runs[nr] = .{ .ring = s.ring.?.ptr, .taps = dr.taps_step + s.v_row0 * TAP_ROW, .n = q.n, .pos0 = s.v_pos0 };
            nr += 1;
        }
        try dr.ingest(e, runs[0..nr]);
        tm = try dr.mark(e, .{ tm, 0 });
        // Context absorption above belongs to every request. Only positive-depth requests enter the block.
        var active: [MAX_S]Req = undefined;
        var S: usize = 0;
        for (reqs) |q| if (q.propose) {
            active[S] = q;
            S += 1;
        };
        if (S == 0) return;
        const proposals = active[0..S];
        for (proposals, 0..) |q, i| {
            const s = &dr.slots.items[q.slot];
            dr.h_rings[i] = s.ring.?.ptr;
            dr.h_starts[i] = q.pos;
            dr.h_froms[i] = s.from;
            for (0..BLOCK) |t| {
                dr.h_ids[i * BLOCK + t] = @intCast(if (t == 0) q.anchor else MASK);
                dr.h_bpos[i * BLOCK + t] = @intCast(q.pos + @as(i64, @intCast(t)));
            }
        }
        const M = S * BLOCK;
        try block.upload(dr, e, dr.ids, std.mem.sliceAsBytes(dr.h_ids[0..M]));
        try block.upload(dr, e, dr.bpos, std.mem.sliceAsBytes(dr.h_bpos[0..M]));
        try block.upload(dr, e, dr.rings, std.mem.sliceAsBytes(dr.h_rings[0..S]));
        try block.upload(dr, e, dr.starts, std.mem.sliceAsBytes(dr.h_starts[0..S]));
        try block.upload(dr, e, dr.froms, std.mem.sliceAsBytes(dr.h_froms[0..S]));
        const rank0 = proposals[0].out != null;
        tm = try dr.mark(e, .{ tm, 1 });
        try block.pass(dr, e, S, rank0);
        tm = try dr.mark(e, .{ tm, 2 });
        if (rank0) {
            try block.download(dr, e, std.mem.sliceAsBytes(dr.cand[0..@import("dflash_candidates.zig").Layout.init(dr.world, M, CW).read]), dr.gtk);
            try block.download(dr, e, std.mem.sliceAsBytes(dr.gh[0 .. M * RANK]), dr.g);
            for (proposals, 0..) |q, i| select.select(dr, i, M, q);
            if (dr.log > dr.calls) for (proposals) |q| std.log.info("dflash2 round {d}: slot {d} pos {d} kept {d} anchor {d} drafts {any}", .{ dr.calls, q.slot, q.pos, q.n, q.anchor, q.out.?[0 .. BLOCK - 1] });
            if (dr.calls % 256 == 255) std.log.info("dflash2: {d} rounds, {d} contexts restarted", .{ dr.calls + 1, dr.lost });
        }
        _ = try dr.mark(e, .{ tm, 3 });
        dr.calls += 1;
        if (dr.prof != null and dr.calls % 64 == 0) {
            const k: f64 = 64;
            std.log.info("dflash2 ms a round: ingest {d:.2} tables {d:.2} block {d:.2} select {d:.2}", .{ dr.phase_ms[0] / k, dr.phase_ms[1] / k, dr.phase_ms[2] / k, dr.phase_ms[3] / k });
            dr.phase_ms = @splat(0);
        }
    }
};

/// Prompt context retained in the window; row0 is only a tap-buffer offset, never a sequence position.
pub fn promptRun(ring: u64, taps: u64, start: usize, rows: usize, len: usize, row0: usize) !?Run {
    if (start > len or rows > len - start) return error.BadPromptContext;
    if (rows > MAX_CTX or row0 > MAX_CTX - rows) return error.DrafterContextTooWide;
    const keep_from = len -| @as(usize, @intCast(WIN));
    if (start + rows <= keep_from) return null;
    const lo = keep_from -| start;
    return .{ .ring = ring, .taps = taps + (row0 + lo) * TAP_ROW, .n = rows - lo, .pos0 = @intCast(start + lo) };
}

pub fn ingestPrompt(dr: anytype, e: anytype, i: u32, start: usize, rows: usize, len: usize, row0: usize) !void {
    const s = try dr.slot(i);
    const run = (try promptRun(s.ring.?.ptr, dr.taps_prefill, start, rows, len, row0)) orelse return;
    try dr.ingest(e, &.{run});
}

test "actual canonical absorb-only request ingests accepted tap rows before a later proposal without an empty block launch" {
    const Memory = @import("cuda_test_driver.zig").Memory;
    const Recording = struct {
        var ingest: usize = 0;
        var proposal: usize = 0;
        var copies: usize = 0;
        var last_taps: u64 = 0;
        var copied_rows: usize = 0;
        fn fill(comptime T: type, d: *const cuda.Driver) T {
            if (T == cuda.Function) return .{ .d = d, .handle = @ptrFromInt(1) };
            if (T == bool) return false;
            var out: T = undefined;
            switch (@typeInfo(T)) {
                .@"struct" => inline for (@typeInfo(T).@"struct".field_names) |name| {
                    @field(out, name) = fill(@TypeOf(@field(out, name)), d);
                },
                .array => for (&out) |*item| {
                    item.* = fill(@TypeOf(item.*), d);
                },
                else => @compileError("unexpected kernel descriptor field"),
            }
            return out;
        }
        fn copy(_: u64, src: u64, n: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
            copies += 1;
            last_taps = src;
            copied_rows = n / TAP_ROW;
            return 0;
        }
        fn upload(_: u64, _: ?*const anyopaque, _: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
            return 0;
        }
        fn launch(f: cuda.abi.Function, x: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: cuda.abi.Stream, _: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
            if (f == @as(cuda.abi.Function, @ptrFromInt(2))) ingest += 1;
            if (f == @as(cuda.abi.Function, @ptrFromInt(3))) {
                std.debug.assert(x == BLOCK); // one proposed stream, not an empty block
                proposal += 1;
            }
            return 0;
        }
    };
    var d = Memory.driver();
    d.api.cuMemcpyDtoDAsync_v2 = Recording.copy;
    d.api.cuMemcpyHtoDAsync_v2 = Recording.upload;
    d.api.cuLaunchKernel = Recording.launch;
    d.api.cuLaunchKernelEx = struct {
        fn launch(c: *const cuda.abi.LaunchConfig, f: cuda.abi.Function, args: ?[*]?*anyopaque, extra: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
            return Recording.launch(f, c.grid_x, c.grid_y, c.grid_z, c.block_x, c.block_y, c.block_z, c.shared_bytes, c.stream, args, extra);
        }
    }.launch;
    const Owner = struct {
        d: *const cuda.Driver,
        stream: cuda.Stream,
        m: struct { glue: @import("cuda_glue.zig").Kernels, mm: mm.Kernels, q4: q4.Kernels },
        w: struct { mtp: ?struct { draft: ?q4.Q4 }, embed: u64 },
        spans: []const u8 = &.{0},
        tp_group: ?struct {
            pub fn check(_: *@This()) !void {
                return error.UnexpectedCollective;
            }
            pub fn allGather(_: *@This(), _: cuda.Stream, _: u64, _: u64, _: usize) !void {
                return error.UnexpectedCollective;
            }
        } = null,
        lib: *const cuda.nccl.Library = undefined,
        comm: cuda.nccl.Comm = null,
        share: struct { rank: usize = 0, vocab: struct { lo: usize = 0 } = .{} } = .{},
    };
    var e: Owner = .{
        .d = &d,
        .stream = .{ .d = &d, .handle = null },
        .m = .{ .glue = Recording.fill(@import("cuda_glue.zig").Kernels, &d), .mm = Recording.fill(mm.Kernels, &d), .q4 = Recording.fill(q4.Kernels, &d) },
        .w = .{ .mtp = .{ .draft = .{ .w = 0, .s = 0, .b = 0, .n = 128, .k = D, .npad = 128 } }, .embed = 0 },
    };
    e.m.glue.embed_b16.handle = @ptrFromInt(3);
    var dr: Drafter = .{ .gpa = std.testing.allocator, .d = &d, .module = undefined, .k = Recording.fill(Kernels, &d), .weights = undefined, .split = false, .ff = FF, .rank = 0, .world = 1 };
    dr.k.ctx_kv.handle = @ptrFromInt(2);
    const dense: mm.Tiled = .{ .w = 0, .n = D, .k = D };
    dr.fc = .{ .bf16 = .{ .w = 0, .n = D, .k = 5 * D } };
    dr.hproj = dense;
    for (&dr.layers) |*layer| layer.* = .{ .in_norm = 0, .post_norm = 0, .qn = 0, .kn = 0, .a_base = 0, .m_base = 0, .qkv = .{ .bf16 = dense }, .kv = .{ .bf16 = dense }, .o = .{ .bf16 = dense }, .gu = .{ .bf16 = dense }, .down = dense, .a_proj = .{ .bf16 = dense }, .m_proj = .{ .bf16 = dense } };
    var positions: [MAX_CTX]i64 = undefined;
    var rings: [MAX_CTX]u64 = undefined;
    dr.h_cpos = &positions;
    dr.h_crings = &rings;
    dr.taps_step = 0x800000;
    const ring: cuda.DeviceBuffer = .{ .d = &d, .ptr = 0x100000, .len = ring_bytes, .borrowed = true };
    try dr.slots.append(std.testing.allocator, .{ .ring = ring, .from = 100 });
    defer dr.slots.deinit(std.testing.allocator);
    defer cuda.copy_source.dropSpares(&d, cuda.copy_source.current());
    Recording.ingest = 0;
    Recording.proposal = 0;
    Recording.copies = 0;
    try dr.noteVerify(0, 0, 3, 100, true);
    try dr.draft(&e, &.{.{ .slot = 0, .pos = 102, .n = 2, .anchor = 2, .propose = false }});
    try std.testing.expectEqual(@as(usize, LAYERS), Recording.ingest);
    try std.testing.expectEqual(@as(usize, 0), Recording.proposal);
    try std.testing.expectEqual(@as(u64, 0), dr.calls);
    try std.testing.expectEqual(@as(usize, 2), Recording.copied_rows);
    try std.testing.expectEqualSlices(i64, &.{ 100, 101 }, dr.h_cpos[0..2]);
    try std.testing.expectEqual(@as(u64, 0), dr.lost);
    // Real noteVerify then real draft: the next accepted row is contiguous with the absorb-only turn.
    try dr.noteVerify(0, 2, 1, 102, true);
    try dr.draft(&e, &.{.{ .slot = 0, .pos = 103, .n = 1, .anchor = 3 }});
    try std.testing.expectEqual(@as(usize, 2 * LAYERS), Recording.ingest);
    try std.testing.expectEqual(@as(usize, 1), Recording.proposal);
    try std.testing.expectEqual(@as(usize, 2), Recording.copies);
    try std.testing.expectEqual(dr.taps_step + 2 * TAP_ROW, Recording.last_taps);
    try std.testing.expectEqual(@as(i64, 102), dr.h_cpos[0]);
    try std.testing.expectEqual(@as(i64, 103), dr.h_starts[0]);
    try std.testing.expectEqual(@as(i64, 100), dr.h_froms[0]);
    try std.testing.expectEqual(@as(u64, 0), dr.lost);
    try std.testing.expectEqual(@as(u64, 1), dr.calls);
    try e.stream.synchronize(); // draft submits asynchronously; complete its owned staging before fixture teardown
    cuda.copy_source.dropSpares(&d, cuda.copy_source.current());
    try std.testing.expect(Memory.empty());
}

test "glm dflash: canonical snapshot save restores filled slots, zeros unused bytes, and refuses a bad header" {
    const Memory = @import("cuda_test_driver.zig").Memory;
    var d = Memory.driver();
    const stream: cuda.Stream = .{ .d = &d, .handle = null };
    var ring = try cuda.DeviceBuffer.alloc(&d, ring_bytes);
    defer ring.free();
    var other = try cuda.DeviceBuffer.alloc(&d, ring_bytes);
    defer other.free();
    @memset(Memory.slice(ring.ptr, ring.len), 0xa5);
    @memset(Memory.slice(other.ptr, other.len), 0x3c);
    const from: i64 = 10;
    var p: i64 = from;
    while (p < 18) : (p += 1) {
        const mark: u8 = 0x40 | @as(u8, @intCast(p - from));
        for (0..LAYERS) |layer| {
            for (0..2) |kv| {
                @memset(Memory.slice(ring.ptr + ringSlotOffset(layer, kv, p), KVW * 2), mark);
            }
        }
    }
    var dr: Drafter = .{ .gpa = std.testing.allocator, .d = &d, .module = undefined, .k = undefined, .weights = undefined, .split = false, .ff = 0, .rank = 0, .world = 1 };
    try dr.slots.append(std.testing.allocator, .{ .ring = ring, .from = from });
    try dr.slots.append(std.testing.allocator, .{ .ring = other, .from = 3, .v_rows = 2 });
    defer dr.slots.deinit(std.testing.allocator);
    try std.testing.expect(dr.snapshotReady(0, 18));
    try std.testing.expect(!dr.snapshotReady(1, 18));
    var snap = try cuda.DeviceBuffer.alloc(&d, snapshot_bytes);
    defer snap.free();
    @memset(Memory.slice(snap.ptr, snap.len), 0xaa);
    try dr.save(0, snap.ptr, 18, stream);
    try std.testing.expect(std.mem.indexOfScalar(u8, Memory.slice(snap.ptr, snap.len), 0xaa) == null);
    try std.testing.expect(std.mem.indexOfScalar(u8, Memory.slice(snap.ptr, snap.len), 0xa5) == null);
    @memset(Memory.slice(ring.ptr, ring.len), 0x11);
    dr.slots.items[0].from = 99;
    try dr.restore(0, snap.ptr, 18, stream);
    try std.testing.expectEqual(from, dr.slots.items[0].from);
    try std.testing.expectEqual(@as(usize, 0), dr.slots.items[0].v_rows);
    p = from;
    while (p < 18) : (p += 1) {
        const mark: u8 = 0x40 | @as(u8, @intCast(p - from));
        for (0..LAYERS) |layer| {
            for (0..2) |kv| {
                try std.testing.expect(std.mem.allEqual(u8, Memory.slice(ring.ptr + ringSlotOffset(layer, kv, p), KVW * 2), mark));
            }
        }
    }
    try std.testing.expect(std.mem.allEqual(u8, Memory.slice(ring.ptr + ringSlotOffset(0, 0, 0), KVW * 2), 0));
    try std.testing.expect(std.mem.allEqual(u8, Memory.slice(other.ptr, other.len), 0x3c));
    Memory.slice(snap.ptr, 8)[0] = 'X';
    try std.testing.expectError(error.BadDraftSnapshot, dr.restore(0, snap.ptr, 18, stream));
    try std.testing.expectEqual(from, dr.slots.items[0].from);
    const Failure = struct {
        var calls: usize = 0;
        var delegate: @TypeOf(d.api.cuMemcpyDtoDAsync_v2) = undefined;
        fn copy(dst: u64, src: u64, n: usize, stream_: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
            calls += 1;
            if (calls == 1) return cuda.abi.error_not_ready;
            return delegate(dst, src, n, stream_);
        }
    };
    Failure.calls = 0;
    Failure.delegate = d.api.cuMemcpyDtoDAsync_v2;
    d.api.cuMemcpyDtoDAsync_v2 = Failure.copy;
    Memory.slice(snap.ptr, 8)[0] = snapshot_magic[0];
    try std.testing.expectError(error.NotReady, dr.restore(0, snap.ptr, 18, stream));
    try std.testing.expectEqual(std.math.maxInt(i64), dr.slots.items[0].from);
    try std.testing.expect(!dr.snapshotReady(0, 18));
    try std.testing.expectEqual(@as(i64, 3), dr.slots.items[1].from);
}

test "glm dflash: restore refuses nonzero unused header fields, padding, and a negative from before slot mutation" {
    const Memory = @import("cuda_test_driver.zig").Memory;
    var d = Memory.driver();
    const stream: cuda.Stream = .{ .d = &d, .handle = null };
    var ring = try cuda.DeviceBuffer.alloc(&d, ring_bytes);
    defer ring.free();
    @memset(Memory.slice(ring.ptr, ring.len), 0xa5);
    var dr: Drafter = .{ .gpa = std.testing.allocator, .d = &d, .module = undefined, .k = undefined, .weights = undefined, .split = false, .ff = 0, .rank = 0, .world = 1 };
    try dr.slots.append(std.testing.allocator, .{ .ring = ring, .from = 10 });
    defer dr.slots.deinit(std.testing.allocator);
    var snap = try cuda.DeviceBuffer.alloc(&d, snapshot_bytes);
    defer snap.free();
    try dr.save(0, snap.ptr, 18, stream);
    const valid = try std.testing.allocator.dupe(u8, Memory.slice(snap.ptr, snapshot_header_bytes));
    defer std.testing.allocator.free(valid);
    const Count = struct {
        var copies: usize = 0;
        var delegate: @TypeOf(d.api.cuMemcpyDtoDAsync_v2) = undefined;
        fn copy(dst: u64, src: u64, n: usize, stream_: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
            copies += 1;
            return delegate(dst, src, n, stream_);
        }
    };
    Count.delegate = d.api.cuMemcpyDtoDAsync_v2;
    d.api.cuMemcpyDtoDAsync_v2 = Count.copy;
    const Kind = enum { v_step, v_row0, v_rows, v_pos0, padding, negative_from };
    for ([_]Kind{ .v_step, .v_row0, .v_rows, .v_pos0, .padding, .negative_from }) |kind| {
        @memcpy(Memory.slice(snap.ptr, snapshot_header_bytes), valid);
        var header = std.mem.bytesToValue(SnapshotHeader, valid[0..@sizeOf(SnapshotHeader)]);
        switch (kind) {
            .v_step => header.v_step = 7,
            .v_row0 => header.v_row0 = 3,
            .v_rows => header.v_rows = 2,
            .v_pos0 => header.v_pos0 = 11,
            .padding => {},
            .negative_from => {
                header.from = -1;
                header.filled = filledCount(-1, 18);
            },
        }
        @memcpy(Memory.slice(snap.ptr, @sizeOf(SnapshotHeader)), std.mem.asBytes(&header));
        if (kind == .padding) Memory.slice(snap.ptr, snapshot_header_bytes)[@sizeOf(SnapshotHeader)] = 1;
        dr.slots.items[0].from = 99;
        dr.slots.items[0].v_rows = 4;
        Count.copies = 0;
        try std.testing.expectError(error.BadDraftSnapshot, dr.restore(0, snap.ptr, 18, stream));
        try std.testing.expectEqual(@as(i64, 99), dr.slots.items[0].from);
        try std.testing.expectEqual(@as(usize, 4), dr.slots.items[0].v_rows);
        try std.testing.expectEqual(@as(usize, 0), Count.copies);
        try std.testing.expectEqual(@as(usize, 1), dr.slots.items.len);
    }
    var empty: Drafter = .{ .gpa = std.testing.allocator, .d = &d, .module = undefined, .k = undefined, .weights = undefined, .split = false, .ff = 0, .rank = 0, .world = 1 };
    defer empty.slots.deinit(std.testing.allocator);
    @memcpy(Memory.slice(snap.ptr, snapshot_header_bytes), valid);
    var dirty = std.mem.bytesToValue(SnapshotHeader, valid[0..@sizeOf(SnapshotHeader)]);
    dirty.v_rows = 1;
    @memcpy(Memory.slice(snap.ptr, @sizeOf(SnapshotHeader)), std.mem.asBytes(&dirty));
    Count.copies = 0;
    try std.testing.expectError(error.BadDraftSnapshot, empty.restore(0, snap.ptr, 18, stream));
    try std.testing.expectEqual(@as(usize, 0), empty.slots.items.len);
    try std.testing.expectEqual(@as(usize, 0), Count.copies);
}
