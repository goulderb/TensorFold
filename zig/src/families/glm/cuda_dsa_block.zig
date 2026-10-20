//! GLM-5.3-Flash's DSA block on CUDA, a rank's latent-path share (the Python family's dsa_block) to an fp32 partial.

const std = @import("std");
const cuda = @import("cuda");
const glue = @import("cuda_glue.zig");
const b16 = @import("cuda_b16.zig");
const latent = @import("cuda_latent.zig");
const sparse = @import("cuda_sparse.zig");
const cache = @import("cuda_cache.zig");

/// A prompt chunk's rows one dense latent attention call takes (geometry.MLA_PROMPT_ATT_ROWS).
pub const prompt_att_rows = 512;

/// The indexer's weights, replicated: [wk | weights_proj] [d + heads][dims], gate, key LayerNorm, pool bias, wq_b.
pub const Index = struct { kw: b16.Proj, gate: u64, ln_w: u64, ln_b: u64, ape: u64, qb: b16.Proj, heads: usize, d: usize };

pub const InputProjection = union(enum) {
    combined_bf16: u64,
    combined: b16.Proj,
    separate: struct { qa: b16.Proj, kva: b16.Proj },
};

/// One DSA layer on a rank of `heads` heads: input projection, norms, q_b, kv_b as wk/wv, o, indexer.
pub const Layer = struct {
    input: InputProjection,
    q_norm: u64,
    kv_norm: u64,
    q_b: b16.Proj,
    wk: u64,
    wv: u64,
    o: b16.Proj,
    index: Index,
    heads: usize,
    qk: usize,
    dv: usize,
    lw: usize,
    q_lora: usize,
    dims: usize,
    eps: f32,
    dense_limit: usize, // the largest context whose top-k selection keeps every visible key (Config.keyWidth)
};

/// Latent and pooled keys share a format; individual index keys and gates stay BF16.
pub const Caches = struct { lc: cache.Cache, index: cache.Index, pk: cache.Cache, pk_rows: usize, owner: ?u32 = null };

/// The block's scratch for up to `rows` rows.
pub const Scratch = struct {
    dp: u64, // bf16 [rows][q_lora + lw]
    qr: u64, // bf16 [rows][q_lora]
    xs_qr: u64, // fp32 [rows][q_lora / 64]
    lat: u64, // bf16 [rows][lw]
    xs_lat: u64, // fp32 [rows][lw / 64]
    q: u64, // bf16 [rows][heads qk]
    ikr: u64, // bf16 [rows][160]
    igr: u64, // fp32 [rows][128]
    router_part: u64, // fp32 [router_ks][rows][128]
    qa: u64, // bf16 [rows][heads][lw]
    ol: u64, // bf16 [rows][heads][lw]
    vn: u64, // bf16 [rows][heads dv]
    qi: u64, // bf16 [rows][index heads d]
    partials: latent.Partials,
    dummy: u64, // an i32
    select: sparse.Scratch,
    tokens: u64, // i32 [rows][sparse.width]
    counts: u64, // i32 [rows]
    mm_part: u64, // fp32 split-K partials
    owner_q: u64 = 0,
    owner_out: u64 = 0,
    at: u64, // an i32: a prompt block's first position (pos + its first row)
};

/// The kernels: the captured Triton set and glm_ops (the token lists and position offsets).
pub const Kernels = struct {
    set: glue.Set,
    ops: cuda.Module,
    owner: ?@import("cuda_kv_owner.zig").Transport = null,
    slot: usize = 0,
};

/// forward.dsa_block (latent path) of `rows` rows at pos to an fp32 partial; `nch`: 512-key chunks, `part_rows`: block.
pub fn block(k: Kernels, stream: cuda.Stream, l: Layer, c: Caches, s: Scratch, x: u64, x_stride: usize, rows: usize, pos: u64, host_pos: usize, nch: usize, part_rows: usize, out: u64) !void {
    try body(k, stream, l, c, s, x, x_stride, rows, pos, host_pos, nch, part_rows);
    try fill(k.set, stream, l, s, 0, rows, out);
}

/// forward.out_proj's fill for the DSA block: o_proj of rows lo..hi (s.vn) into out [hi - lo][dims] fp32.
pub fn fill(set: glue.Set, stream: cuda.Stream, l: Layer, s: Scratch, lo: usize, hi: usize, out: u64) !void {
    const w = l.heads * l.dv;
    try b16.proj(set, stream, s.vn + lo * w * 2, w, l.o, out, s.mm_part, hi - lo, l.dims, w, true);
}

/// The block up to its out_proj (s.vn holds the expanded rows).
pub fn body(k: Kernels, stream: cuda.Stream, l: Layer, c: Caches, s: Scratch, x: u64, x_stride: usize, rows: usize, pos: u64, host_pos: usize, nch: usize, part_rows: usize) !void {
    try prepare(k.set, stream, l, s, x, x_stride, rows, sparseRows(l, host_pos, rows));
    try attend(k, stream, l, c, s, 0, rows, pos, host_pos, nch, part_rows);
    try latent.expandV(k.set, stream, s.ol, l.wv, s.vn, rows, l.heads, l.dv, l.lw);
}

/// Whether a sequence at `host_pos` has rows past the dense limit among its next `rows` (the sparse pass).
pub fn sparseRows(l: Layer, host_pos: usize, rows: usize) bool {
    return host_pos + rows - 1 >= l.dense_limit;
}

const replica_slots = 16;
const replica_floor_rows: usize = 262144;

const Replica = struct { buf: ?cuda.DeviceBuffer = null };

var replicas: [replica_slots]Replica = @splat(.{});

fn localAttn(rows: usize) bool {
    if (rows < latent.prompt_rows) return false;
    const v = std.c.getenv("TF_GLM_PREFILL_ATTN_LOCAL") orelse return false;
    return std.mem.eql(u8, std.mem.span(v), "1");
}

fn ensureReplica(d: *const cuda.Driver, stream: cuda.Stream, slot: usize, bytes: usize) !u64 {
    if (slot >= replica_slots) return error.BadPrefillSlot;
    var slot_buf = &replicas[slot];
    if (slot_buf.buf) |*old| {
        if (old.len >= bytes) return old.ptr;
        var neu = try cuda.DeviceBuffer.alloc(d, bytes);
        errdefer neu.free();
        try d.check(d.api.cuMemcpyDtoDAsync_v2(neu.ptr, old.ptr, old.len, stream.handle), "cuMemcpyDtoDAsync");
        old.free();
        slot_buf.buf = neu;
        return neu.ptr;
    }
    slot_buf.buf = try cuda.DeviceBuffer.alloc(d, bytes);
    return slot_buf.buf.?.ptr;
}

fn attendLocal(k: Kernels, stream: cuda.Stream, l: Layer, c: Caches, s: Scratch, owner: u32, pos: u64, host_pos: usize, rows: usize, nch: usize, part_rows: usize, scale: f32, sparse_any: bool) !void {
    const transport = k.owner orelse return error.NoKvOwnerTransport;
    const stride = cache.Format.rowBytes(c.lc.format, l.lw);
    const need = @max((host_pos + rows) * stride, replica_floor_rows * stride);
    const off = host_pos * stride;
    const n = rows * stride;
    const base = if (transport.rank == owner) c.lc.ptr else try ensureReplica(transport.d, stream, k.slot, need);
    try transport.carry(stream, owner, 1 - owner, base + off, base + off, n);
    var caches = c;
    caches.lc.ptr = base;
    if (host_pos < l.dense_limit) try denseAttention(k, stream, l, caches, s, pos, rows, nch, part_rows, scale, latent.headBlock(rows));
    if (sparse_any) try latent.sparseAttention(k.set, stream, s.qa, caches.lc, s.tokens, s.counts, sparse.width, s.partials, s.ol, rows, l.heads, l.lw, scale);
}

/// The block's row-by-row part for a shared window: projections, norms, q_b, indexer, absorbed query, `qi`'s query.
pub fn prepare(set: glue.Set, stream: cuda.Stream, l: Layer, s: Scratch, x: u64, x_stride: usize, rows: usize, qi: bool) !void {
    try prepareInput(set, stream, l, s, x, x_stride, rows);
    try b16.proj(set, stream, s.qr, l.q_lora, l.q_b, s.q, s.mm_part, rows, l.heads * l.qk, l.q_lora, false);
    const ix = l.index;
    try b16.proj(set, stream, x, x_stride, ix.kw, s.ikr, s.mm_part, rows, ix.d + ix.heads, l.dims, false);
    try glue.router(set, stream, x, x_stride, ix.gate, s.igr, s.router_part, rows, l.dims, ix.d);
    try latent.absorbQ(set, stream, s.q, l.wk, s.qa, rows, l.heads, l.qk, l.lw);
    if (qi) try b16.proj(set, stream, s.qr, l.q_lora, ix.qb, s.qi, s.mm_part, rows, ix.heads * ix.d, l.q_lora, false);
}

/// Keep a prompt's projection dispatch while storing its persistent rows in the packed frame.
pub fn prepareSegment(set: glue.Set, stream: cuda.Stream, l: Layer, s: Scratch, x: u64, x_stride: usize, row0: usize, rows: usize, qi: bool) !void {
    var v = s;
    v.qr += row0 * l.q_lora * 2;
    v.xs_qr += row0 * (l.q_lora / 64) * 4;
    v.lat += row0 * l.lw * 2;
    v.xs_lat += row0 * (l.lw / 64) * 4;
    v.q += row0 * l.heads * l.qk * 2;
    v.ikr += row0 * (l.index.d + l.index.heads) * 2;
    v.igr += row0 * l.index.d * 4;
    v.qa += row0 * l.heads * l.lw * 2;
    v.qi += row0 * l.index.heads * l.index.d * 2;
    try prepare(set, stream, l, v, x, x_stride, rows, qi);
}

fn prepareInput(set: glue.Set, stream: cuda.Stream, l: Layer, s: Scratch, x: u64, x_stride: usize, rows: usize) !void {
    const width = l.q_lora + l.lw;
    switch (l.input) {
        .separate => |pair| {
            const kv = s.dp + rows * l.q_lora * 2;
            try b16.proj(set, stream, x, x_stride, pair.qa, s.dp, s.mm_part, rows, l.q_lora, l.dims, false);
            try b16.proj(set, stream, x, x_stride, pair.kva, kv, s.mm_part, rows, l.lw, l.dims, false);
            try glue.rmsnorm(set, stream, s.dp, l.q_lora, l.q_norm, s.qr, l.q_lora, s.xs_qr, l.eps, rows, l.q_lora);
            try glue.rmsnorm(set, stream, kv, l.lw, l.kv_norm, s.lat, l.lw, s.xs_lat, l.eps, rows, l.lw);
        },
        .combined_bf16 => |proj| {
            try b16.matmul(set, stream, x, x_stride, proj, s.dp, s.mm_part, rows, width, l.dims, false);
            try glue.rmsnorm(set, stream, s.dp, width, l.q_norm, s.qr, l.q_lora, s.xs_qr, l.eps, rows, l.q_lora);
            try glue.rmsnorm(set, stream, s.dp + l.q_lora * 2, width, l.kv_norm, s.lat, l.lw, s.xs_lat, l.eps, rows, l.lw);
        },
        .combined => |proj| {
            try b16.proj(set, stream, x, x_stride, proj, s.dp, s.mm_part, rows, width, l.dims, false);
            try glue.rmsnorm(set, stream, s.dp, width, l.q_norm, s.qr, l.q_lora, s.xs_qr, l.eps, rows, l.q_lora);
            try glue.rmsnorm(set, stream, s.dp + l.q_lora * 2, width, l.kv_norm, s.lat, l.lw, s.xs_lat, l.eps, rows, l.lw);
        },
    }
}

test "combined Q4 DSA input projects once and normalizes strided QA and KV rows" {
    const Trace = struct {
        const Norm = struct { x: u64, stride: i64, out: u64, out_stride: i64, rows: u32 };
        var norms: [2]Norm = undefined;
        var count: usize = 0;
        var projections: usize = 0;
        fn value(comptime T: type, args: [*]?*anyopaque, i: usize) T {
            return @as(*align(1) const T, @ptrCast(args[i].?)).*;
        }
        fn launch(f: cuda.abi.Function, gx: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: cuda.abi.Stream, raw: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
            const id = @intFromPtr(f.?);
            const args = raw.?;
            if (id == 3 or id == 4) {
                norms[count] = .{ .x = value(u64, args, 0), .stride = value(i64, args, 1), .out = value(u64, args, 3), .out_stride = value(i64, args, 4), .rows = gx };
                count += 1;
            } else if (id == 2 or id == 5) projections += 1;
            return 0;
        }
    };
    var driver: cuda.Driver = .{ .lib = undefined, .api = undefined };
    driver.api.cuLaunchKernel = Trace.launch;
    const fun = struct {
        fn of(d: *const cuda.Driver, id: usize) cuda.Function {
            return .{ .d = d, .handle = @ptrFromInt(id) };
        }
    }.of;
    var kernels: glue.Kernels = undefined;
    kernels.group_sums = fun(&driver, 1);
    kernels.rmsnorm_1536 = fun(&driver, 3);
    kernels.rmsnorm_512 = fun(&driver, 4);
    const affine: cuda.qlinear.Affine4Kernels = .{ .group_fn = undefined, .gemv_fn = fun(&driver, 2), .split_fn = undefined, .prefill_fn = fun(&driver, 5), .words_fn = undefined, .scales_fn = undefined, .gemv_blocks = 64, .split = null, .pdl = false };
    const set: glue.Set = .{ .glue = &kernels, .linear = .{ .affine4 = &affine } };
    const stream: cuda.Stream = .{ .d = &driver, .handle = @ptrFromInt(17) };
    var layer: Layer = undefined;
    layer.input = .{ .combined = .{ .q = .{ .affine4 = .{ .w = 100, .s = 200, .b = 300, .n = 2048, .k = 4096, .npad = 2048 } } } };
    layer.q_lora = 1536;
    layer.lw = 512;
    layer.dims = 4096;
    layer.q_norm = 400;
    layer.kv_norm = 500;
    layer.eps = 1e-6;
    var scratch: Scratch = undefined;
    scratch.dp = 8192;
    scratch.qr = 16384;
    scratch.lat = 32768;
    scratch.mm_part = 65536;
    scratch.xs_qr = 131072;
    scratch.xs_lat = 262144;
    for ([_]usize{ 1, 8, 16, 128, 2048 }) |rows| {
        Trace.count = 0;
        Trace.projections = 0;
        try prepareInput(set, stream, layer, scratch, 4096, 4096, rows);
        try std.testing.expectEqual(@as(usize, 1), Trace.projections);
        try std.testing.expectEqual(@as(usize, 2), Trace.count);
        try std.testing.expectEqual(Trace.Norm{ .x = scratch.dp, .stride = 2048, .out = scratch.qr, .out_stride = 1536, .rows = @intCast(rows) }, Trace.norms[0]);
        try std.testing.expectEqual(Trace.Norm{ .x = scratch.dp + 1536 * 2, .stride = 2048, .out = scratch.lat, .out_stride = 512, .rows = @intCast(rows) }, Trace.norms[1]);
    }
}

/// One sequence's rows r0..r0 + rows at pos: latent and index keys written, dense then sparse attention into s.ol.
pub fn attend(k: Kernels, stream: cuda.Stream, l: Layer, c: Caches, s: Scratch, r0: usize, rows: usize, pos: u64, host_pos: usize, nch: usize, part_rows: usize) !void {
    const set = k.set;
    const ix = l.index;
    const kw_rows = ix.d + ix.heads;
    var v = s; // the scratch from row r0
    v.lat += r0 * l.lw * 2;
    v.ikr += r0 * kw_rows * 2;
    v.igr += r0 * ix.d * 4;
    v.qa += r0 * l.heads * l.lw * 2;
    v.ol += r0 * l.heads * l.lw * 2;
    v.qi += r0 * ix.heads * ix.d * 2;
    if (c.owner == null or (k.owner != null and k.owner.?.rank == c.owner.?)) try latent.latentWrite(set, stream, v.lat, l.lw, c.lc, pos, rows, l.lw);
    try sparse.indexUpdate(set, stream, v.ikr, kw_rows, v.igr, ix.ln_w, ix.ln_b, ix.ape, c.index, c.pk, pos, rows);
    const scale: f32 = @floatCast(std.math.pow(f64, @floatFromInt(l.qk), -0.5));
    const sparse_any = sparseRows(l, host_pos, rows);
    if (sparse_any) try sparse.selectTokens(set, k.ops, stream, v.qi, v.ikr + ix.d * 2, kw_rows, c.pk, pos, host_pos, rows, ix.heads, ix.d, c.pk_rows - 2, v.select, v.tokens, v.counts);
    if (c.owner) |owner| {
        if (localAttn(rows)) {
            try attendLocal(k, stream, l, c, v, owner, pos, host_pos, rows, nch, part_rows, scale, sparse_any);
        } else {
            const transport = k.owner orelse return error.NoKvOwnerTransport;
            var first: usize = 0;
            while (first < rows) {
                const n = @min(part_rows, rows - first);
                var tile = v;
                tile.qa += first * l.heads * l.lw * 2;
                tile.ol += first * l.heads * l.lw * 2;
                tile.tokens += first * sparse.width * 4;
                tile.counts += first * 4;
                var at = pos;
                if (first > 0) {
                    var args: cuda.Args = .{};
                    inline for (.{ pos, @as(c_int, @intCast(first)), v.at }) |value| args.add(value);
                    try cuda.launch.launch(try k.ops.function("tf_glm_add_i32"), .{ .grid = .{}, .block = .{ .x = 32 } }, stream, &args);
                    at = v.at;
                }
                var work: Attention = .{ .k = k, .stream = stream, .l = l, .c = c, .s = tile, .pos = at, .host_pos = host_pos + first, .rows = n, .nch = nch, .scale = scale, .hb = latent.headBlock(rows) };
                try transport.attention(stream, owner, tile.qa, tile.ol, v.owner_q, v.owner_out, n * l.heads * l.lw * 2, .{ .ctx = &work, .run = Attention.run });
                first += n;
            }
        }
    } else {
        if (host_pos < l.dense_limit) try denseAttention(k, stream, l, c, v, pos, rows, nch, part_rows, scale, latent.headBlock(rows));
        if (sparse_any) try latent.sparseAttention(set, stream, v.qa, c.lc, v.tokens, v.counts, sparse.width, v.partials, v.ol, rows, l.heads, l.lw, scale);
    }
}

/// forward.dense_attention: latent.attention in `part_rows`-row blocks, each at its first row's position.
fn denseAttention(k: Kernels, stream: cuda.Stream, l: Layer, c: Caches, s: Scratch, pos: u64, rows: usize, nch: usize, part_rows: usize, scale: f32, hb: usize) !void {
    var r0: usize = 0;
    while (r0 < rows) : (r0 += part_rows) {
        const n = @min(part_rows, rows - r0);
        var at = pos;
        if (r0 > 0) {
            var a: cuda.Args = .{};
            a.add(pos);
            a.add(@as(c_int, @intCast(r0)));
            a.add(s.at);
            try cuda.launch.launch(try k.ops.function("tf_glm_add_i32"), .{ .grid = .{ .x = 1 }, .block = .{ .x = 32 } }, stream, &a);
            at = s.at;
        }
        const off = r0 * l.heads * l.lw * 2;
        try latent.attention(k.set, stream, s.qa + off, c.lc, at, s.partials, s.ol + off, s.dummy, n, l.heads, l.lw, nch, scale, hb);
    }
}

const Attention = struct {
    k: Kernels,
    stream: cuda.Stream,
    l: Layer,
    c: Caches,
    s: Scratch,
    pos: u64,
    host_pos: usize,
    rows: usize,
    nch: usize,
    scale: f32,
    hb: usize,
    fn run(ptr: *anyopaque, qa: u64, out: u64) !void {
        const a: *@This() = @ptrCast(@alignCast(ptr));
        var s = a.s;
        s.qa = qa;
        s.ol = out;
        if (a.host_pos < a.l.dense_limit) try denseAttention(a.k, a.stream, a.l, a.c, s, a.pos, a.rows, a.nch, a.rows, a.scale, a.hb);
        if (sparseRows(a.l, a.host_pos, a.rows)) try latent.sparseAttentionBlock(a.k.set, a.stream, qa, a.c.lc, s.tokens, s.counts, sparse.width, s.partials, out, a.rows, a.l.heads, a.l.lw, a.scale, a.hb);
    }
};
