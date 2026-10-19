//! DSA decode kernels whose sequence geometry is read from the target graph's stable device row table.
const std = @import("std");
const cuda = @import("cuda");
const glue = @import("cuda_glue.zig");
const dsa = @import("cuda_dsa_block.zig");
const cache = @import("cuda_cache.zig");
const sparse = @import("cuda_sparse.zig");

pub const chunks = 5;

/// Fixed stride for every captured score row; visible pools are masked on the device.
pub fn scoreColumns(capacity: usize) usize {
    return capacity / sparse.pool + @intFromBool(capacity % sparse.pool != 0);
}

pub const Kernels = struct {
    module: cuda.Module,
    lat_bf16: cuda.Function,
    lat_fp8: cuda.Function,
    index_write: cuda.Function,
    pool_bf16: cuda.Function,
    pool_fp8: cuda.Function,
    scores_bf16: cuda.Function,
    scores_fp8: cuda.Function,
    select_rows: cuda.Function,
    tokens: cuda.Function,
    chunks_bf16: cuda.Function,
    chunks_fp8: cuda.Function,
    merge: cuda.Function,
    pack_peer: cuda.Function,
    owner_select: cuda.Function,

    pub fn init(d: *const cuda.Driver) !Kernels {
        var m = try cuda.Module.load(d, cuda.kernels.glm_target_dsa);
        errdefer m.unload();
        var k: Kernels = undefined;
        k.module = m;
        inline for (.{ "lat_bf16", "lat_fp8", "index_write", "pool_bf16", "pool_fp8", "scores_bf16", "scores_fp8", "select_rows", "tokens", "chunks_bf16", "chunks_fp8", "merge", "pack_peer", "owner_select" }) |name| @field(k, name) = try m.function("tf_glm_target_dsa_" ++ name);
        try k.chunks_bf16.allowDynamicShared(glue.chunksSmem(16));
        try k.chunks_fp8.allowDynamicShared(glue.chunksSmem(16));
        return k;
    }

    pub fn deinit(k: *Kernels) void {
        k.module.unload();
        k.* = undefined;
    }
};

fn shape(l: dsa.Layer, rows: usize, capacity: usize) !void {
    if (rows == 0 or rows > 32 or capacity == 0 or capacity > std.math.maxInt(c_int)) return error.BadTargetDsaWindow;
    if (l.lw != 512 or l.qk != 256 or l.dv != 256 or l.heads == 0 or l.index.d != 128 or l.index.heads == 0 or l.index.heads > 32) return error.UnsupportedTargetDsaShape;
    if (l.dense_limit < sparse.width or l.dense_limit > chunks * 512) return error.UnsupportedTargetDsaDenseLimit;
}

/// Row projections and norms are unchanged; query-index work is always present in the captured topology.
pub fn prepare(set: glue.Set, stream: cuda.Stream, l: dsa.Layer, s: dsa.Scratch, x: u64, hidden: usize, rows: usize) !void {
    try dsa.prepare(set, stream, l, s, x, hidden, rows, true);
}

/// All rows update replicated index storage; only resident rows write latent storage.
/// s.select.scores needs rows * scoreColumns(capacity) floats, independent of current sequence positions.
pub fn update(k: *const Kernels, set: glue.Set, stream: cuda.Stream, l: dsa.Layer, s: dsa.Scratch, rows: usize, row_table: u64, format: cache.Format, capacity: usize) !void {
    _ = set;
    try shape(l, rows, capacity);
    const ix = l.index;
    const stride = ix.d + ix.heads;
    var lat: cuda.Args = .{};
    inline for (.{ s.lat, @as(i64, @intCast(l.lw)), row_table }) |v| lat.add(v);
    try cuda.launch.launch(if (format == .fp8) k.lat_fp8 else k.lat_bf16, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 128 } }, stream, &lat);
    var index: cuda.Args = .{};
    inline for (.{ s.ikr, @as(i64, @intCast(stride)), s.igr, ix.ln_w, ix.ln_b, row_table }) |v| index.add(v);
    try cuda.launch.launch(k.index_write, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 32 } }, stream, &index);
    var pool: cuda.Args = .{};
    inline for (.{ ix.ape, row_table }) |v| pool.add(v);
    try cuda.launch.launch(if (format == .fp8) k.pool_fp8 else k.pool_bf16, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 32 } }, stream, &pool);
    const np = scoreColumns(capacity);
    const nb = (np + 63) / 64;
    var score: cuda.Args = .{};
    inline for (.{ s.qi, s.ikr + ix.d * 2, @as(i64, @intCast(stride)), s.select.scores, row_table, @as(i64, @intCast(np)), @as(i64, @intCast(nb)), @as(f32, @floatCast(std.math.pow(f64, @floatFromInt(ix.d), -0.5))), sparse.headScale(ix.heads), @as(i64, @intCast(ix.heads)) }) |v| score.add(v);
    try cuda.launch.launch(if (format == .fp8) k.scores_fp8 else k.scores_bf16, .{ .grid = .{ .x = @intCast(rows), .y = @intCast(nb) }, .block = .{ .x = 128 } }, stream, &score);
    var select: cuda.Args = .{};
    inline for (.{ s.select.scores, s.select.pools, @as(i64, @intCast(np)), row_table }) |v| select.add(v);
    try cuda.launch.launch(k.select_rows, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 128 } }, stream, &select);
    var tokens: cuda.Args = .{};
    inline for (.{ s.select.pools, row_table, s.tokens, s.counts }) |v| tokens.add(v);
    try cuda.launch.launch(k.tokens, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 256 } }, stream, &tokens);
}

/// One fixed five-chunk pass handles a mixture of dense, sparse and nonresident rows.
/// The row table's pad field holds the eager segment's merge count (1..5), refreshed before replay.
/// Partials are [5][rows][heads][512] floats, with [5][rows][heads] maxima and normalizers.
pub fn attention(k: *const Kernels, set: glue.Set, stream: cuda.Stream, l: dsa.Layer, s: dsa.Scratch, rows: usize, row_table: u64, query: u64, out: u64, rank: usize, format: cache.Format, capacity: usize) !void {
    _ = set; // Captured decode uses HB16, as the eager native decode path does.
    try shape(l, rows, capacity);
    if (rank > std.math.maxInt(u32)) return error.BadTargetDsaRank;
    const scale: f32 = @floatCast(std.math.pow(f64, @floatFromInt(l.qk), -0.5));
    var a: cuda.Args = .{};
    inline for (.{ query, s.tokens, s.counts, s.partials.po, s.partials.pm, s.partials.pl, row_table, @as(i64, @intCast(rows)), @as(i64, @intCast(l.heads)), scale, @as(u32, @intCast(rank)) }) |v| a.add(v);
    try cuda.launch.launch(if (format == .fp8) k.chunks_fp8 else k.chunks_bf16, .{ .grid = .{ .x = @intCast(rows), .y = @intCast((l.heads + 15) / 16), .z = chunks }, .block = .{ .x = 256 }, .shared = glue.chunksSmem(16) }, stream, &a);
    var merge: cuda.Args = .{};
    inline for (.{ s.partials.po, s.partials.pm, s.partials.pl, out, row_table, @as(i64, @intCast(rows)), @as(i64, @intCast(l.heads)), @as(u32, @intCast(rank)) }) |v| merge.add(v);
    try cuda.launch.launch(k.merge, .{ .grid = .{ .x = @intCast(rows), .y = @intCast(l.heads) }, .block = .{ .x = 128 } }, stream, &merge);
}

/// Packs only the peer's head slice, so the result AllGather sends one local-head width per rank.
pub fn packPeer(k: *const Kernels, stream: cuda.Stream, local_results: u64, rank: usize, rows: usize, heads: usize, lw: usize, peer: u64) !void {
    if (rank >= 2 or rows == 0 or rows > 32 or heads == 0 or lw != 512) return error.BadTargetDsaOwnerSelection;
    var a: cuda.Args = .{};
    inline for (.{ local_results, @as(u32, @intCast(rank)), @as(i64, @intCast(rows)), @as(i64, @intCast(heads)), @as(i64, @intCast(lw)), peer }) |v| a.add(v);
    try cuda.launch.launch(k.pack_peer, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 256 } }, stream, &a);
}

/// Bit-copies the owner's local head result or the owner's packed peer result; no floating reduction occurs.
pub fn ownerSelect(k: *const Kernels, stream: cuda.Stream, row_table: u64, own_result: u64, gathered_peer: u64, rank: usize, rows: usize, heads: usize, lw: usize, out: u64) !void {
    if (rank >= 2 or rows == 0 or rows > 32 or heads == 0 or lw != 512) return error.BadTargetDsaOwnerSelection;
    var a: cuda.Args = .{};
    inline for (.{ row_table, own_result, gathered_peer, @as(u32, @intCast(rank)), @as(i64, @intCast(rows)), @as(i64, @intCast(heads)), @as(i64, @intCast(lw)), out }) |v| a.add(v);
    try cuda.launch.launch(k.owner_select, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 256 } }, stream, &a);
}

test "target score stride is immutable capacity geometry, including incomplete last pools" {
    try std.testing.expectEqual(@as(usize, 64), scoreColumns(256));
    try std.testing.expectEqual(@as(usize, 65), scoreColumns(257));
    try std.testing.expectEqual(@as(usize, 65536), scoreColumns(262144));
    try std.testing.expectEqual(@as(usize, 2051), sparse.width);
    try std.testing.expectEqual(@as(usize, 5), chunks);
}
