//! GLM-5.3-Flash's MLA on the 512-wide latent cache (DSA layers): the Python family's latent on BF16, row by row.

const std = @import("std");
const cuda = @import("cuda");
const glue = @import("cuda_glue.zig");
const cache_rows = @import("cuda_cache.zig");

/// latent.CHUNK, KT, HB, HB_WIDE, PROMPT_ROWS, PROMPT_RB.
pub const chunk = 512;
pub const key_tile = 32;
pub const heads_tile = 16;
pub const heads_tile_wide = 32;
pub const prompt_rows = 64;
pub const prompt_rb = 64;

/// latent.head_block: a rank's 32 heads a program in prompt chunks (each key read once), 16 in decode windows.
pub fn headBlock(rows: usize) usize {
    return if (rows >= 64) heads_tile_wide else heads_tile;
}

/// latent.chunks_for: 512-key chunks to cover `length` keys.
pub fn chunksFor(length: usize) usize {
    return (length + chunk - 1) / chunk;
}

fn cdiv(a: usize, b: usize) u32 {
    return @intCast((a + b - 1) / b);
}

/// latent.absorb_q (BF16 kv_b): out [rows][heads][lw] bf16 = q [rows][heads][d] . wk [heads][d][lw], fp32 sums.
pub fn absorbQ(set: glue.Set, stream: cuda.Stream, q: u64, wk: u64, out: u64, rows: usize, heads: usize, d: usize, lw: usize) !void {
    if (d != 256 or lw != 512) return error.LatentShape;
    try rowBlocks(set.glue.absorb_q, stream, q, wk, out, rows, heads, lw / 32);
}

/// The absorb and expand launches: rows in one block in decode windows, prompt_rb a block in prompt chunks.
fn rowBlocks(f: cuda.Function, stream: cuda.Stream, in: u64, w: u64, out: u64, rows: usize, heads: usize, cols: usize) !void {
    const rb = if (rows >= prompt_rows) prompt_rb else rows;
    var a: cuda.Args = .{};
    inline for (.{ in, w, out, @as(i64, @intCast(rows)), @as(i64, @intCast(heads)), @as(i64, @intCast(rb)) }) |v| a.add(v);
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(heads), .y = @intCast(cols), .z = cdiv(rows, rb) }, .block = .{ .x = 128 } }, stream, &a);
}

/// latent.expand_v (BF16 kv_b): out [rows][heads][dv] bf16 = o_lat [rows][heads][lw] . wv [heads][dv][lw].
pub fn expandV(set: glue.Set, stream: cuda.Stream, ol: u64, wv: u64, out: u64, rows: usize, heads: usize, dv: usize, lw: usize) !void {
    if (dv != 256 or lw != 512) return error.LatentShape;
    try rowBlocks(set.glue.expand_v, stream, ol, wv, out, rows, heads, dv / 16);
}

/// latent.latent_write: rows of `lat` (stride `lat_stride`) into slots pos .. pos + rows - 1 (pos an i32 on device).
pub fn latentWrite(set: glue.Set, stream: cuda.Stream, lat: u64, lat_stride: usize, cache: cache_rows.Cache, pos: u64, rows: usize, lw: usize) !void {
    if (lw != 512) return error.LatentShape;
    var a: cuda.Args = .{};
    inline for (.{ lat, @as(i64, @intCast(lat_stride)), cache.ptr, pos, @as(i64, @intCast(lw)) }) |v| a.add(v);
    const f = if (cache.format == .fp8) set.glue.lat_write_fp8 else set.glue.lat_write;
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 128 } }, stream, &a);
}

/// The chunk partials of one attention call: po fp32 [nch][rows][heads][lw], pm and pl fp32 [nch][rows][heads].
pub const Partials = struct { po: u64, pm: u64, pl: u64 };

/// latent.attention: causal qa over the cache in `nch` 512-key chunks into out bf16; `dummy`: the merge's count i32.
pub fn attention(set: glue.Set, stream: cuda.Stream, qa: u64, cache: cache_rows.Cache, pos: u64, p: Partials, out: u64, dummy: u64, rows: usize, heads: usize, lw: usize, nch: usize, scale: f32, hb: usize) !void {
    try chunks(set, stream, qa, cache, pos, 0, 0, p, rows, heads, lw, nch, 0, scale, hb, false);
    try merge(set, stream, p, out, dummy, rows, heads, lw, nch, false);
}

/// latent._dense_chunks (sparse false: keys through pos + r) and _sparse_chunks (row r's `counts` tokens of `tokens`).
fn chunks(set: glue.Set, stream: cuda.Stream, qa: u64, cache: cache_rows.Cache, pos: u64, tokens: u64, counts: u64, p: Partials, rows: usize, heads: usize, lw: usize, nch: usize, w: usize, scale: f32, hb: usize, sparse: bool) !void {
    if (lw != 512) return error.LatentShape;
    const fp8load = useFP8Load(set, cache.format, hb);
    const f = switch (hb) {
        heads_tile => if (cache.format == .fp8) set.glue.chunks_16_fp8 else set.glue.chunks_16,
        heads_tile_wide => if (fp8load) set.glue.chunks_32_fp8load else if (cache.format == .fp8) set.glue.chunks_32_fp8 else set.glue.chunks_32,
        else => return error.HeadBlock,
    };
    var a: cuda.Args = .{};
    inline for (.{ qa, cache.ptr, pos, tokens, counts, p.po, p.pm, p.pl, @as(i64, @intCast(rows)), @as(i64, @intCast(heads)), @as(i64, @intCast(w)), scale, @as(c_int, @intFromBool(sparse)) }) |v| a.add(v);
    const shared = if (fp8load) glue.chunksFP8LoadSmem() else if (hb == heads_tile_wide) glue.chunksSmem(heads_tile_wide) else glue.chunksSmem(heads_tile);
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(rows), .y = cdiv(heads, hb), .z = @intCast(nch) }, .block = .{ .x = 256 }, .shared = shared }, stream, &a);
}

/// latent.sparse_attention: rows with counts > 0 over their selected tokens (i32 [rows][w]); others left alone.
pub fn sparseAttention(set: glue.Set, stream: cuda.Stream, qa: u64, cache: cache_rows.Cache, tokens: u64, counts: u64, w: usize, p: Partials, out: u64, rows: usize, heads: usize, lw: usize, scale: f32) !void {
    return sparseAttentionBlock(set, stream, qa, cache, tokens, counts, w, p, out, rows, heads, lw, scale, headBlock(rows));
}

pub fn sparseAttentionBlock(set: glue.Set, stream: cuda.Stream, qa: u64, cache: cache_rows.Cache, tokens: u64, counts: u64, w: usize, p: Partials, out: u64, rows: usize, heads: usize, lw: usize, scale: f32, hb: usize) !void {
    const nch = chunksFor(w);
    try chunks(set, stream, qa, cache, 0, tokens, counts, p, rows, heads, lw, nch, w, scale, hb, true);
    try merge(set, stream, p, out, counts, rows, heads, lw, nch, true);
}

fn merge(set: glue.Set, stream: cuda.Stream, p: Partials, out: u64, counts: u64, rows: usize, heads: usize, lw: usize, nch: usize, sparse: bool) !void {
    if (lw != 512) return error.LatentShape;
    var a: cuda.Args = .{};
    inline for (.{ p.po, p.pm, p.pl, out, counts, @as(i64, @intCast(rows)), @as(i64, @intCast(heads)), @as(c_int, @intCast(nch)), @as(c_int, @intFromBool(sparse)) }) |v| a.add(v);
    try cuda.launch.launch(set.glue.merge, .{ .grid = .{ .x = @intCast(rows), .y = @intCast(heads) }, .block = .{ .x = 128 } }, stream, &a);
}

fn useFP8Load(set: glue.Set, format: cache_rows.Format, hb: usize) bool {
    return set.attn_fp8load and format == .fp8 and hb == heads_tile_wide;
}

test "FP8 attention load opt-in is confined to HB32 FP8 with matching shared storage" {
    const off: glue.Set = .{ .glue = undefined };
    const on: glue.Set = .{ .glue = undefined, .attn_fp8load = true };
    try std.testing.expect(!useFP8Load(off, .fp8, 32));
    try std.testing.expect(useFP8Load(on, .fp8, 32));
    try std.testing.expect(!useFP8Load(on, .bf16, 32));
    try std.testing.expect(!useFP8Load(on, .fp8, 16));
    try std.testing.expectEqual(@as(u32, 101120), glue.chunksSmem(32));
    try std.testing.expectEqual(@as(u32, 101376), glue.chunksFP8LoadSmem());
}

test "latent.py's tiling rules" {
    try std.testing.expectEqual(@as(usize, 16), headBlock(8));
    try std.testing.expectEqual(@as(usize, 32), headBlock(64));
    try std.testing.expectEqual(@as(usize, 3), chunksFor(1025));
    try std.testing.expectEqual(@as(usize, 1), chunksFor(512));
}
