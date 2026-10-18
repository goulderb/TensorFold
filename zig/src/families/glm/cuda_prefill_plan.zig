//! Planning numbers for GLM prompt prefill: piece cuts, split SendRecv, owner vs local MLA.
const std = @import("std");
const cfg = @import("config.zig");
const cache = @import("cuda_cache.zig");
const latent = @import("cuda_latent.zig");
const sparse = @import("cuda_sparse.zig");
const fw = @import("cuda_forward.zig");
const dsa = @import("cuda_dsa_block.zig");

pub const backbone_layers: usize = 45;
pub const mla_layers: usize = 11;
pub const kda_layers: usize = 34;
pub const moe_layers: usize = 42;
pub const overlap_pieces: usize = 3;
pub const sites: usize = 2;
pub const exchanges: usize = 2;
pub const world: usize = 2;
pub const pass_rows: usize = 2048;
pub const att_rows: usize = dsa.prompt_att_rows;
pub const profile8k_tokens: usize = 8192;

pub fn passesFor(tokens: usize) usize {
    return (tokens + pass_rows - 1) / pass_rows;
}

pub fn splitSendRecvPerPass(layers: usize, pieces: usize) usize {
    return layers * sites * pieces * exchanges;
}

pub fn splitSendRecv(tokens: usize, layers: usize, pieces: usize) usize {
    return splitSendRecvPerPass(layers, pieces) * passesFor(tokens);
}

pub fn rankShare(n: usize, ranks: usize, rank: usize) [2]usize {
    const base = n / ranks;
    const extra = n % ranks;
    const lo = rank * base + @min(rank, extra);
    return .{ lo, lo + base + @intFromBool(rank < extra) };
}

pub fn fp32PartialsBytes(rows: usize, hidden: usize) usize {
    return rows * hidden * 4;
}

pub fn bf16ShareBytes(rows: usize, hidden: usize) usize {
    return rows * hidden * 2;
}

pub fn qaTileBytes(rows: usize, heads: usize, lw: usize) usize {
    return rows * heads * lw * 2;
}

pub fn latentCarryBytes(rows: usize, format: cache.Format, lw: usize) usize {
    return rows * format.rowBytes(lw);
}

pub fn replicaBytes(capacity: usize, format: cache.Format, lw: usize) usize {
    return capacity * format.rowBytes(lw);
}

pub fn ownerSerial(kv_shard: bool, local: bool) bool {
    return kv_shard and !local;
}

pub fn localAttn(env: ?[]const u8) bool {
    const v = env orelse return false;
    return std.mem.eql(u8, v, "1");
}

pub fn ownerChunkLaunches(layers: usize, passes: usize, tiles: usize, ranks: usize) usize {
    return layers * passes * tiles * ranks;
}

pub fn localChunkLaunches(layers: usize, passes: usize, tiles: usize) usize {
    return layers * passes * tiles;
}

pub fn profile8kOwnerChunks() usize {
    const layers = mla_layers;
    const ranks = world;
    const tiles = pass_rows / att_rows;
    const pass0_dense = layers * tiles * ranks;
    const pass1_dense = layers * 1 * ranks;
    const sparse_tiles = layers * tiles * ranks * 3;
    return pass0_dense + pass1_dense + sparse_tiles;
}

pub fn denseNch(pos: usize, rows: usize) usize {
    return latent.chunksFor(pos + rows);
}

pub fn sparseNch() usize {
    return latent.chunksFor(sparse.width);
}

test "split SendRecv count matches the 8192-token nsys table" {
    try std.testing.expectEqual(@as(usize, 4), passesFor(profile8k_tokens));
    try std.testing.expectEqual(@as(usize, 540), splitSendRecvPerPass(backbone_layers, overlap_pieces));
    try std.testing.expectEqual(@as(usize, 2160), splitSendRecv(profile8k_tokens, backbone_layers, overlap_pieces));
}

test "piece cuts of a 2048-row pass on the 128-row step" {
    var cut: [fw.max_pieces][2]usize = undefined;
    const n = fw.ranges(pass_rows, overlap_pieces, &cut);
    try std.testing.expectEqual(@as(usize, 3), n);
    try std.testing.expectEqual(@as(usize, 0), cut[0][0]);
    try std.testing.expectEqual(@as(usize, 768), cut[0][1]);
    try std.testing.expectEqual(@as(usize, 768), cut[1][0]);
    try std.testing.expectEqual(@as(usize, 1536), cut[1][1]);
    try std.testing.expectEqual(@as(usize, 1536), cut[2][0]);
    try std.testing.expectEqual(@as(usize, 2048), cut[2][1]);
}

test "TP=2 owned rows of each piece" {
    try std.testing.expectEqual(@as(usize, 384), rankShare(768, 2, 0)[1] - rankShare(768, 2, 0)[0]);
    try std.testing.expectEqual(@as(usize, 384), rankShare(768, 2, 1)[1] - rankShare(768, 2, 1)[0]);
    try std.testing.expectEqual(@as(usize, 256), rankShare(512, 2, 0)[1] - rankShare(512, 2, 0)[0]);
    try std.testing.expectEqual(@as(usize, 256), rankShare(512, 2, 1)[1] - rankShare(512, 2, 1)[0]);
    try std.testing.expectEqual(@as(usize, 12582912), fp32PartialsBytes(768, 4096));
    try std.testing.expectEqual(@as(usize, 6291456), bf16ShareBytes(768, 4096));
}

test "8K owner chunks_32_fp8 launches match nsys 374" {
    try std.testing.expectEqual(@as(u32, 34), (cfg.Config{}).countKind(.kda));
    try std.testing.expectEqual(@as(u32, 11), (cfg.Config{}).countKind(.mla));
    try std.testing.expectEqual(@as(usize, 512), att_rows);
    try std.testing.expectEqual(@as(usize, 4), denseNch(0, pass_rows));
    try std.testing.expectEqual(@as(usize, 8), denseNch(2048, pass_rows));
    try std.testing.expectEqual(@as(usize, 5), sparseNch());
    try std.testing.expectEqual(@as(usize, 2051), sparse.width);
    try std.testing.expectEqual(@as(usize, 374), profile8kOwnerChunks());
}

test "local MLA halves owner-serialized chunk launches and copies keys not Q" {
    try std.testing.expect(ownerSerial(true, false));
    try std.testing.expect(!ownerSerial(true, true));
    try std.testing.expect(!ownerSerial(false, false));
    try std.testing.expect(!localAttn(null));
    try std.testing.expect(!localAttn("0"));
    try std.testing.expect(localAttn("1"));
    try std.testing.expectEqual(@as(usize, 352), ownerChunkLaunches(mla_layers, 4, 4, world));
    try std.testing.expectEqual(@as(usize, 176), localChunkLaunches(mla_layers, 4, 4));
    try std.testing.expectEqual(@as(usize, 16777216), qaTileBytes(att_rows, 32, 512));
    try std.testing.expectEqual(@as(usize, 1081344), latentCarryBytes(pass_rows, .fp8, 512));
    try std.testing.expectEqual(@as(usize, 138412032), replicaBytes(262144, .fp8, 512));
    try std.testing.expectEqual(@as(usize, 528), cache.Format.fp8.rowBytes(512));
}

test "250K pass count and SendRecv scale linearly" {
    try std.testing.expectEqual(@as(usize, 123), passesFor(250030));
    try std.testing.expectEqual(@as(usize, 66420), splitSendRecv(250030, backbone_layers, overlap_pieces));
    try std.testing.expectEqual(@as(usize, 10736), ownerChunkLaunches(mla_layers, 122, 4, world));
    try std.testing.expectEqual(@as(usize, 5368), localChunkLaunches(mla_layers, 122, 4));
}
