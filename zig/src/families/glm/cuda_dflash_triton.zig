const std = @import("std");
const cuda = @import("cuda");
const aot = cuda.aot;
pub const Tri = struct {
    set: *const aot.Set,
    s: cuda.Stream,
    fn run(t: Tri, name: []const u8, grid: [3]usize, args: []const aot.Arg, constants: []const aot.Const) !void {
        try t.set.run(t.s, name, .{ @intCast(grid[0]), @intCast(grid[1]), @intCast(grid[2]) }, args, constants);
    }
    pub fn embed(t: Tri, ids: u64, table: u64, out: u64, rows: usize, copies: usize) !void {
        try t.run("_embed_b16", .{ rows, 64, 1 }, &.{ aot.ptr("IDS", "*i32", ids), aot.ptr("W", "*bf16", table), aot.ptr("OUT", "*bf16", out) }, &.{ aot.ci("D", 4096), aot.ci("COPIES", @intCast(copies)) });
    }
    pub fn draftRmsnorm(t: Tri, x: u64, stride: usize, weight: u64, out: u64, xs: u64, rows: usize, d: usize, eps: f32) !void {
        try t.run("_rmsnorm", .{ rows, 1, 1 }, &.{ aot.ptr("X", "*bf16", x), aot.int("x_stride", @intCast(stride)), aot.ptr("W", "*bf16", weight), aot.ptr("OUT", "*bf16", out), aot.int("o_stride", @intCast(d)), aot.ptr("XS", "*fp32", xs), aot.float("eps", eps) }, &.{ aot.ci("D", @intCast(d)), aot.ci("BLOCK", @intCast(pow2(d))), aot.ci("SUMS", 1) });
    }
    pub fn groupSums(t: Tri, x: u64, xs: u64, rows: usize, k: usize) !void {
        try t.run("_group_sums", .{ rows, (k / 64 + 15) / 16, 1 }, &.{ aot.ptr("X", "*bf16", x), aot.ptr("XS", "*fp32", xs), aot.int("x_stride", @intCast(k)) }, &.{ aot.ci("K", @intCast(k)), aot.ci("GB", 16) });
    }
    pub fn draftSwiglu(t: Tri, gu: u64, out: u64, xs: u64, rows: usize, width: usize) !void {
        const block = std.math.gcd(@as(usize, 512), width);
        if (block < 64) return error.BadDraftWidth;
        try t.run("_swiglu", .{ rows, width / block, 1 }, &.{ aot.ptr("GU", "*bf16", gu), aot.ptr("OUT", "*bf16", out), aot.ptr("XS", "*fp32", xs), aot.float("LIMIT", 1e30) }, &.{ aot.ci("W", @intCast(width)), aot.ci("BLOCK", @intCast(block)) });
    }
    pub fn draftConv(t: Tri, x: u64, dyn: u64, base: u64, residual: ?u64, out: u64, rows: usize, branch: usize, segment: usize) !void {
        if (branch > 1) return error.BadDraftBranch;
        try t.run("_dconv_kernel", .{ rows, 4, 1 }, &.{ aot.ptr("X", "*bf16", x), aot.ptr("DYN", "*bf16", dyn), aot.ptr("BASE", "*bf16", base), aot.ptr("RES", "*bf16", residual orelse x), aot.ptr("OUT", "*bf16", out) }, &.{ aot.ci("D", 4096), aot.ci("G", 256), aot.ci("GS", 16), aot.ci("BRANCH", @intCast(branch)), aot.ci("HAS_RES", @intFromBool(residual != null)), aot.ci("BLOCK", 1024), aot.ci("SEG", @intCast(segment)) });
    }
    pub fn draftPrep(t: Tri, qkv: u64, qn: u64, kn: u64, cos: u64, sin: u64, qo: u64, ko: u64, vo: u64, rows: usize, heads: usize, kv: usize, eps: f32) !void {
        try t.run("_prep_kernel", .{ rows, heads + 2 * kv, 1 }, &.{ aot.ptr("QKV", "*bf16", qkv), aot.ptr("QN", "*bf16", qn), aot.ptr("KN", "*bf16", kn), aot.ptr("COS", "*fp32", cos), aot.ptr("SIN", "*fp32", sin), aot.ptr("QO", "*bf16", qo), aot.ptr("KO", "*bf16", ko), aot.ptr("VO", "*bf16", vo), aot.int("L", @intCast(rows)), aot.int("stride", @intCast((heads + 2 * kv) * 128)), aot.float("eps", eps) }, &.{ aot.ci("H", @intCast(heads)), aot.ci("HKV", @intCast(kv)), aot.ci("HALF", 64) });
    }
    pub fn draftAttention(t: Tri, q: u64, k: u64, v: u64, out: u64, pos: u64, slots: u64, streams: usize, heads: usize, kv: usize, slot_rows: usize) !void {
        if (kv == 0 or heads % kv != 0) return error.BadDraftHeads;
        try t.run("_dattn_seg_kernel", .{ kv, streams, 1 }, &.{ aot.ptr("Q", "*bf16", q), aot.ptr("K", "*bf16", k), aot.ptr("V", "*bf16", v), aot.ptr("OUT", "*bf16", out), aot.ptr("POS", "*i64", pos), aot.ptr("SLOT", "*i64", slots), aot.int("window", 2047), aot.float("scale", 0.08838834764831845), aot.int("L", @intCast(streams * 8)), aot.int("SLOT_ROWS", @intCast(slot_rows)) }, &.{ aot.ci("N", 8), aot.ci("G", @intCast(heads / kv)), aot.ci("NH", @intCast(heads)), aot.ci("HD", 128), aot.ci("CAP", 2176), aot.ci("BK", 64), aot.ci("CAUSAL", 0), aot.ci("RING", 1) });
    }
};
pub fn pow2(n: usize) usize {
    return std.math.ceilPowerOfTwo(usize, @max(n, 1)) catch unreachable;
}
