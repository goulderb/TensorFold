//! GLM P2/P3: TensorFold's glm5_next/cuda/qmm.py and exl3_mm.py launch policies.
//! Math authored by Ash Hart and TensorFold contributors. Zig CUDA patterns authored by jschmied (PR #443).
const std = @import("std");
const cuda = @import("cuda");

pub const Entries = @import("cuda_dflash_entries.zig").Entries;
pub const Q4 = struct { w: u64, s: u64, b: u64, n: usize, k: usize, scale_stride: usize = 0 };
pub fn splitK(n: usize, k: usize) usize {
    if (k == 4096 and n == 12576) return 4;
    if (k == 4096 and n == 4096) return 2;
    if (k == 4096 and n == 2048) return 4;
    if (n == 8192 and (k == 1536 or k == 512)) return 4;
    if (n == 4096 and k == 8192) return 8;
    const tiles = (n + 63) / 64;
    const groups = k / 64;
    var sk: usize = 1;
    while (sk < 8 and tiles * sk < 192 and groups % (sk * 2) == 0 and groups / (sk * 2) >= 8) sk *= 2;
    return sk;
}
pub fn decTile(n: usize, k: usize) ?usize {
    if (n == 4096 and k == 8192) return 4;
    if ((n == 4096 and (k == 4096 or k == 1024)) or (n == 2048 and k == 4096) or (n == 8192 and k == 1536)) return 2;
    return null;
}
pub const Tile = struct {
    bm: usize,
    bn: usize,
    wm: usize,
    wn: usize,
    stages: usize,
    pub fn shared(t: Tile, prompt: bool) u32 {
        const stage = t.bm * 128 + t.bn * 32 + t.bn * 4 + (if (prompt) @as(usize, 0) else t.bm * 4);
        const partial = t.bm / t.wm / 16 * (t.bn / t.wn / 8) * 4 * (t.wm * t.wn * 32) * 4;
        return @intCast(if (prompt) t.stages * stage else @max(t.stages * stage, partial));
    }
};
pub fn tile(rows: usize, n: usize, k: usize, prompt: bool) Tile {
    if (prompt) return .{ .bm = 128, .bn = 128, .wm = 2, .wn = 2, .stages = 2 };
    if (rows <= 16) if (decTile(n, k)) |cfg| return .{ .bm = 16, .bn = 128, .wm = 1, .wn = if (cfg == 2) 8 else 4, .stages = if (cfg == 2) 4 else 6 };
    return .{ .bm = if (rows <= 16) 16 else if (rows <= 32) 32 else 64, .bn = 64, .wm = 1, .wn = 4, .stages = 4 };
}
fn ci(x: usize) c_int {
    return @intCast(x);
}
fn u(x: usize) u32 {
    return @intCast(x);
}
/// Whether `matmul` may read activations at `x` with row stride `x_stride` as they are: 16-byte aligned rows (the
/// kernels load them with 16-byte asynchronous copies). TP=3's packed KDA projection rows are 8726 / 8341 / 8341
/// elements wide (Geometry.projWidth), so a slice of them fails the stride test; Python's qmm.py makes a contiguous
/// clone there. A caller gets `.copy` and must pass the rows through a scratch of `alignedStride(x_stride)`.
pub const Align = enum { direct, copy };
pub fn alignment(x: u64, x_stride: usize, rows: usize) Align {
    return if (x % 16 != 0 or (rows > 1 and x_stride % 8 != 0)) .copy else .direct;
}
/// The row stride (elements, bf16) of the aligned copy of rows `width` wide.
pub fn alignedStride(width: usize) usize {
    return (width + 7) / 8 * 8;
}
pub const Stage = struct { ptr: u64 = 0, bytes: usize = 0 };
pub fn matmul(e: *const Entries, stream: cuda.Stream, x_in: u64, x_stride_in: usize, xs: u64, q: Q4, out: u64, part: u64, rows: usize, f32_out: bool, clusters: bool, prompt: bool, stage: Stage) !void {
    if (rows == 0 or q.n == 0 or q.k == 0 or q.k % 64 != 0 or (rows > 1 and x_stride_in < q.k)) return error.Invalid;
    _ = stage;
    if (alignment(x_in, x_stride_in, rows) == .copy) return error.Invalid;
    const x = x_in;
    const x_stride = x_stride_in;
    const sk = if (prompt) 1 else splitK(q.n, q.k);
    if (q.k / 64 % sk != 0 or (!clusters and sk > 1 and part == 0)) return error.Invalid;
    const t = tile(rows, q.n, q.k, prompt);
    const rt = (rows + t.bm - 1) / t.bm;
    const npad = (q.n + 127) / 128 * 128;
    const stride = if (q.scale_stride == 0) npad else q.scale_stride;
    if (stride < npad or (prompt and stride != npad)) return error.Invalid;
    const cluster = clusters and sk > 1;
    var name: [80]u8 = undefined;
    const sym: [:0]const u8 = if (prompt) try printZ(&name, "glm_qmm_prefill_{d}", .{@intFromBool(f32_out)}) else if (rows <= 16 and decTile(q.n, q.k) != null) try printZ(&name, "glm_qmm_cfg{d}_{d}_{d}", .{ decTile(q.n, q.k).?, @intFromBool(f32_out), @intFromBool(cluster) }) else try printZ(&name, "glm_qmm_{d}_{d}_{d}", .{ t.bm, @intFromBool(f32_out), @intFromBool(cluster) });
    var a: cuda.Args = .{};
    a.add(x);
    if (!prompt) a.add(xs);
    for ([_]u64{ q.w, q.s, q.b, out }) |v| a.add(v);
    if (!prompt) a.add(part);
    for ([_]usize{ rows, q.n, q.k }) |v| a.add(ci(v));
    if (!prompt) a.add(ci(sk));
    a.add(ci(stride));
    a.add(ci(if (rows == 1) q.k else x_stride));
    a.add(ci(@max(1, @min(rt, (12 << 20) / (t.bm * q.k * 2)))));
    try e.launch(if (prompt) 1 else 0, sym, .{ .grid = .{ .x = u(rt * ((q.n + t.bn - 1) / t.bn)), .y = 1, .z = u(sk) }, .block = .{ .x = u(t.wm * t.wn * 32) }, .shared = t.shared(prompt), .cluster = if (cluster) .{ .x = 1, .y = 1, .z = u(sk) } else null }, stream, &a);
    if (!prompt and sk > 1 and !cluster) {
        var b: cuda.Args = .{};
        b.add(part);
        b.add(out);
        b.add(@as(i64, @intCast(rows * q.n)));
        b.add(ci(sk));
        const red = try printZ(&name, "glm_qmm_reduce_{d}", .{@intFromBool(f32_out)});
        try e.launch(0, red, .{ .grid = .{ .x = u((rows * q.n + 255) / 256) }, .block = .{ .x = 256 } }, stream, &b);
    }
}
fn printZ(buf: []u8, comptime format: []const u8, args: anytype) ![:0]u8 {
    return std.fmt.bufPrintSentinel(buf, format, args, 0);
}
