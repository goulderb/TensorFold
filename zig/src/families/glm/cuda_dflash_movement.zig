const std = @import("std");
const cuda = @import("cuda");
const topk_tile = 4096;
const topk_threads = 256;
pub const Functions = struct {
    hist: cuda.Function,
    digit: cuda.Function,
    count: cuda.Function,
    prefix: cuda.Function,
    compact: cuda.Function,
    to_f32: cuda.Function,
};
pub const Ops = struct {
    mods: [2]cuda.Module,
    f: Functions,
    pub fn load(ctx: *const cuda.Context) !Ops {
        var a = try cuda.Module.load(ctx.d, cuda.kernels.torch_topk);
        errdefer a.unload();
        var b = try cuda.Module.load(ctx.d, cuda.kernels.torch_pointwise);
        errdefer b.unload();
        return .{ .mods = .{ a, b }, .f = .{
            .hist = try a.function("tf_topk_f32_histogram_kernel"),
            .digit = try a.function("tf_topk_f32_choose_digit_kernel"),
            .count = try a.function("tf_topk_f32_count_kernel"),
            .prefix = try a.function("tf_topk_f32_prefix_kernel"),
            .compact = try a.function("tf_topk_f32_compact_kernel"),
            .to_f32 = try b.function("tf_bf16_to_f32_kernel"),
        } };
    }
    pub fn deinit(o: *Ops) void {
        for (&o.mods) |*m| m.unload();
    }
    pub fn on(o: *const Ops, s: cuda.Stream) Torch {
        return .{ .f = &o.f, .s = s };
    }
};
fn tiles(columns: usize) usize {
    return (columns + topk_tile - 1) / topk_tile;
}
fn blocks(count: usize, threads: usize) u32 {
    return @intCast(@max(@min((count + threads - 1) / threads, 65535), 1));
}
pub fn topkScratchBytes(rows: usize, columns: usize) usize {
    return rows * (tiles(columns) * 260 + 3) * 4;
}
pub const Torch = struct {
    f: *const Functions,
    s: cuda.Stream,
    fn go(t: Torch, f: cuda.Function, grid: [2]usize, block: u32, args: *cuda.Args) !void {
        try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(grid[0]), .y = @intCast(grid[1]) }, .block = .{ .x = block } }, t.s, args);
    }
    fn check(t: Torch, rc: anytype, what: []const u8) !void {
        try t.s.d.check(rc, what);
    }
    pub fn copy(t: Torch, dst: u64, src: u64, bytes: usize) !void {
        if (bytes == 0) return;
        try t.check(t.s.d.api.cuMemcpyDtoDAsync_v2(dst, src, bytes, t.s.handle), "cuMemcpyDtoDAsync");
    }
    pub fn fill32(t: Torch, dst: u64, value: u32, count: usize) !void {
        if (count == 0) return;
        try t.check(t.s.d.api.cuMemsetD32Async(dst, value, count, t.s.handle), "cuMemsetD32Async");
    }
    pub fn toF32(t: Torch, in: u64, out: u64, count: usize) !void {
        if (count == 0) return;
        var a: cuda.Args = .{};
        a.add(in);
        a.add(out);
        a.add(@as(u64, count));
        try t.go(t.f.to_f32, .{ blocks(count, 256), 1 }, 256, &a);
    }
    pub fn topk(t: Torch, in: u64, columns: usize, rows: usize, k: usize, values: u64, indices: u64, scratch: u64) !void {
        const n = tiles(columns);
        const row_bytes: u64 = columns * 4;
        const hist = scratch;
        const threshold = hist + rows * n * 256 * 4;
        const remaining = threshold + rows * 4;
        const greater = remaining + rows * 4;
        const equal = greater + rows * n * 4;
        const greater_prefix = equal + rows * n * 4;
        const equal_prefix = greater_prefix + rows * n * 4;
        const greater_total = equal_prefix + rows * n * 4;
        var shift: u32 = 24;
        while (true) : (shift -= 8) {
            var h: cuda.Args = .{};
            for ([_]u64{ in, hist, threshold, columns, n, row_bytes, 4 }) |v| h.add(v);
            h.add(shift);
            try t.go(t.f.hist, .{ n, rows }, topk_threads, &h);
            var d: cuda.Args = .{};
            for ([_]u64{ hist, threshold, remaining, n }) |v| d.add(v);
            d.add(@as(u32, @intCast(k)));
            d.add(shift);
            try t.go(t.f.digit, .{ rows, 1 }, topk_threads, &d);
            if (shift == 0) break;
        }
        var c: cuda.Args = .{};
        for ([_]u64{ in, threshold, greater, equal, columns, n, row_bytes, 4 }) |v| c.add(v);
        try t.go(t.f.count, .{ n, rows }, topk_threads, &c);
        var p: cuda.Args = .{};
        for ([_]u64{ greater, equal, greater_prefix, equal_prefix, greater_total, n }) |v| p.add(v);
        try t.go(t.f.prefix, .{ rows, 1 }, 1, &p);
        var w: cuda.Args = .{};
        for ([_]u64{ in, values, indices, threshold, greater_prefix, equal_prefix, greater_total, columns, n, row_bytes, 4 }) |v| w.add(v);
        w.add(@as(u32, @intCast(k)));
        try t.go(t.f.compact, .{ n, rows }, topk_threads, &w);
    }
};
