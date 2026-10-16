const std = @import("std");
const cuda = @import("cuda");
pub const shared = @import("cuda_dflash_movement.zig");
pub const Ops = struct {
    module: cuda.Module,
    base: shared.Ops,
    sort: cuda.Function,
    pack: cuda.Function,
    index_copy: cuda.Function,
    rotary: cuda.Function,
    ring: cuda.Function,
    rank_add: cuda.Function,
    pub fn load(ctx: *const cuda.Context) !Ops {
        var base = try shared.Ops.load(ctx);
        errdefer base.deinit();
        var m = try cuda.Module.load(ctx.d, cuda.kernels.dflash_ops);
        errdefer m.unload();
        return .{
            .module = m,
            .base = base,
            .sort = try m.function("glm_sort"),
            .pack = try m.function("glm_pack"),
            .index_copy = try m.function("glm_index_copy"),
            .rotary = try m.function("glm_rotary"),
            .ring = try m.function("glm_ring"),
            .rank_add = try m.function("glm_rank_add"),
        };
    }
    pub fn deinit(o: *Ops) void {
        o.module.unload();
        o.base.deinit();
    }
    pub fn on(o: *const Ops, s: cuda.Stream) Torch {
        return .{ .ops = o, .s = s, .base = o.base.on(s) };
    }
};
pub const Torch = struct {
    ops: *const Ops,
    s: cuda.Stream,
    base: shared.Torch,
    fn go(t: Torch, f: cuda.Function, rows: usize, args: *cuda.Args) !void {
        if (rows == 0) return;
        try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 256 } }, t.s, args);
    }
    pub fn topkSorted(t: Torch, input: u64, columns: usize, rows: usize, k: usize, vals: u64, ids: u64, scratch: u64) !void {
        if (k == 0 or k > columns or k > 16384 or rows > 65535) return error.TopkShape;
        try t.base.topk(input, columns, rows, k, vals, ids, scratch);
        var a: cuda.Args = .{};
        a.add(vals);
        a.add(ids);
        a.add(@as(u32, @intCast(k)));
        try t.go(t.ops.sort, rows, &a);
    }
    pub fn pack(t: Torch, vals: u64, ids: u64, out: u64, rows: usize, k: usize, offset: i64) !void {
        var a: cuda.Args = .{};
        for ([_]u64{ vals, ids, out }) |x| a.add(x);
        a.add(@as(u32, @intCast(k)));
        a.add(offset);
        try t.go(t.ops.pack, rows, &a);
    }
    pub fn indexCopy(t: Torch, src: u64, dst: u64, ids: u64, rows: usize, dest_rows: usize, inner_bytes: usize, outer: usize, invalid: u64) !void {
        if (rows == 0 or outer == 0 or inner_bytes == 0) return;
        var a: cuda.Args = .{};
        for ([_]u64{ src, dst, ids, rows, dest_rows, inner_bytes, outer, invalid }) |x| a.add(x);
        try cuda.launch.launch(t.ops.index_copy, .{ .grid = .{ .x = @intCast(rows), .y = @intCast(@min(outer, 65535)) }, .block = .{ .x = 256 } }, t.s, &a);
    }
    pub fn rotary(t: Torch, pos: u64, inv: u64, cos: u64, sin: u64, rows: usize, half: usize) !void {
        var a: cuda.Args = .{};
        for ([_]u64{ pos, inv, cos, sin, rows, half }) |x| a.add(x);
        if (rows * half != 0) try t.go(t.ops.rotary, blocks(rows * half), &a);
    }
    pub fn ring(t: Torch, pos: u64, slots: u64, out: u64, rows: usize, cap: i64) !void {
        if (cap <= 0) return error.BadRing;
        var a: cuda.Args = .{};
        for ([_]u64{ pos, slots, out, rows }) |x| a.add(x);
        a.add(cap);
        if (rows != 0) try t.go(t.ops.ring, blocks(rows), &a);
    }
    pub fn rankAdd(t: Torch, ranks: u64, out: u64, count: usize, world: usize) !void {
        if (world < 1 or world > 4) return error.UnsupportedTp;
        var a: cuda.Args = .{};
        for ([_]u64{ ranks, out, count }) |x| a.add(x);
        a.add(@as(u32, @intCast(world)));
        if (count != 0) try t.go(t.ops.rank_add, blocks(count), &a);
    }
};
fn blocks(n: usize) usize {
    return @min((n + 255) / 256, 65535);
}
