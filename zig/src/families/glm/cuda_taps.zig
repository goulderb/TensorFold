const std = @import("std");

pub const layers = [_]usize{ 5, 14, 24, 33, 42 };
pub const hidden = 4096;

pub fn index(layer: usize) ?usize {
    for (layers, 0..) |wanted, i| if (layer == wanted) return i;
    return null;
}

pub const View = struct {
    ptr: u64,
    plane_rows: usize,
    row0: usize = 0,
    rows: usize,

    pub fn validate(v: View) !void {
        if (v.ptr == 0 or v.rows == 0 or v.row0 > v.plane_rows or v.rows > v.plane_rows - v.row0) return error.BadTapView;
        const bytes = try std.math.mul(u64, try std.math.mul(u64, v.plane_rows, hidden * 2), layers.len);
        _ = try std.math.add(u64, v.ptr, bytes);
    }

    pub fn plane(v: View, i: usize) u64 {
        std.debug.assert(i < layers.len);
        return v.ptr + (i * v.plane_rows + v.row0) * hidden * 2;
    }
};

pub const Copy = struct { source: u64, destination: u64, bytes: usize };

pub fn copies(from: View, to: View) ![layers.len]Copy {
    try from.validate();
    try to.validate();
    if (from.rows != to.rows) return error.BadTapView;
    var out: [layers.len]Copy = undefined;
    for (&out, 0..) |*range, i| range.* = .{ .source = from.plane(i), .destination = to.plane(i), .bytes = from.rows * hidden * 2 };
    return out;
}

test "tap slices retain allocated strides across a short shared segment" {
    const c = try copies(.{ .ptr = 0x1000, .plane_rows = 32, .row0 = 4, .rows = 3 }, .{ .ptr = 0x2000, .plane_rows = 8, .rows = 3 });
    try std.testing.expectEqual(@as(u64, 0x9000), c[0].source);
    try std.testing.expectEqual(@as(u64, 0x49000), c[1].source);
    try std.testing.expectEqual(@as(u64, 0x2000), c[0].destination);
    try std.testing.expectEqual(@as(u64, 0x12000), c[1].destination);
    try std.testing.expectEqual(@as(usize, 24576), c[4].bytes);
    try std.testing.expectEqual(@as(u64, 0x109000), c[4].source);
    try std.testing.expectError(error.BadTapView, copies(.{ .ptr = 1, .plane_rows = 8, .row0 = 7, .rows = 2 }, .{ .ptr = 2, .plane_rows = 8, .rows = 2 }));
}

test "only the five post-layer residuals are drafter taps" {
    try std.testing.expectEqual(@as(?usize, 0), index(5));
    try std.testing.expectEqual(@as(?usize, 4), index(42));
    try std.testing.expectEqual(@as(?usize, null), index(43));
    try std.testing.expectEqual(@as(?usize, null), index(4));
}

const cuda = @import("cuda");
/// Where a forward's taps go: rows [R][5 * hidden] bf16 at `out`, through glm_dflash's df_tap.
pub const Taps = struct {
    lifetime: u64 = 0, // unique allocation/module owner generation, not the device address
    f: cuda.Function,
    out: u64,

    /// After layer `j` of an `R`-row pass on residual `x` (bf16 [R][4][4096]): its tap slot, if `j` is tapped.
    pub fn at(t: Taps, s: cuda.Stream, j: usize, x: u64, R: usize) !void {
        const slot: i32 = for (layers, 0..) |l, i| {
            if (l == j) break @intCast(i);
        } else return;
        var a: cuda.Args = .{};
        a.add(x);
        a.add(t.out);
        a.add(slot);
        a.add(@as(i32, 0)); // the mean of the 4 streams (df_tap's mode 0)
        try cuda.launch.launch(t.f, .{ .grid = .{ .x = @intCast(R), .y = 4096 / 1024 }, .block = .{ .x = 256 } }, s, &a);
    }
};
