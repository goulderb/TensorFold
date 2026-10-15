//! The CUDA oracle's 2-D RoPE, in the same merge-block order as its image patches.
//! There is no learned position table or positional interpolation in this tower.
const std = @import("std");
const Config = @import("cuda_vision_config.zig").Config;

pub const Table = struct {
    rows: usize,
    width: u32,
    cos: []f32,
    sin: []f32,
    pub fn init(a: std.mem.Allocator, c: Config, grid: [3]u32) !Table {
        try c.check();
        if (grid[0] != 1 or grid[1] == 0 or grid[2] == 0 or grid[1] % c.merge != 0 or grid[2] % c.merge != 0)
            return error.InvalidVisionGrid;
        const rows = std.math.mul(usize, grid[1], grid[2]) catch return error.InvalidVisionGrid;
        if (rows > 4 * 8000) return error.InvalidVisionGrid;
        const n = std.math.mul(usize, rows, c.headDim()) catch return error.InvalidVisionGrid;
        const cos = try a.alloc(f32, n);
        errdefer a.free(cos);
        const sin = try a.alloc(f32, n);
        errdefer a.free(sin);
        const spatial = c.headDim() / 2;
        const quarter = spatial / 2;
        var row: usize = 0;
        for (0..grid[1] / c.merge) |bh| for (0..grid[2] / c.merge) |bw| for (0..c.merge) |mh| for (0..c.merge) |mw| {
            const y: f32 = @floatFromInt(bh * c.merge + mh);
            const x: f32 = @floatFromInt(bw * c.merge + mw);
            for (0..c.headDim()) |i| {
                const base = i % spatial;
                const j = base % quarter;
                const exponent = @as(f32, @floatFromInt(2 * j)) / @as(f32, @floatFromInt(spatial));
                const inv: f32 = 1 / std.math.pow(f32, c.theta, exponent);
                const phase = (if (base < quarter) y else x) * inv;
                if (!std.math.isFinite(inv) or !std.math.isFinite(phase)) return error.InvalidVisionConfig;
                cos[row * c.headDim() + i] = @cos(phase);
                sin[row * c.headDim() + i] = @sin(phase);
            }
            row += 1;
        };
        return .{ .rows = rows, .width = c.headDim(), .cos = cos, .sin = sin };
    }
    pub fn deinit(self: *Table, a: std.mem.Allocator) void {
        a.free(self.cos);
        a.free(self.sin);
        self.* = undefined;
    }
};

test "2-D RoPE traverses merge blocks and duplicates the two spatial axes" {
    const a = std.testing.allocator;
    const c = try @import("cuda_vision_config.zig").parse(a, @embedFile("fixtures_cuda_config_exl3.json"));
    var t = try Table.init(a, c, .{ 1, 2, 4 });
    defer t.deinit(a);
    try std.testing.expectEqual(@as(usize, 8), t.rows);
    for (t.cos[0..64], t.sin[0..64]) |co, si| {
        try std.testing.expectEqual(@as(f32, 1), co);
        try std.testing.expectEqual(@as(f32, 0), si);
    }
    // Row four is (y=0,x=2): the x axis is slots 16..31, repeated in 48..63.
    try std.testing.expectApproxEqAbs(@as(f32, -0.41614684), t.cos[4 * 64 + 16], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.90929743), t.sin[4 * 64 + 16], 1e-6);
    for (0..t.rows) |r| {
        try std.testing.expectEqualSlices(f32, t.cos[r * 64 ..][0..32], t.cos[r * 64 + 32 ..][0..32]);
        try std.testing.expectEqualSlices(f32, t.sin[r * 64 ..][0..32], t.sin[r * 64 + 32 ..][0..32]);
    }
    try std.testing.expectError(error.InvalidVisionGrid, Table.init(a, c, .{ 2, 2, 4 }));
    try std.testing.expectError(error.InvalidVisionGrid, Table.init(a, c, .{ 1, 3, 4 }));
}
