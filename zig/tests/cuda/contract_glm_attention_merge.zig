//! GLM's latent-partial merge under the CUDA contract, independently of the chunk producer.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check_ = @import("check.zig");
const Rig = @import("contract_rig.zig").Rig;
const c = core.contract.spec;
const g = core.contract.gates;
const rows = core.contract.rows;
const lw = 512;

fn depth(s: c.Shape) usize {
    return s.k + 4;
}

const dense_shapes = [_]c.Shape{
    .{ .rows = 7, .n = 32, .k = 5 },
    .{ .rows = 1, .n = 1, .k = 1 },
    .{ .rows = 2, .n = 15, .k = 2 },
    .{ .rows = 3, .n = 16, .k = 3 },
    .{ .rows = 4, .n = 17, .k = 4 },
    .{ .rows = 5, .n = 31, .k = 5 },
    .{ .rows = 6, .n = 32, .k = 8 },
    .{ .rows = 7, .n = 33, .k = 64 },
    .{ .rows = 7, .n = 45, .k = 128 },
    .{ .rows = 7, .n = 32, .k = 512 },
    .{ .rows = 63, .n = 32, .k = 4 },
    .{ .rows = 64, .n = 32, .k = 5 },
    .{ .rows = 65, .n = 32, .k = 5 },
    .{ .rows = 512, .n = 32, .k = 8 },
};
const sparse_shapes = dense_shapes ++ [_]c.Shape{.{ .rows = 2048, .n = 32, .k = 5 }};

pub fn spec(sparse: bool) c.Spec {
    return .{
        .name = if (sparse) "cuda GLM tf_glm_merge sparse" else "cuda GLM tf_glm_merge dense",
        .accum = .f32,
        .depth = depth,
        .ops = 16,
        .out = .bf16,
        .nan = .{ .stated = "nonfinite active numerators propagate; sparse count-zero output bytes remain untouched" },
        .tiles = .{ .{ .size = 64 }, .{ .size = 32 }, .{} },
        .served = if (sparse) &sparse_shapes else &dense_shapes,
    };
}

pub const Merge = struct {
    rig: Rig,
    f: cuda.Function,
    sparse: bool,
    shape: c.Shape = .{ .rows = 0, .n = 0, .k = 0 },
    fill: g.Fill = .random,
    host_po: []f32 = &.{},
    host_pm: []f32 = &.{},
    host_pl: []f32 = &.{},
    host_counts: []i32 = &.{},
    po: u64 = 0,
    pm: u64 = 0,
    pl: u64 = 0,
    counts: u64 = 0,
    out: u64 = 0,

    pub fn init(gpu: check_.Gpu, s: cuda.Stream, module: cuda.Module, sparse: bool) !Merge {
        return .{ .rig = .{ .gpu = gpu, .s = s }, .f = try module.function("tf_glm_merge"), .sparse = sparse };
    }

    fn freeHost(self: *Merge) void {
        const a = self.rig.gpu.gpa;
        a.free(self.host_po);
        a.free(self.host_pm);
        a.free(self.host_pl);
        a.free(self.host_counts);
        self.host_po = &.{};
        self.host_pm = &.{};
        self.host_pl = &.{};
        self.host_counts = &.{};
    }

    pub fn deinit(self: *Merge) void {
        self.freeHost();
        self.rig.deinit();
    }

    pub fn kernel(self: *Merge) g.Kernel {
        return .{ .spec = spec(self.sparse), .ctx = self, .vt = &vt };
    }

    const vt = g.hooks(Merge);

    pub fn prepare(self: *Merge, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.rows > 2048 or shape.n == 0 or shape.n > 64 or shape.k == 0 or shape.k > 512) return false;
        self.rig.reset();
        self.freeHost();
        self.shape = shape;
        self.fill = fill;
        const a = self.rig.gpu.gpa;
        const scalars = shape.k * shape.rows * shape.n;
        self.host_po = try a.alloc(f32, scalars * lw);
        self.host_pm = try a.alloc(f32, scalars);
        self.host_pl = try a.alloc(f32, scalars);
        self.host_counts = try a.alloc(i32, shape.rows);
        for (self.host_counts, 0..) |*n, row| n.* = if (self.sparse and fill == .random and row % 16 == 0) 0 else 1;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        for (0..shape.k) |chunk| for (0..shape.rows) |row| for (0..shape.n) |head| {
            const scalar = (chunk * shape.rows + row) * shape.n + head;
            const inactive = fill == .random and shape.k > 1 and chunk == shape.k - 1 and (row + head) % 3 == 0;
            self.host_pm[scalar] = if (inactive) -std.math.inf(f32) else if (fill != .random) 0 else @as(f32, @floatFromInt(@as(i32, @intCast(rng.uintLessThan(u32, 17))) - 8)) / 4;
            self.host_pl[scalar] = if (inactive) 0 else if (fill != .random) 1 else @as(f32, @floatFromInt(1 + rng.uintLessThan(u32, 8))) / 4;
            var pattern: [16]f32 = undefined;
            for (&pattern, 0..) |*v, col| v.* = if (inactive) 0 else if (fill == .stress) blk: {
                const h = shape.k / 2;
                const i = if (chunk < h) chunk else chunk - h;
                const big: f32 = @floatFromInt(16384 + ((i * 37 + row * 13 + head * 7 + col * 17 + seed % 1024) % 8192));
                break :blk if (shape.k % 2 == 1 and chunk == shape.k - 1) 1 else if (chunk < h) big else -big + @as(f32, if (i == 0) 1 else 0);
            } else (rng.float(f32) * 2 - 1) * 8;
            for (self.host_po[scalar * lw ..][0..lw], 0..) |*v, col| v.* = pattern[col % pattern.len];
        };
        if (fill == .nonfinite) {
            const inf = std.math.inf(f32);
            const put = [_]f32{ std.math.nan(f32), inf, -inf, inf };
            for (put, 0..) |v, row| if (row < shape.rows) for (0..shape.n) |head| {
                self.host_po[(row * shape.n + head) * lw + row] = v;
                if (row == 3) self.host_po[((shape.k - 1) * shape.rows * shape.n + row * shape.n + head) * lw + row] = -inf;
            };
        }
        self.po = try self.rig.input(std.mem.sliceAsBytes(self.host_po));
        self.pm = try self.rig.input(std.mem.sliceAsBytes(self.host_pm));
        self.pl = try self.rig.input(std.mem.sliceAsBytes(self.host_pl));
        self.counts = try self.rig.input(std.mem.sliceAsBytes(self.host_counts));
        self.out = try self.rig.output(self.count() * 2);
        return true;
    }

    pub fn run(self: *Merge) !void {
        var args: cuda.Args = .{};
        inline for (.{ self.po, self.pm, self.pl, self.out, self.counts, @as(i64, @intCast(self.shape.rows)), @as(i64, @intCast(self.shape.n)), @as(c_int, @intCast(self.shape.k)), @as(c_int, @intFromBool(self.sparse)) }) |v| args.add(v);
        try cuda.launch.launch(self.f, .{
            .grid = .{ .x = @intCast(self.shape.rows), .y = @intCast(self.shape.n) },
            .block = .{ .x = 128 },
        }, self.rig.s, &args);
        try self.rig.s.synchronize();
    }

    pub fn count(self: *Merge) usize {
        return self.shape.rows * self.shape.n * lw;
    }

    fn one(self: *Merge, row: usize, head: usize, col: usize, acc: c.Accum, order: g.Order) struct { f64, f64 } {
        var terms: [512]f64 = undefined;
        var denominators: [512]f64 = undefined;
        var most: f64 = -std.math.inf(f64);
        for (0..self.shape.k) |chunk| {
            const scalar = (chunk * self.shape.rows + row) * self.shape.n + head;
            if (self.host_pl[scalar] > 0) most = @max(most, self.host_pm[scalar]);
        }
        var mag: f64 = 0;
        for (0..self.shape.k) |chunk| {
            const scalar = (chunk * self.shape.rows + row) * self.shape.n + head;
            const weight: f64 = if (self.host_pl[scalar] > 0) @exp(@as(f64, self.host_pm[scalar]) - most) else 0;
            terms[chunk] = weight * self.host_po[scalar * lw + col];
            denominators[chunk] = weight * self.host_pl[scalar];
            mag += @abs(terms[chunk]);
        }
        const numerator = rows.sumIn(acc, order, terms[0..self.shape.k]);
        const denominator = rows.sumIn(acc, order, denominators[0..self.shape.k]);
        const want = numerator / denominator;
        return .{ want, mag / denominator + @abs(want) };
    }

    pub fn reference(self: *Merge, want: []f64, mag: []f64) !void {
        for (0..self.shape.rows) |row| for (0..self.shape.n) |head| {
            var values: [16]f64 = undefined;
            var magnitudes: [16]f64 = undefined;
            for (0..16) |col| {
                const v = self.one(row, head, col, .none, .sequential);
                values[col] = v[0];
                magnitudes[col] = v[1];
            }
            for (0..lw) |col| {
                const at = (row * self.shape.n + head) * lw + col;
                want[at] = values[col % 16];
                mag[at] = magnitudes[col % 16];
            }
        };
    }

    pub fn simulate(self: *Merge, acc: c.Accum, order: g.Order, out: []f64) !void {
        for (0..self.shape.rows) |row| for (0..self.shape.n) |head| {
            var values: [16]f64 = undefined;
            for (&values, 0..) |*v, col| v.* = self.one(row, head, col, acc, order)[0];
            for (0..lw) |col| out[(row * self.shape.n + head) * lw + col] = values[col % 16];
        };
    }

    pub fn check(self: *Merge, gate: g.Gate) !?[]const u8 {
        if (gate != .nonfinite and gate != .shapes) return null;
        const raw = try self.rig.outputBytes();
        for (0..self.shape.rows) |row| for (0..self.shape.n) |head| {
            var values: [16]f64 = undefined;
            var magnitudes: [16]f64 = undefined;
            for (0..16) |col| {
                const v = self.one(row, head, col, .none, .sequential);
                values[col] = v[0];
                magnitudes[col] = v[1];
            }
            for (0..lw) |col| {
                const at = (row * self.shape.n + head) * lw + col;
                if (self.sparse and self.host_counts[row] == 0) {
                    if (std.mem.readInt(u16, raw[at * 2 ..][0..2], .little) != 0xffff)
                        return "sparse count-zero merge changed its output sentinel";
                    continue;
                }
                const got = c.read(.bf16, raw, at);
                const v = if (gate == .nonfinite) self.one(row, head, col, .none, .sequential) else .{ values[col % 16], magnitudes[col % 16] };
                if (gate == .nonfinite) {
                    if (c.classOf(got) != c.classOf(v[0])) return "merge changed an active numerator's nonfinite class";
                } else if (!(@abs(got - v[0]) <= c.bound(spec(self.sparse), self.shape, v[0], v[1])))
                    return "merge exceeds its independent float64 weighted-sum/division bound";
            }
        };
        return null;
    }
};
