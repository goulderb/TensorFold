//! GLM checkpoint-source projections: MXFP8 and block FP8, the BF16 upcast, and every dense row capacity.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const glm = @import("glm_cuda");
const rig_ = @import("contract_rig.zig");
const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const Dense = contract.dense.Dense;

pub const Kind = enum { mx_f32, mx_bf16, pb_f32, pb_bf16 };

fn isMx(kind: Kind) bool {
    return kind == .mx_f32 or kind == .mx_bf16;
}

fn isWide(kind: Kind) bool {
    return kind == .mx_f32 or kind == .pb_f32;
}

fn depth(s: c.Shape) usize {
    return s.k / 32 + 7;
}

const served = [_]c.Shape{
    .{ .rows = 1, .n = 512, .k = 4096 },
    .{ .rows = 2, .n = 1536, .k = 4096 },
    .{ .rows = 3, .n = 512, .k = 1536 },
    .{ .rows = 4, .n = 4096, .k = 128 },
    .{ .rows = 7, .n = 512, .k = 4096 },
    .{ .rows = 8, .n = 4096, .k = 1536 },
    .{ .rows = 15, .n = 512, .k = 128 },
    .{ .rows = 16, .n = 512, .k = 4096 },
    .{ .rows = 17, .n = 128, .k = 4096 },
    .{ .rows = 56, .n = 128, .k = 4096 },
    .{ .rows = 2048, .n = 32, .k = 128 },
};

/// One tf_exlr8_dense_N's gate 1 and 4 rows; `planted` is the first of fillX's nonfinite rows (0-3) its rows hold.
pub const Capacity = struct { rows: usize, planted: usize = 0 };

/// Gates 1 and 4 pick 5-8 and 1 rows, so each capacity has its own cases; one and two rows split the planted rows.
pub const capacities = [_]Capacity{ .{ .rows = 1 }, .{ .rows = 1, .planted = 3 }, .{ .rows = 2 }, .{ .rows = 2, .planted = 2 }, .{ .rows = 4 }, .{ .rows = 8 }, .{ .rows = 16 } };

const capacity_served = blk: {
    var out: [capacities.len][2]c.Shape = undefined;
    for (capacities, &out) |cap, *s| s.* = .{ .{ .rows = cap.rows, .n = 512, .k = 4096 }, .{ .rows = cap.rows, .n = 4096, .k = 1536 } };
    break :blk out;
};

const base_names = [_][]const u8{ "cuda GLM source MXFP8 fp32", "cuda GLM source MXFP8 bf16", "cuda GLM source FP8_PB_WO fp32", "cuda GLM source FP8_PB_WO bf16" };

const capacity_names = blk: {
    var out: [base_names.len][capacities.len][]const u8 = undefined;
    for (base_names, &out) |base, *row| for (capacities, row) |cap, *name| {
        name.* = std.fmt.comptimePrint("{s} rows {d} planted {d}", .{ base, cap.rows, cap.planted });
    };
    break :blk out;
};

pub fn spec(kind: Kind, sliced: bool, capacity: ?usize) c.Spec {
    const sliced_names = [_][]const u8{
        "cuda GLM source MXFP8 sliced/padded fp32",     "cuda GLM source MXFP8 sliced/padded bf16",
        "cuda GLM source FP8_PB_WO sliced/padded fp32", "cuda GLM source FP8_PB_WO sliced/padded bf16",
    };
    const i = @backingInt(kind);
    return .{
        .name = if (capacity) |at| capacity_names[i][at] else if (sliced) sliced_names[i] else base_names[i],
        .accum = .f32,
        .depth = depth,
        .ops = 1,
        .out = if (isWide(kind)) .f32 else .bf16,
        .tiles = .{ .{ .size = 16 }, .{ .size = 32 }, .{ .size = 128, .step = 128, .min = 128 } },
        .served = if (capacity) |at| &capacity_served[at] else &served,
    };
}

/// fillX for one case: nonfinite rows from `planted` on, so one- and two-row capacities hold the later planted rows.
fn fillRows(a: std.mem.Allocator, d: Dense, cap: Capacity, fill: g.Fill, seed: u64) !void {
    if (fill != .nonfinite or cap.planted == 0) return d.fillX(fill, seed, 2);
    const wide = try Dense.init(a, .{ .rows = cap.planted + d.shape.rows, .n = 1, .k = d.shape.k }, 1);
    defer wide.deinit();
    wide.fillX(fill, seed, 2);
    @memcpy(d.x, wide.x[cap.planted * d.shape.k ..]);
}

fn tensor(dtype: core.checkpoint.DType, n: usize, k: usize, bytes: []const u8) core.checkpoint.Tensor {
    var t: core.checkpoint.Tensor = .{ .dtype = dtype, .rank = 2, .shape = @splat(1), .bytes = bytes };
    t.shape[0] = n;
    t.shape[1] = k;
    return t;
}

pub const Source = struct {
    rig: rig_.Rig,
    dense: *const cuda.exlr8.Dense,
    cast: cuda.Function,
    kind: Kind,
    sliced: bool = false,
    capacity: ?usize = null,
    d: ?Dense = null,
    source: glm.b16.Source = undefined,
    x: u64 = 0,
    part: u64 = 0,
    out: u64 = 0,
    stride: usize = 0,

    pub fn deinit(self: *Source) void {
        if (self.d) |d| d.deinit();
        self.rig.deinit();
    }

    pub fn kernel(self: *Source) g.Kernel {
        return .{ .spec = spec(self.kind, self.sliced, self.capacity), .ctx = self, .vt = &vt };
    }

    const vt = g.hooks(Source);

    pub fn prepare(self: *Source, asked: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (self.d) |d| d.deinit();
        self.d = null;
        self.rig.reset();
        var shape = asked;
        const cap: Capacity = if (self.capacity) |at| capacities[at] else .{ .rows = shape.rows };
        if (fill != .random) shape.rows = cap.rows;
        if (shape.rows == 0 or shape.n == 0 or shape.k == 0 or shape.k % 128 != 0) return false;
        const a = self.rig.gpu.gpa;
        const d = try Dense.init(a, shape, 1);
        self.d = d;
        try fillRows(a, d, cap, fill, seed);
        const mx = isMx(self.kind);
        const row0: usize = if (self.sliced) 123 else 0;
        const col0: usize = if (self.sliced) (if (mx) 32 else 4) else 0;
        const n = row0 + shape.n;
        const k = shape.k + if (self.sliced) @as(usize, 128) else 0;
        const codes = try a.alloc(u8, n * k);
        defer a.free(codes);
        var prng = std.Random.DefaultPrng.init(seed ^ 0x5388);
        const rng = prng.random();
        for (codes) |*q| {
            q.* = rng.int(u8);
            if (q.* & 0x7f == 0x7f) q.* = 0x01;
        }
        const scales = try a.alloc(u8, if (mx) n * (k / 32) else ((n + 127) / 128) * ((k + 127) / 128) * 4);
        defer a.free(scales);
        if (mx) {
            for (scales) |*s| s.* = if (fill == .stress) 127 else 117 + rng.uintLessThan(u8, 10);
        } else {
            for (0..scales.len / 4) |i| {
                const s: f32 = if (fill == .stress) 1 else 0.001 + rng.float(f32) * 0.01;
                std.mem.writeInt(u32, scales[4 * i ..][0..4], @bitCast(s), .little);
            }
        }
        if (fill == .stress) {
            const positive = [_]u8{ 0x38, 0x40, 0x44, 0x48, 0x4a, 0x4c };
            for (0..shape.n) |j| for (0..shape.k / 2) |i| {
                const q = positive[rng.uintLessThan(usize, positive.len)];
                codes[(row0 + j) * k + col0 + i] = q;
                codes[(row0 + j) * k + col0 + shape.k / 2 + i] = q | 0x80;
            };
        }
        const wt = tensor(.f8_e4m3, n, k, codes);
        const st = tensor(if (mx) .u8 else .f32, if (mx) n else (n + 127) / 128, if (mx) k / 32 else (k + 127) / 128, scales);
        const rs: core.fp8_source.Span = .{ .lo = row0, .hi = row0 + shape.n };
        const cs: core.fp8_source.Span = .{ .lo = col0, .hi = col0 + shape.k };
        if (mx) try core.fp8_source.mxfp8(wt, st, rs, cs, d.w) else try core.fp8_source.pbWo(wt, st, rs, cs, d.w);
        var p = try glm.source_dense.pack(a, .{ .weight = wt, .scale = st, .format = if (mx) .mxfp8 else .fp8_pb_wo }, &.{rs}, cs);
        defer p.deinit(a);
        try p.checkCuda();
        const payload = try a.alloc(u8, p.resident());
        defer a.free(payload);
        @memcpy(payload[0..p.codes.len], p.codes);
        @memcpy(payload[p.codes.len..], p.scales);
        const job = p.job();
        const groups = try core.exlr8.job.denseGroups(a, &.{job});
        defer a.free(groups);
        self.source = .{
            .weights = try self.rig.input(payload),
            .jobs = try self.rig.input(std.mem.asBytes(&job)),
            .groups = try self.rig.input(std.mem.sliceAsBytes(groups)),
            .groups_n = @intCast(groups.len),
            .n = @intCast(shape.n),
            .k = @intCast(shape.k),
        };
        self.stride = shape.k + if (self.sliced) @as(usize, 13) else 0;
        const x = try a.alloc(u16, shape.rows * self.stride);
        defer a.free(x);
        @memset(x, 0x7fc0);
        for (0..shape.rows) |r| @memcpy(x[r * self.stride ..][0..shape.k], d.x[r * shape.k ..][0..shape.k]);
        self.x = try self.rig.input(std.mem.sliceAsBytes(x));
        self.part = try self.rig.scratch(glm.b16.sourcePartBytes(shape.rows, shape.k));
        self.out = try self.rig.output(shape.rows * shape.n * c.size(spec(self.kind, self.sliced, self.capacity).out));
        return true;
    }

    pub fn run(self: *Source) !void {
        const d = self.d.?;
        const set: glm.glue.Set = .{ .glue = undefined, .exlr8_dense = self.dense, .source_upcast = self.cast };
        try glm.b16.sourceMatmul(set, self.rig.s, self.x, self.stride, self.source, self.out, self.part, d.shape.rows, d.shape.n, d.shape.k, isWide(self.kind));
        try self.rig.s.synchronize();
    }

    pub fn count(self: *Source) usize {
        return self.d.?.outputs();
    }

    pub fn reference(self: *Source, want: []f64, mag: []f64) !void {
        try self.d.?.reference(want, mag);
    }

    pub fn simulate(self: *Source, acc: c.Accum, order: g.Order, out: []f64) !void {
        try self.d.?.simulate(acc, order, out);
    }
};

test "each exlr8 row capacity's gate 1 cases hold NaN, +Inf and -Inf inputs" {
    const a = std.testing.allocator;
    for ([_]usize{ 1, 2, 4, 8, 16 }) |rows| {
        var seen: [3]bool = @splat(false);
        for (capacities) |cap| if (cap.rows == rows) {
            const d = try Dense.init(a, .{ .rows = rows, .n = 1, .k = 4096 }, 1);
            defer d.deinit();
            try fillRows(a, d, cap, .nonfinite, 7);
            for (d.x) |bits| {
                const v = c.fromBf16(bits);
                for ([_]bool{ std.math.isNan(v), v == std.math.inf(f32), v == -std.math.inf(f32) }, &seen) |hit, *s| s.* = s.* or hit;
            }
        };
        try std.testing.expect(std.mem.allEqual(bool, &seen, true));
    }
}
