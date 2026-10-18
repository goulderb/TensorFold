//! Independent latent-attention inputs and float64 chunk references for the CUDA kernel contract.
const std = @import("std");
const core = @import("core");
const c = core.contract.spec;
const g = core.contract.gates;
const rows = core.contract.rows;

pub const lw: usize = 512;
pub const patterns = 16;
pub const edge_counts = [_]usize{ 0, 1, 31, 32, 33, 511, 512, 513, 1023, 1024, 1025, 2047, 2048, 2049, 2050, 2051 };

pub const Data = struct {
    a: std.mem.Allocator,
    shape: c.Shape,
    sparse: bool,
    fp8: bool,
    fill: g.Fill,
    q: []u16,
    cache: []u8,
    values: []f64,
    scales: []f64,
    counts: []i32,
    tokens: []i32,
    position: i32,
    cache_rows: usize,
    nch: usize,

    pub fn init(a: std.mem.Allocator, shape: c.Shape, sparse: bool, fp8: bool, fill: g.Fill, seed: u64) !Data {
        const cache_rows = if (sparse) 3 * shape.k + 1 else shape.k;
        var d = Data{
            .a = a,
            .shape = shape,
            .sparse = sparse,
            .fp8 = fp8,
            .fill = fill,
            .q = try a.alloc(u16, shape.rows * shape.n * lw),
            .cache = &.{},
            .values = &.{},
            .scales = &.{},
            .counts = &.{},
            .tokens = &.{},
            .position = @intCast(if (sparse) 0 else shape.k - shape.rows),
            .cache_rows = cache_rows,
            .nch = (shape.k + 511) / 512,
        };
        errdefer d.deinit();
        d.cache = try a.alloc(u8, cache_rows * (if (fp8) lw + 16 else lw * 2));
        d.values = try a.alloc(f64, cache_rows * patterns);
        d.scales = try a.alloc(f64, cache_rows);
        d.counts = try a.alloc(i32, shape.rows);
        d.tokens = try a.alloc(i32, shape.rows * shape.k);
        @memset(d.q, 0);
        @memset(d.cache, 0);
        @memset(d.tokens, -1);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        for (0..cache_rows) |key| {
            d.scales[key] = if (fill == .stress) 128 else if (fill == .nonfinite) 1 else std.math.ldexp(@as(f64, 1), @as(i32, @intCast(key % 5)) - 2);
            for (0..patterns) |col| {
                const value: f32 = if (fill == .nonfinite) 1 else if (fill == .stress) blk: {
                    const slot = if (sparse) key / 3 else key;
                    const within = slot % 512;
                    const base: f32 = if (slot % 256 == 0) 1 else decode(@intCast(56 + ((slot % 256) * 37 + col * 13 + seed % 71) % 71));
                    break :blk if (within < 256) base else -base + @as(f32, if (within == 256) 1 else 0);
                } else @as(f32, @floatFromInt(@as(i32, @intCast(rng.uintLessThan(u32, 29))) - 14)) / 8;
                d.values[key * patterns + col] = value;
            }
            const stride: usize = if (fp8) lw + 16 else lw * 2;
            const bytes = d.cache[key * stride ..][0..stride];
            for (0..lw) |col| {
                const value: f32 = @floatCast(d.values[key * patterns + columnPattern(col)] * columnFactor(col));
                if (fp8) bytes[col] = encode(value) else std.mem.writeInt(u16, bytes[col * 2 ..][0..2], c.toBf16(value * @as(f32, @floatCast(d.scales[key]))), .little);
            }
            if (fp8) std.mem.writeInt(u32, bytes[lw..][0..4], @bitCast(@as(f32, @floatCast(d.scales[key]))), .little);
        }
        for (0..shape.rows) |row| {
            d.counts[row] = @intCast(if (!sparse or fill != .random) shape.k else @min(shape.k, edge_counts[row % edge_counts.len]));
            for (0..@as(usize, @intCast(d.counts[row]))) |i| if (sparse) {
                d.tokens[row * shape.k + i] = @intCast(3 * i);
            };
            for (0..shape.n) |head| {
                const q = d.q[(row * shape.n + head) * lw ..][0..lw];
                for (0..patterns) |col| q[col] = c.toBf16(if (fill == .stress) 0 else if (fill == .nonfinite) 1 else @as(f32, @floatFromInt(@as(i32, @intCast((row % 4 + head % 4 + col) % 5)) - 2)) / 8);
                if (fill == .nonfinite and row < 4) {
                    q[0] = c.toBf16(switch (row) {
                        0 => std.math.nan(f32),
                        1, 3 => std.math.inf(f32),
                        else => -std.math.inf(f32),
                    });
                    if (row == 3) q[1] = c.toBf16(-std.math.inf(f32));
                }
            }
        }
        return d;
    }

    pub fn deinit(d: Data) void {
        d.a.free(d.q);
        d.a.free(d.cache);
        d.a.free(d.values);
        d.a.free(d.scales);
        d.a.free(d.counts);
        d.a.free(d.tokens);
    }

    pub fn length(d: Data, row: usize) usize {
        return if (d.sparse) @intCast(d.counts[row]) else @as(usize, @intCast(d.position)) + row + 1;
    }

    pub fn keyAt(d: Data, row: usize, at: usize) usize {
        return if (d.sparse) @intCast(d.tokens[row * d.shape.k + at]) else at;
    }

    pub fn valueAt(d: Data, key_: usize, col: usize) f64 {
        return d.values[key_ * patterns + columnPattern(col)] * columnFactor(col) * d.scales[key_];
    }

    pub fn query(d: Data, row: usize, head: usize, col: usize) f64 {
        return c.fromBf16(d.q[(row * d.shape.n + head) * lw + col]);
    }
};

pub fn columnPattern(col: usize) usize {
    return (col % patterns + (col / patterns) * 7) % patterns;
}

pub fn columnFactor(col: usize) f64 {
    const block = col / patterns;
    return (if (block < 16) @as(f64, 1) else -1) * std.math.ldexp(@as(f64, 1), -@as(i32, @intCast(block % 4)));
}

fn encode(x: f32) u8 {
    if (x == 0) return if (std.math.signbit(x)) 0x80 else 0;
    const sign: u8 = if (x < 0) 0x80 else 0;
    const v = @abs(x);
    for (0..127) |code| {
        const value = decode(@intCast(code));
        if (value == v) return sign | @as(u8, @intCast(code));
    }
    unreachable;
}

fn decode(code: u8) f32 {
    const e: i32 = @as(i32, @intCast(code >> 3)) - 7;
    const mant: f32 = @floatFromInt(code & 7);
    return if (code < 8) mant * 0x1p-9 else (1 + mant / 8) * std.math.ldexp(@as(f32, 1), e);
}

pub const Ref = struct {
    po: [patterns]f64 = @splat(0),
    tol: [patterns]f64 = @splat(0),
    pm: f64 = -std.math.inf(f64),
    pl: f64 = 0,
    pm_tol: f64 = 0,
    pl_tol: f64 = 0,
};

pub fn reference(d: Data, row: usize, head: usize, chunk: usize) Ref {
    const begin = chunk * 512;
    const end = @min(begin + 512, d.length(row));
    if (begin >= end) return .{};
    var scores: [512]f64 = undefined;
    var score_errors: [512]f64 = undefined;
    var result: Ref = .{};
    for (begin..end) |i| {
        const key = d.keyAt(row, i);
        var dot: f64 = 0;
        var magnitude: f64 = 0;
        for (0..patterns) |col| {
            const term = d.query(row, head, col) * d.valueAt(key, col);
            dot += term;
            magnitude += @abs(term);
        }
        scores[i - begin] = dot * 0.125;
        score_errors[i - begin] = c.gamma(520, 0x1p-24) * magnitude * 0.125;
        result.pm = @max(result.pm, scores[i - begin]);
        result.pm_tol = @max(result.pm_tol, score_errors[i - begin]);
    }
    for (begin..end) |i| {
        const key = d.keyAt(row, i);
        const p = @exp(scores[i - begin] - result.pm);
        const err = @exp(score_errors[i - begin] + result.pm_tol) - 1 + 0x1p-21;
        result.pl += p;
        result.pl_tol += p * err;
        for (0..patterns) |col| {
            const v = d.valueAt(key, col);
            result.po[col] += p * v;
            result.tol[col] += p * @abs(v) * (err + 0x1p-8 * (1 + err));
        }
    }
    result.pm_tol += 0x1p-149;
    result.pl_tol += c.gamma(544, 0x1p-24) * (result.pl + result.pl_tol) + 0x1p-149;
    for (&result.tol, result.po) |*t, v| t.* += c.gamma(544, 0x1p-24) * (@abs(v) + t.*) + 0x1p-149;
    return result;
}

pub fn stressTerms(d: Data, row: usize, chunk: usize, col: usize, buf: *[512]f64) []const f64 {
    const first = chunk * 512;
    const end = @min(first + 512, d.length(row));
    const n = end -| first;
    for (buf[0..n], 0..) |*v, i| v.* = d.valueAt(d.keyAt(row, first + i), col);
    return buf[0..n];
}

pub fn sum(d: Data, row: usize, chunk: usize, col: usize, acc: c.Accum, order: g.Order) f64 {
    var buf: [512]f64 = undefined;
    return rows.sumIn(acc, order, stressTerms(d, row, chunk, col, &buf));
}

test "attention stress proves fp32 sums against sequential, pairwise and 32-lane reduced precision" {
    for ([_]bool{ false, true }) |sparse| for ([_]bool{ false, true }) |fp8| {
        const d = try Data.init(std.testing.allocator, .{ .rows = 1, .n = 1, .k = 512 }, sparse, fp8, .stress, 9);
        defer d.deinit();
        for ([_]g.Order{ .sequential, .pairwise, .strided }) |order| {
            var margins = [2]f64{ 0, 0 };
            for (0..patterns) |col| {
                const want = sum(d, 0, 0, col, .none, .sequential);
                try std.testing.expectEqual(@as(f64, 128), want);
                try std.testing.expectEqual(want, sum(d, 0, 0, col, .f32, order));
                for ([_]c.Accum{ .f16, .bf16 }, &margins) |acc, *margin| {
                    const got = sum(d, 0, 0, col, acc, order);
                    const miss = if (std.math.isFinite(got)) @abs(got - want) / (0x1p-24 * @abs(want)) else std.math.inf(f64);
                    margin.* = @max(margin.*, miss);
                }
                var terms: [512]f64 = undefined;
                var magnitude: f64 = 0;
                for (stressTerms(d, 0, 0, col, &terms)) |v| magnitude += @abs(v);
                try std.testing.expect(magnitude > 0x1p20 and magnitude < 0x1p24);
            }
            for (margins) |margin| try std.testing.expect(margin >= 100);
        }
    };
}

test "attention column patterns retain distinct vector blocks and exact FP8 source values" {
    const d = try Data.init(std.testing.allocator, .{ .rows = 1, .n = 1, .k = 32 }, false, true, .random, 19);
    defer d.deinit();
    for (0..d.cache_rows) |key_| for (0..lw) |col| {
        const code = d.cache[key_ * (lw + 16) + col];
        const decoded = decode(code & 127) * (if (code & 128 == 0) @as(f32, 1) else -1);
        try std.testing.expectEqual(d.valueAt(key_, col), @as(f64, decoded) * d.scales[key_]);
    };
    for (0..32) |left| for (left + 1..32) |right| {
        var same = true;
        for (0..patterns) |col| same = same and d.valueAt(0, left * patterns + col) == d.valueAt(0, right * patterns + col);
        try std.testing.expect(!same);
    };
}
