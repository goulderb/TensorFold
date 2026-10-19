//! The keyed host sampler (Python exact_sampling.py) bit for bit: libm's exp and log, numpy's sum order.
const std = @import("std");
const Allocator = std.mem.Allocator;

extern "c" fn exp(x: f64) f64;
extern "c" fn log(x: f64) f64;

pub const Sampling = struct {
    seed: u64,
    temperature: f64 = 1.0,
    top_k: u32 = 20,
    top_p: f64 = 0.95,
    min_p: f64 = 0.0,

    /// ln(min_p), -inf when off.
    pub fn minLog(s: Sampling) f64 {
        return if (s.min_p > 0.0) log(s.min_p) else -std.math.inf(f64);
    }
};

/// A reproducible seed from the prompt: sha256 of the ids joined by commas and "|salt", first 8 bytes, 63 bits.
pub fn seedFor(tokens: []const u32, salt: i64) u64 {
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    var buf: [32]u8 = undefined;
    for (tokens, 0..) |t, i| {
        if (i > 0) h.update(",");
        h.update(std.fmt.bufPrint(&buf, "{d}", .{t}) catch unreachable);
    }
    h.update(std.fmt.bufPrint(&buf, "|{d}", .{salt}) catch unreachable);
    const digest = h.finalResult();
    return std.mem.readInt(u64, digest[0..8], .little) & ((@as(u64, 1) << 63) - 1);
}

pub fn mix(v: u64) u64 {
    var x = v;
    x ^= x >> 30;
    x *%= 0xBF58476D1CE4E5B9;
    x ^= x >> 27;
    x *%= 0x94D049BB133111EB;
    return x ^ (x >> 31);
}

/// splitmix64 of (seed, position, id): the bits every keyed sampler starts from.
pub fn keyBits(seed: u64, position: u64, id: u64) u64 {
    var x = mix(seed +% 0x9E3779B97F4A7C15);
    x = mix(x ^ (position *% 0xD1B54A32D192ED03));
    return mix(x ^ id);
}

/// Uniform (0, 1] double from the key's top 53 bits (Python `uniform`).
pub fn uniform(seed: u64, position: u64, id: u64) f64 {
    const x = keyBits(seed, position, id);
    return @as(f64, @floatFromInt(x >> 11)) * 0x1p-53 + 0x1p-54;
}

/// numpy's float64 pairwise sum (`np.add.reduce` on a contiguous row).
pub fn pairwiseSum(a: []const f64) f64 {
    if (a.len < 8) {
        var res: f64 = -0.0;
        for (a) |v| res += v;
        return res;
    }
    if (a.len <= 128) {
        var r: [8]f64 = a[0..8].*;
        var i: usize = 8;
        while (i < a.len - (a.len % 8)) : (i += 8) {
            inline for (0..8) |j| r[j] += a[i + j];
        }
        var res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while (i < a.len) : (i += 1) res += a[i];
        return res;
    }
    var n2 = a.len / 2;
    n2 -= n2 % 8;
    return pairwiseSum(a[0..n2]) + pairwiseSum(a[n2..]);
}

pub const Candidate = struct { value: f64, id: u64 };

fn byValueThenId(_: void, a: Candidate, b: Candidate) bool {
    return a.value > b.value or (a.value == b.value and a.id < b.id);
}

/// The id drawn among candidates `ids` with float64 logits `values` (Python `choose`, and `choose_rows` row by row).
pub fn choose(gpa: Allocator, values: []const f64, ids: []const u64, position: u64, s: Sampling) !u64 {
    if (s.temperature <= 0 and values.len == ids.len and ids.len != 0) {
        greedy: {
            var best: Candidate = .{ .value = values[0], .id = ids[0] };
            if (!std.math.isFinite(best.value)) break :greedy;
            for (values[1..], ids[1..]) |v, id| {
                if (!std.math.isFinite(v)) break :greedy;
                const c = Candidate{ .value = v, .id = id };
                if (byValueThenId({}, c, best)) best = c;
            }
            return best.id;
        }
    }
    const width = ids.len;
    const order = try gpa.alloc(Candidate, width);
    defer gpa.free(order);
    for (order, values, ids) |*o, v, id| o.* = .{ .value = v, .id = id };
    std.mem.sort(Candidate, order, {}, byValueThenId);
    return chooseOrdered(gpa, order, position, s);
}

/// ``choose`` over candidates already in its order (by value, then the lower id at a tie).
pub fn chooseOrdered(gpa: Allocator, order: []const Candidate, position: u64, s: Sampling) !u64 {
    const width = order.len;
    if (s.temperature <= 0) return order[0].id; // temperature 0 is greedy: the first maximum, ties to the lower id
    const k = @max(1, @min(if (s.top_k != 0) @as(usize, s.top_k) else width, width));
    const scaled = try gpa.alloc(f64, k);
    defer gpa.free(scaled);
    const t = @max(s.temperature, 1e-6);
    for (scaled, order[0..k]) |*x, o| x.* = o.value / t;
    var keep = k;
    if (s.top_p > 0.0 and s.top_p < 1.0) {
        const probs = try gpa.alloc(f64, k);
        defer gpa.free(probs);
        for (probs, scaled) |*p, x| p.* = exp(x - scaled[0]);
        const total = pairwiseSum(probs);
        var cum: f64 = 0.0;
        var below: usize = 0;
        for (probs) |*p| {
            p.* /= total;
            cum += p.*;
            if (cum < s.top_p) below += 1;
        }
        keep = @min(keep, below + 1);
    }
    if (s.min_p > 0.0) {
        const floor = scaled[0] + s.minLog();
        var n: usize = 0;
        for (scaled[0..keep]) |x| {
            if (x >= floor) n += 1;
        }
        keep = n;
    }
    var best: usize = 0;
    var best_score = -std.math.inf(f64);
    for (0..keep) |j| {
        const gumbel = -log(-log(uniform(s.seed, position, order[j].id)));
        const score = scaled[j] + gumbel;
        if (j == 0 or score > best_score) {
            best = j;
            best_score = score;
        }
    }
    return order[best].id;
}

test "seeds keep 63 bits and uniforms fall in (0, 1]" {
    try std.testing.expect(seedFor(&.{ 1, 2, 3 }, 0) < (@as(u64, 1) << 63));
    const u = uniform(7, 11, 13);
    try std.testing.expect(u > 0.0 and u <= 1.0);
}

test "temperature 0 draws greedily, an exact tie going to the lower id" {
    const a = std.testing.allocator;
    const values = [_]f64{ 1.5, 3.0, -2.0, 3.0 };
    const ids = [_]u64{ 10, 35, 50, 93 };
    for (0..64) |position| try std.testing.expectEqual(@as(u64, 35), try choose(a, &values, &ids, position, .{ .seed = 7, .temperature = 0 }));
}

test "finite greedy draw needs no allocation" {
    var fba = std.heap.FixedBufferAllocator.init(&.{});
    const values = [_]f64{ 1.5, 3.0, -2.0, 3.0 };
    const ids = [_]u64{ 10, 35, 50, 93 };
    try std.testing.expectEqual(@as(u64, 35), try choose(fba.allocator(), &values, &ids, 0, .{ .seed = 7, .temperature = 0 }));
}

// Frozen copy of the pre-fastpath choose body (ad247b3da): the test oracle must
// not route through the guarded scan it checks against.
fn legacyChoose(gpa: Allocator, values: []const f64, ids: []const u64, position: u64, s: Sampling) !u64 {
    const width = ids.len;
    const order = try gpa.alloc(Candidate, width);
    defer gpa.free(order);
    for (order, values, ids) |*o, v, id| o.* = .{ .value = v, .id = id };
    std.mem.sort(Candidate, order, {}, byValueThenId);
    if (s.temperature <= 0) return order[0].id;
    const k = @max(1, @min(if (s.top_k != 0) @as(usize, s.top_k) else width, width));
    const scaled = try gpa.alloc(f64, k);
    defer gpa.free(scaled);
    const t = @max(s.temperature, 1e-6);
    for (scaled, order[0..k]) |*x, o| x.* = o.value / t;
    var keep = k;
    if (s.top_p > 0.0 and s.top_p < 1.0) {
        const probs = try gpa.alloc(f64, k);
        defer gpa.free(probs);
        for (probs, scaled) |*p, x| p.* = exp(x - scaled[0]);
        const total = pairwiseSum(probs);
        var cum: f64 = 0.0;
        var below: usize = 0;
        for (probs) |*p| {
            p.* /= total;
            cum += p.*;
            if (cum < s.top_p) below += 1;
        }
        keep = @min(keep, below + 1);
    }
    if (s.min_p > 0.0) {
        const floor = scaled[0] + s.minLog();
        var n: usize = 0;
        for (scaled[0..keep]) |x| {
            if (x >= floor) n += 1;
        }
        keep = n;
    }
    var best: usize = 0;
    var best_score = -std.math.inf(f64);
    for (0..keep) |j| {
        const gumbel = -log(-log(uniform(s.seed, position, order[j].id)));
        const score = scaled[j] + gumbel;
        if (j == 0 or score > best_score) {
            best = j;
            best_score = score;
        }
    }
    return order[best].id;
}

test "finite greedy matches the frozen sorted oracle on arbitrary ids, ties, signed zeros, negatives and singletons" {
    const a = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0xC0FFEE);
    const rand = prng.random();
    const rows = [_]struct { values: []const f64, ids: []const u64 }{
        .{ .values = &.{ 1.5, 3.0, -2.0, 3.0 }, .ids = &.{ 10, 35, 50, 93 } },
        .{ .values = &.{ 3.0, 3.0, 3.0 }, .ids = &.{ 0xFFFF_FFFF_FFFF_0001, 7, 0xFFFF_FFFF_FFFF_0000 } },
        .{ .values = &.{ 0.0, -0.0 }, .ids = &.{ 900, 12 } },
        .{ .values = &.{ -0.0, 0.0 }, .ids = &.{ 900, 12 } },
        .{ .values = &.{ -0.0, 0.0 }, .ids = &.{ 12, 900 } },
        .{ .values = &.{ -1.5, -2.5, -0.25 }, .ids = &.{ 5, 1 << 40, 3 } },
        .{ .values = &.{42.0}, .ids = &.{0xDEAD_BEEF_CAFE_F00D} },
        .{ .values = &.{ -0.0, -0.0, -0.0 }, .ids = &.{ 77, 3, 1 << 41 } },
    };
    for (rows) |r| {
        for ([_]f64{ 0.0, -0.0, -1.0, -1e300 }) |temp| {
            const s = Sampling{ .seed = 0x5EED, .temperature = temp };
            try std.testing.expectEqual(try legacyChoose(a, r.values, r.ids, 9, s), try choose(a, r.values, r.ids, 9, s));
        }
    }
    // Randomized finite rows with arbitrary u64 ids, shuffled and with duplicate ids.
    var values: [64]f64 = undefined;
    var ids: [64]u64 = undefined;
    for (0..200) |case| {
        const width = 1 + rand.uintLessThan(usize, 64);
        for (values[0..width]) |*v| {
            v.* = switch (rand.uintLessThan(u8, 6)) {
                0 => -rand.float(f64) * 1e300,
                1 => rand.float(f64) * 1e300,
                2 => 0.0,
                3 => -0.0,
                4 => @as(f64, @floatFromInt(rand.int(i32))) * std.math.floatMin(f64),
                else => rand.float(f64),
            };
        }
        for (ids[0..width]) |*id| id.* = switch (rand.uintLessThan(u8, 4)) {
            0 => rand.int(u64),
            1 => rand.uintLessThan(u64, 1 << 17),
            2 => 0xFFFF_FFFF_FFFF_FFFF,
            else => 1 << 40,
        };
        rand.shuffle(f64, values[0..width]);
        rand.shuffle(u64, ids[0..width]);
        const s = Sampling{ .seed = rand.int(u64), .temperature = if (case % 2 == 0) 0.0 else -0.5 };
        try std.testing.expectEqual(try legacyChoose(a, values[0..width], ids[0..width], rand.int(u64), s), try choose(a, values[0..width], ids[0..width], rand.int(u64), s));
    }
}

test "finite greedy matches the frozen sorted oracle on f64 extremes and subnormals" {
    const a = std.testing.allocator;
    const rows = [_]struct { values: []const f64, ids: []const u64 }{
        .{ .values = &.{ std.math.floatMax(f64), -std.math.floatMax(f64), std.math.floatMin(f64), -std.math.floatMin(f64) }, .ids = &.{ 4, 1 << 63, 2, 1 } },
        .{ .values = &.{ std.math.floatMin(f64), -std.math.floatMin(f64), 5e-324, -5e-324, 2.5e-324 }, .ids = &.{ 8, 9, 10, 11, 12 } },
        .{ .values = &.{ 1.0, 1.0 + std.math.floatEps(f64), 1.0 + 2 * std.math.floatEps(f64) }, .ids = &.{ 100, 5, 7 } },
        .{ .values = &.{ -std.math.floatMax(f64), -1.7976931348623155e308 }, .ids = &.{ 1, 2 } },
        .{ .values = &.{ 0.0, 5e-324 }, .ids = &.{ 3, 1 } },
    };
    for (rows) |r| {
        for ([_]f64{ 0.0, -0.0, -2.5 }) |temp| {
            const s = Sampling{ .seed = 3, .temperature = temp };
            try std.testing.expectEqual(try legacyChoose(a, r.values, r.ids, 0, s), try choose(a, r.values, r.ids, 0, s));
        }
    }
}

test "bf16-widened bounded seeded rows match the frozen sorted oracle" {
    const a = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0xB16);
    const rand = prng.random();
    var values: [40]f64 = undefined;
    var ids: [40]u64 = undefined;
    for (0..100) |_| {
        const width = 1 + rand.uintLessThan(usize, 40);
        for (values[0..width]) |*v| {
            const bits: u16 = rand.int(u16);
            v.* = @as(f64, @as(f32, @bitCast(@as(u32, bits) << 16)));
        }
        for (ids[0..width]) |*id| id.* = rand.int(u64);
        const s = Sampling{ .seed = rand.int(u64), .temperature = 0.0 };
        try std.testing.expectEqual(try legacyChoose(a, values[0..width], ids[0..width], rand.int(u64), s), try choose(a, values[0..width], ids[0..width], rand.int(u64), s));
    }
}

test "nonfinite rows fall through to the frozen sorted oracle path and still need allocation" {
    const a = std.testing.allocator;
    const nan = std.math.nan(f64);
    const inf = std.math.inf(f64);
    const rows = [_]struct { values: []const f64, ids: []const u64 }{
        .{ .values = &.{ nan, 1.0, 2.0 }, .ids = &.{ 5, 6, 7 } },
        .{ .values = &.{ 1.0, nan, 2.0 }, .ids = &.{ 5, 6, 7 } },
        .{ .values = &.{ 1.0, 2.0, nan }, .ids = &.{ 5, 6, 7 } },
        .{ .values = &.{ inf, 1.0 }, .ids = &.{ 5, 6 } },
        .{ .values = &.{ 1.0, -inf }, .ids = &.{ 5, 6 } },
        .{ .values = &.{ nan, inf, -inf, 0.0 }, .ids = &.{ 1, 2, 3, 4 } },
        .{ .values = &.{ nan, nan, nan }, .ids = &.{ 9, 8, 7 } },
    };
    for (rows) |r| {
        const s = Sampling{ .seed = 11, .temperature = 0.0 };
        try std.testing.expectEqual(try legacyChoose(a, r.values, r.ids, 4, s), try choose(a, r.values, r.ids, 4, s));
    }
    // The fallback allocates: a zero-capacity allocator must still fail with OutOfMemory.
    var fba = std.heap.FixedBufferAllocator.init(&.{});
    try std.testing.expectError(error.OutOfMemory, choose(fba.allocator(), &.{ 1.0, inf }, &.{ 1, 2 }, 0, .{ .seed = 0, .temperature = 0 }));
}

test "sampled and nonfinite draws keep the old allocation behavior" {
    var fba = std.heap.FixedBufferAllocator.init(&.{});
    const fa = fba.allocator();
    const values = [_]f64{ 1.5, 3.0, -2.0, 3.0 };
    const ids = [_]u64{ 10, 35, 50, 93 };
    try std.testing.expectError(error.OutOfMemory, choose(fa, &values, &ids, 0, .{ .seed = 7, .temperature = 1.0 }));
    try std.testing.expectError(error.OutOfMemory, choose(fa, &values, &ids, 0, .{ .seed = 7, .temperature = std.math.nan(f64) }));
    try std.testing.expectError(error.OutOfMemory, choose(fa, &values, ids[0..3], 0, .{ .seed = 7, .temperature = 0.0 }));
}

test "sampled policies match the frozen sorted oracle, nan and inf temperatures included" {
    const a = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x5A7D);
    const rand = prng.random();
    var values: [72]f64 = undefined;
    var ids: [72]u64 = undefined;
    for (0..120) |_| {
        const width = 1 + rand.uintLessThan(usize, 72);
        for (values[0..width]) |*v| v.* = rand.float(f64) * 6.0 - 3.0;
        for (ids[0..width]) |*id| id.* = rand.int(u64);
        const top_ks = [5]u32{ 0, 1, 5, 20, 64 };
        const top_ps = [5]f64{ 0.0, 0.5, 0.9, 0.95, 1.0 };
        const min_ps = [4]f64{ 0.0, 0.0, 0.05, 0.2 };
        const s = Sampling{
            .seed = rand.int(u64),
            .temperature = switch (rand.uintLessThan(u8, 6)) {
                0 => std.math.nan(f64),
                1 => std.math.inf(f64),
                else => 0.05 + rand.float(f64) * 2.0,
            },
            .top_k = top_ks[rand.uintLessThan(usize, 5)],
            .top_p = top_ps[rand.uintLessThan(usize, 5)],
            .min_p = min_ps[rand.uintLessThan(usize, 4)],
        };
        const position = rand.int(u64);
        try std.testing.expectEqual(try legacyChoose(a, values[0..width], ids[0..width], position, s), try choose(a, values[0..width], ids[0..width], position, s));
    }
}
