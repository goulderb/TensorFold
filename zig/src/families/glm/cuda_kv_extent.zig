const std = @import("std");

pub const alignment = 2048;
pub const max_slots = 64;
pub const Region = struct { base: usize, rows: usize };

pub const Pool = struct {
    rows: usize,
    parts: usize,
    slots: [max_slots]?Region = @splat(null),

    pub fn init(rows: usize, parts: usize) !Pool {
        if (rows == 0 or (parts != 1 and parts != 2) or rows % (parts * alignment) != 0) return error.BadKvPool;
        return .{ .rows = rows, .parts = parts };
    }

    pub fn partRows(p: *const Pool) usize {
        return p.rows / p.parts;
    }

    pub fn owner(p: *const Pool, r: Region) usize {
        return r.base / p.partRows();
    }

    pub fn plan(p: *const Pool, slot: usize, tokens: usize) !Region {
        if (slot >= max_slots or tokens == 0) return error.BadKvRequest;
        const n = try round(tokens);
        if (n > p.partRows()) return error.PromptTooLong;
        if (p.slots[slot]) |r| if (r.rows >= n) return r;
        var best: ?Region = null;
        var best_live: usize = std.math.maxInt(usize);
        var best_free: usize = 0;
        for (0..p.parts) |part| {
            var live: usize = 0;
            var free = p.partRows();
            for (p.slots, 0..) |maybe, i| if (i != slot) {
                if (maybe) |r| if (p.owner(r) == part) {
                    live += 1;
                    free -= r.rows;
                };
            };
            var at = part * p.partRows();
            const end = at + p.partRows();
            while (at <= end - n) {
                var next = at;
                for (p.slots, 0..) |maybe, i| if (i != slot) {
                    if (maybe) |r| if (r.base < at + n and at < r.base + r.rows) {
                        next = @max(next, r.base + r.rows);
                    };
                };
                if (next == at) {
                    if (best == null or live < best_live or (live == best_live and free > best_free)) {
                        best = .{ .base = at, .rows = n };
                        best_live = live;
                        best_free = free;
                    }
                    break;
                }
                at = next;
            }
        }
        return best orelse error.LaneMemory;
    }

    /// Apply the leader's exact placement; this also checks a follower's mirrored free ranges.
    pub fn apply(p: *Pool, slot: usize, r: Region) !void {
        if (slot >= max_slots or r.rows == 0 or r.rows % alignment != 0 or r.base % alignment != 0 or r.rows > p.partRows() or r.base > p.rows - r.rows) return error.BadKvPlacement;
        if (p.owner(r) != (r.base + r.rows - 1) / p.partRows()) return error.BadKvPlacement;
        for (p.slots, 0..) |maybe, i| if (i != slot) {
            if (maybe) |other| if (other.base < r.base + r.rows and r.base < other.base + other.rows) return error.BadKvPlacement;
        };
        p.slots[slot] = r;
    }

    pub fn release(p: *Pool, slot: usize) void {
        if (slot < max_slots) p.slots[slot] = null;
    }
};

pub fn requestRows(prompt: usize, output: usize, lookahead: usize, capacity: usize) !usize {
    const total = try std.math.add(usize, try std.math.add(usize, prompt, output), lookahead);
    if (prompt == 0 or total >= capacity) return error.PromptTooLong;
    return round(total);
}

fn round(n: usize) !usize {
    return (try std.math.add(usize, n, alignment - 1)) & ~@as(usize, alignment - 1);
}

test "placement matches the pinned donor across owners, release and uneven requests" {
    var p = try Pool.init(32768, 2);
    const inputs = [_]usize{ 3000, 7000, 2048, 5000 };
    const want = [_]Region{ .{ .base = 0, .rows = 4096 }, .{ .base = 16384, .rows = 8192 }, .{ .base = 4096, .rows = 2048 }, .{ .base = 24576, .rows = 6144 } };
    for (inputs, want, 0..) |n, expected, slot| {
        const got = try p.plan(slot, n);
        try std.testing.expectEqual(expected, got);
        try p.apply(slot, got);
    }
    p.release(1);
    const next = try p.plan(4, 10000);
    try std.testing.expectEqual(Region{ .base = 6144, .rows = 10240 }, next);
    try p.apply(4, next);
    try std.testing.expectError(error.LaneMemory, p.plan(5, 9000));
    try std.testing.expectError(error.PromptTooLong, p.plan(5, 16385));
    try std.testing.expectEqual(next, try p.plan(4, 10000));
}

test "placement refuses overlap and owner crossing without changing the pool" {
    var p = try Pool.init(16384, 2);
    try p.apply(0, .{ .base = 0, .rows = 4096 });
    try std.testing.expectError(error.BadKvPlacement, p.apply(1, .{ .base = 2048, .rows = 4096 }));
    try std.testing.expectError(error.BadKvPlacement, p.apply(1, .{ .base = 6144, .rows = 4096 }));
    try std.testing.expectEqual(@as(?Region, null), p.slots[1]);
    try p.apply(1, .{ .base = 8192, .rows = 4096 });
    p.release(0);
    try std.testing.expectEqual(Region{ .base = 0, .rows = 8192 }, try p.plan(2, 8000));
    try std.testing.expectError(error.BadKvPool, Pool.init(32769, 2));
    try std.testing.expectEqual(@as(usize, 4096), try requestRows(2000, 100, 8, 8192));
    try std.testing.expectError(error.PromptTooLong, requestRows(8000, 200, 8, 8192));
    try std.testing.expectError(error.Overflow, requestRows(std.math.maxInt(usize), 1, 8, 8192));
}
