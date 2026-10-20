//! Prompt split geometry adapted from TensorFold hcsplit.py and Mia's cuda_prefill_split.zig.
//! The active engine owns the streams and buffers; rank ownership is one contiguous padded half.
const std = @import("std");
pub const max_pieces = 8;
pub const Range = struct { lo: usize, hi: usize };
pub const Piece = struct { off: usize, rows: usize };
pub const Plan = struct {
    rows: usize,
    padded: usize,
    half: usize,
    mine: usize,
    peer: usize,
    count: usize,
    pieces: [max_pieces]Piece,

    pub fn init(rows: usize, capacity: usize, rank: usize, count: usize) !Plan {
        if (rows == 0 or rank >= 2 or count == 0 or count > max_pieces) return error.BadPromptSplit;
        const padded = std.mem.alignForward(usize, rows, 2);
        if (padded > capacity) return error.BadPromptSplit;
        const half = padded / 2;
        var size = (half + count - 1) / count;
        if (size > 64) size = @min(half, std.mem.alignForward(usize, size, 64));
        var plan: Plan = .{ .rows = rows, .padded = padded, .half = half, .mine = rank * half, .peer = (1 - rank) * half, .count = 0, .pieces = undefined };
        var off: usize = 0;
        while (off < half) : (off += size) {
            plan.pieces[plan.count] = .{ .off = off, .rows = @min(size, half - off) };
            plan.count += 1;
        }
        return plan;
    }

    pub fn owned(p: Plan) Range {
        return .{ .lo = p.mine, .hi = @min(p.mine + p.half, p.rows) };
    }
};

pub fn applies(level: u8, world: usize, rows: usize, capacity: usize) bool {
    return level != 0 and world == 2 and rows >= 64 and std.mem.alignForward(usize, rows, 2) <= capacity;
}

test "production prompt split owns a contiguous half and pieces that half on 64-row tiles" {
    const p = try Plan.init(2048, 2048, 0, 2);
    try std.testing.expectEqual(@as(usize, 1024), p.half);
    try std.testing.expectEqual(@as(usize, 2), p.count);
    try std.testing.expectEqual(Piece{ .off = 0, .rows = 512 }, p.pieces[0]);
    try std.testing.expectEqual(Piece{ .off = 512, .rows = 512 }, p.pieces[1]);
    const other = try Plan.init(2048, 2048, 1, 2);
    try std.testing.expectEqual(Range{ .lo = 1024, .hi = 2048 }, other.owned());
}

test "odd and short tails cover every real row once and only pad within the buffer" {
    for ([_]usize{ 64, 65, 127, 128, 129, 255, 2047, 2048 }) |rows| {
        for (1..max_pieces + 1) |count| {
            var covered: [2048]u8 = @splat(0);
            for (0..2) |rank| {
                const p = try Plan.init(rows, 2048, rank, count);
                var end: usize = 0;
                for (p.pieces[0..p.count]) |piece| {
                    try std.testing.expectEqual(end, piece.off);
                    end += piece.rows;
                    const lo = p.mine + piece.off;
                    for (lo..@min(lo + piece.rows, rows)) |row| covered[row] += 1;
                }
                try std.testing.expectEqual(p.half, end);
            }
            for (covered[0..rows]) |n| try std.testing.expectEqual(@as(u8, 1), n);
        }
    }
    try std.testing.expect(applies(2, 2, 65, 66));
    try std.testing.expect(!applies(2, 2, 65, 65));
    try std.testing.expect(!applies(0, 2, 2048, 2048));
    try std.testing.expect(!applies(2, 3, 2048, 2048));
    try std.testing.expect(!applies(2, 2, 63, 2048));
    try std.testing.expectError(error.BadPromptSplit, Plan.init(65, 65, 0, 2));
    try std.testing.expectError(error.BadPromptSplit, Plan.init(64, 64, 2, 2));
    try std.testing.expectError(error.BadPromptSplit, Plan.init(64, 64, 0, 0));
}
