//! Where a prompt pass cuts its chunks: a row's bits are the same wherever the cuts fall, so states keep at any row.
const std = @import("std");

pub const shared_rows: usize = 512;

pub fn sharedStep(capacity: usize, lanes: usize) usize {
    return if (lanes > 1) @min(capacity, shared_rows) else capacity;
}

/// The end of the chunk starting at `at`: `step` rows on, the prompt's end, or the first of `cuts` before them.
pub fn end(at: usize, len: usize, step: usize, cuts: []const u32) usize {
    var e = @min(len, at + step);
    for (cuts) |m| if (m > at and m < e) {
        e = m;
    };
    return e;
}

test "chunks run `step` rows from where a pass starts, and a cut ends one" {
    const step = 2048;
    var ends: [4]usize = undefined;
    var at: usize = 0;
    for (&ends) |*c| {
        c.* = end(at, 5000, step, &.{ 4990, 1500 });
        at = c.*;
    }
    try std.testing.expectEqualSlices(usize, &.{ 1500, 3548, 4990, 5000 }, &ends);
    try std.testing.expectEqual(@as(usize, 4148), end(2100, 5000, step, &.{})); // a resumed pass: 2048 rows from its start
    try std.testing.expectEqual(@as(usize, 2048), end(0, 5000, step, &.{2048})); // a cut at a chunk's end adds none
    try std.testing.expectEqual(@as(usize, 1000), end(0, 1000, step, &.{ 0, 1000 })); // cuts at the start or end cut nothing
}

test "shared prompt turns stay bounded and retain fallback and exact history cuts" {
    try std.testing.expectEqual(@as(usize, 2048), sharedStep(2048, 1));
    try std.testing.expectEqual(@as(usize, 512), sharedStep(2048, 2));
    try std.testing.expectEqual(@as(usize, 512), sharedStep(2048, 8));
    try std.testing.expectEqual(@as(usize, 256), sharedStep(256, 2));
    var at: usize = 0;
    var fallback = false;
    var history = false;
    while (at < 6250) {
        const next = end(at, 6250, sharedStep(2048, 2), &.{ 4096, 6244 });
        try std.testing.expect(next > at and next - at <= 512);
        fallback = fallback or next == 4096;
        history = history or next == 6244;
        at = next;
    }
    try std.testing.expect(fallback and history);
    try std.testing.expectEqual(@as(usize, 6250), at);
    try std.testing.expectEqual(@as(usize, 6244), end(6144, 6250, shared_rows, &.{ 4096, 6244 }));
}
