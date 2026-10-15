const std = @import("std");
const modes = @import("cache_modes.zig");

pub const Prompt = struct { tokens: []const u32, spans: []const modes.Span = &.{}, starts: []const u32 = &.{} };

pub fn pending(consumer: Prompt, producer: Prompt, marks: []const u32, after: u32, lookahead: u32, grid: u32) ?u32 {
    var best: ?u32 = null;
    for (marks) |at| {
        if (at <= after or at >= consumer.tokens.len) continue;
        const through = @as(u64, at) + lookahead;
        if (through > consumer.tokens.len or through > producer.tokens.len) continue;
        if (consumer.starts.len > 0) {
            if (std.mem.indexOfScalar(u32, consumer.starts, at) == null) continue;
        } else if (grid == 0 or at % grid != 0) continue;
        if (!modes.equal(consumer.spans, producer.spans, at)) continue;
        const n: usize = @intCast(through);
        if (!std.mem.eql(u32, consumer.tokens[0..n], producer.tokens[0..n])) continue;
        best = @max(best orelse 0, at);
    }
    return best;
}

test "shared prefix waits only for a future reusable mark with the same row arithmetic and lookahead" {
    const a = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    const b = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 9, 9 };
    try std.testing.expectEqual(@as(?u32, 4), pending(.{ .tokens = &b }, .{ .tokens = &a }, &.{ 4, 8 }, 0, 1, 4));
    try std.testing.expectEqual(@as(?u32, null), pending(.{ .tokens = &b }, .{ .tokens = &a }, &.{ 4, 8 }, 4, 1, 4));
    try std.testing.expectEqual(@as(?u32, 8), pending(.{ .tokens = &a }, .{ .tokens = &a }, &.{ 4, 8 }, 0, 1, 4));
    try std.testing.expectEqual(@as(?u32, null), pending(.{ .tokens = &a, .spans = &.{.{ 0, 2 }} }, .{ .tokens = &a }, &.{4}, 0, 1, 4));
    try std.testing.expectEqual(@as(?u32, 4), pending(.{ .tokens = &a, .spans = &.{.{ 7, 8 }} }, .{ .tokens = &a }, &.{4}, 0, 1, 4));
    try std.testing.expectEqual(@as(?u32, null), pending(.{ .tokens = &a, .starts = &.{3} }, .{ .tokens = &a }, &.{4}, 0, 1, 4));
    try std.testing.expectEqual(@as(?u32, null), pending(.{ .tokens = a[0..4] }, .{ .tokens = &a }, &.{4}, 0, 1, 4));
}
