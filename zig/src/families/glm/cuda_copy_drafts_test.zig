const std = @import("std");
const copy = @import("cuda_copy_drafts.zig");
const Settings = copy.Settings;
const CopyDrafts = copy.CopyDrafts;
const t = std.testing;

fn shortSettings(most: usize) Settings {
    return .{ .enabled = true, .match = 2, .most = most };
}

fn longContext() [36]u32 {
    var ctx: [36]u32 = undefined;
    for (ctx[0..28], 0..) |*token, i| token.* = @intCast(i);
    for (ctx[28..], 0..) |*token, i| token.* = @intCast(i);
    return ctx;
}

test "copy settings retain opt-in defaults and validate production limits" {
    const defaults: Settings = .{};
    try t.expect(!defaults.enabled);
    try defaults.validate();
    const production = Settings.production();
    try production.validate();
    try t.expect(production.enabled);
    try t.expectEqual(@as(usize, 8), production.match);
    try t.expectEqual(@as(usize, 15), production.most);
    try t.expectEqual(@as(usize, 16), production.reply_match);
    for ([_]usize{ 0, 1, 65 }) |n| try t.expectError(error.BadCopyMatch, (Settings{ .match = n }).validate());
    for ([_]usize{ 0, 16 }) |n| try t.expectError(error.BadCopyMax, (Settings{ .most = n }).validate());
    for ([_]usize{ 1, 8, 65 }) |n| try t.expectError(error.BadCopyReplyMatch, (Settings{ .reply_match = n }).validate());
    try t.expectError(error.BadCopyMissMax, (Settings{ .most = 4, .miss_most = 5 }).validate());
    const ctx = longContext();
    var off = try CopyDrafts.init(defaults, 28);
    defer off.deinit();
    try t.expectEqual(@as(usize, 0), (try off.propose(&ctx, 15)).len);
}

test "copy proposals exclude the suffix but allow overlap and a single continuation" {
    var s = try CopyDrafts.init(shortSettings(15), 10);
    defer s.deinit();
    try t.expectEqual(@as(usize, 0), (try s.propose(&.{ 1, 2 }, 15)).len);
    try t.expectEqual(@as(usize, 0), (try s.propose(&.{ 0, 1, 2 }, 15)).len);
    try t.expectEqualSlices(u32, &.{1}, try s.propose(&.{ 1, 1, 1 }, 15));
    const p = s.proposer();
    try t.expect(p.vtable.priced);
    try t.expectEqual(@as(i64, 2), p.lastMatch());
    try t.expectEqualSlices(u32, &.{1}, try p.propose(&.{ 1, 1, 1 }, 15));
}

test "copy chooses the latest full occurrence instead of a longer suffix match" {
    var s = try CopyDrafts.init(shortSettings(2), 10);
    defer s.deinit();
    const ctx = [_]u32{ 1, 2, 10, 11, 1, 2, 20, 21, 1, 2 };
    try t.expectEqualSlices(u32, &.{ 20, 21 }, try s.propose(&ctx, 2));
}

test "copy chooses an older full continuation before a newer short one" {
    var s = try CopyDrafts.init(shortSettings(4), 9);
    defer s.deinit();
    const ctx = [_]u32{ 1, 2, 10, 11, 12, 1, 2, 1, 2 };
    try t.expectEqualSlices(u32, &.{ 10, 11, 12, 1 }, try s.propose(&ctx, 4));
}

test "copy falls back to the earliest longest available continuation" {
    var s = try CopyDrafts.init(shortSettings(15), 10);
    defer s.deinit();
    const ctx = [_]u32{ 1, 2, 10, 11, 1, 2, 20, 21, 1, 2 };
    try t.expectEqualSlices(u32, ctx[2..], try s.propose(&ctx, 15));
}

test "reply starts need sixteen matching tokens but prompt starts may cross the boundary" {
    var ctx: [40]u32 = undefined;
    for (0..8) |i| {
        ctx[i] = @intCast(100 + i);
        ctx[8 + i] = @intCast(i);
        ctx[16 + i] = @intCast(200 + i);
        ctx[24 + i] = @intCast(100 + i);
        ctx[32 + i] = @intCast(i);
    }
    var reply = try CopyDrafts.init(Settings.production(), 8);
    defer reply.deinit();
    try t.expectEqualSlices(u32, ctx[16..31], try reply.propose(&ctx, 15));
    ctx[24] = 999;
    try t.expectEqual(@as(usize, 0), (try reply.propose(&ctx, 15)).len);
    var crossing = try CopyDrafts.init(Settings.production(), 9);
    defer crossing.deinit();
    try t.expectEqualSlices(u32, ctx[16..31], try crossing.propose(&ctx, 15));
    var lacking_prefix = try CopyDrafts.init(Settings.production(), 0);
    defer lacking_prefix.deinit();
    const short = longContext();
    try t.expectEqual(@as(usize, 0), (try lacking_prefix.propose(&short, 15)).len);
}

test "copy room reserves are authoritative including zero and one token" {
    const ctx = longContext();
    var s = try CopyDrafts.init(Settings.production(), 28);
    defer s.deinit();
    for ([_]i64{ -1, 0, 1, 14, 15, 99 }) |room| {
        const n: usize = @intCast(@max(0, @min(room, 15)));
        try t.expectEqualSlices(u32, ctx[8 .. 8 + n], try s.propose(&ctx, room));
        try t.expectEqual(n, s.original_copied);
    }
}

test "only standalone copies pad windows wider than eight rows when room permits" {
    const ctx = longContext();
    for ([_]usize{ 7, 8, 14, 15 }) |most| {
        var settings = Settings.production();
        settings.most = most;
        var standalone = try CopyDrafts.initStandalone(settings, 28, &.{ 16, 8, 16, 32 });
        defer standalone.deinit();
        const got = try standalone.propose(&ctx, 15);
        const padded = if (most == 7) most else 15;
        try t.expectEqual(padded, got.len);
        try t.expectEqual(most, standalone.original_copied);
        try t.expectEqualSlices(u32, ctx[8 .. 8 + most], got[0..most]);
        for (got[most..]) |token| try t.expectEqual(ctx[8 + most - 1], token);
        var lane = try CopyDrafts.init(settings, 28);
        defer lane.deinit();
        try t.expectEqual(most, (try lane.proposer().propose(&ctx, 15)).len);
    }
    var settings = Settings.production();
    settings.most = 8;
    var bounded = try CopyDrafts.initStandalone(settings, 28, &.{16});
    defer bounded.deinit();
    for ([_]i64{ 8, 14 }) |room| try t.expectEqual(@as(usize, 8), (try bounded.propose(&ctx, room)).len);
    try t.expectEqual(@as(usize, 15), (try bounded.propose(&ctx, 15)).len);
    var next = try CopyDrafts.initStandalone(settings, 28, &.{ 12, 16 });
    defer next.deinit();
    try t.expectEqual(@as(usize, 11), (try next.propose(&ctx, 15)).len);
}

test "miss accounting excludes padding but includes shared-budget truncation" {
    const ctx = longContext();
    var settings = Settings.production();
    settings.most = 8;
    settings.miss_most = 2;
    var s = try CopyDrafts.initStandalone(settings, 28, &.{16});
    defer s.deinit();
    const p = s.proposer();
    try t.expectEqual(@as(usize, 15), (try p.propose(&ctx, 15)).len);
    p.observe(15, 8);
    try t.expect(!s.missed);
    try t.expectEqual(@as(usize, 15), (try p.propose(&ctx, 15)).len);
    p.observe(2, 2);
    try t.expect(s.missed);
    try t.expectEqual(@as(usize, 2), (try p.propose(&ctx, 15)).len);
    p.observe(2, 2);
    try t.expect(!s.missed);
    try t.expectEqual(@as(usize, 15), (try p.propose(&ctx, 15)).len);
}

test "committed context settles zero-draft trims and the original unpadded copy count" {
    const ctx = longContext();
    var settings = Settings.production();
    settings.most = 8;
    settings.miss_most = 2;
    var trimmed = try CopyDrafts.init(settings, 28);
    defer trimmed.deinit();
    try t.expectEqual(@as(usize, 8), (try trimmed.propose(&ctx, 15)).len);
    var one: [37]u32 = undefined;
    @memcpy(one[0..36], &ctx);
    one[36] = 8;
    // A shared trim to the pending row alone never calls Proposer.observe.
    try t.expectEqualSlices(u32, &.{ 9, 10 }, try trimmed.propose(&one, 15));
    try t.expect(trimmed.missed);
    var all: [45]u32 = undefined;
    @memcpy(all[0..36], &ctx);
    for (all[36..], 8..) |*token, i| token.* = @intCast(i);
    var padded = try CopyDrafts.initStandalone(settings, 28, &.{16});
    defer padded.deinit();
    try t.expectEqual(@as(usize, 15), (try padded.propose(&ctx, 15)).len);
    // Grammar/other committed continuation can be longer than the own-draft count in observe.
    padded.observe(2, 2);
    try t.expect(padded.missed);
    try t.expectEqual(@as(usize, 15), (try padded.propose(&all, 15)).len);
    try t.expect(!padded.missed);
}
