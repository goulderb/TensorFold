//! A clean stop's flush: one pass, each kept state written once with the slot waited for, its outcome counted once.
const std = @import("std");
const api = @import("engine_api");
const ns = @import("nemotron_spill.zig");
const t = @import("nemotron_spill_test.zig");
const pc = api.prompt_cache;
const spill = pc.spill;

const gpa = std.testing.allocator;
const io = std.testing.io;
const rules: pc.Rules = .{ .lookahead = 1, .planned = true, .grid = t.grid, .min_prompt = 0, .min_gap = 1 };

/// A disk and a store holding states at `ats`, flushed with `flush_s`; the close's outcome.
fn stopWith(root: *const t.Root, ats: []const u32, flush_s: f64, prompt: []const u32) !ns.Flushed {
    const l = t.layout(10);
    var settings = try root.settings();
    settings.flush_s = flush_s;
    const d = try ns.Disk.open(gpa, io, gpa, t.device(), l, settings, t.identity_a, 1 << 30);
    t.configure(d);
    d.tier.chunk = 5;
    d.tier.write_file = t.Hold.write;
    var target = t.targetFor(d);
    var store = pc.Store.init(gpa, .{ .ptr = &target, .vtable = &t.Family.vtable }, rules, d.reservation + (l.kvBytes(l.max_len) + l.mamba) * ats.len);
    defer store.deinit();
    d.attach(&target, &store);
    for (ats) |at| try std.testing.expect(store.keep(prompt, at, null, &.{}, &.{}));
    return d.close(&store);
}

fn storedAfter(root: *const t.Root, prompt: []const u32, ats: []const u32) !usize {
    const d = try ns.Disk.open(gpa, io, gpa, t.device(), t.layout(10), try root.settings(), t.identity_a, 1 << 30);
    defer d.abandon(null);
    var n: usize = 0;
    for (ats) |at| {
        if (d.tier.stored(spill.Name.of(prompt[0 .. at + 1], at), prompt[0 .. at + 1])) n += 1;
    }
    return n;
}

test "a clean stop writes every kept state once, waiting for the slot instead of counting busy skips" {
    var root = try t.Root.make();
    defer root.deinit();
    var tok: [32]u32 = undefined;
    const prompt = t.tokensFor(&tok, 20);
    const ats = [_]u32{ 4, 8, 12, 16 };
    const f = try stopWith(&root, &ats, 5, prompt);
    try std.testing.expectEqual(@as(u64, 4), f.written);
    try std.testing.expectEqual(@as(u64, 0), f.left);
    try std.testing.expectEqual(@as(u64, 0), f.lost);
    try std.testing.expectEqual(@as(u64, 0), f.failed);
    try std.testing.expectEqual(@as(u64, 0), f.counts.skipped_busy); // the old retry passes counted 4 * 3 / 2 here
    try std.testing.expectEqual(@as(u64, 4), f.counts.submitted);
    try std.testing.expectEqual(@as(usize, 4), try storedAfter(&root, prompt, &ats));
}

test "a stop whose flush time is spent writes nothing new and counts each kept state left once" {
    var root = try t.Root.make();
    defer root.deinit();
    var tok: [32]u32 = undefined;
    const prompt = t.tokensFor(&tok, 20);
    const ats = [_]u32{ 4, 8, 12 };
    const f = try stopWith(&root, &ats, 0, prompt);
    try std.testing.expectEqual(@as(u64, 0), f.written);
    try std.testing.expectEqual(@as(u64, 3), f.left);
    try std.testing.expectEqual(@as(u64, 0), f.counts.skipped_busy);
    try std.testing.expectEqual(@as(usize, 0), try storedAfter(&root, prompt, &ats));
}

test "a write the tier gives up stops the flush: one capture, one loss, the rest left once each" {
    var root = try t.Root.make();
    defer root.deinit();
    var tok: [32]u32 = undefined;
    const prompt = t.tokensFor(&tok, 20);
    const ats = [_]u32{ 4, 8, 12 };
    t.Hold.fail.store(true, .release);
    defer t.Hold.fail.store(false, .release);
    t.Fake.reset();
    const f = try stopWith(&root, &ats, 5, prompt);
    var fences: usize = 0;
    for (t.Fake.seen()) |c| {
        if (c == .fence) fences += 1;
    }
    try std.testing.expectEqual(@as(usize, 1), fences); // no download for states a failing tier would lose too
    try std.testing.expectEqual(@as(u64, 0), f.written);
    try std.testing.expectEqual(@as(u64, 1), f.lost);
    try std.testing.expectEqual(@as(u64, 2), f.left);
}
