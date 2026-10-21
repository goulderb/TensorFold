//! RAM only (TENSORFOLD_SPILL_RAM_GIB with no TENSORFOLD_SPILL_GIB): every state captured straight into the tier's
//! memory, the least recently used leaving past the cap, nothing written and no flush at a clean stop.
const std = @import("std");
const api = @import("engine_api");
const nemotron = @import("nemotron");
const ns = @import("nemotron_spill.zig");
const t = @import("nemotron_spill_test.zig");
const pc = api.prompt_cache;
const spill = pc.spill;
const N = nemotron.native;

const gpa = std.testing.allocator;
const io = std.testing.io;

const at: u32 = 40; // three states at 40 overflow a cap of two, which still holds one max-window state

fn ramOnly(l: ns.Layout) !spill.Settings {
    const two = 2 * try l.payloadAt(at);
    try std.testing.expect(two >= try l.payload()); // slotless, as RAM only needs
    return .{ .dir = "", .cap = 0, .ram_cap = two, .min_tokens = 1 };
}

fn openRam(l: ns.Layout) !*ns.Disk {
    const d = try ns.Disk.open(gpa, io, gpa, t.device(), l, try ramOnly(l), t.identity_a, 1 << 30);
    t.configure(d);
    d.tier.chunk = 5;
    return d;
}

test "RAM only: states are held straight from the device with no slot, read back whole, and the least recently used leaves" {
    const l = t.layout(10);
    const d = try openRam(l);
    defer d.abandon(null);
    try std.testing.expectEqual(@as(usize, 0), d.slot.len);
    var target = t.targetFor(d);
    var tok: [3][at + 1]u32 = undefined;
    var names: [3]spill.Name = undefined;
    for (0..3) |i| {
        for (&tok[i], 0..) |*x, k| x.* = @intCast(k * 3 + 1 + 1000 * i);
        names[i] = spill.Name.of(&tok[i], at);
        const c = try t.copyAt(l, at, @intCast(i + 1));
        defer t.freeCopy(c);
        try ns.spillWrite(&target, c, d.tier, names[i], &tok[i]);
        try std.testing.expect(d.job == null); // no write was queued
        if (i == 1) { // the first becomes more recent than the second
            const got: *N.Copy = @ptrCast(@alignCast(try ns.spillRead(&target, d.tier, names[0])));
            t.freeCopy(got);
        }
    }
    try std.testing.expect(d.tier.has(names[0]) and d.tier.has(names[2]));
    try std.testing.expect(!d.tier.has(names[1]));
    for ([_]usize{ 0, 2 }) |i| {
        const got: *N.Copy = @ptrCast(@alignCast(try ns.spillRead(&target, d.tier, names[i])));
        defer t.freeCopy(got);
        const want = try t.copyAt(l, at, @intCast(i + 1));
        defer t.freeCopy(want);
        var a: [256]u8 = undefined;
        var b: [256]u8 = undefined;
        try std.testing.expectEqualSlices(u8, t.bytesOf(want.kv, &a), t.bytesOf(got.kv, &b));
        try std.testing.expectEqualSlices(u8, t.bytesOf(want.mamba, &a), t.bytesOf(got.mamba, &b));
        try std.testing.expectEqual(@as(usize, at), got.pos);
    }
    try std.testing.expectEqual(@as(u64, 3), d.counts.held);
    try std.testing.expectEqual(@as(u64, 0), d.counts.submitted);
    try std.testing.expectEqual(@as(u64, 3), d.counts.read_ok);
    const info = d.tier.info();
    try std.testing.expectEqual(@as(u64, 1), info.stats.ram_evicted);
    try std.testing.expectEqual(@as(u64, 0), info.stats.ram_dropped);
    try std.testing.expectEqual(@as(u64, 0), info.stats.written);
    try std.testing.expectEqual(@as(u64, 0), info.cap);
}

test "RAM only: a cap below one max-window state is refused before anything is opened" {
    const l = t.layout(10);
    const s: spill.Settings = .{ .dir = "", .cap = 0, .ram_cap = (try l.payload()) - 1 };
    try std.testing.expectError(error.SpillRamBelowState, ns.Disk.open(gpa, io, gpa, t.device(), l, s, t.identity_a, 1 << 30));
}

test "RAM only: a clean stop captures nothing; the kept states simply end with the process" {
    const l = t.layout(10);
    const d = try openRam(l);
    var target = t.targetFor(d);
    const rules: pc.Rules = .{ .lookahead = 1, .planned = true, .grid = t.grid, .min_prompt = 0, .min_gap = 1 };
    var store = pc.Store.init(gpa, .{ .ptr = &target, .vtable = &t.Family.vtable }, rules, d.reservation + 3 * (l.kvBytes(l.max_len) + l.mamba));
    defer store.deinit();
    d.attach(&target, &store);
    var tok: [32]u32 = undefined;
    const prompt = t.tokensFor(&tok, 20);
    for ([_]u32{ 4, 8, 12 }) |a| try std.testing.expect(store.keep(prompt, a, null, &.{}, &.{}));
    t.Fake.reset();
    const out = try d.close(&store);
    try std.testing.expect(store.tier == null);
    try std.testing.expectEqual(@as(u64, 0), out.counts.held);
    try std.testing.expectEqual(@as(u64, 0), out.written);
    try std.testing.expectEqual(@as(u64, 0), out.left);
    for (t.Fake.seen()) |c| try std.testing.expect(c != .dtoh); // no state left the device
}
