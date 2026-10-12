//! The prompt cache with a spill tier, on the host with the running-sum toy family (no model): a stale session resumed
//! after a restart replies exactly as a cold pass while prefilling only the tokens past the stored state; refusals and
//! failures go cold and are counted; without a tier or spill hooks nothing is written.
const std = @import("std");
const pc = @import("prompt_cache.zig");
const spill = pc.spill;
const Fake = @import("prompt_cache_fake.zig").Fake;
const fresh = @import("prompt_cache_test.zig").fresh;

const io = std.testing.io;
const gpa = std.testing.allocator;
const label = "neutral-toy";
const identity_a: [32]u8 = @splat(0x61);
const identity_b: [32]u8 = @splat(0x62);
const rules: pc.Rules = .{ .min_prompt = 0, .min_gap = 1 };

const Root = struct {
    tmp: std.testing.TmpDir,
    path: []u8,

    fn make() !Root {
        const tmp = std.testing.tmpDir(.{});
        return .{ .tmp = tmp, .path = try std.fmt.allocPrint(gpa, ".zig-cache/tmp/{s}", .{tmp.sub_path}) };
    }

    fn deinit(r: *Root) void {
        gpa.free(r.path);
        r.tmp.cleanup();
    }

    fn tier(r: *const Root, identity: [32]u8) !*spill.Tier {
        return spill.Tier.open(gpa, io, .{ .dir = r.path, .cap = 1 << 20, .min_free = 0, .min_tokens = 64 }, label, identity, 0, 1);
    }

    fn file(r: *const Root, buf: []u8, identity: [32]u8, name: spill.Name) [:0]const u8 {
        return std.fmt.bufPrintSentinel(buf, "{s}/" ++ label ++ "-{x}/rank0/{x:0>16}-{d}.tfs", .{ r.path, identity[0..8], name.key, name.at }, 0) catch unreachable;
    }
};

fn conversation(buf: []u32, seed: u32) []u32 {
    for (buf, 0..) |*t, i| t.* = @intCast((i * 17 + seed) % 5003 + 1);
    return buf;
}

/// One server lifetime: a pass over `prompt` keeping its history at `history`, then a clean stop's flush.
fn firstLife(root: *const Root, prompt: []const u32, history: u32) !void {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const t = try root.tier(identity_a);
    defer t.deinit();
    var f: Fake = .{ .gpa = gpa };
    var s = pc.Store.init(gpa, f.spilling(), rules, 1 << 20);
    defer s.deinit();
    s.attach(t);
    const out = f.pass(&s, prompt, try s.begin(arena.allocator(), prompt, history, &.{}, &.{}, null, &.{}));
    try std.testing.expectEqual(fresh(prompt), out);
    try std.testing.expect(s.flushSpill(10));
    try std.testing.expect(t.info().stats.written >= 1);
}

test "a stale session after a restart replies as a cold pass and prefills only past the stored state" {
    var root = try Root.make();
    defer root.deinit();
    var buf: [700]u32 = undefined;
    const turn2 = conversation(&buf, 9);
    try firstLife(&root, turn2[0..500], 400);
    // a new process: nothing in memory, the tier's index rebuilt from its files
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const t = try root.tier(identity_a);
    defer t.deinit();
    var f: Fake = .{ .gpa = gpa };
    var s = pc.Store.init(gpa, f.spilling(), rules, 1 << 20);
    defer s.deinit();
    s.attach(t);
    const plan = try s.begin(arena.allocator(), turn2, 650, &.{}, &.{}, null, &.{});
    try std.testing.expectEqual(@as(u32, 400), plan.from); // the longest stored prefix, not the 0 a cold start has
    const restored = f.pass(&s, turn2, plan);
    var cold: Fake = .{ .gpa = gpa };
    try std.testing.expectEqual(cold.pass(&s, turn2, .{}), restored); // same reply state as a cold prefill
    try std.testing.expectEqual(fresh(turn2), restored);
    try std.testing.expectEqual(@as(usize, 300), turn2.len - plan.from); // prefilled 300 of 700
    try std.testing.expectEqual(@as(u64, 1), s.spill_counts.recalled);
    try std.testing.expectEqual(@as(u64, 400), s.spill_counts.recalled_tokens); // the avoided prefill
    try std.testing.expectEqual(@as(u64, 400), s.spillInfo().?.stats.loaded_tokens); // I/O, separately counted
}

test "a state read back from disk whose family restore fails avoids no prefill and matches a cold reply" {
    var root = try Root.make();
    defer root.deinit();
    var buf: [700]u32 = undefined;
    const turn2 = conversation(&buf, 11);
    try firstLife(&root, turn2[0..500], 400);
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const t = try root.tier(identity_a);
    defer t.deinit();
    var f: Fake = .{ .gpa = gpa, .fail_restore = true };
    var s = pc.Store.init(gpa, f.spilling(), rules, 1 << 20);
    defer s.deinit();
    s.attach(t);
    const plan = try s.begin(arena.allocator(), turn2, 650, &.{}, &.{}, null, &.{});
    try std.testing.expectEqual(@as(u32, 0), plan.from);
    try std.testing.expectEqual(fresh(turn2), f.pass(&s, turn2, plan));
    try std.testing.expectEqual(@as(u64, 1), s.spill_counts.read_back);
    try std.testing.expectEqual(@as(u64, 400), t.info().stats.loaded_tokens); // read from disk
    try std.testing.expectEqual(@as(u64, 0), s.spill_counts.recalled);
    try std.testing.expectEqual(@as(u64, 0), s.spill_counts.recalled_tokens); // but no prefill avoided
}

test "a state the budget evicts goes to disk and a prompt that extends it reads it back" {
    var root = try Root.make();
    defer root.deinit();
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const t = try root.tier(identity_a);
    defer t.deinit();
    var f: Fake = .{ .gpa = gpa };
    var s = pc.Store.init(gpa, f.spilling(), rules, 100 + 400); // one state of 400 tokens
    defer s.deinit();
    s.attach(t);
    var b1: [600]u32 = undefined;
    var b2: [600]u32 = undefined;
    const one = conversation(&b1, 1);
    const two = conversation(&b2, 2);
    _ = f.pass(&s, one[0..450], try s.begin(a, one[0..450], 400, &.{}, &.{}, null, &.{}));
    _ = f.pass(&s, two[0..450], try s.begin(a, two[0..450], 400, &.{}, &.{}, null, &.{})); // evicts one's state
    try std.testing.expect(t.drain(10));
    try std.testing.expectEqual(@as(u64, 1), t.info().stats.written);
    const plan = try s.begin(a, one, 550, &.{}, &.{}, null, &.{});
    try std.testing.expectEqual(@as(u32, 400), plan.from);
    try std.testing.expectEqual(fresh(one), f.pass(&s, one, plan));
    try std.testing.expectEqual(@as(u64, 1), s.spill_counts.recalled);
}

test "a family read failure, a corrupt file and another identity go cold, counted, and match a cold reply" {
    var root = try Root.make();
    defer root.deinit();
    var buf: [700]u32 = undefined;
    const turn2 = conversation(&buf, 4);
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    { // the family refuses the read: forgotten on disk, prefilled from 0
        try firstLife(&root, turn2[0..500], 400);
        const t = try root.tier(identity_a);
        defer t.deinit();
        var f: Fake = .{ .gpa = gpa, .fail_spill_read = true };
        var s = pc.Store.init(gpa, f.spilling(), rules, 1 << 20);
        defer s.deinit();
        s.attach(t);
        const plan = try s.begin(a, turn2, 650, &.{}, &.{}, null, &.{});
        try std.testing.expectEqual(@as(u32, 0), plan.from);
        try std.testing.expectEqual(@as(usize, 1), f.spill_forgets); // the peers were told to drop their copies
        try std.testing.expectEqual(fresh(turn2), f.pass(&s, turn2, plan));
        try std.testing.expectEqual(@as(u64, 1), s.spill_counts.recall_failed);
        try std.testing.expectEqual(@as(u64, 0), s.spill_counts.recalled);
        try std.testing.expect(!t.has(spill.Name.of(turn2[0..400], 400)));
    }
    { // a corrupt header: the tier refuses it, counted as a load failure and a recall failure
        try firstLife(&root, turn2[0..500], 400);
        const t = try root.tier(identity_a);
        defer t.deinit();
        var p: [600]u8 = undefined; // corrupted after the restart's scan indexed it
        const path = root.file(&p, identity_a, spill.Name.of(turn2[0..400], 400));
        const fd = std.c.open(path, .{ .ACCMODE = .WRONLY }, @as(std.c.mode_t, 0));
        try std.testing.expect(fd >= 0);
        try std.testing.expectEqual(@as(isize, 4), std.c.pwrite(fd, "XXXX", 4, 0));
        _ = std.c.close(fd);
        var f: Fake = .{ .gpa = gpa };
        var s = pc.Store.init(gpa, f.spilling(), rules, 1 << 20);
        defer s.deinit();
        s.attach(t);
        const plan = try s.begin(a, turn2, 650, &.{}, &.{}, null, &.{});
        try std.testing.expectEqual(@as(u32, 0), plan.from);
        try std.testing.expectEqual(fresh(turn2), f.pass(&s, turn2, plan));
        try std.testing.expectEqual(@as(u64, 1), s.spill_counts.recall_failed);
        try std.testing.expect(t.info().stats.load_failed >= 1);
    }
    { // another model, runtime, layout, dtype or TP identity: its own directory, nothing found
        try firstLife(&root, turn2[0..500], 400);
        const t = try root.tier(identity_b);
        defer t.deinit();
        var f: Fake = .{ .gpa = gpa };
        var s = pc.Store.init(gpa, f.spilling(), rules, 1 << 20);
        defer s.deinit();
        s.attach(t);
        const plan = try s.begin(a, turn2, 650, &.{}, &.{}, null, &.{});
        try std.testing.expectEqual(@as(u32, 0), plan.from);
        try std.testing.expectEqual(fresh(turn2), f.pass(&s, turn2, plan));
        try std.testing.expectEqual(@as(u64, 1), t.info().stats.disk_misses);
    }
}

test "short states are not written and counted; without spill hooks or a tier nothing is written" {
    var root = try Root.make();
    defer root.deinit();
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var buf: [300]u32 = undefined;
    const prompt = conversation(&buf, 7);
    const t = try root.tier(identity_a);
    defer t.deinit();
    { // a state under the tier's min_tokens
        var f: Fake = .{ .gpa = gpa };
        var s = pc.Store.init(gpa, f.spilling(), rules, 1 << 20);
        defer s.deinit();
        s.attach(t);
        _ = f.pass(&s, prompt[0..40], try s.begin(a, prompt[0..40], 32, &.{}, &.{}, null, &.{}));
        try std.testing.expect(s.flushSpill(10));
        try std.testing.expectEqual(@as(u64, 1), s.spill_counts.ineligible);
    }
    { // a family with no spill hooks, tier attached: off
        var f: Fake = .{ .gpa = gpa };
        var s = pc.Store.init(gpa, f.snapshots(), rules, 1 << 20);
        defer s.deinit();
        s.attach(t);
        _ = f.pass(&s, prompt, try s.begin(a, prompt, 200, &.{}, &.{}, null, &.{}));
        try std.testing.expect(s.flushSpill(10));
    }
    { // spill hooks, no tier: off, and no status
        var f: Fake = .{ .gpa = gpa };
        var s = pc.Store.init(gpa, f.spilling(), rules, 1 << 20);
        defer s.deinit();
        _ = f.pass(&s, prompt, try s.begin(a, prompt, 200, &.{}, &.{}, null, &.{}));
        try std.testing.expect(s.flushSpill(10));
        try std.testing.expect(s.spillInfo() == null);
    }
    try std.testing.expectEqual(@as(u64, 0), t.info().stats.written);
}

test "through lookup, avoided prefill counts only when the caller reports a good restore" {
    var root = try Root.make();
    defer root.deinit();
    var buf: [700]u32 = undefined;
    const turn2 = conversation(&buf, 13);
    try firstLife(&root, turn2[0..500], 400);
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const t = try root.tier(identity_a);
    defer t.deinit();
    var f: Fake = .{ .gpa = gpa };
    var s = pc.Store.init(gpa, f.spilling(), rules, 1 << 20);
    defer s.deinit();
    s.attach(t);
    var l = try s.lookup(a, turn2, 650, &.{}, &.{}, &.{});
    try std.testing.expectEqual(@as(u32, 400), l.entry.?.at);
    try std.testing.expectEqual(@as(u64, 1), s.spill_counts.read_back);
    try std.testing.expectEqual(@as(u64, 0), s.spill_counts.recalled_tokens); // looked up, not yet restored
    s.resumed(l.entry.?, turn2, false); // the backend's live restore failed
    try std.testing.expectEqual(@as(u64, 0), s.spill_counts.recalled_tokens);
    l = try s.lookup(a, turn2, 650, &.{}, &.{}, &.{}); // dropped from memory; the disk copy reads back again
    try std.testing.expectEqual(@as(u64, 2), s.spill_counts.read_back);
    s.resumed(l.entry.?, turn2, true);
    try std.testing.expectEqual(@as(u64, 1), s.spill_counts.recalled);
    try std.testing.expectEqual(@as(u64, 400), s.spill_counts.recalled_tokens);
}

test "a tier outliving its store keeps no hook into it; a later owner's hook survives an earlier store" {
    var root = try Root.make();
    defer root.deinit();
    const t = try spill.Tier.open(gpa, io, .{ .dir = root.path, .cap = 3 * 4096 + 8192, .min_free = 0, .min_tokens = 1 }, label, identity_a, 0, 1);
    defer t.deinit();
    var f: Fake = .{ .gpa = gpa };
    {
        var first = pc.Store.init(gpa, f.spilling(), rules, 1 << 20);
        var second = pc.Store.init(gpa, f.spilling(), rules, 1 << 20);
        first.attach(t);
        second.attach(t); // second owns the hook now
        first.deinit();
        try std.testing.expect(t.hook != null and t.hook.?.ptr == @as(*anyopaque, &second));
        second.clear(); // a weight change: detached
        try std.testing.expect(t.hook == null and second.tier == null);
        second.attach(t);
        second.deinit();
        try std.testing.expect(t.hook == null);
    }
    // the cap now evicts with no store alive: nothing calls into freed memory
    var tok: [64]u32 = undefined;
    for (&tok, 0..) |*x, i| x.* = @intCast(i);
    const pay: [4096]u8 = @splat(1);
    const Mem = struct {
        fn read(p: *anyopaque, offset: u64, out: []u8) anyerror!void {
            const b: *const [4096]u8 = @ptrCast(@alignCast(p));
            @memcpy(out, b[@intCast(offset)..][0..out.len]);
        }
    };
    for (1..4) |k| {
        const at: u32 = @intCast(k * 10);
        const j = try t.write(spill.Name.of(tok[0..at], at), tok[0..at], "", pay.len, .{ .ptr = @constCast(&pay), .read = Mem.read });
        j.waitCopied();
        try std.testing.expect(t.drain(10));
        j.release();
    }
    try std.testing.expect(t.info().stats.disk_evictions >= 1);
}

test "a pair whose rank-1 copy is still being written reads nothing, forgets nothing, and resumes once it lands" {
    var root = try Root.make();
    defer root.deinit();
    var peer_root = try Root.make();
    defer peer_root.deinit();
    var buf: [700]u32 = undefined;
    const turn2 = conversation(&buf, 17);
    try firstLife(&root, turn2[0..500], 400); // rank 0's copy is on disk
    const peer = try peer_root.tier(identity_a);
    defer peer.deinit();
    const name = spill.Name.of(turn2[0..400], 400);
    const Gate = struct {
        open: std.atomic.Value(bool) = .init(false),
        state: Fake.State,
        fn read(p: *anyopaque, offset: u64, out: []u8) anyerror!void {
            const g: *@This() = @ptrCast(@alignCast(p));
            while (!g.open.load(.acquire)) std.Thread.yield() catch {};
            @memcpy(out, std.mem.asBytes(&g.state)[@intCast(offset)..][0..out.len]);
        }
    };
    var gate: Gate = .{ .state = .{ .at = 400, .sum = fresh(turn2[0..400]) } };
    const j = try peer.write(name, turn2[0..400], "", @sizeOf(Fake.State), .{ .ptr = &gate, .read = Gate.read });
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const t = try root.tier(identity_a);
    defer t.deinit();
    var f: Fake = .{ .gpa = gpa, .peer_tier = peer };
    var s = pc.Store.init(gpa, f.spilling(), rules, 1 << 20);
    defer s.deinit();
    s.attach(t);
    const plan = try s.begin(a, turn2, 650, &.{}, &.{}, null, &.{});
    try std.testing.expectEqual(@as(u32, 0), plan.from);
    try std.testing.expectEqual(@as(u64, 1), s.spill_counts.unavailable);
    try std.testing.expectEqual(@as(u64, 0), s.spill_counts.recall_failed);
    try std.testing.expectEqual(@as(usize, 0), f.spill_forgets); // no rank told to drop anything
    try std.testing.expect(t.has(name) and !t.pending(name)); // rank 0's file kept
    try std.testing.expect(peer.pending(name)); // rank 1's write not cut
    try std.testing.expectEqual(fresh(turn2), f.pass(&s, turn2, plan));
    gate.open.store(true, .release);
    j.waitCopied();
    try std.testing.expect(peer.drain(10));
    try std.testing.expect(j.ok);
    j.release();
    s.clear(); // memory holds the cold pass's states: start the next turn from disk alone
    s.attach(t);
    const again = try s.begin(a, turn2, 650, &.{}, &.{}, null, &.{});
    try std.testing.expectEqual(@as(u32, 400), again.from);
    try std.testing.expectEqual(fresh(turn2), f.pass(&s, turn2, again));
    try std.testing.expectEqual(@as(u64, 400), s.spill_counts.recalled_tokens);
}
