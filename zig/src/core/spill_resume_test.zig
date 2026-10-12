//! Stale resume on the host with a toy two-rank family (no model): longest prefix all ranks hold, restart survival,
//! identity mismatch and corruption go cold, and a failed rank leaves no rank restored.
const std = @import("std");
const spill = @import("spill.zig");
const resume_ = @import("spill_resume.zig");
const Tier = spill.Tier;
const Name = spill.Name;

const io = std.testing.io;
const gpa = std.testing.allocator;
const family_label = "neutral-toy"; // any family's label: the tier keys directories by it, never by a model

const identity_a: [32]u8 = @splat(0x3c);
const identity_b: [32]u8 = @splat(0x4d);

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

    fn settings(r: *const Root) spill.Settings {
        return .{ .dir = r.path, .cap = 1 << 22, .min_free = 0 };
    }

    fn file(r: *const Root, buf: []u8, identity: [32]u8, rank: u32, name: Name) [:0]const u8 {
        return std.fmt.bufPrintSentinel(buf, "{s}/" ++ family_label ++ "-{x}/rank{d}/{x:0>16}-{d}.tfs", .{ r.path, identity[0..8], rank, name.key, name.at }, 0) catch unreachable;
    }
};

/// Two ranks' tiers under one root and identity.
const Pair = struct {
    tiers: [2]*Tier,

    fn open(root: *const Root, identity: [32]u8) !Pair {
        const a = try Tier.open(gpa, io, root.settings(), family_label, identity, 0, 2);
        errdefer a.deinit();
        return .{ .tiers = .{ a, try Tier.open(gpa, io, root.settings(), family_label, identity, 1, 2) } };
    }

    fn deinit(p: *Pair) void {
        for (p.tiers) |t| t.deinit();
    }
};

const Mem = struct {
    bytes: []const u8,

    fn read(ptr: *anyopaque, offset: u64, out: []u8) anyerror!void {
        const m: *Mem = @ptrCast(@alignCast(ptr));
        @memcpy(out, m.bytes[@intCast(offset)..][0..out.len]);
    }
};

fn payload(buf: []u8, rank: u32, at: u32) []u8 {
    for (buf, 0..) |*b, i| b.* = @truncate(i *% 13 +% rank *% 101 +% at);
    return buf;
}

fn put(t: *Tier, tokens: []const u32, at: u32, rank: u32) !void {
    var pay: [3000]u8 = undefined;
    var m: Mem = .{ .bytes = payload(&pay, rank, at) };
    const j = try t.write(Name.of(tokens[0..at], at), tokens[0..at], "", m.bytes.len, .{ .ptr = &m, .read = Mem.read });
    defer j.release();
    j.waitCopied();
    try std.testing.expect(t.drain(10));
    try std.testing.expect(j.ok);
}

fn putAll(p: *Pair, tokens: []const u32, at: u32) !void {
    for (p.tiers, 0..) |t, r| try put(t, tokens, at, @intCast(r));
}

/// A toy family: each rank's restored bytes, a rank that fails, and a position floor it refuses below.
const Toy = struct {
    live: [2]std.ArrayList(u8) = .{ .empty, .empty },
    restored: [2]bool = .{ false, false },
    discards: u32 = 0,
    fail_rank: ?u32 = null,
    refuse_above: ?u32 = null,

    fn family(t: *Toy) resume_.Family {
        return .{ .ptr = t, .vtable = &.{ .eligible = eligible, .restore = restore, .discard = discard } };
    }

    fn deinit(t: *Toy) void {
        for (&t.live) |*l| l.deinit(gpa);
    }

    fn eligible(ptr: *anyopaque, at: u32) bool {
        const t: *Toy = @ptrCast(@alignCast(ptr));
        return if (t.refuse_above) |most| at <= most else true;
    }

    fn restore(ptr: *anyopaque, rank: u32, reading: *spill.Reading) anyerror!void {
        const t: *Toy = @ptrCast(@alignCast(ptr));
        if (t.fail_rank == rank) { // the failing rank half-overwrote its live state first
            try t.live[rank].appendSlice(gpa, "partial");
            t.restored[rank] = true;
            return error.ToyRestoreFailed;
        }
        t.live[rank].clearRetainingCapacity();
        try reading.pump(.{ .ptr = &t.live[rank], .write = sink });
        t.restored[rank] = true;
    }

    fn sink(ptr: *anyopaque, offset: u64, bytes: []const u8) anyerror!void {
        const l: *std.ArrayList(u8) = @ptrCast(@alignCast(ptr));
        if (offset != l.items.len) return error.OutOfOrder;
        try l.appendSlice(gpa, bytes);
    }

    fn discard(ptr: *anyopaque, rank: u32) void {
        const t: *Toy = @ptrCast(@alignCast(ptr));
        t.restored[rank] = false;
        t.live[rank].clearRetainingCapacity();
        t.discards += 1;
    }
};

fn tokensOf(buf: []u32) []u32 {
    for (buf, 0..) |*t, i| t.* = @intCast((i * 11 + 5) % 997);
    return buf;
}

test "after an isolated restart the longest prefix both ranks hold resumes on both ranks with their own bytes" {
    var root = try Root.make();
    defer root.deinit();
    var tok: [900]u32 = undefined;
    const tokens = tokensOf(&tok);
    {
        var p = try Pair.open(&root, identity_a);
        defer p.deinit();
        try putAll(&p, tokens, 300);
        try putAll(&p, tokens, 600);
    }
    var p = try Pair.open(&root, identity_a); // a new process: the index is rebuilt from files only
    defer p.deinit();
    var toy: Toy = .{};
    defer toy.deinit();
    var why: resume_.Refusals = .{};
    const got = resume_.resumeLongest(&p.tiers, tokens, 0, 0, toy.family(), &why);
    try std.testing.expectEqual(resume_.Outcome{ .resumed = 600 }, got);
    var want: [3000]u8 = undefined;
    for (0..2) |r| {
        try std.testing.expect(toy.restored[r]);
        try std.testing.expectEqualSlices(u8, payload(&want, @intCast(r), 600), toy.live[r].items);
    }
    try std.testing.expectEqual(@as(u64, 600), why.resumed_tokens);
    // avoided prefill is the tier's own count on every rank
    for (p.tiers) |t| try std.testing.expectEqual(@as(u64, 600), t.info().stats.loaded_tokens);
    // a prompt that does not extend a stored state is cold
    try std.testing.expectEqual(resume_.Outcome.cold, resume_.resumeLongest(&p.tiers, tokens[0..300], 0, 0, toy.family(), &why));
    try std.testing.expectEqual(@as(u64, 1), why.no_prefix);
}

test "a longer state one rank lacks is skipped for the longest pair; a family's refusal skips it too" {
    var root = try Root.make();
    defer root.deinit();
    var tok: [900]u32 = undefined;
    const tokens = tokensOf(&tok);
    var p = try Pair.open(&root, identity_a);
    defer p.deinit();
    try putAll(&p, tokens, 200);
    try putAll(&p, tokens, 400);
    try put(p.tiers[0], tokens, 700, 0); // rank 1 never wrote its half
    var toy: Toy = .{};
    defer toy.deinit();
    var why: resume_.Refusals = .{};
    try std.testing.expectEqual(resume_.Outcome{ .resumed = 400 }, resume_.resumeLongest(&p.tiers, tokens, 0, 0, toy.family(), &why));
    try std.testing.expectEqual(@as(u64, 1), why.missing_rank);
    toy.refuse_above = 300;
    try std.testing.expectEqual(resume_.Outcome{ .resumed = 200 }, resume_.resumeLongest(&p.tiers, tokens, 0, 0, toy.family(), &why));
    try std.testing.expect(why.ineligible >= 1);
}

test "a different model, runtime or layout identity never reads another's states" {
    var root = try Root.make();
    defer root.deinit();
    var tok: [500]u32 = undefined;
    const tokens = tokensOf(&tok);
    {
        var p = try Pair.open(&root, identity_a);
        defer p.deinit();
        try putAll(&p, tokens, 400);
    }
    var p = try Pair.open(&root, identity_b);
    defer p.deinit();
    var toy: Toy = .{};
    defer toy.deinit();
    var why: resume_.Refusals = .{};
    try std.testing.expectEqual(resume_.Outcome.cold, resume_.resumeLongest(&p.tiers, tokens, 0, 0, toy.family(), &why));
    try std.testing.expect(!toy.restored[0] and !toy.restored[1]);
    try std.testing.expectEqual(@as(u64, 1), why.no_prefix);
}

test "a corrupt rank-1 header goes cold, discards rank 0, and forgets the state on both ranks" {
    var root = try Root.make();
    defer root.deinit();
    var tok: [500]u32 = undefined;
    const tokens = tokensOf(&tok);
    var p = try Pair.open(&root, identity_a);
    defer p.deinit();
    try putAll(&p, tokens, 400);
    const name = Name.of(tokens[0..400], 400);
    var buf: [600]u8 = undefined;
    const path = root.file(&buf, identity_a, 1, name);
    const fd = std.c.open(path, .{ .ACCMODE = .WRONLY }, @as(std.c.mode_t, 0));
    try std.testing.expect(fd >= 0);
    try std.testing.expectEqual(@as(isize, 4), std.c.pwrite(fd, "XXXX", 4, 0)); // the magic
    _ = std.c.close(fd);
    var toy: Toy = .{};
    defer toy.deinit();
    var why: resume_.Refusals = .{};
    try std.testing.expectEqual(resume_.Outcome.cold, resume_.resumeLongest(&p.tiers, tokens, 0, 0, toy.family(), &why));
    try std.testing.expectEqual(@as(u64, 1), why.read_failed);
    try std.testing.expect(!toy.restored[0] and !toy.restored[1]);
    try std.testing.expectEqual(@as(u32, 1), toy.discards);
    for (p.tiers) |t| try std.testing.expect(!t.has(name));
    try std.testing.expect(p.tiers[1].info().stats.load_failed >= 1);
}

test "a rank whose family restore fails leaves no rank restored and the pair forgotten" {
    var root = try Root.make();
    defer root.deinit();
    var tok: [500]u32 = undefined;
    const tokens = tokensOf(&tok);
    var p = try Pair.open(&root, identity_a);
    defer p.deinit();
    try putAll(&p, tokens, 250);
    var toy: Toy = .{ .fail_rank = 1 };
    defer toy.deinit();
    var why: resume_.Refusals = .{};
    try std.testing.expectEqual(resume_.Outcome.cold, resume_.resumeLongest(&p.tiers, tokens, 0, 0, toy.family(), &why));
    try std.testing.expectEqual(@as(u64, 1), why.restore_failed);
    try std.testing.expectEqual(@as(u64, 0), why.resumed);
    try std.testing.expect(!toy.restored[0] and !toy.restored[1]);
    try std.testing.expectEqual(@as(u32, 2), toy.discards); // rank 0's whole restore and rank 1's partial one
    for (toy.live) |l| try std.testing.expectEqual(@as(usize, 0), l.items.len);
    // rank 0 read its file (I/O counted) but no prefill was avoided: the pair never committed
    try std.testing.expectEqual(@as(u64, 250), p.tiers[0].info().stats.loaded_tokens);
    try std.testing.expectEqual(@as(u64, 0), why.resumed_tokens);
    for (p.tiers) |t| try std.testing.expect(!t.has(Name.of(tokens[0..250], 250)));
}

test "persistence stays off unless TENSORFOLD_SPILL_GIB configures a tier" {
    const Env = struct {
        gib: ?[]const u8,
        pub fn get(e: @This(), name: [:0]const u8) ?[]const u8 {
            if (std.mem.eql(u8, name, "TENSORFOLD_SPILL_GIB")) return e.gib;
            if (std.mem.eql(u8, name, "TENSORFOLD_SPILL_DIR")) return "/var/tmp/x";
            return null;
        }
    };
    try std.testing.expect((try spill.Settings.fromEnv(gpa, Env{ .gib = null })) == null);
    try std.testing.expect((try spill.Settings.fromEnv(gpa, Env{ .gib = "0" })) == null);
    const on = (try spill.Settings.fromEnv(gpa, Env{ .gib = "2" })).?;
    defer gpa.free(on.dir);
    try std.testing.expectEqual(@as(u64, 2 << 30), on.cap);
}

/// A payload whose bytes the writer cannot take until `open`: its write stays pending.
const Gate = struct {
    bytes: []const u8,
    open: std.atomic.Value(bool) = .init(false),

    fn read(ptr: *anyopaque, offset: u64, out: []u8) anyerror!void {
        const g: *Gate = @ptrCast(@alignCast(ptr));
        while (!g.open.load(.acquire)) std.Thread.yield() catch {};
        @memcpy(out, g.bytes[@intCast(offset)..][0..out.len]);
    }
};

test "a longest state whose peer copy is still being written yields the shorter pair and both files survive" {
    var root = try Root.make();
    defer root.deinit();
    var tok: [900]u32 = undefined;
    const tokens = tokensOf(&tok);
    var p = try Pair.open(&root, identity_a);
    defer p.deinit();
    try putAll(&p, tokens, 400);
    try put(p.tiers[0], tokens, 700, 0);
    var pay: [3000]u8 = undefined;
    var gate: Gate = .{ .bytes = payload(&pay, 1, 700) };
    const long = Name.of(tokens[0..700], 700);
    const j = try p.tiers[1].write(long, tokens[0..700], "", gate.bytes.len, .{ .ptr = &gate, .read = Gate.read });
    try std.testing.expect(p.tiers[1].pending(long));
    var toy: Toy = .{};
    defer toy.deinit();
    var why: resume_.Refusals = .{};
    const got = resume_.resumeLongest(&p.tiers, tokens, 0, 0, toy.family(), &why);
    gate.open.store(true, .release);
    j.waitCopied();
    try std.testing.expect(p.tiers[1].drain(10));
    try std.testing.expect(j.ok);
    j.release();
    try std.testing.expectEqual(resume_.Outcome{ .resumed = 400 }, got);
    try std.testing.expect(why.missing_rank >= 1);
    try std.testing.expectEqual(@as(u64, 0), why.read_failed);
    for (p.tiers) |t| try std.testing.expect(t.stored(long, tokens[0..700])); // neither rank's long copy was deleted
}

test "a peer file under the same name but other tokens is not a pair" {
    var root = try Root.make();
    defer root.deinit();
    var tok: [900]u32 = undefined;
    const tokens = tokensOf(&tok);
    var other: [900]u32 = undefined;
    for (&other, 0..) |*t, i| t.* = @intCast(i + 3);
    var p = try Pair.open(&root, identity_a);
    defer p.deinit();
    try putAll(&p, tokens, 200);
    try put(p.tiers[0], tokens, 600, 0);
    var pay: [3000]u8 = undefined;
    var m: Mem = .{ .bytes = payload(&pay, 1, 600) };
    const name = Name.of(tokens[0..600], 600); // rank 1 files other tokens under rank 0's name
    const j = try p.tiers[1].write(name, other[0..600], "", m.bytes.len, .{ .ptr = &m, .read = Mem.read });
    j.waitCopied();
    try std.testing.expect(p.tiers[1].drain(10));
    j.release();
    try std.testing.expect(p.tiers[1].has(name) and !p.tiers[1].stored(name, tokens[0..600]));
    var toy: Toy = .{};
    defer toy.deinit();
    var why: resume_.Refusals = .{};
    try std.testing.expectEqual(resume_.Outcome{ .resumed = 200 }, resume_.resumeLongest(&p.tiers, tokens, 0, 0, toy.family(), &why));
    try std.testing.expectEqual(@as(u64, 1), why.missing_rank);
}

test "a tier of another world size reads none of a pair's files" {
    var root = try Root.make();
    defer root.deinit();
    var tok: [500]u32 = undefined;
    const tokens = tokensOf(&tok);
    {
        var p = try Pair.open(&root, identity_a);
        defer p.deinit();
        try putAll(&p, tokens, 400);
    }
    const t = try Tier.open(gpa, io, root.settings(), family_label, identity_a, 0, 4);
    defer t.deinit();
    try std.testing.expect(t.best(tokens, 0, {}, struct {
        fn any(_: void, _: u32) bool {
            return true;
        }
    }.any) == null);
}

test "a spill identity changes with each of weights, runtime, layout, dtype and TP, and refuses empty fields" {
    const base: resume_.Identity = .{ .weights = @splat(7), .runtime = "tf-1.0.7+k1", .layout = "kv-paged-256,la1", .dtype = "kv=fp8", .tp = 2 };
    const d = try base.digest();
    try std.testing.expectEqualSlices(u8, &d, &(try base.digest()));
    var changed = [_]resume_.Identity{ base, base, base, base, base };
    changed[0].weights[31] = 8;
    changed[1].runtime = "tf-1.0.7+k2";
    changed[2].layout = "kv-paged-128,la1";
    changed[3].dtype = "kv=bf16";
    changed[4].tp = 1;
    for (changed) |c| try std.testing.expect(!std.mem.eql(u8, &d, &(try c.digest())));
    // field boundaries are length-prefixed: moving text between fields is a different identity
    var shifted = base;
    shifted.runtime = "tf-1.0.7+k1kv-paged-256";
    shifted.layout = ",la1";
    try std.testing.expect(!std.mem.eql(u8, &d, &(try shifted.digest())));
    var bad = base;
    bad.dtype = "";
    try std.testing.expectError(error.BadSpillIdentity, bad.digest());
    bad = base;
    bad.tp = 0;
    try std.testing.expectError(error.BadSpillIdentity, bad.digest());
    bad = base;
    bad.weights = @splat(0);
    try std.testing.expectError(error.BadSpillIdentity, bad.digest());
    bad = base;
    bad.layout = &[_]u8{ 'a', 0, 'b' };
    try std.testing.expectError(error.BadSpillIdentity, bad.digest());
}
