const std = @import("std");
const pc = @import("prompt_cache.zig");
const Fake = @import("prompt_cache_fake.zig").Fake;

const gpa = std.testing.allocator;
const io = std.testing.io;
pub const tokens: [500]u32 = @splat(1);
const name = pc.spill.Name.of(tokens[0..401], 400);

pub const Family = struct {
    fake: Fake,
    tier: *pc.spill.Tier,
    active: bool = false,
    busy: bool = false,
    fail: bool = false,
    polls: u32 = 0,
    starts: u32 = 0,
    hold: std.atomic.Value(bool) = .init(false),
    cancelled: u32 = 0,

    fn of(ptr: *anyopaque) *Family {
        return @ptrCast(@alignCast(ptr));
    }

    fn base(f: *Family) pc.Snapshots {
        return f.fake.spilling();
    }

    fn bytes(ptr: *anyopaque, at: u32) u64 {
        const f = of(ptr);
        const b = f.base();
        return b.vtable.bytes(b.ptr, at);
    }

    fn save(ptr: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!pc.Saved {
        const b = of(ptr).base();
        return b.vtable.save(b.ptr, owner, at);
    }

    fn restore(ptr: *anyopaque, owner: ?*anyopaque, saved: pc.Saved) anyerror!void {
        const b = of(ptr).base();
        return b.vtable.restore(b.ptr, owner, saved);
    }

    fn drop(ptr: *anyopaque, saved: pc.Saved) void {
        const b = of(ptr).base();
        b.vtable.drop(b.ptr, saved);
    }

    fn begin(ptr: *anyopaque, tier: *pc.spill.Tier, selected: pc.spill.Name) anyerror!pc.SpillRead {
        const f = of(ptr);
        if (f.busy) return error.ReadBusy;
        try std.testing.expect(tier == f.tier and selected.eql(name));
        f.starts += 1;
        f.active = true;
        return .{ .ptr = f, .vtable = &.{ .poll = poll, .cancel = cancel } };
    }

    fn poll(ptr: *anyopaque) anyerror!?pc.Saved {
        const f = of(ptr);
        try std.testing.expect(f.active);
        f.polls += 1;
        if (f.hold.load(.acquire)) return null;
        if (f.polls == 1) return null;
        f.active = false;
        if (f.fail) return error.PayloadHash;
        const b = f.base();
        return try b.vtable.spill_read.?(b.ptr, f.tier, name);
    }

    fn cancel(ptr: *anyopaque) void {
        const f = of(ptr);
        std.debug.assert(f.active);
        f.active = false;
        f.cancelled += 1;
    }

    const vtable: pc.Snapshots.VTable = .{ .bytes = bytes, .save = save, .restore = restore, .drop = drop, .spill_read_async = begin };

    fn snapshots(f: *Family) pc.Snapshots {
        return .{ .ptr = f, .vtable = &vtable };
    }
};

pub const Fixture = struct {
    tmp: std.testing.TmpDir,
    path: []u8,
    tier: *pc.spill.Tier,
    family: Family,
    store: pc.Store,

    pub fn init() !*Fixture {
        const x = try gpa.create(Fixture);
        errdefer gpa.destroy(x);
        x.tmp = std.testing.tmpDir(.{});
        errdefer x.tmp.cleanup();
        x.path = try std.fmt.allocPrint(gpa, ".zig-cache/tmp/{s}/tier", .{x.tmp.sub_path});
        errdefer gpa.free(x.path);
        x.tier = try pc.spill.Tier.open(gpa, io, .{ .dir = x.path, .cap = 1 << 20, .min_free = 0, .min_tokens = 1 }, "async-toy", @splat(0x51), 0, 1);
        errdefer x.tier.deinit();
        x.family = .{ .fake = .{ .gpa = gpa }, .tier = x.tier };
        x.store = pc.Store.init(gpa, x.family.snapshots(), .{ .min_prompt = 0, .min_gap = 1 }, 1 << 20);
        x.store.attach(x.tier);
        const state: Fake.State = .{ .at = 400, .sum = 400 };
        const Source = struct {
            fn read(ptr: *anyopaque, offset: u64, out: []u8) !void {
                const bytes = std.mem.asBytes(@as(*Fake.State, @ptrCast(@alignCast(ptr))));
                @memcpy(out, bytes[@intCast(offset)..][0..out.len]);
            }
        };
        const job = try x.tier.write(name, tokens[0..401], "", @sizeOf(Fake.State), .{ .ptr = @constCast(&state), .read = Source.read });
        defer job.release();
        job.waitDone();
        try std.testing.expect(job.ok);
        return x;
    }

    pub fn deinit(x: *Fixture) void {
        x.store.deinit();
        x.tier.deinit();
        gpa.free(x.path);
        x.tmp.cleanup();
        gpa.destroy(x);
    }

    fn lookup(x: *Fixture, pending: *?pc.PendingRestore) !pc.Lookup {
        return x.store.lookupRewindAsync(gpa, &tokens, tokens.len, 450, 0, &.{}, &.{}, &.{}, pending);
    }
};

test "async disk lookup publishes only after terminal poll" {
    const x = try Fixture.init();
    defer x.deinit();
    var pending: ?pc.PendingRestore = null;
    try std.testing.expectError(error.RestorePending, x.lookup(&pending));
    try std.testing.expectEqual(@as(usize, 0), x.store.entries.items.len);
    try std.testing.expect(x.store.pending_bytes > 0);
    try std.testing.expect((try x.store.pollRestore(&pending.?)) == null);
    try std.testing.expectEqual(@as(usize, 0), x.store.entries.items.len);
    const result = (try x.store.pollRestore(&pending.?)).?;
    pending = null;
    try std.testing.expectEqual(@as(u32, 400), result.entry.?.at);
    try std.testing.expectEqual(@as(u64, 0), x.store.pending_bytes);
    try std.testing.expectEqual(@as(u64, 1), x.store.spill_counts.read_back);
    x.store.release(result.entry.?);
    gpa.free(result.marks);
}

test "a second pending request reuses the first completed prefix without another read" {
    const x = try Fixture.init();
    defer x.deinit();
    var first: ?pc.PendingRestore = null;
    var second: ?pc.PendingRestore = null;
    try std.testing.expectError(error.RestorePending, x.lookup(&first));
    try std.testing.expectError(error.RestorePending, x.lookup(&second));
    try std.testing.expect((try x.store.pollRestore(&first.?)) == null);
    const one = (try x.store.pollRestore(&first.?)).?;
    first = null;
    const two = (try x.store.pollRestore(&second.?)).?;
    second = null;
    try std.testing.expect(one.entry == two.entry);
    try std.testing.expectEqual(@as(u32, 1), x.family.starts);
    try std.testing.expectEqual(@as(u64, 1), x.store.spill_counts.read_back);
    try std.testing.expectEqual(@as(u64, 0), x.store.pending_bytes);
    try std.testing.expectEqual(@as(usize, 0), x.store.pending_count);
    x.store.release(one.entry.?);
    x.store.release(two.entry.?);
    gpa.free(one.marks);
    gpa.free(two.marks);
}

test "busy async reader waits without a new tier lookup and cancellation frees its reservation" {
    const x = try Fixture.init();
    defer x.deinit();
    x.family.busy = true;
    var pending: ?pc.PendingRestore = null;
    try std.testing.expectError(error.RestorePending, x.lookup(&pending));
    try std.testing.expect((try x.store.pollRestore(&pending.?)) == null);
    try std.testing.expectEqual(@as(u32, 0), x.family.polls);
    x.family.busy = false;
    try std.testing.expect((try x.store.pollRestore(&pending.?)) == null);
    x.store.cancelRestore(&pending.?);
    pending = null;
    try std.testing.expectEqual(@as(u32, 1), x.family.cancelled);
    try std.testing.expectEqual(@as(u64, 0), x.store.pending_bytes);
    try std.testing.expectEqual(@as(u64, 0), x.store.spill_counts.recall_failed);
}

test "a pending restore charges both the byte budget and the kept-state slot" {
    const x = try Fixture.init();
    defer x.deinit();
    x.store.budget = 500;
    x.store.most = 1;
    var pending: ?pc.PendingRestore = null;
    try std.testing.expectError(error.RestorePending, x.lookup(&pending));
    try std.testing.expectEqual(@as(u64, 0), x.store.room());
    x.family.fake.at = 1;
    try std.testing.expect(!x.store.keep(&.{ 7, 8 }, 1, null, &.{}, &.{}));
    try std.testing.expectEqual(@as(usize, 0), x.store.entries.items.len);
    x.store.cancelRestore(&pending.?);
    pending = null;
    try std.testing.expectEqual(@as(u64, 500), x.store.room());
}

test "terminal async failure falls back cold and releases its pending budget" {
    const x = try Fixture.init();
    defer x.deinit();
    x.family.fail = true;
    var pending: ?pc.PendingRestore = null;
    try std.testing.expectError(error.RestorePending, x.lookup(&pending));
    try std.testing.expect((try x.store.pollRestore(&pending.?)) == null);
    const result = (try x.store.pollRestore(&pending.?)).?;
    pending = null;
    try std.testing.expect(result.entry == null);
    try std.testing.expectEqual(@as(u64, 1), x.store.spill_counts.recall_failed);
    try std.testing.expectEqual(@as(u64, 0), x.store.pending_bytes);
    gpa.free(result.marks);
}

test "clear before the reader starts invalidates the candidate and replans cold" {
    const x = try Fixture.init();
    defer x.deinit();
    var pending: ?pc.PendingRestore = null;
    try std.testing.expectError(error.RestorePending, x.lookup(&pending));
    x.store.clear();
    const result = (try x.store.pollRestore(&pending.?)).?;
    pending = null;
    try std.testing.expect(result.entry == null);
    try std.testing.expectEqual(@as(u32, 0), x.family.polls);
    try std.testing.expectEqual(@as(u64, 0), x.store.pending_bytes);
    try std.testing.expectEqual(@as(u64, 0), x.store.spill_counts.read_back);
    gpa.free(result.marks);
}

test "clear during a reader cancels it before tier teardown and never publishes its copy" {
    const x = try Fixture.init();
    defer x.deinit();
    var pending: ?pc.PendingRestore = null;
    try std.testing.expectError(error.RestorePending, x.lookup(&pending));
    try std.testing.expect((try x.store.pollRestore(&pending.?)) == null);
    try std.testing.expect(x.family.active);
    x.store.clear();
    try std.testing.expect(!x.family.active);
    try std.testing.expectEqual(@as(u32, 1), x.family.cancelled);
    const result = (try x.store.pollRestore(&pending.?)).?;
    pending = null;
    try std.testing.expect(result.entry == null);
    try std.testing.expectEqual(@as(u32, 1), x.family.polls);
    try std.testing.expectEqual(@as(u64, 0), x.store.pending_bytes);
    try std.testing.expectEqual(@as(u64, 0), x.store.spill_counts.read_back);
    gpa.free(result.marks);
}

test "detaching and reattaching a tier cannot resume an older reader" {
    const x = try Fixture.init();
    defer x.deinit();
    var pending: ?pc.PendingRestore = null;
    try std.testing.expectError(error.RestorePending, x.lookup(&pending));
    try std.testing.expect((try x.store.pollRestore(&pending.?)) == null);
    x.store.detach();
    x.store.attach(x.tier);
    const result = (try x.store.pollRestore(&pending.?)).?;
    pending = null;
    try std.testing.expect(result.entry == null);
    try std.testing.expectEqual(@as(u32, 1), x.family.cancelled);
    try std.testing.expectEqual(@as(u32, 1), x.family.polls);
    try std.testing.expectEqual(@as(u64, 0), x.store.pending_bytes);
    gpa.free(result.marks);
}

test "clear cancels the surviving reader after middle and head pending links are removed" {
    const x = try Fixture.init();
    defer x.deinit();
    x.family.hold.store(true, .release);
    var first: ?pc.PendingRestore = null;
    var middle: ?pc.PendingRestore = null;
    var head: ?pc.PendingRestore = null;
    try std.testing.expectError(error.RestorePending, x.lookup(&first));
    try std.testing.expectError(error.RestorePending, x.lookup(&middle));
    try std.testing.expectError(error.RestorePending, x.lookup(&head));
    try std.testing.expectEqual(@as(usize, 3), x.store.pending_count);
    try std.testing.expect((try x.store.pollRestore(&middle.?)) == null);
    x.store.cancelRestore(&middle.?);
    middle = null;
    try std.testing.expectEqual(@as(usize, 2), x.store.pending_count);
    try std.testing.expect((try x.store.pollRestore(&head.?)) == null);
    x.store.clear();
    try std.testing.expectEqual(@as(u32, 2), x.family.cancelled);
    const last = (try x.store.pollRestore(&head.?)).?;
    head = null;
    try std.testing.expect(last.entry == null);
    gpa.free(last.marks);
    const oldest = (try x.store.pollRestore(&first.?)).?;
    first = null;
    try std.testing.expect(oldest.entry == null);
    gpa.free(oldest.marks);
    try std.testing.expectEqual(@as(usize, 0), x.store.pending_count);
    try std.testing.expect(x.store.pending_head == null);
    try std.testing.expectEqual(@as(u64, 0), x.store.pending_bytes);
}
