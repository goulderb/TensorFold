//! The Nemotron spill adapter on the host: a fake CUDA driver whose device memory is host memory, recording the
//! order of context binds, stream fences, copies and frees; a real spill tier in a scratch directory.
const std = @import("std");
const cuda = @import("cuda");
const api = @import("engine_api");
const nemotron = @import("nemotron");
const ns = @import("nemotron_spill.zig");
const pc = api.prompt_cache;
const spill = pc.spill;
const N = nemotron.native;
const abi = cuda.abi;

const gpa = std.testing.allocator;
const io = std.testing.io;

// ---- the fake driver -------------------------------------------------------------------------------------------

pub const Call = enum { bind, fence, legacy_sync, dtoh, htod, alloc, free };

pub const Fake = struct {
    var calls: [256]Call = undefined;
    var n: usize = 0;
    var live: isize = 0; // device allocations not yet freed
    var uploaded: usize = 0;
    var drv: cuda.Driver = undefined;
    const stream_handle: abi.Stream = @ptrFromInt(0x5100);

    fn put(c: Call) void {
        if (n < calls.len) calls[n] = c;
        n += 1;
    }
    pub fn reset() void {
        n = 0;
        uploaded = 0;
    }
    pub fn seen() []const Call {
        return calls[0..@min(n, calls.len)];
    }
    fn setCurrent(_: abi.Context) callconv(.c) abi.Result {
        put(.bind);
        return abi.success;
    }
    fn sync(s: abi.Stream) callconv(.c) abi.Result {
        put(if (s == null) .legacy_sync else .fence);
        return abi.success;
    }
    fn alloc(p: *abi.DevicePtr, len: usize) callconv(.c) abi.Result {
        put(.alloc);
        const m = std.c.malloc(len) orelse return 2;
        p.* = @intFromPtr(m);
        live += 1;
        return abi.success;
    }
    fn free(p: abi.DevicePtr) callconv(.c) abi.Result {
        put(.free);
        std.c.free(@ptrFromInt(p));
        live -= 1;
        return abi.success;
    }
    fn dtoh(out: ?*anyopaque, src: abi.DevicePtr, len: usize) callconv(.c) abi.Result {
        put(.dtoh);
        @memcpy(@as([*]u8, @ptrCast(out.?))[0..len], @as([*]const u8, @ptrFromInt(src))[0..len]);
        return abi.success;
    }
    fn htod(dst: abi.DevicePtr, src: ?*const anyopaque, len: usize) callconv(.c) abi.Result {
        put(.htod);
        uploaded += len;
        @memcpy(@as([*]u8, @ptrFromInt(dst))[0..len], @as([*]const u8, @ptrCast(src.?))[0..len]);
        return abi.success;
    }
    fn errName(_: abi.Result, s: *?[*:0]const u8) callconv(.c) abi.Result {
        s.* = "FAKE";
        return abi.success;
    }

    fn driver() *const cuda.Driver {
        drv = .{ .lib = undefined, .api = undefined };
        drv.api.cuCtxSetCurrent = setCurrent;
        drv.api.cuStreamSynchronize = sync;
        drv.api.cuMemAlloc_v2 = alloc;
        drv.api.cuMemFree_v2 = free;
        drv.api.cuMemcpyDtoH_v2 = dtoh;
        drv.api.cuMemcpyHtoD_v2 = htod;
        drv.api.cuGetErrorName = errName;
        drv.api.cuGetErrorString = errName;
        return &drv;
    }
};

var context: cuda.Context = undefined;

pub fn device() ns.Device {
    const d = Fake.driver();
    context = .{ .d = d, .device = 0, .handle = @ptrFromInt(0x7100) };
    return .{ .ctx = &context, .stream = .{ .d = d, .handle = Fake.stream_handle } };
}

// ---- fixtures --------------------------------------------------------------------------------------------------

pub const grid: u32 = 4;

pub fn layout(head: u64) ns.Layout {
    return ns.Layout.of(.{ .kv_row = if (head > 0) 4 else 2, .mamba = 24, .head = head > 0, .max_len = 64 }, grid) catch unreachable;
}

pub const identity_a: [32]u8 = @splat(0x4e);
const identity_b: [32]u8 = @splat(0x4f);

pub const Root = struct {
    tmp: std.testing.TmpDir,
    path: []u8,

    pub fn make() !Root {
        const tmp = std.testing.tmpDir(.{});
        return .{ .tmp = tmp, .path = try std.fmt.allocPrint(gpa, ".zig-cache/tmp/{s}", .{tmp.sub_path}) };
    }
    pub fn deinit(r: *Root) void {
        gpa.free(r.path);
        r.tmp.cleanup();
    }
    pub fn settings(r: *const Root) !spill.Settings {
        return .{ .dir = try gpa.dupe(u8, r.path), .cap = 1 << 24, .min_free = 0, .min_tokens = 1, .flush_s = 5 };
    }
};

fn openDisk(root: *const Root, l: ns.Layout, identity: [32]u8) !*ns.Disk {
    const s = try root.settings();
    errdefer gpa.free(s.dir);
    const d = try ns.Disk.open(gpa, io, gpa, device(), l, s, identity, 1 << 30);
    configure(d);
    d.tier.chunk = 5; // transfers cross the state/head boundary and every plane
    return d;
}

/// The fake image has the same three-region contract as upstream cuda_disk.image.
pub fn configure(d: *ns.Disk) void {
    d.image = &fake_image;
    d.drop = Family.drop;
}

const fake_image: pc.rows.Image = .{ .parts = imageParts, .gather = imageGather, .blank = imageBlank, .scatter = imageScatter };
fn imageParts(_: *anyopaque, _: u32, _: *std.ArrayList(pc.rows.image.Part), _: std.mem.Allocator) anyerror!void {
    return error.Unused;
}
fn allocate(l: ns.Layout, at: usize) !*N.Copy {
    const drv = Fake.driver();
    var kv = try cuda.DeviceBuffer.alloc(drv, @intCast(l.kvBytes(at)));
    errdefer kv.free();
    var mamba = try cuda.DeviceBuffer.alloc(drv, @intCast(l.mamba));
    errdefer mamba.free();
    const c = try gpa.create(N.Copy);
    c.* = .{ .kv = kv, .mamba = mamba, .head = l.head, .pos = at, .parity = 0, .rows = 0 };
    return c;
}
fn imageBlank(ptr: *anyopaque, at: u32) anyerror!pc.Saved {
    return try allocate(Family.diskOf(ptr).layout, at);
}
fn imageGather(_: *anyopaque, saved: pc.Saved, ranges: []const pc.rows.image.Range, out: []u8) anyerror!void {
    const c: *N.Copy = @ptrCast(@alignCast(saved));
    const words = [2]u64{ c.parity, c.rows };
    var off: usize = 0;
    for (ranges) |range| {
        const n: usize = @intCast(range.len);
        if (range.off < 16) @memcpy(out[off..][0..n], std.mem.asBytes(&words)[@intCast(range.off)..][0..n]) else if (range.off < 16 + c.kv.len) try c.kv.download(@intCast(range.off - 16), out[off..][0..n]) else try c.mamba.download(@intCast(range.off - 16 - c.kv.len), out[off..][0..n]);
        off += n;
    }
}
fn imageScatter(_: *anyopaque, saved: pc.Saved, ranges: []const pc.rows.image.Range, in: []const u8) anyerror!void {
    const c: *N.Copy = @ptrCast(@alignCast(saved));
    var off: usize = 0;
    for (ranges) |range| {
        const n: usize = @intCast(range.len);
        if (range.off < 16) {
            var words = [2]u64{ c.parity, c.rows };
            @memcpy(std.mem.asBytes(&words)[@intCast(range.off)..][0..n], in[off..][0..n]);
            c.parity = @intCast(words[0]);
            c.rows = @intCast(words[1]);
        } else if (range.off < 16 + c.kv.len) try c.kv.upload(@intCast(range.off - 16), in[off..][0..n]) else try c.mamba.upload(@intCast(range.off - 16 - c.kv.len), in[off..][0..n]);
        off += n;
    }
}

pub fn copyAt(l: ns.Layout, pos: usize, seed: u8) !*N.Copy {
    const c = try allocate(l, pos);
    errdefer freeCopy(c);
    var buf: [256]u8 = undefined;
    for (buf[0..c.kv.len], 0..) |*b, i| b.* = @truncate(i *% 7 +% seed);
    try c.kv.upload(0, buf[0..c.kv.len]);
    for (buf[0..c.mamba.len], 0..) |*b, i| b.* = @truncate(i *% 13 +% seed +% 101);
    try c.mamba.upload(0, buf[0..c.mamba.len]);
    c.rows = 5;
    return c;
}

pub fn freeCopy(c: *N.Copy) void {
    c.kv.free();
    c.mamba.free();
    gpa.destroy(c);
}

pub fn bytesOf(b: cuda.DeviceBuffer, out: []u8) []u8 {
    @memcpy(out[0..b.len], @as([*]const u8, @ptrFromInt(b.ptr))[0..b.len]);
    return out[0..b.len];
}

pub fn tokensFor(buf: []u32, n: usize) []u32 {
    for (buf[0..n], 0..) |*t, i| t.* = @intCast(i * 3 + 1);
    return buf[0..n];
}

/// Target pointing at a Disk: the Snapshots slots find their adapter through it.
pub fn targetFor(d: *ns.Disk) N.Target {
    return .{ .e = undefined, .head = null, .disk = d };
}

fn asyncRoot() !Root {
    var root = try Root.make();
    errdefer root.deinit();
    const path = try std.fmt.allocPrint(gpa, "{s}/async", .{root.path});
    gpa.free(root.path);
    root.path = path;
    return root;
}

test "async restore uploads one bounded chunk per poll and publishes the complete state" {
    var root = try asyncRoot();
    defer root.deinit();
    const l = layout(10);
    const d = try openDisk(&root, l, identity_a);
    defer d.abandon(null);
    var target = targetFor(d);
    var words: [16]u32 = undefined;
    const tokens = tokensFor(&words, 9);
    const name = spill.Name.of(tokens, 8);
    const original = try copyAt(l, 8, 3);
    defer freeCopy(original);
    try ns.spillWrite(&target, original, d.tier, name, tokens);
    d.settleAll();
    const live = Fake.live;
    Fake.reset();
    const pending = try ns.spillReadAsync(&target, d.tier, name);
    var waiting = true;
    defer if (waiting) pending.vtable.cancel(pending.ptr);
    try std.testing.expectEqual(live, Fake.live);
    const until = std.Io.Clock.awake.now(io).toNanoseconds() + 5 * std.time.ns_per_s;
    var got: ?pc.Saved = null;
    var turns: usize = 0;
    while (got == null and std.Io.Clock.awake.now(io).toNanoseconds() < until) {
        const before = Fake.uploaded;
        got = try pending.vtable.poll(pending.ptr);
        try std.testing.expect(Fake.uploaded - before <= d.tier.chunk);
        turns += @intFromBool(Fake.uploaded > before);
        if (got == null) std.Thread.yield() catch {};
    }
    try std.testing.expect(got != null);
    waiting = false;
    const copy: *N.Copy = @ptrCast(@alignCast(got.?));
    defer freeCopy(copy);
    try std.testing.expect(turns > 1);
    try std.testing.expectEqual(@as(u64, 1), d.counts.read_ok);
    var before: [256]u8 = undefined;
    var after: [256]u8 = undefined;
    try std.testing.expectEqualSlices(u8, bytesOf(original.kv, &before), bytesOf(copy.kv, &after));
    try std.testing.expectEqualSlices(u8, bytesOf(original.mamba, &before), bytesOf(copy.mamba, &after));
}

test "async restore cancellation drops an unpublished GPU copy without counting failure" {
    var root = try asyncRoot();
    defer root.deinit();
    const l = layout(10);
    const d = try openDisk(&root, l, identity_a);
    defer d.abandon(null);
    var target = targetFor(d);
    var words: [16]u32 = undefined;
    const tokens = tokensFor(&words, 9);
    const name = spill.Name.of(tokens, 8);
    const original = try copyAt(l, 8, 3);
    defer freeCopy(original);
    try ns.spillWrite(&target, original, d.tier, name, tokens);
    d.settleAll();
    const live = Fake.live;
    Fake.reset();
    const pending = try ns.spillReadAsync(&target, d.tier, name);
    const until = std.Io.Clock.awake.now(io).toNanoseconds() + 5 * std.time.ns_per_s;
    while (Fake.uploaded == 0 and std.Io.Clock.awake.now(io).toNanoseconds() < until) {
        try std.testing.expectEqual(@as(?pc.Saved, null), try pending.vtable.poll(pending.ptr));
        std.Thread.yield() catch {};
    }
    try std.testing.expect(Fake.uploaded > 0);
    pending.vtable.cancel(pending.ptr);
    try std.testing.expectEqual(live, Fake.live);
    try std.testing.expectEqual(@as(u64, 0), d.counts.read_failed);
    try std.testing.expect(d.tier.has(name));
}

// ---- codec -----------------------------------------------------------------------------------------------------

test "the codec writes every field explicitly and reads back only what the loaded layout writes" {
    const l = layout(10);
    var m: [ns.meta_len]u8 = undefined;
    ns.encode(l, .{ .pos = 8 }, &m);
    const name: spill.Name = .{ .key = 1, .at = 8 };
    const s = try ns.decode(l, &m, name);
    try std.testing.expectEqual(@as(u64, 8), s.pos);
    // each field byte by byte: any change is refused (offsets past the magic)
    var changed: usize = 0;
    for (8..ns.meta_len) |i| {
        var x = m;
        x[i] ^= 0x01;
        if (ns.decode(l, &x, name)) |_| {} else |_| changed += 1;
    }
    try std.testing.expectEqual(ns.meta_len - 8, changed);
    // the magic, a short or long record, another position, an off-grid or zero one
    var bad = m;
    bad[0] = 'X';
    try std.testing.expectError(error.SpillMetaMagic, ns.decode(l, &bad, name));
    try std.testing.expectError(error.SpillMetaLength, ns.decode(l, m[0 .. ns.meta_len - 1], name));
    var long: [ns.meta_len + 1]u8 = undefined;
    @memcpy(long[0..ns.meta_len], &m);
    long[ns.meta_len] = 0;
    try std.testing.expectError(error.SpillMetaLength, ns.decode(l, &long, name));
    try std.testing.expectError(error.SpillScalars, ns.decode(l, &m, .{ .key = 1, .at = 4 }));
    var off: [ns.meta_len]u8 = undefined;
    ns.encode(l, .{ .pos = 6 }, &off);
    try std.testing.expectError(error.SpillScalars, ns.decode(l, &off, .{ .key = 1, .at = 6 }));
    var past: [ns.meta_len]u8 = undefined;
    ns.encode(l, .{ .pos = 64 }, &past);
    try std.testing.expectError(error.SpillScalars, ns.decode(l, &past, .{ .key = 1, .at = 64 })); // + lookahead > max_len
    // the engine's own bound: a pass needs its prompt and a window's rows inside the cache (prefillFrom), so a kept
    // state satisfies pos + lookahead + max_rows <= max_len (64 here: 44 fits, 48 cannot have been kept)
    var last: [ns.meta_len]u8 = undefined;
    ns.encode(l, .{ .pos = 44 }, &last);
    _ = try ns.decode(l, &last, .{ .key = 1, .at = 44 });
    var over: [ns.meta_len]u8 = undefined;
    ns.encode(l, .{ .pos = 48 }, &over);
    try std.testing.expectError(error.SpillScalars, ns.decode(l, &over, .{ .key = 1, .at = 48 }));
    // another layout: no head, other plane, other grid, other window
    try std.testing.expectError(error.SpillLayout, ns.decode(layout(0), &m, name));
    var other = l;
    other.kv_row += 2;
    try std.testing.expectError(error.SpillLayout, ns.decode(other, &m, name));
    other = l;
    other.grid = 8;
    try std.testing.expectError(error.SpillLayout, ns.decode(other, &m, name));
    other = l;
    other.max_len = 128;
    try std.testing.expectError(error.SpillLayout, ns.decode(other, &m, name));
    var old = m;
    @memcpy(old[0..8], "NEMOCU01");
    try std.testing.expectError(error.SpillMetaMagic, ns.decode(l, &old, name));
}

test "compact layout rejects empty or partial elements and checked payload overflow" {
    var sh: N.SpillShape = .{ .kv_row = 4, .mamba = 24, .head = false, .max_len = 64 };
    sh.kv_row = 0;
    try std.testing.expectError(error.BadLayout, ns.Layout.of(sh, grid));
    sh.kv_row = 3;
    try std.testing.expectError(error.BadLayout, ns.Layout.of(sh, grid));
    sh.kv_row = 4;
    sh.mamba = 23;
    try std.testing.expectError(error.BadLayout, ns.Layout.of(sh, grid));
    sh.mamba = 24;
    try std.testing.expectError(error.BadLayout, ns.Layout.of(sh, 0));
    sh.kv_row = std.math.maxInt(u64) - 1;
    try std.testing.expectError(error.BadLayout, ns.Layout.of(sh, grid));
    sh.kv_row = 4;
    sh.mamba = std.math.maxInt(u64) - 1;
    try std.testing.expectError(error.BadLayout, ns.Layout.of(sh, grid));
}

// ---- write and read ---------------------------------------------------------------------------------------------

test "a write fences the engine stream before downloading, and a read uploads with the legacy stream synchronized" {
    var root = try Root.make();
    defer root.deinit();
    const l = layout(10);
    const live0 = Fake.live;
    var tok: [16]u32 = undefined;
    const tokens = tokensFor(&tok, 9); // at 8 and one lookahead token
    const name = spill.Name.of(tokens, 8);
    {
        const d = try openDisk(&root, l, identity_a);
        var target = targetFor(d);
        const c = try copyAt(l, 8, 3);
        defer freeCopy(c);
        Fake.reset();
        try ns.spillWrite(&target, c, d.tier, name, tokens);
        try std.testing.expectEqualSlices(Call, &.{ .bind, .fence, .dtoh, .dtoh }, Fake.seen());
        d.job.?.waitDone();
        d.reap();
        try std.testing.expectEqual(@as(u64, 1), d.counts.written);
        d.abandon(null);
    }
    // a new process: the tier rebuilt from its files
    const d = try openDisk(&root, l, identity_a);
    defer d.abandon(null);
    var target = targetFor(d);
    try std.testing.expect(d.tier.stored(name, tokens));
    Fake.reset();
    const got: *N.Copy = @ptrCast(@alignCast(try ns.spillRead(&target, d.tier, name)));
    defer freeCopy(got);
    const seen = Fake.seen();
    try std.testing.expectEqual(Call.bind, seen[0]);
    try std.testing.expectEqual(Call.alloc, seen[1]);
    try std.testing.expectEqual(Call.alloc, seen[2]);
    // every upload is followed at once by the legacy stream's synchronize, never a fence on the engine stream
    var uploads: usize = 0;
    for (seen[3..], 3..) |c, i| switch (c) {
        .htod => {
            uploads += 1;
            try std.testing.expectEqual(Call.legacy_sync, seen[i + 1]);
        },
        .legacy_sync => {},
        else => return error.TestUnexpectedResult,
    };
    try std.testing.expect(uploads >= 56 / 5); // 32 KV and 24 Mamba bytes, split into five-byte chunks; words stay on the host
    const want = try copyAt(l, 8, 3);
    defer freeCopy(want);
    var a: [256]u8 = undefined;
    var b: [256]u8 = undefined;
    try std.testing.expectEqualSlices(u8, bytesOf(want.kv, &a), bytesOf(got.kv, &b));
    try std.testing.expectEqualSlices(u8, bytesOf(want.mamba, &a), bytesOf(got.mamba, &b));
    try std.testing.expectEqual(@as(usize, 8), got.pos);
    try std.testing.expectEqual(@as(usize, 0), got.rows); // canonical: the next window rewrites it
    try std.testing.expectEqual(@as(usize, 0), got.parity);
    try std.testing.expectEqual(@as(u64, 1), d.counts.read_ok);
    _ = live0;
}

test "a state without an MTP head round-trips; a head-less file is refused by a head engine before any allocation" {
    var root = try Root.make();
    defer root.deinit();
    var tok: [16]u32 = undefined;
    const tokens = tokensFor(&tok, 5);
    const name = spill.Name.of(tokens, 4);
    {
        const d = try openDisk(&root, layout(0), identity_a);
        defer d.abandon(null);
        var target = targetFor(d);
        const c = try copyAt(layout(0), 4, 9);
        defer freeCopy(c);
        try ns.spillWrite(&target, c, d.tier, name, tokens);
        d.settleAll();
        const got: *N.Copy = @ptrCast(@alignCast(try ns.spillRead(&target, d.tier, name)));
        defer freeCopy(got);
        try std.testing.expect(!got.head);
        var a: [256]u8 = undefined;
        var b: [256]u8 = undefined;
        try std.testing.expectEqualSlices(u8, bytesOf(c.kv, &a), bytesOf(got.kv, &b));
    }
    const d = try openDisk(&root, layout(10), identity_a);
    defer d.abandon(null);
    var target = targetFor(d);
    const live = Fake.live;
    Fake.reset();
    try std.testing.expectError(error.SpillLayout, ns.spillRead(&target, d.tier, name));
    for (Fake.seen()) |c| try std.testing.expect(c != .alloc);
    try std.testing.expectEqual(live, Fake.live);
    try std.testing.expectEqual(@as(u64, 1), d.counts.read_failed);
}

test "a corrupt payload frees every buffer it allocated and returns no state; NotStored is not a failure" {
    var root = try Root.make();
    defer root.deinit();
    const l = layout(10);
    var tok: [16]u32 = undefined;
    const tokens = tokensFor(&tok, 9);
    const name = spill.Name.of(tokens, 8);
    const d = try openDisk(&root, l, identity_a);
    defer d.abandon(null);
    var target = targetFor(d);
    const c = try copyAt(l, 8, 1);
    defer freeCopy(c);
    try ns.spillWrite(&target, c, d.tier, name, tokens);
    d.settleAll();
    // flip the last payload byte on disk
    var p: [512]u8 = undefined;
    const path = try std.fmt.bufPrintSentinel(&p, "{s}/nemotron_h-{x}/rank0/{x:0>16}-{d}.tfs", .{ root.path, identity_a[0..8], name.key, name.at }, 0);
    const fd = std.c.open(path, .{ .ACCMODE = .RDWR }, @as(std.c.mode_t, 0));
    try std.testing.expect(fd >= 0);
    const size = std.c.lseek(fd, 0, std.c.SEEK.END);
    var byte: [1]u8 = undefined;
    try std.testing.expectEqual(@as(isize, 1), std.c.pread(fd, &byte, 1, size - 1));
    byte[0] ^= 0xff;
    try std.testing.expectEqual(@as(isize, 1), std.c.pwrite(fd, &byte, 1, size - 1));
    _ = std.c.close(fd);
    const live = Fake.live;
    try std.testing.expectError(error.PayloadHash, ns.spillRead(&target, d.tier, name));
    try std.testing.expectEqual(live, Fake.live); // both temporary buffers freed
    try std.testing.expectEqual(@as(u64, 1), d.counts.read_failed);
    const other = spill.Name.of(tokens[0..5], 4);
    try std.testing.expectError(error.NotStored, ns.spillRead(&target, d.tier, other));
    try std.testing.expectEqual(@as(u64, 1), d.counts.read_failed);
}

test "a write refuses a state that is not the layout's, a decode-window state, or tokens without the lookahead" {
    var root = try Root.make();
    defer root.deinit();
    const l = layout(10);
    const d = try openDisk(&root, l, identity_a);
    defer d.abandon(null);
    var target = targetFor(d);
    var tok: [16]u32 = undefined;
    const tokens = tokensFor(&tok, 9);
    const c = try copyAt(l, 8, 2);
    defer freeCopy(c);
    try std.testing.expectError(error.SpillTokens, ns.spillWrite(&target, c, d.tier, spill.Name.of(tokens[0..8], 8), tokens[0..8]));
    c.parity = 1;
    try std.testing.expectError(error.SpillScalars, ns.spillWrite(&target, c, d.tier, spill.Name.of(tokens, 8), tokens));
    c.parity = 0;
    c.whole = false;
    try std.testing.expectError(error.Incomplete, ns.spillWrite(&target, c, d.tier, spill.Name.of(tokens, 8), tokens));
    c.whole = true;
    c.rows = N.max_rows + 1;
    try std.testing.expectError(error.SpillScalars, ns.spillWrite(&target, c, d.tier, spill.Name.of(tokens, 8), tokens));
    c.rows = 0;
    c.head = false;
    try std.testing.expectError(error.SpillLayout, ns.spillWrite(&target, c, d.tier, spill.Name.of(tokens, 8), tokens));
    c.head = true;
    var tok7: [16]u32 = undefined;
    const t7 = tokensFor(&tok7, 7);
    c.pos = 6;
    try std.testing.expectError(error.SpillScalars, ns.spillWrite(&target, c, d.tier, spill.Name.of(t7, 6), t7)); // off grid
    try std.testing.expectEqual(@as(u64, 0), d.counts.submitted);
}

// ---- ownership ---------------------------------------------------------------------------------------------------

/// A disk whose file writes wait for `open`: a submitted job stays un-copied until then.
pub const Hold = struct {
    pub var open: std.atomic.Value(bool) = .init(true);
    pub var fail: std.atomic.Value(bool) = .init(false);
    pub fn write(fd: c_int, bytes: []const u8, at: u64) anyerror!void {
        while (!open.load(.acquire)) std.Thread.yield() catch {};
        if (fail.load(.acquire)) return error.InjectedIo;
        var off: usize = 0;
        while (off < bytes.len) {
            const n = std.c.pwrite(fd, bytes[off..].ptr, bytes.len - off, @intCast(at + off));
            if (n <= 0) return error.InjectedIo;
            off += @intCast(n);
        }
    }
};

test "a write while the slot is still being read is a counted skip, not a failure; the slot is reused once copied" {
    var root = try Root.make();
    defer root.deinit();
    const l = layout(10);
    const d = try openDisk(&root, l, identity_a);
    defer d.abandon(null);
    d.tier.write_file = Hold.write;
    var target = targetFor(d);
    var tok: [32]u32 = undefined;
    const tokens = tokensFor(&tok, 13);
    const c8 = try copyAt(l, 8, 1);
    defer freeCopy(c8);
    const c12 = try copyAt(l, 12, 2);
    defer freeCopy(c12);
    Hold.open.store(false, .release);
    try ns.spillWrite(&target, c8, d.tier, spill.Name.of(tokens[0..9], 8), tokens[0..9]);
    try ns.spillWrite(&target, c12, d.tier, spill.Name.of(tokens[0..13], 12), tokens[0..13]); // busy: returns success
    try std.testing.expectEqual(@as(u64, 1), d.counts.skipped_busy);
    try std.testing.expectEqual(@as(u64, 1), d.counts.submitted);
    try std.testing.expectEqual(@as(u64, 1), d.tier.info().stats.unwritten_incomplete);
    Hold.open.store(true, .release);
    d.job.?.waitCopied();
    try ns.spillWrite(&target, c12, d.tier, spill.Name.of(tokens[0..13], 12), tokens[0..13]); // the slot is free again
    try std.testing.expectEqual(@as(u64, 2), d.counts.submitted);
    d.settleAll();
    try std.testing.expectEqual(@as(u64, 2), d.counts.written);
}

test "a write the disk gives up is counted lost at its outcome, and its reference released once" {
    var root = try Root.make();
    defer root.deinit();
    const l = layout(10);
    const d = try openDisk(&root, l, identity_a);
    defer d.abandon(null);
    d.tier.write_file = Hold.write;
    var target = targetFor(d);
    var tok: [16]u32 = undefined;
    const tokens = tokensFor(&tok, 9);
    const c = try copyAt(l, 8, 1);
    defer freeCopy(c);
    Hold.fail.store(true, .release);
    defer Hold.fail.store(false, .release);
    try ns.spillWrite(&target, c, d.tier, spill.Name.of(tokens, 8), tokens);
    d.settleAll();
    try std.testing.expectEqual(@as(u64, 1), d.counts.lost);
    try std.testing.expectEqual(@as(u64, 0), d.counts.written);
}

// ---- the Store: spill on eviction, flush at close, read back after restart -------------------------------------

/// Snapshots over Copies the test makes: save copies a pattern for `at`, drop frees it.
pub const Family = struct {
    fn bytes(p: *anyopaque, at: u32) u64 {
        const d = diskOf(p);
        return d.layout.kvBytes(at) + d.layout.mamba;
    }
    fn save(p: *anyopaque, _: ?*anyopaque, at: u32) anyerror!pc.Saved {
        return try copyAt(diskOf(p).layout, at, @truncate(at));
    }
    fn restore(_: *anyopaque, _: ?*anyopaque, _: pc.Saved) anyerror!void {}
    fn drop(_: *anyopaque, saved: pc.Saved) void {
        freeCopy(@ptrCast(@alignCast(saved)));
    }
    fn diskOf(p: *anyopaque) *ns.Disk {
        const t: *N.Target = @ptrCast(@alignCast(p));
        return @ptrCast(@alignCast(t.disk.?));
    }
    pub const vtable: pc.Snapshots.VTable = .{ .bytes = bytes, .save = save, .restore = restore, .drop = drop, .spill_write = ns.spillWrite, .spill_read = ns.spillRead, .spare = ns.spare, .trim = ns.trim };
};

test "close flushes kept states one slot at a time, settles every write, and a restart reads them back" {
    var root = try Root.make();
    defer root.deinit();
    const l = layout(10);
    const per = l.kvBytes(l.max_len) + l.mamba;
    var tok: [32]u32 = undefined;
    const prompt = tokensFor(&tok, 20);
    const rules: pc.Rules = .{ .lookahead = 1, .planned = true, .grid = grid, .min_prompt = 0, .min_gap = 1 };
    {
        const d = try openDisk(&root, l, identity_a);
        var target = targetFor(d);
        var store = pc.Store.init(gpa, .{ .ptr = &target, .vtable = &Family.vtable }, rules, d.reservation + 3 * per);
        d.attach(&target, &store);
        try std.testing.expectEqual(d.reservation, store.spare());
        for ([_]u32{ 4, 8, 12 }) |at| try std.testing.expect(store.keep(prompt, at, null, &.{}, &.{}));
        _ = try d.close(&store); // one pass: each state written once, the slot waited for
        try std.testing.expect(store.tier == null);
        store.deinit();
    }
    const d = try openDisk(&root, l, identity_a);
    defer d.abandon(null);
    var target = targetFor(d);
    for ([_]u32{ 4, 8, 12 }) |at| {
        const name = spill.Name.of(prompt[0 .. at + 1], at);
        try std.testing.expect(d.tier.stored(name, prompt[0 .. at + 1]));
        const got: *N.Copy = @ptrCast(@alignCast(try ns.spillRead(&target, d.tier, name)));
        defer freeCopy(got);
        const want = try copyAt(l, at, @truncate(at));
        defer freeCopy(want);
        var a: [256]u8 = undefined;
        var b: [256]u8 = undefined;
        try std.testing.expectEqualSlices(u8, bytesOf(want.kv, &a), bytesOf(got.kv, &b));
    }
}

test "the host ceiling and the tier quota bound the adapter's memory; nothing is opened past the ceiling" {
    var root = try Root.make();
    defer root.deinit();
    const l = layout(10);
    const s = try root.settings();
    defer gpa.free(s.dir);
    const need = try ns.Disk.reservationFor(try l.payload(), 64 << 20);
    try std.testing.expectEqual(@as(u64, 296 + 2 * (64 << 20) + ns.control_bytes), need);
    try std.testing.expectError(error.SpillHostCeiling, ns.Disk.open(gpa, io, gpa, device(), l, s, identity_a, need - 1));
    var p: [512]u8 = undefined;
    const top = try std.fmt.bufPrint(&p, "{s}/nemotron_h-{x}", .{ root.path, identity_a[0..8] });
    try std.testing.expectError(error.FileNotFound, std.Io.Dir.cwd().access(io, top, .{})); // no tier directory made
}

test "a startup unwind detaches the store and the target before the tier and slot go" {
    var root = try Root.make();
    defer root.deinit();
    const l = layout(10);
    const d = try openDisk(&root, l, identity_a);
    var target = targetFor(d);
    target.disk = null;
    var store = pc.Store.init(gpa, .{ .ptr = &target, .vtable = &Family.vtable }, .{ .lookahead = 1, .planned = true, .grid = grid }, 1 << 20);
    defer store.deinit();
    d.attach(&target, &store);
    try std.testing.expect(store.tier != null and target.disk != null);
    d.abandon(&store);
    try std.testing.expect(store.tier == null and target.disk == null);
    try std.testing.expectEqual(@as(u64, 0), ns.spare(&target)); // no disk: no reservation
}

test "a pending write the tier cancels settles as lost, and the slot takes the next write" {
    var root = try Root.make();
    defer root.deinit();
    const l = layout(10);
    const d = try openDisk(&root, l, identity_a);
    defer d.abandon(null);
    d.tier.write_file = Hold.write;
    var target = targetFor(d);
    var tok: [16]u32 = undefined;
    const tokens = tokensFor(&tok, 9);
    const name = spill.Name.of(tokens, 8);
    const c = try copyAt(l, 8, 6);
    defer freeCopy(c);
    Hold.open.store(false, .release);
    try ns.spillWrite(&target, c, d.tier, name, tokens);
    d.tier.forget(name); // the cap or an older build's cleanup took it while it was on its way
    Hold.open.store(true, .release);
    d.settleAll();
    try std.testing.expectEqual(@as(u64, 1), d.counts.lost);
    try std.testing.expect(!d.tier.has(name));
    try ns.spillWrite(&target, c, d.tier, name, tokens);
    d.settleAll();
    try std.testing.expectEqual(@as(u64, 1), d.counts.written);
    try std.testing.expect(d.tier.stored(name, tokens));
}

test "a tier out of its memory quota refuses the write; nothing is queued and the slot stays free" {
    var root = try Root.make();
    defer root.deinit();
    const l = layout(10);
    const d = try openDisk(&root, l, identity_a);
    defer d.abandon(null);
    var target = targetFor(d);
    var tok: [16]u32 = undefined;
    const tokens = tokensFor(&tok, 9);
    const c = try copyAt(l, 8, 7);
    defer freeCopy(c);
    const limit = d.quota.limit;
    d.quota.limit = d.quota.used.load(.monotonic); // no room for the job, its tokens or its metadata
    try std.testing.expectError(error.OutOfMemory, ns.spillWrite(&target, c, d.tier, spill.Name.of(tokens, 8), tokens));
    try std.testing.expect(d.job == null);
    try std.testing.expectEqual(@as(u64, 0), d.counts.submitted);
    d.quota.limit = limit;
    try ns.spillWrite(&target, c, d.tier, spill.Name.of(tokens, 8), tokens);
    d.settleAll();
    try std.testing.expectEqual(@as(u64, 1), d.counts.written);
}

test "the reservation is charged once: the host ceiling holds it, the cache holds it and one state besides" {
    const payload: u64 = 3 << 30;
    const r = try ns.Disk.reservationFor(payload, 64 << 20);
    try std.testing.expectEqual(payload + 2 * (64 << 20) + ns.control_bytes, r);
    try std.testing.expectEqual(r, try ns.Disk.admit(payload, payload, r, r + payload));
    try std.testing.expectError(error.SpillHostCeiling, ns.Disk.admit(payload, payload, r - 1, 1 << 40));
    try std.testing.expectError(error.SpillCacheTooSmall, ns.Disk.admit(payload, payload, r, r + payload - 1));
    try std.testing.expectError(error.BadLayout, ns.Disk.admit(std.math.maxInt(u64), 1, std.math.maxInt(u64), std.math.maxInt(u64)));
}

test "a clean stop offers each kept state once: a state too short for a file is counted once, not once a retry" {
    var root = try Root.make();
    defer root.deinit();
    const l = layout(10);
    const per = l.kvBytes(l.max_len) + l.mamba;
    var tok: [32]u32 = undefined;
    const prompt = tokensFor(&tok, 20);
    const rules: pc.Rules = .{ .lookahead = 1, .planned = true, .grid = grid, .min_prompt = 0, .min_gap = 1 };
    var settings = try root.settings();
    settings.min_tokens = 8; // the state at 4 is not worth a file
    const d = try ns.Disk.open(gpa, io, gpa, device(), l, settings, identity_a, 1 << 30);
    configure(d);
    d.tier.chunk = 5;
    var target = targetFor(d);
    var store = pc.Store.init(gpa, .{ .ptr = &target, .vtable = &Family.vtable }, rules, d.reservation + 4 * per);
    defer store.deinit();
    d.attach(&target, &store);
    for ([_]u32{ 4, 8, 12, 16 }) |at| try std.testing.expect(store.keep(prompt, at, null, &.{}, &.{}));
    _ = try d.close(&store);
    try std.testing.expectEqual(@as(u64, 1), store.spill_counts.ineligible); // one pass, not one per retry pass
    try std.testing.expectEqual(@as(u64, 0), store.spill_counts.write_failed);
}

test "page-rounded device state is admitted separately from logical host payload" {
    const payload: u64 = 72;
    const pages: u64 = 4 << 20;
    const reservation = try ns.Disk.reservationFor(payload, 64 << 20);
    try std.testing.expectError(error.SpillCacheTooSmall, ns.Disk.admit(payload, pages, reservation, reservation + pages - 1));
    try std.testing.expectEqual(reservation, try ns.Disk.admit(payload, pages, reservation, reservation + pages));
}
