//! The spill tier on the host: files written, indexed after a restart and read back byte for byte; cut writes, foreign
//! and corrupt files never read; the cap and the free-disk floor evict the least recently used.
const std = @import("std");
const spill = @import("spill.zig");
const Tier = spill.Tier;
const Name = spill.Name;

const io = std.testing.io;
const gpa = std.testing.allocator;

/// A payload in host memory, read in whatever pieces the writer asks for; `fail_at` cuts it there.
const Mem = struct {
    bytes: []const u8,
    fail_at: ?u64 = null,

    fn source(m: *Mem) spill.Source {
        return .{ .ptr = m, .read = read };
    }

    fn read(ptr: *anyopaque, offset: u64, out: []u8) anyerror!void {
        const m: *Mem = @ptrCast(@alignCast(ptr));
        if (m.fail_at) |f| if (offset + out.len > f) return error.DeviceGone;
        @memcpy(out, m.bytes[@intCast(offset)..][0..out.len]);
    }
};

const Collect = struct {
    out: std.ArrayList(u8) = .empty,

    fn sink(c: *Collect) spill.Sink {
        return .{ .ptr = c, .write = write };
    }

    fn write(ptr: *anyopaque, offset: u64, bytes: []const u8) anyerror!void {
        const c: *Collect = @ptrCast(@alignCast(ptr));
        if (offset != c.out.items.len) return error.OutOfOrder;
        try c.out.appendSlice(gpa, bytes);
    }
};

const Forgot = struct {
    names: std.ArrayList(Name) = .empty,

    fn hook(f: *Forgot) spill.Hook {
        return .{ .ptr = f, .forgot = forgot };
    }

    fn forgot(ptr: *anyopaque, name: Name) void {
        const f: *Forgot = @ptrCast(@alignCast(ptr));
        f.names.append(gpa, name) catch {};
    }
};

const identity_a: [32]u8 = @splat(0xa1);
const identity_b: [32]u8 = @splat(0xb2);

fn anyAt(_: void, _: u32) bool {
    return true;
}

/// A scratch root for one test (relative to the build root, where tests run).
const Root = struct {
    tmp: std.testing.TmpDir,
    path: []u8,

    fn make() !Root {
        const tmp = std.testing.tmpDir(.{});
        // a root the tier makes (and makes private): the test directory itself inherits its parent's access
        return .{ .tmp = tmp, .path = try std.fmt.allocPrint(gpa, ".zig-cache/tmp/{s}/tier", .{tmp.sub_path}) };
    }

    fn deinit(r: *Root) void {
        gpa.free(r.path);
        r.tmp.cleanup();
    }

    fn settings(r: *const Root, cap: u64) spill.Settings {
        return .{ .dir = r.path, .cap = cap, .min_free = 0 };
    }

    fn file(r: *const Root, buf: []u8, identity: [32]u8, rank: u32, name: Name) [:0]const u8 {
        return std.fmt.bufPrintSentinel(buf, "{s}/flashnext-{x}/rank{d}/{x:0>16}-{d}.tfs", .{ r.path, identity[0..8], rank, name.key, name.at }, 0) catch unreachable;
    }
};

fn tokensOf(buf: []u32, seed: u32) []u32 {
    for (buf, 0..) |*t, i| t.* = @intCast((i * 7 + seed) % 1000);
    return buf;
}

fn payloadOf(buf: []u8, seed: u8) []u8 {
    for (buf, 0..) |*b, i| b.* = @truncate(i *% 31 +% seed);
    return buf;
}

/// Writes one state and waits for it to be on disk.
fn put(t: *Tier, tokens: []const u32, at: u32, meta: []const u8, payload: []const u8) !Name {
    var m: Mem = .{ .bytes = payload };
    const name = Name.of(tokens[0..at], at);
    const j = try t.write(name, tokens[0..at], meta, payload.len, m.source());
    defer j.release();
    j.waitCopied();
    try std.testing.expect(t.drain(10));
    try std.testing.expect(j.ok);
    return name;
}

fn readBack(t: *Tier, name: Name, meta_out: *[64]u8) ![]u8 {
    var r = try t.read(name);
    defer r.close();
    const m = r.meta();
    @memcpy(meta_out[0..m.len], m);
    var c: Collect = .{};
    errdefer c.out.deinit(gpa);
    try r.pump(c.sink());
    return c.out.toOwnedSlice(gpa);
}

test "a state written before a restart is found for a prompt that extends it and reads back byte for byte" {
    var root = try Root.make();
    defer root.deinit();
    var tok: [700]u32 = undefined;
    const tokens = tokensOf(&tok, 3);
    var pay: [10_000]u8 = undefined;
    const payload = payloadOf(&pay, 5);
    {
        const t = try Tier.open(gpa, io, root.settings(1 << 20), "flashnext", identity_a, 0, 2);
        defer t.deinit();
        t.chunk = 4096; // three pieces and a tail
        _ = try put(t, tokens, 600, "meta:600", payload);
        try std.testing.expectEqual(@as(u64, 1), t.info().stats.written);
    }
    const t = try Tier.open(gpa, io, root.settings(1 << 20), "flashnext", identity_a, 0, 2);
    defer t.deinit();
    t.chunk = 4096;
    try std.testing.expect(t.best(tokens[0..599], 0, {}, anyAt) == null); // the prompt must hold every stored token and one more
    const hit = t.best(tokens, 0, {}, anyAt).?;
    try std.testing.expectEqual(@as(u32, 600), hit.name.at);
    try std.testing.expectEqualSlices(u32, tokens[0..600], hit.tokens);
    try std.testing.expect(t.best(tokens, 600, {}, anyAt) == null); // nothing past what memory already holds
    var meta: [64]u8 = undefined;
    const got = try readBack(t, hit.name, &meta);
    defer gpa.free(got);
    try std.testing.expectEqualSlices(u8, payload, got);
    try std.testing.expectEqualStrings("meta:600", meta[0..8]);
    const info = t.info();
    try std.testing.expectEqual(@as(u64, 1), info.stats.loaded);
    try std.testing.expectEqual(@as(u64, 600), info.stats.loaded_tokens);
    // the file: a 4096-byte header, 700 x 4 token bytes padded to 4096, then the payload
    try std.testing.expectEqual(@as(u64, 4096 + 4096 + 10_000), info.stored_bytes);
}

test "a cut write leaves only a partial file, deleted at the next start; a file whose size disagrees is deleted" {
    if (@import("builtin").os.tag == .windows) return error.SkipZigTest; // POSIX files, modes, links or paths
    var root = try Root.make();
    defer root.deinit();
    var tok: [100]u32 = undefined;
    const tokens = tokensOf(&tok, 1);
    var pay: [5000]u8 = undefined;
    const payload = payloadOf(&pay, 2);
    var name: Name = undefined;
    {
        const t = try Tier.open(gpa, io, root.settings(1 << 20), "flashnext", identity_a, 0, 1);
        defer t.deinit();
        name = try put(t, tokens, 90, "", payload);
    }
    var p: [512]u8 = undefined;
    const path = root.file(&p, identity_a, 0, name);
    var q: [600]u8 = undefined;
    const partial = try std.fmt.bufPrintSentinel(&q, "{s}.7.partial", .{path}, 0);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = partial, .data = "a write the process never finished" });
    // the stored file loses its last byte (a disk that lied about a sync)
    const fd = std.c.open(path, .{ .ACCMODE = .WRONLY }, @as(std.c.mode_t, 0));
    try std.testing.expect(fd >= 0);
    try std.testing.expectEqual(@as(c_int, 0), std.c.ftruncate(fd, @intCast(4096 + 4096 + 5000 - 1)));
    _ = std.c.close(fd);
    const t = try Tier.open(gpa, io, root.settings(1 << 20), "flashnext", identity_a, 0, 1);
    defer t.deinit();
    try std.testing.expect(t.best(tokens, 0, {}, anyAt) == null);
    try std.testing.expectError(error.FileNotFound, std.Io.Dir.cwd().access(io, partial, .{}));
    try std.testing.expectError(error.FileNotFound, std.Io.Dir.cwd().access(io, path, .{}));
    try std.testing.expectEqual(@as(u64, 1), t.info().stats.pruned);
}

extern "c" fn chown(path: [*:0]const u8, owner: std.c.uid_t, group: std.c.gid_t) c_int;

test "a spill directory that is a link, another user's, or writable by others is refused at start" {
    if (@import("builtin").os.tag == .windows) return error.SkipZigTest; // POSIX files, modes, links or paths
    var root = try Root.make();
    defer root.deinit();
    const s = root.settings(1 << 20);
    var rp: [512]u8 = undefined;
    const top_dir = try std.fmt.bufPrintSentinel(&rp, "{s}", .{root.path}, 0);
    var tp: [512]u8 = undefined;
    const top = try std.fmt.bufPrintSentinel(&tp, "{s}/flashnext-{x}", .{ root.path, identity_a[0..8] }, 0);
    var kp: [512]u8 = undefined;
    const rank = try std.fmt.bufPrintSentinel(&kp, "{s}/rank0", .{top}, 0);
    var mp: [512]u8 = undefined;
    const moved = try std.fmt.bufPrintSentinel(&mp, "{s}.real", .{top}, 0);
    var np: [64]u8 = undefined;
    (try Tier.open(gpa, io, s, "flashnext", identity_a, 0, 1)).deinit();
    // the rank's directory writable by every user
    try std.testing.expectEqual(@as(c_int, 0), std.c.chmod(rank, 0o777));
    try std.testing.expectError(error.SpillDirUnsafe, Tier.open(gpa, io, s, "flashnext", identity_a, 0, 1));
    try std.testing.expectEqual(@as(c_int, 0), std.c.chmod(rank, 0o700));
    // the spill root writable by its group
    try std.testing.expectEqual(@as(c_int, 0), std.c.chmod(top_dir, 0o770));
    try std.testing.expectError(error.SpillDirUnsafe, Tier.open(gpa, io, s, "flashnext", identity_a, 0, 1));
    try std.testing.expectEqual(@as(c_int, 0), std.c.chmod(top_dir, 0o700));
    // the identity's directory a link to a private directory elsewhere
    try std.testing.expectEqual(@as(c_int, 0), std.c.rename(top, moved));
    try std.testing.expectEqual(@as(c_int, 0), std.c.symlink(try std.fmt.bufPrintSentinel(&np, "flashnext-{x}.real", .{identity_a[0..8]}, 0), top));
    try std.testing.expectError(error.SpillDirUnsafe, Tier.open(gpa, io, s, "flashnext", identity_a, 0, 1));
    try std.testing.expectEqual(@as(c_int, 0), std.c.unlink(top));
    try std.testing.expectEqual(@as(c_int, 0), std.c.rename(moved, top));
    // another user's (root alone may hand it over: elsewhere this part is skipped)
    if (chown(rank, 4321, 4321) == 0) {
        try std.testing.expectError(error.SpillDirUnsafe, Tier.open(gpa, io, s, "flashnext", identity_a, 0, 1));
        try std.testing.expectEqual(@as(c_int, 0), chown(rank, std.c.geteuid(), std.c.getegid()));
    }
    (try Tier.open(gpa, io, s, "flashnext", identity_a, 0, 1)).deinit();
}

test "a write never follows a link planted where a partial file or the start stamp goes" {
    if (@import("builtin").os.tag == .windows) return error.SkipZigTest; // POSIX files, modes, links or paths
    var root = try Root.make();
    defer root.deinit();
    var tok: [40]u32 = undefined;
    const tokens = tokensOf(&tok, 4);
    var pay: [1000]u8 = undefined;
    const payload = payloadOf(&pay, 5);
    const outside = "not the tier's";
    var vp: [512]u8 = undefined;
    const victim = try std.fmt.bufPrintSentinel(&vp, "{s}/victim", .{std.fs.path.dirname(root.path).?}, 0);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = victim, .data = outside });
    var ap: [std.c.PATH_MAX]u8 = undefined;
    const target = std.c.realpath(victim, &ap) orelse return error.NoVictim;
    const name = Name.of(tokens[0..30], 30);
    {
        const t = try Tier.open(gpa, io, root.settings(1 << 20), "flashnext", identity_a, 0, 1);
        defer t.deinit();
        // links at the names a write's partial file took when they were `<file>.<n>.partial`
        var fp: [512]u8 = undefined;
        const final = root.file(&fp, identity_a, 0, name);
        for (1..5) |n| {
            var lp: [600]u8 = undefined;
            try std.testing.expectEqual(@as(c_int, 0), std.c.symlink(target, try std.fmt.bufPrintSentinel(&lp, "{s}.{d}.partial", .{ final, n }, 0)));
        }
        try std.testing.expect(name.eql(try put(t, tokens, 30, "", payload)));
        var meta: [64]u8 = undefined;
        const back = try readBack(t, name, &meta);
        defer gpa.free(back);
        try std.testing.expectEqualSlices(u8, payload, back);
    }
    var buf: [64]u8 = undefined;
    try std.testing.expectEqualStrings(outside, try std.Io.Dir.cwd().readFile(io, victim, &buf));
    // the stamp a start writes, a link now
    var sp: [512]u8 = undefined;
    const stamp = try std.fmt.bufPrintSentinel(&sp, "{s}/flashnext-{x}/opened", .{ root.path, identity_a[0..8] }, 0);
    try std.testing.expectEqual(@as(c_int, 0), std.c.unlink(stamp));
    try std.testing.expectEqual(@as(c_int, 0), std.c.symlink(target, stamp));
    const t = try Tier.open(gpa, io, root.settings(1 << 20), "flashnext", identity_a, 0, 1);
    defer t.deinit();
    try std.testing.expect(t.has(name));
    try std.testing.expectEqualStrings(outside, try std.Io.Dir.cwd().readFile(io, victim, &buf));
}

test "another identity's or another rank's file is never read: a copy into this directory is deleted at start" {
    var root = try Root.make();
    defer root.deinit();
    var tok: [100]u32 = undefined;
    const tokens = tokensOf(&tok, 4);
    var pay: [3000]u8 = undefined;
    const payload = payloadOf(&pay, 9);
    var name: Name = undefined;
    {
        const t = try Tier.open(gpa, io, root.settings(1 << 20), "flashnext", identity_b, 1, 2);
        defer t.deinit();
        name = try put(t, tokens, 80, "", payload);
    }
    var a: [512]u8 = undefined;
    var b: [512]u8 = undefined;
    const from = root.file(&a, identity_b, 1, name);
    for ([_]struct { id: [32]u8, rank: u32 }{ .{ .id = identity_a, .rank = 1 }, .{ .id = identity_b, .rank = 0 } }) |c| {
        const t0 = try Tier.open(gpa, io, root.settings(1 << 20), "flashnext", c.id, c.rank, 2);
        t0.deinit(); // makes the directory
        const to = root.file(&b, c.id, c.rank, name);
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, from, gpa, .limited(1 << 20));
        defer gpa.free(bytes);
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = to, .data = bytes });
        const t = try Tier.open(gpa, io, root.settings(1 << 20), "flashnext", c.id, c.rank, 2);
        defer t.deinit();
        try std.testing.expect(t.best(tokens, 0, {}, anyAt) == null);
        try std.testing.expectError(error.FileNotFound, std.Io.Dir.cwd().access(io, to, .{}));
    }
    // its own rank and identity still read it
    const t = try Tier.open(gpa, io, root.settings(1 << 20), "flashnext", identity_b, 1, 2);
    defer t.deinit();
    try std.testing.expectEqual(@as(u32, 80), t.best(tokens, 0, {}, anyAt).?.name.at);
}

test "a corrupt payload byte fails its read after the copy, and the file goes" {
    if (@import("builtin").os.tag == .windows) return error.SkipZigTest; // POSIX files, modes, links or paths
    var root = try Root.make();
    defer root.deinit();
    var tok: [50]u32 = undefined;
    const tokens = tokensOf(&tok, 8);
    var pay: [9000]u8 = undefined;
    const payload = payloadOf(&pay, 1);
    const t = try Tier.open(gpa, io, root.settings(1 << 20), "flashnext", identity_a, 0, 1);
    defer t.deinit();
    const name = try put(t, tokens, 40, "", payload);
    var p: [512]u8 = undefined;
    const path = root.file(&p, identity_a, 0, name);
    const fd = std.c.open(path, .{ .ACCMODE = .RDWR }, @as(std.c.mode_t, 0));
    try std.testing.expect(fd >= 0);
    const flip = [1]u8{payload[5000] ^ 0x40};
    try std.testing.expectEqual(@as(isize, 1), std.c.pwrite(fd, &flip, 1, 4096 + 4096 + 5000));
    _ = std.c.close(fd);
    var meta: [64]u8 = undefined;
    try std.testing.expectError(error.PayloadHash, readBack(t, name, &meta));
    try std.testing.expect(!t.has(name));
    try std.testing.expectError(error.FileNotFound, std.Io.Dir.cwd().access(io, path, .{}));
    const info = t.info();
    try std.testing.expectEqual(@as(u64, 1), info.stats.hash_failed);
    try std.testing.expectEqual(@as(u64, 1), info.stats.load_failed);
}

test "the cap evicts the least recently written or read state first, and the owner hears of each" {
    var root = try Root.make();
    defer root.deinit();
    var tok: [3][40]u32 = undefined;
    var pay: [1000]u8 = undefined;
    const payload = payloadOf(&pay, 3);
    const file_bytes: u64 = 4096 + 4096 + 1000;
    const t = try Tier.open(gpa, io, root.settings(2 * file_bytes), "flashnext", identity_a, 0, 1);
    defer t.deinit();
    var forgot: Forgot = .{};
    defer forgot.names.deinit(gpa);
    t.hook = forgot.hook();
    const a = try put(t, tokensOf(&tok[0], 1), 30, "", payload);
    const b = try put(t, tokensOf(&tok[1], 2), 30, "", payload);
    t.touch(a); // a read: a is now the newer of the two
    const c = try put(t, tokensOf(&tok[2], 3), 30, "", payload);
    try std.testing.expect(t.has(a) and t.has(c) and !t.has(b));
    try std.testing.expectEqual(@as(usize, 1), forgot.names.items.len);
    try std.testing.expect(forgot.names.items[0].eql(b));
    // a state larger than the whole cap evicts nothing
    var big: [3 * 8192]u8 = undefined;
    var m: Mem = .{ .bytes = payloadOf(&big, 1) };
    try std.testing.expectError(error.LargerThanCap, t.write(Name.of(tok[0][0..31], 31), tok[0][0..31], "", big.len, m.source()));
    try std.testing.expect(t.has(a) and t.has(c));
    try std.testing.expectEqual(@as(u64, 1), t.info().stats.full);
}

/// A disk of `disk_bytes` that holds only the tier's directory: its free bytes as statvfs reports them. A file
/// unlinked while this process still has it open keeps its blocks until the last descriptor closes.
var disk_bytes: u64 = 0;
fn modelFree(dir: [:0]const u8) u64 {
    var used: u64 = 0;
    var rp: [std.c.PATH_MAX]u8 = undefined;
    const real = std.mem.sliceTo(std.c.realpath(dir, &rp) orelse return 0, 0);
    if (std.c.opendir(dir)) |d| {
        defer _ = std.c.closedir(d);
        while (std.c.readdir(d)) |ent| {
            if (ent.type != std.c.DT.REG) continue;
            var p: [1200]u8 = undefined;
            used += sizeAt(std.fmt.bufPrintSentinel(&p, "{s}/{s}", .{ dir, std.mem.sliceTo(&ent.name, 0) }, 0) catch continue);
        }
    }
    const fds = std.c.opendir("/proc/self/fd") orelse return 0;
    defer _ = std.c.closedir(fds);
    while (std.c.readdir(fds)) |ent| {
        const n = std.mem.sliceTo(&ent.name, 0);
        const fd = std.fmt.parseInt(c_int, n, 10) catch continue;
        var lp: [64]u8 = undefined;
        var target: [std.c.PATH_MAX]u8 = undefined;
        const len = std.c.readlink(std.fmt.bufPrintSentinel(&lp, "/proc/self/fd/{s}", .{n}, 0) catch continue, &target, target.len);
        if (len <= 0) continue;
        const t = target[0..@intCast(len)];
        if (!std.mem.startsWith(u8, t, real) or !std.mem.endsWith(u8, t, " (deleted)")) continue;
        used += sizeOf(fd);
    }
    return disk_bytes -| used;
}

fn sizeAt(path: [:0]const u8) u64 {
    const fd = std.c.open(path, .{ .ACCMODE = .RDONLY }, @as(std.c.mode_t, 0));
    if (fd < 0) return 0;
    defer _ = std.c.close(fd);
    return sizeOf(fd);
}

fn sizeOf(fd: c_int) u64 {
    const end = std.c.lseek(fd, 0, std.c.SEEK.END); // the tier reads with pread: the offset is nobody's
    return if (end > 0) @intCast(end) else 0;
}

test "the free-disk floor: stored states go until a write leaves the floor free, else the write is refused" {
    if (@import("builtin").os.tag == .windows) return error.SkipZigTest; // POSIX files, modes, links or paths
    var root = try Root.make();
    defer root.deinit();
    var tok: [3][40]u32 = undefined;
    var pay: [1000]u8 = undefined;
    const payload = payloadOf(&pay, 3);
    const file_bytes: u64 = 4096 + 4096 + 1000;
    var s = root.settings(1 << 30);
    s.min_free = 100_000;
    const t = try Tier.open(gpa, io, s, "flashnext", identity_a, 0, 1);
    defer t.deinit();
    t.free_space = modelFree;
    disk_bytes = 1 << 30;
    const a = try put(t, tokensOf(&tok[0], 1), 30, "", payload);
    const b = try put(t, tokensOf(&tok[1], 2), 30, "", payload);
    // the disk now has room for the floor and one more file only after a's bytes come back
    disk_bytes = 100_000 + 3 * file_bytes - 1;
    try std.testing.expectEqual(@as(u64, 100_000 + file_bytes - 1), modelFree(t.dir));
    const c = try put(t, tokensOf(&tok[2], 3), 30, "", payload);
    try std.testing.expect(!t.has(a) and t.has(b) and t.has(c));
    // with every stored state gone it still would not leave the floor free: refused, nothing more goes
    disk_bytes = 0;
    var m: Mem = .{ .bytes = payload };
    try std.testing.expectError(error.NoSpace, t.write(Name.of(tok[0][0..29], 29), tok[0][0..29], "", payload.len, m.source()));
    try std.testing.expectEqual(@as(u64, 1), t.info().stats.no_space);
    try std.testing.expect(t.has(b) and t.has(c));
}

test "the free-disk floor counts only the bytes a deletion gives back: a state being read keeps its blocks" {
    if (@import("builtin").os.tag == .windows) return error.SkipZigTest; // POSIX files, modes, links or paths
    var root = try Root.make();
    defer root.deinit();
    var tok: [2][40]u32 = undefined;
    var pay: [1000]u8 = undefined;
    const payload = payloadOf(&pay, 3);
    const file_bytes: u64 = 4096 + 4096 + 1000;
    var s = root.settings(1 << 30);
    s.min_free = 100_000;
    const t = try Tier.open(gpa, io, s, "flashnext", identity_a, 0, 1);
    defer t.deinit();
    t.free_space = modelFree;
    // room for the floor and two files: a second file fits only once the first's blocks are free
    disk_bytes = 100_000 + 2 * file_bytes - 1;
    const a = try put(t, tokensOf(&tok[0], 1), 30, "", payload);
    // a request reads a back: its file stays open until the read closes, so deleting it frees nothing yet
    var r = try t.read(a);
    var m: Mem = .{ .bytes = payload };
    const later = tokensOf(&tok[1], 2);
    const b = Name.of(later[0..30], 30);
    try std.testing.expectError(error.NoSpace, t.write(b, later[0..30], "", payload.len, m.source()));
    try std.testing.expectEqual(@as(u64, 100_000 + file_bytes - 1), modelFree(t.dir));
    try std.testing.expect(!t.has(b));
    r.close();
    try std.testing.expectEqual(@as(u64, 100_000 + 2 * file_bytes - 1), modelFree(t.dir));
    try std.testing.expect(b.eql(try put(t, later, 30, "", payload)));
}

test "a source that fails mid-copy leaves no file and no entry; a write forgotten on its way is never renamed" {
    var root = try Root.make();
    defer root.deinit();
    var tok: [60]u32 = undefined;
    const tokens = tokensOf(&tok, 6);
    var pay: [20_000]u8 = undefined;
    const payload = payloadOf(&pay, 4);
    const t = try Tier.open(gpa, io, root.settings(1 << 20), "flashnext", identity_a, 0, 1);
    defer t.deinit();
    t.chunk = 4096;
    var m: Mem = .{ .bytes = payload, .fail_at = 9000 };
    const name = Name.of(tokens[0..50], 50);
    const j = try t.write(name, tokens[0..50], "", payload.len, m.source());
    j.waitCopied(); // a failed copy releases its source too
    try std.testing.expect(t.drain(10));
    try std.testing.expect(!j.ok);
    j.release();
    try std.testing.expect(!t.has(name));
    try std.testing.expectEqual(@as(u64, 1), t.info().stats.write_failed);
    // forgotten while queued behind a slow one: cut, no file
    t.forget(name);
    var m2: Mem = .{ .bytes = payload };
    const j2 = try t.write(name, tokens[0..50], "", payload.len, m2.source());
    t.forget(name);
    try std.testing.expect(t.drain(10));
    j2.release();
    var p: [512]u8 = undefined;
    try std.testing.expectError(error.FileNotFound, std.Io.Dir.cwd().access(io, root.file(&p, identity_a, 0, name), .{}));
    try std.testing.expectEqual(@as(usize, 0), t.info().entries);
}

test "a start keeps the newest other builds' directories and deletes the older ones' files" {
    var root = try Root.make();
    defer root.deinit();
    var tok: [40]u32 = undefined;
    const tokens = tokensOf(&tok, 2);
    var pay: [100]u8 = undefined;
    const payload = payloadOf(&pay, 1);
    const ids = [_][32]u8{ @splat(1), @splat(2), @splat(3) };
    var keep_all = root.settings(1 << 20);
    keep_all.keep_builds = 10;
    for (ids, 0..) |id, i| {
        const t = try Tier.open(gpa, io, keep_all, "flashnext", id, 0, 1);
        _ = try put(t, tokens, 30, "", payload);
        t.deinit();
        // the identities' last starts, oldest first
        var p: [512]u8 = undefined;
        const stamp = try std.fmt.bufPrintSentinel(&p, "{s}/flashnext-{x}/opened", .{ root.path, id[0..8] }, 0);
        var w: [8]u8 = undefined;
        std.mem.writeInt(u64, &w, 100 + i, .little);
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = stamp, .data = &w });
    }
    const t = try Tier.open(gpa, io, root.settings(1 << 20), "flashnext", identity_a, 0, 1); // keep_builds 1
    defer t.deinit();
    try std.testing.expectEqual(@as(u64, 2), t.info().stats.pruned);
    for (ids, [_]bool{ false, false, true }) |id, kept| {
        var p: [512]u8 = undefined;
        const dir = try std.fmt.bufPrintSentinel(&p, "{s}/flashnext-{x}", .{ root.path, id[0..8] }, 0);
        const there = if (std.Io.Dir.cwd().access(io, dir, .{})) |_| true else |_| false;
        try std.testing.expectEqual(kept, there);
    }
}

test "settings come from TENSORFOLD_SPILL_*: off without a size, GLM's defaults, bad values refused" {
    if (@import("builtin").os.tag == .windows) return error.SkipZigTest; // POSIX files, modes, links or paths
    const Env = struct {
        vars: []const [2][]const u8,
        pub fn get(e: @This(), name: [:0]const u8) ?[]const u8 {
            for (e.vars) |v| if (std.mem.eql(u8, v[0], name)) return v[1];
            return null;
        }
    };
    try std.testing.expect((try spill.Settings.fromEnv(gpa, Env{ .vars = &.{.{ "HOME", "/home/user" }} })) == null);
    try std.testing.expect((try spill.Settings.fromEnv(gpa, Env{ .vars = &.{ .{ "TENSORFOLD_SPILL_GIB", "0" }, .{ "HOME", "/home/user" } } })) == null);
    const s = (try spill.Settings.fromEnv(gpa, Env{ .vars = &.{ .{ "TENSORFOLD_SPILL_GIB", "200" }, .{ "HOME", "/home/user" } } })).?;
    defer gpa.free(s.dir);
    try std.testing.expectEqualStrings("/home/user/.cache/tensorfold/spill", s.dir);
    try std.testing.expectEqual(@as(u64, 200 << 30), s.cap);
    try std.testing.expectEqual(@as(f64, 0.70), s.highwater);
    try std.testing.expectEqual(@as(u32, 8192), s.min_tokens);
    try std.testing.expectEqual(@as(u64, 50 << 30), s.min_free);
    try std.testing.expectEqual(@as(f64, 60), s.flush_s);
    var buf: [128]u8 = undefined;
    try std.testing.expectEqualStrings("spill=204800/0.7/8192/51200/60/1/0", try s.text(&buf));
    const t = (try spill.Settings.fromEnv(gpa, Env{ .vars = &.{ .{ "TENSORFOLD_SPILL_GIB", "1.5" }, .{ "TENSORFOLD_SPILL_DIR", "/nvme/spill" }, .{ "TENSORFOLD_SPILL_HIGHWATER", "1.0" }, .{ "TENSORFOLD_SPILL_MIN_TOKENS", "4096" }, .{ "TENSORFOLD_SPILL_MIN_FREE_GIB", "10" }, .{ "TENSORFOLD_SPILL_FLUSH_S", "30" } } })).?;
    defer gpa.free(t.dir);
    try std.testing.expectEqualStrings("/nvme/spill", t.dir);
    try std.testing.expectEqual(@as(u64, 3 << 29), t.cap);
    try std.testing.expectEqual(@as(u32, 4096), t.min_tokens);
    try std.testing.expectEqual(@as(u64, 10 << 30), t.min_free);
    try std.testing.expectError(error.BadSpillHighwater, spill.Settings.fromEnv(gpa, Env{ .vars = &.{ .{ "TENSORFOLD_SPILL_GIB", "1" }, .{ "TENSORFOLD_SPILL_HIGHWATER", "1.5" }, .{ "HOME", "/h" } } }));
    try std.testing.expectError(error.SpillDirNotAbsolute, spill.Settings.fromEnv(gpa, Env{ .vars = &.{ .{ "TENSORFOLD_SPILL_GIB", "1" }, .{ "TENSORFOLD_SPILL_DIR", "relative/dir" } } }));
    try std.testing.expectError(error.BadSpillSetting, spill.Settings.fromEnv(gpa, Env{ .vars = &.{ .{ "TENSORFOLD_SPILL_GIB", "lots" }, .{ "HOME", "/h" } } }));
}

test "a header round-trips and refuses a flipped byte, another version, identity, rank or world" {
    const h: spill.Header = .{ .identity = identity_a, .family = @splat(0), .rank = 1, .world = 2, .name = .{ .key = 0x1234, .at = 99 }, .tokens = 100, .payload_len = 777, .tokens_hash = 5, .payload_hash = 6, .saved = 1_760_000_000, .meta_len = 3 };
    var buf: [spill.header_bytes]u8 = undefined;
    h.encode("abc", &buf);
    const back = try spill.Header.decode(&buf, identity_a, 1, 2);
    try std.testing.expectEqual(@as(u64, 777), back.payload_len);
    try std.testing.expectEqual(@as(u32, 99), back.name.at);
    try std.testing.expectEqualStrings("abc", spill.Header.meta(&buf, back));
    try std.testing.expectEqual(@as(u64, 4096 + 4096 + 777), back.fileBytes());
    try std.testing.expectError(error.OtherIdentity, spill.Header.decode(&buf, identity_b, 1, 2));
    try std.testing.expectError(error.OtherRank, spill.Header.decode(&buf, identity_a, 0, 2));
    try std.testing.expectError(error.OtherWorld, spill.Header.decode(&buf, identity_a, 1, 4)); // a TP=4 tier never reads TP=2 shards
    buf[100] ^= 1;
    try std.testing.expectError(error.HeaderHash, spill.Header.decode(&buf, identity_a, 1, 2));
    buf[100] ^= 1;
    buf[8] = 2;
    try std.testing.expectError(error.OtherVersion, spill.Header.decode(&buf, identity_a, 1, 2));
}

test "a hurried write goes before the ones queued ahead of it" {
    var root = try Root.make();
    defer root.deinit();
    const Gated = struct {
        open: *std.atomic.Value(bool),
        started: *std.atomic.Value(u32),
        order: *std.ArrayList(u8),
        tag: u8,
        fn read(ptr: *anyopaque, offset: u64, out: []u8) anyerror!void {
            const g: *@This() = @ptrCast(@alignCast(ptr));
            if (offset == 0) {
                try g.order.append(gpa, g.tag);
                _ = g.started.fetchAdd(1, .release);
            }
            while (!g.open.load(.acquire)) std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
            @memset(out, g.tag);
        }
    };
    var open = std.atomic.Value(bool).init(false);
    var started = std.atomic.Value(u32).init(0);
    var order: std.ArrayList(u8) = .empty;
    defer order.deinit(gpa);
    const t = try Tier.open(gpa, io, root.settings(1 << 20), "flashnext", identity_a, 0, 1);
    defer t.deinit();
    var tok: [3][40]u32 = undefined;
    var g: [3]Gated = undefined;
    var jobs: [3]*spill.Job = undefined;
    for (&g, &jobs, &tok, 0..) |*x, *j, *tk, i| {
        x.* = .{ .open = &open, .started = &started, .order = &order, .tag = @intCast('a' + i) };
        j.* = try t.write(Name.of(tokensOf(tk, @intCast(i))[0..30], 30), tk[0..30], "", 100, .{ .ptr = x, .read = Gated.read });
    }
    while (started.load(.acquire) == 0) std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {}; // the writer is in a's read
    t.hurry(jobs[2]);
    open.store(true, .release);
    try std.testing.expect(t.drain(10));
    for (jobs) |j| j.release();
    try std.testing.expectEqualStrings("acb", order.items);
}

const FullDisk = struct {
    var calls: usize = 0;
    var saw_nospc: bool = false;

    fn write(fd: c_int, bytes: []const u8, at: u64) !void {
        calls += 1;
        if (calls == 3) { // tokens and one payload chunk have reached the partial file
            const full = std.c.open("/dev/full", .{ .ACCMODE = .WRONLY }, @as(std.c.mode_t, 0));
            if (full < 0) return error.FullDeviceUnavailable;
            defer _ = std.c.close(full);
            const rc = std.c.pwrite(full, bytes.ptr, bytes.len, @intCast(at));
            saw_nospc = rc < 0 and std.posix.errno(rc) == .NOSPC;
            return if (saw_nospc) error.SpillWrite else error.ExpectedDiskFull;
        }
        var done: usize = 0;
        while (done < bytes.len) {
            const n = std.c.pwrite(fd, bytes.ptr + done, bytes.len - done, @intCast(at + done));
            if (n <= 0) return error.SpillWrite;
            done += @intCast(n);
        }
    }
};

test "actual injected ENOSPC after a payload chunk releases the source and removes the partial write" {
    if (@import("builtin").os.tag == .windows) return error.SkipZigTest; // POSIX files, modes, links or paths
    if (@import("builtin").os.tag != .linux) return error.SkipZigTest;
    var root = try Root.make();
    defer root.deinit();
    var tok: [60]u32 = undefined;
    const tokens = tokensOf(&tok, 6);
    var pay: [20_000]u8 = undefined;
    const payload = payloadOf(&pay, 4);
    const name = Name.of(tokens[0..50], 50);
    {
        const t = try Tier.open(gpa, io, root.settings(1 << 20), "flashnext", identity_a, 0, 1);
        defer t.deinit();
        t.chunk = 4096;
        FullDisk.calls = 0;
        FullDisk.saw_nospc = false;
        t.write_file = FullDisk.write;
        var m: Mem = .{ .bytes = payload }; // Source.read always succeeds: this is disk I/O failure
        const j = try t.write(name, tokens[0..50], "", payload.len, m.source());
        defer j.release();
        j.waitCopied();
        try std.testing.expect(t.drain(10));
        try std.testing.expect(FullDisk.saw_nospc);
        try std.testing.expectEqual(@as(usize, 3), FullDisk.calls);
        try std.testing.expect(!j.ok);
        try std.testing.expect(j.copied());
        try std.testing.expect(!t.has(name));
        try std.testing.expectEqual(@as(usize, 0), t.info().entries);
        try std.testing.expectEqual(@as(u64, 0), t.info().stored_bytes);
        try std.testing.expectEqual(@as(u64, 1), t.info().stats.write_failed);
        var pb: [512]u8 = undefined;
        try std.testing.expectError(error.FileNotFound, std.Io.Dir.cwd().access(io, root.file(&pb, identity_a, 0, name), .{}));
        // No orphan partial survives while the process is still running (not just restart cleanup).
        const dir = std.c.opendir(t.dir) orelse return error.DirectoryUnavailable;
        defer _ = std.c.closedir(dir);
        while (std.c.readdir(dir)) |entry| {
            try std.testing.expect(!std.mem.endsWith(u8, std.mem.sliceTo(&entry.name, 0), ".partial"));
        }
    }
    // The failed job made no durable index entry, and the same name can be stored and read normally afterwards.
    const t = try Tier.open(gpa, io, root.settings(1 << 20), "flashnext", identity_a, 0, 1);
    defer t.deinit();
    try std.testing.expectEqual(@as(usize, 0), t.info().entries);
    _ = try put(t, tokens, 50, "", payload);
    var meta: [64]u8 = undefined;
    const got = try readBack(t, name, &meta);
    defer gpa.free(got);
    try std.testing.expectEqualSlices(u8, payload, got);
}

// The writer is held by a real source gate. If the old reader tries to wait,
// intercept that exact reader condition wait and release the writer, so the
// regression fails at an assertion rather than depending on a timeout.
const QueuedWriter = struct {
    mutex: std.Io.Mutex = .init,
    cond: std.Io.Condition = .init,
    entered: bool = false,
    released: bool = false,
    tier: ?*Tier = null,
    reader_id: std.Thread.Id,
    read_waits: usize = 0,
    var active: ?*QueuedWriter = null;
    fn copy(ptr: *anyopaque, _: u64, out: []u8) anyerror!void {
        const q: *QueuedWriter = @ptrCast(@alignCast(ptr));
        q.mutex.lockUncancelable(io);
        defer q.mutex.unlock(io);
        q.entered = true;
        q.cond.broadcast(io);
        while (!q.released) q.cond.waitUncancelable(io, &q.mutex);
        @memset(out, 0x5a);
    }
    fn enteredGate(q: *QueuedWriter) !void {
        q.mutex.lockUncancelable(io);
        defer q.mutex.unlock(io);
        const deadline = std.Io.Clock.awake.now(io).toNanoseconds() + 2 * std.time.ns_per_s;
        while (!q.entered) {
            if (std.Io.Clock.awake.now(io).toNanoseconds() >= deadline) return error.WriterDidNotEnterGate;
            q.cond.waitTimeout(io, &q.mutex, .{ .duration = .{ .raw = .fromNanoseconds(10 * std.time.ns_per_ms), .clock = .awake } }) catch {};
        }
    }
    fn release(q: *QueuedWriter) void {
        q.mutex.lockUncancelable(io);
        defer q.mutex.unlock(io);
        q.released = true;
        q.cond.broadcast(io);
    }
    fn wait(userdata: ?*anyopaque, ptr: *const u32, expected: u32) void {
        if (active) |q| if (q.tier) |t| {
            if (std.Thread.getCurrentId() == q.reader_id and ptr == &t.cond.epoch.raw) {
                q.read_waits += 1;
                q.release();
            }
        };
        io.vtable.futexWaitUncancelable(userdata, ptr, expected);
    }
};
test "a queued follower copy returns NotStored without waiting for its blocked writer" {
    var root = try Root.make();
    defer root.deinit();
    var gate: QueuedWriter = .{ .reader_id = std.Thread.getCurrentId() };
    var vtable = io.vtable.*;
    vtable.futexWaitUncancelable = QueuedWriter.wait;
    const probe_io: std.Io = .{ .userdata = io.userdata, .vtable = &vtable };
    const t = try Tier.open(gpa, probe_io, root.settings(1 << 30), "flashnext", identity_a, 1, 2);
    defer t.deinit();
    gate.tier = t;
    QueuedWriter.active = &gate;
    defer QueuedWriter.active = null;
    const tokens = [_]u32{ 1, 2, 3 };
    const name = Name.of(&tokens, tokens.len);
    const job = try t.write(name, &tokens, "meta", 128, .{ .ptr = &gate, .read = QueuedWriter.copy });
    defer job.release();
    defer {
        gate.release();
        job.waitCopied();
        _ = t.drain(10);
    }
    try gate.enteredGate();
    const result = t.read(name);
    if (result) |reading| {
        var r = reading;
        r.close();
    } else |_| {}
    try std.testing.expectEqual(@as(usize, 0), gate.read_waits);
    try std.testing.expectError(error.NotStored, result);
    try std.testing.expect(!gate.released);
    try std.testing.expectEqual(@as(usize, 1), t.info().pending);
    gate.release();
    job.waitCopied();
    try std.testing.expect(t.drain(10));
    try std.testing.expect(job.ok);
    var ready = try t.read(name);
    defer ready.close();
    try std.testing.expectEqualSlices(u32, &tokens, ready.tokens);
}

test "settings refuse non-finite, negative and overflowing numbers before any conversion" {
    if (@import("builtin").os.tag == .windows) return error.SkipZigTest; // POSIX files, modes, links or paths
    const Env = struct {
        vars: []const [2][]const u8,
        pub fn get(e: @This(), name: [:0]const u8) ?[]const u8 {
            for (e.vars) |kv| if (std.mem.eql(u8, kv[0], name)) return kv[1];
            return null;
        }
    };
    const bad = [_][2][]const u8{
        .{ "TENSORFOLD_SPILL_GIB", "nan" },
        .{ "TENSORFOLD_SPILL_GIB", "inf" },
        .{ "TENSORFOLD_SPILL_GIB", "1e300" },
        .{ "TENSORFOLD_SPILL_MIN_FREE_GIB", "nan" },
        .{ "TENSORFOLD_SPILL_MIN_FREE_GIB", "-1" },
        .{ "TENSORFOLD_SPILL_MIN_FREE_GIB", "1e300" },
        .{ "TENSORFOLD_SPILL_FLUSH_S", "inf" },
        .{ "TENSORFOLD_SPILL_FLUSH_S", "-3" },
        .{ "TENSORFOLD_SPILL_FLUSH_S", "1e300" },
        .{ "TENSORFOLD_SPILL_HIGHWATER", "nan" },
    };
    for (bad) |kv| {
        const vars = [_][2][]const u8{ .{ "TENSORFOLD_SPILL_GIB", "1" }, .{ "TENSORFOLD_SPILL_DIR", "/x" }, kv };
        // the later pair wins where the names match: GIB itself is replaced by the bad value
        const got = spill.Settings.fromEnv(gpa, Env{ .vars = if (std.mem.eql(u8, kv[0], "TENSORFOLD_SPILL_GIB")) &.{ kv, .{ "TENSORFOLD_SPILL_DIR", "/x" } } else &vars });
        if (got) |maybe| {
            if (maybe) |st| gpa.free(st.dir);
            return error.TestUnexpectedResult;
        } else |_| {}
    }
}

test "waitDone returns once the write is final, after its source was copied" {
    var root = try Root.make();
    defer root.deinit();
    const t = try Tier.open(gpa, io, root.settings(1 << 20), "toy", identity_a, 0, 1);
    defer t.deinit();
    var tok: [40]u32 = undefined;
    const tokens = tokensOf(&tok, 2);
    var pay: [9000]u8 = undefined;
    var m: Mem = .{ .bytes = payloadOf(&pay, 3) };
    const j = try t.write(Name.of(tokens[0..30], 30), tokens[0..30], "", m.bytes.len, m.source());
    defer j.release();
    j.waitDone();
    try std.testing.expect(j.copied() and j.done() and j.ok);
    j.waitDone(); // already final: returns at once
}

// -- the memory level (TENSORFOLD_SPILL_RAM_GIB) --------------------------------------------------------------------

/// A state handed over whole: its payload copied into tier-owned memory.
fn hold(t: *Tier, tokens: []const u32, at: u32, meta: []const u8, payload: []const u8) !Name {
    const name = Name.of(tokens[0..at], at);
    const owned = try t.ramAlloc(payload.len);
    @memcpy(owned, payload);
    try t.adopt(name, tokens[0..at], meta, owned);
    return name;
}

test "a held state is found and read back byte for byte from memory, and written behind to disk" {
    var root = try Root.make();
    defer root.deinit();
    var tok: [300]u32 = undefined;
    const tokens = tokensOf(&tok, 11);
    var pay: [20_000]u8 = undefined;
    const payload = payloadOf(&pay, 9);
    var s = root.settings(1 << 20);
    s.ram_cap = 1 << 20;
    const t = try Tier.open(gpa, io, s, "flashnext", identity_a, 0, 1);
    defer t.deinit();
    t.chunk = 4096;
    const name = try hold(t, tokens, 200, "held", payload);
    try std.testing.expect(t.has(name));
    try std.testing.expect(t.stored(name, tokens[0..200]));
    const hit = t.best(tokens, 0, {}, anyAt).?;
    try std.testing.expectEqual(@as(u32, 200), hit.name.at);
    var meta: [64]u8 = undefined;
    const got = try readBack(t, name, &meta);
    defer gpa.free(got);
    try std.testing.expectEqualSlices(u8, payload, got);
    try std.testing.expectEqualStrings("held", meta[0..4]);
    try std.testing.expect(t.drain(10));
    const info = t.info();
    try std.testing.expectEqual(@as(u64, 1), info.stats.ram_adopted);
    try std.testing.expectEqual(@as(u64, 1), info.stats.ram_loaded); // memory served it, whatever the writer reached
    try std.testing.expectEqual(@as(usize, 1), info.ram_entries); // still held with its file in place
    try std.testing.expectEqual(@as(u64, 20_000), info.ram_bytes);
    try std.testing.expectEqual(@as(u64, 1), info.stats.written);
    try std.testing.expectEqual(@as(u64, 4096 + 4096 + 20_000), info.stored_bytes);
    try std.testing.expectError(error.AlreadyStored, hold(t, tokens, 200, "again", payload));
}

test "past the memory cap the least recently used held state leaves memory for its file, and a restart finds them all" {
    var root = try Root.make();
    defer root.deinit();
    var tok: [3][100]u32 = undefined;
    var pay: [3][10_000]u8 = undefined;
    var names: [3]Name = undefined;
    var s = root.settings(1 << 20);
    s.ram_cap = 25_000; // two of the three
    {
        const t = try Tier.open(gpa, io, s, "flashnext", identity_a, 0, 1);
        defer t.deinit();
        for (0..3) |i| {
            names[i] = try hold(t, tokensOf(&tok[i], @intCast(100 * i + 1)), 90, "m", payloadOf(&pay[i], @intCast(i + 1)));
            try std.testing.expect(t.drain(10));
        }
        const info = t.info();
        try std.testing.expectEqual(@as(usize, 2), info.ram_entries);
        try std.testing.expectEqual(@as(u64, 20_000), info.ram_bytes);
        try std.testing.expectEqual(@as(u64, 1), info.stats.ram_released);
        try std.testing.expectEqual(@as(usize, 3), info.entries);
        try std.testing.expectEqual(@as(u64, 3), info.stats.written);
        // the oldest went to disk and reads back from there
        var meta: [64]u8 = undefined;
        const got = try readBack(t, names[0], &meta);
        defer gpa.free(got);
        try std.testing.expectEqualSlices(u8, &pay[0], got);
        try std.testing.expectEqual(@as(u64, 0), t.info().stats.ram_loaded);
    }
    const t = try Tier.open(gpa, io, s, "flashnext", identity_a, 0, 1);
    defer t.deinit();
    for (names) |n| try std.testing.expect(t.has(n)); // every one was written behind
    try std.testing.expectEqual(@as(usize, 0), t.info().ram_entries); // memory went with the process
}

test "a held state outlives an abrupt end: its file was written behind before the process went" {
    var root = try Root.make();
    defer root.deinit();
    var tok: [100]u32 = undefined;
    var pay: [10_000]u8 = undefined;
    var s = root.settings(1 << 20);
    s.ram_cap = 1 << 20;
    const name = blk: {
        const t = try Tier.open(gpa, io, s, "flashnext", identity_a, 0, 1);
        defer t.deinit(); // no flushRam: as a kill leaves it, after the writer's queue emptied
        const n = try hold(t, tokensOf(&tok, 7), 90, "m", payloadOf(&pay, 7));
        try std.testing.expect(t.drain(10));
        break :blk n;
    };
    const t = try Tier.open(gpa, io, s, "flashnext", identity_a, 0, 1);
    defer t.deinit();
    var meta: [64]u8 = undefined;
    const got = try readBack(t, name, &meta);
    defer gpa.free(got);
    try std.testing.expectEqualSlices(u8, &pay, got);
}

test "a held state the disk has no room for leaves memory, and the owner hears of it" {
    var root = try Root.make();
    defer root.deinit();
    var tok: [2][100]u32 = undefined;
    var pay: [2][10_000]u8 = undefined;
    var s = root.settings(4096); // smaller than any file
    s.ram_cap = 15_000;
    const t = try Tier.open(gpa, io, s, "flashnext", identity_a, 0, 1);
    defer t.deinit();
    var f: Forgot = .{};
    defer f.names.deinit(gpa);
    t.setHook(f.hook());
    const first = try hold(t, tokensOf(&tok[0], 1), 90, "m", payloadOf(&pay[0], 1));
    _ = try hold(t, tokensOf(&tok[1], 2), 90, "m", payloadOf(&pay[1], 2));
    try std.testing.expect(!t.has(first));
    try std.testing.expectEqual(@as(usize, 1), f.names.items.len);
    try std.testing.expect(f.names.items[0].eql(first));
    try std.testing.expectEqual(@as(u64, 1), t.info().stats.ram_dropped);
}

test "a held state read while it goes on to disk reads whole; forgetting one on its way leaves no file" {
    var root = try Root.make();
    defer root.deinit();
    var tok: [3][100]u32 = undefined;
    var pay: [3][10_000]u8 = undefined;
    var s = root.settings(1 << 20);
    s.ram_cap = 15_000;
    const t = try Tier.open(gpa, io, s, "flashnext", identity_a, 0, 1);
    defer t.deinit();
    const a = try hold(t, tokensOf(&tok[0], 1), 90, "m", payloadOf(&pay[0], 1));
    var r = try t.read(a); // a reader holds a's memory
    _ = try hold(t, tokensOf(&tok[1], 2), 90, "m", payloadOf(&pay[1], 2)); // a goes on to disk
    try std.testing.expect(t.drain(10));
    var c: Collect = .{};
    defer c.out.deinit(gpa);
    try r.pump(c.sink());
    r.close();
    try std.testing.expectEqualSlices(u8, &pay[0], c.out.items);
    // b goes on to disk for c's room and is forgotten at once: whatever the writer reached, no file stays
    const b = Name.of(tok[1][0..90], 90);
    _ = try hold(t, tokensOf(&tok[2], 3), 90, "m", payloadOf(&pay[2], 3));
    t.forget(b);
    try std.testing.expect(t.drain(10));
    try std.testing.expect(!t.has(b));
    var p: [512]u8 = undefined;
    try std.testing.expectError(error.FileNotFound, std.Io.Dir.cwd().access(io, root.file(&p, identity_a, 0, b), .{}));
}

test "a clean stop writes every held state, and the next start finds them all" {
    var root = try Root.make();
    defer root.deinit();
    var tok: [3][100]u32 = undefined;
    var pay: [3][10_000]u8 = undefined;
    var names: [3]Name = undefined;
    var s = root.settings(1 << 20);
    s.ram_cap = 1 << 20;
    {
        const t = try Tier.open(gpa, io, s, "flashnext", identity_a, 0, 1);
        defer t.deinit();
        for (0..3) |i| names[i] = try hold(t, tokensOf(&tok[i], @intCast(100 * i + 1)), 90, "m", payloadOf(&pay[i], @intCast(i + 1)));
        t.flushRam();
        try std.testing.expect(t.drain(10));
        const info = t.info();
        try std.testing.expectEqual(@as(u64, 3), info.stats.written);
    }
    const t = try Tier.open(gpa, io, s, "flashnext", identity_a, 0, 1);
    defer t.deinit();
    for (names, 0..) |n, i| {
        var meta: [64]u8 = undefined;
        const got = try readBack(t, n, &meta);
        defer gpa.free(got);
        try std.testing.expectEqualSlices(u8, &pay[i], got);
    }
}

test "settings take TENSORFOLD_SPILL_RAM_GIB and an absolute directory on this OS" {
    const Env = struct {
        dir: []const u8,
        ram: ?[]const u8,
        pub fn get(e: @This(), name: [:0]const u8) ?[]const u8 {
            if (std.mem.eql(u8, name, "TENSORFOLD_SPILL_GIB")) return "2";
            if (std.mem.eql(u8, name, "TENSORFOLD_SPILL_DIR")) return e.dir;
            if (std.mem.eql(u8, name, "TENSORFOLD_SPILL_RAM_GIB")) return e.ram;
            return null;
        }
    };
    const abs = if (@import("builtin").os.tag == .windows) "C:/spill" else "/spill";
    const s = (try spill.Settings.fromEnv(gpa, Env{ .dir = abs, .ram = "1.5" })).?;
    defer gpa.free(s.dir);
    try std.testing.expectEqual(@as(u64, 3 << 29), s.ram_cap);
    const off = (try spill.Settings.fromEnv(gpa, Env{ .dir = abs, .ram = null })).?;
    defer gpa.free(off.dir);
    try std.testing.expectEqual(@as(u64, 0), off.ram_cap);
    try std.testing.expectError(error.BadSpillSetting, spill.Settings.fromEnv(gpa, Env{ .dir = abs, .ram = "-1" }));
    try std.testing.expectError(error.SpillDirNotAbsolute, spill.Settings.fromEnv(gpa, Env{ .dir = "spill", .ram = null }));
}

test "Windows: a root others may change is refused; the tier's own directories are private" {
    if (@import("builtin").os.tag != .windows) return error.SkipZigTest; // POSIX: the mode test above
    var root = try Root.make();
    defer root.deinit();
    // the test directory inherits its parent's access: on a dev checkout, Authenticated Users may change it
    var pb: [512]u8 = undefined;
    const parent = try std.fmt.bufPrint(&pb, ".zig-cache/tmp/{s}", .{root.tmp.sub_path});
    var s = root.settings(1 << 20);
    const t = try Tier.open(gpa, io, s, "flashnext", identity_a, 0, 1); // made here, so private
    t.deinit();
    s.dir = parent;
    if (Tier.open(gpa, io, s, "flashnext", identity_a, 0, 1)) |x| {
        x.deinit();
        return error.SkipZigTest; // a parent no one else may change (not a shared checkout): nothing to refuse
    } else |err| try std.testing.expectEqual(error.SpillDirUnsafe, err);
}

test "a state read back from disk is held again, so its next read is memory's" {
    var root = try Root.make();
    defer root.deinit();
    var tok: [100]u32 = undefined;
    var pay: [10_000]u8 = undefined;
    var s = root.settings(1 << 20);
    s.ram_cap = 1 << 20;
    const name = blk: {
        const t = try Tier.open(gpa, io, s, "flashnext", identity_a, 0, 1);
        defer t.deinit();
        const n = try hold(t, tokensOf(&tok, 5), 90, "m", payloadOf(&pay, 5));
        try std.testing.expect(t.drain(10));
        break :blk n;
    };
    const t = try Tier.open(gpa, io, s, "flashnext", identity_a, 0, 1);
    defer t.deinit();
    var meta: [64]u8 = undefined;
    const first = try readBack(t, name, &meta); // from its file
    defer gpa.free(first);
    var info = t.info();
    try std.testing.expectEqual(@as(u64, 0), info.stats.ram_loaded);
    try std.testing.expectEqual(@as(u64, 1), info.stats.ram_promoted);
    try std.testing.expectEqual(@as(usize, 1), info.ram_entries);
    const second = try readBack(t, name, &meta); // from memory
    defer gpa.free(second);
    try std.testing.expectEqualSlices(u8, &pay, second);
    try std.testing.expectEqualStrings("m", meta[0..1]);
    info = t.info();
    try std.testing.expectEqual(@as(u64, 1), info.stats.ram_loaded);
}

// -- RAM only (TENSORFOLD_SPILL_RAM_GIB, no TENSORFOLD_SPILL_GIB) -------------------------------------------------------

const MapEnv = struct {
    vars: []const [2][]const u8,
    pub fn get(e: @This(), name: [:0]const u8) ?[]const u8 {
        for (e.vars) |kv| if (std.mem.eql(u8, kv[0], name)) return kv[1];
        return null;
    }
};

test "RAM_GIB without GIB is the RAM-only tier; GIB 0 is off; disk-only settings are refused, not ignored" {
    const s = (try spill.Settings.fromEnv(gpa, MapEnv{ .vars = &.{ .{ "TENSORFOLD_SPILL_RAM_GIB", "2" }, .{ "TENSORFOLD_SPILL_MIN_TOKENS", "4096" } } })).?;
    defer gpa.free(s.dir);
    try std.testing.expect(!s.disk());
    try std.testing.expectEqual(@as(u64, 0), s.cap);
    try std.testing.expectEqual(@as(usize, 0), s.dir.len);
    try std.testing.expectEqual(@as(u64, 2 << 30), s.ram_cap);
    try std.testing.expectEqual(@as(u32, 4096), s.min_tokens);
    var tb: [128]u8 = undefined;
    _ = try s.text(&tb);
    try std.testing.expect((try spill.Settings.fromEnv(gpa, MapEnv{ .vars = &.{} })) == null);
    try std.testing.expect((try spill.Settings.fromEnv(gpa, MapEnv{ .vars = &.{.{ "TENSORFOLD_SPILL_RAM_GIB", "0" }} })) == null);
    try std.testing.expect((try spill.Settings.fromEnv(gpa, MapEnv{ .vars = &.{ .{ "TENSORFOLD_SPILL_GIB", "0" }, .{ "TENSORFOLD_SPILL_RAM_GIB", "2" } } })) == null);
    try std.testing.expectError(error.BadSpillSetting, spill.Settings.fromEnv(gpa, MapEnv{ .vars = &.{.{ "TENSORFOLD_SPILL_RAM_GIB", "-1" }} }));
    try std.testing.expectError(error.BadSpillSetting, spill.Settings.fromEnv(gpa, MapEnv{ .vars = &.{ .{ "TENSORFOLD_SPILL_RAM_GIB", "2" }, .{ "TENSORFOLD_SPILL_MIN_TOKENS", "x" } } }));
    for (spill.Settings.disk_only) |n| {
        try std.testing.expectError(error.SpillDiskSettingWithoutDisk, spill.Settings.fromEnv(gpa, MapEnv{ .vars = &.{ .{ "TENSORFOLD_SPILL_RAM_GIB", "2" }, .{ n, "1" } } }));
    }
    try std.testing.expectError(error.NoSpillLevel, Tier.open(gpa, io, .{ .dir = "", .cap = 0 }, "flashnext", identity_a, 0, 1));
}

/// A RAM-only tier whose settings name `root` nowhere: the test checks nothing appears there.
fn ramOnly(cap: u64) spill.Settings {
    return .{ .dir = "", .cap = 0, .ram_cap = cap };
}

fn exists(path: []const u8) !bool {
    const z = try gpa.dupeSentinel(u8, path, 0);
    defer gpa.free(z);
    var l = @import("spill_fs.zig").Lister.open(z) orelse return false;
    l.close();
    return true;
}

test "RAM only: held states read back byte for byte, the least recently used leaves past the cap, and no file is made" {
    var root = try Root.make();
    defer root.deinit();
    const t = try Tier.open(gpa, io, ramOnly(25_000), "flashnext", identity_a, 0, 1); // two of three 10_000-byte states
    defer t.deinit();
    t.chunk = 4096;
    var forgot: Forgot = .{};
    defer forgot.names.deinit(gpa);
    t.setHook(forgot.hook());
    var tok: [3][100]u32 = undefined;
    var pay: [3][10_000]u8 = undefined;
    const a = try hold(t, tokensOf(&tok[0], 1), 90, "a", payloadOf(&pay[0], 1));
    const b = try hold(t, tokensOf(&tok[1], 101), 90, "b", payloadOf(&pay[1], 2));
    var meta: [64]u8 = undefined;
    const first = try readBack(t, a, &meta); // a is now more recent than b
    gpa.free(first);
    const c = try hold(t, tokensOf(&tok[2], 201), 90, "c", payloadOf(&pay[2], 3));
    try std.testing.expect(t.has(a) and t.has(c));
    try std.testing.expect(!t.has(b));
    try std.testing.expect(t.best(&tok[1], 0, {}, anyAt) == null);
    try std.testing.expectError(error.NotStored, t.read(b));
    try std.testing.expectEqual(@as(usize, 1), forgot.names.items.len);
    try std.testing.expect(forgot.names.items[0].eql(b));
    for ([_]Name{ a, c }, [_]usize{ 0, 2 }) |n, i| {
        const got = try readBack(t, n, &meta);
        defer gpa.free(got);
        try std.testing.expectEqualSlices(u8, &pay[i], got);
    }
    // nowhere to write: a family hands states over whole
    var m: Mem = .{ .bytes = &pay[0] };
    try std.testing.expectError(error.NoDiskLevel, t.write(Name.of(tok[0][0..50], 50), tok[0][0..50], "", pay[0].len, m.source()));
    t.flushRam(); // a clean stop has nothing to write
    try std.testing.expect(t.drain(1));
    const info = t.info();
    try std.testing.expectEqual(@as(u64, 3), info.stats.ram_adopted);
    try std.testing.expectEqual(@as(u64, 1), info.stats.ram_evicted);
    try std.testing.expectEqual(@as(u64, 0), info.stats.ram_dropped); // leaving is the policy, not a failure
    try std.testing.expectEqual(@as(u64, 0), info.stats.ram_demoted);
    try std.testing.expectEqual(@as(u64, 0), info.stats.written);
    try std.testing.expectEqual(@as(u64, 0), info.stats.write_failed);
    try std.testing.expectEqual(@as(u64, 3), info.stats.ram_loaded);
    try std.testing.expectEqual(@as(usize, 2), info.entries);
    try std.testing.expectEqual(@as(usize, 2), info.ram_entries);
    try std.testing.expectEqual(@as(u64, 20_000), info.ram_bytes);
    try std.testing.expectEqual(@as(u64, 0), info.stored_bytes);
    try std.testing.expectEqual(@as(u64, 0), info.cap);
    try std.testing.expectEqual(@as(u64, 0), info.disk_free);
    try std.testing.expect(t.thread == null); // no writer ever started
    try std.testing.expect(!try exists(root.path));
}

test "RAM only: a read under way keeps its state whole when the cap evicts it" {
    const t = try Tier.open(gpa, io, ramOnly(15_000), "flashnext", identity_a, 0, 1); // one 10_000-byte state
    defer t.deinit();
    t.chunk = 4096;
    var tok: [2][100]u32 = undefined;
    var pay: [2][10_000]u8 = undefined;
    const a = try hold(t, tokensOf(&tok[0], 7), 90, "a", payloadOf(&pay[0], 7));
    var r = try t.read(a);
    defer r.close();
    _ = try hold(t, tokensOf(&tok[1], 107), 90, "b", payloadOf(&pay[1], 8)); // a leaves the tier mid-read
    try std.testing.expect(!t.has(a));
    var c: Collect = .{};
    defer c.out.deinit(gpa);
    try r.pump(c.sink());
    try std.testing.expectEqualSlices(u8, &pay[0], c.out.items);
    try std.testing.expectEqual(@as(u64, 10_000), t.info().ram_bytes); // only b's bytes count once a left
}

test "RAM only: a state larger than the cap is refused and nothing held leaves for it" {
    const t = try Tier.open(gpa, io, ramOnly(12_000), "flashnext", identity_a, 0, 1);
    defer t.deinit();
    var tok: [2][100]u32 = undefined;
    var small: [10_000]u8 = undefined;
    var big: [20_000]u8 = undefined;
    const a = try hold(t, tokensOf(&tok[0], 3), 90, "a", payloadOf(&small, 3));
    try std.testing.expect(!t.holds(big.len));
    try std.testing.expectError(error.LargerThanRam, hold(t, tokensOf(&tok[1], 103), 90, "b", payloadOf(&big, 4)));
    try std.testing.expect(t.has(a));
    try std.testing.expectEqual(@as(u64, 0), t.info().stats.ram_evicted);
}
