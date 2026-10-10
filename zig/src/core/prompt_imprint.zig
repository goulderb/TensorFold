//! Prompt-cache states on disk (Imprint): a learned harness state outlives the server and serves later fresh sessions.
const std = @import("std");
const Allocator = std.mem.Allocator;
const builtin = @import("builtin");
const fsys = @import("lanes").learned_fs;

/// One state on disk: the tokens it read (position plus lookahead), the chunk starts below it, and its bytes.
pub const Meta = struct { key: u64, at: u32, tokens: []u32, starts: []u32, bytes: u64, used: u64 = 0 };

const MAGIC: u32 = 0x494d5052; // "IMPR"
const VERSION: u32 = 2;
const HEAD = 9; // a record's words before its tokens: magic, version, key (2), at, token and start counts, bytes (2)

pub const Imprint = struct {
    gpa: Allocator,
    root: []u8,
    dir: [:0]u8, // root/<identity>: the index, the last-open stamp and each state's files (the family's)
    cap: u64, // bytes of learned states this Mac keeps under `root`, every identity together
    admission: @import("lanes").learned_disk.Admission = .{},
    others: u64 = 0, // bytes other identities under `root` still hold
    metas: std.ArrayList(Meta) = .empty,
    clock: u64 = 0, // a meta's `used`: the clock at its last write or read
    index_end: u64 = 0,

    /// The states learned under `identity` in `root` (made when missing); other identities past `cap` go, oldest first.
    pub fn open(gpa: Allocator, root: []const u8, identity: u64, cap: u64) !Imprint {
        var m: Imprint = paths: {
            const r = try gpa.dupe(u8, root);
            errdefer gpa.free(r);
            const dir = try std.fmt.allocPrintSentinel(gpa, "{s}/{x:0>16}", .{ root, identity }, 0);
            errdefer gpa.free(dir);
            try makePath(gpa, dir);
            break :paths .{ .gpa = gpa, .root = r, .dir = dir, .cap = cap };
        };
        errdefer m.deinit();
        @import("lanes").learned_dirs.cleanupParts(root) catch return error.ImprintRead;
        try m.load();
        try @import("lanes").learned_dirs.removeUnindexed(m.dir, m.metas.items);
        m.stamp();
        m.others = @import("lanes").learned_dirs.totalOthers(root, m.dir) catch return error.ImprintRead;
        return m;
    }

    pub fn deinit(m: *Imprint) void {
        for (m.metas.items) |x| free(m.gpa, x);
        m.metas.deinit(m.gpa);
        m.gpa.free(m.dir);
        m.gpa.free(m.root);
    }

    fn free(gpa: Allocator, x: Meta) void {
        gpa.free(x.tokens);
        gpa.free(x.starts);
    }

    pub fn keyOf(tokens: []const u32) u64 {
        return std.hash.Wyhash.hash(0x1e47, std.mem.sliceAsBytes(tokens));
    }

    fn find(m: *const Imprint, key: u64) ?usize {
        for (m.metas.items, 0..) |x, i| if (x.key == key) return i;
        return null;
    }

    pub fn has(m: *const Imprint, key: u64) bool {
        return m.find(key) != null;
    }

    /// The bytes of every state learned under this identity.
    pub fn total(m: *const Imprint) u64 {
        var n: u64 = 0;
        for (m.metas.items) |x| n += x.bytes;
        return n;
    }

    /// Reserve the current index's size too: LRU eviction rewrites it through a temporary file.
    pub fn indexScratchBytes(m: *const Imprint) !u64 {
        var path: [1100]u8 = undefined;
        const fd = fsys.open(try std.fmt.bufPrintSentinel(&path, "{s}/index", .{m.dir}, 0), .{}) catch |err| return if (err == error.Missing) 0 else error.ImprintRead;
        defer fsys.close(fd);
        return fsys.size(fd) catch error.ImprintRead;
    }

    /// Whether `bytes` more fit under the cap; other identities give room first, least recently opened first.
    pub fn fits(m: *Imprint, bytes: u64) bool {
        if (m.others + m.total() + bytes > m.cap) m.others = sweep(m.gpa, m.root, m.dir, m.cap -| (m.total() + bytes));
        return m.others + m.total() + bytes <= m.cap;
    }

    /// The least recently used state learned here: the next to go when the cap needs room.
    pub fn victim(m: *const Imprint) ?u64 {
        var out: ?*const Meta = null;
        for (m.metas.items) |*x| if (out == null or x.used < out.?.used) {
            out = x;
        };
        return if (out) |x| x.key else null;
    }

    /// The longest learned state `prompt` resumes: its tokens a prefix, a row left, `starts` below it the prompt's own.
    pub fn best(m: *const Imprint, prompt: []const u32, starts: []const u32, planned: bool, longer_than: u32) ?*const Meta {
        var out: ?*const Meta = null;
        for (m.metas.items) |*x| {
            if (x.at <= longer_than or x.tokens.len > prompt.len or x.at >= prompt.len) continue;
            if (out != null and x.at <= out.?.at) continue;
            if (!std.mem.eql(u32, prompt[0..x.tokens.len], x.tokens)) continue;
            if (planned and !sameBelow(starts, x.starts, x.at)) continue;
            out = x;
        }
        return out;
    }

    /// State `key` was read: the most recently used now.
    pub fn touch(m: *Imprint, key: u64) void {
        const i = m.find(key) orelse return;
        m.clock += 1;
        m.metas.items[i].used = m.clock;
    }

    pub fn pendingWrite(m: *const Imprint) !?u64 {
        var buf: [1100]u8 = undefined;
        const fd = fsys.open(try std.fmt.bufPrintSentinel(&buf, "{s}/pending", .{m.dir}, 0), .{ .nofollow = true }) catch |err| return if (err == error.Missing) null else error.ImprintRead;
        defer fsys.close(fd);
        if ((fsys.size(fd) catch return error.ImprintRead) != 48) return error.ImprintRead;
        var data: [48]u8 = undefined;
        if (!readAt(fd, &data, 0)) return error.ImprintRead;
        var digest: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(data[0..16], &digest, .{});
        if (!std.mem.eql(u8, &digest, data[16..]) or std.mem.readInt(u64, data[0..8], .little) != 0x494d5052494e5431) return error.ImprintRead;
        return std.mem.readInt(u64, data[8..16], .little);
    }
    pub fn beginWrite(m: *const Imprint, key: u64) !void {
        if (try m.pendingWrite() != null) return error.ImprintPending;
        var buf: [1100]u8 = undefined;
        var dest: [1100]u8 = undefined;
        const tmp = try std.fmt.bufPrintSentinel(&buf, "{s}/pending.part", .{m.dir}, 0);
        const fd = fsys.open(tmp, .{ .access = .write, .create = true, .trunc = true, .nofollow = true }) catch return error.ImprintWrite;
        errdefer fsys.unlink(tmp) catch {};
        {
            defer fsys.close(fd);
            var data: [48]u8 = undefined;
            std.mem.writeInt(u64, data[0..8], 0x494d5052494e5431, .little);
            std.mem.writeInt(u64, data[8..16], key, .little);
            std.crypto.hash.sha2.Sha256.hash(data[0..16], data[16..48], .{});
            try writeAll(fd, &data);
            if (!fsys.sync(fd)) return error.ImprintWrite;
        }
        fsys.rename(tmp, try std.fmt.bufPrintSentinel(&dest, "{s}/pending", .{m.dir}, 0)) catch return error.ImprintWrite;
    }
    pub fn clearPending(m: *const Imprint) !void {
        var buf: [1100]u8 = undefined;
        try @import("lanes").learned_dirs.unlink(try std.fmt.bufPrintSentinel(&buf, "{s}/pending", .{m.dir}, 0));
    }
    /// Record a state its family has written (`bytes` on this Mac): an index record, fsynced, then the entry.
    pub fn add(m: *Imprint, key: u64, at: u32, tokens: []const u32, starts: []const u32, bytes: u64) !void {
        if (tokens.len > 1 << 22 or at == 0 or at > tokens.len or key != keyOf(tokens) or !ordered(starts)) return error.ImprintWrite;
        if (m.has(key)) return;
        const below = cut(starts, at);
        var path: [1100]u8 = undefined;
        const fd = fsys.open(try std.fmt.bufPrintSentinel(&path, "{s}/index", .{m.dir}, 0), .{ .access = .read_write, .create = true, .append = true, .nofollow = true }) catch return error.ImprintWrite;
        defer fsys.close(fd);
        _ = indexSize(fd) catch return error.ImprintWrite;
        var end = fsys.size(fd) catch return error.ImprintWrite;
        if (end != m.index_end) {
            try m.reload(fd);
            if (m.has(key)) return;
            end = fsys.size(fd) catch return error.ImprintWrite;
            if (end != m.index_end) return error.ImprintWrite;
        }
        const t = try m.gpa.dupe(u32, tokens);
        errdefer m.gpa.free(t);
        const s = try m.gpa.dupe(u32, below);
        errdefer m.gpa.free(s);
        try m.metas.ensureUnusedCapacity(m.gpa, 1);
        errdefer _ = fsys.truncate(fd, end);
        try record(fd, key, at, tokens, below, bytes);
        if (!fsys.sync(fd)) return error.ImprintWrite;
        m.index_end = end + HEAD * 4 + 4 * (tokens.len + below.len);
        m.clock += 1;
        m.metas.appendAssumeCapacity(.{ .key = key, .at = at, .tokens = t, .starts = s, .bytes = bytes, .used = m.clock });
    }

    /// Forget state `key`: the index is rewritten without it (its family removes the files).
    pub fn remove(m: *Imprint, key: u64) !void {
        const i = m.find(key) orelse return;
        var path: [1100]u8 = undefined;
        var part: [1100]u8 = undefined;
        const tmp = try std.fmt.bufPrintSentinel(&part, "{s}/index.part", .{m.dir}, 0);
        const fd = fsys.open(tmp, .{ .access = .write, .create = true, .trunc = true }) catch return error.ImprintWrite;
        errdefer fsys.unlink(tmp) catch {};
        {
            defer fsys.close(fd);
            for (m.metas.items, 0..) |x, j| if (j != i) try record(fd, x.key, x.at, x.tokens, x.starts, x.bytes);
            if (!fsys.sync(fd)) return error.ImprintWrite;
        }
        fsys.rename(tmp, try std.fmt.bufPrintSentinel(&path, "{s}/index", .{m.dir}, 0)) catch return error.ImprintWrite;
        free(m.gpa, m.metas.orderedRemove(i));
        m.index_end = 0;
        for (m.metas.items) |x| m.index_end += HEAD * 4 + 4 * (x.tokens.len + x.starts.len);
    }

    /// Keep a valid prefix across torn or invalid suffixes; unknown versions and I/O errors refuse learning.
    fn load(m: *Imprint) !void {
        var path: [1100]u8 = undefined;
        const fd = fsys.open(try std.fmt.bufPrintSentinel(&path, "{s}/index", .{m.dir}, 0), .{ .access = .read_write, .nofollow = true }) catch |err| return if (err == error.Missing) {} else error.ImprintRead;
        defer fsys.close(fd);
        try m.readIndex(fd);
    }

    fn reload(m: *Imprint, fd: fsys.Fd) !void {
        var fresh: Imprint = .{ .gpa = m.gpa, .root = m.root, .dir = m.dir, .cap = m.cap };
        defer {
            for (fresh.metas.items) |x| free(m.gpa, x);
            fresh.metas.deinit(m.gpa);
        }
        try fresh.readIndex(fd);
        for (fresh.metas.items) |*x| {
            if (m.find(x.key)) |i| x.used = m.metas.items[i].used else {
                m.clock += 1;
                x.used = m.clock;
            }
        }
        std.mem.swap(std.ArrayList(Meta), &m.metas, &fresh.metas);
        m.index_end = fresh.index_end;
    }

    fn torn(m: *Imprint, fd: fsys.Fd, whole: u64, observed: u64) !void {
        try repairTail(fd, whole, observed);
        m.index_end = whole;
    }

    fn readIndex(m: *Imprint, fd: fsys.Fd) !void {
        const end = try indexSize(fd);
        var off: u64 = 0;
        while (off < end) {
            if (end - off < HEAD * 4) {
                if (end - off >= 8) {
                    var version: [2]u32 = undefined;
                    if (!readAt(fd, std.mem.sliceAsBytes(&version), off)) return error.ImprintRead;
                    if (version[0] == MAGIC and version[1] != VERSION) return error.ImprintRead;
                }
                return m.torn(fd, off, end);
            }
            var head: [HEAD]u32 = undefined;
            if (!readAt(fd, std.mem.sliceAsBytes(&head), off)) return error.ImprintRead;
            if (head[0] != MAGIC) return m.torn(fd, off, end);
            if (head[1] != VERSION) return error.ImprintRead;
            const n: usize = head[5];
            const k: usize = head[6];
            if (n > 1 << 22 or k > n or head[4] == 0 or head[4] > n) return m.torn(fd, off, end);
            const at = off + @sizeOf(@TypeOf(head));
            const next = at + 4 * (n + k);
            if (next > end) return m.torn(fd, off, end);
            const t = try m.gpa.alloc(u32, n);
            const s = m.gpa.alloc(u32, k) catch |err| {
                m.gpa.free(t);
                return err;
            };
            const x: Meta = .{ .key = word64(head[2..4]), .at = head[4], .tokens = t, .starts = s, .bytes = word64(head[7..9]) };
            if (!readAt(fd, std.mem.sliceAsBytes(t), at) or !readAt(fd, std.mem.sliceAsBytes(s), at + 4 * n)) {
                free(m.gpa, x);
                return error.ImprintRead;
            }
            if (x.key != keyOf(t) or !ordered(s) or (s.len > 0 and s[s.len - 1] >= x.at)) {
                free(m.gpa, x);
                return m.torn(fd, off, end);
            }
            if (m.has(x.key)) free(m.gpa, x) else {
                m.clock += 1;
                var y = x;
                y.used = m.clock;
                m.metas.append(m.gpa, y) catch |err| {
                    free(m.gpa, x);
                    return err;
                };
            }
            off = next;
            m.index_end = off;
        }
    }

    /// This identity was opened: the newest under `root` for the sweep (root/clock counts opens).
    fn stamp(m: *const Imprint) void {
        var path: [1100]u8 = undefined;
        const clock = std.fmt.bufPrintSentinel(&path, "{s}/clock", .{m.root}, 0) catch return;
        const now = readWord(clock) +| 1;
        writeWord(clock, now);
        var used: [1100]u8 = undefined;
        writeWord(std.fmt.bufPrintSentinel(&used, "{s}/used", .{m.dir}, 0) catch return, now);
    }
};

fn indexSize(fd: fsys.Fd) !u64 {
    return fsys.size(fd) catch error.ImprintRead;
}

fn repairTail(fd: fsys.Fd, whole: u64, observed: u64) !void {
    if ((fsys.size(fd) catch return error.ImprintRead) != observed) return error.ImprintRead;
    if (!fsys.truncate(fd, whole) or !fsys.sync(fd)) return error.ImprintWrite;
}

fn ordered(starts: []const u32) bool {
    for (starts, 0..) |at, i| if (i > 0 and at <= starts[i - 1]) return false;
    return true;
}

/// One index record: its head, its tokens, then the chunk starts below its position.
fn record(fd: fsys.Fd, key: u64, at: u32, tokens: []const u32, starts: []const u32, bytes: u64) !void {
    const head = [HEAD]u32{ MAGIC, VERSION, @truncate(key), @truncate(key >> 32), at, @intCast(tokens.len), @intCast(starts.len), @truncate(bytes), @truncate(bytes >> 32) };
    try writeAll(fd, std.mem.sliceAsBytes(&head));
    try writeAll(fd, std.mem.sliceAsBytes(tokens));
    try writeAll(fd, std.mem.sliceAsBytes(starts));
}

fn word64(w: []const u32) u64 {
    return @as(u64, w[0]) | @as(u64, w[1]) << 32;
}

/// The chunk starts below `at`.
fn cut(starts: []const u32, at: u32) []const u32 {
    var n: usize = 0;
    while (n < starts.len and starts[n] < at) n += 1;
    return starts[0..n];
}

/// Whether a prompt's chunk starts below `at` are the learned pass's (a planned family's state depends on them).
fn sameBelow(starts: []const u32, learned: []const u32, at: u32) bool {
    return std.mem.eql(u32, cut(starts, at), learned) and std.mem.indexOfScalar(u32, starts, at) != null;
}

/// An identity directory's name: 16 hex digits.
fn identityName(name: []const u8) bool {
    if (name.len != 16) return false;
    for (name) |ch| if (!std.ascii.isHex(ch)) return false;
    return true;
}

/// Other identities' directories (16 hex digits, their files only) deleted, oldest first, down to `room` bytes.
fn sweep(gpa: Allocator, root: []const u8, keep: []const u8, room: u64) u64 {
    const Dir = struct { name: [16]u8, bytes: u64, used: u64 };
    var dirs: std.ArrayList(Dir) = .empty;
    defer dirs.deinit(gpa);
    var d = fsys.openDir(root) catch return 0;
    var held: u64 = 0;
    while (fsys.next(&d)) |ent| {
        const name = ent.name;
        if (ent.kind != .dir or !identityName(name) or std.mem.endsWith(u8, keep, name)) continue;
        var sub: [1100]u8 = undefined;
        const path = std.fmt.bufPrint(&sub, "{s}/{s}", .{ root, name }) catch continue;
        var buf: [1100]u8 = undefined;
        const used = readWord(std.fmt.bufPrintSentinel(&buf, "{s}/used", .{path}, 0) catch continue);
        const x: Dir = .{ .name = name[0..16].*, .bytes = files(path, false), .used = used };
        held += x.bytes;
        dirs.append(gpa, x) catch break;
    }
    fsys.closeDir(&d);
    std.mem.sort(Dir, dirs.items, {}, struct {
        fn older(_: void, a: Dir, b: Dir) bool {
            return a.used < b.used;
        }
    }.older);
    for (dirs.items) |x| {
        if (held <= room) break;
        var sub: [1100]u8 = undefined;
        const path = std.fmt.bufPrintSentinel(&sub, "{s}/{s}", .{ root, &x.name }, 0) catch continue;
        _ = files(path, true);
        fsys.rmdir(path) catch continue;
        held -|= x.bytes;
    }
    return held;
}

fn files(dir: []const u8, remove: bool) u64 {
    var d = fsys.openDir(dir) catch return 0;
    defer fsys.closeDir(&d);
    var n: u64 = 0;
    while (fsys.next(&d)) |ent| {
        if (ent.kind != .file) continue;
        var p: [1400]u8 = undefined;
        const f = std.fmt.bufPrintSentinel(&p, "{s}/{s}", .{ dir, ent.name }, 0) catch continue;
        const fd = fsys.open(f, .{}) catch continue;
        const end = fsys.size(fd) catch 0;
        fsys.close(fd);
        n += end;
        if (remove) fsys.unlink(f) catch {};
    }
    return n;
}

fn readWord(path: [:0]const u8) u64 {
    const fd = fsys.open(path, .{}) catch return 0;
    defer fsys.close(fd);
    var w: u64 = 0;
    return if (readAt(fd, std.mem.asBytes(&w), 0)) w else 0;
}

fn writeWord(path: [:0]const u8, w: u64) void {
    var buf: [1200]u8 = undefined;
    const part = std.fmt.bufPrintSentinel(&buf, "{s}.part", .{path}, 0) catch return;
    const fd = fsys.open(part, .{ .access = .write, .create = true, .trunc = true, .nofollow = true }) catch return;
    defer fsys.close(fd);
    defer fsys.unlink(part) catch {};
    writeAll(fd, std.mem.asBytes(&w)) catch return;
    if (!fsys.sync(fd)) return;
    fsys.rename(part, path) catch {};
}

pub fn writeAll(fd: fsys.Fd, bytes: []const u8) !void {
    fsys.write(fd, bytes) catch return error.ImprintWrite;
}

pub fn readAt(fd: fsys.Fd, dest: []u8, at: u64) bool {
    return fsys.readAt(fd, dest, at);
}

/// This Mac's OS build (sysctl kern.osversion): its Metal compiler can change a kernel's bits.
pub fn osBuild(buf: []u8) []const u8 {
    if (comptime builtin.os.tag == .windows) return "";
    var n: usize = buf.len;
    if (std.c.sysctlbyname("kern.osversion", buf.ptr, &n, null, 0) != 0) return "";
    return std.mem.sliceTo(buf[0..n], 0);
}

pub fn defaultRoot(a: Allocator) ![]const u8 {
    if (comptime builtin.os.tag == .windows) {
        var base: [900]u8 = undefined;
        const local = fsys.localAppData(&base) orelse return error.ImprintDir;
        return std.fmt.allocPrint(a, "{s}\\tensorfold\\learned", .{local});
    }
    const home = std.c.getenv("HOME") orelse return error.ImprintDir;
    return std.fmt.allocPrint(a, "{s}/.cache/tensorfold/learned", .{std.mem.span(home)});
}

fn makePath(gpa: Allocator, dir: [:0]const u8) !void {
    const tmp = try gpa.dupeSentinel(u8, dir, 0);
    defer gpa.free(tmp);
    for (tmp[1..], 1..) |ch, i| if (ch == '/' or (builtin.os.tag == .windows and ch == '\\')) {
        const sep = tmp[i];
        tmp[i] = 0;
        fsys.mkdir(tmp[0..i :0]) catch {};
        tmp[i] = sep;
    };
    fsys.mkdir(tmp) catch |err| if (err != error.Exists) return error.ImprintDir;
}

/// A learned state's file under `imprint.dir`, read whole into a count (the single-rank reclaim).
pub fn singleReclaim(_: *anyopaque, dir: [:0]const u8, key: u64) !u64 {
    var buf: [1100]u8 = undefined;
    return @import("lanes").learned_dirs.fileBytes(try std.fmt.bufPrintSentinel(&buf, "{s}/{x:0>16}.bin", .{ dir, key }, 0));
}

/// A learned state's file under `imprint.dir` removed (the cap needs room, or it no longer reads back).
pub fn singleForgetChecked(_: *anyopaque, dir: [:0]const u8, key: u64) !void {
    var buf: [1100]u8 = undefined;
    try @import("lanes").learned_dirs.unlink(try std.fmt.bufPrintSentinel(&buf, "{s}/{x:0>16}.bin", .{ dir, key }, 0));
}

test {
    _ = @import("prompt_imprint_test.zig");
    _ = @import("prompt_imprint_fault_test.zig");
    _ = @import("learned_disk.zig");
}

test "a failed tail truncation keeps the bytes and refuses repair" {
    var buf: [128]u8 = undefined;
    const path = try fsys.tempRoot(&buf, "index-repair");
    defer fsys.unlink(path) catch {};
    const writer = try fsys.open(path, .{ .access = .write, .create = true, .trunc = true });
    try writeAll(writer, "record");
    fsys.close(writer);
    const reader = try fsys.open(path, .{});
    defer fsys.close(reader);
    try std.testing.expectError(error.ImprintWrite, repairTail(reader, 0, 6));
    try std.testing.expectEqual(@as(u64, 6), try fsys.size(reader));
    var actual: [6]u8 = undefined;
    try std.testing.expect(readAt(reader, &actual, 0));
    try std.testing.expectEqualStrings("record", &actual);
}
