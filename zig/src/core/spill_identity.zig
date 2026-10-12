//! Linux bootstrap digest cache for enabled spill: local selectors never enter the agreed identity.
const std = @import("std");
const linux = std.os.linux;
const Sha = std.crypto.hash.sha2.Sha256;

pub const record_bytes = 104;
pub const maximum_records = 256;
pub const node = ".identity-digests-v2";
const record_magic = "TFID0002";
const record_body_bytes = 72;
const block_bytes = 64 << 10;
const hex_bytes = 64;
const record_name_bytes = hex_bytes + ".id".len;
const partial_prefix = ".partial-";
const partial_name_bytes = partial_prefix.len + hex_bytes;
const max_header_bytes = 1 << 30;
const at_symlink_nofollow: u32 = 0x100;
const at_empty_path: u32 = 0x1000;
const s_ifmt: u16 = 0o170000;
const s_ifreg: u16 = 0o100000;
const s_ifdir: u16 = 0o040000;
const Name = [record_name_bytes + 1]u8;

pub const Result = struct { size: u64, digest: [32]u8 };
pub const Meta = struct {
    size: u64,
    mtime_s: i64,
    mtime_ns: u32,
    ctime_s: i64,
    ctime_ns: u32,
    inode: u64,
    dev_major: u32,
    dev_minor: u32,
};

const statx_mask: linux.STATX = .{ .TYPE = true, .MODE = true, .NLINK = true, .UID = true, .SIZE = true, .MTIME = true, .CTIME = true, .INO = true };

fn checkedStatx(st: linux.Statx) !linux.Statx {
    if (!st.mask.TYPE or !st.mask.MODE or !st.mask.NLINK or !st.mask.UID or !st.mask.SIZE or !st.mask.MTIME or !st.mask.CTIME or !st.mask.INO) return error.IdentityCacheUnsafe;
    return st;
}

fn statxFd(fd: c_int) !linux.Statx {
    var st: linux.Statx = undefined;
    if (std.c.statx(fd, "", at_empty_path, statx_mask, &st) != 0) return error.WeightsUnreadable;
    return checkedStatx(st);
}

fn statxAt(dirfd: c_int, path: [:0]const u8, flags: u32) !linux.Statx {
    var st: linux.Statx = undefined;
    if (std.c.statx(dirfd, path, flags, statx_mask, &st) != 0) return error.IdentityCacheUnsafe;
    return checkedStatx(st);
}

fn toMeta(st: linux.Statx) Meta {
    return .{
        .size = st.size,
        .mtime_s = st.mtime.sec,
        .mtime_ns = st.mtime.nsec,
        .ctime_s = st.ctime.sec,
        .ctime_ns = st.ctime.nsec,
        .inode = st.ino,
        .dev_major = st.dev_major,
        .dev_minor = st.dev_minor,
    };
}

pub fn metadata(fd: c_int) !Meta {
    const st = try statxFd(fd);
    if (st.mode & s_ifmt != s_ifreg) return error.WeightsUnreadable;
    return toMeta(st);
}

pub fn sameMetadata(a: Meta, b: Meta) bool {
    return std.meta.eql(a, b);
}

fn verifyUnchanged(path: [:0]const u8, fd: c_int, before: Meta) !void {
    const now = metadata(fd) catch return error.SourceChanged;
    if (!sameMetadata(before, now)) return error.SourceChanged;
    const via_path = statxAt(std.posix.AT.FDCWD, path, 0) catch return error.SourceChanged;
    if (!sameMetadata(before, toMeta(via_path))) return error.SourceChanged;
}

fn canonicalPath(fd: c_int, buf: *[4096]u8) ![]const u8 {
    var link: [32:0]u8 = undefined;
    const proc = try std.fmt.bufPrintSentinel(&link, "/proc/self/fd/{d}", .{fd}, 0);
    const n = std.c.readlink(proc, buf, buf.len);
    if (n <= 0 or n >= buf.len) return error.WeightsUnreadable;
    return buf[0..@intCast(n)];
}

fn feed(h: *Sha, comptime T: type, v: T) void {
    var b: [@sizeOf(T)]u8 = undefined;
    std.mem.writeInt(T, &b, v, .little);
    h.update(&b);
}

fn key(path: []const u8, m: Meta, index: [32]u8) [32]u8 {
    var h = Sha.init(.{});
    feed(&h, u64, path.len);
    h.update(path);
    feed(&h, u64, m.size);
    feed(&h, i64, m.mtime_s);
    feed(&h, u32, m.mtime_ns);
    feed(&h, i64, m.ctime_s);
    feed(&h, u32, m.ctime_ns);
    feed(&h, u64, m.inode);
    feed(&h, u32, m.dev_major);
    feed(&h, u32, m.dev_minor);
    h.update(&index);
    return h.finalResult();
}

fn recordName(k: [32]u8) Name {
    var out: Name = @splat(0);
    const hex = std.fmt.bytesToHex(k, .lower);
    @memcpy(out[0..hex_bytes], &hex);
    @memcpy(out[hex_bytes..record_name_bytes], ".id");
    return out;
}

fn partialName(buf: *[partial_name_bytes + 1]u8, k: [32]u8) [:0]u8 {
    const hex = std.fmt.bytesToHex(k, .lower);
    @memcpy(buf[0..partial_prefix.len], partial_prefix);
    @memcpy(buf[partial_prefix.len..partial_name_bytes], &hex);
    buf[partial_name_bytes] = 0;
    return buf[0..partial_name_bytes :0];
}

fn isRecordName(name: []const u8) bool {
    if (name.len != record_name_bytes or !std.mem.endsWith(u8, name, ".id")) return false;
    for (name[0..hex_bytes]) |c| {
        if (!((c >= '0' and c <= '9') or (c >= 'a' and c <= 'f'))) return false;
    }
    return true;
}

fn encodeRecord(k: [32]u8, value: [32]u8) [record_bytes]u8 {
    var rec: [record_bytes]u8 = undefined;
    @memcpy(rec[0..8], record_magic);
    @memcpy(rec[8..40], &k);
    @memcpy(rec[40..72], &value);
    var checksum: [32]u8 = undefined;
    Sha.hash(rec[0..record_body_bytes], &checksum, .{});
    @memcpy(rec[72..104], &checksum);
    return rec;
}

fn decodeRecord(rec: *const [record_bytes]u8, k: [32]u8) ?[32]u8 {
    if (!std.mem.eql(u8, rec[0..8], record_magic)) return null;
    var checksum: [32]u8 = undefined;
    Sha.hash(rec[0..record_body_bytes], &checksum, .{});
    if (!std.mem.eql(u8, rec[72..104], &checksum)) return null;
    if (!std.mem.eql(u8, rec[8..40], &k)) return null;
    var value: [32]u8 = undefined;
    @memcpy(&value, rec[40..72]);
    return value;
}

fn checkNewPrivate(fd: c_int) !void {
    const st = statxFd(fd) catch return error.IdentityCacheUnsafe;
    if (st.mode & 0o777 != 0o700 or st.uid != std.c.getuid()) return error.IdentityCacheUnsafe;
}

fn checkChainDir(fd: c_int) !void {
    const st = statxFd(fd) catch return error.IdentityCacheUnsafe;
    if (st.mode & s_ifmt != s_ifdir) return error.IdentityCacheUnsafe;
    const uid = std.c.getuid();
    if ((st.uid != 0 and st.uid != uid) or st.mode & 0o022 != 0) return error.IdentityCacheUnsafe;
}

/// Walks root one component at a time from a directory fd. Missing components are created
/// private (0700) with mkdirat; every component is opened NOFOLLOW and validated by fd.
/// Existing ancestors must be owned by root or the current uid with no group/world write.
fn openRootChain(root: [:0]const u8) !c_int {
    const absolute = root.len > 0 and root[0] == '/';
    if (!absolute) return error.IdentityCacheUnsafe;
    var fd = std.c.open("/", .{ .ACCMODE = .RDONLY, .DIRECTORY = true, .NOFOLLOW = true, .CLOEXEC = true }, @as(std.c.mode_t, 0));
    if (fd < 0) return error.IdentityCacheUnsafe;
    var parts = std.mem.tokenizeScalar(u8, root, '/');
    var seen = false;
    while (parts.next()) |part| {
        if (std.mem.eql(u8, part, ".")) continue;
        if (part.len > 255) {
            _ = std.c.close(fd);
            return error.IdentityCacheUnsafe;
        }
        seen = true;
        var name_buf: [256]u8 = undefined;
        @memcpy(name_buf[0..part.len], part);
        name_buf[part.len] = 0;
        const name = name_buf[0..part.len :0];
        const created = std.c.mkdirat(fd, name, 0o700) == 0;
        const next = std.c.openat(fd, name, .{ .ACCMODE = .RDONLY, .DIRECTORY = true, .NOFOLLOW = true, .CLOEXEC = true }, @as(std.c.mode_t, 0));
        _ = std.c.close(fd);
        if (next < 0) return error.IdentityCacheUnsafe;
        fd = next;
        checkChainDir(fd) catch |err| {
            _ = std.c.close(fd);
            return err;
        };
        if (created) checkNewPrivate(fd) catch |err| {
            _ = std.c.close(fd);
            return err;
        };
    }
    if (!seen) {
        _ = std.c.close(fd);
        return error.IdentityCacheUnsafe;
    }
    return fd;
}

fn requirePrivateDir(fd: c_int) !void {
    const st = statxFd(fd) catch return error.IdentityCacheUnsafe;
    if (st.mode & s_ifmt != s_ifdir or st.uid != std.c.getuid() or st.mode & 0o777 != 0o700) return error.IdentityCacheUnsafe;
}

const Time = struct {
    s: i64 = 0,
    ns: u32 = 0,
    fn before(a: Time, b: Time) bool {
        return a.s < b.s or (a.s == b.s and a.ns < b.ns);
    }
};

const Scan = struct {
    count: usize = 0,
    oldest: ?Name = null,
    oldest_time: Time = .{},
};

pub const Event = enum { payload_fsync, renamed, directory_fsync };

pub const Store = struct {
    io: std.Io,
    fd: c_int,
    mutex: std.Io.Mutex = .init,
    hashes: u64 = 0,
    hits: u64 = 0,
    advice_calls: u64 = 0,
    limit: usize = maximum_records,
    last_record: Name = @splat(0),
    hook: ?struct {
        ptr: *anyopaque,
        after_block: *const fn (*anyopaque, c_int, u64) void,
        after_lookup: ?*const fn (*anyopaque) void = null,
        critical: ?*const fn (*anyopaque, *Store) void = null,
        dir_sync: ?*const fn (*anyopaque, c_int) void = null,
        open_flags: ?*const fn (*anyopaque, *std.c.O) void = null,
        event: ?*const fn (*anyopaque, Event, c_int) ?c_int = null,
    } = null,

    pub fn open(io: std.Io, root: [:0]const u8) !Store {
        const rfd = try openRootChain(root);
        defer _ = std.c.close(rfd);
        try requirePrivateDir(rfd);
        _ = std.c.mkdirat(rfd, node, 0o700);
        const nfd = std.c.openat(rfd, node, .{ .ACCMODE = .RDONLY, .DIRECTORY = true, .NOFOLLOW = true, .CLOEXEC = true }, @as(std.c.mode_t, 0));
        if (nfd < 0) return error.IdentityCacheUnsafe;
        errdefer _ = std.c.close(nfd);
        try requirePrivateDir(nfd);
        var store: Store = .{ .io = io, .fd = nfd };
        try store.maintain();
        return store;
    }

    pub fn close(s: *Store) void {
        _ = std.c.close(s.fd);
    }

    /// Locked bootstrap maintenance, run on every open so warm all-hit starts also remove
    /// stale partials and cap the footprint.
    pub fn maintain(s: *Store) !void {
        try s.lock();
        defer s.unlock();
        if (s.hook) |hook| if (hook.critical) |probe| probe(hook.ptr, s);
        try s.removeStalePartials();
        try s.prune();
    }

    fn fsyncEvent(s: *Store, ev: Event, fd: c_int) c_int {
        if (s.hook) |hook| if (hook.event) |observe| if (observe(hook.ptr, ev, fd)) |forced| return forced;
        return std.c.fsync(fd);
    }

    fn emitEvent(s: *Store, ev: Event, fd: c_int) void {
        if (s.hook) |hook| {
            if (hook.event) |observe| {
                _ = observe(hook.ptr, ev, fd);
            }
        }
    }

    fn syncDirectory(s: *Store) !void {
        if (s.fsyncEvent(.directory_fsync, s.fd) != 0) return error.IdentityCacheUnsafe;
        if (s.hook) |hook| if (hook.dir_sync) |probe| probe(hook.ptr, s.fd);
    }

    pub fn digest(s: *Store, path: [:0]const u8, index: [32]u8) !Result {
        var flags: std.c.O = .{ .ACCMODE = .RDONLY, .NONBLOCK = true, .CLOEXEC = true };
        if (s.hook) |hook| if (hook.open_flags) |observe| observe(hook.ptr, &flags);
        const fd = std.c.open(path, flags, @as(std.c.mode_t, 0));
        if (fd < 0) return error.WeightsUnreadable;
        defer _ = std.c.close(fd);
        const before = try metadata(fd);
        var canon_buf: [4096]u8 = undefined;
        const canon = try canonicalPath(fd, &canon_buf);
        try verifyUnchanged(path, fd, before);
        try s.checkHeader(fd, before.size);
        const k = key(canon, before, index);
        const name = recordName(k);
        if (try s.lookup(name, k)) |cached| {
            if (s.hook) |hook| if (hook.after_lookup) |after| after(hook.ptr);
            try verifyUnchanged(path, fd, before);
            return .{ .size = before.size, .digest = cached };
        }
        const value = try s.hashSource(fd, before.size);
        try verifyUnchanged(path, fd, before);
        try s.commit(name, k, value);
        return .{ .size = before.size, .digest = value };
    }

    fn lock(s: *Store) !void {
        s.mutex.lockUncancelable(s.io);
        if (linux.flock(s.fd, std.posix.LOCK.EX) != 0) {
            s.mutex.unlock(s.io);
            return error.IdentityCacheUnsafe;
        }
    }

    fn unlock(s: *Store) void {
        _ = linux.flock(s.fd, std.posix.LOCK.UN);
        s.mutex.unlock(s.io);
    }

    fn advise(s: *Store, fd: c_int, offset: u64, len: usize) void {
        _ = linux.fadvise(fd, @intCast(offset), @intCast(len), linux.POSIX_FADV.DONTNEED);
        s.advice_calls += 1;
    }

    fn checkHeader(s: *Store, fd: c_int, size: u64) !void {
        if (size < 8) return error.WeightsUnreadable;
        var head: [8]u8 = undefined;
        if (std.c.pread(fd, &head, head.len, 0) != @as(isize, 8)) return error.WeightsUnreadable;
        s.advise(fd, 0, 8);
        const header_len = std.mem.readInt(u64, &head, .little);
        if (header_len > size - 8 or header_len > max_header_bytes) return error.WeightsUnreadable;
    }

    fn lookup(s: *Store, name: Name, k: [32]u8) !?[32]u8 {
        try s.lock();
        defer s.unlock();
        s.last_record = name;
        if (s.hook) |hook| if (hook.critical) |probe| probe(hook.ptr, s);
        const z = name[0..record_name_bytes :0];
        const st = statxAt(s.fd, z, at_symlink_nofollow) catch return null;
        if (st.mode & s_ifmt != s_ifreg) return error.IdentityCacheUnsafe;
        if (st.nlink != 1 or st.uid != std.c.getuid() or st.mode & 0o777 != 0o600) return error.IdentityCacheUnsafe;
        if (st.size > record_bytes) return error.IdentityCacheUnsafe;
        if (st.size < record_bytes) return null;
        const rfd = std.c.openat(s.fd, z, .{ .ACCMODE = .RDONLY, .NOFOLLOW = true, .CLOEXEC = true }, @as(std.c.mode_t, 0));
        if (rfd < 0) return error.IdentityCacheUnsafe;
        defer _ = std.c.close(rfd);
        const opened = statxFd(rfd) catch return error.IdentityCacheUnsafe;
        if (opened.ino != st.ino or opened.dev_major != st.dev_major or opened.dev_minor != st.dev_minor) return error.IdentityCacheUnsafe;
        var rec: [record_bytes]u8 = undefined;
        if (std.c.pread(rfd, &rec, rec.len, 0) != @as(isize, record_bytes)) return null;
        const value = decodeRecord(&rec, k) orelse return null;
        s.hits += 1;
        return value;
    }

    fn hashSource(s: *Store, fd: c_int, size: u64) ![32]u8 {
        var h = Sha.init(.{});
        var block: [block_bytes]u8 = undefined;
        var at: u64 = 0;
        while (at < size) {
            const want: usize = @intCast(@min(@as(u64, block.len), size - at));
            const got = std.c.pread(fd, &block, want, @intCast(at));
            if (got < 0) return error.WeightsUnreadable;
            if (got == 0) return error.SourceChanged;
            const n: usize = @intCast(got);
            h.update(block[0..n]);
            s.advise(fd, at, n);
            at += n;
            if (s.hook) |hook| hook.after_block(hook.ptr, fd, at);
        }
        return h.finalResult();
    }

    fn openDirFd(s: *Store) !*std.c.DIR {
        const dfd = std.c.openat(s.fd, ".", .{ .ACCMODE = .RDONLY, .DIRECTORY = true, .CLOEXEC = true }, @as(std.c.mode_t, 0));
        if (dfd < 0) return error.IdentityCacheUnsafe;
        return std.c.fdopendir(dfd) orelse {
            _ = std.c.close(dfd);
            return error.IdentityCacheUnsafe;
        };
    }

    fn removeStalePartials(s: *Store) !void {
        const d = try s.openDirFd();
        defer _ = std.c.closedir(d);
        var removed = false;
        while (std.c.readdir(d)) |ent| {
            const entry = std.mem.span(@as([*:0]const u8, @ptrCast(&ent.name)));
            if (!std.mem.startsWith(u8, entry, partial_prefix)) continue;
            _ = std.c.unlinkat(s.fd, entry, 0);
            removed = true;
        }
        if (removed) try s.syncDirectory();
    }

    fn scanRecords(s: *Store) !Scan {
        const d = try s.openDirFd();
        defer _ = std.c.closedir(d);
        var scan: Scan = .{};
        while (std.c.readdir(d)) |ent| {
            const entry = std.mem.span(@as([*:0]const u8, @ptrCast(&ent.name)));
            if (!isRecordName(entry)) continue;
            const st = try statxAt(s.fd, entry, at_symlink_nofollow);
            if (st.mode & s_ifmt != s_ifreg) return error.IdentityCacheUnsafe;
            scan.count += 1;
            const t: Time = .{ .s = st.mtime.sec, .ns = st.mtime.nsec };
            if (scan.oldest == null or t.before(scan.oldest_time)) {
                var n: Name = @splat(0);
                @memcpy(n[0..record_name_bytes], entry);
                scan.oldest = n;
                scan.oldest_time = t;
            }
        }
        return scan;
    }

    fn prune(s: *Store) !void {
        while (true) {
            const scan = try s.scanRecords();
            if (scan.count <= s.limit) return;
            const victim = scan.oldest orelse return;
            if (std.c.unlinkat(s.fd, victim[0..record_name_bytes :0], 0) != 0) return error.IdentityCacheUnsafe;
            try s.syncDirectory();
        }
    }

    fn commit(s: *Store, name: Name, k: [32]u8, value: [32]u8) !void {
        try s.lock();
        defer s.unlock();
        if (s.hook) |hook| if (hook.critical) |probe| probe(hook.ptr, s);
        try s.removeStalePartials();
        var partial_buf: [partial_name_bytes + 1]u8 = undefined;
        const partial = partialName(&partial_buf, k);
        const pfd = std.c.openat(s.fd, partial, .{ .ACCMODE = .WRONLY, .CREAT = true, .EXCL = true, .NOFOLLOW = true, .CLOEXEC = true }, @as(std.c.mode_t, 0o600));
        if (pfd < 0) return error.IdentityCacheUnsafe;
        var published = false;
        defer if (!published) {
            _ = std.c.unlinkat(s.fd, partial, 0);
        };
        defer _ = std.c.close(pfd);
        const rec = encodeRecord(k, value);
        if (std.c.pwrite(pfd, &rec, rec.len, 0) != @as(isize, record_bytes)) return error.IdentityCacheUnsafe;
        if (s.fsyncEvent(.payload_fsync, pfd) != 0) return error.IdentityCacheUnsafe;
        s.advise(pfd, 0, record_bytes);
        if (std.c.renameat(s.fd, partial, s.fd, name[0..record_name_bytes :0]) != 0) return error.IdentityCacheUnsafe;
        s.emitEvent(.renamed, s.fd);
        published = true;
        try s.syncDirectory();
        s.hashes += 1;
        try s.prune();
    }
};

fn exitCodeOf(status: i32) ?u8 {
    if (status & 0x7f != 0) return null;
    return @intCast((status >> 8) & 0xff);
}

const Fixture = struct {
    tmp: std.testing.TmpDir,
    base: [:0]u8,
    root: [:0]u8,
    source: [:0]u8,
    store: Store,
    fn open() !Fixture {
        var tmp = std.testing.tmpDir(.{});
        errdefer tmp.cleanup();
        var cwd: [4096]u8 = undefined;
        const n = std.c.readlink("/proc/self/cwd", &cwd, cwd.len);
        if (n <= 0 or n >= cwd.len) return error.SyntheticRootUnavailable;
        const base = try std.fmt.allocPrintSentinel(std.testing.allocator, "{s}/.zig-cache/tmp/{s}/identity", .{ cwd[0..@intCast(n)], tmp.sub_path }, 0);
        errdefer std.testing.allocator.free(base);
        if (std.c.mkdir(base, 0o700) != 0) return error.SyntheticRootUnavailable;
        const root = try std.fmt.allocPrintSentinel(std.testing.allocator, "{s}/spill", .{base}, 0);
        errdefer std.testing.allocator.free(root);
        const source = try std.fmt.allocPrintSentinel(std.testing.allocator, ".zig-cache/tmp/{s}/source.safetensors", .{tmp.sub_path}, 0);
        errdefer std.testing.allocator.free(source);
        const store = try Store.open(std.testing.io, root);
        return .{ .tmp = tmp, .base = base, .root = root, .source = source, .store = store };
    }
    fn close(f: *Fixture) void {
        f.store.close();
        std.testing.allocator.free(f.base);
        std.testing.allocator.free(f.root);
        std.testing.allocator.free(f.source);
        f.tmp.cleanup();
    }
    fn write(f: *Fixture, value: u8) !void {
        var bytes: [8192]u8 = @splat(value);
        @memset(bytes[0..8], 0); // bounded empty safetensors header, synthetic payload only
        try f.tmp.dir.writeFile(std.testing.io, .{ .sub_path = "source.safetensors", .data = &bytes });
    }
};

test "digest cache key includes canonical path size precise times inode device and index" {
    const m: Meta = .{ .size = 8192, .mtime_s = 10, .mtime_ns = 20, .ctime_s = 30, .ctime_ns = 40, .inode = 50, .dev_major = 60, .dev_minor = 70 };
    const base = key("/synthetic/weights", m, @splat(0));
    inline for (.{ "size", "mtime_s", "mtime_ns", "ctime_s", "ctime_ns", "inode", "dev_major", "dev_minor" }) |field| {
        var changed = m;
        @field(changed, field) += 1;
        try std.testing.expect(!std.mem.eql(u8, &base, &key("/synthetic/weights", changed, @splat(0))));
    }
    try std.testing.expect(!std.mem.eql(u8, &base, &key("/synthetic/other", m, @splat(0))));
    try std.testing.expect(!std.mem.eql(u8, &base, &key("/synthetic/weights", m, @splat(1))));
}

test "digest cache reuses complete hash and restored mtime mutation invalidates" {
    var f = try Fixture.open();
    defer f.close();
    try f.write(7);
    const first = try f.store.digest(f.source, @splat(0));
    try std.testing.expectEqualSlices(u8, &first.digest, &(try f.store.digest(f.source, @splat(0))).digest);
    try std.testing.expectEqual(@as(u64, 1), f.store.hashes);
    try std.testing.expectEqual(@as(u64, 1), f.store.hits);
    const fd = std.c.open(f.source, .{ .ACCMODE = .RDONLY, .CLOEXEC = true }, @as(std.c.mode_t, 0));
    try std.testing.expect(fd >= 0);
    defer _ = std.c.close(fd);
    const before = try metadata(fd);
    try f.write(8);
    const times = [2]std.c.timespec{ .{ .sec = before.mtime_s, .nsec = before.mtime_ns }, .{ .sec = before.mtime_s, .nsec = before.mtime_ns } };
    try std.testing.expectEqual(@as(c_int, 0), std.c.utimensat(std.posix.AT.FDCWD, f.source, &times, 0));
    const after = try metadata(fd);
    try std.testing.expectEqual(before.mtime_s, after.mtime_s);
    try std.testing.expectEqual(before.mtime_ns, after.mtime_ns);
    const changed = try f.store.digest(f.source, @splat(0));
    try std.testing.expect(!std.mem.eql(u8, &first.digest, &changed.digest));
    try std.testing.expectEqual(@as(u64, 2), f.store.hashes);
    _ = try f.store.digest(f.source, @splat(1));
    try std.testing.expectEqual(@as(u64, 3), f.store.hashes);
}

test "digest cache rejects unsafe roots and symlink records" {
    var f = try Fixture.open();
    defer f.close();
    try std.testing.expectEqual(@as(c_int, 0), std.c.chmod(f.root, 0o777));
    const unsafe = Store.open(std.testing.io, f.root);
    if (unsafe) |copy| {
        var owned = copy;
        owned.close();
    } else |_| {}
    try std.testing.expectError(error.IdentityCacheUnsafe, unsafe);
    try std.testing.expectEqual(@as(c_int, 0), std.c.chmod(f.root, 0o700));
    try f.write(7);
    _ = try f.store.digest(f.source, @splat(0));
    const name = f.store.last_record[0..67 :0];
    _ = std.c.unlinkat(f.store.fd, name, 0);
    try std.testing.expectEqual(@as(c_int, 0), std.c.symlinkat("/no/such/synthetic-target", f.store.fd, name));
    try std.testing.expectError(error.IdentityCacheUnsafe, f.store.digest(f.source, @splat(0)));
}

test "digest cache checksum and truncated records rehash instead of trusting partials" {
    var f = try Fixture.open();
    defer f.close();
    try f.write(7);
    const first = try f.store.digest(f.source, @splat(0));
    const name = f.store.last_record[0..67 :0];
    const fd = std.c.openat(f.store.fd, name, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true, .CLOEXEC = true }, @as(std.c.mode_t, 0o600));
    try std.testing.expect(fd >= 0);
    var corrupt: [record_bytes]u8 = @splat(0);
    @memcpy(corrupt[0..8], record_magic);
    try std.testing.expectEqual(@as(isize, record_bytes), std.c.pwrite(fd, &corrupt, corrupt.len, 0));
    _ = std.c.close(fd);
    try std.testing.expectEqualSlices(u8, &first.digest, &(try f.store.digest(f.source, @splat(0))).digest);
    try std.testing.expectEqual(@as(u64, 2), f.store.hashes);
    const short = std.c.openat(f.store.fd, name, .{ .ACCMODE = .WRONLY, .TRUNC = true, .CLOEXEC = true }, @as(std.c.mode_t, 0));
    try std.testing.expect(short >= 0);
    _ = std.c.close(short);
    try std.testing.expectEqualSlices(u8, &first.digest, &(try f.store.digest(f.source, @splat(0))).digest);
    try std.testing.expectEqual(@as(u64, 3), f.store.hashes);
    _ = try f.store.digest(f.source, @splat(0));
    try std.testing.expectEqual(@as(u64, 3), f.store.hashes);
    try std.testing.expectEqual(@as(u64, 1), f.store.hits);
}

test "digest cache rejects source changes while hashing" {
    var f = try Fixture.open();
    defer f.close();
    try f.write(7);
    const writer = std.c.open(f.source, .{ .ACCMODE = .WRONLY, .CLOEXEC = true }, @as(std.c.mode_t, 0));
    try std.testing.expect(writer >= 0);
    defer _ = std.c.close(writer);
    const Hook = struct {
        writer: c_int,
        changed: bool = false,
        fn after(ptr: *anyopaque, _: c_int, _: u64) void {
            const self: *@This() = @ptrCast(@alignCast(ptr));
            if (self.changed) return;
            self.changed = true;
            const byte = [_]u8{9};
            _ = std.c.pwrite(self.writer, &byte, byte.len, 100);
        }
    };
    var hook: Hook = .{ .writer = writer };
    f.store.hook = .{ .ptr = &hook, .after_block = Hook.after };
    try std.testing.expectError(error.SourceChanged, f.store.digest(f.source, @splat(0)));
    try std.testing.expect(hook.changed);
    try std.testing.expectEqual(@as(u64, 0), f.store.hashes);
    try std.testing.expectEqual(@as(u64, 0), f.store.hits);
}

test "digest cache footprint remains bounded with durable fixed records" {
    var f = try Fixture.open();
    defer f.close();
    try f.write(7);
    f.store.limit = 3;
    for (0..6) |i| _ = try f.store.digest(f.source, @splat(@intCast(i)));
    var count: usize = 0;
    var pb: [4096]u8 = undefined;
    const dir = try std.fmt.bufPrintSentinel(&pb, "{s}/{s}", .{ f.root, node }, 0);
    const d = std.c.opendir(dir) orelse return error.TestUnexpectedResult;
    defer _ = std.c.closedir(d);
    while (std.c.readdir(d)) |ent| {
        const name = std.mem.sliceTo(&ent.name, 0);
        if (name[0] == '.') continue;
        try std.testing.expect(std.mem.endsWith(u8, name, ".id"));
        const fd = std.c.openat(f.store.fd, @ptrCast(name.ptr), .{ .ACCMODE = .RDONLY, .NOFOLLOW = true, .CLOEXEC = true }, @as(std.c.mode_t, 0));
        try std.testing.expect(fd >= 0);
        const m = try metadata(fd);
        _ = std.c.close(fd);
        try std.testing.expectEqual(@as(u64, record_bytes), m.size);
        count += 1;
    }
    try std.testing.expectEqual(@as(usize, 3), count);
}

test "digest cache advises pages away after bounded reads and synced writes" {
    var f = try Fixture.open();
    defer f.close();
    try f.write(7);
    _ = try f.store.digest(f.source, @splat(0));
    try std.testing.expect(f.store.advice_calls >= 2);
}

test "digest cache rejects a source changed between hit lookup and reuse" {
    var f = try Fixture.open();
    defer f.close();
    try f.write(7);
    _ = try f.store.digest(f.source, @splat(0));
    const writer = std.c.open(f.source, .{ .ACCMODE = .WRONLY, .CLOEXEC = true }, @as(std.c.mode_t, 0));
    try std.testing.expect(writer >= 0);
    defer _ = std.c.close(writer);
    const Race = struct {
        writer: c_int,
        fn after(ptr: *anyopaque) void {
            const self: *@This() = @ptrCast(@alignCast(ptr));
            const byte = [_]u8{9};
            _ = std.c.pwrite(self.writer, &byte, byte.len, 100);
        }
        fn ignore(_: *anyopaque, _: c_int, _: u64) void {}
    };
    var race: Race = .{ .writer = writer };
    f.store.hook = .{ .ptr = &race, .after_block = Race.ignore, .after_lookup = Race.after };
    try std.testing.expectError(error.SourceChanged, f.store.digest(f.source, @splat(0)));
    try std.testing.expectEqual(@as(u64, 1), f.store.hashes);
}

test "digest cache refuses a source whose safetensors header overruns the file" {
    var f = try Fixture.open();
    defer f.close();
    try f.write(7);
    const writer = std.c.open(f.source, .{ .ACCMODE = .WRONLY, .CLOEXEC = true }, @as(std.c.mode_t, 0));
    try std.testing.expect(writer >= 0);
    defer _ = std.c.close(writer);
    const head: [8]u8 = @splat(0xff);
    try std.testing.expectEqual(@as(isize, 8), std.c.pwrite(writer, &head, head.len, 0));
    try std.testing.expectError(error.WeightsUnreadable, f.store.digest(f.source, @splat(0)));
    try std.testing.expectEqual(@as(u64, 0), f.store.hashes);
}

test "digest cache treats oversized records as unsafe" {
    var f = try Fixture.open();
    defer f.close();
    try f.write(7);
    _ = try f.store.digest(f.source, @splat(0));
    const name = f.store.last_record[0..67 :0];
    const fd = std.c.openat(f.store.fd, name, .{ .ACCMODE = .WRONLY, .TRUNC = true, .CLOEXEC = true }, @as(std.c.mode_t, 0o600));
    try std.testing.expect(fd >= 0);
    var big: [record_bytes + 1]u8 = @splat(0);
    try std.testing.expectEqual(@as(isize, record_bytes + 1), std.c.pwrite(fd, &big, big.len, 0));
    _ = std.c.close(fd);
    try std.testing.expectError(error.IdentityCacheUnsafe, f.store.digest(f.source, @splat(0)));
}

test "digest cache rejects hard-linked records" {
    var f = try Fixture.open();
    defer f.close();
    try f.write(7);
    _ = try f.store.digest(f.source, @splat(0));
    const name = f.store.last_record[0..67 :0];
    try std.testing.expectEqual(@as(c_int, 0), std.c.linkat(f.store.fd, name, f.store.fd, ".hard-alias", 0));
    try std.testing.expectError(error.IdentityCacheUnsafe, f.store.digest(f.source, @splat(0)));
}

test "digest cache refuses a group or world writable ancestor" {
    var f = try Fixture.open();
    defer f.close();
    const parent = try std.fmt.allocPrintSentinel(std.testing.allocator, "{s}", .{f.base}, 0);
    defer std.testing.allocator.free(parent);
    try std.testing.expectEqual(@as(c_int, 0), std.c.chmod(parent, 0o777));
    try std.testing.expectError(error.IdentityCacheUnsafe, Store.open(std.testing.io, f.root));
    try std.testing.expectEqual(@as(c_int, 0), std.c.chmod(parent, 0o755));
}

test "digest cache removes stale partial files before publishing" {
    var f = try Fixture.open();
    defer f.close();
    try f.write(7);
    const stale = ".partial-0000000000000000000000000000000000000000000000000000000000000000";
    const fd = std.c.openat(f.store.fd, stale, .{ .ACCMODE = .WRONLY, .CREAT = true, .EXCL = true, .NOFOLLOW = true, .CLOEXEC = true }, @as(std.c.mode_t, 0o600));
    try std.testing.expect(fd >= 0);
    _ = std.c.close(fd);
    _ = try f.store.digest(f.source, @splat(0));
    try std.testing.expect(std.c.openat(f.store.fd, stale, .{ .ACCMODE = .RDONLY, .NOFOLLOW = true, .CLOEXEC = true }, @as(std.c.mode_t, 0)) < 0);
}

test "digest cache advice counts header read source reads and synced record write" {
    var f = try Fixture.open();
    defer f.close();
    try f.write(7);
    _ = try f.store.digest(f.source, @splat(0));
    try std.testing.expectEqual(@as(u64, 3), f.store.advice_calls);
}

test "digest cache keys symlinked sources by canonical target" {
    var f = try Fixture.open();
    defer f.close();
    try f.write(7);
    const link = try std.fmt.allocPrintSentinel(std.testing.allocator, ".zig-cache/tmp/{s}/alias.safetensors", .{f.tmp.sub_path}, 0);
    defer std.testing.allocator.free(link);
    try std.testing.expectEqual(@as(c_int, 0), std.c.symlinkat("source.safetensors", std.posix.AT.FDCWD, link));
    const first = try f.store.digest(f.source, @splat(0));
    const via_link = try f.store.digest(link, @splat(0));
    try std.testing.expectEqualSlices(u8, &first.digest, &via_link.digest);
    try std.testing.expectEqual(@as(u64, 1), f.store.hits);
    try std.testing.expectEqual(@as(u64, 1), f.store.hashes);
}

test "digest cache rejects a source path retargeted while hashing" {
    var f = try Fixture.open();
    defer f.close();
    try f.write(7);
    const other = try std.fmt.allocPrintSentinel(std.testing.allocator, ".zig-cache/tmp/{s}/other.safetensors", .{f.tmp.sub_path}, 0);
    defer std.testing.allocator.free(other);
    var bytes: [8192]u8 = @splat(9);
    @memset(bytes[0..8], 0);
    try f.tmp.dir.writeFile(std.testing.io, .{ .sub_path = "other.safetensors", .data = &bytes });
    const link = try std.fmt.allocPrintSentinel(std.testing.allocator, ".zig-cache/tmp/{s}/alias.safetensors", .{f.tmp.sub_path}, 0);
    defer std.testing.allocator.free(link);
    try std.testing.expectEqual(@as(c_int, 0), std.c.symlinkat("source.safetensors", std.posix.AT.FDCWD, link));
    const Retarget = struct {
        link: [:0]const u8,
        fn after(ptr: *anyopaque, _: c_int, _: u64) void {
            const self: *@This() = @ptrCast(@alignCast(ptr));
            _ = std.c.unlinkat(std.posix.AT.FDCWD, self.link, 0);
            _ = std.c.symlinkat("other.safetensors", std.posix.AT.FDCWD, self.link);
        }
    };
    var retarget: Retarget = .{ .link = link };
    f.store.hook = .{ .ptr = &retarget, .after_block = Retarget.after };
    try std.testing.expectError(error.SourceChanged, f.store.digest(link, @splat(0)));
    try std.testing.expectEqual(@as(u64, 0), f.store.hashes);
}

test "digest cache rejects a checksummed record carrying another key" {
    var f = try Fixture.open();
    defer f.close();
    try f.write(7);
    const first = try f.store.digest(f.source, @splat(0));
    const name = f.store.last_record[0..67 :0];
    const fd = std.c.openat(f.store.fd, name, .{ .ACCMODE = .WRONLY, .TRUNC = true, .CLOEXEC = true }, @as(std.c.mode_t, 0o600));
    try std.testing.expect(fd >= 0);
    const foreign = encodeRecord(@splat(1), @splat(2));
    try std.testing.expectEqual(@as(isize, record_bytes), std.c.pwrite(fd, &foreign, foreign.len, 0));
    _ = std.c.close(fd);
    try std.testing.expectEqualSlices(u8, &first.digest, &(try f.store.digest(f.source, @splat(0))).digest);
    try std.testing.expectEqual(@as(u64, 2), f.store.hashes);
}

test "digest cache creates missing private parent directories securely" {
    var f = try Fixture.open();
    defer f.close();
    try f.write(7);
    const deep = try std.fmt.allocPrintSentinel(std.testing.allocator, "{s}/missing/parent/spill", .{f.base}, 0);
    defer std.testing.allocator.free(deep);
    var store = Store.open(std.testing.io, deep) catch |err| {
        try std.testing.expectEqual(@as(anyerror, error.Nothing), @as(anyerror, err));
        unreachable;
    };
    defer store.close();
    _ = try store.digest(f.source, @splat(0));
    try std.testing.expectEqual(@as(u64, 1), store.hashes);
}

test "digest cache warm open caps an oversized footprint and removes stale partials" {
    var f = try Fixture.open();
    defer f.close();
    try f.write(7);
    _ = try f.store.digest(f.source, @splat(0));
    var i: usize = 1;
    while (i <= maximum_records) : (i += 1) {
        var k: [32]u8 = @splat(0);
        std.mem.writeInt(u64, k[0..8], i, .little);
        const name = recordName(k);
        const fd = std.c.openat(f.store.fd, name[0..record_name_bytes :0], .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true, .NOFOLLOW = true, .CLOEXEC = true }, @as(std.c.mode_t, 0o600));
        try std.testing.expect(fd >= 0);
        const rec = encodeRecord(k, k);
        try std.testing.expectEqual(@as(isize, record_bytes), std.c.pwrite(fd, &rec, rec.len, 0));
        _ = std.c.close(fd);
    }
    const stale = ".partial-0000000000000000000000000000000000000000000000000000000000000000";
    const sfd = std.c.openat(f.store.fd, stale, .{ .ACCMODE = .WRONLY, .CREAT = true, .EXCL = true, .NOFOLLOW = true, .CLOEXEC = true }, @as(std.c.mode_t, 0o600));
    try std.testing.expect(sfd >= 0);
    _ = std.c.close(sfd);
    var warm = try Store.open(std.testing.io, f.root);
    defer warm.close();
    var count: usize = 0;
    var pb: [4096]u8 = undefined;
    const dir = try std.fmt.bufPrintSentinel(&pb, "{s}/{s}", .{ f.root, node }, 0);
    const d = std.c.opendir(dir) orelse return error.TestUnexpectedResult;
    defer _ = std.c.closedir(d);
    while (std.c.readdir(d)) |ent| {
        const name = std.mem.sliceTo(&ent.name, 0);
        if (name[0] == '.') continue;
        count += 1;
    }
    try std.testing.expectEqual(@as(usize, maximum_records), count);
    try std.testing.expect(std.c.openat(warm.fd, stale, .{ .ACCMODE = .RDONLY, .NOFOLLOW = true, .CLOEXEC = true }, @as(std.c.mode_t, 0)) < 0);
}

test "digest cache rejects a dot-suffixed root with a non-private mode" {
    var f = try Fixture.open();
    defer f.close();
    try std.testing.expectEqual(@as(c_int, 0), std.c.chmod(f.root, 0o755));
    const dotted = try std.fmt.allocPrintSentinel(std.testing.allocator, "{s}/.", .{f.root}, 0);
    defer std.testing.allocator.free(dotted);
    const result = Store.open(std.testing.io, dotted);
    if (result) |copy| {
        var owned = copy;
        owned.close();
    } else |_| {}
    try std.testing.expectError(error.IdentityCacheUnsafe, result);
    try std.testing.expectEqual(@as(c_int, 0), std.c.chmod(f.root, 0o700));
}

test "digest cache rejects a record whose digest bytes no longer match its checksum" {
    var f = try Fixture.open();
    defer f.close();
    try f.write(7);
    const first = try f.store.digest(f.source, @splat(0));
    const name = f.store.last_record[0..67 :0];
    const fd = std.c.openat(f.store.fd, name, .{ .ACCMODE = .WRONLY, .CLOEXEC = true }, @as(std.c.mode_t, 0));
    try std.testing.expect(fd >= 0);
    const flip: [1]u8 = .{first.digest[0] ^ 0xff};
    try std.testing.expectEqual(@as(isize, 1), std.c.pwrite(fd, &flip, 1, 40));
    _ = std.c.close(fd);
    try std.testing.expectEqualSlices(u8, &first.digest, &(try f.store.digest(f.source, @splat(0))).digest);
    try std.testing.expectEqual(@as(u64, 2), f.store.hashes);
}

fn noBlock(_: *anyopaque, _: c_int, _: u64) void {}

const MutexProbe = struct {
    locked: u64 = 0,
    unlocked: u64 = 0,
    fn critical(ptr: *anyopaque, s: *Store) void {
        const self: *@This() = @ptrCast(@alignCast(ptr));
        if (s.mutex.tryLock()) {
            self.unlocked += 1;
            s.mutex.unlock(s.io);
        } else {
            self.locked += 1;
        }
    }
};

const FlockProbe = struct {
    path: [:0]const u8,
    blocked: u64 = 0,
    acquired: u64 = 0,
    child_failed: u64 = 0,
    fn critical(ptr: *anyopaque, _: *Store) void {
        const self: *@This() = @ptrCast(@alignCast(ptr));
        const forked: isize = @bitCast(std.os.linux.fork());
        if (forked == 0) {
            const fd = std.c.open(self.path, .{ .ACCMODE = .RDONLY, .DIRECTORY = true, .NOFOLLOW = true, .CLOEXEC = true }, @as(std.c.mode_t, 0));
            if (fd < 0) std.os.linux.exit_group(3);
            const rc = std.os.linux.flock(fd, std.posix.LOCK.EX | std.posix.LOCK.NB);
            std.os.linux.exit_group(if (rc == 0) 2 else 0);
        }
        if (forked < 0) {
            self.child_failed += 1;
            return;
        }
        var status: i32 = 0;
        const waited: isize = @bitCast(std.os.linux.waitpid(@intCast(forked), &status, 0));
        if (waited != forked) {
            self.child_failed += 1;
            return;
        }
        switch (exitCodeOf(status) orelse 255) {
            0 => self.blocked += 1,
            2 => self.acquired += 1,
            else => self.child_failed += 1,
        }
    }
};

const SyncRecorder = struct {
    const Sample = struct { records: usize, partials: usize };
    samples: [32]Sample = undefined,
    len: usize = 0,
    fn onSync(ptr: *anyopaque, fd: c_int) void {
        const self: *@This() = @ptrCast(@alignCast(ptr));
        if (self.len == self.samples.len) return;
        self.samples[self.len] = .{ .records = countEntries(fd, false), .partials = countEntries(fd, true) };
        self.len += 1;
    }
};

fn countEntries(fd: c_int, partial: bool) usize {
    const dfd = std.c.openat(fd, ".", .{ .ACCMODE = .RDONLY, .DIRECTORY = true, .CLOEXEC = true }, @as(std.c.mode_t, 0));
    if (dfd < 0) return std.math.maxInt(usize);
    const d = std.c.fdopendir(dfd) orelse {
        _ = std.c.close(dfd);
        return std.math.maxInt(usize);
    };
    defer _ = std.c.closedir(d);
    var n: usize = 0;
    while (std.c.readdir(d)) |ent| {
        const entry = std.mem.span(@as([*:0]const u8, @ptrCast(&ent.name)));
        if (partial) {
            if (std.mem.startsWith(u8, entry, partial_prefix)) n += 1;
        } else if (isRecordName(entry)) {
            n += 1;
        }
    }
    return n;
}

test "digest cache holds the store mutex in every locked critical section" {
    var f = try Fixture.open();
    defer f.close();
    try f.write(7);
    var probe: MutexProbe = .{};
    f.store.hook = .{ .ptr = &probe, .after_block = noBlock, .critical = MutexProbe.critical };
    _ = try f.store.digest(f.source, @splat(0));
    _ = try f.store.digest(f.source, @splat(0));
    try f.store.maintain();
    try std.testing.expectEqual(@as(u64, 0), probe.unlocked);
    try std.testing.expect(probe.locked >= 4);
    try std.testing.expect(f.store.mutex.tryLock());
    f.store.mutex.unlock(f.store.io);
}

test "digest cache holds a fresh cross-process directory flock in every locked critical section" {
    var f = try Fixture.open();
    defer f.close();
    try f.write(7);
    const node_path = try std.fmt.allocPrintSentinel(std.testing.allocator, "{s}/{s}", .{ f.root, node }, 0);
    defer std.testing.allocator.free(node_path);
    var probe: FlockProbe = .{ .path = node_path };
    f.store.hook = .{ .ptr = &probe, .after_block = noBlock, .critical = FlockProbe.critical };
    _ = try f.store.digest(f.source, @splat(0));
    _ = try f.store.digest(f.source, @splat(0));
    try std.testing.expectEqual(@as(u64, 0), probe.acquired);
    try std.testing.expectEqual(@as(u64, 0), probe.child_failed);
    try std.testing.expect(probe.blocked >= 3);
    const after = std.c.open(node_path, .{ .ACCMODE = .RDONLY, .DIRECTORY = true, .NOFOLLOW = true, .CLOEXEC = true }, @as(std.c.mode_t, 0));
    try std.testing.expect(after >= 0);
    defer _ = std.c.close(after);
    try std.testing.expectEqual(@as(usize, 0), std.os.linux.flock(after, std.posix.LOCK.EX | std.posix.LOCK.NB));
}

test "digest cache syncs the directory after the record rename" {
    var f = try Fixture.open();
    defer f.close();
    try f.write(7);
    var rec: SyncRecorder = .{};
    f.store.hook = .{ .ptr = &rec, .after_block = noBlock, .dir_sync = SyncRecorder.onSync };
    _ = try f.store.digest(f.source, @splat(0));
    var published = false;
    for (rec.samples[0..rec.len]) |s| {
        if (s.records == 1) published = true;
    }
    try std.testing.expect(published);
}

test "digest cache syncs the directory after prune deletions" {
    var f = try Fixture.open();
    defer f.close();
    try f.write(7);
    f.store.limit = 3;
    var rec: SyncRecorder = .{};
    f.store.hook = .{ .ptr = &rec, .after_block = noBlock, .dir_sync = SyncRecorder.onSync };
    for (0..4) |i| _ = try f.store.digest(f.source, @splat(@intCast(i)));
    var saw_four = false;
    var shrank = false;
    for (rec.samples[0..rec.len]) |s| {
        if (s.records == 4) {
            saw_four = true;
        } else if (saw_four and s.records == 3) {
            shrank = true;
        }
    }
    try std.testing.expect(shrank);
}

test "digest cache syncs the directory after stale partial cleanup" {
    var f = try Fixture.open();
    defer f.close();
    try f.write(7);
    const stale = ".partial-0000000000000000000000000000000000000000000000000000000000000000";
    const sfd = std.c.openat(f.store.fd, stale, .{ .ACCMODE = .WRONLY, .CREAT = true, .EXCL = true, .NOFOLLOW = true, .CLOEXEC = true }, @as(std.c.mode_t, 0o600));
    try std.testing.expect(sfd >= 0);
    _ = std.c.close(sfd);
    var rec: SyncRecorder = .{};
    f.store.hook = .{ .ptr = &rec, .after_block = noBlock, .dir_sync = SyncRecorder.onSync };
    try f.store.maintain();
    try std.testing.expect(rec.len > 0);
    try std.testing.expectEqual(@as(usize, 0), rec.samples[0].partials);
}

test "digest cache rejects a store directory without private mode" {
    var f = try Fixture.open();
    defer f.close();
    const node_path = try std.fmt.allocPrintSentinel(std.testing.allocator, "{s}/{s}", .{ f.root, node }, 0);
    defer std.testing.allocator.free(node_path);
    try std.testing.expectEqual(@as(c_int, 0), std.c.chmod(node_path, 0o755));
    const result = Store.open(std.testing.io, f.root);
    if (result) |copy| {
        var owned = copy;
        owned.close();
    } else |_| {}
    try std.testing.expectError(error.IdentityCacheUnsafe, result);
    try std.testing.expectEqual(@as(c_int, 0), std.c.chmod(node_path, 0o700));
}

test "digest cache rejects an ancestor owned by another uid" {
    if (std.c.geteuid() != 0) return error.SkipZigTest; // only root can give a directory to another uid
    var f = try Fixture.open();
    defer f.close();
    const owned = try std.fmt.allocPrintSentinel(std.testing.allocator, "{s}/owned", .{f.base}, 0);
    defer std.testing.allocator.free(owned);
    _ = std.c.mkdir(owned, 0o755);
    try std.testing.expectEqual(@as(usize, 0), std.os.linux.chown(owned, 1234, 1234));
    const deep = try std.fmt.allocPrintSentinel(std.testing.allocator, "{s}/spill", .{owned}, 0);
    defer std.testing.allocator.free(deep);
    const result = Store.open(std.testing.io, deep);
    if (result) |copy| {
        var store = copy;
        store.close();
    } else |_| {}
    try std.testing.expectError(error.IdentityCacheUnsafe, result);
}

test "digest cache rejects a relative cache root" {
    var f = try Fixture.open();
    defer f.close();
    const rel = try std.fmt.allocPrintSentinel(std.testing.allocator, ".zig-cache/tmp/{s}/spill", .{f.tmp.sub_path}, 0);
    defer std.testing.allocator.free(rel);
    const result = Store.open(std.testing.io, rel);
    if (result) |copy| {
        var owned = copy;
        owned.close();
    } else |_| {}
    try std.testing.expectError(error.IdentityCacheUnsafe, result);
}

fn boundedChildOutcome(store: *Store, path: [:0]const u8, self_kill: bool) ?u8 {
    const pid: isize = @bitCast(std.os.linux.fork());
    if (pid < 0) return null;
    if (pid == 0) {
        if (self_kill) {
            _ = std.os.linux.kill(std.os.linux.getpid(), .KILL);
            std.os.linux.exit_group(4);
        }
        if (store.digest(path, @splat(0))) |_| {
            std.os.linux.exit_group(2);
        } else |err| {
            std.os.linux.exit_group(if (err == error.WeightsUnreadable) 0 else 3);
        }
    }
    var status: i32 = 0;
    var polls: usize = 0;
    while (polls < 200) : (polls += 1) {
        const r: isize = @bitCast(std.os.linux.waitpid(@intCast(pid), &status, std.os.linux.W.NOHANG));
        if (r == pid) return exitCodeOf(status);
        if (r < 0) return null;
        const pause: std.os.linux.timespec = .{ .sec = 0, .nsec = 10_000_000 };
        _ = std.os.linux.nanosleep(&pause, null);
    }
    _ = std.os.linux.kill(@intCast(pid), .KILL);
    _ = std.os.linux.waitpid(@intCast(pid), &status, 0);
    return null;
}

/// True only for a normal exit 0 (WeightsUnreadable) inside the wait budget.
fn digestWithinBudget(store: *Store, path: [:0]const u8) bool {
    return boundedChildOutcome(store, path, false) == @as(?u8, 0);
}

test "digest cache rejects a FIFO source without blocking" {
    var f = try Fixture.open();
    defer f.close();
    const fifo = try std.fmt.allocPrintSentinel(std.testing.allocator, ".zig-cache/tmp/{s}/fifo.safetensors", .{f.tmp.sub_path}, 0);
    defer std.testing.allocator.free(fifo);
    try std.testing.expectEqual(@as(usize, 0), std.os.linux.mknodat(std.posix.AT.FDCWD, fifo, 0o010600, 0));
    try std.testing.expect(digestWithinBudget(&f.store, fifo));
}

test "digest cache rejects a symlink to a FIFO without blocking" {
    var f = try Fixture.open();
    defer f.close();
    const fifo = try std.fmt.allocPrintSentinel(std.testing.allocator, ".zig-cache/tmp/{s}/fifo.safetensors", .{f.tmp.sub_path}, 0);
    defer std.testing.allocator.free(fifo);
    try std.testing.expectEqual(@as(usize, 0), std.os.linux.mknodat(std.posix.AT.FDCWD, fifo, 0o010600, 0));
    const link = try std.fmt.allocPrintSentinel(std.testing.allocator, ".zig-cache/tmp/{s}/fifo-alias.safetensors", .{f.tmp.sub_path}, 0);
    defer std.testing.allocator.free(link);
    try std.testing.expectEqual(@as(c_int, 0), std.c.symlinkat("fifo.safetensors", std.posix.AT.FDCWD, link));
    try std.testing.expect(digestWithinBudget(&f.store, link));
}

const OpenFlagProbe = struct {
    requested_nonblock: ?bool = null,
    fn observe(ptr: *anyopaque, flags: *std.c.O) void {
        const self: *@This() = @ptrCast(@alignCast(ptr));
        self.requested_nonblock = flags.NONBLOCK;
        // Safety only: force the open non-blocking so a regressed build cannot hang the runner.
        flags.NONBLOCK = true;
    }
};

test "digest cache requests a nonblocking source open for FIFO inputs" {
    var f = try Fixture.open();
    defer f.close();
    const fifo = try std.fmt.allocPrintSentinel(std.testing.allocator, ".zig-cache/tmp/{s}/observed-fifo.safetensors", .{f.tmp.sub_path}, 0);
    defer std.testing.allocator.free(fifo);
    try std.testing.expectEqual(@as(usize, 0), std.os.linux.mknodat(std.posix.AT.FDCWD, fifo, 0o010600, 0));
    var probe: OpenFlagProbe = .{};
    f.store.hook = .{ .ptr = &probe, .after_block = noBlock, .open_flags = OpenFlagProbe.observe };
    try std.testing.expectError(error.WeightsUnreadable, f.store.digest(fifo, @splat(0)));
    try std.testing.expectEqual(@as(?bool, true), probe.requested_nonblock);
}

const EventLog = struct {
    seq: [8]Event = undefined,
    fds: [8]c_int = undefined,
    kinds: [8]u32 = undefined,
    len: usize = 0,
    fail_directory: bool = false,
    fn observe(ptr: *anyopaque, ev: Event, fd: c_int) ?c_int {
        const self: *@This() = @ptrCast(@alignCast(ptr));
        if (self.len < self.seq.len) {
            self.seq[self.len] = ev;
            self.fds[self.len] = fd;
            var st: linux.Statx = undefined;
            self.kinds[self.len] = if (std.c.statx(fd, "", at_empty_path, statx_mask, &st) == 0) st.mode & 0o170000 else 0;
            self.len += 1;
        }
        if (ev == .directory_fsync and self.fail_directory) return -1;
        return null;
    }
    fn indexOf(self: *const @This(), ev: Event) ?usize {
        for (self.seq[0..self.len], 0..) |e, i| {
            if (e == ev) return i;
        }
        return null;
    }
};

test "digest cache treats a signaled bounded child as a failure" {
    var f = try Fixture.open();
    defer f.close();
    try std.testing.expectEqual(@as(?u8, null), boundedChildOutcome(&f.store, f.source, true));
}

test "digest cache decodes only normal exits as exit codes" {
    try std.testing.expectEqual(@as(?u8, null), exitCodeOf(9));
    try std.testing.expectEqual(@as(?u8, 0), exitCodeOf(0));
    try std.testing.expectEqual(@as(?u8, 3), exitCodeOf(0x300));
}

test "digest cache orders payload fsync before rename and directory fsync after rename" {
    var f = try Fixture.open();
    defer f.close();
    try f.write(7);
    var log: EventLog = .{};
    f.store.hook = .{ .ptr = &log, .after_block = noBlock, .event = EventLog.observe };
    _ = try f.store.digest(f.source, @splat(0));
    const payload = log.indexOf(.payload_fsync);
    const renamed = log.indexOf(.renamed);
    const directory = log.indexOf(.directory_fsync);
    try std.testing.expect(payload != null and renamed != null and directory != null);
    try std.testing.expect(payload.? < renamed.?);
    try std.testing.expect(renamed.? < directory.?);
    try std.testing.expectEqual(f.store.fd, log.fds[directory.?]);
    try std.testing.expectEqual(@as(u32, 0o100000), log.kinds[payload.?]);
    try std.testing.expect(log.fds[payload.?] != f.store.fd);
    try std.testing.expectEqual(@as(u32, 0o040000), log.kinds[directory.?]);
}

test "digest cache fsync wrapper reports the kernel error for an invalid descriptor" {
    var f = try Fixture.open();
    defer f.close();
    try std.testing.expect(f.store.hook == null);
    try std.testing.expect(f.store.fsyncEvent(.payload_fsync, -1) != 0);
    try std.testing.expect(f.store.fsyncEvent(.payload_fsync, f.store.fd) == 0);
}

test "digest cache fails closed when the directory fsync fails" {
    var f = try Fixture.open();
    defer f.close();
    try f.write(7);
    var log: EventLog = .{ .fail_directory = true };
    f.store.hook = .{ .ptr = &log, .after_block = noBlock, .event = EventLog.observe };
    try std.testing.expectError(error.IdentityCacheUnsafe, f.store.digest(f.source, @splat(0)));
    try std.testing.expectEqual(@as(u64, 0), f.store.hashes);
}
