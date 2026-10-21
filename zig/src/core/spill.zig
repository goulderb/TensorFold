//! Spill tier: kept prompt states written to local disk when they leave the prompt cache, read back when a later
//! prompt extends them, also after a restart (docs/spill-tier.md). Nothing here knows a model: a family hands a state
//! over as its tokens, a meta blob and a payload `Source`, and reads it back through a `Sink`.
//!
//! A port of the GLM spill tier by Robert Wojciechowski (wojo, https://github.com/wojo), Mia's patch 0088 for
//! MiaAI-Lab's GLM-5.3-Flash TensorFold recipe (tensorfold/cuda/spill.py, Apache-2.0): its settings, its LRU by bytes
//! under a cap and a free-disk floor, its write-then-rename files and its one-directory-per-build layout. The file and
//! directory conventions also follow upstream TensorFold's --learn (zig/src/core/prompt_imprint.zig).
//! Local fork changes add model-neutral source/sink interfaces, Nemotron/TP2 adapters and Windows/RAM tiers.
//! Original notices and transitive design credits are retained in LICENSES/donors/README.md.
const std = @import("std");
const builtin = @import("builtin");
const Allocator = std.mem.Allocator;
const fs = @import("spill_fs.zig");
pub const Read = @import("spill_read.zig").Read;

/// The tier's files on POSIX and on Windows (spill_fs.zig): every file made, read and deleted in a directory checked
/// private, no link followed, written then renamed.
pub const supported = true;

pub const version: u32 = 1;
pub const header_bytes = 4096;
const magic = "TFSPILL1";
pub const meta_most = 3072;
const hash_seed: u64 = 0x5b11;
const extension = ".tfs";

/// TENSORFOLD_SPILL_*: every rank reads the same values (the ranks compare `text` at start).
pub const Settings = struct {
    dir: []const u8,
    cap: u64,
    highwater: f64 = 0.70,
    min_tokens: u32 = 8192,
    min_free: u64 = 50 << 30,
    flush_s: f64 = 60,
    keep_builds: u32 = 1,
    /// TENSORFOLD_SPILL_RAM_GIB: states a family hands over whole (`Tier.adopt`) stay in memory up to this many
    /// bytes and are read back from there; each is also written behind to disk, so a stop that ends the process
    /// (a kill, a crash) keeps them. Past the cap the least recently used leave memory once their file is in place.
    /// 0: every state goes to disk alone.
    ram_cap: u64 = 0,

    /// The tier's settings from `env.get(name)`, null when TENSORFOLD_SPILL_GIB is unset or 0; `dir` is allocated.
    /// Every number is finite and in range before it is converted: NaN, infinities, negatives and sizes past
    /// `most_gib` are refused, never clamped or truncated.
    pub fn fromEnv(a: Allocator, env: anytype) !?Settings {
        const gib = try gibs(env.get("TENSORFOLD_SPILL_GIB") orelse return null);
        if (gib == 0) return null;
        var s: Settings = .{ .dir = undefined, .cap = @intFromFloat(gib * (1 << 30)) };
        if (env.get("TENSORFOLD_SPILL_HIGHWATER")) |v| s.highwater = try float(v);
        if (!(s.highwater > 0 and s.highwater <= 1)) return error.BadSpillHighwater;
        if (env.get("TENSORFOLD_SPILL_MIN_TOKENS")) |v| s.min_tokens = std.fmt.parseInt(u32, trim(v), 10) catch return error.BadSpillSetting;
        if (env.get("TENSORFOLD_SPILL_MIN_FREE_GIB")) |v| s.min_free = @intFromFloat(try gibs(v) * (1 << 30));
        if (env.get("TENSORFOLD_SPILL_FLUSH_S")) |v| {
            s.flush_s = try float(v);
            if (s.flush_s < 0 or s.flush_s > most_flush_s) return error.BadSpillSetting;
        }
        if (env.get("TENSORFOLD_SPILL_KEEP_BUILDS")) |v| s.keep_builds = std.fmt.parseInt(u32, trim(v), 10) catch return error.BadSpillSetting;
        if (env.get("TENSORFOLD_SPILL_RAM_GIB")) |v| s.ram_cap = @intFromFloat(try gibs(v) * (1 << 30));
        s.dir = if (env.get("TENSORFOLD_SPILL_DIR")) |d| try a.dupe(u8, trim(d)) else if (fs.windows) blk: {
            const local = env.get("LOCALAPPDATA") orelse return error.NoSpillDir;
            break :blk try std.fmt.allocPrint(a, "{s}/tensorfold/spill", .{local});
        } else blk: {
            const home = env.get("HOME") orelse return error.NoSpillDir;
            break :blk try std.fmt.allocPrint(a, "{s}/.cache/tensorfold/spill", .{home});
        };
        if (!fs.absolute(s.dir)) {
            a.free(s.dir);
            return error.SpillDirNotAbsolute;
        }
        return s;
    }

    /// What both ranks must share, as text for the ranks' settings comparison.
    pub fn text(s: Settings, buf: []u8) ![]const u8 {
        return std.fmt.bufPrint(buf, "spill={d}/{d}/{d}/{d}/{d}/{d}/{d}", .{ s.cap >> 20, s.highwater, s.min_tokens, s.min_free >> 20, s.flush_s, s.keep_builds, s.ram_cap >> 20 });
    }

    /// The largest GiB count a setting takes (1 PiB): its byte count fits u64 with room to spare.
    pub const most_gib: f64 = 1 << 20;
    /// The longest clean-stop flush (a week).
    pub const most_flush_s: f64 = 7 * 24 * 3600;

    fn float(v: []const u8) !f64 {
        const x = std.fmt.parseFloat(f64, trim(v)) catch return error.BadSpillSetting;
        return if (std.math.isFinite(x)) x else error.BadSpillSetting;
    }

    fn gibs(v: []const u8) !f64 {
        const g = try float(v);
        return if (g >= 0 and g <= most_gib) g else error.BadSpillSetting;
    }

    fn trim(v: []const u8) []const u8 {
        return std.mem.trim(u8, v, " \t\r\n");
    }
};

/// The process environment, for Settings.fromEnv.
pub const ProcessEnv = struct {
    pub fn get(_: ProcessEnv, name: [:0]const u8) ?[]const u8 {
        return if (std.c.getenv(name)) |v| std.mem.span(v) else null;
    }
};

/// A stored state's name: its tokens' hash and its position (the file `<key>-<at>.tfs`).
pub const Name = struct {
    key: u64,
    at: u32,

    pub fn of(tokens: []const u32, at: u32) Name {
        return .{ .key = std.hash.Wyhash.hash(hash_seed, std.mem.sliceAsBytes(tokens)), .at = at };
    }

    pub fn eql(a: Name, b: Name) bool {
        return a.key == b.key and a.at == b.at;
    }

    fn file(n: Name, buf: []u8) [:0]const u8 {
        return std.fmt.bufPrintSentinel(buf, "{x:0>16}-{d}" ++ extension, .{ n.key, n.at }, 0) catch unreachable;
    }

    fn parse(text: []const u8) ?Name {
        if (!std.mem.endsWith(u8, text, extension)) return null;
        const stem = text[0 .. text.len - extension.len];
        if (stem.len < 18 or stem[16] != '-') return null;
        return .{
            .key = std.fmt.parseInt(u64, stem[0..16], 16) catch return null,
            .at = std.fmt.parseInt(u32, stem[17..], 10) catch return null,
        };
    }
};

/// Where a payload's bytes come from: `read` fills `out` with bytes [offset, offset + out.len), on the writer thread.
pub const Source = struct { ptr: *anyopaque, read: *const fn (ptr: *anyopaque, offset: u64, out: []u8) anyerror!void };

/// Where a payload's bytes go when read back: in order, on the caller's thread.
pub const Sink = struct { ptr: *anyopaque, write: *const fn (ptr: *anyopaque, offset: u64, bytes: []const u8) anyerror!void };

/// A stored name left the tier (the cap or the floor needed room): the owner's hook, on the thread that wrote.
pub const Hook = struct { ptr: *anyopaque, forgot: *const fn (ptr: *anyopaque, name: Name) void };

pub const Stats = struct {
    // best() lookup outcomes, distinct from RAM hits and successful loads.
    disk_hits: u64 = 0,
    disk_misses: u64 = 0,
    disk_evictions: u64 = 0, // stored entries removed for cap/free-floor pressure
    unlinked_bytes: u64 = 0, // logical sizes successfully unlinked, not free-space credit
    // Observed positive free-space delta across successful unlinks, capped by logical size.
    // Open readers defer reclamation; unrelated filesystem activity can affect observations.
    reclaimed_bytes: u64 = 0,
    write_count: u64 = 0, // timed writer attempts, including failures/cancellation
    load_count: u64 = 0, // terminal timed read attempts, including failures
    written: u64 = 0,
    write_failed: u64 = 0,
    skipped: u64 = 0, // already stored or on its way
    full: u64 = 0, // larger than the cap
    no_space: u64 = 0, // below the free-disk floor even with every stored state gone
    deferred: u64 = 0, // early writes put off while writes were queued
    unwritten_short: u64 = 0,
    unwritten_superseded: u64 = 0,
    unwritten_incomplete: u64 = 0,
    replacement_abandoned: u64 = 0, // authoritative replacements exhausted their synchronous write-failure bound
    loaded: u64 = 0,
    load_failed: u64 = 0,
    hash_failed: u64 = 0,
    forgotten: u64 = 0,
    pruned: u64 = 0,
    written_bytes: u64 = 0,
    loaded_bytes: u64 = 0,
    loaded_tokens: u64 = 0, // I/O: tokens of files this rank read back, before any pair commits; not avoided prefill
    write_ns: u64 = 0,
    load_ns: u64 = 0,
    copy_wait_ns: u64 = 0,
    // the memory level (TENSORFOLD_SPILL_RAM_GIB)
    ram_adopted: u64 = 0, // states handed over whole and held in memory
    ram_loaded: u64 = 0, // reads served from memory (a subset of `loaded`)
    ram_demoted: u64 = 0, // held states written on to disk (behind, or for memory room)
    ram_released: u64 = 0, // held states whose memory went for room, their file in place
    ram_promoted: u64 = 0, // states read back from disk and held again (a hot state's next read is memory's)
    ram_dropped: u64 = 0, // held states that left memory with no file (no disk room, a failed write)
};

/// What /health reports of a tier.
pub const Info = struct { stats: Stats, entries: usize, pending: usize, stored_bytes: u64, queued_bytes: u64, disk_free: u64, cap: u64, ram_entries: usize = 0, ram_bytes: u64 = 0, ram_cap: u64 = 0 };

/// The fixed fields of a file's header.
pub const Header = struct {
    identity: [32]u8,
    family: [16]u8,
    rank: u32,
    world: u32,
    name: Name,
    tokens: u32,
    payload_len: u64,
    tokens_hash: u64,
    payload_hash: u64,
    saved: u64,
    meta_len: u32,

    pub fn payloadOffset(tokens: usize) u64 {
        return std.mem.alignForward(u64, header_bytes + 4 * @as(u64, tokens), 4096);
    }

    pub fn fileBytes(h: Header) u64 {
        return payloadOffset(h.tokens) + h.payload_len;
    }

    /// The header's 4096 bytes, `meta` in them.
    pub fn encode(h: Header, blob: []const u8, out: *[header_bytes]u8) void {
        @memset(out, 0);
        @memcpy(out[0..8], magic);
        put(u32, out, 8, version);
        put(u32, out, 12, header_bytes);
        @memcpy(out[16..48], &h.identity);
        @memcpy(out[48..64], &h.family);
        put(u32, out, 64, h.rank);
        put(u32, out, 68, h.world);
        put(u64, out, 72, h.name.key);
        put(u32, out, 80, h.name.at);
        put(u32, out, 84, h.tokens);
        put(u64, out, 88, payloadOffset(h.tokens));
        put(u64, out, 96, h.payload_len);
        put(u64, out, 104, h.tokens_hash);
        put(u64, out, 112, h.payload_hash);
        put(u64, out, 120, h.saved);
        put(u32, out, 128, @intCast(blob.len));
        @memcpy(out[136..][0..blob.len], blob);
        put(u64, out, header_bytes - 8, std.hash.XxHash3.hash(0, out[0 .. header_bytes - 8]));
    }

    /// A header's fields when it is one this tier wrote (`identity`, `rank` of `world`): else why not.
    pub fn decode(in: *const [header_bytes]u8, identity: [32]u8, rank: u32, world: u32) !Header {
        if (!std.mem.eql(u8, in[0..8], magic)) return error.NotASpillFile;
        if (get(u32, in, 8) != version or get(u32, in, 12) != header_bytes) return error.OtherVersion;
        if (get(u64, in, header_bytes - 8) != std.hash.XxHash3.hash(0, in[0 .. header_bytes - 8])) return error.HeaderHash;
        if (!std.mem.eql(u8, in[16..48], &identity)) return error.OtherIdentity;
        const h: Header = .{
            .identity = in[16..48].*,
            .family = in[48..64].*,
            .rank = get(u32, in, 64),
            .world = get(u32, in, 68),
            .name = .{ .key = get(u64, in, 72), .at = get(u32, in, 80) },
            .tokens = get(u32, in, 84),
            .payload_len = get(u64, in, 96),
            .tokens_hash = get(u64, in, 104),
            .payload_hash = get(u64, in, 112),
            .saved = get(u64, in, 120),
            .meta_len = get(u32, in, 128),
        };
        if (h.rank != rank) return error.OtherRank;
        if (h.world != world) return error.OtherWorld; // another TP size's rank files hold other shards
        if (h.meta_len > meta_most or get(u64, in, 88) != payloadOffset(h.tokens)) return error.BadHeader;
        return h;
    }

    pub fn meta(in: *const [header_bytes]u8, h: Header) []const u8 {
        return in[136..][0..h.meta_len];
    }

    fn put(comptime T: type, out: *[header_bytes]u8, at: usize, v: T) void {
        std.mem.writeInt(T, out[at..][0..@sizeOf(T)], v, .little);
    }

    fn get(comptime T: type, in: *const [header_bytes]u8, at: usize) T {
        return std.mem.readInt(T, in[at..][0..@sizeOf(T)], .little);
    }
};

/// One state on its way to disk. `copied`: every payload byte left the source (its memory may go); `done`: on disk or
/// given up. The tier holds one reference until done, the writer's caller one until `release`.
/// A state's bytes held in memory: its payload and meta as a file would hold them. The entry holds one reference, a
/// read or a write of it to disk one each; the last release frees it.
pub const Ram = struct {
    gpa: Allocator,
    payload: []u8,
    meta: []u8,
    hash: u64,
    saved: u64,
    refs: std.atomic.Value(u32) = .init(1),

    fn retain(r: *Ram) *Ram {
        _ = r.refs.fetchAdd(1, .monotonic);
        return r;
    }

    fn release(r: *Ram) void {
        if (r.refs.fetchSub(1, .acq_rel) != 1) return;
        r.gpa.free(r.payload);
        r.gpa.free(r.meta);
        r.gpa.destroy(r);
    }

    /// The payload as a write's Source (on the writer thread; the job holds a reference).
    fn read(ptr: *anyopaque, offset: u64, out: []u8) anyerror!void {
        const r: *Ram = @ptrCast(@alignCast(ptr));
        if (offset > r.payload.len or out.len > r.payload.len - offset) return error.SpillSourceRange;
        @memcpy(out, r.payload[@intCast(offset)..][0..out.len]);
    }
};

pub const Job = struct {
    tier: *Tier,
    name: Name,
    tokens: []u32,
    meta: []u8,
    payload_len: u64,
    source: Source,
    ram: ?*Ram = null, // a held state written on to disk (its source): released with the job
    keep_ram: bool = false, // written behind: the state stays held once its file is in place
    state: std.atomic.Value(u8) = .init(queued),
    cancel: std.atomic.Value(bool) = .init(false),
    refs: std.atomic.Value(u32) = .init(2),
    ok: bool = false,

    const queued = 0;
    const copied_state = 1;
    const done_state = 2;

    pub fn copied(j: *const Job) bool {
        return j.state.load(.acquire) >= copied_state;
    }

    pub fn done(j: *const Job) bool {
        return j.state.load(.acquire) == done_state;
    }

    /// Blocks until every payload byte left the source.
    pub fn waitCopied(j: *Job) void {
        if (j.copied()) return;
        const t = j.tier;
        const t0 = t.now();
        t.mutex.lockUncancelable(t.io);
        while (!j.copied()) t.cond.waitUncancelable(t.io, &t.mutex);
        t.stats.copy_wait_ns += @intCast(@max(0, t.now() - t0));
        t.mutex.unlock(t.io);
    }

    /// Blocks until the write is on disk or given up (`ok` is then final). Call it before `Tier.shutdown`: shutdown
    /// ends queued writes only as far as its own bookkeeping allocation allows.
    pub fn waitDone(j: *Job) void {
        if (j.done()) return;
        const t = j.tier;
        t.mutex.lockUncancelable(t.io);
        while (!j.done()) t.cond.waitUncancelable(t.io, &t.mutex);
        t.mutex.unlock(t.io);
    }

    pub fn release(j: *Job) void {
        if (j.refs.fetchSub(1, .acq_rel) != 1) return;
        const gpa = j.tier.gpa;
        if (j.ram) |r| r.release();
        gpa.free(j.tokens);
        gpa.free(j.meta);
        gpa.destroy(j);
    }
};

/// A state the tier holds: in memory (`ram`), on disk (`on_disk`, its file renamed in place), or on its way there
/// (`job`, a write not renamed yet). `bytes` is its file's size, which the disk's cap and floor count.
const Entry = struct { generation: u64 = 0, name: Name, tokens: []u32, bytes: u64, used: u64, job: ?*Job = null, source_id: usize = 0, ram: ?*Ram = null, on_disk: bool = false };

/// Whether `e` can be read now: held in memory, or its file in place.
fn readable(e: *const Entry) bool {
    return e.ram != null or (e.on_disk and e.job == null);
}

/// Whether `e` takes disk room: its file in place or on its way.
fn onDisk(e: *const Entry) bool {
    return e.on_disk or e.job != null;
}

/// A stored state the tier can read back for a prompt.
pub const Hit = struct { name: Name, tokens: []const u32 };

pub const Tier = struct {
    gpa: Allocator,
    io: std.Io,
    settings: Settings,
    family: [16]u8,
    identity: [32]u8,
    rank: u32,
    world: u32,
    top: [:0]u8, // dir/<family>-<identity>
    dir: [:0]u8, // top/rank<r>
    /// `dir` checked private: every file of the tier is made, read and deleted in it
    dfd: fs.Dir = undefined,
    mutex: std.Io.Mutex = .init,
    cond: std.Io.Condition = .init,
    entries: std.ArrayList(*Entry) = .empty,
    queue: std.ArrayList(*Job) = .empty,
    busy: ?*Job = null,
    clock: u64 = 0,
    /// states written after this clock are not evicted for room (a batch's own writes never evict each other)
    pin: ?u64 = null,
    thread: ?std.Thread = null,
    reader: ?*Read = null,
    generation: u64 = 0,
    closing: bool = false,
    stats: Stats = .{},
    hook: ?Hook = null,
    /// the allocator held states' bytes come from (`ramAlloc`): the tier's own may be quota-limited to its buffers
    ram_gpa: ?Allocator = null,
    /// bytes the entries hold in memory (a read or a disk write may keep a released one a little longer)
    ram_bytes: u64 = 0,
    /// names a held state's failed disk write dropped on the writer thread: the hook hears of them on the owner's
    gone: std.ArrayList(Name) = .empty,
    /// bytes a copy or a read moves at a time (64 MiB; small in tests)
    chunk: usize = 64 << 20,
    wbuf: []u8 = &.{},
    rbuf: []u8 = &.{},
    /// free bytes on the tier's disk (statvfs; tests replace it)
    free_space: *const fn (dir: [:0]const u8) u64 = fs.diskFree,
    /// File writes, separate from Source.read; tests inject a real ENOSPC after a partial payload.
    write_file: *const fn (fd: fs.File, bytes: []const u8, at: u64) anyerror!void = fs.pwriteAll,
    advise_file: *const fn (fd: fs.File, at: u64, len: u64) void = pageAdvice,

    /// The tier under `settings.dir` for `family` at `identity` and `rank`: made when missing, its partial and foreign
    /// files deleted, other identities past `keep_builds` deleted, its states indexed oldest first.
    pub fn open(gpa: Allocator, io: std.Io, settings: Settings, family: []const u8, identity: [32]u8, rank: u32, world: u32) !*Tier {
        if (family.len == 0 or family.len > 16) return error.BadFamilyName;
        const t = try gpa.create(Tier);
        errdefer gpa.destroy(t);
        var fam: [16]u8 = @splat(0);
        @memcpy(fam[0..family.len], family);
        const top = try std.fmt.allocPrintSentinel(gpa, "{s}/{s}-{x}", .{ settings.dir, family, identity[0..8] }, 0);
        errdefer gpa.free(top);
        const dir = try std.fmt.allocPrintSentinel(gpa, "{s}/rank{d}", .{ top, rank }, 0);
        errdefer gpa.free(dir);
        t.* = .{ .gpa = gpa, .io = io, .settings = settings, .family = fam, .identity = identity, .rank = rank, .world = world, .top = top, .dir = dir };
        try fs.makePath(gpa, dir);
        // the root, identity and rank directories, each opened without following a link and checked private
        const root_path = try gpa.dupeSentinel(u8, settings.dir, 0);
        defer gpa.free(root_path);
        const root_fd = try fs.openPrivate(null, root_path, root_path);
        defer fs.closeDir(root_fd);
        var tn: [40]u8 = undefined;
        const top_fd = try fs.openPrivate(root_fd, try std.fmt.bufPrintSentinel(&tn, "{s}-{x}", .{ family, identity[0..8] }, 0), t.top);
        defer fs.closeDir(top_fd);
        var rn: [24]u8 = undefined;
        t.dfd = try fs.openPrivate(top_fd, try std.fmt.bufPrintSentinel(&rn, "rank{d}", .{rank}, 0), t.dir);
        errdefer fs.closeDir(t.dfd);
        writeStamp(top_fd, @intCast(@max(0, std.Io.Clock.real.now(io).toSeconds())));
        t.stats.pruned += t.pruneBuilds(family);
        try t.scan();
        t.fitCap();
        return t;
    }

    /// Stops the writer (a queued or running write is cut, its partial file deleted) and frees the index.
    pub fn deinit(t: *Tier) void {
        t.shutdown();
        for (t.entries.items) |e| t.freeEntry(e);
        t.entries.deinit(t.gpa);
        t.queue.deinit(t.gpa);
        t.gone.deinit(t.gpa);
        t.gpa.free(t.wbuf);
        t.gpa.free(t.rbuf);
        fs.closeDir(t.dfd);
        t.gpa.free(t.dir);
        t.gpa.free(t.top);
        t.gpa.destroy(t);
    }

    /// The writer stops: queued and running writes are cut. Their sources are read no more once this returns.
    pub fn shutdown(t: *Tier) void {
        t.mutex.lockUncancelable(t.io);
        t.closing = true;
        for (t.queue.items) |j| j.cancel.store(true, .release);
        if (t.busy) |j| j.cancel.store(true, .release);
        t.cond.broadcast(t.io);
        t.mutex.unlock(t.io);
        if (t.reader) |r| {
            r.cancel();
            r.destroy();
            t.reader = null;
        }
        if (t.thread) |th| th.join();
        t.thread = null;
        // writes never started end here, as failed
        t.mutex.lockUncancelable(t.io);
        const left = t.gpa.dupe(*Job, t.queue.items) catch &.{};
        t.queue.clearRetainingCapacity();
        t.mutex.unlock(t.io);
        for (left) |j| t.finish(j, false, null);
        t.gpa.free(left);
    }

    /// Waits until every queued write is on disk or `seconds` passed: true when none is left.
    pub fn drain(t: *Tier, seconds: f64) bool {
        const end = t.now() + @as(i96, @intFromFloat(seconds * 1e9));
        t.mutex.lockUncancelable(t.io);
        defer t.mutex.unlock(t.io);
        while (t.queue.items.len > 0 or t.busy != null) {
            const left = end - t.now();
            if (left <= 0) return false;
            t.cond.waitTimeout(t.io, &t.mutex, .{ .duration = .{ .raw = .fromNanoseconds(@intCast(@min(left, 100 * std.time.ns_per_ms))), .clock = .awake } }) catch {};
        }
        return true;
    }

    fn now(t: *const Tier) i96 {
        return std.Io.Clock.awake.now(t.io).toNanoseconds();
    }

    fn freeEntry(t: *Tier, e: *Entry) void {
        t.releaseRam(e);
        t.gpa.free(e.tokens);
        t.gpa.destroy(e);
    }

    /// The entry's memory copy released (lock held, or before the writer ran).
    fn releaseRam(t: *Tier, e: *Entry) void {
        const r = e.ram orelse return;
        t.ram_bytes -|= r.payload.len;
        e.ram = null;
        r.release();
    }

    fn find(t: *const Tier, name: Name) ?usize {
        for (t.entries.items, 0..) |e, i| if (e.name.eql(name)) return i;
        return null;
    }

    /// Whether `name` is stored or on its way.
    pub fn has(t: *Tier, name: Name) bool {
        t.mutex.lockUncancelable(t.io);
        defer t.mutex.unlock(t.io);
        return t.find(name) != null;
    }

    /// The owner's hook for names the cap or floor removes (null: none). The hook runs on the thread that calls `write`,
    /// never under the tier's lock; one owner at a time, and every `write` while it is set runs on that owner's thread.
    pub fn setHook(t: *Tier, h: ?Hook) void {
        t.mutex.lockUncancelable(t.io);
        defer t.mutex.unlock(t.io);
        t.hook = h;
    }

    /// Clears the hook only when `owner` set it: a later owner's hook stays.
    pub fn releaseHook(t: *Tier, owner: *anyopaque) void {
        t.mutex.lockUncancelable(t.io);
        defer t.mutex.unlock(t.io);
        if (t.hook) |h| if (h.ptr == owner) {
            t.hook = null;
        };
    }

    fn currentHook(t: *Tier) ?Hook {
        t.mutex.lockUncancelable(t.io);
        defer t.mutex.unlock(t.io);
        return t.hook;
    }

    /// Whether `name` is on disk (not on its way) with exactly `tokens`: a copy `read` can return now.
    pub fn stored(t: *Tier, name: Name, tokens: []const u32) bool {
        t.mutex.lockUncancelable(t.io);
        defer t.mutex.unlock(t.io);
        const i = t.find(name) orelse return false;
        const e = t.entries.items[i];
        return readable(e) and std.mem.eql(u32, e.tokens, tokens);
    }

    /// Whether the existing copy is still being written (never waits for that writer).
    pub fn pending(t: *Tier, name: Name) bool {
        t.mutex.lockUncancelable(t.io);
        defer t.mutex.unlock(t.io);
        const i = t.find(name) orelse return false;
        return !readable(t.entries.items[i]);
    }

    /// The current queued or stored copy came from this still-owned job. Scanned copies have no source id.
    pub fn copyOf(t: *Tier, name: Name, job: *const Job) bool {
        if (job.tier != t or !job.name.eql(name)) return false;
        t.mutex.lockUncancelable(t.io);
        defer t.mutex.unlock(t.io);
        const i = t.find(name) orelse return false;
        return t.entries.items[i].source_id == @intFromPtr(job) and !job.cancel.load(.acquire);
    }

    /// `name` becomes the most recently used.
    pub fn touch(t: *Tier, name: Name) void {
        t.mutex.lockUncancelable(t.io);
        defer t.mutex.unlock(t.io);
        const i = t.find(name) orelse return;
        t.clock += 1;
        t.entries.items[i].used = t.clock;
    }

    /// The longest stored state `prompt` extends past `beyond` tokens (at least one prompt token after it), whose
    /// position `usable` accepts; its tokens stay valid until this thread forgets it.
    pub fn best(t: *Tier, prompt: []const u32, beyond: u32, ctx: anytype, comptime usable: fn (@TypeOf(ctx), u32) bool) ?Hit {
        t.mutex.lockUncancelable(t.io);
        defer t.mutex.unlock(t.io);
        var out: ?*Entry = null;
        for (t.entries.items) |e| {
            if (!readable(e) or e.name.at <= beyond or e.name.at >= prompt.len or e.tokens.len > prompt.len) continue;
            if (out != null and e.name.at <= out.?.name.at) continue;
            if (!std.mem.eql(u32, prompt[0..e.tokens.len], e.tokens) or !usable(ctx, e.name.at)) continue;
            out = e;
        }
        if (out != null) t.stats.disk_hits += 1 else t.stats.disk_misses += 1;
        return if (out) |e| .{ .name = e.name, .tokens = e.tokens } else null;
    }

    /// `name` leaves the tier: its file goes (a write on its way is cut and never renamed).
    pub fn forget(t: *Tier, name: Name) void {
        t.mutex.lockUncancelable(t.io);
        defer t.mutex.unlock(t.io);
        const i = t.find(name) orelse return;
        t.drop(i);
        t.stats.forgotten += 1;
    }

    /// Entry i out of the index (lock held): its file unlinked, or its write cut.
    fn drop(t: *Tier, i: usize) void {
        const e = t.entries.orderedRemove(i);
        if (e.job) |j| j.cancel.store(true, .release);
        if (e.on_disk) t.unlink(e.name, e.bytes);
        t.freeEntry(e);
    }

    fn unlink(t: *Tier, name: Name, bytes: u64) void {
        var buf: [32]u8 = undefined;
        const before = t.free_space(t.dir);
        if (fs.unlink(t.dfd, name.file(&buf))) {
            t.stats.unlinked_bytes += bytes;
            t.stats.reclaimed_bytes += @min(bytes, t.free_space(t.dir) -| before);
        }
    }

    /// Bytes the stored and queued states take, and those of the queued ones alone (lock held).
    fn totals(t: *const Tier) struct { all: u64, queued: u64 } {
        var all: u64 = 0;
        var queued: u64 = 0;
        for (t.entries.items) |e| {
            if (!onDisk(e)) continue; // held in memory alone: the memory cap counts it
            all += e.bytes;
            if (e.job != null) queued += e.bytes;
        }
        return .{ .all = all, .queued = queued };
    }

    /// Bytes of writes not on disk yet.
    pub fn queuedBytes(t: *Tier) u64 {
        t.mutex.lockUncancelable(t.io);
        defer t.mutex.unlock(t.io);
        return t.totals().queued;
    }

    /// Queues `name` (its `tokens`, the family's `meta` and `payload_len` bytes from `source`) for the writer. The least
    /// recently used stored states go first until it fits the cap and leaves the free-disk floor; the hook hears of
    /// each. The caller holds a reference to the job until `release`.
    pub fn write(t: *Tier, name: Name, tokens: []const u32, meta: []const u8, payload_len: u64, source: Source) !*Job {
        if (meta.len > meta_most) return error.MetaTooLong;
        const bytes = Header.payloadOffset(tokens.len) + payload_len;
        var gone: std.ArrayList(Name) = .empty;
        defer gone.deinit(t.gpa);
        defer if (t.currentHook()) |h| for (gone.items) |n| h.forgot(h.ptr, n); // read under the lock, called outside it
        const j = try t.gpa.create(Job);
        errdefer t.gpa.destroy(j);
        j.* = .{ .tier = t, .name = name, .tokens = try t.gpa.dupe(u32, tokens), .meta = undefined, .payload_len = payload_len, .source = source };
        errdefer t.gpa.free(j.tokens);
        j.meta = try t.gpa.dupe(u8, meta);
        errdefer t.gpa.free(j.meta);
        const e = try t.gpa.create(Entry);
        errdefer t.gpa.destroy(e);
        e.* = .{ .name = name, .tokens = try t.gpa.dupe(u32, tokens), .bytes = bytes, .used = 0, .job = j, .source_id = @intFromPtr(j) };
        errdefer t.gpa.free(e.tokens);
        t.takeGone(&gone);
        t.mutex.lockUncancelable(t.io);
        defer t.mutex.unlock(t.io);
        if (t.closing) return error.TierClosed;
        if (t.find(name) != null) {
            t.stats.skipped += 1;
            return error.AlreadyStored;
        }
        try t.makeRoom(bytes, &gone);
        try t.entries.ensureUnusedCapacity(t.gpa, 1);
        try t.queue.ensureUnusedCapacity(t.gpa, 1);
        if (t.thread == null) t.thread = try std.Thread.spawn(.{}, run, .{t});
        t.clock += 1;
        e.used = t.clock;
        t.generation += 1;
        e.generation = t.generation;
        t.entries.appendAssumeCapacity(e);
        t.queue.appendAssumeCapacity(j);
        t.cond.broadcast(t.io);
        return j;
    }

    /// Disk room for a file of `bytes` (lock held): the least recently used stored files go until it fits the cap and
    /// leaves the free-disk floor, their names in `gone`; refused when it would not fit with every one gone.
    fn makeRoom(t: *Tier, bytes: u64, gone: *std.ArrayList(Name)) !void {
        if (bytes > t.settings.cap) {
            t.stats.full += 1;
            return error.LargerThanCap;
        }
        var sums = t.totals();
        var free = t.free_space(t.dir);
        // nothing goes for a write that would not fit with every state that may go gone
        var reclaim: u64 = 0;
        for (t.entries.items) |x| if (t.evictable(x)) {
            reclaim += x.bytes;
        };
        if (sums.all - reclaim + bytes > t.settings.cap) {
            t.stats.full += 1;
            return error.TierFull;
        }
        if (free +| reclaim < t.settings.min_free +| sums.queued +| bytes) {
            t.stats.no_space += 1;
            return error.NoSpace;
        }
        while (sums.all + bytes > t.settings.cap or free < t.settings.min_free +| sums.queued +| bytes) {
            const i = t.oldestStored() orelse {
                t.stats.no_space += 1;
                return error.NoSpace;
            };
            const victim = t.entries.items[i];
            sums.all -= victim.bytes;
            try gone.append(t.gpa, victim.name);
            t.drop(i);
            t.stats.disk_evictions += 1;
            t.stats.forgotten += 1;
            // the disk asked again: a file still open (a read of it) or not unlinked gives no bytes back
            free = t.free_space(t.dir);
        }
    }

    /// Names a held state's failed disk write dropped (writer thread) moved to `out` for the owner's hook.
    fn takeGone(t: *Tier, out: *std.ArrayList(Name)) void {
        t.mutex.lockUncancelable(t.io);
        defer t.mutex.unlock(t.io);
        out.appendSlice(t.gpa, t.gone.items) catch return;
        t.gone.clearRetainingCapacity();
    }

    /// Bytes for a state a family hands over whole (`adopt`), from the allocator held states use.
    pub fn ramAlloc(t: *Tier, n: usize) ![]u8 {
        return (t.ram_gpa orelse t.gpa).alloc(u8, n);
    }

    /// Whether a state of `payload_len` bytes can be held in memory (`adopt`): the memory level is on and holds it.
    pub fn holds(t: *const Tier, payload_len: u64) bool {
        return t.settings.ram_cap > 0 and payload_len <= t.settings.ram_cap;
    }

    /// `name` (its `tokens`, the family's `meta`) held in memory from `payload`, which the tier now owns (from
    /// `ramAlloc`; freed on any error). The least recently used held states go on to disk until memory fits its
    /// cap, or leave when the disk has no room; the hook hears of those, here on the owner's thread.
    pub fn adopt(t: *Tier, name: Name, tokens: []const u32, meta: []const u8, payload: []u8) !void {
        const ram_gpa = t.ram_gpa orelse t.gpa;
        var owned = true;
        defer if (owned) ram_gpa.free(payload);
        if (meta.len > meta_most) return error.MetaTooLong;
        if (!t.holds(payload.len)) return error.LargerThanRam;
        var gone: std.ArrayList(Name) = .empty;
        defer gone.deinit(t.gpa);
        defer if (t.currentHook()) |h| for (gone.items) |n| h.forgot(h.ptr, n);
        t.takeGone(&gone);
        const r = try t.gpa.create(Ram);
        errdefer t.gpa.destroy(r);
        r.* = .{ .gpa = ram_gpa, .payload = payload, .meta = try ram_gpa.dupe(u8, meta), .hash = std.hash.XxHash3.hash(0, payload), .saved = @intCast(@max(0, std.Io.Clock.real.now(t.io).toSeconds())) };
        errdefer ram_gpa.free(r.meta);
        const e = try t.gpa.create(Entry);
        errdefer t.gpa.destroy(e);
        e.* = .{ .name = name, .tokens = try t.gpa.dupe(u32, tokens), .bytes = Header.payloadOffset(tokens.len) + payload.len, .used = 0 };
        errdefer t.gpa.free(e.tokens);
        t.mutex.lockUncancelable(t.io);
        defer t.mutex.unlock(t.io);
        if (t.closing) return error.TierClosed;
        if (t.find(name) != null) {
            t.stats.skipped += 1;
            return error.AlreadyStored;
        }
        try t.entries.ensureUnusedCapacity(t.gpa, 1);
        owned = false;
        e.ram = r;
        t.ram_bytes += payload.len;
        t.clock += 1;
        e.used = t.clock;
        t.generation += 1;
        e.generation = t.generation;
        t.entries.appendAssumeCapacity(e);
        t.stats.ram_adopted += 1;
        // written behind: its file goes to disk now while it stays held (no room: it stays in memory alone)
        _ = t.demote(t.entries.items.len - 1, &gone, true);
        t.fitRam(e, &gone);
    }

    /// Held bytes that stay held (lock held): what the memory cap is kept to. A state on its way to disk for room
    /// leaves memory when its file is in place, so it is not counted.
    fn ramCommitted(t: *const Tier) u64 {
        var n: u64 = 0;
        for (t.entries.items) |e| if (e.ram) |r| {
            if (e.job) |j| if (!j.keep_ram) continue;
            n += r.payload.len;
        };
        return n;
    }

    /// Memory back under its cap (lock held): the least recently used held states other than `keep` leave memory
    /// when their file is in place, else go on to disk for room, else leave altogether (their names in `gone`) when
    /// the disk has no room. One written behind and still on its way stays until its file is in place (`trimRam`).
    fn fitRam(t: *Tier, keep: ?*const Entry, gone: *std.ArrayList(Name)) void {
        while (t.ramCommitted() > t.settings.ram_cap) {
            var out: ?usize = null;
            for (t.entries.items, 0..) |e, i| {
                if (e.ram == null or e.job != null or e == keep) continue;
                if (out == null or e.used < t.entries.items[out.?].used) out = i;
            }
            const i = out orelse return;
            if (!t.demote(i, gone, false)) {
                const name = t.entries.items[i].name;
                gone.append(t.gpa, name) catch {};
                t.drop(i);
                t.stats.ram_dropped += 1;
                t.stats.forgotten += 1;
            }
        }
    }

    /// Memory back under its cap as far as files in place allow (lock held, any thread): the least recently used
    /// held states whose file is in place leave memory. Nothing leaves the tier, so no owner hears of it.
    fn trimRam(t: *Tier) void {
        while (t.ramCommitted() > t.settings.ram_cap) {
            var out: ?usize = null;
            for (t.entries.items, 0..) |e, i| {
                if (e.ram == null or !e.on_disk or e.job != null) continue;
                if (out == null or e.used < t.entries.items[out.?].used) out = i;
            }
            const i = out orelse return;
            t.releaseRam(t.entries.items[i]);
            t.stats.ram_released += 1;
        }
    }

    /// Held entry i queued for disk from its memory (lock held), kept held once its file is in place (`behind`) or
    /// released then: false when the disk has no room or no write starts. With its file in place already, its
    /// memory simply goes (unless written behind, which has nothing left to do).
    fn demote(t: *Tier, i: usize, gone: *std.ArrayList(Name), behind: bool) bool {
        const e = t.entries.items[i];
        const r = e.ram.?;
        if (e.on_disk) { // its file is in place already: the memory alone goes
            if (!behind) {
                t.releaseRam(e);
                t.stats.ram_released += 1;
            }
            return true;
        }
        t.makeRoom(e.bytes, gone) catch return false;
        // makeRoom may have taken entries before i: find it again
        const k = t.find(e.name) orelse return false;
        std.debug.assert(t.entries.items[k] == e);
        const j = t.gpa.create(Job) catch return false;
        j.* = .{ .tier = t, .name = e.name, .tokens = t.gpa.dupe(u32, e.tokens) catch {
            t.gpa.destroy(j);
            return false;
        }, .meta = t.gpa.dupe(u8, r.meta) catch {
            t.gpa.free(j.tokens);
            t.gpa.destroy(j);
            return false;
        }, .payload_len = r.payload.len, .source = .{ .ptr = r, .read = Ram.read }, .refs = .init(1), .ram = r.retain(), .keep_ram = behind };
        t.queue.ensureUnusedCapacity(t.gpa, 1) catch {
            j.release();
            return false;
        };
        if (t.thread == null) t.thread = std.Thread.spawn(.{}, run, .{t}) catch {
            j.release();
            return false;
        };
        e.job = j;
        e.source_id = @intFromPtr(j);
        t.queue.appendAssumeCapacity(j);
        t.cond.broadcast(t.io);
        t.stats.ram_demoted += 1;
        return true;
    }

    /// `payload`, read whole and checked from `r`'s file, held again by its entry when the entry is still there, its
    /// file in place and nothing holds it yet; memory is then trimmed to its cap (files in place go first). True when
    /// the entry took `payload`.
    fn promote(t: *Tier, r: *const Reading, payload: []u8) bool {
        const ram_gpa = t.ram_gpa orelse t.gpa;
        const held = t.gpa.create(Ram) catch return false;
        const meta = ram_gpa.dupe(u8, r.meta()) catch {
            t.gpa.destroy(held);
            return false;
        };
        held.* = .{ .gpa = ram_gpa, .payload = payload, .meta = meta, .hash = r.header.payload_hash, .saved = r.header.saved };
        t.mutex.lockUncancelable(t.io);
        defer t.mutex.unlock(t.io);
        const i = t.find(r.name) orelse {
            ram_gpa.free(meta);
            t.gpa.destroy(held);
            return false;
        };
        const e = t.entries.items[i];
        if (e.generation != r.generation or e.ram != null or !e.on_disk or e.job != null) {
            ram_gpa.free(meta);
            t.gpa.destroy(held);
            return false;
        }
        e.ram = held;
        t.ram_bytes += payload.len;
        t.stats.ram_promoted += 1;
        t.clock += 1;
        e.used = t.clock; // the most recently used: others leave memory before it
        t.trimRam();
        return true;
    }

    /// A clean stop: every held state not on disk queued for it, most recently used first (the writers then drain).
    pub fn flushRam(t: *Tier) void {
        var gone: std.ArrayList(Name) = .empty;
        defer gone.deinit(t.gpa);
        defer if (t.currentHook()) |h| for (gone.items) |n| h.forgot(h.ptr, n);
        t.mutex.lockUncancelable(t.io);
        defer t.mutex.unlock(t.io);
        while (true) {
            var out: ?usize = null;
            for (t.entries.items, 0..) |e, i| {
                if (e.ram == null or e.job != null or e.on_disk) continue;
                if (out == null or e.used > t.entries.items[out.?].used) out = i;
            }
            const i = out orelse return;
            if (!t.demote(i, &gone, false)) return; // the disk is full: the rest stay in memory until the tier closes
        }
    }

    /// Whether `e` may go for room: stored (not on its way) and not one of a pinned batch.
    fn evictable(t: *const Tier, e: *const Entry) bool {
        return e.on_disk and e.job == null and (t.pin == null or e.used <= t.pin.?);
    }

    /// The least recently used state that may go for room, lock held.
    fn oldestStored(t: *const Tier) ?usize {
        var out: ?usize = null;
        for (t.entries.items, 0..) |e, i| {
            if (!t.evictable(e)) continue;
            if (out == null or e.used < t.entries.items[out.?].used) out = i;
        }
        return out;
    }

    /// `j` goes next: its source's memory is wanted back (a state left memory while its early write waited).
    pub fn hurry(t: *Tier, j: *Job) void {
        t.mutex.lockUncancelable(t.io);
        defer t.mutex.unlock(t.io);
        const i = std.mem.indexOfScalar(*Job, t.queue.items, j) orelse return;
        _ = t.queue.orderedRemove(i);
        t.queue.insertAssumeCapacity(0, j);
    }

    /// The writes from now until `unpin` form a batch: none of them is evicted to make room for another.
    pub fn pinBatch(t: *Tier) void {
        t.mutex.lockUncancelable(t.io);
        defer t.mutex.unlock(t.io);
        t.pin = t.clock;
    }

    pub fn unpin(t: *Tier) void {
        t.mutex.lockUncancelable(t.io);
        defer t.mutex.unlock(t.io);
        t.pin = null;
    }

    fn run(t: *Tier) void {
        while (true) {
            t.mutex.lockUncancelable(t.io);
            while (t.queue.items.len == 0 and !t.closing) t.cond.waitUncancelable(t.io, &t.mutex);
            if (t.closing) {
                t.mutex.unlock(t.io);
                return;
            }
            const j = t.queue.orderedRemove(0);
            t.busy = j;
            t.mutex.unlock(t.io);
            const t0 = t.now();
            var name: [64]u8 = undefined;
            const partial = t.store(j, &name) catch |err| blk: {
                if (err != error.Cancelled and !builtin.is_test) std.log.warn("spill rank {d}: writing {d} tokens failed ({s})", .{ t.rank, j.name.at, @errorName(err) });
                break :blk null;
            };
            t.mutex.lockUncancelable(t.io);
            t.stats.write_count += 1;
            t.stats.write_ns += @intCast(@max(0, t.now() - t0));
            t.busy = null;
            t.mutex.unlock(t.io);
            t.finish(j, partial != null, partial);
        }
    }

    /// The job's file written to a new partial file in the directory (its name returned) and fsynced; the source's
    /// last bytes read before the last write. The name is random and the file is made only when absent, so nothing
    /// already there (a link included) is written through.
    fn store(t: *Tier, j: *Job, name: *[64]u8) ![:0]const u8 {
        if (j.cancel.load(.acquire)) return error.Cancelled;
        if (t.wbuf.len == 0) t.wbuf = try t.gpa.alloc(u8, t.chunk);
        var nonce: [8]u8 = undefined;
        t.io.random(&nonce);
        var nb: [32]u8 = undefined;
        const partial = try std.fmt.bufPrintSentinel(name, "{s}.{x}.partial", .{ j.name.file(&nb), &nonce }, 0);
        const fd = fs.createExcl(t.dfd, partial) orelse return error.SpillOpen;
        var ok = false;
        defer {
            fs.close(fd);
            if (!ok) _ = fs.unlink(t.dfd, partial);
        }
        const tokens = std.mem.sliceAsBytes(j.tokens);
        try t.write_file(fd, tokens, header_bytes);
        const at = Header.payloadOffset(j.tokens.len);
        var hash = std.hash.XxHash3.init(0);
        var off: u64 = 0;
        while (off < j.payload_len) {
            if (j.cancel.load(.acquire)) return error.Cancelled;
            const n: usize = @intCast(@min(t.chunk, j.payload_len - off));
            j.source.read(j.source.ptr, off, t.wbuf[0..n]) catch |err| {
                t.markCopied(j);
                return err;
            };
            if (off + n == j.payload_len) t.markCopied(j);
            hash.update(t.wbuf[0..n]);
            try t.write_file(fd, t.wbuf[0..n], at + off);
            off += n;
        }
        t.markCopied(j);
        var head: [header_bytes]u8 = undefined;
        const h: Header = .{
            .identity = t.identity,
            .family = t.family,
            .rank = t.rank,
            .world = t.world,
            .name = j.name,
            .tokens = @intCast(j.tokens.len),
            .payload_len = j.payload_len,
            .tokens_hash = std.hash.XxHash3.hash(0, tokens),
            .payload_hash = hash.final(),
            .saved = @intCast(@max(0, std.Io.Clock.real.now(t.io).toSeconds())),
            .meta_len = @intCast(j.meta.len),
        };
        h.encode(j.meta, &head);
        try t.write_file(fd, &head, 0);
        if (!fs.truncate(fd, h.fileBytes())) return error.SpillWrite;
        if (!fs.sync(fd)) return error.SpillSync;
        t.advise_file(fd, 0, h.fileBytes());
        if (j.cancel.load(.acquire)) return error.Cancelled;
        ok = true;
        return partial;
    }

    fn markCopied(t: *Tier, j: *Job) void {
        if (j.copied()) return;
        t.mutex.lockUncancelable(t.io);
        j.state.store(Job.copied_state, .release);
        t.cond.broadcast(t.io);
        t.mutex.unlock(t.io);
    }

    /// A job ends: its partial file renamed when it is still the index's, else deleted.
    fn finish(t: *Tier, j: *Job, wrote: bool, partial: ?[:0]const u8) void {
        var renamed = false;
        t.mutex.lockUncancelable(t.io);
        const i = for (t.entries.items, 0..) |e, k| {
            if (e.job == j) break k;
        } else null;
        if (wrote and i != null and !j.cancel.load(.acquire)) {
            var nb: [32]u8 = undefined;
            if (fs.rename(t.dfd, partial.?, j.name.file(&nb))) {
                renamed = true;
                const e = t.entries.items[i.?];
                e.job = null;
                e.on_disk = true;
                if (j.ram != null and !j.keep_ram) t.releaseRam(e); // went on to disk for memory room: the file holds it now
                t.stats.written += 1;
                t.stats.written_bytes += e.bytes;
                if (j.ram != null) t.trimRam(); // a state written behind may now leave memory for room
            }
        }
        if (!renamed) {
            if (partial) |p| _ = fs.unlink(t.dfd, p);
            if (i) |k| {
                const e = t.entries.orderedRemove(k);
                if (j.keep_ram and e.ram != null and !j.cancel.load(.acquire)) {
                    // written behind and failed: it stays held, with no file (room for it leaves the tier instead)
                    e.job = null;
                    t.entries.insertAssumeCapacity(k, e);
                } else {
                    if (j.ram != null and !j.cancel.load(.acquire)) { // a held state's write for room failed: it leaves, the owner hears of it
                        t.gone.append(t.gpa, e.name) catch {};
                        t.stats.ram_dropped += 1;
                    }
                    t.freeEntry(e);
                }
            }
            if (!j.cancel.load(.acquire)) t.stats.write_failed += 1;
        }
        j.ok = renamed;
        j.state.store(Job.done_state, .release);
        t.cond.broadcast(t.io);
        t.mutex.unlock(t.io);
        if (renamed) fs.syncDir(t.dfd);
        j.release();
    }

    /// Owner-thread submission. One bounded reader remains owned until release and worker completion.
    pub fn readAsync(t: *Tier, name: Name) !*Read {
        if (t.reader) |r| {
            if (!r.reapable()) return error.ReadBusy;
            r.destroy();
            t.reader = null;
        }
        if (t.closing) return error.TierClosed;
        const r = try Read.start(t, name);
        t.reader = r;
        return r;
    }

    pub fn readGeneration(t: *Tier, name: Name) !u64 {
        t.mutex.lockUncancelable(t.io);
        defer t.mutex.unlock(t.io);
        const i = t.find(name) orelse return error.NotStored;
        if (!readable(t.entries.items[i])) return error.NotStored;
        return t.entries.items[i].generation;
    }

    pub fn isRam(t: *Tier, name: Name) bool {
        t.mutex.lockUncancelable(t.io);
        defer t.mutex.unlock(t.io);
        const i = t.find(name) orelse return false;
        return t.entries.items[i].ram != null;
    }

    pub fn forgetGeneration(t: *Tier, name: Name, generation: u64) void {
        t.mutex.lockUncancelable(t.io);
        defer t.mutex.unlock(t.io);
        const i = t.find(name) orelse return;
        if (t.entries.items[i].generation != generation) return;
        t.drop(i);
        t.stats.forgotten += 1;
    }

    /// Reads the header and tokens now, then the payload through Reading.pump.
    pub fn read(t: *Tier, name: Name) !Reading {
        return t.readMode(name, true, null);
    }

    pub fn readDeferred(t: *Tier, name: Name, generation: u64) !Reading {
        return t.readMode(name, false, generation);
    }

    pub fn readFailed(t: *Tier, err: anyerror, started: i96) void {
        if (err == error.ReadCancelled or err == error.NotStored) return;
        t.mutex.lockUncancelable(t.io);
        defer t.mutex.unlock(t.io);
        t.stats.load_failed += 1;
        t.stats.load_count += 1;
        t.stats.load_ns += @intCast(@max(0, t.now() - started));
        if (err == error.PayloadHash) t.stats.hash_failed += 1;
    }

    fn readMode(t: *Tier, name: Name, forget_on_failure: bool, expected: ?u64) !Reading {
        t.mutex.lockUncancelable(t.io);
        const i = t.find(name) orelse {
            t.mutex.unlock(t.io);
            return error.NotStored;
        };
        const e = t.entries.items[i];
        const generation = e.generation;
        if (expected) |want| if (want != generation) {
            t.mutex.unlock(t.io);
            return error.NotStored;
        };
        if (e.ram) |held| {
            var r: Reading = .{ .tier = t, .name = name, .t0 = t.now(), .ram = held.retain(), .generation = generation, .forget_on_failure = forget_on_failure };
            const tokens = t.gpa.dupe(u32, e.tokens) catch |err| {
                t.mutex.unlock(t.io);
                r.close();
                return err;
            };
            t.mutex.unlock(t.io);
            r.tokens = tokens;
            r.header = .{ .identity = t.identity, .family = t.family, .rank = t.rank, .world = t.world, .name = name, .tokens = @intCast(tokens.len), .payload_len = held.payload.len, .tokens_hash = 0, .payload_hash = held.hash, .saved = held.saved, .meta_len = @intCast(held.meta.len) };
            @memcpy(r.head[136..][0..held.meta.len], held.meta);
            return r;
        }
        if (!readable(e)) {
            t.mutex.unlock(t.io);
            return error.NotStored;
        }
        t.mutex.unlock(t.io);
        var r: Reading = .{ .tier = t, .name = name, .t0 = t.now(), .generation = generation, .forget_on_failure = forget_on_failure };
        r.begin() catch |err| {
            r.fail();
            r.close();
            return err;
        };
        return r;
    }

    /// Bytes of every stored file, and the stats (/health).
    pub fn info(t: *Tier) Info {
        t.mutex.lockUncancelable(t.io);
        defer t.mutex.unlock(t.io);
        const sums = t.totals();
        var pending_count: usize = 0;
        var ram_entries: usize = 0;
        for (t.entries.items) |e| {
            pending_count += @intFromBool(!readable(e));
            ram_entries += @intFromBool(e.ram != null);
        }
        return .{ .stats = t.stats, .entries = t.entries.items.len - pending_count, .pending = pending_count, .stored_bytes = sums.all - sums.queued, .queued_bytes = sums.queued, .disk_free = t.free_space(t.dir), .cap = t.settings.cap, .ram_entries = ram_entries, .ram_bytes = t.ram_bytes, .ram_cap = t.settings.ram_cap };
    }

    /// Counts a state the store chose not to write.
    pub fn unwritten(t: *Tier, comptime why: enum { short, superseded, incomplete, deferred, replacement_abandoned }) void {
        t.mutex.lockUncancelable(t.io);
        defer t.mutex.unlock(t.io);
        switch (why) {
            .short => t.stats.unwritten_short += 1,
            .superseded => t.stats.unwritten_superseded += 1,
            .incomplete => t.stats.unwritten_incomplete += 1,
            .deferred => t.stats.deferred += 1,
            .replacement_abandoned => t.stats.replacement_abandoned += 1,
        }
    }

    // -- the directory ---------------------------------------------------------------------------------------------

    /// This rank's files: partial ones deleted, the rest indexed when their header and tokens check (else deleted),
    /// oldest written first.
    fn scan(t: *Tier) !void {
        var d = fs.Lister.open(t.dir) orelse return error.SpillDir;
        defer d.close();
        const Found = struct { e: *Entry, saved: u64 };
        var found: std.ArrayList(Found) = .empty;
        defer found.deinit(t.gpa);
        errdefer for (found.items) |f| t.freeEntry(f.e);
        while (d.next()) |ent| {
            var fb: [1040]u8 = undefined;
            const file = std.fmt.bufPrintSentinel(&fb, "{s}", .{ent.name}, 0) catch continue;
            if (std.mem.endsWith(u8, file, ".partial")) {
                _ = fs.unlink(t.dfd, file);
                continue;
            }
            const name = Name.parse(file) orelse continue;
            const got = t.index(file, name) catch {
                _ = fs.unlink(t.dfd, file);
                t.stats.pruned += 1;
                continue;
            };
            try found.append(t.gpa, .{ .e = got.e, .saved = got.saved });
        }
        std.mem.sort(Found, found.items, {}, struct {
            fn older(_: void, a: Found, b: Found) bool {
                return a.saved < b.saved or (a.saved == b.saved and a.e.name.key < b.e.name.key);
            }
        }.older);
        try t.entries.ensureUnusedCapacity(t.gpa, found.items.len);
        for (found.items) |f| {
            t.clock += 1;
            f.e.used = t.clock;
            t.entries.appendAssumeCapacity(f.e);
        }
        found.clearRetainingCapacity();
    }

    /// The entry of the directory's file `file` named `name`, when its header, size and tokens check.
    fn index(t: *Tier, file: [:0]const u8, name: Name) !struct { e: *Entry, saved: u64 } {
        const fd = fs.openRead(t.dfd, file) orelse return error.SpillOpen;
        defer fs.close(fd);
        var head: [header_bytes]u8 = undefined;
        if (!fs.preadAll(fd, &head, 0)) return error.ShortFile;
        const h = try Header.decode(&head, t.identity, t.rank, t.world);
        if (!h.name.eql(name)) return error.OtherName;
        if (fs.size(fd) != h.fileBytes()) return error.ShortFile;
        const tokens = try t.gpa.alloc(u32, h.tokens);
        errdefer t.gpa.free(tokens);
        if (!fs.preadAll(fd, std.mem.sliceAsBytes(tokens), header_bytes)) return error.ShortFile;
        if (std.hash.XxHash3.hash(0, std.mem.sliceAsBytes(tokens)) != h.tokens_hash) return error.TokensHash;
        if (!Name.of(tokens, h.name.at).eql(name) or h.name.at > tokens.len) return error.OtherName;
        const e = try t.gpa.create(Entry);
        t.generation += 1;
        e.* = .{ .generation = t.generation, .name = name, .tokens = tokens, .bytes = h.fileBytes(), .used = 0, .on_disk = true };
        return .{ .e = e, .saved = h.saved };
    }

    /// The oldest stored states go until the index fits the cap (a smaller cap than the files' last start).
    fn fitCap(t: *Tier) void {
        while (t.totals().all > t.settings.cap) {
            const i = t.oldestStored() orelse return;
            t.drop(i);
            t.stats.disk_evictions += 1;
            t.stats.pruned += 1;
        }
    }

    /// Other identities' directories of this family: the newest `keep_builds` stay, the others lose this rank's files
    /// (and go once every rank's are gone). Returns the files deleted.
    fn pruneBuilds(t: *Tier, family: []const u8) u64 {
        const Build = struct { name: [80]u8, len: usize, stamp: u64 };
        var builds: std.ArrayList(Build) = .empty;
        defer builds.deinit(t.gpa);
        var rb: [1200]u8 = undefined;
        const root = std.fmt.bufPrintSentinel(&rb, "{s}", .{t.settings.dir}, 0) catch return 0;
        var d = fs.Lister.open(root) orelse return 0;
        while (d.next()) |ent| {
            const n = ent.name;
            if (n.len != family.len + 1 + 16 or !std.mem.startsWith(u8, n, family) or n[family.len] != '-') continue;
            if (std.mem.endsWith(u8, t.top, n)) continue;
            var b: Build = .{ .name = undefined, .len = n.len, .stamp = 0 };
            @memcpy(b.name[0..n.len], n);
            var sp: [1300]u8 = undefined;
            b.stamp = readStamp(std.fmt.bufPrintSentinel(&sp, "{s}/{s}", .{ t.settings.dir, n }, 0) catch continue);
            builds.append(t.gpa, b) catch break;
        }
        d.close();
        std.mem.sort(Build, builds.items, {}, struct {
            fn newer(_: void, a: Build, b: Build) bool {
                return a.stamp > b.stamp;
            }
        }.newer);
        var gone: u64 = 0;
        if (builds.items.len <= t.settings.keep_builds) return 0;
        for (builds.items[t.settings.keep_builds..]) |b| {
            var tp: [1300]u8 = undefined;
            const other = std.fmt.bufPrintSentinel(&tp, "{s}/{s}", .{ t.settings.dir, b.name[0..b.len] }, 0) catch continue;
            var rp: [1400]u8 = undefined;
            const mine = std.fmt.bufPrintSentinel(&rp, "{s}/rank{d}", .{ other, t.rank }, 0) catch continue;
            gone += removeFiles(mine);
            fs.rmdir(mine);
            removeIfOnlyStamp(other);
        }
        return gone;
    }
};

/// A stored state being read back: its header's meta, its tokens, then its payload through `pump`.
pub const Reading = struct {
    tier: *Tier,
    name: Name,
    fd: ?fs.File = null,
    head: [header_bytes]u8 = undefined,
    header: Header = undefined,
    tokens: []u32 = &.{},
    t0: i96,
    timed: bool = false,
    generation: u64 = 0,
    forget_on_failure: bool = true,
    ram: ?*Ram = null, // read from memory (a reference held until close), not from its file

    fn begin(r: *Reading) !void {
        const t = r.tier;
        var nb: [32]u8 = undefined;
        const fd = fs.openRead(t.dfd, r.name.file(&nb)) orelse return error.SpillOpen;
        r.fd = fd;
        if (!fs.preadAll(fd, &r.head, 0)) return error.ShortFile;
        t.advise_file(fd, 0, r.head.len);
        r.header = try Header.decode(&r.head, t.identity, t.rank, t.world);
        if (!r.header.name.eql(r.name)) return error.OtherName;
        if (fs.size(fd) != r.header.fileBytes()) return error.ShortFile;
        r.tokens = try t.gpa.alloc(u32, r.header.tokens);
        if (!fs.preadAll(fd, std.mem.sliceAsBytes(r.tokens), header_bytes)) return error.ShortFile;
        t.advise_file(fd, header_bytes, std.mem.sliceAsBytes(r.tokens).len);
        if (std.hash.XxHash3.hash(0, std.mem.sliceAsBytes(r.tokens)) != r.header.tokens_hash) return error.TokensHash;
        if (!Name.of(r.tokens, r.name.at).eql(r.name)) return error.OtherName;
    }

    pub fn meta(r: *const Reading) []const u8 {
        return Header.meta(&r.head, r.header);
    }

    pub fn payloadLen(r: *const Reading) u64 {
        return r.header.payload_len;
    }

    /// The payload into `sink`, its hash checked once all of it is there: a mismatch is an error (the caller frees
    /// what the sink filled; the file goes).
    pub fn pump(r: *Reading, sink: Sink) !void {
        const t = r.tier;
        if (r.ram == null and t.rbuf.len == 0) t.rbuf = t.gpa.alloc(u8, t.chunk) catch |err| return r.failWith(err);
        return r.pumpBuffered(sink, t.rbuf);
    }

    pub fn pumpBuffered(r: *Reading, sink: Sink, buffer: []u8) !void {
        const t = r.tier;
        // Resource failures follow the same conservative discard/cold-fallback policy as
        // other terminal read errors, even when the stored file itself is valid.
        if (r.ram) |held| {
            var off: usize = 0;
            while (off < held.payload.len) {
                const n = @min(t.chunk, held.payload.len - off);
                sink.write(sink.ptr, off, held.payload[off..][0..n]) catch |err| return r.failWith(err);
                off += n;
            }
            t.touch(r.name);
            t.mutex.lockUncancelable(t.io);
            t.stats.loaded += 1;
            t.stats.ram_loaded += 1;
            t.stats.loaded_bytes += held.payload.len;
            t.stats.loaded_tokens += r.name.at;
            r.recordTime();
            t.mutex.unlock(t.io);
            return;
        }
        if (buffer.len == 0) return r.failWith(error.EmptyReadBuffer);
        // held again as it is read when the memory level holds it: the state was wanted, its next read is memory's
        const ram_gpa = t.ram_gpa orelse t.gpa;
        var copy: ?[]u8 = if (t.holds(r.header.payload_len)) ram_gpa.alloc(u8, @intCast(r.header.payload_len)) catch null else null;
        defer if (copy) |c| ram_gpa.free(c);
        var hash = std.hash.XxHash3.init(0);
        const at = Header.payloadOffset(r.header.tokens);
        var off: u64 = 0;
        while (off < r.header.payload_len) {
            const n: usize = @intCast(@min(buffer.len, r.header.payload_len - off));
            if (!fs.preadAll(r.fd.?, buffer[0..n], at + off)) return r.failWith(error.ShortFile);
            t.advise_file(r.fd.?, at + off, n);
            hash.update(buffer[0..n]);
            if (copy) |c| @memcpy(c[@intCast(off)..][0..n], buffer[0..n]);
            sink.write(sink.ptr, off, buffer[0..n]) catch |err| return r.failWith(err);
            off += n;
        }
        if (hash.final() != r.header.payload_hash) {
            if (r.forget_on_failure) {
                t.mutex.lockUncancelable(t.io);
                t.stats.hash_failed += 1;
                t.mutex.unlock(t.io);
            }
            return r.failWith(error.PayloadHash);
        }
        if (copy) |c| if (t.promote(r, c)) {
            copy = null; // the entry owns it
        };
        t.touch(r.name);
        t.mutex.lockUncancelable(t.io);
        t.stats.loaded += 1;
        t.stats.loaded_bytes += r.header.fileBytes();
        t.stats.loaded_tokens += r.header.name.at;
        r.recordTime();
        t.mutex.unlock(t.io);
    }

    fn failWith(r: *Reading, err: anyerror) anyerror {
        if (err == error.ReadCancelled) return err;
        r.fail();
        return err;
    }

    /// A read that failed: its file goes (it is not trusted again).
    fn fail(r: *Reading) void {
        if (!r.forget_on_failure) return;
        const t = r.tier;
        t.mutex.lockUncancelable(t.io);
        t.stats.load_failed += 1;
        r.recordTime();
        t.mutex.unlock(t.io);
        if (r.forget_on_failure) t.forgetGeneration(r.name, r.generation);
    }

    // Tier mutex held. Abandoned reads have no terminal sample.
    fn recordTime(r: *Reading) void {
        if (r.timed) return;
        r.timed = true;
        r.tier.stats.load_count += 1;
        r.tier.stats.load_ns += @intCast(@max(0, r.tier.now() - r.t0));
    }

    pub fn close(r: *Reading) void {
        if (r.fd) |fd| fs.close(fd);
        r.fd = null;
        if (r.ram) |held| held.release();
        r.ram = null;
        r.tier.gpa.free(r.tokens);
        r.tokens = &.{};
    }
};

// -- files ----------------------------------------------------------------------------------------------------------

/// Whether an existing `dir` passes the tier's own private-directory check (no link; on POSIX this user's or root's,
/// not group or world writable; on Windows changeable by no one but its owner, SYSTEM and Administrators), without
/// creating, opening a tier in, or pruning anything.
pub fn checkRoot(dir: [:0]const u8) !void {
    if (!fs.absolute(dir)) return error.SpillDirNotAbsolute;
    fs.closeDir(try fs.openPrivate(null, dir, dir));
}

/// The start stamp in the identity's directory `top`.
fn writeStamp(top: fs.Dir, seconds: u64) void {
    const fd = fs.createTrunc(top, "opened") orelse return;
    defer fs.close(fd);
    var w: [8]u8 = undefined;
    std.mem.writeInt(u64, &w, seconds, .little);
    fs.pwriteAll(fd, &w, 0) catch {};
}

fn readStamp(top: [:0]const u8) u64 {
    var p: [1400]u8 = undefined;
    const path = std.fmt.bufPrintSentinel(&p, "{s}/opened", .{top}, 0) catch return 0;
    const fd = fs.openPath(path) orelse return 0;
    defer fs.close(fd);
    var w: [8]u8 = undefined;
    return if (fs.preadAll(fd, &w, 0)) std.mem.readInt(u64, &w, .little) else 0;
}

/// Regular files directly in `dir` unlinked; how many.
fn removeFiles(dir: [:0]const u8) u64 {
    var d = fs.Lister.open(dir) orelse return 0;
    defer d.close();
    var n: u64 = 0;
    while (d.next()) |ent| {
        if (!ent.file) continue;
        var p: [1600]u8 = undefined;
        const f = std.fmt.bufPrintSentinel(&p, "{s}/{s}", .{ dir, ent.name }, 0) catch continue;
        if (fs.unlinkPath(f)) n += 1;
    }
    return n;
}

/// An identity directory whose ranks' directories are all gone: its stamp, then itself.
fn removeIfOnlyStamp(top: [:0]const u8) void {
    var d = fs.Lister.open(top) orelse return;
    var others = false;
    while (d.next()) |ent| {
        if (std.mem.eql(u8, ent.name, "opened")) continue;
        others = true;
    }
    d.close();
    if (others) return;
    var p: [1500]u8 = undefined;
    _ = fs.unlinkPath(std.fmt.bufPrintSentinel(&p, "{s}/opened", .{top}, 0) catch return);
    fs.rmdir(top);
}

/// Bytes an unprivileged writer may still use on `dir`'s file system (no limit known: the most).
pub const diskFree = fs.diskFree;

test {
    _ = @import("spill_test.zig");
}

fn asyncEvent(r: *Read) !Read.Poll {
    const deadline = std.Io.Clock.awake.now(std.testing.io).toNanoseconds() + 5 * std.time.ns_per_s;
    while (true) {
        const event = r.poll();
        if (event != .pending) return event;
        if (std.Io.Clock.awake.now(std.testing.io).toNanoseconds() >= deadline) return error.TestReadTimeout;
        std.Thread.yield() catch {};
    }
}

test "async spill paused disk IO leaves tier available and cancellation nonblocking" {
    const Pause = struct {
        var entered: std.atomic.Value(bool) = .init(false);
        var released: std.atomic.Value(bool) = .init(false);
        fn advice(_: fs.File, at: u64, _: u64) void {
            if (at != 0) return;
            entered.store(true, .release);
            while (!released.load(.acquire)) std.Thread.yield() catch {};
        }
    };
    var f = try MetricsFixture.open();
    defer f.close();
    const name = try f.put(&.{ 1, 2 }, false);
    f.tier.advise_file = Pause.advice;
    Pause.entered.store(false, .release);
    Pause.released.store(false, .release);
    defer Pause.released.store(true, .release);
    const r = try f.tier.readAsync(name);
    const deadline = std.Io.Clock.awake.now(std.testing.io).toNanoseconds() + 5 * std.time.ns_per_s;
    while (!Pause.entered.load(.acquire)) {
        if (std.Io.Clock.awake.now(std.testing.io).toNanoseconds() >= deadline) return error.TestReadTimeout;
        std.Thread.yield() catch {};
    }
    try std.testing.expect(r.description() == null);
    try std.testing.expect(r.poll() == .pending);
    try std.testing.expectEqual(@as(usize, 1), f.tier.info().entries);
    r.release();
    try std.testing.expectError(error.ReadBusy, f.tier.readAsync(name));
    Pause.released.store(true, .release);
    f.tier.shutdown();
    try std.testing.expectEqual(@as(u64, 0), f.tier.info().stats.load_failed);
    try std.testing.expect(f.tier.has(name));
}

test "async spill mailbox stays bounded and borrowed until consumed" {
    var f = try MetricsFixture.open();
    defer f.close();
    f.tier.chunk = 5;
    const name = try f.put(&.{ 1, 2 }, false);
    const r = try f.tier.readAsync(name);
    defer r.release();
    try std.testing.expectError(error.ReadBusy, f.tier.readAsync(name));
    const first = try asyncEvent(r);
    try std.testing.expect(first == .chunk);
    try std.testing.expectEqual(@as(usize, 5), first.chunk.bytes.len);
    try std.testing.expectEqualSlices(u8, &.{ 7, 7, 7, 7, 7 }, first.chunk.bytes);
    try std.testing.expectEqual(@as(u64, 16), r.description().?.header.payload_len);
    for (0..100) |_| {
        std.Thread.yield() catch {};
        const same = r.poll();
        try std.testing.expectEqual(first.chunk.offset, same.chunk.offset);
        try std.testing.expectEqual(first.chunk.bytes.ptr, same.chunk.bytes.ptr);
    }
    var bytes: u64 = 0;
    while (true) {
        switch (try asyncEvent(r)) {
            .chunk => |c| {
                try std.testing.expectEqual(bytes, c.offset);
                try std.testing.expect(c.bytes.len <= 5);
                bytes += c.bytes.len;
                r.consume();
            },
            .done => break,
            .failed => |err| return err,
            .pending => unreachable,
        }
    }
    try std.testing.expectEqual(@as(u64, 16), bytes);
    try std.testing.expectEqual(@as(u64, 1), f.tier.info().stats.loaded);
    try std.testing.expectEqual(@as(usize, 0), f.tier.rbuf.len);
}

test "async spill cancellation releases blocked producer without corruption" {
    var f = try MetricsFixture.open();
    defer f.close();
    f.tier.chunk = 5;
    const name = try f.put(&.{ 1, 2 }, false);
    const r = try f.tier.readAsync(name);
    try std.testing.expect((try asyncEvent(r)) == .chunk);
    r.cancel();
    r.release();
    f.tier.shutdown();
    try std.testing.expect(f.tier.reader == null);
    try std.testing.expect(f.tier.has(name));
    try std.testing.expectEqual(@as(u64, 0), f.tier.info().stats.load_failed);
    try std.testing.expectEqual(@as(u64, 0), f.tier.info().stats.loaded);
}

test "async spill shutdown cancels an unreleased borrowed chunk" {
    var f = try MetricsFixture.open();
    defer f.close();
    f.tier.chunk = 5;
    const name = try f.put(&.{ 1, 2 }, false);
    const r = try f.tier.readAsync(name);
    try std.testing.expect((try asyncEvent(r)) == .chunk);
    f.tier.shutdown();
    try std.testing.expect(f.tier.reader == null);
    try std.testing.expect(f.tier.has(name));
    try std.testing.expectEqual(@as(u64, 0), f.tier.info().stats.load_failed);
}

test "async spill promotes validated disk payload and reuses RAM" {
    var f = try MetricsFixture.open();
    defer f.close();
    f.tier.chunk = 5;
    f.tier.settings.ram_cap = 32;
    const name = try f.put(&.{ 1, 2 }, false);
    for (0..2) |attempt| {
        const r = try f.tier.readAsync(name);
        var bytes: u64 = 0;
        while (true) switch (try asyncEvent(r)) {
            .chunk => |c| {
                try std.testing.expectEqual(bytes, c.offset);
                for (c.bytes) |b| try std.testing.expectEqual(@as(u8, 7), b);
                bytes += c.bytes.len;
                r.consume();
            },
            .done => break,
            .failed => |err| return err,
            .pending => unreachable,
        };
        try std.testing.expectEqual(@as(u64, 16), bytes);
        r.release();
        const info = f.tier.info();
        try std.testing.expectEqual(@as(u64, 1), info.stats.ram_promoted);
        try std.testing.expectEqual(@as(u64, attempt), info.stats.ram_loaded);
        try std.testing.expectEqual(@as(u64, 16), info.ram_bytes);
        try std.testing.expect(f.tier.isRam(name));
    }
}

test "async spill corruption invalidation cannot forget a replacement" {
    var f = try MetricsFixture.open();
    defer f.close();
    f.tier.chunk = 5;
    const name = try f.put(&.{ 1, 2 }, false);
    var before = try f.tier.read(name);
    var nb: [32]u8 = undefined;
    const fd = fs.createTrunc(f.tier.dfd, name.file(&nb)) orelse return error.TestOpen;
    try fs.pwriteAll(fd, &before.head, 0);
    try fs.pwriteAll(fd, std.mem.sliceAsBytes(before.tokens), header_bytes);
    const bad: [16]u8 = @splat(8);
    try fs.pwriteAll(fd, &bad, Header.payloadOffset(before.tokens.len));
    fs.close(fd);
    before.close();
    const r = try f.tier.readAsync(name);
    defer r.release();
    while (true) switch (try asyncEvent(r)) {
        .chunk => r.consume(),
        .failed => |err| {
            try std.testing.expectEqual(error.PayloadHash, err);
            break;
        },
        else => return error.TestExpectedFailure,
    };
    try std.testing.expect(f.tier.has(name));
    f.tier.forget(name);
    _ = try f.put(&.{ 1, 2 }, false);
    r.forgetFailed();
    r.forgetInvalid();
    try std.testing.expect(f.tier.has(name));
    try std.testing.expectEqual(@as(u64, 1), f.tier.info().stats.hash_failed);
    try std.testing.expectEqual(@as(u64, 0), f.tier.info().stats.loaded);
}

const MetricsFixture = struct {
    tmp: std.testing.TmpDir,
    path: []u8,
    tier: *Tier,

    fn open() !MetricsFixture {
        const tmp = std.testing.tmpDir(.{});
        // a root the tier makes (and makes private): the test directory itself inherits its parent's access
        const path = try std.fmt.allocPrint(std.testing.allocator, ".zig-cache/tmp/{s}/tier", .{tmp.sub_path});
        errdefer std.testing.allocator.free(path);
        const t = try Tier.open(std.testing.allocator, std.testing.io, .{ .dir = path, .cap = 1 << 20, .min_free = 0 }, "metrics", @splat(0x31), 0, 1);
        return .{ .tmp = tmp, .path = path, .tier = t };
    }

    fn close(f: *MetricsFixture) void {
        f.tier.deinit();
        std.testing.allocator.free(f.path);
        f.tmp.cleanup();
    }

    fn source(_: *anyopaque, _: u64, out: []u8) !void {
        @memset(out, 7);
    }
    fn broken(_: *anyopaque, _: u64, _: []u8) !void {
        return error.MetricsSourceFailed;
    }
    fn sink(_: *anyopaque, _: u64, _: []const u8) !void {}
    fn usable(_: void, _: u32) bool {
        return true;
    }
    fn noFree(_: [:0]const u8) u64 {
        return 0;
    }

    fn put(f: *MetricsFixture, tokens: []const u32, fail: bool) !Name {
        const name = Name.of(tokens, @intCast(tokens.len));
        const j = try f.tier.write(name, tokens, "", 16, .{ .ptr = f, .read = if (fail) broken else source });
        defer j.release();
        j.waitCopied();
        try std.testing.expect(f.tier.drain(10));
        // drain can see busy cleared immediately before finish; wait for the job's publication.
        while (!j.done()) std.Thread.yield() catch {};
        try std.testing.expectEqual(!fail, j.ok);
        return name;
    }
};

test "spill metrics lookup outcomes and occupancy snapshot" {
    var f = try MetricsFixture.open();
    defer f.close();
    const name = try f.put(&.{ 1, 2 }, false);
    const t = f.tier;
    try std.testing.expect(t.best(&.{ 1, 2, 3 }, 0, {}, MetricsFixture.usable) != null);
    try std.testing.expect(t.best(&.{ 1, 4, 3 }, 0, {}, MetricsFixture.usable) == null);
    try std.testing.expect(t.best(&.{ 1, 2, 3 }, 2, {}, MetricsFixture.usable) == null);
    var info = t.info();
    try std.testing.expectEqual(@as(u64, 1), info.stats.disk_hits);
    try std.testing.expectEqual(@as(u64, 2), info.stats.disk_misses);
    try std.testing.expectEqual(@as(u64, 8208), info.stored_bytes);
    try std.testing.expectEqual(@as(u64, 0), info.queued_bytes);
    try std.testing.expectEqual(@as(usize, 0), info.pending);
    try std.testing.expectEqual(@as(u64, 1 << 20), info.cap);
    // Model an indexed queued write under the mutex without starting another thread.
    var job: Job = .{ .tier = t, .name = name, .tokens = &.{}, .meta = &.{}, .payload_len = 16, .source = .{ .ptr = &f, .read = MetricsFixture.source } };
    t.mutex.lockUncancelable(t.io);
    t.entries.items[0].job = &job;
    t.mutex.unlock(t.io);
    info = t.info();
    t.mutex.lockUncancelable(t.io);
    t.entries.items[0].job = null;
    t.mutex.unlock(t.io);
    try std.testing.expectEqual(@as(usize, 0), info.entries);
    try std.testing.expectEqual(@as(usize, 1), info.pending);
    try std.testing.expectEqual(@as(u64, 0), info.stored_bytes);
    try std.testing.expectEqual(@as(u64, 8208), info.queued_bytes);
}

test "spill metrics timing includes successful and failed attempts" {
    var f = try MetricsFixture.open();
    defer f.close();
    const name = try f.put(&.{ 1, 2 }, false);
    _ = try f.put(&.{ 4, 5 }, true);
    var r = try f.tier.read(name);
    defer r.close();
    try r.pump(.{ .ptr = &f, .write = MetricsFixture.sink });
    // A stored entry whose file disappeared fails during begin(), not pump().
    var nb: [32]u8 = undefined;
    try std.testing.expect(fs.unlink(f.tier.dfd, name.file(&nb)));
    try std.testing.expectError(error.SpillOpen, f.tier.read(name));
    const stats = f.tier.info().stats;
    try std.testing.expectEqual(@as(u64, 2), stats.write_count);
    try std.testing.expectEqual(@as(u64, 2), stats.load_count);
    try std.testing.expect(stats.write_ns > 0);
    try std.testing.expect(stats.load_ns > 0);
    try std.testing.expectEqual(@as(u64, 1), stats.written);
    try std.testing.expectEqual(@as(u64, 1), stats.write_failed);
    try std.testing.expectEqual(@as(u64, 1), stats.loaded);
    try std.testing.expectEqual(@as(u64, 1), stats.load_failed);
    try std.testing.expectEqual(@as(u64, 8208), stats.written_bytes);
    try std.testing.expectEqual(@as(u64, 8208), stats.loaded_bytes);
}

test "spill metrics unlink separates logical bytes from observed reclaim" {
    var f = try MetricsFixture.open();
    defer f.close();
    const name = try f.put(&.{ 1, 2 }, false);
    var r = try f.tier.read(name);
    defer r.close();
    // Deterministic floor observation, as in existing floor tests: open reader returns no space.
    f.tier.free_space = MetricsFixture.noFree;
    f.tier.forget(name);
    var stats = f.tier.info().stats;
    try std.testing.expectEqual(@as(u64, 8208), stats.unlinked_bytes);
    try std.testing.expectEqual(@as(u64, 0), stats.reclaimed_bytes);
    try r.pump(.{ .ptr = &f, .write = MetricsFixture.sink });
    // Failed unlink must not add either counter.
    f.tier.mutex.lockUncancelable(f.tier.io);
    f.tier.unlink(name, 8208);
    f.tier.mutex.unlock(f.tier.io);
    stats = f.tier.info().stats;
    try std.testing.expectEqual(@as(u64, 8208), stats.unlinked_bytes);
    try std.testing.expectEqual(@as(u64, 0), stats.reclaimed_bytes);
    // Deterministic observed delta greater than logical size: the counter must cap it.
    f.tier.free_space = diskFree;
    const second = try f.put(&.{ 4, 5 }, false);
    f.tier.free_space = struct {
        fn observed(dir: [:0]const u8) u64 {
            var d = fs.Lister.open(dir) orelse unreachable;
            defer d.close();
            while (d.next()) |ent| {
                if (std.mem.endsWith(u8, ent.name, extension)) return 0;
            }
            return 1 << 20;
        }
    }.observed;
    f.tier.forget(second);
    stats = f.tier.info().stats;
    try std.testing.expectEqual(@as(u64, 16416), stats.unlinked_bytes);
    try std.testing.expectEqual(@as(u64, 8208), stats.reclaimed_bytes);
}

test "spill metrics pressure evictions count distinct from explicit forget" {
    var f = try MetricsFixture.open();
    defer f.close();
    _ = try f.put(&.{ 1, 2 }, false);
    f.tier.settings.cap = 8208;
    const next = try f.put(&.{ 3, 4 }, false);
    try std.testing.expectEqual(@as(u64, 1), f.tier.info().stats.disk_evictions);
    f.tier.forget(next);
    try std.testing.expectEqual(@as(u64, 1), f.tier.info().stats.disk_evictions);
    _ = try f.put(&.{ 5, 6 }, false);
    f.tier.settings.cap = 0;
    f.tier.mutex.lockUncancelable(f.tier.io);
    f.tier.fitCap();
    f.tier.mutex.unlock(f.tier.io);
    try std.testing.expectEqual(@as(u64, 2), f.tier.info().stats.disk_evictions);
}

test "spill metrics payload allocation failure records one terminal load" {
    var f = try MetricsFixture.open();
    defer f.close();
    const name = try f.put(&.{ 1, 2 }, false);
    const t = f.tier;
    const original = t.gpa;
    var failing = std.testing.FailingAllocator.init(original, .{ .fail_index = 1 });
    t.gpa = failing.allocator();
    defer t.gpa = original;
    var r = try t.read(name);
    defer r.close();
    // The first allocation (header-validated token storage) succeeded. Only the
    // next allocation, the shared payload buffer, is forced to fail.
    try std.testing.expectEqualSlices(u32, &.{ 1, 2 }, r.tokens);
    try std.testing.expect(!failing.has_induced_failure);
    try std.testing.expectEqual(@as(usize, 0), t.rbuf.len);
    try std.testing.expectError(error.OutOfMemory, r.pump(.{ .ptr = &f, .write = MetricsFixture.sink }));
    try std.testing.expect(failing.has_induced_failure);
    const stats = t.info().stats;
    try std.testing.expectEqual(@as(u64, 1), stats.load_failed);
    try std.testing.expectEqual(@as(u64, 1), stats.load_count);
    try std.testing.expect(stats.load_ns > 0);
    try std.testing.expectEqual(@as(u64, 0), stats.loaded);
    try std.testing.expectEqual(@as(u64, 0), stats.loaded_bytes);
    // Conservatively discard the valid copy, matching native cold fallback.
    try std.testing.expect(!t.has(name));
    var nb: [32]u8 = undefined;
    const fd = fs.openRead(t.dfd, name.file(&nb));
    if (fd) |x| fs.close(x);
    try std.testing.expect(fd == null);
    r.close();
    r.close();
    const closed = t.info().stats;
    try std.testing.expectEqual(stats.load_failed, closed.load_failed);
    try std.testing.expectEqual(stats.load_count, closed.load_count);
    try std.testing.expectEqual(stats.load_ns, closed.load_ns);
}

// Advisory only: failure must never turn a valid spill operation into a cache miss.
pub fn pageAdvice(fd: fs.File, at: u64, len: u64) void {
    if (builtin.os.tag == .linux) _ = posix_fadvise(fd, @intCast(at), @intCast(len), 4);
}

test "spill page advice follows fsynced complete writes" {
    const Observe = struct {
        var calls: std.atomic.Value(u64) = .init(0);
        var start: std.atomic.Value(u64) = .init(0);
        var length: std.atomic.Value(u64) = .init(0);
        fn advise(_: fs.File, at: u64, len: u64) void {
            start.store(at, .release);
            length.store(len, .release);
            _ = calls.fetchAdd(1, .acq_rel);
        }
    };
    Observe.calls.store(0, .release);
    var f = try MetricsFixture.open();
    defer f.close();
    f.tier.chunk = 4;
    f.tier.advise_file = Observe.advise;
    _ = try f.put(&.{ 1, 2 }, false);
    try std.testing.expectEqual(@as(u64, 1), Observe.calls.load(.acquire));
    try std.testing.expectEqual(@as(u64, 0), Observe.start.load(.acquire));
    try std.testing.expectEqual(@as(u64, 8208), Observe.length.load(.acquire));
    // An unfinished source must never receive a post-fsync success advisory.
    _ = try f.put(&.{ 4, 5 }, true);
    try std.testing.expectEqual(@as(u64, 1), Observe.calls.load(.acquire));
}

test "spill page advice follows header tokens and each bounded payload read" {
    const Observe = struct {
        var calls: std.atomic.Value(u64) = .init(0);
        var bytes: std.atomic.Value(u64) = .init(0);
        fn advise(_: fs.File, _: u64, len: u64) void {
            _ = bytes.fetchAdd(len, .acq_rel);
            _ = calls.fetchAdd(1, .acq_rel);
        }
    };
    Observe.calls.store(0, .release);
    Observe.bytes.store(0, .release);
    var f = try MetricsFixture.open();
    defer f.close();
    f.tier.chunk = 4;
    const name = try f.put(&.{ 1, 2 }, false);
    f.tier.advise_file = Observe.advise;
    var r = try f.tier.read(name);
    defer r.close();
    try r.pump(.{ .ptr = &f, .write = MetricsFixture.sink });
    try std.testing.expectEqual(@as(u64, 6), Observe.calls.load(.acquire));
    try std.testing.expectEqual(@as(u64, header_bytes + 8 + 16), Observe.bytes.load(.acquire));
}

const posix_fadvise = if (builtin.os.tag == .linux) struct {
    extern "c" fn posix_fadvise(fd: c_int, offset: i64, len: i64, advice: c_int) c_int;
}.posix_fadvise else undefined;
