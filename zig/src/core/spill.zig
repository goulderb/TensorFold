//! Spill tier: kept prompt states written to local disk when they leave the prompt cache, read back when a later
//! prompt extends them, also after a restart (docs/spill-tier.md). Nothing here knows a model: a family hands a state
//! over as its tokens, a meta blob and a payload `Source`, and reads it back through a `Sink`.
//!
//! A port of the GLM spill tier by Robert Wojciechowski (wojo, https://github.com/wojo), Mia's patch 0088 for
//! MiaAI-Lab's GLM-5.3-Flash TensorFold recipe (tensorfold/cuda/spill.py, Apache-2.0): its settings, its LRU by bytes
//! under a cap and a free-disk floor, its write-then-rename files and its one-directory-per-build layout. The file and
//! directory conventions also follow upstream TensorFold's --learn (zig/src/core/prompt_imprint.zig).
const std = @import("std");
const builtin = @import("builtin");
const Allocator = std.mem.Allocator;

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
        s.dir = if (env.get("TENSORFOLD_SPILL_DIR")) |d| try a.dupe(u8, trim(d)) else blk: {
            const home = env.get("HOME") orelse return error.NoSpillDir;
            break :blk try std.fmt.allocPrint(a, "{s}/.cache/tensorfold/spill", .{home});
        };
        if (s.dir.len == 0 or s.dir[0] != '/') {
            a.free(s.dir);
            return error.SpillDirNotAbsolute;
        }
        return s;
    }

    /// What both ranks must share, as text for the ranks' settings comparison.
    pub fn text(s: Settings, buf: []u8) ![]const u8 {
        return std.fmt.bufPrint(buf, "spill={d}/{d}/{d}/{d}/{d}/{d}", .{ s.cap >> 20, s.highwater, s.min_tokens, s.min_free >> 20, s.flush_s, s.keep_builds });
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
};

/// What /health reports of a tier.
pub const Info = struct { stats: Stats, entries: usize, pending: usize, stored_bytes: u64, queued_bytes: u64, disk_free: u64, cap: u64 };

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
pub const Job = struct {
    tier: *Tier,
    name: Name,
    tokens: []u32,
    meta: []u8,
    payload_len: u64,
    source: Source,
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
        gpa.free(j.tokens);
        gpa.free(j.meta);
        gpa.destroy(j);
    }
};

const Entry = struct { name: Name, tokens: []u32, bytes: u64, used: u64, job: ?*Job = null, source_id: usize = 0 };

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
    /// `dir` opened once checked private: every file of the tier is made, read and deleted relative to it
    dfd: c_int = -1,
    mutex: std.Io.Mutex = .init,
    cond: std.Io.Condition = .init,
    entries: std.ArrayList(*Entry) = .empty,
    queue: std.ArrayList(*Job) = .empty,
    busy: ?*Job = null,
    clock: u64 = 0,
    /// states written after this clock are not evicted for room (a batch's own writes never evict each other)
    pin: ?u64 = null,
    thread: ?std.Thread = null,
    closing: bool = false,
    stats: Stats = .{},
    hook: ?Hook = null,
    /// bytes a copy or a read moves at a time (64 MiB; small in tests)
    chunk: usize = 64 << 20,
    wbuf: []u8 = &.{},
    rbuf: []u8 = &.{},
    /// free bytes on the tier's disk (statvfs; tests replace it)
    free_space: *const fn (dir: [:0]const u8) u64 = diskFree,
    /// File writes, separate from Source.read; tests inject a real ENOSPC after a partial payload.
    write_file: *const fn (fd: c_int, bytes: []const u8, at: u64) anyerror!void = pwriteAll,
    advise_file: *const fn (fd: c_int, at: u64, len: u64) void = pageAdvice,

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
        try makePath(gpa, dir);
        // the root, identity and rank directories, each opened without following a link and checked private
        const root_path = try gpa.dupeSentinel(u8, settings.dir, 0);
        defer gpa.free(root_path);
        const root_fd = try openPrivate(std.posix.AT.FDCWD, root_path);
        defer _ = std.c.close(root_fd);
        var tn: [40]u8 = undefined;
        const top_fd = try openPrivate(root_fd, try std.fmt.bufPrintSentinel(&tn, "{s}-{x}", .{ family, identity[0..8] }, 0));
        defer _ = std.c.close(top_fd);
        var rn: [24]u8 = undefined;
        t.dfd = try openPrivate(top_fd, try std.fmt.bufPrintSentinel(&rn, "rank{d}", .{rank}, 0));
        errdefer _ = std.c.close(t.dfd);
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
        t.gpa.free(t.wbuf);
        t.gpa.free(t.rbuf);
        _ = std.c.close(t.dfd);
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
        t.gpa.free(e.tokens);
        t.gpa.destroy(e);
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
        return e.job == null and std.mem.eql(u32, e.tokens, tokens);
    }

    /// Whether the existing copy is still being written (never waits for that writer).
    pub fn pending(t: *Tier, name: Name) bool {
        t.mutex.lockUncancelable(t.io);
        defer t.mutex.unlock(t.io);
        const i = t.find(name) orelse return false;
        return t.entries.items[i].job != null;
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
            if (e.job != null or e.name.at <= beyond or e.name.at >= prompt.len or e.tokens.len > prompt.len) continue;
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
        if (e.job) |j| j.cancel.store(true, .release) else t.unlink(e.name, e.bytes);
        t.freeEntry(e);
    }

    fn unlink(t: *Tier, name: Name, bytes: u64) void {
        var buf: [32]u8 = undefined;
        const before = t.free_space(t.dir);
        if (std.c.unlinkat(t.dfd, name.file(&buf), 0) == 0) {
            t.stats.unlinked_bytes += bytes;
            t.stats.reclaimed_bytes += @min(bytes, t.free_space(t.dir) -| before);
        }
    }

    /// Bytes the stored and queued states take, and those of the queued ones alone (lock held).
    fn totals(t: *const Tier) struct { all: u64, queued: u64 } {
        var all: u64 = 0;
        var queued: u64 = 0;
        for (t.entries.items) |e| {
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
        t.mutex.lockUncancelable(t.io);
        defer t.mutex.unlock(t.io);
        if (t.closing) return error.TierClosed;
        if (t.find(name) != null) {
            t.stats.skipped += 1;
            return error.AlreadyStored;
        }
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
        try t.entries.ensureUnusedCapacity(t.gpa, 1);
        try t.queue.ensureUnusedCapacity(t.gpa, 1);
        if (t.thread == null) t.thread = try std.Thread.spawn(.{}, run, .{t});
        t.clock += 1;
        e.used = t.clock;
        t.entries.appendAssumeCapacity(e);
        t.queue.appendAssumeCapacity(j);
        t.cond.broadcast(t.io);
        return j;
    }

    /// Whether `e` may go for room: stored (not on its way) and not one of a pinned batch.
    fn evictable(t: *const Tier, e: *const Entry) bool {
        return e.job == null and (t.pin == null or e.used <= t.pin.?);
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
        const fd = std.c.openat(t.dfd, partial, .{ .ACCMODE = .WRONLY, .CREAT = true, .EXCL = true, .NOFOLLOW = true, .CLOEXEC = true }, @as(std.c.mode_t, 0o600));
        if (fd < 0) return error.SpillOpen;
        var ok = false;
        defer {
            _ = std.c.close(fd);
            if (!ok) _ = std.c.unlinkat(t.dfd, partial, 0);
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
        if (std.c.ftruncate(fd, @intCast(h.fileBytes())) != 0) return error.SpillWrite;
        if (std.c.fsync(fd) != 0) return error.SpillSync;
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
            if (std.c.renameat(t.dfd, partial.?, t.dfd, j.name.file(&nb)) == 0) {
                renamed = true;
                t.entries.items[i.?].job = null;
                t.stats.written += 1;
                t.stats.written_bytes += t.entries.items[i.?].bytes;
            }
        }
        if (!renamed) {
            if (partial) |p| _ = std.c.unlinkat(t.dfd, p, 0);
            if (i) |k| t.freeEntry(t.entries.orderedRemove(k));
            if (!j.cancel.load(.acquire)) t.stats.write_failed += 1;
        }
        j.ok = renamed;
        j.state.store(Job.done_state, .release);
        t.cond.broadcast(t.io);
        t.mutex.unlock(t.io);
        if (renamed) _ = std.c.fsync(t.dfd);
        j.release();
    }

    /// Reads stored state `name` back: its header and tokens checked now (the file goes when they fail), its payload
    /// by `Reading.pump`. A write of it still on its way is error.NotStored (never waited for; see `stored`).
    pub fn read(t: *Tier, name: Name) !Reading {
        t.mutex.lockUncancelable(t.io);
        const i = t.find(name) orelse {
            t.mutex.unlock(t.io);
            return error.NotStored;
        };
        if (t.entries.items[i].job != null) {
            t.mutex.unlock(t.io);
            return error.NotStored;
        }
        t.mutex.unlock(t.io);
        var r: Reading = .{ .tier = t, .name = name, .t0 = t.now() };
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
        for (t.entries.items) |e| pending_count += @intFromBool(e.job != null);
        return .{ .stats = t.stats, .entries = t.entries.items.len - pending_count, .pending = pending_count, .stored_bytes = sums.all - sums.queued, .queued_bytes = sums.queued, .disk_free = t.free_space(t.dir), .cap = t.settings.cap };
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
        const d = std.c.opendir(t.dir) orelse return error.SpillDir;
        defer _ = std.c.closedir(d);
        const Found = struct { e: *Entry, saved: u64 };
        var found: std.ArrayList(Found) = .empty;
        defer found.deinit(t.gpa);
        errdefer for (found.items) |f| t.freeEntry(f.e);
        while (std.c.readdir(d)) |ent| {
            const file = std.mem.sliceTo(@as([*:0]const u8, @ptrCast(&ent.name)), 0);
            if (std.mem.endsWith(u8, file, ".partial")) {
                _ = std.c.unlinkat(t.dfd, file, 0);
                continue;
            }
            const name = Name.parse(file) orelse continue;
            const got = t.index(file, name) catch {
                _ = std.c.unlinkat(t.dfd, file, 0);
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
        const fd = std.c.openat(t.dfd, file, .{ .ACCMODE = .RDONLY, .NOFOLLOW = true, .CLOEXEC = true }, @as(std.c.mode_t, 0));
        if (fd < 0) return error.SpillOpen;
        defer _ = std.c.close(fd);
        var head: [header_bytes]u8 = undefined;
        if (!preadAll(fd, &head, 0)) return error.ShortFile;
        const h = try Header.decode(&head, t.identity, t.rank, t.world);
        if (!h.name.eql(name)) return error.OtherName;
        if (fileSize(fd) != h.fileBytes()) return error.ShortFile;
        const tokens = try t.gpa.alloc(u32, h.tokens);
        errdefer t.gpa.free(tokens);
        if (!preadAll(fd, std.mem.sliceAsBytes(tokens), header_bytes)) return error.ShortFile;
        if (std.hash.XxHash3.hash(0, std.mem.sliceAsBytes(tokens)) != h.tokens_hash) return error.TokensHash;
        if (!Name.of(tokens, h.name.at).eql(name) or h.name.at > tokens.len) return error.OtherName;
        const e = try t.gpa.create(Entry);
        e.* = .{ .name = name, .tokens = tokens, .bytes = h.fileBytes(), .used = 0 };
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
        const d = std.c.opendir(root) orelse return 0;
        while (std.c.readdir(d)) |ent| {
            const n = std.mem.sliceTo(&ent.name, 0);
            if (n.len != family.len + 1 + 16 or !std.mem.startsWith(u8, n, family) or n[family.len] != '-') continue;
            if (std.mem.endsWith(u8, t.top, n)) continue;
            var b: Build = .{ .name = undefined, .len = n.len, .stamp = 0 };
            @memcpy(b.name[0..n.len], n);
            var sp: [1300]u8 = undefined;
            b.stamp = readStamp(std.fmt.bufPrintSentinel(&sp, "{s}/{s}", .{ t.settings.dir, n }, 0) catch continue);
            builds.append(t.gpa, b) catch break;
        }
        _ = std.c.closedir(d);
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
            _ = std.c.rmdir(mine);
            removeIfOnlyStamp(other);
        }
        return gone;
    }
};

/// A stored state being read back: its header's meta, its tokens, then its payload through `pump`.
pub const Reading = struct {
    tier: *Tier,
    name: Name,
    fd: c_int = -1,
    head: [header_bytes]u8 = undefined,
    header: Header = undefined,
    tokens: []u32 = &.{},
    t0: i96,
    timed: bool = false,

    fn begin(r: *Reading) !void {
        const t = r.tier;
        var nb: [32]u8 = undefined;
        r.fd = std.c.openat(t.dfd, r.name.file(&nb), .{ .ACCMODE = .RDONLY, .NOFOLLOW = true, .CLOEXEC = true }, @as(std.c.mode_t, 0));
        if (r.fd < 0) return error.SpillOpen;
        if (!preadAll(r.fd, &r.head, 0)) return error.ShortFile;
        t.advise_file(r.fd, 0, r.head.len);
        r.header = try Header.decode(&r.head, t.identity, t.rank, t.world);
        if (!r.header.name.eql(r.name)) return error.OtherName;
        if (fileSize(r.fd) != r.header.fileBytes()) return error.ShortFile;
        r.tokens = try t.gpa.alloc(u32, r.header.tokens);
        if (!preadAll(r.fd, std.mem.sliceAsBytes(r.tokens), header_bytes)) return error.ShortFile;
        t.advise_file(r.fd, header_bytes, std.mem.sliceAsBytes(r.tokens).len);
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
        // Resource failures follow the same conservative discard/cold-fallback policy as
        // other terminal read errors, even when the stored file itself is valid.
        if (t.rbuf.len == 0) t.rbuf = t.gpa.alloc(u8, t.chunk) catch |err| return r.failWith(err);
        var hash = std.hash.XxHash3.init(0);
        const at = Header.payloadOffset(r.header.tokens);
        var off: u64 = 0;
        while (off < r.header.payload_len) {
            const n: usize = @intCast(@min(t.chunk, r.header.payload_len - off));
            if (!preadAll(r.fd, t.rbuf[0..n], at + off)) return r.failWith(error.ShortFile);
            t.advise_file(r.fd, at + off, n);
            hash.update(t.rbuf[0..n]);
            sink.write(sink.ptr, off, t.rbuf[0..n]) catch |err| return r.failWith(err);
            off += n;
        }
        if (hash.final() != r.header.payload_hash) {
            t.mutex.lockUncancelable(t.io);
            t.stats.hash_failed += 1;
            t.mutex.unlock(t.io);
            return r.failWith(error.PayloadHash);
        }
        t.touch(r.name);
        t.mutex.lockUncancelable(t.io);
        t.stats.loaded += 1;
        t.stats.loaded_bytes += r.header.fileBytes();
        t.stats.loaded_tokens += r.header.name.at;
        r.recordTime();
        t.mutex.unlock(t.io);
    }

    fn failWith(r: *Reading, err: anyerror) anyerror {
        r.fail();
        return err;
    }

    /// A read that failed: its file goes (it is not trusted again).
    fn fail(r: *Reading) void {
        const t = r.tier;
        t.mutex.lockUncancelable(t.io);
        t.stats.load_failed += 1;
        r.recordTime();
        t.mutex.unlock(t.io);
        t.forget(r.name);
    }

    // Tier mutex held. Abandoned reads have no terminal sample.
    fn recordTime(r: *Reading) void {
        if (r.timed) return;
        r.timed = true;
        r.tier.stats.load_count += 1;
        r.tier.stats.load_ns += @intCast(@max(0, r.tier.now() - r.t0));
    }

    pub fn close(r: *Reading) void {
        if (r.fd >= 0) _ = std.c.close(r.fd);
        r.fd = -1;
        r.tier.gpa.free(r.tokens);
        r.tokens = &.{};
    }
};

// -- files ----------------------------------------------------------------------------------------------------------

fn pwriteAll(fd: c_int, bytes: []const u8, at: u64) !void {
    var done: usize = 0;
    while (done < bytes.len) {
        const n = std.c.pwrite(fd, bytes.ptr + done, bytes.len - done, @intCast(at + done));
        if (n <= 0) return error.SpillWrite;
        done += @intCast(n);
    }
}

fn preadAll(fd: c_int, dest: []u8, at: u64) bool {
    var done: usize = 0;
    while (done < dest.len) {
        const n = std.c.pread(fd, dest.ptr + done, dest.len - done, @intCast(at + done));
        if (n <= 0) return false;
        done += @intCast(n);
    }
    return true;
}

fn fileSize(fd: c_int) u64 {
    const end = std.c.lseek(fd, 0, std.c.SEEK.END);
    return if (end > 0) @intCast(end) else 0;
}

/// `dir` and its parents, mode 0700 (the files hold prompt token ids).
fn makePath(gpa: Allocator, dir: [:0]const u8) !void {
    const tmp = try gpa.dupeSentinel(u8, dir, 0);
    defer gpa.free(tmp);
    for (tmp[1..], 1..) |ch, i| if (ch == '/') {
        tmp[i] = 0;
        _ = std.c.mkdir(tmp.ptr, 0o700);
        tmp[i] = '/';
    };
    const rc = std.c.mkdir(tmp.ptr, 0o700);
    if (rc != 0 and std.c.errno(rc) != .EXIST) return error.SpillDir;
}

/// Whether an existing `dir` passes the tier's own private-directory check (no link, this user's or root's, not group
/// or world writable), without creating, opening a tier in, or pruning anything.
pub fn checkRoot(dir: [:0]const u8) !void {
    if (dir.len == 0 or dir[0] != '/') return error.SpillDirNotAbsolute;
    _ = std.c.close(try openPrivate(std.c.AT.FDCWD, dir));
}

/// Directory `name` in directory `at`, opened (not through a link) when only this process's user or root can change
/// it; else another user could put a link where the tier writes. The descriptor is the caller's.
fn openPrivate(at: c_int, name: [:0]const u8) !c_int {
    const fd = std.c.openat(at, name, .{ .ACCMODE = .RDONLY, .DIRECTORY = true, .NOFOLLOW = true, .CLOEXEC = true }, @as(std.c.mode_t, 0));
    if (fd < 0) return error.SpillDirUnsafe;
    errdefer _ = std.c.close(fd);
    const st = ownerOf(fd) orelse return error.SpillDir;
    if ((st.uid != std.c.geteuid() and st.uid != 0) or st.mode & 0o022 != 0) return error.SpillDirUnsafe;
    return fd;
}

fn ownerOf(fd: c_int) ?struct { uid: u32, mode: u32 } {
    if (builtin.os.tag == .linux) {
        var sx: std.os.linux.Statx = undefined;
        if (std.c.statx(fd, "", 0x1000, .{ .MODE = true, .UID = true }, &sx) != 0) return null; // AT_EMPTY_PATH
        return .{ .uid = sx.uid, .mode = sx.mode };
    }
    var st: std.c.Stat = undefined;
    if (std.c.fstat(fd, &st) != 0) return null;
    return .{ .uid = st.uid, .mode = @intCast(st.mode) };
}

/// The start stamp in the identity's directory `top`.
fn writeStamp(top: c_int, seconds: u64) void {
    const fd = std.c.openat(top, "opened", .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true, .NOFOLLOW = true, .CLOEXEC = true }, @as(std.c.mode_t, 0o600));
    if (fd < 0) return;
    defer _ = std.c.close(fd);
    var w: [8]u8 = undefined;
    std.mem.writeInt(u64, &w, seconds, .little);
    pwriteAll(fd, &w, 0) catch {};
}

fn readStamp(top: [:0]const u8) u64 {
    var p: [1400]u8 = undefined;
    const path = std.fmt.bufPrintSentinel(&p, "{s}/opened", .{top}, 0) catch return 0;
    const fd = std.c.open(path, .{ .ACCMODE = .RDONLY }, @as(std.c.mode_t, 0));
    if (fd < 0) return 0;
    defer _ = std.c.close(fd);
    var w: [8]u8 = undefined;
    return if (preadAll(fd, &w, 0)) std.mem.readInt(u64, &w, .little) else 0;
}

/// Regular files directly in `dir` unlinked; how many.
fn removeFiles(dir: [:0]const u8) u64 {
    const d = std.c.opendir(dir) orelse return 0;
    defer _ = std.c.closedir(d);
    var n: u64 = 0;
    while (std.c.readdir(d)) |ent| {
        if (ent.type != std.c.DT.REG) continue;
        var p: [1600]u8 = undefined;
        const f = std.fmt.bufPrintSentinel(&p, "{s}/{s}", .{ dir, std.mem.sliceTo(&ent.name, 0) }, 0) catch continue;
        if (std.c.unlink(f) == 0) n += 1;
    }
    return n;
}

/// An identity directory whose ranks' directories are all gone: its stamp, then itself.
fn removeIfOnlyStamp(top: [:0]const u8) void {
    const d = std.c.opendir(top) orelse return;
    var others = false;
    while (std.c.readdir(d)) |ent| {
        const n = std.mem.sliceTo(&ent.name, 0);
        if (std.mem.eql(u8, n, ".") or std.mem.eql(u8, n, "..") or std.mem.eql(u8, n, "opened")) continue;
        others = true;
    }
    _ = std.c.closedir(d);
    if (others) return;
    var p: [1500]u8 = undefined;
    _ = std.c.unlink(std.fmt.bufPrintSentinel(&p, "{s}/opened", .{top}, 0) catch return);
    _ = std.c.rmdir(top);
}

const Statvfs = extern struct {
    bsize: c_ulong,
    frsize: c_ulong,
    blocks: u64,
    bfree: u64,
    bavail: u64,
    files: u64,
    ffree: u64,
    favail: u64,
    fsid: c_ulong,
    flag: c_ulong,
    namemax: c_ulong,
    spare: [6]c_int,
};
extern "c" fn statvfs(path: [*:0]const u8, buf: *Statvfs) c_int;

/// Bytes an unprivileged writer may still use on `dir`'s file system (no limit known: the most).
fn diskFree(dir: [:0]const u8) u64 {
    if (builtin.os.tag != .linux) return std.math.maxInt(u64);
    var st: Statvfs = undefined;
    if (statvfs(dir.ptr, &st) != 0) return std.math.maxInt(u64);
    return st.bavail *| @as(u64, st.frsize);
}

test {
    _ = @import("spill_test.zig");
}

const MetricsFixture = struct {
    tmp: std.testing.TmpDir,
    path: []u8,
    tier: *Tier,

    fn open() !MetricsFixture {
        const tmp = std.testing.tmpDir(.{});
        const path = try std.fmt.allocPrint(std.testing.allocator, ".zig-cache/tmp/{s}", .{tmp.sub_path});
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
    try std.testing.expectEqual(@as(c_int, 0), std.c.unlinkat(f.tier.dfd, name.file(&nb), 0));
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
            const d = std.c.opendir(dir) orelse unreachable;
            defer _ = std.c.closedir(d);
            while (std.c.readdir(d)) |ent| {
                if (std.mem.endsWith(u8, std.mem.sliceTo(&ent.name, 0), extension)) return 0;
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
    const fd = std.c.openat(t.dfd, name.file(&nb), .{ .ACCMODE = .RDONLY, .CLOEXEC = true }, @as(std.c.mode_t, 0));
    if (fd >= 0) _ = std.c.close(fd);
    try std.testing.expect(fd < 0);
    r.close();
    r.close();
    const closed = t.info().stats;
    try std.testing.expectEqual(stats.load_failed, closed.load_failed);
    try std.testing.expectEqual(stats.load_count, closed.load_count);
    try std.testing.expectEqual(stats.load_ns, closed.load_ns);
}

// Advisory only: failure must never turn a valid spill operation into a cache miss.
pub fn pageAdvice(fd: c_int, at: u64, len: u64) void {
    if (@import("builtin").os.tag == .linux) _ = posix_fadvise(fd, @intCast(at), @intCast(len), 4);
}

test "spill page advice follows fsynced complete writes" {
    const Observe = struct {
        var calls: std.atomic.Value(u64) = .init(0);
        var start: std.atomic.Value(u64) = .init(0);
        var length: std.atomic.Value(u64) = .init(0);
        fn advise(_: c_int, at: u64, len: u64) void {
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
        fn advise(_: c_int, _: u64, len: u64) void {
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

extern "c" fn posix_fadvise(fd: c_int, offset: i64, len: i64, advice: c_int) c_int;
