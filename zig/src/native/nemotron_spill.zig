//! Nemotron CUDA kept states on the spill tier (docs/spill-tier.md): an owned host copy, written by the tier's thread.
//!
//! One owner thread: construction, then the lane host's thread (every Store call and every Tier.write runs there),
//! then the closing thread after that thread is joined. `spill_write` binds the context, waits for the engine's
//! non-blocking stream (the snapshot's copies are queued there; a blocking download on the legacy stream would not
//! wait for them), downloads the whole state and head into one host slot it owns, and only then returns. The tier's
//! writer thread reads that slot: it never calls CUDA nor touches a Copy, a Target or a live sequence. The slot is
//! reused once its job has `copied`; a job's outcome is read, and its reference released, once it is `done`.
//! `spill_read` validates the file against the loaded engine before allocating, uploads with the blocking
//! `DeviceBuffer.upload` (it synchronizes the legacy stream), and returns a Copy only after the payload hash checked.
//! While serving, a busy slot is a skip (`Tier.unwritten(.incomplete)`), never a failed write. A clean stop instead
//! waits for the slot, so one flush pass offers every kept state once and its counts are real outcomes, not retries.
const std = @import("std");
const cuda = @import("cuda");
const api = @import("engine_api");
const nemotron = @import("nemotron");
const pc = api.prompt_cache;
const spill = pc.spill;
const N = nemotron.native;

pub const lookahead: u32 = 1;

/// Logical KV bytes per token (including the head), recurrent bytes and the source window.
pub const Layout = struct {
    kv_row: u64,
    mamba: u64,
    head: bool,
    max_len: u64,
    grid: u32,

    pub fn of(sh: N.SpillShape, grid: u32) !Layout {
        const l: Layout = .{ .kv_row = sh.kv_row, .mamba = sh.mamba, .head = sh.head, .max_len = sh.max_len, .grid = grid };
        if (grid == 0 or l.max_len == 0 or l.kv_row == 0 or l.kv_row % 2 != 0 or l.mamba == 0 or l.mamba % 2 != 0) return error.BadLayout;
        _ = try l.payload();
        return l;
    }

    pub fn kvBytes(l: Layout, at: u64) u64 {
        std.debug.assert(at <= l.max_len);
        return l.kv_row * at; // checked at max_len during admission
    }

    pub fn payloadAt(l: Layout, at: u64) !u64 {
        if (at > l.max_len) return error.BadLayout;
        const kv = std.math.mul(u64, l.kv_row, at) catch return error.BadLayout;
        return std.math.add(u64, std.math.add(u64, kv, l.mamba) catch return error.BadLayout, 16) catch error.BadLayout;
    }

    /// The worst-case slot reservation. A file writes only payloadAt(its position).
    pub fn payload(l: Layout) !u64 {
        return l.payloadAt(l.max_len);
    }

    fn ranges(l: Layout, at: u64) [3]pc.rows.image.Range {
        const kv = l.kvBytes(at);
        return .{ .{ .off = 0, .len = 16 }, .{ .off = 16, .len = kv }, .{ .off = 16 + kv, .len = l.mamba } };
    }
};

// ---- the private codec: fixed little-endian fields, nothing from struct memory ------------------------------------

const magic = "NEMOCU02";
const codec_version: u32 = 2;
const flag_head: u32 = 1;
/// magic, version, length, flags, grid, lookahead; KV row, Mamba bytes, window, payload, position.
pub const meta_len: usize = 8 + 4 * 5 + 8 * 5;
pub const Scalars = struct { pos: u64 };

const Writer = struct {
    buf: *[meta_len]u8,
    at: usize = 0,
    fn bytes(w: *Writer, b: []const u8) void {
        @memcpy(w.buf[w.at..][0..b.len], b);
        w.at += b.len;
    }
    fn int(w: *Writer, comptime T: type, v: T) void {
        std.mem.writeInt(T, w.buf[w.at..][0..@sizeOf(T)], v, .little);
        w.at += @sizeOf(T);
    }
};

const Reader = struct {
    buf: []const u8,
    at: usize = 0,
    fn int(r: *Reader, comptime T: type) T {
        const v = std.mem.readInt(T, r.buf[r.at..][0..@sizeOf(T)], .little);
        r.at += @sizeOf(T);
        return v;
    }
};

/// Explicit little-endian metadata for the compact image owned by cuda_disk.image.
pub fn encode(l: Layout, scalar: Scalars, out: *[meta_len]u8) void {
    var w: Writer = .{ .buf = out };
    w.bytes(magic);
    w.int(u32, codec_version);
    w.int(u32, meta_len);
    w.int(u32, if (l.head) flag_head else 0);
    w.int(u32, l.grid);
    w.int(u32, lookahead);
    w.int(u64, l.kv_row);
    w.int(u64, l.mamba);
    w.int(u64, l.max_len);
    w.int(u64, l.payloadAt(scalar.pos) catch unreachable);
    w.int(u64, scalar.pos);
    std.debug.assert(w.at == meta_len);
}

pub fn decode(l: Layout, m: []const u8, name: spill.Name) !Scalars {
    if (m.len != meta_len) return error.SpillMetaLength;
    if (!std.mem.eql(u8, m[0..8], magic)) return error.SpillMetaMagic;
    var r: Reader = .{ .buf = m, .at = 8 };
    if (r.int(u32) != codec_version) return error.SpillMetaVersion;
    if (r.int(u32) != meta_len) return error.SpillMetaLength;
    const flags = r.int(u32);
    if (flags & ~flag_head != 0) return error.SpillMetaFlags;
    if ((flags & flag_head != 0) != l.head) return error.SpillLayout;
    if (r.int(u32) != l.grid or r.int(u32) != lookahead) return error.SpillLayout;
    if (r.int(u64) != l.kv_row or r.int(u64) != l.mamba or r.int(u64) != l.max_len) return error.SpillLayout;
    const bytes = r.int(u64);
    const pos = r.int(u64);
    try checkPosition(l, pos, name.at);
    if (bytes != try l.payloadAt(pos)) return error.SpillLayout;
    return .{ .pos = pos };
}

/// A kept state's position: the name's, a grid point a prompt pass keeps at and resumes from. A pass runs only when
/// its prompt and a window's rows fit the cache (Engine.prefillFrom: prompt.len + max_rows <= max_len), and a kept
/// state lies inside its prompt with the lookahead token after it: pos + lookahead + max_rows <= max_len.
fn checkPosition(l: Layout, pos: u64, at: u32) !void {
    if (pos != at or pos == 0 or pos % l.grid != 0) return error.SpillScalars; // `pos == at` first: no overflow below
    if (pos + lookahead + N.max_rows > l.max_len) return error.SpillScalars;
}

// ---- the adapter -----------------------------------------------------------------------------------------------------

/// The CUDA context and the engine stream the snapshots are queued on.
pub const Device = struct { ctx: *const cuda.Context, stream: cuda.Stream };

/// Host bytes beyond the slot the tier may allocate: two transfer buffers and its index, tokens and jobs.
pub fn tierQuota(chunk: usize, control: u64) u64 {
    return 2 * @as(u64, chunk) + control;
}
/// Index, token and job bytes allowed beside the transfer buffers.
pub const control_bytes: u64 = 64 << 20;

/// Counts the adapter alone sees (the tier's Stats and the Store's SpillCounts hold the rest).
pub const Counts = struct {
    submitted: u64 = 0, // writes handed to the tier (not proof of persistence)
    written: u64 = 0, // submitted writes the tier finished on disk
    lost: u64 = 0, // submitted writes the tier gave up (cancelled, disk full, I/O error)
    skipped_busy: u64 = 0, // spill opportunities skipped: the slot still held an earlier write's bytes
    capture_ns: u64 = 0, // owner time fencing the stream and downloading states
    read_ok: u64 = 0, // states read back whole
    read_failed: u64 = 0, // reads refused or failed (not counting NotStored)
};

const retired_most = 4;

/// What a clean stop's flush did: states written, lost (the tier gave them up), refused (the tier or the capture
/// failed), and left unwritten (the flush time ran out, or an earlier failure stopped it); `counts` as the tier closed.
pub const Flushed = struct { written: u64 = 0, lost: u64 = 0, failed: u64 = 0, left: u64 = 0, counts: Counts = .{} };

/// A clean stop's flush in progress: its deadline for new submissions and what it could not write.
const Flush = struct { end: i96, left: u64 = 0, failed: u64 = 0, stopped: bool = false };

/// The quota-limited allocator the tier gets: a full quota refuses (a write is refused, a read goes cold).
const Quota = struct {
    parent: std.mem.Allocator,
    limit: u64,
    used: std.atomic.Value(u64) = .init(0),

    fn allocator(q: *Quota) std.mem.Allocator {
        return .{ .ptr = q, .vtable = &.{ .alloc = alloc, .resize = resize, .remap = remap, .free = free } };
    }
    fn take(q: *Quota, n: usize) bool {
        var now = q.used.load(.monotonic);
        while (true) {
            if (now + n > q.limit) return false;
            now = q.used.cmpxchgWeak(now, now + n, .monotonic, .monotonic) orelse return true;
        }
    }
    fn alloc(p: *anyopaque, len: usize, a: std.mem.Alignment, ra: usize) ?[*]u8 {
        const q: *Quota = @ptrCast(@alignCast(p));
        if (!q.take(len)) return null;
        return q.parent.rawAlloc(len, a, ra) orelse {
            _ = q.used.fetchSub(len, .monotonic);
            return null;
        };
    }
    fn resize(p: *anyopaque, m: []u8, a: std.mem.Alignment, n: usize, ra: usize) bool {
        const q: *Quota = @ptrCast(@alignCast(p));
        if (n > m.len and !q.take(n - m.len)) return false;
        if (q.parent.rawResize(m, a, n, ra)) {
            if (n < m.len) _ = q.used.fetchSub(m.len - n, .monotonic);
            return true;
        }
        if (n > m.len) _ = q.used.fetchSub(n - m.len, .monotonic);
        return false;
    }
    fn remap(p: *anyopaque, m: []u8, a: std.mem.Alignment, n: usize, ra: usize) ?[*]u8 {
        const q: *Quota = @ptrCast(@alignCast(p));
        if (n > m.len and !q.take(n - m.len)) return null;
        if (q.parent.rawRemap(m, a, n, ra)) |r| {
            if (n < m.len) _ = q.used.fetchSub(m.len - n, .monotonic);
            return r;
        }
        if (n > m.len) _ = q.used.fetchSub(n - m.len, .monotonic);
        return null;
    }
    fn free(p: *anyopaque, m: []u8, a: std.mem.Alignment, ra: usize) void {
        const q: *Quota = @ptrCast(@alignCast(p));
        q.parent.rawFree(m, a, ra);
        _ = q.used.fetchSub(m.len, .monotonic);
    }
};

pub const Disk = struct {
    gpa: std.mem.Allocator,
    copy_gpa: std.mem.Allocator, // the engine's: Copy values the cache's drop destroys with it
    device: Device,
    layout: Layout,
    payload: u64,
    active_payload: u64 = 0,
    image: *const pc.rows.Image = &@import("cuda_disk.zig").image,
    drop: *const fn (*anyopaque, pc.Saved) void = N.snap_drop,
    tier: *spill.Tier,
    quota: *Quota,
    dir: []u8, // the tier's settings directory (owned)
    slot: []u8, // the whole state and head, captured on the owner thread
    reservation: u64, // host bytes charged against the cache's budget (spare)
    job: ?*spill.Job = null, // the write reading the slot (until copied), then its outcome until done
    retired: [retired_most]?*spill.Job = @splat(null), // copied writes waiting for their outcome
    counts: Counts = .{},
    flush_s: f64,
    target: ?*N.Target = null, // the Target `attach` bound (unbound at close)
    flush: ?Flush = null, // set only inside `close`

    /// The host reservation for a layout's payload and a tier's chunk: slot, transfer buffers, control.
    pub fn reservationFor(payload: u64, chunk: usize) !u64 {
        return std.math.add(u64, payload, tierQuota(chunk, control_bytes)) catch error.BadLayout;
    }

    /// The reservation for `payload` when both budgets hold it: the configured host ceiling the reservation, and the
    /// prompt cache (which carries it as its family's spare) the reservation and one kept state besides. One charge:
    /// native admission sizes the cache from what the streams leave and subtracts nothing more.
    pub fn admit(payload: u64, device_bytes: u64, host_ceiling: u64, cache: u64) !u64 {
        const r = try reservationFor(payload, 64 << 20);
        if (host_ceiling < r) return error.SpillHostCeiling;
        if (cache < r +| device_bytes) return error.SpillCacheTooSmall;
        return r;
    }

    /// The tier under `settings` for the loaded engine `identity`, and its slot, within `host_ceiling` bytes.
    /// `settings.dir` passes to the Disk. Fails before any allocation it cannot account.
    pub fn open(gpa: std.mem.Allocator, io: std.Io, copy_gpa: std.mem.Allocator, device: Device, layout: Layout, settings: spill.Settings, identity: [32]u8, host_ceiling: u64) !*Disk {
        const payload = try layout.payload();
        const reservation = try reservationFor(payload, 64 << 20);
        if (reservation > host_ceiling) return error.SpillHostCeiling;
        const d = try gpa.create(Disk);
        errdefer gpa.destroy(d);
        const q = try gpa.create(Quota);
        errdefer gpa.destroy(q);
        q.* = .{ .parent = gpa, .limit = tierQuota(64 << 20, control_bytes) };
        const slot = try gpa.alloc(u8, @intCast(payload));
        errdefer gpa.free(slot);
        const t = try spill.Tier.open(q.allocator(), io, settings, "nemotron_h", identity, 0, 1);
        d.* = .{ .gpa = gpa, .copy_gpa = copy_gpa, .device = device, .layout = layout, .payload = payload, .tier = t, .quota = q, .dir = @constCast(settings.dir), .slot = slot, .reservation = reservation, .flush_s = settings.flush_s };
        return d;
    }

    /// Binds `d` to the family's Target and the Store; the Store must stay where it is until `close`. Called on the
    /// owner thread before the lane host's thread starts (that thread then owns the Store, the Target and `d`).
    pub fn attach(d: *Disk, target: *N.Target, store: *pc.Store) void {
        target.disk = d;
        store.attach(d.tier);
        d.target = target;
    }

    /// Outcomes of finished writes, references released; the slot freed for reuse once its write copied it.
    pub fn reap(d: *Disk) void {
        for (&d.retired) |*r| if (r.*) |j| if (j.done()) {
            d.settle(j);
            r.* = null;
        };
        const j = d.job orelse return;
        if (j.done()) {
            d.settle(j);
            d.job = null;
        } else if (j.copied()) {
            for (&d.retired) |*r| if (r.* == null) {
                r.* = j;
                d.job = null;
                return;
            };
        }
    }

    fn settle(d: *Disk, j: *spill.Job) void {
        if (j.ok) d.counts.written += 1 else {
            d.counts.lost += 1;
            if (d.flush) |*f| f.stopped = true; // the tier could not take a state: the stop asks for no more
        }
        j.release();
    }

    /// Every submitted write final, every reference released (before the tier's shutdown).
    pub fn settleAll(d: *Disk) void {
        if (d.job) |j| {
            j.waitDone();
            d.settle(j);
            d.job = null;
        }
        for (&d.retired) |*r| if (r.*) |j| {
            j.waitDone();
            d.settle(j);
            r.* = null;
        };
    }

    /// The slot's bytes to the tier's writer (its thread): bounds-checked copies of immutable host memory.
    fn sourceRead(p: *anyopaque, offset: u64, out: []u8) anyerror!void {
        const d: *Disk = @ptrCast(@alignCast(p));
        if (offset > d.active_payload or out.len > d.active_payload - offset) return error.SpillSourceRange;
        @memcpy(out, d.slot[@intCast(offset)..][0..out.len]);
    }

    fn check(d: *const Disk, c: *const N.Copy, name: spill.Name) !void {
        try checkPosition(d.layout, c.pos, name.at);
        if (!c.whole) return error.Incomplete;
        if (c.kv.len < d.layout.kvBytes(c.pos) or c.mamba.len < d.layout.mamba or c.head != d.layout.head) return error.SpillLayout;
        if (c.parity != 0 or c.rows > N.max_rows) return error.SpillScalars;
    }

    /// Store's spill_write: the state captured into the slot, then queued; a busy slot is a counted skip.
    fn write(d: *Disk, target: *N.Target, saved: pc.Saved, tier: *spill.Tier, name: spill.Name, tokens: []const u32) !void {
        if (tier != d.tier) return error.SpillTier;
        if (d.flush) |*f| {
            // a clean stop: past the deadline or after a failure nothing new is submitted; else the slot is waited for
            if (!f.stopped and std.Io.Clock.awake.now(tier.io).toNanoseconds() < f.end) {
                if (d.job) |j| j.waitCopied();
                d.reap();
                if (d.job != null) d.settleAll(); // the retired outcomes are full: take them all
            }
            if (f.stopped or std.Io.Clock.awake.now(tier.io).toNanoseconds() >= f.end) {
                f.left += 1;
                return;
            }
        } else {
            d.reap();
            if (d.job != null) {
                d.counts.skipped_busy += 1;
                tier.unwritten(.incomplete);
                return;
            }
        }
        const c: *const N.Copy = @ptrCast(@alignCast(saved));
        if (tokens.len != @as(usize, name.at) + lookahead) return error.SpillTokens;
        try d.check(c, name);
        d.capture(target, c, name, tokens) catch |err| {
            if (d.flush) |*f| {
                f.stopped = true; // a refused write or a failed capture would fail again for the next state
                f.failed += 1;
            }
            return err;
        };
    }

    /// Reuse the upstream image: words, compact KV, Mamba. Its gather fences queued copies.
    fn capture(d: *Disk, target: *N.Target, c: *const N.Copy, name: spill.Name, tokens: []const u32) !void {
        const tier = d.tier;
        const t0 = std.Io.Clock.awake.now(tier.io).toNanoseconds();
        try d.device.ctx.makeCurrent();
        try d.device.stream.synchronize();
        const bytes = try d.layout.payloadAt(c.pos);
        const ranges = d.layout.ranges(c.pos);
        d.image.gather(target, @ptrCast(@constCast(c)), &ranges, d.slot[0..@intCast(bytes)]) catch |err| {
            d.device.stream.synchronize() catch {};
            return err;
        };
        @memset(d.slot[8..16], 0); // a previous window's row count is rewritten before any reader
        d.counts.capture_ns += @intCast(@max(0, std.Io.Clock.awake.now(tier.io).toNanoseconds() - t0));
        var meta: [meta_len]u8 = undefined;
        encode(d.layout, .{ .pos = c.pos }, &meta);
        d.active_payload = bytes;
        d.job = try tier.write(name, tokens, &meta, bytes, .{ .ptr = d, .read = sourceRead });
        d.counts.submitted += 1;
    }

    fn read(d: *Disk, target: *N.Target, tier: *spill.Tier, name: spill.Name) !pc.Saved {
        if (tier != d.tier) return error.SpillTier;
        d.reap();
        const c = d.readCopy(target, tier, name) catch |err| {
            if (err != error.NotStored) d.counts.read_failed += 1;
            return err;
        };
        d.counts.read_ok += 1;
        return c;
    }

    fn readCopy(d: *Disk, target: *N.Target, tier: *spill.Tier, name: spill.Name) !pc.Saved {
        var r = try tier.read(name);
        defer r.close();
        const scalar = try decode(d.layout, r.meta(), name);
        if (@as(u64, r.header.tokens) != @as(u64, name.at) + lookahead) return error.SpillTokens;
        if (r.payloadLen() != try d.layout.payloadAt(scalar.pos)) return error.SpillLayout;
        try d.device.ctx.makeCurrent();
        const saved = try d.image.blank(target, name.at);
        errdefer d.drop(target, saved);
        errdefer d.device.stream.synchronize() catch {};
        var sink: Upload = .{ .disk = d, .target = target, .saved = saved, .at = name.at };
        try r.pump(.{ .ptr = &sink, .write = Upload.write });
        const c: *const N.Copy = @ptrCast(@alignCast(saved));
        try d.check(c, name);
        if (c.rows != 0) return error.SpillScalars;
        return saved;
    }

    /// A Tier chunk may cross image boundaries. Upstream scatter settles each reusable host chunk.
    const Upload = struct {
        disk: *Disk,
        target: *N.Target,
        saved: pc.Saved,
        at: u32,
        fn write(p: *anyopaque, offset: u64, bytes: []const u8) anyerror!void {
            const u: *Upload = @ptrCast(@alignCast(p));
            const total = try u.disk.layout.payloadAt(u.at);
            if (offset > total or bytes.len > total - offset) return error.SpillSourceRange;
            for (u.disk.layout.ranges(u.at)) |part| {
                const lo = @max(offset, part.off);
                const hi = @min(offset + bytes.len, part.off + part.len);
                if (lo >= hi) continue;
                u.disk.image.scatter(u.target, u.saved, &.{.{ .off = lo, .len = hi - lo }}, bytes[@intCast(lo - offset)..@intCast(hi - offset)]) catch |err| {
                    u.disk.device.stream.synchronize() catch {};
                    return err;
                };
            }
        }
    };

    /// The clean stop, on the owner thread after the lane thread joined: one flush pass offers every kept state not on
    /// disk (most recently used first), each write waiting for the slot, none submitted after TENSORFOLD_SPILL_FLUSH_S
    /// or after a write failed; then the Store detached, every write final before the tier stops, and the tier, the
    /// slot and the settings freed. A failed context bind skips the flush; the rest still runs (error.SpillContext).
    /// The caller frees the Store and the device copies after this returns.
    pub fn close(d: *Disk, store: *pc.Store) !Flushed {
        var out: Flushed = .{};
        const bound = if (d.device.ctx.makeCurrent()) |_| true else |_| false;
        if (bound) {
            d.settleAll(); // serving's writes final first: their outcomes are not the stop's
            const written = d.counts.written;
            const lost = d.counts.lost;
            const now = std.Io.Clock.awake.now(d.tier.io).toNanoseconds();
            d.flush = .{ .end = now + @as(i96, @intFromFloat(d.flush_s * 1e9)) };
            _ = store.flushSpill(d.flush_s);
            d.settleAll();
            out.written = d.counts.written - written;
            out.lost = d.counts.lost - lost;
            out.failed = d.flush.?.failed;
            out.left = d.flush.?.left;
            d.flush = null;
        }
        store.detach();
        if (d.target) |t| t.disk = null;
        d.settleAll();
        out.counts = d.counts;
        d.deinit();
        if (!bound) return error.SpillContext;
        return out;
    }

    /// Frees the tier (its writer stopped), the slot and the settings; every job must be settled first.
    fn deinit(d: *Disk) void {
        std.debug.assert(d.job == null);
        for (d.retired) |r| std.debug.assert(r == null);
        d.tier.deinit();
        const gpa = d.gpa;
        gpa.free(d.slot);
        gpa.free(d.dir);
        gpa.destroy(d.quota);
        gpa.destroy(d);
    }

    /// Startup unwind before the lane thread ran: the Store detached, nothing written.
    pub fn abandon(d: *Disk, store: ?*pc.Store) void {
        if (store) |s| s.detach();
        if (d.target) |t| t.disk = null;
        d.settleAll();
        d.deinit();
    }
};

// ---- the family's Snapshots slots ------------------------------------------------------------------------------------

fn diskOf(p: *anyopaque) *Disk {
    return maybeDisk(p).?;
}

fn maybeDisk(p: *anyopaque) ?*Disk {
    const t: *N.Target = @ptrCast(@alignCast(p));
    return @ptrCast(@alignCast(t.disk orelse return null));
}

pub fn spillWrite(p: *anyopaque, saved: pc.Saved, tier: *spill.Tier, name: spill.Name, tokens: []const u32) anyerror!void {
    return diskOf(p).write(@ptrCast(@alignCast(p)), saved, tier, name, tokens);
}

pub fn spillRead(p: *anyopaque, tier: *spill.Tier, name: spill.Name) anyerror!pc.Saved {
    return diskOf(p).read(@ptrCast(@alignCast(p)), tier, name);
}

/// The host reservation, inside the cache's budget for as long as the tier is open.
pub fn spare(p: *anyopaque) u64 {
    return if (maybeDisk(p)) |d| d.reservation else 0;
}

/// The reservation cannot shrink; finished writes are reaped.
pub fn trim(p: *anyopaque, room: u64) void {
    _ = room;
    if (maybeDisk(p)) |d| d.reap();
}

test {
    _ = @import("nemotron_spill_test.zig");
    _ = @import("nemotron_spill_close_test.zig");
}
