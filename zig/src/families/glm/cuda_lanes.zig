//! GLM-5.3-Flash's CUDA lane backend (as nemotron/cuda_lanes.zig): a slot per stream, shared verify, chunked prompts.

const std = @import("std");
const lanes = @import("lanes");
const engine = @import("cuda_engine.zig");
const follow = @import("cuda_follow.zig");
const fw = @import("cuda_forward.zig");
const vision_runtime = @import("cuda_vision_runtime.zig");

const be = lanes.backend;

/// First tokens a handle names.
const ring = 1024;

/// The widest verify window: the pending token and a full draft chain (glm-decode-check's, the AOT row variants).
pub const max_rows = 1 + engine.max_drafts;

/// A stream's place in the backend: its sequence's slot, the rows its last verify left uncommitted, its held drafts.
const Lane = struct {
    slot: u32,
    pending_rows: ?usize = null,
    held: [engine.max_drafts]u32 = undefined,
    held_n: usize = 0,
    fill: ?usize = null, // a prompt pass going a chunk at a time: where its next chunk starts
};

pub const Lanes = struct {
    x: follow.Leader,
    vision: ?*vision_runtime.Service = null,
    whole: bool = false, // prompt passes run whole (TF_GLM_PROMPT_STEPS=0): no other stream's round between chunks
    alone: bool = false, // each window in its own forward (TF_GLM_SHARED=0)
    shared: u64 = 0, // forwards that held several streams' windows (checks read it)
    lanes: std.AutoHashMapUnmanaged(*const lanes.Stream, Lane) = .empty,
    drawn: [ring]u32 = undefined,
    next: u64 = 0,

    pub fn deinit(self: *Lanes) void {
        self.lanes.deinit(self.x.e.gpa);
    }

    pub fn backend(self: *Lanes) be.Backend {
        return .{ .ptr = self, .vtable = if (self.whole) &whole_vt else &steps_vt };
    }

    const whole_vt: be.Backend.VTable = .{
        .prefill = prefillFn,
        .first = firstFn,
        .first_masked = firstMaskedFn,
        .held = heldFn,
        .queue = queueFn,
        .read = readFn,
        .verify = verifyFn,
        .keep = keepFn,
        .draft = draftFn,
        .tree = draftShapeFn,
        .release = releaseFn,
    };
    const steps_vt: be.Backend.VTable = vt: {
        var v = whole_vt;
        v.prefill_step = prefillStepFn;
        break :vt v;
    };

    /// Round loop facts: windows up to max_rows, drafts with the head loaded, shared windows unless TF_GLM_SHARED=0.
    pub fn facts(self: *const Lanes) lanes.Model {
        const drafting = self.x.e.drafting();
        const sharing = drafting and !self.alone;
        return .{
            .exact_width = if (drafting) max_rows else 1,
            .gpu_tokens = false,
            .mtp = drafting,
            .speculate = drafting,
            .speculate_early = false,
            .drafts = engine.max_drafts,
            .hidden_rows = sharing,
            .batch_rows = if (sharing) engine.shared_rows else 0,
            .max_streams = if (sharing) engine.max_shared else 1,
            .draft_streams = sharing and self.x.e.dflash != null,
        };
    }

    fn of(ptr: *anyopaque) *Lanes {
        return @ptrCast(@alignCast(ptr));
    }

    fn take(self: *Lanes, token: u32) u64 {
        const h = self.next;
        self.drawn[h % ring] = token;
        self.next += 1;
        return h;
    }

    fn value(self: *const Lanes, feed: be.Feed) u32 {
        return switch (feed) {
            .handle => |h| self.drawn[h % ring],
            .value => |v| v,
        };
    }

    /// The lowest slot no stream holds.
    fn freeSlot(self: *const Lanes) u32 {
        var slot: u32 = 0;
        while (true) : (slot += 1) {
            var it = self.lanes.valueIterator();
            const taken = while (it.next()) |l| {
                if (l.slot == slot) break true;
            } else false;
            if (!taken) return slot;
        }
    }

    /// The stream's sequence bound, its last verify committed (`kept` rows, else all of them).
    fn bind(self: *Lanes, s: *const lanes.Stream, kept: ?usize) !*Lane {
        const l = self.lanes.getPtr(s) orelse return error.UnknownStream;
        try self.x.bind(l.slot);
        if (l.pending_rows) |rows| try self.x.keep(kept orelse rows);
        l.pending_rows = null;
        return l;
    }

    fn cancelled(ptr: ?*anyopaque) bool {
        return @as(*lanes.Stream, @ptrCast(@alignCast(ptr.?))).isCancelled();
    }

    // -- the vtable ---------------------------------------------------------------------------------------------

    fn prefillFn(ptr: *anyopaque, s: *lanes.Stream) anyerror!void {
        const self = of(ptr);
        const from = try self.opening(s);
        _ = self.take(try self.x.prefill(s.prompt(), from, .{ .ptr = s, .check = cancelled }, marksOf(s)));
    }

    /// One chunk of the stream's prompt pass (fillOne runs other rounds between); true once the first token is drawn.
    fn prefillStepFn(ptr: *anyopaque, s: *lanes.Stream) anyerror!bool {
        const self = of(ptr);
        const going: ?usize = if (self.lanes.getPtr(s)) |l| l.fill else null;
        const from = if (going == null) try self.opening(s) else null;
        const l = self.lanes.getPtr(s).?;
        if (going != null) try self.x.bind(l.slot);
        const at = going orelse if (from) |r| r.at else 0;
        errdefer l.fill = null;
        if (try self.x.chunk(s.prompt(), at, from, .{ .ptr = s, .check = cancelled }, marksOf(s))) |first| {
            _ = self.take(first);
            l.fill = null;
            return true;
        }
        l.fill = at + engine.prompt_rows;
        return false;
    }

    /// A prompt pass begins: the stream's slot bound and its sequence fresh; where the pass resumes (a kept state).
    fn opening(self: *Lanes, s: *lanes.Stream) !?engine.Resume {
        const ids = s.prompt();
        if (ids.len + s.max_new + max_rows >= self.x.e.capacity) return error.PromptTooLong;
        const free_slot = self.freeSlot();
        const gop = try self.lanes.getOrPut(self.x.e.gpa, s);
        if (!gop.found_existing) gop.value_ptr.* = .{ .slot = free_slot };
        gop.value_ptr.pending_rows = null;
        gop.value_ptr.held_n = 0;
        gop.value_ptr.fill = null;
        try self.x.bind(gop.value_ptr.slot);
        if (self.x.e.pool != null) {
            const rows = try @import("cuda_kv_extent.zig").requestRows(ids.len, s.max_new, fw.window_rows, self.x.e.capacity);
            try self.x.reserve(rows);
        }
        try self.x.begin(s.sampling);
        if (s.vision) |picture| {
            const service = self.vision orelse return error.VisionUnavailable;
            try self.x.vision(service, picture.encoded, ids, picture.image_token, picture.replica);
        }
        s.cached = 0;
        return self.resumeFrom(s, ids.len);
    }

    fn marksOf(s: *lanes.Stream) ?engine.Marks {
        return if (s.reuse.marks.len == 0) null else .{ .at = s.reuse.marks, .ctx = s, .call = keptAt };
    }

    /// The stream's kept state, where its prompt pass starts (null: none, off the chunk grid, or no longer held).
    fn resumeFrom(self: *Lanes, s: *lanes.Stream, len: usize) ?engine.Resume {
        const saved = s.reuse.saved orelse return null;
        const k: *const engine.Kept = @ptrCast(@alignCast(saved));
        if (k.at != s.reuse.at or k.at == 0 or k.at >= len or k.at % fw.prefill_rows != 0 or !self.x.e.holds(k.id)) {
            s.reuse_failed = true;
            return null;
        }
        s.cached = k.at;
        return .{ .id = k.id, .at = k.at };
    }

    /// The prompt pass stands at a mark: the lane host keeps the state there (cuda_native's snap_save).
    fn keptAt(ctx: *anyopaque, at: u32) void {
        const s: *lanes.Stream = @ptrCast(@alignCast(ctx));
        if (s.reuse.hook) |k| k.at(k.ptr, s, at);
    }

    fn firstFn(ptr: *anyopaque, s: *lanes.Stream, position: u64) anyerror!u64 {
        const self = of(ptr);
        if (position != s.prompt_len) return error.PositionMismatch;
        return self.next - 1;
    }

    fn firstMaskedFn(ptr: *anyopaque, s: *lanes.Stream, position: u64, bits: []const u32) anyerror!u64 {
        const self = of(ptr);
        if (position != s.prompt_len) return error.PositionMismatch;
        _ = try self.bind(s, null);
        return self.take(try self.x.firstMasked(position, bits));
    }

    fn heldFn(ptr: *anyopaque, s: *lanes.Stream, out: []u32) anyerror!void {
        const l = of(ptr).lanes.getPtr(s) orelse return error.UnknownStream;
        if (out.len > l.held_n) return error.NoHeldDrafts;
        @memcpy(out, l.held[0..out.len]);
    }

    fn draftShapeFn(ptr: *anyopaque, s: *lanes.Stream, _: std.mem.Allocator) anyerror!?lanes.stream.Held {
        const self = of(ptr);
        if (self.x.e.dflash == null) return null;
        const lane = self.lanes.getPtr(s) orelse return error.UnknownStream;
        return .{ .count = @intCast(lane.held_n) };
    }

    fn queueFn(ptr: *anyopaque, s: *lanes.Stream, feed: be.Feed, position: u64) anyerror!u64 {
        _ = ptr;
        _ = s;
        _ = feed;
        _ = position;
        return error.NotPipelined;
    }

    fn readFn(ptr: *anyopaque, handle: u64) anyerror!u32 {
        const self = of(ptr);
        if (handle >= self.next or self.next - handle > ring) return error.NoSuchToken;
        return self.drawn[handle % ring];
    }

    /// Each window in its own forward on its stream's sequence (the core sends one when no forward is shared).
    fn verifyFn(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) anyerror!void {
        const self = of(ptr);
        const constrained = for (windows) |w| {
            if (w.masks.len > 0) break true;
        } else false;
        if (windows.len > 1 and !constrained) return self.verifyShared(windows, out);
        for (windows, out) |w, *o| {
            if (w.parents != null) return error.TreesNotBuilt;
            const l = try self.bind(w.stream, null);
            const rows = w.rows();
            if (rows > max_rows) return error.WindowTooWide;
            if (w.held > l.held_n) return error.NoHeldDrafts;
            const pos = self.x.e.pos();
            for (w.positions, 0..) |p, r| if (p != pos + 1 + r) return error.PositionMismatch;
            var ids: [max_rows]u32 = undefined;
            ids[0] = w.pending;
            const drafts = if (w.held > 0) l.held[0..w.held] else w.tokens;
            @memcpy(ids[1..][0..drafts.len], drafts);
            var sampled: [max_rows]u32 = undefined;
            if (w.masks.len > 0) try self.x.verifyMasked(ids[0..rows], sampled[0..rows], w.masks) else try self.x.verify(ids[0..rows], sampled[0..rows]);
            @memcpy(o.sampled[0..rows], sampled[0..rows]);
            @memcpy(o.drafts[0..drafts.len], drafts);
            l.pending_rows = rows;
        }
    }

    /// A round's windows in one forward (Engine.verifyShared), each stream's rows on its own sequence.
    fn verifyShared(self: *Lanes, windows: []const be.Window, out: []be.Verified) !void {
        if (windows.len > engine.max_shared) return error.WindowTooWide;
        var parts: [engine.max_shared]engine.Part = undefined;
        var ids: [engine.max_shared][max_rows]u32 = undefined;
        var sampled: [engine.max_shared][max_rows]u32 = undefined;
        var lane: [engine.max_shared]*Lane = undefined;
        for (windows, 0..) |w, k| {
            if (w.parents != null) return error.TreesNotBuilt;
            const l = try self.bind(w.stream, null);
            const rows = w.rows();
            if (rows > max_rows) return error.WindowTooWide;
            if (w.held > l.held_n) return error.NoHeldDrafts;
            const pos = self.x.e.pos();
            for (w.positions, 0..) |p, r| if (p != pos + 1 + r) return error.PositionMismatch;
            ids[k][0] = w.pending;
            const drafts = if (w.held > 0) l.held[0..w.held] else w.tokens;
            @memcpy(ids[k][1..][0..drafts.len], drafts);
            parts[k] = .{ .slot = l.slot, .ids = ids[k][0..rows], .out = sampled[k][0..rows] };
            lane[k] = l;
        }
        try self.x.verifyShared(parts[0..windows.len]);
        self.shared += 1;
        for (out[0..windows.len], parts[0..windows.len], lane[0..windows.len]) |*o, p, l| {
            const rows = p.ids.len;
            @memcpy(o.sampled[0..rows], p.out);
            @memcpy(o.drafts[0 .. rows - 1], p.ids[1..rows]);
            l.pending_rows = rows;
        }
    }

    fn keepFn(ptr: *anyopaque, windows: []const be.Window, paths: []const []const u32) anyerror!void {
        const self = of(ptr);
        for (windows, paths) |w, path| {
            if (path.len == 0) return error.EmptyPath;
            for (path, 0..) |r, i| if (r != i) return error.TreesNotBuilt;
            _ = try self.bind(w.stream, path.len);
        }
    }

    fn draftFn(ptr: *anyopaque, requests: []const be.DraftRequest) anyerror!void {
        const self = of(ptr);
        if (!self.x.e.drafting()) return error.NoDraftHead;
        if (self.x.e.dflash != null) return self.draftBatch(requests);
        for (requests) |r| {
            const l = try self.bind(r.stream, if (r.rows) |rows| rows.len else null);
            if (r.position != self.x.e.pos() + 1) return error.PositionMismatch;
            if (r.lanes != null) return error.TreesNotBuilt;
            const depth = @min(r.depth, engine.max_drafts);
            if (r.rows) |rows| {
                for (rows, 0..) |row, i| if (row != i) return error.TreesNotBuilt;
                l.held_n = try self.x.draft(false, r.follow[0..rows.len], depth, &l.held);
            } else {
                const first = self.value(r.first orelse return error.NoFirstToken);
                l.held_n = try self.x.draft(true, &.{first}, depth, &l.held);
            }
        }
    }

    const DraftInput = struct { pending: u32, keep: ?usize };

    fn preflightDraft(self: *const Lanes, l: *const Lane, r: be.DraftRequest, base: usize) !DraftInput {
        if (r.lanes != null) return error.TreesNotBuilt;
        if (r.depth > engine.max_drafts) return error.BadDraftRequest;
        const base_pos: u64 = @intCast(base);
        var end = r.start;
        var keep: ?usize = null;
        const pending = if (r.rows) |rows| blk: {
            if (rows.len == 0 or rows.len > max_rows or r.follow.len < rows.len) return error.BadDraftRequest;
            for (rows, 0..) |row, j| if (row != j) return error.TreesNotBuilt;
            end = std.math.add(u64, r.start, @as(u64, @intCast(rows.len))) catch return error.PositionMismatch;
            if (l.pending_rows) |available| {
                if (base_pos != r.start or rows.len > available) return error.NoDraftWindow;
                keep = rows.len;
            } else if (base_pos != end) return error.NoDraftWindow;
            break :blk r.follow[rows.len - 1];
        } else blk: {
            if (l.pending_rows != null) return error.NoDraftWindow;
            if (base_pos != r.start) return error.PositionMismatch;
            const feed = r.first orelse return error.NoFirstToken;
            break :blk switch (feed) {
                .value => |token| token,
                .handle => |h| handle: {
                    if (h >= self.next or self.next - h > ring) return error.NoSuchToken;
                    break :handle self.drawn[h % ring];
                },
            };
        };
        if (pending >= self.x.e.c.vocab) return error.BadDraftRequest;
        if (r.position != (std.math.add(u64, end, 1) catch return error.PositionMismatch)) return error.PositionMismatch;
        return .{ .pending = pending, .keep = keep };
    }

    fn draftBatch(self: *Lanes, requests: []const be.DraftRequest) !void {
        if (requests.len == 0 or requests.len > engine.max_shared) return error.BadDraftBatch;
        var lanes_by_request: [engine.max_shared]*Lane = undefined;
        var pending: [engine.max_shared]u32 = undefined;
        var keeps: [engine.max_shared]?usize = undefined;
        var seen: [engine.max_shared]bool = @splat(false);
        for (requests, 0..) |r, i| {
            if (r.stream.isCancelled()) return error.Cancelled;
            const l = self.lanes.getPtr(r.stream) orelse return error.UnknownStream;
            if (l.slot >= engine.max_shared or seen[l.slot]) return error.BadDraftSlot;
            seen[l.slot] = true;
            const base = try self.x.e.slotPos(l.slot);
            const input = try self.preflightDraft(l, r, base);
            pending[i] = input.pending;
            keeps[i] = input.keep;
            lanes_by_request[i] = l;
        }
        for (requests, lanes_by_request[0..requests.len], keeps[0..requests.len]) |r, l, keep| {
            _ = try self.bind(r.stream, keep);
            if (try self.x.e.slotPos(l.slot) + 1 != r.position) return error.PositionMismatch;
        }
        var parts: [engine.max_shared]engine.DraftPart = undefined;
        var original: [engine.max_shared]usize = undefined;
        var count: usize = 0;
        for (requests, lanes_by_request[0..requests.len], pending[0..requests.len], 0..) |r, l, token, i| {
            if (r.depth == 0) continue;
            parts[count] = .{ .slot = l.slot, .position = r.position, .pending = token, .depth = r.depth };
            original[count] = i;
            count += 1;
        }
        var results: [engine.max_shared]engine.DraftResult = @splat(.{});
        if (count > 0) try self.x.draftBatch(parts[0..count], results[0..count]);
        for (lanes_by_request[0..requests.len]) |l| l.held_n = 0;
        for (results[0..count], original[0..count]) |result, i| {
            if (requests[i].stream.isCancelled()) continue;
            const l = lanes_by_request[i];
            @memcpy(l.held[0..result.count], result.tokens[0..result.count]);
            l.held_n = result.count;
        }
    }

    /// The stream left the rounds: its sequence is freed on every rank (slot 0 stays for the next stream).
    fn releaseFn(ptr: *anyopaque, s: *lanes.Stream) void {
        const self = of(ptr);
        const kv = self.lanes.fetchRemove(s) orelse return;
        self.x.release(kv.value.slot) catch |err| std.log.err("releasing slot {d}: {s}", .{ kv.value.slot, @errorName(err) });
    }
};

comptime {
    std.debug.assert(max_rows <= fw.window_rows);
}

test "DFlash publishes the actual held count through the existing lane shape callback" {
    var e: engine.Engine = undefined;
    e.gpa = std.testing.allocator;
    e.dflash = @ptrFromInt(256);
    var link: follow.Link = undefined;
    var backend: Lanes = .{ .x = .{ .e = &e, .link = &link } };
    defer backend.deinit();
    var stream: lanes.Stream = undefined;
    try backend.lanes.put(e.gpa, &stream, .{ .slot = 0, .held_n = 2 });
    const b = backend.backend();
    try std.testing.expectEqual(@as(u32, 2), (try b.vtable.tree.?(b.ptr, &stream, e.gpa)).?.count);
    backend.lanes.getPtr(&stream).?.held_n = 0;
    try std.testing.expectEqual(@as(u32, 0), (try b.vtable.tree.?(b.ptr, &stream, e.gpa)).?.count);
    e.dflash = null;
    try std.testing.expectEqual(@as(?lanes.stream.Held, null), try b.vtable.tree.?(b.ptr, &stream, e.gpa));
}

test "DFlash facts enable batched head scheduling only while sharing" {
    var e: engine.Engine = undefined;
    e.mtp = false;
    e.dflash = @ptrFromInt(256);
    var link: follow.Link = undefined;
    var backend: Lanes = .{ .x = .{ .e = &e, .link = &link } };
    var cfg = try lanes.Config.init(std.testing.allocator, backend.facts(), max_rows, engine.max_drafts);
    defer cfg.deinit(std.testing.allocator);
    try std.testing.expect(cfg.family_streams and cfg.draft_streams);
    backend.alone = true;
    try std.testing.expect(!backend.facts().draft_streams);
    backend.alone = false;
    e.dflash = null;
    e.mtp = true;
    try std.testing.expect(!backend.facts().draft_streams);
}

test "DFlash draft preflight distinguishes accepted rows already committed by the core" {
    var e: engine.Engine = undefined;
    e.c = .{};
    var link: follow.Link = undefined;
    const backend: Lanes = .{ .x = .{ .e = &e, .link = &link } };
    var stream: lanes.Stream = undefined;
    const request: be.DraftRequest = .{ .stream = &stream, .follow = &.{ 11, 22 }, .rows = &.{ 0, 1 }, .start = 100, .position = 103, .depth = 1 };
    const committed: Lane = .{ .slot = 0 };
    const already = try backend.preflightDraft(&committed, request, 102);
    try std.testing.expectEqual(@as(u32, 22), already.pending);
    try std.testing.expectEqual(@as(?usize, null), already.keep);
    const pending_window: Lane = .{ .slot = 0, .pending_rows = 3 };
    const uncommitted = try backend.preflightDraft(&pending_window, request, 100);
    try std.testing.expectEqual(@as(?usize, 2), uncommitted.keep);
    try std.testing.expectError(error.NoDraftWindow, backend.preflightDraft(&committed, request, 101));
    try std.testing.expectError(error.NoDraftWindow, backend.preflightDraft(&pending_window, request, 102));
}

test "a later invalid mixed draft request refuses before any bind or paired command" {
    var e: engine.Engine = undefined;
    e.gpa = std.testing.allocator;
    e.c = .{};
    e.mtp = false;
    e.dflash = @ptrFromInt(256);
    e.seqs = .empty;
    defer e.seqs.deinit(std.testing.allocator);
    var first_seq: @TypeOf(e.cur.*) = undefined;
    first_seq.st.pos = 10;
    var second_seq: @TypeOf(e.cur.*) = undefined;
    second_seq.st.pos = 20;
    try e.seqs.append(std.testing.allocator, &first_seq);
    try e.seqs.append(std.testing.allocator, &second_seq);
    var link: follow.Link = undefined;
    var backend: Lanes = .{ .x = .{ .e = &e, .link = &link } };
    defer backend.deinit();
    var first: lanes.Stream = undefined;
    first.cancel_check = null;
    var second: lanes.Stream = undefined;
    second.cancel_check = null;
    try backend.lanes.put(std.testing.allocator, &first, .{ .slot = 0, .held_n = 2 });
    try backend.lanes.put(std.testing.allocator, &second, .{ .slot = 1, .pending_rows = 2, .held_n = 1 });
    const b = backend.backend();
    var requests = [_]be.DraftRequest{
        .{ .stream = &first, .follow = &.{}, .first = .{ .value = 1 }, .rows = null, .start = 10, .position = 11, .depth = 1 },
        .{ .stream = &second, .follow = &.{2}, .rows = &.{0}, .start = 20, .position = 22, .depth = 4 },
    };
    try std.testing.expectError(error.BadDraftRequest, b.draft(&requests));
    try std.testing.expectEqual(@as(usize, 10), first_seq.st.pos);
    try std.testing.expectEqual(@as(usize, 20), second_seq.st.pos);
    try std.testing.expectEqual(@as(usize, 2), backend.lanes.getPtr(&first).?.held_n);
    try std.testing.expectEqual(@as(?usize, 2), backend.lanes.getPtr(&second).?.pending_rows);
    requests[1].depth = 1;
    requests[1].position = 23;
    try std.testing.expectError(error.PositionMismatch, b.draft(&requests));
    requests[1].position = 22;
    requests[1].follow = &.{154880};
    try std.testing.expectError(error.BadDraftRequest, b.draft(&requests));
    requests[1].follow = &.{2};
    backend.lanes.getPtr(&second).?.slot = 0;
    try std.testing.expectError(error.BadDraftSlot, b.draft(&requests));
    try std.testing.expectEqual(@as(usize, 10), first_seq.st.pos);
    try std.testing.expectEqual(@as(usize, 20), second_seq.st.pos);
    backend.lanes.getPtr(&second).?.slot = 1;
    first_seq.st.pos = 12;
    requests[0] = .{ .stream = &first, .follow = &.{ 1, 2 }, .rows = &.{ 0, 1 }, .start = 10, .position = 13, .depth = 1 };
    requests[1].depth = 4;
    try std.testing.expectError(error.BadDraftRequest, b.draft(&requests));
    try std.testing.expectEqual(@as(usize, 12), first_seq.st.pos);
    try std.testing.expectEqual(@as(usize, 20), second_seq.st.pos);
    try std.testing.expectEqual(@as(usize, 2), backend.lanes.getPtr(&first).?.held_n);
    try std.testing.expectEqual(@as(?usize, 2), backend.lanes.getPtr(&second).?.pending_rows);
}
