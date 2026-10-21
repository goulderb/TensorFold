//! GLM-5.3-Flash's CUDA lane backend (as nemotron/cuda_lanes.zig): a slot per stream, shared verify, chunked prompts.

const std = @import("std");
const lanes = @import("lanes");
const engine = @import("cuda_engine.zig");
const follow = @import("cuda_follow.zig");
const fw = @import("cuda_forward.zig");
const vision_runtime = @import("cuda_vision_runtime.zig");
const dflash = @import("dflash.zig");
const prompt_lp = @import("cuda_prompt_logprobs.zig");
const cuda = @import("cuda");

const be = lanes.backend;

/// First tokens a handle names.
const ring = 1024;
const max_held = dflash.BLOCK - 1;
const gb10_window_costs = costTable(max_rows, 52.0, 6.8);
const gb10_shared_costs = costTable(engine.shared_rows, 76.0, 4.1);

fn costTable(comptime n: usize, comptime one: f64, comptime row: f64) [n]lanes.config.Cost {
    var c: [n]lanes.config.Cost = undefined;
    for (&c, 0..) |*x, i| x.* = .{ .width = i + 1, .ms = one + row * @as(f64, @floatFromInt(i)) };
    return c;
}

/// Per-lane target storage: copied windows are wider than the seven-draft neural chain.
pub const max_rows = @import("cuda_verify_plan.zig").target_rows;

/// TF_GLM_LONE_COSTS=ONE,ROW: a lone stream's round modeled as ONE ms plus ROW ms a row past the first. Without a
/// table lanes/depth.zig asks a lone stream for three drafts at most; with one it prices every depth the verify plan
/// allows, and the stream's measured rounds replace the model depth by depth. Shared rounds keep positive-probability
/// nodes at a constant cost. Unlike untimed fill, a zero-probability tail does not improve that rate.
pub const LoneCosts = struct {
    window: [max_rows]lanes.config.Cost,
    flat: [engine.shared_rows]lanes.config.Cost,

    pub fn parse(raw: []const u8) !LoneCosts {
        const comma = std.mem.indexOfScalar(u8, raw, ',') orelse return error.BadLoneCosts;
        const one = try positive(raw[0..comma]);
        const row = try positive(raw[comma + 1 ..]);
        var c: LoneCosts = undefined;
        for (&c.window, 0..) |*x, i| x.* = .{ .width = @intCast(i + 1), .ms = one + row * @as(f64, @floatFromInt(i)) };
        for (&c.flat, 0..) |*x, i| x.* = .{ .width = @intCast(i + 1), .ms = 1.0 };
        return c;
    }

    fn positive(raw: []const u8) !f64 {
        const v = std.fmt.parseFloat(f64, raw) catch return error.BadLoneCosts;
        if (!std.math.isFinite(v) or v <= 0.0) return error.BadLoneCosts;
        return v;
    }
};

/// A stream's place in the backend: its sequence's slot, the rows its last verify left uncommitted, its held drafts.
const Lane = struct {
    slot: u32,
    pending_rows: ?usize = null,
    held: [engine.max_drafts]u32 = undefined,
    chances: [engine.max_drafts]f64 = undefined,
    held_n: usize = 0,
    first: ?u64 = null,
    fill: ?usize = null, // a prompt pass going a chunk at a time: where its next chunk starts
};

const PendingDraftBatch = struct {
    parts: [engine.max_shared]engine.DraftPart = undefined,
    count: usize = 0,

    fn retaining(p: PendingDraftBatch, superseded: []const u32) PendingDraftBatch {
        var result: PendingDraftBatch = .{};
        for (p.parts[0..p.count]) |part| {
            if (std.mem.indexOfScalar(u32, superseded, part.slot) != null) continue;
            result.parts[result.count] = part;
            result.count += 1;
        }
        return result;
    }
};

pub const Lanes = struct {
    x: follow.Leader,
    vision: ?*vision_runtime.Service = null,
    whole: bool = false, // prompt passes run whole (TF_GLM_PROMPT_STEPS=0): no other stream's round between chunks
    alone: bool = false, // each window in its own forward (TF_GLM_SHARED=0)
    lone_costs: ?LoneCosts = null, // TF_GLM_LONE_COSTS
    batch_prefill: bool = false,
    prefill_batches: u64 = 0,
    shared: u64 = 0, // forwards that held several streams' windows (checks read it)
    pending_drafts: PendingDraftBatch = .{},
    lanes: std.AutoHashMapUnmanaged(*const lanes.Stream, Lane) = .empty,
    drawn: [ring]u32 = undefined,
    next: u64 = 0,
    bits: ?cuda.HostBuffer = null, // structured output's grammar words (shared_rows rows), device-read, at first use
    bits_dev: u64 = 0,

    pub fn deinit(self: *Lanes) void {
        if (self.bits) |*b| b.free();
        self.lanes.deinit(self.x.e.gpa);
    }

    /// A window's grammar rows (rows of the vocabulary's words) at row `row` of the bits, device-read; 0 for none.
    pub fn putBits(self: *Lanes, row: usize, rows: usize, masks: []const u32) !u64 {
        if (masks.len == 0) return 0;
        try @import("cuda_sampler.zig").validateMasks(masks, rows, self.x.e.c.vocab);
        const words = (self.x.e.c.vocab + 31) / 32;
        if (masks.len != rows * words or row + rows > engine.shared_rows) return error.InvalidGrammarMasks;
        if (self.bits == null) {
            var buf = try cuda.HostBuffer.allocMapped(self.x.e.d, engine.shared_rows * words * 4);
            errdefer buf.free();
            self.bits_dev = try buf.device();
            self.bits = buf;
        }
        @memcpy(self.bits.?.slice(u32)[row * words ..][0..masks.len], masks);
        return self.bits_dev + row * words * 4;
    }

    pub fn backend(self: *Lanes) be.Backend {
        return .{ .ptr = self, .deferred_draft_shape = self.x.e.settings.draft_prelaunch and (if (self.x.e.dflash) |d| d.asyncAvailable() else false), .vtable = if (self.whole) &whole_vt else if (self.batch_prefill) &batch_vt else &steps_vt };
    }

    const whole_vt: be.Backend.VTable = .{
        .prefill = prefillFn,
        .prefill_many = prefillManyFn,
        .first = firstFn,
        .first_masked = firstMaskedFn,
        .held = heldFn,
        .queue = queueFn,
        .read = readFn,
        .verify = verifyFn,
        .keep = keepFn,
        .draft = draftFn,
        .tree = draftShapeFn,
        .probabilities = chancesFn,
        .release = releaseFn,
        .prompt_rows = true,
    };
    const steps_vt: be.Backend.VTable = vt: {
        var v = whole_vt;
        v.prefill_step = prefillStepFn;
        v.prefill_many = null;
        v.many_rows = 0;
        break :vt v;
    };
    const batch_vt: be.Backend.VTable = vt: {
        var v = steps_vt;
        v.prefill_batch = prefillBatchFn;
        break :vt v;
    };

    /// Round loop facts: windows up to max_rows, drafts with the head loaded, shared windows unless TF_GLM_SHARED=0.
    pub fn facts(self: *const Lanes) lanes.Model {
        const drafting = self.x.e.drafting();
        const sharing = drafting and !self.alone;
        const block = self.x.e.drafter != null;
        const plan = self.x.e.verifyPlan();
        const profile = self.x.e.settings.draft_profile;
        const costs: ?*const LoneCosts = if (drafting) if (self.lone_costs) |*c| c else null else null;
        return .{
            .copy_drafts = if (profile.enabled) profile.copy.enabled else true,
            .window_costs = if (costs) |c| c.window[0 .. 1 + plan.drafts()] else if (block) &gb10_window_costs else &.{},
            .shared_costs = if (costs) |c| &c.flat else if (block) &gb10_shared_costs else &.{},
            .exact_width = if (profile.enabled) @intCast(self.x.e.targetRows()) else if (drafting) @intCast(1 + plan.drafts()) else 1,
            .gpu_tokens = false,
            .mtp = drafting,
            .speculate = drafting,
            .speculate_early = false,
            .drafts = @intCast(plan.drafts()),
            .fixed_drafts = profile.enabled,
            .hidden_rows = sharing,
            .batch_rows = if (sharing) @intCast(self.x.e.sharedLimit()) else 0,
            .max_streams = if (sharing) engine.max_shared else 1,
            .draft_streams = sharing and (self.x.e.dflash != null or block),
            .draft_probabilities = block,
            .mtp_step_ms = if (block) 1.6 / @as(f64, max_held) else 0,
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
        return self.bindUsing(s, kept, self.x);
    }

    fn bindUsing(self: *Lanes, s: *const lanes.Stream, kept: ?usize, x: anytype) !*Lane {
        const l = self.lanes.getPtr(s) orelse return error.UnknownStream;
        try x.bind(l.slot);
        if (l.pending_rows) |rows| try x.keep(kept orelse rows);
        l.pending_rows = null;
        return l;
    }

    fn cancelled(ptr: ?*anyopaque) bool {
        return @as(*lanes.Stream, @ptrCast(@alignCast(ptr.?))).isCancelled();
    }

    pub fn claim(self: *Lanes, s: *const lanes.Stream) !*Lane {
        return self.claimUsing(s, freeSlot);
    }

    pub fn claimUsing(self: *Lanes, s: *const lanes.Stream, comptime choose: anytype) !*Lane {
        if (self.lanes.getPtr(s)) |existing| return existing;
        const slot = choose(self);
        const gop = try self.lanes.getOrPut(self.x.e.gpa, s);
        gop.value_ptr.* = .{ .slot = slot };
        return gop.value_ptr;
    }

    fn draftBlock(self: *Lanes, requests: []const be.DraftRequest, x: anytype) !void {
        // draftFn has validated the entire cohort before any bind, keep, flush or paired command.
        var reqs: [dflash.MAX_S]dflash.Req = undefined;
        var held: [dflash.MAX_S]*Lane = undefined;
        var original: [dflash.MAX_S]usize = undefined;
        var count: usize = 0;
        for (requests, 0..) |r, i| {
            const l = self.lanes.getPtr(r.stream).?;
            const input = try self.preflightDraft(l, r, try x.e.slotPos(l.slot));
            if (r.stream.isCancelled()) {
                l.held_n = 0;
                continue;
            }
            _ = try self.bindUsing(r.stream, input.keep, x);
            l.held_n = 0;
            // Zero depth still ingests accepted target taps; only proposal generation is suppressed.
            reqs[count] = .{ .slot = l.slot, .pos = 0, .n = if (r.rows) |rows| rows.len else 0, .anchor = input.pending, .propose = r.depth != 0, .out = &l.held, .probs = &l.chances };
            held[count] = l;
            original[count] = i;
            count += 1;
        }
        if (count == 0) return;
        try x.block(reqs[0..count]);
        const cut = confidence();
        for (original[0..count], held[0..count]) |i, l| {
            const r = requests[i];
            if (r.stream.isCancelled()) continue;
            l.held_n = r.depth;
            var chain: f64 = 1; // glm_cuda's rule: past the first, a chain stops where its product falls under the cut
            for (l.chances[0..l.held_n], 0..) |*c, j| {
                chain *= c.*;
                if (j > 0 and chain < cut) c.* = 0; // held with chance 0: the window sizing leaves it out
                if (c.* == 0) chain = 0;
            }
        }
    }

    fn confidence() f64 {
        const v = @import("sys").getenv("TF_GLM_DRAFT_CONFIDENCE") orelse return 0.7;
        return std.fmt.parseFloat(f64, v) catch 0.7;
    }

    fn chancesFn(ptr: *anyopaque, s: *lanes.Stream, out: []f64) anyerror!bool {
        const self = of(ptr);
        if (self.x.e.drafter == null) return false;
        const l = self.lanes.getPtr(s) orelse return false;
        if (l.held_n == 0) return false;
        const n = @min(out.len, l.held_n);
        var chain: f64 = 1; // a draft lands only if every one before it does: the selector's chances chained
        for (out[0..n], l.chances[0..n]) |*c, p| {
            chain *= p;
            c.* = chain;
        }
        for (out[n..]) |*c| c.* = 0;
        return true;
    }

    // -- the vtable ---------------------------------------------------------------------------------------------

    fn prefillFn(ptr: *anyopaque, s: *lanes.Stream) anyerror!void {
        const self = of(ptr);
        const from = try self.opening(s);
        try self.images(s, from);
        var sink: Sink = .{ .gpa = self.x.e.gpa, .s = s };
        try sink.prepare();
        self.x.e.prompt_logprobs = sink.want();
        defer self.x.e.prompt_logprobs = null;
        const first = try self.x.prefill(s.prompt(), from, .{ .ptr = s, .check = cancelled }, marksOf(s));
        self.lanes.getPtr(s).?.first = self.take(first);
    }

    fn prefillManyFn(ptr: *anyopaque, ss: []const *lanes.Stream) anyerror!bool {
        const self = of(ptr);
        if (!self.x.e.m.tiled or ss.len > engine.max_together) return false;
        for (ss) |s| if (!canBatchWhole(s) or self.lanes.contains(s) or s.vision != null) return false;
        var opened: usize = 0;
        errdefer for (ss[0..opened]) |s| releaseFn(ptr, s);
        var parts: [engine.max_together]engine.Prompt = undefined;
        var firsts: [engine.max_together]u32 = undefined;
        var lo: usize = 0;
        var rows: usize = 0;
        for (ss, 0..) |s, i| {
            if (rows + s.prompt_len > engine.prompt_rows) { // this pass is full: run it, the next starts here
                try self.many(ss[lo..i], parts[lo..i], firsts[lo..i]);
                lo = i;
                rows = 0;
            }
            if (try self.opening(s) != null) return error.BadPrompts; // no kept state (checked above)
            opened += 1;
            parts[i] = .{ .slot = self.lanes.getPtr(s).?.slot, .ids = s.prompt() };
            rows += s.prompt_len;
        }
        try self.many(ss[lo..], parts[lo..ss.len], firsts[lo..ss.len]);
        return true;
    }

    fn many(self: *Lanes, ss: []const *lanes.Stream, parts: []const engine.Prompt, firsts: []u32) !void {
        try self.x.prefillMany(parts, firsts);
        for (ss, firsts) |s, t| self.lanes.getPtr(s).?.first = self.take(t);
    }

    /// One chunk of the stream's prompt pass (fillOne runs other rounds between); true once the first token is drawn.
    fn prefillStepFn(ptr: *anyopaque, s: *lanes.Stream) anyerror!bool {
        const self = of(ptr);
        const going: ?usize = if (self.lanes.getPtr(s)) |l| l.fill else null;
        const from = if (going == null) try self.opening(s) else null;
        const l = self.lanes.getPtr(s).?;
        if (going == null) try self.images(s, from);
        if (going != null) try self.x.bind(l.slot);
        const at = going orelse if (from) |r| r.at else 0;
        errdefer l.fill = null;
        var sink: Sink = .{ .gpa = self.x.e.gpa, .s = s };
        try sink.prepare();
        self.x.e.prompt_logprobs = sink.want();
        defer self.x.e.prompt_logprobs = null;
        if (try self.x.chunk(s.prompt(), at, from, .{ .ptr = s, .check = cancelled }, marksOf(s))) |first| {
            l.first = self.take(first);
            l.fill = null;
            return true;
        }
        l.fill = at + engine.prompt_rows;
        return false;
    }

    fn canBatch(self: *const Lanes, s: *const lanes.Stream) bool {
        return !self.x.e.mtp and !self.lanes.contains(s) and s.prompt_len > 0 and s.prompt_len <= engine.prompt_batch_rows and
            s.max_new > 0 and s.vision == null and s.images.len == 0 and s.grammar == null and s.gate == null and
            s.logprobs == null and s.prompt_logprobs == null and s.decode_spans.len == 0 and
            s.reuse.saved == null and s.reuse.at == 0 and s.reuse.marks.len == 0 and s.yield_check == null;
    }

    fn prefillBatchFn(ptr: *anyopaque, streams: []const *lanes.Stream, out: []be.PrefillOutcome) anyerror!void {
        const self = of(ptr);
        if (streams.len != out.len) return error.BadPromptBatch;
        @memset(out, .skipped);
        var eligible: usize = 0;
        var rows: usize = 0;
        for (streams) |s| {
            if (eligible == engine.max_shared or !self.canBatch(s) or s.prompt_len > engine.prompt_rows - rows) break;
            eligible += 1;
            rows += s.prompt_len;
        }
        if (eligible < 2) return;
        var parts: [engine.max_shared]engine.Engine.PromptPart = undefined;
        var original: [engine.max_shared]usize = undefined;
        var count: usize = 0;
        for (streams[0..eligible], 0..) |s, i| {
            if (s.isCancelled()) {
                out[i] = .{ .failed = error.Cancelled };
                continue;
            }
            _ = self.opening(s) catch |err| {
                releaseFn(ptr, s);
                out[i] = .{ .failed = err };
                continue;
            };
            if (s.isCancelled()) {
                releaseFn(ptr, s);
                out[i] = .{ .failed = error.Cancelled };
                continue;
            }
            parts[count] = .{ .slot = self.lanes.getPtr(s).?.slot, .ids = s.prompt() };
            original[count] = i;
            count += 1;
        }
        var first: [engine.max_shared]u32 = undefined;
        if (count > 1) {
            try self.x.prefillShared(parts[0..count], first[0..count]);
            self.prefill_batches += 1;
            var actual_rows: usize = 0;
            for (parts[0..count]) |part| actual_rows += part.ids.len;
            std.log.info("GLM cold prompt batch: {d} streams, {d} rows", .{ count, actual_rows });
        } else if (count == 1) {
            const s = streams[original[0]];
            try self.x.bind(parts[0].slot);
            first[0] = (self.x.chunk(s.prompt(), 0, null, .{ .ptr = s, .check = cancelled }, null) catch |err| {
                releaseFn(ptr, s);
                out[original[0]] = .{ .failed = err };
                return;
            }) orelse return error.BadPromptBatch;
        }
        for (original[0..count], first[0..count]) |i, token| {
            self.lanes.getPtr(streams[i]).?.first = self.take(token);
            out[i] = .complete;
        }
    }

    /// A prompt pass begins: the stream's slot bound and its sequence fresh; where the pass resumes (a kept state).
    fn opening(self: *Lanes, s: *lanes.Stream) !?engine.Resume {
        const ids = s.prompt();
        if (ids.len + s.max_new + self.x.e.targetRows() >= self.x.e.capacity) return error.PromptTooLong;
        const l = try self.claim(s);
        l.pending_rows = null;
        l.held_n = 0;
        l.fill = null;
        l.first = null;
        try self.x.bind(l.slot);
        if (self.x.e.pool != null) {
            const rows = try @import("cuda_kv_extent.zig").requestRows(ids.len, s.max_new, fw.window_rows, self.x.e.capacity);
            try self.x.reserve(rows);
        }
        if (self.x.e.images) |v| v.clear();
        try self.x.begin(s.sampling);
        if (s.vision) |picture| {
            const service = self.vision orelse return error.VisionUnavailable;
            try self.x.vision(service, picture.encoded, ids, picture.image_token, picture.replica);
        }
        s.cached = 0;
        s.prompt_rows.clearRetainingCapacity();
        return if (s.prompt_logprobs != null) null else self.resumeFrom(s, ids.len); // prompt rows need every row
    }

    /// A prompt pass with images: rank 0's tower encodes those past `from`, then every rank holds their rows.
    fn images(self: *Lanes, s: *lanes.Stream, from: ?engine.Resume) !void {
        if (s.images.len == 0) return;
        const v = self.x.e.images orelse return error.NoVision;
        try v.begin(self.x.e.slot, s.images, if (from) |r| r.at else 0); // a refusal here leaves every rank as it was
        try self.x.images(v);
    }
    const Sink = @import("cuda_prompt_sink.zig").Sink;

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
        const l = self.lanes.getPtr(s) orelse return error.UnknownStream;
        return l.first orelse error.NoFirstToken;
    }

    /// The first token drawn again among `mask`'s allowed tokens (prefill drew it unmasked); a fresh handle to it.
    fn firstMaskedFn(ptr: *anyopaque, s: *lanes.Stream, position: u64, mask: []const u32) anyerror!u64 {
        const self = of(ptr);
        if (position != s.prompt_len) return error.PositionMismatch;
        const l = self.lanes.getPtr(s) orelse return error.UnknownStream;
        const bits = try self.putBits(0, 1, mask);
        if (bits == 0) return error.InvalidGrammarMasks;
        try self.x.bind(l.slot);
        return self.take(try self.x.redraw(@intCast(position), bits));
    }

    /// The head's drafts for the stream's next window as the host holds them (draftFn), for a grammar to walk.
    fn heldFn(ptr: *anyopaque, s: *lanes.Stream, out: []u32) anyerror!void {
        const l = of(ptr).lanes.getPtr(s) orelse return error.UnknownStream;
        if (out.len > l.held_n) return error.NoHeldDrafts;
        @memcpy(out, l.held[0..out.len]);
    }

    fn draftShapeFn(ptr: *anyopaque, s: *lanes.Stream, _: std.mem.Allocator) anyerror!?lanes.stream.Held {
        const self = of(ptr);
        if (self.x.e.dflash == null) return null;
        try self.finishPendingDrafts();
        const lane = self.lanes.getPtr(s) orelse return error.UnknownStream;
        return .{ .count = @intCast(lane.held_n) };
    }

    fn finishPendingDrafts(self: *Lanes) !void {
        try self.finishPendingDraftsExcept(&.{});
    }

    fn finishPendingDraftsExcept(self: *Lanes, superseded: []const u32) !void {
        return self.finishPendingDraftsUsing(superseded, self.x);
    }

    fn finishPendingDraftsUsing(self: *Lanes, superseded: []const u32, x: anytype) !void {
        const waiting = self.pending_drafts.retaining(superseded);
        var pending: PendingDraftBatch = .{};
        // A cancellation while waiting must not become a new paired finish submission.
        for (waiting.parts[0..waiting.count]) |part| {
            var it = self.lanes.iterator();
            while (it.next()) |entry| {
                if (entry.value_ptr.slot != part.slot) continue;
                if (entry.key_ptr.*.isCancelled()) {
                    entry.value_ptr.held_n = 0;
                    break;
                }
                pending.parts[pending.count] = part;
                pending.count += 1;
                break;
            }
        }
        const count = pending.count;
        self.pending_drafts.count = 0;
        if (count == 0) return;
        var results: [engine.max_shared]engine.DraftResult = @splat(.{});
        try x.draftBatch(pending.parts[0..count], results[0..count]);
        for (pending.parts[0..count], results[0..count]) |part, result| {
            var it = self.lanes.iterator();
            while (it.next()) |entry| {
                const stream = entry.key_ptr.*;
                const lane = entry.value_ptr;
                if (lane.slot != part.slot or stream.isCancelled()) continue;
                if (result.count > lane.held.len) return error.BadDraftBatch;
                @memcpy(lane.held[0..result.count], result.tokens[0..result.count]);
                lane.held_n = result.count;
                break;
            }
        }
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
        if (windows.len != out.len) return error.BadVerifyBatch;
        var total: usize = 0;
        for (windows) |w| {
            if (w.rows() > self.x.e.targetRows()) return error.WindowTooWide;
            total += w.rows();
        }
        if (self.x.e.settings.draft_profile.enabled and total > self.x.e.sharedLimit()) return error.WindowTooWide;
        const constrained = for (windows) |w| {
            if (w.masks.len > 0) break true;
        } else false;
        if (windows.len > 1 and !constrained) return self.verifyShared(windows, out);
        for (windows, out) |w, *o| {
            if (w.parents != null) return error.TreesNotBuilt;
            const l = try self.bind(w.stream, null);
            const rows = w.rows();
            if (rows > self.x.e.targetRows()) return error.WindowTooWide;
            if (w.held > l.held_n) return error.NoHeldDrafts;
            const pos = self.x.e.pos();
            for (w.positions, 0..) |p, r| if (p != pos + 1 + r) return error.PositionMismatch;
            var ids: [max_rows]u32 = undefined;
            ids[0] = w.pending;
            const drafts = if (w.held > 0) l.held[0..w.held] else w.tokens;
            @memcpy(ids[1..][0..drafts.len], drafts);
            var sampled: [max_rows]u32 = undefined;
            const bits = try self.putBits(0, rows, w.masks); // before the step goes out: a refusal after it strands the ranks
            self.x.e.structured = bits != 0;
            defer self.x.e.structured = false;
            try self.x.verify(ids[0..rows], sampled[0..rows], bits);
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
        var row: usize = 0; // the windows' rows so far: each one's grammar rows start there in the bits
        var structured = false;
        for (windows, 0..) |w, k| {
            if (w.parents != null) return error.TreesNotBuilt;
            const l = try self.bind(w.stream, null);
            const rows = w.rows();
            if (rows > self.x.e.targetRows()) return error.WindowTooWide;
            if (w.held > l.held_n) return error.NoHeldDrafts;
            const pos = self.x.e.pos();
            for (w.positions, 0..) |p, r| if (p != pos + 1 + r) return error.PositionMismatch;
            ids[k][0] = w.pending;
            const drafts = if (w.held > 0) l.held[0..w.held] else w.tokens;
            @memcpy(ids[k][1..][0..drafts.len], drafts);
            const bits = try self.putBits(row, rows, w.masks);
            structured = structured or bits != 0;
            parts[k] = .{ .slot = l.slot, .ids = ids[k][0..rows], .out = sampled[k][0..rows], .bits = bits };
            lane[k] = l;
            row += rows;
        }
        self.x.e.structured = structured; // one window under a grammar: the whole step draws from whole rows
        defer self.x.e.structured = false;
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
        return self.draftUsing(requests, self.x);
    }

    // The production Leader and the CPU command recorder execute the same preflight, commit and dispatch body.
    fn draftUsing(self: *Lanes, requests: []const be.DraftRequest, x: anytype) !void {
        if (!x.e.drafting()) return error.NoDraftHead;
        // Validate all requests before selecting either runtime. A late invalid item must not commit an earlier lane.
        try self.preflightCohort(requests, if (x.e.drafter != null) dflash.MAX_S else engine.max_shared);
        if (x.e.drafter != null) return self.draftBlock(requests, x);
        if (x.e.dflash != null) return self.draftBatch(requests, x);
        for (requests) |r| {
            const held = self.lanes.getPtr(r.stream).?;
            if (r.stream.isCancelled()) {
                held.held_n = 0;
                continue;
            }
            const input = try self.preflightDraft(held, r, try x.e.slotPos(held.slot));
            const l = try self.bindUsing(r.stream, input.keep, x);
            l.held_n = 0;
            // Engine.draft absorbs follow rows even at depth zero, maintaining MTP length alignment.
            if (r.position != x.e.pos() + 1) return error.PositionMismatch;
            if (r.lanes != null) return error.TreesNotBuilt;
            const depth = @min(r.depth, x.e.verifyPlan().drafts());
            if (r.rows) |rows| {
                for (rows, 0..) |row, i| if (row != i) return error.TreesNotBuilt;
                l.held_n = try x.draft(false, r.follow[0..rows.len], depth, &l.held);
            } else {
                const first = self.value(r.first orelse return error.NoFirstToken);
                l.held_n = try x.draft(true, &.{first}, depth, &l.held);
            }
            if (r.stream.isCancelled()) l.held_n = 0;
        }
    }

    const DraftInput = struct { pending: u32, keep: ?usize };

    fn preflightCohort(self: *const Lanes, requests: []const be.DraftRequest, limit: usize) !void {
        if (requests.len == 0 or requests.len > limit) return error.BadDraftBatch;
        var seen: [dflash.MAX_S]bool = @splat(false);
        for (requests) |r| {
            const l = self.lanes.getPtr(r.stream) orelse return error.UnknownStream;
            if (l.slot >= limit or seen[l.slot]) return error.BadDraftSlot;
            seen[l.slot] = true;
            _ = try self.preflightDraft(l, r, try self.x.e.slotPos(l.slot));
        }
    }

    fn preflightDraft(self: *const Lanes, l: *const Lane, r: be.DraftRequest, base: usize) !DraftInput {
        if (r.lanes != null) return error.TreesNotBuilt;
        if (r.depth > self.x.e.verifyPlan().drafts()) return error.BadDraftRequest;
        const base_pos: u64 = @intCast(base);
        var end = r.start;
        var keep: ?usize = null;
        const pending = if (r.rows) |rows| blk: {
            if (rows.len == 0 or rows.len > self.x.e.targetRows() or r.follow.len < rows.len) return error.BadDraftRequest;
            for (rows, 0..) |row, j| if (row != j) return error.TreesNotBuilt;
            // MTP embeds every accepted follow token, not just the terminal anchor.
            // This runs for the entire cohort before any target commit or paired command.
            for (r.follow[0..rows.len]) |token| if (token >= self.x.e.c.vocab) return error.BadDraftRequest;
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

    fn draftBatch(self: *Lanes, requests: []const be.DraftRequest, x: anytype) !void {
        if (requests.len == 0 or requests.len > engine.max_shared) return error.BadDraftBatch;
        var lanes_by_request: [engine.max_shared]*Lane = undefined;
        var superseded: [engine.max_shared]u32 = undefined;
        var pending: [engine.max_shared]u32 = undefined;
        var keeps: [engine.max_shared]?usize = undefined;
        var seen: [engine.max_shared]bool = @splat(false);
        for (requests, 0..) |r, i| {
            const l = self.lanes.getPtr(r.stream) orelse return error.UnknownStream;
            if (l.slot >= engine.max_shared or seen[l.slot]) return error.BadDraftSlot;
            seen[l.slot] = true;
            const base = try x.e.slotPos(l.slot);
            const input = try self.preflightDraft(l, r, base);
            pending[i] = input.pending;
            keeps[i] = input.keep;
            lanes_by_request[i] = l;
            superseded[i] = l.slot;
        }
        try self.finishPendingDraftsUsing(superseded[0..requests.len], x);
        for (requests, lanes_by_request[0..requests.len], keeps[0..requests.len]) |r, l, keep| {
            if (r.stream.isCancelled()) {
                l.held_n = 0;
                continue;
            }
            _ = try self.bindUsing(r.stream, keep, x);
            if (try x.e.slotPos(l.slot) + 1 != r.position) return error.PositionMismatch;
        }
        var parts: [engine.max_shared]engine.DraftPart = undefined;
        var original: [engine.max_shared]usize = undefined;
        var count: usize = 0;
        for (requests, lanes_by_request[0..requests.len], pending[0..requests.len], 0..) |r, l, token, i| {
            // Cancellation must suppress both synchronous and deferred paired submissions.
            if (r.stream.isCancelled() or r.depth == 0) continue;
            parts[count] = .{ .slot = l.slot, .position = r.position, .pending = token, .depth = r.depth };
            original[count] = i;
            count += 1;
        }
        for (lanes_by_request[0..requests.len]) |l| l.held_n = 0;
        self.pending_drafts.count = 0;
        if (count > 0) {
            if (!x.e.settings.draft_prelaunch or !x.e.dflash.?.asyncAvailable()) {
                var results: [engine.max_shared]engine.DraftResult = @splat(.{});
                try x.draftBatch(parts[0..count], results[0..count]);
                for (results[0..count], original[0..count]) |result, i| {
                    if (requests[i].stream.isCancelled()) continue;
                    const l = lanes_by_request[i];
                    @memcpy(l.held[0..result.count], result.tokens[0..result.count]);
                    l.held_n = result.count;
                }
                return;
            }
            @memcpy(self.pending_drafts.parts[0..count], parts[0..count]);
            self.pending_drafts.count = count;
            try x.prelaunchBatch(self.pending_drafts.parts[0..count]);
        }
    }

    /// The stream left the rounds: its sequence is freed on every rank (slot 0 stays for the next stream).
    fn releaseFn(ptr: *anyopaque, s: *lanes.Stream) void {
        const self = of(ptr);
        const kv = self.lanes.fetchRemove(s) orelse return;
        self.pending_drafts = self.pending_drafts.retaining(&.{kv.value.slot});
        self.x.release(kv.value.slot) catch |err| std.log.err("releasing slot {d}: {s}", .{ kv.value.slot, @errorName(err) });
    }
};

test "new draft cohorts retain waiting lanes and retire superseded copy or forced lanes" {
    var p: PendingDraftBatch = .{ .count = 2 };
    p.parts[0] = .{ .slot = 5, .position = 101, .pending = 11, .depth = 7 };
    p.parts[1] = .{ .slot = 1, .position = 201, .pending = 22, .depth = 7 };
    const admitted = p.retaining(&.{3});
    try std.testing.expectEqual(@as(usize, 2), admitted.count);
    const next_round = p.retaining(&.{5});
    try std.testing.expectEqual(@as(usize, 1), next_round.count);
    try std.testing.expectEqual(@as(u32, 1), next_round.parts[0].slot);
    try std.testing.expectEqual(@as(u64, 201), next_round.parts[0].position);
    try std.testing.expectEqual(@as(usize, 0), p.retaining(&.{ 1, 5 }).count);
    const released = p.retaining(&.{1});
    try std.testing.expectEqual(@as(usize, 1), released.count);
    try std.testing.expectEqualDeep(p.parts[0], released.parts[0]);
}

comptime {
    std.debug.assert(max_rows <= fw.window_rows);
}

test "prefilled lanes retain distinct first handles after another prompt completes" {
    const a = std.testing.allocator;
    var e: engine.Engine = undefined;
    e.drafter = null;
    e.gpa = a;
    var link: follow.Link = undefined;
    var backend: Lanes = .{ .x = .{ .e = &e, .link = &link } };
    defer backend.deinit();
    var first = try lanes.Stream.init(a, .{ .id = "first", .prompt = &.{ 1, 2 }, .max_new = 4, .eos = &.{} });
    defer first.deinit(a);
    var second = try lanes.Stream.init(a, .{ .id = "second", .prompt = &.{3}, .max_new = 4, .eos = &.{} });
    defer second.deinit(a);
    const one = backend.take(101);
    const two = backend.take(202);
    try backend.lanes.put(a, &first, .{ .slot = 0, .first = one });
    try backend.lanes.put(a, &second, .{ .slot = 1, .first = two });
    const b = backend.backend();
    try std.testing.expectEqual(one, try b.first(&first, 2));
    try std.testing.expectEqual(two, try b.first(&second, 1));
    try std.testing.expectEqual(@as(u32, 101), backend.drawn[one % ring]);
}

test "cold prompt batch eligibility preserves unsupported request state" {
    const a = std.testing.allocator;
    var e: engine.Engine = undefined;
    e.drafter = null;
    e.gpa = a;
    e.mtp = false;
    var link: follow.Link = undefined;
    var backend: Lanes = .{ .x = .{ .e = &e, .link = &link } };
    defer backend.deinit();
    var s = try lanes.Stream.init(a, .{ .id = "cold", .prompt = &.{ 1, 2 }, .max_new = 4, .eos = &.{} });
    defer s.deinit(a);
    try std.testing.expect(backend.canBatch(&s));
    // The HTTP adapter gives ordinary chat a lesson decision row, even without a learner.
    s.decide = 0;
    try std.testing.expect(backend.canBatch(&s));
    s.decide = null;
    s.prompt_len = 128;
    try std.testing.expect(!backend.canBatch(&s));
    s.prompt_len = 2;
    s.reuse.at = 2048;
    try std.testing.expect(!backend.canBatch(&s));
    s.reuse.at = 0;
    s.reuse.marks = &.{1};
    try std.testing.expect(!backend.canBatch(&s));
    s.reuse.marks = &.{};
    s.logprobs = 1;
    try std.testing.expect(!backend.canBatch(&s));
    s.logprobs = null;
    e.mtp = true;
    try std.testing.expect(!backend.canBatch(&s));
}

test "DFlash publishes the actual held count through the existing lane shape callback" {
    var e: engine.Engine = undefined;
    e.drafter = null;
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
    e.drafter = null;
    e.settings = .{};
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

test "TF_GLM_LONE_COSTS takes two positive ms and models each window width" {
    const c = try LoneCosts.parse("38,7.9");
    try std.testing.expectEqual(@as(u32, 1), c.window[0].width);
    try std.testing.expectApproxEqAbs(@as(f64, 38.0), c.window[0].ms, 1e-12);
    try std.testing.expectEqual(@as(u32, max_rows), c.window[max_rows - 1].width);
    try std.testing.expectApproxEqAbs(@as(f64, 38.0 + 7.9 * (max_rows - 1)), c.window[max_rows - 1].ms, 1e-9);
    for ([_][]const u8{ "", "38", "38,", ",7.9", "0,7.9", "38,-1", "inf,1", "nan,1", "38,7.9,1", "a,b", " 38,7.9" }) |raw|
        try std.testing.expectError(error.BadLoneCosts, LoneCosts.parse(raw));
}

fn loneFixture(e: *engine.Engine, link: *follow.Link, costs: ?LoneCosts) Lanes {
    e.drafter = null;
    e.settings = .{ .dflash_verify = .dflash6 };
    e.mtp = false;
    e.dflash = @ptrFromInt(256);
    return .{ .x = .{ .e = e, .link = link }, .lone_costs = costs };
}

fn loneDepth(cfg: *const lanes.Config, p: f64) !i64 {
    const gpa = std.testing.allocator;
    const prior: [engine.max_drafts]f64 = @splat(p);
    var rule: lanes.depth.Rule = .{ .gpa = gpa, .k = cfg.k, .prior = &prior, .most_drafts = cfg.most_drafts, .family_costs = &cfg.family_costs, .shared_costs = &cfg.shared_costs, .mtp_step_ms = cfg.mtp_step_ms, .plain_guard = cfg.plain_guard, .node_probabilities = cfg.node_probabilities, .batch_rows = cfg.batch_rows, .batched_heads = cfg.draft_streams };
    defer rule.deinit();
    var st: ?lanes.depth.State = null;
    defer if (st) |*s| s.deinit(gpa);
    return rule.depth(.{ .state = &st, .draft_room = 1000, .finished = false, .forced = false });
}

test "lone costs let a lone DFlash stream price drafts past three, by acceptance" {
    var e: engine.Engine = undefined;
    e.drafter = null;
    var link: follow.Link = undefined;
    const untimed = loneFixture(&e, &link, null);
    var plain = try lanes.Config.init(std.testing.allocator, untimed.facts(), max_rows, engine.max_drafts);
    defer plain.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(u32, 6), plain.most_drafts);
    try std.testing.expectEqual(@as(i64, 3), try loneDepth(&plain, 0.95)); // depth.zig's untimed rule stops at three
    const timed = loneFixture(&e, &link, try LoneCosts.parse("38,7.9"));
    var cfg = try lanes.Config.init(std.testing.allocator, timed.facts(), max_rows, engine.max_drafts);
    defer cfg.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(?f64, 38.0 + 7.9 * 6), cfg.family_costs.get(7));
    try std.testing.expectEqual(@as(i64, 6), try loneDepth(&cfg, 0.9)); // code-like acceptance: the whole plan pays
    try std.testing.expect(try loneDepth(&cfg, 0.6) <= 3); // prose-like acceptance: shallow stays cheaper a token
}

test "lone costs preserve positive-probability shared draft allocation across row caps and overheads" {
    const gpa = std.testing.allocator;
    var e: engine.Engine = undefined;
    e.drafter = null;
    var link: follow.Link = undefined;
    const timed = loneFixture(&e, &link, try LoneCosts.parse("38,7.9"));
    var cfg = try lanes.Config.init(gpa, timed.facts(), max_rows, engine.max_drafts);
    defer cfg.deinit(gpa);
    const untimed = loneFixture(&e, &link, null);
    var plain = try lanes.Config.init(gpa, untimed.facts(), max_rows, engine.max_drafts);
    defer plain.deinit(gpa);
    try std.testing.expect(plain.shared_costs.empty());
    try std.testing.expectEqual(@as(?f64, 1.0), cfg.shared_costs.get(cfg.batch_rows));
    const sets = [_][]const []const f64{
        &.{ &.{ 0.9, 0.81, 0.73, 0.66, 0.59, 0.53 }, &.{ 0.6, 0.36, 0.22, 0.13, 0.08, 0.05 } },
        &.{ &.{ 0.5, 0.2, 0.05, 0.01, 0.001, 0.0001 }, &.{ 0.95, 0.9, 0.86, 0.81, 0.77, 0.74 }, &.{ 0.7, 0.49, 0.34, 0.24, 0.17, 0.12 } },
    };
    for (sets) |probs| {
        const fixed: [3]u32 = @splat(1);
        for ([_]f64{ 0.0, 8.0, 120.0 }) |overhead| {
            for ([_]u64{ 8, 32, cfg.batch_rows }) |rows| {
                const want = try lanes.allocate.allocateDrafts(gpa, fixed[0..probs.len], probs, &plain.shared_costs, overhead, rows, .{ .batched = true });
                defer gpa.free(want);
                const got = try lanes.allocate.allocateDrafts(gpa, fixed[0..probs.len], probs, &cfg.shared_costs, overhead, rows, .{ .batched = true });
                defer gpa.free(got);
                try std.testing.expectEqualSlices(u32, want, got);
            }
        }
    }
}

test "lone costs omit zero-probability shared tails while untimed fill spends the remaining rows" {
    const gpa = std.testing.allocator;
    var e: engine.Engine = undefined;
    e.drafter = null;
    var link: follow.Link = undefined;
    const timed = loneFixture(&e, &link, try LoneCosts.parse("38,7.9"));
    var cfg = try lanes.Config.init(gpa, timed.facts(), max_rows, engine.max_drafts);
    defer cfg.deinit(gpa);
    const untimed = loneFixture(&e, &link, null);
    var plain = try lanes.Config.init(gpa, untimed.facts(), max_rows, engine.max_drafts);
    defer plain.deinit(gpa);
    const probs = [_][]const f64{ &.{ 0.3, 0.09, 0.03, 0.01, 0.0, 0.0 }, &.{ 0.3, 0.09, 0.03, 0.01, 0.0, 0.0 } };
    for ([_]f64{ 0.0, 8.0, 120.0 }) |overhead| {
        const full = try lanes.allocate.allocateDrafts(gpa, &.{ 1, 1 }, &probs, &plain.shared_costs, overhead, cfg.batch_rows, .{ .batched = true });
        defer gpa.free(full);
        const priced = try lanes.allocate.allocateDrafts(gpa, &.{ 1, 1 }, &probs, &cfg.shared_costs, overhead, cfg.batch_rows, .{ .batched = true });
        defer gpa.free(priced);
        try std.testing.expectEqualSlices(u32, &.{ 6, 6 }, full);
        try std.testing.expectEqualSlices(u32, &.{ 4, 4 }, priced);
        for (probs, full, priced) |p, all, useful| {
            for (p[0..useful]) |probability| try std.testing.expect(probability > 0);
            for (p[useful..all]) |probability| try std.testing.expectEqual(@as(f64, 0), probability);
        }
    }
}

test "DFlash draft preflight distinguishes accepted rows already committed by the core" {
    var e: engine.Engine = undefined;
    e.drafter = null;
    e.settings = .{};
    e.dflash = @ptrFromInt(256);
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

// Command-level control recording only: no numerical draft quality is manufactured here.
// The same draftUsing body dispatches the real Leader in production. MTP length writes below use real State code.
const DraftRecording = struct {
    e: *engine.Engine,
    binds: [2]usize = @splat(0),
    keeps: [2]usize = @splat(0),
    absorbed: [2]usize = @splat(0),
    proposals: [2]usize = @splat(0),
    blocks: usize = 0,
    batches: usize = 0,
    drafts: usize = 0,
    depth_zero: usize = 0,
    prelaunches: usize = 0,
    fn bind(self: *@This(), slot: u32) !void {
        self.binds[slot] += 1;
        try self.e.bind(slot);
    }
    fn keep(self: *@This(), n: usize) !void {
        self.keeps[self.e.slot] += 1;
        self.e.cur.st.pos += n; // target commit recording, not a CUDA numerical substitute
        if (self.e.dflash != null) self.absorbed[self.e.slot] += n; // real legacy Engine.keep calls DFlash.accept
    }
    fn block(self: *@This(), reqs: []dflash.Req) !void {
        self.blocks += 1;
        for (reqs) |q| {
            self.absorbed[q.slot] += q.n;
            if (!q.propose) continue;
            self.proposals[q.slot] += 1;
            @memset(q.out.?, 77);
            @memset(q.probs.?, 0.9);
        }
    }
    fn draftBatch(self: *@This(), parts: []const engine.DraftPart, out: []engine.DraftResult) !void {
        self.batches += 1;
        for (parts, out) |part, *result| {
            self.proposals[part.slot] += 1;
            result.* = .{};
            result.count = part.depth;
            @memset(result.tokens[0..result.count], 77);
        }
    }
    fn prelaunchBatch(self: *@This(), _: []const engine.DraftPart) !void {
        self.prelaunches += 1;
        return error.UnexpectedDeferredSubmission;
    }
    const Forward = struct {
        recording: *DraftRecording,
        pub fn mtp(self: @This(), ids: []const i32, _: []const u64) !void {
            self.recording.absorbed[self.recording.e.slot] += ids.len;
        }
    };
    const Drawing = struct {
        var active: ?*DraftRecording = null;
        fn draw(e: *engine.Engine, _: u64, rows: usize, _: u64, _: u64, out: []u32) !void {
            const self = active.?;
            std.debug.assert(rows == 1 and self.e == e);
            self.proposals[e.slot] += 1;
            out[0] = 77;
        }
    };
    fn draft(self: *@This(), from_prompt: bool, follow_ids: []const u32, depth: usize, out: []u32) !usize {
        std.debug.assert(!from_prompt); // these regressions exercise accepted pending rows, not empty prompt no-ops
        self.drafts += 1;
        if (depth == 0) self.depth_zero += 1;
        Drawing.active = self;
        defer Drawing.active = null;
        return self.e.draftMtpWith(follow_ids, depth, out, 0, Forward{ .recording = self }, Drawing.draw);
    }
};
const DraftFixture = struct {
    const Mode = enum { legacy, canonical, mtp };
    e: engine.Engine,
    sequences: [2]@import("cuda_engine_types.zig").Seq,
    streams: [2]lanes.Stream,
    positions: cuda.DeviceBuffer,
    backend: Lanes,
    recording: DraftRecording,
    fn init(f: *@This(), d: *const cuda.Driver, mode: Mode) !void {
        f.positions = try cuda.DeviceBuffer.alloc(d, 8);
        f.e = undefined;
        f.e.drafter = if (mode == .canonical) @ptrFromInt(512) else null;
        f.e.dflash = if (mode == .legacy) @ptrFromInt(256) else null;
        f.e.mtp = mode == .mtp;
        f.e.settings = .{};
        f.e.gpa = std.testing.allocator;
        f.e.d = d;
        f.e.stream = .{ .d = d, .handle = null };
        f.e.c = .{};
        f.e.c.hidden = 4;
        f.e.c.vocab = 100;
        f.e.bm.logits = 0;
        f.e.bm.fnormed = 0;
        f.e.pool = null;
        f.e.seqs = .empty;
        f.e.slot = 0;
        for (&f.sequences, &f.streams, 0..) |*seq, *stream, i| {
            seq.* = .{ .st = undefined };
            seq.st.pos = 10 + 10 * i;
            seq.st.mtp_len = seq.st.pos;
            seq.st.mtp_pos_dev = f.positions.ptr + 4 * i;
            seq.st.arena = .{ .buf = .{ .d = d, .ptr = f.positions.ptr, .len = f.positions.len, .borrowed = true } };
            try f.e.seqs.append(std.testing.allocator, seq);
            stream.* = undefined;
            stream.cancel_check = null;
        }
        f.e.cur = &f.sequences[0];
        f.backend = .{ .x = .{ .e = &f.e, .link = undefined } };
        for (&f.streams, 0..) |*stream, i| try f.backend.lanes.put(std.testing.allocator, stream, .{ .slot = @intCast(i), .pending_rows = 3, .held_n = 2 });
        f.recording = .{ .e = &f.e };
    }
    fn deinit(f: *@This()) void {
        f.backend.deinit();
        f.e.seqs.deinit(std.testing.allocator);
        f.positions.free();
    }
    fn requests(f: *@This(), depth: u32) [2]be.DraftRequest {
        return .{
            .{ .stream = &f.streams[0], .follow = &.{ 1, 2 }, .rows = &.{ 0, 1 }, .start = 10, .position = 13, .depth = depth },
            .{ .stream = &f.streams[1], .follow = &.{ 1, 2 }, .rows = &.{ 0, 1 }, .start = 20, .position = 23, .depth = depth },
        };
    }
};

test "cancelled pending draft lane publishes no held output or paired work while its healthy neighbor completes; uncancelled control records real dispatch" {
    const Cancel = struct {
        fn check(_: *anyopaque) bool {
            return true;
        }
    };
    const M = @import("cuda_test_driver.zig").Memory;
    var d = M.driver();
    for ([_]DraftFixture.Mode{ .legacy, .canonical, .mtp }) |mode| for ([_]bool{ true, false }) |cancel| {
        var f: DraftFixture = undefined;
        try f.init(&d, mode);
        defer f.deinit();
        var marker: u8 = 0;
        if (cancel) f.streams[0].cancel_check = .{ .ptr = &marker, .check = Cancel.check };
        const requests = f.requests(1); // nonzero work, two accepted pending rows on BOTH streams
        try f.backend.draftUsing(&requests, &f.recording);
        const first = f.backend.lanes.getPtr(&f.streams[0]).?;
        const healthy = f.backend.lanes.getPtr(&f.streams[1]).?;
        try std.testing.expectEqual(@as(usize, 1), healthy.held_n);
        try std.testing.expectEqual(@as(u32, 77), healthy.held[0]);
        try std.testing.expectEqual(@as(?usize, null), healthy.pending_rows);
        try std.testing.expectEqual(@as(usize, 22), f.sequences[1].st.pos);
        try std.testing.expectEqual(@as(usize, 1), f.recording.proposals[1]);
        try std.testing.expectEqual(@as(usize, 1), f.recording.binds[1]);
        try std.testing.expectEqual(@as(usize, 1), f.recording.keeps[1]);
        try std.testing.expectEqual(@as(usize, 2), f.recording.absorbed[1]);
        try std.testing.expectEqual(@as(usize, if (cancel) 0 else 1), f.recording.proposals[0]);
        try std.testing.expectEqual(@as(usize, if (cancel) 0 else 1), f.recording.binds[0]);
        try std.testing.expectEqual(@as(usize, if (cancel) 0 else 1), f.recording.keeps[0]);
        try std.testing.expectEqual(@as(usize, if (cancel) 0 else 2), f.recording.absorbed[0]);
        try std.testing.expectEqual(@as(usize, if (cancel) 0 else 1), first.held_n);
        try std.testing.expectEqual(@as(?usize, if (cancel) 3 else null), first.pending_rows);
        try std.testing.expectEqual(@as(usize, if (cancel) 10 else 12), f.sequences[0].st.pos);
        if (mode == .mtp) try std.testing.expectEqual(@as(usize, 22), f.sequences[1].st.mtp_len);
        try std.testing.expectEqual(@as(usize, 0), f.recording.prelaunches);
    };
    try std.testing.expect(M.empty());
}

test "accepted depth-zero draft turn absorbs before the next positive-depth turn, including real MTP length writes" {
    const M = @import("cuda_test_driver.zig").Memory;
    var d = M.driver();
    for ([_]DraftFixture.Mode{ .legacy, .canonical, .mtp }) |mode| {
        var f: DraftFixture = undefined;
        try f.init(&d, mode);
        defer f.deinit();
        var request = f.requests(0)[0];
        try f.backend.draftUsing(&.{request}, &f.recording);
        try std.testing.expectEqual(@as(usize, 2), f.recording.absorbed[0]);
        try std.testing.expectEqual(@as(usize, 0), f.recording.proposals[0]);
        try std.testing.expectEqual(@as(usize, 0), f.recording.batches);
        try std.testing.expectEqual(@as(usize, 0), f.backend.lanes.getPtr(&f.streams[0]).?.held_n);
        if (mode == .mtp) {
            try std.testing.expectEqual(@as(usize, 1), f.recording.depth_zero);
            try std.testing.expectEqual(@as(usize, 12), f.sequences[0].st.mtp_len);
            const actual: *const u32 = @ptrFromInt(f.sequences[0].st.mtp_pos_dev);
            try std.testing.expectEqual(@as(u32, 12), actual.*);
        }
        // A later verify contributes one more accepted row. The next proposal must follow all three context rows.
        f.backend.lanes.getPtr(&f.streams[0]).?.pending_rows = 2;
        request = .{ .stream = &f.streams[0], .follow = &.{3}, .rows = &.{0}, .start = 12, .position = 14, .depth = 1 };
        try f.backend.draftUsing(&.{request}, &f.recording);
        try std.testing.expectEqual(@as(usize, 3), f.recording.absorbed[0]);
        try std.testing.expectEqual(@as(usize, 1), f.recording.proposals[0]);
        try std.testing.expectEqual(@as(usize, 1), f.backend.lanes.getPtr(&f.streams[0]).?.held_n);
        if (mode == .mtp) try std.testing.expectEqual(@as(usize, 13), f.sequences[0].st.mtp_len);
    }
    try std.testing.expect(M.empty());
}

test "cancelled waiting legacy draft is not resubmitted when a healthy deferred neighbor completes" {
    const Cancel = struct {
        fn check(_: *anyopaque) bool {
            return true;
        }
    };
    const M = @import("cuda_test_driver.zig").Memory;
    var d = M.driver();
    var f: DraftFixture = undefined;
    try f.init(&d, .legacy);
    defer f.deinit();
    var marker: u8 = 0;
    f.streams[0].cancel_check = .{ .ptr = &marker, .check = Cancel.check };
    f.backend.pending_drafts = .{ .count = 2 };
    f.backend.pending_drafts.parts[0] = .{ .slot = 0, .position = 11, .pending = 1, .depth = 1 };
    f.backend.pending_drafts.parts[1] = .{ .slot = 1, .position = 21, .pending = 2, .depth = 1 };
    try f.backend.finishPendingDraftsUsing(&.{}, &f.recording);
    try std.testing.expectEqualSlices(usize, &.{ 0, 1 }, &f.recording.proposals);
    try std.testing.expectEqual(@as(usize, 1), f.recording.batches);
    try std.testing.expectEqual(@as(usize, 0), f.backend.lanes.getPtr(&f.streams[0]).?.held_n);
    try std.testing.expectEqual(@as(usize, 1), f.backend.lanes.getPtr(&f.streams[1]).?.held_n);
    try std.testing.expectEqual(@as(usize, 0), f.backend.pending_drafts.count);
}

test "nonterminal invalid follow on a later cohort item is refused before recorded bind keep absorb or paired submission" {
    const M = @import("cuda_test_driver.zig").Memory;
    var d = M.driver();
    for ([_]DraftFixture.Mode{ .legacy, .canonical, .mtp }) |mode| {
        var f: DraftFixture = undefined;
        try f.init(&d, mode);
        defer f.deinit();
        var requests = f.requests(1);
        requests[1].follow = &.{ 100, 2 }; // last token is valid, but MTP consumes the earlier invalid token as well
        try std.testing.expectError(error.BadDraftRequest, f.backend.draftUsing(&requests, &f.recording));
        try std.testing.expectEqualSlices(usize, &.{ 0, 0 }, &f.recording.binds);
        try std.testing.expectEqualSlices(usize, &.{ 0, 0 }, &f.recording.keeps);
        try std.testing.expectEqualSlices(usize, &.{ 0, 0 }, &f.recording.absorbed);
        try std.testing.expectEqualSlices(usize, &.{ 0, 0 }, &f.recording.proposals);
        try std.testing.expectEqual(@as(usize, 0), f.recording.blocks + f.recording.batches + f.recording.drafts + f.recording.prelaunches);
        for (&f.streams, &f.sequences, 0..) |*stream, *seq, i| {
            const lane = f.backend.lanes.getPtr(stream).?;
            try std.testing.expectEqual(@as(usize, 2), lane.held_n);
            try std.testing.expectEqual(@as(?usize, 3), lane.pending_rows);
            try std.testing.expectEqual(10 + 10 * i, seq.st.pos);
        }
        requests[1].follow = &.{ 1, 2 }; // positive control proves the recorder sees bind/keep and proposal work
        try f.backend.draftUsing(&requests, &f.recording);
        try std.testing.expectEqualSlices(usize, &.{ 1, 1 }, &f.recording.proposals);
        try std.testing.expectEqualSlices(usize, &.{ 1, 1 }, &f.recording.binds);
    }
    try std.testing.expect(M.empty());
}

test "a later invalid mixed draft request refuses before any bind or paired command" {
    // Every dispatch owner preserves the same no-side-effects preflight and cancellation contract.
    inline for ([_]enum { legacy, canonical, mtp }{ .legacy, .canonical, .mtp }) |mode| {
        const canonical = mode == .canonical;
        var e: engine.Engine = undefined;
        e.drafter = if (canonical) @ptrFromInt(512) else null;
        e.settings = .{};
        e.gpa = std.testing.allocator;
        e.c = .{};
        e.mtp = mode == .mtp;
        e.dflash = if (mode == .mtp) null else @ptrFromInt(256);
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
            .{ .stream = &second, .follow = &.{2}, .rows = &.{0}, .start = 20, .position = 22, .depth = if (canonical) 8 else 4 },
        };
        try std.testing.expectError(error.BadDraftRequest, b.draft(&requests));
        try std.testing.expectEqual(@as(usize, 10), first_seq.st.pos);
        try std.testing.expectEqual(@as(usize, 20), second_seq.st.pos);
        try std.testing.expectEqual(@as(usize, 2), backend.lanes.getPtr(&first).?.held_n);
        try std.testing.expectEqual(@as(?usize, 2), backend.lanes.getPtr(&second).?.pending_rows);
        e.settings.dflash_verify = .dflash6;
        requests[1].depth = if (canonical) 8 else 7;
        try std.testing.expectError(error.BadDraftRequest, b.draft(&requests));
        try std.testing.expectEqual(@as(usize, 10), first_seq.st.pos);
        try std.testing.expectEqual(@as(?usize, 2), backend.lanes.getPtr(&second).?.pending_rows);
        e.settings.dflash_verify = .baseline;
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
        requests[1].depth = if (canonical) 8 else 4;
        try std.testing.expectError(error.BadDraftRequest, b.draft(&requests));
        try std.testing.expectEqual(@as(usize, 12), first_seq.st.pos);
        try std.testing.expectEqual(@as(usize, 20), second_seq.st.pos);
        try std.testing.expectEqual(@as(usize, 2), backend.lanes.getPtr(&first).?.held_n);
        try std.testing.expectEqual(@as(?usize, 2), backend.lanes.getPtr(&second).?.pending_rows);
    }
}

/// Eligibility for the maintained fresh-prompt batching path.
pub fn canBatchWhole(s: anytype) bool {
    return s.prompt_logprobs == null and s.images.len == 0 and s.reuse.saved == null and s.prompt_len <= engine.prompt_rows;
}

test "a lane's slot is chosen before its map entry exists" {
    var storage: [8192]u8 = @splat(0);
    var fixed = std.heap.FixedBufferAllocator.init(&storage);
    var e: engine.Engine = undefined;
    e.drafter = null;
    e.gpa = fixed.allocator();
    var self: Lanes = .{ .x = .{ .e = &e, .link = undefined } };
    defer self.lanes.deinit(e.gpa);
    var streams: [3]lanes.Stream = undefined;
    const Probe = struct {
        var bad = false;
        fn choose(l: *const Lanes) u32 {
            bad = l.lanes.count() != 0;
            return 0;
        }
    };
    _ = try self.claimUsing(&streams[0], Probe.choose);
    try std.testing.expect(!Probe.bad);
    try std.testing.expectEqual(@as(u32, 0), (try self.claim(&streams[0])).slot);
    try std.testing.expectEqual(@as(u32, 1), (try self.claim(&streams[1])).slot);
    _ = self.lanes.remove(&streams[0]);
    try std.testing.expectEqual(@as(u32, 0), (try self.claim(&streams[2])).slot);
    try std.testing.expectEqual(@as(u32, 1), (try self.claim(&streams[1])).slot);
}

test "a failed lane insertion leaves no entry, and a retry takes slot zero" {
    var fail = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = 0 });
    var e: engine.Engine = undefined;
    e.drafter = null;
    e.gpa = fail.allocator();
    var self: Lanes = .{ .x = .{ .e = &e, .link = undefined } };
    defer self.lanes.deinit(e.gpa);
    var stream: lanes.Stream = undefined;
    try std.testing.expectError(error.OutOfMemory, self.claim(&stream));
    try std.testing.expectEqual(@as(usize, 0), self.lanes.count());
    fail.fail_index = std.math.maxInt(usize);
    try std.testing.expectEqual(@as(u32, 0), (try self.claim(&stream)).slot);
}
