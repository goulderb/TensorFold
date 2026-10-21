//! TP over processes until #552: rank 0's Leader sends each step (header, ids, slot) first; other ranks `follow` it.

const std = @import("std");
const cuda = @import("cuda");
const engine = @import("cuda_engine.zig");
const sampler = @import("cuda_sampler.zig");
const prompt_wire = @import("cuda_prompt_wire.zig");
const dflash = @import("dflash.zig");
const images_mod = @import("cuda_images.zig");
const vcfg = @import("vision_config.zig");
const Engine = engine.Engine;
const vision_runtime = @import("cuda_vision_runtime.zig");
const vision_binding = @import("cuda_vision_binding.zig");

test {
    _ = @import("cuda_spill_order_test.zig");
}

pub const Kind = enum(u32) { begin = 1, prefill, verify, keep, draft, end, drop, release, chunk, shared, reserve, spill_write, spill_read, spill_forget, vision, redraw_masked, verify_masked, draft_batch, prefill_shared, prelaunch_batch, image, block, many, redraw };

/// One step as it crosses: 64 bytes, then `n` u32 ids.
pub const Header = extern struct {
    kind: u32,
    n: u32 = 0, // ids that follow
    depth: u32 = 0, // draft: drafts to chain; chunk: where it starts; shared: its parts
    flags: u32 = 0, // begin: sampled; draft: from the prompt; block: proposing request bitset; prefill, chunk: prompt_logprobs; verify, shared: whole rows
    a: u64 = 0, // keep: rows; prefill, chunk: the kept state it resumes (0: none); redraw: the prompt's length
    seed: u64 = 0, // begin: the sampling seed; prefill, chunk: the kept state's position
    temperature: f64 = 0,
    top_p: f64 = 0,
    min_p: f64 = 0,
    top_k: u32 = 0, // begin: top_k; chunk: the prompt's length
    slot: u32 = 0, // the stream's sequence the step runs on (Engine.bind)

    comptime {
        std.debug.assert(@sizeOf(Header) == 64);
    }
};

/// tf_glm_put_words' argument: a step's header (16 words) and up to `put_ids` ids.
pub const Words = extern struct { w: [96]u32 };
pub const put_ids = 96 - 16;

const BlockWire = struct { ids: []const u32, proposals: u32 };
fn encodeBlock(reqs: []const dflash.Req, out: *[3 * dflash.MAX_S]u32) !BlockWire {
    if (reqs.len == 0 or reqs.len > dflash.MAX_S) return error.TooManyStreams;
    var proposals: u32 = 0;
    var seen: [dflash.MAX_S]bool = @splat(false);
    for (reqs, 0..) |q, i| {
        if (q.slot >= dflash.MAX_S or seen[q.slot]) return error.BadDraftSlot;
        seen[q.slot] = true;
        if (q.n > dflash.STEP_ROWS or q.anchor >= dflash.VOCAB) return error.BadDraftRequest;
        out[3 * i ..][0..3].* = .{ q.slot, @intCast(q.n), q.anchor };
        if (q.propose) proposals |= @as(u32, 1) << @intCast(i);
    }
    return .{ .ids = out[0 .. 3 * reqs.len], .proposals = proposals };
}
fn decodeBlock(words: []const u32, count: usize, proposals: u32, out: *[dflash.MAX_S]dflash.Req) ![]dflash.Req {
    if (count == 0 or count > dflash.MAX_S or words.len != 3 * count or proposals >> @intCast(count) != 0) return error.BadStep;
    var seen: [dflash.MAX_S]bool = @splat(false);
    for (out[0..count], 0..) |*q, i| {
        const slot = words[3 * i];
        if (slot >= dflash.MAX_S or seen[slot]) return error.BadDraftSlot;
        seen[slot] = true;
        q.* = .{ .slot = slot, .pos = 0, .n = words[3 * i + 1], .anchor = words[3 * i + 2], .propose = proposals & (@as(u32, 1) << @intCast(i)) != 0 };
        if (q.n > dflash.STEP_ROWS or q.anchor >= dflash.VOCAB) return error.BadDraftRequest;
    }
    return out[0..count];
}

/// Follower block dispatch: the exact submitted header owns the proposal collective cohort.
fn dispatchBlock(e: *Engine, h: Header, ids: []const u32) !void {
    if (h.kind != @backingInt(Kind.block) or h.n != ids.len) return error.BadStep;
    var reqs: [dflash.MAX_S]dflash.Req = undefined;
    const decoded = try decodeBlock(ids, h.depth, h.flags, &reqs);
    try e.draftBlock(decoded);
}

test "paired block header and payload round-trip mixed all-proposal and all-absorb cohorts" {
    for ([_]u32{ 0b000, 0b101, 0b111 }) |mask| {
        var leader: [3]dflash.Req = .{
            .{ .slot = 2, .pos = 102, .n = 2, .anchor = 11 },
            .{ .slot = 0, .pos = 200, .n = 0, .anchor = 22 },
            .{ .slot = 1, .pos = 303, .n = 3, .anchor = 33 },
        };
        for (&leader, 0..) |*q, i| q.propose = mask & (@as(u32, 1) << @intCast(i)) != 0;
        var words: [3 * dflash.MAX_S]u32 = undefined;
        const encoded = try encodeBlock(&leader, &words);
        const sent: Header = .{ .kind = @backingInt(Kind.block), .n = @intCast(encoded.ids.len), .depth = leader.len, .flags = encoded.proposals };
        var bytes: [@sizeOf(Header)]u8 = undefined;
        @memcpy(&bytes, std.mem.asBytes(&sent));
        const received = std.mem.bytesToValue(Header, &bytes);
        var payload: [3 * dflash.MAX_S]u32 = undefined;
        @memcpy(payload[0..received.n], encoded.ids);
        var follower: [dflash.MAX_S]dflash.Req = undefined;
        const decoded = try decodeBlock(payload[0..received.n], received.depth, received.flags, &follower);
        try std.testing.expectEqual(mask, received.flags);
        var leader_rows: usize = 0;
        var follower_rows: usize = 0;
        var leader_proposals: usize = 0;
        var follower_proposals: usize = 0;
        for (leader, decoded) |a, b| {
            try std.testing.expectEqual(a.slot, b.slot);
            try std.testing.expectEqual(a.n, b.n);
            try std.testing.expectEqual(a.anchor, b.anchor);
            try std.testing.expectEqual(a.propose, b.propose);
            try std.testing.expect(b.out == null and b.probs == null);
            leader_rows += a.n;
            follower_rows += b.n;
            leader_proposals += @intFromBool(a.propose);
            follower_proposals += @intFromBool(b.propose);
        }
        try std.testing.expectEqual(leader_rows, follower_rows);
        try std.testing.expectEqual(@as(usize, 5), follower_rows);
        try std.testing.expectEqual(leader_proposals, follower_proposals);
        if (mask == 0) try std.testing.expectEqual(@as(usize, 0), follower_proposals);
        try std.testing.expectError(error.BadStep, decodeBlock(encoded.ids, 3, 0b1000, &follower));
        try std.testing.expectError(error.BadStep, decodeBlock(encoded.ids[0..8], 3, mask, &follower));
    }
}

fn encodeDraftBatch(parts: []const engine.DraftPart, out: *[engine.max_shared * 5]u32) ![]const u32 {
    if (parts.len == 0 or parts.len > engine.max_shared) return error.BadDraftBatch;
    for (parts, 0..) |p, i| {
        const at = i * 5;
        out[at] = p.slot;
        out[at + 1] = @truncate(p.position);
        out[at + 2] = @truncate(p.position >> 32);
        out[at + 3] = p.pending;
        out[at + 4] = p.depth;
    }
    return out[0 .. parts.len * 5];
}

fn decodeDraftBatch(words: []const u32, count: usize, out: *[engine.max_shared]engine.DraftPart) ![]const engine.DraftPart {
    if (count == 0 or count > engine.max_shared or words.len != count * 5) return error.BadDraftBatch;
    var seen: [engine.max_shared]bool = @splat(false);
    for (out[0..count], 0..) |*p, i| {
        const at = i * 5;
        const slot = words[at];
        if (slot >= engine.max_shared or seen[slot]) return error.BadDraftSlot;
        seen[slot] = true;
        p.* = .{ .slot = slot, .position = (@as(u64, words[at + 2]) << 32) | words[at + 1], .pending = words[at + 3], .depth = words[at + 4] };
        if (p.position == 0 or p.pending >= 154880 or p.depth > engine.max_drafts) return error.BadDraftRequest;
    }
    return out[0..count];
}

/// The most ids one step carries (a prompt at the engine's capacity).
pub const Link = struct {
    d: *const cuda.Driver,
    lib: *const cuda.nccl.Library,
    comm: cuda.nccl.Comm,
    stream: cuda.Stream,
    buf: cuda.DeviceBuffer, // header, then ids
    ids: []u32, // host copy of the last step's ids
    gpa: std.mem.Allocator,
    wait: bool, // TF_GLM_LINK_WAIT=1: every step uploaded and waited for, as before the put

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, lib: *const cuda.nccl.Library, comm: cuda.nccl.Comm, stream: cuda.Stream, asked_ids: usize) !Link {
        const max_ids = @max(asked_ids, prompt_wire.words);
        var buf = try cuda.DeviceBuffer.alloc(d, 64 + max_ids * 4);
        errdefer buf.free();
        const wait = if (@import("sys").getenv("TF_GLM_LINK_WAIT")) |v| std.mem.eql(u8, v, "1") else false;
        return .{ .d = d, .lib = lib, .comm = comm, .stream = stream, .buf = buf, .ids = try gpa.alloc(u32, max_ids), .gpa = gpa, .wait = wait };
    }

    pub fn deinit(l: *Link) void {
        l.gpa.free(l.ids);
        l.buf.free();
    }

    fn bcast(l: *Link, bytes: usize) !void {
        try l.post(bytes);
        try l.stream.synchronize();
    }

    fn post(l: *Link, bytes: usize) !void {
        try l.lib.check(l.lib.api.ncclBroadcast(l.buf.ptr, l.buf.ptr, bytes, .u8, 0, l.comm, l.stream.handle), "ncclBroadcast");
    }

    /// Rank 0: the step and ids to every rank; up to `put_ids` ids go by `put` unwaited, longer ones upload and wait.
    pub fn send(l: *Link, put: cuda.Function, h: Header, ids: []const u32) !void {
        if (ids.len > l.ids.len) return error.TooManyIds;
        var hh = h;
        hh.n = @intCast(ids.len);
        if (!l.wait and ids.len <= put_ids) {
            var w: Words = undefined;
            @memcpy(w.w[0..16], std.mem.bytesAsSlice(u32, std.mem.asBytes(&hh)));
            @memcpy(w.w[16..][0..ids.len], ids);
            var a: cuda.Args = .{};
            a.add(l.buf.ptr);
            a.add(@as(u32, @intCast(16 + ids.len)));
            a.add(w);
            try cuda.launch.launch(put, .{ .grid = .{ .x = 1 }, .block = .{ .x = 128 } }, l.stream, &a);
            try l.post(64);
            if (ids.len > 0) try l.post(64 + ids.len * 4);
            return;
        }
        try l.buf.uploadAsync(0, std.mem.asBytes(&hh), l.stream.handle);
        try l.bcast(64);
        if (ids.len > 0) {
            try l.buf.uploadAsync(64, std.mem.sliceAsBytes(ids), l.stream.handle);
            try l.bcast(64 + ids.len * 4);
        }
    }

    /// Other ranks: the next step and its ids (valid until the next receive).
    pub fn rows(l: *Link, ptr: u64, bytes: usize) !void {
        if (bytes == 0) return;
        try l.lib.check(l.lib.api.ncclBroadcast(ptr, ptr, bytes, .u8, 0, l.comm, l.stream.handle), "ncclBroadcast");
    }

    pub fn receive(l: *Link) !struct { h: Header, ids: []const u32 } {
        try l.bcast(64);
        var h: Header = undefined;
        try l.buf.download(0, std.mem.asBytes(&h));
        if (h.n > l.ids.len) return error.TooManyIds;
        if (h.n > 0) {
            try l.bcast(64 + h.n * 4);
            try l.buf.download(64, std.mem.sliceAsBytes(l.ids[0..h.n]));
        }
        return .{ .h = h, .ids = l.ids[0..h.n] };
    }

    pub fn features(l: *Link, buffer: cuda.DeviceBuffer) !void {
        try l.lib.check(l.lib.api.ncclBroadcast(buffer.ptr, buffer.ptr, buffer.len, .u8, 0, l.comm, l.stream.handle), "ncclBroadcast vision features");
        try l.stream.synchronize();
    }

    pub fn grammarBits(l: *Link, s: *sampler.Sampler, row_count: usize) !void {
        if (row_count == 0 or row_count > s.rows) return error.InvalidGrammarMasks;
        const bytes = row_count * ((s.vocab + 31) / 32) * 4;
        try l.lib.check(l.lib.api.ncclBroadcast(s.bits.ptr, s.bits.ptr, bytes, .u8, 0, l.comm, l.stream.handle), "ncclBroadcast grammar masks");
        try l.stream.synchronize();
    }
};

/// Rank 0's engine: every step goes to the other ranks first.
pub const Leader = struct {
    e: *Engine,
    link: *Link,

    /// The step on the bound stream's sequence (its slot) to every rank.
    fn send(x: Leader, h: Header, ids: []const u32) !void {
        var hh = h;
        hh.slot = if (h.kind == @backingInt(Kind.release)) h.slot else x.e.slot;
        if (ids.len > x.link.ids.len) return error.TooManyIds; // no command submitted: a caller refusal, not paired failure
        errdefer if (x.e.pool != null or x.e.dflash != null or x.e.drafter != null) x.e.pair_failure(error.PairedCommandFailed);
        try x.link.send(x.e.m.glue.put_words, hh, ids);
    }

    /// A prompt step's flags: 1 when its rows go through the head (the other ranks join each pass's broadcasts).
    fn promptFlags(x: Leader) u32 {
        return @intFromBool(x.e.prompt_logprobs != null);
    }

    /// The stream's sequence for the steps that follow (each step names it, so the other ranks bind it too).
    pub fn bind(x: Leader, slot: u32) !void {
        try x.e.bind(slot);
    }

    pub fn persist(x: Leader, command: engine.persistence.Command) !void {
        const kind: Kind = switch (command.operation) {
            .write => .spill_write,
            .read => .spill_read,
            .forget => .spill_forget,
        };
        try x.send(.{ .kind = @backingInt(kind), .a = command.id, .seed = command.key, .depth = command.at }, command.tokens);
        try x.e.persist(command);
    }

    /// Rank 0 chooses a physical extent before the paired begin/restore command.
    pub fn reserve(x: Leader, rows: usize) !void {
        const r = try x.e.placement(rows) orelse return;
        try x.flush();
        try x.send(.{ .kind = @backingInt(Kind.reserve), .a = r.base, .seed = r.rows }, &.{});
        try x.e.reservePaired(x.e.slot, r);
    }

    fn complete(x: Leader, result: anytype) @TypeOf(result) {
        return result catch |err| {
            x.failed(err);
            return err;
        };
    }

    fn failed(x: Leader, err: anyerror) void {
        if (x.e.pool == null and x.e.dflash == null and x.e.drafter == null) return;
        switch (err) {
            error.Cancelled, error.KeptStateRefused => {},
            else => x.e.pair_failure(err),
        }
    }

    /// Frees a stream's sequence on every rank.
    pub fn release(x: Leader, slot: u32) !void {
        try x.flush();
        try x.send(.{ .kind = @backingInt(Kind.release), .slot = slot }, &.{});
        x.e.release(slot);
    }

    /// States rank 0 dropped since the last step, so the other ranks free theirs (ids that fit one step at a time).
    pub fn flush(x: Leader) !void {
        if (x.e.disk) |p| try p.flush(p.ctx);
        const d = &x.e.dropped;
        while (d.items.len > 0) {
            const n = @min(d.items.len, x.link.ids.len);
            try x.send(.{ .kind = @backingInt(Kind.drop) }, d.items[0..n]);
            d.replaceRangeAssumeCapacity(0, n, &.{});
        }
    }

    pub fn begin(x: Leader, s: ?sampler.Sampling) !void {
        try x.flush();
        const c = sampler.check(s);
        var h: Header = .{ .kind = @backingInt(Kind.begin) };
        if (c) |v| {
            h.flags = 1;
            h.seed = v.seed;
            h.temperature = v.temperature;
            h.top_p = v.top_p;
            h.min_p = v.min_p;
            h.top_k = v.top_k;
        }
        try x.send(h, &.{});
        try x.complete(x.e.begin(c));
    }

    pub fn vision(x: Leader, service: *vision_runtime.Service, encoded: []const []const u8, ids: []const u32, image_token: u32, replica: u64) !void {
        if (x.e.cur.vision != null) return error.VisionAlreadyBound;
        const state = try service.encode(encoded, ids, replica, @intCast(x.e.share.world));
        x.e.cur.vision = .{ .leader = state };
        errdefer x.e.release(x.e.slot);
        const table = x.e.cur.vision.?.table();
        if (table.image_token != image_token) return error.VisionTokenMismatch;
        try x.send(.{ .kind = @backingInt(Kind.vision), .depth = @intCast(table.rows), .a = replica, .seed = image_token }, ids);
        if (!try x.e.pairedReady(true)) return error.PeerVisionUnavailable;
        try x.link.features(table.features);
    }

    /// `cancel` (rank 0's), asked before each chunk, decides for every rank; `from` every rank holds; `marks` rank 0's.
    pub fn images(x: Leader, v: *const images_mod.Images) !void {
        try x.flush();
        const words = try spanWords(v.gpa, v.spans.items, x.link.ids.len);
        defer v.gpa.free(words);
        try x.send(.{ .kind = @backingInt(Kind.image) }, words);
        try x.link.rows(v.rows.ptr, v.rowCount() * vcfg.out_hidden * 2);
    }

    pub fn prefill(x: Leader, ids: []const u32, from: ?engine.Resume, cancel: ?engine.Cancel, marks: ?engine.Marks) !u32 {
        try x.flush();
        try x.send(.{ .kind = @backingInt(Kind.prefill), .flags = x.promptFlags(), .a = if (from) |r| r.id else 0, .seed = if (from) |r| r.at else 0 }, ids);
        return x.complete(x.e.prefill(ids, from, cancel, marks));
    }

    /// One chunk of a prompt pass on every rank (Engine.prefillChunk); the first token once the chunk ends the prompt.
    pub fn chunk(x: Leader, ids: []const u32, start: usize, from: ?engine.Resume, cancel: ?engine.Cancel, marks: ?engine.Marks) !?u32 {
        if (start >= ids.len) return error.BadChunk;
        try x.flush();
        const part = ids[start..@min(ids.len, start + engine.prompt_rows + 1)];
        try x.send(.{ .kind = @backingInt(Kind.chunk), .flags = x.promptFlags(), .a = if (from) |r| r.id else 0, .seed = if (from) |r| r.at else 0, .depth = @intCast(start), .top_k = @intCast(ids.len) }, part);
        return x.complete(x.e.prefillChunk(part, start, ids.len, from, cancel, marks));
    }

    /// `bits`: the rows' grammar words (structured output, with Engine.structured), else 0.
    pub fn verify(x: Leader, ids: []const u32, out: []u32, bits: u64) !void {
        try x.flush();
        try x.send(.{ .kind = @backingInt(Kind.verify), .flags = @intFromBool(x.e.structured) }, ids);
        try x.complete(x.e.verify(ids, out, bits));
    }

    pub fn firstMasked(x: Leader, position: u64, bits: []const u32) !u32 {
        try sampler.validateMasks(bits, 1, x.e.c.vocab);
        try x.flush();
        try x.e.smp.bits.uploadAsync(0, std.mem.sliceAsBytes(bits), x.e.stream.handle);
        try x.send(.{ .kind = @backingInt(Kind.redraw_masked), .a = position }, &.{});
        try x.complete(x.link.grammarBits(&x.e.smp, 1));
        return x.complete(x.e.redrawQueued(position));
    }

    pub fn verifyMasked(x: Leader, ids: []const u32, out: []u32, bits: []const u32) !void {
        if (ids.len > x.e.targetRows()) return error.WindowTooWide;
        try sampler.validateMasks(bits, ids.len, x.e.c.vocab);
        try x.flush();
        try x.e.smp.bits.uploadAsync(0, std.mem.sliceAsBytes(bits), x.e.stream.handle);
        try x.send(.{ .kind = @backingInt(Kind.verify_masked) }, ids);
        try x.complete(x.link.grammarBits(&x.e.smp, ids.len));
        try x.complete(x.e.verifyMasked(ids, out));
    }

    /// Several streams' windows in one forward on every rank (Engine.verifyShared): parts' slots and rows, then ids.
    pub fn verifyShared(x: Leader, parts: []const engine.Part) !void {
        if (parts.len == 0 or parts.len > engine.max_shared) return error.WindowTooWide;
        var rows: usize = 0;
        for (parts) |p| {
            if (p.ids.len == 0 or p.ids.len > x.e.targetRows() or p.ids.len > x.e.sharedCapacity() -| rows) return error.WindowTooWide;
            rows += p.ids.len;
        }
        try x.flush();
        var buf: [2 * engine.max_shared + engine.shared_rows]u32 = undefined;
        var n: usize = 2 * parts.len;
        for (parts, 0..) |p, i| {
            if (n + p.ids.len > buf.len) return error.WindowTooWide;
            buf[2 * i] = p.slot;
            buf[2 * i + 1] = @intCast(p.ids.len);
            @memcpy(buf[n..][0..p.ids.len], p.ids);
            n += p.ids.len;
        }
        try x.send(.{ .kind = @backingInt(Kind.shared), .flags = @intFromBool(x.e.structured), .depth = @intCast(parts.len) }, buf[0..n]);
        try x.complete(x.e.verifyShared(parts));
    }

    pub fn prefillMany(x: Leader, parts: []const engine.Prompt, firsts: []u32) !void {
        var buf: [prompt_wire.words]u32 = undefined;
        const ids = try prompt_wire.encode(parts, x.e.capacity, &buf);
        try x.flush();
        try x.send(.{ .kind = @backingInt(Kind.many), .depth = @intCast(parts.len) }, ids);
        try x.complete(x.e.prefillMany(parts, firsts));
    }

    pub fn block(x: Leader, reqs: []dflash.Req) !void {
        var buf: [3 * dflash.MAX_S]u32 = undefined;
        const encoded = try encodeBlock(reqs, &buf);
        try x.e.validateDraftBlock(reqs); // every local user/slot refusal precedes command submission
        try x.flush();
        try x.send(.{ .kind = @backingInt(Kind.block), .depth = @intCast(reqs.len), .flags = encoded.proposals }, encoded.ids);
        try x.complete(x.e.draftBlock(reqs));
    }

    pub fn keep(x: Leader, n: usize) !void {
        try x.flush();
        try x.send(.{ .kind = @backingInt(Kind.keep), .a = n }, &.{});
        try x.complete(x.e.keep(n));
    }

    pub fn prefillShared(x: Leader, parts: []const engine.Engine.PromptPart, out: []u32) !void {
        if (parts.len < 2 or parts.len > engine.max_shared or out.len != parts.len) return error.BadPromptBatch;
        var words: [engine.prompt_rows + 2 * engine.max_shared]u32 = undefined;
        var at = 2 * parts.len;
        for (parts, 0..) |p, i| {
            if (p.ids.len == 0 or p.ids.len > words.len - at) return error.BadPromptBatch;
            words[2 * i] = p.slot;
            words[2 * i + 1] = @intCast(p.ids.len);
            @memcpy(words[at..][0..p.ids.len], p.ids);
            at += p.ids.len;
        }
        try x.flush();
        try x.send(.{ .kind = @backingInt(Kind.prefill_shared), .depth = @intCast(parts.len) }, words[0..at]);
        try x.complete(x.e.prefillShared(parts, out));
    }

    /// Structured output's first token: every rank gathers the prompt's last row again, rank 0 draws it among `bits`.
    pub fn redraw(x: Leader, len: usize, bits: u64) !u32 {
        try x.flush();
        try x.send(.{ .kind = @backingInt(Kind.redraw), .a = len }, &.{});
        return x.complete(x.e.redrawFirst(len, bits));
    }

    pub fn draft(x: Leader, from_prompt: bool, follow_ids: []const u32, depth: usize, out: []u32) !usize {
        try x.flush();
        try x.send(.{ .kind = @backingInt(Kind.draft), .depth = @intCast(depth), .flags = @intFromBool(from_prompt) }, follow_ids);
        return x.complete(x.e.draft(from_prompt, follow_ids, depth, out));
    }

    pub fn draftBatch(x: Leader, parts: []const engine.DraftPart, out: []engine.DraftResult) !void {
        if (out.len != parts.len) return error.BadDraftBatch;
        try x.complete(x.e.validateDraftBatch(parts));
        var words: [engine.max_shared * 5]u32 = undefined;
        const payload = try encodeDraftBatch(parts, &words);
        try x.flush();
        try x.send(.{ .kind = @backingInt(Kind.draft_batch), .depth = @intCast(parts.len) }, payload);
        try x.complete(x.e.draftBatch(parts, out));
    }

    pub fn prelaunchBatch(x: Leader, parts: []const engine.DraftPart) !void {
        try x.complete(x.e.validateDraftBatch(parts));
        var words: [engine.max_shared * 5]u32 = undefined;
        const payload = try encodeDraftBatch(parts, &words);
        try x.flush();
        try x.send(.{ .kind = @backingInt(Kind.prelaunch_batch), .depth = @intCast(parts.len) }, payload);
        try x.complete(x.e.prelaunchDraftBatch(parts));
    }

    /// The other ranks leave their loops.
    pub fn end(x: Leader) !void {
        try x.send(.{ .kind = @backingInt(Kind.end) }, &.{});
    }
};

/// A prompt step whose rows go through the head (Leader.rows): this rank's columns join each pass's broadcasts.
fn promptRows(h: Header) bool {
    const kind: Kind = @fromBackingInt(@intCast(h.kind));
    return (kind == .prefill or kind == .chunk) and h.flags & 1 != 0;
}

/// A verify step under a grammar (Engine.structured): this rank's columns join the gather of each draw's whole rows.
fn structuredStep(h: Header) bool {
    const kind: Kind = @fromBackingInt(@intCast(h.kind));
    return (kind == .verify or kind == .shared) and h.flags & 1 != 0;
}

/// A rank other than 0 runs rank 0's steps until it ends; a cancelled prefill is error.Cancelled and the loop goes on.
pub fn follow(e: *Engine, link: *Link) !void {
    var drafts: [engine.max_drafts]u32 = undefined;
    var sampled: [16]u32 = undefined;
    while (true) {
        const step = try link.receive();
        const h = step.h;
        const kind: Kind = @fromBackingInt(@intCast(h.kind));
        if (kind == .spill_write or kind == .spill_read or kind == .spill_forget) {
            const operation: engine.persistence.Operation = switch (kind) {
                .spill_write => .write,
                .spill_read => .read,
                .spill_forget => .forget,
                else => unreachable,
            };
            e.persist(.{ .operation = operation, .id = h.a, .key = h.seed, .at = h.depth, .tokens = step.ids }) catch |err| switch (err) {
                error.NotStored, error.SpillRefused => {},
                else => return err,
            };
            continue;
        }
        if (kind == .reserve) {
            e.reservePaired(h.slot, .{ .base = @intCast(h.a), .rows = @intCast(h.seed) }) catch |err| switch (err) {
                error.KvReservationRefused => {},
                else => return err,
            };
            continue;
        }
        if (kind == .vision) {
            var failure: ?anyerror = null;
            e.bind(h.slot) catch |err| {
                failure = err;
            };
            const rows: usize = h.depth;
            var ids: []u32 = undefined;
            var ids_owned = false;
            if (failure == null) {
                if (e.gpa.dupe(u32, step.ids)) |copy| {
                    ids = copy;
                    ids_owned = true;
                } else |err| failure = err;
            }
            var positions: []u32 = undefined;
            var positions_owned = false;
            var features: cuda.DeviceBuffer = undefined;
            if (failure == null) {
                if (rows == 0 or rows > 16384 or ids.len > e.capacity or h.seed >= e.c.vocab or e.cur.vision != null) failure = error.BadVisionCommand;
            }
            if (failure == null) {
                if (e.gpa.alloc(u32, rows)) |copy| {
                    positions = copy;
                    positions_owned = true;
                } else |err| failure = err;
            }
            if (failure == null) {
                var at: usize = 0;
                for (ids, 0..) |id, i| if (id == h.seed) {
                    if (at >= rows) {
                        failure = error.BadVisionPositions;
                        break;
                    }
                    positions[at] = @intCast(i);
                    at += 1;
                };
                if (at != rows) failure = error.BadVisionPositions;
            }
            if (failure == null) {
                features = cuda.DeviceBuffer.alloc(e.d, rows * vision_binding.width * 2) catch |err| blk: {
                    failure = err;
                    break :blk undefined;
                };
            }
            const ready = try e.pairedReady(failure == null);
            if (!ready) {
                if (failure == null) features.free();
                if (positions_owned) e.gpa.free(positions);
                if (ids_owned) e.gpa.free(ids);
                if (failure) |err| std.log.warn("vision request refused on follower: {s}", .{@errorName(err)});
                continue;
            }
            var context: cuda.abi.Context = null;
            try e.d.check(e.d.api.cuCtxGetCurrent(&context), "cuCtxGetCurrent");
            e.cur.vision = .{ .follower = .{ .table = .{ .driver = e.d, .context = context, .stream = e.stream, .features = features, .positions = positions, .ids = ids, .image_token = @intCast(h.seed), .rows = rows, .ready = false, .generation = 1, .replica = h.a, .world = @intCast(e.share.world), .rank = @intCast(e.share.rank) }, .ids = ids, .positions = positions } };
            try link.features(features);
            e.cur.vision.?.table().ready = true;
            _ = try e.cur.vision.?.table().binding();
            continue;
        }
        if (kind == .draft_batch) {
            var parts_buf: [engine.max_shared]engine.DraftPart = undefined;
            const parts = decodeDraftBatch(step.ids, h.depth, &parts_buf) catch |err| {
                _ = try e.pairedReady(false);
                return err;
            };
            var results: [engine.max_shared]engine.DraftResult = undefined;
            try e.draftBatch(parts, results[0..parts.len]);
            continue;
        }
        if (kind == .prelaunch_batch) {
            var parts_buf: [engine.max_shared]engine.DraftPart = undefined;
            const parts = decodeDraftBatch(step.ids, h.depth, &parts_buf) catch |err| {
                _ = try e.pairedReady(false);
                return err;
            };
            try e.prelaunchDraftBatch(parts);
            continue;
        }
        if (kind != .end and kind != .drop and kind != .shared and kind != .prefill_shared and kind != .release and kind != .block and kind != .many) try e.bind(h.slot);
        e.prompt_logprobs = if (promptRows(h)) .{} else null;
        e.structured = structuredStep(h);
        switch (kind) {
            .begin => try e.begin(if (h.flags & 1 != 0) .{ .seed = h.seed, .temperature = h.temperature, .top_p = h.top_p, .min_p = h.min_p, .top_k = h.top_k } else null),
            .prefill => {
                const ids = try e.gpa.dupe(u32, step.ids);
                defer e.gpa.free(ids);
                const from: ?engine.Resume = if (h.a != 0) .{ .id = h.a, .at = h.seed } else null;
                _ = e.prefill(ids, from, null, null) catch |err| switch (err) {
                    error.Cancelled, error.KeptStateRefused => {},
                    else => return err,
                };
            },
            .chunk => {
                const ids = try e.gpa.dupe(u32, step.ids);
                defer e.gpa.free(ids);
                const from: ?engine.Resume = if (h.a != 0) .{ .id = h.a, .at = h.seed } else null;
                _ = e.prefillChunk(ids, h.depth, h.top_k, from, null, null) catch |err| switch (err) {
                    error.Cancelled, error.KeptStateRefused => {},
                    else => return err,
                };
            },
            .verify => try e.verify(step.ids, sampled[0..step.ids.len], 0),
            .redraw_masked => {
                try link.grammarBits(&e.smp, 1);
                _ = try e.redrawQueued(h.a);
            },
            .verify_masked => {
                try link.grammarBits(&e.smp, step.ids.len);
                try e.verifyMasked(step.ids, sampled[0..step.ids.len]);
            },
            .reserve, .spill_write, .spill_read, .spill_forget, .vision, .draft_batch, .prelaunch_batch => unreachable,
            .redraw => _ = try e.redrawFirst(@intCast(h.a), 0),
            .shared => {
                const k = h.depth;
                if (k == 0 or k > engine.max_shared or step.ids.len < 2 * k) return error.BadStep;
                var parts: [engine.max_shared]engine.Part = undefined;
                var outs: [engine.shared_rows]u32 = undefined;
                var at: usize = 2 * k;
                var row: usize = 0;
                for (parts[0..k], 0..) |*p, i| {
                    const n = step.ids[2 * i + 1];
                    if (at + n > step.ids.len or row + n > outs.len) return error.BadStep;
                    p.* = .{ .slot = step.ids[2 * i], .ids = step.ids[at..][0..n], .out = outs[row..][0..n] };
                    at += n;
                    row += n;
                }
                try e.verifyShared(parts[0..k]);
            },
            .prefill_shared => {
                const k = h.depth;
                if (k < 2 or k > engine.max_shared or step.ids.len < 2 * k) return error.BadPromptBatch;
                var parts: [engine.max_shared]engine.Engine.PromptPart = undefined;
                var out: [engine.max_shared]u32 = undefined;
                var at: usize = 2 * k;
                for (parts[0..k], 0..) |*p, i| {
                    const n = step.ids[2 * i + 1];
                    if (n == 0 or n > step.ids.len - at) return error.BadPromptBatch;
                    p.* = .{ .slot = step.ids[2 * i], .ids = step.ids[at..][0..n] };
                    at += n;
                }
                if (at != step.ids.len) return error.BadPromptBatch;
                try e.prefillShared(parts[0..k], out[0..k]);
            },
            .many => {
                var parts: [engine.max_together]engine.Prompt = undefined;
                const decoded = try prompt_wire.decode(step.ids, h.depth, e.capacity, &parts);
                var firsts: [engine.max_together]u32 = undefined;
                try e.prefillMany(decoded, &firsts);
            },
            .keep => try e.keep(@intCast(h.a)),
            .draft => {
                const ids = try e.gpa.dupe(u32, step.ids);
                defer e.gpa.free(ids);
                _ = try e.draft(h.flags & 1 != 0, ids, h.depth, &drafts);
            },
            .block => try dispatchBlock(e, h, step.ids),
            .image => {
                const v = e.images orelse return error.NoVision;
                try v.take(h.slot, step.ids);
                try link.rows(v.rows.ptr, v.rowCount() * vcfg.out_hidden * 2);
            },
            .drop => for (step.ids) |id| e.dropState(id),
            .release => e.release(h.slot),
            .end => return,
        }
    }
}

test "a step's header is 64 bytes and keeps its fields" {
    const h: Header = .{ .kind = @backingInt(Kind.draft), .n = 3, .depth = 2, .flags = 1, .a = 7, .seed = 1 << 62, .temperature = 0.6, .top_p = 0.95, .top_k = 20 };
    const bytes = std.mem.asBytes(&h);
    var back: Header = undefined;
    @memcpy(std.mem.asBytes(&back), bytes);
    try std.testing.expectEqual(h, back);
    try std.testing.expectEqual(Kind.draft, @as(Kind, @fromBackingInt(@intCast(back.kind))));
}

test "draft batch wire preserves unordered slots and positions and refuses a later duplicate" {
    const parts = [_]engine.DraftPart{
        .{ .slot = 5, .position = 1 << 33, .pending = 11, .depth = 3 },
        .{ .slot = 1, .position = 91, .pending = 22, .depth = 0 },
    };
    var words: [engine.max_shared * 5]u32 = undefined;
    const wire = try encodeDraftBatch(&parts, &words);
    try std.testing.expectEqual(@as(usize, 10), wire.len);
    var decoded: [engine.max_shared]engine.DraftPart = undefined;
    try std.testing.expectEqualSlices(engine.DraftPart, &parts, try decodeDraftBatch(wire, 2, &decoded));
    words[5] = 5;
    try std.testing.expectError(error.BadDraftSlot, decodeDraftBatch(words[0..10], 2, &decoded));
    try std.testing.expectError(error.BadDraftBatch, decodeDraftBatch(words[0..9], 2, &decoded));
}

test "leader draft state refusal fails the pair before sending a command" {
    const Failed = struct {
        var reason: ?anyerror = null;
        fn fail(err: anyerror) void {
            reason = err;
        }
    };
    var e: Engine = undefined;
    e.settings = .{};
    e.pool = null;
    e.dflash = @ptrFromInt(256);
    e.seqs = .empty;
    e.pair_failure = Failed.fail;
    var link: Link = undefined;
    const leader: Leader = .{ .e = &e, .link = &link };
    var out: [1]engine.DraftResult = .{.{ .count = 99 }};
    Failed.reason = null;
    try std.testing.expectError(error.BadDraftSlot, leader.draftBatch(&.{.{ .slot = 1, .position = 1, .pending = 7, .depth = 1 }}, &out));
    try std.testing.expectEqual(@as(?anyerror, error.BadDraftSlot), Failed.reason);
    try std.testing.expectEqual(@as(usize, 99), out[0].count);
}
test "flag 1 asks for prompt rows on prefill and chunk steps only (begin's is sampling, draft's from the prompt)" {
    try std.testing.expect(promptRows(.{ .kind = @backingInt(Kind.chunk), .flags = 1, .depth = 2048, .top_k = 4000 }));
    try std.testing.expect(promptRows(.{ .kind = @backingInt(Kind.prefill), .flags = 1 }));
    try std.testing.expect(!promptRows(.{ .kind = @backingInt(Kind.chunk), .depth = 2048, .top_k = 4000 }));
    try std.testing.expect(!promptRows(.{ .kind = @backingInt(Kind.begin), .flags = 1, .seed = 7 }));
    try std.testing.expect(!promptRows(.{ .kind = @backingInt(Kind.draft), .flags = 1, .depth = 3 }));
}

test "flag 1 on verify and shared steps draws from whole rows; a prompt step's flag 1 does not" {
    try std.testing.expect(structuredStep(.{ .kind = @backingInt(Kind.verify), .flags = 1, .n = 4 }));
    try std.testing.expect(structuredStep(.{ .kind = @backingInt(Kind.shared), .flags = 1, .depth = 2 }));
    try std.testing.expect(!structuredStep(.{ .kind = @backingInt(Kind.verify), .n = 4 }));
    try std.testing.expect(!structuredStep(.{ .kind = @backingInt(Kind.chunk), .flags = 1, .depth = 2048 }));
    try std.testing.expect(!structuredStep(.{ .kind = @backingInt(Kind.begin), .flags = 1 }));
}

pub fn spanWords(a: std.mem.Allocator, spans: []const images_mod.Span, max_words: usize) ![]u32 {
    const n = std.math.mul(usize, spans.len, 3) catch return error.TooManyIds;
    if (n > max_words) return error.TooManyIds;
    const words = try a.alloc(u32, n);
    for (spans, 0..) |sp, i| words[3 * i ..][0..3].* = .{ sp.at, sp.n, sp.row };
    return words;
}

test "128 video pairs cross the span serializer and a follower's receiver; one word short is refused" {
    const a = std.testing.allocator;
    var spans: [128]images_mod.Span = undefined;
    for (&spans, 0..) |*sp, i| sp.* = .{ .at = @intCast(i * 2), .n = 1, .row = @intCast(i) };
    const words = try spanWords(a, &spans, 3 * spans.len);
    defer a.free(words);
    var v: images_mod.Images = undefined;
    v.gpa = a;
    v.spans = .empty;
    v.slot = 9;
    defer v.spans.deinit(a);
    v.limits = .{ .image_tokens = 64, .max_images = 8, .video_tokens = 128, .max_frames = 256 };
    v.video_ids = .{ 1, 2, 3 };
    try v.take(9, words);
    try std.testing.expectEqualSlices(images_mod.Span, &spans, v.spans.items);
    try std.testing.expectEqual(@as(u32, 9), v.slot);
    try std.testing.expectEqual(@as(usize, 128), v.rowCount());
    try std.testing.expectError(error.TooManyIds, spanWords(a, &spans, 3 * spans.len - 1));
}

test "a link whose host allocation fails frees its device buffer" {
    const Mock = struct {
        var frees: usize = 0;
        fn alloc(ptr: *cuda.abi.DevicePtr, _: usize) callconv(.c) cuda.abi.Result {
            ptr.* = 4096;
            return 0;
        }
        fn free(_: cuda.abi.DevicePtr) callconv(.c) cuda.abi.Result {
            frees += 1;
            return 0;
        }
    };
    Mock.frees = 0;
    var d: cuda.Driver = undefined;
    d.api.cuMemAlloc_v2 = Mock.alloc;
    d.api.cuMemFree_v2 = Mock.free;
    var lib: cuda.nccl.Library = undefined;
    var failing = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = 0 });
    const before = cuda.usage(false).device;
    defer {
        if (Mock.frees == 0) {
            var leaked: cuda.DeviceBuffer = .{ .d = &d, .ptr = 4096, .len = 64 + 128 * 4 };
            leaked.free();
        }
    }
    try std.testing.expectError(error.OutOfMemory, Link.init(failing.allocator(), &d, &lib, null, .{ .d = &d, .handle = null }, 128));
    try std.testing.expectEqual(@as(usize, 1), Mock.frees);
    try std.testing.expectEqual(before, cuda.usage(false).device);
}

// Real Leader/Link submission and follower block dispatch with CUDA/NCCL command recordings.
// No numerical drafts are manufactured: out=null runs the actual device proposal/collective bodies only.
const CanonicalCommands = struct {
    const Memory = @import("cuda_test_driver.zig").Memory;
    var buffer: u64 = 0;
    var header: Header = undefined;
    var ids: [3 * dflash.MAX_S]u32 = undefined;
    var broadcasts: usize = 0;
    var gathers: usize = 0;
    var proposals: usize = 0;
    var absorbed: usize = 0;
    var fail_send: bool = false;
    var fail_ingest: bool = false;
    var reason: ?anyerror = null;
    var event: usize = 0;
    var failed_at: usize = 0;
    var cleanup_at: usize = 0;
    fn fail(err: anyerror) void {
        reason = err;
        event += 1;
        failed_at = event;
    }
    fn sync(_: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        if (reason != null) {
            std.debug.assert(failed_at > 0);
            event += 1;
            cleanup_at = event;
        }
        return 0;
    }
    fn bcast(src: u64, _: u64, n: usize, _: cuda.nccl.DataType, _: c_int, _: cuda.nccl.Comm, _: cuda.abi.Stream) callconv(.c) cuda.nccl.Result {
        std.debug.assert(reason == null);
        broadcasts += 1;
        event += 1;
        if (fail_send) return 1;
        header = std.mem.bytesToValue(Header, Memory.slice(src, @sizeOf(Header))[0..@sizeOf(Header)]);
        if (n > @sizeOf(Header)) @memcpy(std.mem.sliceAsBytes(ids[0..header.n]), Memory.slice(src + @sizeOf(Header), header.n * 4));
        return 0;
    }
    fn gather(_: u64, _: u64, _: usize, _: cuda.nccl.DataType, _: cuda.nccl.Comm, _: cuda.abi.Stream) callconv(.c) cuda.nccl.Result {
        std.debug.assert(reason == null);
        gathers += 1;
        return 0;
    }
    fn upload(dst: u64, src: ?*const anyopaque, n: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        if (dst >= buffer and dst + n <= buffer + 1024) @memcpy(Memory.slice(dst, n), @as([*]const u8, @ptrCast(src.?))[0..n]);
        return 0;
    }
    fn copy(_: u64, _: u64, n: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        std.debug.assert(reason == null);
        event += 1;
        if (fail_ingest) return cuda.abi.error_not_ready;
        absorbed += n / dflash.TAP_ROW;
        return 0;
    }
    fn launch(f: cuda.abi.Function, x: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: cuda.abi.Stream, _: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
        std.debug.assert(reason == null);
        if (f == @as(cuda.abi.Function, @ptrFromInt(3))) proposals += x / dflash.BLOCK;
        return 0;
    }
    fn launchEx(c: *const cuda.abi.LaunchConfig, f: cuda.abi.Function, args: ?[*]?*anyopaque, extra: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
        return launch(f, c.grid_x, c.grid_y, c.grid_z, c.block_x, c.block_y, c.block_z, c.shared_bytes, c.stream, args, extra);
    }
    fn fill(comptime T: type, d: *const cuda.Driver) T {
        if (T == cuda.Function) return .{ .d = d, .handle = @ptrFromInt(1) };
        if (T == bool) return false;
        var out: T = undefined;
        switch (@typeInfo(T)) {
            .@"struct" => inline for (@typeInfo(T).@"struct".field_names) |name| {
                @field(out, name) = fill(@TypeOf(@field(out, name)), d);
            },
            .array => for (&out) |*item| {
                item.* = fill(@TypeOf(item.*), d);
            },
            else => @compileError("unexpected recording kernel field"),
        }
        return out;
    }
    fn reset() void {
        broadcasts = 0;
        gathers = 0;
        proposals = 0;
        absorbed = 0;
        fail_send = false;
        fail_ingest = false;
        reason = null;
        event = 0;
        failed_at = 0;
        cleanup_at = 0;
    }
};

test "actual canonical Leader.block submission drives follower dispatch for mixed all-proposal and all-absorb cohorts and halts after submitted failures" {
    const C = CanonicalCommands;
    const a = std.testing.allocator;
    var d = C.Memory.driver();
    d.api.cuMemcpyHtoDAsync_v2 = C.upload;
    d.api.cuMemcpyDtoDAsync_v2 = C.copy;
    d.api.cuLaunchKernel = C.launch;
    d.api.cuLaunchKernelEx = C.launchEx;
    d.api.cuStreamSynchronize = C.sync;
    var lib: cuda.nccl.Library = .{ .lib = undefined, .api = undefined };
    lib.api.ncclBroadcast = C.bcast;
    lib.api.ncclAllGather = C.gather;
    var storage = try cuda.DeviceBuffer.alloc(&d, 1024);
    defer storage.free();
    C.buffer = storage.ptr;
    var host_ids: [128]u32 = undefined;
    var link: Link = .{ .d = &d, .lib = &lib, .comm = null, .stream = .{ .d = &d, .handle = null }, .buf = storage, .ids = &host_ids, .gpa = a, .wait = true };
    defer cuda.copy_source.dropSpares(&d, cuda.copy_source.current());
    var dr: dflash.Drafter = .{ .gpa = a, .d = &d, .module = undefined, .k = C.fill(dflash.Kernels, &d), .weights = undefined, .split = true, .ff = dflash.FF / 2, .rank = 0, .world = 2 };
    const dense: @import("cuda_mm.zig").Tiled = .{ .w = 0, .n = dflash.D, .k = dflash.D };
    dr.fc = .{ .bf16 = .{ .w = 0, .n = dflash.D, .k = 5 * dflash.D } };
    dr.hproj = dense;
    for (&dr.layers) |*l| l.* = .{ .in_norm = 0, .post_norm = 0, .qn = 0, .kn = 0, .a_base = 0, .m_base = 0, .qkv = .{ .bf16 = dense }, .kv = .{ .bf16 = dense }, .o = .{ .bf16 = dense }, .gu = .{ .bf16 = dense }, .down = dense, .a_proj = .{ .bf16 = dense }, .m_proj = .{ .bf16 = dense } };
    var positions: [dflash.MAX_CTX]i64 = undefined;
    var rings: [dflash.MAX_CTX]u64 = undefined;
    dr.h_cpos = &positions;
    dr.h_crings = &rings;
    const ring: cuda.DeviceBuffer = .{ .d = &d, .ptr = 0x100000, .len = dflash.ring_bytes, .borrowed = true };
    try dr.slots.appendNTimes(a, .{ .ring = ring, .from = 100 }, 3);
    defer dr.slots.deinit(a);
    var e: Engine = undefined;
    e.settings = .{};
    e.pool = null;
    e.dflash = null;
    e.drafter = &dr;
    e.disk = null;
    e.dropped = .empty;
    e.slot = 0;
    e.d = &d;
    e.stream = link.stream;
    e.lib = &lib;
    e.comm = null;
    e.tp_group = null;
    e.share.rank = 0;
    e.share.vocab.lo = 0;
    var spans: [2]@import("cuda_shard.zig").Span = undefined;
    e.spans = &spans;
    e.m.glue = C.fill(@import("cuda_glue.zig").Kernels, &d);
    e.m.mm = C.fill(@import("cuda_mm.zig").Kernels, &d);
    e.m.q4 = C.fill(@import("cuda_q4.zig").Kernels, &d);
    e.m.glue.embed_b16.handle = @ptrFromInt(3);
    e.w.mtp = @as(@import("cuda_weights.zig").Mtp, undefined);
    e.w.mtp.?.draft = .{ .w = 0, .s = 0, .b = 0, .n = 128, .k = dflash.D, .npad = 128 };
    e.w.embed = 0;
    e.pair_failure = C.fail;
    var seqs: [3]@import("cuda_engine_types.zig").Seq = undefined;
    var slots: [3]?*@import("cuda_engine_types.zig").Seq = undefined;
    for (&seqs, &slots) |*seq, *slot| {
        seq.st.pos = 102;
        slot.* = seq;
    }
    e.seqs = .empty;
    e.seqs.items = &slots;
    e.seqs.capacity = slots.len;
    const leader: Leader = .{ .e = &e, .link = &link };
    for ([_]u32{ 0b000, 0b101, 0b111 }) |mask| {
        C.reset();
        var reqs: [3]dflash.Req = undefined;
        for (&reqs, 0..) |*r, i| r.* = .{ .slot = @intCast(i), .pos = 0, .n = 2, .anchor = @intCast(11 + i), .propose = mask & (@as(u32, 1) << @intCast(i)) != 0 };
        for (0..3) |i| try dr.noteVerify(@intCast(i), 2 * i, 2, 100, i == 0);
        try leader.block(&reqs);
        try std.testing.expectEqual(@as(usize, 2), C.broadcasts);
        try std.testing.expectEqual(mask, C.header.flags); // actual Leader/Link bytes, never a constructed expected input
        const leader_proposals = C.proposals;
        const leader_gathers = C.gathers;
        try std.testing.expectEqual(@as(usize, @popCount(mask)), leader_proposals);
        try std.testing.expectEqual(@as(usize, 6), C.absorbed);
        C.proposals = 0;
        C.gathers = 0;
        C.absorbed = 0;
        e.share.rank = 1;
        try dispatchBlock(&e, C.header, C.ids[0..C.header.n]);
        e.share.rank = 0;
        try std.testing.expectEqual(leader_proposals, C.proposals);
        try std.testing.expectEqual(leader_gathers, C.gathers);
        try std.testing.expectEqual(@as(usize, 6), C.absorbed);
        try std.testing.expectEqual(@as(u64, 0), dr.lost);
        if (mask != 0) try std.testing.expect(leader_gathers >= dflash.LAYERS);
        try e.stream.synchronize();
    }
    // Refuse malformed/user inputs before any command, ingest, collective or paired halt.
    C.reset();
    var reqs = [_]dflash.Req{.{ .slot = 3, .pos = 0, .n = 2, .anchor = 11 }};
    try std.testing.expectError(error.UnknownSlot, leader.block(&reqs));
    try std.testing.expect(C.reason == null and C.broadcasts == 0 and C.absorbed == 0);
    reqs[0].slot = 0;
    for ([_]bool{ true, false }) |send_failure| {
        C.reset();
        C.fail_send = send_failure;
        C.fail_ingest = !send_failure;
        if (send_failure) {
            try std.testing.expectError(error.NcclFailed, leader.block(&reqs));
            try std.testing.expectEqual(@as(?anyerror, error.PairedCommandFailed), C.reason);
            try std.testing.expectEqual(@as(usize, 1), C.broadcasts);
        } else {
            try std.testing.expectError(error.NotReady, leader.block(&reqs));
            try std.testing.expectEqual(@as(?anyerror, error.NotReady), C.reason);
            try std.testing.expectEqual(@as(usize, 2), C.broadcasts);
        }
        try std.testing.expect(C.failed_at > 0);
        // Serving failure terminates the rank; the recorder returns solely to inspect ordering before owner teardown.
        try std.testing.expectEqual(@as(usize, 0), C.cleanup_at);
        try std.testing.expectEqual(@as(usize, 0), C.proposals);
        try std.testing.expectEqual(@as(usize, 0), C.gathers);
        try e.stream.synchronize(); // actual cleanup call crosses the recording driver after paired failure
        try std.testing.expect(C.failed_at < C.cleanup_at);
    }
    cuda.copy_source.dropSpares(&d, cuda.copy_source.current());
    storage.free();
    storage.borrowed = true;
    try std.testing.expect(C.Memory.empty());
}
