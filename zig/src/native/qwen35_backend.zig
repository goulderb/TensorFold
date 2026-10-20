const std = @import("std");
const lanes = @import("lanes");
const be = lanes.backend;
const Allocator = std.mem.Allocator;

pub const facts: lanes.Model = .{
    .exact_width = 1,
    .gpu_tokens = false,
    .mtp = false,
    .hidden_rows = false,
    .streams_exact = false,
    .max_streams = 1,
    .lone_copies = false,
};

pub fn Adapter(comptime Session: type) type {
    return struct {
        const Self = @This();
        gpa: Allocator,
        session: *Session,
        sessions: []Session,
        bound: []?*lanes.Stream,
        prompt_at: []usize,
        capacity: usize,
        /// Prompt tokens per true session.append; default 1 matches FakeSession and legacy Chain.
        prefill_rows: usize = 1,
        active: ?*lanes.Stream = null,
        values: []f64,
        ids: []u64,
        lock: std.atomic.Mutex = .unlocked,

        pub fn init(gpa: Allocator, session: *Session, capacity: usize) !Self {
            return initSessions(gpa, @as(*[1]Session, session), capacity);
        }

        pub fn initSessions(gpa: Allocator, sessions: []Session, capacity: usize) !Self {
            if (sessions.len == 0 or sessions[0].logits.len == 0 or capacity == 0) return error.InvalidSession;
            const values = try gpa.alloc(f64, sessions[0].logits.len);
            errdefer gpa.free(values);
            const ids = try gpa.alloc(u64, sessions[0].logits.len);
            errdefer gpa.free(ids);
            for (ids, 0..) |*id, i| id.* = i;
            const bound = try gpa.alloc(?*lanes.Stream, sessions.len);
            errdefer gpa.free(bound);
            @memset(bound, null);
            const prompt_at = try gpa.alloc(usize, sessions.len);
            @memset(prompt_at, 0);
            return .{ .gpa = gpa, .session = &sessions[0], .sessions = sessions, .bound = bound, .prompt_at = prompt_at, .capacity = capacity, .values = values, .ids = ids };
        }

        pub fn deinit(self: *Self) void {
            self.gpa.free(self.values);
            self.gpa.free(self.ids);
            self.gpa.free(self.bound);
            self.gpa.free(self.prompt_at);
        }

        pub fn backend(self: *Self) be.Backend {
            return .{ .ptr = self, .vtable = &vtable };
        }

        const vtable: be.Backend.VTable = .{
            .prefill = prefill,
            .prefill_step = prefillStep,
            .prefill_batch = prefillBatch,
            .first = first,
            .queue = queue,
            .read = read,
            .verify = verify,
            .keep = keep,
            .draft = draft,
            .release = release,
            .reports_fill = true,
        };

        fn of(ptr: *anyopaque) *Self {
            return @ptrCast(@alignCast(ptr));
        }

        pub fn boundIndex(self: *const Self, s: *lanes.Stream) ?usize {
            for (self.bound, 0..) |b, i| if (b == s) return i;
            if (self.active == s) return 0;
            return null;
        }

        fn indexOf(self: *Self, s: *lanes.Stream) ?usize {
            return self.boundIndex(s);
        }

        fn sessionOf(self: *Self, s: *lanes.Stream) !*Session {
            const i = self.indexOf(s) orelse return error.PositionMismatch;
            return &self.sessions[i];
        }

        fn acquire(self: *Self, s: *lanes.Stream) !usize {
            if (self.indexOf(s)) |i| return i;
            for (self.bound, 0..) |*b, i| if (b.* == null) {
                b.* = s;
                self.prompt_at[i] = 0;
                if (i == 0) self.active = s;
                return i;
            };
            return error.LaneMemory;
        }

        fn draw(self: *Self, s: *lanes.Stream, position: u64) !u64 {
            const session = try self.sessionOf(s);
            if (session.position != position) return error.PositionMismatch;
            const sampling: lanes.Sampling = s.sampling orelse .{ .seed = 0, .temperature = 0 };
            if (sampling.temperature <= 0) {
                if (try session.cachedGreedy(position)) |token| return token;
            }
            for (self.values, session.logits) |*value, bits|
                value.* = @as(f32, @bitCast(@as(u32, bits) << 16));
            return lanes.sampling.choose(self.gpa, self.values, self.ids, position, sampling);
        }

        fn begin(self: *Self, s: *lanes.Stream) !void {
            if (self.indexOf(s) != null) return;
            if (s.prompt_len == 0 or s.prompt_len > self.capacity or s.max_new > self.capacity - s.prompt_len)
                return error.ContextFull;
            if (s.vision != null or s.images.len > 0 or s.grammar != null or
                s.logprobs != null or s.prompt_logprobs != null or s.think_budget != 0 or
                s.decode_spans.len > 0)
                return error.UnsupportedRequest;
            for (s.context.items[0..s.prompt_len]) |id| if (id >= self.ids.len) return error.InvalidToken;
            const i = try self.acquire(s);
            const session = &self.sessions[i];
            if (s.reuse.saved) |saved| {
                if (s.reuse.at == 0 or s.reuse.at >= s.prompt_len) {
                    s.reuse_failed = true;
                    try session.reset();
                    self.prompt_at[i] = 0;
                    s.cached = 0;
                } else if (comptime @hasDecl(Session, "restoreKept")) {
                    if (session.restoreKept(saved)) {
                        self.prompt_at[i] = s.reuse.at;
                        s.cached = s.reuse.at;
                    } else |_| {
                        s.reuse_failed = true;
                        try session.reset();
                        self.prompt_at[i] = 0;
                        s.cached = 0;
                    }
                } else {
                    session.position = s.reuse.at;
                    self.prompt_at[i] = s.reuse.at;
                    s.cached = s.reuse.at;
                }
            } else {
                try session.reset();
                self.prompt_at[i] = 0;
                s.cached = 0;
            }
        }

        fn prefill(ptr: *anyopaque, s: *lanes.Stream) anyerror!void {
            while (!try prefillStep(ptr, s)) {}
        }

        fn prefillStep(ptr: *anyopaque, s: *lanes.Stream) anyerror!bool {
            const self = of(ptr);
            while (!self.lock.tryLock()) std.Thread.yield() catch {};
            defer self.lock.unlock();
            if (s.isCancelled()) return error.Cancelled;
            if (self.indexOf(s) == null) try self.begin(s);
            const i = self.indexOf(s) orelse return error.PositionMismatch;
            const session = &self.sessions[i];
            if (self.prefill_rows == 0) return error.InvalidSession;
            if (self.prompt_at[i] < s.prompt_len) {
                const first_chunk = self.prompt_at[i] == 0;
                const n = @min(self.prefill_rows, s.prompt_len - self.prompt_at[i]);
                try session.append(s.context.items[self.prompt_at[i]..][0..n]);
                self.prompt_at[i] += n;
                s.fill_at.store(@intCast(self.prompt_at[i]), .monotonic);
                if (first_chunk) std.log.info("qwen prefill first chunk rows={d} position={d}", .{ n, session.position });
                if (self.prompt_at[i] < s.prompt_len and std.mem.indexOfScalar(u32, s.reuse.marks, @intCast(self.prompt_at[i])) != null) {
                    if (s.reuse.hook) |k| k.at(k.ptr, s, @intCast(self.prompt_at[i]));
                }
            }
            return self.prompt_at[i] == s.prompt_len;
        }

        fn first(ptr: *anyopaque, s: *lanes.Stream, position: u64) anyerror!u64 {
            const self = of(ptr);
            while (!self.lock.tryLock()) std.Thread.yield() catch {};
            defer self.lock.unlock();
            return self.draw(s, position);
        }

        fn queue(ptr: *anyopaque, s: *lanes.Stream, feed: be.Feed, position: u64) anyerror!u64 {
            const self = of(ptr);
            while (!self.lock.tryLock()) std.Thread.yield() catch {};
            defer self.lock.unlock();
            return self.queueLocked(ptr, s, feed, position);
        }

        fn queueLocked(self: *Self, ptr: *anyopaque, s: *lanes.Stream, feed: be.Feed, position: u64) !u64 {
            const session = try self.sessionOf(s);
            if (session.position + 1 != position) return error.PositionMismatch;
            const token = switch (feed) {
                .value => |t| t,
                .handle => |h| try read(ptr, h),
            };
            try session.append(&.{token});
            return self.draw(s, position);
        }

        fn read(ptr: *anyopaque, handle: u64) anyerror!u32 {
            if (handle >= of(ptr).ids.len) return error.InvalidHandle;
            return @intCast(handle);
        }

        fn plainWindow(w: be.Window, o: be.Verified) bool {
            return w.held == 0 and w.tokens.len == 0 and w.parents == null and !w.early and w.masks.len == 0 and
                w.positions.len == 1 and o.sampled.len == 1 and o.drafts.len == 0 and o.rows.len == 0;
        }

        fn verify(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) anyerror!void {
            if (windows.len != out.len or windows.len == 0 or windows.len > 2) return error.UnsupportedWindow;
            for (windows, out) |w, o| if (!plainWindow(w, o)) return error.UnsupportedWindow;
            const self = of(ptr);
            while (!self.lock.tryLock()) std.Thread.yield() catch {};
            defer self.lock.unlock();
            if (windows.len == 1 or comptime !@hasDecl(Session, "pipelineRows")) {
                for (windows, out) |w, *o| {
                    o.sampled[0] = @intCast(try self.queueLocked(ptr, w.stream, .{ .value = w.pending }, w.positions[0]));
                }
                return;
            }
            const a = try self.sessionOf(windows[0].stream);
            const b = try self.sessionOf(windows[1].stream);
            if (a.position + 1 != windows[0].positions[0] or b.position + 1 != windows[1].positions[0])
                return error.PositionMismatch;
            try Session.pipelineRows(a, &.{windows[0].pending}, b, &.{windows[1].pending}, .last);
            out[0].sampled[0] = @intCast(try self.draw(windows[0].stream, windows[0].positions[0]));
            out[1].sampled[0] = @intCast(try self.draw(windows[1].stream, windows[1].positions[0]));
        }

        fn prefillBatch(ptr: *anyopaque, streams: []const *lanes.Stream, out: []be.PrefillOutcome) anyerror!void {
            if (streams.len != 2 or out.len != 2) return;
            if (comptime !@hasDecl(Session, "pipelineRows")) return;
            const self = of(ptr);
            while (!self.lock.tryLock()) std.Thread.yield() catch {};
            defer self.lock.unlock();
            for (streams) |s| {
                if (s.isCancelled()) return error.Cancelled;
                if (self.indexOf(s) == null) try self.begin(s);
            }
            const ia = self.indexOf(streams[0]) orelse return error.PositionMismatch;
            const ib = self.indexOf(streams[1]) orelse return error.PositionMismatch;
            if (self.prefill_rows == 0) return error.InvalidSession;
            while (self.prompt_at[ia] < streams[0].prompt_len or self.prompt_at[ib] < streams[1].prompt_len) {
                if (streams[0].isCancelled() or streams[1].isCancelled()) return error.Cancelled;
                const na = @min(self.prefill_rows, streams[0].prompt_len -| self.prompt_at[ia]);
                const nb = @min(self.prefill_rows, streams[1].prompt_len -| self.prompt_at[ib]);
                if (na > 0 and nb > 0) {
                    try Session.pipelineRows(
                        &self.sessions[ia],
                        streams[0].context.items[self.prompt_at[ia]..][0..na],
                        &self.sessions[ib],
                        streams[1].context.items[self.prompt_at[ib]..][0..nb],
                        .last,
                    );
                    self.advanceFill(streams[0], ia, na);
                    self.advanceFill(streams[1], ib, nb);
                } else if (na > 0) {
                    try self.sessions[ia].append(streams[0].context.items[self.prompt_at[ia]..][0..na]);
                    self.advanceFill(streams[0], ia, na);
                } else if (nb > 0) {
                    try self.sessions[ib].append(streams[1].context.items[self.prompt_at[ib]..][0..nb]);
                    self.advanceFill(streams[1], ib, nb);
                } else break;
            }
            out[0] = .complete;
            out[1] = .complete;
        }

        fn advanceFill(self: *Self, s: *lanes.Stream, i: usize, n: usize) void {
            const first_chunk = self.prompt_at[i] == 0;
            self.prompt_at[i] += n;
            s.fill_at.store(@intCast(self.prompt_at[i]), .monotonic);
            if (first_chunk) std.log.info("qwen prefill first chunk rows={d} position={d}", .{ n, self.sessions[i].position });
            if (self.prompt_at[i] < s.prompt_len and std.mem.indexOfScalar(u32, s.reuse.marks, @intCast(self.prompt_at[i])) != null) {
                if (s.reuse.hook) |k| k.at(k.ptr, s, @intCast(self.prompt_at[i]));
            }
        }

        fn keep(ptr: *anyopaque, windows: []const be.Window, paths: []const []const u32) anyerror!void {
            const self = of(ptr);
            if (windows.len != 1 or paths.len != 1 or paths[0].len != 1 or paths[0][0] != 0 or
                windows[0].rows() != 1 or self.indexOf(windows[0].stream) == null)
                return error.UnsupportedRollback;
        }

        fn draft(ptr: *anyopaque, requests: []const be.DraftRequest) anyerror!void {
            const self = of(ptr);
            for (requests) |r| if (r.depth != 0 or r.early or r.lanes != null or r.ranks or self.indexOf(r.stream) == null)
                return error.DraftsUnsupported;
        }

        fn release(ptr: *anyopaque, s: *lanes.Stream) void {
            const self = of(ptr);
            if (self.indexOf(s)) |i| {
                self.bound[i] = null;
                self.prompt_at[i] = 0;
            }
            if (self.active == s) self.active = null;
        }
    };
}

const FakeSession = struct {
    position: usize = 0,
    logits: [4]u16 = .{ 0x3f80, 0x4000, 0x4040, 0x4080 },
    resets: usize = 0,
    fail: bool = false,
    greedy: ?u32 = null,
    greedy_reads: usize = 0,
    append_calls: usize = 0,
    last_append: usize = 0,
    append_sizes: [8]usize = .{ 0, 0, 0, 0, 0, 0, 0, 0 },

    fn cachedGreedy(s: *FakeSession, position: u64) !?u32 {
        if (s.fail) return error.GpuFailed;
        if (position != s.position) return error.PositionMismatch;
        s.greedy_reads += 1;
        return s.greedy;
    }

    fn reset(s: *FakeSession) !void {
        s.position = 0;
        s.resets += 1;
        s.fail = false;
        s.greedy = null;
        s.append_calls = 0;
        s.last_append = 0;
        s.append_sizes = .{ 0, 0, 0, 0, 0, 0, 0, 0 };
    }

    fn append(s: *FakeSession, tokens: []const u32) !void {
        if (s.fail) return error.GpuFailed;
        if (s.append_calls < s.append_sizes.len) s.append_sizes[s.append_calls] = tokens.len;
        s.append_calls += 1;
        s.last_append = tokens.len;
        s.position += tokens.len;
    }
};

test "Qwen terminal greedy bypasses host conversion only for an eligible committed pick" {
    const gpa = std.testing.allocator;
    var session: FakeSession = .{ .position = 2, .greedy = 3 };
    var adapter = try Adapter(FakeSession).init(gpa, &session, 32);
    defer adapter.deinit();
    var s = try lanes.Stream.init(gpa, .{ .id = "greedy", .prompt = &.{ 1, 2 }, .max_new = 4 });
    defer s.deinit(gpa);
    adapter.active = &s;
    @memset(adapter.values, -123);
    try std.testing.expectEqual(@as(u64, 3), try adapter.backend().first(&s, 2));
    for (adapter.values) |value| try std.testing.expectEqual(@as(f64, -123), value);
    try std.testing.expectEqual(@as(usize, 1), session.greedy_reads);
    try std.testing.expectError(error.PositionMismatch, adapter.backend().first(&s, 3));
    try std.testing.expectEqual(@as(usize, 1), session.greedy_reads);
    session.greedy = null;
    try std.testing.expectEqual(@as(u64, 3), try adapter.backend().first(&s, 2));
    try std.testing.expectEqualSlices(f64, &.{ 1, 2, 3, 4 }, adapter.values);
    session.logits[0] = 0x7f80;
    const nonfinite = [_]f64{ std.math.inf(f64), 2, 3, 4 };
    const expected = try lanes.sampling.choose(gpa, &nonfinite, adapter.ids, 2, .{ .seed = 0, .temperature = 0 });
    try std.testing.expectEqual(expected, try adapter.backend().first(&s, 2));
    session.logits[0] = 0x3f80;
    session.greedy = 0;
    s.sampling = .{ .seed = 17, .temperature = 1, .top_k = 0, .top_p = 1 };
    const reads = session.greedy_reads;
    try std.testing.expectEqual(try lanes.sampling.choose(gpa, &.{ 1, 2, 3, 4 }, adapter.ids, 2, s.sampling.?), try adapter.backend().first(&s, 2));
    try std.testing.expectEqual(reads, session.greedy_reads);
    s.sampling = null;
    session.fail = true;
    try std.testing.expectError(error.GpuFailed, adapter.backend().first(&s, 2));
    adapter.active = null;
    try std.testing.expectError(error.PositionMismatch, adapter.backend().first(&s, 2));
}

test "Qwen lane adapter uses keyed positions, validates windows and resets between streams" {
    const gpa = std.testing.allocator;
    var session: FakeSession = .{};
    var adapter = try Adapter(FakeSession).init(gpa, &session, 32);
    defer adapter.deinit();
    const backend = adapter.backend();
    var s = try lanes.Stream.init(gpa, .{ .id = "first", .prompt = &.{ 1, 2 }, .max_new = 4, .sampling = .{ .seed = 17, .temperature = 1, .top_k = 0, .top_p = 1 } });
    defer s.deinit(gpa);
    try std.testing.expect(!try backend.vtable.prefill_step.?(backend.ptr, &s));
    try std.testing.expect(try backend.vtable.prefill_step.?(backend.ptr, &s));
    try std.testing.expectEqual(@as(u32, 2), s.fill_at.load(.monotonic));
    const values = [_]f64{ 1, 2, 3, 4 };
    const ids = [_]u64{ 0, 1, 2, 3 };
    try std.testing.expectEqual(try lanes.sampling.choose(gpa, &values, &ids, 2, s.sampling.?), try backend.first(&s, 2));
    var sampled: [1]u32 = undefined;
    const windows = [_]be.Window{.{ .stream = &s, .pending = 3, .held = 0, .tokens = &.{}, .parents = null, .positions = &.{3} }};
    var verified = [_]be.Verified{.{ .sampled = &sampled, .drafts = &.{} }};
    try backend.verify(&windows, &verified);
    try std.testing.expectEqual(try lanes.sampling.choose(gpa, &values, &ids, 3, s.sampling.?), sampled[0]);
    try backend.keep(&windows, &.{&.{0}});
    try std.testing.expectError(error.UnsupportedRollback, backend.keep(&windows, &.{&.{}}));
    try std.testing.expectError(error.DraftsUnsupported, backend.draft(&.{.{ .stream = &s, .follow = &.{}, .rows = null, .start = 3, .position = 4, .depth = 1 }}));
    var malformed = windows;
    malformed[0].positions = &.{ 4, 5 };
    try std.testing.expectError(error.UnsupportedWindow, backend.verify(&malformed, &verified));
    try std.testing.expectEqual(@as(usize, 3), session.position);
    session.fail = true;
    try std.testing.expectError(error.GpuFailed, backend.queue(&s, .{ .value = 1 }, 4));
    backend.release(&s);
    var next = try lanes.Stream.init(gpa, .{ .id = "next", .prompt = &.{0}, .max_new = 2 });
    defer next.deinit(gpa);
    try backend.prefill(&next);
    try std.testing.expectEqual(@as(usize, 2), session.resets);
    try std.testing.expectEqual(@as(usize, 1), session.position);
    backend.release(&next);
}

test "Qwen lane adapter refuses invalid state before resetting and cancellation before appending" {
    const gpa = std.testing.allocator;
    var session: FakeSession = .{};
    var adapter = try Adapter(FakeSession).init(gpa, &session, 4);
    defer adapter.deinit();
    const backend = adapter.backend();
    var invalid = try lanes.Stream.init(gpa, .{ .id = "invalid", .prompt = &.{4}, .max_new = 1 });
    defer invalid.deinit(gpa);
    try std.testing.expectError(error.InvalidToken, backend.prefill(&invalid));
    try std.testing.expectEqual(@as(usize, 0), session.resets);
    const Cancel = struct {
        fn check(_: *anyopaque) bool {
            return true;
        }
    };
    var cancelled = try lanes.Stream.init(gpa, .{ .id = "cancel", .prompt = &.{0}, .max_new = 1, .cancel_check = .{ .ptr = &session, .check = Cancel.check } });
    defer cancelled.deinit(gpa);
    try std.testing.expectError(error.Cancelled, backend.prefill(&cancelled));
    try std.testing.expectEqual(@as(usize, 0), session.resets);
    try std.testing.expectEqual(@as(usize, 0), session.position);
}

test "Qwen adapter serves complete shared-engine rounds and releases zero-output streams" {
    const gpa = std.testing.allocator;
    var session: FakeSession = .{};
    var adapter = try Adapter(FakeSession).init(gpa, &session, 32);
    defer adapter.deinit();
    var cfg = try lanes.Config.init(gpa, facts, 1, 0);
    defer cfg.deinit(gpa);
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, adapter.backend(), clock.clock());
    defer core.deinit();
    for ([_]u32{ 0, 1, 4 }) |max_new| {
        var s = try lanes.Stream.init(gpa, .{ .id = "rounds", .prompt = &.{ 1, 2 }, .max_new = max_new });
        defer s.deinit(gpa);
        var first_step = true;
        while (!try core.fillStream(&s, first_step)) first_step = false;
        while (core.activeCount() > 0) try core.step();
        try std.testing.expect(s.finished);
        try std.testing.expectEqual(lanes.stream.Reason.length, s.reason);
        try std.testing.expectEqual(@as(usize, max_new), s.emitted().len);
        for (s.emitted()) |token| try std.testing.expectEqual(@as(u32, 3), token);
        try std.testing.expect(adapter.active == null);
        try std.testing.expectEqual(@as(usize, 2 + (max_new -| 1)), session.position);
    }
}

test "Qwen call gates feed corrected opener and multi-token names into the model" {
    const gpa = std.testing.allocator;
    const Session = struct {
        position: usize = 0,
        logits: [9]u16 = .{ 0, 0, 0, 0, 0x4080, 0, 0, 0, 0 },
        fed: [32]u32 = undefined,
        end_thinking: bool = false,
        suggest_prefix: bool = false,
        fn cachedGreedy(_: *const @This(), _: u64) !?u32 {
            return null;
        }
        fn reset(s: *@This()) !void {
            s.position = 0;
        }
        fn append(s: *@This(), tokens: []const u32) !void {
            @memcpy(s.fed[s.position..][0..tokens.len], tokens);
            s.position += tokens.len;
            @memset(&s.logits, 0);
            const next: usize = if (s.end_thinking and s.position == 1) 7 else if (s.suggest_prefix) switch (s.position) {
                1 => 1,
                2 => 0,
                3 => 5,
                else => 4,
            } else 4;
            s.logits[next] = 0x4080;
        }
    };
    const Lex = struct {
        fn blank(_: *anyopaque, _: u32) bool {
            return false;
        }
        fn text(_: *anyopaque, _: Allocator, token: u32) ![]const u8 {
            return ([_][]const u8{ "<function=", "<tool_call>", "foo", ">", "wrong", "fo", "<think>", "</think>", "o" })[token];
        }
        fn encode(_: *anyopaque, _: Allocator, value: []const u8) ![]const u32 {
            if (std.mem.eql(u8, value, "<function=")) return &.{0};
            if (std.mem.eql(u8, value, "foo>")) return &.{ 2, 3 };
            if (std.mem.eql(u8, value, "o>")) return &.{ 8, 3 };
            return error.UnexpectedText;
        }
    };
    var session: Session = .{};
    var adapter = try Adapter(Session).init(gpa, &session, 32);
    defer adapter.deinit();
    var cfg = try lanes.Config.init(gpa, facts, 1, 0);
    defer cfg.deinit(gpa);
    try std.testing.expect(!cfg.pipelined);
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, adapter.backend(), clock.clock());
    defer core.deinit();
    const Case = struct {
        names: []const []const u8 = &.{"foo"},
        prompt: []const u32 = &.{0},
        emitted: []const u32 = &.{ 1, 0, 2, 3, 4, 4 },
        fed: []const u32 = &.{ 0, 1, 0, 2, 3, 4 },
        end_thinking: bool = false,
        suggest_prefix: bool = false,
    };
    for ([_]Case{
        .{ .names = &.{ "foo", "other" } },
        .{},
        .{ .prompt = &.{6}, .end_thinking = true, .emitted = &.{ 7, 1, 0, 2, 3, 4 }, .fed = &.{ 6, 7, 1, 0, 2, 3 } },
        .{ .suggest_prefix = true, .emitted = &.{ 1, 0, 5, 8, 3, 4 }, .fed = &.{ 0, 1, 0, 5, 8, 3 } },
    }) |case| {
        session.end_thinking = case.end_thinking;
        session.suggest_prefix = case.suggest_prefix;
        var s = try lanes.Stream.init(gpa, .{
            .id = "call",
            .prompt = case.prompt,
            .max_new = 6,
            .call = .{
                .opener = 1,
                .lead = "<function=",
                .names = case.names,
                .tail = ">",
                .think_open = 6,
                .think_end = 7,
                .lex = .{ .ctx = &session, .blank = Lex.blank, .text = Lex.text, .encode = Lex.encode },
            },
        });
        defer s.deinit(gpa);
        try core.addStream(&s);
        while (core.activeCount() > 0) try core.step();
        try std.testing.expectEqualSlices(u32, case.emitted, s.emitted());
        try std.testing.expectEqualSlices(u32, case.fed, session.fed[0..session.position]);
        try std.testing.expect(adapter.active == null);
    }
}

test "Qwen stepped prefill cancellation releases the slot and the next stream resets it" {
    const gpa = std.testing.allocator;
    var session: FakeSession = .{};
    var adapter = try Adapter(FakeSession).init(gpa, &session, 32);
    defer adapter.deinit();
    var cfg = try lanes.Config.init(gpa, facts, 1, 0);
    defer cfg.deinit(gpa);
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, adapter.backend(), clock.clock());
    defer core.deinit();
    const Cancel = struct {
        fn check(ptr: *anyopaque) bool {
            const s: *FakeSession = @ptrCast(@alignCast(ptr));
            return s.position == 2;
        }
    };
    var interrupted = try lanes.Stream.init(gpa, .{
        .id = "interrupted",
        .prompt = &.{ 0, 1, 2, 3 },
        .max_new = 4,
        .cancel_check = .{ .ptr = &session, .check = Cancel.check },
    });
    defer interrupted.deinit(gpa);
    try std.testing.expect(!try core.fillStream(&interrupted, true));
    try std.testing.expect(!try core.fillStream(&interrupted, false));
    try std.testing.expectError(error.Cancelled, core.fillStream(&interrupted, false));
    try std.testing.expectEqual(@as(usize, 2), session.position);
    try std.testing.expect(adapter.active == null);
    var next = try lanes.Stream.init(gpa, .{ .id = "next", .prompt = &.{1}, .max_new = 2 });
    defer next.deinit(gpa);
    try core.addStream(&next);
    while (core.activeCount() > 0) try core.step();
    try std.testing.expectEqualSlices(u32, &.{ 3, 3 }, next.emitted());
    try std.testing.expectEqual(@as(usize, 2), session.resets);
    try std.testing.expect(adapter.active == null);
}

test "Qwen bounded prefill appends true multi-row chunks with a partial final tail" {
    const gpa = std.testing.allocator;
    var session: FakeSession = .{};
    var adapter = try Adapter(FakeSession).init(gpa, &session, 64);
    defer adapter.deinit();
    adapter.prefill_rows = 16;
    const backend = adapter.backend();
    var exact = try lanes.Stream.init(gpa, .{ .id = "exact16", .prompt = &@as([16]u32, @splat(1)), .max_new = 1 });
    defer exact.deinit(gpa);
    try std.testing.expect(try backend.vtable.prefill_step.?(backend.ptr, &exact));
    try std.testing.expectEqual(@as(usize, 1), session.append_calls);
    try std.testing.expectEqual(@as(usize, 16), session.last_append);
    try std.testing.expectEqual(@as(u32, 16), exact.fill_at.load(.monotonic));
    backend.release(&exact);

    var under = try lanes.Stream.init(gpa, .{ .id = "under15", .prompt = &@as([15]u32, @splat(2)), .max_new = 1 });
    defer under.deinit(gpa);
    try backend.prefill(&under);
    try std.testing.expectEqual(@as(usize, 1), session.append_calls);
    try std.testing.expectEqual(@as(usize, 15), session.last_append);
    try std.testing.expectEqual(@as(u32, 15), under.fill_at.load(.monotonic));
    backend.release(&under);

    var over = try lanes.Stream.init(gpa, .{ .id = "over17", .prompt = &@as([17]u32, @splat(0)), .max_new = 1 });
    defer over.deinit(gpa);
    try std.testing.expect(!try backend.vtable.prefill_step.?(backend.ptr, &over));
    try std.testing.expectEqual(@as(usize, 1), session.append_calls);
    try std.testing.expectEqual(@as(usize, 16), session.append_sizes[0]);
    try std.testing.expectEqual(@as(u32, 16), over.fill_at.load(.monotonic));
    try std.testing.expect(try backend.vtable.prefill_step.?(backend.ptr, &over));
    try std.testing.expectEqual(@as(usize, 2), session.append_calls);
    try std.testing.expectEqual(@as(usize, 1), session.append_sizes[1]);
    try std.testing.expectEqual(@as(u32, 17), over.fill_at.load(.monotonic));
    backend.release(&over);
}

test "Qwen bounded prefill cancels between chunks and rejects a zero bound" {
    const gpa = std.testing.allocator;
    var session: FakeSession = .{};
    var adapter = try Adapter(FakeSession).init(gpa, &session, 64);
    defer adapter.deinit();
    adapter.prefill_rows = 16;
    const backend = adapter.backend();
    const Cancel = struct {
        fn check(ptr: *anyopaque) bool {
            const s: *FakeSession = @ptrCast(@alignCast(ptr));
            return s.position >= 16;
        }
    };
    var interrupted = try lanes.Stream.init(gpa, .{
        .id = "chunk-cancel",
        .prompt = &@as([20]u32, @splat(1)),
        .max_new = 1,
        .cancel_check = .{ .ptr = &session, .check = Cancel.check },
    });
    defer interrupted.deinit(gpa);
    try std.testing.expect(!try backend.vtable.prefill_step.?(backend.ptr, &interrupted));
    try std.testing.expectEqual(@as(usize, 16), session.position);
    try std.testing.expectError(error.Cancelled, backend.vtable.prefill_step.?(backend.ptr, &interrupted));
    backend.release(&interrupted);
    try std.testing.expect(adapter.active == null);

    var zero_session: FakeSession = .{};
    var zero_adapter = try Adapter(FakeSession).init(gpa, &zero_session, 64);
    defer zero_adapter.deinit();
    zero_adapter.prefill_rows = 0;
    var bad = try lanes.Stream.init(gpa, .{ .id = "zero-bound", .prompt = &.{1}, .max_new = 1 });
    defer bad.deinit(gpa);
    try std.testing.expectError(error.InvalidSession, zero_adapter.backend().prefill(&bad));
    try std.testing.expectEqual(@as(usize, 1), zero_session.resets);
    try std.testing.expectEqual(@as(usize, 0), zero_session.position);
}

test "Qwen adapter binds two sessions and refuses a third" {
    const gpa = std.testing.allocator;
    var sessions: [2]FakeSession = .{ .{}, .{} };
    var adapter = try Adapter(FakeSession).initSessions(gpa, &sessions, 32);
    defer adapter.deinit();
    const backend = adapter.backend();
    var a = try lanes.Stream.init(gpa, .{ .id = "a", .prompt = &.{1}, .max_new = 1 });
    defer a.deinit(gpa);
    var b = try lanes.Stream.init(gpa, .{ .id = "b", .prompt = &.{2}, .max_new = 1 });
    defer b.deinit(gpa);
    var c = try lanes.Stream.init(gpa, .{ .id = "c", .prompt = &.{3}, .max_new = 1 });
    defer c.deinit(gpa);
    try backend.prefill(&a);
    try backend.prefill(&b);
    try std.testing.expectEqual(@as(usize, 1), sessions[0].resets);
    try std.testing.expectEqual(@as(usize, 1), sessions[1].resets);
    try std.testing.expectError(error.LaneMemory, backend.prefill(&c));
    backend.release(&a);
    try backend.prefill(&c);
    try std.testing.expectEqual(@as(usize, 2), sessions[0].resets);
    backend.release(&b);
    backend.release(&c);
}

test "Qwen adapter verifies two plain windows on separate sessions" {
    const gpa = std.testing.allocator;
    var sessions: [2]FakeSession = .{ .{ .position = 1 }, .{ .position = 1 } };
    var adapter = try Adapter(FakeSession).initSessions(gpa, &sessions, 32);
    defer adapter.deinit();
    const backend = adapter.backend();
    var a = try lanes.Stream.init(gpa, .{ .id = "a", .prompt = &.{1}, .max_new = 4, .sampling = .{ .seed = 0, .temperature = 0 } });
    defer a.deinit(gpa);
    var b = try lanes.Stream.init(gpa, .{ .id = "b", .prompt = &.{2}, .max_new = 4, .sampling = .{ .seed = 0, .temperature = 0 } });
    defer b.deinit(gpa);
    try backend.prefill(&a);
    try backend.prefill(&b);
    sessions[0].position = 1;
    sessions[1].position = 1;
    var sampled_a: [1]u32 = undefined;
    var sampled_b: [1]u32 = undefined;
    const windows = [_]be.Window{
        .{ .stream = &a, .pending = 1, .held = 0, .tokens = &.{}, .parents = null, .positions = &.{2} },
        .{ .stream = &b, .pending = 2, .held = 0, .tokens = &.{}, .parents = null, .positions = &.{2} },
    };
    var verified = [_]be.Verified{
        .{ .sampled = &sampled_a, .drafts = &.{} },
        .{ .sampled = &sampled_b, .drafts = &.{} },
    };
    try backend.verify(&windows, &verified);
    try std.testing.expectEqual(@as(usize, 2), sessions[0].position);
    try std.testing.expectEqual(@as(usize, 2), sessions[1].position);
    try std.testing.expectEqual(@as(u32, 3), sampled_a[0]);
    try std.testing.expectEqual(@as(u32, 3), sampled_b[0]);
}
