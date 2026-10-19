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
        capacity: usize,
        active: ?*lanes.Stream = null,
        prompt_at: usize = 0,
        values: []f64,
        ids: []u64,

        pub fn init(gpa: Allocator, session: *Session, capacity: usize) !Self {
            if (session.logits.len == 0 or capacity == 0) return error.InvalidSession;
            const values = try gpa.alloc(f64, session.logits.len);
            errdefer gpa.free(values);
            const ids = try gpa.alloc(u64, session.logits.len);
            for (ids, 0..) |*id, i| id.* = i;
            return .{ .gpa = gpa, .session = session, .capacity = capacity, .values = values, .ids = ids };
        }

        pub fn deinit(self: *Self) void {
            self.gpa.free(self.values);
            self.gpa.free(self.ids);
        }

        pub fn backend(self: *Self) be.Backend {
            return .{ .ptr = self, .vtable = &vtable };
        }

        const vtable: be.Backend.VTable = .{
            .prefill = prefill,
            .prefill_step = prefillStep,
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

        fn draw(self: *Self, s: *lanes.Stream, position: u64) !u64 {
            if (self.active != s or self.session.position != position) return error.PositionMismatch;
            for (self.values, self.session.logits) |*value, bits|
                value.* = @as(f32, @bitCast(@as(u32, bits) << 16));
            return lanes.sampling.choose(self.gpa, self.values, self.ids, position, s.sampling orelse .{ .seed = 0, .temperature = 0 });
        }

        fn begin(self: *Self, s: *lanes.Stream) !void {
            if (self.active != null) return error.LaneMemory;
            if (s.prompt_len == 0 or s.prompt_len > self.capacity or s.max_new > self.capacity - s.prompt_len)
                return error.ContextFull;
            if (s.vision != null or s.images.len > 0 or s.grammar != null or
                s.logprobs != null or s.prompt_logprobs != null or s.think_budget != 0 or
                s.decode_spans.len > 0 or s.reuse.saved != null or s.reuse.at != 0)
                return error.UnsupportedRequest;
            for (s.context.items[0..s.prompt_len]) |id| if (id >= self.ids.len) return error.InvalidToken;
            try self.session.reset();
            self.prompt_at = 0;
            self.active = s;
        }

        fn prefill(ptr: *anyopaque, s: *lanes.Stream) anyerror!void {
            while (!try prefillStep(ptr, s)) {}
        }

        fn prefillStep(ptr: *anyopaque, s: *lanes.Stream) anyerror!bool {
            const self = of(ptr);
            if (s.isCancelled()) return error.Cancelled;
            if (self.active != s) try self.begin(s);
            if (self.prompt_at < s.prompt_len) {
                try self.session.append(s.context.items[self.prompt_at..][0..1]);
                self.prompt_at += 1;
                s.fill_at.store(@intCast(self.prompt_at), .monotonic);
            }
            return self.prompt_at == s.prompt_len;
        }

        fn first(ptr: *anyopaque, s: *lanes.Stream, position: u64) anyerror!u64 {
            return of(ptr).draw(s, position);
        }

        fn queue(ptr: *anyopaque, s: *lanes.Stream, feed: be.Feed, position: u64) anyerror!u64 {
            const self = of(ptr);
            if (self.active != s or self.session.position + 1 != position) return error.PositionMismatch;
            const token = switch (feed) {
                .value => |t| t,
                .handle => |h| try read(ptr, h),
            };
            try self.session.append(&.{token});
            return self.draw(s, position);
        }

        fn read(ptr: *anyopaque, handle: u64) anyerror!u32 {
            if (handle >= of(ptr).ids.len) return error.InvalidHandle;
            return @intCast(handle);
        }

        fn verify(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) anyerror!void {
            if (windows.len != 1 or out.len != 1) return error.UnsupportedWindow;
            const w = windows[0];
            if (w.held != 0 or w.tokens.len != 0 or w.parents != null or w.early or w.masks.len != 0 or
                w.positions.len != 1 or out[0].sampled.len != 1 or out[0].drafts.len != 0 or out[0].rows.len != 0)
                return error.UnsupportedWindow;
            out[0].sampled[0] = @intCast(try queue(ptr, w.stream, .{ .value = w.pending }, w.positions[0]));
        }

        fn keep(ptr: *anyopaque, windows: []const be.Window, paths: []const []const u32) anyerror!void {
            if (windows.len != 1 or paths.len != 1 or paths[0].len != 1 or paths[0][0] != 0 or
                windows[0].rows() != 1 or of(ptr).active != windows[0].stream)
                return error.UnsupportedRollback;
        }

        fn draft(ptr: *anyopaque, requests: []const be.DraftRequest) anyerror!void {
            for (requests) |r| if (r.depth != 0 or r.early or r.lanes != null or r.ranks or of(ptr).active != r.stream)
                return error.DraftsUnsupported;
        }

        fn release(ptr: *anyopaque, s: *lanes.Stream) void {
            const self = of(ptr);
            if (self.active == s) self.active = null;
        }
    };
}

const FakeSession = struct {
    position: usize = 0,
    logits: [4]u16 = .{ 0x3f80, 0x4000, 0x4040, 0x4080 },
    resets: usize = 0,
    fail: bool = false,

    fn reset(s: *FakeSession) !void {
        s.position = 0;
        s.resets += 1;
        s.fail = false;
    }

    fn append(s: *FakeSession, tokens: []const u32) !void {
        if (s.fail) return error.GpuFailed;
        s.position += tokens.len;
    }
};

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
        logits: [6]u16 = .{ 0, 0, 0, 0, 0x4080, 0 },
        fed: [32]u32 = undefined,
        fn reset(s: *@This()) !void {
            s.position = 0;
        }
        fn append(s: *@This(), tokens: []const u32) !void {
            @memcpy(s.fed[s.position..][0..tokens.len], tokens);
            s.position += tokens.len;
        }
    };
    const Lex = struct {
        fn blank(_: *anyopaque, _: u32) bool {
            return false;
        }
        fn text(_: *anyopaque, _: Allocator, token: u32) ![]const u8 {
            return ([_][]const u8{ "<function=", "<tool_call>", "foo", ">", "wrong", "other" })[token];
        }
        fn encode(_: *anyopaque, _: Allocator, value: []const u8) ![]const u32 {
            if (std.mem.eql(u8, value, "<function=")) return &.{0};
            if (std.mem.eql(u8, value, "foo>")) return &.{ 2, 3 };
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
    for ([_][]const []const u8{ &.{ "foo", "other" }, &.{"foo"} }) |names| {
        var s = try lanes.Stream.init(gpa, .{
            .id = "call",
            .prompt = &.{0},
            .max_new = 6,
            .call = .{
                .opener = 1,
                .lead = "<function=",
                .names = names,
                .tail = ">",
                .lex = .{ .ctx = &session, .blank = Lex.blank, .text = Lex.text, .encode = Lex.encode },
            },
        });
        defer s.deinit(gpa);
        try core.addStream(&s);
        while (core.activeCount() > 0) try core.step();
        try std.testing.expectEqualSlices(u32, &.{ 1, 0, 2, 3, 4, 4 }, s.emitted());
        try std.testing.expectEqualSlices(u32, &.{ 0, 1, 0, 2, 3, 4 }, session.fed[0..session.position]);
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
