//! Host tests of the C ABI layer over a scripted engine.
const std = @import("std");
const record = @import("record.zig");
const abi = @import("abi.zig");
const Error = abi.Error;
const Status = abi.Status;
const Open = abi.Open;
const Message = abi.Message;
const Info = abi.Info;
const OpenOptions = abi.OpenOptions;
const flag_head_only = abi.flag_head_only;
const hidden_width = abi.hidden_width;
const Exports = abi.Exports;

/// A scripted engine: each call takes the next outcome, writes logits 0.5 * index, and records what it saw.
const Fake = @import("abi_fake.zig").Fake;

const T = Exports(Fake, std.testing.allocator);

const Sample = struct {
    ids: [30]u32 = @splat(3),
    questions: [2]record.C.Question = .{
        .{ .kind = 0, .question = .{ .start = 1, .end = 3 }, .first_option = 0, .option_count = 2 },
        .{ .kind = 2, .question = .{ .start = 10, .end = 12 }, .first_option = 2, .option_count = 1 },
    },
    options: [3]record.Span = .{ .{ .start = 4, .end = 6 }, .{ .start = 7, .end = 8 }, .{ .start = 13, .end = 15 } },
    c: record.C.Record = undefined,

    fn get(s: *Sample) *const record.C.Record {
        s.c = .{ .ids = &s.ids, .n_ids = 30, .questions = &s.questions, .n_questions = 2, .options = &s.options, .n_options = 3 };
        return &s.c;
    }
};

fn openTest(flags: u32) !*T.Handle {
    const o: OpenOptions = .{ .model_dir = "C:/model", .head_sha256 = @splat(0), .device_uuid = @splat(7), .cap_bytes = 1 << 34, .max_tokens = 64, .flags = flags };
    return T.open(&o, null, 0) orelse error.OpenFailed;
}

fn lastText(h: *T.Handle) []const u8 {
    return h.last.text();
}

test "version and null handles are answered without an engine" {
    Fake.reset();
    try std.testing.expectEqual(@as(u32, 1), T.abiVersion());
    var s: Sample = .{};
    var logits: [3]f32 = undefined;
    try std.testing.expectEqual(Status.invalid, T.decide(null, s.get(), &logits, 3));
    try std.testing.expectEqual(Status.invalid, T.head(null, s.get(), null, &logits, 3));
    var info: Info = undefined;
    try std.testing.expectEqual(Status.invalid, T.infoGet(null, &info));
    try std.testing.expectEqual(@as(usize, 0), T.lastError(null, null, 0));
    T.close(null);
    try std.testing.expectEqual(@as(usize, 0), Fake.opened);
}

test "open refuses bad options with a message and never opens the engine" {
    Fake.reset();
    var err: [24]u8 = undefined;
    try std.testing.expect(T.open(null, &err, err.len) == null);
    try std.testing.expectEqualStrings("open options are null", std.mem.sliceTo(&err, 0));
    var o: OpenOptions = .{ .model_dir = null, .head_sha256 = @splat(0), .device_uuid = @splat(0), .cap_bytes = 0, .max_tokens = 64, .flags = 0 };
    try std.testing.expect(T.open(&o, &err, err.len) == null);
    try std.testing.expectEqualStrings("model_dir is null", std.mem.sliceTo(&err, 0));
    o.model_dir = "x";
    o.flags = 2;
    try std.testing.expect(T.open(&o, &err, err.len) == null);
    o.flags = 0;
    o.max_tokens = 4097;
    try std.testing.expect(T.open(&o, &err, 8) == null);
    try std.testing.expectEqualStrings("max_tok", std.mem.sliceTo(&err, 0));
    try std.testing.expectEqual(@as(usize, 0), Fake.opened);
    o.max_tokens = 64;
    Fake.fail_open = true;
    try std.testing.expect(T.open(&o, &err, err.len) == null);
    try std.testing.expectEqualStrings("device 0 not found", std.mem.sliceTo(&err, 0));
}

test "a decision parses the record, derives the option map and returns logits in option order" {
    Fake.reset();
    const h = try openTest(0);
    defer T.close(h);
    var s: Sample = .{};
    var logits: [4]f32 = @splat(-1);
    try std.testing.expectEqual(Status.ok, T.decide(h, s.get(), &logits, 4));
    try std.testing.expectEqualSlices(f32, &.{ 0, 0.5, 1.0, -1 }, &logits);
    try std.testing.expectEqualSlices(u16, &.{ 0, 0, 1 }, Fake.last_option_question[0..3]);
    try std.testing.expectEqual(@as(usize, 30), Fake.last_ids);
}

test "refusals before the engine leave it unchanged and keep a message" {
    Fake.reset();
    const h = try openTest(0);
    defer T.close(h);
    var s: Sample = .{};
    var logits: [3]f32 = undefined;
    try std.testing.expectEqual(Status.invalid, T.decide(h, null, &logits, 3));
    try std.testing.expectEqual(Status.invalid, T.decide(h, s.get(), null, 3));
    try std.testing.expectEqual(Status.invalid, T.decide(h, s.get(), &logits, 2));
    try std.testing.expectEqualStrings("logits need room for 3 options", lastText(h));
    s.c.n_ids = 65;
    try std.testing.expectEqual(Status.capacity, T.decide(h, &s.c, &logits, 3));
    try std.testing.expectEqualStrings("record refused: TooLong", lastText(h));
    s.questions[0].option_count = 0;
    try std.testing.expectEqual(Status.invalid, T.decide(h, s.get(), &logits, 3));
    try std.testing.expectEqual(@as(usize, 0), Fake.calls);
    var buf: [10]u8 = undefined;
    const full = T.lastError(h, &buf, buf.len);
    try std.testing.expectEqual(lastText(h).len, full);
    try std.testing.expectEqualStrings("record re", std.mem.sliceTo(&buf, 0));
}

test "device faults stay reusable, non-finite logits are reported, and a fatal fault poisons the handle" {
    Fake.reset();
    const h = try openTest(0);
    var s: Sample = .{};
    var logits: [3]f32 = undefined;
    Fake.outcomes[0] = error.Device;
    Fake.outcomes[1] = null;
    Fake.outcomes[2] = error.NonFinite;
    Fake.outcomes[3] = error.Fatal;
    try std.testing.expectEqual(Status.device, T.decide(h, s.get(), &logits, 3));
    try std.testing.expectEqualStrings("scripted Device", lastText(h));
    try std.testing.expectEqual(Status.ok, T.decide(h, s.get(), &logits, 3));
    try std.testing.expectEqual(Status.nonfinite, T.decide(h, s.get(), &logits, 3));
    try std.testing.expectEqual(Status.fatal, T.decide(h, s.get(), &logits, 3));
    try std.testing.expectEqual(Status.fatal, T.decide(h, s.get(), &logits, 3));
    try std.testing.expectEqual(Status.fatal, T.head(h, s.get(), &.{}, &logits, 3));
    var info: Info = undefined;
    try std.testing.expectEqual(Status.fatal, T.infoGet(h, &info));
    try std.testing.expectEqual(@as(usize, 4), Fake.calls);
    T.close(h);
    try std.testing.expect(Fake.closed_poisoned);
    try std.testing.expectEqual(@as(usize, 1), Fake.closed);
}

var busy_handle: *T.Handle = undefined;
var busy_sample: Sample = .{};

fn reenterDecide() Status {
    var logits: [3]f32 = undefined;
    return T.decide(busy_handle, busy_sample.get(), &logits, 3);
}

test "a call made while another is inside the engine is refused as busy without touching it" {
    Fake.reset();
    busy_handle = try openTest(0);
    defer T.close(busy_handle);
    Fake.reenter = reenterDecide;
    var logits: [3]f32 = undefined;
    try std.testing.expectEqual(Status.ok, T.decide(busy_handle, busy_sample.get(), &logits, 3));
    try std.testing.expectEqual(Status.busy, Fake.reentered.?);
    try std.testing.expectEqual(@as(usize, 1), Fake.calls);
    Fake.reenter = null;
    try std.testing.expectEqual(Status.ok, T.decide(busy_handle, busy_sample.get(), &logits, 3));
}

test "a head-only engine refuses decisions and runs the head over caller rows" {
    Fake.reset();
    const h = try openTest(flag_head_only);
    defer T.close(h);
    var s: Sample = .{};
    var logits: [3]f32 = undefined;
    try std.testing.expectEqual(Status.invalid, T.decide(h, s.get(), &logits, 3));
    try std.testing.expectEqual(@as(usize, 0), Fake.calls);
    const rows = try std.testing.allocator.alloc(u16, 30 * hidden_width);
    defer std.testing.allocator.free(rows);
    try std.testing.expectEqual(Status.invalid, T.head(h, s.get(), null, &logits, 3));
    try std.testing.expectEqual(Status.ok, T.head(h, s.get(), rows.ptr, &logits, 3));
    try std.testing.expectEqual(@as(usize, 30 * hidden_width), Fake.last_hidden);
    var info: Info = undefined;
    try std.testing.expectEqual(Status.ok, T.infoGet(h, &info));
    try std.testing.expectEqual(@as(u8, 1), info.head_only);
    try std.testing.expectEqual(@as(u32, 32), info.max_options);
    try std.testing.expectEqual(@as(u64, 123), info.budget_bytes);
    try std.testing.expectEqualStrings("test", std.mem.sliceTo(&info.source_commit, 0));
}
