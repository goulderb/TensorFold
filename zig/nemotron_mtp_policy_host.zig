//! Real Nemotron facts and common lane rounds with deterministic head confidences, without device work.
const std = @import("std");
const nemo = @import("nemotron");
const lanes = @import("lanes");
const a = std.testing.allocator;
const be = lanes.backend;
const Fake = lanes.fake.Fake;
const prompt = [_]u32{ 1, 3, 7, 11, 19, 23, 29, 31 };

test "mtp policy: actual family facts carry positive caps and confidence into the common rounds" {
    const e = try a.create(nemo.Engine);
    defer a.destroy(e);
    e.* = undefined;
    e.c.latent = 512;
    e.tp_link = null;
    var head: nemo.Head = undefined;
    var b: nemo.Lanes = undefined;
    b.e = e;
    b.head = &head;
    b.cost_count = 0;
    b.mtp_ms = 0;
    const facts = b.facts();
    try std.testing.expectEqual(@as(u32, 15), facts.drafts);
    try std.testing.expect(facts.mtp and facts.hidden_rows and facts.copy_drafts);
    try std.testing.expectEqual(@as(u32, nemo.state.max_rows), facts.exact_width);
    try std.testing.expect(b.backend().vtable.probabilities == null);
}

const Record = struct {
    var most: usize = 0;
    fn probabilities(_: *anyopaque, _: *lanes.Stream, out: []f64) !bool {
        var chance: f64 = 1;
        for (out) |*v| {
            chance *= 0.8;
            v.* = chance;
        }
        return true;
    }
    fn verify(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) !void {
        const target: *Fake = @ptrCast(@alignCast(ptr));
        for (windows) |w| if (std.mem.eql(u8, w.stream.id, "subject")) {
            most = @max(most, w.held + w.tokens.len);
        };
        try target.backend().vtable.verify(ptr, windows, out);
    }
};
const Options = struct { neighbor: bool = false, resumed: bool = false, handoff: bool = false, sampled: bool = false, drafted: bool = true, floor: ?f64 = 0.5, cap: u32 = 8, width: u32 = 16 };
fn run(o: Options) ![]u32 {
    Record.most = 0;
    const costs = [_]lanes.config.Cost{ .{ .width = 1, .ms = 1 }, .{ .width = 16, .ms = 1 } };
    var cfg = try lanes.Config.init(a, .{ .exact_width = o.width, .copy_drafts = false, .mtp = o.drafted, .speculate = true, .speculate_early = false, .drafts = o.cap, .draft_probabilities = true, .draft_confidence = o.floor, .window_costs = &costs, .hidden_rows = true, .batch_rows = o.width, .max_streams = 4, .draft_streams = true }, o.width, o.width - 1);
    defer cfg.deinit(a);
    var target: Fake = .{ .gpa = a };
    defer target.deinit();
    const original = target.backend();
    var vt = original.vtable.*;
    vt.probabilities = Record.probabilities;
    vt.verify = Record.verify;
    var clock: lanes.fake.FixedClock = .{};
    var e = lanes.Engine.init(a, &cfg, .{ .ptr = original.ptr, .vtable = &vt }, clock.clock());
    defer e.deinit();
    var kept: std.ArrayList(u32) = .empty;
    defer kept.deinit(a);
    try kept.appendSlice(a, prompt[0..4]);
    var subject = try lanes.Stream.init(a, .{ .id = "subject", .prompt = &prompt, .max_new = 80, .eos = &.{}, .drafts = o.drafted, .sampling = if (o.sampled) .{ .seed = 87, .temperature = 0.7 } else null, .reuse = if (o.resumed) .{ .saved = &kept, .at = 4 } else .{} });
    defer subject.deinit(a);
    var neighbor = try lanes.Stream.init(a, .{ .id = "neighbor", .prompt = &.{ 4, 6, 8, 12, 14 }, .max_new = 80, .eos = &.{}, .drafts = o.drafted });
    defer neighbor.deinit(a);
    try e.addStream(&subject);
    if (o.neighbor) try e.addStream(&neighbor);
    var steps: usize = 0;
    while (e.activeCount() > 0) : (steps += 1) {
        if (steps > 200) return error.DidNotFinish;
        try e.step();
        if (o.handoff and steps == 1 and !subject.finished) {
            try std.testing.expect(e.detach(&subject));
            try e.adopt(&subject);
        }
    }
    if (o.drafted) {
        try std.testing.expect(Record.most > 0);
        const ceiling: usize = if (o.floor == 0.5) @min(o.cap, 3) else if (o.floor == 1) 1 else o.cap;
        try std.testing.expect(Record.most <= ceiling);
    }
    return a.dupe(u32, subject.emitted());
}
test "mtp policy: shared confidence controls keep equality across neighbors resume and handoff" {
    for ([_]bool{ false, true }) |sampled| {
        const plain = try run(.{ .drafted = false, .sampled = sampled });
        defer a.free(plain);
        for ([_]bool{ false, true }) |neighbor| for ([_]bool{ false, true }) |resumed| for ([_]bool{ false, true }) |handoff| {
            const got = try run(.{ .neighbor = neighbor, .resumed = resumed, .handoff = handoff, .sampled = sampled });
            defer a.free(got);
            try std.testing.expectEqualSlices(u32, plain, got);
        };
    }
}
test "mtp policy: confidence endpoints and a positive cap are enforced by actual verification" {
    for ([_]?f64{ 0, 0.5, 1 }) |floor| for ([_]u32{ 1, 2, 8 }) |cap| {
        const got = try run(.{ .floor = floor, .cap = cap, .neighbor = true });
        defer a.free(got);
        try std.testing.expectEqual(@as(usize, 80), got.len);
    };
}

test "mtp policy: CUDA family facts do not advertise a separate probability policy; common rounds still honor draft_confidence" {
    const e = try a.create(nemo.Engine);
    defer a.destroy(e);
    e.* = undefined;
    e.tp_link = null;
    var head: nemo.Head = undefined;
    var b: nemo.Lanes = undefined;
    b.e = e;
    b.head = &head;
    b.cost_count = 0;
    b.mtp_ms = 0;
    try std.testing.expect(!b.facts().draft_probabilities);
    try std.testing.expect(b.backend().vtable.probabilities == null);
}

test "mtp policy: explicit cap leaves copy capacity independent and invalid confidence is refused" {
    var cfg = try lanes.Config.init(a, .{ .exact_width = 16, .drafts = 2, .mtp = true, .speculate = true, .draft_confidence = 0.5 }, 16, 15);
    defer cfg.deinit(a);
    try std.testing.expectEqual(@as(u32, 2), cfg.most_drafts);
    try std.testing.expectEqual(@as(u32, 15), cfg.max_copy);
    for ([_]f64{ -1, 2, std.math.nan(f64) }) |floor| try std.testing.expectError(error.InvalidDraftConfidence, lanes.Config.init(a, .{ .exact_width = 16, .draft_confidence = floor }, 16, 15));
    const empty: lanes.table.Table = .{};
    const counts = try lanes.allocate.allocate(a, &.{1}, &.{&.{ 0, 0, 0 }}, &empty, 0, 4);
    defer a.free(counts);
    try std.testing.expectEqualSlices(u32, &.{3}, counts);
}

test "mtp policy: no-head models share windows only with tensor-parallel, not a retired Bolt flag" {
    const e = try a.create(nemo.Engine);
    defer a.destroy(e);
    e.* = undefined;
    e.c.latent = 0;
    e.tp_link = null;
    var b: nemo.Lanes = undefined;
    b.e = e;
    b.head = null;
    b.cost_count = 0;
    b.mtp_ms = 0;
    const lone = b.facts();
    try std.testing.expectEqual(@as(u32, 1), lone.exact_width);
    try std.testing.expect(!lone.mtp and !lone.hidden_rows);
    e.tp_link = @ptrFromInt(8);
    const shared = b.facts();
    try std.testing.expectEqual(@as(u32, nemo.state.max_rows), shared.exact_width);
    try std.testing.expect(shared.hidden_rows and shared.max_streams > 1 and !shared.mtp);
}

test "mtp policy: decode windows stay the current max_rows; Super/Bolt split widths are retired" {
    try std.testing.expectEqual(@as(usize, 16), nemo.state.max_rows);
}

test "mtp policy: the inherited policy works through 31 drafts and a resumed shared 32 row round" {
    const plain = try run(.{ .drafted = false });
    defer a.free(plain);
    for ([_]bool{ false, true }) |neighbor| for ([_]bool{ false, true }) |resumed| for ([_]bool{ false, true }) |handoff| {
        const got = try run(.{ .width = 32, .cap = 31, .floor = 0, .neighbor = neighbor, .resumed = resumed, .handoff = handoff });
        defer a.free(got);
        try std.testing.expectEqualSlices(u32, plain, got);
    };
}

test "mtp policy: rank-local plan ranks survive sibling follow after a resumed sequence is bound" {
    var owner: nemo.state.Buffers = undefined;
    owner.plan = .{ .members = 0, .items = 0, .counts = 0, .rank = 0x300000, .hist = 0x400000 };
    owner.state_bytes = @splat(64);
    inline for (nemo.state.seq_fields, 0..) |name, i| @field(owner, name) = 0x600000 + i * 4096;
    var sibling = owner;
    sibling.plan.rank = 0x700000;
    sibling.plan.hist = 0x800000;
    sibling.follow(&owner);
    try std.testing.expectEqual(owner.k_cache, sibling.k_cache);
    try std.testing.expectEqual(owner.hidden, sibling.hidden);
    try std.testing.expectEqual(@as(u64, 0x700000), sibling.plan.rank);
    try std.testing.expectEqual(@as(u64, 0x800000), sibling.plan.hist);
}
