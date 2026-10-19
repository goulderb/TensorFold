//! A backend-controlled neural prefix is requested whole before the family's shared row trim.
const std = @import("std");
const depth = @import("depth.zig");
const Config = @import("config.zig").Config;
const Engine = @import("engine.zig").Engine;
const fake = @import("fake.zig");
const a = std.testing.allocator;

const Probe = struct {
    copied: bool = false,
    fn copy(ptr: *anyopaque, _: *const depth.Who) bool {
        const p: *Probe = @ptrCast(@alignCast(ptr));
        return p.copied;
    }
    fn probe(p: *Probe) depth.Probe {
        return .{ .ptr = p, .copy = copy };
    }
};

test "fixed backend depth requests seven before shared trimming despite low acceptance or prior grants" {
    var cfg = try Config.init(a, .{ .exact_width = 16, .mtp = true, .speculate = true, .drafts = 7, .fixed_drafts = true, .hidden_rows = true, .batch_rows = 32 }, 16, 15);
    defer cfg.deinit(a);
    try std.testing.expect(cfg.fixed_drafts);
    try std.testing.expectEqual(@as(u32, 7), cfg.most_drafts);
    try std.testing.expect(cfg.draft_confidence == null and !cfg.node_probabilities);
    var target: fake.Fake = .{ .gpa = a };
    defer target.deinit();
    var clock: fake.FixedClock = .{};
    var engine = Engine.init(a, &cfg, target.backend(), clock.clock());
    defer engine.deinit();
    var states: [8]?depth.State = @splat(null);
    defer for (&states) |*s| if (s.*) |*state| state.deinit(a);
    var whos: [8]depth.Who = undefined;
    for (&whos, &states) |*w, *s| w.* = .{ .state = s, .draft_room = 64, .finished = false, .forced = false, .granted = 0 };
    var probe: Probe = .{};
    var out: [8]i64 = undefined;
    try engine.rule.budgets(&whos, probe.probe(), false, &out);
    try std.testing.expectEqualSlices(i64, &[_]i64{ 7, 7, 7, 7, 7, 7, 7, 7 }, &out);
    for (whos) |w| {
        for (0..8) |_| try engine.rule.observeDepth(w, 7, 0);
        try std.testing.expectEqual(@as(i64, 7), try engine.rule.depth(w));
    }
    try engine.rule.budgets(&whos, probe.probe(), true, &out);
    try std.testing.expectEqualSlices(i64, &[_]i64{ 7, 7, 7, 7, 7, 7, 7, 7 }, &out);
    probe.copied = true;
    try engine.rule.budgets(&whos, probe.probe(), false, &out);
    try std.testing.expectEqualSlices(i64, &[_]i64{ 0, 0, 0, 0, 0, 0, 0, 0 }, &out);
    probe.copied = false;
    whos[0].forced = true;
    whos[1].finished = true;
    whos[2].draft_room = 1;
    whos[3].draft_room = 2;
    engine.rule.node_probabilities = true;
    try engine.rule.budgets(&whos, probe.probe(), false, &out);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 0, 1, 7, 7, 7, 7 }, &out);
}
