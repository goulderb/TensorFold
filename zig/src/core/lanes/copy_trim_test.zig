//! Copied prefixes join a family's row budget only when its policy explicitly opts in.
const std = @import("std");
const win = @import("windows.zig");
const Stream = @import("stream.zig").Stream;
const Engine = @import("engine.zig").Engine;
const Config = @import("config.zig").Config;
const fake = @import("fake.zig");
const trim = @import("trim.zig");
const a = std.testing.allocator;

const Policy = struct {
    count: u32,
    nodes: ?[]const u32 = null,
    saw_copy: bool = false,

    fn choose(ptr: *anyopaque, windows: []const trim.Window, _: bool, _: std.mem.Allocator, out: []trim.Choice) !void {
        const p: *Policy = @ptrCast(@alignCast(ptr));
        for (windows, out) |w, *o| {
            p.saw_copy = p.saw_copy or w.copied;
            o.* = .{ .count = p.count, .nodes = p.nodes };
        }
    }
    fn commit(_: *anyopaque, _: trim.Commit) !void {}
    fn rule(p: *Policy, comptime copies: bool) trim.Trim {
        return .{ .ptr = p, .vtable = &.{ .copy_windows = copies, .choose = choose, .commit = commit } };
    }
};

test "copy trim is opt-in, preserves forced rows and keeps the chosen prefix including zero" {
    var cfg = try Config.init(a, .{ .exact_width = 16, .hidden_rows = true }, 16, 7);
    defer cfg.deinit(a);
    var target: fake.Fake = .{ .gpa = a };
    defer target.deinit();
    var clock: fake.FixedClock = .{};
    var engine = Engine.init(a, &cfg, target.backend(), clock.clock());
    defer engine.deinit();
    var s = try Stream.init(a, .{ .id = "copies", .prompt = &.{ 1, 2, 3 }, .max_new = 64 });
    defer s.deinit(a);
    const tokens = [_]u32{ 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18 };
    inline for (.{ false, true }) |copies| {
        for ([_]u32{ 0, 2, 15 }) |count| {
            var policy: Policy = .{ .count = count };
            engine.trim = policy.rule(copies);
            var plans = [_]win.Plan{
                .{ .stream = &s, .position = 3, .kind = .copy, .tokens = &tokens },
                .{ .stream = &s, .position = 3, .kind = .head, .held = 7 },
                .{ .stream = &s, .position = 3, .kind = .forced, .tokens = &tokens, .forced = &tokens },
                .{ .stream = &s, .position = 3, .kind = .none },
            };
            try win.allocate(&engine, &plans);
            try std.testing.expect(policy.saw_copy);
            try std.testing.expectEqual(@as(usize, if (copies) count else 15), plans[0].count());
            try std.testing.expectEqualSlices(u32, tokens[0..if (copies) count else 15], plans[0].tokens);
            try std.testing.expectEqual(if (copies and count == 0) win.Kind.none else win.Kind.copy, plans[0].kind);
            try std.testing.expectEqual(@as(usize, @min(count, 7)), plans[1].count());
            try std.testing.expectEqual(@as(usize, 15), plans[2].count());
            try std.testing.expectEqual(win.Kind.forced, plans[2].kind);
            try std.testing.expectEqual(@as(usize, 0), plans[3].count());
        }
    }
}

test "copy trim refuses node selections rather than changing the copied continuation" {
    var cfg = try Config.init(a, .{ .exact_width = 16 }, 16, 7);
    defer cfg.deinit(a);
    var target: fake.Fake = .{ .gpa = a };
    defer target.deinit();
    var clock: fake.FixedClock = .{};
    var engine = Engine.init(a, &cfg, target.backend(), clock.clock());
    defer engine.deinit();
    var s = try Stream.init(a, .{ .id = "copy-tree", .prompt = &.{1}, .max_new = 16 });
    defer s.deinit(a);
    var policy: Policy = .{ .count = 1, .nodes = &.{1} };
    engine.trim = policy.rule(true);
    var plans = [_]win.Plan{.{ .stream = &s, .position = 1, .kind = .copy, .tokens = &.{ 2, 3 } }};
    try std.testing.expectError(error.CopyTrimNeedsPrefix, win.allocate(&engine, &plans));
    try std.testing.expectEqualSlices(u32, &.{ 2, 3 }, plans[0].tokens);
}
