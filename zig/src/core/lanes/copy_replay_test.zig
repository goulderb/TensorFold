//! Sixteen-row copied windows must commit the same sequence as one-token draws at every rejection boundary.
const std = @import("std");
const Config = @import("config.zig").Config;
const Engine = @import("engine.zig").Engine;
const Stream = @import("stream.zig").Stream;
const Proposer = @import("proposer.zig").Proposer;
const Sampling = @import("sampling.zig").Sampling;
const fake = @import("fake.zig");
const a = std.testing.allocator;

const Copies = struct {
    reject: usize,
    sampling: ?Sampling,
    out: [15]u32 = undefined,
    first_accepted: ?i64 = null,
    fn propose(ptr: *anyopaque, context: []const u32, room: i64) ![]const u32 {
        const p: *Copies = @ptrCast(@alignCast(ptr));
        if (room <= 0) return p.out[0..0];
        const n: usize = @intCast(@min(room, p.out.len));
        var history: [128]u32 = undefined;
        if (context.len + n > history.len) return error.TestUnexpectedResult;
        @memcpy(history[0..context.len], context);
        for (p.out[0..n], 0..) |*t, i| {
            const at = context.len + i;
            t.* = fake.next(history[0..at], p.sampling, at);
            if (i == p.reject) t.* = (t.* + 1) % fake.vocab;
            history[at] = t.*;
        }
        return p.out[0..n];
    }
    fn observe(ptr: *anyopaque, _: i64, accepted: i64) void {
        const p: *Copies = @ptrCast(@alignCast(ptr));
        if (p.first_accepted == null) p.first_accepted = accepted;
    }
    fn lastMatch(_: *anyopaque) i64 {
        return 8;
    }
    fn proposer(p: *Copies) Proposer {
        return .{ .ptr = p, .vtable = &.{ .propose = propose, .observe = observe, .last_match = lastMatch, .priced = true } };
    }
};

test "fifteen copied drafts replay serial greedy and keyed draws after every possible rejection" {
    const prompt = [_]u32{ 8, 3, 5 };
    for ([_]?Sampling{ null, .{ .seed = 23, .temperature = 1 } }) |sampling| {
        for (0..16) |reject| {
            var cfg = try Config.init(a, .{ .exact_width = 16, .gpu_tokens = false, .mtp = false }, 16, 7);
            defer cfg.deinit(a);
            var target: fake.Fake = .{ .gpa = a };
            defer target.deinit();
            var clock: fake.FixedClock = .{};
            var engine = Engine.init(a, &cfg, target.backend(), clock.clock());
            defer engine.deinit();
            var copies: Copies = .{ .reject = reject, .sampling = sampling };
            var s = try Stream.init(a, .{ .id = "wide-copy", .prompt = &prompt, .max_new = 64, .sampling = sampling, .proposer = copies.proposer() });
            defer s.deinit(a);
            try engine.addStream(&s);
            var widest: usize = 0;
            var steps: usize = 0;
            while (engine.activeCount() != 0) {
                if (steps == 128) return error.TestUnexpectedResult;
                try engine.step();
                steps += 1;
                widest = @max(widest, target.last_rows);
            }
            try std.testing.expectEqual(@as(usize, 16), widest);
            try std.testing.expectEqual(@as(i64, @intCast(reject)), copies.first_accepted.?);
            try std.testing.expectEqual(@as(usize, 64), s.emitted().len);
            var history: [128]u32 = undefined;
            @memcpy(history[0..prompt.len], &prompt);
            for (s.emitted(), 0..) |token, i| {
                const at = prompt.len + i;
                try std.testing.expectEqual(fake.next(history[0..at], sampling, at), token);
                history[at] = token;
            }
        }
    }
}
