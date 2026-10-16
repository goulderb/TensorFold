//! Every rank votes after rank-zero vision setup, including on a local failure.
pub const Vote = struct {
    ctx: *anyopaque,
    ready: *const fn (*anyopaque, bool) anyerror!bool,
};

pub fn paired(problem: ?anyerror, vote: Vote) !void {
    const ready = try vote.ready(vote.ctx, problem == null);
    if (!ready) return problem orelse error.PeerVisionOpen;
    if (problem) |err| return err;
}

test "rank-zero vision open failure and peer refusal both vote before proceeding" {
    const std = @import("std");
    const Mock = struct {
        calls: usize = 0,
        valid: bool = true,
        fn ready(ptr: *anyopaque, valid: bool) !bool {
            const self: *@This() = @ptrCast(@alignCast(ptr));
            self.calls += 1;
            self.valid = valid;
            return false;
        }
    };
    var leader: Mock = .{};
    try std.testing.expectError(error.MissingVisionCodecPath, paired(error.MissingVisionCodecPath, .{ .ctx = &leader, .ready = Mock.ready }));
    try std.testing.expectEqual(@as(usize, 1), leader.calls);
    try std.testing.expect(!leader.valid);
    var follower: Mock = .{};
    try std.testing.expectError(error.PeerVisionOpen, paired(null, .{ .ctx = &follower, .ready = Mock.ready }));
    try std.testing.expectEqual(@as(usize, 1), follower.calls);
    try std.testing.expect(follower.valid);
}

test "a failed vote propagates without a second collective" {
    const std = @import("std");
    const Mock = struct {
        calls: usize = 0,
        fn ready(ptr: *anyopaque, _: bool) !bool {
            const self: *@This() = @ptrCast(@alignCast(ptr));
            self.calls += 1;
            return error.CollectiveFailed;
        }
    };
    var mock: Mock = .{};
    try std.testing.expectError(error.CollectiveFailed, paired(error.MissingVisionCodecPath, .{ .ctx = &mock, .ready = Mock.ready }));
    try std.testing.expectEqual(@as(usize, 1), mock.calls);
}
