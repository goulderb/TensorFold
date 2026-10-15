const std = @import("std");
const cuda = @import("cuda");
const engine = @import("cuda_engine.zig");
const follow = @import("cuda_follow.zig");
const Memory = @import("cuda_test_driver.zig").Memory;

test "spill flush precedes follower drops and a failed flush keeps their source ids" {
    const Fixture = struct {
        e: engine.Engine = undefined,
        fail: bool = false,
        flushed: bool = false,
        drops: usize = 0,

        fn flush(ptr: *anyopaque) !void {
            const f: *@This() = @ptrCast(@alignCast(ptr));
            try std.testing.expectEqualSlices(u32, &.{ 17, 23 }, f.e.dropped.items);
            if (f.fail) return error.SpillTransport;
            f.flushed = true;
        }
        fn apply(_: *anyopaque, _: engine.persistence.Command) !void {}
        fn forget(_: *anyopaque, _: u64, _: u32) void {}
        fn close(_: *anyopaque) void {}
        fn broadcast(source: u64, _: u64, bytes: usize, dtype: cuda.nccl.DataType, _: c_int, comm: cuda.nccl.Comm, _: cuda.abi.Stream) callconv(.c) cuda.nccl.Result {
            const f: *@This() = @ptrCast(@alignCast(comm.?));
            std.debug.assert(f.flushed and dtype == .u8);
            const header: *const follow.Header = @ptrFromInt(source);
            std.debug.assert(header.kind == @backingInt(follow.Kind.drop) and header.n == 2);
            if (bytes > @sizeOf(follow.Header)) {
                const ids: [*]const u32 = @ptrFromInt(source + @sizeOf(follow.Header));
                std.debug.assert(ids[0] == 17 and ids[1] == 23);
                f.drops += 1;
            }
            return 0;
        }
    };
    var d = Memory.driver();
    var lib: cuda.nccl.Library = .{ .lib = undefined, .api = undefined };
    lib.api.ncclBroadcast = Fixture.broadcast;
    var f: Fixture = .{};
    f.e.gpa = std.testing.allocator;
    f.e.dropped = .empty;
    defer f.e.dropped.deinit(f.e.gpa);
    try f.e.dropped.appendSlice(f.e.gpa, &.{ 17, 23 });
    f.e.pool = null;
    f.e.slot = 0;
    f.e.m.glue.put_words = .{ .d = &d, .handle = null };
    f.e.disk = .{ .ctx = &f, .tier = &f, .identity = @splat(0), .apply = Fixture.apply, .request = Fixture.apply, .flush = Fixture.flush, .forget = Fixture.forget, .close = Fixture.close };
    var ids: [2]u32 = undefined;
    var link: follow.Link = .{ .d = &d, .lib = &lib, .comm = @ptrCast(&f), .stream = .{ .d = &d, .handle = null }, .buf = try cuda.DeviceBuffer.alloc(&d, 72), .ids = &ids, .gpa = std.testing.allocator, .wait = true };
    defer link.buf.free();
    const leader: follow.Leader = .{ .e = &f.e, .link = &link };
    f.fail = true;
    try std.testing.expectError(error.SpillTransport, leader.flush());
    try std.testing.expectEqual(@as(usize, 0), f.drops);
    try std.testing.expectEqualSlices(u32, &.{ 17, 23 }, f.e.dropped.items);
    f.fail = false;
    try leader.flush();
    try std.testing.expectEqual(@as(usize, 1), f.drops);
    try std.testing.expectEqual(@as(usize, 0), f.e.dropped.items.len);
}
