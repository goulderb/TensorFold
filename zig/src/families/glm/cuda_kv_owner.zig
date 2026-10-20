const std = @import("std");
const cuda = @import("cuda");

pub fn failPair(err: anyerror) void {
    std.log.err("GLM paired cache operation failed ({s}); terminating this rank", .{@errorName(err)});
    std.process.exit(1);
}

pub const Work = struct { ctx: *anyopaque, run: *const fn (*anyopaque, u64, u64) anyerror!void };

pub const Transport = struct {
    d: *const cuda.Driver,
    lib: *const cuda.nccl.Library,
    comm: cuda.nccl.Comm,
    rank: usize,
    tp_group: ?*const cuda.tp_group.Group = null,
    failure: *const fn (anyerror) void = failPair,

    /// Each head rank keeps its own result; only the cache owner executes the existing attention kernels.
    pub fn attention(t: Transport, stream: cuda.Stream, owner: u32, qa: u64, out: u64, staged_q: u64, staged_out: u64, bytes: usize, work: Work) !void {
        errdefer t.failure(error.PairedTransferFailed);
        if (t.tp_group) |group| {
            if (group.world != 2 or group.rank != t.rank or owner >= 2) return error.BadOwnerTransport;
            const queries = [2]u64{ staged_q, staged_q + bytes };
            const local_results = [2]u64{ staged_out, staged_out + bytes };
            const gathered_results = staged_out + 2 * bytes;
            try group.allGather(stream, qa, staged_q, bytes);
            for (0..2) |head_rank| {
                if (t.rank == owner) try work.run(work.ctx, queries[head_rank], local_results[head_rank]);
            }
            const peer = 1 - owner;
            const send = local_results[peer];
            if (t.rank != owner) try t.d.check(t.d.api.cuMemsetD8Async(send, 0, bytes, stream.handle), "clear nonowner attention result");
            try group.allGather(stream, send, gathered_results, bytes);
            const result = if (t.rank == owner) local_results[owner] else gathered_results + owner * bytes;
            try t.d.check(t.d.api.cuMemcpyDtoDAsync_v2(out, result, bytes, stream.handle), "copy owner attention result");
            return;
        }
        for (0..2) |head_rank| {
            if (head_rank == owner) {
                if (t.rank == owner) try work.run(work.ctx, qa, out);
                continue;
            }
            try t.broadcast(stream, qa, staged_q, bytes, head_rank);
            if (t.rank == owner) try work.run(work.ctx, staged_q, staged_out);
            try t.broadcast(stream, staged_out, if (t.rank == head_rank) out else staged_out, bytes, owner);
        }
    }

    pub fn prepared(t: Transport, stream: cuda.Stream, buffer: cuda.DeviceBuffer, valid: bool) !bool {
        errdefer t.failure(error.PairedTransferFailed);
        try t.d.check(t.d.api.cuMemsetD32Async(buffer.ptr, @intFromBool(valid), 1, stream.handle), "cuMemsetD32Async");
        var ready = true;
        for (0..2) |rank| {
            try t.broadcast(stream, buffer.ptr, buffer.ptr + 4, 4, rank);
            try stream.synchronize();
            var value: u32 = 0;
            try buffer.download(4, std.mem.asBytes(&value));
            ready = ready and value != 0;
        }
        return ready;
    }

    pub fn carry(t: Transport, stream: cuda.Stream, source_owner: u32, destination_owner: u32, source: u64, destination: u64, bytes: usize) !void {
        errdefer t.failure(error.PairedTransferFailed);
        return t.carryBounded(stream, source_owner, destination_owner, source, destination, bytes, 64 << 20);
    }

    fn carryBounded(t: Transport, stream: cuda.Stream, source_owner: u32, destination_owner: u32, source: u64, destination: u64, bytes: usize, limit: usize) !void {
        if (source_owner == destination_owner) {
            if (t.rank == source_owner and bytes > 0) try t.d.check(t.d.api.cuMemcpyDtoDAsync_v2(destination, source, bytes, stream.handle), "cuMemcpyDtoDAsync");
            return;
        }
        var offset: usize = 0;
        while (offset < bytes) {
            const n = @min(limit, bytes - offset);
            const from = if (t.rank == source_owner) source else destination;
            const to = if (t.rank == destination_owner) destination else source;
            try t.broadcast(stream, from + offset, to + offset, n, source_owner);
            offset += n;
        }
    }

    fn broadcast(t: Transport, stream: cuda.Stream, source: u64, destination: u64, bytes: usize, root: usize) !void {
        try t.lib.check(t.lib.api.ncclBroadcast(source, destination, bytes, .u8, @intCast(root), t.comm, stream.handle), "ncclBroadcast");
    }
};

const Mock = @import("cuda_test_collectives.zig").Collectives;

test "two ranks retain their own head outputs when either cache owner runs attention and carries encoded rows" {
    const Job = struct {
        transport: Transport,
        stream: cuda.Stream,
        owner: u32,
        q: [16]u8,
        out: [16]u8 = @splat(0xff),
        stage_q: [16]u8 = undefined,
        stage_out: [16]u8 = undefined,
        saved: [37]u8 = @splat(91),
        restored: [37]u8 = @splat(0),
        failed: ?anyerror = null,
        fn kernel(ptr: *anyopaque, q: u64, out: u64) !void {
            const j: *@This() = @ptrCast(@alignCast(ptr));
            if (j.transport.rank != j.owner) return error.NonownerExecutedAttention;
            const src: [*]const u8 = @ptrFromInt(q);
            const dst: [*]u8 = @ptrFromInt(out);
            for (0..16) |i| dst[i] = src[i] +% 17;
        }
        fn run(j: *@This()) void {
            j.execute() catch |err| {
                j.failed = err;
            };
        }
        fn execute(j: *@This()) !void {
            try j.transport.attention(j.stream, j.owner, @intFromPtr(&j.q), @intFromPtr(&j.out), @intFromPtr(&j.stage_q), @intFromPtr(&j.stage_out), 16, .{ .ctx = j, .run = kernel });
            try j.transport.carryBounded(j.stream, j.owner, 1 - j.owner, @intFromPtr(&j.saved), @intFromPtr(&j.restored), 37, 7);
        }
    };
    var d = @import("cuda_test_driver.zig").Memory.driver();
    var lib: cuda.nccl.Library = .{ .lib = undefined, .api = undefined };
    lib.api.ncclBroadcast = Mock.broadcast;
    for (0..2) |owner| {
        var channel: Mock.Channel = .{};
        var ends = [_]Mock.End{ .{ .rank = 0, .channel = &channel }, .{ .rank = 1, .channel = &channel } };
        var jobs: [2]Job = undefined;
        for (&jobs, &ends, 0..) |*j, *endpoint, rank| {
            j.* = .{ .transport = .{ .d = &d, .lib = &lib, .comm = @ptrCast(endpoint), .rank = rank }, .stream = .{ .d = &d, .handle = null }, .owner = @intCast(owner), .q = @splat(@intCast(1 + rank)) };
        }
        const a = try std.Thread.spawn(.{}, Job.run, .{&jobs[0]});
        const b = try std.Thread.spawn(.{}, Job.run, .{&jobs[1]});
        a.join();
        b.join();
        for (&jobs, 0..) |*j, rank| {
            try std.testing.expectEqual(@as(?anyerror, null), j.failed);
            for (j.out) |v| try std.testing.expectEqual(@as(u8, @intCast(18 + rank)), v);
        }
        for (jobs[1 - owner].restored) |v| try std.testing.expectEqual(@as(u8, 91), v);
        try std.testing.expectEqual(@as(usize, 8), channel.epoch.load(.acquire));
    }
}

test "an attention execution failure invokes the fatal pair policy before returning" {
    const Failure = struct {
        var called = false;
        fn fatal(_: anyerror) void {
            called = true;
        }
        fn kernel(_: *anyopaque, _: u64, _: u64) !void {
            return error.InjectedAttentionFailure;
        }
    };
    Failure.called = false;
    var marker: u8 = 0;
    const t: Transport = .{ .d = undefined, .lib = undefined, .comm = null, .rank = 0, .failure = Failure.fatal };
    try std.testing.expectError(error.InjectedAttentionFailure, t.attention(undefined, 0, 0, 0, 0, 0, 0, .{ .ctx = &marker, .run = Failure.kernel }));
    try std.testing.expect(Failure.called);
}

test "borrowed-group attention gathers rank-major queries and returns each rank's own owner result" {
    const Collective = struct {
        const Call = struct { source: u64, destination: u64, bytes: usize };
        const Channel = struct {
            mutex: std.atomic.Mutex = .unlocked,
            epoch: std.atomic.Value(usize) = .init(0),
            arrived: usize = 0,
            calls: [2]Call = undefined,
        };
        const End = struct { rank: usize, channel: *Channel };

        fn allGather(source: u64, destination: u64, bytes: usize, dtype: cuda.nccl.DataType, comm: cuda.nccl.Comm, _: cuda.abi.Stream) callconv(.c) cuda.nccl.Result {
            std.debug.assert(dtype == .u8);
            const endpoint: *End = @ptrCast(@alignCast(comm.?));
            const channel = endpoint.channel;
            while (!channel.mutex.tryLock()) std.Thread.yield() catch {};
            const before = channel.epoch.load(.acquire);
            channel.calls[endpoint.rank] = .{ .source = source, .destination = destination, .bytes = bytes };
            channel.arrived += 1;
            if (channel.arrived == 2) {
                std.debug.assert(channel.calls[0].bytes == bytes and channel.calls[1].bytes == bytes);
                for (channel.calls) |call| {
                    const destination_bytes: [*]u8 = @ptrFromInt(call.destination);
                    for (channel.calls, 0..) |peer, rank| {
                        const source_bytes: [*]const u8 = @ptrFromInt(peer.source);
                        @memcpy(destination_bytes[rank * bytes ..][0..bytes], source_bytes[0..bytes]);
                    }
                }
                channel.arrived = 0;
                channel.epoch.store(before + 1, .release);
            }
            channel.mutex.unlock();
            while (channel.epoch.load(.acquire) == before) std.Thread.yield() catch {};
            return 0;
        }
    };
    const Job = struct {
        transport: Transport,
        stream: cuda.Stream,
        owner: u32,
        q: [16]u8,
        out: [16]u8 = @splat(0xff),
        stage_q: [32]u8 = undefined,
        stage_out: [64]u8 = undefined,
        canary: [16]u8 = @splat(0xa5),
        failed: ?anyerror = null,
        fn kernel(ptr: *anyopaque, q: u64, out: u64) !void {
            const j: *@This() = @ptrCast(@alignCast(ptr));
            if (j.transport.rank != j.owner) return error.NonownerExecutedAttention;
            const src: [*]const u8 = @ptrFromInt(q);
            const dst: [*]u8 = @ptrFromInt(out);
            for (0..16) |i| dst[i] = src[i] +% 17;
        }
        fn run(j: *@This()) void {
            j.transport.attention(j.stream, j.owner, @intFromPtr(&j.q), @intFromPtr(&j.out), @intFromPtr(&j.stage_q), @intFromPtr(&j.stage_out), 16, .{ .ctx = j, .run = kernel }) catch |err| {
                j.failed = err;
            };
        }
    };
    var d = @import("cuda_test_driver.zig").Memory.driver();
    var lib: cuda.nccl.Library = .{ .lib = undefined, .api = undefined };
    lib.api.ncclAllGather = Collective.allGather;
    for (0..2) |owner| {
        var channel: Collective.Channel = .{};
        var ends = [_]Collective.End{ .{ .rank = 0, .channel = &channel }, .{ .rank = 1, .channel = &channel } };
        var groups: [2]cuda.tp_group.Group = undefined;
        var jobs: [2]Job = undefined;
        for (&jobs, &groups, &ends, 0..) |*job, *group, *endpoint, rank| {
            group.* = try cuda.tp_group.Group.borrow(&lib, @ptrCast(endpoint), rank, 2);
            job.* = .{
                .transport = .{ .d = &d, .lib = &lib, .comm = @ptrCast(endpoint), .rank = rank, .tp_group = group },
                .stream = .{ .d = &d, .handle = null },
                .owner = @intCast(owner),
                .q = @splat(@intCast(rank + 1)),
            };
        }
        const a = try std.Thread.spawn(.{}, Job.run, .{&jobs[0]});
        const b = try std.Thread.spawn(.{}, Job.run, .{&jobs[1]});
        a.join();
        b.join();
        for (&jobs, 0..) |*job, rank| {
            try std.testing.expectEqual(@as(?anyerror, null), job.failed);
            for (job.out) |value| try std.testing.expectEqual(@as(u8, @intCast(rank + 18)), value);
            try std.testing.expect(std.mem.allEqual(u8, &job.canary, 0xa5));
        }
        try std.testing.expectEqual(@as(usize, 2), channel.epoch.load(.acquire));
    }
}
