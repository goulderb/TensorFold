const std = @import("std");
const cuda = @import("cuda");

pub const Collectives = struct {
    const Call = struct { source: u64, destination: u64, bytes: usize, root: usize, reduce: bool = false };
    pub const Channel = struct {
        mutex: std.atomic.Mutex = .unlocked,
        epoch: std.atomic.Value(usize) = .init(0),
        arrived: usize = 0,
        calls: [2]Call = undefined,
    };
    pub const End = struct { rank: usize, channel: *Channel };

    pub fn broadcast(source: u64, destination: u64, bytes: usize, dtype: cuda.nccl.DataType, root: c_int, comm: cuda.nccl.Comm, _: cuda.abi.Stream) callconv(.c) cuda.nccl.Result {
        std.debug.assert(dtype == .u8);
        return arrive(source, destination, bytes, @intCast(root), comm, false);
    }

    pub fn allReduce(source: u64, destination: u64, n: usize, dtype: cuda.nccl.DataType, op: cuda.nccl.RedOp, comm: cuda.nccl.Comm, _: cuda.abi.Stream) callconv(.c) cuda.nccl.Result {
        std.debug.assert(n == 1 and dtype == .f64 and op == .max);
        return arrive(source, destination, 8, 0, comm, true);
    }

    fn arrive(source: u64, destination: u64, bytes: usize, root: usize, comm: cuda.nccl.Comm, reduce: bool) cuda.nccl.Result {
        const endpoint: *End = @ptrCast(@alignCast(comm.?));
        const c = endpoint.channel;
        while (!c.mutex.tryLock()) std.Thread.yield() catch {};
        const before = c.epoch.load(.acquire);
        c.calls[endpoint.rank] = .{ .source = source, .destination = destination, .bytes = bytes, .root = root, .reduce = reduce };
        c.arrived += 1;
        if (c.arrived == 2) {
            std.debug.assert(c.calls[0].bytes == c.calls[1].bytes and c.calls[0].root == c.calls[1].root and c.calls[0].reduce == c.calls[1].reduce);
            var maximum: f64 = 0;
            if (reduce) {
                const a: *const f64 = @ptrFromInt(c.calls[0].source);
                const b: *const f64 = @ptrFromInt(c.calls[1].source);
                maximum = @max(a.*, b.*);
            }
            const src = if (reduce) @intFromPtr(&maximum) else c.calls[root].source;
            for (c.calls) |call| if (call.destination != src) {
                const to: [*]u8 = @ptrFromInt(call.destination);
                const from: [*]const u8 = @ptrFromInt(src);
                @memcpy(to[0..bytes], from[0..bytes]);
            };
            c.arrived = 0;
            c.epoch.store(before + 1, .release);
        }
        c.mutex.unlock();
        while (c.epoch.load(.acquire) == before) std.Thread.yield() catch {};
        return 0;
    }
};
