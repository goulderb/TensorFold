const std = @import("std");
const cuda = @import("cuda");
const Span = @import("cuda_shard.zig").Span;
const Exchange = @import("cuda_forward.zig").Exchange;

const Pair = extern struct { score: u32, id: u32 };

pub const Context = struct {
    rank: usize,
    vocab: Span,
    top1: cuda.Function,
    exchange: Exchange,
    group: ?*const cuda.tp_group.Group = null,
};

pub const Greedy = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    world: usize,
    capacity: usize,
    local_bytes: usize,
    pairs: cuda.DeviceBuffer,
    host: []Pair,

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, world: usize, capacity: usize) !Greedy {
        if (world == 0 or world > @import("cuda_ranks.zig").max_world or capacity == 0 or capacity > @import("cuda_verify_plan.zig").max_rows) return error.BadGreedyShape;
        const padded = std.mem.alignForward(usize, capacity, 2);
        const local_bytes = padded * @sizeOf(Pair);
        var pairs = try cuda.DeviceBuffer.alloc(d, (world + 1) * local_bytes);
        errdefer pairs.free();
        try pairs.fill8(0, null);
        const host = try gpa.alloc(Pair, world * padded);
        return .{ .gpa = gpa, .d = d, .world = world, .capacity = capacity, .local_bytes = local_bytes, .pairs = pairs, .host = host };
    }

    pub fn deinit(g: *Greedy) void {
        g.pairs.free();
        g.gpa.free(g.host);
        g.* = undefined;
    }

    pub fn draw(g: *Greedy, stream: cuda.Stream, context: Context, logits: u64, rows: usize, out: []u32) !void {
        if (rows == 0 or rows > g.capacity or out.len < rows or context.rank >= g.world or context.vocab.hi <= context.vocab.lo or context.vocab.hi > std.math.maxInt(u32)) return error.BadGreedyShape;
        if (context.group) |group| {
            if (group.world != g.world or group.rank != context.rank) return error.TransportIdentityMismatch;
        } else if (g.world > 1 and context.exchange == .local) return error.NoGreedyTransport;
        const stride = std.mem.alignForward(usize, rows, 2);
        const bytes = stride * @sizeOf(Pair);
        for (0..rows) |row| {
            var args: cuda.Args = .{};
            args.add(logits + row * context.vocab.len() * 2);
            args.add(@as(i64, @intCast(context.vocab.len())));
            args.add(@as(u32, @intCast(context.vocab.lo)));
            args.add(g.pairs.ptr + row * @sizeOf(Pair));
            try cuda.launch.launch(context.top1, .{ .grid = .{ .x = 1 }, .block = .{ .x = 1024 } }, stream, &args);
        }
        const gathered = if (g.world == 1) g.pairs.ptr else g.pairs.ptr + g.local_bytes;
        if (g.world > 1) {
            if (context.group) |group| {
                try group.allGather(stream, g.pairs.ptr, gathered, bytes);
            } else switch (context.exchange) {
                .nccl => |x| try x.lib.check(x.lib.api.ncclAllGather(g.pairs.ptr, gathered, bytes, .u8, x.comm, stream.handle), "ncclAllGather greedy rows"),
                .local => unreachable,
            }
        }
        try stream.synchronize();
        if (context.group) |group| try group.check();
        const all = g.host[0 .. g.world * stride];
        try g.d.check(g.d.api.cuMemcpyDtoH_v2(all.ptr, gathered, all.len * @sizeOf(Pair)), "cuMemcpyDtoH greedy rows");
        select(all, stride, rows, out);
    }
};

fn select(all: []const Pair, stride: usize, rows: usize, out: []u32) void {
    for (0..rows) |row| {
        var pick = all[row];
        var rank: usize = 1;
        while (rank * stride < all.len) : (rank += 1) {
            const candidate = all[rank * stride + row];
            const value: f32 = @bitCast(candidate.score);
            const best: f32 = @bitCast(pick.score);
            if (value > best or (value == best and candidate.id < pick.id)) pick = candidate;
        }
        out[row] = pick.id;
    }
}

test "packed greedy reduction uses compact padded rank stride and global ID ties" {
    const all = [_]Pair{
        .{ .score = @bitCast(@as(f32, 5)), .id = 100 },
        .{ .score = @bitCast(@as(f32, -0.0)), .id = 20 },
        .{ .score = @bitCast(-std.math.inf(f32)), .id = 3 },
        .{ .score = @bitCast(std.math.inf(f32)), .id = 0 },
        .{ .score = @bitCast(@as(f32, 5)), .id = 4 },
        .{ .score = @bitCast(@as(f32, 0)), .id = 21 },
        .{ .score = @bitCast(std.math.inf(f32)), .id = 30 },
        .{ .score = @bitCast(std.math.inf(f32)), .id = 0 },
    };
    var out: [4]u32 = @splat(0xdeadbeef);
    select(&all, 4, 3, &out);
    try std.testing.expectEqualSlices(u32, &.{ 4, 20, 30, 0xdeadbeef }, &out);
}
