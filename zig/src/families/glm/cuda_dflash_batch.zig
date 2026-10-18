const std = @import("std");
const state = @import("cuda_dflash_state.zig");
const policy = @import("cuda_draft_policy.zig");
const Sampling = @import("cuda_sampler.zig").Sampling;
const verify = @import("cuda_verify_plan.zig");

pub const max_lanes = 8;
pub const max_depth = verify.max_drafts;
pub const vocab = 154880;
pub const Request = struct {
    slot: usize,
    end: usize,
    pending: u32,
    depth: usize,
    sampling: ?Sampling,
};
pub const Result = struct { tokens: [max_depth]u32 = undefined, count: usize = 0 };

pub const Plan = struct {
    requests: [max_lanes]Request = undefined,
    original: [max_lanes]usize = undefined,
    count: usize = 0,
    total: usize = 0,
    capacity: usize = 0,

    pub fn build(lanes: *const [max_lanes]state.Lane, capacity: usize, requests: []const Request) !Plan {
        return buildWithPlan(lanes, capacity, requests, .baseline);
    }

    pub fn buildWithPlan(lanes: *const [max_lanes]state.Lane, capacity: usize, requests: []const Request, verify_plan: verify.Plan) !Plan {
        if (requests.len == 0 or requests.len > max_lanes) return error.BadDraftBatch;
        var seen: [max_lanes]bool = @splat(false);
        var plan: Plan = .{ .total = requests.len, .capacity = capacity };
        for (requests, 0..) |r, i| {
            if (r.slot >= max_lanes or seen[r.slot]) return error.BadDraftSlot;
            seen[r.slot] = true;
            if (r.pending >= vocab or r.depth > verify_plan.drafts()) return error.BadDraftRequest;
            const lane = lanes[r.slot];
            if (!lane.healthy) return error.DraftFailed;
            if (lane.pending != null or lane.end != r.end) return error.BadDraftRequest;
            if (r.end > capacity) return error.BadDraftCapacity;
            if (r.depth == 0 or r.end == 0 or r.end >= capacity -| 1) continue;
            plan.requests[plan.count] = r;
            plan.original[plan.count] = i;
            plan.count += 1;
        }
        return plan;
    }

    pub const Metadata = struct {
        ids: [max_lanes * 8]u32 = undefined,
        positions: [max_lanes * 8]i64 = undefined,
        row_slots: [max_lanes * 8]i64 = undefined,
        begins: [max_lanes]i64 = undefined,
        block_slots: [max_lanes]i64 = undefined,
    };

    pub fn metadata(p: Plan, mask: u32) Metadata {
        var m: Metadata = .{};
        for (p.requests[0..p.count], 0..) |r, j| {
            for (0..8) |row| {
                m.ids[j * 8 + row] = if (row == 0) r.pending else mask;
                m.positions[j * 8 + row] = @intCast(r.end + row);
                m.row_slots[j * 8 + row] = @intCast(r.slot);
            }
            m.begins[j] = @intCast(r.end);
            m.block_slots[j] = @intCast(r.slot);
        }
        return m;
    }
};

pub fn select(p: Plan, rank_major: []const f32, projection: []const f32, world: usize, selector: policy.Selector, out: []Result) !void {
    if (out.len != p.total or p.count == 0 or rank_major.len != world * p.count * 7 * 32 or projection.len != p.count * 7 * 256) return error.BadDraftBatch;
    var merged: [max_lanes * 7 * 16]policy.Candidate = undefined;
    try policy.merge(rank_major, world, p.count * 7, 16, vocab, merged[0 .. p.count * 7 * 16]);
    var selected: [max_lanes]Result = @splat(.{});
    for (p.requests[0..p.count], 0..) |r, j| {
        const max_count = @min(r.depth, p.capacity - r.end - 1);
        selected[j].count = try policy.chain(merged[j * 7 * 16 ..][0 .. 7 * 16], projection[j * 7 * 256 ..][0 .. 7 * 256], selector, r.pending, r.end + 1, r.sampling, .{}, selected[j].tokens[0..max_count], null);
    }
    for (p.original[0..p.count], selected[0..p.count]) |i, result| out[i] = result;
}

test "noncontiguous physical slots map into packed graph rows" {
    var lanes: [8]state.Lane = @splat(.{});
    lanes[5].end = 40;
    lanes[1].end = 400;
    lanes[7].end = 2;
    const requests = [_]Request{
        .{ .slot = 5, .end = 40, .pending = 11, .depth = 3, .sampling = null },
        .{ .slot = 1, .end = 400, .pending = 22, .depth = 1, .sampling = null },
        .{ .slot = 7, .end = 2, .pending = 33, .depth = 2, .sampling = null },
    };
    const p = try Plan.build(&lanes, 1024, &requests);
    try std.testing.expectEqual(@as(usize, 3), p.count);
    const m = p.metadata(154856);
    try std.testing.expectEqualSlices(u32, &.{ 11, 154856, 154856, 154856, 154856, 154856, 154856, 154856, 22 }, m.ids[0..9]);
    try std.testing.expectEqual(@as(i64, 1), m.block_slots[1]);
    try std.testing.expectEqual(@as(i64, 402), m.positions[10]);
    try std.testing.expectEqual(@as(i64, 7), m.row_slots[16]);
}

test "a later duplicate or invalid lane leaves the batch unbuilt" {
    var lanes: [8]state.Lane = @splat(.{});
    lanes[5].end = 40;
    const good: Request = .{ .slot = 5, .end = 40, .pending = 11, .depth = 3, .sampling = null };
    try std.testing.expectError(error.BadDraftBatch, Plan.build(&lanes, 1024, &.{}));
    try std.testing.expectError(error.BadDraftSlot, Plan.build(&lanes, 1024, &.{ good, good }));
    try std.testing.expectError(error.BadDraftRequest, Plan.build(&lanes, 1024, &.{ good, .{ .slot = 1, .end = 0, .pending = vocab, .depth = 1, .sampling = null } }));
}

test "capacity and zero-depth lanes do not consume packed graph rows" {
    var lanes: [8]state.Lane = @splat(.{});
    lanes[5].end = 9;
    lanes[1].end = 9;
    lanes[7].end = 4;
    const p = try Plan.build(&lanes, 10, &.{
        .{ .slot = 5, .end = 9, .pending = 11, .depth = 3, .sampling = null },
        .{ .slot = 1, .end = 9, .pending = 22, .depth = 2, .sampling = null },
        .{ .slot = 7, .end = 4, .pending = 33, .depth = 0, .sampling = null },
    });
    try std.testing.expectEqual(@as(usize, 0), p.count);
}

test "rank-major merged candidates scatter in request order and failed selection publishes nothing" {
    var lanes: [8]state.Lane = @splat(.{});
    lanes[5].end = 40;
    lanes[1].end = 400;
    const p = try Plan.build(&lanes, 1024, &.{
        .{ .slot = 5, .end = 40, .pending = 11, .depth = 3, .sampling = null },
        .{ .slot = 1, .end = 400, .pending = 22, .depth = 1, .sampling = null },
    });
    var raw: [2 * 2 * 7 * 32]f32 = @splat(0);
    for (0..2) |rank| for (0..2) |item| for (0..7) |row| for (0..16) |j| {
        const base = ((rank * 2 + item) * 7 + row) * 32;
        raw[base + j] = if (rank == 1 and j == 0) 100 else -100;
        raw[base + 16 + j] = @bitCast(@as(u32, @intCast(100 + item * 10 + row * 16 + j)));
    };
    const projection: [2 * 7 * 256]f32 = @splat(0);
    const pred = try std.testing.allocator.alloc(f32, 512 * 256);
    defer std.testing.allocator.free(pred);
    const succ = try std.testing.allocator.alloc(f32, 512 * 256);
    defer std.testing.allocator.free(succ);
    @memset(pred, 0);
    @memset(succ, 0);
    const selector: policy.Selector = .{ .pred = pred, .succ = succ };
    var out: [2]Result = .{ .{ .count = 99 }, .{ .count = 99 } };
    try select(p, &raw, &projection, 2, selector, &out);
    try std.testing.expectEqual(@as(usize, 3), out[0].count);
    try std.testing.expectEqual(@as(usize, 1), out[1].count);
    try std.testing.expectEqual(@as(u32, 100), out[0].tokens[0]);
    try std.testing.expectEqual(@as(u32, 110), out[1].tokens[0]);
    raw[((1 * 2 + 1) * 7 + 0) * 32 + 16] = @bitCast(@as(u32, vocab));
    out = .{ .{ .count = 99 }, .{ .count = 99 } };
    try std.testing.expectError(error.BadCandidates, select(p, &raw, &projection, 2, selector, &out));
    try std.testing.expectEqual(@as(usize, 99), out[0].count);
    try std.testing.expectEqual(@as(usize, 99), out[1].count);
}
