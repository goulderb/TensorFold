//! Bounded shared-verify routing snapshots. Diagnostic copies are never benchmark timings.
const std = @import("std");
const cuda = @import("cuda");

pub const max_experts = 1024;
pub const max_samples = 32;

pub fn samples(raw: ?[]const u8) !usize {
    const text = raw orelse return 0;
    const n = std.fmt.parseInt(usize, text, 10) catch return error.BadRouteSamples;
    if (n > max_samples) return error.BadRouteSamples;
    return n;
}

pub fn participants(raw: ?[]const u8) !usize {
    const text = raw orelse return 1;
    const n = std.fmt.parseInt(usize, text, 10) catch return error.BadRouteParticipants;
    if (n == 0 or n > 8) return error.BadRouteParticipants;
    return n;
}

pub const Summary = struct {
    members: [max_experts]u32 = @splat(0),
    active: usize = 0,
    items: usize = 0,
    pairs: usize = 0,
    shared_pairs: usize = 0,
    tile32_items: usize = 0,
    tile64_items: usize = 0,
};

pub fn summarize(counts: [2]i32, triples: []const i32, experts: usize, rows: usize, topk: usize) !Summary {
    if (experts == 0 or experts > max_experts or counts[0] < 0 or counts[1] < 0) return error.BadRoutePlan;
    const n: usize = @intCast(counts[0]);
    if (n > triples.len / 3) return error.BadRoutePlan;
    var result: Summary = .{};
    var offset: usize = 0;
    var used: usize = 0;
    var previous: i32 = -1;
    for (0..n) |i| {
        const e = triples[i * 3];
        const first = triples[i * 3 + 1];
        const count = triples[i * 3 + 2];
        if (e < previous or e < 0 or e > experts or first < 0 or first != offset or count < 1 or count > 16) return error.BadRoutePlan;
        if (e != previous) used += 1;
        previous = e;
        offset += @intCast(count);
        if (e == experts) {
            result.shared_pairs += @intCast(count);
        } else {
            result.members[@intCast(e)] += @intCast(count);
            result.pairs += @intCast(count);
            result.items += 1;
        }
    }
    if (result.pairs != rows * topk or result.shared_pairs != rows or used != counts[1]) return error.BadRoutePlan;
    for (result.members[0..experts]) |count| if (count > 0) {
        result.active += 1;
        result.tile32_items += (count + 31) / 32;
        result.tile64_items += (count + 63) / 64;
    };
    return result;
}

const Entry = struct { layer: usize, experts: usize, topk: usize, capacity: usize };

pub const Trace = struct {
    gpa: std.mem.Allocator,
    pin: cuda.HostBuffer,
    entries: []Entry,
    rank: usize,
    stride: usize,
    max_rows: usize,
    remaining: usize,
    min_participants: usize = 1,
    sample: usize = 0,
    rows: usize = 0,
    used: usize = 0,
    active: bool = false,

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, rank: usize, layers: usize, rows: usize, experts: usize, topk: usize, limit: usize) !Trace {
        if (layers == 0 or layers > 128 or rows == 0 or rows > 64 or experts == 0 or experts > max_experts or topk == 0 or topk > experts or limit == 0 or limit > max_samples) return error.BadRouteBounds;
        const capacity = cuda.grouped.maxItems(rows * (topk + 1), experts + 1, 16);
        const stride = (2 + capacity * 3) * 4;
        const entries = try gpa.alloc(Entry, layers);
        errdefer gpa.free(entries);
        return .{ .gpa = gpa, .pin = try cuda.HostBuffer.alloc(d, layers * stride), .entries = entries, .rank = rank, .stride = stride, .max_rows = rows, .remaining = limit };
    }

    pub fn deinit(t: *Trace) void {
        t.pin.free();
        t.gpa.free(t.entries);
        t.* = undefined;
    }

    pub fn begin(t: *Trace, rows: usize) ?*Trace {
        if (t.remaining == 0 or rows < 8 or rows > t.max_rows) return null;
        t.rows = rows;
        t.used = 0;
        t.active = true;
        t.remaining -= 1;
        t.sample += 1;
        return t;
    }

    pub fn capture(t: *Trace, stream: cuda.Stream, layer: usize, experts: usize, topk: usize, plan: cuda.grouped.Plan) !void {
        if (!t.active or t.used >= t.entries.len) return error.BadRouteCapture;
        const capacity = cuda.grouped.maxItems(t.rows * (topk + 1), experts + 1, 16);
        if ((2 + capacity * 3) * 4 > t.stride) return error.BadRouteCapture;
        const at = t.used * t.stride;
        const counts: cuda.DeviceBuffer = .{ .d = stream.d, .ptr = plan.counts, .len = 8, .borrowed = true };
        const items: cuda.DeviceBuffer = .{ .d = stream.d, .ptr = plan.items, .len = capacity * 12, .borrowed = true };
        try counts.downloadAsync(0, t.pin.bytes[at..][0..8], stream.handle);
        try items.downloadAsync(0, t.pin.bytes[at + 8 ..][0 .. capacity * 12], stream.handle);
        t.entries[t.used] = .{ .layer = layer, .experts = experts, .topk = topk, .capacity = capacity };
        t.used += 1;
    }

    pub fn abort(t: *Trace, stream: cuda.Stream) void {
        stream.synchronize() catch {};
        t.active = false;
    }

    fn read(t: *Trace, index: usize) !Summary {
        const entry = t.entries[index];
        const bytes = t.pin.bytes[index * t.stride ..][0 .. (2 + entry.capacity * 3) * 4];
        const words = std.mem.bytesAsSlice(i32, @as([]align(4) u8, @alignCast(bytes)));
        return summarize(words[0..2].*, words[2..], entry.experts, t.rows, entry.topk);
    }

    pub fn finish(t: *Trace, stream: cuda.Stream) !void {
        if (!t.active) return;
        try stream.synchronize();
        defer t.active = false;
        for (0..t.used) |i| _ = try t.read(i);
        for (t.entries[0..t.used], 0..) |entry, i| {
            const result = try t.read(i);
            const bytes = t.pin.bytes[i * t.stride ..][0 .. (2 + entry.capacity * 3) * 4];
            const words = std.mem.bytesAsSlice(i32, @as([]align(4) u8, @alignCast(bytes)));
            const line = try std.json.Stringify.valueAlloc(t.gpa, .{
                .kind = "layer",
                .rank = t.rank,
                .sample = t.sample,
                .layer = entry.layer,
                .rows = t.rows,
                .tile_pairs = @as(usize, 16),
                .topk = entry.topk,
                .plan_items = result.items + (result.shared_pairs + 15) / 16,
                .routed_items = result.items,
                .active_experts = result.active,
                .routed_pairs = result.pairs,
                .shared_pairs = result.shared_pairs,
                .tile32_items = result.tile32_items,
                .tile64_items = result.tile64_items,
                .members = result.members[0..entry.experts],
                .counts = words[0..2],
                .item_triples = words[2..][0 .. @as(usize, @intCast(words[0])) * 3],
            }, .{});
            defer t.gpa.free(line);
            std.log.info("GLM_ROUTE {s}", .{line});
        }
        std.log.info("GLM_ROUTE {{\"kind\":\"complete\",\"rank\":{d},\"sample\":{d},\"rows\":{d},\"layers\":{d}}}", .{ t.rank, t.sample, t.rows, t.used });
    }
};

test "routing report excludes shared and reconstructs multi-item experts" {
    const items = [_]i32{ 0, 0, 16, 0, 16, 1, 1, 17, 14, 2, 31, 16, 2, 47, 15, -99, -99, -99 };
    const result = try summarize(.{ 5, 3 }, &items, 2, 31, 1);
    try std.testing.expectEqualSlices(u32, &.{ 17, 14 }, result.members[0..2]);
    try std.testing.expectEqual(@as(usize, 3), result.items);
    try std.testing.expectEqual(@as(usize, 2), result.active);
    try std.testing.expectEqual(@as(usize, 31), result.pairs);
    try std.testing.expectEqual(@as(usize, 31), result.shared_pairs);
    try std.testing.expectEqual(@as(usize, 2), result.tile32_items);
    try std.testing.expectError(error.BadRoutePlan, summarize(.{ 7, 3 }, &items, 2, 31, 1));
    try std.testing.expectError(error.BadRoutePlan, summarize(.{ 5, 2 }, &items, 2, 31, 1));
    try std.testing.expectError(error.BadRoutePlan, summarize(.{ 5, 3 }, &items, 2, 30, 1));
    var corrupt = items;
    corrupt[6] = 3;
    try std.testing.expectError(error.BadRoutePlan, summarize(.{ 5, 3 }, &corrupt, 2, 31, 1));
    corrupt = items;
    corrupt[4] = 15;
    try std.testing.expectError(error.BadRoutePlan, summarize(.{ 5, 3 }, &corrupt, 2, 31, 1));
}

test "routing diagnostic participant bounds" {
    try std.testing.expectEqual(@as(usize, 1), try participants(null));
    try std.testing.expectEqual(@as(usize, 8), try participants("8"));
    try std.testing.expectError(error.BadRouteParticipants, participants("0"));
    try std.testing.expectError(error.BadRouteParticipants, participants("9"));
    try std.testing.expectError(error.BadRouteParticipants, participants("all"));
}

test "routing sample bounds and shared-row selection" {
    try std.testing.expectEqual(@as(usize, 0), try samples(null));
    try std.testing.expectEqual(@as(usize, 32), try samples("32"));
    try std.testing.expectError(error.BadRouteSamples, samples("33"));
    try std.testing.expectError(error.BadRouteSamples, samples("-1"));
    var t: Trace = .{ .gpa = undefined, .pin = undefined, .entries = undefined, .rank = 0, .stride = 0, .max_rows = 56, .remaining = 1 };
    try std.testing.expect(t.begin(7) == null);
    try std.testing.expect(t.begin(57) == null);
    try std.testing.expect(t.begin(8) != null);
    try std.testing.expect(t.begin(8) == null);
    std.mem.doNotOptimizeAway(&Trace.capture);
    std.mem.doNotOptimizeAway(&Trace.finish);
}
