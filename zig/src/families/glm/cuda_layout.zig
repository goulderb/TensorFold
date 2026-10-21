const std = @import("std");
const cache = @import("cuda_cache.zig");

pub const Lengths = struct { main: usize, mtp: usize };
pub const Plane = enum { fixed, latent, keys, gates, pooled };
pub const Range = struct { plane: Plane = .fixed, layer: usize = 0, offset: usize = 0, saved: usize, bytes: usize };

pub const Layout = struct {
    capacity: usize,
    ring_rows: usize,
    dsa_layers: usize,
    fixed: [9]usize,
    fixed_end: usize,
    cache_offsets: [4]usize,
    cache_stride: usize,
    latent_stride: usize,
    index_stride: usize,
    pooled_stride: usize,
    mtp_pos: usize,
    total_bytes: usize,
    external_planes: bool = false,

    pub fn init(fixed_sizes: [9]usize, capacity: usize, max_window: usize, dsa_layers: usize, format: cache.Format, latent_width: usize, index_width: usize) !Layout {
        if (capacity == 0 or max_window == 0 or latent_width == 0 or index_width == 0) return error.InvalidGeometry;
        const ring = @min(capacity, (try std.math.add(usize, max_window, 3 + 63)) & ~@as(usize, 63));
        const lat = try rowBytes(format, latent_width);
        const index = try std.math.mul(usize, index_width, 2);
        const pooled = try rowBytes(format, index_width);
        var fixed: [9]usize = undefined;
        var end: usize = 0;
        for (fixed_sizes, &fixed) |n, *offset| {
            offset.* = end;
            end = try std.math.add(usize, end, try aligned(n));
        }
        const sizes = [_]usize{
            try std.math.mul(usize, capacity, lat),
            try std.math.mul(usize, ring, index),
            try std.math.mul(usize, ring, index),
            try std.math.mul(usize, try std.math.add(usize, capacity / 4, 2), pooled),
        };
        var offsets: [4]usize = undefined;
        var stride: usize = 0;
        for (sizes, &offsets) |n, *offset| {
            offset.* = stride;
            stride = try std.math.add(usize, stride, try aligned(n));
        }
        const mtp_pos = try std.math.add(usize, end, try std.math.mul(usize, try std.math.add(usize, dsa_layers, 1), stride));
        return .{ .capacity = capacity, .ring_rows = ring, .dsa_layers = dsa_layers, .fixed = fixed, .fixed_end = end, .cache_offsets = offsets, .cache_stride = stride, .latent_stride = lat, .index_stride = index, .pooled_stride = pooled, .mtp_pos = mtp_pos, .total_bytes = try std.math.add(usize, mtp_pos, 256) };
    }

    /// Keep recurrent state and raw rings private; latent/pooled planes will be bound from the shared pool.
    pub fn external(layout: Layout) !Layout {
        var out = layout;
        const raw = try aligned(try std.math.mul(usize, layout.ring_rows, layout.index_stride));
        out.cache_offsets = .{ 0, 0, raw, 0 };
        out.cache_stride = try std.math.mul(usize, raw, 2);
        out.mtp_pos = try std.math.add(usize, layout.fixed_end, try std.math.mul(usize, layout.dsa_layers + 1, out.cache_stride));
        out.total_bytes = try std.math.add(usize, out.mtp_pos, 256);
        out.external_planes = true;
        return out;
    }

    pub fn address(layout: Layout, r: Range) usize {
        std.debug.assert(!layout.external_planes);
        return r.offset + if (r.plane == .fixed) @as(usize, 0) else layout.cacheOffsets(r.layer)[@backingInt(r.plane) - 1];
    }

    pub fn cacheOffsets(layout: Layout, layer: usize) [4]usize {
        std.debug.assert(layer <= layout.dsa_layers);
        var out = layout.cache_offsets;
        for (&out) |*offset| offset.* += layout.fixed_end + layer * layout.cache_stride;
        return out;
    }

    pub fn snapshot(layout: Layout, lengths: Lengths) !Plan {
        if (lengths.main > layout.capacity or lengths.mtp > layout.capacity) return error.PastCapacity;
        return .{ .layout = layout, .lengths = lengths };
    }
};

pub const Plan = struct {
    layout: Layout,
    lengths: Lengths,

    pub fn ranges(plan: Plan) Iterator {
        return plan.rangesFor(true);
    }

    pub fn rangesFor(plan: Plan, latent_resident: bool) Iterator {
        return .{ .plan = plan, .latent_resident = latent_resident };
    }

    pub fn byteCount(plan: Plan) usize {
        return plan.bytesFor(true);
    }

    pub fn bytesFor(plan: Plan, latent_resident: bool) usize {
        var it = plan.rangesFor(latent_resident);
        var count: usize = 0;
        while (it.next()) |r| count += r.bytes;
        return count;
    }
};

pub const Iterator = struct {
    plan: Plan,
    stage: usize = 0,
    saved: usize = 0,
    latent_resident: bool = true,

    pub fn next(it: *Iterator) ?Range {
        const l = it.plan.layout;
        while (it.stage < 1 + (l.dsa_layers + 1) * 4) {
            const stage = it.stage;
            it.stage += 1;
            var offset: usize = 0;
            var plane: Plane = .fixed;
            var layer: usize = 0;
            var bytes: usize = l.fixed_end;
            if (stage > 0) {
                layer = (stage - 1) / 4;
                const field = (stage - 1) % 4;
                const n = if (layer == l.dsa_layers) it.plan.lengths.mtp else it.plan.lengths.main;
                plane = @fromBackingInt(@intCast(field + 1));
                if (plane == .latent and !it.latent_resident) continue;
                bytes = switch (field) {
                    0 => n * l.latent_stride,
                    1, 2 => n % 4 * l.index_stride,
                    3 => n / 4 * l.pooled_stride,
                    else => unreachable,
                };
                if (field == 1 or field == 2) offset = (n - n % 4) % l.ring_rows * l.index_stride;
            }
            if (bytes == 0) continue;
            const r: Range = .{ .plane = plane, .layer = layer, .offset = offset, .saved = it.saved, .bytes = bytes };
            it.saved += bytes;
            return r;
        }
        return null;
    }
};

fn aligned(n: usize) !usize {
    return (try std.math.add(usize, n, 255)) & ~@as(usize, 255);
}

fn rowBytes(format: cache.Format, width: usize) !usize {
    return switch (format) {
        .bf16 => std.math.mul(usize, width, 2),
        .fp8 => std.math.add(usize, width, 16),
    };
}

test "bounded rings use the widest window plus the prior incomplete pool" {
    const l = try Layout.init(.{ 4, 0, 0, 0, 0, 0, 0, 0, 0 }, 262144, 2048, 1, .fp8, 512, 128);
    try std.testing.expectEqual(@as(usize, 2112), l.ring_rows);
    const small = try Layout.init(.{ 4, 0, 0, 0, 0, 0, 0, 0, 0 }, 17, 2048, 1, .bf16, 512, 128);
    try std.testing.expectEqual(@as(usize, 17), small.ring_rows);
    try std.testing.expectEqual(@as(usize, 256), l.fixed_end);
    try std.testing.expectEqual(@as(usize, 528), l.latent_stride);
    try std.testing.expectEqual(@as(usize, 256), l.index_stride);
    try std.testing.expectEqual(@as(usize, 144), l.pooled_stride);
    try std.testing.expectError(error.InvalidGeometry, Layout.init(@splat(0), 0, 2048, 1, .fp8, 512, 128));
    try std.testing.expectError(error.Overflow, Layout.init(@splat(0), std.math.maxInt(usize), 2048, 1, .bf16, 512, 128));
}

test "main and MTP prefixes retain independent incomplete pools" {
    const l = try Layout.init(.{ 4, 3, 5, 0, 0, 0, 0, 0, 0 }, 8192, 2048, 2, .fp8, 512, 128);
    const plan = try l.snapshot(.{ .main = 2115, .mtp = 7 });
    var it = plan.ranges();
    try std.testing.expectEqual(Range{ .saved = 0, .bytes = 768 }, it.next().?);
    try std.testing.expectEqual(Range{ .plane = .latent, .saved = 768, .bytes = 2115 * 528 }, it.next().?);
    try std.testing.expectEqual(@as(usize, 0), it.next().?.offset); // unfinished pool wraps to ring slots 0..2
    try std.testing.expectEqual(@as(usize, 0), it.next().?.offset);
    try std.testing.expectEqual(@as(usize, 528 * 144), it.next().?.bytes);
    const main_bytes = 2115 * 528 + 2 * 3 * 256 + 528 * 144;
    const mtp_bytes = 7 * 528 + 2 * 3 * 256 + 144;
    try std.testing.expectEqual(@as(usize, 768 + 2 * main_bytes + mtp_bytes), plan.byteCount());
    try std.testing.expectError(error.PastCapacity, l.snapshot(.{ .main = 8193, .mtp = 0 }));
    try std.testing.expectError(error.PastCapacity, l.snapshot(.{ .main = 0, .mtp = 8193 }));
}

test "prefix packing restores only visible bytes after storage is reused" {
    for ([_]cache.Format{ .bf16, .fp8 }) |format| {
        const l = try Layout.init(.{ 4, 17, 31, 9, 5, 3, 2, 1, 0 }, 128, 8, 2, format, 4, 4);
        const live = try std.testing.allocator.alloc(u8, l.total_bytes);
        defer std.testing.allocator.free(live);
        for (0..129) |at| {
            const plan = try l.snapshot(.{ .main = at, .mtp = 128 - at });
            const saved = try std.testing.allocator.alloc(u8, plan.byteCount());
            defer std.testing.allocator.free(saved);
            for (live, 0..) |*b, i| b.* = @truncate(i *% 37 +% at);
            var save = plan.ranges();
            var end: usize = 0;
            while (save.next()) |r| {
                try std.testing.expectEqual(end, r.saved);
                try std.testing.expect(l.address(r) + r.bytes <= live.len);
                @memcpy(saved[r.saved..][0..r.bytes], live[l.address(r)..][0..r.bytes]);
                end += r.bytes;
            }
            try std.testing.expectEqual(plan.byteCount(), end);
            @memset(live, 0xa5);
            var restore = plan.ranges();
            while (restore.next()) |r| @memcpy(live[l.address(r)..][0..r.bytes], saved[r.saved..][0..r.bytes]);
            var verify = plan.ranges();
            while (verify.next()) |r| for (live[l.address(r)..][0..r.bytes], l.address(r)..) |b, i| {
                try std.testing.expectEqual(@as(u8, @truncate(i *% 37 +% at)), b);
            };
            try std.testing.expectEqual(@as(u8, 0xa5), live[l.mtp_pos]);
        }
    }
}

test "shared layouts retain private rings and resolve independent resident snapshot bytes" {
    const full = try Layout.init(.{ 4, 512, 512, 256, 0, 0, 0, 0, 0 }, 8192, 2048, 2, .fp8, 512, 128);
    const shared = try full.external();
    try std.testing.expect(shared.total_bytes < full.total_bytes);
    try std.testing.expectEqual(@as(usize, 2112 * 256 * 2), shared.cache_stride);
    const plan = try shared.snapshot(.{ .main = 2115, .mtp = 7 });
    try std.testing.expectEqual(plan.byteCount() - (2 * 2115 + 7) * 528, plan.bytesFor(false));
    var it = plan.rangesFor(false);
    var end: usize = 0;
    while (it.next()) |r| {
        try std.testing.expect(r.plane != .latent);
        try std.testing.expectEqual(end, r.saved);
        end += r.bytes;
    }
    try std.testing.expectEqual(plan.bytesFor(false), end);
}
