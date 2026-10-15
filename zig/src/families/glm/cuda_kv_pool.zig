const std = @import("std");
const cuda = @import("cuda");
const cache = @import("cuda_cache.zig");
const state = @import("cuda_state.zig");
pub const extent = @import("cuda_kv_extent.zig");

pub const Shape = struct {
    rows: usize,
    parts: usize,
    layers: usize,
    latent_stride: usize,
    pooled_stride: usize,
    latent_bytes: usize,
    layer_bytes: usize,
    bytes: usize,

    pub fn init(rows: usize, parts: usize, layers: usize, format: cache.Format) !Shape {
        const p = try extent.Pool.init(rows, parts);
        if (layers == 0) return error.BadKvPool;
        const lat = format.rowBytes(512);
        const pk = format.rowBytes(128);
        const latent_bytes = try aligned(try std.math.mul(usize, p.partRows(), lat));
        const pooled_bytes = try aligned(try std.math.mul(usize, rows / 4 + 2, pk));
        const layer_bytes = try std.math.add(usize, latent_bytes, pooled_bytes);
        return .{ .rows = rows, .parts = parts, .layers = layers, .latent_stride = lat, .pooled_stride = pk, .latent_bytes = latent_bytes, .layer_bytes = layer_bytes, .bytes = try std.math.mul(usize, layers, layer_bytes) };
    }
};

pub const Pool = struct {
    shape: Shape,
    regions: extent.Pool,
    buf: cuda.DeviceBuffer,
    rank: usize,
    format: cache.Format,

    pub fn init(d: *const cuda.Driver, shape: Shape, rank: usize, format: cache.Format) !Pool {
        var buf = try cuda.DeviceBuffer.alloc(d, shape.bytes);
        errdefer buf.free();
        try buf.fill8(0, null);
        return .{ .shape = shape, .regions = try extent.Pool.init(shape.rows, shape.parts), .buf = buf, .rank = rank, .format = format };
    }

    pub fn deinit(p: *Pool) void {
        p.buf.free();
        p.* = undefined;
    }

    pub fn latent(p: *const Pool, layer: usize, r: extent.Region) u64 {
        if (p.shape.parts > 1 and p.regions.owner(r) != p.rank) return 0;
        const row = if (p.shape.parts > 1) r.base % p.regions.partRows() else r.base;
        return p.buf.ptr + layer * p.shape.layer_bytes + row * p.shape.latent_stride;
    }

    pub fn pooled(p: *const Pool, layer: usize, r: extent.Region) u64 {
        return p.buf.ptr + layer * p.shape.layer_bytes + p.shape.latent_bytes + r.base / 4 * p.shape.pooled_stride;
    }

    pub fn bind(p: *const Pool, st: *state.State, r: extent.Region) void {
        std.debug.assert(st.caches.len + 1 == p.shape.layers);
        for (st.caches, 0..) |*c, layer| p.bindLayer(c, layer, r);
        p.bindLayer(&st.mtp_caches, st.caches.len, r);
        st.capacity = @min(st.layout.capacity, r.rows);
    }

    fn bindLayer(p: *const Pool, c: *@import("cuda_dsa_block.zig").Caches, layer: usize, r: extent.Region) void {
        c.lc = .{ .ptr = p.latent(layer, r), .format = p.format };
        c.pk = .{ .ptr = p.pooled(layer, r), .format = p.format };
        c.pk_rows = r.rows / 4 + 2;
        c.owner = if (p.shape.parts > 1) @intCast(p.regions.owner(r)) else null;
    }
};

fn aligned(n: usize) !usize {
    return (try std.math.add(usize, n, 255)) & ~@as(usize, 255);
}

test "physical shared storage accounts owner latents once and pooled keys on every rank" {
    const replicated = try Shape.init(32768, 1, 3, .fp8);
    const sharded = try Shape.init(32768, 2, 3, .fp8);
    try std.testing.expectEqual(@as(usize, 16384 * 528), sharded.latent_bytes);
    try std.testing.expectEqual(replicated.bytes - 3 * 16384 * 528, sharded.bytes);
    try std.testing.expectError(error.BadKvPool, Shape.init(32769, 2, 3, .fp8));
}

test "two requests bind separate physical regions and reuse released rows without reallocating planes" {
    const M = @import("cuda_test_driver.zig").Memory;
    var d = M.driver();
    var p = try Pool.init(&d, try Shape.init(8192, 1, 2, .fp8), 0, .fp8);
    defer p.deinit();
    const first = try p.regions.plan(0, 2000);
    try p.regions.apply(0, first);
    const second = try p.regions.plan(1, 2000);
    try p.regions.apply(1, second);
    try std.testing.expectEqual(extent.Region{ .base = 2048, .rows = 2048 }, second);
    try std.testing.expectEqual(p.pooled(0, first) + 512 * 144, p.pooled(0, second));
    @memset(M.slice(p.latent(0, first), 2048 * 528), 37);
    for (M.slice(p.latent(0, second), 2048 * 528)) |v| try std.testing.expectEqual(@as(u8, 0), v);
    p.regions.release(0);
    const reused = try p.regions.plan(2, 2000);
    try std.testing.expectEqual(first, reused);
    try p.regions.apply(2, reused);
    try std.testing.expectEqual(p.latent(0, first), p.latent(0, reused));
    for (M.slice(p.latent(0, second), 2048 * 528)) |v| try std.testing.expectEqual(@as(u8, 0), v);
}
