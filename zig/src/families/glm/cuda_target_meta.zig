//! Stable target-graph tables. Sequence addresses are staged each round, never captured as kernel arguments.
const std = @import("std");
const cuda = @import("cuda");
const state = @import("cuda_state.zig");
const weights = @import("cuda_weights.zig");
const kda = @import("cuda_kda.zig");

pub const max_rows = 32;
pub const max_segments = 8;
pub const unowned = std.math.maxInt(u32);
pub const resident: u32 = 1;
pub const sparse: u32 = 2;

pub const Row = extern struct { segment: u32, local: u32 };
pub const KdaSegment = extern struct {
    row0: u32,
    rows: u32,
    conv: u64,
    state_in: u64,
    state_out: u64,
    proj: u64,
    k: u64,
    v: u64,
    g: u64,
    b: u64,
};
pub const DsaRow = extern struct {
    latent: u64,
    index_keys: u64,
    index_gates: u64,
    pooled: u64,
    position: u32,
    ring_rows: u32,
    pooled_rows: u32,
    owner: u32,
    flags: u32,
    pad: u32, // Active merge chunks (1..5), preserving the eager segment's reduction extent.
};
pub const Segment = struct { st: *state.State, row0: usize, rows: usize };

pub const Host = struct {
    gpa: std.mem.Allocator,
    rows: [max_rows]Row = undefined,
    kda: []KdaSegment,
    dsa: []DsaRow,
    kda_layers: usize,
    dsa_layers: usize,

    pub fn init(gpa: std.mem.Allocator, w: *const weights.Weights) !Host {
        var nk: usize = 0;
        var nd: usize = 0;
        for (w.layers) |l| switch (l.mixer) {
            .kda => nk += 1,
            .mla => nd += 1,
        };
        const ks = try gpa.alloc(KdaSegment, try std.math.mul(usize, nk, max_segments));
        errdefer gpa.free(ks);
        const ds = try gpa.alloc(DsaRow, try std.math.mul(usize, nd, max_rows));
        return .{ .gpa = gpa, .kda = ks, .dsa = ds, .kda_layers = nk, .dsa_layers = nd };
    }

    pub fn deinit(h: *Host) void {
        h.gpa.free(h.kda);
        h.gpa.free(h.dsa);
        h.* = undefined;
    }

    /// Validates all geometry before changing a sequence or publishing a device table.
    pub fn build(h: *Host, w: *const weights.Weights, segments: []const Segment, total: usize, sharded: bool) !void {
        if (total == 0 or total > max_rows or segments.len == 0 or segments.len > max_segments) return error.BadTargetGraphWindow;
        if (sharded and (w.share.world != 2 or w.share.rank >= 2)) return error.TargetGraphShardNeedsTp2;
        var at: usize = 0;
        for (segments, 0..) |seg, i| {
            const st = seg.st;
            if (seg.row0 != at or seg.rows == 0 or seg.rows > state.window_rows or seg.rows > total -| at) return error.BadTargetGraphWindow;
            if (st.pos > st.capacity or seg.rows > st.capacity - st.pos) return error.PastCapacity;
            if (st.kda_layers != h.kda_layers or st.caches.len != h.dsa_layers or st.cur.len != h.kda_layers or st.heads != w.share.kda.len()) return error.BadTargetGraphState;
            for (segments[0..i]) |other| if (st == other.st) return error.SlotTwice;
            at += seg.rows;
        }
        if (at != total) return error.BadTargetGraphWindow;
        @memset(&h.rows, std.mem.zeroes(Row));
        @memset(h.kda, std.mem.zeroes(KdaSegment));
        @memset(h.dsa, std.mem.zeroes(DsaRow));
        for (segments, 0..) |seg, index| {
            for (0..seg.rows) |r| h.rows[seg.row0 + r] = .{ .segment = @intCast(index), .local = @intCast(r) };
        }
        var ki: usize = 0;
        var di: usize = 0;
        for (w.layers) |layer| switch (layer.mixer) {
            .kda => {
                for (segments, 0..) |seg, index| {
                    const st = seg.st;
                    const current = st.cur[ki];
                    const kv = ki * state.window_rows * st.heads * kda.dk;
                    h.kda[ki * max_segments + index] = .{
                        .row0 = @intCast(seg.row0),
                        .rows = @intCast(seg.rows),
                        .conv = st.conv + ki * 3 * st.conv_c * 2,
                        .state_in = st.recAt(current, ki),
                        .state_out = st.recAt(1 - current, ki),
                        .proj = st.proj + ki * st.proj_layer * 2,
                        .k = st.save_k + kv * 4,
                        .v = st.save_v + kv * 2,
                        .g = st.save_g + kv * 4,
                        .b = st.save_b + ki * state.window_rows * st.heads * 4,
                    };
                }
                ki += 1;
            },
            .mla => |l| {
                for (segments) |seg| {
                    const st = seg.st;
                    const c = st.caches[di];
                    const owner = c.owner orelse unowned;
                    if ((sharded and owner >= 2) or (!sharded and owner != unowned)) return error.BadTargetGraphOwner;
                    const local = !sharded or owner == w.share.rank;
                    if ((local and c.lc.ptr == 0) or c.index.keys == 0 or c.index.gates == 0 or c.pk.ptr == 0) return error.BadTargetGraphPlacement;
                    const upto = st.pos + seg.rows;
                    // Eager dense attention reduces the whole segment's chunk bound, not each row's causal length.
                    const merge_chunks = if (upto > l.dense_limit) 5 else @min(5, upto / 512 + @intFromBool(upto % 512 != 0));
                    for (0..seg.rows) |r| {
                        const pos = st.pos + r;
                        h.dsa[di * max_rows + seg.row0 + r] = .{
                            .latent = c.lc.ptr,
                            .index_keys = c.index.keys,
                            .index_gates = c.index.gates,
                            .pooled = c.pk.ptr,
                            .position = std.math.cast(u32, pos) orelse return error.PastCapacity,
                            .ring_rows = std.math.cast(u32, c.index.rows) orelse return error.BadTargetGraphState,
                            .pooled_rows = std.math.cast(u32, c.pk_rows) orelse return error.BadTargetGraphState,
                            .owner = owner,
                            .flags = (if (local) resident else 0) | (if (pos >= l.dense_limit) sparse else 0),
                            .pad = @intCast(merge_chunks),
                        };
                    }
                }
                di += 1;
            },
        };
    }
};

pub const Meta = struct {
    host: Host,
    rows: cuda.DeviceBuffer,
    kda: cuda.DeviceBuffer,
    dsa: cuda.DeviceBuffer,

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, w: *const weights.Weights) !Meta {
        var host = try Host.init(gpa, w);
        errdefer host.deinit();
        var rows = try cuda.DeviceBuffer.alloc(d, max_rows * @sizeOf(Row));
        errdefer rows.free();
        var ks = try cuda.DeviceBuffer.alloc(d, @max(1, host.kda.len * @sizeOf(KdaSegment)));
        errdefer ks.free();
        const ds = try cuda.DeviceBuffer.alloc(d, @max(1, host.dsa.len * @sizeOf(DsaRow)));
        return .{ .host = host, .rows = rows, .kda = ks, .dsa = ds };
    }

    pub fn deinit(m: *Meta) void {
        m.dsa.free();
        m.kda.free();
        m.rows.free();
        m.host.deinit();
        m.* = undefined;
    }

    pub fn stage(m: *Meta, stream: cuda.Stream, w: *const weights.Weights, segments: []const Segment, total: usize, sharded: bool) !void {
        try m.host.build(w, segments, total, sharded);
        // uploadAsync owns the source until completion; neither caller stack nor next-round table edits can race it.
        try m.rows.uploadAsync(0, std.mem.sliceAsBytes(&m.host.rows), stream.handle);
        if (m.host.kda.len > 0) try m.kda.uploadAsync(0, std.mem.sliceAsBytes(m.host.kda), stream.handle);
        if (m.host.dsa.len > 0) try m.dsa.uploadAsync(0, std.mem.sliceAsBytes(m.host.dsa), stream.handle);
    }

    pub fn kdaLayer(m: *const Meta, layer: usize) u64 {
        std.debug.assert(layer < m.host.kda_layers);
        return m.kda.ptr + layer * max_segments * @sizeOf(KdaSegment);
    }

    pub fn dsaLayer(m: *const Meta, layer: usize) u64 {
        std.debug.assert(layer < m.host.dsa_layers);
        return m.dsa.ptr + layer * max_rows * @sizeOf(DsaRow);
    }
};

const Fixture = struct {
    cur: [1]u1 = .{0},
    caches: [1]@import("cuda_dsa_block.zig").Caches = undefined,
    layers: [2]weights.Layer = undefined,
    st: state.State = undefined,
    w: weights.Weights = undefined,

    fn init(f: *Fixture) !void {
        const c = @import("config.zig").Config{};
        f.w = .{ .gpa = std.testing.allocator, .config = c, .share = try @import("cuda_shard.zig").Share.of(&c, 0, 2), .layers = &f.layers };
        var kl: @import("cuda_kda_block.zig").Layer = undefined;
        kl.heads = f.w.share.kda.len();
        f.layers[0].mixer = .{ .kda = kl };
        var dl: @import("cuda_dsa_block.zig").Layer = undefined;
        dl.dense_limit = 2051;
        f.layers[1].mixer = .{ .mla = dl };
        f.caches[0] = .{ .lc = .{ .ptr = 0x10000, .format = .fp8 }, .index = .{ .keys = 0x20000, .gates = 0x30000, .rows = 2080 }, .pk = .{ .ptr = 0x40000, .format = .fp8 }, .pk_rows = 2048 };
        f.st.pos = 2048;
        f.st.capacity = 8192;
        f.st.kda_layers = 1;
        f.st.heads = f.w.share.kda.len();
        f.st.cur = &f.cur;
        f.st.caches = &f.caches;
        f.st.conv = 0x50000;
        f.st.conv_c = 3 * f.st.heads * kda.dk;
        f.st.rec = 0x60000;
        f.st.rec_layer = f.st.heads * kda.dk * kda.dv * 4;
        f.st.proj = 0x70000;
        f.st.proj_layer = state.window_rows * (3 * f.st.heads * kda.dk + 2 * kda.dk + f.st.heads);
        f.st.save_k = 0x80000;
        f.st.save_v = 0x90000;
        f.st.save_g = 0xa0000;
        f.st.save_b = 0xb0000;
    }
};

test "target table C ABI offsets and strides stay explicit" {
    try std.testing.expectEqual(@as(usize, 8), @sizeOf(Row));
    try std.testing.expectEqual(@as(usize, 72), @sizeOf(KdaSegment));
    try std.testing.expectEqual(@as(usize, 8), @offsetOf(KdaSegment, "conv"));
    try std.testing.expectEqual(@as(usize, 56), @sizeOf(DsaRow));
    try std.testing.expectEqual(@as(usize, 32), @offsetOf(DsaRow, "position"));
    try std.testing.expectEqual(@as(usize, 48), @offsetOf(DsaRow, "flags"));
}

test "target staging refreshes positions pointers and recurrence parity while clearing inactive segments" {
    var f: Fixture = .{};
    try f.init();
    var h = try Host.init(std.testing.allocator, &f.w);
    defer h.deinit();
    const parts = [_]Segment{.{ .st = &f.st, .row0 = 0, .rows = 4 }};
    try h.build(&f.w, &parts, 4, false);
    try std.testing.expectEqual(Row{ .segment = 0, .local = 3 }, h.rows[3]);
    try std.testing.expectEqual(@as(u32, 0), h.kda[1].rows);
    try std.testing.expectEqual(f.st.rec, h.kda[0].state_in);
    try std.testing.expectEqual(f.st.rec + f.st.rec_layer, h.kda[0].state_out);
    try std.testing.expectEqual(@as(u32, 2051), h.dsa[3].position);
    try std.testing.expectEqual(resident, h.dsa[2].flags);
    try std.testing.expectEqual(resident | sparse, h.dsa[3].flags);
    f.cur[0] = 1;
    f.st.pos = 13;
    f.st.rec += 0x100000;
    f.caches[0].lc.ptr = 0x200000;
    try h.build(&f.w, &parts, 4, false);
    try std.testing.expectEqual(f.st.rec + f.st.rec_layer, h.kda[0].state_in);
    try std.testing.expectEqual(f.st.rec, h.kda[0].state_out);
    try std.testing.expectEqual(@as(u64, 0x200000), h.dsa[0].latent);
    try std.testing.expectEqual(@as(u32, 13), h.dsa[0].position);
}

test "target DSA merge chunks preserve segment extent across dense chunk and sparse boundaries" {
    var f: Fixture = .{};
    try f.init();
    var h = try Host.init(std.testing.allocator, &f.w);
    defer h.deinit();
    const parts = [_]Segment{.{ .st = &f.st, .row0 = 0, .rows = 16 }};
    f.st.pos = 497;
    try h.build(&f.w, &parts, 16, false);
    for (h.dsa[0..16]) |row| try std.testing.expectEqual(@as(u32, 2), row.pad);
    try std.testing.expectEqual(@as(u32, 497), h.dsa[0].position);
    f.st.pos = 2051 - 16;
    try h.build(&f.w, &parts, 16, false);
    for (h.dsa[0..16]) |row| {
        try std.testing.expectEqual(@as(u32, 5), row.pad);
        try std.testing.expectEqual(@as(u32, 0), row.flags & sparse);
    }
    f.st.pos += 1;
    try h.build(&f.w, &parts, 16, false);
    for (h.dsa[0..16]) |row| try std.testing.expectEqual(@as(u32, 5), row.pad);
    try std.testing.expectEqual(@as(u32, 0), h.dsa[14].flags & sparse);
    try std.testing.expectEqual(sparse, h.dsa[15].flags & sparse);
}

test "target staging handles nonresident sharded pointers and refuses invalid geometry before publication" {
    var f: Fixture = .{};
    try f.init();
    var h = try Host.init(std.testing.allocator, &f.w);
    defer h.deinit();
    var parts = [_]Segment{.{ .st = &f.st, .row0 = 0, .rows = 4 }};
    f.caches[0].owner = 1;
    f.caches[0].lc.ptr = 0;
    try h.build(&f.w, &parts, 4, true);
    try std.testing.expectEqual(@as(u32, 1), h.dsa[0].owner);
    try std.testing.expectEqual(@as(u32, 0), h.dsa[0].flags & resident);
    f.w.share.rank = 1;
    try std.testing.expectError(error.BadTargetGraphPlacement, h.build(&f.w, &parts, 4, true));
    f.w.share.rank = 0;
    try std.testing.expectError(error.BadTargetGraphOwner, h.build(&f.w, &parts, 4, false));
    parts[0].row0 = 1;
    try std.testing.expectError(error.BadTargetGraphWindow, h.build(&f.w, &parts, 4, true));
    parts[0].row0 = 0;
    parts[0].rows = 17;
    try std.testing.expectError(error.BadTargetGraphWindow, h.build(&f.w, &parts, 17, true));
    parts[0].rows = 4;
    f.st.pos = f.st.capacity - 3;
    try std.testing.expectError(error.PastCapacity, h.build(&f.w, &parts, 4, true));
    try std.testing.expectError(error.BadTargetGraphWindow, h.build(&f.w, &parts, 33, true));
}
