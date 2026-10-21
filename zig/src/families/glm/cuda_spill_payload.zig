const std = @import("std");
const layout = @import("cuda_layout.zig");
const cache = @import("cuda_cache.zig");

const magic = "GLMCU001";
pub const header_bytes = 88;
pub const max_parities = 1024;
pub const max_meta = header_bytes + max_parities;
const no_owner = std.math.maxInt(u32);

pub fn withDraftBytes(target_bytes: u64, draft_bytes: u64) !u64 {
    if (draft_bytes == 0) return target_bytes;
    const padded = (try std.math.add(u64, target_bytes, 255)) & ~@as(u64, 255);
    return std.math.add(u64, padded, draft_bytes);
}

pub const Shape = struct {
    layout: layout.Layout,
    format: cache.Format,
    mtp: bool,
    sharded: bool,
    rank: u32,
    world: u32,
    grid: u32,
    parities: usize,
    draft_bytes: u64 = 0,
    draft_identity: [32]u8 = @splat(0),

    pub fn digest(s: Shape) [32]u8 {
        var h = std.crypto.hash.sha2.Sha256.init(.{});
        // Payload layout epoch, separate from the unchanged metadata codec version.
        // The ninth fixed field owns the prompt hidden row; old eight-field payloads cannot restore.
        h.update("glm-cuda-snapshot-layout-v2-prompt-hidden");
        inline for (.{ "capacity", "ring_rows", "dsa_layers", "fixed_end", "cache_stride", "latent_stride", "index_stride", "pooled_stride", "mtp_pos", "total_bytes" }) |field| hashInt(&h, @field(s.layout, field));
        for (s.layout.fixed) |n| hashInt(&h, n);
        for (s.layout.cache_offsets) |n| hashInt(&h, n);
        hashInt(&h, @intFromBool(s.layout.external_planes));
        hashInt(&h, @backingInt(s.format));
        hashInt(&h, @intFromBool(s.mtp));
        hashInt(&h, @intFromBool(s.sharded));
        hashInt(&h, s.world);
        hashInt(&h, s.grid);
        hashInt(&h, s.parities);
        hashInt(&h, s.draft_bytes);
        h.update(&s.draft_identity);
        return h.finalResult();
    }

    pub fn payloadBytes(s: Shape, target_bytes: u64) !u64 {
        return withDraftBytes(target_bytes, s.draft_bytes);
    }

    fn check(s: Shape, pos: u64, mtp_len: u64, owner: ?u32) !usize {
        if (s.world == 0 or s.rank >= s.world or s.grid == 0 or s.parities > max_parities) return error.SpillShape;
        if (s.sharded and (!s.layout.external_planes or s.world != 2)) return error.SpillShape;
        if (pos == 0 or pos >= s.layout.capacity or pos % s.grid != 0) return error.SpillPosition;
        if (mtp_len != (if (s.mtp) pos else @as(u64, 0))) return error.SpillPosition;
        if (s.sharded != (owner != null)) return error.SpillOwner;
        if (owner) |r| if (r >= s.world) return error.SpillOwner;
        const plan = try s.layout.snapshot(.{ .main = @intCast(pos), .mtp = @intCast(mtp_len) });
        return s.payloadBytes(plan.bytesFor(owner == null or owner.? == s.rank));
    }
};

pub const State = struct {
    pos: u64,
    mtp_len: u64,
    owner: ?u32,
    bytes: u64,
    cur: []const u8,
};

pub fn encode(s: Shape, pos: u64, mtp_len: u64, owner: ?u32, cur: []const u1, out: []u8) ![]const u8 {
    const bytes = try s.check(pos, mtp_len, owner);
    if (cur.len != s.parities or out.len < header_bytes + cur.len) return error.SpillMetaLength;
    const meta = out[0 .. header_bytes + cur.len];
    @memset(meta, 0);
    @memcpy(meta[0..8], magic);
    put(u32, meta, 8, 1);
    put(u32, meta, 12, @intCast(meta.len));
    @memcpy(meta[16..48], &s.digest());
    put(u64, meta, 48, pos);
    put(u64, meta, 56, mtp_len);
    put(u64, meta, 64, bytes);
    put(u32, meta, 72, owner orelse no_owner);
    put(u32, meta, 76, s.rank);
    put(u32, meta, 80, @intCast(cur.len));
    for (cur, meta[header_bytes..]) |v, *p| p.* = v;
    return meta;
}

pub fn decode(s: Shape, meta: []const u8, at: u32, payload_bytes: u64) !State {
    if (meta.len < header_bytes or meta.len != header_bytes + s.parities) return error.SpillMetaLength;
    if (!std.mem.eql(u8, meta[0..8], magic) or get(u32, meta, 8) != 1) return error.SpillMetaVersion;
    if (get(u32, meta, 12) != meta.len or get(u32, meta, 80) != s.parities) return error.SpillMetaLength;
    if (!std.mem.eql(u8, meta[16..48], &s.digest()) or get(u32, meta, 76) != s.rank or get(u32, meta, 84) != 0) return error.SpillShape;
    const owner = get(u32, meta, 72);
    const state: State = .{ .pos = get(u64, meta, 48), .mtp_len = get(u64, meta, 56), .bytes = get(u64, meta, 64), .owner = if (owner == no_owner) null else owner, .cur = meta[header_bytes..] };
    if (state.pos != at) return error.SpillPosition;
    if (state.bytes != payload_bytes or state.bytes != try s.check(state.pos, state.mtp_len, state.owner)) return error.SpillPayloadLength;
    for (state.cur) |v| if (v > 1) return error.SpillParity;
    return state;
}

fn hashInt(h: *std.crypto.hash.sha2.Sha256, n: u64) void {
    var bytes: [8]u8 = undefined;
    std.mem.writeInt(u64, &bytes, n, .little);
    h.update(&bytes);
}

fn put(comptime T: type, out: []u8, at: usize, value: T) void {
    std.mem.writeInt(T, out[at..][0..@sizeOf(T)], value, .little);
}

fn get(comptime T: type, in: []const u8, at: usize) T {
    return std.mem.readInt(T, in[at..][0..@sizeOf(T)], .little);
}

test "owner and peer roundtrip BF16 and FP8 with separate main and MTP geometry" {
    for ([_]cache.Format{ .bf16, .fp8 }) |format| {
        for ([_]bool{ false, true }) |mtp| {
            const l = try (try layout.Layout.init(.{ 256, 256, 0, 0, 0, 0, 0, 0, 0 }, 8192, 2048, 2, format, 512, 128)).external();
            for (0..2) |rank| {
                const shape: Shape = .{ .layout = l, .format = format, .mtp = mtp, .sharded = true, .rank = @intCast(rank), .world = 2, .grid = 2048, .parities = 3 };
                var out: [max_meta]u8 = undefined;
                const meta = try encode(shape, 4096, if (mtp) 4096 else 0, 1, &.{ 0, 1, 0 }, &out);
                const bytes = (try l.snapshot(.{ .main = 4096, .mtp = if (mtp) 4096 else 0 })).bytesFor(rank == 1);
                const got = try decode(shape, meta, 4096, bytes);
                try std.testing.expectEqual(@as(u64, 4096), got.pos);
                try std.testing.expectEqual(@as(u64, if (mtp) 4096 else 0), got.mtp_len);
                try std.testing.expectEqual(@as(?u32, 1), got.owner);
                try std.testing.expectEqualSlices(u8, &.{ 0, 1, 0 }, got.cur);
                var other = shape;
                other.rank = 1 - shape.rank;
                try std.testing.expectError(error.SpillShape, decode(other, meta, 4096, bytes));
                try std.testing.expectError(error.SpillPayloadLength, decode(shape, meta, 4096, bytes - 1));
                try std.testing.expectError(error.SpillPosition, decode(shape, meta, 2048, bytes));
                out[header_bytes + 1] = 2;
                try std.testing.expectError(error.SpillParity, decode(shape, meta, 4096, bytes));
            }
        }
    }
}

test "codec refuses partial metadata, geometry changes and invalid owner before allocation" {
    const l = try layout.Layout.init(.{ 256, 0, 0, 0, 0, 0, 0, 0, 0 }, 8192, 2048, 1, .bf16, 512, 128);
    const shape: Shape = .{ .layout = l, .format = .bf16, .mtp = false, .sharded = false, .rank = 0, .world = 1, .grid = 2048, .parities = 1 };
    var out: [max_meta]u8 = undefined;
    const meta = try encode(shape, 2048, 0, null, &.{1}, &out);
    const bytes = (try l.snapshot(.{ .main = 2048, .mtp = 0 })).byteCount();
    const got = try decode(shape, meta, 2048, bytes);
    try std.testing.expectEqual(@as(?u32, null), got.owner);
    try std.testing.expectEqualSlices(u8, &.{1}, got.cur);
    for (0..meta.len) |n| try std.testing.expectError(error.SpillMetaLength, decode(shape, meta[0..n], 2048, bytes));
    var changed = shape;
    changed.format = .fp8;
    try std.testing.expectError(error.SpillShape, decode(changed, meta, 2048, bytes));
    try std.testing.expectError(error.SpillOwner, encode(shape, 2048, 0, 0, &.{1}, &out));
    try std.testing.expectError(error.SpillPosition, encode(shape, 2048, 2048, null, &.{1}, &out));
    try std.testing.expectError(error.SpillPosition, encode(shape, 8192, 0, null, &.{1}, &out));
}

test "persisted adapter rejects the historical eight-field layout and separates prompt hidden geometry" {
    const old_layout = try layout.Layout.init(.{ 256, 0, 0, 0, 0, 0, 0, 0, 0 }, 8192, 2048, 1, .bf16, 512, 128);
    const new_layout = try layout.Layout.init(.{ 256, 0, 0, 0, 0, 0, 0, 0, 4096 * 2 }, 8192, 2048, 1, .bf16, 512, 128);
    // Pinned pre-integration layout/codec: 7d547958d5, cuda_layout + cuda_spill_payload.
    // Geometry is the old_layout above with only eight fixed fields; world/grid/parities = 1/2048/1.
    const historical_digest = [_]u8{ 0xd2, 0xb7, 0x9f, 0x0d, 0x78, 0x9b, 0xfa, 0x08, 0x45, 0x5d, 0xb1, 0x7c, 0x75, 0x4b, 0x65, 0x8e, 0x11, 0xf3, 0xab, 0x17, 0xd2, 0xc0, 0xd5, 0x55, 0x75, 0x70, 0xc2, 0x39, 0xe9, 0x08, 0xd4, 0x96 };
    const old_shape: Shape = .{ .layout = old_layout, .format = .bf16, .mtp = false, .sharded = false, .rank = 0, .world = 1, .grid = 2048, .parities = 1 };
    var new_shape = old_shape;
    new_shape.layout = new_layout;
    try std.testing.expect(!std.mem.eql(u8, &old_shape.digest(), &new_shape.digest()));
    try std.testing.expect(!std.mem.eql(u8, &historical_digest, &new_shape.digest()));
    const old_bytes = (try old_layout.snapshot(.{ .main = 2048, .mtp = 0 })).byteCount();
    const new_bytes = (try new_layout.snapshot(.{ .main = 2048, .mtp = 0 })).byteCount();
    try std.testing.expectEqual(@as(usize, 8192), new_bytes - old_bytes);
    var out: [max_meta]u8 = undefined;
    const meta = try encode(old_shape, 2048, 0, null, &.{0}, &out);
    // Current metadata codec, historical payload identity: rejected at decode before any restore occurs.
    @memcpy(out[16..48], &historical_digest);
    try std.testing.expectError(error.SpillShape, decode(new_shape, meta, 2048, old_bytes));
    // The new epoch also rejects old persisted data for zero-sized synthetic hidden-row layouts.
    try std.testing.expectError(error.SpillShape, decode(old_shape, meta, 2048, old_bytes));
    const current = try encode(new_shape, 2048, 0, null, &.{0}, &out);
    _ = try decode(new_shape, current, 2048, new_bytes);
    try std.testing.expectError(error.SpillShape, decode(old_shape, current, 2048, new_bytes));
}

test "DFlash ring tail is budgeted and incompatible drafter state is rejected" {
    const l = try layout.Layout.init(.{ 256, 0, 0, 0, 0, 0, 0, 0, 0 }, 8192, 2048, 1, .bf16, 512, 128);
    const control: Shape = .{ .layout = l, .format = .bf16, .mtp = false, .sharded = false, .rank = 0, .world = 2, .grid = 2048, .parities = 1 };
    var shape = control;
    shape.draft_bytes = 22_282_240;
    shape.draft_identity = @splat(7);
    const target = (try l.snapshot(.{ .main = 2048, .mtp = 0 })).byteCount();
    const total = std.mem.alignForward(u64, target, 256) + 22_282_240;
    try std.testing.expectEqual(total, try shape.payloadBytes(target));
    var out: [max_meta]u8 = undefined;
    const meta = try encode(shape, 2048, 0, null, &.{0}, &out);
    const saved = try decode(shape, meta, 2048, total);
    try std.testing.expectEqual(total, saved.bytes);
    try std.testing.expectError(error.SpillShape, decode(control, meta, 2048, total));
    var other = shape;
    other.draft_identity[0] = 8;
    try std.testing.expectError(error.SpillShape, decode(other, meta, 2048, total));
    try std.testing.expectError(error.SpillPayloadLength, decode(shape, meta, 2048, target));
}

test "canonical saved-state restore refuses content and precision mismatch" {
    const ranks = @import("cuda_ranks.zig");
    const l = try layout.Layout.init(.{ 256, 0, 0, 0, 0, 0, 0, 0, 0 }, 8192, 2048, 1, .bf16, 512, 128);
    const q4 = ranks.CanonicalDraft{ .checkpoint = ranks.hashBytes("saved-q4"), .bits = 4 };
    const other = ranks.CanonicalDraft{ .checkpoint = ranks.hashBytes("saved-other"), .bits = 4 };
    const bf16 = ranks.CanonicalDraft{ .checkpoint = ranks.hashBytes("saved-q4"), .bits = 16 };
    const ring = @import("dflash.zig").ring_bytes;
    const owned = ranks.persistedDraftSnapshot(true, q4, ring, 0, @splat(0));
    var shape: Shape = .{ .layout = l, .format = .bf16, .mtp = false, .sharded = false, .rank = 0, .world = 1, .grid = 2048, .parities = 1, .draft_bytes = owned.bytes, .draft_identity = owned.identity };
    const target = (try l.snapshot(.{ .main = 2048, .mtp = 0 })).byteCount();
    const total = try shape.payloadBytes(target);
    var out: [max_meta]u8 = undefined;
    const meta = try encode(shape, 2048, 0, null, &.{0}, &out);
    _ = try decode(shape, meta, 2048, total);
    var content = shape;
    content.draft_identity = other.snapshotIdentity();
    try std.testing.expectError(error.SpillShape, decode(content, meta, 2048, total));
    var bits = shape;
    bits.draft_identity = bf16.snapshotIdentity();
    try std.testing.expectError(error.SpillShape, decode(bits, meta, 2048, total));
    var legacy = shape;
    legacy.draft_identity = @splat(7);
    try std.testing.expectError(error.SpillShape, decode(legacy, meta, 2048, total));
}
