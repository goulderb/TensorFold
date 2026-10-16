//! Live-process, full-rank BF16 vision row binding. The pipeline owns Table, its buffer and host slices.
//! No encoder, allocator, callback or language embedding substitution is hidden in this module.
const std = @import("std");
const cuda = @import("cuda");
const max_context: usize = 256 * 1024;
const max_rows: usize = @import("cuda_state.zig").prefill_rows;

/// The sequence allocation includes an internal decode-window pad beyond the
/// model's logical context; image positions must remain inside the latter.
pub fn logicalCapacity(allocated: usize) usize {
    return @min(allocated, max_context);
}

fn product(xs: []const usize) !usize {
    var n: usize = 1;
    for (xs) |x| n = try std.math.mul(usize, n, x);
    return n;
}

fn checkTokens(ids: []const i32, vocab: u32) !void {
    for (ids) |id| if (id < 0 or @as(u32, @intCast(id)) >= vocab) return error.BadToken;
}

pub const width: usize = 4096;
pub const streams: usize = 4;
pub const DType = enum { bf16, f32 };

pub fn traceEnabled() bool {
    const value = std.c.getenv("TF_GLM_VISION_TRACE") orelse return false;
    return std.mem.eql(u8, std.mem.span(value), "1");
}

fn deviceSha256(buffer: cuda.DeviceBuffer, offset: usize, len: usize) ![32]u8 {
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    var bytes: [64 * 1024]u8 = undefined;
    var at: usize = 0;
    while (at < len) {
        const n = @min(bytes.len, len - at);
        try buffer.download(offset + at, bytes[0..n]);
        hash.update(bytes[0..n]);
        at += n;
    }
    var digest: [32]u8 = undefined;
    hash.final(&digest);
    return digest;
}

pub fn traceFeatureTable(table: *const Table) !void {
    if (!traceEnabled()) return;
    try table.stream.synchronize();
    const digest = try deviceSha256(table.features, 0, table.features.len);
    const hex = std.fmt.bytesToHex(digest, .lower);
    std.debug.print("[DEBUG-glm-vision-window] phase=features rank={d} replica={d} rows={d} bytes={d} ptr={x} first_position={d} last_position={d} sha256={s}\n", .{
        table.rank, table.replica, table.rows, table.features.len, table.features.ptr, table.positions[0], table.positions[table.positions.len - 1], &hex,
    });
}

/// Stable address required. Owner must clear ready BEFORE reuse/cleanup and never mutate while borrowed.
/// ready certifies native encoder/copy completion. Every TP rank owns a complete, identical feature table;
/// replica identifies the same request, not a device-content checksum or physical-TP agreement protocol.
pub const Table = struct {
    driver: *const cuda.Driver,
    context: cuda.abi.Context,
    stream: cuda.Stream,
    features: cuda.DeviceBuffer,
    positions: []const u32,
    ids: []const u32,
    image_token: u32,
    width: u32 = 4096,
    rows: usize,
    dtype: DType = .bf16,
    ready: bool = false,
    generation: u64 = 0,
    replica: u64,
    world: u32 = 1,
    rank: u32 = 0,

    pub fn binding(self: *const Table) !Binding {
        try validateTable(self);
        return .{ .owner = self, .generation = self.generation };
    }
};

/// Borrows the SAME live Table, never a copied readiness flag. All owners/slices must remain alive and immutable
/// until main computation completes; on failed synchronization they must be retained until cleanup succeeds.
pub const Binding = struct { owner: *const Table, generation: u64 };

pub const Target = struct {
    stream: cuda.Stream,
    context: cuda.abi.Context,
    embedding: cuda.DeviceBuffer,
    start: usize,
    capacity: usize,
    window: usize,
    vocab: usize,
    width: usize = 4096,
    streams: usize = 4,
    world: u32 = 1,
    rank: u32 = 0,
};
pub const Row = struct { window: usize, table: usize };
pub const Plan = struct {
    rows: [max_rows]Row = undefined,
    count: usize = 0,
    source: u64,
    output: u64,
    row_bytes: usize,
    streams: usize,
};

fn validateTable(o: *const Table) !void {
    if (!o.ready) return error.VisionNotReady;
    if (o.generation == 0 or o.replica == 0) return error.StaleVisionBinding;
    if (o.dtype != .bf16 or o.width != width or o.rows == 0 or o.rows > max_context or o.positions.len != o.rows) return error.BadVisionShape;
    if (o.ids.len == 0 or o.ids.len > max_context or o.rows > o.ids.len) return error.BadVisionShape;
    if ((o.world != 1 and o.world != 2) or o.rank >= o.world) return error.VisionRankMismatch;
    if (o.context == null) return error.VisionContextMismatch;
    if (o.features.borrowed) return error.VisionOwnership;
    if (o.features.d != o.driver or o.stream.d != o.driver) return error.DriverMismatch;
    const bytes = try product(&.{ o.rows, width, 2 });
    if (o.features.len != bytes) return error.BadVisionShape;
    _ = try span(o.features, 2);
    if (o.image_token >= 154880) return error.BadToken;
    var count: usize = 0;
    for (o.ids, 0..) |id, pos| {
        if (id >= 154880) return error.BadToken;
        if (id == o.image_token) {
            if (count >= o.positions.len or o.positions[count] != pos) return error.BadVisionPositions;
            count += 1;
        }
    }
    if (count != o.rows) return error.BadVisionPositions;
}

/// All token/position/shape/ownership checks are CPU-only and precede any query, copy or decoder write.
/// The supplied IDs must be the original prompt window, not synthetic row-index IDs.
pub fn plan(b: Binding, target: Target, ids: []const i32) !Plan {
    const o = b.owner;
    try validateTable(o);
    if (b.generation != o.generation) return error.StaleVisionBinding;
    if (target.capacity == 0 or target.capacity > max_context) return error.BadCapacity;
    if (target.window == 0 or target.window > max_rows or ids.len == 0 or ids.len > target.window or ids.len > target.capacity) return error.BadRows;
    if (target.start > target.capacity or ids.len > target.capacity - target.start or o.ids.len > target.capacity) return error.ContextFull;
    if (target.width != width or target.streams != streams) return error.BadVisionShape;
    if (target.vocab == 0 or target.vocab > 154880) return error.BadShape;
    if (o.world != target.world or o.rank != target.rank) return error.VisionRankMismatch;
    if (target.stream.d != o.driver or target.embedding.d != o.driver) return error.DriverMismatch;
    if (target.stream.handle != o.stream.handle) return error.VisionStreamMismatch;
    if (target.context == null or target.context != o.context) return error.VisionContextMismatch;
    if (target.start > o.ids.len or ids.len > o.ids.len - target.start) return error.VisionWindowMismatch;
    for (o.ids) |id| if (id >= target.vocab) return error.BadToken;
    try checkTokens(ids, @intCast(target.vocab));
    for (ids, o.ids[target.start..][0..ids.len]) |id, original| if (@as(u32, @intCast(id)) != original) return error.VisionTokenMismatch;
    const row_bytes = try product(&.{ width, 2 });
    if (target.embedding.len != try product(&.{ target.window, streams, row_bytes })) return error.BadVisionShape;
    try disjoint(o.features, target.embedding);
    var p: Plan = .{ .source = o.features.ptr, .output = target.embedding.ptr, .row_bytes = row_bytes, .streams = streams };
    const end = target.start + ids.len;
    for (o.positions, 0..) |pos, row| {
        if (pos < target.start) continue;
        if (pos >= end) break;
        p.rows[p.count] = .{ .window = pos - target.start, .table = row };
        p.count += 1;
    }
    return p;
}

/// Host metadata agreement only: full replicated shape, exact prompt and row order. Identical GPU content and
/// remote failure/acceptance agreement remain the TP caller's qualification obligation.
pub fn sameReplica(a: Binding, b: Binding) !void {
    try validateTable(a.owner);
    try validateTable(b.owner);
    const x = a.owner;
    const y = b.owner;
    if (a.generation != x.generation or b.generation != y.generation) return error.StaleVisionBinding;
    if (x.replica != y.replica or x.world != y.world or x.rows != y.rows or x.image_token != y.image_token or
        !std.mem.eql(u32, x.ids, y.ids) or !std.mem.eql(u32, x.positions, y.positions)) return error.VisionReplicaMismatch;
    // In a local single-context TP harness, two owning descriptors must not double-own one allocation.
    if (x.context == y.context and x.driver == y.driver) try disjoint(x.features, y.features);
}

/// Native context admission; no writes. Main group admission runs this for EVERY local rank before staging IDs.
pub fn checkContext(stream: cuda.Stream, expected: cuda.abi.Context) !void {
    if (expected == null) return error.VisionContextMismatch;
    var current: cuda.abi.Context = null;
    try stream.d.check(stream.d.api.cuCtxGetCurrent(&current), "cuCtxGetCurrent");
    if (current != expected) return error.VisionContextMismatch;
}

/// Exact archived index_copy_(rows, features.repeat(1,4)), without arithmetic or a new kernel. Copies are ordered
/// on the table/decoder's common stream. This queues work; caller must synchronize before freeing any owner.
pub fn inject(b: Binding, target: Target, ids: []const i32) !void {
    const p = try plan(b, target, ids);
    try checkContext(target.stream, target.context);
    for (p.rows[0..p.count]) |row| {
        const source = p.source + row.table * p.row_bytes;
        for (0..p.streams) |s| {
            const output = p.output + (row.window * p.streams + s) * p.row_bytes;
            try target.stream.d.check(target.stream.d.api.cuMemcpyDtoDAsync_v2(output, source, p.row_bytes, target.stream.handle), "cuMemcpyDtoDAsync");
        }
    }
    if (traceEnabled()) {
        try target.stream.synchronize();
        std.debug.print("[DEBUG-glm-vision-window] phase=inject rank={d} replica={d} start={d} window={d} matched={d} feature_ptr={x} embedding_ptr={x}\n", .{
            target.rank, b.owner.replica, target.start, ids.len, p.count, p.source, p.output,
        });
        if (p.count != 0) {
            const selected = [_]Row{ p.rows[0], p.rows[p.count - 1] };
            for (selected, 0..) |row, selected_index| {
                for (0..p.streams) |shard| {
                    const at = (row.window * p.streams + shard) * p.row_bytes;
                    const digest = try deviceSha256(target.embedding, at, p.row_bytes);
                    const hex = std.fmt.bytesToHex(digest, .lower);
                    std.debug.print("[DEBUG-glm-vision-window] phase=embedding rank={d} replica={d} selected={d} prompt_position={d} feature_row={d} shard={d} sha256={s}\n", .{
                        target.rank, b.owner.replica, selected_index, target.start + row.window, row.table, shard, &hex,
                    });
                }
            }
        }
    }
}

pub fn disjoint(a: cuda.DeviceBuffer, b: cuda.DeviceBuffer) !void {
    if (a.d != b.d) return error.DriverMismatch;
    const ae = try span(a, 2);
    const be = try span(b, 2);
    if (a.ptr < be and b.ptr < ae) return error.VisionAliasedBuffers;
}
fn span(b: cuda.DeviceBuffer, alignment: u64) !u64 {
    if (b.ptr == 0 or b.ptr % alignment != 0 or b.len == 0) return error.BadVisionShape;
    return std.math.add(u64, b.ptr, b.len);
}
