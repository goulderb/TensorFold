const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm_cuda");
const qr = @import("quant_ref_tests.zig");
const check = @import("check.zig");

/// Compare bounded raw rings with full-context storage through wraps, partial pools and a resumed branch.
pub fn run(r: *qr.RigT, set: glm.glue.Set, format: glm.cache.Format) !void {
    const capacity = 8192;
    const ring_rows = 2112;
    const width = 128;
    const stride = format.rowBytes(width);
    const key = try r.bfs(capacity * width, 1.5);
    const gate = try r.a.alloc(f32, capacity * width);
    for (gate) |*v| v.* = (r.rng.float(f32) * 2 - 1) * 3;
    const lnw = try r.bfs(width, 1);
    const lnb = try r.bfs(width, 0.25);
    const ape = try r.bfs(4 * width, 0.5);
    const kr = try r.dev(std.mem.sliceAsBytes(key));
    const gr = try r.dev(std.mem.sliceAsBytes(gate));
    const w = try r.dev(std.mem.sliceAsBytes(lnw));
    const b = try r.dev(std.mem.sliceAsBytes(lnb));
    const a = try r.dev(std.mem.sliceAsBytes(ape));
    const full_k = try r.zeros(capacity * width * 2);
    const full_g = try r.zeros(capacity * width * 2);
    const raw_k = try r.zeros((ring_rows + 2) * width * 2);
    const raw_g = try r.zeros((ring_rows + 2) * width * 2);
    const index: glm.cache.Index = .{ .keys = raw_k + width * 2, .gates = raw_g + width * 2, .rows = ring_rows };
    const pooled_bytes = (capacity / 4 + 2) * stride;
    const full_pk = try r.zeros(pooled_bytes);
    const ring_pk = try r.zeros(pooled_bytes);
    var at: usize = 0;
    for ([_]usize{ 2048, 61, 1, 1, 1, 1, 7, 2048, 3, 2048, 1, 2, 3 }) |rows| {
        try step(r, set, format, kr, gr, w, b, a, full_k, full_g, full_pk, index, ring_pk, at, rows);
        at += rows;
    }
    try step(r, set, format, kr, gr, w, b, a, full_k, full_g, full_pk, index, ring_pk, at, 8);
    at += 2; // only the first two speculative rows were accepted
    // Preserve the incomplete pool, poison the rings as another sequence would, then restore and branch.
    const tail_bytes = at % 4 * width * 2;
    const tail_at = (at - at % 4) % ring_rows * width * 2;
    const saved_k = try r.back(index.keys + tail_at, tail_bytes);
    const saved_g = try r.back(index.gates + tail_at, tail_bytes);
    const zeros = try r.a.alloc(u8, ring_rows * width * 2);
    @memset(zeros, 0);
    const kd: cuda.DeviceBuffer = .{ .d = r.gpu.d, .ptr = index.keys, .len = zeros.len };
    const gd: cuda.DeviceBuffer = .{ .d = r.gpu.d, .ptr = index.gates, .len = zeros.len };
    try kd.uploadAsync(0, zeros, r.s.handle);
    try gd.uploadAsync(0, zeros, r.s.handle);
    try kd.uploadAsync(tail_at, saved_k, r.s.handle);
    try gd.uploadAsync(tail_at, saved_g, r.s.handle);
    try step(r, set, format, kr, gr, w, b, a, full_k, full_g, full_pk, index, ring_pk, at, 8);
    for ([_]u64{ raw_k, raw_g }) |ptr| {
        const bytes = try r.back(ptr, (ring_rows + 2) * width * 2);
        for (bytes[0 .. width * 2]) |v| try check.expect(v == 0xff, "raw ring leading guard", .{});
        for (bytes[(ring_rows + 1) * width * 2 ..]) |v| try check.expect(v == 0xff, "raw ring trailing guard", .{});
    }
    check.pass("{s} raw index rings match full storage through wraps and restored tails", .{@tagName(format)});
}

fn step(r: *qr.RigT, set: glm.glue.Set, format: glm.cache.Format, kr: u64, gr: u64, w: u64, b: u64, ape: u64, full_k: u64, full_g: u64, full_pk: u64, index: glm.cache.Index, ring_pk: u64, at: usize, rows: usize) !void {
    const p: i32 = @intCast(at);
    const pos = try r.dev(std.mem.asBytes(&p));
    const keys = kr + at * 128 * 2;
    const gates = gr + at * 128 * 4;
    var write: cuda.Args = .{};
    inline for (.{ keys, @as(i64, 128), gates, w, b, full_k, full_g, pos, @as(f32, 1e-6) }) |v| write.add(v);
    try cuda.launch.launch(set.glue.index_write, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 32 } }, r.s, &write);
    var pool: cuda.Args = .{};
    inline for (.{ full_k, full_g, ape, full_pk, pos, @as(i64, @intCast(rows)) }) |v| pool.add(v);
    try cuda.launch.launch(if (format == .fp8) set.glue.pool_keys_fp8 else set.glue.pool_keys, .{ .grid = .{ .x = @intCast(rows / 4 + 2) }, .block = .{ .x = 32 } }, r.s, &pool);
    try glm.sparse.indexUpdate(set, r.s, keys, 128, gates, w, b, ape, index, .{ .ptr = ring_pk, .format = format }, pos, rows);
    const count = (at + rows) / 4 * format.rowBytes(128);
    try check.sameBytes("ring pooled prefix equals full cache", try r.back(ring_pk, count), try r.back(full_pk, count));
    const ring_keys = try r.back(index.keys, index.rows * 128 * 2);
    const ring_gates = try r.back(index.gates, index.rows * 128 * 2);
    const full_keys = try r.back(full_k + at * 128 * 2, rows * 128 * 2);
    const full_gates = try r.back(full_g + at * 128 * 2, rows * 128 * 2);
    for (at..at + rows) |row| {
        const slot = row % index.rows * 128 * 2;
        const src = (row - at) * 128 * 2;
        try check.sameBytes("ring normalized keys equal full cache", ring_keys[slot..][0 .. 128 * 2], full_keys[src..][0 .. 128 * 2]);
        try check.sameBytes("ring gates equal full cache", ring_gates[slot..][0 .. 128 * 2], full_gates[src..][0 .. 128 * 2]);
    }
}
