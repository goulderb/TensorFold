//! Six-layer prompt scheduling proof on the same engine, mirrored directly on both ranks.
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm_cuda");
const check = @import("check.zig");
const Engine = glm.engine.Engine;
const guard_bytes = 32;
const guard_value = 0xa5;
const tap_layers = [_]usize{ 5, 14, 24, 33, 42 };

const Guarded = struct {
    allocation: cuda.DeviceBuffer,
    bytes: usize,

    fn init(gpu: check.Gpu, bytes: usize) !Guarded {
        return .{ .allocation = try cuda.DeviceBuffer.alloc(gpu.d, bytes + 2 * guard_bytes), .bytes = bytes };
    }

    fn ptr(g: Guarded) u64 {
        return g.allocation.ptr + guard_bytes;
    }

    fn reset(g: Guarded, e: *Engine) !void {
        try e.d.check(e.d.api.cuMemsetD8Async(g.allocation.ptr, guard_value, g.allocation.len, e.stream.handle), "overlap guard reset");
    }

    fn verify(g: Guarded, e: *Engine, rows: usize) !void {
        var edge: [guard_bytes]u8 = undefined;
        for ([_]u64{ g.allocation.ptr, g.ptr() + g.bytes }) |address| {
            try e.d.check(e.d.api.cuMemcpyDtoH_v2(&edge, address, edge.len), "overlap guard read");
            for (edge) |v| try check.expect(v == guard_value, "rank {d}: prompt output allocation guard intact", .{e.share.rank});
        }
        const padded = (rows + 1) / 2 * 2;
        const offset = padded * e.c.hidden * 2;
        if (offset < g.bytes) {
            const tail = try e.gpa.alloc(u8, g.bytes - offset);
            defer e.gpa.free(tail);
            try e.d.check(e.d.api.cuMemcpyDtoH_v2(tail.ptr, g.ptr() + offset, tail.len), "overlap tail guard read");
            for (tail) |v| try check.expect(v == guard_value, "rank {d}: rows past padded prompt remain untouched", .{e.share.rank});
        }
    }
};

const Witness = struct {
    bytes: std.ArrayList(u8) = .empty,

    fn append(w: *Witness, e: *Engine, bytes: []const u8) !void {
        try w.bytes.appendSlice(e.gpa, bytes);
    }

    fn device(w: *Witness, e: *Engine, pointer: u64, count: usize) !void {
        const at = w.bytes.items.len;
        try w.bytes.resize(e.gpa, at + count);
        if (count > 0) try e.d.check(e.d.api.cuMemcpyDtoH_v2(w.bytes.items[at..].ptr, pointer, count), "overlap witness read");
    }

    fn bf16(w: *Witness, e: *Engine, pointer: u64, count: usize) !void {
        const at = w.bytes.items.len;
        try w.device(e, pointer, count);
        const bytes = w.bytes.items[at..];
        var i: usize = 0;
        while (i < bytes.len) : (i += 2) {
            const bits = std.mem.readInt(u16, bytes[i..][0..2], .little);
            try check.expect(bits & 0x7f80 != 0x7f80, "rank {d}: finite prompt/decode witness element {d}", .{ e.share.rank, i / 2 });
        }
    }

    fn taps(w: *Witness, e: *Engine, b: *glm.forward.Buffers, rows: usize) !void {
        if (b.taps) |storage| for (tap_layers, 0..) |layer, plane| {
            if (layer >= e.w.layers.len) continue;
            try w.bf16(e, storage.ptr + plane * b.rows * e.c.hidden * 2, rows * e.c.hidden * 2);
        };
    }

    fn state(w: *Witness, e: *Engine) !void {
        const st = &e.cur.st;
        try check.expect(st.pending == 0, "rank {d}: prompt/keep has no pending rows", .{e.share.rank});
        const meta = [_]u64{ st.pos, st.pending, st.mtp_len, st.kda_layers, st.caches.len };
        try w.append(e, std.mem.sliceAsBytes(&meta));
        for (st.cur) |parity| try w.bytes.append(e.gpa, @intCast(parity));
        try w.device(e, st.pos_dev, 4);
        try w.device(e, st.conv, st.kda_layers * 3 * st.conv_c * 2);
        for (st.cur, 0..) |parity, layer| try w.device(e, st.recAt(parity, layer), st.rec_layer);
        const owner = if (st.layout.external_planes) st.mtp_caches.owner else null;
        const resident = owner == null or owner.? == e.share.rank;
        const plan = try st.layout.snapshot(.{ .main = st.pos, .mtp = st.mtp_len });
        var ranges = plan.rangesFor(resident);
        while (ranges.next()) |range| {
            if (range.plane == .fixed) continue;
            try w.device(e, st.address(range), range.bytes);
        }
    }

    fn prompt(w: *Witness, e: *Engine, rows: usize, first: u32) !void {
        const token = [_]u32{first};
        try w.append(e, std.mem.sliceAsBytes(&token));
        try w.bf16(e, e.bp.hidden, rows * e.c.hidden * 2);
        try w.bf16(e, e.bp.fnormed, rows * e.c.hidden * 2);
        try w.bf16(e, e.bp.logits, e.share.vocab.len() * 2);
        try w.taps(e, &e.bp, rows);
        try w.state(e);
    }

    fn decode(w: *Witness, e: *Engine) !void {
        var out: [3]u32 = undefined;
        try e.verify(&.{ 177, 196, 215 }, &out);
        try e.stream.synchronize();
        try w.append(e, std.mem.sliceAsBytes(&out));
        try w.bf16(e, e.bd.fnormed, 3 * e.c.hidden * 2);
        try w.bf16(e, e.bd.logits, 3 * e.share.vocab.len() * 2);
        try w.taps(e, &e.bd, 3);
        try e.keep(1);
        try e.stream.synchronize();
        try w.state(e);
        var following: [1]u32 = undefined;
        try e.verify(&.{234}, &following);
        try e.stream.synchronize();
        try w.append(e, std.mem.sliceAsBytes(&following));
        try w.bf16(e, e.bd.fnormed, e.c.hidden * 2);
        try w.bf16(e, e.bd.logits, e.share.vocab.len() * 2);
        try w.taps(e, &e.bd, 1);
        try e.keep(1);
        try e.stream.synchronize();
        try w.state(e);
    }
};

fn prepare(e: *Engine, hidden: Guarded, normed: Guarded, level: u8) !void {
    try e.stream.synchronize();
    e.settings.prefill_overlap = level;
    try e.bind(0);
    try e.begin(null);
    try hidden.reset(e);
    try normed.reset(e);
    if (e.bp.taps) |storage| try e.d.check(e.d.api.cuMemsetD8Async(storage.ptr, 0, storage.len, e.stream.handle), "overlap prompt tap reset");
    if (e.bd.taps) |storage| try e.d.check(e.d.api.cuMemsetD8Async(storage.ptr, 0, storage.len, e.stream.handle), "overlap decode tap reset");
    try e.stream.synchronize();
    try check.expect(try e.pairedReady(true), "both ranks ready for overlap fixture", .{});
}

pub fn run(gpu: check.Gpu, checkpoint: []const u8, rank: c_int, world: c_int, id_path: []const u8) !void {
    try check.expect(world == 2, "prompt overlap proof requires TP2", .{});
    var lib = try cuda.nccl.Library.open();
    defer lib.close();
    const comm = try @import("glm_lanes_tests.zig").openComm(gpu, &lib, rank, world, id_path);
    defer _ = lib.api.ncclCommDestroy(comm);
    var problem: []const u8 = "";
    var settings = try glm.ranks.Settings.read(&glm.ranks.environment, &problem);
    settings.layers = 6;
    settings.draft = .off;
    settings.target_graphs = .off;
    settings.prefill_overlap = 2;
    settings.prefill_pieces = 2;
    if (settings.kv_pool_tokens > 0) settings.kv_pool_tokens = 8192;
    try check.expect(settings.kda_prompt == .wy32, "prompt overlap proof requires TF_GLM_KDA_CHUNKED=1", .{});
    const e = try Engine.init(gpu.gpa, gpu.io, gpu.d, gpu.ctx, checkpoint, &lib, comm, @intCast(rank), @intCast(world), .{ .capacity = 4096, .mtp = false, .settings = settings });
    defer e.deinit();
    try check.expect(e.w.layers.len == 6, "prompt overlap proof loaded exactly six layers", .{});
    try e.bp.enableTaps(gpu.d);
    try e.bd.enableTaps(gpu.d);
    var hidden = try Guarded.init(gpu, e.bp.rows * e.c.hidden * 2);
    defer hidden.allocation.free();
    var normed = try Guarded.init(gpu, e.bp.rows * e.c.hidden * 2);
    defer normed.allocation.free();
    const old_hidden = e.bp.hidden;
    const old_normed = e.bp.fnormed;
    e.bp.hidden = hidden.ptr();
    e.bp.fnormed = normed.ptr();
    defer {
        e.stream.synchronize() catch |err| {
            e.pair_failure(err);
            std.process.exit(1);
        };
        e.pieces.side.synchronize() catch |err| {
            e.pair_failure(err);
            std.process.exit(1);
        };
        e.bp.hidden = old_hidden;
        e.bp.fnormed = old_normed;
    }
    const ids = try gpu.gpa.alloc(u32, 2048);
    defer gpu.gpa.free(ids);
    for (ids, 0..) |*id, i| id.* = @intCast(10 + (i * 17 + 37) % 7000);
    var reference: Witness = .{};
    defer reference.bytes.deinit(gpu.gpa);
    var candidate: Witness = .{};
    defer candidate.bytes.deinit(gpu.gpa);
    var cases: usize = 0;
    for ([_]usize{ 64, 65, 127, 128, 129, 2047, 2048 }) |rows| {
        for (0..3) |level| {
            try prepare(e, hidden, normed, @intCast(level));
            const first = try e.prefill(ids[0..rows], null, null, null);
            try e.stream.synchronize();
            try hidden.verify(e, rows);
            try normed.verify(e, rows);
            candidate.bytes.clearRetainingCapacity();
            try candidate.prompt(e, rows, first);
            try candidate.decode(e);
            if (level == 0) {
                std.mem.swap(Witness, &reference, &candidate);
            } else try check.sameBytes("prompt hidden/logits/taps, active recurrent/cache state and following serial decode", candidate.bytes.items, reference.bytes.items);
            var digest: [32]u8 = undefined;
            std.crypto.hash.sha2.Sha256.hash((if (level == 0) reference else candidate).bytes.items, &digest, .{});
            std.debug.print("OVERLAP_ORACLE rank={d} rows={d} level={d} bytes={d} sha256={s}\n", .{ rank, rows, level, reference.bytes.items.len, std.fmt.bytesToHex(digest, .lower) });
            cases += 1;
        }
        check.pass("rank {d}: rows {d}, overlap 0/1/2 exact semantic witnesses and guards", .{ rank, rows });
    }
    var times: [3][5]f64 = undefined;
    for (0..5) |sample| {
        for (0..3) |step| {
            const level = (sample + step) % 3;
            try prepare(e, hidden, normed, @intCast(level));
            const start = check.now(gpu.io);
            const first = try e.prefill(ids, null, null, null);
            try e.stream.synchronize();
            const elapsed = check.now(gpu.io) - start;
            times[level][sample] = @as(f64, @floatFromInt(elapsed)) / 1e6;
            try hidden.verify(e, 2048);
            try normed.verify(e, 2048);
            candidate.bytes.clearRetainingCapacity();
            try candidate.prompt(e, 2048, first);
            try candidate.decode(e);
            try check.sameBytes("timed overlap arm retains full 2048-row semantic witness", candidate.bytes.items, reference.bytes.items);
            std.debug.print("OVERLAP_SAMPLE rank={d} rows=2048 sample={d} order={d} level={d} host_ms={d:.6}\n", .{ rank, sample, step, level, times[level][sample] });
        }
    }
    for (&times, 0..) |*values, level| {
        const median = check.median(values);
        std.debug.print("OVERLAP_TIMING rank={d} layers=6 rows=2048 level={d} repeats=5 median_ms={d:.6} min_ms={d:.6} max_ms={d:.6}\n", .{ rank, level, median, values[0], values[4] });
    }
    check.pass("rank {d}: {d} correctness arms plus 15 paired timed arms, zero skips/errors; six-layer proof, not whole-model speedup", .{ rank, cases });
}
