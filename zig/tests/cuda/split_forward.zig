//! `tf-cuda-test split-forward MODEL`: the seam arithmetic, checked by moving the seam.
//!
//! A layer split runs rank 0's blocks on one device and rank 1's on the other. The seam is the one place the two
//! differ from an unsplit run, so this test runs the same prompt chunk at two different seam points and requires
//! identical logits. A seam that dropped or double-applied the previous block's residual would differ between them,
//! and the two points are chosen so one carries a dense delta and the other a MoE delta into the seam norm.
//!
//! The seam rule: the sender runs the receiving rank's first block norm, so the handoff carries the residual h, the
//! normed rows y and their group sums xs, and the receiver starts with its first norm already done. That keeps the
//! norm's fused add (x + the previous block's delta) on the device that computed the delta.
//!
//! A single-device reference is not available on this workstation: the checkpoint is 17.3 GiB and the largest card
//! is 15.9 GiB, which is why the split exists. The end-to-end reference is the Spark native run for the same prompt.

const std = @import("std");
const cuda = @import("cuda");
const nemotron = @import("nemotron");
const check = @import("check.zig");
const pair_tests = @import("pair_tests.zig");

const fwd = nemotron.forward;

pub fn run(gpu: check.Gpu, model: []const u8) !void {
    const gpa = gpu.gpa;
    const io = gpu.io;
    const cfg = try nemotron.Config.read(gpa, io, model);
    const rows: usize = 64; // a chunk small enough to compare whole, wide enough to exercise every block kind

    // A deterministic prompt: ids in the model's vocabulary, so the embedding reads real rows.
    const prompt = try gpa.alloc(u32, rows);
    defer gpa.free(prompt);
    var seed: u32 = 12345;
    for (prompt) |*t| {
        seed = seed *% 1664525 +% 1013904223;
        t.* = seed % @as(u32, @intCast(cfg.vocab));
    }

    const pick = (try pair_tests.internalPair(gpu.d, gpa)) orelse {
        std.debug.print("SKIP split-forward: no two devices share a compute capability\n", .{});
        return;
    };

    // Four seam points, alternating in the form of the delta entering the seam norm: block 26 is attention and
    // block 28 mamba, so seams 27 and 29 carry a dense delta; block 27 is MoE and block 29 MoE, so seams 28 and 30
    // carry a MoE one. Every later seam is compared against the first, so a seam that mishandles either form shows.
    const seams = [_]usize{ 27, 28, 29, 30 };
    var results: [seams.len][]u8 = undefined;
    var made: usize = 0;
    defer for (results[0..made]) |r| gpa.free(r);
    for (seams, 0..) |mid, i| {
        results[i] = try split(gpu, model, cfg, prompt, mid, pick);
        made += 1;
        check.pass("split-forward seam {d}: {d} logits bytes, {d} distinct tokens in the top row", .{ mid, results[i].len, distinct(results[i]) });
        if (i > 0) try check.sameBytes("split-forward logits at seam", results[i], results[0]);
    }
    check.pass("split-forward: the logits are identical at seams {d}, {d}, {d} and {d}", .{ seams[0], seams[1], seams[2], seams[3] });
}

/// How many distinct bf16 values the first row of logits holds: a degenerate output (all zero, all one value) is a
/// forward that did not run, not a match.
fn distinct(bytes: []const u8) usize {
    var seen: [64]u16 = undefined;
    var n: usize = 0;
    var i: usize = 0;
    while (i + 2 <= bytes.len and n < seen.len) : (i += 2) {
        const v = std.mem.readInt(u16, bytes[i..][0..2], .little);
        var dup = false;
        for (seen[0..n]) |s| if (s == v) {
            dup = true;
        };
        if (!dup) {
            seen[n] = v;
            n += 1;
        }
    }
    return n;
}

/// The two ranks, each on its own device, driven through the chunk phases with the seam handed across.
fn split(gpu: check.Gpu, model: []const u8, cfg: nemotron.Config, prompt: []const u32, mid: usize, pick: [2]c_int) ![]u8 {
    const gpa = gpu.gpa;
    const io = gpu.io;
    const rows = prompt.len;

    var ctx: [2]cuda.Context = undefined;
    ctx[0] = try cuda.Context.init(gpu.d, pick[0]);
    defer ctx[0].deinit();
    ctx[1] = try cuda.Context.init(gpu.d, pick[1]);
    defer ctx[1].deinit();

    // Rank 0 loads [0, mid) plus the embedding and the receiving rank's first norm for the seam; rank 1 loads
    // [mid, layers) plus the final norm, the head and the MTP head.
    const ranges = [_]nemotron.weights.Range{
        .{ .first = 0, .last = mid, .embed = true, .head = false, .seam_norm = mid },
        .{ .first = mid, .last = cfg.layers, .embed = false, .head = true },
    };
    var engines: [2]*nemotron.Engine = undefined;
    var made: usize = 0;
    defer for (engines[0..made]) |e| e.deinit();
    for (ranges, 0..) |r, rank| {
        const mi = try ctx[rank].memInfo();
        std.debug.print("split-forward rank {d} on device {d}: {d:.2} GiB free before the load\n", .{ rank, pick[rank], gib(mi.free) });
        engines[rank] = try nemotron.Engine.init(gpa, io, &ctx[rank], model, null, .{
            .context = 4096,
            .mtp = false,
            .graphs = false,
            .sampling = null,
            .segments = 1,
            .range = r,
            .norm_done = rank == 1,
        });
        made += 1;
        const after = try ctx[rank].memInfo();
        std.debug.print("split-forward rank {d} loaded: {d:.2} GiB free on device {d} after\n", .{ rank, gib(after.free), pick[rank] });
    }

    // The seam crosses three buffers: the residual h (the next block's norm reads it), the normed rows y and their
    // 64-group sums xs (the receiving block's projections read them, since its norm is skipped).
    const h_bytes = rows * cfg.hidden * 2;
    const xs_bytes = rows * (cfg.hidden / 64) * 4;
    const seam_bytes = 2 * h_bytes + xs_bytes;
    var stage = try cuda.HostBuffer.allocMapped(gpu.d, seam_bytes);
    defer stage.free();

    // Rank 0: the chunk through its blocks, then the seam norm, which leaves h in w.x, y in b.y and xs in b.xs.
    const r0 = engines[0];
    var w0: fwd.Walk = .{ .rows = rows, .pos = 0 };
    try r0.ctx.makeCurrent();
    // The chunk reads its ids from p_ids, which the engine's own prefill uploads; the direct phase calls must too.
    try gpu.d.check(gpu.d.api.cuMemcpyHtoD_v2(r0.b.p_ids, prompt.ptr, rows * 4), "cuMemcpyHtoD");
    try r0.forward(null).chunkBegin(&w0);
    for (0..mid) |i| {
        try r0.forward(null).chunkPre(&w0, i);
        try r0.forward(null).chunkMixer(&w0, i);
        try r0.forward(null).chunkPost(&w0, i);
    }
    try r0.forward(null).finalNorm(&w0);
    try r0.stream.synchronize();
    check.pass("split-forward rank 0: its {d} blocks and the seam norm ran", .{mid});
    // The norm leaves h in the buffer w.x now points at, which alternates between h[0] and h[1].
    try gpu.d.check(gpu.d.api.cuMemcpyDtoH_v2(stage.bytes.ptr, w0.x, h_bytes), "cuMemcpyDtoH");
    try gpu.d.check(gpu.d.api.cuMemcpyDtoH_v2(stage.bytes.ptr + h_bytes, r0.b.y, h_bytes), "cuMemcpyDtoH");
    try gpu.d.check(gpu.d.api.cuMemcpyDtoH_v2(stage.bytes.ptr + 2 * h_bytes, r0.b.xs, xs_bytes), "cuMemcpyDtoH");

    // Rank 1: the seam in, then its blocks and the tail.
    const r1 = engines[1];
    try r1.ctx.makeCurrent();
    try gpu.d.check(gpu.d.api.cuMemcpyHtoD_v2(r1.b.h[0], stage.bytes.ptr, h_bytes), "cuMemcpyHtoD");
    try gpu.d.check(gpu.d.api.cuMemcpyHtoD_v2(r1.b.y, stage.bytes.ptr + h_bytes, h_bytes), "cuMemcpyHtoD");
    try gpu.d.check(gpu.d.api.cuMemcpyHtoD_v2(r1.b.xs, stage.bytes.ptr + 2 * h_bytes, xs_bytes), "cuMemcpyHtoD");
    const start = r0.forward(null).counters();
    var w1: fwd.Walk = .{ .rows = rows, .pos = 0, .x = r1.b.h[0], .delta = .none, .mj = start.mj, .aj = start.aj };
    for (mid..cfg.layers) |i| {
        try r1.forward(null).chunkPre(&w1, i);
        try r1.forward(null).chunkMixer(&w1, i);
        try r1.forward(null).chunkPost(&w1, i);
    }
    try r1.forward(null).chunkFinish(&w1);
    try r1.stream.synchronize();
    check.pass("split-forward rank 1: its {d} blocks and the tail ran", .{cfg.layers - mid});
    return logits(gpu, r1);
}

fn gib(bytes: usize) f64 {
    return @as(f64, @floatFromInt(bytes)) / (1 << 30);
}

fn logits(gpu: check.Gpu, e: *nemotron.Engine) ![]u8 {
    const n = e.c.vocab * 2;
    const out = try gpu.gpa.alloc(u8, n);
    errdefer gpu.gpa.free(out);
    try e.ctx.makeCurrent();
    try e.stream.synchronize();
    try gpu.d.check(gpu.d.api.cuMemcpyDtoH_v2(out.ptr, e.b.p_logits, n), "cuMemcpyDtoH");
    return out;
}
