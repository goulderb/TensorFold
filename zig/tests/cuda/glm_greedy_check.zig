//! Model-free TP=2 proof of the packed GLM greedy collective boundary.
//! Compares the old row-at-a-time top1/NCCL/readback path with Greedy using
//! NCCL and RoCE, then reports isolated 8K correctness and production-width timing.
const std = @import("std");
const cuda = @import("cuda");
const glm_cuda = @import("glm_cuda");
const check = @import("check.zig");

const max_rows = 64;
const small_vocab = 8192; // local tokens per rank; global synthetic vocabulary is 16384
const production_vocab = 77440; // 154880-token checkpoint vocabulary, split evenly across TP=2
const roce_cap = 512 << 10;
const samples = 5;
const calls_per_sample = 10;
const row_shapes = [_]usize{ 1, 3, 8, 16, 31, 32, 64 };
const timing_shapes = [_]usize{ 1, 8, 16, 32 };

pub fn run(gpu: check.Gpu, rank: usize) !void {
    if (rank >= 2) return error.BadRank;
    const a = gpu.gpa;
    var stream = try cuda.Stream.init(gpu.d, false);
    defer stream.deinit();
    var group = try cuda.tp_group.Group.open(rank, 2);
    defer group.close();
    try group.openRoce(a, gpu.io, gpu.d, stream, roce_cap);
    try check.expect(group.fast != null, "glm-greedy: TP2 RoCE gather opened on both ranks", .{});

    var module = try cuda.Module.load(gpu.d, cuda.kernels.glm_glue);
    defer module.unload();
    const top1 = try module.function("tf_glm_top1");

    var local = try cuda.DeviceBuffer.alloc(gpu.d, max_rows * production_vocab * 2);
    defer local.free();
    var old_pairs = try cuda.DeviceBuffer.alloc(gpu.d, 2 * 8);
    defer old_pairs.free();
    const old_host = try a.alloc(u32, 4);
    defer a.free(old_host);
    var picker = try glm_cuda.greedy.Greedy.init(a, gpu.d, 2, max_rows);
    defer picker.deinit();

    const nccl_context: glm_cuda.greedy.Context = .{
        .rank = rank,
        .vocab = .{ .lo = rank * small_vocab, .hi = (rank + 1) * small_vocab },
        .top1 = top1,
        .exchange = .{ .nccl = .{ .lib = &group.lib, .comm = group.comm } },
    };
    const roce_context: glm_cuda.greedy.Context = .{
        .rank = rank,
        .vocab = .{ .lo = rank * small_vocab, .hi = (rank + 1) * small_vocab },
        .top1 = top1,
        .exchange = .{ .nccl = .{ .lib = &group.lib, .comm = group.comm } },
        .group = &group,
    };

    var row_base: usize = 0;
    for (row_shapes) |rows| {
        const host = try a.alloc(u16, rows * small_vocab);
        defer a.free(host);
        fill(host, rank, small_vocab, row_base);
        try local.upload(0, std.mem.sliceAsBytes(host));
        try compareCase(gpu, &group, stream, top1, &picker, nccl_context, roce_context, local.ptr, old_pairs, old_host, rank, small_vocab, rows, row_base);
        row_base += rows;
    }
    check.pass("glm-greedy rank {d}: old row-wise NCCL, packed NCCL and packed RoCE equal CPU oracle for row counts {any}; synthetic local vocab {d}", .{ rank, row_shapes, small_vocab });

    // The timing input is production-shaped by width, but still synthetic and model-free.
    const timing_context_nccl: glm_cuda.greedy.Context = .{
        .rank = rank,
        .vocab = .{ .lo = rank * production_vocab, .hi = (rank + 1) * production_vocab },
        .top1 = top1,
        .exchange = .{ .nccl = .{ .lib = &group.lib, .comm = group.comm } },
    };
    const timing_context_roce: glm_cuda.greedy.Context = .{
        .rank = rank,
        .vocab = .{ .lo = rank * production_vocab, .hi = (rank + 1) * production_vocab },
        .top1 = top1,
        .exchange = .{ .nccl = .{ .lib = &group.lib, .comm = group.comm } },
        .group = &group,
    };
    for (timing_shapes) |rows| {
        const host = try a.alloc(u16, rows * production_vocab);
        defer a.free(host);
        fill(host, rank, production_vocab, 211);
        try local.upload(0, std.mem.sliceAsBytes(host));
        try timeCase(gpu, &group, stream, top1, &picker, timing_context_nccl, timing_context_roce, local.ptr, old_pairs, old_host, rows);
    }
}

fn compareCase(
    gpu: check.Gpu,
    group: *const cuda.tp_group.Group,
    stream: cuda.Stream,
    top1: cuda.Function,
    picker: *glm_cuda.greedy.Greedy,
    nccl_context: glm_cuda.greedy.Context,
    roce_context: glm_cuda.greedy.Context,
    logits: u64,
    old_pairs: cuda.DeviceBuffer,
    old_host: []u32,
    rank: usize,
    width: usize,
    rows: usize,
    row_base: usize,
) !void {
    const a = gpu.gpa;
    const want = try a.alloc(u32, rows);
    defer a.free(want);
    const old = try a.alloc(u32, rows + 2);
    defer a.free(old);
    const by_nccl = try a.alloc(u32, rows + 2);
    defer a.free(by_nccl);
    const by_roce = try a.alloc(u32, rows + 2);
    defer a.free(by_roce);
    @memset(old, 0xa5a5a5a5);
    @memset(by_nccl, 0xa5a5a5a5);
    @memset(by_roce, 0xa5a5a5a5);

    try oldRowwise(gpu, group, stream, top1, logits, width, rows, rank, old_pairs, old_host, old[0..rows]);
    try picker.draw(stream, nccl_context, logits, rows, by_nccl[0..rows]);
    try picker.draw(stream, roce_context, logits, rows, by_roce[0..rows]);
    for (0..rows) |r| want[r] = oracle(width, row_base + r);
    try check.expect(std.mem.eql(u32, old[0..rows], want), "glm-greedy old reference matches CPU oracle for rows={d}, width={d}", .{ rows, width });
    try check.expect(std.mem.eql(u32, by_nccl[0..rows], want), "glm-greedy packed NCCL matches CPU oracle for rows={d}, width={d}", .{ rows, width });
    try check.expect(std.mem.eql(u32, by_roce[0..rows], want), "glm-greedy packed RoCE matches CPU oracle for rows={d}, width={d}", .{ rows, width });
    try check.expect(old[rows] == 0xa5a5a5a5 and old[rows + 1] == 0xa5a5a5a5, "glm-greedy old path preserves output canaries", .{});
    try check.expect(by_nccl[rows] == 0xa5a5a5a5 and by_nccl[rows + 1] == 0xa5a5a5a5, "glm-greedy packed NCCL preserves output canaries", .{});
    try check.expect(by_roce[rows] == 0xa5a5a5a5 and by_roce[rows + 1] == 0xa5a5a5a5, "glm-greedy packed RoCE preserves output canaries", .{});
}

/// Recreates cuda_engine.greedy exactly: one unchanged top1 launch, 8-byte
/// NCCL all-gather, stream wait, group health check and world-pair readback per row.
fn oldRowwise(
    gpu: check.Gpu,
    group: *const cuda.tp_group.Group,
    stream: cuda.Stream,
    top1: cuda.Function,
    logits: u64,
    width: usize,
    rows: usize,
    rank: usize,
    pairs: cuda.DeviceBuffer,
    host: []u32,
    out: []u32,
) !void {
    for (0..rows) |row| {
        var args: cuda.Args = .{};
        args.add(logits + row * width * 2);
        args.add(@as(i64, @intCast(width)));
        args.add(@as(u32, @intCast(rank * width)));
        args.add(pairs.ptr + rank * 8);
        try cuda.launch.launch(top1, .{ .grid = .{ .x = 1 }, .block = .{ .x = 1024 } }, stream, &args);
        try group.lib.check(group.lib.api.ncclAllGather(pairs.ptr + rank * 8, pairs.ptr, 2, .u32, group.comm, stream.handle), "old greedy ncclAllGather");
        try stream.synchronize();
        try group.check();
        try gpu.d.check(gpu.d.api.cuMemcpyDtoH_v2(host.ptr, pairs.ptr, 16), "old greedy pair readback");
        out[row] = reduce(host[0..4]);
    }
}

fn reduce(pairs: []const u32) u32 {
    var best_score: f32 = @bitCast(pairs[0]);
    var best_id = pairs[1];
    var i: usize = 2;
    while (i + 1 < pairs.len) : (i += 2) {
        const score: f32 = @bitCast(pairs[i]);
        const id = pairs[i + 1];
        if (score > best_score or (score == best_score and id < best_id)) {
            best_score = score;
            best_id = id;
        }
    }
    return best_id;
}

fn fill(host: []u16, rank: usize, width: usize, base_row: usize) void {
    const rows = host.len / width;
    const total = width * 2;
    for (0..rows) |r| for (0..width) |i| {
        const id = rank * width + i;
        host[r * width + i] = valueBits(total, base_row + r, id);
    };
}

/// Input values are exact powers-of-two fractions except the explicit IEEE edge cases.
/// The seven scenarios force stride-1024 local ties, cross-rank ties, vocabulary
/// boundaries, signed zeros, infinities, NaNs and the all-invalid -Inf kernel result.
fn valueBits(total: usize, row: usize, id: usize) u16 {
    const mode = row % 7;
    if (mode == 0 and (id == 1024 or id == 2048 or id == 3072 or id == total / 2)) return bf16(12.0);
    if (mode == 1 and id == total - 1) return bf16(16.0);
    if (mode == 1 and id == 0) return bf16(15.0);
    if (mode == 2) return if (id % 2 == 0) 0x0000 else 0x8000;
    if (mode == 3) {
        if (id == 17 or id == total / 2 + 9) return 0x7f80; // +Inf tie across ranks
        return bf16(@as(f32, @floatFromInt(@as(i32, @intCast(id % 31)) - 15)) / 8.0);
    }
    if (mode == 4) return if (id % 3 == 0) 0xff80 else 0x7fc1; // every value is non-winning
    if (mode == 5 and (id == 42 or id == total - 1)) return 0x7fc1;
    if (mode == 5 and id == 900) return bf16(5.0);
    const signed: i32 = @as(i32, @intCast((id * 37 + row * 101) % 1024)) - 512;
    return bf16(@as(f32, @floatFromInt(signed)) / 128.0);
}

fn bf16(value: f32) u16 {
    return @truncate(@as(u32, @bitCast(value)) >> 16);
}

fn oracle(width: usize, row: usize) u32 {
    const total = width * 2;
    var best: f32 = -std.math.inf(f32);
    var best_id: u32 = 0;
    for (0..total) |id| {
        const bits = valueBits(total, row, id);
        const value: f32 = @bitCast(@as(u32, bits) << 16);
        // Mirrors tf_glm_top1: strict comparison from -Inf means NaN and -Inf
        // never seed a local candidate; the host then resolves equal scores by id.
        if (value > best or (value == best and id < best_id)) {
            best = value;
            best_id = @intCast(id);
        }
    }
    return best_id;
}

fn timeCase(
    gpu: check.Gpu,
    group: *const cuda.tp_group.Group,
    stream: cuda.Stream,
    top1: cuda.Function,
    picker: *glm_cuda.greedy.Greedy,
    nccl_context: glm_cuda.greedy.Context,
    roce_context: glm_cuda.greedy.Context,
    logits: u64,
    old_pairs: cuda.DeviceBuffer,
    old_host: []u32,
    rows: usize,
) !void {
    const a = gpu.gpa;
    const old = try a.alloc(u32, rows);
    defer a.free(old);
    const by_nccl = try a.alloc(u32, rows);
    defer a.free(by_nccl);
    const by_roce = try a.alloc(u32, rows);
    defer a.free(by_roce);
    // Warm all arms outside samples; each measured interval starts immediately before
    // invocation and ends only after the token slice is available on the host.
    try oldRowwise(gpu, group, stream, top1, logits, production_vocab, rows, nccl_context.rank, old_pairs, old_host, old);
    try picker.draw(stream, nccl_context, logits, rows, by_nccl);
    try picker.draw(stream, roce_context, logits, rows, by_roce);
    try check.expect(std.mem.eql(u32, old, by_nccl) and std.mem.eql(u32, old, by_roce), "glm-greedy production-width warmup outputs agree for rows={d}", .{rows});

    var ns: [3][samples]f64 = undefined;
    for (0..samples) |sample| {
        // Rotate the arm order on every paired repeat to blunt monotonic clock / thermal drift.
        for (0..3) |turn| {
            const arm = (sample + turn) % 3;
            const start = std.Io.Clock.awake.now(gpu.io).toNanoseconds();
            for (0..calls_per_sample) |_| switch (arm) {
                0 => try oldRowwise(gpu, group, stream, top1, logits, production_vocab, rows, nccl_context.rank, old_pairs, old_host, old),
                1 => try picker.draw(stream, nccl_context, logits, rows, by_nccl),
                2 => try picker.draw(stream, roce_context, logits, rows, by_roce),
                else => unreachable,
            };
            const end = std.Io.Clock.awake.now(gpu.io).toNanoseconds();
            ns[arm][sample] = @as(f64, @floatFromInt(end - start)) / @as(f64, calls_per_sample);
            try check.expect(std.mem.eql(u32, old, by_nccl) and std.mem.eql(u32, old, by_roce), "glm-greedy measured outputs agree for rows={d}, sample={d}, arm={d}", .{ rows, sample, arm });
        }
    }
    const labels = [_][]const u8{ "old-rowwise-NCCL", "packed-NCCL", "packed-RoCE" };
    for (0..3) |arm| {
        const median = check.median(&ns[arm]);
        var sorted = ns[arm];
        std.mem.sort(f64, &sorted, {}, std.sort.asc(f64));
        std.debug.print("TIMING glm-greedy rank {d} local_vocab={d} rows={d} arm={s} ns/call median={d:.0} min={d:.0} max={d:.0} paired_samples={d} calls/sample={d}\n", .{ nccl_context.rank, production_vocab, rows, labels[arm], median, sorted[0], sorted[samples - 1], samples, calls_per_sample });
    }
    std.debug.print("OPS glm-greedy rows={d}: old top1/NCCL-allGather(8B)/sync/group-check/readback per call={d}; packed top1 launches={d}, one allGather(alignForward(rows,2)*8B)/sync/group-check/readback per call\n", .{ rows, rows, rows });
}
