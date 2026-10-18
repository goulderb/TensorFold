const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const X = cuda.experts.Nvfp4Experts;
const Variant = X.Variant;
const variants = std.meta.tags(Variant);
const experts = 288;
const dims = 4096;
const width = 1024;
const slots = 8;
const max_rounds = 31;

const Case = struct {
    rows: usize,
    distinct: bool = false,
    groups: usize = slots,

    fn used(c: Case) usize {
        return if (c.distinct) c.rows * slots else @min(c.rows * slots, c.groups);
    }
};
const cases = [_]Case{
    .{ .rows = 1 },                   .{ .rows = 2 },                 .{ .rows = 4 },                 .{ .rows = 8 },  .{ .rows = 16 },
    .{ .rows = 8, .distinct = true }, .{ .rows = 3 },                 .{ .rows = 7 },                 .{ .rows = 9 },  .{ .rows = 15 },
    .{ .rows = 17 },                  .{ .rows = 30 },                .{ .rows = 32 },                .{ .rows = 48 }, .{ .rows = 56 },
    .{ .rows = 30, .groups = 64 },    .{ .rows = 30, .groups = 128 }, .{ .rows = 56, .groups = 128 },
};

pub fn run(gpu: check.Gpu, args: []const [:0]const u8) !void {
    const launches = if (args.len > 0) try std.fmt.parseInt(usize, args[0], 10) else 144;
    const rounds = if (args.len > 1) try std.fmt.parseInt(usize, args[1], 10) else 7;
    if (args.len > 2 or launches == 0 or launches % (experts / slots) != 0 or rounds < 5 or rounds > max_rounds) return error.BadBenchmarkArguments;
    const sms: usize = @intCast(try gpu.ctx.attribute(.multiprocessor_count));
    var ex = try X.load(gpu.d, sms);
    defer ex.unload();
    try ex.loadCandidates(sms);
    var scale_errors = try cuda.DeviceBuffer.alloc(gpu.d, 4);
    defer scale_errors.free();
    try scale_errors.fill32(0, null);
    var scale_args: cuda.Args = .{};
    scale_args.add(scale_errors.ptr);
    const scale_check = try ex.mod.function("tf_nvfp4_warp_scale_check");
    var scale_stream = try cuda.Stream.init(gpu.d, false);
    defer scale_stream.deinit();
    try cuda.launch.launch(scale_check, .{ .grid = .{ .x = 256 }, .block = .{ .x = 32 } }, scale_stream, &scale_args);
    try scale_stream.synchronize();
    var scale_result: u32 = undefined;
    try scale_errors.download(0, std.mem.asBytes(&scale_result));
    try check.expect(scale_result == 0, "warp scale transport differs in {d} cases", .{scale_result});
    check.pass("warp scales: all 256 FP8 encodings, 131072 float-bit comparisons, including signed zero and NaNs", .{});
    var mod = try cuda.Module.load(gpu.d, cuda.kernels.experts);
    defer mod.unload();
    const router = try cuda.grouped.Router.resolve(mod);
    var stream = try cuda.Stream.init(gpu.d, false);
    defer stream.deinit();
    std.debug.print("BENCH nvfp4-experts sm={d} sms={d} hidden={d} rank_width={d} experts={d} topk={d} launches={d} rounds={d} equality=bit_exact timing=cuda_graph_events traffic=unique_codes_scales_global rotating_banks=true\n", .{
        try gpu.ctx.capability(), sms, dims, width, experts, slots, launches, rounds,
    });
    for ([_]cuda.experts.Epi{ .up, .down_f32, .down_bf16 }) |epi| {
        try projection(gpu, &ex, router, stream, epi, launches, rounds);
    }
    check.pass("nvfp4-experts-bench all candidates equal baseline for every case and weight bank", .{});
}

fn next(state: *u32) u32 {
    state.* = state.* *% 1664525 +% 1013904223;
    return state.*;
}

fn finite(bytes: []const u8, fp32: bool) !void {
    if (fp32) {
        for (std.mem.bytesAsSlice(u32, bytes)) |v| try check.expect(std.math.isFinite(@as(f32, @bitCast(v))), "nonfinite fp32 output", .{});
    } else {
        for (std.mem.bytesAsSlice(u16, bytes)) |v| try check.expect(std.math.isFinite(@as(f32, @bitCast(@as(u32, v) << 16))), "nonfinite bf16 output", .{});
    }
}

fn projection(gpu: check.Gpu, ex: *const X, router: cuda.grouped.Router, s: cuda.Stream, epi: cuda.experts.Epi, launches: usize, rounds: usize) !void {
    const up = epi == .up;
    const parts: usize = if (up) 2 else 1;
    const n: usize = if (up) width else dims;
    const k: usize = if (up) dims else width;
    const one = X.bytes(1, n, k, parts);
    var weights = try cuda.DeviceBuffer.alloc(gpu.d, experts * one);
    defer weights.free();
    const host = try gpu.gpa.alloc(u32, one / 4);
    defer gpu.gpa.free(host);
    var state: u32 = 0x5391024;
    for (0..experts) |e| {
        for (host, 0..) |*v, i| {
            if (i % X.words < 128) {
                v.* = next(&state);
            } else {
                var w: u32 = 0;
                for (0..4) |b| w |= (0x28 + ((next(&state) >> 24) % 25)) << @as(u5, @intCast(8 * b));
                v.* = w;
            }
        }
        try weights.upload(e * one, std.mem.sliceAsBytes(host));
    }
    var globals: [experts * 2]f32 = undefined;
    for (globals[0 .. experts * parts], 0..) |*g, i| g.* = @as(f32, @floatFromInt(1 + i % 7)) / 64.0;
    var scales = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(globals[0 .. experts * parts]));
    defer scales.free();
    const l: cuda.experts.Layer = .{
        .format = .nvfp4,
        .up = weights.ptr,
        .down = weights.ptr,
        .up_scale = scales.ptr,
        .down_scale = scales.ptr,
        .width = width,
        .dims = dims,
        .experts = experts,
    };
    for (cases) |c| try oneCase(gpu, ex, router, s, epi, l, c, one, launches, rounds);
}

fn bankLayer(l: cuda.experts.Layer, c: Case, bank: usize, one: usize, up: bool) cuda.experts.Layer {
    var r = l;
    const offset = bank * c.used();
    r.up += offset * one;
    r.down += offset * one;
    r.up_scale += offset * 8;
    r.down_scale += offset * 4;
    r.experts = c.used();
    r.limit = if (up) 8.0 else 0.0;
    return r;
}

fn oneCase(gpu: check.Gpu, ex: *const X, router: cuda.grouped.Router, s: cuda.Stream, epi: cuda.experts.Epi, l: cuda.experts.Layer, c: Case, one: usize, launches: usize, rounds: usize) !void {
    const up = epi == .up;
    const k: usize = if (up) dims else width;
    const n: usize = if (up) width else dims;
    const output_bytes: usize = if (epi == .down_f32) 4 else 2;
    const pairs = c.rows * slots;
    const banks = experts / c.used();
    const bound = cuda.grouped.maxItems(pairs, experts, 16);
    var picks: [56 * slots]i32 = undefined;
    for (picks[0..pairs], 0..) |*p, i| p.* = @intCast(if (c.distinct) i else i % c.used());
    var dpicks = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(picks[0..pairs]));
    defer dpicks.free();
    var members = try cuda.DeviceBuffer.alloc(gpu.d, pairs * 4);
    defer members.free();
    var items = try cuda.DeviceBuffer.alloc(gpu.d, bound * 12);
    defer items.free();
    var counts = try cuda.DeviceBuffer.alloc(gpu.d, 8);
    defer counts.free();
    const plan: cuda.grouped.Plan = .{ .members = members.ptr, .items = items.ptr, .counts = counts.ptr, .rank = 0, .hist = 0 };
    try router.route(s, dpicks.ptr, pairs, experts, 16, plan);
    try s.synchronize();
    var actual: [2]i32 = undefined;
    try counts.download(0, std.mem.asBytes(&actual));
    const quotient = pairs / c.used();
    const remainder = pairs % c.used();
    const expected_items = (c.used() - remainder) * ((quotient + 15) / 16) + remainder * ((quotient + 16) / 16);
    try check.expect(actual[0] == expected_items, "plan item count {d} expected {d}", .{ actual[0], expected_items });
    const host_x = try gpu.gpa.alloc(u16, (if (up) c.rows else pairs) * k);
    defer gpu.gpa.free(host_x);
    var state: u32 = 0x102453;
    for (host_x) |*v| {
        const f: f32 = @as(f32, @floatFromInt(@as(i32, @intCast(next(&state) >> 24)) - 128)) / 256.0;
        v.* = @truncate(@as(u32, @bitCast(f)) >> 16);
    }
    var input = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(host_x));
    defer input.free();
    const in: cuda.experts.Rows = .{ .x = input.ptr, .stride = k, .slots = if (up) slots else 0 };
    var reference = try cuda.DeviceBuffer.alloc(gpu.d, pairs * n * output_bytes);
    defer reference.free();
    var output = try cuda.DeviceBuffer.alloc(gpu.d, reference.len);
    defer output.free();
    const want = try gpu.gpa.alloc(u8, reference.len);
    defer gpu.gpa.free(want);
    const got = try gpu.gpa.alloc(u8, reference.len);
    defer gpu.gpa.free(got);
    for (0..banks) |bank| {
        const bl = bankLayer(l, c, bank, one, up);
        try reference.fill8(0xff, s.handle);
        try ex.runVariant(.baseline, s, epi, in, bl, plan, reference.ptr, bound, -1);
        try s.synchronize();
        try reference.download(0, want);
        try finite(want, epi == .down_f32);
        for (variants) |v| {
            try output.fill8(0xff, s.handle);
            try ex.runVariant(v, s, epi, in, bl, plan, output.ptr, bound, -1);
            try s.synchronize();
            try output.download(0, got);
            check.sameBytes(@tagName(v), got, want) catch |err| {
                std.debug.print("FAIL equality epi={t} rows={d} distinct={} bank={d}\n", .{ epi, c.rows, c.distinct, bank });
                return err;
            };
        }
    }
    var graphs: [variants.len]cuda.graph.Exec = undefined;
    var made: usize = 0;
    defer for (graphs[0..made]) |*g| g.deinit();
    for (variants, 0..) |v, vi| {
        try cuda.graph.beginCapture(s, .thread_local);
        for (0..launches) |i| try ex.runVariant(v, s, epi, in, bankLayer(l, c, i % banks, one, up), plan, output.ptr, bound, -1);
        var graph = try cuda.graph.endCapture(s);
        defer graph.deinit();
        graphs[vi] = try graph.instantiate();
        made += 1;
        try graphs[vi].upload(s);
        try graphs[vi].launchOn(s);
        try s.synchronize();
    }
    var start = try cuda.Event.init(gpu.d, true);
    defer start.deinit();
    var end = try cuda.Event.init(gpu.d, true);
    defer end.deinit();
    var samples: [variants.len][max_rounds]f64 = undefined;
    for (0..rounds) |round| {
        for (0..variants.len) |j| {
            const vi = if (round % 2 == 0) j else variants.len - 1 - j;
            try start.record(s);
            try graphs[vi].launchOn(s);
            try end.record(s);
            try end.synchronize();
            samples[vi][round] = @as(f64, try cuda.Event.elapsedMs(start, end)) * 1000.0 / @as(f64, @floatFromInt(launches));
        }
    }
    try reference.fill8(0xff, s.handle);
    try ex.runVariant(.baseline, s, epi, in, bankLayer(l, c, (launches - 1) % banks, one, up), plan, reference.ptr, bound, -1);
    try s.synchronize();
    try reference.download(0, want);
    for (graphs) |g| {
        try g.launchOn(s);
        try s.synchronize();
        try output.download(0, got);
        try check.sameBytes("graph final output", got, want);
    }
    const parts: usize = if (up) 2 else 1;
    const touched = c.used() * (one + parts * 4);
    const base = check.median(samples[0][0..rounds]);
    const ei: usize = @backingInt(epi);
    for (variants, 0..) |v, vi| {
        const median = check.median(samples[vi][0..rounds]);
        const resident = if (vi == 0) ex.resident[ei] else ex.candidate_resident[vi - 1][ei];
        const f = if (vi == 0) ex.fns[ei] else ex.candidate_fns[vi - 1][ei];
        const registers = try f.attribute(.num_regs);
        const local_bytes = try f.attribute(.local_size_bytes);
        std.debug.print("RESULT nvfp4-experts epi={t} rows={d} routing={s} used={d} banks={d} variant={t} grid={d} block=128 resident_blocks={d} regs={d} local_B={d} bytes={d} us={d:.3} min_us={d:.3} max_us={d:.3} effective_GBps={d:.3} speedup={d:.4} equality=bit_exact\n", .{
            epi,                                          c.rows,                  if (c.distinct) "distinct" else "shared",           c.used(),      banks,   v,
            X.decodeGrid(v, bound, n / X.cols, resident), resident,                registers,                                          local_bytes,   touched, median,
            samples[vi][0],                               samples[vi][rounds - 1], @as(f64, @floatFromInt(touched)) / median / 1000.0, base / median,
        });
    }
}

test "benchmark GLM TP2 shapes and rotating banks cover 288 experts without overrun" {
    try std.testing.expectEqual(@as(usize, 4096 * 1024 * 9 / 8), X.bytes(1, width, dims, 2));
    for (cases) |c| {
        const banks = experts / c.used();
        try std.testing.expect(banks >= 2);
        try std.testing.expect(banks * c.used() <= experts);
        try std.testing.expect(c.rows <= 56);
        const l: cuda.experts.Layer = .{ .format = .nvfp4, .up = 0, .down = 0, .up_scale = 0, .down_scale = 0, .width = width, .dims = dims, .experts = experts };
        const b = bankLayer(l, c, banks - 1, X.bytes(1, width, dims, 2), true);
        try std.testing.expect(b.up + X.bytes(c.used(), width, dims, 2) <= X.bytes(experts, width, dims, 2));
        try std.testing.expect(b.up_scale + c.used() * 8 <= experts * 8);
    }
    std.mem.doNotOptimizeAway(&X.loadCandidates);
    std.mem.doNotOptimizeAway(&run);
    std.mem.doNotOptimizeAway(&projection);
    std.mem.doNotOptimizeAway(&oneCase);
}
