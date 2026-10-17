const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const glm = @import("glm_cuda");
const check = @import("check.zig");
const qr = @import("quant_ref_tests.zig");
const rows = [_]usize{ 1, 7, 16, 17, 64, 2048 };
const bf = qr.helpers.bfOf;

fn tensor(dtype: core.checkpoint.DType, n: usize, k: usize, bytes: []const u8) core.checkpoint.Tensor {
    var t: core.checkpoint.Tensor = .{ .dtype = dtype, .rank = 2, .shape = @splat(1), .bytes = bytes };
    t.shape[0] = n;
    t.shape[1] = k;
    return t;
}

fn matrix(gpu: check.Gpu, set: glm.glue.Set, stream: cuda.Stream, mx: bool, sliced: bool) !void {
    var arena: std.heap.ArenaAllocator = .init(gpu.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var rng = std.Random.DefaultPrng.init(0x5328 + @as(u64, @intFromBool(mx)));
    var rig: qr.RigT = .{ .gpu = gpu, .a = a, .s = stream, .rng = rng.random() };
    defer {
        for (rig.bufs.items) |*b| b.free();
        rig.bufs.deinit(gpu.gpa);
    }
    const n: usize = if (sliced) 289 else 161;
    const k: usize = if (sliced and !mx) 388 else 384;
    const codes = try a.alloc(u8, n * k);
    for (codes) |*c| {
        c.* = rig.rng.int(u8);
        if (c.* & 0x7f == 0x7f) c.* = 0x01;
    }
    const scales = try a.alloc(u8, if (mx) n * (k / 32) else ((n + 127) / 128) * ((k + 127) / 128) * 4);
    if (mx) {
        for (scales) |*s| s.* = 117 + rig.rng.uintLessThan(u8, 6);
    } else {
        for (0..scales.len / 4) |i| {
            const s: f32 = 0.001 + rig.rng.float(f32) * 0.01;
            std.mem.writeInt(u32, scales[i * 4 ..][0..4], @bitCast(s), .little);
        }
    }
    const rs: core.fp8_source.Span = if (sliced) .{ .lo = 123, .hi = 284 } else .{ .lo = 0, .hi = n };
    const cs: core.fp8_source.Span = if (sliced) .{ .lo = if (mx) 32 else 4, .hi = if (mx) 288 else 260 } else .{ .lo = 0, .hi = k };
    const wt = tensor(.f8_e4m3, n, k, codes);
    const st = tensor(if (mx) .u8 else .f32, if (mx) n else (n + 127) / 128, if (mx) k / 32 else (k + 127) / 128, scales);
    var p = try glm.source_dense.pack(a, .{ .weight = wt, .scale = st, .format = if (mx) .mxfp8 else .fp8_pb_wo }, &.{rs}, cs);
    defer p.deinit(a);
    try p.checkCuda();
    const wd = try a.alloc(f64, p.outs * p.ins);
    if (mx) try core.fp8_source.mxfp8(wt, st, rs, cs, wd) else try core.fp8_source.pbWo(wt, st, rs, cs, wd);
    const stored = try a.alloc(f64, p.out_total * p.in_total);
    try (try p.matrix()).rowsF64(0, p.out_total, stored);
    for (0..p.outs) |o| for (0..p.ins) |i| {
        try check.expect(wd[o * p.ins + i] == stored[(o + p.out0) * p.in_total + i + p.in0], "source FP64 references agree", .{});
    };
    const payload = try a.alloc(u8, p.resident());
    @memcpy(payload[0..p.codes.len], p.codes);
    @memcpy(payload[p.codes.len..], p.scales);
    const j = p.job();
    const groups = try core.exlr8.job.denseGroups(a, &.{j});
    const source: glm.b16.Proj = .{ .source = .{ .weights = try rig.dev(payload), .jobs = try rig.dev(std.mem.asBytes(&j)), .groups = try rig.dev(std.mem.sliceAsBytes(groups)), .groups_n = @intCast(groups.len), .n = p.outs, .k = p.ins } };
    const scratch_bytes = glm.b16.sourcePartBytes(rows[rows.len - 1], p.ins);
    const scratch = try rig.zeros(scratch_bytes + 256);
    // A canary after the exact scratch bound catches writing past the final 16-row tile.
    const canary = try rig.back(scratch + scratch_bytes, 256);
    const xf = try rig.bfs(rows[rows.len - 1] * p.ins, 1);
    const out32 = try rig.zeros(rows[rows.len - 1] * p.outs * 4);
    const out16 = try rig.zeros(rows[rows.len - 1] * p.outs * 2);
    const baseline = try rig.zeros(rows[rows.len - 1] * p.outs * 4);
    var most: f64 = 0;
    for ([_]usize{ p.ins, p.ins + 13 }) |stride| {
        const x = try a.alloc(u16, rows[rows.len - 1] * stride);
        @memset(x, 0x7fc0);
        for (0..rows[rows.len - 1]) |r| @memcpy(x[r * stride ..][0..p.ins], xf[r * p.ins ..][0..p.ins]);
        const dx = try rig.dev(std.mem.sliceAsBytes(x));
        try glm.b16.proj(set, stream, dx, stride, source, baseline, scratch, rows[rows.len - 1], p.outs, p.ins, true);
        const all = try rig.back(baseline, rows[rows.len - 1] * p.outs * 4);
        for (rows) |m| {
            try glm.b16.proj(set, stream, dx, stride, source, out32, scratch, m, p.outs, p.ins, true);
            try glm.b16.proj(set, stream, dx, stride, source, out16, scratch, m, p.outs, p.ins, false);
            const f32s = try rig.back(out32, m * p.outs * 4);
            const b16s = try rig.back(out16, m * p.outs * 2);
            try check.sameBytes("source row buckets equal the 2048-row call", f32s, all[0..f32s.len]);
            for (0..m) |r| for (0..p.outs) |o| {
                var sum: f64 = 0;
                var abs: f64 = 0;
                for (0..p.ins) |i| {
                    const term = bf(x[r * stride + i]) * wd[o * p.ins + i];
                    sum += term;
                    abs += @abs(term);
                }
                const at = r * p.outs + o;
                const got = std.mem.bytesToValue(f32, f32s[4 * at ..][0..4]);
                const rounded = core.quant.convert.bf16(got);
                const bgot = std.mem.bytesToValue(u16, b16s[2 * at ..][0..2]);
                try check.expect(bgot == rounded, "source BF16 equals rounding of FP32", .{});
                // FP32 weight product rounding plus lane FMAs and the five butterfly adds.
                const tol = abs * @as(f64, @floatFromInt(p.ins + 8)) * 0x1p-24 + 1e-30;
                const frac = @abs(@as(f64, got) - sum) / tol;
                most = @max(most, frac);
                try check.expect(std.math.isFinite(got) and frac <= 1, "source FP32 within (K+8)*2^-24*sum(abs(x*w))", .{});
                try check.expect(@abs(bf(bgot) - sum) <= tol + @abs(sum) * 0x1p-8, "source BF16 within FP32 bound plus rounding", .{});
            };
            try check.sameBytes("source scratch canary", try rig.back(scratch + scratch_bytes, 256), canary);
            check.pass("source {s} slice={any} rows={d} stride={d}: FP64, both epilogues, row invariance, scratch", .{ if (mx) "MXFP8" else "FP8_PB_WO", sliced, m, stride });
        }
    }
    check.pass("source {s} slice={any}: max {d:.5} of FP32 bound; scratch {d} bytes", .{ if (mx) "MXFP8" else "FP8_PB_WO", sliced, most, scratch_bytes });
}

pub fn run(gpu: check.Gpu) !void {
    var dm = try cuda.Module.load(gpu.d, cuda.kernels.exlr8_dense);
    defer dm.unload();
    var pm = try cuda.Module.load(gpu.d, cuda.kernels.torch_pointwise);
    defer pm.unload();
    const dense = try cuda.exlr8.Dense.resolve(dm);
    const set: glm.glue.Set = .{ .glue = undefined, .exlr8_dense = &dense, .source_upcast = try pm.function("tf_bf16_to_f32_kernel") };
    var stream = try cuda.Stream.init(gpu.d, true);
    defer stream.deinit();
    for ([_]bool{ true, false }) |mx| for ([_]bool{ false, true }) |sliced| try matrix(gpu, set, stream, mx, sliced);
}
