const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");
const Mode = cuda.ck.NativeMode;
const max_rows = 129;

fn bf(bits: u16) f32 {
    return @bitCast(@as(u32, bits) << 16);
}
fn tobf(v: f32) u16 {
    const bits: u32 = @bitCast(v);
    return @truncate((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
}
fn weightValue(row: usize, col: usize) f32 {
    return @as(f32, @floatFromInt(@as(i32, @intCast((row * 3 + col * 5) % 29)) - 14)) / 8;
}
fn inputValue(row: usize, col: usize) f32 {
    if (row == 0) return 0;
    const amplitudes = [_]f32{ 1, 1.0 / 1024.0, 128, 0.5 };
    return @as(f32, @floatFromInt(@as(i32, @intCast((row * 11 + col * 7) % 31)) - 15)) / 16 * amplitudes[row % 4];
}

fn nibble(v: f32) u8 {
    var best: u8 = 0;
    var distance = std.math.inf(f32);
    for (0..8) |i| {
        const next = @abs(@abs(v) - @as(f32, @floatCast(core.quant.nvfp4.e2m1(@intCast(i)))));
        if (next < distance or (next == distance and i % 2 == 0)) {
            best = @intCast(i);
            distance = next;
        }
    }
    return best | (if (v < 0 and best != 0) @as(u8, 8) else 0);
}

fn referenceRow(mode: Mode, source: []const f32, scale: f32, out: []f32) void {
    const inv: f32 = 1 / scale;
    if (mode == .fp8) {
        for (source, out) |v, *result| result.* = @floatCast(core.quant.fp8.toE4m3(v * inv));
        return;
    }
    for (0..source.len / 16) |block| {
        const values = source[block * 16 ..][0..16];
        var amax: f32 = 0;
        for (values) |v| amax = @max(amax, @abs(v));
        const sf: f32 = @floatCast(core.quant.fp8.toE4m3(inv * (amax * @as(f32, 1.0 / 6.0))));
        const mul: f32 = if (sf == 0) 0 else inv / sf;
        for (values, out[block * 16 ..][0..16]) |v, *result| {
            result.* = @as(f32, @floatCast(core.quant.nvfp4.e2m1(nibble(v * mul)))) * sf;
        }
    }
}

pub fn run(gpu: check.Gpu) !void {
    try check.expect(try gpu.ctx.capability() == 121, "ck-dense needs sm121", .{});
    var stream = try cuda.Stream.init(gpu.d, false);
    defer stream.deinit();
    var ck = try cuda.ck.Ck.load(gpu.d, 12, @intCast(try gpu.ctx.attribute(.multiprocessor_count)));
    defer ck.unload();
    var pack = try cuda.modelopt.Packer.load(gpu.d);
    defer pack.unload();
    try packing(gpu, stream, &ck, &pack);
    for ([_]Mode{ .fp8, .nvfp4 }) |mode| try edges(gpu, stream, &ck, &pack, mode);
    for ([_][2]usize{ .{ 160, 4096 }, .{ 12576, 4096 }, .{ 4096, 128 }, .{ 4096, 6144 } }) |shape| {
        for ([_]Mode{ .fp8, .nvfp4 }) |mode| try dense(gpu, stream, &ck, &pack, mode, shape[0], shape[1]);
    }
}

fn edges(gpu: check.Gpu, s: cuda.Stream, ck: *const cuda.ck.Ck, pack: *const cuda.modelopt.Packer, mode: Mode) !void {
    const n = 160;
    const k = 128;
    const stride = k + 8;
    const npad = 256;
    const row_bits = [_]u16{ 0x0000, 0x0080, 0x0040, 0x7f00, 0x3f80, 0x7fc1, 0xbf80, 0x7f80, 0x4000, 0xff80, 0x3f00, 0xff00, 0x4040, 0x3e80, 0xc000, 0x8000, 0x8040 };
    const rows = row_bits.len;
    var weights: [n * k]u16 = @splat(0);
    for (0..n) |col| weights[col * k + (col * 17) % k] = tobf(if (col % 2 == 0) 1.5 else -1.5);
    var source = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(&weights));
    defer source.free();
    var raw = try cuda.DeviceBuffer.alloc(gpu.d, npad * k);
    defer raw.free();
    var raw_scales = try cuda.DeviceBuffer.alloc(gpu.d, npad * k / 16);
    defer raw_scales.free();
    var codes = try cuda.DeviceBuffer.alloc(gpu.d, npad * k);
    defer codes.free();
    var scales = try cuda.DeviceBuffer.alloc(gpu.d, npad * k / 16);
    defer scales.free();
    try raw.fill8(0, s.handle);
    try raw_scales.fill8(0, s.handle);
    const weight: cuda.qlinear.Weight = if (mode == .fp8) blk: {
        try ck.quantizeWeight8(s, source.ptr, n, k, 1, raw.ptr);
        try pack.fp8Lane(s, raw.ptr, codes.ptr, n, k);
        break :blk .{ .ck8 = .{ .codes = codes.ptr, .scale = 1, .n = n, .k = k, .npad = npad } };
    } else blk: {
        try ck.quantizeWeight4(s, source.ptr, n, k, 1, raw.ptr, raw_scales.ptr);
        try ck.pack(s, raw.ptr, raw_scales.ptr, 1, npad, k, 0, k, codes.ptr, scales.ptr);
        break :blk .{ .ck4 = .{ .words = codes.ptr, .scales = scales.ptr, .scale = 1, .n = n, .k = k, .npad = npad } };
    };
    var inputs: [rows * stride]u16 = @splat(0x7fc1);
    var packed_rows: [rows * k]u16 = undefined;
    for (row_bits, 0..) |bits, row| {
        @memset(inputs[row * stride ..][0..k], if (std.math.isFinite(bf(bits))) bits else 0x3f80);
        inputs[row * stride + k - 1] = bits;
        if (bf(bits) == 0) for (0..k) |col| {
            inputs[row * stride + col] = if (col % 2 == 0) 0x0000 else 0x8000;
        };
        @memcpy(packed_rows[row * k ..][0..k], inputs[row * stride ..][0..k]);
    }
    var x = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(&inputs));
    defer x.free();
    var xp = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(&packed_rows));
    defer xp.free();
    var workspace = try cuda.DeviceBuffer.alloc(gpu.d, try weight.nativeWorkspaceBytes(rows));
    defer workspace.free();
    const work: cuda.ck.Workspace = .{ .ptr = workspace.ptr, .bytes = workspace.len };
    var output = try cuda.DeviceBuffer.alloc(gpu.d, rows * n * 4);
    defer output.free();
    const linear: cuda.qlinear.Linear = .{ .ck = ck };
    try linear.promptF32(s, weight, .{ .x = x.ptr, .ldx = stride, .native = work }, output.ptr, rows);
    try s.synchronize();
    var full: [rows * n]f32 = undefined;
    try output.download(0, std.mem.sliceAsBytes(&full));
    var row_scales: [rows]f32 = undefined;
    const layout = try cuda.ck.workspaceLayout(mode, rows, n, k);
    try workspace.download(layout.scales, std.mem.sliceAsBytes(&row_scales));
    for (row_bits, 0..) |bits, row| {
        if (!std.math.isFinite(bf(bits))) {
            try check.expect(std.math.isNan(row_scales[row]), "{t} nonfinite edge row {d} has NaN scale", .{ mode, row });
        } else if (bf(bits) == 0) {
            try check.expect(row_scales[row] == 1, "{t} signed zero edge row uses literal scale 1", .{mode});
        } else if ((bits & 0x7fff) <= 0x0080) {
            try check.expect(@as(u32, @bitCast(row_scales[row])) == 0x00800000, "{t} tiny normal/subnormal row uses literal FLT_MIN scale", .{mode});
        }
    }
    var narrow: [rows * n]u16 = undefined;
    try linear.prompt(s, weight, .{ .x = x.ptr, .ldx = stride, .native = work }, output.ptr, rows);
    try s.synchronize();
    try output.download(0, std.mem.sliceAsBytes(&narrow));
    for (row_bits, 0..) |bits, row| for (0..n) |col| {
        const v = full[row * n + col];
        const b = narrow[row * n + col];
        if (!std.math.isFinite(bf(bits))) {
            try check.expect(std.math.isNan(v) and std.math.isNan(bf(b)), "{t} exceptional row {d} col {d} is NaN in F32 and BF16", .{ mode, row, col });
        } else if (bf(bits) == 0) {
            try check.expect(v == 0 and bf(b) == 0, "{t} signed zero row has exact zero F32/BF16 output", .{mode});
        } else {
            const want = @as(f64, bf(bits)) * @as(f64, if (col % 2 == 0) 1.5 else -1.5);
            const tolerance = @abs(want) / 16;
            try check.expect(std.math.isFinite(v) and std.math.isFinite(bf(b)) and @abs(@as(f64, v) - want) <= tolerance and @abs(@as(f64, bf(b)) - want) <= tolerance, "{t} finite edge row {d} col {d}: F32 {e}, BF16 {e}, independent ideal {e}", .{ mode, row, col, v, bf(b), want });
            try check.expect(b == tobf(v), "{t} finite edge BF16 rounds its F32 output once", .{mode});
        }
    };
    var got: [rows * n]f32 = undefined;
    var got_bf: [rows * n]u16 = undefined;
    try linear.decodeF32(s, weight, .{ .x = xp.ptr, .ldx = k, .native = work }, output.ptr, rows);
    try s.synchronize();
    try output.download(0, std.mem.sliceAsBytes(&got));
    try linear.decode(s, weight, .{ .x = xp.ptr, .ldx = k, .native = work }, output.ptr, rows);
    try s.synchronize();
    try output.download(0, std.mem.sliceAsBytes(&got_bf));
    for (row_bits, 0..) |bits, row| {
        const start = row * n;
        if (!std.math.isFinite(bf(bits))) {
            for (got[start..][0..n], got_bf[start..][0..n]) |v, b| try check.expect(std.math.isNan(v) and std.math.isNan(bf(b)), "{t} packed exceptional row stays NaN in both epilogues", .{mode});
            continue;
        }
        try check.sameBytes("finite edge packed/strided F32 isolation", std.mem.sliceAsBytes(got[start..][0..n]), std.mem.sliceAsBytes(full[start..][0..n]));
        try check.sameBytes("finite edge packed/strided BF16 isolation", std.mem.sliceAsBytes(got_bf[start..][0..n]), std.mem.sliceAsBytes(narrow[start..][0..n]));
        try linear.decodeF32(s, weight, .{ .x = x.ptr + row * stride * 2, .ldx = stride, .native = work }, output.ptr, 1);
        try s.synchronize();
        try output.download(0, std.mem.sliceAsBytes(got[0..n]));
        try check.sameBytes("finite edge F32 row alone survives exceptional neighbors", std.mem.sliceAsBytes(got[0..n]), std.mem.sliceAsBytes(full[start..][0..n]));
        try linear.decode(s, weight, .{ .x = x.ptr + row * stride * 2, .ldx = stride, .native = work }, output.ptr, 1);
        try s.synchronize();
        try output.download(0, std.mem.sliceAsBytes(got_bf[0..n]));
        try check.sameBytes("finite edge BF16 row alone survives exceptional neighbors", std.mem.sliceAsBytes(got_bf[0..n]), std.mem.sliceAsBytes(narrow[start..][0..n]));
    }
    check.pass("native {t} edge rows: signed zero, scale floor, finite extremes, NaN/Inf, F32/BF16 and neighbor isolation", .{mode});
}

fn packing(gpu: check.Gpu, s: cuda.Stream, ck: *const cuda.ck.Ck, pack: *const cuda.modelopt.Packer) !void {
    const n = 160;
    const k = 128;
    const npad = 256;
    var values: [n * k]u16 = undefined;
    for (&values, 0..) |*v, i| v.* = tobf(@floatFromInt(i % 16));
    var x = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(&values));
    defer x.free();
    var raw = try cuda.DeviceBuffer.alloc(gpu.d, npad * k);
    defer raw.free();
    var codes = try cuda.DeviceBuffer.alloc(gpu.d, npad * k);
    defer codes.free();
    try ck.quantizeWeight8(s, x.ptr, n, k, 1, raw.ptr);
    try pack.fp8Lane(s, raw.ptr, codes.ptr, n, k);
    try s.synchronize();
    var row: [16]u8 = undefined;
    try raw.download(0, &row);
    const literal = [_]u8{ 0x00, 0x38, 0x40, 0x44, 0x48, 0x4a, 0x4c, 0x4e, 0x50, 0x51, 0x52, 0x53, 0x54, 0x55, 0x56, 0x57 };
    try check.sameBytes("native FP8 weight quantizer emits plain row-major codes", &row, &literal);
    var fragment: [8]u8 = undefined;
    try codes.download(0, &fragment);
    const expected = [_]u8{ 0x00, 0x38, 0x50, 0x51, 0x00, 0x38, 0x50, 0x51 };
    try check.sameBytes("native FP8 weight fragment is packed_rows exactly once", &fragment, &expected);
    var raw4: [npad * k / 2]u8 = @splat(0);
    for (raw4[0 .. n * k / 2], 0..) |*v, i| v.* = @as(u8, @intCast((2 * i) % 16)) | (@as(u8, @intCast((2 * i + 1) % 16)) << 4);
    var c4 = try cuda.DeviceBuffer.fromHost(gpu.d, &raw4);
    defer c4.free();
    const scales: [npad * k / 16]u8 = @splat(0x38);
    var s4 = try cuda.DeviceBuffer.fromHost(gpu.d, &scales);
    defer s4.free();
    var tiled = try cuda.DeviceBuffer.alloc(gpu.d, scales.len);
    defer tiled.free();
    try ck.pack(s, c4.ptr, s4.ptr, 1, npad, k, 0, k, codes.ptr, tiled.ptr);
    try s.synchronize();
    var words: [4]u32 = undefined;
    try codes.download(0, std.mem.sliceAsBytes(&words));
    try check.expect(std.mem.eql(u32, &words, &.{ 0x76543210, 0x76543210, 0xfedcba98, 0xfedcba98 }), "CK4 words use contiguous nibble fragments", .{});
    check.pass("native dense packing distinguishes activation FP8 order and weight-only FP4 order", .{});
}

fn dense(gpu: check.Gpu, s: cuda.Stream, ck: *cuda.ck.Ck, pack: *const cuda.modelopt.Packer, mode: Mode, n: usize, k: usize) !void {
    const a = gpu.gpa;
    const npad = try cuda.ck.shape(n, k);
    const stride = k + 8;
    const weights = try a.alloc(u16, n * k);
    defer a.free(weights);
    for (weights, 0..) |*v, i| v.* = tobf(weightValue(i / k, i % k));
    const scale = try cuda.ck.tensorScale(1.75, mode);
    var source = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(weights));
    defer source.free();
    var raw = try cuda.DeviceBuffer.alloc(gpu.d, npad * k);
    defer raw.free();
    var raw_scales = try cuda.DeviceBuffer.alloc(gpu.d, npad * k / 16);
    defer raw_scales.free();
    var codes = try cuda.DeviceBuffer.alloc(gpu.d, npad * k);
    defer codes.free();
    var scales = try cuda.DeviceBuffer.alloc(gpu.d, npad * k / 16);
    defer scales.free();
    try raw.fill8(0, s.handle);
    try raw_scales.fill8(0, s.handle);
    const weight: cuda.qlinear.Weight = if (mode == .fp8) blk: {
        try ck.quantizeWeight8(s, source.ptr, n, k, scale, raw.ptr);
        try pack.fp8Lane(s, raw.ptr, codes.ptr, n, k);
        break :blk .{ .ck8 = .{ .codes = codes.ptr, .scale = scale, .n = @intCast(n), .k = @intCast(k), .npad = @intCast(npad) } };
    } else blk: {
        try ck.quantizeWeight4(s, source.ptr, n, k, scale, raw.ptr, raw_scales.ptr);
        try ck.pack(s, raw.ptr, raw_scales.ptr, 1, npad, k, 0, k, codes.ptr, scales.ptr);
        break :blk .{ .ck4 = .{ .words = codes.ptr, .scales = scales.ptr, .scale = scale, .n = @intCast(n), .k = @intCast(k), .npad = @intCast(npad) } };
    };
    const inputs = try a.alloc(u16, max_rows * stride);
    defer a.free(inputs);
    const packed_rows = try a.alloc(u16, max_rows * k);
    defer a.free(packed_rows);
    for (inputs, 0..) |*v, i| v.* = tobf(if (i % stride < k) inputValue(i / stride, i % stride) else 1024);
    for (0..max_rows) |row| @memcpy(packed_rows[row * k ..][0..k], inputs[row * stride ..][0..k]);
    var x = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(inputs));
    defer x.free();
    var xp = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(packed_rows));
    defer xp.free();
    var workspace = try cuda.DeviceBuffer.alloc(gpu.d, try weight.nativeWorkspaceBytes(max_rows));
    defer workspace.free();
    const work: cuda.ck.Workspace = .{ .ptr = workspace.ptr, .bytes = workspace.len };
    var output = try cuda.DeviceBuffer.alloc(gpu.d, max_rows * n * 4 + 256);
    defer output.free();
    const linear: cuda.qlinear.Linear = .{ .ck = ck };
    const in: cuda.qlinear.Rows = .{ .x = x.ptr, .ldx = stride, .native = work };
    try output.fill8(0x7d, s.handle);
    try linear.promptF32(s, weight, in, output.ptr, max_rows);
    try s.synchronize();
    const full = try a.alloc(f32, max_rows * n);
    defer a.free(full);
    try output.download(0, std.mem.sliceAsBytes(full));
    for (full) |v| try check.expect(std.math.isFinite(v), "{t} {d}x{d} output finite", .{ mode, n, k });
    var guard: [256]u8 = undefined;
    try output.download(max_rows * n * 4, &guard);
    try check.expect(std.mem.allEqual(u8, &guard, 0x7d), "native tail writes stay within output", .{});

    const xr = try a.alloc(f32, k);
    defer a.free(xr);
    const xq = try a.alloc(f32, k);
    defer a.free(xq);
    const wr = try a.alloc(f32, k);
    defer a.free(wr);
    const wq = try a.alloc(f32, k);
    defer a.free(wq);
    for ([_]usize{ 0, 1, 2, 7, 17, 128 }) |row| {
        var amax: f32 = 0;
        for (xr, 0..) |*v, col| {
            v.* = bf(inputs[row * stride + col]);
            amax = @max(amax, @abs(v.*));
        }
        const act = try cuda.ck.tensorScale(amax, mode);
        referenceRow(mode, xr, act, xq);
        for ([_]usize{ 0, 63, 64, n - 32, n - 1 }) |col| {
            for (wr, 0..) |*v, i| v.* = bf(weights[col * k + i]);
            referenceRow(mode, wr, scale, wq);
            var sum: f64 = 0;
            var magnitude: f64 = 0;
            for (xq, wq) |u, v| {
                const product = @as(f64, u) * @as(f64, v);
                sum += product;
                magnitude += @abs(product);
            }
            const factor: f32 = act * scale;
            const want = sum * factor;
            const tolerance = @max(1e-5, magnitude * @abs(factor) * 3e-5);
            try check.expect(@abs(@as(f64, full[row * n + col]) - want) <= tolerance, "{t} {d}x{d} row {d} col {d}: {e} vs FP64 {e}, tolerance {e}", .{ mode, n, k, row, col, full[row * n + col], want, tolerance });
        }
    }

    const got = try a.alloc(f32, full.len);
    defer a.free(got);
    for ([_]usize{ 1, 3, 8, 17, 33, 128 }) |rows| {
        try linear.decodeF32(s, weight, .{ .x = xp.ptr, .ldx = k, .native = work }, output.ptr, rows);
        try s.synchronize();
        try output.download(0, std.mem.sliceAsBytes(got[0 .. rows * n]));
        try check.sameBytes("native packed_rows rows equal strided prompt rows at every batch width", std.mem.sliceAsBytes(got[0 .. rows * n]), std.mem.sliceAsBytes(full[0 .. rows * n]));
    }
    for ([_]usize{ 1, 7, 17, 128 }) |row| {
        try linear.decodeF32(s, weight, .{ .x = x.ptr + row * stride * 2, .ldx = stride, .native = work }, output.ptr, 1);
        try s.synchronize();
        try output.download(0, std.mem.sliceAsBytes(got[0..n]));
        try check.sameBytes("native row alone equals its row in a heterogeneous batch", std.mem.sliceAsBytes(got[0..n]), std.mem.sliceAsBytes(full[row * n ..][0..n]));
    }
    const narrow = try a.alloc(u16, full.len);
    defer a.free(narrow);
    try linear.prompt(s, weight, in, output.ptr, max_rows);
    try s.synchronize();
    try output.download(0, std.mem.sliceAsBytes(narrow));
    for (narrow, full) |bits, v| try check.expect(bits == tobf(v), "native BF16 is one rounding of its F32 epilogue", .{});
    if (mode == .fp8) {
        var amax: f32 = 0;
        for (0..k) |i| amax = @max(amax, @abs(bf(inputs[stride + i])));
        const act = try cuda.ck.tensorScale(amax, .fp8);
        const layout = try cuda.ck.workspaceLayout(.fp8, 1, n, k);
        try ck.quant8(s, x.ptr + stride * 2, stride, 1, k, act, workspace.ptr);
        try ck.lane8(s, workspace.ptr, 1, .{ .mode = .fp8, .codes = codes.ptr, .scales = 0, .scale = scale, .n = @intCast(n), .k = @intCast(k), .npad = @intCast(npad) }, act, output.ptr, workspace.ptr + layout.partials);
        try s.synchronize();
        try output.download(0, std.mem.sliceAsBytes(narrow[0..n]));
        for (narrow[0..n], full[n..][0..n]) |bits, v| try check.expect(bits == tobf(v), "native FP8 agrees with the existing scalar CK8 consumer", .{});
        const clustered = ck.clusters;
        ck.clusters = false;
        defer ck.clusters = clustered;
        try linear.promptF32(s, weight, in, output.ptr, max_rows);
        try s.synchronize();
        try output.download(0, std.mem.sliceAsBytes(got));
        try check.sameBytes("native external split reduction equals cluster reduction", std.mem.sliceAsBytes(got), std.mem.sliceAsBytes(full));
    }
    check.pass("native {t} N={d} K={d}: tensor weight scale, FP64 samples, tails, row/batch/stride invariance and BF16/F32", .{ mode, n, k });
}
