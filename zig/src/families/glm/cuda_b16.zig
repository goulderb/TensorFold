//! GLM-5.3-Flash's BF16 matmuls on CUDA (attention, router, shared experts): the Python family's B16 path.

const std = @import("std");
const cuda = @import("cuda");
const glue = @import("cuda_glue.zig");
const mm = @import("cuda_mm.zig");

/// qmm.SPLIT_TARGET, B16_SPLIT_COLS, B16_BK, geometry.MLA_B16_SLICES and MLA_B16_ROWS_FROM.
pub const split_target = 192;
pub const split_cols = 64;
pub const bk = 64;
pub const max_slices = 8;
pub const rows_from = 128;

/// qmm.b16_split_k: K slices doubled while the 64-column tiles stay under 192 and each slice keeps 4 steps of 64.
pub fn splitK(n: usize, k: usize) usize {
    const tiles = (n + split_cols - 1) / split_cols;
    const steps = k / bk;
    var sk: usize = 1;
    while (sk < max_slices and tiles * sk < split_target and steps % (sk * 2) == 0 and steps / (sk * 2) >= 4) sk *= 2;
    return sk;
}

/// Triton's one-row _bmm (aot.json M = 1, GB10): BLOCK_N, stages by (n, k), one stage -> two; else tf_glm_bmm_decode.
const DecodeTile = struct { n: usize, k: usize, bn: u32, st: u32 };
const decode_tiles = [_]DecodeTile{
    .{ .n = 4096, .k = 4096, .bn = 32, .st = 2 },  .{ .n = 4096, .k = 128, .bn = 32, .st = 2 },
    .{ .n = 4096, .k = 2816, .bn = 32, .st = 2 },  .{ .n = 4096, .k = 5632, .bn = 32, .st = 2 },
    .{ .n = 5376, .k = 1536, .bn = 32, .st = 2 },  .{ .n = 8192, .k = 1536, .bn = 32, .st = 2 },
    .{ .n = 12576, .k = 4096, .bn = 32, .st = 3 }, .{ .n = 4096, .k = 1024, .bn = 32, .st = 3 },
    .{ .n = 160, .k = 4096, .bn = 32, .st = 3 },   .{ .n = 1280, .k = 4096, .bn = 32, .st = 3 },
    .{ .n = 2816, .k = 128, .bn = 32, .st = 3 },   .{ .n = 2688, .k = 128, .bn = 32, .st = 3 },
    .{ .n = 4096, .k = 8192, .bn = 64, .st = 2 },  .{ .n = 4096, .k = 704, .bn = 64, .st = 2 },
    .{ .n = 4096, .k = 16384, .bn = 64, .st = 2 }, .{ .n = 4096, .k = 1536, .bn = 64, .st = 2 },
    .{ .n = 4096, .k = 2048, .bn = 64, .st = 2 },  .{ .n = 16384, .k = 1536, .bn = 64, .st = 2 },
    .{ .n = 2048, .k = 4096, .bn = 64, .st = 2 },  .{ .n = 1408, .k = 4096, .bn = 64, .st = 2 },
    .{ .n = 8192, .k = 128, .bn = 64, .st = 2 },   .{ .n = 24896, .k = 4096, .bn = 64, .st = 2 },
    .{ .n = 77440, .k = 4096, .bn = 64, .st = 2 }, .{ .n = 154880, .k = 4096, .bn = 64, .st = 2 },
};

fn decodeTile(n: usize, k: usize) ?DecodeTile {
    for (decode_tiles) |t| if (t.n == n and t.k == k) return t;
    return null;
}

fn cdiv(a: usize, b: usize) usize {
    return (a + b - 1) / b;
}

/// The fp32 split-K partials a window of `m` rows needs (0 when it sums in registers or takes one slice).
pub fn partBytes(m: usize, n: usize, k: usize) usize {
    const sk = splitK(n, k);
    return if (m >= rows_from or sk == 1) 0 else sk * m * n * 4;
}

/// qmm.matmul for a B16 weight: out [m][n] bf16 (fp32 with `wide`) = x . w; 128+ rows sum in registers, else `part`.
pub fn matmul(set: glue.Set, stream: cuda.Stream, x: u64, x_stride: usize, w: u64, out: u64, part: u64, m: usize, n: usize, k: usize, wide: bool) !void {
    if (k % bk != 0) return error.KNotMultipleOf64;
    if (set.mm) |mk| { // tiled weights, packed or bf16
        const t: mm.Tiled = if (set.packs) |p| p.tiled(w, n, k) else .{ .w = w, .n = n, .k = k };
        return mm.matmul(mk, stream, x, x_stride, t, out, part, m, wide);
    }
    const sk = splitK(n, k);
    var a: cuda.Args = .{};
    a.add(x);
    a.add(w);
    a.add(out);
    if (m < rows_from) a.add(if (sk > 1) part else out);
    inline for (.{ @as(i64, @intCast(m)), @as(i64, @intCast(x_stride)), @as(i64, @intCast(n)), @as(i64, @intCast(k)), @as(c_int, @intCast(sk)), @as(c_int, @intFromBool(wide)) }) |v| a.add(v);
    if (m >= rows_from and x % 16 == 0 and x_stride % 8 == 0 and n % 8 == 0 and out % 16 == 0) { // the cp.async tiles
        // One slice past 256 columns: 128 x 128 blocks, two an SM; K slices or narrow outputs: 128 x 64 blocks.
        const one = sk == 1 and n > 256;
        const f = if (one) set.glue.bmm_tile1 else set.glue.bmm_tile;
        const bn: usize = if (one) 128 else 64;
        const shared: u32 = if (one) glue.bmm_tile1_smem[@intFromBool(wide)] else glue.bmm_tile_smem;
        return cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(cdiv(m, 128)), .y = @intCast(cdiv(n, bn)) }, .block = .{ .x = 256 }, .shared = shared }, stream, &a);
    }
    if (m >= rows_from) return cuda.launch.launch(set.glue.bmm_rows, .{ .grid = .{ .x = @intCast(cdiv(m, 128)), .y = @intCast(cdiv(n, 64)) }, .block = .{ .x = 128 } }, stream, &a);
    if (decodeTile(n, k)) |t| {
        if (x % 16 == 0 and x_stride % 8 == 0 and n % 8 == 0 and out % 16 == 0 and part % 16 == 0) {
            const f = switch (t.bn * 10 + t.st) {
                321 => set.glue.bmm_d32s1,
                322 => set.glue.bmm_d32s2,
                323 => set.glue.bmm_d32s3,
                642 => set.glue.bmm_d64s2,
                else => unreachable,
            };
            const shared: u32 = t.st * (16 + t.bn) * bk * 2; // Tile<16, bn, 64, st, 1, 4>'s stages (the output tile is smaller)
            try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(cdiv(m, 16)), .y = @intCast(cdiv(n, t.bn)), .z = @intCast(sk) }, .block = .{ .x = 128 }, .shared = shared }, stream, &a);
            if (sk > 1) try glue.sliceSum(set, stream, part, out, m * n, sk, wide);
            return;
        }
    }
    try cuda.launch.launch(set.glue.bmm_decode, .{ .grid = .{ .x = @intCast(cdiv(m, 16)), .y = @intCast(cdiv(n, 64)), .z = @intCast(sk) }, .block = .{ .x = 128 } }, stream, &a);
    if (sk > 1) try glue.sliceSum(set, stream, part, out, m * n, sk, wide);
}

/// EXLR8 dense MXFP8 / FP8_BLOCK resident weights (TF_GLM_DENSE=source).
pub const Source = struct { weights: u64, jobs: u64, groups: u64, groups_n: u32, n: u32, k: u32 };

/// A projection: BF16 (bmm), a quantized qlinear weight, or a source-dense EXLR8 job.
pub const Proj = union(enum) { b16: u64, q: cuda.qlinear.Weight, source: Source };

/// matmul for either kind: a quantized weight takes qlinear's decode rows under rows_from rows, its prompt rows above.
pub fn proj(set: glue.Set, stream: cuda.Stream, x: u64, x_stride: usize, p: Proj, out: u64, part: u64, m: usize, n: usize, k: usize, wide: bool) !void {
    switch (p) {
        .b16 => |w| return matmul(set, stream, x, x_stride, w, out, part, m, n, k, wide),
        .source => |s| return sourceMatmul(set, stream, x, x_stride, s, out, part, m, n, k, wide),
        .q => |w| {
            if (w.outputs() != n or w.inputs() != k) return error.ShapeMismatch;
            if (w == .affine4 and m < rows_from) try groupSums(set, stream, x, x_stride, part, m, k);
            const in: cuda.qlinear.Rows = .{ .x = x, .ldx = x_stride, .sums = part, .part = part, .native = .{ .ptr = part, .bytes = try w.nativeWorkspaceBytes(m) } };
            const l = set.linear;
            if (m >= rows_from) return if (wide) l.promptF32(stream, w, in, out, m) else l.prompt(stream, w, in, out, m);
            return if (wide) l.decodeF32(stream, w, in, out, m) else l.decode(stream, w, in, out, m);
        },
    }
}

pub fn sourcePartBytes(rows: usize, k: usize) usize {
    return @min(rows, 16) * k * 4;
}

fn upcast(f: cuda.Function, stream: cuda.Stream, x: u64, out: u64, count: usize) !void {
    var a: cuda.Args = .{};
    a.add(x);
    a.add(out);
    a.add(@as(u64, count));
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(@min(cdiv(count, 256), 65535)) }, .block = .{ .x = 256 } }, stream, &a);
}

/// Reuse the frame's matmul scratch for one compact FP32 tile, in stream order with its projection.
pub fn sourceMatmul(set: glue.Set, stream: cuda.Stream, x: u64, x_stride: usize, s: Source, out: u64, part: u64, m: usize, n: usize, k: usize, wide: bool) !void {
    if (s.n != n or s.k != k) return error.ShapeMismatch;
    if (k == 0 or k % 128 != 0 or n == 0 or x_stride < k) return error.SourceDenseGeometry;
    if (m == 0) return;
    const dense = set.exlr8_dense orelse return error.SourceDenseNotLoaded;
    const cast = set.source_upcast orelse return error.SourceDenseNotLoaded;
    var at: usize = 0;
    while (at < m) {
        const take: u32 = @intCast(@min(m - at, 16));
        if (x_stride == k) {
            try upcast(cast, stream, x + at * x_stride * 2, part, take * k);
        } else for (0..take) |r| try upcast(cast, stream, x + (at + r) * x_stride * 2, part + r * k * 4, k);
        const rows = @import("core").exlr8.job.Rows{ .rows = take, .x_stride = @intCast(k), .y_stride = @intCast(n), .y_f32 = @intFromBool(wide) };
        try dense.project(stream, part, s.weights, s.jobs, s.groups, s.groups_n, rows, out + at * n * @as(usize, if (wide) 4 else 2));
        at += take;
    }
}

test "source dense dispatch refuses shapes and makes zero rows a no-op before driver access" {
    const set: glue.Set = .{ .glue = undefined };
    const stream: cuda.Stream = undefined;
    const p: Proj = .{ .source = .{ .weights = 0, .jobs = 0, .groups = 0, .groups_n = 1, .n = 33, .k = 128 } };
    try proj(set, stream, 0, 128, p, 0, 0, 0, 33, 128, false);
    try std.testing.expectError(error.ShapeMismatch, proj(set, stream, 0, 128, p, 0, 0, 1, 32, 128, false));
    try std.testing.expectError(error.SourceDenseGeometry, proj(set, stream, 0, 127, p, 0, 0, 1, 33, 128, false));
    for ([_]usize{ 1, 7, 16, 17, 64, 2048 }) |m| for ([_]bool{ false, true }) |wide| {
        try std.testing.expectError(error.SourceDenseNotLoaded, proj(set, stream, 0, 128, p, 0, 0, m, 33, 128, wide));
        try std.testing.expectEqual(@as(usize, @min(m, 16)) * 128 * 4, sourcePartBytes(m, 128));
    };
}

test "source dense dispatch upcasts BF16 tiles before FP32 or BF16 projection on the frame stream" {
    const Rows = @import("core").exlr8.job.Rows;
    const Capture = struct {
        const Call = struct { f: usize, x: u64, out: u64, count: u64 = 0, rows: Rows = undefined };
        var calls: [2304]Call = undefined;
        var len: usize = 0;
        fn value(comptime T: type, args: [*]?*anyopaque, i: usize) T {
            return @as(*align(1) const T, @ptrCast(args[i].?)).*;
        }
        fn launch(f: cuda.abi.Function, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, stream: cuda.abi.Stream, args: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
            std.debug.assert(stream == @as(cuda.abi.Stream, @ptrFromInt(17)));
            const id = @intFromPtr(f.?);
            const a = args.?;
            calls[len] = .{ .f = id, .x = value(u64, a, 0), .out = value(u64, a, if (id == 99) 1 else 5) };
            if (id == 99) calls[len].count = value(u64, a, 2) else calls[len].rows = value(Rows, a, 4);
            len += 1;
            return 0;
        }
    };
    var driver: cuda.Driver = .{ .lib = undefined, .api = undefined };
    driver.api.cuLaunchKernel = Capture.launch;
    var dense: cuda.exlr8.Dense = .{ .decode_fn = undefined, .project_fns = undefined };
    for (&dense.project_fns, [_]usize{ 1, 2, 4, 8, 16 }) |*f, cap| f.* = .{ .d = &driver, .handle = @ptrFromInt(cap) };
    const set: glue.Set = .{ .glue = undefined, .exlr8_dense = &dense, .source_upcast = .{ .d = &driver, .handle = @ptrFromInt(99) } };
    const stream: cuda.Stream = .{ .d = &driver, .handle = @ptrFromInt(17) };
    const p: Proj = .{ .source = .{ .weights = 100, .jobs = 200, .groups = 300, .groups_n = 2, .n = 33, .k = 128 } };
    for ([_]usize{ 1, 7, 16, 17, 64, 2048 }) |m| for ([_]usize{ 128, 141 }) |stride| for ([_]bool{ false, true }) |wide| {
        Capture.len = 0;
        try proj(set, stream, 1000, stride, p, 2000, 3000, m, 33, 128, wide);
        var ci: usize = 0;
        var at: usize = 0;
        while (at < m) {
            const take: usize = @min(m - at, 16);
            const casts = if (stride == 128) 1 else take;
            for (0..casts) |r| {
                const call = Capture.calls[ci];
                ci += 1;
                try std.testing.expectEqual(@as(usize, 99), call.f);
                try std.testing.expectEqual(1000 + (at + r) * stride * 2, call.x);
                try std.testing.expectEqual(3000 + r * 128 * 4, call.out);
                try std.testing.expectEqual((if (casts == 1) take else 1) * 128, call.count);
            }
            const call = Capture.calls[ci];
            ci += 1;
            try std.testing.expectEqual(try std.math.ceilPowerOfTwo(usize, take), call.f);
            try std.testing.expectEqual(@as(u64, 3000), call.x);
            try std.testing.expectEqual(2000 + at * 33 * @as(usize, if (wide) 4 else 2), call.out);
            try std.testing.expectEqual(Rows{ .rows = @intCast(take), .x_stride = 128, .y_stride = 33, .y_f32 = @intFromBool(wide) }, call.rows);
            at += take;
        }
        try std.testing.expectEqual(ci, Capture.len);
    };
}

/// `rows` rows of `bytes` from `src` into `dst`, each side's rows its stride apart in bytes (tf_glm_copy_rows).
pub fn copyRows(set: glue.Set, stream: cuda.Stream, src: u64, src_stride: usize, dst: u64, dst_stride: usize, rows: usize, bytes: usize) !void {
    var a: cuda.Args = .{};
    a.add(src);
    a.add(@as(c_longlong, @intCast(src_stride)));
    a.add(dst);
    a.add(@as(c_longlong, @intCast(dst_stride)));
    a.add(@as(c_int, @intCast(rows)));
    a.add(@as(c_int, @intCast(bytes)));
    const f = set.copy_rows orelse return error.NoCopyRows;
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast((rows * bytes + 255) / 256) }, .block = .{ .x = 256 } }, stream, &a);
}

/// qmm.group_sums: (m, k) bf16 rows (stride `x_stride`) -> (m, k / 64) fp32 sums of each 64 inputs.
pub fn groupSums(set: glue.Set, stream: cuda.Stream, x: u64, x_stride: usize, xs: u64, m: usize, k: usize) !void {
    var a: cuda.Args = .{};
    inline for (.{ x, xs, @as(i64, @intCast(x_stride)), @as(i64, @intCast(k / 64)) }) |v| a.add(v);
    try cuda.launch.launch(set.glue.group_sums, .{ .grid = .{ .x = @intCast(m), .y = @intCast(cdiv(k / 64, 16)) }, .block = .{ .x = 128 } }, stream, &a);
}

test "K slices as qmm.py picks them" {
    try std.testing.expectEqual(@as(usize, 1), splitK(12576, 4096)); // 197 tiles: past the target at one slice
    try std.testing.expectEqual(@as(usize, 4), splitK(4096, 4096)); // 64 tiles: 2, then 4 (256 past the target)
    try std.testing.expectEqual(@as(usize, 8), splitK(160, 4096));
    try std.testing.expectEqual(@as(usize, 1), splitK(4096, 128)); // two steps: no slice keeps four
    try std.testing.expectEqual(@as(usize, 0), partBytes(200, 4096, 4096));
}
