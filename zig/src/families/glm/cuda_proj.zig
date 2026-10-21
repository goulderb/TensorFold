//! A GLM-5.3-Flash projection as its checkpoint stores it: bf16 through the glue's matmuls, FP8 kinds through qlinear.

const std = @import("std");
const cuda = @import("cuda");
const glue = @import("cuda_glue.zig");
const b16 = @import("cuda_b16.zig");

/// A rank's projection: bf16 rows for the glue, or FP8_PB_WO's block FP8 or MXFP8, packed for the lane as stored.
pub const Proj = union(enum) { bf16: u64, fp8g: cuda.qmmf.Weight, mxfp8: cuda.qmmf.Weight, q: cuda.qlinear.Weight, source: Source };

/// x [m][k] (rows `x_stride` apart) by `w` into out [m][n]: bf16, or fp32 with `wide`; `part` the bf16 path's slices.
pub fn matmul(set: glue.Set, stream: cuda.Stream, w: Proj, x: u64, x_stride: usize, out: u64, part: u64, m: usize, n: usize, k: usize, wide: bool) !void {
    switch (w) {
        .bf16 => |p| return b16.matmul(set, stream, x, x_stride, p, out, part, m, n, k, wide),
        .source => |s| return sourceMatmul(set, stream, x, x_stride, s, out, part, m, n, k, wide),
        .q => |q| {
            if (q.outputs() != n or q.inputs() != k) return error.ShapeMismatch;
            const prompt = m >= b16.rows_from or (set.prompt and q == .affine4);
            if (q == .affine4 and !prompt) try groupSums(set, stream, x, x_stride, part, m, k);
            const in: cuda.qlinear.Rows = .{ .x = x, .ldx = x_stride, .sums = part, .part = part, .native = .{ .ptr = part, .bytes = try q.nativeWorkspaceBytes(m) } };
            const l = set.linear;
            if (prompt) return if (wide) l.promptF32(stream, q, in, out, m) else l.prompt(stream, q, in, out, m);
            return if (wide) l.decodeF32(stream, q, in, out, m) else l.decode(stream, q, in, out, m);
        },
        .fp8g, .mxfp8 => |q| {
            if (q.n != n or q.k != k) return error.ShapeMismatch;
            const lane = set.linear.lane orelse return error.FormatNotLoaded;
            // GB10's lane meets its K slices in a cluster; a GPU that sums them in memory would need partials held here
            if (lane.partBytes(m, q) != 0) return error.SlicePartialsNotHeld;
            const in: cuda.qlinear.Rows = .{ .x = x, .ldx = x_stride };
            const f: cuda.qlinear.Weight = if (w == .fp8g) .{ .fp8g = q } else .{ .mxfp8 = q };
            return if (wide) set.linear.decodeF32(stream, f, in, out, m) else set.linear.decode(stream, f, in, out, m);
        },
    }
}

/// error.SlicePartialsNotHeld when `matmul` would refuse `w` on `lane` at some row count: the loader's check.
pub fn fits(lane: *const cuda.qmmf.Lane, w: cuda.qmmf.Weight) error{SlicePartialsNotHeld}!void {
    // one row stands for all: under qmmf.fused_rows the plan is the same for every row count, and over it none reduces
    if (lane.partBytes(1, w) != 0) return error.SlicePartialsNotHeld;
}

test "a block-FP8 projection is refused before any launch when the call's shape or the loaded kernels do not fit" {
    const q = cuda.fp8.weight(0, 0, 64, 4096);
    const s: cuda.Stream = undefined; // never reached: the refusals come first
    var set: glue.Set = .{ .glue = undefined };
    try std.testing.expectError(error.ShapeMismatch, matmul(set, s, .{ .fp8g = q }, 0, 4096, 0, 0, 1, 128, 4096, false));
    try std.testing.expectError(error.FormatNotLoaded, matmul(set, s, .{ .fp8g = q }, 0, 4096, 0, 0, 1, 64, 4096, true));
    var lane: cuda.qmmf.Lane = undefined;
    lane.clusters = false; // a GPU before sm_90: 64 outputs from 4,096 inputs take K slices summed in memory
    set.linear = .{ .lane = &lane };
    try std.testing.expectError(error.SlicePartialsNotHeld, matmul(set, s, .{ .fp8g = q }, 0, 4096, 0, 0, 1, 64, 4096, true));
    try std.testing.expectError(error.SlicePartialsNotHeld, matmul(set, s, .{ .mxfp8 = cuda.mxfp8.weight(0, 0, 64, 4096) }, 0, 4096, 0, 0, 1, 64, 4096, true));
}

test "the loader's check refuses an FP8 projection exactly when matmul would at some row count" {
    var flat: cuda.qmmf.Lane = undefined;
    flat.clusters = false; // a GPU before sm_90
    var gb10: cuda.qmmf.Lane = undefined;
    gb10.clusters = true;
    // FP4-Local's stored kda q, kv_a and MLA o, a 64-output cut, an lm_head share, and inputs too few for K slices
    const shapes = [_][2]usize{ .{ 8192, 4096 }, .{ 512, 4096 }, .{ 4096, 16384 }, .{ 64, 4096 }, .{ 77440, 4096 }, .{ 64, 512 } };
    for (shapes) |nk| for ([_]*const cuda.qmmf.Lane{ &flat, &gb10 }) |lane| {
        const q = cuda.fp8.weight(0, 0, nk[0], nk[1]);
        var refused = false; // matmul's own test, at decode rows, partial tiles, the fused rows and a prompt chunk
        for ([_]usize{ 1, 4, 16, 17, 64, 255, 256, 2048 }) |m| refused = refused or lane.partBytes(m, q) != 0;
        if (refused) try std.testing.expectError(error.SlicePartialsNotHeld, fits(lane, q)) else try fits(lane, q);
    };
    try std.testing.expectError(error.SlicePartialsNotHeld, fits(&flat, cuda.fp8.weight(0, 0, 8192, 4096)));
    try std.testing.expectError(error.SlicePartialsNotHeld, fits(&flat, cuda.mxfp8.weight(0, 0, 4096, 16384)));
    try fits(&flat, cuda.fp8.weight(0, 0, 77440, 4096));
    try fits(&gb10, cuda.mxfp8.weight(0, 0, 4096, 16384));
}

fn cdiv(a: usize, b: usize) usize {
    return (a + b - 1) / b;
}

pub const Source = struct { weights: u64, jobs: u64, groups: u64, groups_n: u32, n: u32, k: u32 };

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
    try @import("cuda_proj.zig").matmul(set, stream, p, 0, 128, 0, 0, 0, 33, 128, false);
    try std.testing.expectError(error.ShapeMismatch, @import("cuda_proj.zig").matmul(set, stream, p, 0, 128, 0, 0, 1, 32, 128, false));
    try std.testing.expectError(error.SourceDenseGeometry, @import("cuda_proj.zig").matmul(set, stream, p, 0, 127, 0, 0, 1, 33, 128, false));
    for ([_]usize{ 1, 7, 16, 17, 64, 2048 }) |m| for ([_]bool{ false, true }) |wide| {
        try std.testing.expectError(error.SourceDenseNotLoaded, @import("cuda_proj.zig").matmul(set, stream, p, 0, 128, 0, 0, m, 33, 128, wide));
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
        try @import("cuda_proj.zig").matmul(set, stream, p, 1000, stride, 2000, 3000, m, 33, 128, wide);
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

test "short affine Q4 prompts launch prompt arithmetic while decode and other weights retain their paths" {
    const Trace = struct {
        var calls: [4]usize = undefined;
        var len: usize = 0;
        fn launch(f: cuda.abi.Function, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: cuda.abi.Stream, _: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
            calls[len] = @intFromPtr(f.?);
            len += 1;
            return 0;
        }
    };
    var driver: cuda.Driver = .{ .lib = undefined, .api = undefined };
    driver.api.cuLaunchKernel = Trace.launch;
    const fun = struct {
        fn of(d: *const cuda.Driver, id: usize) cuda.Function {
            return .{ .d = d, .handle = @ptrFromInt(id) };
        }
    }.of;
    var kernels: glue.Kernels = undefined;
    kernels.group_sums = fun(&driver, 1);
    kernels.bmm_d32s2 = fun(&driver, 11);
    kernels.bmm_tile1 = fun(&driver, 12);
    const affine: cuda.qlinear.Affine4Kernels = .{
        .group_fn = fun(&driver, 2),
        .group_f32 = fun(&driver, 4),
        .prefill_fn = fun(&driver, 5),
        .prefill_f32 = fun(&driver, 6),
        .gemv_fn = undefined,
        .split_fn = undefined,
        .words_fn = undefined,
        .scales_fn = undefined,
        .gemv_blocks = 64,
        .split = null,
        .pdl = false,
    };
    var lane: cuda.qmmf.Lane = .{ .mod = undefined, .fns = undefined, .f32_fns = undefined, .reduce = undefined, .clusters = false };
    for (&lane.fns, &lane.f32_fns) |*narrow, *wide| {
        @memset(narrow, fun(&driver, 20));
        @memset(wide, fun(&driver, 21));
    }
    const affine_weight: Proj = .{ .q = .{ .affine4 = .{ .w = 100, .s = 200, .b = 300, .n = 12576, .k = 4096, .npad = 12672 } } };
    const native_weight: Proj = .{ .q = .{ .nvfp4 = .{ .mode = .fp4, .codes = 100, .scales = 200, .n = 12576, .k = 4096, .npad = 12672 } } };
    const stream: cuda.Stream = .{ .d = &driver, .handle = null };
    for ([_]usize{ 1, 8, 31, 64, 127, 128 }) |rows| for ([_]bool{ false, true }) |prompt| {
        const set: glue.Set = .{ .glue = &kernels, .linear = .{ .affine4 = &affine, .lane = &lane }, .prompt = prompt };
        for ([_]bool{ false, true }) |wide| {
            Trace.len = 0;
            try @import("cuda_proj.zig").matmul(set, stream, affine_weight, 4096, 4096, 8192, 16384, rows, 12576, 4096, wide);
            const expected: []const usize = if (prompt or rows >= 128)
                (if (wide) &.{6} else &.{5})
            else if (wide)
                &.{ 1, 4 }
            else
                &.{ 1, 2 };
            try std.testing.expectEqualSlices(usize, expected, Trace.calls[0..Trace.len]);
            if (rows < 128) {
                Trace.len = 0;
                try @import("cuda_proj.zig").matmul(set, stream, native_weight, 4096, 4096, 8192, 16384, rows, 12576, 4096, wide);
                try std.testing.expectEqualSlices(usize, if (wide) &.{21} else &.{20}, Trace.calls[0..Trace.len]);
            }
        }
        Trace.len = 0;
        try @import("cuda_proj.zig").matmul(set, stream, .{ .bf16 = 16384 }, 4096, 128, 32768, 65536, rows, 4096, 128, false);
        try std.testing.expectEqualSlices(usize, if (rows >= 128) &.{12} else &.{11}, Trace.calls[0..Trace.len]);
    };
}
