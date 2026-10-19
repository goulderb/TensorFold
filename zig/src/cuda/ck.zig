//! ModelOpt checkpoints in their own math: FP8 x FP8 projections (ck_lane.cu), NVFP4 x NVFP4 experts (ck_experts.cu).
const std = @import("std");
const driver = @import("driver.zig");
const module = @import("module.zig");
const launch_ = @import("launch.zig");
const stream_ = @import("stream.zig");
const kernels = @import("kernels.zig");
const qmmf = @import("qmmf.zig");
const grouped = @import("grouped.zig");

const Driver = driver.Driver;
const Module = module.Module;
const Function = module.Function;
const Stream = stream_.Stream;
const Plan = grouped.Plan;

pub const NativeMode = enum { fp8, nvfp4 };
pub const numerical_identity = "ck-dense-v1:tensor-weight:row-activation:local-k:scale-floor-f32-normal:fp32-partials";
pub const Fp8Matrix = struct { codes: u64, scale: f32, n: u32, k: u32, npad: u32 };
pub const Fp4Matrix = struct { words: u64, scales: u64, scale: f32, n: u32, k: u32, npad: u32 };
pub const Workspace = struct { ptr: u64 = 0, bytes: usize = 0 };

pub const WorkspaceLayout = struct {
    codes: usize = 0,
    blocks: usize,
    scales: usize,
    partials: usize,
    bytes: usize,
};

pub fn shape(n: usize, k: usize) !usize {
    if (n == 0 or k == 0 or n > std.math.maxInt(i32) - 127 or k > std.math.maxInt(i32) or k % 64 != 0) return error.InvalidNativeShape;
    return std.mem.alignForward(usize, n, 128);
}

pub fn workspaceLayout(mode: NativeMode, rows: usize, n: usize, k: usize) !WorkspaceLayout {
    _ = try shape(n, k);
    if (rows == 0 or rows > std.math.maxInt(i32)) return error.InvalidNativeShape;
    const cells = try std.math.mul(usize, rows, k);
    const codes = if (mode == .fp8) cells else cells / 2;
    const blocks = std.mem.alignForward(usize, codes, 256);
    const scales = std.mem.alignForward(usize, try std.math.add(usize, blocks, if (mode == .nvfp4) cells / 16 else 0), 256);
    const partials = std.mem.alignForward(usize, try std.math.add(usize, scales, try std.math.mul(usize, rows, 4)), 256);
    const sk = if (mode == .fp8) qmmf.splitK(n, k) else 1;
    const part_bytes = if (sk > 1) try std.math.mul(usize, try std.math.mul(usize, rows, n), sk * 4) else 0;
    return .{ .blocks = blocks, .scales = scales, .partials = partials, .bytes = try std.math.add(usize, partials, part_bytes) };
}

pub fn workspaceBytes(mode: NativeMode, rows: usize, n: usize, k: usize) !usize {
    return (try workspaceLayout(mode, rows, n, k)).bytes;
}

pub fn tensorScale(amax: f32, mode: NativeMode) !f32 {
    if (!std.math.isFinite(amax) or amax < 0) return error.InvalidNativeScale;
    if (amax == 0) return 1;
    return @max(amax / @as(f32, if (mode == .fp8) 448 else 2688), std.math.floatMin(f32));
}

fn validateDense(mode: NativeMode, rows: usize, n: usize, k: usize, npad: usize, scale: f32, x: u64, ldx: usize, out: u64, work: Workspace) !WorkspaceLayout {
    if (npad != try shape(n, k)) return error.InvalidNativeShape;
    if (!std.math.isFinite(scale) or scale < std.math.floatMin(f32)) return error.InvalidNativeScale;
    if (x == 0 or out == 0 or x % 16 != 0 or ldx < k or ldx > std.math.maxInt(i32) or ldx % 8 != 0 or out % 4 != 0) return error.InvalidNativeStride;
    const layout = try workspaceLayout(mode, rows, n, k);
    if (work.ptr == 0 or work.ptr % 256 != 0 or work.bytes < layout.bytes) return error.NativeWorkspaceTooSmall;
    return layout;
}

/// Token rows an expert call takes as decode-sized (the _few kernels): a verify window's, up to 32.
pub const few_tokens = 32;

/// Token rows from which a pass's expert calls take the prompt kernels, over plans of `prompt_tile` pairs an item.
pub const prompt_tokens = 128;

/// Pairs a prompt plan item holds: ck_experts.cu's prompt tile rows (decode plans hold 16).
pub const prompt_tile = 128;

/// One fp32 factor: the checkpoint's runtimes multiply the input scale by the weight's in fp32.
pub fn alpha(act: f32, scale: f32) f32 {
    return act * scale;
}

/// 1 / the input scale in fp32, as the quantizers take it.
pub fn inverse(act: f32) f32 {
    return 1.0 / act;
}

/// Bytes an NVFP4 row quantization of `rows` rows of `k` inputs takes: codes, then an e4m3 scale a 16.
pub fn rowsBytes(rows: usize, k: usize) usize {
    return rows * k / 2 + rows * k / 16;
}

/// One layer's experts in the FP4 mma's layout: words and scale tiles a direction, alpha a direction and expert.
pub const Layer = struct {
    up: u64, // [E][width/64][dims/64][8][32][2] u32
    up_s: u64, // [E][width/64][dims/64][64][4] e4m3
    down: u64, // [E][dims/64][width/64][8][32][2]
    down_s: u64,
    up_alpha: u64, // fp32 [E]: up's input scale times each expert's weight scale
    down_alpha: u64,
    up_act: f32, // up's input scale (every expert's the same)
    down_act: f32,
    experts: usize,
    width: usize,
    dims: usize,
};

pub const Ck = struct {
    lane_mod: Module,
    experts_mod: Module,
    lane: [10]Function, // row tiles 16, 32, 64, 64 skipping, 128 skipping (128 x 128 clustered); plain then clustered
    reduce: Function,
    quant8_fn: Function,
    quant4_fn: Function,
    pack4_fn: Function,
    pack4_scales_fn: Function,
    up_fns: [2]Function, // calls under prompt_tokens, then decode-sized calls (ck_experts.cu's _few: 32 columns a warp)
    down_fns: [2]Function,
    resident: [2][2]usize, // blocks the up and down kernels fill, by up_fns' order: per SM times SMs
    prompt_fns: [3]Function, // over prompt plans: relu^2 up to NVFP4 rows, fp32 down, share-summed down (256 threads)
    prompt_resident: [3]usize,
    clusters: bool,
    dense8: [2][8]Function,
    dense8_reduce: Function,
    dense4: [2]Function,
    row_scales_fn: Function,
    quant8_rows_fn: Function,
    quant8_weight_fn: Function,
    quant4_rows_fn: Function,

    pub fn load(d: *const Driver, capability_major: u32, sms: usize) !Ck {
        if (!kernels.available) return error.BuiltWithoutKernels;
        var lane_mod = try Module.load(d, kernels.ck_lane);
        errdefer lane_mod.unload();
        var experts_mod = try Module.load(d, kernels.ck_experts);
        errdefer experts_mod.unload();
        var c: Ck = undefined;
        c.lane_mod = lane_mod;
        c.experts_mod = experts_mod;
        for ([_][:0]const u8{ "tf_ck8_16", "tf_ck8_16c", "tf_ck8_32", "tf_ck8_32c", "tf_ck8_64", "tf_ck8_64c", "tf_ck8_64s", "tf_ck8_64sc", "tf_ck8_128", "tf_ck8_128c" }, &c.lane) |name, *f| f.* = try lane_mod.function(name);
        for (c.lane[8..]) |f| try f.allowDynamicShared(65536);
        c.reduce = try lane_mod.function("tf_ck8_reduce");
        c.quant8_fn = try lane_mod.function("tf_ck_quant8");
        c.quant4_fn = try experts_mod.function("tf_ck_quant4");
        c.pack4_fn = try experts_mod.function("tf_ck_pack4");
        c.pack4_scales_fn = try experts_mod.function("tf_ck_pack4_scales");
        for ([_][2][:0]const u8{ .{ "tf_ck4_up_relu2", "tf_ck4_down_f32" }, .{ "tf_ck4_up_relu2_few", "tf_ck4_down_f32_few" } }, 0..) |names, i| {
            c.up_fns[i] = try experts_mod.function(names[0]);
            c.down_fns[i] = try experts_mod.function(names[1]);
            c.resident[i] = .{ @max(1, try c.up_fns[i].occupancy(128, 0)) * sms, @max(1, try c.down_fns[i].occupancy(128, 0)) * sms };
        }
        for ([_][:0]const u8{ "tf_ck4_up_nvfp4_prompt", "tf_ck4_down_f32_prompt", "tf_ck4_down_sum_prompt" }, &c.prompt_fns, &c.prompt_resident) |name, *f, *n| {
            f.* = try experts_mod.function(name);
            n.* = @max(1, try f.occupancy(256, 0)) * sms;
        }
        c.clusters = capability_major >= 9;
        inline for (.{ "bf16", "f32" }, 0..) |ep, j| {
            inline for (.{ "16", "16c", "32", "32c", "64", "64c", "64s", "64sc" }, 0..) |tile, i| {
                c.dense8[j][i] = try lane_mod.function("tf_ck8_dense" ++ tile ++ "_" ++ ep);
                try c.dense8[j][i].allowDynamicShared(32768);
            }
            c.dense4[j] = try experts_mod.function("tf_ck4_dense_" ++ ep);
        }
        c.dense8_reduce = try lane_mod.function("tf_ck8_dense_reduce");
        c.row_scales_fn = try lane_mod.function("tf_ck_row_scales_or_nan");
        c.quant8_rows_fn = try lane_mod.function("tf_ck_quant8_rows");
        c.quant8_weight_fn = try lane_mod.function("tf_ck_quant8_weight");
        c.quant4_rows_fn = try experts_mod.function("tf_ck_quant4_rows");
        return c;
    }

    pub fn unload(c: *Ck) void {
        c.experts_mod.unload();
        c.lane_mod.unload();
    }

    fn go(f: Function, s: Stream, grid: [3]usize, block: u32, shared: u32, a: *launch_.Args) !void {
        try launch_.launch(f, .{ .grid = .{ .x = @intCast(grid[0]), .y = @intCast(grid[1]), .z = @intCast(grid[2]) }, .block = .{ .x = block }, .shared = shared }, s, a);
    }

    pub fn quantizeWeight8(c: *const Ck, s: Stream, x: u64, n: usize, k: usize, scale: f32, out: u64) !void {
        _ = try shape(n, k);
        if (!std.math.isFinite(scale) or scale < std.math.floatMin(f32)) return error.InvalidNativeScale;
        var a: launch_.Args = .{};
        a.add(x);
        a.add(@as(i32, @intCast(n)));
        a.add(@as(i32, @intCast(k)));
        a.add(inverse(scale));
        a.add(out);
        try go(c.quant8_weight_fn, s, .{ (n * (k / 16) + 127) / 128, 1, 1 }, 128, 0, &a);
    }

    pub fn quantizeWeight4(c: *const Ck, s: Stream, x: u64, n: usize, k: usize, scale: f32, codes: u64, scales: u64) !void {
        _ = try shape(n, k);
        if (!std.math.isFinite(scale) or scale < std.math.floatMin(f32)) return error.InvalidNativeScale;
        var a: launch_.Args = .{};
        a.add(x);
        for ([_]usize{ k, n, k }) |v| a.add(@as(i32, @intCast(v)));
        a.add(inverse(scale));
        a.add(codes);
        a.add(scales);
        try go(c.quant4_fn, s, .{ (n * (k / 16) + 127) / 128, 1, 1 }, 128, 0, &a);
    }

    fn quantizeRows(c: *const Ck, s: Stream, mode: NativeMode, x: u64, ldx: usize, rows: usize, k: usize, work: Workspace, l: WorkspaceLayout) !void {
        var a: launch_.Args = .{};
        a.add(x);
        a.add(@as(i32, @intCast(ldx)));
        a.add(@as(i32, @intCast(k)));
        a.add(@as(f32, if (mode == .fp8) 448 else 2688));
        a.add(work.ptr + l.scales);
        try go(c.row_scales_fn, s, .{ rows, 1, 1 }, 256, 0, &a);
        var q: launch_.Args = .{};
        q.add(x);
        q.add(@as(i32, @intCast(ldx)));
        q.add(@as(i32, @intCast(rows)));
        q.add(@as(i32, @intCast(k)));
        q.add(work.ptr + l.scales);
        q.add(work.ptr);
        if (mode == .nvfp4) q.add(work.ptr + l.blocks);
        try go(if (mode == .fp8) c.quant8_rows_fn else c.quant4_rows_fn, s, .{ (rows * (k / 16) + 127) / 128, 1, 1 }, 128, 0, &q);
    }

    pub fn matmul8(c: *const Ck, s: Stream, w: Fp8Matrix, x: u64, ldx: usize, rows: usize, work: Workspace, out: u64, kind: qmmf.Out) !void {
        const l = try validateDense(.fp8, rows, w.n, w.k, w.npad, w.scale, x, ldx, out, work);
        if (w.codes == 0 or w.codes % 16 != 0) return error.InvalidNativeWeight;
        try c.quantizeRows(s, .fp8, x, ldx, rows, w.k, work, l);
        const sk = qmmf.splitK(w.n, w.k);
        const cluster = sk > 1 and c.clusters;
        const bm: usize = qmmf.bucket(rows);
        const ti: usize = if (bm == 16) 0 else if (bm == 32) 1 else if (rows % 64 == 0) 2 else 3;
        const f = c.dense8[@intFromBool(kind == .f32)][ti * 2 + @intFromBool(cluster)];
        const rows_t = (rows + bm - 1) / bm;
        const band = @max(1, @min(rows_t, (12 << 20) / (bm * w.k)));
        var a: launch_.Args = .{};
        a.add(work.ptr);
        a.add(w.codes);
        a.add(w.scale);
        a.add(out);
        a.add(work.ptr + l.partials);
        for ([_]usize{ rows, w.n, w.k, sk, band }) |v| a.add(@as(i32, @intCast(v)));
        a.add(work.ptr + l.scales);
        try launch_.launch(f, .{
            .grid = .{ .x = @intCast(rows_t * ((w.n + 63) / 64)), .z = sk },
            .block = .{ .x = 128 },
            .shared = @intCast(4 * (bm * 64 + 4096)),
            .cluster = if (cluster) .{ .z = sk } else null,
        }, s, &a);
        if (sk > 1 and !cluster) {
            const total = rows * w.n;
            var r: launch_.Args = .{};
            r.add(work.ptr + l.partials);
            r.add(out);
            r.add(@as(i64, @intCast(total)));
            r.add(@as(i32, @intCast(w.n)));
            r.add(@as(i32, @intCast(sk)));
            r.add(w.scale);
            r.add(work.ptr + l.scales);
            r.add(@as(i32, @intFromBool(kind == .f32)));
            try go(c.dense8_reduce, s, .{ (total + 255) / 256, 1, 1 }, 256, 0, &r);
        }
    }

    pub fn matmul4(c: *const Ck, s: Stream, w: Fp4Matrix, x: u64, ldx: usize, rows: usize, work: Workspace, out: u64, kind: qmmf.Out) !void {
        const l = try validateDense(.nvfp4, rows, w.n, w.k, w.npad, w.scale, x, ldx, out, work);
        if (w.words == 0 or w.words % 16 != 0 or w.scales == 0 or w.scales % 4 != 0) return error.InvalidNativeWeight;
        try c.quantizeRows(s, .nvfp4, x, ldx, rows, w.k, work, l);
        var a: launch_.Args = .{};
        for ([_]u64{ work.ptr, work.ptr + l.blocks, w.words, w.scales, work.ptr + l.scales }) |p| a.add(p);
        a.add(w.scale);
        for ([_]usize{ rows, w.k, w.n }) |v| a.add(@as(i32, @intCast(v)));
        a.add(out);
        const units = ((rows + 15) / 16) * ((w.n + 63) / 64);
        try go(c.dense4[@intFromBool(kind == .f32)], s, .{ @max(1, @min((units + 3) / 4, c.resident[1])), 1, 1 }, 128, 0, &a);
    }

    /// bf16 rows [m, k] (`ldx` apart) to e4m3(x / act) at `out` [m, k], in the FP8 weights' fragment order.
    pub fn quant8(c: *const Ck, s: Stream, x: u64, ldx: usize, m: usize, k: usize, act: f32, out: u64) !void {
        var a: launch_.Args = .{};
        a.add(x);
        a.add(@as(i32, @intCast(ldx)));
        a.add(@as(i32, @intCast(m)));
        a.add(@as(i32, @intCast(k)));
        a.add(inverse(act));
        a.add(out);
        try go(c.quant8_fn, s, .{ (m * (k / 16) + 127) / 128, 1, 1 }, 128, 0, &a);
    }

    /// e4m3 rows [m, k] at `x8` times the `.fp8` weight `w` under input scale `act`, bf16 at `y` [m, n].
    pub fn lane8(c: *const Ck, s: Stream, x8: u64, m: usize, w: qmmf.Weight, act: f32, y: u64, part: u64) !void {
        const ti: usize = if (m >= prompt_tokens) 4 else if (m <= 16) 0 else if (m <= 32) 1 else if (m % 64 == 0) 2 else 3;
        return c.lane8Tile(s, ti, x8, m, w, act, y, part);
    }

    /// lane8 on row tile `ti` of `lane`'s order: 16, 32, 64 (whole tiles of m), 64 skipping, 128 skipping.
    pub fn lane8Tile(c: *const Ck, s: Stream, ti: usize, x8: u64, m: usize, w: qmmf.Weight, act: f32, y: u64, part: u64) !void {
        if (w.mode != .fp8) return error.FormatMismatch;
        const sk = qmmf.splitK(w.n, w.k);
        const cluster = sk > 1 and sk <= 8 and c.clusters;
        if (sk > 1 and !cluster and part == 0) return error.NeedsPartials;
        const prompt = ti == 4;
        const bm: usize = ([_]usize{ 16, 32, 64, 64, 128 })[ti];
        const bn: usize = if (prompt and cluster) 128 else 64;
        const f = c.lane[ti * 2 + @intFromBool(cluster)];
        const stage = (bm * 64 + bn * 64 + 127) / 128 * 128;
        const smem: u32 = @intCast(@max(4 * stage, if (cluster) bm * bn * 4 else 0));
        const rows_t = (m + bm - 1) / bm;
        const group = @max(1, @min(rows_t, (12 << 20) / (bm * w.k)));
        const al = alpha(act, w.scale);
        var a: launch_.Args = .{};
        a.add(x8);
        a.add(w.codes);
        a.add(al);
        a.add(y);
        a.add(part);
        for ([_]usize{ m, w.n, w.k, sk, group }) |v| a.add(@as(i32, @intCast(v)));
        try launch_.launch(f, .{
            .grid = .{ .x = @intCast(rows_t * ((w.n + bn - 1) / bn)), .y = 1, .z = sk },
            .block = .{ .x = if (prompt) 256 else 128, .y = 1, .z = 1 },
            .shared = smem,
            .cluster = if (cluster) .{ .x = 1, .y = 1, .z = sk } else null,
        }, s, &a);
        if (sk > 1 and !cluster) {
            const total: usize = m * w.n;
            var r: launch_.Args = .{};
            r.add(part);
            r.add(y);
            r.add(@as(i64, @intCast(total)));
            r.add(@as(i32, @intCast(sk)));
            r.add(al);
            try go(c.reduce, s, .{ (total + 255) / 256, 1, 1 }, 256, 0, &r);
        }
    }

    /// bf16 rows [m, k] (`ldx` apart) to NVFP4 under input scale `act`: codes [m][k/2] then scales [m][k/16] at `out`.
    pub fn quant4(c: *const Ck, s: Stream, x: u64, ldx: usize, m: usize, k: usize, act: f32, out: u64) !void {
        var a: launch_.Args = .{};
        a.add(x);
        a.add(@as(i32, @intCast(ldx)));
        a.add(@as(i32, @intCast(m)));
        a.add(@as(i32, @intCast(k)));
        a.add(inverse(act));
        a.add(out);
        a.add(out + m * k / 2);
        try go(c.quant4_fn, s, .{ (m * (k / 16) + 127) / 128, 1, 1 }, 128, 0, &a);
    }

    /// `count` experts' NVFP4 bytes [count][n][ks/2] and scales [count][n][ks/16], inputs k0 .. k0 + k, packed.
    pub fn pack(c: *const Ck, s: Stream, codes: u64, scales: u64, count: usize, n: usize, ks: usize, k0: usize, k: usize, words: u64, tiles: u64) !void {
        if (n % 64 != 0 or k % 64 != 0 or k0 % 64 != 0 or k0 + k > ks) return error.UnexpectedTensor;
        var a: launch_.Args = .{};
        a.add(codes);
        for ([_]usize{ n, ks, k0, k }) |v| a.add(@as(i32, @intCast(v)));
        a.add(words);
        try go(c.pack4_fn, s, .{ n / 64, k / 64, count }, 256, 0, &a);
        const total: usize = count * n * (k / 16);
        var b: launch_.Args = .{};
        b.add(scales);
        for ([_]usize{ n, ks, k0, k }) |v| b.add(@as(i32, @intCast(v)));
        b.add(tiles);
        b.add(@as(i64, @intCast(total)));
        try go(c.pack4_scales_fn, s, .{ (total + 255) / 256, 1, 1 }, 256, 0, &b);
    }

    /// Bytes a layer's words and scale tiles take a direction: n outputs from k inputs for `count` experts.
    pub fn packedBytes(count: usize, n: usize, k: usize) [2]usize {
        return .{ count * n * k / 2, count * n * (k / 16) };
    }

    /// Columns of `n` a warp of the up and down kernels takes, by up_fns' order.
    pub const warp_columns = [2]usize{ 64, 32 };

    /// relu^2 up over a plan: NVFP4 rows (quant4's, at `rows`, `k` wide; row = pair / slots) to bf16 [pairs][width].
    pub fn up(c: *const Ck, s: Stream, l: Layer, rows: u64, nrows: usize, slots: usize, p: Plan, out: u64, items: usize) !void {
        const few = @intFromBool(nrows <= few_tokens);
        return run(s, c.up_fns[few], c.resident[few][0], warp_columns[few], rows, nrows, slots, l.up, l.up_s, l.up_alpha, l.dims, l.width, p, out, items);
    }

    /// down over a plan: NVFP4 pair rows (quant4's, at `rows`) of `tokens` token rows to fp32 [pairs][dims].
    pub fn down(c: *const Ck, s: Stream, l: Layer, rows: u64, nrows: usize, tokens: usize, p: Plan, out: u64, items: usize) !void {
        const few = @intFromBool(tokens <= few_tokens);
        return run(s, c.down_fns[few], c.resident[few][1], warp_columns[few], rows, nrows, 0, l.down, l.down_s, l.down_alpha, l.width, l.dims, p, out, items);
    }

    /// `up` over a prompt plan to the NVFP4 rows quant4 makes of its bf16 rows at down's input scale (rowsBytes).
    pub fn promptUp(c: *const Ck, s: Stream, l: Layer, rows: u64, nrows: usize, slots: usize, p: Plan, pairs: usize, out: u64, items: usize) !void {
        var a = args(rows, nrows, slots, l.up, l.up_s, l.up_alpha, l.dims, l.width, p, out);
        a.add(out + pairs * l.width / 2);
        a.add(inverse(l.down_act));
        try runPrompt(s, c.prompt_fns[0], c.prompt_resident[0], l.width, items, &a);
    }

    /// `down` over a prompt plan.
    pub fn promptDown(c: *const Ck, s: Stream, l: Layer, rows: u64, nrows: usize, p: Plan, out: u64, items: usize) !void {
        var a = args(rows, nrows, 0, l.down, l.down_s, l.down_alpha, l.width, l.dims, p, out);
        try runPrompt(s, c.prompt_fns[1], c.prompt_resident[1], l.dims, items, &a);
    }

    /// `down` over a prompt plan, pair p + s on expert s * experts / shares + e: shares added in order, bf16 a pair p.
    pub fn promptSum(c: *const Ck, s: Stream, l: Layer, rows: u64, nrows: usize, p: Plan, out: u64, items: usize, shares: usize) !void {
        var a = args(rows, nrows, 0, l.down, l.down_s, l.down_alpha, l.width, l.dims, p, out);
        a.add(@as(i32, @intCast(shares)));
        a.add(@as(i32, @intCast(l.experts / shares)));
        // the first share's items alone: pair p + s mirrors pair p, so every share's experts plan the same items
        try runPrompt(s, c.prompt_fns[2], c.prompt_resident[2], l.dims, items / shares, &a);
    }

    /// One grouped launch of `f` over a plan: NVFP4 rows at `rows` (`nrows` of `k`), `n` columns of `columns` a warp.
    pub fn run(s: Stream, f: Function, resident: usize, columns: usize, rows: u64, nrows: usize, slots: usize, w: u64, ws: u64, al: u64, k: usize, n: usize, p: Plan, out: u64, items: usize) !void {
        const nb = n / columns;
        const grid = @min((items * nb + 3) / 4, resident);
        if (grid < 1) return;
        var a = args(rows, nrows, slots, w, ws, al, k, n, p, out);
        try go(f, s, .{ grid, 1, 1 }, 128, 0, &a);
    }

    /// A prompt kernel's launch: a block a (plan item, 128 columns of `n`).
    fn runPrompt(s: Stream, f: Function, resident: usize, n: usize, items: usize, a: *launch_.Args) !void {
        const grid = @min(items * ((n / 64 + 1) / 2), resident);
        if (grid < 1) return;
        try go(f, s, .{ grid, 1, 1 }, 256, 0, a);
    }

    /// The expert kernels' leading arguments: input rows and their scales, the weights, the shape, the plan, `out`.
    fn args(rows: u64, nrows: usize, slots: usize, w: u64, ws: u64, al: u64, k: usize, n: usize, p: Plan, out: u64) launch_.Args {
        var a: launch_.Args = .{};
        a.add(rows);
        a.add(rows + nrows * k / 2);
        a.add(@as(i32, @intCast(slots)));
        a.add(w);
        a.add(ws);
        a.add(al);
        a.add(@as(i32, @intCast(k)));
        a.add(@as(i32, @intCast(n)));
        for ([_]u64{ p.items, p.counts, p.members, out }) |v| a.add(v);
        return a;
    }
};

test "the factors and the row bytes" {
    try std.testing.expectEqual(@as(f32, 0.5) * @as(f32, 0.25), alpha(0.5, 0.25));
    try std.testing.expectEqual(@as(usize, 4 * 512 + 4 * 64), rowsBytes(4, 1024));
    try std.testing.expectEqual([2]usize{ 2 * 64 * 1024 / 2, 2 * 64 * 64 }, Ck.packedBytes(2, 64, 1024));
}

test "native scales are finite for zero and subnormal sources and refuse nonfinite tensors" {
    const t = std.testing;
    try t.expectEqual(@as(f32, 1), try tensorScale(0, .fp8));
    try t.expectEqual(@as(f32, 1), try tensorScale(0, .nvfp4));
    try t.expectEqual(@as(f32, 0.5), try tensorScale(224, .fp8));
    try t.expectEqual(@as(f32, 0.5), try tensorScale(1344, .nvfp4));
    try t.expectEqual(std.math.floatMin(f32), try tensorScale(@as(f32, @bitCast(@as(u32, 1))), .fp8));
    try t.expectError(error.InvalidNativeScale, tensorScale(std.math.inf(f32), .fp8));
    try t.expectError(error.InvalidNativeScale, tensorScale(std.math.nan(f32), .nvfp4));
}

test "native workspaces contain separate codes scales and split partials for GLM tails" {
    for ([_]usize{ 160, 12576 }) |n| {
        try std.testing.expectEqual(std.mem.alignForward(usize, n, 128), try shape(n, 4096));
        inline for (.{ NativeMode.fp8, NativeMode.nvfp4 }) |mode| {
            const l = try workspaceLayout(mode, 33, n, 4096);
            try std.testing.expect(l.blocks >= 33 * 4096 / (if (mode == .fp8) @as(usize, 1) else 2));
            try std.testing.expect(l.scales >= l.blocks + (if (mode == .nvfp4) @as(usize, 33 * 4096 / 16) else 0));
            try std.testing.expect(l.partials >= l.scales + 33 * 4);
            try std.testing.expect(l.bytes >= l.partials);
        }
    }
    try std.testing.expectError(error.InvalidNativeShape, workspaceBytes(.fp8, 0, 160, 4096));
    try std.testing.expectError(error.InvalidNativeShape, shape(160, 127));
}

test "native dense refuses invalid buffers before launching and preserves the strided launch ABI" {
    const abi = @import("abi.zig");
    const Fake = struct {
        var calls: usize = 0;
        var seen: [3]usize = undefined;
        var stride: i32 = 0;
        var dims: [3]i32 = undefined;
        fn value(comptime T: type, a: [*]?*anyopaque, i: usize) T {
            return @as(*const T, @ptrCast(@alignCast(a[i].?))).*;
        }
        fn launch(f: abi.Function, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: abi.Stream, args: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) abi.Result {
            const id = @intFromPtr(f.?);
            seen[calls] = id;
            calls += 1;
            if (id == 1) stride = value(i32, args.?, 1);
            if (id == 3) dims = .{ value(i32, args.?, 5), value(i32, args.?, 6), value(i32, args.?, 7) };
            return 0;
        }
    };
    var d: Driver = .{ .lib = undefined, .api = undefined };
    d.api.cuLaunchKernel = Fake.launch;
    const s: Stream = .{ .d = &d, .handle = null };
    var c: Ck = undefined;
    c.clusters = false;
    c.row_scales_fn = .{ .d = &d, .handle = @ptrFromInt(1) };
    c.quant8_rows_fn = .{ .d = &d, .handle = @ptrFromInt(2) };
    c.dense8[1][0] = .{ .d = &d, .handle = @ptrFromInt(3) };
    const w: Fp8Matrix = .{ .codes = 256, .scale = 0.5, .n = 160, .k = 128, .npad = 256 };
    const bytes = try workspaceBytes(.fp8, 3, 160, 128);
    Fake.calls = 0;
    try std.testing.expectError(error.NativeWorkspaceTooSmall, c.matmul8(s, w, 4096, 136, 3, .{ .ptr = 12288, .bytes = bytes - 1 }, 8192, .f32));
    try std.testing.expectError(error.InvalidNativeStride, c.matmul8(s, w, 4096, 129, 3, .{ .ptr = 12288, .bytes = bytes }, 8192, .f32));
    try std.testing.expectError(error.InvalidNativeStride, c.matmul8(s, w, 4096, @as(usize, 1) << 31, 3, .{ .ptr = 12288, .bytes = bytes }, 8192, .f32));
    try std.testing.expectEqual(@as(usize, 0), Fake.calls);
    try c.matmul8(s, w, 4096, 136, 3, .{ .ptr = 12288, .bytes = bytes }, 8192, .f32);
    try std.testing.expectEqualSlices(usize, &.{ 1, 2, 3 }, &Fake.seen);
    try std.testing.expectEqual(@as(i32, 136), Fake.stride);
    try std.testing.expectEqual([3]i32{ 3, 160, 128 }, Fake.dims);
}
