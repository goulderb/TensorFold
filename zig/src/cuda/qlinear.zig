//! Quantized projections behind one call: a weight view per format, with decode rows and prompt rows chosen per format.
const std = @import("std");
const driver = @import("driver.zig");
const module = @import("module.zig");
const launch_ = @import("launch.zig");
const memory = @import("memory.zig");
const stream_ = @import("stream.zig");
const qmmf = @import("qmmf.zig");
const nvfp4 = @import("nvfp4.zig");
const exl3_linear = @import("exl3_linear.zig");
const ck = @import("ck.zig");
const exl3 = @import("exl3.zig");
const ggml = @import("ggml.zig");

const Driver = driver.Driver;
const Module = module.Module;
const Function = module.Function;
const Stream = stream_.Stream;

/// MLX affine 4-bit tiled for qmm: packed words, (kg, npad) scales and biases, n outputs from k inputs.
pub const Affine4 = struct {
    w: u64,
    s: u64,
    b: u64,
    n: usize,
    k: usize,
    npad: usize,

    /// The tiled sizes for n outputs from k inputs: rows padded to 128, then the words', scales' and biases' bytes.
    pub fn layout(n: usize, k: usize) struct { npad: usize, words: usize, scales: usize } {
        const npad = (n + 127) / 128 * 128;
        return .{ .npad = npad, .words = npad * k / 2, .scales = k / 64 * npad * 2 };
    }
};

/// MLX affine codes tiled for affine8.cu at `bits` a weight (2, 3, 5, 6 or 8), with Affine4's padding and tables.
pub const AffineN = struct {
    w: u64,
    s: u64,
    b: u64,
    n: usize,
    k: usize,
    npad: usize,
    bits: u8,

    pub fn layout(n: usize, k: usize, bits: u8) struct { npad: usize, words: usize, scales: usize } {
        const npad = (n + 127) / 128 * 128;
        return .{ .npad = npad, .words = npad * k * bits / 8, .scales = k / 64 * npad * 2 };
    }
};

/// The width an MLX code of `bits` is served at: its own, but 7 on the 8-bit tiles.
pub fn servedBits(bits: u8) u8 {
    return if (bits == 7) 8 else bits;
}

/// The width the embed kernels read a row of `bits` codes at: 4 or 8.
pub fn rowBits(bits: u8) u8 {
    return if (bits <= 4) 4 else 8;
}

/// One projection's weights in any format this module serves; every one but affine4 is a lane matmul mode.
pub const Weight = union(enum) {
    affine4: Affine4,
    fp8g: qmmf.Weight,
    mxfp8: qmmf.Weight, // e4m3 with UE8M0 scale bytes per 32 inputs
    nvfp4: qmmf.Weight,
    fp8: qmmf.Weight, // e4m3 with one fp32 scale (ModelOpt FP8)
    bf16: qmmf.Weight, // unquantized: the checkpoint's bf16 values
    exl3: exl3_linear.Matrix, // trellis tiles as stored, with suh and svh (the GLM family's path)
    exl3_dense: exl3.Weight, // upstream's exl3/linear.cu weights (exl3_upload; no family loads them yet)
    ck8: ck.Fp8Matrix,
    ck4: ck.Fp4Matrix,
    q8_0: ggml.Weight, // ggml Q8_0 packed by ggml.Kernels.pack
    iq2xxs: ggml.Weight, // ggml IQ2_XXS packed by ggml.Kernels.pack

    pub fn outputs(w: Weight) usize {
        return switch (w) {
            .exl3 => |q| q.n,
            .affine4 => |q| q.n,
            .fp8g, .mxfp8, .nvfp4, .fp8, .bf16 => |q| q.n,
            .exl3_dense => |q| q.n,
            .ck8 => |q| q.n,
            .ck4 => |q| q.n,
            .q8_0, .iq2xxs => |q| q.n,
        };
    }

    pub fn inputs(w: Weight) usize {
        return switch (w) {
            .exl3 => |q| q.k,
            .affine4 => |q| q.k,
            .fp8g, .mxfp8, .nvfp4, .fp8, .bf16 => |q| q.k,
            .exl3_dense => |q| q.k,
            .ck8 => |q| q.k,
            .ck4 => |q| q.k,
            .q8_0, .iq2xxs => |q| q.k,
        };
    }

    pub fn nativeWorkspaceBytes(w: Weight, rows: usize) !usize {
        return switch (w) {
            .ck8 => |q| ck.workspaceBytes(.fp8, rows, q.n, q.k),
            .ck4 => |q| ck.workspaceBytes(.nvfp4, rows, q.n, q.k),
            else => 0,
        };
    }
};

/// Rows in: bf16 `x` rows `ldx` apart (0: packed), their 64-group sums, and the lane's slice partials (partBytes).
pub const Rows = struct { x: u64, ldx: usize = 0, sums: u64 = 0, part: u64 = 0, native: ck.Workspace = .{} };

pub const AffinePromptPolicy = enum(u8) { stock, glm_production };
pub const AffinePromptTile = enum(u8) { tile0 = 0, tile3 = 3, tile9 = 9 };

pub fn affinePromptTile(policy: AffinePromptPolicy, rows: usize, n: usize, k: usize) AffinePromptTile {
    if (policy == .stock) return .tile0;
    return if (k >= 1024 and ((rows + 127) / 128) * ((n + 127) / 128) >= 96) .tile9 else .tile3;
}

pub const AffinePromptGeometry = struct { bm: usize, bn: usize, threads: u32, shared: u32 };

pub fn affinePromptGeometry(tile: AffinePromptTile) AffinePromptGeometry {
    return switch (tile) {
        .tile0 => .{ .bm = 128, .bn = 128, .threads = 256, .shared = 62976 },
        .tile3 => .{ .bm = 64, .bn = 64, .threads = 128, .shared = 41984 },
        .tile9 => .{ .bm = 128, .bn = 128, .threads = 128, .shared = 41984 },
    };
}

/// The kernels of every format a family loaded; a weight whose format is not loaded is refused.
pub const Linear = struct {
    affine4: ?*const Affine4Kernels = null,
    affine_prompt_policy: AffinePromptPolicy = .stock,
    lane: ?*const qmmf.Lane = null, // decode rows of every lane format; fp8g and bf16 prompt rows too
    nvfp4_prompt: ?*const nvfp4.Prompt = null, // the prompt GEMM: nvfp4 and fp8 prompt rows (linear.py's prefill)
    exl3: ?*const exl3_linear.Kernels = null, // decode rows 16 a window; prompt rows on the tensor units
    exl3_dense: ?*exl3.Dense = null, // upstream's exl3/linear.cu weights (exl3_upload)
    ck: ?*const ck.Ck = null, // the checkpoint's own math: an fp8 weight with an input scale takes e4m3 rows
    ck_rows: u64 = 0, // its e4m3 rows (at least rows * k bytes), and slice partials where no cluster meets them
    ck_part: u64 = 0,
    ggml: ?*const ggml.Kernels = null, // q8_0 and iq2xxs rows of both kinds

    /// e4m3 rows of x times an fp8 weight under its input scale, bf16 out: decode and prompt rows alike (row-exact).
    fn checkpoint(l: Linear, s: Stream, c: *const ck.Ck, q: qmmf.Weight, in: Rows, out: u64, rows: usize) !void {
        if (l.ck_rows == 0) return error.NoCheckpointRows;
        try c.quant8(s, in.x, if (in.ldx == 0) q.k else in.ldx, rows, q.k, q.act, l.ck_rows);
        return c.lane8(s, l.ck_rows, rows, q, q.act, out, if (l.ck_part != 0) l.ck_part else in.part);
    }

    /// Decode rows: out (rows, n) bf16, each row's bits the same at every row count the format takes.
    pub fn decode(l: Linear, s: Stream, w: Weight, in: Rows, out: u64, rows: usize) !void {
        switch (w) {
            .affine4 => |q| {
                const a = l.affine4 orelse return error.FormatNotLoaded;
                if (rows > 16 or (in.ldx != 0 and in.ldx != q.k)) return a.clusterAs(s, in.x, in.ldx, in.sums, q, out, rows, splitK(q.n, q.k), false);
                return a.decode(s, in.x, in.sums, q, out, rows);
            },
            .fp8g => |q| return l.onLane(s, q, .fp8g, in, out, rows),
            .mxfp8 => |q| return l.onLane(s, q, .mxfp8, in, out, rows),
            .nvfp4 => |q| return l.onLane(s, q, .fp4, in, out, rows),
            .fp8 => |q| return if (q.act != 0 and l.ck != null) l.checkpoint(s, l.ck.?, q, in, out, rows) else l.onLane(s, q, .fp8, in, out, rows),
            .bf16 => |q| return l.onLane(s, q, .bf16, in, out, rows),
            .exl3 => |q| return l.onExl3(s, q, in, out, rows),
            .exl3_dense => |q| {
                const e = l.exl3_dense orelse return error.FormatNotLoaded;
                if (in.ldx != 0 and in.ldx != q.k) return error.RowStride;
                return e.decode(s, q, in.x, .bf16, rows, out, .bf16);
            },
            .ck8, .ck4 => return l.onNative(s, w, in, out, rows, .bf16),
            .q8_0, .iq2xxs => |q| return (l.ggml orelse return error.FormatNotLoaded).decode(s, q, in.x, in.ldx, rows, out, .bf16),
        }
    }

    /// Prompt rows: out (rows, n) bf16 by the format's prompt GEMM.
    pub fn prompt(l: Linear, s: Stream, w: Weight, in: Rows, out: u64, rows: usize) !void {
        switch (w) {
            .affine4 => |q| {
                return l.affinePrompt(s, in, q, out, rows, false);
            },
            .fp8g => |q| return l.onLane(s, q, .fp8g, in, out, rows),
            .mxfp8 => |q| return l.onLane(s, q, .mxfp8, in, out, rows),
            .nvfp4 => |q| return l.onPrompt(s, q, .fp4, in, out, rows),
            .fp8 => |q| return if (q.act != 0 and l.ck != null) l.checkpoint(s, l.ck.?, q, in, out, rows) else l.onPrompt(s, q, .fp8, in, out, rows),
            .bf16 => |q| return l.onLane(s, q, .bf16, in, out, rows),
            .exl3 => |q| return (l.exl3 orelse return error.FormatNotLoaded).prompt(s, in.x, if (in.ldx == 0) q.k else in.ldx, q, out, rows, false),
            .exl3_dense => |q| {
                const e = l.exl3_dense orelse return error.FormatNotLoaded;
                if (in.ldx != 0 and in.ldx != q.k) return error.RowStride;
                return e.prompt(s, q, in.x, .bf16, rows, out, .bf16);
            },
            .ck8, .ck4 => return l.onNative(s, w, in, out, rows, .bf16),
            // prompt rows always take the prompt GEMM: a row's bits must not depend on where a prompt's chunks are cut
            .q8_0, .iq2xxs => |q| return (l.ggml orelse return error.FormatNotLoaded).promptRows(s, q, in.x, in.ldx, rows, out, .bf16),
        }
    }

    /// Decode rows as decode(), out (rows, n) fp32 unrounded: a tensor-parallel rank's partial rows (nvfp4, exl3).
    pub fn decodeF32(l: Linear, s: Stream, w: Weight, in: Rows, out: u64, rows: usize) !void {
        if (w == .ck8 or w == .ck4) return l.onNative(s, w, in, out, rows, .f32);
        if (w == .affine4) return (l.affine4 orelse return error.FormatNotLoaded).clusterAs(s, in.x, in.ldx, in.sums, w.affine4, out, rows, splitK(w.affine4.n, w.affine4.k), true);
        if (w == .fp8g) return l.onLaneAs(s, w.fp8g, .fp8g, in, out, .f32, rows);
        const q = switch (w) {
            .exl3 => |q| return (l.exl3 orelse return error.FormatNotLoaded).rows(s, in.x, if (in.ldx == 0) q.k else in.ldx, q, out, rows, true),
            .exl3_dense => |q| {
                const e = l.exl3_dense orelse return error.FormatNotLoaded;
                if (in.ldx != 0 and in.ldx != q.k) return error.RowStride;
                return e.decode(s, q, in.x, .bf16, rows, out, .f32);
            },
            .nvfp4 => |q| q,
            else => return error.EpilogueNotBuilt,
        };
        return l.onLaneAs(s, q, .fp4, in, out, .f32, rows);
    }

    /// Prompt rows as prompt(), out (rows, n) fp32 unrounded (nvfp4, exl3).
    pub fn promptF32(l: Linear, s: Stream, w: Weight, in: Rows, out: u64, rows: usize) !void {
        if (w == .ck8 or w == .ck4) return l.onNative(s, w, in, out, rows, .f32);
        if (w == .affine4) return l.affinePrompt(s, in, w.affine4, out, rows, true);
        if (w == .fp8g) return l.onLaneAs(s, w.fp8g, .fp8g, in, out, .f32, rows);
        const q = switch (w) {
            .exl3 => |q| return (l.exl3 orelse return error.FormatNotLoaded).prompt(s, in.x, if (in.ldx == 0) q.k else in.ldx, q, out, rows, true),
            .exl3_dense => |q| {
                const e = l.exl3_dense orelse return error.FormatNotLoaded;
                if (in.ldx != 0 and in.ldx != q.k) return error.RowStride;
                return e.prompt(s, q, in.x, .bf16, rows, out, .f32);
            },
            .nvfp4 => |q| q,
            else => return error.EpilogueNotBuilt,
        };
        return l.onPromptAs(s, q, .fp4, in, out, .f32, rows);
    }

    fn affinePrompt(l: Linear, s: Stream, in: Rows, q: Affine4, out: u64, rows: usize, wide: bool) !void {
        const a = l.affine4 orelse return error.FormatNotLoaded;
        return a.promptAsTile(s, in.x, in.ldx, q, out, rows, wide, affinePromptTile(l.affine_prompt_policy, rows, q.n, q.k));
    }

    fn onNative(l: Linear, s: Stream, w: Weight, in: Rows, out: u64, rows: usize, kind: qmmf.Out) !void {
        const c = l.ck orelse return error.FormatNotLoaded;
        const stride = if (in.ldx == 0) w.inputs() else in.ldx;
        return switch (w) {
            .ck8 => |q| c.matmul8(s, q, in.x, stride, rows, in.native, out, kind),
            .ck4 => |q| c.matmul4(s, q, in.x, stride, rows, in.native, out, kind),
            else => unreachable,
        };
    }

    fn onExl3(l: Linear, s: Stream, q: exl3_linear.Matrix, in: Rows, out: u64, rows: usize) !void {
        const e = l.exl3 orelse return error.FormatNotLoaded;
        return e.rows(s, in.x, if (in.ldx == 0) q.k else in.ldx, q, out, rows, false);
    }

    fn onPrompt(l: Linear, s: Stream, q: qmmf.Weight, mode: qmmf.Mode, in: Rows, out: u64, rows: usize) !void {
        return l.onPromptAs(s, q, mode, in, out, .bf16, rows);
    }

    fn onPromptAs(
        l: Linear,
        s: Stream,
        q: qmmf.Weight,
        mode: qmmf.Mode,
        in: Rows,
        out: u64,
        kind: qmmf.Out,
        rows: usize,
    ) !void {
        const p = l.nvfp4_prompt orelse return error.FormatNotLoaded;
        if (q.mode != mode) return error.FormatMismatch;
        return p.matmulOut(s, in.x, if (in.ldx == 0) q.k else in.ldx, rows, q, out, kind);
    }

    fn onLane(l: Linear, s: Stream, q: qmmf.Weight, mode: qmmf.Mode, in: Rows, out: u64, rows: usize) !void {
        return l.onLaneAs(s, q, mode, in, out, .bf16, rows);
    }

    fn onLaneAs(
        l: Linear,
        s: Stream,
        q: qmmf.Weight,
        mode: qmmf.Mode,
        in: Rows,
        out: u64,
        kind: qmmf.Out,
        rows: usize,
    ) !void {
        const lane = l.lane orelse return error.FormatNotLoaded;
        if (q.mode != mode) return error.FormatMismatch;
        return lane.matmulOut(s, in.x, if (in.ldx == 0) q.k else in.ldx, rows, q, out, kind, in.part);
    }
};

/// The affine-4 instantiations' names (cuobjdump -symbols of the qmm_group, qmm_prefill and lane_gemv fatbins).
pub const symbols = struct {
    pub const group = "_ZN12tf_qmm_group12group_kernelILi64ELi16ELi64ELi1ELi4ELi8ELb0ELb0ELb0ELb0EEEvPK13__nv_bfloat16PKfNS_5PartsEiiiii";
    pub const prefill = "_ZN14tf_qmm_prefill14prefill_kernelILi64ELi128ELi128ELi2ELi4ELi3ELb0EEEvPK13__nv_bfloat16PKjS3_S3_Pviiiiii";
    pub const gemv = "_ZN12tf_lane_gemv11gemv_kernelILi64ELi64ELi8EEEvPK13__nv_bfloat16PKfNS_4PartEiii";
    pub const gemv_split = "tf_lane_gemv_split";
    pub const pack_words = "tf_pack_dense";
    pub const pack_scales = "tf_transpose_pad16";
};

/// qmm_group.cu's Part and Parts, passed by value: four projections at most, one used here.
pub const Part = extern struct { w: u64, scales: u64, biases: u64, out: u64, n: c_int, npad: c_int, sk: c_int, tiles: c_int, first: c_int };
pub const Parts = extern struct { p: [4]Part, count: c_int };

/// lane_gemv.cu's Part: one projection whose column tiles the CTAs share out.
pub const GemvPart = extern struct { w: u64, scales: u64, biases: u64, out: u64, n: c_int, npad: c_int, sk: c_int, tiles: c_int };

comptime {
    std.debug.assert(@sizeOf(Part) == 56 and @sizeOf(Parts) == 232 and @sizeOf(GemvPart) == 48);
}

pub const group_smem: u32 = 35328; // LaneTile<64, 16, 64, 1, 4, 8>::SMEM
pub const split_smem: u32 = 17664; // LaneTile<64, 16, 64, 1, 4, 4>::SMEM: four stages, so more split CTAs fit an SM
pub const prefill_smem: u32 = 62976; // Tile<64, 128, 128, 2, 4, 3>::SMEM

/// Split-K lane_gemv's fp32 slice partials and per-tile tickets: splitK keeps tiles * sk under 384 and tiles under 192.
pub const Split = struct {
    work: memory.DeviceBuffer,
    tickets: memory.DeviceBuffer,

    pub const tiles_max = 192;
    const work_bytes = 384 * 16 * 64 * 4;

    pub fn init(d: *const Driver) !Split {
        var work = try memory.DeviceBuffer.alloc(d, work_bytes);
        errdefer work.free();
        var tickets = try memory.DeviceBuffer.alloc(d, tiles_max * 4);
        errdefer tickets.free();
        try d.check(d.api.cuMemsetD32_v2(tickets.ptr, 0, tiles_max), "cuMemsetD32");
        return .{ .work = work, .tickets = tickets };
    }

    pub fn deinit(sp: *Split) void {
        sp.work.free();
        sp.tickets.free();
    }
};

/// qmm.split_k: K slices fixed by the weight's shape, never by the row count.
pub fn splitK(n: usize, k: usize) usize {
    const tiles = (n + 63) / 64;
    const groups = k / 64;
    var sk: usize = 1;
    while (sk < 8 and tiles * sk < 192 and groups % (sk * 2) == 0 and groups / (sk * 2) >= 8) sk *= 2;
    return sk;
}

fn int(x: usize) c_int {
    return @intCast(x);
}

fn u(x: usize) u32 {
    return @intCast(x);
}

/// The affine-4 kernels: qmm_group's cluster tile, lane_gemv and its split-K form, qmm_prefill's prompt GEMM.
pub const Affine4Kernels = struct {
    pub const max_rows = 16; // rows decode() takes: its kernels tile 16 rows

    group_fn: Function,
    group_f32: ?Function = null,
    prefill_f32: ?Function = null,
    prefill3: [2]?Function = @splat(null),
    prefill9: [2]?Function = @splat(null),
    gemv_fn: Function,
    split_fn: Function,
    prefill_fn: Function,
    words_fn: Function, // affine4_pack.cu: MLX words into the tiles
    scales_fn: Function, // affine4_pack.cu: (n, kg) scales or biases into (kg, npad)
    gemv_blocks: usize, // resident lane_gemv CTAs: per SM times SMs
    split: ?Split, // every GPU but GB10: lane_gemv's slices on CTAs of their own (one stream's decode() at a time)
    pdl: bool, // GB10: the decode kernels launch with programmatic dependent launch

    /// From loaded qmm_group, qmm_prefill, lane_gemv and affine4_pack modules; GB10 decodes without the split form.
    pub fn resolve(d: *const Driver, group_mod: Module, prefill_mod: Module, gemv_mod: Module, pack_mod: Module, sms: usize, gb10: bool) !Affine4Kernels {
        var a: Affine4Kernels = undefined;
        a.prefill3 = @splat(null);
        a.prefill9 = @splat(null);
        a.group_fn = try group_mod.function(symbols.group);
        a.prefill_fn = try prefill_mod.function(symbols.prefill);
        a.group_f32 = try group_mod.function("_ZN12tf_qmm_group12group_kernelILi64ELi16ELi64ELi1ELi4ELi8ELb1ELb0ELb0ELb0EEEvPK13__nv_bfloat16PKfNS_5PartsEiiiii");
        a.prefill_f32 = try prefill_mod.function("_ZN14tf_qmm_prefill14prefill_kernelILi64ELi128ELi128ELi2ELi4ELi3ELb1EEEvPK13__nv_bfloat16PKjS3_S3_Pviiiiii");
        try a.group_f32.?.allowDynamicShared(group_smem);
        try a.prefill_f32.?.allowDynamicShared(prefill_smem);
        a.gemv_fn = try gemv_mod.function(symbols.gemv);
        a.split_fn = try gemv_mod.function(symbols.gemv_split);
        a.words_fn = try pack_mod.function(symbols.pack_words);
        a.scales_fn = try pack_mod.function(symbols.pack_scales);
        try a.group_fn.allowDynamicShared(group_smem);
        try a.gemv_fn.allowDynamicShared(group_smem);
        try a.split_fn.allowDynamicShared(split_smem);
        try a.prefill_fn.allowDynamicShared(prefill_smem);
        a.gemv_blocks = @max(1, try a.gemv_fn.occupancy(128, group_smem)) * sms;
        a.pdl = gb10;
        a.split = if (gb10) null else try Split.init(d);
        return a;
    }

    pub fn enablePromptTiles(a: *Affine4Kernels, prefill_mod: Module) !void {
        var tile3: [2]?Function = undefined;
        var tile9: [2]?Function = undefined;
        inline for (0..2) |epilogue| {
            const f3 = try prefill_mod.function(std.fmt.comptimePrint("_ZN14tf_qmm_prefill14prefill_kernelILi64ELi64ELi64ELi1ELi4ELi4ELb{d}EEEvPK13__nv_bfloat16PKjS3_S3_Pviiiiii", .{epilogue}));
            const f9 = try prefill_mod.function(std.fmt.comptimePrint("_ZN14tf_qmm_prefill14prefill_kernelILi64ELi128ELi128ELi2ELi2ELi2ELb{d}EEEvPK13__nv_bfloat16PKjS3_S3_Pviiiiii", .{epilogue}));
            try f3.allowDynamicShared(41984);
            try f9.allowDynamicShared(41984);
            tile3[epilogue] = f3;
            tile9[epilogue] = f9;
        }
        a.prefill3 = tile3;
        a.prefill9 = tile9;
    }

    pub fn deinit(a: *Affine4Kernels) void {
        if (a.split) |*sp| sp.deinit();
    }

    /// qmm_fast.tile: MLX words (n, k/8), scales and biases (n, k/64) on the device into q's tiles (Affine4.layout).
    pub fn pack(a: *const Affine4Kernels, s: Stream, words: u64, scales: u64, biases: u64, q: Affine4) !void {
        const kg = q.k / 64;
        const total: u64 = @as(u64, q.npad / 64) * kg * 512;
        var g: launch_.Args = .{};
        g.add(words);
        for ([_]usize{ q.n, q.k / 8, kg }) |v| g.add(int(v));
        g.add(q.w);
        g.add(@as(c_longlong, @intCast(total)));
        try launch_.launch(a.words_fn, .{ .grid = .{ .x = @intCast((total + 255) / 256) }, .block = .{ .x = 256 } }, s, &g);
        for ([_]u64{ scales, biases }, [_]u64{ q.s, q.b }) |src, out| {
            var t: launch_.Args = .{};
            t.add(src);
            for ([_]usize{ q.n, kg, q.npad }) |v| t.add(int(v));
            t.add(out);
            const cells = @as(u64, kg) * q.npad;
            try launch_.launch(a.scales_fn, .{ .grid = .{ .x = @intCast((cells + 255) / 256) }, .block = .{ .x = 256 } }, s, &t);
        }
    }

    /// qmm.matmul: x (rows, k) bf16 with group sums xs -> out (rows, n) bf16, qmm_group's tile-2 bits on every path.
    pub fn decode(a: *const Affine4Kernels, s: Stream, x: u64, xs: u64, q: Affine4, out: u64, rows: usize) !void {
        if (rows > max_rows) return error.WindowTooWide;
        const sk = splitK(q.n, q.k);
        if (sk > 1) if (a.split) |sp| return a.gemvSplit(s, x, xs, q, out, rows, sk, sp);
        return if (sk > 1) a.gemv(s, x, xs, q, out, rows, sk) else a.cluster(s, x, xs, q, out, rows, sk);
    }

    /// Each CTA writes one K slice's partial from zero; the tile's last CTA sums them in slice order.
    pub fn gemvSplit(a: *const Affine4Kernels, s: Stream, x: u64, xs: u64, q: Affine4, out: u64, rows: usize, sk: usize, sp: Split) !void {
        const tiles = (q.n + 63) / 64;
        if (tiles > Split.tiles_max or tiles * sk * 16 * 64 * 4 > sp.work.len) return error.SplitTooWide;
        var g: launch_.Args = .{};
        g.add(x);
        g.add(xs);
        g.add(GemvPart{ .w = q.w, .scales = q.s, .biases = q.b, .out = out, .n = int(q.n), .npad = int(q.npad), .sk = int(sk), .tiles = int(tiles) });
        for ([_]usize{ rows, q.k, q.k }) |v| g.add(int(v));
        g.add(sp.work.ptr);
        g.add(sp.tickets.ptr);
        try launch_.launch(a.split_fn, .{ .grid = .{ .x = u(tiles * sk) }, .block = .{ .x = 128 }, .shared = split_smem }, s, &g);
    }

    /// lane_gemv: every K slice of a column tile in one CTA, summed in slice order, CTAs looping over the tiles.
    pub fn gemv(a: *const Affine4Kernels, s: Stream, x: u64, xs: u64, q: Affine4, out: u64, rows: usize, sk: usize) !void {
        const tiles = (q.n + 63) / 64;
        var g: launch_.Args = .{};
        g.add(x);
        g.add(xs);
        g.add(GemvPart{ .w = q.w, .scales = q.s, .biases = q.b, .out = out, .n = int(q.n), .npad = int(q.npad), .sk = int(sk), .tiles = int(tiles) });
        for ([_]usize{ rows, q.k, q.k }) |v| g.add(int(v));
        const cfg: launch_.Config = .{ .grid = .{ .x = u(@min(tiles, a.gemv_blocks)) }, .block = .{ .x = 128 }, .shared = group_smem, .pdl = a.pdl };
        try launch_.launch(a.gemv_fn, cfg, s, &g);
    }

    /// qmm_group's tile 2: a cluster of `sk` CTAs a column tile, its K slices summed over distributed shared memory.
    pub fn cluster(a: *const Affine4Kernels, s: Stream, x: u64, xs: u64, q: Affine4, out: u64, rows: usize, sk: usize) !void {
        return a.clusterAs(s, x, q.k, xs, q, out, rows, sk, false);
    }

    pub fn clusterAs(a: *const Affine4Kernels, s: Stream, x: u64, ldx: usize, xs: u64, q: Affine4, out: u64, rows: usize, sk: usize, wide: bool) !void {
        const stride = try affineStride(x, ldx, q.k, rows);
        const tiles = (q.n + 63) / 64;
        var parts: Parts = std.mem.zeroes(Parts);
        parts.count = 1;
        parts.p[0] = .{ .w = q.w, .scales = q.s, .biases = q.b, .out = out, .n = int(q.n), .npad = int(q.npad), .sk = int(sk), .tiles = int(tiles), .first = 0 };
        const rows_t = (rows + 15) / 16;
        const clusters = rows_t * tiles; // one part: the cluster is its sk K slices of one tile
        var g: launch_.Args = .{};
        g.add(x);
        g.add(xs);
        g.add(parts);
        for ([_]usize{ rows, q.k, stride, rows_t, sk }) |v| g.add(int(v));
        const cfg: launch_.Config = .{
            .grid = .{ .x = u(clusters * sk) },
            .block = .{ .x = 128 },
            .shared = group_smem,
            .cluster = if (sk > 1) .{ .x = u(sk) } else null,
            .pdl = a.pdl,
        };
        const f = if (wide) a.group_f32 orelse return error.EpilogueNotBuilt else a.group_fn;
        try launch_.launch(f, cfg, s, &g);
    }

    /// qmm.prefill_matmul (tile 0): weights rounded once to bf16, one fp32 chain over K; bf16 out.
    pub fn prompt(a: *const Affine4Kernels, s: Stream, x: u64, q: Affine4, out: u64, rows: usize) !void {
        return a.promptAs(s, x, q.k, q, out, rows, false);
    }

    pub fn promptAs(a: *const Affine4Kernels, s: Stream, x: u64, ldx: usize, q: Affine4, out: u64, rows: usize, wide: bool) !void {
        return a.promptAsTile(s, x, ldx, q, out, rows, wide, .tile0);
    }

    pub fn promptAsTile(a: *const Affine4Kernels, s: Stream, x: u64, ldx: usize, q: Affine4, out: u64, rows: usize, wide: bool, tile: AffinePromptTile) !void {
        const input_stride = try affineStride(x, ldx, q.k, rows);
        const stride = if (rows == 1 and tile != .tile0) q.k else input_stride;
        const shape = affinePromptGeometry(tile);
        const rows_t = (rows + shape.bm - 1) / shape.bm;
        const band = (12 << 20) / (shape.bm * q.k * 2);
        const group = @max(1, @min(rows_t, band));
        var g: launch_.Args = .{};
        for ([_]u64{ x, q.w, q.s, q.b, out }) |v| g.add(v);
        for ([_]usize{ rows, q.n, q.k, q.npad, stride, group }) |v| g.add(int(v));
        const grid = rows_t * ((q.n + shape.bn - 1) / shape.bn);
        const f = switch (tile) {
            .tile0 => if (wide) a.prefill_f32 orelse return error.EpilogueNotBuilt else a.prefill_fn,
            .tile3 => a.prefill3[@intFromBool(wide)] orelse return error.PromptPolicyNotLoaded,
            .tile9 => a.prefill9[@intFromBool(wide)] orelse return error.PromptPolicyNotLoaded,
        };
        try launch_.launch(f, .{ .grid = .{ .x = u(grid), .y = 1, .z = 1 }, .block = .{ .x = shape.threads }, .shared = shape.shared }, s, &g);
    }
};

fn affineStride(x: u64, ldx: usize, k: usize, rows: usize) !usize {
    const stride = if (ldx == 0) k else ldx;
    if (stride < k or x % 16 != 0 or (rows > 1 and stride % 8 != 0)) return error.UnsupportedAffineStride;
    return stride;
}

test "affine tiled rows reject overlapping or unaligned strides before CUDA" {
    try std.testing.expectEqual(@as(usize, 128), try affineStride(256, 0, 128, 3));
    try std.testing.expectEqual(@as(usize, 136), try affineStride(256, 136, 128, 3));
    try std.testing.expectError(error.UnsupportedAffineStride, affineStride(256, 127, 128, 3));
    try std.testing.expectError(error.UnsupportedAffineStride, affineStride(256, 129, 128, 3));
    try std.testing.expectError(error.UnsupportedAffineStride, affineStride(258, 128, 128, 3));
}

/// affine8.cu's shared bytes a width: Tile<64, 64, 6, bits>::SMEM for gemv, Tile<64, 64, 4, bits>::SMEM for split.
pub fn gemvSmem(bits: u8) u32 {
    return if (bits == 8) 38784 else 14208 + 3072 * @as(u32, bits);
}

pub fn splitSmem(bits: u8) u32 {
    return if (bits == 8) 25856 else 9472 + 2048 * @as(u32, bits);
}

/// affine8.cu's Prefill<bits>::SMEM: the prompt GEMM's three stages.
pub fn prefillSmem(bits: u8) u32 {
    return 50688 + 3072 * @as(u32, bits);
}

/// The widths AffineNKernels serves, in the order of its function tables.
pub const tile_widths = [_]u8{ 2, 3, 5, 6, 8 };

fn widthAt(bits: u8) !usize {
    return std.mem.indexOfScalar(u8, &tile_widths, bits) orelse error.UnsupportedWidth;
}

/// Every MLX width but 4: codes of 2, 3, 5 and 6 bits on tiles of their width, 7 and 8 on 8-bit tiles, unrounded.
pub const AffineNKernels = struct {
    gemv_fns: [tile_widths.len]Function,
    split_fns: [tile_widths.len]Function,
    prefill_fns: [tile_widths.len]Function,
    pack4_fn: Function, // MLX codes of 2 to 4 bits into the affine-4 tiles
    pack8_fn: Function, // MLX codes of 5 to 8 bits into the 8-bit tiles
    packn_fn: Function, // MLX codes of 2, 3, 5 or 6 bits into tiles of that width
    widen_fn: Function, // MLX rows into MLX rows of 4 or 8-bit codes (token embeddings)
    scales_fn: Function, // affine4_pack.cu's tf_transpose_pad16
    gemv_blocks: [tile_widths.len]usize, // resident gemv CTAs a width: per SM times SMs
    split: ?Split,
    pdl: bool,

    /// From loaded affine8, affine_prefill and affine4_pack modules; GB10 decodes without the split form.
    pub fn resolve(d: *const Driver, mod: Module, prefill_mod: Module, pack_mod: Module, sms: usize, gb10: bool) !AffineNKernels {
        var a: AffineNKernels = undefined;
        for (tile_widths, &a.gemv_fns, &a.split_fns, &a.prefill_fns, &a.gemv_blocks) |bits, *g, *sp, *pf, *blocks| {
            var name: [32]u8 = undefined;
            g.* = try mod.function(try std.fmt.bufPrintSentinel(&name, "tf_affine{d}_gemv", .{bits}, 0));
            sp.* = try mod.function(try std.fmt.bufPrintSentinel(&name, "tf_affine{d}_split", .{bits}, 0));
            pf.* = try prefill_mod.function(try std.fmt.bufPrintSentinel(&name, "tf_affine{d}_prefill", .{bits}, 0));
            try g.allowDynamicShared(gemvSmem(bits));
            try sp.allowDynamicShared(splitSmem(bits));
            try pf.allowDynamicShared(prefillSmem(bits));
            blocks.* = @max(1, try g.occupancy(128, gemvSmem(bits))) * sms;
        }
        a.pack4_fn = try mod.function("tf_affine8_pack4");
        a.pack8_fn = try mod.function("tf_affine8_pack8");
        a.packn_fn = try mod.function("tf_affinen_pack");
        a.widen_fn = try mod.function("tf_affine8_widen");
        a.scales_fn = try pack_mod.function(symbols.pack_scales);
        a.pdl = gb10;
        a.split = if (gb10) null else try Split.init(d);
        return a;
    }

    pub fn deinit(a: *AffineNKernels) void {
        if (a.split) |*sp| sp.deinit();
    }

    /// MLX words (n, k * bits / 32) on the device into the tiles at `w` of the width `servedBits(bits)` serves.
    pub fn packWords(a: *const AffineNKernels, s: Stream, words: u64, bits: u8, n: usize, k: usize, w: u64) !void {
        return a.packAt(s, words, bits, n, k, w, servedBits(bits));
    }

    /// `packWords` onto the tiles of width `wide`: `bits` itself, or 4 (Affine4's tiles) or 8 when the codes fit them.
    pub fn packAt(a: *const AffineNKernels, s: Stream, words: u64, bits: u8, n: usize, k: usize, w: u64, wide: u8) !void {
        if (k % 64 != 0 or k * bits % 32 != 0) return error.UnsupportedShape;
        if (wide < bits or (wide != 4 and wide != 8 and wide != bits)) return error.UnsupportedWidth;
        const kg = k / 64;
        const npad = (n + 127) / 128 * 128;
        const per: u64 = switch (wide) {
            4 => 512,
            8 => 1024,
            else => 128 * @as(u64, wide),
        };
        const total: u64 = @as(u64, npad / 64) * kg * per;
        var g: launch_.Args = .{};
        g.add(words);
        for ([_]usize{ bits, n, k * bits / 32, kg }) |v| g.add(int(v));
        g.add(w);
        g.add(@as(c_longlong, @intCast(total)));
        const f = switch (wide) {
            4 => a.pack4_fn,
            8 => a.pack8_fn,
            else => a.packn_fn,
        };
        try launch_.launch(f, .{ .grid = .{ .x = @intCast((total + 255) / 256) }, .block = .{ .x = 256 } }, s, &g);
    }

    /// (n, k/64) scales or biases into the tiles' (k/64, npad) layout.
    pub fn packTable(a: *const AffineNKernels, s: Stream, src: u64, n: usize, k: usize, npad: usize, out: u64) !void {
        const kg = k / 64;
        var t: launch_.Args = .{};
        t.add(src);
        for ([_]usize{ n, kg, npad }) |v| t.add(int(v));
        t.add(out);
        const cells = @as(u64, kg) * npad;
        try launch_.launch(a.scales_fn, .{ .grid = .{ .x = @intCast((cells + 255) / 256) }, .block = .{ .x = 256 } }, s, &t);
    }

    /// MLX rows (n, k) of `bits` codes into MLX rows of `wide`-bit codes at `out`, every code's value kept.
    pub fn widen(a: *const AffineNKernels, s: Stream, words: u64, bits: u8, n: usize, k: usize, out: u64, wide: u8) !void {
        if (wide != 4 and wide != 8) return error.UnsupportedWidth;
        const total: u64 = @as(u64, n) * k * wide / 32;
        var g: launch_.Args = .{};
        g.add(words);
        g.add(int(bits));
        g.add(int(k));
        g.add(out);
        g.add(int(wide));
        g.add(@as(c_longlong, @intCast(total)));
        try launch_.launch(a.widen_fn, .{ .grid = .{ .x = @intCast((total + 255) / 256) }, .block = .{ .x = 256 } }, s, &g);
    }

    /// x (rows, k) bf16 with group sums xs -> out (rows, n) bf16; a row's bits the same at every row count up to 16.
    pub fn decode(a: *const AffineNKernels, s: Stream, x: u64, xs: u64, q: AffineN, out: u64, rows: usize) !void {
        if (rows > 16) return error.WindowTooWide;
        const at = try widthAt(q.bits);
        const sk = splitK(q.n, q.k);
        const tiles = (q.n + 63) / 64;
        const part: GemvPart = .{ .w = q.w, .scales = q.s, .biases = q.b, .out = out, .n = int(q.n), .npad = int(q.npad), .sk = int(sk), .tiles = int(tiles) };
        var g: launch_.Args = .{};
        g.add(x);
        g.add(xs);
        g.add(part);
        for ([_]usize{ rows, q.k, q.k }) |v| g.add(int(v));
        if (sk > 1) if (a.split) |sp| {
            if (tiles > Split.tiles_max or tiles * sk * 16 * 64 * 4 > sp.work.len) return error.SplitTooWide;
            g.add(sp.work.ptr);
            g.add(sp.tickets.ptr);
            return launch_.launch(a.split_fns[at], .{ .grid = .{ .x = u(tiles * sk) }, .block = .{ .x = 128 }, .shared = splitSmem(q.bits) }, s, &g);
        };
        const cfg: launch_.Config = .{ .grid = .{ .x = u(@min(tiles, a.gemv_blocks[at])) }, .block = .{ .x = 128 }, .shared = gemvSmem(q.bits), .pdl = a.pdl };
        try launch_.launch(a.gemv_fns[at], cfg, s, &g);
    }

    /// Prompt rows x (rows, k) -> out (rows, n), bf16: Affine4Kernels.prompt's blocks and bands over AffineN's tiles.
    pub fn prompt(a: *const AffineNKernels, s: Stream, x: u64, q: AffineN, out: u64, rows: usize) !void {
        const at = try widthAt(q.bits);
        const rows_t = (rows + 127) / 128;
        const band = (12 << 20) / (128 * q.k * 2);
        const group = @max(1, @min(rows_t, band));
        var g: launch_.Args = .{};
        for ([_]u64{ x, q.w, q.s, q.b, out }) |v| g.add(v);
        for ([_]usize{ rows, q.n, q.k, q.npad, q.k, group }) |v| g.add(int(v));
        const grid = rows_t * ((q.n + 127) / 128);
        try launch_.launch(a.prefill_fns[at], .{ .grid = .{ .x = u(grid) }, .block = .{ .x = 256 }, .shared = prefillSmem(q.bits) }, s, &g);
    }
};

test "AffineN's tiles hold `bits` a weight over Affine4's padding" {
    const l = AffineN.layout(10304, 2688, 8);
    try std.testing.expectEqual(@as(usize, 10368), l.npad);
    try std.testing.expectEqual(@as(usize, 10368 * 2688), l.words);
    try std.testing.expectEqual(Affine4.layout(10304, 2688).scales, l.scales);
    try std.testing.expectEqual(@as(usize, 10368 * 2688 * 3 / 8), AffineN.layout(10304, 2688, 3).words);
    for ([_]u8{ 2, 3, 4, 5, 6, 8 }) |b| try std.testing.expectEqual(b, servedBits(b));
    try std.testing.expectEqual(@as(u8, 8), servedBits(7));
    for ([_]u8{ 2, 3, 4 }) |b| try std.testing.expectEqual(@as(u8, 4), rowBits(b));
    for ([_]u8{ 5, 6, 8 }) |b| try std.testing.expectEqual(@as(u8, 8), rowBits(b));
    try std.testing.expectEqual(@as(u32, 38784), gemvSmem(8));
    try std.testing.expectEqual(@as(u32, 14208 + 3072 * 2), gemvSmem(2));
    try std.testing.expectError(error.UnsupportedWidth, widthAt(4));
}

test "split_k follows the Python shapes" {
    try std.testing.expectEqual(@as(usize, 2), splitK(10304, 2688));
    try std.testing.expectEqual(@as(usize, 8), splitK(2688, 4096));
    try std.testing.expectEqual(@as(usize, 2), splitK(4608, 2688));
    try std.testing.expectEqual(@as(usize, 4), splitK(2688, 5376));
    try std.testing.expectEqual(@as(usize, 1), splitK(131072, 2688));
}

test "a weight view reports its shape in every format" {
    const a: Weight = .{ .affine4 = .{ .w = 0, .s = 0, .b = 0, .n = 10304, .k = 2688, .npad = 10304 } };
    const f: Weight = .{ .fp8g = .{ .mode = .fp8g, .codes = 0, .scales = 0, .n = 7168, .k = 2560, .npad = 7168 } };
    try std.testing.expectEqual(@as(usize, 10304), a.outputs());
    try std.testing.expectEqual(@as(usize, 2560), f.inputs());
}

test "a format that is not loaded is refused before any launch" {
    const l: Linear = .{};
    const w: Weight = .{ .affine4 = .{ .w = 0, .s = 0, .b = 0, .n = 64, .k = 64, .npad = 64 } };
    const s: Stream = undefined; // never reached: the refusal comes first
    try std.testing.expectError(error.FormatNotLoaded, l.decode(s, w, .{ .x = 0 }, 0, 1));
    try std.testing.expectError(error.FormatNotLoaded, l.prompt(s, w, .{ .x = 0 }, 0, 1));
    const e: Weight = .{ .exl3 = .{ .trellis = 0, .suh = 0, .svh = 0, .k = 128, .n = 128, .bits = 4, .codebook = .mul1 } };
    try std.testing.expectError(error.FormatNotLoaded, l.decode(s, e, .{ .x = 0 }, 0, 1));
    try std.testing.expectError(error.FormatNotLoaded, l.prompt(s, e, .{ .x = 0 }, 0, 1));
    const native: Weight = .{ .ck8 = .{ .codes = 0, .scale = 1, .n = 160, .k = 128, .npad = 256 } };
    try std.testing.expectError(error.FormatNotLoaded, l.decode(s, native, .{ .x = 0 }, 0, 1));
    try std.testing.expectError(error.FormatNotLoaded, l.promptF32(s, native, .{ .x = 0 }, 0, 1));
}

test "the affine-4 tiles' sizes follow qmm_fast.tile" {
    const l = Affine4.layout(10304, 2688); // Nemotron's in_proj: rows to whole 128s, 4 bits a weight, 2 bytes a group
    try std.testing.expectEqual(@as(usize, 10368), l.npad);
    try std.testing.expectEqual(@as(usize, 10368 * 2688 / 2), l.words);
    try std.testing.expectEqual(@as(usize, 42 * 10368 * 2), l.scales);
}

test "a lane weight whose mode is not its format's is refused before any launch" {
    const lane: qmmf.Lane = undefined; // never reached: the refusal comes first
    const l: Linear = .{ .lane = &lane };
    const fp4: qmmf.Weight = .{ .mode = .fp4, .codes = 0, .scales = 0, .n = 64, .k = 64, .npad = 128 };
    const s: Stream = undefined;
    try std.testing.expectError(error.FormatMismatch, l.decode(s, .{ .fp8g = fp4 }, .{ .x = 0 }, 0, 1));
    try std.testing.expectError(error.FormatNotLoaded, l.prompt(s, .{ .nvfp4 = fp4 }, .{ .x = 0 }, 0, 1));
    const f8: qmmf.Weight = .{ .mode = .fp8g, .codes = 0, .scales = 0, .n = 64, .k = 64, .npad = 128 };
    try std.testing.expectError(error.FormatMismatch, l.decodeF32(s, .{ .fp8g = fp4 }, .{ .x = 0 }, 0, 1));
    try std.testing.expectError(error.EpilogueNotBuilt, l.promptF32(s, .{ .bf16 = f8 }, .{ .x = 0 }, 0, 1));
}

test {
    _ = @import("qlinear_test.zig");
}

test "GLM affine prompt policy selects exact production thresholds without changing stock" {
    try std.testing.expectEqual(AffinePromptPolicy.stock, (Linear{}).affine_prompt_policy);
    for ([_]usize{ 1, 14, 31, 128, 512, 1024, 2048 }) |rows| {
        try std.testing.expectEqual(AffinePromptTile.tile0, affinePromptTile(.stock, rows, 12576, 4096));
        try std.testing.expectEqual(AffinePromptTile.tile3, affinePromptTile(.glm_production, rows, 4096, 128));
    }
    try std.testing.expectEqual(AffinePromptTile.tile3, affinePromptTile(.glm_production, 128, 12160, 1024));
    try std.testing.expectEqual(AffinePromptTile.tile9, affinePromptTile(.glm_production, 128, 12288, 1024));
    try std.testing.expectEqual(AffinePromptTile.tile3, affinePromptTile(.glm_production, 128, 12288, 1023));
    try std.testing.expectEqual(AffinePromptTile.tile3, affinePromptTile(.glm_production, 512, 2048, 4096));
    try std.testing.expectEqual(AffinePromptTile.tile9, affinePromptTile(.glm_production, 1024, 2048, 4096));
    try std.testing.expectEqual(AffinePromptTile.tile9, affinePromptTile(.glm_production, 1, 12576, 4096));
    try std.testing.expectEqual(AffinePromptGeometry{ .bm = 128, .bn = 128, .threads = 256, .shared = 62976 }, affinePromptGeometry(.tile0));
    try std.testing.expectEqual(AffinePromptGeometry{ .bm = 64, .bn = 64, .threads = 128, .shared = 41984 }, affinePromptGeometry(.tile3));
    try std.testing.expectEqual(AffinePromptGeometry{ .bm = 128, .bn = 128, .threads = 128, .shared = 41984 }, affinePromptGeometry(.tile9));
}

test "Linear prompt policy routes both epilogues with each tile's actual launch geometry" {
    const abi = @import("abi.zig");
    const Trace = struct {
        var id: usize = 0;
        var grid: u32 = 0;
        var block: u32 = 0;
        var shared: u32 = 0;
        var stride: c_int = 0;
        var group: c_int = 0;
        fn value(raw: [*]?*anyopaque, at: usize) c_int {
            return @as(*align(1) const c_int, @ptrCast(raw[at].?)).*;
        }
        fn launch(f: abi.Function, gx: c_uint, _: c_uint, _: c_uint, bx: c_uint, _: c_uint, _: c_uint, bytes: c_uint, _: abi.Stream, args: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) abi.Result {
            id = @intFromPtr(f.?);
            grid = gx;
            block = bx;
            shared = bytes;
            stride = value(args.?, 9);
            group = value(args.?, 10);
            return 0;
        }
    };
    var d: Driver = .{ .lib = undefined, .api = undefined };
    d.api.cuLaunchKernel = Trace.launch;
    const Fn = struct {
        fn of(owner: *const Driver, id: usize) Function {
            return .{ .d = owner, .handle = @ptrFromInt(id) };
        }
    };
    const a: Affine4Kernels = .{
        .group_fn = undefined,
        .gemv_fn = undefined,
        .split_fn = undefined,
        .prefill_fn = Fn.of(&d, 1),
        .prefill_f32 = Fn.of(&d, 2),
        .prefill3 = .{ Fn.of(&d, 3), Fn.of(&d, 4) },
        .prefill9 = .{ Fn.of(&d, 5), Fn.of(&d, 6) },
        .words_fn = undefined,
        .scales_fn = undefined,
        .gemv_blocks = 0,
        .split = null,
        .pdl = false,
    };
    const stream: Stream = .{ .d = &d, .handle = null };
    var q: Affine4 = .{ .w = 256, .s = 512, .b = 768, .n = 2048, .k = 4096, .npad = 2048 };
    for ([_]bool{ false, true }) |wide| {
        const stock: Linear = .{ .affine4 = &a };
        const tuned: Linear = .{ .affine4 = &a, .affine_prompt_policy = .glm_production };
        const w: Weight = .{ .affine4 = q };
        if (wide) try stock.promptF32(stream, w, .{ .x = 1024 }, 2048, 512) else try stock.prompt(stream, w, .{ .x = 1024 }, 2048, 512);
        try std.testing.expectEqual(@as(usize, 1 + @intFromBool(wide)), Trace.id);
        try std.testing.expectEqual(@as(u32, 64), Trace.grid);
        try std.testing.expectEqual(@as(u32, 256), Trace.block);
        try std.testing.expectEqual(@as(u32, 62976), Trace.shared);
        if (wide) try tuned.promptF32(stream, w, .{ .x = 1024 }, 2048, 512) else try tuned.prompt(stream, w, .{ .x = 1024 }, 2048, 512);
        try std.testing.expectEqual(@as(usize, 3 + @intFromBool(wide)), Trace.id);
        try std.testing.expectEqual(@as(u32, 256), Trace.grid);
        try std.testing.expectEqual(@as(u32, 128), Trace.block);
        try std.testing.expectEqual(@as(u32, 41984), Trace.shared);
        try std.testing.expectEqual(@as(c_int, 8), Trace.group);
        if (wide) try tuned.promptF32(stream, w, .{ .x = 1024 }, 2048, 1024) else try tuned.prompt(stream, w, .{ .x = 1024 }, 2048, 1024);
        try std.testing.expectEqual(@as(usize, 5 + @intFromBool(wide)), Trace.id);
        try std.testing.expectEqual(@as(u32, 128), Trace.grid);
        try std.testing.expectEqual(@as(c_int, 8), Trace.group);
        q.n = 12576;
        q.npad = 12672;
        if (wide) try tuned.promptF32(stream, .{ .affine4 = q }, .{ .x = 1024, .ldx = 8192 }, 2048, 1) else try tuned.prompt(stream, .{ .affine4 = q }, .{ .x = 1024, .ldx = 8192 }, 2048, 1);
        try std.testing.expectEqual(@as(c_int, 4096), Trace.stride);
        try std.testing.expectEqual(@as(u32, 99), Trace.grid);
        q.n = 2048;
        q.npad = 2048;
    }
}
