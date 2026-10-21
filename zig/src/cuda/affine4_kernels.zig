//! Affine-4 decode, verify and prompt launches over the shared tiled weight layout.
const std = @import("std");
const linear = @import("qlinear.zig");
const Driver = @import("driver.zig").Driver;
const Module = @import("module.zig").Module;
const Function = @import("module.zig").Function;
const Stream = @import("stream.zig").Stream;
const launch_ = @import("launch.zig");
const Affine4 = linear.Affine4;
const Split = linear.Split;
const GemvPart = linear.GemvPart;
const Parts = linear.Parts;
const symbols = linear.symbols;
const splitK = linear.splitK;
const group_smem = linear.group_smem;
const split_smem = linear.split_smem;
const prefill_smem = linear.prefill_smem;
fn int(v: usize) c_int {
    return @intCast(v);
}
fn u(v: usize) u32 {
    return @intCast(v);
}

const affineStride = linear.affineStride;
const AffinePromptTile = linear.AffinePromptTile;
const affinePromptGeometry = linear.affinePromptGeometry;

/// The affine-4 kernels: qmm_group's cluster tile, lane_gemv and its split-K form, qmm_prefill's prompt GEMM.
pub const Affine4Kernels = struct {
    pub const max_rows = 16; // rows decode() takes: its kernels tile 16 rows

    group_fn: Function,
    verify_fn: Function = undefined,
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
        a.verify_fn = try group_mod.function(symbols.verify32);
        try a.verify_fn.allowDynamicShared(linear.verify_smem);
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

    /// Keep legacy devices at 16 rows; GB10 verify shares one weight panel across two m16 row tiles.
    pub fn verifyRows(a: *const Affine4Kernels) usize {
        return if (a.pdl) 32 else max_rows;
    }

    pub fn verify(a: *const Affine4Kernels, s: Stream, x: u64, xs: u64, weight: Affine4, out: u64, rows: usize) !void {
        if (rows == 0 or rows > a.verifyRows()) return error.WindowTooWide;
        if (rows <= max_rows) return a.decode(s, x, xs, weight, out, rows);
        return a.cluster32(s, x, xs, weight, out, rows);
    }

    /// The same ordered group and K-slice chains as decode(), with both row tiles reusing the loaded weights.
    pub fn cluster32(a: *const Affine4Kernels, s: Stream, x: u64, xs: u64, weight: Affine4, out: u64, rows: usize) !void {
        if (rows == 0 or rows > 32) return error.WindowTooWide;
        const tiles = (weight.n + 63) / 64;
        const sk = splitK(weight.n, weight.k);
        var parts: Parts = std.mem.zeroes(Parts);
        parts.count = 1;
        parts.p[0] = .{ .w = weight.w, .scales = weight.s, .biases = weight.b, .out = out, .n = int(weight.n), .npad = int(weight.npad), .sk = int(sk), .tiles = int(tiles), .first = 0 };
        var g: launch_.Args = .{};
        g.add(x);
        g.add(xs);
        g.add(parts);
        for ([_]usize{ rows, weight.k, weight.k, 1, sk }) |v| g.add(int(v));
        try launch_.launch(a.verify_fn, .{ .grid = .{ .x = u(tiles * sk) }, .block = .{ .x = 128 }, .shared = linear.verify_smem, .cluster = if (sk > 1) .{ .x = u(sk) } else null, .pdl = a.pdl }, s, &g);
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
