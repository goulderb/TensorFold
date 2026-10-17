const std = @import("std");
const core = @import("core");
const source = @import("cuda_weight_source.zig");

const Allocator = std.mem.Allocator;
const Span = source.Span;
const Format = core.exlr8.dense.Format;
const Tensor = core.checkpoint.Tensor;

pub const Packed = struct {
    format: u32,
    out0: u32,
    outs: u32,
    in0: u32,
    ins: u32,
    out_total: u32,
    in_total: u32,
    codes: []u8,
    scales: []u8,

    pub fn deinit(p: *Packed, a: Allocator) void {
        a.free(p.codes);
        a.free(p.scales);
        p.* = undefined;
    }

    pub fn job(p: Packed) core.exlr8.job.DenseJob {
        return .{
            .format = p.format,
            .out0 = p.out0,
            .outs = p.outs,
            .in0 = p.in0,
            .ins = p.ins,
            .out_total = p.out_total,
            .in_total = p.in_total,
            .w = 0,
            .s = p.codes.len,
        };
    }

    pub fn resident(p: Packed) usize {
        return p.codes.len + p.scales.len;
    }

    pub fn extend(p: *Packed, a: Allocator, q: Packed) !void {
        if (p.format != q.format or p.ins != q.ins or p.out0 != 0 or p.in0 != 0 or q.out0 != 0 or q.in0 != 0 or p.in_total != p.ins or q.in_total != q.ins) return error.SourceDenseGeometry;
        const row_block: u32 = if (p.format == @intFromEnum(Format.mxfp8)) 1 else 128;
        if (p.outs % row_block != 0) return error.SourceDenseGeometry;
        const codes = try a.alloc(u8, p.codes.len + q.codes.len);
        errdefer a.free(codes);
        @memcpy(codes[0..p.codes.len], p.codes);
        @memcpy(codes[p.codes.len..], q.codes);
        const scales = try a.alloc(u8, p.scales.len + q.scales.len);
        errdefer a.free(scales);
        @memcpy(scales[0..p.scales.len], p.scales);
        @memcpy(scales[p.scales.len..], q.scales);
        a.free(p.codes);
        a.free(p.scales);
        p.codes = codes;
        p.scales = scales;
        p.outs = try fit(p.outs + q.outs);
        p.out_total = p.outs;
    }

    pub fn matrix(p: Packed) !core.exlr8.dense.Matrix {
        const fmt: Format = std.enums.fromInt(Format, @as(u8, @intCast(p.format))) orelse return error.SourceDenseGeometry;
        const n = p.out_total;
        const k = p.in_total;
        var m: core.exlr8.dense.Matrix = .{
            .record = .{
                .format = fmt,
                .out = n,
                .in = k,
                .k = 0,
                .side = .replicated,
                .channels = 0,
                .rotation = 0,
                .solve = 0,
            },
            .data = .{
                t2(.u8, n, k, p.codes),
                switch (fmt) {
                    .mxfp8 => t2(.u8, n, k / 32, p.scales),
                    .fp8_block => t2(.f32, std.math.divCeil(usize, n, 128) catch unreachable, std.math.divCeil(usize, k, 128) catch unreachable, p.scales),
                    else => return error.SourceDenseGeometry,
                },
                undefined,
            },
            .checksum = null,
        };
        try m.check();
        return m;
    }
};

fn t2(dtype: core.checkpoint.DType, d0: usize, d1: usize, bytes: []const u8) Tensor {
    var t: Tensor = .{ .dtype = dtype, .rank = 2, .shape = @splat(1), .bytes = bytes };
    t.shape[0] = d0;
    t.shape[1] = d1;
    return t;
}

fn fit(v: usize) !u32 {
    return std.math.cast(u32, v) orelse error.SourceDenseGeometry;
}

fn checkSpan(s: Span, dim: usize) !void {
    if (s.lo > s.hi or s.hi > dim or s.lo == s.hi) return error.SourceDenseGeometry;
}

fn pbScale(scale: Tensor, n: usize, k: usize, r: usize, c: usize) f32 {
    const kb = std.math.divCeil(usize, k, 128) catch unreachable;
    _ = n;
    const si = (r / 128) * kb + (c / 128);
    return @bitCast(std.mem.readInt(u32, scale.bytes[4 * si ..][0..4], .little));
}

fn writeF32(out: []u8, i: usize, v: f32) void {
    std.mem.writeInt(u32, out[4 * i ..][0..4], @bitCast(v), .little);
}

fn mxLayout(scale: Tensor, n: usize, k: usize) !void {
    if (scale.dtype != .u8 and scale.dtype != .f8_e8m0) return error.BadFp8Source;
    if (k % 32 != 0 or scale.rank != 2 or scale.shape[0] != n or scale.shape[1] != k / 32) return error.BadFp8Source;
    if (scale.bytes.len != n * (k / 32)) return error.BadFp8Source;
}

fn pbLayout(scale: Tensor, n: usize, k: usize) !void {
    if (scale.dtype != .f32) return error.BadFp8Source;
    const nb = std.math.divCeil(usize, n, 128) catch unreachable;
    const kb = std.math.divCeil(usize, k, 128) catch unreachable;
    const ok = switch (scale.rank) {
        2 => scale.shape[0] == nb and scale.shape[1] == kb,
        4 => scale.shape[0] == nb and scale.shape[1] == 1 and scale.shape[2] == kb and scale.shape[3] == 1,
        else => false,
    };
    if (!ok or scale.bytes.len != nb * kb * 4) return error.BadFp8Source;
}

pub fn pack(a: Allocator, s: source.Source, rows: []const Span, cols: Span) !Packed {
    const algo = s.format orelse return error.SourceDenseRequiresFp8;
    const w = s.weight;
    const sc = s.scale orelse return error.SourceDenseRequiresFp8;
    if (w.rank != 2 or w.dtype != .f8_e4m3) return error.SourceDenseRequiresFp8;
    const n = w.dim(0);
    const k = w.dim(1);
    if (n == 0 or k == 0 or w.bytes.len != n * k) return error.BadFp8Source;
    if (cols.lo > cols.hi or cols.hi > k or cols.lo == cols.hi) return error.SourceDenseGeometry;
    if (rows.len == 0) return error.SourceDenseGeometry;
    var outs: usize = 0;
    for (rows) |sp| {
        try checkSpan(sp, n);
        outs += sp.hi - sp.lo;
    }
    return switch (algo) {
        .mxfp8 => packMx(a, w, sc, rows, cols, n, k, outs),
        .fp8_pb_wo => packPb(a, w, sc, rows, cols, n, k, outs),
        else => error.SourceDenseRequiresFp8,
    };
}

fn packMx(a: Allocator, w: Tensor, sc: Tensor, rows: []const Span, cols: Span, n: usize, k: usize, outs: usize) !Packed {
    try mxLayout(sc, n, k);
    if (cols.lo % 32 != 0 or cols.hi % 32 != 0) return error.SourceDenseGeometry;
    const ins = cols.hi - cols.lo;
    const codes = try a.alloc(u8, outs * ins);
    errdefer a.free(codes);
    const scales = try a.alloc(u8, outs * (ins / 32));
    errdefer a.free(scales);
    var at: usize = 0;
    for (rows) |sp| for (sp.lo..sp.hi) |r| {
        @memcpy(codes[at * ins ..][0..ins], w.bytes[r * k + cols.lo ..][0..ins]);
        @memcpy(scales[at * (ins / 32) ..][0..(ins / 32)], sc.bytes[r * (k / 32) + cols.lo / 32 ..][0 .. ins / 32]);
        at += 1;
    };
    return .{
        .format = @intFromEnum(Format.mxfp8),
        .out0 = 0,
        .outs = try fit(outs),
        .in0 = 0,
        .ins = try fit(ins),
        .out_total = try fit(outs),
        .in_total = try fit(ins),
        .codes = codes,
        .scales = scales,
    };
}

fn pbAligned(rows: []const Span, cols: Span) bool {
    if (cols.lo % 128 != 0) return false;
    for (rows) |sp| if (sp.lo % 128 != 0) return false;
    return true;
}

fn packPb(a: Allocator, w: Tensor, sc: Tensor, rows: []const Span, cols: Span, n: usize, k: usize, outs: usize) !Packed {
    try pbLayout(sc, n, k);
    if (pbAligned(rows, cols)) return packPbTight(a, w, sc, rows, cols, n, k, outs);
    if (rows.len != 1) return error.SourceDenseGeometry;
    return packPbRect(a, w, sc, rows[0], cols, n, k);
}

fn packPbTight(a: Allocator, w: Tensor, sc: Tensor, rows: []const Span, cols: Span, n: usize, k: usize, outs: usize) !Packed {
    const ins = cols.hi - cols.lo;
    const codes = try a.alloc(u8, outs * ins);
    errdefer a.free(codes);
    const snb = std.math.divCeil(usize, outs, 128) catch unreachable;
    const skb = std.math.divCeil(usize, ins, 128) catch unreachable;
    const scales = try a.alloc(u8, snb * skb * 4);
    errdefer a.free(scales);
    var at: usize = 0;
    for (rows) |sp| for (sp.lo..sp.hi) |r| {
        @memcpy(codes[at * ins ..][0..ins], w.bytes[r * k + cols.lo ..][0..ins]);
        if (at % 128 == 0) {
            const sr = at / 128;
            for (0..skb) |sc_i| writeF32(scales, sr * skb + sc_i, pbScale(sc, n, k, r, cols.lo + sc_i * 128));
        }
        at += 1;
    };
    return .{
        .format = @intFromEnum(Format.fp8_block),
        .out0 = 0,
        .outs = try fit(outs),
        .in0 = 0,
        .ins = try fit(ins),
        .out_total = try fit(outs),
        .in_total = try fit(ins),
        .codes = codes,
        .scales = scales,
    };
}

fn packPbRect(a: Allocator, w: Tensor, sc: Tensor, rows: Span, cols: Span, n: usize, k: usize) !Packed {
    const row0 = rows.lo / 128 * 128;
    const col0 = cols.lo / 128 * 128;
    const row1 = @min(n, (std.math.divCeil(usize, rows.hi, 128) catch unreachable) * 128);
    const col1 = @min(k, (std.math.divCeil(usize, cols.hi, 128) catch unreachable) * 128);
    const out_total = row1 - row0;
    const in_total = col1 - col0;
    const outs = rows.hi - rows.lo;
    const ins = cols.hi - cols.lo;
    const codes = try a.alloc(u8, out_total * in_total);
    errdefer a.free(codes);
    const snb = std.math.divCeil(usize, out_total, 128) catch unreachable;
    const skb = std.math.divCeil(usize, in_total, 128) catch unreachable;
    const scales = try a.alloc(u8, snb * skb * 4);
    errdefer a.free(scales);
    for (0..out_total) |r| @memcpy(codes[r * in_total ..][0..in_total], w.bytes[(row0 + r) * k + col0 ..][0..in_total]);
    for (0..snb) |sr| for (0..skb) |sc_i| {
        writeF32(scales, sr * skb + sc_i, pbScale(sc, n, k, row0 + sr * 128, col0 + sc_i * 128));
    };
    return .{
        .format = @intFromEnum(Format.fp8_block),
        .out0 = try fit(rows.lo - row0),
        .outs = try fit(outs),
        .in0 = try fit(cols.lo - col0),
        .ins = try fit(ins),
        .out_total = try fit(out_total),
        .in_total = try fit(in_total),
        .codes = codes,
        .scales = scales,
    };
}
