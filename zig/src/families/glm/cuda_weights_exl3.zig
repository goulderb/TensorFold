//! A TP rank's GLM-5.3-Flash weights from an EXL3 checkpoint: matrices cut at 128s, KDA and MoE as EXL3 has them.

const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const shard = @import("cuda_shard.zig");
const W = @import("cuda_weights.zig");
const projections = @import("cuda_weights_projs.zig");
const mia = @import("cuda_experts_mia.zig");

fn putBytes(L: *W.Loader, bytes: []const u8, i: u32, comptime part: []const u8) !u64 {
    try L.host.appendSlice(L.gpa, bytes);
    return L.put("layers.{d}.moe.mia." ++ part, .{i});
}

const MiaMatrix = struct {
    trellis: std.ArrayList(u8) = .empty,
    suh: std.ArrayList(u8) = .empty,
    svh: std.ArrayList(u8) = .empty,

    fn deinit(m: *MiaMatrix, gpa: std.mem.Allocator) void {
        m.trellis.deinit(gpa);
        m.suh.deinit(gpa);
        m.svh.deinit(gpa);
    }

    fn append(m: *MiaMatrix, L: *W.Loader, name: []const u8, klo: usize, khi: usize, spans: []const shard.Span, expected_k: usize, expected_n: usize) !void {
        var halves: std.ArrayList(u8) = .empty;
        defer halves.deinit(L.gpa);
        const q = try gather(L.ck, L.gpa, name, klo, khi, spans, &m.trellis, &halves);
        if (q.bits != 4 or q.codebook != .mcg or q.k != expected_k or q.n != expected_n) return error.UnsupportedMiaWeights;
        try m.appendHalves(L.gpa, halves.items, q.k, q.n);
    }

    fn appendHalves(m: *MiaMatrix, gpa: std.mem.Allocator, halves: []const u8, k: usize, n: usize) !void {
        if (halves.len != 2 * (k + n)) return error.UnexpectedExl3Scales;
        try m.suh.appendSlice(gpa, halves[0 .. k * 2]);
        try m.svh.appendSlice(gpa, halves[k * 2 ..]);
    }
};

fn sharedInputHalves(gate: []const u8, up: []const u8, experts: usize, dims: usize) bool {
    const size = dims * @sizeOf(f16);
    if (experts == 0 or gate.len != experts * size or up.len != experts * size) return false;
    const first = gate[0..size];
    for (0..experts) |expert| {
        const offset = expert * size;
        if (!std.mem.eql(u8, first, gate[offset..][0..size]) or !std.mem.eql(u8, first, up[offset..][0..size])) return false;
    }
    return true;
}

test "Mia shares input Hadamard only when every gate and up half matches bitwise" {
    const first = [_]u8{ 0x00, 0x80, 0x00, 0x00 };
    const same = first ++ first;
    try std.testing.expect(sharedInputHalves(&same, &same, 2, 2));
    const changed = first ++ [_]u8{ 0x00, 0x00, 0x00, 0x00 };
    try std.testing.expect(!sharedInputHalves(&changed, &same, 2, 2));
    try std.testing.expect(!sharedInputHalves(&same, &changed, 2, 2));
    try std.testing.expect(!sharedInputHalves(&same, &first, 2, 2));
}

test "Mia input and output halves remain separate across experts" {
    var gate: MiaMatrix = .{};
    defer gate.deinit(std.testing.allocator);
    var up: MiaMatrix = .{};
    defer up.deinit(std.testing.allocator);
    try gate.appendHalves(std.testing.allocator, &.{ 1, 2, 3, 4, 5, 6 }, 2, 1);
    try up.appendHalves(std.testing.allocator, &.{ 7, 8, 9, 10, 11, 12 }, 2, 1);
    try gate.appendHalves(std.testing.allocator, &.{ 13, 14, 15, 16, 17, 18 }, 2, 1);
    try up.appendHalves(std.testing.allocator, &.{ 19, 20, 21, 22, 23, 24 }, 2, 1);
    try std.testing.expectEqualSlices(u8, &.{ 1, 2, 3, 4, 13, 14, 15, 16 }, gate.suh.items);
    try std.testing.expectEqualSlices(u8, &.{ 5, 6, 17, 18 }, gate.svh.items);
    try std.testing.expectEqualSlices(u8, &.{ 7, 8, 9, 10, 19, 20, 21, 22 }, up.suh.items);
    try std.testing.expectEqualSlices(u8, &.{ 11, 12, 23, 24 }, up.svh.items);
    try std.testing.expectError(error.UnexpectedExl3Scales, gate.appendHalves(std.testing.allocator, &.{ 1, 2 }, 2, 1));
}

fn miaMoe(L: *W.Loader, i: u32) !@import("cuda_moe.zig").Layer {
    const c = &L.w.config;
    const cut = L.w.share.moe;
    const sh = L.w.share.shared;
    try mia.validate(c.hidden, cut.len(), c.experts, c.topk + 1);
    if (L.w.exl3) return error.MiaFullExl3SharedUnsupported;
    try L.whole(try L.get("layers.{d}.mlp.gate.weight", .{i}), .bf16);
    const router = try L.put("layers.{d}.moe.router", .{i});
    try L.whole(try L.get("layers.{d}.mlp.gate.e_score_correction_bias", .{i}), .f32);
    const bias = try L.put("layers.{d}.moe.bias", .{i});
    var mats: [3]MiaMatrix = .{ .{}, .{}, .{} };
    defer for (&mats) |*m| m.deinit(L.gpa);
    var nb: [192]u8 = undefined;
    for (0..c.experts) |e| {
        inline for (.{ "gate_proj", "up_proj", "down_proj" }, 0..) |part, j| {
            const name = try std.fmt.bufPrint(&nb, W.prefix ++ "layers.{d}.mlp.experts.{d}.{s}", .{ i, e, part });
            if (j < 2) try mats[j].append(L, name, 0, 0, &.{cut}, c.hidden, cut.len()) else try mats[j].append(L, name, cut.lo, cut.hi, &.{.{ .lo = 0, .hi = c.hidden }}, cut.len(), c.hidden);
        }
    }
    const shared_suh = sharedInputHalves(mats[0].suh.items, mats[1].suh.items, c.experts, c.hidden);
    var weights: mia.Experts = .{
        .gate_trellis = try putBytes(L, mats[0].trellis.items, i, "gate_trellis"),
        .up_trellis = try putBytes(L, mats[1].trellis.items, i, "up_trellis"),
        .down_trellis = try putBytes(L, mats[2].trellis.items, i, "down_trellis"),
        .gate_suh = try putBytes(L, mats[0].suh.items, i, "gate_suh"),
        .up_suh = try putBytes(L, mats[1].suh.items, i, "up_suh"),
        .down_suh = try putBytes(L, mats[2].suh.items, i, "down_suh"),
        .gate_svh = try putBytes(L, mats[0].svh.items, i, "gate_svh"),
        .up_svh = try putBytes(L, mats[1].svh.items, i, "up_svh"),
        .down_svh = try putBytes(L, mats[2].svh.items, i, "down_svh"),
        .dims = c.hidden,
        .width = cut.len(),
        .count = c.experts,
    };
    if (shared_suh) weights.shared_suh = weights.gate_suh;
    inline for (.{ "gate_proj", "up_proj" }) |proj| try L.rows(try L.get("layers.{d}.mlp.shared_experts." ++ proj ++ ".weight", .{i}), sh.lo, sh.hi, c.shared_inter, .bf16);
    const gu = try projections.finish(L, .shared, "layers.{d}.moe.shared.gu", .{i}, 2 * sh.len(), c.hidden);
    try L.cols(try L.get("layers.{d}.mlp.shared_experts.down_proj.weight", .{i}), sh.lo, sh.hi, .bf16);
    const down = try projections.finish(L, .shared, "layers.{d}.moe.shared.down", .{i}, c.hidden, sh.len());
    return .{
        .router = router,
        .bias = bias,
        .routed = .{ .format = .exl3, .up = 0, .down = 0, .width = cut.len(), .dims = c.hidden, .experts = c.experts, .limit = c.swiglu_limit },
        .mia_routed = weights,
        .count = c.experts,
        .shared_gu = gu,
        .shared_down = down,
        .shared = sh.len(),
        .top_k = c.topk,
        .scale = c.routed_scale,
        .norm = c.norm_topk,
    };
}

/// EXL3 matrix `name`'s share: inputs [klo, khi) (khi 0: all), `spans`' columns; tiles to `tiles`, suh+svh `scales`.
pub fn gather(ck: *core.Checkpoint, gpa: std.mem.Allocator, name: []const u8, klo: usize, khi: usize, spans: []const shard.Span, tiles: *std.ArrayList(u8), scales: *std.ArrayList(u8)) !cuda.exl3_linear.Matrix {
    var nb: [192]u8 = undefined;
    const t = try ck.get(try std.fmt.bufPrint(&nb, "{s}.trellis", .{name}));
    const suh = try ck.get(try std.fmt.bufPrint(&nb, "{s}.suh", .{name}));
    const svh = try ck.get(try std.fmt.bufPrint(&nb, "{s}.svh", .{name}));
    const book: cuda.exl3_linear.Codebook = if (ck.has(try std.fmt.bufPrint(&nb, "{s}.mul1", .{name}))) .mul1 else if (ck.has(try std.fmt.bufPrint(&nb, "{s}.mcg", .{name}))) .mcg else .inst3;
    if (t.dtype != .i16 or t.rank != 3 or t.dim(2) % 16 != 0) return error.UnexpectedTensor;
    const tn = t.dim(1);
    const words = t.dim(2);
    const k_all = t.dim(0) * 16;
    if (suh.dtype != .f16 or suh.rank != 1 or suh.dim(0) != k_all or svh.dtype != .f16 or svh.rank != 1 or svh.dim(0) != tn * 16) return error.UnexpectedExl3Scales;
    const k_hi = if (khi == 0) k_all else khi;
    if (klo % 128 != 0 or k_hi % 128 != 0 or k_hi > k_all) return error.Exl3RowCut;
    var n: usize = 0;
    for (spans) |c| {
        if (c.lo % 128 != 0 or c.hi % 128 != 0 or c.hi > tn * 16) return error.Exl3ColumnCut;
        n += c.len();
    }
    // trellis tiles: each 16-row group of the rank's inputs, its spans' 16-column tiles in order
    for (klo / 16..k_hi / 16) |r| for (spans) |c| {
        try tiles.appendSlice(gpa, t.bytes[(r * tn + c.lo / 16) * words * 2 ..][0 .. c.len() / 16 * words * 2]);
    };
    try scales.appendSlice(gpa, suh.bytes[klo * 2 .. k_hi * 2]);
    for (spans) |c| try scales.appendSlice(gpa, svh.bytes[c.lo * 2 .. c.hi * 2]);
    return .{ .trellis = 0, .suh = 0, .svh = 0, .k = k_hi - klo, .n = n, .bits = @intCast(words / 16), .codebook = book };
}

/// gather() of one matrix, uploaded as its trellis, suh and svh.
pub fn matrix(L: *W.Loader, ck: *core.Checkpoint, name: []const u8, klo: usize, khi: usize, spans: []const shard.Span) !cuda.qlinear.Weight {
    var scales: std.ArrayList(u8) = .empty;
    defer scales.deinit(L.gpa);
    var m = try gather(ck, L.gpa, name, klo, khi, spans, &L.host, &scales);
    m.trellis = try L.put("{s}.trellis", .{name});
    try L.host.appendSlice(L.gpa, scales.items[0 .. m.k * 2]);
    m.suh = try L.put("{s}.suh", .{name});
    try L.host.appendSlice(L.gpa, scales.items[m.k * 2 ..]);
    m.svh = try L.put("{s}.svh", .{name});
    return .{ .exl3 = m };
}

/// matrix() on the model's tensors, `fmt` naming the prefix of layer `i`.
pub fn layerMatrix(L: *W.Loader, comptime fmt: []const u8, i: u32, klo: usize, khi: usize, spans: []const shard.Span) !cuda.qlinear.Weight {
    var nb: [192]u8 = undefined;
    return matrix(L, L.ck, try std.fmt.bufPrint(&nb, W.prefix ++ fmt, .{i}), klo, khi, spans);
}

/// An EXL3 matrix's every column.
pub fn whole(L: *W.Loader, comptime fmt: []const u8, i: u32) !cuda.qlinear.Weight {
    var nb: [192]u8 = undefined;
    const name = try std.fmt.bufPrint(&nb, W.prefix ++ fmt, .{i});
    var nb2: [192]u8 = undefined;
    const t = try L.ck.get(try std.fmt.bufPrint(&nb2, "{s}.trellis", .{name}));
    return matrix(L, L.ck, name, 0, 0, &.{.{ .lo = 0, .hi = t.dim(1) * 16 }});
}

/// Loader.kda on an EXL3 checkpoint: qkv_proj by heads, [f_a | g_a | b] BF16, conv1d's q, k and v rows, o by inputs.
pub fn kda(L: *W.Loader, i: u32) !@import("cuda_kda_block.zig").Layer {
    const c = &L.w.config;
    const h = L.w.share.kda;
    const dk = c.kda_dim;
    const all = c.kda_heads * dk;
    const lo = h.lo * dk;
    const hi = h.hi * dk;
    const qkv = try layerMatrix(L, "layers.{d}.self_attn.qkv_proj", i, 0, 0, &.{ .{ .lo = lo, .hi = hi }, .{ .lo = all + lo, .hi = all + hi }, .{ .lo = 2 * all + lo, .hi = 2 * all + hi } });
    inline for (.{ "f_a", "g_a" }) |x| try L.whole(try L.get("layers.{d}.self_attn." ++ x ++ "_proj.weight", .{i}), .bf16);
    try L.rows(try L.get("layers.{d}.self_attn.b_proj.weight", .{i}), h.lo, h.hi, c.kda_heads, .bf16);
    const proj = try L.put("layers.{d}.kda.proj", .{i});
    try L.rows(try L.get("layers.{d}.self_attn.f_b_proj.weight", .{i}), lo, hi, all, .bf16);
    const fb = try L.put("layers.{d}.kda.fb", .{i});
    try L.rows(try L.get("layers.{d}.self_attn.g_b_proj.weight", .{i}), lo, hi, all, .bf16);
    const gb = try L.put("layers.{d}.kda.gb", .{i});
    const conv = try L.get("layers.{d}.self_attn.conv1d.weight", .{i});
    for (0..3) |j| try L.rows(conv, j * all + lo, j * all + hi, 3 * all, .f32);
    const conv_w = try L.put("layers.{d}.kda.conv", .{i});
    try L.rows(try L.get("layers.{d}.self_attn.A_log", .{i}), h.lo, h.hi, c.kda_heads, .f32);
    const a_log = try L.put("layers.{d}.kda.a_log", .{i});
    try L.rows(try L.get("layers.{d}.self_attn.dt_bias", .{i}), lo, hi, all, .f32);
    const dt = try L.put("layers.{d}.kda.dt_bias", .{i});
    try L.whole(try L.get("layers.{d}.self_attn.o_norm.weight", .{i}), .bf16);
    const norm = try L.put("layers.{d}.kda.norm", .{i});
    const o = try layerMatrix(L, "layers.{d}.self_attn.o_proj", i, lo, hi, &.{.{ .lo = 0, .hi = c.hidden }});
    return .{ .proj = .{ .b16 = proj }, .qkv = qkv, .fb = .{ .b16 = fb }, .gb = .{ .b16 = gb }, .o = .{ .q = o }, .conv_w = conv_w, .a_log = a_log, .dt_bias = dt, .norm_w = norm, .eps = c.eps, .lower = c.lower_bound, .heads = h.len(), .dims = c.hidden };
}

/// Loader.moeLayer on EXL3: experts as exl3_experts stacks, gate and up by the rank's columns, down by its inputs.
pub fn moe(L: *W.Loader, i: u32) !@import("cuda_moe.zig").Layer {
    if (L.tools.exl3_routed == .mia_mcg4) return miaMoe(L, i);
    const c = &L.w.config;
    const d: usize = c.hidden;
    const cut = L.w.share.moe;
    try L.whole(try L.get("layers.{d}.mlp.gate.weight", .{i}), .bf16);
    const router = try L.put("layers.{d}.moe.router", .{i});
    try L.whole(try L.get("layers.{d}.mlp.gate.e_score_correction_bias", .{i}), .f32);
    const bias = try L.put("layers.{d}.moe.bias", .{i});
    var t_up: std.ArrayList(u8) = .empty;
    defer t_up.deinit(L.gpa);
    var h_up: std.ArrayList(u8) = .empty;
    defer h_up.deinit(L.gpa);
    var t_dn: std.ArrayList(u8) = .empty;
    defer t_dn.deinit(L.gpa);
    var h_dn: std.ArrayList(u8) = .empty;
    defer h_dn.deinit(L.gpa);
    var widths: cuda.exl3_experts.Widths = .{};
    var nb: [192]u8 = undefined;
    for (0..c.experts) |e| {
        for ([_][]const u8{ "gate_proj", "up_proj", "down_proj" }, 0..) |p, m| {
            const name = try std.fmt.bufPrint(&nb, W.prefix ++ "layers.{d}.mlp.experts.{d}.{s}", .{ i, e, p });
            const q = if (m < 2) try gather(L.ck, L.gpa, name, 0, 0, &.{cut}, &t_up, &h_up) else try gather(L.ck, L.gpa, name, cut.lo, cut.hi, &.{.{ .lo = 0, .hi = d }}, &t_dn, &h_dn);
            if (e > 0 or m > 0) {
                if (q.codebook != widths.codebook or (m < 2 and q.bits != widths.up) or (m == 2 and e > 0 and q.bits != widths.down)) return error.InconsistentExl3Experts;
            }
            if (m < 2) widths.up = q.bits else widths.down = q.bits;
            widths.codebook = q.codebook;
        }
    }
    var ptrs: [4]u64 = undefined;
    for (&ptrs, [_][]const u8{ t_up.items, t_dn.items, h_up.items, h_dn.items }, [_][]const u8{ "up", "down", "up_scale", "down_scale" }) |*p, bytes, what| {
        try L.host.appendSlice(L.gpa, bytes);
        p.* = try L.put("layers.{d}.moe.experts.{s}", .{ i, what });
    }
    const sh = L.w.share.shared;
    if (!L.w.exl3) {
        inline for (.{ "gate_proj", "up_proj" }) |proj| try L.rows(try L.get("layers.{d}.mlp.shared_experts." ++ proj ++ ".weight", .{i}), sh.lo, sh.hi, c.shared_inter, .bf16);
        const gu = try projections.finish(L, .shared, "layers.{d}.moe.shared.gu", .{i}, 2 * sh.len(), d);
        try L.cols(try L.get("layers.{d}.mlp.shared_experts.down_proj.weight", .{i}), sh.lo, sh.hi, .bf16);
        const down = try projections.finish(L, .shared, "layers.{d}.moe.shared.down", .{i}, d, sh.len());
        return .{
            .router = router,
            .bias = bias,
            .routed = .{ .format = .exl3, .up = ptrs[0], .down = ptrs[1], .up_scale = ptrs[2], .down_scale = ptrs[3], .width = cut.len(), .dims = d, .experts = c.experts, .limit = c.swiglu_limit, .exl3 = widths },
            .count = c.experts,
            .shared_gu = gu,
            .shared_down = down,
            .shared = sh.len(),
            .top_k = c.topk,
            .scale = c.routed_scale,
            .norm = c.norm_topk,
        };
    }
    return .{
        .router = router,
        .bias = bias,
        .routed = .{ .format = .exl3, .up = ptrs[0], .down = ptrs[1], .up_scale = ptrs[2], .down_scale = ptrs[3], .width = cut.len(), .dims = d, .experts = c.experts, .limit = c.swiglu_limit, .exl3 = widths },
        .count = c.experts,
        .shared_gu = .{ .b16 = 0 },
        .shared_gate = try layerMatrix(L, "layers.{d}.mlp.shared_experts.gate_proj", i, 0, 0, &.{sh}),
        .shared_up = try layerMatrix(L, "layers.{d}.mlp.shared_experts.up_proj", i, 0, 0, &.{sh}),
        .shared_down = .{ .q = try layerMatrix(L, "layers.{d}.mlp.shared_experts.down_proj", i, sh.lo, sh.hi, &.{.{ .lo = 0, .hi = d }}) },
        .shared = sh.len(),
        .top_k = c.topk,
        .scale = c.routed_scale,
        .norm = c.norm_topk,
    };
}
