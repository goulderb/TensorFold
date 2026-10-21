//! A TP rank's GLM-5.3-Flash weights on CUDA (the Python family's weights.load, NVFP4), named by its paths for digests.

const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const cfg = @import("config.zig");
const shard = @import("cuda_shard.zig");
const X = cuda.experts.Nvfp4Experts;
const kda_block = @import("cuda_kda_block.zig");
const dsa_block = @import("cuda_dsa_block.zig");
const mlp = @import("cuda_mlp.zig");
const moe = @import("cuda_moe.zig");
const quant = @import("cuda_quant.zig");
const q4 = @import("cuda_q4.zig");
const exl3 = @import("cuda_exl3.zig");
const x3 = @import("cuda_weights_exl3.zig");
const precision = @import("cuda_weight_precision.zig");
pub const projections = @import("cuda_weights_projs.zig");
const mm_mod = @import("cuda_mm.zig");
pub const formats = @import("cuda_formats.zig");
const projection = @import("cuda_proj.zig");
const glue = @import("cuda_glue.zig");

const Tensor = core.checkpoint.Tensor;
pub const prefix = "model.language_model.";

/// A hyper-connection site: fn bf16 [s (s + 2)][s d], base and scale widened to fp32.
pub const Hc = struct { fn_w: u64, base: u64, scale: u64 };

pub const Mixer = union(cfg.Kind) { kda: kda_block.Layer, mla: dsa_block.Layer };
pub const Ffn = union(enum) { dense: mlp.Layer, moe: moe.Layer };

pub const Layer = struct { index: u32, attn_hc: Hc, ffn_hc: Hc, in_norm: glue.Norm, post_norm: glue.Norm, mixer: Mixer, ffn: Ffn };

/// The MTP head (weights.MTPW): norms, eh_proj, a DSA + MoE layer (NVFP4 experts), shared_head.norm, 4-bit draft head.
pub const Mtp = struct { enorm: u64, hnorm: u64, eh: projection.Proj, norm: u64, layer: Layer, draft: ?q4.Q4 };

/// The kernels a load runs: affine4_pack's tf_pack_dense, the FP8 lane, and the MTP head's quantizers and 4-bit tiling.
pub const Tools = struct { recipes: ?*const core.modelopt.Recipes = null, precision: precision.Settings = .{}, exl3_routed: @import("cuda_experts_mia.zig").Mode = .generic, affine4: ?*const cuda.qlinear.Affine4Kernels = null, modelopt: ?*const cuda.modelopt.Packer = null, ck: ?*const cuda.ck.Ck = null, pack_dense: cuda.Function, lane: ?*const cuda.qmmf.Lane = null, quant: ?*const quant.Kernels = null, q4: ?*const q4.Kernels = null };

/// A loaded tensor by the Python family's attribute path, for digests against its weights.
pub const Named = struct { name: []u8, ptr: u64, len: usize };

pub const Weights = struct {
    gpa: std.mem.Allocator,
    config: cfg.Config,
    share: shard.Share,
    embed: u64 = 0, // bf16 [vocab][dims]
    layers: []Layer = &.{},
    precision: precision.Settings = .{},
    native_conversions: [precision.groups.len]u32 = @splat(0),
    exl3: bool = false,
    norm: glue.Norm = .{ .bf16 = 0 },
    head: projection.Proj = .{ .bf16 = 0 }, // [share.vocab][dims]: bf16, or MXFP8 for the lane
    mtp: ?Mtp = null,
    exl3_shx: bool = true, // EXL3: every layer's gate and up share one input rotation (a prompt rotates each row once)
    buffers: std.ArrayList(cuda.DeviceBuffer) = .empty,
    named: std.ArrayList(Named) = .empty,

    pub fn deinit(w: *Weights) void {
        for (w.buffers.items) |*b| b.free();
        for (w.named.items) |n| w.gpa.free(n.name);
        w.buffers.deinit(w.gpa);
        w.named.deinit(w.gpa);
        w.gpa.free(w.layers);
        w.* = undefined;
    }

    pub fn quantizedAttention(w: *const Weights) bool {
        return w.exl3;
    }

    /// The upload named `name` (the Python family's attribute path), or null.
    pub fn find(w: *const Weights, name: []const u8) ?Named {
        for (w.named.items) |n| if (std.mem.eql(u8, n.name, name)) return n;
        return null;
    }
};

/// Host bf16 from fp32: round to nearest even (torch's float -> bfloat16), NaN kept quiet.
fn toBf16(v: f32) u16 {
    const b: u32 = @bitCast(v);
    if (std.math.isNan(v)) return @intCast((b >> 16) | 0x40);
    return @intCast((b + 0x7fff + ((b >> 16) & 1)) >> 16);
}

/// A float32 narrowed to bf16 by the loader, or null when bf16 does not hold it exactly (NaN stays NaN).
fn narrow(v: f32) ?u16 {
    const h = toBf16(v);
    return if (std.math.isNan(v) or @as(f32, @bitCast(@as(u32, h) << 16)) == v) h else null;
}

pub const Loader = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    stream: cuda.Stream,
    pack_dense: cuda.Function,
    tools: Tools,
    w: *Weights,
    ck: *core.Checkpoint,
    host: std.ArrayList(u8) = .empty,
    name_buf: [192]u8 = undefined,
    last: []const u8 = "", // the tensor `get` read last, named when its values are refused
    recipes: ?*const core.modelopt.Recipes = null, // MIXED_PRECISION's table; null: the tensors decide (NVFP4)

    pub fn get(L: *Loader, comptime fmt: []const u8, args: anytype) !Tensor {
        L.last = try std.fmt.bufPrint(&L.name_buf, prefix ++ fmt, args);
        return L.ck.get(L.last);
    }

    /// Hidden-size norm `fmt` uploaded as `up`: bf16 as stored, or kept float32 for the _f32w kernels.
    pub fn norm(L: *Loader, comptime fmt: []const u8, args: anytype, comptime up: []const u8) !glue.Norm {
        const t = try L.get(fmt, args);
        const kept = t.dtype == .f32;
        if (kept and t.numel() != L.w.config.hidden) return error.UnexpectedTensor;
        try L.whole(t, if (kept) .f32 else .bf16);
        const w = try L.put(up, args);
        return if (kept) .{ .f32 = w } else .{ .bf16 = w };
    }

    /// The projection `fmt` (a module under the prefix) is stored as `want`, or it is refused by name (formats.of).
    fn format(L: *Loader, comptime fmt: []const u8, args: anytype, want: formats.Format) !void {
        var buf: [192]u8 = undefined;
        return L.formatIn(L.ck, try std.fmt.bufPrint(&buf, prefix ++ fmt, args), want);
    }

    pub fn formatIn(L: *Loader, ck: *core.Checkpoint, module: []const u8, want: formats.Format) !void {
        _ = try L.formatOne(ck, module, &.{want});
    }

    /// `module`'s stored format (formats.of) when it is one of `wants`; any other is refused by name.
    pub fn formatOne(L: *Loader, ck: *core.Checkpoint, module: []const u8, wants: []const formats.Format) !formats.Format {
        var buf: [224]u8 = undefined;
        const w = try ck.get(try std.fmt.bufPrint(&buf, "{s}.weight", .{module}));
        const scale_name = try std.fmt.bufPrint(&buf, "{s}.weight_scale", .{module});
        const s: ?Tensor = if (ck.has(scale_name)) try ck.get(scale_name) else null;
        const stored: formats.Stored = .{
            .weight = .{ .dtype = w.dtype, .shape = w.shape[0..w.rank] },
            .scale = if (s) |*t| .{ .dtype = t.dtype, .shape = t.shape[0..t.rank] } else null,
            .csf = ck.has(try std.fmt.bufPrint(&buf, "{s}.weight_scale.csf32_bases", .{module})),
        };
        const declared = L.declaredAs(module);
        const got = formats.of(declared, stored);
        if (got) |f| {
            for (wants) |want| if (f == want) return f;
        } else |_| {}
        var why: [320]u8 = undefined;
        const said = formats.describe(&why, module, declared, stored);
        if (got) |_| {
            std.log.err("{s}; this loader reads it as {any}", .{ said, wants });
        } else |e| std.log.err("{s} ({t}); this loader reads it as {any}", .{ said, e, wants });
        return error.UnreadableFormat;
    }

    /// What config.json declares `module` to be: its MIXED_PRECISION recipe, or NVFP4 for every quantized linear.
    pub fn declaredAs(L: *const Loader, module: []const u8) formats.Declared {
        return if (L.recipes) |r| .{ .mixed = r.of(module) } else .nvfp4;
    }

    /// The host bytes gathered so far uploaded as one tensor named `fmt`.
    pub fn put(L: *Loader, comptime fmt: []const u8, args: anytype) !u64 {
        var b = try cuda.DeviceBuffer.fromHost(L.d, L.host.items);
        errdefer b.free();
        const name = try std.fmt.allocPrint(L.gpa, fmt, args);
        errdefer L.gpa.free(name);
        try L.w.buffers.ensureUnusedCapacity(L.gpa, 1);
        try L.w.named.append(L.gpa, .{ .name = name, .ptr = b.ptr, .len = b.len });
        L.w.buffers.appendAssumeCapacity(b);
        L.host.clearRetainingCapacity();
        return b.ptr;
    }

    /// Elements [lo, hi) of `t` flat as `to` (bf16 or fp32 from a bf16 or fp32 tensor; other dtypes only as stored).
    pub fn add(L: *Loader, t: Tensor, lo: usize, hi: usize, to: core.checkpoint.DType) !void {
        if (t.dtype == to) {
            const size = t.bytes.len / t.numel();
            return L.host.appendSlice(L.gpa, t.bytes[lo * size .. hi * size]);
        }
        for (lo..hi) |i| {
            const v: f32 = switch (t.dtype) {
                .bf16 => @bitCast(@as(u32, std.mem.readInt(u16, t.bytes[2 * i ..][0..2], .little)) << 16),
                .f32 => std.mem.bytesToValue(f32, t.bytes[4 * i ..][0..4]),
                else => return error.UnexpectedTensor,
            };
            switch (to) {
                .f32 => try L.host.appendSlice(L.gpa, std.mem.asBytes(&v)),
                .bf16 => try L.host.appendSlice(L.gpa, std.mem.asBytes(&(narrow(v) orelse {
                    std.log.err("{s}: a float32 value bf16 does not hold, and its kernels read bf16", .{L.last});
                    return error.InexactBf16;
                }))),
                else => return error.UnexpectedTensor,
            }
        }
    }

    /// Rows [lo, hi) of `t`'s outermost axis (`total` long; a vector's elements): split.py's "row" cut.
    pub fn rows(L: *Loader, t: Tensor, lo: usize, hi: usize, total: usize, to: core.checkpoint.DType) !void {
        const per = t.numel() / total;
        if (per * total != t.numel()) return error.UnexpectedTensor;
        try L.add(t, lo * per, hi * per, to);
    }

    pub fn whole(L: *Loader, t: Tensor, to: core.checkpoint.DType) !void {
        try L.add(t, 0, t.numel(), to);
    }

    /// Columns [lo, hi) of every row of a matrix: split.py's "col" cut.
    pub fn cols(L: *Loader, t: Tensor, lo: usize, hi: usize, to: core.checkpoint.DType) !void {
        if (t.rank != 2) return error.UnexpectedTensor;
        const k = t.dim(1);
        for (0..t.dim(0)) |r| try L.add(t, r * k + lo, r * k + hi, to);
    }

    fn hc(L: *Loader, i: u32, comptime site: []const u8) !Hc {
        try L.whole(try L.get("layers.{d}.hc_" ++ site ++ "_fn", .{i}), .bf16);
        const f = try L.put("layers.{d}.{s}_hc.fn", .{ i, site });
        try L.whole(try L.get("layers.{d}.hc_" ++ site ++ "_base", .{i}), .f32);
        const base = try L.put("layers.{d}.{s}_hc.base", .{ i, site });
        try L.whole(try L.get("layers.{d}.hc_" ++ site ++ "_scale", .{i}), .f32);
        return .{ .fn_w = f, .base = base, .scale = try L.put("layers.{d}.{s}_hc.scale", .{ i, site }) };
    }

    /// These bytes came through the exact BF16 gather, not a quantized-source materializer.
    fn finishBf16(L: *Loader, group: precision.Group, comptime name: []const u8, args: anytype, n: usize, k: usize) !projection.Proj {
        if (L.tools.precision.dense.get(group) == .source) return .{ .bf16 = try L.put(name, args) };
        return projections.finish(L, group, name, args, n, k);
    }

    /// weights.load's kda(i): [q | k | v | f_a | g_a | b], f_b/g_b by heads, o by columns, conv taps, A_log, dt_bias.
    fn kda(L: *Loader, i: u32) !kda_block.Layer {
        if (L.w.exl3) return x3.kda(L, i);
        const c = &L.w.config;
        const h = L.w.share.kda;
        const dk = c.kda_dim;
        const all = c.kda_heads * dk;
        const mx = (try L.get("layers.{d}.self_attn.q_proj.weight", .{i})).dtype == .f8_e4m3; // FP4-Local's q, k, v and o
        inline for (.{ "q", "k", "v" }) |x| try L.format("layers.{d}.self_attn." ++ x ++ "_proj", .{i}, if (mx) .mxfp8 else .bf16);
        const hr: [2]usize = .{ h.lo * dk, h.hi * dk }; // the rank's heads' rows
        var qkv: ?projection.Proj = null; // [q | k | v] as one lane weight, then the stack's bf16 rest
        if (mx) {
            const parts = [_]formats.Part{ .{ .rel = "self_attn.q_proj", .rows = hr }, .{ .rel = "self_attn.k_proj", .rows = hr }, .{ .rel = "self_attn.v_proj", .rows = hr } };
            qkv = try projections.cut(L, .kda, prefix, i, &parts, &.{.mxfp8}, "layers.{d}.kda.qkv");
        } else inline for (.{ "q", "k", "v" }) |x| try L.rows(try L.get("layers.{d}.self_attn." ++ x ++ "_proj.weight", .{i}), hr[0], hr[1], all, .bf16);
        inline for (.{ "f_a", "g_a" }) |x| try L.whole(try L.get("layers.{d}.self_attn." ++ x ++ "_proj.weight", .{i}), .bf16);
        try L.rows(try L.get("layers.{d}.self_attn.b_proj.weight", .{i}), h.lo, h.hi, c.kda_heads, .bf16);
        const proj = if (mx) try L.finishBf16(.kda, "layers.{d}.kda.rest", .{i}, 2 * dk + h.len(), c.hidden) else try L.finishBf16(.kda, "layers.{d}.kda.proj", .{i}, 3 * h.len() * dk + 2 * dk + h.len(), c.hidden);
        try L.rows(try L.get("layers.{d}.self_attn.f_b_proj.weight", .{i}), h.lo * dk, h.hi * dk, all, .bf16);
        const fb = try L.finishBf16(.kda, "layers.{d}.kda.fb", .{i}, h.len() * dk, dk);
        try L.rows(try L.get("layers.{d}.self_attn.g_b_proj.weight", .{i}), h.lo * dk, h.hi * dk, all, .bf16);
        const gb = try L.finishBf16(.kda, "layers.{d}.kda.gb", .{i}, h.len() * dk, dk);
        inline for (.{ "q", "k", "v" }) |x| try L.rows(try L.get("layers.{d}.self_attn." ++ x ++ "_conv1d.weight", .{i}), h.lo * dk, h.hi * dk, all, .f32);
        const conv = try L.put("layers.{d}.kda.conv", .{i});
        inline for (.{ "q", "k", "v" }) |x| try L.rows(try L.get("layers.{d}.self_attn." ++ x ++ "_conv1d.weight", .{i}), h.lo * dk, h.hi * dk, all, .bf16);
        const prompt_conv = try L.put("layers.{d}.kda.prompt_conv", .{i});
        try L.rows(try L.get("layers.{d}.self_attn.A_log", .{i}), h.lo, h.hi, c.kda_heads, .f32);
        const a_log = try L.put("layers.{d}.kda.a_log", .{i});
        try L.rows(try L.get("layers.{d}.self_attn.dt_bias", .{i}), h.lo * dk, h.hi * dk, all, .f32);
        const dt = try L.put("layers.{d}.kda.dt_bias", .{i});
        try L.whole(try L.get("layers.{d}.self_attn.o_norm.weight", .{i}), .bf16);
        const o_norm = try L.put("layers.{d}.kda.norm", .{i});
        const o = try projections.cut(L, .kda, prefix, i, &.{.{ .rel = "self_attn.o_proj", .cols = hr }}, &.{ .bf16, .mxfp8 }, "layers.{d}.kda.o");
        return .{ .proj = proj, .qkv = qkv, .fb = fb, .gb = gb, .o = o, .conv_w = conv, .prompt_conv_w = prompt_conv, .a_log = a_log, .dt_bias = dt, .norm_w = o_norm, .eps = c.eps, .lower = c.lower_bound, .heads = h.len(), .dims = c.hidden };
    }

    /// weights.load's dsa(i) (latent): [q_a | kv_a], norms, q_b by heads, kv_b as wk/wv (AbsorbW), o by cols, indexer.
    pub fn dsa(L: *Loader, i: u32) !dsa_block.Layer {
        const c = &L.w.config;
        const h = L.w.share.mla;
        const qk = c.nope;
        // the MTP layer's may be MXFP8 (FP4-Local's): on the lane as stored, kv_b widened exactly for its halves
        const mtp = i == c.layers;
        const wants: []const formats.Format = if (mtp) &.{ .bf16, .fp8_pb, .mxfp8 } else &.{ .bf16, .fp8_pb };
        var kb: [96]u8 = undefined;
        const kv_name = try std.fmt.bufPrint(&kb, prefix ++ "layers.{d}.self_attn.kv_b_proj", .{i});
        const kv_mx = mtp and (try L.get("layers.{d}.self_attn.kv_b_proj.weight", .{i})).dtype == .f8_e4m3;
        try L.format("layers.{d}.self_attn.kv_b_proj", .{i}, if (kv_mx) .mxfp8 else .bf16);
        const kv_wide: ?formats.mxfp8.Mxfp8 = if (kv_mx) try formats.mxfp8.Mxfp8.read(L.ck, kv_name) else null;
        const input: dsa_block.InputProjection = if (L.w.exl3) .{ .separate = .{
            .qa = .{ .q = try x3.whole(L, "layers.{d}.self_attn.q_a_proj", i) },
            .kva = .{ .q = try x3.whole(L, "layers.{d}.self_attn.kv_a_proj_with_mqa", i) },
        } } else if (L.tools.precision.dense.get(.dsa) == .source) .{ .separate = .{
            .qa = try projections.cut(L, .dsa, prefix, i, &.{.{ .rel = "self_attn.q_a_proj" }}, wants, "layers.{d}.dsa.qa"),
            .kva = try projections.cut(L, .dsa, prefix, i, &.{.{ .rel = "self_attn.kv_a_proj_with_mqa" }}, wants, "layers.{d}.dsa.kva"),
        } } else .{ .combined = try projections.cut(L, .dsa, prefix, i, &.{ .{ .rel = "self_attn.q_a_proj" }, .{ .rel = "self_attn.kv_a_proj_with_mqa" } }, wants, "layers.{d}.dsa.proj") };
        try L.whole(try L.get("layers.{d}.self_attn.q_a_layernorm.weight", .{i}), .bf16);
        const q_norm = try L.put("layers.{d}.dsa.q_norm", .{i});
        try L.whole(try L.get("layers.{d}.self_attn.kv_a_layernorm.weight", .{i}), .bf16);
        const kv_norm = try L.put("layers.{d}.dsa.kv_norm", .{i});
        const q_b: projection.Proj = if (L.w.exl3) .{ .q = try x3.layerMatrix(L, "layers.{d}.self_attn.q_b_proj", i, 0, 0, &.{.{ .lo = h.lo * qk, .hi = h.hi * qk }}) } else try projections.cut(L, .dsa, prefix, i, &.{.{ .rel = "self_attn.q_b_proj", .rows = .{ h.lo * qk, h.hi * qk } }}, wants, "layers.{d}.dsa.q_b");
        const kv_b = try L.get("layers.{d}.self_attn.kv_b_proj.weight", .{i});
        const per = qk + c.v_dim; // a head's kv_b rows: its keys, then its values
        const total = c.mla_heads * per;
        for (h.lo..h.hi) |head| try absorbRows(L, kv_wide, kv_b, kv_name, head * per, head * per + qk, total);
        const wk = try L.put("layers.{d}.dsa.absorb.wk", .{i});
        for (h.lo..h.hi) |head| try absorbRows(L, kv_wide, kv_b, kv_name, head * per + qk, (head + 1) * per, total);
        const wv = try L.put("layers.{d}.dsa.absorb.wv", .{i});
        const o: projection.Proj = if (L.w.exl3) .{ .q = try x3.layerMatrix(L, "layers.{d}.self_attn.o_proj", i, h.lo * c.v_dim, h.hi * c.v_dim, &.{.{ .lo = 0, .hi = c.hidden }}) } else try projections.cut(L, .dsa, prefix, i, &.{.{ .rel = "self_attn.o_proj", .cols = .{ h.lo * c.v_dim, h.hi * c.v_dim } }}, wants, "layers.{d}.dsa.o");
        const iw: []const formats.Format = if (mtp) &.{ .bf16, .mxfp8 } else &.{.bf16};
        const kw = try projections.cut(L, .dsa, prefix, i, &.{ .{ .rel = "self_attn.indexer.wk" }, .{ .rel = "self_attn.indexer.weights_proj" } }, iw, "layers.{d}.dsa.index.kw");
        const qb: projection.Proj = if (L.w.exl3) .{ .q = try x3.whole(L, "layers.{d}.self_attn.indexer.wq_b", i) } else try projections.cut(L, .dsa, prefix, i, &.{.{ .rel = "self_attn.indexer.wq_b" }}, iw, "layers.{d}.dsa.index.qb");
        try L.whole(try L.get("layers.{d}.self_attn.indexer.k_norm.weight", .{i}), .bf16);
        const ln_w = try L.put("layers.{d}.dsa.index.ln_w", .{i});
        try L.whole(try L.get("layers.{d}.self_attn.indexer.k_norm.bias", .{i}), .bf16);
        const ln_b = try L.put("layers.{d}.dsa.index.ln_b", .{i});
        try L.whole(try L.get("layers.{d}.self_attn.indexer.index_kpool_compress_gate", .{i}), .bf16);
        const gate = try L.put("layers.{d}.dsa.index.gate", .{i});
        try L.whole(try L.get("layers.{d}.self_attn.indexer.index_kpool_compress_ape", .{i}), .bf16);
        const ape = try L.put("layers.{d}.dsa.index.ape", .{i});
        return .{
            .input = input,
            .q_norm = q_norm,
            .kv_norm = kv_norm,
            .q_b = q_b,
            .wk = wk,
            .wv = wv,
            .o = o,
            .index = .{ .kw = kw, .gate = gate, .ln_w = ln_w, .ln_b = ln_b, .ape = ape, .qb = qb, .heads = c.i_heads, .d = c.i_dim },
            .heads = h.len(),
            .qk = qk,
            .dv = c.v_dim,
            .lw = c.kv_lora,
            .q_lora = c.q_lora,
            .dims = c.hidden,
            .eps = c.eps,
            .dense_limit = c.keyWidth(),
        };
    }

    /// kv_b's rows [lo, hi) onto the host bytes: bf16 as stored, or MXFP8 (`mx`, the MTP layer's) widened exactly.
    fn absorbRows(L: *Loader, mx: ?formats.mxfp8.Mxfp8, t: Tensor, name: []const u8, lo: usize, hi: usize, total: usize) !void {
        if (mx) |p| return formats.mxfp8.appendWide(L, p, name, lo, hi);
        try L.rows(t, lo, hi, total, .bf16);
    }

    /// An NVFP4 linear's rows [lo, hi), inputs [klo, khi) as Fp4Linear.from_checkpoint (tf_pack_dense, packScales).
    fn fp4(L: *Loader, comptime fmt: []const u8, i: u32, lo: usize, hi: usize, klo: usize, khi: usize) !cuda.qmmf.Weight {
        const name = try std.fmt.bufPrint(&L.name_buf, prefix ++ fmt, .{i});
        const l = try formats.plainLinear(L.ck, name, L.declaredAs(name));
        const n = hi - lo;
        const k = khi - klo;
        for (0..n) |r| try L.host.appendSlice(L.gpa, l.codes[(lo + r) * l.k / 2 + klo / 2 ..][0 .. k / 2]);
        var src = try cuda.DeviceBuffer.fromHost(L.d, L.host.items);
        defer src.free();
        L.host.clearRetainingCapacity();
        const npad = cuda.nvfp4.padded(n);
        const kg = k / 64;
        const total: u64 = @as(u64, npad / 64) * kg * 512;
        const words = try cuda.DeviceBuffer.alloc(L.d, total * 4);
        try L.w.buffers.append(L.gpa, words);
        try L.w.named.append(L.gpa, .{ .name = try std.fmt.allocPrint(L.gpa, fmt ++ ".words", .{i}), .ptr = words.ptr, .len = words.len });
        var args: cuda.Args = .{};
        args.add(src.ptr);
        for ([_]usize{ n, k / 8, kg }) |v| args.add(@as(c_int, @intCast(v)));
        args.add(words.ptr);
        args.add(@as(c_longlong, @intCast(total)));
        try cuda.launch.launch(L.pack_dense, .{ .grid = .{ .x = @intCast((total + 255) / 256) }, .block = .{ .x = 256 } }, L.stream, &args);
        try L.stream.synchronize();
        const scales = try L.gpa.alloc(u8, n * k / 16);
        defer L.gpa.free(scales);
        for (0..n) |r| @memcpy(scales[r * k / 16 ..][0 .. k / 16], l.scales[(lo + r) * l.k / 16 + klo / 16 ..][0 .. k / 16]);
        try L.host.resize(L.gpa, npad * k / 16);
        cuda.nvfp4.packScales(L.host.items, scales, n, k);
        const bs = try L.put(fmt ++ ".scales", .{i});
        return cuda.nvfp4.weight(words.ptr, bs, l.global, n, k);
    }

    /// weights.load's mlp for a dense layer: gate and up by the rank's rows, down by its columns (NVFP4 or MXFP8).
    fn dense(L: *Loader, i: u32) !mlp.Layer {
        var name: [192]u8 = undefined;
        if (L.w.exl3 and L.ck.has(try std.fmt.bufPrint(&name, prefix ++ "layers.{d}.mlp.gate_proj.trellis", .{i}))) return x3.dense(L, i);
        const c = &L.w.config;
        const s = L.w.share.dense;
        const d = c.hidden;
        const module = try std.fmt.bufPrint(&name, prefix ++ "layers.{d}.mlp.gate_proj", .{i});
        const stored_format = try L.formatOne(L.ck, module, &.{ .bf16, .nvfp4, .mxfp8 });
        inline for (.{ "up_proj", "down_proj" }) |x| try L.format("layers.{d}.mlp." ++ x, .{i}, stored_format);
        if (stored_format == .nvfp4) return .{
            .input = .{ .separate = .{
                .gate = .{ .q = .{ .nvfp4 = try L.fp4("layers.{d}.mlp.gate_proj", i, s.lo, s.hi, 0, d) } },
                .up = .{ .q = .{ .nvfp4 = try L.fp4("layers.{d}.mlp.up_proj", i, s.lo, s.hi, 0, d) } },
            } },
            .down = .{ .q = .{ .nvfp4 = try L.fp4("layers.{d}.mlp.down_proj", i, 0, d, s.lo, s.hi) } },
            .width = s.len(),
            .dims = d,
            .limit = c.swiglu_limit,
        };
        const parts = [_]formats.Part{ .{ .rel = "mlp.gate_proj", .rows = .{ s.lo, s.hi } }, .{ .rel = "mlp.up_proj", .rows = .{ s.lo, s.hi } } };
        return .{
            .input = .{ .combined = try projections.cut(L, .mlp, prefix, i, &parts, &.{stored_format}, "layers.{d}.mlp.gu") },
            .down = try projections.cut(L, .mlp, prefix, i, &.{.{ .rel = "mlp.down_proj", .cols = .{ s.lo, s.hi } }}, &.{stored_format}, "layers.{d}.mlp.down"),
            .width = s.len(),
            .dims = d,
            .limit = c.swiglu_limit,
        };
    }

    /// One device allocation the loader frees itself (staging for the GPU's quantizers).
    pub fn scratch(L: *Loader, n: usize) !cuda.DeviceBuffer {
        return cuda.DeviceBuffer.alloc(L.d, n);
    }

    /// moe_nvfp4 of the MTP layer: BF16 experts cut to the rank, GPU-quantized 8 at a time, scale as weight_scale_2.
    fn quantizedExperts(L: *Loader, i: u32, up: []u32, down: []u32, up_scale: []f32, down_scale: []f32) !void {
        const qk = L.tools.quant orelse return error.NoQuantizer;
        const c = &L.w.config;
        const d: usize = c.hidden;
        const cut = L.w.share.moe;
        const ni = cut.len();
        const up_len = X.bytes(1, ni, d, 2) / 4;
        const down_len = X.bytes(1, d, ni, 1) / 4;
        const chunk = 8;
        var wb = try L.scratch(chunk * ni * d * 2);
        defer wb.free();
        var cb = try L.scratch(chunk * ni * d / 2);
        defer cb.free();
        var sb = try L.scratch(chunk * ni * d / 16);
        defer sb.free();
        var gb = try L.scratch(chunk * 4);
        defer gb.free();
        const codes = try L.gpa.alloc(u8, 3 * chunk * ni * d / 2); // gate, up and down apart: packGateUp takes both
        defer L.gpa.free(codes);
        const scales = try L.gpa.alloc(u8, 3 * chunk * ni * d / 16);
        defer L.gpa.free(scales);
        var g: [3][chunk]f32 = undefined;
        var e0: usize = 0;
        while (e0 < c.experts) : (e0 += chunk) {
            const ec = @min(chunk, c.experts - e0);
            inline for (.{ "gate_proj", "up_proj", "down_proj" }, 0..) |proj, m| {
                const n = if (m < 2) ni else d; // gate and up: the rank's rows; down: every row, the rank's columns
                const k = if (m < 2) d else ni;
                for (e0..e0 + ec) |e| {
                    try L.format("layers.{d}.mlp.experts.{d}." ++ proj, .{ i, e }, .bf16);
                    const t = try L.get("layers.{d}.mlp.experts.{d}." ++ proj ++ ".weight", .{ i, e });
                    if (m < 2) try L.rows(t, cut.lo, cut.hi, c.moe_inter, .bf16) else try L.cols(t, cut.lo, cut.hi, .bf16);
                }
                try wb.upload(0, L.host.items);
                L.host.clearRetainingCapacity();
                try quant.nvfp4(qk, L.stream, wb.ptr, ec, n, k, cb.ptr, sb.ptr, gb.ptr);
                try L.stream.synchronize();
                try cb.download(0, codes[m * chunk * n * k / 2 ..][0 .. ec * n * k / 2]);
                try sb.download(0, scales[m * chunk * n * k / 16 ..][0 .. ec * n * k / 16]);
                try gb.download(0, std.mem.sliceAsBytes(g[m][0..ec]));
            }
            const mc = chunk * ni * d / 2; // one matrix's codes for the chunk; scales an eighth of that
            for (0..ec) |j| {
                const e = e0 + j;
                const one = ni * d;
                var cs: [3][]const u8 = undefined;
                var ss: [3][]const u8 = undefined;
                for (0..3) |m| {
                    cs[m] = codes[m * mc + j * one / 2 ..][0 .. one / 2];
                    ss[m] = scales[m * mc / 8 + j * one / 16 ..][0 .. one / 16];
                }
                X.packGateUp(up[e * up_len ..][0..up_len], .{ cs[0], ss[0] }, .{ cs[1], ss[1] }, ni, d);
                X.packOne(down[e * down_len ..][0..down_len], cs[2], ss[2], d, ni);
                up_scale[e * 2] = g[0][j];
                up_scale[e * 2 + 1] = g[1][j];
                down_scale[e] = g[2][j];
            }
        }
    }

    /// weights.load's moe(i) on NVFP4: router and bias, routed experts as Experts4, the BF16 shared expert's share.
    pub fn moeLayer(L: *Loader, i: u32) !moe.Layer {
        if (L.w.exl3 or L.tools.exl3_routed == .mia_mcg4) return x3.moe(L, i);
        const c = &L.w.config;
        const d: usize = c.hidden;
        const cut = L.w.share.moe;
        const ni = cut.len();
        try L.whole(try L.get("layers.{d}.mlp.gate.weight", .{i}), .bf16);
        const router = try L.put("layers.{d}.moe.router", .{i});
        try L.whole(try L.get("layers.{d}.mlp.gate.e_score_correction_bias", .{i}), .f32);
        const bias = try L.put("layers.{d}.moe.bias", .{i});
        const count: usize = c.experts;
        var routed: cuda.experts.Layer = .{ .format = .nvfp4, .up = 0, .down = 0, .width = ni, .dims = d, .experts = count, .limit = c.swiglu_limit };
        var routed_exl3: ?exl3.Experts = null;
        if (c.exl3) { // the MTP layer's experts too
            routed_exl3 = try exl3.load(L, i, count, d, c.moe_inter, cut);
            if (!routed_exl3.?.shx) L.w.exl3_shx = false;
        } else routed = try L.nvfp4Experts(i);
        const sh = L.w.share.shared;
        const sw: []const formats.Format = if (i == c.layers) &.{ .bf16, .fp8_pb, .mxfp8 } else &.{ .bf16, .fp8_pb }; // MTP's MXFP8
        const gu_parts = [_]formats.Part{ .{ .rel = "mlp.shared_experts.gate_proj", .rows = .{ sh.lo, sh.hi } }, .{ .rel = "mlp.shared_experts.up_proj", .rows = .{ sh.lo, sh.hi } } };
        const gu = try projections.cut(L, .shared, prefix, i, &gu_parts, sw, "layers.{d}.moe.shared.gu");
        const sd = try projections.cut(L, .shared, prefix, i, &.{.{ .rel = "mlp.shared_experts.down_proj", .cols = .{ sh.lo, sh.hi } }}, sw, "layers.{d}.moe.shared.down");
        return .{
            .router = router,
            .bias = bias,
            .routed = routed,
            .exl3 = routed_exl3,
            .count = count,
            .shared_gu = gu,
            .shared_down = sd,
            .shared = sh.len(),
            .top_k = c.topk,
            .scale = c.routed_scale,
            .norm = c.norm_topk,
        };
    }

    /// Layer `i`'s routed NVFP4 experts packed as Experts4 (the MTP layer's quantized from its BF16 ones).
    fn nvfp4Experts(L: *Loader, i: u32) !cuda.experts.Layer {
        const c = &L.w.config;
        const d: usize = c.hidden;
        const cut = L.w.share.moe;
        const ni = cut.len();
        const count: usize = c.experts;
        const up_len = X.bytes(1, ni, d, 2) / 4;
        const down_len = X.bytes(1, d, ni, 1) / 4;
        const up = try L.gpa.alloc(u32, count * up_len);
        defer L.gpa.free(up);
        const down = try L.gpa.alloc(u32, count * down_len);
        defer L.gpa.free(down);
        const up_scale = try L.gpa.alloc(f32, count * 2);
        defer L.gpa.free(up_scale);
        const down_scale = try L.gpa.alloc(f32, count);
        defer L.gpa.free(down_scale);
        const codes = try L.gpa.alloc(u8, d * ni / 2);
        defer L.gpa.free(codes);
        const scales = try L.gpa.alloc(u8, d * ni / 16);
        defer L.gpa.free(scales);
        const per = c.moe_inter * d / 16; // a whole gate, up or down's e4m3 scales
        const decoded = try L.gpa.alloc(u8, 3 * per); // csf_32x2_v1's, decoded before the rank's cut
        defer L.gpa.free(decoded);
        // the MTP layer's experts: bf16 (nvidia's), quantized here, or W4A16_NVFP4 (FP4-Local's), read as stored
        const bf16 = i == c.layers and (try L.get("layers.{d}.mlp.experts.0.gate_proj.weight", .{i})).dtype == .bf16;
        if (bf16) try L.quantizedExperts(i, up, down, up_scale, down_scale) else for (0..count) |e| {
            var ls: [3]core.nvfp4.Linear = undefined;
            inline for (.{ "gate_proj", "up_proj", "down_proj" }, 0..) |proj, m| {
                const name = try std.fmt.bufPrint(&L.name_buf, prefix ++ "layers.{d}.mlp.experts.{d}." ++ proj, .{ i, e });
                ls[m] = try formats.expert(L, name, decoded[m * per ..][0..per]);
            }
            // gate and up: the rank's rows; down: every row, the rank's columns
            const gate = .{ ls[0].codes[cut.lo * d / 2 ..][0 .. ni * d / 2], ls[0].scales[cut.lo * d / 16 ..][0 .. ni * d / 16] };
            const upm = .{ ls[1].codes[cut.lo * d / 2 ..][0 .. ni * d / 2], ls[1].scales[cut.lo * d / 16 ..][0 .. ni * d / 16] };
            X.packGateUp(up[e * up_len ..][0..up_len], gate, upm, ni, d);
            const l = ls[2];
            for (0..d) |r| {
                @memcpy(codes[r * ni / 2 ..][0 .. ni / 2], l.codes[r * l.k / 2 + cut.lo / 2 ..][0 .. ni / 2]);
                @memcpy(scales[r * ni / 16 ..][0 .. ni / 16], l.scales[r * l.k / 16 + cut.lo / 16 ..][0 .. ni / 16]);
            }
            X.packOne(down[e * down_len ..][0..down_len], codes, scales, d, ni);
            up_scale[e * 2] = ls[0].global;
            up_scale[e * 2 + 1] = ls[1].global;
            down_scale[e] = l.global;
        }
        var ptrs: [4]u64 = undefined;
        for (&ptrs, [_][]const u8{ std.mem.sliceAsBytes(up), std.mem.sliceAsBytes(down), std.mem.sliceAsBytes(up_scale), std.mem.sliceAsBytes(down_scale) }, [_][]const u8{ "up", "down", "up_scale", "down_scale" }) |*p, bytes, what| {
            try L.host.appendSlice(L.gpa, bytes);
            p.* = try L.put("layers.{d}.moe.experts.{s}", .{ i, what });
        }
        return .{ .format = .nvfp4, .up = ptrs[0], .down = ptrs[1], .up_scale = ptrs[2], .down_scale = ptrs[3], .width = ni, .dims = d, .experts = count, .limit = c.swiglu_limit };
    }

    fn layer(L: *Loader, i: u32) !Layer {
        const c = &L.w.config;
        const in_norm = try L.norm("layers.{d}.input_layernorm.weight", .{i}, "layers.{d}.in_norm");
        const post_norm = try L.norm("layers.{d}.post_attention_layernorm.weight", .{i}, "layers.{d}.post_norm");
        return .{
            .index = i,
            .attn_hc = try L.hc(i, "attn"),
            .ffn_hc = try L.hc(i, "ffn"),
            .in_norm = in_norm,
            .post_norm = post_norm,
            .mixer = switch (c.kind(i)) {
                .kda => .{ .kda = try L.kda(i) },
                .mla => .{ .mla = try L.dsa(i) },
            },
            .ffn = if (c.isMoe(i)) .{ .moe = try L.moeLayer(i) } else .{ .dense = try L.dense(i) },
        };
    }
};

/// weights.load without the MTP head: embedding, layers, final norm, the rank's head rows (bf16, MXFP8); `pack_dense`.
pub fn load(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, stream: cuda.Stream, tools: Tools, dir: []const u8, c: cfg.Config, share: shard.Share, layers: []const u32, with_mtp: bool) !Weights {
    var w: Weights = .{ .gpa = gpa, .config = c, .share = share, .precision = tools.precision };
    errdefer w.deinit();
    var ck = try core.Checkpoint.openModelPrefix(gpa, io, dir, prefix);
    defer ck.close();
    var head_ck = try core.Checkpoint.openModelPrefix(gpa, io, dir, "lm_head.");
    defer head_ck.close();
    var L: Loader = .{ .gpa = gpa, .d = d, .stream = stream, .pack_dense = tools.pack_dense, .tools = tools, .w = &w, .ck = &ck };
    defer L.host.deinit(gpa);
    var recipes = try formats.readRecipes(gpa, io, dir);
    defer if (recipes) |*r| r.deinit();
    if (recipes) |*r| L.recipes = if (r.modules.count() > 0) r else null;
    if (tools.recipes) |r| L.recipes = r;
    w.exl3 = ck.has(prefix ++ "layers.0.self_attn.qkv_proj.trellis");
    w.layers = try gpa.alloc(Layer, layers.len);
    for (layers, w.layers) |i, *l| {
        if (i >= c.layers) return error.BadLayer;
        l.* = try L.layer(i);
        ck.release(); // a rank reads about half the checkpoint: its pages would crowd the GPU's allocations out
    }
    try L.whole(try L.get("embed_tokens.weight", .{}), .bf16);
    w.embed = try L.put("embed", .{});
    w.norm = try L.norm("norm.weight", .{}, "norm");
    const vocab: formats.Part = .{ .rel = "lm_head", .rows = .{ share.vocab.lo, share.vocab.hi } };
    if (w.exl3) {
        w.head = .{ .q = try x3.matrix(&L, &head_ck, "lm_head", 0, 0, &.{.{ .lo = share.vocab.lo, .hi = share.vocab.hi }}) };
        if (with_mtp and c.mtp > 0) ck.add(dir, "mtp.safetensors") catch |e| if (e != error.FileNotFound) return e;
    } else w.head = try formats.cut(&L, &head_ck, &.{"lm_head"}, &.{vocab}, &.{ .bf16, .mxfp8 }, "head", .{});
    if (with_mtp and c.mtp > 0) w.mtp = try @import("cuda_mtp_load.zig").load(&L, &head_ck);
    try projections.head(&L);
    if (L.recipes != null) {
        for (0..c.layers) |i| {
            var selected = false;
            for (layers) |index| if (index == i) {
                selected = true;
                break;
            };
            if (!selected) try ck.skip(try std.fmt.bufPrint(&L.name_buf, prefix ++ "layers.{d}.", .{i}));
        }
        if (!with_mtp or c.mtp == 0) try ck.skip(try std.fmt.bufPrint(&L.name_buf, prefix ++ "layers.{d}.", .{c.layers}));
        if (ck.unused() != 0 or head_ck.unused() != 0) return error.UnusedTensors;
    }
    return w;
}

test "actual mixed loading keeps an MXFP8 head and float32 norm" {
    const gpa = std.testing.allocator;
    const M = @import("cuda_test_driver.zig").Memory;
    const hidden = 64;
    const vocab = 128;
    const embed: [vocab * hidden]u16 = @splat(0x3f80);
    const norm: [hidden]f32 = @splat(@bitCast(@as(u32, 0x3f808001))); // not exactly representable as BF16
    const codes: [vocab * hidden]u8 = @splat(0x38);
    const scales: [vocab * hidden / 32]u8 = @splat(127);
    const tensors = [_]core.shard_edit.New{
        .{ .name = prefix ++ "embed_tokens.weight", .dtype = .bf16, .shape = &.{ vocab, hidden }, .bytes = std.mem.sliceAsBytes(&embed) },
        .{ .name = prefix ++ "norm.weight", .dtype = .f32, .shape = &.{hidden}, .bytes = std.mem.sliceAsBytes(&norm) },
        .{ .name = "lm_head.weight", .dtype = .f8_e4m3, .shape = &.{ vocab, hidden }, .bytes = &codes },
        .{ .name = "lm_head.weight_scale", .dtype = .u8, .shape = &.{ vocab, hidden / 32 }, .bytes = &scales },
    };
    {
        var tmp = std.testing.tmpDir(.{});
        defer tmp.cleanup();
        var dir: [96]u8 = undefined;
        const at = try std.fmt.bufPrint(&dir, ".zig-cache/tmp/{s}", .{tmp.sub_path});
        var path: [128]u8 = undefined;
        try core.shard_edit.create(gpa, try std.fmt.bufPrint(&path, "{s}/model.safetensors", .{at}), &.{}, &tensors);
        try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "config.json", .data =
            \\{"quantization_config":{"quant_method":"modelopt","quant_algo":"MIXED_PRECISION","quantized_layers":{"lm_head":{"quant_algo":"MXFP8","group_size":32}}}}
        });
        var driver = M.driver();
        var lane: cuda.qmmf.Lane = undefined; // only the CPU shape planner is used; no kernel is launched
        lane.clusters = true;
        const tools: Tools = .{ .pack_dense = .{ .d = &driver, .handle = @ptrFromInt(1) }, .lane = &lane };
        const c: cfg.Config = .{ .hidden = hidden, .vocab = vocab, .layers = 0, .mtp = 0 };
        const share = try shard.Share.of(&c, 0, 1);
        var w = try load(gpa, std.testing.io, &driver, .{ .d = &driver, .handle = null }, tools, at, c, share, &.{}, false);
        defer w.deinit();
        try std.testing.expect(w.head == .mxfp8);
        try std.testing.expect(w.norm == .f32);
        try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(&norm), M.slice(w.norm.f32, @sizeOf(@TypeOf(norm))));
        try std.testing.expectEqual(vocab * hidden, w.find("head.codes").?.len);
        try std.testing.expectEqual(vocab * hidden / 32, w.find("head.scales").?.len);
        try std.testing.expect(w.find("head") == null); // no materialized BF16 checkpoint head
    }
    try std.testing.expect(M.empty());
}

test "the actual startup loader refuses a missing checkpoint without device allocation" {
    const Trace = struct {
        var allocations: usize = 0;
        fn alloc(_: *u64, _: usize) callconv(.c) cuda.abi.Result {
            allocations += 1;
            return 2; // an attempted allocation is observable, not real CUDA work
        }
    };
    Trace.allocations = 0;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var path: [160]u8 = undefined;
    const missing = try std.fmt.bufPrint(&path, ".zig-cache/tmp/{s}/missing", .{tmp.sub_path});
    var driver = @import("cuda_test_driver.zig").Memory.driver();
    driver.api.cuMemAlloc_v2 = Trace.alloc;
    const c: cfg.Config = .{};
    const share = try shard.Share.of(&c, 0, 2);
    try std.testing.expectError(error.FileNotFound, load(std.testing.allocator, std.testing.io, &driver, .{ .d = &driver, .handle = null }, .{ .pack_dense = .{ .d = &driver, .handle = @ptrFromInt(1) } }, missing, c, share, &.{0}, false));
    try std.testing.expectEqual(@as(usize, 0), Trace.allocations);
}

test "bf16 rounding of fp32 is to nearest even" {
    try std.testing.expectEqual(@as(u16, 0x3f80), toBf16(1.0));
    try std.testing.expectEqual(@as(u16, 0x3f80), toBf16(@bitCast(@as(u32, 0x3f808000)))); // a tie to even (down)
    try std.testing.expectEqual(@as(u16, 0x3f82), toBf16(@bitCast(@as(u32, 0x3f818000)))); // a tie to even (up)
    try std.testing.expectEqual(@as(u16, 0x3f81), toBf16(@bitCast(@as(u32, 0x3f808001))));
}

test "the loader narrows a float32 to bf16 only when bf16 holds it, unlike FP4-Local's norms" {
    try std.testing.expectEqual(@as(?u16, 0x3f80), narrow(1.0));
    try std.testing.expectEqual(@as(?u16, null), narrow(@bitCast(@as(u32, 0x3f808001))));
    try std.testing.expect(narrow(std.math.nan(f32)) != null);
}

test {
    _ = projections;
    _ = x3;
    _ = @import("cuda_formats.zig");
    _ = @import("cuda_proj.zig");
    _ = @import("cuda_fp8pb.zig");
    _ = @import("cuda_mxfp8.zig");
}
