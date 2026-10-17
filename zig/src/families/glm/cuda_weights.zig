//! A TP rank's GLM-5.3-Flash weights on CUDA (the Python family's weights.load, NVFP4 or EXL3), named by its paths.

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
const b16 = @import("cuda_b16.zig");
const x3 = @import("cuda_weights_exl3.zig");
pub const projections = @import("cuda_weights_projs.zig");

const source = @import("cuda_weight_source.zig");
const Tensor = core.checkpoint.Tensor;
pub const prefix = "model.language_model.";

/// A hyper-connection site: fn bf16 [s (s + 2)][s d], base and scale widened to fp32.
pub const Hc = struct { fn_w: u64, base: u64, scale: u64 };

pub const Mixer = union(cfg.Kind) { kda: kda_block.Layer, mla: dsa_block.Layer };
pub const Ffn = union(enum) { dense: mlp.Layer, moe: moe.Layer };

pub const Layer = struct { index: u32, attn_hc: Hc, ffn_hc: Hc, in_norm: u64, post_norm: u64, mixer: Mixer, ffn: Ffn };

/// The MTP head (weights.MTPW): norms, eh_proj, a DSA + MoE layer, shared_head.norm, 4-bit draft head (EXL3: the head).
pub const Mtp = struct { enorm: u64, hnorm: u64, eh: b16.Proj, norm: u64, layer: Layer, draft: ?q4.Q4 };

/// The kernels a load runs: affine4_pack's tf_pack_dense, and for the MTP head the quantizers and the 4-bit tiling.
pub const Tools = struct { recipes: ?*const core.modelopt.Table = null, pack_dense: cuda.Function, quant: ?*const quant.Kernels = null, q4: ?*const q4.Kernels = null, affine4: ?*const cuda.qlinear.Affine4Kernels = null, modelopt: ?*const cuda.modelopt.Packer = null, ck: ?*const cuda.ck.Ck = null, precision: @import("cuda_weight_precision.zig").Settings = .{}, exl3_routed: @import("cuda_experts_mia.zig").Mode = .generic };

/// A loaded tensor by the Python family's attribute path, for digests against its weights.
pub const Named = struct { name: []u8, ptr: u64, len: usize };

pub const Weights = struct {
    gpa: std.mem.Allocator,
    config: cfg.Config,
    share: shard.Share,
    embed: u64 = 0, // bf16 [vocab][dims]
    layers: []Layer = &.{},
    norm: u64 = 0,
    head: b16.Proj = .{ .b16 = 0 }, // bf16 [share.vocab][dims], or EXL3
    precision: @import("cuda_weight_precision.zig").Settings = .{},
    native_conversions: [@import("cuda_weight_precision.zig").groups.len]u32 = @splat(0),
    exl3: bool = false, // an EXL3 checkpoint: the attention, shared expert, dense MLP and head quantized
    mtp: ?Mtp = null,
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

    /// Whether the attention projections are quantized (an EXL3 checkpoint), so the KDA projection stages its rows.
    pub fn quantizedAttention(w: *const Weights) bool {
        return w.exl3;
    }

    /// The upload named `name` (the Python family's attribute path), or null.
    pub fn find(w: *const Weights, name: []const u8) ?Named {
        for (w.named.items) |n| if (std.mem.eql(u8, n.name, name)) return n;
        return null;
    }
};

const toBf16 = core.quant.convert.bf16;

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

    pub fn get(L: *Loader, comptime fmt: []const u8, args: anytype) !source.Source {
        const name = try std.fmt.bufPrint(&L.name_buf, prefix ++ fmt, args);
        return source.Source.read(L.ck, L.tools.recipes, name);
    }

    /// The host bytes gathered so far uploaded as one tensor named `fmt`.
    pub fn put(L: *Loader, comptime fmt: []const u8, args: anytype) !u64 {
        const b = try cuda.DeviceBuffer.fromHost(L.d, L.host.items);
        try L.w.buffers.append(L.gpa, b);
        try L.w.named.append(L.gpa, .{ .name = try std.fmt.allocPrint(L.gpa, fmt, args), .ptr = b.ptr, .len = b.len });
        L.host.clearRetainingCapacity();
        return b.ptr;
    }

    fn view(t: anytype) source.Source {
        return if (@TypeOf(t) == Tensor) .{ .weight = t } else t;
    }

    fn add(L: *Loader, t: anytype, lo: usize, hi: usize, to: core.checkpoint.DType) !void {
        try view(t).append(L.gpa, &L.host, lo, hi, to);
    }

    pub fn rows(L: *Loader, t: anytype, lo: usize, hi: usize, total: usize, to: core.checkpoint.DType) !void {
        if (total == 0 or lo > hi or hi > total) return error.UnexpectedTensor;
        const count = view(t).weight.numel();
        const per = count / total;
        if (per * total != count) return error.UnexpectedTensor;
        try L.add(t, lo * per, hi * per, to);
    }

    pub fn whole(L: *Loader, t: anytype, to: core.checkpoint.DType) !void {
        try L.add(t, 0, view(t).weight.numel(), to);
    }

    pub fn cols(L: *Loader, t: anytype, lo: usize, hi: usize, to: core.checkpoint.DType) !void {
        const w = view(t).weight;
        if (w.rank != 2 or lo > hi or hi > w.dim(1)) return error.UnexpectedTensor;
        const k = w.dim(1);
        for (0..w.dim(0)) |r| try L.add(t, r * k + lo, r * k + hi, to);
    }

    fn hc(L: *Loader, i: u32, comptime site: []const u8) !Hc {
        try L.whole(try L.get("layers.{d}.hc_" ++ site ++ "_fn", .{i}), .bf16);
        const f = try L.put("layers.{d}.{s}_hc.fn", .{ i, site });
        try L.whole(try L.get("layers.{d}.hc_" ++ site ++ "_base", .{i}), .f32);
        const base = try L.put("layers.{d}.{s}_hc.base", .{ i, site });
        try L.whole(try L.get("layers.{d}.hc_" ++ site ++ "_scale", .{i}), .f32);
        return .{ .fn_w = f, .base = base, .scale = try L.put("layers.{d}.{s}_hc.scale", .{ i, site }) };
    }

    /// weights.load's kda(i): [q | k | v | f_a | g_a | b], f_b/g_b by heads, o by columns, conv taps, A_log, dt_bias.
    fn kda(L: *Loader, i: u32) !kda_block.Layer {
        if (L.w.exl3) return x3.kda(L, i);
        const c = &L.w.config;
        const h = L.w.share.kda;
        const dk = c.kda_dim;
        const all = c.kda_heads * dk;
        inline for (.{ "q", "k", "v" }) |x| try L.rows(try L.get("layers.{d}.self_attn." ++ x ++ "_proj.weight", .{i}), h.lo * dk, h.hi * dk, all, .bf16);
        inline for (.{ "f_a", "g_a" }) |x| try L.whole(try L.get("layers.{d}.self_attn." ++ x ++ "_proj.weight", .{i}), .bf16);
        try L.rows(try L.get("layers.{d}.self_attn.b_proj.weight", .{i}), h.lo, h.hi, c.kda_heads, .bf16);
        const proj = try projections.finish(L, .kda, "layers.{d}.kda.proj", .{i}, 3 * h.len() * dk + 2 * dk + h.len(), c.hidden);
        try L.rows(try L.get("layers.{d}.self_attn.f_b_proj.weight", .{i}), h.lo * dk, h.hi * dk, all, .bf16);
        const fb = try projections.finish(L, .kda, "layers.{d}.kda.fb", .{i}, h.len() * dk, dk);
        try L.rows(try L.get("layers.{d}.self_attn.g_b_proj.weight", .{i}), h.lo * dk, h.hi * dk, all, .bf16);
        const gb = try projections.finish(L, .kda, "layers.{d}.kda.gb", .{i}, h.len() * dk, dk);
        inline for (.{ "q", "k", "v" }) |x| try L.rows(try L.get("layers.{d}.self_attn." ++ x ++ "_conv1d.weight", .{i}), h.lo * dk, h.hi * dk, all, .f32);
        const conv = try L.put("layers.{d}.kda.conv", .{i});
        try L.rows(try L.get("layers.{d}.self_attn.A_log", .{i}), h.lo, h.hi, c.kda_heads, .f32);
        const a_log = try L.put("layers.{d}.kda.a_log", .{i});
        try L.rows(try L.get("layers.{d}.self_attn.dt_bias", .{i}), h.lo * dk, h.hi * dk, all, .f32);
        const dt = try L.put("layers.{d}.kda.dt_bias", .{i});
        try L.whole(try L.get("layers.{d}.self_attn.o_norm.weight", .{i}), .bf16);
        const norm = try L.put("layers.{d}.kda.norm", .{i});
        try L.cols(try L.get("layers.{d}.self_attn.o_proj.weight", .{i}), h.lo * dk, h.hi * dk, .bf16);
        const o = try projections.finish(L, .kda, "layers.{d}.kda.o", .{i}, c.hidden, h.len() * dk);
        return .{ .proj = if (proj == .b16) proj.b16 else 0, .proj_q = if (proj == .q) proj.q else null, .fb = if (fb == .b16) fb.b16 else 0, .fb_q = if (fb == .q) fb.q else null, .gb = if (gb == .b16) gb.b16 else 0, .gb_q = if (gb == .q) gb.q else null, .o = o, .conv_w = conv, .a_log = a_log, .dt_bias = dt, .norm_w = norm, .eps = c.eps, .lower = c.lower_bound, .heads = h.len(), .dims = c.hidden };
    }

    /// weights.load's dsa(i) (latent): [q_a | kv_a], norms, q_b by heads, kv_b as wk/wv (AbsorbW), o by cols, indexer.
    fn dsa(L: *Loader, i: u32) !dsa_block.Layer {
        const c = &L.w.config;
        const h = L.w.share.mla;
        const qk = c.nope;
        const ex = L.w.exl3;
        const input: dsa_block.InputProjection = if (ex) .{ .separate = .{
            .qa = .{ .q = try x3.whole(L, "layers.{d}.self_attn.q_a_proj", i) },
            .kva = .{ .q = try x3.whole(L, "layers.{d}.self_attn.kv_a_proj_with_mqa", i) },
        } } else if (L.tools.precision.dense.get(.dsa) != .bf16) blk: {
            try L.whole(try L.get("layers.{d}.self_attn.q_a_proj.weight", .{i}), .bf16);
            const qa = try projections.finish(L, .dsa, "layers.{d}.dsa.qa", .{i}, c.q_lora, c.hidden);
            try L.whole(try L.get("layers.{d}.self_attn.kv_a_proj_with_mqa.weight", .{i}), .bf16);
            const kva = try projections.finish(L, .dsa, "layers.{d}.dsa.kva", .{i}, c.kv_lora, c.hidden);
            break :blk .{ .separate = .{ .qa = qa, .kva = kva } };
        } else blk: {
            inline for (.{ "q_a_proj", "kv_a_proj_with_mqa" }) |x| try L.whole(try L.get("layers.{d}.self_attn." ++ x ++ ".weight", .{i}), .bf16);
            break :blk .{ .combined_bf16 = try L.put("layers.{d}.dsa.proj", .{i}) };
        };
        try L.whole(try L.get("layers.{d}.self_attn.q_a_layernorm.weight", .{i}), .bf16);
        const q_norm = try L.put("layers.{d}.dsa.q_norm", .{i});
        try L.whole(try L.get("layers.{d}.self_attn.kv_a_layernorm.weight", .{i}), .bf16);
        const kv_norm = try L.put("layers.{d}.dsa.kv_norm", .{i});
        const q_b: b16.Proj = if (ex) .{ .q = try x3.layerMatrix(L, "layers.{d}.self_attn.q_b_proj", i, 0, 0, &.{.{ .lo = h.lo * qk, .hi = h.hi * qk }}) } else blk: {
            try L.rows(try L.get("layers.{d}.self_attn.q_b_proj.weight", .{i}), h.lo * qk, h.hi * qk, c.mla_heads * qk, .bf16);
            break :blk try projections.finish(L, .dsa, "layers.{d}.dsa.q_b", .{i}, h.len() * qk, c.q_lora);
        };
        const kv_b = try L.get("layers.{d}.self_attn.kv_b_proj.weight", .{i});
        const per = qk + c.v_dim; // a head's kv_b rows: its keys, then its values
        const total = c.mla_heads * per;
        for (h.lo..h.hi) |head| try L.rows(kv_b, head * per, head * per + qk, total, .bf16);
        const wk = try L.put("layers.{d}.dsa.absorb.wk", .{i});
        for (h.lo..h.hi) |head| try L.rows(kv_b, head * per + qk, (head + 1) * per, total, .bf16);
        const wv = try L.put("layers.{d}.dsa.absorb.wv", .{i});
        const o: b16.Proj = if (ex) .{ .q = try x3.layerMatrix(L, "layers.{d}.self_attn.o_proj", i, h.lo * c.v_dim, h.hi * c.v_dim, &.{.{ .lo = 0, .hi = c.hidden }}) } else blk: {
            try L.cols(try L.get("layers.{d}.self_attn.o_proj.weight", .{i}), h.lo * c.v_dim, h.hi * c.v_dim, .bf16);
            break :blk try projections.finish(L, .dsa, "layers.{d}.dsa.o", .{i}, c.hidden, h.len() * c.v_dim);
        };
        inline for (.{ "wk", "weights_proj" }) |x| try L.whole(try L.get("layers.{d}.self_attn.indexer." ++ x ++ ".weight", .{i}), .bf16);
        const kw = try projections.finish(L, .dsa, "layers.{d}.dsa.index.kw", .{i}, c.i_dim + c.i_heads, c.hidden);
        const qb: b16.Proj = if (ex) .{ .q = try x3.whole(L, "layers.{d}.self_attn.indexer.wq_b", i) } else blk: {
            try L.whole(try L.get("layers.{d}.self_attn.indexer.wq_b.weight", .{i}), .bf16);
            break :blk try projections.finish(L, .dsa, "layers.{d}.dsa.index.qb", .{i}, c.i_heads * c.i_dim, c.q_lora);
        };
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
            .index = .{ .kw = if (kw == .b16) kw.b16 else 0, .kw_q = if (kw == .q) kw.q else null, .gate = gate, .ln_w = ln_w, .ln_b = ln_b, .ape = ape, .qb = qb, .heads = c.i_heads, .d = c.i_dim },
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

    /// An NVFP4 linear's rows [lo, hi), inputs [klo, khi) as Fp4Linear.from_checkpoint (tf_pack_dense, packScales).
    fn fp4(L: *Loader, comptime fmt: []const u8, i: u32, lo: usize, hi: usize, klo: usize, khi: usize) !cuda.qmmf.Weight {
        const name = try std.fmt.bufPrint(&L.name_buf, prefix ++ fmt, .{i});
        var stored = try source.Fp4.read(L.gpa, L.ck, L.tools.recipes, name);
        defer stored.deinit(L.gpa);
        const l = stored.linear;
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

    /// weights.load's mlp for a dense layer: gate and up by the rank's rows, down by its columns (NVFP4).
    fn dense(L: *Loader, i: u32) !mlp.Layer {
        const c = &L.w.config;
        const s = L.w.share.dense;
        const d = c.hidden;
        var dense_name: [192]u8 = undefined;
        const weight_name = try std.fmt.bufPrint(&dense_name, prefix ++ "layers.{d}.mlp.gate_proj.weight", .{i});
        if (L.ck.has(weight_name)) {
            const gate = try L.ck.get(weight_name);
            const rec = source.recipe(L.tools.recipes, weight_name[0 .. weight_name.len - 7]);
            const bf16 = if (L.tools.recipes != null) (rec == null or rec.?.algo == .mxfp8 or rec.?.algo == .fp8_pb_wo) else (gate.dtype == .bf16 or gate.dtype == .f32);
            if (bf16) return .{
                .gate = try L.denseB16(i, "gate_proj", s.lo, s.hi, 0, d),
                .up = try L.denseB16(i, "up_proj", s.lo, s.hi, 0, d),
                .down = try L.denseB16(i, "down_proj", 0, d, s.lo, s.hi),
                .width = s.len(),
                .dims = d,
                .limit = c.swiglu_limit,
            };
        }
        if (L.w.exl3) return .{
            .gate = .{ .q = try x3.layerMatrix(L, "layers.{d}.mlp.gate_proj", i, 0, 0, &.{s}) },
            .up = .{ .q = try x3.layerMatrix(L, "layers.{d}.mlp.up_proj", i, 0, 0, &.{s}) },
            .down = .{ .q = try x3.layerMatrix(L, "layers.{d}.mlp.down_proj", i, s.lo, s.hi, &.{.{ .lo = 0, .hi = d }}) },
            .width = s.len(),
            .dims = d,
            .limit = c.swiglu_limit,
        };
        return .{
            .gate = .{ .q = .{ .nvfp4 = try L.fp4("layers.{d}.mlp.gate_proj", i, s.lo, s.hi, 0, d) } },
            .up = .{ .q = .{ .nvfp4 = try L.fp4("layers.{d}.mlp.up_proj", i, s.lo, s.hi, 0, d) } },
            .down = .{ .q = .{ .nvfp4 = try L.fp4("layers.{d}.mlp.down_proj", i, 0, d, s.lo, s.hi) } },
            .width = s.len(),
            .dims = d,
            .limit = c.swiglu_limit,
        };
    }

    fn denseB16(L: *Loader, i: u32, comptime proj: []const u8, lo: usize, hi: usize, klo: usize, khi: usize) !b16.Proj {
        const t = try L.get("layers.{d}.mlp." ++ proj ++ ".weight", .{i});
        if (t.weight.rank != 2 or hi > t.weight.dim(0) or khi > t.weight.dim(1)) return error.UnexpectedTensor;
        for (lo..hi) |row| try L.add(t, row * t.weight.dim(1) + klo, row * t.weight.dim(1) + khi, .bf16);
        return projections.finish(L, .mlp, "layers.{d}.mlp." ++ proj, .{i}, hi - lo, khi - klo);
    }

    /// One device allocation the loader frees itself (staging for the GPU's quantizers).
    fn scratch(L: *Loader, n: usize) !cuda.DeviceBuffer {
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
    fn moeLayer(L: *Loader, i: u32) !moe.Layer {
        var routed_name: [192]u8 = undefined;
        if (L.w.exl3 or L.ck.has(try std.fmt.bufPrint(&routed_name, prefix ++ "layers.{d}.mlp.experts.0.gate_proj.trellis", .{i}))) return x3.moe(L, i);
        if (L.tools.exl3_routed == .mia_mcg4) return error.MiaNeedsMcg4Experts;
        const c = &L.w.config;
        const d: usize = c.hidden;
        const cut = L.w.share.moe;
        const ni = cut.len();
        try L.whole(try L.get("layers.{d}.mlp.gate.weight", .{i}), .bf16);
        const router = try L.put("layers.{d}.moe.router", .{i});
        try L.whole(try L.get("layers.{d}.mlp.gate.e_score_correction_bias", .{i}), .f32);
        const bias = try L.put("layers.{d}.moe.bias", .{i});
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
        if (i == c.layers and L.tools.recipes == null) try L.quantizedExperts(i, up, down, up_scale, down_scale) else for (0..count) |e| {
            const globals = try source.packExpert(L.gpa, L.ck, L.tools.recipes, i, e, .{ .lo = cut.lo, .hi = cut.hi }, d, up[e * up_len ..][0..up_len], down[e * down_len ..][0..down_len], codes, scales);
            up_scale[e * 2] = globals[0];
            up_scale[e * 2 + 1] = globals[1];
            down_scale[e] = globals[2];
        }
        var ptrs: [4]u64 = undefined;
        for (&ptrs, [_][]const u8{ std.mem.sliceAsBytes(up), std.mem.sliceAsBytes(down), std.mem.sliceAsBytes(up_scale), std.mem.sliceAsBytes(down_scale) }, [_][]const u8{ "up", "down", "up_scale", "down_scale" }) |*p, bytes, what| {
            try L.host.appendSlice(L.gpa, bytes);
            p.* = try L.put("layers.{d}.moe.experts.{s}", .{ i, what });
        }
        const sh = L.w.share.shared;
        inline for (.{ "gate_proj", "up_proj" }) |x| try L.rows(try L.get("layers.{d}.mlp.shared_experts." ++ x ++ ".weight", .{i}), sh.lo, sh.hi, c.shared_inter, .bf16);
        const gu = try projections.finish(L, .shared, "layers.{d}.moe.shared.gu", .{i}, 2 * sh.len(), d);
        try L.cols(try L.get("layers.{d}.mlp.shared_experts.down_proj.weight", .{i}), sh.lo, sh.hi, .bf16);
        const sd = try projections.finish(L, .shared, "layers.{d}.moe.shared.down", .{i}, d, sh.len());
        return .{
            .router = router,
            .bias = bias,
            .routed = .{ .format = .nvfp4, .up = ptrs[0], .down = ptrs[1], .up_scale = ptrs[2], .down_scale = ptrs[3], .width = ni, .dims = d, .experts = count, .limit = c.swiglu_limit },
            .count = count,
            .shared_gu = if (gu == .b16) gu.b16 else 0,
            .shared_gu_q = if (gu == .q) gu.q else null,
            .shared_down = sd,
            .shared = sh.len(),
            .top_k = c.topk,
            .scale = c.routed_scale,
            .norm = c.norm_topk,
        };
    }

    /// weights.load's MTPW: layer `layers` as plain DSA + MoE, norms, eh_proj, draft head from the rank's head rows.
    fn mtp(L: *Loader) !Mtp {
        const c = &L.w.config;
        const i = c.layers;
        try L.whole(try L.get("layers.{d}.input_layernorm.weight", .{i}), .bf16);
        const in_norm = try L.put("layers.{d}.in_norm", .{i});
        try L.whole(try L.get("layers.{d}.post_attention_layernorm.weight", .{i}), .bf16);
        const post_norm = try L.put("layers.{d}.post_norm", .{i});
        const none: Hc = .{ .fn_w = 0, .base = 0, .scale = 0 };
        const layer_w: Layer = .{ .index = i, .attn_hc = none, .ffn_hc = none, .in_norm = in_norm, .post_norm = post_norm, .mixer = .{ .mla = try L.dsa(i) }, .ffn = .{ .moe = try L.moeLayer(i) } };
        L.ck.release();
        if (L.w.exl3) {
            var o3: [3]u64 = undefined;
            for (&o3, [_][]const u8{ "enorm.weight", "hnorm.weight", "shared_head.norm.weight" }, [_][]const u8{ "enorm", "hnorm", "norm" }) |*p, name, what| {
                try L.whole(try source.Source.read(L.ck, L.tools.recipes, try std.fmt.bufPrint(&L.name_buf, prefix ++ "layers.{d}.{s}", .{ i, name })), .bf16);
                p.* = try L.put("mtp.{s}", .{what});
            }
            return .{ .enorm = o3[0], .hnorm = o3[1], .eh = .{ .q = try x3.whole(L, "layers.{d}.eh_proj", i) }, .norm = o3[2], .layer = layer_w, .draft = null };
        }
        var eh: b16.Proj = undefined;
        var out: [4]u64 = undefined;
        for (&out, [_][]const u8{ "enorm.weight", "hnorm.weight", "eh_proj.weight", "shared_head.norm.weight" }, [_][]const u8{ "enorm", "hnorm", "eh", "norm" }) |*p, name, what| {
            try L.whole(try source.Source.read(L.ck, L.tools.recipes, try std.fmt.bufPrint(&L.name_buf, prefix ++ "layers.{d}.{s}", .{ i, name })), .bf16);
            if (std.mem.eql(u8, what, "eh")) {
                eh = try projections.finish(L, .mtp_eh, "mtp.eh", .{}, c.hidden, 2 * c.hidden);
                p.* = 0;
            } else p.* = try L.put("mtp.{s}", .{what});
        }
        // the draft head: quantize4 of the BF16 head's rows, then make_q4's tiling
        const qk = L.tools.quant orelse return error.NoQuantizer;
        const tk = L.tools.q4 orelse return error.NoQuantizer;
        const n = L.w.share.vocab.len();
        const k: usize = c.hidden;
        var raw: [3]cuda.DeviceBuffer = undefined;
        var made: usize = 0;
        defer for (raw[0..made]) |*b| b.free();
        for (&raw, [_]usize{ n * k / 2, n * k / 32, n * k / 32 }) |*b, size| {
            b.* = try L.scratch(size);
            made += 1;
        }
        try quant.q4(qk, L.stream, L.w.head.b16, n, k, raw[0].ptr, raw[1].ptr, raw[2].ptr);
        var tiled: [3]u64 = undefined;
        for (&tiled, q4.bytes(n, k), [_][]const u8{ "words", "scales", "biases" }) |*p, size, what| {
            const b = try cuda.DeviceBuffer.alloc(L.d, size);
            try L.w.buffers.append(L.gpa, b);
            try L.w.named.append(L.gpa, .{ .name = try std.fmt.allocPrint(L.gpa, "draft_head.{s}", .{what}), .ptr = b.ptr, .len = b.len });
            p.* = b.ptr;
        }
        const draft: q4.Q4 = .{ .w = tiled[0], .s = tiled[1], .b = tiled[2], .n = n, .k = k, .npad = q4.npadOf(n) };
        try q4.pack(tk, L.stream, raw[0].ptr, raw[1].ptr, raw[2].ptr, draft);
        try L.stream.synchronize();
        return .{ .enorm = out[0], .hnorm = out[1], .eh = eh, .norm = out[3], .layer = layer_w, .draft = draft };
    }

    fn layer(L: *Loader, i: u32) !Layer {
        const c = &L.w.config;
        try L.whole(try L.get("layers.{d}.input_layernorm.weight", .{i}), .bf16);
        const in_norm = try L.put("layers.{d}.in_norm", .{i});
        try L.whole(try L.get("layers.{d}.post_attention_layernorm.weight", .{i}), .bf16);
        const post_norm = try L.put("layers.{d}.post_norm", .{i});
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

/// weights.load without the MTP head: embedding, layers, final norm, the rank's BF16 head rows; `pack_dense`.
pub fn load(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, stream: cuda.Stream, tools: Tools, dir: []const u8, c: cfg.Config, share: shard.Share, layers: []const u32, with_mtp: bool) !Weights {
    var w: Weights = .{ .gpa = gpa, .config = c, .share = share, .precision = tools.precision };
    errdefer w.deinit();
    var ck = try core.Checkpoint.openModelPrefix(gpa, io, dir, prefix);
    defer ck.close();
    var head_ck = try core.Checkpoint.openModelPrefix(gpa, io, dir, "lm_head.");
    defer head_ck.close();
    var L: Loader = .{ .gpa = gpa, .d = d, .stream = stream, .pack_dense = tools.pack_dense, .tools = tools, .w = &w, .ck = &ck };
    defer L.host.deinit(gpa);
    w.exl3 = ck.has(prefix ++ "layers.0.self_attn.qkv_proj.trellis");
    w.layers = try gpa.alloc(Layer, layers.len);
    for (layers, w.layers) |i, *l| {
        if (i >= c.layers) return error.BadLayer;
        l.* = try L.layer(i);
        ck.release(); // a rank reads about half the checkpoint: its pages would crowd the GPU's allocations out
    }
    try L.whole(try L.get("embed_tokens.weight", .{}), .bf16);
    w.embed = try L.put("embed", .{});
    try L.whole(try L.get("norm.weight", .{}), .bf16);
    w.norm = try L.put("norm", .{});
    if (w.exl3) {
        w.head = .{ .q = try x3.matrix(&L, &head_ck, "lm_head", 0, 0, &.{.{ .lo = share.vocab.lo, .hi = share.vocab.hi }}) };
        // EXL3 keeps the MTP layer in mtp.safetensors or, in some quants, in the shards
        if (with_mtp and c.mtp > 0) ck.add(dir, "mtp.safetensors") catch |e| if (e != error.FileNotFound) return e;
    } else {
        try L.rows(try source.Source.read(&head_ck, tools.recipes, "lm_head.weight"), share.vocab.lo, share.vocab.hi, c.vocab, .bf16);
        w.head = .{ .b16 = try L.put("head", .{}) };
    }
    if (with_mtp and c.mtp > 0) w.mtp = try L.mtp();
    try projections.head(&L);
    if (tools.recipes != null) {
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

test "bf16 rounding of fp32 is to nearest even" {
    try std.testing.expectEqual(@as(u16, 0x3f80), toBf16(1.0));
    try std.testing.expectEqual(@as(u16, 0x3f80), toBf16(@bitCast(@as(u32, 0x3f808000)))); // a tie to even (down)
    try std.testing.expectEqual(@as(u16, 0x3f82), toBf16(@bitCast(@as(u32, 0x3f818000)))); // a tie to even (up)
    try std.testing.expectEqual(@as(u16, 0x3f81), toBf16(@bitCast(@as(u32, 0x3f808001))));
}
