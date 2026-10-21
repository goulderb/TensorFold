//! Clef's joint schema head on CUDA for one encoded record. It reimplements the forward pass of JointSchemaHead from
//! Cloudflare's clef-flash joint_schema_model.py (Apache-2.0, revision fde727a2); no code is copied. See NOTICE.
const std = @import("std");
const cuda = @import("cuda");
const record = @import("record.zig");
const checkpoint = @import("head_checkpoint.zig");

const cfg = checkpoint.pinned;

/// The record extents every buffer is sized for: the token limit and the most options and questions it admits.
pub const Dims = struct {
    tokens: usize,
    options: usize,
    questions: usize,

    pub fn of(l: record.Limits) Dims {
        return .{ .tokens = l.tokens, .options = l.maxOptions(), .questions = l.maxQuestions() };
    }

    /// Rows of the option and field work buffers, which the two phases share.
    pub fn rows(d: Dims) usize {
        return @max(d.options, d.questions);
    }
};

pub const hidden = cfg.hidden_size;
pub const width = cfg.width;
pub const feedforward = cfg.feedforward;

/// The per-request tables uploaded once from pinned host staging: spans, the option map and question types.
pub const Tables = struct {
    pub const Range = extern struct { first: u32, count: u32 };
    pub const Part = enum { question_spans, option_spans, lexical_spans, option_question, question_options, types };

    offsets: std.EnumArray(Part, usize),
    bytes: usize,

    pub fn layout(d: Dims) !Tables {
        var t: Tables = .{ .offsets = .initFill(0), .bytes = 0 };
        for (std.enums.values(Part)) |p| {
            const n: usize = switch (p) {
                .question_spans => d.questions * @sizeOf(record.Span),
                .option_spans, .lexical_spans => d.options * @sizeOf(record.Span),
                .option_question => d.options * 4,
                .question_options => d.questions * @sizeOf(Range),
                .types => d.questions * 4,
            };
            t.offsets.set(p, t.bytes);
            t.bytes = try alignUp(try std.math.add(usize, t.bytes, n));
        }
        return t;
    }
};

pub const Buf = enum {
    input,
    hn,
    memory,
    memory_norm,
    k,
    v,
    qvec,
    anchor,
    ctx,
    lex,
    acc,
    part,
    routed,
    fields,
    base,
    summary,
    normed,
    q,
    attn,
    out,
    wide,
    gvec,
    residual,
    prior,
    cosine,
    logits,
    tables,
};

/// One device arena for every per-request buffer, sized at open from `Dims`; no request allocates.
pub const Scratch = struct {
    offsets: std.EnumArray(Buf, usize),
    sizes: std.EnumArray(Buf, usize),
    tables: Tables,
    bytes: usize,

    /// `upload_hidden`: head-only engines own a [tokens, hidden] destination for caller rows.
    pub fn layout(d: Dims, upload_hidden: bool) !Scratch {
        const tables = try Tables.layout(d);
        var s: Scratch = .{ .offsets = .initFill(0), .sizes = .initFill(0), .tables = tables, .bytes = 0 };
        const r = d.rows();
        for (std.enums.values(Buf)) |b| {
            const n: usize = switch (b) {
                .input => if (upload_hidden) try bf16s(.{ d.tokens, hidden }) else 0,
                .hn => try bf16s(.{ d.tokens, hidden }),
                .memory, .memory_norm, .k, .v => try bf16s(.{ d.tokens, width }),
                .qvec, .anchor => try bf16s(.{ d.questions, hidden }),
                .ctx, .lex => try bf16s(.{ d.options, hidden }),
                .acc => 4 * @max(try mul(.{ d.tokens, width }), try mul(.{ r, feedforward })),
                .part => try partBytes(d),
                .routed => try bf16s(.{ d.options, width }),
                .fields, .base, .summary => try bf16s(.{ d.questions, width }),
                .normed, .q, .attn, .out => try bf16s(.{ r, width }),
                .wide => try bf16s(.{ r, @max(feedforward, 4 * width) }),
                .gvec => width * 2,
                .residual, .prior, .cosine, .logits => d.options * 2,
                .tables => tables.bytes,
            };
            s.offsets.set(b, s.bytes);
            s.sizes.set(b, n);
            s.bytes = try alignUp(try std.math.add(usize, s.bytes, n));
        }
        return s;
    }
};

/// The lane matmul's split-K partials (fp32 [sk][m][n]) for the largest call below its fused row count, sm_80 and
/// sm_89 having no clusters. Partials grow with the rows below that count and vanish at it, so the bound sits there.
fn partBytes(d: Dims) !usize {
    const Use = struct { rows: usize, n: usize, k: usize };
    const r = d.rows();
    const uses = [_]Use{
        .{ .rows = d.tokens, .n = width, .k = hidden },
        .{ .rows = r, .n = width, .k = hidden },
        .{ .rows = d.tokens, .n = width, .k = width },
        .{ .rows = r, .n = width, .k = width },
        .{ .rows = r, .n = feedforward, .k = width },
        .{ .rows = r, .n = width, .k = feedforward },
        .{ .rows = d.options, .n = width, .k = 4 * width },
        .{ .rows = d.options, .n = 1, .k = width },
    };
    var most: usize = 0;
    for (uses) |u| {
        const m = @min(u.rows, cuda.qmmf.fused_rows - 1);
        if (m == 0) continue;
        const p = cuda.qmmf.plan(m, u.n, u.k, false);
        if (p.reduce) most = @max(most, try mul(.{ p.sk, m, u.n, 4 }));
    }
    return most;
}

fn mul(dims: anytype) !usize {
    var n: usize = 1;
    inline for (dims) |x| n = try std.math.mul(usize, n, x);
    return n;
}

fn bf16s(dims: anytype) !usize {
    return std.math.mul(usize, try mul(dims), 2);
}

fn alignUp(n: usize) !usize {
    return std.math.mul(usize, (try std.math.add(usize, n, 255)) / 256, 256);
}

/// clef_head.cu's image; empty in host-only builds.
pub const image: []const u8 = &Image.bytes;
const Image = struct {
    const bytes align(16) = @embedFile("fatbin_clef_head").*;
};

/// clef_head.cu's kernels; each binds as "tf_clef_" ++ its field name.
pub const Fns = struct {
    layer_norm_4096: cuda.Function,
    span_mean: cuda.Function,
    option_queries: cuda.Function,
    attention: cuda.Function,
    segment_summary: cuda.Function,
    fields: cuda.Function,
    normalize: cuda.Function,
    prior_dot: cuda.Function,
    score_features: cuda.Function,
    logit_combine: cuda.Function,
};

/// The head's kernels: clef_head.cu's, vit.cu's LayerNorm, bias and pointwise, and the BF16 lane matmul.
pub const Kernels = struct {
    module: cuda.Module,
    fns: Fns,
    vit: cuda.vit.Kernels,
    lane: cuda.qmmf.Lane,

    pub fn load(ctx: *const cuda.Context) !Kernels {
        if (image.len == 0 or !cuda.kernels.available) return error.BuiltWithoutKernels;
        const major: u32 = @intCast(try ctx.attribute(.compute_capability_major));
        if (major < 8) return error.Bf16NeedsAmpere;
        var module = try cuda.Module.load(ctx.d, image);
        errdefer module.unload();
        var fns: Fns = undefined;
        inline for (@typeInfo(Fns).@"struct".field_names) |name| @field(fns, name) = try module.function("tf_clef_" ++ name);
        var vit = try cuda.vit.Kernels.load(ctx.d);
        errdefer vit.unload();
        return .{ .module = module, .fns = fns, .vit = vit, .lane = try cuda.qmmf.Lane.load(ctx.d, major) };
    }

    pub fn unload(k: *Kernels) void {
        k.lane.unload();
        k.vit.unload();
        k.module.unload();
        k.* = undefined;
    }
};

/// The head's weight arena: the plan's layout at a device base address.
pub const Weights = struct {
    base: u64,
    layout: checkpoint.Layout,

    fn at(w: Weights, offset: usize) u64 {
        return w.base + offset;
    }
};

/// The upload target of loadWeights: each binding's stored bytes into the arena, packed when it is a projection.
pub const WeightSink = struct {
    arena: cuda.DeviceBuffer,
    stage: cuda.DeviceBuffer,
    stream: cuda.Stream,
    pack: *const cuda.modelopt.Packer,

    pub fn tensor(s: *WeightSink, b: checkpoint.Binding, bytes: []const u8) !void {
        switch (b.placement) {
            .raw => try s.arena.upload(b.device_offset, bytes),
            .lane, .lane_qkv => {
                const parts: usize = if (b.placement == .lane_qkv) 3 else 1;
                const n = b.entry.shape[0] / parts;
                const k = b.entry.shape[1];
                try s.stage.upload(0, bytes);
                for (0..parts) |i| try s.pack.bf16Lane(s.stream, s.stage.ptr + i * n * k * 2, try s.arena.at(b.device_offset + i * checkpoint.laneBytes(n, k)), n, k);
                try s.stream.synchronize();
            },
        }
    }
};

/// Uploads every binding through `plan.upload` (digest first): stored tensors as they are, projections staged and
/// packed by the shared BF16 lane packer, in_proj tensors as three lanes. Synchronous; `stage` holds load_stage bytes.
pub fn loadWeights(plan: *const checkpoint.Plan, io: std.Io, path: []const u8, pin: [32]u8, arena: cuda.DeviceBuffer, stage: cuda.DeviceBuffer, stream: cuda.Stream, pack: *const cuda.modelopt.Packer) !Weights {
    if (arena.len < plan.bytes.device_weights or stage.len < plan.bytes.load_stage) return error.InvalidHeadAllocation;
    var sink: WeightSink = .{ .arena = arena, .stage = stage, .stream = stream, .pack = pack };
    try plan.upload(io, path, pin, &sink);
    return .{ .base = arena.ptr, .layout = plan.layout };
}

/// Writes one record's tables into `host` (Tables.bytes) in the device layout.
pub fn stageTables(t: Tables, rec: record.Record, host: []u8) void {
    const put = struct {
        fn f(comptime T: type, h: []u8, at: usize, i: usize, v: T) void {
            @memcpy(h[at + i * @sizeOf(T) ..][0..@sizeOf(T)], std.mem.asBytes(&v));
        }
    }.f;
    var lexical: u32 = 0;
    for (rec.options, rec.option_question, 0..) |o, q, i| {
        put(record.Span, host, t.offsets.get(.option_spans), i, o);
        const len = o.end - o.start;
        put(record.Span, host, t.offsets.get(.lexical_spans), i, .{ .start = lexical, .end = lexical + len });
        lexical += len;
        put(u32, host, t.offsets.get(.option_question), i, q);
    }
    for (rec.questions, 0..) |q, i| {
        put(record.Span, host, t.offsets.get(.question_spans), i, q.span);
        put(Tables.Range, host, t.offsets.get(.question_options), i, .{ .first = q.first, .count = q.count });
        put(u32, host, t.offsets.get(.types), i, @backingInt(q.kind));
    }
}

pub const Inputs = struct {
    /// [ids.len, hidden] BF16 rows after the backbone's final norm; read only.
    hidden: u64,
    /// [lexical_rows, hidden] BF16 output-table rows of every option span, in option order.
    lexical: u64,
    rec: *const record.Record,
};

const ln_eps: f32 = 1e-5;
const normalize_eps: f32 = 1e-12;
const cosine_eps: f32 = 1e-8;
const ln_100: f32 = 4.605170185988092;
const head_dim = cfg.width / cfg.heads;

/// One record through the head on `stream`, ending with BF16 logits in the scratch `.logits` buffer. Every buffer
/// is the scratch arena's; the tables must already be uploaded. Nothing is allocated or synchronized here.
pub const Head = struct {
    k: *const Kernels,
    w: Weights,
    scratch: Scratch,
    base: u64,
    dims: Dims,

    pub fn at(h: *const Head, b: Buf) u64 {
        return h.base + h.scratch.offsets.get(b);
    }

    fn table(h: *const Head, p: Tables.Part) u64 {
        return h.at(.tables) + h.scratch.tables.offsets.get(p);
    }

    pub fn run(h: *const Head, s: cuda.Stream, in: Inputs) !void {
        const rec = in.rec;
        const t = rec.ids.len;
        const nq = rec.questions.len;
        const no = rec.options.len;
        if (t > h.dims.tokens or nq > h.dims.questions or no > h.dims.options or nq == 0 or no == 0) return error.InvalidHeadShape;
        const l = &h.w.layout;
        const hn = h.at(.hn);
        const global = hn + (t - 1) * hidden * 2;
        const memory = h.at(.memory);
        const routed = h.at(.routed);
        const fields = h.at(.fields);
        const normed = h.at(.normed);
        const q = h.at(.q);
        const attn = h.at(.attn);
        const out = h.at(.out);
        const wide = h.at(.wide);

        // hidden_norm, memory, the global row and the span means
        try h.go(s, h.k.fns.layer_norm_4096, t, 1, 256, 0, .{ in.hidden, h.w.at(l.hidden_norm.weight), h.w.at(l.hidden_norm.bias), hn, int(hidden), ln_eps });
        try h.linear(s, hn, t, l.memory, memory);
        try h.go(s, h.k.fns.span_mean, nq, 1, 256, 0, .{ hn, int(hidden), h.table(.question_spans), h.at(.qvec) });
        try h.go(s, h.k.fns.span_mean, no, 1, 256, 0, .{ hn, int(hidden), h.table(.option_spans), h.at(.ctx) });
        try h.go(s, h.k.fns.span_mean, no, 1, 256, 0, .{ in.lexical, int(hidden), h.table(.lexical_spans), h.at(.lex) });

        // option queries: (context + lexical) + the question's projection
        try h.linear(s, h.at(.ctx), no, l.option_context, q);
        try h.linear(s, h.at(.lex), no, l.option_lexical, attn);
        try h.linear(s, h.at(.qvec), nq, l.option_question, out);
        try h.go(s, h.k.fns.option_queries, no, 1, 256, 0, .{ q, attn, out, h.table(.option_question), routed, int(width) });

        for (&l.routing) |*r| {
            try h.norm(s, routed, no, r.query_norm, normed);
            try h.norm(s, memory, t, r.memory_norm, h.at(.memory_norm));
            try h.attention(s, normed, no, h.at(.memory_norm), t, r.attention, routed);
            try h.norm(s, routed, no, r.feedforward_norm, normed);
            try h.feedforward(s, normed, no, r.up, r.down, routed);
        }

        // fields: base + summary_norm(attention summary of the routed options) + global + type embedding
        const base = h.at(.base);
        try h.linear(s, h.at(.qvec), nq, l.question, base);
        var most: usize = 0;
        for (rec.questions) |x| most = @max(most, x.count);
        try h.go(s, h.k.fns.segment_summary, nq, 1, 256, @intCast(most * 4), .{ routed, base, h.table(.question_options), h.at(.summary), int(width), @sqrt(@as(f32, width)) });
        try h.norm(s, h.at(.summary), nq, l.option_summary_norm, normed);
        try h.linear(s, global, 1, l.global, h.at(.gvec));
        try h.go(s, h.k.fns.fields, nq, 1, 256, 0, .{ base, normed, h.at(.gvec), h.w.at(l.type_embedding), h.table(.types), fields, int(width) });

        // norm-first decoder layers: self-attention over the fields, cross-attention over the un-normed memory
        for (&l.decoders) |*x| {
            try h.norm(s, fields, nq, x.norm1, normed);
            try h.attention(s, normed, nq, normed, nq, x.self_attention, fields);
            try h.norm(s, fields, nq, x.norm2, normed);
            try h.attention(s, normed, nq, memory, t, x.cross_attention, fields);
            try h.norm(s, fields, nq, x.norm3, normed);
            try h.feedforward(s, normed, nq, x.up, x.down, fields);
        }
        try h.norm(s, fields, nq, l.field_norm, base);

        // the lexical prior: normalized (question + global) anchors against normalized lexical means
        try h.go(s, h.k.fns.normalize, nq, 1, 256, 0, .{ h.at(.qvec), global, h.at(.anchor), int(hidden), normalize_eps });
        try h.go(s, h.k.fns.normalize, no, 1, 256, 0, .{ h.at(.lex), @as(u64, 0), h.at(.ctx), int(hidden), normalize_eps });
        try h.go(s, h.k.fns.prior_dot, no, 1, 256, 0, .{ h.at(.ctx), h.at(.anchor), h.table(.option_question), h.at(.prior), int(hidden) });

        // the joint score: cosine of field and option, a residual scorer over their features, the gated sum
        try h.norm(s, routed, no, l.option_norm, normed);
        try h.go(s, h.k.fns.score_features, no, 1, 256, 0, .{ base, normed, h.table(.option_question), h.at(.cosine), wide, int(width), cosine_eps });
        try h.linear(s, wide, no, l.scorer_up, out);
        try h.k.vit.pointwise(s, out, null, out, no * width, .gelu);
        try h.linear(s, out, no, l.scorer_down, h.at(.residual));
        try h.go(s, h.k.fns.logit_combine, (no + 255) / 256, 1, 256, 0, .{ h.w.at(l.prior_logit_scale), h.w.at(l.joint_logit_scale), h.w.at(l.residual_gate), h.at(.prior), h.at(.cosine), h.at(.residual), h.at(.logits), int(no), ln_100 });
    }

    /// torch.nn.MultiheadAttention without masks or weights: Q from `x`, K and V from `kv` (each its own Linear, as
    /// the packed in_proj splits), attention, out_proj, then `residual` += out (BF16).
    fn attention(h: *const Head, s: cuda.Stream, x: u64, rows: usize, kv: u64, keys: usize, a: checkpoint.Attention, residual: u64) !void {
        const q = h.at(.q);
        const k = h.at(.k);
        const v = h.at(.v);
        try h.linear(s, x, rows, a.q, q);
        try h.linear(s, kv, keys, a.k, k);
        try h.linear(s, kv, keys, a.v, v);
        try h.go(s, h.k.fns.attention, rows, cfg.heads, head_dim, 0, .{ q, k, v, h.at(.attn), int(keys), int(width), 1.0 / @sqrt(@as(f32, head_dim)) });
        try h.linear(s, h.at(.attn), rows, a.out, h.at(.out));
        try h.k.vit.pointwise(s, h.at(.out), residual, residual, rows * width, .none);
    }

    /// Linear, exact GELU, Linear, then `residual` += out (BF16).
    fn feedforward(h: *const Head, s: cuda.Stream, x: u64, rows: usize, up: checkpoint.Linear, down: checkpoint.Linear, residual: u64) !void {
        const wide = h.at(.wide);
        try h.linear(s, x, rows, up, wide);
        try h.k.vit.pointwise(s, wide, null, wide, rows * up.n, .gelu);
        try h.linear(s, wide, rows, down, h.at(.out));
        try h.k.vit.pointwise(s, h.at(.out), residual, residual, rows * width, .none);
    }

    /// y = bf16(x W^T + b): the lane matmul's FP32 sums, then one rounding with the bias; bf16 directly without one.
    fn linear(h: *const Head, s: cuda.Stream, x: u64, rows: usize, l: checkpoint.Linear, y: u64) !void {
        const w = cuda.modelopt.bf16Weight(h.w.at(l.weight), l.n, l.k);
        if (l.bias) |b| {
            try h.k.lane.matmulOut(s, x, l.k, rows, w, h.at(.acc), .f32, h.at(.part));
            try h.k.vit.bias(s, h.at(.acc), h.w.at(b), y, rows, l.n);
        } else try h.k.lane.matmulOut(s, x, l.k, rows, w, y, .bf16, h.at(.part));
    }

    fn norm(h: *const Head, s: cuda.Stream, x: u64, rows: usize, n: checkpoint.Norm, y: u64) !void {
        try h.k.vit.layerNorm(s, x, h.w.at(n.weight), h.w.at(n.bias), y, rows, width, ln_eps);
    }

    fn go(h: *const Head, s: cuda.Stream, f: cuda.Function, gx: usize, gy: usize, threads: u32, shared: u32, args: anytype) !void {
        _ = h;
        var a: cuda.Args = .{};
        inline for (args) |x| a.add(x);
        try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(gx), .y = @intCast(gy), .z = 1 }, .block = .{ .x = threads, .y = 1, .z = 1 }, .shared = shared }, s, &a);
    }
};

fn int(n: usize) c_int {
    return @intCast(n);
}

test "scratch at 4096 tokens holds every buffer apart, aligned, and the largest split-K partials" {
    const d = Dims.of(try record.Limits.init(4096, 248320));
    const s = try Scratch.layout(d, false);
    var end: usize = 0;
    for (std.enums.values(Buf)) |b| {
        try std.testing.expect(s.offsets.get(b) % 256 == 0 and s.offsets.get(b) >= end);
        end = s.offsets.get(b) + s.sizes.get(b);
    }
    try std.testing.expectEqual(@as(usize, 0), s.sizes.get(.input));
    try std.testing.expectEqual(@as(usize, 8 * 255 * 1024 * 4), s.sizes.get(.part));
    try std.testing.expectEqual(@as(usize, 32 << 20), s.sizes.get(.acc));
    try std.testing.expectEqual(@as(usize, 2048 * 4096 * 2), s.sizes.get(.wide));
    const with_input = try Scratch.layout(d, true);
    try std.testing.expectEqual(s.bytes + (32 << 20), with_input.bytes);
    const t = try Tables.layout(d);
    try std.testing.expectEqual(@as(usize, 2048 * (8 + 8 + 8 + 4 + 8 + 4)), t.bytes);
    try std.testing.expectEqual(@as(usize, 2048 * (8 + 8 + 8 + 4)), t.offsets.get(.question_options));
}

test "the scratch bound follows the token limit" {
    const small = try Scratch.layout(Dims.of(try record.Limits.init(64, 10)), false);
    const large = try Scratch.layout(Dims.of(try record.Limits.init(4096, 10)), false);
    try std.testing.expect(small.bytes < large.bytes);
    try std.testing.expectEqual(@as(usize, 32 * 4096 * 2), small.sizes.get(.ctx));
}
