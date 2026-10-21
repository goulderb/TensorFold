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
