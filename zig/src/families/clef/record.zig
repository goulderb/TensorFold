//! One encoded Clef record (official encode_record's input ids and spans) validated once at the C boundary.
//! Everything past `parse` trusts a `Record`.
const std = @import("std");

pub const max_tokens = 4096;

pub const Limits = struct {
    tokens: u32,
    vocab: u32,

    pub fn init(tokens: u32, vocab: u32) !Limits {
        if (tokens < 2 or tokens > max_tokens or vocab == 0) return error.InvalidLimits;
        return .{ .tokens = tokens, .vocab = vocab };
    }

    /// Spans are nonempty and disjoint, with template tokens between them, so neither count reaches half the tokens.
    pub fn maxOptions(l: Limits) u32 {
        return l.tokens / 2;
    }

    pub fn maxQuestions(l: Limits) u32 {
        return l.tokens / 2;
    }
};

pub const QuestionType = enum(u8) { noul = 0, choice = 1, score = 2 };

/// [start, end): nonempty and inside the record's ids.
pub const Span = extern struct { start: u32, end: u32 };

pub const Question = struct {
    kind: QuestionType,
    span: Span,
    first: u32,
    count: u32,
};

/// The C record (tfclef.h tf_clef_record and its parts), as the caller lays it out.
pub const C = struct {
    pub const Question = extern struct { kind: u8, question: Span, first_option: u32, option_count: u32 };
    pub const Record = extern struct {
        ids: ?[*]const u32,
        n_ids: u32,
        questions: ?[*]const C.Question,
        n_questions: u32,
        options: ?[*]const Span,
        n_options: u32,
    };
};

/// Caller-owned storage for the derived parts, sized by `Limits`; a record borrows it until the next parse.
pub const Scratch = struct {
    questions: []Question,
    option_question: []u16,

    pub fn alloc(gpa: std.mem.Allocator, l: Limits) !Scratch {
        const questions = try gpa.alloc(Question, l.maxQuestions());
        errdefer gpa.free(questions);
        return .{ .questions = questions, .option_question = try gpa.alloc(u16, l.maxOptions()) };
    }

    pub fn free(s: *Scratch, gpa: std.mem.Allocator) void {
        gpa.free(s.questions);
        gpa.free(s.option_question);
        s.* = undefined;
    }
};

pub const Record = struct {
    ids: []const u32,
    questions: []const Question,
    /// Every question's option spans, flattened in question order.
    options: []const Span,
    /// Option index to question index, derived here and never taken from the caller.
    option_question: []const u16,
    /// The summed option span lengths: the rows of the lexical gather.
    lexical_rows: u32,

    pub const Error = error{ Empty, TooLong, NullPointer, IdOutOfVocab, BadSpan, BadType, OptionsNotTiled, TooMany, LexicalRows };

    /// TooLong is the only capacity refusal; every other error is an invalid record.
    pub fn parse(c: *const C.Record, l: Limits, s: Scratch) Error!Record {
        if (c.n_ids == 0) return error.Empty;
        if (c.n_ids > l.tokens) return error.TooLong;
        if (c.n_questions == 0 or c.n_options == 0) return error.Empty;
        if (c.n_questions > l.maxQuestions() or c.n_options > l.maxOptions()) return error.TooMany;
        if (s.questions.len < l.maxQuestions() or s.option_question.len < l.maxOptions()) return error.TooMany;
        const ids = (c.ids orelse return error.NullPointer)[0..c.n_ids];
        const cq = (c.questions orelse return error.NullPointer)[0..c.n_questions];
        const options = (c.options orelse return error.NullPointer)[0..c.n_options];
        for (ids) |id| if (id >= l.vocab) return error.IdOutOfVocab;
        var next: u32 = 0;
        for (cq, s.questions[0..cq.len], 0..) |q, *out, qi| {
            const kind = std.enums.fromInt(QuestionType, q.kind) orelse return error.BadType;
            try span(q.question, c.n_ids);
            if (q.option_count == 0 or q.first_option != next or q.option_count > c.n_options - next) return error.OptionsNotTiled;
            @memset(s.option_question[next..][0..q.option_count], @intCast(qi));
            next += q.option_count;
            out.* = .{ .kind = kind, .span = q.question, .first = q.first_option, .count = q.option_count };
        }
        if (next != c.n_options) return error.OptionsNotTiled;
        var lexical: u32 = 0;
        for (options) |o| {
            try span(o, c.n_ids);
            lexical += o.end - o.start;
            if (lexical > c.n_ids) return error.LexicalRows;
        }
        return .{ .ids = ids, .questions = s.questions[0..cq.len], .options = options, .option_question = s.option_question[0..options.len], .lexical_rows = lexical };
    }
};

fn span(s: Span, n: u32) !void {
    if (s.start >= s.end or s.end > n) return error.BadSpan;
}

const Builder = struct {
    ids: [64]u32 = @splat(1),
    questions: [4]C.Question = undefined,
    options: [8]Span = undefined,
    c: C.Record = undefined,

    /// noul over 10..12 with options 13..15 and 16..17, then choice over 20..22 with one option 23..26.
    fn valid(b: *Builder) *C.Record {
        b.questions[0] = .{ .kind = 0, .question = .{ .start = 10, .end = 12 }, .first_option = 0, .option_count = 2 };
        b.questions[1] = .{ .kind = 1, .question = .{ .start = 20, .end = 22 }, .first_option = 2, .option_count = 1 };
        b.options[0] = .{ .start = 13, .end = 15 };
        b.options[1] = .{ .start = 16, .end = 17 };
        b.options[2] = .{ .start = 23, .end = 26 };
        b.c = .{ .ids = &b.ids, .n_ids = 30, .questions = &b.questions, .n_questions = 2, .options = &b.options, .n_options = 3 };
        return &b.c;
    }
};

fn testScratch() !Scratch {
    return Scratch.alloc(std.testing.allocator, try Limits.init(64, 100));
}

test "a valid record derives its option-to-question map and lexical rows" {
    var s = try testScratch();
    defer s.free(std.testing.allocator);
    var b: Builder = .{};
    const r = try Record.parse(b.valid(), try Limits.init(64, 100), s);
    try std.testing.expectEqualSlices(u16, &.{ 0, 0, 1 }, r.option_question);
    try std.testing.expectEqual(@as(u32, 6), r.lexical_rows);
    try std.testing.expectEqual(QuestionType.choice, r.questions[1].kind);
    try std.testing.expectEqual(@as(usize, 30), r.ids.len);
}

test "each malformed record is refused with its own reason, and only overlength is capacity" {
    var s = try testScratch();
    defer s.free(std.testing.allocator);
    const l = try Limits.init(64, 100);
    const Case = struct { name: []const u8, err: Record.Error, edit: *const fn (*Builder) void };
    const cases = [_]Case{
        .{ .name = "empty ids", .err = error.Empty, .edit = struct {
            fn f(b: *Builder) void {
                b.c.n_ids = 0;
            }
        }.f },
        .{ .name = "overlength", .err = error.TooLong, .edit = struct {
            fn f(b: *Builder) void {
                b.c.n_ids = 65;
            }
        }.f },
        .{ .name = "no questions", .err = error.Empty, .edit = struct {
            fn f(b: *Builder) void {
                b.c.n_questions = 0;
            }
        }.f },
        .{ .name = "null ids", .err = error.NullPointer, .edit = struct {
            fn f(b: *Builder) void {
                b.c.ids = null;
            }
        }.f },
        .{ .name = "null options", .err = error.NullPointer, .edit = struct {
            fn f(b: *Builder) void {
                b.c.options = null;
            }
        }.f },
        .{ .name = "out of vocabulary", .err = error.IdOutOfVocab, .edit = struct {
            fn f(b: *Builder) void {
                b.ids[29] = 100;
            }
        }.f },
        .{ .name = "empty question span", .err = error.BadSpan, .edit = struct {
            fn f(b: *Builder) void {
                b.questions[1].question = .{ .start = 21, .end = 21 };
            }
        }.f },
        .{ .name = "question span past the ids", .err = error.BadSpan, .edit = struct {
            fn f(b: *Builder) void {
                b.questions[0].question = .{ .start = 28, .end = 31 };
            }
        }.f },
        .{ .name = "empty option span", .err = error.BadSpan, .edit = struct {
            fn f(b: *Builder) void {
                b.options[1] = .{ .start = 16, .end = 16 };
            }
        }.f },
        .{ .name = "option span past the ids", .err = error.BadSpan, .edit = struct {
            fn f(b: *Builder) void {
                b.options[2] = .{ .start = 23, .end = 31 };
            }
        }.f },
        .{ .name = "invalid type", .err = error.BadType, .edit = struct {
            fn f(b: *Builder) void {
                b.questions[0].kind = 3;
            }
        }.f },
        .{ .name = "options out of order", .err = error.OptionsNotTiled, .edit = struct {
            fn f(b: *Builder) void {
                b.questions[1].first_option = 1;
                b.questions[1].option_count = 2;
            }
        }.f },
        .{ .name = "question without options", .err = error.OptionsNotTiled, .edit = struct {
            fn f(b: *Builder) void {
                b.questions[1].option_count = 0;
            }
        }.f },
        .{ .name = "options not covered", .err = error.OptionsNotTiled, .edit = struct {
            fn f(b: *Builder) void {
                b.c.n_options = 4;
                b.options[3] = .{ .start = 27, .end = 28 };
            }
        }.f },
        .{ .name = "options overrun", .err = error.OptionsNotTiled, .edit = struct {
            fn f(b: *Builder) void {
                b.questions[1].option_count = 2;
            }
        }.f },
        .{ .name = "too many options", .err = error.TooMany, .edit = struct {
            fn f(b: *Builder) void {
                b.c.n_options = 33;
            }
        }.f },
        .{ .name = "too many questions", .err = error.TooMany, .edit = struct {
            fn f(b: *Builder) void {
                b.c.n_questions = 33;
            }
        }.f },
        .{ .name = "overlapping options exceed the ids", .err = error.LexicalRows, .edit = struct {
            fn f(b: *Builder) void {
                b.c.n_ids = 5;
                b.questions[0].question = .{ .start = 0, .end = 1 };
                b.questions[1].question = .{ .start = 1, .end = 2 };
                b.options[0] = .{ .start = 0, .end = 5 };
                b.options[1] = .{ .start = 0, .end = 5 };
                b.options[2] = .{ .start = 0, .end = 5 };
            }
        }.f },
    };
    for (cases) |case| {
        var b: Builder = .{};
        _ = b.valid();
        case.edit(&b);
        std.testing.expectError(case.err, Record.parse(&b.c, l, s)) catch |e| {
            std.debug.print("case {s}\n", .{case.name});
            return e;
        };
    }
}

test "limits bound the counts by the token limit and refuse impossible limits" {
    const l = try Limits.init(4096, 248320);
    try std.testing.expectEqual(@as(u32, 2048), l.maxOptions());
    try std.testing.expectEqual(@as(u32, 2048), l.maxQuestions());
    try std.testing.expectError(error.InvalidLimits, Limits.init(4097, 1));
    try std.testing.expectError(error.InvalidLimits, Limits.init(1, 1));
    try std.testing.expectError(error.InvalidLimits, Limits.init(16, 0));
}

test "a record at the token limit with the most options parses without allocation" {
    const l = try Limits.init(4096, 8);
    var s = try Scratch.alloc(std.testing.allocator, l);
    defer s.free(std.testing.allocator);
    const ids = try std.testing.allocator.alloc(u32, 4096);
    defer std.testing.allocator.free(ids);
    @memset(ids, 7);
    const options = try std.testing.allocator.alloc(Span, 2048);
    defer std.testing.allocator.free(options);
    for (options, 0..) |*o, i| o.* = .{ .start = @intCast(2 * i), .end = @intCast(2 * i + 1) };
    const q = [_]C.Question{.{ .kind = 2, .question = .{ .start = 4094, .end = 4096 }, .first_option = 0, .option_count = 2048 }};
    const c: C.Record = .{ .ids = ids.ptr, .n_ids = 4096, .questions = &q, .n_questions = 1, .options = options.ptr, .n_options = 2048 };
    const r = try Record.parse(&c, l, s);
    try std.testing.expectEqual(@as(u32, 2048), r.lexical_rows);
    try std.testing.expectEqual(@as(u16, 0), r.option_question[2047]);
}
