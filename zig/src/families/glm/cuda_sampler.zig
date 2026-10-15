//! Sampled draws on upstream's sample.cu (#542 option C): every rank runs tf_draw.

const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const Span = @import("cuda_shard.zig").Span;
const Exchange = @import("cuda_forward.zig").Exchange;

pub const Sampling = lanes.Sampling;

/// The request's rule as the engine runs it: null (or temperature 0) decodes greedily.
pub fn check(s: ?Sampling) ?Sampling {
    const x = s orelse return null;
    return if (x.temperature > 0) x else null;
}

/// sample.cu's Rule: the seed, 1 / T, top_p, the candidates' near window, ln(min_p) and top_k (0: off).
pub const Rule = extern struct { seed: u64, inv_t: f32, top_p: f32, near: f32, min_log: f32, top_k: u32, pad: u32 = 0 };

/// The rule as nemotron/cuda_sampler.zig fills it.
pub fn rule(s: Sampling) Rule {
    return .{ .seed = s.seed, .inv_t = @floatCast(1.0 / @max(s.temperature, 1e-6)), .top_p = @floatCast(s.top_p), .near = 20.0, .min_log = @floatCast(s.minLog()), .top_k = s.top_k };
}

/// A rank's logits rows: bf16 [rows][its vocabulary span].
pub const Local = struct { rank: usize, logits: u64 };

pub const Sampler = struct {
    d: *const cuda.Driver,
    module: cuda.Module,
    tf_draw: cuda.Function,
    mask_fn: cuda.Function,
    vocab: usize,
    rows: usize,
    full: cuda.DeviceBuffer, // bf16 [rows][vocab]: every rank's columns in place
    rule: cuda.DeviceBuffer,
    meta: cuda.DeviceBuffer,
    tok: cuda.DeviceBuffer,
    prob: cuda.DeviceBuffer,
    bits: cuda.DeviceBuffer,

    pub fn init(d: *const cuda.Driver, vocab: usize, rows: usize) !Sampler {
        var m = try cuda.Module.load(d, cuda.kernels.sample);
        errdefer m.unload();
        const f = try m.function("tf_draw");
        const mask_fn = try m.function("tf_grammar_mask");
        var full = try cuda.DeviceBuffer.alloc(d, rows * vocab * 2);
        errdefer full.free();
        var r = try cuda.DeviceBuffer.alloc(d, @sizeOf(Rule));
        errdefer r.free();
        var meta = try cuda.DeviceBuffer.alloc(d, 16);
        errdefer meta.free();
        var tok = try cuda.DeviceBuffer.alloc(d, rows * 4);
        errdefer tok.free();
        const prob = try cuda.DeviceBuffer.alloc(d, rows * 4);
        errdefer prob.free();
        const bits = try cuda.DeviceBuffer.alloc(d, rows * ((vocab + 31) / 32) * 4);
        return .{ .d = d, .module = m, .tf_draw = f, .mask_fn = mask_fn, .vocab = vocab, .rows = rows, .full = full, .rule = r, .meta = meta, .tok = tok, .prob = prob, .bits = bits };
    }

    pub fn deinit(s: *Sampler) void {
        s.bits.free();
        s.prob.free();
        s.tok.free();
        s.meta.free();
        s.rule.free();
        s.full.free();
        s.module.unload();
    }

    /// `rows` rows of `mine` -> `out`, tokens at positions first, first + 1, ...; `spans`: every rank's, rank order.
    pub fn draw(s: *Sampler, stream: cuda.Stream, exchange: Exchange, spans: []const Span, mine: []const Local, rows: usize, first: u64, r: Rule, out: []u32) !void {
        std.debug.assert(rows <= s.rows and out.len >= rows and first >= 1);
        try s.gather(stream, exchange, spans, mine, rows);
        try s.choose(stream, rows, first, r, out);
    }

    fn gather(s: *Sampler, stream: cuda.Stream, exchange: Exchange, spans: []const Span, mine: []const Local, rows: usize) !void {
        const V = s.vocab;
        for (mine) |l| {
            const span = spans[l.rank];
            for (0..rows) |row| try s.copy(stream, s.full.ptr + (row * V + span.lo) * 2, l.logits + row * span.len() * 2, span.len() * 2);
        }
        switch (exchange) {
            .local => {},
            .nccl => |x| { // each rank's columns of each row from that rank (spans may differ in width: TP=3)
                try x.lib.check(x.lib.api.ncclGroupStart(), "ncclGroupStart");
                for (0..rows) |row| for (spans, 0..) |span, q| {
                    const at = s.full.ptr + (row * V + span.lo) * 2;
                    try x.lib.check(x.lib.api.ncclBroadcast(at, at, span.len(), .bf16, @intCast(q), x.comm, stream.handle), "ncclBroadcast");
                };
                try x.lib.check(x.lib.api.ncclGroupEnd(), "ncclGroupEnd");
            },
        }
    }

    fn choose(s: *Sampler, stream: cuda.Stream, rows: usize, first: u64, r: Rule, out: []u32) !void {
        const V = s.vocab;
        try s.rule.uploadAsync(0, std.mem.asBytes(&r), stream.handle);
        const meta = [4]i32{ @intCast(first - 1), 0, 0, 0 }; // row r keys meta[0] + r + 1
        try s.meta.uploadAsync(0, std.mem.asBytes(&meta), stream.handle);
        var args: cuda.Args = .{};
        args.add(s.full.ptr);
        args.add(@as(u32, @intCast(V)));
        args.add(s.rule.ptr);
        args.add(s.meta.ptr);
        args.add(@as(i32, 0));
        args.add(s.tok.ptr);
        args.add(s.prob.ptr);
        try cuda.launch.launch(s.tf_draw, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 1024 } }, stream, &args);
        try stream.synchronize();
        try s.tok.download(0, std.mem.sliceAsBytes(out[0..rows]));
    }

    /// The caller broadcasts `bits` to all ranks before this draw. Plain draws never consult it.
    pub fn masked(s: *Sampler, stream: cuda.Stream, exchange: Exchange, spans: []const Span, mine: []const Local, rows: usize, first: u64, r: ?Rule, top1: cuda.Function, out: []u32) !void {
        if (rows == 0 or rows > s.rows or out.len < rows or first == 0) return error.InvalidGrammarMasks;
        try s.gather(stream, exchange, spans, mine, rows);
        var args: cuda.Args = .{};
        args.add(s.full.ptr);
        args.add(@as(u32, @intCast(s.vocab)));
        args.add(s.bits.ptr);
        args.add(@as(u32, @intCast((s.vocab + 31) / 32)));
        try cuda.launch.launch(s.mask_fn, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 256 } }, stream, &args);
        if (r) |rule_| return s.choose(stream, rows, first, rule_, out);
        for (0..rows) |row| {
            var a: cuda.Args = .{};
            a.add(s.full.ptr + row * s.vocab * 2);
            a.add(@as(i64, @intCast(s.vocab)));
            a.add(@as(u32, 0));
            a.add(s.meta.ptr);
            try cuda.launch.launch(top1, .{ .grid = .{ .x = 1 }, .block = .{ .x = 1024 } }, stream, &a);
            try stream.synchronize();
            var pair: [2]u32 = undefined;
            try s.meta.download(0, std.mem.asBytes(&pair));
            if (!std.math.isFinite(@as(f32, @bitCast(pair[0])))) return error.NoAllowedToken;
            out[row] = pair[1];
        }
    }

    /// The whole rows a draw read (bf16 [rows][vocab]), for checks against the host rule.
    pub fn rowsRead(s: *const Sampler, rows: usize, host: []u16) !void {
        try s.full.download(0, std.mem.sliceAsBytes(host[0 .. rows * s.vocab]));
    }

    fn copy(s: *const Sampler, stream: cuda.Stream, dst: u64, src: u64, bytes: usize) !void {
        try s.d.check(s.d.api.cuMemcpyDtoDAsync_v2(dst, src, bytes, stream.handle), "cuMemcpyDtoDAsync");
    }
};

/// Reject a dead row before its paired CUDA command; tail bits outside the vocabulary cannot make it live.
pub fn validateMasks(bits: []const u32, rows: usize, vocab: usize) !void {
    const words = (vocab + 31) / 32;
    if (rows == 0 or vocab == 0 or bits.len != rows * words) return error.InvalidGrammarMasks;
    for (0..rows) |row| {
        var any: u32 = 0;
        for (bits[row * words ..][0..words], 0..) |word, i| {
            const valid: u32 = if (i + 1 == words and vocab % 32 != 0) (@as(u32, 1) << @as(u5, @intCast(vocab % 32))) - 1 else 0xffffffff;
            any |= word & valid;
        }
        if (any == 0) return error.NoAllowedToken;
    }
}

test "grammar masks require a live token in every row and ignore vocabulary padding" {
    try validateMasks(&.{ 1, 0, 0, 1 }, 2, 33);
    try std.testing.expectError(error.NoAllowedToken, validateMasks(&.{ 1, 0, 0, 2 }, 2, 33));
    try std.testing.expectError(error.NoAllowedToken, validateMasks(&.{ 0, 0 }, 1, 33));
    try std.testing.expectError(error.InvalidGrammarMasks, validateMasks(&.{1}, 1, 33));
}

test "rules as nemotron/cuda_sampler fills them; temperature 0 is greedy" {
    const s: Sampling = .{ .seed = 1234, .temperature = 0.5, .top_k = 20, .top_p = 0.95 };
    const r = rule(check(s).?);
    try std.testing.expectEqual(@as(f32, 2.0), r.inv_t);
    try std.testing.expectEqual(@as(u32, 20), r.top_k);
    try std.testing.expect(std.math.isInf(r.min_log));
    try std.testing.expectEqual(@as(usize, 32), @sizeOf(Rule));
    try std.testing.expectEqual(@as(?Sampling, null), check(.{ .seed = 0, .temperature = 0 }));
}
