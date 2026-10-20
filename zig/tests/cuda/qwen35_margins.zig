//! tf-qwen35-margins: teacher-forced logit margins of the served Qwen ModelOpt chain against reference attention.
//!
//! Opens the chain as the native server does (internal split, 16-row prefill) three times: reference attention with
//! BF16 KV (the oracle), chunked attention with BF16 KV, then chunked attention with FP8 KV. Each pass feeds every
//! input sequence through appendLogitRows: row r is the next-token logits after tokens[0..r]. Each row's record
//! holds a Wyhash of the whole BF16 logit row, its top four (id, logit) and its logits at the probe ids: the
//! input's own probes for that row (a CPU reference's top two), then the oracle's top two on the chunked passes,
//! then the chunked BF16 pass's top two on the FP8 pass.
//!
//! Usage: tf-qwen35-margins CHECKPOINT INPUT.json OUTPUT.jsonl
//! INPUT.json: {"context": N, "sequences": [{"name": "...", "tokens": [...], "probes": [{"row": r, "ids": [a, b]}]}]}
const std = @import("std");
const qwen = @import("qwen35_cuda");

const Probe = struct { row: usize, ids: []const u32 };
const Sequence = struct { name: []const u8, tokens: []const u32, probes: []const Probe = &.{} };
const Input = struct { context: usize, sequences: []const Sequence };
const Top = [4]struct { id: u32, logit: f32 };

fn logit(bits: u16) f32 {
    return @bitCast(@as(u32, bits) << 16);
}

/// The row's four largest logits, ties to the lower id (the argmax kernels' rule).
fn top4(row: []const u16) !Top {
    var top: Top = @splat(.{ .id = std.math.maxInt(u32), .logit = -std.math.inf(f32) });
    for (row, 0..) |bits, i| {
        const v = logit(bits);
        if (!std.math.isFinite(v)) return error.NonfiniteLogits;
        if (v <= top[3].logit) continue;
        var at: usize = 3;
        while (at > 0 and v > top[at - 1].logit) : (at -= 1) top[at] = top[at - 1];
        top[at] = .{ .id = @intCast(i), .logit = v };
    }
    return top;
}

const Journal = struct {
    a: std.mem.Allocator,
    io: std.Io,
    file: std.Io.File,
    fn write(j: Journal, value: anytype) !void {
        var out: std.Io.Writer.Allocating = .init(j.a);
        defer out.deinit();
        try std.json.Stringify.value(value, .{}, &out.writer);
        try out.writer.writeByte('\n');
        try j.file.writeStreamingAll(j.io, out.written());
    }
};

const Pair = [2]u32;
/// Per sequence, per row: a pass's top two.
const Tops = [][]Pair;

const Pass = struct {
    /// Journal label: "ref" (reference attention, BF16 KV), "bf16" and "fp8" (chunked attention).
    label: []const u8,
    mode: qwen.engine.AttentionMode,
    kv8: bool,
    /// Earlier passes whose top two this pass probes, in order; null slots are skipped.
    probe: [2]?Tops = .{ null, null },
    /// Where this pass records its own top two for later passes.
    record: ?Tops = null,
};

fn run(init: std.process.Init, dir: []const u8, input: Input, p: Pass, j: Journal) !void {
    const a = init.gpa;
    const c = try qwen.Engine.open(a, init.io, dir, .{ .context = input.context, .prefill_rows = 16, .attention_mode = p.mode, .kv8 = p.kv8 });
    defer c.deinit();
    var s = try c.createSession();
    defer s.deinit();
    const vocab = s.logits.len;
    const rows = try a.alloc(u16, 16 * vocab);
    defer a.free(rows);
    try j.write(.{ .kind = "pass", .kv = p.label, .attention = @tagName(p.mode), .kv8 = p.kv8, .partitions = c.partitions.len, .context = c.context, .prefill_rows = c.prefill_rows });
    for (input.sequences, 0..) |seq, si| {
        try s.reset();
        const started = std.Io.Clock.awake.now(init.io).toNanoseconds();
        var at: usize = 0;
        while (at < seq.tokens.len) {
            const end = at + @min(16, seq.tokens.len - at);
            const n = end - at;
            try s.appendLogitRows(seq.tokens[at..end], rows[0 .. n * vocab]);
            for (0..n) |k| {
                const r = at + k;
                const row = rows[k * vocab ..][0..vocab];
                const top = try top4(row);
                if (p.record) |t| t[si][r] = .{ top[0].id, top[1].id };
                var probe_ids: [8]u32 = undefined;
                var probe_logits: [8]f32 = undefined;
                var count: usize = 0;
                for (seq.probes) |q| if (q.row == r) for (q.ids) |id| {
                    if (count == probe_ids.len or id >= vocab) return error.InvalidProbe;
                    probe_ids[count] = id;
                    probe_logits[count] = logit(row[id]);
                    count += 1;
                };
                for (p.probe) |maybe| if (maybe) |t| for (t[si][r]) |id| {
                    if (count == probe_ids.len) return error.InvalidProbe;
                    probe_ids[count] = id;
                    probe_logits[count] = logit(row[id]);
                    count += 1;
                };
                const hash = std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(row));
                try j.write(.{ .kind = "row", .kv = p.label, .seq = seq.name, .row = r, .next = if (r + 1 < seq.tokens.len) seq.tokens[r + 1] else null, .row_hash = hash, .top = top, .probe_ids = probe_ids[0..count], .probe_logits = probe_logits[0..count] });
            }
            at = end;
        }
        const elapsed = std.Io.Clock.awake.now(init.io).toNanoseconds() - started;
        try j.write(.{ .kind = "sequence", .kv = p.label, .seq = seq.name, .tokens = seq.tokens.len, .seconds = @as(f64, @floatFromInt(elapsed)) / 1e9 });
    }
}

fn tops(a: std.mem.Allocator, input: Input) !Tops {
    const t = try a.alloc([]Pair, input.sequences.len);
    for (input.sequences, t) |seq, *rows| rows.* = try a.alloc(Pair, seq.tokens.len);
    return t;
}

pub fn main(init: std.process.Init) !u8 {
    const a = init.arena.allocator();
    const args = try init.minimal.args.toSlice(a);
    if (args.len != 4) return 2;
    const bytes = try std.Io.Dir.cwd().readFileAlloc(init.io, args[2], a, .limited(64 << 20));
    const input = try std.json.parseFromSliceLeaky(Input, a, bytes, .{});
    if (input.sequences.len == 0) return error.InvalidInput;
    for (input.sequences) |seq| if (seq.tokens.len < 2 or seq.tokens.len > input.context) return error.InvalidInput;
    const file = try std.Io.Dir.cwd().createFile(init.io, args[3], .{ .exclusive = true });
    defer file.close(init.io);
    const j: Journal = .{ .a = a, .io = init.io, .file = file };
    const ref = try tops(a, input);
    const bf16 = try tops(a, input);
    try run(init, args[1], input, .{ .label = "ref", .mode = .reference, .kv8 = false, .record = ref }, j);
    try run(init, args[1], input, .{ .label = "bf16", .mode = .chunked, .kv8 = false, .probe = .{ ref, null }, .record = bf16 }, j);
    try run(init, args[1], input, .{ .label = "fp8", .mode = .chunked, .kv8 = true, .probe = .{ ref, bf16 } }, j);
    try j.write(.{ .kind = "done" });
    return 0;
}
