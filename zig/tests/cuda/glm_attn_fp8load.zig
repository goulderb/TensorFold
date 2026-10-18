//! HB32 FP8 attention: old/fast partial and output bytes, then CUDA-event timing of sparse chunks alone.
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm_cuda");
const check = @import("check.zig");
const Rig = @import("quant_ref_tests.zig").RigT;
const glue = glm.glue;
const latent = glm.latent;
const rank_heads = 32;
const lw = 512;
const width = 2051;
const nch = 5;
const cache_rows = 8192;
const scale: f32 = 0.125;
const parity_rows = 64;
const edge_counts = [_]i32{ 2048, 2049, 2050, 2051, 0, 1, 31, 32, 33, 511, 512, 513, 1023, 1024, 1025, 2047 };

const Inputs = struct {
    qa: u64,
    cache: glm.cache.Cache,
    pos: u64,
    tokens: u64,
    counts: u64,
    host_counts: []const i32,
    rows: usize,
    heads: usize,

    fn init(r: *Rig, rows: usize, heads: usize, position: i32, edges: bool) !Inputs {
        const q = try r.bfs(rows * heads * lw, 0.5);
        const qa = try r.dev(std.mem.sliceAsBytes(q));
        const row_bytes = glm.cache.Format.fp8.rowBytes(lw);
        const raw = try r.a.alloc(u8, cache_rows * row_bytes);
        @memset(raw, 0);
        for (0..cache_rows) |row| {
            const bytes = raw[row * row_bytes ..][0..row_bytes];
            // Cover finite E4M3 codes and signed zero; independently randomize the other half to catch column swaps.
            for (bytes[0..lw], 0..) |*byte, col| {
                const code: u8 = if (col < 256) @intCast((row * 37 + col) % 256) else r.rng.int(u8);
                byte.* = if (code & 127 == 127) code - 1 else code;
            }
            std.mem.writeInt(u32, bytes[lw..][0..4], @as(u32, @intCast(120 + row % 8)) << 23, .little);
        }
        const cache: glm.cache.Cache = .{ .ptr = try r.dev(raw), .format = .fp8 };
        const counts = try r.a.alloc(i32, rows);
        const tokens = try r.a.alloc(i32, rows * width);
        @memset(tokens, -1);
        for (counts, 0..) |*count, row| {
            count.* = if (edges) edge_counts[row % edge_counts.len] else @intCast(2048 + row % 4);
            // Sorted, unique, noncontiguous keys with a different offset on each row.
            const offset = r.rng.uintLessThan(usize, cache_rows - 3 * width);
            for (tokens[row * width ..][0..@intCast(count.*)], 0..) |*token, i| token.* = @intCast(offset + 3 * i);
        }
        return .{
            .qa = qa,
            .cache = cache,
            .pos = try r.dev(std.mem.asBytes(&position)),
            .tokens = try r.dev(std.mem.sliceAsBytes(tokens)),
            .counts = try r.dev(std.mem.sliceAsBytes(counts)),
            .host_counts = counts,
            .rows = rows,
            .heads = heads,
        };
    }

    fn attend(in: Inputs, r: *Rig, set: glue.Set, p: latent.Partials, out: u64, sparse: bool) !void {
        if (sparse) {
            try latent.sparseAttentionBlock(set, r.s, in.qa, in.cache, in.tokens, in.counts, width, p, out, in.rows, in.heads, lw, scale, 32);
        } else {
            try latent.attention(set, r.s, in.qa, in.cache, in.pos, p, out, in.counts, in.rows, in.heads, lw, nch, scale, 32);
        }
    }

    fn chunks(in: Inputs, r: *Rig, f: cuda.Function, p: latent.Partials, sparse: bool, fp8load: bool) !void {
        var args: cuda.Args = .{};
        inline for (.{ in.qa, in.cache.ptr, in.pos, in.tokens, in.counts, p.po, p.pm, p.pl, @as(i64, @intCast(in.rows)), @as(i64, @intCast(in.heads)), @as(i64, width), scale, @as(c_int, @intFromBool(sparse)) }) |v| args.add(v);
        try cuda.launch.launch(f, .{
            .grid = .{ .x = @intCast(in.rows), .y = @intCast((in.heads + 31) / 32), .z = nch },
            .block = .{ .x = 256 },
            .shared = if (fp8load) glue.chunksFP8LoadSmem() else glue.chunksSmem(32),
        }, r.s, &args);
    }
};

fn partials(r: *Rig, rows: usize, heads: usize) !latent.Partials {
    const scalars = nch * rows * heads * 4;
    return .{ .po = try r.zeros(scalars * lw), .pm = try r.zeros(scalars), .pl = try r.zeros(scalars) };
}

fn fill(r: *Rig, ptr: u64, len: usize, value: u8) !void {
    const buf: cuda.DeviceBuffer = .{ .d = r.gpu.d, .ptr = ptr, .len = len };
    try buf.fill8(value, r.s.handle);
}

fn release(r: *Rig) void {
    for (r.bufs.items) |*b| b.free();
    r.bufs.deinit(r.gpu.gpa);
}

fn finite(raw: []const u8, bf16: bool, allow_negative_inf: bool) !void {
    const step: usize = if (bf16) 2 else 4;
    var i: usize = 0;
    while (i < raw.len) : (i += step) {
        const value: f32 = if (bf16)
            @bitCast(@as(u32, std.mem.bytesToValue(u16, raw[i..][0..2])) << 16)
        else
            std.mem.bytesToValue(f32, raw[i..][0..4]);
        try check.expect(std.math.isFinite(value) or (allow_negative_inf and value == -std.math.inf(f32)), "attention value at byte {d} is valid", .{i});
    }
}

fn parity(gpu: check.Gpu, set: glue.Set, sparse: bool, position: i32, heads: usize) !void {
    var arena: std.heap.ArenaAllocator = .init(gpu.gpa);
    defer arena.deinit();
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var prng = std.Random.DefaultPrng.init(0x593fa57);
    var r: Rig = .{ .gpu = gpu, .a = arena.allocator(), .s = s, .rng = prng.random() };
    defer release(&r);
    const in = try Inputs.init(&r, parity_rows, heads, position, true);
    const p = try partials(&r, parity_rows, heads);
    const out_bytes = parity_rows * heads * lw * 2;
    const out = try r.zeros(out_bytes);
    try fill(&r, out, out_bytes, 0xff);
    const ptrs = [_]u64{ p.po, p.pm, p.pl, out };
    const scalar_bytes = nch * parity_rows * heads * 4;
    const sizes = [_]usize{ scalar_bytes * lw, scalar_bytes, scalar_bytes, out_bytes };
    const labels = [_][]const u8{ "po", "pm", "pl", "merged BF16" };
    var old: [4][]u8 = undefined;
    try in.attend(&r, set, p, out, sparse);
    for (ptrs, sizes, &old, 0..) |ptr, bytes, *snapshot, i| {
        snapshot.* = try r.back(ptr, bytes);
        if (i < 3) try finite(snapshot.*, false, i == 1);
        // Different poison catches unwritten partials; output keeps its zero-count sentinel.
        try fill(&r, ptr, bytes, if (i == 3) 0xff else 0xfe);
    }
    var fast = set;
    fast.attn_fp8load = true;
    try in.attend(&r, fast, p, out, sparse);
    for (ptrs, sizes, old, labels) |ptr, bytes, want, label| {
        const got = try r.back(ptr, bytes);
        try check.sameBytes(label, got, want);
    }
    // Exercise the candidate symbol directly too, so a broken opt-in dispatch cannot hide its arithmetic.
    for (ptrs[0..3], sizes[0..3]) |ptr, bytes| try fill(&r, ptr, bytes, 0xfd);
    try in.chunks(&r, set.glue.chunks_32_fp8load, p, sparse, true);
    for (ptrs[0..3], sizes[0..3], old[0..3], labels[0..3]) |ptr, bytes, want, label| {
        try check.sameBytes(label, try r.back(ptr, bytes), want);
    }
    const row_bytes = heads * lw * 2;
    for (0..parity_rows) |row| {
        const bytes = old[3][row * row_bytes ..][0..row_bytes];
        if (sparse and in.host_counts[row] == 0) {
            for (bytes) |b| try check.expect(b == 0xff, "zero-count row {d} remains untouched", .{row});
        } else try finite(bytes, true, false);
    }
    check.pass("glm-attn-fp8load {s} pos={d} rows={d} heads={d} nch={d}: po/pm/pl and merged output byte-equal", .{ if (sparse) "sparse edge counts" else "dense", position, parity_rows, heads, nch });
}

fn timing(gpu: check.Gpu, set: glue.Set, rows: usize) !void {
    var arena: std.heap.ArenaAllocator = .init(gpu.gpa);
    defer arena.deinit();
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var prng = std.Random.DefaultPrng.init(0x593fa57);
    var r: Rig = .{ .gpu = gpu, .a = arena.allocator(), .s = s, .rng = prng.random() };
    defer release(&r);
    const in = try Inputs.init(&r, rows, rank_heads, 0, false);
    // One partial set is reused by both variants, including at 2048 rows (about 640 MiB).
    const p = try partials(&r, rows, rank_heads);
    const kernels = [_]cuda.Function{ set.glue.chunks_32_fp8, set.glue.chunks_32_fp8load };
    var start = try cuda.Event.init(gpu.d, true);
    defer start.deinit();
    var end = try cuda.Event.init(gpu.d, true);
    defer end.deinit();
    for (0..2) |_| for (kernels, 0..) |kernel, variant| try in.chunks(&r, kernel, p, true, variant == 1);
    try s.synchronize();
    const reps = 7;
    var samples: [2][reps]f64 = undefined;
    for (0..reps) |rep| {
        for (0..2) |turn| {
            const variant = (rep + turn) % 2;
            try start.record(s);
            try in.chunks(&r, kernels[variant], p, true, variant == 1);
            try end.record(s);
            try end.synchronize();
            samples[variant][rep] = try cuda.Event.elapsedMs(start, end);
            std.debug.print("RESULT glm-attn-fp8load sparse chunks-only variant={s} rows={d} heads={d} nch={d} counts=2048..2051 rep={d} ms={d:.6}\n", .{ if (variant == 0) "old" else "fast", rows, rank_heads, nch, rep, samples[variant][rep] });
        }
    }
    for (&samples, [_][]const u8{ "old", "fast" }) |*values, name| {
        std.debug.print("RESULT glm-attn-fp8load sparse chunks-only variant={s} rows={d} median_ms={d:.6} launches={d}\n", .{ name, rows, check.median(values), reps });
    }
}

pub fn run(gpu: check.Gpu, args: []const [:0]const u8) !void {
    if (args.len > 1) return error.UnexpectedArgument;
    const rows = if (args.len == 1) try std.fmt.parseInt(usize, args[0], 10) else 2048;
    if (rows == 0 or rows > 2048) return error.RowsOutOfRange;
    var mod = try cuda.Module.load(gpu.d, cuda.kernels.glm_glue);
    defer mod.unload();
    const kernels = try glue.Kernels.load(mod);
    const set: glue.Set = .{ .glue = &kernels };
    for ([_]i32{ 0, 480, 2016 }) |position| try parity(gpu, set, false, position, rank_heads);
    try parity(gpu, set, true, 0, rank_heads);
    try parity(gpu, set, true, 0, 45);
    try timing(gpu, set, rows);
    check.pass("glm-attn-fp8load: parity at 64 rows; chunks-only timing at {d} rows (no merge in events)", .{rows});
}
