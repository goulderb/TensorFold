//! tf-qwen35-attn-compare: the HF image's serial reference and chunked attention on identical Q/K/V, scored
//! against fp64 exact attention. The chunked kernels are correct when, on every case:
//!   - a row of at most 128 keys (one chunk) is bit-identical to the serial reference;
//!   - past the split, chunked is no less accurate than the serial reference against fp64 (mean error within
//!     0.01 bf16 ULP a case, correctly rounded fraction within 0.0005 over all split cases);
//!   - the FP8 partial over e4m3 rows is as accurate against fp64 of its dequantized rows as BF16 chunked is.
//! Exact agreement past the split is not expected: chunk partials merge in a different order than the serial
//! running sum, and either order is an equally correct rounding (trial qwen35-attn-kernel-compare-20261020).
//!
//! Usage: CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=UUID0,UUID1 \
//!   tf-qwen35-attn-compare --approved-device ORDINAL --expected-uuid GPU-UUID
//! One device, a context and a few MB beside a serving model; refuses below the 1536 MiB floor. Exits 1 on a
//! failed check, printing the report either way. Building runs no device code.
const std = @import("std");
const cuda = @import("cuda");
const device_admission = @import("qwen35_probe_device.zig");
const Buffer = cuda.DeviceBuffer;
const floor = 1536 << 20;

// The kernels' ABI: qwen35.cuh's Geo/Seg (cuda_engine.zig Geo, cuda_cache.zig Seg), HD, MAXCH.
const Geo = extern struct { hid: c_int = 0, nk: c_int = 0, nv: c_int = 0, qh: c_int, kvh: c_int, nl: c_int = 0 };
const Seg = extern struct { rec: u64 = 0, conv: u64 = 0, kv: u64, tap: u64 = 0, log: u64 = 0, cap: c_int, len: c_int, row0: c_int = 0, rows: c_int, tap_rows: c_int = 0, taps: c_int = 0, delta: c_int = 0 };
const hd = 256;
const max_chunks = 64;
const qh = 16;
const kvh = 2;
const group = qh / kvh;

const Regime = enum { flat, peaky };
const key_counts = [_]usize{ 64, 128, 129, 200, 1000, 4096, 8192 };
const row_counts = [_]usize{ 1, 16 };
const seeds = [_]u64{ 1, 2 };

fn bf16Value(bits: u16) f64 {
    return @as(f32, @bitCast(@as(u32, bits) << 16));
}

/// fp64 to bf16, round to nearest even, from the fp64 value itself (no double rounding through fp32).
fn bf16Round(x: f64) u16 {
    const u: u32 = @bitCast(@as(f32, @floatCast(x)));
    const lo = u & 0xffff0000;
    const hi = lo + 0x10000;
    const dlo = @abs(x - @as(f64, @as(f32, @bitCast(lo))));
    const dhi = @abs(x - @as(f64, @as(f32, @bitCast(hi))));
    if (dlo < dhi) return @intCast(lo >> 16);
    if (dhi < dlo) return @intCast(hi >> 16);
    return @intCast(if ((lo >> 16) & 1 == 1) hi >> 16 else lo >> 16);
}

fn ulp(x: f64) f64 {
    if (x == 0) return std.math.ldexp(@as(f64, 1), -133);
    return std.math.ldexp(@as(f64, 1), std.math.frexp(@abs(x)).exponent - 1 - 7);
}

fn ordered(bits: u16) i32 {
    return if (bits & 0x8000 != 0) -@as(i32, bits & 0x7fff) else bits;
}

/// e4m3fn code to its value (no infinities; 0x7f and 0xff are NaN and never produced here).
fn e4m3(code: u8) f32 {
    const sign: f32 = if (code & 0x80 != 0) -1 else 1;
    const e: i32 = (code >> 3) & 0xf;
    const m: f32 = @floatFromInt(code & 7);
    const mag = if (e == 0) m / 8.0 * std.math.ldexp(@as(f32, 1), -6) else (1 + m / 8.0) * std.math.ldexp(@as(f32, 1), e - 7);
    return sign * mag;
}

/// Nearest e4m3 code to x (|x| <= 448), ties to the even code. `table` holds codes 0..126, increasing.
fn e4m3Round(table: *const [127]f32, x: f32) u8 {
    const mag = @abs(x);
    var lo: usize = 0;
    var hi: usize = table.len - 1;
    while (hi - lo > 1) {
        const mid = (lo + hi) / 2;
        if (table[mid] <= mag) lo = mid else hi = mid;
    }
    const dlo = @abs(mag - table[lo]);
    const dhi = @abs(table[hi] - mag);
    const best: u8 = @intCast(if (dlo < dhi) lo else if (dhi < dlo) hi else if (lo & 1 == 0) lo else hi);
    return if (x < 0) best | 0x80 else best;
}

/// One case's tally. fp8 cases have no serial reference (it reads BF16 rows only): their reference fields stay 0.
const Score = struct {
    keys: usize,
    rows: usize,
    regime: Regime,
    seed: u64,
    kind: []const u8,
    elements: usize = 0,
    equal_to_reference: usize = 0,
    max_ulp_from_reference: i32 = 0,
    reference_mean_error: f64 = 0,
    chunked_mean_error: f64 = 0,
    reference_correctly_rounded: usize = 0,
    chunked_correctly_rounded: usize = 0,
};

const Kernels = struct { partial: cuda.Function, merge: cuda.Function, chunked: cuda.Function, chunked8: cuda.Function, merge_chunked: cuda.Function };

const Rig = struct {
    /// A case's host buffers; reset after each case.
    arena: std.heap.ArenaAllocator,
    a: std.mem.Allocator = undefined,
    ctx: *const cuda.Context,
    stream: cuda.Stream,
    k: Kernels,

    fn alloc(r: *Rig, bytes: usize) !Buffer {
        if ((try r.ctx.memInfo()).free < floor + bytes) return error.GpuMemoryFloor;
        return Buffer.alloc(r.ctx.d, bytes);
    }

    fn run(r: *Rig, f: cuda.Function, grid: [3]usize, block: usize, args: anytype) !void {
        var a: cuda.Args = .{};
        inline for (args) |v| a.add(v);
        const cfg: cuda.Config = .{ .grid = .{ .x = @intCast(grid[0]), .y = @intCast(grid[1]), .z = @intCast(grid[2]) }, .block = .{ .x = @intCast(block), .y = 1, .z = 1 } };
        try cuda.launch.launch(f, cfg, r.stream, &a);
    }
};

/// One case: identical bf16 Q and KV (BF16 rows, or e4m3 rows with fp32 row scales), the window's rows the last.
fn case(r: *Rig, table: *const [127]f32, keys: usize, rows: usize, regime: Regime, seed: u64, fp8: bool) !Score {
    defer _ = r.arena.reset(.retain_capacity);
    r.a = r.arena.allocator();
    var prng = std.Random.DefaultPrng.init(seed *% 1000003 +% keys *% 31 +% rows +% @as(u64, @backingInt(regime)) *% 7919);
    const rand = prng.random();
    const cap = keys;
    const len = keys - rows;
    const qscale: f64 = if (regime == .peaky) 4 else 1;
    // Values the kernels read, as fp64 for the exact attention: bf16 rows, or e4m3 times scale rounded to fp32.
    const key_vals = try r.a.alloc(f64, kvh * cap * hd);
    const value_vals = try r.a.alloc(f64, kvh * cap * hd);
    const q = try r.a.alloc(u16, rows * qh * hd);
    for (q) |*x| x.* = bf16Round(rand.floatNorm(f64) * qscale);
    const kv_bytes = if (fp8) 2 * kvh * cap * (hd + 4) else 2 * kvh * cap * hd * 2;
    const kv = try r.a.alloc(u8, kv_bytes);
    if (fp8) {
        const codes = kv[0 .. 2 * kvh * cap * hd];
        const scales = std.mem.bytesAsSlice(f32, kv[2 * kvh * cap * hd ..]);
        for (0..2) |part| for (0..kvh * cap) |row| {
            var raw: [hd]f32 = undefined;
            var amax: f32 = 0;
            for (&raw) |*x| {
                x.* = @floatCast(rand.floatNorm(f64));
                amax = @max(amax, @abs(x.*));
            }
            const scale = amax / 448.0;
            scales[part * kvh * cap + row] = scale;
            const out = if (part == 0) key_vals else value_vals;
            for (raw, 0..) |x, d| {
                const code = e4m3Round(table, x / scale);
                codes[(part * kvh * cap + row) * hd + d] = code;
                out[row * hd + d] = e4m3(code) * scale; // the kernel's __fmul_rn
            }
        };
    } else {
        const bits = std.mem.bytesAsSlice(u16, kv);
        for (0..2) |part| for (0..kvh * cap * hd) |i| {
            const b = bf16Round(rand.floatNorm(f64));
            bits[part * kvh * cap * hd + i] = b;
            (if (part == 0) key_vals else value_vals)[i] = bf16Value(b);
        };
    }

    const slots = rows * qh * max_chunks;
    var kv_dev = try r.alloc(kv.len);
    defer kv_dev.free();
    var q_dev = try r.alloc(q.len * 2);
    defer q_dev.free();
    var ref_dev = try r.alloc(q.len * 2);
    defer ref_dev.free();
    var chk_dev = try r.alloc(q.len * 2);
    defer chk_dev.free();
    var pm = try r.alloc(slots * 4);
    defer pm.free();
    var pl = try r.alloc(slots * 4);
    defer pl.free();
    var po = try r.alloc(slots * hd * 4);
    defer po.free();
    var seg_dev = try r.alloc(@sizeOf(Seg));
    defer seg_dev.free();
    var rowseg_dev = try r.alloc(rows * 4);
    defer rowseg_dev.free();
    var anc_dev = try r.alloc(rows * 4);
    defer anc_dev.free();
    try kv_dev.upload(0, kv);
    try q_dev.upload(0, std.mem.sliceAsBytes(q));
    const seg: Seg = .{ .kv = kv_dev.ptr, .cap = @intCast(cap), .len = @intCast(len), .rows = @intCast(rows) };
    try seg_dev.upload(0, std.mem.asBytes(&seg));
    const rowseg = try r.a.alloc(c_int, rows);
    @memset(rowseg, 0);
    const anc = try r.a.alloc(u32, rows);
    for (anc, 0..) |*x, m| x.* = 0x80000000 | @as(u32, @intCast(m)); // a contiguous window: row m sees m + 1
    try rowseg_dev.upload(0, std.mem.sliceAsBytes(rowseg));
    try anc_dev.upload(0, std.mem.sliceAsBytes(anc));
    const geo: Geo = .{ .qh = qh, .kvh = kvh };
    const no_params: u64 = 0;
    const layer: c_int = 0;
    const scale: f32 = 1.0 / 16.0;

    if (!fp8) {
        try r.run(r.k.partial, .{ 1, qh, rows }, 256, .{ geo, q_dev.ptr, seg_dev.ptr, rowseg_dev.ptr, anc_dev.ptr, layer, no_params, scale, pm.ptr, pl.ptr, po.ptr });
        try r.run(r.k.merge, .{ qh, rows, 1 }, 256, .{ geo, pm.ptr, pl.ptr, po.ptr, seg_dev.ptr, rowseg_dev.ptr, anc_dev.ptr, no_params, ref_dev.ptr });
    }
    try r.run(if (fp8) r.k.chunked8 else r.k.chunked, .{ max_chunks, kvh, rows }, 32 * group, .{ geo, q_dev.ptr, seg_dev.ptr, rowseg_dev.ptr, anc_dev.ptr, layer, no_params, scale, pm.ptr, pl.ptr, po.ptr });
    try r.run(r.k.merge_chunked, .{ qh, rows, 1 }, 32, .{ geo, pm.ptr, pl.ptr, po.ptr, seg_dev.ptr, rowseg_dev.ptr, anc_dev.ptr, no_params, chk_dev.ptr });
    try r.stream.synchronize();
    const ref = try r.a.alloc(u16, q.len);
    const chk = try r.a.alloc(u16, q.len);
    if (!fp8) try ref_dev.download(0, std.mem.sliceAsBytes(ref));
    try chk_dev.download(0, std.mem.sliceAsBytes(chk));

    var s: Score = .{ .keys = keys, .rows = rows, .regime = regime, .seed = seed, .kind = if (fp8) "fp8" else "bf16" };
    const score = try r.a.alloc(f64, cap);
    var out: [hd]f64 = undefined;
    var ref_sum: f64 = 0;
    var chk_sum: f64 = 0;
    for (0..rows) |m| {
        const total = len + m + 1;
        for (0..qh) |h| {
            const g = h / group;
            const qq = q[(m * qh + h) * hd ..][0..hd];
            var mx = -std.math.inf(f64);
            for (0..total) |t| {
                var dot: f64 = 0;
                for (0..hd) |d| dot += bf16Value(qq[d]) * key_vals[(g * cap + t) * hd + d];
                score[t] = dot / 16.0;
                mx = @max(mx, score[t]);
            }
            var den: f64 = 0;
            @memset(&out, 0);
            for (0..total) |t| {
                const w = @exp(score[t] - mx);
                den += w;
                for (0..hd) |d| out[d] += w * value_vals[(g * cap + t) * hd + d];
            }
            for (0..hd) |d| {
                const exact = out[d] / den;
                const i = (m * qh + h) * hd + d;
                const rn = bf16Round(exact);
                chk_sum += @abs(bf16Value(chk[i]) - exact) / ulp(exact);
                s.chunked_correctly_rounded += @intFromBool(chk[i] == rn);
                if (!fp8) {
                    ref_sum += @abs(bf16Value(ref[i]) - exact) / ulp(exact);
                    s.reference_correctly_rounded += @intFromBool(ref[i] == rn);
                    s.equal_to_reference += @intFromBool(ref[i] == chk[i]);
                    s.max_ulp_from_reference = @max(s.max_ulp_from_reference, @as(i32, @intCast(@abs(ordered(ref[i]) - ordered(chk[i])))));
                }
                s.elements += 1;
            }
        }
    }
    const n: f64 = @floatFromInt(s.elements);
    s.reference_mean_error = ref_sum / n;
    s.chunked_mean_error = chk_sum / n;
    return s;
}

pub fn main(init: std.process.Init) !u8 {
    const a = init.arena.allocator();
    const args = try init.minimal.args.toSlice(a);
    if (args.len != 5 or !std.mem.eql(u8, args[1], "--approved-device") or !std.mem.eql(u8, args[3], "--expected-uuid")) {
        try std.Io.File.stderr().writeStreamingAll(init.io, "usage: tf-qwen35-attn-compare --approved-device ORDINAL --expected-uuid GPU-UUID\nRequires PCI_BUS_ID ordering and two full UUIDs in CUDA_VISIBLE_DEVICES (qwen35_probe_device.zig). Building this executable runs no device code.\n");
        return 2;
    }
    const ordinal = try std.fmt.parseInt(c_int, args[2], 10);
    const admission = try device_admission.Admission.parse(init.environ_map.get("CUDA_DEVICE_ORDER"), init.environ_map.get("CUDA_VISIBLE_DEVICES"), &.{ordinal}, &.{args[4]});
    var driver = try cuda.Driver.open();
    defer driver.close();
    const identities = try admission.verify(a, &driver);
    var context = try cuda.Context.init(&driver, ordinal);
    defer context.deinit();
    if ((try context.memInfo()).free < floor + (64 << 20)) return error.GpuMemoryFloor;
    var stream = try cuda.Stream.init(&driver, false);
    defer stream.deinit();
    var image = try cuda.Module.load(&driver, cuda.kernels.qwen35_hf);
    defer image.unload();
    var rig: Rig = .{ .arena = .init(std.heap.page_allocator), .ctx = &context, .stream = stream, .k = .{
        .partial = try image.function("tf_q35_hf_attn_partial"),
        .merge = try image.function("tf_q35_hf_attn_merge"),
        .chunked = try image.function("tf_q35_hf_attn_partial_chunked"),
        .chunked8 = try image.function("tf_q35_hf_attn_partial8_chunked"),
        .merge_chunked = try image.function("tf_q35_hf_attn_merge_chunked"),
    } };
    defer rig.arena.deinit();
    var table: [127]f32 = undefined;
    for (&table, 0..) |*v, c| v.* = e4m3(@intCast(c));

    var scores: std.ArrayList(Score) = .empty;
    var failures: std.ArrayList([]const u8) = .empty;
    var split_elements: usize = 0;
    var split_ref_cr: usize = 0;
    var split_chk_cr: usize = 0;
    var bf16_split_error: f64 = 0;
    var fp8_split_error: f64 = 0;
    var fp8_split_elements: usize = 0;
    for (seeds) |seed| for ([_]Regime{ .flat, .peaky }) |regime| for (key_counts) |keys| for (row_counts) |rows| for ([_]bool{ false, true }) |fp8| {
        const s = try case(&rig, &table, keys, rows, regime, seed, fp8);
        try scores.append(a, s);
        const label = try std.fmt.allocPrint(a, "{s} {s} keys={d} rows={d} seed={d}", .{ s.kind, @tagName(regime), keys, rows, seed });
        if (fp8) {
            if (keys > 128) {
                fp8_split_error += s.chunked_mean_error * @as(f64, @floatFromInt(s.elements));
                fp8_split_elements += s.elements;
            }
            continue;
        }
        if (keys <= 128 and s.equal_to_reference != s.elements) try failures.append(a, try std.fmt.allocPrint(a, "{s}: one chunk not bit-identical", .{label}));
        if (keys > 128) {
            if (s.chunked_mean_error > s.reference_mean_error + 0.01) try failures.append(a, try std.fmt.allocPrint(a, "{s}: chunked mean error {d:.4} > reference {d:.4} + 0.01", .{ label, s.chunked_mean_error, s.reference_mean_error }));
            split_elements += s.elements;
            split_ref_cr += s.reference_correctly_rounded;
            split_chk_cr += s.chunked_correctly_rounded;
            bf16_split_error += s.chunked_mean_error * @as(f64, @floatFromInt(s.elements));
        }
    };
    const ref_cr = @as(f64, @floatFromInt(split_ref_cr)) / @as(f64, @floatFromInt(split_elements));
    const chk_cr = @as(f64, @floatFromInt(split_chk_cr)) / @as(f64, @floatFromInt(split_elements));
    if (chk_cr < ref_cr - 0.0005) try failures.append(a, try std.fmt.allocPrint(a, "split: chunked correctly rounded {d:.5} < reference {d:.5} - 0.0005", .{ chk_cr, ref_cr }));
    const bf16_error = bf16_split_error / @as(f64, @floatFromInt(split_elements));
    const fp8_error = fp8_split_error / @as(f64, @floatFromInt(fp8_split_elements));
    if (fp8_error > bf16_error + 0.01) try failures.append(a, try std.fmt.allocPrint(a, "split: fp8 chunked mean error {d:.4} > bf16 chunked {d:.4} + 0.01", .{ fp8_error, bf16_error }));

    const report = .{
        .status = if (failures.items.len == 0) "pass" else "fail",
        .device = ordinal,
        .device_identities = identities,
        .reference = "fp64 exact softmax attention over the values each kernel reads",
        .split = .{ .elements = split_elements, .reference_correctly_rounded = ref_cr, .chunked_correctly_rounded = chk_cr, .bf16_chunked_mean_error = bf16_error, .fp8_chunked_mean_error = fp8_error },
        .failures = failures.items,
        .cases = scores.items,
        .free_mib = (try context.memInfo()).free / (1 << 20),
    };
    var output: std.Io.Writer.Allocating = .init(a);
    try std.json.Stringify.value(report, .{ .whitespace = .indent_2 }, &output.writer);
    try output.writer.writeByte('\n');
    try std.Io.File.stdout().writeStreamingAll(init.io, output.written());
    return if (failures.items.len == 0) 0 else 1;
}

test "bf16 and e4m3 host rounding" {
    try std.testing.expectEqual(@as(u16, 0x3f80), bf16Round(1.0));
    try std.testing.expectEqual(@as(u16, 0x3f80), bf16Round(1.0 + 1.0 / 256.0)); // a tie rounds to the even 1.0
    try std.testing.expectEqual(@as(u16, 0x3f82), bf16Round(1.0 + 3.0 / 256.0)); // a tie rounds to the even 0x3f82
    try std.testing.expectEqual(@as(f32, 448), e4m3(0x7e));
    try std.testing.expectEqual(@as(f32, -1), e4m3(0xb8));
    try std.testing.expectEqual(std.math.ldexp(@as(f32, 1), -9), e4m3(0x01));
    var table: [127]f32 = undefined;
    for (&table, 0..) |*v, c| v.* = e4m3(@intCast(c));
    for (0..127) |c| try std.testing.expectEqual(@as(u8, @intCast(c)), e4m3Round(&table, table[c]));
    try std.testing.expectEqual(@as(u8, 0xb8), e4m3Round(&table, -1.01));
    try std.testing.expectEqual(1.0 / 128.0, ulp(1.5));
}
