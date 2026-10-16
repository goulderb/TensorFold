const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm_cuda");
const core = @import("core");
const check = @import("check.zig");
const qr = @import("quant_ref_tests.zig");
const Rig = qr.RigT;
const bf = qr.helpers.bfOf;
const bf16 = qr.helpers.bf16Of;

fn products(r: *Rig, set: glm.glue.Set, weight: cuda.qlinear.Weight, values: []const f64) !void {
    const n = weight.outputs();
    const k = weight.inputs();
    const stride = k + 8;
    const rows = [_]usize{ 1, 3, 17, 40, 128 };
    const maximum = rows[rows.len - 1];
    const x = try r.bfs(maximum * stride, 0.125);
    for (0..maximum) |row| @memset(x[row * stride + k ..][0..8], bf16(100));
    const want = try r.a.alloc(f64, maximum * n);
    const tol = try r.a.alloc(f64, maximum * n);
    for (0..maximum) |row| for (0..n) |j| {
        var sum: f64 = 0;
        var total: f64 = 0;
        for (0..k) |i| {
            const v = bf(x[row * stride + i]) * values[j * k + i];
            sum += v;
            total += @abs(v);
        }
        want[row * n + j] = sum;
        tol[row * n + j] = @abs(sum) / 128 + total / 4096 + 1e-20;
    };
    const input = try r.dev(std.mem.sliceAsBytes(x));
    const scratch = try r.zeros(@max(maximum * k / 64 * 4, 8 * maximum * n * 4));
    for (rows) |m| for ([_]bool{ false, true }) |wide| {
        const out = try r.zeros(m * n * @as(usize, if (wide) 4 else 2));
        try glm.b16.proj(set, r.s, input, stride, .{ .q = weight }, out, scratch, m, n, k, wide);
        var name: [100]u8 = undefined;
        try r.close(try std.fmt.bufPrint(&name, "GLM {s} {d} strided rows wide={any}", .{ @tagName(weight), m, wide }), out, wide, want[0 .. m * n], tol[0 .. m * n]);
    };
}

fn dense(r: *Rig, modules: *const glm.forward.Modules, set: glm.glue.Set, n: usize, k: usize) !void {
    const weights = try r.a.alloc(u16, n * k);
    const values = try r.a.alloc(f64, n * k);
    const raw = try r.a.alloc(u32, n * k / 8);
    @memset(raw, 0);
    for (0..n) |row| for (0..k) |col| {
        const bias: f32 = @as(f32, @floatFromInt(row % 7)) / 4 - 1;
        const v = @as(f32, @floatFromInt(col % 16)) / 4 + bias;
        weights[row * k + col] = bf16(v);
        values[row * k + col] = v;
        raw[row * k / 8 + col / 8] |= @as(u32, @intCast(col % 16)) << @as(u5, @intCast(col % 8 * 4));
    };
    const source = try r.dev(std.mem.sliceAsBytes(weights));
    const words = try r.zeros(n * k / 2);
    const scales = try r.zeros(n * k / 32);
    const biases = try r.zeros(n * k / 32);
    try glm.quant.q4Mse(&modules.quant, r.s, source, n, k, words, scales, biases);
    try check.sameBytes("MSE Q4 literal nibbles", try r.back(words, n * k / 2), std.mem.sliceAsBytes(raw));
    const expected_s = try r.a.alloc(u16, n * k / 64);
    const expected_b = try r.a.alloc(u16, n * k / 64);
    @memset(expected_s, bf16(0.25));
    for (expected_b, 0..) |*v, i| v.* = bf16(@as(f32, @floatFromInt((i / (k / 64)) % 7)) / 4 - 1);
    try check.sameBytes("MSE Q4 literal scales", try r.back(scales, expected_s.len * 2), std.mem.sliceAsBytes(expected_s));
    try check.sameBytes("MSE Q4 literal biases", try r.back(biases, expected_b.len * 2), std.mem.sliceAsBytes(expected_b));
    const layout = cuda.qlinear.Affine4.layout(n, k);
    var model: glm.weights.Weights = .{ .gpa = r.gpu.gpa, .config = undefined, .share = undefined };
    defer model.deinit();
    var tools = modules.tools();
    tools.precision.dense = .q4;
    var loader: glm.weights.Loader = .{ .gpa = r.gpu.gpa, .d = r.gpu.d, .stream = r.s, .pack_dense = modules.pack_dense, .tools = tools, .w = &model, .ck = undefined };
    defer loader.host.deinit(r.gpu.gpa);
    try loader.host.appendSlice(r.gpu.gpa, std.mem.sliceAsBytes(weights));
    const projection = try glm.weights.projections.finish(&loader, "synthetic", .{}, n, k);
    const view = projection.q.affine4;
    var physical: usize = 0;
    for (model.buffers.items) |buffer| physical += buffer.len;
    try check.expect(model.buffers.items.len == 3 and physical == layout.words + 2 * layout.scales, "loader owns only Q4 physical bytes", .{});
    const expected_words = try r.a.alloc(u32, layout.words / 4);
    cuda.nvfp4.packWords(expected_words, std.mem.sliceAsBytes(raw), n, k);
    try check.sameBytes("Q4 packed nibbles and padded rows", try r.back(view.w, layout.words), std.mem.sliceAsBytes(expected_words));
    const tiled_s = try r.a.alloc(u16, layout.scales / 2);
    const tiled_b = try r.a.alloc(u16, layout.scales / 2);
    for (0..k / 64) |g| for (0..layout.npad) |row| {
        tiled_s[g * layout.npad + row] = if (row < n) expected_s[row * (k / 64) + g] else 0;
        tiled_b[g * layout.npad + row] = if (row < n) expected_b[row * (k / 64) + g] else 0;
    };
    try check.sameBytes("Q4 packed scale stride", try r.back(view.s, layout.scales), std.mem.sliceAsBytes(tiled_s));
    try check.sameBytes("Q4 packed bias stride", try r.back(view.b, layout.scales), std.mem.sliceAsBytes(tiled_b));
    try products(r, set, .{ .affine4 = view }, values);
    check.pass("Q4 MSE conversion, upstream packing and projection [{d},{d}]", .{ n, k });
}

fn head(r: *Rig, modules: *const glm.forward.Modules, set: glm.glue.Set) !void {
    const n = 130;
    const k = 256;
    const weights = try r.a.alloc(u16, n * k);
    const codes = try r.a.alloc(u8, n * k);
    const scales = try r.a.alloc(f32, n * k / 128);
    const values = try r.a.alloc(f64, n * k);
    for (scales, 0..) |*s, g| s.* = @as(f32, @floatFromInt(1 + g % 4)) / 8;
    for (0..n) |row| for (0..k) |col| {
        const code: u8 = if (col % 128 == 0) 0x7e else @as(u8, @intCast(0x20 + col % 80)) | if (col % 3 == 0) @as(u8, 0x80) else 0;
        codes[row * k + col] = code;
        values[row * k + col] = core.quant.fp8.e4m3(code) * scales[row * (k / 128) + col / 128];
        weights[row * k + col] = bf16(@floatCast(values[row * k + col]));
    };
    // Torch's scalar division is float32 reciprocal multiplication: 168 * fp32(1/448) is 0x3ec00001.
    const scale_bits = [_]u32{ 0x3e000000, 0x3e800000, 0x3ec00001, 0x3f000000 };
    for (scales, 0..) |*s, g| s.* = @bitCast(scale_bits[g % scale_bits.len]);
    for (values, codes, 0..) |*value, code, i| value.* = core.quant.fp8.e4m3(code) * scales[i / 128];
    const npad = cuda.fp8.padded(n);
    var model: glm.weights.Weights = .{ .gpa = r.gpu.gpa, .config = undefined, .share = undefined };
    defer model.deinit();
    model.config.hidden = k;
    model.share.vocab = .{ .lo = 0, .hi = n };
    var tools = modules.tools();
    tools.precision.head = .fp8;
    var loader: glm.weights.Loader = .{ .gpa = r.gpu.gpa, .d = r.gpu.d, .stream = r.s, .pack_dense = modules.pack_dense, .tools = tools, .w = &model, .ck = undefined };
    defer loader.host.deinit(r.gpu.gpa);
    try loader.host.appendSlice(r.gpu.gpa, std.mem.sliceAsBytes(weights));
    model.head = .{ .b16 = try loader.put("head", .{}) };
    try glm.weights.projections.head(&loader);
    const dcodes = model.head.q.fp8g.codes;
    const dscales = model.head.q.fp8g.scales;
    var physical: usize = 0;
    for (model.buffers.items) |buffer| physical += buffer.len;
    try check.expect(model.buffers.items.len == 2 and physical == npad * k + npad * k / 64 * 4, "FP8 head replaces BF16 allocation with packed physical bytes", .{});
    const packed_codes = try r.a.alloc(u8, npad * k);
    const packed_scales = try r.a.alloc(f32, npad * k / 64);
    cuda.fp8.packCodes(packed_codes, codes, n, k);
    cuda.fp8.packRowScales(packed_scales, scales, n, k);
    try check.sameBytes("FP8 row128 codes and padded rows", try r.back(dcodes, packed_codes.len), packed_codes);
    try check.sameBytes("FP8 row128 scale stride", try r.back(dscales, packed_scales.len * 4), std.mem.sliceAsBytes(packed_scales));
    try products(r, set, .{ .fp8g = cuda.fp8.weight(dcodes, dscales, n, k) }, values);
    check.pass("FP8 row128 conversion, upstream packing and projection", .{});
}

pub fn run(gpu: check.Gpu) !void {
    var arena: std.heap.ArenaAllocator = .init(gpu.gpa);
    defer arena.deinit();
    var stream = try cuda.Stream.init(gpu.d, false);
    defer stream.deinit();
    var rng = std.Random.DefaultPrng.init(53);
    var r: Rig = .{ .gpu = gpu, .a = arena.allocator(), .s = stream, .rng = rng.random() };
    defer {
        for (r.bufs.items) |*buffer| buffer.free();
        r.bufs.deinit(gpu.gpa);
    }
    var modules = try glm.forward.Modules.load(gpu.d, gpu.ctx);
    defer modules.deinit();
    const linear: cuda.qlinear.Linear = .{ .affine4 = &modules.affine4, .lane = &modules.lane };
    const set: glm.glue.Set = .{ .glue = &modules.glue, .linear = linear };
    try dense(&r, &modules, set, 130, 128);
    try dense(&r, &modules, set, 128, 1024);
    try head(&r, &modules, set);
    check.pass("glm-precision: packed bytes and float64 products pass, worst {d:.3} of declared tolerance", .{r.worst});
}

pub fn oracle(gpu: check.Gpu, dir: []const u8) !void {
    var fixture = try @import("fixture.zig").Fixture.open(gpu.gpa, gpu.io, dir);
    defer fixture.deinit();
    var arena: std.heap.ArenaAllocator = .init(gpu.gpa);
    defer arena.deinit();
    var stream = try cuda.Stream.init(gpu.d, false);
    defer stream.deinit();
    var rng = std.Random.DefaultPrng.init(53);
    var r: Rig = .{ .gpu = gpu, .a = arena.allocator(), .s = stream, .rng = rng.random() };
    defer {
        for (r.bufs.items) |*buffer| buffer.free();
        r.bufs.deinit(gpu.gpa);
    }
    const n: usize = @intCast(try fixture.int("n"));
    const k: usize = @intCast(try fixture.int("k"));
    const source = try fixture.bytes("weights");
    defer gpu.gpa.free(source);
    try check.expect(source.len == n * k * 2 and k % 128 == 0, "oracle BF16 dimensions", .{});
    const input = try r.dev(source);
    var module = try cuda.Module.load(gpu.d, cuda.kernels.glm_quant);
    defer module.unload();
    const quant = try glm.quant.Kernels.load(module);
    const words = try r.zeros(n * k / 2);
    const scales = try r.zeros(n * k / 32);
    const biases = try r.zeros(n * k / 32);
    try glm.quant.q4Mse(&quant, stream, input, n, k, words, scales, biases);
    for ([_]u64{ words, scales, biases }, [_][]const u8{ "q4_words", "q4_scales", "q4_biases" }) |ptr, name| {
        const want = try fixture.bytes(name);
        defer gpu.gpa.free(want);
        try check.sameBytes(name, try r.back(ptr, want.len), want);
    }
    const raw_codes = try fixture.bytes("fp8_codes");
    defer gpu.gpa.free(raw_codes);
    const raw_scales = try fixture.bytes("fp8_scales");
    defer gpu.gpa.free(raw_scales);
    const aligned_scales = try r.a.alloc(f32, n * k / 128);
    @memcpy(std.mem.sliceAsBytes(aligned_scales), raw_scales);
    const npad = cuda.fp8.padded(n);
    const expected_codes = try r.a.alloc(u8, npad * k);
    const expected_scales = try r.a.alloc(f32, npad * k / 64);
    cuda.fp8.packCodes(expected_codes, raw_codes, n, k);
    cuda.fp8.packRowScales(expected_scales, aligned_scales, n, k);
    const dcodes = try r.zeros(expected_codes.len);
    const dscales = try r.zeros(expected_scales.len * 4);
    try glm.quant.fp8Rows(&quant, stream, input, n, k, dcodes, dscales);
    try check.sameBytes("production FP8 codes", try r.back(dcodes, expected_codes.len), expected_codes);
    try check.sameBytes("production FP8 scales", try r.back(dscales, expected_scales.len * 4), std.mem.sliceAsBytes(expected_scales));
    check.pass("GLM weight conversions [{d},{d}] equal production qmm.py GPU oracle bytes", .{ n, k });
}

pub fn grammar(gpu: check.Gpu) !void {
    var arena: std.heap.ArenaAllocator = .init(gpu.gpa);
    defer arena.deinit();
    var stream = try cuda.Stream.init(gpu.d, false);
    defer stream.deinit();
    var rng = std.Random.DefaultPrng.init(53);
    var r: Rig = .{ .gpu = gpu, .a = arena.allocator(), .s = stream, .rng = rng.random() };
    defer {
        for (r.bufs.items) |*buffer| buffer.free();
        r.bufs.deinit(gpu.gpa);
    }
    var modules = try glm.forward.Modules.load(gpu.d, gpu.ctx);
    defer modules.deinit();
    const vocab = 65;
    const rows = 5;
    var sampler = try glm.sampler.Sampler.init(gpu.d, vocab, rows);
    defer sampler.deinit();
    var logits: [rows * vocab]u16 = undefined;
    for (&logits, 0..) |*value, i| value.* = bf16(@floatFromInt(vocab - i % vocab));
    const input = try r.dev(std.mem.sliceAsBytes(&logits));
    const wanted = [_]u32{ 64, 32, 7, 0, 63 };
    var masks: [rows * 3]u32 = @splat(0);
    for (wanted, 0..) |token, row| masks[row * 3 + token / 32] |= @as(u32, 1) << @as(u5, @intCast(token % 32));
    for (0..rows) |row| masks[row * 3 + 2] |= 0xfffffffe;
    for ([_]usize{ 1, 3, rows }) |count| for ([_]bool{ false, true }) |sampled| {
        try glm.sampler.validateMasks(masks[0 .. count * 3], count, vocab);
        try sampler.bits.uploadAsync(0, std.mem.sliceAsBytes(masks[0 .. count * 3]), stream.handle);
        var got: [rows]u32 = undefined;
        const rule: ?glm.sampler.Rule = if (sampled) glm.sampler.rule(.{ .seed = 73, .temperature = 0.8 }) else null;
        try sampler.masked(stream, .local, &.{.{ .lo = 0, .hi = vocab }}, &.{.{ .rank = 0, .logits = input }}, count, 17, rule, modules.glue.top1, got[0..count]);
        try check.sameBytes("masked GLM draw excludes larger forbidden logits", std.mem.sliceAsBytes(got[0..count]), std.mem.sliceAsBytes(wanted[0..count]));
    };
    check.pass("glm-grammar: first and multirow masks govern greedy and sampled draws", .{});
}
