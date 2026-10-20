//! Model-free native WY prompt arithmetic against the pinned production kernel's raw fixtures.
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm_cuda");
const check = @import("check.zig");
const Rig = @import("quant_ref_tests.zig").RigT;
const kda = glm.kda;

const Files = struct {
    p: []const u8,
    a: []const u8,
    g: []const u8,
    conv_state: []const u8,
    conv_w: []const u8,
    state_in: []const u8,
    a_log: []const u8,
    dt_bias: []const u8,
    norm_w: []const u8,
    out: []const u8,
    state_out: []const u8,
};
const Case = struct {
    name: []const u8,
    H: usize,
    R: usize,
    pos: usize,
    p_stride: usize,
    b_off: usize,
    a_stride: usize,
    g_stride: usize,
    eps: f32,
    lower: f32,
    files: Files,
};
const Manifest = struct { version: usize, cases: []const Case };
const Guarded = struct { base: u64, ptr: u64, bytes: usize };
const guard_bytes = 32;
const guard_value = 0xa5;

fn read(r: *Rig, dir: []const u8, name: []const u8, bytes: usize) ![]u8 {
    const path = try std.fs.path.join(r.a, &.{ dir, name });
    const data = try std.Io.Dir.cwd().readFileAlloc(r.gpu.io, path, r.a, .limited(256 << 20));
    try check.expect(data.len == bytes, "KDA fixture {s}: {d} bytes, expected {d}", .{ path, data.len, bytes });
    return data;
}

fn guarded(r: *Rig, bytes: usize) !Guarded {
    const data = try r.a.alloc(u8, bytes + 2 * guard_bytes);
    @memset(data, guard_value);
    const base = try r.dev(data);
    return .{ .base = base, .ptr = base + guard_bytes, .bytes = bytes };
}

fn guards(r: *Rig, buf: Guarded) !void {
    const before = try r.back(buf.base, guard_bytes);
    const after = try r.back(buf.ptr + buf.bytes, guard_bytes);
    for (before) |v| try check.expect(v == guard_value, "KDA output leading guard intact", .{});
    for (after) |v| try check.expect(v == guard_value, "KDA output trailing guard intact", .{});
}

fn chainInput(r: *Rig, dir: []const u8, c: Case) !kda.Chain {
    const n = c.H * kda.dk;
    return .{
        .p = try r.dev(try read(r, dir, c.files.p, c.R * c.p_stride * 2)),
        .p_stride = c.p_stride,
        .b_off = c.b_off,
        .a = try r.dev(try read(r, dir, c.files.a, c.R * c.a_stride * 2)),
        .a_stride = c.a_stride,
        .g = try r.dev(try read(r, dir, c.files.g, c.R * c.g_stride * 2)),
        .g_stride = c.g_stride,
        .conv_state = try r.dev(try read(r, dir, c.files.conv_state, 9 * n * 2)),
        .conv_w = try r.dev(try read(r, dir, c.files.conv_w, 3 * n * 4 * 2)),
        .state_in = try r.dev(try read(r, dir, c.files.state_in, c.H * kda.dv * kda.dk * 4)),
        .a_log = try r.dev(try read(r, dir, c.files.a_log, c.H * 4)),
        .dt_bias = try r.dev(try read(r, dir, c.files.dt_bias, n * 4)),
        .norm_w = try r.dev(try read(r, dir, c.files.norm_w, kda.dk * 2)),
        .eps = c.eps,
        .lower = c.lower,
        .heads = c.H,
    };
}

fn suffix(c: kda.Chain, row: usize, state: u64, conv: u64) kda.Chain {
    var s = c;
    s.p += row * c.p_stride * 2;
    s.a += row * c.a_stride * 2;
    s.g += row * c.g_stride * 2;
    s.state_in = state;
    s.conv_state = conv;
    return s;
}

fn serialSaves(r: *Rig, rows: usize, heads: usize) !kda.Saves {
    const n = heads * kda.dk;
    return .{
        .out = try r.zeros(rows * n * 2),
        .state_out = try r.zeros(heads * kda.dv * kda.dk * 4),
        .k = try r.zeros(rows * n * 4),
        .v = try r.zeros(rows * n * 2),
        .g = try r.zeros(rows * n * 4),
        .b = try r.zeros(rows * heads * 4),
    };
}

fn continuation(r: *Rig, ks: *const kda.Kernels, conv_shift: cuda.Function, c: Case, input: kda.Chain, want_out: []const u8, want_state: []const u8) !void {
    const n = c.H * kda.dk;
    const state_bytes = c.H * kda.dv * kda.dk * 4;
    const first_out = try guarded(r, 32 * n * 2);
    const first_state = try guarded(r, state_bytes);
    try kda.chunk(ks, r.s, input, 32, c.pos, first_out.ptr, first_state.ptr);
    try check.sameBytes("KDA aligned first32 output", try r.back(first_out.ptr, 32 * n * 2), want_out[0 .. 32 * n * 2]);

    const conv = try r.dev(try r.back(input.conv_state, 9 * n * 2));
    var args: cuda.Args = .{};
    args.add(conv);
    args.add(input.p);
    args.add(@as(i64, 32));
    args.add(@as(i64, @intCast(9 * n)));
    args.add(@as(i64, @intCast(32 * c.p_stride)));
    args.add(@as(i64, @intCast(c.p_stride)));
    args.add(@as(i64, @intCast(3 * n)));
    // Use the production convolution commit kernel between prompt chunks.
    try cuda.launch.launch(conv_shift, .{ .grid = .{ .x = 1, .y = @intCast((3 * n + 255) / 256) }, .block = .{ .x = 256 } }, r.s, &args);

    const state_host = try r.back(first_state.ptr, state_bytes);
    const conv_host = try r.back(conv, 9 * n * 2);
    const restored_state = try r.dev(state_host);
    const restored_conv = try r.dev(conv_host);
    const next = suffix(input, 32, first_state.ptr, conv);
    const restored = suffix(input, 32, restored_state, restored_conv);
    const final_out = try guarded(r, 32 * n * 2);
    const final_state = try guarded(r, state_bytes);
    const resume_out = try guarded(r, 32 * n * 2);
    const resume_state = try guarded(r, state_bytes);
    try kda.chunk(ks, r.s, next, 32, c.pos + 32, final_out.ptr, final_state.ptr);
    try kda.chunk(ks, r.s, restored, 32, c.pos + 32, resume_out.ptr, resume_state.ptr);
    try check.sameBytes("KDA aligned32+32 suffix output", try r.back(final_out.ptr, 32 * n * 2), want_out[32 * n * 2 ..]);
    try check.sameBytes("KDA aligned32+32 final state", try r.back(final_state.ptr, state_bytes), want_state);
    try check.sameBytes("KDA restored32+32 suffix output", try r.back(resume_out.ptr, 32 * n * 2), want_out[32 * n * 2 ..]);
    try check.sameBytes("KDA restored32+32 final state", try r.back(resume_state.ptr, state_bytes), want_state);
    for ([_]Guarded{ first_out, first_state, final_out, final_state, resume_out, resume_state }) |buf| try guards(r, buf);

    const conv_bf16 = try r.back(input.conv_w, 3 * n * 4 * 2);
    const conv_f32 = try r.a.alloc(f32, 3 * n * 4);
    for (conv_f32, 0..) |*v, i| {
        const bits = std.mem.readInt(u16, conv_bf16[2 * i ..][0..2], .little);
        v.* = @bitCast(@as(u32, bits) << 16);
    }
    const serial_conv = try r.dev(std.mem.sliceAsBytes(conv_f32));
    var serial_next = next;
    serial_next.conv_w = serial_conv;
    var serial_restored = restored;
    serial_restored.conv_w = serial_conv;
    const decode_rows = 3;
    const fused = try serialSaves(r, decode_rows, c.H);
    const wide = try serialSaves(r, decode_rows, c.H);
    const workspace: kda.Wide = .{ .q = try r.zeros(decode_rows * n * 4), .y = try r.zeros(decode_rows * n * 2) };
    try kda.chain(ks, r.s, serial_next, decode_rows, fused, null);
    try kda.chain(ks, r.s, serial_restored, decode_rows, wide, workspace);
    try check.sameBytes("KDA WY to serial decode fused/wide output", try r.back(fused.out, decode_rows * n * 2), try r.back(wide.out, decode_rows * n * 2));
    try check.sameBytes("KDA WY to serial decode fused/wide state", try r.back(fused.state_out, state_bytes), try r.back(wide.state_out, state_bytes));
    const replayed = try r.zeros(state_bytes);
    try kda.replay(ks, r.s, first_state.ptr, fused.k, fused.v, fused.g, fused.b, decode_rows, c.H, replayed);
    try check.sameBytes("KDA WY to serial decode replay", try r.back(replayed, state_bytes), try r.back(fused.state_out, state_bytes));
    check.pass("KDA {s}: aligned32+32, restored state/conv and WY to serial decode/replay", .{c.name});
}

fn one(gpu: check.Gpu, stream: cuda.Stream, ks: *const kda.Kernels, conv_shift: cuda.Function, dir: []const u8, c: Case) !void {
    var arena: std.heap.ArenaAllocator = .init(gpu.gpa);
    defer arena.deinit();
    var rng = std.Random.DefaultPrng.init(0);
    var r: Rig = .{ .gpu = gpu, .a = arena.allocator(), .s = stream, .rng = rng.random() };
    defer {
        for (r.bufs.items) |*b| b.free();
        r.bufs.deinit(gpu.gpa);
    }
    try check.expect(c.H > 0 and c.R > 0 and c.H <= 32 and c.R <= 2048, "KDA fixture dimensions bounded", .{});
    try check.expect(c.p_stride >= 3 * c.H * kda.dk and c.a_stride >= c.H * kda.dk and c.g_stride >= c.H * kda.dk and c.b_off + c.H <= c.p_stride, "KDA fixture strides bounded", .{});
    const input = try chainInput(&r, dir, c);
    const out_bytes = c.R * c.H * kda.dk * 2;
    const state_bytes = c.H * kda.dv * kda.dk * 4;
    const want_out = try read(&r, dir, c.files.out, out_bytes);
    const want_state = try read(&r, dir, c.files.state_out, state_bytes);
    const output = try guarded(&r, out_bytes);
    const state = try guarded(&r, state_bytes);
    const sentinels = try r.a.alloc(u8, c.R * c.H * (kda.dk * 10 + 4));
    @memset(sentinels, 0x6b);
    const untouched = try r.dev(sentinels);
    try kda.chunk(ks, stream, input, c.R, c.pos, output.ptr, state.ptr);
    try check.sameBytes("KDA native WY equals production output", try r.back(output.ptr, out_bytes), want_out);
    try check.sameBytes("KDA native WY equals production state", try r.back(state.ptr, state_bytes), want_state);
    try check.sameBytes("KDA WY leaves unrelated serial replay saves untouched", try r.back(untouched, sentinels.len), sentinels);
    try guards(&r, output);
    try guards(&r, state);
    check.pass("KDA {s}: exact production bytes H={d} R={d} pos={d}, output/state guards and serial saves intact", .{ c.name, c.H, c.R, c.pos });
    if (c.H == 2 and c.R == 64 and c.pos == 0) try continuation(&r, ks, conv_shift, c, input, want_out, want_state);
}

pub fn run(gpu: check.Gpu, dir: []const u8) !void {
    var arena: std.heap.ArenaAllocator = .init(gpu.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const path = try std.fs.path.join(a, &.{ dir, "manifest.json" });
    const raw = try std.Io.Dir.cwd().readFileAlloc(gpu.io, path, a, .limited(1 << 20));
    const parsed = try std.json.parseFromSlice(Manifest, a, raw, .{ .ignore_unknown_fields = true });
    try check.expect(parsed.value.version == 1, "KDA fixture manifest version1", .{});
    var stream = try cuda.Stream.init(gpu.d, false);
    defer stream.deinit();
    var serial_module = try cuda.Module.load(gpu.d, cuda.kernels.glm_kda);
    defer serial_module.unload();
    var chunk_module = try cuda.Module.load(gpu.d, cuda.kernels.glm_kda_chunk);
    defer chunk_module.unload();
    var glue_module = try cuda.Module.load(gpu.d, cuda.kernels.glm_glue);
    defer glue_module.unload();
    const conv_shift = try glue_module.function("tf_glm_conv_shift");
    var ks = try kda.Kernels.load(serial_module);
    try ks.loadChunk(chunk_module);
    var production_shapes: [6]bool = @splat(false);
    const rows = [_]usize{ 31, 32, 33, 128, 512, 2048 };
    var offset31 = false;
    var offset32 = false;
    var aligned = false;
    for (parsed.value.cases) |c| {
        try one(gpu, stream, &ks, conv_shift, dir, c);
        if (c.H == 32 and c.pos == 0) for (rows, 0..) |r, i| {
            if (c.R == r) production_shapes[i] = true;
        };
        offset31 = offset31 or (c.H == 2 and c.pos == 31);
        offset32 = offset32 or (c.H == 2 and c.pos == 32);
        aligned = aligned or (c.H == 2 and c.R == 64 and c.pos == 0);
    }
    for (production_shapes, rows) |seen, r| try check.expect(seen, "KDA required H32 R{d} fixture present", .{r});
    try check.expect(offset31 and offset32 and aligned, "KDA offset31/32 and aligned continuation fixtures present", .{});
    check.pass("glm-kda-chunk-check: {d} pinned production cases, all required shapes and aligned restore/decode checks", .{parsed.value.cases.len});
}
