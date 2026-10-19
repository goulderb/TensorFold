const std = @import("std");
const builtin = @import("builtin");
const cuda = @import("cuda");
const admission = @import("admission");
const options = @import("options");
const floor_bytes = 1536 * 1024 * 1024;
const model_dir = "C:/Development/local-ai-server/models/ornith15-35b-nvfp4-94e431d9";
const Mode = enum { quant, boundaries };
const Input = struct {
    prompt: []const u32,
    token: u32,
    repetitions: usize = 16,
    output_tokens: usize = 128,
    context: usize = 4096,
    cut: usize = 20,
    numeric_seam_fixture: []const u8,
    numeric_seam_fixture_sha256: []const u8,
    mode: Mode,
};
const File = struct { bytes: usize, sha256: []const u8, shape: []const usize, dtype: []const u8 };
const required_files = [_][]const u8{
    "layer-03-pos-001-input_norm.bf16",        "layer-03-pos-001-q_gate.bf16",            "layer-03-pos-001-value.bf16",
    "layer-14-pos-000-input.bf16",             "layer-14-pos-000-mixer_out.bf16",         "layer14.before.bf16",
    "layer14.hf.expected.f32",                 "layer14.norm.plus-one.f32",               "layer14.norm.raw.bf16",
    "layer14.residual.expected.bf16",          "q_proj.expected-activation.fragment.fp8", "q_proj.expected-activation.row.fp8",
    "q_proj.hf.expected.f32",                  "q_proj.scalars.f32",                      "q_proj.weight.fp8",
    "v_proj.expected-activation.fragment.fp8", "v_proj.expected-activation.row.fp8",      "v_proj.hf.expected.f32",
    "v_proj.scalars.f32",                      "v_proj.weight.fp8",
};
const Fixture = struct {
    files: std.StringHashMap([]const u8),
    manifest: std.json.Value,

    fn get(f: Fixture, name: []const u8) ![]const u8 {
        return f.files.get(name) orelse error.MissingFixtureFile;
    }
    fn scalar(f: Fixture, name: []const u8, index: usize) !f32 {
        const bytes = try f.get(name);
        return @bitCast(std.mem.readInt(u32, bytes[index * 4 ..][0..4], .little));
    }
};
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
fn hash(bytes: []const u8) [64]u8 {
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &digest, .{});
    return std.fmt.bytesToHex(digest, .lower);
}
fn verifyHash(bytes: []const u8, expected: []const u8) !void {
    if (!std.mem.eql(u8, &hash(bytes), expected)) return error.HashMismatch;
}
fn path(a: std.mem.Allocator, dir: []const u8, name: []const u8) ![]const u8 {
    return std.fs.path.join(a, &.{ dir, name });
}
fn read(a: std.mem.Allocator, io: std.Io, name: []const u8) ![]const u8 {
    return std.Io.Dir.cwd().readFileAlloc(io, name, a, .limited(32 << 20));
}
fn parse(a: std.mem.Allocator, io: std.Io, name: []const u8) !Input {
    const input = try std.json.parseFromSliceLeaky(Input, a, try read(a, io, name), .{});
    if (input.prompt.len != 64 or input.output_tokens != 128 or input.context != 4096 or input.cut != 20 or input.repetitions == 0) return error.InvalidInput;
    if (!std.fs.path.isAbsolute(input.numeric_seam_fixture) or input.numeric_seam_fixture_sha256.len != 64) return error.InvalidFixture;
    return input;
}
fn requireShape(f: File, expected: []const usize, dtype: []const u8) !void {
    if (!std.mem.eql(usize, f.shape, expected) or !std.mem.eql(u8, f.dtype, dtype)) return error.FixtureShapeMismatch;
    var count: usize = 1;
    for (expected) |n| count *= n;
    const size: usize = if (std.mem.startsWith(u8, dtype, "f32")) 4 else if (std.mem.eql(u8, dtype, "bf16")) 2 else 1;
    if (f.bytes != count * size) return error.FixtureSizeMismatch;
}
fn loadFixture(a: std.mem.Allocator, io: std.Io, input: Input) !Fixture {
    const bytes = try read(a, io, try path(a, input.numeric_seam_fixture, "fixture.json"));
    try verifyHash(bytes, input.numeric_seam_fixture_sha256);
    const manifest = try std.json.parseFromSliceLeaky(std.json.Value, a, bytes, .{});
    const entries = manifest.object.get("files") orelse return error.MissingFixtureFiles;
    if (entries.object.count() != 20) return error.FixtureFileCount;
    var result: Fixture = .{ .files = std.StringHashMap([]const u8).init(a), .manifest = manifest };
    var iter = entries.object.iterator();
    while (iter.next()) |entry| {
        const name = entry.key_ptr.*;
        if (!std.mem.eql(u8, std.fs.path.basename(name), name)) return error.InvalidFixturePath;
        const spec = try std.json.parseFromValueLeaky(File, a, entry.value_ptr.*, .{});
        if (std.mem.eql(u8, name, "q_proj.weight.fp8")) {
            try requireShape(spec, &.{ 8192, 2048 }, "float8_e4m3fn-row-major");
        } else if (std.mem.eql(u8, name, "v_proj.weight.fp8")) {
            try requireShape(spec, &.{ 512, 2048 }, "float8_e4m3fn-row-major");
        } else if (std.mem.endsWith(u8, name, ".scalars.f32")) {
            try requireShape(spec, &.{4}, "f32-act-weight-inverse-alpha");
        } else if (std.mem.endsWith(u8, name, ".fragment.fp8")) {
            try requireShape(spec, &.{2048}, "float8_e4m3fn-fragment-order");
        } else if (std.mem.endsWith(u8, name, ".row.fp8")) {
            try requireShape(spec, &.{2048}, "float8_e4m3fn-row-major");
        } else if (std.mem.endsWith(u8, name, ".hf.expected.f32")) {
            const n: usize = if (std.mem.startsWith(u8, name, "q_proj")) 8192 else if (std.mem.startsWith(u8, name, "v_proj")) 512 else 2048;
            try requireShape(spec, &.{n}, "f32-of-bf16");
        } else if (std.mem.eql(u8, name, "layer14.norm.plus-one.f32")) {
            try requireShape(spec, &.{2048}, "f32");
        } else if (std.mem.endsWith(u8, name, ".bf16")) {
            const n: usize = if (std.mem.eql(u8, name, "layer-03-pos-001-q_gate.bf16")) 8192 else if (std.mem.eql(u8, name, "layer-03-pos-001-value.bf16")) 512 else 2048;
            try requireShape(spec, &.{n}, "bf16");
        } else return error.UnknownFixtureFile;
        const data = try read(a, io, try path(a, input.numeric_seam_fixture, name));
        if (data.len != spec.bytes) return error.FixtureSizeMismatch;
        try verifyHash(data, spec.sha256);
        if (std.mem.eql(u8, spec.dtype, "bf16")) {
            for (0..data.len / 2) |i| {
                const bits = @as(u32, std.mem.readInt(u16, data[i * 2 ..][0..2], .little)) << 16;
                if (!std.math.isFinite(@as(f32, @bitCast(bits)))) return error.NonfiniteFixture;
            }
        } else if (std.mem.startsWith(u8, spec.dtype, "f32")) {
            for (0..data.len / 4) |i| if (!std.math.isFinite(@as(f32, @bitCast(std.mem.readInt(u32, data[i * 4 ..][0..4], .little))))) return error.NonfiniteFixture;
        } else {
            for (data) |code| if (code & 127 == 127) return error.NonfiniteFixture;
        }
        try result.files.put(name, data);
    }
    for (required_files) |name| _ = try result.get(name);
    inline for (.{ "q_proj", "v_proj" }) |p| {
        const act = try result.scalar(p ++ ".scalars.f32", 0);
        const scale = try result.scalar(p ++ ".scalars.f32", 1);
        if (!(act > 0) or !std.math.isFinite(act) or !(scale > 0) or !std.math.isFinite(scale)) return error.InvalidScale;
        if (@as(u32, @bitCast(cuda.ck.inverse(act))) != @as(u32, @bitCast(try result.scalar(p ++ ".scalars.f32", 2))) or @as(u32, @bitCast(cuda.ck.alpha(act, scale))) != @as(u32, @bitCast(try result.scalar(p ++ ".scalars.f32", 3)))) return error.ScalarArithmeticMismatch;
    }
    if (!std.mem.eql(u8, (try result.get("q_proj.scalars.f32"))[0..4], (try result.get("v_proj.scalars.f32"))[0..4])) return error.UnequalActivationScales;
    if (!std.mem.eql(u8, try result.get("q_proj.expected-activation.fragment.fp8"), try result.get("v_proj.expected-activation.fragment.fp8"))) return error.UnequalActivationReferences;
    const images = manifest.object.get("inputs") orelse return error.MissingImageHashes;
    inline for (.{ "ck_lane", "ck_experts", "norms", "qwen35_ops", "qwen35_hf", "modelopt_pack" }) |name| {
        var found = false;
        var image_iter = images.object.iterator();
        while (image_iter.next()) |entry| {
            if (std.mem.endsWith(u8, entry.key_ptr.*, name ++ ".fatbin")) {
                try verifyHash(@field(cuda.kernels, name), entry.value_ptr.object.get("sha256").?.string);
                found = true;
            }
        }
        if (!found) return error.MissingImageHashes;
    }
    return result;
}
fn differences(a: []const u8, b: []const u8) usize {
    std.debug.assert(a.len == b.len);
    var n: usize = 0;
    for (a, b) |x, y| n += @intFromBool(x != y);
    return n;
}
fn fp8Value(code: u8) ?f32 {
    if (code & 127 == 127) return null;
    const exponent = (code >> 3) & 15;
    const scales = [_]f32{ 0.001953125, 0.001953125, 0.00390625, 0.0078125, 0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32 };
    const significand: f32 = @floatFromInt((code & 7) + @as(u8, if (exponent == 0) 0 else 8));
    return significand * scales[exponent] * @as(f32, if (code & 128 == 0) 1 else -1);
}
fn save(j: Journal, out: []const u8, ordinal: c_int, name: []const u8, bytes: []const u8) !void {
    const filename = try std.fmt.allocPrint(j.a, "gpu-{d}-{s}", .{ ordinal, name });
    try std.Io.Dir.cwd().writeFile(j.io, .{ .sub_path = try path(j.a, out, filename), .data = bytes });
    try j.write(.{ .kind = "artifact", .ordinal = ordinal, .file = filename, .bytes = bytes.len, .sha256 = &hash(bytes) });
}
fn memory(ctx: *cuda.Context, j: Journal, phase: []const u8) !void {
    const info = try ctx.memInfo();
    try j.write(.{ .kind = "memory", .ordinal = ctx.device_ordinal, .phase = phase, .free_bytes = info.free, .floor_bytes = floor_bytes, .sample_ns = std.Io.Clock.awake.now(j.io).toNanoseconds() });
    if (info.free < floor_bytes) return error.GpuMemoryFloor;
}
fn download(j: Journal, out: []const u8, ordinal: c_int, name: []const u8, buffer: cuda.DeviceBuffer) ![]const u8 {
    const bytes = try j.a.alloc(u8, buffer.len);
    try buffer.download(0, bytes);
    try save(j, out, ordinal, name, bytes);
    return bytes;
}
fn poisonBf16(a: std.mem.Allocator, buffer: cuda.DeviceBuffer) !void {
    const bytes = try a.alloc(u8, buffer.len);
    defer a.free(bytes);
    @memset(bytes, 0xff);
    try buffer.upload(0, bytes);
}
fn compare(j: Journal, ordinal: c_int, name: []const u8, actual: []const u8, expected: []const u8) !bool {
    const count = differences(actual, expected);
    try j.write(.{ .kind = "comparison", .ordinal = ordinal, .name = name, .bytes = actual.len, .mismatched_bytes = count, .actual_sha256 = &hash(actual), .expected_sha256 = &hash(expected) });
    return count == 0;
}
fn hfCompare(j: Journal, ordinal: c_int, name: []const u8, actual: []const u8, expected: []const u8) !void {
    const expanded = try j.a.alloc(u8, actual.len * 2);
    for (0..actual.len / 2) |i| std.mem.writeInt(u32, expanded[i * 4 ..][0..4], @as(u32, std.mem.readInt(u16, actual[i * 2 ..][0..2], .little)) << 16, .little);
    _ = try compare(j, ordinal, name, expanded, expected);
}
fn quant(d: *cuda.Driver, s: cuda.Stream, ck: *const cuda.ck.Ck, f: Fixture, j: Journal, out: []const u8, ordinal: c_int) !bool {
    var x = try cuda.DeviceBuffer.fromHost(d, try f.get("layer-03-pos-001-input_norm.bf16"));
    defer x.free();
    var encoded = try cuda.DeviceBuffer.alloc(d, 2048);
    defer encoded.free();
    const act = try f.scalar("q_proj.scalars.f32", 0);
    try j.write(.{ .kind = "binding", .ordinal = ordinal, .symbol = "tf_ck_quant8", .function_handle = @intFromPtr(ck.quant8_fn.handle), .grid = .{ 1, 1, 1 }, .block = .{ 128, 1, 1 }, .m = 1, .k = 2048, .ldx = 2048, .act_bits = @as(u32, @bitCast(act)), .inverse_bits = @as(u32, @bitCast(cuda.ck.inverse(act))) });
    var previous: ?[]const u8 = null;
    var good = true;
    for ([_]u8{ 0x7f, 0xff }, 0..) |poison, repeat| {
        const poisoned = try j.a.alloc(u8, 2048);
        @memset(poisoned, poison);
        try encoded.upload(0, poisoned);
        try ck.quant8(s, x.ptr, 2048, 1, 2048, act, encoded.ptr);
        try s.synchronize();
        const codes = try download(j, out, ordinal, try std.fmt.allocPrint(j.a, "quant-{d}.fp8", .{repeat}), encoded);
        good = (try compare(j, ordinal, "quant-reference", codes, try f.get("q_proj.expected-activation.fragment.fp8"))) and good;
        if (previous) |before| good = (try compare(j, ordinal, "quant-poison-determinism", codes, before)) and good;
        previous = codes;
        const expected = try f.get("q_proj.expected-activation.fragment.fp8");
        const inputs = try f.get("layer-03-pos-001-input_norm.bf16");
        const order = [_]usize{ 0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15 };
        const row_codes = try j.a.alloc(u8, codes.len);
        for (codes, 0..) |code, i| row_codes[(i / 16) * 16 + order[i % 16]] = code;
        try save(j, out, ordinal, try std.fmt.allocPrint(j.a, "quant-{d}.row.fp8", .{repeat}), row_codes);
        good = (try compare(j, ordinal, "quant-row-reference", row_codes, try f.get("q_proj.expected-activation.row.fp8"))) and good;
        for (codes, expected, 0..) |actual, reference, i| if (actual != reference) {
            const row_index = (i / 16) * 16 + order[i % 16];
            try j.write(.{ .kind = "quant_mismatch", .ordinal = ordinal, .repeat = repeat, .fragment_index = i, .input_index = row_index, .actual_code = actual, .reference_code = reference, .actual_value = fp8Value(actual), .reference_value = fp8Value(reference), .input_bf16_bits = std.mem.readInt(u16, inputs[row_index * 2 ..][0..2], .little), .act_bits = @as(u32, @bitCast(act)) });
        };
    }
    return good;
}
fn projection(d: *cuda.Driver, s: cuda.Stream, ck: *const cuda.ck.Ck, f: Fixture, j: Journal, out: []const u8, ordinal: c_int, comptime p: []const u8, n: usize, comptime reference: []const u8) !bool {
    var packer = try cuda.modelopt.Packer.load(d);
    defer packer.unload();
    const raw = try f.get(p ++ ".weight.fp8");
    var source = try cuda.DeviceBuffer.fromHost(d, raw);
    defer source.free();
    var packed_weight = try cuda.DeviceBuffer.alloc(d, raw.len);
    defer packed_weight.free();
    try packer.fp8Lane(s, source.ptr, packed_weight.ptr, n, 2048);
    try s.synchronize();
    const actual_pack = try download(j, out, ordinal, p ++ ".packed.fp8", packed_weight);
    const cpu_pack = try j.a.alloc(u8, raw.len);
    cuda.fp8.packCodes(cpu_pack, raw, n, 2048);
    var good = try compare(j, ordinal, p ++ ".pack-control", actual_pack, cpu_pack);
    var x = try cuda.DeviceBuffer.fromHost(d, try f.get("layer-03-pos-001-input_norm.bf16"));
    defer x.free();
    var codes = try cuda.DeviceBuffer.alloc(d, 2048);
    defer codes.free();
    var y = try cuda.DeviceBuffer.alloc(d, n * 2);
    defer y.free();
    var partials = try cuda.DeviceBuffer.alloc(d, n * 4 * 4);
    defer partials.free();
    var w = cuda.modelopt.fp8Weight(packed_weight.ptr, try f.scalar(p ++ ".scalars.f32", 1), n, 2048);
    w.act = try f.scalar(p ++ ".scalars.f32", 0);
    const sk = cuda.qmmf.splitK(w.n, w.k);
    if (!ck.clusters or sk != (if (n == 8192) @as(u32, 2) else 4)) return error.UnexpectedProjectionBinding;
    try poisonBf16(j.a, y);
    try ck.quant8(s, x.ptr, 2048, 1, 2048, w.act, codes.ptr);
    try ck.lane8(s, codes.ptr, 1, w, w.act, y.ptr, partials.ptr);
    try s.synchronize();
    try j.write(.{ .kind = "binding", .ordinal = ordinal, .projection = p, .symbol = "tf_ck8_16c", .function_handle = @intFromPtr(ck.lane[1].handle), .grid = .{ n / 64, 1, sk }, .block = .{ 128, 1, 1 }, .cluster = .{ 1, 1, sk }, .shared_bytes = 20480, .m = 1, .n = n, .k = 2048, .split_k = sk, .act_bits = @as(u32, @bitCast(w.act)), .weight_scale_bits = @as(u32, @bitCast(w.scale)), .alpha_bits = @as(u32, @bitCast(cuda.ck.alpha(w.act, w.scale))) });
    const actual_codes = try download(j, out, ordinal, p ++ ".actual-activation.fp8", codes);
    good = (try compare(j, ordinal, p ++ ".activation", actual_codes, try f.get(p ++ ".expected-activation.fragment.fp8"))) and good;
    const actual = try download(j, out, ordinal, p ++ ".bf16", y);
    good = (try compare(j, ordinal, p ++ ".native-control", actual, try f.get(reference))) and good;
    try hfCompare(j, ordinal, p ++ ".hf-diagnostic", actual, try f.get(p ++ ".hf.expected.f32"));
    return good;
}
fn normalization(d: *cuda.Driver, s: cuda.Stream, f: Fixture, j: Journal, out: []const u8, ordinal: c_int) !bool {
    var norms = try cuda.norms.Norms.load(d);
    defer norms.unload();
    var ops = try cuda.Module.load(d, cuda.kernels.qwen35_ops);
    defer ops.unload();
    var hf = try cuda.Module.load(d, cuda.kernels.qwen35_hf);
    defer hf.unload();
    var h = try cuda.DeviceBuffer.fromHost(d, try f.get("layer14.before.bf16"));
    defer h.free();
    var r = try cuda.DeviceBuffer.fromHost(d, try f.get("layer-14-pos-000-mixer_out.bf16"));
    defer r.free();
    var w = try cuda.DeviceBuffer.fromHost(d, try f.get("layer14.norm.plus-one.f32"));
    defer w.free();
    var y = try cuda.DeviceBuffer.alloc(d, 4096);
    defer y.free();
    var good = true;
    for (0..2) |arm| {
        try h.upload(0, try f.get("layer14.before.bf16"));
        try poisonBf16(j.a, y);
        if (arm == 0) {
            var args: cuda.Args = .{};
            args.add(h.ptr);
            args.add(r.ptr);
            args.add(@as(i32, 2048));
            try cuda.launch.launch(try ops.function("tf_qwen35_add"), .{ .grid = .{ .x = 8 }, .block = .{ .x = 256 } }, s, &args);
            try norms.rms(s, h.ptr, .bf16, w.ptr, y.ptr, .bf16, 1, 2048, 1e-6);
        } else {
            var args: cuda.Args = .{};
            for ([_]u64{ h.ptr, r.ptr, w.ptr, y.ptr }) |ptr| args.add(ptr);
            args.add(@as(f32, 1e-6));
            args.add(@as(i32, 2048));
            args.add(@as(u64, 0));
            try cuda.launch.launch(try hf.function("tf_q35_hf_norm"), .{ .grid = .{ .x = 1 }, .block = .{ .x = 256 } }, s, &args);
        }
        try s.synchronize();
        try j.write(.{ .kind = "binding", .ordinal = ordinal, .arm = arm, .symbol = if (arm == 0) "tf_rms_rows_bf16" else "tf_q35_hf_norm", .grid = .{ 1, 1, 1 }, .block = .{ 256, 1, 1 }, .width = 2048, .eps_bits = @as(u32, @bitCast(@as(f32, 1e-6))), .add_symbol = if (arm == 0) "tf_qwen35_add" else "fused", .add_grid_x = if (arm == 0) @as(usize, 8) else 1 });
        const residual = try download(j, out, ordinal, try std.fmt.allocPrint(j.a, "norm-{d}.residual.bf16", .{arm}), h);
        good = (try compare(j, ordinal, "norm-residual-control", residual, try f.get("layer14.residual.expected.bf16"))) and good;
        const actual = try download(j, out, ordinal, try std.fmt.allocPrint(j.a, "norm-{d}.bf16", .{arm}), y);
        good = (try compare(j, ordinal, "norm-native-control", actual, try f.get("layer-14-pos-000-input.bf16"))) and good;
        try hfCompare(j, ordinal, "norm-hf-diagnostic", actual, try f.get("layer14.hf.expected.f32"));
    }
    return good;
}
fn execute(init: std.process.Init, input: Input, f: Fixture, out: []const u8, j: Journal) !bool {
    const uuids = [_][]const u8{ "GPU-f2c0ea1e-a2be-f0b4-6c23-ffe33b537a06", "GPU-dd2a988c-fe38-24ab-233a-9e9bb3b2645a" };
    const admitted = try admission.Admission.parse(init.environ_map.get("CUDA_DEVICE_ORDER"), init.environ_map.get("CUDA_VISIBLE_DEVICES"), &.{ 0, 1 }, &uuids);
    var d = try cuda.Driver.open();
    defer d.close();
    const identities = try admitted.verify(j.a, &d);
    try j.write(.{ .kind = "start", .source_commit = options.source_commit, .implementation = "qwen-numeric-seam", .identities = identities, .fixture_sha256 = input.numeric_seam_fixture_sha256, .mode = @tagName(input.mode), .scope = "frozen operator controls; HF differences are diagnostic; no model load", .prompt_tokens = input.prompt.len, .output_tokens = input.output_tokens, .context = input.context, .cut = input.cut, .sample_ns = std.Io.Clock.awake.now(j.io).toNanoseconds() });
    try j.write(.{ .kind = "fixture", .manifest = f.manifest });
    var good = true;
    for ([_]Mode{ .quant, .boundaries }) |phase| {
        if (phase == .boundaries and (input.mode != .boundaries or !good)) break;
        for (identities) |identity| {
            var ctx = try cuda.Context.init(&d, identity.ordinal);
            defer ctx.deinit();
            try j.write(.{ .kind = "device_phase", .ordinal = identity.ordinal, .phase = @tagName(phase) });
            try memory(&ctx, j, "before");
            var s = try cuda.Stream.init(&d, true);
            defer s.deinit();
            var ck = try cuda.ck.Ck.load(&d, (try ctx.capability()) / 10, @intCast(try ctx.attribute(.multiprocessor_count)));
            defer ck.unload();
            if (phase == .quant) {
                good = (try quant(&d, s, &ck, f, j, out, identity.ordinal)) and good;
            } else {
                good = (try projection(&d, s, &ck, f, j, out, identity.ordinal, "q_proj", 8192, "layer-03-pos-001-q_gate.bf16")) and good;
                good = (try projection(&d, s, &ck, f, j, out, identity.ordinal, "v_proj", 512, "layer-03-pos-001-value.bf16")) and good;
                good = (try normalization(&d, s, f, j, out, identity.ordinal)) and good;
            }
            try memory(&ctx, j, "after");
        }
    }
    return good;
}
pub fn main(init: std.process.Init) !u8 {
    const a = init.arena.allocator();
    const args = try init.minimal.args.toSlice(a);
    if (args.len == 2 and std.mem.eql(u8, args[1], "--identity")) {
        const j: Journal = .{ .a = a, .io = init.io, .file = std.Io.File.stdout() };
        try j.write(.{ .source_commit = options.source_commit, .engine_optimization = @tagName(options.engine_optimization), .root_optimization = @tagName(builtin.mode), .implementation = "qwen-numeric-seam" });
        return 0;
    }
    if (args.len == 3 and std.mem.eql(u8, args[1], "--check")) {
        _ = try loadFixture(a, init.io, try parse(a, init.io, args[2]));
        return 0;
    }
    if (args.len != 4) return 2;
    const canonical = try std.fs.path.resolve(a, &.{args[1]});
    const expected = try std.fs.path.resolve(a, &.{model_dir});
    if (!std.ascii.eqlIgnoreCase(canonical, expected)) return error.UnexpectedModelDirectory;
    const input = try parse(a, init.io, args[2]);
    const f = try loadFixture(a, init.io, input);
    try std.Io.Dir.cwd().createDir(init.io, args[3], .default_dir);
    const file = try std.Io.Dir.cwd().createFile(init.io, try path(a, args[3], "results.jsonl"), .{ .exclusive = true });
    defer file.close(init.io);
    const j: Journal = .{ .a = a, .io = init.io, .file = file };
    const good = execute(init, input, f, args[3], j) catch |err| {
        try j.write(.{ .kind = "failure", .error_name = @errorName(err) });
        try j.write(.{ .kind = "complete", .correctness_pass = false, .independent_hf_gate_pass = false, .production_qualification = false });
        return 1;
    };
    try j.write(.{ .kind = "complete", .correctness_pass = good, .scope = "activation input parity and frozen native control reproduction only", .independent_hf_gate_pass = false, .production_qualification = false });
    return if (good) 0 else 1;
}
