const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const guard = 64;
const floor_bytes = 1536 << 20;
const FilePin = struct { bytes: usize, sha256: []const u8 };
const Manifest = struct { format_version: usize, files: std.json.ArrayHashMap(FilePin) };
const frozen_ck_sha = "f978ef94102297ea32c199a003fcb8bb9e0a5acb81fa63b95cd3e4a882dcb97e";

fn hash(bytes: []const u8) [64]u8 {
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &digest, .{});
    return std.fmt.bytesToHex(digest, .lower);
}

fn read(gpu: check.Gpu, path: []const u8) ![]u8 {
    return std.Io.Dir.cwd().readFileAlloc(gpu.io, path, gpu.gpa, .limited(40 << 20));
}

fn fixture(gpu: check.Gpu, dir: []const u8, manifest: Manifest, name: []const u8, size: usize) ![]u8 {
    const pin = manifest.files.map.get(name) orelse return error.MissingMmaCaptureFixture;
    if (pin.bytes != size) return error.InvalidMmaCaptureFixture;
    const path = try std.fs.path.join(gpu.gpa, &.{ dir, name });
    defer gpu.gpa.free(path);
    const bytes = try read(gpu, path);
    errdefer gpu.gpa.free(bytes);
    if (bytes.len != size or !std.mem.eql(u8, &hash(bytes), pin.sha256)) return error.MmaCaptureFixtureHash;
    return bytes;
}

fn save(gpu: check.Gpu, journal: anytype, projection: []const u8, arm: []const u8, name: []const u8, bytes: []const u8) !void {
    const filename = try std.fmt.allocPrint(gpu.gpa, "mma_gpu{d}-{s}-{s}-{s}.raw", .{ gpu.ctx.device_ordinal, projection, arm, name });
    defer gpu.gpa.free(filename);
    const path = try std.fs.path.join(gpu.gpa, &.{ journal.directory, filename });
    defer gpu.gpa.free(path);
    const file = try std.Io.Dir.cwd().createFile(gpu.io, path, .{ .exclusive = true });
    defer file.close(gpu.io);
    try file.writeStreamingAll(gpu.io, bytes);
    try journal.write(.{ .kind = "mma_capture_artifact", .ordinal = gpu.ctx.device_ordinal, .projection = projection, .arm = arm, .name = name, .file = filename, .bytes = bytes.len, .sha256 = &hash(bytes) });
}

fn finite(bytes: []const u8, stride: usize) !void {
    if (bytes.len % stride != 0) return error.InvalidMmaCaptureScalar;
    var at: usize = 0;
    while (at < bytes.len) : (at += stride) {
        const bits = if (stride == 4) std.mem.readInt(u32, bytes[at..][0..4], .little) else @as(u32, std.mem.readInt(u16, bytes[at..][0..2], .little)) << 16;
        if (!std.math.isFinite(@as(f32, @bitCast(bits)))) return error.NonfiniteMmaCapture;
    }
}

fn guarded(gpu: check.Gpu, size: usize) !cuda.DeviceBuffer {
    var b = try cuda.DeviceBuffer.alloc(gpu.d, size + guard * 2);
    errdefer b.free();
    try b.fill8(0xa5, null);
    const active: cuda.DeviceBuffer = .{ .d = gpu.d, .ptr = b.ptr + guard, .len = size, .borrowed = true };
    try active.fill8(0xff, null);
    return b;
}

fn payload(raw: []const u8, size: usize) ![]const u8 {
    if (raw.len != size + guard * 2) return error.InvalidMmaCaptureReadback;
    for (raw[0..guard]) |v| if (v != 0xa5) return error.MmaCaptureGuard;
    for (raw[guard + size ..]) |v| if (v != 0xa5) return error.MmaCaptureGuard;
    return raw[guard..][0..size];
}

fn identity(gpu: check.Gpu, journal: anytype) !void {
    const expected = [_][]const u8{ "f2c0ea1ea2bef0b46c23ffe33b537a06", "dd2a988cfe3824ab233a9e9bb3b2645a" };
    if (gpu.ctx.device_ordinal < 0 or gpu.ctx.device_ordinal > 1) return error.UnexpectedMmaCaptureGpu;
    var lib = gpu.d.lib;
    const get = lib.lookup(*const fn (*[16]u8, c_int) callconv(.c) cuda.abi.Result, "cuDeviceGetUuid_v2") orelse return error.MissingSymbol;
    var uuid: [16]u8 = undefined;
    try gpu.d.check(get(&uuid, gpu.ctx.device), "cuDeviceGetUuid_v2");
    const hex = std.fmt.bytesToHex(uuid, .lower);
    if (!std.mem.eql(u8, &hex, expected[@intCast(gpu.ctx.device_ordinal)])) return error.UnexpectedMmaCaptureGpu;
    try journal.write(.{ .kind = "mma_capture_identity", .ordinal = gpu.ctx.device_ordinal, .uuid_hex = &hex, .free_bytes = (try gpu.ctx.memInfo()).free });
}

pub fn run(gpu: check.Gpu, dir: []const u8, manifest_sha: []const u8, image_path: []const u8, image_sha: []const u8, journal: anytype) !void {
    try identity(gpu, journal);
    if ((try gpu.ctx.memInfo()).free < floor_bytes) return error.GpuMemoryFloor;
    const manifest_path = try std.fs.path.join(gpu.gpa, &.{ dir, "manifest.json" });
    defer gpu.gpa.free(manifest_path);
    const manifest_bytes = try read(gpu, manifest_path);
    defer gpu.gpa.free(manifest_bytes);
    if (!std.mem.eql(u8, &hash(manifest_bytes), manifest_sha)) return error.MmaCaptureFixtureHash;
    const parsed = try std.json.parseFromSlice(Manifest, gpu.gpa, manifest_bytes, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    if (parsed.value.format_version != 1 or parsed.value.files.map.count() != 10) return error.InvalidMmaCaptureFixture;
    const image_raw = try read(gpu, image_path);
    defer gpu.gpa.free(image_raw);
    if (!std.mem.eql(u8, &hash(image_raw), image_sha)) return error.MmaCaptureImageHash;
    const image = try gpu.gpa.alignedAlloc(u8, .@"16", image_raw.len);
    defer gpu.gpa.free(image);
    @memcpy(image, image_raw);
    if (!std.mem.eql(u8, &hash(cuda.kernels.ck_lane), frozen_ck_sha)) return error.MmaCaptureFrozenImageHash;
    var original = try cuda.Module.load(gpu.d, cuda.kernels.ck_lane);
    defer original.unload();
    var capture = try cuda.Module.load(gpu.d, image);
    defer capture.unload();
    const functions = [_]cuda.Function{ try original.function("tf_ck8_16c"), try capture.function("tf_q35_mma_control"), try capture.function("tf_q35_mma_capture") };
    const arms = [_][]const u8{ "original", "control", "capture" };
    var stream = try cuda.Stream.init(gpu.d, false);
    defer stream.deinit();
    var coordinates_total: usize = 0;
    try journal.write(.{ .kind = "mma_capture_scope", .ordinal = gpu.ctx.device_ordinal, .fixture_sha256 = manifest_sha, .image_sha256 = image_sha, .frozen_image_sha256 = frozen_ck_sha, .full_model = false, .performance = false, .quality_gate = "FAIL unchanged", .guard_bytes = guard });
    inline for (.{ "q_proj", "v_proj" }, .{ @as(usize, 8192), @as(usize, 512) }, .{ @as(usize, 2), @as(usize, 4) }) |projection, n, split| {
        const coordinate_pin = parsed.value.files.map.get(projection ++ ".coords.i32") orelse return error.MissingMmaCaptureFixture;
        if (coordinate_pin.bytes == 0 or coordinate_pin.bytes > 8 * 4 or coordinate_pin.bytes % 4 != 0) return error.InvalidMmaCaptureCoordinates;
        const coordinate_bytes = try fixture(gpu, dir, parsed.value, projection ++ ".coords.i32", coordinate_pin.bytes);
        defer gpu.gpa.free(coordinate_bytes);
        var coordinate_values: [8]i32 = undefined;
        const coords = coordinate_values[0 .. coordinate_bytes.len / 4];
        for (coords, 0..) |*coordinate, index| {
            coordinate.* = std.mem.readInt(i32, coordinate_bytes[index * 4 ..][0..4], .little);
            if (coordinate.* < 0 or coordinate.* >= @as(i32, @intCast(n)) or std.mem.indexOfScalar(i32, coords[0..index], coordinate.*) != null) return error.InvalidMmaCaptureCoordinates;
        }
        const required: []const i32 = if (n == 8192) &.{ 3950, 4766 } else &.{77};
        for (required) |coordinate| if (std.mem.indexOfScalar(i32, coords, coordinate) == null) return error.InvalidMmaCaptureCoordinates;
        coordinates_total += coords.len;
        const trace_floats = coords.len * (split + 2);
        const packed_weight = try fixture(gpu, dir, parsed.value, projection ++ ".packed.fp8", n * 2048);
        defer gpu.gpa.free(packed_weight);
        const codes = try fixture(gpu, dir, parsed.value, projection ++ ".codes.fp8", 2048);
        defer gpu.gpa.free(codes);
        const expected = try fixture(gpu, dir, parsed.value, projection ++ ".reference.bf16", n * 2);
        defer gpu.gpa.free(expected);
        const scalars = try fixture(gpu, dir, parsed.value, projection ++ ".scalars.f32", 16);
        defer gpu.gpa.free(scalars);
        const alpha: f32 = @bitCast(std.mem.readInt(u32, scalars[12..16], .little));
        try finite(scalars, 4);
        var x = try cuda.DeviceBuffer.fromHost(gpu.d, codes);
        defer x.free();
        var w = try cuda.DeviceBuffer.fromHost(gpu.d, packed_weight);
        defer w.free();
        var coordinate_buffer = try cuda.DeviceBuffer.fromHost(gpu.d, coordinate_bytes);
        defer coordinate_buffer.free();
        var trace = try guarded(gpu, trace_floats * 4);
        defer trace.free();
        for (functions, arms, 0..) |function, arm, ai| {
            var output = try guarded(gpu, n * 2);
            defer output.free();
            var args: cuda.Args = .{};
            args.add(x.ptr);
            args.add(w.ptr);
            args.add(alpha);
            args.add(output.ptr + guard);
            args.add(@as(u64, 0));
            for ([_]usize{ 1, n, 2048, split, 1 }) |v| args.add(@as(i32, @intCast(v)));
            if (ai == 2) {
                args.add(trace.ptr + guard);
                args.add(coordinate_buffer.ptr);
                args.add(@as(i32, @intCast(coords.len)));
            }
            const config: cuda.Config = .{ .grid = .{ .x = n / 64, .y = 1, .z = split }, .block = .{ .x = 128, .y = 1, .z = 1 }, .shared = 20480, .cluster = .{ .x = 1, .y = 1, .z = split } };
            try cuda.launch.launch(function, config, stream, &args);
            try stream.synchronize();
            const raw = try check.download(gpu, output);
            defer gpu.gpa.free(raw);
            try save(gpu, journal, projection, arm, "output", raw);
            if (ai == 2) {
                const captured_trace = try check.download(gpu, trace);
                defer gpu.gpa.free(captured_trace);
                try save(gpu, journal, projection, "capture", "trace", captured_trace);
            }
            var argument_values: [13]u64 = @splat(0);
            for (0..args.count) |index| argument_values[index] = args.integer(index) orelse return error.InvalidMmaCaptureArgument;
            try journal.write(.{ .kind = "mma_capture_binding", .ordinal = gpu.ctx.device_ordinal, .projection = projection, .arm = arm, .symbol = ([_][]const u8{ "tf_ck8_16c", "tf_q35_mma_control", "tf_q35_mma_capture" })[ai], .grid = config.grid, .block = config.block, .cluster = config.cluster, .shared_bytes = config.shared, .argument_count = args.count, .argument_sizes = args.sizes[0..args.count], .argument_values = argument_values[0..args.count], .input_ptr = x.ptr, .weight_ptr = w.ptr, .output_ptr = output.ptr + guard, .trace_ptr = trace.ptr + guard, .coordinate_ptr = coordinate_buffer.ptr, .m = 1, .n = n, .k = 2048, .split_k = split, .group = 1, .coordinates = coords, .alpha_bits = @as(u32, @bitCast(alpha)), .function_handle = @intFromPtr(function.handle), .trace_floats = trace_floats, .regs = try function.attribute(.num_regs), .local_bytes = try function.attribute(.local_size_bytes), .binding_evidence = "launch request values; no graph or cluster-attribute readback" });
            const result = try payload(raw, n * 2);
            try finite(result, 2);
            try check.sameBytes("MMA instrumentation neutrality", result, expected);
            try journal.write(.{ .kind = "mma_capture_neutrality", .ordinal = gpu.ctx.device_ordinal, .projection = projection, .arm = arm, .elements = n, .sha256 = &hash(result), .bit_exact = true, .finite = true, .guards = true });
        }
        const raw_trace = try check.download(gpu, trace);
        defer gpu.gpa.free(raw_trace);
        const values = try payload(raw_trace, trace_floats * 4);
        try finite(values, 4);
        for (coords, 0..) |coordinate, ci| {
            for (0..split + 2) |stage| {
                const at = if (stage < split) ci * split + stage else coords.len * split + (stage - split) * coords.len + ci;
                const value_bits = std.mem.readInt(u32, values[at * 4 ..][0..4], .little);
                try journal.write(.{ .kind = "mma_capture_value", .ordinal = gpu.ctx.device_ordinal, .projection = projection, .coordinate = coordinate, .stage = if (stage < split) "partial" else if (stage == split) "pre_scale" else "post_scale", .slice = @as(?usize, if (stage < split) stage else null), .bits = value_bits, .value = @as(f32, @bitCast(value_bits)) });
            }
        }
        if ((try gpu.ctx.memInfo()).free < floor_bytes) return error.GpuMemoryFloor;
    }
    try journal.write(.{ .kind = "mma_capture_complete", .ordinal = gpu.ctx.device_ordinal, .projections = 2, .coordinates = coordinates_total, .required_coordinates = 3, .output_elements_per_arm = 8704, .arms = 3, .neutrality = true, .finite = true, .quality_gate = "FAIL unchanged" });
}

test "MMA capture guard and finite controls reject corrupt downloads" {
    var bytes: [guard * 2 + 4]u8 = @splat(0xa5);
    @memcpy(bytes[guard..][0..4], &[_]u8{ 0, 0, 0x80, 0x3f });
    try finite(try payload(&bytes, 4), 4);
    bytes[0] = 0;
    try std.testing.expectError(error.MmaCaptureGuard, payload(&bytes, 4));
    try std.testing.expectError(error.NonfiniteMmaCapture, finite(&.{ 0xff, 0xff, 0xff, 0xff }, 4));
}
