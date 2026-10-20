//! Local isolated BF16-boundary screen for TensorFold's attributed HF activation.
//! Expected HF bytes are host comparison data, never candidate device operands.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const guard = 64;
const width = 512;
const row_bytes = width * 2;
const File = struct { name: []const u8, sha256: []const u8 };
const Case = struct { position: usize, gate: File, up: File, native: File, hf: File };
const Manifest = struct { version: u32, rows: usize, width: usize, frozen_image_sha256: []const u8, cases: [6]Case };
const names = [_][:0]const u8{ "tf_q35_hf_mlp_act", "tf_shared_activation_control", "tf_shared_activation_candidate", "tf_shared_activation_intermediates" };

fn hash(bytes: []const u8) [64]u8 {
    var d: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &d, .{});
    return std.fmt.bytesToHex(d, .lower);
}
fn pin(bytes: []const u8, expected: []const u8) !void {
    if (expected.len != 64 or !std.mem.eql(u8, &hash(bytes), expected)) return error.SharedActivationPin;
}
fn floor(gpu: check.Gpu) !usize {
    const free = (try gpu.ctx.memInfo()).free;
    if (free < (1536 << 20) + (1 << 20)) return error.SharedActivationMemoryFloor;
    return free;
}
fn finite(bytes: []const u8, element_bytes: usize) !void {
    if (bytes.len % element_bytes != 0) return error.SharedActivationSize;
    var i: usize = 0;
    while (i < bytes.len) : (i += element_bytes) {
        const nonfinite = if (element_bytes == 2)
            std.mem.readInt(u16, bytes[i..][0..2], .little) & 0x7f80 == 0x7f80
        else
            std.mem.readInt(u32, bytes[i..][0..4], .little) & 0x7f800000 == 0x7f800000;
        if (nonfinite) return error.SharedActivationNonfinite;
    }
}
fn guarded(bytes: []const u8, active: usize, element_bytes: usize) !void {
    if (bytes.len != active + 2 * guard) return error.SharedActivationSize;
    for (bytes[0..guard]) |v| if (v != 0xff) return error.SharedActivationGuard;
    for (bytes[guard + active ..]) |v| if (v != 0xff) return error.SharedActivationGuard;
    try finite(bytes[guard..][0..active], element_bytes);
}
fn differences(actual: []const u8, expected: []const u8) !usize {
    if (actual.len != expected.len or actual.len % 2 != 0) return error.SharedActivationSize;
    var count: usize = 0;
    var i: usize = 0;
    while (i < actual.len) : (i += 2) {
        if (!std.mem.eql(u8, actual[i..][0..2], expected[i..][0..2])) count += 1;
    }
    return count;
}
fn read(gpu: check.Gpu, folder: []const u8, file: File) ![]u8 {
    if (!std.mem.eql(u8, std.fs.path.basename(file.name), file.name)) return error.SharedActivationPath;
    const path = try std.fs.path.join(gpu.gpa, &.{ folder, file.name });
    defer gpu.gpa.free(path);
    const bytes = try std.Io.Dir.cwd().readFileAlloc(gpu.io, path, gpu.gpa, .limited(1 << 20));
    errdefer gpu.gpa.free(bytes);
    try pin(bytes, file.sha256);
    if (bytes.len != row_bytes) return error.SharedActivationSize;
    try finite(bytes, 2);
    return bytes;
}
fn save(gpu: check.Gpu, j: anytype, position: usize, label: []const u8, bytes: []const u8) !void {
    const name = try std.fmt.allocPrint(gpu.gpa, "shared_activation_gpu{d}_pos{d}_{s}.bin", .{ gpu.ctx.device_ordinal, position, label });
    defer gpu.gpa.free(name);
    const path = try std.fs.path.join(gpu.gpa, &.{ j.directory, name });
    defer gpu.gpa.free(path);
    const file = try std.Io.Dir.cwd().createFile(gpu.io, path, .{ .exclusive = true });
    defer file.close(gpu.io);
    try file.writeStreamingAll(gpu.io, bytes);
    try j.write(.{ .kind = "shared_activation_file", .ordinal = gpu.ctx.device_ordinal, .position = position, .label = label, .name = name, .bytes = bytes.len, .sha256 = &hash(bytes) });
}
fn args(gate: u64, up: u64, output: u64) cuda.Args {
    var a: cuda.Args = .{};
    a.add(gate);
    a.add(up);
    a.add(output);
    a.add(@as(i64, width));
    a.add(@as(u64, 0));
    a.add(@as(i32, width));
    return a;
}
fn validateNode(p: cuda.abi.KernelNodeParams, f: cuda.Function, want: cuda.Args) ![64]u8 {
    if (p.func != f.handle or p.grid_x != 2 or p.grid_y != 1 or p.grid_z != 1 or p.block_x != 256 or p.block_y != 1 or p.block_z != 1 or p.shared_bytes != 0) return error.SharedActivationGeometry;
    const actual = p.params orelse return error.SharedActivationArguments;
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    for (0..want.count) |n| {
        const ptr: [*]const u8 = @ptrCast(actual[n] orelse return error.SharedActivationArguments);
        const bytes = want.storage[want.offsets[n]..][0..want.sizes[n]];
        if (!std.mem.eql(u8, ptr[0..bytes.len], bytes)) return error.SharedActivationArguments;
        h.update(ptr[0..bytes.len]);
    }
    var digest: [32]u8 = undefined;
    h.final(&digest);
    return std.fmt.bytesToHex(digest, .lower);
}
fn execute(gpu: check.Gpu, stream: cuda.Stream, f: cuda.Function, a: *cuda.Args, position: usize, arm: usize, j: anytype) !void {
    try cuda.graph.beginCapture(stream, .thread_local);
    cuda.launch.launch(f, .{ .grid = .{ .x = 2, .y = 1, .z = 1 }, .block = .{ .x = 256, .y = 1, .z = 1 } }, stream, a) catch |err| {
        if (cuda.graph.endCapture(stream)) |value| {
            var dead = value;
            dead.deinit();
        } else |_| {}
        return err;
    };
    var graph = try cuda.graph.endCapture(stream);
    defer graph.deinit();
    if (try graph.nodeCount() != 1) return error.SharedActivationNodes;
    var storage: [1]cuda.graph.Node = undefined;
    const nodes = try graph.nodes(&storage);
    var p: cuda.abi.KernelNodeParams = undefined;
    try gpu.d.check(gpu.d.api.cuGraphKernelNodeGetParams_v2(nodes[0], &p), "cuGraphKernelNodeGetParams");
    const digest = try validateNode(p, f, a.*);
    try j.write(.{ .kind = "shared_activation_node", .ordinal = gpu.ctx.device_ordinal, .position = position, .arm = arm, .symbol = names[arm], .nodes = 1, .grid = [_]u32{ p.grid_x, p.grid_y, p.grid_z }, .block = [_]u32{ p.block_x, p.block_y, p.block_z }, .shared_bytes = p.shared_bytes, .abi_count = a.count, .argument_sizes = a.sizes[0..a.count], .arguments_sha256 = &digest, .function_identity_checked = true });
    var exec = try graph.instantiate();
    defer exec.deinit();
    try exec.launchOn(stream);
    try stream.synchronize();
}
pub fn run(gpu: check.Gpu, fixture_dir: []const u8, manifest_sha256: []const u8, image_path: []const u8, image_sha256: []const u8, j: anytype) !void {
    return runInner(gpu, fixture_dir, manifest_sha256, image_path, image_sha256, j) catch |err| {
        j.write(.{ .kind = "shared_activation_failure", .ordinal = gpu.ctx.device_ordinal, .failure = @errorName(err), .correctness_pass = false, .full_model_quality = "FAIL unchanged" }) catch {};
        return err;
    };
}
fn runInner(gpu: check.Gpu, fixture_dir: []const u8, manifest_sha256: []const u8, image_path: []const u8, image_sha256: []const u8, j: anytype) !void {
    const path = try std.fs.path.join(gpu.gpa, &.{ fixture_dir, "manifest.json" });
    defer gpu.gpa.free(path);
    const raw = try std.Io.Dir.cwd().readFileAlloc(gpu.io, path, gpu.gpa, .limited(1 << 20));
    defer gpu.gpa.free(raw);
    try pin(raw, manifest_sha256);
    const parsed = try std.json.parseFromSlice(Manifest, gpu.gpa, raw, .{});
    defer parsed.deinit();
    const m = parsed.value;
    if (m.version != 1 or m.rows != 6 or m.width != width) return error.SharedActivationManifest;
    try pin(cuda.kernels.qwen35_hf, m.frozen_image_sha256);
    const image = try std.Io.Dir.cwd().readFileAlloc(gpu.io, image_path, gpu.gpa, .limited(16 << 20));
    defer gpu.gpa.free(image);
    try pin(image, image_sha256);
    const aligned = try gpu.gpa.alignedAlloc(u8, .@"8", image.len);
    defer gpu.gpa.free(aligned);
    @memcpy(aligned, image);
    try j.write(.{ .kind = "shared_activation_start", .ordinal = gpu.ctx.device_ordinal, .manifest_sha256 = manifest_sha256, .image_sha256 = image_sha256, .frozen_image_sha256 = m.frozen_image_sha256, .rows = m.rows, .width = width, .free_bytes = try floor(gpu), .scope = "quality-only isolated activation; no performance or full-model claim" });
    var original = try cuda.Module.load(gpu.d, cuda.kernels.qwen35_hf);
    defer original.unload();
    var module = try cuda.Module.load(gpu.d, aligned);
    defer module.unload();
    const functions = [_]cuda.Function{ try original.function(names[0]), try module.function(names[1]), try module.function(names[2]), try module.function(names[3]) };
    for (functions, 0..) |f, arm| {
        const local = try f.attribute(.local_size_bytes);
        const shared = try f.attribute(.shared_size_bytes);
        try j.write(.{ .kind = "shared_activation_resource", .ordinal = gpu.ctx.device_ordinal, .arm = arm, .symbol = names[arm], .registers = try f.attribute(.num_regs), .local_bytes = local, .static_shared = shared });
        if (local != 0 or shared != 0) return error.SharedActivationResources;
    }
    var stream = try cuda.Stream.init(gpu.d, true);
    defer stream.deinit();
    var gate = try cuda.DeviceBuffer.alloc(gpu.d, row_bytes + 2 * guard);
    defer gate.free();
    var up = try cuda.DeviceBuffer.alloc(gpu.d, row_bytes + 2 * guard);
    defer up.free();
    var out = try cuda.DeviceBuffer.alloc(gpu.d, row_bytes + 2 * guard);
    defer out.free();
    var fp = try cuda.DeviceBuffer.alloc(gpu.d, width * 6 * 4 + 2 * guard);
    defer fp.free();
    var bf = try cuda.DeviceBuffer.alloc(gpu.d, width * 3 * 2 + 2 * guard);
    defer bf.free();
    var mismatches: usize = 0;
    for (m.cases, 0..) |c, position| {
        if (c.position != position) return error.SharedActivationManifest;
        const g = try read(gpu, fixture_dir, c.gate);
        defer gpu.gpa.free(g);
        const u = try read(gpu, fixture_dir, c.up);
        defer gpu.gpa.free(u);
        const native = try read(gpu, fixture_dir, c.native);
        defer gpu.gpa.free(native);
        const hf = try read(gpu, fixture_dir, c.hf);
        defer gpu.gpa.free(hf);
        try gate.fill8(0xff, stream.handle);
        try up.fill8(0xff, stream.handle);
        try stream.synchronize();
        try gate.upload(guard, g);
        try up.upload(guard, u);
        var outputs: [3][row_bytes]u8 = undefined;
        for (functions[0..3], 0..) |f, arm| {
            try out.fill8(0xff, stream.handle);
            var a = args(gate.ptr + guard, up.ptr + guard, out.ptr + guard);
            try execute(gpu, stream, f, &a, position, arm, j);
            const bytes = try check.download(gpu, out);
            defer gpu.gpa.free(bytes);
            try save(gpu, j, position, names[arm], bytes);
            try guarded(bytes, row_bytes, 2);
            @memcpy(&outputs[arm], bytes[guard..][0..row_bytes]);
        }
        try fp.fill8(0xff, stream.handle);
        try bf.fill8(0xff, stream.handle);
        var a: cuda.Args = .{};
        a.add(gate.ptr + guard);
        a.add(up.ptr + guard);
        a.add(fp.ptr + guard);
        a.add(bf.ptr + guard);
        a.add(@as(i64, width));
        try execute(gpu, stream, functions[3], &a, position, 3, j);
        const floats = try check.download(gpu, fp);
        defer gpu.gpa.free(floats);
        const rounded = try check.download(gpu, bf);
        defer gpu.gpa.free(rounded);
        try save(gpu, j, position, "intermediates_f32", floats);
        try save(gpu, j, position, "intermediates_bf16", rounded);
        try guarded(floats, width * 6 * 4, 4);
        try guarded(rounded, width * 3 * 2, 2);
        for ([_]cuda.DeviceBuffer{ gate, up }, [_][]const u8{ g, u }, [_][]const u8{ "gate_readback", "up_readback" }) |buffer, expected, label| {
            const bytes = try check.download(gpu, buffer);
            defer gpu.gpa.free(bytes);
            try save(gpu, j, position, label, bytes);
            try guarded(bytes, row_bytes, 2);
            if (!std.mem.eql(u8, bytes[guard..][0..row_bytes], expected)) return error.SharedActivationInputMutation;
        }
        const frozen_diff = try differences(&outputs[0], native);
        const control_diff = try differences(&outputs[1], native);
        const candidate_diff = try differences(&outputs[2], hf);
        const diag_control = try differences(rounded[guard + row_bytes ..][0..row_bytes], &outputs[1]);
        const diag_candidate = try differences(rounded[guard + 2 * row_bytes ..][0..row_bytes], &outputs[2]);
        try j.write(.{ .kind = "shared_activation_case", .ordinal = gpu.ctx.device_ordinal, .position = position, .gate_sha256 = c.gate.sha256, .up_sha256 = c.up.sha256, .native_sha256 = c.native.sha256, .hf_sha256 = c.hf.sha256, .frozen_native_differences = frozen_diff, .control_native_differences = control_diff, .candidate_hf_differences = candidate_diff, .diagnostic_control_differences = diag_control, .diagnostic_candidate_differences = diag_candidate, .finite_and_guards = true });
        if (frozen_diff != 0 or control_diff != 0 or diag_control != 0 or diag_candidate != 0) return error.SharedActivationControlMismatch;
        mismatches += candidate_diff;
    }
    try j.write(.{ .kind = "shared_activation_complete", .ordinal = gpu.ctx.device_ordinal, .cases = 6, .candidate_hf_differences = mismatches, .correctness_pass = mismatches == 0, .free_bytes = try floor(gpu), .full_model_quality = "FAIL unchanged" });
    if (mismatches != 0) return error.SharedActivationCandidateMismatch;
}

test "shared activation guards finiteness and exact comparison reject corruption" {
    var bytes: [row_bytes + 2 * guard]u8 = @splat(0xff);
    @memset(bytes[guard..][0..row_bytes], 0);
    try guarded(&bytes, row_bytes, 2);
    bytes[0] = 0;
    try std.testing.expectError(error.SharedActivationGuard, guarded(&bytes, row_bytes, 2));
    bytes[0] = 0xff;
    bytes[guard + 1] = 0x7f;
    bytes[guard] = 0x80;
    try std.testing.expectError(error.SharedActivationNonfinite, guarded(&bytes, row_bytes, 2));
    try std.testing.expectEqual(@as(usize, 1), try differences(&.{ 0, 0 }, &.{ 1, 0 }));
    try std.testing.expectError(error.SharedActivationPin, pin("x", &hash("y")));
}
test "shared activation actual six argument ABI rejects mutation" {
    var a = args(0x1000, 0x2000, 0x3000);
    try std.testing.expectEqual(@as(usize, 6), a.count);
    try std.testing.expectEqualSlices(u16, &.{ 8, 8, 8, 8, 8, 4 }, a.sizes[0..a.count]);
    var p: cuda.abi.KernelNodeParams = std.mem.zeroes(cuda.abi.KernelNodeParams);
    p.grid_x = 2;
    p.grid_y = 1;
    p.grid_z = 1;
    p.block_x = 256;
    p.block_y = 1;
    p.block_z = 1;
    p.params = a.pointers();
    const f: cuda.Function = .{ .d = undefined, .handle = p.func };
    const want = a;
    _ = try validateNode(p, f, want);
    for (0..6) |i| {
        a.storage[a.offsets[i]] ^= 1;
        try std.testing.expectError(error.SharedActivationArguments, validateNode(p, f, want));
        a.storage[a.offsets[i]] ^= 1;
    }
    p.block_x = 128;
    try std.testing.expectError(error.SharedActivationGeometry, validateNode(p, f, want));
}
