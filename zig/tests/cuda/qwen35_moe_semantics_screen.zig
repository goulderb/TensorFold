//! Isolated HF expert cast/order qualification from native operands.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const guard = 64;
const width = 2048;
const File = struct { path: []const u8, bytes: usize, sha256: []const u8, role: []const u8, position: usize, label: []const u8 = "", arm: []const u8 = "", stage: []const u8 = "" };
const Manifest = struct { schema: u32, geometry: struct { width: usize, slots: usize, positions: usize }, files: []const File };
const names = [_][:0]const u8{ "tf_q35_hf_combine", "tf_moe_combine_control", "tf_moe_experts_candidate", "tf_moe_experts_recorder" };

fn hash(bytes: []const u8) [64]u8 {
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &digest, .{});
    return std.fmt.bytesToHex(digest, .lower);
}
fn pin(bytes: []const u8, sha: []const u8) !void {
    if (sha.len != 64 or !std.mem.eql(u8, &hash(bytes), sha)) return error.MoeSemanticsPin;
}
fn find(m: Manifest, position: usize, role: []const u8, label: []const u8, arm: []const u8, stage: []const u8) !File {
    var found: ?File = null;
    for (m.files) |f| {
        if (f.position != position or !std.mem.eql(u8, f.role, role) or !std.mem.eql(u8, f.label, label) or !std.mem.eql(u8, f.arm, arm) or !std.mem.eql(u8, f.stage, stage)) continue;
        if (found != null) return error.MoeSemanticsDuplicate;
        found = f;
    }
    return found orelse error.MoeSemanticsMissing;
}
fn read(gpu: check.Gpu, folder: []const u8, f: File, bytes: usize) ![]u8 {
    if (!std.fs.path.isAbsolute(f.path) or f.bytes != bytes) return error.MoeSemanticsFixture;
    const path = try std.fs.path.join(gpu.gpa, &.{ folder, std.fs.path.basename(f.path) });
    defer gpu.gpa.free(path);
    const raw = try std.Io.Dir.cwd().readFileAlloc(gpu.io, path, gpu.gpa, .limited((1 << 20) + 1));
    errdefer gpu.gpa.free(raw);
    if (raw.len != bytes) return error.MoeSemanticsFixture;
    try pin(raw, f.sha256);
    return raw;
}
fn guarded(bytes: []const u8, active: usize, kind: enum { f32, bf16, i32 }) !void {
    if (bytes.len != active + 2 * guard) return error.MoeSemanticsSize;
    for (bytes[0..guard]) |v| if (v != 0xff) return error.MoeSemanticsGuard;
    for (bytes[guard + active ..]) |v| if (v != 0xff) return error.MoeSemanticsGuard;
    const size: usize = if (kind == .bf16) 2 else 4;
    if (active % size != 0) return error.MoeSemanticsSize;
    var i: usize = 0;
    while (i < active) : (i += size) {
        const p = bytes[guard + i ..];
        if (kind == .bf16 and std.mem.readInt(u16, p[0..2], .little) & 0x7f80 == 0x7f80) return error.MoeSemanticsNonfinite;
        if (kind == .f32 and std.mem.readInt(u32, p[0..4], .little) & 0x7f800000 == 0x7f800000) return error.MoeSemanticsNonfinite;
    }
}
fn save(gpu: check.Gpu, j: anytype, position: usize, label: []const u8, bytes: []const u8) !void {
    const name = try std.fmt.allocPrint(gpu.gpa, "moe_semantics_gpu{d}_pos{d}_{s}.bin", .{ gpu.ctx.device_ordinal, position, label });
    defer gpu.gpa.free(name);
    const path = try std.fs.path.join(gpu.gpa, &.{ j.directory, name });
    defer gpu.gpa.free(path);
    const file = try std.Io.Dir.cwd().createFile(gpu.io, path, .{ .exclusive = true });
    defer file.close(gpu.io);
    try file.writeStreamingAll(gpu.io, bytes);
    try j.write(.{ .kind = "moe_semantics_file", .ordinal = gpu.ctx.device_ordinal, .position = position, .label = label, .name = name, .bytes = bytes.len, .sha256 = &hash(bytes) });
}
fn validateNode(p: cuda.abi.KernelNodeParams, f: cuda.Function, want: cuda.Args) ![64]u8 {
    if (p.func != f.handle or p.grid_x != 8 or p.grid_y != 1 or p.grid_z != 1 or p.block_x != 256 or p.block_y != 1 or p.block_z != 1 or p.shared_bytes != 0) return error.MoeSemanticsGeometry;
    const actual = p.params orelse return error.MoeSemanticsArguments;
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    for (0..want.count) |i| {
        const ptr: [*]const u8 = @ptrCast(actual[i] orelse return error.MoeSemanticsArguments);
        const bytes = want.storage[want.offsets[i]..][0..want.sizes[i]];
        if (!std.mem.eql(u8, ptr[0..bytes.len], bytes)) return error.MoeSemanticsArguments;
        h.update(ptr[0..bytes.len]);
    }
    var digest: [32]u8 = undefined;
    h.final(&digest);
    return std.fmt.bytesToHex(digest, .lower);
}
fn execute(gpu: check.Gpu, stream: cuda.Stream, f: cuda.Function, a: *cuda.Args, position: usize, arm: usize, j: anytype) !void {
    try cuda.graph.beginCapture(stream, .thread_local);
    cuda.launch.launch(f, .{ .grid = .{ .x = 8, .y = 1, .z = 1 }, .block = .{ .x = 256, .y = 1, .z = 1 } }, stream, a) catch |err| {
        if (cuda.graph.endCapture(stream)) |value| {
            var dead = value;
            dead.deinit();
        } else |_| {}
        return err;
    };
    var graph = try cuda.graph.endCapture(stream);
    defer graph.deinit();
    if (try graph.nodeCount() != 1) return error.MoeSemanticsNodes;
    var storage: [1]cuda.graph.Node = undefined;
    const nodes = try graph.nodes(&storage);
    var p: cuda.abi.KernelNodeParams = undefined;
    try gpu.d.check(gpu.d.api.cuGraphKernelNodeGetParams_v2(nodes[0], &p), "cuGraphKernelNodeGetParams");
    const digest = try validateNode(p, f, a.*);
    var values: [9]u64 = undefined;
    for (0..a.count) |i| {
        const ptr: [*]const u8 = @ptrCast(p.params.?[i].?);
        values[i] = std.mem.readInt(u64, ptr[0..8], .little);
    }
    try j.write(.{ .kind = "moe_semantics_node", .ordinal = gpu.ctx.device_ordinal, .position = position, .arm = arm, .symbol = names[arm], .nodes = 1, .grid = [_]u32{ p.grid_x, p.grid_y, p.grid_z }, .block = [_]u32{ p.block_x, p.block_y, p.block_z }, .shared_bytes = p.shared_bytes, .abi_count = a.count, .argument_sizes = a.sizes[0..a.count], .arguments = values[0..a.count], .arguments_sha256 = &digest, .function_identity_checked = true });
    var exec = try graph.instantiate();
    defer exec.deinit();
    try exec.launchOn(stream);
    try stream.synchronize();
}
pub fn run(gpu: check.Gpu, folder: []const u8, manifest_sha: []const u8, image_path: []const u8, image_sha: []const u8, j: anytype) !void {
    const path = try std.fs.path.join(gpu.gpa, &.{ folder, "manifest.json" });
    defer gpu.gpa.free(path);
    const raw = try std.Io.Dir.cwd().readFileAlloc(gpu.io, path, gpu.gpa, .limited(1 << 20));
    defer gpu.gpa.free(raw);
    try pin(raw, manifest_sha);
    const parsed = try std.json.parseFromSlice(Manifest, gpu.gpa, raw, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    const m = parsed.value;
    if (m.schema != 1 or m.geometry.width != width or m.geometry.slots != 8 or m.geometry.positions != 6) return error.MoeSemanticsFixture;
    const image = try std.Io.Dir.cwd().readFileAlloc(gpu.io, image_path, gpu.gpa, .limited(16 << 20));
    defer gpu.gpa.free(image);
    try pin(image, image_sha);
    const aligned = try gpu.gpa.alignedAlloc(u8, .@"8", image.len);
    defer gpu.gpa.free(aligned);
    @memcpy(aligned, image);
    try j.write(.{ .kind = "moe_semantics_start", .ordinal = gpu.ctx.device_ordinal, .manifest_sha256 = manifest_sha, .image_sha256 = image_sha, .frozen_image_sha256 = &hash(cuda.kernels.qwen35_hf), .positions = 6, .width = width, .free_bytes = (try gpu.ctx.memInfo()).free, .expected_hf_device_inputs = false, .full_model_quality = "FAIL unchanged" });
    var original = try cuda.Module.load(gpu.d, cuda.kernels.qwen35_hf);
    defer original.unload();
    var module = try cuda.Module.load(gpu.d, aligned);
    defer module.unload();
    const functions = [_]cuda.Function{ try original.function(names[0]), try module.function(names[1]), try module.function(names[2]), try module.function(names[3]) };
    var stream = try cuda.Stream.init(gpu.d, true);
    defer stream.deinit();
    errdefer cuda.completion.drain(gpu.d, stream, "MoE semantics failure drain");
    for (functions, 0..) |f, arm| {
        const local = try f.attribute(.local_size_bytes);
        const shared = try f.attribute(.shared_size_bytes);
        try j.write(.{ .kind = "moe_semantics_resource", .ordinal = gpu.ctx.device_ordinal, .arm = arm, .symbol = names[arm], .registers = try f.attribute(.num_regs), .local_bytes = local, .static_shared = shared });
        if (local != 0 or shared != 0) return error.MoeSemanticsResources;
    }
    const input_sizes = [_]usize{ 65536, 36, 32, 4096 };
    const input_labels = [_][]const u8{ "pairs.f32", "weights.f32", "picks.i32", "shared_out.bf16" };
    const output_sizes = [_]usize{ 4096, 4096, 4096, 4096, 32, 65536, 65536, 65536, 32 };
    const output_labels = [_][]const u8{ "frozen_full", "control_full", "candidate_experts", "recorder_experts", "recorder_route", "recorder_down", "recorder_product", "recorder_prefix", "recorder_order" };
    for (0..6) |position| {
        var inputs: [4]cuda.DeviceBuffer = undefined;
        var outputs: [9]cuda.DeviceBuffer = undefined;
        var in_count: usize = 0;
        var out_count: usize = 0;
        defer for (inputs[0..in_count]) |*buffer| buffer.free();
        defer for (outputs[0..out_count]) |*buffer| buffer.free();
        for (&inputs, input_sizes, input_labels) |*buffer, bytes, label| {
            const f = try find(m, position, "device_input", label, "", "");
            const data = try read(gpu, folder, f, bytes);
            defer gpu.gpa.free(data);
            if (std.mem.eql(u8, label, "picks.i32")) try validatePicks(data);
            buffer.* = try cuda.DeviceBuffer.alloc(gpu.d, bytes + 2 * guard);
            in_count += 1;
            try buffer.fill8(0xff, null);
            try buffer.upload(guard, data);
        }
        for (&outputs, output_sizes) |*buffer, bytes| {
            buffer.* = try cuda.DeviceBuffer.alloc(gpu.d, bytes + 2 * guard);
            out_count += 1;
            try buffer.fill8(0xff, null);
        }
        for (functions, 0..) |f, arm| {
            var a: cuda.Args = .{};
            a.add(inputs[0].ptr + guard);
            a.add(inputs[1].ptr + guard);
            a.add(inputs[if (arm < 2) 3 else 2].ptr + guard);
            a.add(outputs[arm].ptr + guard);
            if (arm == 3) for (outputs[4..]) |buffer| a.add(buffer.ptr + guard);
            try execute(gpu, stream, f, &a, position, arm, j);
        }
        var mismatches: usize = 0;
        for (outputs, output_sizes, output_labels, 0..) |buffer, bytes, label, index| {
            const data = try check.download(gpu, buffer);
            defer gpu.gpa.free(data);
            try save(gpu, j, position, label, data);
            try guarded(data, bytes, if (index < 4) .bf16 else if (index == 8) .i32 else .f32);
            const stages = [_][]const u8{ "full", "full", "experts", "experts", "route", "down", "product", "prefix", "order" };
            const expected = try read(gpu, folder, try find(m, position, "host_expected", "", if (index < 2) "native" else "candidate", stages[index]), bytes);
            defer gpu.gpa.free(expected);
            if (!std.mem.eql(u8, data[guard..][0..bytes], expected)) mismatches += 1;
            if (index == 2 and position >= 2 and position <= 4) {
                const hf = try read(gpu, folder, try find(m, position, "host_original_hf", "", "", "experts"), bytes);
                defer gpu.gpa.free(hf);
                if (!std.mem.eql(u8, data[guard..][0..bytes], hf)) mismatches += 1;
            }
        }
        for (inputs, input_sizes, [_][]const u8{ "pairs_readback", "weights_readback", "picks_readback", "shared_readback" }, 0..) |buffer, bytes, label, index| {
            const data = try check.download(gpu, buffer);
            defer gpu.gpa.free(data);
            try save(gpu, j, position, label, data);
            try guarded(data, bytes, if (index == 3) .bf16 else if (index == 2) .i32 else .f32);
            const expected = try read(gpu, folder, try find(m, position, "device_input", input_labels[index], "", ""), bytes);
            defer gpu.gpa.free(expected);
            if (!std.mem.eql(u8, data[guard..][0..bytes], expected)) return error.MoeSemanticsMutation;
        }
        try j.write(.{ .kind = "moe_semantics_case", .ordinal = gpu.ctx.device_ordinal, .position = position, .files = 13, .original_hf_gate_eligible = position >= 2 and position <= 4, .mismatched_planes = mismatches, .correctness_pass = mismatches == 0, .numerical_verdict = "requires independent raw audit", .full_model_quality = "FAIL unchanged" });
        if (mismatches != 0) return error.MoeSemanticsMismatch;
    }
    try j.write(.{ .kind = "moe_semantics_complete", .ordinal = gpu.ctx.device_ordinal, .positions = 6, .raw_files = 78, .numerical_verdict = "requires independent raw audit", .free_bytes = (try gpu.ctx.memInfo()).free, .full_model_quality = "FAIL unchanged" });
}
fn validatePicks(bytes: []const u8) !void {
    if (bytes.len != 32) return error.MoeSemanticsPicks;
    var seen: [256]bool = @splat(false);
    for (0..8) |i| {
        const pick = std.mem.readInt(u32, bytes[i * 4 ..][0..4], .little);
        if (pick >= 256 or seen[pick]) return error.MoeSemanticsPicks;
        seen[pick] = true;
    }
}
test "MoE semantic input rejects duplicate and out-of-range picks before CUDA" {
    var picks: [32]u8 = @splat(0);
    for (0..8) |i| std.mem.writeInt(u32, picks[i * 4 ..][0..4], @intCast(i), .little);
    try validatePicks(&picks);
    picks[28] = 0;
    try std.testing.expectError(error.MoeSemanticsPicks, validatePicks(&picks));
    std.mem.writeInt(u32, picks[28..][0..4], 256, .little);
    try std.testing.expectError(error.MoeSemanticsPicks, validatePicks(&picks));
}
test "MoE semantic guards reject nonfinite and overwritten sentinels" {
    var bytes: [128 + 4]u8 = @splat(0xff);
    @memset(bytes[guard..][0..4], 0);
    try guarded(&bytes, 4, .f32);
    bytes[0] = 0;
    try std.testing.expectError(error.MoeSemanticsGuard, guarded(&bytes, 4, .f32));
    bytes[0] = 0xff;
    std.mem.writeInt(u32, bytes[guard..][0..4], 0x7f800000, .little);
    try std.testing.expectError(error.MoeSemanticsNonfinite, guarded(&bytes, 4, .f32));
}
test "MoE semantic graph ABI rejects each recorder argument and geometry mutation" {
    var a: cuda.Args = .{};
    for (0..9) |i| a.add(@as(u64, 0x1000 + i * 64));
    var p: cuda.abi.KernelNodeParams = std.mem.zeroes(cuda.abi.KernelNodeParams);
    p.grid_x = 8;
    p.grid_y = 1;
    p.grid_z = 1;
    p.block_x = 256;
    p.block_y = 1;
    p.block_z = 1;
    p.params = a.pointers();
    const f: cuda.Function = .{ .d = undefined, .handle = p.func };
    const want = a;
    _ = try validateNode(p, f, want);
    for (0..9) |i| {
        a.storage[a.offsets[i]] ^= 1;
        try std.testing.expectError(error.MoeSemanticsArguments, validateNode(p, f, want));
        a.storage[a.offsets[i]] ^= 1;
    }
    p.block_x = 128;
    try std.testing.expectError(error.MoeSemanticsGeometry, validateNode(p, f, want));
}
