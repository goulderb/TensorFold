//! Isolated composition of the attributed HF activation and unchanged NVFP4 shared down.
//! Captured HF tensors are host comparison targets and never device inputs.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const qwen = @import("qwen35_cuda");
const check = @import("check.zig");
const guard = 64;
const width = 512;
const outputs = 2048;
const File = struct { path: []const u8, bytes: usize, sha256: []const u8, role: []const u8 };
const Case = struct { position: usize, gate: File, up: File, native_activation: File, native_down: File, hf_activation: File, hf_down: File };
const Manifest = struct {
    schema: u32,
    origin_manifest: File,
    checkpoint_file: File,
    checkpoint_index: File,
    frozen_image_sha256: []const u8,
    activation_image_sha256: []const u8,
    cases: [6]Case,
};
const names = [_][:0]const u8{ "tf_q35_hf_mlp_act", "tf_shared_activation_control", "tf_shared_activation_candidate" };

fn hash(bytes: []const u8) [64]u8 {
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &digest, .{});
    return std.fmt.bytesToHex(digest, .lower);
}
fn pin(bytes: []const u8, sha: []const u8) !void {
    if (sha.len != 64 or !std.mem.eql(u8, &hash(bytes), sha)) return error.SharedBranchPin;
}
fn read(gpu: check.Gpu, f: File, role: []const u8, size: usize) ![]u8 {
    if (!std.fs.path.isAbsolute(f.path) or f.bytes != size or !std.mem.eql(u8, f.role, role)) return error.SharedBranchRole;
    const bytes = try std.Io.Dir.cwd().readFileAlloc(gpu.io, f.path, gpu.gpa, .limited(4 << 20));
    errdefer gpu.gpa.free(bytes);
    if (bytes.len != size) return error.SharedBranchSize;
    try pin(bytes, f.sha256);
    return bytes;
}
fn finite(bytes: []const u8) !void {
    if (bytes.len % 2 != 0) return error.SharedBranchSize;
    var i: usize = 0;
    while (i < bytes.len) : (i += 2) {
        if (std.mem.readInt(u16, bytes[i..][0..2], .little) & 0x7f80 == 0x7f80) return error.SharedBranchNonfinite;
    }
}
fn guarded(bytes: []const u8, active: usize, numeric: bool) !void {
    if (bytes.len != active + 2 * guard) return error.SharedBranchSize;
    for (bytes[0..guard]) |v| if (v != 0xff) return error.SharedBranchGuard;
    for (bytes[guard + active ..]) |v| if (v != 0xff) return error.SharedBranchGuard;
    if (numeric) try finite(bytes[guard..][0..active]);
}
fn differences(a: []const u8, b: []const u8) !usize {
    if (a.len != b.len or a.len % 2 != 0) return error.SharedBranchSize;
    var count: usize = 0;
    var i: usize = 0;
    while (i < a.len) : (i += 2) {
        if (!std.mem.eql(u8, a[i..][0..2], b[i..][0..2])) count += 1;
    }
    return count;
}
fn save(gpu: check.Gpu, j: anytype, position: usize, label: []const u8, bytes: []const u8) !void {
    const name = try std.fmt.allocPrint(gpu.gpa, "shared_branch_gpu{d}_pos{d}_{s}.bin", .{ gpu.ctx.device_ordinal, position, label });
    defer gpu.gpa.free(name);
    const path = try std.fs.path.join(gpu.gpa, &.{ j.directory, name });
    defer gpu.gpa.free(path);
    const file = try std.Io.Dir.cwd().createFile(gpu.io, path, .{ .exclusive = true });
    defer file.close(gpu.io);
    try file.writeStreamingAll(gpu.io, bytes);
    try j.write(.{ .kind = "shared_branch_file", .ordinal = gpu.ctx.device_ordinal, .position = position, .label = label, .name = name, .bytes = bytes.len, .sha256 = &hash(bytes) });
}
fn actArgs(gate: u64, up: u64, out: u64) cuda.Args {
    var a: cuda.Args = .{};
    a.add(gate);
    a.add(up);
    a.add(out);
    a.add(@as(i64, width));
    a.add(@as(u64, 0));
    a.add(@as(i32, width));
    return a;
}
fn downArgs(x: u64, w: cuda.qmmf.Weight, out: u64) cuda.Args {
    var a: cuda.Args = .{};
    a.add(x);
    a.add(w.codes);
    a.add(w.scales);
    a.add(w.scale);
    a.add(out);
    a.add(@as(u64, 0));
    inline for (.{ 1, outputs, width, 1, outputs, width, 1 }) |v| a.add(@as(i32, v));
    return a;
}
fn validateNode(p: cuda.abi.KernelNodeParams, f: cuda.Function, want: cuda.Args, down: bool) ![64]u8 {
    const grid: u32 = if (down) 32 else 2;
    const block: u32 = if (down) 128 else 256;
    const shared: u32 = if (down) 17408 else 0;
    if (p.func != f.handle or p.grid_x != grid or p.grid_y != 1 or p.grid_z != 1 or p.block_x != block or p.block_y != 1 or p.block_z != 1 or p.shared_bytes != shared) return error.SharedBranchGeometry;
    if (want.count != (if (down) @as(usize, 13) else 6)) return error.SharedBranchArguments;
    const actual = p.params orelse return error.SharedBranchArguments;
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    for (0..want.count) |i| {
        const ptr: [*]const u8 = @ptrCast(actual[i] orelse return error.SharedBranchArguments);
        const bytes = want.storage[want.offsets[i]..][0..want.sizes[i]];
        if (!std.mem.eql(u8, ptr[0..bytes.len], bytes)) return error.SharedBranchArguments;
        h.update(ptr[0..bytes.len]);
    }
    var digest: [32]u8 = undefined;
    h.final(&digest);
    return std.fmt.bytesToHex(digest, .lower);
}
fn inspectLaunch(gpu: check.Gpu, stream: cuda.Stream, graph: cuda.graph.Graph, f: cuda.Function, a: cuda.Args, position: usize, arm: usize, down: bool, j: anytype) !void {
    if (try graph.nodeCount() != 1) return error.SharedBranchNodes;
    var storage: [1]cuda.graph.Node = undefined;
    const nodes = try graph.nodes(&storage);
    var p: cuda.abi.KernelNodeParams = undefined;
    try gpu.d.check(gpu.d.api.cuGraphKernelNodeGetParams_v2(nodes[0], &p), "cuGraphKernelNodeGetParams");
    const digest = try validateNode(p, f, a, down);
    var values: [13]u64 = @splat(0);
    for (0..a.count) |i| {
        const ptr: [*]const u8 = @ptrCast(p.params.?[i].?);
        for (ptr[0..a.sizes[i]], 0..) |v, shift| values[i] |= @as(u64, v) << @intCast(8 * shift);
    }
    try j.write(.{ .kind = "shared_branch_node", .ordinal = gpu.ctx.device_ordinal, .position = position, .arm = arm, .stage = if (down) "down" else "activation", .symbol = if (down) "qmmf fp4 BM16 SK1 BF16" else names[arm], .nodes = 1, .grid = [_]u32{ p.grid_x, p.grid_y, p.grid_z }, .block = [_]u32{ p.block_x, p.block_y, p.block_z }, .shared_bytes = p.shared_bytes, .abi_count = a.count, .argument_sizes = a.sizes[0..a.count], .arguments = values[0..a.count], .arguments_sha256 = &digest, .function_identity_checked = true });
    var exec = try graph.instantiate();
    defer exec.deinit();
    try exec.launchOn(stream);
    try stream.synchronize();
}
fn abortCapture(stream: cuda.Stream) void {
    if (cuda.graph.endCapture(stream)) |value| {
        var graph = value;
        graph.deinit();
    } else |_| {}
}
fn activation(gpu: check.Gpu, stream: cuda.Stream, f: cuda.Function, a: *cuda.Args, position: usize, arm: usize, j: anytype) !void {
    try cuda.graph.beginCapture(stream, .thread_local);
    cuda.launch.launch(f, .{ .grid = .{ .x = 2 }, .block = .{ .x = 256 } }, stream, a) catch |err| {
        abortCapture(stream);
        return err;
    };
    var graph = try cuda.graph.endCapture(stream);
    defer graph.deinit();
    try inspectLaunch(gpu, stream, graph, f, a.*, position, arm, false, j);
}
fn projection(gpu: check.Gpu, stream: cuda.Stream, lane: *cuda.qmmf.Lane, x: u64, w: cuda.qmmf.Weight, out: u64, position: usize, arm: usize, j: anytype) !void {
    const linear: cuda.qlinear.Linear = .{ .lane = lane };
    try cuda.graph.beginCapture(stream, .thread_local);
    linear.decode(stream, .{ .nvfp4 = w }, .{ .x = x }, out, 1) catch |err| {
        abortCapture(stream);
        return err;
    };
    var graph = try cuda.graph.endCapture(stream);
    defer graph.deinit();
    try inspectLaunch(gpu, stream, graph, lane.fns[0][0], downArgs(x, w, out), position, arm, true, j);
}
pub fn run(gpu: check.Gpu, folder: []const u8, manifest_sha: []const u8, image_path: []const u8, image_sha: []const u8, j: anytype) !void {
    return runInner(gpu, folder, manifest_sha, image_path, image_sha, j) catch |err| {
        j.write(.{ .kind = "shared_branch_failure", .ordinal = gpu.ctx.device_ordinal, .failure = @errorName(err), .full_model_quality = "FAIL unchanged" }) catch {};
        return err;
    };
}
fn runInner(gpu: check.Gpu, folder: []const u8, manifest_sha: []const u8, image_path: []const u8, image_sha: []const u8, j: anytype) !void {
    const path = try std.fs.path.join(gpu.gpa, &.{ folder, "manifest.json" });
    defer gpu.gpa.free(path);
    const raw = try std.Io.Dir.cwd().readFileAlloc(gpu.io, path, gpu.gpa, .limited(1 << 20));
    defer gpu.gpa.free(raw);
    try pin(raw, manifest_sha);
    const parsed = try std.json.parseFromSlice(Manifest, gpu.gpa, raw, .{});
    defer parsed.deinit();
    const m = parsed.value;
    if (m.schema != 1 or !std.mem.eql(u8, m.activation_image_sha256, image_sha)) return error.SharedBranchManifest;
    try pin(cuda.kernels.qwen35_hf, m.frozen_image_sha256);
    const origin = try read(gpu, m.origin_manifest, "origin_manifest", m.origin_manifest.bytes);
    defer gpu.gpa.free(origin);
    const ck_bytes = try read(gpu, m.checkpoint_file, "checkpoint", 1643156);
    defer gpu.gpa.free(ck_bytes);
    const ck_index = try read(gpu, m.checkpoint_index, "checkpoint", 527);
    defer gpu.gpa.free(ck_index);
    const directory = std.fs.path.dirname(m.checkpoint_file.path) orelse return error.SharedBranchManifest;
    if (!std.mem.eql(u8, std.fs.path.basename(m.checkpoint_file.path), "model.safetensors") or !std.mem.eql(u8, std.fs.path.basename(m.checkpoint_index.path), "model.safetensors.index.json") or !std.mem.eql(u8, directory, std.fs.path.dirname(m.checkpoint_index.path) orelse return error.SharedBranchManifest)) return error.SharedBranchManifest;
    const image = try std.Io.Dir.cwd().readFileAlloc(gpu.io, image_path, gpu.gpa, .limited(16 << 20));
    defer gpu.gpa.free(image);
    try pin(image, image_sha);
    const aligned = try gpu.gpa.alignedAlloc(u8, .@"8", image.len);
    defer gpu.gpa.free(aligned);
    @memcpy(aligned, image);
    try j.write(.{ .kind = "shared_branch_start", .ordinal = gpu.ctx.device_ordinal, .manifest_sha256 = manifest_sha, .origin_manifest_sha256 = m.origin_manifest.sha256, .image_sha256 = image_sha, .frozen_image_sha256 = m.frozen_image_sha256, .checkpoint_sha256 = m.checkpoint_file.sha256, .checkpoint_index_sha256 = m.checkpoint_index.sha256, .rows = 6, .activation_width = width, .output_width = outputs, .expected_hf_device_inputs = false, .free_bytes = (try gpu.ctx.memInfo()).free, .scope = "isolated activation/down composition, no timing, gate or final MoE claim" });
    var ck = try core.Checkpoint.openModel(gpu.gpa, gpu.io, directory);
    defer ck.close();
    const part = try qwen.checkpoint.readPart(&ck, "model.language_model.layers.0.mlp.shared_expert.down_proj", outputs, width);
    if (part.kind != .nvfp4 or part.w.bytes.len != 524288 or part.blocks.?.bytes.len != 65536 or part.act != 0 or !(part.scale > 0) or !std.math.isFinite(part.scale)) return error.SharedBranchFormat;
    var stream = try cuda.Stream.init(gpu.d, true);
    defer stream.deinit();
    var packer = try cuda.modelopt.Packer.load(gpu.d);
    defer packer.unload();
    var lane = try cuda.qmmf.Lane.load(gpu.d, @intCast(try gpu.ctx.attribute(.compute_capability_major)));
    defer lane.unload();
    const sizes = [_]usize{ 589824, 524288, 65536 };
    var buffers: [3]cuda.DeviceBuffer = undefined;
    var count: usize = 0;
    defer for (buffers[0..count]) |*buffer| buffer.free();
    for (&buffers, sizes) |*buffer, size| {
        buffer.* = try cuda.DeviceBuffer.alloc(gpu.d, size + 2 * guard);
        count += 1;
        try buffer.fill8(0xff, null);
    }
    try buffers[0].upload(guard, part.w.bytes);
    try buffers[0].upload(guard + part.w.bytes.len, part.blocks.?.bytes);
    try packer.fp4Lane(stream, buffers[0].ptr + guard, buffers[0].ptr + guard + part.w.bytes.len, buffers[1].ptr + guard, buffers[2].ptr + guard, outputs, width);
    try stream.synchronize();
    const words = try gpu.gpa.alloc(u32, 524288 / 4);
    defer gpu.gpa.free(words);
    const scales = try gpu.gpa.alloc(u8, 65536);
    defer gpu.gpa.free(scales);
    cuda.nvfp4.packWords(words, part.w.bytes, outputs, width);
    cuda.nvfp4.packScales(scales, part.blocks.?.bytes, outputs, width);
    var before: [3][]u8 = undefined;
    var before_count: usize = 0;
    defer for (before[0..before_count]) |data| gpu.gpa.free(data);
    for (buffers, [_][]const u8{ "checkpoint_staging", "packed_words", "packed_scales" }, 0..) |buffer, label, index| {
        before[index] = try check.download(gpu, buffer);
        before_count += 1;
        try save(gpu, j, 999, label, before[index]);
    }
    for (before, sizes, 0..) |data, size, index| {
        try guarded(data, size, false);
        const payload = data[guard..][0..size];
        const exact = switch (index) {
            0 => std.mem.eql(u8, payload[0..524288], part.w.bytes) and std.mem.eql(u8, payload[524288..], part.blocks.?.bytes),
            1 => std.mem.eql(u8, payload, std.mem.sliceAsBytes(words)),
            2 => std.mem.eql(u8, payload, scales),
            else => unreachable,
        };
        if (!exact) return error.SharedBranchPackedBytes;
    }
    const w = cuda.nvfp4.weight(buffers[1].ptr + guard, buffers[2].ptr + guard, part.scale, outputs, width);
    const plan = cuda.qmmf.plan(1, outputs, width, lane.clusters);
    if (plan.bm != 16 or plan.sk != 1 or plan.fused or plan.cluster or plan.reduce) return error.SharedBranchPlan;
    try j.write(.{ .kind = "shared_branch_packing", .ordinal = gpu.ctx.device_ordinal, .cpu_byte_proof = true, .n = outputs, .k = width, .bm = plan.bm, .sk = plan.sk, .global_scale_bits = @as(u32, @bitCast(w.scale)), .words = w.codes, .scales = w.scales, .no_partials_or_cluster = true, .qmmf_image_sha256 = &hash(cuda.kernels.qmmf), .packer_image_sha256 = &hash(cuda.kernels.modelopt_pack) });
    var original = try cuda.Module.load(gpu.d, cuda.kernels.qwen35_hf);
    defer original.unload();
    var module = try cuda.Module.load(gpu.d, aligned);
    defer module.unload();
    const fns = [_]cuda.Function{ try original.function(names[0]), try module.function(names[1]), try module.function(names[2]) };
    for (m.cases, 0..) |c, position| {
        if (c.position != position) return error.SharedBranchManifest;
        const gate = try read(gpu, c.gate, "device_native_input", 1024);
        defer gpu.gpa.free(gate);
        const up = try read(gpu, c.up, "device_native_input", 1024);
        defer gpu.gpa.free(up);
        try finite(gate);
        try finite(up);
        var inputs: [2]cuda.DeviceBuffer = undefined;
        var acts: [3]cuda.DeviceBuffer = undefined;
        var downs: [3]cuda.DeviceBuffer = undefined;
        var ni: usize = 0;
        var na: usize = 0;
        var nd: usize = 0;
        defer for (inputs[0..ni]) |*buffer| buffer.free();
        defer for (acts[0..na]) |*buffer| buffer.free();
        defer for (downs[0..nd]) |*buffer| buffer.free();
        for (&inputs, [_][]const u8{ gate, up }) |*buffer, payload| {
            buffer.* = try cuda.DeviceBuffer.alloc(gpu.d, 1024 + 2 * guard);
            ni += 1;
            try buffer.fill8(0xff, null);
            try buffer.upload(guard, payload);
        }
        for (&acts) |*buffer| {
            buffer.* = try cuda.DeviceBuffer.alloc(gpu.d, 1024 + 2 * guard);
            na += 1;
            try buffer.fill8(0xff, null);
        }
        for (&downs) |*buffer| {
            buffer.* = try cuda.DeviceBuffer.alloc(gpu.d, 4096 + 2 * guard);
            nd += 1;
            try buffer.fill8(0xff, null);
        }
        for (fns, 0..) |f, arm| {
            var a = actArgs(inputs[0].ptr + guard, inputs[1].ptr + guard, acts[arm].ptr + guard);
            try activation(gpu, stream, f, &a, position, arm, j);
            try projection(gpu, stream, &lane, acts[arm].ptr + guard, w, downs[arm].ptr + guard, position, arm, j);
        }
        var act_rows: [3][1152]u8 = undefined;
        var down_rows: [3][4224]u8 = undefined;
        for (acts, downs, [_][]const u8{ "frozen", "control", "candidate" }, 0..) |act, down, label, arm| {
            const act_data = try check.download(gpu, act);
            defer gpu.gpa.free(act_data);
            const down_data = try check.download(gpu, down);
            defer gpu.gpa.free(down_data);
            const act_label = try std.fmt.allocPrint(gpu.gpa, "{s}_activation", .{label});
            defer gpu.gpa.free(act_label);
            const down_label = try std.fmt.allocPrint(gpu.gpa, "{s}_down", .{label});
            defer gpu.gpa.free(down_label);
            try save(gpu, j, position, act_label, act_data);
            try save(gpu, j, position, down_label, down_data);
            @memcpy(&act_rows[arm], act_data);
            @memcpy(&down_rows[arm], down_data);
        }
        var input_rows: [2][1152]u8 = undefined;
        for (inputs, [_][]const u8{ "gate_readback", "up_readback" }, 0..) |buffer, label, index| {
            const data = try check.download(gpu, buffer);
            defer gpu.gpa.free(data);
            try save(gpu, j, position, label, data);
            @memcpy(&input_rows[index], data);
        }
        var mismatches: usize = 0;
        for (&act_rows, &down_rows, 0..) |act_data, down_data, arm| {
            try guarded(&act_data, 1024, true);
            try guarded(&down_data, 4096, true);
            const role = if (arm == 2) "host_original_hf" else "host_native_control";
            const expected_act = try read(gpu, if (arm == 2) c.hf_activation else c.native_activation, role, 1024);
            defer gpu.gpa.free(expected_act);
            const expected_down = try read(gpu, if (arm == 2) c.hf_down else c.native_down, role, 4096);
            defer gpu.gpa.free(expected_down);
            const act_diff = try differences(act_data[guard..][0..1024], expected_act);
            const down_diff = try differences(down_data[guard..][0..4096], expected_down);
            mismatches += act_diff + down_diff;
            try j.write(.{ .kind = "shared_branch_arm", .ordinal = gpu.ctx.device_ordinal, .position = position, .arm = arm, .activation_differences = act_diff, .down_differences = down_diff, .activation_sha256 = &hash(act_data[guard..][0..1024]), .down_sha256 = &hash(down_data[guard..][0..4096]), .target_role = role, .activation_target_sha256 = (if (arm == 2) c.hf_activation else c.native_activation).sha256, .down_target_sha256 = (if (arm == 2) c.hf_down else c.native_down).sha256, .finite_and_guards = true });
        }
        for (&input_rows, [_][]const u8{ gate, up }) |data, expected| {
            try guarded(&data, 1024, true);
            if (!std.mem.eql(u8, data[guard..][0..1024], expected)) return error.SharedBranchInputMutation;
        }
        try j.write(.{ .kind = "shared_branch_case", .ordinal = gpu.ctx.device_ordinal, .position = position, .mismatches = mismatches, .raw_files = 8, .actual_hf_activation_and_down_eligible = true, .correctness_pass = mismatches == 0 });
        if (mismatches != 0) return error.SharedBranchMismatch;
    }
    var after: [3][]u8 = undefined;
    var after_count: usize = 0;
    defer for (after[0..after_count]) |data| gpu.gpa.free(data);
    for (buffers, [_][]const u8{ "checkpoint_staging_after", "packed_words_after", "packed_scales_after" }, 0..) |buffer, label, index| {
        after[index] = try check.download(gpu, buffer);
        after_count += 1;
        try save(gpu, j, 999, label, after[index]);
    }
    for (after, sizes, 0..) |data, size, index| {
        try guarded(data, size, false);
        const expected: []const u8 = if (index == 1) std.mem.sliceAsBytes(words) else scales;
        if (index != 0 and !std.mem.eql(u8, data[guard..][0..size], expected)) return error.SharedBranchPackedMutation;
        if (index == 0 and (!std.mem.eql(u8, data[guard..][0..524288], part.w.bytes) or !std.mem.eql(u8, data[guard + 524288 ..][0..65536], part.blocks.?.bytes))) return error.SharedBranchPackedMutation;
    }
    try j.write(.{ .kind = "shared_branch_complete", .ordinal = gpu.ctx.device_ordinal, .positions = 6, .raw_files = 54, .actual_hf_activation_and_down = "requires independent raw audit", .free_bytes = (try gpu.ctx.memInfo()).free, .gate_and_final_moe = "not implemented in this screen", .full_model_quality = "FAIL unchanged" });
}

test "shared branch sentinels and exact BF16 boundaries reject corruption" {
    var data: [guard * 2 + 4]u8 = @splat(0xff);
    @memset(data[guard..][0..4], 0);
    try guarded(&data, 4, true);
    data[0] = 0;
    try std.testing.expectError(error.SharedBranchGuard, guarded(&data, 4, true));
    data[0] = 0xff;
    data[guard] = 0x80;
    data[guard + 1] = 0x7f;
    try std.testing.expectError(error.SharedBranchNonfinite, guarded(&data, 4, true));
    try std.testing.expectEqual(@as(usize, 1), try differences(&.{ 0, 0, 1, 0 }, &.{ 0, 0, 0, 0 }));
}
test "shared branch qlinear ABI checks all thirteen values and actual geometry" {
    var a = downArgs(0x1000, cuda.nvfp4.weight(0x2000, 0x3000, 0.25, outputs, width), 0x4000);
    var p: cuda.abi.KernelNodeParams = std.mem.zeroes(cuda.abi.KernelNodeParams);
    p.grid_x = 32;
    p.grid_y = 1;
    p.grid_z = 1;
    p.block_x = 128;
    p.block_y = 1;
    p.block_z = 1;
    p.shared_bytes = 17408;
    p.params = a.pointers();
    const f: cuda.Function = .{ .d = undefined, .handle = p.func };
    const want = a;
    _ = try validateNode(p, f, want, true);
    for (0..13) |i| {
        a.storage[a.offsets[i]] ^= 1;
        try std.testing.expectError(error.SharedBranchArguments, validateNode(p, f, want, true));
        a.storage[a.offsets[i]] ^= 1;
    }
    p.shared_bytes = 0;
    try std.testing.expectError(error.SharedBranchGeometry, validateNode(p, f, want, true));
}
