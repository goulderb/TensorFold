//! Native-input composition of the attributed HF expert and shared branches.
//! Captured HF tensors are host comparison targets and never device inputs.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const qwen = @import("qwen35_cuda");
const check = @import("check.zig");
const guard = 64;
const width = 512;
const outputs = 2048;
const File = struct { path: []const u8, bytes: usize, sha256: []const u8, role: []const u8 = "", position: usize = 999, label: []const u8 = "" };
const Manifest = struct {
    schema: u32,
    origin_manifest: File,
    checkpoint_file: File,
    checkpoint_index: File,
    frozen_image_sha256: []const u8,
    activation_image: File,
    expert_image: File,
    finish_image_sha256: []const u8,
    files: []const File,
};
const names = [_][:0]const u8{ "tf_q35_hf_mlp_act", "tf_shared_activation_control", "tf_shared_activation_candidate" };
const Stage = enum { activation, down, native_combine, experts, finish, recorder };
const Span = struct { base: u64, bytes: usize };
const input_labels = [_][]const u8{ "shared_gate.bf16", "shared_up.bf16", "pairs.f32", "weights.f32", "picks.i32" };
const input_sizes = [_]usize{ 1024, 1024, 65536, 36, 32 };
const output_labels = [_][]const u8{ "frozen_activation", "control_activation", "candidate_activation", "frozen_down", "control_down", "candidate_down", "frozen_full", "control_full", "candidate_experts", "candidate_full", "recorder_full", "rounded_shared_weight", "shared_product", "final_sum_f32" };
const output_sizes = [_]usize{ 1024, 1024, 1024, 4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096, 2, 4096, 8192 };

fn find(m: Manifest, position: usize, label: []const u8, role: []const u8) !File {
    var found: ?File = null;
    for (m.files) |f| {
        if (f.position != position or !std.mem.eql(u8, f.label, label)) continue;
        if (found != null or !std.mem.eql(u8, f.role, role)) return error.FullMoeFixture;
        found = f;
    }
    return found orelse error.FullMoeFixture;
}
fn disjoint(spans: []const Span) !void {
    for (spans, 0..) |left, i| {
        if (left.base == 0 or left.bytes == 0) return error.FullMoeAllocation;
        const end = std.math.add(u64, left.base, left.bytes) catch return error.FullMoeAllocation;
        for (spans[i + 1 ..]) |right| {
            const right_end = std.math.add(u64, right.base, right.bytes) catch return error.FullMoeAllocation;
            if (!(end <= right.base or right_end <= left.base)) return error.FullMoeOverlap;
        }
    }
}
fn allocate(gpu: check.Gpu, position: usize, label: []const u8, bytes: usize, j: anytype) !cuda.DeviceBuffer {
    var result = try cuda.DeviceBuffer.alloc(gpu.d, bytes + 2 * guard);
    errdefer result.free();
    try result.fill8(0xff, null);
    try j.write(.{ .kind = "full_moe_allocation", .ordinal = gpu.ctx.device_ordinal, .position = position, .label = label, .base = result.ptr, .payload = result.ptr + guard, .active_bytes = bytes, .allocation_bytes = result.len });
    return result;
}
fn loadImage(gpu: check.Gpu, f: File) !cuda.Module {
    const bytes = try read(gpu, f, "", f.bytes);
    defer gpu.gpa.free(bytes);
    const aligned = try gpu.gpa.alignedAlloc(u8, .@"8", bytes.len);
    defer gpu.gpa.free(aligned);
    @memcpy(aligned, bytes);
    return cuda.Module.load(gpu.d, aligned);
}

fn hash(bytes: []const u8) [64]u8 {
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &digest, .{});
    return std.fmt.bytesToHex(digest, .lower);
}
fn pin(bytes: []const u8, sha: []const u8) !void {
    if (sha.len != 64 or !std.mem.eql(u8, &hash(bytes), sha)) return error.FullMoePin;
}
fn read(gpu: check.Gpu, f: File, role: []const u8, size: usize) ![]u8 {
    if (!std.fs.path.isAbsolute(f.path) or f.bytes != size or !std.mem.eql(u8, f.role, role)) return error.FullMoeRole;
    const bytes = try std.Io.Dir.cwd().readFileAlloc(gpu.io, f.path, gpu.gpa, .limited(4 << 20));
    errdefer gpu.gpa.free(bytes);
    if (bytes.len != size) return error.FullMoeSize;
    try pin(bytes, f.sha256);
    return bytes;
}
fn finite(bytes: []const u8) !void {
    if (bytes.len % 2 != 0) return error.FullMoeSize;
    var i: usize = 0;
    while (i < bytes.len) : (i += 2) {
        if (std.mem.readInt(u16, bytes[i..][0..2], .little) & 0x7f80 == 0x7f80) return error.FullMoeNonfinite;
    }
}
fn finiteF32(bytes: []const u8) !void {
    if (bytes.len % 4 != 0) return error.FullMoeSize;
    var i: usize = 0;
    while (i < bytes.len) : (i += 4) {
        if (std.mem.readInt(u32, bytes[i..][0..4], .little) & 0x7f800000 == 0x7f800000) return error.FullMoeNonfinite;
    }
}
fn validateInput(bytes: []const u8, index: usize) !void {
    if (bytes.len != input_sizes[index]) return error.FullMoeSize;
    if (index < 2) return finite(bytes);
    if (index < 4) return finiteF32(bytes);
    var seen: [256]bool = @splat(false);
    for (0..8) |i| {
        const pick = std.mem.readInt(u32, bytes[i * 4 ..][0..4], .little);
        if (pick >= 256 or seen[pick]) return error.FullMoePicks;
        seen[pick] = true;
    }
}
fn guarded(bytes: []const u8, active: usize, numeric: bool) !void {
    if (bytes.len != active + 2 * guard) return error.FullMoeSize;
    for (bytes[0..guard]) |v| if (v != 0xff) return error.FullMoeGuard;
    for (bytes[guard + active ..]) |v| if (v != 0xff) return error.FullMoeGuard;
    if (numeric) try finite(bytes[guard..][0..active]);
}
fn differences(a: []const u8, b: []const u8) !usize {
    if (a.len != b.len or a.len % 2 != 0) return error.FullMoeSize;
    var count: usize = 0;
    var i: usize = 0;
    while (i < a.len) : (i += 2) {
        if (!std.mem.eql(u8, a[i..][0..2], b[i..][0..2])) count += 1;
    }
    return count;
}
fn save(gpu: check.Gpu, j: anytype, position: usize, label: []const u8, bytes: []const u8) !void {
    const name = try std.fmt.allocPrint(gpu.gpa, "full_moe_gpu{d}_pos{d}_{s}.bin", .{ gpu.ctx.device_ordinal, position, label });
    defer gpu.gpa.free(name);
    const path = try std.fs.path.join(gpu.gpa, &.{ j.directory, name });
    defer gpu.gpa.free(path);
    const file = try std.Io.Dir.cwd().createFile(gpu.io, path, .{ .exclusive = true });
    defer file.close(gpu.io);
    try file.writeStreamingAll(gpu.io, bytes);
    try j.write(.{ .kind = "full_moe_file", .ordinal = gpu.ctx.device_ordinal, .position = position, .label = label, .name = name, .bytes = bytes.len, .sha256 = &hash(bytes) });
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
fn validateNode(p: cuda.abi.KernelNodeParams, f: cuda.Function, want: cuda.Args, stage: Stage) ![64]u8 {
    const grid: u32 = if (stage == .down) 32 else if (stage == .activation) 2 else 8;
    const block: u32 = if (stage == .down) 128 else 256;
    const shared: u32 = if (stage == .down) 17408 else 0;
    if (p.func != f.handle or p.grid_x != grid or p.grid_y != 1 or p.grid_z != 1 or p.block_x != block or p.block_y != 1 or p.block_z != 1 or p.shared_bytes != shared) return error.FullMoeGeometry;
    if (want.count != (switch (stage) {
        .down => @as(usize, 13),
        .activation => 6,
        .recorder => 7,
        else => 4,
    })) return error.FullMoeArguments;
    const actual = p.params orelse return error.FullMoeArguments;
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    for (0..want.count) |i| {
        const ptr: [*]const u8 = @ptrCast(actual[i] orelse return error.FullMoeArguments);
        const bytes = want.storage[want.offsets[i]..][0..want.sizes[i]];
        if (!std.mem.eql(u8, ptr[0..bytes.len], bytes)) return error.FullMoeArguments;
        h.update(ptr[0..bytes.len]);
    }
    var digest: [32]u8 = undefined;
    h.final(&digest);
    return std.fmt.bytesToHex(digest, .lower);
}
fn inspectLaunch(gpu: check.Gpu, stream: cuda.Stream, graph: cuda.graph.Graph, f: cuda.Function, a: cuda.Args, position: usize, arm: usize, stage: Stage, symbol: []const u8, j: anytype) !void {
    if (try graph.nodeCount() != 1) return error.FullMoeNodes;
    var storage: [1]cuda.graph.Node = undefined;
    const nodes = try graph.nodes(&storage);
    var p: cuda.abi.KernelNodeParams = undefined;
    try gpu.d.check(gpu.d.api.cuGraphKernelNodeGetParams_v2(nodes[0], &p), "cuGraphKernelNodeGetParams");
    const digest = try validateNode(p, f, a, stage);
    var values: [13]u64 = @splat(0);
    for (0..a.count) |i| {
        const ptr: [*]const u8 = @ptrCast(p.params.?[i].?);
        for (ptr[0..a.sizes[i]], 0..) |v, shift| values[i] |= @as(u64, v) << @intCast(8 * shift);
    }
    try j.write(.{ .kind = "full_moe_node", .ordinal = gpu.ctx.device_ordinal, .position = position, .arm = arm, .stage = @tagName(stage), .symbol = symbol, .function_handle = @intFromPtr(p.func), .nodes = 1, .grid = [_]u32{ p.grid_x, p.grid_y, p.grid_z }, .block = [_]u32{ p.block_x, p.block_y, p.block_z }, .shared_bytes = p.shared_bytes, .abi_count = a.count, .argument_sizes = a.sizes[0..a.count], .arguments = values[0..a.count], .arguments_sha256 = &digest, .function_identity_checked = true });
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
fn execute(gpu: check.Gpu, stream: cuda.Stream, f: cuda.Function, a: *cuda.Args, position: usize, arm: usize, stage: Stage, symbol: []const u8, j: anytype) !void {
    try cuda.graph.beginCapture(stream, .thread_local);
    cuda.launch.launch(f, .{ .grid = .{ .x = if (stage == .activation) 2 else 8 }, .block = .{ .x = 256 } }, stream, a) catch |err| {
        abortCapture(stream);
        return err;
    };
    var graph = try cuda.graph.endCapture(stream);
    defer graph.deinit();
    try inspectLaunch(gpu, stream, graph, f, a.*, position, arm, stage, symbol, j);
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
    try inspectLaunch(gpu, stream, graph, lane.fns[0][0], downArgs(x, w, out), position, arm, .down, "qmmf fp4 BM16 SK1 BF16", j);
}
pub fn run(gpu: check.Gpu, folder: []const u8, manifest_sha: []const u8, image_path: []const u8, image_sha: []const u8, j: anytype) !void {
    return runInner(gpu, folder, manifest_sha, image_path, image_sha, j) catch |err| {
        j.write(.{ .kind = "full_moe_failure", .ordinal = gpu.ctx.device_ordinal, .failure = @errorName(err), .full_model_quality = "FAIL unchanged" }) catch {};
        return err;
    };
}
fn runInner(gpu: check.Gpu, folder: []const u8, manifest_sha: []const u8, image_path: []const u8, image_sha: []const u8, j: anytype) !void {
    const path = try std.fs.path.join(gpu.gpa, &.{ folder, "manifest.json" });
    defer gpu.gpa.free(path);
    const raw = try std.Io.Dir.cwd().readFileAlloc(gpu.io, path, gpu.gpa, .limited(1 << 20));
    defer gpu.gpa.free(raw);
    try pin(raw, manifest_sha);
    const parsed = try std.json.parseFromSlice(Manifest, gpu.gpa, raw, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    const m = parsed.value;
    if (m.schema != 1 or m.files.len != 180 or !std.mem.eql(u8, m.finish_image_sha256, image_sha) or !std.mem.eql(u8, m.origin_manifest.sha256, "2e2813cb00c9a351d48e4b9b2f20eb5533509f02e26ffd931ff12d16273fcfe9")) return error.FullMoeManifest;
    try pin(cuda.kernels.qwen35_hf, m.frozen_image_sha256);
    const origin = try read(gpu, m.origin_manifest, "", m.origin_manifest.bytes);
    defer gpu.gpa.free(origin);
    const ck_bytes = try read(gpu, m.checkpoint_file, "", 590244);
    defer gpu.gpa.free(ck_bytes);
    const ck_index = try read(gpu, m.checkpoint_index, "", m.checkpoint_index.bytes);
    defer gpu.gpa.free(ck_index);
    const directory = std.fs.path.dirname(m.checkpoint_file.path) orelse return error.FullMoeManifest;
    if (!std.mem.eql(u8, std.fs.path.basename(m.checkpoint_file.path), "model.safetensors") or !std.mem.eql(u8, std.fs.path.basename(m.checkpoint_index.path), "model.safetensors.index.json") or !std.mem.eql(u8, directory, std.fs.path.dirname(m.checkpoint_index.path) orelse return error.FullMoeManifest)) return error.FullMoeManifest;
    const image = try std.Io.Dir.cwd().readFileAlloc(gpu.io, image_path, gpu.gpa, .limited(16 << 20));
    defer gpu.gpa.free(image);
    try pin(image, image_sha);
    const aligned = try gpu.gpa.alignedAlloc(u8, .@"8", image.len);
    defer gpu.gpa.free(aligned);
    @memcpy(aligned, image);
    try j.write(.{ .kind = "full_moe_start", .ordinal = gpu.ctx.device_ordinal, .manifest_sha256 = manifest_sha, .origin_manifest_sha256 = m.origin_manifest.sha256, .image_sha256 = image_sha, .activation_image_sha256 = m.activation_image.sha256, .expert_image_sha256 = m.expert_image.sha256, .frozen_image_sha256 = m.frozen_image_sha256, .checkpoint_sha256 = m.checkpoint_file.sha256, .checkpoint_index_sha256 = m.checkpoint_index.sha256, .rows = 6, .activation_width = width, .output_width = outputs, .expected_hf_device_inputs = false, .free_bytes = (try gpu.ctx.memInfo()).free, .scope = "isolated native-input full MoE composition; no full-model or timing claim" });
    var ck = try core.Checkpoint.openModel(gpu.gpa, gpu.io, directory);
    defer ck.close();
    const part = try qwen.checkpoint.readPart(&ck, "model.language_model.layers.0.mlp.shared_expert.down_proj", outputs, width);
    if (part.kind != .nvfp4 or part.w.bytes.len != 524288 or part.blocks.?.bytes.len != 65536 or part.act != 0 or !(part.scale > 0) or !std.math.isFinite(part.scale)) return error.FullMoeFormat;
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
        buffer.* = try allocate(gpu, 999, ([_][]const u8{ "checkpoint_staging", "packed_words", "packed_scales" })[count], size, j);
        count += 1;
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
        if (!exact) return error.FullMoePackedBytes;
    }
    const w = cuda.nvfp4.weight(buffers[1].ptr + guard, buffers[2].ptr + guard, part.scale, outputs, width);
    const plan = cuda.qmmf.plan(1, outputs, width, lane.clusters);
    if (plan.bm != 16 or plan.sk != 1 or plan.fused or plan.cluster or plan.reduce) return error.FullMoePlan;
    try j.write(.{ .kind = "full_moe_packing", .ordinal = gpu.ctx.device_ordinal, .cpu_byte_proof = true, .n = outputs, .k = width, .bm = plan.bm, .sk = plan.sk, .global_scale_bits = @as(u32, @bitCast(w.scale)), .words = w.codes, .scales = w.scales, .no_partials_or_cluster = true, .qmmf_image_sha256 = &hash(cuda.kernels.qmmf), .packer_image_sha256 = &hash(cuda.kernels.modelopt_pack) });
    var original = try cuda.Module.load(gpu.d, cuda.kernels.qwen35_hf);
    defer original.unload();
    var module = try cuda.Module.load(gpu.d, aligned);
    defer module.unload();
    var act_module = try loadImage(gpu, m.activation_image);
    defer act_module.unload();
    var expert_module = try loadImage(gpu, m.expert_image);
    defer expert_module.unload();
    const fns = [_]cuda.Function{ try original.function(names[0]), try act_module.function(names[1]), try act_module.function(names[2]) };
    const combine_names = [_][:0]const u8{ "tf_q35_hf_combine", "tf_moe_combine_control" };
    const combine = [_]cuda.Function{ try original.function(combine_names[0]), try expert_module.function(combine_names[1]) };
    const expert = try expert_module.function("tf_moe_experts_candidate");
    const finish = try module.function("tf_full_moe_finish_candidate");
    const recorder = try module.function("tf_full_moe_finish_recorder");
    if (finish.handle == recorder.handle) return error.FullMoeFunctionAlias;
    const resource_fns = [_]cuda.Function{ fns[0], fns[1], fns[2], combine[0], combine[1], expert, finish, recorder, lane.fns[0][0] };
    const resource_names = [_][]const u8{ names[0], names[1], names[2], combine_names[0], combine_names[1], "tf_moe_experts_candidate", "tf_full_moe_finish_candidate", "tf_full_moe_finish_recorder", "qmmf fp4 BM16 SK1 BF16" };
    for (resource_fns, resource_names, 0..) |f, symbol, index| {
        const local = try f.attribute(.local_size_bytes);
        try j.write(.{ .kind = "full_moe_resource", .ordinal = gpu.ctx.device_ordinal, .resource = index, .symbol = symbol, .function_handle = @intFromPtr(f.handle), .registers = try f.attribute(.num_regs), .local_bytes = local, .static_shared = try f.attribute(.shared_size_bytes) });
        if (local != 0) return error.FullMoeLocalStorage;
    }
    for (0..6) |position| {
        var inputs: [5]cuda.DeviceBuffer = undefined;
        var out: [14]cuda.DeviceBuffer = undefined;
        var ni: usize = 0;
        var no: usize = 0;
        defer for (inputs[0..ni]) |*buffer| buffer.free();
        defer for (out[0..no]) |*buffer| buffer.free();
        var spans: [22]Span = undefined;
        for (buffers, 0..) |buffer, index| spans[index] = .{ .base = buffer.ptr, .bytes = buffer.len };
        for (&inputs, input_labels, input_sizes, 0..) |*buffer, label, size, index| {
            const data = try read(gpu, try find(m, position, label, "device_native_input"), "device_native_input", size);
            defer gpu.gpa.free(data);
            try validateInput(data, index);
            buffer.* = try allocate(gpu, position, label, size, j);
            ni += 1;
            spans[3 + index] = .{ .base = buffer.ptr, .bytes = buffer.len };
            try buffer.upload(guard, data);
        }
        for (&out, output_labels, output_sizes, 0..) |*buffer, label, size, index| {
            buffer.* = try allocate(gpu, position, label, size, j);
            no += 1;
            spans[8 + index] = .{ .base = buffer.ptr, .bytes = buffer.len };
        }
        try disjoint(&spans);
        for (fns, 0..) |f, arm| {
            var a = actArgs(inputs[0].ptr + guard, inputs[1].ptr + guard, out[arm].ptr + guard);
            try execute(gpu, stream, f, &a, position, arm, .activation, names[arm], j);
            try projection(gpu, stream, &lane, out[arm].ptr + guard, w, out[3 + arm].ptr + guard, position, arm, j);
        }
        for (combine, 0..) |f, arm| {
            var a: cuda.Args = .{};
            for ([_]u64{ inputs[2].ptr + guard, inputs[3].ptr + guard, out[3 + arm].ptr + guard, out[6 + arm].ptr + guard }) |ptr| a.add(ptr);
            try execute(gpu, stream, f, &a, position, arm, .native_combine, combine_names[arm], j);
        }
        var expert_args: cuda.Args = .{};
        for ([_]u64{ inputs[2].ptr + guard, inputs[3].ptr + guard, inputs[4].ptr + guard, out[8].ptr + guard }) |ptr| expert_args.add(ptr);
        try execute(gpu, stream, expert, &expert_args, position, 2, .experts, "tf_moe_experts_candidate", j);
        var finish_args: cuda.Args = .{};
        for ([_]u64{ out[8].ptr + guard, out[5].ptr + guard, inputs[3].ptr + guard, out[9].ptr + guard }) |ptr| finish_args.add(ptr);
        try execute(gpu, stream, finish, &finish_args, position, 2, .finish, "tf_full_moe_finish_candidate", j);
        var recorder_args: cuda.Args = .{};
        for ([_]u64{ out[8].ptr + guard, out[5].ptr + guard, inputs[3].ptr + guard, out[10].ptr + guard, out[11].ptr + guard, out[12].ptr + guard, out[13].ptr + guard }) |ptr| recorder_args.add(ptr);
        try execute(gpu, stream, recorder, &recorder_args, position, 2, .recorder, "tf_full_moe_finish_recorder", j);
        var reads: [19][]u8 = undefined;
        var nr: usize = 0;
        defer for (reads[0..nr]) |data| gpu.gpa.free(data);
        for (out, output_labels, 0..) |buffer, label, index| {
            reads[index] = try check.download(gpu, buffer);
            nr += 1;
            try save(gpu, j, position, label, reads[index]);
        }
        for (inputs, input_labels, 0..) |buffer, label, index| {
            reads[14 + index] = try check.download(gpu, buffer);
            nr += 1;
            const readback_label = try std.fmt.allocPrint(gpu.gpa, "{s}_readback", .{label});
            defer gpu.gpa.free(readback_label);
            try save(gpu, j, position, readback_label, reads[14 + index]);
        }
        const targets = [_][]const u8{ "native-shared_act.bf16", "native-shared_act.bf16", "hf-shared_expert-down_proj-input.H", "native-shared_out.bf16", "native-shared_out.bf16", "hf-shared_expert-down_proj-output.H", "native-moe.bf16", "native-moe.bf16", "candidate-experts.H", "candidate-full.H", "candidate-full.H", "candidate-shared-weight.H", "candidate-shared-product.H", "candidate-final-sum.f" };
        var mismatches: usize = 0;
        for (reads[0..14], output_sizes, targets, 0..) |data, size, target, index| {
            try guarded(data, size, false);
            if (index == 13) try finiteF32(data[guard..][0..size]) else try finite(data[guard..][0..size]);
            const role = if (index >= 8) "host_synthesized" else if (index == 2 or index == 5) "host_original_hf" else "host_native_control";
            const expected = try read(gpu, try find(m, position, target, role), role, size);
            defer gpu.gpa.free(expected);
            if (!std.mem.eql(u8, data[guard..][0..size], expected)) mismatches += 1;
        }
        for (reads[14..], input_labels, input_sizes, 0..) |data, label, size, index| {
            try guarded(data, size, false);
            try validateInput(data[guard..][0..size], index);
            const expected = try read(gpu, try find(m, position, label, "device_native_input"), "device_native_input", size);
            defer gpu.gpa.free(expected);
            if (!std.mem.eql(u8, data[guard..][0..size], expected)) return error.FullMoeInputMutation;
        }
        if (!std.mem.eql(u8, reads[9], reads[10])) return error.FullMoeRecorderMismatch;
        const hf_expert = try read(gpu, try find(m, position, "hf-experts-0.H", "host_original_hf"), "host_original_hf", 4096);
        defer gpu.gpa.free(hf_expert);
        const hf_full = try read(gpu, try find(m, position, "hf-moe-output.H", "host_original_hf"), "host_original_hf", 4096);
        defer gpu.gpa.free(hf_full);
        const expert_diffs = try differences(reads[8][guard..][0..4096], hf_expert);
        const final_diffs = try differences(reads[9][guard..][0..4096], hf_full);
        const eligible = position >= 2 and position <= 4;
        if (eligible and (expert_diffs != 0 or final_diffs != 0)) mismatches += 1;
        try j.write(.{ .kind = "full_moe_case", .ordinal = gpu.ctx.device_ordinal, .position = position, .raw_files = 19, .live_allocations = spans.len, .live_allocations_disjoint = true, .mismatched_planes = mismatches, .actual_hf_final_gate_eligible = eligible, .candidate_expert_differences_vs_actual_hf = expert_diffs, .candidate_final_differences_vs_actual_hf = final_diffs, .recorder_candidate_exact = true, .correctness_pass = mismatches == 0, .full_model_quality = "FAIL unchanged" });
        if (mismatches != 0) return error.FullMoeMismatch;
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
        if (index != 0 and !std.mem.eql(u8, data[guard..][0..size], expected)) return error.FullMoePackedMutation;
        if (index == 0 and (!std.mem.eql(u8, data[guard..][0..524288], part.w.bytes) or !std.mem.eql(u8, data[guard + 524288 ..][0..65536], part.blocks.?.bytes))) return error.FullMoePackedMutation;
    }
    try j.write(.{ .kind = "full_moe_complete", .ordinal = gpu.ctx.device_ordinal, .positions = 6, .raw_files = 120, .actual_hf_final_positions = [_]usize{ 2, 3, 4 }, .free_bytes = (try gpu.ctx.memInfo()).free, .full_model_quality = "FAIL unchanged" });
}

test "shared branch sentinels and exact BF16 boundaries reject corruption" {
    var data: [guard * 2 + 4]u8 = @splat(0xff);
    @memset(data[guard..][0..4], 0);
    try guarded(&data, 4, true);
    data[0] = 0;
    try std.testing.expectError(error.FullMoeGuard, guarded(&data, 4, true));
    data[0] = 0xff;
    data[guard] = 0x80;
    data[guard + 1] = 0x7f;
    try std.testing.expectError(error.FullMoeNonfinite, guarded(&data, 4, true));
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
    _ = try validateNode(p, f, want, .down);
    for (0..13) |i| {
        a.storage[a.offsets[i]] ^= 1;
        try std.testing.expectError(error.FullMoeArguments, validateNode(p, f, want, .down));
        a.storage[a.offsets[i]] ^= 1;
    }
    p.shared_bytes = 0;
    try std.testing.expectError(error.FullMoeGeometry, validateNode(p, f, want, .down));
}

test "full MoE allocations reject every live overlap and overflow" {
    var spans = [_]Span{ .{ .base = 4096, .bytes = 1152 }, .{ .base = 8192, .bytes = 4224 }, .{ .base = 16384, .bytes = 128 } };
    try disjoint(&spans);
    spans[2].base = 8193;
    try std.testing.expectError(error.FullMoeOverlap, disjoint(&spans));
    spans[2] = .{ .base = std.math.maxInt(u64) - 3, .bytes = 8 };
    try std.testing.expectError(error.FullMoeAllocation, disjoint(&spans));
}
test "full MoE picks retain native identities without duplicates" {
    var data: [32]u8 = @splat(0);
    for (0..8) |i| std.mem.writeInt(u32, data[i * 4 ..][0..4], @intCast(i), .little);
    try validateInput(&data, 4);
    data[28] = 0;
    try std.testing.expectError(error.FullMoePicks, validateInput(&data, 4));
}
test "full MoE finish and recorder validate all actual pointer bytes" {
    for ([_]Stage{ .finish, .recorder }) |stage| {
        var a: cuda.Args = .{};
        const count: usize = if (stage == .finish) 4 else 7;
        for (0..count) |i| a.add(@as(u64, 4096 + i * 8192));
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
        _ = try validateNode(p, f, want, stage);
        for (0..count) |i| {
            a.storage[a.offsets[i]] ^= 1;
            try std.testing.expectError(error.FullMoeArguments, validateNode(p, f, want, stage));
            a.storage[a.offsets[i]] ^= 1;
        }
    }
}
