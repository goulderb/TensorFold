//! Production format proof adapted from the retained TensorFold slice-major screen.
//! Checkpoint bytes feed the real packer; saved outputs remain comparison targets.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const qwen = @import("qwen35_cuda");
const check = @import("check.zig");
const X = cuda.experts.Nvfp4Experts;
const width = 512;
const dims = 2048;
const slots = 8;
const experts = 256;
const guard = 64;
const stride = dims + 64;
const File = struct { path: []const u8, bytes: usize, sha256: []const u8 };
const Part = struct { expert: usize, part: usize, codes_sha256: []const u8, blocks_sha256: []const u8, scale_bits: u32 };
const Manifest = struct {
    schema: usize,
    checkpoint_dir: []const u8,
    checkpoint_index: File,
    origin: File,
    expert_order: [experts]usize,
    files: std.json.ArrayHashMap(File),
    parts: [experts * 2]Part,
    frozen_image: File,
    pack_image_sha256: []const u8,
    expert_image_sha256: []const u8,
};

fn hash(raw: []const u8) [64]u8 {
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(raw, &digest, .{});
    return std.fmt.bytesToHex(digest, .lower);
}
fn pin(raw: []const u8, sha: []const u8) !void {
    if (sha.len != 64 or !std.mem.eql(u8, &hash(raw), sha)) return error.SliceFormatPin;
}
fn read(gpu: check.Gpu, f: File) ![]u8 {
    if (!std.fs.path.isAbsolute(f.path)) return error.SliceFormatPath;
    const raw = try std.Io.Dir.cwd().readFileAlloc(gpu.io, f.path, gpu.gpa, .limited(f.bytes + 1));
    errdefer gpu.gpa.free(raw);
    if (raw.len != f.bytes) return error.SliceFormatSize;
    try pin(raw, f.sha256);
    return raw;
}
fn field(gpu: check.Gpu, m: Manifest, name: []const u8, size: usize) ![]u8 {
    const f = m.files.map.get(name) orelse return error.SliceFormatMissing;
    if (f.bytes != size) return error.SliceFormatSize;
    return read(gpu, f);
}
fn save(gpu: check.Gpu, j: anytype, name: []const u8, raw: []const u8) !void {
    const basename = try std.fmt.allocPrint(gpu.gpa, "slice_format_gpu{d}_{s}.bin", .{ gpu.ctx.device_ordinal, name });
    defer gpu.gpa.free(basename);
    const path = try std.fs.path.join(gpu.gpa, &.{ j.directory, basename });
    defer gpu.gpa.free(path);
    const f = try std.Io.Dir.cwd().createFile(gpu.io, path, .{ .exclusive = true });
    defer f.close(gpu.io);
    try f.writeStreamingAll(gpu.io, raw);
    try j.write(.{ .kind = "slice_format_artifact", .ordinal = gpu.ctx.device_ordinal, .name = name, .file = basename, .bytes = raw.len, .sha256 = &hash(raw) });
}
fn permutation(original: []const u8, candidate: []const u8) !void {
    if (original.len != candidate.len or original.len % 576 != 0) return error.SliceFormatPermutation;
    var offset: usize = 0;
    while (offset < original.len) : (offset += 576) {
        for (0..128) |word| {
            const old = 4 * (word % 32) + word / 32;
            if (!std.mem.eql(u8, candidate[offset + word * 4 ..][0..4], original[offset + old * 4 ..][0..4])) return error.SliceFormatPermutation;
        }
        if (!std.mem.eql(u8, original[offset + 512 ..][0..64], candidate[offset + 512 ..][0..64])) return error.SliceFormatPermutation;
    }
}
const Guarded = struct {
    buffer: cuda.DeviceBuffer,
    bytes: usize,
    fn init(gpu: check.Gpu, bytes: usize) !Guarded {
        var buffer = try cuda.DeviceBuffer.alloc(gpu.d, bytes + guard * 2);
        errdefer buffer.free();
        const b: Guarded = .{ .buffer = buffer, .bytes = bytes };
        try b.poison();
        return b;
    }
    fn ptr(b: Guarded) u64 {
        return b.buffer.ptr + guard;
    }
    fn poison(b: Guarded) !void {
        try b.buffer.fill8(0xa5, null);
        const payload: cuda.DeviceBuffer = .{ .d = b.buffer.d, .ptr = b.ptr(), .len = b.bytes, .borrowed = true };
        try payload.fill8(0xff, null);
    }
    fn active(b: Guarded, raw: []const u8) ![]const u8 {
        if (raw.len != b.bytes + 2 * guard) return error.SliceFormatSize;
        for (raw[0..guard]) |v| if (v != 0xa5) return error.SliceFormatGuard;
        for (raw[guard + b.bytes ..]) |v| if (v != 0xa5) return error.SliceFormatGuard;
        return raw[guard..][0..b.bytes];
    }
    fn deinit(b: *Guarded) void {
        b.buffer.free();
    }
};
fn finite(raw: []const u8) !void {
    if (raw.len % 2 != 0) return error.SliceFormatSize;
    var i: usize = 0;
    while (i < raw.len) : (i += 2) if (std.mem.readInt(u16, raw[i..][0..2], .little) & 0x7f80 == 0x7f80) return error.SliceFormatNonfinite;
}
fn nodeCheck(p: cuda.abi.KernelNodeParams, function: cuda.Function, grid: [3]u32, threads: u32, args: *const cuda.Args) ![64]u8 {
    if (p.func != function.handle or p.grid_x != grid[0] or p.grid_y != grid[1] or p.grid_z != grid[2] or p.block_x != threads or p.block_y != 1 or p.block_z != 1 or p.shared_bytes != 0) return error.SliceFormatGeometry;
    const params = p.params orelse return error.SliceFormatArguments;
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    for (0..args.count) |i| {
        const ptr: [*]const u8 = @ptrCast(params[i] orelse return error.SliceFormatArguments);
        const want = args.storage[args.offsets[i]..][0..args.sizes[i]];
        if (!std.mem.eql(u8, ptr[0..want.len], want)) return error.SliceFormatArguments;
        h.update(ptr[0..want.len]);
    }
    var digest: [32]u8 = undefined;
    h.final(&digest);
    return std.fmt.bytesToHex(digest, .lower);
}
fn captured(gpu: check.Gpu, stream: cuda.Stream, graph: *cuda.graph.Graph, f: cuda.Function, args: *const cuda.Args, grid: [3]u32, threads: u32, phase: []const u8, first: usize, second: usize, third: usize, symbol: []const u8, j: anytype) !void {
    if (try graph.nodeCount() != 1) return error.SliceFormatNodes;
    var store: [1]cuda.graph.Node = undefined;
    const nodes = try graph.nodes(&store);
    var p: cuda.abi.KernelNodeParams = undefined;
    try gpu.d.check(gpu.d.api.cuGraphKernelNodeGetParams_v2(nodes[0], &p), "cuGraphKernelNodeGetParams");
    const sha = try nodeCheck(p, f, grid, threads, args);
    var values: [14]u64 = undefined;
    for (0..args.count) |i| values[i] = args.integer(i) orelse return error.SliceFormatArguments;
    try j.write(.{ .kind = "slice_format_node", .ordinal = gpu.ctx.device_ordinal, .phase = phase, .first = first, .second = second, .third = third, .symbol = symbol, .function_handle = @intFromPtr(p.func), .nodes = 1, .grid = grid, .threads = threads, .shared_bytes = 0, .argument_values = values[0..args.count], .argument_sizes = args.sizes[0..args.count], .argument_sha256 = &sha, .actual_driver_node_readback = true });
    var exec = try graph.instantiate();
    defer exec.deinit();
    try exec.launchOn(stream);
    try stream.synchronize();
}
fn cancel(stream: cuda.Stream) void {
    if (cuda.graph.endCapture(stream)) |value| {
        var graph = value;
        graph.deinit();
    } else |_| {}
}

pub fn run(gpu: check.Gpu, folder: []const u8, manifest_sha: []const u8, j: anytype) !void {
    if (!std.fs.path.isAbsolute(folder)) return error.SliceFormatPath;
    const path = try std.fs.path.join(gpu.gpa, &.{ folder, "manifest.json" });
    defer gpu.gpa.free(path);
    const raw = try std.Io.Dir.cwd().readFileAlloc(gpu.io, path, gpu.gpa, .limited(1 << 20));
    defer gpu.gpa.free(raw);
    try pin(raw, manifest_sha);
    const parsed = try std.json.parseFromSlice(Manifest, gpu.gpa, raw, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    const m = parsed.value;
    if (m.schema != 1 or m.files.map.count() != 5 or !std.fs.path.isAbsolute(m.checkpoint_dir)) return error.SliceFormatManifest;
    var seen: [experts]bool = @splat(false);
    for (m.expert_order) |expert| {
        if (expert >= experts or seen[expert]) return error.SliceFormatManifest;
        seen[expert] = true;
    }
    const origin = try read(gpu, m.origin);
    defer gpu.gpa.free(origin);
    const index = try read(gpu, m.checkpoint_index);
    defer gpu.gpa.free(index);
    try pin(cuda.kernels.modelopt_pack, m.pack_image_sha256);
    try pin(cuda.kernels.nvfp4_experts, m.expert_image_sha256);
    const one = X.bytes(1, width, dims, 2);
    const original = try field(gpu, m, "up.original.bin", experts * one);
    defer gpu.gpa.free(original);
    const expected = try field(gpu, m, "up.slice-major.bin", experts * one);
    defer gpu.gpa.free(expected);
    try permutation(original, expected);
    const scales = try field(gpu, m, "up-scales.bin", experts * 8);
    defer gpu.gpa.free(scales);
    const input = try field(gpu, m, "input.bin", dims * 2);
    defer gpu.gpa.free(input);
    const saved = try field(gpu, m, "hidden.bin", slots * width * 2);
    defer gpu.gpa.free(saved);
    var ck = try core.Checkpoint.openModel(gpu.gpa, gpu.io, m.checkpoint_dir);
    defer ck.close();
    var stream = try cuda.Stream.init(gpu.d, false);
    defer stream.deinit();
    var pack = try cuda.modelopt.Packer.load(gpu.d);
    defer pack.unload();
    try pack.loadSliceMajorExperts();
    var stage = try cuda.DeviceBuffer.alloc(gpu.d, width * dims / 2 + width * dims / 16);
    defer stage.free();
    var format_weights = try Guarded.init(gpu, experts * one);
    defer format_weights.deinit();
    try j.write(.{ .kind = "slice_format_start", .ordinal = gpu.ctx.device_ordinal, .manifest_sha256 = manifest_sha, .pack_image_sha256 = m.pack_image_sha256, .expert_image_sha256 = m.expert_image_sha256, .frozen_image_sha256 = m.frozen_image.sha256, .candidate_ptr = format_weights.ptr(), .candidate_bytes = format_weights.bytes, .stage_ptr = stage.ptr, .stage_bytes = stage.len, .scope = "direct production pack and tagged split-M consumers; no timing or full-model result" });
    var scale_values: [experts * 2]f32 = undefined;
    for (m.expert_order, 0..) |expert, order| {
        inline for (.{ "gate_proj", "up_proj" }, 0..) |suffix, part| {
            const expected_part = m.parts[order * 2 + part];
            if (expected_part.expert != expert or expected_part.part != part) return error.SliceFormatManifest;
            var buf: [256]u8 = undefined;
            const pre = try std.fmt.bufPrint(&buf, "model.language_model.layers.0.mlp.experts.{d}.{s}", .{ expert, suffix });
            const p = try qwen.checkpoint.readPart(&ck, pre, width, dims);
            if (p.kind != .nvfp4 or p.blocks == null or @as(u32, @bitCast(p.scale)) != expected_part.scale_bits) return error.SliceFormatCheckpoint;
            try pin(p.w.bytes, expected_part.codes_sha256);
            try pin(p.blocks.?.bytes, expected_part.blocks_sha256);
            scale_values[order * 2 + part] = p.scale;
            try stage.upload(0, p.w.bytes);
            try stage.upload(p.w.bytes.len, p.blocks.?.bytes);
            try stream.synchronize();
            var args: cuda.Args = .{};
            args.add(stage.ptr);
            args.add(stage.ptr + p.w.bytes.len);
            args.add(format_weights.ptr() + order * one);
            for ([_]usize{ width, dims, 0, dims, 2, part }) |v| args.add(@as(i32, @intCast(v)));
            try cuda.graph.beginCapture(stream, .thread_local);
            pack.fp4ExpertsLayout(stream, stage.ptr, stage.ptr + p.w.bytes.len, format_weights.ptr() + order * one, 1, width, dims, 0, dims, 2, part, .slice_major_v1) catch |err| {
                cancel(stream);
                return err;
            };
            var graph = try cuda.graph.endCapture(stream);
            defer graph.deinit();
            try captured(gpu, stream, &graph, pack.fp4_experts_slice_major.?, &args, .{ 64, 16, 1 }, 144, "pack", order, expert, part, "tf_pack_fp4_experts_slice_major", j);
            try j.write(.{ .kind = "slice_format_part", .ordinal = gpu.ctx.device_ordinal, .order = order, .expert = expert, .part = part, .codes_sha256 = expected_part.codes_sha256, .blocks_sha256 = expected_part.blocks_sha256, .scale_bits = expected_part.scale_bits, .output_offset = order * one, .checkpoint_operands_verified = true });
        }
    }
    try check.sameBytes("slice-format checkpoint global scales", std.mem.sliceAsBytes(&scale_values), scales);
    const downloaded = try check.download(gpu, format_weights.buffer);
    defer gpu.gpa.free(downloaded);
    try save(gpu, j, "direct_packed_guarded", downloaded);
    const active = try format_weights.active(downloaded);
    try permutation(original, active);
    try check.sameBytes("slice-format all format_weights words", active, expected);
    try j.write(.{ .kind = "slice_format_pack_complete", .ordinal = gpu.ctx.device_ordinal, .projections = 512, .experts = experts, .parts = 2, .packed_bytes = active.len, .sha256 = &hash(active), .inverse_sha256 = &hash(original), .scale_sha256 = &hash(scales), .guard_bytes = guard, .one_candidate_allocation = true, .down_loaded = false, .down_qualification_claim = false });
    try consumers(gpu, stream, m, format_weights.ptr(), original, scales, input, saved, j);
    try j.write(.{ .kind = "slice_format_complete", .ordinal = gpu.ctx.device_ordinal, .pack_projections = 512, .banks = 32, .rows = [_]usize{ 1, 2, 16 }, .arms = 3, .consumer_launches = 288, .all_exact = true, .timing_measured = false, .production_qualification = false, .full_model_quality_claim = false });
}

fn consumers(gpu: check.Gpu, stream: cuda.Stream, m: Manifest, packed_ptr: u64, original: []const u8, scales: []const u8, input: []const u8, saved: []const u8, j: anytype) !void {
    var original_device = try cuda.DeviceBuffer.fromHost(gpu.d, original);
    defer original_device.free();
    var scale_device = try cuda.DeviceBuffer.fromHost(gpu.d, scales);
    defer scale_device.free();
    for ([_]cuda.DeviceBuffer{ original_device, scale_device }, [_][]const u8{ original, scales }, [_][]const u8{ "consumed_original", "consumed_scales" }) |buffer, want, label| {
        const got = try check.download(gpu, buffer);
        defer gpu.gpa.free(got);
        try save(gpu, j, label, got);
        try check.sameBytes("slice-format consumer upload", got, want);
    }
    const frozen_raw = try read(gpu, m.frozen_image);
    defer gpu.gpa.free(frozen_raw);
    const aligned = try gpu.gpa.alignedAlloc(u8, .@"16", frozen_raw.len);
    defer gpu.gpa.free(aligned);
    @memcpy(aligned, frozen_raw);
    var frozen = try cuda.Module.load(gpu.d, aligned);
    defer frozen.unload();
    const historic = try frozen.function("tf_nvfp4_n8_split_m_swiglu");
    const sms: usize = @intCast(try gpu.ctx.attribute(.multiprocessor_count));
    var e = try X.loadWarps(gpu.d, sms, 4);
    defer e.unload();
    try e.loadSplitMUp(sms);
    try e.loadSliceMajorUp(sms);
    const names = [_][]const u8{ "frozen", "control", "slice_major" };
    const symbols = [_][]const u8{ "tf_nvfp4_n8_split_m_swiglu", "tf_nvfp4_n8_split_m_swiglu", "tf_nvfp4_n8_split_m_slice_major_swiglu" };
    const functions = [_]cuda.Function{ historic, (e.split_m_up orelse return error.CandidateNotLoaded).function, (e.slice_major_up orelse return error.CandidateNotLoaded).function };
    for (functions, 0..) |f, arm| {
        if (try f.occupancy(256, 0) * sms < 128 or try f.attribute(.shared_size_bytes) != 2048 or try f.attribute(.local_size_bytes) != 0) return error.SliceFormatResources;
        try j.write(.{ .kind = "slice_format_resources", .ordinal = gpu.ctx.device_ordinal, .arm = arm, .name = names[arm], .symbol = symbols[arm], .registers = try f.attribute(.num_regs), .static_shared_bytes = try f.attribute(.shared_size_bytes), .local_bytes = try f.attribute(.local_size_bytes), .resident_per_sm = try f.occupancy(256, 0), .sms = sms });
    }
    const one = X.bytes(1, width, dims, 2);
    for ([_]usize{ 1, 2, 16 }) |rows| {
        const host_rows = try gpu.gpa.alloc(u8, rows * stride * 2);
        defer gpu.gpa.free(host_rows);
        @memset(host_rows, 0xa5);
        for (0..rows) |row| for (0..dims) |col| {
            const bits = std.mem.readInt(u16, input[col * 2 ..][0..2], .little) ^ @as(u16, @intCast((row * 17) & 127));
            std.mem.writeInt(u16, host_rows[(row * stride + col) * 2 ..][0..2], bits, .little);
        };
        var input_device = try cuda.DeviceBuffer.fromHost(gpu.d, host_rows);
        defer input_device.free();
        const input_actual = try check.download(gpu, input_device);
        defer gpu.gpa.free(input_actual);
        try check.sameBytes("slice-format input", input_actual, host_rows);
        const input_name = try std.fmt.allocPrint(gpu.gpa, "rows{d}_input", .{rows});
        defer gpu.gpa.free(input_name);
        try save(gpu, j, input_name, input_actual);
        var members: [16 * slots]i32 = undefined;
        var items: [slots][3]i32 = undefined;
        const counts = [_]i32{ slots, 0 };
        for (0..slots) |expert| {
            items[expert] = .{ @intCast(expert), @intCast(expert * rows), @intCast(rows) };
            for (0..rows) |row| members[expert * rows + row] = @intCast(row * slots + expert);
        }
        var md = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(members[0 .. rows * slots]));
        defer md.free();
        var it = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.asBytes(&items));
        defer it.free();
        var ct = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.asBytes(&counts));
        defer ct.free();
        const plan: cuda.grouped.Plan = .{ .members = md.ptr, .items = it.ptr, .counts = ct.ptr, .rank = 0, .hist = 0 };
        for ([_]cuda.DeviceBuffer{ md, it, ct }, [_][]const u8{ std.mem.sliceAsBytes(members[0 .. rows * slots]), std.mem.asBytes(&items), std.mem.asBytes(&counts) }, 0..) |buffer, want, index| {
            const got = try check.download(gpu, buffer);
            defer gpu.gpa.free(got);
            try check.sameBytes("slice-format plan", got, want);
            const name = try std.fmt.allocPrint(gpu.gpa, "rows{d}_plan{d}", .{ rows, index });
            defer gpu.gpa.free(name);
            try save(gpu, j, name, got);
        }
        var out = try Guarded.init(gpu, rows * slots * width * 2);
        defer out.deinit();
        const reference = try gpu.gpa.alloc(u8, out.bytes);
        defer gpu.gpa.free(reference);
        for (0..32) |bank| for (functions, 0..) |function, arm| {
            try out.poison();
            const layer: cuda.experts.Layer = .{ .format = .nvfp4, .nvfp4_up_layout = if (arm == 2) .slice_major_v1 else .lane_major, .up = (if (arm == 2) packed_ptr else original_device.ptr) + bank * slots * one, .down = 0, .up_scale = scale_device.ptr + bank * slots * 8, .width = width, .dims = dims, .experts = slots };
            const row_input: cuda.experts.Rows = .{ .x = input_device.ptr, .stride = stride, .slots = slots };
            var args = X.decodeArgs(.up, row_input, layer, plan, out.ptr(), -1);
            try cuda.graph.beginCapture(stream, .thread_local);
            if (arm == 0) {
                cuda.launch.launch(function, .{ .grid = .{ .x = 128 }, .block = .{ .x = 256 } }, stream, &args) catch |err| {
                    cancel(stream);
                    return err;
                };
            } else {
                e.runSplitMUp(stream, row_input, layer, plan, out.ptr(), slots, -1) catch |err| {
                    cancel(stream);
                    return err;
                };
            }
            var graph = try cuda.graph.endCapture(stream);
            defer graph.deinit();
            try captured(gpu, stream, &graph, function, &args, .{ 128, 1, 1 }, 256, "consumer", rows, bank, arm, symbols[arm], j);
            const got = try check.download(gpu, out.buffer);
            defer gpu.gpa.free(got);
            const label = try std.fmt.allocPrint(gpu.gpa, "rows{d}_bank{d}_{s}", .{ rows, bank, names[arm] });
            defer gpu.gpa.free(label);
            try save(gpu, j, label, got);
            const active = try out.active(got);
            try finite(active);
            if (arm == 0) @memcpy(reference, active) else try check.sameBytes("slice-format bank consumer", active, reference);
            if (rows == 1 and bank == 0) try check.sameBytes("slice-format historical captured output", active, saved);
            try j.write(.{ .kind = "slice_format_consumer", .ordinal = gpu.ctx.device_ordinal, .rows = rows, .bank = bank, .arm = arm, .name = names[arm], .layout = @tagName(layer.nvfp4_up_layout), .sha256 = &hash(active), .active_bytes = active.len, .guard_bytes = guard, .exact = true, .finite = true, .production_tag_dispatch = arm != 0 });
        };
    }
}

test "slice-format full block permutation preserves scale suffix and rejects corruption" {
    var original: [1152]u8 = undefined;
    var candidate: [1152]u8 = undefined;
    for (0..288) |i| std.mem.writeInt(u32, original[i * 4 ..][0..4], @intCast(i), .little);
    for (0..2) |block| {
        for (0..128) |word| @memcpy(candidate[block * 576 + word * 4 ..][0..4], original[block * 576 + (4 * (word % 32) + word / 32) * 4 ..][0..4]);
        @memcpy(candidate[block * 576 + 512 ..][0..64], original[block * 576 + 512 ..][0..64]);
    }
    try permutation(&original, &candidate);
    for ([_]usize{ 0, 511, 512, 575, 576, 1151 }) |i| {
        candidate[i] ^= 1;
        try std.testing.expectError(error.SliceFormatPermutation, permutation(&original, &candidate));
        candidate[i] ^= 1;
    }
}

test "slice-format node proof rejects each ABI slot and geometry mutation" {
    const f: cuda.Function = .{ .d = undefined, .handle = @ptrFromInt(1) };
    var args: cuda.Args = .{};
    for (0..9) |i| if (i < 3) {
        args.add(@as(u64, i + 100));
    } else {
        args.add(@as(i32, @intCast(i)));
    };
    var actual = args;
    var node: cuda.abi.KernelNodeParams = .{ .func = f.handle, .grid_x = 64, .grid_y = 16, .grid_z = 1, .block_x = 144, .block_y = 1, .block_z = 1, .shared_bytes = 0, .params = actual.pointers(), .extra = null };
    _ = try nodeCheck(node, f, .{ 64, 16, 1 }, 144, &args);
    for (0..9) |i| {
        actual.storage[actual.offsets[i]] ^= 1;
        try std.testing.expectError(error.SliceFormatArguments, nodeCheck(node, f, .{ 64, 16, 1 }, 144, &args));
        actual.storage[actual.offsets[i]] ^= 1;
    }
    inline for (.{ "grid_x", "grid_y", "grid_z", "block_x", "block_y", "block_z", "shared_bytes" }) |field_name| {
        @field(node, field_name) += 1;
        try std.testing.expectError(error.SliceFormatGeometry, nodeCheck(node, f, .{ 64, 16, 1 }, 144, &args));
        @field(node, field_name) -= 1;
    }
}
