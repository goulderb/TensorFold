//! Isolated integrated-image math with a saved native shared-down operand.
const std = @import("std");
const cuda = @import("cuda");
const qwen = @import("qwen35_cuda");
const check = @import("check.zig");
const Bindings = @typeInfo(@FieldType(qwen.shared_engine.Engine, "modelopt")).optional.child;
const Router = @FieldType(Bindings, "router");
const Span = struct { base: u64, bytes: usize };
const guard = 64;
const width = 512;
const File = struct { path: []const u8, bytes: usize, sha256: []const u8, role: []const u8 = "", position: usize = 999, label: []const u8 = "" };
const Manifest = struct { schema: u32, image_sha256: []const u8, original_manifest: File, source_journal: File, files: []const File };
const input_labels = [_][]const u8{ "shared_gate.bf16", "shared_up.bf16", "pairs.f32", "weights.f32", "picks.i32", "saved_native_shared_down.bf16" };
const input_sizes = [_]usize{ 1024, 1024, 65536, 36, 32, 4096 };
const output_labels = [_][]const u8{ "legacy_activation", "candidate_activation", "legacy_full", "candidate_full" };
const output_sizes = [_]usize{ 1024, 1024, 4096, 4096 };
const symbols = [_][:0]const u8{ "tf_q35_hf_mlp_act", "tf_q35_hf_shared_act_bf16", "tf_q35_hf_combine", "tf_q35_hf_combine_bf16" };
fn find(m: Manifest, position: usize, label: []const u8, role: []const u8) !File {
    var found: ?File = null;
    for (m.files) |f| {
        if (f.position != position or !std.mem.eql(u8, f.label, label)) continue;
        if (found != null or !std.mem.eql(u8, f.role, role)) return error.IntegratedMoeFixture;
        found = f;
    }
    return found orelse error.IntegratedMoeFixture;
}
fn disjoint(spans: []const Span) !void {
    for (spans, 0..) |left, i| {
        if (left.base == 0 or left.bytes == 0) return error.IntegratedMoeAllocation;
        const end = std.math.add(u64, left.base, left.bytes) catch return error.IntegratedMoeAllocation;
        for (spans[i + 1 ..]) |right| {
            const right_end = std.math.add(u64, right.base, right.bytes) catch return error.IntegratedMoeAllocation;
            if (!(end <= right.base or right_end <= left.base)) return error.IntegratedMoeOverlap;
        }
    }
}
fn allocate(gpu: check.Gpu, position: usize, label: []const u8, bytes: usize, j: anytype) !cuda.DeviceBuffer {
    var result = try cuda.DeviceBuffer.alloc(gpu.d, bytes + 2 * guard);
    errdefer result.free();
    try result.fill8(0xff, null);
    try j.write(.{ .kind = "integrated_moe_allocation", .ordinal = gpu.ctx.device_ordinal, .position = position, .label = label, .base = result.ptr, .payload = result.ptr + guard, .active_bytes = bytes, .allocation_bytes = result.len });
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
    if (sha.len != 64 or !std.mem.eql(u8, &hash(bytes), sha)) return error.IntegratedMoePin;
}
fn read(gpu: check.Gpu, f: File, role: []const u8, size: usize) ![]u8 {
    if (!std.fs.path.isAbsolute(f.path) or f.bytes != size or !std.mem.eql(u8, f.role, role)) return error.IntegratedMoeRole;
    const bytes = try std.Io.Dir.cwd().readFileAlloc(gpu.io, f.path, gpu.gpa, .limited(4 << 20));
    errdefer gpu.gpa.free(bytes);
    if (bytes.len != size) return error.IntegratedMoeSize;
    try pin(bytes, f.sha256);
    return bytes;
}
fn finite(bytes: []const u8) !void {
    if (bytes.len % 2 != 0) return error.IntegratedMoeSize;
    var i: usize = 0;
    while (i < bytes.len) : (i += 2) {
        if (std.mem.readInt(u16, bytes[i..][0..2], .little) & 0x7f80 == 0x7f80) return error.IntegratedMoeNonfinite;
    }
}
fn finiteF32(bytes: []const u8) !void {
    if (bytes.len % 4 != 0) return error.IntegratedMoeSize;
    var i: usize = 0;
    while (i < bytes.len) : (i += 4) {
        if (std.mem.readInt(u32, bytes[i..][0..4], .little) & 0x7f800000 == 0x7f800000) return error.IntegratedMoeNonfinite;
    }
}
fn validateInput(bytes: []const u8, index: usize) !void {
    if (bytes.len != input_sizes[index]) return error.IntegratedMoeSize;
    if (index < 2 or index == 5) return finite(bytes);
    if (index < 4) return finiteF32(bytes);
    var seen: [256]bool = @splat(false);
    for (0..8) |i| {
        const pick = std.mem.readInt(u32, bytes[i * 4 ..][0..4], .little);
        if (pick >= 256 or seen[pick]) return error.IntegratedMoePicks;
        seen[pick] = true;
    }
}
fn guarded(bytes: []const u8, active: usize, numeric: bool) !void {
    if (bytes.len != active + 2 * guard) return error.IntegratedMoeSize;
    for (bytes[0..guard]) |v| if (v != 0xff) return error.IntegratedMoeGuard;
    for (bytes[guard + active ..]) |v| if (v != 0xff) return error.IntegratedMoeGuard;
    if (numeric) try finite(bytes[guard..][0..active]);
}
fn differences(a: []const u8, b: []const u8) !usize {
    if (a.len != b.len or a.len % 2 != 0) return error.IntegratedMoeSize;
    var count: usize = 0;
    var i: usize = 0;
    while (i < a.len) : (i += 2) {
        if (!std.mem.eql(u8, a[i..][0..2], b[i..][0..2])) count += 1;
    }
    return count;
}
fn save(gpu: check.Gpu, j: anytype, position: usize, label: []const u8, bytes: []const u8) !void {
    const name = try std.fmt.allocPrint(gpu.gpa, "integrated_moe_gpu{d}_pos{d}_{s}.bin", .{ gpu.ctx.device_ordinal, position, label });
    defer gpu.gpa.free(name);
    const path = try std.fs.path.join(gpu.gpa, &.{ j.directory, name });
    defer gpu.gpa.free(path);
    const file = try std.Io.Dir.cwd().createFile(gpu.io, path, .{ .exclusive = true });
    defer file.close(gpu.io);
    try file.writeStreamingAll(gpu.io, bytes);
    try j.write(.{ .kind = "integrated_moe_file", .ordinal = gpu.ctx.device_ordinal, .position = position, .label = label, .name = name, .bytes = bytes.len, .sha256 = &hash(bytes) });
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
fn kernelName(f: cuda.Function) ![]const u8 {
    var lib = f.d.lib;
    const get = lib.lookup(*const fn (*?[*:0]const u8, cuda.abi.Function) callconv(.c) cuda.abi.Result, "cuFuncGetName") orelse return error.MissingSymbol;
    var name: ?[*:0]const u8 = null;
    try f.d.check(get(&name, f.handle), "cuFuncGetName");
    return std.mem.span(name orelse return error.MissingSymbol);
}
fn combineArgs(pairs: u64, weights: u64, shared: u64, out: u64, picks: u64, candidate: bool) cuda.Args {
    var a: cuda.Args = .{};
    for ([_]u64{ pairs, weights, shared, out }) |p| a.add(p);
    if (candidate) a.add(picks);
    return a;
}
fn validateNode(p: cuda.abi.KernelNodeParams, f: cuda.Function, a: cuda.Args, arm: usize) ![64]u8 {
    const activation = arm < 2;
    if (p.func != f.handle or p.grid_x != (if (activation) @as(u32, 2) else 8) or p.grid_y != 1 or p.grid_z != 1 or p.block_x != 256 or p.block_y != 1 or p.block_z != 1 or p.shared_bytes != 0 or p.extra != null) return error.IntegratedMoeGeometry;
    const n: usize = if (activation) 6 else if (arm == 2) 4 else 5;
    if (a.count != n) return error.IntegratedMoeArguments;
    const params = p.params orelse return error.IntegratedMoeArguments;
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    for (0..n) |i| {
        const size: u16 = if (activation and i == 5) 4 else 8;
        if (a.sizes[i] != size) return error.IntegratedMoeArguments;
        const actual: [*]const u8 = @ptrCast(params[i] orelse return error.IntegratedMoeArguments);
        if (!std.mem.eql(u8, actual[0..size], a.storage[a.offsets[i]..][0..size])) return error.IntegratedMoeArguments;
        h.update(actual[0..size]);
    }
    var digest: [32]u8 = undefined;
    h.final(&digest);
    return std.fmt.bytesToHex(digest, .lower);
}
fn inspect(gpu: check.Gpu, stream: cuda.Stream, graph: cuda.graph.Graph, f: cuda.Function, a: cuda.Args, position: usize, arm: usize, j: anytype) !void {
    if (try graph.nodeCount() != 1) return error.IntegratedMoeNodes;
    var storage: [1]cuda.graph.Node = undefined;
    const nodes = try graph.nodes(&storage);
    var lib = gpu.d.lib;
    const dependencies = lib.lookup(*const fn (cuda.graph.Node, ?[*]cuda.graph.Node, *usize) callconv(.c) cuda.abi.Result, "cuGraphNodeGetDependencies") orelse return error.MissingSymbol;
    const get_type = lib.lookup(*const fn (cuda.graph.Node, *c_uint) callconv(.c) cuda.abi.Result, "cuGraphNodeGetType") orelse return error.MissingSymbol;
    var count: usize = 0;
    var kind: c_uint = 999;
    try gpu.d.check(dependencies(nodes[0], null, &count), "cuGraphNodeGetDependencies");
    try gpu.d.check(get_type(nodes[0], &kind), "cuGraphNodeGetType");
    if (count != 0 or kind != 0) return error.IntegratedMoeDependencies;
    var p: cuda.abi.KernelNodeParams = undefined;
    try gpu.d.check(gpu.d.api.cuGraphKernelNodeGetParams_v2(nodes[0], &p), "cuGraphKernelNodeGetParams");
    const digest = try validateNode(p, f, a, arm);
    const name = try kernelName(f);
    if (!std.mem.eql(u8, name, symbols[arm])) return error.IntegratedMoeSymbol;
    var values: [6]u64 = @splat(0);
    for (0..a.count) |i| {
        const ptr: [*]const u8 = @ptrCast(p.params.?[i].?);
        for (ptr[0..a.sizes[i]], 0..) |v, shift| values[i] |= @as(u64, v) << @intCast(8 * shift);
    }
    try j.write(.{ .kind = "integrated_moe_node", .ordinal = gpu.ctx.device_ordinal, .position = position, .arm = arm, .symbol = name, .function_handle = @intFromPtr(p.func), .node_handle = @intFromPtr(nodes[0]), .node_type = kind, .nodes = 1, .dependencies = count, .grid = [_]u32{ p.grid_x, p.grid_y, p.grid_z }, .block = [_]u32{ p.block_x, p.block_y, p.block_z }, .shared_bytes = p.shared_bytes, .abi_count = a.count, .argument_sizes = a.sizes[0..a.count], .arguments = values[0..a.count], .arguments_sha256 = &digest, .function_identity_checked = true });
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
pub fn run(gpu: check.Gpu, folder: []const u8, manifest_sha: []const u8, image_path: []const u8, image_sha: []const u8, j: anytype) !void {
    return runInner(gpu, folder, manifest_sha, image_path, image_sha, j) catch |err| {
        j.write(.{ .kind = "integrated_moe_failure", .ordinal = gpu.ctx.device_ordinal, .failure = @errorName(err) }) catch {};
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
    if (m.schema != 1 or m.files.len != 66 or !std.mem.eql(u8, m.image_sha256, image_sha) or !std.mem.eql(u8, image_sha, "57f2598e2d9dc7b6a9a4731374ac4afb740e48a793f5728e9c6c2b5432eb67ca") or !std.mem.eql(u8, m.original_manifest.sha256, "2e2813cb00c9a351d48e4b9b2f20eb5533509f02e26ffd931ff12d16273fcfe9")) return error.IntegratedMoeManifest;
    const origin = try read(gpu, m.original_manifest, "", m.original_manifest.bytes);
    defer gpu.gpa.free(origin);
    const prior_journal = try read(gpu, m.source_journal, "", m.source_journal.bytes);
    defer gpu.gpa.free(prior_journal);
    const image = try read(gpu, .{ .path = image_path, .sha256 = image_sha, .bytes = 162424 }, "", 162424);
    defer gpu.gpa.free(image);
    const aligned = try gpu.gpa.alignedAlloc(u8, .@"8", image.len);
    defer gpu.gpa.free(aligned);
    @memcpy(aligned, image);
    var module = try cuda.Module.load(gpu.d, aligned);
    defer module.unload();
    const routers = [_]Router{ try Router.resolveWithMath(module, .legacy), try Router.resolveWithMath(module, .bf16_boundaries) };
    const fns = [_]cuda.Function{ try module.function(symbols[0]), try module.function(symbols[1]), routers[0].combine_fn, routers[1].combine_fn };
    for (fns, 0..) |f, arm| {
        if (!std.mem.eql(u8, try kernelName(f), symbols[arm])) return error.IntegratedMoeSymbol;
        const local = try f.attribute(.local_size_bytes);
        const shared = try f.attribute(.shared_size_bytes);
        if (local != 0 or shared != 0) return error.IntegratedMoeResources;
        try j.write(.{ .kind = "integrated_moe_resource", .ordinal = gpu.ctx.device_ordinal, .arm = arm, .symbol = try kernelName(f), .function_handle = @intFromPtr(f.handle), .registers = try f.attribute(.num_regs), .local_bytes = local, .static_shared = shared });
    }
    var stream = try cuda.Stream.init(gpu.d, true);
    defer stream.deinit();
    try j.write(.{ .kind = "integrated_moe_start", .ordinal = gpu.ctx.device_ordinal, .manifest_sha256 = manifest_sha, .image_sha256 = image_sha, .original_manifest_sha256 = m.original_manifest.sha256, .source_journal_sha256 = m.source_journal.sha256, .rows = 6, .expected_hf_device_inputs = false, .scope = "isolated integrated-image fixed-operand math; no fresh complete projection chain or full-model claim" });
    for (0..6) |position| {
        var inputs: [6]cuda.DeviceBuffer = undefined;
        var out: [4]cuda.DeviceBuffer = undefined;
        var ni: usize = 0;
        var no: usize = 0;
        defer for (inputs[0..ni]) |*b| b.free();
        defer for (out[0..no]) |*b| b.free();
        var spans: [10]Span = undefined;
        for (&inputs, input_labels, input_sizes, 0..) |*b, label, size, index| {
            const data = try read(gpu, try find(m, position, label, "device_native_input"), "device_native_input", size);
            defer gpu.gpa.free(data);
            try validateInput(data, index);
            b.* = try allocate(gpu, position, label, size, j);
            ni += 1;
            spans[index] = .{ .base = b.ptr, .bytes = b.len };
            try b.upload(guard, data);
        }
        for (&out, output_labels, output_sizes, 0..) |*b, label, size, index| {
            b.* = try allocate(gpu, position, label, size, j);
            no += 1;
            spans[6 + index] = .{ .base = b.ptr, .bytes = b.len };
        }
        try disjoint(&spans);
        for (fns, 0..) |f, arm| {
            var a = if (arm < 2) actArgs(inputs[0].ptr + guard, inputs[1].ptr + guard, out[arm].ptr + guard) else combineArgs(inputs[2].ptr + guard, inputs[3].ptr + guard, inputs[5].ptr + guard, out[arm].ptr + guard, inputs[4].ptr + guard, arm == 3);
            try cuda.graph.beginCapture(stream, .thread_local);
            if (arm < 2) {
                cuda.launch.launch(f, .{ .grid = .{ .x = 2 }, .block = .{ .x = 256 } }, stream, &a) catch |err| {
                    abortCapture(stream);
                    return err;
                };
            } else {
                routers[arm - 2].combine(stream, inputs[2].ptr + guard, inputs[3].ptr + guard, inputs[5].ptr + guard, out[arm].ptr + guard, inputs[4].ptr + guard) catch |err| {
                    abortCapture(stream);
                    return err;
                };
            }
            var graph = try cuda.graph.endCapture(stream);
            defer graph.deinit();
            try inspect(gpu, stream, graph, f, a, position, arm, j);
        }
        var reads: [10][]u8 = undefined;
        var nr: usize = 0;
        defer for (reads[0..nr]) |data| gpu.gpa.free(data);
        for (out, output_labels, 0..) |b, label, index| {
            reads[index] = try check.download(gpu, b);
            nr += 1;
            try save(gpu, j, position, label, reads[index]);
        }
        for (inputs, input_labels, 0..) |b, label, index| {
            reads[4 + index] = try check.download(gpu, b);
            nr += 1;
            const read_label = try std.fmt.allocPrint(gpu.gpa, "{s}_readback", .{label});
            defer gpu.gpa.free(read_label);
            try save(gpu, j, position, read_label, reads[4 + index]);
        }
        for (reads[0..4], output_labels, output_sizes) |data, label, size| {
            try guarded(data, size, true);
            const target = try read(gpu, try find(m, position, label, "host_target"), "host_target", size);
            defer gpu.gpa.free(target);
            if (!std.mem.eql(u8, data[guard..][0..size], target)) return error.IntegratedMoeMismatch;
        }
        for (reads[4..], input_labels, input_sizes, 0..) |data, label, size, index| {
            try guarded(data, size, false);
            try validateInput(data[guard..][0..size], index);
            const target = try read(gpu, try find(m, position, label, "device_native_input"), "device_native_input", size);
            defer gpu.gpa.free(target);
            if (!std.mem.eql(u8, data[guard..][0..size], target)) return error.IntegratedMoeInputMutation;
        }
        const hf = try read(gpu, try find(m, position, "actual_hf_full", "host_target"), "host_target", 4096);
        defer gpu.gpa.free(hf);
        const diff = try differences(reads[3][guard..][0..4096], hf);
        const eligible = position >= 2 and position <= 4;
        if (diff != ([_]usize{ 2, 1998, 0, 0, 0, 0 })[position]) return error.IntegratedMoeDiagnostic;
        try j.write(.{ .kind = "integrated_moe_case", .ordinal = gpu.ctx.device_ordinal, .position = position, .raw_files = 10, .live_allocations = 10, .live_allocations_disjoint = true, .actual_hf_final_gate_eligible = eligible, .candidate_final_differences_vs_actual_hf = diff, .correctness_pass = true });
    }
    try j.write(.{ .kind = "integrated_moe_complete", .ordinal = gpu.ctx.device_ordinal, .positions = 6, .raw_files = 60, .full_network_hf_gate = "unchanged" });
}
test "integrated complete ABI and geometry detect every argument mutation" {
    for (0..4) |arm| {
        var a = if (arm < 2) actArgs(100, 200, 300) else combineArgs(100, 200, 300, 400, 500, arm == 3);
        var p: cuda.abi.KernelNodeParams = std.mem.zeroes(cuda.abi.KernelNodeParams);
        p.grid_x = if (arm < 2) 2 else 8;
        p.grid_y = 1;
        p.grid_z = 1;
        p.block_x = 256;
        p.block_y = 1;
        p.block_z = 1;
        p.params = a.pointers();
        const f: cuda.Function = .{ .d = undefined, .handle = p.func };
        const want = a;
        _ = try validateNode(p, f, want, arm);
        for (0..a.count) |i| {
            a.storage[a.offsets[i]] ^= 1;
            try std.testing.expectError(error.IntegratedMoeArguments, validateNode(p, f, want, arm));
            a.storage[a.offsets[i]] ^= 1;
        }
        p.shared_bytes = 1;
        try std.testing.expectError(error.IntegratedMoeGeometry, validateNode(p, f, want, arm));
    }
}
test "integrated saved native shared-down is BF16 and all allocations disjoint" {
    var data: [4096]u8 = @splat(0);
    try validateInput(&data, 5);
    data[1] = 0x7f;
    data[0] = 0x80;
    try std.testing.expectError(error.IntegratedMoeNonfinite, validateInput(&data, 5));
    try std.testing.expectError(error.IntegratedMoeOverlap, disjoint(&.{ .{ .base = 100, .bytes = 100 }, .{ .base = 150, .bytes = 100 } }));
}
