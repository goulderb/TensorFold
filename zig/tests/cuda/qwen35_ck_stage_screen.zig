//! Local isolated stage-depth screen, adapting the existing TensorFold row-linear
//! screen and CK dispatch. Original CK kernels and runtime policy are unchanged.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const qwen = @import("qwen35_cuda");
const check = @import("check.zig");
const floor_bytes = 1536 << 20;
const repeats = 4;
const guard = 64;
const row_counts = [_]usize{ 1, 2, 16 };
const scope = "mixed 26 real FP8 projection banks, footprint exceeds twice device L2; preencoded inputs; operator only";
const Kind = enum { gdn_qkv, gdn_z, gdn_out, attn_q, attn_k, attn_v, attn_o };
const Bank = struct {
    layer: usize,
    kind: Kind,
    w: cuda.qmmf.Weight,
    encoded: cuda.DeviceBuffer,
    output: cuda.DeviceBuffer,
    reference: []u8,
    quant: []u8,
    weight_sha256: [64]u8,
    input_sha256: [64]u8,
    fn deinit(b: *Bank, a: std.mem.Allocator) void {
        b.encoded.free();
        b.output.free();
        a.free(b.reference);
        a.free(b.quant);
    }
};
fn hash(bytes: []const u8) [64]u8 {
    var d: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &d, .{});
    return std.fmt.bytesToHex(d, .lower);
}
fn floor(gpu: check.Gpu) !usize {
    const free = (try gpu.ctx.memInfo()).free;
    if (free < floor_bytes) return error.CkStageMemoryFloor;
    return free;
}
fn kernelName(f: cuda.Function) ![]const u8 {
    var lib = f.d.lib;
    const get = lib.lookup(*const fn (*?[*:0]const u8, cuda.abi.Function) callconv(.c) cuda.abi.Result, "cuFuncGetName") orelse return error.MissingSymbol;
    var name: ?[*:0]const u8 = null;
    try f.d.check(get(&name, f.handle), "cuFuncGetName");
    return std.mem.span(name orelse return error.MissingSymbol);
}
fn validateCluster(v: cuda.abi.LaunchAttributeValue, policy: cuda.abi.LaunchAttributeValue, sk: u32) !void {
    if (v.cluster_dim.x != 1 or v.cluster_dim.y != 1 or v.cluster_dim.z != sk or policy.int != 0) return error.CkStageClusterAttributes;
}
fn inputBits(row: usize, column: usize, layer: usize) u16 {
    var x: u32 = @truncate((row + 1) *% 0x9e3779b1 +% column *% 0x85ebca6b +% layer *% 65537);
    x ^= x >> 16;
    x *%= 0x7feb352d;
    x ^= x >> 15;
    return @as(u16, @truncate((x >> 16) & 0x8000)) | 0x3f00 | @as(u16, @truncate(x & 0xff));
}
fn validateOutput(actual: []const u8, reference: []const u8) !void {
    if (actual.len < reference.len or reference.len % 2 != 0) return error.CkStageOutputSize;
    var i: usize = 0;
    while (i < reference.len) : (i += 2) {
        if (std.mem.readInt(u16, actual[i..][0..2], .little) & 0x7f80 == 0x7f80) return error.CkStageNonfinite;
    }
    if (!std.mem.eql(u8, actual[0..reference.len], reference)) return error.CkStageMismatch;
    for (actual[reference.len..]) |v| if (v != 0xff) return error.CkStageGuard;
}
fn args(b: Bank, rows: usize, part: u64) cuda.Args {
    var a: cuda.Args = .{};
    a.add(b.encoded.ptr);
    a.add(b.w.codes);
    a.add(cuda.ck.alpha(b.w.act, b.w.scale));
    a.add(b.output.ptr);
    a.add(part);
    for ([_]usize{ rows, b.w.n, b.w.k, cuda.qmmf.splitK(b.w.n, b.w.k), 1 }) |v| a.add(@as(i32, @intCast(v)));
    return a;
}
fn call(b: Bank, s: cuda.Stream, functions: [2]cuda.Function, stage: usize, rows: usize, part: u64) !void {
    if (rows == 0 or rows > 16 or (stage != 3 and stage != 4)) return error.CkStageAdmission;
    const sk = cuda.qmmf.splitK(b.w.n, b.w.k);
    if (sk > 8) return error.CkStageClusterRequired;
    var a = args(b, rows, part);
    try cuda.launch.launch(functions[@intFromBool(sk > 1)], .{
        .grid = .{ .x = @intCast((b.w.n + 63) / 64), .y = 1, .z = sk },
        .block = .{ .x = 128, .y = 1, .z = 1 },
        .shared = @intCast(stage * 5120),
        .cluster = if (sk > 1) .{ .x = 1, .y = 1, .z = sk } else null,
    }, s, &a);
}
fn save(gpu: check.Gpu, j: anytype, label: []const u8, bank: usize, rows: usize, bytes: []const u8) !void {
    const name = try std.fmt.allocPrint(gpu.gpa, "ck_stage_gpu{d}_bank{d}_rows{d}_{s}.bin", .{ gpu.ctx.device_ordinal, bank, rows, label });
    defer gpu.gpa.free(name);
    const path = try std.fs.path.join(gpu.gpa, &.{ j.directory, name });
    defer gpu.gpa.free(path);
    const file = try std.Io.Dir.cwd().createFile(gpu.io, path, .{ .exclusive = true });
    defer file.close(gpu.io);
    try file.writeStreamingAll(gpu.io, bytes);
    try j.write(.{ .kind = "ck_stage_file", .ordinal = gpu.ctx.device_ordinal, .name = name, .bank = bank, .rows = rows, .bytes = bytes.len, .sha256 = &hash(bytes) });
}
fn addBank(gpu: check.Gpu, e: *qwen.shared_engine.Engine, banks: *std.ArrayList(Bank), layer: usize, kind: Kind, projection: qwen.shared_engine.Proj, part: cuda.DeviceBuffer, j: anytype) !void {
    if (projection != .linear or projection.linear != .fp8) return error.CkStageFormat;
    const w = projection.linear.fp8;
    if (w.mode != .fp8 or !std.math.isFinite(w.act) or w.act <= 0 or !std.math.isFinite(w.scale) or w.scale <= 0) return error.CkStageScale;
    const n = 16 * w.n * 2;
    const input = try gpu.gpa.alloc(u8, 16 * w.k * 2);
    defer gpu.gpa.free(input);
    for (0..16) |r| for (0..w.k) |c| std.mem.writeInt(u16, input[(r * w.k + c) * 2 ..][0..2], inputBits(r, c, layer), .little);
    var x = try cuda.DeviceBuffer.fromHost(gpu.d, input);
    defer x.free();
    var b: Bank = .{ .layer = layer, .kind = kind, .w = w, .encoded = try cuda.DeviceBuffer.alloc(gpu.d, 16 * w.k + guard), .output = undefined, .reference = undefined, .quant = undefined, .weight_sha256 = undefined, .input_sha256 = hash(input) };
    errdefer b.encoded.free();
    b.output = try cuda.DeviceBuffer.alloc(gpu.d, n + guard);
    errdefer b.output.free();
    b.reference = try gpu.gpa.alloc(u8, n);
    errdefer gpu.gpa.free(b.reference);
    b.quant = try gpu.gpa.alloc(u8, b.encoded.len);
    errdefer gpu.gpa.free(b.quant);
    const packed_bytes_host = try gpu.gpa.alloc(u8, @as(usize, w.npad) * w.k);
    defer gpu.gpa.free(packed_bytes_host);
    try e.arena.download(w.codes - e.arena.ptr, packed_bytes_host);
    b.weight_sha256 = hash(packed_bytes_host);
    for (banks.items) |prior| if (std.mem.eql(u8, &prior.weight_sha256, &b.weight_sha256)) return error.CkStageDuplicateWeight;
    const ck = &e.modelopt.?.ck;
    try b.encoded.fill8(0xff, e.stream.handle);
    try b.output.fill8(0xff, e.stream.handle);
    try ck.quant8(e.stream, x.ptr, w.k, 16, w.k, w.act, b.encoded.ptr);
    for (0..16) |r| try ck.lane8(e.stream, b.encoded.ptr + r * w.k, 1, w, w.act, b.output.ptr + r * w.n * 2, part.ptr);
    try e.stream.synchronize();
    try b.output.download(0, b.reference);
    try b.encoded.download(0, b.quant);
    for (b.quant[0 .. 16 * w.k]) |v| if (v & 0x7f == 0x7f) return error.CkStageNonfinite;
    for (b.quant[16 * w.k ..]) |v| if (v != 0xff) return error.CkStageGuard;
    try validateOutput(b.reference, b.reference);
    for (0..16) |r| for (0..r) |p| if (std.mem.eql(u8, b.reference[r * w.n * 2 ..][0 .. w.n * 2], b.reference[p * w.n * 2 ..][0 .. w.n * 2])) return error.CkStageIdenticalRows;
    try save(gpu, j, "reference", banks.items.len, 16, b.reference);
    try save(gpu, j, "encoded", banks.items.len, 16, b.quant);
    try j.write(.{ .kind = "ck_stage_bank", .ordinal = gpu.ctx.device_ordinal, .bank = banks.items.len, .layer = layer, .projection = @tagName(kind), .weight = w, .packed_bytes = packed_bytes_host.len, .packed_sha256 = &b.weight_sha256, .input_sha256 = &b.input_sha256, .encoded_sha256 = &hash(b.quant), .reference_sha256 = &hash(b.reference), .reference_distinct_rows = 16, .encoded_ptr = b.encoded.ptr, .output_ptr = b.output.ptr, .part_ptr = part.ptr, .alpha_bits = @as(u32, @bitCast(cuda.ck.alpha(w.act, w.scale))) });
    try banks.append(gpu.gpa, b);
}
fn verify(gpu: check.Gpu, banks: []const Bank, part: cuda.DeviceBuffer, rows: usize, stage: usize, label: []const u8, dump: bool, j: anytype) !void {
    for (banks, 0..) |b, i| {
        const bytes = try gpu.gpa.alloc(u8, b.output.len);
        defer gpu.gpa.free(bytes);
        try b.output.download(0, bytes);
        validateOutput(bytes, b.reference[0 .. rows * b.w.n * 2]) catch |err| {
            try save(gpu, j, "failure_output", i, rows, bytes);
            return err;
        };
        const quant = try gpu.gpa.alloc(u8, b.quant.len);
        defer gpu.gpa.free(quant);
        try b.encoded.download(0, quant);
        if (!std.mem.eql(u8, quant, b.quant)) {
            try save(gpu, j, "failure_encoded", i, rows, quant);
            return error.CkStageEncodedChanged;
        }
        if (dump) try save(gpu, j, label, i, rows, bytes);
    }
    var scratch: [guard]u8 = undefined;
    try part.download(0, &scratch);
    for (scratch) |v| if (v != 0xff) {
        try save(gpu, j, "failure_partial", 0, rows, &scratch);
        return error.CkStagePartialWrite;
    };
    try j.write(.{ .kind = "ck_stage_check", .ordinal = gpu.ctx.device_ordinal, .rows = rows, .stage = stage, .phase = label, .all_banks_byte_exact = true, .encoded_unchanged = true, .finite = true, .inactive_guard_and_part_untouched = true });
}
fn capture(gpu: check.Gpu, banks: []const Bank, s: cuda.Stream, functions: [2]cuda.Function, stage: usize, rows: usize, part: u64, j: anytype) !cuda.graph.Exec {
    try cuda.graph.beginCapture(s, .thread_local);
    for (0..repeats) |_| for (banks) |b| call(b, s, functions, stage, rows, part) catch |err| {
        if (cuda.graph.endCapture(s)) |value| {
            var dead = value;
            dead.deinit();
        } else |_| {}
        return err;
    };
    var graph = try cuda.graph.endCapture(s);
    defer graph.deinit();
    const count = repeats * banks.len;
    if (try graph.nodeCount() != count) return error.CkStageNodeCount;
    const storage = try gpu.gpa.alloc(cuda.graph.Node, count);
    defer gpu.gpa.free(storage);
    const nodes = try graph.nodes(storage);
    const predecessors = try gpu.gpa.alloc(cuda.graph.Node, count);
    defer gpu.gpa.free(predecessors);
    var lib = gpu.d.lib;
    const get_deps = lib.lookup(*const fn (cuda.abi.GraphNode, ?[*]cuda.abi.GraphNode, *usize) callconv(.c) cuda.abi.Result, "cuGraphNodeGetDependencies") orelse return error.MissingSymbol;
    const get_attr = lib.lookup(*const fn (cuda.abi.GraphNode, cuda.abi.LaunchAttributeId, *cuda.abi.LaunchAttributeValue) callconv(.c) cuda.abi.Result, "cuGraphKernelNodeGetAttribute") orelse return error.MissingSymbol;
    for (nodes, predecessors) |node, *previous| {
        var deps: [2]cuda.graph.Node = .{ null, null };
        var n: usize = 2;
        try gpu.d.check(get_deps(node, &deps, &n), "cuGraphNodeGetDependencies");
        if (n > 1) return error.CkStageDependencies;
        previous.* = if (n == 0) null else deps[0];
    }
    var prior: cuda.graph.Node = null;
    for (0..count) |index| {
        var found: ?usize = null;
        for (predecessors, 0..) |prev, n| if (prev == prior) {
            if (found != null) return error.CkStageDependencies;
            found = n;
        };
        const at = found orelse return error.CkStageDependencies;
        const b = banks[index % banks.len];
        const sk = cuda.qmmf.splitK(b.w.n, b.w.k);
        var p: cuda.abi.KernelNodeParams = undefined;
        try gpu.d.check(gpu.d.api.cuGraphKernelNodeGetParams_v2(nodes[at], &p), "cuGraphKernelNodeGetParams");
        if (p.func != functions[@intFromBool(sk > 1)].handle or p.grid_x != (b.w.n + 63) / 64 or p.grid_y != 1 or p.grid_z != sk or p.block_x != 128 or p.block_y != 1 or p.block_z != 1 or p.shared_bytes != stage * 5120) return error.CkStageGeometry;
        const want = args(b, rows, part);
        const actual = p.params orelse return error.CkStageArguments;
        var h = std.crypto.hash.sha2.Sha256.init(.{});
        for (0..want.count) |n| {
            const ptr: [*]const u8 = @ptrCast(actual[n] orelse return error.CkStageArguments);
            const bytes = want.storage[want.offsets[n]..][0..want.sizes[n]];
            if (!std.mem.eql(u8, ptr[0..bytes.len], bytes)) return error.CkStageArguments;
            h.update(ptr[0..bytes.len]);
        }
        var digest: [32]u8 = undefined;
        h.final(&digest);
        var cluster_value: cuda.abi.LaunchAttributeValue = .{ .pad = @splat(0) };
        var policy_value: cuda.abi.LaunchAttributeValue = .{ .pad = @splat(0) };
        if (sk > 1) {
            try gpu.d.check(get_attr(nodes[at], .cluster_dimension, &cluster_value), "cuGraphKernelNodeGetAttribute");
            try gpu.d.check(get_attr(nodes[at], .cluster_scheduling_policy_preference, &policy_value), "cuGraphKernelNodeGetAttribute");
            try validateCluster(cluster_value, policy_value, sk);
        }
        try j.write(.{ .kind = "ck_stage_node", .ordinal = gpu.ctx.device_ordinal, .stage = stage, .rows = rows, .index = index, .bank = index % banks.len, .grid = [_]u32{ p.grid_x, p.grid_y, p.grid_z }, .block = [_]u32{ p.block_x, p.block_y, p.block_z }, .shared_bytes = p.shared_bytes, .abi_count = want.count, .arguments_sha256 = &std.fmt.bytesToHex(digest, .lower), .function_identity_checked = true, .serial_dependency_checked = true, .cluster_requested = [_]u32{ 1, 1, sk }, .cluster_actual = [_]u32{ cluster_value.cluster_dim.x, cluster_value.cluster_dim.y, cluster_value.cluster_dim.z }, .cluster_policy = policy_value.int, .cluster_attributes_readback = sk > 1 });
        prior = nodes[at];
    }
    var exec = try graph.instantiate();
    errdefer exec.deinit();
    try exec.upload(s);
    try s.synchronize();
    return exec;
}
pub fn run(gpu: check.Gpu, model_dir: []const u8, image_path: []const u8, image_sha256: []const u8, j: anytype) !void {
    return runInner(gpu, model_dir, image_path, image_sha256, j) catch |err| {
        j.write(.{ .kind = "ck_stage_failure", .ordinal = gpu.ctx.device_ordinal, .failure = @errorName(err), .correctness_pass = false }) catch {};
        return err;
    };
}
fn runInner(gpu: check.Gpu, model_dir: []const u8, image_path: []const u8, image_sha256: []const u8, j: anytype) !void {
    const image = try std.Io.Dir.cwd().readFileAlloc(gpu.io, image_path, gpu.gpa, .limited(16 << 20));
    defer gpu.gpa.free(image);
    if (image_sha256.len != 64 or !std.mem.eql(u8, &hash(image), image_sha256)) return error.CkStageImagePin;
    const aligned = try gpu.gpa.alignedAlloc(u8, .@"8", image.len);
    defer gpu.gpa.free(aligned);
    @memcpy(aligned, image);
    var module = try cuda.Module.load(gpu.d, aligned);
    defer module.unload();
    const candidate = [2]cuda.Function{ try module.function("tf_ck8_stage3_16"), try module.function("tf_ck8_stage3_16c") };
    const config = try qwen.checkpoint.Config.read(gpu.gpa, gpu.io, model_dir);
    const admitted: qwen.checkpoint.Admitted = .{ .geometry = config.geometry, .encoding = .{ .modelopt = config } };
    const placement: qwen.checkpoint.Placement = .{ .first = 1, .end = 9, .total = 40 };
    var source = try core.Checkpoint.openModel(gpu.gpa, gpu.io, model_dir);
    const plan = qwen.weights.plan(&source, admitted, placement) catch |err| {
        source.close();
        return err;
    };
    source.close();
    const scratch = try qwen.shared_engine.Engine.scratchBytes(admitted, placement);
    if (try floor(gpu) < plan.weights_bytes + plan.staging_bytes + scratch + (32 << 20) + floor_bytes) return error.CkStageMemoryFloor;
    const e = try qwen.shared_engine.Engine.initPlaced(gpu.gpa, gpu.io, gpu.ctx, model_dir, admitted, placement, false);
    defer e.deinit();
    if (!e.modelopt.?.ck.clusters) return error.CkStageClusterRequired;
    const baseline = [2]cuda.Function{ e.modelopt.?.ck.lane[0], e.modelopt.?.ck.lane[1] };
    for ([_][2]cuda.Function{ baseline, candidate }, 0..) |functions, arm| for (functions, 0..) |f, cluster| {
        const local = try f.attribute(.local_size_bytes);
        if (local != 0) return error.CkStageLocalMemory;
        try j.write(.{ .kind = "ck_stage_resource", .ordinal = gpu.ctx.device_ordinal, .stage = 4 - arm, .cluster = cluster == 1, .symbol = try kernelName(f), .registers = try f.attribute(.num_regs), .local_bytes = local, .static_shared = try f.attribute(.shared_size_bytes), .dynamic_shared = (4 - arm) * 5120, .threads = 128, .resident_blocks_per_sm = try f.occupancy(128, (4 - arm) * 5120) });
    };
    var part = try cuda.DeviceBuffer.alloc(gpu.d, guard);
    defer part.free();
    try part.fill8(0xff, e.stream.handle);
    var banks: std.ArrayList(Bank) = .empty;
    defer {
        for (banks.items) |*b| b.deinit(gpu.gpa);
        banks.deinit(gpu.gpa);
    }
    for (e.layers, 1..) |layer, global| switch (layer.mixer) {
        .delta => |d| {
            try addBank(gpu, e, &banks, global, .gdn_qkv, d.qkv, part, j);
            try addBank(gpu, e, &banks, global, .gdn_z, d.z, part, j);
            try addBank(gpu, e, &banks, global, .gdn_out, d.out, part, j);
        },
        .attention => |a| {
            try addBank(gpu, e, &banks, global, .attn_q, a.q, part, j);
            try addBank(gpu, e, &banks, global, .attn_k, a.k, part, j);
            try addBank(gpu, e, &banks, global, .attn_v, a.v, part, j);
            try addBank(gpu, e, &banks, global, .attn_o, a.o, part, j);
        },
    };
    var packed_bytes: usize = 0;
    for (banks.items) |b| packed_bytes += @as(usize, b.w.npad) * b.w.k;
    const l2 = try gpu.ctx.attribute(.l2_cache_size);
    if (banks.items.len != 26 or l2 <= 0 or packed_bytes <= 2 * @as(usize, @intCast(l2))) return error.CkStageColdFootprint;
    try j.write(.{ .kind = "ck_stage_start", .ordinal = gpu.ctx.device_ordinal, .banks = banks.items.len, .packed_weight_bytes = packed_bytes, .l2_bytes = l2, .first_layer = 1, .end_layer = 9, .rows = row_counts, .repeats = repeats, .image_sha256 = image_sha256, .scope = scope, .free_bytes = try floor(gpu) });
    var graphs: [6]cuda.graph.Exec = undefined;
    var made: usize = 0;
    defer for (graphs[0..made]) |*g| g.deinit();
    for (row_counts, 0..) |rows, ri| for ([_][2]cuda.Function{ baseline, candidate }, 0..) |functions, arm| {
        for (banks.items) |b| try b.output.fill8(0xff, e.stream.handle);
        graphs[2 * ri + arm] = try capture(gpu, banks.items, e.stream, functions, 4 - arm, rows, part.ptr, j);
        made += 1;
        try graphs[2 * ri + arm].launchOn(e.stream);
        try e.stream.synchronize();
        try verify(gpu, banks.items, part, rows, 4 - arm, if (arm == 0) "stage4" else "stage3", true, j);
    };
    var start = try cuda.Event.init(gpu.d, true);
    defer start.deinit();
    var end = try cuda.Event.init(gpu.d, true);
    defer end.deinit();
    for (row_counts, 0..) |rows, ri| for (0..6) |pair| for (0..2) |order| {
        const arm = if (pair % 2 == 0) order else 1 - order;
        for (banks.items) |b| try b.output.fill8(0xff, e.stream.handle);
        try e.stream.synchronize();
        const free_before = try floor(gpu);
        const begin = check.now(gpu.io);
        try start.record(e.stream);
        try graphs[2 * ri + arm].launchOn(e.stream);
        try end.record(e.stream);
        try end.synchronize();
        const wall_ns = check.now(gpu.io) - begin;
        const event_ms = try cuda.Event.elapsedMs(start, end);
        if (wall_ns <= 0 or !std.math.isFinite(event_ms) or event_ms <= 0) return error.CkStageTiming;
        try verify(gpu, banks.items, part, rows, 4 - arm, "timed", false, j);
        try j.write(.{ .kind = "ck_stage_sample", .ordinal = gpu.ctx.device_ordinal, .stage = 4 - arm, .rows = rows, .pair = pair, .order = order, .warmup = pair == 0, .calls = repeats * banks.items.len, .repeats = repeats, .event_ms = event_ms, .wall_ns = wall_ns, .event_us_per_mixed_pass = @as(f64, event_ms) * 1000 / repeats, .wall_us_per_mixed_pass = @as(f64, @floatFromInt(wall_ns)) / 1000 / repeats, .free_before = free_before, .free_after = try floor(gpu), .all_output_bytes_exact = true, .scope = scope });
    };
    try j.write(.{ .kind = "ck_stage_complete", .ordinal = gpu.ctx.device_ordinal, .correctness_pass = true, .banks = banks.items.len, .free_bytes = try floor(gpu), .scope = scope });
}
test "stage screen finite mismatch and poison controls" {
    const ref = [_]u8{ 0, 0x3f };
    var actual = [_]u8{ 0, 0x3f, 0xff };
    try validateOutput(&actual, &ref);
    actual[0] = 1;
    try std.testing.expectError(error.CkStageMismatch, validateOutput(&actual, &ref));
    actual = .{ 0xff, 0xff, 0xff };
    try std.testing.expectError(error.CkStageNonfinite, validateOutput(&actual, &ref));
    actual = .{ 0, 0x3f, 0 };
    try std.testing.expectError(error.CkStageGuard, validateOutput(&actual, &ref));
}
test "stage screen CUDA cluster ABI and mutated dimensions policy" {
    try std.testing.expectEqual(@as(usize, 64), @sizeOf(cuda.abi.LaunchAttributeValue));
    try std.testing.expectEqual(@as(usize, 8), @alignOf(cuda.abi.LaunchAttributeValue));
    try std.testing.expectEqual(@as(u32, 4), @backingInt(cuda.abi.LaunchAttributeId.cluster_dimension));
    try std.testing.expectEqual(@as(u32, 5), @backingInt(cuda.abi.LaunchAttributeId.cluster_scheduling_policy_preference));
    var v: cuda.abi.LaunchAttributeValue = .{ .cluster_dim = .{ .x = 1, .y = 1, .z = 4 } };
    var policy: cuda.abi.LaunchAttributeValue = .{ .int = 0 };
    try validateCluster(v, policy, 4);
    v.cluster_dim.x = 2;
    try std.testing.expectError(error.CkStageClusterAttributes, validateCluster(v, policy, 4));
    v.cluster_dim.x = 1;
    v.cluster_dim.z = 2;
    try std.testing.expectError(error.CkStageClusterAttributes, validateCluster(v, policy, 4));
    v.cluster_dim.z = 4;
    policy.int = 1;
    try std.testing.expectError(error.CkStageClusterAttributes, validateCluster(v, policy, 4));
}
