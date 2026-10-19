const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");

const width = 2048;
const groups = 128;
const banks = 8;
const floor_bytes = 1536 * 1024 * 1024;
const act: f32 = 0.125;
const seed: u32 = 0x351019ad;
const Arm = enum { baseline, shared_quant };
const Case = struct {
    name: []const u8,
    outputs: []const usize,
    per_partition: usize,

    fn columns(self: Case) usize {
        var sum: usize = 0;
        for (self.outputs) |n| sum += n;
        return sum;
    }
    fn weightBytes(self: Case) usize {
        return self.columns() * width;
    }
    fn outputBytes(self: Case) usize {
        return groups * self.columns() * 2;
    }
};
const cases = [_]Case{
    .{ .name = "qkv_z_pair", .outputs = &.{ 8192, 4096 }, .per_partition = 15 },
    .{ .name = "q_k_v_triple", .outputs = &.{ 8192, 512, 512 }, .per_partition = 5 },
};
pub const scope = "synthetic captured FP8 input-quantization seam; current-fork equivalence only; independent HF full-logit gate remains failed";

const InputKey = struct {
    ptr: u64,
    rows: usize = 1,
    columns: usize = width,
    stride: usize = width,
    scale: f32 = act,

    fn eligible(self: InputKey, other: InputKey) bool {
        return self.ptr == other.ptr and self.rows == other.rows and self.columns == other.columns and
            self.stride == other.stride and self.scale > 0 and std.math.isFinite(self.scale) and
            other.scale > 0 and std.math.isFinite(other.scale) and
            @as(u32, @bitCast(cuda.ck.inverse(self.scale))) == @as(u32, @bitCast(cuda.ck.inverse(other.scale)));
    }
};

fn random(state: *u32) u32 {
    state.* ^= state.* << 13;
    state.* ^= state.* >> 17;
    state.* ^= state.* << 5;
    return state.*;
}
fn weights(bytes: []u8, bank: usize, projection: usize) void {
    var state = seed ^ @as(u32, @intCast(1 + bank * 131 + projection * 8191));
    for (bytes) |*v| {
        const bits = random(&state);
        v.* = @as(u8, @intCast(bits % 127)) | @as(u8, @intCast((bits >> 16) & 128));
    }
}
fn input(a: std.mem.Allocator, payload: usize) ![]u8 {
    const bytes = try a.alloc(u8, groups * width * 2);
    var state = seed ^ @as(u32, @intCast(911 + payload * 353));
    const edges = [_]f32{ 0, -0.0, 0.0001220703125, -0.0001220703125, 0.125, 0.1328125, 0.140625, -0.1328125, 54, 56, 58, 60, -54, -56, -58, -60 };
    for (0..groups) |g| for (0..width) |i| {
        const value: f32 = if (i < edges.len) edges[(i + g + payload) % edges.len] else @as(f32, @floatFromInt(@as(i32, @intCast(random(&state) % 16385)) - 8192)) / 1024;
        const bits: u16 = @truncate(@as(u32, @bitCast(value)) >> 16);
        std.mem.writeInt(u16, bytes[(g * width + i) * 2 ..][0..2], bits, .little);
    };
    return bytes;
}
fn finite(bytes: []const u8) !void {
    if (bytes.len % 2 != 0) return error.InvalidQuantOutputSize;
    var i: usize = 0;
    while (i < bytes.len) : (i += 2) {
        const bits = std.mem.readInt(u16, bytes[i..][0..2], .little);
        if (!std.math.isFinite(@as(f32, @bitCast(@as(u32, bits) << 16)))) return error.NonfiniteQuantOutput;
    }
}
fn same(actual: []const u8, expected: []const u8) !void {
    try finite(actual);
    try finite(expected);
    if (!std.mem.eql(u8, actual, expected)) return error.QuantOutputMismatch;
}
fn hash(bytes: []const u8) [64]u8 {
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &digest, .{});
    return std.fmt.bytesToHex(digest, .lower);
}
pub fn checkInput(a: std.mem.Allocator) !void {
    const key: InputKey = .{ .ptr = 4096 };
    if (!key.eligible(key)) return error.QuantEligibilityControl;
    var changed = key;
    changed.scale *= 2;
    if (key.eligible(changed)) return error.QuantUnequalScaleAccepted;
    changed = key;
    changed.ptr += 4096;
    if (key.eligible(changed)) return error.QuantDifferentInputAccepted;
    changed = key;
    changed.stride += 16;
    if (key.eligible(changed)) return error.QuantDifferentStrideAccepted;
    const first = try input(a, 0);
    defer a.free(first);
    const second = try input(a, 1);
    defer a.free(second);
    try finite(first);
    try finite(second);
    for (0..groups) |g| {
        const start = g * width * 2;
        if (std.mem.eql(u8, first[start..][0 .. width * 2], second[start..][0 .. width * 2])) return error.QuantStaleInputControl;
        if (g != 0 and std.mem.eql(u8, first[0 .. width * 2], first[start..][0 .. width * 2])) return error.QuantRepeatedGroupInput;
    }
    var raw: [4096]u8 = undefined;
    var other: [4096]u8 = undefined;
    weights(&raw, 0, 0);
    weights(&other, banks - 1, 1);
    if (std.mem.eql(u8, &raw, &other)) return error.QuantWeightBankControl;
    for (raw) |code| if (code & 127 == 127) return error.NonfiniteQuantWeight;
    const altered = try a.dupe(u8, first);
    defer a.free(altered);
    altered[altered.len - 2] ^= 1;
    if (same(altered, first)) |_| return error.QuantLateOutputControl else |err| if (err != error.QuantOutputMismatch) return err;
    @memset(altered, 255);
    if (finite(altered)) |_| return error.QuantPoisonControl else |err| if (err != error.NonfiniteQuantOutput) return err;
}
fn floor(gpu: check.Gpu) !usize {
    const free = (try gpu.ctx.memInfo()).free;
    if (free < floor_bytes) return error.GpuMemoryFloor;
    return free;
}

const Fixture = struct {
    shape: Case,
    ck: *const cuda.ck.Ck,
    inputs: cuda.DeviceBuffer,
    weight_buffer: cuda.DeviceBuffer,
    encoded: cuda.DeviceBuffer,
    partials: cuda.DeviceBuffer,
    outputs: cuda.DeviceBuffer,

    fn weight(self: Fixture, bank: usize, projection: usize) cuda.qmmf.Weight {
        var offset = bank * self.shape.weightBytes();
        for (self.shape.outputs[0..projection]) |n| offset += n * width;
        var w = cuda.modelopt.fp8Weight(self.weight_buffer.ptr + offset, @as(f32, @floatFromInt(projection + 1)) / 64, self.shape.outputs[projection], width);
        w.act = act;
        return w;
    }
    fn output(self: Fixture, group: usize, projection: usize) u64 {
        var offset = group * self.shape.columns() * 2;
        for (self.shape.outputs[0..projection]) |n| offset += n * 2;
        return self.outputs.ptr + offset;
    }
    fn project(self: Fixture, arm: Arm, s: cuda.Stream, group: usize, bank: usize, projection: usize) !void {
        const w = self.weight(bank, projection);
        if (arm == .baseline) {
            const linear: cuda.qlinear.Linear = .{ .ck = self.ck, .ck_rows = self.encoded.ptr, .ck_part = self.partials.ptr };
            try linear.decode(s, .{ .fp8 = w }, .{ .x = self.inputs.ptr + group * width * 2, .ldx = width }, self.output(group, projection), 1);
        } else {
            try self.ck.lane8(s, self.encoded.ptr, 1, w, w.act, self.output(group, projection), self.partials.ptr);
        }
    }
    fn groupRun(self: Fixture, arm: Arm, s: cuda.Stream, group: usize, bank: usize) !void {
        if (arm == .shared_quant) try self.ck.quant8(s, self.inputs.ptr + group * width * 2, width, 1, width, act, self.encoded.ptr);
        for (0..self.shape.outputs.len) |p| try self.project(arm, s, group, bank, p);
    }
};

fn argument(comptime T: type, params: cuda.abi.KernelNodeParams, i: usize) !T {
    const args = params.params orelse return error.QuantMissingKernelArguments;
    const ptr = args[i] orelse return error.QuantMissingKernelArguments;
    return @as(*const T, @ptrCast(@alignCast(ptr))).*;
}
fn geometry(p: cuda.abi.KernelNodeParams, f: cuda.abi.Function, x: usize, z: usize, shared: usize) !void {
    if (p.func != f or p.grid_x != x or p.grid_y != 1 or p.grid_z != z or p.block_x != 128 or p.block_y != 1 or p.block_z != 1 or p.shared_bytes != shared) return error.QuantKernelGeometryMismatch;
}
fn checkQuant(p: cuda.abi.KernelNodeParams, f: Fixture, group: usize) !void {
    try geometry(p, f.ck.quant8_fn.handle, 1, 1, 0);
    if (try argument(u64, p, 0) != f.inputs.ptr + group * width * 2 or
        try argument(i32, p, 1) != width or try argument(i32, p, 2) != 1 or
        try argument(i32, p, 3) != width or try argument(f32, p, 4) != cuda.ck.inverse(act) or
        try argument(u64, p, 5) != f.encoded.ptr) return error.QuantKernelArgumentMismatch;
}
fn checkLane(p: cuda.abi.KernelNodeParams, f: Fixture, group: usize, bank: usize, projection: usize) !void {
    const w = f.weight(bank, projection);
    const sk = cuda.qmmf.splitK(w.n, w.k);
    const cluster = sk > 1 and sk <= 8 and f.ck.clusters;
    if (sk > 1 and !cluster) return error.QuantScreenRequiresCluster;
    try geometry(p, f.ck.lane[@intFromBool(cluster)].handle, (w.n + 63) / 64, sk, 20480);
    if (try argument(u64, p, 0) != f.encoded.ptr or try argument(u64, p, 1) != w.codes or
        try argument(f32, p, 2) != cuda.ck.alpha(w.act, w.scale) or try argument(u64, p, 3) != f.output(group, projection) or
        try argument(u64, p, 4) != f.partials.ptr or try argument(i32, p, 5) != 1 or
        try argument(i32, p, 6) != w.n or try argument(i32, p, 7) != width or
        try argument(i32, p, 8) != sk or try argument(i32, p, 9) != 1) return error.QuantLaneArgumentMismatch;
}

fn capture(gpu: check.Gpu, f: Fixture, arm: Arm, active_banks: usize, s: cuda.Stream) !cuda.graph.Exec {
    try cuda.graph.beginCapture(s, .thread_local);
    for (0..groups) |g| f.groupRun(arm, s, g, g % active_banks) catch |err| {
        if (cuda.graph.endCapture(s)) |graph| {
            var dead = graph;
            dead.deinit();
        } else |_| {}
        return err;
    };
    var graph = try cuda.graph.endCapture(s);
    defer graph.deinit();
    const per_group = f.shape.outputs.len + if (arm == .baseline) f.shape.outputs.len else @as(usize, 1);
    const count = groups * per_group;
    if (try graph.nodeCount() != count) return error.QuantGraphWorkCountMismatch;
    const storage = try gpu.gpa.alloc(cuda.graph.Node, count);
    defer gpu.gpa.free(storage);
    const nodes = try graph.nodes(storage);
    if (nodes.len != count) return error.QuantGraphWorkCountMismatch;
    var lib = gpu.d.lib;
    const get_deps = lib.lookup(*const fn (cuda.abi.GraphNode, ?[*]cuda.abi.GraphNode, *usize) callconv(.c) cuda.abi.Result, "cuGraphNodeGetDependencies") orelse return error.MissingSymbol;
    const predecessors = try gpu.gpa.alloc(cuda.graph.Node, count);
    defer gpu.gpa.free(predecessors);
    for (nodes, predecessors) |node, *previous| {
        var deps: [2]cuda.graph.Node = .{ null, null };
        var n: usize = deps.len;
        try gpu.d.check(get_deps(node, &deps, &n), "cuGraphNodeGetDependencies");
        if (n > 1) return error.QuantGraphDependencyMismatch;
        previous.* = if (n == 0) null else deps[0];
    }
    var prior: cuda.graph.Node = null;
    for (0..count) |index| {
        var found: ?usize = null;
        for (predecessors, 0..) |previous, j| if (previous == prior) {
            if (found != null) return error.QuantGraphDependencyMismatch;
            found = j;
        };
        const j = found orelse return error.QuantGraphDependencyMismatch;
        var p: cuda.abi.KernelNodeParams = undefined;
        try gpu.d.check(gpu.d.api.cuGraphKernelNodeGetParams_v2(nodes[j], &p), "cuGraphKernelNodeGetParams");
        const g = index / per_group;
        const local = index % per_group;
        const quant = if (arm == .baseline) local % 2 == 0 else local == 0;
        if (quant) try checkQuant(p, f, g) else try checkLane(p, f, g, g % active_banks, if (arm == .baseline) local / 2 else local - 1);
        prior = nodes[j];
    }
    var exec = try graph.instantiate();
    errdefer exec.deinit();
    try exec.upload(s);
    try s.synchronize();
    return exec;
}

fn encodedParity(gpu: check.Gpu, f: Fixture, s: cuda.Stream, active_banks: usize, journal: anytype, payload: usize) !void {
    var baseline: [width]u8 = undefined;
    var candidate: [width]u8 = undefined;
    var previous: [width]u8 = undefined;
    var negative: [width]u8 = undefined;
    for (0..groups) |g| {
        try f.ck.quant8(s, f.inputs.ptr + g * width * 2, width, 1, width, act, f.encoded.ptr);
        try s.synchronize();
        try f.encoded.download(0, &candidate);
        for (candidate) |code| if (code & 127 == 127) return error.NonfiniteEncodedActivation;
        if (g != 0 and std.mem.eql(u8, &candidate, &previous)) return error.QuantStaleEncodedInput;
        previous = candidate;
        for (0..f.shape.outputs.len) |p| {
            try f.project(.baseline, s, g, g % active_banks, p);
            try s.synchronize();
            try f.encoded.download(0, &baseline);
            if (!std.mem.eql(u8, &baseline, &candidate)) return error.QuantEncodedParityMismatch;
        }
        if (g == 0) {
            try f.ck.quant8(s, f.inputs.ptr, width, 1, width, act * 2, f.encoded.ptr);
            try s.synchronize();
            try f.encoded.download(0, &negative);
            if (std.mem.eql(u8, &candidate, &negative)) return error.QuantUnequalScaleControlInsensitive;
        }
    }
    try journal.write(.{ .kind = "quant_reuse_encoded_parity", .ordinal = gpu.ctx.device_ordinal, .case = f.shape.name, .payload = payload, .banks = active_banks, .groups = groups, .bytes_compared_per_projection = width, .baseline_quantizations_checked = groups * f.shape.outputs.len, .shared_quantizations_checked = groups, .all_encoded_bytes_exact = true, .changing_rows_change_encoded_bytes = true, .unequal_scale_encoded_bytes_differ = true, .unequal_scale_eligibility_rejected = true, .independent_weight_scales = true });
}

fn runCase(gpu: check.Gpu, ck: *const cuda.ck.Ck, packer: *const cuda.modelopt.Packer, s: cuda.Stream, shape: Case, journal: anytype) !void {
    var arena = std.heap.ArenaAllocator.init(gpu.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const payloads = [2][]u8{ try input(a, 0), try input(a, 1) };
    var inputs = try cuda.DeviceBuffer.fromHost(gpu.d, payloads[0]);
    defer inputs.free();
    var weight_buffer = try cuda.DeviceBuffer.alloc(gpu.d, banks * shape.weightBytes());
    defer weight_buffer.free();
    var encoded = try cuda.DeviceBuffer.alloc(gpu.d, width);
    defer encoded.free();
    var partials = try cuda.DeviceBuffer.alloc(gpu.d, 8192 * 8 * 4);
    defer partials.free();
    var outputs = try cuda.DeviceBuffer.alloc(gpu.d, shape.outputBytes());
    defer outputs.free();
    const f: Fixture = .{ .shape = shape, .ck = ck, .inputs = inputs, .weight_buffer = weight_buffer, .encoded = encoded, .partials = partials, .outputs = outputs };
    {
        const raw = try gpu.gpa.alloc(u8, 8192 * width);
        defer gpu.gpa.free(raw);
        var staging = try cuda.DeviceBuffer.alloc(gpu.d, raw.len);
        defer staging.free();
        for (0..banks) |bank| for (shape.outputs, 0..) |n, p| {
            weights(raw[0 .. n * width], bank, p);
            try staging.upload(0, raw[0 .. n * width]);
            try packer.fp8Lane(s, staging.ptr, f.weight(bank, p).codes, n, width);
            try s.synchronize();
        };
    }
    const l2: usize = @intCast(try gpu.ctx.attribute(.l2_cache_size));
    if (banks * shape.weightBytes() <= l2) return error.QuantRotationFitsL2;
    try journal.write(.{ .kind = "quant_reuse_fixture", .ordinal = gpu.ctx.device_ordinal, .case = shape.name, .outputs = shape.outputs, .width = width, .rows = 1, .input_scale = act, .weight_scale_rule = "(projection+1)/64", .seed = seed, .pattern_version = 1, .banks = banks, .weight_bytes_per_bank = shape.weightBytes(), .unique_weight_bytes = banks * shape.weightBytes(), .l2_bytes = l2, .groups = groups, .distinct_input_rows = groups, .payloads = 2, .output_bytes = shape.outputBytes(), .pack_path = "modelopt.Packer.fp8Lane + modelopt.fp8Weight", .baseline_path = "qlinear.Linear.decode(.fp8)", .candidate_path = "Ck.quant8 once + same ordered Ck.lane8", .free_bytes = try floor(gpu), .scope = scope });
    const poison = try a.alloc(u8, shape.outputBytes());
    @memset(poison, 255);
    const reference = [2][]u8{ try a.alloc(u8, shape.outputBytes()), try a.alloc(u8, shape.outputBytes()) };
    const actual = try a.alloc(u8, shape.outputBytes());
    var start = try cuda.Event.init(gpu.d, true);
    defer start.deinit();
    var end = try cuda.Event.init(gpu.d, true);
    defer end.deinit();
    for ([_]usize{ banks, 1 }) |active_banks| {
        const cache_scope = if (active_banks == 1) "hot_one_bank" else "rotating_eight_banks";
        for (payloads, 0..) |bytes, payload| {
            try inputs.upload(0, bytes);
            try outputs.upload(0, poison);
            try encodedParity(gpu, f, s, active_banks, journal, payload);
            try outputs.download(0, reference[payload]);
            try finite(reference[payload]);
            try outputs.upload(0, poison);
            for (0..groups) |g| try f.groupRun(.shared_quant, s, g, g % active_banks);
            try s.synchronize();
            try outputs.download(0, actual);
            try same(actual, reference[payload]);
            try journal.write(.{ .kind = "quant_reuse_output_parity", .ordinal = gpu.ctx.device_ordinal, .case = shape.name, .cache_scope = cache_scope, .payload = payload, .all_output_bytes_compared = actual.len, .all_banks_checked = active_banks, .finite = true, .bit_exact = true, .output_sha256 = &hash(actual), .input_sha256 = &hash(bytes) });
        }
        if (std.mem.eql(u8, reference[0], reference[1])) return error.QuantChangedPayloadControlInsensitive;
        var graphs: [2]cuda.graph.Exec = undefined;
        var made: usize = 0;
        defer for (graphs[0..made]) |*graph| graph.deinit();
        for (std.meta.tags(Arm), 0..) |arm, i| {
            graphs[i] = try capture(gpu, f, arm, active_banks, s);
            made += 1;
            const quant_count = groups * if (arm == .baseline) shape.outputs.len else @as(usize, 1);
            try journal.write(.{ .kind = "quant_reuse_kernelwork", .ordinal = gpu.ctx.device_ordinal, .case = shape.name, .cache_scope = cache_scope, .arm = @tagName(arm), .quant_symbol = "tf_ck_quant8", .quant_function_handle = @intFromPtr(ck.quant8_fn.handle), .quant_nodes = quant_count, .gemm_nodes = groups * shape.outputs.len, .total_nodes = quant_count + groups * shape.outputs.len, .function_geometry_arguments_and_serial_dependencies_checked = true, .quantized_rows_per_node = 1, .quantized_columns_per_node = width, .groups = groups, .banks = active_banks });
        }
        var savings: [5]f64 = undefined;
        var wall_savings: [5]f64 = undefined;
        for (0..6) |pair| {
            var times: [2]f64 = undefined;
            var wall: [2]f64 = undefined;
            const payload = pair % 2;
            for (0..2) |order| {
                const index = if (pair % 2 == 0) order else 1 - order;
                const arm = std.meta.tags(Arm)[index];
                try inputs.upload(0, payloads[payload]);
                try outputs.upload(0, poison);
                try s.synchronize();
                const free_before = try floor(gpu);
                const wall_start = check.now(gpu.io);
                try start.record(s);
                try graphs[index].launchOn(s);
                try end.record(s);
                try end.synchronize();
                const elapsed_ns = check.now(gpu.io) - wall_start;
                const elapsed_ms = try cuda.Event.elapsedMs(start, end);
                if (!std.math.isFinite(elapsed_ms) or elapsed_ms <= 0 or elapsed_ns <= 0) return error.InvalidQuantTiming;
                times[index] = @as(f64, elapsed_ms) * 1000 / groups;
                wall[index] = @as(f64, @floatFromInt(elapsed_ns)) / 1000 / groups;
                try outputs.download(0, actual);
                try same(actual, reference[payload]);
                try journal.write(.{ .kind = "quant_reuse_sample", .ordinal = gpu.ctx.device_ordinal, .case = shape.name, .cache_scope = cache_scope, .pair = pair, .order = order, .payload = payload, .arm = @tagName(arm), .warmup = pair == 0, .elapsed_ms = elapsed_ms, .wall_ns = elapsed_ns, .us_per_group = times[index], .wall_us_per_group = wall[index], .graph_launches = 1, .groups = groups, .all_output_bytes_compared = actual.len, .bit_exact = true, .finite = true, .output_sha256 = &hash(actual), .reset_outside_events = true, .free_bytes_before = free_before, .free_bytes = try floor(gpu), .scope = scope });
            }
            if (pair > 0) {
                savings[pair - 1] = times[0] - times[1];
                wall_savings[pair - 1] = wall[0] - wall[1];
            }
        }
        const ordered_savings = try a.dupe(f64, &savings);
        const median_us = check.median(ordered_savings);
        try journal.write(.{ .kind = "quant_reuse_case_complete", .ordinal = gpu.ctx.device_ordinal, .case = shape.name, .cache_scope = cache_scope, .paired_saving_us_per_group = savings, .wall_paired_saving_us_per_group = wall_savings, .median_saving_us_per_group = median_us, .min_saving_us_per_group = ordered_savings[0], .max_saving_us_per_group = ordered_savings[4], .groups_per_append_on_this_partition = shape.per_partition, .estimated_turn_saving_us_this_case_this_partition = 191 * @as(f64, @floatFromInt(shape.per_partition)) * median_us, .projection_formula = "191*(15*pair_gpu0+5*triple_gpu0+15*pair_gpu1+5*triple_gpu1) us", .accepted_turn_seconds = 1.7232142, .required_total_us_per_append = 451.1031937172775, .required_us_per_removed_quant = 9.02206387434555, .measured_pairs = 5, .excluded_warmup_pairs = 1, .correctness_pass = true, .production_qualification = false });
    }
}

pub fn run(gpu: check.Gpu, journal: anytype) !void {
    _ = try floor(gpu);
    try checkInput(gpu.gpa);
    const major: u32 = @intCast(try gpu.ctx.attribute(.compute_capability_major));
    const sms: usize = @intCast(try gpu.ctx.attribute(.multiprocessor_count));
    var ck = try cuda.ck.Ck.load(gpu.d, major, sms);
    defer ck.unload();
    var lib = gpu.d.lib;
    const get_name = lib.lookup(*const fn (*?[*:0]const u8, cuda.abi.Function) callconv(.c) cuda.abi.Result, "cuFuncGetName") orelse return error.MissingSymbol;
    var quant_name: ?[*:0]const u8 = null;
    try gpu.d.check(get_name(&quant_name, ck.quant8_fn.handle), "cuFuncGetName");
    if (!std.mem.eql(u8, std.mem.span(quant_name orelse return error.MissingSymbol), "tf_ck_quant8")) return error.QuantFunctionSymbolMismatch;
    var packer = try cuda.modelopt.Packer.load(gpu.d);
    defer packer.unload();
    var stream = try cuda.Stream.init(gpu.d, false);
    defer stream.deinit();
    for (cases) |shape| try runCase(gpu, &ck, &packer, stream, shape, journal);
    try journal.write(.{ .kind = "quant_reuse_screen_complete", .ordinal = gpu.ctx.device_ordinal, .cases = 2, .cache_scopes = 2, .measured_pairs_per_case_scope = 5, .correctness_pass = true, .full_model_gate_pass = false, .production_qualification = false, .scope = scope });
}

test "quant reuse rejects unequal scales and altered or stale payloads" {
    try checkInput(std.testing.allocator);
}

test "quant reuse shape counts and independent weight scales" {
    try std.testing.expectEqual(@as(usize, 24 * 1024 * 1024), cases[0].weightBytes());
    try std.testing.expectEqual(@as(usize, 18 * 1024 * 1024), cases[1].weightBytes());
    try std.testing.expectEqual(@as(usize, 25), cases[0].per_partition * (cases[0].outputs.len - 1) + cases[1].per_partition * (cases[1].outputs.len - 1));
    try std.testing.expect(cuda.ck.alpha(act, 1.0 / 64.0) != cuda.ck.alpha(act, 2.0 / 64.0));
}
