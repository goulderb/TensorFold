//! Isolated row-count proof using the existing ModelOpt loader and decode operators.
//! Synthetic distinct BF16 inputs; real checkpoint weights. No engine forward or prompt GEMM.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const qwen = @import("qwen35_cuda");
const check = @import("check.zig");
const max_rows = 16;
const repeats = 16;
const guard_bytes = 64;
const floor_bytes = 1536 << 20;
const scope = "real-checkpoint decode row operators with synthetic inputs; resident-weight graph replay; no model prefill gain claim";
const Arm = enum { serial, batch };

fn hash(bytes: []const u8) [64]u8 {
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &digest, .{});
    return std.fmt.bytesToHex(digest, .lower);
}

fn floor(gpu: check.Gpu) !usize {
    const free = (try gpu.ctx.memInfo()).free;
    if (free < floor_bytes) return error.PrefillLinearMemoryFloor;
    return free;
}

fn inputBits(row: usize, column: usize) u16 {
    var x: u32 = @truncate((row + 1) * 0x9e3779b1 +% column *% 0x85ebca6b);
    x ^= x >> 16;
    x *%= 0x7feb352d;
    x ^= x >> 15;
    return @as(u16, @truncate((x >> 16) & 0x8000)) | 0x3f00 | @as(u16, @truncate(x & 0xff));
}

fn finite(bytes: []const u8) !void {
    if (bytes.len % 2 != 0) return error.InvalidLinearOutput;
    var i: usize = 0;
    while (i < bytes.len) : (i += 2) {
        if (std.mem.readInt(u16, bytes[i..][0..2], .little) & 0x7f80 == 0x7f80) return error.NonfiniteLinearOutput;
    }
}

fn distinct(bytes: []const u8, row_bytes: usize, rows: usize) !void {
    for (0..rows) |r| for (0..r) |prior| {
        if (std.mem.eql(u8, bytes[r * row_bytes ..][0..row_bytes], bytes[prior * row_bytes ..][0..row_bytes])) return error.LinearRowsNotDistinct;
    };
}

fn checkOutput(actual: []const u8, reference: []const u8) !void {
    if (actual.len < reference.len) return error.InvalidLinearOutput;
    try finite(actual[0..reference.len]);
    if (!std.mem.eql(u8, actual[0..reference.len], reference)) return error.PrefillLinearMismatch;
    for (actual[reference.len..]) |byte| if (byte != 0xff) return error.PrefillLinearGuardChanged;
}

fn checkEncoded(actual: []const u8, reference: []const u8, used: usize) !void {
    if (used > actual.len or used > reference.len) return error.InvalidLinearEncodedSize;
    if (!std.mem.eql(u8, actual[0..used], reference[0..used])) return error.LinearQuantizedRowsMismatch;
    for (actual[0..used]) |byte| if (byte & 0x7f == 0x7f) return error.NonfiniteLinearEncoded;
    for (actual[used..]) |byte| if (byte != 0xff) return error.LinearQuantizedGuardChanged;
}

fn weight(p: qwen.shared_engine.Proj) !cuda.qmmf.Weight {
    if (p != .linear) return error.PrefillLinearFormat;
    return switch (p.linear) {
        .fp8, .bf16, .nvfp4 => |w| w,
        else => error.PrefillLinearFormat,
    };
}

const Fixture = struct {
    linear: cuda.qlinear.Linear,
    projection: cuda.qlinear.Weight,
    w: cuda.qmmf.Weight,
    input: u64,
    output: u64,
    encoded: u64,
    part: u64,

    fn checkpoint(f: Fixture) bool {
        return f.projection == .fp8 and f.w.act != 0;
    }
    fn call(f: Fixture, stream: cuda.Stream, row: usize, rows: usize) !void {
        var linear = f.linear;
        linear.ck_rows = f.encoded + row * f.w.k;
        linear.ck_part = f.part;
        try linear.decode(stream, f.projection, .{ .x = f.input + row * f.w.k * 2, .part = f.part }, f.output + row * f.w.n * 2, rows);
    }
    fn run(f: Fixture, stream: cuda.Stream, arm: Arm, rows: usize) !void {
        if (arm == .serial) {
            for (0..rows) |row| try f.call(stream, row, 1);
        } else try f.call(stream, 0, rows);
    }
};

const Expected = struct { function: cuda.Function, grid: [3]usize, shared: usize, args: cuda.Args };

fn expected(f: Fixture, row: usize, rows: usize, which: usize) !Expected {
    const w = f.w;
    const x = f.input + row * w.k * 2;
    const y = f.output + row * w.n * 2;
    const encoded = f.encoded + row * w.k;
    var a: cuda.Args = .{};
    if (f.checkpoint()) {
        const ck = f.linear.ck.?;
        const sk = cuda.qmmf.splitK(w.n, w.k);
        const cluster = sk > 1 and sk <= 8 and ck.clusters;
        if (sk > 1 and !cluster) return error.PrefillLinearRequiresClusters;
        if (which == 0) {
            a.add(x);
            for ([_]usize{ w.k, rows, w.k }) |v| a.add(@as(i32, @intCast(v)));
            a.add(cuda.ck.inverse(w.act));
            a.add(encoded);
            return .{ .function = ck.quant8_fn, .grid = .{ (rows * (w.k / 16) + 127) / 128, 1, 1 }, .shared = 0, .args = a };
        }
        a.add(encoded);
        a.add(w.codes);
        a.add(cuda.ck.alpha(w.act, w.scale));
        a.add(y);
        a.add(f.part);
        for ([_]usize{ rows, w.n, w.k, sk, 1 }) |v| a.add(@as(i32, @intCast(v)));
        return .{ .function = ck.lane[@intFromBool(cluster)], .grid = .{ (w.n + 63) / 64, 1, sk }, .shared = 20480, .args = a };
    }
    const lane = f.linear.lane.?;
    const plan = cuda.qmmf.plan(rows, w.n, w.k, lane.clusters);
    if (plan.reduce or plan.fused or plan.bm != 16) return error.PrefillLinearUnexpectedPlan;
    a.add(x);
    a.add(w.codes);
    a.add(w.scales);
    a.add(w.scale);
    a.add(y);
    a.add(@as(u64, 0));
    for ([_]usize{ rows, w.n, w.k, plan.sk, w.npad, w.k, 1 }) |v| a.add(@as(i32, @intCast(v)));
    return .{ .function = lane.fns[@backingInt(w.mode)][@intFromBool(plan.cluster)], .grid = .{ (w.n + 63) / 64, 1, plan.sk }, .shared = cuda.qmmf.sharedBytes(w.mode, 16), .args = a };
}

fn verifyArgs(p: cuda.abi.KernelNodeParams, want: *const cuda.Args) ![64]u8 {
    const args = p.params orelse return error.LinearGraphMissingArgs;
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    for (0..want.count) |i| {
        const actual: [*]const u8 = @ptrCast(args[i] orelse return error.LinearGraphMissingArgs);
        const bytes = want.storage[want.offsets[i]..][0..want.sizes[i]];
        if (!std.mem.eql(u8, actual[0..bytes.len], bytes)) return error.LinearGraphArgumentMismatch;
        h.update(actual[0..bytes.len]);
    }
    var digest: [32]u8 = undefined;
    h.final(&digest);
    return std.fmt.bytesToHex(digest, .lower);
}

fn kernelName(f: cuda.Function) ![]const u8 {
    var lib = f.d.lib;
    const get = lib.lookup(*const fn (*?[*:0]const u8, cuda.abi.Function) callconv(.c) cuda.abi.Result, "cuFuncGetName") orelse return error.MissingSymbol;
    var name: ?[*:0]const u8 = null;
    try f.d.check(get(&name, f.handle), "cuFuncGetName");
    return std.mem.span(name orelse return error.MissingSymbol);
}

fn save(gpu: check.Gpu, journal: anytype, name: []const u8, rows: usize, arm: []const u8, dtype: []const u8, bytes: []const u8) !void {
    const basename = try std.fmt.allocPrint(gpu.gpa, "linear-gpu{d}-{s}-rows{d}-{s}.{s}", .{ gpu.ctx.device_ordinal, name, rows, arm, dtype });
    defer gpu.gpa.free(basename);
    const path = try std.fs.path.join(gpu.gpa, &.{ journal.directory, basename });
    defer gpu.gpa.free(path);
    const file = try std.Io.Dir.cwd().createFile(gpu.io, path, .{ .exclusive = true });
    defer file.close(gpu.io);
    try file.writeStreamingAll(gpu.io, bytes);
    try journal.write(.{ .kind = "prefill_linear_saved", .ordinal = gpu.ctx.device_ordinal, .case = name, .rows = rows, .arm = arm, .dtype = dtype, .basename = basename, .bytes = bytes.len, .sha256 = &hash(bytes) });
}

const Readback = struct {
    output: cuda.DeviceBuffer,
    encoded: cuda.DeviceBuffer,
    part: cuda.DeviceBuffer,
    actual: []u8,
    quant_actual: []u8,
    part_actual: []u8,

    fn verify(b: Readback, gpu: check.Gpu, f: Fixture, reference: []const u8, quant_reference: []const u8, rows: usize, name: []const u8, arm: []const u8, phase: []const u8, journal: anytype) !void {
        try b.output.download(0, b.actual);
        try b.encoded.download(0, b.quant_actual);
        try b.part.download(0, b.part_actual);
        b.validate(f, reference, quant_reference, rows) catch |err| {
            const label = try std.fmt.allocPrint(gpu.gpa, "{s}-{s}-failure", .{ arm, phase });
            defer gpu.gpa.free(label);
            // Failure files include inactive rows and guards, so the offending byte survives.
            try save(gpu, journal, name, rows, label, "bf16", b.actual);
            try save(gpu, journal, name, rows, label, "e4m3", b.quant_actual);
            try save(gpu, journal, name, rows, label, "partial-bytes", b.part_actual);
            try journal.write(.{ .kind = "prefill_linear_mismatch", .ordinal = gpu.ctx.device_ordinal, .case = name, .rows = rows, .arm = arm, .phase = phase, .failure = @errorName(err), .active_output_bytes = reference.len, .first_active_difference = std.mem.indexOfDiff(u8, b.actual[0..reference.len], reference), .failure_files_include_guards = true });
            return err;
        };
    }

    fn validate(b: Readback, f: Fixture, reference: []const u8, quant_reference: []const u8, rows: usize) !void {
        try checkOutput(b.actual, reference);
        if (f.checkpoint()) try checkEncoded(b.quant_actual, quant_reference, rows * f.w.k);
        for (b.part_actual) |byte| if (byte != 0xff) return error.LinearUnexpectedPartialWrite;
    }
};

const Node = struct { index: usize, repeat: usize, row: usize, rows: usize, symbol: []const u8, grid: [3]usize, block: [3]usize, shared_bytes: usize, argument_count: usize, arguments_sha256: [64]u8 };

fn capture(gpu: check.Gpu, f: Fixture, stream: cuda.Stream, arm: Arm, rows: usize, name: []const u8, journal: anytype) !cuda.graph.Exec {
    try cuda.graph.beginCapture(stream, .thread_local);
    for (0..repeats) |_| f.run(stream, arm, rows) catch |err| {
        if (cuda.graph.endCapture(stream)) |value| {
            var dead = value;
            dead.deinit();
        } else |_| {}
        return err;
    };
    var graph = try cuda.graph.endCapture(stream);
    defer graph.deinit();
    const per_call: usize = if (f.checkpoint()) 2 else 1;
    const calls: usize = if (arm == .serial) rows else 1;
    const count = repeats * calls * per_call;
    if (try graph.nodeCount() != count) return error.LinearGraphCountMismatch;
    const storage = try gpu.gpa.alloc(cuda.graph.Node, count);
    defer gpu.gpa.free(storage);
    const nodes = try graph.nodes(storage);
    if (nodes.len != count) return error.LinearGraphCountMismatch;
    const predecessors = try gpu.gpa.alloc(cuda.graph.Node, count);
    defer gpu.gpa.free(predecessors);
    const roster = try gpu.gpa.alloc(Node, count);
    defer gpu.gpa.free(roster);
    var lib = gpu.d.lib;
    const get_deps = lib.lookup(*const fn (cuda.abi.GraphNode, ?[*]cuda.abi.GraphNode, *usize) callconv(.c) cuda.abi.Result, "cuGraphNodeGetDependencies") orelse return error.MissingSymbol;
    for (nodes, predecessors) |node, *previous| {
        var deps: [2]cuda.graph.Node = .{ null, null };
        var n: usize = deps.len;
        try gpu.d.check(get_deps(node, &deps, &n), "cuGraphNodeGetDependencies");
        if (n > 1) return error.LinearGraphDependencyMismatch;
        previous.* = if (n == 0) null else deps[0];
    }
    var prior: cuda.graph.Node = null;
    for (roster, 0..) |*entry, index| {
        var found: ?usize = null;
        for (predecessors, 0..) |previous, j| if (previous == prior) {
            if (found != null) return error.LinearGraphDependencyMismatch;
            found = j;
        };
        const at = found orelse return error.LinearGraphDependencyMismatch;
        var p: cuda.abi.KernelNodeParams = undefined;
        try gpu.d.check(gpu.d.api.cuGraphKernelNodeGetParams_v2(nodes[at], &p), "cuGraphKernelNodeGetParams");
        const row = if (arm == .serial) (index / per_call) % rows else 0;
        const m: usize = if (arm == .serial) 1 else rows;
        const want = try expected(f, row, m, index % per_call);
        if (p.func != want.function.handle or p.grid_x != want.grid[0] or p.grid_y != want.grid[1] or p.grid_z != want.grid[2] or p.block_x != 128 or p.block_y != 1 or p.block_z != 1 or p.shared_bytes != want.shared) return error.LinearGraphGeometryMismatch;
        entry.* = .{ .index = index, .repeat = index / (calls * per_call), .row = row, .rows = m, .symbol = try kernelName(want.function), .grid = want.grid, .block = .{ p.block_x, p.block_y, p.block_z }, .shared_bytes = p.shared_bytes, .argument_count = want.args.count, .arguments_sha256 = try verifyArgs(p, &want.args) };
        prior = nodes[at];
    }
    try journal.write(.{ .kind = "prefill_linear_graph", .ordinal = gpu.ctx.device_ordinal, .case = name, .arm = @tagName(arm), .rows = rows, .repeats = repeats, .node_count = count, .nodes = roster, .serial_dependencies_checked = true, .all_abi_arguments_checked = true, .cluster_attributes_checked = false });
    var exec = try graph.instantiate();
    errdefer exec.deinit();
    try exec.upload(stream);
    try stream.synchronize();
    return exec;
}

fn caseRun(gpu: check.Gpu, e: *qwen.shared_engine.Engine, name: []const u8, projection: qwen.shared_engine.Proj, journal: anytype) !void {
    const w = try weight(projection);
    const in_bytes = max_rows * @as(usize, w.k) * 2;
    const out_bytes = max_rows * @as(usize, w.n) * 2;
    const encoded_bytes = max_rows * @as(usize, w.k);
    const part_bytes = @max(256, e.modelopt.?.lane.partBytes(max_rows, w));
    const need = in_bytes + out_bytes + encoded_bytes + part_bytes + 3 * guard_bytes;
    if (try floor(gpu) < floor_bytes + need) return error.PrefillLinearMemoryFloor;
    const input = try gpu.gpa.alloc(u8, in_bytes);
    defer gpu.gpa.free(input);
    for (0..max_rows) |r| for (0..w.k) |c| std.mem.writeInt(u16, input[(r * w.k + c) * 2 ..][0..2], inputBits(r, c), .little);
    try finite(input);
    try distinct(input, w.k * 2, max_rows);
    var x = try cuda.DeviceBuffer.fromHost(gpu.d, input);
    defer x.free();
    var output = try cuda.DeviceBuffer.alloc(gpu.d, out_bytes + guard_bytes);
    defer output.free();
    var encoded = try cuda.DeviceBuffer.alloc(gpu.d, encoded_bytes + guard_bytes);
    defer encoded.free();
    var part = try cuda.DeviceBuffer.alloc(gpu.d, part_bytes + guard_bytes);
    defer part.free();
    const f: Fixture = .{ .linear = e.modelopt.?.linear(), .projection = projection.linear, .w = w, .input = x.ptr, .output = output.ptr, .encoded = encoded.ptr, .part = part.ptr };
    const reference = try gpu.gpa.alloc(u8, output.len);
    defer gpu.gpa.free(reference);
    const actual = try gpu.gpa.alloc(u8, output.len);
    defer gpu.gpa.free(actual);
    const quant_reference = try gpu.gpa.alloc(u8, encoded.len);
    defer gpu.gpa.free(quant_reference);
    const quant_actual = try gpu.gpa.alloc(u8, encoded.len);
    defer gpu.gpa.free(quant_actual);
    const part_actual = try gpu.gpa.alloc(u8, part.len);
    defer gpu.gpa.free(part_actual);
    const readback: Readback = .{ .output = output, .encoded = encoded, .part = part, .actual = actual, .quant_actual = quant_actual, .part_actual = part_actual };
    var reference_readback = readback;
    reference_readback.actual = reference;
    reference_readback.quant_actual = quant_reference;
    try output.fill8(0xff, e.stream.handle);
    try encoded.fill8(0xff, e.stream.handle);
    try part.fill8(0xff, e.stream.handle);
    try f.run(e.stream, .serial, max_rows);
    try e.stream.synchronize();
    try reference_readback.verify(gpu, f, reference[0..out_bytes], quant_reference, max_rows, name, "serial", "reference", journal);
    try save(gpu, journal, name, max_rows, "serial-reference", "bf16", reference[0..out_bytes]);
    if (f.checkpoint()) try save(gpu, journal, name, max_rows, "serial-reference", "e4m3", quant_reference[0..encoded_bytes]);
    try distinct(reference, w.n * 2, max_rows);
    var row_hashes: [max_rows][64]u8 = undefined;
    for (&row_hashes, 0..) |*h, r| h.* = hash(reference[r * w.n * 2 ..][0 .. w.n * 2]);
    try journal.write(.{ .kind = "prefill_linear_case", .ordinal = gpu.ctx.device_ordinal, .case = name, .format = @tagName(projection.linear), .weight = w, .input_source = "synthetic deterministic inputBits-v1, signed BF16 values; not model activations", .input_sha256 = &hash(input), .input_rows_distinct = true, .reference_rows_distinct = true, .reference_row_sha256 = row_hashes, .reference_sha256 = &hash(reference[0..out_bytes]), .ck_rows_bytes = encoded_bytes, .part_bytes = part_bytes, .guard_bytes = guard_bytes, .free_bytes = try floor(gpu) });
    var start = try cuda.Event.init(gpu.d, true);
    defer start.deinit();
    var end = try cuda.Event.init(gpu.d, true);
    defer end.deinit();
    var graphs: [6]cuda.graph.Exec = undefined;
    var made: usize = 0;
    defer for (graphs[0..made]) |*graph| graph.deinit();
    for ([_]usize{ 1, 2, 16 }, 0..) |rows, ri| {
        const used = rows * w.n * 2;
        for ([_]Arm{ .serial, .batch }, 0..) |arm, index| {
            try output.fill8(0xff, e.stream.handle);
            try encoded.fill8(0xff, e.stream.handle);
            try part.fill8(0xff, e.stream.handle);
            try f.run(e.stream, arm, rows);
            try e.stream.synchronize();
            try readback.verify(gpu, f, reference[0..used], quant_reference, rows, name, @tagName(arm), "eager", journal);
            graphs[2 * ri + index] = try capture(gpu, f, e.stream, arm, rows, name, journal);
            made += 1;
            try output.fill8(0xff, e.stream.handle);
            try encoded.fill8(0xff, e.stream.handle);
            try part.fill8(0xff, e.stream.handle);
            try graphs[2 * ri + index].launchOn(e.stream);
            try e.stream.synchronize();
            try readback.verify(gpu, f, reference[0..used], quant_reference, rows, name, @tagName(arm), "captured", journal);
            if (arm == .batch) {
                try save(gpu, journal, name, rows, "captured-batch", "bf16", actual[0..used]);
                if (f.checkpoint()) try save(gpu, journal, name, rows, "captured-batch", "e4m3", quant_actual[0 .. rows * w.k]);
            }
        }
        try journal.write(.{ .kind = "prefill_linear_correctness", .ordinal = gpu.ctx.device_ordinal, .case = name, .rows = rows, .eager_and_captured_exact = true, .checked_bytes = used, .inactive_and_guard_bytes = actual.len - used, .finite = true, .quantized_rows_checked = f.checkpoint(), .output_sha256 = &hash(reference[0..used]) });
    }
    for ([_]usize{ 1, 2, 16 }, 0..) |rows, ri| {
        const used = rows * w.n * 2;
        var event_times: [2][5]f64 = undefined;
        var wall_times: [2][5]f64 = undefined;
        for (0..6) |pair| for (0..2) |order| {
            const index = if (pair % 2 == 0) order else 1 - order;
            const arm: Arm = if (index == 0) .serial else .batch;
            try output.fill8(0xff, e.stream.handle);
            try encoded.fill8(0xff, e.stream.handle);
            try part.fill8(0xff, e.stream.handle);
            try e.stream.synchronize();
            const free_before = try floor(gpu);
            const begin = check.now(gpu.io);
            try start.record(e.stream);
            try graphs[2 * ri + index].launchOn(e.stream);
            try end.record(e.stream);
            try end.synchronize();
            const wall_ns = check.now(gpu.io) - begin;
            const event_ms = try cuda.Event.elapsedMs(start, end);
            if (wall_ns <= 0 or !std.math.isFinite(event_ms) or event_ms <= 0) return error.InvalidLinearTiming;
            try readback.verify(gpu, f, reference[0..used], quant_reference, rows, name, @tagName(arm), "timed", journal);
            const event_us = @as(f64, event_ms) * 1000 / repeats;
            const wall_us = @as(f64, @floatFromInt(wall_ns)) / 1000 / repeats;
            if (pair != 0) {
                event_times[index][pair - 1] = event_us;
                wall_times[index][pair - 1] = wall_us;
            }
            try journal.write(.{ .kind = "prefill_linear_sample", .ordinal = gpu.ctx.device_ordinal, .case = name, .rows = rows, .arm = @tagName(arm), .pair = pair, .order = order, .warmup = pair == 0, .repeats = repeats, .event_ms = event_ms, .wall_ns = wall_ns, .event_us_per_prefix = event_us, .wall_us_per_prefix = wall_us, .all_output_bytes_exact = true, .finite = true, .poison_checked = true, .quantized_rows_checked = f.checkpoint(), .partial_workspace_untouched = true, .output_sha256 = &hash(actual[0..used]), .free_before = free_before, .free_after = try floor(gpu), .scope = scope });
        };
        var event_saved: [5]f64 = undefined;
        var wall_saved: [5]f64 = undefined;
        for (0..5) |i| {
            event_saved[i] = event_times[0][i] - event_times[1][i];
            wall_saved[i] = wall_times[0][i] - wall_times[1][i];
        }
        try journal.write(.{ .kind = "prefill_linear_summary", .ordinal = gpu.ctx.device_ordinal, .case = name, .rows = rows, .event_serial_median_us = check.median(&event_times[0]), .event_batch_median_us = check.median(&event_times[1]), .wall_serial_median_us = check.median(&wall_times[0]), .wall_batch_median_us = check.median(&wall_times[1]), .paired_event_saved_median_us = check.median(&event_saved), .paired_wall_saved_median_us = check.median(&wall_saved), .scope = scope });
    }
}

fn placementRun(gpu: check.Gpu, dir: []const u8, config: qwen.checkpoint.Config, placement: qwen.checkpoint.Placement, journal: anytype) !void {
    const admitted: qwen.checkpoint.Admitted = .{ .geometry = config.geometry, .encoding = .{ .modelopt = config } };
    var source = try core.Checkpoint.openModel(gpu.gpa, gpu.io, dir);
    const plan = qwen.weights.plan(&source, admitted, placement) catch |err| {
        source.close();
        return err;
    };
    source.close();
    const scratch = try qwen.shared_engine.Engine.scratchBytes(admitted, placement);
    const required = plan.weights_bytes + plan.staging_bytes + scratch + floor_bytes;
    if (try floor(gpu) < required) return error.PrefillLinearMemoryFloor;
    const e = try qwen.shared_engine.Engine.initPlaced(gpu.gpa, gpu.io, gpu.ctx, dir, admitted, placement, false);
    defer e.deinit();
    try journal.write(.{ .kind = "prefill_linear_placement", .ordinal = gpu.ctx.device_ordinal, .first = placement.first, .end = placement.end, .weights_bytes = plan.weights_bytes, .staging_bytes = plan.staging_bytes, .scratch_bytes = scratch, .required_bytes = required, .free_after_load = try floor(gpu) });
    if (placement.first == 1) {
        if (e.layers[0].mixer != .delta or e.layers[2].mixer != .attention or e.layers[0].mlp != .moe or e.layers[0].mlp.moe.shared != .projected) return error.PrefillLinearLayerRoster;
        const gdn = e.layers[0].mixer.delta;
        inline for (.{ "qkv", "z", "a", "b", "out" }) |field| try caseRun(gpu, e, "layer1.gdn." ++ field, @field(gdn, field), journal);
        const attention = e.layers[2].mixer.attention;
        inline for (.{ "q", "k", "v", "o" }) |field| try caseRun(gpu, e, "layer3.attention." ++ field, @field(attention, field), journal);
        const shared = e.layers[0].mlp.moe.shared.projected;
        inline for (.{ "gate", "up", "down" }) |field| try caseRun(gpu, e, "layer1.shared." ++ field, @field(shared, field), journal);
    } else try caseRun(gpu, e, "head", e.head, journal);
}

pub fn run(gpu: check.Gpu, model_dir: []const u8, journal: anytype) !void {
    return runInner(gpu, model_dir, journal) catch |err| {
        journal.write(.{ .kind = "prefill_linear_failure", .ordinal = gpu.ctx.device_ordinal, .failure = @errorName(err), .correctness_pass = false }) catch {};
        return err;
    };
}

fn runInner(gpu: check.Gpu, model_dir: []const u8, journal: anytype) !void {
    const config = try qwen.checkpoint.Config.read(gpu.gpa, gpu.io, model_dir);
    const path = try std.fs.path.join(gpu.gpa, &.{ model_dir, "config.json" });
    defer gpu.gpa.free(path);
    const bytes = try std.Io.Dir.cwd().readFileAlloc(gpu.io, path, gpu.gpa, .limited(1 << 22));
    defer gpu.gpa.free(bytes);
    try journal.write(.{ .kind = "prefill_linear_start", .ordinal = gpu.ctx.device_ordinal, .model_dir = model_dir, .config_sha256 = &hash(bytes), .checkpoint_tensor_pins = "external protected-window input manifest; loader validates actual tensor names, formats and dimensions", .rows = [_]usize{ 1, 2, 16 }, .cases = 13, .warmups = 1, .measured_pairs = 5, .floor_bytes = floor_bytes, .scope = scope });
    try placementRun(gpu, model_dir, config, .{ .first = 1, .end = 4, .total = 40 }, journal);
    try placementRun(gpu, model_dir, config, .{ .first = 39, .end = 40, .total = 40 }, journal);
    try journal.write(.{ .kind = "prefill_linear_complete", .ordinal = gpu.ctx.device_ordinal, .cases = 13, .correctness_pass = true, .free_bytes = try floor(gpu), .scope = scope });
}

test "linear screen inputs are finite distinct at every supported width" {
    for ([_]usize{ 512, 2048, 4096 }) |k| {
        const bytes = try std.testing.allocator.alloc(u8, max_rows * k * 2);
        defer std.testing.allocator.free(bytes);
        for (0..max_rows) |r| for (0..k) |c| std.mem.writeInt(u16, bytes[(r * k + c) * 2 ..][0..2], inputBits(r, c), .little);
        try finite(bytes);
        try distinct(bytes, k * 2, max_rows);
    }
}

test "linear screen rejects output mismatch NaN and trailing writes" {
    const expected_bytes = [_]u8{ 0, 0x3f };
    var actual = [_]u8{ 0, 0x3f, 0xff, 0xff };
    try checkOutput(&actual, &expected_bytes);
    actual[0] = 1;
    try std.testing.expectError(error.PrefillLinearMismatch, checkOutput(&actual, &expected_bytes));
    actual[0] = 0xff;
    actual[1] = 0xff;
    try std.testing.expectError(error.NonfiniteLinearOutput, checkOutput(&actual, &expected_bytes));
    actual = .{ 0, 0x3f, 0, 0xff };
    try std.testing.expectError(error.PrefillLinearGuardChanged, checkOutput(&actual, &expected_bytes));
}

test "linear encoded comparison rejects mismatches NaN and guard writes" {
    const reference = [_]u8{ 0x38, 0x40 };
    var actual = [_]u8{ 0x38, 0x40, 0xff };
    try checkEncoded(&actual, &reference, 2);
    actual[0] = 0x39;
    try std.testing.expectError(error.LinearQuantizedRowsMismatch, checkEncoded(&actual, &reference, 2));
    actual = .{ 0x38, 0x40, 0 };
    try std.testing.expectError(error.LinearQuantizedGuardChanged, checkEncoded(&actual, &reference, 2));
    const nan = [_]u8{0x7f};
    try std.testing.expectError(error.NonfiniteLinearEncoded, checkEncoded(&nan, &nan, 1));
}

test "linear readback rejects unexpected writes anywhere in unused partial storage" {
    var output = [_]u8{ 0, 0x3f, 0xff, 0xff };
    var encoded = [_]u8{0xff};
    var partial = [_]u8{ 0xff, 0xff, 0xff };
    const reference = [_]u8{ 0, 0x3f };
    const w: cuda.qmmf.Weight = .{ .mode = .bf16, .codes = 0, .scales = 0, .n = 1, .k = 1, .npad = 1 };
    const f: Fixture = .{ .linear = .{}, .projection = .{ .bf16 = w }, .w = w, .input = 0, .output = 0, .encoded = 0, .part = 0 };
    const b: Readback = .{ .output = undefined, .encoded = undefined, .part = undefined, .actual = &output, .quant_actual = &encoded, .part_actual = &partial };
    try b.validate(f, &reference, &.{}, 1);
    for (&partial) |*byte| {
        byte.* = 0;
        try std.testing.expectError(error.LinearUnexpectedPartialWrite, b.validate(f, &reference, &.{}, 1));
        byte.* = 0xff;
    }
}

test "linear graph ABI comparison rejects corruption of every parameter" {
    const stub: cuda.Function = .{ .d = undefined, .handle = @ptrFromInt(1) };
    var lane: cuda.qmmf.Lane = undefined;
    lane.clusters = true;
    lane.fns = @splat(@splat(stub));
    var ck: cuda.ck.Ck = undefined;
    ck.clusters = true;
    ck.lane = @splat(stub);
    ck.quant8_fn = stub;
    const w: cuda.qmmf.Weight = .{ .mode = .fp8, .codes = 0x20000, .scales = 0, .scale = 0.5, .n = 512, .k = 2048, .npad = 512, .act = 0.125 };
    var fixture: Fixture = .{ .linear = .{ .lane = &lane, .ck = &ck }, .projection = .{ .fp8 = w }, .w = w, .input = 0x10000, .output = 0x30000, .encoded = 0x40000, .part = 0x50000 };
    for (0..3) |kind| {
        if (kind == 2) {
            fixture.w.mode = .bf16;
            fixture.w.act = 0;
            fixture.projection = .{ .bf16 = fixture.w };
        }
        const want = try expected(fixture, 3, 2, if (kind == 1) 1 else 0);
        try std.testing.expectEqual(@as(usize, if (kind == 0) 6 else if (kind == 1) 10 else 13), want.args.count);
        var actual = want.args;
        var p: cuda.abi.KernelNodeParams = std.mem.zeroes(cuda.abi.KernelNodeParams);
        p.params = actual.pointers();
        _ = try verifyArgs(p, &want.args);
        for (0..actual.count) |i| {
            actual.storage[actual.offsets[i]] ^= 1;
            try std.testing.expectError(error.LinearGraphArgumentMismatch, verifyArgs(p, &want.args));
            actual.storage[actual.offsets[i]] ^= 1;
        }
    }
}
