const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const X = cuda.experts.Nvfp4Experts;
const dims = 2048;
const stride = dims + 64;
const width = 512;
const slots = 8;
const experts = 256;
const banks = experts / slots;
const launches = banks * 4;
const guard_bytes = 64;
const floor_bytes = 1536 << 20;
const FilePin = struct { path: []const u8, bytes: usize = 0, sha256: []const u8 };
const Manifest = struct {
    format_version: usize,
    scope: []const u8,
    representation: []const u8,
    original_representation: []const u8,
    inverse_sha256: []const u8,
    expert_order: []const usize,
    files: std.json.ArrayHashMap(FilePin),
    frozen_image: FilePin,
};
const Arm = enum { frozen, control, slice_major };
const arms = [_]Arm{ .frozen, .control, .slice_major };

fn hash(bytes: []const u8) [64]u8 {
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &digest, .{});
    return std.fmt.bytesToHex(digest, .lower);
}

fn verify(bytes: []const u8, expected: []const u8) !void {
    if (!std.mem.eql(u8, &hash(bytes), expected)) return error.SliceMajorHashMismatch;
}

pub fn validatePaths(dir: []const u8, manifest_sha: []const u8, image: []const u8, image_sha: []const u8) !void {
    if (!std.fs.path.isAbsolute(dir) or !std.fs.path.isAbsolute(image) or manifest_sha.len != 64 or image_sha.len != 64) return error.InvalidSliceMajorInput;
    for (manifest_sha) |c| if (!std.ascii.isHex(c)) return error.InvalidSliceMajorInput;
    for (image_sha) |c| if (!std.ascii.isHex(c)) return error.InvalidSliceMajorInput;
}

fn readFile(gpu: check.Gpu, path: []const u8, limit: usize) ![]u8 {
    return std.Io.Dir.cwd().readFileAlloc(gpu.io, path, gpu.gpa, .limited(limit));
}

fn read(gpu: check.Gpu, manifest: *const Manifest, name: []const u8, size: usize) ![]u8 {
    const pin = manifest.files.map.get(name) orelse return error.MissingSliceMajorFixture;
    if (!std.fs.path.isAbsolute(pin.path) or pin.bytes != size) return error.InvalidSliceMajorFixture;
    const bytes = try readFile(gpu, pin.path, size + 1);
    errdefer gpu.gpa.free(bytes);
    if (bytes.len != size) return error.InvalidSliceMajorFixture;
    try verify(bytes, pin.sha256);
    return bytes;
}

fn save(gpu: check.Gpu, journal: anytype, name: []const u8, bytes: []const u8) !void {
    const filename = try std.fmt.allocPrint(gpu.gpa, "slice_major_gpu{d}_{s}.bin", .{ gpu.ctx.device_ordinal, name });
    defer gpu.gpa.free(filename);
    const path = try std.fs.path.join(gpu.gpa, &.{ journal.directory, filename });
    defer gpu.gpa.free(path);
    const file = try std.Io.Dir.cwd().createFile(gpu.io, path, .{ .exclusive = true });
    defer file.close(gpu.io);
    try file.writeStreamingAll(gpu.io, bytes);
    try journal.write(.{ .kind = "slice_major_artifact", .ordinal = gpu.ctx.device_ordinal, .name = name, .file = filename, .bytes = bytes.len, .sha256 = &hash(bytes) });
}

fn floor(gpu: check.Gpu) !usize {
    const free = (try gpu.ctx.memInfo()).free;
    if (free < floor_bytes) return error.GpuMemoryFloor;
    return free;
}

fn finite(bytes: []const u8, fp32: bool) !void {
    const element: usize = if (fp32) 4 else 2;
    if (bytes.len % element != 0) return error.InvalidSliceMajorScalar;
    var offset: usize = 0;
    while (offset < bytes.len) : (offset += element) {
        const bits = if (fp32) std.mem.readInt(u32, bytes[offset..][0..4], .little) else @as(u32, std.mem.readInt(u16, bytes[offset..][0..2], .little)) << 16;
        if (!std.math.isFinite(@as(f32, @bitCast(bits)))) return error.NonfiniteSliceMajorOutput;
    }
}

fn validateManifest(manifest: *const Manifest) !void {
    if (manifest.format_version != 1 or !std.mem.eql(u8, manifest.representation, "slice-major-u32-per144-v1") or
        !std.mem.eql(u8, manifest.original_representation, "lane-major-u32-per144-v1") or manifest.files.map.count() != 5 or manifest.expert_order.len != experts) return error.InvalidSliceMajorFixture;
    const original = manifest.files.map.get("up.original.bin") orelse return error.MissingSliceMajorFixture;
    if (!std.mem.eql(u8, original.sha256, manifest.inverse_sha256)) return error.InvalidSliceMajorFixture;
    var seen: [experts]bool = @splat(false);
    for (manifest.expert_order) |expert| {
        if (expert >= experts or seen[expert]) return error.InvalidSliceMajorFixture;
        seen[expert] = true;
    }
}

fn originalWord(word: usize) usize {
    return 4 * (word % 32) + word / 32;
}

fn verifyPermutation(original: []const u8, candidate: []const u8) !void {
    if (original.len != candidate.len or original.len % (144 * 4) != 0 or original.ptr == candidate.ptr) return error.InvalidSliceMajorPermutation;
    var offset: usize = 0;
    while (offset < original.len) : (offset += 144 * 4) {
        for (0..128) |word| {
            if (!std.mem.eql(u8, candidate[offset + word * 4 ..][0..4], original[offset + originalWord(word) * 4 ..][0..4])) return error.InvalidSliceMajorPermutation;
        }
        if (!std.mem.eql(u8, candidate[offset + 128 * 4 ..][0 .. 16 * 4], original[offset + 128 * 4 ..][0 .. 16 * 4])) return error.InvalidSliceMajorPermutation;
    }
}

const Guarded = struct {
    buffer: cuda.DeviceBuffer,
    bytes: usize,
    fn init(gpu: check.Gpu, bytes: usize) !Guarded {
        var buffer = try cuda.DeviceBuffer.alloc(gpu.d, bytes + guard_bytes * 2);
        errdefer buffer.free();
        const result: Guarded = .{ .buffer = buffer, .bytes = bytes };
        try result.poison();
        return result;
    }
    fn ptr(b: Guarded) u64 {
        return b.buffer.ptr + guard_bytes;
    }
    fn poison(b: Guarded) !void {
        try b.buffer.fill8(0xa5, null);
        const active: cuda.DeviceBuffer = .{ .d = b.buffer.d, .ptr = b.ptr(), .len = b.bytes, .borrowed = true };
        try active.fill8(0xff, null);
    }
    fn payload(b: Guarded, bytes: []const u8) ![]const u8 {
        if (bytes.len != b.bytes + 2 * guard_bytes) return error.InvalidSliceMajorReadback;
        for (bytes[0..guard_bytes]) |v| if (v != 0xa5) return error.SliceMajorGuardChanged;
        for (bytes[guard_bytes + b.bytes ..]) |v| if (v != 0xa5) return error.SliceMajorGuardChanged;
        return bytes[guard_bytes..][0..b.bytes];
    }
    fn deinit(b: *Guarded) void {
        b.buffer.free();
    }
};

const Binding = struct {
    arm: Arm,
    function: cuda.Function,
    weights: u64,
    scales: u64,
    resident: usize,
    fn grid(b: Binding) usize {
        return @min(b.resident, 128);
    }
    fn layer(b: Binding, bank: usize) cuda.experts.Layer {
        return .{ .format = .nvfp4, .up = b.weights + bank * slots * X.bytes(1, width, dims, 2), .down = 0, .up_scale = b.scales + bank * slots * 8, .width = width, .dims = dims, .experts = slots };
    }
    fn args(b: Binding, rows: cuda.experts.Rows, bank: usize, plan: cuda.grouped.Plan, output: u64, skip: i32) cuda.Args {
        return X.decodeArgs(.up, rows, b.layer(bank), plan, output, skip);
    }
    fn config(b: Binding) cuda.Config {
        return .{ .grid = .{ .x = @intCast(b.grid()), .y = 1, .z = 1 }, .block = .{ .x = 256, .y = 1, .z = 1 } };
    }
    fn launch(b: Binding, stream: cuda.Stream, rows: cuda.experts.Rows, bank: usize, plan: cuda.grouped.Plan, output: u64, skip: i32) !void {
        var arguments = b.args(rows, bank, plan, output, skip);
        try cuda.launch.launch(b.function, b.config(), stream, &arguments);
    }
};

fn distinctRow(original: []const u8, out: []u8, row: usize) void {
    const mask: u16 = @intCast((row * 17) & 127);
    for (0..dims) |column| {
        const bits = std.mem.readInt(u16, original[column * 2 ..][0..2], .little);
        std.mem.writeInt(u16, out[column * 2 ..][0..2], bits ^ mask, .little);
    }
}

fn inputRows(gpu: check.Gpu, original: []const u8, rows: usize) ![]u8 {
    const bytes = try gpu.gpa.alloc(u8, rows * stride * 2);
    errdefer gpu.gpa.free(bytes);
    @memset(bytes, 0xa5);
    for (0..rows) |row| distinctRow(original, bytes[row * stride * 2 ..][0 .. dims * 2], row);
    try finite(bytes, false);
    return bytes;
}

const PlanBuffers = struct {
    members: cuda.DeviceBuffer,
    items: cuda.DeviceBuffer,
    counts: cuda.DeviceBuffer,
    fn init(gpu: check.Gpu, rows: usize, journal: anytype) !PlanBuffers {
        var members: [16 * slots]i32 = undefined;
        var items: [slots][3]i32 = undefined;
        const counts = [_]i32{ slots, 0 };
        for (0..slots) |expert| {
            items[expert] = .{ @intCast(expert), @intCast(expert * rows), @intCast(rows) };
            for (0..rows) |row| members[expert * rows + row] = @intCast(row * slots + expert);
        }
        var p: PlanBuffers = undefined;
        p.members = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(members[0 .. rows * slots]));
        errdefer p.members.free();
        p.items = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.asBytes(&items));
        errdefer p.items.free();
        p.counts = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.asBytes(&counts));
        errdefer p.counts.free();
        const blobs = [_][]const u8{ std.mem.sliceAsBytes(members[0 .. rows * slots]), std.mem.asBytes(&items), std.mem.asBytes(&counts) };
        for ([_]cuda.DeviceBuffer{ p.members, p.items, p.counts }, blobs, 0..) |buffer, expected, index| {
            const got = try check.download(gpu, buffer);
            defer gpu.gpa.free(got);
            const name = try std.fmt.allocPrint(gpu.gpa, "rows{d}_plan{d}", .{ rows, index });
            defer gpu.gpa.free(name);
            try save(gpu, journal, name, got);
            try check.sameBytes("slice-major actual plan", got, expected);
        }
        try journal.write(.{ .kind = "slice_major_plan", .ordinal = gpu.ctx.device_ordinal, .rows = rows, .members = members[0 .. rows * slots], .items = items, .counts = counts, .bindings = p.plan(), .slots = slots });
        return p;
    }
    fn plan(p: PlanBuffers) cuda.grouped.Plan {
        return .{ .members = p.members.ptr, .items = p.items.ptr, .counts = p.counts.ptr, .rank = 0, .hist = 0 };
    }
    fn deinit(p: *PlanBuffers) void {
        p.counts.free();
        p.items.free();
        p.members.free();
    }
};

fn qualify(gpu: check.Gpu, stream: cuda.Stream, bindings: [3]Binding, rows_in: cuda.experts.Rows, plan: cuda.grouped.Plan, output: Guarded, rows: usize, saved: []const u8, phase: []const u8, journal: anytype) ![]u8 {
    const expected = try gpu.gpa.alloc(u8, output.bytes);
    errdefer gpu.gpa.free(expected);
    for (0..banks) |bank| {
        for (bindings, 0..) |binding, ai| {
            try output.poison();
            try binding.launch(stream, rows_in, bank, plan, output.ptr(), -1);
            try stream.synchronize();
            const raw = try check.download(gpu, output.buffer);
            defer gpu.gpa.free(raw);
            const name = try std.fmt.allocPrint(gpu.gpa, "{s}_rows{d}_bank{d}_{s}", .{ phase, rows, bank, @tagName(binding.arm) });
            defer gpu.gpa.free(name);
            try save(gpu, journal, name, raw);
            const active = try output.payload(raw);
            try finite(active, false);
            for (0..slots) |expert| for (0..rows) |row| for (row + 1..rows) |other| {
                const start = (row * slots + expert) * width * 2;
                const next = (other * slots + expert) * width * 2;
                if (std.mem.eql(u8, active[start..][0 .. width * 2], active[next..][0 .. width * 2])) return error.SliceMajorRowsNotDistinct;
            };
            if (ai == 0) @memcpy(expected, active) else try check.sameBytes("slice-major full bank", active, expected);
            if (bank == 0 and rows == 1) try check.sameBytes("slice-major saved bank0", active, saved);
            try journal.write(.{ .kind = "slice_major_bank", .ordinal = gpu.ctx.device_ordinal, .phase = phase, .rows = rows, .bank = bank, .arm = @tagName(binding.arm), .active_bytes = active.len, .guard_bytes = guard_bytes, .sha256 = &hash(active), .finite = true, .bit_exact = true, .saved_bank0 = bank == 0 and rows == 1, .free_bytes = try floor(gpu) });
        }
    }
    return expected;
}

fn kernelName(f: cuda.Function) ![]const u8 {
    var lib = f.d.lib;
    const get_name = lib.lookup(*const fn (*?[*:0]const u8, cuda.abi.Function) callconv(.c) cuda.abi.Result, "cuFuncGetName") orelse return error.MissingSymbol;
    var name: ?[*:0]const u8 = null;
    try f.d.check(get_name(&name, f.handle), "cuFuncGetName");
    return std.mem.span(name orelse return error.MissingSymbol);
}

fn checkNode(actual: cuda.abi.KernelNodeParams, binding: Binding, expected: *const cuda.Args) ![64]u8 {
    const cfg = binding.config();
    if (actual.func != binding.function.handle or actual.grid_x != cfg.grid.x or actual.grid_y != 1 or actual.grid_z != 1 or actual.block_x != 256 or actual.block_y != 1 or actual.block_z != 1 or actual.shared_bytes != 0) return error.SliceMajorGraphGeometry;
    if (expected.count != 14) return error.SliceMajorGraphArguments;
    const params = actual.params orelse return error.SliceMajorGraphArguments;
    var digest = std.crypto.hash.sha2.Sha256.init(.{});
    for (0..14) |index| {
        const bytes: [*]const u8 = @ptrCast(params[index] orelse return error.SliceMajorGraphArguments);
        const want = expected.storage[expected.offsets[index]..][0..expected.sizes[index]];
        if (!std.mem.eql(u8, bytes[0..want.len], want)) return error.SliceMajorGraphArguments;
        digest.update(bytes[0..want.len]);
    }
    var sha: [32]u8 = undefined;
    digest.final(&sha);
    return std.fmt.bytesToHex(sha, .lower);
}

fn graph(gpu: check.Gpu, binding: Binding, rows_in: cuda.experts.Rows, plan: cuda.grouped.Plan, output: u64, rows: usize, journal: anytype) !cuda.graph.Exec {
    var source = try cuda.graph.Graph.init(gpu.d);
    defer source.deinit();
    var nodes: [launches]cuda.graph.Node = undefined;
    var lib = gpu.d.lib;
    const dependencies = lib.lookup(*const fn (cuda.graph.Node, ?[*]cuda.graph.Node, *usize) callconv(.c) cuda.abi.Result, "cuGraphNodeGetDependencies") orelse return error.MissingSymbol;
    for (0..launches) |index| {
        const bank = index % banks;
        var args = binding.args(rows_in, bank, plan, output, -1);
        nodes[index] = try source.addKernel(if (index == 0) &.{} else nodes[index - 1 .. index], binding.function, binding.config(), &args);
        var actual: cuda.abi.KernelNodeParams = undefined;
        try gpu.d.check(gpu.d.api.cuGraphKernelNodeGetParams_v2(nodes[index], &actual), "cuGraphKernelNodeGetParams");
        const digest = try checkNode(actual, binding, &args);
        var count: usize = 1;
        var parent: [1]cuda.graph.Node = undefined;
        try gpu.d.check(dependencies(nodes[index], &parent, &count), "cuGraphNodeGetDependencies");
        if (count != @intFromBool(index != 0) or (index > 0 and parent[0] != nodes[index - 1])) return error.SliceMajorGraphDependency;
        var values: [14]u64 = undefined;
        for (&values, 0..) |*value, argument| value.* = args.integer(argument) orelse return error.SliceMajorGraphArguments;
        try journal.write(.{ .kind = "slice_major_node", .ordinal = gpu.ctx.device_ordinal, .rows = rows, .arm = @tagName(binding.arm), .index = index, .bank = bank, .function_handle = @intFromPtr(actual.func), .symbol = try kernelName(binding.function), .grid = binding.config().grid, .threads = 256, .shared_bytes = actual.shared_bytes, .argument_values = values, .argument_sizes = args.sizes[0..14], .argument_sha256 = &digest, .dependencies = count, .layout = if (binding.arm == .slice_major) "slice-major-u32-per144-v1" else "lane-major-u32-per144-v1" });
    }
    if (try source.nodeCount() != launches) return error.SliceMajorGraphWork;
    return source.instantiate();
}

fn timed(gpu: check.Gpu, stream: cuda.Stream, bindings: [3]Binding, rows_in: cuda.experts.Rows, plan: cuda.grouped.Plan, output: Guarded, rows: usize, expected: []const u8, journal: anytype) !void {
    var graphs: [2]cuda.graph.Exec = undefined;
    var made: usize = 0;
    defer for (graphs[0..made]) |*g| g.deinit();
    for (bindings[1..], 0..) |binding, index| {
        graphs[index] = try graph(gpu, binding, rows_in, plan, output.ptr(), rows, journal);
        made += 1;
        try graphs[index].upload(stream);
    }
    try stream.synchronize();
    var start = try cuda.Event.init(gpu.d, true);
    defer start.deinit();
    var end = try cuda.Event.init(gpu.d, true);
    defer end.deinit();
    var events: [2][5]f64 = undefined;
    var walls: [2][5]f64 = undefined;
    for (0..6) |pair| for (0..2) |order| {
        const ai = (pair + order) % 2;
        try output.poison();
        try stream.synchronize();
        const t0 = check.now(gpu.io);
        try start.record(stream);
        try graphs[ai].launchOn(stream);
        try end.record(stream);
        try end.synchronize();
        const wall_ms = @as(f64, @floatFromInt(check.now(gpu.io) - t0)) / 1e6;
        const event_ms = try cuda.Event.elapsedMs(start, end);
        if (!std.math.isFinite(event_ms) or event_ms <= 0 or !std.math.isFinite(wall_ms) or wall_ms <= 0) return error.InvalidSliceMajorTiming;
        const raw = try check.download(gpu, output.buffer);
        defer gpu.gpa.free(raw);
        const name = try std.fmt.allocPrint(gpu.gpa, "timed_rows{d}_pair{d}_{s}", .{ rows, pair, @tagName(bindings[ai + 1].arm) });
        defer gpu.gpa.free(name);
        try save(gpu, journal, name, raw);
        const active = try output.payload(raw);
        try finite(active, false);
        try check.sameBytes("slice-major timed final bank31", active, expected);
        if (pair > 0) {
            events[ai][pair - 1] = event_ms;
            walls[ai][pair - 1] = wall_ms;
        }
        try journal.write(.{ .kind = "slice_major_sample", .ordinal = gpu.ctx.device_ordinal, .rows = rows, .arm = @tagName(bindings[ai + 1].arm), .pair = pair, .order = order, .warmup = pair == 0, .event_ms = event_ms, .wall_ms = wall_ms, .launches = launches, .event_us_per_call = event_ms * 1000 / launches, .final_bank = banks - 1, .active_bytes = output.bytes, .guard_bytes = guard_bytes, .sha256 = &hash(active), .bit_exact = true, .finite = true, .free_bytes = try floor(gpu) });
    };
    for (0..2) |index| {
        std.mem.sort(f64, &events[index], {}, std.sort.asc(f64));
        std.mem.sort(f64, &walls[index], {}, std.sort.asc(f64));
        try journal.write(.{ .kind = "slice_major_summary", .ordinal = gpu.ctx.device_ordinal, .rows = rows, .arm = @tagName(bindings[index + 1].arm), .samples = 5, .launches_per_sample = launches, .event_ms_median = events[index][2], .event_ms_min = events[index][0], .event_ms_max = events[index][4], .wall_ms_median = walls[index][2], .wall_ms_min = walls[index][0], .wall_ms_max = walls[index][4], .model_speedup_claim = false });
    }
}

fn general(gpu: check.Gpu, stream: cuda.Stream, bindings: [3]Binding, input: []const u8, journal: anytype) !void {
    const count = 18;
    var members: [count]i32 = undefined;
    for (&members, 0..) |*value, index| value.* = @intCast(index);
    const items = [2][3]i32{ .{ 0, 0, 17 }, .{ 1, 17, 1 } };
    const counts = [2]i32{ 2, 0 };
    var mb = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.asBytes(&members));
    defer mb.free();
    var ib = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.asBytes(&items));
    defer ib.free();
    var cb = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.asBytes(&counts));
    defer cb.free();
    const plan: cuda.grouped.Plan = .{ .members = mb.ptr, .items = ib.ptr, .counts = cb.ptr, .rank = 0, .hist = 0 };
    for ([_]cuda.DeviceBuffer{ mb, ib, cb }, [_][]const u8{ std.mem.asBytes(&members), std.mem.asBytes(&items), std.mem.asBytes(&counts) }, 0..) |buffer, expected, index| {
        const actual = try check.download(gpu, buffer);
        defer gpu.gpa.free(actual);
        const name = try std.fmt.allocPrint(gpu.gpa, "general_plan{d}", .{index});
        defer gpu.gpa.free(name);
        try save(gpu, journal, name, actual);
        try check.sameBytes("slice-major general plan", actual, expected);
    }
    const host = try inputRows(gpu, input, count);
    defer gpu.gpa.free(host);
    var device = try cuda.DeviceBuffer.fromHost(gpu.d, host);
    defer device.free();
    const actual_input = try check.download(gpu, device);
    defer gpu.gpa.free(actual_input);
    try save(gpu, journal, "general_input", actual_input);
    try check.sameBytes("slice-major general input", actual_input, host);
    var output = try Guarded.init(gpu, count * width * 2);
    defer output.deinit();
    const expected = try gpu.gpa.alloc(u8, output.bytes);
    defer gpu.gpa.free(expected);
    for ([_]i32{ -1, 0, 1 }) |skip| for (bindings, 0..) |original, ai| {
        var binding = original;
        binding.resident = 1;
        try output.poison();
        try binding.launch(stream, .{ .x = device.ptr, .stride = stride, .slots = 0 }, 0, plan, output.ptr(), skip);
        try stream.synchronize();
        const raw = try check.download(gpu, output.buffer);
        defer gpu.gpa.free(raw);
        const name = try std.fmt.allocPrint(gpu.gpa, "general_skip{d}_{s}", .{ skip, @tagName(binding.arm) });
        defer gpu.gpa.free(name);
        try save(gpu, journal, name, raw);
        const active = try output.payload(raw);
        if (ai == 0) @memcpy(expected, active) else try check.sameBytes("slice-major count17/skip", active, expected);
        for (0..count) |row| {
            const bytes = active[row * width * 2 ..][0 .. width * 2];
            const inactive = if (row < 17) skip == 0 else skip == 1;
            if (inactive) {
                for (bytes) |value| if (value != 0xff) return error.SliceMajorInactiveWrite;
            } else try finite(bytes, false);
        }
        if (skip != 0) for (0..17) |row| for (row + 1..17) |other| {
            if (std.mem.eql(u8, active[row * width * 2 ..][0 .. width * 2], active[other * width * 2 ..][0 .. width * 2])) return error.SliceMajorRowsNotDistinct;
        };
        const actual_args = binding.args(.{ .x = device.ptr, .stride = stride, .slots = 0 }, 0, plan, output.ptr(), skip);
        var values: [14]u64 = undefined;
        for (&values, 0..) |*value, argument| value.* = actual_args.integer(argument) orelse return error.SliceMajorGraphArguments;
        try journal.write(.{ .kind = "slice_major_general", .input_ptr = device.ptr, .output_ptr = output.ptr(), .output_allocation_ptr = output.buffer.ptr, .symbol = try kernelName(binding.function), .function_handle = @intFromPtr(binding.function.handle), .argument_values = values, .argument_sizes = actual_args.sizes[0..14], .launch_binding_evidence = "same Args builder as direct launch; timed graph nodes independently read back", .ordinal = gpu.ctx.device_ordinal, .arm = @tagName(binding.arm), .rows = count, .stride = stride, .slots = 0, .members = members, .items = items, .counts = counts, .bindings = plan, .skip = skip, .grid = 1, .active_bytes = output.bytes, .guard_bytes = guard_bytes, .bit_exact = true, .inactive_poison = true });
    };
}

pub fn run(gpu: check.Gpu, dir: []const u8, manifest_sha: []const u8, image_path: []const u8, image_sha: []const u8, journal: anytype) !void {
    try validatePaths(dir, manifest_sha, image_path, image_sha);
    _ = try floor(gpu);
    const path = try std.fs.path.join(gpu.gpa, &.{ dir, "manifest.json" });
    defer gpu.gpa.free(path);
    const manifest_raw = try readFile(gpu, path, 1 << 20);
    defer gpu.gpa.free(manifest_raw);
    try verify(manifest_raw, manifest_sha);
    const parsed = try std.json.parseFromSlice(Manifest, gpu.gpa, manifest_raw, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    const manifest = &parsed.value;
    try validateManifest(manifest);
    const one = X.bytes(1, width, dims, 2);
    const original = try read(gpu, manifest, "up.original.bin", experts * one);
    defer gpu.gpa.free(original);
    const candidate = try read(gpu, manifest, "up.slice-major.bin", experts * one);
    defer gpu.gpa.free(candidate);
    try verifyPermutation(original, candidate);
    const scales = try read(gpu, manifest, "up-scales.bin", experts * 8);
    defer gpu.gpa.free(scales);
    try finite(scales, true);
    const input = try read(gpu, manifest, "input.bin", dims * 2);
    defer gpu.gpa.free(input);
    try finite(input, false);
    const saved = try read(gpu, manifest, "hidden.bin", slots * width * 2);
    defer gpu.gpa.free(saved);
    try finite(saved, false);
    var original_weights = try cuda.DeviceBuffer.fromHost(gpu.d, original);
    defer original_weights.free();
    var candidate_weights = try cuda.DeviceBuffer.fromHost(gpu.d, candidate);
    defer candidate_weights.free();
    var global_scales = try cuda.DeviceBuffer.fromHost(gpu.d, scales);
    defer global_scales.free();
    for ([_]cuda.DeviceBuffer{ original_weights, candidate_weights, global_scales }, [_][]const u8{ original, candidate, scales }, [_][]const u8{ "consumed_original_weights", "consumed_slice_major_weights", "consumed_scales" }) |buffer, expected, name| {
        const actual = try check.download(gpu, buffer);
        defer gpu.gpa.free(actual);
        try save(gpu, journal, name, actual);
        try check.sameBytes("slice-major consumed upload", actual, expected);
    }
    try verify(cuda.kernels.nvfp4_experts, manifest.frozen_image.sha256);
    var frozen = try cuda.Module.load(gpu.d, cuda.kernels.nvfp4_experts);
    defer frozen.unload();
    const image_raw = try readFile(gpu, image_path, 8 << 20);
    defer gpu.gpa.free(image_raw);
    try verify(image_raw, image_sha);
    const aligned = try gpu.gpa.alignedAlloc(u8, .@"16", image_raw.len);
    defer gpu.gpa.free(aligned);
    @memcpy(aligned, image_raw);
    var image = try cuda.Module.load(gpu.d, aligned);
    defer image.unload();
    const functions = [_]cuda.Function{ try frozen.function("tf_nvfp4_n8_split_m_swiglu"), try image.function("tf_q35_slice_major_control"), try image.function("tf_q35_slice_major_candidate") };
    const sms: usize = @intCast(try gpu.ctx.attribute(.multiprocessor_count));
    var bindings: [3]Binding = undefined;
    for (functions, arms, &bindings) |function, arm, *binding| {
        const blocks = try function.occupancy(256, 0);
        if (blocks == 0 or try function.attribute(.shared_size_bytes) != 2048 or try function.attribute(.local_size_bytes) != 0) return error.SliceMajorKernelResources;
        binding.* = .{ .arm = arm, .function = function, .weights = if (arm == .slice_major) candidate_weights.ptr else original_weights.ptr, .scales = global_scales.ptr, .resident = blocks * sms };
        if (binding.grid() != 128) return error.SliceMajorSchedulingChanged;
        try journal.write(.{ .kind = "slice_major_resources", .ordinal = gpu.ctx.device_ordinal, .arm = @tagName(arm), .function_handle = @intFromPtr(function.handle), .symbol = try kernelName(function), .regs = try function.attribute(.num_regs), .local_bytes = try function.attribute(.local_size_bytes), .static_shared_bytes = try function.attribute(.shared_size_bytes), .dynamic_shared_bytes = 0, .sms = sms, .resident_per_sm = blocks, .resident_blocks = binding.resident, .grid = binding.grid(), .threads = 256, .weight_ptr = binding.weights, .scale_ptr = binding.scales });
    }
    try journal.write(.{ .kind = "slice_major_scope", .ordinal = gpu.ctx.device_ordinal, .manifest_sha256 = manifest_sha, .image_sha256 = image_sha, .frozen_image_sha256 = manifest.frozen_image.sha256, .scope = manifest.scope, .expert_order = manifest.expert_order, .l2_bytes = try gpu.ctx.attribute(.l2_cache_size), .unique_weight_bytes_per_arm = experts * one, .logical_weight_bytes_per_launch = slots * one, .logical_scales_per_launch = slots * 8, .banks = banks, .launches = launches, .guard_bytes = guard_bytes, .free_bytes = try floor(gpu), .performance_scope = "isolated resident-weight up operator; no hardware limiter or model speedup claim" });
    var stream = try cuda.Stream.init(gpu.d, false);
    defer stream.deinit();
    try general(gpu, stream, bindings, input, journal);
    const Shape = struct {
        device_rows: cuda.DeviceBuffer,
        plan: PlanBuffers,
        output: Guarded,
        expected: []u8,
        rows: usize,
        fn rowsIn(shape: @This()) cuda.experts.Rows {
            return .{ .x = shape.device_rows.ptr, .stride = stride, .slots = slots };
        }
        fn deinit(shape: *@This(), gpa: std.mem.Allocator) void {
            gpa.free(shape.expected);
            shape.output.deinit();
            shape.plan.deinit();
            shape.device_rows.free();
        }
    };
    var shapes: [3]Shape = undefined;
    var made: usize = 0;
    defer for (shapes[0..made]) |*shape| shape.deinit(gpu.gpa);
    for ([_]usize{ 1, 2, 16 }, 0..) |rows, index| {
        const host_rows = try inputRows(gpu, input, rows);
        defer gpu.gpa.free(host_rows);
        var device_rows = try cuda.DeviceBuffer.fromHost(gpu.d, host_rows);
        errdefer device_rows.free();
        const actual_rows = try check.download(gpu, device_rows);
        defer gpu.gpa.free(actual_rows);
        const name = try std.fmt.allocPrint(gpu.gpa, "rows{d}_input", .{rows});
        defer gpu.gpa.free(name);
        try save(gpu, journal, name, actual_rows);
        try check.sameBytes("slice-major actual input", actual_rows, host_rows);
        var plan = try PlanBuffers.init(gpu, rows, journal);
        errdefer plan.deinit();
        var output = try Guarded.init(gpu, rows * slots * width * 2);
        errdefer output.deinit();
        const rows_in: cuda.experts.Rows = .{ .x = device_rows.ptr, .stride = stride, .slots = slots };
        try journal.write(.{ .kind = "slice_major_shape_binding", .ordinal = gpu.ctx.device_ordinal, .rows = rows, .input_ptr = rows_in.x, .input_bytes = device_rows.len, .stride = rows_in.stride, .slots = rows_in.slots, .output_ptr = output.ptr(), .output_allocation_ptr = output.buffer.ptr, .output_bytes = output.bytes, .guard_bytes = guard_bytes, .plan = plan.plan() });
        const expected = try qualify(gpu, stream, bindings, rows_in, plan.plan(), output, rows, saved, "before", journal);
        shapes[index] = .{ .device_rows = device_rows, .plan = plan, .output = output, .expected = expected, .rows = rows };
        made += 1;
    }
    try journal.write(.{ .kind = "slice_major_qualified_before_timing", .ordinal = gpu.ctx.device_ordinal, .shapes = 3, .banks_per_shape = banks, .arms = 3, .raw_outputs = 288 });
    for (shapes) |shape| try timed(gpu, stream, bindings, shape.rowsIn(), shape.plan.plan(), shape.output, shape.rows, shape.expected, journal);
    for (shapes) |shape| {
        const after = try qualify(gpu, stream, bindings, shape.rowsIn(), shape.plan.plan(), shape.output, shape.rows, saved, "after", journal);
        defer gpu.gpa.free(after);
        try check.sameBytes("slice-major before/after final bank", after, shape.expected);
    }
    for ([_]cuda.DeviceBuffer{ original_weights, candidate_weights, global_scales }, [_][]const u8{ original, candidate, scales }, [_][]const u8{ "original_weights", "slice_major_weights", "scales" }) |buffer, expected, name| {
        const actual = try check.download(gpu, buffer);
        defer gpu.gpa.free(actual);
        try check.sameBytes("slice-major immutable consumed operands", actual, expected);
        try journal.write(.{ .kind = "slice_major_inputs_unchanged", .ordinal = gpu.ctx.device_ordinal, .name = name, .bytes = actual.len, .sha256 = &hash(actual), .bit_exact = true });
    }
    try journal.write(.{ .kind = "slice_major_complete", .ordinal = gpu.ctx.device_ordinal, .rows = [_]usize{ 1, 2, 16 }, .banks_per_shape_per_phase = banks, .qualification_phases = 2, .timed_arms = 2, .frozen_control_untimed = true, .measured_pairs = 5, .warmup_pairs = 1, .all_outputs_exact = true, .free_bytes = try floor(gpu), .production_qualification = false });
}

test "slice-major permutation has an inverse and excludes all scale words" {
    var original: [576]u8 = undefined;
    var transformed: [576]u8 = undefined;
    for (0..144) |word| std.mem.writeInt(u32, original[word * 4 ..][0..4], @intCast(word), .little);
    for (0..128) |word| @memcpy(transformed[word * 4 ..][0..4], original[originalWord(word) * 4 ..][0..4]);
    @memcpy(transformed[512..], original[512..]);
    try verifyPermutation(&original, &transformed);
    transformed[0] ^= 1;
    try std.testing.expectError(error.InvalidSliceMajorPermutation, verifyPermutation(&original, &transformed));
    transformed[0] ^= 1;
    transformed[512] ^= 1;
    try std.testing.expectError(error.InvalidSliceMajorPermutation, verifyPermutation(&original, &transformed));
}

test "slice-major paths reject before CUDA and output poison is nonfinite" {
    try std.testing.expectError(error.InvalidSliceMajorInput, validatePaths("relative", "a", "relative", "b"));
    try std.testing.expectError(error.NonfiniteSliceMajorOutput, finite(&.{ 0xff, 0xff }, false));
}

test "slice-major capture rejects every ABI and geometry mutation" {
    const binding: Binding = .{ .arm = .slice_major, .function = .{ .d = undefined, .handle = @ptrFromInt(1) }, .weights = 4096, .scales = 128, .resident = 128 };
    const plan: cuda.grouped.Plan = .{ .items = 300, .counts = 400, .members = 500, .rank = 0, .hist = 0 };
    const expected = binding.args(.{ .x = 1024, .stride = stride, .slots = slots }, 31, plan, 600, -1);
    var actual = expected;
    var params: cuda.abi.KernelNodeParams = .{ .func = @ptrFromInt(1), .grid_x = 128, .grid_y = 1, .grid_z = 1, .block_x = 256, .block_y = 1, .block_z = 1, .shared_bytes = 0, .params = actual.pointers(), .extra = null };
    _ = try checkNode(params, binding, &expected);
    for (0..14) |argument| {
        actual.storage[actual.offsets[argument]] ^= 1;
        try std.testing.expectError(error.SliceMajorGraphArguments, checkNode(params, binding, &expected));
        actual.storage[actual.offsets[argument]] ^= 1;
    }
    inline for (.{ "grid_x", "grid_y", "grid_z", "block_x", "block_y", "block_z", "shared_bytes" }) |field| {
        @field(params, field) += 1;
        try std.testing.expectError(error.SliceMajorGraphGeometry, checkNode(params, binding, &expected));
        @field(params, field) -= 1;
    }
    params.func = @ptrFromInt(2);
    try std.testing.expectError(error.SliceMajorGraphGeometry, checkNode(params, binding, &expected));
    params.func = @ptrFromInt(1);
    params.params = null;
    try std.testing.expectError(error.SliceMajorGraphArguments, checkNode(params, binding, &expected));
}

test "slice-major row control preserves BF16 exponent and varies eighteen rows" {
    var original: [dims * 2]u8 = undefined;
    for (0..dims) |column| std.mem.writeInt(u16, original[column * 2 ..][0..2], 0x3f80, .little);
    var rows: [18][dims * 2]u8 = undefined;
    for (&rows, 0..) |*row, index| {
        distinctRow(&original, row, index);
        try finite(row, false);
        for (0..dims) |column| try std.testing.expectEqual(@as(u16, 0x3f80), std.mem.readInt(u16, row[column * 2 ..][0..2], .little) & 0xff80);
        for (rows[0..index]) |prior| try std.testing.expect(!std.mem.eql(u8, row, &prior));
    }
}
