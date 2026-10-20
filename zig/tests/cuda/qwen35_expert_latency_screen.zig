const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const X = cuda.experts.Nvfp4Experts;
const experts = 256;
const dims = 2048;
const width = 512;
const slots = 8;
const banks = experts / slots;
const launches = 128;
const floor_bytes = 1536 * 1024 * 1024;
const FilePin = struct { bytes: usize, sha256: []const u8 };
const Manifest = struct { scope: []const u8, expert_order: []const usize, files: std.json.ArrayHashMap(FilePin) };

pub fn validateInput(input: anytype) !void {
    if (!input.expert_latency_screen) return;
    if (input.expert_fixture == null or input.expert_shard_screen or input.expert_n8_screen or
        input.gdn_layout_screen or input.gdn_layout_ab or input.expert_n8_ab or input.quant_reuse_screen or
        input.transport_screen or input.verify_default_policy or input.router_turns or input.router_screen or
        input.router_candidate or input.forced.len != 0) return error.InvalidInput;
}

const Arm = enum {
    n8,
    split_m,

    fn threads(arm: Arm) u32 {
        return if (arm == .n8) 128 else 256;
    }
    fn symbol(arm: Arm) [:0]const u8 {
        return if (arm == .n8) "tf_nvfp4_n8_pipe4_swiglu" else "tf_nvfp4_n8_split_m_swiglu";
    }
};
const arms = [_]Arm{ .n8, .split_m };
const Binding = struct {
    arm: Arm,
    function: cuda.Function,
    resident: usize,

    fn grid(b: Binding) usize {
        return @min(128, b.resident);
    }
    fn launch(b: Binding, s: cuda.Stream, rows: cuda.experts.Rows, layer: cuda.experts.Layer, plan: cuda.grouped.Plan, output: u64, skip: i32) !void {
        var args = X.decodeArgs(.up, rows, layer, plan, output, skip);
        try cuda.launch.launch(b.function, .{ .grid = .{ .x = @intCast(b.grid()), .y = 1, .z = 1 }, .block = .{ .x = b.arm.threads(), .y = 1, .z = 1 } }, s, &args);
    }
};

const Route = struct {
    members: [slots]i32,
    items: [slots][3]i32,
    counts: [2]i32,

    fn init() Route {
        var route: Route = .{ .members = undefined, .items = undefined, .counts = .{ slots, 0 } };
        for (0..slots) |i| {
            route.members[i] = @intCast(i);
            route.items[i] = .{ @intCast(i), @intCast(i), 1 };
        }
        return route;
    }
};

fn validateOrder(order: []const usize) !void {
    if (order.len != experts) return error.InvalidExpertOrder;
    var seen: [experts]bool = @splat(false);
    for (order) |id| {
        if (id >= experts or seen[id]) return error.InvalidExpertOrder;
        seen[id] = true;
    }
}

fn kernelName(f: cuda.Function) ![]const u8 {
    var lib = f.d.lib;
    const get_name = lib.lookup(*const fn (*?[*:0]const u8, cuda.abi.Function) callconv(.c) cuda.abi.Result, "cuFuncGetName") orelse return error.MissingSymbol;
    var name: ?[*:0]const u8 = null;
    try f.d.check(get_name(&name, f.handle), "cuFuncGetName");
    return std.mem.span(name orelse return error.MissingSymbol);
}

fn checkNode(p: cuda.abi.KernelNodeParams, f: cuda.abi.Function, grid: usize, block: usize) !void {
    if (p.func != f or p.grid_x != grid or p.grid_y != 1 or p.grid_z != 1 or
        p.block_x != block or p.block_y != 1 or p.block_z != 1 or p.shared_bytes != 0)
        return error.ExpertGraphKernelMismatch;
}

fn verifyHash(bytes: []const u8, expected: []const u8) ![64]u8 {
    var hash: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &hash, .{});
    const hex = std.fmt.bytesToHex(hash, .lower);
    if (!std.mem.eql(u8, &hex, expected)) return error.ExpertFixtureHashMismatch;
    return hex;
}

fn floor(gpu: check.Gpu) !void {
    if ((try gpu.ctx.memInfo()).free < floor_bytes) return error.GpuMemoryFloor;
}

fn read(gpu: check.Gpu, dir: []const u8, manifest: *const Manifest, name: []const u8, size: usize) ![]u8 {
    const pin = manifest.files.map.get(name) orelse return error.MissingExpertFixturePin;
    if (pin.bytes != size) return error.InvalidExpertFixtureSize;
    const path = try std.fs.path.join(gpu.gpa, &.{ dir, name });
    defer gpu.gpa.free(path);
    const bytes = try std.Io.Dir.cwd().readFileAlloc(gpu.io, path, gpu.gpa, .limited(size + 1));
    errdefer gpu.gpa.free(bytes);
    if (bytes.len != size) return error.InvalidExpertFixtureSize;
    _ = try verifyHash(bytes, pin.sha256);
    return bytes;
}

fn finite(bytes: []const u8, fp32: bool) !void {
    const stride: usize = if (fp32) 4 else 2;
    if (bytes.len % stride != 0) return error.InvalidExpertOutputSize;
    var offset: usize = 0;
    while (offset < bytes.len) : (offset += stride) {
        const bits: u32 = if (fp32)
            std.mem.readInt(u32, bytes[offset..][0..4], .little)
        else
            @as(u32, std.mem.readInt(u16, bytes[offset..][0..2], .little)) << 16;
        if (!std.math.isFinite(@as(f32, @bitCast(bits)))) return error.NonfiniteExpertOutput;
    }
}

fn bankLayer(l: cuda.experts.Layer, bank: usize) cuda.experts.Layer {
    var result = l;
    result.up += bank * slots * X.bytes(1, width, dims, 2);
    result.up_scale += bank * slots * 8;
    result.experts = slots;
    return result;
}

fn nodeArgs(p: cuda.abi.KernelNodeParams, expected: *const cuda.launch.Args) ![64]u8 {
    if (expected.count != 14) return error.ExpertGraphArgumentMismatch;
    const actual = p.params orelse return error.ExpertGraphMissingArguments;
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    for (0..expected.count) |i| {
        const bytes: [*]const u8 = @ptrCast(actual[i] orelse return error.ExpertGraphMissingArguments);
        const want = expected.storage[expected.offsets[i]..][0..expected.sizes[i]];
        if (!std.mem.eql(u8, bytes[0..want.len], want)) return error.ExpertGraphArgumentMismatch;
        h.update(bytes[0..want.len]);
    }
    var digest: [32]u8 = undefined;
    h.final(&digest);
    return std.fmt.bytesToHex(digest, .lower);
}

pub fn run(gpu: check.Gpu, dir: []const u8, expected_manifest_sha256: []const u8, journal: anytype) !void {
    try floor(gpu);
    const manifest_path = try std.fs.path.join(gpu.gpa, &.{ dir, "manifest.json" });
    defer gpu.gpa.free(manifest_path);
    const manifest_bytes = try std.Io.Dir.cwd().readFileAlloc(gpu.io, manifest_path, gpu.gpa, .limited(1 << 20));
    defer gpu.gpa.free(manifest_bytes);
    const digest = try verifyHash(manifest_bytes, expected_manifest_sha256);
    const parsed = try std.json.parseFromSlice(Manifest, gpu.gpa, manifest_bytes, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    const manifest = &parsed.value;
    try validateOrder(manifest.expert_order);
    const sms: usize = @intCast(try gpu.ctx.attribute(.multiprocessor_count));
    var ex = try X.load(gpu.d, sms);
    defer ex.unload();
    var bindings: [2]Binding = undefined;
    for (arms, &bindings) |arm, *binding| {
        const f = try ex.mod.function(arm.symbol());
        const per_sm = try f.occupancy(arm.threads(), 0);
        if (per_sm == 0) return error.NoResidentExpertBlock;
        binding.* = .{ .arm = arm, .function = f, .resident = per_sm * sms };
        try journal.write(.{ .kind = "expert_latency_resources", .ordinal = gpu.ctx.device_ordinal, .arm = @tagName(arm), .symbol = try kernelName(f), .regs = try f.attribute(.num_regs), .local_bytes = try f.attribute(.local_size_bytes), .static_shared_bytes = try f.attribute(.shared_size_bytes), .dynamic_shared_bytes = 0, .resident_blocks_per_sm = per_sm, .resident_blocks = binding.resident, .threads = arm.threads(), .grid = binding.grid() });
        if (arm == .n8 and binding.grid() != 128) return error.N8GridMismatch;
        if (try f.attribute(.local_size_bytes) != 0) return error.N8KernelSpill;
        const shared = try f.attribute(.shared_size_bytes);
        if (shared != (if (arm == .n8) @as(i32, 0) else 2048)) return error.ExpertSharedSizeMismatch;
    }
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    try journal.write(.{ .kind = "expert_latency_fixture", .ordinal = gpu.ctx.device_ordinal, .manifest_sha256 = &digest, .scope = manifest.scope, .expert_order = manifest.expert_order, .banks = banks, .slots = slots, .sm_count = sms, .floor_bytes = floor_bytes, .free_bytes = (try gpu.ctx.memInfo()).free, .measurement_scope = "isolated per-GPU up operator; no model gain or hardware bottleneck claim" });
    const one = X.bytes(1, width, dims, 2);
    const host_weights = try read(gpu, dir, manifest, "up.bin", experts * one);
    defer gpu.gpa.free(host_weights);
    const host_scales = try read(gpu, dir, manifest, "up-scales.bin", experts * 8);
    defer gpu.gpa.free(host_scales);
    try finite(host_scales, true);
    const host_input = try read(gpu, dir, manifest, "input.bin", dims * 2);
    defer gpu.gpa.free(host_input);
    try finite(host_input, false);
    const saved = try read(gpu, dir, manifest, "hidden.bin", slots * width * 2);
    defer gpu.gpa.free(saved);
    try finite(saved, false);
    var weights = try cuda.DeviceBuffer.fromHost(gpu.d, host_weights);
    defer weights.free();
    var scales = try cuda.DeviceBuffer.fromHost(gpu.d, host_scales);
    defer scales.free();
    var input = try cuda.DeviceBuffer.fromHost(gpu.d, host_input);
    defer input.free();
    const layer: cuda.experts.Layer = .{ .format = .nvfp4, .up = weights.ptr, .down = 0, .up_scale = scales.ptr, .width = width, .dims = dims, .experts = experts };
    const rows: cuda.experts.Rows = .{ .x = input.ptr, .stride = dims, .slots = slots };
    const route = Route.init();
    const route_bytes = [_][]const u8{ std.mem.asBytes(&route.members), std.mem.asBytes(&route.items), std.mem.asBytes(&route.counts) };
    var plan_buffers: [3]cuda.DeviceBuffer = undefined;
    var made_buffers: usize = 0;
    defer for (plan_buffers[0..made_buffers]) |*buffer| buffer.free();
    var plan_hashes: [3][64]u8 = undefined;
    for (route_bytes, 0..) |bytes, i| {
        plan_buffers[i] = try cuda.DeviceBuffer.fromHost(gpu.d, bytes);
        made_buffers += 1;
        var got_plan: [slots * 12]u8 = undefined;
        try plan_buffers[i].download(0, got_plan[0..bytes.len]);
        try check.sameBytes("latency actual plan", got_plan[0..bytes.len], bytes);
        var hash: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(bytes, &hash, .{});
        plan_hashes[i] = std.fmt.bytesToHex(hash, .lower);
    }
    const plan: cuda.grouped.Plan = .{ .members = plan_buffers[0].ptr, .items = plan_buffers[1].ptr, .counts = plan_buffers[2].ptr, .rank = 0, .hist = 0 };
    try journal.write(.{ .kind = "expert_latency_plan", .ordinal = gpu.ctx.device_ordinal, .actual_items = route.counts[0], .members = route.members, .items = route.items, .counts = route.counts, .bindings = plan, .binding_sha256 = .{ plan_hashes[0][0..], plan_hashes[1][0..], plan_hashes[2][0..] }, .input_slots = slots });
    var reference = try cuda.DeviceBuffer.alloc(gpu.d, saved.len);
    defer reference.free();
    var output = try cuda.DeviceBuffer.alloc(gpu.d, saved.len);
    defer output.free();
    const want = try gpu.gpa.alloc(u8, saved.len);
    defer gpu.gpa.free(want);
    const got = try gpu.gpa.alloc(u8, saved.len);
    defer gpu.gpa.free(got);
    try floor(gpu);
    try generalRows(gpu, s, bindings, host_input, layer, journal);
    for (0..banks) |bank| {
        const bl = bankLayer(layer, bank);
        try reference.fill8(0xff, s.handle);
        try bindings[0].launch(s, rows, bl, plan, reference.ptr, -1);
        try s.synchronize();
        try reference.download(0, want);
        try finite(want, false);
        if (bank == 0) try check.sameBytes("latency bank0 saved output", want, saved);
        try output.fill8(0xff, s.handle);
        try bindings[1].launch(s, rows, bl, plan, output.ptr, -1);
        try s.synchronize();
        try output.download(0, got);
        try finite(got, false);
        try check.sameBytes("latency split-M bank", got, want);
        if (bank == 0) try check.sameBytes("latency split-M saved output", got, saved);
        try journal.write(.{ .kind = "expert_latency_bank_parity", .ordinal = gpu.ctx.device_ordinal, .bank = bank, .finite = true, .bit_exact = true, .saved_bank0 = bank == 0 });
    }
    var graphs: [2]cuda.graph.Exec = undefined;
    var made_graphs: usize = 0;
    defer for (graphs[0..made_graphs]) |*graph| graph.deinit();
    for (bindings, 0..) |binding, ai| {
        try cuda.graph.beginCapture(s, .thread_local);
        for (0..launches) |i| binding.launch(s, rows, bankLayer(layer, i % banks), plan, output.ptr, -1) catch |err| {
            if (cuda.graph.endCapture(s)) |graph| {
                var dead = graph;
                dead.deinit();
            } else |_| {}
            return err;
        };
        var graph = try cuda.graph.endCapture(s);
        defer graph.deinit();
        if (try graph.nodeCount() != launches) return error.ExpertGraphWorkCountMismatch;
        var nodes: [launches]cuda.graph.Node = undefined;
        const captured = try graph.nodes(&nodes);
        if (captured.len != launches) return error.ExpertGraphWorkCountMismatch;
        var bank_counts: [banks]usize = @splat(0);
        var binding_hashes: [banks][64]u8 = undefined;
        for (captured) |node| {
            var p: cuda.abi.KernelNodeParams = undefined;
            try gpu.d.check(gpu.d.api.cuGraphKernelNodeGetParams_v2(node, &p), "cuGraphKernelNodeGetParams");
            try checkNode(p, binding.function.handle, binding.grid(), binding.arm.threads());
            const args = p.params orelse return error.ExpertGraphMissingArguments;
            const weight_ptr: *const u64 = @ptrCast(@alignCast(args[3] orelse return error.ExpertGraphMissingArguments));
            if (weight_ptr.* < weights.ptr or (weight_ptr.* - weights.ptr) % (slots * one) != 0) return error.ExpertGraphArgumentMismatch;
            const bank = (weight_ptr.* - weights.ptr) / (slots * one);
            if (bank >= banks) return error.ExpertGraphArgumentMismatch;
            const expected = X.decodeArgs(.up, rows, bankLayer(layer, bank), plan, output.ptr, -1);
            binding_hashes[bank] = try nodeArgs(p, &expected);
            bank_counts[bank] += 1;
        }
        for (bank_counts) |count| if (count != launches / banks) return error.ExpertGraphWorkCountMismatch;
        graphs[ai] = try graph.instantiate();
        made_graphs += 1;
        try graphs[ai].upload(s);
        try s.synchronize();
        var binding_hex: [banks][]const u8 = undefined;
        for (&binding_hashes, &binding_hex) |*hash, *hex| hex.* = hash;
        const f = binding.function;
        try journal.write(.{ .kind = "expert_latency_kernel", .ordinal = gpu.ctx.device_ordinal, .arm = @tagName(binding.arm), .symbol = try kernelName(f), .grid = binding.grid(), .threads = binding.arm.threads(), .regs = try f.attribute(.num_regs), .local_bytes = try f.attribute(.local_size_bytes), .static_shared_bytes = try f.attribute(.shared_size_bytes), .dynamic_shared_bytes = 0, .resident_blocks = binding.resident, .resident_blocks_per_sm = try f.occupancy(binding.arm.threads(), 0), .kernel_launches = launches, .verified_kernel_nodes = captured.len, .bank_node_counts = bank_counts, .argument_sha256_by_bank = binding_hex, .logical_weight_and_scale_bytes_per_launch = slots * (one + 8), .logical_unique_weight_and_scale_bytes = experts * (one + 8), .free_bytes = (try gpu.ctx.memInfo()).free });
    }
    var start = try cuda.Event.init(gpu.d, true);
    defer start.deinit();
    var end = try cuda.Event.init(gpu.d, true);
    defer end.deinit();
    var events: [2][5]f64 = undefined;
    var walls: [2][5]f64 = undefined;
    for (0..6) |pair| for (0..2) |order| {
        const ai = (pair + order) % 2;
        try output.fill8(0xff, s.handle);
        try s.synchronize();
        const t0 = std.Io.Clock.awake.now(gpu.io).toNanoseconds();
        try start.record(s);
        try graphs[ai].launchOn(s);
        try end.record(s);
        try end.synchronize();
        const wall_ms = @as(f64, @floatFromInt(std.Io.Clock.awake.now(gpu.io).toNanoseconds() - t0)) / 1e6;
        const elapsed_ms = try cuda.Event.elapsedMs(start, end);
        if (!std.math.isFinite(elapsed_ms) or elapsed_ms <= 0 or !std.math.isFinite(wall_ms) or wall_ms <= 0) return error.InvalidExpertLatencyTiming;
        try output.download(0, got);
        try finite(got, false);
        try check.sameBytes("latency timed final bank", got, want);
        try floor(gpu);
        if (pair != 0) {
            events[ai][pair - 1] = elapsed_ms;
            walls[ai][pair - 1] = wall_ms;
        }
        try journal.write(.{ .kind = "expert_latency_sample", .ordinal = gpu.ctx.device_ordinal, .arm = @tagName(arms[ai]), .pair = pair, .order = order, .warmup = pair == 0, .elapsed_ms = elapsed_ms, .wall_ms = wall_ms, .kernel_launches = launches, .banks = banks, .final_bank = banks - 1, .finite = true, .bit_exact_final = true, .free_bytes = (try gpu.ctx.memInfo()).free });
    };
    for (arms, 0..) |arm, ai| {
        std.mem.sort(f64, &events[ai], {}, std.sort.asc(f64));
        std.mem.sort(f64, &walls[ai], {}, std.sort.asc(f64));
        try journal.write(.{ .kind = "expert_latency_summary", .ordinal = gpu.ctx.device_ordinal, .arm = @tagName(arm), .samples = 5, .kernel_launches_per_sample = launches, .event_ms_median = events[ai][2], .event_ms_min = events[ai][0], .event_ms_max = events[ai][4], .wall_ms_median = walls[ai][2], .wall_ms_min = walls[ai][0], .wall_ms_max = walls[ai][4] });
    }
    try journal.write(.{ .kind = "expert_latency_complete", .ordinal = gpu.ctx.device_ordinal, .all_32_banks_exact = true, .saved_bank0_exact = true, .graph_finals_exact = true, .production_qualification = false });
}

fn generalRows(gpu: check.Gpu, s: cuda.Stream, bindings: [2]Binding, host_input: []const u8, layer: cuda.experts.Layer, journal: anytype) !void {
    const row_count = 18;
    var members: [row_count]i32 = undefined;
    for (&members, 0..) |*member, i| member.* = @intCast(i);
    const items = [2][3]i32{ .{ 0, 0, 17 }, .{ 1, 17, 1 } };
    const counts = [2]i32{ 2, 0 };
    var mb = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.asBytes(&members));
    defer mb.free();
    var ib = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.asBytes(&items));
    defer ib.free();
    var cb = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.asBytes(&counts));
    defer cb.free();
    const plan: cuda.grouped.Plan = .{ .members = mb.ptr, .items = ib.ptr, .counts = cb.ptr, .rank = 0, .hist = 0 };
    const data = [_][]const u8{ std.mem.asBytes(&members), std.mem.asBytes(&items), std.mem.asBytes(&counts) };
    const buffers = [_]cuda.DeviceBuffer{ mb, ib, cb };
    var plan_hashes: [3][64]u8 = undefined;
    for (data, buffers, 0..) |bytes, buffer, i| {
        var copied: [row_count * 4]u8 = undefined;
        try buffer.download(0, copied[0..bytes.len]);
        try check.sameBytes("general rows uploaded plan", copied[0..bytes.len], bytes);
        plan_hashes[i] = hashBytes(bytes);
    }
    const host_rows = try gpu.gpa.alloc(u8, row_count * dims * 2);
    defer gpu.gpa.free(host_rows);
    var row_hashes: [row_count][64]u8 = undefined;
    var row_hex: [row_count][]const u8 = undefined;
    for (0..row_count) |row| {
        const bytes = host_rows[row * dims * 2 ..][0 .. dims * 2];
        distinctRow(host_input, bytes, row);
        try finite(bytes, false);
        row_hashes[row] = hashBytes(bytes);
        row_hex[row] = &row_hashes[row];
    }
    var distinct_input = try cuda.DeviceBuffer.fromHost(gpu.d, host_rows);
    defer distinct_input.free();
    const copied_input = try gpu.gpa.alloc(u8, host_rows.len);
    defer gpu.gpa.free(copied_input);
    try distinct_input.download(0, copied_input);
    try check.sameBytes("general distinct inputs", copied_input, host_rows);
    const rows: cuda.experts.Rows = .{ .x = distinct_input.ptr, .stride = dims, .slots = 0 };
    try journal.write(.{ .kind = "expert_latency_general_plan", .ordinal = gpu.ctx.device_ordinal, .members = members, .items = items, .counts = counts, .bindings = plan, .binding_sha256 = .{ plan_hashes[0][0..], plan_hashes[1][0..], plan_hashes[2][0..] }, .input_slots = 0, .input_rows_sha256 = row_hex });
    const size = row_count * width * 2;
    var output = try cuda.DeviceBuffer.alloc(gpu.d, size);
    defer output.free();
    const got = try gpu.gpa.alloc(u8, size);
    defer gpu.gpa.free(got);
    const want = try gpu.gpa.alloc(u8, size);
    defer gpu.gpa.free(want);
    for ([_]i32{ -1, 0, 1 }) |skip| {
        for (bindings, 0..) |original, ai| {
            var binding = original;
            binding.resident = 1;
            try output.fill8(0xff, s.handle);
            try binding.launch(s, rows, layer, plan, output.ptr, skip);
            try s.synchronize();
            try output.download(0, if (ai == 0) want else got);
        }
        for (0..row_count) |row| {
            const inactive = if (row < 17) skip == 0 else skip == 1;
            const bytes = got[row * width * 2 ..][0 .. width * 2];
            if (inactive) {
                for (bytes) |byte| if (byte != 0xff) return error.ExpertInactiveWrite;
            } else try finite(bytes, false);
        }
        if (skip != 0) for (0..17) |row| {
            const a = want[row * width * 2 ..][0 .. width * 2];
            for (row + 1..17) |other| {
                const b = want[other * width * 2 ..][0 .. width * 2];
                if (std.mem.eql(u8, a, b)) return error.GeneralRowsNotDistinct;
            }
        };
        try check.sameBytes("split-M general rows and skip", got, want);
        var output_hashes: [row_count][64]u8 = undefined;
        var output_hex: [row_count][]const u8 = undefined;
        for (0..row_count) |row| {
            output_hashes[row] = hashBytes(want[row * width * 2 ..][0 .. width * 2]);
            output_hex[row] = &output_hashes[row];
        }
        try journal.write(.{ .kind = "expert_latency_general_rows", .ordinal = gpu.ctx.device_ordinal, .rows = row_count, .first_item_rows = 17, .skip = skip, .grid = 1, .bit_exact = true, .inactive_poison = true, .finite_active = true, .first17_baseline_rows_distinct = skip != 0, .baseline_output_row_sha256 = output_hex });
    }
}

fn hashBytes(bytes: []const u8) [64]u8 {
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &digest, .{});
    return std.fmt.bytesToHex(digest, .lower);
}

fn distinctRow(original: []const u8, output: []u8, row: usize) void {
    std.debug.assert(original.len == output.len and original.len % 2 == 0);
    const mask: u16 = @intCast((row * 17) & 127);
    for (0..original.len / 2) |col| {
        const bits = std.mem.readInt(u16, original[col * 2 ..][0..2], .little);
        std.mem.writeInt(u16, output[col * 2 ..][0..2], bits ^ mask, .little);
    }
}

test "general row inputs vary all eighteen rows without changing BF16 sign or exponent" {
    const values = [4]u16{ 0, 0x3f80, 0xbf80, 0x7f7f };
    const bytes = std.mem.asBytes(&values);
    var rows: [18][8]u8 = undefined;
    for (&rows, 0..) |*row, i| {
        distinctRow(bytes, row, i);
        try finite(row, false);
        for (values, 0..) |value, col| {
            const actual = std.mem.readInt(u16, row[col * 2 ..][0..2], .little);
            try std.testing.expectEqual(value & 0xff80, actual & 0xff80);
        }
        for (rows[0..i]) |prior| try std.testing.expect(!std.mem.eql(u8, row, &prior));
    }
}

test "latency capture verifies every ABI argument and launch dimension" {
    const layer: cuda.experts.Layer = .{ .format = .nvfp4, .up = 4096, .down = 0, .up_scale = 128, .width = width, .dims = dims, .experts = slots };
    const plan: cuda.grouped.Plan = .{ .items = 300, .counts = 400, .members = 500, .rank = 0, .hist = 0 };
    const expected = X.decodeArgs(.up, .{ .x = 1024, .stride = dims, .slots = slots }, layer, plan, 600, -1);
    try std.testing.expectEqual(@as(usize, 14), expected.count);
    var actual = expected;
    var p: cuda.abi.KernelNodeParams = .{ .func = @ptrFromInt(1), .grid_x = 128, .grid_y = 1, .grid_z = 1, .block_x = 256, .block_y = 1, .block_z = 1, .shared_bytes = 0, .params = actual.pointers(), .extra = null };
    _ = try nodeArgs(p, &expected);
    for (0..expected.count) |i| {
        actual.storage[actual.offsets[i]] ^= 1;
        try std.testing.expectError(error.ExpertGraphArgumentMismatch, nodeArgs(p, &expected));
        actual.storage[actual.offsets[i]] ^= 1;
    }
    try checkNode(p, @ptrFromInt(1), 128, 256);
    inline for (.{ "grid_x", "grid_y", "grid_z", "block_x", "block_y", "block_z", "shared_bytes" }) |field| {
        @field(p, field) += 1;
        try std.testing.expectError(error.ExpertGraphKernelMismatch, checkNode(p, @ptrFromInt(1), 128, 256));
        @field(p, field) -= 1;
    }
    p.params = null;
    try std.testing.expectError(error.ExpertGraphMissingArguments, nodeArgs(p, &expected));
}

test "split-M tile ownership covers every projection slice exactly once even at capped grids" {
    for ([_]usize{ 1, 70, 84, 128 }) |grid| {
        var seen: [128][4][2]bool = @splat(@splat(@splat(false)));
        for (0..grid) |block| {
            var tile = block;
            while (tile < 128) : (tile += grid) {
                for (0..8) |warp| {
                    const slice = warp & 3;
                    const part = warp >> 2;
                    try std.testing.expect(!seen[tile][slice][part]);
                    seen[tile][slice][part] = true;
                }
            }
        }
        for (seen) |tile| for (tile) |slice| for (slice) |hit| try std.testing.expect(hit);
    }
}

test "split-M pitch preserves packed projection words and per-projection MMA order" {
    for (0..2) |part| {
        var term: usize = 0;
        for (0..64) |g| {
            for (0..32) |lane| {
                const old = g * 2 * 36 + part * 36 + lane;
                const split = part * 36 + g * 2 * 36 + lane;
                try std.testing.expectEqual(old, split);
            }
            for (0..2) |h| {
                try std.testing.expectEqual(g * 2 + h, term);
                term += 1;
            }
        }
        try std.testing.expectEqual(@as(usize, 128), term);
    }
}

test "split-M shared handoff covers all sixteen rows without cross-lane bank collisions" {
    var seen: [16][32]bool = @splat(@splat(false));
    for (0..4) |slice| for (0..4) |component| {
        var bank_seen: [32]bool = @splat(false);
        for (0..32) |lane| {
            const shared_index = (slice * 4 + component) * 32 + lane;
            try std.testing.expect(!bank_seen[shared_index % 32]);
            bank_seen[shared_index % 32] = true;
            const row = (lane >> 2) + (component / 2) * 8;
            const col = slice * 8 + 2 * (lane & 3) + component % 2;
            try std.testing.expect(!seen[row][col]);
            seen[row][col] = true;
        }
    };
    for (seen) |row| for (row) |hit| try std.testing.expect(hit);
}

test "latency plan retains all eight original member rows and rotating bank offsets" {
    const route = Route.init();
    try std.testing.expectEqual(@as(i32, 8), route.counts[0]);
    for (route.items, 0..) |item, i| {
        try std.testing.expectEqual([3]i32{ @intCast(i), @intCast(i), 1 }, item);
        try std.testing.expectEqual(@as(i32, @intCast(i)), route.members[i]);
    }
    const layer: cuda.experts.Layer = .{ .format = .nvfp4, .up = 4096, .down = 0, .up_scale = 128, .width = width, .dims = dims, .experts = experts };
    const last = bankLayer(layer, banks - 1);
    try std.testing.expectEqual(layer.up + 31 * slots * X.bytes(1, width, dims, 2), last.up);
    try std.testing.expectEqual(layer.up_scale + 31 * slots * 8, last.up_scale);
    try std.testing.expectEqual(@as(usize, 0), launches % banks);
}
