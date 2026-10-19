const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const X = cuda.experts.Nvfp4Experts;
const Arm = enum {
    wn4_baseline,
    wn4_n16_pipe4,
    wn1_baseline,
    wn4_n8_pipe4,

    fn warps(self: Arm) usize {
        return switch (self) {
            .wn1_baseline => 1,
            else => 4,
        };
    }

    fn variant(self: Arm) X.Variant {
        std.debug.assert(self != .wn4_n8_pipe4);
        return if (self == .wn4_n16_pipe4) .n16_pipe4 else .baseline;
    }

    fn engine(self: Arm, engines: *const [2]X) *const X {
        return &engines[if (self == .wn1_baseline) @as(usize, 1) else 0];
    }
};
const arms = [_]Arm{ .wn4_baseline, .wn4_n16_pipe4, .wn1_baseline };
const n8_arms = [_]Arm{ .wn4_baseline, .wn4_n16_pipe4, .wn4_n8_pipe4 };
const experts = 256;
const dims = 2048;
const width = 512;
const slots = 8;
const banks = experts / slots;
const launches = 128;
const bound = cuda.grouped.maxItems(slots, experts, 16);
const floor_bytes = 1536 * 1024 * 1024;
const FilePin = struct { bytes: usize, sha256: []const u8 };
const Manifest = struct { scope: []const u8, expert_order: []const usize, files: std.json.ArrayHashMap(FilePin) };

const N8Up = struct {
    f: cuda.Function,
    resident: usize,

    fn grid(resident: usize) usize {
        return @min((bound * (width / X.cols) * 4 + 3) / 4, resident);
    }

    fn args(rows: cuda.experts.Rows, layer: cuda.experts.Layer, plan: cuda.grouped.Plan, out: u64) cuda.launch.Args {
        var a: cuda.launch.Args = .{};
        a.add(rows.x);
        a.add(@as(i32, @intCast(rows.stride)));
        a.add(@as(i32, @intCast(rows.slots)));
        a.add(layer.up);
        a.add(layer.up_scale);
        a.add(@as(i32, dims / 32));
        a.add(@as(i32, width / X.cols));
        for ([_]u64{ plan.items, plan.counts, plan.members, out }) |v| a.add(v);
        a.add(@as(i32, width));
        a.add(layer.limit);
        a.add(@as(i32, -1));
        return a;
    }

    fn run(self: N8Up, s: cuda.Stream, rows: cuda.experts.Rows, layer: cuda.experts.Layer, plan: cuda.grouped.Plan, out: u64) !void {
        var a = args(rows, layer, plan, out);
        try cuda.launch.launch(self.f, .{ .grid = .{ .x = @intCast(grid(self.resident)), .y = 1, .z = 1 }, .block = .{ .x = 128, .y = 1, .z = 1 } }, s, &a);
    }
};

fn runArm(arm: Arm, n8: ?N8Up, engines: *const [2]X, s: cuda.Stream, epi: cuda.experts.Epi, rows: cuda.experts.Rows, layer: cuda.experts.Layer, plan: cuda.grouped.Plan, out: u64) !void {
    if (arm == .wn4_n8_pipe4) {
        std.debug.assert(epi == .up);
        return (n8 orelse return error.N8KernelNotLoaded).run(s, rows, layer, plan, out);
    }
    return arm.engine(engines).runVariant(arm.variant(), s, epi, rows, layer, plan, out, bound, -1);
}

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
    result.down += bank * slots * X.bytes(1, dims, width, 1);
    result.up_scale += bank * slots * 8;
    result.down_scale += bank * slots * 4;
    result.experts = slots;
    return result;
}

pub fn run(gpu: check.Gpu, fixture_dir: []const u8, expected_manifest_sha256: []const u8, journal: anytype) !void {
    return runMode(false, gpu, fixture_dir, expected_manifest_sha256, journal);
}

pub fn runN8(gpu: check.Gpu, fixture_dir: []const u8, expected_manifest_sha256: []const u8, journal: anytype) !void {
    return runMode(true, gpu, fixture_dir, expected_manifest_sha256, journal);
}

fn runMode(comptime n8_mode: bool, gpu: check.Gpu, fixture_dir: []const u8, expected_manifest_sha256: []const u8, journal: anytype) !void {
    try floor(gpu);
    const manifest_path = try std.fs.path.join(gpu.gpa, &.{ fixture_dir, "manifest.json" });
    defer gpu.gpa.free(manifest_path);
    const manifest_bytes = try std.Io.Dir.cwd().readFileAlloc(gpu.io, manifest_path, gpu.gpa, .limited(1 << 20));
    defer gpu.gpa.free(manifest_bytes);
    const manifest_sha256 = try verifyHash(manifest_bytes, expected_manifest_sha256);
    const manifest = try std.json.parseFromSlice(Manifest, gpu.gpa, manifest_bytes, .{ .ignore_unknown_fields = true });
    defer manifest.deinit();
    try validateOrder(manifest.value.expert_order);
    try journal.write(.{ .kind = if (n8_mode) "expert_n8_fixture" else "expert_fixture", .ordinal = gpu.ctx.device_ordinal, .manifest_sha256 = &manifest_sha256, .scope = manifest.value.scope, .expert_order = manifest.value.expert_order, .bank0_original_ids = manifest.value.expert_order[0..slots], .bank0_remapped_ids = [8]usize{ 0, 1, 2, 3, 4, 5, 6, 7 } });
    const sms: usize = @intCast(try gpu.ctx.attribute(.multiprocessor_count));
    var engines: [2]X = undefined;
    engines[0] = try X.loadWarps(gpu.d, sms, 4);
    defer engines[0].unload();
    try engines[0].loadCandidates(sms);
    if (!n8_mode) engines[1] = try X.loadWarps(gpu.d, sms, 1);
    defer if (!n8_mode) engines[1].unload();
    const n8: ?N8Up = if (n8_mode) blk: {
        const f = try engines[0].mod.function("tf_nvfp4_n8_pipe4_swiglu");
        if (try f.attribute(.local_size_bytes) != 0) return error.N8KernelSpill;
        const resident = @as(usize, @intCast(try f.occupancy(128, 0))) * sms;
        if (N8Up.grid(resident) != 128) return error.N8GridMismatch;
        break :blk .{ .f = f, .resident = resident };
    } else null;
    var mod = try cuda.Module.load(gpu.d, cuda.kernels.experts);
    defer mod.unload();
    const router = try cuda.grouped.Router.resolve(mod);
    var stream = try cuda.Stream.init(gpu.d, false);
    defer stream.deinit();
    const epis = if (n8_mode) &[_]cuda.experts.Epi{.up} else &[_]cuda.experts.Epi{ .up, .down_f32 };
    for (epis) |epi| try projection(n8_mode, n8, gpu, fixture_dir, &manifest.value, journal, &engines, router, stream, epi);
    try floor(gpu);
}

fn projection(comptime n8_mode: bool, n8: ?N8Up, gpu: check.Gpu, dir: []const u8, manifest: *const Manifest, journal: anytype, engines: *const [2]X, router: cuda.grouped.Router, s: cuda.Stream, epi: cuda.experts.Epi) !void {
    const selected_arms = if (n8_mode) n8_arms else arms;
    const up = epi == .up;
    const n: usize = if (up) width else dims;
    const k: usize = if (up) dims else width;
    const parts: usize = if (up) 2 else 1;
    const one = X.bytes(1, n, k, parts);
    const host_weights = try read(gpu, dir, manifest, if (up) "up.bin" else "down.bin", experts * one);
    defer gpu.gpa.free(host_weights);
    const host_scales = try read(gpu, dir, manifest, if (up) "up-scales.bin" else "down-scales.bin", experts * parts * 4);
    defer gpu.gpa.free(host_scales);
    try finite(host_scales, true);
    const host_input = try read(gpu, dir, manifest, if (up) "input.bin" else "hidden.bin", (if (up) @as(usize, 1) else slots) * k * 2);
    defer gpu.gpa.free(host_input);
    try finite(host_input, false);
    const saved = try read(gpu, dir, manifest, if (up) "hidden.bin" else "pairs.bin", slots * n * (if (up) @as(usize, 2) else 4));
    defer gpu.gpa.free(saved);
    try finite(saved, !up);
    var weights = try cuda.DeviceBuffer.fromHost(gpu.d, host_weights);
    defer weights.free();
    var scales = try cuda.DeviceBuffer.fromHost(gpu.d, host_scales);
    defer scales.free();
    var input = try cuda.DeviceBuffer.fromHost(gpu.d, host_input);
    defer input.free();
    const layer: cuda.experts.Layer = .{ .format = .nvfp4, .up = weights.ptr, .down = weights.ptr, .up_scale = scales.ptr, .down_scale = scales.ptr, .width = width, .dims = dims, .experts = experts, .limit = 0, .relu2 = false };
    const rows: cuda.experts.Rows = .{ .x = input.ptr, .stride = k, .slots = if (up) slots else 0 };
    var picks: [slots]i32 = undefined;
    for (&picks, 0..) |*p, i| p.* = @intCast(i);
    var dpicks = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.asBytes(&picks));
    defer dpicks.free();
    var members = try cuda.DeviceBuffer.alloc(gpu.d, slots * 4);
    defer members.free();
    var items = try cuda.DeviceBuffer.alloc(gpu.d, bound * 12);
    defer items.free();
    var counts = try cuda.DeviceBuffer.alloc(gpu.d, 8);
    defer counts.free();
    const plan: cuda.grouped.Plan = .{ .members = members.ptr, .items = items.ptr, .counts = counts.ptr, .rank = 0, .hist = 0 };
    try router.route(s, dpicks.ptr, slots, experts, 16, plan);
    try s.synchronize();
    var actual: [2]i32 = undefined;
    try counts.download(0, std.mem.asBytes(&actual));
    if (actual[0] != slots or bound != slots) return error.ExpertPlanCountMismatch;
    try journal.write(.{ .kind = if (n8_mode) "expert_n8_plan" else "expert_plan", .ordinal = gpu.ctx.device_ordinal, .epi = @tagName(epi), .actual_items = actual[0], .bound = bound, .picks = picks, .banks = banks, .limit = layer.limit, .relu2 = layer.relu2 });
    var reference = try cuda.DeviceBuffer.alloc(gpu.d, saved.len);
    defer reference.free();
    var output = try cuda.DeviceBuffer.alloc(gpu.d, saved.len);
    defer output.free();
    const want = try gpu.gpa.alloc(u8, saved.len);
    defer gpu.gpa.free(want);
    const got = try gpu.gpa.alloc(u8, saved.len);
    defer gpu.gpa.free(got);
    try floor(gpu);
    for (0..banks) |bank| {
        const bl = bankLayer(layer, bank);
        try reference.fill8(0xff, s.handle);
        try engines[0].runVariant(.baseline, s, epi, rows, bl, plan, reference.ptr, bound, -1);
        try s.synchronize();
        try reference.download(0, want);
        try finite(want, !up);
        if (bank == 0) try check.sameBytes("bank0 saved real expert output", want, saved);
        for (selected_arms[1..]) |arm| {
            try output.fill8(0xff, s.handle);
            try runArm(arm, n8, engines, s, epi, rows, bl, plan, output.ptr);
            try s.synchronize();
            try output.download(0, got);
            try finite(got, !up);
            try check.sameBytes(@tagName(arm), got, want);
            try journal.write(.{ .kind = if (n8_mode) "expert_n8_bank_parity" else "expert_bank_parity", .ordinal = gpu.ctx.device_ordinal, .epi = @tagName(epi), .bank = bank, .variant = @tagName(arm), .warps = arm.warps(), .finite = true, .bit_exact = true, .saved_real_output = bank == 0 });
        }
    }
    var graphs: [selected_arms.len]cuda.graph.Exec = undefined;
    var made: usize = 0;
    defer for (graphs[0..made]) |*graph| graph.deinit();
    for (selected_arms, 0..) |arm, vi| {
        const ex = arm.engine(engines);
        const variant = if (arm == .wn4_n8_pipe4) X.Variant.baseline else arm.variant();
        try cuda.graph.beginCapture(s, .thread_local);
        for (0..launches) |i| runArm(arm, n8, engines, s, epi, rows, bankLayer(layer, i % banks), plan, output.ptr) catch |err| {
            if (cuda.graph.endCapture(s)) |graph| {
                var dead = graph;
                dead.deinit();
            } else |_| {}
            return err;
        };
        var graph = try cuda.graph.endCapture(s);
        defer graph.deinit();
        if (try graph.nodeCount() != launches) return error.ExpertGraphWorkCountMismatch;
        graphs[vi] = try graph.instantiate();
        made += 1;
        try graphs[vi].upload(s);
        try s.synchronize();
        const ei: usize = @backingInt(epi);
        const variant_index = @backingInt(variant);
        const resident = if (arm == .wn4_n8_pipe4) n8.?.resident else if (variant == .baseline) ex.resident[ei] else ex.candidate_resident[variant_index - 1][ei];
        const f = if (arm == .wn4_n8_pipe4) n8.?.f else if (variant == .baseline) ex.fns[ei] else ex.candidate_fns[variant_index - 1][ei];
        const grid = if (arm == .wn4_n8_pipe4) N8Up.grid(resident) else X.decodeGridWarps(variant, bound, n / X.cols, resident, arm.warps());
        const block = arm.warps() * 32;
        var nodes: [launches]cuda.graph.Node = undefined;
        const captured = try graph.nodes(&nodes);
        if (captured.len != launches) return error.ExpertGraphWorkCountMismatch;
        var params: cuda.abi.KernelNodeParams = undefined;
        for (captured) |node| {
            try gpu.d.check(gpu.d.api.cuGraphKernelNodeGetParams_v2(node, &params), "cuGraphKernelNodeGetParams");
            try checkNode(params, f.handle, grid, block);
        }
        try journal.write(.{ .kind = if (n8_mode) "expert_n8_kernel" else "expert_kernel", .ordinal = gpu.ctx.device_ordinal, .epi = @tagName(epi), .variant = @tagName(arm), .warps = arm.warps(), .symbol = try kernelName(f), .grid = params.grid_x, .block = params.block_x, .regs = try f.attribute(.num_regs), .local_bytes = try f.attribute(.local_size_bytes), .resident_blocks = resident, .resident_blocks_per_sm = try f.occupancy(@intCast(block), 0), .kernel_launches = launches, .verified_kernel_nodes = captured.len, .banks = banks, .logical_weight_and_scale_bytes_per_launch = slots * (one + parts * 4), .logical_unique_weight_and_scale_bytes = experts * (one + parts * 4) });
    }
    var start = try cuda.Event.init(gpu.d, true);
    defer start.deinit();
    var end = try cuda.Event.init(gpu.d, true);
    defer end.deinit();
    for (selected_arms[1..], 1..) |arm, candidate_index| {
        for (0..6) |pair| for (0..2) |order| {
            const candidate = (pair + order) % 2 == 1;
            const vi = if (candidate) candidate_index else 0;
            try output.fill8(0xff, s.handle);
            try start.record(s);
            try graphs[vi].launchOn(s);
            try end.record(s);
            try end.synchronize();
            const elapsed_ms = try cuda.Event.elapsedMs(start, end);
            try output.download(0, got);
            try finite(got, !up);
            try check.sameBytes("timed graph final bank output", got, want);
            try floor(gpu);
            try journal.write(.{ .kind = if (n8_mode) "expert_n8_sample" else "expert_sample", .ordinal = gpu.ctx.device_ordinal, .epi = @tagName(epi), .comparison = @tagName(arm), .pair = pair, .warmup = pair == 0, .order = order, .arm = @tagName(selected_arms[vi]), .elapsed_ms = elapsed_ms, .kernel_launches = launches, .banks = banks, .final_bank = (launches - 1) % banks, .finite = true, .bit_exact_final = true });
        };
    }
    try journal.write(.{ .kind = if (n8_mode) "expert_n8_projection_complete" else "expert_projection_complete", .ordinal = gpu.ctx.device_ordinal, .epi = @tagName(epi), .all_bank_prechecks = true, .saved_bank0_exact = true, .graph_finals_exact = true, .timed_pairs_per_candidate = 5 });
}

test "Qwen N8 scalar slices recover every packed source column and scale" {
    const n = 64;
    const k = 64;
    var codes: [2][n * k / 2]u8 = undefined;
    var scales: [2][n * k / 16]u8 = undefined;
    for (0..2) |part| {
        for (&codes[part], 0..) |*code, i| code.* = @truncate(i * 29 + i / 17 + part * 101);
        for (&scales[part], 0..) |*scale, i| scale.* = @truncate(i * 7 + part * 13);
    }
    var words: [n / X.cols * (k / 32) * 2 * X.words]u32 = undefined;
    const shifts = [2][4]u5{ .{ 0, 16, 4, 20 }, .{ 8, 24, 12, 28 } };
    X.packGateUp(&words, .{ &codes[0], &scales[0] }, .{ &codes[1], &scales[1] }, n, k);
    for (0..n / X.cols) |cb| for (0..k / 32) |group| for (0..2) |part| {
        const block = words[((cb * (k / 32) + group) * 2 + part) * X.words ..][0..X.words];
        for (0..32) |lane| for (0..4) |slice| {
            const word = block[lane * 4 + slice];
            const row = cb * X.cols + slice * 8 + lane / 4;
            for (0..2) |half| for (0..4) |q| {
                const input = group * 32 + half * 16 + (lane % 4) * 4 + q;
                const source = codes[part][row * (k / 2) + input / 2];
                const expected: u32 = if (input % 2 == 0) source & 0xf else source >> 4;
                try std.testing.expectEqual(expected, (word >> shifts[half][q]) & 0xf);
            };
        };
        for (0..4) |t| for (0..4) |slice| for (0..2) |half| {
            const word = block[128 + t * 4 + 2 * half + (slice >> 1)];
            const shift: u5 = @intCast(16 * (slice & 1));
            for (0..2) |column| {
                const byte_shift: u5 = shift + @as(u5, @intCast(column * 8));
                const row = cb * X.cols + slice * 8 + t * 2 + column;
                const expected = scales[part][row * (k / 16) + group * 2 + half];
                try std.testing.expectEqual(@as(u32, expected), (word >> byte_shift) & 0xff);
            }
        };
    };
}

test "Qwen N8 warp units cover every output exactly once across four slices" {
    var seen: [slots][width]bool = @splat(@splat(false));
    const slices = 4;
    const nb = width / X.cols;
    const grid = N8Up.grid(10000);
    try std.testing.expectEqual(@as(usize, 512), grid * Arm.wn4_n8_pipe4.warps());
    for (0..grid) |block| for (0..Arm.wn4_n8_pipe4.warps()) |warp| {
        const unit = block * Arm.wn4_n8_pipe4.warps() + warp;
        const tile = unit / slices;
        const slice = unit % slices;
        const item = tile / nb;
        const cb = tile - item * nb;
        for (0..4) |t| for (0..2) |column| {
            const col = cb * X.cols + 2 * t + 8 * slice + column;
            try std.testing.expect(item < slots and col < width);
            try std.testing.expect(!seen[item][col]);
            seen[item][col] = true;
        };
    };
    for (seen) |row| for (row) |covered| try std.testing.expect(covered);
}

test "Qwen N8 up screen keeps the direct decode ABI and bank offsets" {
    const layer: cuda.experts.Layer = .{ .format = .nvfp4, .up = 4096, .down = 0, .up_scale = 8192, .down_scale = 0, .width = width, .dims = dims, .experts = experts };
    const rows: cuda.experts.Rows = .{ .x = 1024, .stride = dims, .slots = slots };
    const plan: cuda.grouped.Plan = .{ .items = 100, .counts = 200, .members = 300, .rank = 0, .hist = 0 };
    const last = bankLayer(layer, banks - 1);
    const a = N8Up.args(rows, last, plan, 400);
    try std.testing.expectEqual(@as(usize, 14), a.count);
    const values = [_]u64{ rows.x, dims, slots, last.up, last.up_scale, 64, 16, plan.items, plan.counts, plan.members, 400, width, 0, 0xffffffff };
    const sizes = [_]u16{ 8, 4, 4, 8, 8, 4, 4, 8, 8, 8, 8, 4, 4, 4 };
    for (values, sizes, 0..) |value, size, i| {
        try std.testing.expectEqual(@as(?u64, value), a.integer(i));
        try std.testing.expectEqual(size, a.sizes[i]);
    }
    try std.testing.expectEqual(@as(usize, 128), N8Up.grid(10000));
    try std.testing.expectEqual(@as(usize, 7), N8Up.grid(7));
    try std.testing.expectEqual(@as(usize, 128), Arm.wn4_n8_pipe4.warps() * 32);
    try std.testing.expectEqual(Arm.wn1_baseline, arms[2]);
    try std.testing.expectEqual(Arm.wn4_n8_pipe4, n8_arms[2]);
}

test "Qwen expert screen uses eight items and covers all real expert banks" {
    try std.testing.expectEqual(@as(usize, 8), bound);
    try std.testing.expectEqual(@as(usize, 32), banks);
    try std.testing.expectEqual(@as(usize, 0), launches % banks);
    const layer: cuda.experts.Layer = .{ .format = .nvfp4, .up = 0, .down = 0, .up_scale = 0, .down_scale = 0, .width = width, .dims = dims, .experts = experts };
    const last = bankLayer(layer, banks - 1);
    try std.testing.expectEqual(X.bytes(experts, width, dims, 2), last.up + X.bytes(slots, width, dims, 2));
    try std.testing.expectEqual(X.bytes(experts, dims, width, 1), last.down + X.bytes(slots, dims, width, 1));
    for (arms, [3]usize{ 32, 64, 128 }, [3]usize{ 128, 256, 512 }) |arm, up_grid, down_grid| {
        try std.testing.expectEqual(up_grid, X.decodeGridWarps(arm.variant(), bound, width / X.cols, 10000, arm.warps()));
        try std.testing.expectEqual(down_grid, X.decodeGridWarps(arm.variant(), bound, dims / X.cols, 10000, arm.warps()));
        try std.testing.expectEqual(@as(usize, 7), X.decodeGridWarps(arm.variant(), bound, width / X.cols, 7, arm.warps()));
    }
    try std.testing.expectEqual(@as(usize, 32), Arm.wn1_baseline.warps() * 32);
    try std.testing.expectEqual(@as(usize, 128), Arm.wn4_n16_pipe4.warps() * 32);
    try std.testing.expectEqual(@as(f32, 0), last.limit);
    try std.testing.expect(!last.relu2);
}

test "Qwen expert screen rejects changed fixture bytes and nonfinite output" {
    const expected = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
    _ = try verifyHash("abc", expected);
    try std.testing.expectError(error.ExpertFixtureHashMismatch, verifyHash("abd", expected));
    try finite(&.{ 0, 0, 0x80, 0x3f }, true);
    try finite(&.{ 0x80, 0x3f }, false);
    try std.testing.expectError(error.NonfiniteExpertOutput, finite(&.{ 0xff, 0xff, 0xff, 0xff }, true));
    try std.testing.expectError(error.NonfiniteExpertOutput, finite(&.{ 0xff, 0xff }, false));
    try std.testing.expectError(error.InvalidExpertOutputSize, finite(&.{0}, false));
    try std.testing.expectError(error.InvalidExpertOutputSize, finite(&.{ 0, 0 }, true));
}

test "Qwen expert screen rejects missing repeated and out of range fixture experts" {
    var order: [experts]usize = undefined;
    for (&order, 0..) |*id, i| id.* = experts - 1 - i;
    try validateOrder(&order);
    try std.testing.expectError(error.InvalidExpertOrder, validateOrder(order[0 .. experts - 1]));
    order[0] = order[1];
    try std.testing.expectError(error.InvalidExpertOrder, validateOrder(&order));
    order[0] = experts;
    try std.testing.expectError(error.InvalidExpertOrder, validateOrder(&order));
}

test "Qwen expert screen manifest requires byte counts and hashes" {
    const bytes =
        \\{"scope":"real layer zero","expert_order":[0],"files":{"input.bin":{"bytes":4096,"sha256":"abc"}},"source":"retained capture"}
    ;
    const parsed = try std.json.parseFromSlice(Manifest, std.testing.allocator, bytes, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    const pin = parsed.value.files.map.get("input.bin").?;
    try std.testing.expectEqual(@as(usize, 4096), pin.bytes);
    try std.testing.expectEqualStrings("abc", pin.sha256);
    try std.testing.expectError(error.MissingField, std.json.parseFromSlice(Manifest, std.testing.allocator,
        \\{"scope":"real layer zero","expert_order":[0],"files":{"input.bin":{"bytes":4096}}}
    , .{}));
    try std.testing.expectError(error.MissingField, std.json.parseFromSlice(Manifest, std.testing.allocator,
        \\{"scope":"real layer zero","expert_order":[0],"files":{"input.bin":{"sha256":"abc"}}}
    , .{}));
}

test "Qwen expert screen rejects a wrong captured kernel or launch geometry" {
    const f: cuda.abi.Function = @ptrFromInt(1);
    const p: cuda.abi.KernelNodeParams = .{ .func = f, .grid_x = 128, .grid_y = 1, .grid_z = 1, .block_x = 32, .block_y = 1, .block_z = 1, .shared_bytes = 0, .params = null, .extra = null };
    try checkNode(p, f, 128, 32);
    try std.testing.expectError(error.ExpertGraphKernelMismatch, checkNode(p, @ptrFromInt(2), 128, 32));
    try std.testing.expectError(error.ExpertGraphKernelMismatch, checkNode(p, f, 32, 32));
    try std.testing.expectError(error.ExpertGraphKernelMismatch, checkNode(p, f, 128, 128));
    var wrong = p;
    wrong.grid_y = 2;
    try std.testing.expectError(error.ExpertGraphKernelMismatch, checkNode(wrong, f, 128, 32));
    wrong = p;
    wrong.shared_bytes = 4;
    try std.testing.expectError(error.ExpertGraphKernelMismatch, checkNode(wrong, f, 128, 32));
}

const Shard = enum {
    full8,
    first4,
    last4,

    fn first(self: Shard) usize {
        return if (self == .last4) 4 else 0;
    }
    fn count(self: Shard) usize {
        return if (self == .full8) 8 else 4;
    }
    fn active(self: Shard, row: usize) bool {
        return row >= self.first() and row < self.first() + self.count();
    }
};
const shard_arms = [_]Shard{ .full8, .first4, .last4 };
const ShardRoute = struct {
    members: [slots]i32,
    items: [slots][3]i32,
    counts: [2]i32,

    fn init(shard: Shard) ShardRoute {
        var route: ShardRoute = .{ .members = undefined, .items = @splat(.{ -1, -1, -1 }), .counts = .{ @intCast(shard.count()), 0 } };
        for (&route.members, 0..) |*member, row| member.* = @intCast(row);
        for (0..shard.count()) |item| {
            const row: i32 = @intCast(shard.first() + item);
            route.items[item] = .{ row, row, 1 };
        }
        return route;
    }
};

pub fn validateShardInput(input: anytype) !void {
    if (!input.expert_shard_screen) return;
    if (input.expert_fixture == null or input.expert_n8_screen or input.gdn_layout_screen or input.gdn_layout_ab or
        input.expert_n8_ab or input.quant_reuse_screen or input.transport_screen or input.verify_default_policy or
        input.router_turns or input.forced.len != 0) return error.InvalidInput;
}

fn shardHash(bytes: []const u8) [64]u8 {
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &digest, .{});
    return std.fmt.bytesToHex(digest, .lower);
}

fn shardOutput(shard: Shard, actual: []const u8, expected: []const u8, row_bytes: usize, fp32: bool) !void {
    if (actual.len != slots * row_bytes or expected.len != actual.len) return error.InvalidExpertOutputSize;
    for (0..slots) |row| {
        const bytes = actual[row * row_bytes ..][0..row_bytes];
        if (shard.active(row)) {
            try finite(bytes, fp32);
            try check.sameBytes("expert shard active output", bytes, expected[row * row_bytes ..][0..row_bytes]);
        } else for (bytes) |byte| if (byte != 0xff) return error.ExpertShardInactiveWrite;
    }
}

fn shardRun(ex: *const X, shard: Shard, s: cuda.Stream, epi: cuda.experts.Epi, rows: cuda.experts.Rows, layer: cuda.experts.Layer, plan: cuda.grouped.Plan, output: u64) !void {
    if (epi == .up) return ex.runN8Up(s, rows, layer, plan, output, shard.count(), -1);
    return ex.run(s, epi, rows, layer, plan, output, shard.count(), -1);
}

fn shardNodeArgs(p: cuda.abi.KernelNodeParams, expected: *const cuda.launch.Args) ![64]u8 {
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

pub fn runShards(gpu: check.Gpu, fixture_dir: []const u8, expected_manifest_sha256: []const u8, journal: anytype) !void {
    try floor(gpu);
    const path = try std.fs.path.join(gpu.gpa, &.{ fixture_dir, "manifest.json" });
    defer gpu.gpa.free(path);
    const bytes = try std.Io.Dir.cwd().readFileAlloc(gpu.io, path, gpu.gpa, .limited(1 << 20));
    defer gpu.gpa.free(bytes);
    const digest = try verifyHash(bytes, expected_manifest_sha256);
    const manifest = try std.json.parseFromSlice(Manifest, gpu.gpa, bytes, .{ .ignore_unknown_fields = true });
    defer manifest.deinit();
    try validateOrder(manifest.value.expert_order);
    const sms: usize = @intCast(try gpu.ctx.attribute(.multiprocessor_count));
    var ex = try X.load(gpu.d, sms);
    defer ex.unload();
    try ex.loadN8Up(sms);
    if (try ex.n8_up.?.function.attribute(.local_size_bytes) != 0) return error.N8KernelSpill;
    if (ex.n8_up.?.grid(8, width / X.cols) != 128 or ex.n8_up.?.grid(4, width / X.cols) != 64) return error.N8GridMismatch;
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    try journal.write(.{ .kind = "expert_shard_fixture", .ordinal = gpu.ctx.device_ordinal, .manifest_sha256 = &digest, .scope = manifest.value.scope, .expert_order = manifest.value.expert_order, .banks = banks, .slots = slots, .sm_count = sms, .floor_bytes = floor_bytes, .free_bytes = (try gpu.ctx.memInfo()).free, .measurement_scope = "isolated per-GPU operator events and host wall; no simultaneous ranks, reduction, or model gain" });
    for ([_]cuda.experts.Epi{ .up, .down_f32 }) |epi| try shardProjection(gpu, fixture_dir, &manifest.value, journal, &ex, s, epi);
    try floor(gpu);
}

fn shardProjection(gpu: check.Gpu, dir: []const u8, manifest: *const Manifest, journal: anytype, ex: *const X, s: cuda.Stream, epi: cuda.experts.Epi) !void {
    const up = epi == .up;
    const n: usize = if (up) width else dims;
    const k: usize = if (up) dims else width;
    const parts: usize = if (up) 2 else 1;
    const row_bytes = n * (if (up) @as(usize, 2) else 4);
    const one = X.bytes(1, n, k, parts);
    const host_weights = try read(gpu, dir, manifest, if (up) "up.bin" else "down.bin", experts * one);
    defer gpu.gpa.free(host_weights);
    const host_scales = try read(gpu, dir, manifest, if (up) "up-scales.bin" else "down-scales.bin", experts * parts * 4);
    defer gpu.gpa.free(host_scales);
    try finite(host_scales, true);
    const host_input = try read(gpu, dir, manifest, if (up) "input.bin" else "hidden.bin", (if (up) @as(usize, 1) else slots) * k * 2);
    defer gpu.gpa.free(host_input);
    try finite(host_input, false);
    const saved = try read(gpu, dir, manifest, if (up) "hidden.bin" else "pairs.bin", slots * row_bytes);
    defer gpu.gpa.free(saved);
    try finite(saved, !up);
    var weights = try cuda.DeviceBuffer.fromHost(gpu.d, host_weights);
    defer weights.free();
    var scales = try cuda.DeviceBuffer.fromHost(gpu.d, host_scales);
    defer scales.free();
    var input = try cuda.DeviceBuffer.fromHost(gpu.d, host_input);
    defer input.free();
    const layer: cuda.experts.Layer = .{ .format = .nvfp4, .up = weights.ptr, .down = weights.ptr, .up_scale = scales.ptr, .down_scale = scales.ptr, .width = width, .dims = dims, .experts = experts, .limit = 0, .relu2 = false };
    const rows: cuda.experts.Rows = .{ .x = input.ptr, .stride = k, .slots = slots };
    var plan_buffers: [3][3]cuda.DeviceBuffer = undefined;
    var made_buffers: usize = 0;
    defer for (0..made_buffers) |i| plan_buffers[i / 3][i % 3].free();
    var plans: [3]cuda.grouped.Plan = undefined;
    for (shard_arms, 0..) |shard, ai| {
        const route = ShardRoute.init(shard);
        const data = [_][]const u8{ std.mem.asBytes(&route.members), std.mem.asBytes(&route.items), std.mem.asBytes(&route.counts) };
        var hashes: [3][64]u8 = undefined;
        for (data, 0..) |host, bi| {
            plan_buffers[ai][bi] = try cuda.DeviceBuffer.fromHost(gpu.d, host);
            made_buffers += 1;
            var downloaded: [slots * 12]u8 = undefined;
            try plan_buffers[ai][bi].download(0, downloaded[0..host.len]);
            try check.sameBytes("uploaded shard plan", downloaded[0..host.len], host);
            hashes[bi] = shardHash(downloaded[0..host.len]);
        }
        plans[ai] = .{ .members = plan_buffers[ai][0].ptr, .items = plan_buffers[ai][1].ptr, .counts = plan_buffers[ai][2].ptr, .rank = 0, .hist = 0 };
        try journal.write(.{ .kind = "expert_shard_plan", .ordinal = gpu.ctx.device_ordinal, .epi = @tagName(epi), .arm = @tagName(shard), .bound = shard.count(), .actual_items = route.counts[0], .members = route.members, .items = route.items[0..shard.count()], .counts = route.counts, .bindings = plans[ai], .binding_sha256 = .{ hashes[0][0..], hashes[1][0..], hashes[2][0..] }, .input_slots = slots, .first_output_row = shard.first(), .output_rows = shard.count(), .bank0_original_ids = manifest.expert_order[shard.first()..][0..shard.count()] });
    }
    var reference = try cuda.DeviceBuffer.alloc(gpu.d, saved.len);
    defer reference.free();
    var output = try cuda.DeviceBuffer.alloc(gpu.d, saved.len);
    defer output.free();
    const want = try gpu.gpa.alloc(u8, saved.len);
    defer gpu.gpa.free(want);
    const got = try gpu.gpa.alloc(u8, saved.len);
    defer gpu.gpa.free(got);
    try floor(gpu);
    for (0..banks) |bank| {
        const bl = bankLayer(layer, bank);
        try reference.fill8(0xff, s.handle);
        try shardRun(ex, .full8, s, epi, rows, bl, plans[0], reference.ptr);
        try s.synchronize();
        try reference.download(0, want);
        try finite(want, !up);
        if (bank == 0) try check.sameBytes("shard full8 saved real output", want, saved);
        for (shard_arms[1..], 1..) |shard, ai| {
            try output.fill8(0xff, s.handle);
            try shardRun(ex, shard, s, epi, rows, bl, plans[ai], output.ptr);
            try s.synchronize();
            try output.download(0, got);
            try shardOutput(shard, got, want, row_bytes, !up);
            if (bank == 0) try shardOutput(shard, got, saved, row_bytes, !up);
            try journal.write(.{ .kind = "expert_shard_bank_parity", .ordinal = gpu.ctx.device_ordinal, .epi = @tagName(epi), .bank = bank, .arm = @tagName(shard), .original_ids = manifest.expert_order[bank * slots + shard.first() ..][0..shard.count()], .active_bytes_exact = true, .inactive_bytes_poison = true, .saved_real_output = bank == 0 });
        }
    }
    var graphs: [3]cuda.graph.Exec = undefined;
    var made_graphs: usize = 0;
    defer for (graphs[0..made_graphs]) |*graph| graph.deinit();
    const f = if (up) ex.n8_up.?.function else ex.fns[1];
    const resident = if (up) ex.n8_up.?.resident else ex.resident[1];
    for (shard_arms, 0..) |shard, ai| {
        try cuda.graph.beginCapture(s, .thread_local);
        for (0..launches) |i| shardRun(ex, shard, s, epi, rows, bankLayer(layer, i % banks), plans[ai], output.ptr) catch |err| {
            if (cuda.graph.endCapture(s)) |graph| {
                var dead = graph;
                dead.deinit();
            } else |_| {}
            return err;
        };
        var graph = try cuda.graph.endCapture(s);
        defer graph.deinit();
        if (try graph.nodeCount() != launches) return error.ExpertGraphWorkCountMismatch;
        const grid = if (up) ex.n8_up.?.grid(shard.count(), n / X.cols) else X.decodeGridWarps(.baseline, shard.count(), n / X.cols, resident, ex.warps);
        if (grid != (if (shard == .full8) @as(usize, 128) else 64)) return error.ExpertShardGridMismatch;
        var nodes: [launches]cuda.graph.Node = undefined;
        const captured = try graph.nodes(&nodes);
        if (captured.len != launches) return error.ExpertGraphWorkCountMismatch;
        var bank_counts: [banks]usize = @splat(0);
        var binding_hashes: [banks][64]u8 = undefined;
        for (captured) |node| {
            var p: cuda.abi.KernelNodeParams = undefined;
            try gpu.d.check(gpu.d.api.cuGraphKernelNodeGetParams_v2(node, &p), "cuGraphKernelNodeGetParams");
            try checkNode(p, f.handle, grid, 128);
            const args = p.params orelse return error.ExpertGraphMissingArguments;
            const weight_ptr: *const u64 = @ptrCast(@alignCast(args[3] orelse return error.ExpertGraphMissingArguments));
            if (weight_ptr.* < weights.ptr or (weight_ptr.* - weights.ptr) % (slots * one) != 0) return error.ExpertGraphArgumentMismatch;
            const bank = (weight_ptr.* - weights.ptr) / (slots * one);
            if (bank >= banks) return error.ExpertGraphArgumentMismatch;
            const expected = X.decodeArgs(epi, rows, bankLayer(layer, bank), plans[ai], output.ptr, -1);
            binding_hashes[bank] = try shardNodeArgs(p, &expected);
            bank_counts[bank] += 1;
        }
        for (bank_counts) |count| if (count != launches / banks) return error.ExpertGraphWorkCountMismatch;
        graphs[ai] = try graph.instantiate();
        made_graphs += 1;
        try graphs[ai].upload(s);
        try s.synchronize();
        var binding_hex: [banks][]const u8 = undefined;
        for (&binding_hashes, &binding_hex) |*digest, *hex| hex.* = digest;
        try journal.write(.{ .kind = "expert_shard_kernel", .ordinal = gpu.ctx.device_ordinal, .epi = @tagName(epi), .arm = @tagName(shard), .symbol = try kernelName(f), .grid = grid, .threads = 128, .regs = try f.attribute(.num_regs), .local_bytes = try f.attribute(.local_size_bytes), .resident_blocks = resident, .resident_blocks_per_sm = try f.occupancy(128, 0), .kernel_launches = launches, .verified_kernel_nodes = captured.len, .bank_node_counts = bank_counts, .argument_sha256_by_bank = binding_hex, .logical_weight_and_scale_bytes_per_launch = shard.count() * (one + parts * 4), .logical_unique_weight_and_scale_bytes = banks * shard.count() * (one + parts * 4), .allocated_weight_and_scale_bytes = experts * (one + parts * 4), .free_bytes = (try gpu.ctx.memInfo()).free });
    }
    var start = try cuda.Event.init(gpu.d, true);
    defer start.deinit();
    var end = try cuda.Event.init(gpu.d, true);
    defer end.deinit();
    for (shard_arms[1..], 1..) |comparison, candidate_index| {
        var events: [2][5]f64 = undefined;
        var walls: [2][5]f64 = undefined;
        for (0..6) |pair| for (0..2) |order| {
            const arm_index = (pair + order) % 2;
            const ai = if (arm_index == 1) candidate_index else 0;
            const shard = shard_arms[ai];
            try output.fill8(0xff, s.handle);
            try s.synchronize();
            const t0 = std.Io.Clock.awake.now(gpu.io).toNanoseconds();
            try start.record(s);
            try graphs[ai].launchOn(s);
            try end.record(s);
            try end.synchronize();
            const wall_ms = @as(f64, @floatFromInt(std.Io.Clock.awake.now(gpu.io).toNanoseconds() - t0)) / 1e6;
            const elapsed_ms = try cuda.Event.elapsedMs(start, end);
            if (!std.math.isFinite(elapsed_ms) or elapsed_ms <= 0 or !std.math.isFinite(wall_ms) or wall_ms <= 0) return error.InvalidExpertShardTiming;
            try output.download(0, got);
            try shardOutput(shard, got, want, row_bytes, !up);
            try floor(gpu);
            if (pair != 0) {
                events[arm_index][pair - 1] = elapsed_ms;
                walls[arm_index][pair - 1] = wall_ms;
            }
            try journal.write(.{ .kind = "expert_shard_sample", .ordinal = gpu.ctx.device_ordinal, .epi = @tagName(epi), .comparison = @tagName(comparison), .pair = pair, .warmup = pair == 0, .order = order, .arm = @tagName(shard), .elapsed_ms = elapsed_ms, .wall_ms = wall_ms, .kernel_launches = launches, .banks = banks, .final_bank = banks - 1, .active_bytes_exact = true, .inactive_bytes_poison = true, .free_bytes = (try gpu.ctx.memInfo()).free });
        };
        for (0..2) |arm_index| {
            std.mem.sort(f64, &events[arm_index], {}, std.sort.asc(f64));
            std.mem.sort(f64, &walls[arm_index], {}, std.sort.asc(f64));
            try journal.write(.{ .kind = "expert_shard_summary", .ordinal = gpu.ctx.device_ordinal, .epi = @tagName(epi), .comparison = @tagName(comparison), .arm = @tagName(shard_arms[if (arm_index == 0) @as(usize, 0) else candidate_index]), .samples = 5, .kernel_launches_per_sample = launches, .event_ms_median = events[arm_index][2], .event_ms_min = events[arm_index][0], .event_ms_max = events[arm_index][4], .wall_ms_median = walls[arm_index][2], .wall_ms_min = walls[arm_index][0], .wall_ms_max = walls[arm_index][4] });
        }
    }
    try journal.write(.{ .kind = "expert_shard_projection_complete", .ordinal = gpu.ctx.device_ordinal, .epi = @tagName(epi), .all_bank_prechecks = true, .saved_bank0_exact = true, .graph_finals_exact = true, .timed_pairs_per_comparison = 5 });
}

test "expert shard plans retain global selected rows without overlap or missing work" {
    var covered: [slots]bool = @splat(false);
    for (shard_arms[1..]) |shard| {
        const route = ShardRoute.init(shard);
        try std.testing.expectEqual(@as(i32, 4), route.counts[0]);
        for (route.items[0..4]) |item| {
            const row: usize = @intCast(route.members[@intCast(item[1])]);
            try std.testing.expectEqual(@as(i32, @intCast(row)), item[0]);
            try std.testing.expectEqual(@as(i32, 1), item[2]);
            try std.testing.expect(shard.active(row));
            try std.testing.expect(!covered[row]);
            covered[row] = true;
        }
    }
    for (covered) |hit| try std.testing.expect(hit);
    const full = ShardRoute.init(.full8);
    try std.testing.expectEqual(@as(i32, 8), full.counts[0]);
    try std.testing.expectEqual([3]i32{ 7, 7, 1 }, full.items[7]);
}

test "expert shard output rejects wrong active bytes and any inactive write" {
    const values: [slots]u16 = @splat(0x3f80);
    const reference = std.mem.asBytes(&values);
    var actual: [slots * 2]u8 = @splat(0xff);
    @memcpy(actual[8..], reference[8..]);
    try shardOutput(.last4, &actual, reference, 2, false);
    actual[0] = 0;
    try std.testing.expectError(error.ExpertShardInactiveWrite, shardOutput(.last4, &actual, reference, 2, false));
    actual[0] = 0xff;
    actual[8] = 0xff;
    actual[9] = 0xff;
    try std.testing.expectError(error.NonfiniteExpertOutput, shardOutput(.last4, &actual, reference, 2, false));
}

test "expert shard capture verifier rejects every changed argument binding" {
    const layer: cuda.experts.Layer = .{ .format = .nvfp4, .up = 4096, .down = 8192, .up_scale = 128, .down_scale = 256, .width = width, .dims = dims, .experts = slots };
    const plan: cuda.grouped.Plan = .{ .items = 300, .counts = 400, .members = 500, .rank = 0, .hist = 0 };
    for ([_]cuda.experts.Epi{ .up, .down_f32 }) |epi| {
        const expected = X.decodeArgs(epi, .{ .x = 1024, .stride = if (epi == .up) dims else width, .slots = slots }, layer, plan, 600, -1);
        var actual = expected;
        const p: cuda.abi.KernelNodeParams = .{ .func = @ptrFromInt(1), .grid_x = 64, .grid_y = 1, .grid_z = 1, .block_x = 128, .block_y = 1, .block_z = 1, .shared_bytes = 0, .params = actual.pointers(), .extra = null };
        _ = try shardNodeArgs(p, &expected);
        for (0..expected.count) |i| {
            actual.storage[actual.offsets[i]] ^= 1;
            try std.testing.expectError(error.ExpertGraphArgumentMismatch, shardNodeArgs(p, &expected));
            actual.storage[actual.offsets[i]] ^= 1;
        }
    }
}
