const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const X = cuda.experts.Nvfp4Experts;
const variants = std.meta.tags(X.Variant);
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
    try floor(gpu);
    const manifest_path = try std.fs.path.join(gpu.gpa, &.{ fixture_dir, "manifest.json" });
    defer gpu.gpa.free(manifest_path);
    const manifest_bytes = try std.Io.Dir.cwd().readFileAlloc(gpu.io, manifest_path, gpu.gpa, .limited(1 << 20));
    defer gpu.gpa.free(manifest_bytes);
    const manifest_sha256 = try verifyHash(manifest_bytes, expected_manifest_sha256);
    const manifest = try std.json.parseFromSlice(Manifest, gpu.gpa, manifest_bytes, .{ .ignore_unknown_fields = true });
    defer manifest.deinit();
    if (manifest.value.expert_order.len != experts) return error.InvalidExpertOrder;
    var seen: [experts]bool = @splat(false);
    for (manifest.value.expert_order) |id| {
        if (id >= experts or seen[id]) return error.InvalidExpertOrder;
        seen[id] = true;
    }
    try journal.write(.{ .kind = "expert_fixture", .ordinal = gpu.ctx.device_ordinal, .manifest_sha256 = &manifest_sha256, .scope = manifest.value.scope, .expert_order = manifest.value.expert_order, .bank0_original_ids = manifest.value.expert_order[0..slots], .bank0_remapped_ids = [8]usize{ 0, 1, 2, 3, 4, 5, 6, 7 } });
    const sms: usize = @intCast(try gpu.ctx.attribute(.multiprocessor_count));
    var ex = try X.load(gpu.d, sms);
    defer ex.unload();
    try ex.loadCandidates(sms);
    var mod = try cuda.Module.load(gpu.d, cuda.kernels.experts);
    defer mod.unload();
    const router = try cuda.grouped.Router.resolve(mod);
    var stream = try cuda.Stream.init(gpu.d, false);
    defer stream.deinit();
    for ([_]cuda.experts.Epi{ .up, .down_f32 }) |epi| try projection(gpu, fixture_dir, &manifest.value, journal, &ex, router, stream, epi);
    try floor(gpu);
}

fn projection(gpu: check.Gpu, dir: []const u8, manifest: *const Manifest, journal: anytype, ex: *const X, router: cuda.grouped.Router, s: cuda.Stream, epi: cuda.experts.Epi) !void {
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
    try journal.write(.{ .kind = "expert_plan", .ordinal = gpu.ctx.device_ordinal, .epi = @tagName(epi), .actual_items = actual[0], .bound = bound, .picks = picks, .banks = banks, .limit = layer.limit, .relu2 = layer.relu2 });
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
        try ex.runVariant(.baseline, s, epi, rows, bl, plan, reference.ptr, bound, -1);
        try s.synchronize();
        try reference.download(0, want);
        try finite(want, !up);
        if (bank == 0) try check.sameBytes("bank0 saved real expert output", want, saved);
        for (variants[1..]) |variant| {
            try output.fill8(0xff, s.handle);
            try ex.runVariant(variant, s, epi, rows, bl, plan, output.ptr, bound, -1);
            try s.synchronize();
            try output.download(0, got);
            try finite(got, !up);
            try check.sameBytes(@tagName(variant), got, want);
            try journal.write(.{ .kind = "expert_bank_parity", .ordinal = gpu.ctx.device_ordinal, .epi = @tagName(epi), .bank = bank, .variant = @tagName(variant), .finite = true, .bit_exact = true, .saved_real_output = bank == 0 });
        }
    }
    var graphs: [variants.len]cuda.graph.Exec = undefined;
    var made: usize = 0;
    defer for (graphs[0..made]) |*graph| graph.deinit();
    for (variants, 0..) |variant, vi| {
        try cuda.graph.beginCapture(s, .thread_local);
        for (0..launches) |i| ex.runVariant(variant, s, epi, rows, bankLayer(layer, i % banks), plan, output.ptr, bound, -1) catch |err| {
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
        const resident = if (vi == 0) ex.resident[ei] else ex.candidate_resident[vi - 1][ei];
        const f = if (vi == 0) ex.fns[ei] else ex.candidate_fns[vi - 1][ei];
        try journal.write(.{ .kind = "expert_kernel", .ordinal = gpu.ctx.device_ordinal, .epi = @tagName(epi), .variant = @tagName(variant), .grid = X.decodeGrid(variant, bound, n / X.cols, resident), .block = 128, .regs = try f.attribute(.num_regs), .local_bytes = try f.attribute(.local_size_bytes), .resident_blocks = resident, .kernel_launches = launches, .banks = banks, .logical_weight_and_scale_bytes_per_launch = slots * (one + parts * 4), .logical_unique_weight_and_scale_bytes = experts * (one + parts * 4) });
    }
    var start = try cuda.Event.init(gpu.d, true);
    defer start.deinit();
    var end = try cuda.Event.init(gpu.d, true);
    defer end.deinit();
    for (variants[1..], 1..) |variant, candidate_index| {
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
            try journal.write(.{ .kind = "expert_sample", .ordinal = gpu.ctx.device_ordinal, .epi = @tagName(epi), .comparison = @tagName(variant), .pair = pair, .warmup = pair == 0, .order = order, .arm = @tagName(variants[vi]), .elapsed_ms = elapsed_ms, .kernel_launches = launches, .banks = banks, .final_bank = (launches - 1) % banks, .finite = true, .bit_exact_final = true });
        };
    }
    try journal.write(.{ .kind = "expert_projection_complete", .ordinal = gpu.ctx.device_ordinal, .epi = @tagName(epi), .all_bank_prechecks = true, .saved_bank0_exact = true, .graph_finals_exact = true, .timed_pairs_per_candidate = 5 });
}

test "Qwen expert screen uses eight items and covers all real expert banks" {
    try std.testing.expectEqual(@as(usize, 8), bound);
    try std.testing.expectEqual(@as(usize, 32), banks);
    try std.testing.expectEqual(@as(usize, 0), launches % banks);
    const layer: cuda.experts.Layer = .{ .format = .nvfp4, .up = 0, .down = 0, .up_scale = 0, .down_scale = 0, .width = width, .dims = dims, .experts = experts };
    const last = bankLayer(layer, banks - 1);
    try std.testing.expectEqual(X.bytes(experts, width, dims, 2), last.up + X.bytes(slots, width, dims, 2));
    try std.testing.expectEqual(X.bytes(experts, dims, width, 1), last.down + X.bytes(slots, dims, width, 1));
    try std.testing.expectEqual(@as(usize, 32), X.decodeGrid(.baseline, bound, width / X.cols, 10000));
    try std.testing.expectEqual(@as(usize, 64), X.decodeGrid(.n16, bound, width / X.cols, 10000));
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
}
