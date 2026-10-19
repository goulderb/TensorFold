const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const Pick = cuda.tree_round.Pick;
const rows = 128;
const vocab = 248320;
const row_bytes = vocab * 2;
const guard = 64;
const Manifest = struct { version: usize, rows: usize, vocab: usize, bytes: usize, path: []const u8, sha256: []const u8, row_sha256: []const []const u8, tokens: []const u32, tree_image_sha256: []const u8, argmax_sha256: []const u8 };
const Arm = enum { host, device };

fn hash(bytes: []const u8) [64]u8 {
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &digest, .{});
    return std.fmt.bytesToHex(digest, .lower);
}
fn verify(bytes: []const u8, expected: []const u8) !void {
    if (!std.mem.eql(u8, &hash(bytes), expected)) return error.TerminalGreedyHashMismatch;
}
pub fn validatePaths(dir: []const u8, sha: []const u8) !void {
    if (!std.fs.path.isAbsolute(dir) or sha.len != 64) return error.InvalidTerminalGreedyInput;
    for (sha) |c| if (!std.ascii.isHex(c)) return error.InvalidTerminalGreedyInput;
}
fn validateManifest(m: Manifest) !void {
    if (m.version != 1 or m.rows != rows or m.vocab != vocab or m.bytes != rows * row_bytes or !std.fs.path.isAbsolute(m.path) or m.tokens.len != rows or m.row_sha256.len != rows) return error.InvalidTerminalGreedyFixture;
    for (m.tokens) |token| if (token >= vocab) return error.InvalidTerminalGreedyFixture;
    for ([_][]const u8{ m.sha256, m.tree_image_sha256, m.argmax_sha256 }) |sha| {
        if (sha.len != 64) return error.InvalidTerminalGreedyFixture;
        for (sha) |c| if (!std.ascii.isHex(c)) return error.InvalidTerminalGreedyFixture;
    }
    for (m.row_sha256) |sha| {
        if (sha.len != 64) return error.InvalidTerminalGreedyFixture;
        for (sha) |c| if (!std.ascii.isHex(c)) return error.InvalidTerminalGreedyFixture;
    }
}
fn argmax(row: []const u16) !u32 {
    var best: usize = 0;
    var best_value: f32 = -std.math.inf(f32);
    for (row, 0..) |bits, i| {
        const value: f32 = @bitCast(@as(u32, bits) << 16);
        if (!std.math.isFinite(value)) return error.NonfiniteLogits;
        if (value > best_value) {
            best = i;
            best_value = value;
        }
    }
    return @intCast(best);
}
fn validatePick(pick: Pick, n: usize) !u32 {
    if (pick.nonfinite & ~@as(u32, 7) != 0) return error.InvalidTerminalPick;
    if (pick.nonfinite != 0) {
        if (pick.token != std.math.maxInt(u32)) return error.InvalidTerminalPick;
        return error.NonfiniteLogits;
    }
    if (pick.token >= n) return error.InvalidTerminalPick;
    return pick.token;
}
fn canaries(bytes: []const u8, active: usize) !void {
    if (bytes.len != active + guard * 2) return error.InvalidTerminalGuard;
    for (bytes[0..guard]) |value| if (value != 0xa5) return error.TerminalGuardChanged;
    for (bytes[guard + active ..]) |value| if (value != 0xa5) return error.TerminalGuardChanged;
}
fn save(gpu: check.Gpu, journal: anytype, name: []const u8, bytes: []const u8) !void {
    const filename = try std.fmt.allocPrint(gpu.gpa, "terminal_greedy_gpu{d}_{s}.bin", .{ gpu.ctx.device_ordinal, name });
    defer gpu.gpa.free(filename);
    const path = try std.fs.path.join(gpu.gpa, &.{ journal.directory, filename });
    defer gpu.gpa.free(path);
    const file = try std.Io.Dir.cwd().createFile(gpu.io, path, .{ .exclusive = true });
    defer file.close(gpu.io);
    try file.writeStreamingAll(gpu.io, bytes);
    try journal.write(.{ .kind = "terminal_greedy_artifact", .ordinal = gpu.ctx.device_ordinal, .name = name, .file = filename, .bytes = bytes.len, .sha256 = &hash(bytes) });
}
fn freeBytes(gpu: check.Gpu) !usize {
    const available = (try gpu.ctx.memInfo()).free;
    if (available < 1536 << 20) return error.GpuMemoryFloor;
    return available;
}
fn kernelName(f: cuda.Function) ![]const u8 {
    var lib = f.d.lib;
    const get_name = lib.lookup(*const fn (*?[*:0]const u8, cuda.abi.Function) callconv(.c) cuda.abi.Result, "cuFuncGetName") orelse return error.MissingSymbol;
    var name: ?[*:0]const u8 = null;
    try f.d.check(get_name(&name, f.handle), "cuFuncGetName");
    return std.mem.span(name orelse return error.MissingSymbol);
}
const Buffers = struct {
    logits: cuda.DeviceBuffer,
    pick: cuda.DeviceBuffer,
    host_logits: cuda.HostBuffer,
    host_pick: cuda.HostBuffer,
    snapshots: []u16,
    picks: [rows]Pick = undefined,
    dtoh_count: usize = 0,
    dtoh_bytes: usize = 0,
    kernel_launches: usize = 0,
    row_sync_count: usize = 0,
    fn init(gpu: check.Gpu, source: []const u8) !Buffers {
        var logits = try cuda.DeviceBuffer.alloc(gpu.d, source.len + guard * 2);
        errdefer logits.free();
        try logits.fill8(0xa5, null);
        try logits.upload(guard, source);
        var pick = try cuda.DeviceBuffer.alloc(gpu.d, @sizeOf(Pick) + guard * 2);
        errdefer pick.free();
        try pick.fill8(0xa5, null);
        var host_logits = try cuda.HostBuffer.alloc(gpu.d, row_bytes + guard * 2);
        errdefer host_logits.free();
        var host_pick = try cuda.HostBuffer.alloc(gpu.d, @sizeOf(Pick) + guard * 2);
        errdefer host_pick.free();
        const snapshots = try gpu.gpa.alloc(u16, rows * vocab);
        return .{ .logits = logits, .pick = pick, .host_logits = host_logits, .host_pick = host_pick, .snapshots = snapshots };
    }
    fn deinit(b: *Buffers, gpu: check.Gpu) void {
        gpu.gpa.free(b.snapshots);
        b.host_pick.free();
        b.host_logits.free();
        b.pick.free();
        b.logits.free();
    }
    fn reset(b: *Buffers) !void {
        b.dtoh_count = 0;
        b.dtoh_bytes = 0;
        b.kernel_launches = 0;
        b.row_sync_count = 0;
        @memset(b.host_logits.bytes, 0xa5);
        @memset(b.host_logits.bytes[guard..][0..row_bytes], 0xff);
        @memset(b.host_pick.bytes, 0xa5);
        @memset(b.host_pick.bytes[guard..][0..@sizeOf(Pick)], 0xff);
        @memset(b.snapshots, 0xffff);
        for (&b.picks) |*p| p.* = .{ .token = std.math.maxInt(u32), .nonfinite = std.math.maxInt(u32) };
        const payload: cuda.DeviceBuffer = .{ .d = b.pick.d, .ptr = b.pick.ptr + guard, .len = @sizeOf(Pick), .borrowed = true };
        try payload.fill8(0xff, null);
    }
    fn row(b: *Buffers, gpu: check.Gpu, kernels: *const cuda.tree_round.Kernels, stream: cuda.Stream, arm: Arm, index: usize) !void {
        const input = b.logits.ptr + guard + index * row_bytes;
        switch (arm) {
            .host => {
                try gpu.d.check(gpu.d.api.cuMemcpyDtoHAsync_v2(b.host_logits.bytes[guard..].ptr, input, row_bytes, stream.handle), "cuMemcpyDtoHAsync");
                b.dtoh_count += 1;
                b.dtoh_bytes += row_bytes;
                try stream.synchronize();
                b.row_sync_count += 1;
                const dst = b.snapshots[index * vocab ..][0..vocab];
                @memcpy(dst, b.host_logits.slice(u16)[guard / 2 ..][0..vocab]);
                b.picks[index] = .{ .token = try argmax(dst), .nonfinite = 0 };
            },
            .device => {
                try kernels.argmax(stream, input, b.pick.ptr + guard, 1, vocab, vocab);
                b.kernel_launches += 1;
                try gpu.d.check(gpu.d.api.cuMemcpyDtoHAsync_v2(b.host_pick.bytes[guard..].ptr, b.pick.ptr + guard, @sizeOf(Pick), stream.handle), "cuMemcpyDtoHAsync");
                b.dtoh_count += 1;
                b.dtoh_bytes += @sizeOf(Pick);
                try stream.synchronize();
                b.row_sync_count += 1;
                const pick = b.host_pick.slice(Pick)[guard / @sizeOf(Pick)];
                b.picks[index] = pick;
                _ = try validatePick(pick, vocab);
            },
        }
    }
};
fn sweep(gpu: check.Gpu, b: *Buffers, kernels: *const cuda.tree_round.Kernels, stream: cuda.Stream, arm: Arm, context: []const u8, journal: anytype) !void {
    for (0..rows) |index| b.row(gpu, kernels, stream, arm, index) catch |err| {
        cuda.completion.drain(gpu.d, stream, "terminal greedy row failed to drain");
        const raw = try check.download(gpu, b.pick);
        defer gpu.gpa.free(raw);
        const blobs = [_][]const u8{ b.host_logits.bytes, b.host_pick.bytes, std.mem.asBytes(&b.picks), std.mem.sliceAsBytes(b.snapshots), raw };
        for (blobs, 0..) |bytes, part| {
            const name = try std.fmt.allocPrint(gpu.gpa, "failure_{s}_row{d}_part{d}", .{ context, index, part });
            defer gpu.gpa.free(name);
            try save(gpu, journal, name, bytes);
        }
        try journal.write(.{ .kind = "terminal_greedy_failure", .ordinal = gpu.ctx.device_ordinal, .context = context, .row = index, .arm = @tagName(arm), .error_name = @errorName(err), .submitted_dtoh_count = b.dtoh_count, .submitted_dtoh_bytes = b.dtoh_bytes, .submitted_kernels = b.kernel_launches, .completed_row_syncs = b.row_sync_count });
        return err;
    };
}
fn evidence(gpu: check.Gpu, b: *Buffers, source: []const u8, expected: []const u32, arm: Arm, name: []const u8, journal: anytype) !void {
    try save(gpu, journal, name, std.mem.asBytes(&b.picks));
    for ([_][]const u8{ b.host_logits.bytes, b.host_pick.bytes }, [_][]const u8{ "host_logits_guard", "host_pick_guard" }) |bytes, suffix| {
        const host_name = try std.fmt.allocPrint(gpu.gpa, "{s}_{s}", .{ name, suffix });
        defer gpu.gpa.free(host_name);
        try save(gpu, journal, host_name, bytes);
    }
    const raw_pick = try check.download(gpu, b.pick);
    defer gpu.gpa.free(raw_pick);
    const guard_name = try std.fmt.allocPrint(gpu.gpa, "{s}_device_guard", .{name});
    defer gpu.gpa.free(guard_name);
    try save(gpu, journal, guard_name, raw_pick);
    if (arm == .host) {
        const snapshot_name = try std.fmt.allocPrint(gpu.gpa, "{s}_copied_rows", .{name});
        defer gpu.gpa.free(snapshot_name);
        const snapshot = std.mem.sliceAsBytes(b.snapshots);
        try save(gpu, journal, snapshot_name, snapshot);
        try check.sameBytes("terminal diagnostic copied rows", snapshot, source);
    }
    for (b.picks, expected) |pick, token| {
        if (try validatePick(pick, vocab) != token) return error.TerminalTokenMismatch;
    }
    try canaries(b.host_logits.bytes, row_bytes);
    try canaries(b.host_pick.bytes, @sizeOf(Pick));
    try canaries(raw_pick, @sizeOf(Pick));
    if (b.dtoh_count != rows or b.row_sync_count != rows or b.dtoh_bytes != rows * @as(usize, if (arm == .host) row_bytes else @sizeOf(Pick)) or b.kernel_launches != @as(usize, if (arm == .device) rows else 0)) return error.TerminalTransferCounts;
}
const semantic_vocab = 513;
const semantic_cases = 8;
fn semanticRow(index: usize, row: []u16) Pick {
    @memset(row, 0xbf80);
    switch (index) {
        0 => {
            row[3] = 0x4000;
            row[257] = 0x4000;
            return .{ .token = 3, .nonfinite = 0 };
        },
        1 => {
            row[0] = 0x8000;
            row[257] = 0;
            return .{ .token = 0, .nonfinite = 0 };
        },
        2 => {
            row[0] = 0;
            row[257] = 0x8000;
            return .{ .token = 0, .nonfinite = 0 };
        },
        3 => {
            row[511] = 0xbf00;
            return .{ .token = 511, .nonfinite = 0 };
        },
        4 => {
            row[3] = 0x7fc1;
            return .{ .token = 0xffffffff, .nonfinite = 1 };
        },
        5 => {
            row[3] = 0x7f80;
            return .{ .token = 0xffffffff, .nonfinite = 2 };
        },
        6 => {
            row[3] = 0xff80;
            return .{ .token = 0xffffffff, .nonfinite = 4 };
        },
        7 => {
            row[1] = 0x7fc0;
            row[257] = 0x7f80;
            row[512] = 0xff80;
            return .{ .token = 0xffffffff, .nonfinite = 7 };
        },
        else => unreachable,
    }
}
fn semantics(gpu: check.Gpu, kernels: *const cuda.tree_round.Kernels, stream: cuda.Stream, b: *Buffers, journal: anytype) !void {
    for (0..semantic_cases) |index| {
        var input: [semantic_vocab]u16 = undefined;
        const expected = semanticRow(index, &input);
        var device = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.asBytes(&input));
        defer device.free();
        errdefer cuda.completion.drain(gpu.d, stream, "terminal greedy semantic row failed to drain");
        try b.reset();
        try kernels.argmax(stream, device.ptr, b.pick.ptr + guard, 1, semantic_vocab, semantic_vocab);
        try gpu.d.check(gpu.d.api.cuMemcpyDtoHAsync_v2(b.host_pick.bytes[guard..].ptr, b.pick.ptr + guard, @sizeOf(Pick), stream.handle), "cuMemcpyDtoHAsync");
        try stream.synchronize();
        const actual = b.host_pick.slice(Pick)[guard / @sizeOf(Pick)];
        const raw = try check.download(gpu, b.pick);
        defer gpu.gpa.free(raw);
        const name = try std.fmt.allocPrint(gpu.gpa, "semantic{d}_pick", .{index});
        defer gpu.gpa.free(name);
        try save(gpu, journal, name, raw);
        const input_name = try std.fmt.allocPrint(gpu.gpa, "semantic{d}_input", .{index});
        defer gpu.gpa.free(input_name);
        const consumed = try check.download(gpu, device);
        defer gpu.gpa.free(consumed);
        try save(gpu, journal, input_name, consumed);
        try check.sameBytes("terminal semantic input", consumed, std.mem.asBytes(&input));
        try canaries(raw, @sizeOf(Pick));
        try canaries(b.host_pick.bytes, @sizeOf(Pick));
        try check.sameBytes("terminal semantic Pick", std.mem.asBytes(&actual), std.mem.asBytes(&expected));
        if (expected.nonfinite == 0) {
            if (try argmax(&input) != actual.token or try validatePick(actual, semantic_vocab) != actual.token) return error.TerminalSemanticMismatch;
        } else {
            if (argmax(&input)) |_| return error.TerminalSemanticMismatch else |err| if (err != error.NonfiniteLogits) return err;
            if (validatePick(actual, semantic_vocab)) |_| return error.TerminalSemanticMismatch else |err| if (err != error.NonfiniteLogits) return err;
        }
        try journal.write(.{ .kind = "terminal_greedy_semantic", .ordinal = gpu.ctx.device_ordinal, .index = index, .vocab = semantic_vocab, .expected = expected, .actual = actual, .input_ptr = device.ptr, .pick_ptr = b.pick.ptr + guard, .direct_request_only = true });
    }
}
pub fn run(gpu: check.Gpu, fixture_dir: []const u8, manifest_sha: []const u8, journal: anytype) !void {
    try validatePaths(fixture_dir, manifest_sha);
    _ = try freeBytes(gpu);
    const path = try std.fs.path.join(gpu.gpa, &.{ fixture_dir, "manifest.json" });
    defer gpu.gpa.free(path);
    const manifest_raw = try std.Io.Dir.cwd().readFileAlloc(gpu.io, path, gpu.gpa, .limited(1 << 20));
    defer gpu.gpa.free(manifest_raw);
    try verify(manifest_raw, manifest_sha);
    const parsed = try std.json.parseFromSlice(Manifest, gpu.gpa, manifest_raw, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    const manifest = parsed.value;
    try validateManifest(manifest);
    const source = try std.Io.Dir.cwd().readFileAlloc(gpu.io, manifest.path, gpu.gpa, .limited(rows * row_bytes + 1));
    defer gpu.gpa.free(source);
    if (source.len != rows * row_bytes) return error.InvalidTerminalGreedyFixture;
    try verify(source, manifest.sha256);
    const host_rows = try gpu.gpa.alloc(u16, rows * vocab);
    defer gpu.gpa.free(host_rows);
    @memcpy(std.mem.sliceAsBytes(host_rows), source);
    for (0..rows) |index| {
        const row = host_rows[index * vocab ..][0..vocab];
        try verify(std.mem.sliceAsBytes(row), manifest.row_sha256[index]);
        if (try argmax(row) != manifest.tokens[index]) return error.TerminalTokenMismatch;
        for (manifest.row_sha256[0..index]) |prior| if (std.mem.eql(u8, prior, manifest.row_sha256[index])) return error.TerminalRowsNotDistinct;
    }
    try verify(cuda.kernels.tree_round, manifest.tree_image_sha256);
    var module = try cuda.Module.load(gpu.d, cuda.kernels.tree_round);
    defer module.unload();
    const kernels = try cuda.tree_round.Kernels.resolve(module);
    var buffers = try Buffers.init(gpu, source);
    defer buffers.deinit(gpu);
    var stream = try cuda.Stream.init(gpu.d, false);
    defer stream.deinit();
    defer cuda.completion.drain(gpu.d, stream, "terminal greedy screen failed to drain");
    const uploaded = try check.download(gpu, buffers.logits);
    defer gpu.gpa.free(uploaded);
    try save(gpu, journal, "consumed_logits", uploaded);
    try canaries(uploaded, source.len);
    try check.sameBytes("terminal uploaded logits", uploaded[guard..][0..source.len], source);
    try journal.write(.{ .kind = "terminal_greedy_scope", .ordinal = gpu.ctx.device_ordinal, .manifest_sha256 = manifest_sha, .fixture_sha256 = manifest.sha256, .tree_image_sha256 = manifest.tree_image_sha256, .argmax_sha256 = manifest.argmax_sha256, .rows = rows, .vocab = vocab, .input_ptr = buffers.logits.ptr + guard, .pick_ptr = buffers.pick.ptr + guard, .input_guard_bytes = guard, .pick_bytes = @sizeOf(Pick), .host_row_bytes = row_bytes, .included = "full-row pinned D2H+row sync+diagnostic memcpy/argmax versus shared GPU argmax+8byte pinned D2H+row sync+CPU status/token validation", .excluded = "Engine cache.logits copy, Session.logits copy, model/head compute and fullmodel saving", .launch_mode = "direct sequential rows; no capture or batched128-row launch", .event_scope = "whole stream span including host gaps; not isolated kernel time", .free_bytes = try freeBytes(gpu) });
    try journal.write(.{ .kind = "terminal_greedy_resources", .ordinal = gpu.ctx.device_ordinal, .symbol = try kernelName(kernels.argmax_fn), .function_handle = @intFromPtr(kernels.argmax_fn.handle), .regs = try kernels.argmax_fn.attribute(.num_regs), .local_bytes = try kernels.argmax_fn.attribute(.local_size_bytes), .static_shared_bytes = try kernels.argmax_fn.attribute(.shared_size_bytes), .dynamic_shared_bytes = 0, .grid = [_]u32{ 1, 1, 1 }, .block = [_]u32{ 256, 1, 1 }, .evidence = "actual resolved function/resources; launch arguments below are request-only, not graph readback" });
    for (0..rows) |index| try journal.write(.{ .kind = "terminal_greedy_request", .ordinal = gpu.ctx.device_ordinal, .row = index, .symbol = "tf_round_bf16_argmax", .argument_values = [_]u64{ buffers.logits.ptr + guard + index * row_bytes, buffers.pick.ptr + guard, vocab, vocab }, .argument_sizes = [_]u32{ 8, 8, 4, 4 }, .grid = [_]u32{ 1, 1, 1 }, .block = [_]u32{ 256, 1, 1 }, .evidence = "shared Kernels.argmax direct request; no actual graph ABI attestation" });
    try semantics(gpu, &kernels, stream, &buffers, journal);
    for ([_]Arm{ .host, .device }) |arm| {
        const name = try std.fmt.allocPrint(gpu.gpa, "before_{s}", .{@tagName(arm)});
        defer gpu.gpa.free(name);
        try buffers.reset();
        try sweep(gpu, &buffers, &kernels, stream, arm, name, journal);
        try evidence(gpu, &buffers, source, manifest.tokens, arm, name, journal);
    }
    try journal.write(.{ .kind = "terminal_greedy_qualified_before_timing", .ordinal = gpu.ctx.device_ordinal, .semantic_cases = semantic_cases, .real_rows = rows, .arms = 2 });
    var start = try cuda.Event.init(gpu.d, true);
    defer start.deinit();
    var end = try cuda.Event.init(gpu.d, true);
    defer end.deinit();
    var events: [2][5]f64 = undefined;
    var walls: [2][5]f64 = undefined;
    for (0..6) |pair| for (0..2) |order| {
        const ai = (pair + order) % 2;
        const arm: Arm = if (ai == 0) .host else .device;
        const name = try std.fmt.allocPrint(gpu.gpa, "pair{d}_{s}", .{ pair, @tagName(arm) });
        defer gpu.gpa.free(name);
        try buffers.reset();
        try stream.synchronize();
        const t0 = check.now(gpu.io);
        try start.record(stream);
        try sweep(gpu, &buffers, &kernels, stream, arm, name, journal);
        try end.record(stream);
        try end.synchronize();
        const wall_ms = @as(f64, @floatFromInt(check.now(gpu.io) - t0)) / 1e6;
        const event_ms = try cuda.Event.elapsedMs(start, end);
        if (!std.math.isFinite(wall_ms) or wall_ms <= 0 or !std.math.isFinite(event_ms) or event_ms <= 0) return error.InvalidTerminalTiming;
        try evidence(gpu, &buffers, source, manifest.tokens, arm, name, journal);
        if (pair > 0) {
            walls[ai][pair - 1] = wall_ms;
            events[ai][pair - 1] = event_ms;
        }
        try journal.write(.{ .kind = "terminal_greedy_sample", .ordinal = gpu.ctx.device_ordinal, .arm = @tagName(arm), .pair = pair, .order = order, .warmup = pair == 0, .rows = rows, .dtoh_count = buffers.dtoh_count, .dtoh_bytes = buffers.dtoh_bytes, .row_sync_count = buffers.row_sync_count, .event_sync_count = 1, .kernel_launches = buffers.kernel_launches, .diagnostic_copy_bytes = if (arm == .host) rows * row_bytes else @as(usize, 0), .cpu_argmax_count = if (arm == .host) rows else @as(usize, 0), .cpu_pick_validation_count = if (arm == .device) rows else @as(usize, 0), .wall_ms = wall_ms, .event_ms = event_ms, .outputs_consumed = rows, .free_bytes = try freeBytes(gpu) });
    };
    for ([_]Arm{ .host, .device }, 0..) |arm, ai| {
        const name = try std.fmt.allocPrint(gpu.gpa, "after_{s}", .{@tagName(arm)});
        defer gpu.gpa.free(name);
        try buffers.reset();
        try sweep(gpu, &buffers, &kernels, stream, arm, name, journal);
        try evidence(gpu, &buffers, source, manifest.tokens, arm, name, journal);
        std.mem.sort(f64, &walls[ai], {}, std.sort.asc(f64));
        std.mem.sort(f64, &events[ai], {}, std.sort.asc(f64));
        try journal.write(.{ .kind = "terminal_greedy_summary", .ordinal = gpu.ctx.device_ordinal, .arm = @tagName(arm), .samples = 5, .wall_ms_median = walls[ai][2], .wall_ms_min = walls[ai][0], .wall_ms_max = walls[ai][4], .event_ms_median = events[ai][2], .event_ms_min = events[ai][0], .event_ms_max = events[ai][4], .model_speedup_claim = false });
    }
    const final_input = try check.download(gpu, buffers.logits);
    defer gpu.gpa.free(final_input);
    try save(gpu, journal, "final_logits", final_input);
    try check.sameBytes("terminal immutable logits and guards", final_input, uploaded);
    try journal.write(.{ .kind = "terminal_greedy_complete", .ordinal = gpu.ctx.device_ordinal, .rows = rows, .semantic_cases = semantic_cases, .measured_pairs = 5, .warmup_pairs = 1, .tokens_exact = true, .production_qualification = false, .free_bytes = try freeBytes(gpu) });
}

test "terminal greedy rejects invalid paths, shapes and Pick statuses" {
    try std.testing.expectError(error.InvalidTerminalGreedyInput, validatePaths("relative", "x"));
    try std.testing.expectError(error.InvalidTerminalPick, validatePick(.{ .token = 1, .nonfinite = 8 }, 5));
    try std.testing.expectError(error.InvalidTerminalPick, validatePick(.{ .token = 5, .nonfinite = 0 }, 5));
    try std.testing.expectError(error.InvalidTerminalPick, validatePick(.{ .token = 1, .nonfinite = 1 }, 5));
    try std.testing.expectError(error.NonfiniteLogits, validatePick(.{ .token = 0xffffffff, .nonfinite = 7 }, 5));
}
test "terminal greedy exact comparator covers ties signed zeros and nonfinite kinds" {
    for (0..semantic_cases) |index| {
        var row: [semantic_vocab]u16 = undefined;
        const expected = semanticRow(index, &row);
        if (expected.nonfinite == 0) try std.testing.expectEqual(expected.token, try argmax(&row)) else try std.testing.expectError(error.NonfiniteLogits, argmax(&row));
    }
}
test "terminal greedy guard mutations are rejected" {
    var bytes: [guard * 2 + 8]u8 = @splat(0xa5);
    try canaries(&bytes, 8);
    bytes[0] ^= 1;
    try std.testing.expectError(error.TerminalGuardChanged, canaries(&bytes, 8));
    bytes[0] ^= 1;
    bytes[bytes.len - 1] ^= 1;
    try std.testing.expectError(error.TerminalGuardChanged, canaries(&bytes, 8));
}

test "terminal greedy manifest dimensions hashes and token bounds reject mutations" {
    const sha: [64]u8 = @splat('a');
    const tokens: [rows]u32 = @splat(0);
    const hashes: [rows][]const u8 = @splat(&sha);
    const good: Manifest = .{ .version = 1, .rows = rows, .vocab = vocab, .bytes = rows * row_bytes, .path = "C:/fixture.bin", .sha256 = &sha, .row_sha256 = &hashes, .tokens = &tokens, .tree_image_sha256 = &sha, .argmax_sha256 = &sha };
    try validateManifest(good);
    inline for (.{ "version", "rows", "vocab", "bytes" }) |field| {
        var bad = good;
        @field(bad, field) += 1;
        try std.testing.expectError(error.InvalidTerminalGreedyFixture, validateManifest(bad));
    }
    var bad = good;
    bad.tokens = tokens[0 .. rows - 1];
    try std.testing.expectError(error.InvalidTerminalGreedyFixture, validateManifest(bad));
    var invalid_tokens = tokens;
    invalid_tokens[127] = vocab;
    bad = good;
    bad.tokens = &invalid_tokens;
    try std.testing.expectError(error.InvalidTerminalGreedyFixture, validateManifest(bad));
    bad = good;
    bad.sha256 = "z";
    try std.testing.expectError(error.InvalidTerminalGreedyFixture, validateManifest(bad));
}
