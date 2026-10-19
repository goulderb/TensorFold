const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");

const heads = 32;
const key_heads = 16;
const dim = 128;
const state_bytes = heads * dim * dim * 4;
const out_bytes = heads * dim * 4;
const launches = 128;
const banks = 64;
const bank_bytes = state_bytes + out_bytes;
const timing_bytes = banks * bank_bytes;
const unique_state_bytes = banks * state_bytes;
const floor_bytes = 1536 * 1024 * 1024;
const seed: u32 = 0x35d019;
pub const scope = "synthetic nonzero GDN operator seam; timed 64-bank working set exceeds L2; not physical DRAM bandwidth, a saved model trace, or full-model performance";
const Arm = enum { old, new };
const Inputs = struct {
    q: [key_heads * dim]f32,
    k: [key_heads * dim]f32,
    v: [heads * dim]u16,
    decay: [heads]f32,
    beta: [heads]f32,

    fn patterned(step: usize) Inputs {
        var result: Inputs = undefined;
        for (&result.q, &result.k, 0..) |*q, *k, i| {
            q.* = pattern(i, step, 17) / 8192;
            k.* = pattern(i, step, 29) / 8192;
        }
        for (&result.v, 0..) |*v, i| {
            const value = pattern(i, step, 43) / 8192;
            v.* = @truncate(@as(u32, @bitCast(value)) >> 16);
        }
        for (&result.decay, &result.beta, 0..) |*decay, *beta, h| {
            decay.* = -@as(f32, @floatFromInt(1 + (h * 7 + step * 3) % 31)) / 1024;
            beta.* = 0.5 + @as(f32, @floatFromInt((h * 11 + step * 5) % 33)) / 64;
        }
        return result;
    }
    fn validate(self: *const Inputs) !void {
        try finite(std.mem.asBytes(&self.q));
        try finite(std.mem.asBytes(&self.k));
        try finite(std.mem.asBytes(&self.decay));
        try finite(std.mem.asBytes(&self.beta));
        for (self.v) |bits| if (!std.math.isFinite(@as(f32, @bitCast(@as(u32, bits) << 16)))) return error.NonfiniteGdnValue;
        for (self.decay, self.beta) |decay, beta| if (decay >= 0 or beta < 0.5 or beta > 1) return error.InvalidGdnCoefficients;
    }
    fn digest(self: *const Inputs) [64]u8 {
        var h = std.crypto.hash.sha2.Sha256.init(.{});
        h.update(std.mem.asBytes(&self.q));
        h.update(std.mem.asBytes(&self.k));
        h.update(std.mem.asBytes(&self.v));
        h.update(std.mem.asBytes(&self.decay));
        h.update(std.mem.asBytes(&self.beta));
        var bytes: [32]u8 = undefined;
        h.final(&bytes);
        return std.fmt.bytesToHex(bytes, .lower);
    }
};
fn pattern(index: usize, step: usize, salt: usize) f32 {
    const value: i32 = @intCast((index * salt + step * 19 + seed) % 63);
    return @floatFromInt(if (value < 31) value - 31 else value - 30);
}
fn hash(bytes: []const u8) [64]u8 {
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &digest, .{});
    return std.fmt.bytesToHex(digest, .lower);
}
fn finite(bytes: []const u8) !void {
    if (bytes.len % 4 != 0) return error.InvalidGdnValueSize;
    var offset: usize = 0;
    while (offset < bytes.len) : (offset += 4) {
        const bits = std.mem.readInt(u32, bytes[offset..][0..4], .little);
        if (!std.math.isFinite(@as(f32, @bitCast(bits)))) return error.NonfiniteGdnValue;
    }
}
fn same(actual: []const u8, expected: []const u8) !void {
    if (actual.len != bank_bytes or expected.len != bank_bytes) return error.InvalidGdnBankSize;
    try finite(actual);
    try finite(expected);
    if (!std.mem.eql(u8, actual[0..state_bytes], expected[0..state_bytes])) return error.GdnStateParityMismatch;
    if (!std.mem.eql(u8, actual[state_bytes..], expected[state_bytes..])) return error.GdnOutputParityMismatch;
}
fn sameBanks(actual: []const u8, expected: []const u8) !void {
    if (actual.len != timing_bytes or expected.len != timing_bytes) return error.InvalidGdnTimingSize;
    for (0..banks) |bank| try same(actual[bank * bank_bytes ..][0..bank_bytes], expected[bank * bank_bytes ..][0..bank_bytes]);
}
fn banksHash(bytes: []const u8, output: bool) [64]u8 {
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    for (0..banks) |bank| {
        const offset = bank * bank_bytes;
        h.update(if (output) bytes[offset + state_bytes ..][0..out_bytes] else bytes[offset..][0..state_bytes]);
    }
    var digest: [32]u8 = undefined;
    h.final(&digest);
    return std.fmt.bytesToHex(digest, .lower);
}
fn initial(a: std.mem.Allocator, count: usize) ![]u8 {
    const image = try a.alloc(u8, count * bank_bytes);
    for (0..count) |bank| {
        const part = image[bank * bank_bytes ..][0..bank_bytes];
        for (0..heads) |h| for (0..dim) |v| for (0..dim) |k| {
            const index = (h * dim + v) * dim + k;
            const value = pattern(h * 1597 + v * 137 + k, 0, 37) / 8192 + @as(f32, @floatFromInt(bank)) / 1048576;
            std.mem.writeInt(u32, part[index * 4 ..][0..4], @bitCast(value), .little);
        };
        @memset(part[state_bytes..], 0xff);
    }
    return image;
}
pub fn checkInput(a: std.mem.Allocator) !void {
    const image = try initial(a, 1);
    defer a.free(image);
    try finite(image[0..state_bytes]);
    if (finite(image[state_bytes..])) |_| return error.GdnPoisonControlInsensitive else |err| if (err != error.NonfiniteGdnValue) return err;
    var input = Inputs.patterned(0);
    try input.validate();
    for (1..16) |step| {
        const changed = Inputs.patterned(step);
        try changed.validate();
        if (std.mem.eql(f32, &input.q, &changed.q) or std.mem.eql(f32, &input.k, &changed.k) or std.mem.eql(u16, &input.v, &changed.v) or std.mem.eql(f32, &input.decay, &changed.decay) or std.mem.eql(f32, &input.beta, &changed.beta)) return error.GdnChangingInputControlInsensitive;
        input = changed;
    }
    const other = try a.dupe(u8, image);
    defer a.free(other);
    @memset(image[state_bytes..], 0);
    @memset(other[state_bytes..], 0);
    other[0] ^= 1;
    if (same(image, other)) |_| return error.GdnStateControlInsensitive else |err| if (err != error.GdnStateParityMismatch) return err;
    other[0] ^= 1;
    other[state_bytes] ^= 1;
    if (same(image, other)) |_| return error.GdnOutputControlInsensitive else |err| if (err != error.GdnOutputParityMismatch) return err;
    if (banks != 64 or launches != banks * 2 or unique_state_bytes != 128 * 1024 * 1024 or timing_bytes != banks * bank_bytes) return error.InvalidGdnBankWork;
    const all = try initial(a, banks);
    defer a.free(all);
    for (0..banks) |bank| {
        const part = all[bank * bank_bytes ..][0..bank_bytes];
        try finite(part[0..state_bytes]);
        @memset(part[state_bytes..], 0);
        if (bank != 0 and std.mem.eql(u8, all[0..state_bytes], part[0..state_bytes])) return error.GdnBankPatternControlInsensitive;
    }
    const late = try a.dupe(u8, all);
    defer a.free(late);
    try sameBanks(late, all);
    late[(banks - 1) * bank_bytes] ^= 1;
    if (sameBanks(late, all)) |_| return error.GdnLateBankStateControlInsensitive else |err| if (err != error.GdnStateParityMismatch) return err;
    late[(banks - 1) * bank_bytes] ^= 1;
    late[(banks - 1) * bank_bytes + state_bytes] ^= 1;
    if (sameBanks(late, all)) |_| return error.GdnLateBankOutputControlInsensitive else |err| if (err != error.GdnOutputParityMismatch) return err;
}
fn floor(gpu: check.Gpu) !usize {
    const free = (try gpu.ctx.memInfo()).free;
    if (free < floor_bytes) return error.GpuMemoryFloor;
    return free;
}
const Buffers = struct {
    inputs: cuda.DeviceBuffer,
    states: [2]cuda.DeviceBuffer,
    fn upload(self: Buffers, input: *const Inputs) !void {
        try self.inputs.upload(0, std.mem.asBytes(&input.q));
        try self.inputs.upload(8192, std.mem.asBytes(&input.k));
        try self.inputs.upload(16384, std.mem.asBytes(&input.v));
        try self.inputs.upload(24576, std.mem.asBytes(&input.decay));
        try self.inputs.upload(24704, std.mem.asBytes(&input.beta));
    }
    fn launch(self: Buffers, f: cuda.Function, arm: usize, bank: usize, stream: cuda.Stream) !void {
        var args: cuda.Args = .{};
        for ([_]u64{ self.inputs.ptr, self.inputs.ptr + 8192, self.inputs.ptr + 16384, self.inputs.ptr + 24576, self.inputs.ptr + 24704, self.states[arm].ptr + bank * bank_bytes, self.states[arm].ptr + bank * bank_bytes + state_bytes }) |address| args.add(address);
        try cuda.launch.launch(f, .{ .grid = .{ .x = heads }, .block = .{ .x = dim } }, stream, &args);
    }
};
fn capture(b: Buffers, f: cuda.Function, arm: usize, s: cuda.Stream) !cuda.graph.Exec {
    try cuda.graph.beginCapture(s, .thread_local);
    for (0..launches) |i| b.launch(f, arm, i % banks, s) catch |err| {
        if (cuda.graph.endCapture(s)) |g| {
            var dead = g;
            dead.deinit();
        } else |_| {}
        return err;
    };
    var graph = try cuda.graph.endCapture(s);
    defer graph.deinit();
    if (try graph.nodeCount() != launches) return error.GdnGraphWorkCountMismatch;
    var exec = try graph.instantiate();
    errdefer exec.deinit();
    try exec.upload(s);
    try s.synchronize();
    return exec;
}
pub fn run(gpu: check.Gpu, journal: anytype) !void {
    _ = try floor(gpu);
    try checkInput(gpu.gpa);
    var arena = std.heap.ArenaAllocator.init(gpu.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const timing_image = try initial(a, banks);
    const image = timing_image[0..bank_bytes];
    const reference = try a.alloc(u8, image.len);
    const actual = try a.alloc(u8, image.len);
    const unwritten = try a.dupe(u8, image);
    var module = try cuda.Module.load(gpu.d, cuda.kernels.qwen35_ops);
    defer module.unload();
    const functions = [_]cuda.Function{ try module.function("tf_qwen35_gdn_step"), try module.function("tf_qwen35_gdn_step_tiled") };
    var stream = try cuda.Stream.init(gpu.d, false);
    defer stream.deinit();
    var inputs = try cuda.DeviceBuffer.alloc(gpu.d, 24832);
    defer inputs.free();
    var old = try cuda.DeviceBuffer.fromHost(gpu.d, timing_image);
    defer old.free();
    var new = try cuda.DeviceBuffer.fromHost(gpu.d, timing_image);
    defer new.free();
    const b: Buffers = .{ .inputs = inputs, .states = .{ old, new } };
    const initial_hash = hash(image[0..state_bytes]);
    const free_after_allocation = try floor(gpu);
    try journal.write(.{ .kind = "gdn_fixture", .ordinal = gpu.ctx.device_ordinal, .seed = seed, .pattern_version = 2, .state_layout = "head,value,key", .value_heads = heads, .key_heads = key_heads, .dimension = dim, .state_bytes = state_bytes, .out_bytes = out_bytes, .q_dtype = "FP32", .k_dtype = "FP32", .v_dtype = "BF16", .state_dtype = "FP32", .out_dtype = "FP32", .q_elements = key_heads * dim, .k_elements = key_heads * dim, .v_elements = heads * dim, .decay_elements = heads, .beta_elements = heads, .initial_state_sha256 = &initial_hash, .free_bytes = free_after_allocation, .floor_bytes = floor_bytes, .scope = scope });
    try old.download(0, actual);
    if (finite(actual[state_bytes..])) |_| return error.GdnPoisonControlInsensitive else |err| if (err != error.NonfiniteGdnValue) return err;
    try journal.write(.{ .kind = "gdn_negative_control", .ordinal = gpu.ctx.device_ordinal, .poisoned_output_rejected = true, .changed_state_rejected = true, .changed_output_rejected = true, .late_bank_state_mismatch_rejected = true, .late_bank_output_mismatch_rejected = true });
    for (0..16) |step| {
        const input = Inputs.patterned(step);
        try input.validate();
        try b.upload(&input);
        for (functions, 0..) |f, arm| {
            try b.states[arm].upload(state_bytes, image[state_bytes..]);
            try b.launch(f, arm, 0, stream);
            try stream.synchronize();
            try b.states[arm].download(0, if (arm == 0) reference else actual);
        }
        try same(actual, reference);
        if (step == 0) {
            @memcpy(unwritten[state_bytes..], reference[state_bytes..]);
            if (same(unwritten, reference)) |_| return error.GdnUnwrittenStateControlInsensitive else |err| if (err != error.GdnStateParityMismatch) return err;
            try journal.write(.{ .kind = "gdn_unwritten_state_control", .ordinal = gpu.ctx.device_ordinal, .unchanged_initial_state_with_correct_output_rejected = true });
        }
        if (std.mem.eql(u8, actual[0..state_bytes], image[0..state_bytes])) return error.GdnUnwrittenStateControlInsensitive;
        const state_hash = hash(actual[0..state_bytes]);
        const out_hash = hash(actual[state_bytes..]);
        const input_hash = input.digest();
        try journal.write(.{ .kind = "gdn_step_parity", .ordinal = gpu.ctx.device_ordinal, .step = step, .inputs_sha256 = &input_hash, .state_sha256 = &state_hash, .output_sha256 = &out_hash, .state_bit_exact = true, .output_bit_exact = true, .finite_state = true, .finite_output = true, .unwritten_initial_state_rejected = true, .free_bytes = try floor(gpu) });
    }
    const fixed = Inputs.patterned(0);
    try b.upload(&fixed);
    const timing_reference = try a.alloc(u8, timing_bytes);
    const timing_actual = try a.alloc(u8, timing_bytes);
    const timing_initial_hash = hash(timing_image);
    const timing_initial_state_hash = banksHash(timing_image, false);
    try old.upload(0, timing_image);
    for (0..launches) |i| try b.launch(functions[0], 0, i % banks, stream);
    try stream.synchronize();
    try old.download(0, timing_reference);
    try finite(timing_reference);
    var graphs: [2]cuda.graph.Exec = undefined;
    var made: usize = 0;
    defer for (graphs[0..made]) |*graph| graph.deinit();
    for (functions, std.meta.tags(Arm), 0..) |f, arm, index| {
        graphs[index] = try capture(b, f, index, stream);
        made += 1;
        try journal.write(.{ .kind = "gdn_graph", .ordinal = gpu.ctx.device_ordinal, .arm = @tagName(arm), .node_count = launches, .kernel_launches = launches, .grid_x = heads, .block_x = dim, .inputs_sha256 = &fixed.digest(), .expected_state_sha256 = &banksHash(timing_reference, false), .expected_output_sha256 = &banksHash(timing_reference, true), .expected_payload_sha256 = &hash(timing_reference), .initial_payload_sha256 = &timing_initial_hash, .banks = banks, .unique_state_bytes = unique_state_bytes, .bank_bytes = bank_bytes, .payload_bytes = timing_bytes, .updates_per_bank = launches / banks, .scope = scope, .evolving_state = true, .free_bytes = try floor(gpu) });
    }
    var start = try cuda.Event.init(gpu.d, true);
    defer start.deinit();
    var end = try cuda.Event.init(gpu.d, true);
    defer end.deinit();
    for (0..6) |pair| for (0..2) |order| {
        const index = if (pair % 2 == 0) order else 1 - order;
        const arm: Arm = @fromBackingInt(@intCast(index));
        const before_free = try floor(gpu);
        try b.states[index].upload(0, timing_image);
        try stream.synchronize();
        try start.record(stream);
        try graphs[index].launchOn(stream);
        try end.record(stream);
        try end.synchronize();
        const elapsed_ms = try cuda.Event.elapsedMs(start, end);
        if (!std.math.isFinite(elapsed_ms) or elapsed_ms <= 0) return error.InvalidGdnElapsedTime;
        try b.states[index].download(0, timing_actual);
        try sameBanks(timing_actual, timing_reference);
        try journal.write(.{ .kind = "gdn_sample", .ordinal = gpu.ctx.device_ordinal, .pair = pair, .order = order, .arm = @tagName(arm), .warmup = pair == 0, .elapsed_ms = elapsed_ms, .kernel_launches = launches, .node_count = launches, .graph_launches = 1, .state_bit_exact = true, .output_bit_exact = true, .finite_state = true, .finite_output = true, .state_sha256 = &banksHash(timing_actual, false), .output_sha256 = &banksHash(timing_actual, true), .payload_sha256 = &hash(timing_actual), .initial_payload_sha256 = &timing_initial_hash, .initial_state_sha256 = &timing_initial_state_hash, .banks = banks, .unique_state_bytes = unique_state_bytes, .bank_bytes = bank_bytes, .payload_bytes = timing_bytes, .updates_per_bank = launches / banks, .reset_before_event = true, .free_bytes_before = before_free, .free_bytes = try floor(gpu), .floor_bytes = floor_bytes, .scope = scope });
    };
    try journal.write(.{ .kind = "gdn_screen_complete", .ordinal = gpu.ctx.device_ordinal, .correctness_steps = 16, .banks = banks, .unique_state_bytes = unique_state_bytes, .updates_per_bank = launches / banks, .measured_pairs = 5, .excluded_warmup_pairs = 1, .correctness_pass = true, .scope = scope, .production_qualification = false });
}
