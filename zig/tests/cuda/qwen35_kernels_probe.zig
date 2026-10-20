const std = @import("std");
const cuda = @import("cuda");
const ops_module = @import("qwen35_ops");
const ref = @import("qwen35_ref");
const device_admission = @import("qwen35_probe_device.zig");
const Ops = ops_module.Ops;
const Buffer = cuda.DeviceBuffer;
const floor = 1536 << 20;

const Tolerance = struct { absolute: f32 = 0, relative: f32 = 0 };
// One BF16 spacing permits the independently ordered FP32 reduction to straddle a rounding midpoint.
const bf16_tolerance: Tolerance = .{ .absolute = 1e-6, .relative = 1.0 / 128.0 };
// Q/K and the four-step FP32 recurrence use different host/device reduction orders.
const qk_tolerance: Tolerance = .{ .absolute = 4e-7, .relative = 3e-5 };
const state_tolerance: Tolerance = .{ .absolute = 5e-6, .relative = 5e-5 };
const gate_tolerance: Tolerance = .{ .absolute = 2e-6, .relative = 2e-6 };

const Result = struct { name: []const u8, elements: usize, absolute: f32, relative: f32, max_absolute: f32, max_relative: f32 };

const Rig = struct {
    a: std.mem.Allocator,
    ctx: *const cuda.Context,
    stream: cuda.Stream,
    ops: *const Ops,
    buffers: std.ArrayList(Buffer) = .empty,
    results: std.ArrayList(Result) = .empty,

    fn deinit(r: *Rig) void {
        for (r.buffers.items) |*buffer| buffer.free();
        r.buffers.deinit(r.a);
        r.results.deinit(r.a);
    }

    fn alloc(r: *Rig, comptime T: type, count: usize) !Buffer {
        const bytes = std.math.mul(usize, count, @sizeOf(T)) catch return error.InvalidSize;
        if ((try r.ctx.memInfo()).free < floor + bytes) return error.GpuMemoryFloor;
        var buffer = try Buffer.alloc(r.ctx.d, bytes);
        errdefer buffer.free();
        try r.buffers.append(r.a, buffer);
        return buffer;
    }

    fn floats(r: *Rig, values: []const f32) !Buffer {
        const buffer = try r.alloc(f32, values.len);
        try buffer.upload(0, std.mem.sliceAsBytes(values));
        return buffer;
    }

    fn bfloats(r: *Rig, values: []const f32) !Buffer {
        const bits = try r.a.alloc(u16, values.len);
        for (values, bits) |x, *out| out.* = @truncate(@as(u32, @bitCast(ref.bf16(x))) >> 16);
        const buffer = try r.alloc(u16, values.len);
        try buffer.upload(0, std.mem.sliceAsBytes(bits));
        return buffer;
    }

    fn download(r: *Rig, comptime T: type, buffer: Buffer, count: usize) ![]T {
        try r.stream.synchronize();
        const out = try r.a.alloc(T, count);
        try buffer.download(0, std.mem.sliceAsBytes(out));
        return out;
    }

    fn check(r: *Rig, name: []const u8, got: []const f32, want: []const f32, tolerance: Tolerance) !void {
        if (got.len != want.len) return error.ProbeShape;
        var result: Result = .{ .name = name, .elements = got.len, .absolute = tolerance.absolute, .relative = tolerance.relative, .max_absolute = 0, .max_relative = 0 };
        for (got, want, 0..) |actual, expected, i| {
            if (!std.math.isFinite(actual) or !std.math.isFinite(expected)) {
                std.debug.print("nonfinite {s}[{d}] actual={d} expected={d}\n", .{ name, i, actual, expected });
                return error.NonfiniteKernelValue;
            }
            const difference = @abs(actual - expected);
            result.max_absolute = @max(result.max_absolute, difference);
            if (expected != 0) result.max_relative = @max(result.max_relative, difference / @abs(expected));
            if (difference > tolerance.absolute + tolerance.relative * @abs(expected)) {
                std.debug.print("mismatch {s}[{d}] actual={d} expected={d} abs={d} allowed={d}\n", .{ name, i, actual, expected, difference, tolerance.absolute + tolerance.relative * @abs(expected) });
                return error.KernelMismatch;
            }
        }
        try r.results.append(r.a, result);
    }

    fn checkF32(r: *Rig, name: []const u8, buffer: Buffer, want: []const f32, tolerance: Tolerance) !void {
        try r.check(name, try r.download(f32, buffer, want.len), want, tolerance);
    }

    fn checkBf16(r: *Rig, name: []const u8, buffer: Buffer, want: []const f32, tolerance: Tolerance) !void {
        const bits = try r.download(u16, buffer, want.len);
        const values = try r.a.alloc(f32, bits.len);
        for (bits, values) |x, *out| out.* = @bitCast(@as(u32, x) << 16);
        try r.check(name, values, want, tolerance);
    }
};

fn pattern(a: std.mem.Allocator, n: usize, salt: usize, scale: f32) ![]f32 {
    const out = try a.alloc(f32, n);
    for (out, 0..) |*value, i| value.* = ref.bf16(@as(f32, @floatFromInt(@as(i32, @intCast((i * 17 + salt * 11) % 97)) - 48)) * scale);
    return out;
}

fn simple(r: *Rig) !void {
    const table = try pattern(r.a, 3 * 2048, 3, 1.0 / 32.0);
    const table_dev = try r.bfloats(table);
    const out = try r.alloc(u16, 2048);
    try r.ops.embed(r.stream, table_dev.ptr, 2, out.ptr);
    try r.checkBf16("embed", out, table[4096..], .{});

    const x = try pattern(r.a, 2048, 5, 1.0 / 32.0);
    const update = try pattern(r.a, 2048, 8, 1.0 / 64.0);
    x[0] = 1;
    x[1] = 1 + 1.0 / 128.0;
    x[2] = -1;
    x[3] = -1 - 1.0 / 128.0;
    update[0] = 1.0 / 256.0;
    update[1] = 1.0 / 256.0;
    update[2] = -1.0 / 256.0;
    update[3] = -1.0 / 256.0;
    const x_dev = try r.bfloats(x);
    const update_dev = try r.bfloats(update);
    const want = try r.a.alloc(f32, x.len);
    for (x, update, want) |left, right, *value| value.* = ref.bf16(left + right);
    try r.ops.add(r.stream, x_dev.ptr, update_dev.ptr, x.len);
    try r.checkBf16("add", x_dev, want, .{});

    const norm = try pattern(r.a, 2048, 2, 1.0 / 128.0);
    const norm_dev = try r.floats(norm);
    var variance: f64 = 0;
    for (want) |value| variance += @as(f64, value) * value;
    const inv = 1 / @sqrt(variance / 2048 + 1e-6);
    const normalized = try r.a.alloc(f32, x.len);
    for (want, norm, normalized) |value, weight, *result| result.* = ref.bf16(@floatCast(@as(f64, value) * inv * weight));
    try r.ops.rms(r.stream, x_dev.ptr, norm_dev.ptr, out.ptr, 2048, 1e-6);
    try r.checkBf16("rms_existing_norms", out, normalized, bf16_tolerance);

    const gate = try pattern(r.a, 512, 4, 1.0 / 16.0);
    const up = try pattern(r.a, 512, 9, 1.0 / 16.0);
    const gate_dev = try r.bfloats(gate);
    const up_dev = try r.bfloats(up);
    const activated = try r.a.alloc(f32, 512);
    for (gate, up, activated) |g, u, *value| value.* = ref.bf16((g / (1 + @exp(-g))) * u);
    try r.ops.swiglu(r.stream, gate_dev.ptr, up_dev.ptr, out.ptr, 512);
    try r.checkBf16("swiglu", out, activated, bf16_tolerance);
}

fn gdn(r: *Rig) !void {
    const weights = try r.a.alloc(f32, 8192 * 4);
    for (0..8192) |ch| @memcpy(weights[ch * 4 ..][0..4], &[_]f32{ -0.125, 0.25, 0.5, 0.125 });
    const history = try pattern(r.a, 3 * 8192, 12, 1.0 / 64.0);
    for (0..3) |row| {
        @memset(history[row * 8192 ..][0..128], 0);
        @memset(history[row * 8192 + 2048 ..][0..128], 0);
    }
    const history_dev = try r.bfloats(history);
    const weights_dev = try r.floats(weights);
    const a = try pattern(r.a, 32, 7, 1.0 / 16.0);
    const b = try pattern(r.a, 32, 6, 1.0 / 16.0);
    const a_log = try pattern(r.a, 32, 5, 1.0 / 64.0);
    const dt_bias = try pattern(r.a, 32, 4, 1.0 / 64.0);
    a[0] = 100;
    a[1] = -100;
    b[0] = -8;
    b[1] = 8;
    const a_dev = try r.bfloats(a);
    const b_dev = try r.bfloats(b);
    const log_dev = try r.floats(a_log);
    const bias_dev = try r.floats(dt_bias);
    const q_dev = try r.alloc(f32, 2048);
    const k_dev = try r.alloc(f32, 2048);
    const v_dev = try r.alloc(u16, 4096);
    const decay_dev = try r.alloc(f32, 32);
    const beta_dev = try r.alloc(f32, 32);
    const y_dev = try r.alloc(f32, 4096);
    const out_dev = try r.alloc(u16, 4096);
    const states = try r.a.alloc([128][128]f32, 32);
    const state_input = try r.a.alloc(f32, 32 * 128 * 128);
    for (states, 0..) |*state, head| {
        for (state, 0..) |*row, key| {
            for (row, 0..) |*cell, value| {
                cell.* = @as(f32, @floatFromInt(@as(i32, @intCast((head + 3 * key + value) % 19)) - 9)) / 4096;
                state_input[(head * 128 + value) * 128 + key] = cell.*;
            }
        }
    }
    const state_dev = try r.floats(state_input);
    const z = try pattern(r.a, 4096, 8, 1.0 / 16.0);
    const norm = try pattern(r.a, 128, 2, 1.0 / 64.0);
    const z_dev = try r.bfloats(z);
    const norm_dev = try r.floats(norm);
    const conv = try r.a.alloc(f32, 8192);
    var q: [16][128]f32 = undefined;
    var k: [16][128]f32 = undefined;
    var decay: [32]f32 = undefined;
    var beta: [32]f32 = undefined;
    var y: [32][128]f32 = undefined;
    var normalized: [32][128]f32 = undefined;
    for (0..4) |step| {
        const input = try pattern(r.a, 8192, step + 1, 1.0 / 64.0);
        @memset(input[0..128], 0);
        @memset(input[2048..2176], 0);
        input[0] = ref.bf16(1e-4);
        input[2048] = ref.bf16(-1e-4);
        const input_dev = try r.bfloats(input);
        ref.conv(input, weights, history, conv);
        for (0..16) |head| {
            q[head] = ref.l2(conv[head * 128 ..][0..128].*, true);
            k[head] = ref.l2(conv[2048 + head * 128 ..][0..128].*, false);
        }
        for (0..32) |head| {
            const gates = ref.gates(a[head], b[head], a_log[head], dt_bias[head]);
            decay[head] = gates.log_decay;
            beta[head] = gates.beta;
        }
        try r.ops.gdnPre(r.stream, input_dev.ptr, weights_dev.ptr, history_dev.ptr, a_dev.ptr, b_dev.ptr, log_dev.ptr, bias_dev.ptr, q_dev.ptr, k_dev.ptr, v_dev.ptr, decay_dev.ptr, beta_dev.ptr);
        try r.checkF32(try std.fmt.allocPrint(r.a, "gdn_pre_q_step{d}", .{step}), q_dev, std.mem.bytesAsSlice(f32, std.mem.asBytes(&q)), qk_tolerance);
        try r.checkF32(try std.fmt.allocPrint(r.a, "gdn_pre_k_step{d}", .{step}), k_dev, std.mem.bytesAsSlice(f32, std.mem.asBytes(&k)), qk_tolerance);
        try r.checkBf16(try std.fmt.allocPrint(r.a, "gdn_pre_v_step{d}", .{step}), v_dev, conv[4096..], bf16_tolerance);
        try r.checkF32(try std.fmt.allocPrint(r.a, "gdn_pre_decay_step{d}", .{step}), decay_dev, &decay, gate_tolerance);
        try r.checkF32(try std.fmt.allocPrint(r.a, "gdn_pre_beta_step{d}", .{step}), beta_dev, &beta, .{});
        try r.checkBf16(try std.fmt.allocPrint(r.a, "gdn_pre_history_step{d}", .{step}), history_dev, history, .{});
        for (0..32) |head| {
            y[head] = ref.gdn(q[head / 2], k[head / 2], conv[4096 + head * 128 ..][0..128].*, decay[head], beta[head], &states[head]);
            for (0..128) |value| {
                for (0..128) |key| state_input[(head * 128 + value) * 128 + key] = states[head][key][value];
            }
        }
        try r.ops.gdnStep(r.stream, q_dev.ptr, k_dev.ptr, v_dev.ptr, decay_dev.ptr, beta_dev.ptr, state_dev.ptr, y_dev.ptr);
        try r.checkF32(try std.fmt.allocPrint(r.a, "gdn_state_step{d}", .{step}), state_dev, state_input, state_tolerance);
        try r.checkF32(try std.fmt.allocPrint(r.a, "gdn_y_step{d}", .{step}), y_dev, std.mem.bytesAsSlice(f32, std.mem.asBytes(&y)), state_tolerance);
        // Compare normalization on identical inputs: accepted FP32 recurrence drift can cross a BF16 midpoint.
        const actual_y = try r.download(f32, y_dev, 4096);
        for (0..32) |head| normalized[head] = ref.gatedNorm(actual_y[head * 128 ..][0..128].*, z[head * 128 ..][0..128].*, norm[0..128].*, 1e-6);
        try r.ops.gdnOut(r.stream, y_dev.ptr, z_dev.ptr, norm_dev.ptr, out_dev.ptr, 1e-6);
        try r.checkBf16(try std.fmt.allocPrint(r.a, "gdn_out_step{d}", .{step}), out_dev, std.mem.bytesAsSlice(f32, std.mem.asBytes(&normalized)), bf16_tolerance);
    }
}

fn attention(r: *Rig) !void {
    const capacity = 4;
    const keys = try r.a.alloc(f32, capacity * 512);
    const values = try r.a.alloc(f32, capacity * 512);
    @memset(keys, -0.75);
    @memset(values, 100);
    const keys_dev = try r.bfloats(keys);
    const values_dev = try r.bfloats(values);
    var q_norm: [256]f32 = undefined;
    var k_norm: [256]f32 = undefined;
    for (&q_norm, &k_norm, 0..) |*q, *k, d| {
        q.* = 1 + @as(f32, @floatFromInt(d % 5)) / 16;
        k.* = 0.5 + @as(f32, @floatFromInt(d % 7)) / 8;
    }
    const q_norm_dev = try r.floats(&q_norm);
    const k_norm_dev = try r.floats(&k_norm);
    const out_dev = try r.alloc(u16, 4096);
    for (0..capacity) |position| {
        const qg = try pattern(r.a, 8192, position + 1, 1.0 / 32.0);
        const key = try pattern(r.a, 512, position + 3, 1.0 / 64.0);
        const value = try pattern(r.a, 512, position + 5, 1.0 / 32.0);
        const qg_dev = try r.bfloats(qg);
        const key_dev = try r.bfloats(key);
        const value_dev = try r.bfloats(value);
        for (0..2) |head| {
            var squared: f64 = 0;
            for (key[head * 256 ..][0..256]) |x| squared += @as(f64, x) * x;
            const inv = 1 / @sqrt(squared / 256 + 1e-6);
            var normalized: [256]f32 = undefined;
            for (&normalized, 0..) |*out, d| out.* = ref.bf16(@floatCast(@as(f64, key[head * 256 + d]) * inv * k_norm[d]));
            @memcpy(keys[position * 512 + head * 256 ..][0..256], &ref.rotary(normalized, position));
        }
        @memcpy(values[position * 512 ..][0..512], value);
        var q_gate: [16][512]f32 = undefined;
        for (&q_gate, 0..) |*head, h| @memcpy(head, qg[h * 512 ..][0..512]);
        const want = ref.attention(q_gate, q_norm, keys, values, position);
        try r.ops.attention(r.stream, qg_dev.ptr, key_dev.ptr, value_dev.ptr, q_norm_dev.ptr, k_norm_dev.ptr, keys_dev.ptr, values_dev.ptr, position, capacity, out_dev.ptr, 1e-6);
        try r.checkBf16(try std.fmt.allocPrint(r.a, "attention_kv_keys_pos{d}", .{position}), keys_dev, keys, bf16_tolerance);
        try r.checkBf16(try std.fmt.allocPrint(r.a, "attention_kv_values_pos{d}", .{position}), values_dev, values, .{});
        try r.checkBf16(try std.fmt.allocPrint(r.a, "attention_pos{d}", .{position}), out_dev, std.mem.bytesAsSlice(f32, std.mem.asBytes(&want)), bf16_tolerance);
    }
}

fn routing(r: *Rig) !void {
    const x = try r.a.alloc(f32, 2048);
    @memset(x, 0);
    x[0] = 1;
    const x_dev = try r.bfloats(x);
    const router = try r.a.alloc(f32, 256 * 2048);
    @memset(router, 0);
    const shared_gate = try r.a.alloc(f32, 2048);
    @memset(shared_gate, 0);
    shared_gate[0] = -0.75;
    const shared_dev = try r.bfloats(shared_gate);
    const picks_dev = try r.alloc(i32, 8);
    const weights_dev = try r.alloc(f32, 9);
    const scores_dev = try r.alloc(f32, 257);
    for (0..2) |round| {
        var logits: [256]f32 = @splat(0);
        if (round == 1) {
            for (&logits, 0..) |*value, expert| value.* = ref.bf16(@as(f32, @floatFromInt(expert % 13)) / 8 - 2);
            logits[7] = 2;
            logits[200] = 2;
        }
        for (logits, 0..) |logit, expert| router[expert * 2048] = logit;
        const router_dev = try r.bfloats(router);
        const want = ref.route(logits, -0.75);
        try r.ops.route(r.stream, x_dev.ptr, router_dev.ptr, shared_dev.ptr, picks_dev.ptr, weights_dev.ptr, scores_dev.ptr);
        const got_picks = try r.download(i32, picks_dev, 8);
        if (!std.mem.eql(i32, got_picks, &want.picks)) return error.RoutePickMismatch;
        try r.checkF32(if (round == 0) "route_equal_ties" else "route_fractional_weights", weights_dev, &want.weights, gate_tolerance);
        const pairs = try r.a.alloc([2048]f32, 8);
        for (pairs, 0..) |*pair, slot| for (pair, 0..) |*value, d| {
            value.* = @as(f32, @floatFromInt(@as(i32, @intCast((slot * 11 + d) % 37)) - 18)) / 64;
        };
        const pair_dev = try r.floats(std.mem.bytesAsSlice(f32, std.mem.sliceAsBytes(pairs)));
        const shared = try pattern(r.a, 2048, 4, 1.0 / 32.0);
        const shared_out = try r.bfloats(shared);
        const out = try r.alloc(u16, 2048);
        const combined = ref.combine(pairs[0..8].*, want.weights, shared[0..2048].*);
        try r.ops.combine(r.stream, pair_dev.ptr, weights_dev.ptr, shared_out.ptr, out.ptr);
        try r.checkBf16(if (round == 0) "combine_equal_ties" else "combine_fractional_shared", out, &combined, bf16_tolerance);
    }
}

pub fn main(init: std.process.Init) !u8 {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 5 or !std.mem.eql(u8, args[1], "--approved-device") or !std.mem.eql(u8, args[3], "--expected-uuid")) {
        try std.Io.File.stderr().writeStreamingAll(init.io, "usage: tf-qwen35-kernels-probe --approved-device ORDINAL --expected-uuid GPU-UUID\nRequires PCI_BUS_ID ordering, two full UUIDs in CUDA_VISIBLE_DEVICES, and an approved device window. Building this executable runs no device code.\n");
        return 2;
    }
    const ordinal = try std.fmt.parseInt(c_int, args[2], 10);
    const admission = try device_admission.Admission.parse(init.environ_map.get("CUDA_DEVICE_ORDER"), init.environ_map.get("CUDA_VISIBLE_DEVICES"), &.{ordinal}, &.{args[4]});
    var driver = try cuda.Driver.open();
    defer driver.close();
    const identities = try admission.verify(init.arena.allocator(), &driver);
    var context = try cuda.Context.init(&driver, ordinal);
    defer context.deinit();
    if ((try context.memInfo()).free < floor + (32 << 20)) return error.GpuMemoryFloor;
    var stream = try cuda.Stream.init(&driver, false);
    defer stream.deinit();
    var ops = try Ops.load(&context);
    defer ops.unload();
    var rig: Rig = .{ .a = init.arena.allocator(), .ctx = &context, .stream = stream, .ops = &ops };
    defer rig.deinit();
    try simple(&rig);
    try gdn(&rig);
    try attention(&rig);
    try routing(&rig);
    try stream.synchronize();
    const report = .{
        .status = "pass",
        .device = ordinal,
        .device_identities = identities,
        .reference = "independent host implementation of pinned HF equations, Transformers e0eb2810ac1b2caf6d082f247d8d37162ecf67a0",
        .reference_vectors = "handwritten host equations; not HF-generated vectors; end-to-end HF comparison remains separate",
        .routing_weight_contract = "FP32 selected-softmax weights; HF eager casts them to BF16",
        .entries = [_][]const u8{ "embed", "add", "swiglu", "gdn_pre", "gdn_step", "gdn_out", "attention_kv", "attention", "route", "combine" },
        .checks = rig.results.items,
        .free_mib = (try context.memInfo()).free / (1 << 20),
    };
    var output: std.Io.Writer.Allocating = .init(init.arena.allocator());
    try std.json.Stringify.value(report, .{ .whitespace = .indent_2 }, &output.writer);
    try output.writer.writeByte('\n');
    try std.Io.File.stdout().writeStreamingAll(init.io, output.written());
    return 0;
}
