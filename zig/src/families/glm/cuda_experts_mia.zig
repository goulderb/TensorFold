//! Mia's MCG4 routed GLM experts. Numerical kernels are pinned from import/mia-zig-f1ee79a6.
const std = @import("std");
const cuda = @import("cuda");

pub const Mode = enum { generic, mia_mcg4 };

pub const Experts = struct {
    gate_trellis: u64,
    up_trellis: u64,
    down_trellis: u64,
    gate_suh: u64,
    up_suh: u64,
    down_suh: u64,
    gate_svh: u64,
    up_svh: u64,
    down_svh: u64,
    shared_suh: u64 = 0,
    dims: usize,
    width: usize,
    count: usize,
};

pub fn validate(dims: usize, width: usize, experts: usize, slots: usize) !void {
    if (dims != 4096 or width != 1024 or experts == 0 or experts > 1024 or slots < 2 or slots > 32) return error.UnsupportedMiaGeometry;
}

pub const Scratch = struct {
    xg: u64,
    xu: u64,
    xd: u64,
    work: u64,
    capacity: usize,
    decode_capacity: usize,
    slots: usize,

    pub fn bytes(capacity: usize, decode_capacity: usize, slots: usize, dims: usize, width: usize) !usize {
        try validate(dims, width, 1, slots);
        if (decode_capacity == 0 or decode_capacity > capacity) return error.InvalidMiaCapacity;
        const pairs = try std.math.mul(usize, capacity, slots);
        const rotations = try std.math.mul(usize, pairs, (2 * dims + width) * 2);
        const decode_pairs = try std.math.mul(usize, decode_capacity, slots);
        const partials = try std.math.mul(usize, decode_pairs, 2 * 4 * @max(dims, width) * 4);
        const prompt = try std.math.mul(usize, pairs, dims * 4);
        return std.math.add(usize, rotations, @max(partials, prompt));
    }

    pub fn fromBase(base: u64, capacity: usize, decode_capacity: usize, slots: usize, dims: usize, width: usize) !Scratch {
        _ = try bytes(capacity, decode_capacity, slots, dims, width);
        const pairs = capacity * slots;
        const xg = base;
        const xu = xg + pairs * dims * 2;
        const xd = xu + pairs * dims * 2;
        const work = xd + pairs * width * 2;
        return .{ .xg = xg, .xu = xu, .xd = xd, .work = work, .capacity = capacity, .decode_capacity = decode_capacity, .slots = slots };
    }
};

pub const Kernels = struct {
    module: cuda.Module,
    rot: cuda.Function,
    rot_shared: cuda.Function,
    decode_gu: cuda.Function,
    decode_gu_shared: cuda.Function,
    gateup: cuda.Function,
    decode_down: cuda.Function,
    prompt_gu: cuda.Function,
    prompt_gu_shared: cuda.Function,
    prompt_down: cuda.Function,
    cast_prompt: cuda.Function,

    pub fn load(d: *const cuda.Driver, image: []const u8) !Kernels {
        var m = try cuda.Module.load(d, image);
        errdefer m.unload();
        const rot = try m.function("glm_exl3_rot_0");
        const rot_shared = try m.function("glm_exl3_rot_1");
        const decode_gu = try m.function("glm_exl3_dec_16_0_0");
        const decode_gu_shared = try m.function("glm_exl3_dec_16_0_1");
        const gateup = try m.function("glm_exl3_gateup");
        const decode_down = try m.function("glm_exl3_dec_16_1_0");
        const prompt_gu = try m.function("glm_exl3_prompt_gu_0");
        const prompt_gu_shared = try m.function("glm_exl3_prompt_gu_1");
        const prompt_down = try m.function("glm_exl3_prompt_down");
        const cast_prompt = try m.function("glm_exl3_slot_f32_to_bf16");
        try prompt_gu.allowDynamicShared(43008);
        try prompt_gu_shared.allowDynamicShared(33792);
        try prompt_down.allowDynamicShared(43008);
        return .{ .module = m, .rot = rot, .rot_shared = rot_shared, .decode_gu = decode_gu, .decode_gu_shared = decode_gu_shared, .gateup = gateup, .decode_down = decode_down, .prompt_gu = prompt_gu, .prompt_gu_shared = prompt_gu_shared, .prompt_down = prompt_down, .cast_prompt = cast_prompt };
    }

    pub fn unload(k: *Kernels) void {
        k.module.unload();
    }
};

fn cfg(x: usize, y: usize, z: usize, threads: u32, shared: u32) cuda.launch.Config {
    return .{ .grid = .{ .x = @intCast(x), .y = @intCast(y), .z = @intCast(z) }, .block = .{ .x = threads }, .shared = shared };
}

fn iv(v: usize) c_int {
    return @intCast(v);
}

fn launch(f: cuda.Function, config: cuda.launch.Config, stream: cuda.Stream, values: anytype) !void {
    var args: cuda.Args = .{};
    inline for (values) |v| args.add(v);
    try cuda.launch.launch(f, config, stream, &args);
}

const Step = enum { rot, rot_shared, decode_gu, decode_gu_shared, gateup, decode_down, prompt_gu, prompt_gu_shared, prompt_down, cast_prompt };

const DeviceDispatch = struct {
    kernels: *const Kernels,
    stream: cuda.Stream,

    fn emit(d: *DeviceDispatch, comptime step: Step, config: cuda.launch.Config, values: anytype) !void {
        return launch(@field(d.kernels.*, @tagName(step)), config, d.stream, values);
    }
};

pub fn routed(k: *const Kernels, stream: cuda.Stream, e: Experts, plan: cuda.grouped.Plan, s: Scratch, x: u64, stride: usize, picks: u64, rows: usize, prompt: bool, limit: f32, output: u64) !void {
    var dispatch: DeviceDispatch = .{ .kernels = k, .stream = stream };
    return routedWith(&dispatch, e, plan, s, x, stride, picks, rows, prompt, limit, output);
}

fn routedWith(dispatch: anytype, e: Experts, plan: cuda.grouped.Plan, s: Scratch, x: u64, stride: usize, picks: u64, rows: usize, prompt: bool, limit: f32, output: u64) !void {
    try validate(e.dims, e.width, e.count, s.slots);
    if (rows == 0 or rows > (if (prompt) s.capacity else s.decode_capacity)) return error.InvalidMiaRows;
    const tile: usize = if (prompt) 64 else 16;
    const pairs = s.decode_capacity * s.slots;
    const items = cuda.grouped.maxItems(rows * s.slots, e.count + 1, tile);
    const shared = e.shared_suh != 0;
    const xu = if (shared) s.xg else s.xu;
    if (shared) {
        try dispatch.emit(.rot_shared, cfg(rows, e.dims / 128, 1, 32, 0), .{ x, iv(stride), @as(u64, 0), e.shared_suh, e.shared_suh, s.xg, s.xg, iv(e.dims), @as(c_int, 1) });
    } else {
        try dispatch.emit(.rot, cfg(rows * s.slots, e.dims / 128, 2, 32, 0), .{ x, iv(stride), picks, e.gate_suh, e.up_suh, s.xg, s.xu, iv(e.dims), iv(s.slots) });
    }
    if (prompt) {
        const gu_cfg = cfg(e.width / 128, items, 1, 256, if (shared) 33792 else 43008);
        if (shared) {
            try dispatch.emit(.prompt_gu_shared, gu_cfg, .{ s.xg, xu, e.gate_trellis, e.up_trellis, plan.items, plan.counts, plan.members, e.gate_svh, e.up_svh, e.down_suh, s.xd, iv(e.dims), iv(e.width), iv(e.count), limit, iv(s.slots), @as(u64, 0) });
        } else {
            try dispatch.emit(.prompt_gu, gu_cfg, .{ s.xg, xu, e.gate_trellis, e.up_trellis, plan.items, plan.counts, plan.members, e.gate_svh, e.up_svh, e.down_suh, s.xd, iv(e.dims), iv(e.width), iv(e.count), limit, iv(s.slots), @as(u64, 0) });
        }
        try dispatch.emit(.prompt_down, cfg(e.dims / 256, items, 1, 256, 43008), .{ s.xd, e.down_trellis, plan.items, plan.counts, plan.members, e.down_svh, s.work, iv(e.width), iv(e.dims), iv(e.count), @as(u64, 0) });
        return dispatch.emit(.cast_prompt, cfg((rows * s.slots * e.dims + 255) / 256, 1, 1, 256, 0), .{ s.work, output, iv(rows), iv(s.slots), iv(e.dims) });
    }
    const gu_cfg = cfg(items, e.width / 128, 8, 128, 0);
    if (shared) {
        try dispatch.emit(.decode_gu_shared, gu_cfg, .{ s.xg, xu, e.gate_trellis, e.up_trellis, plan.items, plan.counts, plan.members, s.work, iv(e.dims), iv(e.width), iv(pairs), @as(c_int, 4), iv(e.count), iv(s.slots), e.gate_svh, e.up_svh, e.down_suh, s.xd, limit, @as(u64, 0) });
    } else {
        try dispatch.emit(.decode_gu, gu_cfg, .{ s.xg, xu, e.gate_trellis, e.up_trellis, plan.items, plan.counts, plan.members, s.work, iv(e.dims), iv(e.width), iv(pairs), @as(c_int, 4), iv(e.count), iv(s.slots), e.gate_svh, e.up_svh, e.down_suh, s.xd, limit, @as(u64, 0) });
    }
    try dispatch.emit(.gateup, cfg(rows * s.slots, e.width / 128, 1, 32, 0), .{ s.work, picks, e.gate_svh, e.up_svh, e.down_suh, s.xd, iv(pairs), iv(e.width), @as(c_int, 4), iv(s.slots), limit });
    try dispatch.emit(.decode_down, cfg(items, e.dims / 128, 1, 128, 0), .{ s.xd, s.xd, e.down_trellis, e.down_trellis, plan.items, plan.counts, plan.members, s.work, iv(e.width), iv(e.dims), iv(pairs), @as(c_int, 1), iv(e.count), iv(s.slots), e.down_svh, e.down_svh, e.down_svh, output, @as(f32, 0), @as(u64, 0) });
}

test "mia geometry and scratch layout reject unsupported routed shapes" {
    try validate(4096, 1024, 288, 9);
    try std.testing.expectError(error.UnsupportedMiaGeometry, validate(4096, 768, 288, 9));
    try std.testing.expectError(error.UnsupportedMiaGeometry, validate(4096, 1024, 288, 1));
    const s = try Scratch.fromBase(4096, 2, 1, 9, 4096, 1024);
    try std.testing.expectEqual(@as(u64, 4096 + 18 * 4096 * 2), s.xu);
    try std.testing.expectEqual(@as(u64, s.xu + 18 * 4096 * 2), s.xd);
    try std.testing.expectEqual(@as(u64, s.xd + 18 * 1024 * 2), s.work);
    const small_end = 4096 + try Scratch.bytes(2, 1, 9, 4096, 1024);
    try std.testing.expectEqual(@as(u64, small_end), s.work + 9 * 2 * 4 * 4096 * 4);
    const prompt = try Scratch.fromBase(4096, 2048, 32, 9, 4096, 1024);
    const prompt_end = 4096 + try Scratch.bytes(2048, 32, 9, 4096, 1024);
    try std.testing.expectEqual(@as(u64, prompt_end), prompt.work + 2048 * 9 * 4096 * 4);
}

test "mia launch trace keeps decode epilogue and prompt output contract" {
    const Record = struct { step: Step, argc: usize, grid: cuda.launch.Config, arg0: ?u64, arg1: ?u64, arg2: ?u64, arg3: ?u64, arg4: ?u64, arg5: ?u64, arg6: ?u64, arg8: ?u64, arg11: ?u64, size14: u16 };
    const Trace = struct {
        records: [4]Record = undefined,
        len: usize = 0,

        fn emit(t: *@This(), comptime step: Step, config: cuda.launch.Config, values: anytype) !void {
            var args: cuda.Args = .{};
            inline for (values) |v| args.add(v);
            t.records[t.len] = .{ .step = step, .argc = args.count, .grid = config, .arg0 = args.integer(0), .arg1 = args.integer(1), .arg2 = args.integer(2), .arg3 = args.integer(3), .arg4 = args.integer(4), .arg5 = args.integer(5), .arg6 = args.integer(6), .arg8 = args.integer(8), .arg11 = args.integer(11), .size14 = if (args.count > 14) args.sizes[14] else 0 };
            t.len += 1;
        }
    };
    const ex: Experts = .{ .gate_trellis = 1, .up_trellis = 2, .down_trellis = 3, .gate_suh = 4, .up_suh = 5, .down_suh = 6, .gate_svh = 7, .up_svh = 8, .down_svh = 9, .dims = 4096, .width = 1024, .count = 288 };
    const plan: cuda.grouped.Plan = .{ .items = 11, .counts = 12, .members = 13, .rank = 14, .hist = 15 };
    const scratch = try Scratch.fromBase(8192, 64, 32, 9, 4096, 1024);
    var decode: Trace = .{};
    try routedWith(&decode, ex, plan, scratch, 21, 4096, 22, 8, false, 10, 23);
    try std.testing.expectEqual(@as(usize, 4), decode.len);
    try std.testing.expectEqualSlices(Step, &.{ .rot, .decode_gu, .gateup, .decode_down }, &.{ decode.records[0].step, decode.records[1].step, decode.records[2].step, decode.records[3].step });
    try std.testing.expectEqual(@as(usize, 20), decode.records[1].argc);
    try std.testing.expectEqual(@as(?u64, 4), decode.records[1].arg11);
    try std.testing.expectEqual(@as(usize, 11), decode.records[2].argc);
    try std.testing.expectEqual(@as(?u64, 4), decode.records[2].arg8);
    try std.testing.expectEqual(@as(usize, 20), decode.records[3].argc);
    try std.testing.expectEqual(@as(?u64, 1), decode.records[3].arg11);
    try std.testing.expectEqual(@as(u32, 76), decode.records[1].grid.grid.x);
    try std.testing.expectEqual(@as(u32, 72), decode.records[2].grid.grid.x);
    try std.testing.expectEqual(@as(u32, 72), decode.records[0].grid.grid.x);
    try std.testing.expectEqual(@as(u32, 2), decode.records[0].grid.grid.z);
    try std.testing.expectEqual(@as(?u64, 22), decode.records[0].arg2);
    try std.testing.expectEqual(@as(?u64, 4), decode.records[0].arg3);
    try std.testing.expectEqual(@as(?u64, 5), decode.records[0].arg4);
    var prompt: Trace = .{};
    try routedWith(&prompt, ex, plan, scratch, 21, 4096, 22, 64, true, 10, 23);
    try std.testing.expectEqual(@as(usize, 4), prompt.len);
    try std.testing.expectEqualSlices(Step, &.{ .rot, .prompt_gu, .prompt_down, .cast_prompt }, &.{ prompt.records[0].step, prompt.records[1].step, prompt.records[2].step, prompt.records[3].step });
    try std.testing.expectEqual(@as(usize, 17), prompt.records[1].argc);
    try std.testing.expectEqual(@as(usize, 11), prompt.records[2].argc);
    try std.testing.expectEqual(@as(usize, 5), prompt.records[3].argc);
    try std.testing.expectEqual(@as(u32, 43008), prompt.records[1].grid.shared);
    try std.testing.expectEqual(@as(u32, 298), prompt.records[1].grid.grid.y);
    try std.testing.expectEqual(@as(u16, 4), prompt.records[1].size14);
    var shared_ex = ex;
    shared_ex.shared_suh = 31;
    var shared_decode: Trace = .{};
    try routedWith(&shared_decode, shared_ex, plan, scratch, 21, 4096, 22, 8, false, 10, 23);
    try std.testing.expectEqualSlices(Step, &.{ .rot_shared, .decode_gu_shared, .gateup, .decode_down }, &.{ shared_decode.records[0].step, shared_decode.records[1].step, shared_decode.records[2].step, shared_decode.records[3].step });
    try std.testing.expectEqual(@as(u32, 8), shared_decode.records[0].grid.grid.x);
    try std.testing.expectEqual(@as(u32, 1), shared_decode.records[0].grid.grid.z);
    try std.testing.expectEqual(@as(?u64, 0), shared_decode.records[0].arg2);
    try std.testing.expectEqual(@as(?u64, 31), shared_decode.records[0].arg3);
    try std.testing.expectEqual(@as(?u64, 31), shared_decode.records[0].arg4);
    try std.testing.expectEqual(@as(?u64, scratch.xg), shared_decode.records[0].arg5);
    try std.testing.expectEqual(@as(?u64, scratch.xg), shared_decode.records[0].arg6);
    try std.testing.expectEqual(@as(?u64, scratch.xg), shared_decode.records[1].arg0);
    try std.testing.expectEqual(@as(?u64, scratch.xg), shared_decode.records[1].arg1);
    try std.testing.expectEqual(@as(u32, 76), shared_decode.records[1].grid.grid.x);
    var shared_prompt: Trace = .{};
    try routedWith(&shared_prompt, shared_ex, plan, scratch, 21, 4096, 22, 64, true, 10, 23);
    try std.testing.expectEqualSlices(Step, &.{ .rot_shared, .prompt_gu_shared, .prompt_down, .cast_prompt }, &.{ shared_prompt.records[0].step, shared_prompt.records[1].step, shared_prompt.records[2].step, shared_prompt.records[3].step });
    try std.testing.expectEqual(@as(u32, 64), shared_prompt.records[0].grid.grid.x);
    try std.testing.expectEqual(@as(u32, 1), shared_prompt.records[0].grid.grid.z);
    try std.testing.expectEqual(@as(?u64, scratch.xg), shared_prompt.records[1].arg0);
    try std.testing.expectEqual(@as(?u64, scratch.xg), shared_prompt.records[1].arg1);
    try std.testing.expectEqual(@as(u32, 33792), shared_prompt.records[1].grid.shared);
    try std.testing.expectEqual(@as(u32, 43008), shared_prompt.records[2].grid.shared);
    try std.testing.expectEqual(@as(u32, 298), shared_prompt.records[1].grid.grid.y);
}
