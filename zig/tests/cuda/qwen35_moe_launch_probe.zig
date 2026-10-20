//! Private eager-forward evidence; the original driver call still performs every launch.
const std = @import("std");
const cuda = @import("cuda");
const qwen = @import("qwen35_cuda");
const Launch = @FieldType(cuda.abi.Api, "cuLaunchKernel");
const Params = ?[*]?*anyopaque;
const Kind = enum { shared_activation, combine };
const Witness = struct {
    rank: usize,
    layer_local: usize,
    kind: Kind,
    function: usize,
    stream: usize,
    grid: [3]u32,
    block: [3]u32,
    shared_bytes: u32,
    argument_count: usize,
    arguments: [6]u64,
    result: cuda.abi.Result,
};
const Target = struct {
    activation: cuda.Function,
    combine: cuda.Function,
    stream: cuda.abi.Stream,
    activation_args: [6]u64,
    combine_args: [5]u64,
    layers: usize,
};
const Capture = struct {
    original: Launch,
    targets: [2]Target,
    witnesses: [80]Witness = undefined,
    used: usize = 0,
    counts: [2]usize = .{ 0, 0 },
    invalid: bool = false,

    fn observe(c: *Capture, function: cuda.abi.Function, grid: [3]u32, block: [3]u32, shared: u32, stream: cuda.abi.Stream, params: Params, extra: Params, result: cuda.abi.Result) void {
        for (c.targets, 0..) |target, rank| {
            if (stream != target.stream) continue;
            const kind: Kind = if (function == target.activation.handle) .shared_activation else if (function == target.combine.handle) .combine else continue;
            if (c.used == c.witnesses.len or params == null or extra != null) {
                c.invalid = true;
                return;
            }
            const count: usize = if (kind == .shared_activation) 6 else 5;
            var args: [6]u64 = @splat(0);
            for (0..count) |i| {
                const ptr: [*]const u8 = @ptrCast(params.?[i] orelse {
                    c.invalid = true;
                    return;
                });
                args[i] = if (kind == .shared_activation and i == 5) std.mem.readInt(u32, ptr[0..4], .little) else std.mem.readInt(u64, ptr[0..8], .little);
            }
            const expected_kind: Kind = if (c.counts[rank] % 2 == 0) .shared_activation else .combine;
            if (kind != expected_kind or c.counts[rank] >= 2 * target.layers or result != cuda.abi.success or shared != 0 or
                !std.mem.eql(u32, &block, &.{ 256, 1, 1 }) or !std.mem.eql(u32, &grid, if (kind == .shared_activation) &.{ 2, 1, 1 } else &.{ 8, 1, 1 })) c.invalid = true;
            if (kind == .shared_activation) {
                if (!std.mem.eql(u64, &args, &target.activation_args)) c.invalid = true;
            } else if (!std.mem.eql(u64, args[0..5], &target.combine_args) or args[5] != 0) c.invalid = true;
            c.witnesses[c.used] = .{ .rank = rank, .layer_local = c.counts[rank] / 2, .kind = kind, .function = @intFromPtr(function), .stream = @intFromPtr(stream), .grid = grid, .block = block, .shared_bytes = shared, .argument_count = count, .arguments = args, .result = result };
            c.used += 1;
            c.counts[rank] += 1;
            return;
        }
    }
};

var active: ?*Capture = null;

fn intercept(f: cuda.abi.Function, gx: c_uint, gy: c_uint, gz: c_uint, bx: c_uint, by: c_uint, bz: c_uint, shared: c_uint, stream: cuda.abi.Stream, args: Params, extra: Params) callconv(.c) cuda.abi.Result {
    const capture = active orelse return 1;
    const result = capture.original(f, gx, gy, gz, bx, by, bz, shared, stream, args, extra);
    capture.observe(f, .{ gx, gy, gz }, .{ bx, by, bz }, shared, stream, args, extra, result);
    return result;
}

pub fn run(s: *qwen.Session, token: u32, journal: anytype) !void {
    const c = s.chain;
    if (active != null or c.partitions.len != 2) return error.InvalidMoeLaunchProbe;
    var capture: Capture = .{ .original = c.driver.api.cuLaunchKernel, .targets = undefined };
    var captured: [2]bool = undefined;
    for (c.partitions, 0..) |part, rank| {
        const e = part.engine;
        if (e.moeMath() != .bf16_boundaries or e.layers.len != 20 or e.g.intermediate != 512) return error.InvalidMoeLaunchProbe;
        const binding = &e.modelopt.?;
        capture.targets[rank] = .{ .activation = binding.shared_act, .combine = binding.router.combine_fn, .stream = e.stream.handle, .activation_args = .{ e.gate, e.up, e.act, 512, e.tables.ptr + qwen.shared_engine.Engine.params_at, 512 }, .combine_args = .{ e.routed.y, e.routed.wts, e.routed.shared, e.r, e.routed.picks }, .layers = e.layers.len };
        captured[rank] = e.captured;
    }
    try s.reset();
    for (c.partitions) |part| part.engine.captured = false;
    defer for (c.partitions, captured) |part, saved| {
        part.engine.captured = saved;
    };
    active = &capture;
    c.driver.api.cuLaunchKernel = intercept;
    {
        defer {
            c.driver.api.cuLaunchKernel = capture.original;
            active = null;
        }
        try s.append(&.{token});
    }
    if (capture.invalid or capture.used != 80 or !std.mem.eql(usize, &capture.counts, &.{ 40, 40 })) return error.MoeLaunchProbeMismatch;
    for (capture.witnesses[0..capture.used]) |witness| try journal.write(.{ .kind = "moe_math_eager_launch", .witness = witness, .instrumented = true, .captured = false, .api_arguments_decoded_by_known_signature = true });
    try journal.write(.{ .kind = "moe_math_eager_complete", .actual_forward_rows = 1, .successful_observed_launches = capture.used, .actual_loaded_math = c.partitions[0].engine.moeMath(), .driver_restored = active == null and c.driver.api.cuLaunchKernel == capture.original, .timing_reported = false });
}

test "eager launch observer forwards every call and checks actual MoE arguments" {
    const Fake = struct {
        var calls: usize = 0;
        var returned: cuda.abi.Result = cuda.abi.success;
        fn launch(_: cuda.abi.Function, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: cuda.abi.Stream, _: Params, _: Params) callconv(.c) cuda.abi.Result {
            calls += 1;
            return returned;
        }
    };
    const function: cuda.Function = .{ .d = undefined, .handle = @ptrFromInt(1) };
    const combine: cuda.Function = .{ .d = undefined, .handle = @ptrFromInt(2) };
    const stream: cuda.abi.Stream = @ptrFromInt(3);
    const target: Target = .{ .activation = function, .combine = combine, .stream = stream, .activation_args = .{ 101, 202, 303, 512, 404, 512 }, .combine_args = .{ 505, 606, 707, 808, 909 }, .layers = 1 };
    var capture: Capture = .{ .original = Fake.launch, .targets = .{ target, target } };
    if (active != null) return error.InvalidMoeLaunchProbe;
    active = &capture;
    defer active = null;
    Fake.calls = 0;
    Fake.returned = cuda.abi.success;
    try std.testing.expectEqual(cuda.abi.success, intercept(@ptrFromInt(99), 7, 1, 1, 128, 1, 1, 16, stream, null, null));
    try std.testing.expectEqual(@as(usize, 1), Fake.calls);
    try std.testing.expectEqual(@as(usize, 0), capture.used);
    var activation: cuda.Args = .{};
    for (target.activation_args[0..5]) |value| activation.add(value);
    activation.add(@as(c_int, 512));
    try std.testing.expectEqual(cuda.abi.success, intercept(function.handle, 2, 1, 1, 256, 1, 1, 0, stream, activation.pointers(), null));
    var combining: cuda.Args = .{};
    for (target.combine_args) |value| combining.add(value);
    try std.testing.expectEqual(cuda.abi.success, intercept(combine.handle, 8, 1, 1, 256, 1, 1, 0, stream, combining.pointers(), null));
    try std.testing.expectEqual(@as(usize, 3), Fake.calls);
    try std.testing.expectEqual(@as(usize, 2), capture.used);
    try std.testing.expect(!capture.invalid);
    Fake.returned = 7;
    try std.testing.expectEqual(@as(cuda.abi.Result, 7), intercept(function.handle, 2, 1, 1, 256, 1, 1, 0, stream, activation.pointers(), null));
    try std.testing.expect(capture.invalid);
}
