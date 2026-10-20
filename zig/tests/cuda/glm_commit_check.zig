//! Weight-free TP2 target commit probe at the checkpoint's actual KDA geometry.
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm_cuda");
const check = @import("check.zig");
const streams = 8;
const pending_rows = 4;
const samples = 5;
const calls = 5;
const guard = 32;
const Shapes = struct {
    layers: usize,
    heads: usize,
    state: usize,
    kv: usize,
    beta: usize,
    conv: usize,
    projection: usize,
    projection_row: usize,
    fn init(layers: usize, heads: usize) Shapes {
        const n = heads * glm.kda.dk;
        const width = 3 * n + 2 * glm.kda.dk + heads;
        return .{
            .layers = layers,
            .heads = heads,
            .state = heads * glm.kda.dv * glm.kda.dk * 4,
            .kv = glm.forward.window_rows * n,
            .beta = glm.forward.window_rows * heads,
            .conv = 9 * n * 2,
            .projection = glm.forward.window_rows * width * 2,
            .projection_row = width,
        };
    }
    fn total(s: Shapes) usize {
        return streams * s.layers * (2 * s.state + s.kv * 10 + s.beta * 4 + s.conv + s.projection) + 4 * guard;
    }
};
const Frame = struct {
    sin: cuda.DeviceBuffer,
    sout: cuda.DeviceBuffer,
    k: cuda.DeviceBuffer,
    v: cuda.DeviceBuffer,
    g: cuda.DeviceBuffer,
    b: cuda.DeviceBuffer,
    conv: cuda.DeviceBuffer,
    projection: cuda.DeviceBuffer,
    fn init(d: *const cuda.Driver, s: Shapes) !Frame {
        var f: Frame = undefined;
        var count: usize = 0;
        errdefer inline for (@typeInfo(Frame).@"struct".field_names, 0..) |name, i| {
            if (i < count) @field(f, name).free();
        };
        const layers = streams * s.layers;
        inline for (@typeInfo(Frame).@"struct".field_names, .{
            layers * s.state,  layers * s.state + 2 * guard, layers * s.kv * 4,           layers * s.kv * 2,
            layers * s.kv * 4, layers * s.beta * 4,          layers * s.conv + 2 * guard, layers * s.projection,
        }) |name, bytes| {
            @field(f, name) = try cuda.DeviceBuffer.alloc(d, bytes);
            count += 1;
        }
        return f;
    }
    fn deinit(f: *Frame) void {
        inline for (@typeInfo(Frame).@"struct".field_names) |name| @field(f, name).free();
    }
    fn output(f: *const Frame) u64 {
        return f.sout.ptr + guard;
    }
    fn history(f: *const Frame) u64 {
        return f.conv.ptr + guard;
    }
};

fn fill(d: *const cuda.Driver, stream: cuda.Stream, ptr: u64, bytes: usize, value: u32) !void {
    try d.check(d.api.cuMemsetD32Async(ptr, value, bytes / 4, stream.handle), "commit probe fill");
}
fn setup(f: *Frame, gpu: check.Gpu, stream: cuda.Stream, s: Shapes) !void {
    const total = streams * s.layers;
    for (0..total) |i| {
        const value: f32 = 0.001 * @as(f32, @floatFromInt(i + 1));
        try fill(gpu.d, stream, f.sin.ptr + i * s.state, s.state, @bitCast(value));
        const half: u32 = 0x3e00 + @as(u32, @intCast(i % 31));
        try fill(gpu.d, stream, f.projection.ptr + i * s.projection, s.projection, half | half << 16);
    }
    try fill(gpu.d, stream, f.k.ptr, f.k.len, @bitCast(@as(f32, 0.01)));
    try fill(gpu.d, stream, f.v.ptr, f.v.len, 0x3e4d3e4d);
    try fill(gpu.d, stream, f.g.ptr, f.g.len, @bitCast(@as(f32, 0.98)));
    try fill(gpu.d, stream, f.b.ptr, f.b.len, @bitCast(@as(f32, 0.3)));
    try stream.synchronize();
}
fn reset(f: *Frame, gpu: check.Gpu, stream: cuda.Stream, s: Shapes) !void {
    try f.sout.fill8(0xa5, stream.handle);
    try f.conv.fill8(0xa5, stream.handle);
    for (0..streams * s.layers) |i| {
        const half: u32 = 0x3d00 + @as(u32, @intCast(i % 31));
        try fill(gpu.d, stream, f.history() + i * s.conv, s.conv, half | half << 16);
    }
    try stream.synchronize();
}
fn commit(f: *const Frame, stream: cuda.Stream, replay: cuda.Function, shift: cuda.Function, s: Shapes, keep: usize, batch: bool) !void {
    const groups: usize = if (batch) 1 else streams;
    const layers = s.layers * (if (batch) @as(usize, streams) else 1);
    for (0..groups) |group| {
        const at = group * layers;
        if (keep < pending_rows) {
            var args: cuda.Args = .{};
            args.add(@as(c_int, @intCast(s.heads)));
            args.add(f.sin.ptr + at * s.state);
            args.add(s.state / 4);
            args.add(f.k.ptr + at * s.kv * 4);
            args.add(f.v.ptr + at * s.kv * 2);
            args.add(f.g.ptr + at * s.kv * 4);
            args.add(f.b.ptr + at * s.beta * 4);
            args.add(s.kv);
            args.add(s.beta);
            args.add(@as(c_int, @intCast(keep)));
            args.add(f.output() + at * s.state);
            try cuda.launch.launch(replay, .{ .grid = .{ .x = @intCast(layers * s.heads) }, .block = .{ .x = 1024 } }, stream, &args);
        }
        var args: cuda.Args = .{};
        args.add(f.history() + at * s.conv);
        args.add(f.projection.ptr + at * s.projection);
        inline for (.{ keep, s.conv / 2, s.projection / 2, s.projection_row, 3 * s.heads * glm.kda.dk }) |value| args.add(@as(i64, @intCast(value)));
        try cuda.launch.launch(shift, .{ .grid = .{ .x = @intCast(layers), .y = @intCast((3 * s.heads * glm.kda.dk + 255) / 256) }, .block = .{ .x = 256 } }, stream, &args);
    }
}

fn download(gpu: check.Gpu, ptr: u64, out: []u8) !void {
    const view: cuda.DeviceBuffer = .{ .d = gpu.d, .ptr = ptr, .len = out.len };
    try view.download(0, out);
}
fn compare(gpu: check.Gpu, ptr: u64, want: []const u8, scratch: []u8, name: []const u8) !void {
    var at: usize = 0;
    while (at < want.len) {
        const n = @min(scratch.len, want.len - at);
        try download(gpu, ptr + at, scratch[0..n]);
        try check.sameBytes(name, scratch[0..n], want[at..][0..n]);
        at += n;
    }
}
fn checkGuards(gpu: check.Gpu, f: *Frame) !void {
    var bytes: [guard]u8 = undefined;
    for ([_]cuda.DeviceBuffer{ f.sout, f.conv }) |b| for ([_]u64{ b.ptr, b.ptr + b.len - guard }) |ptr| {
        try download(gpu, ptr, &bytes);
        for (bytes) |value| try check.expect(value == 0xa5, "commit probe state/history guard intact", .{});
    };
}

pub fn run(gpu: check.Gpu, config_path: []const u8) !void {
    const a = gpu.gpa;
    const raw = try std.Io.Dir.cwd().readFileAlloc(gpu.io, config_path, a, .limited(1 << 20));
    defer a.free(raw);
    const cfg = try glm.config.parse(a, raw);
    const share = try glm.shard.Share.of(&cfg, 0, 2);
    const s = Shapes.init(cfg.countKind(.kda), share.kda.len());
    try check.expect(s.layers > 0 and s.layers <= 45 and (s.heads == 16 or s.heads == 32), "commit checkpoint geometry bounded for TP2", .{});
    try check.expect(s.total() < 2 * (@as(usize, 1) << 30), "commit fixture device allocations below 2 GiB", .{});
    const room = try gpu.ctx.memInfo();
    try check.expect(room.free > s.total() + (1536 << 20), "commit fixture preserves measured 1536 MiB device headroom", .{});
    const host_bytes = streams * s.layers * (s.state + s.conv) + (4 << 20);
    std.debug.print("RESULT glm-commit geometry: checkpoint_kda_layers={d} tp2_local_heads={d} streams={d} pending_rows={d} replay_stride_rows={d} device_bytes={d} host_check_bytes={d} measured_free_bytes={d}\n", .{ s.layers, s.heads, streams, pending_rows, glm.forward.window_rows, s.total(), host_bytes, room.free });
    var stream = try cuda.Stream.init(gpu.d, false);
    defer stream.deinit();
    var module = try cuda.Module.load(gpu.d, cuda.kernels.glm_kda);
    defer module.unload();
    var glue = try cuda.Module.load(gpu.d, cuda.kernels.glm_glue);
    defer glue.unload();
    const replay = try module.function("tf_glm_kda_replay_layers");
    const shift = try glue.function("tf_glm_conv_shift");
    var f = try Frame.init(gpu.d, s);
    defer f.deinit();
    defer stream.synchronize() catch |err| {
        std.debug.print("FAIL commit probe drain before buffer release: {t}\n", .{err});
        std.process.exit(1);
    };
    try setup(&f, gpu, stream, s);
    const expected_state = try a.alloc(u8, streams * s.layers * s.state);
    defer a.free(expected_state);
    const expected_history = try a.alloc(u8, streams * s.layers * s.conv);
    defer a.free(expected_history);
    const scratch = try a.alloc(u8, 4 << 20);
    defer a.free(scratch);
    for ([_]usize{ 1, 2, 3, 4 }) |keep| {
        try reset(&f, gpu, stream, s);
        try commit(&f, stream, replay, shift, s, keep, false);
        try stream.synchronize();
        try download(gpu, f.output(), expected_state);
        try download(gpu, f.history(), expected_history);
        if (keep < pending_rows) for (0..streams * s.layers * s.heads) |head| {
            const offset = head * glm.kda.dv * glm.kda.dk * 4;
            const first: f32 = @bitCast(std.mem.readInt(u32, expected_state[offset..][0..4], .little));
            const last_at = offset + glm.kda.dv * glm.kda.dk * 4 - 4;
            const last: f32 = @bitCast(std.mem.readInt(u32, expected_state[last_at..][0..4], .little));
            try check.expect(first > 0 and last > 0 and std.math.isFinite(first) and std.math.isFinite(last), "commit replay writes every layer/head state", .{});
        };
        for (0..streams * s.layers) |layer| {
            const expected: u16 = 0x3e00 + @as(u16, @intCast(layer % 31));
            const at = layer * s.conv + s.conv - 2;
            try check.expect(std.mem.readInt(u16, expected_history[at..][0..2], .little) == expected, "commit shift writes every layer's final history row", .{});
        }
        try checkGuards(gpu, &f);
        try reset(&f, gpu, stream, s);
        try commit(&f, stream, replay, shift, s, keep, true);
        try stream.synchronize();
        try compare(gpu, f.output(), expected_state, scratch, "batch target replay exact state");
        try compare(gpu, f.history(), expected_history, scratch, "batch target commit exact convolution history");
        try checkGuards(gpu, &f);
        check.pass("glm-commit keep={d}: old eight launches and one batch grid have exact state/history; full keep skips replay", .{keep});
        var old: [samples]f64 = undefined;
        var batch: [samples]f64 = undefined;
        for (0..samples) |sample| for (0..2) |arm| {
            const is_packed = (sample + arm) % 2 == 1;
            try reset(&f, gpu, stream, s);
            const begin = check.now(gpu.io);
            for (0..calls) |_| try commit(&f, stream, replay, shift, s, keep, is_packed);
            try stream.synchronize();
            const elapsed = @as(f64, @floatFromInt(check.now(gpu.io) - begin)) / 1e6 / calls;
            (if (is_packed) &batch else &old)[sample] = elapsed;
            try checkGuards(gpu, &f);
        };
        const old_min = std.mem.min(f64, &old);
        const old_max = std.mem.max(f64, &old);
        const packed_min = std.mem.min(f64, &batch);
        const packed_max = std.mem.max(f64, &batch);
        var sorted_old = old;
        var sorted_batch = batch;
        std.debug.print("RESULT glm-commit keep={d} pending={d} layers={d} local_heads={d} samples={d} calls={d}: old_ms={d:.6} range={d:.6}..{d:.6} packed_ms={d:.6} range={d:.6}..{d:.6} raw_old_ms={any} raw_packed_ms={any}\n", .{ keep, pending_rows, s.layers, s.heads, samples, calls, check.median(&sorted_old), old_min, old_max, check.median(&sorted_batch), packed_min, packed_max, old, batch });
    }
}
