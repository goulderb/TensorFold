//! Host observations of the joint head over a recording CUDA driver: launch order, write extents, no allocation.
const std = @import("std");
const cuda = @import("cuda");
const abi = cuda.abi;
const head = @import("head.zig");
const record = @import("record.zig");
const checkpoint = @import("head_checkpoint.zig");

/// A kernel parameter's C type: device address, int, int64 or float.
const P = enum { p, i, l, f };

const Launch = struct { f: usize, grid: [3]u32, block: u32, shared: u32, args: [16]u64 };

const clef_base = 1;
const vit_base = 20;
const lane_bf16 = 100;
const lane_f32 = 200;
const reduce_base = 300;
const pack_base = 400;

fn signature(f: usize) []const P {
    if (f >= lane_bf16 and f < reduce_base) return &.{ .p, .p, .p, .f, .p, .p, .i, .i, .i, .i, .i, .i, .i };
    if (f >= reduce_base and f < pack_base) return &.{ .p, .p, .l, .i, .f };
    if (f >= pack_base) return &.{ .p, .p, .i, .i, .l };
    return switch (f) {
        clef_base + 0 => &.{ .p, .p, .p, .p, .i, .f },
        clef_base + 1 => &.{ .p, .i, .p, .p },
        clef_base + 2 => &.{ .p, .p, .p, .p, .p, .i },
        clef_base + 3 => &.{ .p, .p, .p, .p, .i, .i, .f },
        clef_base + 4 => &.{ .p, .p, .p, .p, .i, .f },
        clef_base + 5 => &.{ .p, .p, .p, .p, .p, .p, .i },
        clef_base + 6 => &.{ .p, .p, .p, .i, .f },
        clef_base + 7 => &.{ .p, .p, .p, .p, .i },
        clef_base + 8 => &.{ .p, .p, .p, .p, .p, .i, .f },
        clef_base + 9 => &.{ .p, .p, .p, .p, .p, .p, .p, .i, .f },
        vit_base + 0 => &.{ .p, .p, .p, .p, .i, .f },
        vit_base + 1 => &.{ .p, .p, .p, .l, .i },
        vit_base + 2 => &.{ .p, .p, .p, .i },
        else => &.{},
    };
}

const Fake = struct {
    var launches: [4096]Launch = undefined;
    var count: usize = 0;
    var allocations: usize = 0;
    var syncs: usize = 0;
    var copies: usize = 0;
    var copy_dst: [8]u64 = undefined;
    var copy_len: [8]usize = undefined;

    fn reset() void {
        count = 0;
        allocations = 0;
        syncs = 0;
        copies = 0;
    }

    fn launch(f: abi.Function, gx: c_uint, gy: c_uint, gz: c_uint, bx: c_uint, _: c_uint, _: c_uint, shared: c_uint, _: abi.Stream, params: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) abi.Result {
        const id = @intFromPtr(f);
        const sig = signature(id);
        if (sig.len == 0) return 1;
        var l: Launch = .{ .f = id, .grid = .{ gx, gy, gz }, .block = bx, .shared = shared, .args = @splat(0) };
        for (sig, 0..) |kind, i| {
            const ptr = params.?[i].?;
            l.args[i] = switch (kind) {
                .p, .l => @as(*align(1) const u64, @ptrCast(ptr)).*,
                .i, .f => @as(*align(1) const u32, @ptrCast(ptr)).*,
            };
        }
        launches[count] = l;
        count += 1;
        return 0;
    }
    fn alloc(_: *abi.DevicePtr, _: usize) callconv(.c) abi.Result {
        allocations += 1;
        return 2;
    }
    fn hostAlloc(_: *?*anyopaque, _: usize, _: c_uint) callconv(.c) abi.Result {
        allocations += 1;
        return 2;
    }
    fn sync(_: abi.Stream) callconv(.c) abi.Result {
        syncs += 1;
        return 0;
    }
    fn contextSync() callconv(.c) abi.Result {
        syncs += 1;
        return 0;
    }
    fn copy(dst: u64, _: ?*const anyopaque, n: usize) callconv(.c) abi.Result {
        copy_dst[copies] = dst;
        copy_len[copies] = n;
        copies += 1;
        return 0;
    }

    fn driver() cuda.Driver {
        var d: cuda.Driver = undefined;
        d.api.cuLaunchKernel = launch;
        d.api.cuMemAlloc_v2 = alloc;
        d.api.cuMemHostAlloc = hostAlloc;
        d.api.cuStreamSynchronize = sync;
        d.api.cuCtxSynchronize = contextSync;
        d.api.cuMemcpyHtoD_v2 = copy;
        return d;
    }
};

fn function(d: *const cuda.Driver, id: usize) cuda.Function {
    return .{ .d = d, .handle = @ptrFromInt(id) };
}

fn kernels(d: *const cuda.Driver) head.Kernels {
    var k: head.Kernels = undefined;
    inline for (@typeInfo(head.Fns).@"struct".field_names, 0..) |name, i| @field(k.fns, name) = function(d, clef_base + i);
    k.vit = .{ .mod = undefined, .layer_norm_fn = function(d, vit_base), .pointwise_fn = function(d, vit_base + 1), .bias_fn = function(d, vit_base + 2), .unshuffle_fn = function(d, vit_base + 3), .cast_fn = function(d, vit_base + 4), .position_fn = function(d, vit_base + 5) };
    k.lane = .{ .mod = undefined, .fns = undefined, .f32_fns = undefined, .reduce = .{ function(d, reduce_base), function(d, reduce_base + 1) }, .clusters = false };
    for (&k.lane.fns, &k.lane.f32_fns, 0..) |*a, *b, m| for (a, b, 0..) |*x, *y, t| {
        x.* = function(d, lane_bf16 + m * 7 + t);
        y.* = function(d, lane_f32 + m * 7 + t);
    };
    return k;
}

const Region = struct { name: []const u8, at: u64, len: usize };

const weights_base: u64 = 0x1_0000_0000;
const scratch_base: u64 = 0x2_0000_0000;
const hidden_base: u64 = 0x3_0000_0000;
const lexical_base: u64 = 0x4_0000_0000;

const Sample = struct {
    ids: [30]u32 = @splat(3),
    questions: [2]record.C.Question = .{
        .{ .kind = 0, .question = .{ .start = 1, .end = 3 }, .first_option = 0, .option_count = 2 },
        .{ .kind = 2, .question = .{ .start = 10, .end = 12 }, .first_option = 2, .option_count = 1 },
    },
    options: [3]record.Span = .{ .{ .start = 4, .end = 6 }, .{ .start = 7, .end = 8 }, .{ .start = 13, .end = 16 } },
    c: record.C.Record = undefined,
};

const Run = struct {
    d: cuda.Driver,
    k: head.Kernels,
    plan: checkpoint.Plan,
    scratch: record.Scratch,
    regions: std.ArrayList(Region) = .empty,
    h: head.Head,
    rec: record.Record,
    sample: Sample = .{},

    fn init(r: *Run) !void {
        const a = std.testing.allocator;
        r.* = .{ .d = Fake.driver(), .k = undefined, .plan = undefined, .scratch = undefined, .h = undefined, .rec = undefined };
        r.k = kernels(&r.d);
        var arena: std.heap.ArenaAllocator = .init(a);
        errdefer arena.deinit();
        const capture = try std.json.parseFromSliceLeaky(std.json.Value, arena.allocator(), @embedFile("fixtures/capture.json"), .{});
        r.plan = try checkpoint.build(arena.allocator(), @embedFile("fixtures/joint_head_config.json"), @embedFile("fixtures/joint_head.safetensors.header.json"), @intCast(capture.object.get("file_bytes").?.integer));
        r.plan.arena = arena;
        const limits = try record.Limits.init(64, 100);
        r.scratch = try record.Scratch.alloc(a, limits);
        const dims = head.Dims.of(limits);
        const layout = try head.Scratch.layout(dims, true);
        r.h = .{ .k = &r.k, .w = .{ .base = weights_base, .layout = r.plan.layout }, .scratch = layout, .base = scratch_base, .dims = dims };
        r.sample.c = .{ .ids = &r.sample.ids, .n_ids = 30, .questions = &r.sample.questions, .n_questions = 2, .options = &r.sample.options, .n_options = 3 };
        r.rec = try record.Record.parse(&r.sample.c, limits, r.scratch);
        try r.regions.append(a, .{ .name = "weights", .at = weights_base, .len = r.plan.bytes.device_weights });
        try r.regions.append(a, .{ .name = "hidden", .at = hidden_base, .len = 30 * head.hidden * 2 });
        try r.regions.append(a, .{ .name = "lexical", .at = lexical_base, .len = r.rec.lexical_rows * head.hidden * 2 });
        for (std.enums.values(head.Buf)) |b| try r.regions.append(a, .{ .name = @tagName(b), .at = scratch_base + layout.offsets.get(b), .len = layout.sizes.get(b) });
    }

    fn deinit(r: *Run) void {
        r.regions.deinit(std.testing.allocator);
        r.scratch.free(std.testing.allocator);
        r.plan.deinit();
    }

    fn inputs(r: *Run) head.Inputs {
        return .{ .hidden = hidden_base, .lexical = lexical_base, .rec = &r.rec };
    }

    /// The region that holds [at, at + len); a span that crosses or misses every region fails the test.
    fn within(r: *const Run, at: u64, len: usize) ![]const u8 {
        for (r.regions.items) |g| if (at >= g.at and at + len <= g.at + g.len) return g.name;
        std.debug.print("span {x} + {d} is in no buffer\n", .{ at, len });
        return error.OutOfBuffer;
    }
};

/// The write and the reads of one launch, as (address, bytes).
fn spans(l: Launch, out: *[6][2]u64) usize {
    const a = l.args;
    const gx: u64 = l.grid[0];
    var n: usize = 0;
    const put = struct {
        fn f(o: *[6][2]u64, i: *usize, at: u64, len: u64) void {
            if (at == 0) return;
            o[i.*] = .{ at, len };
            i.* += 1;
        }
    }.f;
    switch (l.f) {
        clef_base + 0 => {
            put(out, &n, a[3], gx * a[4] * 2);
            put(out, &n, a[0], gx * a[4] * 2);
        },
        clef_base + 1 => put(out, &n, a[3], gx * a[1] * 2),
        clef_base + 2 => put(out, &n, a[4], gx * a[5] * 2),
        clef_base + 3 => {
            put(out, &n, a[3], gx * a[5] * 2);
            put(out, &n, a[0], gx * a[5] * 2);
            put(out, &n, a[1], a[4] * a[5] * 2);
            put(out, &n, a[2], a[4] * a[5] * 2);
        },
        clef_base + 4 => put(out, &n, a[3], gx * a[4] * 2),
        clef_base + 5 => put(out, &n, a[5], gx * a[6] * 2),
        clef_base + 6 => put(out, &n, a[2], gx * a[3] * 2),
        clef_base + 7 => put(out, &n, a[3], gx * 2),
        clef_base + 8 => {
            put(out, &n, a[3], gx * 2);
            put(out, &n, a[4], gx * 4 * a[5] * 2);
        },
        clef_base + 9 => put(out, &n, a[6], a[7] * 2),
        vit_base + 0 => {
            put(out, &n, a[3], gx * a[4] * 2);
            put(out, &n, a[0], gx * a[4] * 2);
        },
        vit_base + 1 => {
            put(out, &n, a[2], a[3] * 2);
            put(out, &n, a[0], a[3] * 2);
        },
        vit_base + 2 => {
            put(out, &n, a[2], gx * a[3] * 2);
            put(out, &n, a[0], gx * a[3] * 4);
        },
        reduce_base, reduce_base + 1 => put(out, &n, a[1], a[2] * @as(u64, if (l.f == reduce_base) 2 else 4)),
        else => {
            const m = a[6];
            const cols = a[7];
            const bytes: u64 = if (l.f >= lane_f32) 4 else 2;
            if (a[5] != 0) put(out, &n, a[5], a[9] * m * cols * 4) else put(out, &n, a[4], m * cols * bytes);
            put(out, &n, a[0], m * a[8] * 2);
        },
    }
    return n;
}

fn kernelName(f: usize) []const u8 {
    if (f >= clef_base and f < vit_base) return @typeInfo(head.Fns).@"struct".field_names[f - clef_base];
    return switch (f) {
        vit_base => "vit_layer_norm",
        vit_base + 1 => "vit_pointwise",
        vit_base + 2 => "vit_bias",
        reduce_base, reduce_base + 1 => "lane_reduce",
        else => "lane",
    };
}

test "one record runs the head's launches in torch's order, inside its buffers, without allocating or waiting" {
    var r: Run = undefined;
    try r.init();
    defer r.deinit();
    Fake.reset();
    try r.h.run(.{ .d = &r.d, .handle = @ptrFromInt(9) }, r.inputs());
    try std.testing.expectEqual(@as(usize, 0), Fake.allocations);
    try std.testing.expectEqual(@as(usize, 0), Fake.syncs);

    var counts: std.StringHashMapUnmanaged(usize) = .empty;
    defer counts.deinit(std.testing.allocator);
    var order: std.ArrayList([]const u8) = .empty;
    defer order.deinit(std.testing.allocator);
    for (Fake.launches[0..Fake.count]) |l| {
        const n = kernelName(l.f);
        const got = try counts.getOrPut(std.testing.allocator, n);
        got.value_ptr.* = if (got.found_existing) got.value_ptr.* + 1 else 1;
        if (l.f < vit_base) try order.append(std.testing.allocator, n);
        var s: [6][2]u64 = undefined;
        for (s[0..spans(l, &s)], 0..) |span, i| {
            const region = r.within(span[0], span[1]) catch |e| {
                std.debug.print("{s} span {d}\n", .{ n, i });
                return e;
            };
            if (i == 0) try std.testing.expect(!std.mem.eql(u8, region, "weights") and !std.mem.eql(u8, region, "hidden") and !std.mem.eql(u8, region, "lexical"));
        }
    }
    // JointSchemaHead's graph: 6 bias-free projections, 6 biased Linears in each of 2 routing layers, 10 in each of
    // 4 decoder layers and 2 in the residual scorer; LayerNorms 3 + 3 per routing layer and 3 per decoder layer.
    try std.testing.expectEqual(@as(usize, 6 + 2 * 6 + 4 * 10 + 2), counts.get("lane").?);
    try std.testing.expectEqual(@as(usize, 2 * 6 + 4 * 10 + 2), counts.get("vit_bias").?);
    try std.testing.expectEqual(@as(usize, 2 * 3 + 1 + 4 * 3 + 2), counts.get("vit_layer_norm").?);
    try std.testing.expectEqual(@as(usize, 2 * 3 + 4 * 4 + 1), counts.get("vit_pointwise").?);
    const expected = [_][]const u8{
        "layer_norm_4096", "span_mean",     "span_mean",       "span_mean", "option_queries",
        "attention",       "attention",     "segment_summary", "fields",    "attention",
        "attention",       "attention",     "attention",       "attention", "attention",
        "attention",       "attention",     "normalize",       "normalize", "prior_dot",
        "score_features",  "logit_combine",
    };
    try std.testing.expectEqual(expected.len, order.items.len);
    for (expected, order.items) |want, got| try std.testing.expectEqualStrings(want, got);
}

test "each head kernel covers exactly the record's rows, questions and options" {
    var r: Run = undefined;
    try r.init();
    defer r.deinit();
    Fake.reset();
    try r.h.run(.{ .d = &r.d, .handle = @ptrFromInt(9) }, r.inputs());
    var attention: usize = 0;
    var span_means: usize = 0;
    for (Fake.launches[0..Fake.count]) |l| switch (l.f) {
        clef_base + 0 => try std.testing.expectEqual([3]u32{ 30, 1, 1 }, l.grid),
        clef_base + 1 => {
            try std.testing.expectEqual(@as(u32, if (span_means == 0) 2 else 3), l.grid[0]);
            span_means += 1;
        },
        clef_base + 3 => {
            const lq: u32 = if (attention < 2) 3 else 2;
            const lk: u64 = if (attention < 2) 30 else if (attention % 2 == 0) 2 else 30;
            try std.testing.expectEqual([3]u32{ lq, 16, 1 }, l.grid);
            try std.testing.expectEqual(@as(u32, 64), l.block);
            try std.testing.expectEqual(lk, l.args[4]);
            try std.testing.expectEqual(@as(u64, @as(u32, @bitCast(@as(f32, 0.125)))), l.args[6]);
            attention += 1;
        },
        clef_base + 4 => {
            try std.testing.expectEqual(@as(u32, 2), l.grid[0]);
            try std.testing.expectEqual(@as(u32, 2 * 4), l.shared);
            try std.testing.expectEqual(@as(u64, @as(u32, @bitCast(@as(f32, 32.0)))), l.args[5]);
        },
        clef_base + 8 => try std.testing.expectEqual(@as(u32, 3), l.grid[0]),
        clef_base + 9 => {
            try std.testing.expectEqual(@as(u32, 1), l.grid[0]);
            try std.testing.expectEqual(@as(u64, 3), l.args[7]);
        },
        else => {},
    };
    try std.testing.expectEqual(@as(usize, 10), attention);
}

test "a record beyond the head's extents is refused before any launch" {
    var r: Run = undefined;
    try r.init();
    defer r.deinit();
    Fake.reset();
    var big = r.rec;
    const zeros: [65]u32 = @splat(0);
    big.ids = &zeros;
    try std.testing.expectError(error.InvalidHeadShape, r.h.run(.{ .d = &r.d, .handle = null }, .{ .hidden = hidden_base, .lexical = lexical_base, .rec = &big }));
    try std.testing.expectEqual(@as(usize, 0), Fake.count);
}

test "staged tables carry spans, packed lexical offsets, the option map, ranges and types" {
    var r: Run = undefined;
    try r.init();
    defer r.deinit();
    const t = r.h.scratch.tables;
    const host = try std.testing.allocator.alloc(u8, t.bytes);
    defer std.testing.allocator.free(host);
    head.stageTables(t, r.rec, host);
    const read = struct {
        fn f(comptime T: type, h: []const u8, at: usize, i: usize) T {
            return std.mem.bytesToValue(T, h[at + i * @sizeOf(T) ..][0..@sizeOf(T)]);
        }
    }.f;
    const lex = t.offsets.get(.lexical_spans);
    try std.testing.expectEqual(record.Span{ .start = 0, .end = 2 }, read(record.Span, host, lex, 0));
    try std.testing.expectEqual(record.Span{ .start = 2, .end = 3 }, read(record.Span, host, lex, 1));
    try std.testing.expectEqual(record.Span{ .start = 3, .end = 6 }, read(record.Span, host, lex, 2));
    try std.testing.expectEqual(@as(u32, 1), read(u32, host, t.offsets.get(.option_question), 2));
    try std.testing.expectEqual(head.Tables.Range{ .first = 2, .count = 1 }, read(head.Tables.Range, host, t.offsets.get(.question_options), 1));
    try std.testing.expectEqual(@as(u32, 2), read(u32, host, t.offsets.get(.types), 1));
    try std.testing.expectEqual(record.Span{ .start = 10, .end = 12 }, read(record.Span, host, t.offsets.get(.question_spans), 1));
    try std.testing.expectEqual(record.Span{ .start = 13, .end = 16 }, read(record.Span, host, t.offsets.get(.option_spans), 2));
}

test "weight loading uploads stored tensors in place and packs in_proj as three lanes" {
    var d = Fake.driver();
    Fake.reset();
    var pack: cuda.modelopt.Packer = undefined;
    pack.bf16_lane = function(&d, pack_base);
    const arena: cuda.DeviceBuffer = .{ .d = &d, .ptr = 0x10000, .len = 1 << 24, .borrowed = true };
    const stage: cuda.DeviceBuffer = .{ .d = &d, .ptr = 0x20000000, .len = 1 << 20, .borrowed = true };
    var sink: head.WeightSink = .{ .arena = arena, .stage = stage, .stream = .{ .d = &d, .handle = null }, .pack = &pack };
    var entry: @import("core").safetensors.Entry = .{ .dtype = .bf16, .rank = 2, .shape = .{ 384, 64, 1, 1, 1 }, .begin = 0, .end = 384 * 64 * 2 };
    const bytes = try std.testing.allocator.alloc(u8, 384 * 64 * 2);
    defer std.testing.allocator.free(bytes);
    try sink.tensor(.{ .name = "in_proj_weight", .entry = entry, .placement = .lane_qkv, .device_offset = 4096, .device_bytes = 3 * checkpoint.laneBytes(128, 64) }, bytes);
    try std.testing.expectEqual(@as(usize, 3), Fake.count);
    for (Fake.launches[0..3], 0..) |l, i| {
        try std.testing.expectEqual(stage.ptr + i * 128 * 64 * 2, l.args[0]);
        try std.testing.expectEqual(arena.ptr + 4096 + i * checkpoint.laneBytes(128, 64), l.args[1]);
        try std.testing.expectEqual(@as(u64, 128), l.args[2]);
        try std.testing.expectEqual(@as(u64, 64), l.args[3]);
    }
    try std.testing.expectEqual(stage.ptr, Fake.copy_dst[0]);
    entry.rank = 1;
    entry.shape = .{ 3, 1, 1, 1, 1 };
    entry.end = 6;
    try sink.tensor(.{ .name = "bias", .entry = entry, .placement = .raw, .device_offset = 512, .device_bytes = 6 }, bytes[0..6]);
    try std.testing.expectEqual(arena.ptr + 512, Fake.copy_dst[1]);
    try std.testing.expectEqual(@as(usize, 6), Fake.copy_len[1]);
    try std.testing.expectEqual(@as(usize, 3), Fake.count);
}
