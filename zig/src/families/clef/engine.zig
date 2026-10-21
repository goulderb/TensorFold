//! The native Clef decision engine: the shared BF16 Qwen backbone's hidden rows, the host lexical gather and the
//! joint head on one CUDA context, behind the tfclef C ABI.
const std = @import("std");
const qwen = @import("qwen35");
const record = @import("record.zig");
const head = @import("head.zig");
const checkpoint = @import("head_checkpoint.zig");

/// The backbone's share: the full BF16 engine, or only its mapped vocabulary tables for head-only parity runs.
pub const Backbone = union(enum) {
    engine: qwen.engine.Engine.Bf16Budget,
    tables_only: usize,
};

/// Logical device and host allocations at open. Every request reuses them; none allocates.
pub const Budget = struct {
    backbone_steady: usize,
    backbone_load_peak: usize,
    head_weights: usize,
    head_load_stage: usize,
    head_scratch: usize,
    lexical_device: usize,
    host_lexical: usize,
    host_staging: usize,
    host_tables: usize,
    steady: usize,
    load_peak: usize,

    pub fn of(limits: record.Limits, head_bytes: checkpoint.Bytes, backbone: Backbone) !Budget {
        const add = std.math.add;
        const d = head.Dims.of(limits);
        const head_only = backbone == .tables_only;
        const scratch = try head.Scratch.layout(d, head_only);
        const rows = try std.math.mul(usize, d.tokens, head.hidden * 2);
        const staging = try add(usize, scratch.tables.bytes, try std.math.mul(usize, d.options, 2));
        const steady_b, const peak_b, const tables = switch (backbone) {
            .engine => |e| .{ e.steady_device_bytes, e.load_device_peak_bytes, e.host_tables_bytes },
            .tables_only => |t| .{ 0, 0, t },
        };
        const steady = try add(usize, try add(usize, steady_b, head_bytes.device_weights), try add(usize, scratch.bytes, rows));
        const head_load = try add(usize, try add(usize, steady_b, head_bytes.device_weights), head_bytes.load_stage);
        return .{
            .backbone_steady = steady_b,
            .backbone_load_peak = peak_b,
            .head_weights = head_bytes.device_weights,
            .head_load_stage = head_bytes.load_stage,
            .head_scratch = scratch.bytes,
            .lexical_device = rows,
            .host_lexical = rows,
            .host_staging = staging,
            .host_tables = tables,
            .steady = steady,
            .load_peak = @max(steady, @max(peak_b, head_load)),
        };
    }
};

fn pinnedBackbone() qwen.engine.Engine.Bf16Budget {
    // Engine.bf16Budget at 4096 tokens over the pinned checkpoint metadata (docs/clef-bf16.md, its host test).
    return .{
        .weights_bytes = 13_879_009_280,
        .load_stage_bytes = 100_663_296,
        .scratch_bytes = 28_567_552,
        .cache_bytes = 187_403_264,
        .hidden_bytes = 33_554_432,
        .host_input_bytes = 131_072,
        .host_tables_bytes = 4_068_474_880,
        .host_conversion_peak = 201_326_592,
        .steady_device_bytes = 14_128_534_528,
        .load_device_peak_bytes = 14_128_534_528,
    };
}

fn pinnedHead() !checkpoint.Bytes {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const capture = try std.json.parseFromSliceLeaky(std.json.Value, a, @embedFile("fixtures/capture.json"), .{});
    const plan = try checkpoint.build(a, @embedFile("fixtures/joint_head_config.json"), @embedFile("fixtures/joint_head.safetensors.header.json"), @intCast(capture.object.get("file_bytes").?.integer));
    return plan.bytes;
}

test "the 4096-token budget sums its parts with the pinned backbone and head" {
    const limits = try record.Limits.init(4096, 248320);
    const h = try pinnedHead();
    const b = try Budget.of(limits, h, .{ .engine = pinnedBackbone() });
    try std.testing.expectEqual(b.backbone_steady + b.head_weights + b.head_scratch + b.lexical_device, b.steady);
    try std.testing.expectEqual(@as(usize, 4096 * 4096 * 2), b.lexical_device);
    try std.testing.expectEqual(b.lexical_device, b.host_lexical);
    try std.testing.expectEqual(@max(b.steady, b.backbone_steady + b.head_weights + b.head_load_stage), b.load_peak);
    try std.testing.expectEqual(@as(usize, 4_068_474_880), b.host_tables);
    try std.testing.expectEqual((try head.Scratch.layout(head.Dims.of(limits), false)).bytes, b.head_scratch);
    try std.testing.expect(b.steady < 14_700 << 20);
    try std.testing.expectEqual(@as(usize, 14_632_435_712), b.steady);
}

test "a head-only budget carries no backbone and owns its hidden upload rows" {
    const limits = try record.Limits.init(4096, 248320);
    const h = try pinnedHead();
    const full = try Budget.of(limits, h, .{ .engine = pinnedBackbone() });
    const only = try Budget.of(limits, h, .{ .tables_only = 4_068_474_880 });
    try std.testing.expectEqual(@as(usize, 0), only.backbone_steady);
    try std.testing.expectEqual(full.head_scratch + 4096 * 4096 * 2, only.head_scratch);
    try std.testing.expectEqual(only.head_weights + only.head_scratch + only.lexical_device, only.steady);
}

test "budget arithmetic is checked and its maxima follow the token limit" {
    const h = try pinnedHead();
    const small = try Budget.of(try record.Limits.init(64, 248320), h, .{ .engine = pinnedBackbone() });
    const large = try Budget.of(try record.Limits.init(4096, 248320), h, .{ .engine = pinnedBackbone() });
    try std.testing.expect(small.head_scratch < large.head_scratch and small.host_staging < large.host_staging);
    var huge = pinnedBackbone();
    huge.steady_device_bytes = std.math.maxInt(usize);
    try std.testing.expectError(error.Overflow, Budget.of(try record.Limits.init(64, 1), h, .{ .engine = huge }));
}
