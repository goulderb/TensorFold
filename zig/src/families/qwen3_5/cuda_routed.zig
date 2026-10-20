//! Qwen3.5 MoE's routed experts on CUDA (qwen3_5_moe): the router's top-k, the grouped plan, the experts and combine.
const cuda = @import("cuda");
const E = @import("cuda_engine.zig");

const ProjectedOffsets = struct { features: usize, picks: usize, weights: usize, scores: usize, pairs: usize };

fn projectedOffsets(row: usize, hidden: usize, top_k: usize, experts: usize) ProjectedOffsets {
    return .{
        .features = row * hidden * 2,
        .picks = row * top_k * 4,
        .weights = row * (top_k + 1) * 4,
        .scores = row * (experts + 1) * 4,
        .pairs = row * top_k * hidden * 4,
    };
}

/// The routed rows' scratch: logits, picks and weights a row, the plan, the pairs' activations and sums.
pub const Scratch = struct {
    logits: u64 = 0,
    picks: u64 = 0,
    wts: u64 = 0,
    members: u64 = 0,
    items: u64 = 0,
    counts: u64 = 0,
    rank: u64 = 0,
    hist: u64 = 0,
    act: u64 = 0,
    y: u64 = 0,
    shared: u64 = 0,

    /// Carves it from `t` (the engine's scratch taker) for `most` rows.
    pub fn carve(t: anytype, g: E.Geometry, most: usize) !Scratch {
        const pairs = most * g.slots();
        const count = g.experts + 1;
        return .{
            .logits = try t.alloc(most * count * 4),
            .picks = try t.alloc(pairs * 4),
            .wts = try t.alloc(pairs * 4),
            .members = try t.alloc(pairs * 4),
            .items = try t.alloc(3 * cuda.grouped.maxItems(pairs, count, 16) * 4),
            .counts = try t.alloc(2 * 4),
            .rank = try t.alloc(pairs * 4),
            .hist = try t.alloc(((pairs + 1023) / 1024) * count * 4),
            .act = try t.alloc(pairs * g.intermediate * 2),
            .y = try t.alloc(pairs * g.hidden * 4),
            .shared = try t.alloc(most * g.hidden * 2),
        };
    }
};

/// Routed experts over `total` normed rows at x into out; `wide` rows take the prefill kernels, else decode's.
pub fn rows(e: *E.Engine, router: u64, layer: cuda.experts.Layer, shared: @import("cuda_layers.zig").SharedExpert, x: u64, out: u64, total: usize, wide: bool, expert_up_n8: bool, expert_up_split_m: bool) !void {
    try validateSplitMDispatch(wide, expert_up_n8, expert_up_split_m);
    if (layer.nvfp4_up_layout == .slice_major_v1 and (wide or !expert_up_n8 or !expert_up_split_m or total == 0 or total > E.hf_prefill_rows)) return error.IncompatibleExpertUpLayout;
    const g = e.g;
    const s = e.routed;
    const slots = if (shared == .projected) g.top_k else g.slots();
    const count = if (shared == .projected) g.experts else g.experts + 1;
    const pairs = total * slots;
    const tile: usize = if (wide) 64 else 16;
    const items = cuda.grouped.maxItems(pairs, count, tile);
    const plan: cuda.grouped.Plan = .{ .members = s.members, .items = s.items, .counts = s.counts, .rank = s.rank, .hist = s.hist };
    const grouped: cuda.experts.Grouped = .{ .affine4 = &e.experts, .affine3 = &e.experts3, .nvfp4 = if (e.modelopt) |*m| &m.nvfp4_experts else null };
    if (shared == .projected) {
        for (0..total) |row| {
            const at = projectedOffsets(row, g.hidden, g.top_k, g.experts);
            try e.modelopt.?.router.route(e.stream, x + at.features, router, router + g.experts * g.hidden * 2, s.picks + at.picks, s.wts + at.weights, s.logits + at.scores);
        }
    } else {
        try e.moe.route(e.stream, x, g.hidden, total, router, s.logits, g.hidden, count);
        try e.moe.pick(e.stream, s.logits, s.picks, s.wts, total, g.experts, g.top_k);
    }
    try e.mark(.route);
    try e.plan.route(e.stream, s.picks, pairs, count, tile, plan);
    try e.mark(.expert_plan);
    const rows_in: cuda.experts.Rows = .{ .x = x, .stride = g.hidden, .slots = slots };
    const act_in: cuda.experts.Rows = .{ .x = s.act, .stride = g.intermediate };
    if (wide) {
        try grouped.prompt(e.stream, .up, rows_in, layer, plan, s.act, items, -1);
        try e.mark(.expert_up);
        try grouped.prompt(e.stream, .down_bf16, act_in, layer, plan, s.y, items, -1);
    } else {
        if (expert_up_split_m) {
            try e.modelopt.?.nvfp4_experts.runSplitMUp(e.stream, rows_in, layer, plan, s.act, items, -1);
        } else if (expert_up_n8) {
            try e.modelopt.?.nvfp4_experts.runN8Up(e.stream, rows_in, layer, plan, s.act, items, -1);
        } else try grouped.decode(e.stream, .up, rows_in, layer, plan, s.act, items, -1);
        try e.mark(.expert_up);
        try grouped.decode(e.stream, .down_f32, act_in, layer, plan, s.y, items, -1);
    }
    try e.mark(.expert_down);
    switch (shared) {
        .stacked => try e.moe.combine(e.stream, s.y, !wide, s.wts, out, total, slots, g.hidden),
        .projected => |m| {
            const width = m.gate.outputs();
            try e.proj(m.gate, x, e.gate, total, false);
            try e.proj(m.up, x, e.up, total, false);
            try e.run(e.modelopt.?.shared_act, .{ (total * width + 255) / 256, 1, 1 }, .{ 256, 1, 1 }, .{ e.gate, e.up, e.act, @as(c_longlong, @intCast(total * width)), e.tables.ptr + E.Engine.params_at, @as(c_int, @intCast(width)) });
            try e.proj(m.down, e.act, s.shared, total, false);
            try e.mark(.shared);
            for (0..total) |row| {
                const at = projectedOffsets(row, g.hidden, g.top_k, g.experts);
                try e.modelopt.?.router.combine(e.stream, s.y + at.pairs, s.wts + at.weights, s.shared + at.features, out + at.features, s.picks + at.picks);
            }
        },
    }
    try e.mark(.combine);
}

fn validateSplitMDispatch(wide: bool, n8: bool, split_m: bool) !void {
    if (split_m and (wide or !n8)) return error.UnsupportedSplitMExpertUp;
}

test "projected HF routing keeps compact picks and separate shared weights" {
    const std = @import("std");
    for (0..16) |row| {
        const at = projectedOffsets(row, 2048, 8, 256);
        try std.testing.expectEqual(row * 4096, at.features);
        try std.testing.expectEqual(row * 32, at.picks);
        try std.testing.expectEqual(row * 36, at.weights);
        try std.testing.expectEqual(row * 1028, at.scores);
        try std.testing.expectEqual(row * 65536, at.pairs);
        const next = projectedOffsets(row + 1, 2048, 8, 256);
        try std.testing.expectEqual(at.picks + 8 * 4, next.picks);
        try std.testing.expectEqual(at.weights + 9 * 4, next.weights);
        try std.testing.expect(at.picks + 8 * 4 <= 16 * 9 * 4);
        try std.testing.expect(at.weights + 9 * 4 <= 16 * 9 * 4);
        try std.testing.expect(at.scores + 257 * 4 <= 16 * 257 * 4);
        try std.testing.expect(at.pairs + 8 * 2048 * 4 <= 16 * 9 * 2048 * 4);
    }
    const first = projectedOffsets(0, 2048, 8, 256);
    try std.testing.expectEqual(ProjectedOffsets{ .features = 0, .picks = 0, .weights = 0, .scores = 0, .pairs = 0 }, first);
}

test "split-M dispatch refuses wide or non-N8 scheduling before CUDA" {
    const std = @import("std");
    try validateSplitMDispatch(false, true, true);
    for ([_]bool{ false, true }) |n8| for ([_]bool{ false, true }) |wide| {
        try validateSplitMDispatch(wide, n8, false);
    };
    try std.testing.expectError(error.UnsupportedSplitMExpertUp, rows(undefined, 0, undefined, undefined, 0, 0, 1, false, false, true));
    try std.testing.expectError(error.UnsupportedSplitMExpertUp, rows(undefined, 0, undefined, undefined, 0, 0, 1, true, true, true));
    try std.testing.expectError(error.UnsupportedSplitMExpertUp, rows(undefined, 0, undefined, undefined, 0, 0, 1, true, false, true));
}

test "slice-major routed dispatch refuses incompatible scheduling before router or CUDA access" {
    const std = @import("std");
    const l: cuda.experts.Layer = .{ .format = .nvfp4, .nvfp4_up_layout = .slice_major_v1, .up = 101, .down = 102, .width = 512, .dims = 2048, .experts = 256 };
    try std.testing.expectError(error.IncompatibleExpertUpLayout, rows(undefined, 0, l, undefined, 0, 0, 1, false, true, false));
    try std.testing.expectError(error.IncompatibleExpertUpLayout, rows(undefined, 0, l, undefined, 0, 0, 1, true, false, false));
    try std.testing.expectError(error.IncompatibleExpertUpLayout, rows(undefined, 0, l, undefined, 0, 0, 17, false, true, true));
}
