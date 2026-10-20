//! HF arithmetic on the upstream Geo/Seg ABI. Admission restricts this profile to one row and BF16 KV.
const cuda = @import("cuda");
const layers = @import("cuda_layers.zig");

pub const MoeMath = enum(u1) { legacy, bf16_boundaries };

/// ModelOpt attention: HF serial reference, or upstream chunked BF16 kernels.
pub const AttentionMode = enum(u1) { reference, chunked };

/// Partial/merge symbol names that match a mode. Reference stays on HF; chunked reuses qwen35.cu.
pub fn attnKernelNames(mode: AttentionMode) struct { partial: [:0]const u8, merge: [:0]const u8 } {
    return switch (mode) {
        .reference => .{ .partial = "tf_q35_hf_attn_partial", .merge = "tf_q35_hf_attn_merge" },
        .chunked => .{ .partial = "tf_q35_attn_partial", .merge = "tf_q35_attn_merge" },
    };
}

/// Launch geometry for one attention mode. HF reference is serial; chunked matches ordinary qwen35.cu.
pub const AttnLaunch = struct {
    partial_grid: [3]usize,
    partial_block: [3]usize,
    merge_block: usize,
};

pub fn attnLaunch(mode: AttentionMode, q_heads: usize, kv_heads: usize, rows: usize, group: usize, max_chunks: usize) AttnLaunch {
    return switch (mode) {
        .reference => .{ .partial_grid = .{ 1, q_heads, rows }, .partial_block = .{ 256, 1, 1 }, .merge_block = 256 },
        .chunked => .{ .partial_grid = .{ max_chunks, kv_heads, rows }, .partial_block = .{ 32 * group, 1, 1 }, .merge_block = 32 },
    };
}

/// These dimensions are part of the numerical binding, including the reduction order.
pub const Launches = struct {
    norm: usize = 256,
    gdn_pre: usize = 128,
    gdn_post: usize = 128,
    gdn_chain: [3]usize = .{ 128, 1, 1 },
    gdn_chain_grid_y: usize = 1,
    value_rows_per_block: usize = 128,
    head_norm: usize = 256,
    attn_partial: [3]usize = .{ 256, 1, 1 },
    attn_merge: usize = 256,
};

/// Argument order matches Fns. Norm weights, conv weights and dt_bias are FP32.
/// GDN Q/K/Y are FP32; V/beta and Seg.conv are BF16; Seg.rec is FP32.
/// GDN G holds log-decay, exponentiated once by chain. All layer indices are local.
/// Attention partial launches (1,q_heads,rows) and writes only MAXCH-strided slot zero.
/// Merge launches (q_heads,rows,1). The ordinary conv_commit and attn_gate remain bound.
/// Callers that select chunked attention rebind attn_partial/merge from qwen35.cu afterward.
pub fn bind(image: cuda.Module, fns: *layers.Fns) !void {
    inline for (.{ "embed", "norm", "gdn_pre", "gdn_chain", "gdn_post", "head_norm", "rope", "attn_partial", "attn_merge", "mlp_act" }) |name| {
        @field(fns, name) = try image.function("tf_q35_hf_" ++ name);
    }
}

/// One row of 2048 BF16 features, 256 router rows, eight routed FP32 outputs and a BF16 shared output.
pub const Router = struct {
    products_fn: cuda.Function,
    select_fn: cuda.Function,
    combine_fn: cuda.Function,
    math: MoeMath = .legacy,

    pub fn resolve(image: cuda.Module) !Router {
        return resolveWithMath(image, .legacy);
    }

    pub fn resolveWithMath(image: cuda.Module, math: MoeMath) !Router {
        return .{
            .products_fn = try image.function("tf_q35_hf_route_products"),
            .select_fn = try image.function("tf_q35_hf_route_select_parallel"),
            .combine_fn = try image.function(if (math == .bf16_boundaries) "tf_q35_hf_combine_bf16" else "tf_q35_hf_combine"),
            .math = math,
        };
    }

    /// picks has eight i32 entries. weights has nine FP32 entries, with the shared sigmoid last.
    /// scores has 257 FP32 entries for the ordered dot products.
    pub fn route(self: Router, stream: cuda.Stream, x: u64, router: u64, shared_gate: u64, picks: u64, weights: u64, scores: u64) !void {
        var args: cuda.Args = .{};
        for ([_]u64{ x, router, shared_gate, scores }) |address| args.add(address);
        try cuda.launch.launch(self.products_fn, .{ .grid = .{ .x = 257, .y = 1, .z = 1 }, .block = .{ .x = 256 } }, stream, &args);
        var selection: cuda.Args = .{};
        for ([_]u64{ scores, picks, weights }) |address| selection.add(address);
        try cuda.launch.launch(self.select_fn, .{ .grid = .{ .x = 1, .y = 1, .z = 1 }, .block = .{ .x = 256 } }, stream, &selection);
    }

    /// pairs is FP32 [8,2048] in pick order. shared and out are BF16 [2048].
    fn combineArgs(math: MoeMath, pairs: u64, weights: u64, shared: u64, out: u64, picks: u64) cuda.Args {
        var args: cuda.Args = .{};
        for ([_]u64{ pairs, weights, shared, out }) |address| args.add(address);
        if (math == .bf16_boundaries) args.add(picks);
        return args;
    }

    pub fn combine(self: Router, stream: cuda.Stream, pairs: u64, weights: u64, shared: u64, out: u64, picks: u64) !void {
        var args = combineArgs(self.math, pairs, weights, shared, out, picks);
        try cuda.launch.launch(self.combine_fn, .{ .grid = .{ .x = 8, .y = 1, .z = 1 }, .block = .{ .x = 256 } }, stream, &args);
    }
};

test "HF combine keeps the legacy ABI and appends actual picks only for BF16 boundaries" {
    const std = @import("std");
    for ([_]MoeMath{ .legacy, .bf16_boundaries }) |math| {
        const args = Router.combineArgs(math, 101, 202, 303, 404, 505);
        const count: usize = if (math == .legacy) 4 else 5;
        try std.testing.expectEqual(count, args.count);
        try std.testing.expectEqual(count * 8, args.used);
        for ([_]u64{ 101, 202, 303, 404, 505 }, 0..) |value, i| {
            if (i < count) {
                try std.testing.expectEqual(@as(u16, 8), args.sizes[i]);
                try std.testing.expectEqual(value, args.integer(i).?);
            } else try std.testing.expect(args.integer(i) == null);
        }
    }
}

test "attention mode selects HF reference or ordinary chunked kernel names and geometry" {
    const std = @import("std");
    const reference = attnKernelNames(.reference);
    try std.testing.expectEqualStrings("tf_q35_hf_attn_partial", reference.partial);
    try std.testing.expectEqualStrings("tf_q35_hf_attn_merge", reference.merge);
    const chunked = attnKernelNames(.chunked);
    try std.testing.expectEqualStrings("tf_q35_attn_partial", chunked.partial);
    try std.testing.expectEqualStrings("tf_q35_attn_merge", chunked.merge);
    const ref = attnLaunch(.reference, 16, 2, 8, 8, 64);
    try std.testing.expectEqual(@as(usize, 1), ref.partial_grid[0]);
    try std.testing.expectEqual(@as(usize, 16), ref.partial_grid[1]);
    try std.testing.expectEqual(@as(usize, 8), ref.partial_grid[2]);
    try std.testing.expectEqual(@as(usize, 256), ref.partial_block[0]);
    try std.testing.expectEqual(@as(usize, 256), ref.merge_block);
    const ch = attnLaunch(.chunked, 16, 2, 8, 8, 64);
    try std.testing.expectEqual(@as(usize, 64), ch.partial_grid[0]);
    try std.testing.expectEqual(@as(usize, 2), ch.partial_grid[1]);
    try std.testing.expectEqual(@as(usize, 8), ch.partial_grid[2]);
    try std.testing.expectEqual(@as(usize, 256), ch.partial_block[0]);
    try std.testing.expectEqual(@as(usize, 32), ch.merge_block);
    const defaults: struct { mode: AttentionMode = .reference } = .{};
    try std.testing.expectEqual(AttentionMode.reference, defaults.mode);
}
