//! HF arithmetic on the upstream Geo/Seg ABI. Admission restricts this profile to one row and BF16 KV.
const cuda = @import("cuda");
const layers = @import("cuda_layers.zig");

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

    pub fn resolve(image: cuda.Module) !Router {
        return .{
            .products_fn = try image.function("tf_q35_hf_route_products"),
            .select_fn = try image.function("tf_q35_hf_route_select_parallel"),
            .combine_fn = try image.function("tf_q35_hf_combine"),
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
    pub fn combine(self: Router, stream: cuda.Stream, pairs: u64, weights: u64, shared: u64, out: u64) !void {
        var args: cuda.Args = .{};
        for ([_]u64{ pairs, weights, shared, out }) |address| args.add(address);
        try cuda.launch.launch(self.combine_fn, .{ .grid = .{ .x = 8, .y = 1, .z = 1 }, .block = .{ .x = 256 } }, stream, &args);
    }
};
