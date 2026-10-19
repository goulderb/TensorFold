//! Serial Qwen3.5 35B CUDA operations. Every address belongs to the caller's context.
const cuda = @import("cuda");

pub const hidden = 2048;
pub const key_heads = 16;
pub const value_heads = 32;
pub const gdn_dim = 128;
pub const conv_width = 8192;
pub const attention_heads = 16;
pub const kv_heads = 2;
pub const head_dim = 256;
pub const picks = 8;
pub const experts = 256;

pub const Ops = struct {
    image: cuda.Module,
    plan_image: cuda.Module,
    norms: cuda.norms.Norms,
    lane: cuda.qmmf.Lane,
    prompt: cuda.nvfp4.Prompt,
    nvfp4_experts: cuda.experts.Nvfp4Experts,
    expert_up_variant: cuda.experts.Nvfp4Experts.Variant = .baseline,
    pack: cuda.modelopt.Packer,
    ck: cuda.ck.Ck,
    router: cuda.grouped.Router,
    embed_fn: cuda.Function,
    add_fn: cuda.Function,
    swiglu_fn: cuda.Function,
    gdn_pre_fn: cuda.Function,
    gdn_step_fn: cuda.Function,
    gdn_out_fn: cuda.Function,
    attention_kv_fn: cuda.Function,
    attention_fn: cuda.Function,
    route_fn: cuda.Function,
    route_products_fn: cuda.Function,
    route_select_fn: cuda.Function,
    cooperative_route: bool = false,
    combine_fn: cuda.Function,

    pub fn load(ctx: *const cuda.Context) !Ops {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        const major: u32 = @intCast(try ctx.attribute(.compute_capability_major));
        if (major < 12) return error.CheckpointMathNeedsBlackwell;
        const sms: usize = @intCast(try ctx.attribute(.multiprocessor_count));
        var image = try cuda.Module.load(ctx.d, cuda.kernels.qwen35_ops);
        errdefer image.unload();
        var plan_image = try cuda.Module.load(ctx.d, cuda.kernels.experts);
        errdefer plan_image.unload();
        var norms = try cuda.norms.Norms.load(ctx.d);
        errdefer norms.unload();
        var lane = try cuda.qmmf.Lane.load(ctx.d, major);
        errdefer lane.unload();
        var prompt = try cuda.nvfp4.Prompt.load(ctx.d);
        errdefer prompt.unload();
        var nvfp4_experts = try cuda.experts.Nvfp4Experts.load(ctx.d, sms);
        errdefer nvfp4_experts.unload();
        var pack = try cuda.modelopt.Packer.load(ctx.d);
        errdefer pack.unload();
        var ck = try cuda.ck.Ck.load(ctx.d, major, sms);
        errdefer ck.unload();
        return .{
            .image = image,
            .plan_image = plan_image,
            .norms = norms,
            .lane = lane,
            .prompt = prompt,
            .nvfp4_experts = nvfp4_experts,
            .pack = pack,
            .ck = ck,
            .router = try cuda.grouped.Router.resolve(plan_image),
            .embed_fn = try image.function("tf_qwen35_embed"),
            .add_fn = try image.function("tf_qwen35_add"),
            .swiglu_fn = try image.function("tf_qwen35_swiglu"),
            .gdn_pre_fn = try image.function("tf_qwen35_gdn_pre"),
            .gdn_step_fn = try image.function("tf_qwen35_gdn_step"),
            .gdn_out_fn = try image.function("tf_qwen35_gdn_out"),
            .attention_kv_fn = try image.function("tf_qwen35_attention_kv"),
            .attention_fn = try image.function("tf_qwen35_attention"),
            .route_fn = try image.function("tf_qwen35_route"),
            .route_products_fn = try image.function("tf_qwen35_route_products"),
            .route_select_fn = try image.function("tf_qwen35_route_select"),
            .combine_fn = try image.function("tf_qwen35_combine"),
        };
    }

    pub fn unload(self: *Ops) void {
        self.ck.unload();
        self.pack.unload();
        self.nvfp4_experts.unload();
        self.prompt.unload();
        self.lane.unload();
        self.norms.unload();
        self.plan_image.unload();
        self.image.unload();
        self.* = undefined;
    }

    pub fn linear(self: *const Ops, ck_rows: u64, ck_part: u64) cuda.qlinear.Linear {
        return .{ .lane = &self.lane, .nvfp4_prompt = &self.prompt, .ck = &self.ck, .ck_rows = ck_rows, .ck_part = ck_part };
    }

    pub fn grouped(self: *const Ops) cuda.experts.Grouped {
        return .{ .nvfp4 = &self.nvfp4_experts };
    }

    fn go(f: cuda.Function, s: cuda.Stream, grid: usize, threads: u32, args: *cuda.Args) !void {
        try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(grid), .y = 1, .z = 1 }, .block = .{ .x = threads } }, s, args);
    }

    pub fn embed(self: *const Ops, s: cuda.Stream, table: u64, token: u32, out: u64) !void {
        var args: cuda.Args = .{};
        args.add(table);
        args.add(token);
        args.add(out);
        try go(self.embed_fn, s, hidden / 256, 256, &args);
    }

    pub fn rms(self: *const Ops, s: cuda.Stream, x: u64, weight_f32: u64, out: u64, width: usize, eps: f32) !void {
        try self.norms.rms(s, x, .bf16, weight_f32, out, .bf16, 1, width, eps);
    }

    pub fn add(self: *const Ops, s: cuda.Stream, residual: u64, update: u64, width: usize) !void {
        if (width == 0) return;
        var args: cuda.Args = .{};
        for ([_]u64{ residual, update }) |v| args.add(v);
        args.add(@as(i32, @intCast(width)));
        try go(self.add_fn, s, (width + 255) / 256, 256, &args);
    }

    /// q/k FP32 [16,128], v BF16 [32,128], log-decay/beta FP32 [32].
    /// history is BF16 [3,8192], oldest first, updated with the raw qkv row.
    pub fn gdnPre(self: *const Ops, s: cuda.Stream, qkv: u64, conv_weights: u64, history: u64, a: u64, b: u64, a_log: u64, dt_bias: u64, q: u64, k: u64, v: u64, decay: u64, beta: u64) !void {
        var args: cuda.Args = .{};
        for ([_]u64{ qkv, conv_weights, history, a, b, a_log, dt_bias, q, k, v, decay, beta }) |address| args.add(address);
        try go(self.gdn_pre_fn, s, 2 * key_heads + value_heads, gdn_dim, &args);
    }

    /// State is FP32 [32,value128,key128], and out is FP32 [32,128].
    pub fn gdnStep(self: *const Ops, s: cuda.Stream, q: u64, k: u64, v: u64, decay: u64, beta: u64, recurrent_state: u64, out: u64) !void {
        var args: cuda.Args = .{};
        for ([_]u64{ q, k, v, decay, beta, recurrent_state, out }) |address| args.add(address);
        try go(self.gdn_step_fn, s, value_heads, gdn_dim, &args);
    }

    pub fn gdnOut(self: *const Ops, s: cuda.Stream, y_f32: u64, z_bf16: u64, norm_f32: u64, out_bf16: u64, eps: f32) !void {
        var args: cuda.Args = .{};
        for ([_]u64{ y_f32, z_bf16, norm_f32, out_bf16 }) |address| args.add(address);
        args.add(eps);
        try go(self.gdn_out_fn, s, value_heads, gdn_dim, &args);
    }

    /// BF16 caches are packed [capacity,2,256]. The current row is written before attention.
    pub fn attention(self: *const Ops, s: cuda.Stream, q_gate: u64, k: u64, v: u64, q_norm_f32: u64, k_norm_f32: u64, keys: u64, values: u64, position: usize, capacity: usize, out_bf16: u64, eps: f32) !void {
        if (position >= capacity or capacity > 262144) return error.InvalidAttentionPosition;
        var prep: cuda.Args = .{};
        for ([_]u64{ k, v, k_norm_f32, keys, values }) |address| prep.add(address);
        prep.add(@as(i32, @intCast(position)));
        prep.add(eps);
        try go(self.attention_kv_fn, s, kv_heads, head_dim, &prep);
        var args: cuda.Args = .{};
        for ([_]u64{ q_gate, q_norm_f32, keys, values }) |address| args.add(address);
        args.add(@as(i32, @intCast(position)));
        args.add(out_bf16);
        args.add(eps);
        try go(self.attention_fn, s, attention_heads, head_dim, &args);
    }

    pub fn route(self: *const Ops, s: cuda.Stream, x: u64, router_bf16: u64, shared_gate_bf16: u64, picks_i32: u64, weights_f32: u64, scores_f32: u64) !void {
        if (self.cooperative_route) {
            if (scores_f32 == 0) return error.MissingRouteScores;
            var products: cuda.Args = .{};
            for ([_]u64{ x, router_bf16, shared_gate_bf16, scores_f32 }) |address| products.add(address);
            try go(self.route_products_fn, s, 257, 256, &products);
            var select: cuda.Args = .{};
            for ([_]u64{ scores_f32, picks_i32, weights_f32 }) |address| select.add(address);
            return go(self.route_select_fn, s, 1, 256, &select);
        }
        var args: cuda.Args = .{};
        for ([_]u64{ x, router_bf16, shared_gate_bf16, picks_i32, weights_f32 }) |address| args.add(address);
        try go(self.route_fn, s, 1, 256, &args);
    }

    pub fn combine(self: *const Ops, s: cuda.Stream, pairs_f32: u64, weights_f32: u64, shared_bf16: u64, out_bf16: u64) !void {
        var args: cuda.Args = .{};
        for ([_]u64{ pairs_f32, weights_f32, shared_bf16, out_bf16 }) |address| args.add(address);
        try go(self.combine_fn, s, hidden / 256, 256, &args);
    }

    pub fn swiglu(self: *const Ops, s: cuda.Stream, gate: u64, up: u64, out: u64, width: usize) !void {
        if (width == 0) return;
        var args: cuda.Args = .{};
        for ([_]u64{ gate, up, out }) |address| args.add(address);
        args.add(@as(i32, @intCast(width)));
        try go(self.swiglu_fn, s, (width + 255) / 256, 256, &args);
    }
};
