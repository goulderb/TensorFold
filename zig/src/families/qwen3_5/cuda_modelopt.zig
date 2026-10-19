const cuda = @import("cuda");
const hf = @import("cuda_hf.zig");
const layers = @import("cuda_layers.zig");

pub const Bindings = struct {
    image: cuda.Module,
    gdn_layout_image: cuda.Module,
    gdn_chain_transposed: cuda.Function,
    lane: cuda.qmmf.Lane,
    prompt: cuda.nvfp4.Prompt,
    nvfp4_experts: cuda.experts.Nvfp4Experts,
    pack: cuda.modelopt.Packer,
    ck: cuda.ck.Ck,
    router: hf.Router,
    rows: u64 = 0,
    part: u64 = 0,

    pub fn load(ctx: *const cuda.Context, fns: *layers.Fns) !Bindings {
        const major: u32 = @intCast(try ctx.attribute(.compute_capability_major));
        if (major < 12) return error.CheckpointMathNeedsBlackwell;
        const sms: usize = @intCast(try ctx.attribute(.multiprocessor_count));
        var image = try cuda.Module.load(ctx.d, cuda.kernels.qwen35_hf);
        errdefer image.unload();
        var gdn_layout_image = try cuda.Module.load(ctx.d, cuda.kernels.qwen35_gdn_layout);
        errdefer gdn_layout_image.unload();
        const gdn_chain_transposed = try gdn_layout_image.function("tf_q35_hf_gdn_chain_transposed");
        var lane = try cuda.qmmf.Lane.load(ctx.d, major);
        errdefer lane.unload();
        var prompt = try cuda.nvfp4.Prompt.load(ctx.d);
        errdefer prompt.unload();
        var experts = try cuda.experts.Nvfp4Experts.load(ctx.d, sms);
        errdefer experts.unload();
        try experts.loadN8Up(sms);
        var pack = try cuda.modelopt.Packer.load(ctx.d);
        errdefer pack.unload();
        var ck = try cuda.ck.Ck.load(ctx.d, major, sms);
        errdefer ck.unload();
        try hf.bind(image, fns);
        return .{ .image = image, .gdn_layout_image = gdn_layout_image, .gdn_chain_transposed = gdn_chain_transposed, .lane = lane, .prompt = prompt, .nvfp4_experts = experts, .pack = pack, .ck = ck, .router = try hf.Router.resolve(image) };
    }

    pub fn unload(b: *Bindings) void {
        b.ck.unload();
        b.pack.unload();
        b.nvfp4_experts.unload();
        b.prompt.unload();
        b.lane.unload();
        b.gdn_layout_image.unload();
        b.image.unload();
    }

    pub fn linear(b: *const Bindings) cuda.qlinear.Linear {
        return .{ .lane = &b.lane, .nvfp4_prompt = &b.prompt, .ck = &b.ck, .ck_rows = b.rows, .ck_part = b.part };
    }
};
