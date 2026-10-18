const std = @import("std");
const api = @import("engine_api");
const lanes = @import("lanes");
const qwen = @import("qwen35_cuda");
const lane = @import("qwen35_backend.zig");
const Allocator = std.mem.Allocator;

pub const family: api.Family = .{ .model_type = "qwen3_5_moe", .formats = &.{"modelopt-mixed-nvfp4"} };

pub fn refusal(o: api.Open) ?[]const u8 {
    if (o.lanes_fixed and o.lanes != 1) return "the Qwen CUDA correctness baseline serves one request at a time; use --parallel 1";
    if (o.tp != 1) return "the Qwen CUDA baseline uses a layer split, with no tensor-parallel collectives";
    if (o.vision != null) return "the Qwen CUDA baseline serves text; its vision tower is not ported";
    if (o.drafter != null or o.mtp_drafts != null or o.mtp_confidence != null) return "the Qwen CUDA baseline has no drafter or MTP execution";
    if (o.slide or o.recall or o.learn != null or o.learn_conversations or o.record != null) return "the Qwen CUDA baseline has no learning, recording or persistent cache";
    if (o.keep != null or o.prompt_cache_gib != null) return "the Qwen CUDA baseline keeps independent request state with no prefix cache";
    if (o.precision != null or o.policy.len > 0) return "the Qwen CUDA baseline uses the checkpoint's FP8 activations, W4A16 experts and BF16 KV";
    if (o.segments != null and o.segments.? != 1) return "the Qwen CUDA baseline prefills one token per engine step; use --segments 1";
    return null;
}

pub fn requestRefusal(r: *const api.Request) ?[]const u8 {
    if (r.call != null or r.structure != null) return "Qwen CUDA call gates and structured decoding are not qualified";
    if (r.images.len > 0 or r.vision != null) return "Qwen CUDA serves text only";
    if (r.decode_spans.len > 0) return "Qwen CUDA decoded prompt rows are not qualified";
    if (r.logprobs != null or r.prompt_logprobs != null) return "Qwen CUDA logprob responses are not qualified";
    if (r.think_budget != 0) return "Qwen CUDA thinking budgets are not qualified";
    if (r.decide != null) return "Qwen CUDA has no routed lessons";
    return null;
}

fn getenv(name: [:0]const u8) ?[]const u8 {
    return if (std.c.getenv(name)) |v| std.mem.span(v) else null;
}

fn environmentRefusal(get: *const fn ([:0]const u8) ?[]const u8) ?[]const u8 {
    if (get("TENSORFOLD_TP")) |v| if (!std.mem.eql(u8, v, "1") and !std.mem.eql(u8, v, "0"))
        return "Qwen CUDA uses TENSORFOLD_SPLIT=1; unset TENSORFOLD_TP for the layer-split baseline";
    if (get("TENSORFOLD_SPLIT")) |v| if (!std.mem.eql(u8, v, "1") and !std.mem.eql(u8, v, "0"))
        return "Qwen CUDA TENSORFOLD_SPLIT accepts 0 or 1";
    if (get("TF_CUDA_SEGMENTS")) |v| if (!std.mem.eql(u8, v, "1"))
        return "Qwen CUDA prefills one token per engine step; unset TF_CUDA_SEGMENTS";
    inline for (.{ "TENSORFOLD_KV_POOL_TOKENS", "TENSORFOLD_SPILL_GIB", "TENSORFOLD_SPILL_HOST_GIB", "TENSORFOLD_SPILL_HIGHWATER", "TENSORFOLD_SPILL_MIN_TOKENS", "TENSORFOLD_SPILL_MIN_FREE_GIB", "TENSORFOLD_SPILL_FLUSH_S", "TENSORFOLD_SPILL_KEEP_BUILDS" }) |name| {
        if (get(name) != null) return "Qwen CUDA has no shared KV pool or spill tier; unset TENSORFOLD_KV_POOL_TOKENS and TENSORFOLD_SPILL_*";
    }
    return null;
}

pub const Loaded = struct {
    ctx: *anyopaque,
    release: *const fn (*anyopaque) void,
    backend: lanes.backend.Backend,
    context: usize,
};

pub const facts = lane.facts;

const Owner = struct {
    gpa: Allocator,
    model: *qwen.Engine,
    session: qwen.Session,
    adapter: lane.Adapter(qwen.Session),

    fn release(ptr: *anyopaque) void {
        const self: *@This() = @ptrCast(@alignCast(ptr));
        self.adapter.deinit();
        self.session.deinit();
        self.model.deinit();
        self.gpa.destroy(self);
    }
};

pub fn load(a: Allocator, gpa: Allocator, io: std.Io, o: api.Open, problem: *[]const u8) !?Loaded {
    if (refusal(o) orelse environmentRefusal(getenv)) |why| {
        problem.* = why;
        return null;
    }
    const cfg = qwen.Config.read(gpa, io, o.dir) catch |err| {
        problem.* = try std.fmt.allocPrint(a, "Qwen ModelOpt checkpoint refused: {s}", .{@errorName(err)});
        return null;
    };
    const requested = o.context orelse 4096;
    if (requested < 0 or requested > cfg.context) {
        problem.* = "Qwen context must fit the checkpoint window";
        return null;
    }
    const context: usize = if (requested == 0) cfg.context else @intCast(requested);
    const device = std.math.cast(c_int, o.device orelse 0) orelse {
        problem.* = "Qwen CUDA device index exceeds the CUDA driver range";
        return null;
    };
    const devices: []const c_int = if (getenv("TENSORFOLD_SPLIT")) |v|
        (if (std.mem.eql(u8, v, "1")) &.{ 0, 1 } else &.{device})
    else
        &.{device};
    const model = qwen.Engine.open(gpa, io, o.dir, .{ .devices = devices, .context = context }) catch |err| {
        problem.* = try std.fmt.allocPrint(a, "Qwen CUDA load failed: {s}; use a GPU with sufficient free memory or TENSORFOLD_SPLIT=1 for the internal pair", .{@errorName(err)});
        return null;
    };
    errdefer model.deinit();
    const owner = try gpa.create(Owner);
    errdefer gpa.destroy(owner);
    owner.* = .{ .gpa = gpa, .model = model, .session = try model.createSession(), .adapter = undefined };
    errdefer owner.session.deinit();
    owner.adapter = try lane.Adapter(qwen.Session).init(gpa, &owner.session, context);
    return .{ .ctx = owner, .release = Owner.release, .backend = owner.adapter.backend(), .context = context };
}

test "unsupported Qwen capabilities refuse before loading CUDA" {
    const base: api.Open = .{ .dir = "missing", .model_type = "qwen3_5_moe" };
    try std.testing.expect(refusal(base) == null);
    var o = base;
    o.vision = .{};
    try std.testing.expect(refusal(o) != null);
    o = base;
    o.lanes_fixed = true;
    try std.testing.expect(refusal(o) != null);
    o.lanes = 1;
    try std.testing.expect(refusal(o) == null);
    o.learn_conversations = true;
    try std.testing.expect(refusal(o) != null);
    o = base;
    o.segments = 2;
    try std.testing.expect(refusal(o) != null);
}

test "Qwen refuses unsupported direct requests and inherited pool or spill settings" {
    var r: api.Request = .{ .prompt = &.{1}, .max_tokens = 1, .loop_guard = true };
    try std.testing.expect(requestRefusal(&r) == null);
    r.decode_spans = &.{.{ 0, 1 }};
    try std.testing.expect(requestRefusal(&r) != null);
    const Env = struct {
        fn empty(_: [:0]const u8) ?[]const u8 {
            return null;
        }
        fn pool(name: [:0]const u8) ?[]const u8 {
            return if (std.mem.eql(u8, name, "TENSORFOLD_KV_POOL_TOKENS")) "131072" else null;
        }
        fn spill(name: [:0]const u8) ?[]const u8 {
            return if (std.mem.eql(u8, name, "TENSORFOLD_SPILL_HOST_GIB")) "12" else null;
        }
    };
    try std.testing.expect(environmentRefusal(Env.empty) == null);
    try std.testing.expect(environmentRefusal(Env.pool) != null);
    try std.testing.expect(environmentRefusal(Env.spill) != null);
}

test {
    _ = lane;
}
