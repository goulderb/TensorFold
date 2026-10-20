const std = @import("std");
const api = @import("engine_api");
const lanes = @import("lanes");
const qwen = @import("qwen35_cuda");
const lane = @import("qwen35_backend.zig");
const Allocator = std.mem.Allocator;

pub const family: api.Family = .{ .model_type = "qwen3_5_moe", .formats = &.{"modelopt-mixed-nvfp4"} };

pub fn claims(gpa: Allocator, io: std.Io, dir: []const u8) !bool {
    return qwen.checkpoint.claims(gpa, io, dir);
}

pub fn refusal(o: api.Open) ?[]const u8 {
    if (o.lanes_fixed and (o.lanes < 1 or o.lanes > 2)) return "the Qwen CUDA baseline serves one or two requests at a time; use --parallel 1 or 2";
    if (o.tp != 1) return "the Qwen CUDA baseline uses a layer split, with no tensor-parallel collectives";
    if (o.vision != null) return "the Qwen CUDA baseline serves text; its vision tower is not ported";
    if (o.drafter != null or o.mtp_drafts != null or o.mtp_confidence != null) return "the Qwen CUDA baseline has no drafter or MTP execution";
    if (o.slide or o.recall or o.learn != null or o.learn_conversations or o.record != null) return "the Qwen CUDA baseline has no learning, recording or persistent cache";
    if (o.keep != null) return "the Qwen CUDA baseline has no --checkpoint-slots keep store";
    if (o.kv8) return "the Qwen ModelOpt baseline supports BF16 KV only; omit --kv-cache fp8";
    if (o.precision != null or o.policy.len > 0) return "the Qwen CUDA baseline uses the checkpoint's FP8 activations, W4A16 experts and BF16 KV";
    if (o.segments != null and o.segments.? != 1) return "the Qwen CUDA baseline prefills one token per engine step; use --segments 1";
    return null;
}

pub fn requestRefusal(r: *const api.Request) ?[]const u8 {
    if (r.structure != null) return "Qwen CUDA structured decoding is not qualified";
    if (r.images.len > 0 or r.vision != null) return "Qwen CUDA serves text only";
    if (r.logprobs != null or r.prompt_logprobs != null) return "Qwen CUDA logprob responses are not qualified";
    if (r.think_budget != 0) return "Qwen CUDA thinking budgets are not qualified";
    return null;
}

fn getenv(name: [:0]const u8) ?[]const u8 {
    return if (std.c.getenv(name)) |v| std.mem.span(v) else null;
}

fn attentionModeFromOpen(o: api.Open) qwen.shared_engine.AttentionMode {
    return switch (o.qwen_attention orelse .chunked) {
        .reference => .reference,
        .chunked => .chunked,
    };
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
    lanes: u32,
};

pub const facts = lane.facts;
pub const snap_host = true;
pub const snap_lookahead: u32 = 0;
pub const snap_grid: u32 = 2048;

const Owner = struct {
    gpa: Allocator,
    model: *qwen.Engine,
    sessions: []qwen.Session,
    adapter: lane.Adapter(qwen.Session),

    fn release(ptr: *anyopaque) void {
        const self: *@This() = @ptrCast(@alignCast(ptr));
        self.adapter.deinit();
        for (self.sessions) |*session| session.deinit();
        self.gpa.free(self.sessions);
        self.model.deinit();
        self.gpa.destroy(self);
    }
};

pub fn snap_bytes(ptr: *anyopaque, at: u32) u64 {
    const self: *Owner = @ptrCast(@alignCast(ptr));
    return self.model.keptBytes(at);
}

pub fn snap_save(ptr: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!*anyopaque {
    const self: *Owner = @ptrCast(@alignCast(ptr));
    const stream: *lanes.Stream = @ptrCast(@alignCast(owner orelse return error.NoStream));
    const i = self.adapter.boundIndex(stream) orelse return error.UnknownStream;
    return self.sessions[i].saveKept(at);
}

pub fn snap_restore(ptr: *anyopaque, owner: ?*anyopaque, saved: *anyopaque) anyerror!void {
    const self: *Owner = @ptrCast(@alignCast(ptr));
    const stream: *lanes.Stream = @ptrCast(@alignCast(owner orelse return error.NoStream));
    const i = self.adapter.boundIndex(stream) orelse return error.UnknownStream;
    return self.sessions[i].restoreKept(saved);
}

pub fn snap_drop(ptr: *anyopaque, saved: *anyopaque) void {
    const self: *Owner = @ptrCast(@alignCast(ptr));
    self.sessions[0].dropKept(saved);
}

pub fn load(a: Allocator, gpa: Allocator, io: std.Io, o: api.Open, problem: *[]const u8) !?Loaded {
    if (refusal(o) orelse environmentRefusal(getenv)) |why| {
        problem.* = why;
        return null;
    }
    const prefill_rows: usize = o.prefill_rows orelse 1;
    if (prefill_rows == 0 or prefill_rows > qwen.shared_engine.hf_prefill_rows) {
        problem.* = "Qwen CUDA --prefill-rows must be from 1 to 16";
        return null;
    }
    const attention_mode = attentionModeFromOpen(o);
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
    const model = qwen.Engine.open(gpa, io, o.dir, .{ .devices = devices, .context = context, .prefill_rows = prefill_rows, .attention_mode = attention_mode }) catch |err| {
        problem.* = try std.fmt.allocPrint(a, "Qwen CUDA load failed: {s}; use a GPU with sufficient free memory or TENSORFOLD_SPLIT=1 for the internal pair", .{@errorName(err)});
        return null;
    };
    errdefer model.deinit();
    const lanes_n: u32 = if (o.lanes_fixed) o.lanes else 1;
    if (lanes_n < 1 or lanes_n > 2) {
        problem.* = "the Qwen CUDA baseline serves one or two requests at a time; use --parallel 1 or 2";
        return null;
    }
    const owner = try gpa.create(Owner);
    errdefer gpa.destroy(owner);
    const sessions = try gpa.alloc(qwen.Session, lanes_n);
    errdefer gpa.free(sessions);
    var made: usize = 0;
    errdefer {
        for (sessions[0..made]) |*session| session.deinit();
    }
    while (made < lanes_n) : (made += 1) {
        sessions[made] = try model.createSession();
    }
    owner.* = .{ .gpa = gpa, .model = model, .sessions = sessions, .adapter = undefined };
    owner.adapter = try lane.Adapter(qwen.Session).initSessions(gpa, owner.sessions, context);
    owner.adapter.prefill_rows = owner.sessions[0].chain.prefill_rows;
    return .{ .ctx = owner, .release = Owner.release, .backend = owner.adapter.backend(), .context = context, .lanes = lanes_n };
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
    o.lanes = 2;
    try std.testing.expect(refusal(o) == null);
    o.lanes = 3;
    try std.testing.expect(refusal(o) != null);
    o.lanes = 1;
    o.learn_conversations = true;
    try std.testing.expect(refusal(o) != null);
    o = base;
    o.kv8 = true;
    try std.testing.expect(refusal(o) != null);
    o = base;
    o.segments = 2;
    try std.testing.expect(refusal(o) != null);
    o = base;
    o.prefill_rows = 16;
    try std.testing.expect(refusal(o) == null);
    o = base;
    o.prompt_cache_gib = 4;
    try std.testing.expect(refusal(o) == null);
}

test "Qwen load refuses an invalid prefill-rows bound before CUDA open" {
    // no Io/dir: the bound check runs before Config.read and Engine.open
    const unused_io: std.Io = undefined;
    for ([_]?u32{ 0, 17 }) |rows| {
        var problem: []const u8 = "";
        const loaded = try load(std.testing.allocator, std.testing.allocator, unused_io, .{ .dir = "missing", .model_type = "qwen3_5_moe", .prefill_rows = rows }, &problem);
        try std.testing.expect(loaded == null);
        try std.testing.expectEqualStrings("Qwen CUDA --prefill-rows must be from 1 to 16", problem);
    }
}

test "Qwen Open maps --qwen-attention through load before CUDA open" {
    const unused_io: std.Io = undefined;
    var problem: []const u8 = "";
    // Invalid prefill still trips first; attention mapping itself is compile-checked by the open call above.
    _ = attentionModeFromOpen(.{ .dir = "missing", .model_type = "qwen3_5_moe" });
    _ = attentionModeFromOpen(.{ .dir = "missing", .model_type = "qwen3_5_moe", .qwen_attention = .reference });
    _ = attentionModeFromOpen(.{ .dir = "missing", .model_type = "qwen3_5_moe", .qwen_attention = .chunked });
    try std.testing.expectEqual(qwen.shared_engine.AttentionMode.chunked, attentionModeFromOpen(.{ .dir = "x", .model_type = "qwen3_5_moe" }));
    try std.testing.expectEqual(qwen.shared_engine.AttentionMode.reference, attentionModeFromOpen(.{ .dir = "x", .model_type = "qwen3_5_moe", .qwen_attention = .reference }));
    try std.testing.expectEqual(qwen.shared_engine.AttentionMode.chunked, attentionModeFromOpen(.{ .dir = "x", .model_type = "qwen3_5_moe", .qwen_attention = .chunked }));
    const loaded = try load(std.testing.allocator, std.testing.allocator, unused_io, .{ .dir = "missing", .model_type = "qwen3_5_moe", .prefill_rows = 0, .qwen_attention = .chunked }, &problem);
    try std.testing.expect(loaded == null);
}

test "Qwen accepts an ordinary chat prompt carrying a decision position" {
    const request: api.Request = .{ .prompt = &.{ 1, 2, 3 }, .max_tokens = 4, .decide = 2 };
    try std.testing.expect(requestRefusal(&request) == null);
}

test "Qwen accepts chat-history spans for the default host's prompt arithmetic" {
    const request: api.Request = .{ .prompt = &.{ 1, 2, 3 }, .max_tokens = 4, .decide = 2, .decode_spans = &.{.{ 0, 2 }} };
    try std.testing.expect(requestRefusal(&request) == null);
}

test "Qwen refuses unsupported direct requests and inherited pool or spill settings" {
    var r: api.Request = .{ .prompt = &.{1}, .max_tokens = 1, .loop_guard = true };
    try std.testing.expect(requestRefusal(&r) == null);
    r.logprobs = 0;
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
