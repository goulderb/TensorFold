const std = @import("std");
const api = @import("engine_api");
const cuda = @import("cuda");
const qwen = @import("qwen35_cuda");
const sampling = @import("lanes").sampling;
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
    return null;
}

pub fn open(a: Allocator, gpa: Allocator, io: std.Io, o: api.Open, problem: *[]const u8) !?api.Opened {
    if (refusal(o)) |why| {
        problem.* = why;
        return null;
    }
    if (std.c.getenv("TENSORFOLD_TP")) |v| if (std.mem.eql(u8, std.mem.span(v), "2")) {
        problem.* = "Qwen CUDA uses TENSORFOLD_SPLIT=1; unset TENSORFOLD_TP for the layer-split baseline";
        return null;
    };
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
    const devices: []const c_int = if (std.c.getenv("TENSORFOLD_SPLIT")) |v|
        (if (std.mem.eql(u8, std.mem.span(v), "1")) &.{ 0, 1 } else &.{device})
    else
        &.{device};
    const model = qwen.Engine.open(gpa, io, o.dir, .{ .devices = devices, .context = context }) catch |err| {
        problem.* = try std.fmt.allocPrint(a, "Qwen CUDA load failed: {s}; use a GPU with sufficient free memory or TENSORFOLD_SPLIT=1 for the internal pair", .{@errorName(err)});
        return null;
    };
    errdefer model.deinit();
    const h = try gpa.create(Host);
    errdefer gpa.destroy(h);
    h.* = .{ .gpa = gpa, .io = io, .model = model, .session = try model.createSession(), .max_queued = o.max_queued orelse 8, .values = undefined, .ids = undefined };
    errdefer h.session.deinit();
    h.values = try gpa.alloc(f64, 248320);
    errdefer gpa.free(h.values);
    h.ids = try gpa.alloc(u64, 248320);
    errdefer gpa.free(h.ids);
    for (h.ids, 0..) |*id, i| id.* = i;
    h.thread = try std.Thread.spawn(.{}, Host.run, .{h});
    return .{ .engine = .{ .ctx = h, .vtable = &Host.vtable }, .ctx = h, .close = Host.close };
}

const Job = struct { id: api.Id, request: *const api.Request, sink: api.Sink, cancelled: bool = false };

const Host = struct {
    gpa: Allocator,
    io: std.Io,
    model: *qwen.Engine,
    session: qwen.Session,
    max_queued: u32,
    mutex: std.Io.Mutex = .init,
    wake: std.Io.Condition = .init,
    queue: std.ArrayList(Job) = .empty,
    current: ?api.Id = null,
    cancel_current: bool = false,
    closing: bool = false,
    thread: ?std.Thread = null,
    values: []f64,
    ids: []u64,
    tokens: std.ArrayList(u32) = .empty,

    const vtable: api.Engine.VTable = .{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory };

    fn of(ctx: *anyopaque) *Host {
        return @ptrCast(@alignCast(ctx));
    }

    fn info(ctx: *anyopaque) api.Info {
        const h = of(ctx);
        return .{ .name = "qwen35-cuda-serial", .lanes = 1, .context_window = @intCast(h.model.capacity), .thinking_budget = false, .max_queued = h.max_queued };
    }

    fn submit(ctx: *anyopaque, id: api.Id, request: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const h = of(ctx);
        h.mutex.lockUncancelable(h.io);
        defer h.mutex.unlock(h.io);
        if (h.closing) return error.Closed;
        if (h.queue.items.len >= h.max_queued) return error.Full;
        h.queue.append(h.gpa, .{ .id = id, .request = request, .sink = sink }) catch return error.Full;
        h.wake.signal(h.io);
    }

    fn cancel(ctx: *anyopaque, id: api.Id) void {
        const h = of(ctx);
        h.mutex.lockUncancelable(h.io);
        defer h.mutex.unlock(h.io);
        if (h.current == id) h.cancel_current = true;
        for (h.queue.items) |*job| if (job.id == id) {
            job.cancelled = true;
        };
        h.wake.signal(h.io);
    }

    fn status(ctx: *anyopaque, out: *api.Status, _: []u32) void {
        const h = of(ctx);
        h.mutex.lockUncancelable(h.io);
        defer h.mutex.unlock(h.io);
        out.* = .{ .running = if (h.current != null) 1 else 0, .waiting = @intCast(h.queue.items.len) };
    }

    fn memory(_: *anyopaque, reset_peak: bool) ?api.Memory {
        const u = cuda.usage(reset_peak);
        return .{ .active = u.device, .peak = u.peak, .pinned = u.host };
    }

    fn cancelled(h: *Host) bool {
        h.mutex.lockUncancelable(h.io);
        defer h.mutex.unlock(h.io);
        return h.closing or h.cancel_current;
    }

    fn event(job: Job, e: api.Event) void {
        job.sink.event(job.sink.ctx, job.id, &e);
    }

    fn execute(h: *Host, job: Job) !api.Reason {
        const r = job.request;
        if (job.cancelled or h.cancelled()) return .cancelled;
        if (r.prompt.len == 0 or r.prompt.len > h.model.capacity or r.max_tokens > h.model.capacity - r.prompt.len) return error.ContextFull;
        if (r.call != null or r.structure != null or r.images.len > 0 or r.decode_spans.len > 0 or r.logprobs != null or r.prompt_logprobs != null or r.think_budget != 0 or r.loop_guard) return error.UnsupportedRequest;
        try h.session.reset();
        h.tokens.clearRetainingCapacity();
        for (r.prompt) |id| {
            if (h.cancelled()) return .cancelled;
            try h.session.append(&.{id});
        }
        event(job, .{ .prefilled = 0 });
        for (0..r.max_tokens) |i| {
            if (h.cancelled()) return .cancelled;
            for (h.values, h.session.logits) |*value, bits| value.* = @as(f32, @bitCast(@as(u32, bits) << 16));
            const id: u32 = @intCast(try sampling.choose(h.gpa, h.values, h.ids, h.session.position, r.sampling orelse .{ .seed = 0, .temperature = 0 }));
            try h.tokens.append(h.gpa, id);
            event(job, .{ .tokens = &.{id} });
            if (std.mem.indexOfScalar(u32, r.eos, id) != null) return .stop;
            if (r.stop) |stop| if (stop.check(stop.ctx, h.tokens.items)) return .stop;
            if (i + 1 < r.max_tokens) try h.session.append(&.{id});
        }
        return .length;
    }

    fn run(h: *Host) void {
        while (true) {
            h.mutex.lockUncancelable(h.io);
            while (h.queue.items.len == 0 and !h.closing) h.wake.waitUncancelable(h.io, &h.mutex);
            if (h.queue.items.len == 0 and h.closing) {
                h.mutex.unlock(h.io);
                return;
            }
            const job = h.queue.orderedRemove(0);
            h.current = job.id;
            h.cancel_current = false;
            h.mutex.unlock(h.io);
            const result = h.execute(job);
            const finished: api.Event = if (result) |reason|
                .{ .finished = .{ .reason = reason, .stats = .{ .rounds = h.tokens.items.len, .min_rows = 1 } } }
            else |err|
                .{ .finished = .{ .reason = .failed, .message = @errorName(err) } };
            h.mutex.lockUncancelable(h.io);
            h.current = null;
            h.mutex.unlock(h.io);
            event(job, finished);
        }
    }

    fn close(ctx: *anyopaque) void {
        const h = of(ctx);
        h.mutex.lockUncancelable(h.io);
        h.closing = true;
        h.wake.signal(h.io);
        h.mutex.unlock(h.io);
        if (h.thread) |thread| thread.join();
        h.queue.deinit(h.gpa);
        h.tokens.deinit(h.gpa);
        h.gpa.free(h.values);
        h.gpa.free(h.ids);
        h.session.deinit();
        h.model.deinit();
        h.gpa.destroy(h);
    }
};

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
}
