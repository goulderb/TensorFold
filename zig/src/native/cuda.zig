//! CUDA family registration, device setup, memory admission and the native lane host.
const openDrama = @import("drama_open.zig").open;
const std = @import("std");
const platform = @import("sys");
const cuda = @import("cuda");
const api = @import("engine_api");
const lanes = @import("lanes");
const nemotron = @import("nemotron");
const flashnext = @import("flashnext");
const family_memory = @import("cuda_family_memory.zig");
const pool = family_memory.pool;
const family_options = @import("cuda_options.zig");
pub const follow = @import("cuda_follow.zig").follow;
const lfm2 = @import("lfm2.zig");
const glm = @import("glm_cuda");
const qwen35 = @import("qwen35");
const Allocator = std.mem.Allocator;
const budget = @import("cuda_memory.zig");
const lane_backend = @import("cuda_bound.zig");
const snaps = @import("cuda_snaps.zig");
const nemotron_slide = @import("nemotron_slide.zig");
const glm_spill = @import("glm_spill.zig");
const cuda_disk = @import("cuda_disk.zig");
const vision_offer = @import("cuda_vision_offer.zig");
const nspill = @import("nemotron_spill.zig");
const spill_facts = @import("nemotron_spill_facts.zig");
const drafted = @import("cuda_drafted.zig");
const kv_pages = @import("cuda_kv_pool.zig");

/// CUDA families provide metadata, open their lane backend and explain their refusals.
const registry = .{ nemotron.native, nemotron.omni, glm.native, qwen35.native, qwen35.native_moe, flashnext.native };
/// --record opens here (engines.zig refuses it for a backend without this).
pub const records = true;

/// The same copies with a spill tier (TENSORFOLD_SPILL_*): evicted states go to local disk through an owned host slot.
const nemotron_spill_snaps: api.prompt_cache.Snapshots.VTable = .{
    .bytes = nemotron.native.snap_bytes,
    .save = nemotron.native.snap_save,
    .restore = nemotron.native.snap_restore,
    .drop = nemotron.native.snap_drop,
    .spill_write = nspill.spillWrite,
    .spill_read = nspill.spillRead,
    .spare = nspill.spare,
    .trim = nspill.trim,
    .rows = &cuda_disk.image,
};

pub const backends: []const []const u8 = &.{"cuda"};
pub const serves: []const []const u8 = &.{};
pub const families: []const api.Family = blk: {
    var out: [registry.len + 2]api.Family = undefined;
    for (registry, 0..) |F, i| out[i] = .{ .model_type = F.model_type, .formats = F.formats };
    out[registry.len] = lfm2.family;
    out[registry.len + 1] = .{ .model_type = "fish_omni", .formats = &.{"fp8-block128"} };
    const final = out;
    break :blk &final;
};
const gib: f64 = 1 << 30;
const cuda_device = @import("cuda_device.zig");
pub const chip = cuda_device.chip;
pub const deviceName = cuda_device.deviceName;
const getenv = cuda_device.getenv;
const setting = cuda_device.setting;

/// TENSORFOLD_SPLIT=1 asks for the two-GPU layer split. Unset or 0: one device, the existing path.
/// TENSORFOLD_TP=2: two-GPU tensor parallel in one process (cuda_tp2.zig).
fn tpRequested() bool {
    const v = getenv("TENSORFOLD_TP") orelse return false;
    return std.mem.eql(u8, v, "2");
}

/// A layer split or two-GPU TP: either one runs one stream over two devices, with the host's per-device admission.
fn pairRequested() bool {
    return splitOnly() or tpRequested();
}

fn splitOnly() bool {
    const t = std.mem.trim(u8, getenv("TENSORFOLD_SPLIT") orelse return false, " ");
    return t.len > 0 and !std.mem.eql(u8, t, "0");
}

/// The model's window (config.json's max_position_embeddings, text_config's first), 0 when it names none.
fn modelContext(a: Allocator, io: std.Io, dir: []const u8) i64 {
    const path = std.fs.path.join(a, &.{ dir, "config.json" }) catch return 0;
    const bytes = std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(16 << 20)) catch return 0;
    const doc = std.json.parseFromSliceLeaky(std.json.Value, a, bytes, .{}) catch return 0;
    if (doc != .object) return 0;
    const text = if (doc.object.get("text_config")) |t| (if (t == .object) t else doc) else doc;
    const limit = text.object.get("max_position_embeddings") orelse doc.object.get("max_position_embeddings") orelse return 0;
    return if (limit == .integer and limit.integer > 0) limit.integer else 0;
}

/// Prompt plus reply tokens a request may use: --context (0: the model's window), else the family default within it.
fn contextWindow(requested: ?i64, native: i64, default: i64) error{ Negative, NoNative, PastNative }!i64 {
    const r = requested orelse return if (native > 0) @min(default, native) else default;
    if (r < 0) return error.Negative;
    if (r == 0) return if (native > 0) native else error.NoNative;
    if (native > 0 and r > native) return error.PastNative;
    return r;
}

/// A lone driver returns false at completion or true when yield hands it to the lane core.
const LoneRun = *const fn (ctx: *anyopaque, s: *lanes.Stream, hooks: *anyopaque, committed: *const fn (*anyopaque) void, yield: *const fn (*anyopaque) bool, resumed: bool) anyerror!bool;

/// The context a lane thread needs current: the lane host steps rounds on its own thread, CUDA binds per thread.
threadlocal var bound: ?*const cuda.Context = null;

const failed_load = @import("cuda_load_gpu.zig");
const Gpu = failed_load.Gpu;

/// One loaded model behind the lane host: everything the engine thread reads lives here.
pub const Host = struct {
    gpa: Allocator,
    gpu: ?*Gpu, // null when the family binds its own contexts, or in host tests
    family: *anyopaque,
    release: *const fn (*anyopaque) void,
    inner: lanes.backend.Backend,
    vtable: lanes.backend.Backend.VTable,
    cfg: lanes.Config,
    clock: lanes.backend.WallClock,
    core: lanes.Engine,
    host: api.LaneHost,
    startup: []u8 = &.{},
    lone: ?LoneRun = null, // the family's driver for a lone drafted stream, called with `family`
    trim: ?lanes.Trim = null, // optional family row policy, installed before the lane thread starts
    proposers: @FieldType(api.LaneHost, "proposers") = null, // request-local copies through the upstream factory
    store: ?*api.prompt_cache.Store = null, // kept Nemotron states; null when the budget is 0
    spill_owner: ?*glm_spill.Adapter = null,
    disk: ?*nspill.Disk = null, // the store's spill tier (TENSORFOLD_SPILL_*); null when none is configured
    target: ?*nemotron.native.Target = null, // the loaded engine the observers read (engine thread only)
    learn: ?api.Learner = null, // the family's Sliding Weights learner (--slide), called with the context current
    slide: ?*anyopaque = null, // that learner's adapter, freed by slide_free before the family
    slide_free: ?*const fn (Allocator, *anyopaque) void = null,
    dir: []u8 = &.{}, // the model folder the learner writes its lessons into
    follow_fn: ?*const fn (*anyopaque) anyerror!void = null,
    health: ?api.HealthSource = null,
    recording: ?*record.Recording = null, // --record: the recorder in front of the family's backend
    served: drafted.Served = undefined, // the family's backend, wrapped with --drafter's draft model when given
    memory_source: ?api.MemorySource = null,
    request_refusal: ?*const fn (*const api.Request) ?[]const u8 = null,

    fn followFamily(p: *anyopaque) anyerror!void {
        const h: *Host = @ptrCast(@alignCast(p));
        if (h.gpu) |g| try g.ctx.makeCurrent();
        return h.follow_fn.?(h.family);
    }

    /// A family's readiness reads host state only: no CUDA binding or blocking device work.
    fn readiness(h: *Host, comptime F: type) void {
        if (@hasDecl(F, "healthy")) h.health = .{ .ctx = h.family, .read = F.healthy };
    }

    pub fn close(p: *anyopaque) void {
        const h: *Host = @ptrCast(@alignCast(p));
        h.host.stop();
        h.core.deinit();
        h.cfg.deinit(h.gpa);
        if (h.recording) |r| r.free(h.gpa);
        if (h.gpu) |g| g.ctx.makeCurrent() catch {};
        // the lane thread is joined: this thread owns the store and the tier. Every spill write settles before the
        // tier stops; only then may the store free its device copies and the family its context.
        if (h.disk) |d| {
            h.disk = null; // freed by its close, whatever it returns
            const f = d.close(h.store.?) catch |e| {
                std.log.err("spill tier: closing without a bound CUDA context ({s}); device memory is left to the process exit", .{@errorName(e)});
                return; // freeing device memory against an unbound context is not safe
            };
            // the stop's real outcome, once: a state left here is not on disk for the next start
            std.log.info("spill tier: stopped; {d} kept state{s} written now, {d} lost, {d} refused, {d} left unwritten (flush time or a failure); serving wrote {d}, skipped {d} while busy", .{ f.written, if (f.written == 1) "" else "s", f.lost, f.failed, f.left, f.counts.written - f.written, f.counts.skipped_busy });
        }
        if (h.store) |s| {
            if (h.spill_owner) |owner| owner.finish(s);
            const rows = s.disk;
            s.deinit();
            h.gpa.destroy(s);
            if (rows) |r| r.deinit(); // after the store: its entries leave the write queue as they go
        }
        failed_load.closeEntry(if (h.gpu) |g| &g.ctx else null) catch {};
        if (h.slide) |s| h.slide_free.?(h.gpa, s);
        h.gpa.free(h.dir);
        h.served.close(h.gpa);
        h.release(h.family);
        if (h.spill_owner) |owner| owner.deinit();
        if (h.gpu) |g| {
            g.ctx.deinit();
            g.driver.close();
            h.gpa.destroy(g);
        }
        h.gpa.free(h.startup);
        h.gpa.destroy(h);
    }

    /// The store's and the tier's counts with the adapter's, on the engine thread.
    fn observeSpill(p: *anyopaque, out: *api.SpillStatus) bool {
        const h: *Host = @ptrCast(@alignCast(p));
        const store = h.store orelse return false;
        if (!api.spillStatus(store, out)) return false;
        if (h.disk) |d| out.writes = .{ .submitted = d.counts.submitted, .written = d.counts.written, .lost = d.counts.lost, .skipped_busy = d.counts.skipped_busy, .capture_ns = d.counts.capture_ns, .read_ok = d.counts.read_ok, .read_failed = d.counts.read_failed };
        return true;
    }

    fn observePrefill(p: *anyopaque) api.PrefillCounts {
        const h: *Host = @ptrCast(@alignCast(p));
        const t = h.target orelse return .{};
        const c = nemotron.native.prefillCounts(t);
        return .{ .submitted_rows = c.submitted_rows, .completed_rows = c.completed_rows, .passes = c.passes };
    }

    pub fn bind(p: *anyopaque) *Host {
        const h: *Host = @ptrCast(@alignCast(p));
        if (h.gpu) |g| if (bound != &g.ctx or pairRequested()) {
            g.ctx.makeCurrent() catch |e| std.log.err("cuCtxSetCurrent on the lane thread: {s}", .{@errorName(e)});
            bound = &g.ctx;
        };
        return h;
    }

    fn familyPolicies(h: *Host, loaded: anytype) void {
        const L = @TypeOf(loaded);
        h.trim = if (@hasField(L, "trim")) loaded.trim else null;
        h.proposers = if (@hasField(L, "proposers")) loaded.proposers else null;
    }

    /// The round loop and the lane host over the family's backend, its thread started.
    pub fn serve(h: *Host, io: std.Io, facts: lanes.Model, rows: u32, info: api.Info, explain: ?api.Explain) !void {
        h.cfg = try lanes.Config.init(h.gpa, facts, rows, rows - 1);
        errdefer h.cfg.deinit(h.gpa);
        // a drafter's trees: the planner picks a lone stream's lanes, as deep as it drafts
        if (facts.head_trees) h.cfg.head_lanes = rows - 1;
        if (facts.head_trees) h.cfg.head_depth = @max(1, facts.drafts);
        h.clock = .{ .io = io };
        h.core = lanes.Engine.init(h.gpa, &h.cfg, h.backend(), h.clock.clock());
        errdefer h.core.deinit();
        h.core.trim = h.trim;
        h.host = api.LaneHost.init(h.gpa, io, &h.core, info);
        h.host.proposers = h.proposers;
        h.host.memory = h.memory_source orelse (if (h.gpu != null) api.MemorySource{ .read = family_memory.read } else null);
        h.host.request_refusal = h.request_refusal;
        h.host.explain = explain;
        h.host.health = h.health;
        h.host.cache = h.store;
        h.host.warm_quiet = if (info.warm_turns) nemotron.native.warm_quiet else 0;
        if (h.lone != null) h.host.lone = .{ .ctx = h, .run = loneRun, .sampled = true, .logprobs = h.inner.vtable.first_row != null, .structures = h.inner.vtable.first_masked != null, .min_budget = nemotron.native.lone_min_budget, .resumes = true };
        h.host.observe = .{ .ctx = h, .spill = observeSpill, .prefill = observePrefill };
        if (h.learn != null) h.host.learner = h.learner();
        try h.host.start();
    }

    /// The family's learner, each call made with the context current on the lane thread.
    fn learner(h: *Host) api.Learner {
        return .{
            .ctx = h,
            .begin = struct {
                fn f(p: *anyopaque, request: *const api.LearnRequest, sink: api.LearnSink) anyerror!void {
                    const l = bind(p).learn.?;
                    return l.begin(l.ctx, request, sink);
                }
            }.f,
            .step = struct {
                fn f(p: *anyopaque) api.Learner.Step {
                    const l = bind(p).learn.?;
                    return l.step(l.ctx);
                }
            }.f,
            .abort = struct {
                fn f(p: *anyopaque) void {
                    const l = bind(p).learn.?;
                    l.abort(l.ctx);
                }
            }.f,
        };
    }

    /// The family's lone driver with the context current on the lane thread.
    fn loneRun(p: *anyopaque, s: *lanes.Stream, hooks: api.LoneHooks) anyerror!bool {
        const x = bind(p);
        return x.lone.?(x.family, s, hooks.ctx, hooks.committed, hooks.yield, hooks.resumed);
    }

    /// The family's backend, each call made with the context current on the calling thread.
    fn backend(h: *Host) lanes.backend.Backend {
        h.vtable = lane_backend.vtable(Host, h.inner.vtable.*);
        return .{ .ptr = h, .vtable = &h.vtable };
    }
};

const refusals = @import("cuda_refusal.zig");
const record = @import("cuda_record.zig");

/// The engine for `o.dir`, or null with `problem` set when `refusal` names a flag or no CUDA family reads it.
pub fn open(a: Allocator, gpa: Allocator, io: std.Io, o: api.Open, problem: *[]const u8) !?api.Opened {
    if (refusals.backend(o)) |why| {
        problem.* = why;
        return null;
    }
    if (std.mem.eql(u8, o.model_type, "fish_omni")) return openDrama(a, gpa, io, o, problem);
    inline for (registry) |F| {
        if (std.mem.eql(u8, o.model_type, F.model_type)) return openWith(F, a, gpa, io, o, problem);
    }
    if (std.mem.eql(u8, o.model_type, lfm2.family.model_type)) {
        if (refusals.lfm2(o)) |why| {
            problem.* = why;
            return null;
        }
        return lfm2.open(a, gpa, io, o, problem);
    }
    problem.* = try std.fmt.allocPrint(a, "the native CUDA engine has no backend for {s} checkpoints yet", .{o.model_type});
    return null;
}

fn openQwen(a: Allocator, gpa: Allocator, io: std.Io, o: api.Open, problem: *[]const u8) !?api.Opened {
    const loaded = try qwen35.load(a, gpa, io, o, problem) orelse return null;
    errdefer loaded.release(loaded.ctx);
    const h = try gpa.create(Host);
    errdefer gpa.destroy(h);
    h.* = .{
        .gpa = gpa,
        .gpu = null,
        .family = loaded.ctx,
        .release = loaded.release,
        .inner = loaded.backend,
        .vtable = undefined,
        .cfg = undefined,
        .clock = undefined,
        .core = undefined,
        .host = undefined,
        .memory_source = .{ .read = readMemory },
        .request_refusal = qwen35.requestRefusal,
    };
    try h.serve(io, qwen35.facts, 1, .{
        .name = "qwen35-cuda-lanes",
        .lanes = 1,
        .context_window = @intCast(loaded.context),
        .max_queued = o.max_queued orelse 8,
        .plain_only = true,
        .thinking_budget = false,
    }, null);
    return .{ .engine = h.host.engine(), .close = Host.close, .ctx = h };
}

/// The admit's refusals: no stream fits, or --parallel asks for more than the budget holds.
const no_stream = "the CUDA memory budget fits no stream: one at a {d}-token window takes {d:.2} GiB, " ++
    "and {d:.2} GiB is left after the model's {d:.2} GiB and the {d:.1} GiB reserve{s}; " ++
    "lower --context, or free device memory";
const too_many = "--parallel {d} needs {d:.2} GiB for its streams at a {d}-token window, " ++
    "and the CUDA memory budget leaves {d:.2} GiB{s}: serve --parallel {d}, or lower --context";

fn openWith(comptime F: type, a: Allocator, gpa: Allocator, io: std.Io, o: api.Open, problem: *[]const u8) !?api.Opened {
    if (refusals.family(F, o)) |why| {
        problem.* = why;
        return null;
    }
    const native = modelContext(a, io, o.dir);
    const window = contextWindow(o.context, native, F.default_context) catch |e| {
        problem.* = switch (e) {
            error.Negative => try std.fmt.allocPrint(a, "--context {d}: a token count, or 0 for the model's window", .{o.context.?}),
            error.NoNative => "--context 0 asks for the model's window, and its config.json names none: give a token count",
            error.PastNative => try std.fmt.allocPrint(a, "--context {d} exceeds this model's {d}-token window", .{ o.context.?, native }),
        };
        return null;
    };
    // the spill tier: off unless TENSORFOLD_SPILL_GIB asks for it, read before any CUDA work
    var disk_config: ?spill_facts.Config = null;
    if (F != glm.native) {
        disk_config = spill_facts.Config.fromEnv(gpa, api.prompt_cache.spill.ProcessEnv{}) catch |e| {
            problem.* = try std.fmt.allocPrint(a, "TENSORFOLD_SPILL_*: {s} (an enabled tier needs TENSORFOLD_SPILL_GIB and TENSORFOLD_SPILL_HOST_GIB; docs/spill-tier.md)", .{@errorName(e)});
            return null;
        };
    }
    defer if (disk_config) |c| if (c.tier.dir.len > 0) gpa.free(c.tier.dir);
    if (disk_config != null and (F != nemotron.native or o.learn_conversations)) {
        problem.* = "TENSORFOLD_SPILL_GIB: this adapter serves Nemotron without --learn-conversations; choose one disk store";
        return null;
    }

    if (disk_config != null and o.slide) {
        problem.* = "TENSORFOLD_SPILL_GIB with --slide: stored states hold the checkpoint's arithmetic, and Living Weights change it; serve one or the other";
        return null;
    }
    if (o.slide) {
        if (!@hasDecl(F, "slide")) {
            problem.* = try std.fmt.allocPrint(a, "--slide: {s} checkpoints do not learn on CUDA yet", .{o.model_type});
            return null;
        }
        if (try nemotron_slide.refusal(F, a, io, o.dir, "`cp -RL` (a copy that follows links)")) |why| {
            problem.* = why;
            return null;
        }
    }
    const device = try setting(a, o.device, "TF_CUDA_DEVICE", 0, problem) orelse return null;
    const segments = try setting(a, o.segments, "TF_CUDA_SEGMENTS", 1, problem) orelse return null;
    if (segments < 1 or segments > F.max_segments) {
        const named = if (o.segments != null) "--segments " else "TF_CUDA_SEGMENTS=";
        problem.* = try std.fmt.allocPrint(a, "{s}{d}: whole prompt chunks a call, 1 to {d}", .{ named, segments, F.max_segments });
        return null;
    }
    if (o.slide and segments > 1) {
        problem.* = "--slide runs a prompt one chunk at a time, its change's scratch shared by every layer: serve it without --segments or TF_CUDA_SEGMENTS";
        return null;
    }
    if (failed_load.recover() != 0) {
        problem.* = "a CUDA loader is retained until its context recovers; retry after recovery";
        return null;
    }
    const g = try gpa.create(Gpu);
    g.gpa = gpa;
    g.next = null;
    var opened = false;
    defer if (!opened) gpa.destroy(g);
    g.driver = cuda.Driver.open() catch |e| {
        problem.* = try std.fmt.allocPrint(a, "no CUDA driver ({s})", .{@errorName(e)});
        return null;
    };
    defer if (!opened) g.driver.close();
    const count = try g.driver.deviceCount();
    if (device >= count) {
        const named = if (o.device != null) "--device " else "TF_CUDA_DEVICE=";
        problem.* = try std.fmt.allocPrint(a, "{s}{d}: this machine has {d} CUDA device{s} (0 to {d})", .{ named, device, count, if (count == 1) "" else "s", @max(count, 1) - 1 });
        return null;
    }
    g.ctx = try cuda.Context.init(&g.driver, @intCast(device));
    defer if (!opened) g.ctx.deinit();
    bound = &g.ctx;
    // a failed open frees this context (and g): the thread must not keep it bound
    defer if (!opened) {
        bound = null;
    };
    const capability = try g.ctx.capability();
    var name_buf: [256]u8 = undefined;
    const name = g.ctx.name(&name_buf) catch "GPU";
    if (o.slide and capability != 121) {
        problem.* = try std.fmt.allocPrint(a, "--slide learns on a GB10 (sm_121) so far, and this {s} is sm_{d}: serve it without --slide", .{ name, capability });
        return null;
    }
    const kernels: ?[]const u8 = switch (try cuda.aot.pick(a, io, getenv("TENSORFOLD_CUDA_KERNELS"), capability)) {
        .native => null,
        .captured => |dir| dir,
        .missing => |dir| {
            problem.* = try std.fmt.allocPrint(a, "TENSORFOLD_CUDA_KERNELS={s} holds no aot.json: give a captured set's folder, or native for our own kernels", .{dir});
            return null;
        },
    };
    // the spill tier's identity: the files the load reads, stat'ed before it and hashed whole after it
    var model_files: ?spill_facts.Artifacts = null;
    defer if (model_files) |*m| m.deinit(gpa);
    var exe_files: ?spill_facts.Artifacts = null;
    defer if (exe_files) |*m| m.deinit(gpa);
    if (F == nemotron.native) if (disk_config != null) {
        const paths = F.artifacts(gpa, io, o.dir, o.drafts) catch |e| {
            problem.* = try std.fmt.allocPrint(a, "spill tier: the checkpoint's files in {s} cannot be listed ({s})", .{ o.dir, @errorName(e) });
            return null;
        };
        model_files = spill_facts.Artifacts.take(gpa, paths) catch |e| { // takes `paths`, freed on failure too
            problem.* = try std.fmt.allocPrint(a, "spill tier: the checkpoint's files in {s} cannot be identified ({s})", .{ o.dir, @errorName(e) });
            return null;
        };
        exe_files = spill_facts.executable(gpa, io, kernels) catch |e| {
            problem.* = try std.fmt.allocPrint(a, "spill tier: this executable or its kernel set cannot be identified ({s})", .{@errorName(e)});
            return null;
        };
    };
    const before = try pool(a, io, &g.ctx, problem) orelse return null;
    // a family split over hosts holds only its rank's share (GLM-5.3-Flash's TP)
    const weights = try family_memory.weights(F, a, io, o, problem) orelse return null;
    const held0 = cuda.usage(false).device;
    // a family over hosts reads every rank's room after its load, by this budget, and admits by the smallest
    const Probe = struct {
        a: Allocator,
        io: std.Io,
        ctx: *const cuda.Context,
        held: u64,
        problem: *[]const u8,
        fn read(p: *anyopaque) ?u64 {
            const self: *@This() = @ptrCast(@alignCast(p));
            const now = (pool(self.a, self.io, self.ctx, self.problem) catch return null) orelse return null;
            return now.room(cuda.usage(false).device - self.held);
        }
    };
    var probe: Probe = .{ .a = a, .io = io, .ctx = &g.ctx, .held = held0, .problem = problem };
    const split = if (comptime @hasField(F.Options, "split")) pairRequested() else false;
    const tp = if (comptime @hasField(F.Options, "split")) tpRequested() else false;
    if (split and !tp and (o.drafts or o.slide or o.learn != null or segments != 1 or disk_config != null or (o.prompt_cache_gib orelse 0) != 0 or (o.lanes_fixed and o.lanes != 1))) {
        problem.* = "TENSORFOLD_SPLIT is a single-stream layer-split trial, not TP: use --no-drafts --parallel 1 --prompt-cache-gib 0 without learning, spill or segments";
        return null;
    }
    if (tp and (o.drafts or o.slide or o.learn != null or segments != 1)) {
        problem.* = "TENSORFOLD_TP=2 serves without drafts, Sliding Weights, learned states or segments: use --no-drafts";
        return null;
    }
    var opts: F.Options = .{ .context = @intCast(window), .drafts = o.drafts, .segments = segments, .slide = o.slide };
    if (!family_options.set(F, o, &opts, problem)) return null;
    if (@hasField(F.Options, "split")) opts.split = splitOnly();
    if (comptime @hasField(F.Options, "tp")) {
        if (comptime @FieldType(F.Options, "tp") == bool) opts.tp = tp;
    }
    if (@hasField(F.Options, "room")) opts.room = .{ .ctx = &probe, .read = Probe.read };
    var spill_owner: ?*glm_spill.Adapter = null;
    defer if (!opened) if (spill_owner) |owner| owner.deinit();
    if (comptime F == glm.native) {
        spill_owner = try glm_spill.Adapter.prepare(gpa, io, o.dir);
        if (spill_owner) |owner| opts.persistence = .{ .ctx = owner, .attach = glm_spill.Adapter.attach };
    }
    // A layer split admits rank 0's share against this device, not the whole checkpoint: rank 1's share lands on the
    // other card, which the peer's own open checks. The layer count is not read yet, so the share is the same
    // fraction the split uses, over the checkpoint's bytes.
    const share = if (split) weights * 54 / 100 else weights;
    if (share > before.room(held0)) {
        problem.* = try std.fmt.allocPrint(a, "the checkpoint's {d:.1} GiB of weights{s} do not fit the {d:.1} GiB the CUDA memory budget grants ({d:.1} GiB free less a {d:.1} GiB reserve{s}); free device memory or adjust TENSORFOLD_MEMORY_RESERVE_GIB / TENSORFOLD_CUDA_MEMORY_LIMIT_GB", .{ toGib(weights), if (split) " (this rank's share)" else "", toGib(before.room(held0)), toGib(before.free), toGib(before.reserve), if (before.limit != null) ", under TENSORFOLD_CUDA_MEMORY_LIMIT_GB" else "" });
        return null;
    }
    const loaded = F.open(gpa, io, &g.ctx, o.dir, kernels, opts) catch |e| {
        cuda.tp_watch.fail(@errorName(e)); // a family over hosts: its other ranks stop with this error
        if (@hasDecl(F, "explains_open")) if (F.explain(null, e)) |text| {
            problem.* = text;
            return null;
        };
        switch (@as(anyerror, e)) { // every family's open: GLM's error set has no ModeloptNeedsSm90
            error.LoadRetained => {
                opened = true;
                failed_load.keep(g);
                problem.* = "CUDA loading failed without completion; its destinations and GPU owner are retained";
                return null;
            },
            error.ModeloptNeedsSm90 => {
                problem.* = try std.fmt.allocPrint(
                    a,
                    "{s} is sm_{d}: {s} (the 4-bit checkpoints run on it)",
                    .{ name, capability, nemotron.kernels.modeloptRefusal(capability).? },
                );
                return null;
            },
            else => {
                problem.* = try std.fmt.allocPrint(
                    a,
                    "the native CUDA engine cannot load {s} with {s} glue kernels ({s})",
                    .{ o.dir, kernels orelse "its own", @errorName(e) },
                );
                return null;
            },
        }
    };
    defer if (!opened) loaded.deinit(loaded.ctx);
    const taps = record.taps(loaded); // --record is refused before the host is made, so a refusal holds nothing
    if (o.record) |dir| if (try record.refusal(a, io, o.model_type, dir, taps)) |why| {
        problem.* = why;
        return null;
    };
    if (F == nemotron.native) if (disk_config != null and loaded.living) {
        problem.* = try std.fmt.allocPrint(a, "TENSORFOLD_SPILL_GIB: {s} holds Living Weights, which change the arithmetic stored states would keep; serve without the tier", .{o.dir});
        return null;
    };
    var served = try drafted.wrap(F, a, gpa, io, loaded, o, problem) orelse return null;
    defer if (!opened) served.close(gpa);
    const model = cuda.usage(false).device - held0;
    const after = try pool(a, io, &g.ctx, problem) orelse return null;
    const L = @TypeOf(loaded); // a family's Loaded may leave out a learner and a lone driver
    const learn_bytes: usize = if (@hasField(L, "learn_bytes")) loaded.learn_bytes else 0;
    const host_cache = try kv_pages.placeHost(F, a, loaded, o, after.room(model), @intCast(window), served.stream_bytes, problem) orelse return null;
    const host_budget = if (host_cache and @hasDecl(F, "snap_save")) budget.hostCacheBytes(a, io, o.prompt_cache_gib) else 0;
    // a split's model bytes span both devices: this one's room is its free memory less the larger of the reserve and
    // the 1536 MiB floor (the peer checked its own floor at open)
    var room = if (split) after.free -| @max(after.reserve, 1536 << 20) else try kv_pages.startupRoom(F, a, loaded, o, after.room(model), host_budget, problem) orelse return null;
    var low: []const u8 = ""; // names the rank whose room bounds the streams, when it is not this one
    if (@hasField(@TypeOf(loaded), "peer_room")) if (loaded.peer_room) |pr| if (pr.bytes < room) {
        room = pr.bytes;
        low = try std.fmt.allocPrint(a, " on rank {d}, the smallest room of the ranks", .{pr.rank});
    };
    const free = loaded.free_streams;
    const layout = kv_pages.layout(loaded);
    var paged: ?kv_pages.Plan = null;
    const lane_limit = if (@hasField(@TypeOf(loaded), "lane_limit")) loaded.lane_limit else std.math.maxInt(u32);
    const asked = budget.laneCount(o.lanes, o.lanes_fixed, lane_limit) catch {
        problem.* = try std.fmt.allocPrint(a, "this CUDA cache supports at most {d} lanes; lower --parallel {d}", .{ lane_limit, o.lanes });
        return null;
    };
    const streams: u32 = if (split and !tp) 1 else if (layout) |l| blk: {
        paged = try kv_pages.plan(F, a, l, room, o, @intCast(window), problem, host_cache) orelse return null;
        break :blk paged.?.lanes;
    } else budget.admit(room, served.stream_bytes, free, asked, o.lanes_fixed) catch |e| {
        problem.* = switch (e) {
            error.NoStream => try std.fmt.allocPrint(a, no_stream, .{
                window,
                toGib(served.stream_bytes),
                toGib(room),
                toGib(model),
                toGib(after.reserve),
                low,
            }),
            error.TooMany => try std.fmt.allocPrint(a, too_many, .{
                o.lanes,
                toGib(served.stream_bytes * (o.lanes -| free)),
                window,
                toGib(room),
                low,
                free + room / served.stream_bytes,
            }),
        };
        return null;
    };
    const held = served.stream_bytes * (streams -| free);
    const kept = if (split and !tp) kv_pages.Kept{ .snaps = 0, .copied = 0, .snap_bytes = 0 } else kv_pages.kept(F, loaded, paged, o, room, held, host_budget, host_cache);
    const store_cache = if (@hasDecl(F, "snap_save")) try snaps.cacheBudget(F, loaded.target, kept.copied) else 0;
    // Mamba spares are prepaid; the tier reservation is charged once inside the remaining Store budget.
    const disk: ?*nspill.Disk = if (F != nemotron.native) null else if (disk_config) |*c| (try openDisk(F, a, gpa, io, g, loaded, c, &model_files.?, &exe_files.?, kernels, segments, store_cache, after.unified, problem)) orelse return null else null;
    defer if (!opened) if (disk) |d| d.abandon(null); // after the store's own unwind, which detaches it
    if (disk != null) disk_config.?.tier.dir = ""; // the tier owns it now
    if (!try kv_pages.attach(F, a, loaded.ctx, paged, problem)) return null;
    served.reserve(streams) catch |e| {
        problem.* = try std.fmt.allocPrint(a, "the native CUDA engine cannot reserve {d} streams' caches at a {d}-token window ({s}): serve fewer with --parallel, or lower --context", .{ streams, window, @errorName(e) });
        return null;
    };
    const h = try gpa.create(Host);
    errdefer gpa.destroy(h);
    h.* = .{ .gpa = gpa, .gpu = g, .family = loaded.ctx, .release = loaded.deinit, .inner = served.backend, .served = served, .vtable = undefined, .cfg = undefined, .clock = undefined, .core = undefined, .host = undefined, .lone = if (@hasField(L, "lone")) loaded.lone else null };
    h.spill_owner = spill_owner;
    h.familyPolicies(loaded);
    h.readiness(F);
    const shown = try kv_pages.text(F, a, served.stream_bytes, layout, paged, @intCast(window), kept, o.kv8);
    h.startup = try std.fmt.allocPrint(gpa, "CUDA sm_{d} device {d} ({s}{s}): model {d:.2} GiB{s}; {d} stream{s} at once, {s}, of {d:.1} GiB left after a {d:.1} GiB reserve; prompts in passes of up to {d} rows{s}{s}; {s}", .{
        capability, device,                                                                                                       name, if (after.unified) ", memory shared with the host" else "", toGib(model), if (learn_bytes > 0) try std.fmt.allocPrint(a, " and {d:.2} GiB kept for --slide's trainer", .{toGib(learn_bytes)}) else "", streams, if (streams == 1) "" else "s", shown[0], toGib(room), toGib(after.reserve), F.prompt_rows, if (segments > 1) try std.fmt.allocPrint(a, ", {d} staggered segments a call", .{segments}) else "",
        shown[1],   if (kernels) |dir| try std.fmt.allocPrint(a, "glue kernels captured at {s}", .{dir}) else "own glue kernels",
    });
    errdefer gpa.free(h.startup);
    if (@hasDecl(F, "snap_save") and (kept.copied > 0 or kept.snaps > 0)) {
        const pc = api.prompt_cache;
        const store = try gpa.create(pc.Store);
        errdefer gpa.destroy(store);
        // rules and budget from the family's snap_* declarations (cuda_snaps.zig)
        const image: ?*const pc.rows.Image = if (F == nemotron.native) &cuda_disk.image else null; // its disk rows
        const copies: pc.Snapshots = .{ .ptr = loaded.target, .vtable = if (F == nemotron.native and disk != null) &nemotron_spill_snaps else &snaps.Snaps(F, image).vtable };
        var rules = snaps.cacheRules(F);
        if (disk != null) {
            rules.planned = true;
            rules.grid = F.prompt_rows;
        }
        store.* = pc.Store.init(gpa, copies, rules, store_cache);
        errdefer {
            const rows = store.disk;
            store.deinit();
            if (rows) |r| r.deinit();
            h.store = null;
        }
        try kv_pages.store(F, store, loaded.target, kept); // its snapshot slots, before a disk probe takes one
        h.store = store;
        if (F == nemotron.native and o.learn_conversations) {
            if (o.learn) |root| try cuda_disk.attach(a, io, &g.ctx, o, root, kernels, loaded, store);
        }
        if (F == nemotron.native) if (disk) |d| d.attach(loaded.target, store);
        if (spill_owner) |owner| store.rules.spill_timing = owner.timing;
        if (@hasDecl(F, "snap_tier")) if (F.snap_tier(loaded.target)) |tier| store.attach(@ptrCast(@alignCast(tier)));
    }
    errdefer if (h.store) |store| {
        store.deinit(); // detaches the tier; the disk itself goes in the deferred abandon below this frame's start
        gpa.destroy(store);
    };
    if (@hasDecl(F, "slide")) if (loaded.learns != null) {
        const Slide = nemotron_slide.Adapter(F.slide);
        h.dir = try gpa.dupe(u8, o.dir);
        const s = try gpa.create(Slide);
        s.* = Slide.init(gpa, io, F.learnModel(loaded.ctx, h.dir));
        h.slide = s;
        h.slide_free = struct {
            fn f(al: Allocator, p: *anyopaque) void {
                const x: *Slide = @ptrCast(@alignCast(p));
                x.deinit();
                al.destroy(x);
            }
        }.f;
        h.learn = s.hook();
    };
    errdefer if (h.slide) |s| {
        h.slide_free.?(gpa, s);
        gpa.free(h.dir);
    };
    if (o.record) |dir| {
        const r = try record.open(gpa, io, dir, taps, served.backend);
        h.recording = r;
        h.inner = r.recorded.backend();
        h.lone = null; // every token through the lane rounds, where the recorder sees it
        std.log.info("recording each stream's rows under {s}/{s}", .{ dir, r.recorder.name });
    }
    errdefer if (h.recording) |r| r.free(gpa);
    // the family cuts its own prompt grid from position 0, as `tensorfold run` does: prefill_step 0
    h.disk = disk;
    h.target = if (F == nemotron.native) loaded.target else null;
    const admit = admission(F, loaded, paged);
    try h.serve(io, served.facts, loaded.rows, .{
        .lanes = streams,
        .max_queued = o.max_queued,
        .call_gates = true,
        .context_window = @intCast(window),
        .startup = h.startup,
        .prompt_cache = h.store != null,
        .warm_turns = h.store != null and @hasDecl(F, "warm_quiet"),
        .fill_arrival = F == nemotron.native, // agent turns grow: shortest-first starved the longest ones
        .logprobs = served.backend.vtable.first_row != null,
        .prompt_logprobs = served.backend.vtable.prompt_rows,
        .structures = served.backend.vtable.first_masked != null,
        .vision = if (@hasField(@TypeOf(loaded), "vision")) vision_offer.offer(loaded.vision, o.vision) else null,
        .encoded_vision = if (@hasField(@TypeOf(loaded), "vision_probe")) if (loaded.vision_probe) |image_probe| .{ .ctx = image_probe.ctx, .rows = image_probe.rows } else null else null,
        .admit = admit,
    }, .{ .ctx = loaded.ctx, .text = F.explain });
    opened = true;
    if (@hasField(@TypeOf(loaded), "follow")) if (loaded.follow) |f| {
        h.follow_fn = f;
        return .{ .engine = h.host.engine(), .close = Host.close, .ctx = h, .follow = Host.followFamily };
    };
    return .{ .engine = h.host.engine(), .close = Host.close, .ctx = h };
}

const toGib = family_memory.toGib;

/// A retained family's pool owns its admission; paged families expose it through the common plan.
fn admission(comptime F: type, loaded: anytype, paged: ?kv_pages.Plan) ?api.Admit {
    if (comptime @hasField(@TypeOf(loaded), "admit")) {
        if (loaded.admit) |ad| return .{ .ctx = ad.ctx, .fits = ad.fits };
    }
    if (comptime @hasDecl(F, "fits")) {
        if (paged != null) return .{ .ctx = loaded.ctx, .fits = F.fits };
    }
    return null;
}

test "optional family trim and proposer factories reach the served lane host without changing other families" {
    const Hooks = struct {
        fn release(_: *anyopaque) void {}
        fn choose(_: *anyopaque, _: []const lanes.trim.Window, _: bool, _: Allocator, _: []lanes.trim.Choice) !void {}
        fn commit(_: *anyopaque, _: lanes.trim.Commit) !void {}
        fn make(_: *anyopaque, _: Allocator, _: []const u32, _: []const u32) !lanes.proposer.Proposer {
            return error.TestUnexpectedResult;
        }
        fn free(_: *anyopaque, _: lanes.proposer.Proposer) void {}
    };
    var target: lanes.fake.Fake = .{ .gpa = std.testing.allocator };
    defer target.deinit();
    var h: Host = .{ .gpa = std.testing.allocator, .gpu = null, .family = &target, .release = Hooks.release, .inner = target.backend(), .vtable = undefined, .cfg = undefined, .clock = undefined, .core = undefined, .host = undefined };
    h.familyPolicies(.{
        .trim = @as(?lanes.Trim, .{ .ptr = &target, .vtable = &.{ .choose = Hooks.choose, .commit = Hooks.commit } }),
        .proposers = @as(@FieldType(api.LaneHost, "proposers"), .{ .ctx = &target, .make = Hooks.make, .free = Hooks.free }),
    });
    try h.serve(std.testing.io, .{ .exact_width = 16, .hidden_rows = true }, 16, .{ .lanes = 2 }, null);
    defer {
        h.host.stop();
        h.core.deinit();
        h.cfg.deinit(h.gpa);
    }
    try std.testing.expectEqual(@as(*anyopaque, &target), h.core.trim.?.ptr);
    try std.testing.expectEqual(@as(*anyopaque, &target), h.host.proposers.?.ctx);
    h.familyPolicies(.{});
    try std.testing.expect(h.trim == null and h.proposers == null);
}

test "retained family pool admission survives the common page-plan adapter" {
    const F = struct {};
    const Gate = struct {
        fn fits(p: *anyopaque, prompt: u32, output: u32) bool {
            const limit: *const u32 = @ptrCast(@alignCast(p));
            return @as(u64, prompt) + output <= limit.*;
        }
    };
    var limit: u32 = 64;
    const loaded = .{ .admit = @as(?api.Admit, .{ .ctx = &limit, .fits = Gate.fits }) };
    const ad = admission(F, loaded, null) orelse return error.TestUnexpectedResult;
    try std.testing.expect(ad.fits(ad.ctx, 32, 32));
    try std.testing.expect(!ad.fits(ad.ctx, 32, 33));
    try std.testing.expect(admission(F, .{}, null) == null);
}

test "context windows: the family default inside the model's, 0 for the model's, never past it" {
    try std.testing.expectEqual(@as(i64, 16384), try contextWindow(null, 262144, 16384));
    try std.testing.expectEqual(@as(i64, 4096), try contextWindow(null, 4096, 16384));
    try std.testing.expectEqual(@as(i64, 262144), try contextWindow(0, 262144, 16384));
    try std.testing.expectEqual(@as(i64, 65536), try contextWindow(65536, 262144, 16384));
    try std.testing.expectError(error.PastNative, contextWindow(300000, 262144, 16384));
    try std.testing.expectError(error.Negative, contextWindow(-5, 262144, 16384));
    try std.testing.expectError(error.NoNative, contextWindow(0, 0, 16384));
}

test {
    _ = family_memory;
    _ = budget; // cuda_memory.zig's own tests, which no other test binary reaches
    _ = refusals;
    _ = lane_backend;
    _ = snaps;
    _ = cuda_device; // the device and variable tests
    _ = nspill;
    _ = spill_facts;
}

test "the lane core sees a family's prefill_step through the lane backend, so a served prompt can step" {
    var target: lanes.fake.Fake = .{ .gpa = std.testing.allocator, .prefill_chunks = 3 };
    defer target.deinit();
    var h: Host = undefined;
    h.inner = target.stepped();
    try std.testing.expect(h.backend().vtable.prefill_step != null);
}

test "every registered family and d1's decision engine are listed for capabilities" {
    try std.testing.expectEqual(@as(usize, registry.len + 2), families.len);
    try std.testing.expectEqualStrings("nemotron_h", families[0].model_type);
    try std.testing.expectEqualStrings("lfm2_vl", families[registry.len].model_type);
}

test "the Qwen host profile preserves memory reporting, direct refusals and zero-output completion" {
    const gpa = std.testing.allocator;
    var target: lanes.fake.Fake = .{ .gpa = gpa, .prefill_chunks = 3 };
    defer target.deinit();
    const Fixture = struct {
        fn release(_: *anyopaque) void {}
        fn memory(_: ?*anyopaque, _: bool) ?api.Memory {
            return .{ .active = 23 };
        }
        mutex: std.Io.Mutex = .init,
        done: ?api.Reason = null,
        terminals: usize = 0,
        tokens: usize = 0,
        fn event(ctx: *anyopaque, _: api.Id, ev: *const api.Event) void {
            const self: *@This() = @ptrCast(@alignCast(ctx));
            self.mutex.lockUncancelable(std.testing.io);
            defer self.mutex.unlock(std.testing.io);
            switch (ev.*) {
                .tokens => |t| self.tokens += t.len,
                .finished => |f| {
                    self.done = f.reason;
                    self.terminals += 1;
                },
                else => {},
            }
        }
        fn wait(self: *@This()) !api.Reason {
            for (0..5000) |_| {
                self.mutex.lockUncancelable(std.testing.io);
                const done = self.done;
                self.mutex.unlock(std.testing.io);
                if (done) |reason| return reason;
                try std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake);
            }
            return error.TestTimedOut;
        }
    };
    const h = try gpa.create(Host);
    defer gpa.destroy(h);
    h.* = .{
        .gpa = gpa,
        .gpu = null,
        .family = &target,
        .release = Fixture.release,
        .inner = target.stepped(),
        .vtable = undefined,
        .cfg = undefined,
        .clock = undefined,
        .core = undefined,
        .host = undefined,
        .memory_source = .{ .read = Fixture.memory },
        .request_refusal = qwen35.requestRefusal,
    };
    try h.serve(std.testing.io, qwen35.facts, 1, .{ .lanes = 1, .plain_only = true, .thinking_budget = false }, null);
    defer {
        h.host.stop();
        h.core.deinit();
        h.cfg.deinit(gpa);
    }
    const e = h.host.engine();
    try std.testing.expectEqual(@as(u64, 23), e.memory(false).?.active);
    try std.testing.expect(e.info().plain_only and e.info().loop_guard);
    try std.testing.expectEqual(@as(u32, 0), e.info().prefill_step);
    try std.testing.expect(!h.host.decoded_rows);
    var refused: Fixture = .{};
    const bad: api.Request = .{ .prompt = &.{ 2, 7, 1 }, .max_tokens = 2, .logprobs = 0 };
    try e.submit(1, &bad, .{ .ctx = &refused, .event = Fixture.event });
    try std.testing.expectEqual(api.Reason.failed, try refused.wait());
    try std.testing.expectEqual(@as(usize, 0), target.prefill_count);
    var empty: Fixture = .{};
    const zero: api.Request = .{ .prompt = bad.prompt, .max_tokens = 0 };
    try e.submit(2, &zero, .{ .ctx = &empty, .event = Fixture.event });
    try std.testing.expectEqual(api.Reason.length, try empty.wait());
    try std.testing.expectEqual(@as(usize, 0), empty.tokens);
    var next: Fixture = .{};
    const valid: api.Request = .{ .prompt = bad.prompt, .max_tokens = 3, .decide = 2, .decode_spans = &.{.{ 0, 2 }} };
    try e.submit(3, &valid, .{ .ctx = &next, .event = Fixture.event });
    try std.testing.expectEqual(api.Reason.length, try next.wait());
    try std.testing.expectEqual(@as(usize, 3), next.tokens);
    try std.testing.expectEqual(@as(usize, 1), refused.terminals);
    try std.testing.expectEqual(@as(usize, 1), empty.terminals);
    try std.testing.expectEqual(@as(usize, 1), next.terminals);
}

test "a cancel during a long prompt stops it, and the next request's reply is unchanged (#291)" {
    const gpa = std.testing.allocator;
    var target: lanes.fake.Fake = .{ .gpa = gpa, .prefill_chunks = 10 };
    defer target.deinit();
    const Nothing = struct {
        fn release(_: *anyopaque) void {}
    };
    const h = try gpa.create(Host);
    defer gpa.destroy(h);
    h.* = .{ .gpa = gpa, .gpu = null, .family = &target, .release = Nothing.release, .inner = target.backend(), .vtable = undefined, .cfg = undefined, .clock = undefined, .core = undefined, .host = undefined };
    try h.serve(std.testing.io, .{ .exact_width = 8, .mtp = true, .speculate = true, .drafts = 7, .hidden_rows = true }, 8, .{ .lanes = 2 }, null);
    defer {
        h.host.stop();
        h.core.deinit();
        h.cfg.deinit(gpa);
    }
    const e = h.host.engine();
    try std.testing.expect(e.memory(false) == null); // no device to count
    const Box = struct {
        mutex: std.Io.Mutex = .init,
        tokens: std.ArrayList(u32) = .empty,
        done: ?api.Reason = null,
        fn event(ctx: *anyopaque, _: api.Id, ev: *const api.Event) void {
            const b: *@This() = @ptrCast(@alignCast(ctx));
            b.mutex.lockUncancelable(std.testing.io);
            defer b.mutex.unlock(std.testing.io);
            switch (ev.*) {
                .tokens => |t| b.tokens.appendSlice(gpa, t) catch {},
                .finished => |f| b.done = f.reason,
                else => {},
            }
        }
        fn wait(b: *@This()) api.Reason {
            while (true) {
                b.mutex.lockUncancelable(std.testing.io);
                const d = b.done;
                b.mutex.unlock(std.testing.io);
                if (d) |r| return r;
                std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
            }
        }
    };
    const Breaker = struct {
        engine: api.Engine,
        id: api.Id,
        fn call(ctx: *anyopaque, _: *lanes.Stream, chunk: usize) void {
            const c: *@This() = @ptrCast(@alignCast(ctx));
            if (chunk == 2 and c.id != 0) c.engine.cancel(c.id);
        }
    };
    const prompt = [_]u32{ 8, 6, 7, 5, 3, 0, 9, 2, 1, 4 };
    const request: api.Request = .{ .prompt = &prompt, .max_tokens = 40 };
    var breaker: Breaker = .{ .engine = e, .id = 1 };
    target.prefill_hook = Breaker.call;
    target.prefill_hook_ctx = &breaker;
    var broken: Box = .{};
    defer broken.tokens.deinit(gpa);
    try e.submit(1, &request, .{ .ctx = &broken, .event = Box.event });
    try std.testing.expectEqual(api.Reason.cancelled, broken.wait());
    try std.testing.expect(target.prefill_count <= 3); // stopped at the next chunk, not after all ten
    try std.testing.expectEqual(@as(usize, 0), target.lanes.count()); // its lane released
    try std.testing.expectEqual(@as(usize, 0), broken.tokens.items.len);
    breaker.id = 0;
    var restored: Box = .{};
    defer restored.tokens.deinit(gpa);
    try e.submit(2, &request, .{ .ctx = &restored, .event = Box.event });
    try std.testing.expectEqual(api.Reason.length, restored.wait());
    var history: std.ArrayList(u32) = .empty;
    defer history.deinit(gpa);
    try history.appendSlice(gpa, &prompt);
    for (restored.tokens.items) |t| {
        try std.testing.expectEqual(lanes.fake.next(history.items, null, history.items.len), t);
        try history.append(gpa, t);
    }
    try std.testing.expectEqual(@as(usize, 40), restored.tokens.items.len);
}

/// The spill tier for a loaded family, or null with `problem` set: the layout from the engine, the reservation inside
/// the cache, the identity from the files the load read and the facts it ran with.
fn openDisk(comptime F: type, a: Allocator, gpa: Allocator, io: std.Io, g: *Gpu, loaded: F.Loaded, c: *spill_facts.Config, model_files: *const spill_facts.Artifacts, exe_files: *const spill_facts.Artifacts, kernels: ?[]const u8, segments: u32, cache: u64, unified: bool, problem: *[]const u8) !?*nspill.Disk {
    const t = loaded.target;
    // two-GPU TP: each file holds both ranks' shares, which the kept states copy through each rank's own engine
    const ranks: u32 = if (t.e.tp_link != null) 2 else 1;
    const layout = (nspill.Layout.of(F.planes(t), F.prompt_rows) catch |e| {
        problem.* = try std.fmt.allocPrint(a, "spill tier: the engine's state layout is not one the tier stores ({s})", .{@errorName(e)});
        return null;
    }).over(ranks) catch |e| {
        problem.* = try std.fmt.allocPrint(a, "spill tier: the engine's state layout is not one the tier stores ({s})", .{@errorName(e)});
        return null;
    };
    const payload = try layout.payload();
    const slot: u64 = if (nspill.Disk.slotless(payload, c.tier)) 0 else payload;
    const reservation = try nspill.Disk.reservationFor(slot, 64 << 20);
    // a rank's kept state; the cache's budget is a rank's (both hold the same bytes)
    const device_bytes = F.snap_bytes(t, @intCast(layout.max_len));
    // the host reservation comes out of the cache only where host memory is the device's: a discrete GPU's cache is
    // device memory it never takes
    _ = nspill.Disk.admit(slot, device_bytes, c.host_bytes, if (unified) cache else cache +| reservation) catch |e| {
        problem.* = switch (e) {
            error.SpillHostCeiling => try std.fmt.allocPrint(a, "TENSORFOLD_SPILL_HOST_GIB={d:.2} is below the tier's {d:.2} GiB host reservation (one {d:.2} GiB state, two transfer buffers, its index): raise it or lower --context", .{ toGib(c.host_bytes), toGib(reservation), toGib(payload) }),
            error.SpillCacheTooSmall => try std.fmt.allocPrint(a, "the {d:.2} GiB prompt cache cannot hold the spill tier's {d:.2} GiB host reservation and one {d:.2} GiB kept state: raise --prompt-cache-gib or lower --context", .{ toGib(cache), toGib(reservation), toGib(device_bytes) }),
            else => try std.fmt.allocPrint(a, "spill tier: {s}", .{@errorName(e)}),
        };
        return null;
    };
    var driver: c_int = 0;
    g.driver.check(g.driver.api.cuDriverGetVersion(&driver), "cuDriverGetVersion") catch |e| {
        problem.* = try std.fmt.allocPrint(a, "spill tier: no driver version ({s})", .{@errorName(e)});
        return null;
    };
    const weights = model_files.digest(gpa) catch |e| {
        problem.* = try std.fmt.allocPrint(a, "spill tier: the checkpoint's files changed or failed to read while loading ({s})", .{@errorName(e)});
        return null;
    };
    const exe = exe_files.digest(gpa) catch |e| {
        problem.* = try std.fmt.allocPrint(a, "spill tier: the executable or its kernel set changed or failed to read ({s})", .{@errorName(e)});
        return null;
    };
    const capability = g.ctx.capability() catch |e| {
        problem.* = try std.fmt.allocPrint(a, "spill tier: no compute capability ({s})", .{@errorName(e)});
        return null;
    };
    const facts: spill_facts.Facts = .{ .weights = weights, .executable = exe, .kernels = kernels orelse "embedded", .driver = driver, .capability = capability, .format = F.formatName(t), .planes = .{ layout.kv_row, layout.mamba, 0, 0, 0, 0, 0 }, .head = @intFromBool(layout.head), .max_len = layout.max_len, .grid = layout.grid, .lookahead = nspill.lookahead, .segments = @intCast(segments), .kv = if (t.e.c.kv_fp8) "fp8-e4m3" else "bf16", .tp = ranks };
    const identity = try facts.identity();
    const d = nspill.Disk.open(gpa, io, gpa, .{ .ctx = &g.ctx, .stream = t.e.stream }, layout, c.tier, identity, c.host_bytes) catch |e| {
        problem.* = try std.fmt.allocPrint(a, "spill tier at {s}: {s}", .{ c.tier.dir, @errorName(e) });
        return null;
    };
    d.charged = unified;
    std.log.info("spill tier: {s}, {d:.1} GiB on disk, {d:.1} GiB in host memory, {d} rank{s} a state, {s}", .{ c.tier.dir, toGib(c.tier.cap), toGib(c.tier.ram_cap), ranks, if (ranks == 1) "" else "s", if (slot == 0) "captured straight into memory" else "through a host slot" });
    return d;
}

test "family readiness reaches production lane admission without binding CUDA" {
    try @import("cuda_health_test.zig").check(Host, Host.readiness, Host.serve);
}

test "Flash Next refuses tensor-rank draft caching before files or CUDA" {
    try @import("cuda_flashnext_refusal_test.zig").cache(openWith);
}

test "Flash Next takes --vision at family admission" {
    try @import("cuda_flashnext_refusal_test.zig").vision();
}

test {
    _ = @import("cuda_vision_offer.zig");
    _ = kv_pages;
    _ = drafted;
}
