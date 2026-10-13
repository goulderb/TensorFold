//! GET /health, /v1/models, /metrics, and the opt-in dashboard.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const live = @import("live.zig");
const routes = @import("routes.zig");
const Server = @import("server.zig").Server;
const Conn = @import("http_conn.zig").Conn;
const Value = json.Value;
const Allocator = std.mem.Allocator;

/// With keys on, health names nothing about the model.
pub fn health(srv: *Server, conn: *Conn, a: Allocator) !void {
    if (srv.keys) |k| if (k.enabled()) return conn.sendJson(200, "{\"status\": \"ok\"}");
    routes.sendValue(conn, a, 200, try openHealth(srv, a, std.mem.indexOf(u8, conn.path, "reset_peak=1") != null));
}

/// The open health body: model, lanes, memory and live counts, then whichever of release, backend and GPU it knows.
pub fn openHealth(srv: *Server, a: Allocator, reset_peak: bool) !Value {
    const o = try json.newObject(a);
    try o.put(a, "status", .{ .string = "ok" });
    try o.put(a, "model", .{ .string = srv.config.served_name });
    const ids = try a.alloc(Value, srv.config.model_ids.len);
    for (srv.config.model_ids, ids) |id, *slot| slot.* = .{ .string = id };
    try o.put(a, "model_ids", .{ .array = ids });
    try o.put(a, "max_batch_size", try json.intValue(a, srv.info.lanes));
    if (srv.info.policy.len > 0) try o.put(a, "policy", .{ .string = srv.info.policy });
    var status: api.Status = .{};
    srv.engine.status(&status, &.{});
    try o.put(a, "warming", .{ .bool = status.warming });
    const memory = try json.newObject(a);
    if (srv.engine.memory(reset_peak)) |mem| {
        try memory.put(a, "active", try json.intValue(a, mem.active));
        try memory.put(a, "cache", try json.intValue(a, mem.cache));
        try memory.put(a, "peak", try json.intValue(a, mem.peak));
        if (mem.limit) |limit| try memory.put(a, "limit", try json.intValue(a, limit));
    }
    try o.put(a, "memory", .{ .object = memory });
    try o.put(a, "live", try live.snapshot(a, srv.engine));
    const ident = srv.config.identity;
    for ([_][2][]const u8{
        .{ "version", ident.version },
        .{ "backend", ident.backend },
        .{ "device", ident.device },
    }) |field|
        if (field[1].len > 0) try o.put(a, field[0], .{ .string = field[1] });
    if (status.prefill) |pf| {
        const p = try json.newObject(a);
        try p.put(a, "submitted_rows", try json.intValue(a, pf.submitted_rows));
        try p.put(a, "completed_rows", try json.intValue(a, pf.completed_rows));
        try p.put(a, "passes", try json.intValue(a, pf.passes));
        try o.put(a, "prefill", .{ .object = p });
    }
    var sp: api.SpillStatus = undefined;
    if (srv.engine.spill(&sp)) try o.put(a, "spill", try spillValue(a, sp));
    return .{ .object = o };
}

/// The spill tier's sizes and stats and the prompt cache's resume outcomes, as /health shows them.
pub fn spillValue(a: Allocator, sp: api.SpillStatus) !Value {
    const o = try json.newObject(a);
    try o.put(a, "entries", try json.intValue(a, sp.tier.entries));
    try o.put(a, "pending", try json.intValue(a, sp.tier.pending));
    try o.put(a, "stored_bytes", try json.intValue(a, sp.tier.stored_bytes));
    try o.put(a, "queued_bytes", try json.intValue(a, sp.tier.queued_bytes));
    try o.put(a, "cap_bytes", try json.intValue(a, sp.tier.cap));
    try o.put(a, "disk_free_bytes", try json.intValue(a, sp.tier.disk_free));
    inline for (@typeInfo(@TypeOf(sp.tier.stats)).@"struct".field_names) |name| try o.put(a, name, try json.intValue(a, @field(sp.tier.stats, name)));
    const c = try json.newObject(a);
    inline for (@typeInfo(@TypeOf(sp.cache)).@"struct".field_names) |name| try c.put(a, name, try json.intValue(a, @field(sp.cache, name)));
    try o.put(a, "resume", .{ .object = c });
    const w = try json.newObject(a);
    inline for (@typeInfo(@TypeOf(sp.writes)).@"struct".field_names) |name| try w.put(a, name, try json.intValue(a, @field(sp.writes, name)));
    try o.put(a, "writes", .{ .object = w });
    try o.put(a, "age_ms", try json.intValue(a, sp.age_ms));
    return .{ .object = o };
}

test "the spill status names the tier's stats and the cache's resume outcomes" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var sp: api.SpillStatus = .{ .tier = .{ .stats = .{ .loaded_tokens = 400, .load_failed = 1 }, .entries = 2, .pending = 0, .stored_bytes = 9000, .queued_bytes = 0, .disk_free = 1 << 30, .cap = 1 << 31 } };
    sp.cache.recalled_tokens = 400;
    sp.cache.recall_failed = 1;
    const v = try spillValue(a, sp);
    try std.testing.expectEqual(@as(?i64, 400), v.get("loaded_tokens").?.int64());
    try std.testing.expectEqual(@as(?i64, 1), v.get("load_failed").?.int64());
    try std.testing.expectEqual(@as(?i64, 2), v.get("entries").?.int64());
    const r = v.get("resume").?;
    try std.testing.expectEqual(@as(?i64, 400), r.get("recalled_tokens").?.int64());
    try std.testing.expectEqual(@as(?i64, 1), r.get("recall_failed").?.int64());
    sp.writes.skipped_busy = 4;
    sp.age_ms = 12;
    const v2 = try spillValue(a, sp);
    try std.testing.expectEqual(@as(?i64, 4), v2.get("writes").?.get("skipped_busy").?.int64());
    try std.testing.expectEqual(@as(?i64, 12), v2.get("age_ms").?.int64());
}

pub fn models(srv: *Server, conn: *Conn, a: Allocator) !void {
    const created = std.Io.Clock.real.now(srv.io).toSeconds();
    const data = try a.alloc(Value, srv.config.model_ids.len);
    const thinks = srv.thinking_switch or srv.effort_levels.len > 0 or srv.think_close.len > 0;
    const reasoning: ?Value = if (thinks)
        try reasoningValue(a, srv.config.enable_thinking, srv.thinking_switch, srv.effortFor(null), srv.effort_levels)
    else
        null;
    for (srv.config.model_ids, data) |id, *slot| {
        slot.* = try modelValue(a, id, srv.info, created, reasoning);
    }
    const o = try json.newObject(a);
    try o.put(a, "object", .{ .string = "list" });
    try o.put(a, "data", .{ .array = data });
    routes.sendValue(conn, a, 200, .{ .object = o });
}

/// Discovery extensions report the loaded engine's prompt-plus-reply limit; zero leaves it unspecified.
fn modelValue(a: Allocator, id: []const u8, info: api.Info, created: i64, reasoning: ?Value) Allocator.Error!Value {
    const m = try json.newObject(a);
    try m.put(a, "id", .{ .string = id });
    try m.put(a, "object", .{ .string = "model" });
    try m.put(a, "created", try json.intValue(a, created));
    try m.put(a, "owned_by", .{ .string = "tensorfold" });
    if (info.context_window > 0) {
        try m.put(a, "context_length", try json.intValue(a, info.context_window));
        try m.put(a, "max_model_len", try json.intValue(a, info.context_window));
    }
    if (reasoning) |r| try m.put(a, "reasoning", r);
    return .{ .object = m };
}

/// /v1/models' ``reasoning``: the thinking default and the template's levels.
fn reasoningValue(
    a: Allocator,
    enable: bool,
    switchable: bool,
    default: ?[]const u8,
    levels: []const []const u8,
) Allocator.Error!Value {
    const r = try json.newObject(a);
    try r.put(a, "default_enabled", .{ .bool = enable });
    if (default) |e| try r.put(a, "default_effort", .{ .string = e });
    if (levels.len > 0) {
        const named = try a.alloc(Value, levels.len + @intFromBool(switchable));
        for (levels, named[0..levels.len]) |l, *slot| slot.* = .{ .string = l };
        if (switchable) named[levels.len] = .{ .string = "none" };
        try r.put(a, "supported_efforts", .{ .array = named });
    }
    return .{ .object = r };
}

/// Prometheus text, version 0.0.4.
pub fn metrics(srv: *Server, conn: *Conn, a: Allocator) void {
    var out: std.Io.Writer.Allocating = .init(a);
    srv.metrics.render(srv.io, &out.writer, srv.engine, srv.info.context_window) catch return;
    const body = out.written();
    conn.startResponse(200, null) catch return;
    conn.addHeader("Content-Type", "text/plain; version=0.0.4; charset=utf-8") catch return;
    var len: [24]u8 = undefined;
    conn.addHeader("Content-Length", std.fmt.bufPrint(&len, "{d}", .{body.len}) catch return) catch return;
    conn.finish(body) catch {};
}

pub fn dashboard(srv: *Server, conn: *Conn, a: Allocator) void {
    if (!srv.config.dashboard) return routes.unknown(conn, a);
    conn.startResponse(200, null) catch return;
    conn.addHeader("Content-Type", "text/html; charset=utf-8") catch return;
    var len: [24]u8 = undefined;
    conn.addHeader("Content-Length", std.fmt.bufPrint(&len, "{d}", .{dashboard_page.len}) catch return) catch return;
    conn.finish(dashboard_page) catch {};
}

pub fn stats(srv: *Server, conn: *Conn, a: Allocator) void {
    if (!srv.config.dashboard) return routes.unknown(conn, a);
    var status: api.Status = .{};
    srv.engine.status(&status, &.{});
    const memory = srv.engine.memory(false);
    srv.metrics.mutex.lockUncancelable(srv.io);
    const prompt = srv.metrics.prompt;
    const generation = srv.metrics.generation;
    const drafted = srv.metrics.drafted;
    const accepted = srv.metrics.accepted;
    const rounds = srv.metrics.rounds;
    srv.metrics.mutex.unlock(srv.io);
    const body = statsSnapshot(a, status, memory, prompt, generation, drafted, accepted, rounds) catch return;
    conn.quiet_log = true;
    routes.sendValue(conn, a, 200, body);
}

pub fn statsSnapshot(a: Allocator, status: api.Status, memory: ?api.Memory, prompt: u64, generation: u64, drafted: u64, accepted: u64, rounds: u64) Allocator.Error!Value {
    const o = try json.newObject(a);
    try o.put(a, "running", try json.intValue(a, status.running));
    try o.put(a, "waiting", try json.intValue(a, status.waiting));
    try o.put(a, "decode_tokens_per_second", .{ .float = status.decode_tokens_per_second });
    try o.put(a, "prefill_tokens_per_second", .{ .float = status.prefill_tokens_per_second });
    try o.put(a, "generation_tokens_running", try json.intValue(a, status.generation_tokens));
    if (status.fill_prompt_tokens != null) {
        try o.put(a, "fill_prompt_tokens", try json.intValue(a, status.fill_prompt_tokens.?));
        try o.put(a, "fill_tokens_left", try json.intValue(a, status.fill_tokens_left orelse 0));
        try o.put(a, "fill_run", try json.intValue(a, status.fill_run));
    } else if (status.fill_unknown) {
        try o.put(a, "fill_unknown", .{ .bool = true }); // prefilling, but this backend reports no progress
        try o.put(a, "fill_run", try json.intValue(a, status.fill_run));
    }
    try o.put(a, "prompt_tokens_total", try json.intValue(a, prompt));
    try o.put(a, "generation_tokens_total", try json.intValue(a, generation));
    try o.put(a, "mtp_drafted_total", try json.intValue(a, drafted));
    try o.put(a, "mtp_accepted_total", try json.intValue(a, accepted));
    try o.put(a, "decode_rounds_total", try json.intValue(a, rounds));
    const mem = try json.newObject(a);
    if (memory) |m| {
        try mem.put(a, "active", try json.intValue(a, m.active));
        try mem.put(a, "cache", try json.intValue(a, m.cache));
        try mem.put(a, "peak", try json.intValue(a, m.peak));
        if (m.limit) |limit| try mem.put(a, "limit", try json.intValue(a, limit));
    }
    try o.put(a, "memory", .{ .object = mem });
    return .{ .object = o };
}

const dashboard_page = @embedFile("dashboard.html");

test "model discovery reports fitted engine context equally for its served name and aliases" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const info: api.Info = .{ .context_window = 8192, .context_fitted = true };
    for ([_][]const u8{ "served-model", "model-alias" }) |id| {
        const value = try modelValue(arena.allocator(), id, info, 123, null);
        try std.testing.expectEqualStrings(id, value.get("id").?.string);
        try std.testing.expectEqual(@as(i64, 8192), value.get("context_length").?.int64().?);
        try std.testing.expectEqual(value.get("context_length").?.int64(), value.get("max_model_len").?.int64());
        try std.testing.expectEqual(@as(i64, 123), value.get("created").?.int64().?);
    }
}

test "model discovery omits an unspecified context and preserves the full u32 limit" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const unlimited = try modelValue(arena.allocator(), "model", .{}, 0, null);
    try std.testing.expect(unlimited.get("context_length") == null);
    try std.testing.expect(unlimited.get("max_model_len") == null);
    const bounded = try modelValue(arena.allocator(), "model", .{ .context_window = std.math.maxInt(u32) }, 0, null);
    try std.testing.expectEqual(@as(i64, std.math.maxInt(u32)), bounded.get("max_model_len").?.int64().?);
}

test "model discovery adds reasoning to an entry only when the model has it" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const plain = try modelValue(a, "model", .{}, 0, null);
    try std.testing.expect(plain.get("reasoning") == null);
    const thinking = try modelValue(a, "model", .{}, 0, try reasoningValue(a, true, true, null, &.{"low"}));
    try std.testing.expect(thinking.get("reasoning").?.get("default_enabled").?.bool);
}

test "model discovery lists the template's levels, and none only behind a thinking switch" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const qwen = try reasoningValue(a, true, true, null, &.{ "xhigh", "medium", "low" });
    const listed = qwen.get("supported_efforts").?.array;
    try std.testing.expectEqual(@as(usize, 4), listed.len);
    const want = [_][]const u8{ "xhigh", "medium", "low", "none" };
    for (want, listed) |name, v| try std.testing.expectEqualStrings(name, v.string);
    const glm = try reasoningValue(a, true, false, null, &.{ "max", "high", "low" });
    try std.testing.expectEqual(@as(usize, 3), glm.get("supported_efforts").?.array.len);
    const bare = try reasoningValue(a, true, true, null, &.{});
    try std.testing.expect(bare.get("supported_efforts") == null);
}

test "model discovery names a default effort only when the server was started with one" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const levels = [_][]const u8{ "xhigh", "medium", "low" };
    const plain = try reasoningValue(a, true, true, null, &levels);
    try std.testing.expect(plain.get("default_effort") == null);
    try std.testing.expect(plain.get("default_enabled").?.bool);
    const started = try reasoningValue(a, false, true, "low", &levels);
    try std.testing.expectEqualStrings("low", started.get("default_effort").?.string);
    try std.testing.expect(!started.get("default_enabled").?.bool);
}

test "stats snapshot exposes counters and omits unknown memory" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const value = try statsSnapshot(arena.allocator(), .{ .running = 1, .generation_tokens = 2 }, null, 3, 4, 5, 6, 7);
    const object = value.object;
    for ([_][]const u8{ "running", "waiting", "generation_tokens_running", "prompt_tokens_total", "generation_tokens_total", "mtp_drafted_total", "mtp_accepted_total", "decode_rounds_total", "memory" }) |key|
        try std.testing.expect(object.contains(key));
    try std.testing.expectEqual(@as(usize, 0), object.get("memory").?.object.count());
    try std.testing.expect(object.get("fill_prompt_tokens") == null); // no fill in progress: the keys stay out
}

test "stats snapshot carries the filling prompt's tokens while a prefill runs" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const status: api.Status = .{ .fill_prompt_tokens = 900, .fill_tokens_left = 300 };
    const value = try statsSnapshot(arena.allocator(), status, null, 0, 0, 0, 0, 0);
    const object = value.object;
    try std.testing.expectEqual(@as(i64, 900), object.get("fill_prompt_tokens").?.int64().?);
    try std.testing.expectEqual(@as(i64, 300), object.get("fill_tokens_left").?.int64().?);
}
test "stats snapshot marks an unknown-progress fill without counters" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const status: api.Status = .{ .fill_unknown = true, .fill_run = 7 };
    const value = try statsSnapshot(arena.allocator(), status, null, 0, 0, 0, 0, 0);
    const object = value.object;
    try std.testing.expect(object.get("fill_unknown").?.bool);
    try std.testing.expectEqual(@as(i64, 7), object.get("fill_run").?.int64().?);
    try std.testing.expect(object.get("fill_prompt_tokens") == null); // no counters: the keys stay out
}

test "stats snapshot's memory carries a configured limit and omits none" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const with = try statsSnapshot(
        arena.allocator(),
        .{},
        .{ .active = 5, .cache = 1, .peak = 9, .pinned = 2, .limit = 100 },
        0,
        0,
        0,
        0,
        0,
    );
    const memory = with.object.get("memory").?.object;
    try std.testing.expectEqual(@as(i64, 5), memory.get("active").?.int64().?);
    try std.testing.expectEqual(@as(i64, 9), memory.get("peak").?.int64().?);
    try std.testing.expectEqual(@as(i64, 100), memory.get("limit").?.int64().?);
    const without = try statsSnapshot(arena.allocator(), .{}, .{ .active = 5, .peak = 9 }, 0, 0, 0, 0, 0);
    // no limit configured: the key stays out
    try std.testing.expect(without.object.get("memory").?.object.get("limit") == null);
}
