//! Prometheus text for GET /metrics, family for family as the Python server writes it.
const std = @import("std");
const process = @import("process_memory.zig");
const api = @import("engine_api");

const prefix = "tensorfold:";
/// Upper edges shared by the request and time-to-first-token histograms; +Inf is added when rendered.
pub const buckets = [_]f64{ 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0 };
pub const tpot_buckets = [_]f64{ 0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0 };

fn HistogramType(comptime edges: []const f64) type {
    return struct {
        pub const bucket_edges = edges;
        counts: [edges.len + 1]u64 = @splat(0),
        total: f64 = 0,
        n: u64 = 0,

        pub fn observe(h: *@This(), raw: f64) void {
            const value = @max(0, raw);
            h.n += 1;
            h.total += value;
            for (edges, 0..) |edge, i| if (value <= edge) {
                h.counts[i] += 1;
                return;
            };
            h.counts[edges.len] += 1;
        }
    };
}

pub const Histogram = HistogramType(&buckets);
pub const TpotHistogram = HistogramType(&tpot_buckets);

/// Counters of finished requests; gauges are read from the engine at scrape time.
pub const Metrics = struct {
    gpa: std.mem.Allocator,
    mutex: std.Io.Mutex = .init,
    prompt: u64 = 0,
    cached: u64 = 0,
    generation: u64 = 0,
    drafted: u64 = 0,
    accepted: u64 = 0,
    rounds: u64 = 0,
    disconnects: u64 = 0,
    latency: Histogram = .{},
    ttft: Histogram = .{},
    decode: Histogram = .{},
    prefill: Histogram = .{},
    tpot: TpotHistogram = .{},
    requests: std.ArrayList(struct { key: []const u8, status: u16, count: u64 }) = .empty,

    pub fn deinit(m: *Metrics) void {
        for (m.requests.items) |r| m.gpa.free(r.key);
        m.requests.deinit(m.gpa);
    }

    pub fn note(m: *Metrics, io: std.Io, prompt: usize, cached: usize, generation: usize, drafted: u64, accepted: u64, rounds: u64, latency: f64, ttft: ?f64, decode: ?f64, prefill: ?f64, tpot: ?f64) void {
        m.mutex.lockUncancelable(io);
        defer m.mutex.unlock(io);
        m.prompt += prompt;
        m.cached += cached;
        m.generation += generation;
        m.drafted += drafted;
        m.accepted += accepted;
        m.rounds += rounds;
        m.latency.observe(latency);
        if (ttft) |t| m.ttft.observe(t);
        if (decode) |d| m.decode.observe(d);
        if (prefill) |p| m.prefill.observe(p);
        if (tpot) |t| m.tpot.observe(t);
    }

    pub fn tpotValue(_: *const Metrics, first: ?i96, last: ?i96, generated: usize) ?f64 {
        if (generated < 2 or first == null or last == null) return null;
        return @max(0, @as(f64, @floatFromInt(last.? - first.?)) / 1e9) / @as(f64, @floatFromInt(generated - 1));
    }

    /// One HTTP reply under its key label, counted once per request.
    pub fn httpRequest(m: *Metrics, io: std.Io, key: []const u8, status: u16) void {
        m.mutex.lockUncancelable(io);
        defer m.mutex.unlock(io);
        for (m.requests.items) |*r| if (r.status == status and std.mem.eql(u8, r.key, key)) {
            r.count += 1;
            return;
        };
        const owned = m.gpa.dupe(u8, key) catch return;
        m.requests.append(m.gpa, .{ .key = owned, .status = status, .count = 1 }) catch m.gpa.free(owned);
    }

    pub fn disconnected(m: *Metrics, io: std.Io) void {
        m.mutex.lockUncancelable(io);
        defer m.mutex.unlock(io);
        m.disconnects += 1;
    }

    /// The scrape body, ending in a newline.
    pub fn render(m: *Metrics, io: std.Io, w: *std.Io.Writer, engine: api.Engine, window: u32) !void {
        var status: api.Status = .{};
        var streams: [512]u32 = undefined;
        engine.status(&status, &streams);
        var snap: Metrics = undefined;
        const requests = blk: {
            m.mutex.lockUncancelable(io);
            defer m.mutex.unlock(io);
            snap = m.*;
            break :blk try m.gpa.dupe(@TypeOf(m.requests.items[0]), m.requests.items);
        };
        defer m.gpa.free(requests);
        std.mem.sort(@TypeOf(requests[0]), requests, {}, struct {
            fn less(_: void, x: @TypeOf(requests[0]), y: @TypeOf(requests[0])) bool {
                const o = std.mem.order(u8, x.key, y.key);
                return o == .lt or (o == .eq and x.status < y.status);
            }
        }.less);
        if (requests.len > 0) {
            try family(w, "requests_total", "counter", "HTTP replies by API key label and status.");
            for (requests) |r| try w.print("{s}requests_total{{key=\"{s}\",status=\"{d}\"}} {d}\n", .{ prefix, r.key, r.status, r.count });
        }
        try gauge(w, "requests_running", "gauge", "Requests in prefill or decode.", status.running);
        try gauge(w, "requests_waiting", "gauge", "Requests queued or held until a lane is free.", status.waiting);
        try gauge(w, "generation_tokens_running", "gauge", "Generated tokens held by live streams.", status.generation_tokens);
        try gauge(w, "prompt_tokens_total", "counter", "Prompt tokens of finished requests.", snap.prompt);
        const keeps = engine.info().prompt_cache;
        if (keeps) try gauge(w, "prompt_tokens_cached_total", "counter", "Prompt tokens of finished requests restored from kept prompt states.", snap.cached);
        try gauge(w, "generation_tokens_total", "counter", "Generated tokens of finished requests.", snap.generation);
        const live = streams[0..@min(status.streams, streams.len)];
        try family(w, "kv_cache_usage_ratio", "gauge", "Tokens in a stream cache divided by that stream's context window.");
        try pools(w, "kv_cache_usage_ratio", "pool", live, window);
        try gauge(w, "mtp_drafted_total", "counter", "Draft tokens verified on finished requests.", snap.drafted);
        try gauge(w, "mtp_accepted_total", "counter", "Draft tokens kept on finished requests.", snap.accepted);
        try histogram(w, "request_latency_seconds", "Seconds from arrival to the reply leaving.", snap.latency);
        try histogram(w, "time_to_first_token_seconds", "Seconds from arrival to the first generated token.", snap.ttft);
        try histogram(w, "request_decode_seconds", "Seconds a finished request spent decoding. Its sum over generation_tokens_total is the decode rate.", snap.decode);
        try histogram(w, "request_time_per_output_token_seconds", "Seconds from the first generated token to the last, divided by the tokens less one.", snap.tpot);
        try gauge(w, "decode_rounds_total", "counter", "Decode rounds used by finished requests.", snap.rounds);
        try histogram(w, "request_prefill_seconds", "Seconds a finished request's prompt pass took.", snap.prefill);
        try processGauges(w, process.read());
        if (engine.memory(false)) |mem| {
            try gauge(w, "device_memory_bytes", "gauge", "Device memory the engine holds now: weights, caches and live streams; only where the backend counts it (CUDA).", mem.active);
            try gauge(w, "device_memory_peak_bytes", "gauge", "The most device memory the engine has held since start or the last /health?reset_peak=1.", mem.peak);
            try gauge(w, "pinned_memory_bytes", "gauge", "Pinned host memory the engine holds now: staging buffers and windows; only where the backend counts it (CUDA).", mem.pinned);
        }
        try gauge(w, "num_requests_running", "gauge", "Requests in prefill or decode. A mirror of tensorfold:requests_running.", status.running);
        try gauge(w, "num_requests_waiting", "gauge", "Requests queued or held until a lane is free. A mirror of tensorfold:requests_waiting.", status.waiting);
        try family(w, "kv_cache_usage_perc", "gauge", "A stream's cache occupancy under vLLM's name; same streams and ratios as tensorfold:kv_cache_usage_ratio.");
        try pools(w, "kv_cache_usage_perc", "stream", live, window);
        try gauge(w, "spec_decode_num_draft_tokens_total", "counter", "Draft tokens verified on finished requests, this server's single draft counter.", snap.drafted);
        try gauge(w, "spec_decode_num_accepted_tokens_total", "counter", "Draft tokens kept on finished requests.", snap.accepted);
        if (keeps) {
            try gauge(w, "prefix_cache_queries_total", "counter", "Prompt tokens of finished requests, looked up in the kept prompt states, under vLLM's name.", snap.prompt);
            try gauge(w, "prefix_cache_hits_total", "counter", "Prompt tokens of finished requests restored from kept prompt states, under vLLM's name.", snap.cached);
        }
        try histogram(w, "e2e_request_latency_seconds", "Seconds from arrival to the reply leaving, under vLLM's name.", snap.latency);
        try histogram(w, "request_decode_time_seconds", "Seconds a finished request spent decoding, under vLLM's name.", snap.decode);
        try histogram(w, "request_prefill_time_seconds", "Seconds a finished request's prompt pass took, under vLLM's name.", snap.prefill);
        try gauge(w, "client_disconnections_total", "counter", "Requests a client left before the reply left the server.", snap.disconnects);
        if (status.preemptions) |p| try gauge(w, "preemptions_total", "counter", "Requests that had to give a lane up to a later one.", p);
        if (status.shared_prefix_waits) |n| try gauge(w, "shared_prefix_waits_total", "counter", "Distinct requests that waited for a compatible in-flight prefix.", n);
        if (status.prefill) |pf| {
            try gauge(w, "prefill_rows_submitted_total", "counter", "Prompt rows the primary model queued for computation (resumed rows excluded).", pf.submitted_rows);
            try gauge(w, "prefill_rows_completed_total", "counter", "Prompt rows of prompt passes the primary model finished (resumed rows excluded).", pf.completed_rows);
            try gauge(w, "prefill_passes_total", "counter", "Prompt passes the primary model finished.", pf.passes);
        }
        var sp: api.SpillStatus = undefined;
        if (engine.spill(&sp)) {
            try gauge(w, "spill_disk_read_states_total", "counter", "Kept states read back from the spill tier into memory (raw reads; a later restore may fail).", sp.cache.read_back);
            try gauge(w, "spill_disk_read_tokens_total", "counter", "Prompt tokens of spill files read back, before any restore commits.", sp.tier.stats.loaded_tokens);
            try gauge(w, "spill_ram_bytes", "gauge", "Bytes of kept states the spill tier holds in host memory.", sp.tier.ram_bytes);
            try gauge(w, "spill_ram_entries", "gauge", "Kept states the spill tier holds in host memory.", sp.tier.ram_entries);
            try gauge(w, "spill_ram_read_states_total", "counter", "Reads the spill tier served from host memory, not disk.", sp.tier.stats.ram_loaded);
            try gauge(w, "spill_ram_demoted_total", "counter", "Held states written on to disk for host-memory room.", sp.tier.stats.ram_demoted);
            try gauge(w, "spill_ram_evicted_total", "counter", "RAM-only tier: held states that left for host-memory room (no disk level keeps them).", sp.tier.stats.ram_evicted);
            try gauge(w, "spill_restored_states_total", "counter", "Read-back states whose restore committed and a prompt pass resumed from.", sp.cache.recalled);
            try gauge(w, "spill_restored_tokens_total", "counter", "Prompt tokens committed restores from the spill tier covered.", sp.cache.recalled_tokens);
            try gauge(w, "spill_write_submitted_total", "counter", "States handed to the spill tier's writer (not proof of persistence).", sp.writes.submitted);
            try gauge(w, "spill_write_written_total", "counter", "Submitted states the tier finished writing.", sp.writes.written);
            try gauge(w, "spill_write_lost_total", "counter", "Submitted states the tier gave up.", sp.writes.lost);
            try gauge(w, "spill_write_skipped_total", "counter", "Spill opportunities skipped while the adapter's slot was busy.", sp.writes.skipped_busy);
            try gauge(w, "spill_write_failed_total", "counter", "Spill writes the family refused or failed.", sp.cache.write_failed);
            try gauge(w, "spill_recall_failed_total", "counter", "Stored states that would not read back (forgotten).", sp.cache.recall_failed);
            try gauge(w, "spill_unavailable_total", "counter", "Stored states skipped because a copy was still being written.", sp.cache.unavailable);
            try gauge(w, "spill_status_age_ms", "gauge", "Milliseconds since the engine thread took this spill snapshot.", sp.age_ms);
        }
    }
};

fn family(w: *std.Io.Writer, name: []const u8, kind: []const u8, help: []const u8) !void {
    try w.print("# HELP {s}{s} {s}\n# TYPE {s}{s} {s}\n", .{ prefix, name, help, prefix, name, kind });
}

fn gauge(w: *std.Io.Writer, name: []const u8, kind: []const u8, help: []const u8, value: u64) !void {
    try family(w, name, kind, help);
    try w.print("{s}{s} {d}\n", .{ prefix, name, value });
}

fn processGauges(w: *std.Io.Writer, sample: ?process.Snapshot) !void {
    const p = sample orelse return;
    try gauge(w, "process_footprint_bytes", "gauge", "Current OS physical footprint including Metal buffers (macOS).", p.physical_footprint_bytes);
    try gauge(w, "process_footprint_peak_bytes", "gauge", "Kernel lifetime peak physical footprint including Metal buffers; never reset by health (macOS).", p.lifetime_peak_physical_footprint_bytes);
}

test "process gauges omit unknown samples and keep current and lifetime peak distinct" {
    var out: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer out.deinit();
    try processGauges(&out.writer, null);
    try std.testing.expectEqual(@as(usize, 0), out.written().len);
    try processGauges(&out.writer, .{ .physical_footprint_bytes = 0, .lifetime_peak_physical_footprint_bytes = 12 });
    try std.testing.expect(std.mem.indexOf(u8, out.written(), "tensorfold:process_footprint_bytes 0\n") != null);
    try std.testing.expect(std.mem.indexOf(u8, out.written(), "tensorfold:process_footprint_peak_bytes 12\n") != null);
}

fn pools(w: *std.Io.Writer, name: []const u8, label: []const u8, lengths: []const u32, window: u32) !void {
    if (lengths.len == 0) return w.print("{s}{s}{{{s}=\"0\"}} 0\n", .{ prefix, name, label });
    for (lengths, 0..) |n, i| {
        const ratio: f64 = if (window == 0) 0 else @min(1.0, @as(f64, @floatFromInt(n)) / @as(f64, @floatFromInt(window)));
        var buf: [40]u8 = undefined;
        try w.print("{s}{s}{{{s}=\"{d}\"}} {s}\n", .{ prefix, name, label, i, num(&buf, ratio) });
    }
}

fn histogram(w: *std.Io.Writer, name: []const u8, help: []const u8, h: anytype) !void {
    try w.print("# HELP {s}{s} {s}\n# TYPE {s}{s} histogram\n", .{ prefix, name, help, prefix, name });
    var cumulative: u64 = 0;
    const edges = @TypeOf(h).bucket_edges;
    for (edges, 0..) |edge, i| {
        cumulative += h.counts[i];
        var buf: [40]u8 = undefined;
        try w.print("{s}{s}_bucket{{le=\"{s}\"}} {d}\n", .{ prefix, name, trimmed(&buf, edge, 4), cumulative });
    }
    try w.print("{s}{s}_bucket{{le=\"+Inf\"}} {d}\n", .{ prefix, name, cumulative + h.counts[edges.len] });
    var buf: [40]u8 = undefined;
    try w.print("{s}{s}_sum {s}\n{s}{s}_count {d}\n", .{ prefix, name, num(&buf, h.total), prefix, name, h.n });
}

/// ``_num``: an integral value as an int, else six decimals without trailing zeros.
fn num(buf: []u8, value: f64) []const u8 {
    if (value == @trunc(value) and @abs(value) < 1e18) return std.fmt.bufPrint(buf, "{d}", .{@as(i64, @intFromFloat(value))}) catch "0";
    return trimmed(buf, value, 6);
}

fn trimmed(buf: []u8, value: f64, comptime places: u8) []const u8 {
    const text = std.fmt.bufPrint(buf, "{d:." ++ std.fmt.comptimePrint("{d}", .{places}) ++ "}", .{value}) catch return "0";
    return std.mem.trimEnd(u8, std.mem.trimEnd(u8, text, "0"), ".");
}

test "edges and numbers" {
    var buf: [40]u8 = undefined;
    try std.testing.expectEqualStrings("0.01", trimmed(&buf, 0.01, 4));
    try std.testing.expectEqualStrings("120", trimmed(&buf, 120.0, 4));
    try std.testing.expectEqualStrings("0.333333", num(&buf, 1.0 / 3.0));
    try std.testing.expectEqualStrings("2", num(&buf, 2.0));
}

test "metrics record rounds prefill and per-token timing" {
    const gpa = std.testing.allocator;
    var m: Metrics = .{ .gpa = gpa };
    m.note(std.testing.io, 10, 0, 3, 0, 0, 2, 1.0, null, null, 0.25, m.tpotValue(1_000_000_000, 1_024_000_000, 3));
    try std.testing.expectEqual(@as(u64, 2), m.rounds);
    try std.testing.expectEqual(@as(u64, 1), m.prefill.n);
    try std.testing.expectEqual(@as(u64, 1), m.tpot.n);
    try std.testing.expectEqual(@as(u64, 1), m.prefill.counts[3]);
    try std.testing.expectEqual(@as(u64, 1), m.tpot.counts[4]);
    var out: std.Io.Writer.Allocating = .init(gpa);
    defer out.deinit();
    try histogram(&out.writer, "request_prefill_seconds", "prefill", m.prefill);
    try std.testing.expect(std.mem.indexOf(u8, out.written(), "request_prefill_seconds_sum 0.25") != null);
}

const RenderStub = struct {
    pub fn info(ctx: *anyopaque) api.Info {
        const s: *@This() = @ptrCast(@alignCast(ctx));
        return s.info_value;
    }
    pub fn submit(_: *anyopaque, _: api.Id, _: *const api.Request, _: api.Sink) api.SubmitError!void {
        unreachable;
    }
    pub fn cancel(_: *anyopaque, _: api.Id) void {}
    pub fn status(ctx: *anyopaque, out: *api.Status, _: []u32) void {
        const s: *@This() = @ptrCast(@alignCast(ctx));
        out.* = s.status_value;
    }
    pub fn memory(ctx: *anyopaque, _: bool) ?api.Memory {
        const s: *@This() = @ptrCast(@alignCast(ctx));
        return s.memory_value;
    }
    info_value: api.Info = .{},
    status_value: api.Status = .{ .running = 1, .waiting = 2, .generation_tokens = 3 },
    memory_value: ?api.Memory = null,

    pub fn engine(s: *@This()) api.Engine {
        return .{ .ctx = s, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }
};

test "render exposes live counters, rounds, prefill and TPOT without cache family" {
    const gpa = std.testing.allocator;
    var m: Metrics = .{ .gpa = gpa };
    m.note(std.testing.io, 10, 0, 3, 0, 0, 2, 1.0, null, null, 0.25, m.tpotValue(1_000_000_000, 3_000_000_000, 3));
    var stub: RenderStub = .{};
    stub.status_value.shared_prefix_waits = 9;
    var out: std.Io.Writer.Allocating = .init(gpa);
    defer out.deinit();
    try m.render(std.testing.io, &out.writer, stub.engine(), 4096);
    const body = out.written();
    try std.testing.expect(std.mem.indexOf(u8, body, "tensorfold:generation_tokens_running 3") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "tensorfold:shared_prefix_waits_total 9\n") != null);
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const health = try @import("live.zig").snapshot(arena.allocator(), stub.engine());
    try std.testing.expectEqual(@as(?i64, 9), health.get("shared_prefix_waits").?.int64());
    try std.testing.expect(std.mem.indexOf(u8, body, "tensorfold:decode_rounds_total 2") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "tensorfold:request_prefill_seconds_sum 0.25") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "tensorfold:request_time_per_output_token_seconds_sum 1") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "prompt_tokens_cached_total") == null);
    try std.testing.expect(std.mem.indexOf(u8, body, "prefix_cache_hits_total") == null);
    try std.testing.expect(std.mem.indexOf(u8, body, "device_memory_bytes") == null); // unknown memory: no gauge
    try std.testing.expect(std.mem.indexOf(u8, body, "pinned_memory_bytes") == null);
    stub.memory_value = .{ .active = 7 << 30, .peak = 8 << 30, .pinned = 1 << 30 };
    var known: std.Io.Writer.Allocating = .init(gpa);
    defer known.deinit();
    try m.render(std.testing.io, &known.writer, stub.engine(), 4096);
    try std.testing.expect(std.mem.indexOf(u8, known.written(), "tensorfold:device_memory_bytes 7516192768\n") != null);
    try std.testing.expect(std.mem.indexOf(u8, known.written(), "tensorfold:device_memory_peak_bytes 8589934592\n") != null);
    try std.testing.expect(std.mem.indexOf(u8, known.written(), "tensorfold:pinned_memory_bytes 1073741824\n") != null);
}

test "render counts restored prompt tokens where the engine keeps prompt states" {
    const gpa = std.testing.allocator;
    var m: Metrics = .{ .gpa = gpa };
    m.note(std.testing.io, 100, 0, 3, 0, 0, 2, 1.0, null, null, null, null);
    m.note(std.testing.io, 120, 96, 3, 0, 0, 2, 1.0, null, null, null, null);
    var stub: RenderStub = .{ .info_value = .{ .prompt_cache = true } };
    var out: std.Io.Writer.Allocating = .init(gpa);
    defer out.deinit();
    try m.render(std.testing.io, &out.writer, stub.engine(), 4096);
    const body = out.written();
    try std.testing.expect(std.mem.indexOf(u8, body, "tensorfold:prompt_tokens_total 220\n") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "tensorfold:prompt_tokens_cached_total 96\n") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "tensorfold:prefix_cache_queries_total 220\n") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "tensorfold:prefix_cache_hits_total 96\n") != null);
}

test "metrics snapshot allocation failure releases the mutex" {
    const io = std.testing.io;
    var failing = std.testing.FailingAllocator.init(std.testing.allocator, .{});
    const gpa = failing.allocator();
    var m: Metrics = .{ .gpa = gpa };
    defer m.deinit();
    m.httpRequest(io, "client", 200);
    try std.testing.expectEqual(@as(usize, 1), m.requests.items.len);
    var stub: RenderStub = .{};
    var out: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer out.deinit();
    failing.fail_index = failing.alloc_index;
    try std.testing.expectError(error.OutOfMemory, m.render(io, &out.writer, stub.engine(), 4096));
    const available = m.mutex.tryLock();
    // No other thread uses m: either tryLock acquired it, or render retained it.
    // Release before asserting so the unfixed regression terminates without blocking.
    m.mutex.unlock(io);
    try std.testing.expect(available);
    try std.testing.expect(failing.has_induced_failure);
    failing.fail_index = std.math.maxInt(usize);
    m.httpRequest(io, "client", 200);
    m.note(io, 10, 0, 3, 0, 0, 2, 1.0, null, null, 0.25, null);
    try m.render(io, &out.writer, stub.engine(), 4096);
    try std.testing.expect(std.mem.indexOf(u8, out.written(), "requests_total{key=\"client\",status=\"200\"} 2") != null);
    try std.testing.expect(std.mem.indexOf(u8, out.written(), "prompt_tokens_total 10") != null);
}

test "request counter append failure frees its label" {
    var bytes: [1024]u8 = undefined;
    var buffer = std.heap.FixedBufferAllocator.init(&bytes);
    var failing = std.testing.FailingAllocator.init(buffer.allocator(), .{ .fail_index = 1 });
    const gpa = failing.allocator();
    var m: Metrics = .{ .gpa = gpa };
    defer m.deinit();
    m.httpRequest(std.testing.io, "client", 200);
    try std.testing.expect(failing.has_induced_failure);
    try std.testing.expectEqual(@as(usize, 6), failing.allocated_bytes);
    try std.testing.expectEqual(@as(usize, 0), m.requests.items.len);
    try std.testing.expectEqual(failing.allocated_bytes, failing.freed_bytes);

    failing.fail_index = std.math.maxInt(usize);
    m.httpRequest(std.testing.io, "client", 200);
    try std.testing.expectEqual(@as(usize, 1), m.requests.items.len);
    try std.testing.expectEqualStrings("client", m.requests.items[0].key);
    const allocations = failing.allocations;
    m.httpRequest(std.testing.io, "client", 200);
    try std.testing.expectEqual(@as(u64, 2), m.requests.items[0].count);
    try std.testing.expectEqual(allocations, failing.allocations);
}

test "render separates raw spill reads, committed restores, write outcomes and executed prefill rows" {
    const gpa = std.testing.allocator;
    var m: Metrics = .{ .gpa = gpa };
    const Spilling = struct {
        stub: RenderStub = .{ .status_value = .{ .prefill = .{ .submitted_rows = 6144, .completed_rows = 4096, .passes = 2 } } },
        fn spill(_: *anyopaque, out: *api.SpillStatus) bool {
            out.* = .{ .tier = .{ .stats = .{ .loaded_tokens = 8192, .ram_evicted = 3 }, .entries = 1, .pending = 0, .stored_bytes = 1, .queued_bytes = 0, .disk_free = 1, .cap = 1 }, .age_ms = 7 };
            out.cache.read_back = 2;
            out.cache.recalled_tokens = 4096;
            out.writes.skipped_busy = 5;
            out.writes.lost = 1;
            return true;
        }
        fn engine(x: *@This()) api.Engine {
            return .{ .ctx = &x.stub, .vtable = &.{ .info = RenderStub.info, .submit = RenderStub.submit, .cancel = RenderStub.cancel, .status = RenderStub.status, .memory = RenderStub.memory, .spill = spill } };
        }
    };
    var x: Spilling = .{};
    var out: std.Io.Writer.Allocating = .init(gpa);
    defer out.deinit();
    try m.render(std.testing.io, &out.writer, x.engine(), 4096);
    const body = out.written();
    for ([_][]const u8{
        "tensorfold:prefill_rows_submitted_total 6144\n",
        "tensorfold:prefill_rows_completed_total 4096\n",
        "tensorfold:prefill_passes_total 2\n",
        "tensorfold:spill_disk_read_states_total 2\n",
        "tensorfold:spill_disk_read_tokens_total 8192\n",
        "tensorfold:spill_restored_tokens_total 4096\n",
        "tensorfold:spill_write_skipped_total 5\n",
        "tensorfold:spill_write_lost_total 1\n",
        "tensorfold:spill_status_age_ms 7\n",
        "tensorfold:spill_ram_evicted_total 3\n",
    }) |want| try std.testing.expect(std.mem.indexOf(u8, body, want) != null);
}
