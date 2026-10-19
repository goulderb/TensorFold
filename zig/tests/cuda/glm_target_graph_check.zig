//! Full-target/FFN graph correctness against the eager engine, not a serving promotion gate.
//! Uses the retained GLM test/engine interfaces; both ranks run the same calls directly.
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm_cuda");
const check = @import("check.zig");
const Engine = glm.engine.Engine;
const slots = [_]u32{ 7, 1, 5, 0, 6, 2, 4, 3 };

const Fixture = struct {
    widths: [8]usize = @splat(0),
    count: usize,
    variant: usize,
    keep: usize = 1,
    lone: bool = false,
    mixed_sparse: bool = false,
    prompt_base: usize = 57,

    fn shared(total: usize, variant: usize, cap: usize) Fixture {
        const count = if (variant == 0) @min(total, 8) else (total + cap - 1) / cap;
        var f: Fixture = .{ .count = count, .variant = variant };
        for (f.widths[0..count], 0..) |*n, i| n.* = total / count + @intFromBool(i < total % count);
        return f;
    }

    fn slot(f: Fixture, i: usize) u32 {
        return slots[(i + f.variant * 3) % slots.len];
    }
};

const Round = struct {
    ids: [8][16]u32 = undefined,
    out: [8][16]u32 = undefined,
    parts: [8]glm.engine.Part = undefined,

    fn prepare(r: *Round, e: *Engine, f: Fixture) !void {
        try e.stream.synchronize();
        // New sequence owners each pass: graphs must not retain sequence pointers.
        for (slots) |slot| e.release(slot);
        for ([_]*glm.forward.Buffers{ &e.bp, &e.bd, &e.bs.? }) |b| {
            if (b.taps) |tap| try e.d.check(e.d.api.cuMemsetD8Async(tap.ptr, 0, tap.len, e.stream.handle), "clear graph proof taps");
        }
        for (0..f.count) |i| {
            const slot = f.slot(i);
            const sparse = f.mixed_sparse and i % 2 == f.variant % 2;
            const length = (if (sparse) @as(usize, 2060) else f.prompt_base) + (i + f.variant * 2) % 5;
            try e.bind(slot);
            if (try e.placement(length + 32)) |region| try e.reserveAt(region);
            const st = &e.cur.st;
            try e.d.check(e.d.api.cuMemsetD8Async(st.arena.buf.ptr, 0, st.arena.buf.len, e.stream.handle), "clear graph proof state");
            try e.begin(if (f.variant == 0) null else .{ .seed = 991 + slot });
            const prompt = try e.gpa.alloc(u32, length);
            defer e.gpa.free(prompt);
            for (prompt, 0..) |*id, j| id.* = @intCast(10 + (i * 37 + j * 7 + f.variant * 101) % 1000);
            const first = try e.prefill(prompt, null, null, null);
            if (f.variant != 0) {
                // Change recurrence parity as well as token values and committed positions.
                var next: [1]u32 = undefined;
                try e.verify(&.{first}, &next);
                try e.keep(1);
            }
            for (&r.ids[i], 0..) |*id, j| id.* = @intCast(100 + (i * 23 + j * 13 + f.variant * 71) % 1000);
            r.parts[i] = .{ .slot = slot, .ids = r.ids[i][0..f.widths[i]], .out = r.out[i][0..f.widths[i]] };
        }
        try e.stream.synchronize();
    }

    fn verify(r: *Round, e: *Engine, f: Fixture) !void {
        if (f.lone) {
            try e.bind(f.slot(0));
            try e.verify(r.parts[0].ids, r.parts[0].out);
        } else try e.verifyShared(r.parts[0..f.count]);
    }
};

const Witness = struct {
    bytes: std.ArrayList(u8) = .empty,

    fn append(w: *Witness, a: std.mem.Allocator, bytes: []const u8) !void {
        try w.bytes.appendSlice(a, bytes);
    }

    fn device(w: *Witness, a: std.mem.Allocator, e: *Engine, ptr: u64, len: usize) !void {
        const at = w.bytes.items.len;
        try w.bytes.resize(a, at + len);
        if (len > 0) try e.d.check(e.d.api.cuMemcpyDtoH_v2(w.bytes.items[at..].ptr, ptr, len), "graph proof download");
    }

    fn bf16(w: *Witness, a: std.mem.Allocator, e: *Engine, ptr: u64, len: usize) !void {
        const at = w.bytes.items.len;
        try w.device(a, e, ptr, len);
        const data = w.bytes.items[at..];
        var i: usize = 0;
        while (i < data.len) : (i += 2) {
            const bits = std.mem.readInt(u16, data[i..][0..2], .little);
            try check.expect(bits & 0x7f80 != 0x7f80, "rank {d}: non-finite BF16 witness at element {d}", .{ e.share.rank, i / 2 });
        }
    }

    fn rows(w: *Witness, a: std.mem.Allocator, e: *Engine, b: *glm.forward.Buffers, n: usize, tokens: []const u32) !void {
        try w.bf16(a, e, b.logits, n * e.share.vocab.len() * 2);
        if (b.taps) |tap| try w.bf16(a, e, tap.ptr, tap.len);
        try w.append(a, std.mem.sliceAsBytes(tokens));
    }

    fn state(w: *Witness, a: std.mem.Allocator, e: *Engine) !void {
        const st = &e.cur.st;
        try check.expect(st.pending == 0, "slot {d}: keep settled all pending rows", .{e.slot});
        const owner = if (st.layout.external_planes) st.mtp_caches.owner else null;
        const resident = owner == null or owner.? == e.share.rank;
        const meta = [_]u64{ e.slot, st.pos, st.pending, st.mtp_len, if (owner) |rank| rank else std.math.maxInt(u64) };
        try w.append(a, std.mem.sliceAsBytes(&meta));
        for (st.cur) |parity| try w.bytes.append(a, @intCast(parity));
        try w.device(a, e, st.arena.buf.ptr, st.arena.buf.len);
        // Also resolve external latent/pooled planes when the optional KV pool is enabled.
        const plan = try st.layout.snapshot(.{ .main = st.pos, .mtp = st.mtp_len });
        var ranges = plan.rangesFor(resident);
        while (ranges.next()) |range| try w.device(a, e, st.address(range), range.bytes);
    }
};

fn expectedReplays(e: *const Engine) u64 {
    return if (e.settings.target_graphs == .full) 1 else @intCast(e.w.layers.len);
}

fn fixture(gpu: check.Gpu, e: *Engine, f: Fixture) !void {
    const a = gpu.gpa;
    // Keep the reference and one reusable working image. An arena retained every
    // ArrayList growth allocation and all three multi-sequence witness images.
    var reference: Witness = .{};
    defer reference.bytes.deinit(a);
    var w: Witness = .{};
    defer w.bytes.deinit(a);
    var captures: u64 = 0;
    var replays: u64 = 0;
    var round: Round = .{};
    for (0..3) |pass| {
        e.target_graphs_active = pass != 0;
        try round.prepare(e, f);
        if (pass == 2) {
            captures = e.targetGraphStats().site_captures;
            replays = e.targetGraphStats().site_replays;
        }
        const before = e.targetGraphStats();
        const start = check.now(gpu.io);
        try round.verify(e, f);
        try e.stream.synchronize();
        const elapsed = check.now(gpu.io) - start;
        const after_verify = e.targetGraphStats();
        const memory = try gpu.ctx.memInfo();
        std.debug.print("target-graph fixture rank={d} pass={d} lone={} widths={any} elapsed_ns={d} capture_delta={d} replay_delta={d} free_bytes={d} total_bytes={d}\n", .{ e.share.rank, pass, f.lone, f.widths[0..f.count], elapsed, after_verify.site_captures - before.site_captures, after_verify.site_replays - before.site_replays, memory.free, memory.total });
        if (pass == 2) {
            try check.expect(after_verify.site_captures == captures and after_verify.site_replays == replays + expectedReplays(e), "third pass replays the configured target path at the exact buffer/width without capture", .{});
        }
        w.bytes.clearRetainingCapacity();
        if (pass != 0) try w.bytes.ensureTotalCapacityPrecise(a, reference.bytes.items.len);
        const b = if (f.lone) &e.bd else &e.bs.?;
        var total: usize = 0;
        for (f.widths[0..f.count]) |n| total += n;
        try w.rows(a, e, b, total, &.{});
        for (round.parts[0..f.count]) |part| try w.append(a, std.mem.sliceAsBytes(part.out));
        for (round.parts[0..f.count]) |part| {
            try e.bind(part.slot);
            try e.keep(@min(f.keep, part.ids.len));
        }
        try e.stream.synchronize();
        for (round.parts[0..f.count]) |part| {
            try e.bind(part.slot);
            try w.state(a, e);
            // Follow each partial commit with a real target step, not just metadata inspection.
            const token = part.out[@min(f.keep, part.ids.len) - 1];
            var following: [1]u32 = undefined;
            try e.verify(&.{token}, &following);
            try e.stream.synchronize();
            try w.rows(a, e, &e.bd, 1, &following);
            try e.keep(1);
            try e.stream.synchronize();
            try w.state(a, e);
        }
        if (pass == 0) std.mem.swap(Witness, &reference, &w) else try check.sameBytes("same-shape eager/target-graph logits, taps, tokens, committed state and following decode", w.bytes.items, reference.bytes.items);
        if (pass == 2) {
            const after = e.targetGraphStats();
            try check.expect(after.site_captures == captures and after.site_replays > replays, "width {d}, keep {d}: third pass replays without new capture", .{ total, f.keep });
        }
    }
    check.pass("rank {d}: {s} geometry {any}, variant {d}, keep {d}: eager/capture/replay byte witnesses agree", .{ e.share.rank, if (f.lone) "lone" else "shared", f.widths[0..f.count], f.variant, f.keep });
}

fn timing(gpu: check.Gpu, e: *Engine) !void {
    const f = Fixture.shared(32, 0, e.targetRows());
    var round: Round = .{};
    // Pair zero warms both sides. Preparation and proof downloads are not timed.
    for (0..6) |pair| for (0..2) |side| {
        const graph = (side + pair) % 2 == 1;
        e.target_graphs_active = graph;
        try round.prepare(e, f);
        const before = e.targetGraphStats();
        const start = check.now(gpu.io);
        try round.verify(e, f);
        try e.stream.synchronize();
        const elapsed = check.now(gpu.io) - start;
        const after = e.targetGraphStats();
        if (graph) try check.expect(after.site_captures == before.site_captures and after.site_replays == before.site_replays + expectedReplays(e), "timed graph width 32 replays the configured target path without capture", .{});
        if (pair > 0) std.debug.print("target-graph timing rank={d} pair={d} graph={} rows=32 elapsed_ns={d} replay_delta={d}\n", .{ e.share.rank, pair, graph, elapsed, after.site_replays - before.site_replays });
    };
}

pub fn run(gpu: check.Gpu, checkpoint: []const u8, rank: c_int, world: c_int, id_path: []const u8) !void {
    if (rank < 0 or world < 1 or rank >= world) return error.BadGraphProofRank;
    var problem: []const u8 = "";
    const settings = try glm.ranks.Settings.read(&glm.ranks.environment, &problem);
    if (settings.target_graphs == .off) return error.GraphProofNeedsTargetGraphs;
    if (settings.layers == 0 or settings.layers > 6) return error.GraphProofNeedsBoundedLayerPrefix;
    var lib = try cuda.nccl.Library.open();
    defer lib.close();
    const comm = try @import("glm_lanes_tests.zig").openComm(gpu, &lib, rank, world, id_path);
    defer _ = lib.api.ncclCommDestroy(comm);
    const e = try Engine.init(gpu.gpa, gpu.io, gpu.d, gpu.ctx, checkpoint, &lib, comm, @intCast(rank), @intCast(world), .{ .capacity = 4096, .mtp = true, .settings = settings, .draft_kernels = glm.ranks.environment("TF_GLM_DFLASH_KERNELS"), .partial_dflash_check = true });
    defer e.deinit();
    if (e.bs == null or (e.ffn_graphs == null and e.full_graphs == null)) return error.GraphProofNeedsSharedDraftBuffers;
    try e.bd.enableTaps(gpu.d);
    try e.bs.?.enableTaps(gpu.d);
    std.debug.print("target-graph proof: mode={s}, layers={d}, pool={}, KV shard={}; prefix correctness, not capacity/spill/vision or full-model qualification\n", .{ @tagName(settings.target_graphs), e.w.layers.len, e.pool != null, settings.kv_shard });
    for (1..33) |width| for (0..2) |variant| try fixture(gpu, e, Fixture.shared(width, variant, e.targetRows()));
    for (1..17) |keep| {
        var f: Fixture = .{ .count = 1, .variant = keep % 2, .keep = keep, .lone = true };
        f.widths[0] = 16;
        try fixture(gpu, e, f);
        if (e.targetRows() >= 16) {
            f.lone = false;
            f.count = 2;
            f.widths[1] = 16;
            try fixture(gpu, e, f);
        }
    }
    if (settings.target_graphs == .full) {
        for (0..2) |variant| {
            var f: Fixture = .{ .count = 2, .variant = variant, .mixed_sparse = true };
            f.widths[0] = @min(e.targetRows(), 16);
            f.widths[1] = f.widths[0];
            try fixture(gpu, e, f);
            f.mixed_sparse = false;
            f.prompt_base = 511 - f.widths[0] + 1;
            try fixture(gpu, e, f);
        }
    }
    try timing(gpu, e);
    const stats = e.targetGraphStats();
    check.pass("rank {d}: bounded {s} target graph proof complete; eager={d} captures={d} replays={d}; timing is verifyShared plus synchronization, not full-model speedup", .{ rank, @tagName(settings.target_graphs), stats.site_eager, stats.site_captures, stats.site_replays });
}
