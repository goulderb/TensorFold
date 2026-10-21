const std = @import("std");
const core = @import("core");
const cuda = @import("cuda");
const source = @import("cuda_weight_source.zig");
const dense = @import("cuda_weight_source_dense.zig");
const W = @import("cuda_weights.zig");
const Config = @import("config.zig");
const Share = @import("cuda_shard.zig").Share;
const vision = @import("cuda_vision_weights.zig");
const X = cuda.experts.Nvfp4Experts;

fn ends(name: []const u8, suffix: []const u8) bool {
    return std.mem.endsWith(u8, name, suffix);
}

fn sourceCuts(name: []const u8, c: *const Config.Config, s: Share, w: core.checkpoint.Tensor, buf: *[128]source.Span) ?struct { n: usize, cols: source.Span } {
    if (w.rank != 2 or !ends(name, ".weight")) return null;
    const rank_cut = if (std.mem.indexOf(u8, name, ".shared_experts.") != null) s.shared else s.dense;
    var n: usize = 1;
    var cols: source.Span = .{ .lo = 0, .hi = w.dim(1) };
    if (ends(name, ".kv_b_proj.weight")) {
        const per = c.nope + c.v_dim;
        n = 0;
        for (s.mla.lo..s.mla.hi) |head| {
            buf[n] = .{ .lo = head * per, .hi = head * per + c.nope };
            n += 1;
        }
        for (s.mla.lo..s.mla.hi) |head| {
            buf[n] = .{ .lo = head * per + c.nope, .hi = (head + 1) * per };
            n += 1;
        }
    } else if (ends(name, ".q_b_proj.weight")) {
        buf[0] = .{ .lo = s.mla.lo * c.nope, .hi = s.mla.hi * c.nope };
    } else if (ends(name, ".q_proj.weight") or ends(name, ".k_proj.weight") or ends(name, ".v_proj.weight") or ends(name, ".f_b_proj.weight") or ends(name, ".g_b_proj.weight")) {
        buf[0] = .{ .lo = s.kda.lo * c.kda_dim, .hi = s.kda.hi * c.kda_dim };
    } else if (ends(name, ".b_proj.weight")) {
        buf[0] = .{ .lo = s.kda.lo, .hi = s.kda.hi };
    } else if (ends(name, ".o_proj.weight")) {
        buf[0] = .{ .lo = 0, .hi = w.dim(0) };
        cols = if (w.dim(1) == c.kdaWidth()) .{ .lo = s.kda.lo * c.kda_dim, .hi = s.kda.hi * c.kda_dim } else .{ .lo = s.mla.lo * c.v_dim, .hi = s.mla.hi * c.v_dim };
    } else if (ends(name, ".gate_proj.weight") or ends(name, ".up_proj.weight")) {
        buf[0] = .{ .lo = rank_cut.lo, .hi = rank_cut.hi };
    } else if (ends(name, ".down_proj.weight")) {
        buf[0] = .{ .lo = 0, .hi = w.dim(0) };
        cols = .{ .lo = rank_cut.lo, .hi = rank_cut.hi };
    } else buf[0] = .{ .lo = 0, .hi = w.dim(0) };
    return .{ .n = n, .cols = cols };
}

fn parentRow(spans: []const source.Span, packed_r: usize) usize {
    var at: usize = 0;
    for (spans) |sp| {
        const n = sp.hi - sp.lo;
        if (packed_r < at + n) return sp.lo + (packed_r - at);
        at += n;
    }
    unreachable;
}

fn samplePacked(a: std.mem.Allocator, p: dense.Packed, t: source.Source, spans: []const source.Span, cols: source.Span) !void {
    const m = try p.matrix();
    const row = try a.alloc(f64, p.in_total);
    defer a.free(row);
    const want = try a.alloc(f64, p.ins);
    defer a.free(want);
    const picks = [_]usize{ 0, p.outs / 2, p.outs - 1 };
    for (picks) |pr| {
        if (p.outs == 0) return;
        try m.rowsF64(p.out0 + pr, 1, row);
        const parent = parentRow(spans, pr);
        switch (t.format.?) {
            .mxfp8 => try core.fp8_source.mxfp8(t.weight, t.scale.?, .{ .lo = parent, .hi = parent + 1 }, cols, want),
            .fp8_pb_wo => try core.fp8_source.pbWo(t.weight, t.scale.?, .{ .lo = parent, .hi = parent + 1 }, cols, want),
            else => return error.SourceDenseRequiresFp8,
        }
        for (0..p.ins) |c| {
            const got = row[p.in0 + c];
            if (std.math.isNan(want[c]) or std.math.isNan(got)) {
                if (!(std.math.isNan(got) and std.math.isNan(want[c]))) return error.SourceDecodeMismatch;
            } else if (got != want[c]) return error.SourceDecodeMismatch;
        }
    }
}

const PackStats = struct { source_bytes: usize, bf16_bytes: usize };

fn gather(L: *W.Loader, name: []const u8, c: *const Config.Config, s: Share) !PackStats {
    const t = try source.Source.read(L.ck, L.tools.recipes, name);
    const w = t.weight;
    const rank_cut = if (std.mem.indexOf(u8, name, ".shared_experts.") != null) s.shared else s.dense;
    var stats: PackStats = .{ .source_bytes = 0, .bf16_bytes = 0 };
    if (w.rank == 2 and ends(name, ".weight")) {
        if (ends(name, ".kv_b_proj.weight")) {
            const per = c.nope + c.v_dim;
            for (s.mla.lo..s.mla.hi) |head| try L.rows(t, head * per, head * per + c.nope, c.mla_heads * per, .bf16);
            for (s.mla.lo..s.mla.hi) |head| try L.rows(t, head * per + c.nope, (head + 1) * per, c.mla_heads * per, .bf16);
        } else if (ends(name, ".q_b_proj.weight")) {
            try L.rows(t, s.mla.lo * c.nope, s.mla.hi * c.nope, c.mla_heads * c.nope, .bf16);
        } else if (ends(name, ".q_proj.weight") or ends(name, ".k_proj.weight") or ends(name, ".v_proj.weight") or ends(name, ".f_b_proj.weight") or ends(name, ".g_b_proj.weight")) {
            try L.rows(t, s.kda.lo * c.kda_dim, s.kda.hi * c.kda_dim, c.kda_heads * c.kda_dim, .bf16);
        } else if (ends(name, ".b_proj.weight")) {
            try L.rows(t, s.kda.lo, s.kda.hi, c.kda_heads, .bf16);
        } else if (ends(name, ".o_proj.weight")) {
            if (w.dim(1) == c.kdaWidth()) try L.cols(t, s.kda.lo * c.kda_dim, s.kda.hi * c.kda_dim, .bf16) else try L.cols(t, s.mla.lo * c.v_dim, s.mla.hi * c.v_dim, .bf16);
        } else if (ends(name, ".gate_proj.weight") or ends(name, ".up_proj.weight")) {
            try L.rows(t, rank_cut.lo, rank_cut.hi, w.dim(0), .bf16);
        } else if (ends(name, ".down_proj.weight")) {
            try L.cols(t, rank_cut.lo, rank_cut.hi, .bf16);
        } else try L.whole(t, .bf16);
    } else if (ends(name, "_conv1d.weight")) {
        try L.rows(t, s.kda.lo * c.kda_dim, s.kda.hi * c.kda_dim, c.kdaWidth(), .f32);
    } else if (ends(name, ".A_log")) {
        try L.rows(t, s.kda.lo, s.kda.hi, c.kda_heads, .f32);
    } else if (ends(name, ".dt_bias")) {
        try L.rows(t, s.kda.lo * c.kda_dim, s.kda.hi * c.kda_dim, c.kdaWidth(), .f32);
    } else try L.whole(t, if (ends(name, "_base") or ends(name, "_scale") or ends(name, "e_score_correction_bias")) .f32 else .bf16);
    if (t.format != null) {
        var span_buf: [128]source.Span = undefined;
        if (sourceCuts(name, c, s, w, &span_buf)) |cut| {
            var p = try dense.pack(L.gpa, t, span_buf[0..cut.n], cut.cols);
            defer p.deinit(L.gpa);
            try samplePacked(L.gpa, p, t, span_buf[0..cut.n], cut.cols);
            stats = .{ .source_bytes = p.resident(), .bf16_bytes = @as(usize, p.outs) * p.ins * 2 };
        }
    }
    return stats;
}

fn layer(a: std.mem.Allocator, io: std.Io, dir: []const u8, table: *const core.modelopt.Recipes, c: Config.Config, s: Share, index: u32) !void {
    var buf: [128]u8 = undefined;
    const prefix = try std.fmt.bufPrint(&buf, "model.language_model.layers.{d}.", .{index});
    var ck = try core.Checkpoint.openModelPrefix(a, io, dir, prefix);
    defer ck.close();
    var weights: W.Weights = .{ .gpa = a, .config = c, .share = s };
    defer weights.deinit();
    var loader: W.Loader = .{ .gpa = a, .d = undefined, .stream = undefined, .pack_dense = undefined, .tools = .{ .pack_dense = undefined, .recipes = table }, .w = &weights, .ck = &ck };
    defer loader.host.deinit(a);
    var hash: std.hash.XxHash64 = .init(0);
    var gathered: usize = 0;
    var matrices: usize = 0;
    var peak: usize = 0;
    var source_bytes: usize = 0;
    var bf16_bytes: usize = 0;
    if (c.isMoe(index)) {
        const ni = s.moe.len();
        const d = c.hidden;
        const up = try a.alloc(u32, X.bytes(1, ni, d, 2) / 4);
        defer a.free(up);
        const down = try a.alloc(u32, X.bytes(1, d, ni, 1) / 4);
        defer a.free(down);
        const codes = try a.alloc(u8, d * ni / 2);
        defer a.free(codes);
        const scales = try a.alloc(u8, d * ni / 16);
        defer a.free(scales);
        peak = up.len * 4 + down.len * 4 + codes.len + scales.len + (if (index == c.layers) @as(usize, 0) else 3 * @as(usize, d) * c.moe_inter / 16);
        for (0..c.experts) |expert| {
            const globals = try source.packExpert(a, &ck, table, index, expert, .{ .lo = s.moe.lo, .hi = s.moe.hi }, d, up, down, codes, scales);
            hash.update(std.mem.sliceAsBytes(up));
            hash.update(std.mem.sliceAsBytes(down));
            hash.update(std.mem.asBytes(&globals));
            gathered += (up.len + down.len) * 4 + 12;
            matrices += 3;
        }
    }
    for (ck.files.items) |*file| for (file.names.keys()) |name| {
        if (ck.used.contains(name)) continue;
        if (std.mem.indexOf(u8, name, ".weight_scale") != null or ends(name, ".input_scale")) continue;
        const src_pack = try gather(&loader, name, &c, s);
        source_bytes += src_pack.source_bytes;
        bf16_bytes += src_pack.bf16_bytes;
        peak = @max(peak, loader.host.capacity);
        gathered += loader.host.items.len;
        if (ends(name, ".weight") and source.recipe(table, name[0 .. name.len - 7]) != null) matrices += 1;
        hash.update(loader.host.items);
        loader.host.clearRetainingCapacity();
    };
    const unused = ck.unused();
    if (unused != 0) return error.UnusedTensors;
    std.debug.print("layer={d} kind={s} rank={d}/2 tensors={d} matrices={d} gathered_bytes={d} source_bytes={d} bf16_bytes={d} saved={d} scratch_bound={d} hash={x} unused={d}\n", .{ index, @tagName(c.kind(index)), s.rank, ck.used.count(), matrices, gathered, source_bytes, bf16_bytes, bf16_bytes -| source_bytes, peak, hash.final(), unused });
}

pub fn main(init: std.process.Init) !void {
    const a = std.heap.page_allocator;
    const argv = try init.minimal.args.toSlice(init.arena.allocator());
    if (argv.len != 2) return error.CheckpointArgumentRequired;
    const dir = argv[1];
    const path = try std.fs.path.join(a, &.{ dir, "config.json" });
    defer a.free(path);
    const json = try std.Io.Dir.cwd().readFileAlloc(init.io, path, a, .limited(1 << 24));
    defer a.free(json);
    var why: core.modelopt.Why = .{};
    var admitted = try @import("cuda_checkpoint.zig").Admission.read(a, json, &why);
    defer admitted.deinit();
    const c = try Config.parse(a, json);
    for (0..2) |rank| {
        const s = try Share.of(&c, rank, 2);
        for ([_]u32{ 0, 3, 45 }) |i| try layer(a, init.io, dir, &admitted.recipes.?, c, s, i);
        {
            var head_ck = try core.Checkpoint.openModelPrefix(a, init.io, dir, "lm_head.");
            defer head_ck.close();
            const head = try source.Source.read(&head_ck, &admitted.recipes.?, "lm_head.weight");
            var p = try dense.pack(a, head, &.{.{ .lo = s.vocab.lo, .hi = s.vocab.hi }}, .{ .lo = 0, .hi = c.hidden });
            defer p.deinit(a);
            try samplePacked(a, p, head, &.{.{ .lo = s.vocab.lo, .hi = s.vocab.hi }}, .{ .lo = 0, .hi = c.hidden });
            if (head_ck.unused() != 0) return error.UnusedTensors;
            std.debug.print("head rank={d}/2 tensors={d} source_bytes={d} bf16_bytes={d} saved={d} unused=0\n", .{ s.rank, head_ck.used.count(), p.resident(), @as(usize, p.outs) * p.ins * 2, @as(usize, p.outs) * p.ins * 2 - p.resident() });
        }
    }
    var ck = try core.Checkpoint.openModelPrefix(a, init.io, dir, "model.visual.");
    defer ck.close();
    var vw = try vision.Weights.loadMixed(a, try @import("cuda_vision_config.zig").parse(a, json), &ck, &admitted.recipes.?);
    defer vw.deinit(a);
    var hash: std.hash.XxHash64 = .init(0);
    var bytes: usize = 0;
    var peak: usize = 0;
    var quantized: usize = 0;
    for (vw.tensors, 0..) |_, i| {
        const values = try vw.converted(a, i);
        defer a.free(values);
        bytes += values.len * 2;
        peak = @max(peak, values.len * 2);
        hash.update(std.mem.sliceAsBytes(values));
        quantized += @intFromBool(vw.quantized[i] != null);
    }
    std.debug.print("vision tensors={d} weights={d} quantized_linears={d} bf16_bytes={d} scratch_bound={d} hash={x} unused={d}\n", .{ ck.used.count(), vw.tensors.len, quantized, bytes, peak, hash.final(), ck.unused() });
}
