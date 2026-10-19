//! Complete target graphs over fixed scratch and staged device tables. Request scheduling and commit stay eager.
const std = @import("std");
const cuda = @import("cuda");
const meta = @import("cuda_target_meta.zig");
const graph = @import("cuda_target_graph.zig");
const state = @import("cuda_state.zig");
const weights = @import("cuda_weights.zig");
const glue = @import("cuda_glue.zig");
const cache = @import("cuda_cache.zig");
const kda_block = @import("cuda_kda_block.zig");
const dsa_block = @import("cuda_dsa_block.zig");
const kda_target = @import("cuda_target_kda.zig");
const dsa_target = @import("cuda_target_dsa.zig");
const latent = @import("cuda_latent.zig");
const sparse = @import("cuda_sparse.zig");

pub const Runtime = struct {
    bank: graph.Bank,
    tables: meta.Meta,
    kda_kernels: kda_target.Kernels,
    dsa_kernels: dsa_target.Kernels,
    projected: cuda.DeviceBuffer,
    partial_out: cuda.DeviceBuffer,
    partial_max: cuda.DeviceBuffer,
    partial_sum: cuda.DeviceBuffer,
    scores: cuda.DeviceBuffer,
    queries: cuda.DeviceBuffer,
    local_results: cuda.DeviceBuffer,
    peer_results: cuda.DeviceBuffer,
    gathered_results: cuda.DeviceBuffer,
    format: cache.Format,
    capacity: usize,
    sharded: bool,
    rank: usize,
    world: usize,

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, w: *const weights.Weights, kda_module: cuda.Module, format: cache.Format, capacity: usize, sharded: bool) !Runtime {
        if (capacity == 0 or w.share.world == 0 or w.share.world > 2 or w.share.rank >= w.share.world) return error.BadTargetGraphShape;
        if (sharded and w.share.world != 2) return error.TargetGraphShardNeedsTp2;
        if (w.config.kda_dim != 128 or w.config.kv_lora != 512) return error.BadTargetGraphShape;
        for (w.layers) |layer| switch (layer.mixer) {
            .kda => |k| if (k.heads != w.share.kda.len()) return error.BadTargetGraphShape,
            .mla => |a| if (a.heads != w.share.mla.len() or a.dense_limit != sparse.width) return error.BadTargetGraphShape,
        };
        var tables = try meta.Meta.init(gpa, d, w);
        errdefer tables.deinit();
        var bank = try graph.Bank.init(gpa, 1);
        errdefer bank.deinit();
        const kda_kernels = try kda_target.Kernels.init(kda_module);
        var dsa_kernels = try dsa_target.Kernels.init(d);
        errdefer dsa_kernels.deinit();
        const n = try std.math.mul(usize, w.share.kda.len(), 128);
        const kw = try std.math.add(usize, try std.math.add(usize, try std.math.mul(usize, 3, n), 256), w.share.kda.len());
        var projected = try cuda.DeviceBuffer.alloc(d, try bytes(&.{ meta.max_rows, kw, 2 }));
        errdefer projected.free();
        // Both dense rows (at most 2051 keys) and sparse rows (at most 2051 selected keys) use five chunks.
        const partials = try bytes(&.{ 5, meta.max_rows, w.share.mla.len() });
        var partial_out = try cuda.DeviceBuffer.alloc(d, try bytes(&.{ partials, 512, 4 }));
        errdefer partial_out.free();
        var partial_max = try cuda.DeviceBuffer.alloc(d, try std.math.mul(usize, partials, 4));
        errdefer partial_max.free();
        var partial_sum = try cuda.DeviceBuffer.alloc(d, try std.math.mul(usize, partials, 4));
        errdefer partial_sum.free();
        const pools = try std.math.add(usize, capacity / 4, 2);
        var scores = try cuda.DeviceBuffer.alloc(d, try bytes(&.{ meta.max_rows, pools, 4 }));
        errdefer scores.free();
        const query_bytes = if (sharded) try bytes(&.{ 2, meta.max_rows, w.share.mla.len(), 512, 2 }) else 1;
        var queries = try cuda.DeviceBuffer.alloc(d, query_bytes);
        errdefer queries.free();
        var local_results = try cuda.DeviceBuffer.alloc(d, query_bytes);
        errdefer local_results.free();
        var peer_results = try cuda.DeviceBuffer.alloc(d, if (sharded) query_bytes / 2 else 1);
        errdefer peer_results.free();
        const gathered_results = try cuda.DeviceBuffer.alloc(d, query_bytes);
        return .{
            .bank = bank,
            .tables = tables,
            .kda_kernels = kda_kernels,
            .dsa_kernels = dsa_kernels,
            .projected = projected,
            .partial_out = partial_out,
            .partial_max = partial_max,
            .partial_sum = partial_sum,
            .scores = scores,
            .queries = queries,
            .local_results = local_results,
            .peer_results = peer_results,
            .gathered_results = gathered_results,
            .format = format,
            .capacity = capacity,
            .sharded = sharded,
            .rank = w.share.rank,
            .world = w.share.world,
        };
    }

    /// The engine completes the stream first; graph executables die before every referenced allocation and module.
    pub fn deinit(r: *Runtime) void {
        r.bank.deinit();
        r.gathered_results.free();
        r.peer_results.free();
        r.local_results.free();
        r.queries.free();
        r.scores.free();
        r.partial_sum.free();
        r.partial_max.free();
        r.partial_out.free();
        r.projected.free();
        r.dsa_kernels.deinit();
        r.tables.deinit();
        r.* = undefined;
    }

    pub fn run(r: *Runtime, f: anytype, ids: []const i32) !void {
        if (f.ranks.len != 1 or ids.len == 0 or ids.len > meta.max_rows or f.ranks[0].b.prefill or f.route_trace != null or f.target_graphs != null) return error.BadTargetGraphWindow;
        const rank = f.ranks[0];
        var segments: [meta.max_segments]meta.Segment = undefined;
        const count = if (rank.segs.len == 0) 1 else rank.segs.len;
        if (count > segments.len) return error.BadTargetGraphWindow;
        if (rank.segs.len == 0) segments[0] = .{ .st = rank.st, .row0 = 0, .rows = ids.len } else {
            for (rank.segs, segments[0..count]) |part, *dst| dst.* = .{ .st = part.st, .row0 = part.row0, .rows = part.rows };
        }
        try r.tables.stage(f.stream, rank.w, segments[0..count], ids.len, r.sharded);
        try f.prepareTarget(ids);
        const Call = struct {
            runtime: *Runtime,
            forward: @TypeOf(f),
            fn compute(call: *@This(), _: usize, rows: usize) !void {
                try call.forward.computeTarget(call.runtime, rows);
            }
        };
        var call: Call = .{ .runtime = r, .forward = f };
        try r.bank.run(f.stream, .{ .buffer = f.target_buffer, .layer = 0, .rows = ids.len }, &call, Call.compute);
    }

    pub fn kda(r: *Runtime, set: glue.Set, stream: cuda.Stream, b: *state.Buffers, layer: kda_block.Layer, index: usize, rows: usize, out: u64, wide: bool) !void {
        var scratch = b.kda;
        scratch.p = r.projected.ptr;
        try kda_target.run(set, &r.kda_kernels, stream, layer, scratch, b.normed, layer.dims, rows, r.tables.rows.ptr, r.tables.kdaLayer(index), out, wide);
    }

    pub fn dsa(r: *Runtime, set: glue.Set, stream: cuda.Stream, exchange: anytype, b: *state.Buffers, layer: dsa_block.Layer, index: usize, rows: usize, out: u64) !void {
        var scratch = b.dsa;
        scratch.partials = .{ .po = r.partial_out.ptr, .pm = r.partial_max.ptr, .pl = r.partial_sum.ptr };
        scratch.select.scores = r.scores.ptr;
        const table = r.tables.dsaLayer(index);
        try dsa_target.prepare(set, stream, layer, scratch, b.normed, layer.dims, rows);
        try dsa_target.update(&r.dsa_kernels, set, stream, layer, scratch, rows, table, r.format, r.capacity);
        if (r.sharded) {
            const nccl = switch (exchange) {
                .nccl => |n| n,
                .local => return error.TargetGraphShardNeedsNccl,
            };
            const elements = rows * layer.heads * layer.lw;
            const per_head = elements * 2;
            // Every replay performs the same collectives, regardless of sequence placement or ownership.
            try nccl.lib.check(nccl.lib.api.ncclAllGather(scratch.qa, r.queries.ptr, elements, .bf16, nccl.comm, stream.handle), "ncclAllGather target queries");
            for (0..2) |head_rank| try dsa_target.attention(&r.dsa_kernels, set, stream, layer, scratch, rows, table, r.queries.ptr + head_rank * per_head, r.local_results.ptr + head_rank * per_head, r.rank, r.format, r.capacity);
            try dsa_target.packPeer(&r.dsa_kernels, stream, r.local_results.ptr, r.rank, rows, layer.heads, layer.lw, r.peer_results.ptr);
            try nccl.lib.check(nccl.lib.api.ncclAllGather(r.peer_results.ptr, r.gathered_results.ptr, elements, .bf16, nccl.comm, stream.handle), "ncclAllGather target owner results");
            // Only the peer's head slice crosses the link; selection preserves the owner's result bits.
            try dsa_target.ownerSelect(&r.dsa_kernels, stream, table, r.local_results.ptr + r.rank * per_head, r.gathered_results.ptr, r.rank, rows, layer.heads, layer.lw, scratch.ol);
        } else try dsa_target.attention(&r.dsa_kernels, set, stream, layer, scratch, rows, table, scratch.qa, scratch.ol, r.rank, r.format, r.capacity);
        try latent.expandV(set, stream, scratch.ol, layer.wv, scratch.vn, rows, layer.heads, layer.dv, layer.lw);
        try dsa_block.fill(set, stream, layer, scratch, 0, rows, out);
    }
};

fn bytes(dimensions: []const usize) !usize {
    var n: usize = 1;
    for (dimensions) |dimension| n = try std.math.mul(usize, n, dimension);
    return n;
}

test "full target workspace arithmetic checks aggregate rows and overflow" {
    try std.testing.expectEqual(@as(usize, 10485760), try bytes(&.{ 5, 32, 32, 512, 4 }));
    try std.testing.expectError(error.Overflow, bytes(&.{ std.math.maxInt(usize), 2 }));
}
