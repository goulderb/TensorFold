//! Metadata-driven KDA for the full target graph; sequence projection/replay saves stay compatible with eager keep.

const std = @import("std");
const cuda = @import("cuda");
const glue = @import("cuda_glue.zig");
const kda = @import("cuda_kda.zig");
const kda_block = @import("cuda_kda_block.zig");
const meta = @import("cuda_target_meta.zig");

pub const Row = meta.Row;
pub const KdaSegment = meta.KdaSegment;

pub const Kernels = struct {
    scatter: cuda.Function,
    prep: cuda.Function,
    step: cuda.Function,
    chain: cuda.Function,
    out: cuda.Function,

    pub fn init(module: cuda.Module) !Kernels {
        var ks: Kernels = undefined;
        inline for (.{ "scatter", "prep", "step", "chain" }) |name| @field(ks, name) = try module.function("tf_glm_target_kda_" ++ name);
        ks.out = try module.function("tf_glm_kda_out");
        return ks;
    }
};

fn windowRows(rows: usize) !u32 {
    if (rows == 0 or rows > meta.max_rows) return error.BadTargetGraphWindow;
    return @intCast(rows);
}

/// The runtime supplies stable max-window scratch (including s.p), a row map and this layer's eight segment records.
/// Only the records contain sequence addresses. Staging, partial keep/replay and convolution commits stay outside capture.
pub fn run(set: glue.Set, ks: *const Kernels, stream: cuda.Stream, layer: kda_block.Layer, scratch: kda_block.Scratch, normed: u64, hidden: usize, rows: usize, row_map: u64, segments: u64, out: u64, wide: bool) !void {
    const nr = try windowRows(rows);
    if (row_map == 0 or segments == 0) return error.BadTargetGraphMetadata;
    const h: c_int = @intCast(layer.heads);
    const n: c_int = @intCast(layer.heads * kda.dk);
    const width: c_int = @intCast(layer.width());
    const beta: c_int = @intCast(3 * layer.heads * kda.dk + 2 * kda.dk);
    const grid: cuda.launch.Dim3 = .{ .x = nr, .y = @intCast(layer.heads) };
    try kda_block.project(set, stream, layer, scratch, normed, hidden, rows);

    var scatter: cuda.Args = .{};
    inline for (.{ row_map, segments, scratch.p, width }) |v| scatter.add(v);
    try cuda.launch.launch(ks.scatter, .{ .grid = .{ .x = nr, .y = @intCast((layer.width() + 255) / 256) }, .block = .{ .x = 256 } }, stream, &scatter);

    if (wide) {
        var prep: cuda.Args = .{};
        inline for (.{ h, row_map, segments, width, beta, scratch.ka, n, layer.conv_w, layer.a_log, layer.dt_bias, layer.lower, scratch.wide.q }) |v| prep.add(v);
        try cuda.launch.launch(ks.prep, .{ .grid = grid, .block = .{ .x = 512 } }, stream, &prep);

        var step: cuda.Args = .{};
        inline for (.{ h, segments, scratch.wide.q, scratch.wide.y }) |v| step.add(v);
        try cuda.launch.launch(ks.step, .{ .grid = .{ .x = @intCast(layer.heads), .y = kda.dv / 4 / 8, .z = meta.max_segments }, .block = .{ .x = 8 * 32 } }, stream, &step);

        var finish: cuda.Args = .{};
        inline for (.{ h, scratch.wide.y, scratch.kg, n, layer.norm_w, layer.eps, scratch.saves.out }) |v| finish.add(v);
        try cuda.launch.launch(ks.out, .{ .grid = grid, .block = .{ .x = kda.dv } }, stream, &finish);
    } else {
        var chain: cuda.Args = .{};
        inline for (.{ h, segments, width, beta, scratch.ka, n, scratch.kg, n, layer.conv_w, layer.a_log, layer.dt_bias, layer.norm_w, layer.eps, layer.lower, scratch.saves.out }) |v| chain.add(v);
        try cuda.launch.launch(ks.chain, .{ .grid = .{ .x = @intCast(layer.heads), .y = meta.max_segments }, .block = .{ .x = 1024 } }, stream, &chain);
    }
    try kda_block.fill(set, stream, layer, scratch, 0, rows, out);
}

test "target KDA admits exactly the graph bank's aggregate widths" {
    try std.testing.expectError(error.BadTargetGraphWindow, windowRows(0));
    try std.testing.expectError(error.BadTargetGraphWindow, windowRows(meta.max_rows + 1));
    for (1..meta.max_rows + 1) |rows| try std.testing.expectEqual(@as(u32, @intCast(rows)), try windowRows(rows));
}
