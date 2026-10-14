//! `tf-cuda-test split-load MODEL`: load each rank's half of a layer split on its own device and report the bytes.
//!
//! This is the weight half of the split, checked on the real checkpoint: rank 0 reads layers [0, mid) plus the
//! embedding, rank 1 reads [mid, layers) plus the final norm, the head and the MTP head. Each rank loads on its own
//! device through its own context, so the per-rank device bytes are the number the memory admission will use.

const std = @import("std");
const cuda = @import("cuda");
const nemotron = @import("nemotron");
const check = @import("check.zig");
const pair_tests = @import("pair_tests.zig");

/// The layer each rank owns, by the model's own layer count.
///
/// Not the midpoint: rank 1 also carries the final norm, the head and the MTP head, so the balance point sits past
/// the middle. At 52 layers the midpoint gives 7.95 against 9.38 GiB; 28 gives 8.64 against 8.61, measured.
pub fn splitAt(layers: usize) usize {
    return layers / 2 + 2;
}

pub fn run(gpu: check.Gpu, model: []const u8) !void {
    const gpa = gpu.gpa;
    const io = gpu.io;
    const pick = (try pair_tests.internalPair(gpu.d, gpa)) orelse {
        std.debug.print("SKIP split-load: no two devices share a compute capability\n", .{});
        return;
    };
    const cfg = try nemotron.Config.read(gpa, io, model);
    const mid = splitAt(cfg.layers);
    std.debug.print("split-load: {d} layers, split at {d}\n", .{ cfg.layers, mid });

    // Rank 0: layers [0, mid) and the embedding. Rank 1: [mid, layers), the norm, the head and the MTP head.
    const ranges = [_]nemotron.weights.Range{
        .{ .first = 0, .last = mid, .embed = true, .head = false },
        .{ .first = mid, .last = cfg.layers, .embed = false, .head = true },
    };
    var total: u64 = 0;
    for (ranges, 0..) |r, rank| {
        var ctx = try cuda.Context.init(gpu.d, pick[rank]);
        defer ctx.deinit();
        const k = try nemotron.kernels.Kernels.load(gpa, io, &ctx, null);
        defer {
            var x = k;
            x.deinit();
        }
        const stream = try cuda.Stream.init(gpu.d, true);
        defer {
            var x = stream;
            x.deinit();
        }
        const before = cuda.usage(false).device;
        var w = try nemotron.weights.loadRange(gpa, io, .{ .k = &k, .s = stream }, model, cfg, true, r);
        defer w.deinit();
        const held = cuda.usage(false).device - before;
        total += held;
        var name_buf: [128]u8 = undefined;
        const name = try ctx.name(&name_buf);
        check.pass("split-load rank {d}: layers [{d}, {d}) embed={} head={} on {s}, {d:.2} GiB", .{ rank, r.first, r.last, r.embed, r.head, name, gib(held) });
    }
    std.debug.print("RESULT split-load {d:.2} GiB across both ranks\n", .{gib(total)});
}

fn gib(bytes: u64) f64 {
    return @as(f64, @floatFromInt(bytes)) / (1 << 30);
}
