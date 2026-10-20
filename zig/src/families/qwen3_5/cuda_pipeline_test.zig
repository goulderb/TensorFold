const std = @import("std");
const pipeline = @import("cuda_pipeline.zig");
const checkpoint = @import("cuda_checkpoint.zig");
const engine = @import("cuda_engine.zig");

fn admitted() !checkpoint.Admitted {
    return checkpoint.admit(std.testing.allocator, @embedFile("fixtures/ornith_nvfp4/config.json"));
}

test "pipeline rejects invalid placement and capacity before opening checkpoint weights or CUDA" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "config.json", .data = @embedFile("fixtures/ornith_nvfp4/config.json") });
    var path: [128]u8 = undefined;
    const dir = try std.fmt.bufPrint(&path, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    const a = std.testing.allocator;
    try std.testing.expectError(error.InvalidContext, pipeline.Chain.open(a, std.testing.io, dir, .{ .context = 0 }));
    try std.testing.expectError(error.InvalidContext, pipeline.Chain.open(a, std.testing.io, dir, .{ .context = 262145 }));
    try std.testing.expectError(error.InvalidDevices, pipeline.Chain.open(a, std.testing.io, dir, .{ .devices = &.{} }));
    try std.testing.expectError(error.InvalidPartition, pipeline.Chain.open(a, std.testing.io, dir, .{ .devices = &.{ 0, 0 } }));
    try std.testing.expectError(error.InvalidPartition, pipeline.Chain.open(a, std.testing.io, dir, .{ .cut = 0 }));
    try std.testing.expectError(error.InvalidPartition, pipeline.Chain.open(a, std.testing.io, dir, .{ .cut = 40 }));
}

test "uneven stage budgets count only their own recurrent and KV state" {
    const model = try admitted();
    const first = try pipeline.memoryBudget(.{ .weights_bytes = 1024, .staging_bytes = 256 }, model, .{ .first = 0, .end = 3, .total = 40 }, 4096, 1);
    const last = try pipeline.memoryBudget(.{ .weights_bytes = 1024, .staging_bytes = 256 }, model, .{ .first = 3, .end = 40, .total = 40 }, 4096, 1);
    const recurrence = 32 * 128 * 128 * 4;
    const tail = 3 * 8192 * 2;
    const log = 32 * 4 + 32 * 128 * 4 + 16 * 128 * 2 + tail;
    const kv = 10 * 2 * 2 * 4096 * 256 * 2;
    try std.testing.expectEqual(@as(usize, 3 * (recurrence + tail) + ((3 * log + 255) / 256) * 256), first.stream_bytes);
    try std.testing.expectEqual(@as(usize, 27 * (recurrence + tail) + ((27 * log + 255) / 256) * 256 + kv), last.stream_bytes);
    try std.testing.expectEqual(first.steady_bytes + pipeline.floor_bytes, first.required_bytes);
    try pipeline.checkCapacity(first.required_bytes, first.required_bytes);
    try std.testing.expectError(error.InsufficientGpuMemory, pipeline.checkCapacity(first.required_bytes - 1, first.required_bytes));
    try std.testing.expectError(error.Overflow, pipeline.memoryBudget(.{ .weights_bytes = std.math.maxInt(usize), .staging_bytes = 256 }, model, .{ .first = 0, .end = 3, .total = 40 }, 4096, 1));
}

test "session validates the entire append before advancing either stage" {
    var chain: pipeline.Chain = undefined;
    chain.context = 4;
    var logits: [8]u16 = @splat(0x3f80);
    var session: pipeline.Session = .{ .chain = &chain, .caches = undefined, .position = 1, .logits = &logits };
    try std.testing.expectError(error.InvalidToken, session.append(&.{ 1, 8 }));
    try std.testing.expectError(error.ContextFull, session.append(&.{ 1, 2, 3, 4 }));
    try std.testing.expectEqual(@as(usize, 1), session.position);
    try std.testing.expect(!session.poisoned);
    for (logits) |value| try std.testing.expectEqual(@as(u16, 0x3f80), value);
    try session.append(&.{});
    try std.testing.expectEqual(@as(usize, 1), session.position);
}

test "poisoned sessions refuse even empty appends until reset succeeds" {
    var session: pipeline.Session = .{ .chain = undefined, .caches = undefined, .logits = undefined, .poisoned = true };
    try std.testing.expectError(error.PoisonedSession, session.append(&.{}));
    try std.testing.expectError(error.PoisonedSession, session.append(&.{0}));
}

test "phase observers refuse captured execution before a graph can launch" {
    const Callback = struct {
        fn mark(_: *anyopaque, _: engine.Phase, _: usize) !void {
            return error.UnexpectedObserverCall;
        }
    };
    var sentinel: u8 = 0;
    var e: engine.Engine = undefined;
    e.observer = .{ .ptr = &sentinel, .mark = Callback.mark };
    e.captured = true;
    try std.testing.expectError(error.QwenObserverRequiresEager, e.execute(undefined));
}
