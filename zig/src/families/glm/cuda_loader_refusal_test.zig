//! Expected-diagnostic subprocess: assertions must pass, while the production unused-tensor error log forces exit 1.
const std = @import("std");
const core = @import("core");
const cuda = @import("cuda");
const W = @import("cuda_weights.zig");
const M = @import("cuda_test_driver.zig").Memory;

test "GLM loader refusal: an unknown mixed-checkpoint tensor is rejected and all allocations are released" {
    const gpa = std.testing.allocator;
    const hidden = 64;
    const vocab = 128;
    const embed: [vocab * hidden]u16 = @splat(0x3f80);
    const norm: [hidden]u16 = @splat(0x3f80);
    const codes: [vocab * hidden]u8 = @splat(0x38);
    const scales: [vocab * hidden / 32]u8 = @splat(127);
    const tensors = [_]core.shard_edit.New{
        .{ .name = W.prefix ++ "embed_tokens.weight", .dtype = .bf16, .shape = &.{ vocab, hidden }, .bytes = std.mem.sliceAsBytes(&embed) },
        .{ .name = W.prefix ++ "norm.weight", .dtype = .bf16, .shape = &.{hidden}, .bytes = std.mem.sliceAsBytes(&norm) },
        .{ .name = "lm_head.weight", .dtype = .f8_e4m3, .shape = &.{ vocab, hidden }, .bytes = &codes },
        .{ .name = "lm_head.weight_scale", .dtype = .u8, .shape = &.{ vocab, hidden / 32 }, .bytes = &scales },
        .{ .name = W.prefix ++ "unknown", .dtype = .u8, .shape = &.{1}, .bytes = &.{0} },
    };
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var dir: [96]u8 = undefined;
    const at = try std.fmt.bufPrint(&dir, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    var path: [128]u8 = undefined;
    try core.shard_edit.create(gpa, try std.fmt.bufPrint(&path, "{s}/model.safetensors", .{at}), &.{}, &tensors);
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "config.json", .data =
        \\{"quantization_config":{"quant_method":"modelopt","quant_algo":"MIXED_PRECISION","quantized_layers":{"lm_head":{"quant_algo":"MXFP8","group_size":32}}}}
    });
    var driver = M.driver();
    var lane: cuda.qmmf.Lane = undefined; // only the CPU shape planner is read; no kernel is launched
    lane.clusters = true;
    const tools: W.Tools = .{ .pack_dense = .{ .d = &driver, .handle = @ptrFromInt(1) }, .lane = &lane };
    const c: @import("config.zig").Config = .{ .hidden = hidden, .vocab = vocab, .layers = 0, .mtp = 0 };
    const share = try @import("cuda_shard.zig").Share.of(&c, 0, 1);
    const loaded = W.load(gpa, std.testing.io, &driver, .{ .d = &driver, .handle = null }, tools, at, c, share, &.{}, false);
    if (loaded) |value| {
        var unexpected = value;
        unexpected.deinit();
        return error.ExpectedUnusedTensorRefusal;
    } else |err| try std.testing.expectEqual(error.UnusedTensors, err);
    try std.testing.expect(M.empty());
}
