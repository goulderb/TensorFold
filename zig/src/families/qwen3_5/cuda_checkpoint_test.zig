const std = @import("std");
const core = @import("core");
const ck = @import("cuda_checkpoint.zig");
const Tensor = core.checkpoint.Tensor;

pub const Header = struct {
    arena: std.mem.Allocator,
    entries: std.json.ObjectMap,
    used: std.StringHashMapUnmanaged(void) = .empty,

    pub fn parse(arena: std.mem.Allocator, bytes: []const u8) !Header {
        var input: std.Io.Reader = .fixed(bytes);
        var gz: std.compress.flate.Decompress = .init(&input, .gzip, &.{});
        var out: std.Io.Writer.Allocating = .init(arena);
        _ = try gz.reader.streamRemaining(&out.writer);
        const doc = try std.json.parseFromSliceLeaky(std.json.Value, arena, out.written(), .{});
        return .{ .arena = arena, .entries = doc.object };
    }

    pub fn has(h: *const Header, name: []const u8) bool {
        return h.entries.contains(name);
    }

    pub fn get(h: *Header, name: []const u8) !Tensor {
        const pair = (h.entries.get(name) orelse return error.MissingTensor).array.items;
        var t: Tensor = .{ .dtype = core.checkpoint.DType.parse(pair[0].string).?, .rank = @intCast(pair[1].array.items.len), .shape = @splat(1), .bytes = &.{} };
        for (pair[1].array.items, 0..) |d, i| t.shape[i] = @intCast(d.integer);
        try h.used.put(h.arena, h.entries.getKey(name).?, {});
        return t;
    }
};

test "cuda Qwen checkpoint admits both official ModelOpt configs and refuses compressed-tensors" {
    const a = std.testing.allocator;
    inline for (.{ "ornith_nvfp4", "nvidia_qwen36_nvfp4" }) |fixture| {
        const cfg = try ck.Config.parse(a, @embedFile("fixtures/" ++ fixture ++ "/config.json"));
        try std.testing.expectEqual(@as(usize, 40), cfg.spec.n_layers);
        try std.testing.expectEqual(@as(usize, 262144), cfg.context);
        try std.testing.expectEqual(@as(usize, 256), cfg.spec.experts);
        try std.testing.expectEqual(@as(usize, 4096), cfg.spec.valueWidth());
    }
    inline for (.{ "unsloth_qwen36_nvfp4", "unsloth_qwen36_nvfp4_fast" }) |fixture|
        try std.testing.expectError(error.UnsupportedQwenQuantization, ck.Config.parse(a, @embedFile("fixtures/" ++ fixture ++ "/config.json")));
    const config = @embedFile("fixtures/ornith_nvfp4/config.json");
    const changes = .{
        .{ "\"num_experts\": 256", "\"num_experts\": 128" },
        .{ "\"num_hidden_layers\": 40", "\"num_hidden_layers\": 39" },
        .{ "\"head_dim\": 256", "\"head_dim\": 128" },
        .{ "\"num_key_value_heads\": 2", "\"num_key_value_heads\": 0" },
        .{ "\"max_position_embeddings\": 262144", "\"max_position_embeddings\": 524288" },
    };
    inline for (changes) |change| {
        const bad = try std.mem.replaceOwned(u8, a, config, change[0], change[1]);
        defer a.free(bad);
        if (ck.Config.parse(a, bad)) |_| return error.TestExpectedError else |_| {}
    }
    const activation_fp4 = try std.mem.replaceOwned(u8, a, config, "W4A16_NVFP4", "NVFP4");
    defer a.free(activation_fp4);
    try std.testing.expectError(error.UnsupportedQwenQuantization, ck.Config.parse(a, activation_fp4));
}

test "cuda Qwen checkpoint validates every official text header and optional NVFP4 input scales" {
    inline for (.{ "ornith_nvfp4", "nvidia_qwen36_nvfp4" }) |fixture| {
        var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
        defer arena.deinit();
        const cfg = try ck.Config.parse(arena.allocator(), @embedFile("fixtures/" ++ fixture ++ "/config.json"));
        var h = try Header.parse(arena.allocator(), @embedFile("fixtures/" ++ fixture ++ "/headers.json.gz"));
        try ck.validateText(&h, cfg, 0, 40, true, true);
        var count: usize = 0;
        var it = h.entries.iterator();
        while (it.next()) |entry| {
            const n = entry.key_ptr.*;
            if (!std.mem.startsWith(u8, n, "model.language_model.") and !std.mem.startsWith(u8, n, "lm_head.")) continue;
            try std.testing.expect(h.used.contains(n));
            count += 1;
        }
        try std.testing.expect(count > 92000);
        const input = "model.language_model.layers.0.mlp.experts.0.gate_proj.input_scale";
        try std.testing.expectEqual(!std.mem.eql(u8, fixture, "ornith_nvfp4"), h.has(input));
        const conv = try h.get("model.language_model.layers.0.linear_attn.conv1d.weight");
        try std.testing.expectEqualSlices(usize, &.{ 8192, 1, 4 }, conv.shape[0..conv.rank]);
        const key = "model.language_model.layers.0.mlp.experts.0.gate_proj.weight_scale";
        h.entries.getPtr(key).?.array.items[1].array.items[1].integer = 127;
        try std.testing.expectError(error.UnexpectedTensor, ck.projectionLayout(&h, "model.language_model.layers.0.mlp.experts.0.gate_proj", 512, 2048));
    }
}

test "cuda Qwen checkpoint refuses dynamic FP4 group metadata even with W4A16 layer tags" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const doc = try std.json.parseFromSliceLeaky(std.json.Value, a, @embedFile("fixtures/nvidia_qwen36_nvfp4/config.json"), .{});
    const quant = doc.object.get("quantization_config").?.object;
    const group = quant.get("config_groups").?.object.get("group_1").?.object;
    var activation = group.get("input_activations").?.object;
    activation.getPtr("dynamic").?.* = .{ .bool = true };
    var serialized: std.Io.Writer.Allocating = .init(a);
    try std.json.Stringify.value(doc, .{}, &serialized.writer);
    try std.testing.expectError(error.UnsupportedQwenQuantization, ck.Config.parse(a, serialized.written()));
}

test "cuda Qwen checkpoint preserves raw GDN parameters and converts only zero-centered norm roles" {
    const a = std.testing.allocator;
    const bytes = [_]u8{ 0x00, 0x00, 0x80, 0x3f, 0x00, 0xbf, 0x00, 0x40 };
    const t: Tensor = .{ .dtype = .bf16, .rank = 1, .shape = .{ 4, 1, 1, 1, 1 }, .bytes = &bytes };
    const raw = try ck.floats(a, t, .raw);
    defer a.free(raw);
    const norm = try ck.floats(a, t, .zero_centered_norm);
    defer a.free(norm);
    try std.testing.expectEqualSlices(f32, &.{ 0, 1, -0.5, 2 }, raw);
    try std.testing.expectEqualSlices(f32, &.{ 1, 2, 0.5, 3 }, norm);
    var scalar: Tensor = .{ .dtype = .f32, .rank = 0, .shape = @splat(1), .bytes = std.mem.asBytes(&@as(f32, 0.5)) };
    try std.testing.expectEqual(@as(f32, 0.5), try ck.scalar(scalar));
    scalar.bytes = std.mem.asBytes(&std.math.nan(f32));
    try std.testing.expectError(error.InvalidQwenScale, ck.scalar(scalar));
    scalar.bytes = std.mem.asBytes(&@as(f32, -0.5));
    try std.testing.expectError(error.InvalidQwenScale, ck.scalar(scalar));
}

test "cuda Qwen checkpoint mapped projections retain static FP8 scale and weight-only NVFP4" {
    const header =
        \\{"q4.weight":{"dtype":"U8","shape":[2,32],"data_offsets":[0,64]},
        \\ "q4.weight_scale":{"dtype":"F8_E4M3","shape":[2,4],"data_offsets":[64,72]},
        \\ "q4.weight_scale_2":{"dtype":"F32","shape":[],"data_offsets":[72,76]},
        \\ "q8.weight":{"dtype":"F8_E4M3","shape":[2,64],"data_offsets":[76,204]},
        \\ "q8.weight_scale":{"dtype":"F32","shape":[],"data_offsets":[204,208]},
        \\ "q8.input_scale":{"dtype":"F32","shape":[],"data_offsets":[208,212]},
        \\ "bf.weight":{"dtype":"BF16","shape":[2,64],"data_offsets":[212,468]}}
    ;
    var image: [8 + header.len + 468]u8 = @splat(0);
    std.mem.writeInt(u64, image[0..8], header.len, .little);
    @memcpy(image[8..][0..header.len], header);
    const data = image[8 + header.len ..];
    @memcpy(data[72..76], std.mem.asBytes(&@as(f32, 0.25)));
    @memcpy(data[204..208], std.mem.asBytes(&@as(f32, 0.5)));
    @memcpy(data[208..212], std.mem.asBytes(&@as(f32, 3.0)));
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "model.safetensors", .data = &image });
    var path: [96]u8 = undefined;
    var mapped = try core.Checkpoint.openModel(std.testing.allocator, std.testing.io, try std.fmt.bufPrint(&path, ".zig-cache/tmp/{s}", .{tmp.sub_path}));
    defer mapped.close();
    const view: ck.View = .{ .ck = &mapped };
    const q4 = try view.part("q4", 2, 64);
    const q8 = try view.part("q8", 2, 64);
    const bf = try view.part("bf", 2, 64);
    try std.testing.expectEqual(ck.Kind.nvfp4, q4.kind);
    try std.testing.expectEqual(@as(f32, 0.25), q4.scale);
    try std.testing.expectEqual(@as(f32, 0), q4.act);
    try std.testing.expectEqual(@as(f32, 0.5), q8.scale);
    try std.testing.expectEqual(@as(f32, 3), q8.act);
    try std.testing.expectEqual(ck.Kind.bf16, bf.kind);
    try std.testing.expectEqual(@as(usize, 0), mapped.unused());
}

test "cuda Qwen checkpoint refuses extra tensors only in the loaded text partition" {
    const header =
        \\{"model.language_model.layers.0.extra":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]},
        \\ "model.language_model.layers.20.extra":{"dtype":"BF16","shape":[1],"data_offsets":[2,4]},
        \\ "model.language_model.embed_tokens.weight":{"dtype":"BF16","shape":[1],"data_offsets":[4,6]},
        \\ "model.visual.extra":{"dtype":"BF16","shape":[1],"data_offsets":[6,8]},
        \\ "mtp.extra":{"dtype":"BF16","shape":[1],"data_offsets":[8,10]},
        \\ "lm_head.weight":{"dtype":"BF16","shape":[1],"data_offsets":[10,12]}}
    ;
    var image: [8 + header.len + 12]u8 = @splat(0);
    std.mem.writeInt(u64, image[0..8], header.len, .little);
    @memcpy(image[8..][0..header.len], header);
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "model.safetensors", .data = &image });
    var path: [96]u8 = undefined;
    var mapped = try core.Checkpoint.openModel(std.testing.allocator, std.testing.io, try std.fmt.bufPrint(&path, ".zig-cache/tmp/{s}", .{tmp.sub_path}));
    defer mapped.close();
    try ck.refuseUnusedText(&mapped, 1, 20, false, false);
    try std.testing.expectError(error.UnexpectedQwenTensor, ck.refuseUnusedText(&mapped, 0, 20, false, false));
    _ = try mapped.get("model.language_model.layers.0.extra");
    try ck.refuseUnusedText(&mapped, 0, 20, false, false);
    try std.testing.expectError(error.UnexpectedQwenTensor, ck.refuseUnusedText(&mapped, 0, 20, false, true));
    _ = try mapped.get("lm_head.weight");
    try std.testing.expectError(error.UnexpectedQwenTensor, ck.refuseUnusedText(&mapped, 0, 20, true, true));
    _ = try mapped.get("model.language_model.embed_tokens.weight");
    try std.testing.expectError(error.UnexpectedQwenTensor, ck.refuseUnusedText(&mapped, 20, 40, true, true));
    _ = try mapped.get("model.language_model.layers.20.extra");
    try ck.refuseUnusedText(&mapped, 0, 40, true, true);
}
