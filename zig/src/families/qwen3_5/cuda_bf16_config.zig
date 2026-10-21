const std = @import("std");
const dimensions = @import("cuda_geometry.zig");
const checks = @import("config_checks.zig");

pub fn parse(gpa: std.mem.Allocator, bytes: []const u8) !dimensions.Geometry {
    const doc = try std.json.parseFromSlice(std.json.Value, gpa, bytes, .{});
    defer doc.deinit();
    if (doc.value != .object) return error.UnsupportedQwenConfig;
    const root = doc.value.object;
    const value = root.get("text_config") orelse return error.UnsupportedQwenConfig;
    if (value != .object) return error.UnsupportedQwenConfig;
    const tower = value.object;
    for ([_]std.json.ObjectMap{ root, tower }) |o| {
        if (o.contains("quantization") or o.contains("quantization_config")) return error.UnsupportedQwenQuantization;
        if (!string(o.get("dtype"), "bfloat16")) return error.UnsupportedQwenDtype;
        if (o.get("torch_dtype")) |v| if (!string(v, "bfloat16")) return error.UnsupportedQwenDtype;
        const tied = o.get("tie_word_embeddings") orelse return error.UnsupportedTiedHead;
        if (tied != .bool or tied.bool) return error.UnsupportedTiedHead;
    }
    const g = try dimensions.dimensions(root);
    if (g.moe() or g.hidden != 4096 or g.intermediate != 12288 or g.layers != 32 or
        g.vocab != 248320 or g.k_heads != 16 or g.v_heads != 32 or g.q_heads != 16 or
        g.kv_heads != 4 or g.eps != 1e-6 or g.context > 262144) return error.UnsupportedQwenGeometry;
    for (g.linear[0..g.layers], 0..) |linear, i| if (linear != (i % 4 != 3)) return error.UnsupportedQwenGeometry;
    try checks.computes(tower);
    if (!string(tower.get("mamba_ssm_dtype"), "float32") or
        !integer(tower.get("full_attention_interval"), 4) or
        !integer(tower.get("mtp_num_hidden_layers"), 0) or
        !zero(tower.get("attention_dropout"))) return error.UnsupportedQwenConfig;
    inline for (.{ "num_experts", "num_experts_per_tok", "shared_expert_intermediate_size" }) |key| {
        if (tower.get(key)) |v| if (!integer(v, 0)) return error.UnsupportedQwenGeometry;
    }
    const mlp = tower.get("mlp_only_layers") orelse return error.UnsupportedQwenConfig;
    if (mlp != .array or mlp.array.items.len != 0) return error.UnsupportedQwenGeometry;
    const rope = tower.get("rope_parameters").?.object;
    const interleaved = rope.get("mrope_interleaved") orelse return error.UnsupportedQwenRotary;
    if (interleaved != .bool or !interleaved.bool) return error.UnsupportedQwenRotary;
    const section = rope.get("mrope_section") orelse return error.UnsupportedQwenRotary;
    if (section != .array or section.array.items.len != 3) return error.UnsupportedQwenRotary;
    for (section.array.items, [_]i64{ 11, 11, 10 }) |item, want| if (!integer(item, want)) return error.UnsupportedQwenRotary;
    return g;
}

test "plain BF16 admits the pinned dense Clef geometry without MLX widths" {
    const g = try parse(std.testing.allocator, @embedFile("fixtures/clef_flash/config.json"));
    try std.testing.expectEqual(@as(usize, 4096), g.hidden);
    try std.testing.expectEqual(@as(usize, 24), g.deltas());
    try std.testing.expectEqual(@as(usize, 8), g.attentions());
    try std.testing.expect(!g.moe());
    try std.testing.expect(!g.tied);
}

test "plain BF16 refuses unsupported declarations before payload access" {
    const a = std.testing.allocator;
    const fixture = @embedFile("fixtures/clef_flash/config.json");
    const changes = .{
        .{ "\"dtype\": \"bfloat16\"", "\"dtype\": \"float16\"" },
        .{ "\"hidden_size\": 4096", "\"hidden_size\": 2048" },
        .{ "\"intermediate_size\": 12288", "\"intermediate_size\": 6144" },
        .{ "\"num_key_value_heads\": 4", "\"num_key_value_heads\": 2" },
        .{ "\"tie_word_embeddings\": false", "\"tie_word_embeddings\": true" },
        .{ "\"attention_dropout\": 0.0", "\"attention_dropout\": 0.1" },
        .{ "\"mtp_num_hidden_layers\": 0", "\"mtp_num_hidden_layers\": 1" },
        .{ "\"mamba_ssm_dtype\": \"float32\"", "\"mamba_ssm_dtype\": \"bfloat16\"" },
        .{ "\"mrope_interleaved\": true", "\"mrope_interleaved\": false" },
        .{ "\"full_attention_interval\": 4", "\"full_attention_interval\": 8" },
        .{ "\"mlp_only_layers\": []", "\"mlp_only_layers\": [0]" },
        .{ "\"dtype\": \"bfloat16\"", "\"dtype\": \"bfloat16\", \"quantization\": null" },
        .{ "\"dtype\": \"bfloat16\"", "\"dtype\": \"bfloat16\", \"quantization_config\": {}" },
        .{ "\"dtype\": \"bfloat16\"", "\"dtype\": \"bfloat16\", \"torch_dtype\": \"float32\"" },
        .{ "\"max_position_embeddings\": 262144", "\"max_position_embeddings\": 524288" },
    };
    inline for (changes) |change| {
        try std.testing.expect(std.mem.indexOf(u8, fixture, change[0]) != null);
        const bad = try std.mem.replaceOwned(u8, a, fixture, change[0], change[1]);
        defer a.free(bad);
        if (parse(a, bad)) |_| return error.TestExpectedError else |_| {}
    }
    try std.testing.expectError(error.UnsupportedQwenConfig, parse(a, "[]"));
}

test "plain BF16 requires explicit supported declarations and layer order" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const fixture = @embedFile("fixtures/clef_flash/config.json");
    const required = [_]struct { tower: bool, key: []const u8 }{
        .{ .tower = false, .key = "dtype" },
        .{ .tower = true, .key = "dtype" },
        .{ .tower = false, .key = "tie_word_embeddings" },
        .{ .tower = true, .key = "tie_word_embeddings" },
        .{ .tower = true, .key = "mamba_ssm_dtype" },
        .{ .tower = true, .key = "mtp_num_hidden_layers" },
        .{ .tower = true, .key = "attention_dropout" },
    };
    for (required) |field| {
        var doc = try std.json.parseFromSliceLeaky(std.json.Value, a, fixture, .{});
        const holder = if (field.tower) &doc.object.getPtr("text_config").?.object else &doc.object;
        try std.testing.expect(holder.swapRemove(field.key));
        var serialized: std.Io.Writer.Allocating = .init(a);
        try std.json.Stringify.value(doc, .{}, &serialized.writer);
        if (parse(a, serialized.written())) |_| return error.TestExpectedError else |_| {}
    }
    const doc = try std.json.parseFromSliceLeaky(std.json.Value, a, fixture, .{});
    doc.object.getPtr("text_config").?.object.getPtr("layer_types").?.array.items[3] = .{ .string = "linear_attention" };
    var serialized: std.Io.Writer.Allocating = .init(a);
    try std.json.Stringify.value(doc, .{}, &serialized.writer);
    try std.testing.expectError(error.UnsupportedQwenGeometry, parse(a, serialized.written()));
}

fn string(v: ?std.json.Value, want: []const u8) bool {
    const x = v orelse return false;
    return x == .string and std.mem.eql(u8, x.string, want);
}

fn integer(v: ?std.json.Value, want: i64) bool {
    const x = v orelse return false;
    return x == .integer and x.integer == want;
}

fn zero(v: ?std.json.Value) bool {
    const x = v orelse return false;
    return switch (x) {
        .integer => x.integer == 0,
        .float => x.float == 0,
        else => false,
    };
}
