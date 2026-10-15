//! Admission of the native family's ModelOpt NVFP4 and EXL3 checkpoint metadata.
const std = @import("std");
const modelopt = @import("core").modelopt;

pub const Format = enum { nvfp4, exl3, routed_exl3 };

pub fn admit(gpa: std.mem.Allocator, bytes: []const u8, why: *modelopt.Why) !Format {
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, bytes, .{});
    defer parsed.deinit();
    const root = parsed.value;
    if (root != .object) return error.InvalidCheckpointConfig;
    const text = root.object.get("text_config") orelse root;
    const q = root.object.get("quantization_config") orelse
        (if (text == .object) text.object.get("quantization_config") else null) orelse return error.MissingQuantization;
    if (q != .object) return error.InvalidQuantization;
    const method = q.object.get("quant_method") orelse return error.InvalidQuantization;
    if (method != .string) return error.InvalidQuantization;
    if (std.ascii.eqlIgnoreCase(method.string, "modelopt")) {
        try modelopt.admit(gpa, bytes, why);
        return .nvfp4;
    }
    if (!std.mem.eql(u8, method.string, "exl3")) return error.UnsupportedQuant;
    if (q.object.get("scope")) |scope| {
        if (scope != .string or !std.mem.eql(u8, scope.string, "glm53_routed_experts_only")) return error.UnsupportedExl3Scope;
        const bits = q.object.get("bits") orelse return error.InvalidQuantization;
        const book = q.object.get("codebook") orelse return error.InvalidQuantization;
        const head = q.object.get("head_bits") orelse return error.InvalidQuantization;
        if (bits != .integer or bits.integer != 4 or book != .string or !std.mem.eql(u8, book.string, "mcg") or head != .integer or head.integer != 16) return error.UnsupportedMixedExl3;
        return .routed_exl3;
    }
    return .exl3;
}

test "the routed-only checkpoint keeps its dense and head precision" {
    var why: modelopt.Why = .{};
    try std.testing.expectEqual(Format.routed_exl3, try admit(std.testing.allocator, @embedFile("fixtures/mia-routed-exl3-config.json"), &why));
    try std.testing.expectEqual(Format.routed_exl3, try admit(std.testing.allocator,
        \\{"quantization_config":{"quant_method":"exl3","bits":4,"codebook":"mcg","head_bits":16,"scope":"glm53_routed_experts_only"}}
    , &why));
    try std.testing.expectError(error.UnsupportedMixedExl3, admit(std.testing.allocator,
        \\{"quantization_config":{"quant_method":"exl3","bits":4,"codebook":"mcg","head_bits":8,"scope":"glm53_routed_experts_only"}}
    , &why));
}

test "EXL3 metadata is distinct from ModelOpt and unknown scopes refuse" {
    var why: modelopt.Why = .{};
    try std.testing.expectEqual(Format.exl3, try admit(std.testing.allocator, "{\"quantization_config\":{\"quant_method\":\"exl3\"}}", &why));
    try std.testing.expectError(error.UnsupportedExl3Scope, admit(std.testing.allocator, "{\"quantization_config\":{\"quant_method\":\"exl3\",\"scope\":\"other\"}}", &why));
    try std.testing.expectError(error.UnsupportedQuant, admit(std.testing.allocator, "{\"quantization_config\":{\"quant_method\":\"modelopt\",\"quant_algo\":\"FP8\"}}", &why));
}
