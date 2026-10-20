const std = @import("std");

pub fn validate(input: anytype) !void {
    if (!input.terminal_ab) return;
    if (input.prompt.len != 64 or input.output_tokens != 128 or input.context != 4096 or input.cut != 20 or input.forced.len != 0) return error.InvalidInput;
    inline for (@typeInfo(@TypeOf(input)).@"struct".field_names) |name| {
        if (comptime !std.mem.eql(u8, name, "terminal_ab")) {
            const T = @FieldType(@TypeOf(input), name);
            if (T == bool and @field(input, name)) return error.InvalidInput;
            if (@typeInfo(T) == .optional and @field(input, name) != null) return error.InvalidInput;
        }
    }
}

const Fixture = struct {
    terminal_ab: bool = true,
    prompt: []const u32 = &@as([64]u32, @splat(1)),
    output_tokens: usize = 128,
    context: usize = 4096,
    cut: usize = 20,
    forced: []const u32 = &.{},
    layout_ab: bool = false,
    fixture: ?u8 = null,
};

test "terminal trial accepts only its exact workload and exclusive mode" {
    try validate(Fixture{});
    try validate(Fixture{ .terminal_ab = false, .layout_ab = true });
    try std.testing.expectError(error.InvalidInput, validate(Fixture{ .layout_ab = true }));
    try std.testing.expectError(error.InvalidInput, validate(Fixture{ .fixture = 1 }));
    try std.testing.expectError(error.InvalidInput, validate(Fixture{ .forced = &.{1} }));
    try std.testing.expectError(error.InvalidInput, validate(Fixture{ .prompt = &.{1} }));
    inline for (.{ "output_tokens", "context", "cut" }) |field| {
        var invalid: Fixture = .{};
        @field(invalid, field) -= 1;
        try std.testing.expectError(error.InvalidInput, validate(invalid));
    }
}

test "terminal trial JSON rejects unknown fields and nonboolean selectors" {
    const a = std.testing.allocator;
    try std.testing.expectError(error.UnknownField, std.json.parseFromSlice(Fixture, a, "{\"terminal_ab\":true,\"unknown\":true}", .{}));
    try std.testing.expectError(error.UnexpectedToken, std.json.parseFromSlice(Fixture, a, "{\"terminal_ab\":1}", .{}));
}
