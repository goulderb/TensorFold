const std = @import("std");

pub fn validate(input: anytype) !void {
    if (input.moe_math == .legacy) return;
    inline for (@typeInfo(@TypeOf(input)).@"struct".field_names) |name| {
        const T = @FieldType(@TypeOf(input), name);
        if (T == bool and @field(input, name)) return error.InvalidInput;
        if (@typeInfo(T) == .optional and @field(input, name) != null) return error.InvalidInput;
    }
}

test "candidate math is confined to the ordinary private full-model probe" {
    const Fixture = struct {
        moe_math: enum { legacy, bf16_boundaries } = .legacy,
        terminal_ab: bool = false,
        boundary_screen: ?u8 = null,
    };
    try validate(Fixture{});
    try validate(Fixture{ .moe_math = .bf16_boundaries });
    try validate(Fixture{ .terminal_ab = true });
    try std.testing.expectError(error.InvalidInput, validate(Fixture{ .moe_math = .bf16_boundaries, .terminal_ab = true }));
    try std.testing.expectError(error.InvalidInput, validate(Fixture{ .moe_math = .bf16_boundaries, .boundary_screen = 1 }));
}
