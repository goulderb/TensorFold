const std = @import("std");

pub fn validate(input: anytype) !void {
    if (!input.layout_ab) return;
    if (input.prompt.len != 64 or input.output_tokens != 128 or input.context != 4096 or input.cut != 20 or input.forced.len != 0) return error.InvalidInput;
    inline for (@typeInfo(@TypeOf(input)).@"struct".field_names) |name| {
        if (comptime !std.mem.eql(u8, name, "layout_ab")) {
            const T = @FieldType(@TypeOf(input), name);
            if (T == bool and @field(input, name)) return error.InvalidInput;
            if (@typeInfo(T) == .optional and @field(input, name) != null) return error.InvalidInput;
        }
    }
}

test "layout trial rejects mixed modes and unmatched workload" {
    const Input = struct {
        layout_ab: bool = true,
        prompt: []const u32 = &@as([64]u32, @splat(1)),
        output_tokens: usize = 128,
        context: usize = 4096,
        cut: usize = 20,
        forced: []const u32 = &.{},
        other_mode: bool = false,
        fixture: ?u8 = null,
    };
    try validate(Input{});
    try validate(Input{ .layout_ab = false, .other_mode = true });
    try std.testing.expectError(error.InvalidInput, validate(Input{ .other_mode = true }));
    try std.testing.expectError(error.InvalidInput, validate(Input{ .fixture = 1 }));
    try std.testing.expectError(error.InvalidInput, validate(Input{ .forced = &.{1} }));
    try std.testing.expectError(error.InvalidInput, validate(Input{ .prompt = &.{1} }));
    try std.testing.expectError(error.InvalidInput, validate(Input{ .output_tokens = 127 }));
    try std.testing.expectError(error.InvalidInput, validate(Input{ .context = 4095 }));
    try std.testing.expectError(error.InvalidInput, validate(Input{ .cut = 19 }));
}
