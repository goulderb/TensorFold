const std = @import("std");

pub fn validateInput(input: anytype) !void {
    const modes = @as(usize, @intFromBool(input.prefill_operator_screen)) + @intFromBool(input.prefill_ab) + @intFromBool(input.combined_ab);
    if (modes == 0) return;
    if (modes != 1) return error.InvalidInput;
    if (input.expert_fixture != null or input.expert_n8_screen or input.expert_shard_screen or
        input.expert_latency_screen or input.gdn_layout_screen or input.gdn_layout_ab or
        input.expert_n8_ab or input.quant_reuse_screen or input.transport_screen or
        input.verify_default_policy or input.router_turns or input.router_screen or
        input.router_candidate or input.forced.len != 0) return error.InvalidInput;
}

pub fn run(gpu: @import("check.zig").Gpu, model_dir: []const u8, journal: anytype) !void {
    try @import("qwen35_prefill_linear_screen.zig").run(gpu, model_dir, journal);
    try @import("qwen35_prefill_gdn_screen.zig").run(gpu, model_dir, journal);
}

test "prefill operator admission excludes every competing diagnostic mode" {
    const Input = struct {
        prefill_operator_screen: bool = true,
        prefill_ab: bool = false,
        combined_ab: bool = false,
        expert_fixture: ?[]const u8 = null,
        expert_n8_screen: bool = false,
        expert_shard_screen: bool = false,
        expert_latency_screen: bool = false,
        gdn_layout_screen: bool = false,
        gdn_layout_ab: bool = false,
        expert_n8_ab: bool = false,
        quant_reuse_screen: bool = false,
        transport_screen: bool = false,
        verify_default_policy: bool = false,
        router_turns: bool = false,
        router_screen: bool = false,
        router_candidate: bool = false,
        forced: []const u32 = &.{},
    };
    try validateInput(Input{});
    try validateInput(Input{ .prefill_operator_screen = false, .prefill_ab = true });
    try std.testing.expectError(error.InvalidInput, validateInput(Input{ .prefill_ab = true }));
    try validateInput(Input{ .prefill_operator_screen = false, .combined_ab = true });
    try std.testing.expectError(error.InvalidInput, validateInput(Input{ .combined_ab = true }));
    try std.testing.expectError(error.InvalidInput, validateInput(Input{ .prefill_operator_screen = false, .prefill_ab = true, .combined_ab = true }));
    inline for (.{ "expert_n8_screen", "expert_shard_screen", "expert_latency_screen", "gdn_layout_screen", "gdn_layout_ab", "expert_n8_ab", "quant_reuse_screen", "transport_screen", "verify_default_policy", "router_turns", "router_screen", "router_candidate" }) |flag| {
        var input: Input = .{};
        @field(input, flag) = true;
        try std.testing.expectError(error.InvalidInput, validateInput(input));
    }
    try std.testing.expectError(error.InvalidInput, validateInput(Input{ .expert_fixture = "fixture" }));
    try std.testing.expectError(error.InvalidInput, validateInput(Input{ .forced = &.{1} }));
}
