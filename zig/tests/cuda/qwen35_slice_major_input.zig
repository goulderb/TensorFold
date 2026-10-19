const std = @import("std");

pub const Screen = struct {
    fixture: []const u8,
    fixture_sha256: []const u8,
    image: []const u8,
    image_sha256: []const u8,
};

fn pin(path: []const u8, sha: []const u8) !void {
    if (!std.fs.path.isAbsolute(path) or sha.len != 64) return error.InvalidInput;
    for (sha) |c| if (!(c >= '0' and c <= '9') and !(c >= 'a' and c <= 'f')) return error.InvalidInput;
}

pub fn validateInput(input: anytype) !void {
    const screen = input.slice_major_screen orelse return;
    try pin(screen.fixture, screen.fixture_sha256);
    try pin(screen.image, screen.image_sha256);
    if (input.operator_suite != null or input.expert_fixture != null or input.forced.len != 0) return error.InvalidInput;
    inline for (.{ "prefill_operator_screen", "prefill_ab", "combined_ab", "expert_n8_screen", "expert_shard_screen", "expert_latency_screen", "gdn_layout_screen", "gdn_layout_ab", "expert_n8_ab", "quant_reuse_screen", "transport_screen", "verify_default_policy", "router_turns", "router_screen", "router_candidate" }) |flag| {
        if (@field(input, flag)) return error.InvalidInput;
    }
}

test "slice-major admission requires both immutable input identities" {
    const sha = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    const path = if (@import("builtin").os.tag == .windows) "C:/fixture" else "/fixture";
    try pin(path, sha);
    try std.testing.expectError(error.InvalidInput, pin("", sha));
    try std.testing.expectError(error.InvalidInput, pin("relative", sha));
    try std.testing.expectError(error.InvalidInput, pin(path, "short"));
    try std.testing.expectError(error.InvalidInput, pin(path, "G123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"));
}

test "slice-major admission excludes every other diagnostic" {
    const sha = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    const path = if (@import("builtin").os.tag == .windows) "C:/fixture" else "/fixture";
    const Input = struct {
        slice_major_screen: ?Screen = .{ .fixture = path, .fixture_sha256 = sha, .image = path, .image_sha256 = sha },
        operator_suite: ?u8 = null,
        expert_fixture: ?[]const u8 = null,
        forced: []const u32 = &.{},
        prefill_operator_screen: bool = false,
        prefill_ab: bool = false,
        combined_ab: bool = false,
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
    };
    try validateInput(Input{});
    try validateInput(Input{ .slice_major_screen = null, .operator_suite = 1 });
    inline for (.{ "prefill_operator_screen", "prefill_ab", "combined_ab", "expert_n8_screen", "expert_shard_screen", "expert_latency_screen", "gdn_layout_screen", "gdn_layout_ab", "expert_n8_ab", "quant_reuse_screen", "transport_screen", "verify_default_policy", "router_turns", "router_screen", "router_candidate" }) |flag| {
        var input: Input = .{};
        @field(input, flag) = true;
        try std.testing.expectError(error.InvalidInput, validateInput(input));
    }
    try std.testing.expectError(error.InvalidInput, validateInput(Input{ .operator_suite = 1 }));
    try std.testing.expectError(error.InvalidInput, validateInput(Input{ .expert_fixture = "fixture" }));
    try std.testing.expectError(error.InvalidInput, validateInput(Input{ .forced = &.{1} }));
}
