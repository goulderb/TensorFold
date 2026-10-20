const std = @import("std");

pub const Image = struct { path: []const u8, sha256: []const u8 };
pub const Mma = struct { fixture: []const u8, fixture_sha256: []const u8, image: []const u8, image_sha256: []const u8 };
pub const Suite = struct { stage3: Image, mma: Mma };

fn pin(path: []const u8, sha: []const u8) !void {
    if (path.len == 0 or sha.len != 64) return error.InvalidInput;
    for (sha) |c| if (!(c >= '0' and c <= '9') and !(c >= 'a' and c <= 'f')) return error.InvalidInput;
}

pub fn validateInput(input: anytype) !void {
    const suite = input.operator_suite orelse return;
    try pin(suite.stage3.path, suite.stage3.sha256);
    try pin(suite.mma.fixture, suite.mma.fixture_sha256);
    try pin(suite.mma.image, suite.mma.image_sha256);
    if (input.expert_fixture != null or input.forced.len != 0) return error.InvalidInput;
    inline for (.{ "prefill_operator_screen", "prefill_ab", "combined_ab", "expert_n8_screen", "expert_shard_screen", "expert_latency_screen", "gdn_layout_screen", "gdn_layout_ab", "expert_n8_ab", "quant_reuse_screen", "transport_screen", "verify_default_policy", "router_turns", "router_screen", "router_candidate" }) |flag| {
        if (@field(input, flag)) return error.InvalidInput;
    }
}

test "operator suite requires all external image and fixture identities" {
    const sha = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    try pin("image", sha);
    try std.testing.expectError(error.InvalidInput, pin("", sha));
    try std.testing.expectError(error.InvalidInput, pin("image", "short"));
    try std.testing.expectError(error.InvalidInput, pin("image", "g123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"));
}

test "operator suite excludes every existing diagnostic mode" {
    const sha = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    const Input = struct {
        operator_suite: ?Suite = .{ .stage3 = .{ .path = "stage3", .sha256 = sha }, .mma = .{ .fixture = "fixture", .fixture_sha256 = sha, .image = "mma", .image_sha256 = sha } },
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
    try validateInput(Input{ .operator_suite = null, .combined_ab = true });
    inline for (.{ "prefill_operator_screen", "prefill_ab", "combined_ab", "expert_n8_screen", "expert_shard_screen", "expert_latency_screen", "gdn_layout_screen", "gdn_layout_ab", "expert_n8_ab", "quant_reuse_screen", "transport_screen", "verify_default_policy", "router_turns", "router_screen", "router_candidate" }) |flag| {
        var input: Input = .{};
        @field(input, flag) = true;
        try std.testing.expectError(error.InvalidInput, validateInput(input));
    }
    try std.testing.expectError(error.InvalidInput, validateInput(Input{ .expert_fixture = "fixture" }));
    try std.testing.expectError(error.InvalidInput, validateInput(Input{ .forced = &.{1} }));
}
