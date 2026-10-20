const std = @import("std");

pub const Fixture = struct { fixture: []const u8, fixture_sha256: []const u8 };
pub const Activation = struct { fixture: []const u8, fixture_sha256: []const u8, image: []const u8, image_sha256: []const u8 };
pub const Screen = union(enum) { terminal_greedy: Fixture, terminal_graph: Fixture, terminal_ownership: Fixture, shared_activation: Activation, moe_semantics: Activation, shared_branch: Activation, full_moe: Activation, integrated_moe: Activation, slice_format: Fixture };

fn pin(path: []const u8, sha: []const u8) !void {
    if (!std.fs.path.isAbsolute(path) or sha.len != 64) return error.InvalidInput;
    for (sha) |c| if (!(c >= '0' and c <= '9') and !(c >= 'a' and c <= 'f')) return error.InvalidInput;
}

pub fn scope(screen: Screen) []const u8 {
    return switch (screen) {
        .terminal_greedy => "isolated terminal greedy transfer and selection; no model throughput claim",
        .terminal_graph => "isolated matched carrier graph terminal selection; no model throughput claim",
        .terminal_ownership => "isolated full-logit ownership and native greedy selection; no model throughput claim",
        .shared_activation => "isolated shared activation HF boundary; no model quality or performance claim",
        .moe_semantics => "isolated expert HF arithmetic boundaries; no full-model quality or performance claim",
        .shared_branch => "isolated activation and shared-down HF composition; no full-model quality or performance claim",
        .slice_format => "isolated production expert packing and tagged consumers; no model performance or quality claim",
        .integrated_moe => "isolated integrated-image fixed-operand math; no fresh complete projection chain or full-model claim",
        .full_moe => "isolated native-input full-MoE HF composition; no full-model quality or performance claim",
    };
}

pub fn validateInput(input: anytype) !void {
    const screen = input.boundary_screen orelse return;
    switch (screen) {
        .terminal_greedy, .terminal_graph, .terminal_ownership, .slice_format => |f| try pin(f.fixture, f.fixture_sha256),
        .shared_activation, .moe_semantics, .shared_branch, .full_moe, .integrated_moe => |f| {
            try pin(f.fixture, f.fixture_sha256);
            try pin(f.image, f.image_sha256);
        },
    }
    if (screen == .integrated_moe) {
        inline for (@typeInfo(@TypeOf(input)).@"struct".field_names) |name| {
            const T = @FieldType(@TypeOf(input), name);
            if (T == bool and @field(input, name)) return error.InvalidInput;
            if (@typeInfo(T) == .optional and !std.mem.eql(u8, name, "boundary_screen") and @field(input, name) != null) return error.InvalidInput;
        }
        if (@hasField(@TypeOf(input), "moe_math")) {
            if (input.moe_math != .legacy) return error.InvalidInput;
        }
    }
    if (input.operator_suite != null or input.slice_major_screen != null or input.expert_fixture != null or input.forced.len != 0) return error.InvalidInput;
    inline for (.{ "prefill_operator_screen", "prefill_ab", "combined_ab", "expert_n8_screen", "expert_shard_screen", "expert_latency_screen", "gdn_layout_screen", "gdn_layout_ab", "expert_n8_ab", "quant_reuse_screen", "transport_screen", "verify_default_policy", "router_turns", "router_screen", "router_candidate" }) |flag| {
        if (@field(input, flag)) return error.InvalidInput;
    }
}

test "boundary screen identities are absolute lowercase SHA256 pins" {
    const sha = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    const path = if (@import("builtin").os.tag == .windows) "C:/fixture" else "/fixture";
    try pin(path, sha);
    try std.testing.expectError(error.InvalidInput, pin("relative", sha));
    try std.testing.expectError(error.InvalidInput, pin(path, "short"));
    try std.testing.expectError(error.InvalidInput, pin(path, "G123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"));
}

test "each boundary screen excludes every other diagnostic" {
    const sha = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    const path = if (@import("builtin").os.tag == .windows) "C:/fixture" else "/fixture";
    const Input = struct {
        boundary_screen: ?Screen = null,
        operator_suite: ?u8 = null,
        slice_major_screen: ?u8 = null,
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
    for ([_]Screen{ .{ .integrated_moe = .{ .fixture = path, .fixture_sha256 = sha, .image = path, .image_sha256 = sha } }, .{ .terminal_ownership = .{ .fixture = path, .fixture_sha256 = sha } }, .{ .full_moe = .{ .fixture = path, .fixture_sha256 = sha, .image = path, .image_sha256 = sha } }, .{ .terminal_greedy = .{ .fixture = path, .fixture_sha256 = sha } }, .{ .terminal_graph = .{ .fixture = path, .fixture_sha256 = sha } }, .{ .slice_format = .{ .fixture = path, .fixture_sha256 = sha } }, .{ .shared_activation = .{ .fixture = path, .fixture_sha256 = sha, .image = path, .image_sha256 = sha } }, .{ .moe_semantics = .{ .fixture = path, .fixture_sha256 = sha, .image = path, .image_sha256 = sha } }, .{ .shared_branch = .{ .fixture = path, .fixture_sha256 = sha, .image = path, .image_sha256 = sha } } }) |screen| {
        try validateInput(Input{ .boundary_screen = screen });
        inline for (.{ "prefill_operator_screen", "prefill_ab", "combined_ab", "expert_n8_screen", "expert_shard_screen", "expert_latency_screen", "gdn_layout_screen", "gdn_layout_ab", "expert_n8_ab", "quant_reuse_screen", "transport_screen", "verify_default_policy", "router_turns", "router_screen", "router_candidate" }) |flag| {
            var input: Input = .{ .boundary_screen = screen };
            @field(input, flag) = true;
            try std.testing.expectError(error.InvalidInput, validateInput(input));
        }
        try std.testing.expectError(error.InvalidInput, validateInput(Input{ .boundary_screen = screen, .operator_suite = 1 }));
        try std.testing.expectError(error.InvalidInput, validateInput(Input{ .boundary_screen = screen, .slice_major_screen = 1 }));
        try std.testing.expectError(error.InvalidInput, validateInput(Input{ .boundary_screen = screen, .expert_fixture = "fixture" }));
        try std.testing.expectError(error.InvalidInput, validateInput(Input{ .boundary_screen = screen, .forced = &.{1} }));
    }
}

test "integrated tag reflects every Input mode and keeps legacy tags unchanged" {
    const sha = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    const path = if (@import("builtin").os.tag == .windows) "C:/fixture" else "/fixture";
    const Input = struct {
        boundary_screen: ?Screen = null,
        operator_suite: ?u8 = null,
        slice_major_screen: ?u8 = null,
        expert_fixture: ?[]const u8 = null,
        expert_fixture_sha256: ?[]const u8 = null,
        forced: []const u32 = &.{},
        moe_math: enum { legacy, bf16_boundaries } = .legacy,
        layout_ab: bool = false,
        terminal_ab: bool = false,
        future_mode: bool = false,
        future_fixture: ?u8 = null,
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
    const screen: Screen = .{ .integrated_moe = .{ .fixture = path, .fixture_sha256 = sha, .image = path, .image_sha256 = sha } };
    const encoded = try std.json.Stringify.valueAlloc(std.testing.allocator, screen, .{});
    defer std.testing.allocator.free(encoded);
    const parsed = try std.json.parseFromSlice(Screen, std.testing.allocator, encoded, .{});
    defer parsed.deinit();
    try validateInput(Input{ .boundary_screen = parsed.value });
    try validateInput(Input{ .boundary_screen = screen });
    inline for (@typeInfo(Input).@"struct".field_names) |name| {
        const T = @FieldType(Input, name);
        if (T == bool) {
            var input: Input = .{ .boundary_screen = screen };
            @field(input, name) = true;
            try std.testing.expectError(error.InvalidInput, validateInput(input));
        }
    }
    try std.testing.expectError(error.InvalidInput, validateInput(Input{ .boundary_screen = screen, .future_fixture = 1 }));
    try std.testing.expectError(error.InvalidInput, validateInput(Input{ .boundary_screen = screen, .expert_fixture_sha256 = sha }));
    try std.testing.expectError(error.InvalidInput, validateInput(Input{ .boundary_screen = screen, .moe_math = .bf16_boundaries }));
    try validateInput(Input{ .boundary_screen = .{ .full_moe = .{ .fixture = path, .fixture_sha256 = sha, .image = path, .image_sha256 = sha } }, .future_mode = true });
}

test "integrated JSON requires exactly its pinned payload" {
    const a = std.testing.allocator;
    try std.testing.expectError(error.UnknownField, std.json.parseFromSlice(Screen, a, "{\"unknown\":{}}", .{}));
    try std.testing.expectError(error.MissingField, std.json.parseFromSlice(Screen, a, "{\"integrated_moe\":{}}", .{}));
    try std.testing.expectError(error.UnexpectedToken, std.json.parseFromSlice(Screen, a, "{\"integrated_moe\":true}", .{}));
    try std.testing.expectError(error.UnknownField, std.json.parseFromSlice(Screen, a, "{\"integrated_moe\":{\"fixture\":\"x\",\"fixture_sha256\":\"x\",\"image\":\"x\",\"image_sha256\":\"x\",\"unknown\":true}}", .{}));
}
