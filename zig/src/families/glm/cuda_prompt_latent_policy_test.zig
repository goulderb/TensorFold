const std = @import("std");
const ranks = @import("cuda_ranks.zig");

fn env(comptime value: ?[]const u8) *const fn ([:0]const u8) ?[]const u8 {
    return &struct {
        fn get(name: [:0]const u8) ?[]const u8 {
            return if (std.mem.eql(u8, name, "TF_GLM_PROMPT_LATENT_AOT")) value else null;
        }
    }.get;
}

test "prompt latent policy is explicit and rejects ambiguous values" {
    var problem: []const u8 = "";
    try std.testing.expect(!(try ranks.Settings.read(env(null), &problem)).prompt_latent_aot);
    try std.testing.expect(!(try ranks.Settings.read(env("0"), &problem)).prompt_latent_aot);
    try std.testing.expect((try ranks.Settings.read(env("1"), &problem)).prompt_latent_aot);
    inline for (.{ "", "true", "2", " 1" }) |bad| {
        try std.testing.expectError(error.BadSetting, ranks.Settings.read(env(bad), &problem));
        try std.testing.expectEqualStrings("TF_GLM_PROMPT_LATENT_AOT: 0 or 1", problem);
    }
}

test "prompt arithmetic separates rank agreement and persisted numerical policy" {
    const zero: [32]u8 = @splat(0);
    const off: ranks.Settings = .{};
    const on: ranks.Settings = .{ .prompt_latent_aot = true };
    var all = [_]ranks.Fingerprint{
        ranks.Fingerprint.of(2, 0, off, 8192, true, zero, zero, ranks.hashBytes("checkpoint")),
        ranks.Fingerprint.of(2, 1, on, 8192, true, zero, zero, ranks.hashBytes("checkpoint")),
    };
    try std.testing.expectEqual(ranks.Mismatch{ .field = .prompt_latent_aot, .rank = 1 }, ranks.mismatch(&all).?);
    all[0] = ranks.Fingerprint.of(2, 0, on, 8192, true, zero, zero, ranks.hashBytes("checkpoint"));
    try std.testing.expect(ranks.mismatch(&all) == null);
    var a = std.crypto.hash.sha2.Sha256.init(.{});
    var b = std.crypto.hash.sha2.Sha256.init(.{});
    off.hashNumerics(&a);
    on.hashNumerics(&b);
    try std.testing.expect(!std.mem.eql(u8, &a.finalResult(), &b.finalResult()));
}
