//! The FP8 attention load opt-in is parsed once and agreed by every rank without changing cache numerics.
const std = @import("std");
const ranks = @import("cuda_ranks.zig");

fn env(comptime value: ?[]const u8) *const fn ([:0]const u8) ?[]const u8 {
    return &struct {
        fn get(name: [:0]const u8) ?[]const u8 {
            return if (std.mem.eql(u8, name, "TF_GLM_ATTN_FP8LOAD")) value else null;
        }
    }.get;
}

test "FP8 attention load opt-in: unset and zero keep old kernel, only one enables" {
    var problem: []const u8 = "";
    try std.testing.expect(!(try ranks.Settings.read(env(null), &problem)).attn_fp8load);
    try std.testing.expect(!(try ranks.Settings.read(env("0"), &problem)).attn_fp8load);
    try std.testing.expect((try ranks.Settings.read(env("1"), &problem)).attn_fp8load);
    inline for (.{ "", "true", "2", " 1" }) |bad| {
        try std.testing.expectError(error.BadSetting, ranks.Settings.read(env(bad), &problem));
        try std.testing.expectEqualStrings("TF_GLM_ATTN_FP8LOAD: 0 or 1", problem);
    }
}

test "FP8 attention load opt-in: ranks must agree but cache numerical identity stays identical" {
    const hash: [32]u8 = @splat(0);
    const off: ranks.Settings = .{};
    const on: ranks.Settings = .{ .attn_fp8load = true };
    var all = [_]ranks.Fingerprint{
        ranks.Fingerprint.of(2, 0, off, 8192, hash, hash, "checkpoint"),
        ranks.Fingerprint.of(2, 1, on, 8192, hash, hash, "checkpoint"),
    };
    try std.testing.expectEqual(ranks.Mismatch{ .field = .attn_fp8load, .rank = 1 }, ranks.mismatch(&all).?);
    all[0] = ranks.Fingerprint.of(2, 0, on, 8192, hash, hash, "checkpoint");
    try std.testing.expect(ranks.mismatch(&all) == null);
    var a = std.crypto.hash.sha2.Sha256.init(.{});
    var b = std.crypto.hash.sha2.Sha256.init(.{});
    off.hashNumerics(&a);
    on.hashNumerics(&b);
    try std.testing.expectEqualSlices(u8, &a.finalResult(), &b.finalResult());
}
