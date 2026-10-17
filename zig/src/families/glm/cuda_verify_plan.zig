const std = @import("std");

pub const max_drafts = 6;
pub const max_streams = 8;
pub const max_rows = max_streams * (1 + max_drafts);

pub const Plan = enum(u8) {
    baseline,
    dflash6,

    pub fn parse(raw: []const u8) !Plan {
        if (std.mem.eql(u8, raw, "3")) return .baseline;
        if (std.mem.eql(u8, raw, "6")) return .dflash6;
        return error.BadVerifyPlan;
    }

    pub fn active(p: Plan, dflash: bool) Plan {
        return if (dflash) p else .baseline;
    }

    pub fn drafts(p: Plan) usize {
        return switch (p) {
            .baseline => 3,
            .dflash6 => 6,
        };
    }

    pub fn rows(p: Plan) usize {
        return max_streams * (1 + p.drafts());
    }
};

test "verify plans retain baseline and keep eight wider lanes below the HC switch" {
    try std.testing.expectEqual(Plan.baseline, try Plan.parse("3"));
    try std.testing.expectEqual(Plan.dflash6, try Plan.parse("6"));
    for ([_][]const u8{ "", "0", "4", "7", "06", " 6" }) |raw| try std.testing.expectError(error.BadVerifyPlan, Plan.parse(raw));
    try std.testing.expectEqual(@as(usize, 3), Plan.baseline.drafts());
    try std.testing.expectEqual(@as(usize, 32), Plan.baseline.rows());
    try std.testing.expectEqual(@as(usize, 6), Plan.dflash6.drafts());
    try std.testing.expectEqual(@as(usize, 56), Plan.dflash6.rows());
    try std.testing.expectEqual(Plan.baseline, Plan.dflash6.active(false));
    try std.testing.expectEqual(Plan.dflash6, Plan.dflash6.active(true));
    try std.testing.expect(max_rows < 64);
}
