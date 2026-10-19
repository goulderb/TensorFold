const std = @import("std");
const lanes = @import("lanes");
const p = @import("cuda_draft_profile.zig");
const verify = @import("cuda_verify_plan.zig");
const state = @import("cuda_dflash_state.zig");
const t = std.testing;
const Entry = struct { name: []const u8, value: []const u8 };
const Env = struct {
    var values: []const Entry = &.{};
    fn get(name: [:0]const u8) ?[]const u8 {
        for (values) |v| if (std.mem.eql(u8, v.name, name)) return v.value;
        return null;
    }
};

test "production profile is explicit and keeps target, neural, shared and allocation ceilings distinct" {
    var problem: []const u8 = "";
    Env.values = &.{.{ .name = "TF_GLM_COPY_MAX", .value = "not-a-number" }};
    try t.expect(!(try p.Settings.read(Env.get, &problem)).enabled);
    Env.values = &.{.{ .name = "TF_GLM_DRAFT_PROFILE", .value = "production" }};
    const s = try p.Settings.read(Env.get, &problem);
    try t.expect(s.enabled and s.copy.enabled);
    try t.expectEqual(@as(usize, 15), s.copy.most);
    try t.expectEqual(@as(usize, 8), s.copy.match);
    try t.expectEqual(@as(usize, 16), s.copy.reply_match);
    try t.expectEqual(@as(usize, 16), p.target_rows);
    try t.expectEqual(@as(usize, 7), p.neural_drafts);
    try t.expectEqual(@as(usize, 32), p.shared_rows);
    try t.expectEqual(@as(usize, 64), verify.max_rows);
    const enabled_hash = s.digest();
    const disabled_hash = (p.Settings{}).digest();
    try t.expect(!std.mem.eql(u8, &enabled_hash, &disabled_hash));
}

test "production profile refuses ignored or contradictory draft controls" {
    var problem: []const u8 = "";
    for ([_]Entry{
        .{ .name = "TF_GLM_MAX_ROWS", .value = "8" },
        .{ .name = "TF_GLM_MULTI_WINDOW", .value = "64" },
        .{ .name = "TF_GLM_DFLASH_POLICY", .value = "fnc7:0.7" },
        .{ .name = "TF_GLM_DFLASH_VERIFY_DRAFTS", .value = "6" },
        .{ .name = "TF_GLM_MULTI_DEPTH", .value = "joint" },
        .{ .name = "TF_GLM_MULTI_LONE", .value = "1" },
        .{ .name = "TF_GLM_SHARED", .value = "0" },
        .{ .name = "TF_GLM_WIDE_GRAPHS", .value = "32" },
        .{ .name = "TF_GLM_LONE_COSTS", .value = "38,7.9" },
        .{ .name = "TF_GLM_COPY_MAX", .value = "16" },
        .{ .name = "TF_GLM_COPY_MATCH", .value = "1" },
        .{ .name = "TF_GLM_COPY_REPLY_MATCH", .value = "8" },
        .{ .name = "TF_GLM_COPY_MISS_MAX", .value = "16" },
        .{ .name = "TF_GLM_COPY_DRAFTS", .value = "yes" },
    }) |bad| {
        Env.values = &.{ .{ .name = "TF_GLM_DRAFT_PROFILE", .value = "production" }, bad };
        try t.expectError(error.BadSetting, p.Settings.read(Env.get, &problem));
        try t.expect(problem.len != 0);
    }
    Env.values = &.{.{ .name = "TF_GLM_DRAFT_PROFILE", .value = "unknown" }};
    try t.expectError(error.BadSetting, p.Settings.read(Env.get, &problem));
}

test "copy settings can disable copies without restoring generic suffix lookup" {
    var problem: []const u8 = "";
    Env.values = &.{ .{ .name = "TF_GLM_DRAFT_PROFILE", .value = "production" }, .{ .name = "TF_GLM_COPY_DRAFTS", .value = "0" } };
    var settings = try p.Settings.read(Env.get, &problem);
    const factory = settings.proposers().?;
    const proposer = try factory.make(factory.ctx, t.allocator, &.{ 1, 2, 1, 2 }, &.{99});
    defer factory.free(factory.ctx, proposer);
    try t.expect(proposer.vtable.priced);
    try t.expectEqual(@as(usize, 0), (try proposer.propose(&.{ 1, 2, 1, 2 }, 15)).len);
}

test "production factory request lifetime is allocator owned and never enables standalone padding" {
    var settings = p.Settings.production();
    settings.copy.match = 2;
    settings.copy.reply_match = 0;
    const factory = settings.proposers().?;
    const proposer = try factory.make(factory.ctx, t.allocator, &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2 }, &.{});
    defer factory.free(factory.ctx, proposer);
    const got = try proposer.propose(&.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2 }, 15);
    try t.expectEqualSlices(u32, &.{ 3, 4, 5, 6, 7, 8, 9, 1, 2 }, got);
    try t.expectEqual(@as(usize, 9), got.len);
}

test "shared production trim keeps three drafts on eight equal seven-draft lanes including copies" {
    var s: lanes.Stream = undefined;
    var windows: [8]lanes.trim.Window = undefined;
    for (&windows, 0..) |*w, i| w.* = .{ .stream = &s, .drafted = i % 2 == 0, .copied = i % 2 != 0, .rows = 8, .parents = null };
    var out: [8]lanes.trim.Choice = undefined;
    try p.cap(&windows, 32, &out);
    for (out) |o| try t.expectEqual(@as(u32, 3), o.count);
    try t.expectError(error.SharedPendingRows, p.cap(&windows, 7, &out));
}

test "shared production trim removes the latest longest tail first and preserves forced rows" {
    var s: lanes.Stream = undefined;
    const windows = [_]lanes.trim.Window{
        .{ .stream = &s, .drafted = true, .rows = 5, .parents = null },
        .{ .stream = &s, .drafted = false, .copied = true, .rows = 5, .parents = null },
        .{ .stream = &s, .drafted = false, .rows = 3, .parents = null },
    };
    var out: [3]lanes.trim.Choice = undefined;
    try p.cap(&windows, 12, &out);
    try t.expectEqual(@as(u32, 4), out[0].count);
    try t.expectEqual(@as(u32, 3), out[1].count);
    try t.expectEqual(@as(u32, 2), out[2].count);
    try t.expectError(error.SharedPendingRows, p.cap(&windows, 4, &out));
}

test "profile facts use sixteen target rows and fixed seven neural drafts, not legacy adaptive limits" {
    const engine = @import("cuda_engine.zig");
    const backend = @import("cuda_lanes.zig");
    var e: engine.Engine = undefined;
    e.settings = .{};
    e.mtp = true;
    e.dflash = null;
    var b: backend.Lanes = .{ .x = .{ .e = &e, .link = undefined } };
    const legacy = b.facts();
    try t.expectEqual(@as(u32, 4), legacy.exact_width);
    try t.expectEqual(@as(u32, 32), legacy.batch_rows);
    try t.expect(!legacy.fixed_drafts);
    e.settings.draft_profile = p.Settings.production();
    e.settings.dflash_verify = .dflash7;
    e.dflash = @ptrFromInt(16);
    e.mtp = false;
    const production = b.facts();
    try t.expectEqual(@as(u32, 16), production.exact_width);
    try t.expectEqual(@as(u32, 7), production.drafts);
    try t.expect(production.fixed_drafts and production.draft_streams);
    try t.expectEqual(@as(u32, 32), production.batch_rows);
    var cfg = try lanes.Config.init(t.allocator, production, 16, 15);
    defer cfg.deinit(t.allocator);
    try t.expectEqual(@as(u32, 15), cfg.max_copy);
    try t.expectEqual(@as(u32, 7), cfg.most_drafts);
    try t.expectEqual(@as(usize, 64), e.verifyPlan().rows());
    try t.expectEqual(@as(usize, 32), e.sharedLimit());
}

test "production profile is shared by TP fingerprints and requires the DFlash loader" {
    const ranks = @import("cuda_ranks.zig");
    var problem: []const u8 = "";
    Env.values = &.{ .{ .name = "TF_GLM_DRAFT_PROFILE", .value = "production" }, .{ .name = "TF_GLM_DFLASH_DIR", .value = "/fixture/dflash" } };
    const settings = try ranks.Settings.read(Env.get, &problem);
    try t.expectEqual(ranks.Drafter.dflash2, settings.draft);
    try t.expectEqual(verify.Plan.dflash7, settings.dflash_verify);
    var all = [_]ranks.Fingerprint{
        ranks.Fingerprint.of(2, 0, settings, 4096, @splat(0), @splat(0), "fixture"),
        ranks.Fingerprint.of(2, 1, settings, 4096, @splat(0), @splat(0), "fixture"),
    };
    try t.expect(ranks.mismatch(&all) == null);
    var different = settings;
    different.draft_profile.copy.enabled = false;
    all[1] = ranks.Fingerprint.of(2, 1, different, 4096, @splat(0), @splat(0), "fixture");
    try t.expectEqual(ranks.Fingerprint.Field.draft_profile, ranks.mismatch(&all).?.field);
    Env.values = &.{ .{ .name = "TF_GLM_DRAFT_PROFILE", .value = "production" }, .{ .name = "TF_GLM_DRAFTER", .value = "mtp" } };
    try t.expectError(error.BadSetting, ranks.Settings.read(Env.get, &problem));
}

test "sixteen-row copy tap windows commit only the accepted prefix at every rejection boundary" {
    for (1..17) |kept| {
        var l: state.Lane = .{ .end = 100 };
        try l.validateStage(100, 16, 256);
        l.pending = .{ .start = 100, .rows = 16 };
        try l.validateAccept(kept, 100 + kept);
        try t.expectError(error.BadDraftCommit, l.validateAccept(kept, 101 + kept));
        l.end += kept;
        l.pending = null;
        try t.expect(l.ready(100 + kept));
        try t.expectError(error.NoDraftWindow, l.validateAccept(kept, 100 + kept));
    }
}
