//! Opt-in serving policy adapted from the pinned TensorFold GLM production multi decoder.
//! `multi.py` trims longest draft tails, latest lane first on ties, and never pads CopyDrafts.
//! Production image d7e14f8799ad21f7e93ba95ea78826426c98ab27d5fbfdf81bed876a2e09ce6e;
//! original TensorFold and donor notices apply. This config does not implement target CUDA graphs.
const std = @import("std");
const lanes = @import("lanes");
const copies = @import("cuda_copy_drafts.zig");

pub const target_rows = 16;
pub const shared_rows = 32;
pub const neural_drafts = 7;

pub const Settings = struct {
    enabled: bool = false,
    copy: copies.Settings = .{},

    pub fn production() Settings {
        return .{ .enabled = true, .copy = copies.Settings.production() };
    }

    pub fn read(get: *const fn ([:0]const u8) ?[]const u8, problem: *[]const u8) !Settings {
        const name = get("TF_GLM_DRAFT_PROFILE") orelse return .{};
        if (!std.mem.eql(u8, name, "production")) {
            problem.* = "TF_GLM_DRAFT_PROFILE: production (or unset for the legacy policy)";
            return error.BadSetting;
        }
        var s: Settings = .{ .enabled = true, .copy = copies.Settings.production() };
        if (get("TF_GLM_COPY_DRAFTS")) |v| {
            if (std.mem.eql(u8, v, "0")) s.copy.enabled = false else if (!std.mem.eql(u8, v, "1")) {
                problem.* = "TF_GLM_COPY_DRAFTS: 0 or 1";
                return error.BadSetting;
            }
        }
        s.copy.most = try integer(get, "TF_GLM_COPY_MAX", s.copy.most, problem);
        s.copy.match = try integer(get, "TF_GLM_COPY_MATCH", s.copy.match, problem);
        s.copy.reply_match = try integer(get, "TF_GLM_COPY_REPLY_MATCH", s.copy.reply_match, problem);
        s.copy.miss_most = try integer(get, "TF_GLM_COPY_MISS_MAX", s.copy.miss_most, problem);
        s.copy.validate() catch {
            problem.* = "TF_GLM_COPY_*: max 1..15, match 2..64, reply match 0 or greater than match (at most 64), miss max 0..max";
            return error.BadSetting;
        };
        try exact(get, "TF_GLM_MAX_ROWS", "16", problem);
        try exact(get, "TF_GLM_MULTI_WINDOW", "32", problem);
        try exact(get, "TF_GLM_DFLASH_POLICY", "fnc7:0.3", problem);
        try exact(get, "TF_GLM_DFLASH_VERIFY_DRAFTS", "7", problem);
        try exact(get, "TF_GLM_MULTI_DEPTH", "policy", problem);
        try exact(get, "TF_GLM_MULTI_LONE", "0", problem);
        try exact(get, "TF_GLM_SHARED", "1", problem);
        // The serial setting is accepted but does not pad the multi-lane serving path or create native graphs.
        try exact(get, "TF_GLM_WIDE_GRAPHS", "16", problem);
        if (get("TF_GLM_LONE_COSTS") != null) {
            problem.* = "TF_GLM_DRAFT_PROFILE=production does not use TF_GLM_LONE_COSTS";
            return error.BadSetting;
        }
        return s;
    }

    pub fn hash(s: Settings, h: *std.crypto.hash.sha2.Sha256) void {
        if (!s.enabled) return;
        h.update("glm-production-draft-profile-v1\x00");
        h.update(&.{@intFromBool(s.copy.enabled)});
        for ([_]usize{ s.copy.most, s.copy.match, s.copy.reply_match, s.copy.miss_most, target_rows, shared_rows, neural_drafts }) |v| {
            const n: u64 = @intCast(v);
            h.update(std.mem.asBytes(&n));
        }
    }

    pub fn digest(s: Settings) [32]u8 {
        if (!s.enabled) return @splat(0);
        var h = std.crypto.hash.sha2.Sha256.init(.{});
        s.hash(&h);
        return h.finalResult();
    }

    pub fn proposers(s: *Settings) ?lanes.proposer.Factory {
        return .{ .ctx = s, .make = make, .free = free };
    }

    pub fn rowPolicy(s: *Settings) lanes.Trim {
        return .{ .ptr = s, .vtable = &.{ .copy_windows = true, .choose = choose, .commit = commit } };
    }
};

fn integer(get: *const fn ([:0]const u8) ?[]const u8, comptime name: [:0]const u8, fallback: usize, problem: *[]const u8) !usize {
    const raw = get(name) orelse return fallback;
    return std.fmt.parseInt(usize, raw, 10) catch {
        problem.* = name ++ ": an unsigned integer";
        return error.BadSetting;
    };
}
fn exact(get: *const fn ([:0]const u8) ?[]const u8, comptime name: [:0]const u8, comptime expected: []const u8, problem: *[]const u8) !void {
    if (get(name)) |v| if (!std.mem.eql(u8, v, expected)) {
        problem.* = name ++ ": " ++ expected ++ " for the production draft profile";
        return error.BadSetting;
    };
}

const Request = struct { gpa: std.mem.Allocator, copy: copies.CopyDrafts };
fn make(ctx: *anyopaque, gpa: std.mem.Allocator, prompt: []const u32, _: []const u32) !lanes.proposer.Proposer {
    const s: *const Settings = @ptrCast(@alignCast(ctx));
    const r = try gpa.create(Request);
    errdefer gpa.destroy(r);
    r.* = .{ .gpa = gpa, .copy = try copies.CopyDrafts.init(s.copy, prompt.len) };
    return r.copy.proposer();
}
fn free(_: *anyopaque, p: lanes.proposer.Proposer) void {
    const copy: *copies.CopyDrafts = @ptrCast(@alignCast(p.ptr));
    const r: *Request = @fieldParentPtr("copy", copy);
    const gpa = r.gpa;
    r.copy.deinit();
    gpa.destroy(r);
}
fn choose(_: *anyopaque, windows: []const lanes.trim.Window, _: bool, _: std.mem.Allocator, out: []lanes.trim.Choice) !void {
    try cap(windows, shared_rows, out);
}
fn commit(_: *anyopaque, _: lanes.trim.Commit) !void {}

/// Production multi._trim: every lane keeps its pending row; only copy/head tails may shrink.
pub fn cap(windows: []const lanes.trim.Window, limit: usize, out: []lanes.trim.Choice) !void {
    if (windows.len != out.len) return error.BadTrimBatch;
    var total: usize = 0;
    for (windows, out) |w, *o| {
        if (w.rows == 0) return error.BadTrimWindow;
        if ((w.drafted or w.copied) and w.parents != null) return error.TreesNotBuilt;
        o.* = .{ .count = w.rows - 1 };
        total += w.rows;
    }
    while (total > limit) {
        var longest: u32 = 0;
        var at: ?usize = null;
        for (windows, out, 0..) |w, o, i| {
            if (!(w.drafted or w.copied) or o.count == 0) continue;
            if (o.count >= longest) {
                longest = o.count;
                at = i;
            }
        }
        out[at orelse return error.SharedPendingRows].count -= 1;
        total -= 1;
    }
}

test {
    _ = @import("cuda_draft_profile_test.zig");
}
