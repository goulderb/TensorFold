//! GLM CopyDrafts adapted from TensorFold production `families/glm5_next/cuda/copy_drafts.py`.
//! Donor SHA256 a8ab7c52f7ba5d7466f27aaef1dba6120930f9fee7e579a0fa32ea74fc591161,
//! production image d7e14f8799ad21f7e93ba95ea78826426c98ab27d5fbfdf81bed876a2e09ce6e.
//! This port borrows the lane's committed context and owns only its proposal; original donor notices remain applicable.
const std = @import("std");
const Proposer = @import("lanes").proposer.Proposer;

pub const max_copies = 15;
const graphed_rows = 8;

pub const Settings = struct {
    enabled: bool = false,
    match: usize = 8,
    most: usize = 5,
    reply_match: usize = 0,
    miss_most: usize = 0,

    pub fn production() Settings {
        return .{ .enabled = true, .most = 15, .reply_match = 16 };
    }

    pub fn validate(s: Settings) !void {
        if (s.match < 2 or s.match > 64) return error.BadCopyMatch;
        if (s.most < 1 or s.most > max_copies) return error.BadCopyMax;
        if (s.reply_match > 64 or (s.reply_match != 0 and s.reply_match <= s.match)) return error.BadCopyReplyMatch;
        if (s.miss_most > s.most) return error.BadCopyMissMax;
    }
};

pub const CopyDrafts = struct {
    settings: Settings,
    prompt_len: usize,
    out: [max_copies]u32 = undefined,
    len: usize = 0,
    last_match: i64 = 0,
    original_copied: usize = 0,
    proposed_at: usize = 0,
    missed: bool = false,
    pads: [max_copies + 2]bool = @splat(false),

    /// The production multi-lane serving path never pads copies, even with only one active lane.
    pub fn init(settings: Settings, prompt_len: usize) !CopyDrafts {
        try settings.validate();
        return .{ .settings = settings, .prompt_len = prompt_len };
    }

    /// Only the standalone decoder pads to its separately captured graph widths.
    pub fn initStandalone(settings: Settings, prompt_len: usize, widths: []const usize) !CopyDrafts {
        var s = try init(settings, prompt_len);
        for (widths) |width| if (width > graphed_rows and width <= max_copies + 1) {
            s.pads[width] = true;
        };
        return s;
    }

    pub fn deinit(s: *CopyDrafts) void {
        s.* = undefined;
    }

    pub fn proposer(s: *CopyDrafts) Proposer {
        return .{ .ptr = s, .vtable = &vtable };
    }

    pub const vtable: Proposer.VTable = .{
        .propose = proposeFn,
        .last_match = lastMatchFn,
        .observe = observeFn,
        .priced = true,
    };

    fn proposeFn(ptr: *anyopaque, context: []const u32, room: i64) anyerror![]const u32 {
        const s: *CopyDrafts = @ptrCast(@alignCast(ptr));
        return s.propose(context, room);
    }

    fn lastMatchFn(ptr: *anyopaque) i64 {
        const s: *CopyDrafts = @ptrCast(@alignCast(ptr));
        return s.last_match;
    }

    fn observeFn(ptr: *anyopaque, proposed: i64, accepted: i64) void {
        const s: *CopyDrafts = @ptrCast(@alignCast(ptr));
        s.observe(proposed, accepted);
    }

    /// `accepted` excludes the bonus token; compare against the original copy, not a trimmed or padded width.
    pub fn observe(s: *CopyDrafts, proposed: i64, accepted: i64) void {
        _ = proposed;
        if (s.original_copied != 0) s.missed = accepted < @as(i64, @intCast(s.original_copied));
    }

    /// Context includes the pending token. The returned storage stays valid until the next proposal.
    pub fn propose(s: *CopyDrafts, context: []const u32, room: i64) ![]const u32 {
        // A zero-draft trim has no observe callback. Actual committed growth also includes any grammar continuation.
        if (s.original_copied != 0 and context.len > s.proposed_at) {
            s.missed = context.len - s.proposed_at <= s.original_copied;
        }
        s.original_copied = 0;
        s.len = 0;
        s.last_match = 0;
        const n = s.settings.match;
        if (!s.settings.enabled or room <= 0 or context.len <= n) return s.out[0..0];
        const available: usize = @intCast(@min(room, max_copies));
        const most = if (s.missed and s.settings.miss_most != 0) @min(s.settings.most, s.settings.miss_most) else s.settings.most;
        const k = @min(most, available);
        const suffix = context.len - n;
        const extra = s.settings.reply_match -| n;
        var earliest: ?usize = null;
        var latest_full: ?usize = null;
        for (0..suffix) |at| {
            if (context[at + n - 1] != context[context.len - 1]) continue;
            if (!std.mem.eql(u32, context[at .. at + n - 1], context[suffix .. context.len - 1])) continue;
            if (extra != 0 and at >= s.prompt_len) {
                if (at < extra or suffix < extra) continue;
                if (!std.mem.eql(u32, context[at - extra .. at], context[suffix - extra .. suffix])) continue;
            }
            if (earliest == null) earliest = at;
            if (context.len - (at + n) >= k) latest_full = at;
        }
        const start = (latest_full orelse earliest orelse return s.out[0..0]) + n;
        s.len = @min(k, context.len - start);
        @memcpy(s.out[0..s.len], context[start .. start + s.len]);
        s.original_copied = s.len;
        s.proposed_at = context.len;
        s.last_match = @intCast(n);
        const rows = s.len + 1;
        if (rows > graphed_rows) {
            for (rows..max_copies + 2) |width| {
                if (!s.pads[width]) continue;
                if (width > rows and width - 1 <= available) {
                    @memset(s.out[s.len .. width - 1], s.out[s.len - 1]);
                    s.len = width - 1;
                }
                break;
            }
        }
        return s.out[0..s.len];
    }
};

test {
    _ = @import("cuda_copy_drafts_test.zig");
}
