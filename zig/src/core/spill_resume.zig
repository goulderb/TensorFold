//! Stale-session resume from the spill tier, for any family: the longest stored prefix every rank holds, restored on
//! every rank or on none. A family plugs in through `Family`; with no family adapter and no tier nothing persists.
//! The tier (spill.zig) already checks identity, rank, header and tokens; this file adds the cross-rank rule and the
//! refusal counts the tier cannot see. Snapshot formats stay the family's own: the payload bytes are opaque here.
const std = @import("std");
const spill = @import("spill.zig");

/// A family's restore seam. `restore` reads rank `rank`'s payload from `reading` into that rank's live state; `discard`
/// undoes a restore that already ran (another rank failed). `eligible` may refuse a position (null: every position).
pub const Family = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        eligible: ?*const fn (ptr: *anyopaque, at: u32) bool = null,
        restore: *const fn (ptr: *anyopaque, rank: u32, reading: *spill.Reading) anyerror!void,
        discard: *const fn (ptr: *anyopaque, rank: u32) void,
    };
};

/// Why a resume went cold, beyond the tier's own stats (disk_misses, load_failed, hash_failed).
pub const Refusals = struct {
    no_prefix: u64 = 0, // no stored state the prompt extends on the first rank
    missing_rank: u64 = 0, // a longer state skipped because a peer rank does not hold it
    ineligible: u64 = 0, // a longer state skipped because the family refused its position
    read_failed: u64 = 0, // a rank's header or tokens failed: every rank's copy forgotten, cold
    restore_failed: u64 = 0, // a rank's payload or family restore failed: restored ranks discarded, cold
    resumed: u64 = 0,
    resumed_tokens: u64 = 0,
};

pub const Outcome = union(enum) {
    cold,
    resumed: u32, // prompt tokens now held on every rank
};

const Probe = struct {
    tiers: []const *spill.Tier,
    prompt: []const u32,
    family: Family,
    refusals: *Refusals,

    fn usable(p: *const Probe, at: u32) bool {
        if (p.family.vtable.eligible) |ok| if (!ok(p.family.ptr, at)) {
            p.refusals.ineligible += 1;
            return false;
        };
        const name = spill.Name.of(p.prompt[0..at], at);
        // tiers[0]'s lock is held while best() calls this; the peers' locks are their own
        for (p.tiers[1..]) |t| if (!t.has(name)) {
            p.refusals.missing_rank += 1;
            return false;
        };
        return true;
    }
};

/// Resume `prompt` past `beyond` tokens from the longest state every tier (one per rank, rank order) holds and the
/// family accepts. All ranks restore or none stays restored; a failed state is forgotten on every rank.
pub fn resumeLongest(tiers: []const *spill.Tier, prompt: []const u32, beyond: u32, family: Family, refusals: *Refusals) Outcome {
    std.debug.assert(tiers.len > 0);
    const probe: Probe = .{ .tiers = tiers, .prompt = prompt, .family = family, .refusals = refusals };
    const hit = tiers[0].best(prompt, beyond, &probe, Probe.usable) orelse {
        refusals.no_prefix += 1;
        return .cold;
    };
    const name = hit.name;
    for (tiers, 0..) |t, i| {
        const rank: u32 = @intCast(i);
        var reading = t.read(name) catch {
            undo(tiers, family, name, rank);
            refusals.read_failed += 1;
            return .cold;
        };
        family.vtable.restore(family.ptr, rank, &reading) catch {
            reading.close();
            undo(tiers, family, name, rank + 1);
            refusals.restore_failed += 1;
            return .cold;
        };
        reading.close();
    }
    for (tiers) |t| t.touch(name);
    refusals.resumed += 1;
    refusals.resumed_tokens += name.at;
    return .{ .resumed = name.at };
}

/// Ranks below `restored` discarded; `name` forgotten on every rank so no later lookup pairs a half.
fn undo(tiers: []const *spill.Tier, family: Family, name: spill.Name, restored: u32) void {
    var r: u32 = 0;
    while (r < restored) : (r += 1) family.vtable.discard(family.ptr, r);
    for (tiers) |t| t.forget(name);
}

test {
    _ = @import("spill_resume_test.zig");
}
