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
    lookahead: u32,
    family: Family,
    refusals: *Refusals,

    fn usable(p: *const Probe, at: u32) bool {
        if (p.family.vtable.eligible) |ok| if (!ok(p.family.ptr, at)) {
            p.refusals.ineligible += 1;
            return false;
        };
        // best() compared the stored tokens (at + lookahead of them) with the prompt before asking
        const tokens = p.prompt[0 .. at + p.lookahead];
        const name = spill.Name.of(tokens, at);
        // tiers[0]'s lock is held while best() calls this; the peers' locks are their own. A peer copy counts only when
        // it is on disk (not on its way) with exactly these tokens, not merely the same hash.
        for (p.tiers[1..]) |t| if (!t.stored(name, tokens)) {
            p.refusals.missing_rank += 1;
            return false;
        };
        return true;
    }
};

/// Resume `prompt` past `beyond` tokens from the longest state every tier (one per rank, rank order) holds and the
/// family accepts; `lookahead` is the tokens a state read past its position (prompt_cache Rules.lookahead). All ranks
/// restore or none stays restored; a failed state is forgotten on every rank.
pub fn resumeLongest(tiers: []const *spill.Tier, prompt: []const u32, beyond: u32, lookahead: u32, family: Family, refusals: *Refusals) Outcome {
    std.debug.assert(tiers.len > 0);
    const probe: Probe = .{ .tiers = tiers, .prompt = prompt, .lookahead = lookahead, .family = family, .refusals = refusals };
    const hit = tiers[0].best(prompt, beyond, &probe, Probe.usable) orelse {
        refusals.no_prefix += 1;
        return .cold;
    };
    const name = hit.name;
    for (tiers, 0..) |t, i| {
        const rank: u32 = @intCast(i);
        var reading = t.read(name) catch |err| {
            // a copy that went pending or away since the lookup is another writer's: left alone, not forgotten
            undo(tiers, family, name, rank, err != error.NotStored);
            refusals.read_failed += 1;
            return .cold;
        };
        family.vtable.restore(family.ptr, rank, &reading) catch {
            reading.close();
            undo(tiers, family, name, rank + 1, true);
            refusals.restore_failed += 1;
            return .cold;
        };
        reading.close();
    }
    for (tiers) |t| t.touch(name);
    // the only avoided-prefill count: every rank committed (the tiers' loaded_tokens count I/O, failed pairs too)
    refusals.resumed += 1;
    refusals.resumed_tokens += name.at;
    return .{ .resumed = name.at };
}

/// Ranks below `restored` discarded (a failing rank's partial restore too); with `bad`, `name` forgotten on every rank
/// so no later lookup pairs a half.
fn undo(tiers: []const *spill.Tier, family: Family, name: spill.Name, restored: u32, bad: bool) void {
    var r: u32 = 0;
    while (r < restored) : (r += 1) family.vtable.discard(family.ptr, r);
    if (bad) for (tiers) |t| t.forget(name);
}

/// What a spill identity binds, for any family: states restore only into the same weights, runtime, layout, dtype and
/// tensor-parallel shape. `weights` is the family's digest of its checkpoint (spill_identity.zig caches it); the text
/// fields are the family's own canonical names. `digest` is Tier.open's identity.
pub const Identity = struct {
    weights: [32]u8,
    runtime: []const u8, // engine build and kernel set, e.g. a version and kernel digest
    layout: []const u8, // state layout: cache kinds, block sizes, lookahead
    dtype: []const u8, // kept-state element types, e.g. "kv=fp8,state=bf16"
    tp: u32, // ranks the state is sharded over (Tier world)

    /// Domain-tagged SHA-256 over every field, each length-prefixed; empty, oversized or NUL-bearing fields refused.
    pub fn digest(i: Identity) error{BadSpillIdentity}![32]u8 {
        if (i.tp == 0 or std.mem.allEqual(u8, &i.weights, 0)) return error.BadSpillIdentity;
        var h = std.crypto.hash.sha2.Sha256.init(.{});
        h.update("tensorfold-spill-identity-v1");
        h.update(&i.weights);
        for ([_][]const u8{ i.runtime, i.layout, i.dtype }) |f| {
            if (f.len == 0 or f.len > 256 or std.mem.indexOfScalar(u8, f, 0) != null) return error.BadSpillIdentity;
            var n: [4]u8 = undefined;
            std.mem.writeInt(u32, &n, @intCast(f.len), .little);
            h.update(&n);
            h.update(f);
        }
        var tp: [4]u8 = undefined;
        std.mem.writeInt(u32, &tp, i.tp, .little);
        h.update(&tp);
        return h.finalResult();
    }
};

test {
    _ = @import("spill_resume_test.zig");
}
