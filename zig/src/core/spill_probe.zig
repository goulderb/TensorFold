//! `spill-probe FAMILY IDENTITY_HEX RANK WORLD`: the model-neutral spill startup checks on the CPU. It reads the
//! TENSORFOLD_SPILL_* settings as a server does, opens the tier (private directory, no links, partial and foreign files
//! pruned) and prints one marker line per stage. It loads no model and touches no device. A family's own startup
//! derives the identity from its weights, runtime, layout, dtype and TP (spill_identity.zig); here it is given.
//! Exit 0: off or opened; 2: usage; 3: a check failed (SPILL_PROBE_FAIL error=NAME).
const std = @import("std");
const spill = @import("spill.zig");

fn say(comptime fmt: []const u8, args: anytype) void {
    var buf: [512]u8 = undefined;
    const line = std.fmt.bufPrint(&buf, fmt ++ "\n", args) catch return;
    _ = std.c.write(1, line.ptr, line.len);
}

pub fn main(init: std.process.Init) !u8 {
    const a = init.arena.allocator();
    const argv = try init.minimal.args.toSlice(a);
    if (argv.len != 5) {
        say("usage: spill-probe FAMILY IDENTITY_HEX(64) RANK WORLD", .{});
        return 2;
    }
    var identity: [32]u8 = undefined;
    _ = std.fmt.hexToBytes(&identity, argv[2]) catch {
        say("SPILL_PROBE_FAIL stage=usage error=BadIdentity", .{});
        return 2;
    };
    if (argv[2].len != 64) {
        say("SPILL_PROBE_FAIL stage=usage error=BadIdentity", .{});
        return 2;
    }
    const rank = std.fmt.parseInt(u32, argv[3], 10) catch return 2;
    const world = std.fmt.parseInt(u32, argv[4], 10) catch return 2;
    if (world == 0 or rank >= world) return 2;
    const settings = spill.Settings.fromEnv(a, spill.ProcessEnv{}) catch |e| {
        say("SPILL_PROBE_FAIL stage=settings error={s}", .{@errorName(e)});
        return 3;
    } orelse {
        say("SPILL_OFF", .{});
        return 0;
    };
    var text: [128]u8 = undefined;
    say("SPILL_SETTINGS {s}", .{settings.text(&text) catch "?"});
    const t = spill.Tier.open(init.gpa, init.io, settings, argv[1], identity, rank, world) catch |e| {
        say("SPILL_PROBE_FAIL stage=open error={s}", .{@errorName(e)});
        return 3;
    };
    defer t.deinit();
    const info = t.info();
    say("SPILL_OPENED entries={d} stored_bytes={d} cap_bytes={d} pruned={d}", .{ info.entries, info.stored_bytes, info.cap, info.stats.pruned });
    return 0;
}
