//! `spill-probe FAMILY IDENTITY_HEX RANK WORLD`: the model-neutral spill startup checks on the CPU, never destructive.
//! It reads the TENSORFOLD_SPILL_* settings as a server does and checks the configured root as the tier would, without
//! opening a tier there: a tier's start prunes partial, foreign and old-build files, so the probe opens its tier only
//! inside a new private directory it makes under the root (mkdtemp, 0700) and removes only that directory afterwards.
//! Existing builds and files under the root are never touched. It loads no model and touches no device. A family's
//! own startup derives the identity (spill_resume.Identity); here it is given.
//! Exit 0: off or every check passed; 2: usage; 3: a check failed (SPILL_PROBE_FAIL stage=NAME error=NAME).
const std = @import("std");
const spill = @import("spill.zig");

extern "c" fn mkdtemp(template: [*:0]u8) ?[*:0]u8;

fn say(comptime fmt: []const u8, args: anytype) void {
    var buf: [512]u8 = undefined;
    const line = std.fmt.bufPrint(&buf, fmt ++ "\n", args) catch return;
    _ = std.c.write(1, line.ptr, line.len);
}

pub fn main(init: std.process.Init) !u8 {
    const a = init.arena.allocator();
    const argv = try init.minimal.args.toSlice(a);
    if (argv.len != 5 or argv[2].len != 64) {
        say("usage: spill-probe FAMILY IDENTITY_HEX(64) RANK WORLD", .{});
        return 2;
    }
    var identity: [32]u8 = undefined;
    _ = std.fmt.hexToBytes(&identity, argv[2]) catch {
        say("SPILL_PROBE_FAIL stage=usage error=BadIdentity", .{});
        return 2;
    };
    const rank = std.fmt.parseInt(u32, argv[3], 10) catch return 2;
    const world = std.fmt.parseInt(u32, argv[4], 10) catch return 2;
    if (world == 0 or rank >= world) return 2;
    var settings = spill.Settings.fromEnv(a, spill.ProcessEnv{}) catch |e| {
        say("SPILL_PROBE_FAIL stage=settings error={s}", .{@errorName(e)});
        return 3;
    } orelse {
        say("SPILL_OFF", .{});
        return 0;
    };
    var text: [128]u8 = undefined;
    say("SPILL_SETTINGS {s}", .{settings.text(&text) catch "?"});
    if (!settings.disk()) {
        say("SPILL_RAM_ONLY", .{}); // no directory to probe
        return 0;
    }
    const root = try a.dupeSentinel(u8, settings.dir, 0);
    spill.checkRoot(root) catch |e| {
        say("SPILL_PROBE_FAIL stage=root error={s}", .{@errorName(e)});
        return 3;
    };
    say("SPILL_ROOT_PRIVATE", .{});
    const template = try std.fmt.allocPrintSentinel(a, "{s}/.spill-probe-XXXXXX", .{root}, 0);
    const scratch = mkdtemp(template.ptr) orelse {
        say("SPILL_PROBE_FAIL stage=scratch error=MkdtempFailed", .{});
        return 3;
    };
    const scratch_path = std.mem.span(scratch);
    defer std.Io.Dir.cwd().deleteTree(init.io, scratch_path) catch say("SPILL_PROBE_LEFT {s}", .{scratch_path});
    settings.dir = scratch_path;
    const t = spill.Tier.open(init.gpa, init.io, settings, argv[1], identity, rank, world) catch |e| {
        say("SPILL_PROBE_FAIL stage=open error={s}", .{@errorName(e)});
        return 3;
    };
    defer t.deinit();
    const info = t.info();
    say("SPILL_OPENED scratch entries={d} cap_bytes={d} disk_free_bytes={d}", .{ info.entries, info.cap, info.disk_free });
    return 0;
}
