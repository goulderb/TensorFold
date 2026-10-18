//! CPU-only gate 5: compare declared launches and recorded usage with the candidate's fatbin ELF metadata.
const std = @import("std");
const core = @import("core");
const contract = core.contract;
const registry = contract.registry;
const limits = contract.limits;

pub const Selection = struct {
    dir: []const u8,
    sm: ?u32 = null,
    only: ?[]const u8 = null,

    pub fn parse(args: []const [:0]const u8) !Selection {
        if (args.len == 0) return error.MissingArgument;
        var out = Selection{ .dir = args[0] };
        var i: usize = 1;
        while (i < args.len) : (i += 2) {
            if (i + 1 >= args.len) return error.MissingArgument;
            if (std.mem.eql(u8, args[i], "--sm")) {
                if (out.sm != null) return error.DuplicateArgument;
                out.sm = try std.fmt.parseInt(u32, args[i + 1], 10);
                if (limits.target(out.sm.?) == null) return error.UnknownTarget;
            } else if (std.mem.eql(u8, args[i], "--only")) {
                if (out.only != null) return error.DuplicateArgument;
                if (args[i + 1].len == 0) return error.EmptySelection;
                out.only = args[i + 1];
            } else return error.UnknownArgument;
        }
        return out;
    }

    fn applies(self: Selection, r: registry.CudaRecord, sm: u32) bool {
        if (!r.applies(sm)) return false;
        if (self.sm) |selected| if (selected != sm) return false;
        return if (self.only) |text| std.mem.indexOf(u8, r.launch.name, text) != null else true;
    }
};

const Tally = struct {
    launches: usize = 0,
    compiles: usize = 0,
    controls_red: usize = 0,
    controls_missed: usize = 0,
    problems: usize = 0,
};

/// A target filter limits the claim to that target; a name filter prints PARTIAL and never claims the full gate.
pub fn run(a: std.mem.Allocator, io: std.Io, args: []const [:0]const u8) !void {
    const selected = try Selection.parse(args);
    var t: Tally = .{};
    for (registry.cuda ++ registry.cuda_controls) |r| {
        var applicable = false;
        for (limits.sms) |sm| applicable = applicable or selected.applies(r, sm.sm);
        if (!applicable) continue;
        t.launches += 1;
        const path = try std.fmt.allocPrint(a, "{s}/{s}.fatbin", .{ selected.dir, r.launch.image });
        defer a.free(path);
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 28));
        defer a.free(bytes);
        const images = try contract.cubin.images(a, bytes);
        defer a.free(images);
        for (limits.sms, 0..) |sm, i| {
            if (!selected.applies(r, sm.sm)) continue;
            const img = for (images) |image| {
                if (image.sm == sm.sm) break image;
            } else {
                std.debug.print("FAIL limits {s}: {s} holds no declared sm_{d} image\n", .{ r.launch.name, path, sm.sm });
                t.problems += 1;
                continue;
            };
            const kernels = try contract.cubin.kernels(a, img.elf, sm.sm);
            defer a.free(kernels);
            const kernel = for (kernels) |k| {
                if (std.mem.eql(u8, k.name, r.launch.symbol)) break k;
            } else {
                std.debug.print("FAIL limits sm_{d} {s}: symbol absent from image\n", .{ sm.sm, r.launch.name });
                t.problems += 1;
                continue;
            };
            try compare(a, r, i, sm, kernel.usage, &t);
        }
    }
    if (t.launches == 0) return error.EmptySelection;
    std.debug.print("SUMMARY limits: {d} launches, {d} compiled records, {d} controls red, {d} controls missed, {d} problems; {s}", .{ t.launches, t.compiles, t.controls_red, t.controls_missed, t.problems, if (selected.only != null) "PARTIAL name filter" else "all selected targets" });
    if (selected.sm) |sm| std.debug.print(" sm_{d}", .{sm});
    std.debug.print("\n", .{});
    if (t.problems != 0) return error.TestFailed;
}

fn compare(a: std.mem.Allocator, r: registry.CudaRecord, index: usize, sm: limits.Sm, usage: limits.Usage, t: *Tally) !void {
    t.compiles += 1;
    const control = std.mem.startsWith(u8, r.launch.name, "control");
    var findings: std.ArrayList(u8) = .empty;
    defer findings.deinit(a);
    const problems = try limits.cuda(r.launch, usage, sm, &findings, a);
    const current = std.meta.eql(usage, r.usage[index]);
    var word: []const u8 = undefined;
    if (control) {
        if (problems == 0) {
            t.controls_missed += 1;
            t.problems += 1;
            word = "MISSED";
        } else {
            t.controls_red += 1;
            word = "RED";
        }
    } else {
        t.problems += problems + @intFromBool(!current);
        word = if (problems == 0 and current) "PASS" else "FAIL";
    }
    std.debug.print("{s} limits sm_{d} {s}: {d} registers, {d} static shared bytes, bound {d} (record {s})\n", .{ word, sm.sm, r.launch.name, usage.regs, usage.static, usage.bound, if (current) "current" else "stale" });
    if (findings.items.len != 0) std.debug.print("{s}", .{findings.items});
    std.debug.print("RECORD sm_{d} {s}: .{{ .regs = {d}, .static = {d}, .bound = {d}, .local = {d} }}\n", .{ sm.sm, r.launch.name, usage.regs, usage.static, usage.bound, usage.local });
}

test "offline target selection is explicit and malformed filters are refused" {
    const s = try Selection.parse(&.{ "fatbins", "--sm", "121", "--only", "GLM" });
    try std.testing.expectEqual(@as(?u32, 121), s.sm);
    try std.testing.expectEqualStrings("GLM", s.only.?);
    try std.testing.expectError(error.UnknownTarget, Selection.parse(&.{ "fatbins", "--sm", "75" }));
    try std.testing.expectError(error.MissingArgument, Selection.parse(&.{ "fatbins", "--sm" }));
    try std.testing.expectError(error.DuplicateArgument, Selection.parse(&.{ "fatbins", "--sm", "121", "--sm", "121" }));
    try std.testing.expectError(error.UnknownArgument, Selection.parse(&.{ "fatbins", "--other", "121" }));
    try std.testing.expectError(error.EmptySelection, Selection.parse(&.{ "fatbins", "--only", "" }));
}

test "a stale compile is a failure even when its launch fits, and every target must catch each control" {
    const a = std.testing.allocator;
    const good: limits.Usage = .{ .regs = 32, .static = 0, .bound = 128 };
    var r = registry.CudaRecord{ .launch = .{ .name = "sample", .symbol = "sample", .image = "sample", .threads = 128 }, .usage = @splat(good) };
    var t: Tally = .{};
    try compare(a, r, 4, limits.target(121).?, good, &t);
    try std.testing.expectEqual(@as(usize, 0), t.problems);
    r.usage[4].regs = 40;
    try compare(a, r, 4, limits.target(121).?, good, &t);
    try std.testing.expectEqual(@as(usize, 1), t.problems);
    r.launch.name = "control sample";
    try compare(a, r, 4, limits.target(121).?, good, &t);
    try std.testing.expectEqual(@as(usize, 1), t.controls_missed);
    r.launch.threads = 256;
    try compare(a, r, 4, limits.target(121).?, good, &t);
    try std.testing.expectEqual(@as(usize, 1), t.controls_red);
}
