const std = @import("std");
const f = @import("nemotron_spill_facts.zig");

const gpa = std.testing.allocator;
const io = std.testing.io;

const Env = struct {
    vars: []const [2][]const u8,
    pub fn get(e: Env, name: [:0]const u8) ?[]const u8 {
        for (e.vars) |kv| if (std.mem.eql(u8, kv[0], name)) return kv[1];
        return null;
    }
};

test "no spill settings is no tier; an orphan setting, a missing or bad host ceiling are errors; GIB 0 is off" {
    try std.testing.expect((try f.Config.fromEnv(gpa, Env{ .vars = &.{} })) == null);
    try std.testing.expectError(error.SpillSettingWithoutGib, f.Config.fromEnv(gpa, Env{ .vars = &.{.{ "TENSORFOLD_SPILL_DIR", "/x" }} }));
    try std.testing.expectError(error.SpillSettingWithoutGib, f.Config.fromEnv(gpa, Env{ .vars = &.{.{ "TENSORFOLD_SPILL_HOST_GIB", "4" }} }));
    try std.testing.expect((try f.Config.fromEnv(gpa, Env{ .vars = &.{ .{ "TENSORFOLD_SPILL_GIB", "0" }, .{ "TENSORFOLD_SPILL_DIR", "/x" } } })) == null);
    try std.testing.expectError(error.SpillHostGibRequired, f.Config.fromEnv(gpa, Env{ .vars = &.{ .{ "TENSORFOLD_SPILL_GIB", "2" }, .{ "TENSORFOLD_SPILL_DIR", "/x" } } }));
    for ([_][]const u8{ "0", "-1", "nan", "inf", "1e300", "lots" }) |bad| {
        try std.testing.expectError(error.BadSpillHostGib, f.Config.fromEnv(gpa, Env{ .vars = &.{ .{ "TENSORFOLD_SPILL_GIB", "2" }, .{ "TENSORFOLD_SPILL_DIR", "/x" }, .{ "TENSORFOLD_SPILL_HOST_GIB", bad } } }));
    }
    try std.testing.expectError(error.BadSpillSetting, f.Config.fromEnv(gpa, Env{ .vars = &.{ .{ "TENSORFOLD_SPILL_GIB", "nan" }, .{ "TENSORFOLD_SPILL_HOST_GIB", "1" } } }));
    const c = (try f.Config.fromEnv(gpa, Env{ .vars = &.{ .{ "TENSORFOLD_SPILL_GIB", "2" }, .{ "TENSORFOLD_SPILL_DIR", "/nvme/spill" }, .{ "TENSORFOLD_SPILL_HOST_GIB", "1.5" } } })).?;
    defer gpa.free(c.tier.dir);
    try std.testing.expectEqual(@as(u64, 2 << 30), c.tier.cap);
    try std.testing.expectEqual(@as(u64, 3 << 29), c.host_bytes);
    try std.testing.expectEqualStrings("/nvme/spill", c.tier.dir);
}

test "RAM_GIB without GIB is the RAM-only tier: it still needs the host ceiling, and refuses the disk's settings" {
    const c = (try f.Config.fromEnv(gpa, Env{ .vars = &.{ .{ "TENSORFOLD_SPILL_RAM_GIB", "12" }, .{ "TENSORFOLD_SPILL_HOST_GIB", "1" }, .{ "TENSORFOLD_SPILL_MIN_TOKENS", "2048" } } })).?;
    defer gpa.free(c.tier.dir);
    try std.testing.expect(!c.tier.disk());
    try std.testing.expectEqual(@as(u64, 12 << 30), c.tier.ram_cap);
    try std.testing.expectEqual(@as(u32, 2048), c.tier.min_tokens);
    try std.testing.expectEqual(@as(u64, 1 << 30), c.host_bytes);
    try std.testing.expectError(error.SpillHostGibRequired, f.Config.fromEnv(gpa, Env{ .vars = &.{.{ "TENSORFOLD_SPILL_RAM_GIB", "12" }} }));
    try std.testing.expectError(error.SpillDiskSettingWithoutDisk, f.Config.fromEnv(gpa, Env{ .vars = &.{ .{ "TENSORFOLD_SPILL_RAM_GIB", "12" }, .{ "TENSORFOLD_SPILL_HOST_GIB", "1" }, .{ "TENSORFOLD_SPILL_DIR", "/x" } } }));
    // RAM_GIB 0 with no GIB is no tier, and the orphan check still applies
    try std.testing.expectError(error.SpillSettingWithoutGib, f.Config.fromEnv(gpa, Env{ .vars = &.{ .{ "TENSORFOLD_SPILL_RAM_GIB", "0" }, .{ "TENSORFOLD_SPILL_HOST_GIB", "1" } } }));
}

const Scratch = struct {
    tmp: std.testing.TmpDir,
    path: []u8,
    fn make() !Scratch {
        const tmp = std.testing.tmpDir(.{});
        return .{ .tmp = tmp, .path = try std.fmt.allocPrint(gpa, ".zig-cache/tmp/{s}", .{tmp.sub_path}) };
    }
    fn deinit(s: *Scratch) void {
        gpa.free(s.path);
        s.tmp.cleanup();
    }
    fn file(s: *const Scratch, name: []const u8, data: []const u8) ![:0]u8 {
        const p = try std.fmt.allocPrintSentinel(gpa, "{s}/{s}", .{ s.path, name }, 0);
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = p, .data = data });
        return p;
    }
};

fn list(paths: []const [:0]u8) ![][:0]u8 {
    const out = try gpa.alloc([:0]u8, paths.len);
    for (out, paths) |*o, p| o.* = try gpa.dupeSentinel(u8, p, 0);
    return out;
}

test "artifact digests cover every byte and refuse a file changed between the load and the hash" {
    var s = try Scratch.make();
    defer s.deinit();
    const a = try s.file("config.json", "{\"a\":1}");
    defer gpa.free(a);
    const b = try s.file("model.safetensors", "weights-bytes");
    defer gpa.free(b);
    {
        const file = try std.Io.Dir.cwd().openFile(io, b, .{ .mode = .read_write });
        defer file.close(io);
        // The rewrite must cross a timestamp tick even on filesystems with coarse write times.
        try file.setTimestamps(io, .{ .modify_timestamp = .{ .new = .fromNanoseconds(946684800 * std.time.ns_per_s) } });
    }
    var one = try f.Artifacts.take(gpa, try list(&.{ a, b }));
    defer one.deinit(gpa);
    const d1 = try one.digest(gpa);
    try std.testing.expectEqualSlices(u8, &d1, &(try one.digest(gpa)));
    // same content, other order: another digest
    var swapped = try f.Artifacts.take(gpa, try list(&.{ b, a }));
    defer swapped.deinit(gpa);
    try std.testing.expect(!std.mem.eql(u8, &d1, &(try swapped.digest(gpa))));
    // a rewrite after `take` (the load saw the old bytes) refuses the tier
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = b, .data = "weights-BYTES" });
    try std.testing.expectError(error.SpillArtifactChanged, one.digest(gpa));
    var fresh = try f.Artifacts.take(gpa, try list(&.{ a, b }));
    defer fresh.deinit(gpa);
    try std.testing.expect(!std.mem.eql(u8, &d1, &(try fresh.digest(gpa))));
    // a missing file refuses at once
    const gone = try std.fmt.allocPrintSentinel(gpa, "{s}/none", .{s.path}, 0);
    defer gpa.free(gone);
    try std.testing.expectError(error.SpillArtifactOpen, f.Artifacts.take(gpa, try list(&.{gone})));
}

test "an AOT set's files are aot.json and the cubins it names; a hash naming another path is refused" {
    var s = try Scratch.make();
    defer s.deinit();
    const j = try s.file("aot.json", "{\"kernels\":[{\"hash\":\"abc\",\"name\":\"k\"},{\"hash\":\"def\"}],\"other\":1}");
    defer gpa.free(j);
    const files = try f.aotFiles(gpa, io, s.path);
    defer {
        for (files) |p| gpa.free(p);
        gpa.free(files);
    }
    try std.testing.expectEqual(@as(usize, 3), files.len);
    try std.testing.expect(std.mem.endsWith(u8, files[1], "/cubins/abc.cubin"));
    try std.testing.expect(std.mem.endsWith(u8, files[2], "/cubins/def.cubin"));
    const bad = try s.file("aot.json", "{\"kernels\":[{\"hash\":\"../x\"}]}");
    defer gpa.free(bad);
    try std.testing.expectError(error.BadAotHash, f.aotFiles(gpa, io, s.path));
}

test "the identity changes with weights, executable, kernels, driver, device, format, any plane, head, window, grid or segments" {
    const base: f.Facts = .{ .weights = @splat(1), .executable = @splat(2), .kernels = "embedded", .driver = 13000, .capability = 121, .format = "mlx", .planes = .{ 8, 8, 16, 6, 6, 6, 12 }, .head = 10, .max_len = 64, .grid = 2048, .lookahead = 1, .segments = 1 };
    const d = try base.identity();
    try std.testing.expectEqualSlices(u8, &d, &(try base.identity()));
    var changed: [13]f.Facts = @splat(base);
    changed[0].weights[0] = 9;
    changed[1].executable[0] = 9;
    changed[2].kernels = "/aot";
    changed[3].driver = 13010;
    changed[4].capability = 120;
    changed[5].format = "modelopt";
    changed[6].planes[2] = 20;
    changed[7].head = 0;
    changed[8].max_len = 128;
    changed[9].grid = 1024;
    changed[10].segments = 2;
    changed[11].planes[6] = 16;
    changed[12].lookahead = 0;
    for (changed) |c| try std.testing.expect(!std.mem.eql(u8, &d, &(try c.identity())));
}

fn takeThenAdd(al: std.mem.Allocator, first: [:0]const u8, second: [:0]const u8) !void {
    const paths = try al.alloc([:0]u8, 1);
    paths[0] = al.dupeSentinel(u8, first, 0) catch |e| {
        al.free(paths);
        return e;
    };
    var a = try f.Artifacts.take(al, paths);
    defer a.deinit(al);
    try a.add(al, second);
    try std.testing.expectEqual(@as(usize, 2), a.paths.len);
}

test "adding an artifact under any failed allocation leaves the list whole: no leak, no double free" {
    var s = try Scratch.make();
    defer s.deinit();
    const a = try s.file("config.json", "{}");
    defer gpa.free(a);
    const b = try s.file("aot.json", "{\"kernels\":[]}");
    defer gpa.free(b);
    try std.testing.checkAllAllocationFailures(gpa, takeThenAdd, .{ a, b });
}
