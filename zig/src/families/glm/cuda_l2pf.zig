//! GLM L2 prefetch runtime adapted from jayleaton/glm53-tensorfold-spark patch 0460.
const std = @import("std");
const cuda = @import("cuda");
const plan = @import("cuda_l2pf_plan.zig");

pub const Hook = struct {
    runtime: *Prefetch,
    layer: usize,
    rows: usize,
    prompt: bool = false,

    pub fn fire(h: Hook, stream: cuda.Stream) !void {
        try h.runtime.site(stream, h.layer, .o, h.rows, h.prompt);
    }
};

pub const Prefetch = struct {
    gpa: std.mem.Allocator,
    module: cuda.Module,
    kernel: cuda.Function,
    mode: plan.Mode,
    max_rows: usize,
    threads: u32,
    table: cuda.DeviceBuffer,
    sink: cuda.DeviceBuffer,
    side: cuda.Stream,
    fork: cuda.Event,
    done: cuda.Event,
    sites: []const ?plan.PlannedSite,
    pending: bool = false,

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, builder: *const plan.Builder) !Prefetch {
        const mode = builder.settings.mode orelse return error.PrefetchDisabled;
        var module = try cuda.Module.load(d, cuda.kernels.glm_l2pf);
        errdefer module.unload();
        const kernel = try module.function("glm_l2pf");
        const table_bytes = std.mem.sliceAsBytes(builder.rows.items);
        var table = try cuda.DeviceBuffer.fromHost(d, table_bytes);
        errdefer table.free();
        var sink = try cuda.DeviceBuffer.alloc(d, @sizeOf(u32));
        errdefer sink.free();
        try sink.fill8(0, null);
        var side = try cuda.Stream.init(d, true);
        errdefer side.deinit();
        var fork = try cuda.Event.init(d, false);
        errdefer fork.deinit();
        var done = try cuda.Event.init(d, false);
        errdefer done.deinit();
        const sites = try gpa.dupe(?plan.PlannedSite, builder.sites.items);
        errdefer gpa.free(sites);
        return .{
            .gpa = gpa,
            .module = module,
            .kernel = kernel,
            .mode = mode,
            .max_rows = builder.settings.max_rows,
            .threads = builder.settings.threads,
            .table = table,
            .sink = sink,
            .side = side,
            .fork = fork,
            .done = done,
            .sites = sites,
        };
    }

    pub fn deinit(p: *Prefetch) !void {
        try p.side.synchronize();
        p.done.deinit();
        p.fork.deinit();
        p.side.deinit();
        p.sink.free();
        p.table.free();
        p.gpa.free(p.sites);
        p.module.unload();
        p.* = undefined;
    }

    pub fn site(p: *Prefetch, main: cuda.Stream, layer: usize, which: plan.Site, rows: usize, prompt: bool) !void {
        if (prompt or rows == 0 or rows > p.max_rows or layer >= p.sites.len / 3) return;
        const site_plan = p.sites[layer * 3 + @backingInt(which)] orelse return;
        try p.fork.record(main);
        try p.side.wait(p.fork);
        var args: cuda.Args = .{};
        args.add(p.table.ptr + @as(u64, site_plan.first) * @sizeOf(plan.Range));
        args.add(@as(c_int, @intCast(site_plan.count)));
        args.add(@as(c_int, @intCast(@backingInt(p.mode))));
        args.add(p.sink.ptr);
        try cuda.launch.launch(p.kernel, .{ .grid = .{ .x = site_plan.grid }, .block = .{ .x = p.threads } }, p.side, &args);
        p.pending = true;
    }

    pub fn join(p: *Prefetch, main: cuda.Stream) !void {
        if (!p.pending) return;
        try p.done.record(p.side);
        try main.wait(p.done);
        p.pending = false;
    }
};

test "prefetch forks at enabled decode sites and joins once before forward completion" {
    const Trace = struct {
        var calls: [16]u8 = undefined;
        var count: usize = 0;
        fn add(call: u8) void {
            calls[count] = call;
            count += 1;
        }
        fn record(event: cuda.abi.Event, stream: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
            add(if (@intFromPtr(event.?) == 1 and @intFromPtr(stream.?) == 10) 1 else if (@intFromPtr(event.?) == 2 and @intFromPtr(stream.?) == 20) 4 else 9);
            return 0;
        }
        fn wait(stream: cuda.abi.Stream, event: cuda.abi.Event, _: c_uint) callconv(.c) cuda.abi.Result {
            add(if (@intFromPtr(stream.?) == 20 and @intFromPtr(event.?) == 1) 2 else if (@intFromPtr(stream.?) == 10 and @intFromPtr(event.?) == 2) 5 else 9);
            return 0;
        }
        fn launch(_: cuda.abi.Function, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, stream: cuda.abi.Stream, _: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
            add(if (@intFromPtr(stream.?) == 20) 3 else 9);
            return 0;
        }
    };
    Trace.count = 0;
    var d: cuda.Driver = .{ .lib = undefined, .api = undefined };
    d.api.cuEventRecord = Trace.record;
    d.api.cuStreamWaitEvent = Trace.wait;
    d.api.cuLaunchKernel = Trace.launch;
    const sites = [_]?plan.PlannedSite{ .{ .first = 0, .count = 1, .grid = 1, .bytes = 4096 }, null, null };
    var p: Prefetch = .{
        .gpa = std.testing.allocator,
        .module = undefined,
        .kernel = .{ .d = &d, .handle = @ptrFromInt(30) },
        .mode = .bulk,
        .max_rows = 64,
        .threads = 128,
        .table = .{ .d = &d, .ptr = 4096, .len = 16 },
        .sink = .{ .d = &d, .ptr = 8192, .len = 4 },
        .side = .{ .d = &d, .handle = @ptrFromInt(20) },
        .fork = .{ .d = &d, .handle = @ptrFromInt(1) },
        .done = .{ .d = &d, .handle = @ptrFromInt(2) },
        .sites = &sites,
    };
    const main: cuda.Stream = .{ .d = &d, .handle = @ptrFromInt(10) };
    try p.site(main, 0, .a, 8, true);
    try p.site(main, 0, .a, 65, false);
    try p.site(main, 0, .f, 8, false);
    try p.join(main);
    try std.testing.expectEqual(@as(usize, 0), Trace.count);
    try p.site(main, 0, .a, 8, false);
    try p.join(main);
    try p.join(main);
    try std.testing.expectEqualSlices(u8, &.{ 1, 2, 3, 4, 5 }, Trace.calls[0..Trace.count]);
}
