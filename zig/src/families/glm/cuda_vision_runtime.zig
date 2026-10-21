//! Retained native-codec/request-binding adapter over the canonical stored-format vision Tower.
//! The old materializing loader and independent BF16 device runtime are not a serving path;
//! their explicit conversion and numerical qualification APIs retain their original coverage.
const std = @import("std");
const cuda = @import("cuda");
const probe_mod = @import("cuda_vision_probe.zig");
const vision = @import("cuda_vision.zig");
const images_mod = @import("cuda_images.zig");
const pipeline = @import("cuda_vision_pipeline.zig");

test "borrowed service retains the published optional Tower owner and never destroys it" {
    const Fixture = struct {
        fn run(a: std.mem.Allocator) !void {
            var d = @import("cuda_test_driver.zig").Memory.driver();
            const images = try a.create(images_mod.Images);
            defer a.destroy(images);
            images.* = undefined;
            images.d = &d;
            images.run = .{ .stream = .{ .d = &d, .handle = null }, .f = undefined };
            // Move the unpublished value into its stable owner before taking any loan.
            const unpublished: vision.Tower = .{ .gpa = a, .d = &d, .lt = undefined, .mem = undefined, .ql = .{}, .inv_freq = undefined, .linears = .init(a), .ws = undefined };
            images.tower = unpublished;
            defer images.tower.?.linears.deinit();
            var probe: probe_mod.Probe = undefined; // borrowed, never used by open/close
            const service = try Service.open(a, std.testing.io, &d, null, images.run.stream, "not-read", &probe, .{}, images);
            const actual = if (images.tower) |*owner| owner else unreachable;
            try std.testing.expect(service.tower == actual);
            try std.testing.expect(service.owned_tower == null and service.module == null);
            service.tower.bytes = 173;
            service.tower.busy.store(true, .release);
            try std.testing.expectEqual(@as(u64, 173), actual.bytes);
            try std.testing.expect(actual.busy.load(.acquire));
            service.close(); // drains the borrowed stream, but does not deinit Tower
            try std.testing.expect(actual.busy.load(.acquire));
            try std.testing.expectEqual(@as(u64, 173), actual.bytes);
            actual.busy.store(false, .release);
        }
    };
    try std.testing.checkAllAllocationFailures(std.testing.allocator, Fixture.run, .{});
}

pub const Service = struct {
    gpa: std.mem.Allocator,
    context: cuda.abi.Context,
    probe: *probe_mod.Probe,
    tower: *vision.Tower,
    run: vision.Run,
    owned_tower: ?vision.Tower = null,
    module: ?cuda.Module = null,

    /// Reuse Images' canonical owner when present; otherwise load that same stored-format Tower.
    /// `probe`, `shared` and qlinear's modules outlive this adapter and every encode it issues.
    pub fn open(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, context: cuda.abi.Context, stream: cuda.Stream, dir: []const u8, probe: *probe_mod.Probe, ql: cuda.qlinear.Linear, shared: ?*images_mod.Images) !*Service {
        const self = try gpa.create(Service);
        errdefer gpa.destroy(self);
        self.* = .{ .gpa = gpa, .context = context, .probe = probe, .tower = undefined, .run = undefined };
        if (shared) |v| {
            if (v.d != d or v.run.stream.d != d or v.run.stream.handle != stream.handle) return error.VisionPipelineDriverMismatch;
            self.tower = if (v.tower) |*t| t else return error.NoVisionTower;
            self.run = v.run;
            return self;
        }
        var module = try cuda.Module.load(d, cuda.kernels.glm_vision);
        errdefer module.unload();
        const functions = try vision.Funcs.load(module);
        self.owned_tower = try vision.Tower.load(gpa, io, d, dir, ql);
        self.tower = if (self.owned_tower) |*t| t else unreachable;
        self.module = module;
        self.run = .{ .stream = stream, .f = functions };
        return self;
    }

    pub fn close(self: *Service) void {
        @import("cuda_load_owner.zig").drain(self.run.stream, self.run.stream.d);
        if (self.owned_tower) |*t| t.deinit();
        if (self.module) |*m| m.unload();
        const gpa = self.gpa;
        gpa.destroy(self);
    }

    pub const Encoded = struct {
        gpa: std.mem.Allocator,
        request: *pipeline.Request,

        pub fn close(self: *Encoded) !void {
            // Request owns every CPU patch/position/id and the completed device feature table.
            // Failed fencing/free retains that ownership for a retry.
            try self.request.deinit();
            self.gpa.destroy(self.request);
            self.* = undefined;
        }
    };

    pub fn encode(self: *Service, encoded: []const []const u8, ids: []const u32, replica: u64, world: u32) !Encoded {
        const request = try self.gpa.create(pipeline.Request);
        errdefer self.gpa.destroy(request);
        self.probe.mutex.lockUncancelable(self.probe.io);
        const prepared = pipeline.Request.prepare(self.gpa, &self.probe.library, self.probe.config, encoded, ids, .{ .prompt_limit = ids.len + 1 });
        self.probe.mutex.unlock(self.probe.io);
        request.* = try prepared;
        errdefer request.deinit() catch {
            // A failed context fence cannot authorize releasing request-owned CUDA/host sources.
            @import("cuda_load_owner.zig").drain(self.run.stream, self.run.stream.d);
            request.deinit() catch @panic("vision request cleanup failed after drain");
        };
        try request.encodeTower(self.tower, &self.run, .{ .stream = self.run.stream, .replica = replica, .world = world, .rank = 0, .expected_context = self.context });
        return .{ .gpa = self.gpa, .request = request };
    }
};
