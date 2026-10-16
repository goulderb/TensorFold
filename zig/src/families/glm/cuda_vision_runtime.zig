//! Rank-zero resident vision weights and per-request native encoder state.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const probe_mod = @import("cuda_vision_probe.zig");
const host = @import("cuda_vision_weights.zig");
const memory = @import("cuda_vision_engine_memory.zig");
const tower = @import("cuda_vision_engine.zig");
const pipeline = @import("cuda_vision_pipeline.zig");
const cudnn = @import("cudnn");
const sdpa = @import("vision_sdpa");

pub const Service = struct {
    gpa: std.mem.Allocator,
    context: cuda.abi.Context,
    stream: cuda.Stream,
    probe: *probe_mod.Probe,
    weights: memory.DeviceWeights,
    conv: cudnn.Library,
    attention: sdpa.Library,
    linear: cuda.cublaslt.Library,

    pub fn open(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, context: cuda.abi.Context, stream: cuda.Stream, dir: []const u8, probe: *probe_mod.Probe) !*Service {
        const self = try gpa.create(Service);
        errdefer gpa.destroy(self);
        var checkpoint = try core.Checkpoint.openModelPrefix(gpa, io, dir, "model.visual.");
        defer checkpoint.close();
        var host_weights = try host.Weights.load(gpa, probe.config, checkpoint.files.items);
        defer host_weights.deinit(gpa);
        const weights = try memory.DeviceWeights.init(gpa, d, &host_weights, 2 * 1024 * 1024 * 1024);
        errdefer {
            var w = weights;
            w.deinit();
        }
        var conv = try cudnn.Library.open();
        errdefer conv.close();
        var attention = try sdpa.Library.open();
        errdefer attention.close();
        var linear = try cuda.cublaslt.Library.open();
        errdefer linear.close();
        self.* = .{ .gpa = gpa, .context = context, .stream = stream, .probe = probe, .weights = weights, .conv = conv, .attention = attention, .linear = linear };
        return self;
    }

    pub fn close(self: *Service) void {
        self.linear.close();
        self.attention.close();
        self.conv.close();
        self.weights.deinit();
        const gpa = self.gpa;
        gpa.destroy(self);
    }

    pub const Encoded = struct {
        gpa: std.mem.Allocator,
        request: *pipeline.Request,
        encoders: []*tower.Engine,

        pub fn close(self: *Encoded) !void {
            try self.request.deinit();
            self.gpa.destroy(self.request);
            for (self.encoders) |encoder| {
                try encoder.deinit();
                self.gpa.destroy(encoder);
            }
            self.gpa.free(self.encoders);
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
        errdefer request.deinit() catch {};
        var encoders: std.ArrayList(*tower.Engine) = .empty;
        errdefer {
            for (encoders.items) |encoder| {
                encoder.deinit() catch {};
                self.gpa.destroy(encoder);
            }
            encoders.deinit(self.gpa);
        }
        for (request.images) |image| {
            const found = for (encoders.items) |encoder| {
                const geometry = encoder.activations.layout.geometry;
                if (geometry.gh == image.grid[1] and geometry.gw == image.grid[2]) break true;
            } else false;
            if (found) continue;
            if (encoders.items.len == 4) return error.TooManyVisionGeometries;
            const encoder = try self.gpa.create(tower.Engine);
            encoder.* = tower.Engine.init(self.gpa, &self.weights, self.stream, .{ .groups = 1, .gh = image.grid[1], .gw = image.grid[2] }, .{
                .convolution = &self.conv.api,
                .sdpa = &self.attention.api,
                .linear = &self.linear,
                .image = cuda.kernels.glm_vision_ops,
                .package = .target_qualification,
            }, .{ .workspace_limit = 64 * 1024 * 1024, .max_activation_bytes = 1024 * 1024 * 1024 }) catch |err| {
                self.gpa.destroy(encoder);
                return err;
            };
            encoders.append(self.gpa, encoder) catch |err| {
                encoder.deinit() catch {};
                self.gpa.destroy(encoder);
                return err;
            };
        }
        try request.encode(encoders.items, .{ .stream = self.stream, .replica = replica, .world = world, .rank = 0, .expected_context = self.context });
        return .{ .gpa = self.gpa, .request = request, .encoders = try encoders.toOwnedSlice(self.gpa) };
    }
};
