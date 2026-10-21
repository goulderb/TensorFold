const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const manifest = @import("cuda_bf16_checkpoint.zig");
const checkpoint = @import("cuda_checkpoint.zig");
const layers = @import("cuda_layers.zig");
const Geometry = @import("cuda_geometry.zig").Geometry;

pub const Loaded = struct {
    gpa: std.mem.Allocator,
    stream: cuda.Stream,
    arena: cuda.DeviceBuffer,
    layers: []layers.Layer,
    final_norm: u64,
    host: manifest.HostTables,

    pub fn deinit(w: *Loaded) void {
        cuda.completion.drain(w.arena.d, w.stream, "BF16 weights still have an active device consumer");
        w.host.deinit();
        w.arena.free();
        w.gpa.free(w.layers);
        w.* = undefined;
    }
};

pub fn load(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, plan: *const manifest.Plan, d: *const cuda.Driver, stream: cuda.Stream, pack: *const cuda.modelopt.Packer) !Loaded {
    var arena = try cuda.DeviceBuffer.alloc(d, plan.bytes().device_weights);
    errdefer arena.free();
    var stage = try cuda.DeviceBuffer.alloc(d, plan.bytes().device_load_stage);
    defer stage.free();
    errdefer cuda.completion.drain(d, stream, "a failed BF16 weight load did not drain");
    var sink = try Sink(Device).init(gpa, plan.geometry, .{ .arena = arena, .stage = stage, .stream = stream, .pack = pack });
    errdefer gpa.free(sink.layers);
    const host = try plan.upload(gpa, io, dir, &sink);
    return .{ .gpa = gpa, .stream = stream, .arena = arena, .layers = sink.layers, .final_norm = sink.final_norm, .host = host };
}

const Device = struct {
    arena: cuda.DeviceBuffer,
    stage: cuda.DeviceBuffer,
    stream: cuda.Stream,
    pack: *const cuda.modelopt.Packer,

    fn address(d: Device, offset: usize, len: usize) !u64 {
        if (offset > d.arena.len or len > d.arena.len - offset) return error.InvalidBf16Allocation;
        return d.arena.at(offset);
    }

    fn upload(d: Device, offset: usize, bytes: []const u8) !void {
        try d.arena.upload(offset, bytes);
    }

    fn projection(d: Device, offset: usize, bytes: []const u8, n: usize, k: usize) !void {
        if (bytes.len > d.stage.len) return error.StageTooSmall;
        try d.stage.upload(0, bytes);
        try d.pack.bf16Lane(d.stream, d.stage.ptr, try d.arena.at(offset), n, k);
        try d.stream.synchronize();
    }
};

fn Sink(comptime Backend: type) type {
    return struct {
        const Self = @This();
        gpa: std.mem.Allocator,
        backend: Backend,
        layers: []layers.Layer,
        final_norm: u64 = 0,

        fn init(gpa: std.mem.Allocator, geometry: Geometry, backend: Backend) !Self {
            const local = try gpa.alloc(layers.Layer, geometry.layers);
            const p: layers.Proj = .{ .linear = .{ .bf16 = cuda.modelopt.bf16Weight(0, 0, 0) } };
            for (local, 0..) |*layer, i| layer.* = .{
                .input_norm = 0,
                .post_norm = 0,
                .mlp = .{ .dense = .{ .gate = p, .up = p, .down = p } },
                .mixer = if (geometry.linear[i])
                    .{ .delta = .{ .qkv = p, .z = p, .a = p, .b = p, .out = p, .conv = 0, .norm = 0, .a_log = 0, .dt_bias = 0 } }
                else
                    .{ .attention = .{ .q = p, .k = p, .v = p, .o = p, .q_norm = 0, .k_norm = 0 } },
            };
            return .{ .gpa = gpa, .backend = backend, .layers = local };
        }

        pub fn tensor(s: *Self, binding: anytype, t: core.safetensors.Tensor) !void {
            const offset = binding.device_offset orelse return error.InvalidBf16Binding;
            const ptr = try s.backend.address(offset, binding.allocation_bytes);
            switch (binding.conversion) {
                .host_rows => return error.InvalidBf16Binding,
                .bf16_lane_pack => {
                    const n = t.shape[0];
                    const k = t.shape[1];
                    try s.backend.projection(offset, t.bytes, n, k);
                    const p: layers.Proj = .{ .linear = .{ .bf16 = cuda.modelopt.bf16Weight(ptr, n, k) } };
                    const layer = &s.layers[binding.layer.?];
                    switch (binding.role) {
                        .gate => layer.mlp.dense.gate = p,
                        .up => layer.mlp.dense.up = p,
                        .down => layer.mlp.dense.down = p,
                        .q => layer.mixer.attention.q = p,
                        .k => layer.mixer.attention.k = p,
                        .v => layer.mixer.attention.v = p,
                        .o => layer.mixer.attention.o = p,
                        .qkv => layer.mixer.delta.qkv = p,
                        .z => layer.mixer.delta.z = p,
                        .a => layer.mixer.delta.a = p,
                        .b => layer.mixer.delta.b = p,
                        .linear_out => layer.mixer.delta.out = p,
                        else => return error.InvalidBf16Binding,
                    }
                },
                .raw_f32, .zero_centered_norm_f32 => {
                    const values = try checkpoint.floats(s.gpa, t, if (binding.conversion == .raw_f32) .raw else .zero_centered_norm);
                    defer s.gpa.free(values);
                    try s.backend.upload(offset, std.mem.sliceAsBytes(values));
                    if (binding.role == .final_norm) {
                        s.final_norm = ptr;
                        return;
                    }
                    const layer = &s.layers[binding.layer.?];
                    switch (binding.role) {
                        .input_norm => layer.input_norm = ptr,
                        .post_norm => layer.post_norm = ptr,
                        .q_norm => layer.mixer.attention.q_norm = ptr,
                        .k_norm => layer.mixer.attention.k_norm = ptr,
                        .conv => layer.mixer.delta.conv = ptr,
                        .a_log => layer.mixer.delta.a_log = ptr,
                        .dt_bias => layer.mixer.delta.dt_bias = ptr,
                        .gdn_norm => layer.mixer.delta.norm = ptr,
                        else => return error.InvalidBf16Binding,
                    }
                },
            }
        }
    };
}

test "BF16 payload loader has a complete host-buildable implementation" {
    std.mem.doNotOptimizeAway(&load);
    std.mem.doNotOptimizeAway(&Loaded.deinit);
}

const Binding = @typeInfo(@FieldType(manifest.Plan, "bindings")).pointer.child;
const Role = @FieldType(Binding, "role");
const Recording = struct {
    offset: usize = 0,
    n: usize = 0,
    k: usize = 0,
    calls: usize = 0,
    bytes: [16]u8 = @splat(0),
    fail: bool = false,

    fn address(_: *Recording, offset: usize, len: usize) !u64 {
        if (offset > 8192 or len > 8192 - offset) return error.InvalidBf16Allocation;
        return 4096 + offset;
    }
    fn upload(r: *Recording, offset: usize, bytes: []const u8) !void {
        if (r.fail) return error.InjectedUpload;
        r.calls += 1;
        r.offset = offset;
        @memcpy(r.bytes[0..bytes.len], bytes);
    }
    fn projection(r: *Recording, offset: usize, bytes: []const u8, n: usize, k: usize) !void {
        r.n = n;
        r.k = k;
        try r.upload(offset, bytes);
    }
};

fn testGeometry() !Geometry {
    var g = try @import("cuda_bf16_config.zig").parse(std.testing.allocator, @embedFile("fixtures/clef_flash/config.json"));
    g.layers = 2;
    g.linear[0] = true;
    g.linear[1] = false;
    return g;
}

fn testBinding(role: Role, layer: ?u16, projection: bool) Binding {
    return .{
        .role = role,
        .layer = layer,
        .name = "validated",
        .shard = 0,
        .file_offset = 0,
        .source = .{ .dtype = .bf16, .rank = if (projection) 2 else 1, .shape = .{ 2, 1, 1, 1, 1 }, .begin = 0, .end = 4 },
        .conversion = if (projection) .bf16_lane_pack else .zero_centered_norm_f32,
        .allocation_bytes = if (projection) 256 else 8,
        .device_offset = 256,
    };
}

fn testTensor(projection: bool) core.safetensors.Tensor {
    return .{ .dtype = .bf16, .rank = if (projection) 2 else 1, .shape = .{ 2, 1, 1, 1, 1 }, .bytes = &.{ 0xc0, 0x3f, 0x00, 0xbf } };
}

test "BF16 sink packs shared projection views and binds dense GDN and attention destinations" {
    var record: Recording = .{};
    var sink = try Sink(*Recording).init(std.testing.allocator, try testGeometry(), &record);
    defer std.testing.allocator.free(sink.layers);
    const cases = .{ .{ Role.gate, 0 }, .{ Role.up, 0 }, .{ Role.down, 0 }, .{ Role.qkv, 0 }, .{ Role.z, 0 }, .{ Role.a, 0 }, .{ Role.b, 0 }, .{ Role.linear_out, 0 }, .{ Role.q, 1 }, .{ Role.k, 1 }, .{ Role.v, 1 }, .{ Role.o, 1 } };
    inline for (cases) |c| {
        try sink.tensor(testBinding(c[0], c[1], true), testTensor(true));
        const layer = sink.layers[c[1]];
        const p = switch (c[0]) {
            .gate => layer.mlp.dense.gate,
            .up => layer.mlp.dense.up,
            .down => layer.mlp.dense.down,
            .qkv => layer.mixer.delta.qkv,
            .z => layer.mixer.delta.z,
            .a => layer.mixer.delta.a,
            .b => layer.mixer.delta.b,
            .linear_out => layer.mixer.delta.out,
            .q => layer.mixer.attention.q,
            .k => layer.mixer.attention.k,
            .v => layer.mixer.attention.v,
            .o => layer.mixer.attention.o,
            else => unreachable,
        };
        try std.testing.expectEqual(@as(u64, 4352), p.linear.bf16.codes);
        try std.testing.expectEqual(@as(u32, 2), p.linear.bf16.n);
        try std.testing.expectEqual(@as(u32, 1), p.linear.bf16.k);
        try std.testing.expectEqual(@as(u32, 128), p.linear.bf16.npad);
        try std.testing.expectEqualSlices(u8, testTensor(true).bytes, record.bytes[0..4]);
    }
    try std.testing.expectEqual(cases.len, record.calls);
}

test "BF16 sink widens norm and state parameters with distinct zero center semantics" {
    var record: Recording = .{};
    var sink = try Sink(*Recording).init(std.testing.allocator, try testGeometry(), &record);
    defer std.testing.allocator.free(sink.layers);
    inline for (.{ Role.final_norm, Role.input_norm, Role.post_norm, Role.q_norm, Role.k_norm, Role.conv, Role.a_log, Role.dt_bias, Role.gdn_norm }) |role| {
        var binding = testBinding(role, if (role == .final_norm) null else if (role == .q_norm or role == .k_norm) 1 else 0, false);
        const raw = role == .conv or role == .a_log or role == .dt_bias or role == .gdn_norm;
        if (raw) binding.conversion = .raw_f32;
        try sink.tensor(binding, testTensor(false));
        const a: f32 = @bitCast(std.mem.readInt(u32, record.bytes[0..4], .little));
        const b: f32 = @bitCast(std.mem.readInt(u32, record.bytes[4..8], .little));
        try std.testing.expectEqual(@as(f32, if (raw) 1.5 else 2.5), a);
        try std.testing.expectEqual(@as(f32, if (raw) -0.5 else 0.5), b);
        const ptr = switch (role) {
            .final_norm => sink.final_norm,
            .input_norm => sink.layers[0].input_norm,
            .post_norm => sink.layers[0].post_norm,
            .q_norm => sink.layers[1].mixer.attention.q_norm,
            .k_norm => sink.layers[1].mixer.attention.k_norm,
            .conv => sink.layers[0].mixer.delta.conv,
            .a_log => sink.layers[0].mixer.delta.a_log,
            .dt_bias => sink.layers[0].mixer.delta.dt_bias,
            .gdn_norm => sink.layers[0].mixer.delta.norm,
            else => unreachable,
        };
        try std.testing.expectEqual(@as(u64, 4352), ptr);
    }
}

test "BF16 sink rejects invalid parameters or allocation spans before upload and never binds failed uploads" {
    var record: Recording = .{};
    var sink = try Sink(*Recording).init(std.testing.allocator, try testGeometry(), &record);
    defer std.testing.allocator.free(sink.layers);
    var binding = testBinding(.final_norm, null, false);
    var t = testTensor(false);
    t.bytes = &.{ 0x80, 0x7f, 0x00, 0x00 };
    try std.testing.expectError(error.InvalidQwenParameter, sink.tensor(binding, t));
    try std.testing.expectEqual(@as(usize, 0), record.calls);
    binding.device_offset = 8191;
    try std.testing.expectError(error.InvalidBf16Allocation, sink.tensor(binding, testTensor(false)));
    binding.device_offset = 256;
    record.fail = true;
    try std.testing.expectError(error.InjectedUpload, sink.tensor(binding, testTensor(false)));
    try std.testing.expectError(error.InjectedUpload, sink.tensor(testBinding(.a, 0, true), testTensor(true)));
    try std.testing.expectEqual(@as(u64, 0), sink.final_norm);
    try std.testing.expectEqual(@as(u64, 0), sink.layers[0].mixer.delta.a.linear.bf16.codes);
}
