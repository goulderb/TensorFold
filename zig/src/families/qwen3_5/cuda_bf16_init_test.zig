//! Recording-driver initialization with real mapped BF16 payloads and shared weight ownership.
const std = @import("std");
const cuda = @import("cuda");
const bf16 = @import("cuda_bf16_checkpoint.zig");
const bindings = @import("cuda_bf16.zig");
const layers = @import("cuda_layers.zig");
const abi = cuda.abi;

const Fake = struct {
    var device: [8]?[]align(256) u8 = @splat(null);
    var host: [8]?[]align(16) u8 = @splat(null);
    var allocations: usize = 0;
    var fail_at: ?usize = null;
    var unloads: usize = 0;
    var streams: usize = 0;
    var syncs: usize = 0;

    fn refuse() bool {
        const index = allocations;
        allocations += 1;
        return fail_at == index;
    }
    fn alloc(out: *abi.DevicePtr, len: usize) callconv(.c) abi.Result {
        if (refuse()) return abi.error_not_ready;
        for (&device) |*slot| if (slot.* == null) {
            const bytes = std.testing.allocator.alignedAlloc(u8, .fromByteUnits(256), len) catch return abi.error_not_ready;
            slot.* = bytes;
            out.* = @intFromPtr(bytes.ptr);
            return 0;
        };
        return abi.error_not_ready;
    }
    fn free(ptr: abi.DevicePtr) callconv(.c) abi.Result {
        for (&device) |*slot| if (slot.*) |bytes| {
            if (@intFromPtr(bytes.ptr) == ptr) {
                std.testing.allocator.free(bytes);
                slot.* = null;
                return 0;
            }
        };
        return 1;
    }
    fn hostAlloc(out: *?*anyopaque, len: usize, _: c_uint) callconv(.c) abi.Result {
        if (refuse()) return abi.error_not_ready;
        for (&host) |*slot| if (slot.* == null) {
            const bytes = std.testing.allocator.alignedAlloc(u8, .@"16", len) catch return abi.error_not_ready;
            slot.* = bytes;
            out.* = bytes.ptr;
            return 0;
        };
        return abi.error_not_ready;
    }
    fn hostFree(ptr: ?*anyopaque) callconv(.c) abi.Result {
        for (&host) |*slot| if (slot.*) |bytes| {
            if (@intFromPtr(bytes.ptr) == @intFromPtr(ptr)) {
                std.testing.allocator.free(bytes);
                slot.* = null;
                return 0;
            }
        };
        return 1;
    }
    fn upload(dst: u64, src: ?*const anyopaque, len: usize) callconv(.c) abi.Result {
        @memcpy(@as([*]u8, @ptrFromInt(dst))[0..len], @as([*]const u8, @ptrCast(src.?))[0..len]);
        return 0;
    }
    fn fill(dst: u64, value: u8, len: usize) callconv(.c) abi.Result {
        @memset(@as([*]u8, @ptrFromInt(dst))[0..len], value);
        return 0;
    }
    fn sync(_: abi.Stream) callconv(.c) abi.Result {
        syncs += 1;
        return 0;
    }
    fn createStream(out: *abi.Stream, _: c_uint) callconv(.c) abi.Result {
        streams += 1;
        out.* = @ptrFromInt(1);
        return 0;
    }
    fn destroyStream(_: abi.Stream) callconv(.c) abi.Result {
        streams -= 1;
        return 0;
    }
    fn unload(_: abi.Module) callconv(.c) abi.Result {
        unloads += 1;
        return 0;
    }
    pub fn load(ctx: *const cuda.Context, fns: *layers.Fns) !bindings.Bindings {
        const module: cuda.Module = .{ .d = ctx.d, .handle = @ptrFromInt(1) };
        inline for (@typeInfo(layers.Fns).@"struct".field_names) |name|
            @field(fns, name) = .{ .d = ctx.d, .handle = @ptrFromInt(1) };
        var result: bindings.Bindings = undefined;
        result.image = module;
        result.glue = module;
        result.lane.mod = module;
        result.lane.clusters = false;
        result.pack.mod = module;
        result.part = 0;
        return result;
    }
    fn driver() cuda.Driver {
        var d: cuda.Driver = undefined;
        d.api.cuMemAlloc_v2 = alloc;
        d.api.cuMemFree_v2 = free;
        d.api.cuMemHostAlloc = hostAlloc;
        d.api.cuMemFreeHost = hostFree;
        d.api.cuMemcpyHtoD_v2 = upload;
        d.api.cuMemsetD8_v2 = fill;
        d.api.cuStreamSynchronize = sync;
        d.api.cuStreamCreate = createStream;
        d.api.cuStreamDestroy_v2 = destroyStream;
        d.api.cuModuleUnload = unload;
        return d;
    }
    fn released() !void {
        for (device) |slot| try std.testing.expect(slot == null);
        for (host) |slot| try std.testing.expect(slot == null);
        try std.testing.expectEqual(@as(usize, 0), streams);
        try std.testing.expectEqual(@as(usize, 4), unloads);
    }
};

pub fn exercise(comptime initialize: anytype) !void {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const header = "{\"embedding\":{\"dtype\":\"BF16\",\"shape\":[2,4096],\"data_offsets\":[0,16384]},\"output\":{\"dtype\":\"BF16\",\"shape\":[2,4096],\"data_offsets\":[16384,32768]},\"norm\":{\"dtype\":\"BF16\",\"shape\":[4096],\"data_offsets\":[32768,40960]}}";
    var payload: [40960]u8 = @splat(0);
    @memset(payload[0..16384], 0x11);
    @memset(payload[16384..32768], 0x22);
    const file = try tmp.dir.createFile(io, "weights.safetensors", .{ .read = true });
    defer file.close(io);
    var prefix: [8]u8 = undefined;
    std.mem.writeInt(u64, &prefix, header.len, .little);
    try file.writePositionalAll(io, &prefix, 0);
    try file.writePositionalAll(io, header, 8);
    try file.writePositionalAll(io, &payload, 8 + header.len);
    var shards: [1]@typeInfo(@FieldType(bf16.Plan, "shards")).pointer.child = undefined;
    shards[0] = .{ .name = "weights.safetensors", .file_bytes = 8 + header.len + payload.len, .header_sha256 = undefined };
    std.crypto.hash.sha2.Sha256.hash(header, &shards[0].header_sha256, .{});
    var roles: [3]@typeInfo(@FieldType(bf16.Plan, "bindings")).pointer.child = undefined;
    for (&roles, 0..) |*role, i| role.* = .{
        .role = if (i == 0) .embedding else if (i == 1) .output else .final_norm,
        .layer = null,
        .name = if (i == 0) "embedding" else if (i == 1) "output" else "norm",
        .shard = 0,
        .file_offset = 8 + header.len + i * 16384,
        .source = .{ .dtype = .bf16, .rank = if (i < 2) 2 else 1, .shape = if (i < 2) .{ 2, 4096, 1, 1, 1 } else .{ 4096, 1, 1, 1, 1 }, .begin = i * 16384, .end = if (i < 2) (i + 1) * 16384 else 40960 },
        .conversion = if (i < 2) .host_rows else .zero_centered_norm_f32,
        .allocation_bytes = 16384,
        .device_offset = if (i < 2) null else 0,
    };
    var g = try @import("cuda_bf16_config.zig").parse(a, @embedFile("fixtures/clef_flash/config.json"));
    g.layers = 0;
    var plan: bf16.Plan = .{ .arena = .init(a), .geometry = g, .shards = &shards, .bindings = &roles, .excluded_visual_count = 0, .budget = .{ .device_weights = 16384, .host_tables = 32768 } };
    defer plan.deinit();
    const dir = try tmp.dir.realPathFileAlloc(io, ".", a);
    defer a.free(dir);
    var d = Fake.driver();
    const ctx: cuda.Context = .{ .d = &d, .handle = cuda.copy_source.current(), .device = 0 };
    for (0..6) |attempt| {
        Fake.allocations = 0;
        Fake.fail_at = if (attempt < 5) attempt else null;
        Fake.unloads = 0;
        Fake.syncs = 0;
        if (initialize(Fake, a, io, &ctx, dir, &plan)) |e| {
            try std.testing.expectEqual(@as(usize, 5), attempt);
            try std.testing.expectEqual(@as(usize, 5), Fake.allocations);
            try std.testing.expect(e.terminal_weights == .hidden_only);
            try std.testing.expect(e.modelopt == null);
            try std.testing.expect(e.learning == null);
            try std.testing.expect(e.mlp == null);
            try std.testing.expect(!e.pockets.any());
            try std.testing.expectEqual(g.layers, e.pockets.layers.len);
            try std.testing.expectEqual(@as(u64, 0), e.logits);
            try std.testing.expectEqual(@as(usize, 0), e.host_logits.bytes.len);
            try std.testing.expectEqual(@as(usize, 131072), e.bf16.?.input.bytes.len);
            var row: [8192]u8 = undefined;
            try e.gatherOutputRows(&.{1}, &row);
            for (row) |byte| try std.testing.expectEqual(@as(u8, 0x22), byte);
            try e.bf16.?.weights.host.gather(.embedding, &.{0}, &row);
            for (row) |byte| try std.testing.expectEqual(@as(u8, 0x11), byte);
            e.deinit();
        } else |err| {
            try std.testing.expect(attempt < 5);
            try std.testing.expectEqual(error.NotReady, err);
        }
        try Fake.released();
        const size = try file.length(io);
        try file.setLength(io, size - 1);
        try file.setLength(io, size);
        try file.writePositionalAll(io, payload[payload.len - 1 ..], size - 1);
    }
}
