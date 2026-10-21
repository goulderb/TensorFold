const std = @import("std");
const core = @import("core");
const config = @import("cuda_bf16_config.zig");
const Geometry = @import("cuda_geometry.zig").Geometry;
const st = core.safetensors;

pub const Bytes = struct {
    device_weights: usize = 0,
    device_load_stage: usize = 0,
    host_embedding: usize = 0,
    host_output: usize = 0,
    host_tables: usize = 0,
    host_conversion_peak: usize = 0,
    /// One input gather and one lexical gather, each bounded to one 4096-token record.
    host_row_staging: usize = 0,
};

const Conversion = enum { host_rows, bf16_lane_pack, raw_f32, zero_centered_norm_f32 };
const Role = enum {
    embedding,
    output,
    final_norm,
    input_norm,
    post_norm,
    gate,
    up,
    down,
    q,
    k,
    v,
    o,
    q_norm,
    k_norm,
    qkv,
    z,
    a,
    b,
    linear_out,
    conv,
    a_log,
    dt_bias,
    gdn_norm,
};
const Binding = struct {
    role: Role,
    layer: ?u16,
    name: []const u8,
    shard: usize,
    file_offset: u64,
    source: st.Entry,
    conversion: Conversion,
    allocation_bytes: usize,
    device_offset: ?usize,
};
const Shard = struct { name: []const u8, header: []const u8, file_bytes: u64 };
const ShardIdentity = struct { name: []const u8, file_bytes: u64, header_sha256: [32]u8 };
const Located = struct { entry: st.Entry, shard: usize, offset: u64 };

const PayloadMap = struct {
    file: std.Io.File,
    map: std.Io.File.MemoryMap,

    fn open(io: std.Io, path: []const u8, identity: ShardIdentity) !PayloadMap {
        const file = try std.Io.Dir.cwd().openFile(io, path, .{});
        errdefer file.close(io);
        const length = try file.length(io);
        if (length != identity.file_bytes) return error.CheckpointChanged;
        const len = std.math.cast(usize, length) orelse return error.Overflow;
        if (len < 8) return error.BadHeader;
        var map = try std.Io.File.MemoryMap.create(io, file, .{ .len = len, .protection = .{ .read = true, .write = false }, .populate = false });
        errdefer map.destroy(io);
        const header_len = std.mem.readInt(u64, map.memory[0..8], .little);
        if (header_len > 16 << 20 or header_len > len - 8) return error.BadHeader;
        var digest: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(map.memory[8..][0..@intCast(header_len)], &digest, .{});
        if (!std.mem.eql(u8, &digest, &identity.header_sha256)) return error.CheckpointChanged;
        return .{ .file = file, .map = map };
    }

    fn deinit(self: *PayloadMap, io: std.Io) void {
        self.map.destroy(io);
        self.file.close(io);
        self.* = undefined;
    }
};

pub const HostTables = struct {
    pub const Kind = enum { embedding, output };
    const Table = struct { bytes: []const u8, rows: usize, row_bytes: usize, shard: usize };

    gpa: std.mem.Allocator,
    io: std.Io,
    maps: []?PayloadMap,
    tables: [2]?Table = .{ null, null },

    pub fn deinit(self: *HostTables) void {
        for (self.maps) |*slot| if (slot.*) |*mapping| mapping.deinit(self.io);
        self.gpa.free(self.maps);
        self.* = undefined;
    }

    /// Whole retained shard mappings, including payload outside the logical tables.
    pub fn mappedBytes(self: *const HostTables) usize {
        var total: usize = 0;
        for (self.maps) |slot| if (slot) |mapping| {
            total += mapping.map.memory.len;
        };
        return total;
    }

    pub fn gather(self: *const HostTables, kind: Kind, ids: []const u32, out: []u8) !void {
        if (ids.len > 4096) return error.TooManyRows;
        const table = self.tables[@backingInt(kind)] orelse return error.MissingHostTable;
        if (out.len != try mul(ids.len, table.row_bytes)) return error.BadGatherSize;
        for (ids) |id| if (id >= table.rows) return error.TokenOutOfRange;
        for (ids, 0..) |id, i| {
            const source = try mul(id, table.row_bytes);
            const dest = try mul(i, table.row_bytes);
            @memcpy(out[dest..][0..table.row_bytes], table.bytes[source..][0..table.row_bytes]);
        }
    }
};

pub const Plan = struct {
    arena: std.heap.ArenaAllocator,
    geometry: Geometry,
    shards: []const ShardIdentity,
    bindings: []const Binding,
    excluded_visual_count: usize,
    budget: Bytes,

    pub fn inspect(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !Plan {
        var arena: std.heap.ArenaAllocator = .init(gpa);
        errdefer arena.deinit();
        const a = arena.allocator();
        const cfg = try readSmall(a, io, dir, "config.json", 4 << 20);
        const index = try readSmall(a, io, dir, "model.safetensors.index.json", 8 << 20);
        const owners = try indexMap(a, index);
        var names: std.StringHashMapUnmanaged(void) = .empty;
        var shards: std.ArrayList(Shard) = .empty;
        for (owners.values()) |value| {
            const name = value.string;
            if ((try names.getOrPut(a, name)).found_existing) continue;
            if (shards.items.len >= 128) return error.TooManyShards;
            const f = try std.Io.Dir.cwd().openFile(io, try std.fs.path.join(a, &.{ dir, name }), .{});
            defer f.close(io);
            try shards.append(a, try readShard(a, io, f, name));
        }
        var plan = try build(a, cfg, index, shards.items);
        plan.arena = arena;
        return plan;
    }

    pub fn deinit(self: *Plan) void {
        self.arena.deinit();
        self.* = undefined;
    }
    pub fn bytes(self: *const Plan) Bytes {
        return self.budget;
    }
    pub fn tensorCount(self: *const Plan) usize {
        return self.bindings.len;
    }

    /// Requires an immutable externally pinned checkpoint. Header identity is not a payload hash.
    /// The synchronous target must consume borrowed tensor bytes before returning.
    pub fn upload(self: *const Plan, gpa: std.mem.Allocator, io: std.Io, dir: []const u8, target: anytype) !HostTables {
        var host: HostTables = .{ .gpa = gpa, .io = io, .maps = try gpa.alloc(?PayloadMap, self.shards.len) };
        @memset(host.maps, null);
        errdefer host.deinit();
        var total_mapped: usize = 0;
        for (self.shards, 0..) |identity, i| {
            const path = try std.fs.path.join(gpa, &.{ dir, identity.name });
            defer gpa.free(path);
            host.maps[i] = try PayloadMap.open(io, path, identity);
            total_mapped = try add(total_mapped, host.maps[i].?.map.memory.len);
        }
        for (self.bindings) |binding| {
            const memory = host.maps[binding.shard].?.map.memory;
            const begin = std.math.cast(usize, binding.file_offset) orelse return error.Overflow;
            const end = try add(begin, binding.source.end - binding.source.begin);
            if (end > memory.len) return error.CheckpointChanged;
            const tensor: st.Tensor = .{ .dtype = binding.source.dtype, .rank = binding.source.rank, .shape = binding.source.shape, .bytes = memory[begin..end] };
            if (binding.conversion == .host_rows) {
                const kind: HostTables.Kind = switch (binding.role) {
                    .embedding => .embedding,
                    .output => .output,
                    else => unreachable,
                };
                host.tables[@backingInt(kind)] = .{ .bytes = tensor.bytes, .rows = tensor.shape[0], .row_bytes = try mul(tensor.shape[1], 2), .shard = binding.shard };
            } else try target.tensor(binding, tensor);
        }
        const embedding = host.tables[0] orelse return error.MissingHostTable;
        const output = host.tables[1] orelse return error.MissingHostTable;
        for (host.maps, 0..) |*slot, i| {
            if (i == embedding.shard or i == output.shard) continue;
            slot.*.?.deinit(io);
            slot.* = null;
        }
        return host;
    }
};

fn readSmall(a: std.mem.Allocator, io: std.Io, dir: []const u8, name: []const u8, limit: usize) ![]const u8 {
    return std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(a, &.{ dir, name }), a, .limited(limit));
}

fn readShard(a: std.mem.Allocator, io: std.Io, f: std.Io.File, name: []const u8) !Shard {
    const size = try f.length(io);
    var prefix: [8]u8 = undefined;
    if (try f.readPositionalAll(io, &prefix, 0) != 8) return error.BadHeader;
    const len = std.mem.readInt(u64, &prefix, .little);
    const offset = try add64(8, len);
    if (len > 16 << 20 or offset > size) return error.BadHeader;
    const header = try a.alloc(u8, std.math.cast(usize, len) orelse return error.Overflow);
    if (try f.readPositionalAll(io, header, 8) != header.len) return error.BadHeader;
    return .{ .name = name, .header = header, .file_bytes = size };
}

fn safeBasename(name: []const u8) bool {
    if (name.len == 0 or name.len > 255 or name[0] == '.' or
        !std.mem.endsWith(u8, name, ".safetensors") or std.mem.indexOf(u8, name, "..") != null) return false;
    for (name) |c| if (!std.ascii.isAlphanumeric(c) and c != '.' and c != '-' and c != '_') return false;
    const stem = name[0..std.mem.indexOfScalar(u8, name, '.').?];
    inline for (.{ "CON", "PRN", "AUX", "NUL", "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9", "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9" }) |reserved| {
        if (std.ascii.eqlIgnoreCase(stem, reserved)) return false;
    }
    return true;
}

fn indexMap(a: std.mem.Allocator, bytes: []const u8) !std.json.ObjectMap {
    const root = try std.json.parseFromSliceLeaky(std.json.Value, a, bytes, .{});
    if (root != .object) return error.BadIndex;
    for (root.object.keys()) |key| if (!std.mem.eql(u8, key, "metadata") and !std.mem.eql(u8, key, "weight_map")) return error.BadIndex;
    if (root.object.get("metadata")) |meta| if (meta != .object) return error.BadIndex;
    const map = root.object.get("weight_map") orelse return error.BadIndex;
    if (map != .object or map.object.count() == 0) return error.BadIndex;
    for (map.object.values()) |value| {
        if (value != .string or !safeBasename(value.string)) return error.UnsafeShardPath;
    }
    return map.object;
}

fn build(a: std.mem.Allocator, cfg: []const u8, index: []const u8, input_shards: []const Shard) !Plan {
    const geometry = try config.parse(a, cfg);
    const owners = try indexMap(a, index);
    var catalog: std.StringHashMapUnmanaged(Located) = .empty;
    var shard_names: std.StringHashMapUnmanaged(void) = .empty;
    const shards = try a.alloc(ShardIdentity, input_shards.len);
    for (input_shards, 0..) |s, i| {
        if (!safeBasename(s.name)) return error.UnsafeShardPath;
        if ((try shard_names.getOrPut(a, s.name)).found_existing) return error.DuplicateShard;
        const offset = try add64(8, std.math.cast(u64, s.header.len) orelse return error.Overflow);
        if (offset > s.file_bytes) return error.BadHeader;
        const data_len = std.math.cast(usize, s.file_bytes - offset) orelse return error.Overflow;
        const header = try st.parseHeader(a, s.header, data_len);
        const ranges = try a.dupe(st.Entry, header.values());
        std.mem.sort(st.Entry, ranges, {}, struct {
            fn less(_: void, lhs: st.Entry, rhs: st.Entry) bool {
                return lhs.begin < rhs.begin;
            }
        }.less);
        var previous_end: usize = 0;
        for (ranges) |entry| {
            if (entry.begin == entry.end) continue;
            if (entry.begin < previous_end) return error.OverlappingTensors;
            previous_end = entry.end;
        }
        var digest: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(s.header, &digest, .{});
        shards[i] = .{ .name = try a.dupe(u8, s.name), .file_bytes = s.file_bytes, .header_sha256 = digest };
        for (header.keys(), header.values()) |name, entry| {
            const item = try catalog.getOrPut(a, name);
            if (item.found_existing) return error.DuplicateTensor;
            const owner = owners.get(name) orelse return error.IndexOwnershipMismatch;
            if (!std.mem.eql(u8, owner.string, s.name)) return error.IndexOwnershipMismatch;
            item.value_ptr.* = .{ .entry = entry, .shard = i, .offset = try add64(offset, entry.begin) };
        }
    }
    if (catalog.count() != owners.count()) return error.IndexOwnershipMismatch;
    var builder: Builder = .{ .a = a, .catalog = &catalog };
    try builder.tensor("model.language_model.embed_tokens.weight", .embedding, null, &.{ geometry.vocab, geometry.hidden }, .host_rows);
    try builder.tensor("lm_head.weight", .output, null, &.{ geometry.vocab, geometry.hidden }, .host_rows);
    try builder.tensor("model.language_model.norm.weight", .final_norm, null, &.{geometry.hidden}, .zero_centered_norm_f32);
    for (0..geometry.layers) |layer| {
        const pre = try std.fmt.allocPrint(a, "model.language_model.layers.{d}", .{layer});
        inline for (.{ .{ "input_layernorm.weight", Role.input_norm }, .{ "post_attention_layernorm.weight", Role.post_norm } }) |p|
            try builder.tensor(try std.fmt.allocPrint(a, "{s}.{s}", .{ pre, p[0] }), p[1], layer, &.{geometry.hidden}, .zero_centered_norm_f32);
        inline for (.{ .{ "gate_proj", Role.gate, 12288, 4096 }, .{ "up_proj", Role.up, 12288, 4096 }, .{ "down_proj", Role.down, 4096, 12288 } }) |p|
            try builder.tensor(try std.fmt.allocPrint(a, "{s}.mlp.{s}.weight", .{ pre, p[0] }), p[1], layer, &.{ p[2], p[3] }, .bf16_lane_pack);
        if (geometry.linear[layer]) {
            inline for (.{ .{ "in_proj_qkv", Role.qkv, 8192, 4096 }, .{ "in_proj_z", Role.z, 4096, 4096 }, .{ "in_proj_a", Role.a, 32, 4096 }, .{ "in_proj_b", Role.b, 32, 4096 }, .{ "out_proj", Role.linear_out, 4096, 4096 } }) |p|
                try builder.tensor(try std.fmt.allocPrint(a, "{s}.linear_attn.{s}.weight", .{ pre, p[0] }), p[1], layer, &.{ p[2], p[3] }, .bf16_lane_pack);
            try builder.tensor(try std.fmt.allocPrint(a, "{s}.linear_attn.conv1d.weight", .{pre}), .conv, layer, &.{ 8192, 1, 4 }, .raw_f32);
            inline for (.{ .{ "A_log", Role.a_log }, .{ "dt_bias", Role.dt_bias } }) |p|
                try builder.tensor(try std.fmt.allocPrint(a, "{s}.linear_attn.{s}", .{ pre, p[0] }), p[1], layer, &.{32}, .raw_f32);
            try builder.tensor(try std.fmt.allocPrint(a, "{s}.linear_attn.norm.weight", .{pre}), .gdn_norm, layer, &.{128}, .raw_f32);
        } else {
            inline for (.{ .{ "q_proj", Role.q, 8192, 4096 }, .{ "k_proj", Role.k, 1024, 4096 }, .{ "v_proj", Role.v, 1024, 4096 }, .{ "o_proj", Role.o, 4096, 4096 } }) |p|
                try builder.tensor(try std.fmt.allocPrint(a, "{s}.self_attn.{s}.weight", .{ pre, p[0] }), p[1], layer, &.{ p[2], p[3] }, .bf16_lane_pack);
            inline for (.{ .{ "q_norm.weight", Role.q_norm }, .{ "k_norm.weight", Role.k_norm } }) |p|
                try builder.tensor(try std.fmt.allocPrint(a, "{s}.self_attn.{s}", .{ pre, p[0] }), p[1], layer, &.{256}, .zero_centered_norm_f32);
        }
    }
    var visual_count: usize = 0;
    var rest = catalog.iterator();
    while (rest.next()) |kv| {
        if (!std.mem.startsWith(u8, kv.key_ptr.*, "model.visual.")) return error.UnexpectedTensor;
        visual_count = try add(visual_count, 1);
    }
    builder.budget.host_tables = try add(builder.budget.host_embedding, builder.budget.host_output);
    builder.budget.device_weights = try aligned(builder.budget.device_weights, 256);
    builder.budget.device_load_stage = try aligned(builder.budget.device_load_stage, 256);
    builder.budget.host_row_staging = try mul(try mul(4096, geometry.hidden), 4);
    return .{ .arena = undefined, .geometry = geometry, .shards = shards, .bindings = builder.bindings.items, .excluded_visual_count = visual_count, .budget = builder.budget };
}

const Builder = struct {
    a: std.mem.Allocator,
    catalog: *std.StringHashMapUnmanaged(Located),
    bindings: std.ArrayList(Binding) = .empty,
    budget: Bytes = .{},

    fn tensor(b: *Builder, name: []const u8, role: Role, layer: ?usize, shape: []const usize, conversion: Conversion) !void {
        const found = b.catalog.fetchRemove(name) orelse return error.MissingTensor;
        const located = found.value;
        const e = located.entry;
        if (e.dtype != .bf16 or e.rank != shape.len or !std.mem.eql(usize, e.shape[0..e.rank], shape)) return error.UnexpectedTensor;
        var elements: usize = 1;
        for (shape) |dim| {
            _ = std.math.cast(c_int, dim) orelse return error.Overflow;
            elements = try mul(elements, dim);
        }
        const raw = try mul(elements, 2);
        var allocation = raw;
        var device_offset: ?usize = null;
        switch (conversion) {
            .host_rows => switch (role) {
                .embedding => b.budget.host_embedding = raw,
                .output => b.budget.host_output = raw,
                else => unreachable,
            },
            .bf16_lane_pack => {
                allocation = try mul(try mul(try aligned(shape[0], 128), shape[1]), 2);
                b.budget.device_load_stage = @max(b.budget.device_load_stage, raw);
                b.budget.host_conversion_peak = @max(b.budget.host_conversion_peak, try add(raw, allocation));
            },
            .raw_f32, .zero_centered_norm_f32 => {
                allocation = try mul(elements, 4);
                b.budget.host_conversion_peak = @max(b.budget.host_conversion_peak, try add(raw, allocation));
            },
        }
        if (conversion != .host_rows) {
            device_offset = try aligned(b.budget.device_weights, 256);
            b.budget.device_weights = try add(device_offset.?, allocation);
        }
        try b.bindings.append(b.a, .{ .role = role, .layer = if (layer) |l| std.math.cast(u16, l) orelse return error.Overflow else null, .name = try b.a.dupe(u8, name), .shard = located.shard, .file_offset = located.offset, .source = e, .conversion = conversion, .allocation_bytes = allocation, .device_offset = device_offset });
    }
};

fn add(a: usize, b: usize) !usize {
    return std.math.add(usize, a, b);
}
fn add64(a: u64, b: u64) !u64 {
    return std.math.add(u64, a, b);
}
fn mul(a: usize, b: usize) !usize {
    return std.math.mul(usize, a, b);
}
fn aligned(n: usize, alignment: usize) !usize {
    return mul((try add(n, alignment - 1)) / alignment, alignment);
}

const fixture_config = @embedFile("fixtures/clef_flash/config.json");
const fixture_index = @embedFile("fixtures/clef_flash/headers/model.safetensors.index.json");
const fixture_shards = [_]Shard{
    .{ .name = "model-00001-of-00004.safetensors", .header = @embedFile("fixtures/clef_flash/headers/model-00001-of-00004.safetensors.header.json"), .file_bytes = 4942706120 },
    .{ .name = "model-00002-of-00004.safetensors", .header = @embedFile("fixtures/clef_flash/headers/model-00002-of-00004.safetensors.header.json"), .file_bytes = 4987757928 },
    .{ .name = "model-00003-of-00004.safetensors", .header = @embedFile("fixtures/clef_flash/headers/model-00003-of-00004.safetensors.header.json"), .file_bytes = 4954810240 },
    .{ .name = "model-00004-of-00004.safetensors", .header = @embedFile("fixtures/clef_flash/headers/model-00004-of-00004.safetensors.header.json"), .file_bytes = 3934446832 },
};

fn fixturePlan(gpa: std.mem.Allocator, index: []const u8, shards: []const Shard) !Plan {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    errdefer arena.deinit();
    var plan = try build(arena.allocator(), fixture_config, index, shards);
    plan.arena = arena;
    return plan;
}

test "original full metadata produces owned roles and distinct host tables" {
    var plan = try fixturePlan(std.testing.allocator, fixture_index, &fixture_shards);
    defer plan.deinit();
    try std.testing.expectEqual(@as(usize, 427), plan.tensorCount());
    try std.testing.expectEqual(@as(usize, 333), plan.excluded_visual_count);
    const b = plan.bytes();
    try std.testing.expectEqual(@as(usize, 2034237440), b.host_embedding);
    try std.testing.expectEqual(b.host_embedding, b.host_output);
    try std.testing.expectEqual(@as(usize, 4068474880), b.host_tables);
    try std.testing.expectEqual(@as(usize, 100663296), b.device_load_stage);
    try std.testing.expectEqual(@as(usize, 13879009280), b.device_weights);
    try std.testing.expectEqual(@as(usize, 201326592), b.host_conversion_peak);
    try std.testing.expectEqual(@as(usize, 67108864), b.host_row_staging);
    var count_a: usize = 0;
    var end: usize = 0;
    for (plan.bindings) |binding| {
        if (binding.role == .a) {
            count_a += 1;
            try std.testing.expectEqual(@as(usize, 1048576), binding.allocation_bytes);
            try std.testing.expectEqual(Conversion.bf16_lane_pack, binding.conversion);
        }
        if (binding.device_offset) |offset| {
            try std.testing.expect(offset >= end and offset % 256 == 0);
            end = offset + binding.allocation_bytes;
        } else try std.testing.expect(binding.role == .embedding or binding.role == .output);
        try std.testing.expect(binding.file_offset + binding.source.end - binding.source.begin <= plan.shards[binding.shard].file_bytes);
    }
    try std.testing.expectEqual(@as(usize, 24), count_a);
    try std.testing.expectEqual(try aligned(end, 256), b.device_weights);
}

test "metadata fixture hashes and lengths match the immutable capture receipt" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const receipt = try std.json.parseFromSliceLeaky(std.json.Value, arena.allocator(), @embedFile("fixtures/clef_flash/headers/capture.json"), .{});
    const files = receipt.object.get("files").?.object;
    var plan = try fixturePlan(std.testing.allocator, fixture_index, &fixture_shards);
    defer plan.deinit();
    for (fixture_shards, 0..) |s, i| {
        const record = files.get(s.name).?.object;
        try std.testing.expectEqual(@as(u64, @intCast(record.get("file_bytes").?.integer)), s.file_bytes);
        try std.testing.expectEqual(@as(usize, @intCast(record.get("header_bytes").?.integer)), s.header.len);
        var digest: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(s.header, &digest, .{});
        try std.testing.expectEqualSlices(u8, &digest, &plan.shards[i].header_sha256);
        try std.testing.expectEqualStrings(record.get("header_sha256").?.string, &std.fmt.bytesToHex(digest, .lower));
    }
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(fixture_index, &digest, .{});
    try std.testing.expectEqualStrings(receipt.object.get("index_sha256").?.string, &std.fmt.bytesToHex(digest, .lower));
}

fn allocationProbe(gpa: std.mem.Allocator) !void {
    var plan = try fixturePlan(gpa, fixture_index, &fixture_shards);
    defer plan.deinit();
    try std.testing.expectEqual(@as(usize, 427), plan.tensorCount());
}

test "every manifest allocation failure releases owned metadata" {
    try std.testing.checkAllAllocationFailures(std.testing.allocator, allocationProbe, .{});
}

test "checked allocation arithmetic refuses overflow" {
    try std.testing.expectError(error.Overflow, add(std.math.maxInt(usize), 1));
    try std.testing.expectError(error.Overflow, add64(std.math.maxInt(u64), 1));
    try std.testing.expectError(error.Overflow, mul(std.math.maxInt(usize), 2));
    try std.testing.expectError(error.Overflow, aligned(std.math.maxInt(usize), 128));
    try std.testing.expectError(error.Overflow, aligned(std.math.maxInt(usize), 256));
}

test "shard index paths are local safe basenames and JSON duplicates fail closed" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_][]const u8{ "../x.safetensors", "C:x.safetensors", "x/y.safetensors", "x\\y.safetensors", ".x.safetensors", "CON.safetensors", "lpt1.safetensors", "x..safetensors", "x.safetensors:stream", "x.txt", "" }) |name| {
        try std.testing.expect(!safeBasename(name));
    }
    try std.testing.expect(safeBasename("model-00001-of-00004.safetensors"));
    try std.testing.expectError(error.DuplicateField, indexMap(a, "{\"weight_map\":{},\"weight_map\":{}}"));
    try std.testing.expectError(error.DuplicateField, indexMap(a, "{\"weight_map\":{\"x\":\"a.safetensors\",\"x\":\"b.safetensors\"}}"));
    try std.testing.expectError(error.BadIndex, indexMap(a, "{\"weight_map\":{},\"extra\":1}"));
}

test "catalog refuses duplicates overlap truncation wrong ownership and size overflow" {
    const cases = .{
        .{ "{\"x\":{\"dtype\":\"BF16\",\"shape\":[1],\"data_offsets\":[0,2]},\"x\":{\"dtype\":\"BF16\",\"shape\":[1],\"data_offsets\":[0,2]}}", error.DuplicateField },
        .{ "{\"x\":{\"dtype\":\"BF16\",\"shape\":[1],\"data_offsets\":[0,2]},\"y\":{\"dtype\":\"BF16\",\"shape\":[1],\"data_offsets\":[0,2]}}", error.OverlappingTensors },
        .{ "{\"x\":{\"dtype\":\"BF16\",\"shape\":[2],\"data_offsets\":[0,4]}}", error.BadSafetensors },
        .{ "{\"y\":{\"dtype\":\"BF16\",\"shape\":[1],\"data_offsets\":[0,2]}}", error.IndexOwnershipMismatch },
        .{ "{\"x\":{\"dtype\":\"BF16\",\"shape\":[4611686018427387904,4],\"data_offsets\":[0,0]}}", error.BadSafetensors },
    };
    inline for (cases) |case| {
        const shards = [_]Shard{.{ .name = "a.safetensors", .header = case[0], .file_bytes = 8 + case[0].len + 2 }};
        try std.testing.expectError(case[1], fixturePlan(std.testing.allocator, "{\"weight_map\":{\"x\":\"a.safetensors\"}}", &shards));
    }
    const h = "{\"x\":{\"dtype\":\"BF16\",\"shape\":[1],\"data_offsets\":[0,2]}}";
    const s: Shard = .{ .name = "a.safetensors", .header = h, .file_bytes = h.len + 10 };
    const other: Shard = .{ .name = "b.safetensors", .header = h, .file_bytes = h.len + 10 };
    try std.testing.expectError(error.DuplicateTensor, fixturePlan(std.testing.allocator, "{\"weight_map\":{\"x\":\"a.safetensors\"}}", &.{ s, other }));
    try std.testing.expectError(error.IndexOwnershipMismatch, fixturePlan(std.testing.allocator, "{\"weight_map\":{\"x\":\"b.safetensors\"}}", &.{s}));
    try std.testing.expectError(error.IndexOwnershipMismatch, fixturePlan(std.testing.allocator, "{\"weight_map\":{\"x\":\"a.safetensors\",\"y\":\"a.safetensors\"}}", &.{s}));
}

test "full plan refuses dtype shape convolution missing and extra text namespaces" {
    const Mutation = enum { dtype, shape, convolution, missing, extra_text, unknown_root };
    inline for (std.meta.tags(Mutation)) |mutation| {
        var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
        defer arena.deinit();
        const a = arena.allocator();
        var shards = fixture_shards;
        var index = try std.json.parseFromSliceLeaky(std.json.Value, a, fixture_index, .{});
        const target = "model.language_model.layers.0.linear_attn.conv1d.weight";
        var changed = false;
        for (&shards) |*s| {
            var doc = try std.json.parseFromSliceLeaky(std.json.Value, a, s.header, .{});
            const entry = doc.object.getPtr(target) orelse continue;
            switch (mutation) {
                .dtype => try entry.object.put(a, "dtype", .{ .string = "F16" }),
                .shape => {
                    entry.object.getPtr("shape").?.array.items[0].integer = 4096;
                    entry.object.getPtr("shape").?.array.items[2].integer = 8;
                },
                .convolution => {
                    entry.object.getPtr("shape").?.array.items[1].integer = 4;
                    entry.object.getPtr("shape").?.array.items[2].integer = 1;
                },
                .missing => {
                    _ = doc.object.swapRemove(target);
                    _ = index.object.getPtr("weight_map").?.object.swapRemove(target);
                },
                .extra_text, .unknown_root => {
                    const name = if (mutation == .extra_text) "model.language_model.surprise.weight" else "surprise.weight";
                    const tail = s.file_bytes - 8 - s.header.len;
                    const extra_json = try std.fmt.allocPrint(a, "{{\"dtype\":\"BF16\",\"shape\":[1],\"data_offsets\":[{d},{d}]}}", .{ tail, tail + 2 });
                    const extra = try std.json.parseFromSliceLeaky(std.json.Value, a, extra_json, .{});
                    try doc.object.put(a, name, extra);
                    try index.object.getPtr("weight_map").?.object.put(a, name, .{ .string = s.name });
                    s.file_bytes += 2;
                },
            }
            const serialized = try std.json.Stringify.valueAlloc(a, doc, .{});
            s.file_bytes = s.file_bytes - s.header.len + serialized.len;
            s.header = serialized;
            changed = true;
            break;
        }
        try std.testing.expect(changed);
        const index_bytes = try std.json.Stringify.valueAlloc(a, index, .{});
        try std.testing.expectError(if (mutation == .missing) error.MissingTensor else error.UnexpectedTensor, fixturePlan(std.testing.allocator, index_bytes, &shards));
    }
}

test "header reader reads only prefix and bounded metadata and refuses truncation" {
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const a = std.testing.allocator;
    const f = try tmp.dir.createFile(io, "tiny.safetensors", .{ .read = true });
    defer f.close(io);
    const header = "{\"x\":{\"dtype\":\"BF16\",\"shape\":[1],\"data_offsets\":[0,2]}}";
    var prefix: [8]u8 = undefined;
    std.mem.writeInt(u64, &prefix, header.len, .little);
    try f.writePositionalAll(io, &prefix, 0);
    try f.writePositionalAll(io, header, 8);
    try f.writePositionalAll(io, &.{ 0xff, 0xff }, 8 + header.len);
    const shard = try readShard(a, io, f, "tiny.safetensors");
    defer a.free(shard.header);
    try std.testing.expectEqualStrings(header, shard.header);
    try std.testing.expectEqual(@as(u64, header.len + 10), shard.file_bytes);
    try f.setLength(io, 7);
    try std.testing.expectError(error.BadHeader, readShard(a, io, f, "tiny.safetensors"));
    try f.writePositionalAll(io, &prefix, 0);
    try std.testing.expectError(error.BadHeader, readShard(a, io, f, "tiny.safetensors"));
    std.mem.writeInt(u64, &prefix, (16 << 20) + 1, .little);
    try f.writePositionalAll(io, &prefix, 0);
    try std.testing.expectError(error.BadHeader, readShard(a, io, f, "tiny.safetensors"));
    std.mem.writeInt(u64, &prefix, std.math.maxInt(u64), .little);
    try f.writePositionalAll(io, &prefix, 0);
    try std.testing.expectError(error.Overflow, readShard(a, io, f, "tiny.safetensors"));
}

test "public inspector validates real files before refusing an incomplete inventory" {
    const io = std.testing.io;
    const a = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = fixture_config });
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data = "{\"weight_map\":{\"x\":\"tiny.safetensors\"}}" });
    const header = "{\"x\":{\"dtype\":\"BF16\",\"shape\":[1],\"data_offsets\":[0,2]}}";
    var prefix: [8]u8 = undefined;
    std.mem.writeInt(u64, &prefix, header.len, .little);
    const image = try std.mem.concat(a, u8, &.{ &prefix, header, &.{ 0xff, 0xff } });
    defer a.free(image);
    try tmp.dir.writeFile(io, .{ .sub_path = "tiny.safetensors", .data = image });
    const dir = try tmp.dir.realPathFileAlloc(io, ".", a);
    defer a.free(dir);
    try std.testing.expectError(error.MissingTensor, Plan.inspect(a, io, dir));
    try std.testing.checkAllAllocationFailures(a, inspectIncomplete, .{ io, dir });
}

fn inspectIncomplete(a: std.mem.Allocator, io: std.Io, dir: []const u8) !void {
    if (Plan.inspect(a, io, dir)) |result| {
        var plan = result;
        plan.deinit();
        return error.TestExpectedError;
    } else |err| switch (err) {
        error.MissingTensor => {},
        else => return err,
    }
}

const payload_test_names = [_][]const u8{ "input.safetensors", "output.safetensors", "device.safetensors" };
const payload_test_headers = [_][]const u8{
    "{\"input\":{\"dtype\":\"BF16\",\"shape\":[3,2],\"data_offsets\":[0,12]}}",
    "{\"output\":{\"dtype\":\"BF16\",\"shape\":[3,2],\"data_offsets\":[0,12]}}",
    "{\"norm\":{\"dtype\":\"BF16\",\"shape\":[2],\"data_offsets\":[0,4]},\"gate\":{\"dtype\":\"BF16\",\"shape\":[2],\"data_offsets\":[4,8]},\"bias\":{\"dtype\":\"BF16\",\"shape\":[2],\"data_offsets\":[8,12]}}",
};
const payload_test_data = [_][12]u8{
    .{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 },
    .{ 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43 },
    .{ 0x80, 0x3f, 0, 0xc0, 0x81, 0x7f, 0, 0x80, 0xff, 0xff, 0, 0 },
};

fn writePayloadFixture(dir: std.Io.Dir) !void {
    for (payload_test_names, payload_test_headers, payload_test_data) |name, header, data| {
        const file = try dir.createFile(std.testing.io, name, .{});
        defer file.close(std.testing.io);
        var prefix: [8]u8 = undefined;
        std.mem.writeInt(u64, &prefix, header.len, .little);
        try file.writePositionalAll(std.testing.io, &prefix, 0);
        try file.writePositionalAll(std.testing.io, header, 8);
        try file.writePositionalAll(std.testing.io, &data, 8 + header.len);
    }
}

fn payloadTestPlan(gpa: std.mem.Allocator) !Plan {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    errdefer arena.deinit();
    const a = arena.allocator();
    const shards = try a.alloc(ShardIdentity, 3);
    for (payload_test_names, payload_test_headers, 0..) |name, header, i| {
        var digest: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(header, &digest, .{});
        shards[i] = .{ .name = try a.dupe(u8, name), .file_bytes = 8 + header.len + 12, .header_sha256 = digest };
    }
    const bindings = try a.alloc(Binding, 5);
    for ([_]Role{ .embedding, .output, .gate, .final_norm, .dt_bias }, 0..) |role, i| {
        const shard = @min(i, 2);
        const offset: usize = switch (role) {
            .gate => 4,
            .dt_bias => 8,
            else => 0,
        };
        const host = i < 2;
        bindings[i] = .{
            .role = role,
            .layer = if (host) null else 0,
            .name = try a.dupe(u8, @tagName(role)),
            .shard = shard,
            .file_offset = 8 + payload_test_headers[shard].len + offset,
            .source = .{ .dtype = .bf16, .rank = if (host) 2 else 1, .shape = if (host) .{ 3, 2, 1, 1, 1 } else .{ 2, 1, 1, 1, 1 }, .begin = offset, .end = offset + if (host) @as(usize, 12) else 4 },
            .conversion = switch (role) {
                .embedding, .output => .host_rows,
                .gate => .bf16_lane_pack,
                .final_norm => .zero_centered_norm_f32,
                else => .raw_f32,
            },
            .allocation_bytes = if (host) 12 else 8,
            .device_offset = if (host) null else i * 256,
        };
    }
    return .{ .arena = arena, .geometry = undefined, .shards = shards, .bindings = bindings, .excluded_visual_count = 0, .budget = .{ .host_embedding = 12, .host_output = 12, .host_tables = 24 } };
}

const PayloadRecorder = struct {
    count: usize = 0,
    fail_at: ?usize = null,

    pub fn tensor(self: *@This(), binding: anytype, value: st.Tensor) !void {
        const roles = [_]Role{ .gate, .final_norm, .dt_bias };
        const offsets = [_]usize{ 4, 0, 8 };
        if (self.count >= roles.len) return error.UnexpectedCallback;
        try std.testing.expectEqual(roles[self.count], binding.role);
        try std.testing.expectEqual(@as(?u16, 0), binding.layer);
        try std.testing.expectEqualStrings(@tagName(roles[self.count]), binding.name);
        try std.testing.expectEqual(@as(u64, 8 + payload_test_headers[2].len + offsets[self.count]), binding.file_offset);
        try std.testing.expect(value.is(.bf16, &.{2}));
        try std.testing.expectEqualSlices(u8, payload_test_data[2][offsets[self.count]..][0..4], value.bytes);
        const fail = self.fail_at == self.count;
        self.count += 1;
        if (fail) return error.TargetFailed;
    }
};

fn expectPayloadUnmapped(dir: std.Io.Dir, names: []const []const u8) !void {
    for (names) |name| {
        const file = try dir.openFile(std.testing.io, name, .{ .mode = .read_write });
        defer file.close(std.testing.io);
        const len = try file.length(std.testing.io);
        var last: [1]u8 = undefined;
        try std.testing.expectEqual(@as(usize, 1), try file.readPositionalAll(std.testing.io, &last, len - 1));
        // Windows refuses truncation while any section still maps the file.
        try file.setLength(std.testing.io, len - 1);
        try file.setLength(std.testing.io, len);
        try file.writePositionalAll(std.testing.io, &last, len - 1);
    }
}

test "payload callbacks borrow exact BF16 ranges in binding order and host tables outlive plan" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try writePayloadFixture(tmp.dir);
    const dir = try tmp.dir.realPathFileAlloc(io, ".", a);
    defer a.free(dir);
    var plan = try payloadTestPlan(a);
    var recorder: PayloadRecorder = .{};
    var host = plan.upload(a, io, dir, &recorder) catch |err| {
        plan.deinit();
        return err;
    };
    plan.deinit();
    defer host.deinit();
    try std.testing.expectEqual(@as(usize, 3), recorder.count);
    try std.testing.expectEqual(@as(usize, 16 + payload_test_headers[0].len + payload_test_headers[1].len + 24), host.mappedBytes());
    try expectPayloadUnmapped(tmp.dir, payload_test_names[2..]);
    var rows: [12]u8 = undefined;
    try host.gather(.embedding, &.{ 2, 0, 2 }, &rows);
    try std.testing.expectEqualSlices(u8, &.{ 8, 9, 10, 11, 0, 1, 2, 3, 8, 9, 10, 11 }, &rows);
    try host.gather(.output, &.{ 0, 2, 1 }, &rows);
    try std.testing.expectEqualSlices(u8, &.{ 32, 33, 34, 35, 40, 41, 42, 43, 36, 37, 38, 39 }, &rows);
    try host.gather(.embedding, &.{}, &.{});
    @memset(&rows, 0xa5);
    try std.testing.expectError(error.TokenOutOfRange, host.gather(.output, &.{ 0, 1, 3 }, &rows));
    try std.testing.expectError(error.TokenOutOfRange, host.gather(.embedding, &.{ 0, 1, std.math.maxInt(u32) }, &rows));
    try std.testing.expectError(error.BadGatherSize, host.gather(.embedding, &.{0}, &rows));
    const too_many: [4097]u32 = @splat(0);
    try std.testing.expectError(error.TooManyRows, host.gather(.embedding, &too_many, &rows));
    try std.testing.expectEqualSlices(u8, &@as([12]u8, @splat(0xa5)), &rows);
    const max_ids: [4096]u32 = @splat(2);
    var max_rows: [4096 * 4]u8 = undefined;
    try host.gather(.output, &max_ids, &max_rows);
    for (0..4096) |i| try std.testing.expectEqualSlices(u8, &.{ 40, 41, 42, 43 }, max_rows[i * 4 ..][0..4]);
}

test "every shard identity is verified before the first payload callback" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try tmp.dir.realPathFileAlloc(io, ".", a);
    defer a.free(dir);
    var plan = try payloadTestPlan(a);
    defer plan.deinit();
    for (0..6) |mutation| {
        try writePayloadFixture(tmp.dir);
        const name = payload_test_names[2];
        if (mutation == 5) {
            try tmp.dir.deleteFile(io, name);
        } else {
            const file = try tmp.dir.openFile(io, name, .{ .mode = .read_write });
            defer file.close(io);
            switch (mutation) {
                0 => try file.writePositionalAll(io, "!", 8),
                1 => try file.setLength(io, plan.shards[2].file_bytes - 1),
                2, 3, 4 => {
                    var prefix: [8]u8 = undefined;
                    const len: u64 = switch (mutation) {
                        2 => std.math.maxInt(u64),
                        3 => 16 << 20 | 1,
                        else => plan.shards[2].file_bytes,
                    };
                    std.mem.writeInt(u64, &prefix, len, .little);
                    try file.writePositionalAll(io, &prefix, 0);
                },
                else => unreachable,
            }
        }
        var recorder: PayloadRecorder = .{};
        const result = plan.upload(a, io, dir, &recorder);
        const expected: anyerror = switch (mutation) {
            0, 1 => error.CheckpointChanged,
            5 => error.FileNotFound,
            else => error.BadHeader,
        };
        try std.testing.expectError(expected, result);
        try std.testing.expectEqual(@as(usize, 0), recorder.count);
        try expectPayloadUnmapped(tmp.dir, payload_test_names[0..if (mutation == 5) @as(usize, 2) else 3]);
    }
}

fn payloadAllocationCase(a: std.mem.Allocator, plan: *const Plan, dir: []const u8, fixture_dir: std.Io.Dir) !void {
    var recorder: PayloadRecorder = .{};
    const result = plan.upload(a, std.testing.io, dir, &recorder);
    if (result) |value| {
        var host = value;
        host.deinit();
    } else |err| {
        try expectPayloadUnmapped(fixture_dir, &payload_test_names);
        return err;
    }
    try expectPayloadUnmapped(fixture_dir, &payload_test_names);
}

test "payload callback and allocation failures release every mapping and file" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try tmp.dir.realPathFileAlloc(io, ".", a);
    defer a.free(dir);
    var plan = try payloadTestPlan(a);
    defer plan.deinit();
    for (0..3) |fail_at| {
        try writePayloadFixture(tmp.dir);
        var recorder: PayloadRecorder = .{ .fail_at = fail_at };
        try std.testing.expectError(error.TargetFailed, plan.upload(a, io, dir, &recorder));
        try std.testing.expectEqual(fail_at + 1, recorder.count);
        try expectPayloadUnmapped(tmp.dir, &payload_test_names);
    }
    try writePayloadFixture(tmp.dir);
    try std.testing.checkAllAllocationFailures(a, payloadAllocationCase, .{ &plan, dir, tmp.dir });
}
