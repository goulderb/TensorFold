//! Clef's joint schema head checkpoint: one role walk over joint_head_config.json names every tensor, checks it
//! against the safetensors header and places it on the device. The roles follow the module tree of Cloudflare's
//! clef-flash joint_schema_model.py (Apache-2.0, revision fde727a2); this is a reimplementation, no code is copied.
const std = @import("std");
const core = @import("core");
const cuda = @import("cuda");
const st = core.safetensors;

pub const Config = struct {
    hidden_size: u32,
    width: u32,
    routing_layers: u32,
    layers: u32,
    heads: u32,
    feedforward: u32,
};

/// The only admitted geometry: the kernels assume head dim 64, a 4096-wide hidden norm and these layer counts.
pub const pinned: Config = .{ .hidden_size = 4096, .width = 1024, .routing_layers = 2, .layers = 4, .heads = 16, .feedforward = 4096 };
pub const tensor_count = 122;

pub fn parseConfig(a: std.mem.Allocator, bytes: []const u8) !Config {
    const root = try std.json.parseFromSliceLeaky(std.json.Value, a, bytes, .{});
    const names = @typeInfo(Config).@"struct".field_names;
    if (root != .object or root.object.count() != names.len) return error.BadHeadConfig;
    var c: Config = undefined;
    inline for (names) |name| {
        const v = root.object.get(name) orelse return error.BadHeadConfig;
        if (v != .integer) return error.BadHeadConfig;
        @field(c, name) = std.math.cast(u32, v.integer) orelse return error.BadHeadConfig;
    }
    if (!std.meta.eql(c, pinned)) return error.UnsupportedHeadGeometry;
    return c;
}

/// Device offsets into the head's weight arena. Projections are packed BF16 lane weights; the rest are stored BF16.
pub const Norm = struct { weight: usize, bias: usize };
pub const Linear = struct { weight: usize, n: u32, k: u32, bias: ?usize };
pub const Attention = struct { q: Linear, k: Linear, v: Linear, out: Linear };
pub const Routing = struct { query_norm: Norm, memory_norm: Norm, attention: Attention, feedforward_norm: Norm, up: Linear, down: Linear };
pub const Decoder = struct { norm1: Norm, self_attention: Attention, norm2: Norm, cross_attention: Attention, norm3: Norm, up: Linear, down: Linear };

pub const Layout = struct {
    hidden_norm: Norm,
    memory: Linear,
    question: Linear,
    option_question: Linear,
    global: Linear,
    option_context: Linear,
    option_lexical: Linear,
    type_embedding: usize,
    routing: [pinned.routing_layers]Routing,
    option_summary_norm: Norm,
    decoders: [pinned.layers]Decoder,
    field_norm: Norm,
    option_norm: Norm,
    scorer_up: Linear,
    scorer_down: Linear,
    prior_logit_scale: usize,
    joint_logit_scale: usize,
    residual_gate: usize,
};

/// `raw`: the stored BF16 bytes. `lane`: one packed projection. `lane_qkv`: a packed in_proj split into three lanes.
pub const Placement = enum { raw, lane, lane_qkv };

pub const Binding = struct {
    name: []const u8,
    entry: st.Entry,
    placement: Placement,
    device_offset: usize,
    device_bytes: usize,
};

pub const Bytes = struct {
    device_weights: usize = 0,
    /// The largest stored projection, staged on the device before packing; freed after load.
    load_stage: usize = 0,
    file_bytes: u64 = 0,
    header_bytes: usize = 0,
};

pub const Plan = struct {
    arena: std.heap.ArenaAllocator,
    config: Config,
    bindings: []const Binding,
    layout: Layout,
    bytes: Bytes,
    header_sha256: [32]u8,

    /// Reads joint_head_config.json and the bounded header of joint_head.safetensors; no payload.
    pub fn inspect(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !Plan {
        var arena: std.heap.ArenaAllocator = .init(gpa);
        errdefer arena.deinit();
        const a = arena.allocator();
        const config_bytes = try std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(a, &.{ dir, "joint_head_config.json" }), a, .limited(64 << 10));
        const f = try std.Io.Dir.cwd().openFile(io, try std.fs.path.join(a, &.{ dir, "joint_head.safetensors" }), .{});
        defer f.close(io);
        const file_bytes = try f.length(io);
        var prefix: [8]u8 = undefined;
        if (try f.readPositionalAll(io, &prefix, 0) != 8) return error.BadHeader;
        const len = std.mem.readInt(u64, &prefix, .little);
        if (len > max_header or len > file_bytes - @min(file_bytes, 8)) return error.BadHeader;
        const header = try a.alloc(u8, @intCast(len));
        if (try f.readPositionalAll(io, header, 8) != header.len) return error.BadHeader;
        var plan = try build(a, config_bytes, header, file_bytes);
        plan.arena = arena;
        return plan;
    }

    pub fn deinit(self: *Plan) void {
        self.arena.deinit();
        self.* = undefined;
    }

    /// Maps the file, hashes the whole mapping against `pin` and rechecks the header before the first callback,
    /// then hands `target.tensor(binding, bytes)` each binding's stored bytes. The bytes are borrowed for the call.
    pub fn upload(self: *const Plan, io: std.Io, path: []const u8, pin: [32]u8, target: anytype) !void {
        const file = try std.Io.Dir.cwd().openFile(io, path, .{});
        defer file.close(io);
        if (try file.length(io) != self.bytes.file_bytes) return error.CheckpointChanged;
        const len = std.math.cast(usize, self.bytes.file_bytes) orelse return error.Overflow;
        var map = try std.Io.File.MemoryMap.create(io, file, .{ .len = len, .protection = .{ .read = true, .write = false }, .populate = false });
        defer map.destroy(io);
        var digest: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(map.memory, &digest, .{});
        if (!std.mem.eql(u8, &digest, &pin)) return error.HeadDigestMismatch;
        const memory = map.memory;
        if (std.mem.readInt(u64, memory[0..8], .little) != self.bytes.header_bytes) return error.CheckpointChanged;
        std.crypto.hash.sha2.Sha256.hash(memory[8..][0..self.bytes.header_bytes], &digest, .{});
        if (!std.mem.eql(u8, &digest, &self.header_sha256)) return error.CheckpointChanged;
        const data = 8 + self.bytes.header_bytes;
        for (self.bindings) |b| try target.tensor(b, memory[data + b.entry.begin .. data + b.entry.end]);
    }
};

const max_header = 1 << 20;

pub fn build(a: std.mem.Allocator, config_bytes: []const u8, header: []const u8, file_bytes: u64) !Plan {
    const config = try parseConfig(a, config_bytes);
    if (header.len > max_header) return error.BadHeader;
    const data_offset = try std.math.add(u64, 8, header.len);
    if (data_offset > file_bytes) return error.BadHeader;
    const data_len = std.math.cast(usize, file_bytes - data_offset) orelse return error.Overflow;
    var catalog = try st.parseHeader(a, header, data_len);
    var w: Walk = .{ .a = a, .catalog = &catalog };
    const layout = try w.head(config);
    if (catalog.count() != 0) return error.UnexpectedTensor;
    if (w.bindings.items.len != tensor_count) return error.UnexpectedTensor;
    const ranges = try a.alloc(st.Entry, w.bindings.items.len);
    for (ranges, w.bindings.items) |*r, b| r.* = b.entry;
    std.mem.sort(st.Entry, ranges, {}, struct {
        fn less(_: void, lhs: st.Entry, rhs: st.Entry) bool {
            return lhs.begin < rhs.begin;
        }
    }.less);
    var covered: usize = 0;
    for (ranges) |e| {
        if (e.begin < covered) return error.OverlappingTensors;
        if (e.begin != covered) return error.UnaccountedPayload;
        covered = e.end;
    }
    if (covered != data_len) return error.UnaccountedPayload;
    w.bytes.device_weights = try aligned(w.bytes.device_weights);
    w.bytes.file_bytes = file_bytes;
    w.bytes.header_bytes = header.len;
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(header, &digest, .{});
    return .{ .arena = undefined, .config = config, .bindings = w.bindings.items, .layout = layout, .bytes = w.bytes, .header_sha256 = digest };
}

/// The role walk. Each step takes its tensor out of the header catalog, so a leftover is an unexpected tensor.
const Walk = struct {
    a: std.mem.Allocator,
    catalog: *st.Header,
    bindings: std.ArrayList(Binding) = .empty,
    bytes: Bytes = .{},

    fn head(w: *Walk, c: Config) !Layout {
        var l: Layout = undefined;
        const h = c.hidden_size;
        const d = c.width;
        l.hidden_norm = try w.norm("hidden_norm", h);
        inline for (.{ "memory", "question", "option_question", "global", "option_context", "option_lexical" }) |name|
            @field(l, name) = try w.linear(name ++ "_projection", d, h, false);
        l.type_embedding = try w.raw("type_embedding.weight", &.{ 3, d }, .raw);
        for (&l.routing, 0..) |*r, i| {
            const p = try std.fmt.allocPrint(w.a, "evidence_layers.{d}", .{i});
            r.query_norm = try w.norm(try w.join(p, "query_norm"), d);
            r.memory_norm = try w.norm(try w.join(p, "memory_norm"), d);
            r.attention = try w.attention(try w.join(p, "attention"), d);
            r.feedforward_norm = try w.norm(try w.join(p, "feedforward_norm"), d);
            r.up = try w.linear(try w.join(p, "feedforward.0"), c.feedforward, d, true);
            r.down = try w.linear(try w.join(p, "feedforward.3"), d, c.feedforward, true);
        }
        l.option_summary_norm = try w.norm("option_summary_norm", d);
        for (&l.decoders, 0..) |*x, i| {
            const p = try std.fmt.allocPrint(w.a, "layers.{d}", .{i});
            x.norm1 = try w.norm(try w.join(p, "norm1"), d);
            x.self_attention = try w.attention(try w.join(p, "self_attn"), d);
            x.norm2 = try w.norm(try w.join(p, "norm2"), d);
            x.cross_attention = try w.attention(try w.join(p, "multihead_attn"), d);
            x.norm3 = try w.norm(try w.join(p, "norm3"), d);
            x.up = try w.linear(try w.join(p, "linear1"), c.feedforward, d, true);
            x.down = try w.linear(try w.join(p, "linear2"), d, c.feedforward, true);
        }
        l.field_norm = try w.norm("field_norm", d);
        l.option_norm = try w.norm("option_norm", d);
        l.scorer_up = try w.linear("residual_scorer.0", d, 4 * d, true);
        l.scorer_down = try w.linear("residual_scorer.3", 1, d, true);
        inline for (.{ "prior_logit_scale", "joint_logit_scale", "residual_gate" }) |name|
            @field(l, name) = try w.raw(name, &.{}, .raw);
        return l;
    }

    fn join(w: *Walk, prefix: []const u8, name: []const u8) ![]const u8 {
        return std.fmt.allocPrint(w.a, "{s}.{s}", .{ prefix, name });
    }

    fn norm(w: *Walk, prefix: []const u8, width: u32) !Norm {
        return .{ .weight = try w.raw(try w.join(prefix, "weight"), &.{width}, .raw), .bias = try w.raw(try w.join(prefix, "bias"), &.{width}, .raw) };
    }

    fn linear(w: *Walk, prefix: []const u8, n: u32, k: u32, bias: bool) !Linear {
        return .{
            .weight = try w.raw(try w.join(prefix, "weight"), &.{ n, k }, .lane),
            .n = n,
            .k = k,
            .bias = if (bias) try w.raw(try w.join(prefix, "bias"), &.{n}, .raw) else null,
        };
    }

    /// torch.nn.MultiheadAttention's packed in_proj (q, k, v rows in that order) and its out_proj.
    fn attention(w: *Walk, prefix: []const u8, d: u32) !Attention {
        const weight = try w.raw(try w.join(prefix, "in_proj_weight"), &.{ 3 * d, d }, .lane_qkv);
        const bias = try w.raw(try w.join(prefix, "in_proj_bias"), &.{3 * d}, .raw);
        const part = laneBytes(d, d);
        var a: Attention = undefined;
        inline for (.{ "q", "k", "v" }, 0..) |name, i|
            @field(a, name) = .{ .weight = weight + i * part, .n = d, .k = d, .bias = bias + i * @as(usize, d) * 2 };
        a.out = try w.linear(try w.join(prefix, "out_proj"), d, d, true);
        return a;
    }

    fn raw(w: *Walk, name: []const u8, shape: []const u32, placement: Placement) !usize {
        const found = w.catalog.fetchSwapRemove(name) orelse return error.MissingTensor;
        const e = found.value;
        if (e.dtype != .bf16 or e.rank != shape.len) return error.UnexpectedTensor;
        for (shape, e.shape[0..e.rank]) |want, got| if (want != got) return error.UnexpectedTensor;
        const stored = e.end - e.begin;
        const allocation = switch (placement) {
            .raw => stored,
            .lane => laneBytes(shape[0], shape[1]),
            .lane_qkv => 3 * laneBytes(shape[0] / 3, shape[1]),
        };
        if (placement != .raw) w.bytes.load_stage = @max(w.bytes.load_stage, stored);
        const offset = try aligned(w.bytes.device_weights);
        w.bytes.device_weights = try std.math.add(usize, offset, allocation);
        try w.bindings.append(w.a, .{ .name = name, .entry = e, .placement = placement, .device_offset = offset, .device_bytes = allocation });
        return offset;
    }
};

/// Bytes of one packed BF16 lane weight: rows padded to whole 128-row blocks (modelopt.Packer.bf16Lane's layout).
pub fn laneBytes(n: usize, k: usize) usize {
    return cuda.modelopt.padded(n) * k * 2;
}

fn aligned(n: usize) !usize {
    return std.math.mul(usize, (try std.math.add(usize, n, 255)) / 256, 256);
}

const fixture_config = @embedFile("fixtures/joint_head_config.json");
const fixture_header = @embedFile("fixtures/joint_head.safetensors.header.json");
const fixture_capture = @embedFile("fixtures/capture.json");

fn captureInt(a: std.mem.Allocator, name: []const u8) !u64 {
    const v = try std.json.parseFromSliceLeaky(std.json.Value, a, fixture_capture, .{});
    return @intCast(v.object.get(name).?.integer);
}

fn fixturePlan(gpa: std.mem.Allocator, config: []const u8, header: []const u8, file_bytes: u64) !Plan {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    errdefer arena.deinit();
    var plan = try build(arena.allocator(), config, header, file_bytes);
    plan.arena = arena;
    return plan;
}

fn fixtureFileBytes() !u64 {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    return captureInt(arena.allocator(), "file_bytes");
}

test "the pinned head header yields every role the walk derives, with no tensor left over" {
    var plan = try fixturePlan(std.testing.allocator, fixture_config, fixture_header, try fixtureFileBytes());
    defer plan.deinit();
    try std.testing.expectEqual(@as(usize, 122), plan.bindings.len);
    var names: std.StringHashMapUnmanaged(void) = .empty;
    defer names.deinit(std.testing.allocator);
    var lanes: usize = 0;
    var split: usize = 0;
    var end: usize = 0;
    for (plan.bindings) |b| {
        try std.testing.expect(!(try names.getOrPut(std.testing.allocator, b.name)).found_existing);
        try std.testing.expect(b.device_offset % 256 == 0 and b.device_offset >= end);
        end = b.device_offset + b.device_bytes;
        switch (b.placement) {
            .lane => lanes += 1,
            .lane_qkv => split += 1,
            .raw => try std.testing.expectEqual(b.entry.end - b.entry.begin, b.device_bytes),
        }
    }
    try std.testing.expectEqual(@as(usize, 6 + 2 * 3 + 4 * 4 + 2), lanes);
    try std.testing.expectEqual(@as(usize, 2 + 4 * 2), split);
    try std.testing.expectEqual(try aligned(end), plan.bytes.device_weights);
    try std.testing.expectEqual(@as(usize, 243_786_752), plan.bytes.device_weights);
    try std.testing.expectEqual(@as(usize, 8 << 20), plan.bytes.load_stage);
    const l = plan.layout;
    try std.testing.expectEqual(@as(u32, 1), l.scorer_down.n);
    try std.testing.expectEqual(@as(u32, 4096), l.scorer_up.k);
    const qkv = l.decoders[3].cross_attention;
    try std.testing.expectEqual(qkv.q.weight + laneBytes(1024, 1024), qkv.k.weight);
    try std.testing.expectEqual(qkv.q.bias.? + 4096, qkv.v.bias.?);
    try std.testing.expect(l.memory.bias == null and l.routing[1].up.bias != null);
}

test "the header fixture matches the immutable capture receipt" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const v = try std.json.parseFromSliceLeaky(std.json.Value, a, fixture_capture, .{});
    try std.testing.expectEqualStrings("fde727a287004204b7518dcc983fe64379776712", v.object.get("revision").?.string);
    try std.testing.expectEqual(@as(i64, fixture_header.len), v.object.get("header_bytes").?.integer);
    try std.testing.expectEqual(@as(i64, tensor_count), v.object.get("tensors").?.integer);
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(fixture_header, &digest, .{});
    try std.testing.expectEqualStrings(v.object.get("header_sha256").?.string, &std.fmt.bytesToHex(digest, .lower));
    std.crypto.hash.sha2.Sha256.hash(fixture_config, &digest, .{});
    try std.testing.expectEqualStrings(v.object.get("config_sha256").?.string, &std.fmt.bytesToHex(digest, .lower));
}

test "config admission accepts only the pinned geometry with exactly its six integer fields" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try std.testing.expectEqual(pinned, try parseConfig(a, fixture_config));
    const bad = [_]struct { []const u8, anyerror }{
        .{ "{\"hidden_size\":4096,\"width\":1024,\"routing_layers\":2,\"layers\":4,\"heads\":16}", error.BadHeadConfig },
        .{ "{\"hidden_size\":4096,\"width\":1024,\"routing_layers\":2,\"layers\":4,\"heads\":16,\"feedforward\":4096,\"dropout\":0}", error.BadHeadConfig },
        .{ "{\"hidden_size\":4096,\"width\":1024,\"routing_layers\":2,\"layers\":4,\"heads\":16,\"feedforward\":4096.0}", error.BadHeadConfig },
        .{ "{\"hidden_size\":4096,\"width\":1024,\"routing_layers\":2,\"layers\":4,\"heads\":8,\"feedforward\":4096}", error.UnsupportedHeadGeometry },
        .{ "{\"hidden_size\":4096,\"width\":1024,\"routing_layers\":2,\"layers\":4,\"heads\":-16,\"feedforward\":4096}", error.BadHeadConfig },
        .{ "[]", error.BadHeadConfig },
        .{ "{\"hidden_size\":4096,\"hidden_size\":4096,\"width\":1024,\"routing_layers\":2,\"layers\":4,\"heads\":16}", error.DuplicateField },
    };
    for (bad) |case| try std.testing.expectError(case[1], parseConfig(a, case[0]));
}

const Mutation = enum { missing, extra, misshaped, wrong_dtype, scalar_rank, duplicate, overlap, gap, trailing };

fn mutated(a: std.mem.Allocator, m: Mutation) !struct { header: []const u8, file_bytes: u64 } {
    var doc = try std.json.parseFromSliceLeaky(std.json.Value, a, fixture_header, .{});
    var file_bytes: u64 = 8 + fixture_header.len + 243_525_640;
    const target = "layers.2.norm3.bias";
    const entry = doc.object.getPtr(target).?;
    switch (m) {
        .missing => _ = doc.object.orderedRemove(target),
        .extra => {
            const tail: i64 = 243_525_640;
            try doc.object.put(a, "layers.2.norm4.bias", try std.json.parseFromSliceLeaky(std.json.Value, a, try std.fmt.allocPrint(a, "{{\"dtype\":\"BF16\",\"shape\":[1],\"data_offsets\":[{d},{d}]}}", .{ tail, tail + 2 }), .{}));
            file_bytes += 2;
        },
        .misshaped => {
            entry.object.getPtr("shape").?.array.items[0].integer = 512;
            try entry.object.getPtr("shape").?.array.append(.{ .integer = 2 });
        },
        .wrong_dtype => try entry.object.put(a, "dtype", .{ .string = "F16" }),
        .scalar_rank => {
            const s = doc.object.getPtr("residual_gate").?;
            try s.object.getPtr("shape").?.array.append(.{ .integer = 1 });
        },
        .duplicate => {},
        .overlap => {
            const offsets = entry.object.getPtr("data_offsets").?.array.items;
            offsets[0].integer -= 2;
            offsets[1].integer -= 2;
        },
        .gap => {
            const offsets = doc.object.getPtr("type_embedding.weight").?.object.getPtr("data_offsets").?.array.items;
            offsets[0].integer += 2;
            offsets[1].integer += 2;
            file_bytes += 2;
        },
        .trailing => file_bytes += 2,
    }
    var header = try std.json.Stringify.valueAlloc(a, doc, .{});
    if (m == .duplicate) {
        const tail = std.mem.lastIndexOfScalar(u8, header, '}').?;
        header = try std.fmt.allocPrint(a, "{s},\"residual_gate\":{{\"dtype\":\"BF16\",\"shape\":[],\"data_offsets\":[235126788,235126790]}}}}", .{header[0..tail]});
    }
    file_bytes = file_bytes - fixture_header.len + header.len;
    return .{ .header = header, .file_bytes = file_bytes };
}

test "missing extra misshaped retyped duplicated overlapping gapped and trailing tensors are refused" {
    const expected = [_]struct { Mutation, anyerror }{
        .{ .missing, error.MissingTensor },
        .{ .extra, error.UnexpectedTensor },
        .{ .misshaped, error.UnexpectedTensor },
        .{ .wrong_dtype, error.UnexpectedTensor },
        .{ .scalar_rank, error.UnexpectedTensor },
        .{ .duplicate, error.DuplicateField },
        .{ .overlap, error.OverlappingTensors },
        .{ .gap, error.UnaccountedPayload },
        .{ .trailing, error.UnaccountedPayload },
    };
    for (expected) |case| {
        var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
        defer arena.deinit();
        const m = try mutated(arena.allocator(), case[0]);
        std.testing.expectError(case[1], fixturePlan(std.testing.allocator, fixture_config, m.header, m.file_bytes)) catch |err| {
            std.debug.print("mutation {s}\n", .{@tagName(case[0])});
            return err;
        };
    }
    try std.testing.expectError(error.BadHeader, fixturePlan(std.testing.allocator, fixture_config, fixture_header, fixture_header.len + 7));
    try std.testing.expectError(error.BadSafetensors, fixturePlan(std.testing.allocator, fixture_config, fixture_header, fixture_header.len + 8 + 100));
}

fn allocationProbe(gpa: std.mem.Allocator, file_bytes: u64) !void {
    var plan = try fixturePlan(gpa, fixture_config, fixture_header, file_bytes);
    plan.deinit();
}

test "every plan allocation failure releases owned metadata" {
    try std.testing.checkAllAllocationFailures(std.testing.allocator, allocationProbe, .{try fixtureFileBytes()});
}

const Recorder = struct {
    calls: usize = 0,
    first: [4]u8 = undefined,
    last_name: []const u8 = "",
    last_len: usize = 0,

    pub fn tensor(r: *Recorder, b: Binding, bytes: []const u8) !void {
        if (r.calls == 0) @memcpy(&r.first, bytes[0..4]);
        r.calls += 1;
        r.last_name = b.name;
        r.last_len = bytes.len;
        try std.testing.expectEqual(b.entry.end - b.entry.begin, bytes.len);
    }
};

/// A two-tensor head-shaped file is too small for the walk, so upload tests use a plan whose header is real but tiny.
fn tinyPlan(a: std.mem.Allocator, header: []const u8, payload: usize) !Plan {
    var arena: std.heap.ArenaAllocator = .init(a);
    errdefer arena.deinit();
    const aa = arena.allocator();
    const catalog = try st.parseHeader(aa, header, payload);
    const bindings = try aa.alloc(Binding, catalog.count());
    for (catalog.keys(), catalog.values(), bindings) |name, e, *b| b.* = .{ .name = name, .entry = e, .placement = .raw, .device_offset = 0, .device_bytes = e.end - e.begin };
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(header, &digest, .{});
    return .{ .arena = arena, .config = pinned, .bindings = bindings, .layout = undefined, .bytes = .{ .file_bytes = 8 + header.len + payload, .header_bytes = header.len }, .header_sha256 = digest };
}

test "upload refuses a digest, length or header change before the first callback, then lends exact ranges" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const header = "{\"a\":{\"dtype\":\"BF16\",\"shape\":[2],\"data_offsets\":[0,4]},\"b\":{\"dtype\":\"BF16\",\"shape\":[],\"data_offsets\":[4,6]}}";
    var prefix: [8]u8 = undefined;
    std.mem.writeInt(u64, &prefix, header.len, .little);
    const image = try std.mem.concat(a, u8, &.{ &prefix, header, &.{ 1, 2, 3, 4, 5, 6 } });
    defer a.free(image);
    var pin: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(image, &pin, .{});
    var plan = try tinyPlan(a, header, 6);
    defer plan.deinit();
    const dir = try tmp.dir.realPathFileAlloc(io, ".", a);
    defer a.free(dir);
    const path = try std.fs.path.join(a, &.{ dir, "joint_head.safetensors" });
    defer a.free(path);

    try tmp.dir.writeFile(io, .{ .sub_path = "joint_head.safetensors", .data = image });
    var wrong = pin;
    wrong[0] ^= 1;
    var r: Recorder = .{};
    try std.testing.expectError(error.HeadDigestMismatch, plan.upload(io, path, wrong, &r));
    try std.testing.expectEqual(@as(usize, 0), r.calls);

    const longer = try std.mem.concat(a, u8, &.{ image, &.{0} });
    defer a.free(longer);
    try tmp.dir.writeFile(io, .{ .sub_path = "joint_head.safetensors", .data = longer });
    try std.testing.expectError(error.CheckpointChanged, plan.upload(io, path, pin, &r));

    const changed = try a.dupe(u8, image);
    defer a.free(changed);
    changed[8 + 2] = 'c';
    var changed_pin: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(changed, &changed_pin, .{});
    try tmp.dir.writeFile(io, .{ .sub_path = "joint_head.safetensors", .data = changed });
    try std.testing.expectError(error.CheckpointChanged, plan.upload(io, path, changed_pin, &r));
    try std.testing.expectEqual(@as(usize, 0), r.calls);

    try tmp.dir.writeFile(io, .{ .sub_path = "joint_head.safetensors", .data = image });
    try plan.upload(io, path, pin, &r);
    try std.testing.expectEqual(@as(usize, 2), r.calls);
    try std.testing.expectEqualSlices(u8, &.{ 1, 2, 3, 4 }, &r.first);
    try std.testing.expectEqualStrings("b", r.last_name);
    try std.testing.expectEqual(@as(usize, 2), r.last_len);
}

test "public inspector reads only the config and the bounded header" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "joint_head_config.json", .data = fixture_config });
    const f = try tmp.dir.createFile(io, "joint_head.safetensors", .{ .read = true });
    defer f.close(io);
    var prefix: [8]u8 = undefined;
    std.mem.writeInt(u64, &prefix, fixture_header.len, .little);
    try f.writePositionalAll(io, &prefix, 0);
    try f.writePositionalAll(io, fixture_header, 8);
    try f.setLength(io, try fixtureFileBytes());
    const dir = try tmp.dir.realPathFileAlloc(io, ".", a);
    defer a.free(dir);
    var plan = try Plan.inspect(a, io, dir);
    defer plan.deinit();
    try std.testing.expectEqual(@as(usize, 122), plan.bindings.len);
    std.mem.writeInt(u64, &prefix, max_header + 1, .little);
    try f.writePositionalAll(io, &prefix, 0);
    try std.testing.expectError(error.BadHeader, Plan.inspect(a, io, dir));
}
