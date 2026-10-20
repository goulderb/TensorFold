//! Matched full-row terminal ownership component; adapted from qwen35_terminal_graph_screen.zig.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const qwen = @import("qwen35_cuda");
const lanes = @import("lanes");
const native = @import("qwen35_adapter");
const Engine = qwen.shared_engine.Engine;
const Adapter = native.Adapter(qwen.Session);
const Pick = cuda.tree_round.Pick;
const rows = 128;
const vocab = 248320;
const row_bytes = vocab * 2;
const guard = 64;
const Manifest = struct { version: usize, rows: usize, vocab: usize, bytes: usize, path: []const u8, sha256: []const u8, row_sha256: []const []const u8, tokens: []const u32, tree_image_sha256: []const u8, argmax_sha256: []const u8 };
const Arm = enum { host, device };

fn hash(bytes: []const u8) [64]u8 {
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &digest, .{});
    return std.fmt.bytesToHex(digest, .lower);
}
fn verify(bytes: []const u8, expected: []const u8) !void {
    if (!std.mem.eql(u8, &hash(bytes), expected)) return error.TerminalGraphHashMismatch;
}
pub fn validatePaths(dir: []const u8, sha: []const u8) !void {
    if (!std.fs.path.isAbsolute(dir) or sha.len != 64) return error.InvalidTerminalGraphInput;
    for (sha) |c| if (!std.ascii.isHex(c)) return error.InvalidTerminalGraphInput;
}
fn validateManifest(m: Manifest) !void {
    if (m.version != 1 or m.rows != rows or m.vocab != vocab or m.bytes != rows * row_bytes or !std.fs.path.isAbsolute(m.path) or m.tokens.len != rows or m.row_sha256.len != rows) return error.InvalidTerminalGraphFixture;
    for (m.tokens) |token| if (token >= vocab) return error.InvalidTerminalGraphFixture;
    for ([_][]const u8{ m.sha256, m.tree_image_sha256, m.argmax_sha256 }) |sha| {
        if (sha.len != 64) return error.InvalidTerminalGraphFixture;
        for (sha) |c| if (!std.ascii.isHex(c)) return error.InvalidTerminalGraphFixture;
    }
    for (m.row_sha256) |sha| {
        if (sha.len != 64) return error.InvalidTerminalGraphFixture;
        for (sha) |c| if (!std.ascii.isHex(c)) return error.InvalidTerminalGraphFixture;
    }
}
fn argmax(row: []const u16) !u32 {
    var best: usize = 0;
    var best_value: f32 = -std.math.inf(f32);
    for (row, 0..) |bits, i| {
        const value: f32 = @bitCast(@as(u32, bits) << 16);
        if (!std.math.isFinite(value)) return error.NonfiniteLogits;
        if (value > best_value) {
            best = i;
            best_value = value;
        }
    }
    return @intCast(best);
}
fn validatePick(pick: Pick, n: usize) !u32 {
    if (pick.nonfinite & ~@as(u32, 7) != 0) return error.InvalidTerminalPick;
    if (pick.nonfinite != 0) {
        if (pick.token != std.math.maxInt(u32)) return error.InvalidTerminalPick;
        return error.NonfiniteLogits;
    }
    if (pick.token >= n) return error.InvalidTerminalPick;
    return pick.token;
}
fn canaries(bytes: []const u8, active: usize) !void {
    if (bytes.len != active + guard * 2) return error.InvalidTerminalGuard;
    for (bytes[0..guard]) |value| if (value != 0xa5) return error.TerminalGuardChanged;
    for (bytes[guard + active ..]) |value| if (value != 0xa5) return error.TerminalGuardChanged;
}
fn save(gpu: check.Gpu, journal: anytype, name: []const u8, bytes: []const u8) !void {
    const filename = try std.fmt.allocPrint(gpu.gpa, "terminal_ownership_gpu{d}_{s}.bin", .{ gpu.ctx.device_ordinal, name });
    defer gpu.gpa.free(filename);
    const path = try std.fs.path.join(gpu.gpa, &.{ journal.directory, filename });
    defer gpu.gpa.free(path);
    const file = try std.Io.Dir.cwd().createFile(gpu.io, path, .{ .exclusive = true });
    defer file.close(gpu.io);
    try file.writeStreamingAll(gpu.io, bytes);
    try journal.write(.{ .kind = "terminal_ownership_artifact", .ordinal = gpu.ctx.device_ordinal, .name = name, .file = filename, .bytes = bytes.len, .sha256 = &hash(bytes) });
}
fn freeBytes(gpu: check.Gpu) !usize {
    const available = (try gpu.ctx.memInfo()).free;
    if (available < 1536 << 20) return error.GpuMemoryFloor;
    return available;
}
fn kernelName(f: cuda.Function) ![]const u8 {
    var lib = f.d.lib;
    const get_name = lib.lookup(*const fn (*?[*:0]const u8, cuda.abi.Function) callconv(.c) cuda.abi.Result, "cuFuncGetName") orelse return error.MissingSymbol;
    var name: ?[*:0]const u8 = null;
    try f.d.check(get_name(&name, f.handle), "cuFuncGetName");
    return std.mem.span(name orelse return error.MissingSymbol);
}
fn argsFor(input: u64, pick: u64, n: usize) cuda.Args {
    var args: cuda.Args = .{};
    args.add(input);
    args.add(pick);
    args.add(@as(u32, @intCast(n)));
    args.add(@as(u32, @intCast(n)));
    return args;
}
fn checkNode(actual: cuda.abi.KernelNodeParams, function: cuda.Function, expected: *const cuda.Args) ![64]u8 {
    if (actual.func != function.handle or actual.grid_x != 1 or actual.grid_y != 1 or actual.grid_z != 1 or actual.block_x != 256 or actual.block_y != 1 or actual.block_z != 1 or actual.shared_bytes != 0 or actual.extra != null) return error.TerminalGraphGeometry;
    if (expected.count != 4) return error.TerminalGraphArguments;
    const params = actual.params orelse return error.TerminalGraphArguments;
    var digest = std.crypto.hash.sha2.Sha256.init(.{});
    for ([_]u16{ 8, 8, 4, 4 }, 0..) |size, index| {
        if (expected.sizes[index] != size) return error.TerminalGraphArguments;
        const bytes: [*]const u8 = @ptrCast(params[index] orelse return error.TerminalGraphArguments);
        const want = expected.storage[expected.offsets[index]..][0..size];
        if (!std.mem.eql(u8, bytes[0..size], want)) return error.TerminalGraphArguments;
        digest.update(bytes[0..size]);
    }
    var sha: [32]u8 = undefined;
    digest.final(&sha);
    return std.fmt.bytesToHex(sha, .lower);
}
fn checkDependency(node: cuda.graph.Node, producer: cuda.graph.Node, arm: Arm, count: usize, parent: cuda.graph.Node) !void {
    if (node == producer) {
        if (count != 0) return error.TerminalGraphDependency;
    } else if (arm != .device or count != 1 or parent != producer) return error.TerminalGraphDependency;
}
fn capture(gpu: check.Gpu, kernels: *const cuda.tree_round.Kernels, stream: cuda.Stream, input: u64, hot: u64, pick: u64, n: usize, arm: Arm, context: []const u8, index: usize, journal: anytype) !cuda.graph.Exec {
    try cuda.graph.beginCapture(stream, .thread_local);
    var capturing = true;
    defer if (capturing) {
        if (cuda.graph.endCapture(stream)) |graph_value| {
            var abandoned = graph_value;
            abandoned.deinit();
        } else |_| {}
    };
    try gpu.d.check(gpu.d.api.cuMemcpyDtoDAsync_v2(hot, input, n * 2, stream.handle), "capture carrier cuMemcpyDtoDAsync");
    if (arm == .device) try kernels.argmax(stream, hot, pick, 1, n, n);
    var graph = try cuda.graph.endCapture(stream);
    capturing = false;
    defer graph.deinit();
    const want_nodes: usize = if (arm == .host) 1 else 2;
    if (try graph.nodeCount() != want_nodes) return error.TerminalGraphNodeCount;
    var storage: [2]cuda.graph.Node = undefined;
    const nodes = try graph.nodes(&storage);
    if (nodes.len != want_nodes) return error.TerminalGraphNodeCount;
    var lib = gpu.d.lib;
    const get_type = lib.lookup(*const fn (cuda.graph.Node, *c_uint) callconv(.c) cuda.abi.Result, "cuGraphNodeGetType") orelse return error.MissingSymbol;
    const dependencies = lib.lookup(*const fn (cuda.graph.Node, ?[*]cuda.graph.Node, *usize) callconv(.c) cuda.abi.Result, "cuGraphNodeGetDependencies") orelse return error.MissingSymbol;
    var producer: cuda.graph.Node = null;
    var kernel: cuda.graph.Node = null;
    var types: [2]u32 = @splat(std.math.maxInt(u32));
    for (nodes, 0..) |node, ni| {
        try gpu.d.check(get_type(node, &types[ni]), "cuGraphNodeGetType");
        switch (types[ni]) {
            0 => {
                if (kernel != null or arm != .device) return error.TerminalGraphNodeType;
                kernel = node;
            },
            1 => {
                if (producer != null) return error.TerminalGraphNodeType;
                producer = node;
            },
            else => return error.TerminalGraphNodeType,
        }
    }
    if (producer == null or (arm == .device and kernel == null)) return error.TerminalGraphNodeType;
    for (nodes) |node| {
        var count: usize = 2;
        var parents: [2]cuda.graph.Node = @splat(null);
        try gpu.d.check(dependencies(node, &parents, &count), "cuGraphNodeGetDependencies");
        try checkDependency(node, producer, arm, count, parents[0]);
    }
    var values: [4]u64 = @splat(0);
    var sizes: [4]u32 = @splat(0);
    var digest: [64]u8 = @splat('0');
    var actual: cuda.abi.KernelNodeParams = std.mem.zeroes(cuda.abi.KernelNodeParams);
    if (arm == .device) {
        var expected = argsFor(hot, pick, n);
        try gpu.d.check(gpu.d.api.cuGraphKernelNodeGetParams_v2(kernel, &actual), "cuGraphKernelNodeGetParams");
        digest = try checkNode(actual, kernels.argmax_fn, &expected);
        for (&values, 0..) |*value, ai| {
            const bytes: [*]const u8 = @ptrCast(actual.params.?[ai].?);
            value.* = if (ai < 2) std.mem.readInt(u64, bytes[0..8], .little) else std.mem.readInt(u32, bytes[0..4], .little);
        }
        sizes = .{ 8, 8, 4, 4 };
    }
    try journal.write(.{ .kind = "terminal_ownership_graph", .ordinal = gpu.ctx.device_ordinal, .context = context, .row = index, .arm = @tagName(arm), .graph_handle = @intFromPtr(graph.handle), .node_count = nodes.len, .node_types = types[0..nodes.len], .producer_node = @intFromPtr(producer), .producer_dependencies = 0, .source_ptr = input, .hot_ptr = hot, .producer_bytes = n * 2, .memcpy_params_request_only = true, .memcpy_evidence = "actual memcpy node type and zero dependencies; src/dst/bytes are capture request only because this binding has no memcpy-param readback ABI", .kernel_node = @intFromPtr(kernel), .kernel_dependencies = @intFromBool(arm == .device), .kernel_parent = if (arm == .device) @intFromPtr(producer) else @as(usize, 0), .function_handle = @intFromPtr(actual.func), .requested_symbol = if (arm == .device) "tf_round_bf16_argmax" else "", .symbol = if (arm == .device) try kernelName(kernels.argmax_fn) else "", .grid = [_]u32{ actual.grid_x, actual.grid_y, actual.grid_z }, .block = [_]u32{ actual.block_x, actual.block_y, actual.block_z }, .shared_bytes = actual.shared_bytes, .argument_values = values, .argument_sizes = sizes, .argument_sha256 = &digest, .kernel_evidence = "actual captured ABI/function/geometry/dependency readback before instantiate", .immutable_after_instantiate = true });
    return graph.instantiate();
}

const Ownership = struct {
    gpa: std.mem.Allocator,
    engine: Engine,
    chain: qwen.Engine,
    session: qwen.Session,
    stages: [2]qwen.engine.Stage,
    adapter: Adapter,
    lane: lanes.Stream,
    cache_memory: []u16,
    session_memory: []u16,

    fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, stream: cuda.Stream, hot: u64, pinned: cuda.HostBuffer) !*Ownership {
        const self = try gpa.create(Ownership);
        errdefer gpa.destroy(self);
        self.* = undefined;
        self.gpa = gpa;
        self.cache_memory = try gpa.alloc(u16, vocab + guard);
        errdefer gpa.free(self.cache_memory);
        self.session_memory = try gpa.alloc(u16, vocab + guard);
        errdefer gpa.free(self.session_memory);
        self.engine.d = d;
        self.engine.stream = stream;
        self.engine.placement = .{ .first = 1, .end = 2, .total = 2 };
        self.engine.g.vocab = vocab;
        self.engine.logits = hot;
        self.engine.host_logits = pinned;
        self.chain.partitions = &self.stages;
        self.session.chain = &self.chain;
        self.session.logits = self.session_memory[guard / 2 ..][0..vocab];
        self.session.caches[0].logits = &.{};
        self.session.caches[1].logits = self.cache_memory[guard / 2 ..][0..vocab];
        self.adapter = try Adapter.init(gpa, &self.session, 1024);
        errdefer self.adapter.deinit();
        self.lane = try lanes.Stream.init(gpa, .{ .id = "terminal-ownership", .prompt = &.{1}, .max_new = 512, .sampling = .{ .seed = 0, .temperature = 0 } });
        self.adapter.active = &self.lane;
        self.reset();
        return self;
    }

    fn deinit(self: *Ownership) void {
        const gpa = self.gpa;
        self.lane.deinit(gpa);
        self.adapter.deinit();
        gpa.free(self.session_memory);
        gpa.free(self.cache_memory);
        gpa.destroy(self);
    }

    fn reset(self: *Ownership) void {
        @memset(self.cache_memory, 0xa5a5);
        @memset(self.session_memory, 0xa5a5);
        @memset(self.session.caches[1].logits, 0xffff);
        @memset(self.session.logits, 0xffff);
        @memset(self.engine.host_logits.bytes, 0xff);
        @memset(self.adapter.values, -123.0);
        self.session.position = 0;
        self.session.poisoned = false;
        for (&self.session.caches) |*cache| {
            cache.len = 0;
            cache.written = 0;
            cache.last = null;
            cache.terminal_greedy = null;
        }
    }

    fn consume(self: *Ownership, arm: Arm) !u32 {
        const shape: Engine.Shape = .{ .total = 1, .nseg = 1, .record = false, .prompt = true, .taps = false, .terminal_greedy = arm == .device };
        const terminal = [_]qwen.shared_engine.Part{.{ .cache = &self.session.caches[1], .rows = 1 }};
        try self.engine.finish(&terminal, .last, shape);
        for (&self.session.caches) |*cache| {
            const parts = [_]qwen.shared_engine.Part{.{ .cache = cache, .rows = 1 }};
            Engine.commit(&parts, false);
        }
        @memcpy(self.session.logits, self.session.caches[1].logits);
        self.session.position += 1;
        return @intCast(try self.adapter.backend().first(&self.lane, self.session.position));
    }

    fn qualify(self: *Ownership, expected: []const u8, arm: Arm, token: u32) !void {
        for ([_][]const u8{ self.engine.host_logits.bytes[0..row_bytes], std.mem.sliceAsBytes(self.session.caches[1].logits), std.mem.sliceAsBytes(self.session.logits) }) |actual| try check.sameBytes("full-row ownership", actual, expected);
        try canaries(std.mem.sliceAsBytes(self.cache_memory), row_bytes);
        try canaries(std.mem.sliceAsBytes(self.session_memory), row_bytes);
        for (self.session.caches) |cache| {
            if (cache.len != self.session.position or cache.written != self.session.position or cache.last != null) return error.OwnershipPositionMismatch;
        }
        if (self.session.caches[0].terminal_greedy != null) return error.OwnershipStageLeak;
        const pick = try self.session.cachedGreedy(self.session.position);
        if (arm == .device) {
            if (pick != token) return error.OwnershipPickMismatch;
            for (self.adapter.values) |value| if (value != -123.0) return error.OwnershipUnexpectedHostScan;
        } else if (pick != null) return error.OwnershipPickMismatch;
    }
};

const Buffers = struct {
    input: cuda.DeviceBuffer,
    hot: cuda.DeviceBuffer,
    pinned: cuda.HostBuffer,
    owner: *Ownership,
    snapshots: [3][]u16,
    tokens: [rows]u32 = undefined,
    positions: [rows]u64 = undefined,

    fn init(gpu: check.Gpu, stream: cuda.Stream, source: []const u8) !Buffers {
        var input = try cuda.DeviceBuffer.alloc(gpu.d, source.len + 2 * guard);
        errdefer input.free();
        try input.fill8(0xa5, null);
        try input.upload(guard, source);
        var hot = try cuda.DeviceBuffer.alloc(gpu.d, row_bytes + 8 + 2 * guard);
        errdefer hot.free();
        try hot.fill8(0xa5, null);
        var pinned = try cuda.HostBuffer.alloc(gpu.d, row_bytes + 8 + 2 * guard);
        errdefer pinned.free();
        @memset(pinned.bytes, 0xa5);
        const owner = try Ownership.init(gpu.gpa, gpu.d, stream, hot.ptr + guard, .{ .d = gpu.d, .bytes = @alignCast(pinned.bytes[guard..][0 .. row_bytes + 8]) });
        errdefer owner.deinit();
        var snapshots: [3][]u16 = undefined;
        var made: usize = 0;
        errdefer for (snapshots[0..made]) |s| gpu.gpa.free(s);
        for (&snapshots) |*s| {
            s.* = try gpu.gpa.alloc(u16, rows * vocab);
            made += 1;
        }
        return .{ .input = input, .hot = hot, .pinned = pinned, .owner = owner, .snapshots = snapshots };
    }
    fn deinit(b: *Buffers, gpu: check.Gpu) void {
        for (b.snapshots) |s| gpu.gpa.free(s);
        b.owner.deinit();
        b.pinned.free();
        b.hot.free();
        b.input.free();
    }
    fn reset(b: *Buffers) !void {
        b.owner.reset();
        @memset(&b.tokens, 0xffffffff);
        @memset(&b.positions, 0);
        for (b.snapshots) |s| @memset(s, 0xffff);
        const payload: cuda.DeviceBuffer = .{ .d = b.hot.d, .ptr = b.hot.ptr + guard, .len = row_bytes + 8, .borrowed = true };
        try payload.fill8(0xff, null);
    }
};

const Graphs = struct {
    execs: [2][rows]cuda.graph.Exec = undefined,
    made: [2]usize = .{ 0, 0 },
    fn init(gpu: check.Gpu, kernels: *const cuda.tree_round.Kernels, stream: cuda.Stream, b: *Buffers, journal: anytype) !Graphs {
        var result: Graphs = .{};
        errdefer {
            cuda.completion.drain(gpu.d, stream, "ownership graph creation");
            result.deinit();
        }
        for ([_]Arm{ .host, .device }, 0..) |arm, ai| for (0..rows) |index| {
            result.execs[ai][index] = try capture(gpu, kernels, stream, b.input.ptr + guard + index * row_bytes, b.hot.ptr + guard, b.hot.ptr + guard + row_bytes, vocab, arm, "real", index, journal);
            result.made[ai] += 1;
            try result.execs[ai][index].upload(stream);
        };
        try stream.synchronize();
        return result;
    }
    fn deinit(g: *Graphs) void {
        for (&g.execs, g.made) |*arm, count| for (arm[0..count]) |*exec| exec.deinit();
    }
};

fn sweep(b: *Buffers, graphs: *const Graphs, stream: cuda.Stream, arm: Arm, source: []const u8, qualify: bool) !void {
    for (0..rows) |index| {
        try graphs.execs[@backingInt(arm)][index].launchOn(stream);
        b.tokens[index] = try b.owner.consume(arm);
        b.positions[index] = b.owner.session.position;
        if (qualify) {
            try b.owner.qualify(source[index * row_bytes ..][0..row_bytes], arm, b.tokens[index]);
            const views = [_][]const u16{ b.owner.engine.rowLogits(0), b.owner.session.caches[1].logits, b.owner.session.logits };
            for (b.snapshots, views) |archive, view| @memcpy(archive[index * vocab ..][0..vocab], view);
        }
    }
}

fn evidence(gpu: check.Gpu, b: *Buffers, source: []const u8, expected: []const u32, name: []const u8, qualified: bool, journal: anytype) !void {
    try check.sameBytes("ownership tokens", std.mem.asBytes(&b.tokens), std.mem.sliceAsBytes(expected));
    for (b.positions, 1..) |position, wanted| if (position != wanted) return error.OwnershipPositionMismatch;
    const token_name = try std.fmt.allocPrint(gpu.gpa, "{s}_tokens", .{name});
    defer gpu.gpa.free(token_name);
    try save(gpu, journal, token_name, std.mem.asBytes(&b.tokens));
    const position_name = try std.fmt.allocPrint(gpu.gpa, "{s}_positions", .{name});
    defer gpu.gpa.free(position_name);
    try save(gpu, journal, position_name, std.mem.asBytes(&b.positions));
    if (qualified) for (b.snapshots, [_][]const u8{ "engine", "cache", "session" }) |snapshot, suffix| {
        const label = try std.fmt.allocPrint(gpu.gpa, "{s}_{s}_rows", .{ name, suffix });
        defer gpu.gpa.free(label);
        try check.sameBytes("ownership archived rows", std.mem.sliceAsBytes(snapshot), source);
        try save(gpu, journal, label, std.mem.sliceAsBytes(snapshot));
    };
    const device = try check.download(gpu, b.hot);
    defer gpu.gpa.free(device);
    for ([_][]const u8{ b.pinned.bytes, device, std.mem.sliceAsBytes(b.owner.cache_memory), std.mem.sliceAsBytes(b.owner.session_memory) }, [_][]const u8{ "pinned", "device", "cache", "session" }, 0..) |bytes, suffix, i| {
        try canaries(bytes, if (i < 2) row_bytes + 8 else row_bytes);
        try check.sameBytes("ownership final row", bytes[guard..][0..row_bytes], source[(rows - 1) * row_bytes ..]);
        const label = try std.fmt.allocPrint(gpu.gpa, "{s}_{s}_guard", .{ name, suffix });
        defer gpu.gpa.free(label);
        try save(gpu, journal, label, bytes);
    }
}

fn semantics(gpu: check.Gpu, kernels: *const cuda.tree_round.Kernels, stream: cuda.Stream, b: *Buffers, journal: anytype) !void {
    const row = try gpu.gpa.alloc(u16, vocab);
    defer gpu.gpa.free(row);
    for (0..semantic_cases) |index| {
        const expected = semanticRow(index, row);
        var input = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(row));
        defer input.free();
        defer cuda.completion.drain(gpu.d, stream, "ownership semantic input");
        var baseline: u32 = undefined;
        for ([_]Arm{ .host, .device }) |arm| {
            b.owner.reset();
            var graph = try capture(gpu, kernels, stream, input.ptr, b.hot.ptr + guard, b.hot.ptr + guard + row_bytes, vocab, arm, "semantic", index, journal);
            defer graph.deinit();
            defer cuda.completion.drain(gpu.d, stream, "ownership semantic graph");
            try graph.upload(stream);
            try stream.synchronize();
            try graph.launchOn(stream);
            const token = try b.owner.consume(arm);
            if (arm == .host) baseline = token else if (token != baseline) return error.OwnershipSemanticFallbackMismatch;
            if (expected.nonfinite == 0 and token != expected.token) return error.TerminalSemanticMismatch;
            const cached = try b.owner.session.cachedGreedy(1);
            if (arm == .device and expected.nonfinite == 0) {
                if (cached != token) return error.OwnershipPickMismatch;
            } else if (cached != null) return error.OwnershipPickMismatch;
            const views = [_][]const u16{ b.owner.engine.rowLogits(0), b.owner.session.caches[1].logits, b.owner.session.logits };
            for (views, [_][]const u8{ "engine", "cache", "session" }) |view, suffix| {
                try check.sameBytes("ownership semantic full row", std.mem.sliceAsBytes(view), std.mem.sliceAsBytes(row));
                const name = try std.fmt.allocPrint(gpu.gpa, "semantic{d}_{s}_{s}", .{ index, @tagName(arm), suffix });
                defer gpu.gpa.free(name);
                try save(gpu, journal, name, std.mem.sliceAsBytes(view));
            }
            const actual: Pick = if (arm == .device) std.mem.bytesToValue(Pick, b.owner.engine.host_logits.bytes[row_bytes..][0..8]) else .{ .token = token, .nonfinite = 0 };
            if (arm == .device) try check.sameBytes("ownership semantic strict Pick", std.mem.asBytes(&actual), std.mem.asBytes(&expected));
            try canaries(b.pinned.bytes, row_bytes + 8);
            try canaries(std.mem.sliceAsBytes(b.owner.cache_memory), row_bytes);
            try canaries(std.mem.sliceAsBytes(b.owner.session_memory), row_bytes);
            try journal.write(.{ .kind = "terminal_ownership_semantic", .ordinal = gpu.ctx.device_ordinal, .index = index, .arm = @tagName(arm), .vocab = vocab, .expected = expected, .actual = actual, .token = token, .cached_token = cached, .position = b.owner.session.position, .fullrow_copies_exact = 3, .dtoh_bytes = row_bytes + @as(usize, if (arm == .device) 8 else 0), .graph_launches = 1, .row_sync_count = 1 });
        }
        const consumed = try check.download(gpu, input);
        defer gpu.gpa.free(consumed);
        try check.sameBytes("ownership semantic input immutable", consumed, std.mem.sliceAsBytes(row));
        const name = try std.fmt.allocPrint(gpu.gpa, "semantic{d}_input", .{index});
        defer gpu.gpa.free(name);
        try save(gpu, journal, name, consumed);
    }
}

pub fn run(gpu: check.Gpu, fixture_dir: []const u8, manifest_sha: []const u8, journal: anytype) !void {
    try validatePaths(fixture_dir, manifest_sha);
    _ = try freeBytes(gpu);
    const path = try std.fs.path.join(gpu.gpa, &.{ fixture_dir, "manifest.json" });
    defer gpu.gpa.free(path);
    const manifest_raw = try std.Io.Dir.cwd().readFileAlloc(gpu.io, path, gpu.gpa, .limited(1 << 20));
    defer gpu.gpa.free(manifest_raw);
    try verify(manifest_raw, manifest_sha);
    const parsed = try std.json.parseFromSlice(Manifest, gpu.gpa, manifest_raw, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    const m = parsed.value;
    try validateManifest(m);
    const source = try std.Io.Dir.cwd().readFileAlloc(gpu.io, m.path, gpu.gpa, .limited(rows * row_bytes + 1));
    defer gpu.gpa.free(source);
    if (source.len != rows * row_bytes) return error.InvalidTerminalGraphFixture;
    try verify(source, m.sha256);
    for (0..rows) |index| {
        try verify(source[index * row_bytes ..][0..row_bytes], m.row_sha256[index]);
        for (m.row_sha256[0..index]) |prior| if (std.mem.eql(u8, prior, m.row_sha256[index])) return error.TerminalRowsNotDistinct;
    }
    try verify(cuda.kernels.tree_round, m.tree_image_sha256);
    var module = try cuda.Module.load(gpu.d, cuda.kernels.tree_round);
    defer module.unload();
    const kernels = try cuda.tree_round.Kernels.resolve(module);
    var stream = try cuda.Stream.init(gpu.d, false);
    defer stream.deinit();
    var b = try Buffers.init(gpu, stream, source);
    defer b.deinit(gpu);
    defer cuda.completion.drain(gpu.d, stream, "terminal ownership screen");
    var graphs = try Graphs.init(gpu, &kernels, stream, &b, journal);
    defer {
        cuda.completion.drain(gpu.d, stream, "terminal ownership graph release");
        graphs.deinit();
    }
    const uploaded = try check.download(gpu, b.input);
    defer gpu.gpa.free(uploaded);
    try canaries(uploaded, source.len);
    try check.sameBytes("ownership uploaded fixture", uploaded[guard..][0..source.len], source);
    try save(gpu, journal, "consumed_logits", uploaded);
    try journal.write(.{ .kind = "terminal_ownership_scope", .ordinal = gpu.ctx.device_ordinal, .manifest_sha256 = manifest_sha, .fixture_sha256 = m.sha256, .tree_image_sha256 = m.tree_image_sha256, .argmax_sha256 = m.argmax_sha256, .rows = rows, .vocab = vocab, .row_bytes = row_bytes, .candidate_bytes = row_bytes + 8, .stages = 2, .input_ptr = b.input.ptr + guard, .hot_ptr = b.hot.ptr + guard, .pick_ptr = b.hot.ptr + guard + row_bytes, .included = "common captured D2D; actual Engine.finish(.last) one full-row D2H/sync and Cache copy; actual Engine.commit both caches; production explicit Cache-to-Session copy; actual Session.cachedGreedy through native Adapter.backend.first", .excluded = "model/head computation, first-stage execution, Session.stepRows orchestration; no model performance claim", .copy_count_evidence = "requested operations from pinned actual finish source; CUDA graph memcpy node type/dependencies read back, memcpy arguments request-only", .event_scope = "whole stream span including host gaps; not isolated kernel time", .archive_copies_in_timing = false, .free_bytes = try freeBytes(gpu) });
    try journal.write(.{ .kind = "terminal_ownership_resources", .ordinal = gpu.ctx.device_ordinal, .symbol = try kernelName(kernels.argmax_fn), .function_handle = @intFromPtr(kernels.argmax_fn.handle), .regs = try kernels.argmax_fn.attribute(.num_regs), .local_bytes = try kernels.argmax_fn.attribute(.local_size_bytes), .static_shared_bytes = try kernels.argmax_fn.attribute(.shared_size_bytes), .dynamic_shared_bytes = 0 });
    try semantics(gpu, &kernels, stream, &b, journal);
    for ([_][]const u8{ "before", "after" }, 0..) |phase, pi| {
        for ([_]Arm{ .host, .device }) |arm| {
            const name = try std.fmt.allocPrint(gpu.gpa, "{s}_{s}", .{ phase, @tagName(arm) });
            defer gpu.gpa.free(name);
            try b.reset();
            try sweep(&b, &graphs, stream, arm, source, true);
            try evidence(gpu, &b, source, m.tokens, name, true, journal);
            try journal.write(.{ .kind = "terminal_ownership_qualified", .ordinal = gpu.ctx.device_ordinal, .phase = phase, .arm = @tagName(arm), .rows = rows, .copies_exact = 3, .tokens_exact = true });
        }
        if (pi != 0) break;
        var start = try cuda.Event.init(gpu.d, true);
        defer start.deinit();
        var end = try cuda.Event.init(gpu.d, true);
        defer end.deinit();
        for (0..6) |pair| for (0..2) |order| {
            const arm: Arm = if ((pair + order) % 2 == 0) .host else .device;
            const name = try std.fmt.allocPrint(gpu.gpa, "pair{d}_{s}", .{ pair, @tagName(arm) });
            defer gpu.gpa.free(name);
            try b.reset();
            try stream.synchronize();
            const t0 = check.now(gpu.io);
            try start.record(stream);
            try sweep(&b, &graphs, stream, arm, source, false);
            try end.record(stream);
            try end.synchronize();
            const wall_ms = @as(f64, @floatFromInt(check.now(gpu.io) - t0)) / 1e6;
            const event_ms = try cuda.Event.elapsedMs(start, end);
            if (!std.math.isFinite(wall_ms) or wall_ms <= 0 or !std.math.isFinite(event_ms) or event_ms <= 0) return error.InvalidTerminalTiming;
            try evidence(gpu, &b, source, m.tokens, name, false, journal);
            try journal.write(.{ .kind = "terminal_ownership_sample", .ordinal = gpu.ctx.device_ordinal, .arm = @tagName(arm), .pair = pair, .order = order, .warmup = pair == 0, .rows = rows, .graph_launches = rows, .producer_copies = rows, .producer_bytes = rows * row_bytes, .dtoh_count = rows, .dtoh_bytes = rows * (row_bytes + @as(usize, if (arm == .device) 8 else 0)), .row_sync_count = rows, .event_sync_count = 1, .cache_copy_bytes = rows * row_bytes, .session_copy_bytes = rows * row_bytes, .diagnostic_copy_bytes = 0, .cpu_argmax_count = if (arm == .host) rows else @as(usize, 0), .cpu_pick_validation_count = if (arm == .device) rows else @as(usize, 0), .adapter_draws = rows, .committed_stages = 2, .wall_ms = wall_ms, .event_ms = event_ms, .outputs_consumed = rows, .copy_counts_request_only = true, .free_bytes = try freeBytes(gpu) });
        };
    }
    const final_input = try check.download(gpu, b.input);
    defer gpu.gpa.free(final_input);
    try save(gpu, journal, "final_logits", final_input);
    try check.sameBytes("ownership input immutable", final_input, uploaded);
    try journal.write(.{ .kind = "terminal_ownership_complete", .ordinal = gpu.ctx.device_ordinal, .rows = rows, .measured_pairs = 5, .warmup_pairs = 1, .tokens_exact = true, .production_qualification = false, .free_bytes = try freeBytes(gpu) });
}

const semantic_vocab = 513;
const semantic_cases = 8;

test "actual finish commit Session getter and native Adapter preserve ownership and fallback" {
    const Fake = struct {
        var copies: usize = 0;
        var bytes_copied: usize = 0;
        var syncs: usize = 0;
        fn sync(_: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
            syncs += 1;
            return cuda.abi.success;
        }
        fn copy(out: ?*anyopaque, input: cuda.abi.DevicePtr, bytes: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
            const dst: [*]u8 = @ptrCast(out.?);
            const src: [*]const u8 = @ptrFromInt(input);
            @memcpy(dst[0..bytes], src[0..bytes]);
            copies += 1;
            bytes_copied += bytes;
            return cuda.abi.success;
        }
    };
    const a = std.testing.allocator;
    const raw = try a.alignedAlloc(u8, .@"16", row_bytes + 8 + guard * 2);
    defer a.free(raw);
    const pinned = try a.alignedAlloc(u8, .@"16", row_bytes + 8 + guard * 2);
    defer a.free(pinned);
    @memset(raw, 0xa5);
    @memset(pinned, 0xa5);
    var driver: cuda.Driver = undefined;
    driver.api.cuMemcpyDtoHAsync_v2 = Fake.copy;
    driver.api.cuStreamSynchronize = Fake.sync;
    const owner = try Ownership.init(a, &driver, .{ .d = &driver, .handle = null }, @intFromPtr(raw.ptr + guard), .{ .d = &driver, .bytes = @alignCast(pinned[guard..][0 .. row_bytes + 8]) });
    defer owner.deinit();
    const input = std.mem.bytesAsSlice(u16, raw[guard..][0..row_bytes]);
    for (0..semantic_cases) |index| {
        const pick = semanticRow(index, input);
        @memcpy(raw[guard + row_bytes ..][0..8], std.mem.asBytes(&pick));
        var baseline: u32 = undefined;
        for ([_]Arm{ .host, .device }) |arm| {
            owner.reset();
            Fake.copies = 0;
            Fake.bytes_copied = 0;
            Fake.syncs = 0;
            const token = try owner.consume(arm);
            if (arm == .host) baseline = token else try std.testing.expectEqual(baseline, token);
            if (pick.nonfinite == 0) try std.testing.expectEqual(pick.token, token);
            try std.testing.expectEqual(1, Fake.copies);
            try std.testing.expectEqual(row_bytes + @as(usize, if (arm == .device) 8 else 0), Fake.bytes_copied);
            try std.testing.expectEqual(1, Fake.syncs);
            try std.testing.expectEqual(1, owner.session.position);
            for (owner.session.caches) |cache| {
                try std.testing.expectEqual(1, cache.len);
                try std.testing.expectEqual(1, cache.written);
            }
            for ([_][]const u16{ owner.engine.rowLogits(0), owner.session.caches[1].logits, owner.session.logits }) |row| try std.testing.expectEqualSlices(u16, input, row);
            try canaries(raw, row_bytes + 8);
            try canaries(pinned, row_bytes + 8);
            try canaries(std.mem.sliceAsBytes(owner.cache_memory), row_bytes);
            try canaries(std.mem.sliceAsBytes(owner.session_memory), row_bytes);
            if (arm == .device and pick.nonfinite == 0) {
                try std.testing.expectEqual(token, (try owner.session.cachedGreedy(1)).?);
                for (owner.adapter.values) |value| try std.testing.expectEqual(-123.0, value);
            } else {
                try std.testing.expectEqual(@as(?u32, null), try owner.session.cachedGreedy(1));
                try std.testing.expect(owner.adapter.values[0] != -123.0);
            }
            try std.testing.expectError(error.PositionMismatch, owner.adapter.backend().first(&owner.lane, 0));
            owner.session.poisoned = true;
            try std.testing.expectError(error.PoisonedSession, owner.adapter.backend().first(&owner.lane, 1));
        }
    }
    for ([_]Pick{ .{ .token = vocab, .nonfinite = 0 }, .{ .token = 0, .nonfinite = 1 }, .{ .token = 0xffffffff, .nonfinite = 8 } }) |pick| {
        owner.reset();
        @memcpy(raw[guard + row_bytes ..][0..8], std.mem.asBytes(&pick));
        try std.testing.expectError(error.InvalidTerminalPick, owner.consume(.device));
        try std.testing.expectEqual(0, owner.session.position);
        try std.testing.expectEqual(0, owner.session.caches[1].len);
        try std.testing.expect(owner.session.caches[1].terminal_greedy == null);
    }
}

fn semanticRow(index: usize, row: []u16) Pick {
    @memset(row, 0xbf80);
    switch (index) {
        0 => {
            row[3] = 0x4000;
            row[257] = 0x4000;
            return .{ .token = 3, .nonfinite = 0 };
        },
        1 => {
            row[0] = 0x8000;
            row[257] = 0;
            return .{ .token = 0, .nonfinite = 0 };
        },
        2 => {
            row[0] = 0;
            row[257] = 0x8000;
            return .{ .token = 0, .nonfinite = 0 };
        },
        3 => {
            row[511] = 0xbf00;
            return .{ .token = 511, .nonfinite = 0 };
        },
        4 => {
            row[3] = 0x7fc1;
            return .{ .token = 0xffffffff, .nonfinite = 1 };
        },
        5 => {
            row[3] = 0x7f80;
            return .{ .token = 0xffffffff, .nonfinite = 2 };
        },
        6 => {
            row[3] = 0xff80;
            return .{ .token = 0xffffffff, .nonfinite = 4 };
        },
        7 => {
            row[1] = 0x7fc0;
            row[257] = 0x7f80;
            row[512] = 0xff80;
            return .{ .token = 0xffffffff, .nonfinite = 7 };
        },
        else => unreachable,
    }
}

test "terminal ownership rejects invalid paths, shapes and Pick statuses" {
    try std.testing.expectError(error.InvalidTerminalGraphInput, validatePaths("relative", "x"));
    try std.testing.expectError(error.InvalidTerminalPick, validatePick(.{ .token = 1, .nonfinite = 8 }, 5));
    try std.testing.expectError(error.InvalidTerminalPick, validatePick(.{ .token = 5, .nonfinite = 0 }, 5));
    try std.testing.expectError(error.InvalidTerminalPick, validatePick(.{ .token = 1, .nonfinite = 1 }, 5));
    try std.testing.expectError(error.NonfiniteLogits, validatePick(.{ .token = 0xffffffff, .nonfinite = 7 }, 5));
}
test "terminal ownership exact comparator covers ties signed zeros and nonfinite kinds" {
    for (0..semantic_cases) |index| {
        var row: [semantic_vocab]u16 = undefined;
        const expected = semanticRow(index, &row);
        if (expected.nonfinite == 0) try std.testing.expectEqual(expected.token, try argmax(&row)) else try std.testing.expectError(error.NonfiniteLogits, argmax(&row));
    }
}
test "terminal ownership guard mutations are rejected" {
    var bytes: [guard * 2 + 8]u8 = @splat(0xa5);
    try canaries(&bytes, 8);
    bytes[0] ^= 1;
    try std.testing.expectError(error.TerminalGuardChanged, canaries(&bytes, 8));
    bytes[0] ^= 1;
    bytes[bytes.len - 1] ^= 1;
    try std.testing.expectError(error.TerminalGuardChanged, canaries(&bytes, 8));
}

test "terminal ownership manifest dimensions hashes and token bounds reject mutations" {
    const sha: [64]u8 = @splat('a');
    const tokens: [rows]u32 = @splat(0);
    const hashes: [rows][]const u8 = @splat(&sha);
    const good: Manifest = .{ .version = 1, .rows = rows, .vocab = vocab, .bytes = rows * row_bytes, .path = "C:/fixture.bin", .sha256 = &sha, .row_sha256 = &hashes, .tokens = &tokens, .tree_image_sha256 = &sha, .argmax_sha256 = &sha };
    try validateManifest(good);
    inline for (.{ "version", "rows", "vocab", "bytes" }) |field| {
        var bad = good;
        @field(bad, field) += 1;
        try std.testing.expectError(error.InvalidTerminalGraphFixture, validateManifest(bad));
    }
    var bad = good;
    bad.tokens = tokens[0 .. rows - 1];
    try std.testing.expectError(error.InvalidTerminalGraphFixture, validateManifest(bad));
    var invalid_tokens = tokens;
    invalid_tokens[127] = vocab;
    bad = good;
    bad.tokens = &invalid_tokens;
    try std.testing.expectError(error.InvalidTerminalGraphFixture, validateManifest(bad));
    bad = good;
    bad.sha256 = "z";
    try std.testing.expectError(error.InvalidTerminalGraphFixture, validateManifest(bad));
}

test "terminal ownership rejects every captured kernel ABI and geometry mutation" {
    const function: cuda.Function = .{ .d = undefined, .handle = @ptrFromInt(1) };
    var expected = argsFor(0x100040, 0x200040, vocab);
    var actual_args = argsFor(0x100040, 0x200040, vocab);
    var actual: cuda.abi.KernelNodeParams = .{ .func = function.handle, .grid_x = 1, .grid_y = 1, .grid_z = 1, .block_x = 256, .block_y = 1, .block_z = 1, .shared_bytes = 0, .params = actual_args.pointers(), .extra = null };
    _ = try checkNode(actual, function, &expected);
    for (0..4) |ai| {
        const offset = actual_args.offsets[ai];
        actual_args.storage[offset] ^= 1;
        try std.testing.expectError(error.TerminalGraphArguments, checkNode(actual, function, &expected));
        actual_args.storage[offset] ^= 1;
    }
    inline for (.{ "grid_x", "grid_y", "grid_z", "block_x", "block_y", "block_z", "shared_bytes" }) |field| {
        @field(actual, field) += 1;
        try std.testing.expectError(error.TerminalGraphGeometry, checkNode(actual, function, &expected));
        @field(actual, field) -= 1;
    }
    actual.func = @ptrFromInt(2);
    try std.testing.expectError(error.TerminalGraphGeometry, checkNode(actual, function, &expected));
    actual.func = function.handle;
    actual.params = null;
    try std.testing.expectError(error.TerminalGraphArguments, checkNode(actual, function, &expected));
    actual.params = actual_args.pointers();
    actual_args.ptrs[0] = null;
    try std.testing.expectError(error.TerminalGraphArguments, checkNode(actual, function, &expected));
    actual.params = actual_args.pointers();
    expected.sizes[3] = 8;
    try std.testing.expectError(error.TerminalGraphArguments, checkNode(actual, function, &expected));
    expected.sizes[3] = 4;
    expected.count = 3;
    try std.testing.expectError(error.TerminalGraphArguments, checkNode(actual, function, &expected));
}

test "terminal ownership dependency rejects a missing carrier edge and unexpected parent" {
    const producer: cuda.graph.Node = @ptrFromInt(1);
    const kernel: cuda.graph.Node = @ptrFromInt(2);
    try checkDependency(producer, producer, .host, 0, null);
    try checkDependency(producer, producer, .device, 0, null);
    try checkDependency(kernel, producer, .device, 1, producer);
    try std.testing.expectError(error.TerminalGraphDependency, checkDependency(producer, producer, .device, 1, kernel));
    try std.testing.expectError(error.TerminalGraphDependency, checkDependency(kernel, producer, .host, 1, producer));
    try std.testing.expectError(error.TerminalGraphDependency, checkDependency(kernel, producer, .device, 0, null));
    try std.testing.expectError(error.TerminalGraphDependency, checkDependency(kernel, producer, .device, 2, producer));
    try std.testing.expectError(error.TerminalGraphDependency, checkDependency(kernel, producer, .device, 1, kernel));
}
