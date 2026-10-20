//! Original TensorFold/Ash Hart prompt kernels, embedded unchanged; local typed ownership and eligibility.
//! Source, Apache-2.0 license and captured-image provenance are retained in prompt_aot/.
const std = @import("std");
const cuda = @import("cuda");
const cache_mod = @import("cuda_cache.zig");
const Sha256 = std.crypto.hash.sha2.Sha256;
var launches: [3]std.atomic.Value(u64) = .{ .init(0), .init(0), .init(0) };

pub fn dispatchCounts() [3]u64 {
    var counts: [3]u64 = undefined;
    for (&launches, &counts) |*counter, *count| count.* = counter.load(.monotonic);
    return counts;
}

const absorb_image align(16) = @embedFile("prompt_aot/absorb.cubin").*;
const expand_image align(16) = @embedFile("prompt_aot/expand.cubin").*;
const sparse_image align(16) = @embedFile("prompt_aot/onepass.cubin").*;
const Artifact = struct {
    symbol: [:0]const u8,
    image: []const u8,
    metadata: []const u8,
    image_sha256: []const u8,
    metadata_sha256: []const u8,
};
const inventory = [_]Artifact{
    .{ .symbol = "_absorb_mma", .image = &absorb_image, .metadata = @embedFile("prompt_aot/absorb.json"), .image_sha256 = "e0d640075673e53cf6a3bc6a2ff6a6e171f3c1bf21753b1ebd3c9a1885c26744", .metadata_sha256 = "21fde4d90b97c1911c34e198368be1d97b8de6d233318640d1439323a22da5d1" },
    .{ .symbol = "_expand_mma", .image = &expand_image, .metadata = @embedFile("prompt_aot/expand.json"), .image_sha256 = "d66e8138e663f5d81843fe575ce6ce5e7349d9bee0d3e998648efac9d3360425", .metadata_sha256 = "d02cabb9630d91e8d16c992237b604aec630c4ffb653fba8d7d0ba2593a8b1c2" },
    .{ .symbol = "_sparse_onepass", .image = &sparse_image, .metadata = @embedFile("prompt_aot/onepass.json"), .image_sha256 = "c6af4058bd7cc02d744b1b7150fbacd8062a89200dbe7f1be3a2869d4ccd2e08", .metadata_sha256 = "cb0e8acf4b44543fdae6e82751960aac02ddf58fbf2901c9b5a06fea1fa06f3d" },
};

fn identity(artifacts: []const Artifact) [32]u8 {
    var h = Sha256.init(.{});
    h.update("glm-prompt-latent-aot-v2:sm121:mma-r2048-only:sparse-r1..2048:dense-prefix2051:h32:d256:lw512:aligned16:fp8rs528:w2051:scale1/16:independent-fallback\x00");
    for (artifacts) |a| {
        h.update(a.symbol);
        h.update("\x00");
        h.update(a.image_sha256);
        h.update(a.metadata_sha256);
    }
    return h.finalResult();
}

pub const numerical_identity: [32]u8 = blk: {
    @setEvalBranchQuota(100000);
    break :blk identity(&inventory);
};

fn verifyBytes(bytes: []const u8, expected: []const u8) !void {
    var digest: [32]u8 = undefined;
    Sha256.hash(bytes, &digest, .{});
    const text = std.fmt.bytesToHex(digest, .lower);
    if (!std.mem.eql(u8, &text, expected)) return error.BadPromptAotArtifact;
}

fn metadata(gpa: std.mem.Allocator, artifact: Artifact) !cuda.triton.Meta {
    try verifyBytes(artifact.image, artifact.image_sha256);
    try verifyBytes(artifact.metadata, artifact.metadata_sha256);
    var parsed = try cuda.triton.parseMeta(gpa, artifact.metadata);
    defer parsed.deinit();
    var m = parsed.value;
    if (!std.mem.eql(u8, m.name, artifact.symbol) or m.num_ctas != 1 or m.global_scratch_size != 0 or m.profile_scratch_size != 0 or m.launch_cooperative_grid or m.launch_pdl) return error.BadPromptAotMetadata;
    // Kernel retains Meta: its name must outlive the temporary JSON parse.
    m.name = artifact.symbol;
    return m;
}

pub const MmaArgs = struct {
    input: u64,
    weight: u64,
    out: u64,
    rows: usize,
    heads: usize,
    inner: usize,
    columns: usize,
};
pub const SparseArgs = struct {
    qa: u64,
    cache: cache_mod.Cache,
    tokens: u64,
    counts: u64,
    out: u64,
    rows: usize,
    heads: usize,
    lw: usize,
    width: usize,
    scale: f32,
};

fn aligned(pointers: []const u64) bool {
    for (pointers) |p| if (p == 0 or p % 16 != 0) return false;
    return true;
}

fn mmaEligible(a: MmaArgs, inner: usize, columns: usize) bool {
    return a.rows == 2048 and a.heads == 32 and a.inner == inner and a.columns == columns and aligned(&.{ a.input, a.weight, a.out });
}

fn sparseEligible(a: SparseArgs) bool {
    return a.rows > 0 and a.rows <= 2048 and a.heads == 32 and a.lw == 512 and a.width == 2051 and a.scale == 0.0625 and a.cache.format == .fp8 and a.cache.format.rowBytes(a.lw) == 528 and aligned(&.{ a.qa, a.cache.ptr, a.cache.ptr, a.tokens, a.counts, a.out });
}

pub const Kernels = struct {
    absorb: cuda.triton.Kernel,
    expand: cuda.triton.Kernel,
    sparse: cuda.triton.Kernel,

    pub fn load(gpa: std.mem.Allocator, d: *const cuda.Driver, ctx: *const cuda.Context) !Kernels {
        if (try ctx.capability() != 121) return error.UnsupportedPromptAotDevice;
        var metas: [inventory.len]cuda.triton.Meta = undefined;
        for (inventory, 0..) |artifact, i| metas[i] = try metadata(gpa, artifact);
        var absorb = try cuda.triton.Kernel.load(d, ctx.device, inventory[0].image, metas[0], inventory[0].symbol);
        errdefer absorb.unload();
        var expand = try cuda.triton.Kernel.load(d, ctx.device, inventory[1].image, metas[1], inventory[1].symbol);
        errdefer expand.unload();
        const sparse = try cuda.triton.Kernel.load(d, ctx.device, inventory[2].image, metas[2], inventory[2].symbol);
        return .{ .absorb = absorb, .expand = expand, .sparse = sparse };
    }

    /// The owner drains the rank stream before releasing these borrowed functions/modules.
    pub fn unload(self: *Kernels) void {
        self.sparse.unload();
        self.expand.unload();
        self.absorb.unload();
        self.* = undefined;
    }

    pub fn tryAbsorb(self: *const Kernels, stream: cuda.Stream, a: MmaArgs) !bool {
        if (!mmaEligible(a, 256, 512)) return false;
        try mma(self.absorb, stream, a, 4);
        _ = launches[0].fetchAdd(1, .monotonic);
        return true;
    }

    pub fn tryExpand(self: *const Kernels, stream: cuda.Stream, a: MmaArgs) !bool {
        if (!mmaEligible(a, 512, 256)) return false;
        try mma(self.expand, stream, a, 2);
        _ = launches[1].fetchAdd(1, .monotonic);
        return true;
    }

    fn mma(kernel: cuda.triton.Kernel, stream: cuda.Stream, a: MmaArgs, z: u32) !void {
        var args: cuda.Args = .{};
        args.add(a.input);
        args.add(a.weight);
        args.add(a.out);
        args.add(@as(i32, @intCast(a.rows)));
        try kernel.launchOn(.{ .x = 32, .y = @intCast((a.rows + 63) / 64), .z = z }, stream, &args, .{}, &.{ 0, 1, 2 });
    }

    pub fn trySparse(self: *const Kernels, stream: cuda.Stream, a: SparseArgs) !bool {
        if (!sparseEligible(a)) return false;
        var args: cuda.Args = .{};
        // kv8.parts' LS is the same row base as LC; the captured kernel adds512.
        for ([_]u64{ a.qa, a.cache.ptr, a.cache.ptr, a.tokens, a.counts, a.out }) |p| args.add(p);
        try self.sparse.launchOn(.{ .x = @intCast(a.rows) }, stream, &args, .{}, &.{ 0, 1, 2, 3, 4, 5 });
        _ = launches[2].fetchAdd(1, .monotonic);
        return true;
    }
};

test "embedded prompt inventory verifies metadata and numerical identity covers every artifact" {
    for (inventory) |a| {
        const m = try metadata(std.testing.allocator, a);
        try std.testing.expectEqualStrings(a.symbol, m.name);
        try std.testing.expect(@intFromPtr(a.image.ptr) % 16 == 0);
    }
    try std.testing.expectEqual(numerical_identity, identity(&inventory));
    const previous_identity = [_]u8{ 0xeb, 0x0f, 0x48, 0xbf, 0x37, 0x52, 0x38, 0x95, 0x3d, 0x2c, 0xf1, 0x8e, 0xb6, 0xe3, 0x39, 0x2d, 0x5b, 0xe7, 0xcb, 0xec, 0xb9, 0xd5, 0xeb, 0x6b, 0x22, 0x5a, 0xb5, 0x28, 0x0c, 0x24, 0x10, 0xfb };
    try std.testing.expect(!std.mem.eql(u8, &previous_identity, &numerical_identity));
    var changed = inventory;
    changed[2].image_sha256 = "different payload";
    try std.testing.expect(!std.mem.eql(u8, &numerical_identity, &identity(&changed)));
    try std.testing.expectError(error.BadPromptAotArtifact, verifyBytes("changed bytes", inventory[0].image_sha256));
}

const Mock = struct {
    var calls: usize = 0;
    var values: [8]u64 = undefined;
    var grid: [3]u32 = undefined;
    var threads: u32 = 0;
    var shared: u32 = 0;
    var status: cuda.abi.Result = cuda.abi.success;
    fn launch(f: cuda.abi.Function, gx: c_uint, gy: c_uint, gz: c_uint, bx: c_uint, _: c_uint, _: c_uint, sm: c_uint, _: cuda.abi.Stream, params: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
        calls += 1;
        grid = .{ gx, gy, gz };
        threads = bx;
        shared = sm;
        const sparse = @intFromPtr(f) == 3;
        const count: usize = if (sparse) 8 else 6;
        for (0..count) |i| values[i] = if (!sparse and i == 3) @as(*align(1) const u32, @ptrCast(params.?[i].?)).* else @as(*align(1) const u64, @ptrCast(params.?[i].?)).*;
        return status;
    }

    fn kernels(d: *cuda.Driver) !Kernels {
        d.api.cuLaunchKernel = launch;
        var out: [3]cuda.triton.Kernel = undefined;
        for (inventory, 0..) |a, i| out[i] = .{ .module = .{ .d = d, .handle = null }, .function = .{ .d = d, .handle = @ptrFromInt(i + 1) }, .meta = try metadata(std.testing.allocator, a) };
        calls = 0;
        status = cuda.abi.success;
        return .{ .absorb = out[0], .expand = out[1], .sparse = out[2] };
    }
};

test "prompt AOT raw driver ABI retains runtime i32 rows and both scratch suffixes" {
    var d: cuda.Driver = undefined;
    const k = try Mock.kernels(&d);
    const stream: cuda.Stream = .{ .d = &d, .handle = null };
    const a: MmaArgs = .{ .input = 16, .weight = 32, .out = 48, .rows = 2048, .heads = 32, .inner = 256, .columns = 512 };
    try std.testing.expect(try k.tryAbsorb(stream, a));
    try std.testing.expectEqualSlices(u64, &.{ 16, 32, 48, 2048, 0, 0 }, Mock.values[0..6]);
    try std.testing.expectEqualSlices(u32, &.{ 32, 32, 4 }, &Mock.grid);
    try std.testing.expectEqual(@as(u32, 128), Mock.threads);
    try std.testing.expectEqual(@as(u32, 49152), Mock.shared);
    var e = a;
    e.inner = 512;
    e.columns = 256;
    try std.testing.expect(try k.tryExpand(stream, e));
    try std.testing.expectEqualSlices(u32, &.{ 32, 32, 2 }, &Mock.grid);
    const s: SparseArgs = .{ .qa = 16, .cache = .{ .ptr = 32, .format = .fp8 }, .tokens = 48, .counts = 64, .out = 80, .rows = 58, .heads = 32, .lw = 512, .width = 2051, .scale = 0.0625 };
    try std.testing.expect(try k.trySparse(stream, s));
    try std.testing.expectEqualSlices(u64, &.{ 16, 32, 32, 48, 64, 80, 0, 0 }, &Mock.values);
    try std.testing.expectEqualSlices(u32, &.{ 58, 1, 1 }, &Mock.grid);
    try std.testing.expectEqual(@as(u32, 256), Mock.threads);
    try std.testing.expectEqual(@as(u32, 67584), Mock.shared);
}

test "unsupported prompt layouts return false without launching" {
    var d: cuda.Driver = undefined;
    const k = try Mock.kernels(&d);
    const stream: cuda.Stream = .{ .d = &d, .handle = null };
    const good: MmaArgs = .{ .input = 16, .weight = 32, .out = 48, .rows = 2048, .heads = 32, .inner = 256, .columns = 512 };
    var a = good;
    a.input = 17;
    try std.testing.expect(!try k.tryAbsorb(stream, a));
    a = good;
    a.weight = 0;
    try std.testing.expect(!try k.tryAbsorb(stream, a));
    a = good;
    a.rows = 2049;
    try std.testing.expect(!try k.tryAbsorb(stream, a));
    a = good;
    a.heads = 16;
    try std.testing.expect(!try k.tryAbsorb(stream, a));
    var s: SparseArgs = .{ .qa = 16, .cache = .{ .ptr = 32, .format = .fp8 }, .tokens = 52, .counts = 64, .out = 80, .rows = 1, .heads = 32, .lw = 512, .width = 2051, .scale = 0.0625 };
    try std.testing.expect(!try k.trySparse(stream, s));
    s.tokens = 48;
    s.counts = 68;
    try std.testing.expect(!try k.trySparse(stream, s));
    s.counts = 64;
    s.cache.format = .bf16;
    try std.testing.expect(!try k.trySparse(stream, s));
    try std.testing.expectEqual(@as(usize, 0), Mock.calls);
}

test "only measured full prompt chunks launch either MMA operation" {
    var d: cuda.Driver = undefined;
    const k = try Mock.kernels(&d);
    const stream: cuda.Stream = .{ .d = &d, .handle = null };
    for ([_]usize{ 0, 1, 3, 16, 58, 65, 83, 2049 }) |rows| {
        var a: MmaArgs = .{ .input = 16, .weight = 32, .out = 48, .rows = rows, .heads = 32, .inner = 256, .columns = 512 };
        try std.testing.expect(!try k.tryAbsorb(stream, a));
        a.inner = 512;
        a.columns = 256;
        try std.testing.expect(!try k.tryExpand(stream, a));
    }
    try std.testing.expectEqual(@as(usize, 0), Mock.calls);
}

test "eligible prompt launch errors propagate without a second launch" {
    var d: cuda.Driver = undefined;
    const k = try Mock.kernels(&d);
    const stream: cuda.Stream = .{ .d = &d, .handle = null };
    Mock.status = cuda.abi.error_not_ready;
    try std.testing.expectError(error.NotReady, k.tryAbsorb(stream, .{ .input = 16, .weight = 32, .out = 48, .rows = 2048, .heads = 32, .inner = 256, .columns = 512 }));
    try std.testing.expectEqual(@as(usize, 1), Mock.calls);
}
