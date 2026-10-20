const std = @import("std");
const cuda = @import("cuda");
const abi = cuda.abi;

pub const row_values = 2048;
pub const row_bytes = row_values * 4;
pub const boundary_part_bytes = 4096;
pub const boundary_bytes = 2 * boundary_part_bytes;
pub const collective_count = 80;
const seam_count = 32;
const logit_bytes = 248320 * 2;
const floor_bytes = 1536 * 1024 * 1024;
const bank_bytes = (collective_count + 1) * row_bytes;

const Create = *const fn (*abi.Context, c_uint, abi.Device) callconv(.c) abi.Result;
const Destroy = *const fn (abi.Context) callconv(.c) abi.Result;
const CanPeer = *const fn (*c_int, abi.Device, abi.Device) callconv(.c) abi.Result;
const EnablePeer = *const fn (abi.Context, c_uint) callconv(.c) abi.Result;
const CopyPeer = *const fn (u64, abi.Context, u64, abi.Context, usize, abi.Stream) callconv(.c) abi.Result;
const Peer = struct {
    can: CanPeer,
    enable: EnablePeer,
    copy: CopyPeer,

    fn resolve(lib: anytype) ?Peer {
        return .{
            .can = lib.lookup(CanPeer, "cuDeviceCanAccessPeer") orelse return null,
            .enable = lib.lookup(EnablePeer, "cuCtxEnablePeerAccess") orelse return null,
            .copy = lib.lookup(CopyPeer, "cuMemcpyPeerAsync") orelse return null,
        };
    }
};

const Rank = struct {
    d: *const cuda.Driver,
    ctx: abi.Context,
    dev: abi.Device,
    destroy: Destroy,
    stream: cuda.Stream,
    banks: cuda.DeviceBuffer,
    scratch: cuda.DeviceBuffer,
    returned: cuda.DeviceBuffer,
    logits: cuda.DeviceBuffer,
    sum: cuda.rank_scatter.RankScatter,

    fn init(d: *const cuda.Driver, ordinal: c_int, create: Create, destroy: Destroy) !Rank {
        var dev: abi.Device = 0;
        try d.check(d.api.cuDeviceGet(&dev, ordinal), "cuDeviceGet");
        var ctx: abi.Context = null;
        try d.check(create(&ctx, 0, dev), "cuCtxCreate_v2");
        errdefer _ = destroy(ctx);
        const stream = try cuda.Stream.init(d, true);
        errdefer _ = d.api.cuStreamDestroy_v2(stream.handle);
        var banks = try cuda.DeviceBuffer.alloc(d, bank_bytes);
        errdefer banks.free();
        var scratch = try cuda.DeviceBuffer.alloc(d, 2 * row_bytes);
        errdefer scratch.free();
        var returned = try cuda.DeviceBuffer.alloc(d, bank_bytes);
        errdefer returned.free();
        var logits = try cuda.DeviceBuffer.alloc(d, logit_bytes);
        errdefer logits.free();
        return .{ .d = d, .ctx = ctx, .dev = dev, .destroy = destroy, .stream = stream, .banks = banks, .scratch = scratch, .returned = returned, .logits = logits, .sum = try cuda.rank_scatter.RankScatter.load(d) };
    }

    fn bind(r: Rank) !void {
        try r.d.check(r.d.api.cuCtxSetCurrent(r.ctx), "cuCtxSetCurrent");
    }

    fn sync(r: Rank) !void {
        try r.bind();
        try r.d.check(r.d.api.cuStreamSynchronize(r.stream.handle), "cuStreamSynchronize");
    }

    fn deinit(r: *Rank) void {
        defer _ = r.destroy(r.ctx);
        r.bind() catch return;
        _ = r.d.api.cuCtxSynchronize();
        r.sum.unload();
        r.logits.free();
        r.returned.free();
        r.scratch.free();
        r.banks.free();
        _ = r.d.api.cuStreamDestroy_v2(r.stream.handle);
    }

    fn upload(r: Rank, dst: u64, bytes: []const u8) !void {
        try r.bind();
        try r.d.check(r.d.api.cuMemcpyHtoDAsync_v2(dst, bytes.ptr, bytes.len, r.stream.handle), "cuMemcpyHtoDAsync_v2");
    }

    fn download(r: Rank, src: u64, bytes: []u8) !void {
        try r.bind();
        try r.d.check(r.d.api.cuMemcpyDtoHAsync_v2(bytes.ptr, src, bytes.len, r.stream.handle), "cuMemcpyDtoHAsync_v2");
    }

    fn poison(r: Rank, b: cuda.DeviceBuffer) !void {
        try r.bind();
        try r.d.check(r.d.api.cuMemsetD8Async(b.ptr, 0xff, b.len, r.stream.handle), "cuMemsetD8Async");
    }

    fn memory(r: Rank) !usize {
        try r.bind();
        var free: usize = 0;
        var total: usize = 0;
        try r.d.check(r.d.api.cuMemGetInfo_v2(&free, &total), "cuMemGetInfo_v2");
        if (free < floor_bytes) return error.GpuMemoryFloor;
        return free;
    }
};

fn now(io: std.Io) i96 {
    return std.Io.Clock.awake.now(io).toNanoseconds();
}

fn hash(bytes: []const u8) [64]u8 {
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &digest, .{});
    return std.fmt.bytesToHex(digest, .lower);
}

fn pattern(bytes: []u8, seed: usize, rank: usize) void {
    for (0..bytes.len / 4) |i| {
        const numerator: i32 = @as(i32, @intCast((i * 37 + seed * 101 + rank * 71) % 2003)) - 1001;
        const value: f32 = @as(f32, @floatFromInt(numerator)) / 1024;
        std.mem.writeInt(u32, bytes[i * 4 ..][0..4], @bitCast(value), .little);
    }
}

fn addRows(left: []const u8, right: []const u8, out: []u8) void {
    @setFloatMode(.strict);
    for (0..row_values) |i| {
        const a: f32 = @bitCast(std.mem.readInt(u32, left[i * 4 ..][0..4], .little));
        const b: f32 = @bitCast(std.mem.readInt(u32, right[i * 4 ..][0..4], .little));
        const sum: f32 = a + b;
        std.mem.writeInt(u32, out[i * 4 ..][0..4], @bitCast(sum), .little);
    }
}

fn same(actual: []const u8, expected: []const u8) !void {
    if (!std.mem.eql(u8, actual, expected)) return error.TransportMismatch;
}

pub fn checkInput(a: std.mem.Allocator) !void {
    const bytes = try a.alloc(u8, 3 * row_bytes);
    defer a.free(bytes);
    pattern(bytes[0..row_bytes], 1, 0);
    pattern(bytes[row_bytes..][0..row_bytes], 1, 1);
    addRows(bytes[0..row_bytes], bytes[row_bytes..][0..row_bytes], bytes[2 * row_bytes ..]);
    var poisoned: [row_bytes]u8 = undefined;
    @memcpy(&poisoned, bytes[2 * row_bytes ..]);
    poisoned[row_bytes - 1] ^= 1;
    if (same(&poisoned, bytes[2 * row_bytes ..])) |_| return error.TransportNegativeControlInsensitive else |err| if (err != error.TransportMismatch) return err;
}

fn floors(ranks: [2]Rank) ![2]usize {
    return .{ try ranks[0].memory(), try ranks[1].memory() };
}

const Method = enum { pinned_gpu_sum, cpu_sum, peer_gpu_sum };

fn collective(ranks: [2]Rank, host: [2]cuda.HostBuffer, method: Method, peer: ?Peer, step: usize) !void {
    const at = step * row_bytes;
    if (method == .peer_gpu_sum) {
        for (ranks, 0..) |r, rank| {
            const other = ranks[1 - rank];
            try r.bind();
            try r.d.check(peer.?.copy(r.scratch.ptr + (1 - rank) * row_bytes, r.ctx, other.banks.ptr + at, other.ctx, row_bytes, r.stream.handle), "cuMemcpyPeerAsync");
        }
        for (ranks) |r| try r.sync();
    } else {
        for (ranks, host) |r, h| try r.download(r.banks.ptr + at, h.bytes[0..row_bytes]);
        for (ranks) |r| try r.sync();
        if (method == .cpu_sum) {
            addRows(host[0].bytes[0..row_bytes], host[1].bytes[0..row_bytes], host[0].bytes[row_bytes..][0..row_bytes]);
            for (ranks) |r| try r.upload(r.banks.ptr + at + row_bytes, host[0].bytes[row_bytes..][0..row_bytes]);
        } else {
            for (ranks, 0..) |r, rank| try r.upload(r.scratch.ptr + (1 - rank) * row_bytes, host[1 - rank].bytes[0..row_bytes]);
        }
    }
    if (method != .cpu_sum) for (ranks, 0..) |r, rank| {
        try r.bind();
        try r.sum.rankSum(r.stream, r.scratch.ptr, r.banks.ptr + at, 2, rank, row_values, r.banks.ptr + at + row_bytes);
    };
    for (ranks) |r| try r.sync();
}

fn sumSample(a: std.mem.Allocator, io: std.Io, ranks: [2]Rank, host: [2]cuda.HostBuffer, method: Method, peer: ?Peer, sample: usize, journal: anytype) !i96 {
    const expected = try a.alloc(u8, bank_bytes);
    defer a.free(expected);
    for (ranks, host, 0..) |r, h, rank| {
        pattern(h.bytes[0..row_bytes], sample + 17, rank);
        try r.poison(r.banks);
        try r.poison(r.scratch);
        try r.upload(r.banks.ptr, h.bytes[0..row_bytes]);
        try r.sync();
    }
    const input_hashes = [2][64]u8{ hash(host[0].bytes[0..row_bytes]), hash(host[1].bytes[0..row_bytes]) };
    addRows(host[0].bytes[0..row_bytes], host[1].bytes[0..row_bytes], expected[row_bytes..][0..row_bytes]);
    for (2..collective_count + 1) |step| addRows(expected[(step - 1) * row_bytes ..][0..row_bytes], expected[(step - 1) * row_bytes ..][0..row_bytes], expected[step * row_bytes ..][0..row_bytes]);
    const before = try floors(ranks);
    const start = now(io);
    for (0..collective_count) |step| try collective(ranks, host, method, peer, step);
    const elapsed = now(io) - start;
    if (elapsed <= 0) return error.InvalidTransportClockInterval;
    const after = try floors(ranks);
    var output_hashes: [2][64]u8 = undefined;
    for (ranks, host, 0..) |r, h, rank| {
        try r.download(r.banks.ptr + row_bytes, h.bytes[0 .. collective_count * row_bytes]);
        try r.sync();
        try same(h.bytes[0 .. collective_count * row_bytes], expected[row_bytes..]);
        output_hashes[rank] = hash(h.bytes[0 .. collective_count * row_bytes]);
    }
    try journal.write(.{ .kind = "transport_sum_sample", .method = @tagName(method), .sample = sample, .warmup = sample == 0, .wall_ns = elapsed, .collectives = collective_count, .wall_ns_per_collective = @divTrunc(elapsed, collective_count), .row_bytes_per_rank = row_bytes, .rank_sum_launches = if (method == .cpu_sum) @as(usize, 0) else 2 * collective_count, .cpu_fp32_additions = if (method == .cpu_sum) collective_count * row_values else @as(usize, 0), .dtoh_calls = if (method == .peer_gpu_sum) @as(usize, 0) else 2 * collective_count, .htod_calls = if (method == .peer_gpu_sum) @as(usize, 0) else 2 * collective_count, .peer_copy_calls = if (method == .peer_gpu_sum) 2 * collective_count else @as(usize, 0), .all_output_banks_exact = true, .checked_bytes = 2 * collective_count * row_bytes, .input_sha256 = [2][]const u8{ &input_hashes[0], &input_hashes[1] }, .output_sha256 = [2][]const u8{ &output_hashes[0], &output_hashes[1] }, .free_before = before, .free_after = after });
    return elapsed;
}

const Direction = enum { forward, reverse, roundtrip };

fn seamLeg(src: Rank, dst: Rank, source: u64, target: u64, host: cuda.HostBuffer, peer: ?Peer) !void {
    if (peer) |p| {
        try dst.bind();
        for (0..2) |part| try dst.d.check(p.copy(target + part * boundary_part_bytes, dst.ctx, source + part * boundary_part_bytes, src.ctx, boundary_part_bytes, dst.stream.handle), "cuMemcpyPeerAsync");
        try dst.sync();
    } else {
        try src.bind();
        for (0..2) |part| try src.d.check(src.d.api.cuMemcpyDtoHAsync_v2(host.bytes.ptr + part * boundary_part_bytes, source + part * boundary_part_bytes, boundary_part_bytes, src.stream.handle), "cuMemcpyDtoHAsync_v2");
        try src.sync();
        try dst.bind();
        for (0..2) |part| try dst.d.check(dst.d.api.cuMemcpyHtoDAsync_v2(target + part * boundary_part_bytes, host.bytes.ptr + part * boundary_part_bytes, boundary_part_bytes, dst.stream.handle), "cuMemcpyHtoDAsync_v2");
        try dst.sync();
    }
}

fn seamSample(io: std.Io, ranks: [2]Rank, host: [2]cuda.HostBuffer, peer: ?Peer, direction: Direction, sample: usize, journal: anytype) !i96 {
    const source_rank: usize = if (direction == .reverse) 1 else 0;
    const src = ranks[source_rank];
    const dst = ranks[1 - source_rank];
    const n = seam_count * boundary_bytes;
    for (0..seam_count) |bank| pattern(host[1].bytes[bank * boundary_bytes ..][0..boundary_bytes], sample * seam_count + bank + 41, source_rank);
    const input_hash = hash(host[1].bytes[0..n]);
    try src.upload(src.banks.ptr, host[1].bytes[0..n]);
    try src.poison(src.returned);
    try dst.poison(dst.returned);
    for (ranks) |r| try r.sync();
    const before = try floors(ranks);
    const start = now(io);
    for (0..seam_count) |bank| {
        const at = bank * boundary_bytes;
        try seamLeg(src, dst, src.banks.ptr + at, dst.returned.ptr + at, host[0], peer);
        if (direction == .roundtrip) try seamLeg(dst, src, dst.returned.ptr + at, src.returned.ptr + at, host[0], peer);
    }
    const elapsed = now(io) - start;
    if (elapsed <= 0) return error.InvalidTransportClockInterval;
    const after = try floors(ranks);
    try dst.download(dst.returned.ptr, host[0].bytes[0..n]);
    try dst.sync();
    try same(host[0].bytes[0..n], host[1].bytes[0..n]);
    if (direction == .roundtrip) {
        try src.download(src.returned.ptr, host[0].bytes[0..n]);
        try src.sync();
        try same(host[0].bytes[0..n], host[1].bytes[0..n]);
    }
    const output_hash = hash(host[0].bytes[0..n]);
    const legs: usize = if (direction == .roundtrip) 2 else 1;
    try journal.write(.{ .kind = "transport_boundary_sample", .method = if (peer != null) "peer" else "pinned_host", .direction = @tagName(direction), .sample = sample, .warmup = sample == 0, .wall_ns = elapsed, .operations = seam_count, .wall_ns_per_operation = @divTrunc(elapsed, seam_count), .payload_bytes_per_leg = boundary_bytes, .logical_copy_bytes_per_leg = boundary_bytes * @as(usize, if (peer == null) 2 else 1), .legs_per_operation = legs, .all_bytes_exact = true, .input_sha256 = &input_hash, .output_sha256 = &output_hash, .free_before = before, .free_after = after });
    return elapsed;
}

fn summary(journal: anytype, name: []const u8, values: [5]i96, count: usize) !void {
    var sorted = values;
    std.mem.sort(i96, &sorted, {}, std.sort.asc(i96));
    try journal.write(.{ .kind = "transport_summary", .case = name, .samples_ns = values, .median_ns = sorted[2], .min_ns = sorted[0], .max_ns = sorted[4], .operations = count, .median_ns_per_operation = @divTrunc(sorted[2], @as(i96, @intCast(count))) });
}

fn overhead(io: std.Io, ranks: [2]Rank, host: [2]cuda.HostBuffer, journal: anytype) !void {
    var copies: [2][5]i96 = undefined;
    var queries: [5]i96 = undefined;
    for (0..6) |sample| {
        const before = try floors(ranks);
        const query_start = now(io);
        _ = try floors(ranks);
        const query_ns = now(io) - query_start;
        if (query_ns <= 0) return error.InvalidTransportClockInterval;
        if (sample > 0) queries[sample - 1] = query_ns;
        for (ranks, 0..) |r, rank| {
            pattern(host[0].bytes[0..logit_bytes], sample + 991, rank);
            try r.upload(r.logits.ptr, host[0].bytes[0..logit_bytes]);
            try r.sync();
            @memset(host[1].bytes[0..logit_bytes], 0xff);
            const start = now(io);
            try r.download(r.logits.ptr, host[1].bytes[0..logit_bytes]);
            try r.sync();
            const elapsed = now(io) - start;
            if (elapsed <= 0) return error.InvalidTransportClockInterval;
            try same(host[0].bytes[0..logit_bytes], host[1].bytes[0..logit_bytes]);
            if (sample > 0) copies[rank][sample - 1] = elapsed;
            try journal.write(.{ .kind = "transport_host_overhead_sample", .sample = sample, .warmup = sample == 0, .rank = rank, .logit_copy_bytes = logit_bytes, .logit_copy_wall_ns = elapsed, .two_meminfo_wall_ns = query_ns, .all_bytes_exact = true, .free_before = before, .free_after = try floors(ranks) });
        }
    }
    try summary(journal, "rank0_final_bf16_logit_download", copies[0], 1);
    try summary(journal, "rank1_final_bf16_logit_download", copies[1], 1);
    try summary(journal, "two_context_meminfo_queries", queries, 1);
}

pub fn run(a: std.mem.Allocator, io: std.Io, ordinals: [2]c_int, journal: anytype) !void {
    if (ordinals[0] == ordinals[1]) return error.TransportRequiresTwoDevices;
    try checkInput(a);
    var d = try cuda.Driver.open();
    defer d.close();
    const create = d.lib.lookup(Create, "cuCtxCreate_v2") orelse return error.MissingPrivateContextSymbol;
    const destroy = d.lib.lookup(Destroy, "cuCtxDestroy_v2") orelse return error.MissingPrivateContextSymbol;
    var old: abi.Context = null;
    try d.check(d.api.cuCtxGetCurrent(&old), "cuCtxGetCurrent");
    defer _ = d.api.cuCtxSetCurrent(old);
    var ranks: [2]Rank = undefined;
    ranks[0] = try Rank.init(&d, ordinals[0], create, destroy);
    defer ranks[0].deinit();
    ranks[1] = try Rank.init(&d, ordinals[1], create, destroy);
    defer ranks[1].deinit();
    var host = [2]cuda.HostBuffer{ try cuda.HostBuffer.alloc(&d, bank_bytes), undefined };
    defer host[0].free();
    host[1] = try cuda.HostBuffer.alloc(&d, bank_bytes);
    defer host[1].free();
    var peer = Peer.resolve(&d.lib);
    const symbols_available = peer != null;
    var capability: [2]c_int = .{ 0, 0 };
    var enable_results: [2]c_int = .{ -1, -1 };
    if (peer) |p| {
        for (ranks, 0..) |r, rank| try d.check(p.can(&capability[rank], r.dev, ranks[1 - rank].dev), "cuDeviceCanAccessPeer");
        if (capability[0] != 0 and capability[1] != 0) {
            for (ranks, 0..) |r, rank| {
                try r.bind();
                enable_results[rank] = p.enable(ranks[1 - rank].ctx, 0);
            }
            if (enable_results[0] != 0 or enable_results[1] != 0) peer = null;
        } else peer = null;
    }
    try journal.write(.{ .kind = "transport_manifest", .ordinals = ordinals, .private_contexts = true, .peer_symbols_available = symbols_available, .peer_capability = capability, .peer_enable_results = enable_results, .peer_available = peer != null, .rank_sum_function = "tf_rank_sum", .rank_sum_bindings = [2]usize{ @intFromPtr(ranks[0].sum.sum.handle), @intFromPtr(ranks[1].sum.sum.handle) }, .negative_control = "last output byte mutation rejected", .boundary_parts = 2, .boundary_part_bytes = boundary_part_bytes, .collectives_per_sample = collective_count, .sum_row_bytes_per_rank = row_bytes, .warmups = 1, .measured_samples = 5, .memory_floor_bytes = floor_bytes, .free_bytes = try floors(ranks), .question = "Does host-coordinated transport fit within an optimistic approximately 4 ms/token serial-halving budget?", .scope = "host-coordinated primitives only; 80 sums assumes two reductions per layer across 40 layers; no proof of actual TP design", .projection = "removed serial compute and host work minus shard compute minus 80 times collective cost minus routing and coordination; asymmetric cards and reduced expert CTAs require measurement", .dma_accounting = "logical copy bytes, not measured physical bandwidth" });
    for (std.meta.tags(Direction)) |direction| {
        var times: [2][5]i96 = undefined;
        for (0..6) |sample| for (0..if (peer != null) @as(usize, 2) else 1) |turn| {
            const arm = if (peer != null and sample % 2 == 1) 1 - turn else turn;
            const elapsed = try seamSample(io, ranks, host, if (arm == 1) peer else null, direction, sample, journal);
            if (sample > 0) times[arm][sample - 1] = elapsed;
        };
        const name = try std.fmt.allocPrint(a, "pinned_boundary_{s}", .{@tagName(direction)});
        defer a.free(name);
        try summary(journal, name, times[0], seam_count);
        if (peer != null) {
            const peer_name = try std.fmt.allocPrint(a, "peer_boundary_{s}", .{@tagName(direction)});
            defer a.free(peer_name);
            try summary(journal, peer_name, times[1], seam_count);
        }
    }
    var sum_times: [3][5]i96 = undefined;
    const methods = [_]Method{ .pinned_gpu_sum, .cpu_sum, .peer_gpu_sum };
    const method_count: usize = if (peer != null) 3 else 2;
    for (0..6) |sample| for (0..method_count) |turn| {
        const arm = if (sample % 2 == 1) method_count - 1 - turn else turn;
        const elapsed = try sumSample(a, io, ranks, host, methods[arm], peer, sample, journal);
        if (sample > 0) sum_times[arm][sample - 1] = elapsed;
    };
    for (methods[0..method_count], 0..) |method, arm| try summary(journal, @tagName(method), sum_times[arm], collective_count);
    try overhead(io, ranks, host, journal);
    try journal.write(.{ .kind = "transport_complete", .all_bytes_exact = true, .free_bytes = try floors(ranks), .model_loaded = false });
}

test "transport geometry and deterministic rank-order reference detect final-byte corruption" {
    try std.testing.expectEqual(@as(usize, 8192), row_bytes);
    try std.testing.expectEqual(@as(usize, 8192), boundary_bytes);
    try std.testing.expectEqual(@as(usize, 496640), logit_bytes);
    try checkInput(std.testing.allocator);
    var a: [row_bytes]u8 = undefined;
    var b: [row_bytes]u8 = undefined;
    var sum: [row_bytes]u8 = undefined;
    pattern(&a, 19, 0);
    pattern(&b, 19, 1);
    try std.testing.expect(!std.mem.eql(u8, &a, &b));
    addRows(&a, &b, &sum);
    for (0..row_values) |i| {
        const av: f32 = @bitCast(std.mem.readInt(u32, a[i * 4 ..][0..4], .little));
        const bv: f32 = @bitCast(std.mem.readInt(u32, b[i * 4 ..][0..4], .little));
        try std.testing.expectEqual(@as(u32, @bitCast(av + bv)), std.mem.readInt(u32, sum[i * 4 ..][0..4], .little));
    }
    const first = hash(&a);
    pattern(&a, 20, 0);
    try std.testing.expect(!std.mem.eql(u8, &first, &hash(&a)));
    for (0..collective_count - 1) |_| addRows(&sum, &sum, &sum);
    for (0..row_values) |i| try std.testing.expect(std.math.isFinite(@as(f32, @bitCast(std.mem.readInt(u32, sum[i * 4 ..][0..4], .little)))));
}

test "missing optional peer symbols leaves host methods available" {
    const Missing = struct {
        fn lookup(_: @This(), comptime T: type, _: [:0]const u8) ?T {
            return null;
        }
    };
    try std.testing.expect(Peer.resolve(Missing{}) == null);
}
