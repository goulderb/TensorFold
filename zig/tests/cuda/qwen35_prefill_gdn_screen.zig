const std = @import("std");
const cuda = @import("cuda");
const qwen = @import("qwen35_cuda");
const check = @import("check.zig");
const Engine = qwen.shared_engine.Engine;
const Seg = @typeInfo(@TypeOf(qwen.shared_engine.Cache.seg)).@"fn".return_type.?;
const Layout = qwen.shared_engine.Cache.GdnLayout;
const guard_bytes = 256;
const max_rows = 17;
const floor_bytes = 1536 << 20;

fn hash(bytes: []const u8) [64]u8 {
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &digest, .{});
    return std.fmt.bytesToHex(digest, .lower);
}

fn save(gpu: check.Gpu, journal: anytype, name: []const u8, bytes: []const u8) !void {
    const path = try std.fs.path.join(gpu.gpa, &.{ journal.directory, name });
    defer gpu.gpa.free(path);
    const file = try std.Io.Dir.cwd().createFile(gpu.io, path, .{ .exclusive = true });
    defer file.close(gpu.io);
    try file.writeStreamingAll(gpu.io, bytes);
}

fn finite(bytes: []const u8, stride: usize) !void {
    if (bytes.len % stride != 0) return error.InvalidGdnScalarBytes;
    var i: usize = 0;
    while (i < bytes.len) : (i += stride) {
        const bits = if (stride == 4) std.mem.readInt(u32, bytes[i..][0..4], .little) else @as(u32, std.mem.readInt(u16, bytes[i..][0..2], .little)) << 16;
        if (!std.math.isFinite(@as(f32, @bitCast(bits)))) return error.NonfiniteGdnOutput;
    }
}

fn distinct(bytes: []const u8, rows: usize) !void {
    const width = bytes.len / rows;
    for (0..rows) |i| for (0..i) |j| {
        if (std.mem.eql(u8, bytes[i * width ..][0..width], bytes[j * width ..][0..width])) return error.IdenticalGdnRows;
    };
}

fn inputBits(row: usize, column: usize, slot: usize) u16 {
    const centered: i32 = @as(i32, @intCast((column * 19 + row * 37 + slot * 11) % 127)) - 63;
    const f: f32 = @as(f32, @floatFromInt(centered)) / 128.0 + @as(f32, @floatFromInt(row)) / 64.0;
    const bits: u32 = @bitCast(f);
    return @truncate(bits >> 16);
}

const Guarded = struct {
    buffer: cuda.DeviceBuffer,
    bytes: usize,

    fn init(gpu: check.Gpu, bytes: usize) !Guarded {
        var buffer = try cuda.DeviceBuffer.alloc(gpu.d, bytes + 2 * guard_bytes);
        errdefer buffer.free();
        try buffer.fill8(0xa5, null);
        return .{ .buffer = buffer, .bytes = bytes };
    }
    fn ptr(b: Guarded) u64 {
        return b.buffer.ptr + guard_bytes;
    }
    fn poison(b: Guarded) !void {
        try b.buffer.fill8(0xa5, null);
        const payload: cuda.DeviceBuffer = .{ .d = b.buffer.d, .ptr = b.ptr(), .len = b.bytes, .borrowed = true };
        try payload.fill8(0xff, null);
    }
    fn upload(b: Guarded, bytes: []const u8) !void {
        if (bytes.len != b.bytes) return error.InvalidGdnUpload;
        try b.buffer.upload(guard_bytes, bytes);
    }
    fn readWhole(b: Guarded, gpu: check.Gpu) ![]u8 {
        const all = try check.download(gpu, b.buffer);
        errdefer gpu.gpa.free(all);
        try b.guards(all);
        return all;
    }
    fn guards(b: Guarded, all: []const u8) !void {
        if (all.len != b.bytes + 2 * guard_bytes) return error.InvalidGdnReadback;
        for (all[0..guard_bytes]) |v| if (v != 0xa5) return error.GdnLeadingGuard;
        for (all[guard_bytes + b.bytes ..]) |v| if (v != 0xa5) return error.GdnTrailingGuard;
    }
    fn read(b: Guarded, gpu: check.Gpu) ![]u8 {
        const all = try b.readWhole(gpu);
        defer gpu.gpa.free(all);
        return gpu.gpa.dupe(u8, all[guard_bytes..][0..b.bytes]);
    }
    fn deinit(b: *Guarded) void {
        b.buffer.free();
    }
};

const Inputs = struct {
    qkv: Guarded,
    a: Guarded,
    b: Guarded,
    z: Guarded,

    fn init(gpu: check.Gpu, e: *Engine, journal: anytype) !Inputs {
        var result: Inputs = undefined;
        var made: usize = 0;
        errdefer inline for (@typeInfo(Inputs).@"struct".field_names, 0..) |field, i| {
            if (i < made) @field(result, field).deinit();
        };
        const widths = [_]usize{ e.g.convWidth(), e.g.v_heads, e.g.v_heads, e.g.valueWidth() };
        inline for (@typeInfo(Inputs).@"struct".field_names, 0..) |field, index| {
            const width = widths[index];
            @field(result, field) = try Guarded.init(gpu, max_rows * width * 2);
            made += 1;
            const host = try gpu.gpa.alloc(u16, max_rows * width);
            defer gpu.gpa.free(host);
            for (host, 0..) |*v, i| {
                v.* = inputBits(i / width, i % width, index);
            }
            const bytes = std.mem.sliceAsBytes(host);
            try finite(bytes, 2);
            try distinct(bytes, max_rows);
            try @field(result, field).upload(bytes);
            const file = try std.fmt.allocPrint(gpu.gpa, "gdn-{d}-input-{s}.bin", .{ gpu.ctx.device_ordinal, field });
            defer gpu.gpa.free(file);
            try save(gpu, journal, file, bytes);
            try journal.write(.{ .kind = "prefill_gdn_input", .ordinal = gpu.ctx.device_ordinal, .name = field, .rows = max_rows, .width = width, .bytes = bytes.len, .sha256 = &hash(bytes), .file = file, .synthetic = true, .finite = true, .all_rows_distinct = true });
        }
        return result;
    }
    fn verify(in: *Inputs, gpu: check.Gpu, journal: anytype) !void {
        inline for (@typeInfo(Inputs).@"struct".field_names, 0..) |field, slot| {
            const b = @field(in, field);
            const bytes = try b.read(gpu);
            defer gpu.gpa.free(bytes);
            const width = bytes.len / (2 * max_rows);
            for (0..bytes.len / 2) |i| {
                if (std.mem.readInt(u16, bytes[i * 2 ..][0..2], .little) != inputBits(i / width, i % width, slot)) return error.GdnInputMutation;
            }
            const file = try std.fmt.allocPrint(gpu.gpa, "gdn-{d}-input-final-{s}.bin", .{ gpu.ctx.device_ordinal, field });
            defer gpu.gpa.free(file);
            try save(gpu, journal, file, bytes);
            try journal.write(.{ .kind = "prefill_gdn_input_unchanged", .ordinal = gpu.ctx.device_ordinal, .name = field, .bytes = bytes.len, .sha256 = &hash(bytes), .file = file, .guards = true, .bit_exact = true });
        }
    }
    fn deinit(in: *Inputs) void {
        inline for (@typeInfo(Inputs).@"struct".field_names) |field| @field(in, field).deinit();
    }
};

const Work = struct {
    q: Guarded,
    k: Guarded,
    v: Guarded,
    g: Guarded,
    beta: Guarded,
    y: Guarded,
    post: Guarded,
    state: Guarded,

    fn init(gpu: check.Gpu, e: *Engine) !Work {
        var out: Work = undefined;
        var made: usize = 0;
        errdefer inline for (@typeInfo(Work).@"struct".field_names, 0..) |field, i| {
            if (i < made) @field(out, field).deinit();
        };
        const sizes = [_]usize{ max_rows * e.g.k_heads * 128 * 4, max_rows * e.g.k_heads * 128 * 4, max_rows * e.g.valueWidth() * 2, max_rows * e.g.v_heads * 4, max_rows * e.g.v_heads * 2, max_rows * e.g.valueWidth() * 4, max_rows * e.g.valueWidth() * 2, e.g.stateBytes() };
        inline for (@typeInfo(Work).@"struct".field_names, 0..) |field, i| {
            @field(out, field) = try Guarded.init(gpu, sizes[i]);
            made += 1;
        }
        return out;
    }
    fn reset(w: *Work, state: []const u8) !void {
        inline for (@typeInfo(Work).@"struct".field_names) |field| try @field(w, field).poison();
        try w.state.upload(state);
    }
    fn deinit(w: *Work) void {
        inline for (@typeInfo(Work).@"struct".field_names) |field| @field(w, field).deinit();
    }
};

fn initialState(gpu: check.Gpu, e: *Engine, layout: Layout) ![]u8 {
    const bytes = try gpu.gpa.alloc(u8, e.g.stateBytes());
    errdefer gpu.gpa.free(bytes);
    for (0..e.g.v_heads) |h| for (0..128) |key| for (0..128) |value| {
        const canonical = (h * 128 + value) * 128 + key;
        const physical = if (layout == .value_key) canonical else (h * 128 + key) * 128 + value;
        const centered: i32 = @as(i32, @intCast((canonical * 13 + 7) % 251)) - 125;
        const f: f32 = @as(f32, @floatFromInt(centered)) / 16384.0;
        std.mem.writeInt(u32, bytes[physical * 4 ..][0..4], @bitCast(f), .little);
    };
    for (0..e.g.convBytes() / 2) |i| {
        const centered: i32 = @as(i32, @intCast((i * 7 + 31) % 127)) - 63;
        const f: f32 = @as(f32, @floatFromInt(centered)) / 256.0;
        const bits: u32 = @bitCast(f);
        std.mem.writeInt(u16, bytes[e.g.recBytes() + i * 2 ..][0..2], @truncate(bits >> 16), .little);
    }
    return bytes;
}

const Spec = struct { function: cuda.Function, cfg: cuda.Config, args: cuda.Args };
const Plan = struct {
    specs: [68]Spec = undefined,
    count: usize = 0,

    fn add(p: *Plan, f: cuda.Function, grid: [3]usize, threads: u32, values: anytype) void {
        var args: cuda.Args = .{};
        inline for (values) |v| args.add(v);
        p.specs[p.count] = .{ .function = f, .cfg = .{ .grid = .{ .x = @intCast(grid[0]), .y = @intCast(grid[1]), .z = @intCast(grid[2]) }, .block = .{ .x = threads, .y = 1, .z = 1 } }, .args = args };
        p.count += 1;
    }
    fn graph(p: *Plan, gpu: check.Gpu, journal: anytype, layout: Layout, rows: usize, arm: []const u8) !cuda.graph.Exec {
        var g = try cuda.graph.Graph.init(gpu.d);
        defer g.deinit();
        var nodes: [68]cuda.graph.Node = undefined;
        var lib = gpu.d.lib;
        const dependencies = lib.lookup(*const fn (cuda.graph.Node, ?[*]cuda.graph.Node, *usize) callconv(.c) cuda.abi.Result, "cuGraphNodeGetDependencies") orelse return error.MissingSymbol;
        const name = lib.lookup(*const fn (*?[*:0]const u8, cuda.abi.Function) callconv(.c) cuda.abi.Result, "cuFuncGetName") orelse return error.MissingSymbol;
        for (p.specs[0..p.count], 0..) |*s, i| {
            nodes[i] = try g.addKernel(if (i == 0) &.{} else nodes[i - 1 .. i], s.function, s.cfg, &s.args);
            var actual: cuda.abi.KernelNodeParams = undefined;
            try gpu.d.check(gpu.d.api.cuGraphKernelNodeGetParams_v2(nodes[i], &actual), "cuGraphKernelNodeGetParams");
            if (actual.func != s.function.handle or actual.grid_x != s.cfg.grid.x or actual.grid_y != s.cfg.grid.y or actual.grid_z != s.cfg.grid.z or actual.block_x != s.cfg.block.x or actual.block_y != 1 or actual.block_z != 1 or actual.shared_bytes != 0) return error.GdnGraphGeometry;
            const args = actual.params orelse return error.GdnGraphArguments;
            var sha = std.crypto.hash.sha2.Sha256.init(.{});
            for (0..s.args.count) |a| {
                const got: [*]const u8 = @ptrCast(args[a] orelse return error.GdnGraphArguments);
                const want = s.args.storage[s.args.offsets[a]..][0..s.args.sizes[a]];
                if (!std.mem.eql(u8, got[0..want.len], want)) return error.GdnGraphArguments;
                sha.update(want);
            }
            var n: usize = 1;
            var parent: [1]cuda.graph.Node = undefined;
            try gpu.d.check(dependencies(nodes[i], &parent, &n), "cuGraphNodeGetDependencies");
            if (n != @intFromBool(i > 0) or (i > 0 and parent[0] != nodes[i - 1])) return error.GdnGraphDependency;
            var symbol: ?[*:0]const u8 = null;
            try gpu.d.check(name(&symbol, actual.func), "cuFuncGetName");
            var digest: [32]u8 = undefined;
            sha.final(&digest);
            const hex = std.fmt.bytesToHex(digest, .lower);
            try journal.write(.{ .kind = "prefill_gdn_node", .ordinal = gpu.ctx.device_ordinal, .layout = @tagName(layout), .rows = rows, .arm = arm, .node = i, .symbol = std.mem.span(symbol orelse return error.MissingSymbol), .grid = s.cfg.grid, .block = s.cfg.block, .argument_count = s.args.count, .argument_sha256 = &hex, .dependency_count = n, .abi_exact = true });
        }
        if (try g.nodeCount() != p.count) return error.GdnGraphNodeCount;
        return g.instantiate();
    }
};

const Tables = struct {
    segs: cuda.DeviceBuffer,
    parents: cuda.DeviceBuffer,
    rowseg: cuda.DeviceBuffer,
    params: cuda.DeviceBuffer,

    fn init(gpu: check.Gpu, e: *Engine, w: *Work, rows: usize, batch: bool, layout: Layout, journal: anytype) !Tables {
        var segs: [max_rows + 1]Seg = undefined;
        var parents: [max_rows]c_int = @splat(-1);
        const rowseg: [max_rows]c_int = @splat(0);
        for (0..rows + 1) |r| {
            segs[r] = .{ .rec = w.state.ptr(), .conv = w.state.ptr() + e.g.recBytes(), .kv = 0, .tap = 0, .log = 0, .cap = 64, .len = @intCast(7 + r), .row0 = @intCast(if (batch and r < rows) r else 0), .rows = 1, .tap_rows = 0, .taps = 0, .delta = 0 };
            if (batch and r < rows) parents[r] = @as(c_int, @intCast(r)) - 1;
        }
        segs[rows + 1] = segs[0];
        segs[rows + 1].rows = @intCast(rows);
        const params = [_]c_int{ 0, 0, @intCast(if (batch) rows else 1), 1, 0, 0, 0, 0 };
        var t: Tables = undefined;
        t.segs = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(segs[0 .. rows + 2]));
        errdefer t.segs.free();
        t.parents = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.asBytes(&parents));
        errdefer t.parents.free();
        t.rowseg = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.asBytes(&rowseg));
        errdefer t.rowseg.free();
        t.params = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.asBytes(&params));
        errdefer t.params.free();
        const buffers = [_]cuda.DeviceBuffer{ t.segs, t.parents, t.rowseg, t.params };
        const wanted = [_][]const u8{ std.mem.sliceAsBytes(segs[0 .. rows + 2]), std.mem.asBytes(&parents), std.mem.asBytes(&rowseg), std.mem.asBytes(&params) };
        for (buffers, wanted, 0..) |b, bytes, i| {
            const got = try check.download(gpu, b);
            defer gpu.gpa.free(got);
            try check.sameBytes("GDN table upload", got, bytes);
            const file = try std.fmt.allocPrint(gpu.gpa, "gdn-{d}-{s}-{d}-{s}-table-{d}.bin", .{ gpu.ctx.device_ordinal, @tagName(layout), rows, if (batch) "batch" else "serial", i });
            defer gpu.gpa.free(file);
            try save(gpu, journal, file, got);
            try journal.write(.{ .kind = "prefill_gdn_table", .ordinal = gpu.ctx.device_ordinal, .layout = @tagName(layout), .batch = batch, .rows = rows, .table = i, .bytes = bytes.len, .sha256 = &hash(bytes), .file = file, .upload_exact = true });
        }
        try journal.write(.{ .kind = "prefill_gdn_segment_views", .ordinal = gpu.ctx.device_ordinal, .layout = @tagName(layout), .batch = batch, .rows = rows, .segments = segs[0 .. rows + 2], .parents = parents, .params = params });
        return t;
    }
    fn deinit(t: *Tables) void {
        t.params.free();
        t.rowseg.free();
        t.parents.free();
        t.segs.free();
    }
};

fn appendOps(plan: *Plan, e: *Engine, in: *const Inputs, w: *Work, t: Tables, layout: Layout, rows: usize, batch: bool, first: usize, count: usize) void {
    const m = e.layers[0].mixer.delta;
    const chain = if (layout == .value_key) e.fns.gdn_chain else e.modelopt.?.gdn_chain_transposed;
    const geo = e.geo;
    const layer: c_int = 0;
    const cw = e.g.convWidth();
    const vw = e.g.valueWidth();
    const kw = e.g.k_heads * 128;
    const iterations = if (batch) 1 else count;
    for (0..iterations) |i| {
        const r = first + i;
        const n = if (batch) count else 1;
        const seg = t.segs.ptr + (if (batch) rows + 1 else r) * @sizeOf(Seg);
        const p = t.params.ptr;
        plan.add(e.fns.gdn_pre, .{ 1, 2 * e.g.k_heads + e.g.v_heads, n }, 128, .{ geo, in.qkv.ptr() + r * cw * 2, seg, t.rowseg.ptr, t.parents.ptr, layer, m.conv, in.a.ptr() + r * e.g.v_heads * 2, in.b.ptr() + r * e.g.v_heads * 2, m.a_log, m.dt_bias, w.q.ptr() + r * kw * 4, w.k.ptr() + r * kw * 4, w.v.ptr() + r * vw * 2, w.g.ptr() + r * e.g.v_heads * 4, w.beta.ptr() + r * e.g.v_heads * 2, p });
        plan.add(e.fns.conv_commit, .{ (cw + 255) / 256, 1, 1 }, 256, .{ geo, in.qkv.ptr() + r * cw * 2, seg, t.parents.ptr, layer, p });
        for (0..n) |j| {
            const chain_seg = t.segs.ptr + (r + j) * @sizeOf(Seg);
            plan.add(chain, .{ 1, 1, e.g.v_heads }, 128, .{ geo, w.q.ptr() + (if (batch) 0 else r * kw * 4), w.k.ptr() + (if (batch) 0 else r * kw * 4), w.v.ptr() + (if (batch) 0 else r * vw * 2), w.g.ptr() + (if (batch) 0 else r * e.g.v_heads * 4), w.beta.ptr() + (if (batch) 0 else r * e.g.v_heads * 2), chain_seg, t.parents.ptr, layer, w.y.ptr() + (if (batch) 0 else r * vw * 4), p });
        }
        plan.add(e.fns.gdn_post, .{ 1, e.g.v_heads, n }, 128, .{ geo, w.y.ptr() + r * vw * 4, in.z.ptr() + r * vw * 2, m.norm, e.g.eps, w.post.ptr() + r * vw * 2, p });
    }
}

fn compare(gpu: check.Gpu, serial: *Work, batch: *Work, active: usize, rec_bytes: usize, journal: anytype, layout: Layout, rows: usize, continuation: bool) !void {
    inline for (@typeInfo(Work).@"struct".field_names) |field| {
        const a = @field(serial, field);
        const b = @field(batch, field);
        const serial_raw = try check.download(gpu, a.buffer);
        defer gpu.gpa.free(serial_raw);
        const batch_raw = try check.download(gpu, b.buffer);
        defer gpu.gpa.free(batch_raw);
        const want = serial_raw[guard_bytes..][0..a.bytes];
        const got = batch_raw[guard_bytes..][0..b.bytes];
        const state = comptime std.mem.eql(u8, field, "state");
        const n = if (state) a.bytes else a.bytes / max_rows * active;
        const serial_file = try std.fmt.allocPrint(gpu.gpa, "gdn-{d}-{s}-{d}-{s}-{s}-serial.bin", .{ gpu.ctx.device_ordinal, @tagName(layout), rows, if (continuation) "continuation" else "prefix", field });
        defer gpu.gpa.free(serial_file);
        const batch_file = try std.fmt.allocPrint(gpu.gpa, "gdn-{d}-{s}-{d}-{s}-{s}-batch.bin", .{ gpu.ctx.device_ordinal, @tagName(layout), rows, if (continuation) "continuation" else "prefix", field });
        defer gpu.gpa.free(batch_file);
        try save(gpu, journal, serial_file, serial_raw);
        try save(gpu, journal, batch_file, batch_raw);
        try journal.write(.{ .kind = "prefill_gdn_readback", .ordinal = gpu.ctx.device_ordinal, .layout = @tagName(layout), .rows = rows, .continuation = continuation, .buffer = field, .bytes = got.len, .active_bytes = n, .serial_file = serial_file, .batch_file = batch_file, .raw_bytes = serial_raw.len, .serial_raw_sha256 = &hash(serial_raw), .batch_raw_sha256 = &hash(batch_raw), .guard_bytes = guard_bytes });
        try a.guards(serial_raw);
        try b.guards(batch_raw);
        try check.sameBytes("GDN serial/batch " ++ field, got, want);
        const fp32 = comptime std.mem.eql(u8, field, "q") or std.mem.eql(u8, field, "k") or std.mem.eql(u8, field, "g") or std.mem.eql(u8, field, "y");
        if (!state) {
            try finite(got[0..n], if (fp32) 4 else 2);
            for (got[n..]) |v| if (v != 0xff) return error.GdnInactiveWrite;
        } else {
            try finite(got[0..rec_bytes], 4);
            try finite(got[rec_bytes..], 2);
        }
        if (comptime std.mem.eql(u8, field, "y") or std.mem.eql(u8, field, "post")) try distinct(got[0..n], active);
        try journal.write(.{ .kind = "prefill_gdn_parity", .ordinal = gpu.ctx.device_ordinal, .layout = @tagName(layout), .rows = rows, .continuation = continuation, .buffer = field, .bytes = got.len, .active_bytes = n, .sha256 = &hash(got), .serial_file = serial_file, .batch_file = batch_file, .raw_bytes = serial_raw.len, .serial_raw_sha256 = &hash(serial_raw), .batch_raw_sha256 = &hash(batch_raw), .guard_bytes = guard_bytes, .bit_exact = true, .guards = true, .inactive_poison = !state, .finite_active = true });
    }
}

pub fn run(gpu: check.Gpu, model_dir: []const u8, journal: anytype) !void {
    if ((try gpu.ctx.memInfo()).free < floor_bytes) return error.GpuMemoryFloor;
    const config = try qwen.checkpoint.Config.read(gpu.gpa, gpu.io, model_dir);
    const admitted: qwen.checkpoint.Admitted = .{ .geometry = config.geometry, .encoding = .{ .modelopt = config } };
    const e = try Engine.initPlaced(gpu.gpa, gpu.io, gpu.ctx, model_dir, admitted, .{ .first = 1, .end = 2, .total = config.geometry.layers }, false);
    defer e.deinit();
    if (e.layers.len != 1 or e.layers[0].mixer != .delta or e.g.deltas() != 1) return error.ExpectedOneGdnLayer;
    if ((try gpu.ctx.memInfo()).free < floor_bytes) return error.GpuMemoryFloor;
    const m = e.layers[0].mixer.delta;
    const coefficients = [_]struct { name: []const u8, ptr: u64, bytes: usize }{
        .{ .name = "conv", .ptr = m.conv, .bytes = e.g.convWidth() * 4 * 4 },
        .{ .name = "norm", .ptr = m.norm, .bytes = 128 * 4 },
        .{ .name = "a_log", .ptr = m.a_log, .bytes = e.g.v_heads * 4 },
        .{ .name = "dt_bias", .ptr = m.dt_bias, .bytes = e.g.v_heads * 4 },
    };
    for (coefficients) |c| {
        const b: cuda.DeviceBuffer = .{ .d = gpu.d, .ptr = c.ptr, .len = c.bytes, .borrowed = true };
        const bytes = try check.download(gpu, b);
        defer gpu.gpa.free(bytes);
        try finite(bytes, 4);
        const file = try std.fmt.allocPrint(gpu.gpa, "gdn-{d}-coefficient-{s}.bin", .{ gpu.ctx.device_ordinal, c.name });
        defer gpu.gpa.free(file);
        try save(gpu, journal, file, bytes);
        try journal.write(.{ .kind = "prefill_gdn_coefficient", .ordinal = gpu.ctx.device_ordinal, .name = c.name, .bytes = c.bytes, .sha256 = &hash(bytes), .file = file, .actual_checkpoint = true, .finite = true });
    }
    var in = try Inputs.init(gpu, e, journal);
    defer in.deinit();
    try journal.write(.{ .kind = "prefill_gdn_scope", .ordinal = gpu.ctx.device_ordinal, .checkpoint = model_dir, .placement_first = 1, .placement_end = 2, .synthetic_inputs = true, .actual_checkpoint_coefficients = true, .recurrence_bytes = e.g.recBytes(), .conv_bytes = e.g.convBytes(), .initial_position = 7, .guard_bytes = guard_bytes, .operator_only = true });
    for ([_]Layout{ .value_key, .key_value }) |layout| for ([_]usize{ 2, 16 }) |rows| {
        const initial = try initialState(gpu, e, layout);
        defer gpu.gpa.free(initial);
        try finite(initial[0..e.g.recBytes()], 4);
        try finite(initial[e.g.recBytes()..], 2);
        const initial_file = try std.fmt.allocPrint(gpu.gpa, "gdn-{d}-{s}-{d}-initial-state.bin", .{ gpu.ctx.device_ordinal, @tagName(layout), rows });
        defer gpu.gpa.free(initial_file);
        try save(gpu, journal, initial_file, initial);
        try journal.write(.{ .kind = "prefill_gdn_initial_state", .ordinal = gpu.ctx.device_ordinal, .layout = @tagName(layout), .rows = rows, .bytes = initial.len, .sha256 = &hash(initial), .file = initial_file, .synthetic = true, .finite = true, .nonzero_recurrence = true, .nonzero_conv_tail = true });
        var serial = try Work.init(gpu, e);
        defer serial.deinit();
        var batch = try Work.init(gpu, e);
        defer batch.deinit();
        var ts = try Tables.init(gpu, e, &serial, rows, false, layout, journal);
        defer ts.deinit();
        var tb = try Tables.init(gpu, e, &batch, rows, true, layout, journal);
        defer tb.deinit();
        var sp: Plan = .{};
        var bp: Plan = .{};
        appendOps(&sp, e, &in, &serial, ts, layout, rows, false, 0, rows);
        appendOps(&bp, e, &in, &batch, tb, layout, rows, true, 0, rows);
        var sg = try sp.graph(gpu, journal, layout, rows, "serial");
        defer sg.deinit();
        var bg = try bp.graph(gpu, journal, layout, rows, "batch");
        defer bg.deinit();
        try serial.reset(initial);
        try batch.reset(initial);
        try sg.launchOn(e.stream);
        try bg.launchOn(e.stream);
        try e.stream.synchronize();
        try compare(gpu, &serial, &batch, rows, e.g.recBytes(), journal, layout, rows, false);
        var sc: Plan = .{};
        var bc: Plan = .{};
        appendOps(&sc, e, &in, &serial, ts, layout, rows, false, rows, 1);
        appendOps(&bc, e, &in, &batch, tb, layout, rows, false, rows, 1);
        var scg = try sc.graph(gpu, journal, layout, rows, "serial_continuation");
        defer scg.deinit();
        var bcg = try bc.graph(gpu, journal, layout, rows, "batch_continuation");
        defer bcg.deinit();
        // Continuation is one row, so its pre/post row guard must also be one.
        const one = [_]c_int{ 0, 0, 1, 1, 0, 0, 0, 0 };
        try tb.params.upload(0, std.mem.asBytes(&one));
        const no_parent: [max_rows]c_int = @splat(-1);
        try tb.parents.upload(0, std.mem.asBytes(&no_parent));
        for ([_]cuda.DeviceBuffer{ tb.params, tb.parents }, [_][]const u8{ std.mem.asBytes(&one), std.mem.asBytes(&no_parent) }, 0..) |b, expected, index| {
            const actual = try check.download(gpu, b);
            defer gpu.gpa.free(actual);
            const file = try std.fmt.allocPrint(gpu.gpa, "gdn-{d}-{s}-{d}-continuation-table-{d}.bin", .{ gpu.ctx.device_ordinal, @tagName(layout), rows, index });
            defer gpu.gpa.free(file);
            try save(gpu, journal, file, actual);
            try check.sameBytes("GDN continuation table", actual, expected);
            try journal.write(.{ .kind = "prefill_gdn_continuation_table", .ordinal = gpu.ctx.device_ordinal, .layout = @tagName(layout), .rows = rows, .table = index, .bytes = actual.len, .sha256 = &hash(actual), .file = file, .upload_exact = true });
        }
        try journal.write(.{ .kind = "prefill_gdn_continuation_views", .ordinal = gpu.ctx.device_ordinal, .layout = @tagName(layout), .rows = rows, .params = one, .parents = no_parent });
        try scg.launchOn(e.stream);
        try bcg.launchOn(e.stream);
        try e.stream.synchronize();
        try compare(gpu, &serial, &batch, rows + 1, e.g.recBytes(), journal, layout, rows, true);
        if ((try gpu.ctx.memInfo()).free < floor_bytes) return error.GpuMemoryFloor;
        try journal.write(.{ .kind = "prefill_gdn_complete", .ordinal = gpu.ctx.device_ordinal, .layout = @tagName(layout), .rows = rows, .serial_nodes = sp.count, .batch_nodes = bp.count, .continuation_nodes = sc.count, .bit_exact = true, .free_bytes = (try gpu.ctx.memInfo()).free, .timed = false });
    };
    try in.verify(gpu, journal);
}

test "GDN input rows differ and scalar poison is rejected" {
    try distinct(&.{ 1, 2, 3, 4 }, 2);
    try std.testing.expectError(error.IdenticalGdnRows, distinct(&.{ 1, 2, 1, 2 }, 2));
    try finite(&.{ 0, 0, 0x80, 0x3f }, 4);
    try std.testing.expectError(error.NonfiniteGdnOutput, finite(&.{ 0xff, 0xff, 0xff, 0xff }, 4));
    try std.testing.expectError(error.NonfiniteGdnOutput, finite(&.{ 0xff, 0xff }, 2));
}

test "GDN guards reject either boundary and a truncated readback" {
    const b: Guarded = .{ .buffer = undefined, .bytes = 4 };
    var raw: [2 * guard_bytes + 4]u8 = @splat(0xa5);
    try b.guards(&raw);
    raw[0] = 0;
    try std.testing.expectError(error.GdnLeadingGuard, b.guards(&raw));
    raw[0] = 0xa5;
    raw[raw.len - 1] = 0;
    try std.testing.expectError(error.GdnTrailingGuard, b.guards(&raw));
    try std.testing.expectError(error.InvalidGdnReadback, b.guards(raw[0 .. raw.len - 1]));
}

test "GDN synthetic rows remain finite and distinct at the smallest input width" {
    var rows: [max_rows * 32]u16 = undefined;
    for (&rows, 0..) |*v, i| v.* = inputBits(i / 32, i % 32, 0);
    try finite(std.mem.asBytes(&rows), 2);
    try distinct(std.mem.asBytes(&rows), max_rows);
}
