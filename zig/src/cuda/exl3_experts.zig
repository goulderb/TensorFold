//! exl3_experts.cu's EXL3 routed experts on grouped.zig's plan: gate and up under SwiGLU, down to fp32 or bf16 rows.
const std = @import("std");
const driver = @import("driver.zig");
const module = @import("module.zig");
const launch_ = @import("launch.zig");
const stream_ = @import("stream.zig");
const kernels = @import("kernels.zig");
const grouped = @import("grouped.zig");
const experts = @import("experts.zig");
const exl3 = @import("exl3_linear.zig");

const Driver = driver.Driver;
const Module = module.Module;
const Function = module.Function;
const Stream = stream_.Stream;

/// exl3_experts.cu's TfExl3Experts, passed by value.
pub const Geometry = extern struct { k: u32, n: u32, bits: u32, tiles_n: u32, mats: u32, epi: u32, book: u32, pad: u32 = 0, stride: u64 };

/// A layer's EXL3 widths and codebook; every expert of a layer shares them (a family stacks its shared ones apart).
pub const Widths = struct { up: u32 = 0, down: u32 = 0, codebook: exl3.Codebook = .mul1 };

/// Tensor decode uses the prompt kernel's FP16 rotated inputs, retaining the decode plan and epilogue.
pub const Decode = enum { scalar, tensor };

/// Halves an expert matrix carries: suh (k) then svh (n).
pub fn halves(k: usize, n: usize) usize {
    return k + n;
}

/// The device stacks a layer view points at: up [E][2] (gate, up) and down [E] trellises, each with its halves.
pub const Stacks = struct {
    pub fn upBytes(count: usize, dims: usize, width: usize, bits: u32) usize {
        return count * 2 * exl3.Matrix.trellisBytes(dims, width, bits);
    }
    pub fn downBytes(count: usize, dims: usize, width: usize, bits: u32) usize {
        return count * exl3.Matrix.trellisBytes(width, dims, bits);
    }
    pub fn upHalves(count: usize, dims: usize, width: usize) usize {
        return count * 2 * halves(dims, width);
    }
    pub fn downHalves(count: usize, dims: usize, width: usize) usize {
        return count * halves(width, dims);
    }
};

pub const Experts = struct {
    mod: Module,
    decode_fn: Function, // every width and codebook
    prompt_fn: Function, // prompt rows on the tensor units
    resident: [2]usize, // blocks the grid may hold, decode and prompt: per SM times SMs
    decode: Decode = .scalar,

    pub fn load(d: *const Driver, sms: usize) !Experts {
        if (!kernels.available) return error.BuiltWithoutKernels;
        var mod = try Module.load(d, kernels.exl3_experts);
        errdefer mod.unload();
        const decode = try mod.function("tf_exl3_experts");
        const prompt = try mod.function("tf_exl3_experts_prompt");
        const e: Experts = .{ .mod = mod, .decode_fn = decode, .prompt_fn = prompt, .resident = .{ @max(1, try decode.occupancy(256, 0)) * sms, @max(1, try prompt.occupancy(256, 0)) * sms } };
        return e;
    }

    pub fn unload(e: *Experts) void {
        e.mod.unload();
    }

    /// The geometry of one call: up takes D inputs to NI outputs (gate and up), down NI to D.
    pub fn geometry(epi: experts.Epi, l: experts.Layer) !Geometry {
        const up = epi == .up;
        const k = if (up) l.dims else l.width;
        const n = if (up) l.width else l.dims;
        const bits = if (up) l.exl3.up else l.exl3.down;
        try (exl3.Matrix{ .trellis = 0, .suh = 0, .svh = 0, .k = k, .n = n, .bits = bits, .codebook = l.exl3.codebook }).check();
        return .{
            .k = @intCast(k),
            .n = @intCast(n),
            .bits = bits,
            .tiles_n = @intCast(n / 16),
            .mats = if (up) 2 else 1,
            .epi = switch (epi) {
                .up => 2,
                .down_f32 => 0,
                .down_bf16 => 3,
            },
            .book = @backingInt(l.exl3.codebook),
            .stride = exl3.Matrix.trellisBytes(k, n, bits),
        };
    }

    /// Decode rows 16 pairs at a time, prompt rows 64 on the tensor units; a row's bits are its path's at every plan.
    pub fn run(e: *const Experts, s: Stream, epi: experts.Epi, in: experts.Rows, l: experts.Layer, p: grouped.Plan, out: u64, items: usize, skip: i32, prompt: bool) !void {
        const g = try geometry(epi, l);
        const tensor = prompt or e.decode == .tensor;
        const pi: usize = @intFromBool(tensor);
        const grid = @min(items * (g.n / 128), e.resident[pi]);
        if (grid < 1) return;
        const up = epi == .up;
        var a: launch_.Args = .{};
        a.add(in.x);
        a.add(@as(i32, @intCast(in.stride)));
        a.add(@as(i32, @intCast(if (up) in.slots else 0)));
        a.add(if (up) l.up else l.down);
        a.add(if (up) l.up_scale else l.down_scale);
        for ([_]u64{ p.items, p.counts, p.members, out }) |v| a.add(v);
        a.add(if (up) l.limit else @as(f32, 0.0));
        a.add(skip);
        a.add(g);
        try launch_.launch(if (tensor) e.prompt_fn else e.decode_fn, .{ .grid = .{ .x = @intCast(grid), .y = 1, .z = 1 }, .block = .{ .x = 256 } }, s, &a);
    }
};

comptime {
    std.debug.assert(@sizeOf(Geometry) == 40);
}

test "an EXL3 layer's geometry follows the epilogue, and a shape exl3.cu cannot take is refused" {
    const l: experts.Layer = .{ .format = .exl3, .up = 0, .down = 0, .width = 2048, .dims = 4096, .experts = 288, .exl3 = .{ .up = 4, .down = 4 } };
    const g = try Experts.geometry(.up, l);
    try std.testing.expectEqual(@as(u32, 2), g.mats);
    try std.testing.expectEqual(@as(u64, 256 * 128 * 64 * 2), g.stride); // a gate_proj: trellis [256, 128, 64] int16
    const d = try Experts.geometry(.down_bf16, l);
    try std.testing.expectEqual(@as(u32, 2048), d.k);
    try std.testing.expectEqual(@as(u32, 3), d.epi);
    var bad = l;
    bad.width = 2000;
    try std.testing.expectError(error.InvalidExl3Matrix, Experts.geometry(.up, bad));
    try std.testing.expectEqual(@as(usize, 288 * 2 * (4096 + 2048)), Stacks.upHalves(288, 4096, 2048));
}

test "tensor decode preserves the grouped decode plan, F32 epilogue and shared skip at the driver boundary" {
    const abi = @import("abi.zig");
    const Mock = struct {
        var function: usize = 0;
        var grid: u32 = 0;
        var values: [11]u64 = undefined;
        var geom: Geometry = undefined;
        fn launch(f: abi.Function, gx: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: abi.Stream, params: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) abi.Result {
            function = @intFromPtr(f);
            grid = gx;
            for (0..11) |i| values[i] = switch (i) {
                1, 2, 9, 10 => @as(*align(1) const u32, @ptrCast(params.?[i].?)).*,
                else => @as(*align(1) const u64, @ptrCast(params.?[i].?)).*,
            };
            geom = @as(*align(1) const Geometry, @ptrCast(params.?[11].?)).*;
            return abi.success;
        }
    };
    var d: Driver = undefined;
    d.api.cuLaunchKernel = Mock.launch;
    var e: Experts = .{ .mod = undefined, .decode_fn = .{ .d = &d, .handle = @ptrFromInt(1) }, .prompt_fn = .{ .d = &d, .handle = @ptrFromInt(2) }, .resident = .{ 9999, 9999 } };
    const g: experts.Grouped = .{ .exl3 = &e };
    const s: Stream = .{ .d = &d, .handle = null };
    const p: grouped.Plan = .{ .items = 101, .counts = 102, .members = 103, .rank = 104, .hist = 105 };
    const l: experts.Layer = .{ .format = .exl3, .up = 201, .down = 202, .up_scale = 203, .down_scale = 204, .dims = 4096, .width = 1024, .experts = 256, .exl3 = .{ .up = 4, .down = 4 } };
    const items = grouped.maxItems(32 * 9, 257, 16);
    for ([_]Decode{ .scalar, .tensor }) |policy| {
        e.decode = policy;
        for ([_]experts.Epi{ .up, .down_f32 }) |epi| {
            try g.decode(s, epi, .{ .x = 301, .stride = 4096, .slots = 9 }, l, p, 302, items, 256);
            try std.testing.expectEqual(@as(usize, if (policy == .scalar) 1 else 2), Mock.function);
            try std.testing.expectEqual(@as(u32, @intCast(items * (if (epi == .up) @as(usize, 8) else 32))), Mock.grid);
            try std.testing.expectEqualSlices(u64, &.{ 101, 102, 103, 302 }, Mock.values[5..9]);
            try std.testing.expectEqual(@as(u64, 256), Mock.values[10]);
            try std.testing.expectEqual(@as(u32, if (epi == .up) 2 else 0), Mock.geom.epi);
            try std.testing.expectEqual(@as(u64, if (epi == .up) 9 else 0), Mock.values[2]);
        }
    }
    e.decode = .scalar;
    try g.prompt(s, .down_bf16, .{ .x = 301, .stride = 1024 }, l, p, 302, 1, 256);
    try std.testing.expectEqual(@as(usize, 2), Mock.function);
    try std.testing.expectEqual(@as(u32, 3), Mock.geom.epi);
}
