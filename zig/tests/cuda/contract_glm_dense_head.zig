//! DFlash's existing private affine4 head kernels, including the 16-, 32- and 64-row proposal buckets.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const c = core.contract.spec;
const g = core.contract.gates;
const Affine = core.contract.affine.Affine;

fn depth(s: c.Shape) usize {
    return 64 + 64 + 2 * (s.k / 64);
}

pub fn spec(bm: usize) c.Spec {
    return .{
        .name = switch (bm) {
            16 => "cuda DFlash target head glm_qmm_16_0_0",
            32 => "cuda DFlash target head glm_qmm_32_0_0",
            64 => "cuda DFlash target head glm_qmm_64_0_0",
            else => unreachable,
        },
        .accum = .f32,
        .depth = depth,
        .out = .bf16,
        .tiles = .{ .{ .size = bm }, .{ .size = 64 }, .{ .size = 64, .step = 64, .min = 64 } },
        .served = switch (bm) {
            16 => &.{ .{ .rows = 7, .n = 77440, .k = 4096 }, .{ .rows = 14, .n = 77440, .k = 4096 } },
            32 => &.{ .{ .rows = 21, .n = 77440, .k = 4096 }, .{ .rows = 28, .n = 77440, .k = 4096 } },
            64 => &.{ .{ .rows = 35, .n = 77440, .k = 4096 }, .{ .rows = 42, .n = 77440, .k = 4096 }, .{ .rows = 49, .n = 77440, .k = 4096 }, .{ .rows = 56, .n = 77440, .k = 4096 } },
            else => unreachable,
        },
    };
}

fn shared(bm: usize) u32 {
    return switch (bm) {
        16 => 17664,
        32 => 26112,
        64 => 43008,
        else => unreachable,
    };
}

pub const Head = struct {
    rig: rig_.Rig,
    affine4: *const cuda.qlinear.Affine4Kernels,
    f: cuda.Function,
    bm: usize,
    host: ?Affine = null,
    shape: c.Shape = undefined,
    x: u64 = 0,
    xs: u64 = 0,
    q: cuda.qlinear.Affine4 = undefined,
    out: u64 = 0,

    pub fn deinit(self: *Head) void {
        if (self.host) |f| f.deinit();
        self.rig.deinit();
    }

    pub fn kernel(self: *Head) g.Kernel {
        return .{ .spec = spec(self.bm), .ctx = self, .vt = &vt };
    }

    const vt = g.hooks(Head);

    pub fn prepare(self: *Head, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (self.host) |f| f.deinit();
        self.host = null;
        self.rig.reset();
        if (shape.rows == 0 or shape.rows > self.bm or shape.n == 0 or shape.k == 0 or shape.k % 64 != 0) return false;
        const a = self.rig.gpu.gpa;
        self.shape = shape;
        // A prime output period crosses tile boundaries; identical columns reuse their exact reference, not a sample.
        const f = try Affine.init(a, .{ .rows = shape.rows, .n = @min(shape.n, 31), .k = shape.k }, 4, 64);
        self.host = f;
        try f.fill(fill, seed);
        const words = try a.alloc(u32, shape.n * shape.k / 8);
        defer a.free(words);
        const sample = try a.alloc(u32, f.shape.n * shape.k / 8);
        defer a.free(sample);
        f.words(sample);
        const groups = shape.k / 64;
        const scales = try a.alloc(u16, shape.n * groups);
        defer a.free(scales);
        const biases = try a.alloc(u16, scales.len);
        defer a.free(biases);
        for (0..shape.n) |j| {
            const ref = j % f.shape.n;
            @memcpy(words[j * (shape.k / 8) ..][0 .. shape.k / 8], sample[ref * (shape.k / 8) ..][0 .. shape.k / 8]);
            @memcpy(scales[j * groups ..][0..groups], f.scales[ref * groups ..][0..groups]);
            @memcpy(biases[j * groups ..][0..groups], f.biases[ref * groups ..][0..groups]);
        }
        const sums = try a.alloc(f32, shape.rows * groups);
        defer a.free(sums);
        for (sums, 0..) |*v, i| {
            var total: f32 = 0;
            for (f.x[i * 64 ..][0..64]) |x| total += c.fromBf16(x);
            v.* = total;
        }
        self.x = try self.rig.input(std.mem.sliceAsBytes(f.x));
        self.xs = try self.rig.input(std.mem.sliceAsBytes(sums));
        const lay = cuda.qlinear.Affine4.layout(shape.n, shape.k);
        const zeros = try a.alloc(u8, lay.words);
        defer a.free(zeros);
        @memset(zeros, 0);
        self.q = .{
            .w = try self.rig.input(zeros),
            .s = try self.rig.input(zeros[0..lay.scales]),
            .b = try self.rig.input(zeros[0..lay.scales]),
            .n = shape.n,
            .k = shape.k,
            .npad = lay.npad,
        };
        const raw_w = try self.rig.scratch(words.len * 4);
        const raw_s = try self.rig.scratch(scales.len * 2);
        const raw_b = try self.rig.scratch(biases.len * 2);
        for ([_]u64{ raw_w, raw_s, raw_b }, [_][]const u8{ std.mem.sliceAsBytes(words), std.mem.sliceAsBytes(scales), std.mem.sliceAsBytes(biases) }) |ptr, bytes| {
            const b: cuda.DeviceBuffer = .{ .d = self.rig.gpu.d, .ptr = ptr, .len = bytes.len };
            try b.upload(0, bytes);
        }
        try self.affine4.pack(self.rig.s, raw_w, raw_s, raw_b, self.q);
        try self.rig.s.synchronize();
        self.out = try self.rig.output(shape.rows * shape.n * 2);
        return true;
    }

    pub fn run(self: *Head) !void {
        const shape = self.shape;
        var args: cuda.Args = .{};
        for ([_]u64{ self.x, self.xs, self.q.w, self.q.s, self.q.b, self.out, 0 }) |v| args.add(v);
        for ([_]usize{ shape.rows, shape.n, shape.k, 1, self.q.npad, shape.k, 1 }) |v| args.add(@as(c_int, @intCast(v)));
        try self.f.allowDynamicShared(shared(self.bm));
        try cuda.launch.launch(self.f, .{
            .grid = .{ .x = @intCast(((shape.rows + self.bm - 1) / self.bm) * ((shape.n + 63) / 64)) },
            .block = .{ .x = 128 },
            .shared = shared(self.bm),
        }, self.rig.s, &args);
        try self.rig.s.synchronize();
    }

    pub fn count(self: *Head) usize {
        return self.shape.rows * self.shape.n;
    }

    fn expand(self: *Head, sample: []const f64, out: []f64) void {
        const n = self.host.?.shape.n;
        for (0..self.shape.rows) |r| for (0..self.shape.n) |j| {
            out[r * self.shape.n + j] = sample[r * n + j % n];
        };
    }

    pub fn reference(self: *Head, want: []f64, mag: []f64) !void {
        const a = self.rig.gpu.gpa;
        const f = self.host.?;
        const small_want = try a.alloc(f64, f.outputs());
        defer a.free(small_want);
        const small_mag = try a.alloc(f64, f.outputs());
        defer a.free(small_mag);
        try f.reference(small_want, small_mag);
        self.expand(small_want, want);
        self.expand(small_mag, mag);
    }

    pub fn simulate(self: *Head, acc: c.Accum, order: g.Order, out: []f64) !void {
        const a = self.rig.gpu.gpa;
        const f = self.host.?;
        const small = try a.alloc(f64, f.outputs());
        defer a.free(small);
        try f.simulate(acc, order, small);
        self.expand(small, out);
    }
};
