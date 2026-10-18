//! The checkpoint-source path's exact BF16-to-FP32 conversion, with no accumulator to qualify.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const c = core.contract.spec;
const g = core.contract.gates;

fn depth(_: c.Shape) usize {
    return 0;
}

pub fn spec() c.Spec {
    return .{
        .name = "cuda GLM source tf_bf16_to_f32_kernel",
        .accum = .none,
        .depth = depth,
        .out = .f32,
        .tiles = .{ .{}, .{ .size = 256 }, .{} },
        .served = &.{
            .{ .rows = 1, .n = 128, .k = 1 },
            .{ .rows = 1, .n = 255, .k = 1 },
            .{ .rows = 1, .n = 256, .k = 1 },
            .{ .rows = 1, .n = 257, .k = 1 },
            .{ .rows = 16, .n = 16384, .k = 1 },
        },
    };
}

pub const Cast = struct {
    rig: rig_.Rig,
    f: cuda.Function,
    host: []u16 = &.{},
    x: u64 = 0,
    out: u64 = 0,

    pub fn deinit(self: *Cast) void {
        self.rig.gpu.gpa.free(self.host);
        self.rig.deinit();
    }

    pub fn kernel(self: *Cast) g.Kernel {
        return .{ .spec = spec(), .ctx = self, .vt = &vt };
    }

    const vt = g.hooks(Cast);

    pub fn prepare(self: *Cast, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        const a = self.rig.gpu.gpa;
        a.free(self.host);
        self.host = &.{};
        self.rig.reset();
        if (shape.rows == 0 or shape.n == 0 or shape.k != 1) return false;
        self.host = try a.alloc(u16, shape.rows * shape.n);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        for (self.host) |*h| h.* = c.toBf16(rng.float(f32) * 4 - 2);
        if (fill == .nonfinite) {
            const put = [_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32), 0.0, -0.0 };
            for (put, 0..) |v, i| if (i < self.host.len) {
                self.host[i] = c.toBf16(v);
            };
        }
        self.x = try self.rig.input(std.mem.sliceAsBytes(self.host));
        self.out = try self.rig.output(self.host.len * 4);
        return true;
    }

    pub fn run(self: *Cast) !void {
        var args: cuda.Args = .{};
        args.add(self.x);
        args.add(self.out);
        args.add(@as(u64, self.host.len));
        try cuda.launch.launch(self.f, .{
            .grid = .{ .x = @intCast(@min((self.host.len + 255) / 256, 65535)) },
            .block = .{ .x = 256 },
        }, self.rig.s, &args);
        try self.rig.s.synchronize();
    }

    pub fn count(self: *Cast) usize {
        return self.host.len;
    }

    pub fn reference(self: *Cast, want: []f64, mag: []f64) !void {
        for (self.host, want, mag) |h, *w, *m| {
            w.* = c.fromBf16(h);
            m.* = @abs(w.*);
        }
    }

    pub fn check(self: *Cast, _: c.Gate) !?[]const u8 {
        const bytes = try self.rig.outputBytes();
        for (self.host, 0..) |h, i| {
            const got = std.mem.readInt(u32, bytes[4 * i ..][0..4], .little);
            if (got != @as(u32, h) << 16) return "BF16 upcast changed a value's exact bits";
        }
        return null;
    }
};
