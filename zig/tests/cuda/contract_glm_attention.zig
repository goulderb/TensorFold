//! GLM latent chunks under the CUDA contract; FP8 vector loads must preserve all baseline partial bytes.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const glm = @import("glm_cuda");
const check_ = @import("check.zig");
const Rig = @import("contract_rig.zig").Rig;
const data = @import("contract_glm_attention_data.zig");
const c = core.contract.spec;
const g = core.contract.gates;
const lw = data.lw;

pub const Kind = enum {
    bf16_16_dense,
    bf16_16_sparse,
    bf16_32_dense,
    bf16_32_sparse,
    fp8_16_dense,
    fp8_16_sparse,
    fp8_32_dense,
    fp8_32_sparse,
    fp8load_dense,
    fp8load_sparse,
};

fn isSparse(kind: Kind) bool {
    return switch (kind) {
        .bf16_16_sparse, .bf16_32_sparse, .fp8_16_sparse, .fp8_32_sparse, .fp8load_sparse => true,
        else => false,
    };
}

fn isFP8(kind: Kind) bool {
    return switch (kind) {
        .bf16_16_dense, .bf16_16_sparse, .bf16_32_dense, .bf16_32_sparse => false,
        else => true,
    };
}

fn headBlock(kind: Kind) usize {
    return switch (kind) {
        .bf16_16_dense, .bf16_16_sparse, .fp8_16_dense, .fp8_16_sparse => 16,
        else => 32,
    };
}

fn baselineShared(kind: Kind) u32 {
    return switch (headBlock(kind)) {
        16 => glm.glue.chunksSmem(16),
        else => glm.glue.chunksSmem(32),
    };
}

fn fast(kind: Kind) bool {
    return kind == .fp8load_dense or kind == .fp8load_sparse;
}

/// The production loader's handle, so its dynamic shared opt-in is the one these launches rely on.
pub fn function(k: *const glm.glue.Kernels, kind: Kind) cuda.Function {
    return switch (kind) {
        .bf16_16_dense, .bf16_16_sparse => k.chunks_16,
        .bf16_32_dense, .bf16_32_sparse => k.chunks_32,
        .fp8_16_dense, .fp8_16_sparse => k.chunks_16_fp8,
        .fp8_32_dense, .fp8_32_sparse => k.chunks_32_fp8,
        .fp8load_dense, .fp8load_sparse => k.chunks_32_fp8load,
    };
}

fn depth(_: c.Shape) usize {
    return 544;
}

const dense_shapes = [_]c.Shape{
    .{ .rows = 7, .n = 32, .k = 2051 },
    .{ .rows = 1, .n = 1, .k = 1 },
    .{ .rows = 2, .n = 15, .k = 31 },
    .{ .rows = 3, .n = 16, .k = 32 },
    .{ .rows = 4, .n = 17, .k = 33 },
    .{ .rows = 5, .n = 31, .k = 511 },
    .{ .rows = 6, .n = 32, .k = 512 },
    .{ .rows = 7, .n = 33, .k = 513 },
    .{ .rows = 63, .n = 32, .k = 1023 },
    .{ .rows = 64, .n = 45, .k = 1024 },
    .{ .rows = 65, .n = 32, .k = 1025 },
    .{ .rows = 512, .n = 32, .k = 2051 },
};

const sparse_shapes = dense_shapes ++ [_]c.Shape{
    .{ .rows = 2048, .n = 32, .k = 2051 },
    .{ .rows = 64, .n = 45, .k = 2047 },
    .{ .rows = 64, .n = 32, .k = 2048 },
    .{ .rows = 64, .n = 32, .k = 2049 },
    .{ .rows = 64, .n = 32, .k = 2050 },
};

pub fn spec(kind: Kind) c.Spec {
    return .{
        .name = switch (kind) {
            inline else => |k| "cuda GLM attention chunks " ++ @tagName(k),
        },
        .accum = .f32,
        .depth = depth,
        .out = .f32,
        .replaces = if (fast(kind)) "tf_glm_chunks_32_fp8" else null,
        .nan = .{ .stated = "full all-NaN key tiles yield NaN sums and -Inf max; masked all-NaN tails and all -Inf tiles are inactive" },
        .tiles = .{ .{ .size = 64 }, .{ .size = headBlock(kind) }, .{ .size = 512 } },
        .served = if (isSparse(kind)) &sparse_shapes else &dense_shapes,
    };
}

pub const Attention = struct {
    rig: Rig,
    kind: Kind,
    f: cuda.Function,
    baseline: cuda.Function,
    d: ?data.Data = null,
    q: u64 = 0,
    cache: u64 = 0,
    pos: u64 = 0,
    tokens: u64 = 0,
    counts: u64 = 0,
    po: u64 = 0,
    pm: u64 = 0,
    pl: u64 = 0,
    refs: []data.Ref = &.{},
    ready: []bool = &.{},
    old: std.ArrayList(u8) = .empty,

    pub fn init(gpu: check_.Gpu, s: cuda.Stream, k: *const glm.glue.Kernels, kind: Kind) Attention {
        const baseline = if (fast(kind)) k.chunks_32_fp8 else function(k, kind);
        return .{ .rig = .{ .gpu = gpu, .s = s }, .kind = kind, .f = function(k, kind), .baseline = baseline };
    }

    pub fn deinit(self: *Attention) void {
        if (self.d) |d| d.deinit();
        const a = self.rig.gpu.gpa;
        a.free(self.refs);
        a.free(self.ready);
        self.old.deinit(a);
        self.rig.deinit();
    }

    pub fn kernel(self: *Attention) g.Kernel {
        return .{ .spec = spec(self.kind), .ctx = self, .vt = &vt };
    }

    const vt = g.hooks(Attention);

    pub fn prepare(self: *Attention, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.rows > 2048 or shape.n == 0 or shape.n > 64 or shape.k == 0 or shape.k > 2051) return false;
        if (!isSparse(self.kind) and (shape.rows > 512 or shape.rows > shape.k)) return false;
        self.rig.reset();
        if (self.d) |d| d.deinit();
        self.d = null;
        const a = self.rig.gpu.gpa;
        a.free(self.refs);
        a.free(self.ready);
        self.refs = &.{};
        self.ready = &.{};
        const d = try data.Data.init(a, shape, isSparse(self.kind), isFP8(self.kind), fill, seed);
        self.d = d;
        const memo_rows = if (d.sparse) @min(shape.rows, data.edge_counts.len) else shape.rows;
        self.refs = try a.alloc(data.Ref, memo_rows * 5 * d.nch);
        self.ready = try a.alloc(bool, self.refs.len);
        @memset(self.ready, false);
        self.q = try self.rig.input(std.mem.sliceAsBytes(d.q));
        self.cache = try self.rig.input(d.cache);
        self.pos = try self.rig.input(std.mem.asBytes(&d.position));
        self.tokens = try self.rig.input(std.mem.sliceAsBytes(d.tokens));
        self.counts = try self.rig.input(std.mem.sliceAsBytes(d.counts));
        const scalar_count = d.nch * shape.rows * shape.n;
        self.po = try self.rig.output(scalar_count * lw * 4);
        self.pm = try self.rig.output(scalar_count * 4);
        self.pl = try self.rig.output(scalar_count * 4);
        return true;
    }

    fn launch(self: *Attention, f: cuda.Function, candidate: bool) !void {
        const d = self.d.?;
        var args: cuda.Args = .{};
        inline for (.{ self.q, self.cache, self.pos, self.tokens, self.counts, self.po, self.pm, self.pl, @as(i64, @intCast(d.shape.rows)), @as(i64, @intCast(d.shape.n)), @as(i64, @intCast(d.shape.k)), @as(f32, 0.125), @as(c_int, @intFromBool(d.sparse)) }) |v| args.add(v);
        const hb = headBlock(self.kind);
        try cuda.launch.launch(f, .{
            .grid = .{ .x = @intCast(d.shape.rows), .y = @intCast((d.shape.n + hb - 1) / hb), .z = @intCast(d.nch) },
            .block = .{ .x = 256 },
            .shared = if (candidate) glm.glue.chunksFP8LoadSmem() else baselineShared(self.kind),
        }, self.rig.s, &args);
        try self.rig.s.synchronize();
    }

    pub fn run(self: *Attention) !void {
        return self.launch(self.f, fast(self.kind));
    }

    pub fn replaced(self: *Attention) ![]const u8 {
        try self.launch(self.baseline, false);
        self.old.clearRetainingCapacity();
        const scalar_bytes = self.scalars() * 4;
        for ([_]u64{ self.po, self.pm, self.pl }, [_]usize{ scalar_bytes * lw, scalar_bytes, scalar_bytes }) |ptr, bytes| {
            try self.old.appendSlice(self.rig.gpu.gpa, try self.rig.bytes(ptr, bytes));
        }
        return self.old.items;
    }

    fn scalars(self: *Attention) usize {
        const d = self.d.?;
        return d.nch * d.shape.rows * d.shape.n;
    }

    pub fn count(self: *Attention) usize {
        return self.scalars() * (if (self.d.?.fill == .stress) @as(usize, lw) else lw + 2);
    }

    fn refFor(self: *Attention, row: usize, head: usize, chunk: usize) data.Ref {
        const d = self.d.?;
        const rr = if (d.sparse) row % data.edge_counts.len else row;
        const kind = (row % 4 + head % 4) % 5;
        const at = (rr * 5 + kind) * d.nch + chunk;
        if (!self.ready[at]) {
            self.refs[at] = data.reference(d, row, head, chunk);
            self.ready[at] = true;
        }
        return self.refs[at];
    }

    pub fn reference(self: *Attention, want: []f64, mag: []f64) !void {
        const d = self.d.?;
        if (d.fill != .stress) return error.AttentionShapesUseSensitivityBound;
        for (0..d.nch) |chunk| for (0..d.shape.rows) |row| for (0..d.shape.n) |head| {
            const base = ((chunk * d.shape.rows + row) * d.shape.n + head) * lw;
            for (0..lw) |col| {
                var buf: [512]f64 = undefined;
                const terms = data.stressTerms(d, row, chunk, col, &buf);
                var value: f64 = 0;
                var abs: f64 = 0;
                for (terms) |v| {
                    value += v;
                    abs += @abs(v);
                }
                want[base + col] = value;
                mag[base + col] = abs;
            }
        };
    }

    pub fn simulate(self: *Attention, acc: c.Accum, order: g.Order, out: []f64) !void {
        const d = self.d.?;
        for (0..d.nch) |chunk| for (0..d.shape.rows) |row| for (0..d.shape.n) |head| {
            const base = ((chunk * d.shape.rows + row) * d.shape.n + head) * lw;
            for (0..lw) |col| out[base + col] = data.sum(d, row, chunk, col, acc, order);
        };
    }

    pub fn check(self: *Attention, gate: g.Gate) !?[]const u8 {
        if (gate != .nonfinite and gate != .shapes) return null;
        const raw = try self.rig.outputBytes();
        if (fast(self.kind) and gate == .nonfinite) {
            const old = try self.replaced();
            return if (std.mem.eql(u8, raw, old)) null else "FP8 key loads changed baseline nonfinite partial bytes";
        }
        const d = self.d.?;
        const n = self.scalars();
        for (0..d.nch) |chunk| for (0..d.shape.rows) |row| for (0..d.shape.n) |head| {
            const scalar = (chunk * d.shape.rows + row) * d.shape.n + head;
            const pm = c.read(.f32, raw, n * lw + scalar);
            const pl = c.read(.f32, raw, n * (lw + 1) + scalar);
            const active = chunk * 512 < d.length(row);
            if (gate == .nonfinite) {
                const full_tile = d.length(row) -| (chunk * 512) >= 32;
                const bad = active and (row == 1 or ((row == 0 or row == 3) and full_tile));
                const max_class: c.Class = if (active and row == 1) .pos_inf else if (!active or row < 4) .neg_inf else .finite;
                if (c.classOf(pm) != max_class) return "chunk maximum violates the stated fmax nonfinite rule";
                if (c.classOf(pl) != (if (bad) c.Class.nan else .finite)) return "chunk denominator violates the stated fmax nonfinite rule";
                for (0..lw) |col| {
                    const value = c.read(.f32, raw, scalar * lw + col);
                    if (c.classOf(value) != (if (bad) c.Class.nan else .finite)) return "chunk numerator violates the stated fmax nonfinite rule";
                    if ((!active or row == 2) and value != 0) return "inactive chunk must write a zero numerator";
                }
                if ((!active or row == 2) and pl != 0) return "inactive chunk must write a zero denominator";
            } else {
                const r = self.refFor(row, head, chunk);
                if (!active) {
                    if (pm != -std.math.inf(f64) or pl != 0) return "empty chunk maximum/denominator was not initialized";
                } else {
                    if (!(@abs(pm - r.pm) <= r.pm_tol)) return "chunk maximum exceeds the independent float64 score bound";
                    if (!(@abs(pl - r.pl) <= r.pl_tol)) return "chunk denominator exceeds the independent float64 softmax bound";
                }
                for (0..lw) |col| {
                    const value = c.read(.f32, raw, scalar * lw + col);
                    const pattern = data.columnPattern(col);
                    const factor = data.columnFactor(col);
                    if (!(@abs(value - r.po[pattern] * factor) <= r.tol[pattern] * @abs(factor) + 0x1p-149))
                        return "chunk numerator exceeds the independent float64 softmax/BF16 probability bound";
                }
            }
        };
        return null;
    }
};
