//! NVFP4 expert decode contracts: four baseline epilogues and byte-exact pipeline/column-slice replacements.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const data = @import("contract_glm_experts_data.zig");
const c = core.contract.spec;
const g = core.contract.gates;
const X = cuda.experts.Nvfp4Experts;

pub const Epi = data.Epi;
pub const Variant = X.Variant;
pub const variants = std.meta.tags(Variant);
pub const epilogues = std.meta.tags(Epi);
pub const kernel_count = variants.len * epilogues.len;
pub const clamped_count = variants.len;

fn names() [variants.len][epilogues.len][]const u8 {
    var result: [variants.len][epilogues.len][]const u8 = undefined;
    inline for (variants, 0..) |v, vi| inline for (epilogues, 0..) |e, ei| {
        result[vi][ei] = "cuda nvfp4 expert " ++ @tagName(v) ++ " " ++ @tagName(e);
    };
    return result;
}
const labels = names();
const clamped_labels = blk: {
    var result: [variants.len][]const u8 = undefined;
    for (variants, 0..) |v, i| result[i] = "cuda nvfp4 expert " ++ @tagName(v) ++ " swiglu clamp10";
    break :blk result;
};

fn depth(s: c.Shape) usize {
    return s.k / 16 + 16;
}

const up_shapes = [_]c.Shape{
    .{ .rows = 7, .n = 1024, .k = 4096 },
    .{ .rows = 56, .n = 1024, .k = 4096 },
    .{ .rows = 5, .n = 64, .k = 32 },
    .{ .rows = 6, .n = 96, .k = 64 },
    .{ .rows = 17, .n = 32, .k = 96 },
    .{ .rows = 33, .n = 64, .k = 128 },
    .{ .rows = 55, .n = 96, .k = 160 },
};
const down_shapes = [_]c.Shape{
    .{ .rows = 7, .n = 4096, .k = 1024 },
    .{ .rows = 56, .n = 4096, .k = 1024 },
    .{ .rows = 5, .n = 64, .k = 32 },
    .{ .rows = 6, .n = 96, .k = 64 },
    .{ .rows = 17, .n = 32, .k = 96 },
    .{ .rows = 33, .n = 64, .k = 128 },
    .{ .rows = 55, .n = 96, .k = 160 },
};

pub fn spec(variant: Variant, epi: Epi) c.Spec {
    const activation = epi == .swiglu or epi == .relu2;
    return .{
        .name = labels[@backingInt(variant)][@backingInt(epi)],
        .accum = .f32,
        .depth = depth,
        .ops = if (epi == .swiglu) 8 else if (epi == .relu2) 2 else 1,
        .mids = if (epi == .swiglu) 3 else if (epi == .relu2) 2 else 0,
        .out = if (epi == .down_f32) .f32 else .bf16,
        .replaces = if (variant == .baseline) null else labels[0][@backingInt(epi)],
        .nan = if (epi == .relu2) .{ .stated = "ReLU squares maxNumber(bf16(dot), 0): NaN and negative infinity yield zero" } else .propagate,
        .tiles = .{ .{ .size = 16 }, .{ .size = 32, .step = 32, .min = 32 }, .{ .size = 128, .step = 32, .min = 32 } },
        .served = if (activation) &up_shapes else &down_shapes,
    };
}

const prompt_up_shapes = [_]c.Shape{
    .{ .rows = 64, .n = 1024, .k = 4096 },
    .{ .rows = 2048, .n = 1024, .k = 4096 },
    .{ .rows = 63, .n = 32, .k = 32 },
    .{ .rows = 65, .n = 96, .k = 96 },
};
const prompt_down_shapes = [_]c.Shape{
    .{ .rows = 64, .n = 4096, .k = 1024 },
    .{ .rows = 2048, .n = 4096, .k = 1024 },
    .{ .rows = 63, .n = 32, .k = 32 },
    .{ .rows = 65, .n = 96, .k = 96 },
};

pub fn promptSpec(epi: Epi, clamped: bool) c.Spec {
    var s = spec(.baseline, epi);
    s.name = switch (epi) {
        .relu2 => "cuda nvfp4 expert prompt relu2",
        .swiglu => if (clamped) "cuda nvfp4 expert prompt swiglu clamp10" else "cuda nvfp4 expert prompt swiglu",
        .down_bf16 => "cuda nvfp4 expert prompt down_bf16",
        .down_f32 => unreachable,
    };
    s.served = if (epi == .down_bf16) &prompt_down_shapes else &prompt_up_shapes;
    s.tiles[0].size = 64;
    s.nan = nanRule(epi, clamped);
    return s;
}

fn nanRule(epi: Epi, clamped: bool) c.Nan {
    if (epi == .swiglu and clamped) return .{ .stated = "SwiGLU clamps bf16 gate by minNumber(g,10), up by maxNumber/minNumber(u,-10,10), including NaN" };
    return if (epi == .relu2) .{ .stated = "ReLU squares maxNumber(bf16(dot), 0): NaN and negative infinity yield zero" } else .propagate;
}

/// The expert plan is built independently on the CPU and every bound input is guarded and immutable.
pub const Expert = struct {
    rig: rig_.Rig,
    experts: *const X,
    variant: Variant,
    epi: Epi,
    clamped: bool = false,
    prompt: bool = false,
    host: ?data.Host = null,
    layer: cuda.experts.Layer = undefined,
    plan: cuda.grouped.Plan = undefined,
    input: cuda.experts.Rows = undefined,
    items: usize = 0,
    out: u64 = 0,
    baseline: u64 = 0,

    pub fn kernel(self: *Expert) g.Kernel {
        return .{ .spec = self.registration(), .ctx = self, .vt = &vt };
    }

    fn registration(self: *Expert) c.Spec {
        if (self.prompt) return promptSpec(self.epi, self.clamped);
        var s = spec(self.variant, self.epi);
        if (self.clamped) {
            std.debug.assert(self.epi == .swiglu);
            s.name = clamped_labels[@backingInt(self.variant)];
            s.replaces = if (self.variant == .baseline) null else clamped_labels[0];
            s.nan = nanRule(self.epi, true);
        }
        return s;
    }

    const vt = g.hooks(Expert);

    fn drop(self: *Expert) void {
        if (self.host) |h| h.deinit();
        self.host = null;
        self.rig.reset();
    }

    pub fn deinit(self: *Expert) void {
        if (self.host) |h| h.deinit();
        self.rig.deinit();
    }

    pub fn prepare(self: *Expert, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        self.drop();
        if (shape.rows == 0 or shape.rows > (if (self.prompt) @as(usize, 2048) else 56) or shape.n == 0 or shape.n % 32 != 0 or shape.k == 0 or shape.k % 32 != 0) return false;
        const a = self.rig.gpu.gpa;
        var h = try data.Host.init(a, shape, self.epi, fill, seed);
        h.limit = if (self.clamped and fill != .stress) 10 else 0;
        if (h.limit > 0) for (h.x) |*v| {
            v.* = c.toBf16(c.fromBf16(v.*) * 64);
        };
        self.host = h;
        const words = try h.packedWords();
        defer a.free(words);
        const weights = try self.rig.input(std.mem.sliceAsBytes(words));
        var scales: [data.experts * 2]f32 = undefined;
        for (0..data.experts) |e| for (0..h.parts()) |m| {
            scales[e * h.parts() + m] = h.scale(e, m);
        };
        const global = try self.rig.input(std.mem.sliceAsBytes(scales[0 .. data.experts * h.parts()]));
        self.layer = .{
            .format = .nvfp4,
            .up = weights,
            .down = weights,
            .up_scale = global,
            .down_scale = global,
            .width = if (h.up()) shape.n else shape.k,
            .dims = if (h.up()) shape.k else shape.n,
            .experts = data.experts,
            .relu2 = self.epi == .relu2,
            .limit = h.limit,
        };
        self.input = .{ .x = try self.rig.input(std.mem.sliceAsBytes(h.x)), .stride = h.stride(), .slots = if (h.up()) data.slots else 0 };
        try self.makePlan(h);
        const bytes = self.count() * c.size(self.registration().out);
        self.out = try self.rig.output(bytes);
        self.baseline = try self.rig.scratch(bytes);
        return true;
    }

    fn makePlan(self: *Expert, h: data.Host) !void {
        const a = self.rig.gpu.gpa;
        const pairs = h.shape.rows * data.slots;
        const tile: usize = if (self.prompt) 64 else 16;
        const bound = cuda.grouped.maxItems(pairs, data.experts, tile);
        const members = try a.alloc(i32, pairs);
        defer a.free(members);
        const items = try a.alloc(i32, bound * 3);
        defer a.free(items);
        @memset(items, -1);
        var total: usize = 0;
        var item_count: usize = 0;
        for (0..data.experts) |e| {
            const start = total;
            for (0..pairs) |p| if (h.expert(pairs - 1 - p) == e) {
                members[total] = @intCast(pairs - 1 - p);
                total += 1;
            };
            var first = start;
            while (first < total) : (first += tile) {
                const item = items[item_count * 3 ..][0..3];
                item.* = .{ @intCast(e), @intCast(first), @intCast(@min(tile, total - first)) };
                item_count += 1;
            }
        }
        if (total != pairs or item_count > bound) return error.InvalidContractPlan;
        const counts = [2]i32{ @intCast(item_count), @intCast(pairs) };
        self.plan = .{
            .members = try self.rig.input(std.mem.sliceAsBytes(members)),
            .items = try self.rig.input(std.mem.sliceAsBytes(items)),
            .counts = try self.rig.input(std.mem.asBytes(&counts)),
            .rank = 0,
            .hist = 0,
        };
        self.items = bound;
    }

    fn launch(self: *Expert, variant: Variant, out: u64) !void {
        const epi: cuda.experts.Epi = switch (self.epi) {
            .swiglu, .relu2 => .up,
            .down_f32 => .down_f32,
            .down_bf16 => .down_bf16,
        };
        if (self.prompt) {
            if (variant != .baseline) return error.PromptReplacementNotBuilt;
            try self.experts.runPrompt(self.rig.s, epi, self.input, self.layer, self.plan, out, self.items, -1);
        } else try self.experts.runVariant(variant, self.rig.s, epi, self.input, self.layer, self.plan, out, self.items, -1);
        try self.rig.s.synchronize();
    }

    pub fn run(self: *Expert) !void {
        try self.launch(self.variant, self.out);
    }

    pub fn replaced(self: *Expert) ![]const u8 {
        if (self.variant == .baseline) return error.NotAReplacement;
        try self.launch(.baseline, self.baseline);
        return self.rig.bytes(self.baseline, self.count() * c.size(self.registration().out));
    }

    pub fn count(self: *Expert) usize {
        const h = self.host.?;
        return h.shape.rows * data.slots * h.shape.n;
    }

    pub fn reference(self: *Expert, want: []f64, mag: []f64) !void {
        try self.host.?.calculate(null, .sequential, want, mag);
    }

    /// The stated clamp rule may make every planted nonfinite finite, so check each class explicitly.
    pub fn check(self: *Expert, at: g.Gate) !?[]const u8 {
        if (at != .nonfinite and at != .shapes) return null;
        const a = self.rig.gpu.gpa;
        const want = try a.alloc(f64, self.count());
        defer a.free(want);
        const mag = try a.alloc(f64, self.count());
        defer a.free(mag);
        try self.reference(want, mag);
        const out = try self.rig.outputBytes();
        const s = self.registration();
        for (want, mag, 0..) |w, m, i| {
            const got = c.read(s.out, out, i);
            if (at == .nonfinite) {
                if (c.classOf(got) != c.classOf(w)) return "output class differs from independent float64 epilogue rule";
            } else if (!(@abs(got - w) <= c.bound(s, self.host.?.shape, w, m))) {
                return "output exceeds independent float64 per-element bound";
            }
        }
        if (at == .nonfinite and self.variant != .baseline) {
            const base = try self.replaced();
            if (!std.mem.eql(u8, out, base)) return "replacement differs from baseline bytes on nonfinite inputs";
        }
        return null;
    }

    pub fn simulate(self: *Expert, acc: c.Accum, order: g.Order, out: []f64) !void {
        try self.host.?.calculate(acc, order, out, null);
    }
};

test "expert stress references prove both low precision accumulators inadequate in all three orders" {
    const a = std.testing.allocator;
    for (epilogues) |epi| {
        const s = spec(.baseline, epi);
        const shape: c.Shape = .{ .rows = 1, .n = 64, .k = if (epi == .down_f32 or epi == .down_bf16) 1024 else 4096 };
        const h = try data.Host.init(a, shape, epi, .stress, 0x6b63);
        defer h.deinit();
        const n = shape.rows * data.slots * shape.n;
        const want = try a.alloc(f64, n);
        defer a.free(want);
        const mag = try a.alloc(f64, n);
        defer a.free(mag);
        const got = try a.alloc(f64, n);
        defer a.free(got);
        try h.calculate(null, .sequential, want, mag);
        for ([_]g.Order{ .sequential, .pairwise, .strided }) |order| {
            try h.calculate(.f32, order, got, null);
            for (got, want) |v, w| try std.testing.expect(@abs(v - w) <= c.boundExact(s, w));
            for ([_]c.Accum{ .f16, .bf16 }) |acc| {
                try h.calculate(acc, order, got, null);
                var worst: f64 = 0;
                for (got, want) |v, w| worst = @max(worst, if (std.math.isFinite(v)) @abs(v - w) / c.boundExact(s, w) else std.math.inf(f64));
                try std.testing.expect(worst >= 100);
            }
        }
    }
}
