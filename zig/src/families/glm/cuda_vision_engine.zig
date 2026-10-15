//! Native 24-block vision tower and projector, not registered for serving.
//! Source da6282f9 vision/glm.py _block/encode; no Torch, codec or fallback.
//! Runtime package evaluation is explicit; GPU algorithms/numerics are unqualified.
//! Borrowed weights, stream/context and native libraries must remain alive and
//! unmoved until deinit succeeds. Results borrow storage until the next encode.
const std = @import("std");
pub const memory = @import("cuda_vision_engine_memory.zig");
pub const cuda = @import("cuda");
pub const ops = @import("vision_ops");
pub const conv = @import("cuda_vision_cudnn.zig");
pub const attention = @import("cuda_vision_sdpa.zig");
const pre = @import("cuda_vision_preprocess.zig");
const lt = cuda.cublaslt;
const Config = memory.Config;
const Slot = memory.Slot;
pub const Runtime = struct {
    convolution: *const conv.cudnn.Api,
    sdpa: *const attention.sdpa.Api,
    linear: *const lt.Library,
    image: []const u8,
    /// Evaluation-only package attestation, never production qualification.
    package: enum { unqualified, target_qualification } = .unqualified,
};
pub const Options = struct { workspace_limit: usize, max_activation_bytes: usize };
const LinearSpec = struct { n: u32, k: u32, weight: memory.host.Block, bias: memory.host.Block };
const block_specs = [_]LinearSpec{
    .{ .n = 3072, .k = 1024, .weight = .qkv_weight, .bias = .qkv_bias },
    .{ .n = 1024, .k = 1024, .weight = .attention_weight, .bias = .attention_bias },
    .{ .n = 4096, .k = 1024, .weight = .gate_weight, .bias = .gate_bias },
    .{ .n = 4096, .k = 1024, .weight = .up_weight, .bias = .up_bias },
    .{ .n = 1024, .k = 4096, .weight = .down_weight, .bias = .down_bias },
};
const merger_specs = [_]struct { n: u32, k: u32, weight: memory.host.Root }{
    .{ .n = 4096, .k = 4096, .weight = .projection },
    .{ .n = 10240, .k = 4096, .weight = .gate },
    .{ .n = 10240, .k = 4096, .weight = .up },
    .{ .n = 4096, .k = 10240, .weight = .down },
};
fn exact(c: Config) !void {
    try c.check();
    if (c.hidden != 1024 or c.heads != 16 or c.depth != 24 or c.intermediate != 4096 or c.output != 4096 or c.projection_intermediate != 10240) return error.UnsupportedVisionTowerConfig;
}
fn linearSpan(b: cuda.DeviceBuffer) lt.DeviceSpan {
    return .{ .ptr = b.ptr, .len = b.len };
}
fn convSpan(b: cuda.DeviceBuffer) conv.cudnn.Span {
    return .{ .ptr = b.ptr, .len = b.len };
}
fn kernelSpan(b: cuda.DeviceBuffer, dtype: ops.DataType) ops.Span {
    return .{ .ptr = b.ptr, .len = b.len, .dtype = dtype };
}

pub const Engine = struct {
    a: std.mem.Allocator,
    weights: *const memory.DeviceWeights,
    stream: cuda.Stream,
    kernels: ops.Kernels,
    projections: conv.Projections,
    sdpa: attention.Attention,
    linears: []lt.Linear,
    activations: memory.Activations,
    mutex: std.atomic.Mutex = .unlocked,
    poisoned: bool = false,
    published: bool = false,
    /// All 124 linear owners (120 biased block projections, four unbiased
    /// merger projections) are initialized before activation allocation.
    /// Missing/unqualified runtime or an unsupported algorithm never yields an
    /// executable partial owner. This does not open libraries/create a context.
    pub fn init(a: std.mem.Allocator, weights: *const memory.DeviceWeights, stream: cuda.Stream, g: memory.Geometry, runtime: ?Runtime, options: Options) !Engine {
        const r = runtime orelse return error.MissingVisionRuntime;
        if (r.package != .target_qualification) return error.UnqualifiedVisionRuntime;
        const c = weights.layout.c;
        try exact(c);
        if (stream.d != weights.pool.d) return error.InvalidOperatorStream;
        const rows = try g.rows();
        _ = try memory.ActivationLayout.init(c, g, .{}, options.max_activation_bytes);
        if (weights.layout.regions.len != 347) return error.InvalidVisionWeightTable;
        // Validate every borrowed span before operator construction.
        for (weights.layout.regions, 0..) |region, i| {
            if (region.dtype != .bf16) return error.UnsupportedVisionDType;
            const b = try weights.tensor(i);
            if (b.ptr == 0 or b.ptr % 256 != 0) return error.InvalidVisionDeviceAllocation;
        }
        const linears = try a.alloc(lt.Linear, 24 * 5 + 4);
        errdefer a.free(linears);
        var initialized: usize = 0;
        errdefer for (0..initialized) |i| linears[initialized - 1 - i].deinit();
        var kernels = try ops.Kernels.load(stream.d, r.image);
        errdefer kernels.deinit();
        var projections = try conv.Projections.init(r.convolution, c, rows, .{ .workspace_limit = options.workspace_limit });
        errdefer projections.deinit();
        var sdpa = try attention.Attention.init(r.sdpa, stream.d, c, .{ .batch = g.groups, .sequence = try g.sequence() }, .separate_qkv, .{ .workspace_limit = options.workspace_limit });
        errdefer sdpa.deinit();
        var linear_workspace: usize = 0;
        for (0..24) |layer| for (block_specs) |spec| {
            const bias = try weights.block(@intCast(layer), spec.bias);
            linears[initialized] = try lt.Linear.initBias(r.linear, rows, spec.n, spec.k, linearSpan(bias), options.workspace_limit);
            linear_workspace = @max(linear_workspace, linears[initialized].workspace_need);
            initialized += 1;
        };
        for (merger_specs) |spec| {
            linears[initialized] = try lt.Linear.init(r.linear, rows / 4, spec.n, spec.k, .bf16, options.workspace_limit);
            linear_workspace = @max(linear_workspace, linears[initialized].workspace_need);
            initialized += 1;
        }
        const activations = try memory.Activations.init(a, stream.d, c, g, .{
            .convolution = @max(projections.patch.workspace_need, projections.downsample.workspace_need),
            .attention = sdpa.operator.workspace_need,
            .linear = linear_workspace,
        }, options.max_activation_bytes);
        return .{ .a = a, .weights = weights, .stream = stream, .kernels = kernels, .projections = projections, .sdpa = sdpa, .linears = linears, .activations = activations };
    }
    fn buffer(self: *const Engine, slot: Slot) !cuda.DeviceBuffer {
        return self.activations.buffer(slot);
    }
    fn bf(self: *const Engine, slot: Slot) !ops.Span {
        return kernelSpan(try self.buffer(slot), .bf16);
    }
    fn root(self: *const Engine, kind: memory.host.Root) !ops.Span {
        return kernelSpan(try self.weights.root(kind), .bf16);
    }
    fn blockWeight(self: *const Engine, layer: u32, kind: memory.host.Block) !ops.Span {
        return kernelSpan(try self.weights.block(layer, kind), .bf16);
    }
    fn linear(self: *const Engine, index: usize, x: Slot, w: cuda.DeviceBuffer, out: Slot) !void {
        try self.linears[index].runChecked(linearSpan(try self.buffer(x)), linearSpan(w), linearSpan(try self.buffer(out)), linearSpan(try self.buffer(.workspace)), self.stream.handle);
    }
    fn convolution(self: *const Engine, op: *const conv.cudnn.Convolution, input: Slot, weight: memory.host.Root, bias: memory.host.Root, output: Slot) !void {
        try op.run(.{ .input = convSpan(try self.buffer(input)), .filter = convSpan(try self.weights.root(weight)), .bias = convSpan(try self.weights.root(bias)), .output = convSpan(try self.buffer(output)), .workspace = convSpan(try self.buffer(.workspace)) }, self.stream.handle);
    }
    fn dispatch(self: *Engine) !void {
        const c = self.weights.layout.c;
        const g = self.activations.layout.geometry;
        const rows = try g.rows();
        const matrix: ops.Matrix = .{ .rows = rows, .cols = 1024 };
        try self.kernels.cast(self.stream, @as(u64, rows) * 1176, kernelSpan(try self.buffer(.pixels), .f32), try self.bf(.patches));
        try self.convolution(&self.projections.patch, .patches, .patch_weight, .patch_bias, .x);
        for (0..24) |layer| {
            const l: u32 = @intCast(layer);
            const base = layer * 5;
            try self.kernels.rms(self.stream, matrix, c.eps, try self.bf(.x), try self.blockWeight(l, .norm1), try self.bf(.norm));
            try self.linear(base, .norm, try self.weights.block(l, .qkv_weight), .qkv);
            try self.kernels.qkvRope(self.stream, .{ .groups = g.groups, .sequence = try g.sequence(), .heads = 16, .dim = 64 }, c.eps, try self.bf(.qkv), try self.blockWeight(l, .q_norm), try self.blockWeight(l, .k_norm), kernelSpan(try self.buffer(.cosine), .f32), kernelSpan(try self.buffer(.sine), .f32), try self.bf(.q), try self.bf(.k), try self.bf(.v));
            try self.sdpa.run(.{ .separate_qkv = .{ .q = try self.buffer(.q), .k = try self.buffer(.k), .v = try self.buffer(.v), .output = try self.buffer(.attention), .workspace = try self.buffer(.workspace) } }, self.stream);
            try self.linear(base + 1, .attention, try self.weights.block(l, .attention_weight), .update);
            try self.kernels.residual(self.stream, @as(u64, rows) * 1024, try self.bf(.x), try self.bf(.update), try self.bf(.x));
            try self.kernels.rms(self.stream, matrix, c.eps, try self.bf(.x), try self.blockWeight(l, .norm2), try self.bf(.norm));
            try self.linear(base + 2, .norm, try self.weights.block(l, .gate_weight), .gate);
            try self.linear(base + 3, .norm, try self.weights.block(l, .up_weight), .up);
            try self.kernels.swiglu(self.stream, @as(u64, rows) * 4096, c.limit, try self.bf(.gate), try self.bf(.up), try self.bf(.gate));
            try self.linear(base + 4, .gate, try self.weights.block(l, .down_weight), .update);
            try self.kernels.residual(self.stream, @as(u64, rows) * 1024, try self.bf(.x), try self.bf(.update), try self.bf(.x));
        }
        try self.kernels.rms(self.stream, matrix, c.eps, try self.bf(.x), try self.root(.post_norm), try self.bf(.norm));
        try self.convolution(&self.projections.downsample, .norm, .downsample_weight, .downsample_bias, .merged);
        try self.linear(120, .merged, try self.weights.root(.projection), .projected);
        // The helper explicitly casts LayerNorm affine output to BF16 BEFORE
        // erf-GELU, preserving both contiguous source operations' boundary.
        try self.kernels.projectorNormGelu(self.stream, .{ .rows = rows / 4, .cols = 4096 }, try self.bf(.projected), try self.root(.projection_norm), try self.root(.projection_bias), try self.bf(.projected));
        try self.linear(121, .projected, try self.weights.root(.gate), .projector_gate);
        try self.linear(122, .projected, try self.weights.root(.up), .projector_up);
        try self.kernels.swiglu(self.stream, @as(u64, rows / 4) * 10240, c.limit, try self.bf(.projector_gate), try self.bf(.projector_up), try self.bf(.projector_gate));
        try self.linear(123, .projector_gate, try self.weights.root(.down), .output);
    }
    fn begin(self: *Engine) !void {
        if (!self.mutex.tryLock()) return error.VisionBusy;
        if (self.poisoned) {
            self.mutex.unlock();
            return error.VisionEncoderPoisoned;
        }
        self.published = false;
        self.stream.synchronize() catch |err| {
            self.poisoned = true;
            self.mutex.unlock();
            return err;
        };
    }
    fn finish(self: *Engine) !cuda.DeviceBuffer {
        self.dispatch() catch |err| {
            self.poisoned = true;
            return err;
        };
        self.stream.synchronize() catch |err| {
            self.poisoned = true;
            return err;
        };
        const out = try self.buffer(.output);
        self.published = true;
        return out;
    }
    /// Publishes ONLY after complete native dispatch and stream completion.
    /// The borrowed result is invalidated by the next encode or successful deinit.
    pub fn encodePixels(self: *Engine, pixels: []const f32) !cuda.DeviceBuffer {
        try self.begin();
        defer self.mutex.unlock();
        self.activations.stagePixels(pixels) catch |err| {
            if (err != error.InvalidVisionGrid and err != error.NonFiniteVisionPixels) self.poisoned = true;
            return err;
        };
        return self.finish();
    }
    pub fn encodePrepared(self: *Engine, images: []const pre.Prepared) !cuda.DeviceBuffer {
        try self.begin();
        defer self.mutex.unlock();
        self.activations.stagePrepared(images) catch |err| {
            if (err != error.InvalidVisionGrid and err != error.NonFiniteVisionPixels) self.poisoned = true;
            return err;
        };
        return self.finish();
    }
    /// On sync failure, retains EVERY owner so pending work cannot use freed
    /// resources. Caller must recover/complete context work and retry cleanup.
    pub fn deinit(self: *Engine) !void {
        if (!self.mutex.tryLock()) return error.VisionBusy;
        self.stream.synchronize() catch |err| {
            self.published = false;
            self.poisoned = true;
            self.mutex.unlock();
            return err;
        };
        self.published = false;
        for (0..self.linears.len) |i| self.linears[self.linears.len - 1 - i].deinit();
        self.a.free(self.linears);
        self.sdpa.deinit();
        self.projections.deinit();
        self.kernels.deinit();
        self.activations.deinit();
        self.* = undefined;
    }
};
test {
    _ = @import("cuda_vision_engine_test.zig");
}
