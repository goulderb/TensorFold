const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const checkpoint = @import("cuda_checkpoint.zig");
const Config = checkpoint.Config;
const Tensor = core.checkpoint.Tensor;
const Allocator = std.mem.Allocator;
const Io = std.Io;
const Buffer = cuda.DeviceBuffer;
const Nvfp4Experts = cuda.experts.Nvfp4Experts;

pub const Projection = cuda.qlinear.Weight;
pub const Dense = struct { gate: Projection, up: Projection, down: Projection };
pub const Moe = struct {
    router: u64,
    shared_gate: u64,
    experts: cuda.experts.Layer,
    shared: Dense,
};
pub const Full = struct {
    q: Projection,
    k: Projection,
    v: Projection,
    out: Projection,
    q_norm: u64,
    k_norm: u64,
};
pub const Gdn = struct {
    qkv: Projection,
    z: Projection,
    a: Projection,
    b: Projection,
    out: Projection,
    conv: u64,
    a_log: u64,
    dt_bias: u64,
    norm: u64,
};
pub const Layer = struct {
    input_norm: u64,
    post_norm: u64,
    moe: Moe,
    mixer: union(enum) { full: Full, gdn: Gdn },
};

pub const gpu_floor_bytes: usize = 1536 * 1024 * 1024;

pub const MemoryPlan = struct {
    weights: usize = 0,
    staging: usize = 0,

    fn tensor(p: *MemoryPlan, name: []const u8, t: Tensor) !void {
        if (std.mem.endsWith(u8, name, ".input_scale")) return;
        const expert = std.mem.indexOf(u8, name, ".mlp.experts.") != null;
        if (std.mem.endsWith(u8, name, ".weight_scale_2")) {
            if (expert) p.weights = try std.math.add(usize, p.weights, 4);
            return;
        }
        if (std.mem.endsWith(u8, name, ".weight_scale")) return;
        var elements: usize = 1;
        for (t.shape[0..t.rank]) |dim| elements = try std.math.mul(usize, elements, dim);
        const raw = std.mem.eql(u8, name, "model.language_model.embed_tokens.weight") or
            std.mem.endsWith(u8, name, ".mlp.gate.weight") or
            std.mem.endsWith(u8, name, ".mlp.shared_expert_gate.weight");
        var bytes: usize = undefined;
        if (raw) {
            bytes = try std.math.mul(usize, elements, t.dtype.size());
        } else if (t.rank == 2) {
            const n = t.dim(0);
            const k = t.dim(1);
            if (expert) {
                bytes = Nvfp4Experts.bytes(1, n, try std.math.mul(usize, k, 2), 1);
            } else {
                const padded_elements = try std.math.mul(usize, cuda.modelopt.padded(n), k);
                bytes = try std.math.mul(usize, padded_elements, t.dtype.size());
                if (t.dtype == .u8) bytes = try std.math.add(usize, bytes, padded_elements / 8);
            }
            var staging = try std.math.mul(usize, elements, t.dtype.size());
            if (t.dtype == .u8) staging = try std.math.add(usize, staging, elements / 8);
            p.staging = @max(p.staging, staging);
        } else {
            bytes = try std.math.mul(usize, elements, @sizeOf(f32));
        }
        p.weights = try std.math.add(usize, p.weights, bytes);
    }
};

pub fn memoryPlan(source: anytype, cfg: Config, first: usize, end: usize, with_embed: bool, with_head: bool) !MemoryPlan {
    // Admit the shapes before the packed-layout allocation arithmetic uses them.
    try checkpoint.validateText(source, cfg, first, end, with_embed, with_head);
    const Source = @TypeOf(source);
    const Planned = struct {
        source: Source,
        plan: MemoryPlan = .{},

        pub fn has(s: *const @This(), name: []const u8) bool {
            return s.source.has(name);
        }

        pub fn get(s: *@This(), name: []const u8) !Tensor {
            const t = try s.source.get(name);
            try s.plan.tensor(name, t);
            return t;
        }
    };
    var planned: Planned = .{ .source = source };
    try checkpoint.validateText(&planned, cfg, first, end, with_embed, with_head);
    return planned.plan;
}

pub fn planPartition(gpa: Allocator, io: Io, dir: []const u8, cfg: Config, first: usize, end: usize) !MemoryPlan {
    var ck = try core.Checkpoint.openModel(gpa, io, dir);
    defer ck.close();
    const plan = try memoryPlan(&ck, cfg, first, end, first == 0, end == 40);
    try checkpoint.refuseUnusedText(&ck, first, end, first == 0, end == 40);
    return plan;
}

pub const Weights = struct {
    layers: []Layer,
    embed: ?u64 = null,
    head: ?Projection = null,
    norm: ?u64 = null,
    device_bytes: usize = 0,
    gpa: Allocator,
    ctx: *const cuda.Context,
    buffers: std.ArrayList(Buffer) = .empty,

    pub fn load(gpa: Allocator, io: Io, ctx: *const cuda.Context, dir: []const u8, cfg: Config, first: usize, end: usize, with_embed: bool, with_head: bool) !Weights {
        if (first > end or end > cfg.spec.n_layers) return error.InvalidPartition;
        var ck = try core.Checkpoint.openModel(gpa, io, dir);
        defer ck.close();
        try checkpoint.validateText(&ck, cfg, first, end, with_embed, with_head);
        try checkpoint.refuseUnusedText(&ck, first, end, with_embed, with_head);
        try ctx.makeCurrent();
        try floor(ctx, 0);
        var out: Weights = .{ .layers = try gpa.alloc(Layer, end - first), .gpa = gpa, .ctx = ctx };
        errdefer out.deinit();
        var stream = try cuda.Stream.init(ctx.d, true);
        defer stream.deinit();
        var pack = try cuda.modelopt.Packer.load(ctx.d);
        defer pack.unload();
        var loader: Loader = .{ .weights = &out, .stream = stream, .pack = &pack, .view = .{ .ck = &ck } };
        if (with_embed) out.embed = try loader.raw(try ck.expect("model.language_model.embed_tokens.weight", .bf16, &.{ 248320, 2048 }));
        if (with_head) {
            out.head = try loader.projection("lm_head", 248320, 2048);
            out.norm = try loader.floatTensor(try ck.expect("model.language_model.norm.weight", .bf16, &.{2048}), .zero_centered_norm);
        }
        for (out.layers, first..) |*layer, i| layer.* = try loader.layer(i, cfg.spec.full(i));
        try stream.synchronize();
        try floor(ctx, 0);
        return out;
    }

    pub fn totalDeviceBytes(self: *const Weights) usize {
        return self.device_bytes;
    }

    pub fn deinit(self: *Weights) void {
        defer {
            self.buffers.deinit(self.gpa);
            self.gpa.free(self.layers);
            self.* = undefined;
        }
        self.ctx.makeCurrent() catch return;
        self.ctx.synchronize() catch {};
        for (self.buffers.items) |*buffer| buffer.free();
    }
};

fn floor(ctx: *const cuda.Context, need: usize) !void {
    const memory = try ctx.memInfo();
    if (memory.free < gpu_floor_bytes or need > memory.free - gpu_floor_bytes) return error.GpuMemoryFloor;
}

const Loader = struct {
    weights: *Weights,
    stream: cuda.Stream,
    pack: *const cuda.modelopt.Packer,
    view: checkpoint.View,

    fn alloc(l: *Loader, bytes: usize) !u64 {
        try floor(l.weights.ctx, bytes);
        try l.weights.buffers.ensureUnusedCapacity(l.weights.gpa, 1);
        var buffer = try Buffer.alloc(l.weights.ctx.d, bytes);
        errdefer buffer.free();
        try floor(l.weights.ctx, 0);
        l.weights.buffers.appendAssumeCapacity(buffer);
        l.weights.device_bytes += bytes;
        return buffer.ptr;
    }

    fn staging(l: *Loader, bytes: []const u8) !Buffer {
        try floor(l.weights.ctx, bytes.len);
        var buffer = try Buffer.alloc(l.weights.ctx.d, bytes.len);
        errdefer buffer.free();
        try floor(l.weights.ctx, 0);
        try buffer.upload(0, bytes);
        return buffer;
    }

    fn upload(l: *Loader, bytes: []const u8) !u64 {
        const ptr = try l.alloc(bytes.len);
        const buffer: Buffer = .{ .d = l.weights.ctx.d, .ptr = ptr, .len = bytes.len, .borrowed = true };
        try buffer.upload(0, bytes);
        return ptr;
    }

    fn raw(l: *Loader, t: Tensor) !u64 {
        return l.upload(t.bytes);
    }

    fn floatTensor(l: *Loader, t: Tensor, role: checkpoint.FloatRole) !u64 {
        const values = try checkpoint.floats(l.weights.gpa, t, role);
        defer l.weights.gpa.free(values);
        return l.upload(std.mem.sliceAsBytes(values));
    }

    fn projection(l: *Loader, pre: []const u8, n: usize, k: usize) !Projection {
        const p = try l.view.part(pre, n, k);
        var src = try l.staging(p.w.bytes);
        defer src.free();
        defer l.stream.synchronize() catch {};
        switch (p.kind) {
            .bf16 => {
                const out = try l.alloc(cuda.modelopt.padded(n) * k * 2);
                try l.pack.bf16Lane(l.stream, src.ptr, out, n, k);
                try l.stream.synchronize();
                return .{ .bf16 = cuda.modelopt.bf16Weight(out, n, k) };
            },
            .fp8 => {
                const out = try l.alloc(cuda.modelopt.padded(n) * k);
                try l.pack.fp8Lane(l.stream, src.ptr, out, n, k);
                try l.stream.synchronize();
                var weight = cuda.modelopt.fp8Weight(out, p.scale, n, k);
                weight.act = p.act;
                return .{ .fp8 = weight };
            },
            .nvfp4 => {
                var scales = try l.staging(p.blocks.?.bytes);
                defer scales.free();
                defer l.stream.synchronize() catch {};
                const words = try l.alloc(cuda.modelopt.padded(n) * k / 2);
                const tiles = try l.alloc(cuda.modelopt.padded(n) * k / 16);
                try l.pack.fp4Lane(l.stream, src.ptr, scales.ptr, words, tiles, n, k);
                try l.stream.synchronize();
                return .{ .nvfp4 = cuda.nvfp4.weight(words, tiles, p.scale, n, k) };
            },
        }
    }

    fn dense(l: *Loader, pre: []const u8) !Dense {
        var buf: [256]u8 = undefined;
        return .{
            .gate = try l.projection(try checkpoint.name(&buf, "{s}.gate_proj", .{pre}), 512, 2048),
            .up = try l.projection(try checkpoint.name(&buf, "{s}.up_proj", .{pre}), 512, 2048),
            .down = try l.projection(try checkpoint.name(&buf, "{s}.down_proj", .{pre}), 2048, 512),
        };
    }

    fn packExpert(l: *Loader, p: checkpoint.Part, out: u64, parts: usize, part: usize) !void {
        if (p.kind != .nvfp4) return error.UnsupportedQwenExperts;
        var codes = try l.staging(p.w.bytes);
        defer codes.free();
        var scales = try l.staging(p.blocks.?.bytes);
        defer scales.free();
        defer l.stream.synchronize() catch {};
        try l.pack.fp4Experts(l.stream, codes.ptr, scales.ptr, out, 1, p.n, p.k, 0, p.k, parts, part);
        try l.stream.synchronize();
    }

    fn experts(l: *Loader, pre: []const u8) !cuda.experts.Layer {
        const up_one = Nvfp4Experts.bytes(1, 512, 2048, 2);
        const down_one = Nvfp4Experts.bytes(1, 2048, 512, 1);
        const up = try l.alloc(256 * up_one);
        const down = try l.alloc(256 * down_one);
        var up_scales: [256 * 2]f32 = undefined;
        var down_scales: [256]f32 = undefined;
        var buf: [256]u8 = undefined;
        for (0..256) |e| {
            inline for (.{ "gate_proj", "up_proj" }, 0..) |suffix, m| {
                const p = try l.view.part(try checkpoint.name(&buf, "{s}.experts.{d}.{s}", .{ pre, e, suffix }), 512, 2048);
                try l.packExpert(p, up + e * up_one, 2, m);
                up_scales[2 * e + m] = p.scale;
            }
            const p = try l.view.part(try checkpoint.name(&buf, "{s}.experts.{d}.down_proj", .{ pre, e }), 2048, 512);
            try l.packExpert(p, down + e * down_one, 1, 0);
            down_scales[e] = p.scale;
        }
        return .{
            .format = .nvfp4,
            .up = up,
            .down = down,
            .up_scale = try l.upload(std.mem.sliceAsBytes(&up_scales)),
            .down_scale = try l.upload(std.mem.sliceAsBytes(&down_scales)),
            .width = 512,
            .dims = 2048,
            .experts = 256,
            .relu2 = false,
        };
    }

    fn moe(l: *Loader, pre: []const u8) !Moe {
        var buf: [256]u8 = undefined;
        var shared_buf: [192]u8 = undefined;
        return .{
            .router = try l.raw(try l.view.tensor(try checkpoint.name(&buf, "{s}.gate.weight", .{pre}), .bf16, &.{ 256, 2048 })),
            .shared_gate = try l.raw(try l.view.tensor(try checkpoint.name(&buf, "{s}.shared_expert_gate.weight", .{pre}), .bf16, &.{ 1, 2048 })),
            .experts = try l.experts(pre),
            .shared = try l.dense(try checkpoint.name(&shared_buf, "{s}.shared_expert", .{pre})),
        };
    }

    fn layer(l: *Loader, i: usize, full: bool) !Layer {
        var pre_buf: [128]u8 = undefined;
        var buf: [256]u8 = undefined;
        const pre = try checkpoint.name(&pre_buf, "model.language_model.layers.{d}", .{i});
        var out: Layer = undefined;
        out.input_norm = try l.floatTensor(try l.view.tensor(try checkpoint.name(&buf, "{s}.input_layernorm.weight", .{pre}), .bf16, &.{2048}), .zero_centered_norm);
        out.post_norm = try l.floatTensor(try l.view.tensor(try checkpoint.name(&buf, "{s}.post_attention_layernorm.weight", .{pre}), .bf16, &.{2048}), .zero_centered_norm);
        out.moe = try l.moe(try checkpoint.name(&buf, "{s}.mlp", .{pre}));
        if (full) {
            var f: Full = undefined;
            f.q = try l.projection(try checkpoint.name(&buf, "{s}.self_attn.q_proj", .{pre}), 8192, 2048);
            f.k = try l.projection(try checkpoint.name(&buf, "{s}.self_attn.k_proj", .{pre}), 512, 2048);
            f.v = try l.projection(try checkpoint.name(&buf, "{s}.self_attn.v_proj", .{pre}), 512, 2048);
            f.out = try l.projection(try checkpoint.name(&buf, "{s}.self_attn.o_proj", .{pre}), 2048, 4096);
            f.q_norm = try l.floatTensor(try l.view.tensor(try checkpoint.name(&buf, "{s}.self_attn.q_norm.weight", .{pre}), .bf16, &.{256}), .zero_centered_norm);
            f.k_norm = try l.floatTensor(try l.view.tensor(try checkpoint.name(&buf, "{s}.self_attn.k_norm.weight", .{pre}), .bf16, &.{256}), .zero_centered_norm);
            out.mixer = .{ .full = f };
        } else {
            var g: Gdn = undefined;
            g.qkv = try l.projection(try checkpoint.name(&buf, "{s}.linear_attn.in_proj_qkv", .{pre}), 8192, 2048);
            g.z = try l.projection(try checkpoint.name(&buf, "{s}.linear_attn.in_proj_z", .{pre}), 4096, 2048);
            g.a = try l.projection(try checkpoint.name(&buf, "{s}.linear_attn.in_proj_a", .{pre}), 32, 2048);
            g.b = try l.projection(try checkpoint.name(&buf, "{s}.linear_attn.in_proj_b", .{pre}), 32, 2048);
            g.out = try l.projection(try checkpoint.name(&buf, "{s}.linear_attn.out_proj", .{pre}), 2048, 4096);
            g.conv = try l.floatTensor(try l.view.tensor(try checkpoint.name(&buf, "{s}.linear_attn.conv1d.weight", .{pre}), .bf16, &.{ 8192, 1, 4 }), .raw);
            g.a_log = try l.floatTensor(try l.view.tensor(try checkpoint.name(&buf, "{s}.linear_attn.A_log", .{pre}), .bf16, &.{32}), .raw);
            g.dt_bias = try l.floatTensor(try l.view.tensor(try checkpoint.name(&buf, "{s}.linear_attn.dt_bias", .{pre}), .bf16, &.{32}), .raw);
            g.norm = try l.floatTensor(try l.view.tensor(try checkpoint.name(&buf, "{s}.linear_attn.norm.weight", .{pre}), .bf16, &.{128}), .raw);
            out.mixer = .{ .gdn = g };
        }
        return out;
    }
};
