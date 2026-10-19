const std = @import("std");
const core = @import("core");
const general = @import("config.zig");
const checks = @import("config_checks.zig");
const dimensions = @import("cuda_geometry.zig");
pub const Geometry = dimensions.Geometry;
const Tensor = core.checkpoint.Tensor;
const DType = core.checkpoint.DType;

pub const Numerics = enum { mlx_sanitized, hf_modelopt };
pub const Encoding = union(enum) { mlx, modelopt: Config };
pub const Admitted = struct {
    geometry: Geometry,
    encoding: Encoding,

    pub fn numerics(a: Admitted) Numerics {
        return switch (a.encoding) {
            .mlx => .mlx_sanitized,
            .modelopt => .hf_modelopt,
        };
    }
};

pub const Placement = struct {
    first: usize,
    end: usize,
    total: usize,

    pub fn validate(p: Placement) !void {
        if (p.first >= p.end or p.end > p.total or p.total > dimensions.max_layers) return error.InvalidPartition;
    }

    pub fn ownsEmbedding(p: Placement) bool {
        return p.first == 0;
    }
    pub fn ownsHead(p: Placement) bool {
        return p.end == p.total;
    }

    pub fn localGeometry(p: Placement, global: Geometry) !Geometry {
        try p.validate();
        if (p.total != global.layers) return error.InvalidPartition;
        var local = global;
        local.layers = p.end - p.first;
        local.linear = @splat(false);
        for (p.first..p.end, 0..) |i, j| {
            local.linear[j] = global.linear[i];
            local.bits.proj[j] = global.bits.proj[i];
        }
        return local;
    }
};

pub fn admit(gpa: std.mem.Allocator, bytes: []const u8) !Admitted {
    if (try claimsBytes(gpa, bytes)) {
        const cfg = try Config.parse(gpa, bytes);
        return .{ .geometry = cfg.geometry, .encoding = .{ .modelopt = cfg } };
    }
    return .{ .geometry = try dimensions.parse(gpa, bytes), .encoding = .mlx };
}

/// Claim the encoding before geometry validation so malformed ModelOpt never falls through to MLX.
pub fn claims(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !bool {
    const path = try std.fs.path.join(gpa, &.{ dir, "config.json" });
    defer gpa.free(path);
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 22));
    defer gpa.free(bytes);
    return claimsBytes(gpa, bytes);
}

pub fn claimsBytes(gpa: std.mem.Allocator, bytes: []const u8) !bool {
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, bytes, .{ .allocate = .alloc_always });
    defer parsed.deinit();
    if (parsed.value != .object) return error.UnsupportedQwenQuantization;
    const quant = parsed.value.object.get("quantization_config") orelse return false;
    if (quant != .object) return error.UnsupportedQwenQuantization;
    if (isString(quant.object.get("quant_method"), "modelopt")) return true;
    return quant.object.contains("quant_algo") or quant.object.contains("quantized_layers") or quant.object.contains("config_groups");
}

pub const Config = struct {
    spec: general.Spec,
    context: usize,
    geometry: Geometry,

    pub fn read(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !Config {
        const path = try std.fs.path.join(gpa, &.{ dir, "config.json" });
        defer gpa.free(path);
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 22));
        defer gpa.free(bytes);
        return parse(gpa, bytes);
    }

    pub fn parse(gpa: std.mem.Allocator, bytes: []const u8) !Config {
        var arena: std.heap.ArenaAllocator = .init(gpa);
        defer arena.deinit();
        const doc = try general.declared(arena.allocator(), bytes);
        const tower = try general.textTower(doc);
        try checks.computes(tower);
        const expected_dimensions = .{
            .{ "hidden_size", 2048 },          .{ "vocab_size", 248320 },         .{ "num_hidden_layers", 40 },
            .{ "num_attention_heads", 16 },    .{ "num_key_value_heads", 2 },     .{ "head_dim", 256 },
            .{ "linear_num_key_heads", 16 },   .{ "linear_num_value_heads", 32 }, .{ "linear_key_head_dim", 128 },
            .{ "linear_value_head_dim", 128 }, .{ "linear_conv_kernel_dim", 4 },  .{ "num_experts", 256 },
            .{ "num_experts_per_tok", 8 },     .{ "moe_intermediate_size", 512 }, .{ "shared_expert_intermediate_size", 512 },
        };
        inline for (expected_dimensions) |d| if (!isInt(tower.get(d[0]), d[1])) return error.UnsupportedQwenGeometry;
        const g = try dimensions.dimensions(doc);
        var spec: general.Spec = .{
            .hidden = g.hidden,
            .intermediate = try optionalCount(tower, "intermediate_size"),
            .n_layers = g.layers,
            .heads = g.q_heads,
            .kv_heads = g.kv_heads,
            .head_dim = dimensions.head_dim,
            .key_heads = g.k_heads,
            .value_heads = g.v_heads,
            .key_dim = dimensions.dk,
            .value_dim = dimensions.dk,
            .conv = dimensions.taps,
            .vocab = g.vocab,
            .eps = switch (tower.get("rms_norm_eps").?) {
                .float => |v| v,
                .integer => |v| @floatFromInt(v),
                else => return error.UnsupportedQwenGeometry,
            },
            .rope_theta = dimensions.theta,
            .rotary_dim = 64,
            .full_every = 4,
            .bits = 4,
            .group = 16,
            .experts = g.experts,
            .top_k = g.top_k,
            .moe_width = g.intermediate,
            .shared_width = g.intermediate,
            .mtp_layers = try optionalCount(tower, "mtp_num_hidden_layers"),
            .max_position = g.context,
        };
        for (0..g.layers) |i| spec.full_layers[i] = !g.linear[i];
        if (tower.get("full_attention_interval")) |v| if (!isInt(v, 4)) return error.UnsupportedQwenGeometry;
        if (tower.get("norm_topk_prob")) |v| if (!isBool(v, true)) return error.UnsupportedQwenGeometry;
        try admitGeometry(spec);
        if (!isString(doc.get("model_type"), "qwen3_5_moe") or
            !isString(tower.get("model_type"), "qwen3_5_moe_text")) return error.UnsupportedQwenGeometry;
        for ([_]std.json.ObjectMap{ doc, tower }) |o| {
            const tied = o.get("tie_word_embeddings") orelse return error.UnsupportedTiedHead;
            if (tied != .bool or tied.bool) return error.UnsupportedTiedHead;
        }
        const rope = try object(tower.get("rope_parameters") orelse return error.UnsupportedQwenRotary);
        spec.mrope = try checks.mrope(rope, spec.rotary_dim);
        if (!isString(rope.get("rope_type"), "default") or
            !std.meta.eql(spec.mrope.section, [3]usize{ 11, 11, 10 }) or !spec.mrope.interleaved)
            return error.UnsupportedQwenRotary;
        const q = try object(doc.get("quantization_config") orelse return error.UnsupportedQwenQuantization);
        if (!isString(q.get("quant_method"), "modelopt") or !isString(q.get("quant_algo"), "MIXED_PRECISION"))
            return error.UnsupportedQwenQuantization;
        const groups = try object(q.get("config_groups") orelse return error.UnsupportedQwenQuantization);
        var fp8 = false;
        var nvfp4 = false;
        var it = groups.iterator();
        while (it.next()) |entry| {
            const group = try object(entry.value_ptr.*);
            const w = try object(group.get("weights") orelse return error.UnsupportedQwenQuantization);
            if (!isString(w.get("type"), "float") or !isBool(w.get("dynamic"), false)) return error.UnsupportedQwenQuantization;
            if (isInt(w.get("num_bits"), 8)) {
                const a = try object(group.get("input_activations") orelse return error.UnsupportedQwenQuantization);
                if (!isInt(a.get("num_bits"), 8) or !isString(a.get("type"), "float") or !isBool(a.get("dynamic"), false))
                    return error.UnsupportedQwenQuantization;
                fp8 = true;
            } else if (isInt(w.get("num_bits"), 4) and isInt(w.get("group_size"), 16)) {
                // ModelOpt's per-layer W4A16_NVFP4 tag overrides the generic group's activation metadata.
                if (group.get("input_activations")) |activation| {
                    const a = try object(activation);
                    if (!isInt(a.get("num_bits"), 4) or !isString(a.get("type"), "float") or
                        !isBool(a.get("dynamic"), false) or !isInt(a.get("group_size"), 16))
                        return error.UnsupportedQwenQuantization;
                }
                nvfp4 = true;
            } else return error.UnsupportedQwenQuantization;
        }
        if (!fp8 or !nvfp4) return error.UnsupportedQwenQuantization;
        const linears = try object(q.get("quantized_layers") orelse return error.UnsupportedQwenQuantization);
        if (linears.count() == 0) return error.UnsupportedQwenQuantization;
        var projections = linears.iterator();
        while (projections.next()) |entry| {
            const p = try object(entry.value_ptr.*);
            if (isString(p.get("quant_algo"), "FP8")) continue;
            if (!isString(p.get("quant_algo"), "W4A16_NVFP4") or !isInt(p.get("group_size"), 16))
                return error.UnsupportedQwenQuantization;
        }
        return .{ .spec = spec, .context = spec.max_position, .geometry = g };
    }
};

fn admitGeometry(s: general.Spec) !void {
    if (s.hidden != 2048 or s.vocab != 248320 or s.n_layers != 40 or
        s.heads != 16 or s.kv_heads != 2 or s.head_dim != 256 or
        s.key_heads != 16 or s.value_heads != 32 or s.key_dim != 128 or s.value_dim != 128 or
        s.conv != 4 or s.experts != 256 or s.top_k != 8 or s.moe_width != 512 or s.shared_width != 512 or
        s.full_every != 4 or s.rotary_dim != 64 or s.rope_theta != 10_000_000 or s.eps != 1e-6 or
        s.max_position > 262144) return error.UnsupportedQwenGeometry;
    for (0..s.n_layers) |i| if (s.full(i) != ((i + 1) % 4 == 0)) return error.UnsupportedQwenGeometry;
}

fn object(v: std.json.Value) !std.json.ObjectMap {
    return if (v == .object) v.object else error.UnsupportedQwenQuantization;
}
fn optionalCount(o: std.json.ObjectMap, key: []const u8) !usize {
    const v = o.get(key) orelse return 0;
    if (v != .integer or v.integer < 0) return error.UnsupportedQwenGeometry;
    return @intCast(v.integer);
}
fn isString(v: ?std.json.Value, text: []const u8) bool {
    const x = v orelse return false;
    return x == .string and std.mem.eql(u8, x.string, text);
}
fn isInt(v: ?std.json.Value, n: i64) bool {
    const x = v orelse return false;
    return x == .integer and x.integer == n;
}
fn isBool(v: ?std.json.Value, b: bool) bool {
    const x = v orelse return false;
    return x == .bool and x.bool == b;
}

pub const Kind = enum { bf16, fp8, nvfp4 };
pub const Part = struct {
    kind: Kind,
    w: Tensor,
    blocks: ?Tensor = null,
    scale: f32 = 1,
    act: f32 = 0,
    n: usize,
    k: usize,
};

pub fn name(buf: []u8, comptime fmt: []const u8, args: anytype) ![]const u8 {
    return std.fmt.bufPrint(buf, fmt, args);
}

fn expect(t: Tensor, dtype: DType, shape: []const usize) !void {
    if (!t.is(dtype, shape)) return error.UnexpectedTensor;
}
fn scalarShape(t: Tensor) !void {
    if (t.dtype != .f32 or !(t.rank == 0 or (t.rank == 1 and t.dim(0) == 1))) return error.UnexpectedTensor;
}

/// Only metadata is read here, so captured checkpoint headers exercise the same admission as mapped weights.
pub fn projectionLayout(source: anytype, pre: []const u8, n: usize, k: usize) !Kind {
    var buf: [256]u8 = undefined;
    const w = try source.get(try name(&buf, "{s}.weight", .{pre}));
    const kind: Kind = switch (w.dtype) {
        .bf16 => .bf16,
        .f8_e4m3 => .fp8,
        .u8 => .nvfp4,
        else => return error.UnsupportedQwenProjection,
    };
    try expect(w, w.dtype, &.{ n, if (kind == .nvfp4) k / 2 else k });
    if (n == 0 or k == 0 or k % 64 != 0) return error.UnexpectedTensor;
    switch (kind) {
        .bf16 => {
            for ([_][]const u8{ "weight_scale", "weight_scale_2", "input_scale" }) |suffix|
                if (source.has(try name(&buf, "{s}.{s}", .{ pre, suffix }))) return error.UnexpectedTensor;
        },
        .fp8 => {
            try scalarShape(try source.get(try name(&buf, "{s}.weight_scale", .{pre})));
            try scalarShape(try source.get(try name(&buf, "{s}.input_scale", .{pre})));
            if (source.has(try name(&buf, "{s}.weight_scale_2", .{pre}))) return error.UnexpectedTensor;
        },
        .nvfp4 => {
            try expect(try source.get(try name(&buf, "{s}.weight_scale", .{pre})), .f8_e4m3, &.{ n, k / 16 });
            try scalarShape(try source.get(try name(&buf, "{s}.weight_scale_2", .{pre})));
            const act = try name(&buf, "{s}.input_scale", .{pre});
            if (source.has(act)) try scalarShape(try source.get(act));
        },
    }
    return kind;
}

pub const View = struct {
    ck: *core.Checkpoint,

    pub fn tensor(v: View, n: []const u8, dtype: DType, shape: []const usize) !Tensor {
        return v.ck.expect(n, dtype, shape);
    }

    pub fn part(v: View, pre: []const u8, n: usize, k: usize) !Part {
        return readPart(v.ck, pre, n, k);
    }
};

pub fn readPart(source: anytype, pre: []const u8, n: usize, k: usize) !Part {
    const kind = try projectionLayout(source, pre, n, k);
    var buf: [256]u8 = undefined;
    var p: Part = .{ .kind = kind, .w = try source.get(try name(&buf, "{s}.weight", .{pre})), .n = n, .k = k };
    switch (kind) {
        .bf16 => {},
        .fp8 => {
            p.scale = try scalar(try source.get(try name(&buf, "{s}.weight_scale", .{pre})));
            p.act = try scalar(try source.get(try name(&buf, "{s}.input_scale", .{pre})));
        },
        .nvfp4 => {
            p.blocks = try source.get(try name(&buf, "{s}.weight_scale", .{pre}));
            p.scale = try scalar(try source.get(try name(&buf, "{s}.weight_scale_2", .{pre})));
            const act = try name(&buf, "{s}.input_scale", .{pre});
            if (source.has(act)) p.act = try scalar(try source.get(act));
        },
    }
    return p;
}

pub fn refuseUnusedText(ck: *const core.Checkpoint, first: usize, end: usize, with_embed: bool, with_head: bool) !void {
    for (ck.files.items) |file| for (file.names.keys()) |n| {
        if (ck.used.contains(n)) continue;
        if (std.mem.startsWith(u8, n, "lm_head.")) {
            if (with_head) return error.UnexpectedQwenTensor;
            continue;
        }
        const base = "model.language_model.";
        if (!std.mem.startsWith(u8, n, base)) continue;
        if (std.mem.eql(u8, n, base ++ "embed_tokens.weight") and !with_embed) continue;
        if (std.mem.eql(u8, n, base ++ "norm.weight") and !with_head) continue;
        const layers = base ++ "layers.";
        if (std.mem.startsWith(u8, n, layers)) {
            const suffix = n[layers.len..];
            const dot = std.mem.indexOfScalar(u8, suffix, '.') orelse return error.UnexpectedQwenTensor;
            const index = std.fmt.parseInt(usize, suffix[0..dot], 10) catch return error.UnexpectedQwenTensor;
            if (index >= 40) return error.UnexpectedQwenTensor;
            if (index < first or index >= end) continue;
        }
        return error.UnexpectedQwenTensor;
    };
}

pub fn scalar(t: Tensor) !f32 {
    try scalarShape(t);
    if (t.bytes.len != 4) return error.UnexpectedTensor;
    const f: f32 = @bitCast(std.mem.readInt(u32, t.bytes[0..4], .little));
    if (!std.math.isFinite(f) or f <= 0) return error.InvalidQwenScale;
    return f;
}

pub const FloatRole = enum { raw, zero_centered_norm };

/// HF RMSNorm stores a zero-centered multiplier; GDN's gated norm and state parameters do not.
pub fn floats(gpa: std.mem.Allocator, t: Tensor, role: FloatRole) ![]f32 {
    if (t.dtype != .bf16 and t.dtype != .f32) return error.UnexpectedTensor;
    if (t.bytes.len % t.dtype.size() != 0) return error.UnexpectedTensor;
    const values = try gpa.alloc(f32, t.bytes.len / t.dtype.size());
    errdefer gpa.free(values);
    for (values, 0..) |*v, i| {
        v.* = if (t.dtype == .bf16)
            @bitCast(@as(u32, std.mem.readInt(u16, t.bytes[2 * i ..][0..2], .little)) << 16)
        else
            @bitCast(std.mem.readInt(u32, t.bytes[4 * i ..][0..4], .little));
        if (!std.math.isFinite(v.*)) return error.InvalidQwenParameter;
        if (role == .zero_centered_norm) v.* += 1;
    }
    return values;
}

/// Required text tensors only. Vision and MTP weights belong to explicitly unimplemented capabilities.
pub fn validateText(source: anytype, cfg: Config, first: usize, end: usize, with_embed: bool, with_head: bool) !void {
    if (first > end or end > cfg.spec.n_layers) return error.InvalidPartition;
    if (with_embed) try expect(try source.get("model.language_model.embed_tokens.weight"), .bf16, &.{ 248320, 2048 });
    if (with_head) {
        _ = try projectionLayout(source, "lm_head", 248320, 2048);
        try expect(try source.get("model.language_model.norm.weight"), .bf16, &.{2048});
    }
    var buf: [256]u8 = undefined;
    var pre_buf: [128]u8 = undefined;
    for (first..end) |i| {
        const pre = try name(&pre_buf, "model.language_model.layers.{d}", .{i});
        inline for (.{ "input_layernorm.weight", "post_attention_layernorm.weight" }) |suffix|
            try expect(try source.get(try name(&buf, "{s}.{s}", .{ pre, suffix })), .bf16, &.{2048});
        try expect(try source.get(try name(&buf, "{s}.mlp.gate.weight", .{pre})), .bf16, &.{ 256, 2048 });
        try expect(try source.get(try name(&buf, "{s}.mlp.shared_expert_gate.weight", .{pre})), .bf16, &.{ 1, 2048 });
        for (0..257) |e| {
            var expert_buf: [192]u8 = undefined;
            const expert = if (e < 256) try name(&expert_buf, "{s}.mlp.experts.{d}", .{ pre, e }) else try name(&expert_buf, "{s}.mlp.shared_expert", .{pre});
            inline for (.{ "gate_proj", "up_proj", "down_proj" }) |suffix| {
                const down = comptime std.mem.eql(u8, suffix, "down_proj");
                const kind = try projectionLayout(source, try name(&buf, "{s}.{s}", .{ expert, suffix }), if (down) 2048 else 512, if (down) 512 else 2048);
                if (e < 256 and kind != .nvfp4) return error.UnsupportedQwenExperts;
            }
        }
        if (cfg.spec.full(i)) {
            inline for (.{ .{ "q_proj", 8192, 2048 }, .{ "k_proj", 512, 2048 }, .{ "v_proj", 512, 2048 }, .{ "o_proj", 2048, 4096 } }) |p|
                _ = try projectionLayout(source, try name(&buf, "{s}.self_attn.{s}", .{ pre, p[0] }), p[1], p[2]);
            inline for (.{ "q_norm.weight", "k_norm.weight" }) |suffix|
                try expect(try source.get(try name(&buf, "{s}.self_attn.{s}", .{ pre, suffix })), .bf16, &.{256});
        } else {
            inline for (.{ .{ "in_proj_qkv", 8192, 2048 }, .{ "in_proj_z", 4096, 2048 }, .{ "in_proj_a", 32, 2048 }, .{ "in_proj_b", 32, 2048 }, .{ "out_proj", 2048, 4096 } }) |p|
                _ = try projectionLayout(source, try name(&buf, "{s}.linear_attn.{s}", .{ pre, p[0] }), p[1], p[2]);
            try expect(try source.get(try name(&buf, "{s}.linear_attn.conv1d.weight", .{pre})), .bf16, &.{ 8192, 1, 4 });
            inline for (.{ "A_log", "dt_bias" }) |suffix|
                try expect(try source.get(try name(&buf, "{s}.linear_attn.{s}", .{ pre, suffix })), .bf16, &.{32});
            try expect(try source.get(try name(&buf, "{s}.linear_attn.norm.weight", .{pre})), .bf16, &.{128});
        }
    }
}
