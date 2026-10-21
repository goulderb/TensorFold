//! ModelOpt packing into the shared engine arena. Planning walks the same allocations without device access.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const checkpoint = @import("cuda_checkpoint.zig");
const layers = @import("cuda_layers.zig");
const Tensor = core.checkpoint.Tensor;
const Proj = layers.Proj;
const Dense = layers.Dense;
const Layer = layers.Layer;
const Nvfp4Experts = cuda.experts.Nvfp4Experts;

pub const Plan = struct { weights_bytes: usize = 0, staging_bytes: usize = 0 };

pub fn plan(source: anytype, cfg: checkpoint.Config, placement: checkpoint.Placement) !Plan {
    _ = try placement.localGeometry(cfg.geometry);
    try checkpoint.validateText(source, cfg, placement.first, placement.end, placement.ownsEmbedding(), placement.ownsHead());
    var l: Loader(true, @TypeOf(source), void) = .{ .source = source };
    try l.walk(cfg, placement);
    if (@TypeOf(source) == *core.Checkpoint)
        try checkpoint.refuseUnusedText(source, placement.first, placement.end, placement.ownsEmbedding(), placement.ownsHead());
    return l.sizes;
}

pub fn load(e: anytype, ck: *core.Checkpoint, stage: *cuda.DeviceBuffer, cfg: checkpoint.Config, placement: checkpoint.Placement) !void {
    _ = try placement.localGeometry(cfg.geometry);
    if (e.layers.len != placement.end - placement.first) return error.InvalidPartition;
    try checkpoint.validateText(ck, cfg, placement.first, placement.end, placement.ownsEmbedding(), placement.ownsHead());
    try checkpoint.refuseUnusedText(ck, placement.first, placement.end, placement.ownsEmbedding(), placement.ownsHead());
    var l: Loader(false, *core.Checkpoint, @TypeOf(e)) = .{ .source = ck, .e = e, .stage = stage, .up_layout = e.expert_up_layout };
    try l.walk(cfg, placement);
}

fn Loader(comptime planning: bool, comptime Source: type, comptime Target: type) type {
    return struct {
        const Self = @This();
        source: Source,
        e: Target = if (planning) {} else undefined,
        stage: if (planning) void else *cuda.DeviceBuffer = if (planning) {} else undefined,
        sizes: Plan = .{},
        up_layout: cuda.experts.Nvfp4UpLayout = .lane_major,

        fn alloc(l: *Self, bytes: usize) !u64 {
            const aligned = try std.math.add(usize, bytes, 255);
            l.sizes.weights_bytes = try std.math.add(usize, l.sizes.weights_bytes, aligned & ~@as(usize, 255));
            return if (planning) @intCast(l.sizes.weights_bytes) else l.e.alloc(bytes);
        }

        fn stageNeed(l: *Self, bytes: usize) !void {
            l.sizes.staging_bytes = @max(l.sizes.staging_bytes, bytes);
            if (!planning and bytes > l.stage.len) return error.StageTooSmall;
        }

        fn tensor(l: *Self, name: []const u8, dtype: core.checkpoint.DType, shape: []const usize) !Tensor {
            const t = try l.source.get(name);
            if (!t.is(dtype, shape)) return error.UnexpectedTensor;
            return t;
        }

        fn elements(t: Tensor) !usize {
            var n: usize = 1;
            for (t.shape[0..t.rank]) |d| n = try std.math.mul(usize, n, d);
            return n;
        }

        fn upload(l: *Self, bytes: []const u8) !u64 {
            const ptr = try l.alloc(bytes.len);
            if (!planning) try l.e.arena.upload(ptr - l.e.arena.ptr, bytes);
            return ptr;
        }

        fn raw(l: *Self, t: Tensor) !u64 {
            const ptr = try l.alloc(try std.math.mul(usize, try elements(t), t.dtype.size()));
            if (!planning) try l.e.arena.upload(ptr - l.e.arena.ptr, t.bytes);
            return ptr;
        }

        fn floatTensor(l: *Self, t: Tensor, role: checkpoint.FloatRole) !u64 {
            const ptr = try l.alloc(try std.math.mul(usize, try elements(t), 4));
            if (!planning) {
                const values = try checkpoint.floats(l.e.gpa, t, role);
                defer l.e.gpa.free(values);
                try l.e.arena.upload(ptr - l.e.arena.ptr, std.mem.sliceAsBytes(values));
            }
            return ptr;
        }

        fn part(l: *Self, pre: []const u8, n: usize, k: usize) !checkpoint.Part {
            if (!planning) return checkpoint.readPart(l.source, pre, n, k);
            const kind = try checkpoint.projectionLayout(l.source, pre, n, k);
            var buf: [256]u8 = undefined;
            return .{ .kind = kind, .w = try l.source.get(try checkpoint.name(&buf, "{s}.weight", .{pre})), .n = n, .k = k };
        }

        fn projection(l: *Self, pre: []const u8, n: usize, k: usize) !Proj {
            const p = try l.part(pre, n, k);
            const raw_bytes = n * k * (if (p.kind == .bf16) @as(usize, 2) else 1) / (if (p.kind == .nvfp4) @as(usize, 2) else 1);
            try l.stageNeed(raw_bytes + (if (p.kind == .nvfp4) n * k / 16 else 0));
            if (!planning) try l.stage.upload(0, p.w.bytes);
            switch (p.kind) {
                .bf16 => {
                    const out = try l.alloc(cuda.modelopt.padded(n) * k * 2);
                    if (!planning) {
                        try l.e.modelopt.?.pack.bf16Lane(l.e.stream, l.stage.ptr, out, n, k);
                        try l.e.stream.synchronize();
                    }
                    return .{ .linear = .{ .bf16 = cuda.modelopt.bf16Weight(out, n, k) } };
                },
                .fp8 => {
                    const out = try l.alloc(cuda.modelopt.padded(n) * k);
                    if (!planning) {
                        try l.e.modelopt.?.pack.fp8Lane(l.e.stream, l.stage.ptr, out, n, k);
                        try l.e.stream.synchronize();
                    }
                    var weight = cuda.modelopt.fp8Weight(out, p.scale, n, k);
                    weight.act = p.act;
                    return .{ .linear = .{ .fp8 = weight } };
                },
                .nvfp4 => {
                    const words = try l.alloc(cuda.modelopt.padded(n) * k / 2);
                    const tiles = try l.alloc(cuda.modelopt.padded(n) * k / 16);
                    if (!planning) {
                        try l.stage.upload(raw_bytes, p.blocks.?.bytes);
                        try l.e.modelopt.?.pack.fp4Lane(l.e.stream, l.stage.ptr, l.stage.ptr + raw_bytes, words, tiles, n, k);
                        try l.e.stream.synchronize();
                    }
                    return .{ .linear = .{ .nvfp4 = cuda.nvfp4.weight(words, tiles, p.scale, n, k) } };
                },
            }
        }

        fn walk(l: *Self, cfg: checkpoint.Config, p: checkpoint.Placement) !void {
            if (p.ownsEmbedding()) {
                const embed = try l.raw(try l.tensor("model.language_model.embed_tokens.weight", .bf16, &.{ 248320, 2048 }));
                if (!planning) {
                    l.e.embed_w = embed;
                    l.e.embed_s = 0;
                    l.e.embed_b = 0;
                    l.e.embed_bits = 16;
                }
            }
            if (p.ownsHead()) {
                const head = try l.projection("lm_head", 248320, 2048);
                const norm = try l.floatTensor(try l.tensor("model.language_model.norm.weight", .bf16, &.{2048}), .zero_centered_norm);
                if (!planning) {
                    l.e.terminal_weights = .{ .vocabulary = head };
                    l.e.final_norm = norm;
                }
            }
            for (p.first..p.end, 0..) |global, local| {
                const value = try l.layer(global, cfg.spec.full(global));
                if (!planning) l.e.layers[local] = value;
            }
        }

        fn dense(l: *Self, pre: []const u8) !Dense {
            var buf: [256]u8 = undefined;
            return .{
                .gate = try l.projection(try checkpoint.name(&buf, "{s}.gate_proj", .{pre}), 512, 2048),
                .up = try l.projection(try checkpoint.name(&buf, "{s}.up_proj", .{pre}), 512, 2048),
                .down = try l.projection(try checkpoint.name(&buf, "{s}.down_proj", .{pre}), 2048, 512),
            };
        }

        fn packExpert(l: *Self, p: checkpoint.Part, out: u64, parts: usize, index: usize, layout: cuda.experts.Nvfp4UpLayout) !void {
            if (p.kind != .nvfp4) return error.UnsupportedQwenExperts;
            try l.stageNeed(p.n * p.k / 2 + p.n * p.k / 16);
            if (!planning) {
                try l.stage.upload(0, p.w.bytes);
                try l.stage.upload(p.w.bytes.len, p.blocks.?.bytes);
                try l.e.modelopt.?.pack.fp4ExpertsLayout(l.e.stream, l.stage.ptr, l.stage.ptr + p.w.bytes.len, out, 1, p.n, p.k, 0, p.k, parts, index, layout);
                try l.e.stream.synchronize();
            }
        }

        fn experts(l: *Self, pre: []const u8) !cuda.experts.Layer {
            const up_one = Nvfp4Experts.bytes(1, 512, 2048, 2);
            const down_one = Nvfp4Experts.bytes(1, 2048, 512, 1);
            const up = try l.alloc(256 * up_one);
            const down = try l.alloc(256 * down_one);
            var up_scales: [256 * 2]f32 = undefined;
            var down_scales: [256]f32 = undefined;
            var buf: [256]u8 = undefined;
            for (0..256) |e| {
                inline for (.{ "gate_proj", "up_proj" }, 0..) |suffix, m| {
                    const p = try l.part(try checkpoint.name(&buf, "{s}.experts.{d}.{s}", .{ pre, e, suffix }), 512, 2048);
                    try l.packExpert(p, up + e * up_one, 2, m, l.up_layout);
                    up_scales[2 * e + m] = p.scale;
                }
                const p = try l.part(try checkpoint.name(&buf, "{s}.experts.{d}.down_proj", .{ pre, e }), 2048, 512);
                try l.packExpert(p, down + e * down_one, 1, 0, .lane_major);
                down_scales[e] = p.scale;
            }
            return .{
                .format = .nvfp4,
                .nvfp4_up_layout = l.up_layout,
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

        fn moe(l: *Self, pre: []const u8) !layers.Mlp {
            var buf: [256]u8 = undefined;
            var shared_buf: [192]u8 = undefined;
            const router = try l.tensor(try checkpoint.name(&buf, "{s}.gate.weight", .{pre}), .bf16, &.{ 256, 2048 });
            const gate = try l.tensor(try checkpoint.name(&buf, "{s}.shared_expert_gate.weight", .{pre}), .bf16, &.{ 1, 2048 });
            const ptr = try l.alloc(257 * 2048 * 2);
            if (!planning) {
                try l.e.arena.upload(ptr - l.e.arena.ptr, router.bytes);
                try l.e.arena.upload(ptr - l.e.arena.ptr + router.bytes.len, gate.bytes);
            }
            return .{ .moe = .{
                .router = ptr,
                .experts = try l.experts(pre),
                .shared = .{ .projected = try l.dense(try checkpoint.name(&shared_buf, "{s}.shared_expert", .{pre})) },
            } };
        }

        fn layer(l: *Self, i: usize, full: bool) !Layer {
            var pre_buf: [128]u8 = undefined;
            var buf: [256]u8 = undefined;
            const pre = try checkpoint.name(&pre_buf, "model.language_model.layers.{d}", .{i});
            var out: Layer = undefined;
            out.input_norm = try l.floatTensor(try l.tensor(try checkpoint.name(&buf, "{s}.input_layernorm.weight", .{pre}), .bf16, &.{2048}), .zero_centered_norm);
            out.post_norm = try l.floatTensor(try l.tensor(try checkpoint.name(&buf, "{s}.post_attention_layernorm.weight", .{pre}), .bf16, &.{2048}), .zero_centered_norm);
            out.mlp = try l.moe(try checkpoint.name(&buf, "{s}.mlp", .{pre}));
            if (full) {
                var f: layers.Attention = undefined;
                f.q = try l.projection(try checkpoint.name(&buf, "{s}.self_attn.q_proj", .{pre}), 8192, 2048);
                f.k = try l.projection(try checkpoint.name(&buf, "{s}.self_attn.k_proj", .{pre}), 512, 2048);
                f.v = try l.projection(try checkpoint.name(&buf, "{s}.self_attn.v_proj", .{pre}), 512, 2048);
                f.o = try l.projection(try checkpoint.name(&buf, "{s}.self_attn.o_proj", .{pre}), 2048, 4096);
                f.q_norm = try l.floatTensor(try l.tensor(try checkpoint.name(&buf, "{s}.self_attn.q_norm.weight", .{pre}), .bf16, &.{256}), .zero_centered_norm);
                f.k_norm = try l.floatTensor(try l.tensor(try checkpoint.name(&buf, "{s}.self_attn.k_norm.weight", .{pre}), .bf16, &.{256}), .zero_centered_norm);
                out.mixer = .{ .attention = f };
            } else {
                var g: layers.Delta = undefined;
                g.qkv = try l.projection(try checkpoint.name(&buf, "{s}.linear_attn.in_proj_qkv", .{pre}), 8192, 2048);
                g.z = try l.projection(try checkpoint.name(&buf, "{s}.linear_attn.in_proj_z", .{pre}), 4096, 2048);
                g.a = try l.projection(try checkpoint.name(&buf, "{s}.linear_attn.in_proj_a", .{pre}), 32, 2048);
                g.b = try l.projection(try checkpoint.name(&buf, "{s}.linear_attn.in_proj_b", .{pre}), 32, 2048);
                g.out = try l.projection(try checkpoint.name(&buf, "{s}.linear_attn.out_proj", .{pre}), 2048, 4096);
                g.conv = try l.floatTensor(try l.tensor(try checkpoint.name(&buf, "{s}.linear_attn.conv1d.weight", .{pre}), .bf16, &.{ 8192, 1, 4 }), .raw);
                g.a_log = try l.floatTensor(try l.tensor(try checkpoint.name(&buf, "{s}.linear_attn.A_log", .{pre}), .bf16, &.{32}), .raw);
                g.dt_bias = try l.floatTensor(try l.tensor(try checkpoint.name(&buf, "{s}.linear_attn.dt_bias", .{pre}), .bf16, &.{32}), .raw);
                g.norm = try l.floatTensor(try l.tensor(try checkpoint.name(&buf, "{s}.linear_attn.norm.weight", .{pre}), .bf16, &.{128}), .raw);
                out.mixer = .{ .delta = g };
            }
            return out;
        }
    };
}

test "ModelOpt arena plans account for every aligned allocation and ownership on official headers" {
    const Header = @import("cuda_checkpoint_test.zig").Header;
    inline for (.{ "ornith_nvfp4", "nvidia_qwen36_nvfp4" }) |fixture| {
        var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
        defer arena.deinit();
        const a = arena.allocator();
        const cfg = try checkpoint.Config.parse(a, @embedFile("fixtures/" ++ fixture ++ "/config.json"));
        var h = try Header.parse(a, @embedFile("fixtures/" ++ fixture ++ "/headers.json.gz"));
        const all = try plan(&h, cfg, .{ .first = 0, .end = 40, .total = 40 });
        const first = try plan(&h, cfg, .{ .first = 0, .end = 20, .total = 40 });
        const last = try plan(&h, cfg, .{ .first = 20, .end = 40, .total = 40 });
        try std.testing.expectEqual(all.weights_bytes, first.weights_bytes + last.weights_bytes);
        try std.testing.expectEqual(@as(usize, 10_790_999_040), first.weights_bytes);
        try std.testing.expectEqual(@as(usize, 10_059_953_152), last.weights_bytes);
        try std.testing.expectEqual(@as(usize, 16_777_216), first.staging_bytes);
        try std.testing.expectEqual(@as(usize, 286_064_640), last.staging_bytes);
        const uneven_first = try plan(&h, cfg, .{ .first = 0, .end = 3, .total = 40 });
        const uneven_last = try plan(&h, cfg, .{ .first = 3, .end = 40, .total = 40 });
        try std.testing.expectEqual(all.weights_bytes, uneven_first.weights_bytes + uneven_last.weights_bytes);
        var walker: Loader(true, *Header, void) = .{ .source = &h };
        const layer = try walker.layer(3, true);
        try std.testing.expectEqual(@as(usize, 256), layer.mlp.moe.experts.experts);
        try std.testing.expect(layer.mlp.moe.shared == .projected);
        try std.testing.expect(layer.mixer == .attention);
        try std.testing.expectEqual(@as(usize, 8192), layer.mixer.attention.q.outputs());
        try std.testing.expectEqual(@as(usize, 2048), layer.mixer.attention.q.inputs());
        try std.testing.expectEqual(@as(usize, 512), layer.mlp.moe.shared.projected.gate.outputs());
        try std.testing.expectEqual(@as(usize, 2048), layer.mlp.moe.shared.projected.gate.inputs());
    }
}

test "recording arena loader matches planned allocation bytes for full and partial official checkpoints" {
    const Header = @import("cuda_checkpoint_test.zig").Header;
    const Recording = struct {
        const Allocation = struct { offset: usize, bytes: usize };
        gpa: std.mem.Allocator,
        arena: struct {
            ptr: u64 = 4096,
            pub fn upload(_: @This(), _: usize, _: []const u8) !void {}
        } = .{},
        stream: struct {
            pub fn synchronize(_: @This()) !void {}
        } = .{},
        modelopt: ?struct { pack: struct {
            var expected_up_layout: cuda.experts.Nvfp4UpLayout = .lane_major;
            var up_calls: usize = 0;
            var down_calls: usize = 0;
            pub fn reset(layout: cuda.experts.Nvfp4UpLayout) void {
                expected_up_layout = layout;
                up_calls = 0;
                down_calls = 0;
            }
            pub fn check(layer_count: usize) !void {
                try std.testing.expectEqual(layer_count * 512, up_calls);
                try std.testing.expectEqual(layer_count * 256, down_calls);
            }
            pub fn bf16Lane(_: @This(), _: anytype, _: u64, _: u64, _: usize, _: usize) !void {}
            pub fn fp8Lane(_: @This(), _: anytype, _: u64, _: u64, _: usize, _: usize) !void {}
            pub fn fp4Lane(_: @This(), _: anytype, _: u64, _: u64, _: u64, _: u64, _: usize, _: usize) !void {}
            pub fn fp4ExpertsLayout(_: @This(), _: anytype, _: u64, _: u64, _: u64, _: usize, _: usize, _: usize, _: usize, _: usize, parts: usize, _: usize, layout: cuda.experts.Nvfp4UpLayout) !void {
                if (parts == 1) {
                    down_calls += 1;
                    try std.testing.expectEqual(cuda.experts.Nvfp4UpLayout.lane_major, layout);
                } else {
                    up_calls += 1;
                    try std.testing.expectEqual(@as(usize, 2), parts);
                    try std.testing.expectEqual(expected_up_layout, layout);
                }
            }
        } = .{} } = .{},
        allocations: std.ArrayList(Allocation) = .empty,
        arena_at: usize = 0,
        layers: []Layer,
        embed_w: u64 = 0,
        embed_s: u64 = 0,
        embed_b: u64 = 0,
        embed_bits: u8 = 0,
        terminal_weights: layers.TerminalWeights = .none,
        final_norm: u64 = 0,

        pub fn alloc(r: *@This(), bytes: usize) !u64 {
            const start = std.mem.alignForward(usize, r.arena_at, 256);
            try r.allocations.append(r.gpa, .{ .offset = start, .bytes = bytes });
            r.arena_at = start + bytes;
            return r.arena.ptr + start;
        }
        fn copy(_: u64, _: ?*const anyopaque, _: usize) callconv(.c) cuda.abi.Result {
            return cuda.abi.success;
        }
    };
    const Values = struct {
        header: *Header,
        zeros: [65536]u8 = @splat(0),
        one: f32 = 1,
        pub fn has(v: *@This(), name: []const u8) bool {
            return v.header.has(name);
        }
        pub fn get(v: *@This(), name: []const u8) !Tensor {
            var t = try v.header.get(name);
            if (t.dtype == .f32 and (t.rank == 0 or (t.rank == 1 and t.dim(0) == 1))) {
                t.bytes = std.mem.asBytes(&v.one);
            } else if (t.rank == 1 or t.rank == 3) {
                var size: usize = t.dtype.size();
                for (t.shape[0..t.rank]) |dim| size *= dim;
                t.bytes = v.zeros[0..size];
            }
            return t;
        }
    };
    inline for (.{ "ornith_nvfp4", "nvidia_qwen36_nvfp4" }) |fixture| {
        var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
        defer arena.deinit();
        const a = arena.allocator();
        const cfg = try checkpoint.Config.parse(a, @embedFile("fixtures/" ++ fixture ++ "/config.json"));
        var header = try Header.parse(a, @embedFile("fixtures/" ++ fixture ++ "/headers.json.gz"));
        var values: Values = .{ .header = &header };
        var driver: cuda.Driver = .{ .api = undefined, .lib = undefined };
        driver.api.cuMemcpyHtoD_v2 = Recording.copy;
        var stage: cuda.DeviceBuffer = .{ .d = &driver, .ptr = 1, .len = 1 << 30, .borrowed = true };
        for ([_]checkpoint.Placement{
            .{ .first = 0, .end = 40, .total = 40 },
            .{ .first = 0, .end = 3, .total = 40 },
            .{ .first = 3, .end = 40, .total = 40 },
        }) |placement| for ([_]cuda.experts.Nvfp4UpLayout{ .lane_major, .slice_major_v1 }) |layout| {
            const expected = try plan(&header, cfg, placement);
            var recorded: Recording = .{ .gpa = a, .layers = try a.alloc(Layer, placement.end - placement.first) };
            @TypeOf(recorded.modelopt.?.pack).reset(layout);
            var loader: Loader(false, *Values, *Recording) = .{ .source = &values, .e = &recorded, .stage = &stage, .up_layout = layout };
            try loader.walk(cfg, placement);
            try @TypeOf(recorded.modelopt.?.pack).check(placement.end - placement.first);
            try std.testing.expectEqual(expected, loader.sizes);
            try std.testing.expectEqual(expected.weights_bytes, std.mem.alignForward(usize, recorded.arena_at, 256));
            var cursor: usize = 0;
            for (recorded.allocations.items) |allocation| {
                try std.testing.expectEqual(cursor, allocation.offset);
                cursor += std.mem.alignForward(usize, allocation.bytes, 256);
            }
            try std.testing.expectEqual(expected.weights_bytes, cursor);
            try std.testing.expectEqual(placement.ownsEmbedding(), recorded.embed_w != 0);
            try std.testing.expectEqual(placement.ownsHead(), recorded.final_norm != 0);
            for (recorded.layers, placement.first..) |layer, global| {
                try std.testing.expectEqual(cfg.spec.full(global), layer.mixer == .attention);
                try std.testing.expect(layer.mlp.moe.shared == .projected);
                try std.testing.expectEqual(layout, layer.mlp.moe.experts.nvfp4_up_layout);
            }
        };
    }
}
