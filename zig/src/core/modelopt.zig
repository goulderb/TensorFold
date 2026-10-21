//! ModelOpt NVFP4 (groups of 16) and MIXED_PRECISION per-module recipes (NVFP4, W4A16_NVFP4, MXFP8, FP8_PB_WO).

const std = @import("std");

/// The NVFP4 group size ModelOpt writes for this format.
pub const nvfp4_group = 16;

/// A MIXED_PRECISION module's recipe, as this engine reads it (the activations stay bf16 whatever the recipe names).
pub const Recipe = enum {
    /// NVFP4: FP4 weights, an e4m3 scale every 16 inputs, a global scale.
    nvfp4,
    /// NVFP4 whose block scales are stored as csf_32x2_v1 (quant/csf.zig decodes them).
    nvfp4_csf,
    /// W4A16_NVFP4: NVFP4 weights without an input scale.
    w4a16_nvfp4,
    /// MXFP8: e4m3 weights, a UE8M0 scale every 32 inputs.
    mxfp8,
    /// FP8_PB_WO: e4m3 weights, an f32 scale every 128 x 128 block.
    fp8_pb_wo,

    pub fn algo(r: Recipe) Algo {
        return switch (r) {
            .nvfp4, .nvfp4_csf => .nvfp4,
            .w4a16_nvfp4 => .w4a16_nvfp4,
            .mxfp8 => .mxfp8,
            .fp8_pb_wo => .fp8_pb_wo,
        };
    }

    pub fn scaleEncoding(r: Recipe) ScaleEncoding {
        return if (r == .nvfp4_csf) .csf_32x2_v1 else .none;
    }
};

/// The modules a MIXED_PRECISION checkpoint quantizes and their recipes; none for a single-format NVFP4 one.
pub const Recipes = struct {
    arena: std.heap.ArenaAllocator,
    modules: std.StringArrayHashMapUnmanaged(Recipe) = .empty,

    /// The recipe of `name`, a module or one of its tensors: the longest declared module it is or lies under.
    pub fn of(self: *const Recipes, name: []const u8) ?Recipe {
        var n = name;
        while (true) {
            if (self.modules.get(n)) |r| return r;
            n = n[0 .. std.mem.lastIndexOfScalar(u8, n, '.') orelse return null];
        }
    }

    pub fn exact(self: *const Recipes, module_path: []const u8) ?Recipe {
        return self.modules.get(module_path);
    }

    pub fn deinit(self: *Recipes) void {
        self.arena.deinit();
    }
};

/// Why the check refused the checkpoint, kept for the caller's log line (tests read it instead).
pub const Why = struct {
    buf: [320]u8 = undefined,
    len: usize = 0,

    /// Keep the reason, cut short if it is long.
    pub fn set(self: *Why, comptime fmt: []const u8, args: anytype) void {
        var w: std.Io.Writer = .fixed(&self.buf);
        w.print(fmt, args) catch {};
        self.len = w.end;
    }

    pub fn text(self: *const Why) []const u8 {
        return self.buf[0..self.len];
    }
};

pub const Algo = enum { nvfp4, w4a16_nvfp4, mxfp8, fp8_pb_wo };

pub const ScaleEncoding = enum { none, csf_32x2_v1 };

/// config.json's quantization_config (top level, else text_config's) checked as ModelOpt NVFP4 or MIXED_PRECISION.
pub fn admit(gpa: std.mem.Allocator, config_json: []const u8, why: *Why) !void {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const q = try quantBlock(arena.allocator(), config_json, why);
    const algo = text(q.get("quant_algo")) orelse "";
    if (std.ascii.eqlIgnoreCase(algo, "NVFP4")) return admitNvfp4(q, why);
    if (std.ascii.eqlIgnoreCase(algo, "MIXED_PRECISION")) {
        var t = try admitRecipes(gpa, config_json, why);
        t.deinit();
        return;
    }
    const named = if (algo.len > 0) algo else "without quant_algo";
    why.set("ModelOpt {s}: this engine reads ModelOpt NVFP4 (nvidia/GLM-5.3-Flash-NVFP4's layout)", .{named});
    return error.UnsupportedQuant;
}

/// MIXED_PRECISION `quantized_layers` as a recipe table.
pub fn admitRecipes(gpa: std.mem.Allocator, config_json: []const u8, why: *Why) !Recipes {
    var t: Recipes = .{ .arena = .init(gpa) };
    errdefer t.deinit();
    const q = try quantBlock(t.arena.allocator(), config_json, why);
    const algo = text(q.get("quant_algo")) orelse "";
    if (!std.ascii.eqlIgnoreCase(algo, "MIXED_PRECISION")) {
        if (!std.ascii.eqlIgnoreCase(algo, "NVFP4")) {
            why.set("ModelOpt {s}: expected NVFP4 or MIXED_PRECISION", .{algo});
            return error.UnsupportedQuant;
        }
        try admitNvfp4(q, why);
        return t;
    }
    try fill(&t, q, why);
    return t;
}

fn quantBlock(a: std.mem.Allocator, config_json: []const u8, why: *Why) !std.json.ObjectMap {
    const root = try std.json.parseFromSliceLeaky(std.json.Value, a, config_json, .{});
    return block(root) orelse {
        why.set("config.json has no ModelOpt quantization_config", .{});
        return error.NotModelOpt;
    };
}

fn admitNvfp4(q: std.json.ObjectMap, why: *Why) !void {
    const groups = object(q.get("config_groups")) orelse {
        why.set("ModelOpt NVFP4 without config_groups", .{});
        return error.UnsupportedQuant;
    };
    var it = groups.iterator();
    while (it.next()) |g| {
        const w = if (object(g.value_ptr.*)) |o| object(o.get("weights")) else null;
        const bits = if (w) |o| int(o.get("num_bits")) else null;
        const size = if (w) |o| int(o.get("group_size")) else null;
        const kind = if (w) |o| text(o.get("type")) orelse "" else "";
        if (bits != 4 or size != nvfp4_group or !std.mem.eql(u8, kind, "float")) {
            why.set(
                "ModelOpt NVFP4 group {s}: weights {?d}-bit {s} in groups of {?d}, " ++
                    "expected 4-bit float in groups of 16",
                .{ g.key_ptr.*, bits, kind, size },
            );
            return error.UnsupportedQuant;
        }
    }
}

fn fill(t: *Recipes, q: std.json.ObjectMap, why: *Why) !void {
    try excludeOk(q, why);
    const layers = object(q.get("quantized_layers")) orelse {
        why.set("ModelOpt MIXED_PRECISION without quantized_layers", .{});
        return error.UnsupportedQuant;
    };
    const a = t.arena.allocator();
    var it = layers.iterator();
    while (it.next()) |kv| {
        const o = object(kv.value_ptr.*) orelse {
            why.set("ModelOpt MIXED_PRECISION {s}: recipe is not an object", .{kv.key_ptr.*});
            return error.UnsupportedQuant;
        };
        const rec = try parseRecipe(kv.key_ptr.*, o, why);
        try t.modules.put(a, try a.dupe(u8, kv.key_ptr.*), rec);
    }
}

fn excludeOk(q: std.json.ObjectMap, why: *Why) !void {
    const v = q.get("exclude_modules") orelse return;
    if (v != .array) {
        why.set("ModelOpt MIXED_PRECISION exclude_modules: expected a string list", .{});
        return error.UnsupportedQuant;
    }
    if (v.array.items.len == 0) return;
    const first = if (v.array.items[0] == .string) v.array.items[0].string else "non-string";
    why.set("ModelOpt MIXED_PRECISION exclude_modules {s}: this engine has no exclude list", .{first});
    return error.UnsupportedQuant;
}

const Block = union(enum) { missing, pair: [2]i64, bad };

fn parseRecipe(path: []const u8, o: std.json.ObjectMap, why: *Why) !Recipe {
    const algo = text(o.get("quant_algo")) orelse {
        why.set("ModelOpt MIXED_PRECISION {s}: without quant_algo", .{path});
        return error.UnsupportedQuant;
    };
    const enc = text(o.get("weight_scale_encoding"));
    const groups = int(o.get("group_size"));
    const b = block2(o.get("weight_block_size"));
    if (std.ascii.eqlIgnoreCase(algo, "NVFP4")) return parseNvfp4(path, enc, groups, b, why);
    if (std.ascii.eqlIgnoreCase(algo, "W4A16_NVFP4")) return parseW4a16(path, enc, groups, b, why);
    if (std.ascii.eqlIgnoreCase(algo, "MXFP8")) return parseMxfp8(path, enc, groups, b, why);
    if (std.ascii.eqlIgnoreCase(algo, "FP8_PB_WO")) return parsePbWo(path, enc, groups, b, why);
    why.set("ModelOpt MIXED_PRECISION {s}: unknown quant_algo {s}", .{ path, algo });
    return error.UnsupportedQuant;
}

fn parseNvfp4(path: []const u8, enc: ?[]const u8, groups: ?i64, b: Block, why: *Why) !Recipe {
    if (b != .missing) return unexpectedBlock(path, "NVFP4", b, why);
    if (groups != nvfp4_group) {
        why.set("ModelOpt MIXED_PRECISION {s}: NVFP4 group_size {?d}, expected 16", .{ path, groups });
        return error.UnsupportedQuant;
    }
    if (enc) |e| {
        if (!std.mem.eql(u8, e, "csf_32x2_v1")) {
            why.set("ModelOpt MIXED_PRECISION {s}: unknown weight_scale_encoding {s}", .{ path, e });
            return error.UnsupportedQuant;
        }
        return Recipe.nvfp4_csf;
    }
    return Recipe.nvfp4;
}

fn parseW4a16(path: []const u8, enc: ?[]const u8, groups: ?i64, b: Block, why: *Why) !Recipe {
    if (enc) |e| {
        why.set("ModelOpt MIXED_PRECISION {s}: unexpected weight_scale_encoding {s}", .{ path, e });
        return error.UnsupportedQuant;
    }
    if (b != .missing) return unexpectedBlock(path, "W4A16_NVFP4", b, why);
    if (groups != nvfp4_group) {
        why.set("ModelOpt MIXED_PRECISION {s}: W4A16_NVFP4 group_size {?d}, expected 16", .{ path, groups });
        return error.UnsupportedQuant;
    }
    return Recipe.w4a16_nvfp4;
}

fn parseMxfp8(path: []const u8, enc: ?[]const u8, groups: ?i64, b: Block, why: *Why) !Recipe {
    if (enc) |e| {
        why.set("ModelOpt MIXED_PRECISION {s}: unexpected weight_scale_encoding {s}", .{ path, e });
        return error.UnsupportedQuant;
    }
    switch (b) {
        .pair => |p| {
            if (p[0] != 1 or p[1] != 32) {
                why.set("ModelOpt MIXED_PRECISION {s}: MXFP8 weight_block_size [{d},{d}], expected [1,32]", .{ path, p[0], p[1] });
                return error.UnsupportedQuant;
            }
            if (groups) |g| if (g != 32) {
                why.set("ModelOpt MIXED_PRECISION {s}: MXFP8 group_size {d}, expected 32", .{ path, g });
                return error.UnsupportedQuant;
            };
        },
        .missing => {
            if (groups != 32) {
                why.set("ModelOpt MIXED_PRECISION {s}: MXFP8 needs weight_block_size [1,32] or group_size 32", .{path});
                return error.UnsupportedQuant;
            }
        },
        .bad => {
            why.set("ModelOpt MIXED_PRECISION {s}: MXFP8 weight_block_size is not two integers", .{path});
            return error.UnsupportedQuant;
        },
    }
    return Recipe.mxfp8;
}

fn parsePbWo(path: []const u8, enc: ?[]const u8, groups: ?i64, b: Block, why: *Why) !Recipe {
    if (enc) |e| {
        why.set("ModelOpt MIXED_PRECISION {s}: unexpected weight_scale_encoding {s}", .{ path, e });
        return error.UnsupportedQuant;
    }
    if (groups) |g| {
        why.set("ModelOpt MIXED_PRECISION {s}: FP8_PB_WO group_size {d}, expected weight_block_size [128,128]", .{ path, g });
        return error.UnsupportedQuant;
    }
    switch (b) {
        .pair => |p| if (p[0] != 128 or p[1] != 128) {
            why.set("ModelOpt MIXED_PRECISION {s}: FP8_PB_WO weight_block_size [{d},{d}], expected [128,128]", .{ path, p[0], p[1] });
            return error.UnsupportedQuant;
        },
        .missing => {
            why.set("ModelOpt MIXED_PRECISION {s}: FP8_PB_WO without weight_block_size [128,128]", .{path});
            return error.UnsupportedQuant;
        },
        .bad => {
            why.set("ModelOpt MIXED_PRECISION {s}: FP8_PB_WO weight_block_size is not two integers", .{path});
            return error.UnsupportedQuant;
        },
    }
    return Recipe.fp8_pb_wo;
}

fn unexpectedBlock(path: []const u8, algo: []const u8, b: Block, why: *Why) error{UnsupportedQuant} {
    switch (b) {
        .pair => |p| why.set("ModelOpt MIXED_PRECISION {s}: {s} weight_block_size [{d},{d}], unexpected", .{ path, algo, p[0], p[1] }),
        .bad => why.set("ModelOpt MIXED_PRECISION {s}: {s} weight_block_size is not two integers", .{ path, algo }),
        .missing => unreachable,
    }
    return error.UnsupportedQuant;
}

fn block2(v: ?std.json.Value) Block {
    const x = v orelse return .missing;
    if (x != .array or x.array.items.len != 2) return .bad;
    const a = int(x.array.items[0]) orelse return .bad;
    const b = int(x.array.items[1]) orelse return .bad;
    return .{ .pair = .{ a, b } };
}

fn block(root: std.json.Value) ?std.json.ObjectMap {
    const top = object(root) orelse return null;
    for ([_]?std.json.ObjectMap{ top, object(top.get("text_config")) }) |source| {
        const q = object((source orelse continue).get("quantization_config")) orelse continue;
        if (std.ascii.eqlIgnoreCase(text(q.get("quant_method")) orelse "", "modelopt")) return q;
    }
    return null;
}

fn object(v: ?std.json.Value) ?std.json.ObjectMap {
    const x = v orelse return null;
    return if (x == .object) x.object else null;
}

fn text(v: ?std.json.Value) ?[]const u8 {
    const x = v orelse return null;
    return if (x == .string) x.string else null;
}

fn int(v: ?std.json.Value) ?i64 {
    const x = v orelse return null;
    return if (x == .integer) x.integer else null;
}

test "nvidia/GLM-5.3-Flash-NVFP4's config is admitted" {
    var why: Why = .{};
    try admit(std.testing.allocator, @embedFile("fixtures/glm53_flash_nvfp4/config.json"), &why);
}

test "nvidia/GLM-5.3-Flash-FP4-Local's MIXED_PRECISION config is admitted" {
    var why: Why = .{};
    const json = @embedFile("fixtures/glm53_flash_fp4_local/config.json");
    try admit(std.testing.allocator, json, &why);
    var t = try admitRecipes(std.testing.allocator, json, &why);
    defer t.deinit();
    var n_nvfp4: usize = 0;
    var n_w4: usize = 0;
    var n_mx: usize = 0;
    var n_pb: usize = 0;
    for (t.modules.values()) |r| switch (r.algo()) {
        .nvfp4 => {
            try std.testing.expectEqual(Recipe.nvfp4_csf, r);
            n_nvfp4 += 1;
        },
        .w4a16_nvfp4 => n_w4 += 1,
        .mxfp8 => n_mx += 1,
        .fp8_pb_wo => n_pb += 1,
    };
    try std.testing.expectEqual(@as(usize, 42), n_nvfp4);
    try std.testing.expectEqual(@as(usize, 77), n_w4);
    try std.testing.expectEqual(@as(usize, 205), n_mx);
    try std.testing.expectEqual(@as(usize, 170), n_pb);
    try std.testing.expectEqual(@as(usize, 494), t.modules.count());
    try std.testing.expectEqual(Recipe.nvfp4_csf, t.exact("model.language_model.layers.3.mlp.experts").?);
    try std.testing.expectEqual(Recipe.w4a16_nvfp4, t.exact("model.language_model.layers.45.mlp.experts").?);
    try std.testing.expectEqual(Recipe.mxfp8, t.exact("lm_head").?);
    try std.testing.expectEqual(Recipe.mxfp8, t.exact("model.language_model.layers.45.self_attn.indexer.wk").?);
    try std.testing.expectEqual(Recipe.fp8_pb_wo, t.exact("model.language_model.layers.10.mlp.shared_experts.down_proj").?);
    try std.testing.expectEqual(Recipe.w4a16_nvfp4, t.exact("model.visual.blocks.0.mlp.down_proj").?);
    try std.testing.expect(t.exact("model.language_model.layers.0.mlp.experts") == null);
}

/// admit() must refuse `json` with `err`, its reason naming `word`.
fn expectRefused(json: []const u8, err: anyerror, word: []const u8) !void {
    var why: Why = .{};
    try std.testing.expectError(err, admit(std.testing.allocator, json, &why));
    if (std.mem.indexOf(u8, why.text(), word) == null) {
        std.debug.print("refusal \"{s}\" does not name \"{s}\"\n", .{ why.text(), word });
        return error.TestUnexpectedResult;
    }
}

test "other ModelOpt formats, and configs without ModelOpt, are refused by name" {
    try expectRefused("{\"text_config\": {}}", error.NotModelOpt, "no ModelOpt");
    try expectRefused("{\"quantization_config\": {\"quant_method\": \"exl3\"}}", error.NotModelOpt, "no ModelOpt");
    try expectRefused(
        "{\"quantization_config\": {\"quant_method\": \"modelopt\", \"quant_algo\": \"FP8\"}}",
        error.UnsupportedQuant,
        "ModelOpt FP8",
    );
    try expectRefused(
        "{\"quantization_config\": {\"quant_method\": \"modelopt\", \"quant_algo\": \"MIXED_PRECISION\"}}",
        error.UnsupportedQuant,
        "quantized_layers",
    );
    try expectRefused(
        \\{"quantization_config": {"quant_method": "modelopt", "quant_algo": "MIXED_PRECISION",
        \\ "quantized_layers": {"lm_head": {"quant_algo": "INT8"}}}}
    , error.UnsupportedQuant, "INT8");
    try expectRefused(
        \\{"quantization_config": {"quant_method": "modelopt", "quant_algo": "MIXED_PRECISION",
        \\ "exclude_modules": ["lm_head"], "quantized_layers": {}}}
    , error.UnsupportedQuant, "lm_head");
    try expectRefused(
        \\{"quantization_config": {"quant_method": "modelopt", "quant_algo": "MIXED_PRECISION",
        \\ "quantized_layers": {"x": {"quant_algo": "MXFP8", "weight_block_size": [1, 64]}}}}
    , error.UnsupportedQuant, "[1,64]");
    try expectRefused(
        \\{"quantization_config": {"quant_method": "modelopt", "quant_algo": "MIXED_PRECISION",
        \\ "quantized_layers": {"x": {"quant_algo": "NVFP4", "group_size": 16, "weight_scale_encoding": "foo"}}}}
    , error.UnsupportedQuant, "foo");
    try expectRefused(
        \\{"quantization_config": {"quant_method": "modelopt", "quant_algo": "NVFP4", "config_groups":
        \\ {"g": {"weights": {"num_bits": 4, "type": "float", "group_size": 32}}}}}
    , error.UnsupportedQuant, "groups of 32");
    var why: Why = .{};
    try admit(std.testing.allocator,
        \\{"text_config": {"quantization_config": {"quant_method": "modelopt", "quant_algo": "nvfp4", "config_groups":
        \\ {"g": {"weights": {"num_bits": 4, "type": "float", "group_size": 16}}}}}}
    , &why);
    try admit(std.testing.allocator,
        \\{"quantization_config": {"quant_method": "modelopt", "quant_algo": "MIXED_PRECISION", "quantized_layers":
        \\ {"lm_head": {"quant_algo": "MXFP8", "group_size": 32}}}}
    , &why);
}

test "nvidia/GLM-5.3-Flash-FP4-Local's recipes: a module's, and its tensors' under it" {
    var why: Why = .{};
    var r = try admitRecipes(std.testing.allocator, @embedFile("fixtures/glm53_flash_fp4_local/config.json"), &why);
    defer r.deinit();
    var count = std.EnumArray(Recipe, usize).initFill(0);
    for (r.modules.values()) |v| count.getPtr(v).* += 1;
    try std.testing.expectEqualSlices(usize, &.{ 0, 42, 77, 205, 170 }, &count.values); // the vision tower's MLPs too
    const at = "model.language_model.layers.";
    const cases = [_]struct { []const u8, ?Recipe }{
        .{ at ++ "3.mlp.experts.17.gate_proj.weight", .nvfp4_csf }, // routed experts: one entry a layer
        .{ at ++ "45.mlp.experts.0.down_proj.weight_scale", .w4a16_nvfp4 }, // MTP's
        .{ at ++ "0.self_attn.q_proj.weight", .mxfp8 }, // KDA, weight_block_size [1, 32]
        .{ at ++ "45.self_attn.kv_b_proj.weight", .mxfp8 }, // MTP's, group_size 32
        .{ "lm_head.weight", .mxfp8 },
        .{ at ++ "3.self_attn.q_a_proj.weight_scale", .fp8_pb_wo }, // MLA
        .{ at ++ "43.mlp.shared_experts.down_proj.weight", .fp8_pb_wo },
        .{ at ++ "3.self_attn.kv_b_proj.weight", null }, // bf16
        .{ at ++ "0.self_attn.f_a_proj.weight", null },
        .{ at ++ "3.mlp.gate.weight", null }, // the router
    };
    for (cases) |c| try std.testing.expectEqual(c[1], r.of(c[0]));
}

test "a recipe covers its module and what lies under it, at a dot" {
    var r: Recipes = .{ .arena = .init(std.testing.allocator) };
    defer r.deinit();
    try r.modules.put(r.arena.allocator(), "m.experts", .nvfp4);
    try std.testing.expectEqual(@as(?Recipe, .nvfp4), r.of("m.experts"));
    try std.testing.expectEqual(@as(?Recipe, .nvfp4), r.of("m.experts.1.up_proj.weight"));
    try std.testing.expectEqual(@as(?Recipe, null), r.of("m.experts_bias"));
    try std.testing.expectEqual(@as(?Recipe, null), r.of("m"));
    try std.testing.expectEqual(@as(?Recipe, .nvfp4), r.exact("m.experts"));
    try std.testing.expectEqual(@as(?Recipe, null), r.exact("m.experts.1.up_proj"));
    try r.modules.put(r.arena.allocator(), "m.experts.1.up_proj", .mxfp8);
    try std.testing.expectEqual(@as(?Recipe, .mxfp8), r.of("m.experts.1.up_proj.weight"));
    try std.testing.expectEqual(@as(?Recipe, .nvfp4), r.of("m.experts.2.up_proj.weight"));
}

test "a single-format NVFP4 config declares no recipes" {
    var why: Why = .{};
    var r = try admitRecipes(std.testing.allocator, @embedFile("fixtures/glm53_flash_nvfp4/config.json"), &why);
    defer r.deinit();
    try std.testing.expectEqual(@as(u32, 0), r.modules.count());
}

/// admitRecipes() must refuse a MIXED_PRECISION config of these `layers` as unsupported, its reason naming `word`.
fn expectRecipesRefused(comptime layers: []const u8, word: []const u8) !void {
    const json = "{\"quantization_config\": {\"quant_method\": \"modelopt\", \"quant_algo\": \"MIXED_PRECISION\"" ++
        layers ++ "}}";
    var why: Why = .{};
    try std.testing.expectError(error.UnsupportedQuant, admitRecipes(std.testing.allocator, json, &why));
    if (std.mem.indexOf(u8, why.text(), word) == null) {
        std.debug.print("refusal \"{s}\" does not name \"{s}\"\n", .{ why.text(), word });
        return error.TestUnexpectedResult;
    }
}

test "recipes this engine does not read are refused by module" {
    try expectRecipesRefused("", "without quantized_layers");
    try expectRecipesRefused(
        \\, "quantized_layers": {"m.q": {"quant_algo": "INT4_AWQ", "group_size": 128}}
    , "m.q: unknown quant_algo INT4_AWQ");
    try expectRecipesRefused( // the slab encoding (csf) is not read
        \\, "quantized_layers": {"m.experts": {"quant_algo": "NVFP4", "group_size": 16, "weight_scale_encoding": "csf"}}
    , "m.experts: unknown weight_scale_encoding csf");
    try expectRecipesRefused(
        \\, "quantized_layers": {"m.e": {"quant_algo": "W4A16_NVFP4", "group_size": 16, "weight_scale_encoding": "csf_32x2_v1"}}
    , "m.e: unexpected weight_scale_encoding csf_32x2_v1");
    try expectRecipesRefused(
        \\, "quantized_layers": {"m.o": {"quant_algo": "MXFP8", "group_size": 16}}
    , "m.o: MXFP8");
    try expectRecipesRefused(
        \\, "quantized_layers": {"m.o": {"quant_algo": "MXFP8", "group_size": 32, "weight_block_size": [1, 64]}}
    , "m.o: MXFP8");
    try expectRecipesRefused(
        \\, "quantized_layers": {"m.s": {"quant_algo": "FP8_PB_WO", "weight_block_size": [64, 64]}}
    , "m.s: FP8_PB_WO");
    try expectRecipesRefused(
        \\, "quantized_layers": {"m.s": 7}
    , "m.s: recipe is not an object");
    var why: Why = .{};
    try std.testing.expectError(error.UnsupportedQuant, admitRecipes(std.testing.allocator,
        \\{"quantization_config": {"quant_method": "modelopt", "quant_algo": "FP8"}}
    , &why));
    var r = try admitRecipes(std.testing.allocator,
        \\{"quantization_config": {"quant_method": "modelopt", "quant_algo": "MIXED_PRECISION", "quantized_layers":
        \\ {"m.experts": {"quant_algo": "NVFP4", "group_size": 16}}}}
    , &why);
    defer r.deinit();
    try std.testing.expectEqual(@as(?Recipe, .nvfp4), r.of("m.experts.0.gate_proj.weight"));
}
