const std = @import("std");
const core = @import("core");
const cuda = @import("cuda");
const source = @import("cuda_weight_source.zig");
const dense = @import("cuda_weight_source_dense.zig");
const admission = @import("cuda_checkpoint.zig");
const a = std.testing.allocator;
const io = std.testing.io;
const X = cuda.experts.Nvfp4Experts;

pub const Item = struct { name: []const u8, dtype: []const u8, shape: []const usize, bytes: []const u8 };

pub fn write(dir: std.Io.Dir, items: []const Item) !void {
    var h: std.Io.Writer.Allocating = .init(a);
    defer h.deinit();
    const w = &h.writer;
    try w.writeAll("{");
    var offset: usize = 0;
    for (items, 0..) |item, i| {
        if (i != 0) try w.writeAll(",");
        try w.print("\"{s}\":{{\"dtype\":\"{s}\",\"shape\":[", .{ item.name, item.dtype });
        for (item.shape, 0..) |dim, j| {
            if (j != 0) try w.writeAll(",");
            try w.print("{d}", .{dim});
        }
        try w.print("],\"data_offsets\":[{d},{d}]}}", .{ offset, offset + item.bytes.len });
        offset += item.bytes.len;
    }
    try w.writeAll("}");
    var image: std.ArrayList(u8) = .empty;
    defer image.deinit(a);
    var size: [8]u8 = undefined;
    std.mem.writeInt(u64, &size, h.written().len, .little);
    try image.appendSlice(a, &size);
    try image.appendSlice(a, h.written());
    for (items) |item| try image.appendSlice(a, item.bytes);
    try dir.writeFile(io, .{ .sub_path = "model.safetensors", .data = image.items });
}

fn table() !core.modelopt.Recipes {
    var why: core.modelopt.Why = .{};
    return core.modelopt.admitRecipes(a,
        \\{"quantization_config":{"quant_method":"modelopt","quant_algo":"MIXED_PRECISION","quantized_layers":{
        \\"kda":{"quant_algo":"MXFP8","weight_block_size":[1,32]},
        \\"mla":{"quant_algo":"FP8_PB_WO","weight_block_size":[128,128]},
        \\"model.language_model.layers.3.mlp.experts":{"quant_algo":"NVFP4","group_size":16,"weight_scale_encoding":"csf_32x2_v1"},
        \\"mtp":{"quant_algo":"W4A16_NVFP4","group_size":16}}}}
    , &why);
}

test "mixed loader admits the real fixture as a distinct format with owned recipes" {
    var why: core.modelopt.Why = .{};
    var admitted = try admission.Admission.read(a, @embedFile("fixtures/fp4-local-config.json"), &why);
    defer admitted.deinit();
    try std.testing.expectEqual(admission.Format.modelopt_mixed, admitted.format);
    try std.testing.expectEqual(@as(usize, 494), admitted.recipes.?.modules.count());
    try std.testing.expectEqual(core.modelopt.Recipe.nvfp4_csf, source.recipe(&admitted.recipes.?, "model.language_model.layers.3.mlp.experts.287.gate_proj").?);
}

test "mixed loader gathers MXFP8 and block FP8 spans and interleaved MLA head rows" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const mx: [4 * 64]u8 = @splat(0x38);
    const ms = [_]u8{ 127, 128, 126, 129, 127, 125, 130, 127 };
    const pb: [256 * 256]u8 = @splat(0xc0);
    const ps = [_]f32{ 0.5, 2, 4, 0.25 };
    const norm = [_]f32{ 1, -2 };
    try write(tmp.dir, &.{
        .{ .name = "kda.weight", .dtype = "F8_E4M3", .shape = &.{ 4, 64 }, .bytes = &mx },
        .{ .name = "kda.weight_scale", .dtype = "U8", .shape = &.{ 4, 2 }, .bytes = &ms },
        .{ .name = "mla.weight", .dtype = "F8_E4M3", .shape = &.{ 256, 256 }, .bytes = &pb },
        .{ .name = "mla.weight_scale", .dtype = "F32", .shape = &.{ 2, 1, 2, 1 }, .bytes = std.mem.sliceAsBytes(&ps) },
        .{ .name = "norm.weight", .dtype = "F32", .shape = &.{2}, .bytes = std.mem.sliceAsBytes(&norm) },
    });
    var path: [128]u8 = undefined;
    var ck = try core.Checkpoint.openModel(a, io, try std.fmt.bufPrint(&path, ".zig-cache/tmp/{s}", .{tmp.sub_path}));
    defer ck.close();
    var recipes = try table();
    defer recipes.deinit();
    const kda = try source.Source.read(&ck, &recipes, "kda.weight");
    var out: [8]u16 = undefined;
    try kda.bf16(.{ .lo = 1, .hi = 3 }, .{ .lo = 30, .hi = 34 }, &out);
    try std.testing.expectEqualSlices(u16, &.{ 0x3f00, 0x3f00, 0x4080, 0x4080, 0x3f80, 0x3f80, 0x3e80, 0x3e80 }, &out);
    const mla = try source.Source.read(&ck, &recipes, "mla.weight");
    try mla.bf16(.{ .lo = 127, .hi = 129 }, .{ .lo = 126, .hi = 130 }, &out);
    try std.testing.expectEqualSlices(u16, &.{ 0xbf80, 0xbf80, 0xc080, 0xc080, 0xc100, 0xc100, 0xbf00, 0xbf00 }, &out);
    var host: std.ArrayList(u8) = .empty;
    defer host.deinit(a);
    try kda.append(a, &host, 64, 128, .bf16);
    try kda.append(a, &host, 192, 256, .bf16);
    try std.testing.expectEqual(@as(usize, 256), host.items.len);
    const values = std.mem.bytesAsSlice(u16, host.items);
    try std.testing.expectEqual(@as(u16, 0x3f00), values[0]);
    try std.testing.expectEqual(@as(u16, 0x4080), values[32]);
    try std.testing.expectEqual(@as(u16, 0x4100), values[64]);
    const ns = try source.Source.read(&ck, &recipes, "norm.weight");
    host.clearRetainingCapacity();
    try ns.append(a, &host, 0, 2, .bf16);
    try std.testing.expectEqualSlices(u16, &.{ 0x3f80, 0xc000 }, @alignCast(std.mem.bytesAsSlice(u16, host.items)));
    try std.testing.expectEqual(@as(usize, 0), ck.unused());
}

test "mixed loader CSF reaches the expert packer identically to plain W4A16 scales" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const base = "model.language_model.layers.3.mlp.experts.0.gate_proj";
    const codes: [32 * 16]u8 = @splat(0x22);
    const bases: [32]u8 = @splat(0x38);
    const fields: [32]u8 = @splat(0);
    const pages = [_]u32{0};
    const offsets = [_]u16{ 0, 0 };
    const plain: [64]u8 = @splat(0x38);
    const global = [_]f32{0.25};
    try write(tmp.dir, &.{
        .{ .name = base ++ ".weight", .dtype = "U8", .shape = &.{ 32, 16 }, .bytes = &codes },
        .{ .name = base ++ ".weight_scale.csf32_bases", .dtype = "U8", .shape = &.{32}, .bytes = &bases },
        .{ .name = base ++ ".weight_scale.csf32_codes", .dtype = "U8", .shape = &.{32}, .bytes = &fields },
        .{ .name = base ++ ".weight_scale.csf32_page_bases", .dtype = "U32", .shape = &.{1}, .bytes = std.mem.sliceAsBytes(&pages) },
        .{ .name = base ++ ".weight_scale.csf32_offsets", .dtype = "U16", .shape = &.{2}, .bytes = std.mem.sliceAsBytes(&offsets) },
        .{ .name = base ++ ".weight_scale.csf32_exceptions", .dtype = "U16", .shape = &.{0}, .bytes = &.{} },
        .{ .name = base ++ ".weight_scale_2", .dtype = "F32", .shape = &.{1}, .bytes = std.mem.sliceAsBytes(&global) },
        .{ .name = base ++ ".input_scale", .dtype = "F32", .shape = &.{}, .bytes = std.mem.sliceAsBytes(&global) },
        .{ .name = "mtp.weight", .dtype = "U8", .shape = &.{ 32, 16 }, .bytes = &codes },
        .{ .name = "mtp.weight_scale", .dtype = "F8_E4M3", .shape = &.{ 32, 2 }, .bytes = &plain },
        .{ .name = "mtp.weight_scale_2", .dtype = "F32", .shape = &.{}, .bytes = std.mem.sliceAsBytes(&global) },
    });
    var path: [128]u8 = undefined;
    var ck = try core.Checkpoint.openModel(a, io, try std.fmt.bufPrint(&path, ".zig-cache/tmp/{s}", .{tmp.sub_path}));
    defer ck.close();
    var recipes = try table();
    defer recipes.deinit();
    var csf = try source.Fp4.read(a, &ck, &recipes, base);
    defer csf.deinit(a);
    var mtp = try source.Fp4.read(a, &ck, &recipes, "mtp");
    defer mtp.deinit(a);
    try std.testing.expectEqualSlices(u8, &plain, csf.linear.scales);
    var tiles: [144]u32 = undefined;
    var expected: [144]u32 = undefined;
    X.packOne(&tiles, csf.linear.codes, csf.linear.scales, 32, 32);
    X.packOne(&expected, mtp.linear.codes, mtp.linear.scales, 32, 32);
    try std.testing.expectEqualSlices(u32, &expected, &tiles);
    for (tiles[0..128]) |word| try std.testing.expectEqual(@as(u32, 0x22222222), word);
    for (tiles[128..]) |word| try std.testing.expectEqual(@as(u32, 0x38383838), word);
    var gu: [288]u32 = undefined;
    X.packGateUp(&gu, .{ csf.linear.codes, csf.linear.scales }, .{ mtp.linear.codes, mtp.linear.scales }, 32, 32);
    try std.testing.expectEqualSlices(u32, &tiles, gu[0..144]);
    try std.testing.expectEqualSlices(u32, &tiles, gu[144..]);
    var bf: [4]u16 = undefined;
    try mtp.bf16(.{ .lo = 1, .hi = 2 }, .{ .lo = 14, .hi = 18 }, &bf);
    try std.testing.expectEqualSlices(u16, &.{ 0x3e80, 0x3e80, 0x3e80, 0x3e80 }, &bf);
    try std.testing.expectEqual(@as(usize, 0), ck.unused());
}

test "mixed loader refuses a source disagreeing with its recipe and leaves unknown tensors unused" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const bytes: [64]u8 = @splat(0);
    try write(tmp.dir, &.{
        .{ .name = "kda.weight", .dtype = "BF16", .shape = &.{ 1, 32 }, .bytes = &bytes },
        .{ .name = "kda.weight_scale", .dtype = "U8", .shape = &.{ 1, 1 }, .bytes = &.{127} },
        .{ .name = "unknown.weight", .dtype = "U8", .shape = &.{64}, .bytes = &bytes },
    });
    var path: [128]u8 = undefined;
    var ck = try core.Checkpoint.openModel(a, io, try std.fmt.bufPrint(&path, ".zig-cache/tmp/{s}", .{tmp.sub_path}));
    defer ck.close();
    var recipes = try table();
    defer recipes.deinit();
    try std.testing.expectError(error.BadFp8Source, source.Source.read(&ck, &recipes, "kda.weight"));
    try std.testing.expectError(error.WeightRecipeMismatch, source.Fp4.read(a, &ck, &recipes, "kda"));
    try std.testing.expect(!ck.used.contains("unknown.weight"));
}

test "explicit source conversion gathers FP8 spans while the default Loader refuses BF16 materialization" {
    const W = @import("cuda_weights.zig");
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const codes: [4 * 64]u8 = @splat(0x38);
    const scales = [_]u8{ 127, 128, 126, 129, 127, 125, 130, 127 };
    try write(tmp.dir, &.{
        .{ .name = "model.language_model.kda.weight", .dtype = "F8_E4M3", .shape = &.{ 4, 64 }, .bytes = &codes },
        .{ .name = "model.language_model.kda.weight_scale", .dtype = "U8", .shape = &.{ 4, 2 }, .bytes = &scales },
    });
    var why: core.modelopt.Why = .{};
    var recipes = try core.modelopt.admitRecipes(a,
        \\{"quantization_config":{"quant_method":"modelopt","quant_algo":"MIXED_PRECISION","quantized_layers":{"model.language_model.kda":{"quant_algo":"MXFP8","group_size":32}}}}
    , &why);
    defer recipes.deinit();
    var path: [128]u8 = undefined;
    var ck = try core.Checkpoint.openModel(a, io, try std.fmt.bufPrint(&path, ".zig-cache/tmp/{s}", .{tmp.sub_path}));
    defer ck.close();
    var model: W.Weights = .{ .gpa = a, .config = undefined, .share = undefined };
    defer model.deinit();
    var L: W.Loader = .{ .gpa = a, .d = undefined, .stream = undefined, .pack_dense = undefined, .tools = .{ .pack_dense = undefined, .recipes = &recipes }, .w = &model, .ck = &ck };
    defer L.host.deinit(a);
    const t = try L.get("kda.weight", .{});
    try std.testing.expectEqual(core.checkpoint.DType.f8_e4m3, t.dtype);
    try std.testing.expectError(error.UnexpectedTensor, L.cols(t, 31, 33, .bf16));
    try std.testing.expectEqual(@as(usize, 0), L.host.items.len);
    const explicit = try source.Source.read(&ck, &recipes, "model.language_model.kda.weight");
    for (0..4) |row| try explicit.append(a, &L.host, row * 64 + 31, row * 64 + 33, .bf16);
    try std.testing.expectEqualSlices(u16, &.{ 0x3f80, 0x4000, 0x3f00, 0x4080, 0x3f80, 0x3e80, 0x4100, 0x3f80 }, @alignCast(std.mem.bytesAsSlice(u16, L.host.items)));
    L.host.clearRetainingCapacity();
    try std.testing.expectError(error.UnexpectedTensor, L.rows(t, 1, 2, 4, .bf16));
    try explicit.append(a, &L.host, 64, 128, .bf16);
    try std.testing.expectEqual(@as(usize, 128), L.host.items.len);
    try std.testing.expectEqual(@as(u16, 0x3f00), std.mem.readInt(u16, L.host.items[0..2], .little));
    try std.testing.expectEqual(@as(usize, 0), ck.unused());
}

test "mixed loader packs a TP2 CSF expert and a plain MTP expert through the shared host gather" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    const alloc = arena.allocator();
    var items: std.ArrayList(Item) = .empty;
    const plain: [256]u8 = @splat(0x38);
    const bases: [64]u8 = @splat(0x38);
    const fields: [128]u8 = @splat(0);
    const offsets: [5]u16 = @splat(0);
    const pages = [_]u32{0};
    const globals = [_]f32{ 0.25, 0.5, 2 };
    for ([_]u32{ 3, 45 }) |layer| {
        inline for (.{ "gate_proj", "up_proj", "down_proj" }, [_]u8{ 0x22, 0x44, 0x33 }, 0..) |proj, code, m| {
            const module = try std.fmt.allocPrint(alloc, "model.language_model.layers.{d}.mlp.experts.0.{s}", .{ layer, proj });
            const codes = try alloc.alloc(u8, 2048);
            @memset(codes, code);
            try items.append(alloc, .{ .name = try std.fmt.allocPrint(alloc, "{s}.weight", .{module}), .dtype = "U8", .shape = &.{ 64, 32 }, .bytes = codes });
            try items.append(alloc, .{ .name = try std.fmt.allocPrint(alloc, "{s}.weight_scale_2", .{module}), .dtype = "F32", .shape = &.{}, .bytes = std.mem.asBytes(&globals[m]) });
            if (layer == 45) {
                try items.append(alloc, .{ .name = try std.fmt.allocPrint(alloc, "{s}.weight_scale", .{module}), .dtype = "F8_E4M3", .shape = &.{ 64, 4 }, .bytes = &plain });
            } else inline for (.{ "bases", "codes", "page_bases", "offsets", "exceptions" }, .{ "U8", "U8", "U32", "U16", "U16" }, .{ @as([]const u8, &bases), @as([]const u8, &fields), std.mem.sliceAsBytes(&pages), std.mem.sliceAsBytes(&offsets), @as([]const u8, &.{}) }, .{ @as([]const usize, &.{64}), @as([]const usize, &.{128}), @as([]const usize, &.{1}), @as([]const usize, &.{5}), @as([]const usize, &.{0}) }) |suffix, dtype, bytes, shape| {
                try items.append(alloc, .{ .name = try std.fmt.allocPrint(alloc, "{s}.weight_scale.csf32_{s}", .{ module, suffix }), .dtype = dtype, .shape = shape, .bytes = bytes });
            }
        }
    }
    try write(tmp.dir, items.items);
    var why: core.modelopt.Why = .{};
    var recipes = try core.modelopt.admitRecipes(a,
        \\{"quantization_config":{"quant_method":"modelopt","quant_algo":"MIXED_PRECISION","quantized_layers":{"model.language_model.layers.3.mlp.experts":{"quant_algo":"NVFP4","group_size":16,"weight_scale_encoding":"csf_32x2_v1"},"model.language_model.layers.45.mlp.experts":{"quant_algo":"W4A16_NVFP4","group_size":16}}}}
    , &why);
    defer recipes.deinit();
    var path: [128]u8 = undefined;
    var ck = try core.Checkpoint.openModel(a, io, try std.fmt.bufPrint(&path, ".zig-cache/tmp/{s}", .{tmp.sub_path}));
    defer ck.close();
    var up: [576]u32 = undefined;
    var down: [288]u32 = undefined;
    var codes: [1024]u8 = undefined;
    var scales: [128]u8 = undefined;
    for ([_]u32{ 3, 45 }) |layer| {
        const g = try source.packExpert(a, &ck, &recipes, layer, 0, .{ .lo = 32, .hi = 64 }, 64, &up, &down, &codes, &scales);
        try std.testing.expectEqualSlices(f32, &globals, &g);
        for (0..4) |block| {
            for (up[block * 144 ..][0..128]) |word| try std.testing.expectEqual(if (block % 2 == 0) @as(u32, 0x22222222) else @as(u32, 0x44444444), word);
            for (up[block * 144 + 128 ..][0..16]) |word| try std.testing.expectEqual(@as(u32, 0x38383838), word);
        }
        for (0..2) |block| {
            for (down[block * 144 ..][0..128]) |word| try std.testing.expectEqual(@as(u32, 0x33333333), word);
            for (down[block * 144 + 128 ..][0..16]) |word| try std.testing.expectEqual(@as(u32, 0x38383838), word);
        }
    }
    try std.testing.expectEqual(@as(usize, 0), ck.unused());
}

test "mixed loader vision converts attention and W4A16 matrices without owning BF16 checkpoint copies" {
    const V = @import("cuda_vision_weights.zig");
    const c = try @import("cuda_vision_config.zig").parse(a, @embedFile("fixtures/fp4-local-config.json"));
    var small = c;
    small.hidden = 32;
    small.heads = 4;
    small.depth = 1;
    small.intermediate = 32;
    small.output = 32;
    small.projection_intermediate = 32;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    const alloc = arena.allocator();
    var items: std.ArrayList(Item) = .empty;
    var specs = try V.Specs.init(small);
    var buf: [256]u8 = undefined;
    while (try specs.next(&buf)) |spec| {
        const mx = std.mem.eql(u8, spec.name, "model.visual.blocks.0.attn.qkv.weight");
        const fp4 = std.mem.eql(u8, spec.name, "model.visual.merger.gate_proj.weight");
        const shape = try alloc.dupe(usize, spec.dims());
        var count: usize = 1;
        for (shape) |dim| count *= dim;
        if (fp4) shape[1] /= 2;
        const bytes = try alloc.alloc(u8, if (mx) count else if (fp4) count / 2 else count * 2);
        @memset(bytes, if (mx) 0x38 else if (fp4) 0x22 else 0);
        const name = try alloc.dupe(u8, spec.name);
        try items.append(alloc, .{ .name = name, .dtype = if (mx) "F8_E4M3" else if (fp4) "U8" else "BF16", .shape = shape, .bytes = bytes });
        if (mx or fp4) {
            const n = spec.shape[0];
            const k = spec.shape[1];
            const scale = try alloc.alloc(u8, n * k / (if (mx) @as(usize, 32) else 16));
            @memset(scale, if (mx) 128 else 0x38);
            try items.append(alloc, .{ .name = try std.fmt.allocPrint(alloc, "{s}_scale", .{name}), .dtype = if (mx) "U8" else "F8_E4M3", .shape = try alloc.dupe(usize, &.{ n, k / (if (mx) @as(usize, 32) else 16) }), .bytes = scale });
            if (fp4) try items.append(alloc, .{ .name = try std.fmt.allocPrint(alloc, "{s}_scale_2", .{name}), .dtype = "F32", .shape = &.{}, .bytes = std.mem.asBytes(&@as(f32, 0.25)) });
        }
    }
    try write(tmp.dir, items.items);
    var why: core.modelopt.Why = .{};
    var recipes = try core.modelopt.admitRecipes(a,
        \\{"quantization_config":{"quant_method":"modelopt","quant_algo":"MIXED_PRECISION","quantized_layers":{"model.visual.blocks.0.attn.qkv":{"quant_algo":"MXFP8","group_size":32},"model.visual.merger.gate_proj":{"quant_algo":"W4A16_NVFP4","group_size":16}}}}
    , &why);
    defer recipes.deinit();
    var path: [128]u8 = undefined;
    var ck = try core.Checkpoint.openModel(a, io, try std.fmt.bufPrint(&path, ".zig-cache/tmp/{s}", .{tmp.sub_path}));
    defer ck.close();
    var w = try V.Weights.loadMixed(a, small, &ck, &recipes);
    defer w.deinit(a);
    const gu = try w.converted(a, @backingInt(V.Root.gate));
    defer a.free(gu);
    for (gu) |value| try std.testing.expectEqual(@as(u16, 0x3e80), value);
    const qkv = try w.converted(a, @typeInfo(V.Root).@"enum".field_names.len + @backingInt(V.Block.qkv_weight));
    defer a.free(qkv);
    for (qkv) |value| try std.testing.expectEqual(@as(u16, 0x4000), value);
    try std.testing.expectEqual(@as(usize, 0), w.root(.gate).bytes.len);
    try std.testing.expectEqual(@as(usize, 0), ck.unused());
}

fn eq64(got: f64, want: f64) !void {
    if (std.math.isNan(want) or std.math.isNan(got)) {
        try std.testing.expect(std.math.isNan(got) and std.math.isNan(want));
        return;
    }
    try std.testing.expectEqual(want, got);
}

fn expectPacked(p: dense.Packed, s: source.Source, rows: []const source.Span, cols: source.Span) !void {
    const m = try p.matrix();
    const stored = try a.alloc(f64, p.out_total * p.in_total);
    defer a.free(stored);
    try m.rowsF64(0, p.out_total, stored);
    var want_n: usize = 0;
    for (rows) |sp| want_n += sp.hi - sp.lo;
    const cn = cols.hi - cols.lo;
    try std.testing.expectEqual(want_n, p.outs);
    try std.testing.expectEqual(cn, p.ins);
    const want = try a.alloc(f64, want_n * cn);
    defer a.free(want);
    var at: usize = 0;
    for (rows) |sp| {
        const rn = sp.hi - sp.lo;
        switch (s.format.?) {
            .mxfp8 => try core.fp8_source.mxfp8(s.weight, s.scale.?, sp, cols, want[at * cn ..][0 .. rn * cn]),
            .fp8_pb_wo => try core.fp8_source.pbWo(s.weight, s.scale.?, sp, cols, want[at * cn ..][0 .. rn * cn]),
            else => unreachable,
        }
        at += rn;
    }
    const j = p.job();
    try std.testing.expectEqual(p.format, j.format);
    try std.testing.expectEqual(p.out0, j.out0);
    try std.testing.expectEqual(p.outs, j.outs);
    try std.testing.expectEqual(p.in0, j.in0);
    try std.testing.expectEqual(p.ins, j.ins);
    try std.testing.expectEqual(p.out_total, j.out_total);
    try std.testing.expectEqual(p.in_total, j.in_total);
    try std.testing.expectEqual(@as(u64, 0), j.w);
    try std.testing.expectEqual(@as(u64, p.codes.len), j.s);
    for (0..p.outs) |r| for (0..p.ins) |c| {
        try eq64(stored[(p.out0 + r) * p.in_total + (p.in0 + c)], want[r * p.ins + c]);
    };
}

test "source dense packs whole and sliced MXFP8 and matches fp64 decode" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var codes: [4 * 64]u8 = @splat(0x38);
    codes[31] = 0xc0;
    codes[32] = 0x40;
    codes[64 + 33] = 0x01;
    const scales = [_]u8{ 127, 128, 126, 129, 127, 125, 130, 127 };
    try write(tmp.dir, &.{
        .{ .name = "kda.weight", .dtype = "F8_E4M3", .shape = &.{ 4, 64 }, .bytes = &codes },
        .{ .name = "kda.weight_scale", .dtype = "U8", .shape = &.{ 4, 2 }, .bytes = &scales },
    });
    var path: [128]u8 = undefined;
    var ck = try core.Checkpoint.openModel(a, io, try std.fmt.bufPrint(&path, ".zig-cache/tmp/{s}", .{tmp.sub_path}));
    defer ck.close();
    var recipes = try table();
    defer recipes.deinit();
    const s = try source.Source.read(&ck, &recipes, "kda.weight");
    var whole = try dense.pack(a, s, &.{core.fp8_source.all(4)}, core.fp8_source.all(64));
    defer whole.deinit(a);
    try expectPacked(whole, s, &.{core.fp8_source.all(4)}, core.fp8_source.all(64));
    try std.testing.expectEqual(@as(u32, 3), whole.format);
    try std.testing.expectEqual(@as(usize, 4 * 64 + 4 * 2), whole.resident());
    var part = try dense.pack(a, s, &.{.{ .lo = 1, .hi = 3 }}, .{ .lo = 32, .hi = 64 });
    defer part.deinit(a);
    try expectPacked(part, s, &.{.{ .lo = 1, .hi = 3 }}, .{ .lo = 32, .hi = 64 });
    try std.testing.expectEqual(@as(u32, 0), part.out0);
    try std.testing.expectEqual(@as(u32, 32), part.ins);
    try std.testing.expectError(error.SourceDenseGeometry, dense.pack(a, s, &.{core.fp8_source.all(4)}, .{ .lo = 16, .hi = 48 }));
}

test "source dense packs FP8_PB_WO whole, unaligned, and head-interleaved kv_b_proj" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var pb: [130 * 130]u8 = @splat(0x38);
    pb[128] = 0xc0;
    pb[128 * 130 + 128] = 0x40;
    const ps = [_]f32{ 0.5, 2, 3, 4 };
    var kv: [512 * 256]u8 = @splat(0x38);
    kv[0] = 0x40;
    kv[256 * 256] = 0xc0;
    kv[128 * 256 + 128] = 0x01;
    const ks = [_]f32{ 0.25, 0.5, 1, 2, 3, 4, 5, 6 };
    try write(tmp.dir, &.{
        .{ .name = "mla.weight", .dtype = "F8_E4M3", .shape = &.{ 130, 130 }, .bytes = &pb },
        .{ .name = "mla.weight_scale", .dtype = "F32", .shape = &.{ 2, 1, 2, 1 }, .bytes = std.mem.sliceAsBytes(&ps) },
        .{ .name = "kv.weight", .dtype = "F8_E4M3", .shape = &.{ 512, 256 }, .bytes = &kv },
        .{ .name = "kv.weight_scale", .dtype = "F32", .shape = &.{ 4, 2 }, .bytes = std.mem.sliceAsBytes(&ks) },
    });
    var path: [128]u8 = undefined;
    var ck = try core.Checkpoint.openModel(a, io, try std.fmt.bufPrint(&path, ".zig-cache/tmp/{s}", .{tmp.sub_path}));
    defer ck.close();
    var why: core.modelopt.Why = .{};
    var recipes = try core.modelopt.admitRecipes(a,
        \\{"quantization_config":{"quant_method":"modelopt","quant_algo":"MIXED_PRECISION","quantized_layers":{"mla":{"quant_algo":"FP8_PB_WO","weight_block_size":[128,128]},"kv":{"quant_algo":"FP8_PB_WO","weight_block_size":[128,128]}}}}
    , &why);
    defer recipes.deinit();
    const mla = try source.Source.read(&ck, &recipes, "mla.weight");
    var whole = try dense.pack(a, mla, &.{core.fp8_source.all(130)}, core.fp8_source.all(130));
    defer whole.deinit(a);
    try expectPacked(whole, mla, &.{core.fp8_source.all(130)}, core.fp8_source.all(130));
    try std.testing.expectEqual(@as(u32, 4), whole.format);
    try std.testing.expectEqual(@as(u32, 130), whole.ins);
    try std.testing.expectError(error.SourceDenseGeometry, whole.checkCuda());
    var mid = try dense.pack(a, mla, &.{.{ .lo = 120, .hi = 130 }}, .{ .lo = 120, .hi = 130 });
    defer mid.deinit(a);
    try expectPacked(mid, mla, &.{.{ .lo = 120, .hi = 130 }}, .{ .lo = 120, .hi = 130 });
    try std.testing.expectEqual(@as(u32, 120), mid.out0);
    try std.testing.expectEqual(@as(u32, 120), mid.in0);
    try std.testing.expectEqual(@as(u32, 10), mid.outs);
    const kv_s = try source.Source.read(&ck, &recipes, "kv.weight");
    const spans = [_]source.Span{ .{ .lo = 256, .hi = 384 }, .{ .lo = 384, .hi = 512 } };
    var heads = try dense.pack(a, kv_s, &spans, core.fp8_source.all(256));
    defer heads.deinit(a);
    try expectPacked(heads, kv_s, &spans, core.fp8_source.all(256));
    try std.testing.expectEqual(@as(u32, 0), heads.out0);
    try std.testing.expectEqual(@as(u32, 256), heads.outs);
    try heads.checkCuda();
    try std.testing.expectError(error.SourceDenseGeometry, dense.pack(a, kv_s, &.{ .{ .lo = 0, .hi = 64 }, .{ .lo = 128, .hi = 256 } }, core.fp8_source.all(256)));
    try std.testing.expectError(error.SourceDenseGeometry, dense.pack(a, mla, &.{ .{ .lo = 120, .hi = 130 }, .{ .lo = 0, .hi = 10 } }, core.fp8_source.all(130)));
}

test "source dense refuses a BF16 linear and a non-mixed gather" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const bytes: [64]u8 = @splat(0);
    try write(tmp.dir, &.{
        .{ .name = "norm.weight", .dtype = "BF16", .shape = &.{ 1, 32 }, .bytes = &bytes },
    });
    var path: [128]u8 = undefined;
    var ck = try core.Checkpoint.openModel(a, io, try std.fmt.bufPrint(&path, ".zig-cache/tmp/{s}", .{tmp.sub_path}));
    defer ck.close();
    var recipes = try table();
    defer recipes.deinit();
    const n = try source.Source.read(&ck, &recipes, "norm.weight");
    try std.testing.expectEqual(@as(?core.modelopt.Algo, null), n.format);
    try std.testing.expectError(error.SourceDenseRequiresFp8, dense.pack(a, n, &.{core.fp8_source.all(1)}, core.fp8_source.all(32)));
}

test "source dense CUDA refusal and upload account only checkpoint bytes and job metadata" {
    const W = @import("cuda_weights.zig");
    const Memory = @import("cuda_test_driver.zig").Memory;
    var driver = Memory.driver();
    const Upload = struct {
        fn copy(dst: u64, src: ?*const anyopaque, n: usize) callconv(.c) cuda.abi.Result {
            const from: [*]const u8 = @ptrCast(src.?);
            @memcpy(Memory.slice(dst, n), from[0..n]);
            return 0;
        }
    };
    driver.api.cuMemcpyHtoD_v2 = Upload.copy;
    var codes: [33 * 128]u8 = @splat(0x38);
    var scales: [33 * 4]u8 = @splat(127);
    var p: dense.Packed = .{ .format = 3, .out0 = 0, .outs = 33, .in0 = 0, .ins = 128, .out_total = 33, .in_total = 128, .codes = &codes, .scales = &scales };
    try p.checkCuda();
    var model: W.Weights = .{ .gpa = a, .config = undefined, .share = undefined };
    defer model.deinit();
    var L: W.Loader = .{ .gpa = a, .d = &driver, .stream = undefined, .pack_dense = undefined, .tools = .{ .pack_dense = undefined, .precision = .{ .dense = .uniform(.source) } }, .w = &model, .ck = undefined };
    defer L.host.deinit(a);
    p.ins = 32;
    try std.testing.expectError(error.SourceDenseGeometry, W.projections.sourceFinish(&L, .kda, "bad", .{}, p));
    p.ins = 128;
    p.in0 = 1;
    try std.testing.expectError(error.SourceDenseGeometry, p.checkCuda());
    p.in0 = 0;
    p.in_total = 130;
    try std.testing.expectError(error.SourceDenseGeometry, p.checkCuda());
    p.in_total = 128;
    for ([_]u8{ 0, 255 }) |bad| {
        scales[0] = bad;
        try std.testing.expectError(error.SourceDenseScale, W.projections.sourceFinish(&L, .kda, "bad", .{}, p));
    }
    scales[0] = 127;
    try std.testing.expectEqual(@as(usize, 0), model.buffers.items.len);
    const proj = try W.projections.sourceFinish(&L, .kda, "source", .{}, p);
    try std.testing.expect(proj == .source);
    try std.testing.expectEqual(@as(usize, 3), model.buffers.items.len);
    const expected = p.resident() + @sizeOf(core.exlr8.job.DenseJob) + 2 * @sizeOf([2]u32);
    var total: usize = 0;
    for (model.buffers.items) |b| total += b.len;
    try std.testing.expectEqual(expected, total);
    try std.testing.expectEqual(p.resident(), model.find("source").?.len);
    try std.testing.expectEqualSlices(u8, &codes, Memory.slice(proj.source.weights, codes.len));
    try std.testing.expectEqualSlices(u8, &scales, Memory.slice(proj.source.weights + codes.len, scales.len));
    try std.testing.expect(total < 33 * 128 * 2);
}
