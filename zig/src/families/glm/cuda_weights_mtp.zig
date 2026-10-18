//! The GLM MTP layer, norms and rank-local draft head loader.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const weights = @import("cuda_weights.zig");
const Loader = weights.Loader;
const Mtp = weights.Mtp;
const Hc = weights.Hc;
const Layer = weights.Layer;
const prefix = weights.prefix;
const source = @import("cuda_weight_source.zig");
const b16 = @import("cuda_b16.zig");
const projections = @import("cuda_weights_projs.zig");
const x3 = @import("cuda_weights_exl3.zig");
const quant = @import("cuda_quant.zig");
const q4 = @import("cuda_q4.zig");
const toBf16 = core.quant.convert.bf16;

/// weights.load's MTPW: layer `layers` as plain DSA + MoE, norms, eh_proj, draft head from the rank's head rows.
pub fn load(L: *Loader) !Mtp {
    const c = &L.w.config;
    const i = c.layers;
    try L.whole(try L.get("layers.{d}.input_layernorm.weight", .{i}), .bf16);
    const in_norm = try L.put("layers.{d}.in_norm", .{i});
    try L.whole(try L.get("layers.{d}.post_attention_layernorm.weight", .{i}), .bf16);
    const post_norm = try L.put("layers.{d}.post_norm", .{i});
    const none: Hc = .{ .fn_w = 0, .base = 0, .scale = 0 };
    const layer_w: Layer = .{ .index = i, .attn_hc = none, .ffn_hc = none, .in_norm = in_norm, .post_norm = post_norm, .mixer = .{ .mla = try L.dsa(i) }, .ffn = .{ .moe = try L.moeLayer(i) } };
    L.ck.release();
    if (L.w.exl3) {
        var o3: [3]u64 = undefined;
        for (&o3, [_][]const u8{ "enorm.weight", "hnorm.weight", "shared_head.norm.weight" }, [_][]const u8{ "enorm", "hnorm", "norm" }) |*p, name, what| {
            try L.whole(try source.Source.read(L.ck, L.tools.recipes, try std.fmt.bufPrint(&L.name_buf, prefix ++ "layers.{d}.{s}", .{ i, name })), .bf16);
            p.* = try L.put("mtp.{s}", .{what});
        }
        return .{ .enorm = o3[0], .hnorm = o3[1], .eh = .{ .q = try x3.whole(L, "layers.{d}.eh_proj", i) }, .norm = o3[2], .layer = layer_w, .draft = null };
    }
    var eh: b16.Proj = undefined;
    var out: [4]u64 = undefined;
    for (&out, [_][]const u8{ "enorm.weight", "hnorm.weight", "eh_proj.weight", "shared_head.norm.weight" }, [_][]const u8{ "enorm", "hnorm", "eh", "norm" }) |*p, name, what| {
        if (std.mem.eql(u8, what, "eh") and L.tools.precision.dense.get(.mtp_eh) == .source) {
            eh = try L.packFinish(.mtp_eh, "mtp.eh", .{}, try source.Source.read(L.ck, L.tools.recipes, try std.fmt.bufPrint(&L.name_buf, prefix ++ "layers.{d}.{s}", .{ i, name })), &.{.{ .lo = 0, .hi = c.hidden }}, .{ .lo = 0, .hi = 2 * c.hidden });
            p.* = 0;
            continue;
        }
        try L.whole(try source.Source.read(L.ck, L.tools.recipes, try std.fmt.bufPrint(&L.name_buf, prefix ++ "layers.{d}.{s}", .{ i, name })), .bf16);
        if (std.mem.eql(u8, what, "eh")) {
            eh = try projections.finish(L, .mtp_eh, "mtp.eh", .{}, c.hidden, 2 * c.hidden);
            p.* = 0;
        } else p.* = try L.put("mtp.{s}", .{what});
    }
    // the draft head: quantize4 of the BF16 head's rows, then make_q4's tiling
    const qk = L.tools.quant orelse return error.NoQuantizer;
    const tk = L.tools.q4 orelse return error.NoQuantizer;
    const n = L.w.share.vocab.len();
    const k: usize = c.hidden;
    var raw: [3]cuda.DeviceBuffer = undefined;
    var made: usize = 0;
    defer for (raw[0..made]) |*b| b.free();
    for (&raw, [_]usize{ n * k / 2, n * k / 32, n * k / 32 }) |*b, size| {
        b.* = try L.scratch(size);
        made += 1;
    }
    try quant.q4(qk, L.stream, L.w.head.b16, n, k, raw[0].ptr, raw[1].ptr, raw[2].ptr);
    var tiled: [3]u64 = undefined;
    for (&tiled, q4.bytes(n, k), [_][]const u8{ "words", "scales", "biases" }) |*p, size, what| {
        const b = try cuda.DeviceBuffer.alloc(L.d, size);
        try L.w.buffers.append(L.gpa, b);
        try L.w.named.append(L.gpa, .{ .name = try std.fmt.allocPrint(L.gpa, "draft_head.{s}", .{what}), .ptr = b.ptr, .len = b.len });
        p.* = b.ptr;
    }
    const draft: q4.Q4 = .{ .w = tiled[0], .s = tiled[1], .b = tiled[2], .n = n, .k = k, .npad = q4.npadOf(n) };
    try q4.pack(tk, L.stream, raw[0].ptr, raw[1].ptr, raw[2].ptr, draft);
    try L.stream.synchronize();
    return .{ .enorm = out[0], .hnorm = out[1], .eh = eh, .norm = out[3], .layer = layer_w, .draft = draft };
}

test "bf16 rounding of fp32 is to nearest even" {
    try std.testing.expectEqual(@as(u16, 0x3f80), toBf16(1.0));
    try std.testing.expectEqual(@as(u16, 0x3f80), toBf16(@bitCast(@as(u32, 0x3f808000)))); // a tie to even (down)
    try std.testing.expectEqual(@as(u16, 0x3f82), toBf16(@bitCast(@as(u32, 0x3f818000)))); // a tie to even (up)
    try std.testing.expectEqual(@as(u16, 0x3f81), toBf16(@bitCast(@as(u32, 0x3f808001))));
}
