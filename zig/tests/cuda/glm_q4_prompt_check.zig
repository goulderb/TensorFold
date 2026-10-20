//! Model-free native proof of the production GLM affine prompt tiles against the original tile.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const qlinear = cuda.qlinear;
const guard = 64;
const max_rows = 2048;

pub fn run(gpu: check.Gpu) !void {
    const room = try gpu.ctx.memInfo();
    try check.expect(room.free > (1536 + 512) * (@as(usize, 1) << 20), "Q4 prompt probe preserves 1536 MiB headroom with a 512 MiB allocation bound", .{});
    var s = try cuda.Stream.init(gpu.d, true);
    defer s.deinit();
    var mods: [4]cuda.Module = undefined;
    var loaded: usize = 0;
    defer for (mods[0..loaded]) |*m| m.unload();
    for ([_][]const u8{ cuda.kernels.qmm_group, cuda.kernels.qmm_prefill, cuda.kernels.lane_gemv, cuda.kernels.affine4_pack }, 0..) |image, i| {
        mods[i] = try cuda.Module.load(gpu.d, image);
        loaded += 1;
    }
    const sms: usize = @intCast(try gpu.ctx.attribute(.multiprocessor_count));
    var kernels = try qlinear.Affine4Kernels.resolve(gpu.d, mods[0], mods[1], mods[2], mods[3], sms, (try gpu.ctx.capability()) == 121);
    defer kernels.deinit();
    try kernels.enablePromptTiles(mods[1]);
    var rng = std.Random.DefaultPrng.init(0x53_04_71_1e);
    const shapes = [_][2]usize{ .{ 12576, 4096 }, .{ 2048, 4096 }, .{ 4096, 128 }, .{ 12288, 4096 } };
    for (shapes) |shape| try matrix(gpu, s, &kernels, rng.random(), shape[0], shape[1]);
    check.pass("glm-q4-prompt: 56 native BF16/FP32 products byte-exact to stock; both production tiles, partial N/M, input strides and destination guards", .{});
}

fn bf(v: f32) u16 {
    return @truncate(@as(u32, @bitCast(v)) >> 16);
}

fn matrix(gpu: check.Gpu, s: cuda.Stream, kernels: *const qlinear.Affine4Kernels, rng: std.Random, n: usize, k: usize) !void {
    var arena = std.heap.ArenaAllocator.init(gpu.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const layout = qlinear.Affine4.layout(n, k);
    const host_w = try a.alloc(u8, layout.words);
    rng.bytes(host_w);
    const host_s = try a.alloc(u16, layout.scales / 2);
    const host_b = try a.alloc(u16, layout.scales / 2);
    for (host_s, host_b) |*scale, *bias| {
        scale.* = bf(0.001 + rng.float(f32) * 0.01);
        bias.* = bf(-0.04 + rng.float(f32) * 0.02);
    }
    var dw = try cuda.DeviceBuffer.fromHost(gpu.d, host_w);
    defer dw.free();
    var ds = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(host_s));
    defer ds.free();
    var db = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(host_b));
    defer db.free();
    const q: qlinear.Affine4 = .{ .w = dw.ptr, .s = ds.ptr, .b = db.ptr, .n = n, .k = k, .npad = layout.npad };
    const ldx = if (k == 128) @as(usize, 12576) else k + 64;
    const host_x = try a.alloc(u16, max_rows * ldx);
    @memset(host_x, 0x7fc0);
    for (0..max_rows) |row| for (host_x[row * ldx ..][0..k]) |*v| {
        v.* = bf(rng.float(f32) - 0.5);
    };
    var dx = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(host_x));
    defer dx.free();
    const out_bytes = max_rows * n * 4 + 2 * guard;
    var original = try cuda.DeviceBuffer.alloc(gpu.d, out_bytes);
    defer original.free();
    var selected = try cuda.DeviceBuffer.alloc(gpu.d, out_bytes);
    defer selected.free();
    defer s.synchronize() catch |err| {
        std.debug.print("FAIL Q4 prompt drain before buffer release: {t}\n", .{err});
        std.process.exit(1);
    };
    const expected = try a.alloc(u8, out_bytes);
    const got = try a.alloc(u8, out_bytes);
    const stock: qlinear.Linear = .{ .affine4 = kernels };
    const production: qlinear.Linear = .{ .affine4 = kernels, .affine_prompt_policy = .glm_production };
    for ([_]usize{ 1, 14, 16, 64, 512, 1024, 2048 }) |rows| {
        for ([_]bool{ false, true }) |wide| {
            try original.fill8(0xa5, s.handle);
            try selected.fill8(0xa5, s.handle);
            const input: qlinear.Rows = .{ .x = dx.ptr, .ldx = ldx };
            if (wide) {
                try stock.promptF32(s, .{ .affine4 = q }, input, original.ptr + guard, rows);
                try production.promptF32(s, .{ .affine4 = q }, input, selected.ptr + guard, rows);
            } else {
                try stock.prompt(s, .{ .affine4 = q }, input, original.ptr + guard, rows);
                try production.prompt(s, .{ .affine4 = q }, input, selected.ptr + guard, rows);
            }
            try s.synchronize();
            try original.download(0, expected);
            try selected.download(0, got);
            const end = guard + rows * n * @as(usize, if (wide) 4 else 2);
            try check.sameBytes("native GLM tile versus original prompt tile, including full destination allocation", got, expected);
            for (got[0..guard]) |v| try check.expect(v == 0xa5, "Q4 prompt prefix guard", .{});
            for (got[end..]) |v| try check.expect(v == 0xa5, "Q4 prompt unrequested rows and suffix guard", .{});
            for (0..rows) |row| {
                const start = guard + row * n * @as(usize, if (wide) 4 else 2);
                var changed = false;
                for (got[start .. start + n * @as(usize, if (wide) 4 else 2)]) |v| changed = changed or v != 0xa5;
                try check.expect(changed, "Q4 prompt every active row was written", .{});
                for (0..n) |col| {
                    const at = start + col * @as(usize, if (wide) 4 else 2);
                    const bits = if (wide) std.mem.readInt(u32, got[at..][0..4], .little) else @as(u32, std.mem.readInt(u16, got[at..][0..2], .little)) << 16;
                    try check.expect(std.math.isFinite(@as(f32, @bitCast(bits))), "Q4 prompt finite product with poisoned input stride padding", .{});
                }
            }
            std.debug.print("PASS glm-q4-prompt n={d} k={d} rows={d} ldx={d} f32={any} tile={d}\n", .{ n, k, rows, ldx, wide, @backingInt(qlinear.affinePromptTile(.glm_production, rows, n, k)) });
        }
    }
    const x_after = try a.alloc(u8, host_x.len * 2);
    try dx.download(0, x_after);
    try check.sameBytes("Q4 prompt input rows and stride padding unchanged", x_after, std.mem.sliceAsBytes(host_x));
    const w_after = try a.alloc(u8, host_w.len);
    try dw.download(0, w_after);
    try check.sameBytes("Q4 prompt weight codes unchanged", w_after, host_w);
    const tables_after = try a.alloc(u8, host_s.len * 2);
    try ds.download(0, tables_after);
    try check.sameBytes("Q4 prompt scales unchanged", tables_after, std.mem.sliceAsBytes(host_s));
    try db.download(0, tables_after);
    try check.sameBytes("Q4 prompt biases unchanged", tables_after, std.mem.sliceAsBytes(host_b));
}
