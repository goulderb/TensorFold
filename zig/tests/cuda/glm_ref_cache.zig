const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const glm = @import("glm_cuda");
const qr = @import("quant_ref_tests.zig");
const check = @import("check.zig");
const bf = qr.helpers.bfOf;

pub const Uploaded = struct { dev: glm.cache.Cache, f64s: []f64 };

pub fn upload(r: *qr.RigT, set: glm.glue.Set, input: []const u16, width: usize, format: glm.cache.Format) !Uploaded {
    const rows = input.len / width;
    const values = try r.a.alloc(f64, input.len);
    for (values, input) |*v, b| v.* = bf(b);
    if (format == .bf16) return .{ .dev = .{ .ptr = try r.dev(std.mem.sliceAsBytes(input)) }, .f64s = values };
    const stride = format.rowBytes(width);
    const raw = try r.zeros((rows + 2) * stride);
    const cache: glm.cache.Cache = .{ .ptr = raw + stride, .format = format };
    const zero: i32 = 0;
    const pos = try r.dev(std.mem.asBytes(&zero));
    if (width == 512) {
        const src = try r.dev(std.mem.sliceAsBytes(input));
        try glm.latent.latentWrite(set, r.s, src, width, cache, pos, rows, width);
    } else {
        try check.expect(width == 128, "pooled cache test width", .{});
        const ik = try r.a.alloc(u16, rows * 4 * width);
        for (0..rows) |row| for (0..4) |part| {
            @memcpy(ik[(row * 4 + part) * width ..][0..width], input[row * width ..][0..width]);
        };
        const gates = try r.a.alloc(u16, ik.len);
        @memset(gates, 0);
        const ape: [4 * 128]u16 = @splat(0);
        var a: cuda.Args = .{};
        inline for (.{ try r.dev(std.mem.sliceAsBytes(ik)), try r.dev(std.mem.sliceAsBytes(gates)), try r.dev(std.mem.sliceAsBytes(&ape)), cache.ptr, pos, @as(i64, @intCast(rows * 4)) }) |v| a.add(v);
        try cuda.launch.launch(set.glue.pool_keys_fp8, .{ .grid = .{ .x = @intCast(rows + 2) }, .block = .{ .x = 32 } }, r.s, &a);
    }
    const bytes = try r.back(raw, (rows + 2) * stride);
    for (bytes[0..stride]) |b| try check.expect(b == 0xff, "cache writer keeps leading guard", .{});
    for (bytes[(rows + 1) * stride ..]) |b| try check.expect(b == 0xff, "cache writer keeps trailing guard", .{});
    for (0..rows) |row| {
        const row_bytes = bytes[(row + 1) * stride ..][0..stride];
        var largest: f64 = 0;
        for (input[row * width ..][0..width]) |v| largest = @max(largest, @abs(bf(v)));
        const exponent = if (largest == 0) -126 else @max(-126, @ceil(std.math.log2(largest)) - 8);
        const want_scale = std.math.pow(f64, 2, exponent);
        const scale = std.mem.bytesToValue(f32, row_bytes[width..][0..4]);
        try check.expect(scale == want_scale, "FP8 row {d} scale matches host exponent reference", .{row});
        for (row_bytes[width + 4 ..]) |b| try check.expect(b == 0, "FP8 padding is deterministic", .{});
        for (0..width) |col| {
            const got = core.quant.fp8.e4m3(row_bytes[col]);
            const want = core.quant.fp8.toE4m3(bf(input[row * width + col]) / want_scale);
            try check.expect(got == want, "FP8 row {d} column {d}: code {d}, host {d}", .{ row, col, got, want });
            values[row * width + col] = got * scale;
        }
    }
    check.pass("FP8 {d} rows x {d}: codes, scales, padding and guards match the host reference", .{ rows, width });
    return .{ .dev = cache, .f64s = values };
}
