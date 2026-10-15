//! Finite-data CPU equation reference for the native vision operator units.
//! BF16 stage boundaries match da6282f9 vision/glm.py. This is not a CUDA parity oracle or encoder.
//! Pinned vision/glm.py SHA256: 07da2a80bd37a95c5728026024ccdddbc36725d1d84066aee116860987b93e01.
//! FP32 reductions mirror this slice's lane tree, not Torch's chosen reduction.
//! CPU 1/sqrt, exp and libc erff may differ from CUDA rsqrtf/expf/erff.
//! Inputs must be finite. Use disjoint slices except exact elementwise/RMS/LN
//! input/output alias; the native wrapper, not this equation helper, checks spans.
const std = @import("std");
pub const native = @import("vision_ops");
extern "c" fn erff(f32) callconv(.c) f32;

pub fn bf16(value: f32) u16 {
    const bits: u32 = @bitCast(value);
    if (bits & 0x7fffffff > 0x7f800000) return 0x7fc0;
    return @truncate((bits +% (0x7fff + ((bits >> 16) & 1))) >> 16);
}
pub fn float(bits: u16) f32 {
    return @bitCast(@as(u32, bits) << 16);
}
fn positive(x: f32) !void {
    if (!std.math.isFinite(x) or x <= 0) return error.InvalidOperatorScalar;
}
fn finite(bits: u16) !f32 {
    const x = float(bits);
    if (!std.math.isFinite(x)) return error.NonFiniteVisionReferenceInput;
    return x;
}
fn inverse(x: []const u16, eps: f32, lanes: usize) !f32 {
    @setFloatMode(.strict);
    try positive(eps);
    var sums: [256]f32 = @splat(0);
    for (x, 0..) |bits, i| {
        const v = try finite(bits);
        sums[i % lanes] += v * v;
    }
    var step = lanes / 2;
    while (step > 0) : (step /= 2) for (0..step) |i| {
        sums[i] += sums[i + step];
    };
    const mean = sums[0] / @as(f32, @floatFromInt(x.len));
    if (!std.math.isFinite(mean)) return error.NonFiniteVisionReferenceInput;
    return 1 / @sqrt(mean + eps);
}
fn norm(x: u16, weight: u16, inv: f32) u16 {
    @setFloatMode(.strict);
    return bf16(float(bf16(float(x) * inv)) * float(weight));
}
pub fn cast(x: []const f32, out: []u16) !void {
    if (x.len == 0 or x.len != out.len) return error.InvalidOperatorShape;
    for (x, out) |v, *dst| {
        if (!std.math.isFinite(v)) return error.NonFiniteVisionReferenceInput;
        dst.* = bf16(v);
    }
}
pub fn rms(shape: native.Matrix, eps: f32, x: []const u16, weight: []const u16, out: []u16) !void {
    const n = try shape.elements();
    if (x.len != n or out.len != n or weight.len != shape.cols) return error.InvalidOperatorShape;
    for (weight) |v| _ = try finite(v);
    for (0..shape.rows) |row| {
        const input = x[row * shape.cols ..][0..shape.cols];
        const inv = try inverse(input, eps, 256);
        for (input, weight, out[row * shape.cols ..][0..shape.cols]) |v, w, *dst| dst.* = norm(v, w, inv);
    }
}
pub fn qkvRope(shape: native.Qkv, eps: f32, dense: []const u16, qw: []const u16, kw: []const u16, cosine: []const f32, sine: []const f32, q: []u16, k: []u16, v: []u16) !void {
    @setFloatMode(.strict);
    const n = try shape.elements();
    const rows = try shape.rows();
    const hidden = @as(usize, shape.heads) * shape.dim;
    const angles = @as(usize, shape.sequence) * shape.dim;
    if (dense.len != 3 * n or qw.len != shape.dim or kw.len != shape.dim or cosine.len != angles or sine.len != angles or q.len != n or k.len != n or v.len != n) return error.InvalidOperatorShape;
    for (qw, kw) |a, z| {
        _ = try finite(a);
        _ = try finite(z);
    }
    for (cosine, sine) |a, z| if (!std.math.isFinite(a) or !std.math.isFinite(z)) return error.NonFiniteVisionReferenceInput;
    for (0..rows) |row| for (0..shape.heads) |head| {
        const qi = (row * 3) * hidden + head * shape.dim;
        const ki = qi + hidden;
        const vi = qi + 2 * hidden;
        const qinv = try inverse(dense[qi..][0..shape.dim], eps, 128);
        const kinv = try inverse(dense[ki..][0..shape.dim], eps, 128);
        for (0..shape.dim) |j| {
            const other = (j + shape.dim / 2) % shape.dim;
            const sign: f32 = if (j < shape.dim / 2) -1 else 1;
            const position = (row % shape.sequence) * shape.dim + j;
            const dst = row * hidden + head * shape.dim + j;
            const q0 = float(norm(dense[qi + j], qw[j], qinv));
            const q1 = sign * float(norm(dense[qi + other], qw[other], qinv));
            const k0 = float(norm(dense[ki + j], kw[j], kinv));
            const k1 = sign * float(norm(dense[ki + other], kw[other], kinv));
            q[dst] = bf16(q0 * cosine[position] + q1 * sine[position]);
            k[dst] = bf16(k0 * cosine[position] + k1 * sine[position]);
            v[dst] = dense[vi + j];
        }
    };
}
pub fn residual(x: []const u16, update: []const u16, out: []u16) !void {
    @setFloatMode(.strict);
    if (x.len == 0 or update.len != x.len or out.len != x.len) return error.InvalidOperatorShape;
    for (x, update, out) |a, z, *dst| dst.* = bf16(try finite(a) + try finite(z));
}
pub fn swiglu(gate: []const u16, up: []const u16, out: []u16, limit: f32) !void {
    @setFloatMode(.strict);
    try positive(limit);
    if (gate.len == 0 or up.len != gate.len or out.len != gate.len) return error.InvalidOperatorShape;
    for (gate, up, out) |gb, ub, *dst| {
        const g = float(bf16(@min(try finite(gb), limit)));
        const u = float(bf16(std.math.clamp(try finite(ub), -limit, limit)));
        const activated = bf16(g / (1 + @exp(-g)));
        dst.* = bf16(float(activated) * u);
    }
}
const Moments = struct { mean: f32 = 0, m2: f32 = 0, count: u32 = 0 };
fn combine(a: Moments, z: Moments) Moments {
    @setFloatMode(.strict);
    if (z.count == 0) return a;
    if (a.count == 0) return z;
    const delta = z.mean - a.mean;
    const count = a.count + z.count;
    const fraction = @as(f32, @floatFromInt(z.count)) / @as(f32, @floatFromInt(count));
    return .{ .mean = a.mean + delta * fraction, .m2 = (a.m2 + z.m2) + (delta * delta) * (@as(f32, @floatFromInt(a.count)) * fraction), .count = count };
}
pub fn projectorNormGelu(shape: native.Matrix, x: []const u16, weight: []const u16, bias: []const u16, out: []u16) !void {
    @setFloatMode(.strict);
    const n = try shape.elements();
    if (x.len != n or out.len != n or weight.len != shape.cols or bias.len != shape.cols) return error.InvalidOperatorShape;
    for (weight, bias) |w, z| {
        _ = try finite(w);
        _ = try finite(z);
    }
    for (0..shape.rows) |row| {
        var moments: [256]Moments = @splat(.{});
        for (x[row * shape.cols ..][0..shape.cols], 0..) |bits, j| moments[j % 256] = combine(moments[j % 256], .{ .mean = try finite(bits), .count = 1 });
        var step: usize = 128;
        while (step > 0) : (step /= 2) for (0..step) |i| {
            moments[i] = combine(moments[i], moments[i + step]);
        };
        const inverse_std = 1 / @sqrt(moments[0].m2 / @as(f32, @floatFromInt(shape.cols)) + native.projector_eps);
        if (!std.math.isFinite(inverse_std)) return error.NonFiniteVisionReferenceInput;
        for (0..shape.cols) |j| {
            const normalized = (float(x[row * shape.cols + j]) - moments[0].mean) * inverse_std;
            const affine = normalized * float(weight[j]) + float(bias[j]);
            const y = float(bf16(affine));
            out[row * shape.cols + j] = bf16((y * 0.5) * (1 + erff(y * @as(f32, 0.7071067811865475244))));
        }
    }
}

test {
    _ = @import("cuda_vision_forward_ops_test.zig");
}
