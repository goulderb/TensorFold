//! Owning native GLM vision operator image and checked launches. Not a vision encoder.
//! Caller owns an already-current context, stream and device buffers through completion.
//! Span.len is bytes; every layout is contiguous. Successful calls enqueue work only:
//! the caller must synchronize/check asynchronous errors. Device data is not scanned.
//! RMS and activation BF16 cast boundaries are intentional, but FP32 reduction,
//! affine, rsqrt, exp and erf agreement with target Torch is NOT numerically qualified.
//! No attention, GEMM, weight upload, codec or full-tower execution is provided.
const std = @import("std");
const cuda = @import("cuda");
pub const symbols = [_][:0]const u8{ "tf_glm_vision_cast", "tf_glm_vision_rms", "tf_glm_vision_qkv_rope", "tf_glm_vision_residual", "tf_glm_vision_swiglu", "tf_glm_vision_layernorm_gelu" };
pub const projector_eps: f32 = 1e-5;
pub const DataType = enum { bf16, f32 };
pub const Span = struct { ptr: u64, len: usize, dtype: DataType = .bf16 };
pub const Matrix = struct {
    rows: u32,
    cols: u32,
    pub fn elements(self: Matrix) !usize {
        if (self.rows == 0 or self.rows > 32000 or self.cols == 0 or self.cols > 4096) return error.InvalidOperatorShape;
        return elementCount(@as(u64, self.rows) * self.cols);
    }
};
pub const Qkv = struct {
    groups: u32,
    sequence: u32,
    heads: u32,
    dim: u32,
    pub fn rows(self: Qkv) !u32 {
        if (self.groups == 0 or self.sequence == 0 or self.sequence % 4 != 0 or self.heads == 0 or self.heads > 128 or
            self.dim == 0 or self.dim > 256 or self.dim % 4 != 0) return error.InvalidOperatorShape;
        const n = @as(u64, self.groups) * self.sequence;
        if (n > 32000) return error.InvalidOperatorShape;
        _ = try elementCount(n * self.heads * self.dim * 3);
        return @intCast(n);
    }
    pub fn elements(self: Qkv) !usize {
        return @as(usize, try self.rows()) * self.heads * self.dim;
    }
};
fn elementCount(n: u64) !usize {
    if (n == 0 or n > std.math.maxInt(c_int)) return error.InvalidOperatorShape;
    return @intCast(n);
}
fn finitePositive(value: f32) !void {
    if (!std.math.isFinite(value) or value <= 0) return error.InvalidOperatorScalar;
}
fn checked(span: Span, dtype: DataType, elements: usize) !usize {
    if (span.dtype != dtype) return error.UnsupportedOperatorDType;
    const bytes = std.math.mul(usize, elements, if (dtype == .bf16) @as(usize, 2) else 4) catch return error.InvalidOperatorShape;
    if (span.ptr == 0 or span.len < bytes or span.ptr % (if (dtype == .bf16) @as(u64, 2) else 4) != 0) return error.InvalidOperatorSpan;
    _ = std.math.add(u64, span.ptr, bytes) catch return error.InvalidOperatorSpan;
    return bytes;
}
fn disjoint(out: Span, n: usize, input: Span, m: usize, inplace: bool) !void {
    if (inplace and out.ptr == input.ptr and n == m) return;
    if (out.ptr < input.ptr + m and input.ptr < out.ptr + n) return error.OperatorAliasing;
}
const Op = enum { cast, rms, qkv, residual, swiglu, layernorm };
pub const Kernels = struct {
    module: cuda.Module,
    functions: [symbols.len]cuda.Function,
    /// Driver/context outlive this image. All six symbols are required before any operator is usable.
    pub fn load(d: *const cuda.Driver, image: []const u8) !Kernels {
        if (image.len == 0 or @intFromPtr(image.ptr) % 8 != 0) return error.MissingVisionKernelImage;
        var module = try cuda.Module.load(d, image);
        if (module.handle == null) return error.InvalidVisionModule;
        errdefer module.unload();
        var functions: [symbols.len]cuda.Function = undefined;
        for (symbols, 0..) |symbol, i| {
            var f: cuda.abi.Function = null;
            const status = d.api.cuModuleGetFunction(&f, module.handle, symbol.ptr);
            if (status != cuda.abi.success or f == null) return error.MissingVisionKernel;
            functions[i] = .{ .d = d, .handle = f };
        }
        return .{ .module = module, .functions = functions };
    }
    pub fn deinit(self: *Kernels) void {
        self.module.unload();
        self.* = undefined;
    }
    fn launch(self: *const Kernels, op: Op, stream: cuda.Stream, blocks: u32, threads: u32, args: *cuda.Args) !void {
        if (stream.d != self.module.d) return error.InvalidOperatorStream;
        try cuda.launch.launch(self.functions[@backingInt(op)], .{ .grid = .{ .x = blocks }, .block = .{ .x = threads } }, stream, args);
    }
    pub fn cast(self: *const Kernels, stream: cuda.Stream, count: u64, x: Span, out: Span) !void {
        const n = try elementCount(count);
        const xn = try checked(x, .f32, n);
        const yn = try checked(out, .bf16, n);
        try disjoint(out, yn, x, xn, false);
        var args: cuda.Args = .{};
        args.add(x.ptr);
        args.add(out.ptr);
        args.add(@as(c_int, @intCast(n)));
        try self.launch(.cast, stream, @intCast((n + 255) / 256), 256, &args);
    }
    pub fn rms(self: *const Kernels, stream: cuda.Stream, shape: Matrix, eps: f32, x: Span, weight: Span, out: Span) !void {
        const n = try shape.elements();
        try finitePositive(eps);
        const xn = try checked(x, .bf16, n);
        const wn = try checked(weight, .bf16, shape.cols);
        const yn = try checked(out, .bf16, n);
        try disjoint(out, yn, x, xn, true);
        try disjoint(out, yn, weight, wn, false);
        var a: cuda.Args = .{};
        a.add(x.ptr);
        a.add(weight.ptr);
        a.add(out.ptr);
        a.add(@as(c_int, @intCast(shape.rows)));
        a.add(@as(c_int, @intCast(shape.cols)));
        a.add(eps);
        try self.launch(.rms, stream, shape.rows, 256, &a);
    }
    pub fn qkvRope(self: *const Kernels, stream: cuda.Stream, shape: Qkv, eps: f32, dense: Span, qw: Span, kw: Span, cosine: Span, sine: Span, q: Span, k: Span, v: Span) !void {
        const n = try shape.elements();
        const rows = try shape.rows();
        try finitePositive(eps);
        const reads = [_]Span{ dense, qw, kw, cosine, sine };
        const lengths = [_]usize{ try checked(dense, .bf16, n * 3), try checked(qw, .bf16, shape.dim), try checked(kw, .bf16, shape.dim), try checked(cosine, .f32, @as(usize, shape.sequence) * shape.dim), try checked(sine, .f32, @as(usize, shape.sequence) * shape.dim) };
        const outputs = [_]Span{ q, k, v };
        var sizes: [3]usize = undefined;
        for (outputs, 0..) |out, i| {
            sizes[i] = try checked(out, .bf16, n);
            for (reads, lengths) |input, m| try disjoint(out, sizes[i], input, m, false);
            for (0..i) |j| try disjoint(out, sizes[i], outputs[j], sizes[j], false);
        }
        var a: cuda.Args = .{};
        for ([_]Span{ dense, qw, kw, cosine, sine, q, k, v }) |span| a.add(span.ptr);
        a.add(@as(c_int, @intCast(rows)));
        a.add(@as(c_int, @intCast(shape.sequence)));
        a.add(@as(c_int, @intCast(shape.heads)));
        a.add(@as(c_int, @intCast(shape.dim)));
        a.add(eps);
        try self.launch(.qkv, stream, rows * shape.heads, 128, &a);
    }
    pub fn residual(self: *const Kernels, stream: cuda.Stream, count: u64, x: Span, update: Span, out: Span) !void {
        const n = try elementCount(count);
        const xn = try checked(x, .bf16, n);
        const un = try checked(update, .bf16, n);
        const yn = try checked(out, .bf16, n);
        try disjoint(out, yn, x, xn, true);
        try disjoint(out, yn, update, un, true);
        var a: cuda.Args = .{};
        a.add(x.ptr);
        a.add(update.ptr);
        a.add(out.ptr);
        a.add(@as(c_int, @intCast(n)));
        try self.launch(.residual, stream, @intCast((n + 255) / 256), 256, &a);
    }
    pub fn swiglu(self: *const Kernels, stream: cuda.Stream, count: u64, limit: f32, gate: Span, up: Span, out: Span) !void {
        const n = try elementCount(count);
        try finitePositive(limit);
        const gn = try checked(gate, .bf16, n);
        const un = try checked(up, .bf16, n);
        const yn = try checked(out, .bf16, n);
        try disjoint(out, yn, gate, gn, true);
        try disjoint(out, yn, up, un, true);
        var a: cuda.Args = .{};
        a.add(gate.ptr);
        a.add(up.ptr);
        a.add(out.ptr);
        a.add(@as(c_int, @intCast(n)));
        a.add(limit);
        try self.launch(.swiglu, stream, @intCast((n + 255) / 256), 256, &a);
    }
    pub fn projectorNormGelu(self: *const Kernels, stream: cuda.Stream, shape: Matrix, x: Span, weight: Span, bias: Span, out: Span) !void {
        const n = try shape.elements();
        const xn = try checked(x, .bf16, n);
        const wn = try checked(weight, .bf16, shape.cols);
        const bn = try checked(bias, .bf16, shape.cols);
        const yn = try checked(out, .bf16, n);
        try disjoint(out, yn, x, xn, true);
        try disjoint(out, yn, weight, wn, false);
        try disjoint(out, yn, bias, bn, false);
        var a: cuda.Args = .{};
        a.add(x.ptr);
        a.add(weight.ptr);
        a.add(bias.ptr);
        a.add(out.ptr);
        a.add(@as(c_int, @intCast(shape.rows)));
        a.add(@as(c_int, @intCast(shape.cols)));
        a.add(projector_eps);
        try self.launch(.layernorm, stream, shape.rows, 256, &a);
    }
};

test {
    _ = @import("glm_vision_ops_test.zig");
}
