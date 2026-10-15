//! Exact BF16 convolution descriptors for archived da6282f9 / Mia b5808ac6 vision/glm.py.
//! Patch input must already be converted from prepared FP32 to BF16. Downsample input
//! is the post-RMSNorm merge-block buffer, not a channels-first copy or matmul substitute.
const std = @import("std");
pub const cudnn = @import("cudnn");
const Config = @import("cuda_vision_config.zig").Config;

fn dim(value: u32) cudnn.Error!c_int {
    return std.math.cast(c_int, value) orelse error.InvalidShape;
}
pub fn patch(c: Config, rows: u32) !cudnn.Spec {
    try c.check();
    if (rows == 0 or rows > 32000 or rows % 4 != 0) return error.InvalidShape;
    const n = try dim(rows);
    const hidden = try dim(c.hidden);
    const spec: cudnn.Spec = .{
        .input = try cudnn.Tensor.contiguous(&.{ n, 3, 2, 14, 14 }),
        .filter = try cudnn.Tensor.contiguous(&.{ hidden, 3, 2, 14, 14 }),
        .output = try cudnn.Tensor.contiguous(&.{ n, hidden, 1, 1, 1 }),
        .bias = try cudnn.Tensor.contiguous(&.{ 1, hidden, 1, 1, 1 }),
        .stride = .{ 2, 14, 14 },
    };
    try spec.check();
    return spec;
}
pub fn downsample(c: Config, rows: u32) !cudnn.Spec {
    try c.check();
    if (rows == 0 or rows > 8000) return error.InvalidShape;
    const n = try dim(rows);
    const hidden = try dim(c.hidden);
    const output = try dim(c.output);
    const row_stride = std.math.mul(c_int, hidden, 4) catch return error.InvalidShape;
    const spec: cudnn.Spec = .{
        .input = .{ .rank = 4, .dims = .{ n, hidden, 2, 2, 1 }, .strides = .{ row_stride, 1, @divTrunc(row_stride, 2), hidden, 1 } },
        .filter = try cudnn.Tensor.contiguous(&.{ output, hidden, 2, 2 }),
        .output = try cudnn.Tensor.contiguous(&.{ n, output, 1, 1 }),
        .bias = try cudnn.Tensor.contiguous(&.{ 1, output, 1, 1 }),
        .stride = .{ 2, 2, 1 },
    };
    try spec.check();
    return spec;
}

pub const Projections = struct {
    patch: cudnn.Convolution,
    downsample: cudnn.Convolution,
    /// API/library and the current CUDA context must outlive both operators.
    /// This is not a tower: the caller still performs 24 blocks and post-RMSNorm.
    pub fn init(api: *const cudnn.Api, c: Config, patch_rows: u32, options: cudnn.Options) !Projections {
        const p = try patch(c, patch_rows);
        const d = try downsample(c, patch_rows / 4);
        var first = try cudnn.Convolution.init(api, p, options);
        errdefer first.deinit();
        return .{ .patch = first, .downsample = try cudnn.Convolution.init(api, d, options) };
    }
    pub fn deinit(self: *Projections) void {
        self.downsample.deinit();
        self.patch.deinit();
        self.* = undefined;
    }
};

test {
    _ = @import("cuda_vision_cudnn_test.zig");
}
