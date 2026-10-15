//! Pinned da6282f9 vision/glm.py _block noncausal SDPA view adapter, not a tower.
//! vision/glm.py SHA256 07da2a80bd37a95c5728026024ccdddbc36725d1d84066aee116860987b93e01.
//! D=1024, H=16, d=64; scale=d**-0.5=0.125; every canvas patch attends within its image/frame pair.
//! Q,K must already be normalized and 2D-rotated; V unchanged. Packed and separate layouts are explicit.
const std = @import("std");
pub const sdpa = @import("vision_sdpa");
const cuda = @import("cuda");
const Config = @import("cuda_vision_config.zig").Config;
pub const hidden = 1024;
pub const heads = 16;
pub const head_dim = 64;
pub const scale: f32 = 0.125;
pub const Layout = enum { packed_qkv, separate_qkv };
pub const Shape = struct {
    batch: u32,
    sequence: u32,
    fn check(self: Shape) !void {
        // 8,000 visual tokens -> 32,000 canvas patches per image; independent image/frame-pair batch lanes.
        if (self.batch == 0 or self.batch > 256 or self.sequence == 0 or self.sequence > 32000 or self.sequence % 4 != 0)
            return error.InvalidShape;
        const rows = std.math.mul(u32, self.batch, self.sequence) catch return error.InvalidShape;
        if (rows > 32000) return error.InvalidShape;
    }
    pub fn bytes(self: Shape, layout: Layout) !usize {
        try self.check();
        const rows = std.math.mul(usize, self.batch, self.sequence) catch return error.InvalidShape;
        const width: usize = if (layout == .packed_qkv) 3 * hidden else hidden;
        const n = std.math.mul(usize, rows, width) catch return error.InvalidShape;
        return std.math.mul(usize, n, 2) catch error.InvalidShape;
    }
};
fn makeSpec(c: Config, shape: Shape, layout: Layout) !sdpa.Spec {
    try c.check();
    try shape.check();
    if (c.hidden != hidden or c.heads != heads or c.headDim() != head_dim) return error.InvalidShape;
    const batch: i64 = shape.batch;
    const seq: i64 = shape.sequence;
    const row: i64 = if (layout == .packed_qkv) 3 * hidden else hidden;
    const input: sdpa.Tensor = .{ .dims = .{ batch, heads, seq, head_dim }, .strides = .{ seq * row, head_dim, row, 1 } };
    const output: sdpa.Tensor = .{ .dims = input.dims, .strides = .{ seq * hidden, head_dim, hidden, 1 } };
    const spec: sdpa.Spec = .{ .q = input, .k = input, .v = input, .output = output, .scale = scale };
    try spec.check();
    return spec;
}
/// Original Linear qkv -> view(B,S,3,H,d), with Q/K updated in the same strided storage.
pub fn packedSpec(c: Config, shape: Shape) !sdpa.Spec {
    return makeSpec(c, shape, .packed_qkv);
}
/// Native qkvRope producer outputs: each BF16 Q/K/V is independent contiguous BSHD.
pub fn separateSpec(c: Config, shape: Shape) !sdpa.Spec {
    return makeSpec(c, shape, .separate_qkv);
}

pub const Packed = struct { qkv: cuda.DeviceBuffer, output: cuda.DeviceBuffer, workspace: cuda.DeviceBuffer };
pub const Separate = struct { q: cuda.DeviceBuffer, k: cuda.DeviceBuffer, v: cuda.DeviceBuffer, output: cuda.DeviceBuffer, workspace: cuda.DeviceBuffer };
pub const Buffers = union(Layout) { packed_qkv: Packed, separate_qkv: Separate };
fn span(buffer: cuda.DeviceBuffer) sdpa.Span {
    return .{ .ptr = buffer.ptr, .len = buffer.len };
}
fn sameDriver(expected: *const cuda.Driver, buffers: []const cuda.DeviceBuffer) !void {
    for (buffers) |buffer| if (buffer.d != expected) return error.DriverMismatch;
}
/// Checked borrowed subviews; no repacking or ownership transfer.
pub fn packedViews(shape: Shape, buffers: Packed) !sdpa.Buffers {
    try sameDriver(buffers.qkv.d, &.{ buffers.output, buffers.workspace });
    try span(buffers.qkv).checked(try shape.bytes(.packed_qkv), 16);
    try span(buffers.output).checked(try shape.bytes(.separate_qkv), 16);
    const k_offset = hidden * 2;
    const v_offset = 2 * hidden * 2;
    return .{
        .q = span(buffers.qkv),
        .k = .{ .ptr = buffers.qkv.ptr + k_offset, .len = buffers.qkv.len - k_offset },
        .v = .{ .ptr = buffers.qkv.ptr + v_offset, .len = buffers.qkv.len - v_offset },
        .output = span(buffers.output),
        .workspace = span(buffers.workspace),
    };
}
/// Distinct BSHD buffers remain distinct; packed strides are never applied to them.
pub fn separateViews(shape: Shape, buffers: Separate) !sdpa.Buffers {
    try sameDriver(buffers.q.d, &.{ buffers.k, buffers.v, buffers.output, buffers.workspace });
    const bytes = try shape.bytes(.separate_qkv);
    for ([_]cuda.DeviceBuffer{ buffers.q, buffers.k, buffers.v, buffers.output }) |buffer| try span(buffer).checked(bytes, 16);
    return .{ .q = span(buffers.q), .k = span(buffers.k), .v = span(buffers.v), .output = span(buffers.output), .workspace = span(buffers.workspace) };
}

pub const Attention = struct {
    driver: *const cuda.Driver,
    shape: Shape,
    layout: Layout,
    operator: sdpa.Attention,
    /// CUDA context is caller-owned/current; API/library outlive the operator.
    pub fn init(api: *const sdpa.Api, driver: *const cuda.Driver, c: Config, shape: Shape, layout: Layout, options: sdpa.Options) !Attention {
        const spec = try makeSpec(c, shape, layout);
        return .{ .driver = driver, .shape = shape, .layout = layout, .operator = try sdpa.Attention.init(api, spec, options) };
    }
    pub fn deinit(self: *Attention) void {
        self.operator.deinit();
        self.* = undefined;
    }
    /// Borrows buffers and stream; caller retains them and this object until GPU completion.
    /// Non-thread-safe. workspace_need is available as operator.workspace_need after init.
    pub fn run(self: *const Attention, buffers: Buffers, stream: cuda.Stream) !void {
        if (std.meta.activeTag(buffers) != self.layout) return error.InvalidSdpaLayout;
        if (stream.d != self.driver) return error.DriverMismatch;
        const views = switch (buffers) {
            .packed_qkv => |interleaved| blk: {
                try sameDriver(self.driver, &.{ interleaved.qkv, interleaved.output, interleaved.workspace });
                break :blk try packedViews(self.shape, interleaved);
            },
            .separate_qkv => |separate| blk: {
                try sameDriver(self.driver, &.{ separate.q, separate.k, separate.v, separate.output, separate.workspace });
                break :blk try separateViews(self.shape, separate);
            },
        };
        try self.operator.run(views, stream.handle);
    }
};

test {
    _ = @import("cuda_vision_sdpa_test.zig");
}
