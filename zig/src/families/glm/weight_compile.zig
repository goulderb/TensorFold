const std = @import("std");
const cuda = @import("cuda");
const W = @import("cuda_weights.zig");
const E = @import("cuda_engine.zig");
const V = @import("cuda_vision_runtime.zig");

export fn glmLoaderCompile(a: *const std.mem.Allocator, io: *const std.Io, d: *const cuda.Driver, stream: *const cuda.Stream, tools: *const W.Tools, dir: [*:0]const u8, c: *const @import("config.zig").Config, share: *const @import("cuda_shard.zig").Share, layers: [*]const u32, count: usize, out: *W.Weights) bool {
    out.* = W.load(a.*, io.*, d, stream.*, tools.*, std.mem.span(dir), c.*, share.*, layers[0..count], true) catch return false;
    return true;
}

export fn glmEngineCompile(a: *const std.mem.Allocator, io: *const std.Io, d: *const cuda.Driver, ctx: *const cuda.Context, dir: [*:0]const u8, lib: *const cuda.nccl.Library, comm: cuda.nccl.Comm, rank: usize, world: usize, options: *const E.Options) ?*E.Engine {
    return E.Engine.init(a.*, io.*, d, ctx, std.mem.span(dir), lib, comm, rank, world, options.*) catch null;
}

export fn glmVisionCompile(a: *const std.mem.Allocator, io: *const std.Io, d: *const cuda.Driver, context: cuda.abi.Context, stream: *const cuda.Stream, dir: [*:0]const u8, probe: *@import("cuda_vision_probe.zig").Probe) ?*V.Service {
    return V.Service.open(a.*, io.*, d, context, stream.*, std.mem.span(dir), probe) catch null;
}

pub fn main() void {}
