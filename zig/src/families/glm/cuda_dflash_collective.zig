const std = @import("std");
const cuda = @import("cuda");
pub const Coll = struct {
    lib: *const cuda.nccl.Library,
    comm: cuda.nccl.Comm,
    rank: usize,
    world: usize,
    tp_group: ?*const cuda.tp_group.Group = null,
    pub fn allGather(c: Coll, src: u64, dst: u64, n: usize, dtype: cuda.nccl.DataType, stream: cuda.abi.Stream) !void {
        if (c.tp_group) |group| {
            if (group.world != c.world or group.rank != c.rank) return error.TransportIdentityMismatch;
            return group.allGatherRaw(stream, src, dst, try std.math.mul(usize, n, switch (dtype) {
                .i8, .u8 => 1,
                .f16, .bf16 => 2,
                .i32, .u32, .f32 => 4,
                .i64, .u64, .f64 => 8,
            }));
        }
        try c.lib.check(c.lib.api.ncclAllGather(src, dst, n, dtype, c.comm, stream), "DFlash all gather");
    }
};
