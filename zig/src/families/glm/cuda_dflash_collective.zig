const cuda = @import("cuda");
pub const Coll = struct {
    lib: *const cuda.nccl.Library,
    comm: cuda.nccl.Comm,
    rank: usize,
    world: usize,
    pub fn allGather(c: Coll, src: u64, dst: u64, n: usize, dtype: cuda.nccl.DataType, stream: cuda.abi.Stream) !void {
        try c.lib.check(c.lib.api.ncclAllGather(src, dst, n, dtype, c.comm, stream), "DFlash all gather");
    }
};
