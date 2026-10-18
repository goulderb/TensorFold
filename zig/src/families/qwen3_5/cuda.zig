pub const checkpoint = @import("cuda_checkpoint.zig");
pub const weights = @import("cuda_weights.zig");
pub const kernels = @import("cuda_kernels.zig");
pub const engine = @import("cuda_engine.zig");
pub const Config = checkpoint.Config;
pub const Engine = engine.Engine;
pub const Session = engine.Session;

test {
    _ = @import("cuda_checkpoint_test.zig");
    _ = @import("cuda_weights_test.zig");
    _ = @import("cuda_kernels_test.zig");
    _ = checkpoint;
    _ = weights;
    _ = kernels;
    _ = engine;
    @import("std").testing.refAllDecls(Engine);
    @import("std").testing.refAllDecls(Session);
    @import("std").testing.refAllDecls(weights.Weights);
    @import("std").testing.refAllDecls(kernels.Ops);
}
