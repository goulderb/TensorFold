//! tfclef's library root: the C names (include/tfclef.h) over the device engine.
const std = @import("std");
const abi = @import("abi.zig");
const engine = @import("engine.zig");

comptime {
    abi.exportAll(abi.Exports(engine.Engine, std.heap.c_allocator));
}
