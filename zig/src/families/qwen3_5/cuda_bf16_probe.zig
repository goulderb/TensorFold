const std = @import("std");
const checkpoint = @import("cuda_bf16_checkpoint.zig");

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 2) return error.ExpectedCheckpointDirectory;
    var plan = try checkpoint.Plan.inspect(init.gpa, init.io, args[1]);
    defer plan.deinit();
    std.debug.print("text_tensors={d} excluded_visual_tensors={d}\nbytes={any}\n", .{
        plan.tensorCount(), plan.excluded_visual_count, plan.bytes(),
    });
}
