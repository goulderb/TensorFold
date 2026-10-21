const std = @import("std");
const core = @import("core");
const checkpoint = @import("cuda_bf16_checkpoint.zig");

const RangeCounter = struct {
    count: usize = 0,
    bytes: usize = 0,

    pub fn tensor(self: *@This(), binding: anytype, value: core.safetensors.Tensor) !void {
        if (binding.conversion == .host_rows or binding.role == .embedding or binding.role == .output)
            return error.UnexpectedHostCallback;
        self.count = try std.math.add(usize, self.count, 1);
        self.bytes = try std.math.add(usize, self.bytes, value.bytes.len);
    }
};

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 2) return error.ExpectedCheckpointDirectory;
    var logical: usize = undefined;
    var row_bytes: usize = undefined;
    var counter: RangeCounter = .{};
    var host = host: {
        var plan = try checkpoint.Plan.inspect(init.gpa, init.io, args[1]);
        defer plan.deinit();
        logical = plan.bytes().host_tables;
        row_bytes = try std.math.mul(usize, plan.geometry.hidden, 2);
        break :host try plan.upload(init.gpa, init.io, args[1], &counter);
    };
    defer host.deinit();
    const rows = try init.gpa.alloc(u8, row_bytes);
    defer init.gpa.free(rows);
    std.debug.print("callback_ranges={d} callback_logical_bytes={d} callback_payload_bytes_read=0\nlogical_host_table_bytes={d} retained_shard_mapped_bytes={d}\n", .{ counter.count, counter.bytes, logical, host.mappedBytes() });
    inline for (.{ checkpoint.HostTables.Kind.embedding, checkpoint.HostTables.Kind.output }) |kind| {
        try host.gather(kind, &.{0}, rows);
        var digest: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(rows, &digest, .{});
        std.debug.print("{s} row=0 gathered_bytes={d} sha256={x}\n", .{ @tagName(kind), rows.len, digest });
    }
    std.debug.print("Read bounded shard metadata and one BF16 row per host table. No device calls or full payload hash.\n", .{});
}
