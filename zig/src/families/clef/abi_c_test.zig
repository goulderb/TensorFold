//! include/tfclef.h compiled beside the export layer: its layouts against the Zig structs, then a C call sequence
//! through the exported names, all over the scripted engine. No device.
const std = @import("std");
const abi = @import("abi.zig");
const record = @import("record.zig");
const Fake = @import("abi_fake.zig").Fake;

comptime {
    abi.exportAll(abi.Exports(Fake, std.testing.allocator));
}

extern fn tfclef_c_layout(count: *usize) [*]const usize;
extern fn tfclef_c_scenario() c_int;

fn expected() []const usize {
    const Q = record.C.Question;
    const R = record.C.Record;
    const O = abi.OpenOptions;
    const I = abi.Info;
    const S = record.Span;
    const sets = .{
        .{ S, .{ "start", "end" } },
        .{ Q, .{ "kind", "question", "first_option", "option_count" } },
        .{ R, .{ "ids", "n_ids", "questions", "n_questions", "options", "n_options" } },
        .{ O, .{ "model_dir", "head_sha256", "device_uuid", "cap_bytes", "max_tokens", "flags" } },
        .{ I, .{ "abi_version", "max_tokens", "max_questions", "max_options", "budget_bytes", "device_free_bytes", "device_total_bytes", "device_uuid", "source_commit", "head_only" } },
    };
    comptime var out: []const usize = &.{};
    inline for (sets) |set| {
        out = out ++ .{@sizeOf(set[0])};
        inline for (set[1]) |field| out = out ++ .{ @offsetOf(set[0], field), @sizeOf(@FieldType(set[0], field)) };
    }
    out = out ++ .{@sizeOf(abi.Status)};
    return out;
}

test "the C header lays out every ABI struct exactly as the export layer reads it" {
    var n: usize = 0;
    const got = tfclef_c_layout(&n);
    try std.testing.expectEqualSlices(usize, expected(), got[0..n]);
}

test "a C caller opens, inspects, refuses, runs the head and closes through the exported names" {
    Fake.reset();
    try std.testing.expectEqual(@as(c_int, 0), tfclef_c_scenario());
    try std.testing.expectEqual(@as(usize, 1), Fake.opened);
    try std.testing.expectEqual(@as(usize, 1), Fake.closed);
    try std.testing.expect(!Fake.closed_poisoned);
}
