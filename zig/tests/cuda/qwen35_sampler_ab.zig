const std = @import("std");
const builtin = @import("builtin");
const build_options = @import("qwen_bench_options");
const qwen = @import("qwen35_cuda");
const cuda = @import("cuda");
const admission = @import("qwen35_probe_device.zig");
const turn = @import("qwen35_sampler_turn.zig");

comptime {
    if (build_options.engine_optimization != .fast)
        @compileError("Sampler comparisons require engine and current sampler modules built with ReleaseFast");
}

const Journal = struct {
    a: std.mem.Allocator,
    io: std.Io,
    file: std.Io.File,

    fn write(self: Journal, value: anytype) !void {
        var buffer: std.Io.Writer.Allocating = .init(self.a);
        defer buffer.deinit();
        try std.json.Stringify.value(value, .{}, &buffer.writer);
        try buffer.writer.writeByte('\n');
        try self.file.writeStreamingAll(self.io, buffer.written());
    }
};

pub fn main(init: std.process.Init) !u8 {
    const a = init.arena.allocator();
    const args = try init.minimal.args.toSlice(a);
    if (args.len == 2 and std.mem.eql(u8, args[1], "--identity")) {
        var output: std.Io.Writer.Allocating = .init(a);
        try std.json.Stringify.value(.{
            .source_commit = build_options.source_commit,
            .engine_optimization = @tagName(build_options.engine_optimization),
            .root_optimization = @tagName(builtin.mode),
        }, .{}, &output.writer);
        try output.writer.writeByte('\n');
        try std.Io.File.stdout().writeStreamingAll(init.io, output.written());
        return 0;
    }
    const check = args.len == 3 and std.mem.eql(u8, args[1], "--check");
    if (!check and args.len != 4) {
        try std.Io.File.stderr().writeStreamingAll(init.io, "usage: tf-qwen35-sampler-ab --identity\n       tf-qwen35-sampler-ab --check INPUT.json\n       tf-qwen35-sampler-ab MODEL_DIR INPUT.json NEW_OUTPUT_DIR\n");
        return 2;
    }
    if (check) {
        const bytes = try std.Io.Dir.cwd().readFileAlloc(init.io, args[2], a, .limited(4 << 20));
        _ = try turn.parse(a, bytes);
        try std.Io.File.stdout().writeStreamingAll(init.io, "Sampler input valid; CPU only, no model or CUDA initialization.\n");
        return 0;
    }
    // Exclusive creation prevents mixing a failed or previous run with this trial.
    try std.Io.Dir.cwd().createDir(init.io, args[3], .default_dir);
    const file = try std.Io.Dir.cwd().createFile(init.io, try std.fs.path.join(a, &.{ args[3], "results.jsonl" }), .{ .exclusive = true });
    defer file.close(init.io);
    const journal: Journal = .{ .a = a, .io = init.io, .file = file };
    const bytes = std.Io.Dir.cwd().readFileAlloc(init.io, args[2], a, .limited(4 << 20)) catch |err| return failed(journal, err);
    const input = turn.parse(a, bytes) catch |err| return failed(journal, err);
    execute(init, args[1], input, bytes, journal) catch |err| return failed(journal, err);
    return 0;
}

fn failed(journal: Journal, err: anyerror) !u8 {
    try journal.write(.{ .kind = "failure", .error_name = @errorName(err), .success = false });
    try std.Io.File.stderr().writeStreamingAll(journal.io, @errorName(err));
    return 1;
}

fn floor(model: *qwen.Engine) ![2]usize {
    var result: [2]usize = undefined;
    for (model.partitions, 0..) |*part, i| {
        try part.context.makeCurrent();
        const memory = try part.context.memInfo();
        if (memory.free < 1536 * (1 << 20)) return error.GpuMemoryFloor;
        result[i] = memory.free >> 20;
    }
    return result;
}

fn equalTokens(expected: []const u32, actual: []const u32) !void {
    if (!std.mem.eql(u32, expected, actual)) return error.TokenMismatch;
}

fn checkRows(old: []const u16, current: []const u16) !void {
    if (!std.mem.eql(u16, old, current)) return error.RawLogitMismatch;
    for (old) |bits| if ((bits & 0x7f80) == 0x7f80) return error.NonfiniteLogits;
}

fn hash(bytes: []const u8) [64]u8 {
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &digest, .{});
    return std.fmt.bytesToHex(digest, .lower);
}

fn median(values: [5]i96) i96 {
    var sorted = values;
    std.mem.sort(i96, &sorted, {}, std.sort.asc(i96));
    return sorted[2];
}

fn execute(init: std.process.Init, model_dir: []const u8, input: turn.Input, input_bytes: []const u8, journal: Journal) !void {
    const a = init.arena.allocator();
    const pinned = try admission.Admission.parse(init.environ_map.get("CUDA_DEVICE_ORDER"), init.environ_map.get("CUDA_VISIBLE_DEVICES"), input.devices, input.expected_device_uuids);
    const identities = blk: {
        var driver = try cuda.Driver.open();
        defer driver.close();
        break :blk try pinned.verify(a, &driver);
    };
    const input_sha = hash(input_bytes);
    try journal.write(.{
        .kind = "start",
        .scope = "synthetic direct single-session generation; no HTTP, tool-gate, cache or concurrency qualification",
        .input_sha256 = &input_sha,
        .input = input,
        .source_commit = build_options.source_commit,
        .root_optimization = @tagName(builtin.mode),
        .engine_optimization = @tagName(build_options.engine_optimization),
        .old_sampler_optimization = "ReleaseFast",
        .current_sampler_optimization = @tagName(build_options.engine_optimization),
        .old_sampler_revision = "ad247b3daf045015d59bd106a88591bff961e488",
        .device_identities = identities,
        .layout = "layer split at 20; not tensor parallelism",
        .warmup_pairs = 1,
        .measured_pairs = 5,
        .materiality = "decode-heavy median total reduction >=5% and >=4/5 positive paired gains",
        .eos = "ignored for fixed work",
        .reset_in_total = false,
    });
    const model = try qwen.Engine.open(init.gpa, init.io, model_dir, .{ .devices = input.devices, .context = input.context, .cut = input.cut });
    defer model.deinit();
    var session = try model.createSession();
    defer session.deinit();
    var sampler = try turn.Sampler.init(init.gpa, session.logits.len);
    defer sampler.deinit();
    var primary_pass = false;
    for (input.cases, 0..) |case, case_index| {
        const old_ids = try a.alloc(u32, case.max_tokens);
        const current_ids = try a.alloc(u32, case.max_tokens);
        const reference = try a.alloc(u32, case.max_tokens);
        const row_count = case.max_tokens * session.logits.len;
        const old_rows = try init.gpa.alloc(u16, row_count);
        defer init.gpa.free(old_rows);
        const current_rows = try init.gpa.alloc(u16, row_count);
        defer init.gpa.free(current_rows);
        _ = try floor(model);
        _ = try turn.run(true, init.io, &session, &sampler, .old, case.prompt, old_ids, old_rows);
        _ = try floor(model);
        _ = try turn.run(true, init.io, &session, &sampler, .current, case.prompt, current_ids, current_rows);
        _ = try floor(model);
        const row_sha = hash(std.mem.sliceAsBytes(old_rows));
        const current_row_sha = hash(std.mem.sliceAsBytes(current_rows));
        try journal.write(.{ .kind = "parity_outputs", .case_name = case.name, .old_tokens = old_ids, .current_tokens = current_ids, .old_raw_rows_sha256 = &row_sha, .current_raw_rows_sha256 = &current_row_sha });
        try equalTokens(old_ids, current_ids);
        try checkRows(old_rows, current_rows);
        @memcpy(reference, old_ids);
        try journal.write(.{ .kind = "parity", .case_name = case.name, .consumed_raw_bf16_rows_equal = true, .row_count = case.max_tokens, .raw_rows_sha256 = &row_sha, .all_output_ids_equal = true });
        var old_times: [5]i96 = undefined;
        var current_times: [5]i96 = undefined;
        var paired_gains: [5]f64 = undefined;
        var positive: usize = 0;
        for (0..6) |pair| {
            const order: [2]turn.Arm = if (pair % 2 == 0) .{ .old, .current } else .{ .current, .old };
            var times: [2]turn.Times = undefined;
            for (order, 0..) |arm, order_index| {
                const free_before = try floor(model);
                const output = if (arm == .old) old_ids else current_ids;
                const measured = try turn.run(false, init.io, &session, &sampler, arm, case.prompt, output, &.{});
                const free_after = try floor(model);
                times[@backingInt(arm)] = measured;
                try journal.write(.{
                    .kind = "turn",
                    .case_name = case.name,
                    .warmup = pair == 0,
                    .pair = pair,
                    .order_index = order_index,
                    .arm = @tagName(arm),
                    .times = measured,
                    .decode_tokens_per_second = @as(f64, @floatFromInt(measured.decode_tokens)) * 1e9 / @as(f64, @floatFromInt(measured.decode_ns)),
                    .tokens = output,
                    .free_mib_before = free_before,
                    .free_mib_after = free_after,
                });
                try equalTokens(reference, output);
            }
            try equalTokens(old_ids, current_ids);
            try journal.write(.{ .kind = "pair_verified", .case_name = case.name, .pair = pair, .complete_output_identity = true, .repeat_determinism = true });
            if (pair > 0) {
                const old_time = times[0].total_ns;
                const current_time = times[1].total_ns;
                if (old_time <= 0 or current_time <= 0) return error.InvalidTiming;
                old_times[pair - 1] = old_time;
                current_times[pair - 1] = current_time;
                paired_gains[pair - 1] = 1 - @as(f64, @floatFromInt(current_time)) / @as(f64, @floatFromInt(old_time));
                if (current_time < old_time) positive += 1;
            }
        }
        const old_median = median(old_times);
        const current_median = median(current_times);
        const reduction = 1 - @as(f64, @floatFromInt(current_median)) / @as(f64, @floatFromInt(old_median));
        const material = reduction >= 0.05 and positive >= 4;
        if (case_index == 0) primary_pass = material;
        try journal.write(.{
            .kind = "case_summary",
            .case_name = case.name,
            .old_median_total_ns = old_median,
            .current_median_total_ns = current_median,
            .median_total_reduction = reduction,
            .paired_total_reductions = paired_gains,
            .positive_pairs = positive,
            .material = material,
        });
    }
    try journal.write(.{ .kind = "complete", .correctness_pass = true, .primary_materiality_pass = primary_pass, .production_qualified = false });
}
