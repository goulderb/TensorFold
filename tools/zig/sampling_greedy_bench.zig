const std = @import("std");
const sampling = @import("sampling");

// Keep the pre-optimization comparator and allocation to measure the removed work.
const Candidate = struct { value: f64, id: u64 };
fn before(a: std.mem.Allocator, values: []const f64, ids: []const u64) !u64 {
    const order = try a.alloc(Candidate, ids.len);
    defer a.free(order);
    for (order, values, ids) |*o, value, id| o.* = .{ .value = value, .id = id };
    std.mem.sort(Candidate, order, {}, struct {
        fn less(_: void, x: Candidate, y: Candidate) bool {
            return x.value > y.value or (x.value == y.value and x.id < y.id);
        }
    }.less);
    return order[0].id;
}

const Sample = struct { row: usize, pair: usize, baseline_first: bool, baseline_ns: i64, candidate_ns: i64, token: u64 };
fn measure(io: std.Io, gpa: std.mem.Allocator, values: []const f64, ids: []const u64, baseline: bool) !struct { ns: i64, token: u64 } {
    std.mem.doNotOptimizeAway(values);
    std.mem.doNotOptimizeAway(ids);
    const start = std.Io.Clock.awake.now(io).toNanoseconds();
    const token = if (baseline) try before(gpa, values, ids) else try sampling.choose(gpa, values, ids, 0, .{ .seed = 0, .temperature = 0 });
    std.mem.doNotOptimizeAway(token);
    return .{ .ns = @intCast(std.Io.Clock.awake.now(io).toNanoseconds() - start), .token = token };
}

pub fn main(init: std.process.Init) !void {
    const a = init.arena.allocator();
    const args = try init.minimal.args.toSlice(a);
    if (args.len < 3) return error.Usage;
    var samples: std.ArrayList(Sample) = .empty;
    var hashes: std.ArrayList([64]u8) = .empty;
    for (args[2..], 0..) |path, row| {
        const bytes = try std.Io.Dir.cwd().readFileAlloc(init.io, path, a, .limited(2 << 20));
        if (bytes.len != 248320 * 4) return error.InvalidFixture;
        var digest: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(bytes, &digest, .{});
        try hashes.append(a, std.fmt.bytesToHex(digest, .lower));
        const values = try a.alloc(f64, 248320);
        const ids = try a.alloc(u64, 248320);
        for (values, ids, 0..) |*value, *id, i| {
            const x: f32 = @bitCast(std.mem.readInt(u32, bytes[4 * i ..][0..4], .little));
            if (!std.math.isFinite(x)) return error.NonfiniteFixture;
            value.* = x;
            id.* = i;
        }
        for (0..12) |iteration| {
            const baseline_first = iteration % 2 == 0;
            const first = try measure(init.io, init.gpa, values, ids, baseline_first);
            const second = try measure(init.io, init.gpa, values, ids, !baseline_first);
            if (first.token != second.token) return error.TokenMismatch;
            if (iteration < 2) continue;
            const baseline = if (baseline_first) first else second;
            const candidate = if (baseline_first) second else first;
            try samples.append(a, .{ .row = row, .pair = iteration - 2, .baseline_first = baseline_first, .baseline_ns = baseline.ns, .candidate_ns = candidate.ns, .token = baseline.token });
        }
    }
    var output: std.Io.Writer.Allocating = .init(a);
    try std.json.Stringify.value(.{
        .scope = "CPU greedy sampler only; finite saved logits; no GPU or serving speedup established",
        .baseline_source = "ad247b3da:zig/src/core/lanes/sampling.zig temperature<=0 sorted path",
        .warmup_pairs_per_row = 2,
        .measured_pairs_per_row = 10,
        .fixture_sha256 = hashes.items,
        .samples = samples.items,
        .correct = true,
        .measured_calls = samples.items.len * 2,
        .errors = 0,
    }, .{ .whitespace = .indent_2 }, &output.writer);
    try output.writer.writeByte('\n');
    try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = args[1], .data = output.written() });
}
