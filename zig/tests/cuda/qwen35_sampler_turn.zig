const std = @import("std");
const lanes = @import("lanes");
const before = @import("sampler_before");

pub const vocab = 248320;
pub const expected_uuids = [_][]const u8{ "GPU-f2c0ea1e-a2be-f0b4-6c23-ffe33b537a06", "GPU-dd2a988c-fe38-24ab-233a-9e9bb3b2645a" };
pub const Arm = enum { old, current };
pub const Case = struct { name: []const u8, prompt: []const u32, max_tokens: usize };
pub const Input = struct {
    context: usize,
    devices: []const c_int,
    expected_device_uuids: []const []const u8,
    cut: usize,
    cases: []const Case,
    provenance: struct {
        scope: []const u8,
        tokenizer_sha256: []const u8,
        description_sha256: []const u8,
        description: []const u8,
        seed: u64,
        temperature: f64,
    },
};

pub fn parse(a: std.mem.Allocator, bytes: []const u8) !Input {
    const input = try std.json.parseFromSliceLeaky(Input, a, bytes, .{ .ignore_unknown_fields = true });
    try validate(input);
    return input;
}

pub fn validate(input: Input) !void {
    if (input.context != 4096 or input.cut != 20 or
        !std.mem.eql(c_int, input.devices, &.{ 0, 1 }) or input.expected_device_uuids.len != 2 or
        input.cases.len != 2 or input.provenance.seed != 0 or input.provenance.temperature != 0 or
        input.provenance.scope.len == 0 or input.provenance.description.len == 0)
        return error.InvalidBenchmarkInput;
    for (input.expected_device_uuids, expected_uuids) |actual, expected|
        if (!std.mem.eql(u8, actual, expected)) return error.InvalidBenchmarkInput;
    for ([_][]const u8{ input.provenance.tokenizer_sha256, input.provenance.description_sha256 }) |hash| {
        if (hash.len != 64) return error.InvalidBenchmarkInput;
        for (hash) |c| _ = std.fmt.charToDigit(c, 16) catch return error.InvalidBenchmarkInput;
    }
    const lengths = [_]usize{ 64, 275 };
    const outputs = [_]usize{ 128, 27 };
    const names = [_][]const u8{ "decode-heavy", "prefill-heavy" };
    for (input.cases, 0..) |case, i| {
        if (!std.mem.eql(u8, case.name, names[i]) or case.prompt.len != lengths[i] or case.max_tokens != outputs[i]) return error.InvalidBenchmarkInput;
        for (case.prompt) |id| if (id >= vocab) return error.InvalidToken;
    }
}

pub const Times = struct {
    reset_ns: i96,
    prefill_ns: i96,
    ttft_ns: i96,
    decode_ns: i96,
    total_ns: i96,
    decode_tokens: usize,
    appends: usize,
    draws: usize,
};

pub const Sampler = struct {
    allocator: std.mem.Allocator,
    values: []f64,
    ids: []u64,

    pub fn init(a: std.mem.Allocator, width: usize) !Sampler {
        if (width == 0) return error.EmptyLogits;
        const values = try a.alloc(f64, width);
        errdefer a.free(values);
        const ids = try a.alloc(u64, width);
        for (ids, 0..) |*id, i| id.* = i;
        return .{ .allocator = a, .values = values, .ids = ids };
    }

    pub fn deinit(self: *Sampler) void {
        self.allocator.free(self.values);
        self.allocator.free(self.ids);
    }

    fn draw(self: *Sampler, arm: Arm, row: []const u16, position: usize) !u32 {
        for (self.values, row) |*value, bits| value.* = @as(f32, @bitCast(@as(u32, bits) << 16));
        return @intCast(switch (arm) {
            .old => try before.choose(self.allocator, self.values, self.ids, position, .{ .seed = 0, .temperature = 0 }),
            .current => try lanes.sampling.choose(self.allocator, self.values, self.ids, position, .{ .seed = 0, .temperature = 0 }),
        });
    }
};

fn now(io: std.Io) i96 {
    return std.Io.Clock.awake.now(io).toNanoseconds();
}

// Capture is used only in the untimed parity pass; measured turns instantiate it off.
pub fn run(comptime capture: bool, io: std.Io, session: anytype, sampler: anytype, arm: Arm, prompt: []const u32, output: []u32, rows: []u16) !Times {
    return runWithAppender(capture, io, session, session, sampler, arm, prompt, output, rows);
}

pub fn runWithAppender(comptime capture: bool, io: std.Io, session: anytype, appender: anytype, sampler: anytype, arm: Arm, prompt: []const u32, output: []u32, rows: []u16) !Times {
    if (prompt.len == 0 or output.len == 0 or sampler.values.len != session.logits.len) return error.InvalidTurn;
    if (capture and rows.len != output.len * session.logits.len) return error.InvalidCapture;
    for (prompt) |id| if (id >= session.logits.len) return error.InvalidToken;
    const reset_start = now(io);
    try session.reset();
    const start = now(io);
    for (prompt) |token| try appender.append(&.{token});
    const prefilled = now(io);
    var first: i96 = 0;
    for (output, 0..) |*token, i| {
        if (i != 0) try appender.append(output[i - 1 .. i]);
        if (capture) @memcpy(rows[i * session.logits.len ..][0..session.logits.len], session.logits[0..]);
        token.* = try sampler.draw(arm, session.logits[0..], session.position);
        if (i == 0) first = now(io);
    }
    const end = now(io);
    return .{
        .reset_ns = start - reset_start,
        .prefill_ns = prefilled - start,
        .ttft_ns = first - start,
        .decode_ns = end - first,
        .total_ns = end - start,
        .decode_tokens = output.len - 1,
        .appends = prompt.len + output.len - 1,
        .draws = output.len,
    };
}

const FakeSession = struct {
    logits: [4]u16 = .{ 0x3f80, 0x4000, 0x4040, 0x4080 },
    position: usize = 0,
    resets: usize = 0,
    appends: usize = 0,
    consumed: [400]u32 = @splat(0),

    fn reset(s: *FakeSession) !void {
        s.position = 0;
        s.appends = 0;
        s.resets += 1;
    }

    fn append(s: *FakeSession, tokens: []const u32) !void {
        if (tokens.len != 1) return error.NotSingleToken;
        s.consumed[s.position] = tokens[0];
        s.position += 1;
        s.appends += 1;
    }
};

test "fixed turns consume P+N-1 single tokens and draw N even when every draw is EOS" {
    var sampler = try Sampler.init(std.testing.allocator, 4);
    defer sampler.deinit();
    for ([_]Arm{ .old, .current }) |arm| {
        var session: FakeSession = .{};
        var prompt: [275]u32 = @splat(1);
        var output: [128]u32 = undefined;
        for ([_][2]usize{ .{ 64, 128 }, .{ 275, 27 }, .{ 1, 1 } }) |shape| {
            const result = try run(false, std.testing.io, &session, &sampler, arm, prompt[0..shape[0]], output[0..shape[1]], &.{});
            try std.testing.expectEqual(shape[0] + shape[1] - 1, session.appends);
            try std.testing.expectEqual(session.appends, session.position);
            try std.testing.expectEqual(shape[1], result.draws);
            try std.testing.expectEqual(shape[1] - 1, result.decode_tokens);
            for (output[0..shape[1]]) |id| try std.testing.expectEqual(@as(u32, 3), id);
            for (session.consumed[0..shape[0]]) |id| try std.testing.expectEqual(@as(u32, 1), id);
            for (session.consumed[shape[0]..session.position]) |id| try std.testing.expectEqual(@as(u32, 3), id);
            try std.testing.expectEqual(result.total_ns, result.ttft_ns + result.decode_ns);
            try std.testing.expect(result.ttft_ns >= result.prefill_ns);
        }
        try std.testing.expectEqual(@as(usize, 3), session.resets);
    }
}

test "untimed capture includes each consumed raw BF16 row and both arms match" {
    var sampler = try Sampler.init(std.testing.allocator, 4);
    defer sampler.deinit();
    var session: FakeSession = .{};
    var old: [3]u32 = undefined;
    var current: [3]u32 = undefined;
    var old_rows: [12]u16 = undefined;
    var current_rows: [12]u16 = undefined;
    _ = try run(true, std.testing.io, &session, &sampler, .old, &.{ 1, 2 }, &old, &old_rows);
    _ = try run(true, std.testing.io, &session, &sampler, .current, &.{ 1, 2 }, &current, &current_rows);
    try std.testing.expectEqualSlices(u32, &old, &current);
    try std.testing.expectEqualSlices(u16, &old_rows, &current_rows);
    for (0..3) |i| try std.testing.expectEqualSlices(u16, &session.logits, old_rows[i * 4 ..][0..4]);
}

test "fixed work invokes exactly N draws at the positions consumed by the serving adapter" {
    const CountingSampler = struct {
        values: [4]f64 = @splat(0),
        positions: [128]usize = undefined,
        draws: usize = 0,

        fn draw(self: *@This(), _: Arm, _: []const u16, position: usize) !u32 {
            self.positions[self.draws] = position;
            self.draws += 1;
            return 3;
        }
    };
    const prompt: [275]u32 = @splat(1);
    var output: [128]u32 = undefined;
    for ([_][2]usize{ .{ 64, 128 }, .{ 275, 27 }, .{ 1, 1 } }) |shape| {
        var session: FakeSession = .{};
        var sampler: CountingSampler = .{};
        _ = try run(false, std.testing.io, &session, &sampler, .old, prompt[0..shape[0]], output[0..shape[1]], &.{});
        try std.testing.expectEqual(shape[1], sampler.draws);
        for (sampler.positions[0..sampler.draws], 0..) |position, i|
            try std.testing.expectEqual(shape[0] + i, position);
        try std.testing.expectEqual(shape[0] + shape[1] - 1, session.appends);
    }
}

test "invalid turn is rejected before session reset or append" {
    var sampler = try Sampler.init(std.testing.allocator, 4);
    defer sampler.deinit();
    var session: FakeSession = .{};
    var output: [1]u32 = undefined;
    try std.testing.expectError(error.InvalidTurn, run(false, std.testing.io, &session, &sampler, .old, &.{}, &output, &.{}));
    try std.testing.expectError(error.InvalidTurn, run(false, std.testing.io, &session, &sampler, .old, &.{1}, &.{}, &.{}));
    try std.testing.expectError(error.InvalidToken, run(false, std.testing.io, &session, &sampler, .old, &.{4}, &output, &.{}));
    try std.testing.expectEqual(@as(usize, 0), session.resets);
    try std.testing.expectEqual(@as(usize, 0), session.appends);
}

test "input parsing rejects malformed types and missing fields without any device dependency" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    try std.testing.expectError(error.MissingField, parse(arena.allocator(), "{}"));
    try std.testing.expectError(error.UnexpectedToken, parse(arena.allocator(), "{\"context\":false}"));
}

test "benchmark admission rejects wrong workload topology physical devices and sampling before CUDA" {
    const short_prompt: [64]u32 = @splat(1);
    const long_prompt: [275]u32 = @splat(1);
    const cases = [_]Case{
        .{ .name = "decode-heavy", .prompt = &short_prompt, .max_tokens = 128 },
        .{ .name = "prefill-heavy", .prompt = &long_prompt, .max_tokens = 27 },
    };
    const valid: Input = .{
        .context = 4096,
        .devices = &.{ 0, 1 },
        .expected_device_uuids = &expected_uuids,
        .cut = 20,
        .cases = &cases,
        .provenance = .{
            .scope = "synthetic",
            .tokenizer_sha256 = "1111111111111111111111111111111111111111111111111111111111111111",
            .description_sha256 = "2222222222222222222222222222222222222222222222222222222222222222",
            .description = "provenance.json",
            .seed = 0,
            .temperature = 0,
        },
    };
    try validate(valid);
    var invalid = valid;
    invalid.context = 4095;
    try std.testing.expectError(error.InvalidBenchmarkInput, validate(invalid));
    invalid = valid;
    invalid.cut = 19;
    try std.testing.expectError(error.InvalidBenchmarkInput, validate(invalid));
    invalid = valid;
    invalid.devices = &.{ 1, 0 };
    try std.testing.expectError(error.InvalidBenchmarkInput, validate(invalid));
    invalid = valid;
    invalid.expected_device_uuids = &.{ expected_uuids[1], expected_uuids[0] };
    try std.testing.expectError(error.InvalidBenchmarkInput, validate(invalid));
    invalid = valid;
    invalid.provenance.temperature = 1;
    try std.testing.expectError(error.InvalidBenchmarkInput, validate(invalid));
    invalid = valid;
    invalid.provenance.seed = 1;
    try std.testing.expectError(error.InvalidBenchmarkInput, validate(invalid));
    invalid = valid;
    invalid.provenance.tokenizer_sha256 = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx";
    try std.testing.expectError(error.InvalidBenchmarkInput, validate(invalid));
    var wrong_cases = cases;
    wrong_cases[0].max_tokens = 127;
    invalid = valid;
    invalid.cases = &wrong_cases;
    try std.testing.expectError(error.InvalidBenchmarkInput, validate(invalid));
    wrong_cases = cases;
    wrong_cases[1].prompt = long_prompt[0..274];
    try std.testing.expectError(error.InvalidBenchmarkInput, validate(invalid));
    var bad_token_prompt = short_prompt;
    bad_token_prompt[63] = vocab;
    wrong_cases = cases;
    wrong_cases[0].prompt = &bad_token_prompt;
    try std.testing.expectError(error.InvalidToken, validate(invalid));
}
