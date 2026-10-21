//! Qwen3.5's hybrids on CUDA: qlinear projections, qwen35.cu glue, one captured forward over every stream's window.
//! Local modifications: explicit dense BF16 hidden-only initialization and request-owned normalized rows.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const Round = @import("cuda_round.zig");
const geometry = @import("cuda_geometry.zig");
const weights = @import("cuda_weights.zig");
const caches = @import("cuda_cache.zig");
const routed = @import("cuda_routed.zig");
const checkpoint = @import("cuda_checkpoint.zig");
const modelopt = @import("cuda_modelopt.zig");
const hf = @import("cuda_hf.zig");
const bf16_manifest = @import("cuda_bf16_checkpoint.zig");
const bf16_weights = @import("cuda_bf16_weights.zig");
const bf16_bindings = @import("cuda_bf16.zig");
pub const hidden = @import("cuda_hidden.zig");
const hooks = @import("cuda_learn_hooks.zig");
const pocket = @import("cuda_pocket.zig");
const profile = @import("cuda_startup_profile.zig");

pub const Geometry = geometry.Geometry;
pub const MoeMath = hf.MoeMath;
pub const AttentionMode = hf.AttentionMode;
pub const Cache = caches.Cache;
pub const tap_rows = caches.tap_rows;
const Seg = caches.Seg;
const DeviceBuffer = cuda.DeviceBuffer;
const head_dim = geometry.head_dim;

/// Rows of one forward at most: a prompt chunk, or a shared round's windows.
pub const max_rows = 128;
/// Streams' windows in one forward at most.
pub const max_segs = Round.max_segs;
/// Rows of a verify forward at most: each row's logits are kept and drawn.
pub const most_verify = Round.most_verify;
/// tf_q35_argmax_slices' blocks a row (ARGMAX_SLICE entries each).
fn argmaxSlices(vocab: usize) usize {
    return (vocab + 4095) / 4096;
}
/// A window's rows at most (exact: one-row steps' bits; qwen35.cuh's MAXW, the rows a stream's log holds).
pub const most_window = Round.most_window;
/// Experimental committed HF prefixes retain the decode operators' row arithmetic.
pub const hf_prefill_rows = 16;

fn hfOrder(numerics: checkpoint.Numerics) bool {
    return numerics == .hf_modelopt or numerics == .hf_dense_bf16;
}

pub fn streamStateCopies(numerics: checkpoint.Numerics) usize {
    return if (numerics == .hf_dense_bf16) 1 else 2;
}

pub fn streamLogRows(numerics: checkpoint.Numerics) usize {
    return if (hfOrder(numerics)) 1 else most_window;
}

pub fn validateMoeMath(admitted: checkpoint.Admitted, math: MoeMath) !void {
    if (math == .legacy) return;
    const g = admitted.geometry;
    if (admitted.numerics() != .hf_modelopt or g.hidden != 2048 or g.intermediate != 512 or g.experts != 256 or g.top_k != 8)
        return error.UnsupportedQwenMoeMath;
    if (admitted.encoding.modelopt.spec.shared_width != 512) return error.UnsupportedQwenMoeMath;
}

fn validateMoeMathShape(configured: MoeMath, requested: MoeMath) !void {
    if (configured != requested) return error.QwenMoeMathShapeMismatch;
}

/// ModelOpt's serial reference attention reads BF16 rows only; e4m3 rows take the chunked partial.
pub fn validateAttentionMode(admitted: checkpoint.Admitted, mode: AttentionMode, kv8: bool) !void {
    if (admitted.numerics() != .hf_modelopt) {
        if (mode != .reference) return error.UnsupportedQwenAttentionMode;
        return;
    }
    if (mode == .reference and kv8) return error.QwenReferenceAttentionNeedsBf16Kv;
}

fn validateAttentionModeShape(configured: AttentionMode, requested: AttentionMode) !void {
    if (configured != requested) return error.QwenAttentionModeShapeMismatch;
}
const hf_part_bytes = 8 << 20;
pub const ExpertGraphNode = struct {
    shape_key: u32,
    total: u8,
    heads_all: bool,
    layer_local: usize,
    layer_global: usize,
    arm: enum { baseline, n8, split_m, slice_major },
    up_layout: cuda.experts.Nvfp4UpLayout = .lane_major,
    grid: [3]u32,
    block: [3]u32,
    shared_bytes: u32,
    abi_count: usize,
    argument_bytes: usize,
    args_sha256: [64]u8,
};
/// Partial attention chunks a row at most (qwen35.cuh's MAXCH): each row's chunk width grows with its own keys.
const max_chunks = 64;

/// Device allocations are 256-byte aligned, as the driver's are.
pub fn aligned(n: usize) usize {
    return std.mem.alignForward(usize, n, 256);
}

const TerminalOutput = struct {
    logits_bytes: usize,
    pick_at: usize,
    bytes: usize,

    fn init(heads: usize, vocab: usize) TerminalOutput {
        const bytes = heads * vocab * 2;
        const pick_at = std.mem.alignForward(usize, bytes, @alignOf(cuda.tree_round.Pick));
        return .{ .logits_bytes = bytes, .pick_at = pick_at, .bytes = pick_at + @sizeOf(cuda.tree_round.Pick) };
    }

    fn token(output: TerminalOutput, host: []const u8, vocab: usize) !?u32 {
        const pick = std.mem.bytesToValue(cuda.tree_round.Pick, host[output.pick_at..][0..@sizeOf(cuda.tree_round.Pick)]);
        if (pick.nonfinite & ~@as(u32, 7) != 0) return error.InvalidTerminalPick;
        if (pick.nonfinite != 0) {
            if (pick.token != std.math.maxInt(u32)) return error.InvalidTerminalPick;
            return null;
        }
        if (pick.token >= vocab) return error.InvalidTerminalPick;
        return pick.token;
    }
};

test "BF16 initialization owns mappings scratch and bindings through every allocation failure" {
    try @import("cuda_bf16_init_test.zig").exercise(Engine.initBf16WithBinder);
}

test "BF16 constructor and owned runtime bodies are compiled on the host" {
    std.mem.doNotOptimizeAway(&Engine.initBf16);
    std.mem.doNotOptimizeAway(&Engine.initBf16Inspected);
    std.mem.doNotOptimizeAway(&Engine.initBf16WithBinder);
    std.mem.doNotOptimizeAway(&Engine.gatherInput);
    std.mem.doNotOptimizeAway(&Engine.gatherOutputRows);
    std.mem.doNotOptimizeAway(&Engine.deinit);
    std.mem.doNotOptimizeAway(&Engine.warmCache);
    std.mem.doNotOptimizeAway(&Engine.warmupPromptRows);
    std.mem.doNotOptimizeAway(&Engine.body);
    std.mem.doNotOptimizeAway(&Engine.deltaScan);
}

test "BF16 budgeting and scratch use pinned dense widths without a vocabulary allocation" {
    const g = try @import("cuda_bf16_config.zig").parse(std.testing.allocator, @embedFile("fixtures/clef_flash/config.json"));
    var plan: bf16_manifest.Plan = undefined;
    plan.geometry = g;
    plan.budget = .{ .device_weights = 13_879_009_280, .device_load_stage = 100_663_296, .host_tables = 4_068_474_880, .host_conversion_peak = 201_326_592 };
    const budget = try Engine.bf16Budget(&plan, 4096);
    try std.testing.expectEqual(@as(usize, 28_558_336), budget.scratch_bytes - aligned(Engine.tables_len));
    try std.testing.expectEqual(@as(usize, 28_558_336) + aligned(Engine.tables_len), budget.scratch_bytes);
    try std.testing.expectEqual(@as(usize, 13_879_009_280) + budget.scratch_bytes + 187_403_264 + 33_554_432, budget.steady_device_bytes);
    try std.testing.expectEqual(@as(usize, 187_403_264), budget.cache_bytes);
    try std.testing.expectEqual(@as(usize, 33_554_432), budget.hidden_bytes);
    try std.testing.expectEqual(@as(usize, 131_072), budget.host_input_bytes);
    try std.testing.expectEqual(@as(usize, 4_068_474_880), budget.host_tables_bytes);
    try std.testing.expectEqual(budget.weights_bytes + budget.scratch_bytes + budget.cache_bytes + budget.hidden_bytes, budget.steady_device_bytes);
    try std.testing.expectError(error.InvalidQwenHiddenShape, Engine.bf16Budget(&plan, 0));
    try std.testing.expectError(error.InvalidQwenHiddenShape, Engine.bf16Budget(&plan, 4097));
    plan.budget.device_weights = std.math.maxInt(usize);
    try std.testing.expectError(error.Overflow, Engine.bf16Budget(&plan, 4096));

    var e: Engine = undefined;
    e.g = g;
    e.numerics = .hf_dense_bf16;
    e.arena = .{ .d = undefined, .ptr = 4096, .len = 28_558_336, .borrowed = true };
    e.arena_at = 0;
    e.bf16 = .{ .weights = undefined, .bindings = undefined, .input = undefined };
    try std.testing.expectEqual(@as(usize, 28_558_336), try e.scratch(.allocate));
    try std.testing.expectEqual(e.arena.len, e.arena_at);
    try std.testing.expectEqual(@as(u64, 0), e.logits);
    try std.testing.expectEqual(@as(u64, 16 * 4096 * 2), e.x - e.h);
    try std.testing.expectEqual(@as(u64, 16 * 4096 * 4), e.mix - e.y);
    try std.testing.expectEqual(@as(u64, 16 * 16 * 128 * 4), e.k - e.q);
    try std.testing.expectEqual(@as(u64, 16 * 8192 * 2), e.gate - e.qkv);
    try std.testing.expectEqual(@as(u64, 16 * 32 * 2), e.b - e.a);
    try std.testing.expectEqual(@as(u64, 16 * 32 * 4), e.qraw - e.g_);
    try std.testing.expectEqual(e.arena.ptr + e.arena.len, e.bf16.?.bindings.part + hf_part_bytes);
}

test "terminal output layout preserves row stride and aligns a contiguous Pick suffix" {
    for ([_]usize{ 1, 3, 257, 248320 }) |vocab| {
        const capacity = TerminalOutput.init(most_verify, vocab).bytes;
        for (1..most_verify + 1) |heads| {
            const output = TerminalOutput.init(heads, vocab);
            try std.testing.expectEqual(heads * vocab * 2, output.logits_bytes);
            try std.testing.expectEqual(@as(usize, 0), output.pick_at % @alignOf(cuda.tree_round.Pick));
            try std.testing.expect(output.pick_at >= output.logits_bytes);
            try std.testing.expect(output.pick_at - output.logits_bytes <= 2);
            try std.testing.expect(output.bytes <= capacity);
        }
    }
    try std.testing.expectEqual(@as(usize, 496648), TerminalOutput.init(1, 248320).bytes);
    try std.testing.expectEqual(@as(usize, 7946248), TerminalOutput.init(16, 248320).bytes);
}

test "terminal Pick validation preserves nonfinite fallback and rejects malformed results" {
    var bytes: [16]u8 = @splat(0);
    const output = TerminalOutput.init(1, 3);
    for ([_]cuda.tree_round.Pick{ .{ .token = 2, .nonfinite = 0 }, .{ .token = 0xffffffff, .nonfinite = 1 }, .{ .token = 0xffffffff, .nonfinite = 7 } }) |pick| {
        @memcpy(bytes[output.pick_at..][0..8], std.mem.asBytes(&pick));
        const token = try output.token(&bytes, 3);
        if (pick.nonfinite == 0) try std.testing.expectEqual(@as(?u32, 2), token) else try std.testing.expect(token == null);
    }
    for ([_]cuda.tree_round.Pick{ .{ .token = 3, .nonfinite = 0 }, .{ .token = 0xffffffff, .nonfinite = 0 }, .{ .token = 1, .nonfinite = 1 }, .{ .token = 0xffffffff, .nonfinite = 8 } }) |pick| {
        @memcpy(bytes[output.pick_at..][0..8], std.mem.asBytes(&pick));
        try std.testing.expectError(error.InvalidTerminalPick, output.token(&bytes, 3));
    }
}

test "hidden terminal graph identity cannot alias vocabulary or greedy terminals" {
    for ([_]usize{ 1, 2, 16, 128 }) |rows| for ([_]bool{ false, true }) |all| for ([_]bool{ false, true }) |greedy| {
        var s: Engine.Shape = .{ .total = @intCast(rows), .nseg = 1, .record = false, .prompt = true, .taps = false, .heads_all = all, .terminal_greedy = greedy };
        const before = s.key();
        s.normalized_hidden = true;
        try std.testing.expectEqual(@as(u32, 1 << 28), before ^ s.key());
    };
}

test "terminal graph key occupies bit 25 without changing prior shape dimensions" {
    for (1..17) |rows| for ([_]bool{ false, true }) |heads_all| {
        var shape: Engine.Shape = .{ .total = @intCast(rows), .nseg = 1, .record = false, .prompt = true, .taps = false, .heads_all = heads_all, .expert_up_layout = .slice_major_v1, .expert_up_split_m = true, .reuse_fp8_quant = true, .expert_up_n8 = true, .gdn_layout = .key_value };
        const before = shape.key();
        shape.terminal_greedy = true;
        try std.testing.expectEqual(@as(u32, 1 << 25), before ^ shape.key());
    };
    try std.testing.expect(!(Engine.InitOptions{}).terminal_greedy);
}

test "terminal finish uses one extended copy and publishes finite picks only at commit" {
    const Fake = struct {
        var copies: usize = 0;
        var syncs: usize = 0;
        var size: usize = 0;
        var fail_sync: bool = false;
        var fail_copy: bool = false;
        fn sync(_: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
            syncs += 1;
            return if (fail_sync) cuda.abi.error_not_ready else cuda.abi.success;
        }
        fn copy(out: ?*anyopaque, source: cuda.abi.DevicePtr, bytes: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
            copies += 1;
            size = bytes;
            if (fail_copy) return cuda.abi.error_not_ready;
            const dst: [*]u8 = @ptrCast(out.?);
            const src: [*]const u8 = @ptrFromInt(source);
            @memcpy(dst[0..bytes], src[0..bytes]);
            return cuda.abi.success;
        }
    };
    var d: cuda.Driver = undefined;
    d.api.cuStreamSynchronize = Fake.sync;
    d.api.cuMemcpyDtoHAsync_v2 = Fake.copy;
    var device: [112]u8 align(16) = @splat(0xa5);
    var host: [112]u8 align(16) = @splat(0x5a);
    var terminal: [3]u16 = @splat(99);
    var cache: Cache = undefined;
    cache.logits = &terminal;
    var e: Engine = undefined;
    e.d = &d;
    e.stream = .{ .d = &d, .handle = null };
    e.placement = .{ .first = 0, .end = 1, .total = 1 };
    e.g.vocab = 3;
    e.numerics = .hf_modelopt;
    e.logits = @intFromPtr(&device);
    e.host_logits = .{ .d = &d, .bytes = &host };
    for ([_]usize{ 1, 16 }) |rows| for ([_]bool{ false, true }) |all| {
        const heads = if (all) rows else 1;
        const output = TerminalOutput.init(heads, 3);
        @memset(&device, 0xa5);
        @memset(&host, 0x5a);
        for (std.mem.bytesAsSlice(u16, device[0..output.logits_bytes]), 0..) |*word, i| word.* = @intCast(i + 1);
        const pick: cuda.tree_round.Pick = .{ .token = 2, .nonfinite = 0 };
        @memcpy(device[output.pick_at..][0..8], std.mem.asBytes(&pick));
        cache.len = 7;
        cache.terminal_greedy = .{ .position = 7, .token = 0 };
        Fake.copies = 0;
        Fake.syncs = 0;
        Fake.fail_sync = false;
        Fake.fail_copy = false;
        const parts = [_]Part{.{ .cache = &cache, .rows = rows }};
        const shape: Engine.Shape = .{ .total = @intCast(rows), .nseg = 1, .record = false, .prompt = true, .taps = false, .heads_all = all, .terminal_greedy = true };
        const keep: Keep = if (all) .prompt_logits else .last;
        try e.finish(&parts, keep, shape);
        try std.testing.expectEqual(@as(usize, 1), Fake.copies);
        try std.testing.expectEqual(@as(usize, 1), Fake.syncs);
        try std.testing.expectEqual(output.bytes, Fake.size);
        try std.testing.expectEqualSlices(u8, device[0..output.logits_bytes], host[0..output.logits_bytes]);
        for (host[output.bytes..]) |byte| try std.testing.expectEqual(@as(u8, 0x5a), byte);
        try std.testing.expectEqualSlices(u16, try e.rowLogits(heads - 1), &terminal);
        try std.testing.expectEqual(@as(usize, 7), cache.len);
        try std.testing.expectEqual(7 + rows, cache.terminal_greedy.?.position);
        Engine.commit(&parts, false);
        try std.testing.expectEqual(cache.len, cache.terminal_greedy.?.position);

        const bad: cuda.tree_round.Pick = .{ .token = 0, .nonfinite = 1 };
        @memcpy(device[output.pick_at..][0..8], std.mem.asBytes(&bad));
        terminal = @splat(99);
        try std.testing.expectError(error.InvalidTerminalPick, e.finish(&parts, keep, shape));
        try std.testing.expect(cache.terminal_greedy == null);
        try std.testing.expectEqualSlices(u16, &.{ 99, 99, 99 }, &terminal);

        const nonfinite: cuda.tree_round.Pick = .{ .token = 0xffffffff, .nonfinite = 7 };
        @memcpy(device[output.pick_at..][0..8], std.mem.asBytes(&nonfinite));
        try e.finish(&parts, keep, shape);
        try std.testing.expect(cache.terminal_greedy == null);
        try std.testing.expectEqualSlices(u16, try e.rowLogits(heads - 1), &terminal);

        var disabled = shape;
        disabled.terminal_greedy = false;
        cache.terminal_greedy = .{ .position = cache.len, .token = 1 };
        try e.finish(&parts, keep, disabled);
        try std.testing.expectEqual(output.logits_bytes, Fake.size);
        try std.testing.expect(cache.terminal_greedy == null);
        try std.testing.expectEqualSlices(u16, try e.rowLogits(heads - 1), &terminal);

        cache.terminal_greedy = .{ .position = cache.len, .token = 1 };
        Fake.fail_copy = true;
        Fake.syncs = 0;
        try std.testing.expectError(error.NotReady, e.finish(&parts, keep, shape));
        try std.testing.expect(cache.terminal_greedy == null);
        try std.testing.expectEqual(@as(usize, 0), Fake.syncs);
        Fake.fail_copy = false;

        cache.terminal_greedy = .{ .position = cache.len, .token = 1 };
        Fake.fail_sync = true;
        try std.testing.expectError(error.NotReady, e.finish(&parts, keep, shape));
        try std.testing.expect(cache.terminal_greedy == null);
    };
    Fake.fail_sync = false;
}

test "terminal argmax launch uses the shared function and terminal row with exact ABI" {
    const Fake = struct {
        var calls: usize = 0;
        var row: u64 = 0;
        var pick: u64 = 0;
        fn launch(f: cuda.abi.Function, gx: c_uint, gy: c_uint, gz: c_uint, bx: c_uint, by: c_uint, bz: c_uint, shared: c_uint, _: cuda.abi.Stream, params: ?[*]?*anyopaque, extra: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
            if (f != @as(cuda.abi.Function, @ptrFromInt(123)) or gx != 1 or gy != 1 or gz != 1 or bx != 256 or by != 1 or bz != 1 or shared != 0 or extra != null) return 1;
            const args = params orelse return 1;
            const input: *const u64 = @ptrCast(@alignCast(args[0].?));
            const output: *const u64 = @ptrCast(@alignCast(args[1].?));
            const vocab: *const u32 = @ptrCast(@alignCast(args[2].?));
            const stride: *const u32 = @ptrCast(@alignCast(args[3].?));
            if (input.* != row or output.* != pick or vocab.* != 3 or stride.* != 3) return 1;
            calls += 1;
            return cuda.abi.success;
        }
    };
    var d: cuda.Driver = undefined;
    d.api.cuLaunchKernel = Fake.launch;
    var e: Engine = undefined;
    e.g.vocab = 3;
    e.logits = 4096;
    e.stream = .{ .d = &d, .handle = null };
    e.terminal = .{ .module = undefined, .kernels = undefined };
    e.terminal.?.kernels.argmax_fn = .{ .d = &d, .handle = @ptrFromInt(123) };
    Fake.calls = 0;
    for ([_]usize{ 1, 16 }) |heads| {
        Fake.row = 4096 + (heads - 1) * 6;
        Fake.pick = 4096 + TerminalOutput.init(heads, 3).pick_at;
        try e.pickTerminal(heads);
    }
    try std.testing.expectEqual(@as(usize, 2), Fake.calls);
}

/// qwen35.cuh's Geo.
pub const Geo = extern struct { hid: c_int, nk: c_int, nv: c_int, qh: c_int, kvh: c_int, nl: c_int };

const layers = @import("cuda_layers.zig");
pub const Proj = layers.Proj;
pub const Delta = layers.Delta;
pub const Attention = layers.Attention;
pub const Layer = layers.Layer;
pub const Mlp = layers.Mlp;
const Fns = layers.Fns;
pub const Keeps = layers.Keeps;
const Kv8 = layers.Kv8;

/// A window of one stream in a forward: `parents`, for a tree, names each row's parent row (-1: the first row).
pub const Part = struct { cache: *Cache, rows: usize, parents: ?[]const i32 = null, images: ?*const Images = null };

/// An image prompt's rows for its prompt chunks: device feature rows, each prompt row's feature row, its positions.
pub const Images = struct {
    features: u64, // bf16 [image rows, hidden]
    src: []const i32, // [prompt]: a feature row, or -1 for text
    pos: []const i32, // [3][prompt]: t, h and w rotary positions (get_rope_index)
};

test "GDN layouts have distinct graph keys for identical forward geometry" {
    const baseline: Engine.Shape = .{ .total = 1, .nseg = 1, .record = false, .prompt = true, .taps = false };
    var candidate = baseline;
    candidate.gdn_layout = .key_value;
    try std.testing.expectEqual(@as(u32, 1 << 19), baseline.key() ^ candidate.key());
}

test "expert up choices have distinct graph keys independent of GDN layout" {
    const baseline: Engine.Shape = .{ .total = 1, .nseg = 1, .record = false, .prompt = true, .taps = false, .gdn_layout = .key_value };
    var candidate = baseline;
    candidate.expert_up_n8 = true;
    try std.testing.expectEqual(@as(u32, 1 << 20), baseline.key() ^ candidate.key());
    candidate.gdn_layout = .value_key;
    try std.testing.expectEqual(@as(u32, (1 << 20) | (1 << 19)), baseline.key() ^ candidate.key());
}

test "N8 up rejects MLX before CUDA or graph lookup" {
    var e: Engine = undefined;
    e.numerics = .mlx_sanitized;
    const shape: Engine.Shape = .{ .total = 1, .nseg = 1, .record = false, .prompt = true, .taps = false, .expert_up_n8 = true };
    try std.testing.expectError(error.UnsupportedQwenExpertUp, e.execute(shape));
    var cache: Cache = undefined;
    cache.gdn_layout = .value_key;
    e.expert_up_n8 = true;
    try std.testing.expectError(error.UnsupportedQwenExpertUp, e.prepare(&.{.{ .cache = &cache, .rows = 1 }}, &.{0}, .last));
}

test "mixed GDN layouts fail before settling caches or touching CUDA" {
    var e: Engine = undefined;
    var baseline: Cache = undefined;
    baseline.gdn_layout = .value_key;
    var candidate: Cache = undefined;
    candidate.gdn_layout = .key_value;
    const parts = [_]Part{ .{ .cache = &baseline, .rows = 1 }, .{ .cache = &candidate, .rows = 1 } };
    try std.testing.expectError(error.MixedQwenGdnLayouts, e.prepare(&parts, &.{ 0, 0 }, .last));
}

/// What a forward keeps: a prompt chunk its last row's logits, a verify window every row's draw or logits and log.
pub const Keep = union(enum) { last, draws: []const Draw, logits, prompt_logits, hidden: hidden.Window, learn };

fn validateCombined(numerics: checkpoint.Numerics, n8: bool, split: bool, reuse: bool) !void {
    if (split and (numerics != .hf_modelopt or !n8)) return error.UnsupportedQwenExpertUp;
    if (reuse and numerics != .hf_modelopt) return error.UnsupportedQwenQuantReuse;
}

fn records(keep: Keep) bool {
    return keep == .draws or keep == .logits or keep == .learn;
}

fn validateHfWindow(numerics: checkpoint.Numerics, bounded: bool, parts: []const Part, ids: usize, keep: Keep) !void {
    if (keep == .prompt_logits and numerics != .hf_modelopt) return error.UnsupportedQwenWindow;
    if (!hfOrder(numerics)) return;
    const limit: usize = if (bounded) hf_prefill_rows else 1;
    if (parts.len != 1 or ids == 0 or ids > limit or parts[0].rows != ids or
        (keep != .last and keep != .hidden and keep != .learn and !(bounded and keep == .prompt_logits))) return error.UnsupportedQwenWindow;
    if (parts[0].parents != null or parts[0].images != null) return error.UnsupportedQwenWindow;
}

fn hfInputWidth(g: Geometry) usize {
    return @max(g.convWidth(), g.valueWidth(), g.attnWidth(), g.hidden, g.intermediate);
}

fn chainView(base: Seg, row: usize) Seg {
    var view = base;
    view.row0 = base.row0 + @as(c_int, @intCast(row));
    view.len = base.len + @as(c_int, @intCast(row));
    view.rows = 1;
    return view;
}

fn validateChainViews(base: Seg, views: []const Seg) !void {
    if (base.row0 != 0 or base.rows != views.len or views.len == 0 or views.len > hf_prefill_rows) return error.QwenGdnTableMismatch;
    for (views, 0..) |view, row| {
        const want = chainView(base, row);
        inline for (@typeInfo(Seg).@"struct".field_names) |name| {
            if (@field(view, name) != @field(want, name)) return error.QwenGdnTableMismatch;
        }
    }
}

const GdnRoster = struct {
    layers: usize,
    rows: usize,
    count: usize = 0,

    fn accept(r: *GdnRoster, layer: usize, row: usize) !void {
        if (r.rows == 0 or r.count >= r.layers * r.rows or layer != r.count / r.rows or row != r.count % r.rows) return error.QwenGdnGraphLayerMismatch;
        r.count += 1;
    }

    fn finish(r: GdnRoster) !void {
        if (r.count != r.layers * r.rows) return error.QwenGdnGraphWorkCountMismatch;
    }
};

/// A kept row's draw on the device: sample.cu's keyed rule at `position`, or greedy without one.
pub const Draw = struct { rule: ?cuda.sample.Rule, position: u64 };

pub const Phase = enum { begin, embedding, input_norm, mixer_inputs, gdn_pre, conv_commit, gdn_chain, gdn_post, attention, mixer_output, post_norm, route, expert_plan, expert_up, expert_down, shared, combine, dense, head };
pub const Observer = struct { ptr: *anyopaque, mark: *const fn (*anyopaque, Phase, usize) anyerror!void };

test "terminal capture verification is opt in" {
    const info = @typeInfo(Engine).@"struct";
    inline for (info.field_names, info.field_types, info.field_attrs) |name, Field, attrs| {
        if (comptime std.mem.eql(u8, name, "verify_terminal_graphs")) {
            try std.testing.expect(!attrs.defaultValue(Field).?);
            return;
        }
    }
    @compileError("terminal verifier flag missing");
}

pub const Engine = struct {
    startup: profile.Times = .{},
    gpa: std.mem.Allocator,
    dir: []const u8,
    pockets: pocket.Pockets,
    learning: ?*const hooks.Keep = null,
    mlp: ?hooks.MlpAdd = null,
    taught: u64 = 0,
    d: *const cuda.Driver,
    g: Geometry,
    placement: checkpoint.Placement,
    numerics: checkpoint.Numerics,
    modelopt: ?modelopt.Bindings = null,
    bf16: ?struct {
        weights: bf16_weights.Loaded,
        bindings: bf16_bindings.Bindings,
        input: cuda.copy_source.Buffer,
    } = null,
    captured: bool = true,
    graph_nodes: usize = 0,
    graph_launches: usize = 0,
    expert_up_layout: cuda.experts.Nvfp4UpLayout = .lane_major,
    expert_up_n8: bool = false,
    expert_up_split_m: bool = false,
    reuse_fp8_quant: bool = false,
    bounded_hf_prefill: bool = false,
    attention_mode: AttentionMode = .reference,
    terminal: ?struct { module: cuda.Module, kernels: cuda.tree_round.Kernels } = null,
    verify_terminal_graphs: bool = false,
    terminal_graph_roster: std.ArrayList(TerminalGraphRecord) = .empty,
    verify_expert_graphs: bool = false,
    expert_graph_nodes_baseline: usize = 0,
    expert_graph_nodes_n8: usize = 0,
    expert_graph_nodes_split_m: usize = 0,
    expert_graph_roster: std.ArrayList(ExpertGraphNode) = .empty,
    quant_graph_nodes_baseline: usize = 0,
    quant_graph_nodes_reuse: usize = 0,
    quant_graphs_verified: usize = 0,
    verify_gdn_graphs: bool = false,
    gdn_graph_nodes_baseline: usize = 0,
    gdn_graph_nodes_candidate: usize = 0,
    observer: ?Observer = null,
    observed_layer: usize = 0,
    geo: Geo,
    context: cuda.abi.Context,
    stream: cuda.Stream,
    mods: [16]cuda.Module,
    kv8: bool, // attention caches as e4m3 rows with a scale each
    fp8: Kv8,
    draws: cuda.sample.Draws,
    affine: cuda.qlinear.Affine4Kernels,
    wide: cuda.qlinear.AffineNKernels, // the tiles of every width but 4, and their packs
    experts: cuda.experts.Affine4Experts,
    experts3: cuda.experts3.Affine3Experts,
    plan: cuda.grouped.Router,
    moe: cuda.moe.Fns,
    fns: Fns,
    keeps: Keeps,
    /// Every weight and scratch buffer in one allocation: WDDM's launch cost grows with the allocation count.
    arena: DeviceBuffer,
    arena_at: usize = 0,
    /// Rows a verify forward takes at most, and a window's rows at most.
    verify_rows: usize,
    window_rows: usize,
    // weights
    embed_w: u64,
    embed_s: u64,
    embed_b: u64,
    embed_bits: u8, // the embedding's rows' width: 4 or 8
    terminal_weights: layers.TerminalWeights = .none,
    final_norm: u64,
    layers: []Layer,
    // scratch, max_rows rows each
    h: u64,
    x: u64,
    r: u64,
    xs: u64,
    qkv: u64,
    z: u64,
    a: u64,
    b: u64,
    q: u64,
    k: u64,
    v: u64,
    g_: u64,
    beta: u64,
    y: u64,
    mix: u64,
    qraw: u64,
    kraw: u64,
    vraw: u64,
    qn: u64,
    kn: u64,
    gate: u64,
    up: u64,
    act: u64,
    pm: u64,
    pl: u64,
    po: u64,
    logits: u64,
    routed: routed.Scratch = .{}, // qwen3_5_moe's routed rows
    amax: u64, // the argmax slices' bests: most_verify rows of argmaxSlices(vocab) (f32, i32)
    // per-forward tables (ids, segments, rows' segment, parent and ancestors, the draws' rules) staged in pinned memory
    tables: DeviceBuffer,
    staged: cuda.HostBuffer,
    host_logits: cuda.HostBuffer,
    /// The kept rows' device draws, when forward was given them.
    tokens: [most_verify]u32 = undefined,
    /// The captured forwards by shape (`Shape.key`): the launches from the embedding to the head.
    graphs: std.AutoHashMapUnmanaged(u32, cuda.graph.Exec) = .empty,
    /// A drafter's tapped layers (`setTaps`): each row's residual after them lands in its stream's tap ring.
    taps: [8]u32 = undefined,
    n_taps: usize = 0,
    /// During an image prompt chunk's forward: its feature rows (0: none); the chunk's src and positions are staged.
    features: u64 = 0,

    pub const ids_at = Round.ids_at;
    pub const segs_at = Round.segs_at;
    pub const rowseg_at = Round.rowseg_at;
    pub const rowpar_at = Round.rowpar_at;
    pub const rowanc_at = Round.rowanc_at;
    const rules_at = Round.rules_at;
    const meta_at = Round.meta_at;
    const tokens_at = Round.tokens_at;
    pub const params_at = Round.params_at;
    pub const chain_segs_at = aligned(params_at + 8 * 4);
    pub const src_at = chain_segs_at + hf_prefill_rows * @sizeOf(Seg);
    pub const mpos_at = src_at + max_rows * 4;
    pub const tables_len = mpos_at + 3 * max_rows * 4;

    pub fn init(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, kv8: bool) !*Engine {
        const g = try Geometry.read(gpa, io, dir);
        return initPlaced(gpa, io, ctx, dir, .{ .geometry = g, .encoding = .mlx }, .{ .first = 0, .end = g.layers, .total = g.layers }, kv8);
    }

    pub const InitOptions = struct { expert_up_layout: cuda.experts.Nvfp4UpLayout = .lane_major, terminal_greedy: bool = false, moe_math: MoeMath = .legacy, attention_mode: AttentionMode = .reference };

    pub fn initPlaced(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, admitted: checkpoint.Admitted, placement: checkpoint.Placement, kv8: bool) !*Engine {
        return initPlacedWithOptions(gpa, io, ctx, dir, admitted, placement, kv8, .{});
    }

    pub fn initPlacedWithOptions(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, admitted: checkpoint.Admitted, placement: checkpoint.Placement, kv8: bool, options: InitOptions) !*Engine {
        try admitted.requireExecutable();
        try validateMoeMath(admitted, options.moe_math);
        try validateAttentionMode(admitted, options.attention_mode, kv8);
        const g = try placement.localGeometry(admitted.geometry);
        if (options.terminal_greedy and admitted.numerics() != .hf_modelopt) return error.UnsupportedQwenTerminalGreedy;
        if (options.expert_up_layout == .slice_major_v1 and (admitted.numerics() != .hf_modelopt or g.hidden != 2048 or g.intermediate != 512 or g.experts != 256)) return error.UnsupportedQwenExpertUpLayout;
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        const started = profile.now(io);
        var ck = try core.Checkpoint.openModel(gpa, io, dir);
        defer ck.close();
        const allocation = try weights.plan(&ck, admitted, placement);
        const d = ctx.d;
        const e = try gpa.create(Engine);
        errdefer gpa.destroy(e);
        e.configure(gpa, ctx, g, placement, admitted.numerics(), kv8, options);
        e.dir = try gpa.dupe(u8, dir);
        errdefer gpa.free(e.dir);
        e.pockets = try pocket.Pockets.init(gpa, d, g.hidden, g.layers);
        errdefer e.pockets.deinit();
        e.learning = null;
        e.mlp = null;
        e.taught = 0;
        e.stream = try cuda.Stream.init(d, true);
        errdefer e.stream.deinit();
        e.arena = try DeviceBuffer.alloc(d, allocation.weights_bytes + try e.scratch(.measure));
        errdefer e.arena.free();
        try e.arena.fill8(0, null); // touched once now: the driver commits memory on first touch, slowly
        const kk = cuda.kernels;
        const images = [_][]const u8{ kk.qmm_group, kk.qmm_prefill, kk.lane_gemv, kk.affine4_pack, kk.qwen35, kk.sample, kk.qwen35_keep, kk.affine8, kk.affine_prefill, kk.qwen35_kv8, kk.experts, kk.experts_prefill, kk.experts_pack, kk.moe, kk.experts3, kk.experts3_prefill };
        var loaded: usize = 0;
        errdefer for (e.mods[0..loaded]) |*m| m.unload();
        for (images, 0..) |img, i| {
            e.mods[i] = try cuda.Module.load(d, img);
            loaded += 1;
        }
        if (options.terminal_greedy and placement.ownsHead()) {
            var module = try cuda.Module.load(d, kk.tree_round);
            errdefer module.unload();
            e.terminal = .{ .module = module, .kernels = try cuda.tree_round.Kernels.resolve(module) };
        }
        errdefer if (e.terminal) |*t| t.module.unload();
        const sms: usize = @intCast(try ctx.attribute(.multiprocessor_count));
        const gb10 = try ctx.attribute(.compute_capability_major) == 12 and try ctx.attribute(.compute_capability_minor) == 1;
        e.affine = try cuda.qlinear.Affine4Kernels.resolve(d, e.mods[0], e.mods[1], e.mods[2], e.mods[3], sms, gb10);
        errdefer e.affine.deinit();
        e.wide = try cuda.qlinear.AffineNKernels.resolve(d, e.mods[7], e.mods[8], e.mods[3], sms, gb10);
        errdefer e.wide.deinit();
        e.draws = try cuda.sample.Draws.resolve(e.mods[5]);
        e.experts = try cuda.experts.Affine4Experts.resolve(e.mods[10], e.mods[11], e.mods[12], sms);
        e.experts3 = try cuda.experts3.Affine3Experts.resolve(e.mods[14], e.mods[15], sms);
        e.plan = try cuda.grouped.Router.resolve(e.mods[10]);
        e.moe = try cuda.moe.Fns.resolve(e.mods[13]);
        inline for (@typeInfo(Fns).@"struct".field_names) |name| @field(e.fns, name) = try e.mods[4].function("tf_q35_" ++ name);
        inline for (@typeInfo(Keeps).@"struct".field_names) |name| @field(e.keeps, name) = try e.mods[6].function("tf_q35k_" ++ name);
        e.fp8 = .{ .rope = try e.mods[9].function("tf_q35_rope8"), .attn_partial = try e.mods[9].function("tf_q35_attn_partial8"), .compact = try e.mods[9].function("tf_q35k_compact8") };
        if (e.numerics == .hf_modelopt) {
            e.modelopt = try modelopt.Bindings.loadWithOptions(ctx, &e.fns, .{ .up_layout = options.expert_up_layout, .moe_math = options.moe_math });
            // HF bind installs the serial reference; chunked takes the HF image's chunk-grid kernels (same per-key math).
            if (e.attention_mode == .chunked) {
                const names = hf.attnKernelNames(.chunked);
                e.fns.attn_partial = try e.modelopt.?.image.function(names.partial);
                e.fns.attn_merge = try e.modelopt.?.image.function(names.merge);
                e.fp8.attn_partial = try e.modelopt.?.image.function(hf.attn_partial8_chunked);
            }
        }
        errdefer if (e.modelopt) |*m| m.unload();
        e.layers = try gpa.alloc(Layer, g.layers);
        errdefer gpa.free(e.layers);
        {
            // each projection as stored, packed from here; freed before the server counts what streams can use
            var stage = try DeviceBuffer.alloc(d, allocation.staging_bytes);
            defer stage.free();
            try weights.loadPlaced(e, &ck, &stage, admitted, placement);
        }
        if (e.numerics == .hf_modelopt) try e.validateHfWorkspaces();
        _ = try e.scratch(.allocate);
        e.tables = try DeviceBuffer.alloc(d, tables_len);
        errdefer e.tables.free();
        e.staged = try cuda.HostBuffer.alloc(d, tables_len);
        errdefer e.staged.free();
        e.host_logits = try cuda.HostBuffer.alloc(d, TerminalOutput.init(most_verify, g.vocab).bytes);
        errdefer e.host_logits.free();
        try e.stream.synchronize();
        const warm_start = profile.now(io);
        e.startup = .{ .load = warm_start - started };
        if (warmsInConstructor(placement)) e.startup.add(try profile.withCache(gpa, io, e, Cache));
        return e;
    }

    /// Split pipeline stages warm through Chain's session, not a full-engine constructor forward.
    pub fn warmsInConstructor(placement: checkpoint.Placement) bool {
        return placement.ownsEmbedding() and placement.ownsHead();
    }

    fn configure(e: *Engine, gpa: std.mem.Allocator, ctx: *const cuda.Context, g: Geometry, placement: checkpoint.Placement, numerics: checkpoint.Numerics, kv8: bool, options: InitOptions) void {
        const d = ctx.d;
        e.* = undefined;
        e.gpa = gpa;
        e.d = d;
        e.context = ctx.handle;
        e.g = g;
        e.placement = placement;
        e.numerics = numerics;
        e.modelopt = null;
        e.bf16 = null;
        e.terminal_weights = .none;
        e.captured = true;
        e.graph_nodes = 0;
        e.graph_launches = 0;
        e.expert_up_n8 = e.numerics == .hf_modelopt;
        e.expert_up_layout = options.expert_up_layout;
        e.expert_up_split_m = options.expert_up_layout == .slice_major_v1;
        e.reuse_fp8_quant = false;
        e.bounded_hf_prefill = false;
        e.attention_mode = if (e.numerics == .hf_modelopt) options.attention_mode else .reference;
        e.terminal = null;
        e.verify_terminal_graphs = false;
        e.terminal_graph_roster = .empty;
        e.verify_expert_graphs = false;
        e.expert_graph_nodes_baseline = 0;
        e.expert_graph_nodes_n8 = 0;
        e.expert_graph_nodes_split_m = 0;
        e.expert_graph_roster = .empty;
        e.quant_graph_nodes_baseline = 0;
        e.quant_graph_nodes_reuse = 0;
        e.quant_graphs_verified = 0;
        e.verify_gdn_graphs = false;
        e.gdn_graph_nodes_baseline = 0;
        e.gdn_graph_nodes_candidate = 0;
        e.observer = null;
        e.observed_layer = 0;
        e.geo = .{ .hid = @intCast(g.hidden), .nk = @intCast(g.k_heads), .nv = @intCast(g.v_heads), .qh = @intCast(g.q_heads), .kvh = @intCast(g.kv_heads), .nl = @intCast(g.deltas()) };
        e.verify_rows = most_verify;
        e.kv8 = kv8;
        e.window_rows = most_window;
        e.arena_at = 0;
        e.graphs = .empty;
        e.n_taps = 0;
        e.features = 0;
    }

    pub const Bf16Budget = struct {
        weights_bytes: usize,
        load_stage_bytes: usize,
        scratch_bytes: usize,
        cache_bytes: usize,
        hidden_bytes: usize,
        host_input_bytes: usize,
        host_tables_bytes: usize,
        host_conversion_peak: usize,
        steady_device_bytes: usize,
        load_device_peak_bytes: usize,
    };

    /// Logical allocations only. Mapped tables and driver/module residency are not measured here.
    pub fn bf16Budget(plan: *const bf16_manifest.Plan, tokens: usize) !Bf16Budget {
        if (tokens == 0 or tokens > hidden.max_rows) return error.InvalidQwenHiddenShape;
        const g = plan.geometry;
        var e: Engine = undefined;
        e.g = g;
        e.numerics = .hf_dense_bf16;
        const add = std.math.add;
        const mul = std.math.mul;
        const scratch_bytes = try add(usize, try e.scratch(.measure), aligned(tables_len));
        const row_bytes = try mul(usize, g.hidden, 2);
        const hidden_bytes = try mul(usize, tokens, row_bytes);
        const host_input = try mul(usize, hf_prefill_rows, row_bytes);
        const state = try mul(usize, g.deltas(), try add(usize, g.recBytes(), g.convBytes()));
        const kv = try mul(usize, try mul(usize, g.attentions(), 2 * g.kv_heads * head_dim * 2), tokens);
        const cache = try add(usize, try add(usize, state, kv), g.logBytes(1));
        const bytes = plan.bytes();
        const fixed = try add(usize, bytes.device_weights, scratch_bytes);
        const steady = try add(usize, fixed, try add(usize, cache, hidden_bytes));
        return .{
            .weights_bytes = bytes.device_weights,
            .load_stage_bytes = bytes.device_load_stage,
            .scratch_bytes = scratch_bytes,
            .cache_bytes = cache,
            .hidden_bytes = hidden_bytes,
            .host_input_bytes = host_input,
            .host_tables_bytes = bytes.host_tables,
            .host_conversion_peak = bytes.host_conversion_peak,
            .steady_device_bytes = steady,
            .load_device_peak_bytes = @max(steady, try add(usize, bytes.device_weights, bytes.device_load_stage)),
        };
    }

    /// Development-only dense Clef execution. No vocabulary head, pipeline, generation or FP8 KV.
    /// The caller verifies the pinned checkpoint hashes and owns the current context until deinit.
    pub fn initBf16(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8) !*Engine {
        var plan = try bf16_manifest.Plan.inspect(gpa, io, dir);
        defer plan.deinit();
        _ = try bf16Budget(&plan, hidden.max_rows);
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        if (ctx.handle != cuda.copy_source.current()) return error.QwenHiddenDeviceMismatch;
        return initBf16Inspected(gpa, io, ctx, dir, &plan);
    }

    fn initBf16Inspected(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, plan: *const bf16_manifest.Plan) !*Engine {
        return initBf16WithBinder(bf16_bindings.Bindings, gpa, io, ctx, dir, plan);
    }

    fn initBf16WithBinder(comptime Binder: type, gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, plan: *const bf16_manifest.Plan) !*Engine {
        const g = plan.geometry;
        const e = try gpa.create(Engine);
        errdefer gpa.destroy(e);
        e.configure(gpa, ctx, g, .{ .first = 0, .end = g.layers, .total = g.layers }, .hf_dense_bf16, false, .{});
        e.pockets = try pocket.Pockets.init(gpa, ctx.d, g.hidden, g.layers);
        errdefer e.pockets.deinit();
        e.learning = null;
        e.mlp = null;
        e.taught = 0;
        e.dir = &.{};
        e.bounded_hf_prefill = true;
        e.verify_rows = 0;
        e.window_rows = 1;
        e.terminal_weights = .hidden_only;
        e.embed_w = 0;
        e.embed_s = 0;
        e.embed_b = 0;
        e.embed_bits = 16;
        e.stream = try cuda.Stream.init(ctx.d, true);
        errdefer e.stream.deinit();
        var bindings = try Binder.load(ctx, &e.fns);
        errdefer bindings.unload();
        var loaded = try bf16_weights.load(gpa, io, dir, plan, ctx.d, e.stream, &bindings.pack);
        errdefer loaded.deinit();
        e.layers = loaded.layers;
        e.final_norm = loaded.final_norm;
        e.arena = try DeviceBuffer.alloc(ctx.d, try e.scratch(.measure));
        errdefer e.arena.free();
        errdefer cuda.completion.drain(ctx.d, e.stream, "failed BF16 initialization did not drain");
        try e.arena.fill8(0, null);
        var input = try cuda.copy_source.Buffer.alloc(ctx.d, hf_prefill_rows * g.hidden * 2);
        errdefer input.free();
        e.bf16 = .{ .weights = loaded, .bindings = bindings, .input = input };
        _ = try e.scratch(.allocate);
        try e.validateHfWorkspaces();
        e.tables = try DeviceBuffer.alloc(ctx.d, tables_len);
        errdefer e.tables.free();
        e.staged = try cuda.HostBuffer.alloc(ctx.d, tables_len);
        errdefer e.staged.free();
        e.host_logits = .{ .d = ctx.d, .bytes = &.{} };
        return e;
    }

    fn gatherInput(e: *Engine, ids: []const u32) !void {
        const b = if (e.bf16) |*b| b else return error.QwenFormatNotLoaded;
        if (ids.len == 0 or ids.len > hf_prefill_rows) return error.InvalidQwenHiddenShape;
        const bytes = try std.math.mul(usize, ids.len, e.g.hidden * 2);
        try b.input.wait();
        const out = b.input.bytes[0..bytes];
        try b.weights.host.gather(.embedding, ids, out);
        try b.input.copyTo(e.arena, e.h - e.arena.ptr, out, e.stream);
    }

    /// Untied lexical rows for the later joint head. This does not enqueue device work.
    pub fn gatherOutputRows(e: *const Engine, ids: []const u32, out: []u8) !void {
        const b = if (e.bf16) |*b| b else return error.QwenFormatNotLoaded;
        try b.weights.host.gather(.output, ids, out);
    }

    /// Prompt rows admitted during open: ModelOpt default 1, bounded HF 16, MLX 32.
    pub fn warmupPromptRows(e: *const Engine) usize {
        if (!hfOrder(e.numerics)) return 32;
        return if (e.bounded_hf_prefill) hf_prefill_rows else 1;
    }

    /// Every path a request takes, run once now: WDDM stalls a process's first forwards for seconds (a 20 s launch).
    pub fn warmCache(e: *Engine, warm_cache: *Cache) !void {
        if (!warmsInConstructor(e.placement)) return;
        const prompt_rows = e.warmupPromptRows();
        var ids: [32]u32 = @splat(0);
        try e.forward(&.{.{ .cache = warm_cache, .rows = prompt_rows }}, ids[0..prompt_rows], .last);
        if (hfOrder(e.numerics)) {
            try e.stream.synchronize();
            return;
        }
        const greedy: Draw = .{ .rule = null, .position = 33 };
        try e.forward(&.{.{ .cache = warm_cache, .rows = 1 }}, ids[0..1], .{ .draws = &.{greedy} });
        try e.forward(&.{.{ .cache = warm_cache, .rows = 1 }}, ids[0..1], .{ .draws = &.{.{ .rule = .{ .seed = 1, .inv_t = 1.0 / 0.7, .top_p = 0.95, .near = 20.0, .min_log = -std.math.inf(f32), .top_k = 20 }, .position = 34 }} });
        try e.forward(&.{.{ .cache = warm_cache, .rows = 3, .parents = &.{ -1, 0, 0 } }}, ids[0..3], .{ .draws = &.{ greedy, greedy, greedy } });
        try warm_cache.keep(e, &.{ 0, 2 });
        try e.stream.synchronize();
    }

    pub fn deinit(e: *Engine) void {
        cuda.completion.drain(e.d, e.stream, "Qwen engine still has active device work");
        e.terminal_graph_roster.deinit(e.gpa);
        e.expert_graph_roster.deinit(e.gpa);
        var it = e.graphs.valueIterator();
        while (it.next()) |x| x.deinit();
        e.graphs.deinit(e.gpa);
        if (e.terminal) |*t| t.module.unload();
        if (e.numerics != .hf_dense_bf16) e.host_logits.free();
        e.staged.free();
        e.tables.free();
        e.arena.free();
        e.pockets.deinit();
        if (e.bf16) |*b| {
            b.input.free();
            b.weights.deinit();
            b.bindings.unload();
        } else {
            e.gpa.free(e.layers);
            if (e.modelopt) |*m| m.unload();
            e.gpa.free(e.dir);
            e.affine.deinit();
            e.wide.deinit();
            for (&e.mods) |*m| m.unload();
        }
        e.stream.deinit();
        e.gpa.destroy(e);
    }

    /// A stream's device bytes at a `cap`-token window: recurrences, convolution tails, keys and values, taps, log.
    pub fn streamBytes(e: *const Engine, cap: usize) usize {
        const state = streamStateCopies(e.numerics) * e.g.stateBytes();
        return state + e.g.kvBytes(cap, e.kv8) + e.tapBytes() + e.g.logBytes(streamLogRows(e.numerics));
    }

    /// A stream's tap ring: tap_rows rows of the tapped layers' residuals.
    pub fn tapBytes(e: *const Engine) usize {
        return tap_rows * e.n_taps * e.g.hidden * 2;
    }

    /// The layers a drafter taps, told before the streams' caches exist; each forward then fills their rings.
    pub fn setTaps(e: *Engine, taps: []const u32) !void {
        if (hfOrder(e.numerics)) return error.UnsupportedQwenTaps;
        if (taps.len > e.taps.len) return error.TooManyTaps;
        for (taps) |t| if (t >= e.g.layers) return error.BadTap;
        @memcpy(e.taps[0..taps.len], taps);
        e.n_taps = taps.len;
    }

    /// The next `bytes` of the arena, 256-byte aligned.
    pub fn alloc(e: *Engine, bytes: usize) !u64 {
        const at = aligned(e.arena_at);
        if (at + bytes > e.arena.len) return error.ArenaExhausted;
        e.arena_at = at + bytes;
        return e.arena.ptr + at;
    }

    pub fn scratchBytes(admitted: checkpoint.Admitted, placement: checkpoint.Placement) !usize {
        try admitted.requireExecutable();
        var e: Engine = undefined;
        e.g = try placement.localGeometry(admitted.geometry);
        e.numerics = admitted.numerics();
        return aligned(tables_len) + try e.scratch(.measure);
    }

    /// The scratch rows' bytes (`measure`: summed, nothing taken), or the rows themselves from the arena.
    fn scratch(e: *Engine, mode: enum { measure, allocate }) !usize {
        var total: usize = 0;
        const Take = struct {
            e: *Engine,
            mode: @TypeOf(mode),
            total: *usize,
            pub fn alloc(t: @This(), bytes: usize) !u64 {
                t.total.* = try std.math.add(usize, t.total.*, (try std.math.add(usize, bytes, 255)) & ~@as(usize, 255));
                return if (t.mode == .measure) 0 else try t.e.alloc(bytes);
            }
        };
        const t = Take{ .e = e, .mode = mode, .total = &total };
        const g = e.g;
        const R: usize = if (hfOrder(e.numerics)) hf_prefill_rows else max_rows;
        const fp: usize = if (hfOrder(e.numerics)) 4 else 2;
        const wide = @max(g.valueWidth(), g.attnWidth(), g.hidden); // y and mix: either mixer's rows
        inline for (.{ "h", "x", "r" }) |name| @field(e, name) = try t.alloc(R * g.hidden * 2);
        e.y = try t.alloc(R * wide * fp);
        inline for (.{ "mix", "z", "v" }) |name| @field(e, name) = try t.alloc(R * wide * 2);
        inline for (.{ "q", "k" }) |name| @field(e, name) = try t.alloc(R * g.k_heads * geometry.dk * fp);
        e.qkv = try t.alloc(R * g.convWidth() * 2);
        inline for (.{ "gate", "up", "act" }) |name| @field(e, name) = try t.alloc(R * g.intermediate * 2);
        e.xs = try t.alloc(R * (@max(g.intermediate, wide) / 64) * 4);
        inline for (.{ "a", "b", "beta" }) |name| @field(e, name) = try t.alloc(R * g.v_heads * 2);
        e.g_ = try t.alloc(R * g.v_heads * 4);
        e.qraw = try t.alloc(R * 2 * g.attnWidth() * 2);
        e.qn = try t.alloc(R * g.attnWidth() * 2);
        inline for (.{ "kraw", "vraw", "kn" }) |name| @field(e, name) = try t.alloc(R * g.kv_heads * head_dim * 2);
        e.pm = try t.alloc(R * g.q_heads * max_chunks * 4);
        e.pl = try t.alloc(R * g.q_heads * max_chunks * 4);
        e.po = try t.alloc(R * g.q_heads * max_chunks * head_dim * 4);
        e.logits = if (e.numerics == .hf_dense_bf16) 0 else try t.alloc(TerminalOutput.init(most_verify, g.vocab).bytes);
        if (e.numerics == .hf_modelopt) {
            const rows = try t.alloc(R * hfInputWidth(g));
            const part = try t.alloc(hf_part_bytes);
            if (mode == .allocate) {
                e.modelopt.?.rows = rows;
                e.modelopt.?.part = part;
            }
        }
        if (e.numerics == .hf_dense_bf16) {
            const part = try t.alloc(hf_part_bytes);
            if (mode == .allocate) e.bf16.?.bindings.part = part;
        }
        if (g.moe()) e.routed = try routed.Scratch.carve(t, g, R);
        e.amax = if (e.numerics == .hf_dense_bf16) 0 else try t.alloc(most_verify * argmaxSlices(g.vocab) * 8);
        return total;
    }

    fn validateHfProjection(e: *const Engine, p: Proj) !void {
        if (p != .linear or p.inputs() > hfInputWidth(e.g)) return error.QwenLinearWorkspace;
        const w = switch (p.linear) {
            .nvfp4, .fp8, .bf16 => |w| w,
            else => return error.QwenLinearWorkspace,
        };
        if (e.numerics == .hf_dense_bf16 and p.linear != .bf16) return error.QwenLinearWorkspace;
        const lane = if (e.numerics == .hf_dense_bf16) e.bf16.?.bindings.lane else e.modelopt.?.lane;
        var bytes = lane.partBytes(hf_prefill_rows, w);
        if (p.linear == .fp8 and w.act != 0) {
            const sk = cuda.qmmf.splitK(w.n, w.k);
            if (sk > 1 and !e.modelopt.?.ck.clusters) bytes = @max(bytes, hf_prefill_rows * @as(usize, w.n) * sk * 4);
        }
        if (bytes > hf_part_bytes) return error.QwenLinearWorkspace;
    }

    fn validateHfWorkspaces(e: *const Engine) !void {
        for (e.layers) |layer| {
            switch (layer.mixer) {
                .delta => |m| inline for (.{ "qkv", "z", "a", "b", "out" }) |name| try e.validateHfProjection(@field(m, name)),
                .attention => |m| inline for (.{ "q", "k", "v", "o" }) |name| try e.validateHfProjection(@field(m, name)),
            }
            switch (layer.mlp) {
                .dense => |m| inline for (.{ "gate", "up", "down" }) |name| try e.validateHfProjection(@field(m, name)),
                .moe => |m| if (m.shared == .projected) {
                    inline for (.{ "gate", "up", "down" }) |name| try e.validateHfProjection(@field(m.shared.projected, name));
                },
            }
        }
        if (e.terminal_weights == .vocabulary) try e.validateHfProjection(e.terminal_weights.vocabulary);
    }

    pub fn run(e: *Engine, f: cuda.Function, grid: [3]usize, block: [3]usize, args: anytype) !void {
        var a: cuda.Args = .{};
        inline for (args) |v| a.add(v);
        const cfg: cuda.Config = .{ .grid = .{ .x = @intCast(grid[0]), .y = @intCast(grid[1]), .z = @intCast(grid[2]) }, .block = .{ .x = @intCast(block[0]), .y = @intCast(block[1]), .z = @intCast(block[2]) } };
        try cuda.launch.launch(f, cfg, e.stream, &a);
    }

    /// out (rows, n) = x (rows, k) through `l`: row tiles with batch-independent bits or a prompt GEMM.
    pub fn proj(e: *Engine, l: Proj, x: u64, out: u64, rows: usize, prompt: bool) !void {
        return e.projSums(l, x, e.xs, out, rows, prompt, e.tables.ptr + params_at);
    }

    /// `proj` with group sums `xs` and device counts `params` (0: all real); prompts past 16 rows take a prompt GEMM.
    pub fn projSums(e: *Engine, l: Proj, x: u64, xs: u64, out: u64, rows: usize, prompt: bool, params: u64) !void {
        if (e.numerics == .hf_dense_bf16) {
            if (l != .linear or l.linear != .bf16) return error.QwenFormatNotLoaded;
            const b = if (e.bf16) |*b| b else return error.QwenFormatNotLoaded;
            return b.bindings.linear().decode(e.stream, l.linear, .{ .x = x, .part = b.bindings.part }, out, rows);
        }
        if (l == .linear) {
            const m = &(e.modelopt orelse return error.QwenFormatNotLoaded);
            return m.linear().decode(e.stream, l.linear, .{ .x = x, .part = m.part }, out, rows);
        }
        return e.project(l, x, xs, out, rows, prompt, params, true);
    }

    pub fn projAgain(e: *Engine, l: Proj, x: u64, out: u64, rows: usize, prompt: bool) !void {
        return e.project(l, x, e.xs, out, rows, prompt, e.tables.ptr + params_at, false);
    }

    fn project(e: *Engine, l: Proj, x: u64, xs: u64, out: u64, rows: usize, prompt: bool, params: u64, sums: bool) !void {
        const n_out = l.outputs();
        const k = l.inputs();
        if (prompt and rows > 16) return switch (l) {
            .a4 => |q| e.affine.prompt(e.stream, x, q, out, rows),
            .an => |q| e.wide.prompt(e.stream, x, q, out, rows),
            .linear => unreachable,
        };
        if (sums) try e.run(e.fns.sums, .{ rows, (k / 64 + 63) / 64, 1 }, .{ 64, 1, 1 }, .{ x, xs, @as(c_int, @intCast(k)), params });
        const tile: usize = switch (l) {
            .a4 => e.affine.verifyRows(),
            .an => 16,
            .linear => unreachable,
        };
        var at: usize = 0;
        while (at < rows) : (at += tile) {
            const n = @min(tile, rows - at);
            const xa = x + at * k * 2;
            const sa = xs + at * (k / 64) * 4;
            const oa = out + at * n_out * 2;
            switch (l) {
                .a4 => |q| try e.affine.verify(e.stream, xa, sa, q, oa, n),
                .an => |q| try e.wide.decode(e.stream, xa, sa, q, oa, n),
                .linear => unreachable,
            }
        }
    }

    /// Token rows of the embedding into `out` (rows, hidden) bf16, at the width its words are served.
    pub fn embed(e: *Engine, ids: u64, out: u64, rows: usize, params: u64) !void {
        if (e.numerics == .hf_dense_bf16) return error.QwenHostEmbeddingRequired;
        const f = if (e.embed_bits == 8) e.fns.embed8 else e.fns.embed;
        try e.run(f, .{ rows, 1, 1 }, .{ 256, 1, 1 }, .{ ids, e.embed_w, e.embed_s, e.embed_b, out, @as(c_int, @intCast(e.g.hidden)), params });
    }

    /// Logits rows first..first + n's greedy tokens into toks: the highest logit, ties to the lower id.
    fn argmax(e: *Engine, first: usize, n: usize, toks: u64) !void {
        if (n == 0) return;
        const slices = argmaxSlices(e.g.vocab);
        const pv = e.amax;
        const pi = e.amax + n * slices * 4;
        try e.run(e.fns.argmax_slices, .{ slices, n, 1 }, .{ 256, 1, 1 }, .{ e.logits + first * e.g.vocab * 2, @as(c_int, @intCast(e.g.vocab)), pv, pi });
        try e.run(e.fns.argmax_merge, .{ n, 1, 1 }, .{ 32, 1, 1 }, .{ pv, pi, @as(c_int, @intCast(slices)), toks + first * 4 });
    }

    pub fn norm(e: *Engine, residual: bool, w: u64, rows: usize) !void {
        try e.run(e.fns.norm, .{ rows, 1, 1 }, .{ if (hfOrder(e.numerics)) (hf.Launches{}).norm else 256, 1, 1 }, .{ e.h, if (residual) e.r else @as(u64, 0), w, e.x, e.g.eps, @as(c_int, @intCast(e.g.hidden)), e.tables.ptr + params_at, e.xs });
    }

    /// A request owns these rows across forward scratch reuse.
    pub fn hiddenRows(e: *Engine, tokens: usize) !hidden.Rows {
        return hidden.Rows.init(e, tokens);
    }

    pub fn validateHiddenProfile(e: *const Engine) !void {
        switch (e.numerics) {
            .hf_dense_bf16 => {
                if (e.bf16 == null or e.terminal_weights != .hidden_only or e.g.moe()) return error.UnsupportedQwenPlainExecution;
                if (e.verify_gdn_graphs or e.verify_expert_graphs) return error.UnsupportedQwenHiddenProfile;
            },
            .mlx_sanitized, .hf_modelopt => {},
        }
        if (!e.placement.ownsEmbedding() or !e.placement.ownsHead() or e.kv8 or e.n_taps != 0 or
            e.verify_terminal_graphs or (e.observer != null and e.captured)) return error.UnsupportedQwenHiddenProfile;
        if (e.context != cuda.copy_source.current()) return error.QwenHiddenDeviceMismatch;
    }

    fn validateHidden(e: *const Engine, parts: []const Part, ids: []const u32, w: hidden.Window) !void {
        try e.validateHiddenProfile();
        if (parts.len != 1 or parts[0].parents != null or parts[0].images != null or
            ids.len == 0 or ids.len > max_rows or parts[0].rows != ids.len) return error.InvalidQwenHiddenShape;
        try w.validate(e, ids.len);
        try w.validateCache(parts[0].cache);
    }

    /// One forward over `parts`' windows, `ids` in order; `keep` names the kept rows; an unkept verify is kept first.
    pub fn forward(e: *Engine, parts: []const Part, ids: []const u32, keep: Keep) !void {
        if (!e.placement.ownsEmbedding() or !e.placement.ownsHead()) return error.QwenRequiresPipeline;
        const shape = try e.prepareForward(parts, ids, keep);
        errdefer {
            for (parts) |p| p.cache.terminal_greedy = null;
            if (keep == .hidden) keep.hidden.fail();
        }
        if (e.numerics == .hf_dense_bf16) try e.gatherInput(ids);
        try e.executeForward(shape);
        try e.finish(parts, keep, shape);
        commit(parts, shape.record);
        if (keep == .hidden) keep.hidden.commit();
    }

    /// Hidden output is a whole forward transaction, not a pipeline terminal.
    pub fn prepare(e: *Engine, parts: []const Part, ids: []const u32, keep: Keep) !Shape {
        if (e.numerics == .hf_dense_bf16) return error.QwenHiddenRequiresForward;
        if (keep == .hidden) return error.QwenHiddenRequiresForward;
        return e.prepareForward(parts, ids, keep);
    }

    fn prepareForward(e: *Engine, parts: []const Part, ids: []const u32, keep: Keep) !Shape {
        if (e.numerics == .hf_dense_bf16 and keep != .hidden) return error.QwenHiddenOnly;
        if (keep == .hidden) try e.validateHidden(parts, ids, keep.hidden);
        if (parts.len == 0 or parts.len > max_segs) return error.InvalidQwenSegments;
        const gdn_layout = parts[0].cache.gdn_layout;
        for (parts[1..]) |p| if (p.cache.gdn_layout != gdn_layout) return error.MixedQwenGdnLayouts;
        try gdn_layout.validate(e.numerics);
        if (e.expert_up_n8 and e.numerics != .hf_modelopt) return error.UnsupportedQwenExpertUp;
        try validateCombined(e.numerics, e.expert_up_n8, e.expert_up_split_m, e.reuse_fp8_quant);
        try e.validateUpLayout(e.expert_up_layout, e.expert_up_n8, e.expert_up_split_m, ids.len);
        const g = e.g;
        try validateHfWindow(e.numerics, e.bounded_hf_prefill, parts, ids.len, keep);
        var total: usize = 0;
        for (parts, 0..) |p, index| {
            for (parts[0..index]) |prior| if (prior.cache == p.cache) return error.InvalidQwenSegments;
            if (p.rows == 0 or p.cache.len + p.rows > p.cache.cap) return error.QwenContextFull;
            total += p.rows;
        }
        if (parts.len == 0 or parts.len > max_segs or total != ids.len or total > max_rows) return error.InvalidQwenSegments;
        if (e.learning != null and total != 1) return error.BadLearningRows;
        const record = records(keep);
        if (record and total > e.verify_rows) return error.InvalidQwenSegments;
        if (!record and parts.len != 1) return error.InvalidQwenSegments;
        if (keep == .draws and keep.draws.len != total) return error.InvalidQwenSegments;
        for (ids) |id| if (id >= g.vocab) return error.InvalidQwenToken;
        for (parts) |p| if (p.parents != null and !record) return error.InvalidQwenParents;
        for (parts) |p| if (p.images != null and record) return error.InvalidQwenImages;
        for (parts) |p| if (p.cache.hidden_invalid) return error.QwenHiddenNeedsReset;
        if (keep == .hidden) keep.hidden.begin(parts[0].cache);
        errdefer if (keep == .hidden) keep.hidden.fail();
        for (parts) |p| p.cache.terminal_greedy = null;
        errdefer {
            for (parts) |p| p.cache.terminal_greedy = null;
        }
        for (parts) |p| try p.cache.settle(e);
        // the prompt GEMM only for prompt chunks: a window's rows keep the bits of one-row steps
        const prompt = !record;
        // the tables
        const st = e.staged.bytes;
        @memcpy(std.mem.bytesAsSlice(u32, st[ids_at..][0 .. total * 4]), ids);
        const slot_peak = try Round.build(st, parts, record);
        const segs = std.mem.bytesAsSlice(Seg, st[segs_at..][0 .. parts.len * @sizeOf(Seg)]);
        var row0: usize = 0;
        for (parts, 0..) |p, i| {
            segs[i] = p.cache.seg(e, p.cache.len, row0, p.rows);
            row0 += p.rows;
        }
        if (hfOrder(e.numerics) and total > 1) {
            const views = std.mem.bytesAsSlice(Seg, st[chain_segs_at..][0 .. total * @sizeOf(Seg)]);
            for (views, 0..) |*view, row| view.* = chainView(segs[0], row);
            try validateChainViews(segs[0], views);
        }
        if (keep == .draws) {
            const rules = std.mem.bytesAsSlice(cuda.sample.Rule, st[rules_at..][0 .. total * @sizeOf(cuda.sample.Rule)]);
            const meta = std.mem.bytesAsSlice(c_int, st[meta_at..][0 .. total * 4]);
            for (keep.draws, rules, meta) |p, *r, *m| {
                if (p.rule) |x| r.* = x;
                m.* = @intCast(p.position -% 1); // sample.cu keys row r at meta + r + 1
            }
        }
        // words 0 and 1 are reserved (each row's attention chunks follow its own keys: qwen35.cu's chunk_keys)
        const params = std.mem.bytesAsSlice(c_int, st[params_at..][0..32]);
        params[0] = 0;
        params[1] = 0;
        params[2] = @intCast(total);
        params[3] = @intCast(parts.len);
        params[4] = @intFromBool(record);
        defer e.taught = 0;
        try @import("cuda_forward_extra.zig").stageScales(e, parts, total);
        // an image prompt's chunk (one part, its rows at prompt positions len..): feature rows and t, h, w positions
        const images: ?*const Images = if (record) null else parts[0].images;
        if (images) |im| {
            const p0 = parts[0].cache.len;
            if (p0 + total > im.src.len or im.pos.len != 3 * im.src.len) return error.InvalidQwenImages;
            const n = im.src.len;
            @memcpy(std.mem.bytesAsSlice(i32, st[src_at..][0 .. total * 4]), im.src[p0..][0..total]);
            const mpos = std.mem.bytesAsSlice(i32, st[mpos_at..][0 .. 3 * max_rows * 4]);
            for (0..3) |a| @memcpy(mpos[a * max_rows ..][0..total], im.pos[a * n + p0 ..][0..total]);
        }
        try e.tables.uploadAsync(0, st[0..if (images != null or (hfOrder(e.numerics) and total > 1)) tables_len else src_at], e.stream.handle);
        // a captured forward covers a row bucket and every window; kernels read the true counts; a lone step is exact
        const lone = total == 1 and parts.len == 1;
        const bucket: usize = if (lone) 1 else if (total <= 16) 16 else most_verify;
        const windows: usize = if (lone) 1 else max_segs;
        const shape: Shape = .{ .total = @intCast(if (prompt) total else bucket), .nseg = @intCast(if (prompt) parts.len else windows), .record = record, .prompt = prompt, .taps = e.n_taps > 0, .gdn_layout = gdn_layout, .expert_up_n8 = e.expert_up_n8, .heads_all = keep == .prompt_logits, .expert_up_split_m = e.expert_up_split_m, .reuse_fp8_quant = e.reuse_fp8_quant, .expert_up_layout = e.expert_up_layout, .terminal_greedy = e.terminal != null and !record and keep != .hidden and keep != .learn, .moe_math = e.moeMath(), .attention_mode = e.attentionMode(), .normalized_hidden = keep == .hidden or keep == .learn, .slots = @intCast(Round.variant(slot_peak)) };
        e.features = if (images) |im| im.features else 0;
        return shape;
    }

    pub fn execute(e: *Engine, shape: Shape) !void {
        if (e.numerics == .hf_dense_bf16) return error.QwenHiddenRequiresForward;
        return e.executeForward(shape);
    }

    fn executeForward(e: *Engine, shape: Shape) !void {
        if (e.numerics == .hf_dense_bf16 and !shape.normalized_hidden) return error.QwenHiddenOnly;
        if (shape.normalized_hidden) {
            try e.validateHiddenProfile();
            if (!shape.prompt or shape.record or shape.taps or shape.heads_all or shape.terminal_greedy or
                shape.nseg != 1 or shape.total == 0 or shape.total > max_rows) return error.InvalidQwenHiddenShape;
        }
        if (shape.terminal_greedy and (e.numerics != .hf_modelopt or !e.placement.ownsHead() or e.terminal == null or shape.record or !shape.prompt)) return error.UnsupportedQwenTerminalGreedy;
        try validateCombined(e.numerics, shape.expert_up_n8, shape.expert_up_split_m, shape.reuse_fp8_quant);
        if (shape.expert_up_n8 and (e.numerics != .hf_modelopt or (shape.prompt and shape.total > hf_prefill_rows))) return error.UnsupportedQwenExpertUp;
        if (shape.heads_all and e.numerics != .hf_modelopt) return error.UnsupportedQwenWindow;
        if (hfOrder(e.numerics) and (shape.total == 0 or shape.total > hf_prefill_rows or shape.nseg != 1 or shape.record or !shape.prompt or shape.taps or
            (!e.bounded_hf_prefill and (shape.total != 1 or shape.heads_all)))) return error.UnsupportedQwenWindow;
        if (e.observer != null and e.captured) return error.QwenObserverRequiresEager;
        try e.validateUpLayout(shape.expert_up_layout, shape.expert_up_n8, shape.expert_up_split_m, shape.total);
        try validateMoeMathShape(e.moeMath(), shape.moe_math);
        try validateAttentionModeShape(e.attentionMode(), shape.attention_mode);
        try shape.gdn_layout.validate(e.numerics);
        if (e.verify_gdn_graphs and e.numerics == .hf_modelopt and shape.total > 1) try e.verifyHfChainTables(shape);
        if (shape.prompt and e.numerics == .mlx_sanitized or !e.captured) {
            // uncaptured: a prompt chunk's GEMMs dwarf its launches, and each length's graph would pay the stall
            try e.body(shape);
        } else {
            const gop = try e.graphs.getOrPut(e.gpa, shape.key());
            if (!gop.found_existing) {
                errdefer _ = e.graphs.remove(shape.key());
                // captured, nothing runs: the replay below is this shape's first run
                try cuda.graph.beginCapture(e.stream, .thread_local);
                e.body(shape) catch |err| {
                    var broken = cuda.graph.endCapture(e.stream) catch return err;
                    broken.deinit();
                    return err;
                };
                var captured = try cuda.graph.endCapture(e.stream);
                defer captured.deinit();
                const nodes = try captured.nodeCount();
                const gdn_nodes = if (e.verify_gdn_graphs) try e.verifyGdnGraph(captured, shape, nodes) else 0;
                const expert_nodes = if (e.verify_expert_graphs) try e.verifyExpertGraph(captured, shape, nodes) else 0;
                const quant_nodes = if (e.verify_expert_graphs) try e.verifyQuantGraph(captured, shape, nodes) else 0;
                if (e.verify_terminal_graphs) {
                    const readback = try TerminalReadback.load(e.d);
                    gop.value_ptr.* = try e.instantiateTerminalGraph(captured, shape, nodes, readback);
                } else gop.value_ptr.* = try captured.instantiate();
                if (e.verify_expert_graphs) {
                    e.quant_graphs_verified += 1;
                    if (shape.reuse_fp8_quant) e.quant_graph_nodes_reuse += quant_nodes else e.quant_graph_nodes_baseline += quant_nodes;
                }
                if (shape.expert_up_split_m) {
                    e.expert_graph_nodes_split_m += expert_nodes;
                } else if (shape.expert_up_n8) {
                    e.expert_graph_nodes_n8 += expert_nodes;
                } else e.expert_graph_nodes_baseline += expert_nodes;
                e.graph_nodes += nodes;
                switch (shape.gdn_layout) {
                    .value_key => e.gdn_graph_nodes_baseline += gdn_nodes,
                    .key_value => e.gdn_graph_nodes_candidate += gdn_nodes,
                }
            }
            try gop.value_ptr.launchOn(e.stream);
            e.graph_launches += 1;
        }
    }

    fn validateUpLayout(e: *const Engine, layout: cuda.experts.Nvfp4UpLayout, n8: bool, split: bool, total: usize) !void {
        if (layout != e.expert_up_layout) return error.QwenExpertUpLayoutMismatch;
        if (layout == .slice_major_v1 and (e.numerics != .hf_modelopt or !e.expert_up_n8 or !e.expert_up_split_m or !n8 or !split or total == 0 or total > hf_prefill_rows)) return error.UnsupportedQwenExpertUpLayout;
        for (e.layers) |layer| if (layer.mlp == .moe and layer.mlp.moe.experts.nvfp4_up_layout != layout) return error.QwenExpertUpLayoutMismatch;
    }

    fn verifyHfChainTables(e: *const Engine, shape: Shape) !void {
        try e.stream.synchronize();
        var base: Seg = undefined;
        try e.tables.download(segs_at, std.mem.asBytes(&base));
        const staged = std.mem.bytesAsSlice(Seg, e.staged.bytes[segs_at..][0..@sizeOf(Seg)])[0];
        inline for (@typeInfo(Seg).@"struct".field_names) |name| {
            if (@field(base, name) != @field(staged, name)) return error.QwenGdnTableMismatch;
        }
        var views: [hf_prefill_rows]Seg = undefined;
        try e.tables.download(chain_segs_at, std.mem.sliceAsBytes(views[0..shape.total]));
        try validateChainViews(base, views[0..shape.total]);
        var params: [8]c_int = undefined;
        try e.tables.download(params_at, std.mem.asBytes(&params));
        if (params[2] != shape.total or params[3] != 1 or params[4] != 0) return error.QwenGdnTableMismatch;
        var parents: [hf_prefill_rows]c_int = undefined;
        var segments: [hf_prefill_rows]c_int = undefined;
        var ancestors: [hf_prefill_rows]Round.Ancestor = undefined;
        try e.tables.download(rowpar_at, std.mem.sliceAsBytes(parents[0..shape.total]));
        try e.tables.download(rowseg_at, std.mem.sliceAsBytes(segments[0..shape.total]));
        try e.tables.download(rowanc_at, std.mem.sliceAsBytes(ancestors[0..shape.total]));
        for (0..shape.total) |row| {
            const want: Round.Ancestor = .{ .bits = 0, .depth = @intCast(row), .chain = 1 };
            if (segments[row] != 0 or parents[row] != @as(c_int, @intCast(row)) - 1 or
                ancestors[row].bits != want.bits or ancestors[row].depth != want.depth or ancestors[row].chain != want.chain)
                return error.QwenGdnTableMismatch;
        }
    }

    fn gdnArguments(e: *const Engine, shape: Shape, layer: c_int, row: usize) cuda.Args {
        var args: cuda.Args = .{};
        args.add(e.geo);
        for ([_]u64{ e.q, e.k, e.v, e.g_, e.beta }) |pointer| args.add(pointer);
        args.add(e.tables.ptr + if (shape.total > 1) chain_segs_at + row * @sizeOf(Seg) else segs_at);
        args.add(e.tables.ptr + rowpar_at);
        args.add(layer);
        args.add(e.y);
        args.add(e.tables.ptr + params_at);
        return args;
    }

    fn checkGdnNode(e: *const Engine, p: cuda.abi.KernelNodeParams, shape: Shape, function: cuda.abi.Function) !struct { layer: usize, row: usize } {
        const launch: hf.Launches = .{};
        if (p.func != function or p.grid_x != 1 or p.grid_y != launch.gdn_chain_grid_y or
            p.grid_z != e.g.v_heads * shape.nseg or p.block_x != launch.gdn_chain[0] or
            p.block_y != launch.gdn_chain[1] or p.block_z != launch.gdn_chain[2] or p.shared_bytes != 0) return error.QwenGdnGraphKernelMismatch;
        const actual = p.params orelse return error.QwenGdnGraphMissingArguments;
        const layer: *const c_int = @ptrCast(@alignCast(actual[8] orelse return error.QwenGdnGraphMissingArguments));
        const segment: *const u64 = @ptrCast(@alignCast(actual[6] orelse return error.QwenGdnGraphMissingArguments));
        if (layer.* < 0 or layer.* >= e.g.deltas()) return error.QwenGdnGraphLayerMismatch;
        var row: usize = 0;
        if (shape.total > 1) {
            const base = e.tables.ptr + chain_segs_at;
            if (segment.* < base or (segment.* - base) % @sizeOf(Seg) != 0) return error.QwenGdnGraphArgumentMismatch;
            row = @intCast((segment.* - base) / @sizeOf(Seg));
            if (row >= shape.total) return error.QwenGdnGraphArgumentMismatch;
        }
        const want = e.gdnArguments(shape, layer.*, row);
        for (0..want.count) |i| {
            const got: [*]const u8 = @ptrCast(actual[i] orelse return error.QwenGdnGraphMissingArguments);
            const bytes = want.storage[want.offsets[i]..][0..want.sizes[i]];
            if (!std.mem.eql(u8, got[0..bytes.len], bytes)) return error.QwenGdnGraphArgumentMismatch;
        }
        return .{ .layer = @intCast(layer.*), .row = row };
    }

    fn verifyGdnGraph(e: *const Engine, graph: cuda.graph.Graph, shape: Shape, node_count: usize) !usize {
        const bindings = e.modelopt orelse return error.UnsupportedQwenGdnLayout;
        const baseline = e.fns.gdn_chain.handle;
        const candidate = bindings.gdn_chain_transposed.handle;
        if (baseline == candidate) return error.QwenGdnGraphKernelMismatch;
        const expected = switch (shape.gdn_layout) {
            .value_key => baseline,
            .key_value => candidate,
        };
        const storage = try e.gpa.alloc(cuda.graph.Node, node_count);
        defer e.gpa.free(storage);
        const nodes = try graph.nodes(storage);
        if (nodes.len != node_count) return error.QwenGdnGraphWorkCountMismatch;
        const predecessors = try e.gpa.alloc(cuda.graph.Node, node_count);
        defer e.gpa.free(predecessors);
        var lib = e.d.lib;
        const get_deps = lib.lookup(*const fn (cuda.abi.GraphNode, ?[*]cuda.abi.GraphNode, *usize) callconv(.c) cuda.abi.Result, "cuGraphNodeGetDependencies") orelse return error.MissingSymbol;
        for (nodes, predecessors) |node, *previous| {
            var deps: [2]cuda.graph.Node = .{ null, null };
            var count: usize = deps.len;
            try e.d.check(get_deps(node, &deps, &count), "cuGraphNodeGetDependencies");
            if (count > 1) return error.QwenGdnGraphDependencyMismatch;
            previous.* = if (count == 0) null else deps[0];
        }
        var roster: GdnRoster = .{ .layers = e.g.deltas(), .rows = if (shape.total > 1) shape.total else 1 };
        var prior: cuda.graph.Node = null;
        for (0..node_count) |_| {
            var found: ?usize = null;
            for (predecessors, 0..) |previous, i| if (previous == prior) {
                if (found != null) return error.QwenGdnGraphDependencyMismatch;
                found = i;
            };
            const index = found orelse return error.QwenGdnGraphDependencyMismatch;
            var params: cuda.abi.KernelNodeParams = undefined;
            try e.d.check(e.d.api.cuGraphKernelNodeGetParams_v2(nodes[index], &params), "cuGraphKernelNodeGetParams");
            prior = nodes[index];
            if (params.func != baseline and params.func != candidate) continue;
            const item = try e.checkGdnNode(params, shape, expected);
            try roster.accept(item.layer, item.row);
        }
        try roster.finish();
        return roster.count;
    }

    const CkProjection = struct { weight: cuda.qmmf.Weight, x: u64, out: u64, rows: usize, quant: bool };

    fn addCkProjection(e: *const Engine, list: *std.ArrayList(CkProjection), p: Proj, x: u64, out: u64, rows: usize, quant: bool) !void {
        if (p == .linear and p.linear == .fp8 and p.linear.fp8.act != 0) {
            try list.append(e.gpa, .{ .weight = p.linear.fp8, .x = x, .out = out, .rows = rows, .quant = quant });
        }
    }

    fn addCkGroup(e: *const Engine, list: *std.ArrayList(CkProjection), projections: []const Proj, outputs: []const u64, shape: Shape) !void {
        var targets: [3]cuda.qlinear.Target = undefined;
        for (projections, outputs, 0..) |p, out, i| {
            if (p != .linear) return error.QwenFormatNotLoaded;
            targets[i] = .{ .weight = p.linear, .output = out };
        }
        const together = shape.reuse_fp8_quant and e.modelopt.?.linear().canDecodeTogether(targets[0..projections.len], .{ .x = e.x, .part = e.modelopt.?.part }, shape.total);
        for (projections, outputs, 0..) |p, out, i| try e.addCkProjection(list, p, e.x, out, shape.total, !together or i == 0);
    }

    fn ckPlan(e: *const Engine, shape: Shape) !std.ArrayList(CkProjection) {
        var list: std.ArrayList(CkProjection) = .empty;
        errdefer list.deinit(e.gpa);
        for (e.layers) |layer| {
            switch (layer.mixer) {
                .delta => |m| {
                    try e.addCkGroup(&list, &.{ m.qkv, m.z }, &.{ e.qkv, e.z }, shape);
                    try e.addCkProjection(&list, m.a, e.x, e.a, shape.total, true);
                    try e.addCkProjection(&list, m.b, e.x, e.b, shape.total, true);
                    try e.addCkProjection(&list, m.out, e.mix, e.r, shape.total, true);
                },
                .attention => |m| {
                    try e.addCkGroup(&list, &.{ m.q, m.k, m.v }, &.{ e.qraw, e.kraw, e.vraw }, shape);
                    try e.addCkProjection(&list, m.o, e.mix, e.r, shape.total, true);
                },
            }
            const dense: ?layers.Dense = switch (layer.mlp) {
                .dense => |m| m,
                .moe => |m| if (m.shared == .projected) m.shared.projected else null,
            };
            if (dense) |m| {
                try e.addCkProjection(&list, m.gate, e.x, e.gate, shape.total, true);
                try e.addCkProjection(&list, m.up, e.x, e.up, shape.total, true);
                try e.addCkProjection(&list, m.down, e.act, if (layer.mlp == .dense) e.r else e.routed.shared, shape.total, true);
            }
        }
        if (e.placement.ownsHead() and !shape.normalized_hidden) {
            const heads: usize = if (shape.record or shape.heads_all) shape.total else 1;
            try e.addCkProjection(&list, try e.vocabulary(), e.x + (shape.total - heads) * e.g.hidden * 2, e.logits, heads, true);
        }
        return list;
    }

    const CkNode = struct {
        function: cuda.abi.Function,
        grid: [3]usize,
        shared: usize = 0,
        threads: usize = 128,
        args: cuda.Args = .{},

        fn check(want: *const CkNode, got: cuda.abi.KernelNodeParams) !void {
            if (got.func != want.function or got.grid_x != want.grid[0] or got.grid_y != want.grid[1] or got.grid_z != want.grid[2] or
                got.block_x != want.threads or got.block_y != 1 or got.block_z != 1 or got.shared_bytes != want.shared) return error.QwenQuantGraphKernelMismatch;
            const args = got.params orelse return error.QwenQuantGraphArgumentMismatch;
            for (0..want.args.count) |i| {
                const actual: [*]const u8 = @ptrCast(args[i] orelse return error.QwenQuantGraphArgumentMismatch);
                const bytes = want.args.storage[want.args.offsets[i]..][0..want.args.sizes[i]];
                if (!std.mem.eql(u8, actual[0..bytes.len], bytes)) return error.QwenQuantGraphArgumentMismatch;
            }
        }
    };

    fn ckNode(e: *const Engine, p: CkProjection, kind: enum { quant, lane, reduce }) CkNode {
        const m = e.modelopt.?;
        const q = p.weight;
        const sk = cuda.qmmf.splitK(q.n, q.k);
        const cluster = sk > 1 and sk <= 8 and m.ck.clusters;
        var node: CkNode = .{ .function = null, .grid = .{ 1, 1, 1 } };
        switch (kind) {
            .quant => {
                node.function = m.ck.quant8_fn.handle;
                node.grid[0] = (p.rows * (q.k / 16) + 127) / 128;
                node.args.add(p.x);
                for ([_]usize{ q.k, p.rows, q.k }) |v| node.args.add(@as(i32, @intCast(v)));
                node.args.add(cuda.ck.inverse(q.act));
                node.args.add(m.rows);
            },
            .lane => {
                node.function = m.ck.lane[@intFromBool(cluster)].handle;
                node.grid = .{ (q.n + 63) / 64, 1, sk };
                node.shared = 4 * (16 * 64 + 64 * 64);
                node.args.add(m.rows);
                node.args.add(q.codes);
                node.args.add(cuda.ck.alpha(q.act, q.scale));
                node.args.add(p.out);
                node.args.add(m.part);
                for ([_]usize{ p.rows, q.n, q.k, sk, 1 }) |v| node.args.add(@as(i32, @intCast(v)));
            },
            .reduce => {
                node.function = m.ck.reduce.handle;
                node.grid[0] = (p.rows * q.n + 255) / 256;
                node.threads = 256;
                node.args.add(m.part);
                node.args.add(p.out);
                node.args.add(@as(i64, @intCast(p.rows * q.n)));
                node.args.add(@as(i32, @intCast(sk)));
                node.args.add(cuda.ck.alpha(q.act, q.scale));
            },
        }
        return node;
    }

    const GetDependencies = *const fn (cuda.abi.GraphNode, ?[*]cuda.abi.GraphNode, *usize) callconv(.c) cuda.abi.Result;

    fn orderedGraphNodes(e: *const Engine, graph: cuda.graph.Graph, node_count: usize) ![]cuda.graph.Node {
        var lib = e.d.lib;
        const get = lib.lookup(GetDependencies, "cuGraphNodeGetDependencies") orelse return error.MissingSymbol;
        return e.orderedGraphNodesWith(graph, node_count, get);
    }

    fn orderedGraphNodesWith(e: *const Engine, graph: cuda.graph.Graph, node_count: usize, get: GetDependencies) ![]cuda.graph.Node {
        const nodes = try e.gpa.alloc(cuda.graph.Node, node_count);
        defer e.gpa.free(nodes);
        if ((try graph.nodes(nodes)).len != node_count) return error.QwenQuantGraphWorkCountMismatch;
        const previous = try e.gpa.alloc(cuda.graph.Node, node_count);
        defer e.gpa.free(previous);
        for (nodes, previous) |node, *pred| {
            var deps: [2]cuda.graph.Node = .{ null, null };
            var count: usize = deps.len;
            try e.d.check(get(node, &deps, &count), "cuGraphNodeGetDependencies");
            if (count > 1) return error.QwenQuantGraphDependencyMismatch;
            pred.* = if (count == 0) null else deps[0];
        }
        const ordered = try e.gpa.alloc(cuda.graph.Node, node_count);
        errdefer e.gpa.free(ordered);
        var prior: cuda.graph.Node = null;
        for (ordered) |*out| {
            var found: ?usize = null;
            for (previous, 0..) |pred, i| if (pred == prior) {
                if (found != null) return error.QwenQuantGraphDependencyMismatch;
                found = i;
            };
            out.* = nodes[found orelse return error.QwenQuantGraphDependencyMismatch];
            prior = out.*;
        }
        return ordered;
    }

    const TerminalReadback = struct {
        dependencies: GetDependencies,
        name: *const fn (*?[*:0]const u8, cuda.abi.Function) callconv(.c) cuda.abi.Result,

        fn load(d: *const cuda.Driver) !TerminalReadback {
            var lib = d.lib;
            return .{
                .dependencies = lib.lookup(GetDependencies, "cuGraphNodeGetDependencies") orelse return error.MissingSymbol,
                .name = lib.lookup(@FieldType(TerminalReadback, "name"), "cuFuncGetName") orelse return error.MissingSymbol,
            };
        }

        fn symbol(r: TerminalReadback, d: *const cuda.Driver, function: cuda.abi.Function) ![]const u8 {
            var value: ?[*:0]const u8 = null;
            try d.check(r.name(&value, function), "cuFuncGetName");
            return std.mem.span(value orelse return error.QwenTerminalGraphSymbolMissing);
        }
    };

    pub const TerminalGraphKernel = struct {
        node: usize,
        predecessor: usize,
        requested_function: usize,
        actual_function: usize,
        requested_symbol: []const u8,
        actual_symbol: [128]u8 = @splat(0),
        actual_symbol_len: usize,
        grid: [3]u32,
        block: [3]u32,
        shared_bytes: u32,
        argument_count: usize,
        argument_sizes: [16]usize = @splat(0),
        argument_bytes: [128]u8 = @splat(0),
        argument_bytes_len: usize = 0,
        argument_sha256: [32]u8,

        fn read(node: cuda.graph.Node, predecessor: cuda.graph.Node, got: cuda.abi.KernelNodeParams, want: *const CkNode, symbol: []const u8, actual_symbol: []const u8) !TerminalGraphKernel {
            if (got.extra != null) return error.QwenTerminalGraphArgumentMismatch;
            if (!std.mem.eql(u8, symbol, actual_symbol)) return error.QwenTerminalGraphSymbolMismatch;
            try want.check(got);
            if (actual_symbol.len > 128 or want.args.count > 16) return error.QwenTerminalGraphRosterOverflow;
            var result: TerminalGraphKernel = .{
                .node = @intFromPtr(node),
                .predecessor = @intFromPtr(predecessor),
                .requested_function = @intFromPtr(want.function),
                .actual_function = @intFromPtr(got.func),
                .requested_symbol = symbol,
                .actual_symbol_len = actual_symbol.len,
                .grid = .{ got.grid_x, got.grid_y, got.grid_z },
                .block = .{ got.block_x, got.block_y, got.block_z },
                .shared_bytes = got.shared_bytes,
                .argument_count = want.args.count,
                .argument_sha256 = undefined,
            };
            @memcpy(result.actual_symbol[0..actual_symbol.len], actual_symbol);
            for (0..want.args.count) |i| {
                const size = want.args.sizes[i];
                if (result.argument_bytes_len + size > result.argument_bytes.len) return error.QwenTerminalGraphRosterOverflow;
                result.argument_sizes[i] = size;
                const actual: [*]const u8 = @ptrCast(got.params.?[i].?);
                @memcpy(result.argument_bytes[result.argument_bytes_len..][0..size], actual[0..size]);
                result.argument_bytes_len += size;
            }
            std.crypto.hash.sha2.Sha256.hash(result.argument_bytes[0..result.argument_bytes_len], &result.argument_sha256, .{});
            return result;
        }
    };

    /// One successful capture per shape key; failed verification/instantiation never publishes a record.
    pub const TerminalGraphRecord = struct {
        shape_key: u32,
        captured_graph: usize,
        node_count: usize,
        terminal_count: usize = 0,
        enabled: bool,
        owns_head: bool,
        head_rows: usize,
        head_kind: enum { absent, lane, reduce, nvfp4_lane } = .absent,
        head_image: enum { absent, ck, qmmf } = .absent,
        head_input: u64 = 0,
        head_output: u64 = 0,
        head_weight: u64 = 0,
        head_scales: u64 = 0,
        head_scale: f32 = 0,
        head: ?TerminalGraphKernel = null,
        terminal: ?TerminalGraphKernel = null,
    };

    fn instantiateTerminalGraph(e: *Engine, graph: cuda.graph.Graph, shape: Shape, node_count: usize, readback: TerminalReadback) !cuda.graph.Exec {
        const start = e.terminal_graph_roster.items.len;
        errdefer e.terminal_graph_roster.items.len = start;
        try e.terminal_graph_roster.append(e.gpa, try e.verifyTerminalGraph(graph, shape, node_count, readback));
        return graph.instantiate();
    }

    fn verifyTerminalGraph(e: *const Engine, graph: cuda.graph.Graph, shape: Shape, node_count: usize, readback: TerminalReadback) !TerminalGraphRecord {
        const nodes = try e.orderedGraphNodesWith(graph, node_count, readback.dependencies);
        defer e.gpa.free(nodes);
        const heads: usize = if (shape.record or shape.heads_all) shape.total else 1;
        var result: TerminalGraphRecord = .{
            .shape_key = shape.key(),
            .captured_graph = @intFromPtr(graph.handle),
            .node_count = node_count,
            .enabled = shape.terminal_greedy,
            .owns_head = e.placement.ownsHead(),
            .head_rows = heads,
        };
        const strict_symbol = "tf_round_bf16_argmax";
        const strict_function = if (e.terminal) |t| t.kernels.argmax_fn.handle else null;
        var terminal_index: ?usize = null;
        for (nodes, 0..) |node, i| {
            var p: cuda.abi.KernelNodeParams = undefined;
            try e.d.check(e.d.api.cuGraphKernelNodeGetParams_v2(node, &p), "cuGraphKernelNodeGetParams");
            const name = try readback.symbol(e.d, p.func);
            if ((strict_function != null and p.func == strict_function) or std.mem.eql(u8, name, strict_symbol)) {
                result.terminal_count += 1;
                terminal_index = i;
                if (strict_function == null or p.func != strict_function or !std.mem.eql(u8, name, strict_symbol)) return error.QwenTerminalGraphFunctionMismatch;
            }
        }
        if (!shape.terminal_greedy) {
            if (result.terminal_count != 0) return error.QwenTerminalGraphWorkCountMismatch;
        } else {
            if (e.numerics != .hf_modelopt or !result.owns_head or e.terminal == null or shape.record or !shape.prompt) return error.UnsupportedQwenTerminalGreedy;
            if (result.terminal_count != 1) return error.QwenTerminalGraphWorkCountMismatch;
            if (nodes.len < 2 or terminal_index.? != nodes.len - 1) return error.QwenTerminalGraphOrderMismatch;
            const output = TerminalOutput.init(heads, e.g.vocab);
            var expected: CkNode = .{ .function = strict_function, .grid = .{ 1, 1, 1 }, .threads = 256 };
            expected.args.add(e.logits + (heads - 1) * e.g.vocab * 2);
            expected.args.add(e.logits + output.pick_at);
            expected.args.add(@as(u32, @intCast(e.g.vocab)));
            expected.args.add(@as(u32, @intCast(e.g.vocab)));
            var p: cuda.abi.KernelNodeParams = undefined;
            try e.d.check(e.d.api.cuGraphKernelNodeGetParams_v2(nodes[nodes.len - 1], &p), "cuGraphKernelNodeGetParams");
            result.terminal = try TerminalGraphKernel.read(nodes[nodes.len - 1], nodes[nodes.len - 2], p, &expected, strict_symbol, try readback.symbol(e.d, p.func));
        }
        if (result.owns_head) {
            if (e.numerics != .hf_modelopt or e.modelopt == null or e.terminal_weights != .vocabulary or e.terminal_weights.vocabulary != .linear) return error.QwenTerminalGraphHeadUnsupported;
            const weight = switch (e.terminal_weights.vocabulary.linear) {
                .fp8 => |w| if (w.act != 0) w else return error.QwenTerminalGraphHeadUnsupported,
                .nvfp4 => |w| w,
                else => return error.QwenTerminalGraphHeadUnsupported,
            };
            const head: CkProjection = .{ .weight = weight, .x = e.x + (shape.total - heads) * e.g.hidden * 2, .out = e.logits, .rows = heads, .quant = true };
            var expected: CkNode = undefined;
            var symbol: []const u8 = undefined;
            if (e.terminal_weights.vocabulary.linear == .nvfp4) {
                const plan = cuda.qmmf.plan(heads, weight.n, weight.k, e.modelopt.?.lane.clusters);
                if (weight.mode != .fp4 or weight.act != 0 or weight.codes == 0 or weight.scales == 0 or !(weight.scale > 0) or !std.math.isFinite(weight.scale) or
                    weight.n != e.g.vocab or weight.k != e.g.hidden or weight.npad != cuda.nvfp4.padded(weight.n) or weight.k % cuda.qmmf.group != 0 or
                    plan.sk != 1 or plan.fused or plan.cluster or plan.reduce) return error.QwenTerminalGraphHeadUnsupported;
                const tile: usize = if (plan.bm == 16) 0 else if (plan.bm == 32) 1 else 2;
                const symbols = [_][]const u8{
                    "_ZN7tf_qmmf11qmmf_kernelILi0ELi16ELi64ELi1ELi4ELi4ELb0ELb0ELb0EEEvPK13__nv_bfloat16PKhS5_fPvPfiiiiiii",
                    "_ZN7tf_qmmf11qmmf_kernelILi0ELi32ELi64ELi1ELi4ELi4ELb0ELb0ELb0EEEvPK13__nv_bfloat16PKhS5_fPvPfiiiiiii",
                    "_ZN7tf_qmmf11qmmf_kernelILi0ELi64ELi64ELi1ELi4ELi4ELb0ELb0ELb0EEEvPK13__nv_bfloat16PKhS5_fPvPfiiiiiii",
                };
                const row_tiles = (heads + plan.bm - 1) / plan.bm;
                const band = @max(1, @min(row_tiles, (12 << 20) / (@as(usize, plan.bm) * weight.k * 2)));
                expected = .{ .function = e.modelopt.?.lane.fns[0][tile * 2].handle, .grid = .{ row_tiles * ((weight.n + 63) / 64), 1, plan.sk }, .shared = cuda.qmmf.sharedBytes(.fp4, plan.bm) };
                // qmmf.Lane.matmulOut's BF16-output ABI, qualified here only for its unsplit lane.
                expected.args.add(head.x);
                expected.args.add(weight.codes);
                expected.args.add(weight.scales);
                expected.args.add(weight.scale);
                expected.args.add(head.out);
                expected.args.add(@as(u64, 0));
                for ([_]usize{ heads, weight.n, weight.k, plan.sk, weight.npad, weight.k, band }) |v| expected.args.add(@as(i32, @intCast(v)));
                symbol = symbols[tile];
                result.head_kind = .nvfp4_lane;
                result.head_image = .qmmf;
            } else {
                const sk = cuda.qmmf.splitK(weight.n, weight.k);
                const cluster = sk > 1 and sk <= 8 and e.modelopt.?.ck.clusters;
                const reduce = sk > 1 and !cluster;
                expected = e.ckNode(head, if (reduce) .reduce else .lane);
                symbol = if (reduce) "tf_ck8_reduce" else if (cluster) "tf_ck8_16c" else "tf_ck8_16";
                result.head_kind = if (reduce) .reduce else .lane;
                result.head_image = .ck;
            }
            const suffix: usize = @intFromBool(shape.terminal_greedy);
            if (nodes.len <= suffix) return error.QwenTerminalGraphOrderMismatch;
            const index = nodes.len - suffix - 1;
            var p: cuda.abi.KernelNodeParams = undefined;
            try e.d.check(e.d.api.cuGraphKernelNodeGetParams_v2(nodes[index], &p), "cuGraphKernelNodeGetParams");
            result.head = try TerminalGraphKernel.read(nodes[index], if (index == 0) null else nodes[index - 1], p, &expected, symbol, try readback.symbol(e.d, p.func));
            result.head_input = head.x;
            result.head_output = head.out;
            result.head_weight = head.weight.codes;
            result.head_scales = head.weight.scales;
            result.head_scale = head.weight.scale;
        }
        return result;
    }

    fn verifyQuantGraph(e: *const Engine, graph: cuda.graph.Graph, shape: Shape, node_count: usize) !usize {
        if (e.numerics != .hf_modelopt) return error.UnsupportedQwenQuantReuse;
        var plan = try e.ckPlan(shape);
        defer plan.deinit(e.gpa);
        var expected: std.ArrayList(CkNode) = .empty;
        defer expected.deinit(e.gpa);
        var quants: usize = 0;
        for (plan.items) |p| {
            if (p.quant) {
                try expected.append(e.gpa, e.ckNode(p, .quant));
                quants += 1;
            }
            try expected.append(e.gpa, e.ckNode(p, .lane));
            const sk = cuda.qmmf.splitK(p.weight.n, p.weight.k);
            if (sk > 1 and !(sk <= 8 and e.modelopt.?.ck.clusters)) try expected.append(e.gpa, e.ckNode(p, .reduce));
        }
        const nodes = try e.orderedGraphNodes(graph, node_count);
        defer e.gpa.free(nodes);
        const c = e.modelopt.?.ck;
        var count: usize = 0;
        for (nodes) |node| {
            var p: cuda.abi.KernelNodeParams = undefined;
            try e.d.check(e.d.api.cuGraphKernelNodeGetParams_v2(node, &p), "cuGraphKernelNodeGetParams");
            var relevant = p.func == c.quant8_fn.handle or p.func == c.reduce.handle;
            for (c.lane) |f| relevant = relevant or p.func == f.handle;
            if (!relevant) continue;
            if (count >= expected.items.len) return error.QwenQuantGraphWorkCountMismatch;
            try expected.items[count].check(p);
            count += 1;
        }
        if (count != expected.items.len) return error.QwenQuantGraphWorkCountMismatch;
        return quants;
    }

    fn verifyExpertGraph(e: *Engine, graph: cuda.graph.Graph, shape: Shape, node_count: usize) !usize {
        const roster_start = e.expert_graph_roster.items.len;
        errdefer e.expert_graph_roster.items.len = roster_start;
        const bindings = e.modelopt orelse return error.UnsupportedQwenExpertUp;
        const experts = bindings.nvfp4_experts;
        const n8 = experts.n8_up orelse return error.CandidateNotLoaded;
        const baseline = experts.fns[0].handle;
        const candidate = n8.function.handle;
        if (baseline == candidate) return error.QwenExpertGraphKernelMismatch;
        const legacy_split_function = if (experts.split_m_up) |v| v.function.handle else null;
        const slice_function = if (experts.slice_major_up) |v| v.function.handle else null;
        const split = if (shape.expert_up_layout == .slice_major_v1) experts.slice_major_up else experts.split_m_up;
        const expected = if (shape.expert_up_split_m) (split orelse return error.UnsupportedQwenExpertUp).function.handle else if (shape.expert_up_n8) candidate else baseline;
        const storage = try e.gpa.alloc(cuda.graph.Node, node_count);
        defer e.gpa.free(storage);
        const nodes = try graph.nodes(storage);
        if (nodes.len != node_count) return error.QwenExpertGraphWorkCountMismatch;
        var layers_seen: [geometry.max_layers]bool = @splat(false);
        var count: usize = 0;
        for (nodes) |node| {
            var params: cuda.abi.KernelNodeParams = undefined;
            try e.d.check(e.d.api.cuGraphKernelNodeGetParams_v2(node, &params), "cuGraphKernelNodeGetParams");
            if (params.func != baseline and params.func != candidate and params.func != legacy_split_function and params.func != slice_function) continue;
            if (params.func != expected) return error.QwenExpertGraphKernelMismatch;
            const args = params.params orelse return error.QwenExpertGraphMissingArguments;
            const up: *const u64 = @ptrCast(@alignCast(args[3] orelse return error.QwenExpertGraphMissingArguments));
            var matched = false;
            for (e.layers, 0..) |layer, index| {
                if (layer.mlp != .moe or layer.mlp.moe.experts.up != up.*) continue;
                if (layers_seen[index]) return error.QwenExpertGraphLayerMismatch;
                const m = layer.mlp.moe;
                try cuda.experts.Nvfp4Experts.validateUpGeometry(m.experts);
                if (m.experts.nvfp4_up_layout != shape.expert_up_layout) return error.QwenExpertUpLayoutMismatch;
                if (shape.expert_up_layout == .slice_major_v1 and !shape.expert_up_split_m) return error.QwenExpertGraphKernelMismatch;
                const slots = if (m.shared == .projected) e.g.top_k else e.g.slots();
                const expert_count = if (m.shared == .projected) e.g.experts else e.g.experts + 1;
                const items = cuda.grouped.maxItems(@as(usize, shape.total) * slots, expert_count, 16);
                const nb = m.experts.width / cuda.experts.Nvfp4Experts.cols;
                const grid = if (shape.expert_up_split_m) split.?.grid(items, nb) else if (shape.expert_up_n8) n8.grid(items, nb) else cuda.experts.Nvfp4Experts.decodeGridWarps(.baseline, items, nb, experts.resident[0], experts.warps);
                const threads = if (shape.expert_up_split_m) cuda.experts.Nvfp4Experts.SplitMUp.threads else if (shape.expert_up_n8) cuda.experts.Nvfp4Experts.N8Up.threads else experts.warps * 32;
                if (params.grid_x != grid or params.grid_y != 1 or params.grid_z != 1 or params.block_x != threads or
                    params.block_y != 1 or params.block_z != 1 or params.shared_bytes != 0) return error.QwenExpertGraphKernelMismatch;
                const plan: cuda.grouped.Plan = .{ .members = e.routed.members, .items = e.routed.items, .counts = e.routed.counts, .rank = e.routed.rank, .hist = e.routed.hist };
                const expected_args = cuda.experts.Nvfp4Experts.decodeArgs(.up, .{ .x = e.x, .stride = e.g.hidden, .slots = slots }, m.experts, plan, e.routed.act, -1);
                var digest = std.crypto.hash.sha2.Sha256.init(.{});
                var argument_bytes: usize = 0;
                for (0..expected_args.count) |a| {
                    const actual: [*]const u8 = @ptrCast(args[a] orelse return error.QwenExpertGraphMissingArguments);
                    const bytes = expected_args.storage[expected_args.offsets[a]..][0..expected_args.sizes[a]];
                    if (!std.mem.eql(u8, actual[0..bytes.len], bytes)) return error.QwenExpertGraphArgumentMismatch;
                    digest.update(actual[0..bytes.len]);
                    argument_bytes += bytes.len;
                }
                var hash: [32]u8 = undefined;
                digest.final(&hash);
                try e.expert_graph_roster.append(e.gpa, .{
                    .shape_key = shape.key(),
                    .total = shape.total,
                    .heads_all = shape.heads_all,
                    .layer_local = index,
                    .layer_global = e.placement.first + index,
                    .up_layout = shape.expert_up_layout,
                    .arm = if (params.func == slice_function) .slice_major else if (params.func == legacy_split_function) .split_m else if (params.func == candidate) .n8 else .baseline,
                    .grid = .{ params.grid_x, params.grid_y, params.grid_z },
                    .block = .{ params.block_x, params.block_y, params.block_z },
                    .shared_bytes = params.shared_bytes,
                    .abi_count = expected_args.count,
                    .argument_bytes = argument_bytes,
                    .args_sha256 = std.fmt.bytesToHex(hash, .lower),
                });
                layers_seen[index] = true;
                matched = true;
                count += 1;
                break;
            }
            if (!matched) return error.QwenExpertGraphLayerMismatch;
        }
        if (count != e.g.layers) return error.QwenExpertGraphWorkCountMismatch;
        return count;
    }

    pub fn commit(parts: []const Part, record: bool) void {
        for (parts) |p| {
            p.cache.last = if (record and p.rows > 1) .{ .start = p.cache.len, .rows = p.rows } else null;
            p.cache.len += p.rows;
            p.cache.written = p.cache.len;
            if (p.cache.terminal_greedy) |pick| {
                if (record or pick.position != p.cache.len) p.cache.terminal_greedy = null;
            }
        }
    }

    pub fn finish(e: *Engine, parts: []const Part, keep: Keep, shape: Shape) !void {
        if (e.numerics == .hf_dense_bf16 and (keep != .hidden or !shape.normalized_hidden)) return error.QwenHiddenOnly;
        for (parts) |p| p.cache.terminal_greedy = null;
        if (!e.placement.ownsHead()) {
            try e.stream.synchronize();
            return;
        }
        const g = e.g;
        var total: usize = 0;
        for (parts) |p| total += p.rows;
        const heads: usize = if (shape.record or shape.heads_all) total else 1;
        switch (keep) {
            .hidden => |w| {
                if (!shape.normalized_hidden) return error.InvalidQwenHiddenShape;
                try w.copy(e.x);
            },
            .draws => |pk| {
                const toks = e.tables.ptr + tokens_at;
                var greedy: usize = 0; // the run of greedy rows ending at i, argmaxed together
                for (pk, 0..) |p, i| {
                    if (p.rule == null) {
                        greedy += 1;
                        continue;
                    }
                    try e.argmax(i - greedy, greedy, toks);
                    greedy = 0;
                    try e.draws.draw(e.stream, e.logits + i * g.vocab * 2, g.vocab, e.tables.ptr + rules_at + i * @sizeOf(cuda.sample.Rule), e.tables.ptr + meta_at + i * 4, 0, toks + i * 4, 1, null, null);
                }
                try e.argmax(pk.len - greedy, greedy, toks);
                try e.d.check(e.d.api.cuMemcpyDtoHAsync_v2(e.host_logits.bytes.ptr, toks, total * 4, e.stream.handle), "cuMemcpyDtoHAsync");
                try e.stream.synchronize();
                @memcpy(e.tokens[0..total], e.host_logits.slice(u32)[0..total]);
            },
            .learn => {},
            .logits, .last, .prompt_logits => {
                const output = TerminalOutput.init(heads, g.vocab);
                const out = e.host_logits.bytes[0..if (shape.terminal_greedy) output.bytes else output.logits_bytes];
                try e.d.check(e.d.api.cuMemcpyDtoHAsync_v2(out.ptr, e.logits, out.len, e.stream.handle), "cuMemcpyDtoHAsync");
                try e.stream.synchronize();
                const token = if (shape.terminal_greedy) try output.token(out, g.vocab) else null;
                if (keep == .last or keep == .prompt_logits) {
                    const cache = parts[0].cache;
                    @memcpy(cache.logits, try e.rowLogits(heads - 1));
                    if (token) |id| cache.terminal_greedy = .{ .position = cache.len + parts[0].rows, .token = id };
                }
            },
        }
    }

    /// A forward's shape, which fixes its launches: its tables and the caches' lengths are read on the device.
    pub const Shape = struct {
        total: u8,
        nseg: u8,
        record: bool,
        prompt: bool,
        taps: bool,
        gdn_layout: Cache.GdnLayout = .value_key,
        expert_up_n8: bool = false,
        expert_up_split_m: bool = false,
        expert_up_layout: cuda.experts.Nvfp4UpLayout = .lane_major,
        reuse_fp8_quant: bool = false,
        heads_all: bool = false,
        terminal_greedy: bool = false,
        moe_math: MoeMath = .legacy,
        attention_mode: AttentionMode = .reference,
        normalized_hidden: bool = false,
        slots: u8 = 1,

        pub fn init(rows: usize, segments: usize, record: bool, taps: bool, slots: u8) Shape {
            std.debug.assert(rows > 0 and rows <= most_verify and segments > 0 and segments <= rows);
            const bucket: usize = if (rows == 1) 1 else std.mem.alignForward(usize, rows, 16);
            const windows = std.math.ceilPowerOfTwo(usize, segments) catch unreachable;
            return .{ .total = @intCast(if (record) bucket else rows), .nseg = @intCast(if (record) windows else segments), .record = record, .prompt = !record, .taps = taps, .slots = slots };
        }

        pub fn key(s: Shape) u32 {
            return @as(u32, s.total) | @as(u32, s.nseg) << 8 | @as(u32, @intFromBool(s.record)) << 16 | @as(u32, @intFromBool(s.prompt)) << 17 | @as(u32, @intFromBool(s.taps)) << 18 | @as(u32, @backingInt(s.gdn_layout)) << 19 | @as(u32, @intFromBool(s.expert_up_n8)) << 20 | @as(u32, @intFromBool(s.heads_all)) << 21 | @as(u32, @intFromBool(s.expert_up_split_m)) << 22 | @as(u32, @intFromBool(s.reuse_fp8_quant)) << 23 | @as(u32, @backingInt(s.expert_up_layout)) << 24 | @as(u32, @intFromBool(s.terminal_greedy)) << 25 | @as(u32, @backingInt(s.moe_math)) << 26 | @as(u32, @backingInt(s.attention_mode)) << 27 | @as(u32, @intFromBool(s.normalized_hidden)) << 28 | @as(u32, @ctz(@as(u32, s.slots))) << 29;
        }
    };

    /// The same scan selected by body, exposed for host launch-contract fixtures.
    pub fn deltaScan(e: *Engine, record: bool, slot_count: u8, segments: usize, layer: c_int) !void {
        const scan = if (!record) e.fns.gdn_chain else switch (slot_count) {
            1 => e.fns.gdn_slots_1,
            2 => e.fns.gdn_slots_2,
            4 => e.fns.gdn_slots_4,
            8 => e.fns.gdn_slots_8,
            16 => e.fns.gdn_slots_16,
            else => return error.InvalidGdnSlots,
        };
        const input = e.tables.ptr + @as(usize, if (record) Round.steps_at else rowpar_at);
        try e.run(scan, .{ 1, geometry.dk / 4, e.g.v_heads * segments }, .{ 32, 4, 1 }, .{ e.geo, e.q, e.k, e.v, e.g_, e.beta, e.tables.ptr + segs_at, input, layer, e.y, e.tables.ptr + params_at });
    }

    pub fn mark(e: *Engine, phase: Phase) !void {
        if (e.observer) |o| try o.mark(o.ptr, phase, e.observed_layer);
    }

    pub fn moeMath(e: *const Engine) MoeMath {
        return if (e.numerics == .hf_modelopt) if (e.modelopt) |m| m.router.math else .legacy else .legacy;
    }

    pub fn attentionMode(e: *const Engine) AttentionMode {
        return if (e.numerics == .hf_modelopt) e.attention_mode else .reference;
    }

    /// The launches of a forward of `shape`, from the embedding to the kept rows' logits (captured once a shape).
    fn body(e: *Engine, shape: Shape) !void {
        const g = e.g;
        const total: usize = shape.total;
        const nseg: usize = shape.nseg;
        const record = shape.record;
        const prompt = shape.prompt;
        const heads: usize = if (record or shape.heads_all) total else 1;
        const t_ids = e.tables.ptr + ids_at;
        const t_segs = e.tables.ptr + segs_at;
        const t_rows = e.tables.ptr + rowseg_at;
        const t_par = e.tables.ptr + rowpar_at;
        const t_anc = e.tables.ptr + rowanc_at;
        const t_params = e.tables.ptr + params_at;
        const geo = e.geo;
        const group = g.q_heads / g.kv_heads;
        const is_hf = hfOrder(e.numerics);
        const launch: hf.Launches = .{};
        const attn_mode: AttentionMode = if (is_hf) shape.attention_mode else .chunked;
        const attn = hf.attnLaunch(attn_mode, g.q_heads, g.kv_heads, total, group, max_chunks);
        const gdn_chain = switch (shape.gdn_layout) {
            .value_key => e.fns.gdn_chain,
            .key_value => e.modelopt.?.gdn_chain_transposed,
        };

        e.observed_layer = e.placement.first;
        try e.mark(.begin);
        if (e.placement.ownsEmbedding() and e.numerics != .hf_dense_bf16) try e.embed(t_ids, e.h, total, t_params);
        // an image chunk's rows take the tower's features, and rotate at their staged t, h, w positions
        const mpos: u64 = if (e.features != 0) e.tables.ptr + mpos_at else 0;
        if (e.features != 0) try e.run(e.fns.splice, .{ total, 1, 1 }, .{ 256, 1, 1 }, .{ e.h, e.features, e.tables.ptr + src_at, @as(c_int, @intCast(g.hidden)), t_params });
        try e.mark(.embedding);
        var di: usize = 0;
        var ai: c_int = 0;
        for (e.layers, 0..) |l, i| {
            e.observed_layer = e.placement.first + i;
            try e.norm(e.placement.first + i != 0, l.input_norm, total);
            try e.mark(.input_norm);
            switch (l.mixer) {
                .delta => |m| {
                    const layer: c_int = @intCast(di);
                    if (shape.reuse_fp8_quant) {
                        _ = try e.modelopt.?.linear().decodeTogether(e.stream, &.{ .{ .weight = m.qkv.linear, .output = e.qkv }, .{ .weight = m.z.linear, .output = e.z } }, .{ .x = e.x, .part = e.modelopt.?.part }, total);
                    } else {
                        try e.proj(m.qkv, e.x, e.qkv, total, prompt);
                        try e.proj(m.z, e.x, e.z, total, prompt);
                    }
                    try e.proj(m.a, e.x, e.a, total, prompt);
                    try e.proj(m.b, e.x, e.b, total, prompt);
                    try e.mark(.mixer_inputs);
                    try e.run(e.fns.gdn_pre, .{ 1, 2 * g.k_heads + g.v_heads, total }, .{ if (is_hf) launch.gdn_pre else 32, 1, 1 }, .{ geo, e.qkv, t_segs, t_rows, t_par, layer, m.conv, e.a, e.b, m.a_log, m.dt_bias, e.q, e.k, e.v, e.g_, e.beta, t_params });
                    try e.mark(.gdn_pre);
                    try e.run(e.fns.conv_commit, .{ (g.convWidth() + 255) / 256, nseg, 1 }, .{ 256, 1, 1 }, .{ geo, e.qkv, t_segs, t_par, layer, t_params });
                    try e.mark(.conv_commit);
                    if (is_hf and total > 1) {
                        for (0..total) |row| try e.run(gdn_chain, .{ 1, launch.gdn_chain_grid_y, g.v_heads }, launch.gdn_chain, .{ geo, e.q, e.k, e.v, e.g_, e.beta, e.tables.ptr + chain_segs_at + row * @sizeOf(Seg), t_par, layer, e.y, t_params });
                    } else if (is_hf) {
                        try e.run(gdn_chain, .{ 1, launch.gdn_chain_grid_y, g.v_heads * nseg }, launch.gdn_chain, .{ geo, e.q, e.k, e.v, e.g_, e.beta, t_segs, t_par, layer, e.y, t_params });
                    } else try e.deltaScan(record, shape.slots, nseg, layer);
                    try e.mark(.gdn_chain);
                    try e.run(e.fns.gdn_post, .{ 1, g.v_heads, total }, .{ if (is_hf) launch.gdn_post else 32, 1, 1 }, .{ geo, e.y, e.z, m.norm, g.eps, e.mix, t_params });
                    try e.mark(.gdn_post);
                    try e.proj(m.out, e.mix, e.r, total, prompt);
                    di += 1;
                },
                .attention => |m| {
                    const aw = g.attnWidth();
                    if (shape.reuse_fp8_quant) {
                        _ = try e.modelopt.?.linear().decodeTogether(e.stream, &.{ .{ .weight = m.q.linear, .output = e.qraw }, .{ .weight = m.k.linear, .output = e.kraw }, .{ .weight = m.v.linear, .output = e.vraw } }, .{ .x = e.x, .part = e.modelopt.?.part }, total);
                    } else {
                        try e.proj(m.q, e.x, e.qraw, total, prompt);
                        try e.proj(m.k, e.x, e.kraw, total, prompt);
                        try e.proj(m.v, e.x, e.vraw, total, prompt);
                    }
                    try e.mark(.mixer_inputs);
                    try e.run(e.fns.head_norm, .{ total * g.q_heads, 1, 1 }, .{ if (is_hf) launch.head_norm else 64, 1, 1 }, .{ e.qraw, m.q_norm, e.qn, @as(c_int, @intCast(g.q_heads)), @as(c_int, @intCast(2 * aw)), @as(c_int, 2 * head_dim), g.eps, t_params });
                    try e.run(e.fns.head_norm, .{ total * g.kv_heads, 1, 1 }, .{ if (is_hf) launch.head_norm else 64, 1, 1 }, .{ e.kraw, m.k_norm, e.kn, @as(c_int, @intCast(g.kv_heads)), @as(c_int, @intCast(g.kv_heads * head_dim)), @as(c_int, head_dim), g.eps, t_params });
                    try e.run(if (e.kv8) e.fp8.rope else e.fns.rope, .{ total, 1, 1 }, .{ head_dim, 1, 1 }, .{ geo, e.qn, e.kn, e.vraw, t_segs, t_rows, t_anc, ai, t_params, mpos });
                    try e.run(if (e.kv8) e.fp8.attn_partial else e.fns.attn_partial, attn.partial_grid, attn.partial_block, .{ geo, e.qn, t_segs, t_rows, t_anc, ai, t_params, @as(f32, 1.0 / 16.0), e.pm, e.pl, e.po });
                    try e.run(e.fns.attn_merge, .{ g.q_heads, total, 1 }, .{ attn.merge_block, 1, 1 }, .{ geo, e.pm, e.pl, e.po, t_segs, t_rows, t_anc, t_params, e.y });
                    try e.run(e.fns.attn_gate, .{ (total * aw + 255) / 256, 1, 1 }, .{ 256, 1, 1 }, .{ geo, e.y, e.qraw, e.mix, @as(c_int, @intCast(total)), t_params });
                    try e.mark(.attention);
                    try e.proj(m.o, e.mix, e.r, total, prompt);
                    ai += 1;
                },
            }
            try e.mark(.mixer_output);
            try e.norm(true, l.post_norm, total);
            try e.mark(.post_norm);
            switch (l.mlp) {
                .dense => |m| {
                    try e.proj(m.gate, e.x, e.gate, total, prompt);
                    try e.proj(m.up, e.x, e.up, total, prompt);
                    const n: c_longlong = @intCast(total * g.intermediate);
                    try e.run(e.fns.mlp_act, .{ (total * g.intermediate + 255) / 256, 1, 1 }, .{ 256, 1, 1 }, .{ e.gate, e.up, e.act, n, t_params, @as(c_int, @intCast(g.intermediate)) });
                    try e.proj(m.down, e.act, e.r, total, prompt);
                    if (e.mlp) |k| try k.f(k.ctx, i, e.x, e.r, total, e.taught) else try e.pockets.add(e.stream, i, e.x, e.r, total, e.taught);
                    try e.mark(.dense);
                },
                .moe => |m| try routed.rows(e, m.router, m.experts, m.shared, e.x, e.r, total, @import("cuda_forward.zig").promptMoe(prompt, total), shape.expert_up_n8, shape.expert_up_split_m),
            }
            if (shape.taps) for (e.taps[0..e.n_taps], 0..) |t, ti| if (t == e.placement.first + i) {
                try e.run(e.fns.tap, .{ total, 1, 1 }, .{ 256, 1, 1 }, .{ e.h, e.r, t_segs, t_rows, @as(c_int, @intCast(ti)), @as(c_int, @intCast(g.hidden)), t_params });
            };
        }
        if (!e.placement.ownsHead()) return;
        try e.norm(true, e.final_norm, total);
        if (shape.normalized_hidden) return;
        // the head over the kept rows
        try e.proj(try e.vocabulary(), e.x + (total - heads) * g.hidden * 2, e.logits, heads, false);
        if (shape.terminal_greedy) try e.pickTerminal(heads);
        try e.mark(.head);
    }

    fn pickTerminal(e: *Engine, heads: usize) !void {
        const output = TerminalOutput.init(heads, e.g.vocab);
        try e.terminal.?.kernels.argmax(e.stream, e.logits + (heads - 1) * e.g.vocab * 2, e.logits + output.pick_at, 1, e.g.vocab, e.g.vocab);
    }

    pub fn routedRows(e: *Engine, router: u64, layer: cuda.experts.Layer, shared: layers.SharedExpert, x: u64, out: u64, total: usize, wide_rows: bool) !void {
        if (e.numerics == .hf_dense_bf16) return error.UnsupportedQwenExperts;
        return routed.rows(e, router, layer, shared, x, out, total, wide_rows, false, false);
    }

    pub fn vocabulary(e: *const Engine) !Proj {
        return switch (e.terminal_weights) {
            .vocabulary => |p| p,
            .hidden_only => error.QwenHiddenOnly,
            .none => error.QwenVocabularyNotLoaded,
        };
    }

    /// Row `i`'s logits from the newest forward that kept logits.
    pub fn rowLogits(e: *const Engine, i: usize) ![]const u16 {
        if (e.numerics == .hf_dense_bf16) return error.QwenHiddenOnly;
        if (i >= most_verify) return error.InvalidQwenLogitRow;
        return e.host_logits.slice(u16)[i * e.g.vocab ..][0..e.g.vocab];
    }
};

test "terminal capture readback checks FP8 and real NVFP4 ABI, final head producer, ordering and absent arms" {
    const Fake = struct {
        var params: [4]cuda.abi.KernelNodeParams = undefined;
        var count: usize = 3;
        var predecessors: [4]usize = .{ 0, 1, 2, 3 };
        var branch = false;
        var wrong_name = false;
        var instantiate_fail = false;
        var instantiate_count: usize = 0;
        var lane_args: cuda.Args = .{};

        fn launch(f: cuda.abi.Function, gx: c_uint, gy: c_uint, gz: c_uint, bx: c_uint, by: c_uint, bz: c_uint, shared: c_uint, _: cuda.abi.Stream, args: ?[*]?*anyopaque, extra: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
            lane_args = .{};
            inline for (0..13) |i| {
                if (i == 0 or i == 1 or i == 2 or i == 4 or i == 5) {
                    lane_args.add(@as(*const u64, @ptrCast(@alignCast(args.?[i].?))).*);
                } else {
                    lane_args.add(@as(*const u32, @ptrCast(@alignCast(args.?[i].?))).*);
                }
            }
            params[1] = .{ .func = f, .grid_x = gx, .grid_y = gy, .grid_z = gz, .block_x = bx, .block_y = by, .block_z = bz, .shared_bytes = shared, .params = lane_args.pointers(), .extra = extra };
            return cuda.abi.success;
        }

        fn nodes(_: cuda.abi.Graph, out: ?[*]cuda.graph.Node, n: *usize) callconv(.c) cuda.abi.Result {
            if (out) |dst| for (0..@min(n.*, count)) |i| {
                dst[i] = @ptrFromInt(count - i);
            };
            n.* = count;
            return cuda.abi.success;
        }
        fn get(node: cuda.graph.Node, out: *cuda.abi.KernelNodeParams) callconv(.c) cuda.abi.Result {
            out.* = params[@intFromPtr(node) - 1];
            return cuda.abi.success;
        }
        fn dependencies(node: cuda.graph.Node, out: ?[*]cuda.graph.Node, n: *usize) callconv(.c) cuda.abi.Result {
            const pred = predecessors[@intFromPtr(node) - 1];
            n.* = if (branch) 2 else if (pred == 0) 0 else 1;
            if (out) |dst| if (pred != 0) {
                dst[0] = @ptrFromInt(pred);
            };
            return cuda.abi.success;
        }
        fn name(out: *?[*:0]const u8, f: cuda.abi.Function) callconv(.c) cuda.abi.Result {
            out.* = switch (@intFromPtr(f)) {
                10, 99 => if (wrong_name) "wrong_argmax" else "tf_round_bf16_argmax",
                20 => "tf_ck8_16",
                21 => "tf_ck8_16c",
                22 => "tf_ck8_reduce",
                40 => "_ZN7tf_qmmf11qmmf_kernelILi0ELi16ELi64ELi1ELi4ELi4ELb0ELb0ELb0EEEvPK13__nv_bfloat16PKhS5_fPvPfiiiiiii",
                42 => "_ZN7tf_qmmf11qmmf_kernelILi0ELi32ELi64ELi1ELi4ELi4ELb0ELb0ELb0EEEvPK13__nv_bfloat16PKhS5_fPvPfiiiiiii",
                44 => "_ZN7tf_qmmf11qmmf_kernelILi0ELi64ELi64ELi1ELi4ELi4ELb0ELb0ELb0EEEvPK13__nv_bfloat16PKhS5_fPvPfiiiiiii",
                else => "body",
            };
            return cuda.abi.success;
        }
        fn from(want: *Engine.CkNode) cuda.abi.KernelNodeParams {
            return .{ .func = want.function, .grid_x = @intCast(want.grid[0]), .grid_y = @intCast(want.grid[1]), .grid_z = @intCast(want.grid[2]), .block_x = @intCast(want.threads), .block_y = 1, .block_z = 1, .shared_bytes = @intCast(want.shared), .params = want.args.pointers(), .extra = null };
        }
        fn instantiate(out: *cuda.abi.GraphExec, _: cuda.abi.Graph, _: c_ulonglong) callconv(.c) cuda.abi.Result {
            instantiate_count += 1;
            if (instantiate_fail) return cuda.abi.error_not_ready;
            out.* = @ptrFromInt(200);
            return cuda.abi.success;
        }
    };
    var driver: cuda.Driver = undefined;
    driver.api.cuGraphGetNodes = Fake.nodes;
    driver.api.cuGraphKernelNodeGetParams_v2 = Fake.get;
    driver.api.cuGraphInstantiateWithFlags = Fake.instantiate;
    var e: Engine = undefined;
    e.d = &driver;
    e.gpa = std.testing.allocator;
    e.terminal_graph_roster = .empty;
    defer e.terminal_graph_roster.deinit(e.gpa);
    e.numerics = .hf_modelopt;
    e.placement = .{ .first = 0, .end = 1, .total = 1 };
    e.g = undefined;
    e.g.hidden = 2048;
    e.g.vocab = 248320;
    e.x = 4096;
    e.logits = 8192;
    e.modelopt = @as(modelopt.Bindings, undefined);
    e.modelopt.?.rows = 16384;
    e.modelopt.?.part = 32768;
    e.modelopt.?.ck.lane[0] = .{ .d = &driver, .handle = @ptrFromInt(20) };
    e.modelopt.?.ck.lane[1] = .{ .d = &driver, .handle = @ptrFromInt(21) };
    e.modelopt.?.ck.reduce = .{ .d = &driver, .handle = @ptrFromInt(22) };
    e.terminal = .{ .module = undefined, .kernels = undefined };
    e.terminal.?.kernels.argmax_fn = .{ .d = &driver, .handle = @ptrFromInt(10) };
    const graph: cuda.graph.Graph = .{ .d = &driver, .handle = @ptrFromInt(100) };
    const readback: Engine.TerminalReadback = .{ .dependencies = Fake.dependencies, .name = Fake.name };
    for ([_]usize{ 1, 16 }) |rows| for ([_]bool{ false, true }) |all| for ([_]usize{ 248320, 64 }) |width| for ([_]bool{ false, true }) |cluster_enabled| {
        e.modelopt.?.ck.clusters = cluster_enabled;
        e.g.vocab = width;
        e.terminal_weights = .{ .vocabulary = .{ .linear = .{ .fp8 = .{ .mode = .fp8, .codes = 65536, .scales = 0, .act = 0.5, .n = @intCast(width), .npad = @intCast(width), .k = 2048 } } } };
        const heads: usize = if (all) rows else 1;
        const output = TerminalOutput.init(heads, width);
        const projection: Engine.CkProjection = .{ .weight = e.terminal_weights.vocabulary.linear.fp8, .x = e.x + (rows - heads) * e.g.hidden * 2, .out = e.logits, .rows = heads, .quant = true };
        const sk = cuda.qmmf.splitK(width, 2048);
        var head = e.ckNode(projection, if (sk > 1 and !cluster_enabled) .reduce else .lane);
        var terminal: Engine.CkNode = .{ .function = @ptrFromInt(10), .grid = .{ 1, 1, 1 }, .threads = 256 };
        terminal.args.add(e.logits + (heads - 1) * width * 2);
        terminal.args.add(e.logits + output.pick_at);
        terminal.args.add(@as(u32, @intCast(width)));
        terminal.args.add(@as(u32, @intCast(width)));
        var body: Engine.CkNode = .{ .function = @ptrFromInt(30), .grid = .{ 1, 1, 1 } };
        Fake.params = .{ Fake.from(&body), Fake.from(&head), Fake.from(&terminal), Fake.from(&body) };
        Fake.count = 3;
        Fake.predecessors = .{ 0, 1, 2, 3 };
        Fake.branch = false;
        Fake.wrong_name = false;
        var shape: Engine.Shape = .{ .total = @intCast(rows), .nseg = 1, .record = false, .prompt = true, .taps = false, .heads_all = all, .terminal_greedy = true };
        const record = try e.verifyTerminalGraph(graph, shape, 3, readback);
        e.terminal_graph_roster.clearRetainingCapacity();
        Fake.instantiate_count = 0;
        Fake.instantiate_fail = true;
        try std.testing.expectError(error.NotReady, e.instantiateTerminalGraph(graph, shape, 3, readback));
        try std.testing.expectEqual(0, e.terminal_graph_roster.items.len);
        Fake.instantiate_fail = false;
        const exec = try e.instantiateTerminalGraph(graph, shape, 3, readback);
        try std.testing.expectEqual(@as(usize, 200), @intFromPtr(exec.handle));
        try std.testing.expectEqual(1, e.terminal_graph_roster.items.len);
        try std.testing.expectEqual(2, Fake.instantiate_count);
        Fake.wrong_name = true;
        try std.testing.expectError(error.QwenTerminalGraphFunctionMismatch, e.instantiateTerminalGraph(graph, shape, 3, readback));
        try std.testing.expectEqual(1, e.terminal_graph_roster.items.len);
        try std.testing.expectEqual(2, Fake.instantiate_count);
        Fake.wrong_name = false;
        try std.testing.expectEqual(shape.key(), record.shape_key);
        try std.testing.expectEqual(100, record.captured_graph);
        try std.testing.expectEqual(1, record.terminal_count);
        try std.testing.expectEqual(2, record.terminal.?.predecessor);
        try std.testing.expectEqual(2, record.head.?.node);
        try std.testing.expectEqual(heads, record.head_rows);
        try std.testing.expectEqual(4, record.terminal.?.argument_count);
        try std.testing.expectEqual(24, record.terminal.?.argument_bytes_len);
        try std.testing.expectEqualSlices(usize, &.{ 8, 8, 4, 4 }, record.terminal.?.argument_sizes[0..4]);
        try std.testing.expectEqualStrings("tf_round_bf16_argmax", record.terminal.?.requested_symbol);
        try std.testing.expectEqualStrings(record.terminal.?.requested_symbol, record.terminal.?.actual_symbol[0..record.terminal.?.actual_symbol_len]);
        var hash: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(record.terminal.?.argument_bytes[0..24], &hash, .{});
        try std.testing.expectEqualSlices(u8, &hash, &record.terminal.?.argument_sha256);

        for (0..4) |arg| {
            terminal.args.storage[terminal.args.offsets[arg]] ^= 1;
            try std.testing.expectError(error.QwenQuantGraphArgumentMismatch, e.verifyTerminalGraph(graph, shape, 3, readback));
            terminal.args.storage[terminal.args.offsets[arg]] ^= 1;
        }
        for (0..head.args.count) |arg| {
            head.args.storage[head.args.offsets[arg]] ^= 1;
            try std.testing.expectError(error.QwenQuantGraphArgumentMismatch, e.verifyTerminalGraph(graph, shape, 3, readback));
            head.args.storage[head.args.offsets[arg]] ^= 1;
        }
        inline for (.{ "grid_x", "grid_y", "grid_z", "block_x", "block_y", "block_z", "shared_bytes" }) |field| {
            @field(Fake.params[2], field) += 1;
            try std.testing.expectError(error.QwenQuantGraphKernelMismatch, e.verifyTerminalGraph(graph, shape, 3, readback));
            @field(Fake.params[2], field) -= 1;
        }
        Fake.params[2].params = null;
        try std.testing.expectError(error.QwenQuantGraphArgumentMismatch, e.verifyTerminalGraph(graph, shape, 3, readback));
        Fake.params[2] = Fake.from(&terminal);
        Fake.params[2].extra = terminal.args.pointers();
        try std.testing.expectError(error.QwenTerminalGraphArgumentMismatch, e.verifyTerminalGraph(graph, shape, 3, readback));
        Fake.params[2] = Fake.from(&terminal);
        Fake.params[2].func = @ptrFromInt(99);
        try std.testing.expectError(error.QwenTerminalGraphFunctionMismatch, e.verifyTerminalGraph(graph, shape, 3, readback));
        Fake.params[2].func = @ptrFromInt(10);
        Fake.wrong_name = true;
        try std.testing.expectError(error.QwenTerminalGraphFunctionMismatch, e.verifyTerminalGraph(graph, shape, 3, readback));
        Fake.wrong_name = false;
        Fake.params[0] = Fake.from(&terminal);
        try std.testing.expectError(error.QwenTerminalGraphWorkCountMismatch, e.verifyTerminalGraph(graph, shape, 3, readback));
        Fake.params[0] = Fake.from(&body);
        Fake.count = 2;
        try std.testing.expectError(error.QwenTerminalGraphWorkCountMismatch, e.verifyTerminalGraph(graph, shape, 2, readback));
        Fake.count = 4;
        try std.testing.expectError(error.QwenTerminalGraphOrderMismatch, e.verifyTerminalGraph(graph, shape, 4, readback));
        Fake.count = 3;
        Fake.params[1] = Fake.from(&body);
        try std.testing.expectError(error.QwenTerminalGraphSymbolMismatch, e.verifyTerminalGraph(graph, shape, 3, readback));
        Fake.params[1] = Fake.from(&head);
        Fake.predecessors[2] = 1;
        try std.testing.expectError(error.QwenQuantGraphDependencyMismatch, e.verifyTerminalGraph(graph, shape, 3, readback));
        Fake.predecessors[2] = 2;
        Fake.branch = true;
        try std.testing.expectError(error.QwenQuantGraphDependencyMismatch, e.verifyTerminalGraph(graph, shape, 3, readback));
        Fake.branch = false;
        shape.terminal_greedy = false;
        try std.testing.expectError(error.QwenTerminalGraphWorkCountMismatch, e.verifyTerminalGraph(graph, shape, 3, readback));
        Fake.count = 2;
        const baseline = try e.verifyTerminalGraph(graph, shape, 2, readback);
        try std.testing.expectEqual(0, baseline.terminal_count);
        try std.testing.expect(baseline.terminal == null);
        try std.testing.expect(baseline.head != null);
        const saved_terminal = e.terminal;
        e.terminal = null;
        _ = try e.verifyTerminalGraph(graph, shape, 2, readback);
        Fake.count = 3;
        try std.testing.expectError(error.QwenTerminalGraphFunctionMismatch, e.verifyTerminalGraph(graph, shape, 3, readback));
        e.terminal = saved_terminal;
        e.placement.end = 0;
        Fake.count = 1;
        const nonhead = try e.verifyTerminalGraph(graph, shape, 1, readback);
        try std.testing.expect(nonhead.head == null);
        try std.testing.expectEqual(0, nonhead.terminal_count);
        e.placement.end = 1;
    };
    const fixture = try std.json.parseFromSlice(std.json.Value, e.gpa, @embedFile("fixtures/ornith_nvfp4/config.json"), .{});
    defer fixture.deinit();
    const quant = fixture.value.object.get("quantization_config").?.object.get("quantized_layers").?.object.get("lm_head").?;
    try std.testing.expectEqualStrings("W4A16_NVFP4", quant.object.get("quant_algo").?.string);
    driver.api.cuLaunchKernel = Fake.launch;
    e.modelopt.?.lane.clusters = true;
    for (0..7) |i| e.modelopt.?.lane.fns[0][i] = .{ .d = &driver, .handle = @ptrFromInt(40 + i) };
    e.g.vocab = 248320;
    e.terminal_weights = .{ .vocabulary = .{ .linear = .{ .nvfp4 = cuda.nvfp4.weight(65536, 131072, 0.125, e.g.vocab, e.g.hidden) } } };
    try std.testing.expectEqual(@as(f32, 0), e.terminal_weights.vocabulary.linear.nvfp4.act);
    const linear: cuda.qlinear.Linear = .{ .lane = &e.modelopt.?.lane };
    for ([_]usize{ 1, 16, 32, 64 }) |rows| for ([_]bool{ false, true }) |all| {
        const heads: usize = if (all) rows else 1;
        const output = TerminalOutput.init(heads, e.g.vocab);
        var terminal: Engine.CkNode = .{ .function = @ptrFromInt(10), .grid = .{ 1, 1, 1 }, .threads = 256 };
        terminal.args.add(e.logits + (heads - 1) * e.g.vocab * 2);
        terminal.args.add(e.logits + output.pick_at);
        terminal.args.add(@as(u32, @intCast(e.g.vocab)));
        terminal.args.add(@as(u32, @intCast(e.g.vocab)));
        var body: Engine.CkNode = .{ .function = @ptrFromInt(30), .grid = .{ 1, 1, 1 } };
        Fake.params = .{ Fake.from(&body), undefined, Fake.from(&terminal), Fake.from(&body) };
        Fake.count = 3;
        Fake.predecessors = .{ 0, 1, 2, 3 };
        const x = e.x + (rows - heads) * e.g.hidden * 2;
        try linear.decode(.{ .d = &driver, .handle = null }, e.terminal_weights.vocabulary.linear, .{ .x = x, .part = e.modelopt.?.part }, e.logits, heads);
        var shape: Engine.Shape = .{ .total = @intCast(rows), .nseg = 1, .record = false, .prompt = true, .taps = false, .heads_all = all, .terminal_greedy = true };
        const record = try e.verifyTerminalGraph(graph, shape, 3, readback);
        try std.testing.expectEqual(13, record.head.?.argument_count);
        try std.testing.expectEqual(72, record.head.?.argument_bytes_len);
        try std.testing.expectEqualSlices(usize, &.{ 8, 8, 8, 4, 8, 8, 4, 4, 4, 4, 4, 4, 4 }, record.head.?.argument_sizes[0..record.head.?.argument_count]);
        try std.testing.expectEqual(x, record.head_input);
        try std.testing.expectEqual(e.logits, record.head_output);
        try std.testing.expectEqual(e.terminal_weights.vocabulary.linear.nvfp4.codes, record.head_weight);
        try std.testing.expectEqual(.nvfp4_lane, record.head_kind);
        try std.testing.expectEqual(.qmmf, record.head_image);
        try std.testing.expectEqual(e.terminal_weights.vocabulary.linear.nvfp4.scales, record.head_scales);
        try std.testing.expectEqual(e.terminal_weights.vocabulary.linear.nvfp4.scale, record.head_scale);
        try std.testing.expectEqual(2, record.terminal.?.predecessor);
        var head_hash: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(record.head.?.argument_bytes[0..record.head.?.argument_bytes_len], &head_hash, .{});
        try std.testing.expectEqualSlices(u8, &head_hash, &record.head.?.argument_sha256);
        for (0..13) |arg| {
            Fake.lane_args.storage[Fake.lane_args.offsets[arg]] ^= 1;
            try std.testing.expectError(error.QwenQuantGraphArgumentMismatch, e.verifyTerminalGraph(graph, shape, 3, readback));
            Fake.lane_args.storage[Fake.lane_args.offsets[arg]] ^= 1;
        }
        inline for (.{ "grid_x", "grid_y", "grid_z", "block_x", "block_y", "block_z", "shared_bytes" }) |field| {
            @field(Fake.params[1], field) += 1;
            try std.testing.expectError(error.QwenQuantGraphKernelMismatch, e.verifyTerminalGraph(graph, shape, 3, readback));
            @field(Fake.params[1], field) -= 1;
        }
        const saved_function = Fake.params[1].func;
        Fake.params[1].func = @ptrFromInt(30);
        try std.testing.expectError(error.QwenTerminalGraphSymbolMismatch, e.verifyTerminalGraph(graph, shape, 3, readback));
        Fake.params[1].func = saved_function;
        const tile: usize = if (heads <= 16) 0 else if (heads <= 32) 1 else 2;
        const saved_binding = e.modelopt.?.lane.fns[0][tile * 2].handle;
        e.modelopt.?.lane.fns[0][tile * 2].handle = @ptrFromInt(99);
        try std.testing.expectError(error.QwenQuantGraphKernelMismatch, e.verifyTerminalGraph(graph, shape, 3, readback));
        e.modelopt.?.lane.fns[0][tile * 2].handle = saved_binding;
        Fake.params[1].extra = Fake.lane_args.pointers();
        try std.testing.expectError(error.QwenTerminalGraphArgumentMismatch, e.verifyTerminalGraph(graph, shape, 3, readback));
        Fake.params[1].extra = null;
        const saved_weight = e.terminal_weights.vocabulary.linear.nvfp4;
        inline for (.{ "codes", "scales", "scale" }) |field| {
            @field(e.terminal_weights.vocabulary.linear.nvfp4, field) = 0;
            try std.testing.expectError(error.QwenTerminalGraphHeadUnsupported, e.verifyTerminalGraph(graph, shape, 3, readback));
            e.terminal_weights.vocabulary.linear.nvfp4 = saved_weight;
        }
        e.terminal_weights.vocabulary.linear.nvfp4.scale = std.math.nan(f32);
        try std.testing.expectError(error.QwenTerminalGraphHeadUnsupported, e.verifyTerminalGraph(graph, shape, 3, readback));
        e.terminal_weights.vocabulary.linear.nvfp4 = saved_weight;
        e.terminal_weights.vocabulary.linear.nvfp4.act = 0.5;
        try std.testing.expectError(error.QwenTerminalGraphHeadUnsupported, e.verifyTerminalGraph(graph, shape, 3, readback));
        e.terminal_weights.vocabulary.linear.nvfp4 = saved_weight;
        e.g.vocab = 64;
        e.terminal_weights.vocabulary.linear.nvfp4 = cuda.nvfp4.weight(saved_weight.codes, saved_weight.scales, saved_weight.scale, 64, e.g.hidden);
        shape.terminal_greedy = false;
        Fake.count = 2;
        try std.testing.expectError(error.QwenTerminalGraphHeadUnsupported, e.verifyTerminalGraph(graph, shape, 2, readback));
        e.g.vocab = 248320;
        e.terminal_weights.vocabulary.linear.nvfp4 = saved_weight;
        Fake.count = 3;
        shape.terminal_greedy = false;
        try std.testing.expectError(error.QwenTerminalGraphWorkCountMismatch, e.verifyTerminalGraph(graph, shape, 3, readback));
        Fake.count = 2;
        const baseline = try e.verifyTerminalGraph(graph, shape, 2, readback);
        try std.testing.expect(baseline.head != null and baseline.terminal == null);
        e.placement.end = 0;
        Fake.count = 1;
        const nonhead = try e.verifyTerminalGraph(graph, shape, 1, readback);
        try std.testing.expect(nonhead.head == null and nonhead.terminal == null);
        e.placement.end = 1;
    };
}

test "expert graph verification rejects opposite arms, wrong arguments, duplicates and missing work" {
    const Fake = struct {
        var params: cuda.abi.KernelNodeParams = undefined;
        var count: usize = 1;
        fn nodes(_: cuda.abi.Graph, out: ?[*]cuda.graph.Node, n: *usize) callconv(.c) cuda.abi.Result {
            if (out) |dst| for (0..@min(n.*, count)) |i| {
                dst[i] = @ptrFromInt(i + 1);
            };
            n.* = count;
            return cuda.abi.success;
        }
        fn get(_: cuda.graph.Node, out: *cuda.abi.KernelNodeParams) callconv(.c) cuda.abi.Result {
            out.* = params;
            return cuda.abi.success;
        }
    };
    var driver: cuda.Driver = undefined;
    driver.api.cuGraphGetNodes = Fake.nodes;
    driver.api.cuGraphKernelNodeGetParams_v2 = Fake.get;
    var bindings: modelopt.Bindings = undefined;
    bindings.nvfp4_experts.fns[0] = .{ .d = &driver, .handle = @ptrFromInt(1) };
    bindings.nvfp4_experts.resident[0] = 192;
    bindings.nvfp4_experts.warps = 4;
    bindings.nvfp4_experts.n8_up = .{ .function = .{ .d = &driver, .handle = @ptrFromInt(2) }, .resident = 192 };
    bindings.nvfp4_experts.split_m_up = .{ .function = .{ .d = &driver, .handle = @ptrFromInt(3) }, .resident = 168 };
    bindings.nvfp4_experts.slice_major_up = .{ .function = .{ .d = &driver, .handle = @ptrFromInt(4) }, .resident = 168 };
    const expert_layer: cuda.experts.Layer = .{ .format = .nvfp4, .up = 101, .down = 102, .up_scale = 103, .down_scale = 104, .width = 512, .dims = 2048, .experts = 256 };
    var local_layers: [1]Layer = undefined;
    local_layers[0].mlp = .{ .moe = .{ .router = 0, .experts = expert_layer, .shared = .{ .projected = undefined } } };
    var e: Engine = undefined;
    e.gpa = std.testing.allocator;
    e.expert_graph_roster = .empty;
    defer e.expert_graph_roster.deinit(e.gpa);
    e.placement = .{ .first = 5, .end = 6, .total = 40 };
    e.d = &driver;
    e.modelopt = bindings;
    e.g = undefined;
    e.g.layers = 1;
    e.g.top_k = 8;
    e.g.experts = 256;
    e.g.hidden = 2048;
    e.layers = &local_layers;
    e.x = 301;
    e.routed = .{ .items = 201, .counts = 202, .members = 203, .rank = 204, .hist = 205, .act = 302 };
    const plan: cuda.grouped.Plan = .{ .items = 201, .counts = 202, .members = 203, .rank = 204, .hist = 205 };
    var args = cuda.experts.Nvfp4Experts.decodeArgs(.up, .{ .x = 301, .stride = 2048, .slots = 8 }, expert_layer, plan, 302, -1);
    Fake.params = .{ .func = @ptrFromInt(1), .grid_x = 32, .grid_y = 1, .grid_z = 1, .block_x = 128, .block_y = 1, .block_z = 1, .shared_bytes = 0, .params = args.pointers(), .extra = null };
    Fake.count = 1;
    const graph: cuda.graph.Graph = .{ .d = &driver, .handle = null };
    var shape: Engine.Shape = .{ .total = 1, .nseg = 1, .record = false, .prompt = true, .taps = false };
    try std.testing.expectEqual(@as(usize, 1), try e.verifyExpertGraph(graph, shape, 1));
    shape.expert_up_n8 = true;
    try std.testing.expectError(error.QwenExpertGraphKernelMismatch, e.verifyExpertGraph(graph, shape, 1));
    Fake.params.func = @ptrFromInt(2);
    Fake.params.grid_x = 128;
    try std.testing.expectEqual(@as(usize, 1), try e.verifyExpertGraph(graph, shape, 1));
    Fake.params.block_x = 32;
    try std.testing.expectError(error.QwenExpertGraphKernelMismatch, e.verifyExpertGraph(graph, shape, 1));
    Fake.params.block_x = 128;
    args.storage[args.offsets[10]] ^= 1;
    try std.testing.expectError(error.QwenExpertGraphArgumentMismatch, e.verifyExpertGraph(graph, shape, 1));
    args.storage[args.offsets[10]] ^= 1;
    shape.expert_up_split_m = true;
    try std.testing.expectError(error.QwenExpertGraphKernelMismatch, e.verifyExpertGraph(graph, shape, 1));
    Fake.params.func = @ptrFromInt(3);
    Fake.params.block_x = 256;
    try std.testing.expectEqual(@as(usize, 1), try e.verifyExpertGraph(graph, shape, 1));
    for (0..args.count) |i| {
        args.storage[args.offsets[i]] ^= 1;
        if (i == 3) {
            try std.testing.expectError(error.QwenExpertGraphLayerMismatch, e.verifyExpertGraph(graph, shape, 1));
        } else try std.testing.expectError(error.QwenExpertGraphArgumentMismatch, e.verifyExpertGraph(graph, shape, 1));
        args.storage[args.offsets[i]] ^= 1;
    }
    Fake.count = 2;
    try std.testing.expectError(error.QwenExpertGraphLayerMismatch, e.verifyExpertGraph(graph, shape, 2));
    Fake.count = 0;
    try std.testing.expectError(error.QwenExpertGraphWorkCountMismatch, e.verifyExpertGraph(graph, shape, 0));
    shape.expert_up_layout = .slice_major_v1;
    Fake.count = 1;
    Fake.params.func = @ptrFromInt(4);
    try std.testing.expectError(error.QwenExpertUpLayoutMismatch, e.verifyExpertGraph(graph, shape, 1));
    local_layers[0].mlp.moe.experts.nvfp4_up_layout = .slice_major_v1;
    try std.testing.expectEqual(@as(usize, 1), try e.verifyExpertGraph(graph, shape, 1));
    Fake.params.func = @ptrFromInt(3);
    try std.testing.expectError(error.QwenExpertGraphKernelMismatch, e.verifyExpertGraph(graph, shape, 1));
    try std.testing.expectEqual(@as(usize, 4), e.expert_graph_roster.items.len);
    const roster = e.expert_graph_roster.items;
    try std.testing.expectEqual(.baseline, roster[0].arm);
    try std.testing.expectEqual(.n8, roster[1].arm);
    try std.testing.expectEqual(.split_m, roster[2].arm);
    try std.testing.expectEqual(.slice_major, roster[3].arm);
    try std.testing.expectEqual(.slice_major_v1, roster[3].up_layout);
    try std.testing.expectEqual(@as(usize, 5), roster[2].layer_global);
    try std.testing.expectEqual(@as(usize, 14), roster[2].abi_count);
    try std.testing.expectEqual([3]u32{ 128, 1, 1 }, roster[2].grid);
    try std.testing.expectEqual([3]u32{ 256, 1, 1 }, roster[2].block);
    try std.testing.expectEqualSlices(u8, &roster[0].args_sha256, &roster[2].args_sha256);
}

test "constructor cache warmup runs only for a complete embedding-and-head placement" {
    const first: checkpoint.Placement = .{ .first = 0, .end = 10, .total = 40 };
    const middle: checkpoint.Placement = .{ .first = 10, .end = 20, .total = 40 };
    const last: checkpoint.Placement = .{ .first = 20, .end = 40, .total = 40 };
    const full: checkpoint.Placement = .{ .first = 0, .end = 40, .total = 40 };
    try std.testing.expect(first.ownsEmbedding() and !first.ownsHead());
    try std.testing.expect(!middle.ownsEmbedding() and !middle.ownsHead());
    try std.testing.expect(!last.ownsEmbedding() and last.ownsHead());
    try std.testing.expect(full.ownsEmbedding() and full.ownsHead());
    try std.testing.expect(!Engine.warmsInConstructor(first));
    try std.testing.expect(!Engine.warmsInConstructor(middle));
    try std.testing.expect(!Engine.warmsInConstructor(last));
    try std.testing.expect(Engine.warmsInConstructor(full));
    var partial: Engine = undefined;
    partial.placement = first;
    try partial.warmCache(@as(*Cache, undefined));
    partial.placement = middle;
    try partial.warmCache(@as(*Cache, undefined));
    partial.placement = last;
    try partial.warmCache(@as(*Cache, undefined));
}

test "plain BF16 execution refuses before checkpoint I/O or CUDA access" {
    const admitted = try checkpoint.admit(std.testing.allocator, @embedFile("fixtures/clef_flash/config.json"));
    const placement: checkpoint.Placement = .{ .first = 0, .end = 32, .total = 32 };
    try std.testing.expectError(error.UnsupportedQwenPlainExecution, Engine.initPlacedWithOptions(std.testing.allocator, std.testing.io, undefined, "missing-checkpoint", admitted, placement, false, .{}));
    try std.testing.expectError(error.UnsupportedQwenPlainExecution, Engine.scratchBytes(admitted, placement));
    try std.testing.expectError(error.UnsupportedQwenPlainExecution, weights.plan(@as(*core.Checkpoint, undefined), admitted, placement));
    try std.testing.expectError(error.UnsupportedQwenPlainExecution, weights.loadPlaced(undefined, undefined, undefined, admitted, placement));
}

test "combined candidates require HF and split-M requires the N8 contract" {
    try validateCombined(.hf_modelopt, true, true, true);
    try validateCombined(.hf_modelopt, false, false, true);
    try validateCombined(.mlx_sanitized, false, false, false);
    try std.testing.expectError(error.UnsupportedQwenExpertUp, validateCombined(.hf_modelopt, false, true, false));
    try std.testing.expectError(error.UnsupportedQwenExpertUp, validateCombined(.mlx_sanitized, true, true, false));
    try std.testing.expectError(error.UnsupportedQwenQuantReuse, validateCombined(.mlx_sanitized, false, false, true));
}

test "quant graph plans derive compatible groups and attest all quant lane reduction arguments" {
    var e: Engine = undefined;
    e.gpa = std.testing.allocator;
    e.x = 1024;
    var m: modelopt.Bindings = undefined;
    m.rows = 1048576;
    m.part = 2097152;
    m.ck.clusters = false;
    m.ck.quant8_fn.handle = @ptrFromInt(1);
    m.ck.lane[0].handle = @ptrFromInt(2);
    m.ck.reduce.handle = @ptrFromInt(3);
    e.modelopt = m;
    const w: cuda.qlinear.Weight = .{ .fp8 = .{ .mode = .fp8, .codes = 256, .scales = 0, .n = 64, .k = 4096, .npad = 128, .act = 0.5, .scale = 0.25 } };
    var projections = [_]Proj{ .{ .linear = w }, .{ .linear = w }, .{ .linear = w } };
    var list: std.ArrayList(Engine.CkProjection) = .empty;
    defer list.deinit(e.gpa);
    var shape: Engine.Shape = .{ .total = 16, .nseg = 1, .record = false, .prompt = true, .taps = false, .reuse_fp8_quant = true };
    try e.addCkGroup(&list, &projections, &.{ 3145728, 4194304, 5242880 }, shape);
    try std.testing.expectEqual(@as(usize, 3), list.items.len);
    try std.testing.expect(list.items[0].quant and !list.items[1].quant and !list.items[2].quant);
    inline for (.{ .quant, .lane, .reduce }) |kind| {
        var want = e.ckNode(list.items[1], kind);
        const pristine = want;
        var actual: cuda.abi.KernelNodeParams = .{ .func = want.function, .grid_x = @intCast(want.grid[0]), .grid_y = @intCast(want.grid[1]), .grid_z = @intCast(want.grid[2]), .block_x = @intCast(want.threads), .block_y = 1, .block_z = 1, .shared_bytes = @intCast(want.shared), .params = want.args.pointers(), .extra = null };
        try pristine.check(actual);
        for (0..want.args.count) |i| {
            want.args.storage[want.args.offsets[i]] ^= 1;
            try std.testing.expectError(error.QwenQuantGraphArgumentMismatch, pristine.check(actual));
            want.args.storage[want.args.offsets[i]] ^= 1;
        }
        actual.grid_x += 1;
        try std.testing.expectError(error.QwenQuantGraphKernelMismatch, pristine.check(actual));
    }
    list.clearRetainingCapacity();
    projections[2].linear.fp8.act = 0.25;
    try e.addCkGroup(&list, &projections, &.{ 3145728, 4194304, 5242880 }, shape);
    for (list.items) |p| try std.testing.expect(p.quant);
    list.clearRetainingCapacity();
    projections[2].linear = .{ .bf16 = w.fp8 };
    try e.addCkGroup(&list, &projections, &.{ 3145728, 4194304, 5242880 }, shape);
    try std.testing.expectEqual(@as(usize, 2), list.items.len);
    for (list.items) |p| try std.testing.expect(p.quant);
    list.clearRetainingCapacity();
    projections[2].linear = w;
    shape.reuse_fp8_quant = false;
    try e.addCkGroup(&list, &projections, &.{ 3145728, 4194304, 5242880 }, shape);
    for (list.items) |p| try std.testing.expect(p.quant);
}

test "bounded HF admission is opt in and does not imply recorded windows" {
    var cache: Cache = undefined;
    var parts = [_]Part{.{ .cache = &cache, .rows = 1 }};
    try validateHfWindow(.hf_modelopt, false, &parts, 1, .last);
    try std.testing.expectError(error.UnsupportedQwenWindow, validateHfWindow(.hf_modelopt, false, &parts, 1, .prompt_logits));
    for (1..17) |rows| {
        parts[0].rows = rows;
        try validateHfWindow(.hf_modelopt, true, &parts, rows, .last);
        try validateHfWindow(.hf_modelopt, true, &parts, rows, .prompt_logits);
        if (rows > 1) try std.testing.expectError(error.UnsupportedQwenWindow, validateHfWindow(.hf_modelopt, false, &parts, rows, .last));
        try std.testing.expectError(error.UnsupportedQwenWindow, validateHfWindow(.hf_modelopt, true, &parts, rows, .logits));
    }
    for ([_]usize{ 0, 17 }) |rows| {
        parts[0].rows = rows;
        try std.testing.expectError(error.UnsupportedQwenWindow, validateHfWindow(.hf_modelopt, true, &parts, rows, .last));
    }
    parts[0].rows = 2;
    try std.testing.expectError(error.UnsupportedQwenWindow, validateHfWindow(.hf_modelopt, true, &parts, 1, .last));
    parts[0].parents = &.{ -1, 0 };
    try std.testing.expectError(error.UnsupportedQwenWindow, validateHfWindow(.hf_modelopt, true, &parts, 2, .last));
    parts[0].parents = null;
    parts[0].images = @ptrFromInt(16);
    try std.testing.expectError(error.UnsupportedQwenWindow, validateHfWindow(.hf_modelopt, true, &parts, 2, .last));
    parts[0].images = null;
    try std.testing.expectError(error.UnsupportedQwenWindow, validateHfWindow(.mlx_sanitized, true, &parts, 2, .prompt_logits));
    try validateHfWindow(.mlx_sanitized, false, &parts, 2, .logits);
    try std.testing.expect(!records(.last) and !records(.prompt_logits));
    try std.testing.expect(records(.logits) and records(.{ .draws = &.{} }));
}

test "bounded HF graph keys separate every row count layout expert arm head and math mode" {
    var keys: [4096]u32 = undefined;
    var count: usize = 0;
    for (1..17) |rows| for ([_]Cache.GdnLayout{ .value_key, .key_value }) |layout| for ([_]bool{ false, true }) |n8| for ([_]bool{ false, true }) |heads| for ([_]bool{ false, true }) |split| for ([_]bool{ false, true }) |reuse| for ([_]cuda.experts.Nvfp4UpLayout{ .lane_major, .slice_major_v1 }) |up_layout| for ([_]bool{ false, true }) |terminal| for ([_]MoeMath{ .legacy, .bf16_boundaries }) |math| {
        const shape: Engine.Shape = .{ .total = @intCast(rows), .nseg = 1, .record = false, .prompt = true, .taps = false, .gdn_layout = layout, .expert_up_n8 = n8, .heads_all = heads, .expert_up_split_m = split, .reuse_fp8_quant = reuse, .expert_up_layout = up_layout, .terminal_greedy = terminal, .moe_math = math };
        const key = shape.key();
        try std.testing.expect(std.mem.indexOfScalar(u32, keys[0..count], key) == null);
        keys[count] = key;
        count += 1;
    };
    try std.testing.expectEqual(@as(usize, 4096), count);
}

test "BF16 MoE math refuses unsupported models before Engine or CUDA allocation" {
    try validateMoeMath(undefined, .legacy);
    try std.testing.expectEqual(MoeMath.legacy, (Engine.InitOptions{}).moe_math);
    var admitted: checkpoint.Admitted = .{ .geometry = undefined, .encoding = .{ .modelopt = undefined } };
    admitted.geometry.hidden = 2048;
    admitted.geometry.intermediate = 512;
    admitted.geometry.experts = 256;
    admitted.geometry.top_k = 8;
    admitted.encoding.modelopt.spec.shared_width = 512;
    try validateMoeMath(admitted, .bf16_boundaries);
    inline for (.{ "hidden", "intermediate", "experts", "top_k" }) |field| {
        var invalid = admitted;
        @field(invalid.geometry, field) -= 1;
        try std.testing.expectError(error.UnsupportedQwenMoeMath, Engine.initPlacedWithOptions(undefined, undefined, undefined, "", invalid, undefined, false, .{ .moe_math = .bf16_boundaries }));
    }
    var invalid_shared = admitted;
    invalid_shared.encoding.modelopt.spec.shared_width = 511;
    try std.testing.expectError(error.UnsupportedQwenMoeMath, validateMoeMath(invalid_shared, .bf16_boundaries));
    admitted.encoding = .mlx;
    try std.testing.expectError(error.UnsupportedQwenMoeMath, validateMoeMath(admitted, .bf16_boundaries));
}

test "MoE math bindings and shape agree and own only graph bit 26" {
    const baseline: Engine.Shape = .{ .total = 1, .nseg = 1, .record = false, .prompt = true, .taps = false };
    var candidate = baseline;
    candidate.moe_math = .bf16_boundaries;
    try std.testing.expectEqual(@as(u32, 1 << 26), baseline.key() ^ candidate.key());
    candidate.terminal_greedy = true;
    try std.testing.expectEqual(@as(u32, (1 << 26) | (1 << 25)), baseline.key() ^ candidate.key());
    for ([_]MoeMath{ .legacy, .bf16_boundaries }) |configured| for ([_]MoeMath{ .legacy, .bf16_boundaries }) |requested| {
        if (configured == requested) try validateMoeMathShape(configured, requested) else try std.testing.expectError(error.QwenMoeMathShapeMismatch, validateMoeMathShape(configured, requested));
    };
}

test "attention mode owns graph bit 27 and rejects Engine/Shape disagreement" {
    try std.testing.expectEqual(AttentionMode.reference, (Engine.InitOptions{}).attention_mode);
    const baseline: Engine.Shape = .{ .total = 1, .nseg = 1, .record = false, .prompt = true, .taps = false };
    var candidate = baseline;
    candidate.attention_mode = .chunked;
    try std.testing.expectEqual(@as(u32, 1 << 27), baseline.key() ^ candidate.key());
    candidate.moe_math = .bf16_boundaries;
    try std.testing.expectEqual(@as(u32, (1 << 27) | (1 << 26)), baseline.key() ^ candidate.key());
    for ([_]AttentionMode{ .reference, .chunked }) |configured| for ([_]AttentionMode{ .reference, .chunked }) |requested| {
        if (configured == requested) try validateAttentionModeShape(configured, requested) else try std.testing.expectError(error.QwenAttentionModeShapeMismatch, validateAttentionModeShape(configured, requested));
    };
    var admitted: checkpoint.Admitted = .{ .geometry = undefined, .encoding = .{ .modelopt = undefined } };
    try validateAttentionMode(admitted, .reference, false);
    try validateAttentionMode(admitted, .chunked, false);
    try validateAttentionMode(admitted, .chunked, true);
    try std.testing.expectError(error.QwenReferenceAttentionNeedsBf16Kv, validateAttentionMode(admitted, .reference, true));
    admitted.encoding = .mlx;
    try validateAttentionMode(admitted, .reference, false);
    try validateAttentionMode(admitted, .reference, true);
    try std.testing.expectError(error.UnsupportedQwenAttentionMode, validateAttentionMode(admitted, .chunked, false));
    const ref = hf.attnLaunch(.reference, 16, 2, 4, 8, max_chunks);
    const ch = hf.attnLaunch(.chunked, 16, 2, 4, 8, max_chunks);
    try std.testing.expect(ref.partial_grid[0] != ch.partial_grid[0] or ref.partial_grid[1] != ch.partial_grid[1]);
    try std.testing.expectEqual(@as(usize, 256), ref.merge_block);
    try std.testing.expectEqual(@as(usize, 32), ch.merge_block);
}

test "bounded HF chain views retain every state binding and reject field corruption" {
    const base: Seg = .{ .rec = 101, .conv = 102, .kv = 103, .tap = 0, .log = 104, .before = 0, .cap = 4096, .len = 127, .row0 = 0, .rows = 16, .tap_rows = 0, .taps = 0, .delta = -3 };
    var views: [hf_prefill_rows]Seg = undefined;
    for (&views, 0..) |*view, row| view.* = chainView(base, row);
    try validateChainViews(base, &views);
    for (&views) |*view| {
        inline for (@typeInfo(Seg).@"struct".field_names) |name| {
            @field(view, name) += 1;
            try std.testing.expectError(error.QwenGdnTableMismatch, validateChainViews(base, &views));
            @field(view, name) -= 1;
        }
    }
    try std.testing.expectError(error.QwenGdnTableMismatch, validateChainViews(base, views[0..15]));
    try std.testing.expect(Engine.chain_segs_at % 256 == 0);
    try std.testing.expect(Engine.src_at >= Engine.chain_segs_at + @sizeOf(@TypeOf(views)));
}

test "bounded HF verifies uploaded row views parents ancestors and counts" {
    const Fake = struct {
        fn sync(_: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
            return cuda.abi.success;
        }
        fn copy(out: ?*anyopaque, source: cuda.abi.DevicePtr, bytes: usize) callconv(.c) cuda.abi.Result {
            const dst: [*]u8 = @ptrCast(out.?);
            const src: [*]const u8 = @ptrFromInt(source);
            @memcpy(dst[0..bytes], src[0..bytes]);
            return cuda.abi.success;
        }
    };
    var driver: cuda.Driver = undefined;
    driver.api.cuStreamSynchronize = Fake.sync;
    driver.api.cuMemcpyDtoH_v2 = Fake.copy;
    var host: [Engine.tables_len]u8 align(16) = @splat(0);
    var device: [Engine.tables_len]u8 align(16) = @splat(0);
    const base: Seg = .{ .rec = 101, .conv = 102, .kv = 103, .tap = 0, .log = 104, .before = 0, .cap = 4096, .len = 127, .row0 = 0, .rows = 2, .tap_rows = 0, .taps = 0, .delta = 0 };
    @memcpy(host[Engine.segs_at..][0..@sizeOf(Seg)], std.mem.asBytes(&base));
    const views = std.mem.bytesAsSlice(Seg, host[Engine.chain_segs_at..][0 .. 2 * @sizeOf(Seg)]);
    for (views, 0..) |*view, row| view.* = chainView(base, row);
    var cache: Cache = undefined;
    _ = try Round.build(&host, &.{.{ .cache = &cache, .rows = 2 }}, false);
    const params = std.mem.bytesAsSlice(c_int, host[Engine.params_at..][0..32]);
    params[2] = 2;
    params[3] = 1;
    device = host;
    var e: Engine = undefined;
    e.stream = .{ .d = &driver, .handle = null };
    e.tables = .{ .d = &driver, .ptr = @intFromPtr(&device), .len = device.len };
    e.staged = .{ .d = &driver, .bytes = &host };
    const shape: Engine.Shape = .{ .total = 2, .nseg = 1, .record = false, .prompt = true, .taps = false };
    try e.verifyHfChainTables(shape);
    for ([_]usize{ Engine.segs_at, Engine.chain_segs_at, Engine.chain_segs_at + @sizeOf(Seg), Engine.params_at + 8, Engine.params_at + 12, Engine.params_at + 16, Engine.rowseg_at, Engine.rowpar_at + 4, Engine.rowanc_at + 4 }) |offset| {
        device[offset] ^= 1;
        try std.testing.expectError(error.QwenGdnTableMismatch, e.verifyHfChainTables(shape));
        device[offset] ^= 1;
    }
}

test "GDN prefix roster rejects duplicates missing rows and wrong order" {
    var roster: GdnRoster = .{ .layers = 2, .rows = 16 };
    try std.testing.expectError(error.QwenGdnGraphLayerMismatch, roster.accept(0, 1));
    try roster.accept(0, 0);
    try std.testing.expectError(error.QwenGdnGraphLayerMismatch, roster.accept(0, 0));
    try std.testing.expectError(error.QwenGdnGraphLayerMismatch, roster.accept(1, 0));
    try std.testing.expectError(error.QwenGdnGraphWorkCountMismatch, roster.finish());
    for (1..16) |row| try roster.accept(0, row);
    for (0..16) |row| try roster.accept(1, row);
    try roster.finish();
    try std.testing.expectError(error.QwenGdnGraphLayerMismatch, roster.accept(2, 0));
}

test "GDN prefix node verifier checks all eleven ABI arguments" {
    var e: Engine = undefined;
    e.geo = .{ .hid = 2048, .nk = 16, .nv = 32, .qh = 16, .kvh = 2, .nl = 1 };
    e.g = undefined;
    e.g.layers = 1;
    e.g.linear = @splat(true);
    e.g.v_heads = 32;
    e.q = 101;
    e.k = 102;
    e.v = 103;
    e.g_ = 104;
    e.beta = 105;
    e.y = 106;
    e.tables.ptr = 4096;
    const shape: Engine.Shape = .{ .total = 2, .nseg = 1, .record = false, .prompt = true, .taps = false };
    var args = e.gdnArguments(shape, 0, 1);
    try std.testing.expectEqual(@as(usize, 11), args.count);
    var p: cuda.abi.KernelNodeParams = .{ .func = @ptrFromInt(1), .grid_x = 1, .grid_y = 1, .grid_z = 32, .block_x = 128, .block_y = 1, .block_z = 1, .shared_bytes = 0, .params = args.pointers(), .extra = null };
    const item = try e.checkGdnNode(p, shape, @ptrFromInt(1));
    try std.testing.expectEqual(@as(usize, 1), item.row);
    for (0..args.count) |i| {
        args.storage[args.offsets[i]] ^= 1;
        if (i == 8) {
            try std.testing.expectError(error.QwenGdnGraphLayerMismatch, e.checkGdnNode(p, shape, @ptrFromInt(1)));
        } else try std.testing.expectError(error.QwenGdnGraphArgumentMismatch, e.checkGdnNode(p, shape, @ptrFromInt(1)));
        args.storage[args.offsets[i]] ^= 1;
    }
    p.grid_z = 64;
    try std.testing.expectError(error.QwenGdnGraphKernelMismatch, e.checkGdnNode(p, shape, @ptrFromInt(1)));
}

test "prompt logits finish copies every row and commits terminal cache logits only" {
    const Fake = struct {
        fn sync(_: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
            return cuda.abi.success;
        }
        fn copy(out: ?*anyopaque, source: cuda.abi.DevicePtr, bytes: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
            const dst: [*]u8 = @ptrCast(out.?);
            const src: [*]const u8 = @ptrFromInt(source);
            @memcpy(dst[0..bytes], src[0..bytes]);
            return cuda.abi.success;
        }
    };
    var driver: cuda.Driver = undefined;
    driver.api.cuStreamSynchronize = Fake.sync;
    driver.api.cuMemcpyDtoHAsync_v2 = Fake.copy;
    var device = [_]u16{ 1, 2, 3, 4, 5, 6 };
    var host: [12]u8 align(16) = @splat(0);
    var terminal: [2]u16 = @splat(0);
    var cache: Cache = undefined;
    cache.logits = &terminal;
    var e: Engine = undefined;
    e.d = &driver;
    e.stream = .{ .d = &driver, .handle = null };
    e.placement = .{ .first = 0, .end = 1, .total = 1 };
    e.g.vocab = 2;
    e.numerics = .hf_modelopt;
    e.logits = @intFromPtr(&device);
    e.host_logits = .{ .d = &driver, .bytes = &host };
    const parts = [_]Part{.{ .cache = &cache, .rows = 3 }};
    var shape: Engine.Shape = .{ .total = 3, .nseg = 1, .record = false, .prompt = true, .taps = false, .heads_all = true };
    try e.finish(&parts, .prompt_logits, shape);
    try std.testing.expectEqualSlices(u16, device[4..6], &terminal);
    for (0..3) |row| try std.testing.expectEqualSlices(u16, device[row * 2 ..][0..2], try e.rowLogits(row));
    terminal = .{ 99, 98 };
    shape.record = true;
    shape.heads_all = false;
    try e.finish(&parts, .logits, shape);
    try std.testing.expectEqualSlices(u16, &.{ 99, 98 }, &terminal);
}

test "immutable expert representation rejects stale shapes disabled consumers and mismatched layer tags before CUDA" {
    var e: Engine = undefined;
    e.numerics = .hf_modelopt;
    e.expert_up_layout = .slice_major_v1;
    e.expert_up_n8 = true;
    e.expert_up_split_m = true;
    var local: [1]Layer = undefined;
    local[0].mlp = .{ .moe = .{ .router = 0, .experts = .{ .format = .nvfp4, .nvfp4_up_layout = .slice_major_v1, .up = 101, .down = 102, .width = 512, .dims = 2048, .experts = 256 }, .shared = .{ .projected = undefined } } };
    e.layers = &local;
    try e.validateUpLayout(.slice_major_v1, true, true, 16);
    try std.testing.expectError(error.QwenExpertUpLayoutMismatch, e.validateUpLayout(.lane_major, true, true, 1));
    try std.testing.expectError(error.UnsupportedQwenExpertUpLayout, e.validateUpLayout(.slice_major_v1, false, true, 1));
    try std.testing.expectError(error.UnsupportedQwenExpertUpLayout, e.validateUpLayout(.slice_major_v1, true, false, 1));
    for ([_]usize{ 0, 17, 128 }) |rows| try std.testing.expectError(error.UnsupportedQwenExpertUpLayout, e.validateUpLayout(.slice_major_v1, true, true, rows));
    e.expert_up_split_m = false;
    try std.testing.expectError(error.UnsupportedQwenExpertUpLayout, e.validateUpLayout(.slice_major_v1, true, true, 1));
    e.expert_up_split_m = true;
    e.expert_up_n8 = false;
    try std.testing.expectError(error.UnsupportedQwenExpertUpLayout, e.validateUpLayout(.slice_major_v1, true, true, 1));
    e.expert_up_n8 = true;
    local[0].mlp.moe.experts.nvfp4_up_layout = .lane_major;
    try std.testing.expectError(error.QwenExpertUpLayoutMismatch, e.validateUpLayout(.slice_major_v1, true, true, 1));
}

test "warmup prompt rows follow ModelOpt admission and MLX 32" {
    var e: Engine = undefined;
    e.numerics = .mlx_sanitized;
    e.bounded_hf_prefill = false;
    try std.testing.expectEqual(@as(usize, 32), e.warmupPromptRows());
    e.numerics = .hf_modelopt;
    try std.testing.expectEqual(@as(usize, 1), e.warmupPromptRows());
    e.bounded_hf_prefill = true;
    try std.testing.expectEqual(@as(usize, hf_prefill_rows), e.warmupPromptRows());
    e.numerics = .hf_dense_bf16;
    try std.testing.expectEqual(@as(usize, hf_prefill_rows), e.warmupPromptRows());
}

test "live Engine prompt expert dispatch keeps 16-row cutoff and 64-row verify" {
    const moe = @import("cuda_forward.zig").promptMoe;
    try std.testing.expectEqual(@as(usize, 16), hf_prefill_rows);
    try std.testing.expectEqual(@as(usize, 64), most_window);
    try std.testing.expect(!moe(true, 16));
    try std.testing.expect(moe(true, 17));
    try std.testing.expect(moe(true, 64));
    try std.testing.expect(moe(true, 65));
    try std.testing.expect(!moe(false, 16));
    try std.testing.expect(!moe(false, 64));
    try std.testing.expect(!moe(false, 65));
}

test "production row tables store 16-byte ancestors for prompt parent and long chains" {
    var st: [Engine.tables_len]u8 align(16) = @splat(0);
    var cache: Cache = undefined;
    try std.testing.expectEqual(@as(usize, 16), @sizeOf(Round.Ancestor));
    try std.testing.expectEqual(Engine.rowanc_at, Round.rowanc_at);
    _ = try Round.build(&st, &.{.{ .cache = &cache, .rows = 2 }}, false);
    var anc = Round.ancestors(&st);
    try std.testing.expectEqual(@as(u32, 1), anc[0].chain);
    try std.testing.expectEqual(@as(u32, 0), anc[0].depth);
    try std.testing.expectEqual(@as(u32, 1), anc[1].depth);
    var p33: [33]i32 = undefined;
    p33[0] = -1;
    for (1..33) |i| p33[i] = @intCast(i - 1);
    _ = try Round.build(&st, &.{.{ .cache = &cache, .rows = 33, .parents = &p33 }}, true);
    anc = Round.ancestors(&st);
    try std.testing.expectEqual(@as(u32, 0), anc[32].chain);
    try std.testing.expectEqual(@as(u32, 32), anc[32].depth);
    var p64: [64]i32 = undefined;
    p64[0] = -1;
    for (1..64) |i| p64[i] = @intCast(i - 1);
    _ = try Round.build(&st, &.{.{ .cache = &cache, .rows = 64, .parents = &p64 }}, true);
    anc = Round.ancestors(&st);
    try std.testing.expectEqual(@as(u32, 63), anc[63].depth);
    try std.testing.expectEqual(@as(u64, 1) << 63, anc[63].bits & (@as(u64, 1) << 63));
    _ = try Round.build(&st, &.{.{ .cache = &cache, .rows = 65 }}, false);
    anc = Round.ancestors(&st);
    try std.testing.expectEqual(@as(u32, 1), anc[64].chain);
    try std.testing.expectEqual(@as(u32, 64), anc[64].depth);
    try std.testing.expectEqual(@as(u64, 0), anc[64].bits);
}
