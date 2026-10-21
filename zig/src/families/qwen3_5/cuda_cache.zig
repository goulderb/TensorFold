//! A stream's state on CUDA: recurrences, conv tails, keys and values, the drafter's tap ring, the last window's log.
//! Local modification: failed hidden requests require a clean cache reset.
const std = @import("std");
const cuda = @import("cuda");
const geometry = @import("cuda_geometry.zig");
const engine = @import("cuda_engine.zig");

const Engine = engine.Engine;
const DeviceBuffer = cuda.DeviceBuffer;

/// Rows a drafter's tap ring holds a stream: its window and a prompt chunk more.
pub const tap_rows = 2048 + engine.max_rows;

/// qwen35.cuh's Seg and Path.
pub const Seg = extern struct { rec: u64, conv: u64, kv: u64, tap: u64, log: u64, cap: c_int, len: c_int, row0: c_int, rows: c_int, tap_rows: c_int, taps: c_int, delta: c_int };
pub const Path = extern struct { n: c_int, rows: [engine.most_window]c_int };

pub const Cache = struct {
    pub const GdnLayout = enum(u1) {
        value_key,
        key_value,

        pub fn validate(layout: GdnLayout, numerics: @import("cuda_checkpoint.zig").Numerics) !void {
            if (layout == .key_value and numerics != .hf_modelopt) return error.UnsupportedQwenGdnLayout;
        }
    };

    /// Fixed when allocated. Reset preserves the recurrence's physical axes.
    gdn_layout: GdnLayout = .value_key,
    owner: ?*const Engine = null,
    hidden_invalid: bool = false,
    reset_epoch: u64 = 0,
    state: DeviceBuffer,
    kv: DeviceBuffer,
    /// A drafter's tapped rows (`Engine.taps`), a ring of tap_rows: the row at position p in slot p % tap_rows.
    tap: ?DeviceBuffer = null,
    /// The newest verify window's rows' records (qwen35.cuh's LogRow), most_window rows a linear layer.
    log: DeviceBuffer,
    cap: usize,
    len: usize = 0,
    /// Positions the newest forward wrote up to: a tap ring holds them until the next forward, whatever `keep` lands.
    written: usize = 0,
    logits: []u16,
    terminal_greedy: ?struct { position: usize, token: u32 } = null,
    /// The newest verify window of more than one row, until its path is kept: where it started, its rows.
    last: ?struct { start: usize, rows: usize } = null,
    /// An image prompt's rotary shift (get_rope_index's rope_delta): rows after its prompt rotate at position + delta.
    delta: i32 = 0,

    pub fn init(gpa: std.mem.Allocator, e: *const Engine, cap: usize) !Cache {
        return initWithLayout(gpa, e, cap, .value_key);
    }

    pub fn initWithLayout(gpa: std.mem.Allocator, e: *const Engine, cap: usize, layout: GdnLayout) !Cache {
        try layout.validate(e.numerics);
        var state = try DeviceBuffer.alloc(e.d, e.g.stateBytes());
        errdefer state.free();
        try state.fill8(0, null);
        var kv = try DeviceBuffer.alloc(e.d, e.g.kvBytes(cap, e.kv8));
        errdefer kv.free();
        try kv.fill8(0, null); // touched now, not as the first request's keys land (the driver commits on first touch)
        var log = try DeviceBuffer.alloc(e.d, e.g.logBytes(if (e.numerics == .hf_modelopt) 1 else engine.most_window));
        errdefer log.free();
        try log.fill8(0, null);
        var tap: ?DeviceBuffer = null;
        if (e.n_taps > 0) {
            tap = try DeviceBuffer.alloc(e.d, e.tapBytes());
            try tap.?.fill8(0, null);
        }
        errdefer if (tap) |*t| t.free();
        return .{ .gdn_layout = layout, .owner = e, .state = state, .kv = kv, .tap = tap, .log = log, .cap = cap, .logits = try gpa.alloc(u16, e.g.vocab) };
    }

    /// Ready for another stream: no rows, the recurrences and tails zero (on the engine's stream, before its forwards).
    pub fn reset(self: *Cache, e: *const Engine) !void {
        self.hidden_invalid = true;
        self.terminal_greedy = null;
        self.len = 0;
        self.written = 0;
        self.last = null;
        self.delta = 0;
        try self.state.fill8(0, e.stream.handle);
        self.reset_epoch = try std.math.add(u64, self.reset_epoch, 1);
        self.hidden_invalid = false;
    }

    pub fn deinit(self: *Cache, gpa: std.mem.Allocator) void {
        self.state.free();
        self.kv.free();
        self.log.free();
        if (self.tap) |*t| t.free();
        gpa.free(self.logits);
    }

    /// The stream's window of `rows` rows from forward row `row0`, after `len` cached positions.
    pub fn seg(self: *const Cache, e: *const Engine, len: usize, row0: usize, rows: usize) Seg {
        const g = e.g;
        return .{ .rec = self.state.ptr, .conv = self.state.ptr + g.deltas() * g.recBytes(), .kv = self.kv.ptr, .tap = if (self.tap) |t| t.ptr else 0, .log = self.log.ptr, .cap = @intCast(self.cap), .len = @intCast(len), .row0 = @intCast(row0), .rows = @intCast(rows), .tap_rows = tap_rows, .taps = @intCast(e.n_taps), .delta = self.delta };
    }

    /// Keeps `path` (window rows, ascending from 0): the state as that chain would leave it, a tree's keys moved down.
    pub fn keep(self: *Cache, e: *Engine, path: []const u32) !void {
        if (self.hidden_invalid) return error.QwenHiddenNeedsReset;
        if (e.numerics == .hf_modelopt) return error.UnsupportedQwenRewind;
        const last = self.last orelse return error.NothingToKeep;
        if (path.len == 0 or path.len > last.rows or path[0] != 0) return error.NothingToKeep;
        var p: Path = .{ .n = @intCast(path.len), .rows = @splat(0) };
        var prefix = true;
        for (path, 0..) |r, i| {
            if (r >= last.rows or (i > 0 and r <= path[i - 1])) return error.NothingToKeep;
            p.rows[i] = @intCast(r);
            prefix = prefix and r == i;
        }
        self.last = null;
        self.len = last.start + path.len;
        self.terminal_greedy = null;
        const g = e.g;
        const sg = self.seg(e, last.start, 0, last.rows);
        try e.run(e.keeps.apply, .{ 1, geometry.dk / 4, g.v_heads * g.deltas() }, .{ 32, 4, 1 }, .{ e.geo, sg, p });
        try e.run(e.keeps.conv, .{ g.deltas(), (g.convBytes() / 2 + 255) / 256, 1 }, .{ 256, 1, 1 }, .{ e.geo, sg, @as(c_int, @intCast(path[path.len - 1])) });
        if (!prefix and g.attentions() > 0) try e.run(if (e.kv8) e.fp8.compact else e.keeps.compact, .{ g.attentions(), g.kv_heads, 1 }, .{ geometry.head_dim, 1, 1 }, .{ e.geo, sg, p });
    }

    /// The newest window kept whole (the round loop keeps nothing when every row landed), before the next forward.
    pub fn settle(self: *Cache, e: *Engine) !void {
        if (self.hidden_invalid) return error.QwenHiddenNeedsReset;
        const last = self.last orelse return;
        var all: [engine.most_window]u32 = undefined;
        for (all[0..last.rows], 0..) |*r, i| r.* = @intCast(i);
        try self.keep(e, all[0..last.rows]);
    }
};

test "MLX rejects candidate GDN state before allocating device memory" {
    var e: Engine = undefined;
    e.numerics = .mlx_sanitized;
    try std.testing.expectError(error.UnsupportedQwenGdnLayout, Cache.initWithLayout(std.testing.allocator, &e, 1, .key_value));
}

test "cache reset invalidates a terminal pick before a failed device clear" {
    const Fake = struct {
        fn fill(_: cuda.abi.DevicePtr, _: u8, _: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
            return cuda.abi.error_not_ready;
        }
    };
    var d: cuda.Driver = undefined;
    d.api.cuMemsetD8Async = Fake.fill;
    var e: Engine = undefined;
    e.stream = .{ .d = &d, .handle = @ptrFromInt(1) };
    var cache: Cache = undefined;
    cache.state = .{ .d = &d, .ptr = 4096, .len = 16 };
    cache.terminal_greedy = .{ .position = 8, .token = 1 };
    try std.testing.expectError(error.NotReady, cache.reset(&e));
    try std.testing.expect(cache.terminal_greedy == null);
    try std.testing.expectEqual(@as(usize, 0), cache.len);
}
