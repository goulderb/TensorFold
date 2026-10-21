//! Request-owned BF16 final backbone rows. No request address belongs in a captured graph.
const std = @import("std");
const cuda = @import("cuda");
const engine = @import("cuda_engine.zig");
const Cache = @import("cuda_cache.zig").Cache;

pub const max_rows = 4096;

test {
    _ = @import("cuda_hidden_test.zig");
}

/// Single-threaded, engine-bound owner. Keep it and its bound cache alive until deinit.
pub const Rows = struct {
    engine: *engine.Engine,
    buffer: cuda.DeviceBuffer,
    rows: usize,
    width: usize,
    generation: u64 = 0,
    filled: usize = 0,
    state: enum { collecting, writing, complete, invalid } = .collecting,
    cache: ?*Cache = null,
    cache_epoch: u64 = 0,

    pub fn init(e: *engine.Engine, rows: usize) !Rows {
        try e.validateHiddenProfile();
        if (rows == 0 or rows > max_rows or e.g.hidden == 0) return error.InvalidQwenHiddenShape;
        const bytes = try std.math.mul(usize, try std.math.mul(usize, rows, e.g.hidden), 2);
        return .{ .engine = e, .buffer = try cuda.DeviceBuffer.alloc(e.d, bytes), .rows = rows, .width = e.g.hidden };
    }

    /// A failed context bind retains the allocation so teardown can be retried safely.
    pub fn deinit(self: *Rows) void {
        self.cancel() catch return;
        cuda.completion.drain(self.engine.d, self.engine.stream, "hidden rows still have an active device consumer");
        self.buffer.free();
        self.* = undefined;
    }

    fn bind(self: *const Rows) !void {
        const e = self.engine;
        if (e.context == cuda.copy_source.current()) return;
        try e.d.check(e.d.api.cuCtxSetCurrent(e.context), "cuCtxSetCurrent");
        cuda.copy_source.setCurrent(e.context);
    }

    pub fn window(self: *Rows, offset: usize, rows: usize) !Window {
        const w: Window = .{ .owner = self, .generation = self.generation, .offset = offset, .rows = rows };
        try w.validate(self.engine, rows);
        return w;
    }

    pub fn completed(self: *Rows) !Record {
        const r: Record = .{ .owner = self, .generation = self.generation };
        _ = try r.device();
        return r;
    }

    /// Also resets the bound cache. A failed reset leaves this record unusable.
    pub fn reset(self: *Rows) !void {
        if (self.state == .writing) return error.QwenHiddenInFlight;
        self.state = .invalid;
        self.generation = try std.math.add(u64, self.generation, 1);
        try self.bind();
        cuda.completion.drain(self.engine.d, self.engine.stream, "hidden reset did not drain");
        errdefer cuda.completion.drain(self.engine.d, self.engine.stream, "failed hidden reset did not drain");
        if (self.cache) |cache| {
            if (cache.reset_epoch == self.cache_epoch) {
                try cache.reset(self.engine);
                cuda.completion.drain(self.engine.d, self.engine.stream, "hidden cache reset did not drain");
                self.cache_epoch = cache.reset_epoch;
            } else self.cache = null;
        }
        self.filled = 0;
        self.state = .collecting;
    }

    pub fn cancel(self: *Rows) !void {
        try self.bind();
        if (self.state == .writing) cuda.completion.drain(self.engine.d, self.engine.stream, "hidden cancellation did not drain");
        self.state = .invalid;
        if (self.cache) |cache| {
            if (cache.reset_epoch == self.cache_epoch) cache.hidden_invalid = true;
        }
    }
};

pub const Window = struct {
    owner: *Rows,
    generation: u64,
    offset: usize,
    rows: usize,

    pub fn validate(w: Window, e: *const engine.Engine, rows: usize) !void {
        const o = w.owner;
        if (w.generation != o.generation) return error.StaleQwenHiddenView;
        if (o.state != .collecting) return error.QwenHiddenNotCollecting;
        if (o.engine != e or o.buffer.d != e.d) return error.QwenHiddenDeviceMismatch;
        if (o.width != e.g.hidden or o.rows == 0 or o.rows > max_rows or w.rows != rows or rows == 0 or
            w.offset > o.rows or rows > o.rows - w.offset) return error.InvalidQwenHiddenShape;
        const bytes = try std.math.mul(usize, try std.math.mul(usize, o.rows, o.width), 2);
        if (o.buffer.len != bytes or o.buffer.ptr == 0) return error.InvalidQwenHiddenShape;
        if (w.offset != o.filled) return error.QwenHiddenWriteOrder;
    }

    pub fn validateCache(w: Window, cache: *const Cache) !void {
        const o = w.owner;
        if (cache.owner != o.engine) return error.QwenHiddenDeviceMismatch;
        if (cache.hidden_invalid) return error.QwenHiddenNeedsReset;
        if (cache.len != w.offset or cache.written != cache.len or cache.last != null or cache.delta != 0 or
            cache.cap < o.rows or cache.cap > std.math.maxInt(c_int)) return error.InvalidQwenHiddenCache;
        if (o.cache) |bound| {
            if (bound != cache or o.cache_epoch != cache.reset_epoch) return error.QwenHiddenCacheMismatch;
        } else if (w.offset != 0) return error.InvalidQwenHiddenCache;
    }

    pub fn begin(w: Window, cache: *Cache) void {
        if (w.generation != w.owner.generation or w.owner.state != .collecting or w.offset != w.owner.filled) return;
        w.owner.cache = cache;
        w.owner.cache_epoch = cache.reset_epoch;
        w.owner.state = .writing;
    }

    pub fn fail(w: Window) void {
        if (w.generation != w.owner.generation or w.owner.state == .invalid) return;
        w.owner.state = .invalid;
        if (w.owner.cache) |cache| {
            if (cache.reset_epoch == w.owner.cache_epoch) cache.hidden_invalid = true;
        }
        cuda.completion.drain(w.owner.engine.d, w.owner.engine.stream, "failed hidden forward did not drain");
    }

    /// Called after graph replay, never during capture. Completion precedes publication and scratch reuse.
    pub fn copy(w: Window, source: u64) !void {
        const o = w.owner;
        if (w.generation != o.generation) return error.StaleQwenHiddenView;
        if (o.state != .writing) return error.QwenHiddenNotWriting;
        errdefer w.fail();
        const row_bytes = o.width * 2;
        try o.buffer.copyFrom(w.offset * row_bytes, source, w.rows * row_bytes, @as(?cuda.abi.Stream, o.engine.stream.handle));
        try o.engine.stream.synchronize();
    }

    pub fn commit(w: Window) void {
        if (w.generation != w.owner.generation or w.owner.state != .writing or w.offset != w.owner.filled) return;
        w.owner.filled += w.rows;
        w.owner.state = if (w.owner.filled == w.owner.rows) .complete else .collecting;
    }
};

/// Borrowed until the next reset or cancellation. Access checks the owner's generation and completion.
pub const Record = struct {
    owner: *const Rows,
    generation: u64,

    pub fn device(r: Record) !u64 {
        const o = r.owner;
        if (r.generation != o.generation) return error.StaleQwenHiddenView;
        if (o.state != .complete or o.filled != o.rows) return error.QwenHiddenIncomplete;
        if (o.engine.context != cuda.copy_source.current()) return error.QwenHiddenDeviceMismatch;
        const cache = o.cache orelse return error.QwenHiddenIncomplete;
        if (cache.hidden_invalid or cache.reset_epoch != o.cache_epoch or cache.len != o.rows) return error.QwenHiddenNeedsReset;
        return o.buffer.ptr;
    }
};
