//! Encoded still pictures -> native canonical pixels -> exact CPU patches ->
//! canonical stored-format Tower -> stable request-owned BF16 rows and checked binding.
//! The older Engine adapter is retained only for explicit conversion/numerical qualification.
const std = @import("std");
const codec = @import("cuda_vision_codec.zig");
const pre = @import("cuda_vision_preprocess.zig");
const resize = @import("cuda_vision_resize.zig");
const encoder = @import("cuda_vision_engine.zig");
const binding = @import("cuda_vision_binding.zig");
pub const cuda = encoder.cuda;
pub const Config = @import("cuda_vision_config.zig").Config;
const Allocator = std.mem.Allocator;

test {
    _ = @import("cuda_vision_pipeline_test.zig");
}

pub const Options = struct {
    max_images: usize = 50,
    image_tokens: u32 = 2048,
    request_image_tokens: u32 = 16384,
    min_tokens: u32 = 16,
    image_limits: pre.Limits = .{},
    max_total_encoded_bytes: ?usize = null,
    max_total_pixels: ?usize = null,
    max_prepared_bytes: usize = 512 * 1024 * 1024,
    prompt_limit: usize,
    fn check(self: Options) !void {
        if (self.max_images == 0 or self.max_images > 256 or self.min_tokens == 0 or
            self.image_tokens < self.min_tokens or self.image_tokens > 8000 or
            self.request_image_tokens < 16 or self.request_image_tokens > 262144 or
            self.max_prepared_bytes == 0 or self.prompt_limit == 0 or
            self.image_limits.max_dimension == 0 or self.image_limits.max_dimension > 8192 or
            self.image_limits.max_pixels == 0 or self.image_limits.max_pixels > 16 * 1024 * 1024)
            return error.InvalidVisionPipelineBudget;
        if (self.max_total_encoded_bytes) |n| if (n == 0) return error.InvalidVisionPipelineBudget;
        if (self.max_total_pixels) |n| if (n == 0) return error.InvalidVisionPipelineBudget;
    }
    pub fn pictureBudget(self: Options, count: usize) pre.Budget {
        return .{ .min_tokens = self.min_tokens, .max_tokens = @max(self.min_tokens, @min(self.image_tokens, self.request_image_tokens / @as(u32, @intCast(@max(1, count))))) };
    }
    fn byteCap(self: Options) usize {
        return self.max_total_encoded_bytes orelse if (self.max_images <= 4) 20 * 1024 * 1024 else 64 * 1024 * 1024;
    }
    fn pixelCap(self: Options) usize {
        return self.max_total_pixels orelse @max(@as(usize, 32 * 1024 * 1024), self.max_images * 8 * 1024 * 1024);
    }
};
pub const EncodeOptions = struct {
    /// Also the main decoder binding stream. The caller owns it and its context.
    stream: cuda.Stream,
    replica: u64,
    world: u32 = 1,
    rank: u32 = 0,
    expected_context: cuda.abi.Context = null,
};
pub const Pass = struct { engine: *encoder.Engine, first_image: usize, images: usize, first_row: usize, rows: usize };
pub const Plan = struct {
    a: Allocator,
    storage: []Pass,
    count: usize,
    pub fn passes(self: Plan) []const Pass {
        return self.storage[0..self.count];
    }
    pub fn deinit(self: *Plan) void {
        self.a.free(self.storage);
        self.* = undefined;
    }
};

fn current(d: *const cuda.Driver, expected: cuda.abi.Context) !cuda.abi.Context {
    var context: cuda.abi.Context = null;
    try d.check(d.api.cuCtxGetCurrent(&context), "cuCtxGetCurrent");
    if (context == null or (expected != null and context != expected)) return error.InvalidVisionPipelineContext;
    return context;
}
fn checkPrompt(c: Config, ids: []const u32, limit: usize) !void {
    if (limit == 0 or ids.len == 0 or ids.len >= limit or ids.len > 262144) return error.ImagePromptTooLong;
    if (c.image_token >= 154880) return error.BadToken;
    for (ids) |id| if (id >= 154880) return error.BadToken;
}
fn rowsBytes(rows: usize, width: u32) !usize {
    const values = std.math.mul(usize, rows, width) catch return error.InvalidVisionPipelineShape;
    return std.math.mul(usize, values, 2) catch error.InvalidVisionPipelineShape;
}
fn overlaps(x: cuda.DeviceBuffer, y: cuda.DeviceBuffer) !bool {
    const xe = std.math.add(u64, x.ptr, x.len) catch return error.InvalidVisionPipelineAllocation;
    const ye = std.math.add(u64, y.ptr, y.len) catch return error.InvalidVisionPipelineAllocation;
    return x.len != 0 and y.len != 0 and x.ptr < ye and y.ptr < xe;
}

pub const Request = struct {
    a: Allocator,
    c: Config,
    images: []pre.Prepared,
    ids: []u32,
    positions: []u32,
    prompt_limit: usize,
    rows: usize,
    table: ?binding.Table = null,
    state: enum { prepared, encoding, failed, ready } = .prepared,
    pending: bool = false,
    /// A failed native call may still reference this exclusively borrowed owner.
    pending_encoder: ?*encoder.Engine = null,

    /// Codec library is borrowed during preparation only. Encoded bytes and caller
    /// token IDs can be released after return; every CPU result is request-owned.
    pub fn prepare(a: Allocator, library: *codec.Library, c: Config, encoded: []const []const u8, ids: []const u32, options: Options) !Request {
        return prepareBackend(a, library.backend(), c, encoded, ids, options);
    }
    /// Uses the same native codec API table; exposed for CPU native-API recordings,
    /// not an encoder callback or a substitute for real codec qualification.
    pub fn prepareBackend(a: Allocator, backend: codec.Backend, c: Config, encoded: []const []const u8, ids: []const u32, options: Options) !Request {
        try options.check();
        try c.check();
        if (c.output != 4096) return error.VisionTextWidthMismatch;
        if (encoded.len == 0 or encoded.len > options.max_images) return error.InvalidVisionImageCount;
        try checkPrompt(c, ids, options.prompt_limit);
        var total_encoded: usize = 0;
        for (encoded) |bytes| {
            if (bytes.len == 0 or bytes.len > codec.max_encoded_bytes) return error.ImageEncodedByteLimit;
            total_encoded = std.math.add(usize, total_encoded, bytes.len) catch return error.ImageEncodedByteLimit;
            if (total_encoded > options.byteCap()) return error.ImageEncodedByteLimit;
        }
        const images = try a.alloc(pre.Prepared, encoded.len);
        errdefer a.free(images);
        var initialized: usize = 0;
        errdefer for (images[0..initialized]) |*p| p.deinit(a);
        var pixels: usize = 0;
        var prepared_bytes: usize = 0;
        var rows: usize = 0;
        for (encoded, images) |bytes, *prepared| {
            if (pixels >= options.pixelCap()) return error.ImageDecodedPixelLimit;
            var limits = options.image_limits;
            limits.max_pixels = @min(limits.max_pixels, options.pixelCap() - pixels);
            var rgb = try backend.decode(a, bytes, limits);
            defer rgb.deinit(a);
            pixels = try std.math.add(usize, pixels, @as(usize, rgb.shape.height) * rgb.shape.width);
            const budget = options.pictureBudget(encoded.len);
            const shape = try pre.canvas(rgb.shape, budget);
            const patch_rows = try std.math.mul(usize, shape.height / c.patch, shape.width / c.patch);
            const values = try std.math.mul(usize, patch_rows, c.patchWidth());
            prepared_bytes = try std.math.add(usize, prepared_bytes, try std.math.mul(usize, values, @sizeOf(f32)));
            if (prepared_bytes > options.max_prepared_bytes) return error.VisionPreparedBudgetExceeded;
            prepared.* = try pre.prepareRgb(a, c, rgb.borrowed(), budget, limits, resize.resizer);
            initialized += 1;
            rows = try std.math.add(usize, rows, prepared.visualTokens());
        }
        const positions = try pre.imagePositions(a, c, ids, images, options.prompt_limit);
        errdefer a.free(positions);
        const owned_ids = try a.dupe(u32, ids);
        return .{ .a = a, .c = c, .images = images, .ids = owned_ids, .positions = positions, .prompt_limit = options.prompt_limit, .rows = rows };
    }
    fn validate(self: *const Request) !void {
        try self.c.check();
        if (self.c.output != 4096 or self.images.len == 0 or self.images.len > 256 or self.rows == 0) return error.InvalidVisionPipelineShape;
        try checkPrompt(self.c, self.ids, self.prompt_limit);
        var rows: usize = 0;
        for (self.images) |p| {
            try p.check(self.c);
            for (p.pixels) |x| if (!std.math.isFinite(x)) return error.NonFiniteVisionPixels;
            rows = try std.math.add(usize, rows, p.visualTokens());
        }
        if (rows != self.rows) return error.InvalidVisionPipelineShape;
        const positions = try pre.imagePositions(self.a, self.c, self.ids, self.images, self.prompt_limit);
        defer self.a.free(positions);
        if (!std.mem.eql(u32, positions, self.positions)) return error.ImageTokenCountMismatch;
        _ = try rowsBytes(self.rows, self.c.output);
    }
    /// Consecutive equal-grid pictures can form independent rectangular groups.
    /// Different grids are never padded/flattened into one attention sequence.
    /// Every pass and every owner is validated before any CUDA write/allocation.
    pub fn plan(self: *const Request, d: *const cuda.Driver, engines: []const *encoder.Engine) !Plan {
        try self.validate();
        if (engines.len == 0) return error.MissingVisionEncoder;
        for (engines) |e| {
            const g = e.activations.layout.geometry;
            _ = try g.rows();
            if (e.poisoned) return error.VisionEncoderPoisoned;
            if (!std.meta.eql(e.weights.layout.c, self.c) or !std.meta.eql(e.activations.layout.c, self.c)) return error.VisionPipelineConfigMismatch;
            if (e.stream.d != d or e.weights.pool.d != d or e.activations.pool.d != d) return error.VisionPipelineDriverMismatch;
            if (e.activations.pool.ptr == 0 or e.activations.pool.len != e.activations.layout.bytes) return error.InvalidVisionPipelineAllocation;
        }
        const passes = try self.a.alloc(Pass, self.images.len);
        errdefer self.a.free(passes);
        var count: usize = 0;
        var first: usize = 0;
        var row: usize = 0;
        while (first < self.images.len) {
            const start = first;
            const grid = self.images[start].grid;
            var end = start + 1;
            while (end < self.images.len and std.mem.eql(u32, &grid, &self.images[end].grid)) : (end += 1) {}
            // Complete tiling, not a greedy dead end (six pictures with 4/3-group
            // owners require 3+3). Bounded by the admitted 256-picture request.
            var viable: [257]bool = @splat(false);
            const run = end - start;
            viable[run] = true;
            var at = run;
            while (at != 0) {
                at -= 1;
                for (engines) |e| {
                    const g = e.activations.layout.geometry;
                    if (g.gh == grid[1] and g.gw == grid[2] and g.groups <= run - at and viable[at + g.groups]) viable[at] = true;
                }
            }
            if (!viable[0]) return error.MissingMatchingVisionEncoder;
            while (first < end) {
                var selected: ?*encoder.Engine = null;
                var groups: usize = 0;
                for (engines) |e| {
                    const g = e.activations.layout.geometry;
                    if (g.gh == grid[1] and g.gw == grid[2] and g.groups > groups and g.groups <= end - first and viable[first - start + g.groups]) {
                        selected = e;
                        groups = g.groups;
                    }
                }
                const e = selected orelse return error.MissingMatchingVisionEncoder;
                const rows = @as(usize, try e.activations.layout.geometry.rows()) / 4;
                passes[count] = .{ .engine = e, .first_image = first, .images = groups, .first_row = row, .rows = rows };
                count += 1;
                first += groups;
                row += rows;
            }
        }
        if (row != self.rows) return error.InvalidVisionPipelineShape;
        return .{ .a = self.a, .storage = passes, .count = count };
    }

    /// The caller grants EXCLUSIVE use of every supplied Engine through this call
    /// and, on error, through successful recover/deinit. No direct encode/deinit
    /// or another pipeline may use those owners concurrently. encodePrepared's
    /// internal mutex alone cannot protect its returned output loan.
    /// Request must not move after the first binding is borrowed.
    pub fn encode(self: *Request, engines: []const *encoder.Engine, options: EncodeOptions) !void {
        if (self.state != .prepared or self.table != null) return error.VisionPipelineAlreadyEncoded;
        if (options.replica == 0 or (options.world != 1 and options.world != 2) or options.rank >= options.world) return error.InvalidVisionPipelineReplica;
        const d = options.stream.d;
        var passes = try self.plan(d, engines);
        defer passes.deinit();
        const context = try current(d, options.expected_context);
        const bytes = try rowsBytes(self.rows, self.c.output);
        var ptr: cuda.abi.DevicePtr = 0;
        try d.check(d.api.cuMemAlloc_v2(&ptr, bytes), "cuMemAlloc");
        cuda.accountExternalDevice(bytes);
        // Raw ownership permits checked free/retry, unlike DeviceBuffer.free's
        // fire-and-forget destructor. This allocation does not update its metrics.
        self.table = .{ .driver = d, .context = context, .stream = options.stream, .features = .{ .d = d, .ptr = ptr, .len = bytes, .borrowed = false }, .positions = self.positions, .ids = self.ids, .image_token = self.c.image_token, .width = self.c.output, .rows = self.rows, .ready = false, .generation = 1, .replica = options.replica, .world = options.world, .rank = options.rank };
        self.state = .encoding;
        errdefer {
            self.state = .failed;
            self.table.?.ready = false;
        }
        if (ptr == 0 or ptr % 256 != 0) return error.InvalidVisionPipelineAllocation;
        _ = std.math.add(u64, ptr, bytes) catch return error.InvalidVisionPipelineAllocation;
        for (passes.passes()) |pass| {
            if (try overlaps(self.table.?.features, pass.engine.activations.pool) or
                try overlaps(self.table.?.features, pass.engine.weights.pool)) return error.AliasedVisionPipelineOutput;
        }
        for (passes.passes()) |pass| {
            _ = try current(d, context);
            self.pending = true;
            self.pending_encoder = pass.engine;
            const out = try pass.engine.encodePrepared(self.images[pass.first_image..][0..pass.images]);
            const size = try rowsBytes(pass.rows, self.c.output);
            if (out.d != d or out.ptr == 0 or !out.borrowed or out.len != size) return error.InvalidVisionEncoderOutput;
            _ = std.math.add(u64, out.ptr, out.len) catch return error.InvalidVisionEncoderOutput;
            if (try overlaps(self.table.?.features, out)) return error.AliasedVisionPipelineOutput;
            try self.table.?.features.copyFrom(try rowsBytes(pass.first_row, self.c.output), out.ptr, size, options.stream.handle);
            // Cross-stream reads finish before the borrowed source can be reused.
            try options.stream.synchronize();
            self.pending = false;
            self.pending_encoder = null;
        }
        self.table.?.ready = true;
        self.state = .ready;
    }
    /// Default serving uses the canonical stored-format Tower, writing directly into this request's owned rows.
    /// The retained Engine-based encode above is an explicit conversion/numerical-qualification adapter only.
    pub fn encodeTower(self: *Request, tower: *@import("cuda_vision.zig").Tower, run: *const @import("cuda_vision.zig").Run, options: EncodeOptions) !void {
        if (self.state != .prepared or self.table != null) return error.VisionPipelineAlreadyEncoded;
        if (options.replica == 0 or (options.world != 1 and options.world != 2) or options.rank >= options.world) return error.InvalidVisionPipelineReplica;
        try self.validate();
        try @import("cuda_vision.zig").checkTowerConfig(self.c);
        if (self.c.eps != tower.eps or self.c.limit != tower.limit or self.c.theta != tower.theta) return error.VisionPipelineConfigMismatch;
        const d = options.stream.d;
        if (tower.d != d or run.stream.d != d or run.stream.handle != options.stream.handle) return error.VisionPipelineDriverMismatch;
        const context = try current(d, options.expected_context);
        const bytes = try rowsBytes(self.rows, self.c.output);
        var ptr: cuda.abi.DevicePtr = 0;
        try d.check(d.api.cuMemAlloc_v2(&ptr, bytes), "cuMemAlloc");
        cuda.accountExternalDevice(bytes);
        self.table = .{ .driver = d, .context = context, .stream = options.stream, .features = .{ .d = d, .ptr = ptr, .len = bytes, .borrowed = false }, .positions = self.positions, .ids = self.ids, .image_token = self.c.image_token, .width = self.c.output, .rows = self.rows, .ready = false, .generation = 1, .replica = options.replica, .world = options.world, .rank = options.rank };
        self.state = .encoding;
        errdefer {
            self.state = .failed;
            self.table.?.ready = false;
        }
        if (ptr == 0 or ptr % 256 != 0) return error.InvalidVisionPipelineAllocation;
        _ = std.math.add(u64, ptr, bytes) catch return error.InvalidVisionPipelineAllocation;
        inline for (.{ "mem", "inv_freq", "ws" }) |field| if (try overlaps(self.table.?.features, @field(tower, field))) return error.AliasedVisionPipelineOutput;
        var row: usize = 0;
        for (self.images) |image| {
            _ = try current(d, context);
            self.pending = true;
            try tower.encode(run, image.pixels, .{ image.grid[0], image.grid[1], image.grid[2] }, ptr + try rowsBytes(row, self.c.output));
            row += image.visualTokens();
            self.pending = false;
        }
        self.table.?.ready = true;
        self.state = .ready;
    }

    pub fn borrowedBinding(self: *Request) !binding.Binding {
        if (self.state != .ready or self.pending) return error.VisionPipelineNotReady;
        const table = if (self.table) |*table| table else return error.VisionPipelineNotReady;
        return table.binding();
    }
    /// Stronger than a stream fence: also covers failed legacy host uploads and
    /// encoder work on another stream. It does not unpoison an encoder or retry it.
    pub fn recover(self: *Request) !void {
        const table = if (self.table) |*table| table else return;
        _ = try current(table.driver, table.context);
        try table.driver.check(table.driver.api.cuCtxSynchronize(), "cuCtxSynchronize");
        self.pending = false;
        self.pending_encoder = null;
    }
    /// Caller stops main/binding use before cleanup. A failed context fence or
    /// native free retains every CPU/GPU owner and permits a later cleanup retry.
    pub fn deinit(self: *Request) !void {
        if (self.table) |*table| {
            table.ready = false;
            self.state = .failed;
            try self.recover();
            if (table.features.ptr != 0) {
                try table.driver.check(table.driver.api.cuMemFree_v2(table.features.ptr), "cuMemFree");
                cuda.unaccountExternalDevice(table.features.len);
            }
            table.features.ptr = 0;
        }
        for (self.images) |*p| p.deinit(self.a);
        self.a.free(self.images);
        self.a.free(self.positions);
        self.a.free(self.ids);
        self.* = undefined;
    }
};
