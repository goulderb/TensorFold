//! Owning native vision checkpoint uploads and activation storage, NOT an encoder.
//! No buffers returned here contain encoded embeddings. Pipeline composition and
//! target runtime/numerical qualification are separate. Current CUDA context and
//! driver outlive owners; caller serializes use and completes all work before
//! re-staging input, reusing buffers or deinit. These owners do not own a stream.
const std = @import("std");
const builtin = @import("builtin");
pub const cuda = @import("cuda");
pub const host = @import("cuda_vision_weights.zig");
pub const Config = @import("cuda_vision_config.zig").Config;
const pre = @import("cuda_vision_preprocess.zig");
const rope = @import("cuda_vision_rope.zig");
const root_count = @typeInfo(host.Root).@"enum".field_names.len;
const block_count = @typeInfo(host.Block).@"enum".field_names.len;
const alignment = 256;
pub const DType = enum {
    bf16,
    f32,
    bytes,
    fn size(self: DType) usize {
        return switch (self) {
            .bf16 => 2,
            .f32 => 4,
            .bytes => 1,
        };
    }
};
pub const Region = struct {
    offset: usize = 0,
    len: usize = 0,
    dtype: DType = .bf16,
    shape: [5]usize = @splat(1),
    rank: u8 = 0,
};
fn append(total: *usize, dtype: DType, dims: []const usize) !Region {
    if (dims.len == 0 or dims.len > 5) return error.InvalidVisionMemoryShape;
    var n = dtype.size();
    for (dims) |dim| n = std.math.mul(usize, n, dim) catch return error.InvalidVisionMemoryShape;
    const at = std.math.add(usize, total.*, alignment - 1) catch return error.InvalidVisionMemoryShape;
    const offset = at & ~@as(usize, alignment - 1);
    total.* = std.math.add(usize, offset, n) catch return error.InvalidVisionMemoryShape;
    var r: Region = .{ .offset = offset, .len = n, .dtype = dtype, .rank = @intCast(dims.len) };
    @memcpy(r.shape[0..dims.len], dims);
    return r;
}
fn budget(total: usize, maximum: usize) !void {
    if (maximum == 0 or total > maximum) return error.VisionMemoryBudgetExceeded;
}
fn nativeHost() !void {
    if (builtin.target.cpu.arch.endian() != .little) return error.UnsupportedVisionHostEndian;
}
fn allocate(d: *const cuda.Driver, len: usize) !cuda.DeviceBuffer {
    var b = try cuda.DeviceBuffer.alloc(d, len);
    errdefer b.free();
    if (len == 0 or b.ptr == 0 or b.ptr % alignment != 0) return error.InvalidVisionDeviceAllocation;
    _ = std.math.add(u64, b.ptr, len) catch return error.InvalidVisionDeviceAllocation;
    return b;
}
fn upload(pool: cuda.DeviceBuffer, offset: usize, bytes: []const u8) !void {
    pool.upload(offset, bytes) catch |err| {
        cuda.completion.drain(pool.d, .{ .d = pool.d, .handle = null }, "a vision upload's default stream and context did not drain");
        return err;
    };
}
fn view(pool: cuda.DeviceBuffer, r: Region) !cuda.DeviceBuffer {
    if (r.offset > pool.len or r.len > pool.len - r.offset) return error.InvalidVisionMemoryRegion;
    const ptr = std.math.add(u64, pool.ptr, r.offset) catch return error.InvalidVisionMemoryRegion;
    _ = std.math.add(u64, ptr, r.len) catch return error.InvalidVisionMemoryRegion;
    return .{ .d = pool.d, .ptr = if (r.len == 0) 0 else ptr, .len = r.len, .borrowed = true };
}

pub const WeightLayout = struct {
    a: std.mem.Allocator,
    c: Config,
    regions: []Region,
    bytes: usize,
    pub fn init(a: std.mem.Allocator, c: Config, max_bytes: usize) !WeightLayout {
        var specs = try host.Specs.init(c);
        const regions = try a.alloc(Region, specs.count());
        errdefer a.free(regions);
        var bytes: usize = 0;
        var name: [256]u8 = undefined;
        var i: usize = 0;
        while (try specs.next(&name)) |spec| {
            regions[i] = try append(&bytes, .bf16, spec.dims());
            try budget(bytes, max_bytes);
            i += 1;
        }
        return .{ .a = a, .c = c, .regions = regions, .bytes = bytes };
    }
    pub fn deinit(self: *WeightLayout) void {
        self.a.free(self.regions);
        self.* = undefined;
    }
    fn check(self: *const WeightLayout, w: *const host.Weights) !void {
        if (!std.meta.eql(self.c, w.c) or self.regions.len != w.tensors.len or (w.quantized.len != 0 and w.quantized.len != w.tensors.len)) return error.InvalidVisionWeightTable;
        for (self.regions, w.tensors, 0..) |r, t, i| {
            if (t.rank != r.rank or !std.mem.eql(usize, t.shape[0..t.rank], r.shape[0..r.rank])) return error.InvalidVisionWeightShape;
            if (t.dtype != .bf16 and t.dtype != .f16 and t.dtype != .f32) return error.UnsupportedVisionDType;
            const bytes = std.math.mul(usize, r.len / 2, t.dtype.size()) catch return error.InvalidVisionWeightShape;
            if ((w.quantized.len == 0 or w.quantized[i] == null) and t.bytes.len != bytes) return error.InvalidVisionWeightShape;
        }
    }
};

pub const DeviceWeights = struct {
    layout: WeightLayout,
    pool: cuda.DeviceBuffer,
    /// Rechecks the entire borrowed table. All layouts and finite conversions
    /// pass before the first device allocation. Conversion scratch is bounded by
    /// the largest tensor, at the cost of two CPU conversion passes. Uploads are
    /// synchronous: neither maps nor conversion scratch need outlive init.
    pub fn init(a: std.mem.Allocator, d: *const cuda.Driver, w: *const host.Weights, max_bytes: usize) !DeviceWeights {
        try nativeHost();
        var layout = try WeightLayout.init(a, w.c, max_bytes);
        errdefer layout.deinit();
        try layout.check(w);
        for (w.tensors, 0..) |_, i| {
            const converted = try w.converted(a, i);
            a.free(converted);
        }
        var pool = try allocate(d, layout.bytes);
        errdefer pool.free();
        for (layout.regions, 0..) |r, i| {
            const converted = try w.converted(a, i);
            defer a.free(converted);
            try upload(pool, r.offset, std.mem.sliceAsBytes(converted));
        }
        return .{ .layout = layout, .pool = pool };
    }
    pub fn tensor(self: *const DeviceWeights, index: usize) !cuda.DeviceBuffer {
        if (index >= self.layout.regions.len) return error.InvalidVisionWeightIndex;
        return view(self.pool, self.layout.regions[index]);
    }
    pub fn root(self: *const DeviceWeights, kind: host.Root) !cuda.DeviceBuffer {
        return self.tensor(@backingInt(kind));
    }
    pub fn block(self: *const DeviceWeights, layer: u32, kind: host.Block) !cuda.DeviceBuffer {
        if (layer >= self.layout.c.depth) return error.InvalidVisionWeightIndex;
        return self.tensor(root_count + @as(usize, layer) * block_count + @backingInt(kind));
    }
    pub fn deinit(self: *DeviceWeights) void {
        self.pool.free();
        self.layout.deinit();
        self.* = undefined;
    }
};

pub const Geometry = struct {
    groups: u32,
    gh: u32,
    gw: u32,
    pub fn sequence(self: Geometry) !u32 {
        if (self.groups == 0 or self.groups > 256 or self.gh == 0 or self.gw == 0 or self.gh % 2 != 0 or self.gw % 2 != 0) return error.InvalidVisionGrid;
        const s = std.math.mul(u32, self.gh, self.gw) catch return error.InvalidVisionGrid;
        if (s > 32000 or @as(u64, s) * self.groups > 32000) return error.InvalidVisionGrid;
        return s;
    }
    pub fn rows(self: Geometry) !u32 {
        return (try self.sequence()) * self.groups;
    }
};
pub const Workspaces = struct {
    convolution: usize = 0,
    attention: usize = 0,
    linear: usize = 0,
    pub fn maximum(self: Workspaces) usize {
        return @max(self.convolution, self.attention, self.linear);
    }
};
pub const Slot = enum {
    pixels,
    patches,
    x,
    norm,
    qkv,
    q,
    k,
    v,
    attention,
    update,
    gate,
    up,
    merged,
    projected,
    projector_gate,
    projector_up,
    output,
    cosine,
    sine,
    workspace,
};
const slot_count = @typeInfo(Slot).@"enum".field_names.len;
pub const ActivationLayout = struct {
    c: Config,
    geometry: Geometry,
    regions: [slot_count]Region,
    bytes: usize,
    pub fn init(c: Config, g: Geometry, work: Workspaces, max_bytes: usize) !ActivationLayout {
        try c.check();
        const rows = try g.rows();
        const sequence = try g.sequence();
        const features = rows / 4;
        if (c.hidden > 4096 or c.output > 4096 or c.heads > 128 or c.headDim() > 256 or c.intermediate > std.math.maxInt(c_int) or c.projection_intermediate > std.math.maxInt(c_int)) return error.UnsupportedVisionMemoryShape;
        var self: ActivationLayout = .{ .c = c, .geometry = g, .regions = undefined, .bytes = 0 };
        for (std.enums.values(Slot)) |slot| {
            var dims: [5]usize = @splat(1);
            var rank: usize = 2;
            var dtype: DType = .bf16;
            switch (slot) {
                .pixels, .patches => {
                    dims[0] = rows;
                    dims[1] = c.patchWidth();
                    if (slot == .pixels) dtype = .f32;
                },
                .x, .norm, .q, .k, .v, .attention, .update => {
                    dims[0] = rows;
                    dims[1] = c.hidden;
                },
                .qkv => {
                    dims[0] = rows;
                    dims[1] = 3;
                    dims[2] = c.heads;
                    dims[3] = c.headDim();
                    rank = 4;
                },
                .gate, .up => {
                    dims[0] = rows;
                    dims[1] = c.intermediate;
                },
                .merged, .projected, .output => {
                    dims[0] = features;
                    dims[1] = c.output;
                },
                .projector_gate, .projector_up => {
                    dims[0] = features;
                    dims[1] = c.projection_intermediate;
                },
                .cosine, .sine => {
                    dims[0] = sequence;
                    dims[1] = c.headDim();
                    dtype = .f32;
                },
                .workspace => {
                    dims[0] = work.maximum();
                    rank = 1;
                    dtype = .bytes;
                },
            }
            const r = try append(&self.bytes, dtype, dims[0..rank]);
            if (dtype != .bytes and r.len / dtype.size() > std.math.maxInt(c_int)) return error.UnsupportedVisionMemoryShape;
            self.regions[@backingInt(slot)] = r;
            try budget(self.bytes, max_bytes);
        }
        return self;
    }
    pub fn region(self: *const ActivationLayout, slot: Slot) Region {
        return self.regions[@backingInt(slot)];
    }
};

pub const Activations = struct {
    layout: ActivationLayout,
    pool: cuda.DeviceBuffer,
    pixels_ready: bool = false,
    /// One rectangular B,S pass; groups attend independently. RoPE has S rows,
    /// not B*S. A single shared workspace is valid ONLY for serialized operators
    /// on one stream. All other activation/output regions remain uninitialized.
    pub fn init(a: std.mem.Allocator, d: *const cuda.Driver, c: Config, g: Geometry, work: Workspaces, max_bytes: usize) !Activations {
        try nativeHost();
        const layout = try ActivationLayout.init(c, g, work, max_bytes);
        var table = try rope.Table.init(a, c, .{ 1, g.gh, g.gw });
        defer table.deinit(a);
        var pool = try allocate(d, layout.bytes);
        errdefer pool.free();
        try upload(pool, layout.region(.cosine).offset, std.mem.sliceAsBytes(table.cos));
        try upload(pool, layout.region(.sine).offset, std.mem.sliceAsBytes(table.sin));
        return .{ .layout = layout, .pool = pool };
    }
    pub fn buffer(self: *const Activations, slot: Slot) !cuda.DeviceBuffer {
        return view(self.pool, self.layout.region(slot));
    }
    /// Already normalized FP32 merge-block patches. This only stages pixels,
    /// not BF16 casting, convolution, attention or the tower/projector.
    pub fn stagePixels(self: *Activations, pixels: []const f32) !void {
        self.pixels_ready = false;
        const r = self.layout.region(.pixels);
        if (pixels.len != r.len / 4) return error.InvalidVisionGrid;
        for (pixels) |x| if (!std.math.isFinite(x)) return error.NonFiniteVisionPixels;
        try upload(self.pool, r.offset, std.mem.sliceAsBytes(pixels));
        self.pixels_ready = true;
    }
    /// Still-picture groups must share one canvas grid. Validate the whole batch
    /// before copying any input; caller retains each borrowed Prepared only until
    /// this synchronous staging call returns, even on error.
    pub fn stagePrepared(self: *Activations, images: []const pre.Prepared) !void {
        self.pixels_ready = false;
        const g = self.layout.geometry;
        if (images.len != g.groups) return error.InvalidVisionGrid;
        for (images) |image| {
            try image.check(self.layout.c);
            if (image.grid[1] != g.gh or image.grid[2] != g.gw) return error.InvalidVisionGrid;
            for (image.pixels) |x| if (!std.math.isFinite(x)) return error.NonFiniteVisionPixels;
        }
        var at = self.layout.region(.pixels).offset;
        for (images) |image| {
            const bytes = std.mem.sliceAsBytes(image.pixels);
            try upload(self.pool, at, bytes);
            at += bytes.len;
        }
        self.pixels_ready = true;
    }
    pub fn deinit(self: *Activations) void {
        self.pool.free();
        self.* = undefined;
    }
};

test {
    _ = @import("cuda_vision_engine_memory_test.zig");
}
