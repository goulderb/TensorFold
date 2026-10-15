//! CPU helpers and recordings of the actual native Engine/API path. No GPU,
//! model/library is loaded, and no numerical embeddings are manufactured.
const std = @import("std");
const p = @import("cuda_vision_pipeline.zig");
const e = @import("cuda_vision_engine.zig");
const codec = @import("cuda_vision_codec.zig");
const fixtures = @import("cuda_vision_codec_fixtures.zig");
const pre = @import("cuda_vision_preprocess.zig");
const resize = @import("cuda_vision_resize.zig");
const binding = @import("cuda_vision_binding.zig");
const cuda = e.cuda;
const lt = cuda.cublaslt;
const CM = e.conv.cudnn.test_support.Mock;
const SM = e.attention.sdpa.test_support.Mock;
const t = std.testing;
const a = t.allocator;

// The mock returns independent target RGBA DATA, not decoded-byte parity. Real
// orientation/white composition/CPU resize and normalized packing run afterwards.
const Pixels = struct {
    probes: usize = 0,
    decodes: usize = 0,
    fail_decode: usize = 0,
    fn of(context: ?*anyopaque) *Pixels {
        return @ptrCast(@alignCast(context.?));
    }
    fn fixture(bytes: [*c]const u8, n: usize) *const fixtures.Fixture {
        for (&fixtures.all) |*f| if (std.mem.eql(u8, f.encoded, bytes[0..n])) return f;
        unreachable;
    }
    fn probe(context: ?*anyopaque, _: u32, bytes: [*c]const u8, n: usize, _: u32, _: usize, w: [*c]u32, h: [*c]u32) callconv(.c) c_int {
        const self = of(context);
        self.probes += 1;
        const f = fixture(bytes, n);
        w.* = f.shape.width;
        h.* = f.shape.height;
        return codec.c.TF_GLM_CODEC_OK;
    }
    fn decode(context: ?*anyopaque, _: u32, bytes: [*c]const u8, n: usize, _: u32, _: u32, out: [*c]u8, size: usize) callconv(.c) c_int {
        const self = of(context);
        self.decodes += 1;
        if (self.decodes == self.fail_decode) return codec.c.TF_GLM_CODEC_INVALID;
        const f = fixture(bytes, n);
        std.debug.assert(size == f.rgba.len);
        @memcpy(out[0..size], f.rgba);
        return codec.c.TF_GLM_CODEC_OK;
    }
    fn features(_: ?*const anyopaque) callconv(.c) u32 {
        return codec.c.TF_GLM_CODEC_PNG | codec.c.TF_GLM_CODEC_JPEG | codec.c.TF_GLM_CODEC_WEBP;
    }
    const api: codec.Api = .{ .tf_glm_codec_abi = undefined, .tf_glm_codec_open = undefined, .tf_glm_codec_close = undefined, .tf_glm_codec_features = features, .tf_glm_codec_probe = probe, .tf_glm_codec_decode = decode };
    fn library(self: *Pixels) codec.Library {
        return .{ .lib = undefined, .api = api, .context = self };
    }
    fn backend(self: *Pixels) codec.Backend {
        return .{ .context = self, .api = &api, .features = features(self) };
    }
};
fn config() !p.Config {
    return @import("cuda_vision_config.zig").parse(a, @embedFile("fixtures_cuda_config_exl3.json"));
}
const cpu_options: p.Options = .{ .prompt_limit = 4096 };
fn prepare(alloc: std.mem.Allocator, pixels: *Pixels, c: p.Config, selected: []const usize, options: p.Options) !p.Request {
    const encoded = try alloc.alloc([]const u8, selected.len);
    defer alloc.free(encoded);
    var rows: usize = 0;
    for (selected, encoded) |index, *bytes| {
        const f = fixtures.all[index];
        bytes.* = f.encoded;
        const shape = try pre.canvas(f.oriented, options.pictureBudget(selected.len));
        rows += @as(usize, shape.height / c.patch) * (shape.width / c.patch) / 4;
    }
    const ids = try alloc.alloc(u32, rows + 2);
    defer alloc.free(ids);
    @memset(ids, c.image_token);
    ids[0] = 5;
    ids[ids.len - 1] = 7;
    var native_library = pixels.library();
    return p.Request.prepare(alloc, &native_library, c, encoded, ids, options);
}
fn indexOf(name: []const u8) usize {
    for (fixtures.all, 0..) |f, i| if (std.mem.eql(u8, name, f.name)) return i;
    unreachable;
}

const Driver = struct {
    const Copy = struct { dst: u64, src: u64, size: usize, stream: cuda.abi.Stream };
    var live: usize = 0;
    var modules: usize = 0;
    var next: u64 = 0x400000000;
    var context: cuda.abi.Context = @ptrFromInt(0x1230);
    var allocs: usize = 0;
    var frees: usize = 0;
    var uploads: usize = 0;
    var launches: usize = 0;
    var lookups: usize = 0;
    var syncs: usize = 0;
    var fences: usize = 0;
    var copies: usize = 0;
    var records: [1024]Copy = undefined;
    var fail_alloc: bool = false;
    var fail_free: bool = false;
    var fail_fence: bool = false;
    var fail_target_sync: bool = false;
    var fail_upload: usize = 0;
    var fail_launch: usize = 0;
    var fail_copy: usize = 0;
    var forced_ptr: ?u64 = null;
    var unfenced_copy: bool = false;
    var bad_reuse: bool = false;
    fn load(out: *cuda.abi.Module, _: ?*const anyopaque) callconv(.c) cuda.abi.Result {
        modules += 1;
        out.* = @ptrFromInt(0x1000);
        return 0;
    }
    fn unload(_: cuda.abi.Module) callconv(.c) cuda.abi.Result {
        modules -= 1;
        return 0;
    }
    fn function(out: *cuda.abi.Function, _: cuda.abi.Module, name: [*:0]const u8) callconv(.c) cuda.abi.Result {
        const i = lookups % 6;
        std.debug.assert(std.mem.eql(u8, std.mem.span(name), e.ops.symbols[i]));
        lookups += 1;
        out.* = @ptrFromInt(0x8000 + i * 16);
        return 0;
    }
    fn alloc(out: *u64, _: usize) callconv(.c) cuda.abi.Result {
        allocs += 1;
        if (fail_alloc) return cuda.abi.error_not_ready;
        live += 1;
        out.* = forced_ptr orelse next;
        next += 0x200000000;
        return 0;
    }
    fn free(_: u64) callconv(.c) cuda.abi.Result {
        frees += 1;
        if (fail_free) return cuda.abi.error_not_ready;
        live -= 1;
        return 0;
    }
    fn upload(_: u64, _: ?*const anyopaque, _: usize) callconv(.c) cuda.abi.Result {
        if (unfenced_copy) bad_reuse = true;
        uploads += 1;
        return if (uploads == fail_upload) cuda.abi.error_not_ready else 0;
    }
    fn copy(dst: u64, src: u64, size: usize, stream: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        records[copies] = .{ .dst = dst, .src = src, .size = size, .stream = stream };
        copies += 1;
        unfenced_copy = true;
        return if (copies == fail_copy) cuda.abi.error_not_ready else 0;
    }
    fn sync(stream: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        syncs += 1;
        if (stream == @as(cuda.abi.Stream, @ptrFromInt(2))) {
            if (fail_target_sync) return cuda.abi.error_not_ready;
            unfenced_copy = false;
        }
        return 0;
    }
    fn getContext(out: *cuda.abi.Context) callconv(.c) cuda.abi.Result {
        out.* = context;
        return 0;
    }
    fn fence() callconv(.c) cuda.abi.Result {
        fences += 1;
        if (fail_fence) return cuda.abi.error_not_ready;
        unfenced_copy = false;
        return 0;
    }
    fn launch(_: cuda.abi.Function, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: cuda.abi.Stream, _: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
        launches += 1;
        return if (launches == fail_launch) cuda.abi.error_not_ready else 0;
    }
    fn value() cuda.Driver {
        var api: cuda.abi.Api = undefined;
        api.cuModuleLoadData = load;
        api.cuModuleUnload = unload;
        api.cuModuleGetFunction = function;
        api.cuMemAlloc_v2 = alloc;
        api.cuMemFree_v2 = free;
        api.cuMemcpyHtoD_v2 = upload;
        api.cuMemcpyDtoDAsync_v2 = copy;
        api.cuStreamSynchronize = sync;
        api.cuCtxGetCurrent = getContext;
        api.cuCtxSynchronize = fence;
        api.cuLaunchKernel = launch;
        return .{ .lib = undefined, .api = api };
    }
};
const LT = struct {
    var next: usize = 1;
    var live: usize = 0;
    var matmuls: usize = 0;
    fn create(comptime T: type, out: *T) c_int {
        out.* = @ptrFromInt(next);
        next += 1;
        live += 1;
        return 0;
    }
    fn destroy(_: anytype) c_int {
        live -= 1;
        return 0;
    }
    fn handle(out: *lt.Handle) callconv(.c) c_int {
        return create(lt.Handle, out);
    }
    fn unhandle(v: lt.Handle) callconv(.c) c_int {
        return destroy(v);
    }
    fn version() callconv(.c) usize {
        return 133073;
    }
    fn name(_: c_int) callconv(.c) ?[*:0]const u8 {
        return "recorded failure";
    }
    fn desc(out: *lt.MatmulDesc, _: c_int, _: lt.DataType) callconv(.c) c_int {
        return create(lt.MatmulDesc, out);
    }
    fn undesc(v: lt.MatmulDesc) callconv(.c) c_int {
        return destroy(v);
    }
    fn descAttr(_: lt.MatmulDesc, _: c_int, _: ?*const anyopaque, _: usize) callconv(.c) c_int {
        return 0;
    }
    fn layout(out: *lt.Layout, _: lt.DataType, _: u64, _: u64, _: i64) callconv(.c) c_int {
        return create(lt.Layout, out);
    }
    fn unlayout(v: lt.Layout) callconv(.c) c_int {
        return destroy(v);
    }
    fn layoutAttr(_: lt.Layout, _: c_int, _: ?*const anyopaque, _: usize) callconv(.c) c_int {
        return 0;
    }
    fn pref(out: *lt.Preference) callconv(.c) c_int {
        return create(lt.Preference, out);
    }
    fn unpref(v: lt.Preference) callconv(.c) c_int {
        return destroy(v);
    }
    fn prefAttr(_: lt.Preference, _: c_int, _: ?*const anyopaque, _: usize) callconv(.c) c_int {
        return 0;
    }
    fn heuristic(_: lt.Handle, _: lt.MatmulDesc, _: lt.Layout, _: lt.Layout, _: lt.Layout, _: lt.Layout, _: lt.Preference, _: c_int, out: [*]lt.Heuristic, count: *c_int) callconv(.c) c_int {
        count.* = 1;
        out[0] = .{ .algo = .{ .data = @splat(0) }, .workspace_size = 512, .state = 0, .waves_count = 1, .reserved = @splat(0) };
        return 0;
    }
    fn matmul(_: lt.Handle, _: lt.MatmulDesc, _: ?*const anyopaque, _: u64, _: lt.Layout, _: u64, _: lt.Layout, _: ?*const anyopaque, _: u64, _: lt.Layout, _: u64, _: lt.Layout, _: ?*const lt.Algo, _: u64, _: usize, _: cuda.abi.Stream) callconv(.c) c_int {
        matmuls += 1;
        return 0;
    }
    const api: lt.Api = .{ .cublasLtCreate = handle, .cublasLtDestroy = unhandle, .cublasLtGetVersion = version, .cublasLtGetStatusName = name, .cublasLtMatmulDescCreate = desc, .cublasLtMatmulDescDestroy = undesc, .cublasLtMatmulDescSetAttribute = descAttr, .cublasLtMatrixLayoutCreate = layout, .cublasLtMatrixLayoutDestroy = unlayout, .cublasLtMatrixLayoutSetAttribute = layoutAttr, .cublasLtMatmulPreferenceCreate = pref, .cublasLtMatmulPreferenceDestroy = unpref, .cublasLtMatmulPreferenceSetAttribute = prefAttr, .cublasLtMatmulAlgoGetHeuristic = heuristic, .cublasLtMatmul = matmul };
};
var library: lt.Library = .{ .lib = undefined, .api = LT.api };
const kernel_image: [8]u8 align(8) = @splat(0);
fn reset() void {
    std.debug.assert(Driver.live == 0 and Driver.modules == 0 and LT.live == 0);
    CM.reset();
    SM.reset();
    LT.next = 1;
    LT.matmuls = 0;
    Driver.next = 0x400000000;
    Driver.context = @ptrFromInt(0x1230);
    Driver.allocs = 0;
    Driver.frees = 0;
    Driver.uploads = 0;
    Driver.launches = 0;
    Driver.lookups = 0;
    Driver.syncs = 0;
    Driver.fences = 0;
    Driver.copies = 0;
    Driver.fail_alloc = false;
    Driver.fail_free = false;
    Driver.fail_fence = false;
    Driver.fail_target_sync = false;
    Driver.fail_upload = 0;
    Driver.fail_launch = 0;
    Driver.fail_copy = 0;
    Driver.forced_ptr = null;
    Driver.unfenced_copy = false;
    Driver.bad_reuse = false;
}
fn weights(d: *const cuda.Driver, c: p.Config) !e.memory.DeviceWeights {
    const layout = try e.memory.WeightLayout.init(a, c, 2 << 30);
    // A borrowed native address is never read as model/host data in these tests.
    return .{ .layout = layout, .pool = .{ .d = d, .ptr = 0x100000000, .len = layout.bytes, .borrowed = true } };
}
fn engine(w: *const e.memory.DeviceWeights, g: e.memory.Geometry) !e.Engine {
    return e.Engine.init(a, w, .{ .d = w.pool.d, .handle = @ptrFromInt(1) }, g, .{ .convolution = &CM.api, .sdpa = &SM.api, .linear = &library, .image = &kernel_image, .package = .target_qualification }, .{ .workspace_limit = 1024, .max_activation_bytes = 1 << 26 });
}
fn geometry(request: *const p.Request, image: usize, groups: u32) e.memory.Geometry {
    return .{ .groups = groups, .gh = request.images[image].grid[1], .gw = request.images[image].grid[2] };
}
fn encodeOptions(d: *const cuda.Driver) p.EncodeOptions {
    return .{ .stream = .{ .d = d, .handle = @ptrFromInt(2) }, .replica = 77, .world = 2, .rank = 1 };
}
fn clean() !void {
    try t.expectEqual(@as(usize, 0), Driver.live);
    try t.expectEqual(@as(usize, 0), Driver.modules);
    try t.expectEqual(@as(usize, 0), LT.live);
    try t.expectEqual(@as(usize, 0), CM.live);
    try t.expectEqual(@as(usize, 0), SM.live);
}

test "vision pipeline budget follows per-picture source floor and shared cap" {
    const o: p.Options = .{ .prompt_limit = 262144 };
    try t.expectEqual(@as(u32, 2048), o.pictureBudget(1).max_tokens);
    try t.expectEqual(@as(u32, 327), o.pictureBudget(50).max_tokens);
    var low = o;
    low.request_image_tokens = 16;
    try t.expectEqual(@as(u32, 16), low.pictureBudget(50).max_tokens);
}
test "vision pipeline real CPU composition orientation resize and packing for every target fixture" {
    const c = try config();
    for (fixtures.all, 0..) |f, i| {
        var pixels: Pixels = .{};
        var request = try prepare(a, &pixels, c, &.{i}, cpu_options);
        defer request.deinit() catch unreachable;
        var expected = try pre.prepareRgb(a, c, .{ .shape = f.oriented, .bytes = f.rgb }, cpu_options.pictureBudget(1), .{}, resize.resizer);
        defer expected.deinit(a);
        try t.expectEqualDeep(expected.canvas, request.images[0].canvas);
        try t.expectEqualDeep(expected.content, request.images[0].content);
        try t.expectEqualDeep(expected.grid, request.images[0].grid);
        try t.expectEqualSlices(f32, expected.pixels, request.images[0].pixels);
        try t.expectEqual(expected.visualTokens(), request.rows);
        try t.expectEqual(request.rows, request.positions.len);
        for (request.positions, 0..) |pos, row| try t.expectEqual(@as(u32, @intCast(row + 1)), pos);
        try t.expectEqual(@as(usize, 1), pixels.probes);
        try t.expectEqual(@as(usize, 1), pixels.decodes);
    }
}
test "vision pipeline caller IDs are cloned and count refusal unwinds CPU allocations" {
    const c = try config();
    var pixels: Pixels = .{};
    const shape = try pre.canvas(fixtures.all[0].oriented, cpu_options.pictureBudget(1));
    const rows = @as(usize, shape.height / c.patch) * (shape.width / c.patch) / 4;
    const ids = try a.alloc(u32, rows);
    defer a.free(ids);
    @memset(ids, c.image_token);
    var request = try p.Request.prepareBackend(a, pixels.backend(), c, &.{fixtures.all[0].encoded}, ids, cpu_options);
    defer request.deinit() catch unreachable;
    @memset(ids, 0);
    try t.expectEqual(c.image_token, request.ids[0]);
    try t.expectError(error.ImageTokenCountMismatch, p.Request.prepareBackend(a, pixels.backend(), c, &.{fixtures.all[0].encoded}, &.{0}, cpu_options));
}
test "vision pipeline all CPU allocator failure positions unwind including second picture" {
    const c = try config();
    var succeeded = false;
    for (0..128) |fail| {
        var failing = t.FailingAllocator.init(a, .{ .fail_index = fail });
        var pixels: Pixels = .{};
        var request = prepare(failing.allocator(), &pixels, c, &.{ 0, 0 }, cpu_options) catch |err| {
            try t.expectEqual(error.OutOfMemory, err);
            try t.expectEqual(failing.allocated_bytes, failing.freed_bytes);
            continue;
        };
        try request.deinit();
        try t.expectEqual(failing.allocated_bytes, failing.freed_bytes);
        succeeded = true;
        break;
    }
    try t.expect(succeeded);
}
test "vision pipeline native decode second-image error releases every prepared owner" {
    var pixels: Pixels = .{ .fail_decode = 2 };
    try t.expectError(error.InvalidEncodedImage, prepare(a, &pixels, try config(), &.{ 0, 0 }, cpu_options));
    try t.expectEqual(@as(usize, 2), pixels.decodes);
}
test "vision pipeline byte pixel prepared prompt token and count admission" {
    const c = try config();
    var pixels: Pixels = .{};
    var o = cpu_options;
    o.max_total_encoded_bytes = fixtures.all[0].encoded.len - 1;
    try t.expectError(error.ImageEncodedByteLimit, prepare(a, &pixels, c, &.{0}, o));
    try t.expectEqual(@as(usize, 0), pixels.probes);
    o = cpu_options;
    o.max_total_pixels = 6;
    try t.expectError(error.ImageDecodedPixelLimit, prepare(a, &pixels, c, &.{ 0, 0 }, o));
    try t.expectEqual(@as(usize, 1), pixels.decodes);
    o = cpu_options;
    o.max_prepared_bytes = 1;
    try t.expectError(error.VisionPreparedBudgetExceeded, prepare(a, &pixels, c, &.{0}, o));
    o = cpu_options;
    o.max_images = 1;
    try t.expectError(error.InvalidVisionImageCount, prepare(a, &pixels, c, &.{ 0, 0 }, o));
    try t.expectError(error.BadToken, p.Request.prepareBackend(a, pixels.backend(), c, &.{fixtures.all[0].encoded}, &.{154880}, cpu_options));
    o = cpu_options;
    o.prompt_limit = 1;
    try t.expectError(error.ImagePromptTooLong, p.Request.prepareBackend(a, pixels.backend(), c, &.{fixtures.all[0].encoded}, &.{0}, o));
    try t.expectError(error.InvalidVisionImageCount, p.Request.prepareBackend(a, pixels.backend(), c, &.{}, &.{0}, cpu_options));
    o = cpu_options;
    o.image_tokens = 0;
    try t.expectError(error.InvalidVisionPipelineBudget, prepare(a, &pixels, c, &.{0}, o));
}
test "vision pipeline complete grouping avoids greedy dead ends using concrete encoder owners" {
    reset();
    var d = Driver.value();
    var pixels: Pixels = .{};
    var request = try prepare(a, &pixels, try config(), &.{ 0, 0, 0, 0, 0, 0 }, cpu_options);
    var w = try weights(&d, request.c);
    var four = try engine(&w, geometry(&request, 0, 4));
    var three = try engine(&w, geometry(&request, 0, 3));
    var plan = try request.plan(&d, &.{ &four, &three });
    try t.expectEqual(@as(usize, 2), plan.count);
    for (plan.passes(), 0..) |pass, i| {
        try t.expectEqual(&three, pass.engine);
        try t.expectEqual(@as(usize, 3), pass.images);
        try t.expectEqual(i * 3, pass.first_image);
        try t.expectEqual(i * 3 * request.images[0].visualTokens(), pass.first_row);
    }
    plan.deinit();
    try request.encode(&.{ &four, &three }, encodeOptions(&d));
    try t.expectEqual(@as(usize, 2), Driver.copies);
    try t.expect(!Driver.bad_reuse and !Driver.unfenced_copy);
    try t.expectEqual(@as(usize, 248), LT.matmuls);
    try request.deinit();
    try three.deinit();
    try four.deinit();
    w.deinit();
    try clean();
}
test "vision pipeline rectangular heterogeneous order encodes all passes then binds actual language copies" {
    reset();
    var d = Driver.value();
    var pixels: Pixels = .{};
    const rotated = indexOf("oriented_png_6");
    var request = try prepare(a, &pixels, try config(), &.{ 0, rotated, 0 }, cpu_options);
    try t.expect(request.images[0].grid[1] != request.images[1].grid[1]);
    var w = try weights(&d, request.c);
    var first = try engine(&w, geometry(&request, 0, 1));
    var second = try engine(&w, geometry(&request, 1, 1));
    const opts = encodeOptions(&d);
    try t.expectError(error.VisionPipelineNotReady, request.borrowedBinding());
    try request.encode(&.{ &first, &second }, opts);
    const b = try request.borrowedBinding();
    try t.expectEqual(@as(usize, 3), Driver.copies);
    var at: usize = 0;
    for (request.images, 0..) |image, i| {
        const owner = if (i == 1) &second else &first;
        const out = try owner.activations.buffer(.output);
        const record = Driver.records[i];
        try t.expectEqual(request.table.?.features.ptr + at, record.dst);
        try t.expectEqual(out.ptr, record.src);
        try t.expectEqual(image.visualTokens() * 4096 * 2, record.size);
        try t.expectEqual(opts.stream.handle, record.stream);
        at += record.size;
    }
    try t.expectEqual(@as(usize, 372), LT.matmuls);
    try t.expect(Driver.launches > 300 and CM.forwarded and SM.executed);
    try t.expect(!Driver.bad_reuse and !Driver.unfenced_copy);
    try t.expectEqual(@as(u64, 77), b.owner.replica);
    try t.expectEqual(@as(u32, 2), b.owner.world);
    try t.expectEqual(@as(u32, 1), b.owner.rank);
    const ids = try a.alloc(i32, request.ids.len);
    defer a.free(ids);
    for (request.ids, ids) |id, *out| out.* = @intCast(id);
    const window = 8;
    var target: binding.Target = .{ .stream = opts.stream, .context = Driver.context, .embedding = .{ .d = &d, .ptr = 0x7000000000, .len = window * 4 * 4096 * 2, .borrowed = true }, .start = 0, .capacity = 4096, .window = window, .vocab = 154880, .world = 2, .rank = 1 };
    var start: usize = 0;
    while (start < ids.len) : (start += window) {
        target.start = start;
        try binding.inject(b, target, ids[start..@min(start + window, ids.len)]);
    }
    try t.expectEqual(@as(usize, 3) + request.rows * 4, Driver.copies);
    for (request.positions, 0..) |pos, row| for (0..4) |stream| {
        const copy = Driver.records[3 + row * 4 + stream];
        try t.expectEqual(b.owner.features.ptr + row * 8192, copy.src);
        try t.expectEqual(target.embedding.ptr + (@as(usize, pos) % window * 4 + stream) * 8192, copy.dst);
        try t.expectEqual(@as(usize, 8192), copy.size);
    };
    try opts.stream.synchronize();
    try t.expectError(error.VisionPipelineAlreadyEncoded, request.encode(&.{&first}, opts));
    try request.deinit();
    try second.deinit();
    try first.deinit();
    w.deinit();
    try clean();
}
test "vision pipeline missing heterogeneous owner refuses before any CUDA allocation or write" {
    reset();
    var d = Driver.value();
    var pixels: Pixels = .{};
    var request = try prepare(a, &pixels, try config(), &.{ 0, indexOf("oriented_png_6") }, cpu_options);
    var w = try weights(&d, request.c);
    var en = try engine(&w, geometry(&request, 0, 1));
    const before = .{ Driver.allocs, Driver.uploads, Driver.launches, Driver.copies };
    try t.expectError(error.MissingMatchingVisionEncoder, request.encode(&.{&en}, encodeOptions(&d)));
    try t.expectEqualDeep(before, .{ Driver.allocs, Driver.uploads, Driver.launches, Driver.copies });
    try t.expect(request.table == null);
    try request.deinit();
    try en.deinit();
    w.deinit();
    try clean();
}
test "vision pipeline prompt and encoder preflight refuses before any CUDA writes" {
    reset();
    var d = Driver.value();
    var other = Driver.value();
    var pixels: Pixels = .{};
    var request = try prepare(a, &pixels, try config(), &.{0}, cpu_options);
    var w = try weights(&d, request.c);
    var en = try engine(&w, geometry(&request, 0, 1));
    const opts = encodeOptions(&d);
    const before = .{ Driver.allocs, Driver.uploads, Driver.launches, Driver.copies };
    try t.expectError(error.MissingVisionEncoder, request.encode(&.{}, opts));
    en.poisoned = true;
    try t.expectError(error.VisionEncoderPoisoned, request.encode(&.{&en}, opts));
    en.poisoned = false;
    en.stream.d = &other;
    try t.expectError(error.VisionPipelineDriverMismatch, request.encode(&.{&en}, opts));
    en.stream.d = &d;
    const token = request.positions[0];
    request.positions[0] += 1;
    try t.expectError(error.ImageTokenCountMismatch, request.encode(&.{&en}, opts));
    request.positions[0] = token;
    const x = request.images[0].pixels[0];
    request.images[0].pixels[0] = std.math.nan(f32);
    try t.expectError(error.NonFiniteVisionPixels, request.encode(&.{&en}, opts));
    request.images[0].pixels[0] = x;
    request.ids[0] = 154880;
    try t.expectError(error.BadToken, request.encode(&.{&en}, opts));
    request.ids[0] = 5;
    var bad = opts;
    bad.replica = 0;
    try t.expectError(error.InvalidVisionPipelineReplica, request.encode(&.{&en}, bad));
    Driver.context = null;
    try t.expectError(error.InvalidVisionPipelineContext, request.encode(&.{&en}, opts));
    Driver.context = @ptrFromInt(0x1230);
    bad = opts;
    bad.expected_context = @ptrFromInt(0x9990);
    try t.expectError(error.InvalidVisionPipelineContext, request.encode(&.{&en}, bad));
    try t.expectEqualDeep(before, .{ Driver.allocs, Driver.uploads, Driver.launches, Driver.copies });
    try request.deinit();
    try en.deinit();
    w.deinit();
    try clean();
}
test "vision pipeline checked native allocation refusal can retry without partial owner" {
    reset();
    var d = Driver.value();
    var pixels: Pixels = .{};
    var request = try prepare(a, &pixels, try config(), &.{0}, cpu_options);
    var w = try weights(&d, request.c);
    var en = try engine(&w, geometry(&request, 0, 1));
    Driver.fail_alloc = true;
    try t.expectError(error.NotReady, request.encode(&.{&en}, encodeOptions(&d)));
    try t.expect(request.table == null and !request.pending);
    Driver.fail_alloc = false;
    try request.encode(&.{&en}, encodeOptions(&d));
    try request.deinit();
    try en.deinit();
    w.deinit();
    try clean();
}
test "vision pipeline alias and alignment admission publishes no native inference" {
    for (0..2) |case| {
        reset();
        var d = Driver.value();
        var pixels: Pixels = .{};
        var request = try prepare(a, &pixels, try config(), &.{0}, cpu_options);
        var w = try weights(&d, request.c);
        var en = try engine(&w, geometry(&request, 0, 1));
        Driver.forced_ptr = if (case == 0) en.activations.pool.ptr else 0x600000001;
        try t.expectError(if (case == 0) error.AliasedVisionPipelineOutput else error.InvalidVisionPipelineAllocation, request.encode(&.{&en}, encodeOptions(&d)));
        try t.expectEqual(@as(usize, 0), Driver.launches);
        try t.expectEqual(@as(usize, 0), Driver.copies);
        try t.expectError(error.VisionPipelineNotReady, request.borrowedBinding());
        try request.deinit();
        Driver.forced_ptr = null;
        try en.deinit();
        w.deinit();
        try clean();
    }
}
test "vision pipeline failed native staging launch DtoD or fence retains all owners until cleanup" {
    for (0..4) |case| {
        reset();
        var d = Driver.value();
        var pixels: Pixels = .{};
        var request = try prepare(a, &pixels, try config(), &.{0}, cpu_options);
        var w = try weights(&d, request.c);
        var en = try engine(&w, geometry(&request, 0, 1));
        switch (case) {
            0 => Driver.fail_upload = Driver.uploads + 1,
            1 => Driver.fail_launch = 1,
            2 => Driver.fail_copy = 1,
            3 => Driver.fail_target_sync = true,
            else => unreachable,
        }
        try t.expectError(error.NotReady, request.encode(&.{&en}, encodeOptions(&d)));
        try t.expect(request.pending and request.pending_encoder == &en);
        try t.expect(!request.table.?.ready);
        const cpu = request.images[0].pixels.ptr;
        const gpu = request.table.?.features.ptr;
        const live = Driver.live;
        const frees = Driver.frees;
        Driver.fail_fence = true;
        try t.expectError(error.NotReady, request.deinit());
        try t.expectEqual(cpu, request.images[0].pixels.ptr);
        try t.expectEqual(gpu, request.table.?.features.ptr);
        try t.expectEqual(live, Driver.live);
        try t.expectEqual(frees, Driver.frees);
        try t.expect(request.pending and request.pending_encoder == &en);
        Driver.fail_fence = false;
        try request.recover();
        try t.expect(!request.pending and request.pending_encoder == null);
        try t.expectError(error.VisionPipelineNotReady, request.borrowedBinding());
        Driver.fail_free = true;
        try t.expectError(error.NotReady, request.deinit());
        try t.expectEqual(cpu, request.images[0].pixels.ptr);
        try t.expectEqual(gpu, request.table.?.features.ptr);
        try t.expectEqual(live, Driver.live);
        Driver.fail_free = false;
        try request.deinit();
        Driver.fail_target_sync = false;
        try en.deinit();
        w.deinit();
        try clean();
    }
}
test "vision pipeline later-pass copy refusal never publishes an earlier successful subset" {
    reset();
    var d = Driver.value();
    var pixels: Pixels = .{};
    var request = try prepare(a, &pixels, try config(), &.{ 0, 0 }, cpu_options);
    var w = try weights(&d, request.c);
    var en = try engine(&w, geometry(&request, 0, 1));
    Driver.fail_copy = 2;
    try t.expectError(error.NotReady, request.encode(&.{&en}, encodeOptions(&d)));
    try t.expectEqual(@as(usize, 2), Driver.copies);
    try t.expectEqual(@as(usize, 248), LT.matmuls);
    try t.expect(!Driver.bad_reuse);
    try t.expect(request.pending and request.pending_encoder == &en);
    try t.expect(!request.table.?.ready);
    try t.expectError(error.VisionPipelineNotReady, request.borrowedBinding());
    try t.expectError(error.VisionPipelineAlreadyEncoded, request.encode(&.{&en}, encodeOptions(&d)));
    try request.deinit();
    try en.deinit();
    w.deinit();
    try clean();
}
test "vision pipeline cleanup wrong current context retains ready owner and invalidates binding" {
    reset();
    var d = Driver.value();
    var pixels: Pixels = .{};
    var request = try prepare(a, &pixels, try config(), &.{0}, cpu_options);
    var w = try weights(&d, request.c);
    var en = try engine(&w, geometry(&request, 0, 1));
    try request.encode(&.{&en}, encodeOptions(&d));
    const b = try request.borrowedBinding();
    const gpu = request.table.?.features.ptr;
    Driver.context = @ptrFromInt(0x9990);
    try t.expectError(error.InvalidVisionPipelineContext, request.deinit());
    try t.expectEqual(gpu, request.table.?.features.ptr);
    try t.expect(!b.owner.ready);
    try t.expectError(error.VisionNotReady, b.owner.binding());
    Driver.context = @ptrFromInt(0x1230);
    try request.deinit();
    try en.deinit();
    w.deinit();
    try clean();
}
