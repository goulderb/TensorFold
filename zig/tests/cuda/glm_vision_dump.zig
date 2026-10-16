//! Export the actual native tower's BF16 projector rows for an independent oracle.
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm_cuda");
const check = @import("check.zig");

pub fn run(gpu: check.Gpu, checkpoint: []const u8, image_path: []const u8, output_path: []const u8) !void {
    const shim = std.c.getenv("TF_GLM_VISION_CODEC") orelse return error.MissingVisionCodecPath;
    const image = try std.Io.Dir.cwd().readFileAlloc(gpu.io, image_path, gpu.gpa, .limited(glm.vision_codec.max_encoded_bytes));
    defer gpu.gpa.free(image);
    var probe = try glm.vision_probe.Probe.open(gpu.gpa, gpu.io, checkpoint, std.mem.span(shim));
    defer probe.close();
    const rows = try glm.vision_probe.Probe.rows(probe, image, 1);
    const ids = try gpu.gpa.alloc(u32, @as(usize, rows) + 2);
    defer gpu.gpa.free(ids);
    ids[0] = probe.config.image_start;
    @memset(ids[1 .. ids.len - 1], probe.config.image_token);
    ids[ids.len - 1] = probe.config.image_end;

    var stream = try cuda.Stream.init(gpu.d, false);
    defer stream.deinit();
    const service = try glm.vision_runtime.Service.open(gpu.gpa, gpu.io, gpu.d, gpu.ctx.handle, stream, checkpoint, probe);
    defer service.close();
    var encoded = try service.encode(&.{image}, ids, 1, 1);
    defer encoded.close() catch {};
    const table = &(encoded.request.table orelse return error.VisionPipelineNotReady);
    if (!table.ready or table.rows != rows or table.width != 4096 or table.dtype != .bf16) return error.VisionPipelineNotReady;
    const bytes = try gpu.gpa.alloc(u8, table.features.len);
    defer gpu.gpa.free(bytes);
    try stream.synchronize();
    try table.features.download(0, bytes);
    try std.Io.Dir.cwd().writeFile(gpu.io, .{ .sub_path = output_path, .data = bytes });
    std.debug.print("RESULT GLM vision rows={d} width=4096 dtype=bf16 bytes={d} output={s}\n", .{ rows, bytes.len, output_path });
}
