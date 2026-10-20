const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    if (optimize != .fast) @panic("the CUDA seam requires ReleaseFast");
    const source = b.option([]const u8, "source_commit", "Full source commit") orelse @panic("source_commit is required");
    if (source.len != 40) @panic("source_commit must be 40 lowercase hex characters");
    for (source) |c| if (!(c >= '0' and c <= '9') and !(c >= 'a' and c <= 'f')) @panic("invalid source_commit");
    const fatbins = b.option([]const u8, "fatbins", "Absolute directory of frozen CUDA images") orelse @panic("fatbins is required");
    if (!std.fs.path.isAbsolute(fatbins)) @panic("fatbins must be absolute");
    const cuda = b.createModule(.{ .root_source_file = b.path("../../../zig/src/cuda/root.zig"), .target = target, .optimize = optimize, .link_libc = true });
    cuda.addImport("sys", b.createModule(.{ .root_source_file = b.path("../../../zig/src/platform/sys.zig"), .target = target, .optimize = optimize, .link_libc = true }));
    const kernels = b.addOptions();
    kernels.addOption(bool, "with_kernels", true);
    cuda.addOptions("kernel_options", kernels);
    cuda.addImport("stagger", b.createModule(.{ .root_source_file = b.path("../../../zig/src/core/stagger.zig"), .target = target, .optimize = optimize }));
    for ([_][]const u8{ "ck_lane", "ck_experts", "norms", "qwen35_ops", "qwen35_hf", "modelopt_pack" }) |name| {
        cuda.addAnonymousImport(b.fmt("fatbin_{s}", .{name}), .{ .root_source_file = b.graph.cwdRelativePath(b.pathJoin(&.{ fatbins, b.fmt("{s}.fatbin", .{name}) })) });
    }
    const root = b.createModule(.{ .root_source_file = b.path("main.zig"), .target = target, .optimize = .safe, .link_libc = true });
    root.addImport("cuda", cuda);
    const admission = b.createModule(.{ .root_source_file = b.path("../../../zig/tests/cuda/qwen35_probe_device.zig"), .target = target, .optimize = .safe });
    admission.addImport("cuda", cuda);
    root.addImport("admission", admission);
    const options = b.addOptions();
    options.addOption([]const u8, "source_commit", source);
    options.addOption(std.builtin.OptimizeMode, "engine_optimization", optimize);
    root.addOptions("options", options);
    b.installArtifact(b.addExecutable(.{ .name = "tf-qwen35-decode-profile", .root_module = root }));
}
