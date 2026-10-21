//! tfclef: native Clef Flash decisions (zig/src/families/clef) as a C library, and its host checks.
const std = @import("std");

/// The family module; `image` is clef_head.cu's fatbin, or null for host checks (an empty image); `source` is the
/// TensorFold commit the library reports.
pub fn module(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, core: *std.Build.Module, qwen35: *std.Build.Module, image: ?std.Build.LazyPath, source: []const u8) *std.Build.Module {
    const m = b.createModule(.{
        .root_source_file = b.path("zig/src/families/clef/root.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = core }, .{ .name = "qwen35", .module = qwen35 } },
    });
    m.addAnonymousImport("fatbin_clef_head", .{ .root_source_file = image orelse b.addWriteFiles().add("clef_head-host-only.fatbin", "") });
    const options = b.addOptions();
    options.addOption([]const u8, "source_commit", source);
    m.addOptions("clef_build", options);
    return m;
}

/// `test-clef-host`: the family's host tests and the C header against the export layer, no device.
pub fn hostTests(b: *std.Build, cuda: *std.Build.Module, core: *std.Build.Module, qwen35: *std.Build.Module, all: *std.Build.Step) void {
    const host = b.graph.host;
    const step = b.step("test-clef-host", "Clef head checkpoint, record, budget and C ABI host tests without a GPU");
    const family = b.addRunArtifact(b.addTest(.{ .root_module = module(b, host, .debug, cuda, core, qwen35, null, "host-test") }));
    family.has_side_effects = true;
    step.dependOn(&family.step);
    const c_abi = b.createModule(.{ .root_source_file = b.path("zig/src/families/clef/abi_c_test.zig"), .target = host, .optimize = .debug, .link_libc = true });
    c_abi.addIncludePath(b.path("include"));
    c_abi.addCSourceFile(.{ .file = b.path("zig/tests/clef/tfclef_abi_test.c"), .flags = &.{ "-std=c11", "-Wall", "-Wextra", "-Werror" } });
    const header = b.addRunArtifact(b.addTest(.{ .root_module = c_abi }));
    header.has_side_effects = true;
    step.dependOn(&header.step);
    all.dependOn(step);
}
