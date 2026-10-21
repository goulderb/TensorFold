//! tfclef: native Clef Flash decisions (zig/src/families/clef) as a C library, and its host checks.
const std = @import("std");

pub fn module(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, core: *std.Build.Module, qwen35: *std.Build.Module) *std.Build.Module {
    return b.createModule(.{
        .root_source_file = b.path("zig/src/families/clef/root.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = core }, .{ .name = "qwen35", .module = qwen35 } },
    });
}

/// `test-clef-host`: the family's host tests and the C header against the export layer, no device.
pub fn hostTests(b: *std.Build, cuda: *std.Build.Module, core: *std.Build.Module, qwen35: *std.Build.Module, all: *std.Build.Step) void {
    const host = b.graph.host;
    const step = b.step("test-clef-host", "Clef head checkpoint, record, budget and C ABI host tests without a GPU");
    const family = b.addRunArtifact(b.addTest(.{ .root_module = module(b, host, .debug, cuda, core, qwen35) }));
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
