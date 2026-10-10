//! Windows server tests: the vision refusal runs in `zig build test`; the console control has a step of its own.
const std = @import("std");

/// `server` is tensorfold-native for this target. The refusal test runs it; a console test needs its own console.
pub fn steps(b: *std.Build, target: std.Build.ResolvedTarget, test_step: *std.Build.Step, server: *std.Build.Step.Compile) void {
    const vision = b.addExecutable(.{ .name = "tf-vision-refusal-test", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/server/windows/vision_refusal_test.zig"),
        .target = target,
        .optimize = .Debug,
    }) });
    const vision_run = b.addRunArtifact(vision);
    vision_run.addArtifactArg(server);
    test_step.dependOn(&vision_run.step);
    const win32_mod = b.createModule(.{ .root_source_file = b.path("zig/src/server/win32.zig"), .target = target, .optimize = .Debug });
    const console = b.addExecutable(.{ .name = "tf-console-test", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/server/windows/console_test.zig"),
        .target = target,
        .optimize = .Debug,
        .imports = &.{.{ .name = "win32", .module = win32_mod }},
    }) });
    b.step("test-server-console", "The server's console control test (Windows): run it from a console").dependOn(&b.addRunArtifact(console).step);
}
