//! Existing GLM family recording controls on the CPU runtime.
const std = @import("std");

pub fn steps(b: *std.Build, cuda: *std.Build.Module, glm: *std.Build.Module, mods: anytype, step: *std.Build.Step) void {
    const host = b.graph.host;
    const loader = b.createModule(.{ .root_source_file = b.path("zig/src/families/glm/cuda_weights.zig"), .target = host, .optimize = .debug, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = mods.core } } });
    loader.addImport("canon", glm.import_table.get("canon").?);
    const loader_step = b.step("test-glm-loader-host", "Mixed-format loader, projection adapters and precision refusals without GPU work");
    loader_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = loader })).step);
    const refusal = b.createModule(.{ .root_source_file = b.path("zig/src/families/glm/cuda_loader_refusal_test.zig"), .target = host, .optimize = .debug, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = mods.core }, .{ .name = "canon", .module = glm.import_table.get("canon").? } } });
    const refusal_test = b.addTest(.{ .root_module = refusal, .filters = &.{"GLM loader refusal:"} });
    // Run as an ordinary subprocess, not Zig's test-server protocol: the intentional production error log must exit 1.
    const refusal_run = std.Build.Step.Run.create(b, "check GLM unused-tensor refusal diagnostic");
    refusal_run.addArtifactArg(refusal_test);
    refusal_run.expectExitCode(1);
    refusal_run.expectStdErrMatch("unused checkpoint tensor model.language_model.unknown");
    refusal_run.expectStdErrMatch("All 2 tests passed."); // the named refusal and weights' unconditional import-coverage test
    refusal_run.expectStdErrMatch("\n1 errors were logged.\n");
    refusal_run.expectStdOutEqual("");
    b.step("test-glm-loader-refusal-host", "Assert unused-tensor rejection, teardown and the expected diagnostic exit").dependOn(&refusal_run.step);
    loader_step.dependOn(&refusal_run.step);
    const source_gather = b.createModule(.{ .root_source_file = b.path("zig/src/families/glm/cuda_weight_source_test.zig"), .target = host, .optimize = .debug, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = mods.core }, .{ .name = "canon", .module = glm.import_table.get("canon").? } } });
    loader_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = source_gather, .filters = &.{"explicit source conversion"} })).step);
    inline for (.{ "state", "forward", "ranks", "spill_payload" }) |unit| {
        const root = b.createModule(.{ .root_source_file = b.path("zig/src/families/glm/cuda_" ++ unit ++ ".zig"), .target = host, .optimize = .debug, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = mods.core }, .{ .name = "canon", .module = glm.import_table.get("canon").? } } });
        root.addImport("heat", mods.heat);
        root.addImport("lanes", mods.lanes);
        inline for (.{ "cudnn", "vision_sdpa", "vision_ops", "image_codec_headers" }) |name|
            root.addImport(name, glm.import_table.get(name).?);
        b.step("test-glm-" ++ unit ++ "-host", "Actual GLM " ++ unit ++ " behavior with recording CUDA, without GPU work").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = root })).step);
    }
    const api = b.createModule(.{ .root_source_file = b.path("zig/src/core/engine_api.zig"), .target = host, .optimize = .debug, .link_libc = true, .imports = &.{.{ .name = "lanes", .module = mods.lanes }} });
    const admission = b.createModule(.{ .root_source_file = b.path("zig/glm_admission_host.zig"), .target = host, .optimize = .debug, .link_libc = true, .imports = &.{ .{ .name = "glm", .module = glm }, .{ .name = "lanes", .module = mods.lanes }, .{ .name = "engine_api", .module = api } } });
    const admission_run = b.addRunArtifact(b.addTest(.{ .root_module = admission, .filters = &.{"GLM CUDA admission:"} }));
    b.step("test-glm-admission-host", "Real GLM admission capabilities through the lane worker without GPU work").dependOn(&admission_run.step);
    step.dependOn(&admission_run.step);
    b.step("test-glm-prompt-settings-host", "Actual opt-in settings and rank agreement on the CPU").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = glm, .filters = &.{"prompt K experiment"} })).step);
    const mm_record = b.createModule(.{ .root_source_file = b.path("zig/test_support/glm_mm_cuda.zig"), .target = host, .optimize = .debug });
    const mm_actual = b.createModule(.{ .root_source_file = b.path("zig/src/families/glm/cuda_mm.zig"), .target = host, .optimize = .debug, .imports = &.{.{ .name = "cuda", .module = mm_record }} });
    const mm_policy = b.createModule(.{ .root_source_file = b.path("zig/glm_mm_policy_host.zig"), .target = host, .optimize = .debug, .imports = &.{ .{ .name = "cuda", .module = mm_record }, .{ .name = "mm", .module = mm_actual } } });
    b.step("test-glm-mm-policy-host", "Actual prompt K policy dispatch without GPU work").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = mm_policy })).step);
    const drafter_host = b.createModule(.{ .root_source_file = b.path("zig/glm_dflash_host.zig"), .target = host, .optimize = .debug, .link_libc = true, .imports = &.{ .{ .name = "glm", .module = glm }, .{ .name = "core", .module = mods.core } } });
    b.step("test-glm-dflash-host", "DFlash rank buffers, partial loader ownership and pending records on the CPU").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = drafter_host, .filters = &.{"glm dflash:"} })).step);
    const packed_host = b.createModule(.{ .root_source_file = b.path("zig/glm_packed_host.zig"), .target = host, .optimize = .debug, .link_libc = true, .imports = &.{ .{ .name = "glm", .module = glm }, .{ .name = "core", .module = mods.core } } });
    b.step("test-glm-packed-host", "Packed GLM launch records, cap parsing and native loader on the CPU").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = packed_host, .filters = &.{"glm packed:"} })).step);
    const ring_host = b.createModule(.{ .root_source_file = b.path("zig/glm_ring_host.zig"), .target = host, .optimize = .debug, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "glm", .module = glm } } });
    b.step("test-glm-ring-host", "Actual GLM MTP rollback pool through recording CUDA copies").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = ring_host, .filters = &.{"glm ring:"} })).step);
    const family_run = b.addRunArtifact(b.addTest(.{ .root_module = glm }));
    step.dependOn(&family_run.step);
    b.step("test-glm-family-host", "Complete normal GLM family root with recording CUDA, without GPU work").dependOn(&family_run.step);
    // Named subprocess evidence supplements (never replaces/filters) the complete normal family gate.
    const retained_cases = b.addTest(.{ .root_module = glm, .filters = &.{ "format sniffing is bounded and covers every independent target fixture", "vision binding rejects stale readiness, ownership, dtype, geometry and every malformed placeholder map" } });
    const retained_run = std.Build.Step.Run.create(b, "verify retained codec and stale vision binding cases execute");
    retained_run.addArtifactArg(retained_cases);
    retained_run.expectExitCode(0);
    retained_run.expectStdErrMatch("format sniffing is bounded and covers every independent target fixture...OK");
    retained_run.expectStdErrMatch("vision binding rejects stale readiness, ownership, dtype, geometry and every malformed placeholder map...OK");
    b.step("test-glm-vision-retained-host", "Prove registered codec and stale borrowed-binding tests execute by name").dependOn(&retained_run.step);
    const prompt_host = b.createModule(.{ .root_source_file = b.path("zig/exl3_prompt_host.zig"), .target = host, .optimize = .debug, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "glm", .module = glm }, .{ .name = "lanes", .module = mods.lanes } } });
    const prompt_run = b.addRunArtifact(b.addTest(.{ .root_module = prompt_host, .filters = &.{"exl3 prompt:"} }));
    prompt_run.has_side_effects = true;
    b.step("test-exl3-prompt-host", "Actual common prompt dispatch and scratch with recording CUDA calls").dependOn(&prompt_run.step);
}

pub fn compile(b: *std.Build, target: std.Build.ResolvedTarget, cuda: *std.Build.Module, mods: anytype, comptime makeGlm: anytype) void {
    const glm_open = b.addObject(.{ .name = "glm-open-check", .root_module = b.createModule(.{ .root_source_file = b.path("zig/glm_open_compile.zig"), .target = target, .optimize = .debug, .link_libc = true, .imports = &.{.{ .name = "glm", .module = makeGlm(b, target, .debug, cuda, mods.core, mods.lanes, mods.heat) }} }) });
    _ = glm_open.getEmittedBin();
    b.step("compile-glm-native-open", "Compile the actual GLM Linux open body with empty GPU images").dependOn(&glm_open.step);
    const prompt_gate = b.createModule(.{ .root_source_file = b.path("zig/tests/cuda/glm_mm_fixed_tests.zig"), .target = target, .optimize = .debug, .link_libc = true, .imports = &.{ .{ .name = "glm_cuda", .module = makeGlm(b, target, .debug, cuda, mods.core, mods.lanes, mods.heat) }, .{ .name = "cuda", .module = cuda } } });
    const prompt_gate_compile = b.addObject(.{ .name = "glm-prompt-gate-compile", .root_module = b.createModule(.{ .root_source_file = b.path("zig/glm_prompt_gate_compile.zig"), .target = target, .optimize = .debug, .link_libc = true, .imports = &.{.{ .name = "gate", .module = prompt_gate }} }) });
    _ = prompt_gate_compile.getEmittedBin();
    b.step("compile-glm-prompt-gate-host", "Compile the actual device prompt comparison gate with inert images").dependOn(&prompt_gate_compile.step);
}
