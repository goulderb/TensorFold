//! The CUDA half of the root build: kernel fatbins with their Python extensions' nvcc flags, the runtime, the CLI.
const std = @import("std");
const exlr8_build = @import("exlr8.zig");
const golden_build = @import("golden.zig");
const catalog = @import("cuda_kernels.zig");
const Kernel = catalog.Kernel;
const torch_flags = catalog.torch_flags;
const kernels = catalog.kernels;
fn stagger(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) *std.Build.Module {
    return b.createModule(.{ .root_source_file = b.path("zig/src/core/stagger.zig"), .target = target, .optimize = optimize });
}

/// The runtime module for `target`; `with_kernels` false builds it host-only (empty images).
pub fn runtime(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, images: []const ?std.Build.LazyPath) *std.Build.Module {
    const options = b.addOptions();
    var with = images.len > 0;
    for (images) |i| with = with and i != null;
    options.addOption(bool, "with_kernels", with);
    const cuda = b.createModule(.{ .root_source_file = b.path("zig/src/cuda/root.zig"), .target = target, .optimize = optimize, .link_libc = true });
    cuda.addOptions("kernel_options", options);
    cuda.addImport("stagger", stagger(b, target, optimize));
    if (with) for (kernels, images) |k, image| cuda.addAnonymousImport(b.fmt("fatbin_{s}", .{k.name}), .{ .root_source_file = image.? });
    return cuda;
}

fn family(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, draft_ids: *std.Build.Module) struct { core: *std.Build.Module, lanes: *std.Build.Module, nemotron: *std.Build.Module, qwen35: *std.Build.Module, qwen_vision: *std.Build.Module, tokenizer: *std.Build.Module, heat: *std.Build.Module, radio: *std.Build.Module } {
    const tokenizer = b.createModule(.{ .root_source_file = b.path("zig/src/core/tokenizer/tokenizer.zig"), .target = target, .optimize = optimize, .link_libc = true });
    const lanes = b.createModule(.{ .root_source_file = b.path("zig/src/core/lanes/lanes.zig"), .target = target, .optimize = optimize, .link_libc = true });
    const core = b.createModule(.{ .root_source_file = b.path("zig/src/core/root.zig"), .target = target, .optimize = optimize, .link_libc = true });
    core.addImport("tokenizer", tokenizer);
    core.addImport("lanes", lanes);
    const heat = b.createModule(.{ .root_source_file = b.path("zig/src/core/heat.zig"), .target = target, .optimize = optimize, .link_libc = true });
    const nemotron = b.createModule(.{ .root_source_file = b.path("zig/src/families/nemotron/cuda.zig"), .target = target, .optimize = optimize, .link_libc = true });
    nemotron.addImport("cuda", cuda);
    nemotron.addImport("core", core);
    nemotron.addImport("lanes", lanes);
    nemotron.addImport("heat", heat);
    nemotron.addImport("nemotron_draft_ids", draft_ids);
    const radio = radioModule(b, target, optimize, cuda, core, lanes);
    nemotron.addImport("radio", radio);
    const qwen35 = b.createModule(.{ .root_source_file = b.path("zig/src/families/qwen3_5/cuda.zig"), .target = target, .optimize = optimize, .link_libc = true });
    qwen35.addImport("cuda", cuda);
    qwen35.addImport("core", core);
    qwen35.addImport("lanes", lanes);
    const qwen_vision = qwenVisionModule(b, target, optimize, cuda, core, lanes);
    qwen35.addImport("qwen_vision", qwen_vision);
    return .{ .core = core, .lanes = lanes, .nemotron = nemotron, .qwen35 = qwen35, .qwen_vision = qwen_vision, .tokenizer = tokenizer, .heat = heat, .radio = radio };
}

/// Linux targets: fatbins (-Dnvcc builds them, -Dfatbins embeds prebuilt ones), `tensorfold` and `tf-cuda-test`.
pub fn targets(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, draft_ids: *std.Build.Module, build_options: *std.Build.Step.Options) *std.Build.Step.Compile {
    const nvcc = b.option([]const u8, "nvcc", "nvcc (or a wrapper) that builds the CUDA kernel fatbins");
    const prebuilt = b.option([]const u8, "fatbins", "absolute directory of prebuilt <name>.fatbin files to embed");
    const sms = b.option([]const u8, "sm", "SASS targets, comma separated (default 121; 80, 86, 89, 120 and 121 build)") orelse "121";
    const strip = b.option(bool, "strip", "No debug info in the CUDA executables: no build machine paths leave with them") orelse false;
    // the compiler's version text is an input of every fatbin, so a new nvcc rebuilds them all
    const version: ?std.Build.LazyPath = if (prebuilt == null and nvcc != null) blk: {
        const run = b.addSystemCommand(&.{ nvcc.?, "--version" });
        run.has_side_effects = true;
        break :blk run.captureStdOut(.{});
    } else null;
    var images: [kernels.len]?std.Build.LazyPath = @splat(null);
    const fatbin_step = b.step("fatbins", "Build and install the CUDA kernel fatbins alone");
    for (kernels, &images) |k, *image| {
        if (prebuilt) |dir| {
            image.* = b.graph.cwdRelativePath(b.pathJoin(&.{ dir, b.fmt("{s}.fatbin", .{k.name}) }));
        } else if (nvcc) |tool| {
            image.* = fatbin(b, tool, version.?, k, sms);
        }
        if (image.*) |file| fatbin_step.dependOn(&b.addInstallFile(file, b.fmt("fatbin/{s}.fatbin", .{k.name})).step);
    }
    // MiniMax H3's kernel image: its library and the contract's runner take it
    const h3_image = if (nvcc) |tool| fatbin(b, tool, version orelse blk: {
        const run = b.addSystemCommand(&.{ tool, "--version" });
        run.has_side_effects = true;
        break :blk run.captureStdOut(.{});
    }, .{ .name = "h3", .flags = &.{"-O3"} }, sms) else null;
    @import("h3_host.zig").library(b, target, optimize, h3_image);
    const cuda = runtime(b, target, optimize, if (nvcc != null or prebuilt != null) &images else &.{});
    const mods = family(b, target, optimize, cuda, draft_ids);
    const cli = b.createModule(.{ .root_source_file = b.path("zig/src/cli/cuda_main.zig"), .target = target, .optimize = optimize, .link_libc = true, .strip = strip });
    cli.addImport("cuda", cuda);
    cli.addImport("core", mods.core);
    cli.addImport("lanes", mods.lanes);
    cli.addImport("nemotron", mods.nemotron);
    const native = engines(b, target, optimize, cuda, mods.core, mods.lanes, mods.nemotron, mods.heat, mods.qwen35, mods.qwen_vision);
    cli.addOptions("build_options", build_options);
    cli.addImport("checkpoint_cli", checkpointCli(b, target, optimize, strip, native.engines, mods.core));
    b.installArtifact(b.addExecutable(.{ .name = "tensorfold", .root_module = cli }));
    b.getInstallStep().dependOn(&b.addInstallDirectory(.{ .source_dir = b.path("zig/kernels/dflash2-aot"), .install_dir = .prefix, .install_subdir = "share/tensorfold/dflash2/sm121" }).step);
    const runner = b.createModule(.{ .root_source_file = b.path("zig/tests/cuda/main.zig"), .target = target, .optimize = optimize, .link_libc = true, .strip = strip });
    runner.addImport("cuda", cuda);
    runner.addImport("lanes", mods.lanes);
    runner.addImport("nemotron", mods.nemotron);
    runner.addImport("core", mods.core);
    runner.addImport("engine_api", native.api);
    runner.addImport("cuda_bound", b.createModule(.{ .root_source_file = b.path("zig/src/native/cuda_bound.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{.{ .name = "lanes", .module = mods.lanes }} }));
    runner.addImport("flashnext", flashnextModule(b, target, optimize, cuda, mods.core, mods.lanes, mods.qwen_vision));
    runner.addImport("lfm2", lfm2Module(b, target, optimize, cuda, mods.core));
    runner.addImport("radio", mods.radio);
    const exlr8 = exlr8_build.coreAlias(b, target, optimize, mods.core);
    runner.addImport("exlr8", exlr8);
    const exlr8_tests = exlr8_build.testModules(b, target, optimize, exlr8);
    for (exlr8_tests) |m| runner.addImport(m.name, m.module);
    runner.addImport("exlr8_frozen", b.createModule(.{ .root_source_file = b.path("zig/tests/fixtures/exlr8/frozen.zig"), .target = target, .optimize = optimize }));
    runner.addImport("exlr8_frozen_dense", b.createModule(.{ .root_source_file = b.path("zig/tests/fixtures/exlr8/frozen_dense.zig"), .target = target, .optimize = optimize }));
    for (exlr8_build.denseModules(b, target, optimize, exlr8, exlr8_tests[0])) |m| runner.addImport(m.name, m.module);
    const glm_cuda = glmCuda(b, target, optimize, cuda, mods.core, mods.lanes, mods.heat);
    runner.addImport("glm_cuda", glm_cuda);
    const runner_exe = b.addExecutable(.{ .name = "tf-cuda-test", .root_module = runner });
    b.installArtifact(runner_exe);
    // tf-tp-watch-check: the ranks' lines with no GPU or NCCL; a failing or dying rank stops every rank
    b.installArtifact(b.addExecutable(.{ .name = "tf-tp-watch-check", .root_module = b.createModule(.{ .root_source_file = b.path("zig/tests/cuda/tp_watch_check.zig"), .target = target, .optimize = optimize, .link_libc = true, .strip = strip, .imports = &.{.{ .name = "cuda", .module = cuda }} }) }));
    flashnextChecks(b, target, optimize, cuda, mods.core, mods.lanes, mods.qwen_vision);
    b.step("compile-cuda-tests", "Compile CUDA runner without executing a GPU").dependOn(&runner_exe.step);
    const startup = b.addObject(.{ .name = "modelopt-startup", .root_module = b.createModule(.{ .root_source_file = b.path("zig/modelopt_startup_compile.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{.{ .name = "native", .module = native.engines }} }) });
    _ = startup.getEmittedBin();
    b.step("compile-modelopt-startup", "Compile actual native startup with test-only absent GPU images").dependOn(&startup.step);
    const controls = if (nvcc) |tool| fatbin(b, tool, version.?, .{ .name = "contract_controls", .flags = &.{"-O3"} }, sms) else null;
    @import("qwen_contract.zig").images(b, runner_exe, controls, h3_image, kernels, &images);
    const server_exe = nativeServer(b, target, optimize, cuda, mods.core, mods.lanes, mods.nemotron, mods.heat, mods.qwen35, mods.qwen_vision, mods.tokenizer, build_options, true, strip);
    b.step("compile-config-server", "Compile the actual CUDA server config entry point without installing or running").dependOn(&server_exe.step);
    // tf-qwen35-cuda: the Qwen CUDA forward's rows for a reference to compare, and its tower on picture files
    const q35 = b.createModule(.{ .root_source_file = b.path("zig/tests/qwen35_cuda_main.zig"), .target = target, .optimize = optimize, .link_libc = true, .strip = strip });
    q35.addImport("cuda", cuda);
    q35.addImport("core", mods.core);
    q35.addImport("qwen35", mods.qwen35);
    b.installArtifact(b.addExecutable(.{ .name = "tf-qwen35-cuda", .root_module = q35 }));
    const d3_tool = b.createModule(.{ .root_source_file = b.path("zig/src/families/drama3/generate.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{ .{ .name = "core", .module = mods.core }, .{ .name = "drama3", .module = @import("drama3_host.zig").module(b, target, optimize, cuda, mods.core) } } });
    const d3_exe = b.addExecutable(.{ .name = "drama3-generate", .root_module = d3_tool });
    b.step("drama3-generate", "Drama generation and replay controls on CUDA").dependOn(&b.addInstallArtifact(d3_exe, .{}).step);
    const qwen_probe = b.addExecutable(.{ .name = "tf-qwen35-probe", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/cuda/qwen35_probe.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "qwen35_cuda", .module = qwen35Module(b, target, optimize, cuda, mods.core, mods.qwen35) }, .{ .name = "core", .module = mods.core }, .{ .name = "cuda", .module = cuda } },
    }) });
    b.step("qwen35-probe", "Build the native Qwen35 checkpoint and teacher-forcing probe").dependOn(&b.addInstallArtifact(qwen_probe, .{}).step);
    const qwen_kernels_probe = b.addExecutable(.{ .name = "tf-qwen35-kernels-probe", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/cuda/qwen35_kernels_probe.zig"),
        .target = target,
        .optimize = .ReleaseSafe,
        .link_libc = true,
        .imports = &.{
            .{ .name = "cuda", .module = cuda },
            .{ .name = "qwen35_ops", .module = b.createModule(.{ .root_source_file = b.path("zig/src/families/qwen3_5/cuda_kernels.zig"), .target = target, .optimize = optimize, .imports = &.{.{ .name = "cuda", .module = cuda }} }) },
            .{ .name = "qwen35_ref", .module = b.createModule(.{ .root_source_file = b.path("zig/src/families/qwen3_5/cuda_kernels_test.zig"), .target = target, .optimize = .ReleaseSafe }) },
        },
    }) });
    b.step("qwen35-kernels-probe", "Build only the Qwen35 device-kernel parity probe; never run it").dependOn(&b.addInstallArtifact(qwen_kernels_probe, .{}).step);
    const qwen_bench_options = b.addOptions();
    qwen_bench_options.addOption(std.builtin.OptimizeMode, "engine_optimization", optimize);
    qwen_bench_options.addOption([]const u8, "source_commit", b.option([]const u8, "qwen35-probe-source", "Source revision embedded in the Qwen tuning probes") orelse "unrecorded");
    const qwen_decode_profile = b.addExecutable(.{ .name = "tf-qwen35-decode-profile", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/cuda/qwen35_decode_profile.zig"),
        .target = target,
        .optimize = .ReleaseSafe,
        .link_libc = true,
        .imports = &.{ .{ .name = "qwen35_adapter", .module = qwen35AdapterModule(b, target, optimize, mods.lanes) }, .{ .name = "qwen35_cuda", .module = qwen35Module(b, target, optimize, cuda, mods.core, mods.qwen35) }, .{ .name = "cuda", .module = cuda }, .{ .name = "lanes", .module = mods.lanes }, .{ .name = "core", .module = mods.core } },
    }) });
    qwen_decode_profile.root_module.addOptions("qwen_bench_options", qwen_bench_options);
    b.step("qwen35-decode-profile", "Build only the Qwen35 eager/graph diagnostic; never run it").dependOn(&b.addInstallArtifact(qwen_decode_profile, .{}).step);
    return server_exe;
}

/// The GLM-5.3-Flash CUDA family over the runtime, the core, the lane core, canon's shares and core/heat.zig.
fn glmCuda(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, core: *std.Build.Module, lanes: *std.Build.Module, heat: *std.Build.Module) *std.Build.Module {
    const canon = b.createModule(.{ .root_source_file = b.path("zig/src/cluster/canon.zig"), .target = target, .optimize = optimize });
    const m = b.createModule(.{ .root_source_file = b.path("zig/src/families/glm/cuda.zig"), .target = target, .optimize = optimize, .link_libc = true });
    m.addImport("cuda", cuda);
    m.addImport("core", core);
    m.addImport("canon", canon);
    m.addImport("lanes", lanes);
    m.addImport("heat", heat);
    const cudnn_headers = b.createModule(.{ .root_source_file = b.path("zig/src/cuda/vision_headers/cudnn_headers.zig"), .target = target, .optimize = optimize });
    const codec_headers = b.createModule(.{ .root_source_file = b.path("zig/src/cuda/vision_headers/image_codec_headers.zig"), .target = target, .optimize = optimize });
    const cudnn = b.createModule(.{ .root_source_file = b.path("zig/src/cuda/glm_vision_cudnn.zig"), .target = target, .optimize = optimize, .link_libc = true });
    cudnn.addImport("cudnn_headers", cudnn_headers);
    cudnn.addImport("cuda", cuda);
    const sdpa = b.createModule(.{ .root_source_file = b.path("zig/src/cuda/glm_vision_sdpa.zig"), .target = target, .optimize = optimize, .link_libc = true });
    sdpa.addImport("cudnn_headers", cudnn_headers);
    sdpa.addImport("cuda", cuda);
    const vision_ops = b.createModule(.{ .root_source_file = b.path("zig/src/cuda/glm_vision_ops.zig"), .target = target, .optimize = optimize, .link_libc = true });
    vision_ops.addImport("cuda", cuda);
    m.addImport("cudnn", cudnn);
    m.addImport("vision_sdpa", sdpa);
    m.addImport("vision_ops", vision_ops);
    m.addImport("image_codec_headers", codec_headers);
    return m;
}

/// LFM2 (d1's decision model) on CUDA: config, weights, kernels and the forward that scores labels.
fn lfm2Module(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, core: *std.Build.Module) *std.Build.Module {
    return b.createModule(.{ .root_source_file = b.path("zig/src/families/lfm2/lfm2.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = core } } });
}

fn qwen35Module(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, core: *std.Build.Module, qwen35: *std.Build.Module) *std.Build.Module {
    return b.createModule(.{ .root_source_file = b.path("zig/src/families/qwen3_5/modelopt_cuda.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = core }, .{ .name = "qwen35", .module = qwen35 } } });
}

fn qwen35AdapterModule(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, lanes: *std.Build.Module) *std.Build.Module {
    return b.createModule(.{ .root_source_file = b.path("zig/src/native/qwen35_backend.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{.{ .name = "lanes", .module = lanes }} });
}

/// RADIO image input (Nemotron Omni checkpoints): the processor, the tower and projector, the prompt rows.
fn radioModule(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, core: *std.Build.Module, lanes: *std.Build.Module) *std.Build.Module {
    return b.createModule(.{ .root_source_file = b.path("zig/src/families/radio/radio.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = core }, .{ .name = "lanes", .module = lanes } } });
}

/// The Qwen-VL tower on CUDA that Qwen and Flash Next share, one module for a build so both import the same one.
fn qwenVisionModule(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, core: *std.Build.Module, lanes: *std.Build.Module) *std.Build.Module {
    return b.createModule(.{ .root_source_file = b.path("zig/src/vision/qwen_cuda.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = core }, .{ .name = "lanes", .module = lanes } } });
}

/// Flash Next's CUDA family, sharing the maintained runtime and lane core.
fn flashnextModule(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, core: *std.Build.Module, lanes: *std.Build.Module, qwen_vision: *std.Build.Module) *std.Build.Module {
    const m = b.createModule(.{ .root_source_file = b.path("zig/src/families/flashnext/cuda_launch.zig"), .target = target, .optimize = optimize, .link_libc = true });
    m.addImport("cuda", cuda);
    m.addImport("core", core);
    m.addImport("lanes", lanes);
    m.addImport("qwen_vision", qwen_vision);
    return m;
}

/// The CUDA engines a native server opens (native/cuda.zig), over the given runtime and families.
fn engines(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, core: *std.Build.Module, lanes: *std.Build.Module, nemotron: *std.Build.Module, heat: *std.Build.Module, qwen35: *std.Build.Module, qwen_vision: *std.Build.Module) struct { api: *std.Build.Module, engines: *std.Build.Module } {
    const api = b.createModule(.{ .root_source_file = b.path("zig/src/core/engine_api.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{.{ .name = "lanes", .module = lanes }} });
    const mod = b.createModule(.{
        .root_source_file = b.path("zig/src/native/cuda.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{
            .{ .name = "cuda", .module = cuda },
            .{ .name = "engine_api", .module = api },
            .{ .name = "lanes", .module = lanes },
            .{ .name = "nemotron", .module = nemotron },
            .{ .name = "drama3", .module = @import("drama3_host.zig").module(b, target, optimize, cuda, core) },
            .{ .name = "lfm2", .module = lfm2Module(b, target, optimize, cuda, core) },
            .{ .name = "qwen35_cuda", .module = qwen35Module(b, target, optimize, cuda, core, qwen35) },
            .{ .name = "core", .module = core },
            .{ .name = "flashnext", .module = flashnextModule(b, target, optimize, cuda, core, lanes, qwen_vision) },
            .{ .name = "glm_cuda", .module = glmCuda(b, target, optimize, cuda, core, lanes, heat) },
            .{ .name = "qwen35", .module = qwen35 },
        },
    });
    return .{ .api = api, .engines = mod };
}

/// The checkpoint subcommands' module: `models`, `info` and `pull` over the CUDA families.
fn checkpointCli(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, strip: ?bool, engines_mod: *std.Build.Module, core: *std.Build.Module) *std.Build.Module {
    return b.createModule(.{ .root_source_file = b.path("zig/src/cli/cli.zig"), .target = target, .optimize = optimize, .link_libc = true, .strip = strip, .imports = &.{ .{ .name = "native_engines", .module = engines_mod }, exlr8_build.cliImport(b, target, optimize, core), .{ .name = "core", .module = core } } });
}

/// `zig build native`: tensorfold-native with the CUDA engines into zig-out/native/bin, as the Metal build makes it.
fn nativeServer(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, core: *std.Build.Module, lanes: *std.Build.Module, nemotron: *std.Build.Module, heat: *std.Build.Module, qwen35: *std.Build.Module, qwen_vision: *std.Build.Module, tokenizer: *std.Build.Module, build_options: *std.Build.Step.Options, install_native: bool, strip: ?bool) *std.Build.Step.Compile {
    const m = engines(b, target, optimize, cuda, core, lanes, nemotron, heat, qwen35, qwen_vision);
    // the HTTP side keeps its safety checks; the engine below it runs at `optimize` (the tokenizer is the family's)
    const template = b.createModule(.{ .root_source_file = b.path("zig/src/core/template/template.zig"), .target = target, .optimize = .ReleaseSafe, .link_libc = true });
    const platform = b.createModule(.{ .root_source_file = b.path("zig/src/core/platform.zig"), .target = target, .optimize = .ReleaseSafe, .link_libc = true });
    const exe = b.addExecutable(.{ .name = "tensorfold-native", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/src/server/main.zig"),
        .target = target,
        .optimize = .ReleaseSafe,
        .link_libc = true,
        .imports = &.{ .{ .name = "engine_api", .module = m.api }, .{ .name = "tokenizer", .module = tokenizer }, .{ .name = "template", .module = template }, .{ .name = "platform", .module = platform }, .{ .name = "native_engines", .module = m.engines }, .{ .name = "checkpoint_cli", .module = checkpointCli(b, target, .ReleaseSafe, null, m.engines, core) } },
    }) });
    exe.root_module.addOptions("build_options", build_options);
    _ = @import("server.zig").addTo(b, exe.root_module, target, .ReleaseSafe);
    if (!install_native) {
        exe.root_module.strip = true;
        return exe;
    }
    // before the install step, which decides then whether a .pdb comes with the executable
    exe.root_module.strip = strip;
    const install = b.addInstallArtifact(exe, .{ .dest_dir = .{ .override = .{ .custom = "native/bin" } } });
    const native_step = b.step("native", "tensorfold-native with the CUDA engines into zig-out/native/bin");
    native_step.dependOn(&install.step);
    native_step.dependOn(&b.addInstallDirectory(.{ .source_dir = b.path("zig/kernels/dflash2-aot"), .install_dir = .prefix, .install_subdir = "native/share/tensorfold/dflash2/sm121" }).step);
    return exe;
}

/// Host tests of the CUDA runtime, the backend-neutral core, the lane core and the CUDA family (no GPU), on any host.
pub fn hostTests(b: *std.Build, draft_ids: *std.Build.Module, build_options: *std.Build.Step.Options, all: *std.Build.Step, server_steps: bool) void {
    const step = b.step("test-cuda-host", "The CUDA side's host unit tests alone (no GPU work)");
    all.dependOn(step);
    const host = b.graph.host;
    step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = b.createModule(.{ .root_source_file = b.path("zig/build/check_fatbins.zig"), .target = host }) })).step);
    const cuda = runtime(b, host, .debug, &.{});
    const bundle_operators = b.addRunArtifact(b.addTest(.{ .root_module = cuda, .filters = &.{ "split-M", "atomic FP8" } }));
    bundle_operators.has_side_effects = true;
    b.step("test-qwen35-bundle-operators-host", "Split-M and atomic FP8 group tests (CPU only)").dependOn(&bundle_operators.step);
    @import("h3_host.zig").targets(b, cuda, step);
    const mods = family(b, host, .debug, cuda, draft_ids);
    const deferred_run = b.addRunArtifact(b.addTest(.{ .root_module = mods.lanes, .filters = &.{"deferred draft shape"} }));
    b.step("test-deferred-draft-host", "Deferred head selection after host work and copy precedence").dependOn(&deferred_run.step);
    const contract_glm = glmCuda(b, host, .debug, cuda, mods.core, mods.lanes, mods.heat);
    const contract_host = b.createModule(.{ .root_source_file = b.path("zig/tests/cuda/contract_host.zig"), .target = host, .optimize = .debug, .link_libc = true, .imports = &.{ .{ .name = "core", .module = mods.core }, .{ .name = "cuda", .module = cuda }, .{ .name = "glm_cuda", .module = contract_glm } } });
    const contract_run = b.addRunArtifact(b.addTest(.{ .root_module = contract_host }));
    step.dependOn(&contract_run.step);
    b.step("test-kernel-contract-host", "Contract reference fixtures and offline metadata tests (CPU only)").dependOn(&contract_run.step);

    @import("qwen_contract.zig").host(b, mods.core, step);
    // The retained Nemotron has no upstream MTP-policy/subset-head implementation. Its own module tests run below;
    // do not expose upstream-family targets whose filters would match nothing, or whose API is absent.
    const exl3_host = b.step("test-exl3-cuda-host", "EXL3 loader and CUDA host regressions without a GPU");
    const exl3_compile = b.addObject(.{ .name = "exl3_host_compile", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/exl3_compile.zig"),
        .target = host,
        .link_libc = true,
        .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = mods.core } },
    }) });
    step.dependOn(&exl3_compile.step);
    exl3_host.dependOn(&exl3_compile.step);
    for ([_]*std.Build.Module{ cuda, mods.core }) |m| {
        exl3_host.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = m })).step);
    }
    const contract_arch = b.createModule(.{ .root_source_file = b.path("zig/tests/cuda/contract_arch_test.zig"), .target = host, .optimize = .debug, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = mods.core } } });
    step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = contract_arch })).step);
    const writeback_contract = b.addRunArtifact(b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/cuda/contract_flashnext_writeback.zig"),
        .target = host,
        .optimize = .debug,
        .link_libc = true,
        .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = mods.core } },
    }) }));
    step.dependOn(&writeback_contract.step);
    const utility_contract = b.addRunArtifact(b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/cuda/contract_utility_ref.zig"),
        .target = host,
        .optimize = .debug,
        .link_libc = true,
        .imports = &.{.{ .name = "core", .module = mods.core }},
    }) }));
    step.dependOn(&utility_contract.step);
    const flash_contract = b.step("test-flashnext-contract-host", "Flash Next contract stress and rounding references without a GPU");
    flash_contract.dependOn(&writeback_contract.step);
    flash_contract.dependOn(&utility_contract.step);
    const row_best_contract = b.addRunArtifact(b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/cuda/contract_row_best_ref.zig"),
        .target = host,
        .optimize = .debug,
        .link_libc = true,
        .imports = &.{.{ .name = "core", .module = mods.core }},
    }) }));
    step.dependOn(&row_best_contract.step);
    flash_contract.dependOn(&row_best_contract.step);
    const qsa_contract = b.addRunArtifact(b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/cuda/contract_qsa_ref.zig"),
        .target = host,
        .optimize = .debug,
        .link_libc = true,
        .imports = &.{.{ .name = "core", .module = mods.core }},
    }) }));
    step.dependOn(&qsa_contract.step);
    flash_contract.dependOn(&qsa_contract.step);
    const reference_step = b.step("test-cuda-reference", "CUDA's FP64 comparison with CPU-only readback (no GPU)");
    const projection_step = b.step("test-cuda-projection-fixture", "Projection fixture dispatch controls with CPU-only CUDA calls");
    for ([_]std.builtin.OptimizeMode{ .debug, .fast }) |mode| {
        const reference = b.addRunArtifact(b.addTest(.{ .root_module = b.createModule(.{
            .root_source_file = b.path("zig/tests/cuda/quant_ref_host_tests.zig"),
            .target = host,
            .optimize = mode,
            .link_libc = true,
            .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = mods.core } },
        }) }));
        step.dependOn(&reference.step);
        reference_step.dependOn(&reference.step);
        const projection = b.addRunArtifact(b.addTest(.{ .root_module = b.createModule(.{
            .root_source_file = b.path("zig/tests/cuda/projection_f32_host_tests.zig"),
            .target = host,
            .optimize = mode,
            .link_libc = true,
            .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = mods.core } },
        }) }));
        step.dependOn(&projection.step);
        projection_step.dependOn(&projection.step);
    }
    const bolt = b.step("test-bolt-host", "Nemotron Bolt and shared lane host contracts");
    const cuda_target = b.resolveTargetQuery(.{ .cpu_arch = host.result.cpu.arch, .os_tag = .linux, .abi = .gnu });
    const check_cuda = runtime(b, cuda_target, .debug, &.{});
    const check_mods = family(b, cuda_target, .debug, check_cuda, draft_ids);
    const compile_bolt = b.addObject(.{ .name = "bolt_host_compile", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/bolt_compile.zig"),
        .target = cuda_target,
        .link_libc = true,
        .imports = &.{ .{ .name = "cuda", .module = check_cuda }, .{ .name = "nemotron", .module = check_mods.nemotron } },
    }) });
    bolt.dependOn(&compile_bolt.step);
    step.dependOn(&compile_bolt.step);
    for ([_]*std.Build.Module{ mods.nemotron, mods.lanes, mods.radio }) |m|
        bolt.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = m })).step);
    const video_step = b.step("test-omni-video-host", "Actual RADIO and Bolt video boundaries without a GPU or decoder library");
    const video_filter = b.option([]const u8, "omni-video-filter", "One Omni video regression");
    for ([_]*std.Build.Module{ mods.nemotron, mods.radio }) |m| {
        const run = b.addRunArtifact(b.addTest(.{ .root_module = m, .filters = if (video_filter) |f| &.{f} else &.{} }));
        run.has_side_effects = true;
        video_step.dependOn(&run.step);
    }
    const drama3 = @import("drama3_host.zig").hostTests(b, step, cuda, mods.core);
    const native_modules = engines(b, host, .debug, cuda, mods.core, mods.lanes, mods.nemotron, mods.heat, mods.qwen35, mods.qwen_vision);
    const qwen35 = qwen35Module(b, host, .debug, cuda, mods.core, mods.qwen35);
    const qwen35_run = b.addRunArtifact(b.addTest(.{ .root_module = qwen35 }));
    qwen35_run.has_side_effects = true;
    const qwen35_step = b.step("test-qwen35-cuda-host", "Qwen35 CUDA checkpoint and execution contracts without GPU work");
    qwen35_step.dependOn(&qwen35_run.step);
    qwen35_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = mods.qwen35 })).step);
    const sampler_turn = b.createModule(.{
        .root_source_file = b.path("zig/tests/cuda/qwen35_sampler_turn.zig"),
        .target = host,
        .optimize = .ReleaseSafe,
        .link_libc = true,
        .imports = &.{
            .{ .name = "lanes", .module = mods.lanes },
            .{ .name = "sampler_before", .module = b.createModule(.{ .root_source_file = b.path("zig/tests/fixtures/qwen35/sampler_before.zig"), .target = host, .optimize = .ReleaseFast }) },
        },
    });
    b.step("test-qwen35-sampler-turn", "CPU-only sampler turn call-count and fixed-work contracts").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = sampler_turn })).step);
    const qwen35_probe_admission = b.createModule(.{
        .root_source_file = b.path("zig/tests/cuda/qwen35_probe_device.zig"),
        .target = host,
        .optimize = .debug,
        .link_libc = true,
        .imports = &.{.{ .name = "cuda", .module = cuda }},
    });
    const qwen35_screens = b.createModule(.{
        .root_source_file = b.path("zig/tests/cuda/qwen35_screens_host.zig"),
        .target = host,
        .optimize = .ReleaseSafe,
        .link_libc = true,
        .imports = &.{ .{ .name = "lanes", .module = mods.lanes }, .{ .name = "qwen35_adapter", .module = qwen35AdapterModule(b, host, .ReleaseSafe, mods.lanes) }, .{ .name = "cuda", .module = cuda }, .{ .name = "qwen35_cuda", .module = qwen35 }, .{ .name = "core", .module = mods.core } },
    });
    b.step("test-qwen35-screens-host", "CPU-only expert and GDN screen admission and layout controls").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = qwen35_screens })).step);
    qwen35_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = qwen35_probe_admission })).step);
    const qwen35_adapter = b.createModule(.{
        .root_source_file = b.path("zig/src/native/qwen35_cuda.zig"),
        .target = host,
        .optimize = .debug,
        .link_libc = true,
        .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "qwen35_cuda", .module = qwen35 }, .{ .name = "engine_api", .module = native_modules.api }, .{ .name = "lanes", .module = mods.lanes } },
    });
    qwen35_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = qwen35_adapter })).step);
    step.dependOn(&qwen35_run.step);
    const native = native_modules.engines;
    const qwen_native_run = b.addRunArtifact(b.addTest(.{ .root_module = native, .filters = &.{ "the Qwen host profile", "every registered family", "Qwen ModelOpt checkpoint refusal" } }));
    b.step("test-qwen35-native-host", "ModelOpt lane-host integration and combined native catalog without a GPU").dependOn(&qwen_native_run.step);
    const windows_port = @import("windows_host.zig").steps(b, cuda, native_modules.api);
    @import("drama3_host.zig").native(b, step, drama3, mods.core, native_modules.api, cuda, "drama_cuda_native_host");
    @import("drama3_host.zig").metal(b, mods.core, native_modules.api);
    const safety = b.step("test-kv-safety-host", "Actual native KV startup budget and paged constructors under allocation/completion faults");
    safety.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = native, .filters = &.{"kv safety:"} })).step);
    const merge_checks = b.step("test-kv-merge-host", "KV page integration with landed snapshot modes and budgets");
    merge_checks.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = native, .filters = &.{"kv merge:"} })).step);
    const load_check = b.step("test-bolt-load-host", "Bolt load destination and native-parent retention without a GPU");
    const load_filter = b.option([]const u8, "bolt-load-filter", "Select one load regression");
    const load_module = b.createModule(.{ .root_source_file = b.path("zig/src/native/cuda_load_gpu.zig"), .target = host, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "nemotron", .module = mods.nemotron } } });
    for ([_]*std.Build.Module{ load_module, mods.radio }) |m| {
        const check = b.addRunArtifact(b.addTest(.{ .root_module = m, .filters = if (load_filter) |f| &.{f} else &.{} }));
        check.has_side_effects = true;
        load_check.dependOn(&check.step);
    }

    const offer_tests = b.addRunArtifact(b.addTest(.{ .root_module = b.createModule(.{ .root_source_file = b.path("zig/src/native/cuda_vision_offer.zig"), .target = host, .link_libc = true, .imports = &.{ .{ .name = "engine_api", .module = native_modules.api }, .{ .name = "nemotron", .module = mods.nemotron }, .{ .name = "qwen35", .module = mods.qwen35 } } }), .filters = if (video_filter) |f| &.{f} else &.{} }));
    offer_tests.has_side_effects = true;
    video_step.dependOn(&offer_tests.step);
    const compile_modules = engines(b, cuda_target, .debug, check_cuda, check_mods.core, check_mods.lanes, check_mods.nemotron, check_mods.heat, check_mods.qwen35, check_mods.qwen_vision);
    const glm_open = b.addObject(.{ .name = "glm-native-open", .root_module = b.createModule(.{ .root_source_file = b.path("zig/glm_open_compile.zig"), .target = cuda_target, .optimize = .debug, .link_libc = true, .imports = &.{.{ .name = "glm", .module = glmCuda(b, cuda_target, .debug, check_cuda, check_mods.core, check_mods.lanes, check_mods.heat) }} }) });
    _ = glm_open.getEmittedBin();
    b.step("compile-glm-native-open", "Compile the actual GLM Linux open body with empty GPU images").dependOn(&glm_open.step);
    const open_object = b.addObject(.{ .name = "omni-video-native-compile", .root_module = b.createModule(.{ .root_source_file = b.path("zig/tests/omni_video_compile.zig"), .target = cuda_target, .link_libc = true, .imports = &.{.{ .name = "native", .module = compile_modules.engines }} }) });
    _ = open_object.getEmittedBin();
    b.step("compile-omni-video-host", "Compile actual native CUDA startup and offer for Linux without images or GPU execution").dependOn(&open_object.step);
    const h3_checkpoint = b.createModule(.{ .root_source_file = b.path("zig/src/families/h3/checkpoint.zig"), .target = host, .optimize = .debug, .link_libc = true });
    step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = h3_checkpoint })).step);
    // the HTTP server's unit tests (zig/src/server/root.zig), which the macOS build runs, on Linux hosts too
    const template = b.createModule(.{ .root_source_file = b.path("zig/src/core/template/template.zig"), .target = host, .optimize = .debug, .link_libc = true });
    const platform = b.createModule(.{ .root_source_file = b.path("zig/src/core/platform.zig"), .target = host, .optimize = .debug, .link_libc = true });
    // -Dtest-filter=TEXT runs only the server tests whose names contain TEXT (the stall and half-close tests by name)
    const filters: []const []const u8 = if (b.option([]const u8, "test-filter", "Run only the server tests whose names contain this text")) |f| b.allocator.dupe([]const u8, &.{f}) catch @panic("OOM") else &.{};
    const server_tests = b.createModule(.{
        .root_source_file = b.path("zig/src/server/root.zig"),
        .target = host,
        .optimize = .debug,
        .link_libc = true,
        .imports = &.{ .{ .name = "engine_api", .module = native_modules.api }, .{ .name = "tokenizer", .module = mods.tokenizer }, .{ .name = "template", .module = template }, .{ .name = "platform", .module = platform } },
    });
    const server_json = @import("server.zig").addTo(b, server_tests, host, .debug);
    windows_port.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = server_tests, .filters = &.{"hardware names"} })).step);
    b.step("test-benchmark-manifest-host", "Manifest identity through the HTTP server without shaders").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = server_tests, .filters = &.{"manifest"} })).step);
    b.step("test-issue-934", "Benchmark config admission and receipt regressions").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = server_tests, .filters = &.{"issue 934:"} })).step);
    b.step("test-issue-969", "First-EOS classification and incomplete benchmark messages without a GPU").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = server_tests, .filters = &.{"issue 969"} })).step);
    b.step("test-tool-required-host", "Required tool parsing, grammar form and constrained lane replies on the CPU").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = server_tests, .filters = &.{"required tools:"} })).step);
    const q27 = @import("qwen27_required.zig").modules(b, mods.lanes);
    server_tests.addImport("qwen27_mask_fake", q27.fake);
    server_tests.addImport("qwen27_masks", q27.masks);
    b.step("test-qwen27-required-host", "Qwen27 required tool routes through the masked lane fake").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = server_tests, .filters = &.{"qwen27 tool calls:"} })).step);
    const server_test = b.addRunArtifact(b.addTest(.{ .root_module = server_tests, .filters = filters }));
    const json_test = b.addRunArtifact(b.addTest(.{ .root_module = server_json }));
    const server_step = b.step(
        "test-server-cpu",
        "The HTTP server's unit tests (routes, templates, tool parsing) without a GPU",
    );
    for ([_]*std.Build.Step.Run{ server_test, json_test }) |run| {
        step.dependOn(&run.step);
        server_step.dependOn(&run.step);
    }
    // the server goldens and Cortheon's fuzz step here when the target is not macOS, whose Metal build defines them
    if (server_steps) {
        golden_build.step(b, host, server_tests, native_modules.api, mods.tokenizer, template);
        @import("cortheon_fuzz.zig").add(b, host, native_modules.api, mods.tokenizer, template);
    }
    // the tokenizer's own tests (memo, steps, regex, unicode) run here: no other test binary reaches them
    for ([_]*std.Build.Module{ cuda, mods.core, mods.lanes, mods.nemotron, mods.qwen35, mods.qwen_vision, flashnextModule(b, host, .debug, cuda, mods.core, mods.lanes, mods.qwen_vision), lfm2Module(b, host, .debug, cuda, mods.core), mods.radio, mods.heat, mods.tokenizer, native_modules.api, native, stagger(b, host, .debug) }) |m| step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = m })).step);
    // Upstream's modelopt-host fixtures are not part of the retained Nemotron implementation.
    // the GLM-5.3-Flash CUDA family's host side
    const glm = glmCuda(b, host, .debug, cuda, mods.core, mods.lanes, mods.heat);
    // Upstream's DFlash loader, packed-linear and ring-cache roots exercise a different GLM family. The retained
    // family has its own DFlash/MCG4/spill tests in this module, not those upstream implementations.
    const glm_host_run = b.addRunArtifact(b.addTest(.{ .root_module = glm }));
    step.dependOn(&glm_host_run.step);
    b.step("test-glm-cuda-host", "GLM CUDA family host tests without a GPU").dependOn(&glm_host_run.step);
    const binding_run = b.addRunArtifact(b.addTest(.{ .root_module = glm, .filters = &.{"padded native sequence capacity"} }));
    b.step("test-glm-vision-binding", "Padded native sequence versus logical vision context (CPU only)").dependOn(&binding_run.step);
    const codec_headers = b.createModule(.{ .root_source_file = b.path("zig/src/cuda/vision_headers/image_codec_headers.zig"), .target = host, .optimize = .debug });
    const codec_cases = b.createModule(.{ .root_source_file = b.path("zig/src/families/glm/cuda_vision_codec_test.zig"), .target = host, .optimize = .debug, .link_libc = true });
    codec_cases.addImport("cuda", cuda);
    codec_cases.addImport("image_codec_headers", codec_headers);
    const codec_run = b.addRunArtifact(b.addTest(.{ .root_module = codec_cases, .filters = &.{"actual"} }));
    b.step("test-glm-vision-codec", "PNG, JPEG and WebP native shim against Pillow pixel goldens (CPU only)").dependOn(&codec_run.step);
    // The retained GLM does not call upstream's common EXL3 prompt dispatcher; test-exl3-cuda-host checks the
    // shared loaders/runtime, while the retained family's own tests cover its actual dispatch.
    // the shares a GLM-5.3-Flash rank takes (cluster/canon.zig against the Python family's)
    const shares = b.createModule(.{ .root_source_file = b.path("zig/src/cluster/canon_glm_test.zig"), .target = host, .optimize = .debug });
    step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = shares })).step);
    // the tool-call parser's tests, run beside the server's on Macs: the GLM calls the CUDA port serves
    const parse_tests = b.createModule(.{ .root_source_file = b.path("zig/src/server/tool_parse_test.zig"), .target = host, .optimize = .debug, .link_libc = true, .imports = &.{ .{ .name = "engine_api", .module = native_modules.api }, .{ .name = "tokenizer", .module = mods.tokenizer }, .{ .name = "template", .module = template } } });
    _ = @import("server.zig").addTo(b, parse_tests, host, .debug);
    step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = parse_tests })).step);
    const cli = b.createModule(.{ .root_source_file = b.path("zig/src/cli/cuda_main.zig"), .target = host, .optimize = .debug, .link_libc = true });
    cli.addImport("cuda", cuda);
    cli.addImport("core", mods.core);
    cli.addImport("lanes", mods.lanes);
    cli.addImport("nemotron", mods.nemotron);
    cli.addOptions("build_options", build_options);
    cli.addImport("checkpoint_cli", checkpointCli(b, host, .debug, false, native_modules.engines, mods.core));
    b.step("test-info-config", "Info shares the checkpoint config reader and diagnostics").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = cli, .filters = &.{"issue 934:"} })).step);
    step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = cli })).step);
}

/// nvcc -fatbin with torch's flags, the kernel's own and one -gencode per SASS target, as the Python build passes them.
fn fatbin(b: *std.Build, nvcc: []const u8, version: std.Build.LazyPath, k: Kernel, sms: []const u8) std.Build.LazyPath {
    const run = b.addSystemCommand(&.{ nvcc, "-fatbin" });
    run.addFileInput(version);
    run.addArgs(&torch_flags);
    // with cl.exe as host compiler, CUDA 13's headers need MSVC's conforming preprocessor, and M_SQRT2 its math define
    if (b.graph.host.result.os.tag == .windows) run.addArgs(&.{ "-Xcompiler=/Zc:preprocessor", "-D_USE_MATH_DEFINES" });
    run.addArgs(k.flags);
    var it = std.mem.tokenizeScalar(u8, sms, ',');
    while (it.next()) |sm| {
        // arch-specific features (the block-scaled FP4 mma) from sm_120 on; older targets build the trapping fallback
        const a = if (k.arch_specific and (std.fmt.parseInt(u32, sm, 10) catch 0) >= 120) "a" else "";
        run.addArg(b.fmt("-gencode=arch=compute_{s}{s},code=sm_{s}{s}", .{ sm, a, sm, a }));
    }
    run.addArgs(&.{ "-MD", "-MF" });
    _ = run.addDepFileOutputArg2(b.fmt("{s}.d", .{k.name}), .{});
    run.addArg("-o");
    const out = run.addOutputFileArg(b.fmt("{s}.fatbin", .{k.name}));
    run.addFileArg(b.path(b.fmt("zig/kernels/cuda/{s}.cu", .{k.src orelse k.name})));
    return out;
}

pub const TpPartialImage = struct { image: std.Build.LazyPath, depfile: std.Build.LazyPath, flags: []const []const u8 };

/// The TP2 MoE partial image alone (zig/tp_moe_gpu_oracle_build.zig): torch's flags, the glue kernels' flags and one
/// SASS target, with the generated ABI header's directory on the include path. The caller resolves `source`, because
/// a standalone build's root is zig/ and `fatbin` above assumes the repository's. It is not one of `kernels`, so no
/// root target embeds, installs or checks it. `flags` is every argument but the paths, as a provenance record.
pub fn tpPartialImage(b: *std.Build, nvcc: []const u8, version: std.Build.LazyPath, source: std.Build.LazyPath, abi_dir: std.Build.LazyPath, sm: []const u8) TpPartialImage {
    const flags = tpPartialFlags(b, sm);
    const run = b.addSystemCommand(&.{nvcc});
    run.addFileInput(version);
    run.addArgs(flags);
    run.addPrefixedDirectoryArg("-I", abi_dir);
    run.addArgs(&.{ "-MD", "-MF" });
    const dep = run.addDepFileOutputArg2("nemotron_tp_partial.d", .{});
    run.addArg("-o");
    const out = run.addOutputFileArg("nemotron_tp_partial.fatbin");
    run.addFileArg(source);
    return .{ .image = out, .depfile = dep, .flags = flags };
}

/// The TP partial image's nvcc arguments other than paths: what a prebuilt image's manifest must have recorded.
pub fn tpPartialFlags(b: *std.Build, sm: []const u8) []const []const u8 {
    var flags: std.ArrayList([]const u8) = .empty;
    flags.append(b.allocator, "-fatbin") catch @panic("OOM");
    flags.appendSlice(b.allocator, &torch_flags) catch @panic("OOM");
    flags.appendSlice(b.allocator, catalog.glue) catch @panic("OOM");
    flags.append(b.allocator, b.fmt("-gencode=arch=compute_{s},code=sm_{s}", .{ sm, sm })) catch @panic("OOM");
    return flags.items;
}

/// Cross-build a release server, embedding the same fatbins on either host CPU.
pub fn distServer(b: *std.Build, target: std.Build.ResolvedTarget, draft_ids: *std.Build.Module, build_options: *std.Build.Step.Options, prebuilt: ?[]const u8) *std.Build.Step.Compile {
    var images: [kernels.len]?std.Build.LazyPath = @splat(null);
    if (prebuilt) |dir| for (kernels, &images) |k, *image| {
        image.* = b.graph.cwdRelativePath(b.pathJoin(&.{ dir, b.fmt("{s}.fatbin", .{k.name}) }));
    };
    const cuda = runtime(b, target, .fast, if (prebuilt != null) &images else &.{});
    const mods = family(b, target, .fast, cuda, draft_ids);
    return nativeServer(b, target, .fast, cuda, mods.core, mods.lanes, mods.nemotron, mods.heat, mods.qwen35, mods.qwen_vision, mods.tokenizer, build_options, false, null);
}

/// SASS targets every release fatbin holds: Ampere (sm_80, sm_86), RTX 50/PRO Blackwell (sm_120), GB10 (sm_121).
pub const release_sms = "80,86,120,121";

/// Validate the complete named input set, including images unused by today's server, and each file's release targets.
pub fn checkDistFatbins(b: *std.Build, dir: []const u8) *std.Build.Step {
    const tool = b.addExecutable(.{ .name = "check-fatbins", .root_module = b.createModule(.{ .root_source_file = b.path("zig/build/check_fatbins.zig"), .target = b.graph.host }) });
    const check = b.addRunArtifact(tool);
    check.addArg(release_sms);
    for (kernels) |k| check.addFileArg(b.graph.cwdRelativePath(b.pathJoin(&.{ dir, b.fmt("{s}.fatbin", .{k.name}) })));
    return &check.step;
}

/// Explicit model diagnostics; these tools do not qualify a checkpoint merely by compiling.
fn flashnextChecks(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, core: *std.Build.Module, lanes: *std.Build.Module, qwen_vision: *std.Build.Module) void {
    const flashnext = flashnextModule(b, target, optimize, cuda, core, lanes, qwen_vision);
    // not installed: one embedding row through the captured _embed cubin
    const embed_check = b.createModule(.{ .root_source_file = b.path("zig/src/families/flashnext/cuda_embed_main.zig"), .target = target, .optimize = optimize, .link_libc = true });
    embed_check.addImport("cuda", cuda);
    embed_check.addImport("core", core);
    embed_check.addImport("flashnext", flashnext);
    const embed_run = b.addRunArtifact(b.addExecutable(.{ .name = "flashnext-embed", .root_module = embed_check }));
    const embed_model = b.option([]const u8, "flashnext_model", "Flash Next checkpoint directory for the embedding check");
    const embed_kernels = b.option([]const u8, "flashnext_kernels", "captured Triton pack for the embedding check");
    const embed_token = b.option([]const u8, "flashnext_token", "token id for the embedding check") orelse "7";
    if (embed_model != null and embed_kernels != null) embed_run.addArgs(&.{ embed_model.?, embed_kernels.?, embed_token });
    b.step("flashnext-embed", "Upload the Flash Next embedding table and run _embed for one token").dependOn(&embed_run.step);
    const prompt_check = b.createModule(.{ .root_source_file = b.path("zig/src/families/flashnext/cuda_prompt_main.zig"), .target = target, .optimize = optimize, .link_libc = true });
    prompt_check.addImport("cuda", cuda);
    prompt_check.addImport("core", core);
    prompt_check.addImport("flashnext", flashnext);
    const prompt_run = b.addRunArtifact(b.addExecutable(.{ .name = "flashnext-prompt", .root_module = prompt_check }));
    const prompt_ids = b.option([]const u8, "flashnext_ids", "file of prompt token ids for the Flash Next prompt check");
    const prompt_count = b.option([]const u8, "flashnext_count", "greedy tokens to emit") orelse "8";
    if (embed_model != null and embed_kernels != null and prompt_ids != null) prompt_run.addArgs(&.{ embed_model.?, embed_kernels.?, prompt_ids.?, prompt_count });
    b.step("flashnext-prompt", "Run one Flash Next prompt on the captured CUDA kernels").dependOn(&prompt_run.step);
    // not installed: our own glue against a captured set on one loaded engine (scores of every prompt row, replies)
    const compare_check = b.createModule(.{ .root_source_file = b.path("zig/src/families/flashnext/cuda_compare_main.zig"), .target = target, .optimize = optimize, .link_libc = true });
    compare_check.addImport("cuda", cuda);
    compare_check.addImport("flashnext", flashnext);
    const compare_run = b.addRunArtifact(b.addExecutable(.{ .name = "flashnext-compare", .root_module = compare_check }));
    if (embed_model != null and embed_kernels != null and prompt_ids != null) compare_run.addArgs(&.{ embed_model.?, embed_kernels.?, prompt_ids.? });
    b.step("flashnext-compare", "Our Flash Next glue against a captured set: every prompt row's scores, perplexity, greedy replies").dependOn(&compare_run.step);
    // not installed: the native block selection against Triton's lists (capture_sparse.sh's fixtures)
    const select_check = b.createModule(.{ .root_source_file = b.path("zig/src/families/flashnext/cuda_select_main.zig"), .target = target, .optimize = optimize, .link_libc = true });
    select_check.addImport("cuda", cuda);
    const select_run = b.addRunArtifact(b.addExecutable(.{ .name = "flashnext-select", .root_module = select_check }));
    if (b.option([]const u8, "flashnext_fixtures", "capture_sparse.sh's selection fixtures folder")) |dir| select_run.addArgs(&.{dir});
    b.step("flashnext-select", "fn_qsa_select against Triton's _select / _select_tiles lists on captured score rows").dependOn(&select_run.step);
}
