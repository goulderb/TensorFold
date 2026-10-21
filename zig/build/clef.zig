//! tfclef: native Clef Flash decisions (zig/src/families/clef) as a C library, and its host checks.
const std = @import("std");
const catalog = @import("cuda_kernels.zig");
const cuda_build = @import("cuda.zig");

pub const Root = enum { family, library };

/// The catalog images the library loads: the BF16 backbone's bindings and the head's vit and lane kernels.
const shared_images = [_][]const u8{ "qwen35", "qwen35_hf", "qmmf", "modelopt_pack", "vit" };
/// Named by the shared projection dispatch's compile closure but never loaded on the BF16 path: embedded empty.
const unreachable_images = [_][]const u8{"exl3_candidate"};
/// clef_head.cu with the glue kernels' IEEE flags, as vit.cu builds.
pub const clef_head: catalog.Kernel = .{ .name = "clef_head", .flags = catalog.glue };

pub const ImageSource = union(enum) {
    nvcc: struct { tool: []const u8, version: std.Build.LazyPath, sms: []const u8 },
    /// A directory of <name>.fatbin files.
    prebuilt: []const u8,
};

fn image(b: *std.Build, source: ImageSource, k: catalog.Kernel) std.Build.LazyPath {
    return switch (source) {
        .prebuilt => |dir| b.graph.cwdRelativePath(b.pathJoin(&.{ dir, b.fmt("{s}.fatbin", .{k.name}) })),
        .nvcc => |n| cuda_build.fatbin(b, n.tool, n.version, k, n.sms),
    };
}

fn kernel(name: []const u8) catalog.Kernel {
    for (catalog.kernels) |k| if (std.mem.eql(u8, k.name, name)) return k;
    @panic("not a catalog kernel");
}

/// The CUDA runtime holding only the library's images; referencing any other image fails to compile.
fn runtime(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, source: ImageSource) *std.Build.Module {
    const options = b.addOptions();
    options.addOption(bool, "with_kernels", true);
    const cuda = b.createModule(.{ .root_source_file = b.path("zig/src/cuda/root.zig"), .target = target, .optimize = optimize, .link_libc = true });
    cuda.addOptions("kernel_options", options);
    cuda.addImport("stagger", b.createModule(.{ .root_source_file = b.path("zig/src/core/stagger.zig"), .target = target, .optimize = optimize }));
    for (shared_images) |name| cuda.addAnonymousImport(b.fmt("fatbin_{s}", .{name}), .{ .root_source_file = image(b, source, kernel(name)) });
    for (unreachable_images) |name| cuda.addAnonymousImport(b.fmt("fatbin_{s}", .{name}), .{ .root_source_file = b.addWriteFiles().add(b.fmt("{s}-unreachable.fatbin", .{name}), "") });
    return cuda;
}

/// `zig build tfclef -Dnvcc=... -Dsm=89` (or -Dfatbins=DIR): the library with its kernel images, in zig-out/native/lib.
pub fn library(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, draft_ids: *std.Build.Module, source: ?ImageSource) void {
    const step = b.step("tfclef", "tfclef: native Clef decisions as a C library with its kernel images (-Dnvcc or -Dfatbins)");
    const commit = b.option([]const u8, "clef-source", "The TensorFold commit tfclef reports in tf_clef_info") orelse "unrecorded";
    const images = source orelse {
        step.dependOn(&b.addFail("tfclef needs -Dnvcc (with -Dsm) or -Dfatbins").step);
        return;
    };
    const cuda = runtime(b, target, optimize, images);
    const mods = cuda_build.family(b, target, optimize, cuda, draft_ids);
    const lib = b.addLibrary(.{ .name = "tfclef", .linkage = .dynamic, .root_module = module(b, target, optimize, cuda, mods.core, mods.qwen35, image(b, images, clef_head), commit, .library) });
    step.dependOn(&b.addInstallArtifact(lib, .{ .dest_dir = .{ .override = .{ .custom = "native/lib" } } }).step);
}

/// The family module, or the library root over it; `head_image` is clef_head.cu's fatbin, or null for host checks
/// (an empty image); `source` is the TensorFold commit the library reports.
pub fn module(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, core: *std.Build.Module, qwen35: *std.Build.Module, head_image: ?std.Build.LazyPath, source: []const u8, root: Root) *std.Build.Module {
    const m = b.createModule(.{
        .root_source_file = b.path(switch (root) {
            .family => "zig/src/families/clef/root.zig",
            .library => "zig/src/families/clef/lib.zig",
        }),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = core }, .{ .name = "qwen35", .module = qwen35 } },
    });
    m.addAnonymousImport("fatbin_clef_head", .{ .root_source_file = head_image orelse b.addWriteFiles().add("clef_head-host-only.fatbin", "") });
    const options = b.addOptions();
    options.addOption([]const u8, "source_commit", source);
    m.addOptions("clef_build", options);
    return m;
}

/// `test-clef-host`: the family's host tests and the C header against the export layer, no device.
pub fn hostTests(b: *std.Build, cuda: *std.Build.Module, core: *std.Build.Module, qwen35: *std.Build.Module, all: *std.Build.Step) void {
    const host = b.graph.host;
    const step = b.step("test-clef-host", "Clef head checkpoint, record, budget and C ABI host tests without a GPU");
    const family = b.addRunArtifact(b.addTest(.{ .root_module = module(b, host, .debug, cuda, core, qwen35, null, "host-test", .family) }));
    family.has_side_effects = true;
    step.dependOn(&family.step);
    const c_abi = b.createModule(.{ .root_source_file = b.path("zig/src/families/clef/abi_c_test.zig"), .target = host, .optimize = .debug, .link_libc = true });
    c_abi.addIncludePath(b.path("include"));
    c_abi.addCSourceFile(.{ .file = b.path("zig/tests/clef/tfclef_abi_test.c"), .flags = &.{ "-std=c11", "-Wall", "-Wextra", "-Werror" } });
    const header = b.addRunArtifact(b.addTest(.{ .root_module = c_abi }));
    header.has_side_effects = true;
    step.dependOn(&header.step);
    all.dependOn(step);
    const compile = b.addLibrary(.{ .name = "tfclef_host_compile_only", .linkage = .dynamic, .root_module = module(b, host, .debug, cuda, core, qwen35, null, "host-compile", .library) });
    const compile_step = b.step("compile-clef-library", "Compile the complete tfclef library with empty kernel images, without nvcc or a device");
    compile_step.dependOn(&compile.step);
    all.dependOn(compile_step);
}
