//! tf-cuda-test kernel-contract: the registered CUDA kernels through gates 1-4, the planted controls, and gate 5.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const options = @import("contract_options");
const check = @import("check.zig");
const rows_k = @import("contract_rows.zig");
const router_k = @import("contract_router.zig");
const mm_k = @import("contract_mm.zig");
const rank_k = @import("contract_rank.zig");

const contract = core.contract;
const g = contract.gates;
const limits = contract.limits;
const registry = contract.registry;
const kk = cuda.kernels;

const controls = struct {
    pub const bytes align(16) = @embedFile("fatbin_contract_controls").*;
};

/// The embedded image a registration names (the controls' only when this binary was built with -Dnvcc).
fn image(name: []const u8) []const u8 {
    inline for (.{ "norms", "nemotron_route", "qmm_group", "lane_gemv", "prompt16", "qmmf", "rank_sum", "prefill_attention", "vit", "glm_glue", "nvfp4_experts", "exlr8_dense", "torch_pointwise", "dflash_qmm" }) |n| if (std.mem.eql(u8, name, n)) return @field(kk, n);
    return if (options.controls) &controls.bytes else &.{};
}

/// Gates 1-4 on every registered CUDA kernel and each control at its own gate, then gate 5 on this GPU.
pub fn run(gpu: check.Gpu) !void {
    const a = gpu.gpa;
    if (!kk.available) return error.BuiltWithoutKernels;
    if (!options.controls) {
        std.debug.print("FAIL kernel-contract: the controls are built only with -Dnvcc; without them no gate is proven\n", .{});
        return error.TestFailed;
    }
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var norms = try cuda.norms.Norms.load(gpu.d);
    defer norms.unload();
    var mods: [6]cuda.Module = undefined;
    var loaded: usize = 0;
    defer for (mods[0..loaded]) |*m| m.unload();
    for ([_][]const u8{ kk.nemotron_route, kk.qmm_group, kk.qmm_prefill, kk.lane_gemv, kk.affine4_pack, &controls.bytes }, 0..) |img, i| {
        mods[i] = try cuda.Module.load(gpu.d, img);
        loaded += 1;
    }
    const cap = try gpu.ctx.capability();
    const sms: usize = @intCast(try gpu.ctx.attribute(.multiprocessor_count));
    var affine4 = try cuda.qlinear.Affine4Kernels.resolve(gpu.d, mods[1], mods[2], mods[3], mods[4], sms, cap == 121);
    defer affine4.deinit();
    var split = try cuda.qlinear.Split.init(gpu.d); // GB10 serves without it; other GPUs take it for K slices
    defer split.deinit();
    var prompt = try cuda.nvfp4.Prompt.load(gpu.d);
    defer prompt.unload();
    var lane = try cuda.qmmf.Lane.load(gpu.d, cap / 10); // sm_90 up: K slices meet in a cluster
    defer lane.unload();
    var lane_reduce = try cuda.qmmf.Lane.load(gpu.d, 8); // no clusters: the slices meet in reduce_kernel
    defer lane_reduce.unload();
    var rank_sum = try cuda.rank_scatter.RankScatter.load(gpu.d);
    defer rank_sum.unload();
    var kernels: std.ArrayList(g.Kernel) = .empty;
    defer kernels.deinit(a);
    var row_list: [7]rows_k.Row = undefined;
    var nrows: usize = 0;
    defer for (row_list[0..nrows]) |*r| r.deinit();
    for (std.enums.values(rows_k.Which)) |w| {
        row_list[nrows] = try rows_k.Row.init(gpu, s, &norms, mods[5], w);
        nrows += 1;
    }
    for (row_list[0..nrows]) |*r| try kernels.append(a, r.kernel());
    var logits = router_k.Logits{ .rig = .{ .gpu = gpu, .s = s }, .f = try mods[0].function("tf_nemo_router") };
    defer logits.deinit();
    var topk = router_k.TopK{ .rig = .{ .gpu = gpu, .s = s }, .f = try mods[0].function("tf_nemo_topk") };
    defer topk.deinit();
    try kernels.append(a, logits.kernel());
    try kernels.append(a, topk.kernel());
    const kinds = comptime std.enums.values(mm_k.Kind);
    var mm: [kinds.len]mm_k.Mm = undefined;
    for (kinds, &mm) |kind, *m| {
        const by_lane: ?*const cuda.qmmf.Lane = switch (kind) {
            .lane_f32 => &lane,
            .lane_f32_reduce => &lane_reduce,
            else => null,
        };
        m.* = .{ .rig = .{ .gpu = gpu, .s = s }, .kind = kind, .affine4 = &affine4, .split = &split, .prompt = &prompt, .lane = by_lane };
    }
    defer for (&mm) |*m| m.deinit();
    for (&mm) |*m| try kernels.append(a, m.kernel());
    var rank = rank_k.Rank{ .rig = .{ .gpu = gpu, .s = s }, .sum = &rank_sum };
    defer rank.deinit();
    try kernels.append(a, rank.kernel());
    var t: g.Tally = .{};
    const t0 = check.now(gpu.io);
    try g.runAll(a, kernels.items, 0x6b63, &t);
    if (cap == 121) {
        try @import("contract_glm.zig").run(gpu, s, &affine4, &t);
    } else std.debug.print("N/A GLM FP4-Local contracts: registered for sm_121 only, not sm_{d}\n", .{cap});
    const t1 = check.now(gpu.io);
    const lim = try live(gpu);
    const t2 = check.now(gpu.io);
    std.debug.print("SUMMARY {d} kernels: {d} pass gates 1-4, {d} fail, {d} known findings; {d} controls: {d} red as planted, {d} missed; {d} limit problems; gates {d} ms, limits {d} ms\n", .{ t.kernels, t.passed, t.failed, t.known, t.controls, t.caught, t.missed, lim, @divTrunc(t1 - t0, std.time.ns_per_ms), @divTrunc(t2 - t1, std.time.ns_per_ms) });
    try check.expect(t.problems() + lim == 0, "kernel-contract: {d} gate failures, {d} controls missed, {d} limit problems", .{ t.failed, t.missed, lim });
}

/// Gate 5 on this GPU: the table's row for it against the driver, each record against the loaded function.
fn live(gpu: check.Gpu) !usize {
    var n = try registry.checkCuda(gpu.gpa, &registry.cuda, false);
    std.debug.print("{s} limits: {d} registered CUDA launches against sm_80/86/89/120/121, {d} problems\n", .{ if (n == 0) "PASS" else "FAIL", registry.cuda.len, n });
    const sm: u32 = @intCast(try gpu.ctx.capability());
    const at = for (limits.sms, 0..) |t, i| {
        if (t.sm == sm) break i;
    } else {
        std.debug.print("N/A  limits live: sm_{d} is not a target the records cover\n", .{sm});
        return n;
    };
    const t = limits.sms[at];
    const dev = [_]struct { []const u8, cuda.abi.DeviceAttribute, u32 }{
        .{ "most shared bytes a block (opt-in)", .max_shared_memory_per_block_optin, t.smem_optin },
        .{ "threads a block", .max_threads_per_block, t.threads },
    };
    for (dev) |d| {
        const got: u32 = @intCast(try gpu.ctx.attribute(d[1]));
        if (got != d[2]) n += 1;
        std.debug.print("{s} limits live sm_{d} {s}: the driver says {d}, the table {d}\n", .{ if (got == d[2]) "PASS" else "FAIL", sm, d[0], got, d[2] });
    }
    for (registry.cuda ++ registry.cuda_controls) |r| {
        if (!r.applies(sm)) continue;
        const control = std.mem.startsWith(u8, r.launch.name, "control");
        var mod = try cuda.Module.load(gpu.d, image(r.launch.image));
        defer mod.unload();
        const sym = try gpu.gpa.dupeSentinel(u8, r.launch.symbol, 0);
        defer gpu.gpa.free(sym);
        const f = try mod.function(sym);
        const regs: u32 = @intCast(try f.attribute(.num_regs));
        const static: u32 = @intCast(try f.attribute(.shared_size_bytes));
        const most: u32 = @intCast(try f.attribute(.max_threads_per_block));
        const fits = r.launch.threads <= most and static + r.launch.dynamic <= t.smem_optin;
        const current = r.usage[at].regs == regs and r.usage[at].static == static;
        if (!control and !(fits and current)) n += 1;
        if (control and fits) n += 1;
        const word = if (control) (if (fits) "MISSED" else "RED ") else if (fits and current) "PASS" else "FAIL";
        std.debug.print("{s} limits live {s}: {d} threads launched, the driver allows {d}; {d} registers, {d} + {d} shared bytes (recorded {d}, {d})\n", .{ word, r.launch.name, r.launch.threads, most, regs, static, r.launch.dynamic, r.usage[at].regs, r.usage[at].static });
    }
    return n;
}

/// Compatibility wrapper; the CLI enters the offline checker before opening the CUDA driver.
pub fn fatbins(gpu: check.Gpu, dir: []const u8) !void {
    const path = try gpu.gpa.dupeZ(u8, dir);
    defer gpu.gpa.free(path);
    try @import("contract_limits.zig").run(gpu.gpa, gpu.io, &.{path});
}
