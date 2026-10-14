//! The two-GPU layer split's foundation: two contexts, one per device, and a host-staged handoff between them.
//!
//! The split runs rank 0's layers on one device and rank 1's on the other, so the hidden state crosses devices once a
//! forward. This device pair is that crossing. It stages through pinned host memory rather than a peer copy: the X1
//! probe found `can_use_host_pointer_for_registered_mem` is 0 on this driver, so a registered region is addressed
//! through its device alias, and inter-GPU peer bandwidth is not qualified. ExLlamaV3 bounces the same way for the
//! same reason (util/device_copy.py).

const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");

const Failed = check.Failed;
const expect = check.expect;

/// Two devices, each with its own primary context, and one pinned staging buffer per direction.
pub const Pair = struct {
    d: *const cuda.Driver,
    gpa: std.mem.Allocator,
    io: std.Io,
    ctx: [2]cuda.Context,
    /// Staging for a handoff: rank 0 writes here, rank 1 reads it, and the reverse for the reply.
    stage: [2]cuda.HostBuffer,
    stage_len: usize,

    /// Opens `a` and `b` as the two ranks. Both contexts are made current on this thread in turn; the caller must
    /// not assume either is current after a call, since a handoff switches between them.
    pub fn open(d: *const cuda.Driver, gpa: std.mem.Allocator, io: std.Io, a: c_int, b: c_int, stage_len: usize) !Pair {
        var p: Pair = undefined;
        p.d = d;
        p.gpa = gpa;
        p.io = io;
        p.stage_len = stage_len;
        p.ctx[0] = try cuda.Context.init(d, a);
        errdefer p.ctx[0].deinit();
        p.ctx[1] = try cuda.Context.init(d, b);
        errdefer p.ctx[1].deinit();
        // each staging buffer is allocated under the context that writes it, so its device alias is that device's
        try p.ctx[0].makeCurrent();
        p.stage[0] = try cuda.HostBuffer.allocMapped(d, stage_len);
        errdefer p.stage[0].free();
        try p.ctx[1].makeCurrent();
        p.stage[1] = try cuda.HostBuffer.allocMapped(d, stage_len);
        errdefer p.stage[1].free();
        return p;
    }

    pub fn close(p: *Pair) void {
        _ = p.ctx[0].makeCurrent() catch {};
        p.stage[0].free();
        _ = p.ctx[1].makeCurrent() catch {};
        p.stage[1].free();
        p.ctx[1].deinit();
        p.ctx[0].deinit();
    }

    /// Copies `bytes` from `src` (a device pointer on rank `from`) to `dst` (a device pointer on rank `to`), staged
    /// through pinned host memory. `src` is downloaded under the source context, then uploaded under the target's.
    ///
    /// The source context is synchronized first: a kernel queued on the source's stream can still be writing `src`
    /// when the download runs, and the download would then move stale bytes.
    pub fn handoff(p: *Pair, from: usize, to: usize, src: u64, dst: u64, bytes: usize) !void {
        if (bytes > p.stage_len) return error.StageTooSmall;
        try p.ctx[from].makeCurrent();
        try p.ctx[from].synchronize();
        try p.d.check(p.d.api.cuMemcpyDtoH_v2(p.stage[from].bytes.ptr, src, bytes), "cuMemcpyDtoH");
        try p.ctx[to].makeCurrent();
        try p.d.check(p.d.api.cuMemcpyHtoD_v2(dst, p.stage[from].bytes.ptr, bytes), "cuMemcpyHtoD");
    }
};

/// The two devices a split should use: the first two with the same compute capability, which on this workstation is
/// the pair of internal cards. The eGPU enumerates first but is sm_89, so picking by ordinal would put a layer split
/// across a Thunderbolt link.
pub fn internalPair(d: *const cuda.Driver, gpa: std.mem.Allocator) !?[2]c_int {
    _ = gpa;
    const count = try d.deviceCount();
    var caps: [8]u32 = undefined;
    var n: usize = 0;
    var i: c_int = 0;
    // The capability comes from the device, not a context: opening one per device would leave the last current
    // after its release, and the caller's own binding would then be stale.
    while (i < count and n < caps.len) : (i += 1) {
        var dev: cuda.abi.Device = 0;
        try d.check(d.api.cuDeviceGet(&dev, i), "cuDeviceGet");
        var v: c_int = 0;
        try d.check(d.api.cuDeviceGetAttribute(&v, .compute_capability_major, dev), "cuDeviceGetAttribute");
        var minor: c_int = 0;
        try d.check(d.api.cuDeviceGetAttribute(&minor, .compute_capability_minor, dev), "cuDeviceGetAttribute");
        caps[n] = @intCast(10 * v + minor);
        n += 1;
    }
    for (0..n) |a| for (a + 1..n) |b| if (caps[a] == caps[b]) return .{ @intCast(a), @intCast(b) };
    return null;
}

/// `tf-cuda-test pair`: the device pair copies bytes both ways with no corruption.
pub fn pair(gpu: check.Gpu) !void {
    const count = try gpu.d.deviceCount();
    const pick = (try internalPair(gpu.d, gpu.gpa)) orelse {
        std.debug.print("SKIP pair: this machine has {d} CUDA device{s}, and no two share a compute capability\n", .{ count, if (count == 1) "" else "s" });
        return;
    };
    std.debug.print("pair: devices {d} and {d} (the same compute capability)\n", .{ pick[0], pick[1] });
    const n: usize = 1 << 20; // 1 MiB: a hidden-state handoff at the widest window is far smaller
    var p = try Pair.open(gpu.d, gpu.gpa, gpu.io, pick[0], pick[1], n);
    defer p.close();

    // A pattern the sender writes on device 0 and the receiver reads on device 1.
    const want = try gpu.gpa.alloc(u8, n);
    defer gpu.gpa.free(want);
    var seed: u32 = 0x9e3779b9;
    for (want) |*x| {
        seed = seed *% 1664525 +% 1013904223;
        x.* = @truncate(seed >> 24);
    }

    try p.ctx[0].makeCurrent();
    const a = try cuda.DeviceBuffer.fromHost(gpu.d, want);
    defer {
        var x = a;
        x.free();
    }
    try p.ctx[1].makeCurrent();
    const b = try cuda.DeviceBuffer.alloc(gpu.d, n);
    defer {
        var x = b;
        x.free();
    }

    try p.handoff(0, 1, a.ptr, b.ptr, n);
    try p.ctx[1].makeCurrent();
    const got = try gpu.gpa.alloc(u8, n);
    defer gpu.gpa.free(got);
    try b.download(0, got);
    try check.sameBytes("pair 0 -> 1", got, want);
    check.pass("pair 0 -> 1: {d} bytes staged through host memory, 0 wrong", .{n});

    // And back, so a reply can cross too. The destination belongs to rank 0, the receiving side.
    try p.ctx[0].makeCurrent();
    const c = try cuda.DeviceBuffer.alloc(gpu.d, n);
    defer {
        var x = c;
        x.free();
    }
    try c.fill8(0, null);
    try p.handoff(1, 0, b.ptr, c.ptr, n);
    try p.ctx[0].makeCurrent();
    @memset(got, 0);
    try c.download(0, got);
    try check.sameBytes("pair 1 -> 0", got, want);
    check.pass("pair 1 -> 0: {d} bytes staged through host memory, 0 wrong", .{n});

    // Each device reports its own free memory, which the per-rank admission will read.
    var name_buf: [128]u8 = undefined;
    for (0..2) |i| {
        const mi = try p.ctx[i].memInfo();
        const name = try p.ctx[i].name(&name_buf);
        check.pass("pair device {d}: {s}, {d:.1} GiB free of {d:.1} GiB", .{ i, name, gib(mi.free), gib(mi.total) });
    }
}

fn gib(bytes: usize) f64 {
    return @as(f64, @floatFromInt(bytes)) / (1 << 30);
}
