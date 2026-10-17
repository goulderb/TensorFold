//! A shared pool of KV cache pages (CUDA virtual memory). Each sequence reserves address space for its whole context
//! but holds physical pages only for the rows it has written; the pages come from one pool every sequence on the
//! device shares, created whole at load so the memory budget sees them. Kernels address a sequence's planes exactly
//! as before: the reservation is contiguous, only its backing is sparse.

const std = @import("std");
const abi = @import("abi.zig");
const Driver = @import("driver.zig").Driver;

pub const Error = error{ KvPoolFull, KvPoolUnsupported } || @import("driver.zig").Error || std.mem.Allocator.Error;

pub const Pool = struct {
    gpa: std.mem.Allocator,
    d: *const Driver,
    device: abi.Device,
    gran: usize,
    /// Physical pages waiting for a reservation.
    free: std.ArrayList(abi.MemHandle) = .empty,
    /// Every page the pool created (released at deinit).
    all: std.ArrayList(abi.MemHandle) = .empty,
    /// Pages mapped into reservations now.
    in_use: usize = 0,

    /// `bytes` of pages (rounded up to whole pages) on `device`, created now; the device's context is current.
    pub fn init(gpa: std.mem.Allocator, d: *const Driver, device: abi.Device, size: usize) Error!*Pool {
        var supported: c_int = 0;
        try d.check(d.api.cuDeviceGetAttribute(&supported, .virtual_memory_management_supported, device), "cuDeviceGetAttribute");
        if (supported == 0) return error.KvPoolUnsupported;
        const p = try gpa.create(Pool);
        errdefer gpa.destroy(p);
        p.* = .{ .gpa = gpa, .d = d, .device = device, .gran = 0 };
        const prop = p.props();
        try d.check(d.api.cuMemGetAllocationGranularity(&p.gran, &prop, 0), "cuMemGetAllocationGranularity");
        const pages = (size + p.gran - 1) / p.gran;
        errdefer p.releaseAll();
        try p.all.ensureTotalCapacity(gpa, pages);
        try p.free.ensureTotalCapacity(gpa, pages);
        for (0..pages) |_| {
            var h: abi.MemHandle = 0;
            try d.check(d.api.cuMemCreate(&h, p.gran, &prop, 0), "cuMemCreate");
            p.all.appendAssumeCapacity(h);
            p.free.appendAssumeCapacity(h);
        }
        return p;
    }

    fn props(p: *const Pool) abi.MemAllocationProp {
        return .{ .type = 1, .location = .{ .type = 1, .id = p.device } }; // pinned, on this device
    }

    fn releaseAll(p: *Pool) void {
        for (p.all.items) |h| _ = p.d.api.cuMemRelease(h);
        p.all.deinit(p.gpa);
        p.free.deinit(p.gpa);
    }

    /// The pool's pages and those mapped now, in bytes.
    pub fn bytes(p: *const Pool) usize {
        return p.all.items.len * p.gran;
    }

    pub fn usedBytes(p: *const Pool) usize {
        return p.in_use * p.gran;
    }

    /// Every reservation must be freed first.
    pub fn deinit(p: *Pool) void {
        p.releaseAll();
        p.gpa.destroy(p);
    }

    /// Address space for `len` bytes, nothing mapped yet.
    pub fn reserve(p: *Pool, len: usize) Error!*Range {
        const size = std.mem.alignForward(usize, @max(len, 1), p.gran);
        const r = try p.gpa.create(Range);
        errdefer p.gpa.destroy(r);
        r.* = .{ .pool = p, .va = 0, .len = size, .pages = try p.gpa.alloc(abi.MemHandle, size / p.gran) };
        errdefer p.gpa.free(r.pages);
        @memset(r.pages, 0);
        try p.d.check(p.d.api.cuMemAddressReserve(&r.va, size, 0, 0, 0), "cuMemAddressReserve");
        return r;
    }
};

/// One reservation: its address space and the page mapped at each page of it (0: none).
pub const Range = struct {
    pool: *Pool,
    va: abi.DevicePtr,
    len: usize,
    pages: []abi.MemHandle,

    /// Bytes [off, off + n) backed by pages (mapped now where they were not). KvPoolFull when the pool runs out; the
    /// pages mapped before then stay.
    pub fn ensure(r: *Range, off: usize, n: usize) Error!void {
        if (n == 0) return;
        const p = r.pool;
        const first = off / p.gran;
        const last = @min((off + n + p.gran - 1) / p.gran, r.pages.len);
        const access: abi.MemAccessDesc = .{ .location = .{ .type = 1, .id = p.device }, .flags = 3 }; // read-write
        for (first..last) |i| {
            if (r.pages[i] != 0) continue;
            const h = p.free.pop() orelse return error.KvPoolFull;
            const at = r.va + i * p.gran;
            p.d.check(p.d.api.cuMemMap(at, p.gran, 0, h, 0), "cuMemMap") catch |e| {
                p.free.appendAssumeCapacity(h);
                return e;
            };
            p.d.check(p.d.api.cuMemSetAccess(at, p.gran, (&access)[0..1], 1), "cuMemSetAccess") catch |e| {
                _ = p.d.api.cuMemUnmap(at, p.gran);
                p.free.appendAssumeCapacity(h);
                return e;
            };
            r.pages[i] = h;
            p.in_use += 1;
        }
    }

    /// Every page back to the pool; the address space stays reserved. The caller's work on it must have finished.
    pub fn trim(r: *Range) void {
        const p = r.pool;
        for (r.pages, 0..) |*h, i| if (h.* != 0) {
            _ = p.d.api.cuMemUnmap(r.va + i * p.gran, p.gran);
            p.free.appendAssumeCapacity(h.*);
            h.* = 0;
            p.in_use -= 1;
        };
    }

    /// Pages mapped now.
    pub fn mapped(r: *const Range) usize {
        var n: usize = 0;
        for (r.pages) |h| n += @intFromBool(h != 0);
        return n;
    }

    pub fn free(r: *Range) void {
        r.trim();
        const p = r.pool;
        _ = p.d.api.cuMemAddressFree(r.va, r.len);
        p.gpa.free(r.pages);
        p.gpa.destroy(r);
    }
};
