const std = @import("std");
const cuda = @import("cuda");

pub const Memory = struct {
    var mutex: std.atomic.Mutex = .unlocked;
    var blocks: [128]?[]align(256) u8 = @splat(null);

    pub fn driver() cuda.Driver {
        var d: cuda.Driver = .{ .lib = undefined, .api = undefined };
        d.api.cuMemAlloc_v2 = alloc;
        d.api.cuMemFree_v2 = free;
        d.api.cuMemcpyDtoDAsync_v2 = copy;
        d.api.cuMemcpy2DAsync_v2 = copy2D;
        d.api.cuMemcpyDtoH_v2 = download;
        d.api.cuMemcpyHtoD_v2 = uploadSync;
        d.api.cuMemcpyHtoDAsync_v2 = upload;
        d.api.cuMemHostAlloc = hostAlloc;
        d.api.cuMemFreeHost = hostFree;
        d.api.cuEventCreate = eventCreate;
        d.api.cuEventDestroy_v2 = eventDone;
        d.api.cuEventRecord = eventRecord;
        d.api.cuEventQuery = eventDone;
        d.api.cuEventSynchronize = eventDone;
        d.api.cuCtxSynchronize = contextSync;
        d.api.cuCtxSetCurrent = setCurrent;
        d.api.cuMemsetD8_v2 = fill8;
        d.api.cuMemsetD8Async = fill8Async;
        d.api.cuMemsetD32Async = fill32;
        d.api.cuStreamSynchronize = sync;
        return d;
    }

    pub fn slice(ptr: u64, n: usize) []u8 {
        const p: [*]u8 = @ptrFromInt(ptr);
        return p[0..n];
    }

    pub fn empty() bool {
        for (blocks) |b| if (b != null) return false;
        return true;
    }

    fn copy2D(c: *const cuda.abi.Memcpy2D, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        std.debug.assert(c.src_type == 2 and c.dst_type == 2 and c.src_array == null and c.dst_array == null);
        for (0..c.height) |row| @memcpy(slice(c.dst + (c.dst_y + row) * c.dst_pitch + c.dst_x, c.width), slice(c.src + (c.src_y + row) * c.src_pitch + c.src_x, c.width));
        return 0;
    }

    fn alloc(ptr: *u64, n: usize) callconv(.c) cuda.abi.Result {
        while (!mutex.tryLock()) std.Thread.yield() catch {};
        defer mutex.unlock();
        for (&blocks) |*b| if (b.* == null) {
            b.* = std.testing.allocator.alignedAlloc(u8, .fromByteUnits(256), n) catch return 2;
            ptr.* = @intFromPtr(b.*.?.ptr);
            return 0;
        };
        return 2;
    }
    fn free(ptr: u64) callconv(.c) cuda.abi.Result {
        while (!mutex.tryLock()) std.Thread.yield() catch {};
        defer mutex.unlock();
        for (&blocks) |*b| if (b.*) |bytes| if (@intFromPtr(bytes.ptr) == ptr) {
            std.testing.allocator.free(bytes);
            b.* = null;
            return 0;
        };
        unreachable;
    }
    fn copy(dst: u64, src: u64, n: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        @memcpy(slice(dst, n), slice(src, n));
        return 0;
    }
    fn uploadSync(dst: u64, src: ?*const anyopaque, n: usize) callconv(.c) cuda.abi.Result {
        const from: [*]const u8 = @ptrCast(src.?);
        @memcpy(slice(dst, n), from[0..n]);
        return 0;
    }
    fn upload(dst: u64, src: ?*const anyopaque, n: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        return uploadSync(dst, src, n);
    }
    fn hostAlloc(ptr: *?*anyopaque, n: usize, _: c_uint) callconv(.c) cuda.abi.Result {
        var address: u64 = 0;
        const result = alloc(&address, n);
        if (result == 0) ptr.* = @ptrFromInt(address);
        return result;
    }
    fn hostFree(ptr: ?*anyopaque) callconv(.c) cuda.abi.Result {
        return free(@intFromPtr(ptr.?));
    }
    fn eventCreate(event: *cuda.abi.Event, _: c_uint) callconv(.c) cuda.abi.Result {
        event.* = @ptrFromInt(1);
        return 0;
    }
    fn eventDone(_: cuda.abi.Event) callconv(.c) cuda.abi.Result {
        return 0;
    }
    fn eventRecord(_: cuda.abi.Event, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        return 0;
    }
    fn contextSync() callconv(.c) cuda.abi.Result {
        return 0;
    }
    fn setCurrent(_: cuda.abi.Context) callconv(.c) cuda.abi.Result {
        return 0;
    }
    fn download(dst: ?*anyopaque, src: u64, n: usize) callconv(.c) cuda.abi.Result {
        const to: [*]u8 = @ptrCast(dst.?);
        @memcpy(to[0..n], slice(src, n));
        return 0;
    }
    fn fill8(dst: u64, value: u8, n: usize) callconv(.c) cuda.abi.Result {
        @memset(slice(dst, n), value);
        return 0;
    }
    fn fill8Async(dst: u64, value: u8, n: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        return fill8(dst, value, n);
    }
    fn fill32(dst: u64, value: c_uint, n: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        const p: [*]u32 = @ptrFromInt(dst);
        @memset(p[0..n], value);
        return 0;
    }
    fn sync(_: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        return 0;
    }
};
