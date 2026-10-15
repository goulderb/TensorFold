const std = @import("std");
const cuda = @import("cuda");

pub const Memory = struct {
    var blocks: [128]?[]align(256) u8 = @splat(null);

    pub fn driver() cuda.Driver {
        var d: cuda.Driver = .{ .lib = undefined, .api = undefined };
        d.api.cuMemAlloc_v2 = alloc;
        d.api.cuMemFree_v2 = free;
        d.api.cuMemcpyDtoDAsync_v2 = copy;
        d.api.cuMemcpyDtoH_v2 = download;
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

    fn alloc(ptr: *u64, n: usize) callconv(.c) cuda.abi.Result {
        for (&blocks) |*b| if (b.* == null) {
            b.* = std.testing.allocator.alignedAlloc(u8, .fromByteUnits(256), n) catch return 2;
            ptr.* = @intFromPtr(b.*.?.ptr);
            return 0;
        };
        return 2;
    }
    fn free(ptr: u64) callconv(.c) cuda.abi.Result {
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
