//! One device's primary context, the one CUDA's runtime and PyTorch share, made current on the calling thread.

const std = @import("std");
const abi = @import("abi.zig");
const Driver = @import("driver.zig").Driver;
const Error = @import("driver.zig").Error;

pub const Context = struct {
    d: *const Driver,
    device: abi.Device,
    handle: abi.Context,
    /// The ordinal this context was opened on, so a caller can tell which device it holds.
    device_ordinal: c_int = 0,

    /// How the host waits for the device (CU_CTX_SCHED_*); host scheduling only, no launch or value changes.
    pub const Schedule = enum(c_uint) { auto = 0, spin = 1, yield = 2, blocking = 4 };

    /// The device's primary context's schedule, before any `init` of it (the driver ignores it on an active context).
    pub fn setSchedule(d: *const Driver, ordinal: c_int, s: Schedule) Error!void {
        var dev: abi.Device = 0;
        try d.check(d.api.cuDeviceGet(&dev, ordinal), "cuDeviceGet");
        try d.check(d.api.cuDevicePrimaryCtxSetFlags_v2(dev, @backingInt(s)), "cuDevicePrimaryCtxSetFlags");
    }

    pub fn init(d: *const Driver, ordinal: c_int) Error!Context {
        var dev: abi.Device = 0;
        try d.check(d.api.cuDeviceGet(&dev, ordinal), "cuDeviceGet");
        var ctx: abi.Context = null;
        try d.check(d.api.cuDevicePrimaryCtxRetain(&ctx, dev), "cuDevicePrimaryCtxRetain");
        errdefer _ = d.api.cuDevicePrimaryCtxRelease_v2(dev);
        try d.check(d.api.cuCtxSetCurrent(ctx), "cuCtxSetCurrent");
        @import("copy_source.zig").setCurrent(ctx);
        return .{ .d = d, .device = dev, .handle = ctx, .device_ordinal = ordinal };
    }

    /// For a library in a host's process: the current device's primary context (device 0's if none), or error.Invalid.
    pub fn shared(d: *const Driver) Error!Context {
        var current: abi.Context = null;
        try d.check(d.api.cuCtxGetCurrent(&current), "cuCtxGetCurrent");
        if (current == null) return init(d, 0);
        var dev: abi.Device = 0;
        try d.check(d.api.cuCtxGetDevice(&dev), "cuCtxGetDevice");
        var ctx: abi.Context = null;
        try d.check(d.api.cuDevicePrimaryCtxRetain(&ctx, dev), "cuDevicePrimaryCtxRetain");
        if (ctx != current) {
            _ = d.api.cuDevicePrimaryCtxRelease_v2(dev);
            return error.Invalid;
        }
        @import("copy_source.zig").setCurrent(ctx); // the host made it current: copy sources are tagged with it
        return .{ .d = d, .device = dev, .handle = ctx, .device_ordinal = dev };
    }

    /// Waits for all work, then drops this retain; resources made in it must be released first.
    /// A failed bind keeps this retain and its spare sources so teardown can be retried safely.
    pub fn deinit(self: *Context) void {
        self.makeCurrent() catch return;
        self.d.synchronize() catch {};
        @import("copy_source.zig").dropSpares(self.d, self.handle); // their pinned bytes go with the context
        _ = self.d.api.cuDevicePrimaryCtxRelease_v2(self.device);
        self.* = undefined;
    }

    /// Makes the context current on another thread before it calls the driver.
    pub fn makeCurrent(self: *const Context) Error!void {
        try self.d.check(self.d.api.cuCtxSetCurrent(self.handle), "cuCtxSetCurrent");
        @import("copy_source.zig").setCurrent(self.handle);
    }

    pub fn synchronize(self: *const Context) Error!void {
        try self.d.synchronize();
    }

    pub fn attribute(self: *const Context, a: abi.DeviceAttribute) Error!c_int {
        var v: c_int = 0;
        try self.d.check(self.d.api.cuDeviceGetAttribute(&v, a, self.device), "cuDeviceGetAttribute");
        return v;
    }

    /// Compute capability as 10 * major + minor (GB10: 121).
    pub fn capability(self: *const Context) Error!u32 {
        const major = try self.attribute(.compute_capability_major);
        const minor = try self.attribute(.compute_capability_minor);
        return @intCast(10 * major + minor);
    }

    pub fn name(self: *const Context, buf: []u8) Error![]const u8 {
        if (buf.len < 2) return error.Invalid;
        try self.d.check(self.d.api.cuDeviceGetName(buf.ptr, @intCast(buf.len), self.device), "cuDeviceGetName");
        return std.mem.sliceTo(buf, 0);
    }

    pub const MemInfo = struct { free: usize, total: usize };

    pub fn memInfo(self: *const Context) Error!MemInfo {
        // cuMemGetInfo reports the current context's device, so bind this one first: a caller asking about a device
        // other than the current one would otherwise read the wrong card's numbers.
        try self.makeCurrent();
        var m: MemInfo = .{ .free = 0, .total = 0 };
        try self.d.check(self.d.api.cuMemGetInfo_v2(&m.free, &m.total), "cuMemGetInfo");
        return m;
    }
};

test "context teardown binds its own device before synchronization and release" {
    const Fake = struct {
        var current: abi.Context = null;
        var synced: abi.Context = null;
        var released: abi.Device = -1;
        fn bind(ctx: abi.Context) callconv(.c) abi.Result {
            current = ctx;
            return abi.success;
        }
        fn sync() callconv(.c) abi.Result {
            synced = current;
            return abi.success;
        }
        fn release(dev: abi.Device) callconv(.c) abi.Result {
            released = dev;
            return abi.success;
        }
    };
    var d: Driver = .{ .lib = undefined, .api = undefined };
    d.api.cuCtxSetCurrent = Fake.bind;
    d.api.cuCtxSynchronize = Fake.sync;
    d.api.cuDevicePrimaryCtxRelease_v2 = Fake.release;
    const source = @import("copy_source.zig");
    defer source.setCurrent(null);
    Fake.current = @ptrFromInt(1);
    Fake.synced = null;
    Fake.released = -1;
    var ctx: Context = .{ .d = &d, .device = 3, .handle = @ptrFromInt(2), .device_ordinal = 3 };
    ctx.deinit();
    try std.testing.expectEqual(@as(abi.Context, @ptrFromInt(2)), Fake.synced);
    try std.testing.expectEqual(@as(abi.Device, 3), Fake.released);
}
