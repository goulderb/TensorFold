//! A tensor-parallel group over NCCL (NCCL_COMM_ID; rows in rank order); two ranks add RoCE small gathers (roce.zig).
const std = @import("std");
const nccl = @import("nccl.zig");
const abi = @import("abi.zig");
const driver = @import("driver.zig");
const stream_ = @import("stream.zig");
const memory = @import("memory.zig");
const roce = @import("roce.zig");

const Stream = stream_.Stream;

pub const Group = struct {
    lib: nccl.Library,
    borrowed_lib: ?*const nccl.Library = null,
    comm: nccl.Comm,
    rank: usize,
    world: usize,
    fast: ?*roce.Gather = null, // two ranks' small all-gathers by RoCE writes (openRoce), when both ranks brought it up

    /// Rank `rank` of `world` (NCCL opened at run time; NCCL_COMM_ID names rank 0's address, every rank the same).
    pub fn open(rank: usize, world: usize) !Group {
        if (std.c.getenv("NCCL_COMM_ID") == null) return error.NoCommId;
        var lib = try nccl.Library.open();
        errdefer lib.close();
        var id: nccl.UniqueId = undefined;
        try lib.check(lib.api.ncclGetUniqueId(&id), "ncclGetUniqueId");
        var comm: nccl.Comm = null;
        try lib.check(lib.api.ncclCommInitRank(&comm, @intCast(world), id, @intCast(rank)), "ncclCommInitRank");
        return .{ .lib = lib, .comm = comm, .rank = rank, .world = world };
    }

    /// Borrow an engine-owned communicator without changing its library or communicator lifetime.
    pub fn borrow(lib: *const nccl.Library, comm: nccl.Comm, rank: usize, world: usize) !Group {
        if (world == 0 or rank >= world or comm == null) return error.BadWorld;
        return .{ .lib = undefined, .borrowed_lib = lib, .comm = comm, .rank = rank, .world = world };
    }

    fn library(g: *const Group) *const nccl.Library {
        return g.borrowed_lib orelse &g.lib;
    }

    /// Release only RoCE resources. A borrowing owner must call this after its streams drain and before NCCL teardown.
    pub fn closeFast(g: *Group) void {
        if (comptime @import("builtin").os.tag == .linux) {
            if (g.fast) |f| f.deinit();
        }
        g.fast = null;
    }

    pub fn close(g: *Group) void {
        g.closeFast();
        if (g.borrowed_lib == null) {
            _ = g.lib.api.ncclCommDestroy(g.comm);
            g.lib.close();
        }
        g.* = undefined;
    }

    /// Two ranks: gathers up to `max_bytes` a rank go by RoCE writes if both ranks bring it up; else NCCL keeps them.
    pub fn openRoce(g: *Group, gpa: std.mem.Allocator, io: std.Io, d: *const driver.Driver, s: Stream, max_bytes: usize) !void {
        if (comptime @import("builtin").os.tag != .linux) return error.UnsupportedRocePlatform;
        if (g.world != 2 or g.fast != null) return;
        g.fast = try roce.Gather.init(gpa, io, d, g, s, roce.wanted(), max_bytes);
    }

    /// A failed RoCE gather's error (its rows hold nothing valid): call it after a synchronize that reads a gather.
    pub fn check(g: *const Group) !void {
        if (g.fast) |f| try f.check();
    }

    /// `bytes` at `mine` from every rank into `all` in rank order (world * bytes there), on `s`.
    pub fn allGather(g: *const Group, s: Stream, mine: u64, all: u64, bytes: usize) !void {
        return g.allGatherRaw(s.handle, mine, all, bytes);
    }

    pub fn allGatherRaw(g: *const Group, stream: abi.Stream, mine: u64, all: u64, bytes: usize) !void {
        if (g.fast) |f| if (f.fits(bytes)) return f.gather(stream, mine, all, bytes);
        const lib = g.library();
        try lib.check(lib.api.ncclAllGather(mine, all, bytes, .u8, g.comm, stream), "ncclAllGather");
    }

    /// `bytes` at `ptr` to rank `peer`, on `s` (its matching recv takes the same count).
    pub fn send(g: *const Group, s: Stream, ptr: u64, bytes: usize, peer: usize) !void {
        const lib = g.library();
        try lib.check(lib.api.ncclSend(ptr, bytes, .u8, @intCast(peer), g.comm, s.handle), "ncclSend");
    }

    /// `bytes` from rank `peer` into `ptr`, on `s`.
    pub fn recv(g: *const Group, s: Stream, ptr: u64, bytes: usize, peer: usize) !void {
        const lib = g.library();
        try lib.check(lib.api.ncclRecv(ptr, bytes, .u8, @intCast(peer), g.comm, s.handle), "ncclRecv");
    }

    /// Rank 0 and rank `peer` connect both ways with one small exchange (`ptr`: 128 bytes, 64 out and 64 in).
    pub fn connect(g: *const Group, s: Stream, ptr: u64, peer: usize) !void {
        const lib = g.library();
        try lib.check(lib.api.ncclGroupStart(), "ncclGroupStart");
        const posted: anyerror!void = post: {
            g.send(s, ptr, 64, peer) catch |err| break :post err;
            g.recv(s, ptr + 64, 64, peer) catch |err| break :post err;
            break :post {};
        };
        try lib.check(lib.api.ncclGroupEnd(), "ncclGroupEnd"); // the group closes on every path
        try posted;
        try s.synchronize();
    }

    /// Every rank's `mine` in rank order into `out` (world of them), waited for.
    pub fn gather(g: *const Group, comptime T: type, d: *const driver.Driver, s: Stream, mine: T, out: []T) !void {
        if (out.len != g.world) return error.BadWorld;
        const size = @sizeOf(T);
        var dev = try memory.DeviceBuffer.alloc(d, size * g.world);
        defer dev.free();
        try dev.uploadAsync(g.rank * size, std.mem.asBytes(&mine), s.handle);
        try g.allGather(s, dev.ptr + g.rank * size, dev.ptr, size);
        try s.synchronize();
        try g.check();
        try dev.download(0, std.mem.sliceAsBytes(out));
    }
};

test "closing a borrowed group never destroys its owner's communicator or library" {
    const Fake = struct {
        var destroys: usize = 0;
        fn destroy(_: nccl.Comm) callconv(.c) nccl.Result {
            destroys += 1;
            return 0;
        }
    };
    Fake.destroys = 0;
    var lib: nccl.Library = .{ .lib = undefined, .api = undefined };
    lib.api.ncclCommDestroy = Fake.destroy;
    var group = try Group.borrow(&lib, @ptrFromInt(1), 0, 2);
    group.close();
    try std.testing.expectEqual(@as(usize, 0), Fake.destroys);
}
