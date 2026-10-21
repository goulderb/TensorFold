//! A bounded reader mailbox. The worker owns disk IO; its caller owns every payload consumer.
const std = @import("std");
const spill = @import("spill.zig");

pub const Read = struct {
    pub const chunk_bytes = 8 << 20;
    pub const Description = struct {
        header: spill.Header,
        head: [spill.header_bytes]u8,

        pub fn meta(d: *const Description) []const u8 {
            return spill.Header.meta(&d.head, d.header);
        }
    };
    pub const Chunk = struct { offset: u64, bytes: []const u8 };
    pub const Poll = union(enum) { pending, chunk: Chunk, done, failed: anyerror };
    const Terminal = union(enum) { running, done, failed: anyerror };

    tier: *spill.Tier,
    name: spill.Name,
    generation: u64,
    started: i96,
    mutex: std.Io.Mutex = .init,
    changed: std.Io.Condition = .init,
    thread: ?std.Thread = null,
    producer: []u8,
    mailbox: []u8,
    description_: ?Description = null,
    ready: ?Chunk = null,
    terminal: Terminal = .running,
    cancelled: bool = false,
    released: bool = false,

    pub fn start(tier: *spill.Tier, name: spill.Name) !*Read {
        const generation = try tier.readGeneration(name);
        const r = try tier.gpa.create(Read);
        errdefer tier.gpa.destroy(r);
        const n = @min(chunk_bytes, tier.chunk);
        if (n == 0) return error.EmptyReadBuffer;
        const producer = try tier.gpa.alloc(u8, n);
        errdefer tier.gpa.free(producer);
        const mailbox = try tier.gpa.alloc(u8, n);
        errdefer tier.gpa.free(mailbox);
        r.* = .{ .tier = tier, .name = name, .generation = generation, .started = std.Io.Clock.awake.now(tier.io).toNanoseconds(), .producer = producer, .mailbox = mailbox };
        r.thread = try std.Thread.spawn(.{}, run, .{r});
        return r;
    }

    pub fn description(r: *Read) ?Description {
        r.mutex.lockUncancelable(r.tier.io);
        defer r.mutex.unlock(r.tier.io);
        return r.description_;
    }

    /// The chunk stays immutable until consume; poll never waits for IO.
    pub fn poll(r: *Read) Poll {
        r.mutex.lockUncancelable(r.tier.io);
        defer r.mutex.unlock(r.tier.io);
        if (r.cancelled) return .{ .failed = error.ReadCancelled };
        if (r.ready) |c| return .{ .chunk = c };
        return switch (r.terminal) {
            .running => .pending,
            .done => .done,
            .failed => |err| .{ .failed = err },
        };
    }

    pub fn consume(r: *Read) void {
        r.mutex.lockUncancelable(r.tier.io);
        defer r.mutex.unlock(r.tier.io);
        r.ready = null;
        r.changed.broadcast(r.tier.io);
    }

    pub fn cancel(r: *Read) void {
        r.mutex.lockUncancelable(r.tier.io);
        defer r.mutex.unlock(r.tier.io);
        r.cancelled = true;
        r.changed.broadcast(r.tier.io);
    }

    /// The Tier retains the allocation until the worker exits. Never joins the worker here.
    pub fn release(r: *Read) void {
        r.mutex.lockUncancelable(r.tier.io);
        defer r.mutex.unlock(r.tier.io);
        r.released = true;
        r.cancelled = true;
        r.changed.broadcast(r.tier.io);
    }

    pub fn reapable(r: *Read) bool {
        r.mutex.lockUncancelable(r.tier.io);
        defer r.mutex.unlock(r.tier.io);
        return r.released and r.terminal != .running;
    }

    /// Owner-only invalidation, guarded against replacing an entry during the read.
    pub fn forgetFailed(r: *Read) void {
        r.mutex.lockUncancelable(r.tier.io);
        const failed = switch (r.terminal) {
            .failed => |err| err != error.ReadCancelled and err != error.NotStored,
            else => false,
        };
        r.mutex.unlock(r.tier.io);
        if (failed) r.tier.forgetGeneration(r.name, r.generation);
    }

    pub fn forgetInvalid(r: *Read) void {
        r.tier.forgetGeneration(r.name, r.generation);
    }

    /// Tier-only, after cancellation or reapable. No owner may use this handle afterwards.
    pub fn destroy(r: *Read) void {
        if (r.thread) |thread| thread.join();
        const gpa = r.tier.gpa;
        gpa.free(r.producer);
        gpa.free(r.mailbox);
        gpa.destroy(r);
    }

    fn run(r: *Read) void {
        const result = r.read();
        r.mutex.lockUncancelable(r.tier.io);
        defer r.mutex.unlock(r.tier.io);
        r.terminal = if (r.cancelled) .{ .failed = error.ReadCancelled } else if (result) |_| .done else |err| failed: {
            r.tier.readFailed(err, r.started);
            break :failed .{ .failed = err };
        };
        r.changed.broadcast(r.tier.io);
    }

    fn read(r: *Read) !void {
        var reading = try r.tier.readDeferred(r.name, r.generation);
        defer reading.close();
        r.mutex.lockUncancelable(r.tier.io);
        if (r.cancelled) {
            r.mutex.unlock(r.tier.io);
            return error.ReadCancelled;
        }
        r.description_ = .{ .header = reading.header, .head = reading.head };
        r.mutex.unlock(r.tier.io);
        try reading.pumpBuffered(.{ .ptr = r, .write = put }, r.producer);
    }

    fn put(ptr: *anyopaque, offset: u64, bytes: []const u8) !void {
        const r: *Read = @ptrCast(@alignCast(ptr));
        var off: usize = 0;
        while (off < bytes.len) {
            r.mutex.lockUncancelable(r.tier.io);
            defer r.mutex.unlock(r.tier.io);
            while (r.ready != null and !r.cancelled) r.changed.waitUncancelable(r.tier.io, &r.mutex);
            if (r.cancelled) return error.ReadCancelled;
            const n = @min(r.mailbox.len, bytes.len - off);
            @memcpy(r.mailbox[0..n], bytes[off..][0..n]);
            r.ready = .{ .offset = offset + off, .bytes = r.mailbox[0..n] };
            r.changed.broadcast(r.tier.io);
            while (r.ready != null and !r.cancelled) r.changed.waitUncancelable(r.tier.io, &r.mutex);
            if (r.cancelled) return error.ReadCancelled;
            off += n;
        }
    }
};
