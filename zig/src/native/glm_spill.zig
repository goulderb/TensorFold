const std = @import("std");
const builtin = @import("builtin");
const cuda = @import("cuda");
const glm = @import("glm_cuda");
const api = @import("engine_api");
const spill = api.prompt_cache.spill;
const identity = api.spill_identity;
const persistence = glm.engine.persistence;

const Artifact = struct { path: [:0]u8, before: identity.Meta };
const reserve_bytes = 192 << 20;

pub const Adapter = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    settings: spill.Settings,
    host_limit: u64,
    timing: @FieldType(api.prompt_cache.Rules, "spill_timing") = .eviction,
    artifacts: std.ArrayList(Artifact) = .empty,
    state: union(enum) { prepared, attached: struct { e: *glm.engine.Engine, link: *glm.follow.Link, tier: *spill.Tier } } = .prepared,
    mutex: std.Io.Mutex = .init,
    forgotten: [64]spill.Name = undefined,
    forgotten_len: usize = 0,
    queued: ?struct { command: persistence.Command, host: glm.engine.Engine.HostSnapshot, tokens: []u32 } = null,
    mode: enum { serving, closing } = .serving,

    pub fn prepare(gpa: std.mem.Allocator, io: std.Io, checkpoint: []const u8) !?*Adapter {
        const settings = try spill.Settings.fromEnv(gpa, spill.ProcessEnv{}) orelse return null;
        errdefer gpa.free(settings.dir);
        const raw = std.c.getenv("TENSORFOLD_SPILL_HOST_GIB") orelse return error.SpillHostGibRequired;
        const gib = std.fmt.parseFloat(f64, std.mem.span(raw)) catch return error.BadSpillHostGib;
        if (!std.math.isFinite(gib) or gib <= 0 or gib > 1024) return error.BadSpillHostGib;
        const host: u64 = @intFromFloat(gib * (1 << 30));
        if (host <= reserve_bytes) return error.BadSpillHostGib;
        const self = try gpa.create(Adapter);
        self.* = .{ .gpa = gpa, .io = io, .settings = settings, .host_limit = host - reserve_bytes };
        errdefer {
            self.freeArtifacts();
            gpa.destroy(self);
        }
        if (std.c.getenv("TF_GLM_SPILL_EAGER")) |raw_timing| {
            const v = std.mem.span(raw_timing);
            if (std.mem.eql(u8, v, "1")) self.timing = .retention else if (!std.mem.eql(u8, v, "0")) return error.BadSpillTiming;
        }
        var dir = try std.Io.Dir.cwd().openDir(io, checkpoint, .{ .iterate = true });
        defer dir.close(io);
        var it = dir.iterate();
        while (try it.next(io)) |entry| {
            if (!std.mem.endsWith(u8, entry.name, ".safetensors") and !std.mem.eql(u8, entry.name, "config.json")) continue;
            const path = try std.fmt.allocPrintSentinel(gpa, "{s}/{s}", .{ checkpoint, entry.name }, 0);
            errdefer gpa.free(path);
            try self.artifacts.append(gpa, .{ .path = path, .before = try metadata(path) });
        }
        if (self.artifacts.items.len < 2) return error.SpillWeightsMissing;
        std.mem.sort(Artifact, self.artifacts.items, {}, struct {
            fn less(_: void, a: Artifact, b: Artifact) bool {
                return std.mem.lessThan(u8, a.path, b.path);
            }
        }.less);
        return self;
    }

    pub fn deinit(a: *Adapter) void {
        closePort(a);
        a.freeArtifacts();
        a.gpa.free(a.settings.dir);
        a.gpa.destroy(a);
    }

    fn freeArtifacts(a: *Adapter) void {
        for (a.artifacts.items) |file| a.gpa.free(file.path);
        a.artifacts.deinit(a.gpa);
    }

    pub fn attach(ptr: *anyopaque, e: *glm.engine.Engine, link: *glm.follow.Link) !persistence.Port {
        const a: *Adapter = @ptrCast(@alignCast(ptr));
        var digest = std.crypto.hash.sha2.Sha256.init(.{});
        digest.update("glm-cuda-spill-v1");
        const root = try a.gpa.dupeSentinel(u8, a.settings.dir, 0);
        defer a.gpa.free(root);
        var hashes = try identity.Store.open(a.io, root);
        defer hashes.close();
        for (a.artifacts.items) |file| {
            if (!identity.sameMetadata(file.before, try metadata(file.path))) return error.SpillArtifactChanged;
            const hash = if (std.mem.endsWith(u8, file.path, ".safetensors")) (try hashes.digest(file.path, @splat(0))).digest else try glm.ranks.hashFile(a.gpa, a.io, file.path);
            if (!identity.sameMetadata(file.before, try metadata(file.path))) return error.SpillArtifactChanged;
            digest.update(std.fs.path.basename(file.path));
            digest.update(&hash);
        }
        digest.update(&try glm.ranks.hashFile(a.gpa, a.io, "/proc/self/exe"));
        digest.update(&e.spillShape().digest());
        e.settings.hashNumerics(&digest);
        const mine = digest.finalResult();
        const tier = try spill.Tier.open(a.gpa, a.io, a.settings, "glm-cuda", mine, @intCast(e.share.rank), @intCast(e.share.world));
        a.state = .{ .attached = .{ .e = e, .link = link, .tier = tier } };
        std.log.info("GLM spill enabled: cap {d} GiB, minimum {d} tokens, free floor {d} GiB, host capture {d} MiB", .{ a.settings.cap >> 30, a.settings.min_tokens, a.settings.min_free >> 30, a.host_limit >> 20 });
        return .{ .ctx = a, .tier = tier, .identity = mine, .apply = apply, .request = request, .flush = flush, .forget = forget, .close = closePort };
    }

    fn closePort(ptr: *anyopaque) void {
        const a: *Adapter = @ptrCast(@alignCast(ptr));
        if (a.state == .attached) {
            _ = a.state.attached.tier.drain(a.settings.flush_s);
            a.state.attached.tier.deinit();
            a.state = .prepared;
        }
        a.dropQueued();
    }

    fn dropQueued(a: *Adapter) void {
        if (a.queued) |q| {
            a.gpa.free(q.host.bytes);
            a.gpa.free(q.tokens);
            a.queued = null;
        }
    }

    pub fn finish(a: *Adapter, store: *api.prompt_cache.Store) void {
        a.mode = .closing;
        flush(a) catch {};
        _ = store.flushSpill(a.settings.flush_s);
        flush(a) catch {};
    }

    fn apply(ptr: *anyopaque, command: persistence.Command) !void {
        const a: *Adapter = @ptrCast(@alignCast(ptr));
        const s = a.state.attached;
        const name: spill.Name = .{ .key = command.key, .at = command.at };
        switch (command.operation) {
            .write => {
                if (s.tier.has(name)) return;
                if (command.tokens.len != @as(u64, command.at) + 1 or !spill.Name.of(command.tokens, command.at).eql(name)) return error.SpillTokens;
                var host = if (s.e.share.rank == 0) (a.queued orelse return error.NoQueuedSpill).host else try s.e.exportState(command.id, a.host_limit);
                defer if (s.e.share.rank != 0) a.gpa.free(host.bytes);
                const Source = struct {
                    fn read(p: *anyopaque, at: u64, out: []u8) !void {
                        const data: *[]u8 = @ptrCast(@alignCast(p));
                        if (at > data.len or out.len > data.len - at) return error.SpillPayloadLength;
                        @memcpy(out, data.*[@intCast(at)..][0..out.len]);
                    }
                };
                const job = try s.tier.write(name, command.tokens, host.meta[0..host.meta_len], host.bytes.len, .{ .ptr = @ptrCast(&host.bytes), .read = Source.read });
                defer job.release();
                job.waitCopied();
                if (job.done() and !job.ok) return error.SpillWriteFailed;
            },
            .read => {
                var reading = try s.tier.read(name);
                defer reading.close();
                try s.e.importState(command.id, command.at, reading.meta(), reading.payloadLen(), &reading, pump);
            },
            .forget => s.tier.forget(name),
        }
    }

    fn pump(ptr: *anyopaque, buffer: *cuda.DeviceBuffer) !void {
        const reading: *spill.Reading = @ptrCast(@alignCast(ptr));
        try reading.pump(.{ .ptr = buffer, .write = struct {
            fn write(p: *anyopaque, offset: u64, bytes: []const u8) !void {
                const buf: *cuda.DeviceBuffer = @ptrCast(@alignCast(p));
                try buf.upload(@intCast(offset), bytes);
            }
        }.write });
    }

    fn request(ptr: *anyopaque, command: persistence.Command) !void {
        const a: *Adapter = @ptrCast(@alignCast(ptr));
        const s = a.state.attached;
        const leader: glm.follow.Leader = .{ .e = s.e, .link = s.link };
        if (command.operation == .write) {
            if (a.queued != null) {
                s.tier.unwritten(.incomplete);
                return error.Incomplete;
            }
            const tokens = try a.gpa.dupe(u32, command.tokens);
            const host = s.e.exportState(command.id, a.host_limit) catch |err| {
                a.gpa.free(tokens);
                return err;
            };
            var owned = command;
            owned.tokens = tokens;
            a.queued = .{ .command = owned, .tokens = tokens, .host = host };
            if (a.mode == .closing) try flush(a);
            return;
        }
        try leader.flush();
        try leader.persist(command);
    }

    fn flush(ptr: *anyopaque) !void {
        const a: *Adapter = @ptrCast(@alignCast(ptr));
        const s = a.state.attached;
        const leader: glm.follow.Leader = .{ .e = s.e, .link = s.link };
        if (a.queued) |q| {
            defer a.dropQueued();
            leader.persist(q.command) catch |err| switch (err) {
                error.SpillRefused, error.NotStored => {},
                else => return err,
            };
        }
        var pending: [64]spill.Name = undefined;
        a.mutex.lockUncancelable(a.io);
        const n = a.forgotten_len;
        @memcpy(pending[0..n], a.forgotten[0..n]);
        a.forgotten_len = 0;
        a.mutex.unlock(a.io);
        for (pending[0..n]) |name| try leader.persist(.{ .operation = .forget, .key = name.key, .at = name.at });
    }

    fn forget(ptr: *anyopaque, key: u64, at: u32) void {
        const a: *Adapter = @ptrCast(@alignCast(ptr));
        a.mutex.lockUncancelable(a.io);
        defer a.mutex.unlock(a.io);
        if (a.forgotten_len == a.forgotten.len) {
            std.log.warn("GLM spill peer-forget queue full; peer file remains under its local disk cap", .{});
            return;
        }
        a.forgotten[a.forgotten_len] = .{ .key = key, .at = at };
        a.forgotten_len += 1;
    }
};

fn metadata(path: [:0]const u8) !identity.Meta {
    if (comptime builtin.os.tag == .windows) return metadataWindows(path);
    const fd = std.c.open(path, .{ .ACCMODE = .RDONLY, .CLOEXEC = true }, @as(std.c.mode_t, 0));
    if (fd < 0) return error.SpillArtifactOpen;
    defer _ = std.c.close(fd);
    return identity.metadata(fd);
}

const WinMeta = if (builtin.os.tag != .windows) struct {} else struct {
const ByHandleInfo = extern struct {
    attributes: u32,
    creation: [2]u32,
    access: [2]u32,
    write: [2]u32,
    volume_serial: u32,
    size_high: u32,
    size_low: u32,
    links: u32,
    index_high: u32,
    index_low: u32,
};
extern "kernel32" fn GetFileInformationByHandle(h: *anyopaque, info: *ByHandleInfo) callconv(.winapi) i32;

fn fileTime(ft: [2]u32) struct { s: i64, ns: u32 } {
    const ticks: u64 = @as(u64, ft[1]) << 32 | ft[0];
    const epoch: u64 = 116444736000000000;
    if (ticks < epoch) return .{ .s = 0, .ns = 0 };
    const unix_100ns = ticks - epoch;
    return .{ .s = @intCast(unix_100ns / 10_000_000), .ns = @intCast((unix_100ns % 10_000_000) * 100) };
}

extern "kernel32" fn CreateFileW(name: [*:0]const u16, access: u32, share: u32, security: ?*anyopaque, disposition: u32, flags: u32, template: ?*anyopaque) callconv(.winapi) ?*anyopaque;
extern "kernel32" fn CloseHandle(h: *anyopaque) callconv(.winapi) i32;

fn metadataWindows(path: [:0]const u8) !identity.Meta {
    var wide: [1024:0]u16 = undefined;
    const n = std.unicode.utf8ToUtf16Le(wide[0..1023], path) catch return error.SpillArtifactOpen;
    wide[n] = 0;
    const h = CreateFileW(&wide, 0x80000000, 0x7, null, 3, 0, null) orelse return error.SpillArtifactOpen;
    if (@intFromPtr(h) == std.math.maxInt(usize)) return error.SpillArtifactOpen;
    defer _ = CloseHandle(h);
    var info: ByHandleInfo = undefined;
    if (GetFileInformationByHandle(h, &info) == 0) return error.SpillArtifactOpen;
    const written = fileTime(info.write);
    const created = fileTime(info.creation);
    return .{
        .size = @as(u64, info.size_high) << 32 | info.size_low,
        .mtime_s = written.s,
        .mtime_ns = written.ns,
        .ctime_s = created.s,
        .ctime_ns = created.ns,
        .inode = @as(u64, info.index_high) << 32 | info.index_low,
        .dev_major = info.volume_serial,
        .dev_minor = 0,
    };
}

};

fn metadataWindows(path: [:0]const u8) !identity.Meta {
    if (comptime builtin.os.tag == .windows) return WinMeta.metadataWindows(path);
    unreachable;
}

test "GLM spill queue refuses a second capture and failed capture leaves no queued allocation" {
    const gpa = std.testing.allocator;
    var e: glm.engine.Engine = undefined;
    e.gpa = gpa;
    e.snaps = .empty;
    defer e.snaps.deinit(gpa);
    var link: glm.follow.Link = undefined;
    var tier: spill.Tier = undefined;
    tier.io = std.testing.io;
    tier.mutex = .init;
    tier.stats = .{};
    var a: Adapter = .{ .gpa = gpa, .io = std.testing.io, .settings = .{ .dir = "/unused", .cap = 1 << 20 }, .host_limit = 1024, .state = .{ .attached = .{ .e = &e, .link = &link, .tier = &tier } } };
    const command: persistence.Command = .{ .operation = .write, .id = 1, .key = 7, .at = 1, .tokens = &.{ 5, 9 } };
    try std.testing.expectError(error.NoKeptState, Adapter.request(&a, command));
    try std.testing.expect(a.queued == null);
    const tokens = try gpa.dupe(u32, command.tokens);
    a.queued = .{ .command = command, .tokens = tokens, .host = .{ .bytes = try gpa.dupe(u8, "owned snapshot"), .meta = undefined, .meta_len = 0 } };
    defer a.dropQueued();
    try std.testing.expectError(error.Incomplete, Adapter.request(&a, .{ .operation = .write, .id = 2, .key = 8, .at = 1, .tokens = &.{ 4, 2 } }));
    try std.testing.expectEqual(@as(u64, 1), tier.stats.unwritten_incomplete);
    try std.testing.expectEqual(@as(u64, 1), a.queued.?.command.id);
    try std.testing.expectEqualSlices(u32, &.{ 5, 9 }, a.queued.?.tokens);
    try std.testing.expectEqualStrings("owned snapshot", a.queued.?.host.bytes);
}
