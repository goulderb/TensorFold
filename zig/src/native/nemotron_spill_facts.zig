//! The Nemotron spill tier's configuration and identity: what the environment asks for, and what the engine loaded.
//!
//! Configuration is opt-in: no TENSORFOLD_SPILL_GIB means no tier and no work (no hashing, no allocation). A set GIB of
//! 0 turns it off. Any other TENSORFOLD_SPILL_* setting without GIB is an error, as is an enabled tier without a valid
//! TENSORFOLD_SPILL_HOST_GIB (the host bytes its slot, transfer buffers and index may use).
//!
//! Identity binds a stored state to the bytes the engine computed with: each artifact file is stat'ed before the load
//! and hashed in full after it, its metadata unchanged throughout (a file replaced or rewritten in between refuses
//! the tier); the executable (its embedded kernels and draft vocabulary) and any captured AOT kernel set the engine
//! loaded are hashed the same way; runtime and layout facts are hashed into the identity's text fields.
const std = @import("std");
const builtin = @import("builtin");
const api = @import("engine_api");
const spill = api.prompt_cache.spill;
const Identity = api.spill_resume.Identity;

/// The file identity below is Linux's: statx, pread and a raw fd. Elsewhere the tier refuses, as `openRead` says.
const is_linux = builtin.os.tag == .linux;

pub const Config = struct {
    tier: spill.Settings, // `dir` allocated
    host_bytes: u64,

    const names = [_][:0]const u8{ "TENSORFOLD_SPILL_DIR", "TENSORFOLD_SPILL_HIGHWATER", "TENSORFOLD_SPILL_MIN_TOKENS", "TENSORFOLD_SPILL_MIN_FREE_GIB", "TENSORFOLD_SPILL_FLUSH_S", "TENSORFOLD_SPILL_KEEP_BUILDS", "TENSORFOLD_SPILL_HOST_GIB" };

    /// The configured tier, null when none is configured; a wrong or incomplete configuration is an error.
    pub fn fromEnv(gpa: std.mem.Allocator, env: anytype) !?Config {
        if (env.get("TENSORFOLD_SPILL_GIB") == null) {
            for (names) |n| if (env.get(n) != null) return error.SpillSettingWithoutGib;
            return null;
        }
        const tier = (try spill.Settings.fromEnv(gpa, env)) orelse return null;
        errdefer gpa.free(tier.dir);
        const text = env.get("TENSORFOLD_SPILL_HOST_GIB") orelse return error.SpillHostGibRequired;
        const g = std.fmt.parseFloat(f64, std.mem.trim(u8, text, " \t\r\n")) catch return error.BadSpillHostGib;
        if (!std.math.isFinite(g) or g <= 0 or g > spill.Settings.most_gib) return error.BadSpillHostGib;
        return .{ .tier = tier, .host_bytes = @intFromFloat(g * (1 << 30)) };
    }
};

/// A file's identity as the kernel reports it: a rewrite, a rename over it or a truncation changes it.
const Meta = struct {
    dev_major: u32,
    dev_minor: u32,
    ino: u64,
    size: u64,
    mtime: [2]i64,
    ctime: [2]i64,

    fn of(fd: c_int) !Meta {
        if (!is_linux) return error.Unsupported;
        const linux = std.os.linux;
        var sx: linux.Statx = undefined;
        if (std.c.statx(fd, "", 0x1000, .{ .INO = true, .SIZE = true, .MTIME = true, .CTIME = true }, &sx) != 0) return error.SpillArtifactStat; // AT_EMPTY_PATH
        return .{ .dev_major = sx.dev_major, .dev_minor = sx.dev_minor, .ino = sx.ino, .size = sx.size, .mtime = .{ sx.mtime.sec, sx.mtime.nsec }, .ctime = .{ sx.ctime.sec, sx.ctime.nsec } };
    }

    fn eql(a: Meta, b: Meta) bool {
        return std.meta.eql(a, b);
    }
};

fn openRead(path: [:0]const u8) !c_int {
    if (!is_linux) return error.Unsupported;
    const fd = std.c.open(path, .{ .ACCMODE = .RDONLY, .CLOEXEC = true }, @as(std.c.mode_t, 0));
    return if (fd < 0) error.SpillArtifactOpen else fd;
}

/// `std.c.close` takes a pointer on Windows, so the Linux-only callers close through here.
fn closeRead(fd: c_int) void {
    if (is_linux) _ = std.c.close(fd);
}

/// A positional read of `buf` at `off`; only the Linux tier reaches this.
fn preadRead(fd: c_int, buf: []u8, off: u64) isize {
    if (!is_linux) return -1;
    return std.c.pread(fd, buf.ptr, buf.len, @intCast(off));
}

fn metaOf(path: [:0]const u8) !Meta {
    const fd = try openRead(path);
    defer closeRead(fd);
    return Meta.of(fd);
}

/// Files whose bytes decide the engine's arithmetic, stat'ed before the load that reads them.
pub const Artifacts = struct {
    paths: [][:0]u8,
    before: []Meta,

    /// Takes ownership of `paths` (gpa-allocated, as the family's list returns them).
    pub fn take(gpa: std.mem.Allocator, paths: [][:0]u8) !Artifacts {
        errdefer free(gpa, paths);
        const before = try gpa.alloc(Meta, paths.len);
        errdefer gpa.free(before);
        for (paths, before) |p, *m| m.* = try metaOf(p);
        return .{ .paths = paths, .before = before };
    }

    /// One more file, stat'ed now; on failure `a` is unchanged.
    pub fn add(a: *Artifacts, gpa: std.mem.Allocator, path: []const u8) !void {
        const p = try gpa.dupeSentinel(u8, path, 0);
        errdefer gpa.free(p);
        const m = try metaOf(p);
        const paths = try gpa.alloc([:0]u8, a.paths.len + 1);
        errdefer gpa.free(paths);
        const before = try gpa.alloc(Meta, a.before.len + 1);
        @memcpy(paths[0..a.paths.len], a.paths);
        paths[a.paths.len] = p;
        @memcpy(before[0..a.before.len], a.before);
        before[a.before.len] = m;
        gpa.free(a.paths);
        gpa.free(a.before);
        a.paths = paths;
        a.before = before;
    }

    pub fn deinit(a: *Artifacts, gpa: std.mem.Allocator) void {
        free(gpa, a.paths);
        gpa.free(a.before);
        a.* = undefined;
    }

    fn free(gpa: std.mem.Allocator, paths: [][:0]u8) void {
        for (paths) |p| gpa.free(p);
        gpa.free(paths);
    }

    /// Every file's full SHA-256 in order (with its base name and size), each file unchanged since `take`.
    pub fn digest(a: *const Artifacts, gpa: std.mem.Allocator) ![32]u8 {
        var all = std.crypto.hash.sha2.Sha256.init(.{});
        all.update("tensorfold-nemotron-artifacts-v1");
        const buf = try gpa.alloc(u8, 4 << 20);
        defer gpa.free(buf);
        for (a.paths, a.before) |p, m| {
            const fd = try openRead(p);
            defer closeRead(fd);
            if (!Meta.eql(try Meta.of(fd), m)) return error.SpillArtifactChanged;
            var h = std.crypto.hash.sha2.Sha256.init(.{});
            var off: u64 = 0;
            while (true) {
                const n = preadRead(fd, buf, off);
                if (n < 0) return error.SpillArtifactRead;
                if (n == 0) break;
                h.update(buf[0..@intCast(n)]);
                off += @intCast(n);
            }
            if (off != m.size or !Meta.eql(try Meta.of(fd), m)) return error.SpillArtifactChanged;
            const base = std.fs.path.basename(p);
            var len: [8]u8 = undefined;
            std.mem.writeInt(u64, &len, base.len, .little);
            all.update(&len);
            all.update(base);
            std.mem.writeInt(u64, &len, off, .little);
            all.update(&len);
            all.update(&h.finalResult());
        }
        return all.finalResult();
    }
};

/// The running executable (its embedded kernels and draft vocabulary) and, when the engine loaded one, its captured
/// AOT kernel set's files, stat'ed now.
pub fn executable(gpa: std.mem.Allocator, io: std.Io, kernels: ?[]const u8) !Artifacts {
    const paths = try gpa.alloc([:0]u8, 1);
    paths[0] = gpa.dupeSentinel(u8, "/proc/self/exe", 0) catch |e| {
        gpa.free(paths);
        return e;
    };
    var a = try Artifacts.take(gpa, paths);
    errdefer a.deinit(gpa);
    if (kernels) |dir| {
        const files = try aotFiles(gpa, io, dir);
        defer Artifacts.free(gpa, files);
        for (files) |f| try a.add(gpa, f);
    }
    return a;
}

/// A captured kernel set's files as `cuda.aot.Set.load` reads them: aot.json and each kernel's cubin.
pub fn aotFiles(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) ![][:0]u8 {
    var out: std.ArrayList([:0]u8) = .empty;
    errdefer {
        for (out.items) |p| gpa.free(p);
        out.deinit(gpa);
    }
    const json = try std.fmt.allocPrintSentinel(gpa, "{s}/aot.json", .{dir}, 0);
    try out.append(gpa, json);
    const text = try std.Io.Dir.cwd().readFileAlloc(io, json, gpa, .limited(1 << 24));
    defer gpa.free(text);
    const Set = struct { kernels: []const struct { hash: []const u8 } };
    const parsed = try std.json.parseFromSlice(Set, gpa, text, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    for (parsed.value.kernels) |k| {
        if (k.hash.len == 0 or std.mem.indexOfAny(u8, k.hash, "/\\\x00") != null or std.mem.eql(u8, k.hash, "..")) return error.BadAotHash;
        try out.append(gpa, try std.fmt.allocPrintSentinel(gpa, "{s}/cubins/{s}.cubin", .{ dir, k.hash }, 0));
    }
    return out.toOwnedSlice(gpa);
}

/// Hex SHA-256 of `text`: structured facts in an Identity text field (each at most 256 bytes).
fn hexDigest(text: []const u8, out: *[64]u8) []const u8 {
    var d: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(text, &d, .{});
    _ = std.fmt.bufPrint(out, "{x}", .{d}) catch unreachable;
    return out;
}

/// What the engine computes with beyond its files.
pub const Facts = struct {
    weights: [32]u8, // Artifacts.digest of the model's files
    executable: [32]u8, // Artifacts.digest of the running executable and any AOT kernel set
    kernels: []const u8, // "embedded" or the AOT set's folder
    driver: i32, // cuDriverGetVersion
    capability: u32, // 10 * major + minor
    format: []const u8, // the loaded weights' format
    planes: [7]u64,
    head: u64,
    max_len: u64,
    grid: u32,
    lookahead: u32,
    segments: u32,

    /// The tier's identity: equal only for the same weights, executable, kernels, device class and state layout.
    pub fn identity(f: Facts) ![32]u8 {
        var buf: [1024]u8 = undefined;
        var rh: [64]u8 = undefined;
        var lh: [64]u8 = undefined;
        var kh: [64]u8 = undefined;
        const kernels = hexDigest(f.kernels, &kh); // a folder path of any length: its digest
        const runtime = try std.fmt.bufPrint(&buf, "exe={x} kernels={s} driver={d} sm={d} segments={d}", .{ f.executable, kernels, f.driver, f.capability, f.segments });
        const r = hexDigest(runtime, &rh);
        const layout = try std.fmt.bufPrint(&buf, "nemotron_h planes={d},{d},{d},{d},{d},{d},{d} head={d} max_len={d} grid={d} lookahead={d}", .{ f.planes[0], f.planes[1], f.planes[2], f.planes[3], f.planes[4], f.planes[5], f.planes[6], f.head, f.max_len, f.grid, f.lookahead });
        const l = hexDigest(layout, &lh);
        var db: [96]u8 = undefined;
        const dtype = try std.fmt.bufPrint(&db, "weights={s} kv=bf16 ssm=f32 conv=bf16 dt=f32 head=bf16", .{f.format});
        return (Identity{ .weights = f.weights, .runtime = r, .layout = l, .dtype = dtype, .tp = 1 }).digest();
    }
};

test {
    _ = @import("nemotron_spill_facts_test.zig");
}
