//! TENSORFOLD_QUAL_IDS_DIR: an opt-in qualification record of each completed non-streamed reply's generated token ids.
//!
//! Unset (the default): nothing is hashed, opened or written. Set: at startup the directory must be absolute, an
//! existing directory (not a link) owned by this user with mode 0700, and outside the model folder, or the server
//! does not start. Each completed non-streamed reply then gets one file, `<sha256(reply id)>.json`, written to a new
//! private temporary file, synced and renamed into place: its reply id, finish reason, the committed generated ids
//! before any end-token stripping, their count and the SHA-256 of their canonical text (core/ids_json.zig, the text
//! every engine's token SHA is taken over). A record that cannot be written is logged and absent, never partial.
const std = @import("std");

/// A token list as Python's json.dumps writes it, "[1, 2, 3]": core/ids_json.zig's canonical text, written here
/// because the server's module cannot share that file with the engines' (it would sit in two modules).
fn idsJson(gpa: std.mem.Allocator, ids: []const u32) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(gpa);
    try out.append(gpa, '[');
    for (ids, 0..) |t, i| {
        if (i > 0) try out.appendSlice(gpa, ", ");
        try out.print(gpa, "{d}", .{t});
    }
    try out.append(gpa, ']');
    return out.toOwnedSlice(gpa);
}

/// Mode and owner of an open file (statx: no stat ABI differences between libcs).
fn modeOwner(fd: c_int) !struct { mode: u32, uid: u32 } {
    if (@import("builtin").os.tag != .linux) return error.Unsupported;
    var sx: std.os.linux.Statx = undefined;
    if (std.c.statx(fd, "", 0x1000, .{ .MODE = true, .UID = true }, &sx) != 0) return error.QualIdsDirUnusable; // AT_EMPTY_PATH
    return .{ .mode = sx.mode, .uid = sx.uid };
}

var serial: std.atomic.Value(u64) = .init(0);

pub const QualIds = struct {
    dfd: c_int,
    path: []u8,

    /// The configured recorder, null when TENSORFOLD_QUAL_IDS_DIR is unset; a configured but unsafe directory is an error.
    pub fn fromEnv(gpa: std.mem.Allocator, model_dir: []const u8) !?*QualIds {
        const raw = std.c.getenv("TENSORFOLD_QUAL_IDS_DIR") orelse return null;
        return open(gpa, std.mem.span(raw), model_dir);
    }

    pub fn open(gpa: std.mem.Allocator, dir: []const u8, model_dir: []const u8) !*QualIds {
        if (dir.len == 0 or dir[0] != '/') return error.QualIdsDirNotAbsolute;
        const z = try gpa.dupeSentinel(u8, dir, 0);
        defer gpa.free(z);
        const fd = std.c.open(z, .{ .ACCMODE = .RDONLY, .DIRECTORY = true, .NOFOLLOW = true, .CLOEXEC = true }, @as(std.c.mode_t, 0));
        if (fd < 0) return error.QualIdsDirUnusable;
        errdefer _ = std.c.close(fd);
        const st = try modeOwner(fd);
        if (st.uid != std.c.geteuid() or (st.mode & 0o777) != 0o700) return error.QualIdsDirUnsafe;
        if (inside(gpa, z, model_dir)) return error.QualIdsDirInModel;
        const q = try gpa.create(QualIds);
        errdefer gpa.destroy(q);
        q.* = .{ .dfd = fd, .path = try gpa.dupe(u8, dir) };
        return q;
    }

    pub fn deinit(q: *QualIds, gpa: std.mem.Allocator) void {
        _ = std.c.close(q.dfd);
        gpa.free(q.path);
        gpa.destroy(q);
    }

    /// Whether `dir` and the model folder resolve to the same place or one inside the other.
    fn inside(gpa: std.mem.Allocator, dir: [:0]const u8, model_dir: []const u8) bool {
        if (model_dir.len == 0) return false;
        var a: [std.c.PATH_MAX]u8 = undefined;
        var b: [std.c.PATH_MAX]u8 = undefined;
        const m = gpa.dupeSentinel(u8, model_dir, 0) catch return true;
        defer gpa.free(m);
        const ra = std.c.realpath(dir, &a) orelse return true;
        const rb = std.c.realpath(m, &b) orelse return false; // no such model folder: nothing to be inside
        const x = std.mem.span(ra);
        const y = std.mem.span(rb);
        return under(x, y) or under(y, x);
    }

    fn under(child: []const u8, parent: []const u8) bool {
        if (!std.mem.startsWith(u8, child, parent)) return false;
        return child.len == parent.len or parent.len == 1 or child[parent.len] == '/';
    }

    /// Writes the record of a completed reply; on failure nothing is left and the caller logs `error`.
    pub fn record(q: *const QualIds, gpa: std.mem.Allocator, reply_id: []const u8, finish: []const u8, ids: []const u32) !void {
        const text = try idsJson(gpa, ids);
        defer gpa.free(text);
        var d: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(text, &d, .{});
        var name_hash: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(reply_id, &name_hash, .{});
        const hex = std.fmt.bytesToHex(name_hash, .lower);
        var body: std.Io.Writer.Allocating = .init(gpa);
        defer body.deinit();
        const w = &body.writer;
        try w.writeAll("{\"version\": 1, \"reply_id\": ");
        try std.json.Stringify.value(reply_id, .{}, w);
        try w.writeAll(", \"finish_reason\": ");
        try std.json.Stringify.value(finish, .{}, w);
        try w.print(", \"count\": {d}, \"ids_sha256\": \"{x}\", \"ids\": ", .{ ids.len, d });
        try w.writeAll(text);
        try w.writeAll("}\n");
        var tmp_buf: [96]u8 = undefined;
        var final_buf: [80]u8 = undefined;
        // unique to this process and call; O_EXCL refuses any file already there
        const tmp = try std.fmt.bufPrintSentinel(&tmp_buf, ".{s}.{d}.{d}.tmp", .{ hex, std.c.getpid(), serial.fetchAdd(1, .monotonic) }, 0);
        const final = try std.fmt.bufPrintSentinel(&final_buf, "{s}.json", .{hex}, 0);
        const fd = std.c.openat(q.dfd, tmp, .{ .ACCMODE = .WRONLY, .CREAT = true, .EXCL = true, .NOFOLLOW = true, .CLOEXEC = true }, @as(std.c.mode_t, 0o600));
        if (fd < 0) return error.QualIdsCreate;
        var keep = false;
        defer if (!keep) {
            _ = std.c.unlinkat(q.dfd, tmp, 0);
        };
        {
            defer _ = std.c.close(fd);
            const bytes = body.written();
            var off: usize = 0;
            while (off < bytes.len) {
                const n = std.c.write(fd, bytes[off..].ptr, bytes.len - off);
                if (n <= 0) return error.QualIdsWrite;
                off += @intCast(n);
            }
            if (std.c.fsync(fd) != 0) return error.QualIdsSync;
        }
        if (std.c.renameat(q.dfd, tmp, q.dfd, final) != 0) return error.QualIdsRename;
        keep = true;
        _ = std.c.fsync(q.dfd);
    }
};

test "an unset directory records nothing; an unsafe, relative or in-model directory refuses to start" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var cwd_buf: [std.c.PATH_MAX]u8 = undefined;
    const cwd = std.mem.sliceTo(std.c.getcwd(&cwd_buf, cwd_buf.len).?, 0);
    const base = try std.fmt.allocPrint(gpa, "{s}/.zig-cache/tmp/{s}", .{ cwd, tmp.sub_path });
    defer gpa.free(base);
    try std.testing.expectError(error.QualIdsDirNotAbsolute, QualIds.open(gpa, "relative/ids", ""));
    const open_dir = try std.fmt.allocPrintSentinel(gpa, "{s}/open", .{base}, 0);
    defer gpa.free(open_dir);
    try std.Io.Dir.cwd().createDirPath(io, open_dir);
    _ = std.c.chmod(open_dir, 0o755);
    try std.testing.expectError(error.QualIdsDirUnsafe, QualIds.open(gpa, open_dir, ""));
    const model = try std.fmt.allocPrintSentinel(gpa, "{s}/model", .{base}, 0);
    defer gpa.free(model);
    const in_model = try std.fmt.allocPrintSentinel(gpa, "{s}/model/ids", .{base}, 0);
    defer gpa.free(in_model);
    try std.Io.Dir.cwd().createDirPath(io, in_model);
    _ = std.c.chmod(in_model, 0o700);
    try std.testing.expectError(error.QualIdsDirInModel, QualIds.open(gpa, in_model, model));
    const link = try std.fmt.allocPrintSentinel(gpa, "{s}/link", .{base}, 0);
    defer gpa.free(link);
    const private = try std.fmt.allocPrintSentinel(gpa, "{s}/ids", .{base}, 0);
    defer gpa.free(private);
    try std.Io.Dir.cwd().createDirPath(io, private);
    _ = std.c.chmod(private, 0o700);
    try std.testing.expectEqual(@as(c_int, 0), std.c.symlink(private, link));
    try std.testing.expectError(error.QualIdsDirUnusable, QualIds.open(gpa, link, model));
}

test "a reply's record holds its id, every committed id, their count and canonical SHA, privately and whole" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var cwd_buf: [std.c.PATH_MAX]u8 = undefined;
    const cwd = std.mem.sliceTo(std.c.getcwd(&cwd_buf, cwd_buf.len).?, 0);
    const dir = try std.fmt.allocPrintSentinel(gpa, "{s}/.zig-cache/tmp/{s}/ids", .{ cwd, tmp.sub_path }, 0);
    defer gpa.free(dir);
    try std.Io.Dir.cwd().createDirPath(io, dir);
    _ = std.c.chmod(dir, 0o700);
    const q = try QualIds.open(gpa, dir, "");
    defer q.deinit(gpa);
    const ids = [_]u32{ 101, 7, 2 }; // the end token 2 stays: the record is the committed reply, before stripping
    try q.record(gpa, "chatcmpl-x/../y", "stop", &ids);
    var name_hash: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash("chatcmpl-x/../y", &name_hash, .{});
    const path = try std.fmt.allocPrintSentinel(gpa, "{s}/{s}.json", .{ dir, std.fmt.bytesToHex(name_hash, .lower) }, 0);
    defer gpa.free(path);
    const text = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 16));
    defer gpa.free(text);
    var canon: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash("[101, 7, 2]", &canon, .{});
    var want_buf: [512]u8 = undefined;
    const want = try std.fmt.bufPrint(&want_buf, "{{\"version\": 1, \"reply_id\": \"chatcmpl-x/../y\", \"finish_reason\": \"stop\", \"count\": 3, \"ids_sha256\": \"{x}\", \"ids\": [101, 7, 2]}}\n", .{canon});
    try std.testing.expectEqualStrings(want, text);
    const fd = std.c.open(path, .{ .ACCMODE = .RDONLY }, @as(std.c.mode_t, 0));
    try std.testing.expect(fd >= 0);
    defer _ = std.c.close(fd);
    try std.testing.expectEqual(@as(u32, 0o600), (try modeOwner(fd)).mode & 0o777);
    // the canonical text is json.dumps' (core/ids_json.zig's): the same vectors pin both copies
    const vectors = [_]struct { ids: []const u32, text: []const u8 }{
        .{ .ids = &.{}, .text = "[]" },
        .{ .ids = &.{0}, .text = "[0]" },
        .{ .ids = &.{4294967295}, .text = "[4294967295]" },
        .{ .ids = &.{ 1, 22, 333 }, .text = "[1, 22, 333]" },
    };
    for (vectors) |v| {
        const t = try idsJson(gpa, v.ids);
        defer gpa.free(t);
        try std.testing.expectEqualStrings(v.text, t);
    }
    var ref: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash("[1, 22, 333]", &ref, .{});
    try std.testing.expectEqualStrings("276d8737d9d5a80805e0731e8d5ae18972a8e3ba721f4988f67151c6e9159068", &std.fmt.bytesToHex(ref, .lower));
}
