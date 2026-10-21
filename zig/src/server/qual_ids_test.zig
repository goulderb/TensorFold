//! TENSORFOLD_QUAL_IDS_DIR through the chat route: a completed non-streamed reply leaves one record named by its id
//! with every committed id (the end token included); a streamed reply leaves none.
const std = @import("std");
const server = @import("root.zig");
const json = server.json;
const openai = @import("openai.zig");
const Conn = @import("http_conn.zig").Conn;
const QualIds = @import("qual_ids.zig").QualIds;
const lp = @import("logprobs_test.zig");

const gpa = std.testing.allocator;

fn files(dir: [:0]const u8) !usize {
    const d = std.c.opendir(dir) orelse return error.OpenDir;
    defer _ = std.c.closedir(d);
    var n: usize = 0;
    while (std.c.readdir(d)) |ent| {
        const name = std.mem.sliceTo(@as([*:0]const u8, @ptrCast(&ent.name)), 0);
        if (!std.mem.eql(u8, name, ".") and !std.mem.eql(u8, name, "..")) n += 1;
    }
    return n;
}

fn chat(srv: *server.Server, body: []const u8) !lp.Got {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var conn: Conn = .{ .fd = @import("sys").sock.none, .peer = "", .buf = &.{}, .gpa = a };
    var got: lp.Got = .{};
    openai.run(srv, a, got.out(), .{ .conn = &conn }, true, (try json.parse(a, body)).ok);
    return got;
}

test "a completed non-streamed reply leaves one record under its id; a streamed reply leaves none" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var cwd_buf: [std.c.PATH_MAX]u8 = undefined;
    const cwd = std.mem.sliceTo(std.c.getcwd(&cwd_buf, cwd_buf.len).?, 0);
    const dir = try std.fmt.allocPrintSentinel(gpa, "{s}/.zig-cache/tmp/{s}/ids", .{ cwd, tmp.sub_path }, 0);
    defer gpa.free(dir);
    try std.Io.Dir.cwd().createDirPath(std.testing.io, dir);
    _ = std.c.chmod(dir, 0o700);
    const reply = lp.tokensOf("Hi") ++ [_]u32{lp.eos};
    var text: lp.Text = .{};
    var backend: lp.Script = .{ .reply = &reply };
    var srv = try server.Server.init(gpa, std.testing.io, backend.engine(), text.text(), .{ .served_name = "test-model", .model_ids = &.{"test-model"}, .enable_thinking = false, .use_drafts = false }, null);
    defer srv.deinit();
    try std.testing.expect(srv.qual_ids == null); // unset: nothing recorded
    srv.qual_ids = try QualIds.open(gpa, dir, "");

    var got = try chat(srv, "{\"model\":\"test-model\",\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"max_tokens\":8}");
    defer got.deinit();
    try std.testing.expectEqual(@as(u16, 200), got.status);
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, got.body, .{});
    defer parsed.deinit();
    const id = parsed.value.object.get("id").?.string;
    try std.testing.expectEqualStrings("Hi", parsed.value.object.get("choices").?.array.items[0].object.get("message").?.object.get("content").?.string);
    var h: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(id, &h, .{});
    const path = try std.fmt.allocPrintSentinel(gpa, "{s}/{s}.json", .{ dir, std.fmt.bytesToHex(h, .lower) }, 0);
    defer gpa.free(path);
    const record = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, path, gpa, .limited(1 << 16));
    defer gpa.free(record);
    const r = try std.json.parseFromSlice(std.json.Value, gpa, record, .{});
    defer r.deinit();
    try std.testing.expectEqualStrings(id, r.value.object.get("reply_id").?.string);
    try std.testing.expectEqualStrings("stop", r.value.object.get("finish_reason").?.string);
    try std.testing.expectEqual(@as(i64, 3), r.value.object.get("count").?.integer);
    const ids = r.value.object.get("ids").?.array.items;
    try std.testing.expectEqual(@as(usize, 3), ids.len);
    for ([_]i64{ 'H', 'i', lp.eos }, ids) |want, v| try std.testing.expectEqual(want, v.integer); // the end token kept
    var canon: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash("[72, 105, 301]", &canon, .{});
    try std.testing.expectEqualStrings(&std.fmt.bytesToHex(canon, .lower), r.value.object.get("ids_sha256").?.string);
    try std.testing.expectEqual(@as(usize, 1), try files(dir));

    var streamed = try chat(srv, "{\"model\":\"test-model\",\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"max_tokens\":8,\"stream\":true}");
    defer streamed.deinit();
    try std.testing.expectEqual(@as(usize, 1), try files(dir)); // no record for a streamed reply
}
