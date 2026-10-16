//! TENSORFOLD_REQUEST_LOG: each chat and completion body appended as one JSON line, images redacted, for replays.
const std = @import("std");
const json = @import("json");
const platform = @import("platform");
const Value = json.Value;
const Allocator = std.mem.Allocator;

/// The media part kind whose URL is redacted (image_url or video_url), or null for any other value.
fn redactKind(v: Value) ?[]const u8 {
    for ([_][]const u8{ "image_url", "video_url" }) |kind| if (v.typeIs(kind)) return kind;
    return null;
}

/// The body with every image and video part's URL replaced, the rest kept.
fn redact(a: Allocator, v: Value) Allocator.Error!Value {
    switch (v) {
        .array => |items| {
            const out = try a.alloc(Value, items.len);
            for (items, out) |item, *slot| slot.* = try redact(a, item);
            return .{ .array = out };
        },
        .object => |o| {
            if (redactKind(v)) |kind| {
                const copy = try json.copyObject(a, o);
                const url = try json.newObject(a);
                try url.put(a, "url", .{ .string = "<redacted>" });
                try copy.put(a, kind, .{ .object = url });
                return .{ .object = copy };
            }
            const out = try json.newObject(a);
            for (o.keys(), o.values()) |k, item| try out.put(a, k, try redact(a, item));
            return .{ .object = out };
        },
        else => return v,
    }
}

/// Appends ``body`` to ``path`` unless it is background work (batch jobs are not client traffic).
pub fn append(a: Allocator, path: []const u8, body: Value) void {
    if (body.strField("priority")) |p| if (std.mem.eql(u8, p, "background")) return;
    const line = json.stringify(a, redact(a, body) catch return, .{}) catch return;
    const framed = std.mem.concat(a, u8, &.{ line, "\n" }) catch return;
    platform.appendFile(a, path, framed);
}

test "the request log is created owner-only and keeps image URLs out" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}/requests.jsonl", .{tmp.sub_path});
    const url = try json.newObject(a);
    try url.put(a, "url", .{ .string = "https://example.test/private.png" });
    const part = try json.newObject(a);
    try part.put(a, "type", .{ .string = "image_url" });
    try part.put(a, "image_url", .{ .object = url });
    const body = try json.newObject(a);
    try body.put(a, "messages", .{ .array = try a.dupe(Value, &.{.{ .object = part }}) });
    append(a, path, .{ .object = body });
    const st = try std.Io.Dir.cwd().statFile(std.testing.io, path, .{}); // Linux's libc has no stat for std.c
    try std.testing.expectEqual(@as(u32, 0o600), @as(u32, @intCast(st.permissions.toMode())) & 0o777);
    const text = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, path, a, .limited(1 << 16));
    try std.testing.expect(std.mem.indexOf(u8, text, "<redacted>") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "example.test") == null);
}

test "a video part's URL is redacted, as an image part's is" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const raw = (try json.parse(a, "[{\"type\":\"video_url\",\"video_url\":{\"url\":\"https://x.test/clip.mp4\"}},{\"type\":\"image_url\",\"image_url\":{\"url\":\"https://x.test/a.png\"}}]")).ok;
    const out = try redact(a, raw);
    for (out.array) |part| {
        const key = if (part.get("video_url") != null) "video_url" else "image_url";
        try std.testing.expectEqualStrings("<redacted>", part.get(key).?.get("url").?.string);
    }
}

test "a redacted video part keeps its video_url key and no image_url key" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const raw = (try json.parse(a, "[{\"type\":\"video_url\",\"video_url\":{\"url\":\"https://x.test/clip.mp4\"}}]")).ok;
    const part = (try redact(a, raw)).array[0];
    try std.testing.expect(part.get("video_url") != null);
    try std.testing.expect(part.get("image_url") == null);
    try std.testing.expectEqualStrings("video_url", part.get("type").?.string);
}
