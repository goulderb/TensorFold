const std = @import("std");
const json = @import("json.zig");
const errors = @import("errors.zig");
const Value = json.Value;
const Cx = errors.Cx;

pub const Probe = struct {
    ctx: *anyopaque,
    rows: *const fn (*anyopaque, []const u8, usize) anyerror!u32,
};

pub const Prepared = struct {
    messages: Value,
    encoded: []const []const u8,
    rows: []const u32,

    pub fn hasImages(self: Prepared) bool {
        return self.encoded.len != 0;
    }
};

fn imageUrl(part: Value) ?[]const u8 {
    const image = part.get("image_url") orelse return null;
    return switch (image) {
        .string => |s| s,
        .object => image.strField("url"),
        else => null,
    };
}

/// HTTP boundary for inline payloads; it follows the native preprocessor's
/// data-URL limits while keeping that CUDA-family file in one Zig module graph.
fn inlineBytes(a: std.mem.Allocator, url: []const u8, max_bytes: usize) ![]u8 {
    if (max_bytes == 0 or max_bytes > 10 * 1024 * 1024) return error.InvalidImageBudget;
    if (!std.mem.startsWith(u8, url, "data:")) return error.ImageUrlsDisabled;
    const comma = std.mem.indexOfScalar(u8, url, ',') orelse return error.InvalidImageDataUrl;
    if (comma > 256) return error.InvalidImageDataUrl;
    const payload = url[comma + 1 ..];
    for (payload) |ch| if (ch > 127) return error.InvalidImageDataUrl;
    var media = std.mem.splitScalar(u8, url[5..comma], ';');
    _ = media.next();
    const encoding = media.next();
    if (media.next() != null) return error.InvalidImageDataUrl;
    if (encoding) |e| {
        if (!std.ascii.eqlIgnoreCase(e, "base64")) return error.InvalidImageDataUrl;
        if (payload.len > 4 * ((max_bytes + 2) / 3)) return error.ImageEncodedByteLimit;
        const decoder = std.base64.standard.Decoder;
        const size = decoder.calcSizeForSlice(payload) catch return error.InvalidImageDataUrl;
        if (size == 0 or size > max_bytes) return error.ImageEncodedByteLimit;
        const out = try a.alloc(u8, size);
        errdefer a.free(out);
        decoder.decode(out, payload) catch return error.InvalidImageDataUrl;
        return out;
    }
    if (payload.len > max_bytes * 3) return error.ImageEncodedByteLimit;
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(a);
    var i: usize = 0;
    while (i < payload.len) {
        var ch = payload[i];
        if (ch == '%' and i + 2 < payload.len) {
            ch = std.fmt.parseInt(u8, payload[i + 1 .. i + 3], 16) catch return error.InvalidImageDataUrl;
            i += 2;
        } else if (ch == '%') return error.InvalidImageDataUrl;
        if (out.items.len == max_bytes) return error.ImageEncodedByteLimit;
        try out.append(a, ch);
        i += 1;
    }
    if (out.items.len == 0) return error.ImageEncodedByteLimit;
    return out.toOwnedSlice(a);
}

/// The existing text template sees GLM's actual special-token spellings. The
/// encoded payload stays separate from that text and follows the request to rank 0.
pub fn prepare(cx: *Cx, original: Value, probe: Probe) errors.Refused!Prepared {
    if (original != .array) return cx.refuse("messages must be a non-empty list");
    const messages = try cx.a.dupe(Value, original.array);
    var encoded: std.ArrayList([]const u8) = .empty;
    var rows: std.ArrayList(u32) = .empty;
    var total_rows: usize = 0;
    var total_images: usize = 0;
    var total_encoded_bytes: usize = 0;
    for (original.array) |message| {
        const content = message.get("content") orelse continue;
        if (content != .array) continue;
        for (content.array) |part| if (part.typeIs("image_url")) {
            total_images += 1;
            if (total_images > 50) return cx.refuse("too many images in one request");
        };
    }
    for (original.array, messages) |message, *dst| {
        const content = message.get("content") orelse continue;
        if (content != .array) continue;
        var text: std.ArrayList(u8) = .empty;
        var images_in_message: usize = 0;
        for (content.array) |part| {
            if (part.typeIs("text")) {
                const value = part.strField("text") orelse return cx.refuse("a text content part must contain a text string");
                try text.appendSlice(cx.a, value);
                continue;
            }
            if (!part.typeIs("image_url")) return cx.refuse("this server accepts text and still image parts only");
            const role = message.strField("role") orelse "";
            if (!std.mem.eql(u8, role, "user") and !std.mem.eql(u8, role, "tool")) return cx.refuse("only user and tool messages can contain image parts");
            if (encoded.items.len >= 50) return cx.refuse("too many images in one request");
            const url = imageUrl(part) orelse return cx.refuse("an image_url part needs a URL");
            const bytes = inlineBytes(cx.a, url, 10 * 1024 * 1024) catch |err| return cx.fail(.request, "invalid image data URL ({s})", .{@errorName(err)});
            total_encoded_bytes = std.math.add(usize, total_encoded_bytes, bytes.len) catch return cx.refuse("image request exceeds the total byte limit");
            if (total_encoded_bytes > 64 * 1024 * 1024) return cx.refuse("image request exceeds the total byte limit");
            const count = probe.rows(probe.ctx, bytes, total_images) catch |err| return cx.fail(.request, "cannot decode image ({s})", .{@errorName(err)});
            if (count == 0 or count > 8000) return cx.refuse("image produced an invalid number of feature rows");
            total_rows = std.math.add(usize, total_rows, count) catch return cx.refuse("too many image rows");
            if (total_rows > 16384) return cx.refuse("too many image rows");
            try encoded.append(cx.a, bytes);
            try rows.append(cx.a, count);
            images_in_message += 1;
            try text.appendSlice(cx.a, "<|begin_of_image|>");
            for (0..count) |_| try text.appendSlice(cx.a, "<|image|>");
            try text.appendSlice(cx.a, "<|end_of_image|>");
        }
        if (images_in_message == 0) continue;
        const copy = try json.copyObject(cx.a, message.object);
        try copy.put(cx.a, "content", .{ .string = text.items });
        dst.* = .{ .object = copy };
    }
    return .{ .messages = .{ .array = messages }, .encoded = encoded.items, .rows = rows.items };
}

test "a still image keeps its ordered text, expands exact marker rows and preserves bytes" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    var cx: Cx = .{ .a = arena.allocator() };
    const parsed = try json.parse(cx.a,
        \\[{"role":"user","content":[{"type":"text","text":"What? "},{"type":"image_url","image_url":{"url":"data:image/png;base64,YWJj"}},{"type":"text","text":" Thanks"}]}]
    );
    const messages = switch (parsed) { .ok => |v| v, .err => return error.TestUnexpectedResult };
    const mock = struct {
        fn rows(_: *anyopaque, bytes: []const u8, images: usize) !u32 {
            try std.testing.expectEqualStrings("abc", bytes);
            try std.testing.expectEqual(@as(usize, 1), images);
            return 2;
        }
    };
    var context: u8 = 0;
    const result = try prepare(&cx, messages, .{ .ctx = &context, .rows = mock.rows });
    try std.testing.expect(result.hasImages());
    try std.testing.expectEqual(@as(usize, 1), result.encoded.len);
    try std.testing.expectEqual(@as(u32, 2), result.rows[0]);
    try std.testing.expectEqualStrings("What? <|begin_of_image|><|image|><|image|><|end_of_image|> Thanks", result.messages.array[0].strField("content").?);
}

test "tool screenshots use the same ordered markers" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    var cx: Cx = .{ .a = arena.allocator() };
    const parsed = try json.parse(cx.a,
        \\[{"role":"tool","tool_call_id":"t1","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,YWJj"}}]}]
    );
    const messages = switch (parsed) { .ok => |v| v, .err => return error.TestUnexpectedResult };
    const mock = struct {
        fn rows(_: *anyopaque, _: []const u8, _: usize) !u32 { return 1; }
    };
    var context: u8 = 0;
    const result = try prepare(&cx, messages, .{ .ctx = &context, .rows = mock.rows });
    try std.testing.expectEqualStrings("<|begin_of_image|><|image|><|end_of_image|>", result.messages.array[0].strField("content").?);
}

test "multiple pictures retain message order and the shared per-picture budget" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    var cx: Cx = .{ .a = arena.allocator() };
    const parsed = try json.parse(cx.a,
        \\[{"role":"user","content":[{"type":"text","text":"First "},{"type":"image_url","image_url":{"url":"data:image/png;base64,YQ=="}}]},{"role":"tool","tool_call_id":"t1","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,Yg=="}}]}]
    );
    const messages = switch (parsed) { .ok => |v| v, .err => return error.TestUnexpectedResult };
    const mock = struct {
        fn rows(_: *anyopaque, _: []const u8, images: usize) !u32 {
            try std.testing.expectEqual(@as(usize, 2), images);
            return 1;
        }
    };
    var context: u8 = 0;
    const result = try prepare(&cx, messages, .{ .ctx = &context, .rows = mock.rows });
    try std.testing.expectEqual(@as(usize, 2), result.encoded.len);
    try std.testing.expectEqualStrings("a", result.encoded[0]);
    try std.testing.expectEqualStrings("b", result.encoded[1]);
    try std.testing.expectEqualStrings("First <|begin_of_image|><|image|><|end_of_image|>", result.messages.array[0].strField("content").?);
    try std.testing.expectEqualStrings("<|begin_of_image|><|image|><|end_of_image|>", result.messages.array[1].strField("content").?);
}
