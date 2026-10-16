//! Schema features passed to the pinned xgrammar compiler. Unknown constraints fail before generation.
const std = @import("std");
const json = @import("json.zig");
const errors = @import("errors.zig");

pub fn validate(cx: *errors.Cx, value: json.Value) errors.Refused!void {
    var count: usize = 0;
    try node(cx, value, 0, &count);
}

fn isType(value: json.Value) bool {
    if (value != .string) return false;
    for ([_][]const u8{ "object", "array", "string", "number", "integer", "boolean", "null" }) |name|
        if (std.mem.eql(u8, value.string, name)) return true;
    return false;
}

fn matchesType(value: json.Value, kind: json.Value) bool {
    if (kind == .array) return for (kind.array) |item| {
        if (matchesType(value, item)) break true;
    } else false;
    if (kind != .string) return false;
    if (std.mem.eql(u8, kind.string, "number")) return value == .int or value == .float;
    if (std.mem.eql(u8, kind.string, "integer")) return value == .int or (value == .float and @floor(value.float) == value.float);
    const name: []const u8 = switch (value) {
        .null => "null",
        .bool => "boolean",
        .string => "string",
        .array => "array",
        .object => "object",
        else => "",
    };
    return std.mem.eql(u8, kind.string, name);
}

fn node(cx: *errors.Cx, value: json.Value, depth: usize, count: *usize) errors.Refused!void {
    count.* += 1;
    if (depth > 64 or count.* > 4096) return cx.refuse("JSON schema exceeds the supported nesting or node limit");
    if (value != .object) return cx.refuse("each JSON schema must be an object");
    const selected: ?[]const u8 = if (value.has("const")) "const" else if (value.has("enum")) "enum" else if (value.has("anyOf")) "anyOf" else null;
    if (selected) |selector| {
        for (value.object.keys()) |key| {
            if (std.mem.eql(u8, key, selector) or std.mem.eql(u8, key, "title") or std.mem.eql(u8, key, "description") or std.mem.eql(u8, key, "$schema")) continue;
            if (std.mem.eql(u8, key, "type") and !std.mem.eql(u8, selector, "anyOf")) continue;
            return cx.fail(.request, "JSON schema {s} combined with {s} is not supported; the constraint cannot be enforced", .{ selector, key });
        }
        if (value.get("type")) |kind| {
            if (value.get("const")) |literal| if (!matchesType(literal, kind)) return cx.refuse("JSON schema const does not match its type");
            if (value.get("enum")) |choices| if (choices == .array) {
                for (choices.array) |literal| if (!matchesType(literal, kind)) return cx.refuse("JSON schema enum does not match its type");
            };
        }
    }
    for (value.object.keys(), value.object.values()) |key, v| {
        if (std.mem.eql(u8, key, "type")) {
            if (v == .array) {
                if (v.array.len == 0) return cx.refuse("JSON schema type must not be empty");
                for (v.array) |t| if (!isType(t)) return cx.refuse("JSON schema type contains an unsupported type");
            } else if (!isType(v)) return cx.refuse("JSON schema type is unsupported");
        } else if (std.mem.eql(u8, key, "properties")) {
            if (v != .object) return cx.refuse("JSON schema properties must be an object");
            for (v.object.values()) |child| try node(cx, child, depth + 1, count);
        } else if (std.mem.eql(u8, key, "required")) {
            if (!value.has("properties") and !value.has("additionalProperties") and !value.typeIs("object")) return cx.refuse("JSON schema required needs an explicit object schema");
            if (v != .array) return cx.refuse("JSON schema required must be an array of property names");
            for (v.array) |name| if (name != .string) return cx.refuse("JSON schema required must contain only property names");
        } else if (std.mem.eql(u8, key, "additionalProperties")) {
            if (v != .bool) try node(cx, v, depth + 1, count);
        } else if (std.mem.eql(u8, key, "items")) {
            try node(cx, v, depth + 1, count);
        } else if (std.mem.eql(u8, key, "minItems") or std.mem.eql(u8, key, "maxItems")) {
            if (!value.typeIs("array")) return cx.refuse("JSON schema item bounds need type array");
            const bound = v.int64() orelse return cx.refuse("JSON schema item bounds must be integers");
            if (bound < 0 or bound > 1048576) return cx.refuse("JSON schema item bound is outside the supported range");
        } else if (std.mem.eql(u8, key, "anyOf")) {
            if (v != .array or v.array.len == 0) return cx.refuse("JSON schema anyOf must be a non-empty array of schemas");
            for (v.array) |child| try node(cx, child, depth + 1, count);
        } else if (std.mem.eql(u8, key, "enum")) {
            if (v != .array or v.array.len == 0) return cx.refuse("JSON schema enum must be a non-empty array");
        } else if (std.mem.eql(u8, key, "const")) {
            // Literal objects' keys are values, not schema keywords.
        } else if (std.mem.eql(u8, key, "title") or std.mem.eql(u8, key, "description") or std.mem.eql(u8, key, "$schema")) {
            if (v != .string) return cx.fail(.request, "JSON schema {s} must be a string", .{key});
        } else return cx.fail(.request, "JSON schema keyword '{s}' is not supported; the constraint cannot be enforced", .{key});
    }
}

test "schema subset accepts nested required objects arrays null unions and UTF8 literals" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    var cx: errors.Cx = .{ .a = arena.allocator() };
    const text =
        \\{"type":"object","properties":{"city":{"enum":["東京","Zürich"]},"rows":{"type":"array","items":{"anyOf":[{"type":"integer"},{"type":"null"}]}},"note":{"type":["string","null"]}},"required":["city","rows"],"additionalProperties":false}
    ;
    try validate(&cx, (try json.parse(cx.a, text)).ok);
    try validate(&cx, (try json.parse(cx.a, "{\"const\":{\"pattern\":\"literal value\"}}")).ok);
}

test "schema subset rejects ignored constraints at every schema nesting level" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    var cx: errors.Cx = .{ .a = arena.allocator() };
    const rejected = [_][]const u8{
        "{\"type\":\"number\",\"minimum\":3}",
        "{\"type\":\"string\",\"pattern\":\"abc\"}",
        "{\"properties\":{\"x\":{\"maxLength\":2}}}",
        "{\"items\":{\"$ref\":\"#/$defs/A\"}}",
        "{\"anyOf\":[{\"format\":\"date\"}]}",
        "{\"additionalProperties\":{\"multipleOf\":2}}",
        "{\"type\":\"imaginary\"}",
        "{\"required\":[3]}",
        "{\"enum\":[]}",
        "{\"type\":\"string\",\"enum\":[3]}",
        "{\"const\":{},\"required\":[\"x\"]}",
        "{\"type\":\"integer\",\"anyOf\":[{\"type\":\"string\"}]}",
        "{\"required\":[\"x\"]}",
    };
    for (rejected) |text| try std.testing.expectError(error.Refused, validate(&cx, (try json.parse(cx.a, text)).ok));
}
