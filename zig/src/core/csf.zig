//! Lossless csf_32x2_v1 NVFP4 scale bytes, independent of checkpoint storage.

const std = @import("std");

pub const Error = error{
    InvalidRows,
    InvalidColumns,
    SizeOverflow,
    InvalidBasesLength,
    InvalidBase,
    InvalidCodesLength,
    InvalidPageBasesLength,
    InvalidOffsetsLength,
    InvalidExceptionsLength,
    ArenaTooLarge,
    InvalidPageBase,
    NonzeroPrefix,
    DecreasingPrefix,
    RecordCountPastArena,
    RecordsCrossPage,
    InvalidPosition,
    UnsortedPositions,
    DuplicatePosition,
    UnusedExceptionRecords,
    InvalidOutputLength,
};

const Geometry = struct {
    rows: usize,
    columns: usize,
    elements: usize,
    tile_columns: usize,
    tiles: usize,
    pages: usize,
    prefixes: usize,

    fn init(rows: usize, columns: usize) Error!Geometry {
        if (rows == 0 or rows % 32 != 0) return error.InvalidRows;
        if (columns == 0) return error.InvalidColumns;
        const elements = std.math.mul(usize, rows, columns) catch return error.SizeOverflow;
        const tile_columns = columns / 4 + @intFromBool(columns % 4 != 0);
        const tiles = std.math.mul(usize, rows / 128 + @intFromBool(rows % 128 != 0), tile_columns) catch return error.SizeOverflow;
        const pages = tiles / 32 + @intFromBool(tiles % 32 != 0);
        const blocks = std.math.mul(usize, rows / 32, columns / 2 + columns % 2) catch return error.SizeOverflow;
        const prefixes = std.math.add(usize, blocks, pages) catch return error.SizeOverflow;
        _ = std.math.mul(usize, prefixes, 2) catch return error.SizeOverflow;
        _ = std.math.mul(usize, pages, 4) catch return error.SizeOverflow;
        return .{ .rows = rows, .columns = columns, .elements = elements, .tile_columns = tile_columns, .tiles = tiles, .pages = pages, .prefixes = prefixes };
    }
};

const Block = struct { row: usize, column: usize, width: usize, page: usize };

const Blocks = struct {
    g: Geometry,
    tile: usize = 0,
    pair: usize = 0,
    row_block: usize = 0,

    fn next(self: *Blocks) ?Block {
        if (self.tile == self.g.tiles) return null;
        const row = self.tile / self.g.tile_columns * 128;
        const column = self.tile % self.g.tile_columns * 4;
        const result: Block = .{
            .row = row + self.row_block * 32,
            .column = column + self.pair * 2,
            .width = @min(2, self.g.columns - column - self.pair * 2),
            .page = self.tile / 32,
        };
        self.row_block += 1;
        if (self.row_block == @min(128, self.g.rows - row) / 32) {
            self.row_block = 0;
            self.pair += 1;
            if (self.pair == (@min(4, self.g.columns - column) + 1) / 2) {
                self.pair = 0;
                self.tile += 1;
            }
        }
        return result;
    }
};

fn u16le(bytes: []const u8, index: usize) u16 {
    return std.mem.readInt(u16, bytes[index * 2 ..][0..2], .little);
}

fn u32le(bytes: []const u8, index: usize) u32 {
    return std.mem.readInt(u32, bytes[index * 4 ..][0..4], .little);
}

/// Validate exact component sizes, palettes, page ownership and all exception records.
pub fn validate(bases: []const u8, codes: []const u8, page_bases: []const u8, offsets: []const u8, exceptions: []const u8, rows: usize, columns: usize) Error!void {
    const g = try Geometry.init(rows, columns);
    if (bases.len != rows) return error.InvalidBasesLength;
    if (codes.len != g.elements / 2) return error.InvalidCodesLength;
    if (page_bases.len != g.pages * 4) return error.InvalidPageBasesLength;
    if (offsets.len != g.prefixes * 2) return error.InvalidOffsetsLength;
    if (exceptions.len % 2 != 0) return error.InvalidExceptionsLength;
    if (exceptions.len >= @as(u64, 1) << 32) return error.ArenaTooLarge;
    for (bases) |base| if (base > 240) return error.InvalidBase;

    var blocks: Blocks = .{ .g = g };
    var page: ?usize = null;
    var prefix: usize = 0;
    var origin: usize = 0;
    var limit: usize = 0;
    var end: usize = 0;
    while (blocks.next()) |b| {
        if (page == null or page.? != b.page) {
            if (page != null) {
                if (end != limit) return error.InvalidPageBase;
                prefix += 1;
            }
            page = b.page;
            origin = u32le(page_bases, b.page);
            if (origin != end or origin % 2 != 0 or origin > exceptions.len) return error.InvalidPageBase;
            limit = if (b.page + 1 < g.pages) u32le(page_bases, b.page + 1) else exceptions.len;
            if (limit % 2 != 0 or limit < origin or limit > exceptions.len) return error.InvalidPageBase;
            if (u16le(offsets, prefix) != 0) return error.NonzeroPrefix;
        }
        const start_count: usize = u16le(offsets, prefix);
        const end_count: usize = u16le(offsets, prefix + 1);
        if (end_count < start_count) return error.DecreasingPrefix;
        if (end_count > (exceptions.len - origin) / 2) return error.RecordCountPastArena;
        end = origin + end_count * 2;
        if (end > limit) return error.RecordsCrossPage;
        var previous: ?u8 = null;
        var record = origin + start_count * 2;
        while (record < end) : (record += 2) {
            const position = exceptions[record];
            if (position >= 64 or position % 2 >= b.width) return error.InvalidPosition;
            if (previous) |p| {
                if (position == p) return error.DuplicatePosition;
                if (position < p) return error.UnsortedPositions;
            }
            previous = position;
        }
        prefix += 1;
    }
    if (end != exceptions.len) return error.UnusedExceptionRecords;
}

/// Write exact row-major [rows, columns] E4M3 bytes. Inputs and output must not overlap.
/// Validation completes before any output write; no allocation or floating-point conversion.
pub fn decode(bases: []const u8, codes: []const u8, page_bases: []const u8, offsets: []const u8, exceptions: []const u8, rows: usize, columns: usize, output: []u8) Error!void {
    const g = try Geometry.init(rows, columns);
    if (output.len != g.elements) return error.InvalidOutputLength;
    try validate(bases, codes, page_bases, offsets, exceptions, rows, columns);
    var blocks: Blocks = .{ .g = g };
    var field: usize = 0;
    var prefix: usize = 0;
    var page: usize = 0;
    while (blocks.next()) |b| {
        if (page != b.page) {
            prefix += 1;
            page = b.page;
        }
        for (0..32) |r| {
            for (0..b.width) |c| {
                const shift: u3 = @intCast((field % 2) * 4);
                output[(b.row + r) * columns + b.column + c] = bases[b.row + r] + ((codes[field / 2] >> shift) & 15);
                field += 1;
            }
        }
        const origin: usize = u32le(page_bases, b.page);
        var record = origin + @as(usize, u16le(offsets, prefix)) * 2;
        const end = origin + @as(usize, u16le(offsets, prefix + 1)) * 2;
        while (record < end) : (record += 2) {
            const position = exceptions[record];
            output[(b.row + position / 2) * columns + b.column + position % 2] = exceptions[record + 1];
        }
        prefix += 1;
    }
}

const TestPlane = struct {
    parts: [5][]u8 = .{ &.{}, &.{}, &.{}, &.{}, &.{} },
    rows: usize,
    columns: usize,

    fn deinit(self: *TestPlane) void {
        for (self.parts) |p| std.testing.allocator.free(p);
    }

    fn replace(self: *TestPlane, index: usize, bytes: []const u8) !void {
        const copy = try std.testing.allocator.dupe(u8, bytes);
        std.testing.allocator.free(self.parts[index]);
        self.parts[index] = copy;
    }

    fn check(self: TestPlane) Error!void {
        const p = self.parts;
        return validate(p[0], p[1], p[2], p[3], p[4], self.rows, self.columns);
    }

    fn expand(self: TestPlane, output: []u8) Error!void {
        const p = self.parts;
        return decode(p[0], p[1], p[2], p[3], p[4], self.rows, self.columns, output);
    }

    fn refused(self: TestPlane, expected: Error) !void {
        try std.testing.expectError(expected, self.check());
        const output = try std.testing.allocator.alloc(u8, self.rows * self.columns);
        defer std.testing.allocator.free(output);
        @memset(output, 0xa5);
        try std.testing.expectError(expected, self.expand(output));
        for (output) |byte| try std.testing.expectEqual(@as(u8, 0xa5), byte);
    }
};

fn appendInt(list: *std.ArrayList(u8), comptime T: type, n: usize) !void {
    var bytes: [@sizeOf(T)]u8 = undefined;
    std.mem.writeInt(T, &bytes, @intCast(n), .little);
    try list.appendSlice(std.testing.allocator, &bytes);
}

// Test encoder deliberately uses nested tile loops rather than the decoder's iterator.
fn testEncode(source: []const u8, rows: usize, columns: usize) !TestPlane {
    var result: TestPlane = .{ .rows = rows, .columns = columns };
    errdefer result.deinit();
    const a = std.testing.allocator;
    result.parts[0] = try a.alloc(u8, rows);
    result.parts[1] = try a.alloc(u8, source.len / 2);
    @memset(result.parts[1], 0);
    for (0..rows) |r| {
        var histogram: [256]usize = @splat(0);
        for (source[r * columns ..][0..columns]) |v| histogram[v] += 1;
        var count: usize = 0;
        for (histogram[0..16]) |n| count += n;
        var best = count;
        var base: u8 = 0;
        for (1..241) |b| {
            count = count - histogram[b - 1] + histogram[b + 15];
            if (count > best) {
                best = count;
                base = @intCast(b);
            }
        }
        result.parts[0][r] = base;
    }
    var origins: std.ArrayList(u8) = .empty;
    defer origins.deinit(a);
    var offsets: std.ArrayList(u8) = .empty;
    defer offsets.deinit(a);
    var records: std.ArrayList(u8) = .empty;
    defer records.deinit(a);
    var tile: usize = 0;
    var page_origin: usize = 0;
    var field: usize = 0;
    var r0: usize = 0;
    while (r0 < rows) : (r0 += 128) {
        var c0: usize = 0;
        while (c0 < columns) : (c0 += 4) {
            if (tile % 32 == 0) {
                if (tile != 0) try appendInt(&offsets, u16, (records.items.len - page_origin) / 2);
                page_origin = records.items.len;
                try appendInt(&origins, u32, page_origin);
            }
            var c1 = c0;
            while (c1 < @min(columns, c0 + 4)) : (c1 += 2) {
                var r1 = r0;
                while (r1 < @min(rows, r0 + 128)) : (r1 += 32) {
                    try appendInt(&offsets, u16, (records.items.len - page_origin) / 2);
                    for (r1..r1 + 32) |r| {
                        for (c1..@min(columns, c1 + 2)) |c| {
                            const v = source[r * columns + c];
                            const delta = @as(i16, v) - result.parts[0][r];
                            const inside = delta >= 0 and delta <= 15;
                            if (!inside) try appendInt(&records, u16, (r - r1) * 2 + c - c1 + @as(usize, v) * 256);
                            const shift: u3 = @intCast(field % 2 * 4);
                            result.parts[1][field / 2] |= @as(u8, if (inside) @intCast(delta) else 0) << shift;
                            field += 1;
                        }
                    }
                }
            }
            tile += 1;
        }
    }
    try appendInt(&offsets, u16, (records.items.len - page_origin) / 2);
    result.parts[2] = try origins.toOwnedSlice(a);
    result.parts[3] = try offsets.toOwnedSlice(a);
    result.parts[4] = try records.toOwnedSlice(a);
    return result;
}

fn zeroPlane(rows: usize, columns: usize) !TestPlane {
    const source = try std.testing.allocator.alloc(u8, rows * columns);
    defer std.testing.allocator.free(source);
    @memset(source, 0);
    return testEncode(source, rows, columns);
}

fn roundTrip(source: []const u8, rows: usize, columns: usize) !void {
    var encoded = try testEncode(source, rows, columns);
    defer encoded.deinit();
    const output = try std.testing.allocator.alloc(u8, source.len);
    defer std.testing.allocator.free(output);
    try encoded.expand(output);
    try std.testing.expectEqualSlices(u8, source, output);
}

test "all palette bases and nibble values are exact, including base 240 and offset 15" {
    var plane = try zeroPlane(32, 2);
    defer plane.deinit();
    var output: [64]u8 = undefined;
    for (0..241) |base| {
        @memset(plane.parts[0], @intCast(base));
        for (0..16) |lo| {
            for (0..16) |hi| {
                @memset(plane.parts[1], @intCast(lo | hi << 4));
                try plane.expand(&output);
                for (0..32) |r| {
                    try std.testing.expectEqual(@as(u8, @intCast(base + lo)), output[r * 2]);
                    try std.testing.expectEqual(@as(u8, @intCast(base + hi)), output[r * 2 + 1]);
                }
            }
        }
    }
}

test "every exception position and byte including both NaN patterns is preserved" {
    var plane = try zeroPlane(32, 2);
    defer plane.deinit();
    try plane.replace(3, &.{ 0, 0, 1, 0 });
    try plane.replace(4, &.{ 0, 0 });
    var output: [64]u8 = undefined;
    for (0..64) |position| {
        for (0..256) |value| {
            plane.parts[4][0] = @intCast(position);
            plane.parts[4][1] = @intCast(value);
            try plane.expand(&output);
            for (output, 0..) |actual, i| try std.testing.expectEqual(@as(u8, if (i == position) @intCast(value) else 0), actual);
        }
    }
}

test "single-column codes pack successive rows, while exceptions keep even positions" {
    const bases: [32]u8 = @splat(240);
    const codes: [16]u8 = @splat(0xf1);
    var output: [32]u8 = undefined;
    try decode(&bases, &codes, &.{ 0, 0, 0, 0 }, &.{ 0, 0, 2, 0 }, &.{ 0, 0x7f, 62, 0xff }, 32, 1, &output);
    for (output, 0..) |actual, r| {
        const expected: u8 = if (r == 0) 0x7f else if (r == 31) 0xff else if (r % 2 == 0) 241 else 255;
        try std.testing.expectEqual(expected, actual);
    }
}

test "random byte matrices round trip across all boundary tiles and page transitions" {
    var random = std.Random.DefaultPrng.init(0xc5f32);
    const a = std.testing.allocator;
    for ([_]usize{ 32, 64, 96, 128, 160, 256 }) |rows| {
        for ([_]usize{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 127, 128, 129, 256, 257 }) |columns| {
            const source = try a.alloc(u8, rows * columns);
            defer a.free(source);
            random.random().bytes(source);
            try roundTrip(source, rows, columns);
        }
    }
}

test "all-equal rows for every byte and every row needing exceptions" {
    var source: [160 * 5]u8 = undefined;
    for (0..256) |value| {
        @memset(&source, @intCast(value));
        try roundTrip(&source, 160, 5);
    }
    for (0..160) |r| {
        @memcpy(source[r * 5 ..][0..5], &[_]u8{ 0, 15, 16, 127, 255 });
    }
    var encoded = try testEncode(&source, 160, 5);
    defer encoded.deinit();
    try std.testing.expectEqual(@as(usize, 160 * 3 * 2), encoded.parts[4].len);
    try roundTrip(&source, 160, 5);
}

test "empty exception pages can precede and follow occupied pages" {
    const rows = 128;
    const columns = 384;
    const source = try std.testing.allocator.alloc(u8, rows * columns);
    defer std.testing.allocator.free(source);
    @memset(source, 0);
    for (0..rows) |r| source[r * columns + 128] = 255;
    var encoded = try testEncode(source, rows, columns);
    defer encoded.deinit();
    try std.testing.expectEqualSlices(u8, &.{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0 }, encoded.parts[2]);
    try roundTrip(source, rows, columns);
}

test "a full exception page uses record prefixes and byte origins" {
    var plane = try zeroPlane(128, 260);
    defer plane.deinit();
    const a = std.testing.allocator;
    var offsets: std.ArrayList(u8) = .empty;
    defer offsets.deinit(a);
    var records: std.ArrayList(u8) = .empty;
    defer records.deinit(a);
    for (0..257) |block| try appendInt(&offsets, u16, block * 64);
    for (0..257) |block| try appendInt(&offsets, u16, block * 64);
    for (0..9) |block| try appendInt(&offsets, u16, block * 64);
    for (0..520) |_| {
        for (0..64) |position| try appendInt(&records, u16, position + (255 - position) * 256);
    }
    try plane.replace(2, &.{ 0, 0, 0, 0, 0, 128, 0, 0, 0, 0, 1, 0 });
    try plane.replace(3, offsets.items);
    try plane.replace(4, records.items);
    const output = try a.alloc(u8, 128 * 260);
    defer a.free(output);
    try plane.expand(output);
    for (0..128) |r| {
        for (0..260) |c| try std.testing.expectEqual(@as(u8, @intCast(255 - (r % 32 * 2 + c % 2))), output[r * 260 + c]);
    }
}

fn fixtureValue(row: usize, column: usize, columns: usize) u8 {
    return if (column == columns - 1 and row % 97 == 0) 255 else @intCast((row % 15) * 16 + (if (column % 2 == 1) @as(usize, 15) else 0));
}

fn checkFixture(bytes: []const u8) !void {
    var plane: TestPlane = .{ .rows = u32le(bytes, 0), .columns = u32le(bytes, 1) };
    defer plane.deinit();
    var size: usize = 0;
    for (0..5) |i| size += u32le(bytes, i + 2);
    const raw = try std.testing.allocator.alloc(u8, size);
    defer std.testing.allocator.free(raw);
    var cursor: usize = 0;
    var i: usize = 28;
    while (i < bytes.len) : (i += 5) {
        const count = std.mem.readInt(u32, bytes[i..][0..4], .little);
        @memset(raw[cursor..][0..count], bytes[i + 4]);
        cursor += count;
    }
    try std.testing.expectEqual(size, cursor);
    cursor = 0;
    for (0..5) |part| {
        const count = u32le(bytes, part + 2);
        try plane.replace(part, raw[cursor..][0..count]);
        cursor += count;
    }
    const output = try std.testing.allocator.alloc(u8, plane.rows * plane.columns);
    defer std.testing.allocator.free(output);
    try plane.expand(output);
    for (0..plane.rows) |r| {
        for (0..plane.columns) |c| try std.testing.expectEqual(fixtureValue(r, c, plane.columns), output[r * plane.columns + c]);
    }
}

test "independent Python fixtures at both GLM shapes and three boundary geometries" {
    try checkFixture(@embedFile("fixtures/csf/2048x256.rle"));
    try checkFixture(@embedFile("fixtures/csf/4096x128.rle"));
    try checkFixture(@embedFile("fixtures/csf/32x1.rle"));
    try checkFixture(@embedFile("fixtures/csf/160x5.rle"));
    try checkFixture(@embedFile("fixtures/csf/160x3.rle"));
}

test "invalid row and column extents are refused" {
    for ([_]usize{ 0, 1, 31, 33, 159 }) |rows| try std.testing.expectError(error.InvalidRows, validate(&.{}, &.{}, &.{}, &.{}, &.{}, rows, 1));
    try std.testing.expectError(error.InvalidColumns, validate(&.{}, &.{}, &.{}, &.{}, &.{}, 32, 0));
}

test "overflowing geometry is refused before accessing components" {
    try std.testing.expectError(error.SizeOverflow, validate(&.{}, &.{}, &.{}, &.{}, &.{}, 32, std.math.maxInt(usize)));
    try std.testing.expectError(error.SizeOverflow, decode(&.{}, &.{}, &.{}, &.{}, &.{}, std.math.maxInt(usize) - 31, 2, &.{}));
}

test "base component length must equal rows" {
    var p = try zeroPlane(32, 3);
    defer p.deinit();
    try p.replace(0, &.{0});
    try p.refused(error.InvalidBasesLength);
}

test "every base above 240 is refused" {
    var p = try zeroPlane(32, 3);
    defer p.deinit();
    for (241..256) |v| {
        p.parts[0][31] = @intCast(v);
        try p.refused(error.InvalidBase);
    }
}

test "code component length must equal rows times columns over two" {
    var p = try zeroPlane(32, 3);
    defer p.deinit();
    try p.replace(1, &.{0});
    try p.refused(error.InvalidCodesLength);
}

test "page origin count must equal P" {
    var p = try zeroPlane(32, 3);
    defer p.deinit();
    try p.replace(2, &.{ 0, 0, 0 });
    try p.refused(error.InvalidPageBasesLength);
}

test "prefix count must equal B plus P" {
    var p = try zeroPlane(32, 3);
    defer p.deinit();
    try p.replace(3, &.{ 0, 0, 0, 0 });
    try p.refused(error.InvalidOffsetsLength);
}

test "exception records must occupy whole U16 values" {
    var p = try zeroPlane(32, 3);
    defer p.deinit();
    try p.replace(4, &.{0});
    try p.refused(error.InvalidExceptionsLength);
}

test "exception arena cannot reach 4 GiB" {
    if (@sizeOf(usize) < 8) return error.SkipZigTest;
    var p = try zeroPlane(32, 3);
    defer p.deinit();
    const inaccessible: [*]const u8 = @ptrFromInt(1);
    try std.testing.expectError(error.ArenaTooLarge, validate(p.parts[0], p.parts[1], p.parts[2], p.parts[3], inaccessible[0 .. @as(usize, 1) << 32], 32, 3));
}

test "page origins start at zero and must be aligned and inside arena" {
    var p = try zeroPlane(32, 129);
    defer p.deinit();
    for ([_][8]u8{
        .{ 2, 0, 0, 0, 2, 0, 0, 0 },
        .{ 0, 0, 0, 0, 1, 0, 0, 0 },
        .{ 0, 0, 0, 0, 4, 0, 0, 0 },
    }) |origins| {
        try p.replace(2, &origins);
        try p.replace(4, &.{ 0, 127 });
        try p.refused(error.InvalidPageBase);
    }
}

test "page origins cannot leave gaps or decrease" {
    var p = try zeroPlane(32, 257);
    defer p.deinit();
    try p.replace(4, &.{ 0, 127, 0, 255 });
    try p.replace(2, &.{ 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0 });
    try p.refused(error.InvalidPageBase);
    try p.replace(2, &.{ 0, 0, 0, 0, 2, 0, 0, 0, 4, 0, 0, 0 });
    try p.refused(error.InvalidPageBase);
}

test "each page prefix starts at zero, including empty later pages" {
    var p = try zeroPlane(32, 129);
    defer p.deinit();
    p.parts[3][0] = 1;
    try p.refused(error.NonzeroPrefix);
    p.parts[3][0] = 0;
    p.parts[3][65 * 2] = 1;
    try p.refused(error.NonzeroPrefix);
}

test "decreasing record prefixes are refused" {
    var p = try zeroPlane(32, 3);
    defer p.deinit();
    try p.replace(3, &.{ 0, 0, 1, 0, 0, 0 });
    try p.replace(4, &.{ 0, 127 });
    try p.refused(error.DecreasingPrefix);
}

test "record counts cannot run past exception arena" {
    var p = try zeroPlane(32, 3);
    defer p.deinit();
    p.parts[3][2] = 1;
    try p.refused(error.RecordCountPastArena);
}

test "records cannot cross a page origin" {
    var p = try zeroPlane(32, 129);
    defer p.deinit();
    p.parts[3][2] = 1;
    try p.replace(4, &.{ 0, 127 });
    try p.refused(error.RecordsCrossPage);
}

test "positions outside 32 rows are refused" {
    var p = try zeroPlane(32, 2);
    defer p.deinit();
    try p.replace(3, &.{ 0, 0, 1, 0 });
    try p.replace(4, &.{ 64, 127 });
    for (64..256) |position| {
        p.parts[4][0] = @intCast(position);
        try p.refused(error.InvalidPosition);
    }
}

test "odd exception positions are invalid in a final single-column block" {
    var p = try zeroPlane(32, 3);
    defer p.deinit();
    try p.replace(3, &.{ 0, 0, 0, 0, 1, 0 });
    try p.replace(4, &.{ 1, 127 });
    for (0..32) |r| {
        p.parts[4][0] = @intCast(r * 2 + 1);
        try p.refused(error.InvalidPosition);
    }
}

test "unsorted exception positions are refused" {
    var p = try zeroPlane(32, 2);
    defer p.deinit();
    try p.replace(3, &.{ 0, 0, 2, 0 });
    try p.replace(4, &.{ 2, 127, 1, 255 });
    try p.refused(error.UnsortedPositions);
}

test "duplicate exception positions are refused" {
    var p = try zeroPlane(32, 2);
    defer p.deinit();
    try p.replace(3, &.{ 0, 0, 2, 0 });
    try p.replace(4, &.{ 2, 127, 2, 255 });
    try p.refused(error.DuplicatePosition);
}

test "unused trailing exception records are refused" {
    var p = try zeroPlane(32, 2);
    defer p.deinit();
    try p.replace(4, &.{ 0, 127 });
    try p.refused(error.UnusedExceptionRecords);
}

test "output extent must equal logical scale extent" {
    var p = try zeroPlane(32, 2);
    defer p.deinit();
    var output: [65]u8 = undefined;
    try std.testing.expectError(error.InvalidOutputLength, p.expand(output[0..63]));
    try std.testing.expectError(error.InvalidOutputLength, p.expand(&output));
}

test "all components can be unaligned byte slices" {
    var p = try zeroPlane(32, 1);
    defer p.deinit();
    try p.replace(3, &.{ 0, 0, 1, 0 });
    try p.replace(4, &.{ 62, 255 });
    var storage: [5][128]u8 align(8) = undefined;
    var slices: [5][]const u8 = undefined;
    for (p.parts, 0..) |part, i| {
        @memcpy(storage[i][1..][0..part.len], part);
        slices[i] = storage[i][1..][0..part.len];
    }
    var output: [32]u8 = undefined;
    try decode(slices[0], slices[1], slices[2], slices[3], slices[4], 32, 1, &output);
    for (output, 0..) |byte, r| try std.testing.expectEqual(@as(u8, if (r == 31) 255 else 0), byte);
}

test "both truncated and oversized components are refused" {
    var p = try zeroPlane(32, 5);
    defer p.deinit();
    const errors = [_]Error{ error.InvalidBasesLength, error.InvalidCodesLength, error.InvalidPageBasesLength, error.InvalidOffsetsLength };
    for (errors, 0..) |expected, component| {
        const original = try std.testing.allocator.dupe(u8, p.parts[component]);
        defer std.testing.allocator.free(original);
        try p.replace(component, original[0 .. original.len - 1]);
        try p.refused(expected);
        const longer = try std.testing.allocator.alloc(u8, original.len + 1);
        defer std.testing.allocator.free(longer);
        @memset(longer, 0);
        try p.replace(component, longer);
        try p.refused(expected);
        try p.replace(component, original);
    }
}

test "deterministic malformed component mutations never panic or partially write" {
    var random = std.Random.DefaultPrng.init(0xbad32);
    var source: [160 * 5]u8 = undefined;
    random.random().bytes(&source);
    var p = try testEncode(&source, 160, 5);
    defer p.deinit();
    var output: [160 * 5]u8 = undefined;
    for (0..4000) |_| {
        const component = random.random().uintLessThan(usize, 5);
        const index = random.random().uintLessThan(usize, p.parts[component].len);
        const saved = p.parts[component][index];
        p.parts[component][index] = random.random().int(u8);
        @memset(&output, 0xa5);
        if (p.expand(&output)) |_| {} else |_| {
            for (output) |byte| try std.testing.expectEqual(@as(u8, 0xa5), byte);
        }
        p.parts[component][index] = saved;
    }
}
