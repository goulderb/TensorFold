//! Lossless csf_32x2_v1 NVFP4 scale bytes, independent of checkpoint storage.

// New code implementing Local Inference Lab's documented format (flashinfer docs/csf-scale-blocks.md at 7e203806,

// Apache-2.0); credits in LICENSES/donors/README.md.

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

// Validation completes before any output write; no allocation or floating-point conversion.

/// Write exact row-major [rows, columns] E4M3 bytes. Inputs and output must not overlap.
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

test {
    _ = @import("csf_test.zig");
}
