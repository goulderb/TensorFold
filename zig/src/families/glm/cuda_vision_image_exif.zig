//! Bounded, allocation-free EXIF orientation inspection. This is NOT pixel decoding.
//! Only classic TIFF IFD0 orientation is interpreted; sub-IFDs/thumbnails/XMP are not traversed.
//! Container framing is checked, but compressed pixels, image dimensions, frame count and
//! color profiles still require a real codec/admission layer. A successful result is NOT a valid image.
//! Missing EXIF/orientation returns null; malformed EXIF, duplicate blocks/tags and values outside
//! 1..8 explicitly fail instead of silently becoming orientation 1.
const std = @import("std");
pub const Orientation = @import("cuda_vision_image.zig").Orientation;

pub const MetadataLimits = struct {
    max_encoded_bytes: usize = 10 * 1024 * 1024,
    max_exif_bytes: usize = 256 * 1024,
    max_chunks: usize = 4096,
    max_ifd_entries: usize = 512,

    fn check(self: MetadataLimits) !void {
        if (self.max_encoded_bytes == 0 or self.max_exif_bytes == 0 or
            self.max_chunks == 0 or self.max_ifd_entries == 0) return error.InvalidImageMetadataBudget;
    }
};
const png_signature = "\x89PNG\r\n\x1a\n";
const exif_signature = "Exif\x00\x00";

fn span(bytes: []const u8, offset: usize, count: usize) ![]const u8 {
    if (offset > bytes.len or count > bytes.len - offset) return error.InvalidImageMetadata;
    return bytes[offset..][0..count];
}
fn u16at(bytes: []const u8, offset: usize, endian: std.builtin.Endian) !u16 {
    return std.mem.readInt(u16, (try span(bytes, offset, 2))[0..2], endian);
}
fn u32at(bytes: []const u8, offset: usize, endian: std.builtin.Endian) !u32 {
    return std.mem.readInt(u32, (try span(bytes, offset, 4))[0..4], endian);
}
fn tick(count: *usize, limits: MetadataLimits) !void {
    if (count.* >= limits.max_chunks) return error.ImageMetadataLimitExceeded;
    count.* += 1;
}

/// Raw TIFF bytes, without an Exif\0\0 prefix. No allocation or pointer chasing.
/// IFD0 tables/value spans are checked even for tags that are not orientation.
/// Next-IFD references are range-checked, not traversed; nested EXIF semantics remain opaque.
pub fn tiffOrientation(bytes: []const u8, limits: MetadataLimits) !?Orientation {
    try limits.check();
    if (bytes.len > limits.max_encoded_bytes) return error.ImageEncodedByteLimit;
    if (bytes.len > limits.max_exif_bytes) return error.ImageMetadataLimitExceeded;
    const header = try span(bytes, 0, 8);
    const endian: std.builtin.Endian = if (std.mem.eql(u8, header[0..2], "II")) .little else if (std.mem.eql(u8, header[0..2], "MM")) .big else return error.InvalidImageMetadata;
    if (try u16at(bytes, 2, endian) != 42) return error.InvalidImageMetadata;
    const ifd: usize = try u32at(bytes, 4, endian);
    if (ifd == 0) return null;
    if (ifd < 8 or ifd % 2 != 0) return error.InvalidImageMetadata;
    const count: usize = try u16at(bytes, ifd, endian);
    if (count > limits.max_ifd_entries) return error.ImageMetadataLimitExceeded;
    const entries_bytes = std.math.mul(usize, count, 12) catch return error.InvalidImageMetadata;
    const entries_at = std.math.add(usize, ifd, 2) catch return error.InvalidImageMetadata;
    _ = try span(bytes, entries_at, entries_bytes);
    const next_at = std.math.add(usize, entries_at, entries_bytes) catch return error.InvalidImageMetadata;
    const next: usize = try u32at(bytes, next_at, endian);
    const table_end = std.math.add(usize, next_at, 4) catch return error.InvalidImageMetadata;
    if (next != 0) {
        if (next < 8 or next % 2 != 0 or (next >= ifd and next < table_end)) return error.InvalidImageMetadata;
        _ = try span(bytes, next, 2);
    }
    var orientation: ?Orientation = null;
    for (0..count) |i| {
        const at = entries_at + i * 12;
        const tag = try u16at(bytes, at, endian);
        const kind = try u16at(bytes, at + 2, endian);
        const n: usize = try u32at(bytes, at + 4, endian);
        const unit: usize = switch (kind) {
            1, 2, 6, 7 => 1,
            3, 8 => 2,
            4, 9, 11, 13 => 4,
            5, 10, 12 => 8,
            else => return error.InvalidImageMetadata,
        };
        const size = std.math.mul(usize, n, unit) catch return error.InvalidImageMetadata;
        if (size > 4) {
            const offset: usize = try u32at(bytes, at + 8, endian);
            if (offset < 8 or offset % 2 != 0) return error.InvalidImageMetadata;
            _ = try span(bytes, offset, size);
            // Values cannot alias the IFD0 table they purport to describe.
            if (offset < table_end and offset + size > ifd) return error.InvalidImageMetadata;
        }
        if (tag == 0x0112) {
            if (orientation != null) return error.DuplicateExifMetadata;
            if (kind != 3 or n != 1) return error.InvalidImageMetadata;
            orientation = try Orientation.fromExif(try u16at(bytes, at + 8, endian));
        }
    }
    return orientation;
}

const Found = struct {
    seen: bool = false,
    orientation: ?Orientation = null,
    fn add(self: *Found, bytes: []const u8, limits: MetadataLimits) !void {
        if (self.seen) return error.DuplicateExifMetadata;
        self.seen = true;
        self.orientation = try tiffOrientation(bytes, limits);
    }
};

/// JPEG marker walk, including bounded entropy scanning (stuffing/restart markers).
/// This checks marker lengths and requires EOI; it does not validate/decompress scan data.
fn jpeg(bytes: []const u8, limits: MetadataLimits) !?Orientation {
    var at: usize = 2;
    var chunks: usize = 0;
    var entropy = false;
    var found: Found = .{};
    while (true) {
        const marker = try jpegMarker(bytes, &at, entropy, &chunks, limits);
        try tick(&chunks, limits);
        switch (marker) {
            0xd9 => {
                if (at != bytes.len) return error.InvalidImageMetadata;
                return found.orientation;
            },
            0xd8, 0xd0...0xd7 => return error.InvalidImageMetadata,
            0x01 => continue,
            else => {},
        }
        const length: usize = try u16at(bytes, at, .big);
        if (length < 2) return error.InvalidImageMetadata;
        const segment = try span(bytes, at, length);
        const payload = segment[2..];
        at += length;
        if (marker == 0xe1 and std.mem.startsWith(u8, payload, "Exif")) {
            if (!std.mem.startsWith(u8, payload, exif_signature)) return error.InvalidImageMetadata;
            try found.add(payload[exif_signature.len..], limits);
        }
        if (marker == 0xda) {
            if (payload.len < 6 or payload[0] == 0 or payload[0] > 4 or
                payload.len != 4 + @as(usize, payload[0]) * 2) return error.InvalidImageMetadata;
            entropy = true;
        } else if (marker != 0xdc) entropy = false;
    }
}
fn jpegMarker(bytes: []const u8, at: *usize, entropy: bool, chunks: *usize, limits: MetadataLimits) !u8 {
    while (at.* < bytes.len) {
        if (bytes[at.*] != 0xff) {
            if (!entropy) return error.InvalidImageMetadata;
            at.* += 1;
            continue;
        }
        while (at.* < bytes.len and bytes[at.*] == 0xff) at.* += 1;
        if (at.* == bytes.len) return error.InvalidImageMetadata;
        const marker = bytes[at.*];
        at.* += 1;
        if (marker == 0 or (marker >= 0xd0 and marker <= 0xd7)) {
            if (!entropy) return error.InvalidImageMetadata;
            if (marker != 0) try tick(chunks, limits);
            continue;
        }
        return marker;
    }
    return error.InvalidImageMetadata;
}

/// PNG chunk framing, CRCs and terminal IEND, without interpreting compressed IDAT.
fn png(bytes: []const u8, limits: MetadataLimits) !?Orientation {
    var at: usize = png_signature.len;
    var chunks: usize = 0;
    var found: Found = .{};
    while (at < bytes.len) {
        try tick(&chunks, limits);
        const header = try span(bytes, at, 8);
        const length: usize = std.mem.readInt(u32, header[0..4], .big);
        if (length > std.math.maxInt(i32)) return error.InvalidImageMetadata;
        const kind = header[4..8];
        for (kind) |ch| if (!std.ascii.isAlphabetic(ch)) return error.InvalidImageMetadata;
        if (kind[2] < 'A' or kind[2] > 'Z') return error.InvalidImageMetadata;
        if (chunks == 1) {
            if (!std.mem.eql(u8, kind, "IHDR") or length != 13) return error.InvalidImageMetadata;
        } else if (std.mem.eql(u8, kind, "IHDR")) return error.InvalidImageMetadata;
        const payload_at = at + 8; // header span proves this addition is in range
        const payload = try span(bytes, payload_at, length);
        const crc_at = payload_at + length;
        const expected = try u32at(bytes, crc_at, .big);
        var crc = std.hash.Crc32.init();
        crc.update(kind);
        crc.update(payload);
        if (crc.final() != expected) return error.InvalidImageMetadata;
        at = crc_at + 4;
        if (std.mem.eql(u8, kind, "eXIf")) try found.add(payload, limits);
        if (std.mem.eql(u8, kind, "IEND")) {
            if (length != 0 or at != bytes.len) return error.InvalidImageMetadata;
            return found.orientation;
        }
    }
    return error.InvalidImageMetadata;
}

/// RIFF WebP chunk sizes and required odd-byte padding, without interpreting VP8/VP8L/VP8X.
fn webp(bytes: []const u8, limits: MetadataLimits) !?Orientation {
    const header = try span(bytes, 0, 12);
    if (!std.mem.eql(u8, header[8..12], "WEBP")) return error.UnsupportedImageFormat;
    const riff_size: usize = std.mem.readInt(u32, header[4..8], .little);
    if (riff_size < 4 or riff_size != bytes.len - 8) return error.InvalidImageMetadata;
    var at: usize = 12;
    var chunks: usize = 0;
    var found: Found = .{};
    while (at < bytes.len) {
        try tick(&chunks, limits);
        const chunk = try span(bytes, at, 8);
        const length: usize = std.mem.readInt(u32, chunk[4..8], .little);
        const payload_at = at + 8;
        const payload = try span(bytes, payload_at, length);
        at = payload_at + length;
        if (length % 2 != 0) {
            const padding = try span(bytes, at, 1);
            if (padding[0] != 0) return error.InvalidImageMetadata;
            at += 1;
        }
        if (std.mem.eql(u8, chunk[0..4], "EXIF")) {
            const tiff = if (std.mem.startsWith(u8, payload, exif_signature)) payload[exif_signature.len..] else payload;
            try found.add(tiff, limits);
        }
    }
    if (chunks == 0) return error.InvalidImageMetadata;
    return found.orientation;
}

/// JPEG, PNG or RIFF WebP metadata only. null means absent, never that pixels decoded successfully.
/// Callers use `(try encodedOrientation(...)) orelse .normal` only AFTER metadata success.
pub fn encodedOrientation(bytes: []const u8, limits: MetadataLimits) !?Orientation {
    try limits.check();
    if (bytes.len > limits.max_encoded_bytes) return error.ImageEncodedByteLimit;
    if (std.mem.startsWith(u8, bytes, "\xff\xd8")) return jpeg(bytes, limits);
    if (std.mem.startsWith(u8, bytes, png_signature)) return png(bytes, limits);
    if (std.mem.startsWith(u8, bytes, "RIFF")) return webp(bytes, limits);
    return error.UnsupportedImageFormat;
}
