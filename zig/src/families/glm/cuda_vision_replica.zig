const std = @import("std");

pub const max_prompt: usize = 256 * 1024;
pub const max_image_rows: usize = 16 * 1024;
pub const width: usize = 4096;

pub const Manifest = struct {
    ids: []const u32,
    positions: []const u32,
    image_token: u32,
    replica: u64,

    pub fn featureBytes(self: Manifest) !usize {
        try self.validate();
        return std.math.mul(usize, self.positions.len, width * @sizeOf(u16)) catch error.InvalidVisionTransfer;
    }

    pub fn validate(self: Manifest) !void {
        if (self.replica == 0 or self.ids.len == 0 or self.ids.len > max_prompt or
            self.positions.len == 0 or self.positions.len > max_image_rows or
            self.positions.len > self.ids.len or self.image_token >= 154880)
            return error.InvalidVisionTransfer;
        var row: usize = 0;
        for (self.ids, 0..) |id, i| {
            if (id >= 154880) return error.InvalidVisionTransfer;
            if (id != self.image_token) continue;
            if (row >= self.positions.len or self.positions[row] != i) return error.InvalidVisionTransfer;
            row += 1;
        }
        if (row != self.positions.len) return error.InvalidVisionTransfer;
    }
};

fn length(h: *std.crypto.hash.sha2.Sha256, n: usize) void {
    var bytes: [8]u8 = undefined;
    std.mem.writeInt(u64, &bytes, @intCast(n), .little);
    h.update(&bytes);
}

/// The source bytes distinguish image turns whose expanded token IDs are identical.
pub fn requestDigest(ids: []const u32, images: []const []const u8) ![32]u8 {
    if (ids.len == 0 or ids.len > max_prompt or images.len == 0 or images.len > 256)
        return error.InvalidVisionTransfer;
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    length(&h, ids.len);
    h.update(std.mem.sliceAsBytes(ids));
    length(&h, images.len);
    for (images) |encoded| {
        if (encoded.len == 0 or encoded.len > 10 * 1024 * 1024) return error.InvalidVisionTransfer;
        length(&h, encoded.len);
        h.update(encoded);
    }
    var out: [32]u8 = undefined;
    h.final(&out);
    return out;
}

pub fn replicaId(digest: [32]u8) u64 {
    return std.mem.readInt(u64, digest[0..8], .little) | 1;
}

test "different image payloads cannot share a request identity" {
    const ids = [_]u32{ 7, 8, 42, 42, 9 };
    const a = try requestDigest(&ids, &.{"image a"});
    const b = try requestDigest(&ids, &.{"image b"});
    try std.testing.expect(!std.mem.eql(u8, &a, &b));
    try std.testing.expect(replicaId(a) != replicaId(b));
    const c = try requestDigest(&ids, &.{"image a"});
    try std.testing.expectEqualSlices(u8, &a, &c);
}

test "rank manifest requires exactly the image positions in the expanded prompt" {
    const ids = [_]u32{ 7, 8, 42, 42, 9 };
    const good: Manifest = .{ .ids = &ids, .positions = &.{ 2, 3 }, .image_token = 42, .replica = 1 };
    try std.testing.expectEqual(@as(usize, 2 * width * 2), try good.featureBytes());
    try std.testing.expectError(error.InvalidVisionTransfer, (Manifest{ .ids = &ids, .positions = &.{2}, .image_token = 42, .replica = 1 }).featureBytes());
    try std.testing.expectError(error.InvalidVisionTransfer, (Manifest{ .ids = &ids, .positions = &.{ 3, 2 }, .image_token = 42, .replica = 1 }).featureBytes());
    try std.testing.expectError(error.InvalidVisionTransfer, (Manifest{ .ids = &ids, .positions = &.{ 2, 3 }, .image_token = 42, .replica = 0 }).featureBytes());
}
