const std = @import("std");
const cuda = @import("cuda");

const Uuid = extern struct { bytes: [16]u8 };
const Selection = struct { ordinal: c_int, uuid: [16]u8 };

pub const Identity = struct { ordinal: c_int, uuid: []const u8, name: []const u8 };

pub const Admission = struct {
    selected: [2]Selection = undefined,
    len: usize,

    pub fn parse(order: ?[]const u8, visible: ?[]const u8, ordinals: []const c_int, expected: []const []const u8) !Admission {
        if (order == null or !std.mem.eql(u8, order.?, "PCI_BUS_ID")) return error.UnpinnedCudaDeviceOrder;
        var fields = std.mem.splitScalar(u8, visible orelse return error.UnpinnedCudaDevices, ',');
        const first_uuid = try parseUuid(fields.next() orelse return error.UnpinnedCudaDevices);
        const second_uuid = try parseUuid(fields.next() orelse return error.UnpinnedCudaDevices);
        if (fields.next() != null or std.mem.eql(u8, &first_uuid, &second_uuid)) return error.UnpinnedCudaDevices;
        const pinned = [_][16]u8{ first_uuid, second_uuid };
        if (ordinals.len == 0 or ordinals.len > 2 or expected.len != ordinals.len) return error.InvalidDeviceSelection;
        var result: Admission = .{ .len = ordinals.len };
        for (ordinals, expected, 0..) |ordinal, expected_uuid, i| {
            if (ordinal < 0 or ordinal >= pinned.len) return error.InvalidDeviceSelection;
            if (i > 0 and ordinal == ordinals[0]) return error.InvalidDeviceSelection;
            const uuid = try parseUuid(expected_uuid);
            if (!std.mem.eql(u8, &uuid, &pinned[@intCast(ordinal)])) return error.DeviceIdentityMismatch;
            result.selected[i] = .{ .ordinal = ordinal, .uuid = uuid };
        }
        return result;
    }

    pub fn verify(self: Admission, a: std.mem.Allocator, driver: *cuda.Driver) ![]Identity {
        const get_uuid = driver.lib.lookup(*const fn (*Uuid, cuda.abi.Device) callconv(.c) cuda.abi.Result, "cuDeviceGetUuid_v2") orelse return error.MissingDeviceUuidSymbol;
        const identities = try a.alloc(Identity, self.len);
        for (self.selected[0..self.len], identities) |selected, *identity| {
            var device: cuda.abi.Device = 0;
            try driver.check(driver.api.cuDeviceGet(&device, selected.ordinal), "cuDeviceGet");
            var actual: Uuid = undefined;
            try driver.check(get_uuid(&actual, device), "cuDeviceGetUuid_v2");
            if (!std.mem.eql(u8, &actual.bytes, &selected.uuid)) return error.DeviceIdentityMismatch;
            var name: [256]u8 = undefined;
            identity.* = .{ .ordinal = selected.ordinal, .uuid = try formatUuid(a, actual.bytes), .name = try a.dupe(u8, try driver.deviceName(selected.ordinal, &name)) };
        }
        return identities;
    }
};

fn parseUuid(text: []const u8) ![16]u8 {
    if (text.len != 40 or !std.mem.startsWith(u8, text, "GPU-")) return error.UnpinnedCudaDevices;
    var result: [16]u8 = undefined;
    var digit: usize = 0;
    for (text[4..], 0..) |char, i| {
        if (i == 8 or i == 13 or i == 18 or i == 23) {
            if (char != '-') return error.UnpinnedCudaDevices;
            continue;
        }
        const nibble = std.fmt.charToDigit(char, 16) catch return error.UnpinnedCudaDevices;
        if (digit % 2 == 0) result[digit / 2] = nibble << 4 else result[digit / 2] |= nibble;
        digit += 1;
    }
    return result;
}

fn formatUuid(a: std.mem.Allocator, bytes: [16]u8) ![]const u8 {
    const hex = std.fmt.bytesToHex(bytes, .lower);
    return std.fmt.allocPrint(a, "GPU-{s}-{s}-{s}-{s}-{s}", .{ hex[0..8], hex[8..12], hex[12..16], hex[16..20], hex[20..32] });
}

const first = "GPU-01234567-89ab-cdef-0123-456789abcdef";
const second = "GPU-fedcba98-7654-3210-fedc-ba9876543210";
const visible_pair = first ++ "," ++ second;

test "probe refuses absent or ordinal-only CUDA admission before opening a driver" {
    try std.testing.expectError(error.UnpinnedCudaDeviceOrder, Admission.parse(null, visible_pair, &.{0}, &.{first}));
    try std.testing.expectError(error.UnpinnedCudaDeviceOrder, Admission.parse("FASTEST_FIRST", visible_pair, &.{0}, &.{first}));
    try std.testing.expectError(error.UnpinnedCudaDevices, Admission.parse("PCI_BUS_ID", null, &.{0}, &.{first}));
    try std.testing.expectError(error.UnpinnedCudaDevices, Admission.parse("PCI_BUS_ID", "0,1", &.{0}, &.{first}));
}

test "probe refuses duplicate truncated and malformed physical UUID lists" {
    for ([_][]const u8{ first ++ "," ++ first, "GPU-01234567," ++ second, first, visible_pair ++ "," ++ first, first ++ ",GPU-fedcba98-7654-3210-fedc-ba987654321x" }) |visible| {
        try std.testing.expectError(error.UnpinnedCudaDevices, Admission.parse("PCI_BUS_ID", visible, &.{0}, &.{first}));
    }
}

test "probe binds each selected ordinal to an independently supplied expected physical UUID" {
    const admission = try Admission.parse("PCI_BUS_ID", visible_pair, &.{ 1, 0 }, &.{ second, first });
    try std.testing.expectEqual(@as(usize, 2), admission.len);
    try std.testing.expectError(error.DeviceIdentityMismatch, Admission.parse("PCI_BUS_ID", visible_pair, &.{0}, &.{second}));
    try std.testing.expectError(error.InvalidDeviceSelection, Admission.parse("PCI_BUS_ID", visible_pair, &.{2}, &.{first}));
    try std.testing.expectError(error.InvalidDeviceSelection, Admission.parse("PCI_BUS_ID", visible_pair, &.{-1}, &.{first}));
    try std.testing.expectError(error.InvalidDeviceSelection, Admission.parse("PCI_BUS_ID", visible_pair, &.{ 0, 0 }, &.{ first, first }));
    try std.testing.expectError(error.InvalidDeviceSelection, Admission.parse("PCI_BUS_ID", visible_pair, &.{0}, &.{}));
}
