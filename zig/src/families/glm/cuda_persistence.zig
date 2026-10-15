pub const Operation = enum { write, read, forget };
pub const Command = struct { operation: Operation, id: u64 = 0, key: u64, at: u32, tokens: []const u32 = &.{} };
pub const Port = struct {
    ctx: *anyopaque,
    tier: *anyopaque,
    identity: [32]u8,
    apply: *const fn (*anyopaque, Command) anyerror!void,
    request: *const fn (*anyopaque, Command) anyerror!void,
    flush: *const fn (*anyopaque) anyerror!void,
    forget: *const fn (*anyopaque, u64, u32) void,
    close: *const fn (*anyopaque) void,
};
