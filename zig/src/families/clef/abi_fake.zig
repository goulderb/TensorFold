//! A scripted engine for host tests of the C ABI layer: each call takes the next outcome and writes logits 0.5 * index.
const std = @import("std");
const record = @import("record.zig");
const abi = @import("abi.zig");
const Error = abi.Error;
const Status = abi.Status;
const Open = abi.Open;
const Message = abi.Message;
const Info = abi.Info;

pub const Fake = struct {
    pub var opened: usize = 0;
    pub var closed: usize = 0;
    pub var closed_poisoned: bool = false;
    pub var fail_open: bool = false;
    pub var outcomes: [8]?Error = @splat(null);
    pub var calls: usize = 0;
    pub var last_ids: usize = 0;
    pub var last_hidden: usize = 0;
    pub var last_option_question: [4]u16 = @splat(0);
    pub var reenter: ?*const fn () Status = null;
    pub var reentered: ?Status = null;

    gpa: std.mem.Allocator,
    head_only: bool,

    pub fn reset() void {
        opened = 0;
        closed = 0;
        closed_poisoned = false;
        fail_open = false;
        outcomes = @splat(null);
        calls = 0;
        reenter = null;
        reentered = null;
    }

    pub fn open(a: std.mem.Allocator, o: Open, m: *Message) !*Fake {
        if (fail_open) {
            m.set("device {x} not found", .{o.device_uuid[0]});
            return error.DeviceNotFound;
        }
        const f = try a.create(Fake);
        f.* = .{ .gpa = a, .head_only = o.head_only };
        opened += 1;
        return f;
    }

    pub fn limits(_: *Fake) record.Limits {
        return record.Limits.init(64, 100) catch unreachable;
    }

    fn step(rec: record.Record, out: []f32, m: *Message) Error!void {
        defer calls += 1;
        if (reenter) |f| reentered = f();
        last_ids = rec.ids.len;
        for (rec.option_question[0..@min(4, rec.option_question.len)], 0..) |q, i| last_option_question[i] = q;
        if (outcomes[calls]) |e| {
            m.set("scripted {s}", .{@errorName(e)});
            return e;
        }
        for (out, 0..) |*x, i| x.* = 0.5 * @as(f32, @floatFromInt(i));
    }

    pub fn decide(_: *Fake, rec: record.Record, out: []f32, m: *Message) Error!void {
        return step(rec, out, m);
    }

    pub fn head(_: *Fake, rec: record.Record, rows: []const u16, out: []f32, m: *Message) Error!void {
        last_hidden = rows.len;
        return step(rec, out, m);
    }

    pub fn info(_: *Fake, out: *Info) void {
        out.budget_bytes = 123;
        @memcpy(out.source_commit[0..4], "test");
    }

    pub fn close(f: *Fake, poisoned: bool) void {
        closed += 1;
        closed_poisoned = poisoned;
        f.gpa.destroy(f);
    }
};
