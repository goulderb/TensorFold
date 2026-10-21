//! tfclef's C ABI (include/tfclef.h). This layer only translates: it parses the C record, serializes calls with the
//! busy and poison states, and maps engine errors to statuses. `Exports(E)` is generic so host tests run it whole.
const std = @import("std");
const record = @import("record.zig");

pub const version: u32 = 1;

pub const Status = enum(c_int) { ok = 0, invalid = 1, capacity = 2, device = 3, fatal = 4, busy = 5, nonfinite = 6 };

/// What an engine call may fail with. Device: a fault after which the engine drained and is reusable.
/// Fatal: a fault it could not drain; the handle is poisoned until closed.
pub const Error = error{ Invalid, Capacity, Device, Fatal, NonFinite };

pub fn status(e: Error) Status {
    return switch (e) {
        error.Invalid => .invalid,
        error.Capacity => .capacity,
        error.Device => .device,
        error.Fatal => .fatal,
        error.NonFinite => .nonfinite,
    };
}

pub const flag_head_only: u32 = 1;

pub const OpenOptions = extern struct {
    model_dir: ?[*:0]const u8,
    head_sha256: [32]u8,
    device_uuid: [16]u8,
    cap_bytes: u64,
    max_tokens: u32,
    flags: u32,
};

pub const Info = extern struct {
    abi_version: u32,
    max_tokens: u32,
    max_questions: u32,
    max_options: u32,
    budget_bytes: u64,
    device_free_bytes: u64,
    device_total_bytes: u64,
    device_uuid: [16]u8,
    source_commit: [41]u8,
    head_only: u8,
};

/// Validated open options in engine terms.
pub const Open = struct {
    model_dir: []const u8,
    head_sha256: [32]u8,
    device_uuid: [16]u8,
    cap_bytes: u64,
    max_tokens: u32,
    head_only: bool,
};

/// A bounded diagnostic an engine call fills; the handle keeps the last non-OK one.
pub const Message = struct {
    buf: [512]u8 = undefined,
    len: usize = 0,

    pub fn set(m: *Message, comptime fmt: []const u8, args: anytype) void {
        m.len = if (std.fmt.bufPrint(&m.buf, fmt, args)) |t| t.len else |_| m.buf.len;
    }

    pub fn text(m: *const Message) []const u8 {
        return m.buf[0..m.len];
    }
};

const State = enum(u8) { ready, busy, poisoned };

/// The engine contract: open(gpa, Open, *Message) !*E, limits(*E) Limits, decide(*E, Record, []f32, *Message)
/// Error!void, head(*E, Record, []const u16, []f32, *Message) Error!void, info(*E, *Info) void, close(*E, poisoned).
pub fn Exports(comptime E: type, comptime gpa: std.mem.Allocator) type {
    return struct {
        pub const Handle = struct {
            engine: *E,
            limits: record.Limits,
            scratch: record.Scratch,
            head_only: bool,
            state: std.atomic.Value(State) = .init(.ready),
            last: Message = .{},

            fn fail(h: *Handle, s: Status, comptime fmt: []const u8, args: anytype) Status {
                h.last.set(fmt, args);
                return s;
            }
        };

        pub fn abiVersion() callconv(.c) u32 {
            return version;
        }

        pub fn open(options: ?*const OpenOptions, err: ?[*]u8, err_len: usize) callconv(.c) ?*Handle {
            var m: Message = .{};
            const h = openChecked(options, &m) catch |e| {
                if (m.len == 0) m.set("{s}", .{@errorName(e)});
                if (err) |out| if (err_len > 0) {
                    const n = @min(m.len, err_len - 1);
                    @memcpy(out[0..n], m.buf[0..n]);
                    out[n] = 0;
                };
                return null;
            };
            return h;
        }

        fn openChecked(options: ?*const OpenOptions, m: *Message) !*Handle {
            const o = options orelse {
                m.set("open options are null", .{});
                return error.InvalidOptions;
            };
            const dir = o.model_dir orelse {
                m.set("model_dir is null", .{});
                return error.InvalidOptions;
            };
            if (o.flags & ~flag_head_only != 0) {
                m.set("unknown open flags 0x{x}", .{o.flags});
                return error.InvalidOptions;
            }
            if (o.max_tokens < 2 or o.max_tokens > record.max_tokens) {
                m.set("max_tokens {d} is outside 2..{d}", .{ o.max_tokens, record.max_tokens });
                return error.InvalidOptions;
            }
            const head_only = o.flags & flag_head_only != 0;
            const e = try E.open(gpa, .{ .model_dir = std.mem.span(dir), .head_sha256 = o.head_sha256, .device_uuid = o.device_uuid, .cap_bytes = o.cap_bytes, .max_tokens = o.max_tokens, .head_only = head_only }, m);
            errdefer e.close(false);
            const limits = e.limits();
            var scratch = try record.Scratch.alloc(gpa, limits);
            errdefer scratch.free(gpa);
            const h = try gpa.create(Handle);
            h.* = .{ .engine = e, .limits = limits, .scratch = scratch, .head_only = head_only };
            return h;
        }

        fn enter(h: *Handle) ?Status {
            const seen = h.state.cmpxchgStrong(.ready, .busy, .acquire, .monotonic) orelse return null;
            return if (seen == .poisoned)
                h.fail(.fatal, "the engine was poisoned by an earlier fatal fault; close it", .{})
            else
                h.fail(.busy, "another call is in progress", .{});
        }

        fn leave(h: *Handle, s: Status) Status {
            h.state.store(if (s == .fatal) .poisoned else .ready, .release);
            return s;
        }

        fn parse(h: *Handle, r: ?*const record.C.Record) union(enum) { ok: record.Record, refused: Status } {
            const c = r orelse return .{ .refused = h.fail(.invalid, "record is null", .{}) };
            const rec = record.Record.parse(c, h.limits, h.scratch) catch |e| return .{ .refused = h.fail(if (e == error.TooLong) .capacity else .invalid, "record refused: {s}", .{@errorName(e)}) };
            return .{ .ok = rec };
        }

        fn outputs(h: *Handle, rec: record.Record, logits: ?[*]f32, cap: u32) ?[]f32 {
            const out = logits orelse return null;
            if (cap < rec.options.len) return null;
            _ = h;
            return out[0..rec.options.len];
        }

        pub fn decide(handle: ?*Handle, r: ?*const record.C.Record, logits: ?[*]f32, cap: u32) callconv(.c) Status {
            const h = handle orelse return .invalid;
            if (enter(h)) |s| return s;
            return leave(h, decideEntered(h, r, logits, cap));
        }

        fn decideEntered(h: *Handle, r: ?*const record.C.Record, logits: ?[*]f32, cap: u32) Status {
            const rec = switch (parse(h, r)) {
                .ok => |x| x,
                .refused => |s| return s,
            };
            if (h.head_only) return h.fail(.invalid, "a head-only engine has no backbone; call tf_clef_head", .{});
            const out = outputs(h, rec, logits, cap) orelse return h.fail(.invalid, "logits need room for {d} options", .{rec.options.len});
            var m: Message = .{};
            h.engine.decide(rec, out, &m) catch |e| return keep(h, status(e), &m);
            return .ok;
        }

        pub fn head(handle: ?*Handle, r: ?*const record.C.Record, hidden: ?[*]const u16, logits: ?[*]f32, cap: u32) callconv(.c) Status {
            const h = handle orelse return .invalid;
            if (enter(h)) |s| return s;
            return leave(h, headEntered(h, r, hidden, logits, cap));
        }

        fn headEntered(h: *Handle, r: ?*const record.C.Record, hidden: ?[*]const u16, logits: ?[*]f32, cap: u32) Status {
            const rec = switch (parse(h, r)) {
                .ok => |x| x,
                .refused => |s| return s,
            };
            const rows = hidden orelse return h.fail(.invalid, "hidden rows are null", .{});
            const out = outputs(h, rec, logits, cap) orelse return h.fail(.invalid, "logits need room for {d} options", .{rec.options.len});
            var m: Message = .{};
            h.engine.head(rec, rows[0 .. rec.ids.len * hidden_width], out, &m) catch |e| return keep(h, status(e), &m);
            return .ok;
        }

        fn keep(h: *Handle, s: Status, m: *const Message) Status {
            h.last = m.*;
            if (h.last.len == 0) h.last.set("{s}", .{@tagName(s)});
            return s;
        }

        pub fn infoGet(handle: ?*Handle, out: ?*Info) callconv(.c) Status {
            const h = handle orelse return .invalid;
            const o = out orelse return .invalid;
            if (enter(h)) |s| return s;
            o.* = std.mem.zeroes(Info);
            o.abi_version = version;
            o.max_tokens = h.limits.tokens;
            o.max_questions = h.limits.maxQuestions();
            o.max_options = h.limits.maxOptions();
            o.head_only = @intFromBool(h.head_only);
            h.engine.info(o);
            return leave(h, .ok);
        }

        /// Copies the last non-OK message (NUL-terminated, truncated) and returns its full length.
        pub fn lastError(handle: ?*Handle, buf: ?[*]u8, len: usize) callconv(.c) usize {
            const h = handle orelse return 0;
            const text = h.last.text();
            if (buf) |out| if (len > 0) {
                const n = @min(text.len, len - 1);
                @memcpy(out[0..n], text[0..n]);
                out[n] = 0;
            };
            return text.len;
        }

        /// Drains and frees through the engine; a poisoned engine releases host state only. NULL is a no-op.
        pub fn close(handle: ?*Handle) callconv(.c) void {
            const h = handle orelse return;
            h.engine.close(h.state.load(.acquire) == .poisoned);
            h.scratch.free(gpa);
            gpa.destroy(h);
        }
    };
}

pub const hidden_width = 4096;

/// The C names of one `Exports` instance, for the library root or a host test binary.
pub fn exportAll(comptime X: type) void {
    @export(&X.abiVersion, .{ .name = "tf_clef_abi_version" });
    @export(&X.open, .{ .name = "tf_clef_open" });
    @export(&X.decide, .{ .name = "tf_clef_decide" });
    @export(&X.head, .{ .name = "tf_clef_head" });
    @export(&X.infoGet, .{ .name = "tf_clef_info_get" });
    @export(&X.lastError, .{ .name = "tf_clef_last_error" });
    @export(&X.close, .{ .name = "tf_clef_close" });
}
