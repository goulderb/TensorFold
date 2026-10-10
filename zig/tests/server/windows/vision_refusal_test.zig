//! The Windows server refuses --vision before it looks for the model: the vision helper talks to Python over POSIX pipes,
//! so the Windows build has none. Run as a program against the built server:
//!
//!   tf-vision-refusal-test PATH_TO_TENSORFOLD_NATIVE
//!
//! The model named here does not exist, so only the refusal can answer: exit 1 with the refusal's message on stderr.
//! The red control runs the same model without --vision: the refusal's message must be absent there.

const std = @import("std");

const want_message = "--vision is not built for Windows";

fn run(init: std.process.Init, server: []const u8, with_vision: bool) !std.process.RunResult {
    const argv: []const []const u8 = if (with_vision)
        &.{ server, "serve", "tf-no-such-model-4e1c", "--vision" }
    else
        &.{ server, "serve", "tf-no-such-model-4e1c" };
    return std.process.run(init.gpa, init.io, .{
        .argv = argv,
        .stderr_limit = .limited(64 * 1024),
        .stdout_limit = .limited(64 * 1024),
    });
}

pub fn main(init: std.process.Init) !u8 {
    const arena = init.arena.allocator();
    const argv = try init.minimal.args.toSlice(arena);
    if (argv.len != 2) {
        std.debug.print("usage: tf-vision-refusal-test PATH_TO_TENSORFOLD_NATIVE\n", .{});
        return 2;
    }
    const result = try run(init, argv[1], true);
    const code: ?u8 = switch (result.term) {
        .exited => |c| c,
        else => null,
    };
    const exit_ok = code != null and code.? == 1;
    const message_ok = std.mem.indexOf(u8, result.stderr, want_message) != null;
    std.debug.print("{s} the server exits 1 on --vision (got {?d})\n", .{ if (exit_ok) "PASS" else "FAIL", code });
    std.debug.print("{s} the refusal names --vision and the reason (stderr: {s})\n", .{ if (message_ok) "PASS" else "FAIL", std.mem.trim(u8, result.stderr, " \r\n") });
    const control = try run(init, argv[1], false);
    const control_ok = std.mem.indexOf(u8, control.stderr, want_message) == null;
    std.debug.print("{s} red control: without --vision the refusal is absent\n", .{if (control_ok) "PASS" else "FAIL"});
    const ok = exit_ok and message_ok and control_ok;
    std.debug.print("{s}\n", .{if (ok) "PASS: 0 failed" else "FAIL"});
    return if (ok) 0 else 1;
}
