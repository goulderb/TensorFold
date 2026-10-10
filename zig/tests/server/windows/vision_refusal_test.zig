//! The Windows server refuses --vision before it looks for the model: the vision helper talks to Python over POSIX pipes,
//! so the Windows build has none. Run as a program against the built server:
//!
//!   tf-vision-refusal-test PATH_TO_TENSORFOLD_NATIVE
//!
//! The model named here does not exist, so only the refusal can answer: exit 1 with the refusal's message on stderr.

const std = @import("std");

const want_message = "--vision is not built for Windows";

pub fn main(init: std.process.Init) !u8 {
    const arena = init.arena.allocator();
    const argv = try init.minimal.args.toSlice(arena);
    if (argv.len != 2) {
        std.debug.print("usage: tf-vision-refusal-test PATH_TO_TENSORFOLD_NATIVE\n", .{});
        return 2;
    }
    const result = try std.process.run(init.gpa, init.io, .{
        .argv = &.{ argv[1], "serve", "tf-no-such-model-4e1c", "--vision" },
        .stderr_limit = .limited(64 * 1024),
        .stdout_limit = .limited(64 * 1024),
    });
    const code: ?u8 = switch (result.term) {
        .exited => |c| c,
        else => null,
    };
    const exit_ok = code != null and code.? == 1;
    const message_ok = std.mem.indexOf(u8, result.stderr, want_message) != null;
    std.debug.print("{s} the server exits 1 on --vision (got {?d})\n", .{ if (exit_ok) "PASS" else "FAIL", code });
    std.debug.print("{s} the refusal names --vision and the reason (stderr: {s})\n", .{ if (message_ok) "PASS" else "FAIL", std.mem.trim(u8, result.stderr, " \r\n") });
    const ok = exit_ok and message_ok;
    std.debug.print("{s}\n", .{if (ok) "PASS: 0 failed" else "FAIL"});
    return if (ok) 0 else 1;
}
