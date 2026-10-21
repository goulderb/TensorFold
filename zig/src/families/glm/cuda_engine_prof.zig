//! GLM profiler lifetime operation used by unwind and close.
const Engine = @import("cuda_engine.zig").Engine;
pub fn freeProfs(e: *Engine) void {
    for (&e.profs) |*p| {
        if (p.*) |v| v.deinit();
        p.* = null;
    }
    for (&e.prompt_profs) |*p| {
        if (p.*) |v| v.deinit();
        p.* = null;
    }
}

pub fn heatRoot() []const u8 {
    const s = @import("sys").getenv("TF_HEAT_ROOT") orelse return @import("heat").default_root;
    return if (s.len == 0) @import("heat").default_root else s;
}
