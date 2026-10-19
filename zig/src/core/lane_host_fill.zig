//! Prefer foreground and short prompts, with bounded waiting for every admitted fill.
const LaneHost = @import("lane_host.zig").LaneHost;

pub const preference_rounds = 8;

/// The same priority for one chunk and a batch cohort; aging outranks prompt length and foreground priority.
pub fn before(h: *const LaneHost, a: *LaneHost.Job, b: *LaneHost.Job) bool {
    const aged = a.fill_wait >= preference_rounds;
    const best_aged = b.fill_wait >= preference_rounds;
    if (aged != best_aged) return aged;
    if (aged and a.fill_wait != b.fill_wait) return a.fill_wait > b.fill_wait;
    if (aged) return a.seq < b.seq;
    if (a.request.background != b.request.background) return !a.request.background;
    if (h.info_.fill_arrival) return a.seq < b.seq;
    const left = a.stream.prompt_len - a.stream.cached;
    const right = b.stream.prompt_len - b.stream.cached;
    return left < right or (left == right and a.seq < b.seq);
}

/// The filling job to give the next chunk: foreground before background, the shortest remaining prompt first (so a
/// short prompt never waits on a long one), the earliest admitted of equals with `Info.fill_arrival`. After
/// `preference_rounds` skipped choices, the oldest wait wins; each fill runs within eight plus the admitted lanes.
pub fn next(h: *const LaneHost) usize {
    var at: usize = 0;
    for (h.filling.items, 0..) |j, i| {
        if (before(h, j, h.filling.items[at])) at = i;
    }
    for (h.filling.items, 0..) |j, i| j.fill_wait = if (i == at) 0 else j.fill_wait +| 1;
    return at;
}

/// Takes `job` off the filling list, wherever it is.
pub fn unfill(h: *LaneHost, job: anytype) void {
    for (h.filling.items, 0..) |j, i| if (j == job) {
        _ = h.filling.orderedRemove(i);
        return;
    };
}
