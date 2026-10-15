//! Which queued request a lane host admits next: the head, or one behind a head that waits for pool room.
const api = @import("engine_api.zig");
const LaneHost = @import("lane_host.zig").LaneHost;
const Job = LaneHost.Job;
const prefix = @import("prefix_wait.zig");

/// The queued job to admit (its index), or null to hold; the caller holds the lock and a lane is free.
pub fn pick(h: *LaneHost) ?usize {
    if (waiting(h, h.queued.items[0])) {
        for (h.queued.items[1..], 1..) |job, i| {
            if (waiting(h, job)) continue;
            if (h.info_.admit) |a| if (!fits(a, job)) continue;
            return i;
        }
        return null;
    }
    const a = h.info_.admit orelse return 0;
    if (h.core.activeCount() == 0) return 0; // alone: the whole pool is free (kept prefixes give their blocks back)
    const head = h.queued.items[0];
    if (fits(a, head)) return 0;
    if (head.waits_since == null) head.waits_since = h.admissions;
    // it waits on the streams admitted before it waited: while one runs, later requests that fit go first; then none do
    for (h.admitted.items) |j| if (j.seq < head.waits_since.?) {
        for (h.queued.items[1..], 1..) |j2, i| if (fits(a, j2) and !waiting(h, j2)) return i;
        return null;
    };
    return null;
}

fn waiting(h: *LaneHost, job: *Job) bool {
    const store = h.cache orelse return false;
    if (!store.rules.shared_wait or job.request.untaught or job.request.prompt_logprobs != null) return false;
    const r = job.request;
    const spans = if (h.decoded_rows) r.decode_spans else &.{};
    const have = if (store.find(r.prompt, r.chunks, spans)) |e| e.at else 0;
    for (h.filling.items) |producer| {
        if (producer.request.untaught or producer.request.prompt_logprobs != null or producer.request.background) continue;
        if (prefix.pending(.{ .tokens = r.prompt, .spans = spans, .starts = r.chunks }, .{ .tokens = producer.request.prompt, .spans = if (h.decoded_rows) producer.request.decode_spans else &.{} }, producer.marks, @max(have, producer.kept_through), store.rules.lookahead, store.rules.grid)) |at| {
            if (store.family.vtable.bytes(store.family.ptr, at) <= store.budget) {
                if (job.prefix_wait_at == null) {
                    job.prefix_wait_at = at;
                    h.shared_prefix_waits += 1;
                }
                return true;
            }
        }
    }
    return false;
}

fn fits(a: api.Admit, job: *const Job) bool {
    return a.fits(a.ctx, @intCast(job.request.prompt.len), job.request.max_tokens);
}
