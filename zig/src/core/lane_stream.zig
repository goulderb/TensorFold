//! A request's stream on the lane host: its kept prefix looked up, its drafter made, the stream set up.
const std = @import("std");
const lanes = @import("lanes");
const host = @import("lane_host.zig");
const recalls = @import("lane_host_recall.zig");
const pc = @import("prompt_cache.zig");
const LaneHost = host.LaneHost;
const Job = LaneHost.Job;

/// A job's drafter freed: the family's (LaneHost.proposers) or the built-in one.
pub fn freeProposer(h: *LaneHost, job: *Job) void {
    if (job.custom) |c| h.proposers.?.free(h.proposers.?.ctx, c) else job.proposer.deinit();
    job.custom = null;
}

/// An admitted job's stream (its kept prefix looked up, its drafter): false when it failed (the job is finished).
pub fn open(h: *LaneHost, job: *Job) bool {
    const r = job.request;
    var reuse: lanes.stream.Reuse = .{};
    // the entry stays alive until the backend restores it: nothing keeps between here and this stream's own pass
    job.prepareKey() catch return !h.drop(job, "out of memory");
    if (h.cache) |store| if (r.prompt_logprobs == null and !r.untaught) if (store.lookupRewind(h.gpa, job.cacheKey(), r.bornLen(), r.history_len, r.rewind_len, r.shared_prefixes, r.chunks, h.spans(r))) |l| {
        reuse = setLookup(job, store, l);
    } else |_| {};
    return startStream(h, job, reuse);
}

pub const Opening = enum { ready, pending, failed };

/// Start the cache lookup, retaining an unstarted job while a disk restore runs.
pub fn openAsync(h: *LaneHost, job: *Job) Opening {
    const r = job.request;
    job.prepareKey() catch {
        _ = h.drop(job, "out of memory");
        return .failed;
    };
    if (h.cache) |store| if (r.prompt_logprobs == null and !r.untaught) {
        const l = store.lookupRewindAsync(h.gpa, job.cacheKey(), r.bornLen(), r.history_len, r.rewind_len, r.shared_prefixes, r.chunks, h.spans(r), &job.pending_restore) catch |err| {
            if (err == error.RestorePending) return .pending;
            return if (startStream(h, job, .{})) .ready else .failed;
        };
        return if (startStream(h, job, setLookup(job, store, l))) .ready else .failed;
    };
    return if (startStream(h, job, .{})) .ready else .failed;
}

/// A pending restore has become a normal cache lookup; only now initialize the stream.
pub fn resumeAsync(h: *LaneHost, job: *Job, l: pc.Lookup) bool {
    return startStream(h, job, setLookup(job, h.cache.?, l));
}

fn setLookup(job: *Job, store: *pc.Store, l: pc.Lookup) lanes.stream.Reuse {
    job.entry = l.entry;
    job.kept0 = store.counts.kept;
    job.marks = l.marks;
    return .{ .saved = if (l.entry) |e| e.saved else null, .at = if (l.entry) |e| e.at else 0, .marks = l.marks, .hook = .{ .ptr = job, .at = Job.kept } };
}

fn startStream(h: *LaneHost, job: *Job, reuse: lanes.stream.Reuse) bool {
    const r = job.request;
    if (h.proposers) |ps| {
        job.custom = ps.make(ps.ctx, h.gpa, r.prompt, r.eos) catch return !h.drop(job, "the drafter could not start");
    } else job.proposer = lanes.SuffixLookup.init(h.gpa, .{ .min_match = h.min_match }) catch return !h.drop(job, "the drafter could not start");
    const recall = recalls.reader(h, job);
    job.stream = lanes.Stream.init(h.gpa, .{
        .id = "request",
        .prompt = r.prompt,
        .vision = r.vision,
        .max_new = r.max_tokens,
        .eos = r.eos,
        .sampling = r.sampling,
        .drafts = r.drafts,
        .proposer = job.custom orelse job.proposer.proposer(),
        .recall = recall,
        .stop_check = if (r.stop) |s| .{ .ptr = s.ctx, .check = s.check } else null,
        .cancel_check = .{ .ptr = job, .check = LaneHost.cancelled },
        .yield_check = if (r.background and r.max_tokens == 0) .{ .ptr = job, .check = LaneHost.yieldsTo } else null,
        .think_budget = r.think_budget,
        .think_close = r.think_close,
        .think_end = if (r.think_end) |t| t else -1,
        .logprobs = r.logprobs,
        .prompt_logprobs = r.prompt_logprobs,
        .loop_guard = r.loop_guard,
        .call = if (r.call) |c| .{ .opener = c.opener, .think_open = if (c.think_open) |t| @as(i64, t) else -1, .think_end = if (c.think_end) |t| @as(i64, t) else -1, .lead = c.lead, .names = c.names, .tail = c.tail, .lex = c.lex } else null,
        .chunks = r.chunks,
        .history_len = r.history_len,
        .shared_prefixes = r.shared_prefixes,
        .reuse = reuse,
        .grammar = if (r.structure) |st| h.constraint(st) catch |e| {
            freeProposer(h, job);
            recalls.release(job);
            return !h.drop(job, if (e == error.OutOfMemory) "out of memory" else "the reply's grammar could not start (xgrammar)");
        } else null,
        .decode_spans = h.spans(r),
        .decide = r.decide,
        .untaught = r.untaught,
        .images = r.images,
        .key = r.cache_key,
    }) catch {
        freeProposer(h, job);
        recalls.release(job);
        return !h.drop(job, "out of memory");
    };
    job.started = true;
    job.fill_on.store(true, .release); // status readers count the fill from here until the pass reports
    job.began = std.Io.Clock.awake.now(h.io).toNanoseconds();
    return true;
}
