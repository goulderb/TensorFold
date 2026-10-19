//! Exact-width target verification, using eight slots and deterministic reset/replay references.
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm_cuda");
const check = @import("check.zig");

pub fn run(gpu: check.Gpu, checkpoint: []const u8) !void {
    var arena = std.heap.ArenaAllocator.init(gpu.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const place = try glm.native.place();
    var problem: []const u8 = "";
    const settings = try glm.ranks.Settings.read(&glm.ranks.environment, &problem);
    var lib = try cuda.nccl.Library.open();
    var comm_id: cuda.nccl.UniqueId = undefined;
    try lib.check(lib.api.ncclGetUniqueId(&comm_id), "ncclGetUniqueId");
    var comm: cuda.nccl.Comm = null;
    try lib.check(lib.api.ncclCommInitRank(&comm, @intCast(place.world), comm_id, @intCast(place.rank)), "ncclCommInitRank");
    const e = try glm.engine.Engine.init(gpu.gpa, gpu.io, gpu.d, gpu.ctx, checkpoint, &lib, comm, place.rank, place.world, .{ .capacity = 2048, .mtp = true, .settings = settings, .draft_kernels = glm.ranks.environment("TF_GLM_DFLASH_KERNELS"), .partial_dflash_check = true });
    var link = try glm.follow.Link.init(gpu.gpa, gpu.d, &lib, comm, e.stream, 2048);
    const x: glm.follow.Leader = .{ .e = e, .link = &link };
    if (e.verifyPlan() != .dflash7 or e.pool != null) return error.BadBoundaryProofProfile;
    // Trained prefixes omit some tap layers; keep those planes deterministic for the replay.
    for ([_]*glm.forward.Buffers{ &e.bp, &e.bd, &e.bm, &e.bs.? }) |b| {
        if (b.taps) |tap| try gpu.d.check(gpu.d.api.cuMemsetD8Async(tap.ptr, 0, tap.len, e.stream.handle), "clear proof taps");
    }
    if (place.rank != 0) return glm.follow.follow(e, &link);
    defer x.end() catch {};
    const vocab = e.share.vocab.len();
    const shapes = [_][8]usize{
        .{ 4, 4, 4, 4, 4, 4, 4, 4 },
        .{ 7, 7, 7, 7, 7, 7, 7, 7 },
        .{ 8, 8, 8, 8, 8, 8, 8, 7 },
        .{ 8, 8, 8, 8, 8, 8, 8, 8 },
    };
    const slots = [_]u32{ 7, 1, 5, 0, 6, 2, 4, 3 };
    for (shapes) |shape| {
        var total: usize = 0;
        for (shape) |n| total += n;
        var ids: [8][8]u32 = undefined;
        var outputs: [8][8]u32 = undefined;
        var drafts: [8][7]u32 = undefined;
        var draft_counts: [8]usize = undefined;
        var draft_hashes: [8][32]u8 = undefined;
        var reference_logits: [8][]u16 = undefined;
        var reference_state: [8][]u8 = undefined;
        var reference_pos: [8]usize = undefined;
        var reference_cur: [8][]u1 = undefined;
        var parts: [8]glm.engine.Part = undefined;
        for (&parts, shape, slots, 0..) |*p, n, slot, i| {
            for (&ids[i], 0..) |*id, r| id.* = @intCast(100 + (i * 23 + r * 13) % 1000);
            p.* = .{ .slot = slot, .ids = ids[i][0..n], .out = outputs[i][0..n] };
        }
        for (0..2) |pass| {
            for (slots, 0..) |slot, i| {
                try x.bind(slot);
                const st = &e.cur.st;
                try gpu.d.check(gpu.d.api.cuMemsetD8Async(st.arena.buf.ptr, 0, st.arena.buf.len, e.stream.handle), "clear proof state");
                try x.begin(null);
                var prompt: [58]u32 = undefined;
                for (&prompt, 0..) |*id, r| id.* = @intCast(10 + (i * 17 + r * 7) % 1000);
                _ = try x.prefill(&prompt, null, null, null);
            }
            if (pass == 0) try x.verifyShared(&parts);
            var row0: usize = 0;
            for (parts, shape, slots, 0..) |p, n, slot, i| {
                try x.bind(slot);
                var lone: [8]u32 = undefined;
                if (pass == 1) try x.verify(p.ids, lone[0..n]);
                try e.stream.synchronize();
                const logits = try a.alloc(u16, n * vocab);
                const source = if (pass == 0) e.bs.?.logits + row0 * vocab * 2 else e.bd.logits;
                try gpu.d.check(gpu.d.api.cuMemcpyDtoH_v2(logits.ptr, source, logits.len * 2), "proof logits");
                if (pass == 0) reference_logits[i] = logits else {
                    try check.expect(std.mem.eql(u16, reference_logits[i], logits), "{d} shared rows: lane {d} logits equal lone", .{ total, slot });
                    try check.expect(std.mem.eql(u32, p.out, lone[0..n]), "{d} shared rows: lane {d} tokens equal lone", .{ total, slot });
                }
                const keep = if (i % 2 == 0) n else 1;
                try x.keep(keep);
                var held: [7]u32 = undefined;
                const count = try x.draft(false, p.out[0..keep], 7, &held);
                const drafter = e.dflash.?;
                var hash = std.crypto.hash.sha2.Sha256.init(.{});
                hash.update(std.mem.sliceAsBytes(drafter.forward.candidatesHost(1)));
                hash.update(std.mem.sliceAsBytes(drafter.forward.projectionHost(1)));
                const raw = hash.finalResult();
                if (pass == 0) {
                    @memcpy(drafts[i][0..count], held[0..count]);
                    draft_counts[i] = count;
                    draft_hashes[i] = raw;
                } else {
                    try check.expect(count == draft_counts[i] and std.mem.eql(u32, drafts[i][0..count], held[0..count]), "{d} shared rows: lane {d} adaptive drafts equal lone", .{ total, slot });
                    try check.expect(std.mem.eql(u8, &draft_hashes[i], &raw), "{d} shared rows: lane {d} all seven draft candidate/projection rows equal lone", .{ total, slot });
                }
                try e.stream.synchronize();
                const st = &e.cur.st;
                const bytes = try a.alloc(u8, st.arena.buf.len);
                try st.arena.buf.download(0, bytes);
                if (pass == 0) {
                    reference_state[i] = bytes;
                    reference_pos[i] = st.pos;
                    reference_cur[i] = try a.dupe(u1, st.cur);
                } else {
                    try check.expect(st.pending == 0 and reference_pos[i] == st.pos and std.mem.eql(u1, reference_cur[i], st.cur), "{d} shared rows: lane {d} committed position/parity equal lone", .{ total, slot });
                    try check.expect(std.mem.eql(u8, reference_state[i], bytes), "{d} shared rows: lane {d} committed state equals lone", .{ total, slot });
                }
                row0 += n;
            }
        }
        try check.expect(e.verify_widths[total] > 0, "actual successful verification width {d}", .{total});
        check.pass("shared width {d}: logits, tokens, seven-row draft readback, adaptive proposals and full committed state equal replay", .{total});
    }
    if (e.settings.exl3_routed == .mia_mcg4) {
        const scratch = e.bs.?.moe.mia_scratch orelse return error.MiaNotExercised;
        try check.expect(scratch.decode_capacity == 64, "Mia shared decode capacity is 64", .{});
        var layers: usize = 0;
        for (e.w.layers) |layer| switch (layer.ffn) {
            .moe => |moe| {
                if (moe.mia_routed != null) layers += 1;
            },
            .dense => {},
        };
        try check.expect(layers > 0, "Mia expert layers executed at width 64", .{});
    }
}
