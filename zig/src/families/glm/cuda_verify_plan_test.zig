const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const engine = @import("cuda_engine.zig");
const backend = @import("cuda_lanes.zig");
const batch = @import("cuda_dflash_batch.zig");
const state = @import("cuda_dflash_state.zig");
const verify = @import("cuda_verify_plan.zig");

test "verify plan scheduler facts are literal for baseline DFlash wider DFlash and MTP" {
    var e: engine.Engine = undefined;
    e.drafter = null;
    e.mtp = false;
    e.dflash = @ptrFromInt(256);
    e.settings = .{ .draft = .dflash2 };
    const b: backend.Lanes = .{ .x = .{ .e = &e, .link = undefined } };
    var cfg = try lanes.Config.init(std.testing.allocator, b.facts(), backend.max_rows, engine.max_drafts);
    defer cfg.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(u32, 4), cfg.family_width);
    try std.testing.expectEqual(@as(u32, 3), cfg.most_drafts);
    try std.testing.expectEqual(@as(u32, 32), cfg.batch_rows);
    try std.testing.expectEqual(@as(u32, 8), cfg.batch_streams);
    e.settings.dflash_verify = .dflash6;
    var wide = try lanes.Config.init(std.testing.allocator, b.facts(), backend.max_rows, engine.max_drafts);
    defer wide.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(u32, 7), wide.family_width);
    try std.testing.expectEqual(@as(u32, 6), wide.most_drafts);
    try std.testing.expectEqual(@as(u32, 56), wide.batch_rows);
    try std.testing.expectEqual(@as(u32, 8), wide.batch_streams);
    const probs = [_][]const f64{ &.{ 1, 1, 1, 1, 1, 1 }, &.{ 1, 1, 1, 1, 1, 1 }, &.{ 1, 1, 1, 1, 1, 1 }, &.{ 1, 1, 1, 1, 1, 1 }, &.{ 1, 1, 1, 1, 1, 1 }, &.{ 1, 1, 1, 1, 1, 1 }, &.{ 1, 1, 1, 1, 1, 1 }, &.{ 1, 1, 1, 1, 1, 1 } };
    const counts = try lanes.allocate.allocate(std.testing.allocator, &.{ 1, 1, 1, 1, 1, 1, 1, 1 }, &probs, &wide.shared_costs, 0, wide.batch_rows);
    defer std.testing.allocator.free(counts);
    try std.testing.expectEqualSlices(u32, &.{ 6, 6, 6, 6, 6, 6, 6, 6 }, counts);
    e.dflash = null;
    e.mtp = true;
    e.settings.draft = .mtp;
    try std.testing.expectEqual(@as(u32, 4), b.facts().exact_width);
    try std.testing.expectEqual(@as(u32, 3), b.facts().drafts);
    try std.testing.expectEqual(@as(u32, 32), b.facts().batch_rows);
}

test "verify plan eight shuffled lanes retain their packed graph rows" {
    var ls: [8]state.Lane = @splat(.{});
    for (&ls) |*l| l.end = 40;
    var requests: [8]batch.Request = undefined;
    for (&requests, [_]usize{ 7, 1, 5, 0, 6, 2, 4, 3 }) |*r, slot| r.* = .{ .slot = slot, .end = 40, .pending = @intCast(slot + 1), .depth = 6, .sampling = null };
    try std.testing.expectError(error.BadDraftRequest, batch.Plan.build(&ls, 1024, &requests));
    const p = try batch.Plan.buildWithPlan(&ls, 1024, &requests, .dflash6);
    try std.testing.expectEqual(@as(usize, 8), p.count);
    try std.testing.expectEqualSlices(usize, &.{ 0, 1, 2, 3, 4, 5, 6, 7 }, p.original[0..8]);
    const m = p.metadata(154856);
    try std.testing.expectEqualSlices(i64, &.{ 7, 1, 5, 0, 6, 2, 4, 3 }, m.block_slots[0..8]);
    try std.testing.expectEqualSlices(u32, &.{ 8, 154856, 154856, 154856, 154856, 154856, 154856, 154856 }, m.ids[0..8]);
    try std.testing.expectEqualSlices(i64, &.{ 40, 41, 42, 43, 44, 45, 46, 47 }, m.positions[56..64]);
    requests[7].depth = 7;
    try std.testing.expectError(error.BadDraftRequest, batch.Plan.buildWithPlan(&ls, 1024, &requests, .dflash6));
    try std.testing.expectEqual(@as(usize, 56), verify.Plan.dflash6.rows());
}

test "verify plan selector uses six existing candidates and truncates capacity tails" {
    var ls: [8]state.Lane = @splat(.{});
    ls[5].end = 40;
    ls[1].end = 42;
    const requests = [_]batch.Request{
        .{ .slot = 5, .end = 40, .pending = 11, .depth = 6, .sampling = null },
        .{ .slot = 1, .end = 42, .pending = 22, .depth = 6, .sampling = null },
    };
    var raw: [2 * 2 * 7 * 32]f32 = @splat(0);
    for (0..2) |rank| for (0..2) |item| for (0..7) |row| for (0..16) |j| {
        const base = ((rank * 2 + item) * 7 + row) * 32;
        raw[base + j] = if (rank == 1 and j == 0) 100 else -100;
        raw[base + 16 + j] = @bitCast(@as(u32, @intCast(100 + item * 10 + row * 16 + j)));
    };
    const projection: [2 * 7 * 256]f32 = @splat(0);
    const codes = try std.testing.allocator.alloc(f32, 512 * 256);
    defer std.testing.allocator.free(codes);
    @memset(codes, 0);
    const selector: @import("cuda_draft_policy.zig").Selector = .{ .pred = codes, .succ = codes };
    var out: [2]batch.Result = @splat(.{});
    const p = try batch.Plan.buildWithPlan(&ls, 1024, &requests, .dflash6);
    try batch.select(p, &raw, &projection, 2, selector, &out);
    try std.testing.expectEqualSlices(u32, &.{ 100, 116, 132, 148, 164, 180 }, out[0].tokens[0..out[0].count]);
    try std.testing.expectEqualSlices(u32, &.{ 110, 126, 142, 158, 174, 190 }, out[1].tokens[0..out[1].count]);
    const tail = try batch.Plan.buildWithPlan(&ls, 46, &requests, .dflash6);
    try batch.select(tail, &raw, &projection, 2, selector, &out);
    try std.testing.expectEqualSlices(u32, &.{ 100, 116, 132, 148, 164 }, out[0].tokens[0..out[0].count]);
    try std.testing.expectEqualSlices(u32, &.{ 110, 126, 142 }, out[1].tokens[0..out[1].count]);
    raw[(1 * 2 + 1) * 7 * 32 + 16] = @bitCast(@as(u32, batch.vocab));
    out = @splat(.{ .count = 99 });
    try std.testing.expectError(error.BadCandidates, batch.select(p, &raw, &projection, 2, selector, &out));
    try std.testing.expectEqual(@as(usize, 99), out[0].count);
    try std.testing.expectEqual(@as(usize, 99), out[1].count);
}

test "verify plan settings are opt in and differ in rank and numerical identities" {
    const ranks = @import("cuda_ranks.zig");
    const Env = struct {
        fn get(name: [:0]const u8) ?[]const u8 {
            if (std.mem.eql(u8, name, "TF_GLM_DRAFTER")) return "dflash2";
            if (std.mem.eql(u8, name, "TF_GLM_DFLASH_DIR")) return "/draft";
            return if (std.mem.eql(u8, name, "TF_GLM_DFLASH_VERIFY_DRAFTS")) "6" else null;
        }
        fn invalid(name: [:0]const u8) ?[]const u8 {
            return if (std.mem.eql(u8, name, "TF_GLM_DFLASH_VERIFY_DRAFTS")) "8" else null;
        }
    };
    var problem: []const u8 = "";
    const selected = try ranks.Settings.read(Env.get, &problem);
    try std.testing.expectEqual(verify.Plan.dflash6, selected.dflash_verify);
    try std.testing.expectError(error.BadSetting, ranks.Settings.read(Env.invalid, &problem));
    try std.testing.expectEqualStrings("TF_GLM_DFLASH_VERIFY_DRAFTS: 3, 6 or 7", problem);
    var base = selected;
    base.dflash_verify = .baseline;
    const zero: [32]u8 = @splat(0);
    const pair = [_]ranks.Fingerprint{ ranks.Fingerprint.of(2, 0, base, 8192, true, zero, zero, ranks.hashBytes("target")), ranks.Fingerprint.of(2, 1, selected, 8192, true, zero, zero, ranks.hashBytes("target")) };
    try std.testing.expectEqual(ranks.Fingerprint.Field.dflash_verify, ranks.mismatch(&pair).?.field);
    var a = std.crypto.hash.sha2.Sha256.init(.{});
    var b = std.crypto.hash.sha2.Sha256.init(.{});
    base.hashNumerics(&a);
    selected.hashNumerics(&b);
    try std.testing.expect(!std.mem.eql(u8, &a.finalResult(), &b.finalResult()));
    try std.testing.expectEqual(@as(usize, 80), 2 * engine.max_shared + engine.shared_rows);
    try std.testing.expect(2 * engine.max_shared + engine.shared_rows <= @import("cuda_follow.zig").put_ids);
}

test "seven draft plan advertises and fills all 64 target rows" {
    var e: engine.Engine = undefined;
    e.drafter = null;
    e.mtp = false;
    e.dflash = @ptrFromInt(256);
    e.settings = .{ .draft = .dflash2, .dflash_verify = .dflash7 };
    const b: backend.Lanes = .{ .x = .{ .e = &e, .link = undefined } };
    var cfg = try lanes.Config.init(std.testing.allocator, b.facts(), backend.max_rows, engine.max_drafts);
    defer cfg.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(u32, 8), cfg.family_width);
    try std.testing.expectEqual(@as(u32, 7), cfg.most_drafts);
    try std.testing.expectEqual(@as(u32, 64), cfg.batch_rows);
    const probs: [8][]const f64 = @splat(&.{ 1, 1, 1, 1, 1, 1, 1 });
    const counts = try lanes.allocate.allocate(std.testing.allocator, &.{ 1, 1, 1, 1, 1, 1, 1, 1 }, &probs, &cfg.shared_costs, 0, cfg.batch_rows);
    defer std.testing.allocator.free(counts);
    try std.testing.expectEqualSlices(u32, &.{ 7, 7, 7, 7, 7, 7, 7, 7 }, counts);
    var ls: [8]state.Lane = @splat(.{});
    var requests: [8]batch.Request = undefined;
    for (&ls) |*l| l.end = 40;
    for (&requests, 0..) |*r, i| r.* = .{ .slot = 7 - i, .end = 40, .pending = @intCast(i), .depth = 7, .sampling = null };
    const plan = try batch.Plan.buildWithPlan(&ls, 1024, &requests, .dflash7);
    try std.testing.expectEqual(@as(usize, 8), plan.count);
    try std.testing.expectEqual(@as(i64, 47), plan.metadata(154856).positions[63]);
    try std.testing.expectError(error.BadDraftRequest, batch.Plan.buildWithPlan(&ls, 1024, &requests, .dflash6));
    requests[7].depth = 8;
    try std.testing.expectError(error.BadDraftRequest, batch.Plan.buildWithPlan(&ls, 1024, &requests, .dflash7));
    const ranks = @import("cuda_ranks.zig");
    const six: ranks.Settings = .{ .dflash_verify = .dflash6 };
    var seven = six;
    seven.dflash_verify = .dflash7;
    var a = std.crypto.hash.sha2.Sha256.init(.{});
    var c = std.crypto.hash.sha2.Sha256.init(.{});
    six.hashNumerics(&a);
    seven.hashNumerics(&c);
    try std.testing.expect(!std.mem.eql(u8, &a.finalResult(), &c.finalResult()));
}

test "Mia decode scratch fits all shared verification boundary shapes" {
    const mia = @import("cuda_experts_mia.zig");
    for ([_]usize{ 32, 56, 63, 64 }) |rows| {
        const bytes = try mia.Scratch.bytes(rows, rows, 9, 4096, 1024);
        const scratch = try mia.Scratch.fromBase(0x1000, rows, rows, 9, 4096, 1024);
        try std.testing.expectEqual(rows, scratch.decode_capacity);
        try std.testing.expect(scratch.work < 0x1000 + bytes);
    }
}

test "canonical default shared verification allocates 64 rows and admits five eight-row windows" {
    const dflash = @import("dflash.zig");
    var e: engine.Engine = undefined;
    e.drafter = @ptrFromInt(8);
    e.dflash = null;
    e.mtp = true;
    e.settings = .{};
    e.bs = null;
    try std.testing.expectEqual(@as(usize, 64), e.verifyPlan().rows());
    try std.testing.expectEqual(@as(usize, 64), e.sharedLimit());
    try std.testing.expectEqual(@as(usize, 8), e.targetRows());
    const allocated = engine.Engine.sharedBufferRows(e.settings, true);
    try std.testing.expectEqual(@as(usize, 64), allocated);
    try std.testing.expectEqual(allocated, e.sharedLimit());
    try std.testing.expectEqual(@as(usize, 64), dflash.STEP_ROWS);
    var buf: @import("cuda_forward.zig").Buffers = undefined;
    buf.rows = allocated;
    e.bs = buf;
    try std.testing.expectEqual(@as(usize, 64), e.sharedCapacity());
    var rows: usize = 0;
    const windows: [5]usize = @splat(8);
    for (windows) |n| {
        try std.testing.expect(n > 0 and n <= e.targetRows());
        try std.testing.expect(rows + n <= e.sharedCapacity());
        rows += n;
    }
    try std.testing.expectEqual(@as(usize, 40), rows);
    try std.testing.expect(rows > 32);
    const baseline = e.settings.dflash_verify.active(false).rows();
    try std.testing.expectEqual(@as(usize, 32), baseline);
    try std.testing.expect(rows > baseline);
    buf.rows = baseline;
    e.bs = buf;
    try std.testing.expectEqual(baseline, e.sharedCapacity());
    try std.testing.expect(rows > e.sharedCapacity());
    var dr: dflash.Drafter = undefined;
    dr.gpa = std.testing.allocator;
    dr.slots = .empty;
    defer dr.slots.deinit(std.testing.allocator);
    dr.verifies = 0;
    const ring: cuda.DeviceBuffer = .{ .d = undefined, .ptr = 0x100000, .len = dflash.ring_bytes, .borrowed = true };
    try dr.slots.append(std.testing.allocator, .{ .ring = ring });
    try dr.noteVerify(0, 0, 40, 100, true);
    try std.testing.expectEqual(@as(usize, 40), dr.slots.items[0].v_rows);
    try std.testing.expectError(error.DrafterContextTooWide, dr.noteVerify(0, 0, 65, 100, true));
}
