//! Production-shaped GLM L2 range planning, adapted from jayleaton/glm53-tensorfold-spark patch 0460.
const std = @import("std");
const cuda = @import("cuda");
const b16 = @import("cuda_b16.zig");
const weights = @import("cuda_weights.zig");

pub const alignment: u64 = 16;
pub const default_piece: u64 = 32 << 10;
pub const max_q4_chunks: usize = 192;

pub const Mode = enum(u32) { bulk = 0, lines = 1, touch = 2 };
pub const Site = enum(u2) { a, f, o };

pub const Allocation = struct { base: u64, bytes: u64 };
pub const Tensor = struct { ptr: u64, bytes: u64, allocation: Allocation };
pub const Q4 = struct { n: usize, k: usize, split_k: usize };
pub const Group = struct { tensors: []const Tensor, weight: ?usize = null, q4: ?Q4 = null };
pub const Range = extern struct { addr: u64, bytes: u64 };
pub const PlannedSite = struct { first: u32, count: u32, grid: u32, bytes: u64 };

pub const Settings = struct {
    mode: ?Mode = null,
    mib: f64 = 8.0,
    sites: [3]bool = .{ true, true, true },
    max_rows: usize = 64,
    blocks: u32 = 0,
    threads: u32 = 128,
    piece: u64 = default_piece,

    pub const ParseError = error{ BadL2pfSetting, BadL2pfGeometry };

    pub fn parse(get: *const fn ([:0]const u8) ?[]const u8) ParseError!Settings {
        var s: Settings = .{};
        if (get("TF_GLM_L2PF")) |v| {
            if (std.mem.eql(u8, v, "0") or std.ascii.eqlIgnoreCase(v, "off")) {
                s.mode = null;
            } else if (std.mem.eql(u8, v, "1") or std.ascii.eqlIgnoreCase(v, "bulk")) {
                s.mode = .bulk;
            } else if (std.ascii.eqlIgnoreCase(v, "lines")) {
                s.mode = .lines;
            } else if (std.ascii.eqlIgnoreCase(v, "touch")) {
                s.mode = .touch;
            } else return error.BadL2pfSetting;
        }
        if (get("TF_GLM_L2PF_MB")) |v| s.mib = std.fmt.parseFloat(f64, v) catch return error.BadL2pfSetting;
        if (!(s.mib > 0 and s.mib <= 64)) return error.BadL2pfSetting;
        if (get("TF_GLM_L2PF_SITES")) |v| {
            s.sites = .{ false, false, false };
            if (v.len == 0) return error.BadL2pfSetting;
            for (v) |ch| switch (std.ascii.toLower(ch)) {
                'a' => s.sites[0] = true,
                'f' => s.sites[1] = true,
                'o' => s.sites[2] = true,
                else => return error.BadL2pfSetting,
            };
        }
        if (get("TF_GLM_L2PF_ROWS")) |v| s.max_rows = std.fmt.parseInt(usize, v, 10) catch return error.BadL2pfSetting;
        if (get("TF_GLM_L2PF_BLOCKS")) |v| s.blocks = std.fmt.parseInt(u32, v, 10) catch return error.BadL2pfSetting;
        if (get("TF_GLM_L2PF_THREADS")) |v| s.threads = std.fmt.parseInt(u32, v, 10) catch return error.BadL2pfSetting;
        if (get("TF_GLM_L2PF_KB")) |v| {
            const kb = std.fmt.parseInt(u64, v, 10) catch return error.BadL2pfSetting;
            s.piece = std.math.mul(u64, kb, 1024) catch return error.BadL2pfSetting;
        }
        if (s.max_rows == 0 or s.blocks > 65535 or s.threads < 32 or s.threads > 1024 or s.threads % 32 != 0 or s.piece < 1024 or s.piece > (1 << 20) or s.piece % alignment != 0) return error.BadL2pfGeometry;
        return s;
    }

    pub fn fromEnv() ParseError!Settings {
        const Env = struct {
            fn get(name: [:0]const u8) ?[]const u8 {
                return if (std.c.getenv(name)) |v| std.mem.span(v) else null;
            }
        };
        return parse(&Env.get);
    }

    pub fn budget(s: Settings) u64 {
        return @intFromFloat(s.mib * (1 << 20));
    }

    pub fn digest(s: Settings) [32]u8 {
        const mode = s.mode orelse return @splat(0);
        var h = std.crypto.hash.sha2.Sha256.init(.{});
        h.update("glm-l2pf-shared-first-v1\x00");
        h.update(@tagName(mode));
        inline for (.{ s.budget(), @as(u64, @intCast(s.max_rows)), s.blocks, s.threads, s.piece }) |v| h.update(std.mem.asBytes(&v));
        for (s.sites) |enabled| h.update(&.{@intFromBool(enabled)});
        return h.finalResult();
    }

    pub fn grid(s: Settings, piece_count: usize) u32 {
        if (s.blocks != 0) return s.blocks;
        const per: usize = if (s.mode.? == .bulk) s.threads else s.threads / 32;
        return @intCast(@max(1, @min(48, (piece_count + per - 1) / per)));
    }
};

fn span(t: Tensor, budget: u64) ?Range {
    if (t.bytes == 0 or t.ptr < t.allocation.base) return null;
    const offset = t.ptr - t.allocation.base;
    if (offset > t.allocation.bytes or t.bytes > t.allocation.bytes - offset) return null;
    const start = std.math.add(u64, t.ptr, alignment - 1) catch return null;
    const lo = start - start % alignment;
    const tensor_end = std.math.add(u64, t.ptr, t.bytes) catch return null;
    const allocation_end = std.math.add(u64, t.allocation.base, t.allocation.bytes) catch return null;
    const hi_unaligned = @min(tensor_end, allocation_end);
    const hi = hi_unaligned - hi_unaligned % alignment;
    if (hi <= lo) return null;
    const bytes = @min(hi - lo, budget - budget % alignment);
    return if (bytes == 0) null else .{ .addr = lo, .bytes = bytes };
}

fn q4Heads(gpa: std.mem.Allocator, out: *std.ArrayList(Range), weight: Tensor, q: Q4, budget: u64) !bool {
    if (q.n == 0 or q.k == 0 or q.k % 64 != 0 or q.split_k == 0) return false;
    const tiles = std.math.divCeil(usize, q.n, 64) catch return false;
    const chunks = std.math.mul(usize, tiles, q.split_k) catch return false;
    if (chunks == 0 or chunks > max_q4_chunks or weight.bytes % chunks != 0 or weight.ptr % alignment != 0 or weight.bytes == 0) return false;
    if (weight.ptr < weight.allocation.base) return false;
    const offset = weight.ptr - weight.allocation.base;
    if (offset > weight.allocation.bytes or weight.bytes > weight.allocation.bytes - offset) return false;
    const chunk_bytes = weight.bytes / chunks;
    if (chunk_bytes % alignment != 0) return false;
    const per = @min(chunk_bytes, budget / chunks) / alignment * alignment;
    if (per < 512) return false;
    for (0..chunks) |i| {
        const chunk_offset = std.math.mul(u64, @intCast(i), chunk_bytes) catch return false;
        const addr = std.math.add(u64, weight.ptr, chunk_offset) catch return false;
        const end = std.math.add(u64, addr, per) catch return false;
        const allocation_end = std.math.add(u64, weight.allocation.base, weight.allocation.bytes) catch return false;
        if (addr < weight.allocation.base or end > allocation_end) return false;
        try out.append(gpa, .{ .addr = addr, .bytes = per });
    }
    return true;
}

fn less(_: void, a: Tensor, b: Tensor) bool {
    return a.bytes < b.bytes;
}

pub fn ranges(gpa: std.mem.Allocator, out: *std.ArrayList(Range), groups: []const Group, budget_bytes: u64) !void {
    var left = budget_bytes;
    for (groups) |group| {
        const ordered = try gpa.dupe(Tensor, group.tensors);
        defer gpa.free(ordered);
        std.mem.sort(Tensor, ordered, {}, less);
        var heads: std.ArrayList(Range) = .empty;
        defer heads.deinit(gpa);
        var skip_weight = false;
        if (group.q4) |q| if (group.weight) |weight_index| {
            if (weight_index < group.tensors.len) {
                const weight = group.tensors[weight_index];
                var small: u64 = 0;
                for (group.tensors, 0..) |t, i| if (i != weight_index) {
                    small = std.math.add(u64, small, t.bytes) catch std.math.maxInt(u64);
                };
                if (small < left and try q4Heads(gpa, &heads, weight, q, left - small)) skip_weight = true;
            }
        };
        for (ordered) |t| {
            if (left < alignment) return;
            if (skip_weight and group.weight != null and t.ptr == group.tensors[group.weight.?].ptr) continue;
            if (span(t, left)) |r| {
                try out.append(gpa, r);
                left -= r.bytes;
            }
        }
        var heads_bytes: u64 = 0;
        for (heads.items) |h| heads_bytes = std.math.add(u64, heads_bytes, h.bytes) catch return error.BudgetOverflow;
        if (heads.items.len > 0 and heads_bytes <= left) {
            try out.appendSlice(gpa, heads.items);
            left -= heads_bytes;
        }
    }
}

pub fn pieces(gpa: std.mem.Allocator, out: *std.ArrayList(Range), spans: []const Range, piece: u64) !void {
    if (piece < alignment or piece % alignment != 0) return error.InvalidPiece;
    for (spans) |r| {
        if (r.addr % alignment != 0 or r.bytes % alignment != 0) return error.UnalignedRange;
        var offset: u64 = 0;
        while (offset < r.bytes) {
            const bytes = @min(piece, r.bytes - offset);
            try out.append(gpa, .{ .addr = try std.math.add(u64, r.addr, offset), .bytes = bytes });
            offset += bytes;
        }
    }
}

pub const Builder = struct {
    gpa: std.mem.Allocator,
    settings: Settings,
    rows: std.ArrayList(Range) = .empty,
    sites: std.ArrayList(?PlannedSite) = .empty,

    pub fn init(gpa: std.mem.Allocator, settings: Settings, layer_count: usize) !Builder {
        const n = std.math.mul(usize, layer_count, 3) catch return error.TooManyLayers;
        var b = Builder{ .gpa = gpa, .settings = settings };
        try b.sites.resize(gpa, n);
        @memset(b.sites.items, null);
        return b;
    }

    pub fn deinit(b: *Builder) void {
        b.rows.deinit(b.gpa);
        b.sites.deinit(b.gpa);
    }

    pub fn add(b: *Builder, layer: usize, site: Site, groups: []const Group) !void {
        if (layer * 3 + @backingInt(site) >= b.sites.items.len) return error.LayerOutOfRange;
        if (b.settings.mode == null or !b.settings.sites[@backingInt(site)] or groups.len == 0) return;
        var planned: std.ArrayList(Range) = .empty;
        defer planned.deinit(b.gpa);
        try ranges(b.gpa, &planned, groups, b.settings.budget());
        var total: u64 = 0;
        for (planned.items) |r| total = try std.math.add(u64, total, r.bytes);
        const first = b.rows.items.len;
        try pieces(b.gpa, &b.rows, planned.items, b.settings.piece);
        const count = b.rows.items.len - first;
        if (count == 0) return;
        if (first > std.math.maxInt(u32) or count > std.math.maxInt(u32)) return error.TableTooLarge;
        b.sites.items[layer * 3 + @backingInt(site)] = .{ .first = @intCast(first), .count = @intCast(count), .grid = b.settings.grid(count), .bytes = total };
    }
};

pub const Collector = struct {
    arena: std.mem.Allocator,
    buffers: []const cuda.DeviceBuffer,
    groups: *std.ArrayList(Group),

    fn tensor(c: Collector, ptr: u64) !?Tensor {
        if (ptr == 0) return null;
        for (c.buffers) |buffer| {
            const len: u64 = @intCast(buffer.len);
            if (ptr >= buffer.ptr and ptr - buffer.ptr < len) return .{
                .ptr = ptr,
                .bytes = len - (ptr - buffer.ptr),
                .allocation = .{ .base = buffer.ptr, .bytes = len },
            };
        }
        return error.WeightAllocationMissing;
    }

    fn append(c: Collector, pointers: []const u64, q4: ?Q4) !void {
        const tensors = try c.arena.alloc(Tensor, pointers.len);
        var count: usize = 0;
        for (pointers) |ptr| if (try c.tensor(ptr)) |t| {
            tensors[count] = t;
            count += 1;
        };
        if (count == 0) return;
        if (q4 != null and count != pointers.len) return error.Q4AllocationMissing;
        try c.groups.append(c.arena, .{
            .tensors = tensors[0..count],
            .weight = if (q4 != null) 0 else null,
            .q4 = if (count > 0) q4 else null,
        });
    }

    pub fn plain(c: Collector, pointers: []const u64) !void {
        try c.append(pointers, null);
    }

    pub fn norm(c: Collector, n: @import("cuda_glue.zig").Norm) !void {
        // Collect the actual allocation extent, including f32 storage; never narrow a norm for prefetch.
        switch (n) {
            .bf16, .f32 => |ptr| try c.plain(&.{ptr}),
        }
    }

    pub fn projection(c: Collector, p: @import("cuda_proj.zig").Proj) !void {
        switch (p) {
            .bf16 => |ptr| try c.plain(&.{ptr}),
            .fp8g, .mxfp8 => |q| try c.plain(&.{ q.codes, q.scales }),
            .source => |s| try c.plain(&.{ s.weights, s.jobs, s.groups }),
            .q => |weight| switch (weight) {
                .affine4 => |q| try c.append(&.{ q.w, q.s, q.b }, .{ .n = q.n, .k = q.k, .split_k = cuda.qlinear.splitK(q.n, q.k) }),
                .fp8g, .mxfp8, .nvfp4, .fp8, .bf16 => |q| try c.plain(&.{ q.codes, q.scales }),
                .exl3_trellis => |q| try c.plain(&.{ q.trellis, q.suh, q.svh }),
                .exl3 => |q| try c.plain(&.{ q.trellis, q.suh, q.svh, q.bias }),
                .ck8 => |q| try c.plain(&.{q.codes}),
                .ck4 => |q| try c.plain(&.{ q.words, q.scales }),
                .q8_0, .iq2xxs => |q| try c.plain(&.{q.w}),
            },
        }
    }
};

fn appendHc(c: Collector, hc: weights.Hc) !void {
    try c.plain(&.{ hc.fn_w, hc.base, hc.scale });
}

fn appendFfn(c: Collector, ffn: weights.Ffn) !void {
    switch (ffn) {
        .dense => |mlp| {
            switch (mlp.input) {
                .combined => |p| try c.projection(p),
                .separate => |pair| {
                    try c.projection(pair.gate);
                    try c.projection(pair.up);
                },
            }
            try c.projection(mlp.down);
        },
        .moe => |moe| {
            try c.plain(&.{ moe.router, moe.bias });
            if (moe.shared_gate) |gate| {
                try c.projection(.{ .q = gate });
                try c.projection(.{ .q = moe.shared_up orelse return error.MissingSharedUp });
            } else try c.projection(moe.shared_gu);
            try c.projection(moe.shared_down);
        },
    }
}

fn appendNextMixer(c: Collector, mixer: weights.Mixer) !void {
    switch (mixer) {
        .kda => |kda| {
            try c.projection(kda.fb);
            try c.projection(kda.gb);
            try c.plain(&.{ kda.conv_w, kda.a_log, kda.dt_bias });
            try c.projection(kda.proj);
            if (kda.qkv) |qkv| try c.projection(qkv);
        },
        .mla => |dsa| {
            try c.plain(&.{ dsa.q_norm, dsa.kv_norm });
            switch (dsa.input) {
                .combined => |p| try c.projection(p),
                .combined_bf16 => |ptr| try c.plain(&.{ptr}),
                .separate => |p| {
                    try c.projection(p.qa);
                    try c.projection(p.kva);
                },
            }
            try c.projection(dsa.q_b);
            try c.projection(dsa.index.kw);
            try c.plain(&.{dsa.index.gate});
        },
    }
}

fn appendOutput(c: Collector, mixer: weights.Mixer) !void {
    switch (mixer) {
        .kda => |kda| try c.projection(kda.o),
        .mla => |dsa| {
            try c.plain(&.{dsa.wv});
            try c.projection(dsa.o);
        },
    }
}

pub fn build(gpa: std.mem.Allocator, settings: Settings, model: *const weights.Weights) !Builder {
    var builder = try Builder.init(gpa, settings, model.layers.len);
    errdefer builder.deinit();
    if (settings.mode == null) return builder;
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    var groups: std.ArrayList(Group) = .empty;
    const c: Collector = .{ .arena = arena_state.allocator(), .buffers = model.buffers.items, .groups = &groups };
    for (model.layers, 0..) |layer, i| {
        if (settings.sites[0]) {
            groups.clearRetainingCapacity();
            try appendHc(c, layer.ffn_hc);
            try c.norm(layer.post_norm);
            try appendFfn(c, layer.ffn);
            try builder.add(i, .a, groups.items);
        }
        if (settings.sites[1]) {
            groups.clearRetainingCapacity();
            if (i + 1 < model.layers.len) {
                const next = model.layers[i + 1];
                try appendHc(c, next.attn_hc);
                try c.norm(next.in_norm);
                try appendNextMixer(c, next.mixer);
            } else {
                try c.norm(model.norm);
                try c.projection(model.head);
            }
            try builder.add(i, .f, groups.items);
        }
        if (settings.sites[2]) {
            groups.clearRetainingCapacity();
            try appendOutput(c, layer.mixer);
            try builder.add(i, .o, groups.items);
        }
    }
    return builder;
}

test "range planner keeps bytes inside allocation and applies the per-site budget" {
    const gpa = std.testing.allocator;
    var out: std.ArrayList(Range) = .empty;
    defer out.deinit(gpa);
    const t = Tensor{ .ptr = 0x1003, .bytes = 0x1000, .allocation = .{ .base = 0x1000, .bytes = 0x1003 } };
    try ranges(gpa, &out, &.{.{ .tensors = &.{t} }}, 1024);
    try std.testing.expectEqual(@as(usize, 1), out.items.len);
    try std.testing.expectEqual(@as(u64, 0x1010), out.items[0].addr);
    try std.testing.expectEqual(@as(u64, 1024), out.items[0].bytes);
    out.clearRetainingCapacity();
    const invalid = Tensor{ .ptr = 0x2000, .bytes = 64, .allocation = .{ .base = 0x1000, .bytes = 0x100 } };
    try ranges(gpa, &out, &.{.{ .tensors = &.{invalid} }}, 4096);
    try std.testing.expectEqual(@as(usize, 0), out.items.len);
}

test "Q4 one-wave heads respect chunk layout and bounded allocation" {
    const gpa = std.testing.allocator;
    var out: std.ArrayList(Range) = .empty;
    defer out.deinit(gpa);
    const w = Tensor{ .ptr = 0x1000000, .bytes = 2048 * 4096 / 2, .allocation = .{ .base = 0x1000000, .bytes = 2048 * 4096 / 2 } };
    const scale = Tensor{ .ptr = 0x2000000, .bytes = 2048 * 64 * 2, .allocation = .{ .base = 0x2000000, .bytes = 2048 * 64 * 2 } };
    const bias = Tensor{ .ptr = 0x3000000, .bytes = 2048 * 64 * 2, .allocation = .{ .base = 0x3000000, .bytes = 2048 * 64 * 2 } };
    const group: Group = .{ .tensors = &.{ w, scale, bias }, .weight = 0, .q4 = .{ .n = 2048, .k = 4096, .split_k = 4 } };
    try ranges(gpa, &out, &.{group}, 8 << 20);
    try std.testing.expectEqual(@as(usize, 130), out.items.len);
    for (out.items) |r| try std.testing.expect(r.addr >= 0x1000000 and r.addr + r.bytes <= 0x1000000 + w.bytes or r.addr >= scale.ptr and r.addr + r.bytes <= scale.ptr + scale.bytes or r.addr >= bias.ptr and r.addr + r.bytes <= bias.ptr + bias.bytes);
    var total: u64 = 0;
    for (out.items) |r| total += r.bytes;
    try std.testing.expect(total <= 8 << 20);
}

test "pieces and settings preserve the production defaults while defaulting off" {
    const Env = struct {
        var pairs: []const [2][]const u8 = &.{};
        fn get(name: [:0]const u8) ?[]const u8 {
            for (pairs) |pair| if (std.mem.eql(u8, pair[0], name)) return pair[1];
            return null;
        }
    };
    Env.pairs = &.{};
    const off = try Settings.parse(&Env.get);
    try std.testing.expect(off.mode == null);
    try std.testing.expectEqual(@as(u64, 8 << 20), off.budget());
    Env.pairs = &.{.{ "TF_GLM_L2PF", "1" }};
    try std.testing.expectEqual(Mode.bulk, (try Settings.parse(&Env.get)).mode.?);
    Env.pairs = &.{ .{ "TF_GLM_L2PF", "touch" }, .{ "TF_GLM_L2PF_SITES", "fo" } };
    const enabled = try Settings.parse(&Env.get);
    try std.testing.expectEqual(Mode.touch, enabled.mode.?);
    try std.testing.expect(!enabled.sites[0] and enabled.sites[1] and enabled.sites[2]);
    var pieces_out: std.ArrayList(Range) = .empty;
    defer pieces_out.deinit(std.testing.allocator);
    try pieces(std.testing.allocator, &pieces_out, &.{.{ .addr = 0x4000, .bytes = 100 << 10 }}, 32 << 10);
    try std.testing.expectEqual(@as(usize, 4), pieces_out.items.len);
    try std.testing.expectEqual(@as(u64, 4 << 10), pieces_out.items[3].bytes);
}
