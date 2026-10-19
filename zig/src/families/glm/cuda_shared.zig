//! GLM-5.3-Flash's shared rows: several streams in one forward, chains per segment.
//! Local extension: bounded prompt segments commit their own recurrence and convolution state.

const std = @import("std");
const cuda = @import("cuda");
const fw = @import("cuda_forward.zig");
const weights = @import("cuda_weights.zig");
const kda = @import("cuda_kda.zig");
const latent = @import("cuda_latent.zig");
const kda_block = @import("cuda_kda_block.zig");
const dsa_block = @import("cuda_dsa_block.zig");
const glue = @import("cuda_glue.zig");

/// Checks a shared window's segments against its `rows` rows, then each sequence holds its rows until its commit.
pub fn begin(r: fw.Rank, rows: usize) !void {
    var at: usize = 0;
    const limit: usize = if (r.b.prefill) fw.prefill_rows else fw.window_rows;
    for (r.segs, 0..) |g, i| {
        if (g.row0 != at or g.rows == 0 or g.rows > limit) return error.BadWindow;
        if (g.st.pos + g.rows > g.st.capacity) return error.PastCapacity;
        for (r.segs[0..i]) |prior| if (prior.st == g.st) return error.SlotTwice;
        at += g.rows;
    }
    if (at != rows) return error.BadWindow;
    for (r.segs) |g| {
        g.st.pending = g.rows;
        g.st.prompt = r.b.prefill;
    }
}

/// Shared mixer: prompts keep each KDA projection's original row count and arithmetic.
/// Decode projections and DSA preparation still span the shared window.
pub fn mixer(f: *const fw.Forward, r: fw.Rank, l: weights.Layer, li: usize, di: usize, rows: usize, out: u64) !void {
    const c = &r.w.config;
    const b = r.b;
    switch (l.mixer) {
        .kda => |k| {
            const n = k.heads * kda.dk;
            const w = k.width();
            var s = b.kda;
            s.p = b.kproj;
            if (!b.prefill) try kda_block.project(f.set(), f.stream, k, s, b.normed, c.hidden, rows);
            for (r.segs) |g| {
                const st = g.st;
                const cur = st.cur[li];
                var v = s;
                v.p = if (b.prefill) b.kproj + g.row0 * w * 2 else st.proj + li * st.proj_layer * 2;
                if (!b.prefill) try f.d.check(f.d.api.cuMemcpyDtoDAsync_v2(v.p, b.kproj + g.row0 * w * 2, g.rows * w * 2, f.stream.handle), "cuMemcpyDtoDAsync");
                v.ka += g.row0 * n * 2;
                v.kg += g.row0 * n * 2;
                v.saves.out += g.row0 * n * 2;
                v.saves.state_out = st.recAt(1 - cur, li);
                if (b.prefill) {
                    v.saves.k += g.row0 * n * 4;
                    v.saves.v += g.row0 * n * 2;
                    v.saves.g += g.row0 * n * 4;
                    v.saves.b += g.row0 * k.heads * 4;
                    v.wide.q += g.row0 * n * 4;
                    v.wide.y += g.row0 * n * 2;
                } else {
                    const kv = li * fw.window_rows * st.heads * kda.dk;
                    v.saves.k = st.save_k + kv * 4;
                    v.saves.v = st.save_v + kv * 2;
                    v.saves.g = st.save_g + kv * 4;
                    v.saves.b = st.save_b + li * fw.window_rows * st.heads * 4;
                }
                const conv = st.conv + li * 3 * st.conv_c * 2;
                // Q4 prompt/decode projection kernels round differently across row thresholds.
                if (b.prefill) try kda_block.project(f.set(), f.stream, k, v, b.normed + g.row0 * c.hidden * 2, c.hidden, g.rows);
                try kda_block.chainRows(&f.m.kda, f.stream, k, v, g.rows, conv, st.recAt(cur, li), !b.prefill and f.kda_wide);
                if (b.prefill) {
                    try glue.convShift(f.set(), f.stream, conv, v.p, g.rows, g.rows * w, w, 1, st.conv_c);
                    st.cur[li] = 1 - cur;
                    try kda_block.fill(f.set(), f.stream, k, v, 0, g.rows, out + g.row0 * c.hidden * 4);
                }
            }
            if (!b.prefill) try kda_block.fill(f.set(), f.stream, k, s, 0, rows, out);
        },
        .mla => |a| {
            var sparse_any = false;
            for (r.segs) |g| sparse_any = sparse_any or dsa_block.sparseRows(a, g.st.pos, g.rows);
            try dsa_block.prepare(f.set(), f.stream, a, b.dsa, b.normed, c.hidden, rows, sparse_any);
            for (r.segs) |g| {
                const nch = @min(latent.chunksFor(g.st.pos + g.rows), b.nch);
                try dsa_block.attend(f.dsaKernels(r), f.stream, a, g.st.caches[di], b.dsa, g.row0, g.rows, g.st.pos_dev, g.st.pos, nch, b.part_rows);
            }
            try latent.expandV(f.set(), f.stream, b.dsa.ol, a.wv, b.dsa.vn, rows, a.heads, a.dv, a.lw);
            try dsa_block.fill(f.set(), f.stream, a, b.dsa, 0, rows, out);
        },
    }
}
