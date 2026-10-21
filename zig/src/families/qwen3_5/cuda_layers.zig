//! The Qwen3.5 CUDA engine's layers (their projections on the tiles of their widths) and its kernel tables.
const cuda = @import("cuda");

const Affine4 = cuda.qlinear.Affine4;
const AffineN = cuda.qlinear.AffineN;

/// MLX affine tiles or a native qlinear projection.
pub const Proj = union(enum) {
    a4: Affine4,
    an: AffineN,
    linear: cuda.qlinear.Weight,

    pub fn outputs(p: Proj) usize {
        return switch (p) {
            .linear => |q| q.outputs(),
            inline else => |q| q.n,
        };
    }

    pub fn inputs(p: Proj) usize {
        return switch (p) {
            .linear => |q| q.inputs(),
            inline else => |q| q.k,
        };
    }
};

/// Final normalization is owned independently of the optional vocabulary projection.
pub const TerminalWeights = union(enum) { none, vocabulary: Proj, hidden_only };

pub const Delta = struct { qkv: Proj, z: Proj, a: Proj, b: Proj, out: Proj, conv: u64, norm: u64, a_log: u64, dt_bias: u64 };
pub const Attention = struct { q: Proj, k: Proj, v: Proj, o: Proj, q_norm: u64, k_norm: u64 };
pub const Dense = struct { gate: Proj, up: Proj, down: Proj };
pub const SharedExpert = union(enum) { stacked, projected: Dense };
pub const Mlp = union(enum) { dense: Dense, moe: struct { router: u64, experts: cuda.experts.Layer, shared: SharedExpert = .stacked } };
pub const Layer = struct {
    input_norm: u64,
    post_norm: u64,
    mlp: Mlp,
    mixer: union(enum) { delta: Delta, attention: Attention },
};

/// qwen35.cu's kernels, each tf_q35_ and its name.
pub const Fns = struct {
    embed: cuda.Function,
    embed8: cuda.Function,
    norm: cuda.Function,
    sums: cuda.Function,
    gdn_pre: cuda.Function,
    conv_commit: cuda.Function,
    gdn_chain: cuda.Function,
    gdn_post: cuda.Function,
    head_norm: cuda.Function,
    rope: cuda.Function,
    attn_partial: cuda.Function,
    attn_merge: cuda.Function,
    attn_gate: cuda.Function,
    mlp_act: cuda.Function,
    argmax: cuda.Function,
    tap: cuda.Function,
    splice: cuda.Function,
};

/// qwen35_keep.cu's kernels (Cache.keep).
pub const Keeps = struct { apply: cuda.Function, conv: cuda.Function, compact: cuda.Function };

/// qwen35_kv8.cu's kernels: the cache writes, attention reads and keep's moves over e4m3 rows (--kv-cache fp8).
pub const Kv8 = struct { rope: cuda.Function, attn_partial: cuda.Function, compact: cuda.Function };
