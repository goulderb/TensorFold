// Clef's joint schema head: the kernels its forward needs beyond vit.cu's LayerNorm, bias and pointwise kernels and
// the BF16 lane matmul (cuda/clef/head.zig). The arithmetic reimplements JointSchemaHead from Cloudflare's clef-flash
// joint_schema_model.py (Apache-2.0, revision fde727a2); no code is copied. Every value torch's eager BF16 graph
// materializes is rounded to BF16 at the same point; reductions run in FP32 in a fixed order.

#include <cuda_bf16.h>
#include <math.h>
#include <stdint.h>

#include "glue_math.cuh"

using namespace tf_glue;

namespace {

constexpr int NT = 256;
constexpr int HEAD_DIM = 64;
// The memory-efficient SDPA kernel torch picks for BF16 at head dim 64 on sm_80+: 64 keys a tile, max and sum online.
constexpr int KEYS = 64;
constexpr float LOG2E = 1.4426950408889634074f;

struct Span {
    uint32_t start, end;
};
struct Range {
    uint32_t first, count;
};

__device__ __forceinline__ __nv_bfloat16 to_bf(float v) { return __float2bfloat16_rn(v); }
__device__ __forceinline__ float neg_inf() { return __int_as_float(0xff800000u); }

// Every thread's value summed in a fixed order: warps by butterfly, then the warps in order. `red`: NT / 32 floats.
// vit.cu's block_sum.
__device__ __forceinline__ float block_sum(float v, float* red) {
    v = warp_sum(v);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
    __syncthreads();
    float s = red[0];
    for (int w = 1; w < NT / 32; ++w) s = __fadd_rn(s, red[w]);
    __syncthreads();
    return s;
}

__device__ __forceinline__ float block_max(float v, float* red) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, off));
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
    __syncthreads();
    float m = red[0];
    for (int w = 1; w < NT / 32; ++w) m = fmaxf(m, red[w]);
    __syncthreads();
    return m;
}

// torch's linalg_vector_norm(2) of a row: the FP32 sum of squares, its square root, one BF16 rounding.
__device__ __forceinline__ float row_norm(float squares, float* red) { return rbf(__fsqrt_rn(block_sum(squares, red))); }

}  // namespace

// hidden_norm, 4096 wide: vit.cu's tf_vit_layer_norm operation for operation (y = bf16((x - mean) * rstd * w + b),
// mean and variance in two fixed-order passes), holding 16 elements a thread where vit.cu holds 8. vit.cu itself is
// unchanged: sharing its body through a header changed sm_120 code generation.
extern "C" __global__ void __launch_bounds__(NT) tf_clef_layer_norm_4096(const __nv_bfloat16* __restrict__ X, const __nv_bfloat16* __restrict__ W,
                                                                         const __nv_bfloat16* __restrict__ B, __nv_bfloat16* __restrict__ Y, int D,
                                                                         float eps) {
    constexpr int MAXE = 16;
    __shared__ float red[NT / 32];
    const int64_t at = static_cast<int64_t>(blockIdx.x) * D;
    float x[MAXE];
    float acc = 0.0f;
#pragma unroll
    for (int k = 0; k < MAXE; ++k) {
        const int i = threadIdx.x + k * NT;
        if (i >= D) continue;
        x[k] = bf(X[at + i]);
        acc = __fadd_rn(acc, x[k]);
    }
    const float mean = __fdiv_rn(block_sum(acc, red), static_cast<float>(D));
    acc = 0.0f;
#pragma unroll
    for (int k = 0; k < MAXE; ++k) {
        if (threadIdx.x + k * NT >= D) continue;
        x[k] = __fsub_rn(x[k], mean);
        acc = __fadd_rn(acc, __fmul_rn(x[k], x[k]));
    }
    const float var = __fdiv_rn(block_sum(acc, red), static_cast<float>(D));
    const float rstd = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(var, eps)));
#pragma unroll
    for (int k = 0; k < MAXE; ++k) {
        const int i = threadIdx.x + k * NT;
        if (i < D) Y[at + i] = __float2bfloat16_rn(__fadd_rn(__fmul_rn(__fmul_rn(x[k], rstd), bf(W[i])), bf(B[i])));
    }
}

// Block r: y[r] = bf16(sum_{i in span r} x[i] * (1 / len)), the sum FP32 in index order. torch's mean over dim 0
// multiplies its FP32 sum by the float reciprocal of the count.
extern "C" __global__ void __launch_bounds__(NT) tf_clef_span_mean(const __nv_bfloat16* __restrict__ X, int width, const Span* __restrict__ spans,
                                                                   __nv_bfloat16* __restrict__ Y) {
    const Span s = spans[blockIdx.x];
    const float inv = __fdiv_rn(1.0f, static_cast<float>(s.end - s.start));
    for (int c = threadIdx.x; c < width; c += NT) {
        float acc = 0.0f;
        for (uint32_t i = s.start; i < s.end; ++i) acc = __fadd_rn(acc, bf(X[static_cast<int64_t>(i) * width + c]));
        Y[static_cast<int64_t>(blockIdx.x) * width + c] = to_bf(__fmul_rn(acc, inv));
    }
}

// Block o: y[o] = bf16(bf16(a[o] + b[o]) + c[question(o)]), the option queries' two adds in torch's order.
extern "C" __global__ void __launch_bounds__(NT) tf_clef_option_queries(const __nv_bfloat16* __restrict__ A, const __nv_bfloat16* __restrict__ Bv,
                                                                        const __nv_bfloat16* __restrict__ C, const uint32_t* __restrict__ question,
                                                                        __nv_bfloat16* __restrict__ Y, int width) {
    const int64_t o = blockIdx.x;
    const int64_t q = question[o];
    for (int c = threadIdx.x; c < width; c += NT) {
        const float ab = rbf(__fadd_rn(bf(A[o * width + c]), bf(Bv[o * width + c])));
        Y[o * width + c] = to_bf(__fadd_rn(ab, bf(C[q * width + c])));
    }
}

// Block (query, head), 64 threads: unmasked attention of one query row over lk keys at head dim 64, rows `width`
// apart with head h at column 64 h. It follows the memory-efficient SDPA forward torch selects for these shapes:
// FP32 scores scaled by `scale`, then by log2(e); 64-key tiles with an online maximum; P = exp2(s - m) rounded to
// BF16 before P.V; the row sum from the unrounded P; both rescaled by exp2(m_old - m_new) at each tile; the output
// times 1 / sum, rounded to BF16.
extern "C" __global__ void __launch_bounds__(HEAD_DIM) tf_clef_attention(const __nv_bfloat16* __restrict__ Q, const __nv_bfloat16* __restrict__ K,
                                                                         const __nv_bfloat16* __restrict__ V, __nv_bfloat16* __restrict__ O, int lk,
                                                                         int width, float scale) {
    __shared__ float q[HEAD_DIM];
    __shared__ float p[KEYS];
    __shared__ float red[2];
    const int64_t row = blockIdx.x;
    const int col = blockIdx.y * HEAD_DIM;
    const int j = threadIdx.x;
    q[j] = bf(Q[row * width + col + j]);
    __syncthreads();
    float m = neg_inf();
    float sum = 0.0f;
    float acc = 0.0f;
    for (int base = 0; base < lk; base += KEYS) {
        const int key = base + j;
        float s = neg_inf();
        if (key < lk) {
            const __nv_bfloat16* k = K + static_cast<int64_t>(key) * width + col;
            float dot = 0.0f;
#pragma unroll 16
            for (int d = 0; d < HEAD_DIM; ++d) dot = __fmaf_rn(q[d], bf(k[d]), dot);
            s = __fmul_rn(__fmul_rn(dot, scale), LOG2E);
        }
        float tile = s;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) tile = fmaxf(tile, __shfl_xor_sync(0xffffffffu, tile, off));
        if ((j & 31) == 0) red[j >> 5] = tile;
        __syncthreads();
        const float m_new = fmaxf(m, fmaxf(red[0], red[1]));
        const float rescale = exp2f(__fsub_rn(m, m_new));
        const float e = key < lk ? exp2f(__fsub_rn(s, m_new)) : 0.0f;
        p[j] = e;
        __syncthreads();
        float tile_sum = 0.0f;
        for (int i = 0; i < KEYS; ++i) tile_sum = __fadd_rn(tile_sum, p[i]);
        sum = __fadd_rn(__fmul_rn(sum, rescale), tile_sum);
        acc = __fmul_rn(acc, rescale);
        const int n = min(KEYS, lk - base);
        for (int i = 0; i < n; ++i) acc = __fmaf_rn(rbf(p[i]), bf(V[static_cast<int64_t>(base + i) * width + col + j]), acc);
        m = m_new;
        __syncthreads();
    }
    O[row * width + col + j] = to_bf(__fmul_rn(__fdiv_rn(1.0f, sum), acc));
}

// Block q over its options [first, first + count): w = softmax_o(bf16(bf16(options[o] . field) / 32)), then
// summary = bf16(sum_o bf16(w[o] * options[o])), each torch op's BF16 result in its order. `w`: count floats.
extern "C" __global__ void __launch_bounds__(NT) tf_clef_segment_summary(const __nv_bfloat16* __restrict__ options, const __nv_bfloat16* __restrict__ fields,
                                                                         const Range* __restrict__ ranges, __nv_bfloat16* __restrict__ summary,
                                                                         int width, float divisor) {
    extern __shared__ float w[];
    __shared__ float red[NT / 32];
    const Range r = ranges[blockIdx.x];
    const __nv_bfloat16* field = fields + static_cast<int64_t>(blockIdx.x) * width;
    const int lane = threadIdx.x & 31;
    for (uint32_t o = threadIdx.x >> 5; o < r.count; o += NT / 32) {
        const __nv_bfloat16* x = options + static_cast<int64_t>(r.first + o) * width;
        float dot = 0.0f;
        for (int c = lane; c < width; c += 32) dot = __fmaf_rn(bf(x[c]), bf(field[c]), dot);
        dot = warp_sum(dot);
        if (lane == 0) w[o] = rbf(__fdiv_rn(rbf(dot), divisor));
    }
    __syncthreads();
    float m = neg_inf();
    for (uint32_t o = threadIdx.x; o < r.count; o += NT) m = fmaxf(m, w[o]);
    m = block_max(m, red);
    float total = 0.0f;
    for (uint32_t o = threadIdx.x; o < r.count; o += NT) total = __fadd_rn(total, expf(__fsub_rn(w[o], m)));
    total = block_sum(total, red);
    for (uint32_t o = threadIdx.x; o < r.count; o += NT) w[o] = rbf(__fdiv_rn(expf(__fsub_rn(w[o], m)), total));
    __syncthreads();
    for (int c = threadIdx.x; c < width; c += NT) {
        float acc = 0.0f;
        for (uint32_t o = 0; o < r.count; ++o) acc = __fadd_rn(acc, rbf(__fmul_rn(w[o], bf(options[static_cast<int64_t>(r.first + o) * width + c]))));
        summary[static_cast<int64_t>(blockIdx.x) * width + c] = to_bf(acc);
    }
}

// Block q: fields = bf16(bf16(bf16(base + summary_norm) + global) + type_embedding[type]), torch's left-to-right adds.
extern "C" __global__ void __launch_bounds__(NT) tf_clef_fields(const __nv_bfloat16* __restrict__ base, const __nv_bfloat16* __restrict__ summary,
                                                                const __nv_bfloat16* __restrict__ global, const __nv_bfloat16* __restrict__ types_table,
                                                                const uint32_t* __restrict__ types, __nv_bfloat16* __restrict__ Y, int width) {
    const int64_t q = blockIdx.x;
    const int64_t type = types[q];
    for (int c = threadIdx.x; c < width; c += NT) {
        float v = rbf(__fadd_rn(bf(base[q * width + c]), bf(summary[q * width + c])));
        v = rbf(__fadd_rn(v, bf(global[c])));
        Y[q * width + c] = to_bf(__fadd_rn(v, bf(types_table[type * width + c])));
    }
}

// Block r: F.normalize of row r (or of bf16(x[r] + add) when `add` is given): y = bf16(x / bf16(max(||x||, eps))).
extern "C" __global__ void __launch_bounds__(NT) tf_clef_normalize(const __nv_bfloat16* __restrict__ X, const __nv_bfloat16* __restrict__ add,
                                                                   __nv_bfloat16* __restrict__ Y, int width, float eps) {
    __shared__ float red[NT / 32];
    const int64_t at = static_cast<int64_t>(blockIdx.x) * width;
    float squares = 0.0f;
    for (int c = threadIdx.x; c < width; c += NT) {
        const float v = add ? rbf(__fadd_rn(bf(X[at + c]), bf(add[c]))) : bf(X[at + c]);
        squares = __fmaf_rn(v, v, squares);
    }
    const float denom = rbf(fmaxf(row_norm(squares, red), eps));
    for (int c = threadIdx.x; c < width; c += NT) {
        const float v = add ? rbf(__fadd_rn(bf(X[at + c]), bf(add[c]))) : bf(X[at + c]);
        Y[at + c] = to_bf(__fdiv_rn(v, denom));
    }
}

// Block o: prior_dot[o] = bf16(lexical_anchor[o] . anchor[question(o)]), torch's matmul of the two normalized rows.
extern "C" __global__ void __launch_bounds__(NT) tf_clef_prior_dot(const __nv_bfloat16* __restrict__ lexical, const __nv_bfloat16* __restrict__ anchors,
                                                                   const uint32_t* __restrict__ question, __nv_bfloat16* __restrict__ Y, int width) {
    __shared__ float red[NT / 32];
    const int64_t o = blockIdx.x;
    const int64_t q = question[o];
    float dot = 0.0f;
    for (int c = threadIdx.x; c < width; c += NT) dot = __fmaf_rn(bf(lexical[o * width + c]), bf(anchors[q * width + c]), dot);
    dot = block_sum(dot, red);
    if (threadIdx.x == 0) Y[o] = to_bf(dot);
}

// Block o with field f = fields[question(o)] and option x = options[o]:
//   cosine[o] = bf16(sum bf16(bf16(f / bf16(max(||f||, eps))) * bf16(x / bf16(max(||x||, eps)))))
//   features[o] = [f, x, bf16(f * x), |bf16(f - x)|]
// cosine_similarity as torch 2.8 decomposes it: each side divided by its clamped norm, then the product summed.
extern "C" __global__ void __launch_bounds__(NT) tf_clef_score_features(const __nv_bfloat16* __restrict__ fields, const __nv_bfloat16* __restrict__ options,
                                                                       const uint32_t* __restrict__ question, __nv_bfloat16* __restrict__ cosine,
                                                                       __nv_bfloat16* __restrict__ features, int width, float eps) {
    __shared__ float red[NT / 32];
    const int64_t o = blockIdx.x;
    const __nv_bfloat16* f = fields + static_cast<int64_t>(question[o]) * width;
    const __nv_bfloat16* x = options + o * width;
    __nv_bfloat16* out = features + o * 4 * width;
    float ff = 0.0f;
    float xx = 0.0f;
    for (int c = threadIdx.x; c < width; c += NT) {
        const float a = bf(f[c]);
        const float b = bf(x[c]);
        ff = __fmaf_rn(a, a, ff);
        xx = __fmaf_rn(b, b, xx);
        out[c] = f[c];
        out[width + c] = x[c];
        out[2 * width + c] = to_bf(__fmul_rn(a, b));
        out[3 * width + c] = to_bf(fabsf(rbf(__fsub_rn(a, b))));
    }
    const float nf = rbf(fmaxf(row_norm(ff, red), eps));
    const float nx = rbf(fmaxf(row_norm(xx, red), eps));
    float dot = 0.0f;
    for (int c = threadIdx.x; c < width; c += NT) dot = __fadd_rn(dot, rbf(__fmul_rn(rbf(__fdiv_rn(bf(f[c]), nf)), rbf(__fdiv_rn(bf(x[c]), nx)))));
    dot = block_sum(dot, red);
    if (threadIdx.x == 0) cosine[o] = to_bf(dot);
}

// Element o: logits = bf16(prior + bf16(bf16(sigmoid(gate)) * joint)), with
//   prior = bf16(bf16(exp(bf16(min(prior_scale, ln 100)))) * prior_dot)
//   joint = bf16(bf16(bf16(exp(bf16(min(joint_scale, ln 100)))) * cosine) + residual)
// scalars: prior_logit_scale, joint_logit_scale, residual_gate (BF16, as stored). torch's exp and sigmoid run in FP32.
extern "C" __global__ void __launch_bounds__(NT) tf_clef_logit_combine(const __nv_bfloat16* __restrict__ prior_scale, const __nv_bfloat16* __restrict__ joint_scale,
                                                                       const __nv_bfloat16* __restrict__ gate, const __nv_bfloat16* __restrict__ prior_dot,
                                                                       const __nv_bfloat16* __restrict__ cosine, const __nv_bfloat16* __restrict__ residual,
                                                                       __nv_bfloat16* __restrict__ logits, int n, float bound) {
    const int o = blockIdx.x * NT + threadIdx.x;
    if (o >= n) return;
    const float ps = rbf(expf(rbf(fminf(bf(*prior_scale), bound))));
    const float js = rbf(expf(rbf(fminf(bf(*joint_scale), bound))));
    const float g = rbf(__fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-bf(*gate)))));
    const float prior = rbf(__fmul_rn(ps, bf(prior_dot[o])));
    const float joint = rbf(__fadd_rn(rbf(__fmul_rn(js, bf(cosine[o]))), bf(residual[o])));
    logits[o] = to_bf(__fadd_rn(prior, rbf(__fmul_rn(g, joint))));
}
