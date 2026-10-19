// Target-graph metadata adapters for the retained TensorFold GLM DSA arithmetic.
// Index/FP8/score/chunk helpers retain their native headers, notices and Triton operation order.
// Device row-table orchestration follows the pinned production segmented design; this is not the TP3-only HQ24 port.
#include "glm_target_meta.cuh"
#include "glm_glue_latent.cuh"
#include "glm_glue_index.cuh"
#include "glm_glue_attn.cuh"

namespace glm_target_dsa {
constexpr uint32_t resident = 1, sparse = 2;
constexpr int width = 2051, pools = 512, nch = 5;

__device__ __forceinline__ bool owns(const TfGlmTargetDsaRow& row, uint32_t rank) {
  return row.owner == UINT32_MAX || row.owner == rank;
}

template <bool FP8>
__device__ void latent_write(const __nv_bfloat16* lat, int64_t stride, const TfGlmTargetDsaRow* rows) {
  const int r = blockIdx.x;
  const auto row = rows[r];
  // Non-owner latents have no allocation. Predicate before forming or loading a cache address.
  if (!(row.flags & resident)) return;
  float x[4];
  for (int j = 0; j < 4; ++j) x[j] = glm_triton::bf(lat[(int64_t)r * stride + threadIdx.x * 4 + j]);
  glm_cache::store<512, 4, FP8>(reinterpret_cast<void*>(row.latent), row.position, x);
}

template <bool FP8>
__device__ void pool_keys(const __nv_bfloat16* ape, const TfGlmTargetDsaRow* rows) {
  constexpr int D = 128;
  const auto row = rows[blockIdx.x];
  if ((row.position & 3u) != 3u) return;
  const int64_t p = row.position / 4;
  if (p >= row.pooled_rows) return;
  const auto* ik = reinterpret_cast<const __nv_bfloat16*>(row.index_keys);
  const auto* ig = reinterpret_cast<const __nv_bfloat16*>(row.index_gates);
  float pooled[4];
  for (int j = 0; j < 4; ++j) {
    const int d = threadIdx.x * 4 + j;
    float l[4], e[4];
    for (int q = 0; q < 4; ++q) l[q] = glm_triton::bf(ig[((4 * p + q) % row.ring_rows) * D + d]) + glm_triton::bf(ape[q * D + d]);
    const float m = fmaxf(fmaxf(l[0], l[1]), fmaxf(l[2], l[3]));
    for (int q = 0; q < 4; ++q) e[q] = glm_triton::exp_approx(l[q] - m);
    const float s = ((e[0] + e[1]) + e[2]) + e[3];
    float t = 0.0f;
    for (int q = 0; q < 4; ++q) {
      const float tq = glm_triton::rbf(__fmul_rn(glm_triton::rbf(glm_triton::div_full(e[q], s)), glm_triton::bf(ik[((4 * p + q) % row.ring_rows) * D + d])));
      t = q == 0 ? tq : t + tq;
    }
    pooled[j] = t;
  }
  glm_cache::store<D, 4, FP8>(reinterpret_cast<void*>(row.pooled), p, pooled);
}

template <bool FP8>
__device__ void scores(const __nv_bfloat16* qi, const __nv_bfloat16* wts, int64_t w_stride,
                       float* out, const TfGlmTargetDsaRow* rows, int64_t np, int64_t nb,
                       float scale, float wscale, int64_t heads) {
  const int r = blockIdx.x;
  const auto row = rows[r];
  if (!(row.flags & sparse)) return;
  // The retained helper adds blockIdx.x to this scalar; load each row's absolute position on device.
  const int base = static_cast<int>(row.position) - r;
  glm_scores<FP8>(qi, wts, w_stride, reinterpret_cast<const void*>(row.pooled), out, &base, np, nb,
                   scale, wscale, heads, 1);
}

template <bool FP8>
__device__ void chunks(const __nv_bfloat16* query, const int* tokens, const int* counts,
                       float* po, float* pm, float* pl, const TfGlmTargetDsaRow* rows,
                       int64_t total, int64_t heads, float scale, uint32_t rank) {
  const int r = blockIdx.x;
  const auto row = rows[r];
  if (!owns(row, rank)) return;
  const int base = static_cast<int>(row.position) - r;
  glm_attn::chunks_wide<16, FP8>(query, reinterpret_cast<const void*>(row.latent), &base,
                                 tokens, counts, po, pm, pl, total, heads, width, scale,
                                 (row.flags & sparse) != 0);
}
}  // namespace glm_target_dsa

extern "C" __global__ void __launch_bounds__(128)
    tf_glm_target_dsa_lat_bf16(const __nv_bfloat16* lat, int64_t stride, const TfGlmTargetDsaRow* rows) {
  glm_target_dsa::latent_write<false>(lat, stride, rows);
}
extern "C" __global__ void __launch_bounds__(128)
    tf_glm_target_dsa_lat_fp8(const __nv_bfloat16* lat, int64_t stride, const TfGlmTargetDsaRow* rows) {
  glm_target_dsa::latent_write<true>(lat, stride, rows);
}
extern "C" __global__ void __launch_bounds__(32)
    tf_glm_target_dsa_index_write(const __nv_bfloat16* kr, int64_t stride, const float* gr,
                                 const __nv_bfloat16* lnw, const __nv_bfloat16* lnb,
                                 const TfGlmTargetDsaRow* rows) {
  const int r = blockIdx.x;
  const auto row = rows[r];
  const int base = static_cast<int>(row.position) - r;
  glm_index_write(kr, stride, gr, lnw, lnb, reinterpret_cast<__nv_bfloat16*>(row.index_keys),
                  reinterpret_cast<__nv_bfloat16*>(row.index_gates), &base, 1e-6f, row.ring_rows);
}
extern "C" __global__ void __launch_bounds__(32)
    tf_glm_target_dsa_pool_bf16(const __nv_bfloat16* ape, const TfGlmTargetDsaRow* rows) {
  glm_target_dsa::pool_keys<false>(ape, rows);
}
extern "C" __global__ void __launch_bounds__(32)
    tf_glm_target_dsa_pool_fp8(const __nv_bfloat16* ape, const TfGlmTargetDsaRow* rows) {
  glm_target_dsa::pool_keys<true>(ape, rows);
}
extern "C" __global__ void __launch_bounds__(128)
    tf_glm_target_dsa_scores_bf16(const __nv_bfloat16* qi, const __nv_bfloat16* wts, int64_t stride,
                                  float* out, const TfGlmTargetDsaRow* rows, int64_t np, int64_t nb,
                                  float scale, float wscale, int64_t heads) {
  glm_target_dsa::scores<false>(qi, wts, stride, out, rows, np, nb, scale, wscale, heads);
}
extern "C" __global__ void __launch_bounds__(128)
    tf_glm_target_dsa_scores_fp8(const __nv_bfloat16* qi, const __nv_bfloat16* wts, int64_t stride,
                                 float* out, const TfGlmTargetDsaRow* rows, int64_t np, int64_t nb,
                                 float scale, float wscale, int64_t heads) {
  glm_target_dsa::scores<true>(qi, wts, stride, out, rows, np, nb, scale, wscale, heads);
}

// Native exact radix selection, with row visibility supplied by the device metadata instead of one host sequence.
extern "C" __global__ void __launch_bounds__(128)
    tf_glm_target_dsa_select_rows(const float* scores, int64_t* out, int64_t np, const TfGlmTargetDsaRow* rows) {
  constexpr int k = glm_target_dsa::pools;
  __shared__ int hist[256];
  __shared__ uint32_t sh_prefix;
  __shared__ int sh_need, wsum[2][4];
  const int64_t r = blockIdx.x;
  if (!(rows[r].flags & glm_target_dsa::sparse)) return;
  const float* row = scores + r * np;
  const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
  const unsigned below = (1u << lane) - 1;
  const int64_t vis = ((int64_t)rows[r].position + 1) / 4;
  const int64_t limit = np < (vis > k ? vis : k) ? np : (vis > k ? vis : k);
  const bool vec = (reinterpret_cast<uintptr_t>(row) & 15) == 0;
  auto keys = [&](int64_t c, uint32_t u[8]) {
    const int64_t i0 = c + 8 * t;
    if (vec && i0 + 8 <= limit) {
      const float4 a = *reinterpret_cast<const float4*>(row + i0), b = *reinterpret_cast<const float4*>(row + i0 + 4);
      const float v[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
      for (int j = 0; j < 8; ++j) u[j] = glm_index::order_key(v[j]);
    } else {
      for (int j = 0; j < 8; ++j) u[j] = i0 + j < limit ? glm_index::order_key(row[i0 + j]) : 0u;
    }
  };
  uint32_t prefix = 0, fixed = 0;
  int need = k;
  for (int shift = 24; shift >= 0; shift -= 8) {
    for (int b = t; b < 256; b += blockDim.x) hist[b] = 0;
    __syncthreads();
    for (int64_t c = 0; c < limit; c += 1024) {
      uint32_t u[8];
      keys(c, u);
      for (int j = 0; j < 8; ++j) {
        const int b = c + 8 * t + j < limit && (u[j] & fixed) == prefix ? (int)((u[j] >> shift) & 0xFF) : 256;
        if (__any_sync(0xffffffffu, b < 256)) {
          const unsigned same = __match_any_sync(0xffffffffu, b);
          if (b < 256 && (same & below) == 0) atomicAdd(&hist[b], __popc(same));
        }
      }
    }
    __syncthreads();
    if (warp == 0) {
      int h[8], own = 0;
      for (int j = 0; j < 8; ++j) own += h[j] = hist[255 - 8 * lane - j];
      int upto = own;
      for (int o = 1; o < 32; o <<= 1) {
        const int v = __shfl_up_sync(0xffffffffu, upto, o);
        if (lane >= o) upto += v;
      }
      const unsigned hit = __ballot_sync(0xffffffffu, upto >= need);
      if (lane == (hit ? __ffs(hit) - 1 : 31)) {
        int above = upto - own, digit = 0;
        for (int j = 0; j < 8; ++j) {
          if (above + h[j] >= need) {
            digit = 255 - 8 * lane - j;
            break;
          }
          above += h[j];
        }
        sh_need = need - above;
        sh_prefix = prefix | ((uint32_t)digit << shift);
      }
    }
    __syncthreads();
    need = sh_need;
    prefix = sh_prefix;
    fixed |= 0xFFu << shift;
    __syncthreads();
  }
  int run_take = 0, run_eq = 0;
  for (int64_t c = 0; c < limit; c += 1024) {
    uint32_t u[8];
    keys(c, u);
    bool eq[8], gt[8];
    int eqs = 0;
    for (int j = 0; j < 8; ++j) {
      const bool in = c + 8 * t + j < limit;
      eq[j] = in && u[j] == prefix;
      gt[j] = in && u[j] > prefix;
      eqs += eq[j];
    }
    int all_eq, all_take;
    int seen = run_eq + glm_index::block_scan(eqs, wsum, 0, &all_eq);
    bool take[8];
    int takes = 0;
    for (int j = 0; j < 8; ++j) {
      take[j] = gt[j] || (eq[j] && seen < need);
      seen += eq[j];
      takes += take[j];
    }
    int at = run_take + glm_index::block_scan(takes, wsum, 1, &all_take);
    for (int j = 0; j < 8; ++j)
      if (take[j]) out[r * k + at++] = c + 8 * t + j;
    run_eq += all_eq;
    run_take += all_take;
  }
}

extern "C" __global__ void tf_glm_target_dsa_tokens(const int64_t* pools, const TfGlmTargetDsaRow* rows,
                                                   int* tokens, int* counts) {
  constexpr int K = glm_target_dsa::pools, W = glm_target_dsa::width;
  const int64_t r = blockIdx.x, position = rows[r].position;
  int* row = tokens + r * W;
  if (!(rows[r].flags & glm_target_dsa::sparse)) {
    for (int i = threadIdx.x; i < W; i += blockDim.x) row[i] = -1;
    if (threadIdx.x == 0) counts[r] = 0;
    return;
  }
  const int64_t npool = (position + 1) / 4;
  for (int i = threadIdx.x; i < 4 * K; i += blockDim.x)
    row[i] = static_cast<int>(pools[r * K + i / 4] * 4 + i % 4);
  if (threadIdx.x < 3) {
    const int64_t t = npool * 4 + threadIdx.x;
    row[4 * K + threadIdx.x] = t <= position ? static_cast<int>(t) : -1;
  }
  if (threadIdx.x == 0) counts[r] = npool > K ? 4 * K + static_cast<int>((position + 1) % 4) : 0;
}

extern "C" __global__ void __launch_bounds__(256)
    tf_glm_target_dsa_chunks_bf16(const __nv_bfloat16* query, const int* tokens, const int* counts,
                                  float* po, float* pm, float* pl, const TfGlmTargetDsaRow* rows,
                                  int64_t total, int64_t heads, float scale, uint32_t rank) {
  glm_target_dsa::chunks<false>(query, tokens, counts, po, pm, pl, rows, total, heads, scale, rank);
}
extern "C" __global__ void __launch_bounds__(256)
    tf_glm_target_dsa_chunks_fp8(const __nv_bfloat16* query, const int* tokens, const int* counts,
                                 float* po, float* pm, float* pl, const TfGlmTargetDsaRow* rows,
                                 int64_t total, int64_t heads, float scale, uint32_t rank) {
  glm_target_dsa::chunks<true>(query, tokens, counts, po, pm, pl, rows, total, heads, scale, rank);
}

// Same ascending-chunk merge and BF16 rounding as the native latent._merge; nonowners never read stale partials.
extern "C" __global__ void __launch_bounds__(128)
    tf_glm_target_dsa_merge(const float* po, const float* pm, const float* pl, __nv_bfloat16* out,
                            const TfGlmTargetDsaRow* rows, int64_t total, int64_t heads, uint32_t rank) {
  constexpr int LW = glm_attn::LW;
  const int64_t r = blockIdx.x, h = blockIdx.y;
  const int k0 = threadIdx.x * 4;
  if (!glm_target_dsa::owns(rows[r], rank)) {
    for (int j = 0; j < 4; ++j) out[(r * heads + h) * LW + k0 + j] = __float2bfloat16_rn(0.0f);
    return;
  }
  float m = -INFINITY, l = 0.0f, o[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  // pad stages the eager segment's merge count, preserving even its empty-chunk signed-zero arithmetic.
  for (int c = 0; c < glm_target_dsa::nch && c < rows[r].pad; ++c) {
    const int64_t base = (c * total + r) * heads + h;
    const float cm = pm[base], cl = pl[base];
    const bool active = cl > 0.0f;
    const float nm = active ? fmaxf(m, cm) : m;
    const float b = active ? glm_triton::exp_approx(cm - nm) : 0.0f;
    if (c == 0) {
      for (int j = 0; j < 4; ++j) o[j] = __fmaf_rn(b, po[base * LW + k0 + j], 0.0f);
      l = __fmaf_rn(b, cl, 0.0f);
    } else {
      const float a = active ? (m == -INFINITY ? 0.0f : glm_triton::exp_approx(m - nm)) : 1.0f;
      for (int j = 0; j < 4; ++j) o[j] = __fmaf_rn(o[j], a, __fmul_rn(b, po[base * LW + k0 + j]));
      l = __fmaf_rn(l, a, __fmul_rn(b, cl));
    }
    m = nm;
  }
  for (int j = 0; j < 4; ++j) out[(r * heads + h) * LW + k0 + j] = __float2bfloat16_rn(glm_triton::div_full(o[j], l));
}

extern "C" __global__ void tf_glm_target_dsa_pack_peer(const uint16_t* local_results, uint32_t rank,
                                                      int64_t total, int64_t heads, int64_t lw,
                                                      uint16_t* peer) {
  const int64_t r = blockIdx.x, width = heads * lw;
  for (int64_t column = threadIdx.x; column < width; column += blockDim.x)
    peer[r * width + column] = local_results[((1 - rank) * total + r) * width + column];
}

extern "C" __global__ void tf_glm_target_dsa_owner_select(const TfGlmTargetDsaRow* rows,
                                                         const uint16_t* own_result,
                                                         const uint16_t* gathered_peer, uint32_t rank,
                                                         int64_t total, int64_t heads, int64_t lw,
                                                         uint16_t* out) {
  const int64_t r = blockIdx.x, width = heads * lw;
  const uint32_t owner = rows[r].owner;
  for (int64_t column = threadIdx.x; column < width; column += blockDim.x) {
    const int64_t at = r * width + column;
    out[at] = owner == rank ? own_result[at] : (owner < 2 ? gathered_peer[static_cast<int64_t>(owner) * total * width + at] : 0);
  }
}
