// Qwen math adapted from Hugging Face Transformers, revision
// e0eb2810ac1b2caf6d082f247d8d37162ecf67a0, modeling_qwen3_5_moe.py.
// Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// Local changes adapt qwen35_ops.cu's arithmetic to upstream Geo/Seg tables,
// read-only convolution tails, head-major KV, and one-slot attention partials.
// The residual norm retains norms.cu's BF16 vector reduction order.

#include <cuda_runtime.h>
#include <math.h>
#include "qwen35.cuh"

using namespace tf_q35;
using B = __nv_bfloat16;

namespace {
__device__ B b(float x) { return __float2bfloat16_rn(x); }
__device__ float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

template<int N>
__device__ float sum(float x, float* scratch) {
    const int t = threadIdx.x;
    scratch[t] = x;
    __syncthreads();
    for (int stride = N / 2; stride; stride /= 2) {
        if (t < stride) scratch[t] += scratch[t + stride];
        __syncthreads();
    }
    const float result = scratch[0];
    __syncthreads();
    return result;
}

__device__ B rotate_hf(const B* x, int d, const int pos[3]) {
    if (d >= 64) return x[d];
    const int i = d % 32;
    const float angle = float(pos[mrope_axis(i)]) * powf(10000000.0f, -float(i) / 32.0f);
    const float cs = rbf(cosf(angle)), sn = rbf(sinf(angle));
    const float lhs = rbf(bf(x[d]) * cs);
    const float rhs = rbf(bf(x[d < 32 ? d + 32 : d - 32]) * sn);
    return b(d < 32 ? lhs - rhs : lhs + rhs);
}
}

extern "C" __global__ void __launch_bounds__(256) tf_q35_hf_embed(
    const int* ids, const B* W, const B* S, const B* Bias, B* out, int D, const int* P) {
    if (past_rows(P, blockIdx.x)) return;
    for (int d = threadIdx.x; d < D; d += blockDim.x)
        out[size_t(blockIdx.x) * D + d] = W[size_t(ids[blockIdx.x]) * D + d];
}

extern "C" __global__ void __launch_bounds__(256) tf_q35_hf_norm(
    B* H, const B* R, const float* W, B* X, float eps, int D, const int* P) {
    if (past_rows(P, blockIdx.x)) return;
    __shared__ float warps[8];
    const size_t base = size_t(blockIdx.x) * D;
    float values[4][8], part = 0.0f;
    for (int j = 0; j < 4; ++j) {
        const int chunk = threadIdx.x + j * 256;
        if (chunk >= D / 8) continue;
        for (int i = 0; i < 8; ++i) {
            const int d = 8 * chunk + i;
            B h = H[base + d];
            if (R != nullptr) {
                h = b(bf(h) + bf(R[base + d]));
                H[base + d] = h;
            }
            values[j][i] = bf(h);
            part = __fmaf_rn(values[j][i], values[j][i], part);
        }
    }
    for (int m = 16; m > 0; m >>= 1) part = __fadd_rn(part, __shfl_xor_sync(0xffffffffu, part, m));
    if ((threadIdx.x & 31) == 0) warps[threadIdx.x >> 5] = part;
    __syncthreads();
    float total = (threadIdx.x & 31) < 8 ? warps[threadIdx.x & 31] : 0.0f;
    for (int m = 4; m > 0; m >>= 1) total = __fadd_rn(total, __shfl_xor_sync(0xffffffffu, total, m));
    total = __shfl_sync(0xffffffffu, total, 0);
    const float inv = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(__fdiv_rn(total, float(D)), eps)));
    for (int j = 0; j < 4; ++j) {
        const int chunk = threadIdx.x + j * 256;
        if (chunk >= D / 8) continue;
        for (int i = 0; i < 8; ++i) {
            const int d = 8 * chunk + i;
            X[base + d] = b(__fmul_rn(__fmul_rn(values[j][i], inv), W[d]));
        }
    }
}

extern "C" __global__ void __launch_bounds__(128) tf_q35_hf_gdn_pre(
    Geo geo, const B* QKV, const Seg* segs, const int* rowseg, const int* rowpar, int layer,
    const float* CW, const B* Ain, const B* Bin, const float* ALOG, const float* DT,
    float* Q, float* K, B* V, float* G, B* BETA, const int* P) {
    if (past_rows(P, blockIdx.z)) return;
    __shared__ float scratch[DK];
    const int head = blockIdx.y, d = threadIdx.x, row = blockIdx.z;
    const int C = geo.conv(), ch = head * DK + d;
    const Seg sg = segs[rowseg[row]];
    const B* tail = reinterpret_cast<const B*>(sg.conv) + size_t(layer) * geo.conv_rows();
    float acc = 0.0f;
    for (int tap = 0; tap < TAPS; ++tap) {
        const int source = back(rowpar, row, TAPS - 1 - tap);
        const float value = source < 0 ? bf(tail[size_t(-1 - source) * C + ch]) : bf(QKV[size_t(source) * C + ch]);
        acc += value * CW[ch * TAPS + tap];
    }
    const float conv = rbf(acc);
    const float activated = rbf(conv * sigmoid(conv));
    if (head < 2 * geo.nk) {
        const float inv = rsqrtf(sum<DK>(activated * activated, scratch) + 1.0e-6f);
        if (head < geo.nk) Q[(size_t(row) * geo.nk + head) * DK + d] = (activated * inv) / sqrtf(float(DK));
        else K[(size_t(row) * geo.nk + head - geo.nk) * DK + d] = activated * inv;
    } else {
        const int h = head - 2 * geo.nk;
        V[(size_t(row) * geo.nv + h) * DK + d] = b(activated);
        if (d == 0) {
            const float dt = bf(Ain[row * geo.nv + h]) + DT[h];
            const float softplus = dt > 20.0f ? dt : log1pf(expf(dt));
            G[row * geo.nv + h] = -expf(ALOG[h]) * softplus;
            BETA[row * geo.nv + h] = b(sigmoid(bf(Bin[row * geo.nv + h])));
        }
    }
}

extern "C" __global__ void __launch_bounds__(128) tf_q35_hf_gdn_chain(
    Geo geo, const float* Q, const float* K, const B* V, const float* G, const B* BETA,
    const Seg* segs, const int* rowpar, int layer, float* Y, const int* P) {
    const int segment = blockIdx.z / geo.nv;
    if (P != nullptr && segment >= P[3]) return;
    const Seg sg = segs[segment];
    __shared__ float qs[DK], ks[DK];
    const int h = blockIdx.z % geo.nv, value = threadIdx.x;
    const int hk = h / (geo.nv / geo.nk), row = sg.row0;
    qs[value] = Q[(size_t(row) * geo.nk + hk) * DK + value];
    ks[value] = K[(size_t(row) * geo.nk + hk) * DK + value];
    __syncthreads();
    float* state = reinterpret_cast<float*>(sg.rec) + size_t(layer) * geo.rec() + (h * DK + value) * DK;
    const float retain = expf(G[row * geo.nv + h]);
    float memory = 0.0f;
    for (int key = 0; key < DK; ++key) {
        state[key] *= retain;
        memory += state[key] * ks[key];
    }
    const float delta = (bf(V[(size_t(row) * geo.nv + h) * DK + value]) - memory) * bf(BETA[row * geo.nv + h]);
    float out = 0.0f;
    for (int key = 0; key < DK; ++key) {
        state[key] += ks[key] * delta;
        out += state[key] * qs[key];
    }
    Y[(size_t(row) * geo.nv + h) * DK + value] = out;
}

extern "C" __global__ void __launch_bounds__(128) tf_q35_hf_gdn_post(
    Geo geo, const float* Y, const B* Z, const float* NW, float eps, B* OUT, const int* P) {
    if (past_rows(P, blockIdx.z)) return;
    __shared__ float scratch[DK];
    const int d = threadIdx.x;
    const size_t i = (size_t(blockIdx.z) * geo.nv + blockIdx.y) * DK + d;
    const float yr = rbf(Y[i]);
    const float inv = rsqrtf(sum<DK>(yr * yr, scratch) / float(DK) + eps);
    const float normalized = rbf(yr * inv);
    const float weighted = rbf(normalized * NW[d]);
    const float gate = bf(Z[i]);
    OUT[i] = b(weighted * (gate * sigmoid(gate)));
}

extern "C" __global__ void __launch_bounds__(256) tf_q35_hf_head_norm(
    const B* X, const float* W, B* OUT, int heads, int stride, int hstride, float eps, const int* P) {
    const int head = blockIdx.x % heads, row = blockIdx.x / heads, d = threadIdx.x;
    if (past_rows(P, row)) return;
    __shared__ float scratch[HD];
    const float x = bf(X[size_t(row) * stride + head * hstride + d]);
    const float inv = rsqrtf(sum<HD>(x * x, scratch) / float(HD) + eps);
    OUT[(size_t(row) * heads + head) * HD + d] = b((x * inv) * W[d]);
}

extern "C" __global__ void __launch_bounds__(256) tf_q35_hf_rope(
    Geo geo, B* Qn, const B* Kn, const B* Vr, const Seg* segs, const int* rowseg,
    const unsigned* rowanc, int layer, const int* P, const int* M) {
    if (past_rows(P, blockIdx.x)) return;
    const int d = threadIdx.x, row = blockIdx.x;
    const Seg sg = segs[rowseg[row]];
    int pos[3];
    row_positions(sg, rowanc[row], row, M, pos);
    __shared__ B queries[MAXQ * HD];
    for (int h = 0; h < geo.qh; ++h) queries[h * HD + d] = Qn[(size_t(row) * geo.qh + h) * HD + d];
    __syncthreads();
    for (int h = 0; h < geo.qh; ++h) Qn[(size_t(row) * geo.qh + h) * HD + d] = rotate_hf(queries + h * HD, d, pos);
    B* keys = reinterpret_cast<B*>(sg.kv) + size_t(layer) * 2 * geo.kvh * sg.cap * HD;
    B* values = keys + size_t(geo.kvh) * sg.cap * HD;
    const int slot = sg.len + row - sg.row0;
    for (int h = 0; h < geo.kvh; ++h) {
        const size_t dst = (size_t(h) * sg.cap + slot) * HD + d;
        keys[dst] = rotate_hf(Kn + (size_t(row) * geo.kvh + h) * HD, d, pos);
        values[dst] = Vr[(size_t(row) * geo.kvh + h) * HD + d];
    }
}

extern "C" __global__ void __launch_bounds__(256) tf_q35_hf_attn_partial(
    Geo geo, const B* Q, const Seg* segs, const int* rowseg, const unsigned* rowanc,
    int layer, const int* P, float scale, float* PM, float* PL, float* PO) {
    if (past_rows(P, blockIdx.z)) return;
    __shared__ float scratch[HD];
    const int h = blockIdx.y, row = blockIdx.z, d = threadIdx.x;
    const int kv = h / (geo.qh / geo.kvh);
    const Seg sg = segs[rowseg[row]];
    const unsigned anc = rowanc[row];
    const B* keys = reinterpret_cast<const B*>(sg.kv) + size_t(layer) * 2 * geo.kvh * sg.cap * HD;
    const B* values = keys + size_t(geo.kvh) * sg.cap * HD;
    const float query = bf(Q[(size_t(row) * geo.qh + h) * HD + d]);
    float maximum = -INFINITY, denominator = 0.0f, result = 0.0f;
    for (int token = 0; token < sg.len + rows_seen(anc); ++token) {
        const int pos = token < sg.len ? token : sg.len + seen_row(anc, token - sg.len);
        const size_t off = (size_t(kv) * sg.cap + pos) * HD + d;
        const float score = sum<HD>(query * bf(keys[off]), scratch) * scale;
        const float next = fmaxf(maximum, score);
        const float old_weight = expf(maximum - next), weight = expf(score - next);
        result = result * old_weight + weight * bf(values[off]);
        denominator = denominator * old_weight + weight;
        maximum = next;
    }
    const size_t slot = (size_t(row) * geo.qh + h) * MAXCH;
    if (d == 0) {
        PM[slot] = maximum;
        PL[slot] = denominator;
    }
    PO[slot * HD + d] = result;
}

extern "C" __global__ void __launch_bounds__(256) tf_q35_hf_attn_merge(
    Geo geo, const float* PM, const float* PL, const float* PO, const Seg* segs,
    const int* rowseg, const unsigned* rowanc, const int* P, B* OUT) {
    if (past_rows(P, blockIdx.y)) return;
    const size_t head = size_t(blockIdx.y) * geo.qh + blockIdx.x;
    const size_t slot = head * MAXCH;
    OUT[head * HD + threadIdx.x] = b(PO[slot * HD + threadIdx.x] / PL[slot]);
}

extern "C" __global__ void tf_q35_hf_mlp_act(
    const B* GATE, const B* UP, B* OUT, long long total, const int* P, int D) {
    if (P != nullptr) total = (long long)P[2] * D;
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const float gate = bf(GATE[i]);
    OUT[i] = b((gate * sigmoid(gate)) * bf(UP[i]));
}

// Local HF adaptation retains the shared activation's intermediate BF16 value.
extern "C" __global__ void tf_q35_hf_shared_act_bf16(
    const B* GATE, const B* UP, B* OUT, long long total, const int* P, int D) {
    if (P != nullptr) total = (long long)P[2] * D;
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const float gate = bf(GATE[i]);
    OUT[i] = b(rbf(gate * sigmoid(gate)) * bf(UP[i]));
}

extern "C" __global__ void tf_q35_hf_route_products(
    const B* x, const B* router, const B* shared_gate, float* scores) {
    __shared__ float products[2048];
    const int expert = blockIdx.x;
    const B* row = expert < 256 ? router + expert * 2048 : shared_gate;
    for (int d = threadIdx.x; d < 2048; d += blockDim.x)
        products[d] = __fmul_rn(bf(x[d]), bf(row[d]));
    __syncthreads();
    if (threadIdx.x != 0) return;
    float value = 0.0f;
    for (int d = 0; d < 2048; ++d) value = __fadd_rn(value, products[d]);
    scores[expert] = expert < 256 ? rbf(value) : value;
}

extern "C" __global__ void tf_q35_hf_route_select_parallel(
    const float* scores, int32_t* picks, float* weights) {
    __shared__ float logits[256], chosen[8];
    __shared__ unsigned warp_score_bits[8];
    __shared__ int warp_indices[8], selected_id;
    const int expert = threadIdx.x, lane = expert & 31, warp = expert >> 5;
    const float score = scores[expert];
    logits[expert] = score;
    if (__syncthreads_or(!isfinite(score))) {
        if (threadIdx.x != 0) return;
        float chosen[8];
        int selected[8] = {-1, -1, -1, -1, -1, -1, -1, -1};
        #pragma unroll
        for (int slot = 0; slot < 8; ++slot) {
            int best = -1;
            for (int candidate = 0; candidate < 256; ++candidate) {
                bool used = false;
                #pragma unroll
                for (int prior = 0; prior < 8; ++prior) used |= selected[prior] == candidate;
                if (!used && (best < 0 || logits[candidate] > logits[best])) best = candidate;
            }
            selected[slot] = best;
            chosen[slot] = logits[best];
        }
        #pragma unroll
        for (int slot = 0; slot < 8; ++slot) picks[slot] = selected[slot];
        float denominator = 0.0f;
        for (int slot = 0; slot < 8; ++slot) {
            weights[slot] = expf(chosen[slot] - chosen[0]);
            denominator += weights[slot];
        }
        for (int slot = 0; slot < 8; ++slot) weights[slot] /= denominator;
        weights[8] = sigmoid(rbf(scores[256]));
        return;
    }
    bool selected = false;
    for (int slot = 0; slot < 8; ++slot) {
        unsigned best_bits = selected ? 0xff800000u : __float_as_uint(score);
        int best = selected ? 256 : expert;
        #pragma unroll
        for (int offset = 16; offset; offset >>= 1) {
            const unsigned other_bits = __shfl_down_sync(0xffffffffu, best_bits, offset);
            const int other = __shfl_down_sync(0xffffffffu, best, offset);
            const float value = __uint_as_float(best_bits), other_value = __uint_as_float(other_bits);
            if (other_value > value || (other_value == value && other < best)) {
                best_bits = other_bits;
                best = other;
            }
        }
        if (lane == 0) {
            warp_score_bits[warp] = best_bits;
            warp_indices[warp] = best;
        }
        __syncthreads();
        if (warp == 0) {
            best_bits = lane < 8 ? warp_score_bits[lane] : 0xff800000u;
            best = lane < 8 ? warp_indices[lane] : 256;
            #pragma unroll
            for (int offset = 16; offset; offset >>= 1) {
                const unsigned other_bits = __shfl_down_sync(0xffffffffu, best_bits, offset);
                const int other = __shfl_down_sync(0xffffffffu, best, offset);
                const float value = __uint_as_float(best_bits), other_value = __uint_as_float(other_bits);
                if (other_value > value || (other_value == value && other < best)) {
                    best_bits = other_bits;
                    best = other;
                }
            }
            if (lane == 0) {
                selected_id = best;
                picks[slot] = best;
                chosen[slot] = __uint_as_float(best_bits);
            }
        }
        __syncthreads();
        selected |= expert == selected_id;
    }
    if (expert != 0) return;
    float denominator = 0.0f;
    for (int slot = 0; slot < 8; ++slot) {
        weights[slot] = expf(chosen[slot] - chosen[0]);
        denominator += weights[slot];
    }
    for (int slot = 0; slot < 8; ++slot) weights[slot] /= denominator;
    weights[8] = sigmoid(rbf(scores[256]));
}

extern "C" __global__ void tf_q35_hf_combine(
    const float* pairs, const float* weights, const B* shared, B* out) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= 2048) return;
    float y = 0.0f;
    for (int slot = 0; slot < 8; ++slot) y += pairs[slot * 2048 + d] * weights[slot];
    out[d] = b(y + bf(shared[d]) * weights[8]);
}

// Local HF adaptation rounds each expert operation and accumulates by expert ID.
extern "C" __global__ void tf_q35_hf_combine_bf16(
    const float* pairs, const float* weights, const B* shared, B* out,
    const int* picks) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= 2048) return;
    float acc = 0.0f;
    int previous = -1;
    for (int iteration = 0; iteration < 8; ++iteration) {
        int slot = -1, best = 256;
        for (int candidate = 0; candidate < 8; ++candidate)
            if (picks[candidate] > previous && picks[candidate] < best) {
                best = picks[candidate];
                slot = candidate;
            }
        const float weight = rbf(weights[slot]);
        const float value = rbf(pairs[slot * 2048 + d]);
        const float term = rbf(__fmul_rn(value, weight));
        acc = rbf(__fadd_rn(acc, term));
        previous = best;
    }
    const float shared_product = rbf(__fmul_rn(bf(shared[d]), rbf(weights[8])));
    out[d] = b(__fadd_rn(acc, shared_product));
}
