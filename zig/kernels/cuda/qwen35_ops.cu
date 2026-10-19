// Qwen math adapted to serial CUDA from Hugging Face Transformers, revision
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
// Local changes: fixed 35B geometry, serial persistent state, online causal
// attention, deterministic top-eight routing and address-only CUDA entry points.
// The cooperative router stages products while retaining ordered FP32 additions.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

namespace {
using B = __nv_bfloat16;
constexpr int Hidden = 2048, KeyHeads = 16, ValueHeads = 32, D = 128;
constexpr int Conv = (2 * KeyHeads + ValueHeads) * D;
constexpr int Heads = 16, KvHeads = 2, HeadDim = 256, Rotary = 64;

__device__ float f(B x) { return __bfloat162float(x); }
__device__ B b(float x) { return __float2bfloat16_rn(x); }
__device__ float rounded(float x) { return f(b(x)); }
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
    const float out = scratch[0];
    __syncthreads();
    return out;
}

__device__ float rotary(float x, float partner, int dim, int position) {
    if (dim >= Rotary) return x;
    const int pair = dim % (Rotary / 2);
    const float angle = float(position) * powf(10000000.0f, -float(pair) / float(Rotary / 2));
    const float c = rounded(cosf(angle)), s = rounded(sinf(angle));
    const float lhs = rounded(x * c), rhs = rounded(partner * s);
    return rounded(dim < Rotary / 2 ? lhs - rhs : lhs + rhs);
}
}

extern "C" __global__ void tf_qwen35_embed(const B* table, uint32_t token, B* out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < Hidden) out[i] = table[size_t(token) * Hidden + i];
}

extern "C" __global__ void tf_qwen35_add(B* residual, const B* update, int width) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < width) residual[i] = b(f(residual[i]) + f(update[i]));
}

extern "C" __global__ void tf_qwen35_swiglu(const B* gate, const B* up, B* out, int width) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < width) {
        const float g = f(gate[i]);
        out[i] = b((g * sigmoid(g)) * f(up[i]));
    }
}

extern "C" __global__ void tf_qwen35_gdn_pre(
    const B* qkv, const float* conv_weights, B* history, const B* a, const B* beta_in,
    const float* a_log, const float* dt_bias, float* q, float* k, B* v, float* decay, float* beta) {
    __shared__ float scratch[D];
    const int head = blockIdx.x, d = threadIdx.x, ch = head * D + d;
    float acc = 0.0f;
    for (int row = 0; row < 3; ++row) acc += f(history[row * Conv + ch]) * conv_weights[ch * 4 + row];
    acc += f(qkv[ch]) * conv_weights[ch * 4 + 3];
    // F.conv1d and its SiLU each return the model dtype in the reference fallback.
    const float conv = rounded(acc);
    const float activated = rounded(conv * sigmoid(conv));
    const B old1 = history[Conv + ch], old2 = history[2 * Conv + ch];
    history[ch] = old1;
    history[Conv + ch] = old2;
    history[2 * Conv + ch] = qkv[ch];
    if (head < 2 * KeyHeads) {
        const float inv = rsqrtf(sum<D>(activated * activated, scratch) + 1.0e-6f);
        if (head < KeyHeads) q[head * D + d] = (activated * inv) / sqrtf(float(D));
        else k[(head - KeyHeads) * D + d] = activated * inv;
    } else {
        const int h = head - 2 * KeyHeads;
        v[h * D + d] = b(activated);
        if (d == 0) {
            const float dt = f(a[h]) + dt_bias[h];
            const float softplus = dt > 20.0f ? dt : log1pf(expf(dt));
            decay[h] = -expf(a_log[h]) * softplus;
            beta[h] = rounded(sigmoid(f(beta_in[h])));
        }
    }
}

extern "C" __global__ void tf_qwen35_gdn_step(
    const float* q, const float* k, const B* v, const float* decay, const float* beta,
    float* state, float* out) {
    __shared__ float qs[D], ks[D];
    const int h = blockIdx.x, value = threadIdx.x, key_head = h / (ValueHeads / KeyHeads);
    qs[value] = q[key_head * D + value];
    ks[value] = k[key_head * D + value];
    __syncthreads();
    float* row = state + (h * D + value) * D;
    const float retain = expf(decay[h]);
    float memory = 0.0f;
    for (int key = 0; key < D; ++key) {
        row[key] *= retain;
        memory += row[key] * ks[key];
    }
    const float delta = (f(v[h * D + value]) - memory) * beta[h];
    float y = 0.0f;
    for (int key = 0; key < D; ++key) {
        row[key] += ks[key] * delta;
        y += row[key] * qs[key];
    }
    out[h * D + value] = y;
}

extern "C" __global__ void tf_qwen35_gdn_out(
    const float* y, const B* z, const float* norm, B* out, float eps) {
    __shared__ float scratch[D];
    const int h = blockIdx.x, d = threadIdx.x, i = h * D + d;
    const float yr = rounded(y[i]);
    const float inv = rsqrtf(sum<D>(yr * yr, scratch) / float(D) + eps);
    const float normalized = rounded(yr * inv);
    const float weighted = rounded(normalized * norm[d]);
    const float gate = f(z[i]);
    out[i] = b(weighted * (gate * sigmoid(gate)));
}

extern "C" __global__ void tf_qwen35_attention_kv(
    const B* key, const B* value, const float* norm, B* keys, B* values, int position, float eps) {
    __shared__ float scratch[HeadDim], normalized[HeadDim];
    const int h = blockIdx.x, d = threadIdx.x, i = h * HeadDim + d;
    const float x = f(key[i]);
    const float inv = rsqrtf(sum<HeadDim>(x * x, scratch) / float(HeadDim) + eps);
    normalized[d] = rounded((x * inv) * norm[d]);
    __syncthreads();
    const int partner = d < Rotary / 2 ? d + Rotary / 2 : d < Rotary ? d - Rotary / 2 : d;
    const size_t dst = (size_t(position) * KvHeads + h) * HeadDim + d;
    keys[dst] = b(rotary(normalized[d], normalized[partner], d, position));
    values[dst] = value[i];
}

extern "C" __global__ void tf_qwen35_attention(
    const B* q_gate, const float* norm, const B* keys, const B* values, int position, B* out, float eps) {
    __shared__ float scratch[HeadDim], normalized[HeadDim], query[HeadDim];
    const int h = blockIdx.x, d = threadIdx.x, kv = h / (Heads / KvHeads);
    const float x = f(q_gate[h * 2 * HeadDim + d]);
    const float inv = rsqrtf(sum<HeadDim>(x * x, scratch) / float(HeadDim) + eps);
    normalized[d] = rounded((x * inv) * norm[d]);
    __syncthreads();
    const int partner = d < Rotary / 2 ? d + Rotary / 2 : d < Rotary ? d - Rotary / 2 : d;
    query[d] = rotary(normalized[d], normalized[partner], d, position);
    __syncthreads();
    float maximum = -__int_as_float(0x7f800000), denominator = 0.0f, result = 0.0f;
    for (int token = 0; token <= position; ++token) {
        const size_t off = (size_t(token) * KvHeads + kv) * HeadDim + d;
        const float score = sum<HeadDim>(query[d] * f(keys[off]), scratch) * (1.0f / 16.0f);
        const float next = fmaxf(maximum, score);
        const float old_weight = expf(maximum - next), weight = expf(score - next);
        result = result * old_weight + weight * f(values[off]);
        denominator = denominator * old_weight + weight;
        maximum = next;
    }
    const float attended = rounded(result / denominator);
    const float gate = rounded(sigmoid(f(q_gate[h * 2 * HeadDim + HeadDim + d])));
    out[h * HeadDim + d] = b(attended * gate);
}

extern "C" __global__ void tf_qwen35_route(
    const B* x, const B* router, const B* shared_gate, int32_t* picks, float* weights) {
    __shared__ float logits[256];
    const int expert = threadIdx.x;
    float logit = 0.0f;
    for (int d = 0; d < Hidden; ++d) logit += f(x[d]) * f(router[expert * Hidden + d]);
    logits[expert] = rounded(logit);
    __syncthreads();
    if (expert != 0) return;
    float chosen[8];
    for (int slot = 0; slot < 8; ++slot) {
        int best = -1;
        for (int candidate = 0; candidate < 256; ++candidate) {
            bool used = false;
            for (int prior = 0; prior < slot; ++prior) used |= picks[prior] == candidate;
            if (!used && (best < 0 || logits[candidate] > logits[best])) best = candidate;
        }
        picks[slot] = best;
        chosen[slot] = logits[best];
    }
    float denominator = 0.0f;
    for (int slot = 0; slot < 8; ++slot) {
        weights[slot] = expf(chosen[slot] - chosen[0]);
        denominator += weights[slot];
    }
    for (int slot = 0; slot < 8; ++slot) weights[slot] /= denominator;
    float shared = 0.0f;
    for (int d = 0; d < Hidden; ++d) shared += f(x[d]) * f(shared_gate[d]);
    weights[8] = sigmoid(rounded(shared));
}

extern "C" __global__ void tf_qwen35_route_products(
    const B* x, const B* router, const B* shared_gate, float* scores) {
    __shared__ float products[Hidden];
    const int expert = blockIdx.x;
    const B* row = expert < 256 ? router + expert * Hidden : shared_gate;
    for (int d = threadIdx.x; d < Hidden; d += blockDim.x)
        products[d] = __fmul_rn(f(x[d]), f(row[d]));
    __syncthreads();
    if (threadIdx.x != 0) return;
    float value = 0.0f;
    for (int d = 0; d < Hidden; ++d) value = __fadd_rn(value, products[d]);
    scores[expert] = expert < 256 ? rounded(value) : value;
}

extern "C" __global__ void tf_qwen35_route_select(
    const float* scores, int32_t* picks, float* weights) {
    __shared__ float logits[256];
    logits[threadIdx.x] = scores[threadIdx.x];
    __syncthreads();
    if (threadIdx.x != 0) return;
    float chosen[8];
    for (int slot = 0; slot < 8; ++slot) {
        int best = -1;
        for (int candidate = 0; candidate < 256; ++candidate) {
            bool used = false;
            for (int prior = 0; prior < slot; ++prior) used |= picks[prior] == candidate;
            if (!used && (best < 0 || logits[candidate] > logits[best])) best = candidate;
        }
        picks[slot] = best;
        chosen[slot] = logits[best];
    }
    float denominator = 0.0f;
    for (int slot = 0; slot < 8; ++slot) {
        weights[slot] = expf(chosen[slot] - chosen[0]);
        denominator += weights[slot];
    }
    for (int slot = 0; slot < 8; ++slot) weights[slot] /= denominator;
    weights[8] = sigmoid(rounded(scores[256]));
}

extern "C" __global__ void tf_qwen35_combine(
    const float* pairs, const float* weights, const B* shared, B* out) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= Hidden) return;
    float y = 0.0f;
    for (int slot = 0; slot < 8; ++slot) y += pairs[slot * Hidden + d] * weights[slot];
    out[d] = b(y + f(shared[d]) * weights[8]);
}
