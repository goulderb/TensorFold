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
// Local diagnostic change keeps qwen35_hf.cu arithmetic and Geo/Seg ABI,
// storing persistent recurrence as [head,key,value] for adjacent value owners.

#include <cuda_runtime.h>
#include <math.h>
#include "qwen35.cuh"

using namespace tf_q35;
using B = __nv_bfloat16;

extern "C" __global__ void __launch_bounds__(128) tf_q35_hf_gdn_chain_transposed(
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
    float* state = reinterpret_cast<float*>(sg.rec) + size_t(layer) * geo.rec() + size_t(h) * DK * DK + value;
    const float retain = expf(G[row * geo.nv + h]);
    float memory = 0.0f;
    for (int key = 0; key < DK; ++key) {
        state[key * DK] *= retain;
        memory += state[key * DK] * ks[key];
    }
    const float delta = (bf(V[(size_t(row) * geo.nv + h) * DK + value]) - memory) * bf(BETA[row * geo.nv + h]);
    float out = 0.0f;
    for (int key = 0; key < DK; ++key) {
        state[key * DK] += ks[key] * delta;
        out += state[key * DK] * qs[key];
    }
    Y[(size_t(row) * geo.nv + h) * DK + value] = out;
}
