#pragma once
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>

namespace glm_cache {

__host__ __device__ __forceinline__ uint32_t scale_bits(uint32_t amax) {
  int e = ((amax >> 23) & 255) + ((amax & 0x7fffff) != 0) - 8;
  e = e < 1 ? 1 : (e > 254 ? 254 : e);
  return static_cast<uint32_t>(e) << 23;
}

__host__ __device__ __forceinline__ uint8_t encode(float v) {
  return __nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3);
}

__host__ __device__ __forceinline__ float decode(uint8_t code) {
  __nv_fp8_e4m3 v;
  v.__x = code;
  return static_cast<float>(v);
}

#ifdef __CUDACC__
template <int W>
__device__ __forceinline__ float scale(const void* cache, int64_t row) {
  return *reinterpret_cast<const float*>(static_cast<const uint8_t*>(cache) + row * (W + 16) + W);
}

template <int W>
__device__ __forceinline__ void load8(__nv_bfloat16* dst, const void* cache, int64_t row, int column, bool ok) {
  const uint8_t* src = static_cast<const uint8_t*>(cache) + row * (W + 16) + column;
  for (int j = 0; j < 8; ++j) dst[j] = __float2bfloat16_rn(ok ? decode(src[j]) : 0.0f);
}

// One row per block, N adjacent columns per thread. Round pooled values to BF16 before quantization.
template <int W, int N, bool FP8>
__device__ __forceinline__ void store(void* cache, int64_t row, const float (&values)[N]) {
  constexpr int THREADS = W / N;
  float x[N], largest = 0.0f;
  for (int j = 0; j < N; ++j) {
    x[j] = __bfloat162float(__float2bfloat16_rn(values[j]));
    largest = fmaxf(largest, fabsf(x[j]));
  }
  if constexpr (!FP8) {
    for (int j = 0; j < N; ++j)
      static_cast<__nv_bfloat16*>(cache)[row * W + threadIdx.x * N + j] = __float2bfloat16_rn(x[j]);
  } else {
    for (int offset = 16; offset > 0; offset >>= 1)
      largest = fmaxf(largest, __shfl_xor_sync(0xffffffffu, largest, offset));
    if constexpr (THREADS > 32) {
      __shared__ float maxima[THREADS / 32];
      if ((threadIdx.x & 31) == 0) maxima[threadIdx.x / 32] = largest;
      __syncthreads();
      largest = 0.0f;
      for (int warp = 0; warp < THREADS / 32; ++warp) largest = fmaxf(largest, maxima[warp]);
    }
    const uint32_t bits = scale_bits(__float_as_uint(largest));
    const float inverse = __uint_as_float((254 - (bits >> 23)) << 23);
    uint8_t* dst = static_cast<uint8_t*>(cache) + row * (W + 16);
    for (int j = 0; j < N; ++j) dst[threadIdx.x * N + j] = encode(__fmul_rn(x[j], inverse));
    if (threadIdx.x == 0) {
      *reinterpret_cast<uint32_t*>(dst + W) = bits;
      for (int j = 4; j < 16; ++j) dst[W + j] = 0;
    }
  }
}

#endif
}
