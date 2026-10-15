// Native GLM vision operator units. Compile without fast-math and with --fmad=false.
// BF16 stage boundaries follow da6282f9 vision/glm.py; CUDA reduction/activation
// numerical parity still requires the target GPU qualification window.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>

using bf16 = __nv_bfloat16;
__device__ __forceinline__ float f(bf16 x) { return __bfloat162float(x); }
__device__ __forceinline__ bf16 b(float x) { return __float2bfloat16_rn(x); }
__device__ __forceinline__ bf16 norm_value(bf16 x, bf16 weight, float inverse) {
    // _rms casts normalized x BEFORE multiplying the learned BF16 weight.
    return b(__fmul_rn(f(b(__fmul_rn(f(x), inverse))), f(weight)));
}

extern "C" __global__ void tf_glm_vision_cast(const float* x, bf16* out, int n) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < static_cast<unsigned int>(n)) out[i] = b(x[i]);
}
extern "C" __global__ void tf_glm_vision_rms(const bf16* x, const bf16* weight, bf16* out, int rows, int width, float eps) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    __shared__ float sums[256];
    const int lane = threadIdx.x;
    float sum = 0.0f;
    for (int j = lane; j < width; j += 256) {
        const float v = f(x[row * width + j]);
        sum = __fadd_rn(sum, __fmul_rn(v, v));
    }
    sums[lane] = sum;
    __syncthreads();
    for (int step = 128; step > 0; step >>= 1) {
        if (lane < step) sums[lane] = __fadd_rn(sums[lane], sums[lane + step]);
        __syncthreads();
    }
    const float inverse = rsqrtf(__fadd_rn(sums[0] / float(width), eps));
    for (int j = lane; j < width; j += 256) out[row * width + j] = norm_value(x[row * width + j], weight[j], inverse);
}

// Packed QKV [groups*S,3,heads,D] -> three contiguous [groups*S,heads,D] buffers.
// cos/sin are FP32 [S,D], in merge-block order and reused for every image group.
extern "C" __global__ void tf_glm_vision_qkv_rope(
    const bf16* packed, const bf16* qweight, const bf16* kweight,
    const float* cosine, const float* sine, bf16* q, bf16* k, bf16* v,
    int rows, int sequence, int heads, int dim, float eps) {
    const int row = blockIdx.x / heads;
    const int head = blockIdx.x % heads;
    if (row >= rows) return;
    const int lane = threadIdx.x;
    const int hidden = heads * dim;
    const int qi = row * 3 * hidden + head * dim;
    const int ki = qi + hidden;
    const int vi = qi + 2 * hidden;
    const int output = row * hidden + head * dim;
    __shared__ float sums[2][128];
    float qs = 0.0f, ks = 0.0f;
    for (int j = lane; j < dim; j += 128) {
        const float qj = f(packed[qi + j]), kj = f(packed[ki + j]);
        qs = __fadd_rn(qs, __fmul_rn(qj, qj));
        ks = __fadd_rn(ks, __fmul_rn(kj, kj));
    }
    sums[0][lane] = qs; sums[1][lane] = ks;
    __syncthreads();
    for (int step = 64; step > 0; step >>= 1) {
        if (lane < step) {
            sums[0][lane] = __fadd_rn(sums[0][lane], sums[0][lane + step]);
            sums[1][lane] = __fadd_rn(sums[1][lane], sums[1][lane + step]);
        }
        __syncthreads();
    }
    const float qinv = rsqrtf(__fadd_rn(sums[0][0] / float(dim), eps));
    const float kinv = rsqrtf(__fadd_rn(sums[1][0] / float(dim), eps));
    for (int j = lane; j < dim; j += 128) {
        const int other = (j + dim / 2) % dim;
        const float sign = j < dim / 2 ? -1.0f : 1.0f;
        const int position = (row % sequence) * dim + j;
        const float co = cosine[position], si = sine[position];
        const float q0 = f(norm_value(packed[qi + j], qweight[j], qinv));
        const float q1 = __fmul_rn(sign, f(norm_value(packed[qi + other], qweight[other], qinv)));
        const float k0 = f(norm_value(packed[ki + j], kweight[j], kinv));
        const float k1 = __fmul_rn(sign, f(norm_value(packed[ki + other], kweight[other], kinv)));
        q[output + j] = b(__fadd_rn(__fmul_rn(q0, co), __fmul_rn(q1, si)));
        k[output + j] = b(__fadd_rn(__fmul_rn(k0, co), __fmul_rn(k1, si)));
        v[output + j] = packed[vi + j];
    }
}
extern "C" __global__ void tf_glm_vision_residual(const bf16* x, const bf16* update, bf16* out, int n) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < static_cast<unsigned int>(n)) out[i] = b(__fadd_rn(f(x[i]), f(update[i])));
}
extern "C" __global__ void tf_glm_vision_swiglu(const bf16* gate, const bf16* up, bf16* out, int n, float limit) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= static_cast<unsigned int>(n)) return;
    const float g = f(b(fminf(f(gate[i]), limit)));
    const float u = f(b(fminf(fmaxf(f(up[i]), -limit), limit)));
    const bf16 activated = b(g / __fadd_rn(1.0f, expf(-g)));
    out[i] = b(__fmul_rn(f(activated), u));
}

struct Moments { float mean, m2; int count; };
__device__ __forceinline__ Moments combine(Moments a, Moments z) {
    if (z.count == 0) return a;
    if (a.count == 0) return z;
    const float delta = __fadd_rn(z.mean, -a.mean);
    const int count = a.count + z.count;
    const float fraction = float(z.count) / float(count);
    const float mean = __fadd_rn(a.mean, __fmul_rn(delta, fraction));
    const float correction = __fmul_rn(__fmul_rn(delta, delta), __fmul_rn(float(a.count), fraction));
    return {mean, __fadd_rn(__fadd_rn(a.m2, z.m2), correction), count};
}
extern "C" __global__ void tf_glm_vision_layernorm_gelu(
    const bf16* x, const bf16* weight, const bf16* bias, bf16* out, int rows, int width, float eps) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    const int lane = threadIdx.x;
    __shared__ Moments moments[256];
    Moments local{0.0f, 0.0f, 0};
    for (int j = lane; j < width; j += 256) local = combine(local, {f(x[row * width + j]), 0.0f, 1});
    moments[lane] = local;
    __syncthreads();
    for (int step = 128; step > 0; step >>= 1) {
        if (lane < step) moments[lane] = combine(moments[lane], moments[lane + step]);
        __syncthreads();
    }
    const float inverse = rsqrtf(__fadd_rn(moments[0].m2 / float(width), eps));
    for (int j = lane; j < width; j += 256) {
        const float normalized = __fmul_rn(__fadd_rn(f(x[row * width + j]), -moments[0].mean), inverse);
        const float affine = __fadd_rn(__fmul_rn(normalized, f(weight[j])), f(bias[j]));
        // F.layer_norm writes BF16, then F.gelu(approximate="none") consumes it.
        const float y = f(b(affine));
        const float z = __fmul_rn(__fmul_rn(y, 0.5f), __fadd_rn(1.0f, erff(__fmul_rn(y, 0.7071067811865475244f))));
        out[row * width + j] = b(z);
    }
}
