// Numerical subset of Mia f013e1ae glm_ops.cu. Ash Hart and TensorFold contributors.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <math.h>
extern "C" __global__ void glm_sort(float* vals, int64_t* ids, uint32_t k) {
    uint64_t base = uint64_t(blockIdx.x)*k;
    for (uint32_t phase=0; phase<k; ++phase) {
        for (uint32_t i=2*threadIdx.x+(phase&1); i+1<k; i+=2*blockDim.x) {
            float a=vals[base+i], b=vals[base+i+1];
            if ((!isnan(a) && isnan(b)) || a<b) {
                vals[base+i]=b; vals[base+i+1]=a;
                int64_t id=ids[base+i]; ids[base+i]=ids[base+i+1]; ids[base+i+1]=id;
            }
        }
        __syncthreads();
    }
}
extern "C" __global__ void glm_pack(const float* v, const int64_t* ids, float* out,
                                    uint32_t k, int64_t offset) {
    uint64_t r=blockIdx.x;
    for(uint32_t j=threadIdx.x;j<k;j+=blockDim.x) {
        out[r*2*k+j]=v[r*k+j];
        out[r*2*k+k+j]=__int_as_float(int32_t(ids[r*k+j]+offset));
    }
}
extern "C" __global__ void glm_index_copy(const char* src, char* dst, const int64_t* idx,
    uint64_t rows, uint64_t dest_rows, uint64_t inner_bytes, uint64_t outer, uint32_t* invalid) {
    uint64_t r=blockIdx.x; int64_t to=idx[r];
    if(to<0 || uint64_t(to)>=dest_rows) { if(threadIdx.x==0) atomicExch(invalid,1u); return; }
    for(uint64_t o=blockIdx.y;o<outer;o+=gridDim.y)
        for(uint64_t b=threadIdx.x;b<inner_bytes;b+=blockDim.x)
            dst[(o*dest_rows+to)*inner_bytes+b]=src[(o*rows+r)*inner_bytes+b];
}
extern "C" __global__ void glm_rotary(const int64_t* pos, const float* inv, float* c, float* s,
                                      uint64_t rows, uint64_t half) {
    for(uint64_t j=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;j<rows*half;j+=uint64_t(gridDim.x)*blockDim.x) {
        float angle=float(pos[j/half])*inv[j%half];
        c[j]=cosf(angle); s[j]=sinf(angle);
    }
}
extern "C" __global__ void glm_ring(const int64_t* pos, const int64_t* slots, int64_t* out,
                                    uint64_t rows, int64_t cap) {
    for(uint64_t j=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;j<rows;j+=uint64_t(gridDim.x)*blockDim.x) {
        int64_t rem=pos[j]%cap; if(rem<0) rem+=cap;
        out[j]=slots[j]*cap+rem;
    }
}
extern "C" __global__ void glm_rank_add(const float* ranks, __nv_bfloat16* out, uint64_t count, uint32_t world) {
    for(uint64_t j=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;j<count;j+=uint64_t(gridDim.x)*blockDim.x) {
        float acc=ranks[j]; for(uint32_t r=1;r<world;++r) acc=acc+ranks[uint64_t(r)*count+j];
        out[j]=__float2bfloat16_rn(acc);
    }
}