// Adapted from jayleaton/glm53-tensorfold-spark patch 0460, Apache-2.0, Copyright 2026 Jay Leaton.
#include <cuda_runtime.h>
extern "C" __global__ void glm_l2pf(const unsigned long long* __restrict__ table, int n, int mode, unsigned* __restrict__ sink) {
    const int lane = threadIdx.x & 31;
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int warps = (gridDim.x * blockDim.x) >> 5;
    if (mode == 0) {
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
            const unsigned long long addr = table[2 * i];
            const unsigned bytes = (unsigned)table[2 * i + 1];
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
            asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(addr), "r"(bytes) : "memory");
#else
            for (unsigned offset = 0; offset < bytes; offset += 128)
                asm volatile("prefetch.global.L2 [%0];" :: "l"(addr + offset));
#endif
        }
        return;
    }
    unsigned acc = 0;
    for (int i = warp; i < n; i += warps) {
        const unsigned long long addr = table[2 * i];
        const unsigned bytes = (unsigned)table[2 * i + 1];
        for (unsigned offset = lane * 128u; offset < bytes; offset += 32u * 128u) {
            if (mode == 1) {
                asm volatile("prefetch.global.L2::evict_last [%0];" :: "l"(addr + offset));
            } else {
                unsigned x, y, z, w;
                asm volatile("ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%4];"
                             : "=r"(x), "=r"(y), "=r"(z), "=r"(w) : "l"(addr + offset));
                acc ^= x ^ y ^ z ^ w;
            }
        }
    }
    if (mode == 2 && acc == 0x9e3779b9u && sink != nullptr) sink[0] = acc;
}
