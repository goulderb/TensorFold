# GLM native CUDA cache precision

The native GLM family accepts `TF_GLM_KV=bf16` or `TF_GLM_KV=fp8`.
Unset or empty means BF16. Every TP rank must use the same setting;
the startup fingerprint rejects a mismatch before loading weights.

FP8 applies to each DSA layer's latent cache and pooled index keys, including
the MTP layer. Raw index keys and gates remain BF16. The existing upstream
engine, scheduler, projections, sparse selection and chunk merge remain in use.

## Row representation and arithmetic

The format follows the deployed wojo runtime's `kv8.py`. Each FP8 row contains
E4M3 codes, a float32 power-of-two scale at byte offset `width`, and twelve
zero bytes. Its stride is `width + 16`. BF16 rows use `2 * width` bytes.
Latent width is 512; pooled index width is 128.

Writers round values to BF16 before computing the maximum magnitude and
quantizing. The scale is `2^(ceil(log2(amax)) - 8)`, clamped to a normal
float32 exponent. Zero rows use `2^-126`. CUDA uses exponent bits to select
that scale and its exact reciprocal.

Attention loads codes exactly into the existing BF16 shared tiles. It applies
each key's scale to FP32 score dots and to probabilities before their BF16
conversion for the value product. The softmax denominator uses unscaled
probabilities. Pooled-key scoring applies row scales before the existing
score transforms and head reduction.

`cuda_cache.Cache` keeps the pointer and format together. State allocation
derives row strides from that format for all DSA and MTP caches. Scales live
inside the state arena, so existing engine-local RAM snapshots copy them
with the codes. Engine settings and capacity stay fixed for a kept state's
lifetime. This does not implement a cross-process snapshot format.

## Verification

The host-only format check compiles the same conversion helpers the CUDA
writers use. It needs CUDA headers and a C++ compiler, but links no CUDA
runtime and performs no GPU calls:

```sh
g++ -x c++ -std=c++20 -I/usr/local/cuda/include \
  zig/tests/cuda/glm_cache_format_test.cu -o /tmp/glm-cache-format-test
/tmp/glm-cache-format-test
```

Host tests cover row strides, setting parsing and rank-format disagreement.
Build the real kernels and native executable with the pinned Zig compiler:

```sh
zig build test-cuda-host -j2
zig build native -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121 -j2
```

The existing `tf-cuda-test glm-ref` checks both cache formats. FP8 writers
are compared with an independent host quantizer, including code values,
scales, zero rows, padding and adjacent allocation guards. Dense and sparse
attention and pooled-key selection consume those caches and compare against
the dequantized host reference. Existing window/head-block comparisons and
zero-count sparse masking run for both formats.

The pooled-writer fixture uses identical keys and zero gates to isolate its
format conversion. Device qualification must also cover varied keys and
gates, MTP windows, and kept-state restore on the actual model.

CPU checks and compilation do not qualify device numerics, generation,
throughput or capacity. GPU comparisons remain pending. FP8 is lossy and
can change replies and sparse selections relative to BF16. Sharded KV/carry,
shared dynamic capacity, prefix wait, durable spill and vision remain separate
production requirements.
