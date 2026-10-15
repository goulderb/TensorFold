# GLM native CUDA cache storage

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
inside the state arena. Engine-local snapshots retain them with the codes
in a separate device allocation. Engine settings and capacity stay fixed for a kept state's
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
can change replies and sparse selections relative to BF16. Owner KV and shared capacity are described below; their device qualification
remains pending. Prefix wait, durable spill and vision remain separate
production requirements.

## Raw index rings and retained prefixes

`cuda_layout.Layout` owns both the live arena offsets and the ranges a prompt
snapshot retains. Raw index keys and gates use at most 2112 rows: the 2048-row
prefill window plus three preceding rows, rounded up to 64. CUDA writes and
pool reads wrap those raw rows; pooled keys and latent rows keep absolute
context positions. The ring is capped at the sequence capacity.

A kept prompt state copies the fixed recurrent region, each visible latent
prefix, completed pooled keys, and the unfinished four-token pool's raw keys
and gates. MTP uses its own length. Both recurrent copies and pending-window
save areas remain in the fixed region. Save requires a committed prompt mark
with no drafted MTP rows. The byte estimate and both copy directions iterate
the same plan. Restore validates the requested resume position before copying
and regenerates device positions from host metadata.

Snapshots remain device-backed and process-local. Durable persistence remains a separate feature.

Host layout tests pack, poison and restore synthetic arenas in both formats,
including independent MTP lengths, ring wrap, invalid lengths and overflow.
A host driver mock also exercises the actual engine save/restore calls in
both formats, with MTP enabled and disabled. It checks exact saved bytes,
restored recurrent parity and positions, cleared pending/draft state, and
leaves invisible rows poisoned.
The compiled `glm-ref` suite also compares bounded-ring kernels with the
original full-context kernels using varied keys, gates and positional biases,
then restores an unfinished tail after reusing the rings. These device checks
must be executed during qualification before relying on the new layout.

## Shared capacity and owner-only latent storage

Both options default off. `TF_GLM_KV_POOL_TOKENS` enables one physical cache
pool with a positive token count aligned to 2048. `TF_GLM_KV_SHARD=1`
requires that pool, TP2, equal MLA head counts and a pool divisible by 4096.
Settings and aggregate capacity participate in rank agreement. This source
has not been device-qualified; the options do not select a production recipe.

Shared mode allocates latent and pooled-key planes once. Fixed recurrent
state and raw index rings remain private per slot. An admitted request
reserves prompt plus output allowance plus an eight-row write margin,
rounded to 2048. Owner mode divides the pool into two rank partitions;
a reservation cannot cross a partition. Placement favors fewer live streams,
then more free rows, then rank and base. Rank 0 announces the exact extent
before begin, and followers validate and apply it. Release returns the extent,
including slot zero. The existing lane host waits for temporary pool pressure;
permanently oversized requests fail during opening even when idle admission
bypasses the fit callback. Reserved requests do not need midstream growth.

Pool allocation uses the smallest post-load peer budget, including existing
memory limits and reserve. Scratch is allocated before this survey. Pool
allocation success is exchanged before startup continues. The native wrapper
then admits additional streams by their private state cost; pool bytes have
already reduced the observed room. Retained snapshots remain separate copies
and consume the existing snapshot budget, never held or aliased pool extents.

Only the latent owner writes and reads a conversation's latent rows. Raw index
updates and pooled keys stay replicated. For each bounded row tile, the other
head rank sends its absorbed queries to the owner and receives its result.
The owner also computes its own heads. Existing attention kernels, head-block
choices, output projections, hidden exchanges, sampler and scheduler remain
in use. The same attention seam handles prompt chunks, shared-window segments
and MTP. This baseline trades extra communication for bounded scratch; its
latency and throughput are unmeasured.

Snapshot data capture uses local copies, retaining latent rows on their
owner and the remaining ranges on every rank. Byte estimates conservatively
charge the worst owner. During paired restore, both ranks validate the saved
state and destination before any data transfer. Encoded latent rows and scales
move in bounded chunks if the new reservation has another owner; private and
pooled ranges restore locally. Host positions and parity publish after the
copies complete. No absent nonowner latent pointer is dereferenced.

Pooled views expose two masked padding rows. Those rows may overlap the next
extent's pooled prefix: the pooling writer writes only complete pools inside
the reservation. No padding holes are charged per extent.

Host checks exercise real allocation/binding and engine capture/restore with
mock CUDA memory and two communicating CPU threads. They cover placement
against the pinned donor, adjacent/reused extents, both ownership directions,
BF16/FP8, MTP, local query outputs and refusal before an invalid peer transfer.
The normal native build compiles the kernel-facing prefill/shared/MTP paths.
Device attention numerics, full TP2 generation, cancellation/peer loss and
performance remain pending. Live compaction/growth carry, parked-prefix
sharing, twins, compact prompt-history fetch and durable spill are deferred.

The initial shared pool supports at most 64 lanes. Automatic lane admission caps
at that limit; an explicit larger `--parallel` is refused. In sharded mode a request must fit within one owner's half of the pool.

Both ranks acknowledge each new reservation and prompt snapshot. A snapshot ID
is returned to the prompt cache only after both copies succeed; a failed capture
frees any successful peer copy. Invalid restores are refused before data transfer.
Once an owner attention or restore transfer fails, the rank terminates rather
than serving another request with a damaged pair. Device numerical parity,
concurrent full-length occupancy and failure recovery still require qualification.
