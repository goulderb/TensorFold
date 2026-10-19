# Moving beyond 81.7 tok/s

This is a source- and evidence-based plan after independent Astra, Fable5.1
and KimiK3 consultation. Source is 2daf7f4ef00753013137b57ba7b02629b9a77033.
The [completed device investigation](gpu-investigation-20261019.md) remains
the measured baseline. No new GPU experiment or outage occurred for this plan.

## Outcome and limits

Optimize resident, single-request decode and complete-turn latency first;
measure TTFT/prefill and aggregate lane throughput separately. The baseline
is 81.668 tok/s, 12.245 ms per decoded token, on synthetic 64/128 greedy c1,
4096 context, two serial GPU layer partitions. It is a diagnostic harness,
not the production server. Independent reference-model numerical gates fail.

The approximately 409 tok/s weight-stream calculation assumes one read of
2.255 GB of logical checkpoint weights/scales at peak per-card bandwidth.
It does not measure actual traffic, cache reuse, arithmetic, issue stalls,
state/KV traffic or synchronization. The 20% speed ratio is not bandwidth
utilization. No retained counters establish a DRAM-bound or issue-bound
verdict. The candidate phase graph itself adds 19.93% instrumentation overhead.

At this baseline, 100 tok/s requires 2.245 ms less decode time; 150 requires
5.578 ms less; 200 requires 7.245 ms less. These are requirements, not forecasts.
Each isolated expensive full trial must plausibly clear 5% complete-turn
latency (0.113 seconds from 2.264 seconds). A 5% decode-only screen is 0.612 ms
per decoded token; these gates are not interchangeable. For work affecting all
191 appends, 0.113 seconds corresponds to about 0.593 ms saved per append.

## Ordered experiments

1. **Expert tile/pipeline variants.** Qwen dispatches baseline at
   `zig/src/cuda/experts.zig:352`; pipe4, n16 and n16_pipe4 already exist.
   At top8, hidden2048 and intermediate512, up has 32 blocks while n16 exposes
   64 before the resident cap. Down has 128 baseline blocks. This is a concrete
   occupancy hypothesis, not measured SM utilization. Up/down phase intervals
   are 2.326/0.634 ms. Adapt the existing rotating-bank benchmark's hard-coded
   GLM geometry, preserving Qwen's SwiGLU and FP32 down outputs. Use real routes
   and multiple weight banks on both GPUs. Exact outputs, finite values and
   five alternating pairs are mandatory; no repeat-cached-bank speed claims.

2. **Split the residual router cost, then optimize the winner.** Its 3.106 ms
   interval combines two kernels. Time products and selection separately with
   an uninstrumented control. Selection still scans 256 scores eight times
   on one thread. Test parallel selection preserving strict greater-than,
   lowest-index ties, chosen order and the literal softmax accumulation.
   Define and test exceptional-value behavior before calling it exact; final
   finite logits do not establish finite intermediate router scores. Product
   staging can prefetch/unroll loads while retaining ascending __fadd_rn;
   inspect generated code first. No ordered-sum-to-tree substitution, and no
   last-block counter fusion before standalone payoff and race proofs.

3. **Coalesce GDN state while preserving math.** Each current thread owns a
   value row in head/value/key order; adjacent threads are 512 bytes apart
   for a fixed key. Test head/key/value storage with identical per-thread
   ascending key arithmetic and FP32 Q/K. Keep both state passes initially.
   Compare nonzero, changing multi-step outputs and inverse-transposed logical
   state exactly. Its 1.277 ms diagnostic cost needs approximately 48% saving
   to clear the decode screen. Register/shared staging or a warp-reduction
   donor is a separate experiment; the latter changes reduction order and the
   donor's BF16 Q/K cannot replace the current FP32 interface unchanged.

4. **Share compatible input quantization; tune actual projection shapes.**
   A header/scalar check finds equal input_scale values for QKV/Z in all30 GDN
   layers and Q/K/V in all10 attention layers of this checkpoint. Their weight
   scales differ. Reuse the same activation codes while preserving every
   projection's weight scale and accumulation order. GDN a/b weights are BF16
   in all30 layers, so a four-way FP8 GEMM is not an exact replacement. Candidate
   fusion needs per-segment weight scaling and appropriate mixed dtypes.
   This could save 50 quantization launches/token, but graph replay already
   removes most host dispatch cost; GPU saving must be measured. The 2.868 ms
   input/output mixer category is not a free fusion budget.

5. **Broader kernels only after identifying the resource limit.** Capture
   DRAM/L2 traffic, achieved occupancy, memory-sector efficiency, register
   spills, issue stalls and kernel gaps on winning representative seams if
   counter tooling is available. Use a separate profiling run; exclude it
   from timing. Native block-scaled FP4 MMA may accelerate expert arithmetic,
   but W4A4 adds activation quantization and needs a Qwen SwiGLU path and an
   independent numerical contract. Existing W4A16 expert weight bytes remain
   four-bit already. FP8 mixer weights are approximately 1.289 GB/token and
   do not become FP4 merely by changing expert kernels.

The first three proof runs use the smallest real saved-fixture seams, with
at most ten minutes of device execution each and separate protected recovery.
Prepare CPU inputs, builds, pins and commands first. Timed GPU work must be
isolated from production load under a bounded controller; CPU replay cannot
measure CUDA kernel speed. A losing proof gets no full-model trial. A winner
must pass the existing five-pair full changing-token/position turn protocol,
including untimed consumed rows, timed tokens/final state, work counters,
device UUIDs and the 1536 MiB floor on each internal GPU.

## Serving and larger changes

Integrate the already proven router and changing-position graph into the
experimental server as a separate verifiable unit. Bind graph lifetime to
session allocations and configuration; preserve admission, poison/reset,
position, logits, scheduler ownership and isolation. A graph captured with a
different route mode or stale state pointers is not reusable. HTTP, interleaved
lanes, cache/state restore and restart tests are required before deployment.

Run independent numerical diagnosis in parallel with exact-performance work.
The retained first differences include FP8 projection and normalization
rounding boundaries; route tie alignment alone did not close the full-logit
gate. Preserve existing thresholds and use an independently pinned serving
reference. Native FP4 activation math, new reduction trees and TP partial sums
must not be validated solely against the already failing port.

Proper batched prefill is a distinct TTFT workstream: the measured 0.709-second
prefill executes 64 one-token appends. It requires causal attention and GDN
state equivalence, not merely dispatching a larger matrix multiplication.
Multi-lane scheduling/pipelining could improve aggregate throughput, but needs
independent lane state, buffers and ownership. The current serial append does
not already pipeline requests, and no free doubling is established.

True TP may help c1 or aggregate throughput, but neither benefit is established.
Measure the existing Windows transport at actual 4KiB payloads and collective
counts on this pair before a port. Forty4KiB payloads at hypothetical25GB/s
take 6.55 microseconds of payload time, not6 milliseconds. Software, driver,
synchronization and collective latency can dominate. The ideal823tok/s
weight-only parallel model ignores these costs and smaller per-rank tiles.
Moving the cut between serial partitions cannot add their bandwidths for c1.

## Consultation reconciliation

The retained [Astra summary](evidence/bandwidth-strategy-20261019/astra-review-summary.md),
[Fable report](evidence/bandwidth-strategy-20261019/fable-r1.md) and
[Kimi report](evidence/bandwidth-strategy-20261019/kimi-r1.md) are proposals,
not three independent performance measurements.

Mechanical checks supersede these review claims:

- Kimi's runtime packing factor is not additional expert scale traffic over
  the audit: a32x32 tile is576bytes both before and after packing when its
  checkpoint scales are included. Padding exists elsewhere, but material
  overall inflation was not demonstrated by that argument.
- Kimi's four-FP8-projection premise is false; a/b are BF16. Activation scales
  match within the checked groups, while weight scales differ. A single
  global-scale concatenated GEMM is not established as exact.
- Kimi's TP payload arithmetic is off by1000, and aggregate-only TP benefit
  is unsupported. Neither consultation proves PCIe latency on this host.
- Fable's 2–3microsecond in-graph kernel floor, 75% bandwidth assumption,
  150–180tok/s forecast and attribution of the fixed-step/generation gap to
  seams lack measurements. Its phase targets are hypotheses, not promises.
- Fable's free two-lane pipelining/doubling claim requires scheduler and state
  work, and its CUDA fixture timing cannot be performed on the CPU.
- Neither a parallel tree GDN reduction nor a native W4A4 operator is exact
  merely because its algebra matches. Bit equality remains a measured gate
  for claimed exact variants.

All source checks are CPU-only. No serving defaults, quantization settings,
production pins or services are changed by this plan.
