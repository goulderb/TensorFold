# Numerical diagnosis after shared-host integration

The corrected shared host passes thirteen HTTP checks on the internal GPU
pair. That result closes the bounded lifecycle unit. It does not close
the four failed full-logit rows or establish production readiness.

The user authorized an hour of work and retesting beginning at 00:07:24 UTC
on October 19. The corrected lifecycle attempt restored Nemotron with exact
proxy READY at 00:11:35 UTC. A later numerical capture restored the preserved
runtime and verified proxy READY at 00:18:50 UTC. Clef remained unchanged.
A second diagnostic restored it with exact proxy READY at 00:34:44 UTC.

## Capture the first unexplained difference

The temporary diagnostic uses source `973428fb` plus a retained patch that
only adds synchronization and first-layer downloads. Native probe build
passes 56/56 steps. The diagnostic executable has SHA256
`37a6775ff53d67e6d43e4c80248b9afc4b2289829867c5759680e397f9726ed0`.
The patch has SHA256
`2de7e6e47c1424c23427e60fa7336503c7331ec61fe56b7dfdc70921c0133486`.
The exact original engine source bytes were restored after building. No
instrumentation remains in the task source or immutable serving runtime.

The model probe exits zero, passes reset and interleaved-session isolation,
and produces both short-case logit rows byte-identical to the preserved
final serial rows. This validates capture neutrality for those two rows.
It does not rerun the chat case. The detached controller owns its probe
handle, samples physical memory once per second, and restores Nemotron in
`finally`. Its receipt reports no cleanup or sampler failure and unchanged
Clef identity.

The capture contains sixty arrays across six first-layer positions: MoE
input, picked expert IDs, routing weights, expert activations, FP32 down
outputs, shared gate/up/activation/output, and final MoE output. Receipts,
patch, scripts and arrays are under
`.artifacts/numerical-hour-trace-20261019/`.

## Explain the operator difference before changing math

The native NVFP4 grouped kernel computes sixteen-term dots on unscaled FP4
codes and applies each FP8 block scale through an ordered FP32 `fmaf`.
The CPU adapter dequantizes block-scaled weights before its full-width dot.
The equations are algebraically equivalent but have different rounding and
reduction order. See `zig/kernels/cuda/nvfp4_experts.cu:74` and
`tools/zig/qwen35_reference.py:164`.

The CPU replay on identical captured inputs compares both policies. Applying
the scale after each sixteen-term dot matches all 24,576 BF16 expert
activations exactly. Dequantizing first differs in three elements across
those positions. On captured native activations, the block-scaled down
replay still has 844 differing FP32 outputs out of 98,304, with maximum
absolute error 2.2351741790771484e-8. The replay uses an ideal local sum;
the remaining local MMA reduction behavior is not independently reproduced.

Combining the replayed block-scaled down outputs with captured native routing
weights and shared output reproduces five of six complete MoE rows exactly.
The sixth differs in one of 2048 BF16 elements by 9.5367431640625e-7. This
explains the first observed activation divergence and narrows the remaining
operator discrepancy. It does not bound error across forty layers.

Explicit fused-versus-separate final combination does not change the earlier
BF16 discrepancy. The Qwen glue build already sets `--fmad=false`.
Astra independently confirmed the block-scale policy difference and found
no concrete indexing or equation bug in the reviewed paths. Its CPU SiLU
sweep covers 34,152 finite BF16 patterns below magnitude 90; the tested CPU
formulas agree. That does not establish CUDA `expf` identity.

## Treat router ties as a separate policy

On identical captured position-one input, native routing selects expert 72
at the cutoff and the pinned HF router formula selects expert 231. The BF16
cutoff logits tie. A sequential CPU replay with lowest-ID tie selection
matches all forty-eight captured native selections. The different order at
position five also appears without changing the selected set.

PyTorch documents that tied `topk` indices can vary. Its
[2.9 API contract](https://docs.pytorch.org/docs/2.9/generated/torch.topk.html)
does not promise a fixed tie choice. This source fact and the same-input
observation prevent interpreting every routed full-logit difference as a
layout bug. They do not establish model quality or authorize a looser gate.

## Keep the full-model gate explicit

A separate CPU diagnostic uses the observed block-scale policy for routed
experts and lowest-ID router ties, plus the earlier native-style FP32 MoE
and attention variant. It preserves the original HF captures and all limits.
Both short rows still fail: RMSE 0.119473/0.162541, maximum error
0.6875/0.9375 and KL 0.007487/0.001079, with equal argmax. Both chat rows also
fail: RMSE 0.452978/0.294949, maximum error 1.96875/1.46875 and KL
0.001057/0.000049, with equal argmax. The chat comparison uses the retained
prior CUDA rows; no new chat device capture is implied.

The premise that aligning these operator policies alone closes the full
gate is falsified. No further arithmetic change should be justified solely
by shrinking the same four logit errors. Instrument the first remaining
same-input divergence and establish an independent serving/quality baseline.
This diagnostic is another CPU arithmetic reference, not that independent
runtime. Its scripts, receipts and comparisons are under
`.artifacts/numerical-hour-full-reference-20261019/`.

## Inspect the remaining dense and normalization boundaries

The aligned CPU trace matches all first-position layer outputs through layer
13. Its first difference is at layer 14. At position one the first differing
layer is 3. A second temporary capture therefore records those two layers
at all six positions, producing 174 arrays. Probe build passes 56/56 steps;
reset and interleaved-session checks pass; both short logit rows remain
byte-identical to the original CUDA rows. This capture again leaves no source
instrumentation behind. The diagnostic binary hash is
`359c4f7dd91a61b4a36a982792a789e0ee4c52931ab580696368d157b1f68dd7`;
its patch hash is
`d18c9c67636f03d0b9d8a21b025825c75c5122f9d467ba47c1d50095056e4cb4`.

At layer 3, position one, normalized input matches exactly. Two of 8192
Q-projection outputs and one of 512 V-projection outputs differ in BF16;
all 512 K outputs match. At layer 14, position zero, input normalization
and mixer output match exactly, then one of 2048 post-attention normalized
values differs. These are earlier boundaries than attention or MoE mixing.

Higher precision CPU replays put the affected projection coordinates about
1.4 to 3.5 FP32 ulps from BF16 rounding midpoints. The layer-14 normalization
coordinate is about 0.56 FP32 ulp from its midpoint. CPU is closer to the
FP64 replay in three affected coordinates and native in one. This is
consistent with reduction and rounding differences, rather than evidence
that either implementation is universally more accurate. GPU quantized input
codes were not captured, so the analysis does not prove their equality.
The observations do not establish harmless accumulated error or model quality.

The controller restored the preserved Nemotron executable with proxy HTTP200,
exact READY and no paid spillover at 00:34:44 UTC. Clef PID 21392 is unchanged;
the receipt reports no sampler or cleanup failures. Complete comparison,
higher precision script, source patch and recovery evidence are under
`.artifacts/numerical-hour-trace2-20261019/`.

Kimi's independent numerical review exceeded its timebox and was cancelled.
It supplied no completed review and is not credited as one. Astra's completed
review and the captures support the narrower operator findings above.

The original fixed limits remain RMSE 0.05, maximum error 0.5, KL 0.001 and
equal argmax. No production promotion, sizing claim or speed claim follows.
Multiple lanes, useful batched prefill, cache, full-window occupancy, soak
and restart qualification remain open. The full fleet promotion gate is
reserved for a candidate that passes the prerequisite correctness gates.
