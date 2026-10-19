# Measure eager submission and Qwen phase costs during the approved investigation

The user explicitly authorizes taking Nemotron down for this investigation on
2026-10-19. Prepare the probe while it serves. Each device run uses the existing
detached outage controller with a fresh absolute deadline, physical-UUID floor
monitoring and mandatory restored canonical completion. Clef and proxies stay up.

The initial question is whether identical fixed-step graph replay removes a
material share of the 32.7 ms decode path. The paired benchmark uses the real
layer body, both GPUs, the same host seam, logits download and per-token floor
queries. It holds token and position fixed while recurrent state evolves.
This is a submission experiment, not changing-token generation throughput.

Architecture candidates were independently reviewed by gpt-6-sol and
gpt-6-astra. The selected shape uses compile-time phase taps in the existing
block and a small diagnostic partition-enqueue method. The ordinary block
instantiates empty taps. The alternative was a driver API trampoline table
with a global function registry; its ABI/global-state complexity is unnecessary
for the first question. A fresh gpt-6-sol judge agrees with the selected shape.

Use an uninstrumented eager/graph pair for timing. A separate captured graph
records preallocated phase events, so intervals inside that graph do not
include late host submission between phases. Event instrumentation is still
extra GPU work and its timings are diagnostic, not headline throughput.

Establish nonzero state with 64 prompt tokens. Save and restore all persistent
state, scratch and seam input before each arm. Compare the diagnostic eager
step with the original append, all 16 intermediate graph/eager state and logits
digests, final digests for each pair, and event-graph versus ordinary-graph
digests. Use one warmup pair followed by five alternating measured pairs.
Finiteness and floor failures stop the probe. The independent numerical gate
remains open even when this self-parity passes.

The existing controller now accepts either its original head/sampler sequence
or a single pinned `tf-qwen35-decode-profile.exe`. All original source/binary/input
pins, timeout bounds, device UUIDs, private port checks and recovery contracts
remain. A test rejected the new single-profile plan before the parser changed;
the targeted controller suite then passed with 41 collected cases.

The first probe build failed because the family module root did not export the
new diagnostic constants. A separate Windows/build worker owns those compile
fixes. No device run or outage starts until the committed source, built binary
identity and CPU-only input validation agree.

The build worker resolved the facade exports and added cached-fatbin support.
The preliminary probe build passed 57 steps and the cached rebuild passed five.
The CPU-only host target passed 47 tests across eight build steps. The fleet
fast gate and 41 controller tests pass. Source is checkpointed before the final
source-pinned rebuild and the first device execution.

## First device run and bounded instrumentation repair

The first detached controller started at 13:48:15 UTC with source `db51c8f9`
and a 14:08:15 recovery deadline. It observed exactly 953 captured nodes and
passed the ordinary-append comparison, all sixteen intermediate graph/eager
state digests, and all six pair comparisons. The fixed-step paired median was
31.827 ms eager versus 28.340 ms graph. These remain partial diagnostic results
because phase timing then failed and the probe did not complete successfully.

`cuEventElapsedTime` returned CUDA_ERROR_INVALID_VALUE for default-captured
events. NVIDIA documents default events as graph dependency edges. The bounded
repair records explicit external event nodes with `cuEventRecordWithFlags`.
The ordinary eager/graph comparison is unchanged. A fresh pinned run repeats
the whole experiment before accepting phase results.

The controller restored Nemotron at 13:49:42 UTC as PID 16488 and obtained a
fresh canonical `READY`, request `9c2173d4-34cb-4603-9942-64371b17f4b9`, with
paid fallback disabled. Clef remained unchanged. The overall first window is
marked failed because its diagnostic child failed, despite successful recovery.

## Completed phase measurement and router experiment

The repaired run (`826b665e`) passed the entire detached window. Five measured
alternating pairs gave fixed-step medians 31.679 ms eager and 28.325 ms graph
(10.59% less latency). Explicit-event instrumentation adds 7.86% in its separate
control comparison. Its largest category is route: 20.058 ms across forty layers;
routed up is 2.329 ms, mixer inputs 2.011 ms, and GDN step 1.282 ms. These are
diagnostic intervals, not exact fractions of uninstrumented latency. Six pair
comparisons and sixteen intermediate state/logits digests match exactly.
Minimum physical free memory was 5358/6105 MiB. Nemotron restored at 13:53:39 UTC
as PID 19624 with canonical READY request d64f4219-04b8-4ba4-a367-36ecd794eefd.
Raw receipt and results remain in the fleet task worktree's
`.artifacts/qwen35-decode-profile-window-r2`; the analyzer rejects failed R1.

The router is the earned optimization target. Independent Astra and Sol
sketches considered cooperative product staging versus transposed weight
storage; a fresh Sol judge selected product staging to avoid a new checkpoint
layout. The build flags disable FMA contraction: exact parity requires a
separately rounded FP32 multiply, followed by the original ascending FP32 sum.
Each of 257 blocks computes one expert/shared gate's 2048 products into shared
memory, then its lane zero sums in the original order. A second block applies
the literal original strict-greater, ascending-index top-eight selection and
softmax order. The BF16 boundary and shared-gate sigmoid stay unchanged.
This preserves existing Qwen/Hugging Face attribution; the cooperative staging
is local work, with no external kernel code imported.

The original router stays the default. A diagnostic-only runtime flag selects
the candidate. The probe compares both routers for saved rows and zero ties
at all forty layers, all sixteen evolving fixed-step states, and every consumed
BF16 row and output token of a synthetic 64-input/128-output greedy turn.
Five alternating measured pairs then time both real complete turns and fixed
steps; warmup and correctness capture are excluded. Fixed-step graph timings
remain distinct from generation throughput. Independent reference-model
numerical qualification remains open regardless of self-parity or speed.

R3 (`c22bdf7f`) passed: five alternating real greedy pairs give 30.709 to
44.315 tok/s, with complete-turn medians 6.145 to 4.233 seconds. Every consumed
BF16 row and generated token matched. Its fixed-step graph median is 18.229 ms
for the candidate versus 28.336 ms baseline. This is material but short of the
80 tok/s objective. Nemotron restored at 14:09:03 UTC as PID 2148, fresh READY
request b0cdfaae-99ca-4421-ade4-051e53cd0867; floor 5356/6103 MiB.

The next isolated kernel change keeps selected IDs locally and unrolls its
eight slots, instead of consulting global output picks inside every candidate
scan. Traversal, strict comparison, chosen values and FP32 math stay unchanged.
Astra's source review identifies the 7168 prior-pick comparisons and repeated
pointer reads as credible cost, but source alone cannot prove generated load
counts or their runtime share. Candidate phase profiling now repeats the same
separate event/control graph protocol to expose the residual route cost.
