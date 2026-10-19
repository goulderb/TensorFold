# Qwen 35B decode leaves graph replay and mature mixer kernels unused

The current port takes 32.674 ms per decoded token in the saved direct greedy
64-input/128-output trial. Reaching 80 tok/s requires 12.5 ms per token, so
20.174 ms, or 61.74% of current latency, must go. The source contains substantial
optimization opportunities. Their individual timings remain unmeasured.

Fable 5.1 and Astra independently reviewed the decode path. Fable prioritizes
host submission and graph replay. Astra identifies inefficient Qwen-specific
router, recurrent-update, and full-attention kernels as competing causes.
The next experiment must distinguish these causes before a large kernel port.
The condensed [Astra findings](evidence/performance-deep-dive-20261019/astra-r1-notes.md)
retain its exact donor pins, accounting corrections, and experiment thresholds.

The preceding head micro measured the whole resident-residual through
normalization, projection, synchronization, and logits download at 0.435 ms.
That isolated cost is too small to explain the 20.174 ms gap. The shared greedy
scan already reduced the sampler's first-draw cost from about 17.5 to 0.31 ms.
Further sampler or head work has a much smaller plausible payoff than the
remaining model path.

## Logical weight traffic does not rule out 80 tok/s

The baseline is five alternating paired measurements from source
`312d4f5d950a2c95cd2028e9136c742928841523`. Current decode is median 30.605 tok/s,
with 32.595 to 32.702 ms per token. It uses one direct synthetic greedy session,
a layer split at 20, and context capacity 4096. It does not establish HTTP,
concurrency, long-context, or model-quality performance. This investigation
replays those saved measurements and does not execute a new device trial.

The CPU-only checkpoint-header audit finds 2,254,637,556 logical weight and
scale bytes per token, including eight selected experts per layer, the head,
one embedding row, and final norm. All 256 expert shapes are checked before
using eight representative experts for the byte sum. The mixer projections
are largely FP8, so treating all active weights as FP4 understates the work.

The 20/20 partition has approximately 0.984 and 1.270 GB of these bytes.
At the RTX 5080 and 5070 Ti's advertised 960 and 896 GB/s, a serial,
uncached, one-read streaming calculation gives 2.443 ms. This is an idealized
reference, not a measured DRAM cost or a critical-path lower bound. Repacking,
padding, repeated reads, activations, recurrent state, KV, and caches change
actual traffic. The calculation makes 12.5 ms plausible but cannot promise it.
The audit assumes partition 0 uses the 960 GB/s card and partition 1 the
896 GB/s card. It does not independently verify CUDA ordinal assignment.
[NVIDIA's published bandwidths](https://www.nvidia.com/en-us/geforce/news/rtx-50-series-graphics-cards-gpu-laptop-announcements/?linkId=100000326874210)
support the hardware inputs.

## The serial path has several substantial candidates

`zig/src/families/qwen3_5/cuda_engine.zig:169` performs every layer's mixer and
MoE work serially. `Session.append` at line 383 finishes the first partition,
copies a 4 KiB residual through pinned host memory, then executes the second
partition and downloads 496,640 bytes of BF16 logits. This is layer splitting,
not tensor parallelism.
The benchmark also prefills through repeated one-token append calls. There is
no batched prompt path in this measurement. Decode tuning and proper batched
prefill are separate work needed for a runnable production instance.

The static selected-path launch ledger is 953 per token. This count is derived
from source and checkpoint projection formats, not an observed API trace:

| Work | GDN layer | Full-attention layer |
| --- | --- | --- |
| Input RMS | 1 | 1 |
| Input projections | 6, two FP8 pairs plus BF16 a and b | 6, three FP8 pairs |
| Mixer kernels | 3 | 2 |
| Output FP8 projection | 2 | 2 |
| Residual adds and post RMS | 3 | 3 |
| Router and grouped plan | 2 | 2 |
| Routed expert up and down | 2 | 2 |
| Separate shared expert | 4 | 4 |
| Combine | 1 | 1 |
| Per-layer total | 24 | 23 |

Thirty GDN layers and ten full-attention layers give 950 launches. Embedding,
final RMS, and the head add three. Closing the whole gap by submission alone
would require approximately 21.17 microseconds of removable time per launch.
This is a threshold equation, not a measured launch cost.

| Priority | Source observation | Smallest useful discriminator |
| --- | --- | --- |
| 1 | No graph capture in the Qwen engine; hundreds of individual launches per token. Nemotron already uses the shared graph API. | Compare current submission with an identical fixed-sequence graph replay on the real stream, with output and recurrent-state checks. |
| 2 | The Qwen router uses one 256-thread CTA. Each thread computes a serial 2048-term dot product, then thread zero selects eight experts and computes the shared gate. | Time the actual 2048 by 256 router seam, including top-eight and shared gate. Compare the pinned mature tensor-core router with explicit rounding controls. |
| 3 | GDN update launches 32 CTAs. Each thread owns one value row and loops over 128 keys twice with strided state accesses. The mature GDN kernel distributes contiguous keys across warp lanes. | Compare nontrivial changing recurrent state and outputs against the existing CPU reference before timing both implementations. |
| 4 | Full attention launches 16 CTAs and scans positions serially. Each position invokes a shared-memory reduction with ten block barriers. | Measure the real head geometry at short and longer positions to separate fixed latency from context growth. |
| 5 | Shared experts run separately, and the selected grouped-expert variant is not the only available donor variant. | Screen existing variants at Qwen's actual 2048/512 geometry and SwiGLU semantics. |

These priorities choose experiments, not measured causal shares. Removing the
whole 20.174 ms gap solely through 40 routers requires 0.504 ms saved per
router. Through 30 GDN updates it requires 0.672 ms per update. Through ten
attention calls it requires 2.017 ms per call. These equations help reject a
candidate once a timing exists.

The preferred donors are already pinned locally. Upstream v0.6.6
`cb2ebf0540f42604e2759b2ddef497861e928248` contains the warp-distributed GDN
and tensor-core router/shared-expert implementations. The current tree also
has graph support in `zig/src/cuda/graph.zig` and Nemotron's engine. Existing
expert variants originate in `87b322035fd68e5ba4be04c3a6ae8371fe14f25e`.
These are concrete reuse candidates, not a request to invent another stack.

The final audit also found an in-family donor we initially missed. Qwen's HIP
backend already captures lane rounds by shape in
`backend/hip/engine/round_graphs.zig` and `round.zig:132`, including staged token
and position metadata before replay. Relevant history is
`c67a80678e7534f867e251d21ecf6973a563f78c`, by BlivionIaG, present at the frozen
runtime pin. Reuse that family's changing-input design with the shared CUDA
graph API. HIP calls and TP collectives are not direct CUDA replacements.
Graph absence here refers specifically to the measured CUDA engine.

GDN's existing specialization uses BF16 Q/K where current Qwen uses FP32.
The historical router ranks FP32 logits and rounds route weights to BF16;
the current router ranks BF16-rounded logits and retains FP32 weights.
Those differences require explicit specialization or numerical validation.
Retain upstream, Apple/MLX, DeltaNet, Qwen/Hugging Face, and other transitive
credits if adapting code. This investigation imports no donor code.

## Claims from the first Fable report that need correction

The verbatim [Fable report](evidence/performance-deep-dive-20261019/fable-r1.md)
preserves the independent analysis. Its opening conclusion that this is
definitively a dispatch problem exceeds the evidence. Neither a host API trace
nor per-kernel timings exist for the saved trial. The proposed 15 to 22 ms
submission saving is an estimate without a local measurement.

`qmmf.Lane.load` explicitly enables clusters for capability major 9 and above.
The suggested missing-cluster reduce pass is therefore rejected by source.
The actual FP8 checkpoint path is `quant8` followed by `ck.lane8`, not the
generic qmmf FP8 path. The current expert kernels unpack FP4 pairs to BF16 and
use BF16 tensor-core MMA. Native FP4 arithmetic is a separate mathematical and
kernel-compatibility question.

There are three explicit stream synchronizations in the complete token path,
including the legacy/null-stream synchronization inside residual upload.
There are four context-bind calls because each of the two memory queries binds
its context internally. The path also has two blocking device-to-host copies,
one host-to-device copy, and two memory queries. These are call counts, not
measured latency attribution. `HostBuffer` is pinned by construction.

Fable's bandwidth calculation assumes too many projections are FP4. Use the
checkpoint-header audit above. Nemotron's production rate is a useful local
engineering precedent for graphs, but it is a different model and configuration
and does not establish a Qwen speedup. No threshold such as 150 tok/s for
needing tensor parallelism is supported.

## The next experiment answers one decision

The question is whether reducing submission overhead or replacing a
Qwen-specific kernel has the larger material payoff. Keep the existing model,
checkpoint, quantization, partition split, outputs, and numerical gates fixed.

Prepare a benchmark-only event/API ledger with preallocated events. Record CPU
submit time, the three synchronization intervals, copies, memory-query time,
and GPU stream intervals around router, GDN, attention, expert, and projection
groups. Read events after the existing synchronization boundaries. Do not add
a synchronization after every layer or kernel.
Keep the per-token GPU floor guard throughout the initial experiment. Timing
the query does not authorize removing or weakening the 1536 MiB floor.

Broad CUDA-event intervals include stream idle time caused by late host
submission. They are not equivalent to kernel-busy time. Separate the current
submit trace from a bounded identical graph replay or a proper API/kernel trace
before concluding that host launch overhead dominates. Record instrumentation
overhead against an uninstrumented control; do not publish profiled throughput
as a speedup.

[NVIDIA's graph-launch analysis](https://developer.nvidia.com/blog/constant-time-launch-for-straight-line-cuda-graphs-and-other-performance-enhancements)
supports testing repeat graph launch for a long serial kernel chain. Its
published microseconds are from different hardware and Ubuntu, so none are
transferred to this Windows result. The
[CUDA 12.8 event API](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-driver-api/group__CUDA__EVENT.html)
defines elapsed stream-event time; it is not a device utilization measurement.

Use a short thin-kernel proof for the largest measured component, with five
alternating pairs and fixed correctness checks. Only a candidate with a
repeatable material payoff receives an unprofiled full-model A/B. Preserve
1536 MiB free on each internal GPU. A new production outage needs a new approved
window; the previous one ended at 12:49:24 UTC.

Prepare fixed router inputs from the six saved numerical captures, nonzero
multi-step GDN fixtures, and expert shapes 2048 by 512 with eight real routes.
Use rotating weight banks to avoid letting artificial cache reuse decide a
variant. Screen attention at 64, 128, 288, 1024, and 4096 positions. First test
identical fixed-input eager versus graph submission, then the router and GDN
if GPU execution remains material. A measured 1.63 ms/token removable share
clears a 5% decode screen. The integrated trial must separately show at least
5% lower complete-turn latency and unchanged fixed work and output checks.

Exact graph replay must preserve changing tokens, positions, KV and recurrent
state, reset behavior, and consumed BF16 rows. A fixed-position graph can prove
submission cost only. Replaying stale token or position arguments is not a
valid decode speedup. A changed reduction order needs its own fixed numerical
reference gate. Existing full-model numerical comparison failures remain open.

## Reproduce the CPU evidence

Run with the fleet Python from the TensorFold task worktree:

```powershell
& C:/Development/local-ai-server/runtime/proxy-venv/Scripts/python.exe docs/plans/qwen35-ornith/evidence/performance-deep-dive-20261019/replay_budget.py
& C:/Development/local-ai-server/runtime/proxy-venv/Scripts/python.exe docs/plans/qwen35-ornith/evidence/performance-deep-dive-20261019/checkpoint_byte_audit.py
```

`replay_budget.py --require-target` must fail with exit 1 against the saved
30.6 tok/s result. Its archive is already retained in
`evidence/device-window-20261019T1230/raw-evidence.zip`. The header audit requires
the staged immutable Ornith checkpoint at the path named in the script. It
reads metadata only, never initializes CUDA, and writes its JSON beside itself.

Production was restored after the preceding trial and is untouched by this
investigation. No runtime code, recipe, deployment, or serving configuration
changes are part of this unit.

The independent [Fable trail audit](evidence/performance-deep-dive-20261019/review-fable-r2.md)
accepts the investigation without blockers. Its flags and the parent responses
are retained in [review-response-r2.md](evidence/performance-deep-dive-20261019/review-response-r2.md).
