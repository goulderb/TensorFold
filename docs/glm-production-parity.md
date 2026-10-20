# GLM production execution parity

The native GLM candidate must match the production execution profile before
another round of kernel tuning. The comparison includes native FP4 Local as
the serving target and the same routed-only EXL3 checkpoint on both engines.
Checkpoint differences cannot explain an implementation result in that control.

The selected design uses the second architecture candidate. It keeps deferred
draft shape in the lane contract and one owned pending pass in GLM. Resolution
happens only when a queued neural head wins over a forced or copied continuation,
before allocating verification rows. Existing fixed-draft depth planning already
skips neural work for copy winners. Preserve that behavior and its proposer
accounting rather than introducing a second copy preflight.

The transport borrows the existing communicator through `cuda.tp_group.Group`.
Its optional RoCE resources have a shorter lifetime than the communicator.
Control and cache protocols retain their existing NCCL operations. Collective
order must also hold across graph replay and prompt side streams.

Q4 DSA uses one combined QA and KVA projection, followed by two strided norms.
The loader retains separate source-format and full-EXL3 projections, and the
BF16 recovery layout. Q4 input layout has a new numerical identity so saved
states from the earlier two-projection implementation cannot be reused silently.
Startup receipts report the loaded projection formats rather than only flags.

The production loader also combines dense MLP gate and up projections. Native
Q4 uses that representation and projects directly into the interleaved gate-up
rows, avoiding the separate output copies. Other formats retain their separate
input representation. Its numerical identity includes the combined layout.

L2 prefetch follows the production attention, FFN, and output sites, including
shared-expert computation before routed experts on the enabled decode path.
The planner owns immutable bounded weight spans. Its side stream and events
outlive every graph that references them, and every forward joins the stream.

Verification proceeds through host dispatch and scheduling tests, bounded paired
GPU proofs, then an unprofiled matched HTTP comparison. Device proofs cover Q4
numerics, deferred draft invalidation, RoCE graph replay, and prefetch event
ordering. Whole-model checks include tools, vision, cancellation, occupied
capacity, cache eviction, first restore after both restart types, and a soak.
Local-attention prefill now owns its replicas in the engine and invalidates
coverage on sequence release and restore. Focused pooled fixtures first
reproduced the stale-prefix failure, then passed with the repair at six and
eight layers. The repaired source passed first disk resumes after restart,
including a near-full request, the serving battery and a mixed cohort. Cold
HTTP first-token latency fell from 3.50 to 3.02 seconds at 4K and from 13.76 to
10.93 seconds at 16K, recovering the earlier fast path. Those results belong
to source `a7c93cdca2a65234bf07b4623169fdbf5725bfba`, not every later build.

Packed greedy selection preserves the existing local top-one kernel and tie
rule while gathering candidates for all rows once. Its TP2 model-free proof
passed exact token comparisons, including ties and nonfinite logits. At 32
rows it reduced component latency from 0.868 to 0.193 milliseconds; the
absolute saving is too small to explain the remaining whole-model deficit.
That is component evidence at `1d8f5efb4`, not a measured serving speedup.

The next source comparison identified missing production behaviors: 32-row
WY KDA prompt recurrence, overlap-level-2 front computation, and explicit Q4
prompt arithmetic below 128 rows. Native row partitioning alone did not
implement production's overlap. The port now preserves prompt mode instead of
inferring it from a sliced row count, and splits one contiguous padded half
per rank. The next block's owned-row front runs on the main stream after its
row swap is queued on the side stream. Dependent work waits for peer rows;
MoE fills consume prepared shared-expert output without recomputing it.

WY has a prompt-only interface with required absolute position and only
output/state destinations. Its BF16 convolution weights have separate owned
storage from serial decode's FP32 taps. Numerical identity separates WY,
explicit Q4 prompt arithmetic and overlap settings from older persisted
states. The existing 2048-token prompt checkpoint grid remains aligned to
WY's 32-row partition. Serial verification and prefix replay are unchanged.

The first model-free native WY proof matched pinned production output and
state bytes in twelve fixtures on both ranks, including boundary offsets,
aligned state/history restore and subsequent serial decode/replay. The
production-only isolated serial/WY A/B at 2048 rows measured 4.177 versus
1.699 milliseconds (five alternating samples). This is component evidence,
not a measured whole-model gain; the native serial launch differs and overlap
has a separate correctness/performance gate. Passing a source build or a
component proof does not establish production parity or authorize promotion.

The Q4 prompt path also has an explicit GLM-only production tile policy.
`TF_GLM_PROMPT_TILE=1` selects production tile 9 for K at least 1024 and
at least 96 logical 128-by-128 output blocks, otherwise tile 3. The default
retains tile 0. Both BF16 and FP32 prompt outputs use the policy, including
direct MLP callers. Tile selection owns the matching block dimensions,
shared-memory allowance and launch grouping. Other families and decode keep
their existing dispatch. Rank agreement and persisted numerical identity
include the setting.

Before this port, a model-free screen of the pinned production image compared
tile 0 with the selected production tile on twelve GLM projection shapes and
both output types. All 24 comparisons matched output bytes, and five
alternating samples showed lower component time on each tested shape.
Those measurements do not establish an HTTP prefill speedup. A separate
60-case screen of production's forced decode split-K table mostly increased
component time, so that table was not adopted as a performance change.
