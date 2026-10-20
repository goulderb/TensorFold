# ModelOpt as an upstream Qwen overlay

The user requires upstream tracking with an overlay suitable for a future
upstream submission. The separate ModelOpt backend retained at rebase checkpoint
6d382e0d2 is a frozen comparison artifact, not the target architecture. This
decision supersedes the temporary two-backend direction in the rebase report.
The Windows compatibility base is 986f826603e7cc008053060061be878b603b45b1.

## Acceptance and workflow

One generic Qwen layer body must execute MLX and ModelOpt checkpoints. The
ModelOpt native adapter and new performance probe must call this body through
the shared engine. Remove the isolated engine's callers after migration.
Keep historical source, notices, fixtures and measurements reproducible from
the archived branch. No old speed result qualifies the new implementation.

The port consists of four verifiable units: admission and arena loading;
numerical primitive bindings; ranged shared engines and their coordinator;
then native/probe caller migration. CPU shape, refusal, allocation and MLX
regression checks precede device work. First device acceptance is one row,
one lane, BF16 KV, no tree/record/vision/MTP/cache/spill, on the actual Ornith
checkpoint across both internal GPUs. Unsupported features fail before CUDA.
Compare intermediate operators, logits and state with retained captures.
Preserve the independent reference thresholds and their currently failed state.

After that gate, establish a fresh captured/eager five-pair baseline and resume
the scaling workflow. The previous 100 tok/s milestone is not a stop condition.
Account for every remaining material hypothesis before declaring diminishing
returns. A failed build, missing profiler permission or numerical blocker is
not evidence of a hardware scaling wall.

## Synthesis of the Astra and Sol designs

Retain upstream `cuda_engine.Engine` as one device's engine. Its existing
`init`/`forward` and MLX native/cache/MTP/vision interfaces remain wrappers for
a full-model placement. Add placement and numerical-profile initialization,
then a thin `cuda_pipeline.Chain` containing two ranged Engine instances.
Both instances call the same layer body. The coordinator contains ownership,
transfer and commit logic only. Nemotron's pinned seam is a precedent; its
model-specific forward or TP arithmetic is not copied.

Each stage has a global layer range and local geometry/state counts. Stage 0
owns embedding; stage 1 owns final norm and head. Pass both BF16 h and pending
r across the cut. The receiving first norm uses the global layer index. Do
not introduce an additional residual sum or rounding at the seam. Each stage
owns its graph map and device tables. Transfers remain outside capture.
Logical session lengths commit only after both stages succeed; partial failure
poisons the paired session until both caches reset.

The shared data contract is:

```zig
// cuda_checkpoint.zig, exported once through the upstream qwen35 module.
pub const Numerics = enum { mlx_sanitized, hf_modelopt };
pub const Encoding = union(enum) { mlx, modelopt: Config };
pub const Admitted = struct { geometry: Geometry, encoding: Encoding };
// Admitted.numerics() derives the numerical profile from encoding.
pub const Placement = struct { first: usize, end: usize, total: usize };
// Validate the half-open range and derive ownsEmbedding/ownsHead/local geometry.
pub fn admit(gpa: Allocator, bytes: []const u8) !Admitted;

// cuda_layers.zig. Existing MLX variants and default shared representation stay.
pub const Proj = union(enum) { a4: Affine4, an: AffineN, linear: cuda.qlinear.Weight };
pub const Dense = struct { gate: Proj, up: Proj, down: Proj };
pub const SharedExpert = union(enum) { stacked, projected: Dense };
// Mlp.moe adds shared: SharedExpert = .stacked.

// cuda_weights.zig and its private ModelOpt packing helper.
pub const Plan = struct { weights_bytes: usize, staging_bytes: usize };
pub fn plan(source: anytype, admitted: Admitted, placement: Placement) !Plan;
pub fn loadPlaced(e: *Engine, ck: *core.Checkpoint, stage: *DeviceBuffer,
                  admitted: Admitted, placement: Placement) !void;
```

Retain opaque norm/embedding GPU addresses in the existing ABI. The admitted
Numerics selects their dtype, scratch widths and primitive bindings together.
ModelOpt norms are FP32 effective multipliers; MLX norms retain their existing
representation. This avoids changing unrelated drafter/vision pointer fields.
The loader validates that all allocations match the chosen profile.

Use `core.Checkpoint`, existing CUDA packers, qlinear and grouped NVFP4 experts.
The ModelOpt helper loads into `Engine.alloc` and owns no second device arena,
stream, engine or layer container. Plan every aligned allocation exactly as
load does. Keep a projected shared expert separate from the routed NVFP4 table
to preserve its output rounding and mixed projection representation.

Keep geometry independent of checkpoint encoding through one dimensional
parser. Existing MLX quantization metadata can remain in its compatibility
view while admission adds ModelOpt metadata. A claimed malformed ModelOpt
checkpoint never falls through to MLX. The old separate module facade must
re-export the upstream checkpoint module rather than own duplicate source files.

## Numerical primitive contract

Keep the upstream Geo/Seg/table ABI and one layer loop. Bind ModelOpt variants
for FP32 effective norms, FP32 normalized GDN Q/K and recurrence arithmetic,
HF rotary rounding, serial attention and exact routing/combine arithmetic.
Carry each primitive's launch dimensions with its binding. Function-address
substitution alone is insufficient because block and chunk geometry differ.

HF GDN pre reads the BF16 tail through Seg.conv; existing conv_commit owns
mutation. Chain uses Seg.rec with a local delta index and writes FP32 y.
One-row attention can write slot zero and use a one-slot merge. Existing
attention gate is reused if parity permits. ModelOpt routing BF16-rounds
logits before stable top-eight selection and retains FP32 weights/shared gate.
Its combine uses ordered multiplication/addition. Preserve original notices
and state adapted code's source and changes. Compile HF glue with fmad=false.

Keep MLX's warm itinerary. ModelOpt warms only supported one-row operations.
Its first cached graph may capture record=false body calls with fresh tables
outside capture. Tree logs and rewind remain refused until their FP32-key
representation is implemented and proved.

## Verification and ownership

Data owner: geometry, checkpoint admission, layers, arena loader and fixtures.
Primitive owner: new HF CUDA source, symbols, launch/ABI contract and compile.
Engine owner: shared forward/routing, pipeline, native binding and new probe.
The coordinator reviews diffs, owns integration/build windows, runs actual
device trials and keeps the decision trail. File ownership prevents concurrent
writes. Commit green units and record any interface deviation before adapting.

GPU gates include per-device peak admission, the 1536 MiB floor, finite exact
primitive rows, eager/captured parity, alternating-session state isolation,
reset/poison behavior and a canonical restored Nemotron completion. Profiled
runs remain separate from performance measurements. Fleet full verification
is reserved for a candidate that passes prerequisite production gates.

## Continue tuning after shared-path acceptance

Preserve the earned cooperative router products and parallel stable selector
when adapting their ABI. The old up-only n16_pipe4 choice remains a candidate
until it passes a new shared-path comparison. A source port cannot inherit the
old 107.144 tok/s result. Rebuild a changing-token, changing-position baseline
and inspect actual graph functions and launch dimensions before comparing it.

The next experiments each change one mechanism:

| Mechanism | Cheapest useful experiment | Condition for a full model trial |
| --- | --- | --- |
| Expert CTA grouping | Compare the existing one-warp decode with four-warp baseline and n16_pipe4 on retained real expert banks. Keep up and down decisions separate. | Exact outputs and a repeatable saving whose call-count projection clears the turn-time gate. |
| GDN persistent layout | Transpose state at initialization and compare all logical state and output bytes through changing inputs over 64 rotating states. Keep ordered scalar arithmetic. | A projected saving above 5% of the new matched turn time. The rejected shared-staging experiment does not answer this layout question. |
| Dense projection input quantization | Audit actual shared input scales, then time reuse of one quantized activation across compatible projections. The a and b projections are BF16. | The affected time must justify integration. Do not concatenate projections with incompatible scaling or promote a sub-1% bound into a full trial. |
| Tensor parallel transport | Measure pinned-host round trips at the real residual payload and collective count before designing tensor splits. | A critical-path model, including collective latency and reduced kernel efficiency, predicts a material gain on this physical pair. |
| Remaining memory or issue bottleneck | Collect a bounded exact-symbol counter report only when it distinguishes the next two candidate changes. | Validate kernel, device and finite counter rows separately from report existence. Use an unprofiled run for the speed comparison. |

The historical 409 tok/s calculation is an uncached logical weight-stream
reference for the serial layer split. The ratio of observed throughput to
that reference is not measured bandwidth utilization. Repacked weights,
state, KV traffic, cache residency, launch cost and idle time need separate
accounting. Two GPUs' peak bandwidths cannot be added for serial stages.

Stop at diminishing returns only when the material mechanisms have measured
outcomes or falsified payoff bounds on the accepted path. Keep numerical
correctness, serving features, capacity and promotion as explicit gates.
Missing features or a failed reference comparison do not demonstrate a
hardware scaling limit.
