# GLM expert routing diagnostic

Set `TF_GLM_ROUTE_SAMPLES=32` on both ranks of a private GLM native run to
snapshot the first 32 shared target decode forwards with at least eight rows.
The maximum is 32; unset or zero disables the probe. Prompts, ordinary serial
verification, the MTP head, and the DFlash drafter do not consume samples.

Rank logs contain `GLM_ROUTE` JSON records, one per target MoE layer followed
by a `complete` record for each sampled forward. Layer records include rank,
sample, target layer, rows, top-k, exact per-expert members, routed item count,
active experts, routed pairs, shared pairs, and hypothetical item counts at
32 and 64 pairs per item. A sample without its completion marker is incomplete.
Never combine ranks as independent tokens or use instrumented latency as a
performance result.

The engine owns the pinned buffer. `verifyShared` selects a bounded forward
and supplies an optional `Trace` to `Forward`. The whole target loop calls
`capture` after the MoE FFN while the plan is still intact. Each layer's count
and bounded item array are copied asynchronously on the forward stream before
the next layer reuses that scratch. `finish` synchronizes once, validates all
snapshots before emitting records, and prints a completion marker. Error paths
drain pending copies before the capture can be reused. Engine teardown also
synchronizes before freeing the pinned allocation.

The copied plan contains `(expert, first, count)` triples. `counts[0]` is the
actual item count; the host capacity bound is not an observation. `counts[1]`
includes the shared expert. Every shared item is excluded from routed totals.
The reducer checks ordered IDs, contiguous member offsets, item sizes, exact
routed and shared pair sums, and the distinct expert count. The re-read model
is `routed_items / active_experts`; it measures logical 16-pair items per
expert, not physical DRAM traffic. Compare `tile32_items` and `tile64_items`
only as weight-load models before testing a larger kernel tile.

## Design decision

Two independent sketches and a read-only cross-judge compared asynchronous
host snapshots, device snapshots followed by a bulk flush, and a new device
histogram kernel. The host-snapshot sketch was selected for its smaller API
and use of the existing plan. Engine-owned bounded shared-forward selection
and completion markers were taken from the second sketch. Review identified
the need to size from the active shared buffer (up to 56 verifier rows), to
drain copies on errors, and to exclude all shared items. A new CUDA reduction
symbol and device allocation add no accuracy for this finite diagnostic.

`cuda_route_trace.zig` owns buffer layout, selection, copies, validation, and
reporting. The hook does not change expert arithmetic, router kernels, their
ABI, precision, checkpoint, or serving defaults. When disabled it allocates,
copies, and synchronizes nothing; a nullable branch remains in the target loop.

## Verification

`weight_checks.py regression` includes the trace tests. The tests reconstruct
a 17-member expert split across two items, another 14-member expert, and a
31-member shared expert; reject invalid counts, IDs, offsets, and pair sums;
and check sampling bounds. Function analysis also covers the copy/report path.
Device acceptance additionally requires both rank logs, all completed samples'
pair-sum checks, and identical generated outputs with sampling off and on.

Source inspections and sketch artifacts are under
`C:/Development/prep/glm53-expert-tuning/design-{a,b}`. The parent selected the
shape; review agreement is not a substitute for runtime validation.

## October 18 device result

The bounded probe completed 32 shared forwards on each Spark rank, each with
42 MoE layers. Both ranks passed every exact membership check, and all 16
greedy code/prose outputs matched between tracing and the exhausted quota.
These are 64 rank observations of 32 shared forwards, not 64 independent
forwards. Across the sampled routed groups, 81.1% had at most four members,
and logical 16-pair items per active expert averaged 1.0155. A 32-pair tile
would save only 1.38% of logical routed items on this sample.

The expert microbenchmark therefore includes 30- and 56-row sparse cases
spread across 64 or 128 experts, as well as dense shared-group cases through
56 rows. Weight banks rotate beyond L2. At 30 rows / 128 experts, a
warp-distributed scale-conversion prototype matched baseline arithmetic but
did not improve the seven-round timings: gate/up 2643.286 to 2643.177 us;
down-FP32 1367.700 to 1377.947 us. The prototype was reverted; its source and
negative result remain in commit `a77a3ee1c` and the fleet trial
`glm53-expert-routing-mtu-20261018`. Default kernel selection is unchanged.

Unique-weight bytes divided by time is an effective traffic rate, not a
hardware bandwidth measurement. A separate Nsight Compute run reported
21.77% SM throughput and a 2675.616 us gate/up launch; the requested DRAM
metrics were absent on this device. This does not prove a DRAM ceiling or
establish how much further kernel headroom remains.
