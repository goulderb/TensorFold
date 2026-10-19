# GLM target CUDA graph parity (2026-10-19)

Acceptance: **full production parity**, including complete target-forward CUDA
graph replay at exact aggregate widths 1..32. The user's 2026-10-19 correction
supersedes the assistant's earlier decision to defer that port behind a partial
FFN payoff experiment. Partial capture is neither the finish line nor a
substitute gate. Final production comparison follows full-path correctness and
qualification; a prefix timing threshold does not decide whether to do the port.

Status: the full-target source port has passed the **six-layer TP2 device replay
gate in all four required configurations**, on `task/glm53-v110-base`, based on
`10f6b355442e628b50db0c0078d03e4d269269ff`. Each configuration passed 100 exact-byte
fixtures per rank, with 48 startup captures and 1,022 replays per rank. This is
bounded device qualification, not complete-model state equivalence or performance.
The full FP4 model also completed all 48 startup captures on both ranks. After
repairing the test container's spill-directory UID mismatch, it served and passed
27 functional cases; cancellation was skipped by the probe budget, so the overall
functional gate remains false. The full/eager/production timing comparison did
not run. See the device receipts below; complete serving and performance remain
gates. Current production was restored and verified through the public proxy.
Final CPU tests, kernel/executable compilation and pinned formatting pass. Bounded
independent static review found no concrete reproducible defect in the reviewed
interfaces, geometry, lifetime and proof accounting; it did not rederive every
retained kernel's arithmetic. The user authorized the 2026-10-19 device outage.
The graph port, separate teardown repair and proof allocation repair are committed
for integration into `trunk` at the user's request. The retained device proofs
qualify their bounded scope. No serving pins changed. Complete-model validation
and production performance remain open; no promotion is claimed.

## Complete target boundary

`TF_GLM_TARGET_GRAPHS=full` selects complete target capture. Unset or `off`
retains the legacy eager target. `ffn` retains the partial predecessor for
research, not as the full-parity implementation. Mode parsing is strict, enters
the TP fingerprint and changes numerical identity; that identity is not evidence
of numerical qualification. Graph modes reject `TF_GLM_ROUTE_SAMPLES` other than
zero, rather than capturing diagnostic copies or diverging between ranks.

The full computation contains:

1. Token embedding and stream expansion.
2. Every target layer: mixer HC pre, segmented KDA or DSA, partial gather,
   mixer HC post, FFN HC pre, dense or routed/shared experts, FFN partial gather,
   HC post, and DFlash tap output where configured.
3. Final stream mean, RMS normalization and local vocabulary head.

Scheduling/admission, ID and metadata staging, grammar/sampling, acceptance,
partial KDA commit/replay, host positions/parities and DFlash context commit stay
outside capture. Prompt execution, vision feature injection during prompt
admission, and MTP/neural proposal execution remain separate. Target decoding
after a vision prompt passed the bounded full-model red/blue functional checks;
complete-model state equivalence remains separate.

The graph key is **persistent scratch kind (`bd`/`bs`) and exact total rows**.
There is one full-computation site, not a graph per layer, sequence address,
position, owner or segment geometry. A 64-slot bank bounds handles (two buffers
by 32 widths), not measured graph memory. Lone scratch admits at most 16 rows;
shared scratch admits aggregate widths 1..32 for the production profile.
Unsupported full-mode geometry fails; it does not silently run a partial graph
or fall back to eager on one rank.

A bank miss executes the real computation eagerly once, completes the stream,
then records and instantiates it **without replaying it that round**. Hits replay
once, without invoking the host compute callback. Sequence-dependent host state
is never retained by the executable. Capture/instantiate/launch failures fail the
pair; a unilateral eager fallback would break captured NCCL ordering. Engine
verification also treats later asynchronous completion failures as pair failures.
Executables die after stream completion and before their scratch, modules,
weights or communicator. Counters count bank-managed eager misses, captures and
replays, not requests or disabled-graph control passes.

## Stable device metadata and scratch

`cuda_target_meta.zig` and `glm_target_meta.cuh` share explicit extern layouts:

- Row: 8 bytes, segment and local row.
- KDA segment: 72 bytes, row span, convolution, input/output recurrence and
  projection/save-plane addresses.
- DSA row: 56 bytes, latent/index/gate/pooled addresses, absolute position,
  physical ring/pool sizes, owner, resident/sparse flags and merge chunk bound.

Engine-owned tables cover 32 rows and eight segments, with records per loaded
mixer layer. Host staging validates coverage, capacity, ownership and state
geometry, clears inactive entries and refreshes pointers/parity each round.
Owned asynchronous upload sources cannot race the next host table update.
Captured kernels receive stable table addresses and load current contents.
Release/rebind and KV-pool relocation therefore do not become graph keys.

The segmented KDA path scatters aggregate projected rows into each sequence's
projection plane, loads current recurrence/conv pointers from the table, and
preserves the retained native FP32 convolution and wide/fused recurrence paths.
Saved K/V/G/B rows remain sequence-local for the existing partial-keep replay.
Inactive segments return before dereferencing their addresses.

The DSA path extends retained native latent/index/pooling, exact radix selection,
BF16/FP8 and attention arithmetic. It uses capacity-strided score scratch and a
fixed five-chunk attention grid. The final table field, historically named
`pad`, carries the eager **segment-end merge count**, not unused padding:

```text
upto = sequence_position + segment_rows
merge_chunks = upto > dense_limit ? 5 : min(5, ceil(upto / 512))
```

All rows in a segment carry that bound, including earlier dense rows in a
segment containing sparse rows. Empty chunks are initialized, but merging
surplus inactive chunks can still change signed zero. The device merge honors
the staged bound to retain eager reduction extent without graphing host geometry.

The runtime owns full-32-row projection, attention-partial and score scratch.
This is necessary: legacy lone scratch has only a placeholder projection buffer,
and legacy shared attention scratch is sized for a 16-row segment, not the
whole aggregate. Runtime scratch and metadata outlive every captured executable.

### Fixed TP2 owner communication

For each sharded DSA layer, independent of current owners/partitions:

1. AllGather each rank's absorbed local-head queries: RH BF16 elements per rank.
2. Each cache rank attends both head-rank slices, predicating ownership before
   touching a possibly zero/nonresident latent pointer.
3. Pack only the peer head slice and AllGather RH BF16 result elements per rank.
4. Bit-select the local own-head result or the owner's gathered peer result.

No floating sum is used for owner selection. This preserves the owner's result
bits and matches production's result payload instead of sending both head
slices. It uses NCCL, not an imported RoCEnante transport; transport performance
is unqualified. Production's `seg_chunks.cu` HQ24 specialization is not its TP2
path and was not imported unchanged as TP2 support.

## Pre-admission production warmup

With both `TF_GLM_TARGET_GRAPHS=full` and `TF_GLM_DRAFT_PROFILE=production`,
`Engine.init` warms/captures all reachable shapes before returning:

- Shared aggregate rows 1..32.
- Lone rows 1..16.
- Exactly 48 new executable captures, checked before admissions.

Warmup uses at most two sequence slots, splitting shared rows at 16. Paired
32-token placements use the pool's existing 2048-row physical alignment; it does
not reserve an entire context for eight temporary streams. Target and drafter
state reset before each shape and after the last shape. Direct forward calls
avoid sampling and DFlash request-window staging/counters. After synchronization,
temporary slot 1 is freed and both pool reservations are released; clean slot 0
remains as the existing engine invariant requires. Captures reference persistent
tables, not those temporary allocations.

Startup logs capture count and elapsed nanoseconds; optional context free/total
bytes are settled observations, not continuous peaks or isolated graph memory.
Nonproduction full mode explicitly logs lazy first-use capture. It is not the
production-profile readiness behavior.

## Bounded device proof, executed on 2026-10-19

```text
glm-target-graph-check MODEL RANK WORLD IDFILE
```

Both ranks execute the same direct engine calls. Use a new NCCL ID file per run:
rank 0 writes it; the launcher must transfer the complete 256-byte ID before
rank 1 opens its local copy. A command or expired trial recipe is not permission
to start a model.

The full gate requires `TF_GLM_TARGET_GRAPHS=full`,
`TF_GLM_ROUTE_SAMPLES=0`, and explicit `TF_GLM_LAYERS=1..6`. Select six layers
for the trained-prefix gate so layer 5 exercises a real DFlash tap. The existing
pinned DFlash checkpoint/kernels and `TF_GLM_DRAFT_PROFILE=production` must be
supplied for the production-profile gate. Keep the model config intact; the
prefix setting limits loading. Engine proof capacity is 4096 to cross real
attention boundaries, not to qualify production capacity.

Coverage:

- Every shared aggregate width 1..32 with two compositions, shuffled slots,
  changed IDs/positions/parities and release/rebind between passes.
- Lone 16-row windows at every keep 1..16, plus production shared 16+16 windows
  at every keep length.
- Mixed short/dense and 2060+-token/sparse sequences with changed slot/owner
  composition; dense windows crossing the 512-key chunk boundary.
- Legacy eager control, enabled graph pass, then mandatory replay without a new
  capture. Startup-prewarmed production fixtures can replay on both enabled
  passes. Full mode requires one complete-target replay; predecessor FFN mode
  requires one site replay per loaded layer.

Witnesses compare all active local-vocabulary logits, complete tap storage,
drawn tokens, sequence arena bytes, committed snapshot ranges, positions,
parities and owners, then a following target decode and committed state. BF16
logits/taps must be finite even if both modes would produce equal NaN bits.
Non-owner ranks omit only nonresident latent planes using the engine's own
snapshot-residency rule, retaining fixed/raw/pooled ranges.

Pool/shard coverage requires a separate TP2 launch with
`TF_GLM_KV_POOL_TOKENS=16384` and `TF_GLM_KV_SHARD=1`; pool-off does not qualify
it. The proof does not establish DFlash proposal, spill, vision or full-model
state equivalence.

Each fixture logs verify-plus-stream-completion time, capture/replay deltas and
settled context free/total bytes. The timing block discards one warm pair and
emits five alternating eager/graph pairs at aggregate 32, excluding preparation
and witness downloads. Measured graph blocks must replay without capture.
Timing includes sampling/communication within `verifyShared` and synchronization,
not just kernels. Prefix/rank timings cannot establish a full-model speedup.

## Device replay receipts

Raw timestamped logs, original failure receipts and immutable build/staging inputs
are retained in `C:/Development/prep/glm53-fullgraph-20261019/`.
`proofs-qualified.json` rechecks the retained logs rather than trusting pass markers.
All four runs used the intact checkpoint configs and six loaded target layers:

- FP4, unpooled, wide KDA: `edd8dc71e61e`.
- FP4, 16,384-token pooled/sharded KV, wide KDA: `a6c47cb34c3c`.
- FP4, unpooled, fused KDA (`wide=0`): `6556e7a2f097`.
- EXL3, 16,384-token pooled/sharded KV, wide KDA, `mia-mcg4`: `24825942b414`.

Both ranks exited zero in every run. All 100 geometry fixtures per rank passed,
including every shared width, shuffled/rebound slots, every keep length and the
mixed sparse/dense/512-key boundary cases. The final counters on every rank were
`eager=48 captures=48 replays=1022`. Startup capture took 456.5–508.4 ms for the
six-layer prefix. Alternating R32 timings remain prefix verify-plus-sync evidence,
not a whole-model speedup.

The initial run `3d0191b3acdd` exceeded the harness's 600-second limit while still
progressing; it is **not** a complete pass. Its witness ArenaAllocator retained
all ArrayList growth allocations and all three witness images. The repaired
runner keeps the reference and one reusable working ArrayList, retaining every
byte/finite check and all fixtures. No engine, CUDA kernel or serving input
changed. All four modes were rerun from scratch after that repair.

The host-only runner rebuild `v110-20261019-133827-cd57fd` passed 8/8 steps, exit 0,
not OOM; pinned formatting `v110-20261019-134022-83cbed` passed. Source archive SHA256
`3264bc09ba881dedee3669e4c2439165ac5f35f1fff59daf1dc8939cd1b5b8a7`;
log SHA256 `9928c2786c7d8d3d9a73aa4ba4deaee1e46c26373b20f38865fbc6aa4196b389`.
The repaired `tf-cuda-test` is 69,337,528 bytes, SHA256
`e3c322785a9f83ffb9ca6658448319e4766f2a3b5cd16630137ad0f1e753b6ca`.
Only `zig/tests/cuda/glm_target_graph_check.zig` differs from the original 18
code-input pins; the other 17 and the serving executable remain unchanged.

Full-model run `148fbe22c679` completed all 48 startup captures on each rank in
about 4.252 seconds. Settled CUDA free memory was 9,249,386,496 / 9,166,594,048 bytes
(rank 0 / rank 1), not under-load peaks. It then failed `IdentityCacheUnsafe` before
HTTP readiness: the image defaults to root but private `/spill` belongs to UID
1000 with mode 0700. The safety check was not weakened. This failed serving gate
must not be reported as functional qualification or a performance result.

The fleet task lifecycle then added a validated optional numeric `runtime.user`
and actual-container-user verification; no engine/CUDA input changed. Retry
`426313455dc3` used `1000:1000` and enabled private spill without bypassing safety
checks. It captured 48 shapes per rank in about 4.269 seconds, then served with
262,144 context, eight lanes, a 2,097,152-token sharded KV pool and a requested
2 GiB prompt cache. The reported cold load, including spill-identity bootstrap,
was 544.2 seconds. Rank 0 logged 44 full-target replays at teardown and no new
captures beyond startup.

All 27 executed functional cases passed: text, reasoning, tools/abstention, JSON,
SSE, red/blue vision and one/eight-client distinct/shared-prefix checks. Eight
requests reported 4,096 restored prompt tokens. The final cancellation case was
`skipped_budget`: the report is **27/28, all_passed=false**, not a complete gate.
The last probe budget was 32 seconds; an already-started concurrency case
completed beyond it, giving 63.4 seconds total. Spill initialization passed, but
these prompts do not qualify spill eviction/readback/restart or full occupancy.

No full/eager/pinned-production timing battery or mandatory same-source output
comparison ran. The original work deadline was not extended. Exact-owned
candidates/tunnels were removed, the original task recipe restored byte-for-byte,
and current production's pinned image restored. A local-only public proxy
completion passed at 18:47:15 UTC: HTTP 200, canonical `glm-5.3-flash`, exact
`READY`, parallel 8, no paid fallback or attribution leakage. Postflight verified
the production unit active and both recovery timer/service inactive/unloaded.
The original controller remains a nonzero historical result (initial timeout;
its final stop also complained that the never-loaded recovery service was absent).
That exit does not override the independently verified public recovery.

Fleet evidence: `fleet/trials/glm53-full-target-graphs-20261019.json` and
`docs/plans/glm53-native/full-target-graphs-20261019.md` in the local-ai-server task
worktree. `fleet.verify fast` and all 46 lifecycle/probe module tests passed;
seven UID regression cases are included in that module total. No commit, new
trunk merge, publication or promotion occurred.

## Remaining gates

- CPU tests, sm_121 kernel compilation, native server/GPU runner compilation,
  pinned formatting and exact code-input matching passed; see receipts below.
  Bounded independent static review is complete, with no concrete source defect
  found in its reviewed scope. The four required TP2 prefix device modes also
  passed; later engine changes require the applicable proofs again.
- Complete full-model serving and same-source full/eager output qualification.
  Stop on numerical mismatch; do not weaken byte checks silently.
- Measure capture/readiness cost, warmed timings and memory/headroom. A
  noise-sized speed result does not replace the full-parity scope or authorize
  an extended outage.
- Qualify the complete model and its capacity, cache/spill, vision, sustained
  occupancy and public serving path; compare against pinned current production.
  Windows and the broader shared-base GPU changes remain separately unqualified.
- Promotion is a separate decision. No device trial starts without fresh outage
  approval. Complete the approved scope, keep control detached, restore
  production and verify a local-only public proxy completion afterward.

## Donor provenance

Pinned production image:
`sha256:d7e14f8799ad21f7e93ba95ea78826426c98ab27d5fbfdf81bed876a2e09ce6e`.
The retained source receipt under
`C:/Development/prep/glm53-expert-tuning/production-source-receipt.json` was
rehash-checked for all 12 graph/draft reference files. Important graph references
under `families/glm5_next/cuda/`:

- `verify.py`: `e0d450b84fe6b4e285ae0bb5942e92aa86762a554e29aae8e38d53c447b1e13f`.
- `segments.py`: `e0f9026ad97c7d766b9b59e4353ea1bf140595f67bdc749c04f6a550a8c577bd`.
- `forward.py`: `6be4e1ec75b0484f81aa2e64d83f729c26483c62c5a69a387770c0466257c4df`.
- `kda.py`: `65d522d5d0f9c70d77445d1485ebd4b876529d46001b2b857997b03eef8e2af4`.
- `kda.cu`: `f74056a38e6f18a5d1995cf994e0f3e0231b24a530da827a350f221aedfbdafc`.
- `latent.py`: `73edd5297525ee5fc457c04e4e622e0413b17666c194427d9ddb993e0ce000e6`.
- `sparse.py`: `80b8de2c2aaccc3c128fdaf65d83b25f66973bb76b58693f2d95b399b723d1cd`.
- `shard.py`: `204577f02fb60dd8a9fb724ed5818dc256415c8c79aa4091daa44ada0386fdd1`.

The port reuses upstream `cuda.graph`, adapts production segmented KDA
orchestration and extends retained native DSA arithmetic/metadata. It does not
import RoCEnante or relabel a TP3 attention specialization as TP2. Retained
TensorFold, Mia, wojo and transitive notices remain applicable. Byte-exact
production-lineage legal bundles and their manifest hashes are under
`LICENSES/donors/mia-b5808ac6/` and `LICENSES/donors/wojo-7fac72b3/`.
See `NOTICE`, `THIRD_PARTY_NOTICES.md` and `LICENSES/donors/README.md` for the
source/change distinction. No publication or relicensing is authorized.

## Full-port CPU/build receipts

All successful runs used pinned Zig 0.17.0 in bounded, networkless CPU-only
Spark 2 containers (`--runtime=runc`, `NVIDIA_VISIBLE_DEVICES=void`, two CPUs).
No kernel or model was executed. Full logs, source archives and private receipts
are retained under `C:/Development/prep/glm53-v110-merge/`.

- Final kernels: `v110-20261019-113912-f08870`, sm_121,
  **184/184 steps succeeded**, exit 0, not OOM. Archive SHA256
  `b8b9da8cce2a59f2fdea46627934686da177157be166056cc7ad419d44326c9e`;
  log SHA256
  `467e178ba6fdf6dd12f2e915062e8e8a35b54d90109ed3b5ee06adb712cd7838`.
- Final host/server CPU tests: `v110-20261019-114728-a1c95c`,
  **81/81 steps succeeded; 946 tests passed, 13 skipped** among the reported
  non-cached tests (plus cached steps, not a fresh aggregate suite count).
  Archive SHA256
  `000266d7664701c161cb07bd1fe0ac8a6f20178092a3a96064c201d8f0711adc`;
  log SHA256
  `52ac1d69d88466127461b30f7b5611805f266391f4b533049a0bf6578fc70fc8`.
- Final formatting: `v110-20261019-114728-102c6f`, all 15 changed/new Zig files,
  exit 0. Its input archive matches the CPU-test archive. Empty log SHA256
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
- Final executables: `v110-20261019-115136-7f5121`,
  `compile-config-server compile-cuda-tests -Dfatbins=/fatbins`,
  **12/12 steps succeeded**, exit 0, not OOM. New kernel images came from the
  final kernel run, mounted read-only. Archive SHA256
  `0bafd289cf22dd20a2d660c5133a8eea1b91fdd4abec9690705f8d2433e4b453`;
  log SHA256
  `f53e197d12ee608322d0323055e4843a35b01e19276c64dc6ee6b006d013fdca`.

The first executable attempt (`v110-20261019-114728-58a6a2`) failed: the server
compiled, but the runner compiler was killed at the 6 GiB container limit
(`OOMKilled=true`, exit 1). The successful retry used a fixed 8 GiB cap,
unchanged two-CPU/single-job limits, no swap expansion, and a host-memory guard
requiring a further 4 GiB available before launch. It did not stop production or
use a GPU. This resource failure is not counted as a passing gate.

`full-target-source-inputs-20261019-115136.json` hashes all 18 changed/new
Zig/CUDA/header inputs and compares all 4,992 regular source files in the
snapshots with the task worktree. CPU, formatter and executable snapshots match
all code inputs; only this expanded document differs. The kernel snapshot
predates the final Zig metadata edit, but all CUDA/header/catalog inputs match
exactly. No final-kernel claim relies on an older DSA merge image.

`full-target-fatbins-v110-20261019-113912-f08870.json` pins all 91 images:

- `glm_kda.fatbin`: 151,648 bytes,
  `4398f42e236ba4fb68d2e1b27457aede5b019f8a83e2ee9d02cbe94fe15640ce`.
- `glm_target_dsa.fatbin`: 628,624 bytes,
  `11b2d8edc2eb13aab47a949190bcd408c9382ddba9274437fe09e97ab1a4f56a`.

Private completed-build copies under
`graph-artifacts-v110-20261019-115136-7f5121/` were checked for full-target
engine/proof markers and rehashed after transfer:

- `tf-cuda-test`: 69,286,984 bytes,
  `0d232fdeeeba7c5e2ebb0a37cea7dc6a75e579737425e6a7a53bbe3b9fc06ae7`.
- `tensorfold-native`: 72,441,912 bytes,
  `196cea2a59f9a4455213e9d13efa4f9b24db8b710a04df7df2fd0f3485d0a31b`.

The directory's `receipt.json` records build exit/OOM state, source/log hashes,
cache paths, binary hashes and markers. These are the original pre-device pins;
the later host-witness-only runner rebuild and actual model runs are recorded
above. Neither executable was promoted. CPU/build success alone does not qualify
CUDA/NCCL graph replay, numerical equivalence, memory/headroom or production
performance.

## Historical partial-predecessor receipts

These are **not full-port binaries or validation**. Full logs, unique archives
and receipts remain under `C:/Development/prep/glm53-v110-merge/`.

- Partial CPU gate `v110-20261019-104102-2bf610`: 81/81 steps, 342/342 reported
  tests plus cached steps; archive
  `b0cc8e5795c05580507e8e078faec49536af2e50924f1280d4fd87b1da4af89f`, log
  `55fec75a0710d93f0ccea0c76d0015fbfab91c41757b0d0a962ed85123f6423b`.
- Partial executable build `v110-20261019-103729-5460a3`: 12/12 steps; archive
  `30e890d0499df4dd25964e93a7c1e15ab0dff4735f67afebbe5e34400efe3f7a`, log
  `f53e197d12ee608322d0323055e4843a35b01e19276c64dc6ee6b006d013fdca`.
- Partial pinned format check `v110-20261019-103729-cceea6`: nine Zig files,
  exit 0. Historical source receipt: `graph-source-inputs-20261019-104102.json`.
- Private partial `tf-cuda-test`: 67,976,688 bytes,
  `f0b12609c68b5a90058a6958cf1db90a09d4f9625814807c98a8ea918b827be4`.
- Private partial `tensorfold-native`: 71,130,744 bytes,
  `fde5d7b6eb9dd27202631a12c3ba515e8eea302282fe30c4832402a2b98e8a66`.

The separately found failed-context-bind teardown regression failed before the
repair (`v110-20261019-101512-9ad62c`: expected one synchronization, observed two)
and passed after (`v110-20261019-101658-62a6de`: 7/7 steps, 487 passed, one
skipped). A failed bind now leaves that context's retain/spares intact for retry
rather than synchronizing/releasing resources while another context is current.
Negative-control diagnostic `FAIL` lines in aggregate tests are expected; real
process exit, OOM state and build summary determine the gate outcome.

Historical instrumented prose traces had about a 2.5% native non-kernel
complement, but those unmatched periods included following draft proposal work.
They do not bound collective/kernel/scheduling effects or explain a whole-turn
gap. The assistant's former 5% partial-benefit screen is not an acceptance gate.

## Completed full-model comparison, 2026-10-19

The graph and teardown changes are integrated at `b5fac6aee`. The continuation
kept all 18 code-input hashes and the retained native/proof executables unchanged.
It exercised both full target graphs and HTTP prompt batching with production's
seven neural drafts, fifteen copy drafts and aggregate 32-row verification budget.
Native's admitted geometry was eight 262,144-token contexts, a 2,113,536-token
live-only pool, and a separate 12.5 GiB saved-state budget.

The combined configuration passed 28 functional cases. Eight requests each
completed 253,840 input plus 8,192 output tokens, with at least 1,012.5 seconds
of actual eight-stream overlap. A 1,806.6-second soak completed 1,528 requests
with zero serving errors. Distinct saved states and an evicted reference restored
correctly within the running process. These checks do not imply restart parity.

Five measured groups after a discarded warmup, through the private HTTP path,
gave full/eager median c8 rates of 154.99/144.48 tokens/s for code cap128,
97.38/88.81 for prose cap128 and 93.05/84.91 for prose cap512. The measured
7.3–9.6% graph benefit has disjoint c8 ranges. All 130 measured native response
pairs match exactly. Single-client differences were within variation; no material
prefill benefit was measured. Loads were sequential rather than alternating
process pairs, and rates include admission/prefill and count actual output tokens.

The pinned production EXL3 comparator reached 217.45 and 126.47 tokens/s on
the same fixed cap128 c8 code/prose work. Native full remains 28.7% and 23.0%
behind. Native FP4 and production EXL3 are different checkpoints: this is a
serving comparison, not an isolated weight-format experiment. The production
benchmark disabled snapshot retention with a SHA-pinned, CPU-tested module-load
overlay while retaining its 12.5 GiB pool-sizing budget and full 256K context;
all timing responses had zero cached tokens. An earlier reduced-pool control is
retained separately and excluded from the primary comparison.

First near-full disk restores after both clean restart and active-request SIGKILL
returned wrong marker answers with `TF_GLM_PREFILL_ATTN_LOCAL=1`. A fresh cold
prefill masked the failure, so only first restores before that prefill count.
Changing only the flag to `0`, with the same binary and saved files, restored
both failing near-full fixtures correctly at 260,096 cached tokens, followed by
three matching replays. This supports the missing-prefix hypothesis in the
non-owner per-layer attention replicas; no replica repair is claimed here.

The candidate is rejected for production on measured performance and the enabled
local-attention path's restart failure. Supervised native public-path and reboot
promotion gates were not run after failed acceptance. The complete fleet record
is `docs/plans/glm53-native/full-model-parity-20261019.md` and
`fleet/trials/glm53-full-model-parity-20261019.json` in local-ai-server, with
raw receipts retained under
`C:/Development/prep/glm53-fullgraph-complete-20261019/`.
