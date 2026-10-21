# Clef BF16 on the shared Qwen backbone

## Problem

Clef Flash uses Qwen 3.5 hidden states and a joint schema head. It is not a
language-model answer-token scorer. Its plain BF16 checkpoint must not enter
the guarded ModelOpt MoE loader or the MLX affine loader.

Checkpoint admission, validated planning, mapped payload ownership, shared CUDA
weight loading and a request-owned hidden terminal are implemented. An explicit
development-only BF16 initializer now connects them to the shared backbone.
It does not enable a native serving route. Numerical backbone qualification
and the joint head remain separate units.

## Usage

Metadata-only caller, with no CUDA context or weight payload mapping. The shared
Qwen CUDA module exports `checkpoint` and `bf16`.

```zig
const admitted = try checkpoint.admit(gpa, config_bytes);
try std.testing.expect(admitted.encoding == .plain_bf16);
var plan = try bf16.Plan.inspect(gpa, io, model_dir);
defer plan.deinit();
try std.testing.expectEqual(@as(usize, 427), plan.tensorCount());
try std.testing.expectEqual(@as(usize, 4_068_474_880), plan.bytes().host_tables);
```

Existing engine callers retain their existing admission and weight-planning
operations. Ordinary plain BF16 execution remains refused by
`requireExecutable()`, ordinary weight planning/loading and
`initPlacedWithOptions()`. Config admission is not permission to execute.
Only the explicit development constructor described below accepts this profile.

The shared forward path retains every token's final normalized hidden state.
A later Clef adapter will run the official joint head once over the whole encoded
record. Input embedding rows and output lexical rows remain separate, untied
host tables. No full vocabulary projection is needed.

## Shape

Storage encoding and numerical execution have distinct named identities.
Admission chooses valid pairs. Callers cannot request arbitrary arithmetic
flags. The first plain profile is the pinned dense 32-layer Clef geometry,
BF16 checkpoint tensors, HF-order arithmetic and FP32 recurrent state.

The BF16 checkpoint module owns its tensor-role schema and an immutable plan.
One role walk derives required names, shapes, conversion and allocation. The
payload loader consumes those bindings, rather than maintaining another list.
The public inspector hides shard names, header parsing, validation and checked
byte arithmetic. Existing MLX and ModelOpt walkers are not redesigned.

Planning keeps device weights, load stage, host vocabulary and conversion
scratch separate. Logical mapped host bytes are not a resident-memory claim.
Cache, request scratch, all-token output and joint-head weights are additional
budgets. A metadata plan does not prove eGPU fit or numerical parity.

## Admission boundary

- Explicit quantization declarations are classified before the plain path.
  Unknown, conflicting, null or malformed declarations fail closed.
- Existing ModelOpt geometry and quantization guards remain unchanged.
- Plain config admission checks explicit BF16, dense geometry, layer pattern,
  rotary layout, untied tables, supported state dtype and no MTP.
- Tensor admission checks every required dtype, shape and byte range. It
  rejects missing or extra text tensors and duplicate names. Vision tensors
  are explicitly excluded from text execution, not silently treated as text.
- Header I/O is bounded and reads no payload. Shard paths, index ownership,
  overlaps, truncation and all size arithmetic are checked.

## Synthesis decision

The parent and independent Sonnet cross-judge selected the Opus manifest
design over the Sol variant because one inspect operation hides intermediate
catalog coordination.
Both independent candidates preferred a validated manifest over a byte-only
encoding-specific walker. Sol's separate BF16 checkpoint module keeps the
shared checkpoint file from accumulating another full loader schema.

The cross-judge scored Opus 20/20 and Sol 19/20; the parent scored them 19/20
and 17/20 using the same five criteria. Both preferred the smaller caller API.
The synthesis also adopts Sol's separate host-conversion and row-staging
budgets, and its rule against blanket HF predicate rewrites. A third GLM
candidate produced no output and was cancelled to bound design latency.
No conclusions are attributed to it.

The synthesis rejects a universal loader rewrite, public tensor-role plumbing,
and freely combinable numerical flags. Keep future loader and hidden-output
signatures in this design record rather than adding unimplemented production
entry points.

## Tradeoffs and alternatives

We accept owned metadata in exchange for testing the actual loading contract
without GPU access. We accept a narrow initial geometry instead of admitting
untested BF16 variants. We accept separate plain and legacy inspection paths
instead of changing existing production loader behavior.

A byte-only planning walker is smaller initially, but loses validated tensor
bindings and host-table ownership. A duplicate Clef backbone loses shared Qwen
fixes. Per-answer token scoring does not compute the official joint head.

## Metadata verification

The Windows header-only probe reads the real staged checkpoint through the same
`Plan.inspect()` used by tests. It reads bounded prefixes and headers, not tensor
payloads. The pinned release validates 427 text tensors and explicitly excludes
333 visual tensors.

The current checked plan reports 13,879,009,280 device-weight bytes, a
100,663,296-byte device load stage, and two distinct 2,034,237,440-byte host
vocabulary tables. Conservative host conversion peak is 201,326,592 bytes.
Two 4096-row host gathers budget another 67,108,864 bytes. These are logical
allocation budgets, not measured residency or device-fit evidence.

Each planned shard retains its file size and original-header SHA256. The
payload loader verifies those identities on the same mappings that provide
its borrowed tensor bytes. All shards are checked before the first upload
callback. Full weight hashes and file immutability remain caller requirements.

```sh
zig run -lc --dep core -Mroot=zig/src/families/qwen3_5/cuda_bf16_probe.zig -Mcore=zig/src/core/root.zig -- /path/to/clef-flash
```

## Payload ownership and verification

`Plan.upload(gpa, io, dir, target)` visits each validated binding once. The
synchronous target consumes non-host BF16 tensor bytes before returning.
`cuda_bf16_weights.load()` converts parameters through the existing
`checkpoint.floats()` helper and packs projections through the existing
`modelopt.Packer.bf16Lane()`. It fills shared `Layer` values and returns device
weights, final norm and owned host tables. It drains failed device operations
before freeing the stage or arena. The ordinary engine still refuses this
profile, so this loader is preparation rather than a serving entry point.

`HostTables` outlives its plan. Its separate embedding and output gathers
validate every token ID and exact output length before writing. Each gather
is bounded to 4096 rows. Whole retained shard mappings can be larger than the
logical tables; `mappedBytes()` reports that separately.

The real-checkpoint CPU probe counted 425 non-host ranges and 13,839,131,648
logical bytes without dereferencing their payloads. After destroying the plan,
it gathered one 8192-byte row from each host table. Retained shard spans
totalled 4,942,706,120 mapped bytes, against 4,068,474,880 logical table bytes. Neither number measures resident host memory.

```sh
zig run -lc --dep core -Mroot=zig/src/families/qwen3_5/cuda_bf16_payload_probe.zig -Mcore=zig/src/core/root.zig -- /path/to/clef-flash
```

The preparation host target passed 137 tests with another subrun cached.
The direct complete weight-loader root passed 32 ReleaseFast tests. These
roots overlap. Tests cover exact callback ranges, host lifetime, gather
refusals, header changes, allocation/callback cleanup and parameter conversion.
Independent admission, metadata and payload reviews found no defects. A
review note led to preserving fixture bytes during the Windows unmapping test;
the 28-test payload ReleaseFast root passed again after that test-only fix.

A private SM89 probe loaded the HF, packer and lane bindings on the decision
eGPU. GPU packing matched all 16,384 CPU reference words, including padding.
Analytically exact 32-by-4096 projections passed for 1, 17, 65 and 256 rows,
covering noncluster split-K reduction and the fused path. This is primitive
kernel evidence, not a full checkpoint load, Qwen forward or Clef probability
comparison. No new CUDA arithmetic was needed for these checks.

## Request-owned hidden output

```zig
var output = try e.hiddenRows(total_tokens);
defer output.deinit();

const window = try output.window(offset, ids.len);
try e.forward(&.{.{ .cache = &cache, .rows = ids.len }}, ids, .{ .hidden = window });
const record = try output.completed();
const device_bf16_rows = try record.device();
```

`Rows` owns up to 4096 BF16 rows. Keep the owner, engine and bound cache alive
while using its windows or completed record. Queue consumers on the engine
stream, or complete them before resetting or destroying the owner. Windows
append in order and check width, bounds, device, cache ownership and reset epoch. Partial, failed,
cancelled and stale records cannot be obtained through `completed()`. A failed
accepted forward poisons the bound cache. `reset()` drains work, resets that
cache and invalidates old views. Preflight refusals leave cache state unchanged.

The terminal supports a complete single-engine text placement with BF16 KV.
MLX chunks retain their existing 128-row limit. HF ModelOpt retains one-row
execution, or up to 16 rows with bounded HF prefill enabled. The explicit dense
BF16 constructor supports up to 16 rows per chunk. Images, parents, taps,
recorded verification, terminal-graph verification and pipeline preparation
are refused for hidden output.

The shared body applies its final RMSNorm once and skips the vocabulary head.
Hidden mode has a separate graph-key bit. The current request destination is
copied outside graph capture and synchronized before cache commit and scratch
reuse. Recording-driver tests cover control flow and pointer extents, not
numerical backbone parity. `TerminalWeights` distinguishes absent, vocabulary
and hidden-only terminals. Final normalization is independent of that union.
Existing loaders, workspace checks and pipeline callers use the same source
of terminal weights. The hidden-only engine allocates no vocabulary logits.

Independent review found that teardown needed an unconditional drain for later
consumers of completed rows. The owner now proves completion before freeing,
including after cancellation. Internal transaction helpers also reject stale
generations, incompatible states and append offsets. New regressions cover
those cases and context fallback after a failed stream wait. A fresh scoped
verification review confirmed the fixes and their regression tests.

The parent reran the shared host target with a fresh cache after those fixes.
All 152 tests and ten build steps passed. The direct hidden-root ReleaseFast
run passed 84 tests. These roots overlap. Neither run executed a numerical
backbone on CUDA.

## Weight-only Ada proof

A private candidate loaded all 32 backbone layers into the planned
13,879,009,280-byte arena on the pinned decision eGPU. Device readback matched
all 4096 final-norm FP32 words and all 524,288 packed words of layer zero's
GDN A projection against the checkpoint and existing CPU conversion/packing
helpers. Both host tables remained available through retained mappings. The
production verifier checked the complete pinned checkpoint hashes before the
trial. No backbone forward or joint-head computation ran. Weight-only success
does not qualify full-engine residency, 4096-token capacity or output parity.

## Explicit hidden-only initialization

The development entry is
`Engine.initBf16(gpa, io, ctx, model_dir) !*Engine`. The caller verifies the
pinned checkpoint payload hashes before loading and keeps the current CUDA
context alive through teardown. Valid requests contain 1 through 4096 tokens.
The engine uses dense `.hf_dense_bf16` numerics, BF16 KV and `value_key` GDN.
It accepts one sequence per forward and chunks of at most 16 rows.

```zig
const e = try qwen.engine.Engine.initBf16(gpa, io, ctx, model_dir);
defer e.deinit();
var cache = try qwen.engine.Cache.init(gpa, e, ids.len);
defer cache.deinit(gpa);
var output = try e.hiddenRows(ids.len);
defer output.deinit();

var at: usize = 0;
while (at < ids.len) {
    const n = @min(qwen.engine.hf_prefill_rows, ids.len - at);
    try e.forward(
        &.{.{ .cache = &cache, .rows = n }},
        ids[at..][0..n],
        .{ .hidden = try output.window(at, n) },
    );
    at += n;
}
const device_rows = try (try output.completed()).device();
```

Host embedding gathers and uploads run outside graph capture through persistent
pinned staging. The shared body skips device embedding for this profile.
`gatherOutputRows(ids, out)` separately copies untied lexical rows into exact-sized
caller-owned host bytes. Neither operation uploads a full vocabulary table.
Generation, logits, pipeline execution, taps, media, parents, verification and
transposed GDN remain unsupported for this constructor and fail during preflight.

`Engine.bf16Budget(plan, tokens)` reports checked logical allocations. A CPU-only
run against the pinned staged checkpoint produced these values for 4096 tokens.

| Allocation | Bytes |
| --- | ---: |
| Device weights | 13,879,009,280 |
| Temporary load stage | 100,663,296 |
| Scratch and aligned device metadata | 28,567,552 |
| Request cache | 187,403,264 |
| Request-owned hidden rows | 33,554,432 |
| Steady device budget | 14,128,534,528 |
| Persistent pinned input staging | 131,072 |
| Logical host tables | 4,068,474,880 |
| Conservative host conversion peak | 201,326,592 |

Driver/module overhead and resident mapped pages are not included. The computed
load-device peak is the larger of the steady budget and weights plus temporary
stage. These budgets do not prove full-engine device fit or occupied capacity.

The runtime binding owner requires four real images: `qwen35`, `qwen35_hf`,
`qmmf` and `modelopt_pack`. Ordinary convolution commit and attention gating bind
from `qwen35` before the HF overrides. Offline export checks and kernel-enabled
constructor/forward compilation passed. The shared qlinear compile closure also
imports an empty EXL3 image in the private compile setup. That image is not a
runnable backend. The BF16 projection entry rejects other weight encodings before
backend access and carries no EXL3 binding. No shared kernel arithmetic changed.
Numerical execution remains a separate device gate.

Independent source/CPU review found no concrete initializer defect. An isolated
cache run passed 158 CUDA-host tests and 11 native-host tests, plus both compile
targets. These counts overlap the earlier preparation roots. A parent rerun
passed all 19 steps with one executed test and cached subruns. Private rebuild
receipts bind before/after repository source fingerprints and the listed build
inputs, not a complete system-toolchain or reproducible-build attestation.
The separate automated comment-only check remains failed/open after two invalid
reports. Required legal notices and API ownership contracts remain intact.
No automated style pass is claimed.

## Public-history port

The reviewed preparation from archived source
`9f3fe49a78fadb6f4dfe146a6109206d4cc4e510` was ported as source changes onto
public trunk `a04b20c8dd27e545f74f1975835435667480a131` on 2026-10-20.
No private Git ancestry, raw evidence, probe binaries or checkpoint weights
were imported. Existing public attribution edits were retained alongside the
Clef additions. Original donor documents and checkpoint metadata keep their
pinned bytes.

The public-history host, native-host and startup gates passed all 17 build
steps and 160 executed tests, with another test subrun cached. The staged
privacy and pending-identity gate reported zero problems. These are CPU source
checks. They do not rerun the earlier device evidence or establish backbone
numerics, joint-head behavior or serving qualification.

## Verification sequence

1. Admit the pinned config, mutate unsupported fields and format claims, and
   prove plain execution is refused before device access.
2. Inspect captured original shard headers and the staged checkpoint. Verify
   427 text tensors, separate host tables, checked byte totals and refusals.
3. Implement payload loading and recording-uploader tests against the plan.
4. Add all-token hidden output with chunk-boundary and lifetime tests.
5. Compare independent backbone states, head logits and probabilities on the
   decision eGPU. Matching choices alone is insufficient.

Sequence Work into Verifiable Units puts a runnable gate before each next
layer. Model the Domain keeps encoding, arithmetic and head semantics distinct.
Test Behavior, Not Implementation makes the returned geometry and budgets the
contract, not source-string checks.
