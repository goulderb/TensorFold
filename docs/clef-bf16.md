# Clef BF16 on the shared Qwen backbone

## Problem

Clef Flash uses Qwen 3.5 hidden states and a joint schema head. It is not a
language-model answer-token scorer. Its plain BF16 checkpoint must not enter
the guarded ModelOpt MoE loader or the MLX affine loader.

Checkpoint admission, validated planning, mapped payload ownership, shared CUDA
weight loading and a request-owned hidden terminal are implemented. An explicit
development-only BF16 initializer now connects them to the shared backbone.
The native joint head and the `tfclef` C library are implemented and checked on
the CPU only (see [Native joint head and tfclef](#native-joint-head-and-tfclef)).
No serving route uses them. Numerical backbone qualification, head parity and
device fit remain separate units.

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

## Bounded hidden-backbone diagnostic

The first full text-backbone run completed on the decision eGPU on 2026-10-21
UTC, using initializer source `98c1b613`. The private official-reference harness
captured all-token post-final-RMSNorm rows before the joint head. Native captures
used the same seven inputs at chunk sizes 1 and 16, with lengths 1, 2, 15, 16,
17, 33 and 151. Every capture returned finite BF16 rows. The two native captures
are bitwise identical for every case, and all six native prefixes match their
full-record prefix exactly.

The native/reference outputs are not bitwise equal. The 151-token record has
normalized RMS error 0.0267971, minimum row cosine 0.9876798 and maximum absolute
error 2.125. Token 61 has the largest squared error and row-normalized RMS error
0.1569954. The reference's short-prefix rows also differ from its own full-record
prefixes. That sequence-length sensitivity does not explain or excuse the
native/reference discrepancy. No numerical acceptance threshold or parity verdict
is assigned. Layer-level localization and a controlled reference-order comparison
remain open.

The approved Clef-only window restored the original Python BF16 service. Direct
and public requests returned HTTP 200, with the public request served locally.
Nemotron and proxy listener identities were unchanged. Across 116 physical GPU
samples during test children, the internal GPUs retained at least 4518/4760 MiB
free and the decision eGPU at least 2540 MiB. These bounded samples do not qualify
full 4096-token capacity, the joint head, probabilities or serving. No production
pin or route changed.

The fleet records sanitized results in
`fleet/trials/clef-tensorfold-hidden-backbone-20261021.json`. Raw files and build
receipts remain private. This is a diagnostic checkpoint, not a promotion.

## Native joint head and tfclef

`tfclef` is a C library for the existing Python decision service. It takes one
record from the official `encode_record` and returns one logit per option.
Python keeps request validation, tokenization, softmax and `systemone_answer`.
The library owns the CUDA context, the BF16 backbone, the host lexical gather
and the joint head. The head reimplements `JointSchemaHead.forward` from
Cloudflare's `joint_schema_model.py`; no Cloudflare code is copied.

### Build

```sh
# from a shell where nvcc and the host compiler work (vcvars64 on Windows)
zig build tfclef -Dnvcc=/path/to/nvcc -Dsm=89 -Dclef-source=$(git rev-parse HEAD)
# or embed prebuilt <name>.fatbin images
zig build tfclef -Dfatbins=/abs/dir -Dclef-source=...
```

The library lands in `zig-out/native/lib`. It embeds six images: `qwen35`,
`qwen35_hf`, `qmmf`, `modelopt_pack` and `vit` from the catalog, and
`clef_head` (`zig/kernels/cuda/clef_head.cu`, glue flags). `exl3_candidate` is
named by the shared projection dispatch's compile closure and embedded empty;
the BF16 path never loads it. Any other referenced image fails to compile.
`compile-clef-library` compiles the whole library with empty images, and
`test-clef-host` runs the family and C-header tests; both are part of
`test-cuda-host`.

### Usage

```c
#include "tfclef.h"

tf_clef_open_options o = {0};
o.model_dir = "/models/clef-flash";
memcpy(o.head_sha256, pin, 32);   /* joint_head.safetensors, from the verified HF metadata */
memcpy(o.device_uuid, uuid, 16);  /* the decision GPU */
o.cap_bytes = 14700ull << 20;
o.max_tokens = 4096;
char err[256];
tf_clef_engine *e = tf_clef_open(&o, err, sizeof err);

tf_clef_record r = {ids, n_ids, questions, n_questions, options, n_options};
float logits[64];
tf_clef_status s = tf_clef_decide(e, &r, logits, 64);
tf_clef_close(e);
```

`TF_CLEF_OPEN_HEAD_ONLY` opens the head weights and the output table without
the backbone. `tf_clef_head` then runs the same `Head.run` over caller hidden
rows: `n_ids x 4096` BF16 bits after the backbone's final norm. It is the
isolated head-parity entry. A full engine refuses `tf_clef_head`, and a
head-only engine refuses `tf_clef_decide`.

### ABI 1

| Function | Contract |
| --- | --- |
| `tf_clef_abi_version()` | Returns 1. Callers refuse a mismatch before open. |
| `tf_clef_open(options, err, len)` | Selects the device by UUID, then checks the budget against `cap_bytes` and free memory. It caps runtime allocations at `cap_bytes`, then loads the backbone and the head; the head file is hashed on its mapping against the pin before any upload. Every buffer is allocated here. Returns NULL with a message on failure. |
| `tf_clef_decide(e, record, logits, cap)` | Backbone and head. Writes one BF16 value widened to float per option, in record order. |
| `tf_clef_head(e, record, hidden, logits, cap)` | Head only, over caller rows. |
| `tf_clef_info_get(e, out)` | ABI version, limits, budget bytes, free and total device memory, UUID, source commit. |
| `tf_clef_last_error(e, buf, len)` | The last non-OK message; returns its full length. |
| `tf_clef_close(e)` | Frees everything when the engine drains; NULL is a no-op. |

| Status | Meaning |
| --- | --- |
| `OK` | Logits written. |
| `INVALID` | The call or record failed validation, or a mode mismatch. The engine is unchanged. |
| `CAPACITY` | `n_ids` exceeds `max_tokens`. The engine is unchanged. |
| `DEVICE` | A device step failed and the stream or context drained. The cache resets on the next call. |
| `FATAL` | A fault the engine could not drain. Every later call returns `FATAL`; close it and exit. |
| `BUSY` | Another call is inside the engine. Nothing ran. |
| `NONFINITE` | A logit is NaN or infinite. The engine stays usable. |

`Record.parse` is the only validation boundary. It requires 1 to `max_tokens`
ids, each below the vocabulary size, and 1 to `max_tokens / 2` questions and
options. Spans must be nonempty and inside the ids. Question types are 0 to 2.
The questions' option ranges must tile the options in order. The summed
option span lengths, the lexical rows, may not exceed the ids. The
option-to-question map is derived here, never taken from the caller.

A request restarts the one hidden-row owner, which resets its bound cache.
It forwards the ids in windows of at most `hf_prefill_rows` (16) rows and takes
the completed device rows. The output-table rows of every option span are
gathered on the host into pinned staging. One stream then uploads the request
tables and lexical rows, runs the head and copies the logits back, with one
synchronization. No request allocates. A device failure is probed with stream,
then context synchronization, never with the shared drain-or-exit helper: a
drain gives `DEVICE`, no drain gives `FATAL`. A sticky fault inside the
backbone forward still ends the process through the shared hidden terminal's
own drain rule, with exit status 1; the stack wrapper restarts it. Closing a
poisoned engine releases host state only and leaves device objects to process
exit.

### Budget at 4096 tokens

Logical allocations, computed by `Budget.of` on the CPU from the pinned
metadata (`engine.zig` tests pin these numbers):

| Allocation | Decision engine | Head-only engine |
| --- | ---: | ---: |
| Backbone steady (`Engine.bf16Budget`) | 14,128,534,528 | 0 |
| Head weights (30 lanes, 10 split in_proj, rest stored) | 243,786,752 | 243,786,752 |
| Head load stage (freed after open) | 8,388,608 | 8,388,608 |
| Head scratch | 226,560,000 | 260,114,432 |
| Lexical rows on the device | 33,554,432 | 33,554,432 |
| Steady device total | 14,632,435,712 | 537,455,616 |
| Load peak | 14,632,435,712 | 537,455,616 |
| Pinned host (lexical stage, tables and logits, backbone input) | 33,771,520 | 33,640,448 |
| Counted by the allocation cap at the peak | 14,666,207,232 | 571,096,064 |
| Mapped host vocabulary tables (logical) | 4,068,474,880 | 4,068,474,880 |

The decision engine's steady total is 13,955 MiB, under today's 14,700 MiB
process cap. Driver, context and module overhead and resident mapped pages are
not included; this is not fit evidence. The head-only scratch adds a
`[4096, 4096]` BF16 upload destination for caller rows.

### Numerical policy

Each value torch's eager BF16 graph materializes is rounded to BF16 at the
same point; reductions run in FP32 in a fixed order. Sources are torch 2.8.0
as installed in the Python service's runtime; where only compiled code holds
the rule, the policy is labelled inferred.

| Step | torch semantics mirrored | Native |
| --- | --- | --- |
| Linear | `F.linear`, one BF16 rounding of `xW^T + b` (cuBLAS epilogue; compiled, inferred) | BF16 lane FP32 sums, then `tf_vit_bias` once; bias-free Linears round in the lane |
| LayerNorm | `nn.LayerNorm`, eps 1e-5 (`normalization.py:180`, `transformer.py:1018`); kernel compiled | `tf_vit_layer_norm`; a 4096-wide copy of its arithmetic for `hidden_norm` |
| GELU | `approximate="none"` (`activation.py:733`, `transformer.py:1189`): erf | `tf_vit_pointwise` erf GELU |
| Span mean | `aten.mean.dim` (traced), FP32 accumulation (`AccumulateType.h:130`); multiply by the count's reciprocal (compiled, inferred) | `tf_clef_span_mean`: FP32 sum in index order times `1/len` |
| Decoder layer | norm-first order (`transformer.py:1110-1120`); self-attention takes the MHA fast path in eval (`activation.py:1250`, `1320`); cross-attention and routing take `F.multi_head_attention_forward` (`functional.py:5694-5729`, `6484`) | Three Q/K/V lane Linears, attention, out Linear, BF16 residual adds in torch's order |
| Attention | Memory-efficient SDPA on Windows (no flash kernel there, per the service's `attn_config.py`; selection inferred). First kernel at head dim 64 is 64x64 (`cutlassF.h:26-29`). Scores times `scale`, then `log2e` (`kernel_forward.h:799-802`, `916`, `1220`); `exp2f` against an online maximum (`1257`, `1291`); row sum from unrounded P (`1293-1300`); P stored as BF16 for P.V (`mma_from_smem.h:1524-1537`); output times `1/sum` (`epilogue_rescale_output.h:182-189`) | `tf_clef_attention` with the same 64-key tiles and rounding points; FP32 dot products in a fixed order rather than tensor-core order |
| Routing summary | `mv`, divide by `sqrt(1024)`, `_softmax` over dim 0, BF16 product, `sum` (traced) | `tf_clef_segment_summary` |
| normalize | `input / norm.clamp_min(1e-12)` (`functional.py:5567`, traced) | `tf_clef_normalize` |
| cosine | torch 2.8 decomposition: each side divided by `max(norm, 1e-8)`, product, sum (traced) | `tf_clef_score_features` |
| Scales and gate | `clamp(max=ln 100).exp()`, `sigmoid` (traced); the clamp never binds for the pinned 2.296875 scales | `tf_clef_logit_combine` |

Recording-driver host tests check launch order against the graph, every write
extent against named buffers, and zero allocations and syncs per head run.
These checks bound control flow and extents, not values.

### Unqualified

- No kernel, head or engine has run on a device. Head parity against the
  official head on identical hidden rows, the kernel units against FP64
  references, and the noise-floor thresholds remain the next device unit.
- The SDPA backend torch selects on the service, its key-tile size and the
  cuBLAS reduction policy were not observed at runtime.
- Physical device fit at 4096 tokens, open time, latency and the attention
  kernel's speed are unmeasured.
- The backbone's native/reference discrepancy above remains open.
- The Python service integration (`NativeClefEngine`, its owner thread and the
  launcher branch) is a separate fleet unit.

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
