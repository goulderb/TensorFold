# Local fork credits and changes

This addendum describes the local TensorFold fork, including the v1.0.12-based
source at `114beb79335d4b33833bc842ed37f1bf4723541d` and the in-progress
v1.1.0 shared-layer integration at upstream `76aafcf27` (2026-10-19).
The latter retains the fork's GLM and Nemotron families, adapts their shared
APIs, and keeps their incompatible EXL3/FP8-KV kernel ABIs explicit. The GLM
expert kernel is renamed `exl3_experts_glm.cu`, not attributed as new work;
upstream's newer pointer-table implementation keeps its original source and
credits. Retained checkpoint-math dense operators remain extensions of the
upstream shared CUDA layers. Integration is not fresh GPU qualification.
It supplements the upstream `NOTICE`, `THIRD_PARTY_NOTICES.md`, and
`docs/credits.md`. It does not replace their copyright or license terms.

## TensorFold and the native GLM family

[TensorFold](https://github.com/ashhart/TensorFold) is by Ash Hart
([ashhart](https://github.com/ashhart)) and the TensorFold contributors.
Upstream v1.0.12 is `833bbfcb`; the fork retains upstream Apache-2.0 and the
MIT notice for older code in `LICENSE` and `LICENSES/MIT.txt`.

The native GLM family also incorporates work by
[Bizuayeu](https://github.com/Bizuayeu/TensorFold) from PR #804,
`pr/glm53-flash-cuda`, head
`6380c8daec97dcf6fa7f18ac95906b411134b09d`. The local history records the
integration at `cd829da2cf5ebd47c46fb9b4460dc390a78d9e16` and
`8fb24a84f`. This includes family serving and all-rank memory admission.

[BHCC2025](https://github.com/BHCC2025) (BHCC Spark Lab) wrote the native
CUDA EXL3 support on the fork's `glm-exl3-wip` branch,
[BHCC2025/TensorFold](https://github.com/BHCC2025/TensorFold), head
`9a099a7ce80781c227b418fe08212ff54f9745fc`. Five of its commits were
cherry-picked onto `trunk` with BHCC2025 kept as the Git author:

| Local | Source | Change |
| --- | --- | --- |
| `0eb3551d9` | `8f2fc5a074cde101911ec7dde4be9deebcdeefb3` | EXL3 projections behind qlinear |
| `d35a6565c` | `4e0238d8d56ce27aaaefe7d646f8bececa25df8c` | EXL3 routed experts behind experts.Grouped |
| `9826affae` | `5180c729cf359999be4ad8a04db39c84c73ce7cf` | EXL3 projections write FP32 for TP partial sums |
| `9a10630c3` | `e61015bcd3862011c179ca9e7a2aad7b7d00f83a` | GLM EXL3 checkpoints on the CUDA family, including MTP |
| `b3c1f8e02` | `9a099a7ce80781c227b418fe08212ff54f9745fc` | EXL3 test for matrices stored as plain `.weight` |

The same work also appears on the fork's `exl3-f32` and `exl3-experts`
branches. Later local EXL3 commits, such as routed-only checkpoint loading and
tensor-core decode, build on these files and are local changes. BHCC2025's
separate Intel Arc XPU commits arrived through upstream v1.0.12 and keep their
upstream authorship. The vocabulary split patch posted on TensorFold tracker
#593 is credited to BHCC2025 by local commit `6745dc228`.
These donor revisions' license and notice files are retained beside this
addendum. Source IDs identify provenance, not a promise that every original
commit remains an ancestor after a rebase or cherry-pick.

## NVIDIA GLM-5.3-Flash-FP4-Local support

`zig/src/core/csf.zig` is a new implementation of the `csf_32x2_v1`
compressed NVFP4 scale format documented by
[Local Inference Lab](https://github.com/local-inference-lab) in
`docs/csf-scale-blocks.md` of
[their FlashInfer fork](https://github.com/local-inference-lab/flashinfer),
commit `7e203806a82b588b9113f5a0866a028564847741`, Apache-2.0. No code from
that repository is included. Its reference encoder and decoder
(`scripts/encode_csf.py`) were used outside this tree only to verify the
decoder against real checkpoint data. The test fixtures under
`zig/src/core/fixtures/csf/` come from an independent encoder.

`modelopt.zig`'s mixed-precision recipe table, `fp8_source.zig` and the GLM
loader read NVIDIA's ModelOpt `MIXED_PRECISION` layout as published in
[nvidia/GLM-5.3-Flash-FP4-Local](https://huggingface.co/nvidia/GLM-5.3-Flash-FP4-Local).
`zig/src/core/fixtures/glm53_flash_fp4_local/config.json` and
`zig/src/families/glm/fixtures/fp4-local-config.json` are copies of that
checkpoint's `config.json` at revision
`76348ddb2be2e35e39ecfc00e75dc07703f4f3d5`. The checkpoint is governed by the
NVIDIA Open Model Agreement and its GLM-5.3-Flash base model by MIT. No
weights are included.

## Mia's native arithmetic and captured kernels

The donor is [MiaAI-Lab's GLM-5.3-Flash TensorFold recipe](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold),
`zig-engine` commit `f1ee79a6bf5319775079412c727f5f6b0ebb1e19`.
The seven applied native patches are preserved at
`f013e1aec57dbee167f43cccb66186da5bacc37f`. Local import
`0556c315b1241e441200d9a843443f52d079567b` records both revisions.

The original [NOTICE](mia-f1ee79a6/NOTICE),
[LICENSE](mia-f1ee79a6/LICENSE), and [CREDITS.md](mia-f1ee79a6/CREDITS.md)
are unmodified copies from that exact donor revision.
`mia-f1ee79a6/source.json` records their SHA256 digests.
Their references to the donor's project describe that project, not the
qualification or configuration of this fork.

The import adapts DFlash2 and fused MCG4 expert arithmetic to the fork's GLM
engine. The donor credits Ash Hart and the TensorFold contributors for the
engine and arithmetic, Mia's AI Lab for the native port, and Jürgen Schmied
([jschmied](https://github.com/jschmied)) for the native CUDA serving path and
family registry in TensorFold PR #443, commit
`59e77e8f4b875ce0e863a8c896fc8e424bc539ac`.
The EXL3 format and codebooks are by Turboderp under MIT. The adapted expert
load path also credits Jay Leaton in `zig/kernels/cuda/glm_exl3.cu`.

Local changes connect the imported numerical components to the existing
family engine, rank link, lane sampler, accepted-state tracking, snapshot
and spill paths. They add the prompt-slot FP32-to-BF16 conversion and adapt
the import to the v1.0.12 APIs. `docs/glm-cuda-mia.md` describes the integration
and its qualification limits. The importer is not the original author.

The local seven-draft extension sizes Mia decode scratch for up to 64 shared
rows and adds boundary replay checks. It preserves the donor's expert
arithmetic and captured DFlash kernels.

The source tree includes 24 captured DFlash2 cubins under
`zig/kernels/dflash2-aot/`. Its `provenance.json` identifies the donor archive
and file hashes. These are software artifacts, not checkpoint weights.
Their presence does not establish GPU qualification or permission to
redistribute every related model or container dependency.

The retained donor credits also distinguish design references from copied
code. They credit local-inference-lab's b12x/RoCEnante transport, Jay Leaton's
GLM work, and specified design contributions from drowzeys and BertholomusAI
(Albert Lee, GitHub `bertholomus`). Do not reduce these to a single Mia credit
or label design-only references as copied code.

## Production GLM copy and row-policy adaptation (2026-10-19)

`zig/src/families/glm/cuda_copy_drafts.zig` adapts TensorFold's Python
`families/glm5_next/cuda/copy_drafts.py` from the pinned Mia/TensorFold
production image
`sha256:d7e14f8799ad21f7e93ba95ea78826426c98ab27d5fbfdf81bed876a2e09ce6e`.
The exact donor file SHA256 is
`a8ab7c52f7ba5d7466f27aaef1dba6120930f9fee7e579a0fa32ea74fc591161`.
The shared longest-tail row policy follows that image's `multi.py`, SHA256
`1386e5275e6c6b0578a698a639e0d578f8d1bfbb69207624829cc2d58576bda2`.
The workstation evidence is
`C:/Development/prep/glm53-expert-tuning/production-source-receipt.json`.
These image/file pins identify the adapted content; they do not invent an
individual author or Git revision for an image snapshot.

Credit remains with Ash Hart and the TensorFold contributors and the
production recipe's retained Mia and transitive donor notices above. The local
changes use the native upstream proposer factory, borrow committed context
instead of owning a second history, preserve original-copy miss accounting,
and separate the unpadded multi-lane path from optional standalone padding.
The opt-in shared-policy extension preserves existing head-only callers.
No Python donor code executes in the native engine. This adaptation is not a
new authorship claim, graph-execution equivalence, or GPU qualification.

## Partial GLM target graphs (2026-10-19)

The opt-in native FFN-site graph bank uses upstream TensorFold's `cuda.graph`
wrapper and the retained GLM HC/router/expert/collective path credited above.
Local changes extract that stable site, manage graph lifetime by engine-owned
scratch/layer/exact rows, and add paired failure handling and replay proofs.
The partial predecessor used pinned production `verify.py`, `segments.py` and
`forward.py` as design references only. The full target port below subsequently
adapts segmented kernels and metadata. Existing Ash Hart/TensorFold, Mia, wojo
and transitive credits remain; neither path is a new authorship claim for their
arithmetic or a GPU qualification.

## Complete GLM target-forward graph port (2026-10-19)

The native full-target port follows the production image pinned above and adapts
its segmented KDA prep/recurrence orchestration in `zig/kernels/cuda/glm_kda.cu`.
The exact donor `families/glm5_next/cuda/kda.cu` SHA256 is
`f74056a38e6f18a5d1995cf994e0f3e0231b24a530da827a350f221aedfbdafc`.
Local changes load absolute per-sequence state/replay pointers from staged device
tables and preserve the retained native FP32 convolution weights and update4
arithmetic, rather than imposing the production contiguous-arena layout.

`glm_target_dsa.cu` and the native target adapters extend the retained GLM latent,
index, selection and attention arithmetic with per-row device metadata. The
production segmented math and sharded query/owner protocol are references from
`latent.py` (SHA256
`73edd5297525ee5fc457c04e4e622e0413b17666c194427d9ddb993e0ce000e6`),
`sparse.py` (`80b8de2c2aaccc3c128fdaf65d83b25f66973bb76b58693f2d95b399b723d1cd`)
and `shard.py` (`204577f02fb60dd8a9fb724ed5818dc256415c8c79aa4091daa44ada0386fdd1`).
The native port uses fixed NCCL collectives and byte-preserving device owner
selection; it does not import RoCEnante/b12x transport or the TP3-only
`seg_chunks.cu` specialization as a TP2 implementation.

Original production-lineage legal documents are retained unmodified at
[mia-b5808ac6](mia-b5808ac6/NOTICE) and [wojo-7fac72b3](wojo-7fac72b3/NOTICE),
including each revision's LICENSE, CREDITS.md and exact source/hash manifest.
These identify legal lineage, not a claim that an image file equals an entire
Git revision. Keep the upstream TensorFold/Ash Hart, MiaAI-Lab, Robert
Wojciechowski and all retained transitive credits. No model weights, model
licenses or permission to publish are supplied by this source adaptation.

The engine-owned tables, stable aggregate-row scratch, full-forward graph
lifecycle, paired failure handling and eager/segmented/replay probes are local
integration changes. Arithmetic, output parity, graph memory, cache/spill/vision
and full-model performance require their own device qualification.

## GLM production DSA projection layout (2026-10-20)

The combined Q4 QA and KVA input follows `families/glm5_next/cuda/weights.py`
from production image
`sha256:d7e14f8799ad21f7e93ba95ea78826426c98ab27d5fbfdf81bed876a2e09ce6e`.
The exact file SHA256 is
`7c1070e1d83b2a1eeccb7cfa5cd39069a2d1f779005eda77deeea49b800ed013`.
This layout adaptation uses the retained native quantizer and projection APIs.
Local changes add the combined quantized weight representation, strided norm
dispatch, loaded-format receipts, and a distinct numerical identity. Existing
Ash Hart/TensorFold, MiaAI-Lab, wojo, and transitive credits remain unchanged.
The production-lineage notices above apply; this is not a new authorship claim
for the quantizer, arithmetic, or model weights.

## GLM production scheduling, transport and L2 prefetch (2026-10-20)

The deferred DFlash launch follows the pinned production image above, including
its ordered batch and copy-first scheduling. Native code retains Ash Hart and
TensorFold's lane engine and adapts the existing Mia DFlash implementation.
The local changes defer host readback, preserve slot generations, pair launch
commands, and resolve selected shape before allocating verification rows.

The L2 planner, runtime, and `glm_l2pf.cu` adapt the retained Mia native donor
at Git import `f013e1aec57dbee167f43cccb66186da5bacc37f`, from MiaAI-Lab
revision `f1ee79a6bf5319775079412c727f5f6b0ebb1e19`. Its legal files remain
unmodified in `mia-f1ee79a6/`. The production planner credits
`jayleaton/glm53-tensorfold-spark`, patch `0460`, Apache-2.0, Copyright 2026
Jay Leaton. Preserve that original and transitive credit. Local changes use
current typed weight formats and allocation bounds, combine Q4 DSA and dense
MLP inputs, own graph-lived stream events, and keep prompt paths distinct.
The production planner's SHA256 is recorded with the build receipts.

Transport reuses the fork's existing `cuda.roce` and `roce_gather.cu`, which
retain b12x's RoCEnante attribution (`local-inference-lab/b12x`, Apache-2.0).
Local changes borrow the existing NCCL communicator and connect GLM's target,
DFlash, and owner attention gathers. The side-stream prompt protocol retains
NCCL. No new transport implementation or new authorship is claimed.

## GLM chunked KDA prompt recurrence (2026-10-20)

`zig/kernels/cuda/glm_kda_chunk.cu` reuses the native extraction retained from
MiaAI-Lab's `zig-engine` revision `f1ee79a6bf5319775079412c727f5f6b0ebb1e19`
and import `f013e1aec57dbee167f43cccb66186da5bacc37f`, identified above. Its
header credits Ash Hart and the TensorFold contributors for the original
`families/glm5_next/cuda/kda_chunk.cu` arithmetic. The device body was also
compared against the production image pinned by wojo recipe revision
`7fac72b346eb85549edc0d1783b6bbb3d998a5af`, image
`sha256:d7e14f8799ad21f7e93ba95ea78826426c98ab27d5fbfdf81bed876a2e09ce6e`.
The captured production file has SHA256
`aa6530d28d3adbc847654b939b30e58fb7b06235b88cd95561236953c800a29c`;
the compared device body differs only in trailing whitespace.

Local changes adapt its launch to the current typed KDA inputs, own the CUDA
module, and separate prompt output/state from serial verification replay
saves. They add absolute prompt position, rank agreement and numerical cache
identity. They do not claim new authorship of the WY algorithm or device
arithmetic. Retained upstream, MiaAI-Lab and wojo notices continue to apply.

The prompt split planner and front scheduling also adapt that same donor's
`cuda_prefill_split.zig` and `cuda_prefill.zig`, and the pinned production
`hcsplit.py`/`forward.py` behavior. The local port retains the current engine,
streams, typed projections and state ownership. It uses contiguous rank halves,
queues row-independent fronts during normed-row swaps, preserves rank-ordered
partial sums, and carries prompt arithmetic through short Q4 row ranges.
These are adaptations of the credited production and donor components, not a
new prefill engine or a new attribution claim for their algorithms.

## The shared spill tier and Nemotron adapters

`zig/src/core/spill.zig` and `docs/spill-tier.md` identify the GLM spill tier
by Robert Wojciechowski ([wojo](https://github.com/wojo)), Mia patch `0088`,
as the source of the settings, eviction behavior, and multi-rank rules.
The local implementation is new Zig, with model-neutral source and sink
interfaces. Later work adds the qualified Nemotron adapter, Windows file
handling, a host-memory cache level, and TP=2 persistence.
Those local changes do not make the inherited design a local invention.

Mia's retained credits identify the earlier lineage:

- Jay Leaton's patch `0250` supplies the multi-rank design.
- `gabewillen` proposed per-rank NVMe persistence in the vLLM kit. Mia explicitly
  credits the idea only, with no code copied from that AGPL project.
- `gilby` and `raymondkpwong` supply upstream spill conventions and proposals.
- `Alexbob0` supplies superseded-first eviction behavior.
- `jakejharris`'s JSpark3 v2.0.1 supplies session-tier safety conventions.
- Thomas Wade (`ThomasWadeZ`) contributes the `0089` kept-state limits.

The Python GLM production recipe's later wojo block-store work is a separate
source series. Its BadAd84 and wojo credits belong to that recipe; this
addendum does not claim that the native fork implements every later patch.

## GLM production Q4 prompt tile policy (2026-10-20)

The GLM selector and additional tile instantiations adapt production patch
`0102-glm-prompt-matmul-tile`, credited to
[BadAd84](https://github.com/BadAd84) in the retained wojo and Mia credits.
The patch SHA256 in the wojo `7fac72b346eb85549edc0d1783b6bbb3d998a5af`
input manifest is
`da03a9c3d2a94ef74fca01cea6976448ad4c758c3af2c1ebbe4c3a36ce7a2676`.
The compared production image is
`sha256:d7e14f8799ad21f7e93ba95ea78826426c98ab27d5fbfdf81bed876a2e09ce6e`;
its `cuda/kernels/qmm_prefill.cu` SHA256 is
`be112df7a73c47530f998caaf44fd75a8490e59ec1152f89d170d72f10c13f9b`.

The existing Ash Hart/TensorFold prefill arithmetic and fragment helpers are
retained. Local changes add an opt-in GLM policy, resolve the additional
BF16/FP32 tile symbols through the existing module owner, match their launch
geometry, isolate persisted numerical identity, and add guarded device proofs.
Other families and the decode policy retain their defaults. Original,
MiaAI-Lab, wojo and transitive notices continue to apply.

## GLM prompt latent AOT kernels (2026-10-20)

`zig/src/families/glm/prompt_aot/` retains unmodified production Triton
`_sparse_onepass`, `_absorb_mma` and `_expand_mma` cubins, PTX, metadata and
their `latent.py`/`kv8.py` sources. These are Ash Hart/TensorFold contributor
algorithms from the patched TensorFold production distribution. Its source
lineage is TensorFold `c4646171139ee8a3c38103eaa1699dad226ec12b`, Mia
`b5808ac620c795ec9c0640f3ca2a2f7be21aaa63`, and
[wojo's patch series](https://github.com/wojo/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold)
`7fac72b346eb85549edc0d1783b6bbb3d998a5af`. The exact compared distribution is
image `sha256:d7e14f8799ad21f7e93ba95ea78826426c98ab27d5fbfdf81bed876a2e09ce6e`;
the module's `provenance.json` pins both actual source files, Triton 3.7.1,
SM121 and each captured payload. The original, Mia, wojo and transitive notices
retained above continue to apply; capture and import do not establish authorship.

Local changes add a native Zig owner over the existing Triton CUDA loader,
strict embedded-byte validation, explicit prompt eligibility and fallback,
rank agreement, persisted numerical identity and bounded component proofs.
The original arithmetic and captured compiled bytes are retained unchanged.
The payloads do not include model weights or a new license grant for them.

## Qwen CUDA mathematical reference

`zig/kernels/cuda/qwen35_ops.cu` and
`zig/src/families/qwen3_5/cuda_kernels_test.zig` adapt the Qwen text equations
from Hugging Face Transformers, revision
`e0eb2810ac1b2caf6d082f247d8d37162ecf67a0`, source
`src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py`.
The original copyright names the Qwen Team and the HuggingFace Inc. team,
Copyright 2025. The Apache-2.0 license is retained unmodified in
`transformers-e0eb2810/LICENSE`, and the adapted files retain its header.

Local changes implement fixed 35B CUDA entry points, serial persistent GDN
state, online causal attention, deterministic top-eight routing and small
independent host reference fixtures. The existing CUDA packing and matmul
kernels retain their original TensorFold and donor credits. TensorFold
`upstream/v0.6.6`, commit `cb2ebf0540f42604e2759b2ddef497861e928248`, was
also inspected as a semantic reference. Its older Qwen GDN rounding was not
copied into this implementation. These additions do not establish numerical
parity with a particular fused implementation or approve a model deployment.

`tf_q35_hf_attn_partial_chunked`, `tf_q35_hf_attn_partial8_chunked` and
`tf_q35_hf_attn_merge_chunked` in `zig/kernels/cuda/qwen35_hf.cu` (sources
`349615f8b` and `12ef22ecb`, merged at `6b558f76c`) are local changes. They
keep the serial HF partial's per-key arithmetic: the pairwise dot order,
unfused products and `expf`. They run it on upstream TensorFold's chunked
attention geometry from `zig/kernels/cuda/qwen35.cu` and `qwen35.cuh`:
`chunk_keys`, `MAXCH` and the partial/merge grid. Upstream commits
`3c57c1433` and `02d3e2ed1` record BobClawblaw as Git author. The FP8
partial reads `qwen35_kv8.cu`'s e4m3 row and scale layout (`58b767435`, same
recorded author, merged as #886). The Qwen/Hugging Face notice and the
upstream TensorFold notices continue to apply.

`zig/tests/cuda/qwen35_attn_compare.zig` and `qwen35_margins.zig` are
local diagnostic tools with no external donor.

The four Qwen checkpoint fixtures contain configuration and tensor metadata
only. Each `source.json` names the producer, immutable model revision and
header hashes. Checkpoint weights are not included and retain their producers'
separate terms.

## Clef Flash configuration and tensor metadata

`zig/src/families/qwen3_5/fixtures/clef_flash/` retains the unmodified
configuration, safetensors index and JSON shard headers from
[Cloudflare/clef-flash](https://huggingface.co/Cloudflare/clef-flash), revision
`fde727a287004204b7518dcc983fe64379776712`. Cloudflare's model card identifies
the Apache-2.0 license and the `Qwen/Qwen3.5-9B` base. The original license and
model card are retained unchanged in `clef-flash-fde727a2/`. `source.json`
records their SHA256 digests and the config digest. `headers/capture.json` pins
the original index/header hashes and shard sizes; `headers/provenance.json`
states the capture scope. Header capture does not rehash checkpoint payloads.

Local changes add strict BF16 config admission, a CPU-only validated tensor
manifest and allocation plan, mapped payload ownership and host row gathers.
The weight loader consumes that manifest through the existing TensorFold BF16
packer, projection views and parameter conversion. A separate binding owner
reuses the existing HF operators without the ModelOpt-only expert bundle.
The shared backbone also gains request-owned hidden output with host lifetime
and graph-identity tests. A development-only dense initializer owns BF16 weights,
separate host vocabulary tables, bounded pinned input staging and existing kernel
bindings. An explicit terminal-weight union keeps final normalization separate
from vocabulary projection. Ordinary plain BF16 loading and native serving remain
closed. No Cloudflare model implementation, checkpoint weights, joint-head
arithmetic or CUDA kernels are copied. CPU, primitive and weight-only checks do
not establish backbone parity or full-engine GPU fit.
Existing Qwen/Hugging Face, TensorFold and
transitive notices remain applicable.

## Clef Flash joint schema head design

`zig/src/families/clef/` and `include/tfclef.h` adapt the design of
`joint_schema_model.py` from
[Cloudflare/clef-flash](https://huggingface.co/Cloudflare/clef-flash), revision
`fde727a287004204b7518dcc983fe64379776712`, Apache-2.0, whose license and model
card stay in `clef-flash-fde727a2/`. Type of reuse: design only. The head
checkpoint walk derives its 122 tensor names and shapes from that file's
`JointSchemaHead`, `EvidenceRoutingLayer` and PyTorch `TransformerDecoderLayer`
module tree and `joint_head_config.json`. `head.zig` and
`zig/kernels/cuda/clef_head.cu` reimplement `JointSchemaHead.forward` for one
record. The C record carries `encode_record`'s input ids and question and
option spans. Nothing is copied from Cloudflare code into Zig, C or CUDA.
Numerical semantics follow PyTorch 2.8 (BSD-3-Clause, `../PyTorch-BSD-3.txt`)
as read from the installed `torch/nn` sources and ATen headers, including the
memory-efficient SDPA forward's 64-key tiles; no PyTorch code is copied. `zig/src/families/clef/fixtures/`
retains the unmodified `joint_head_config.json` and `joint_head.safetensors`
JSON header bytes; `capture.json` records their SHA256 digests, the file size
and the pinned file digest from the revision's Hugging Face metadata. No head
weights are included.

Local changes: strict config and header admission, a device packing plan
through the existing BF16 lane packer, a digest check on the mapped file before
any upload, record validation with a derived option-to-question map, a checked
device budget and the versioned C export layer with host tests. The forward
reuses vit.cu's LayerNorm, bias and pointwise kernels and the BF16 lane matmul;
`clef_head.cu` adds span means, the option-query and field adds, attention,
the per-question routing summary, normalization, the score features and the
logit combination, plus a 4096-wide copy of vit.cu's LayerNorm arithmetic.
The decision engine drives the shared BF16 backbone in 16-row windows; the
shared hidden-row owner gains `restart(rows)` so one allocation serves every
request. `zig/build/clef.zig` builds the `tfclef` dynamic library with its six
kernel images.

## Qwen host sampler tuning

The local change at `b864564587f00f308ff9215c9870bffcd11da61f`
extends TensorFold's existing `zig/src/core/lanes/sampling.zig` from fork
source `ad247b3daf045015d59bd106a88591bff961e488`. Finite greedy rows use
an allocation-free maximum scan with the original comparator. The original
sorting and sampled body is retained for all fallback cases. TensorFold's
original notices, contributors and seed-salt attribution remain in place;
this change claims no authorship of the existing sampler or model math.

Generated Python sampler vectors are preserved under
`docs/plans/qwen35-ornith/evidence/tuning-20261019/`. They were recorded by
the existing `tools/zig/record_lanes.py` and `lanes_units.py` from TensorFold's
Python `engine/exact_sampling.py`, checkout
`a601ddee69f2eb41bf8faf28ab14a4255e3a8f98`. Its latest file revision is
`8a836f2d927c19c9296634d454490a045ddf0b06`, Jürgen Schmied's seed-salt
change, already credited in `docs/credits.md`; that commit is not a claim
of original authorship of the whole file. The fixture manifest records the
source and generated-byte hashes. No Python source or new external kernel
is copied into this unit. Existing TensorFold and transitive donor licenses
and notices remain unchanged.

## Retain notices in future changes and releases

The October 19 Qwen operator evidence adapts TensorFold contributors'
`zig/kernels/cuda/ck_lane.cu`, `mma4.cuh`, `qmm_frag.cuh`,
`nvfp4_experts.cu` and `experts.cuh`, preserved at fork revision
`62a2a679878e5d57e6949d0fd6ce009d408fc211`. The diagnostic source snapshots
retain their original headers. The stage-depth clone changes the pipeline
depth and entry names; the MMA clone adds FP32 trace stores and separate
entry names. The slice-major experiment permutes complete packed words and
changes the corresponding scalar address in a renamed split-M consumer.
These are local diagnostic adaptations, not original authorship of the
operators or their mathematical references. TensorFold's copyright,
Apache-2.0 license, earlier MIT notices where applicable, NVIDIA runtime
dependency terms and all existing transitive credits continue to apply.
The archives pin actual source/header/compiler bytes and state the local
changes; they do not package or relicense the checkpoint or CUDA runtime.
The terminal-greedy screen reuses TensorFold's existing tree-round BF16
argmax and retains the diagnostic host comparator as a source-pinned control.
The shared-activation screen extracts the existing Qwen HF activation,
renames its control and adds a diagnostic BF16 intermediate round plus a
separate recorder. It retains the original mathematical-reference credits
above; the diagnostic candidate is not a newly authored model equation.
The isolated expert-combine screen retains the same Qwen/Transformers
reference and native combine control. Its candidate makes BF16 route,
down, product and accumulator boundaries explicit and visits the selected
experts in ascending ID order. A separate recorder checks those boundaries.
Expected HF outputs remain host comparison data. This is a local diagnostic
adaptation, not a new model equation or a change to checkpoint licensing.

The next tuning probes retain the entire original shared sampler at local
revision `ad247b3daf045015d59bd106a88591bff961e488` as
`zig/tests/fixtures/qwen35/sampler_before.zig`. It is a byte-identical
benchmark control, not newly authored sampling logic. Its SHA256 is
`c8656618cdcabd582388cf25dca2dc0b1d05333266311aebf1cfb0b61b206c41`.
The head probe reuses the existing Qwen loader and shared norm/NVFP4 CUDA
operators. `tools/zig/record_qwen35_head.py` reuses the existing Qwen CPU
reference implementation to generate residual, norm and head-output data;
the fixture manifest pins that reference source. No new external kernel,
checkpoint weight or license text is imported by these probes. Original
and transitive TensorFold, Qwen, NVIDIA and other retained donor credits
continue to apply to their existing implementations.

Keep original copyright headers, licenses, `NOTICE`, and credit files.
Record immutable source revisions and state what changed. Preserve transitive
donor credits and distinguish adaptations from design-only references.
Mark modified third-party files as their licenses require.

The native release packager copies `LICENSES` recursively, including this
addendum and the donor documents. The packaging regression checks the bytes
in the resulting archives. Recheck the exact artifact before publication.
Model licenses, compiled dependencies, and container licenses need separate
review. No public fork or release is approved by this attribution backfill.


The opt-in Qwen slice-major expert-up representation adapts TensorFold's existing
`modelopt_pack.cu` packer and the retained `nvfp4/experts.cu` split-M arithmetic.
It writes the code-word permutation directly into the original up allocation and
adds an explicit split-M consumer with slice-contiguous scalar loads. Scale words,
down packing and arithmetic retain the original implementation. This local tuning
adaptation imports no new donor kernel. All original and transitive notices above
remain applicable, and checkpoint provenance and licensing remain separate.


## Asynchronous tier restore (2026-10-21)

The local asynchronous reader extends the existing wojo/Mia-derived spill tier
and upstream TensorFold compact Image adapter credited above. It separates file
reads and validation from owner-thread GPU uploads, reserves pending cache state,
and makes cancellation and invalidation generation-aware. It adds no donor code
or CUDA kernels; original and transitive notices remain applicable.

## Publication privacy boundary

This source-only publication omits private workstation evidence and run receipts.
References above to generated vectors, journals and diagnostic snapshots refer
to the canonical private evidence, not files included in this copy. All original
credit text and source pins above are retained. See [PUBLICATION.md](../../PUBLICATION.md).
