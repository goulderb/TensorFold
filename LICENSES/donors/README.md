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

## Retain notices in future changes and releases

Keep original copyright headers, licenses, `NOTICE`, and credit files.
Record immutable source revisions and state what changed. Preserve transitive
donor credits and distinguish adaptations from design-only references.
Mark modified third-party files as their licenses require.

The native release packager copies `LICENSES` recursively, including this
addendum and the donor documents. The packaging regression checks the bytes
in the resulting archives. Recheck the exact artifact before publication.
Model licenses, compiled dependencies, and container licenses need separate
review. No public fork or release is approved by this attribution backfill.
