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

The four Qwen checkpoint fixtures contain configuration and tensor metadata
only. Each `source.json` names the producer, immutable model revision and
header hashes. Checkpoint weights are not included and retain their producers'
separate terms.

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

Keep original copyright headers, licenses, `NOTICE`, and credit files.
Record immutable source revisions and state what changed. Preserve transitive
donor credits and distinguish adaptations from design-only references.
Mark modified third-party files as their licenses require.

The native release packager copies `LICENSES` recursively, including this
addendum and the donor documents. The packaging regression checks the bytes
in the resulting archives. Recheck the exact artifact before publication.
Model licenses, compiled dependencies, and container licenses need separate
review. No public fork or release is approved by this attribution backfill.
