# Mia DFlash2 and routed EXL3 in the GLM CUDA engine

The GLM task branch imports DFlash2 and the fused MCG4 expert arithmetic from
[Mia's zig-engine branch](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/tree/f1ee79a6bf5319775079412c727f5f6b0ebb1e19).
The pinned recipe commit is `f1ee79a6bf5319775079412c727f5f6b0ebb1e19`.
Its seven native patches, applied to upstream v1.0.7, are preserved at
`import/mia-zig-f1ee79a6`, commit `f013e1aec57dbee167f43cccb66186da5bacc37f`.
The prior GLM candidate is preserved at `archive/glm53-before-mia-20261016`.

This imports numerical components into the existing `glm` family. The current
Engine, rank link, target sampler, lane acceptance, vision and persistent store
remain the execution path. The GLM branch now integrates the shared v1.0.12 base.

## Runtime controls

| Setting | Values | Default |
| --- | --- | --- |
| `TF_GLM_DRAFTER` | `off`, `mtp`, `dflash2` | `mtp`, disabled when the existing drafts option is off |
| `TF_GLM_DFLASH_DIR` | DFlash2 checkpoint directory | Required for `dflash2` |
| `TF_GLM_DFLASH_KERNELS` | Pinned captured-kernel directory | Installed DFlash2 assets |
| `TF_GLM_EXL3_ROUTED` | `generic`, `mia-mcg4` | `generic` |

DFlash2 requires the existing drafts option, TP2, sm121 and a dense-source
`lm_head.weight` in the target checkpoint. The routed-only EXL3 production
checkpoint has that head. A full-EXL3 head is refused. DFlash2 evaluates the
donor's eight-position block and returns at most three proposals through the
current lane interface. The returned count can be smaller. The target sampler
and acceptance loop decide which proposals to keep. Checkpoint configuration
and weight contents enter paired startup and spill identity.

The lane backend submits active DFlash requests in one paired command. It packs
arbitrary physical slots into the matching one-through-eight-lane captured
graph, merges the rank-major candidate gather, and selects with each lane's own
sampling state. Zero-depth requests commit accepted rows without proposing.
The whole collection is checked before any lane commit, and held proposals are
published after the entire batch succeeds. Rank 0 logs a new maximum packed
width only when a wider graph succeeds.

`mia-mcg4` requires routed 4-bit MCG EXL3, hidden size 4096 and local expert
width 1024. Unsupported books, widths, full-EXL3 shared experts and non-EXL3
routed weights are refused. The existing shared expert fill remains in use.
Generic scalar and tensor EXL3 remain available on separate loads.

The five target dense conversion settings and output-head policy are independent
of these controls. DFlash2 retains the donor's private affine Q4 arithmetic
because its split reduction differs from the target affine4 implementation.
Mia expert FP16 rotation and half2 reconstruction have a separate numerical
identity from the generic expert path.

## Accepted state and persistence

Target forwards capture post-layer stream means at zero-based layers 5, 14, 24, 33 and 42.
Whole, shared and pieced forwards use the allocated plane stride. Split
prefills exchange rank-owned tap slices with the normalized row slices.
Each lane retains its own pending window before shared scratch is reused.
Only accepted target rows enter the drafter. A failure after target commit
terminates the rank pair rather than continuing with a stale drafter.

The package reserves eight partitioned rings at load time. Their resident
allocation is included in measured engine memory. Each lane snapshot appends
the complete five-layer K/V ring tail, 22,282,240 bytes per TP2 rank, to the
existing target payload. Saving requires target and drafter positions to agree.
Restore copies the rings before publishing the restored drafter end. Prefix
snapshots and NVMe payloads carry the new geometry and checkpoint identity.

## Captured kernels and attribution

`zig/kernels/dflash2-aot` contains the finite 24-variant subset needed by DFlash2.
`provenance.json` records the donor archive and every file SHA256. Runtime
loading checks the manifest and cubins against compiled hashes. CUDA build
outputs install these assets beside both CLI and native-server binaries.
An explicit kernel directory must contain that exact subset.

The imported arithmetic credits Ash Hart and TensorFold contributors, the Mia
integration, and the CUDA launch patterns identified in the donor headers.
Checkpoint weights retain their separate checkpoint license. No weights are
included here. The CUDA expert file preserves the donor source and appends
only the prompt slot FP32-to-BF16 conversion required by the current engine.
The [local fork addendum](../LICENSES/donors/README.md) maps the original
authors, transitive donors, exact source revisions, and local changes. The
original donor `NOTICE`, `LICENSE`, and `CREDITS.md` are retained beside it.

## Qualification boundary

Host tests exercise the loader slices, packed weight layouts, launch ABI,
variable proposal counts, lane-owned tap copies, accepted-only commits, complete
ring restore into another slot, failed-restore publication and payload identity.
CPU compilation includes the CUDA images and native serving binary.

Device numerical parity, target tap parity across split prefills, DFlash
acceptance, routed outputs, eight-lane memory, ring wrap and live NVMe resume
require an authorized GPU trial. Compilation and CPU copies do not establish
production quality or a speedup. Both imported runtime modes remain opt-in.

## v1.0.12 integration

Shared server JSON, snapshot row images, observer publication and checkpoint
math use the v1.0.12 APIs. GLM CUDA retains its encoded-image probe and rank-owner
decoder through `Info.encoded_vision`. Other families retain upstream prepared
image input through `Info.vision`. The two paths share the existing request
lifetime and server boundaries. GLM spill and Nemotron compact snapshots retain
separate family adapters over the shared tier. Imported execution modes remain
opt-in; the integration does not establish new GPU qualification.

## Native dense arithmetic

`TF_GLM_DENSE` and its five family overrides also accept `fp8_native` and
`nvfp4_native`. These explicit modes extend the shared upstream CK operators.
They currently require sm121. Existing `fp8` and `nvfp4` modes retain compressed
weights with BF16 MMA operands; the new modes use FP8 or block-scaled FP4
operands for both weights and activations. Dense defaults remain BF16, and the
output-head policy is independent.

Native FP8 converts each eligible rank-local BF16 matrix using one tensor
scale. Native NVFP4 uses one global weight scale and the format's sixteen-value
block scales. The shared CK packer owns its distinct word layout. Activations
use dynamic row scales, so another lane cannot change a row's quantization.
Zero rows use scale one, small scales have an FP32 normal floor, and nonfinite
activation rows produce NaN outputs. Nonfinite weight sources are refused.

Both modes provide BF16 outputs and unrounded FP32 tensor-parallel partials.
Execution frames own bounded quantization scratch; native modes can increase
its size. The mode and numerical version enter paired startup and spill
identity. Startup reports materialized native matrices by family on each rank.
Stored quantized checkpoint tensors retain their representation, so a requested
native policy can convert no matrices. Verify those counts before comparing it.

The model-free device check is `tf-cuda-test ck-dense`. It covers packing,
GLM output tails, sampled numerical references, strided input, row/batch
invariance, BF16/FP32 outputs and the existing scalar CK8 interface. The command
must run on the GPU before the native modes are qualified. CPU compilation and
mock launch checks do not establish output quality or performance.
