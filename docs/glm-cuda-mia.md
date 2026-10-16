# Mia DFlash2 and routed EXL3 in the GLM CUDA engine

The GLM task branch imports DFlash2 and the fused MCG4 expert arithmetic from
[Mia's zig-engine branch](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/tree/f1ee79a6bf5319775079412c727f5f6b0ebb1e19).
The pinned recipe commit is `f1ee79a6bf5319775079412c727f5f6b0ebb1e19`.
Its seven native patches, applied to upstream v1.0.7, are preserved at
`import/mia-zig-f1ee79a6`, commit `f013e1aec57dbee167f43cccb66186da5bacc37f`.
The prior GLM candidate is preserved at `archive/glm53-before-mia-20261016`.

This imports numerical components into the existing `glm` family. The current
Engine, rank link, target sampler, lane acceptance, vision and persistent store
remain the execution path. The shared v1.0.12 trunk has separate source pins.

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

`mia-mcg4` requires routed 4-bit MCG EXL3, hidden size4096 and local expert
width1024. Unsupported books, widths, full-EXL3 shared experts and non-EXL3
routed weights are refused. The existing shared expert fill remains in use.
Generic scalar and tensor EXL3 remain available on separate loads.

The five target dense conversion settings and output-head policy are independent
of these controls. DFlash2 retains the donor's private affine Q4 arithmetic
because its split reduction differs from the target affine4 implementation.
Mia expert FP16 rotation and half2 reconstruction have a separate numerical
identity from the generic expert path.

## Accepted state and persistence

Target forwards capture post-layer stream means at zero-based layers5,14,24,33 and42.
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

`zig/kernels/dflash2-aot` contains the finite24-variant subset needed by DFlash2.
`provenance.json` records the donor archive and every file SHA256. Runtime
loading checks the manifest and cubins against compiled hashes. CUDA build
outputs install these assets beside both CLI and native-server binaries.
An explicit kernel directory must contain that exact subset.

The imported arithmetic credits Ash Hart and TensorFold contributors, the Mia
integration, and the CUDA launch patterns identified in the donor headers.
Checkpoint weights retain their separate checkpoint license. No weights are
included here. The CUDA expert file preserves the donor source and appends
only the prompt slot FP32-to-BF16 conversion required by the current engine.

## Qualification boundary

Host tests exercise the loader slices, packed weight layouts, launch ABI,
variable proposal counts, lane-owned tap copies, accepted-only commits, complete
ring restore into another slot, failed-restore publication and payload identity.
CPU compilation includes the CUDA images and native serving binary.

Device numerical parity, target tap parity across split prefills, DFlash
acceptance, routed outputs, eight-lane memory, ring wrap and live NVMe resume
require an authorized GPU trial. Compilation and CPU copies do not establish
production quality or a speedup. Both imported runtime modes remain opt-in.
