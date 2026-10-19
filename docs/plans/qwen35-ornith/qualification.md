# Qwen35 CUDA qualification

The Ornith 1.5 35B NVFP4 port runs through the native TensorFold server at
the tested one-lane, 4096-token configuration. Bounded functional checks
passed. Full-logit comparison against the corrected Hugging Face reference
fails the unchanged diagnostic limits. This is an experimental testing path;
numerical parity and production qualification remain open.

The [2026-10-19 performance investigation](gpu-investigation-20261019.md)
reaches 81.668 decode tok/s on a synthetic 64/128 greedy c1 turn with exact
eager/graph self-parity. Its graph path remains diagnostic, and its router is
off by default. These results do not change the independent reference gate or
qualify the server, long context, concurrent occupancy, sizing or promotion.

The user approved stopping workstation Nemotron for this trial. It stopped
at 2026-10-18 01:52:49 UTC. The trial was stopped and the existing production
task restarted at 02:22:22 UTC. At 02:23:22 UTC the production proxy returned
`READY` from `nemotron-3.5-lightning`, with eight lanes and paid spillover
disabled. The production executable is still `tensorfold-native-8047d1a0917e`.
Clef's process remained running. No deployment, component, scheduled task,
registry, or proxy selection was changed.

## Source and artifacts

- Fork store `C:/Development/tensorfold.git`, worktree
  `C:/Development/tf/qwen35-ornith-nvfp4`, branch `task/qwen35-ornith-nvfp4`,
  based on trunk `a92010d6e`.
- Initial port `8f6a04efe`, reviewed device-admission checkpoint `343486999`.
- Windows host fixes `b91d70d3`; kernel-oracle correction and compiled source
  checkpoint `bc1773e2`.
- Zig 0.17.0, CUDA 12.8, MSVC 14.44.35207 and embedded `sm_120` kernels.
  The compiler helper changes only its child environment.
- Current immutable runtime directory
  `C:/Development/local-ai-server/runtime/tensorfold-qwen35-native-4d370dcedc46`.

| Executable | Bytes | SHA256 |
|---|---:|---|
| `tensorfold-native.exe` | 22504960 | `4d370dcedc464ccc12e3328209652f9043402c81060a0d66066ffa0f479de40a` |
| `tf-qwen35-probe.exe` | 4670464 | `1b4c77d033fa2af308e42c5fc6ee3f24e5e3e5290d1395d3ba6a88f8645bf687` |
| `tf-qwen35-kernels-probe.exe` | 4613632 | `b43548d3a0b32198df12581b869c86bcf9364e7ddc75b709e6b9b9d45c58fd5c` |

The final three-target build passed 14/14 steps using the retained prebuilt
fatbins. Their source paths, sizes and SHA256 digests are recorded in
`fatbins-sm120-manifest.json`. The source CUDA math is unchanged from the
initial device checkpoint. Both immutable bundles retain the original root
license, notices and byte-verified archive of all nineteen `LICENSES/` paths.
`SOURCE_COMMIT` records the full compiled checkpoint. Every runtime file and
all seventeen checkpoint files are pinned in the inactive fleet recipe.

Unless another root is stated, run evidence below is under
`.artifacts/qualification-20261018-0151/`. Initial preparation and three Claude
Opus 5.5 review rounds remain in Git history at `82ae266a6` and in
`.artifacts/review-r1.md`, `review-r2.md`, and `review-r3.md`.

## Checkpoint and CPU checks

The model is `ornith-ai/Ornith-1.5-35B-A3B-NVFP4` at revision
`94e431d9cc47fa1986a7a1a4e9a80f7f118b03aa`. All three shard hashes and byte
counts match immutable Hugging Face LFS metadata. Both Ornith and NVIDIA
ModelOpt header fixtures pass the text schema. The NVIDIA Qwen3.6 checkpoint
has not been run on a GPU. Unsloth and Unsloth Fast compressed-tensors exports
were inspected and are explicitly refused by this loader.

The text model has forty layers, thirty GDN and ten full-attention, hidden
width 2048, 256 experts and top-eight routing. It uses static FP8 projections
and NVFP4 W4A16 experts, shared MLP and output head. The tested split is twenty
layers on each internal GPU. Packed text weights total 20,850,944,512 bytes;
that is a layout calculation, not physical VRAM usage.

The original fresh Qwen host gate passed all 39 tests and eight build steps.
The final Windows aggregate passed 43/43 steps, 1147 tests and 84 explicit
skips. The subsequent Qwen target passed eight steps with 36 reported tests
and one cached admission run. See [Windows host qualification](windows-host-qualification.md)
for skip reasons, remaining Zig networking diagnostics and Linux coverage
limits. The fleet fast gate passed before commits. The initial launcher,
documentation and attribution suite passed 228 tests, including 67 launcher
tests; process creation was mocked for its actual-recipe CLI case.

## Kernel and session checks

Both kernel probes verify the physical device UUID and name before allocating
a CUDA context. The tested devices are the RTX 5080
`GPU-f2c0ea1e-a2be-f0b4-6c23-ffe33b537a06` and RTX 5070 Ti
`GPU-dd2a988c-fe38-24ab-233a-9e9bb3b2645a`. The final clean probe passed all
56 array comparisons on each, covering ten Qwen entry points and reused RMS
normalization. See `kernels-clean-device0.stdout` and
`kernels-clean-device1.stdout`.

The first probe failed the chained GDN output at step two. CPU and GPU FP32
recurrence outputs differed by about 2.8e-9 at a BF16 midpoint, producing
adjacent BF16 inputs to normalization. The correction compares normalization
on identical, already independently checked device inputs. Recurrence state
and output checks retain their original tolerances. No tolerance or CUDA math
was changed. The original failure and same-input diagnostic remain in
`kernels-device0.stderr` and `kernels-same-input-device0.stderr`.

Both complete-model cases passed exact reset and interleaved two-session
isolation checks. The final staged probe reran both cases successfully.
All four final logit rows are byte-identical to their earlier corresponding
rows; see `rebuilt-logits-identical.json` and `model-final-*/result.json`.
The committed short and chat input fixtures force the same token stream
used by the reference. Forced tokens themselves are not an accuracy check.

## Reference correction and remaining numerical failure

The reference uses the pinned Transformers source at
`e0eb2810ac1b2caf6d082f247d8d37162ecf67a0`, Transformers 5.16.1 and Torch
2.9.0 in the existing CPU capture environment. It hides CUDA and uses the
original PyTorch fallback functions. It is an HF arithmetic reference, not
an independently qualified ModelOpt serving runtime.

Layer captures exposed premature BF16 rounding in the first reference
adapter. The corrected adapter computes quantized dot products in FP32,
applies tensor scales after reduction, and returns BF16 projection outputs.
Static FP8 quantization multiplies by the FP32 reciprocal before E4M3
conversion. Lazy `F.linear` operands preserve the pinned HF expert forwards.
Eight tests pass, including regressions that failed before the correction.
An actual pinned-HF expert-forward CPU probe also passes. The capture receipts
record script SHA256
`afbe5ec943d2d7551e5b8158625d1cdaa00239ea9f022103c997098b6cb49474`
and adapter `modelopt-linear-fp32-postscale-bf16-v1`.

Fifteen same-input first-layer projections match the corrected arithmetic
exactly, in `projection-comparison-v3.log`. This does not establish whole-model
parity. The native engine uses FP32 attention probabilities, routing weights
and expert accumulation where HF eager has BF16 boundaries. A diagnostic
reference variant covering those differences also fails the final-logit
limits. Layer traces show tiny early differences growing across the network;
the remaining discrepancy has not been isolated sufficiently to classify it
as harmless rounding. Temporary engine instrumentation was removed, with its
patch, executable and identities retained in the run directory.

| Case and row | RMSE | Maximum absolute error | Reference-to-native KL | Argmax equal |
|---|---:|---:|---:|---|
| Short 0 | 0.098534 | 0.546875 | 0.005401 | yes |
| Short 1 | 0.154493 | 1.059082 | 0.001756 | yes |
| Chat 0 | 0.498408 | 2.062500 | 0.001874 | yes |
| Chat 1 | 0.366464 | 2.031250 | 0.000096 | yes |

The unchanged limits are RMSE 0.05, maximum absolute error 0.5, KL 0.001,
and identical argmax for every row. All four rows fail. See
`comparison-corrected-short.json` and `comparison-corrected-chat.json`.
The initial failed comparisons remain retained separately. Matching the
four top predictions does not turn this numerical gate green.

## Bounded serving evidence

The inactive recipe was launched through `python -m fleet.launch` from
`C:/Development/local-ai-server/.worktrees/task/ornith35-tf-trial`.
It served `ornith-1.5-35b-test` only on loopback port 11439 with one lane,
4096-token context, BF16 KV and FP32 GDN state.

The completed run contains all ten expected checks: raw completion, thinking
off and on, two queued requests, streaming disconnect, subsequent reuse,
HTTP 400 context refusal, reuse after refusal and an 1835-token prefill that
recovered `ORCHID742`. Both BFCL smoke fixtures passed; this is two fixtures,
not a broad model-quality benchmark. The separate engine log records the
disconnected request ending with `reason=client-left` after one generated
token of a requested 1024, followed by successful reuse.

Review found that the temporary serving script could report success after
an early exception. The script now requires normal completion and the exact
expected check set. `serving-assessment.json` independently checks the original
complete ten-case receipt and early-stop log line, recording their hashes.
No GPU rerun is implied by this assessment. The rebuilt final server also
passed a separate real greeting request through the fleet launcher.

One-second `nvidia-smi` samples during serving had minima of 5358/6105 MiB
free. The final model-probe samples had minima of 5288/6035 MiB. These observed
workloads stayed above the 1536 MiB floor on both GPUs. CUDA context memory
accounting is recorded separately. Under Windows WDDM, `cuMemGetInfo` alone
does not establish physical free-memory admission under arbitrary contention;
neither these samples nor the runtime guards prove a continuous physical
minimum for other workloads.

## Verdict and limits

A fresh independent GPT-6 Astra review supports controlled experimental use
at the tested configuration, with the numerical gate still failing. It found
no new implementation defect in the reference adapter or GDN component
oracle. The Windows fix review is separate. No merge, production promotion,
public route, or performance claim follows from these results.

The next correctness gate is diagnosis of the remaining numerical discrepancy.
Production admission under Windows contention, full-window occupancy,
longer-context quality, throughput, vision, MTP, prefix caching, spill and
tensor parallel execution are unqualified or unsupported on this path.

Restoration evidence is `restoration-requested.json`,
`production-restored-proxy.json`, and `production-restored-processes.json`.
The task is running the original Nemotron runtime with eight 128K lanes;
the Ornith listener is absent. The initial preparation bundle remains
immutable at `tensorfold-qwen35-native-247110a921c0`.
