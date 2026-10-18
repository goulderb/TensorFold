# Test Qwen3.5 / Qwen3.6 35B ModelOpt on native CUDA

This local fork adds a serial CUDA text path for the official Ornith 1.5 35B
and NVIDIA Qwen3.6 35B mixed ModelOpt layouts. CPU loader tests and compilation
do not qualify model output. Run device tests only in an approved window.

The first path has BF16 KV, FP32 GDN state, top-eight routed SwiGLU and a
separate sigmoid-gated shared expert. FP8 projections use their stored static
activation scales. NVFP4 projections use BF16 activations and their stored
weight scales. A two-device layer split fits the checkpoint on the internal
pair. It is not tensor parallel execution. Prompts currently run one token
at a time. This path has no throughput qualification, vision, MTP, prefix
reuse, NVMe spill or production deployment.

## Build and inspect without a device run

Use pinned Zig 0.17.0, CUDA 12.8 and MSVC 14.44.35207. The Windows helper
selects the compiler environment only for its child build process.

```powershell
./tools/zig/build_qwen35_windows.ps1 -Target test-qwen35-cuda-host -WithoutKernels
./tools/zig/build_qwen35_windows.ps1 -Target native
./tools/zig/build_qwen35_windows.ps1 -Target qwen35-probe
./zig-out/bin/tf-qwen35-probe.exe --inspect C:/Development/local-ai-server/models/ornith15-35b-nvfp4-94e431d9
```

The inspection command maps the actual checkpoint, validates every text
tensor and rejects unexplained text entries. It creates no CUDA context.
The broader Windows host suite has inherited failures documented in the
task's qualification record. The targeted Qwen gate remains separate.

## Checkpoints inspected

The first target checkpoint is
`ornith-ai/Ornith-1.5-35B-A3B-NVFP4` at
`94e431d9cc47fa1986a7a1a4e9a80f7f118b03aa`. Its three shards contain
23,407,457,492 tensor bytes. Its attention and GDN projections are FP8 with
scalar weight and input scales. Experts, shared experts and the output head
use NVFP4 with E4M3 block scales and scalar `weight_scale_2`. These NVFP4
projections omit `input_scale`, which is valid for W4A16.

`nvidia/Qwen3.6-35B-A3B-NVFP4` at
`1355db6a052410cfd62085d94b58866fd0f2c3c5` uses the same text geometry and
mixed ModelOpt layout. Its NVFP4 entries also carry optional input scales.
Its generic NVFP4 group declares static four-bit activations, but every
per-layer quantization tag says `W4A16_NVFP4`. The loader treats those explicit
per-layer tags as authoritative. It validates and consumes the optional
input-scale metadata, then runs BF16 activations for these projections.
Actual `NVFP4` activation tags and dynamic FP4 groups are refused.
Both complete text headers pass the same loader validation rules. NVIDIA's
card identifies ModelOpt 0.44.0 and CNN/DailyMail plus NVIDIA's
Nemotron-Post-Training-Dataset-v2 as calibration sources.

`unsloth/Qwen3.6-35B-A3B-NVFP4` at
`739af1e7aac320af1682ed1e0cce369af4c5265d` and its `NVFP4-Fast` variant at
`1c3f884bc99aac2524f6d49bcbac8c88401afd66` use compressed-tensors.
Their FP8 linears have BF16 per-output-channel scales and dynamic token
activation scaling. NVFP4 weights use `weight_packed`, `weight_scale`,
`weight_global_scale` and `input_global_scale`. Their tensor bytes total
26,473,821,704 and 23,631,656,648 respectively. The normal export retains
more BF16 weights. Unsloth's cards identify an Unsloth/UltraChat calibration
mixture. Their published benchmark results are producer claims; this port
has not measured their accuracy or speed.

The captured metadata for all four exports lives in
`zig/src/families/qwen3_5/fixtures/`. Each source receipt records immutable
revision and header hashes. Reproduce it with `tools/zig/qwen35_headers.py`.
The current runtime explicitly refuses compressed-tensors instead of
interpreting its scales as ModelOpt scalars. Unsloth execution needs a
separate reader and arithmetic qualification.

## Device probe in an approved window

Write an input file containing token ids from the pinned checkpoint's own
tokenizer and chat template. Teacher forcing uses `forced`; later rows use
greedy selection.

```json
{
  "prompt": [760, 6511, 314, 9338, 369],
  "forced": [11751, 13],
  "max_tokens": 2,
  "context": 4096,
  "devices": [0, 1],
  "expected_device_uuids": [
    "GPU-f2c0ea1e-a2be-f0b4-6c23-ffe33b537a06",
    "GPU-dd2a988c-fe38-24ab-233a-9e9bb3b2645a"
  ],
  "cut": 20
}
```

```powershell
./zig-out/bin/tf-qwen35-probe.exe MODEL INPUT.json OUTPUT_DIR
```

The executable writes FP32 views of each next-token logit row and a result
receipt. It checks repeated-prompt determinism, independent session logits,
interleaved continuation and reset. Every allocation checks the 1536 MiB
free-memory floor on its device. A fixed cut of twenty layers is the initial
placement, not a measured optimum.

The output's session checks do not prove teacher parity. Compare its rows
and generated ids to the same quantized checkpoint in a trusted runtime.
Pay particular attention to FP8 activation quantization, Qwen's raw
zero-centered norms, convolution BF16 boundaries, GDN normalization epsilon
and attention softmax rounding. The baseline uses online FP32 attention.
It also retains FP32 routing weights and routed-expert sums. HF eager casts
the routing weights to BF16 and accumulates expert outputs in BF16.

Before the outage, capture a CPU reference with the pinned HF equations.
`qwen35_reference.py` loads that exact source by SHA256 and uses the installed
Transformers 5.16.1 integration. It disables hub kernels and unwraps the
optional FLA/causal-convolution wrappers to use the source's PyTorch fallbacks.
It decodes only the selected experts on demand and never exposes CUDA devices.
It is an HF eager arithmetic reference, not an independently qualified
ModelOpt serving runtime. FP8 inputs are statically quantized; NVFP4 stays
weight-only. Dequantized weights are BF16.

```powershell
$py='C:/Development/local-ai-server/runtime/glimmer-capture-venv/Scripts/python.exe'
& $py tools/zig/qwen35_reference.py capture MODEL INPUT.json REFERENCE_DIR
./tools/zig/build_qwen35_windows.ps1 -Target qwen35-kernels-probe
```

Reference capture requires explicit forced ids for every row. The saved
short Ornith case uses prompt ids `[760,6511,314,9338,369]` for
`The capital of France is` and forced ids `[11751,13]` for ` Paris.` from
the pinned checkpoint's tokenizer. Full model parity and longer cases remain
unqualified. No tolerance has been tuned using device output.

In the approved window, run the kernel checks on each internal GPU before
loading the model. The executable reports fixed per-array tolerances and
requires every value to be finite. This command does initialize CUDA.
Pin the two internal physical GPUs in PCI order. Ordinals zero and one then
refer only to that pair. The decision eGPU remains outside the visible set.
The fleet launcher already applies these same UUID pins to the server.
Both probes refuse missing or ambiguous environment pins before initializing
CUDA. They compare the expected UUIDs to the driver's physical identities
before creating a context or loading the model, and report UUIDs and names.

```powershell
$env:CUDA_DEVICE_ORDER='PCI_BUS_ID'
$env:CUDA_VISIBLE_DEVICES='GPU-f2c0ea1e-a2be-f0b4-6c23-ffe33b537a06,GPU-dd2a988c-fe38-24ab-233a-9e9bb3b2645a'
./zig-out/bin/tf-qwen35-kernels-probe.exe --approved-device 0 --expected-uuid GPU-f2c0ea1e-a2be-f0b4-6c23-ffe33b537a06
./zig-out/bin/tf-qwen35-kernels-probe.exe --approved-device 1 --expected-uuid GPU-dd2a988c-fe38-24ab-233a-9e9bb3b2645a
./zig-out/bin/tf-qwen35-probe.exe MODEL INPUT.json ACTUAL_DIR
& $py tools/zig/qwen35_reference.py compare ACTUAL_DIR REFERENCE_DIR comparison.json --max-rmse 0.05 --max-abs 0.5 --max-kl 0.001
```

The model comparison requires matching config/index hashes, prompt and forced
streams, row counts and vocabulary. It reports RMSE, maximum absolute logit
error, reference-to-candidate KL divergence, top-ten overlap and top-one ids.
The initial diagnostic limits above allow BF16 reduction drift but require
identical top-one ids on every row. They are proposed qualification limits,
not an established accuracy guarantee. A failure requires investigation;
do not raise a limit simply to make this candidate pass.

The native server uses the same engine:

```powershell
$env:TENSORFOLD_SPLIT='1'
./zig-out/native/bin/tensorfold-native.exe serve MODEL --host 127.0.0.1 --port 11439 --name ornith-test --parallel 1 --context 4096 --no-drafts
```

Unset `TENSORFOLD_TP` for this layer-split path. It queues requests behind one
active request, checks cancellation between tokens and resets state before
each request. Stop the test listener when qualification ends. Restore the
production recipe and confirm a proxy completion before ending an outage.

## Provenance

The Qwen Team and HuggingFace team's Transformers equations are pinned to
`e0eb2810ac1b2caf6d082f247d8d37162ecf67a0`. See
`LICENSES/donors/README.md` for retained notices and local adaptations.
The design and decisions are in `docs/plans/qwen35-ornith/`.
