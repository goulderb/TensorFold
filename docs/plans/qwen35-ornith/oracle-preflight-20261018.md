# Independent ModelOpt reference preparation

The preserved package is `.artifacts/oracle-preflight-20261018/`. Its report,
hash inventory, source snapshots and verification scripts separate static
compatibility from execution. No serving model, GPU runtime, container image,
or production service was started for this work.

## Serving reference candidate

The package pins vLLM 0.28.0 at
`2cf0a6915ce544dc493a0990f2ea38d81601128a`, its official recipe at
`2338e9edec1d5df5f823abf3c1f19e7ba0b2ed0e`, and NVIDIA ModelOpt 0.45.0 at
`ec87a82927d003986d44fb7f4fa8b3d10c31b095`. The Linux amd64 image is
`vllm/vllm-openai@sha256:2286e8533ca8b6bc777594bae30524f1426ba46ca21797524e06df6a94b06635`.
Its OCI revision matches the inspected vLLM source. Image metadata was read;
the 8.6 GB of compressed layers were not pulled.

Both original and remapped module prefix probes pass 231/231 checks for the
exact Ornith revision `94e431d9cc47fa1986a7a1a4e9a80f7f118b03aa`.
Saved source supports this mixed static FP8 and W4A16 NVFP4 layout. Local
shard-header hashes match the preserved checkpoint receipt. These checks do
not construct the loader or execute the model.

vLLM's fusion changes the original per-projection FP8 weight scales in 39
groups: all ten full-attention Q/K/V groups and 29 of 30 GDN QKV/Z groups.
Their largest selected/original scale ratios are 2.2307692 and 1.79807693.
Those ratios describe scale choices, not measured error. The fused input
scales are equal. All inspected shared and routed gate/up NVFP4 global-scale
pairs are equal. The focused scalar inspection reads 82,960 payload bytes;
it does not read packed weights or block scales.

Therefore vLLM is an independent serving comparison candidate, but its
full-model logits would not by themselves be a bitwise oracle for arithmetic
that preserves each original FP8 projection's scale. This finding does not
explain or close TensorFold's failed HF logit comparisons.

Prepared offline launch and raw-logit capture scripts remain unexecuted. No
approved Linux GPU host is available. The workstation only has a stopped
Docker Desktop WSL distribution, and its internal GPUs serve production.
Neither heterogeneous TP=2 execution nor single-GPU capacity is qualified.

## Executed CPU components

The existing `runtime/glimmer-capture-venv/Scripts/python.exe` ran narrow,
hash-checked AST extracts of NVIDIA's original `NVFP4QTensor.dequantize`
and `_fp8_eager`. CUDA was hidden before Torch import. All arrays were on CPU;
no ModelOpt package or Transformers model implementation was imported. No
dependencies were installed. NVIDIA's original source and licenses are
preserved in the artifact package; no donor implementation is added to the
fork by this documentation.

For layer zero, expert zero, gate weights, the original NVIDIA function
produced 1,048,576 finite FP32 values. The 4,194,304-byte array has SHA256
`21ad8beb0490c967a8b8b86e2285b7fb6b141088f5c2d8ba6f31ff4cbd438ac0`.
The selected original tensor payloads total 589,832 bytes. Source, input and
output hashes and exact commands are in
`cpu-components-execution-20261018/execution-receipt.json` and its report.

The separate `cpu-comparison-20261018/REPORT.md` compares that original
NVIDIA function with the actual corrected `Store.linear` from reference
adapter SHA256
`afbe5ec943d2d7551e5b8158625d1cdaa00239ea9f022103c997098b6cb49474`.
For three identical BF16 input rows, all 1,536 BF16 projection outputs match
bitwise. Both paths use the same CPU PyTorch matmul, so this checks decoding
and scale placement, not independent matmul kernels. Seven FP32 and seven
BF16-rounded FP8 input cases also match bitwise for the selected stored scale.

Standalone weight representations differ by signed zero and multiplication
order. With global scale one, decoded weights are numerically equal, with
304,702 signed-zero bit differences. Expanded FP32 weights with the original
global scale differ numerically at 41,842 entries, maximum absolute error
3.725290298e-9. Returning final BF16 weights before matmul instead yields
15,503 differing weight values and 165 differing BF16 outputs; that diagnostic
policy is not the corrected `Store.linear` path. Keep these policies explicit.

Execution and artifact validation pass. These bounded CPU comparisons do not
establish CUDA projections, attention, GDN, routing, model quality, or whole
model equivalence. The four failed full-logit rows and their fixed limits
remain unchanged. Independent serving execution and numerical qualification
remain open.
