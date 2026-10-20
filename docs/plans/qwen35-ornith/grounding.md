# Qwen3.5 / Qwen3.6 35B CUDA preparation

The requested outcome is native TensorFold inference of Ornith 1.5 35B-A3B in
NVFP4, using the existing local fork. Production remains running during
preparation. Model qualification requires a separately approved outage.

## Source and checkpoint

- Task branch `task/qwen35-ornith-nvfp4` starts at trunk `a92010d6e`.
- Official checkpoint `ornith-ai/Ornith-1.5-35B-A3B-NVFP4` revision
  `94e431d9cc47fa1986a7a1a4e9a80f7f118b03aa`.
- The checkpoint has three shards totaling 23,407,457,492 tensor bytes.
- `qwen3_5_moe` uses 40 layers, hidden 2048, 30 GDN layers and 10 full
  attention layers, 256 routed experts with top 8 and a shared expert.
- Expert/shared widths are 512. GDN has 16 key heads and 32 value heads,
  each dimension 128. Full attention has 16 query heads, 2 KV heads and
  dimension 256. Vocabulary is 248320. Context is 262144.
- ModelOpt MIXED_PRECISION stores attention projections in FP8 and
  experts/shared/head in W4A16_NVFP4. Unquantized gates/norms/embeddings
  remain floating. NVFP4 input_scale tensors are absent.

## Existing architecture

The generic Qwen loader (`zig/src/families/qwen3_5/loader.zig`) parses the
family geometry, opens a normalized tensor Table, then reads host projections
and routed experts. The complete HIP backend uploads them and runs GDN or
full attention followed by the MoE. Its projections use `core.quant` with
only MLX affine and dense formats. Its routed-expert reader expects MLX
switch_mlp stacks. This loader cannot ingest the official mixed export.

The separate older Metal Qwen3.5-2B implementation is pinned to dense 2B
geometry. The Qwen27 Metal implementation is also unsuitable as a CUDA MoE
engine. CUDA registers Nemotron, GLM and LFM2, with no Qwen family yet.

CUDA already has GDN tree/replay kernels from the old Python engine,
ModelOpt checkpoint dense FP8/NVFP4 readers, lane matmuls, NVFP4 expert
packing and kernels, routing primitives, norms and attention. These are
operators, not an assembled Qwen engine. Nemotron's CUDA family is a useful
pattern for checkpoint ownership, backend state and native serving.

## Proof required

1. A real immutable checkpoint schema fixture passes geometry and tensor
   validation, including both quantized formats and unsanitized HF names.
2. Targeted CPU tests verify scale math, shape refusal, routing, rotary,
   GDN and persistent state contracts with failures on the previous source.
3. The native Windows CUDA target and all added fatbins build with pinned
   Zig 0.17.0 and CUDA 12.8 for sm_120 without touching production.
4. A dedicated probe compares teacher-forced logits and greedy continuation
   to a trusted implementation on the same quantized checkpoint.
5. Device qualification exercises a real completion, concurrent isolation,
   cancel/reuse and long prefill with the measured per-GPU floor.

Steps 4 and 5 need device capacity. A CPU pass or native build does not
establish that the model is ready to serve. Vision and speculative decoding
are outside the first text inference milestone.
