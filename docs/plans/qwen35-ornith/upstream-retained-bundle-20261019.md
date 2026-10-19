# Retained split-M and FP8 activation reuse bundle

The user requested measuring compatible sub-5% candidates together after
larger changes. Qualified [bounded prefill](upstream-bounded-prefill-20261019.md)
is the new control. The first bundle combines the retained split-M expert
up kernel with identical-scale FP8 activation reuse. No sum of their old
isolated projections is treated as a measured full-model gain.

`Engine.expert_up_split_m` and `Engine.reuse_fp8_quant` default to false.
Both extend the existing shared body and have separate graph-key bits.
Split-M requires the HF ModelOpt path and N8 scheduling. Its optional
binding tolerates only a missing symbol; other CUDA errors propagate.
Selection without the binding fails loudly. The existing frozen symbol
uses 256 threads and `min(items*nb,resident,128)` blocks. Ordinary N8 keeps
its existing resident-based grid, which can exceed the old 128-block
microbenchmark control. The full trial must measure that actual comparison.

`Linear.decodeTogether` owns one encoded input through an ordered group
of consumers. Admission checks every format, K, positive finite exact
activation-scale bits, row/workspace compatibility and nonaliasing spans.
An incompatible group executes the original independent decodes in order.
Only GDN qkv/z and attention q/k/v are grouped. Each consumer keeps its
original weight, scaling, output and split-K arithmetic. No CUDA kernel or
checkpoint changes are introduced by this integration.

Qualification checks actual captured expert functions, launch geometry and
fourteen argument byte spans, and the metadata-derived ordered quant,
GEMM and reducer sequence. The diagnostic journal retains actual expert
nodes and driver-reported resources outside timing. Cluster launch
attributes remain explicitly unverified. These graph attestations are
source-enforced checks, distinct from independent raw-output analysis.

The bounded trial compares both flags off versus both on with prefill16
on both timed arms: 131 body calls per rank and 262 graph launches for the
64-prompt/128-output fixture. It retains the old one-row native historical
anchor, every prefix and continuation row, complete canonical state,
session isolation and partial-stage failure/reset controls. Every graph
shape is qualified before one warmup and five alternating measured pairs;
validators are disabled and cache counts fixed during timing.

The narrower direct CUDA host target runs the core split-M and atomic FP8
tests. Importing the CUDA dependency from the family/screen root did not
execute those tests; the earlier 124-test draft build is not evidence that
they ran. Final CPU and device receipts will be appended after execution.

The independent HF full-logit quality gate remains FAIL. This opt-in
experiment is neither serving qualification nor production promotion.
