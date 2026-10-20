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

The final combined CPU build passed 24 steps and 137 fresh tests. Its direct
CUDA host target ran all six new named split-M and atomic FP8 tests plus
seven anonymous import blocks. A named rerun confirmed their discovery.
Importing the CUDA dependency from the earlier family/screen root did not
execute those core tests; the earlier 124-test draft build is not evidence
that they ran. All 22 CLI admission cases passed.

The frozen trial uses source a1240bb262d7430d979cb46cfb5559feceb2efc7,
profile SHA256 83ecb48000dd4dded201965b7cde26fda4beef00e441df346daedd3f39c1b927,
input SHA256 9f76a1e214355ecb7d66e857869350dc6992a25e239d1bc7e51a882ea32f9d2d,
and the unchanged 93 CUDA images. Every prefix, continuation, complete-turn,
session isolation, partial-stage failure and reset control passed. The
historical native b78 consumed-logit and 7fb canonical-state anchors remain
exact. Timed work and graph caches remain fixed as specified above.

Five alternating measured pairs produced these medians:

| Engine phase | Bundle off | Bundle on |
|---|---:|---:|
| 64-token prefill | 0.101862 s | 0.1074846 s |
| 127 subsequent appends | 1.1218116 s | 1.071332 s |
| Complete 64/128 turn | 1.2233779 s | 1.1787787 s |

Paired turn latency saving is 3.62974% median, range 3.45196–4.52056%,
with all five pairs positive. Complete-turn ranges are 1.2213057–1.2353404 s
and 1.1769755–1.1811474 s. Decode throughput is approximately 113.2 versus
118.5 subsequent tokens/s at this host-greedy boundary. Prefill regresses
by about 5.6 ms while decode saves about 50.5 ms. Retain the measured net
gain and the phase regression; no ablation has yet attributed each
member's contribution. A row-dependent split-M policy is a future candidate,
not a demonstrated fix. Historical micro percentages are not added.

The independent audit passed 104 tests with zero skips and checks 685
ordered records, 154 raw files totaling
1,488,993,791 bytes and 400 actual expert nodes. Each rank has ten graph
keys, 1170 GDN nodes, forty N8 nodes, 160 split-M nodes, 130 baseline quant
nodes and 320 reused quant nodes. Source-enforced graph attestations verify
all fourteen expert arguments. Actual row-one grids are 128 for both
kernels; prefill16 uses N8 grids 252/210 versus split-M 128/128. N8 uses
146 registers and no static shared memory; split-M uses 112 registers and
2048 static shared bytes. Neither reports local memory. The metadata audit
independently reads 311 checkpoint projections and confirms all forty
reuse groups have equal input-scale bits. BF16 a/b and NVFP4 shared/head
projections are not included in the FP8 reuse count.

The timing includes host BF16 greedy scans and consumed-logit copies. It
excludes HTTP, tokenization, queueing and reset. Raw canonical anchors
independently cover reference and warmup full turns plus prefix and
continuation anchors. Other measured state comparisons remain enforced by
the pinned executable and retained hashes. This is an exact native
comparison, not an independent numerical-quality acceptance.

Nemotron was restored as PID 31200 at 21:39:05.485872 UTC and canonical
request edcccf38-2b04-43cd-b0a2-b578b2612c9c returned READY. Clef PID 21392
was unchanged. Minimum sampled free memory was 5220/5867 MiB. Controller
2752 and probe 27200 ended; the receipt reports no cleanup or sampler
failures. Fable found no source blocker; its checkpoint-format assumption
was resolved by the actual metadata/scalar audit. Cluster attributes remain
outside this proof.

The [evidence archive](evidence/upstream-retained-bundle-20261019/README.txt)
retains 131 members, 9,362,393 compressed bytes, SHA256
bb6f2e49373288647148bdece019a8421aca9834ed3e2acb5aa3f971e5722b2b.
The 138 larger raw readbacks remain external, totaling 1,477,074,431 bytes,
with their original paths and hashes retained. The strict audit verifies
2565 historical source objects and all 93 images. Runtime executables,
images and checkpoint files remain separately pinned external inputs.

The independent HF full-logit quality gate remains FAIL. This opt-in
experiment is neither serving qualification nor production promotion.
