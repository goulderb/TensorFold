# Quantization and transport screens

FP8 input-quantization reuse is rejected for integration at the current 5%
complete-turn materiality gate. All outputs and encoded activations matched,
but the rotating-eight-bank projection saves only 9.9724 ms per 191-append
turn, about 0.62% of the accepted N8 turn median of 1.6088229 seconds.

Thirty QKV/Z pairs and ten Q/K/V triples share exactly equal activation-scale
bits in this checkpoint. Reuse removes fifty quantization calls per append.
The synthetic screen compares the existing qlinear decode with quantization
once and the same ordered FP8 matrix kernels. It covers 128 changing input
rows, two payloads, eight weight banks exceeding L2, actual graph functions,
arguments, dependencies and all output bytes. One warmup precedes five
alternating measured pairs per GPU, case and cache scope.

| Rotating case saving | RTX 5080 median | RTX 5070 Ti median |
| --- | --- | --- |
| Pair | 1.0235 us | 1.0267 us |
| Triple | 2.1785 us | 2.1130 us |

These are group savings including the real GEMMs, not the duration of an
isolated quantization kernel. The projection is
`191*(15*pair_gpu0+5*triple_gpu0+15*pair_gpu1+5*triple_gpu1)`.
The journal retains its historical GDN denominator of 1.7232142 seconds;
the analyzer separately uses the accepted N8 baseline for the decision.
No runtime quantization-reuse path is added.

## Physical transport

Both CUDA peer-capability queries returned zero. This run leaves policy and
driver settings unchanged. Host-coordinated timings use private contexts,
changing data, all-byte comparison, one excluded warmup and five measured
samples. They include context binding and synchronization.

| Operation | Median | Measured range |
| --- | --- | --- |
| Forward boundary, both 4096-byte residual buffers | 46.421 us | 45.566 to 46.544 us |
| 80 dependent pinned transfers and GPU sums | 5.4781 ms | 5.1991 to 5.5486 ms |
| 80 dependent host sums with download and upload | 2.5887 ms | 2.5068 to 2.6492 ms |
| Final 496640-byte BF16 logit download on stage 1 | 43.0 us | 27.3 to 49.3 us |
| Two context memory-info queries | 1.5 us | 1.5 to 1.7 us |

Eighty reductions assumes two per layer across forty layers. Shard computation,
asymmetric cards, smaller expert grids, routing and overlap remain unmeasured.
These timings challenge a simple layerwise parallel rewrite; they cannot reject
every parallel layout. The present one-boundary coordinator is a small fraction
of the roughly 8.80 ms inclusive N8 decode append. No physical bandwidth
utilization is inferred from logical copy bytes or these wall times.

The source and executable are the same frozen pins as the N8 full trial.
Controller and probe exited. Nemotron PID 30468 was restored and canonical
request `0c306c57-7e0d-484e-a730-1fa6ec619867` returned `READY` at
19:24:24 UTC. Clef was unchanged. The controller reported no cleanup or sampler
errors. The independent HF full-logit quality gate remains failed.

The first analyzer version incorrectly required different reduced-output
hashes on the two ranks. The helper computes an all-reduce, so both ranks must
agree. The real journal exposed that parser error. The corrected analyzer
requires equal rank outputs and equal method outputs, plus evolving inputs
and outputs. Eighteen CPU tests include both negative controls and pass.
