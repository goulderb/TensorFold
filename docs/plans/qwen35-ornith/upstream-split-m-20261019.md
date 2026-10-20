# Split gate/up warp prototype

Split-M passes exact operator checks and is retained for the compatible
combined batch. The measured saving projects to 4.20644% of the historical
complete turn. No full-model gain or hardware limiter is established.

The candidate doubles projection warps by assigning gate and up to separate
warps in one CTA. It keeps the existing packed layout, fourteen-argument ABI,
per-projection K accumulation, scales and SwiGLU/BF16 rounding. A shared FP32
handoff adds no floating-point reduction. Original template defaults remain
unchanged, and the new symbol has no production runtime caller.

| GPU | N8 event median and range | Split-M event median and range | Paired saving median |
|---|---:|---:|---:|
| RTX 5080 | 37.68425 us, 37.63500–37.84650 | 28.71325 us, 28.68125–28.82975 | 8.953748 us |
| RTX 5070 Ti | 37.83375 us, 37.81800–37.84850 | 29.08250 us, 29.04975–29.11750 | 8.761998 us |

Each GPU has one alternating warmup pair and five measured alternating pairs,
with 128 launches rotating through 32 real expert banks per sample. All five
pairs improve. Host-wall savings agree. The conditional calculation is
`191 * 20 * (8.953748 + 8.761998) / 1e6 = 0.06767415 seconds`, or 4.20644%
of the 1.6088229-second reference. It assumes this layer represents all forty
layers. Integration cost, batching interactions and whole-model effect are
unmeasured.

All banks match the N8 outputs exactly; bank zero also matches its saved
native control. An eighteen-row control uses distinct finite input rows,
proves the first seventeen baseline outputs differ, and tests a second row
batch, grid-one traversal and skipped-expert poison. Every captured graph's
128 nodes, launch geometry and fourteen arguments are checked. The strict
analyzer independently checks actual plan hashes and argument-digest coverage;
raw pointer values are not in the journal, so argument validation also relies
on the pinned executable's checks.

The actual device launch is 128 blocks on both GPUs. Split-M uses 256 threads,
112 registers, zero local bytes and 2048 static shared bytes; CUDA reports a
two-block-per-SM resident limit. N8 uses 128 threads, 146 registers and a
three-block-per-SM limit. These limits are not achieved-occupancy counters.
The compiled image has no stack use or spills. SASS inspection was unavailable.

Frozen source: `9a3d058636e6361a6c5e9d06437603f7d2e61664`.
Executable SHA256:
`edb6be09dfee99082231b73c3f0df236187aa5ceae64f23d5ca7d264f51f1304`.
Changed CUDA image SHA256:
`f468ad2f84b9cbb05a9ac36b10680f22f7e8803631e03d846718b4942269cda8`.
All 93 originals and the other 92 copied images are unchanged. CPU checks
pass 26 host tests, 22 executable admission cases and eight analyzer test
methods with journal, identity, input, restoration and memory mutations.
The strict audit verifies all 116 records and 2283 historical source pins.

Astra designed and implemented the prototype; Fable's source review found no
blocker after the distinct-row control was strengthened. The archive also
retains the preceding Fable/Kimi coalescing reviews and root corrections:
four slices already share a CTA, and their address pattern does not prove
fourfold physical memory traffic. Source-level equivalence does not by itself
prove identical compiled instructions.

Nemotron was restored at 20:28:25 UTC, PID 27460, canonical READY request
`bb9d2b2c-3ab1-418c-9d6e-fc7f60b1c13e`. Clef was unchanged, cleanup and sampler
errors were empty, owned controller/probe/compiler jobs ended, and all GPU
memory floors passed. The independent HF full-logit gate remains FAIL.

See [the archive](evidence/upstream-split-m-20261019/archive.json) and
[small-candidate ledger](small-candidate-ledger-20261019.md). The next larger
work is bounded exact HF prefill through the shared body; split-M's behavior
on that workload must be revalidated before combination.
