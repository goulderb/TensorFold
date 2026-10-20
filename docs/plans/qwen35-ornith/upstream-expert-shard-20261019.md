# Expert work partition screen

The four-versus-eight routed expert screen passes operator correctness but
fails the material-payoff screen. Halving expert work barely reduces elapsed
kernel time. The homogeneous-layer projection saves only 0.113205 ms per
append, about 1.344% of the reference turn, before transport. This does not
justify an expert-parallel engine port and does not establish a hardware wall.

The frozen source is `68f8e9506a5d0483727f0634924979bc27948160`; executable
SHA256 `44ac61b934034c9da37aec924717a5fd7e58b55141e4320f99d103fabb8632de`.
The fixture manifest SHA256 is
`c41ab1d6363d21f785f0e8d17f8a8030f041dd614aacf823169a3f9787fc366e`.

Each comparison uses production N8 up or unchanged baseline down on one GPU.
The first and last four experts retain their original members and output
rows. The actual count is four, with 64 blocks; full eight uses 128 blocks.
Every active output matches its full-work reference exactly, every inactive
row remains poisoned, and the saved bank-zero control matches. Thirty-two
rotating banks cover all 256 real experts and exceed L2 capacity. The helper
checks all 128 graph nodes, geometry, production function and fourteen ABI
arguments. The analyzer independently reconstructs plan-buffer hashes and
checks argument-digest coverage; raw pointer values are not in JSON, so
argument validation relies on the source-pinned executable's checks.

The medians below use the smaller full-eight reference across comparisons
and the slower first/last half on each GPU. Each comparison has one
alternating warmup pair and five measured alternating pairs. The archived
analysis retains all event and wall-time samples and ranges.

| Projection | GPU | Full eight | Four experts |
|---|---|---:|---:|
| N8 up | RTX 5080 | 37.6507 us | 36.5165 us |
| N8 up | RTX 5070 Ti | 37.7865 us | 36.1672 us |
| Down | RTX 5080 | 11.6038 us | 10.0508 us |
| Down | RTX 5070 Ti | 11.7607 us | 10.0543 us |

For each projection the conditional comparison is
`20 * (full_gpu0 + full_gpu1)` versus `40 * max(half_gpu0, half_gpu1)`.
It assumes all layers resemble this fixture and simultaneous ranks behave
like isolated ranks. Neither simultaneous execution nor reductions were
measured here. Five percent of the 1.6088229-second, 191-append reference turn
requires 0.421158 ms saved per append. The 0.113205 ms compute-only projection
misses that threshold. Charging forty reductions at half the measured
eighty-operation CPU-chain time (2.5887 ms) gives a conditional cost of
1.29435 ms/append and negative net saving. That forty-operation cost is a
projection, not an actual EP benchmark or general TP rejection.

The strict audit verifies the exact 400-record roster, source/executable/input,
fixture and all 93 image pins, output controls, timing summaries, GPU floors
and restoration. Six test methods include 37 journal mutations, 11 controller
mutations, five byte-corruption cases, three malformed JSON cases and
independent projection arithmetic.

Nemotron was restored at 20:07:55 UTC, PID 31312, canonical READY
`6b3fb2e5-2ffc-4e99-a878-a84f20ab93b2`. Clef was unchanged, errors were empty,
and controller/probe processes ended. An initial launch refused the output
directory already created by CPU preflight before any outage; the successful
run uses a fresh `device-window`. Both receipts are retained in
[the archive](evidence/upstream-expert-shard-20261019/archive.json).

Next candidates must address the kernel's elapsed-time floor or expose more
useful independent work. No engine architecture or production pins changed.
