# Opt-in bounded HF prefill through the shared Engine

The isolated [operator proof](upstream-prefill-operators-20261019.md) passed.
This implementation carries that scheduling into the existing Engine body,
router and pipeline. It introduces no CUDA kernel or separate model body.
The bounded full-model device and matched-turn gates passed. The serving
and independent HF quality gates remain separate.

`Options.prefill_rows` defaults to one and accepts one through sixteen.
The pipeline enables `Engine.bounded_hf_prefill` only above one row. The
ordinary default retains one-row admission and terminal-only logits. HF
scratch can hold sixteen rows; actual loaded projection workspace bounds
are checked against the existing partial allocation. Expanded scratch is
included in memory-budget calculations before a pipeline loads; workspace
checks use the actual projections after loading their weights.

The existing decode projections keep their row arithmetic. GDN preparation
and convolution commit run over a chunk; the current recurrence kernel
advances ordered immutable row views. Diagnostic graph checks attest the
actual uploaded tables, expected eleven arguments, function, geometry and
layer/row ordering. The scalar HF router retains compact eight-pick rows,
nine-weight rows and 257-score rows. The existing grouped experts and N8
decode kernels consume the resulting plan.

`Keep.prompt_logits` exposes all rows of an unrecorded committed prefix.
It has a separate graph-key bit and does not reuse snapshot semantics.
`Session.appendLogitRows` validates output size and the whole token request
before mutation, then uses the same pipeline step as ordinary append. Both
partitions finish and pass floor checks before either cache length commits.
A partial-stage failure poisons the session until reset.

The passed device gate uses the existing 64-prompt/128-output fixture:

- All prompt logits, canonical recurrence/conv/KV state, metadata and three
  forced continuation rows at prefix lengths 1, 2, 3, 15, 16, 17, 31, 32
  and 64, from positions zero and seven, eager and captured.
- Distinct interleaved sessions, a failed sixteen-row second stage,
  unchanged host commit metadata, poison refusal and complete reset.
- The accepted one-row logit and canonical-state hashes as a historical
  anchor, so agreement between two changed arms cannot hide a shared drift.
- Untimed qualification of all normal and diagnostic graphs, followed by
  one warmup and five alternating measured turns. Table validators are off
  during timing; graph-cache counts must remain unchanged. Logical work is
  191 appends in either arm; actual body calls are 191 versus 131 per rank,
  with 382 versus 262 graph launches across the two ranks.

The combined CPU build passed 21 steps and 118 tests. A second build of the
final harness passed, with one fresh test and the other host checks cached
from that run.

The frozen run used source `040baadcffc558ff00a35cdf589c2f2820df3f22`,
profile SHA256 `b06fc185bdb8ff430281983ceb141991886fa280dff31ba12b4e074c08e1cbfc`,
and the same 93 pinned CUDA images. The independent audit passed 59 tests
and checked 281 exact-roster journal records, 154 raw files totaling
1,488,993,791 bytes, 2,565 historical source objects and all image/build/input
pins. Both ranks verified eight graph shapes, 915 current GDN chain nodes
and 160 N8 up nodes, with zero baseline bindings. Timed cache counts did
not change. The original one-row `b78...` consumed-logit and `7fb...`
canonical-state anchors remain exact.

Five alternating measured pairs produced these medians:

| Engine phase | Serial1 | Batch16 |
|---|---:|---:|
| 64-token prefill | 0.491047 s | 0.1017207 s |
| 127 subsequent appends | 1.121438 s | 1.121458 s |
| Complete 64/128 turn | 1.6118556 s | 1.2233359 s |

Paired turn latency saving is 24.1039% median, range 23.8455–24.2778%.
Serial prefill range is 0.4904176–0.4913478 s; candidate range is
0.1015779–0.1021051 s. Decode is unchanged within variation. The gain
comes from amortizing prompt operators and reducing body/seam/head calls;
the measured 79% prefill reduction fits its approximately 30.5% share of
the prior turn. This is concurrency-one engine timing with host greedy and
full-row logit copies, excluding HTTP, tokenization, queueing and reset.

All prompt/continuation raw rows are independently finite and byte-exact.
Raw canonical anchors cover recurrence, convolution tails, valid KV, cache
metadata and terminal logits. Measured-turn comparisons are enforced by
the pinned executable and recorded as hashes; independent raw full-turn
checks cover its reference and warmup arms. Nonzero-position prefix cases
append seven initial tokens and then the first R tokens again, deliberately
testing positions rather than representing a contiguous real prompt.

Nemotron was restored as PID 16124 and returned canonical `READY` request
`bcc0be7d-87ee-476c-995e-c02d26a48300` at 21:12:40.542232 UTC. Clef PID
21392 was unchanged. The controller recorded no cleanup or sampler errors;
the minimum sampled free memory was 5222/5867 MiB. Controller 17856 and
probe 31752 ended. The archive and external-readback inventory are under
[upstream-bounded-prefill evidence](evidence/upstream-bounded-prefill-20261019/README.txt).

Fable 5.1 found no blocking source defect. Its warning that attention/KV
had no isolated multirow proof was resolved by the full-prefix checks.
The admitted checkpoint's linear head passed actual workspace validation;
this does not add support for arbitrary tied-head layouts.

The independent HF full-logit gate remains FAIL. No production selection,
capacity qualification or promotion follows from this opt-in implementation.
