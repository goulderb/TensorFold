# Opt-in bounded HF prefill through the shared Engine

The isolated [operator proof](upstream-prefill-operators-20261019.md) passed.
This implementation carries that scheduling into the existing Engine body,
router and pipeline. It introduces no CUDA kernel or separate model body.
Full-model device verification remains pending at this checkpoint.

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

The planned device gate uses the existing 64-prompt/128-output fixture:

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
from that run. The source is ready for a frozen-build device trial; CPU
checks do not establish full-model parity or a performance improvement.

The independent HF full-logit gate remains FAIL. No production selection,
capacity qualification or promotion follows from this opt-in implementation.
