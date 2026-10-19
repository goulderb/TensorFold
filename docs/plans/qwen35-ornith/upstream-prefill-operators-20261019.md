# Bounded HF prefill operator proof

HF prefill currently calls the shared Engine once per token. The next
candidate batches at most sixteen committed prefix rows through that same
body. This receipt starts with isolated operator proofs; it does not open
the Engine's one-row admission guard or select a serving deployment.

The selected design batches row-local preparation and projections while
advancing GDN with ordered launches of its existing chain symbol. A new
persistent recurrence kernel is deferred until that control is exact.
Both the current key/value state layout and the older value/key control
remain part of the isolated proof.

The linear screen loads actual checkpoint layers 1 through 3, then layer
39 and the vocabulary head, through the existing loader. Thirteen named
projections cover FP8, BF16 and NVFP4 at one, two and sixteen rows. It uses
the existing decode path, separate row scratch, distinct synthetic inputs,
poisoned inactive output, and a serial one-row reference. Eager and captured
outputs must match before paired graph timings are collected. Those timings
describe resident-weight operators, not full-model prefill throughput.

The GDN screen loads actual layer 1 coefficients. At two and sixteen rows,
it compares serial versus batched preparation and ordered recurrence,
every intermediate row, full recurrence and convolution state, and a
continuation row. Nonzero initial state, distinct synthetic inputs, finite
output checks and allocation canaries are mandatory. It is a correctness
screen and does not claim a GDN speedup.

Both screens attest graph function identity, ordinary launch geometry,
every ABI argument and direct dependencies. The linear screen cannot
inspect cluster launch attributes through the current API; its journal
explicitly records that limitation. Raw reference and candidate readbacks
are retained so a separate CPU audit can compare bytes and finiteness.

Source inspection corrected one review assumption: HF prompt shapes are
captured when capture is enabled. Only MLX prompt shapes automatically take
the eager branch. Integration must distinguish exact row count and
diagnostic all-row heads in graph identity, and verify repeated chain
bindings. It must also preserve the pipeline's commit-after-both-stages
and poison-on-partial-failure behavior.

Device evidence is pending. The independent full-logit HF quality gate
continues to fail. This operator proof cannot establish production quality,
capacity, serving behavior, or an overall scaling wall. The retained
small-candidate batch remains queued after the larger prefill work.

The existing matched default-policy run bounds the payoff: across five
captured 64-prompt/128-output turns, median prefill is 0.4923853 s
(0.4901837–0.4978404), median turn is 1.6165476 s
(1.6117714–1.623679), and median paired prefill share is 30.524%.
A 5% saving against that median turn requires 80.82738 ms, or about
16.4155% of median prefill. This is a prioritization bound from source
5bbca7643, not a measured batching gain or a claim about HTTP performance.
