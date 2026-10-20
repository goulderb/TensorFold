# Current default policy device verification

The shared Engine defaults were verified on merged source
`5bbca7643f4e06a8dd2c6a5a1594e1bafd1da253`, including trunk `b5fac6aee`.
The frozen executable SHA256 is
`08d3b07bdea6d6fd81b982f486ed72b4b5c225b135b1b6aca8763ce3e7243def`.
This closes the current-default device-evidence gap in Fable's N8/quant R2
review; it does not qualify production or establish another causal speedup.

The input enables graph verification without setting the expert or state-layout
policies. Initialized HF engines select N8 and new sessions select `key_value`.
Both ranks verify 20 actual N8 expert nodes and 15 actual GDN candidate nodes,
with zero baseline bindings. Every eager/captured consumed BF16 logit row,
greedy token and persistent state matches, and session controls pass. The
consumed-logit SHA256 remains
`b78bb49d36d6abcc509679be4673442594072d65f53b5f0998e1474ce0582839`.

Five measured pairs follow one alternating warmup pair. Captured decode takes
8.860772 ms per append (range 8.824613–8.865704), about 112.86 tok/s.
The 64-token prompt plus 128-token output turn takes median 1.616548 s
(range 1.611771–1.623679). This includes host argmax and logit downloads;
it is a C=1 Engine profile, not HTTP throughput. Earlier N8 A/B acceptance
remains the matched evidence for its 6.61% turn-time saving.

The first attempt at source `64a5cb5d` passed output controls but failed the
strict binding proof: Chain.open had already cached a graph before validators
were enabled. The diagnostic driver now warms eagerly for this mode, then
enables validators before first capture. The retry's first eager warmup
correctly has no cached graph. The analyzer originally required graph nodes
for that eager turn; its check now requires nodes for captured turns and a
valid nonnegative cached count for eager turns. Twenty analyzer tests pass,
including zero-before-capture acceptance and zero-captured-node rejection.
Neither correction changes model arithmetic or runtime policy.

Nemotron was restored at 20:00:37 UTC, PID 17784, canonical READY request
`20b6ccda-6301-4584-93ea-c1cc14f23a96`. Clef was unchanged; owned jobs exited,
cleanup and sampler errors are empty, and both internal GPU floors passed.
The independent HF full-logit quality gate remains FAIL.

Original attempts, builds, raw journals, strict analysis and tests are retained
in [the evidence archive](evidence/upstream-default-policy-20261019/archive.json).
