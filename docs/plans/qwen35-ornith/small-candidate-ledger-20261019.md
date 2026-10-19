# Smaller candidates and the combined trial

On 2026-10-19 the user asked to retain sub-5% changes and try a combined batch
after the larger changes. The 5% threshold now prioritizes individual full
trials; it does not discard otherwise useful, compatible gains. Earlier
rejections remain historical decisions with their original evidence.

Each candidate below must retain its actual workload, correctness boundary,
absolute-time estimate and interactions. Percentages cannot simply be added:
the baseline will change, candidates can overlap, and integration can add
cost. Correctness failures and measured regressions remain separate from
small positive results.

| Candidate | Evidence and current bound | Batch disposition | Interactions / required work |
|---|---|---|---|
| Split gate/up work across warps | Exact 32-bank and distinct-row device checks pass. The new full bundle with quant reuse saves 3.62974% median turn latency, with native outputs/state exact. | Qualified in an opt-in shared-Engine bundle; defaults unchanged. | Individual member attribution remains open. One-row grids are 128 in both arms; larger prefill grids differ and bundle prefill regresses about 5.6 ms. [Operator](upstream-split-m-20261019.md), [bundle](upstream-retained-bundle-20261019.md). |
| Reuse FP8 activation quantization for compatible QKV/Z pairs and Q/K/V triples | Exact changing activations and outputs in the isolated screen, and full-model native parity in the split-M bundle. The measured combined saving supersedes the old 0.620% isolated projection for decisions. | Qualified in the same opt-in bundle; do not count its contribution twice. | Atomic qlinear group checks exact input-scale bits and preserves each lane's arithmetic/order. All forty checkpoint groups independently qualify. Individual contribution is unmeasured. [Operator](upstream-quant-transport-20261019.md), [bundle](upstream-retained-bundle-20261019.md). |
| Elide final norm/head/copy for intermediate prompt positions whose logits are unused | Covered by qualified bounded prefill's terminal-only chunk head. Full 64/128 turn latency fell 24.1039% as a combined prefill change; no separate head gain was isolated. | Already covered; do not add the historical projection to the small-candidate bundle. | All-row diagnostics, canonical state and subsequent decode pass. Normal prefill keeps one head per chunk. [Evidence](upstream-bounded-prefill-20261019.md). |
| Device greedy argmax / avoid the full BF16 download on ordinary greedy turns | Direct submission regressed on head GPU1. A new matched carrier graph with one submission per arm and no timed archive copy passes802records/220raws; all10pairs positive. GPU1 paired saving11.751ms/128draws, range11.5498–12.7942. | Retain for real shared-Engine integration; no measured model gain yet. | The synthetic candidate elides full-row transfer. Preserving full host logits during integration changes that boundary and requires a fresh matched measurement. Preserve nonfinite sampler fallback, ties, session ownership and state. Event spans include host gaps; no limiter attribution. [Carrier](upstream-terminal-carrier-20261019.md), [earlier direct boundary](upstream-boundary-screens-20261019.md). |
| Slice-major split-M code-word packing | Exact all32banks/rows1/2/16 controls pass. All30event and30wall pairs positive; row1 paired medians4.5976%/3.6925% operator savings. Conditional mixed-turn projection6.187710ms/0.524926%, not a measured model gain. | Retain for the compatible batch; production format integration remains unqualified. | Original/transformed allocations and tagged bindings fence legacy consumers. Every production fallback/prompt consumer needs an explicit format contract. Four warp slices still cover the same total sectors; no fourfold DRAM claim. [Evidence](upstream-slice-major-20261019.md). |
| CK FP8 pipeline depth four to three | Exact 26-bank outputs and actual graph attributes pass for rows1/2/16 on both GPUs. Five alternating pairs show no useful saving: row1 paired medians -2.712/-0.328 us per mixed pass. | Keep depth four; no positive candidate to batch. | Shared storage falls 20480 to15360 bytes without useful measured gain. This does not establish a bandwidth wall. [Evidence](upstream-ck-operator-suite-20261019.md). |
| Split routed experts across GPUs | Half-work operator screen passes; homogeneous compute-only projection 0.113205 ms/append / 1.344% of historical turn. Conditional forty-reduction charge is 1.29435 ms/append, exceeding that saving. | Recorded, but not a positive composable gain on current transport. | Needs a materially different transport/overlap architecture before reconsideration. Do not add the gross compute saving to the batch total while omitting its communication cost. [Evidence](upstream-expert-shard-20261019.md). |
| N16 / WN2 / one-warp expert alternatives | N16 previously projected a small positive gain against the old kernel, but accepted N8 replaces it and is faster. WN2 is slower than its known N16 control; one-warp down regressed. | Preserve as controls; do not stack mutually exclusive kernels or measured regressions. | Reconsider only for a distinct shape or new observation. [Evidence](upstream-expert-tiles-20261019.md). |

After the larger changes settle:

1. Freeze a new matched baseline, workload and source/executable/image pins.
2. Refresh cheap seams and choose compatible candidates with passing output
   checks. Use absolute saved time; document overlaps and added costs.
3. Implement them behind explicit diagnostic switches on the shared Engine,
   retaining an all-off control. Keep arithmetic and state contracts intact.
4. Run one unprofiled full-model all-off/all-on comparison with one warmup
   pair and at least five alternating measured pairs. Include the 64/128 turn
   and a prefill-heavy case so prompt-only effects are visible.
5. Verify actual dispatch, every consumed output, canonical state, session
   controls, GPU floors and restoration. If the bundle regresses or drifts,
   ablate its members to identify the interaction. Report measured total
   benefit rather than the sum of isolated projections.

Bounded shared prefill is now qualified against the unchanged native
baseline. The first compatible batch, split-M plus identical-scale FP8
reuse with prefill_rows=16 in both arms, passed the complete native gate.
Five paired turns save 3.62974% median latency; decode improves while
prefill regresses. Keep the measured net gain, retain the phase regression
as a future ablation target, and do not add the old isolated projections.
Slice-major packing remains separately unmeasured on the live model. The
direct device greedy path was rejected on head GPU1, but the new matched
carrier graph is positive. Retain it for actual-body integration and a new
matched measurement; its earlier CPU-only estimate is not a model result.
The combined trial does not replace
independent HF quality, serving, capacity or promotion gates.
