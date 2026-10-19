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
| Reuse FP8 activation quantization for compatible QKV/Z pairs and Q/K/V triples | Exact changing activations and outputs in the isolated screen. Projected 9.9724 ms saved per 191-append turn, 0.620% of the historical 1.6088229 s N8 turn. | Keep for the first compatible batch. | Add a narrow shared qlinear entry that preserves each projection's scale and GEMM order. Require identical input-scale bits. Revalidate after larger kernel/prefill changes. [Evidence](upstream-quant-transport-20261019.md). |
| Elide final norm/head/copy for intermediate prompt positions whose logits are unused | Historical complete head boundary: 0.4349 ms/call, range 0.4338–0.4361. The old 275/27 workload projected 119 ms / 1.21%; this is not a current shared-Engine measurement. | Retain; refresh the current boundary before batching. | Likely overlaps bounded batched prefill's last-row head. Do not count both. Prove all persistent state, terminal prompt logits and subsequent decode outputs; all-row diagnostic mode remains available. [Evidence](device-tuning-20261019.md). |
| Device greedy argmax / avoid the full BF16 download on ordinary greedy turns | Current transport micro measured 43 us for one full logit download. The prior sampler draw boundary was about 0.310 ms; that includes host work and is a historical estimate, not a current device-argmax saving. | Retain as a candidate requiring an isolated operator measurement. | Use shared upstream sampling where it fits. Preserve tie, nonfinite and token semantics and independently compare every full row outside timing. This changes the measured host boundary, so report the new scope explicitly. |
| Slice-major N8 code-word packing | Strided scalar addresses verified; no measured saving. Four slice warps already share a CTA/L1. No counter evidence supports a fourfold physical-traffic claim. | Deferred optional combination after split-M. | Separate format affects prompt and decode consumers. First prove exact operand mapping in isolation. Split-M may alter the relevant latency and load pressure; the two benefits may overlap. Retain Fable/Kimi findings and root corrections. |
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

The combined trial remains pending larger changes. It does not replace the
independent HF quality, serving, capacity or promotion gates.
