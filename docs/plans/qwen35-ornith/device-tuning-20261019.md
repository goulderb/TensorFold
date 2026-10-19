# Greedy sampling wins the direct turn trial; head elision fails the payoff screen

The once-loaded Ornith 1.5 35B NVFP4 trial reduced the complete synthetic
64-input/128-output greedy turn from 8.412 to 6.164 seconds, a 26.7% latency
reduction. All five measured pairs improved and produced identical tokens.
This is a direct single-session engine result at temperature zero, not an
HTTP, concurrent-serving or model-quality qualification.

The second case improved by 4.6%, below the predeclared 5% materiality gate.
Keep the already implemented shared greedy scan. Do not implement head elision:
its isolated measurement suggests only about 1.2% of the prefill-heavy turn.
That fraction is a scenario calculation, not an integrated speedup or bound.

## Matched complete-turn results

Each case has one excluded warmup pair and five alternating measured pairs.
Reset is measured separately and excluded. TTFT includes the first draw.
Decode rate uses N-1 outputs after the first token, not all N outputs.

| Prompt/output tokens | Original median turn, range | Scan median turn, range | Less latency | Original/scan median decode |
| --- | --- | --- | --- | --- |
| 64/128 | 8.412 s, 8.372–8.464 | 6.164 s, 6.150–6.169 | 26.7% | 19.88/30.61 tok/s |
| 275/27 | 10.295 s, 10.280–10.300 | 9.820 s, 9.810–9.828 | 4.6% | 19.29/29.29 tok/s |

The paired latency reductions span 26.37–27.16% and 4.48–4.74%, respectively.
The raw-row parity pass compared 128 and 27 consumed BF16 rows. Every timed
output matched its untimed reference, including after every reset. The
analyzer verifies 24 turn records, 12 verified pairs, both parity records,
alternating order and P+N-1 append/N-draw counts before reporting medians.
These counts include four warmup turns and two warmup pairs.

The model, state reset, one-token append pattern, compiled engine and kernels
are shared between arms. The complete frozen pre-scan sampler from `ad247b3d`
is the control; the current arm calls the real shared sampler. Both sampler
modules and the engine are Fast; control/widening code is Safe. Source is
`312d4f5d950a2c95cd2028e9136c742928841523`, a two-GPU layer split at layer 20,
context 4096 and one session. This is not tensor parallelism.

## Attribution and the rejected head candidate

For the 64/128 case, TTFT minus prefill measures the first draw at median
17.494 ms with the original sampler and 0.310 ms with the scan. Multiplying
that difference by 128 gives a 2.200-second estimate, close to the observed
2.248-second complete-turn saving. This extrapolation explains the scale;
it is not a profile of every draw. Median prefill was 2.012/2.014 seconds,
so it did not improve. Current decode still takes about 32.7 ms per next
token, dominated by the serial model path rather than the small greedy draw.

Four synthetic residual/norm/head fixture rows passed the fixed CPU reference
tolerances, head top-ID agreement and exact repeat checks. This qualifies only
those isolated cases. Full-model numerical comparisons still fail.

The resident residual through final RMS, NVFP4 head, synchronization and full
BF16 download measured median 0.4349 ms/call. The range of 20 measured block
means was 0.4338–0.4361 ms/call, with 32 calls per block. Projection-only CUDA
event time was 0.3465 ms/call and omits RMS/download. Checks and serialization
were outside these timers. Empty event overhead was recorded, not subtracted.

Multiplying 0.4349 ms by the 274 non-final prompt positions gives about
119 ms, or 1.21% of the 275/27 current turn. Repeated isolated calls do not
reproduce the integrated cache/scheduling environment. The measurement does
not earn head-elision implementation under the 5% screen. Investigate the
serial prefill/model path next, with a cheap profile before changing kernels.

Coarse telemetry includes 81 samples with an active child. Host CPU was
median 12.0%, range 7.5–23.6%, on 24 logical CPUs. Other host load was present;
alternation and the gap relative to run variation support the measured sampler
gain. These counters do not identify a particular kernel bottleneck.

## Approved execution and restored production

The user approved the fresh 20-minute window ending 12:49:24 UTC. Controller
execution began 12:30:25 UTC. It completed both probes and verified restored
production at 12:35:27.656 UTC, roughly five minutes later. It used the existing
controller at fleet source `1669e9f5`; no service configuration changed.

All 257 physical GPU samples passed. Minimum NVML free memory was 5358/6105 MiB
on the internal RTX 5080/RTX 5070 Ti. The stricter in-context readings in sampler
turns were 4607/5353 MiB, also above 1536 MiB on each card. The receipt contains
no child error, safety-monitor failure, cleanup error or deadline overrun.

Nemotron returned a fresh exact `READY` through the canonical proxy with paid
fallback disabled. Its direct model list and preserved executable matched;
the restored PID is 14348. Clef remained PID 21392. Spark and the proxies were
not restarted. The public provider header was absent, so it is not independent
provider attestation. Restoration uses the combined direct identity, local-only
route policy and exact fresh completion evidence.

## Reproducible evidence and limits

[`analysis.json`](evidence/device-window-20261019T1230/analysis.json) stores the
full medians, ranges, paired results, head screen and restoration receipt.
[`raw-evidence.zip`](evidence/device-window-20261019T1230/raw-evidence.zip)
retains 26 individually hashed members. Its
[`inventory`](evidence/device-window-20261019T1230/inventory.json) pins every
member and the archive. The fleet manifest is
`fleet/trials/ornith15-qwen35-sampler-head-20261019.json` on the fleet task branch.

`tools/zig/analyze_qwen35_tuning.py` rejects an incomplete/unrestored window.
That rejection was observed before restoration; the completed receipt passed.
CPU checks also reject eight corruptions of record counts, work, token output,
pair order, parity, driver median, head sample count and reference errors.
There was no additional device run to generate these negative cases.

No model was promoted, no new production capacity was sized and no full fleet
promotion gate ran. Optimized HTTP behavior, concurrency, cache, long context,
soak/restart and quality remain unqualified. The original full-model numerical
thresholds remain unchanged. Additional device work requires its own approved
scope/window after this completed trial.
