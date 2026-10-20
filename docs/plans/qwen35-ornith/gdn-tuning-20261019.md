# GDN memory-access experiment

The accepted router/up graph result is 107.144 tok/s. This separate experiment
tests whether cooperative staging improves the strided GDN state accesses while
keeping persistent FP32 state layout and ordered scalar arithmetic unchanged.
The candidate has 32 blocks of 128 threads, 5632 shared bytes and 97 barriers
per block. Barrier overhead may outweigh fewer global-memory transactions.

The [CPU build provenance](evidence/gdn-tuning-20261019/preparation/tiled-fatbin-manifest.json)
records two identical builds, unchanged original kernel bodies and 50 unchanged
other fatbins. The optional symbol is diagnostic only. No serving default changes.

Sixteen changing-input correctness steps compare all state/output bytes from
nonzero finite patterned initial state, poisoning output before every step.
Timed graphs rotate 64 independent 2 MiB states over 128 nodes: two updates per
bank and 128 MiB of unique recurrent state. This working set exceeds L2 size;
it does not establish actual DRAM traffic or physical bandwidth utilization.
The original one-bank CPU harness was revised before any GPU execution because
repeated updates of a 2 MiB state would fit cache and hide the intended effect.

Each timed arm resets all banks outside its CUDA event. All final state/output
bytes must match an untimed baseline, with poison and altered-state controls.
One excluded warmup pair precedes five alternating measured pairs per GPU.
Actual free memory must exceed 1536 MiB throughout the device screen.

Full-turn payoff is projected from 191 appends and 15 GDN layers per partition:
191 * 15 * (saving_gpu0_us + saving_gpu1_us) / 1e6 seconds. A repeatable saving
above 5% of the 1.711066-second optimized turn earns a model trial. This is
approximately 14.93 microseconds per call averaged over the two GPUs. A screen
alone cannot establish model throughput or production correctness.

## Result: reject shared staging

The [strict analyzer](evidence/gdn-tuning-20261019/r1/analysis.json) accepts the
complete pinned trial at e60ac0c0ddbc80929fdb95a76439d6728f7340d0. Every
measured pair is slower on both GPUs. Original/tiled median microseconds per
call are 37.917/38.698 on GPU0 and 38.0415/38.9325 on GPU1. Original ranges
are 37.885-37.925 and 38.0055-38.1125; tiled ranges are 38.628-38.713 and
38.9207-39.0192. Median paired savings are negative: -0.7865/-0.89425 us.

All 32 changing-input correctness steps and all 24 timed final bank payloads
pass finite, bit-exact equality. Both poison and altered-state/output controls
pass. Actual minimum in-process free memory is 14721/14773 MiB. The projection
is 4.82 ms added per turn (0.28%), and fails the 5% full-trial gate. No model
trial or production performance claim follows. Improved access grouping did
not overcome this implementation's staging/barrier cost; no profiler counters
were collected to apportion the loss.

The candidate kernel, profile mode and harness are reverted from the active
source. Commit e60ac0c0d retains the exact reproducer and build provenance;
the accepted router/up implementation remains unchanged. The independent
analyzer and corruption tests retain the evidence contract.

The controller completed and all owned processes ended. Nemotron was restored
at 16:52:19 UTC, canonical local completion READY with request
5d7f4cf8-83ef-4ea8-abd6-6f5420dc36d4, production PID2792; Clef stayed unchanged.

Persistent state transposition remains a distinct hypothesis, with migration
and snapshot costs to justify first. For production, independent reference
numerics and live server integration remain ahead of further kernel tuning.
