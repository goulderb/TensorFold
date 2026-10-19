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
