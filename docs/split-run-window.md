# Windows layer-split qualification status

This branch implements an experimental contiguous layer split. It is not tensor parallel TP2 and is not ready for promotion.

## Reproduced serving failure

The owned trial script reproduced HTTP 500 after prefill produced one token. The first decode step failed with CUDA illegal address 700. Its receipt is `.audit/runs/baseline-b-20261014/`. The script stopped its own server and retained its exit status.

The earlier `.audit/runs/baseline-20261014/` receipt failed before startup because the native CLI rejects an explicit `--segments` argument. It is not a decode trial.

## Corrected serving trial

On 2026-10-14, the corrected candidate passed four short HTTP requests through
`fleet.launch nemotron35-tf-windows-qual --foreground`. The local receipt is
`.audit/runs/explicit-phases-20261014/`. The executable SHA256 was
`d5dfbe0ff11a328b636afa3b29542d81dc2c2050aaa13dd028beef323628e1b2`.

The counting request returned all integers from 1 to 20. It generated 70 tokens
with 69 continuing decode rounds. Repeated math and counting responses matched
with cache disabled. This is serving and reset evidence, not independent
numerical parity or full generated-token-ID equality.

At 0.5-second sampling intervals, the minimum free memory was 6142 MiB on the
RTX 5080 and 7358 MiB on the RTX 5070 Ti. Short requests and sampled readings do
not qualify sustained load or exclude transient peaks. The driver stopped its
owned launcher and candidate. Exit code 1 reflects forced cleanup, not a tested
graceful native shutdown.

## Request-covered rerun

The stricter local trial `.audit/runs/request-covered-20261014/` passed six
HTTP requests with binary SHA256
`9a4f42b4c1b7674ef9acfd2990e6b5986090e82ccbe0d1daeabb778ed26f0403`.
Two prompts contained 2345 tokens and crossed the 2048-row prefill boundary.
All repeated response pairs matched with cache off. Each request had 6 to 17
samples per internal GPU from the persistent 50 ms sampler. No sampler errors
were reported. Sampled minima were again 6142 and 7358 MiB.

This corrects the older trial's coverage gap. Its 0.5-second sampler had not
sampled every request window and could miss a dead reader. Neither trial proves
continuous peak memory, sustained load or independent numerical parity.
The stricter driver checks the recipe's executable/checkpoint pin and labels
the owned PID as the fleet launcher. Both launcher and sampler were stopped.

## Baseline contract

`TENSORFOLD_SPLIT=1` selects the layer-split experiment. The experiment accepts one stream, no drafts, no prompt cache, no learned states, and one serial prompt segment. It rejects unsupported combinations rather than silently copying rank 0 alone.

Rank 0 owns the embedding and the vocabulary head. Each rank owns a contiguous block range and its own recurrent and attention state. The sender consumes the deferred block output with the receiving block's norm. The handoff carries the updated residual, normalized rows, and their FP32 group sums. The peer runs its blocks and final norm. It returns normalized rows and group sums to rank 0 for the head and sampling.

The current hardware selection matches two devices by compute capability. On this workstation they are CUDA ordinal 1 for RTX 5080 and ordinal 2 for RTX 5070 Ti. This heuristic is not a production hardware-identity contract.

## Evidence limits

The earlier four-seam manual check measured internal consistency under its then-current source and head-on-peer topology. It did not call the native HTTP serving path or use an independent numerical reference. Changes to the topology or normalization invalidate its relevance until it is rerun.

Old measured weight totals of 8.64 and 8.69 GiB used head-on-peer placement. They do not qualify the current layout. A process-global allocation total is not a per-device memory measurement.

The observed production VRAM readings changed after a request. The exact cause of the residency change has not been independently established. Idle free-memory readings do not prove safe coexistence with a loaded trial.

## Remaining promotion gates

- Genuine per-layer tensor shards and collectives on both internal GPUs.
- Independent same-checkpoint logits and generated token IDs through prefill and continued decode.
- Complete paired lifecycle for every enabled feature, including concurrency and prefix state reuse.
- Per-device admission with packing, state, scratch, and cache budgets.
- At least 1536 MiB measured free on each internal GPU under every qualified load.
- Cancellation, failed startup, teardown, and restart evidence.
- Linux native and single-device regression evidence. Prior object-only cross-compiles are insufficient.
- Sustained performance, fleet verification, pinned operational recipe, and a real proxy completion.

Production remains the existing llama.cpp deployment. Only an approved outage permits GPU trials. Spark GLM has not been stopped for a reference comparison. Every stopped production engine must be restored and verified through a proxy completion after the approved outage work.
