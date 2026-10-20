# First shared-engine device proof

The shared Qwen engine at `ae946848757237c75ca5709dcf57660fdaf5d93f` reproduces
both retained native logit rows exactly on the internal RTX 5080 and RTX 5070
Ti. Each row contains 248320 BF16 values. The five-token prompt and two forced
continuations match the original capture. No coordinate changes in either
row. This is bounded equivalence to the old native implementation. The
independent HF comparison remains failed on its original limits.

The same process passes one warmup and five alternating eager/captured pairs,
comparing every consumed row, token and live persistent-state hash. It also
passes interleaved session isolation, reset, and an injected second-stage
failure. The failure leaves the first stage's state mutated, refuses further
appends while poisoned, and reproduces the rows after resetting both caches.
Graph work counts match two launches per append. Phase instrumentation runs
eagerly and is excluded from the paired measurements.

The run uses one lane, BF16 KV, context 4096 and cut 20. Twenty-two controller
samples observe at least 5336 MiB and 5885 MiB free on the respective internal
GPUs. In-process checks enforce the 1536 MiB floor at allocations and stage
completion. These measurements do not size full-window occupancy or sustained
service load.

The protected controller enters at 18:14:39 UTC and confirms restored Nemotron
at 18:15:45 UTC. The canonical local-only route returns exact `READY`, request
`3dd30f13-5a53-495f-8d23-25fc72826983`, from restored PID 28544. Clef retains
its prior identity. The controller and trial processes have exited, with no
cleanup or sampler failure.

The binary reports the full source SHA, engine mode `fast`, root mode `safe`
and `shared-qwen-engine`. The build records all tracked Zig source hashes
before and after compilation and confirms they are unchanged. All 92 embedded
CUDA images match their current sources, dependencies and SM120 targets.
The actual checkpoint also passes CPU inspection and memory planning.

The local evidence directory is
`.artifacts/qwen35-upstream-20261019/`. It contains `build.json`,
`source-pins.json`, `actual-plan.json`, the pinned first-proof input and plan,
the window receipt and results, and `first-proof-comparison.json`.
`compare_retained.py` verifies the preserved row hashes, input parity and
successful restoration before comparing. Its identical-row, changed-bit and
nonfinite controls pass on the CPU.

No speed claim follows from this two-row case. The next baseline uses a
64-token prompt and 128 greedy outputs, with forced continuation available
only when explicitly supplied. Its inclusive turn timing contains host
argmax and logit-row copies. Pure GPU event timing stays separate. Production
serving, larger contexts, batching, useful concurrency, soak and restart gates
remain open.
