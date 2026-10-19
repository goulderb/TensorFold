# Qwen tuning: measure the surviving sampler change and screen head cost

The next unit prepares two bounded probes. No new candidate device result
exists yet. Production remains on the preserved Nemotron runtime. The
earlier reference-logit failure remains a promotion blocker.

The first probe loads one checkpoint and alternates the frozen old sampler
with the current shared scan. The earlier CPU proof earns this full-model
experiment; no new production sampler flag is needed. The second probe loads
only the actual head and final norm to measure a possible prefill saving
before any head-elision implementation.

## Chosen boundaries

Sol and Astra independently compared a direct once-loaded driver with
repeated immutable HTTP binaries. Their Sol cross-judge selected Astra
(25/30 versus 23/30), with two grafts from Sol: explicitly synthetic workload
labels and an untimed pass checking every consumed BF16 logit row. The judge
also corrected reset timing and required the complete final norm/head/copy
boundary in the head-cost screen. Both designs and the synthesis are retained
under `evidence/next-tuning-20261019/`.

The generation driver mirrors the adapter's one-token prefill appends and
direct BF16-to-f64 draw. N outputs require exactly P+N-1 appends, including
the prompt and excluding an append of the final generated token. Reset is
measured separately; primary prefill, first-token, decode and total clocks
start after reset. Decode rate uses N-1 over the interval after first token.
This is a two-GPU layer split at layer 20, not tensor parallelism.

Cases contain fixed synthetic chat token arrays at 64 prompt/128 output and
275/27. Tokenizer and decoded-prompt provenance are retained. These shapes
test decode-heavy and prefill-heavy turns; they do not establish actual tool
gates, HTTP or lane-host serving performance. EOS does not shorten this
fixed-work test. Serving EOS behavior is unchanged.

An untimed paired pass must compare every consumed raw BF16 logit row and
every output ID. Then each case gets one warmup pair and five measured pairs,
with alternating order, complete output identity and repeat determinism.
Logit hashing, disk writes and result comparison stay outside turn timing.
The declared materiality gate is at least 5% less median complete 64/128
turn time, with consistent paired results and every correctness gate passing.

## Head screen and CPU fixtures

Actual staged metadata declares an NVFP4 U8 head of [248320,1024], F8 scales
of [248320,128], and scalar global scale. BF16 logits do not imply BF16
weights. `Weights.load(...,40,40,false,true)` admits and loads just the head
and norm through the existing packing path. The probe uses the real one-row
qlinear dispatch on the physical RTX5070Ti, where the full engine's head runs.

The primary measured boundary is a deterministic residual through the actual
final RMS, head, synchronization and full BF16 download. Projection-only
CUDA-event time is a separate diagnostic. A repeated isolated head result
is a scenario screen, not a strict bound or a measured integrated speedup.
Operation-count fractions do not establish elapsed-time savings. Any later
head elision needs material payoff and integrated state/final-logit parity.

`tools/zig/record_qwen35_head.py` generated four CPU-only cases: zero and
three finite nonzero residuals, their zero-centered final-RMS outputs and
their full head outputs. It reuses the current CPU `Store.linear` math;
it loads no transformer and creates no CUDA context. The archive contains
only generated input/output fixtures and provenance, not model weights.
Its 12 binary arrays have verified sizes and SHA256 hashes. BF16 tolerances
were fixed before device execution at absolute 1e-6 plus relative 1/128,
following the existing kernel probe. They are not a replacement for, or a
relaxation of, the full-model numerical gate.

## Execution gate

CPU tests, compiled probes, immutable binary/source/input pins and the
controller's default preparation must pass before requesting a device window.
The proposed controller allows 600 seconds of experiments, clipped to the
approved absolute end, with a separately reserved bounded restoration period.
It owns only its spawned PIDs, samples both physical internal GPUs, aborts
below 1536 MiB on either, and preserves Clef's process identity. A fresh
direct and proxy completion must verify the preserved Nemotron after any
approved stop. The actual canonical proxy omits `x-local-ai-provider`, so
an absent header is not treated as independent provider attribution.

The fork source contract says: “Any new outage needs fresh approval.” The
previous hour has ended. No new stop or candidate model execution is approved
by this preparation. Build/test results and the review disposition will be
appended when available; neither probe is currently device-qualified.
