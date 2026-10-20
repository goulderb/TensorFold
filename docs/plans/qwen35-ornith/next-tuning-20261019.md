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
First-token time includes the first sampler draw, which is part of this A/B.
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
by this preparation. Build/test results and the review disposition appear
below; neither probe is currently device-qualified.

## Frozen CPU preparation

Experiment source is `312d4f5d950a2c95cd2028e9136c742928841523`.
Controller source is fleet commit `1669e9f5`. The source, fleet task tree and
default fleet checkout were tracked-clean at the freeze. The complete output
of the clean-tree commands, source identity, binary and input hashes, compiler
versions and command-to-log map are in
[`preparation.json`](evidence/next-tuning-20261019/preparation.json).
Later documentation commits do not change the frozen executable source.

Both embedded Windows executable builds passed 57/57 build steps. A fresh
CPU test cache reran the six sampler loop/admission tests, all passing.
The frozen head executable passed all ten input-validation cases. Five
additional frozen CLI checks verified both embedded identities, both valid
inputs and the sampler's malformed-input failure receipt, with CUDA
visibility empty and an intentionally invalid CUDA ordering setting. The
Debug-engine head build is refused at compilation as intended. These are
CPU checks, not device tests. The controller's 60 targeted mocked tests and
the fleet fast gate passed.

The default controller preparation against the real pins passed. It checked
the executable/input identities and every unchanged checkpoint/runtime pin;
its receipt says `preflight_valid: true` and `outage_started: false`. No
scheduled task, candidate CUDA initialization or model execution occurred.

The local executable and input freeze is
`.artifacts/qwen35-next-tuning-20261019/frozen-312d4f5d/` in the TensorFold
task tree. Its `plan.json` is the concrete controller input. Executables stay
local; the durable evidence stores their exact SHA256 hashes. CUDA images are
embedded and covered by each executable hash. The controller rechecks them
and their CPU `--identity` before a window.

[`raw-preparation.zip`](evidence/next-tuning-20261019/raw-preparation.zip)
retains 69 verified members, including build/test logs, freeze script,
controller receipts and preflight. Its
[`inventory`](evidence/next-tuning-20261019/preparation-inventory.json)
maps every member to its size and SHA256 and maps verification commands to
their logs. The existing `head-fixtures.zip` retains all 12 BF16 arrays.
To reconstruct the head-check runner's layout, extract those arrays into
`head-fixtures-full/` beside the raw archive's `head-implementation/` folder.
The old worker reports inside the raw archive describe their intermediate
builds. The final freeze and final logs supersede their executable hashes.

Fable's [second review](evidence/next-tuning-20261019/review-fable-r2.md)
accepts requesting a window with no blocker. Its context concern is closed
by `Session.reset` and `Session.append` selecting each partition's context
themselves. The post-window analysis must count 24 turn records and 12
verified-pair records, including warmups, before reporting sampler medians.
There should be 20 measured turn records and ten measured pairs across the
two cases. A completion record alone is not a performance analysis.

The 02:43 UTC read-only snapshot still found Nemotron PID 16512 on its exact
preserved executable and Clef PID 21392 unchanged. The internal cards had
5050/5293 MiB free. Host CPU was 24.2% during compilation, so that snapshot
does not characterize noise during a future trial. Record host load again
at the approved experiment, alternate sides, inspect work/error counts and
report median and spread. The earlier exact proxy READY receipt is retained;
its absent provider header is not an independent attribution claim.

No model promotion, capacity sizing or full fleet promotion gate is claimed.
The full-model numerical gate remains failed. A fresh 20-minute workstation
window is the remaining execution gate for these experiments. Its controller
permits at most 600 seconds of work with 500 seconds reserved for bounded
restoration, preserving Clef and the Spark pair. A recoverable completed head
failure still allows the independent sampler trial; safety-monitor and
timeout failures end the experiments. The whole window remains failed if
either experiment fails, even when Nemotron restoration succeeds.
