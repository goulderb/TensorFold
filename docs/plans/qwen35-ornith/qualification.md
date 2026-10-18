# Qwen35 CUDA preparation evidence

This is preparation for an Ornith 1.5 35B NVFP4 trial. No Qwen kernel or
model has run on a GPU in this task. Numerical parity, GPU memory headroom,
serving behavior and performance remain unqualified. Production has stayed
running. Neither the task branch nor the inactive trial recipe is promoted.

## Source under verification

- Fork store `C:/Development/tensorfold.git`.
- Worktree `C:/Development/tf/qwen35-ornith-nvfp4`.
- Branch `task/qwen35-ornith-nvfp4`, from trunk `a92010d6e`.
- Initial port commit `8f6a04efe`.
- Reviewed preparation checkpoint
  `4ba2e07314d61c7b060c46710d32363d83c1bcbd`.
- Final code checkpoint with physical device admission
  `343486999375947fbdd69b4cf625ab63c9452b49`.
- Windows tools are Zig 0.17.0, CUDA 12.8 and MSVC 14.44.35207. The helper
  changes only its child compiler environment.

## CPU evidence

The fresh, isolated host gate passed 39 tests and all eight build steps on
the final committed code checkpoint. This includes real producer header
fixtures, refusal behavior, scale handling, independent math contracts,
memory budgets, mock cleanup faults and native-adapter admission.

```powershell
& 'C:/Users/developer/tools/zig-0.17.0/zig-x86_64-windows-0.17.0/zig.exe' build test-qwen35-cuda-host -j2 --cache-dir .artifacts/final-identity-host-cache --summary all
```

Evidence is `.artifacts/final-host-34348699.log`. The earlier 36-test gate
at `4ba2e073` preceded the three device-admission tests. The three Python reference
tests passed in `.artifacts/reference-unit-identity.log`. Ruff and whitespace
checks passed. The fleet repository's `fleet.verify fast` passed before the
source commits, in `.artifacts/fleet-fast-review-fixes.log` and
`.artifacts/fleet-fast-identity-fixes.log`.

The broad trunk Windows `test-cuda-host` gate was already failing before
the port. `.artifacts/baseline-host.log` records Windows environment-library
compile errors, POSIX spill-test issues and an inherited identity test
failure. It is not a green aggregate gate. No merge or promotion readiness
is claimed from the targeted pass.

## Final Windows artifacts

All final builds below passed at code checkpoint `34348699` with embedded
`sm_120` CUDA kernels. Each helper invocation exits zero. The durable step
receipt is `.artifacts/background-preparation-status.json`.

| Target | Result | Evidence |
|---|---|---|
| `native` | 60/60 steps | `.artifacts/background-native-34348699.log` |
| `qwen35-probe` | 56/56 steps | `.artifacts/background-qwen35-probe-34348699.log` |
| `qwen35-kernels-probe` | 56/56 steps | `.artifacts/background-qwen35-kernels-probe-34348699.log` |

| Executable | Bytes | SHA256 |
|---|---:|---|
| `tensorfold-native.exe` | 22507008 | `247110a921c007cb2d1c6d4bf895e057d9d2634e630e34727c54a12e4f606247` |
| `tf-qwen35-probe.exe` | 4670464 | `eec0a90bbe0feb81c76203042b6664270c0c19bfe94e7d0a6e0f2d32e344a66a` |
| `tf-qwen35-kernels-probe.exe` | 4612096 | `2315123f368527bf574a021477fc967c2e7f2ab79c75421d59292804afb1d021` |

The final model probe's `--inspect` passed on the complete local Ornith
checkpoint, without creating a CUDA context. Evidence is
`.artifacts/background-inspect-34348699.log`. Both executable entry points
also passed six process checks covering usage, absent environment pins and
ordinal-only masks. Expected errors occurred before driver initialization;
no output directory was created. Evidence is
`.artifacts/probe-refusals-34348699.json`.

`.artifacts/committed-source-check.json` verifies that application/tool
sources match the committed checkpoint and all four fixture capture-script
digests equal the committed script's SHA256. An earlier interrupted final
native attempt recorded no exit status and is explicitly excluded from
the passed evidence. The retry helper runs hidden and records each exit
status independently of the tool session.

The immutable bundle is staged at
`C:/Development/local-ai-server/runtime/tensorfold-qwen35-native-247110a921c0/`.
Its eight files include the three executables, `SOURCE_COMMIT`, the original
`LICENSE`, `NOTICE`, `THIRD_PARTY_NOTICES.md`, and `LICENSES.zip`. The archive
retains all nineteen files under their original `LICENSES/` paths, including
the Transformers donor license. Every archive member was compared byte for
byte with its source. `.artifacts/staged-runtime.json` records all file pins.
The helper completed every build, CPU inspection, reference capture and
staging step with exit zero.

## Inactive fleet recipe

The fleet worktree is
`C:/Development/local-ai-server/.worktrees/task/ornith35-tf-trial`, branch
`task/ornith35-tf-trial`. Launcher commit `b5ead91a` adds the serial layer
split. Commit `7a1f457a` adds the prepared recipe
`recipes/workstation/ornith15-qwen35-tf-cuda/recipe.json`, documentation,
credits and the actual-recipe CLI test.

The recipe has experimental lifecycle and prepared status. It selects one
lane, a 4096-token context, BF16 KV, the two internal UUIDs and loopback
`127.0.0.1:11439`, with served id `ornith-1.5-35b-test`. There is no deployment,
component, task, registry or proxy selection change. Its real CPU preflight
verified all eight runtime and seventeen checkpoint files by bytes and SHA256.
The exact source commit and all nineteen license archive paths were also
verified. No process was created by that check.

The launcher, documentation and attribution suite passed 228 tests,
including 67 launcher tests and the actual-recipe CLI with process creation
mocked. The fast gate and commit hook passed. Evidence lives in the fleet
worktree's `.artifacts/real-recipe-preflight.json`,
`.artifacts/license-archive-verified.json` and
`.artifacts/final-targeted-tests.txt`. These are preparation results, not a
live server probe or the full promotion gate.

Model the Domain kept format-specific checkpoint structures and topology
choices explicit. Boundary Discipline put refusal before device allocation.
Test Behavior, Not Implementation drove observable state, routing, cleanup,
argument and environment assertions. Prove It Works requires real device
and live-serving evidence before the port is called qualified.

## Checkpoint preparation

All four immutable checkpoint fixtures were recaptured with the committed
`tools/zig/qwen35_headers.py`. Their receipts now include its SHA256.
`.artifacts/fixture-recapture.json` records the captures. Config and raw
tensor-header hashes did not change. The compressed-tensors Unsloth and
Unsloth Fast formats were inspected and are explicitly refused. Ornith and
NVIDIA ModelOpt fixtures pass the text schema.

All three downloaded Ornith shard bodies match the immutable repository's
LFS SHA256 values. `.artifacts/ornith-weight-hashes.json` records bytes and
digests. The inactive fleet recipe pins these and fourteen metadata files.

The text memory plan derived from both official header fixtures is
20,850,944,512 packed weight bytes. At cut twenty, the embedding partition
uses 10,790,995,200 weight bytes and the final-head partition uses
10,059,949,312. Their peak temporary packing allocations are 16,777,216 and
286,064,640 bytes. These are calculations from the layouts, not measured
device allocations. Admission also budgets scratch, one session's GDN/KV
state and a 1536 MiB floor on each device before any weight uploads.

Cleanup regression tests prove device frees are attempted after a failed
synchronization. Context-selection failures release host ownership and
attempt to release the primary retain without freeing against a wrong
context. Shared CUDA free/unload/destroy APIs discard driver errors.
A fatal driver fault can still require process exit for complete reclamation.

## Reference tooling

`tools/zig/qwen35_reference.py` runs the exact Transformers source at
`e0eb2810ac1b2caf6d082f247d8d37162ecf67a0`, checked against SHA256
`d9f88f2d048492c6a787119c7466add91cb311d7699aea5701ffb8a3ba3015a5`.
It uses the installed Transformers 5.16.1 integration and Torch 2.9.0+cu128.
CUDA devices and hub kernels are disabled in the capture process. The four
optional kernel wrappers are unwrapped to the source's PyTorch fallbacks.
The original model forwards are retained. Linear modules and expert weight
indexing decode the checkpoint lazily to fit host memory. FP8 activations
use stored static scales. NVFP4 activations remain BF16.

The committed short fixture has five prompt ids and two forced ids. The
capture at final source `34348699` passed and produced reference top-one ids
`11751` then `13`, corresponding to ` Paris.`. Evidence is
`.artifacts/final-reference-34348699.log` and
`.artifacts/reference-34348699/result.json`. A second case uses the pinned
chat template with thinking disabled for `Say hello.`. Its fifteen-token
prompt and two forced rows produced top-one ids `9419` then `0`, or `Hello!`.
Its input and receipt are in `.artifacts/ornith-forced-chat.json` and
`.artifacts/reference-chat-34348699-background/`. The complete exit-zero
chat capture is `.artifacts/background-reference-chat-34348699.log`.

The pinned source emits documentation diagnostics for two unused vision
methods when loaded with the installed newer integration. Both CPU captures
exit zero and write all expected rows. The reference is HF eager arithmetic
with BF16 dequantized weights, not an independently qualified ModelOpt
serving runtime. Its dequantization and static FP8 emulation also need
comparison with a second quantized implementation before broad accuracy
claims.

Round two incorrectly inferred that the documented capture interpreter could
not run because the imported packages reside under the ExLlama recovery
runtime. The command did run successfully. An explicit check in
`.artifacts/reference-environment.log` confirms `sys.executable` and
`sys.prefix` are the documented `glimmer-capture-venv`. Its imports resolve
to the existing recovery packages. Future receipts also record interpreter,
prefixes and optional package versions. No package installation or change to
the recovery runtime was made.

The row comparator checks matching config/index hashes, token stream,
vocabulary, row counts and finite values. Initial diagnostic limits are
RMSE 0.05, maximum absolute error 0.5 and reference-to-candidate KL 0.001,
with identical top-one ids for every row. They were selected before any
device output. They are not a measured accuracy guarantee. FP32 attention,
routing weights and expert accumulation intentionally differ from HF eager
BF16 boundaries. Investigate a failure instead of loosening a limit to pass.

## Review and audit limits

Claude Opus 5.5 reviewed the code and retained evidence. The complete raw
conversation was not supplied, so its audit is limited to files and logs.
Round one's result is `.artifacts/review-r1.md`. It identified missing
device comparisons, missing teacher tooling, stale evidence links,
inaccurate initial timestamps, fault cleanup, memory preflight, NVIDIA
activation-metadata ambiguity, fixture script provenance and a build-entry
credit. Source fixes and comparison tools address the preparation findings.
Device results remain open.

Round two is `.artifacts/review-r2.md`. Its P1 requires physical GPU UUID
admission in both probes. Fleet server launches already pin the internal
pair. The probe fix adds explicit expected UUIDs and verifies driver UUIDs
before any context allocation, then records UUIDs and names. Query and key
norm vectors in the
attention test are now distinct. The device oracle receipt explicitly calls
its math a separate host implementation, not HF-generated golden vectors.
The HF end-to-end capture supplies the separate framework comparison.

The model probe creates two sessions to check isolation. Initial engine
admission budgets one; the second session checks its complete allocation
and floor before allocating state. That can still refuse after weights have
loaded. The text weight split leaves a calculated margin on idle 16 GiB
cards, but actual fit remains unmeasured. Layer-level hidden-state dumps
and a reference variant isolating the FP32 routing/accumulation drift are
not implemented. A final-logit mismatch may require adding that diagnostic
instrumentation during qualification.

Round three is `.artifacts/review-r3.md`. It found no new P0, P1 or P2
defects in the probe admission changes or the fleet launcher patch. Its
remaining conditions are final committed-source probe builds with recorded
digests, a fully pinned inactive recipe and server launch through the fleet
wrapper. It withdrew the reference-interpreter objection after reading the
environment evidence. This review does not establish any device result.

The decision log is append-only. Its initial four timestamps were estimates,
not captured event times. Later rows supersede those times using recorded
filesystem observations, and correct the inspection evidence pointer and
the implication that earlier builds covered the final source.

## Remaining device gate

After approval to stop workstation inference, run both internal GPUs through
the compiled per-kernel probe. It covers all ten Qwen entry points and the
reused RMS kernel with 56 array comparisons and sixteen exact routing ids.
Then run the model probe on both forced cases, compare all logit rows,
exercise reset and interleaved session isolation, and inspect failures.

Launch the inactive loopback recipe through `python -m fleet.launch`.
Check real text/reasoning/tool requests, queued requests, cancellation and
reuse, context refusal and a longer prefill. Measure free memory under load
on both GPUs and enforce 1536 MiB on each. These steps have not happened.
Performance claims require the fleet benchmark workflow and its own evidence.

Stop the trial, restore the production recipe under its existing wrapper,
and confirm a proxy completion before ending the approved outage. Any merge,
deployment selection or public model route follows qualification separately.
