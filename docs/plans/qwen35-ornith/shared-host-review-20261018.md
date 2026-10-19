# Shared-host implementation review

Fable 5.1 reviewed the implementation and decision trail read-only, through
the T3 delegated task `qwen35-shared-host-fable-review-r1-20261018`. Its full
response is `.artifacts/shared-host-review-20261018/fable-r1.md`.
The code review found no blocking defect. It verified owner cleanup,
partition context selection, keyed sample positions, fill reporting,
one-row keep paths, cancellation/release and pre-stream request refusal.

## Findings and disposition

1. **A failed receipt is misleadingly named green.**
   `shared-core-green.log` records the fixed zero-output test followed by the
   pre-fix Windows spill fixture failure. It is retained as historical red
   evidence, not a passing aggregate. The original trail row also cites
   `spill-timestamp-green.log` and a replay, but abbreviated paths obscure
   that distinction. An append-only correction gives complete red and green
   paths. `shared-host-aggregate.log` and `spill-timestamp-green.log` each
   report 45/45 successful steps, with cached roots explicitly recorded.

2. **The zero-output trail row abbreviates a path.**
   The append-only correction supplies the full aggregate path and the
   separate original red regression path. No history was rewritten.

3. **Zero-output completion affects other shared-core families.**
   The design now states the GLM direct-engine impact. `warmPass` preserves
   warm-turn behavior. The HTTP layer maps zero to its default and requests
   at least one token, so this core correction does not change that HTTP
   contract. Source searches found no dependent GLM caller. This is not
   device qualification for the changed shared core.

4. **Keep validation is strict but valid.**
   The reviewer traced the shared core: legitimate one-row keep paths remain
   `[0]` even when a stop or loop guard cuts the round. No change is needed.
   Multi-row rollback remains explicitly refused.

5. **The review brief's targeted count became stale.**
   A later cancellation test raised the receipt from 43 to 44 reported passes.
   `shared-qwen-final.log` is authoritative; the three-test admission root is
   cached and is not added to the reported fresh count.

The reviewer also flagged printed `FAIL limits sm_121 sample ... (record
stale)` text. Its source is `zig/tests/cuda/contract_limits.zig:125`, the unit
test that deliberately creates a stale synthetic record and checks failure
detection. It also emits `MISSED` and `RED` control messages. This is expected
CPU negative-control output, not an actual fatbin validation failure. The
source excerpt and explanation are saved in
`.artifacts/production-prep-20261018/contract-limits-negative-control.json`.

`Info.name` changes from `qwen35-cuda-serial` to `qwen35-cuda-lanes`. Searches
of TensorFold and fleet code found no behavior keyed to the old name; its
fleet occurrence is a historical receipt. Adapter capacity and model capacity
both derive from the same validated context passed to `Engine.open`.
Environment refusals have pure getter tests; executing the actual launcher
environment remains part of the device trial.

## Attention retained at the pretrial review

At this review, all new session/lifecycle tests used fake sessions or backends.
The later approved trial exercised real CUDA listing and raw completion before
ordinary chat failed. Other device lifecycle behavior remains unverified.
The four whole-model logit rows still fail their fixed diagnostic limits.
The independent NVIDIA CPU comparison covers one expert and selected inputs;
both projection paths share CPU PyTorch matmul. vLLM separately changes 39
FP8 fusion groups, so a later serving discrepancy requires interpretation.
These remain explicit qualification boundaries; no limits were relaxed and
no production promotion was made.

## Final trail review and live-host correction

Fable's second read-only pass verified the candidate file pins, baseline
provenance and restore-helper bounds. Its full response is
`.artifacts/shared-host-review-20261018/fable-r2.md`. It found a real
controller defect: ownership lookup matched ports on all addresses, including
the tailnet forwarder. Reading that privileged forwarder's executable fails,
so the trial would abort before its outage and restoration identity checks
would also be wrong.

Fleet commit `cc846d46` filters ownership to exact `127.0.0.1` LISTEN sockets.
The mixed-address regression fails before the fix and passes afterward.
Six runner behavior tests and 140 combined targeted tests pass. Ruff,
`fleet.verify fast`, and the actual default pinned preflight pass after this
final source change. Fleet artifacts `ornith-loopback-red.log`,
`ornith-loopback-green.log`, `ornith-loopback-targeted.log`,
`ornith-loopback-preflight.log`, and `ornith-loopback-live-identities.json`
record the proof. The live check finds the preserved Nemotron and Clef
loopback engines and an unused trial port, without reading the forwarder's
executable or taking any service action.

The review also requested explicit sampler and baseline boundaries. The
runbook now states that continuous samples cover candidate load/requests;
the restoration helper records a separate final physical-floor checkpoint.
It also states that a hash mismatch against the earlier source's serial
receipt requires diagnosis and cannot distinguish a host regression from
arithmetic drift by itself. Limits and expected hashes remain fixed.

The first controller safety review separately replaced child discovery by
polling with immediate ownership of the `Popen` handle returned by the
existing `fleet.launch.start` API. A readiness-failure mock verifies cleanup
and attempted restoration. These fixes are CPU-tested; their device behavior
was part of the subsequently approved first window.

## Post-trial review and disposition

Fable 5.1 reviewed the failed trial and first chat correction in round 3,
then the history correction and expanded probe in round 4. The complete
responses are `.artifacts/production-trial-20261018/fable-r3.md` and
`fable-r4.md` in that directory. Round 4 reports no actionable blockers.

- **Ordinary chat history would still fail.** Round 3 found that prior reply
  spans would hit another premature refusal. Commit `973428fb` removes that
  request-level guard. The regression fails before the change and passes
  afterward; the shared-host fixture carries both metadata fields and asserts
  decoded rows are disabled. The host still discards spans before prefill,
  and the low-level backend still refuses explicit decoded-row execution.
- **Automatic tool requests might hit the call guard.** Source inspection
  shows that the server constructs call gates only for required or named tool
  choice. Automatic choice does not hit the guard. No change or device tool
  qualification is claimed.
- **Builds were still running and fleet evidence was uncommitted.** These
  timing observations were accurate when the reviewer read the worktrees.
  The final source now has observed 60/60 native and 56/56 probe build
  summaries. Fleet commits `4217a658` and `d48f7474` preserve the rejected
  trial and original evidence bytes; `93963735` pins the corrected recipe.
  The rejected manifest retains its original source and twelve planned checks.
- **The corrected recipe needs independent pins.** The new inactive
  `ornith15-qwen35-tf-chatfix` recipe has its own eight runtime pins and the
  unchanged seventeen checkpoint pins. The actual safe preflight verifies
  those pins without an outage. The failed recipe's pins are preserved.

The reviewer retains three qualification boundaries. The four full-logit
rows still fail their fixed limits. Nine historical lifecycle checks did not
execute. Corrected chat, including history, has not run on hardware. The
next probe has thirteen checks and requires a fresh approved outage because
the first window ended with verified production restoration. CPU preparation
and review do not close those boundaries.
