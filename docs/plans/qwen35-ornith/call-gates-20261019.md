# Required and named call gates

Source `aa267a2f919aebadbea114f9be883ec04167abe0` enables Qwen's shared
required and named tool gates. It removes the call-only request and adapter
refusals and sets `api.Info.call_gates` in the native Qwen profile. Grammar,
structured output, logprob and thinking-budget refusals remain explicit.
No CUDA arithmetic or sampler changes.

Astra's design review selected shared gate ownership over native masking.
Qwen has `gpu_tokens=false`, so its path is synchronous `familyRound`.
Verification feeds the prior pending token to the session, then the core
corrects the newly sampled output before making it pending. Width one keeps
path `{0}` and does not require model-state rollback. The complete review is
`.artifacts/call-gate-review-astra-20261019/report.md`.

The real-adapter regression fails before the fix with `UnsupportedRequest`.
Afterward the focused gate reports 47/47 passes; the CUDA host aggregate
passes 45/45 steps with 37 reported passes and other roots cached. A later
test-only extension covers the transition from an open thinking block and
a valid sampled name prefix followed by an invalid token. It asserts both
emitted tokens and the exact tokens appended to the fake model session.
It also passes the 47/47 gate. These extra cases are CPU evidence; the device
tools requests below explicitly disable thinking.

## Bounded device result

Native compilation passes 60/60 steps. The immutable server has SHA256
`a810398a82ed4e2f4efaeec74866827dfafe1c19ccbb017d41183e4d0d37df91`
under `runtime/tensorfold-qwen35-native-a810398a82ed` in the fleet repository.
Its source marker pins `aa267a2f`. The unchanged diagnostic probes retain
their source `973428fb` and exact previous hashes; no new probe build is claimed.
The bundle preserves all original license and donor notices.

The inactive `ornith15-qwen35-tf-callgate` recipe retains one lane, 4096
tokens, BF16 KV, FP32 GDN and layer cut 20 across the two internal GPUs.
Its exact pinned preflight passes. The detached controller limits the
experiment to five minutes, retains the existing recovery bounds, and
refuses starts after 00:52:24 UTC within the user's hour ending 01:07:24 UTC.

All thirteen unchanged lifecycle checks pass on the new binary, including
exact raw/chat token signatures, cancellation and deterministic reuse.
Two additional nonstreaming requests use two offered functions. Required
choice returns `greet_person`; named choice returns the requested
`wave_person`. Each returns exactly one parsed tool call with valid JSON
arguments `{"name":"Brett"}` and `finish_reason=tool_calls`.

The server separately records queued cancellation with zero generated tokens
and active cancellation with one. Fifty-one one-second physical samples
have minima 5358/6105 MiB free. Unsampled peaks remain unknown. The controller
reports no sampler or cleanup errors. Nemotron returns exact local-only proxy
READY at 00:50:40.427128 UTC from the preserved executable, PID 16512;
Clef PID 21392 is unchanged. The candidate and controller have exited.

Fleet trial `fleet/trials/ornith15-qwen35-callgate-20261019.json` retains the
raw receipts and informational verdict. The exact-byte archive has SHA256
`15c4ba181fecb47ba0c0ff20a1b620e9edffd6bdc81edc6d927ae23fbb9ee654`.
The TensorFold [approved-hour evidence](evidence/approved-hour-20261019/README.md)
also preserves the diagnostic and tool receipts. Fable round 6 independently
observed the completed restoration; its review findings and disposition are
recorded in [the shared-host review](shared-host-review-20261018.md).

This is bounded call-gate integration evidence. Streamed tool deltas,
reasoning-enabled device tools, malformed arguments, long tool loops and
broad model quality remain unqualified. The four full-logit rows still fail
unchanged limits. The short serial correctness requests do not establish
production performance or capacity. Multiple lanes, batched prefill, cache,
full context, soak, restart and fleet promotion verification remain ahead.
