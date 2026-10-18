# Qwen35 shared-host integration

The first integration keeps the existing CUDA `Host` in `native/cuda.zig`.
Qwen loads its own model and CUDA contexts, then gives that same host a
one-session `lanes.backend.Backend`. The host owns its existing `lanes.Engine`
and `api.LaneHost`. Qwen's private request queue and worker are removed.

Two independent sketches compared this shape with extracting a new generic
runtime owner. The cross-judge chose the existing host, 23 to 22 across six
criteria, because extraction would move working Nemotron/GLM startup and
exceptional disk teardown before Qwen needs that change. Retain those paths.
Candidate artifacts and the judge's response are under
`.artifacts/shared-host-design-20261018/` and the session's subagent results.

The selected shape adapts three details from the extraction candidate:

- A pure optional request-refusal hook runs before `LaneHost` converts a
  request to a stream. It preserves refusal of `decode_spans`, which the host
  otherwise discards when decoded-row execution is disabled.
- `Info.prefill_step` stays zero. The backend's `prefill_step` callback yields
  between serial prompt tokens and publishes `fill_at`; it does not define
  the HTTP prompt chunk plan or establish faster prefill.
- The backend implements real one-row `verify`, identity-only `keep`, and
  explicit refusal of draft depth. Shared sampling and token commit semantics
  remain in the existing core.

The Qwen adapter owns one active stream slot. Binding resets the existing
session. Release clears the slot; the next binding resets even a poisoned
session. The model and sampler buffers live until the shared host has joined
its thread. Startup transfers model ownership only after the host starts.
Failure unwinds the unstarted host and family exactly once.

The native host does not own a CUDA context for Qwen. `Session.reset`,
`append`, and `deinit` already select each partition's context. The sampler
reads downloaded host logits. Memory reporting uses an explicit source rather
than treating a host-owned context as the only possible CUDA allocation owner.

Initially report one lane, plain width-one execution, no cache, call gates,
thinking budget, structured response output, logprobs, vision, or drafts.
Automatic tool output and thinking toggles continue through the template.
The shared loop guard is supported by the core. Keep layer-split, precision,
segments, spill, and other capability refusals until their later units pass.

Both sketches missed the shared zero-output path: it draws one token for a
zero-token request, while the serial adapter emits none. Preserve the serial
contract through a tested zero-output completion in the shared core. Warm
prefill behavior remains on its existing host path.

This corrects the direct engine contract for every family that reaches the
core, including GLM without warm turns. Warm-turn engines take `warmPass`
before the core; that behavior is unchanged. The OpenAI HTTP layer treats
`max_tokens: 0` as the configured default and always requests at least one
output token (`server/openai.zig:148`, `server/chat.zig:108`). It therefore
does not expose this engine-only zero-output path. Source searches found no
GLM caller depending on the former extra token. This is source evidence, not
a GLM device qualification. Preserve the existing HTTP fallback semantics in
the device trial.

CPU tests first cover direct request refusal before prefill, exact-once
completion, cancellation during stepped prefill, reuse after refusal and
failure, zero output, one-row backend contracts, and keyed sample positions.
Fake sessions prove adapter and lifecycle behavior, not CUDA arithmetic.
Compile and host-test the migrated native/server roots before staging a
candidate for an approved device window. The failed numerical gate and
production qualification remain open.

## CPU implementation evidence, 2026-10-18

The adapter now implements this design. `native/qwen35_cuda.zig` owns the
loaded model, session and sampler buffers; `native/cuda.zig` starts and closes
the existing host over that backend. Its former private job queue and worker
are removed. A pure optional `LaneHost.request_refusal` runs before stream
construction. Other families leave it unset. Memory reporting can come from
a family that owns its CUDA contexts.

The zero-output regression failed before the core fix with `expected 0,
found 1`. Both whole and stepped prefill now finish without sampling. The
Qwen fake-session tests cover zero, one and four output tokens through the
shared round loop, keyed sampler positions, one-row verification, refused
rollback/drafts, cancellation before append, and a fresh reset after failure.
Native-host tests exercise the Qwen profile's explicit memory source,
pre-stream refusal, exactly one terminal event, zero output and subsequent
reuse. The existing native-host cancellation regression also passes.

Receipts under `.artifacts/production-prep-20261018/`:

- `zero-output-red.log`: the original regression fails as intended.
- `shared-qwen-rounds.log`: 8/8 build steps; 43 reported tests pass, with the
  admission root cached.
- `shared-qwen-final.log`: 8/8 steps; 44 reported tests pass after adding
  cancellation midway through the actual adapter's stepped prefill and reset
  of the next stream. The admission root remains cached.
- `shared-host-aggregate.log`: 45/45 steps; 95 reported tests pass and one
  skips, with the remaining roots cached. Cached roots are not fresh runs.
- `shared-native-first.log`: 60/60 build steps, including the Windows native
  server and embedded SM120 fatbins. This is compilation, not device execution.
- `windows-net-replay.log`: the Windows address diagnostic is from a passing
  negative proxy test; its executable exits zero, 377 pass and 29 skip.
- `shared-core-green.log`: the zero-output fix passes, but an unrelated
  same-length artifact rewrite test fails intermittently because its snapshot
  metadata does not change. The Windows worker gives that fixture an earlier
  modification time; `spill-timestamp-green.log` and eight executable replays
  pass. Production artifact identity code is unchanged.

These checks exercise CPU request handling. They do not execute the migrated
CUDA session, establish performance, or close the failed whole-model logit
gate. The first device integration trial remains one lane at 4K; the proposed
two-lane 64K service profile requires subsequent batched execution and cache
work.
