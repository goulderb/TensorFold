# Shared-host trial and ordinary-chat correction

The user approved the prepared hardware test on 2026-10-18. The detached
controller started at 23:45:59 UTC with source
`0934d59f8b4d6c8d9dec1c3c3b72e77cd13f5dde` and binary
`ced689d1877eef43057b7a7869dbfeaf8b92cbe59683c483c534f23a516bd6a6`.

The trial failed its ordinary chat check. The private-model check and raw
completion passed. All eight raw output tokens match the original serial hash
`465dab42a5fb38434efc537ad5a49a85e9e3be580d0a175fe156bd6aff06d0cb`.
The thinking-off chat request then returned HTTP 500 with
`Qwen CUDA has no routed lessons`, before prefill or any output token. The
remaining nine lifecycle checks did not execute. This is a failed integration
trial, not a successful twelve-check qualification.

The controller cleaned up both owned processes and restored Nemotron through
the existing task. At 23:47:23 UTC the local-only proxy returned HTTP 200 and
exact `READY` from `nemotron-3.5-lightning`, with paid spillover disabled.
The restored executable is the preserved `8047d1a0917e` runtime. Clef's process
identity is unchanged and the private trial listener is gone. Both cleanup
and sampler error lists are empty.

Twenty-five physical-memory samples cover candidate startup and requests.
Their minima are 5358 MiB on the internal RTX 5080 and 6105 MiB on the internal
RTX 5070 Ti, above the required 1536 MiB floor. The restoration checkpoint is
5572/5815 MiB. These are sampled readings, not continuous minima or a capacity
qualification. No new performance or whole-model numerical claim follows.

Raw evidence is in the fleet task worktree under
`.artifacts/ornith-shared-host-20261018-234559/`; the durable fleet trial record
copies the original controller, HTTP, server and restoration evidence.

## Root cause and CPU correction

`server/chat.zig:151` attaches `Request.decide` to ordinary foreground chat,
using the last assistant-header position from `prompt.decideOf`. The marker
identifies a possible lesson decision row. Its presence does not request or
enable learning. Qwen's new request-refusal hook wrongly refused every request
with this normal metadata field. Raw completions do not carry it, explaining
the split observed on hardware.

The correction removes that single refusal. Unsupported learning remains
refused by the model-open flags; the Qwen backend never enables a lesson
implementation. No CUDA arithmetic, checkpoint layout, sampler or numerical
limit changes.

The focused regression, `Qwen accepts an ordinary chat prompt carrying a
decision position`, fails before the correction. The shared native-host
fixture's valid request also carries a decision position, proving completion
through its real refusal hook and lane lifecycle. Targeted tests then pass
45 reported tests, with the admission root cached. The aggregate passes 45/45
steps, with 97 reported passes, one skip and other roots cached. Receipts are
`.artifacts/production-trial-20261018/chat-position-{red,green,aggregate}.log`.

This establishes the CPU fix for the observed rejection. Successful raw CUDA
execution does not prove corrected chat, streaming, cancellation, reuse or
refusal behavior on a device. Those require the corrected candidate's next
approved test. The first window is complete and production is restored.
The existing four failed full-logit rows, multi-lane work, cache and production
qualification remain open.
