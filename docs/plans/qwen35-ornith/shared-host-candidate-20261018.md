# First shared-host device integration candidate

This candidate ran in the approved October 18 window and failed ordinary
chat. Private listing and raw completion passed; nine later checks did not
execute. Nemotron was restored and verified through the proxy at 23:47:23 UTC.
The [trial result](shared-host-trial-20261018.md) preserves that outcome.
Use the separately pinned [corrected candidate](chat-fixed-candidate-20261018.md)
for the next proposed trial. Neither candidate is qualified for production.

The source is `0934d59f8b4d6c8d9dec1c3c3b72e77cd13f5dde` on
`task/qwen35-ornith-nvfp4`. At staging, later commits only added documentation
and review evidence; the staging script checked that no source code differed
from this pin. Subsequent code corrections are pinned separately at `973428fb`.
The native executable has SHA256
`ced689d1877eef43057b7a7869dbfeaf8b92cbe59683c483c534f23a516bd6a6`.

The immutable directory is
`C:/Development/local-ai-server/runtime/tensorfold-qwen35-native-ced689d1877e`.
Its eight pinned files include the server, both diagnostic probes,
`SOURCE_COMMIT`, root license and notices, and a newly archived `LICENSES`
tree. Every archive entry was compared byte-for-byte with its source. The
original trial and production runtime directories are preserved.

The complete file manifest and build receipts are in
`.artifacts/production-prep-20261018/staged-candidate.json`.
`stage-candidate.py` in that directory stages the bundle without launching a
model. Its replay verifies existing file hashes without overwriting the
bundle. Native server compilation passes 60/60 steps; both probe builds pass
56/56. The final CPU aggregate passes 45/45 steps, with 96 reported tests
passing, one skipped, and the remaining roots cached.

The native `--help` command passes. `capabilities --json` also passes; it
opens the CUDA driver and reads device metadata without creating a context,
allocating device memory or executing a kernel. With the internal GPU UUIDs
visible, it reports `nvidia-sm120` and the `qwen3_5_moe` mixed ModelOpt format.
The unrestricted metadata query selects the machine's ordinal-zero SM89
device; the trial launcher must set the two internal UUIDs explicitly.

The separately pinned inactive fleet recipe
`ornith15-qwen35-tf-shared-host` was used for the one-lane, 4096-token integration
trial at loopback port 11439. The fleet task worktree's
`docs/plans/ornith35-shared-host-trial-20261018.md` owns the exact HTTP checks,
physical-memory abort guard, bounded window and restoration procedure. That
recipe and procedure received CPU validation before the approved window.
That validation is recorded in fleet commits `ce244f57` and
`cc846d46`: 140 targeted
tests pass, Ruff and `fleet.verify fast` pass, and the coordinator's default
preflight hashes every runtime and checkpoint pin without starting an outage.
A separate safety review and six runner tests exercise missing/below-floor
readings, mixed-address ownership, readiness failure and restoration mocks.
The controller uses the existing
`fleet.launch.start` API to own the native process handle immediately; no
production launcher implementation changes were needed. Its approved run
starts detached, aborts on a missing or below-floor physical GPU reading, and
cleans up the owned candidate before restoring Nemotron. The trial budget is
ten minutes, followed by at most ten minutes of planned recovery. The trial
exercised listing and raw completion before chat failed; cleanup and production
recovery passed. No active fleet selection, route or scheduled task changed.

A current production health request at 23:40:53 UTC returned HTTP 200 and
exact `READY` from `nemotron-3.5-lightning` through the local-only proxy, with
paid spillover disabled. Its sampled physical free memory was 5248 and 5491
MiB on the internal GPUs. This was a normal request to the existing production
engine; no outage or restoration occurred. The receipt is
`.artifacts/production-prep-20261018/current-production/production-restored-proxy.json`.

The trial asked whether shared-host requests preserve the original bounded
serial behavior. Numerical acceptance, multi-lane execution, useful batched
prefill, GPU prefix cache, the proposed two-lane 64K service profile and fleet
promotion remain separate uncompleted units. The failed full-logit limits
remain unchanged.
