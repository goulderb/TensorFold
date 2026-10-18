# Shared-host device integration candidate

This candidate is prepared for a new approved device window. It has not run a
model. It is not numerically qualified or selected for production.

The source is `0934d59f8b4d6c8d9dec1c3c3b72e77cd13f5dde` on
`task/qwen35-ornith-nvfp4`. Later commits only add documentation and review
evidence; the staging script checks that no source code differs from this pin.
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

Use the separately pinned inactive fleet recipe
`ornith15-qwen35-tf-shared-host` for a one-lane, 4096-token integration trial
at loopback port 11439. The fleet task worktree's
`docs/plans/ornith35-shared-host-trial-20261018.md` owns the exact HTTP checks,
physical-memory abort guard, bounded window and restoration procedure. That
recipe and procedure require their own CPU validation before requesting the
window. No active fleet selection, route or scheduled task changes here.

The trial asks whether shared-host requests preserve the original bounded
serial behavior. Numerical acceptance, multi-lane execution, useful batched
prefill, GPU prefix cache, the proposed two-lane 64K service profile and fleet
promotion remain separate uncompleted units. The failed full-logit limits
remain unchanged.
