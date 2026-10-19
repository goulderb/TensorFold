# Corrected shared-host integration candidate

The first shared-host trial failed ordinary chat and restored Nemotron at
23:47:23 UTC on October 18. The two passive chat-metadata refusals are now
corrected. This candidate is prepared for another bounded integration trial;
it has not run a model and is not qualified for production.

The source pin is `973428fb85203e636a94c92c80fb8bb34a79369e`. The executable
SHA256 is `d1f462a330bb04a13c40f47ad6ff5316d1a3cd8e5062721607488951b0413744`.
The immutable directory is
`C:/Development/local-ai-server/runtime/tensorfold-qwen35-native-d1f462a330bb`.
Its eight pinned files include the native server, both diagnostic probes,
source marker, original license and notices, and the complete license archive.
Archive entries were compared byte-for-byte with the source tree. Staging
replay verifies the existing files without overwriting them.

The complete staging manifest is
`.artifacts/production-trial-20261018/staged-chat-fixed-candidate.json`.
Final native compilation passes 60/60 steps, and both diagnostic probe builds
pass 56/56. Focused Qwen host tests report 46 passes, with the admission root
cached. The aggregate passes 45/45 steps with 98 reported passes, one skip and
other roots cached. The corresponding `chat-history-*.log` receipts are in
the same artifact directory. These are CPU tests and compilation results.

Fleet commit `939637359c1a5a376e1a8d4af618272019951b2f` adds the inactive
`ornith15-qwen35-tf-chatfix` recipe. It pins all eight runtime files and the
unchanged seventeen checkpoint files. The final targeted fleet tests pass
145 cases; Ruff and `fleet.verify fast` pass. The actual default coordinator
preflight hashes every pin and reports `outage_started=false`. Its receipt
is `.artifacts/ornith-chatfix-preflight.log` in the fleet task worktree.

The prepared fleet procedure is
`docs/plans/ornith35-shared-host-trial-20261018.md`. It keeps one lane, 4096
context tokens, BF16 KV and FP32 GDN state, split after twenty layers across
the two internal GPUs. This is layer splitting, not tensor parallelism.
The thirteen-check HTTP probe adds a prior assistant turn. That new case
checks successful nonempty output without claiming uncaptured serial parity.

No second outage has run. Fresh approval is required before stopping Nemotron
for the ten-minute experiment, with ten further minutes reserved for recovery.
The current production runtime and all failed-candidate pins are preserved.
The four failed numerical rows, multi-lane execution, batched prefill, cache
and the proposed two-lane 64K service profile remain unqualified.
