# The two-GPU layer split: the proving run

This is the run that turns the split from a checked mechanism into a served model. It needs an outage window on
both hosts, because production is resident on every card it uses.

## Why it needs a window

- The workstation's internal GPUs hold llama.cpp's Nemotron layer split: 13648 MiB on the RTX 5080 and 12928 MiB on
  the RTX 5070 Ti after a request, leaving 2330 and 3068 MiB free. The split needs 8.64 GiB on rank 0 and 8.69 GiB on
  rank 1.
- An idle llama-server pages its VRAM out under WDDM, so the cards read nearly empty until a request lands. Check
  free memory after a completion, not before.
- Spark 1 serves GLM 5.3 Flash and has about 8 GB available. The reference run needs its production stopped too.

## The run

1. Stop workstation inference: `Stop-ScheduledTask -TaskName 'LLM - Local Inference'`. Confirm both internal cards
   are free with `nvidia-smi`.
2. Serve the split:
   ```
   TENSORFOLD_SPLIT=1 TF_CUDA_DEVICE=1 \
     out-win/native/bin/tensorfold-native.exe serve \
     C:/Development/prep/nemotron-20261012/staging-d9d758fb \
     --context 8192 --temperature 0 --no-thinking --port 11499
   ```
   `TF_CUDA_DEVICE=1` is the RTX 5080, the first of the two sm_120 devices; the split opens the RTX 5070 Ti as rank 1.
   The startup line must report rank 0's model bytes, not the whole checkpoint.
3. Check `/health`, `/v1/models`, and one chat completion. Record the generated token ids.
4. Stop the split server.
5. On Spark 1, stop the GLM pair through its user unit, run the qualified native binary on the same checkpoint and
   prompt, and record the token ids. That is the independent reference the seam check cannot substitute for.
6. Restore both: the GLM pair on Spark 1, workstation inference through its task. Confirm a proxy completion on each.
7. Record the result as a manifest in `fleet/trials/`.

## What the run proves, and what it does not

Proves: the split serves a real prompt on the workstation's two internal GPUs, and its tokens match the Spark native
run for the same checkpoint and prompt.

Does not prove: decode continuation parity, multi-stream behaviour, the prompt cache, the MTP head (the split
disables it), graph capture (the split runs eagerly), or sustained load. Each is a later unit.

## Known gaps before promotion

- The split runs eagerly with graphs off and the MTP head disabled. Both are deliberate: a graph captures one
  device's kernels and cannot span the seam, and the head lives on rank 0 while its embedding reads rank 1's rows.
- Per-device admission, reserve and peak are not qualified. The startup line reports rank 0's bytes; rank 1's are
  checked only by its own open.
- The single-device regression has been checked by object-level cross-compiles only, not a Linux native build.
