# Qwen expert variant screen

The target is lower resident c1 complete-turn time on the existing 64-input,
128-output workload, with exact native regression outputs and the existing
1536 MiB free-memory floor on each internal GPU. The next decode milestone is
100 tok/s from 81.668 tok/s. Reaching it requires 2.245 ms less decode time;
this experiment does not assume that expert variants alone can supply it.

## Chosen seam

The design compared a parameterized standalone tf-cuda-test command against a
specialized expert mode in the existing pinned decode-profile executable.
The latter keeps the existing physical device admission, identity, journal,
deadline, owned-process cleanup and protected production recovery. It avoids
changing the GLM benchmark or the controller for a two-epilogue proof.

The profile input supplies a fixture directory and its manifest SHA256. The
expert mode branches before Engine.open, creates one admitted GPU context at
a time, and checks the real Qwen shapes through the existing expert runtime.
No full checkpoint is loaded on the GPUs. The fixture is about 432 MiB of
actual layer-zero weights, packed on the CPU using the existing ModelOpt
packing layout. Model weights, scalar scales and the saved native row are
recorded by hash. No new donor kernel is introduced.

The fixture builder is tools/zig/qwen35_expert_fixture.py. It checks the saved
capture hashes retained by the earlier component comparison and performs a
CPU code/scale packing round trip. The real captured experts occupy bank 0,
in their original slot order. The remaining experts occupy 31 more banks.
Those other banks use controlled route substitutions on the captured input;
they are not 31 additional observed production routes.

## Prediction and gates

The up grid has 32 baseline blocks or 64 n16 blocks before residency capping.
The prediction is that one of the existing pipe4, n16 or n16_pipe4 variants
will reduce device time enough to justify a full-turn trial. Achieved
occupancy and the limiting hardware resource remain unmeasured.

Use hidden 2048, intermediate 512, 256 experts, eight distinct selected items,
SwiGLU without clamping, BF16 up outputs and FP32 down outputs. Require finite
and byte-identical baseline/candidate outputs for every bank. Bank 0 must
also match the retained native hidden and down outputs. Poison outputs and
check graph results after timing. Each graph executes 128 expert launches,
cycling all 32 banks four times. Warm each graph, then take at least five
explicit baseline/candidate pairs with alternating order on both GPUs. Keep
raw samples and launch/work counts. Logical weight bytes divided by time do
not measure physical DRAM bandwidth.

Only a repeatable gain exceeding variability and plausibly saving at least
0.113 seconds from the 2.264-second full turn earns integration. A rough
projection is 191 appends times the sum of 20 layers' per-call saving on each
GPU. The thin seam omits interleaving with other operators, so the projection
must be tested on the full changing-token/position graph before claiming a
model speedup. A failed correctness gate rejects the variant. A harness
failure gets one bounded repair if its cause is established.

Prepare builds, executable/input/fixture pins and commands before stopping
production. Run through tools.run_ornith_tuning_window with at most 600
seconds of device experiments and 500 seconds reserved for recovery. Preserve
Clef. Restore Nemotron and require the exact local proxy completion.

This is native regression and performance work. Independent full-logit
qualification remains failed, and this screen cannot authorize promotion.
