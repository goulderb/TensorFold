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

## R1 result

The [retained journal and analysis](evidence/expert-tuning-20261019/r1/analysis.json)
pass every exactness gate on both GPUs. There are 384 bank parity records,
144 raw timing records (24 warmup and 120 measured), and four successful
projection completion records. The measured source is
81ea3e0b6b7fff280a400102103e651ab566c5de, executable SHA256
17ed2bc1df8e2f1ef545af7a43fbac9b3b3cdc5e03911d89dee4a3a616f71326.

For n16_pipe4, up median is 53.475 to 42.979 microseconds on GPU 0 and
53.374 to 42.860 on GPU 1. Its paired up savings range from 10.463 to
10.814 microseconds on GPU 0 and 10.412 to 10.828 on GPU 1. Down savings
are much smaller, median 0.266 and 0.242 microseconds. All five pairs
improve for each of those four groups. The independent review found a roughly
0.2-microsecond order effect in down, so these small down deltas do not support
a down-kernel choice. Retain the clear up result only. n16 alone has similar
up savings. pipe4 alone regresses up by about 3 microseconds.

The wider variant changes the up grid from 32 to 64 blocks. That is a
plausible explanation for the latency change, not a measurement of achieved
occupancy or bandwidth. The seam holds one real input fixed and substitutes
routes through actual weights. It does not reproduce forty layers' working
sets or their evolving activations.

Even transferring n16_pipe4's savings unchanged to every layer projects only
0.0822 seconds per turn, or 3.63%. Retain it as a measured candidate for a
later combined trial; it does not earn a full model load by itself. No
serving or runtime default changed. Next, split products and selection in
the remaining router cost before choosing a kernel change.

The [window receipt](evidence/expert-tuning-20261019/r1/window.json) passes.
Minimum controller-sampled free memory was 15963/15499 MiB, from only two
samples. GPU 0 was not sampled while holding the short-lived fixture; the
in-process allocation and sample checks enforced the floor. These minima
are not a sustained under-load memory qualification. Production was restored
and the local proxy returned exact READY at 16:15:53.606914 UTC, request
45be6335-abf8-4e5a-aa97-d94be56bba22. Clef was unchanged, cleanup and sampler
errors were empty, and both owned controller and child processes ended.

## Router split preparation

The next proof changes no kernel. It separately times the existing products,
selection and combined two-kernel graphs using the actual first-layer router
and six captured native input/pick/weight rows. Each correctness row must
match the saved IDs and FP32 route weights exactly and produce 257 finite,
repeatable scores. A poisoned-score negative control must fail.

Timing uses position zero with fixed inputs and a warm, roughly 1 MiB router
matrix. The three graph types perform 128 route repetitions, with 128 nodes
for each component or 256 for the combined path. An excluded warmup and five
repetitions alternate stage order. Every timed output is checked after the
events. This is an arithmetic diagnostic with warm L2, not a model bandwidth
measurement. Its purpose is to identify which component can plausibly save
enough complete-turn time to deserve a candidate kernel.

## Router split result and candidate

The [router split](evidence/router-tuning-20261019/r1/analysis.json) passes all
six saved rows on both GPUs. Combined median is 71.738/71.760 microseconds;
products are 6.158/6.144 and selection is 65.587/65.666. Selection therefore
dominates this measured seam. Leave products unchanged. Nemotron was restored
with exact local proxy READY at 16:22:52.522906 UTC, request
16028f22-205c-412d-842f-874f1855a7a3; Clef was unchanged.

Compare per-candidate ranking over all 256 scores with eight hierarchical
warp/CTA argmax rounds. The latter avoids a full 256-by-256 comparison matrix
and is the selected candidate. Carry score bits and original index together;
break finite ties toward the lowest index. Keep the existing ordered softmax
and shared-gate calculation. Any nonfinite expert score uses the copied
legacy serial selection body, preserving its exceptional-value behavior.

The falsifiable screen is at least 15 microseconds less combined route time
on each GPU with all five pairs improving, exact captured outputs, and exact
old/new outputs for planted ties, signed zeros and exceptional values. At
40 layers and 191 appends, a 15-microsecond reduction projects 0.1146 seconds,
just over the 5% full-turn gate. This remains a projection until the matched
full changing-token/position graph trial passes. The new symbol is optional
and no serving path selects it during the thin proof.

## Parallel selector proof

The [candidate operator proof](evidence/router-tuning-20261019/r2/analysis.json)
passes. Both GPUs match all six saved native rows and all ten planted
tie/zero/Inf/NaN controls byte for byte. Every measured pair improves.
Combined route median is 71.729 to 10.026 microseconds on GPU 0 and 71.752
to 10.010 on GPU 1. Candidate combined ranges are 10.000-10.034 and
10.004-10.145 microseconds, five samples per arm after an excluded warmup.

Selection alone is approximately 65.6 to 3.94 microseconds. GPU 0 has one
16.91-microsecond candidate selection outlier; it remains in the raw samples
and reported range. The combined path is the relevant materiality screen.
The demonstrated limiter was the legacy single-thread selection code; this
does not establish physical memory-bandwidth utilization elsewhere.

Transferring the combined savings unchanged projects 0.472 seconds per turn,
or 20.8%. This earns a matched full changing-token/position comparison.
Capture baseline and parallel graphs separately over the same session
allocations, and verify the selected function occurs in every layer. Then
compare the parallel graph with an additional n16_pipe4 up-only graph;
down remains baseline because the tiny down deltas were order-sensitive.
Require exact untimed consumed rows and timed tokens/final state, all work
counters, five alternating pairs, and actual in-process memory readings.
Independent numerical qualification and serving integration remain open.
