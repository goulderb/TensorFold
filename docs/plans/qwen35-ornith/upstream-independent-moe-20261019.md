# Fixed-input execution of the original HF MoE

The strongest next quality seam is the shared activation's missing BF16
rounding boundary. This investigation ran the unmodified pinned HF layer0
MoE on six saved native inputs. It changes no native arithmetic or reference
forward. The independent full-logit quality gate remains FAIL.

The earlier native-style full-reference experiment replaced HF expert,
MoE and attention semantics and still failed all four logit rows. Its
first-difference locations describe that aligned diagnostic reference,
not the unmodified HF oracle. The current experiment instead executes
the original pinned SparseMoeBlock, TopKRouter, Experts and MLP forwards,
with the existing unchanged lazy checkpoint operand adapters.

Source/admission checks, a call profiler and read-only stage hooks verify
six calls to each original forward and 168 calls to the original
Store.linear code. All 372 successful raw artifacts retain shape, dtype
and SHA256, and accessed checkpoint tensors retain payload/source pins.
This uses the same CPU Torch and quantization adapter as the original
oracle; it is a single-block semantic comparison, not another qualified
serving runtime.

| Saved position | Different MoE BF16 elements /2048 | Local RMSE | Different shared activation elements /512 |
|---|---:|---:|---:|
| 0 | 1285 | 0.0000705169 | 147 |
| 1 | 2006 | 0.000717891 | 140 |
| 2 | 1118 | 0.0000212211 | 134 |
| 3 | 1269 | 0.0000266244 | 151 |
| 4 | 1307 | 0.0000184244 | 130 |
| 5 | 1207 | 0.0000261582 | 150 |

Shared gate/up projections match native exactly for all six rows. Native
activation computes BF16(silu_FP32(gate) * up); the HF tensor operations
round the SiLU result to BF16 before multiplication. The shared down
output then differs in 963–1488 elements. This gives a concrete first
shared-branch dtype boundary to test with an isolated CUDA candidate.

At position1, HF selects expert231 where native selects72, with both
BF16 router logits equal to -4.21875. Position5 preserves the selected
set but changes tie order. A single observed CPU topk tie outcome is not
a universal policy requirement. The fixed oracle nevertheless remains
unchanged; the experiment does not hardcode a selected expert.

A separate standard-library analyzer reconstructs every native and actual
HF expert/final output exactly. Its diagnostic counterfactual combining
HF casts/order with the actual HF shared output leaves two differing
elements at position0, none at positions2/3/4/5, and 1998 at position1.
The latter retains native picks and the cutoff tie. Routing-weight casting
alone changes little. These counterfactuals inject saved intermediate data
to isolate branches; they are neither executable candidates nor quality
acceptance. Root replayed the checker and reproduced byte-identical analysis
SHA256 `d323a19e147b33c194d816260fa15fda222eefe36b68494847bfedb9f520d07f`.

Next: qualify a renamed shared-activation candidate with one explicit BF16
round after SiLU. Compare all six 512-element outputs against actual HF,
with the frozen native kernel and a matched control. If it differs, inspect
intermediates rather than adding another change. Other HF cast/accumulation
boundaries and routing require separate qualification. Only then is a
math-changing shared-Engine variant justified for the unchanged full-logit
oracle. Historical native anchors remain the control; fitted MMA emulation
and intermediate output injection cannot replace that oracle.

The first two CPU attempts are rejected and preserved. Their optional
FLA/Triton import path initialized the CUDA runtime despite hidden devices.
No numerical GPU work was requested, but no zero-GPU-API claim is made.
The successful third run blocks CUDA lazy initialization before import;
FLA's existing exception handler chooses CPU and the original numerical
forwards execute on CPU. Admission and completion report CUDA uninitialized.
This does not trace all possible driver availability queries. Actual Torch
is 2.9.0+cu128; the capture interpreter resolves it from the existing Tabby
runtime, rather than a CPU-only build. All three processes ended. The
successful run used four compute threads and one interop thread.

Canonical health after the rejected imports passed at 22:31:09.099196 UTC,
request `8223cbd7-b59d-4dd2-9d41-cba8fc481da3`, with Nemotron READY,
5572/5815 MiB physically free and Clef unchanged. No service was stopped
for this CPU investigation.

The verified [evidence archive](evidence/upstream-independent-moe-20261019/README.txt)
contains 549 members, 1,723,078 compressed bytes, SHA256
`1472ab3a7022aa00da1add15b872c753de7367f0589e744d7c24516b0a48e7f4`.
It retains successful and rejected raw captures, source/admission/call
proofs, scripts, local reconstructions, licenses/notices and health evidence.
The separately pinned 1,140,149-byte CK suite journal is already preserved
in the earlier operator archive. Runtime libraries and checkpoint weights
remain external inputs. This evidence qualifies no full model, capacity,
serving or promotion claim.
