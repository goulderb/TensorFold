# Frozen numeric operands and norm reduction

The isolated replay reproduces saved native Q/V and norm outputs on both
internal GPUs. It closes the activation, packed-weight and residual-operand
uncertainty at these seams. The independent full-model HF quality gate remains
FAIL; no threshold, model arithmetic or production setting changed.

The executable is frozen at source
`365e5d28b2022236825b30c2e2c443a34eeb9997`, SHA256
`e3dc2ca97227577b8481920b7974233f2ef56ef4ca064198855eb3d38f5cf2a5`.
The fixture SHA256 is
`aa83901efbe7841c717424f217e3c8d9cd1d5aa0c94bc0299c8b74f15ac248ed`.
It uses the frozen production operators and never loads the complete model.

The strict analyzer independently regenerates FP8 activation codes and their
fragment order, full Q/V packed layouts, BF16 residual addition and FP32
gamma-plus-one. All 28 downloads and 38 journal comparisons agree with those
controls and the saved native outputs. Seven checkpoint tensor ranges,
six images, helper sources, input, executable, phases and GPU identities are
checked. Thirteen mutation tests pass, covering corrupted operands, bindings,
identities, missing comparisons, memory and restoration failures, and false
quality acceptance.

The HF differences reproduce on each GPU: two Q elements (3950, 4766), one V
element (77), and one norm element (304) under both frozen norm bindings.
Counts are elements, not the journal's byte counts. This is a same-input
replay, not a capture of historical full-model live pointers.

A separate CPU emulation reproduces all 2048 BF16 norm outputs for both
bindings on both GPUs. It follows the CUDA kernel's ordered FP32 FMA,
256-thread vector accumulation and XOR reductions, then explicit rounded
division/square root and multiply order. The predicted mean square is
1.2590551376342773; pinned Torch CPU produces 1.259055256843567.
Using Torch rsqrt with the native mean changes no BF16 outputs; using Torch's
mean with explicit sqrt/div changes the single coordinate and matches saved
HF. This attributes that local norm discrepancy to reduction arithmetic.
Predicted intermediate values come from an emulation validated against all
downloaded outputs; they are not GPU register captures. Q/V accumulation and
the full-model quality difference still need independent qualification.

The controller restored Nemotron at 19:57:00 UTC, PID 8352, canonical READY
`d42cbe5c-821d-4577-a814-81a115cc2535`. Clef was unchanged, owned jobs ended,
and cleanup/sampler errors were empty. Operator minimum free memory was
14983/15035 MiB, above the 1536 MiB floor on each GPU.

The [archive](evidence/upstream-numeric-operands-20261019/archive.json) contains
the raw journals, small operands/results, source/build pins, strict analyzer,
mutation tests, norm emulation and coordinate records. Large identical
raw/packed weight copies are separately pinned by size and SHA256; live
reanalysis also requires the original checkpoint ranges, saved traces,
executable and images listed in the inventory. This archive is not a model
package or promotion receipt.
