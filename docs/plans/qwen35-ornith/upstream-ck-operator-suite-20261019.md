# CK pipeline depth and FP32 capture screens

Stage depth three produced no useful operator gain. Keep the original depth
four selected and continue to the isolated slice-major expert-up candidate.
The separate FP32 diagnostic distinguishes the remaining local accumulation
models; it does not change the independent full-model quality verdict.

The retry ran source `b67e0fe3579a88e9d97fe8c34cd5f88114fdceb0`, profile SHA256
`6f23c2e58e4318c00a8040bb86963eb924ef03092d3b32b85902a44b20f28eaf`,
with the unchanged 93 embedded images and separately pinned stage-three and
instrumented MMA images. One shared Engine remains. Both screens are
isolated diagnostics; no complete model was loaded or timed.

## Failed metadata check and bounded repair

The first attempt at `62a2a679878e5d57e6949d0fd6ce009d408fc211` stopped before
graph instantiation, timing or MMA capture. Actual function, geometry and
arguments matched, but the harness expected default cluster policy zero.
The two attribute API calls succeeded; the predicate failed. No performance
result follows from that attempt.

A metadata-only probe then captured 24 combinations of GPU, original/candidate
function, split-K 2/4/8 and omitted/explicit default policy. All graph
dimensions were exact and all graph policies read back SPREAD (1), while the
function default remained zero. The probe instantiated and executed no graph.
This observes driver 610.88 behavior; it is not a general CUDA API promise.

The repair checks the observed policy one strictly and records both 64-byte
raw attribute values before validation. Corresponding member bytes must match
the semantic fields; unused union bytes remain opaque. Launch configuration,
kernel arithmetic, images and numerical checks did not change. The original
analyzers and first-attempt artifacts are retained. Fable's earlier inference
that a default request implied zero readback was disproven by this probe.
Dimensions are independently read back; policy one is a request-independent
fingerprint of this driver. A driver update may trip this fail-closed screen.
The omitted-default request itself is a source-level fact.

The repaired build passed 40 host tests, all 27 CLI admission cases and the
fleet fast gate. Component tests passed separately: six union tests, fifteen
stage methods with 47 subtests, and 36 MMA tests. The stage worker's combined
unittest run additionally includes three metadata-probe tests (18 methods).

## Actual stage timing

Each GPU loaded 26 actual FP8 projection banks from eight layers, totaling
255,852,544 packed bytes (244 MiB), exceeding twice each measured L2 size.
Each mixed pass visits every bank. This footprint does not prove every load
missed cache. Both arms reuse identical prequantized operands. The comparison
changes only pipeline depth four to three and associated dynamic shared
storage (20,480 to 15,360 bytes), with matching source/compiler provenance.

Every graph attests 104 serial kernel nodes, exact function, geometry, ten
argument spans, readback cluster dimensions and that driver policy fingerprint.
All saved active outputs match
the serial references, and guards and encoded inputs remain intact for rows
1/2/16. Every timed sample also checks outputs outside timing. One warmup pair
precedes five alternating measured pairs per shape/GPU. The table reports
event microseconds per 26-bank mixed pass; wall timings are retained separately.

| GPU | Rows | Depth 4 median (range), us | Depth 3 median (range), us | Median paired saving, us |
|---|---:|---:|---:|---:|
| RTX 5080 | 1 | 402.896 (402.312–403.584) | 405.168 (404.896–410.256) | -2.712 |
| RTX 5080 | 2 | 405.584 (404.288–405.832) | 406.064 (405.784–407.096) | -1.264 |
| RTX 5080 | 16 | 419.392 (418.320–423.696) | 421.056 (419.760–426.368) | -1.656 |
| RTX 5070 Ti | 1 | 428.064 (427.392–428.120) | 428.016 (427.720–432.672) | -0.328 |
| RTX 5070 Ti | 2 | 431.592 (430.568–432.600) | 431.304 (431.080–436.688) | +0.488 |
| RTX 5070 Ti | 16 | 446.440 (445.832–446.536) | 447.080 (446.288–447.832) | -0.544 |

This is an aggregate mixed-operator screen. It establishes no full-model
gain, per-shape cold-memory attribution or hardware bandwidth wall. The
candidate provides no repeatable useful saving across the intended path.
Reduced shared storage did not yield the hypothesized gain. The exact
hardware limiter remains unmeasured; restricted counters were not retried.

## FP32 accumulation discriminator

The same window ran frozen original, same-image control and instrumented
MMA functions on Q and V fixtures. All 8,704 BF16 elements per arm match the
saved native outputs on both GPUs. Output/trace guards and finiteness pass.
Source normalization removes only the explicitly permitted trace stores;
the control and instrumented functions each compile with 40 registers and
zero stack/spills. SASS equivalence remains unverified. Binding records
attest launch requests, not independent graph/cluster readback for this screen.

For Q coordinates 3950, 4766 and 3245 and V coordinate 77, the independent
integer/Fraction predictor compares four predefined accumulation models.
The intersection across all four coordinates on both GPUs contains only
`mma32_exact_rz_fused_acc`. Q3950 and Q4766 each admit two models and V77
admits all four; Q3245 carries the discrimination.
At Q3245 both observe partial bits `[3305180512,3316962952]`, joined bits
`3321191736`, post-alpha bits `3205985419` and BF16 bits `48919`. This separates
the former two surviving RZ models despite their equal BF16 output.

This is evidence for four covered coordinates. It does not prove a universal
MMA rule, the unobserved original FP32 values, or localization of all
full-network divergence. The calibrated predictor is not an independent
replacement for the pinned HF oracle. Its full-logit gate remains FAIL.

The first combined post-trial audit rejected the integrated input because
the immutable MMA numerical checker still expected standalone field names.
A new hash-enforced adapter validates the exact integrated suite schema and
constructs the standalone view for the unchanged checker. Its 92 new tests
pass with no skips. The original pre-trial approval remains byte-identical;
the explicit post-trial audit revision adds five pins. Six root union tests
pass. The final audit verifies 1,968 ordered records, 432 raw files, 2,568
historical source objects, all 93 images, controller provenance and exhaustive
disjoint component coverage. Raw journal SHA256 is
`8fc1f2063f045dea6db0035e317af76590ac8df056d73d3358b755393838210a`.

## Recovery and evidence

Controller 24824 and profile 29204 ended. The controller passed with no
cleanup errors and nine physical memory samples, minimum 11,938/11,991 MiB
free while the operator window ran. Nemotron restored as PID 28580 at
22:16:52.407571 UTC; canonical request
`cef58fa2-8097-4203-a244-f5af73b0f7b6` returned READY. Restored physical free
memory was 5,572/5,815 MiB. Clef PID 21392 remained unchanged. CUDA context
memory-info readings from the metadata probe are not physical global headroom.

The [evidence archive](evidence/upstream-ck-operator-suite-20261019/README.txt)
preserves both attempts, the metadata probe, source/build pins, analyzer
revisions, raw small readbacks and independent predictions. Larger files and
runtime images/executables remain separately pinned external inputs.
The ZIP contains 1,008 members, 27,661,056 bytes, SHA256
`c0cd88b8c64f5bc48922bb9a800b3c72058bf8eaebf506bab3c9692091235413`.
The archive verifier passes, all fifteen external files (97,730,693 bytes)
rehash unchanged, and the post-archive historical audit passes again.
The separately retained [Fable follow-up](evidence/upstream-ck-operator-suite-20261019/fable-r2.json)
reproduced the audit byte-for-byte and found no blocker. Its two wording
qualifications are incorporated above. The original ZIP remains unchanged.

No production configuration changed. The next exact tuning candidate is
slice-major code-word packing in an isolated split-M consumer. Device greedy
remains a smaller retained candidate. Quality, serving/capacity and promotion
verification remain open; no production qualification follows from this suite.
