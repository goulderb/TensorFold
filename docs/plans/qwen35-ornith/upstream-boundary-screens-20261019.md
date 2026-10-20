# Terminal selection and HF activation boundary

Keep host greedy selection on the current head GPU. The existing shared
device argmax passes semantic/output checks but regresses on that GPU.
Separately, the isolated shared-activation candidate matches all six saved
HF rows exactly on both GPUs. This is a verified local semantic repair;
the independent full-model HF gate remains FAIL.

Both trials use source930cf50a84f3820bd9b96a29f851ed53a0c7c9cb,
profile SHA2565eb31e5b95304c56b1f15701d7d63ddea8f729048da5ff02125d64bc611e4989,
and the unchanged93 embedded images. The tagged boundary_screen admits
exactly one of terminal_greedy or shared_activation, excluding every other
diagnostic. The final coordinated build passed55 host tests. All30 greedy
and36 activation CLI cases passed. Root union tests passed6 methods;
independent greedy checks passed8 methods/36 subtests, and activation checks
passed11 methods/16 subtests. Both protected preflights passed without outage.

## Actual terminal path screen

The screen uploads128 distinct full native BF16 logit rows outside timing.
Each arm processes them sequentially. Host selection downloads496640 bytes,
synchronizes, makes the source-exact diagnostic copy, then scans the hot
persistent source row. Device selection launches the existing tree-round
BF16 argmax, downloads an eight-byte Pick, synchronizes and validates status
and token. Both consume every result. Engine cache-logit and Session-logit
copies are excluded. The pinned staging buffer is an isolated seam; it is
not the complete live pipeline or HTTP path.

Independent source review caught the initial scan of the diagnostic
destination before device testing. The worker changed only that operand to
the persistent hot source and added a mocked Buffers.row regression. The
old source and red/green evidence remain preserved; no trial ran the old
baseline. Source/runtime counts, fixture, candidate and semantics were
unchanged. The corrected source is committed and its clean executable frozen.

One warmup and five alternating measured pairs each sweep all128 rows.
Every real token agrees with the source-exact host comparator. Independent
ties, signed zeros, NaN and both infinity probes pass, including combined
nonfinite masks. Raw copied rows, Picks, poison and guards pass. Direct
launch argument receipts are source-level request evidence, not driver ABI
readback. Wall-clock endpoints are not archived; samples and exact summary
arithmetic are checked. CUDA event spans include host gaps and are not
kernel time.

| GPU | Host median/range ms per128 | Device median/range ms per128 | Paired wall saving median/range ms |
|---|---:|---:|---:|
| RTX5080 |33.5562 /33.3761–33.6157|25.0529 /23.7706–25.3308|8.4875 /8.0453–9.7856|
| RTX5070Ti |33.5360 /33.4685–33.6356|39.6371 /39.5738–39.7006|-6.1011 /-6.1613–-6.0367|

That is approximately262.16 to195.73us/draw on GPU0 and262.00 to309.66 on
GPU1. All five pairs favor device selection on GPU0; all five regress on
GPU1. The current20/20 pipeline places the head on GPU1, so do not select
this device path. The hardware/driver limiter is unmeasured. Reuse and
inspect existing upstream reductions before another kernel design; possible
placement changes require their own exact state/output and memory checks.
No full-model speedup or slowdown is measured by this seam.

The root audit passes490 records and180 raw files with exhaustive coverage,
all source/build/image pins, memory census and controller recovery. Controller
25360 and profile8132 ended. Nemotron restored as PID1588 at
22:51:57.083744UTC; canonical request
008be5f8-4b29-4e2f-b5e7-378db3248609 returned READY. Clef remained unchanged.
Restored physical free memory was5572/5815MiB.

## Exact local HF activation candidate

The earlier unmodified-HF block experiment found identical shared gate/up
projections and different activation outputs. The candidate adds exactly
one BF16 round after SiLU before multiplication by up. A matched new-image
control preserves the original expression. The frozen embedded activation
is another control; a separate recorder emits FP32/BF16 intermediates.
Expected HF values remain host comparison data and are never candidate
device operands.

All six512-element candidate rows match actual unmodified HF outputs exactly
on both GPUs:6144 checked candidate elements. Frozen original and matched
control remain byte-exact to saved native outputs. Recorder output parity,
input immutability, guards and finiteness pass before interpreting planes.
Actual captured graph checks cover function identity, geometry and every
argument byte span. Control/candidate use16 registers, recorder30, with no
stack/spills. No performance was measured.

Kimi independently validated the original-HF evidence, fixture lineage and
one-expression candidate. Image preparation, helper and executable commits
are separate inputs. Root rechecks the CUDA source/header/catalog at all
three commits: all nine historical source objects match the image snapshot.
The helper's SHA also matches the final executable source. This closes the
provenance question without requiring equal commit IDs or rebuilding an
unchanged kernel. The separate Kimi review/response remains preserved.

The full activation audit requires both GPUs and exact candidate equality.
A hypothetical early mismatch would retain raw artifacts but produce no
passing audit; partial forensic analysis would be manual. This actual run
passes158 records and84 raw files with exhaustive coverage. Controller9220
and profile23832 ended. Nemotron restored as PID32308 at22:53:48.241216UTC;
canonical request19b3ed91-e6fe-4329-a77e-0f9a60d41e32 returned READY, with
5572/5815MiB physically free and Clef unchanged. Neither trial reported
cleanup errors. The physical census in each retained controller stays above
the1536MiB floor on both GPUs.

The [evidence archive](evidence/upstream-boundary-screens-20261019/README.txt)
contains488 members,6856742 compressed bytes, SHA256
114776b3e0b3c12a0e7746c12578424190f6b208bf7af37a72fe278046f012ca.
Twenty-five larger files/runtime binaries remain externally pinned,
1303775368 bytes. It retains source/build/fixture proofs, independent audits,
the pretrial hot-buffer correction, Kimi review and both recoveries.

The activation result justifies carrying this semantic boundary into a
qualified shared-Engine variant. Other MoE cast/accumulation and routing
differences remain. Original full-logit oracle, thresholds and historical
native anchors stay unchanged. This is neither full-model accuracy nor
serving/capacity/promotion acceptance.
