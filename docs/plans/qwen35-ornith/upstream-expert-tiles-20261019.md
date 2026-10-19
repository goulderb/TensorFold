# Expert tile screens after the upstream port

The two-warp N16 pipe4 candidate is rejected. Its outputs matched across all
32 retained expert banks on both GPUs, but it was slower than the existing
four-warp N16 control beyond their observed ranges.

| Up projection | RTX 5080 median | RTX 5070 Ti median |
| --- | --- | --- |
| WN4 baseline in the WN2 comparison | 53.6048 us | 53.4243 us |
| WN4 N16 pipe4 control | 42.8765 us | 42.9182 us |
| WN2 N16 pipe4 | 43.2742 us | 43.2920 us |

Each comparison used one excluded warmup pair and five alternating measured
pairs. WN2 ranges were 43.2690 to 43.2820 us and 43.2862 to 43.3125 us.
Control ranges were 42.8533 to 42.9000 us and 42.8758 to 42.9422 us.
The two candidate comparisons are independent. Their samples are not matched
against each other.

The projected WN2 saving was 0.078148 seconds per 191-append turn, or 4.535%
of the accepted GDN baseline's 1.7232142 seconds. The projection multiplies
each GPU's median paired saving by 20 local layers and 191 appends, adding
the serial stages once. It assumes the layer-zero saving transfers to other
layers. This is an operator screen, not a measured full-model gain or physical
bandwidth measurement. It fails both the 5% materiality gate and the known
control comparison.

The candidate used 162 registers, no local memory, 128 blocks and 64 threads
per block. Its calculated resident-block limit was 6 per SM. That is not
achieved occupancy. Every captured graph's 128 nodes was checked against the
selected function and launch dimensions. The timed graphs rotate all 256
experts across 32 banks. Saved bank-zero output and all-bank finite, exact
outputs passed. The packed up weights and scales span 301,991,936 logical
bytes, which must not be relabelled as measured DRAM traffic.

The experiment source is `4a13c64a68ed47e1a9b9ac606f14c334312f5dc1`.
The active diagnostic replaces WN2 with a separate N8 hypothesis. WN2's source,
binary pins and original evidence remain recoverable. N8 tests reduced
per-warp fragments and more total warps, rather than regrouping the same
256 warps. Its outcome is pending.

The controller restored Nemotron PID 22716. Canonical request
`4fce8123-9bb0-4122-aba8-edae067dd123` returned `READY` at 18:53:58 UTC.
Clef was unchanged. Controller PID 14944 and the probe exited without cleanup
or sampler failures.

`evidence/upstream-wn2-20261019` preserves 171 original files, including source
snapshots, static CUDA image checks, build pins, controller receipts, raw
journal and analysis. Verify its archive with the existing archive reader.
The active analyzer retains WN2 support so the rejection remains rerunnable.
