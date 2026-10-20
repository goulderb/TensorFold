# Shared-engine GDN state layout

The persistent `[head,key,value]` recurrence layout reduced the matched
64-prompt/128-greedy turn by a median paired 7.343%. All five measured pairs
improved, with savings from 7.219% to 7.529%. This clears the 5% acceptance
threshold. ModelOpt `Chain.createSession()` now selects this layout. Explicit
`createSessionWithLayout(.value_key)` retains the baseline for comparisons.
The upstream MLX cache default remains unchanged.

| Metric | Baseline | Candidate |
| --- | --- | --- |
| Complete turn median | 1.8607677 s | 1.7232142 s |
| Complete turn range | 1.8588226 to 1.8614854 s | 1.7206772 to 1.7266284 s |
| Decode append median | 98.6963/s | 106.2375/s |
| Decode append range | 98.6119 to 98.7860/s | 105.9862 to 106.3903/s |

Both arms ran captured on one loaded checkpoint at source
`86eeaa78eaad449cb6b2a17b46f9df851f8e817d`. One alternating warmup pair was
excluded, followed by five measured alternating pairs. Each turn has 191
appends, including 64 prefill and 127 decode appends. Timings include host
argmax and consumed-logit copies. State download, canonicalization and hashing
occur outside timing. This is a synthetic single-lane full-engine comparison,
not HTTP performance, production qualification or measured bandwidth use.

The operator screen predicted 0.134707 s saved per turn. The full trial's
median turn difference is 0.137554 s. Coalescing persistent state access was
therefore a useful lever on this workload. It does not establish that the
remaining decode is bandwidth limited.

The full trial checked all 128 consumed BF16 logit rows and greedy tokens,
plus complete canonical recurrence, convolution tails, used head-major KV,
session lengths and logits. Eager and captured qualification passed for both
layouts. Capture inspection verified exactly 15 distinct GDN layer nodes for
each layout on each stage, including function handles and launch dimensions.
Graph identity includes the cache layout. Reset, injected stage-two failure,
poison refusal, uncommitted lengths, different-prompt interleaving and a new
candidate session followed by baseline replay passed.

Consumed logits retain SHA256
`b78bb49d36d6abcc509679be4673442594072d65f53b5f0998e1474ce0582839`.
Canonical state version 2 has 68,797,513 bytes and SHA256
`7fb71e4f454a6dd058d243337ddbcd4a189032aa44c35e3bc3289ff920acfd89`.
The independent HF full-logit gate remains failed and is not replaced by
equivalence to the retained implementation.

The controller sampled at least 5188/5885 MiB free on the internal GPUs.
Nemotron PID 27380 was restored and canonical request
`04a90a83-94b9-4b5d-9001-0e4273f33ff9` returned `READY` at 18:46:35 UTC.
Clef was unchanged. Controller and owned probe exited. There were no cleanup
or sampler failures.

The isolated expert screen passed parity but did not earn a full trial.
N16 pipe4 projected 4.39% combined turn savings. One-warp up improved less,
and one-warp down regressed. Those implementations remain diagnostic arms.
A two-warp N16 screen is a new test of CTA distribution. Dense quantization
reuse and real transport cost still need cheap measurements before declaring
diminishing returns.

Original receipts, journals, plans, build pins and analysis are preserved in
`evidence/upstream-operator-20261019` and
`evidence/upstream-layout-20261019`. Their ZIPs contain 20 and 22 original
files respectively. Verify each with the existing Windows evidence reader:

```powershell
python docs/plans/qwen35-ornith/evidence/windows-v110-20261019/verify_archive.py docs/plans/qwen35-ornith/evidence/upstream-layout-20261019
```

The full 63,569,920-byte consumed-logits payload remains local at the path and
hash in the archive index. Restore ZIP entries under `.artifacts/` to rerun
the strict analyzer with that payload and the pinned executable. The analyzer
first rejected its own incomplete expected output-file list. Its repaired
check requires both the journal and consumed-logits payload; numerical and
performance thresholds were unchanged.
