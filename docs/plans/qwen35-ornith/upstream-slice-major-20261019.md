# Slice-major expert-up layout

Retain this small positive candidate for the compatible batch. All thirty
measured event pairs and all thirty wall pairs favor the slice-major layout,
with exact isolated outputs. The full-model payoff remains unmeasured and
production representation ownership is not implemented.

Within each 144-u32 block, the candidate moves only the first 128 words:
`new[32*slice+lane] = old[4*lane+slice]`. The final sixteen scale words,
gate/up interleaving, MP=2 group pitch, global scales, MMA/FMA traversal,
gate exchange and SwiGLU rounding are unchanged. Fixed-slice warp addresses
touch four rather than sixteen 32-byte sectors; all four warps together
still touch the same sixteen unique sectors. This does not demonstrate a
fourfold DRAM reduction or identify the hardware limiter.

The isolated screen uploads original and transformed copies of one real
288 MiB layer0 up tensor into separate allocations. Bank0 retains its actual
captured route/input and saved output; the other31 banks substitute experts
to cover all256. The new representation is explicitly tagged and never
bound to the legacy full-width, ordinary N8 or prompt consumer. A matched
new image contains original-layout control and slice-major candidate. The
frozen split-M image remains an untimed output control. Control uses112
registers and candidate110, both with2048 static shared bytes and no stack
or spills. Identical arithmetic source/opcode counts are evidence of the
intended change, not SASS equivalence or a bandwidth diagnosis.

Trial source is `d4814e94237fc33e6b0d8a2756d1afd4a364a577`, profile SHA256
`570205f434e95f5f13656efb3d2bce0e5b7a660823318e595d2d34277928b37b`,
input SHA256 `fc3dc044e03bf4a552248f2111e560e1a5e4150e9a97f56ee11391601c338783`.
The coordinated build passed46 host tests and30 CLI admission cases. The
fixture/source validator's six methods check every524288 block, inverse
hash, all8388608 scale words and source/build provenance. The original
independent analyzer passed84 tests before device execution. Fable found
no blocker and verified the mapping, ABI, buffers, counts and source pins.

All rows1/2/16, all32 banks and all three arms qualify before any timing.
Every active output is finite and byte-exact, including the independent
bank0 anchor and18-distinct-row/count17/stride/skip controls. Guards, plan,
input and weights remain unchanged. Graph readback checks each actual
function, geometry, fourteen argument spans, bank/layout addresses and
serial dependency chain. Driver resources retain128 actual CTAs per arm.
Upload-to-pointer provenance partly relies on pinned source inspection;
the independent raw-output and address checks provide additional controls.

One warmup pair precedes five alternating measured pairs per shape/GPU.
Each sample replays128 serial bank-rotating calls. Raw timing is retained
with complete final-bank output outside timing. Event microseconds/call:

| GPU | Rows | Control median | Candidate median | Paired saving median (range) |
|---|---:|---:|---:|---:|
| RTX5080 |1|28.748751|27.472500|4.5976% (3.9115–4.6292%)|
| RTX5080 |2|28.741000|27.471749|4.3683% (4.2796–4.7249%)|
| RTX5080 |16|30.485000|28.742749|5.7151% (5.6329–5.7783%)|
| RTX5070Ti |1|29.086500|28.031999|3.6925% (3.5755–3.7321%)|
| RTX5070Ti |2|29.122250|28.019750|3.7858% (3.3004–4.0104%)|
| RTX5070Ti |16|30.805999|29.203249|5.1812% (4.9365–5.3147%)|

Homogeneously projecting twenty copies of this layer per rank yields
46.615us saved per one-row body call and66.900us per16-row body call.
For127 subsequent appends plus four prefill chunks, that is6.187710ms,
or0.524926% of the qualified1.1787787s bundle turn. This conditional
projection assumes additive savings without routing/cache/overlap or new
loader costs. It is not an upper bound or measured model improvement.
Do not add it to historical percentages. Production integration needs an
explicit format contract for every consumer and matched full turns.

The first post-trial checker rejected its derived per-call event value:
the helper computes in FP32, while the checker used FP64. The original
checker, its ten pretrial pins and raw data remain unchanged. An explicit
r2 checks `f32(f32(event_ms*1000)/128)` exactly, without epsilon or ULP
tolerance. All other gates remain unchanged. Its94 tests include exact
regression, nonfinite and one-ULP corruption rejection. Both independent
full audits pass4102 records and1280 raw files totaling1272273408 bytes;
the final output agrees with root's separate run. The revised approval
adds four pins to the original ten and records the post-trial reason.

Controller4820 and profile21076 ended. Restoration passed at
22:36:30.084714UTC, Nemotron PID21064, canonical READY request
`2bb08ae3-beed-4a87-a2e6-2db3dd39f730`. Clef PID21392 was unchanged.
Six physical memory samples stayed at or above15158/15211MiB free; after
restoration the GPUs had5572/5815MiB free. No cleanup errors were recorded.

The [verified archive](evidence/upstream-slice-major-20261019/README.txt)
contains1430 members,39558714 compressed bytes, SHA256
`593626f95745b65a26819e81cdf4599a95fb4898396720aee0d99f4498dc189f`.
Eleven larger files/binaries remain externally pinned,1543366255 bytes.
It preserves source/build/fixture proofs, Fable review, both scalar-checker
versions and original approvals, complete small readbacks and restoration.
This is an isolated operator result. Independent HF full-model quality
remains FAIL; capacity, serving and promotion remain unqualified.
