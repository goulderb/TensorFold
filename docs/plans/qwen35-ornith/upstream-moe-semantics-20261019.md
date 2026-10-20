# Expert casts and accumulation order

The isolated expert arithmetic candidate passes on both GPUs. It matches
the original HF expert outputs on positions2/3/4, whose actual input
boundaries independently qualify. Every recorded cast/product/accumulation
plane matches the pinned expectation. This is a local semantic qualification.
The independent full-model HF gate remains FAIL.

The candidate consumes only saved native FP32 expert pairs, native routing
weights and native picks. It rounds route weights and down outputs to BF16,
rounds each product to BF16, then accumulates with BF16 rounding in ascending
selected expert-ID order. It does not receive HF output data as operands.
The candidate and separate recorder produce identical expert results.
Frozen native and newly compiled, textually identical native combine controls
remain exact against all six captured native full-MoE outputs.

The original HF forward and all372 saved artifact hashes are unchanged.
All138 fixture payloads verify. At positions2/3/4, the BF16-rounded native
down outputs and routing weights equal the original HF inputs byte-for-byte.
The resulting6144 expert values per GPU,12288 total, match original HF exactly.
Omitting any single intended cast or the expert iteration order changes
628–1227 outputs on these eligible rows. Kimi independently recomputed all
these lineage, eligibility and discrimination claims before the device run.

Positions0/1/5 remain diagnostics. The pinned native-input candidate differs
from original HF expert outputs by2/2008/0 elements respectively. Position0
has upstream expert down differences; position1 has a changed selected set
at the cutoff tie; position5 has one different down input even though the
final expert output happens to match. Equal final bytes do not qualify an
unequal input boundary. Shared-down recomputation and final shared combine
are separate later gates.

## Frozen execution and independent audit

The source is478e054998bc818e5d675ac272815807bc8fd67f. Profile SHA256 is
18b1adcf2628d972ce3e7445e2232ce5ffb5fd4fee7a8e0b35a1cca76d6712a7.
The frozen build pins every tracked source and93 embedded CUDA images,
including both new helpers. The earlier developer CPU runner's five-file
pin list is narrower; it is not the protected executable's provenance record.
All65 host tests and36 arithmetic CLI admission cases pass.

The new standalone image SHA256 is
91b6af363437672e987fed34810d1d591c4978acbb4521872599b4e0b929309a.
Matched control/candidate/recorder use28/29/40 registers, no stack or spills,
and launch eight256-thread blocks. The original generic resource parser
incorrectly supplied a128-thread request default. Its original receipt is
preserved. A separately pinned resource-contract.json corrects that metadata
to256 without changing the image. Actual captured geometry is independently
checked at runtime. The frozen original combine reports28 registers on both
GPUs; its exact count lacks a separate pretrial disassembly witness.

Preparation and executable commits differ, but all six relevant historical
CUDA source/header/catalog objects match byte-for-byte. Both helper objects
join the complete frozen build roster, and all120 recorded compilation
dependencies rehash. The sibling builder's replay hash is retained as an
additional regeneration input. Future regeneration needs fresh pins.

The independent file audit checks all156 guarded raw files, including all
intermediate planes, native inputs and final outputs. The joined metadata
audit checks48 captured nodes, every argument byte/value/hash, selected
function identity, geometry, pointer aliases and distinct guarded spans.
Its17 methods/98 subcases pass. The separate file checker passes nine
methods. Root's outer audit accepts all230 journal records with exhaustive
unique record/file coverage, complete source/image/executable joins and
protected recovery. No performance is measured by this screen.

Actual node checks attest captured kernel parameters before instantiate.
Source-to-upload pointer ownership remains frozen-helper provenance.
CUDA free-memory values are observations and virtual admission checks.
The short controller trial has one physical sample,15963/15982MiB free;
this is not sustained under-load or full-model capacity qualification.

Controller28740 ended and the profile exited0. Nemotron restored as PID15748
at23:18:37.254110UTC. Canonical local-only request
1ab9c436-ac19-46a2-8a74-5833d10c5577 returned READY, with5572/5815MiB
physically free. Clef remained unchanged and cleanup reported no errors.

The [evidence archive](evidence/upstream-moe-semantics-20261019/README.txt)
contains1937 members,25898032 compressed bytes, SHA256
70789602d634705e18b8dfb6b48ff93aca83de5e1fadd73fca47795e79311ebd.
The19456-byte diagnostic CUDA image remains externally pinned. The archive
retains the original source and notices, fixture/lineage, rejected preparation,
resource correction, Kimi review/response, build, raw outputs and audits.
Its verifier passes. No oracle, threshold, native anchor, serving default,
capacity acceptance or promotion decision changes.
