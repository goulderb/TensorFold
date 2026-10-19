# v1.1.0 shared-layer integration (2026-10-19)

## Selected boundary

The user selected the shared v1.1.0 layers with the fork's GLM implementation
retained. This is a development task, not a serving-source promotion.

- Upstream release: v1.1.0, commit `76aafcf27`.
- Fork base: `d29335ccd62caed69159f5c25bfcc77d96ca7a92` (`trunk` when started).
- Task: `task/glm53-v110-base`, `C:/Development/tf/glm53-v110-base`.
- Initial merge: `7b0982f7e`; its commit explicitly does not establish compilation.
  Follow-up repairs are committed alongside this record. That initial merge
  alone is not a usable candidate.
- The later `task/glm53-dual-hotpath` nine-commit series through `d7ad99ece`
  was subsequently ported here in the follow-up changes. Its earlier GPU results
  remain evidence for the earlier source only, not this v1.1.0 integration.
- The user authorized a local source merge to `trunk` on 2026-10-19. This does
  not authorize a serving promotion, production outage, push or publication.
  Before committing, all 4,975 candidate files were compared with both final
  verification snapshots below: only this integration record differed. Archive
  and full-log SHA256 values were rechecked against the retained artifacts.

Shared core, server, CUDA, build and `sys` platform changes come from upstream.
The entire GLM and Nemotron family directories were restored from the fork base
before adapting shared API callers: resolving only conflicted files retained an
inconsistent mix of unrelated family APIs. Nemotron is retained to preserve the
fork's TP and FP8-KV extensions; this does not change its production source pins.

### Explicit ABI differences

- Upstream EXL3 dense matrices use `qlinear.Weight.exl3_dense` and mutable
  `cuda.exl3.Dense`. Retained GLM matrices use `.exl3` and
  `cuda.exl3_linear.Kernels`.
- Upstream pointer-table experts use `experts.Format.exl3_common`,
  `Layer.exl3_common`, `Grouped.exl3_common`, and
  `cuda.exl3_experts_common`. Retained GLM's stacked experts use `.exl3` and
  `cuda.exl3_experts`.
- Upstream's `exl3_experts.cu` is preserved. The retained kernel is renamed
  `exl3_experts_glm.cu`, with its original flags and a distinct embedded image.
  This is an explicit compatibility boundary, not a claim of new authorship or
  a completed migration onto the common pointer-table implementation.
- Nemotron attention retains the fork's FP8/KV-scale launch ABI. Upstream
  page-table declarations and measurements do not qualify that source.
- Shared checkpoint-math operators retain the fork's dense FP8/NVFP4 extensions.
  Their changed launch occupancy and ported dense kernel require GPU checks.

Original notices and transitive donor credits remain in `THIRD_PARTY_NOTICES.md`
and `LICENSES/donors/README.md`. No publication or relicensing is authorized.

## Behavior reconciled during repair

Restored the complete upstream kernel registry and compiler sets, plus scoped
fork records; restored spill mocks rather than disabling spill tests. Server
stream admission uses the upstream `begin` hook, with no headers before
admission. The fork's required-call fallback, unknown-template refusal, and
slide gating remain. Native family admission must take precedence over generic
page-budget admission, and boolean versus numeric TP options remain distinct.

Upstream copy-source custody remains active, including pinned asynchronous
sources, failed-completion quarantine, and capture guards. Retained test drivers
must implement those APIs; bypassing custody in tests would not prove the port.
A failed capture end retains its source guard until a successful end. Context
teardown binds the context being released before synchronization.

The upstream untimed allocation tie rule can include zero-probability tail nodes.
With opt-in `TF_GLM_LONE_COSTS`, flat priced shared rounds still choose the
shorter equal-rate allocation. Tests distinguish this from positive-probability
allocation equivalence; it is not an output-parity measurement.

## Coverage deliberately not claimed

Retained families do not implement upstream Bolt, subset-head/routed-window,
upstream GLM ring/packed/DFlash loader, or Omni video APIs. Inapplicable standalone
targets are not advertised as passing tests. Family aggregates and original fork
probes remain; upstream common EXL3 probes have explicit separate names. Upstream
load-owner and Omni-video fixtures report skips for absent capabilities.

Failed retained-family loads without the upstream recovery owner cannot safely
release their CUDA parent using that owner. Their parent is retained rather than
freed speculatively. Recovery of that exceptional path is not qualified here.

Imported registry metadata describes its original source/images. Historical
receipts remain historical. Kernel compilation is recorded below; measured
limits, numerical contracts and model/capacity/cache/spill/vision qualification
are still required. No Linux CPU result establishes Windows or GPU readiness.

## Verification artifacts

CPU runs use pinned Zig 0.17.0 on Spark 2 in a CPU-limited, networkless, read-only
container with no GPU access. Source snapshots have unique directories; a run's
real container exit status is recorded only after it stops. Build logs are kept
in `C:/Development/prep/glm53-v110-merge/` on the workstation.

Initial diagnostic run `v110-20261019-090528-ae8268`:

- Command: `zig build test-cuda-host -j2 --summary all` (dedicated cache).
- Actual exit: 1; 92/110 build steps succeeded.
- 2374/2394 tests passed, 5 skipped, 4 failed, 11 crashed; compile errors also
  prevented some test executables from running.
- Source archive SHA256:
  `b1217b8f181efad46439209ce605cd5bf3728b693adf94437eddf35a7575181a`.
- Full log SHA256:
  `440035637b7ace6eb664aaac3b58e2f4b14d77d6a45891cc8501d44ed6ec27f9`.

Later repairs are not covered by that failed diagnostic snapshot. Subsequent
CPU-only results (all actual container exits; none OOM-killed):

| Run suffix (all `v110-20261019-`) | Requested gate | Result |
| --- | --- | --- |
| `091223-42c0ae` | `test-server-cpu test-exl3-cuda-host` | exit 0; 12/12 steps, 953 passed / 1 skipped |
| `091558-b65066` | `fatbins -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121` | exit 0; 182/182 steps; compilation only |
| `091910-bd80a0` | `compile-config-server -Dfatbins=/fatbins` | exit 0; 6/6 steps; native executable compiled with the preceding fresh images |
| `092030-ad9a4b` | `test-cuda-host` | exit 0; 80/80 steps; 828 passed / 14 skipped reported for uncached executions, plus cached test steps |
| `092738-c9f566` | `compile-cuda-tests -Dfatbins=/fatbins` | exit 0; 8/8 steps; GPU runner compiled with fresh images, not executed |

The aggregate used a previous run's owned cache. Its reported count excludes
cached executions; it is not a fresh full-suite test count. Eleven native skips
explicitly cover absent upstream failed-load-owner and video capabilities. The
smaller aggregate graph removes the accidental duplicate test loop and standalone
fixtures requiring upstream family implementations; it does not replace them
with empty passing filters. Deliberately planted failure controls print `FAIL`
and warning diagnostics even when the build exits successfully.

SHA256 pairs, source archive then full log (kept beside the named logs):

- `091223-42c0ae`:
  `7fcafeb35e29f95ded51c6832510a8645896c0a673d1c93f6a367a11d1b7bc46`;
  `25b36da8f05a82a47583e037b26a07d2dfc23fa3f417b81e421b2960b7cacee0`.
- `091558-b65066`:
  `fc4ce2a8119db24541ff41ffacd6748681ff0596bb512104d41a1bdab7602b16`;
  `6ddbf98a6a32ae523a9bde3c2c3f2443098b230b594bb98b158100f63a218612`.
- `091910-bd80a0`:
  `7d1558cd8991a9bc7b3a92b4562ed30cfa1687a20e5ce17c79e62e740a91d806`;
  `72220fd3763e592f4def1cf5234dd66668becdcaaa2f0cfe057faa90a03253d5`.
- `092030-ad9a4b`:
  `3565c66d017138f505f8ba22175f713e340f36dc5a4a2b7b10d953a725da2ff1`;
  `898bc30321663c6b0c51db9698cdc094d2b02da13265d958570c3e466224b962`.

GPU-runner snapshots `091422-044215` and `092600-186250` failed compilation
on upstream-only probe calls into retained family APIs. The former also exposed
the repaired CK down-call token-count argument. These are diagnostic failures,
not device test results. Reconciliation passed compilation in `092738-c9f566`;
its source/log SHA256 pair is
`e62acb2e35701ffede29a92b4bcc7f64a73b1423463320f9a1cfb943182e2dfe` /
`9928c2786c7d8d3d9a73aa4ba4deaee1e46c26373b20f38865fbc6aa4196b389`.

The retained Nemotron kernel audit found all 26 used exports across latent,
ops, route and Mamba images retain their fork signatures. Latent/ops add unused
exports; route wraps the original top-k body as its eight-choice specialization;
Mamba scan stages at most 32 rows, covering retained windows/keeps capped at 16.
This is static ABI/arithmetic-order review, not GPU numerical qualification.

## Hot-path port and production draft profile

The original hot-path commits remain on `task/glm53-dual-hotpath`:
`0dbb813c1`, `b4aae003e`, `a701ba22d`, `5048576c5`, `eaebccc0c`, `4d7c2300f`,
`0a3633d87`, `9556ffffc`, `d7ad99ece`. Their retained-family changes and tests
were ported, not requalified by applying them. Shared batching preserves
upstream `prefill_many` (all-or-none fallback) and adds partial-outcome
`prefill_batch` for per-request setup failures versus fatal shared execution.
First-token commit hooks and bounded fill aging remain; fresh short cohorts
must not starve an aged long or background prompt.

`TF_GLM_DRAFT_PROFILE=production` explicitly selects the new development
profile; leaving it unset retains the previous advertised lane widths and
adaptive policy. The profile defaults and validates the production settings:

| Concept | Profile |
| --- | --- |
| Copied continuation | enabled; at most 15 tokens; exact suffix match 8 |
| Reply-sourced occurrence gate | total match 16; fixed original prompt boundary |
| Neural DFlash request ceiling | 7, then the existing noise-aware `fnc7:0.3` selector |
| Per-sequence target capacity | 16 rows, independent of neural depth |
| Shared execution limit | 32 rows including each pending row |
| Aggregate backing allocation | 64 rows, not permission to execute 64 in this profile |

The fixed neural request is not the generic confidence-floor option. Copies
suppress neural proposal execution for that round; zero-depth backend calls
still settle kept target/DFlash state. The existing chain implements the
production 0.6 edge, 0.7 selection noise, beta-2 confidence noise and cumulative
0.3 threshold; algebraic agreement does not prove floating-point parity.

Copy selection follows the pinned production source, not generic SuffixLookup:
latest occurrence with a full requested continuation, otherwise earliest with
the longest available continuation; overlaps and one-token proposals are valid.
The authoritative proposer avoids generic match gates, rejection silence,
width ramps and Recall fallback. A disabled profile copy setting returns an
empty custom proposal rather than silently restoring generic copies.

The eight-lane production recipe has `TF_GLM_MULTI_LONE=0`, so its copies are
**unpadded even at occupancy one**. Optional standalone padding is a separate
proposer API, not used by the served factory. Original copied length is kept
for miss accounting before any later trimming. A request-local factory uses
upstream lane-host ownership; its shared type lives in `lanes.proposer.Factory`
and the existing `lane_rounds.Proposers` name remains an alias.

Production shared allocation trims both copy and neural tails: remove one
from the longest tail, with the latest lane winning ties, until rows fit 32.
The opt-in `Trim.VTable.copy_windows` extension permits those copy-prefix cuts
without changing existing head-only policies or forced windows. Eight seven-
draft windows become eight three-draft windows at 32 total rows.

Maximum target/readback/sampler/replay buffers now support 16 rows while neural
held storage stays at seven. Those maximum allocations also grow for legacy
GLM settings; unchanged advertised behavior is **not** unchanged memory use.
Both profiles need device memory and numerical requalification.

Production's multi-lane target captures aggregate graph widths 1 through 32.
Native target graph execution is **not implemented by this configuration
change**. Neither serial `WIDE_GRAPHS=16`, target capacity 16, nor backing
allocation 64 proves graph equivalence or performance parity.

Post-port diagnostic failures (preserved as failures, not relabeled):

- `093638-81aa9e`: exit 1; 1988 passed / 3 skipped, with the native test binary
  blocked by a misplaced `LaneHost.Proposers` type reference. The type was fixed.
- `094638-56ab22` and `094649-39eb04`: exit 1, blocked by family references to
  `core.LaneHost`, which is not exported by the checkpoint/core module. Moving
  the same factory interface into the already-shared lanes module fixes the
  module boundary; reruns are required.
- `094415-2b3ca1`: server-only CPU gate exit 0, 467 passed. This target does not
  execute the new lane-core tests and must not be described as proving them.

## Post-profile verification

- `094909-d987aa`: `test-cuda-host test-server-cpu`, exit 0, 81/81 steps;
  1502 passed / 13 skipped in uncached executions, with other steps cached.
- `094909-08ecd5`: `compile-config-server compile-cuda-tests`, real freshly
  compiled sm_121 images embedded, `-j1`, exit 0, 12/12 steps. The native server
  and GPU runner compile; neither executed a model or GPU probe. These two runs
  share source archive SHA256
  `551781576451bac3f145c482285b9b320194cbab8af51902d2c2a7d1a56d7e38`.
  Their respective log hashes are
  `80e85df790593b3817577c05d2f605f90960a8db7566edffed41b742d160d26b` and
  `f53e197d12ee608322d0323055e4843a35b01e19276c64dc6ee6b006d013fdca`.
- Pinned Zig formatting initially rejected three files (`095126-703803`);
  applying formatter-only changes fixed them. `095555-b59842` checks every
  changed/new Zig file with `zig fmt --check`, exit 0.
- **Fresh-cache formatted-source run `095353-c4a690`:**
  `test-cuda-host test-server-cpu`, exit 0, **81/81 steps, 2085 passed / 14
  skipped**. This is the complete reported count, not cached partial counts.
  Source archive SHA256:
  `27acdfad30c1f5028c7f6e1d39a41d5c66a036b8cb1137e455de4762e3418b66`;
  full log SHA256:
  `103ba73cecb035587706feff3147fb6b33837e01cd272df70ca61a72c30c4938`.
- **Formatted-source executable run `095457-66e3b5`:**
  `compile-config-server compile-cuda-tests -Dfatbins=/fatbins -j1`, exit 0,
  **12/12 steps**. Fresh `091558-b65066` images are embedded. Source archive
  SHA256 `964ebd6aed194af564dbb215bd922bee23ed38ba2df6ed1030dea19439cf1562`;
  full log SHA256
  `f53e197d12ee608322d0323055e4843a35b01e19276c64dc6ee6b006d013fdca`.
- All 139 CUDA kernel/catalog inputs compare byte-for-byte with the successful
  `091558-b65066` compilation snapshot. There were no subsequent CUDA source
  changes; this does not replace symbol, limits or numerical hardware checks.
- Fleet attribution documentation passes the repo Python's `fleet.verify fast`
  in its existing `.worktrees/task/glm53-dual-hotpath` task worktree. The default
  fleet checkout remains clean. No serving recipe or production pin changed.

The fresh CPU run includes authoritative 15-copy/16-row serial-equivalence
fixtures at every rejection boundary (greedy and keyed draws), fixed-depth and
copy-prefix policy tests, and request-factory completion/cancellation/grammar-
open/factory-failure cleanup. These use fake backends, not KDA/model execution.
A separate focused static review found no concrete profile/copy/16-row defect;
it traced forced-row admission before shared trimming and inspected general
row-count KDA loops. That review does not qualify hardware replay or memory use.
The separate broad shared-base safety review was stopped after exceeding its
bounded pass without delivering findings; it is incomplete and is not counted
as a passed review. The focused profile review is not a substitute for it.

## Next gates

1. Complete a bounded shared-base/native safety review before any serving
   promotion of this source integration; CPU gates and executable builds are
   complete. Landing source on `trunk` does not pass this outstanding gate.
2. Run independent GPU limits, numerical contracts, every-rejection replay and
   model/capacity/cache/spill/vision qualification on the exact new source.
3. Only with a valid new outage approval, run a controlled matched production
   parity trial; restore production and verify a public proxy completion.

Production was not stopped or reconfigured for this integration work.
