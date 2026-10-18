# Qwen35 production plan review

Review target: `c908bcf2ce1eac3469a5adaed8620783ed9c2110`,
[production.md](production.md). Trunk observed during review:
`d29335ccd62caed69159f5c25bfcc77d96ca7a92`.

The user requested Fable 5.1, Kimi K3, and Astra. All three independently
reviewed the same standalone brief and returned within the ten-minute
timebox. Reviewers read files and returned findings without edits, builds,
tests, model runs, or service actions. Fable also fetched the linked upstream
recipe. The coordinator checked the findings against source and recorded this
judgment. The production plan and implementation remain unchanged.

## Intent

Deliver native Windows TensorFold production serving for Qwen3.5/3.6 35B MoE,
starting with Ornith 1.5 ModelOpt NVFP4. Reuse the shared engine, scheduler,
sampler, kernels, and fleet lifecycle. Qualify the exact checkpoint, concurrent
sessions, prompt performance, hybrid-state cache restore, memory admission,
and recovery before promotion.

## Reviewers and evidence

| Reviewer | Actual model and provider | Findings |
|---|---|---:|
| Fable 5.1 | `claude-fable-5-1`, `claudeAgent`, xhigh | 9 |
| Kimi K3 | `kimi-k3`, `claudeAgent`, max | 8 |
| Astra | `gpt-6-astra`, native Codex subagent, xhigh | 2 |

The dispatch receipt, complete T3 responses, Astra summary, and analytical
capacity screen are retained under
`.artifacts/production-plan-review-20261018/`. The screen is not measured
capacity or permission to admit a configuration.

## Lead verdict

Keep the shared-engine approach. Amend its execution plan before implementing
multi-lane serving and persistence. The strongest finding is a verified
integration constraint: the shared CUDA host and fleet launcher both enforce
the original one-lane layer-split trial. Family backend conformance alone
cannot lift those restrictions.

The remaining numerical discrepancy is unresolved. Independent-reference
feasibility and acceptance policy need concrete deliverables. CPU integration
can proceed against the preserved baseline, but preservation of its logits
does not establish correctness. The correctness gate must block production
qualification and promotion.

## Act on

### Extend the actual shared-host and launcher boundaries

Fable F1 and Astra A1 independently found this constraint. Kimi K3 also
identified the associated batched-forward obligation.

`zig/src/native/cuda.zig:434` refuses multi-lane layer split, prompt caching,
spill, and multiple segments. Line 526 forces one stream and line 547 makes
the cache budget zero. The trial worktree's `fleet/launch.py:216` rejects a
layer-split pool, cache, spill, or non-BF16 KV; line 241 rejects multiple lanes.
These guards are appropriate for the current trial.

Add an explicit unit for family-declared placement capabilities, exact weight
shares, admission on both physical devices, and supported cache adapters.
Update launcher validation only for combinations the candidate implements.
Preserve existing refusals for other combinations. Prove two independent
streams and a nonzero cache budget through the actual shared host, then the
real launcher. Do not select the TP flag as a workaround for missing Qwen TP.

### Make the reference executable and calibrate acceptance independently

Fable F4/F5 and Kimi K2 identify related gaps. Astra agrees exact-Ornith
compatibility is unproven, while correctly observing that the plan already
labels vLLM as a candidate.

Name the reference runtime revision, exact checkpoint, host, dependencies,
hardware requirements, compatibility probe, and resource window. A metadata
check can reject an incompatible format cheaply. It cannot establish NVFP4
kernel correctness or successful model execution. If the candidate reference
cannot execute the checkpoint, choose and validate another path before calling
it an oracle.

Keep the original four failed rows and limits as diagnostic history. Establish
expected arithmetic with independent evidence, select a broader corpus and
quality criteria before evaluating the new candidate, and measure expected
implementation differences where qualified references exist. Two variants
sharing the same unqualified CPU adapter cannot validate each other or justify
new limits. Do not substitute a comparison between those variants for an
independent correctness check.

Make dependencies explicit. Preserve one-lane arithmetic during CPU integration.
Require an accepted correctness baseline before performance qualification or
promotion. Correctness investigations may still use a bounded GPU window.

### Budget capacity before scheduling its measurement matrix

Fable F3 and Kimi K4/K8 converge here. The current BF16 attention state costs
2048 bytes per token per full-attention layer. Ten such layers give 20 KiB per
token across the pair. At 128K, one session needs 2.5 GiB of KV plus 61.40625
MiB of FP32 GDN matrices and BF16 convolution history. These figures follow
`cuda_engine.zig:72` and `:286`; they exclude scratch, cache copies, and other
allocations.

Produce a per-device budget with weights, load-time staging, batched workspaces,
session state, cache storage, transfer slots, allocator costs, and the floor.
Choose dense versus existing shared `cuda.KvPool` backing and define token-pool
admission. Evaluate value precision before key precision. Filter the proposed
lane/window matrix with this budget before device trials.

Treat sampled free VRAM from the 4K trial as bounded evidence. It does not
prove that four 64K sessions fit with the new workspaces and cache. Define a
physical-memory measurement and controlled contention check for Windows.
`cuMemGetInfo` remains insufficient as the physical admission proof.

### Prove batching facts and state commit semantics

Fable F6 and Kimi K3 identify this obligation. `lanes/config.zig:78` enables
shared forwards only when width is at least two, `streams_exact` is true, and
`hidden_rows` is true. The latter declares a real multi-stream forward.

Name the row-batched forward and its GDN, attention, expert, sampling, and
transfer operations. Validate batched versus serial state and logits before
advertising exactness. Separate interleaved scheduling from a shared batched
forward. Either implement the current exact-width contract or extend the
shared capability boundary without falsely claiming exact two-token windows.

Implement required `verify`, `keep`, and `draft` entries with defined behavior
for the enabled request modes. Shared rounds call `keep` even without MTP.
Copied or forced multi-row windows need valid GDN/history rollback, or must be
disabled explicitly. A no-op is valid only where the admitted path proves it.

### Screen performance and placement before full persistence integration

Astra A2 is the clearest finding. Define cold TTFT, active-session pauses,
turn latency, and throughput acceptance before the placement-sensitive work.
Use the cheap hypothesis loop to bound payoff from source and saved evidence,
then test the smallest representative prompt or shared-decode operation with
correctness checks and a fixed budget. Screen layer split and any proposed TP
alternative before building their complete persistence adapters. Reserve full
model matrices for candidates that survive this screen.

### Specify cache tiers and API features in the first production contract

Fable F2/F7, Astra A1, and the actionable part of Kimi K5 apply here.

Select the first recipe's enabled cache tiers. The generic Store and compact
image interfaces are reusable, but the native CUDA disk attachment is
Nemotron-specific: `native/cuda.zig:550` and `:760` bind its target, layout, and
spill adapter. Add a Qwen adapter or a minimal family capability extension to
the existing store. Do not assume `rows_store.zig`'s Windows path supplies the
production spill implementation.

Define allowed prefill/decode policies and cache-key identity. Prove exact
restore continuation at a fixed complete policy. Require compatibility proof
or a cache miss for state produced under another policy. Check quality across
the supported policy combinations separately.

List API support for automatic, required, and forced tool choice, structured
output, thinking toggles and budgets, logprobs, and loop guards. Explicitly
refuse unsupported modes. The two BFCL fixtures do not qualify every tool mode.

### Prepare concrete cutover and recovery evidence

Fable F8 and part of Kimi K7 apply. Name the served ID and exposure, exact
direct/proxy probes, measured per-UUID supervisor memory expectations,
approval scope, window budget, completion criteria, and recovery commands.
Define unsafe conditions that end an experiment. A failed numerical gate
blocks promotion; bounded planned diagnostics can continue while safe.
Complete approved outage work before restoration unless an unrecoverable
failure prevents completion, as the fleet instructions require.

## Consider

- Kimi K6 and Fable F9 recommend a separate trunk-alignment checkpoint. Accept
  a CPU build/host-test checkpoint and explicit source diff before integration.
  Qualify changed kernels and the final runtime in approved device windows.
  Do not require an extra full-model outage solely because commits moved.
- Fable F2 leaves the initial cache tier set undecided. A narrower qualified
  tier set can be a useful first production milestone if its service contract
  states the limitation. It must still meet the selected workload's goals.

## Noted

- Kimi K7 asks for the old trial recipe's disposition. Keeping it pinned,
  inactive, and explicitly experimental is valid. A new qualified recipe can
  supersede it without deleting recovery or diagnostic evidence.
- The original plan already calls the capacity matrix proposed and requires
  exact-checkpoint qualification. Neither item is a claim of existing support.

## Dismissed or narrowed

- Kimi K1 calls the plan a critical shipment of an unqualified engine. The
  plan explicitly forbids that promotion. Accept explicit gate dependencies,
  but reject the claim that CPU integration must stop until numerics is green.
- Kimi K3 treats the current serial kernel path as a contradiction of future
  batching work. It is a real implementation obligation, not proof the route
  is impossible. Fable F6 overstates the contract as requiring two rows per
  stream in every shared round; one-row windows are also valid.
- Kimi K5's proposed cross-policy bit-equality requirement is too broad.
  Restore must preserve the stored execution policy exactly, while incompatible
  policies can miss the cache. Baseline disagreement does not justify weaker
  restore integrity within a fixed policy.
- Kimi K6 overstates trunk's shared-interface churn. The relevant net kernel
  diff includes `nvfp4_experts.cu` and `glm_glue_attn.cuh`; the named lane, host,
  and sampler interfaces do not have substantive changes. Old binary evidence
  remains evidence for that binary, not qualification of a rebuilt runtime.
- Fable F3's inferred fit table is not measured admission. Fable F4's two
  custom-reference variants are not independent calibration. Fable F5's
  suggestion to demote the oracle without a validated replacement would leave
  the central correctness question unresolved.

## Agreement and next units

Fable and Astra independently found the same shared-host restriction. Fable
also traced the launcher restriction. All reviewers identify concrete work
between the serial family engine and production multi-session serving. Fable
and Kimi agree that reference setup and memory arithmetic need more detail.
Astra and Kimi agree that workload acceptance belongs earlier. Their severity
ratings differ; the lead judgment above follows verified execution paths.

The next verifiable units are:

1. Reconcile trunk, define the workload and API/tier contract, and produce the
   per-device memory and placement screen. Keep this preparation CPU-only.
2. Validate an executable exact-checkpoint reference path and prepare bounded
   same-input discrepancy checks with explicit correctness exit gates.
3. Implement one-lane shared-host integration with preserved arithmetic, then
   extend host/launcher capabilities and prove a small batched seam before
   the full cache and production-capacity work.

These are recommended amendments. This review does not apply them to
`production.md`, change serving code, select a deployment, or authorize a GPU
outage.
