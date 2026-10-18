# Deliver Qwen35-family native production serving

The user set runnable production serving as the delivery target on 2026-10-18.
Use the existing TensorFold fork and fleet lifecycle to deliver a selectable
Qwen3.5/3.6 35B MoE deployment, starting with Ornith 1.5 35B ModelOpt NVFP4.
The serial CUDA port is the starting point. Its
[qualification record](qualification.md) still has a failed numerical gate.

This plan defines the next work. It does not qualify a runtime, select a
deployment, or authorize another production outage. The previous approved
window ended with Nemotron restored and a successful proxy completion.

## Set the first production contract

Target native Windows CUDA on the internal RTX 5080 and RTX 5070 Ti. Run one
selected checkpoint across the pair. Deliver text, thinking on and off, tool
calls, streaming, cancellation, bounded queues, and explicit context admission
through the existing proxy and supervised fleet launcher.

Include concurrent agent sessions, chunked prompt processing that lets active
sessions continue decoding, and correct prompt-cache restore. The first
production candidate enables a bounded GPU prefix cache. Host-memory and NVMe
tiers follow as separately qualified additions to the same store. Neither is
advertised by the first candidate. Qualify every tier a recipe enables.

The initial qualification profile is two active lanes with a 65536-token total
window each and a 131072-token shared KV budget. Reserve 16384 tokens per
request for output in fleet admission. Bound the waiting queue at eight.
Start with BF16 KV and FP32 GDN state. Budget a 512 MiB GPU prefix cache on
each device. These are engineering targets; admission must refuse them if the
complete per-device budget and physical readings cannot preserve the floor.

The first API contract includes chat and raw completions, streaming, automatic,
required, and forced tool choice, thinking on and off, cancellation, queue
refusal, and reuse after errors. Thinking budgets, structured response output,
logprobs, and loop guards remain disabled until their own API cases pass.
Advertise support through `api.Info` only after the relevant tests pass.

Set these initial workload gates before candidate timings: a 32768-token cold
prompt with 256 output tokens has TTFT at most 30 seconds; a 16384-token reused
prefix with a 1024-token suffix has TTFT at most 5 seconds; two active sessions
each decode at least 20 tokens per second; another session's prefill causes
no decode pause over 1 second. These are chosen acceptance targets, not prior
measurements or user-specified service guarantees. Record paired samples and
their spread before deciding whether a candidate meets them.

Screen one, two, and four active lanes at 32K and 64K total tokens per request,
then 128K, using the CPU checkpoint-layout budget first. Run only feasible
cells on a device. These are qualification candidates, not capacity promises.
Publish the chosen maximum request window, shared token
pool, lane count, output reserve, queue limit, and measured latency together.
Keep at least 1536 MiB physically free on each internal GPU under the intended
load. Preserve GDN recurrent-state precision while evaluating KV precision,
and reduce value precision before key precision.

Use an explicit Ornith model route for initial production qualification.
Changing `local-small` needs a separate routing decision that accounts for its
current context admission and remote escalation. The 4K trial cannot satisfy
that route's existing contract. Vision and speculative decoding remain later
extensions with their own qualification.

## Deliver verifiable engineering units

1. **Align source and freeze the contract.** Merge the selected current trunk
   into this task branch, preserving the original evidence and runtime pins.
   Run the targeted host baseline and record the kernel and interface diff.
   Run `tf-qwen35-probe --plan MODEL` to derive weights, staging, scratch,
   complete session state, and the floor from the actual checkpoint. This is
   a layout budget, not physical admission. Add batched workspaces, cache
   copies, transfer slots, allocator overhead, and external use before deciding
   which placement and matrix cells deserve a GPU experiment.

2. **Establish the arithmetic baseline.** Preserve the failed comparisons and
   their fixed limits. Extend same-input operator comparisons with identical
   attention and GDN state until the first unexplained difference is isolated.
   Establish a separately pinned serving reference for the exact checkpoint
   and quantization semantics. The custom CPU adapter is useful evidence but
   is not an independently qualified ModelOpt runtime. Check logits and model
   quality across a wider corpus. Explain any acceptance-policy change with
   independent evidence before applying it to the candidate. Name and pin the
   reference runtime, host, dependencies, exact checkpoint, compatibility probe,
   and resource window. Metadata compatibility is a screening result, not a
   successful model execution. If that path cannot execute the checkpoint,
   validate a replacement before calling it a reference. Variants sharing the
   same custom CPU adapter cannot independently calibrate each other.
   Numerical acceptance gates performance qualification and promotion. CPU
   integration may proceed against the preserved baseline while diagnosis
   continues. A byte-identical migration is not numerical qualification.

3. **Connect Qwen to the shared lane host.** Reconcile the existing task branch
   with current `trunk`, retaining its checkpoint loader, packed weights, and
   family operators. Implement `lanes.backend.Backend` and the family `Loaded`
   contract used by `families/nemotron/cuda_native.zig`. Route Qwen through
   `native/cuda.zig`'s shared `Host`, `lanes.Engine`, and `api.LaneHost`.
   Migrate request lifecycle out of the private serial `Host` in
   `native/qwen35_cuda.zig`. First prove a one-lane integration with the same
   arithmetic. This is an intermediate engineering checkpoint.

4. **Extend shared host and launcher capabilities.** The existing host forces
   layer split to one stream and zero cache, and refuses spill and multiple
   segments. The fleet launcher separately enforces the trial limits. Extend
   these boundaries through family-declared placement, per-device weight and
   state budgets, and persistence capabilities. Retain existing refusals for
   unqualified combinations. Prove two independent streams and a nonzero cache
   budget through the shared host and real launcher. Do not use a TP flag to
   bypass a missing Qwen implementation.

5. **Add stream-owned state and useful prompt execution.** Give each stream
   its own full-attention KV, GDN matrices, convolution history, and position.
   Add admission that budgets weights, workspaces, all admitted sessions,
   cache storage, and the per-device reserve. Use shared scheduling and
   sampling. Add real chunked or batched Qwen prompt operations, then shared
   decode rounds without requiring TP or MTP. Qwen's lane facts must admit
   these shared rounds for the layer-split placement. Yielding between serial
   token calls proves fairness but does not establish adequate long-prompt
   performance. Name and prove `streams_exact`, `hidden_rows`, and the admitted
   exact-width contract. Implement `verify`, `keep`, and `draft` for enabled
   request paths. Disable copied or forced multi-row windows until complete
   GDN and history rollback is proven. Shared rounds still call `keep` without
   MTP. Use the existing Qwen
   HIP lane/state design as a reference and retain existing CUDA kernels
   wherever their equations and layouts fit.

   Before broad integration, bound the payoff and falsify the smallest batched
   prompt and decode operation at representative shapes. Use fixed correctness
   checks and a ten-minute experiment budget plus recovery for a proof window.
   Choose placement before its full persistence implementation. Reserve the
   full model matrix for candidates that survive the cheap screen.

6. **Implement complete cache restore.** Connect Qwen to the shared snapshot
   and compact-image interfaces. Capture attention KV, GDN matrices,
   convolution history, position, and any logits or execution metadata needed
   to resume. Key stored state to checkpoint, arithmetic, layout, and topology.
   Prove uninterrupted and restored continuation equivalence at a fixed
   execution policy. Separately validate different prefill chunk and batch
   policies against the arithmetic and quality gates. Prove cross-session
   isolation, eviction, cancellation, corrupt-image refusal, clean restart,
   and abrupt-stop recovery for every enabled cache tier. Multi-device restore
   must commit all partitions or fall back to a fresh prefill. Include prefill
   and decode policy in compatibility checks; an incompatible cached state
   misses rather than silently crossing arithmetic policies. The CUDA disk
   attachment is currently Nemotron-specific. A later host/NVMe addition needs
   a Qwen adapter or a minimal family capability extension to the existing
   store; Windows `rows_store` alone is not that spill implementation.

7. **Qualify the production profile.** Run the exact checkpoint and immutable
   runtime through the capability and agentic paths. Exercise every admitted
   lane at the chosen request limit and shared-pool limit, mixed long prefill
   and decode, saturation, disconnect, cancellation, refusal, and subsequent
   reuse. Measure physical GPU headroom under load and Windows contention.
   Record cold and cached prompt latency, decode rate, tail pauses, host-memory
   pressure, spill I/O, and a sustained soak. Set workload latency and
   throughput acceptance before comparing candidates. Include supervisor
   restart and a reboot drill before claiming unattended persistence.

8. **Package and promote the selected checkpoint.** Pin the final source,
   executable, kernels, checkpoint, tokenizer, template, and retained notices.
   Add the deployment, component, supervised-task settings, registry entry,
   admission contract, and recovery recipe together in a task worktree.
   Run `fleet.current_state` before fleet configuration changes. Run targeted
   checks and the final full promotion gate, then complete direct and proxy
   probes in an approved window. Preserve the working Nemotron recovery inputs
   and prove the recovery procedure. Use `ornith-1.5-35b` as the proposed new
   concrete route; qualify direct and public proxy paths before exposing it.
   Do not repoint `local-small`. Derive the supervisor's per-UUID memory
   expectations from this candidate's measured peaks, not Nemotron's.
   Prepare a restore deadline and explicit unsafe-condition stop criteria.
   A numerical failure blocks promotion; planned bounded diagnostics may
   continue while safe. Keep the existing trial recipe inactive and pinned as
   diagnostic evidence. Record the outcome in `fleet/trials/`.

## Qualify checkpoints and formats separately

Deliver the pinned Ornith ModelOpt checkpoint first. Run NVIDIA Qwen3.6 NVFP4
through the same engine next, using its own immutable checkpoint and recipe.
Select and qualify a concrete Qwen3.5 35B checkpoint before claiming base-model
support. Configuration compatibility and header validation do not establish
whole-model execution or quality.

Unsloth and Unsloth Fast use compressed-tensors with different scale layouts
and activation quantization. Add a separate reader and projection arithmetic
path after the first ModelOpt deployment, with format-specific tests and
whole-checkpoint qualification. Keep the shared family forward and serving
contracts. Do not interpret their metadata as ModelOpt scalars.

Keep the current contiguous layer split as the first placement candidate.
If its measured latency or throughput misses the chosen workload contract,
evaluate TP or expert parallel execution through existing collectives and
kernel interfaces. Either placement must pass the same state, memory, and
quality gates. CUDA graphs and MTP follow a qualified baseline.

## Prepare without stopping production

Source integration, CPU reference work, host tests, builds, checkpoint
inspection, recipe preparation, and workload definitions can proceed while
Nemotron serves. Reuse this task worktree and the existing fleet trial
worktree. Keep immutable runtime bundles separate from source.

The fork source contract requires fresh approval for a new outage. Prepare
the exact candidate, test commands, time budget, and recovery commands before
requesting that window. GPU execution and replacement of the selected engine
wait for that approval. Restore production and verify a proxy completion
when a qualification window ends unless a later promotion decision selects
the qualified candidate.

## Source references

- `zig/src/native/cuda.zig`, `zig/src/core/lanes/backend.zig`, and
  `zig/src/core/lanes/engine.zig` define the shared hosting boundary.
- `zig/src/families/nemotron/cuda_native.zig` and `cuda_lanes.zig` demonstrate
  the existing CUDA family integration.
- `zig/src/families/qwen3_5/backend/hip/engine/` and `forward/` demonstrate
  Qwen state ownership and lane planning on HIP.
- `zig/src/core/prompt_cache.zig` and `rows_store.zig` define shared cache
  interfaces. CUDA family adapters supply complete state images.
- [Current Qwen qualification](qualification.md) records observed results.
- [vLLM's Qwen3.6 recipe](https://github.com/vllm-project/recipes/blob/main/models/Qwen/Qwen3.6-35B-A3B.yaml)
  documents a potential independent NVIDIA ModelOpt reference. Pin the tested
  revision and verify exact Ornith checkpoint compatibility before using it
  as an oracle. Its published hardware results do not qualify this port.
