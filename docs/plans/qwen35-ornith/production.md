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
sessions continue decoding, and correct prompt-cache restore. Use the existing
GPU, host-memory, and disk cache infrastructure where Qwen supplies a complete
state adapter. Qualify every tier that the production recipe enables.

Measure one, two, and four active lanes at 32K and 64K total tokens per request,
then test 128K where memory permits. These are qualification candidates, not
capacity promises. Publish the chosen maximum request window, shared token
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

1. **Establish the arithmetic baseline.** Preserve the failed comparisons and
   their fixed limits. Extend same-input operator comparisons with identical
   attention and GDN state until the first unexplained difference is isolated.
   Establish a separately pinned serving reference for the exact checkpoint
   and quantization semantics. The custom CPU adapter is useful evidence but
   is not an independently qualified ModelOpt runtime. Check logits and model
   quality across a wider corpus. Explain any acceptance-policy change with
   independent evidence before applying it to the candidate.

2. **Connect Qwen to the shared lane host.** Reconcile the existing task branch
   with current `trunk`, retaining its checkpoint loader, packed weights, and
   family operators. Implement `lanes.backend.Backend` and the family `Loaded`
   contract used by `families/nemotron/cuda_native.zig`. Route Qwen through
   `native/cuda.zig`'s shared `Host`, `lanes.Engine`, and `api.LaneHost`.
   Migrate request lifecycle out of the private serial `Host` in
   `native/qwen35_cuda.zig`. First prove a one-lane integration with the same
   arithmetic. This is an intermediate engineering checkpoint.

3. **Add stream-owned state and useful prompt execution.** Give each stream
   its own full-attention KV, GDN matrices, convolution history, and position.
   Add admission that budgets weights, workspaces, all admitted sessions,
   cache storage, and the per-device reserve. Use shared scheduling and
   sampling. Add real chunked or batched Qwen prompt operations, then shared
   decode rounds without requiring TP or MTP. Qwen's lane facts must admit
   these shared rounds for the layer-split placement. Yielding between serial
   token calls proves fairness but does not establish adequate long-prompt
   performance. Use the existing Qwen
   HIP lane/state design as a reference and retain existing CUDA kernels
   wherever their equations and layouts fit.

4. **Implement complete cache restore.** Connect Qwen to the shared snapshot
   and compact-image interfaces. Capture attention KV, GDN matrices,
   convolution history, position, and any logits or execution metadata needed
   to resume. Key stored state to checkpoint, arithmetic, layout, and topology.
   Prove uninterrupted and restored continuation equivalence at a fixed
   execution policy. Separately validate different prefill chunk and batch
   policies against the arithmetic and quality gates. Prove cross-session
   isolation, eviction, cancellation, corrupt-image refusal, clean restart,
   and abrupt-stop recovery for every enabled cache tier. Multi-device restore
   must commit all partitions or fall back to a fresh prefill.

5. **Qualify the production profile.** Run the exact checkpoint and immutable
   runtime through the capability and agentic paths. Exercise every admitted
   lane at the chosen request limit and shared-pool limit, mixed long prefill
   and decode, saturation, disconnect, cancellation, refusal, and subsequent
   reuse. Measure physical GPU headroom under load and Windows contention.
   Record cold and cached prompt latency, decode rate, tail pauses, host-memory
   pressure, spill I/O, and a sustained soak. Set workload latency and
   throughput acceptance before comparing candidates. Include supervisor
   restart and a reboot drill before claiming unattended persistence.

6. **Package and promote the selected checkpoint.** Pin the final source,
   executable, kernels, checkpoint, tokenizer, template, and retained notices.
   Add the deployment, component, supervised-task settings, registry entry,
   admission contract, and recovery recipe together in a task worktree.
   Run `fleet.current_state` before fleet configuration changes. Run targeted
   checks and the final full promotion gate, then complete direct and proxy
   probes in an approved window. Preserve the working Nemotron recovery inputs
   and prove the recovery procedure. Record the outcome in `fleet/trials/`.

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
