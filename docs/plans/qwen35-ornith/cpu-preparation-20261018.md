# Qwen35 production CPU preparation

The task branch merged trunk `d29335ccd` as `1db3bfe36` without conflicts.
The original runtime and checkpoint pins remain unchanged. The merged Qwen
host baseline passed all 39 tests in eight build steps. This verifies the
host baseline after integration, not the changed CUDA kernels on a device.

## Reproduce the layout budget

Build the probe with the pinned Windows toolchain, then inspect the actual
checkpoint without initializing CUDA:

```powershell
./tools/zig/build_qwen35_windows.ps1 -Target qwen35-probe
./zig-out/bin/tf-qwen35-probe.exe --plan C:/Development/local-ai-server/models/ornith15-35b-nvfp4-94e431d9
```

The probe emits twelve cells: one, two, and four streams at 4K, 32K, 64K, and
128K. Each partition's weights and staging come from the checked tensor layout.
The same `memoryBudget` function supplies the engine's one-stream requirement.
It includes both KV planes and complete GDN state for every requested stream.

At the proposed two-lane 64K profile, each stream holds 703283200 state bytes
per device. Weights, current scratch, both streams, and the 1536 MiB floor
require 13817247232 bytes on the embedding partition and 13086201344 bytes on
the head partition. These are computed layout requirements. They exclude CUDA
modules, allocator overhead, other processes, display allocations, cache
copies, spill slots, and future batch workspaces. Physical admission remains
unverified.

The four-lane 128K BF16 layout exceeds a nominal 16 GiB on the embedding
partition even before those exclusions. Reject that candidate in this layout
before scheduling a full model trial.

## Verification evidence

Evidence is under `.artifacts/production-prep-20261018/`:

- `qwen-host-baseline.log` records the merged 39-test baseline.
- `memory-budget-red.log` records the missing new budget contract before its
  implementation.
- `memory-budget-final-host.log` records eight successful steps, 38 reported
  passing tests and a cached three-test admission run. The first host test
  binary reports 37 tests, two more than the baseline's 35.
- `budget-probe-build.log` records all 56 build steps passing, including the
  native probe and `sm_120` fatbins. This is compilation, not device execution.
- `ornith-memory-plan.json` is the CPU command's output with config and index
  hashes, the cut, floor, exclusions, and all twelve cells.

The unchanged arithmetic path still has the failed numerical comparisons in
[qualification.md](qualification.md). The accepted plan amendments are in
[production.md](production.md). Reference compatibility and shared-host
integration are separate checkpoints; neither a CPU budget nor compilation
qualifies inference or production capacity.
