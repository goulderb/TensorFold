# N8 counter investigation

Hardware-counter collection remains unavailable. After three report-directory
harness failures, the fourth bounded attempt reached both internal GPUs and
NVIDIA returned `ERR_NVGPUCTRPERM` for each. No counter rows were obtained.
This is a profiling permission limit, not a measured hardware scaling wall.
The accepted unprofiled N8 full-model result remains separate.

The exact-symbol plan pins the measured `a5b417a82` executable and fixture.
It skips 32 correctness launches and 128 warmup nodes, then requests four N8
launches per GPU. Eight metrics cover duration, L2 sectors, active warps,
issue activity, SM throughput and the installed GB203 DRAM read/write names.
Application clocks and caches are uncontrolled; any eventual report must
retain replay/cache conditions and cannot replace an unprofiled comparison.

## Failures and repairs

1. NCU opened `profile-results/counters.ncu-repz` before the application could
   create `profile-results`. The output parent did not exist.
2. Creating that directory in the controller conflicted with the bootstrap's
   requirement that the application output be absent.
3. Moving creation into the bootstrap passed its contract but violated the
   application's exclusive `createDir`. The CPU stub had not modelled that
   third boundary. The application stopped before GPU work.
4. Separating the report from the application directory removed that conflict.
   NCU now writes `counters.ncu-repz` in the existing trial directory, beside
   `profile-results`, which only the application creates. The driver then
   denied counter access on both GPUs.

The final repair deletes the extra directory-creation API. A real Windows Job
CPU test models the profiler opening its report before an application that
exclusively creates its own directory. Its negative control reproduces the
former nested-path failure. Both controller and bootstrap reject existing
reports without overwriting them or stopping production. Eighty-six targeted
tests passed with zero failures/errors/skips, recorded in JUnit; Ruff and the
fleet fast gate passed. The fleet fix is `3bd4301cf09fe025913a7e76dc28e23e21739a13`.

Every controller/profiler job exited and production restoration was verified.
The final attempt restored Nemotron PID 21972; canonical request
`5497bb2a-c46c-4182-868f-cde346f335e1` returned `READY` at 19:43:14 UTC.
Clef was unchanged and the controller reported no cleanup or sampler errors.
The preceding attempt restored at 19:38:54 UTC with canonical request
`09796df8-963b-4e8b-a9dc-524e4a2127cd`.

NVIDIA documents elevated execution or administrator-enabled counter access
as the remedies for this error in its
[counter-permission guidance](https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters).
No driver permission or global setting was changed. Remaining unprofiled
operator experiments and the numerical/prefill work can proceed independently.

## Independent review

Fable 5.1 verified the N8 full-model timings, all twelve turn hashes, actual
expert graph counts and the quantization projection against original artifacts.
It correctly flagged the failed third counter attempt and the unverified
current-HEAD defaults. Its report predates that attempt's verified restoration.
The response preserves that timing distinction and retains both controller and
in-process free-memory readings. Head default qualification, serial prefill,
capacity and the independent HF full-logit gate remain open.
