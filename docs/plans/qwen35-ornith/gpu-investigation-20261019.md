# Measure eager submission and Qwen phase costs during the approved investigation

The user explicitly authorizes taking Nemotron down for this investigation on
2026-10-19. Prepare the probe while it serves. Each device run uses the existing
detached outage controller with a fresh absolute deadline, physical-UUID floor
monitoring and mandatory restored canonical completion. Clef and proxies stay up.

The initial question is whether identical fixed-step graph replay removes a
material share of the 32.7 ms decode path. The paired benchmark uses the real
layer body, both GPUs, the same host seam, logits download and per-token floor
queries. It holds token and position fixed while recurrent state evolves.
This is a submission experiment, not changing-token generation throughput.

Architecture candidates were independently reviewed by gpt-6-sol and
gpt-6-astra. The selected shape uses compile-time phase taps in the existing
block and a small diagnostic partition-enqueue method. The ordinary block
instantiates empty taps. The alternative was a driver API trampoline table
with a global function registry; its ABI/global-state complexity is unnecessary
for the first question. A fresh gpt-6-sol judge agrees with the selected shape.

Use an uninstrumented eager/graph pair for timing. A separate captured graph
records preallocated phase events, so intervals inside that graph do not
include late host submission between phases. Event instrumentation is still
extra GPU work and its timings are diagnostic, not headline throughput.

Establish nonzero state with 64 prompt tokens. Save and restore all persistent
state, scratch and seam input before each arm. Compare the diagnostic eager
step with the original append, all 16 intermediate graph/eager state and logits
digests, final digests for each pair, and event-graph versus ordinary-graph
digests. Use one warmup pair followed by five alternating measured pairs.
Finiteness and floor failures stop the probe. The independent numerical gate
remains open even when this self-parity passes.

The existing controller now accepts either its original head/sampler sequence
or a single pinned `tf-qwen35-decode-profile.exe`. All original source/binary/input
pins, timeout bounds, device UUIDs, private port checks and recovery contracts
remain. A test rejected the new single-profile plan before the parser changed;
the targeted controller suite then passed with 41 collected cases.

The first probe build failed because the family module root did not export the
new diagnostic constants. A separate Windows/build worker owns those compile
fixes. No device run or outage starts until the committed source, built binary
identity and CPU-only input validation agree.

The build worker resolved the facade exports and added cached-fatbin support.
The preliminary probe build passed 57 steps and the cached rebuild passed five.
The CPU-only host target passed 47 tests across eight build steps. The fleet
fast gate and 41 controller tests pass. Source is checkpointed before the final
source-pinned rebuild and the first device execution.
