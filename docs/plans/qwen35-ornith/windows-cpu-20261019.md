# Windows CPU portability result

The delegated Windows worker closed all four broad-test failures found during
this hour. Commit `4f0377c9` ports EXLR8 environment lookups and learned-state
fixtures to existing portable helpers. Its combined focused run reports 158
passes and 30 skips. Three POSIX symlink fixtures explicitly skip Windows;
Windows reparse-point behavior remains outside that coverage. Linux semantic
compilation passes; no Linux test execution is claimed.

Commit `45e303381` closes the remaining XPU and fake-GLM roots. XPU keeps its
public CRT-descriptor interface. Windows uses UTF-16 binary opens and native
positional reads through the descriptor's handle; Unix `pread` is preserved.
The regression covers Unicode paths, binary bytes, repeated offsets, short
spans, missing files and EOF offsets above 32 bits. The focused XPU target
reports eighteen passes across three roots.

Fake fabric uses raw page mapping on Windows, preserving zeroed and aligned
mailboxes, with Unix `mmap` unchanged. A typed allocation initially poisoned
fresh pages in Debug and failed peer-control behavior; the zero/alignment
regression and raw mapping fix address that observed cause. Three fake-fabric
tests pass. GLM control uses portable test-clock sleeps on Windows while
preserving its live Unix sleep path. Ten expiry tests and seven direct control
tests pass, including delayed ordering. XPU and control roots compile
semantically for Linux. Two architect candidates and a separate judge selected
the descriptor-preserving design; their receipt is archived in `design.json`.

The broad `zig build test -j2 --summary all` then passes 70/70 steps with 109
reported passes and one skip, with other roots cached. It exposes an existing
allocation leak in the Windows vision-refusal fixture despite exiting zero.
Separate commit `02818b408` frees both subprocess capture buffers for both
results. The red replay reproduces the 91-byte and 97-byte leaks. The green
replay keeps all three assertions passing and has no leak diagnostics. The
final cached broad run passes 70/70 steps with 37 reported passes and other
roots cached, with no allocator-leak diagnostics. Formatting and diff checks
pass. The worker and all three design children finished; no owned job remains.

The exact-byte Windows logs and design receipt are in
[the approved-hour evidence](evidence/approved-hour-20261019/README.md).
These fixes are separate from the immutable device-tested Qwen binary pinned
to `aa267a2f`. They do not qualify Intel GPU execution, Windows production GLM,
or a new serving binary. The four failed Qwen numerical rows and all production
capacity prerequisites remain open. Full fleet promotion verification has not
run because the candidate has not reached numerical and operational acceptance.
