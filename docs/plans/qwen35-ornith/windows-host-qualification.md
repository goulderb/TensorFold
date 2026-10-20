# Windows host verification

The Qwen task exposed seven failing steps in the existing Windows CPU host
suite. The portability fixes preserve loopback HTTP coverage, use the
existing Windows filesystem helpers, and exercise explicit refusal for
features whose implementations already do not support Windows.

Production fixes cover Windows EAQ shard-path containment, readable benchmark
lock handles, Recall refusal before allocation, and learned-directory cleanup.
Cleanup pins directory handles against write and delete sharing, rejects
identity reparse points, and enumerates and deletes through handle-derived
canonical paths. Only the configured root may be an alias. Tests check owned
names, indexed and foreign payload preservation, and replacement refusal.

Verified on 2026-10-18 with Zig 0.17.0 from
`C:/Users/developer/tools/zig-0.17.0/zig-x86_64-windows-0.17.0/zig.exe`:

```text
zig build test-cuda-host -j2 --summary all
exit 0; 43/43 steps; 1147/1231 tests passed; 84 skipped

zig build test-qwen35-cuda-host -j2 --summary all
exit 0; 8/8 steps; 36 reported tests passed; one admission run cached
```

The server target separately passed 377 tests with 29 skips, including the
100-round benchmark lock race. Direct lanes tests passed 124 with 10 skips.
`fleet.verify fast` passed in the local-ai-server trial worktree. A separate
read-only review found no P0, P1 or P2 defect in the scoped Windows diff.

The aggregate's 84 skips comprise 29 server, 40 engine API, 10 lanes, three
core, one Nemotron and one native-host case. They cover unsupported Windows
Recall, Originals, row and qualification persistence, POSIX resource limits,
signals, shell and terminal fixtures, direct I/O, and unavailable symlink
privileges. Missing libjpeg-turbo also leaves JPEG pixel cases unqualified.
Eleven inherited Windows skips were removed. No Linux suite was rerun.

The successful aggregate still emits Zig HTTP-library
`INVALID_ADDRESS_COMPONENT` diagnostics and warning-associated `failed command`
lines before its successful summary. These logs are retained; a successful
host target does not qualify every Windows networking or managed benchmark
child-lifecycle path. No GPU, model or serving qualification follows from it.

Local evidence is retained in `.artifacts/windows-host-fix-before.log`,
`.artifacts/windows-host-fix-final-canonical.log`,
`.artifacts/windows-host-fix-qwen-final.log`, and
`.artifacts/windows-host-fix-report.md`.
