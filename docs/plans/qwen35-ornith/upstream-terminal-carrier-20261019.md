# Matched graph terminal selection

Retain graph-integrated strict device greedy selection for a real shared
Engine comparison. The matched carrier screen passes on both GPUs, and all
ten measured wall pairs favor device selection. This supersedes rejection
of that topology based only on the earlier direct-launch screen. It is a
synthetic terminal result, not a measured complete-model speedup.

Both arms submit one immutable graph per draw. Each graph copies the same
real saved BF16 row into a common hot device row. The candidate adds the
existing tree-round argmax inside that graph. The host arm downloads the
full496640-byte row, synchronizes and scans it. The candidate downloads one
eight-byte Pick, synchronizes and validates status/token. Both consume all
128 distinct real rows. There are no graph-node updates or candidate-only
host kernel submissions during timing.

Before device execution, Fable identified the diagnostic archive memcpy in
the host timing window. The corrected helper archives full rows only during
untimed before/after qualification on both arms. Timed sweeps still check all
128 tokens. This changes copy work and cache warming; no copy saving was
predicted. Original source, checker and prospective pins are preserved, with
a new helper, checker, executable and protected input. No old data changed.

## Actual measurements

One warmup pair precedes five alternating measured pairs. Each sample covers
128 sequential graph launches, transfers and row synchronizations.

| GPU | Host wall median [range], ms | Device wall median [range], ms | Paired saving median [range], ms |
|---|---:|---:|---:|
| RTX5080 |38.4015 [35.3405,40.9802]|24.3614 [23.6733,29.7079]|11.6271 [9.1024,16.6188]|
| RTX5070Ti |35.2522 [35.1750,36.3169]|23.5035 [23.4537,23.6252]|11.7510 [11.5498,12.7942]|

The paired savings are about90.84 and91.80us per draw. The wider GPU0 spread
is retained; every pair still favors the candidate. These event spans include
host gaps and synchronization. They are not kernel times. Raw wall-clock
endpoints were not retained, while samples and exact summary arithmetic are
independently checked. No hardware limiter or driver-cost attribution is
established.

The previous direct trial used another submission topology and included a
timed archival copy. Multiple boundaries differ, so the change of verdict
does not isolate the cost of one host launch or one memcpy. This matched
trial establishes a useful candidate under its own stated boundary.

The current20/20 model places the head on GPU1. A subsequent integration
must run through that actual body graph and preserve sampler semantics,
session ownership, logits and state. A conservative design that retains
full-row downloads cannot assume the saving measured here, which includes
transfer elision. Lazy host-logit materialization is a separate API change.
Do not add this isolated delta directly to previous model improvements.

## Qualification and recovery

Source b69ba65b0add172692216fbaf9a6ba7669e2b1e3, profile SHA256
fe922db7626ba8d5a7f234fa2cc8e2cf07199c7ad28d28a8ef04f42a709c8406,
and the93 pinned historical images are frozen. This screen uses unchanged
tree-round image3ead36eb3d671670b26c3d808f4ef9236158a781054f7fae58dcbc160c01ff1c.
The off-by-default slice-format source is present in the executable's
shared Engine but is not exercised or qualified by this terminal screen.
The actual frozen build records every tracked source, including both new
helpers. Sixty-five host tests and30 CLI admission cases pass.

The independent r2 checker passes15 methods/163 subtests and accepts all802
records and220 raw files. Actual graph checks cover node types, dependencies,
argmax function, geometry and every argument byte/value/hash. Memcpy source,
destination and size remain request evidence because this binding lacks
memcpy-parameter readback. Every produced full row is checked before/after
timing, supplementing those requests. Real tokens, strict signed-zero/tie
and NaN/infinity masks, poison, guards and immutable inputs all pass.
Executable-node readback is unavailable; parameters are read before instantiate.
Root independently verifies exhaustive source/build/image/fixture, record/file,
physical-memory sample and protected-restoration coverage.

Controller3272 ended and the profile exited0. Its four physical samples
show minima15648/15701MiB free. These short samples are not sustained model
capacity qualification. Nemotron restored as PID19020 at23:28:39.091740UTC.
Canonical request ec1979a8-54b6-4282-92d4-879768a4e0d2 returned READY, with
5572/5815MiB physically free. Clef remained unchanged and cleanup was empty.

The [evidence archive](evidence/upstream-terminal-carrier-20261019/README.txt)
contains404 members,19887097 compressed bytes, SHA256
7d58a02743808787f233f65fd976f562bdc179bbb76fc96df876f8eea6be5e96.
Twelve larger readbacks remain externally pinned,762839552 bytes. The ZIP
retains original and corrected prospective contracts, Fable review/response,
source/builds, all smaller raw files and both protected preparation histories.
The archive verifier passes. Independent full-model HF quality remains FAIL;
serving, capacity and promotion remain separate gates.
