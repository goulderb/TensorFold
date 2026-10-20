# Source publication boundary

This is a privacy-cleaned publication copy of the development fork based on
canonical source `84dc2e285a559edfb07d1e90110335dc350f59a8`. Public upstream and donor history is retained
unchanged. Local commit author/committer email metadata uses the account-ID-based
GitHub no-reply identity; contributor names, dates, parent ordering and immutable
source references are preserved. Rewritten local commit IDs necessarily differ.

## Omitted private evidence

Private workstation run receipts, evidence ZIPs, raw captures and debug symbols
under `docs/plans/qwen35-ornith/evidence/` are deliberately excluded from local
history. The standalone `zig/src/families/glm/fixtures/weight-loader-verification.json`
is also a private run receipt, not a model input fixture. Canonical originals
remain private and unchanged; no edited archive is presented as an original
qualification artifact.

The private evidence archiver, its pinned-row replay wrapper and the two analysis
tests that require those omitted journals are excluded as well. General analysis
code, model/source test fixtures and the actual runtime kernel assets remain.
Historical references to private receipts are provenance, not downloadable public
proof. This preparation did not rerun model or hardware qualification.

## Build and source integrity

Engine and kernel code, runtime kernel assets and retained model fixtures are
unchanged apart from anonymizing compiler-origin comments in two generated
vision headers. No executable header declarations change. Licenses, original
donor notices, author credits and immutable source pins are retained. Exact
metadata edits replace local network addresses in spill documentation and
personal compiler-home paths in qualification documents and developer helpers.

The PowerShell build helper requires an explicit `-Zig` path. The optional Python
weight-check helper resolves `zig` from PATH; use the version pinned by this tree.
These developer-helper changes do not change inference logic or deployment pins.
Private evidence is not needed by the engine build or source release packaging.

The upstream desktop application's unrelated dependency/build status has not been
qualified by this publication preparation. No model weights, newly built release
binaries or container images are added by this export.
