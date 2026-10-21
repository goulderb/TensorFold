# Forward-only source privacy

The canonical development store retains private historical commits and qualification
receipts. It is not safe to push raw. Forward-only exclusions do not remove tracked
files or erase their history. Keep historical evidence intact and keep production
source/runtime pins unchanged. Publish only an independently verified clean copy.

Local `trunk` can track public history while the same store retains private archive
and task refs. Preserve those refs and their pinned objects. Start new task branches
from the public-history `trunk`. Port reviewed changes from old tasks onto those new
branches; never merge private ancestry into public `trunk`. Keep the canonical push
guard active and use a clean publication clone for outgoing pull requests.

## Local defaults and hooks

From a task worktree, install the reviewed checker with your repository Python:

```sh
python3 -I tools/install_privacy_hooks.py --canonical
```

`--canonical` is for the private-history store only: its pre-push hook refuses every
push, including a direct URL that bypasses remote push settings. Do not use that
option in an independently verified public-history clone. Other contributors can
pass `--email` with their own approved GitHub no-reply identity. The installer's
identity choice must agree with any protected identity in the policy.

The installer changes only repository-local defaults. It sets the interpreter to
the Python running the installer, installs shared hooks and a checker/policy snapshot
under the common Git directory, and backs up previous setting values there. Old
linked worktrees are guarded even if their branch predates the new checker files.
An existing hook path, unrelated hook or locally modified managed file stops the
installer rather than being overwritten. Reinstall after reviewing checker/policy
updates; changing the versioned script alone does not update the installed snapshot.
Do not commit the common Git directory, installation state or private policy.

The pre-commit hook checks the actual pending identity and staged Git blobs, not
working-tree files. Commit author overrides and partial staging therefore remain
visible to the guard. Protected owner identities must use their approved public
address; unrelated donor authorship is not rewritten. The local committer needs a
public no-reply address. Empty commits still receive the identity check.

## Private artifacts and portable inputs

Keep new qualification evidence outside tracked source: use private artifact
storage or ignored `.private-evidence/` and `.artifacts/` directories. Commit only
reproduction commands, non-confidential configuration, sanitized summaries and
content hashes. Keep canonical originals unchanged; do not edit an evidence ZIP
and present it as the original receipt. The exclusions do not untrack legacy files.

PDBs, profiler captures, the legacy private evidence tree and the weight-loader run
receipt are rejected. Unapproved archives are rejected; genuine compressed fixtures
need the same exact approval as other binary inputs. Required runtime kernels and real binary
fixtures have explicit path/size/SHA256 approvals pinned to the reviewed public
baseline. Changed or new binary assets need a separately reviewed approval and
retained notices; an approval never waives content scanning or private-path bans.
Do not add broad binary, JSON or ZIP ignores that hide necessary runtime inputs.

Use CLI/environment inputs for tool selection and machine configuration. The
Windows Qwen helper requires `-Zig`; the Python weight-check helper resolves `zig`
from PATH. Compiler-origin comments in the two generated vision headers use a
neutral home prefix; their executable declarations are unchanged. Sanitize such
metadata when regenerating sources, not just human-written docs. Use reserved
example identities, generic user paths and documentation network addresses.

## Checks and private literals

```sh
python3 -I tools/privacy_check.py --staged --identity
python3 -I tools/privacy_check.py --range BASE..HEAD
python3 tools/zig/lean_check.py
python3 -I -m unittest discover -s tools -p 'test_privacy*.py'
```

The default and `--staged` modes inspect the index. `--identity` additionally checks
the pending local author/committer. A two-dot `--range` checks every outgoing
commit's identity, message and changed blobs, including intermediate commits whose
private content was later removed. Merge-parent diffs are included. The local lean
command includes the staged privacy gate; `--lean-only` exists solely for comparing
lean baselines. CI runs the outgoing privacy gate independently of that comparison.

The checker inspects complete allowed-size blobs, including encoded Windows path
spellings and UTF-16 text. Private home paths, private/CGNAT IPv4 addresses, common
credential formats and sensitive filenames are rejected. Findings identify the
rule and location but never print matched values. Oversized/unapproved binary
objects and malformed Git/policy input fail closed. Credential-fixture exceptions
require an exact path, rule, line SHA256 and review reason; they are not wildcards.
These checks are not a universal secret detector or a substitute for artifact review.

For additional known private values, keep a JSON file outside the worktree with
`{"forbidden_literals": ["personal@example.invalid"]}` and configure its absolute
path with repository-local `privacy.privatePolicy`. A configured missing/invalid
file fails closed. The canonical installation uses a private file under its common
Git directory; these private values are not shipped in source or supplied to CI.
The generic and protected-identity checks still run without that local supplement.

## Public branch and workflows

Public `trunk` is append-only: deletion and force pushes are blocked and changes
use pull requests. No approving-review count or unavailable CI status check is
required. GitHub secret scanning/push protection supplement, but do not replace,
the local privacy guard. Keep donor/upstream push URLs blocked.

The no-GPU workflow includes regression tests and a full PR commit-range privacy
check, with read-only permissions, full history and no persisted checkout credential.
GitHub Actions remain disabled on the public fork until all inherited workflows
and their permissions/publish steps have been reviewed. This change does not
activate Actions, deploy an engine, publish binaries or qualify hardware/models.
