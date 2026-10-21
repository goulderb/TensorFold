# Forward-only source privacy

The public fork is the development source of truth. Local `trunk` tracks its public
history. The former private Git store is a separate local archive for historical
qualification evidence, old worktrees and pinned production commits.

Start new task branches from `trunk`. Port reviewed source changes from old tasks
onto those new branches; never merge private ancestry into `trunk`. Keep archived
evidence intact and production source/runtime pins unchanged. New private output
belongs outside Git. Ignore rules do not remove files from an existing history.

## Local defaults and hooks

From a task worktree, install the reviewed checker with your repository Python:

```sh
python3 -I tools/install_privacy_hooks.py
```

Use `--private-history` only for the separate archived store. Its pre-push hook
refuses every push, including direct URLs. The old `--canonical` spelling remains
an alias for that private-history mode; it is not for the public development store.
Use `--email` to supply an approved GitHub no-reply identity. The choice must agree
with any protected identity in the policy.

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

Push verified personal-fork changes directly to `trunk`; pull requests are not
required. Deletion and force pushes remain blocked. GitHub secret scanning and
push protection supplement the local privacy guard. Keep donor and upstream
push URLs blocked.

The no-GPU workflow includes regression tests and a full PR commit-range privacy
check, with read-only permissions, full history and no persisted checkout credential.
GitHub Actions remain disabled on the public fork until all inherited workflows
and their permissions/publish steps have been reviewed. This change does not
activate Actions, deploy an engine, publish binaries or qualify hardware/models.
