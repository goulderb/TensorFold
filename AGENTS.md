# Local TensorFold fork

- Preserve upstream and donor copyright headers, license texts, `NOTICE`, and
  credits. Keep retained donor documents unmodified in `LICENSES/donors/`.
- When importing or adapting work, record its repository, immutable revision,
  original authors, type of reuse, and local changes in
  [the fork addendum](LICENSES/donors/README.md). Preserve transitive credits.
  The importer is not necessarily the original author.
- Follow MiaAI-Lab's redistribution rule: keep the donor's `NOTICE` and state
  what changed. Mark modified third-party files as their licenses require.
- For historical imports, add credit now without rewriting authors, commit
  dates, or shared history. Never guess missing license terms or author names.
- Keep the root `NOTICE`, `THIRD_PARTY_NOTICES.md`, and source headers consistent
  with code and captured kernels actually included in the fork.
- Before sharing an artifact, inspect its retained notices and licenses.
  `tools/release/test_package.py` checks native archive notice retention.
  Checkpoint weights and runtime dependencies keep separate license terms.
- Work on a task branch from `trunk` in `C:/Development/tensorfold.git`, with a
  worktree under `C:/Development/tf/`. Do not change deployment pins or stop
  production for attribution work. Publishing requires a separate decision.
