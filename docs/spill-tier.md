# Spill tier: kept prompt states on local disk (Flash Next on CUDA)

The prompt cache keeps the state of a prompt at a few marks (a conversation's history, shared system blocks) so a later
request that extends those tokens resumes there instead of prefilling them again. Its budget is device memory, so a
busy server evicts states that a returning conversation would have resumed. The spill tier writes those states to the
machine's NVMe as they leave memory and reads one back when a request extends it, also after a clean restart.

This is a port of the GLM spill tier written by Robert Wojciechowski ([wojo](https://github.com/wojo)) for
MiaAI-Lab's GLM-5.3-Flash TensorFold recipe (Mia's patch 0088, `tensorfold/cuda/spill.py`, Apache-2.0). Its settings,
eviction order and multi-rank rules are wojo's. The file and directory conventions borrow from upstream TensorFold's
`--learn` (`zig/src/core/prompt_imprint.zig`). The code here is new Zig. `zig/src/core/spill.zig` knows no model;
the Flash Next hooks live in `zig/src/families/flashnext/cuda_lanes.zig` and `cuda_engine.zig`.

## What a kept state holds, per rank

At two ranks each rank keeps its own half: its KV heads, its GDN value heads, its share of every layer. A kept state
is the engine's `Snapshot` and a keeper sequence that holds the prompt's first `at` cache rows
(`cuda_lanes.Kept`). Both ranks keep the same states, in the same order, through the mirror's notes.

| Part | Where | Bytes per rank at TP=2 |
| --- | --- | --- |
| GDN recurrent state, 36 linear layers, fp32 | Snapshot `rec` | 36 x 24 heads x 128 x 128 x 4 = 56,623,104 |
| GDN conv windows (kernel 4) | Snapshot `conv` | 36 x 3 x 5,120 x 2 = 1,105,920 |
| n-gram (PLE) conv tail and history ids | Snapshot `ple_tail`, `history` | 3 x 3 x 10,240 x 2 = 184,320 |
| last row's hyper-connection streams (the MTP head reads it on resume) | Snapshot `tail` | 10,240 x 2 = 20,480 |
| indexer keys of each cache layer's incomplete block | Snapshot `ikc` | 13 x 4 x 128 x 2 = 13,312 |
| attention keys, 12 layers and the MTP layer | keeper `kc` | rows x kRow (bf16 512, FP8 272 with scales) |
| attention values | keeper `vc` | rows x vRow (bf16 512, FP8 256) |
| pooled indexer blocks | keeper `pooled` | rows / 4 x 256 |
| indexer keys of the incomplete block (ring sequences hold no older ones) | keeper `ikc` | at most 3 x 256 a layer |

The MTP layer holds `mtp_len` rows (`at - 1`: a resume absorbs the next prompt token at row `at - 1`), the others
`at`. The Snapshot is one 256-byte-aligned allocation of 57,947,136 bytes.

For a 100,000-token prompt, per rank:

| KV format | KV and pooled rows | Snapshot | Payload |
| --- | --- | --- | --- |
| FP8 (production) | 13 x 100,000 x (272 + 256 + 64) = 769.6 MB | 57.9 MB | 827,547,120 bytes (0.77 GiB) |
| bf16 | 13 x 100,000 x (512 + 512 + 64) = 1,414.4 MB | 57.9 MB | 1,472,346,624 bytes (1.37 GiB) |

The weights' format (NVFP4, INT4-AutoRound, EXL3) does not enter a kept state. It enters the file's identity, so a
state written by one checkpoint is never read by another.

## Files

`TENSORFOLD_SPILL_DIR/flashnext-<identity>/rank<r>/<key>-<at>.tfs`, directories mode 0700 and files 0600, because the
files hold prompt token ids. At start the tier opens `TENSORFOLD_SPILL_DIR`, the identity's directory and the rank's
directory one inside the other, refusing a link, and refuses to start (`SpillDirUnsafe`) unless each belongs to the
server's user or root and is not writable by group or others. Another user who can write there could otherwise put a
link where the server writes. Every file is then made, read, renamed and deleted relative to the rank's directory and
never through a link. `<identity>` is 16 hex digits of a SHA-256 over what the bits depend on: the checkpoint's
`config.json`, `model.safetensors.index.json` and `tokenizer.json`; every weight file's name, size and whole-shard
SHA-256 over every byte; the KV format, world size, context, drafts, parallel limit and YaRN,
the kernel set's `aot.json`, executable digest, VMM-ring mode, prompt chunk rows and sliced-fill setting. It also
includes the sorted, length-delimited names and values of all `TF_FLASHNEXT_*` and `FZ_*` environment variables,
GPU name, SM count, CUDA API compatibility version and the SHA-256 of the actual NVIDIA package/module version text
from `/proc/driver/nvidia/version`. Missing required identity inputs, including unreadable weights, executable or
driver-version text, fail closed when spill is enabled. Spill OFF skips identity, device/module and digest-cache work.
Ambiguous duplicate selected environment names (`TF_FLASHNEXT_*` or `FZ_*`) refuse enabled startup, even identical
values: libc's first-occurrence lookup must not be hidden by sorting. All unrelated duplicate names are ignored by
this identity selection. Identity diagnostics expose no raw environment values.
Enabled spill identities describe checkpoint bytes, not Hub snapshot directory labels or blob-link labels: those
labels and local source paths do not enter the agreed digest. The same metadata/shard names, sizes and content
produce the same identity through different Hub-style snapshot paths. A source mutation still invalidates identity;
a revision-looking directory name is never trusted as proof of immutable shard content.
`<key>` is a 64-bit Wyhash of the state's tokens.

Whole-shard digests are cached in the private `<TENSORFOLD_SPILL_DIR>/.identity-digests-v2` directory, opened before
the final tier identity is known (putting it inside an identity directory would be circular). A cached record is
selected by the canonical absolute source path, size, mtime in nanoseconds, ctime in nanoseconds, inode/device and
weight-index hash (the SHA-256 of `model.safetensors.index.json`). These source metadata fields are local cache
selectors only: they do not enter the agreed final identity, which consumes shard name, size and full content digest.
The source file's metadata must match before reuse; a cache miss or missing/checksum-failing/truncated record causes
a bounded-memory read hashing every byte of that shard. Unsafe or oversized cache records refuse enabled startup
rather than being followed or accepted. Weight sources are opened nonblocking and must be regular files, so a FIFO
cannot hang enabled startup. Metadata is checked before and after the read and on a cache hit, using both the open
file descriptor and the freshly resolved source path; a retargeted Hub symlink or replaced source is refused.
Checkpoint/index/tokenizer/kernel metadata reads use 64 KiB blocks, also checking the open descriptor and a freshly
opened source path before accepting the content digest. The weight-index selector hashes raw index bytes. Changed
ctime/inode/device invalidates a cached digest even after a same-size mutation restores the old mtime; stale metadata
is never a reason to reuse a sampled digest. This removes the old unsampled-byte limitation without rereading every
unchanged shard on every start.

The digest-cache root is absolute-only. Bootstrap validates every ancestor by no-follow directory descriptors,
refusing group/other-writable or foreign-owned ancestors; missing components are created privately. Final spill and
digest-cache directories must be mode 0700 and owned by the server UID. Unsafe links are refused. Records are
versioned and checksummed, written atomically with file fsync, rename and directory fsync, and capped at 256 records
with oldest-record removal under a file lock. Maintenance also runs at every cache open, including all-hit warm
starts, removing stale partials and enforcing the cap; deletion is followed by directory fsync. Source metadata
differs legitimately between the two machines, while
identical shard content contributes the same final digest. After bounded shard reads and spill payload reads, and
after fsync of writes, `DONTNEED` advice asks the OS to release those file pages. It is advisory, not a guarantee of
immediate page-cache eviction or measured free memory; qualification must measure that separately.

A file, little endian:

| Offset | Field |
| --- | --- |
| 0 | magic `TFSPILL1`, version 1, header size 4096 |
| 16 | identity (32 bytes), family name (16 bytes), rank, world |
| 72 | key, `at`, token count |
| 88 | payload offset and length |
| 104 | XXH3 of the tokens, XXH3 of the payload, the unix time it was written |
| 128 | the family's meta: length, then its bytes |
| 4088 | XXH3 of the header's first 4088 bytes |
| 4096 | the token ids (u32), padded to 4096 |
| payload | Flash Next: the Snapshot allocation, then per cache layer its K rows, V rows, incomplete-block indexer keys and pooled blocks |

Flash Next's meta holds the meta version, `at`, `mtp_len`, the n-gram history, the ring flag, and the row widths
and Snapshot size it was written with. A reader checks all of them before it allocates anything.

## Two ranks

Rank 0's store decides everything; rank 1 does what it is told on its own disk. The existing note channel
(`cuda_lanes.Note`, carried by `cuda_mirror` with the next call that has collectives) gains three notes:

- `spill key name`: write your copy of kept state `key` as file `name`.
- `unspill name`: delete file `name` (the cap needed room, or a read failed).
- `load name key`: read file `name` into a new kept state and call it `key`.

At the request's lookup rank 0 opens its file (header, identity and tokens checked), allocates the state's Snapshot
and sends `load`; rank 1 does the same with its own file when the note arrives, before the prefill call. The rows come
in the prefill that resumes the state, on both ranks in the same call (`cuda_lanes.resumeFrom`): the keeper sequence's
growth is a collective, so both ranks make it there, and both read their files side by side. Three agreements decide:
both ranks hold the state, both have room for its keeper, and both reads succeeded. Without room on either rank both
prefill from token 0 and the files stay for a later request. When a rank lacks the state or a read fails, both prefill
from token 0, rank 0's store forgets the entry and deletes its file, and rank 1 is told to delete its own. A read that
rank 0 refuses for memory at the lookup deletes nothing. Rank 1 never
decides from its own writes' progress. Its directory is capped at the same size as rank 0's, oldest first, so files
orphaned by a rank-0 failure go eventually.

## Policy

The settings are read from the environment on both ranks and are part of the settings text the ranks compare at
start, so mismatched ranks refuse to start together.

| Variable | Default | GLM name | Meaning |
| --- | --- | --- | --- |
| `TENSORFOLD_SPILL_GIB` | 0 (off) | `SPILL_GIB` | spill files a rank keeps, in GiB |
| `TENSORFOLD_SPILL_DIR` | `$HOME/.cache/tensorfold/spill` | `SPILL_DIR` | the same path on every machine, on local disk |
| `TENSORFOLD_SPILL_HIGHWATER` | 0.70 | `SPILL_HIGHWATER` | past this share of the prompt cache's budget, write the next victims early |
| `TENSORFOLD_SPILL_MIN_TOKENS` | 8192 | `SPILL_MIN_TOKENS` | shorter states are not written |
| `TENSORFOLD_SPILL_MIN_FREE_GIB` | 50 | `SPILL_MIN_FREE_GIB` | no write leaves less free disk than this |
| `TENSORFOLD_SPILL_FLUSH_S` | 60 | `SPILL_FLUSH_S` | a clean stop writes kept states for this long |
| `TENSORFOLD_SPILL_KEEP_BUILDS` | 1 | `--spill-keep-builds` | other identities' directories kept at start |

- A state is written when the store evicts it (its budget, a pass making room, admission shedding), if it holds at
  least `MIN_TOKENS` tokens, is not on disk yet, and is not superseded: no longer kept state of the same conversation is
  in memory and stored or writable. A late keep whose keeper is not made yet is not written.
- At idle, past `HIGHWATER`, the states eviction would take next are written in the background while they stay in
  memory, so a later eviction costs nothing. A state idle could not write (too short, superseded, the tier full) is
  not tried again at idle; it is counted as unwritten only when it leaves memory.
- Disk space is a byte LRU: before a write the least recently written or read files go until the new one fits
  `SPILL_GIB` and leaves `MIN_FREE_GIB` free beside the writes still queued.
- A lookup reads a disk state back only when it is longer than what memory holds for the prompt.
- On a clean stop (SIGTERM to rank 0's server) the kept states not on disk are written newest first for up to
  `FLUSH_S`, and the writers drain.
- At start, `.partial` files and files that fail the header checks are deleted, and other identities' directories past
  `KEEP_BUILDS` are deleted, the least recently opened first.
- States dropped by the per-conversation cap or the kept-bytes cap are not written: they are superseded turns or over a
  memory budget, not states leaving a full cache (Mia's patch 0089).

Writes run on a writer thread per rank. It copies device memory to a 64 MiB host buffer on a stream of its own and
writes it to `<name>.<random>.partial`, a file it creates and that must not exist yet, then fsyncs, renames and fsyncs the directory. A state being written keeps its device
memory until the next prompt pass, on both ranks alike: that pass waits for the copy and frees it before anything
else, because it may need that memory, and a write's progress never makes the ranks' memory differ. When such a state
leaves memory its write moves to the front of the queue (`Tier.hurry`). Admission does not count those states, since
the pass frees them first. Early and stop-time writes form a batch whose files never evict each other for room
(`Tier.pinBatch`). Reads run on the engine thread, inside the prefill, through a 64 MiB buffer into the new keeper and
Snapshot.

Admission sheds kept states before it refuses a request (`Store.admit`), and the states it sheds go to the tier like
any eviction. A keep that would take memory an admitted stream may still grow into is refused (`keepAt`); both ranks
agree on that refusal before a keeper is made, because rank 0's store may have freed states mid-pass that rank 1 frees
only on the next call. The per-conversation and kept-bytes caps apply before the next request's lookup or at idle,
never mid-pass, for the same reason.

## Health and metrics

`/health` gains a `prompt_cache` object: states, held and budget bytes, `kept_bytes`, `kept_bytes_cap`, `kept_mix`
(own turn states, the rest, conversations, tokens, the eight conversations holding the most), every store count
(hits, misses, kept, evicted, refused, failed, shed, recalled, capped_chat, capped_bytes) and a `spill` object: entries,
pending writes, stored, queued and cap bytes, free disk, and the tier's counters (written, write_failed, skipped, full,
no_space, deferred, unwritten_short, unwritten_superseded, unwritten_incomplete, loaded, load_failed, hash_failed,
forgotten, pruned, written_bytes, loaded_bytes, loaded_tokens, write_ns, load_ns, copy_wait_ns). `/metrics` gains
`tensorfold:prompt_cache_*` and `tensorfold:spill_*`. Rank 1 serves no HTTP; its log has its tier's start line.

## Failure modes

Every failure leaves a request prefilling from token 0, the path a cold cache takes.

| Failure | What happens |
| --- | --- |
| Another checkpoint, KV format, world size, kernel set or engine binary | Another identity directory: the file is never opened. |
| A file from this identity whose header, identity, rank, name or size does not match | Deleted at start or at read. |
| A cut write (crash, kill, power) | Only `.partial` files are cut; they are deleted at start. A renamed file was fsynced first. |
| Corrupt bytes | The token, header and payload hashes are checked; the payload hash after the copy, and a mismatch frees the state and deletes the file. |
| Disk full, or below the floor | The write is skipped or fails, the partial file is deleted, and the state is lost as before. |
| Rank 1 has no copy, or its read fails | Both ranks prefill from 0; both files go. |
| A 64-bit token hash collision | The header's tokens are compared with the prompt before anything is read. |

## Deviations from the GLM tier

- One writer thread per rank, with no O_DIRECT. GLM runs two writers with O_DIRECT.
- Reads run on the engine thread, so other streams pause while one reads (GLM reads beside decoding streams).
- Rank 0 decides and rank 1 follows its notes. GLM has every rank take the same decision from an identical index,
  with one all-gather per write to agree on the file size and the free-disk floor, and a reconcile of the ranks'
  indexes at start. Here each rank checks its own floor, and a rank-1 file missing after a restart fails its first
  read safely.
- XXH3 over the tokens, the header and the payload, instead of CRC-32 per 8 MiB.
- Environment variables only; no CLI flags.
- Rank 1's shutdown flush needs rank 0 to stop first: a SIGTERM to rank 1 still ends it at once.
- No probe of the engine's bits at start (upstream `--learn` hashes a probe prompt's state into its identity). The
  identity hashes the binary and the kernel set instead, so an upgrade starts a new directory.
- Files are written as the server's user. GLM gives them to the directory's owner. Here the directory must belong
  to the server's user or root: a host directory owned by a login user and mounted into a container that runs as
  root is refused. The qualification wrapper below runs as the host/helper owning UID:GID instead of container root.

## Authoritative replacement failure accounting

A replacement intent abandons after **any three synchronous `spillWrite` errors**, including `Incomplete`,
allocation failure and disk admission errors. Busy older writers or copies that are not ready are eligibility
checks, not attempts; asynchronous writer failures are not synchronous attempts. The cumulative
`spill_replacement_abandoned_total` counter records each abandoned intent once. A clean drain reports failure
for an abandonment during that drain, not for an earlier abandonment counted before the drain began.

## Remaining qualification limits

- Slow startup: the cold whole-shard hash runs before installStop() and before the link (cuda_native.zig:91-96). As PID 1 in the container, SIGTERM is ignored for those minutes. A failure on one rank leaves the other waiting at the link for up to 10 minutes.
- Unhashed runtime inputs: CUDA libraries from the image and the cubin bytes are not hashed. Cubins are covered only by name through aot.json. The plan's manual kernel inventory check covers this.
- Stale digest (P3): a stale hit would need a same-size in-place rewrite inside one coarse kernel timestamp tick with the same inode. The plan mounts the checkpoint read-only.

## GPU test plan (not run yet)

This needs an outage window: the test pair takes both Sparks' GPUs, so production stops first, with the user's
go-ahead (AGENTS.md). Spark-1 (192.0.2.10) runs rank 0, spark-2 (192.0.2.11) rank 1, as in production. The pair below
uses the ADM4 production image and kernel set, checkpoint, context, parallel limit, FP8, vision and TP settings,
with the separately built binary under test mounted over the image's, on port 8100. The test deliberately changes
the port, prompt-cache budget and spill settings; the interval-mark switch is varied below. Production is interval ON.
Use the reviewed `local-ai-server/tools/nvme_cache_measure.py` measurement tool (the schema-3 implementation at
local-ai-server commit `0bbc4664a20d2abfa6538b92fe623f8a659f62b7`) rather than the old multi-size spill checker. Transfer that committed script
separately to `~/spill-test/nvme_cache_measure.py`; record its source commit and SHA-256 with the evidence.
Its default is an offline plan. Engine phases send requests only with `--execute` and the approved-window,
nonproduction and private-spill acknowledgments. The isolate phase instead copies a selected stopped candidate file
offline with an additional stopped acknowledgment. The tool never manages server lifecycle.

Each candidate selects exactly one nominal size: 8192, 32768 or 131072. Actual prompt token counts and the measured
kept mark are recorded, not assumed equal to nominal size. Every size/interval combination has its own target
cache budget, spill minimum, new private spill directory and artifact prefix. Never run several sizes through one
fixed cache budget: two shorter states may coexist, leaving the supposed disk restore as a RAM hit.

### Before the window (no GPU)

Use the binary from the separately approved build of
`/home/bgoulder/tf-work/nvme-flashnext-spill-adm4`; do not reuse a binary from an older spill checkout.
The build and artifact transfer are prerequisites, not GPU qualification steps performed by this note.

On both Sparks, save the relevant production environment including its exact kernel path, and make the test
directories. Run the wrapper and offline isolation helper as the same owning host UID:GID. The container uses that
UID:GID, so the server can create its own private tier under the seven-argument host mount and can validate the
eight-argument helper-owned isolated root without relaxing ownership checks. A root-run container with login-user
mount ownership is not this layout. Checkpoint files are mounted read-only at `/models/huggingface`, not beneath
container `/root` (which the non-root UID cannot traverse). The kernel path and every required artifact must also
be readable by this UID. Use new private directories; do not chown or loosen existing root-owned/foreign artifacts.

```sh
umask 077
mkdir -p ~/spill-test/spill ~/spill-test/nvcache
docker inspect qwen38-fn-tf --format '{{range .Config.Env}}{{println .}}{{end}}' \
  | grep -E '^(TF_|TENSORFOLD_|NCCL_|HF_HUB_OFFLINE=)' > ~/spill-test/prod.env
```

On spark-1, copy the approved binary (transfer the reviewed local-ai-server measurement tool separately as above),
then check that both hosts hold the same binary:

```sh
scp bgoulder@192.0.2.11:tf-work/nvme-flashnext-spill-adm4/zig-out/native/bin/tensorfold-native ~/spill-test/
ssh bgoulder@192.0.2.11 'mkdir -p ~/spill-test && cp ~/tf-work/nvme-flashnext-spill-adm4/zig-out/native/bin/tensorfold-native ~/spill-test/'
sha256sum ~/spill-test/tensorfold-native; ssh bgoulder@192.0.2.11 sha256sum spill-test/tensorfold-native
```

Before launching the test pair in the separately approved window, record both ranks' kernel content hashes
(a sorted inventory of every kernel artifact's relative path and SHA-256, including `aot.json`) and the binary SHA-256.
Require the rank copies to agree and the kernel inventory to match the frozen ADM4 production recipe, not an old spill
kernel directory. Record the actual ADM4 image hash and its `tf.patches` label on both ranks, not just the tag; compare
with `local-ai-server/recipes/spark/flash-next-tp2-tensorfold/recipe.json`: image
`sha256:b75ed24091efa8ba6ead78bd0053c9191a9d985fd8c2fafeb42c6bb974d4f8c5`, patch set `5c37b6a458f7`.
A changed or missing artifact is a failed prerequisite: do not launch against a guessed kernel path or relabel an image.
These are future window checks; the CPU-only synthetic launch regression is separate evidence and does not
qualify production artifact contents, GPU execution, a link/listener, or inference. It does not query or change
production services/containers. Record its probe/source hashes, literal documented argv and owned-container
startup/absence artifacts separately; a CPU startup marker is not a GPU qualification result.

On both Sparks, write `~/spill-test/run.sh`:

```sh
#!/bin/sh
# run.sh RANK MODE INTERVAL SIZE CACHE_GIB MIN_TOKENS STORE [SPILL_ROOT]: the test pair's rank RANK (0 on spark-1, 1 on spark-2); MODE fresh (no prompt cache)
# or spill (the selected per-size measured one-state cache and spill tier)
RANK=$1; MODE=$2; INTERVAL=${3:-on}
OWNER_UID=$(id -u); OWNER_GID=$(id -g)
SIZE=${4:?select one nominal size}; CACHE_GIB=${5:?supply this candidate target cache budget}
MIN_TOKENS=${6:?supply this candidate spill minimum}; STORE=${7:?new private per-size store name}
SPILL_ROOT=${8:-$HOME/spill-test/spill/$STORE}
case "$SPILL_ROOT" in /*) ;; *) echo "SPILL_ROOT must be absolute" >&2; exit 2 ;; esac
case "$SIZE" in 8192|32768|131072) ;; *) echo "unsupported SIZE" >&2; exit 2 ;; esac
case "$INTERVAL" in
  off) MARK="" ;;
  on) MARK="--prompt-interval-mark" ;;
  *) echo "INTERVAL must be off or on" >&2; exit 2 ;;
esac
# Preserve the exact production kernel path captured in prod.env; never fall back to an older image's kernels.
KERNELS=$(sed -n 's/^TENSORFOLD_CUDA_KERNELS=//p' "$HOME/spill-test/prod.env")
test -n "$KERNELS" || { echo "production kernel path missing from prod.env" >&2; exit 2; }
if [ "$#" -ge 8 ]; then
  test -d "$SPILL_ROOT" || { echo "isolated SPILL_ROOT missing" >&2; exit 2; }
  SPILL_MOUNT="$SPILL_ROOT:/spill/tier"
else
  (umask 077; mkdir -p "$SPILL_ROOT")
  SPILL_MOUNT="$SPILL_ROOT:/spill" # engine creates the private tier child for a new disk candidate
fi
if [ "$MODE" = fresh ]; then CACHE=0; SPILL="-e TENSORFOLD_SPILL_GIB=0"; else CACHE="$CACHE_GIB"; SPILL="-e TENSORFOLD_SPILL_GIB=200 -e TENSORFOLD_SPILL_DIR=/spill/tier"; fi
docker rm -f qwen38-fn-spill >/dev/null 2>&1
exec docker run -d --name qwen38-fn-spill --user "$OWNER_UID:$OWNER_GID" --gpus all --network host --ipc host --device /dev/infiniband \
  --ulimit memlock=-1 --cap-add IPC_LOCK --shm-size 64m --env-file "$HOME/spill-test/prod.env" $SPILL \
  -e TENSORFOLD_SPILL_MIN_TOKENS="$MIN_TOKENS" -e TENSORFOLD_CUDA_KERNELS="$KERNELS" -e CUDA_CACHE_PATH=/cache/nv -v "$HOME/.cache/huggingface:/models/huggingface:ro" \
  -v "$HOME/spill-test/nvcache:/cache" -v "$SPILL_MOUNT" \
  -v "$HOME/spill-test/tensorfold-native:/usr/local/bin/tensorfold-native:ro" \
  tensorfold-qwen38fn:zig-db28187-adm4-candidate tensorfold-native serve \
  /models/huggingface/hub/models--nvidia--Qwen3.8-Flash-Next-NVFP4/snapshots/fc694b54fb0174e0913e6adf86691ef85a4ead47 \
  --name qwen-3.8-flash-next --host 127.0.0.1 --port 8100 --context 262144 --parallel 14 --max-tokens 32768 \
  --thinking --kv-dtype fp8 --vision --vision-max-images 50 --vision-max-videos 4 --vision-image-tokens 16384 \
  --tp 2 --rank "$RANK" --master 192.0.2.10 --master-port 29551 --prompt-cache-gib "$CACHE" $MARK
```

The pair starts with rank 1 first; the checker waits for nothing, so wait for `/health` yourself. A clean stop is
rank 0 first, so rank 1 hears the stop and drains its own writes:

```sh
# start (on spark-1)
ssh bgoulder@192.0.2.11 sh spill-test/run.sh 1 MODE INTERVAL SIZE CACHE_GIB MIN_TOKENS STORE SPILL_ROOT_RANK1 \
  && sh ~/spill-test/run.sh 0 MODE INTERVAL SIZE CACHE_GIB MIN_TOKENS STORE SPILL_ROOT_RANK0
until curl -sf 127.0.0.1:8100/health | grep -q '"warming": *false'; do sleep 10; done
# clean stop (on spark-1): rank 0's flush, then rank 1 follows
docker stop -t 120 qwen38-fn-spill; ssh bgoulder@192.0.2.11 'timeout 180 docker wait qwen38-fn-spill; docker rm -f qwen38-fn-spill'
docker rm -f qwen38-fn-spill
```

### Candidate-only GPU prerequisites (future approved window; not qualified here)

The stored production launch sources `local-ai-server/logs/tensorfold/20261008/zig/dual/start.sh`,
`scripts/config.sh` and `scripts/prepare.sh` select `/opt/tensorfold/share/tensorfold/cuda/sm121`,
`--gpus all`, host IPC/network, `/dev/infiniband`, `IPC_LOCK` and unlimited memlock. These are stored-source
facts, not observations of a live container. Do not query or change `qwen38-fn-tf` for these checks.
Perform the following only inside the separately approved, nonproduction `CANDIDATE` on each rank, before
sending any qualification request; save literal commands, stdout/stderr, exit codes and candidate identity.

- Record `id` with UID:GID **1000:1000** and all supplementary groups. Enumerate the actual character-device
  nodes under `/dev/nvidia*` (including `/dev/nvidia-caps/*`) and `/dev/infiniband/*`; record each node's
  owner, group, mode and major/minor numbers. As 1000:1000, open and close **each** node with
  `O_RDWR | O_CLOEXEC`, recording success or the precise errno. Device presence alone is not access proof;
  any required-node refusal fails the prerequisite. These device opens are GPU-window work, not CPU dry-run checks.
- Record `/proc/self/status` `CapEff` and whether effective `IPC_LOCK`, bit 14 (`0x4000`), is actually present.
  Independently record `/proc/self/limits` and require both soft and hard **Max locked memory** limits to be
  unlimited. `--ulimit memlock=-1` does not grant the capability, `--cap-add IPC_LOCK` does not raise the limits,
  and the requested capability may not survive `--user`. If the effective bit is absent, establish that the
  required-size locked/pinned allocation actually succeeds under the limits, or fail and explicitly adjust the
  approved candidate. A small `mlock` test does not qualify a large pinned allocation.
- Record the literal container `HOME`, the 1000:1000 passwd lookup/home, and traversal permissions for their
  ancestors and all configured cache/spill paths. The serving launch has no explicit `HOME`; the download helper's
  `HOME=/tmp` is not the serving environment, and `--user` does not rewrite `HOME`. Demonstrate as 1000:1000 that
  the configured cache/spill directories are writable and private where required. An inaccessible `/root` or
  foreign-owned ancestor is a failed prerequisite, not permission to chown or loosen existing artifacts; resolve
  it with a separately reviewed explicit path/environment before launching the candidate.
- Require literal `TENSORFOLD_CUDA_KERNELS=/opt/tensorfold/share/tensorfold/cuda/sm121`. As 1000:1000, read its
  `aot.json` and **every** referenced `cubins/<hash>.cubin`; inventory relative names and full SHA-256 content
  hashes on both ranks and compare with the frozen ADM4 inventory. Require the image's
  `/opt/tensorfold/native/bin/tensorfold-native` and the mounted candidate binary to be readable/executable by
  1000:1000. A readable directory or manifest alone does not prove cubin readability or contents.

No device access, capability/memlock, serving HOME or production-kernel check above was executed by the CPU
startup probe. Its markers do not qualify any of these GPU prerequisites.

### In the window

Only after the user approves the production outage, use the pair start/clean-stop sequence above and run each
checker phase detached on spark-1. Keep the same binary, checkpoint, ADM4 kernel set, prompt chunk rows, sliced-fill
setting and both ranks' environment throughout a comparison. Save both ranks' start lines, identity paths, settings,
`/health` counters before/after each request and checker outputs; do not infer a disk load from timing alone.

### Interval OFF then ON, and crossing files

Run OFF first, then ON second, separately at nominal sizes 8192, 32768 and 131072. For each size qualify fresh, disk
and restart with a per-size configuration. A new private spill directory is required for each size/interval run on
BOTH ranks, for example a unique `STORE=8192-off-RUNID`; do not point at production or old spill files. Prefix every
artifact with the same size/interval/run identifier (`8192-off-RUNID-fresh.json`, `...-disk.json`, `...-restart.json`).
The illustrative names `${SIZE}-off-fresh.json`, `${SIZE}-on-fresh.json`, `${SIZE}-off-disk.json`,
`${SIZE}-on-disk.json`, `${SIZE}-off-restart.json` and `${SIZE}-on-restart.json` below also include a unique run id
in actual evidence. Restart/crossing use fresh paired isolated roots from that same size's source evidence, not the
original writable store. The launch override SPILL_ROOT_RANK0/SPILL_ROOT_RANK1 must equal the corresponding manifest
root and is mounted at /spill/tier; omit the override only for a new cold/disk candidate store.

Before starting a size, print the tool's offline plan using one `--size` and `--interval-mark`; choose `CACHE_GIB`
and `MIN_TOKENS` from its candidate configuration, and apply those exact values to both ranks and every phase's tool
arguments. Default spill minimum is 4096 for size 8192, 8192 for the other sizes. Estimated payload-derived budgets
are starting candidates, NOT qualification. The disk phase must verify actual measured charges: the target and
evictor individually fit the measured cache budget but cannot coexist. It checks the actual measured kept mark,
cache-held/budget bytes, `cache_evicted`, spill writes and `loaded_bytes` to prove a disk read, not a memory hit.
If those checks fail, discard that candidate's results and choose a new per-size budget/minimum and a new private
store; rerun cold -> disk -> restart. Do not relax counters, silently change configuration or reuse the old baseline.

For one size, record an immutable `CANDIDATE` identifier containing source/binary/image/checkpoint/kernel hashes.
Use exactly the same candidate identity, model, generation limit, size, target cache budget, spill minimum and
interval option across its cold, disk and restart measurements. The actual cold server alone has cache/spill OFF;
the cold JSON still binds the intended disk/restart target configuration.

```sh
# Future approved-window example on spark-1; manual pair lifecycle is still required between phases.
# SIZE is ONE of 8192/32768/131072; all values below come from this candidate's offline plan and measured setup.
python3 ~/spill-test/nvme_cache_measure.py --phase cold --size "$SIZE" --interval-mark "$INTERVAL" \
  --prompt-cache-gib "$CACHE_GIB" --spill-min-tokens "$MIN_TOKENS" --candidate-id "$CANDIDATE" \
  --url http://127.0.0.1:8100 --max-tokens 64 \
  --execute --ack-window approved-nonproduction-window --ack-nonproduction --ack-private-spill \
  > "$PREFIX-fresh.json"
# Clean-stop fresh pair, then manually start MODE=spill with this same SIZE/config and a NEW empty STORE.
python3 ~/spill-test/nvme_cache_measure.py --phase disk --size "$SIZE" --interval-mark "$INTERVAL" \
  --prompt-cache-gib "$CACHE_GIB" --spill-min-tokens "$MIN_TOKENS" --candidate-id "$CANDIDATE" \
  --url http://127.0.0.1:8100 --max-tokens 64 \
  --execute --ack-window approved-nonproduction-window --ack-nonproduction --ack-private-spill \
  < "$PREFIX-fresh.json" > "$PREFIX-disk.json"
# Clean-stop both ranks and drain their writes. Do NOT restart against the original writable store.
# Run the offline isolation steps below, then start the same-mode candidate with each rank's manifest root.
python3 ~/spill-test/nvme_cache_measure.py --phase restart --size "$SIZE" --interval-mark "$INTERVAL" \
  --prompt-cache-gib "$CACHE_GIB" --spill-min-tokens "$MIN_TOKENS" --candidate-id "$CANDIDATE" \
  --url http://127.0.0.1:8100 --max-tokens 64 \
  --execute --ack-window approved-nonproduction-window --ack-nonproduction --ack-private-spill \
  < "$PREFIX-restart-input.json" > "$PREFIX-restart.json"
```

### Offline selected-file isolation before restart and crossing

A historical disk restore at I can promote a longer mark H during the final disk request; the clean stop may
persist H. Reusing that original root could select H on restart, so historical disk evidence is not current file
selection evidence. Restart and crossing require post-flush evidence for one selected persistent file on both ranks.
The expected cut is the validated, explicitly isolated I, not the historical assumption that no promotion occurred.
If the I file vanished, reject the candidate; never substitute H or silently relabel a historical I artifact.

Only after a MANUAL clean stop and write drain, use `--phase isolate` separately on each Spark. The helper never
stops engines, changes ownership, or chooses a different file. Source file and its final private parent must already
be owned by the helper UID; scratch parent must also be private and owned by that UID. Ancestors must be owned by
root or the helper UID and not group/other-writable. Source must be a regular private file (normally 0600) with
nlink=1; source path/parent must pass the helper's no-follow/private-owner rules. The wrapper's explicit owning UID:GID must match the helper UID that produced each rank's manifest. If existing
files or ancestors are incompatible with that UID, stop qualification; do not automatically chown files, relax
privacy checks, or escalate.
No production/old spill root is allowed. The isolation command requires `--ack-candidate-stopped` in addition to the
standard window/nonproduction/private-spill acknowledgments, none of which grants human outage approval.

For each restart or crossing, create a NEW paired isolation with the same 32-lowercase-hex `ISOLATION_ID` on both
Sparks. Supply the entire validated source disk JSON on stdin and explicitly select the historical I `.tfs` file.
The helper creates exclusive/no-follow roots under `SCRATCH_PARENT`, copies exactly that file, validates full-file
SHA-256, header rank/world/at/key/identity and actual file token bytes, and refuses missing/unsafe/mismatched inputs.
The new root contains only that one chosen file on each rank, with no longer H file; never reuse a mutated isolation
root from a previous completion.

```sh
# Future offline author-only example, run on EACH Spark after manual candidate stop; RANK is 0 or 1.
# SOURCE_STATE is that rank's private stopped candidate .tfs at I; all source candidate args are unchanged.
python3 ~/spill-test/nvme_cache_measure.py --phase isolate --size "$SIZE" --interval-mark "$SOURCE_INTERVAL" \
  --rank "$RANK" --source-state "$SOURCE_STATE" --scratch-parent "$SCRATCH_PARENT" \
  --isolation-id "$ISOLATION_ID" --ack-candidate-stopped \
  --prompt-cache-gib "$CACHE_GIB" --spill-min-tokens "$MIN_TOKENS" --candidate-id "$CANDIDATE" \
  --max-tokens 64 --execute --ack-window approved-nonproduction-window --ack-nonproduction --ack-private-spill \
  < "$SOURCE_DISK" > "$PREFIX-isolate-rank$RANK.json"
```

The phase=isolate/schema1 manifest records rank/world/at/key/identity, token_count, token_prefix_sha256,
file_sha256, file_bytes, root, isolation_id, source_disk_sha256, copy_written_bytes and files (one relative path).
The root layout is `<scratch-parent>/nvme-isolate-<ID>/flashnext-<identity[:16hex]>/rankR/<key16hex>-<I>.tfs`:
the identity suffix is the first eight identity BYTES rendered as 16 lowercase hex characters, not eight characters.
The helper's SHA-256 proves copy/readback equality, not native XXH3 checksum validity; successful paired engine
loading must still qualify native checksums. Pair `rank0` and `rank1` manifests with common identity/key/I/prefix
digest/isolation id and the exact source disk
artifact hash. Each rank's file SHA and size are checked independently; rank payloads need not be byte-identical.
Absolute root paths may differ between hosts. Map each manifest root to that rank's launch SPILL_ROOT; never assume
rank 1 uses rank 0's path, and do not launch against the original store instead.

The schema-3 source measurement binds `/v1/tokenize` for the exact request body to a full LE-u32 prompt SHA-256
and cached-I prefix SHA-256, with completion usage token count/type agreement. Actual `.tfs` token bytes must match
that `cached_prefix_sha256`; at == token_count == I under CUDA cache_rules.lookahead=0. A prompt text SHA alone is
not this token-prefix proof. Old schema-2 artifacts and unpaired/missing manifests are rejected.

```sh
# On the coordinator, assemble restart evidence from the source disk artifact and both offline manifests.
python3 -I - "$SOURCE_DISK" "$RANK0_MANIFEST" "$RANK1_MANIFEST" <<'PY' > "$PREFIX-restart-input.json"
import json, sys
with open(sys.argv[1]) as source, open(sys.argv[2]) as rank0, open(sys.argv[3]) as rank1:
    json.dump({"source_disk": json.load(source),
               "isolation": {"rank0": json.load(rank0), "rank1": json.load(rank1)}}, sys.stdout)
PY
# Manually launch same-mode candidate using the TWO manifest roots, then run the restart command above.
```

Disk takes its own measured cold JSON on stdin; restart takes source validated disk JSON plus paired isolation
manifest evidence, not a bare disk JSON, cold JSON or estimate. The tool rejects different candidate configuration or generation arguments. Compare ON and OFF cold
`token_sha` at the same nominal size, then every memory/disk/restart digest to its own destination-mode cold digest.
Require exact generated-token and actual-prompt equality, not merely matching reply text or approximate speed.

For OFF -> ON, freshly isolate the source OFF file I on BOTH ranks and start interval ON against those manifest
roots. Use the explicit `--phase crossing` interface with source validated disk JSON, destination cold baseline
and paired isolation evidence. For ON -> OFF, reverse that direction, freshly isolating ON I and supplying OFF cold
evidence. The source retained mark I is the expected restored cut; the independent destination same-mode disk run
may have retained a different cut J. Do not require J == I or use destination disk evidence as the crossing baseline.
Save separate crossing outputs and never edit a baseline to make a configuration mismatch pass.

The crossing envelope has exactly three inputs: `source_disk`, `destination_cold`, and `isolation` containing
`rank0` and `rank1`. The interval-only configuration exception allows only `candidate_config.interval_mark` to
toggle OFF <-> ON, and requires it actually to differ. Candidate-id, model, max_tokens, nominal size,
prompt_cache_gib, spill_min_tokens, kv_dtype, TP and payload estimate must match. Both inputs must identify the same
actual prompt tokens/count. The source retained mark must remain eligible under both spill minima and fit the
one-state cache capacity. If configurations differ beyond the toggle, qualify a new common per-size configuration
rather than altering evidence. Crossing must restore exactly I from the selected isolated file, count exactly one
disk load with positive loaded bytes, and match destination cold generated-token digest and output count.
The normal restart remains strict: identical interval/configuration and the exact isolated I from its source disk
baseline. Restore telemetry must begin with an empty memory cache and exactly one disk entry; native loaded_tokens
and loaded_bytes must equal the chosen I and selected file_bytes. Isolation proves the chosen persisted cut; actual
disk-load counters and output checks still prove reuse.

```sh
# Future approved-window example: FIRST make a fresh paired isolation, then launch destination INTERVAL
# using each manifest root, not the original source store or a previously used isolation root.
python3 -I - "$SOURCE_DISK" "$DESTINATION_COLD" "$RANK0_MANIFEST" "$RANK1_MANIFEST" <<'PY' > "$PREFIX-crossing-input.json"
import json, sys
with open(sys.argv[1]) as source, open(sys.argv[2]) as destination, open(sys.argv[3]) as rank0, open(sys.argv[4]) as rank1:
    json.dump({"source_disk": json.load(source), "destination_cold": json.load(destination),
               "isolation": {"rank0": json.load(rank0), "rank1": json.load(rank1)}}, sys.stdout)
PY
python3 ~/spill-test/nvme_cache_measure.py --phase crossing --size "$SIZE" --interval-mark "$INTERVAL" \
  --prompt-cache-gib "$CACHE_GIB" --spill-min-tokens "$MIN_TOKENS" --candidate-id "$CANDIDATE" \
  --url http://127.0.0.1:8100 --max-tokens 64 \
  --execute --ack-window approved-nonproduction-window --ack-nonproduction --ack-private-spill \
  < "$PREFIX-crossing-input.json" > "$PREFIX-crossing.json"
```

The reviewed schema-3 tool source at local-ai-server commit `0bbc4664a20d2abfa6538b92fe623f8a659f62b7` implements
token binding and paired isolation. Record its source SHA and transferred script SHA-256 before qualification.
Source inspection verified the isolate CLI, manifest/envelope checks, token-prefix binding, one-entry start and
native loaded token/byte checks. This is not runtime qualification; no actual file isolation or GPU measurement
was performed while updating this document. Earlier schema-2 source `794ebbf545d011aa06f19357ef9172f1a0a42129`
is not sufficient for this persistence proof.

Check the same identity on each rank across the toggle: interval marks are planner-only, so toggling only
`--prompt-interval-mark` does not change file identity while all hashed inputs stay fixed. For each successful
cross-read require `cached_tokens > 0`, positive `loaded`/`loaded_bytes` deltas and `token_sha` exactly equal to the
destination-mode cold baseline. No eligible file means cold prefill, not a corrupt-file claim, and is not proof of
crossing qualification.

Also qualify identity isolation: retain the old files and change a hashed input in the approved test pair (for
example the actual kernel set or binary), then obtain that configuration's own fresh baseline for that same size.
Restart with the retained store and require a different identity, `cached_tokens = 0` and `loaded delta = 0` for an
otherwise cold request; its `token_sha` must match its own cold baseline. Old files remain unopened under their old
identity, rather than being accepted, relabelled or reported as a successful load. This negative cold check is not a
successful `--phase restart` result (which requires a disk read); record its raw request/health evidence separately.
Record before/after identity paths and input hashes so an accidental binary/kernel/environment change cannot
masquerade as an interval-only test.

These are required future GPU qualification checks, not results already obtained. The tool's acknowledgments do
not themselves grant outage consent. After every requested per-size phase and crossing check completes, restore
production and confirm one completion for `qwen-3.8-flash-next` through the proxy, as every outage ends.

### What passes

- Every per-size cold/disk/restart JSON has `measured: true` and the exact expected candidate configuration. Tool
  failure, a plan-only JSON, a failed one-state/eviction check or changed configuration is not a result to promote.
- Disk proof shows each state individually fits but their actual measured charges cannot coexist, an eviction and
  spill write occurred, and exactly one disk load with positive read bytes restored the measured kept mark.
- Generated-token digest, prompt digest/count and output count agree with the corresponding measured cold baseline;
  restart and crossing match the paired isolated source disk mark; crossing output matches destination cold.
  Counter deltas, floors and write budgets pass; rank-0 HTTP counters
  do not claim rank-1 totals, so collect both ranks' logs and disk/identity evidence separately.
- Each size/interval/crossing output retains observed TTFT and its cold-to-restore ratio. Do not infer correctness
  from speed or reuse one larger size's cache budget for smaller prompts.
- Both ranks' start logs identify their own tier/identity and no failed note; binary/kernel hashes and actual image
  hash/patch label match the approved frozen candidate. Identity-isolation cold evidence is retained separately.

### After the window

The files hold prompt token ids. Delete only the approved private qualification artifacts. The command below
cleans the original test-store subtree only, not isolated roots elsewhere; retain each manifested isolated-root
path for separately approved cleanup by its owning UID after evidence is saved. Do not inventory/delete broader
scratch or production roots. New example stores belong to the same host/helper UID used by the container; cleanup
through the image must use that UID:GID too. Existing incompatible artifacts are not silently reassigned:

```sh
docker run --rm --network none --user "$(id -u):$(id -g)" -v ~/spill-test/spill:/spill --entrypoint sh tensorfold-qwen38fn:zig-db28187-adm4-candidate -c 'rm -rf /spill/*'
ssh bgoulder@192.0.2.11 'docker run --rm --network none --user "$(id -u):$(id -g)" -v ~/spill-test/spill:/spill --entrypoint sh tensorfold-qwen38fn:zig-db28187-adm4-candidate -c "rm -rf /spill/*"'
```

Record the results as a manifest in `fleet/trials/` of local-ai-server.
