# Windows startup object compile and real GLM image evidence

The actual native startup object compile passes all four build steps with the real 94-image bundle. Commit `fcca5ecb41d366491ccf5ce1a16d132f953d4a64` changes only `zig/src/families/flashnext/cuda_ple.zig`. The guard skips the unsupported Windows POSIX random-mapping hint. On Linux, the existing `std.posix.madvise` call and its arguments remain unchanged. Checkpoint validation, admission, tensor selection and inference arithmetic are unchanged by this guard.

The [original evidence archive](evidence/upstream-windows-startup-20261020/archive.json) preserves the red and green compile logs, source before/after bytes, exact guard diff, compiler logs, dependency/source rosters, original failed receipts, successful static replay, fmt and lint receipts. The [external readbacks](evidence/upstream-windows-startup-20261020/external-readbacks.json) pin real binaries and workstation dependencies by size and SHA256. The owned inventory verifies 316 files and 128 external compiler/header/reference dependencies. Original attribution and inherited image provenance are retained.

The old accepted bundle has 93 images. The new isolated bundle copies all 93 byte-for-byte and adds `glm_target_dsa.fatbin`. Its size is 635688 bytes and SHA256 is `f2a68978452f2771e0aa860a783cf558151c9f9635c1b3e5aa21664ad16e17dd`. The original bundle is unchanged. CUDA 12.8 V12.8.93 with MSVC 14.44.35207 compiled the new image for `sm_120` using the actual catalog flags. The frozen source is `f67d2b4d91241da5aebef3ef1a1d059be89b5f29`. All 144 CUDA source/catalog files match the earlier accepted image-source snapshot.

The consumed source/header closure declares 34 global exports. All 34 appear exactly once in the real ptxas log, including every one of the 14 exports requested by the GLM consumer. The static verifier preserves each entry's register, shared-memory, stack and spill records. The compiler log retains existing floating-point sentinel warnings. No CUDA source changed. `cuobjdump` was unavailable, so this result makes no disassembly or runtime geometry claim.

All failed attempts remain visible. The red startup compile fails on the missing image and unsupported Windows `MADV.RANDOM`. The first image builder compiles successfully but its post-compile check fails because it expects only the 14 direct consumer exports. The independent verifier accounts for 20 further exports from consumed headers and matches the complete 34-entry roster. The first Ruff run fails on import order and `re.S` spelling. Exact executed scripts are retained beside the corrected maintained scripts. The final static verifier replay reproduces the original accepted verification object exactly. No failed receipt is rewritten as green.

The green Zig command compiles the real `native.open` entry and returned close body into a fast/native startup object. Its receipt reports 4/4 steps succeeded. Zigfmt, fleet fast checks and maintained artifact Ruff pass in the retained receipts. Compiler, build and check PIDs are observed ended. No GPU, model, controller or service runs during this compatibility work or evidence packaging.

```text
C:/Users/developer/tools/zig-0.17.0/zig-x86_64-windows-0.17.0/zig.exe build compile-modelopt-startup -Dfatbins=C:/Development/tf/qwen35-ornith-nvfp4/.artifacts/qwen35-windows-startup-buildfix-20261020/fatbins-sm120 -j2 --summary all
```

Packaging reruns only the independent static verifier, pin checks and archive verifier. It does not rerun the compiler. From the TensorFold root, the following commands replay those static checks with the repository Python and a fresh output path.

```text
C:/Development/local-ai-server/runtime/proxy-venv/Scripts/python.exe .artifacts/qwen35-windows-startup-buildfix-20261020/verify_image.py --output <new-windows-image-verification.json>
C:/Development/local-ai-server/runtime/proxy-venv/Scripts/python.exe docs/plans/qwen35-ornith/evidence/windows-v110-20261019/verify_archive.py docs/plans/qwen35-ornith/evidence/upstream-windows-startup-20261020
```

The archive verifier checks preserved bytes. Static semantic replay requires the original workspace paths and all external pins. A native executable/server build and explicitly approved runtime qualification remain required. Startup object compilation does not qualify GPU numerics, model quality, serving, performance, capacity or promotion. The original independent HF gate remains FAIL.
