# Trunk refresh after isolated boundary trials

The task branch now includes trunk fe90b9defaec82e830872f405762465ce8f3e960.
The actual rebase ran on Spark 2. Root imported the verified result at
5d4ef50e37986d14ac8cc351bff9ad0b4d615084. The sole committed tree difference
from the pre-rebase source930cf50a84f3820bd9b96a29f851ed53a0c7c9cb is the
inherited docs/glm-cuda-target-graphs.md addition of48 lines. No executable
source changed, so existing measured binaries retain their original pins.

All122 original commits map to122 result commits with original authors and
messages preserved. Ninety-two object IDs remain unchanged and30 were
rewritten. Full comparison covers5467 tracked files and196985374 checkout
bytes. Every file except that GLM document retains its blob, mode and bytes.
The GLM document's result blob equals trunk's blob. The shared upstream
Engine and its local overlay remain the implementation boundary.

Root preserved the original history at
archive/qwen35-before-trunk-fe90-20261019. The imported remote result is also
retained at import/qwen35-spark2-trunk-fe90-20261019. Root verified the bundle,
its148168522 bytes and SHA256
a7e2611fe1d45207459514c91f5e003178eb348da7053aeda5e35c96fce00803,
then used reset --keep to select the remotely rebased result. No local rebase
ran. Both pending tracked documentation files kept their exact hashes, and
the index remained free of staged changes. Historical source objects used
by frozen audits remain reachable through the archive ref.

The [evidence archive](evidence/upstream-trunk-refresh-20261019/README.txt)
contains67 files,1891884 compressed bytes, SHA256
2bf5e211e55d418f19db12b0a1b8c8978a0ac6f415bd4e5f4aa0dffd3a083586.
It includes remote command receipts, the complete commit mapping and source
roster, worker checks and root import assertions. Input and result Git bundles
remain externally pinned. The archive verifier passes. The owned Spark 2
directory and all original refs/history remain preserved. All owned commands
ended successfully. No compiler, GPU or production service action ran.

This content-equivalent import does not qualify a new runtime or change the
independent full-logit HF failure. Existing native parity, timing and isolated
semantic evidence retain their original scope and source pins.
