"""Capture immutable Qwen checkpoint headers without downloading weight bodies."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import struct
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path


def remote_bytes(url: str, start: int, size: int) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"Range": f"bytes={start}-{start + size - 1}", "Accept-Encoding": "identity"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        if response.status != 206:
            raise ValueError(f"range request returned HTTP {response.status}")
        content_range = response.headers.get("Content-Range", "")
        if not content_range.startswith(f"bytes {start}-{start + size - 1}/"):
            raise ValueError(f"unexpected range {content_range!r}")
        data = response.read(size + 1)
    if len(data) != size:
        raise ValueError(f"expected {size} bytes, received {len(data)}")
    return data


def capture(repo: str, revision: str, metadata: Path, output: Path, local: Path | None) -> dict:
    if not re.fullmatch(r"[a-f0-9]{40}", revision):
        raise ValueError("an immutable 40-character revision is required")
    index = json.loads((metadata / "model.safetensors.index.json").read_text(encoding="utf-8"))
    headers = {}
    sources = []
    for shard in sorted(set(index["weight_map"].values())):
        if Path(shard).name != shard or not shard.endswith(".safetensors"):
            raise ValueError(f"unsafe shard name {shard!r}")
        path = local / shard if local else None
        url = f"https://huggingface.co/{repo}/resolve/{revision}/{urllib.parse.quote(shard)}"
        if path and path.is_file():
            with path.open("rb") as source:
                first = source.read(8)
                size = struct.unpack("<Q", first)[0]
                if size > 64 << 20:
                    raise ValueError("header exceeds 64 MiB")
                data = source.read(size)
        else:
            size = struct.unpack("<Q", remote_bytes(url, 0, 8))[0]
            if size > 64 << 20:
                raise ValueError("header exceeds 64 MiB")
            data = remote_bytes(url, 8, size)
        if len(data) != size:
            raise ValueError(f"truncated header {shard}")
        tensors = json.loads(data)
        count = 0
        for name, tensor in tensors.items():
            if name == "__metadata__":
                continue
            if name in headers or index["weight_map"].get(name) != shard:
                raise ValueError(f"duplicate or misindexed tensor {name}")
            headers[name] = [tensor["dtype"], tensor["shape"]]
            count += 1
        sources.append({"file": shard, "header_bytes": size, "header_sha256": hashlib.sha256(data).hexdigest(), "tensors": count})
    if headers.keys() != index["weight_map"].keys():
        raise ValueError("header tensor names do not match the index")
    output.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(headers, sort_keys=True, separators=(",", ":")).encode()
    (output / "headers.json.gz").write_bytes(gzip.compress(raw, mtime=0))
    config_bytes = (metadata / "config.json").read_bytes()
    (output / "config.json").write_bytes(config_bytes)
    config = json.loads(config_bytes)
    quant = config.get("quantization_config", {})
    report = {
        "repo": repo, "revision": revision, "sources": sources,
        "capture_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "headers_sha256": hashlib.sha256(raw).hexdigest(),
        "tensors": len(headers), "dtype_counts": dict(Counter(t[0] for t in headers.values())),
        "quant_method": quant.get("quant_method"), "quant_algo": quant.get("quant_algo"),
        "tensor_bytes": index.get("metadata", {}).get("total_size"),
    }
    (output / "source.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo")
    parser.add_argument("revision")
    parser.add_argument("metadata", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--local", type=Path)
    args = parser.parse_args()
    print(json.dumps(capture(args.repo, args.revision, args.metadata, args.output, args.local), indent=2))


if __name__ == "__main__":
    main()
