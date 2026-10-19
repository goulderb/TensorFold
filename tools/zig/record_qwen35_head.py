"""Record CPU-only head inputs and ModelOpt reference outputs for a thin CUDA probe."""
from __future__ import annotations

import argparse
import hashlib
import json
from contextlib import ExitStack
from pathlib import Path

import torch
from qwen35_reference import Store, environment_receipt, sha


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    columns = torch.arange(2048, dtype=torch.int64)
    inputs = torch.stack([torch.zeros(2048), *[
        (((columns * 17 + salt * 11) % 97).float() - 48) / 32
        for salt in (7, 13, 31)
    ]]).bfloat16()
    with ExitStack() as stack:
        store = Store(args.model, stack)
        weight = store.get("lm_head.weight")
        if weight.dtype != torch.uint8 or tuple(weight.shape) != (248320, 1024):
            raise ValueError("this fixture requires the pinned Ornith NVFP4 head")
        norm = store.get("model.language_model.norm.weight").float() + 1
        normalized = (
            inputs.float()
            * torch.rsqrt(inputs.float().square().mean(-1, keepdim=True) + 1e-6)
            * norm
        ).bfloat16()
        outputs = store.linear("lm_head", normalized)
    if not bool(torch.isfinite(outputs).all()) or bool(outputs[0].count_nonzero()):
        raise ValueError("invalid head reference outputs")
    rows = []
    for index, (source, normed, expected) in enumerate(zip(inputs, normalized, outputs, strict=True)):
        record = {"name": f"row-{index}", "zero": index == 0}
        for field, tensor in (("input", source), ("normalized", normed), ("expected", expected)):
            data = tensor.contiguous().view(torch.uint16).numpy().astype("<u2").tobytes()
            name = f"{field}-{index}.bf16"
            (args.output / name).write_bytes(data)
            record[field] = name
            record[field + "_sha256"] = hashlib.sha256(data).hexdigest()
        rows.append(record)
    manifest = {
        "scope": "synthetic BF16 residuals -> actual zero-centered final RMS -> NVFP4 head; no transformer",
        "model_revision": "94e431d9cc47fa1986a7a1a4e9a80f7f118b03aa",
        "config_sha256": sha(args.model / "config.json"),
        "index_sha256": sha(args.model / "model.safetensors.index.json"),
        "reference_source_sha256": sha(Path(__file__).with_name("qwen35_reference.py")),
        "environment": environment_receipt(),
        "devices": [1],
        "expected_device_uuids": ["GPU-dd2a988c-fe38-24ab-233a-9e9bb3b2645a"],
        "rows": rows,
        "absolute_tolerance": 1e-6,
        "relative_tolerance": 1 / 128,
        "tolerance_basis": "existing Qwen BF16 kernel probe spacing; fixed before device execution",
        "cuda_work": False,
    }
    (args.output / "head-input.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps({"rows": len(rows), "elements_per_output": 248320, "cuda_work": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
