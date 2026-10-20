"""Capture CPU HF logits from the ModelOpt checkpoint and compare native probe rows."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import os
import sys
import urllib.request
from contextlib import ExitStack
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["USE_HUB_KERNELS"] = "0"

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

REVISION = "e0eb2810ac1b2caf6d082f247d8d37162ecf67a0"
SOURCE_SHA = "d9f88f2d048492c6a787119c7466add91cb311d7699aea5701ffb8a3ba3015a5"
SOURCE_URL = f"https://raw.githubusercontent.com/huggingface/transformers/{REVISION}/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py"
ARITHMETIC_ADAPTER = "modelopt-linear-fp32-postscale-bf16-v1"
FP4 = (0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6)


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def environment_receipt() -> dict:
    packages = {}
    for name in (
        "transformers",
        "torch",
        "safetensors",
        "flash-linear-attention",
        "fla-core",
        "kernels",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "executable": sys.executable,
        "prefix": sys.prefix,
        "base_prefix": sys.base_prefix,
        "python": sys.version,
        "packages": packages,
    }


def source_module(path: Path):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(SOURCE_URL, timeout=60) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != SOURCE_SHA:
            raise ValueError("reference source hash mismatch")
        path.write_bytes(data)
    if sha(path) != SOURCE_SHA:
        raise ValueError("reference source hash mismatch")
    name = "transformers.models.qwen3_5_moe.pinned_qwen35_reference"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    for name in (
        "causal_conv1d_fn",
        "causal_conv1d_update",
        "torch_chunk_gated_delta_rule",
        "torch_recurrent_gated_delta_rule",
    ):
        setattr(module, name, inspect.unwrap(getattr(module, name)))
    return module


def unpack_fp4(
    packed: torch.Tensor, blocks: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    if packed.dtype != torch.uint8 or blocks.dtype != torch.float8_e4m3fn:
        raise ValueError("unsupported NVFP4 storage")
    n, half_k = packed.shape
    if half_k % 8 or blocks.shape != (n, half_k // 8):
        raise ValueError("unexpected NVFP4 shape")
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).reshape(n, half_k * 2)
    values = torch.tensor(FP4, dtype=torch.float32)[codes.long()]
    return (
        (values.reshape(n, -1, 16) * blocks.float().unsqueeze(-1) * scale.float())
        .reshape(n, -1)
        .bfloat16()
    )


class Store:
    def __init__(self, root: Path, stack: ExitStack):
        index = json.loads(
            (root / "model.safetensors.index.json").read_text(encoding="utf-8")
        )
        self.names = index["weight_map"]
        self.shards = {
            name: stack.enter_context(
                safe_open(root / name, framework="pt", device="cpu")
            )
            for name in sorted(set(self.names.values()))
        }

    def get(self, name: str) -> torch.Tensor:
        return self.shards[self.names[name]].get_tensor(name)

    def scalar(self, name: str) -> torch.Tensor:
        value = self.get(name).float()
        if (
            value.numel() != 1
            or not bool(torch.isfinite(value).all())
            or not bool((value > 0).all())
        ):
            raise ValueError(f"invalid scale {name}")
        return value.reshape(())

    def linear(self, prefix: str, x: torch.Tensor) -> torch.Tensor:
        if x.device.type != "cpu" or x.dtype != torch.bfloat16:
            raise ValueError("reference requires CPU BF16 activations")
        w = self.get(prefix + ".weight")
        if w.dtype == torch.bfloat16:
            return torch.cat(
                [
                    F.linear(x, w[start : start + 4096])
                    for start in range(0, w.shape[0], 4096)
                ],
                dim=-1,
            )
        activation = x.float()
        if w.dtype == torch.float8_e4m3fn:
            scale = self.scalar(prefix + ".input_scale")
            activation = (
                (activation * (1 / scale))
                .clamp(-448, 448)
                .to(torch.float8_e4m3fn)
                .float()
            )
            output_scale = scale * self.scalar(prefix + ".weight_scale")
        elif w.dtype == torch.uint8:
            output_scale = self.scalar(prefix + ".weight_scale_2")
        else:
            raise ValueError(f"unsupported projection {prefix}: {w.dtype}")
        pieces = []
        for start in range(0, w.shape[0], 4096):
            weight = w[start : start + 4096]
            if w.dtype == torch.uint8:
                weight = unpack_fp4(
                    weight,
                    self.get(prefix + ".weight_scale")[start : start + 4096],
                    torch.tensor(1.0),
                )
            pieces.append(
                (F.linear(activation, weight.float()) * output_scale).bfloat16()
            )
        return torch.cat(pieces, dim=-1)


class LazyLinear(torch.nn.Module):
    def __init__(self, store: Store, prefix: str):
        super().__init__()
        self.store, self.prefix = store, prefix

    def forward(self, x):
        return self.store.linear(self.prefix, x)


class ExpertProjection:
    def __init__(self, store: Store, prefixes: tuple[str, ...]):
        self.store, self.prefixes = store, prefixes

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if func is not F.linear:
            return NotImplemented
        return cls._linear(*args, **(kwargs or {}))

    @staticmethod
    def _linear(input, weight, bias=None):
        if bias is not None:
            raise ValueError("expert projections require bias=None")
        return torch.cat(
            [weight.store.linear(prefix, input) for prefix in weight.prefixes], dim=-1
        )


class ExpertWeights:
    def __init__(self, store: Store, prefix: str, projections: tuple[str, ...]):
        self.store, self.prefix, self.projections = store, prefix, projections

    def __getitem__(self, expert):
        expert = int(expert)
        if not 0 <= expert < 256:
            raise ValueError("invalid expert id")
        return ExpertProjection(
            self.store,
            tuple(
                f"{self.prefix}.{expert}.{projection}"
                for projection in self.projections
            ),
        )


def make_model(root: Path, store: Store, reference):
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
        Qwen3_5MoeTextConfig,
    )

    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    quant = config.get("quantization_config", {})
    if quant.get("quant_algo") != "MIXED_PRECISION" or any(
        v.get("quant_algo") not in ("FP8", "W4A16_NVFP4")
        for v in quant.get("quantized_layers", {}).values()
    ):
        raise ValueError("reference requires ModelOpt FP8 and W4A16_NVFP4")
    cfg = Qwen3_5MoeTextConfig.from_dict(config["text_config"])
    cfg._attn_implementation = "eager"
    cfg._experts_implementation = "eager"
    dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        with torch.device("meta"):
            model = reference.Qwen3_5MoeForCausalLM(cfg)
    finally:
        torch.set_default_dtype(dtype)
    modules = list(model.named_modules())
    for name, module in modules:
        if isinstance(module, torch.nn.Linear):
            prefix = (
                name.replace("model.", "model.language_model.", 1)
                if name.startswith("model.")
                else name
            )
            model.set_submodule(name, LazyLinear(store, prefix))
        elif isinstance(module, reference.Qwen3_5MoeExperts):
            prefix = name.replace("model.", "model.language_model.", 1)
            del module.gate_up_proj
            del module.down_proj
            module.gate_up_proj = ExpertWeights(store, prefix, ("gate_proj", "up_proj"))
            module.down_proj = ExpertWeights(store, prefix, ("down_proj",))
    for name, parameter in list(model.named_parameters()):
        prefix = (
            name.replace("model.", "model.language_model.", 1)
            if name.startswith("model.")
            else name
        )
        tensor = store.get(prefix)
        if tensor.shape != parameter.shape or tensor.dtype != torch.bfloat16:
            raise ValueError(f"unexpected unquantized parameter {prefix}")
        parent, _, key = name.rpartition(".")
        setattr(
            model.get_submodule(parent),
            key,
            torch.nn.Parameter(tensor, requires_grad=False),
        )
    model.model.rotary_emb = reference.Qwen3_5MoeTextRotaryEmbedding(cfg)
    if any(p.is_meta for p in model.parameters()) or any(
        b.is_meta for b in model.buffers()
    ):
        raise ValueError("uninitialized reference state")
    return model.eval()


def capture(args) -> None:
    import transformers

    if transformers.__version__ != "5.16.1":
        raise ValueError("use the verified Transformers 5.16.1 integration environment")
    torch.set_num_threads(args.threads)
    reference = source_module(args.source)
    spec = json.loads(args.input.read_text(encoding="utf-8"))
    prompt, forced, count = spec["prompt"], spec.get("forced", []), spec["max_tokens"]
    if (
        not prompt
        or count < 1
        or len(forced) != count
        or any(type(t) is not int or not 0 <= t < 248320 for t in prompt + forced)
    ):
        raise ValueError(
            "reference capture requires a nonempty prompt and max_tokens explicit forced ids"
        )
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise ValueError("reference output directory must be empty")
    with ExitStack() as stack, torch.inference_mode():
        store = Store(args.model, stack)
        model = make_model(args.model, store, reference)
        state = None
        for position, token in enumerate(prompt + forced):
            if position == len(prompt) + count - 1:
                break
            result = model(
                input_ids=torch.tensor([[token]], dtype=torch.long),
                past_key_values=state,
                use_cache=True,
                logits_to_keep=1,
            )
            state = result.past_key_values
            if position < len(prompt) - 1:
                continue
            row = result.logits[0, -1].float().numpy()
            if not np.isfinite(row).all():
                raise ValueError("nonfinite reference logits")
            row_index = position - len(prompt) + 1
            row.astype("<f4").tofile(args.output / f"row-{row_index:04d}.f32")
            print(f"reference row {row_index}: argmax {int(row.argmax())}", flush=True)
    write_json(
        args.output / "result.json",
        {
            "prompt": prompt,
            "forced": forced,
            "tokens": forced,
            "vocab": 248320,
            "input_sha256": sha(args.input),
            "config_sha256": sha(args.model / "config.json"),
            "index_sha256": sha(args.model / "model.safetensors.index.json"),
            "reference_revision": REVISION,
            "reference_source_sha256": SOURCE_SHA,
            "reference_script_sha256": sha(Path(__file__)),
            "arithmetic_adapter": ARITHMETIC_ADAPTER,
            "transformers": transformers.__version__,
            "torch": torch.__version__,
            "environment": environment_receipt(),
            "device": "cpu",
            "dtype": "bfloat16",
            "activation_quantization": "static FP8 only; NVFP4 W4A16",
            "implementation": "unmodified pinned HF model forwards; lazy ModelOpt F.linear operands",
            "scope": "HF eager arithmetic reference; not an independently qualified ModelOpt serving runtime",
        },
    )


def row_metrics(actual: np.ndarray, expected: np.ndarray) -> dict:
    if (
        actual.shape != expected.shape
        or actual.ndim != 1
        or not np.isfinite(actual).all()
        or not np.isfinite(expected).all()
    ):
        raise ValueError("incompatible or nonfinite logits")
    x, y = actual.astype(np.float64), expected.astype(np.float64)
    delta = x - y
    x -= x.max()
    y -= y.max()
    log_x = x - np.log(np.exp(x).sum())
    log_y = y - np.log(np.exp(y).sum())
    expected_top = np.argsort(expected)[-10:]
    actual_top = np.argsort(actual)[-10:]
    return {
        "max_abs": float(np.abs(delta).max()),
        "rmse": float(np.sqrt(np.mean(delta**2))),
        "kl_reference_to_actual": float(np.sum(np.exp(log_y) * (log_y - log_x))),
        "argmax_equal": bool(actual.argmax() == expected.argmax()),
        "actual_argmax": int(actual.argmax()),
        "reference_argmax": int(expected.argmax()),
        "top10_overlap": len(set(expected_top.tolist()) & set(actual_top.tolist())),
    }


def compare(args) -> int:
    actual = json.loads((args.actual / "result.json").read_text(encoding="utf-8"))
    expected = json.loads((args.reference / "result.json").read_text(encoding="utf-8"))
    if any(
        actual[key] != expected[key]
        for key in ("prompt", "tokens", "vocab", "config_sha256", "index_sha256")
    ):
        raise ValueError(
            "reference and candidate must use identical prompt and forced token streams"
        )
    count = len(expected["tokens"])
    if count == 0:
        raise ValueError("comparison requires at least one logit row")
    if (
        len(list(args.actual.glob("row-*.f32"))) != count
        or len(list(args.reference.glob("row-*.f32"))) != count
    ):
        raise ValueError("missing or extra logit rows")
    rows = []
    for index in range(count):
        name = f"row-{index:04d}.f32"
        x, y = (
            np.fromfile(directory / name, dtype="<f4")
            for directory in (args.actual, args.reference)
        )
        if len(x) != expected["vocab"] or len(y) != expected["vocab"]:
            raise ValueError("wrong logit row byte count")
        metrics = row_metrics(x, y)
        metrics["pass"] = (
            metrics["rmse"] <= args.max_rmse
            and metrics["max_abs"] <= args.max_abs
            and metrics["kl_reference_to_actual"] <= args.max_kl
            and (metrics["argmax_equal"] or args.allow_argmax_difference)
        )
        rows.append({"row": index, **metrics})
    report = {
        "pass": all(row["pass"] for row in rows),
        "rows": rows,
        "limits": {
            "rmse": args.max_rmse,
            "max_abs": args.max_abs,
            "kl": args.max_kl,
            "argmax_equal": not args.allow_argmax_difference,
        },
        "reference_receipt_sha256": sha(args.reference / "result.json"),
        "actual_receipt_sha256": sha(args.actual / "result.json"),
    }
    write_json(args.output, report)
    print(json.dumps(report, indent=2))
    return 0 if report["pass"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    cap = commands.add_parser("capture")
    cap.add_argument("model", type=Path)
    cap.add_argument("input", type=Path)
    cap.add_argument("output", type=Path)
    cap.add_argument(
        "--source",
        type=Path,
        default=Path(".artifacts/modeling_qwen3_5_moe-e0eb2810.py"),
    )
    cap.add_argument("--threads", type=int, default=4)
    cmp = commands.add_parser("compare")
    cmp.add_argument("actual", type=Path)
    cmp.add_argument("reference", type=Path)
    cmp.add_argument("output", type=Path)
    cmp.add_argument("--max-rmse", type=float, required=True)
    cmp.add_argument("--max-abs", type=float, required=True)
    cmp.add_argument("--max-kl", type=float, required=True)
    cmp.add_argument("--allow-argmax-difference", action="store_true")
    args = parser.parse_args()
    if args.command == "compare":
        for limit in (args.max_rmse, args.max_abs, args.max_kl):
            if not np.isfinite(limit) or limit < 0:
                raise ValueError("comparison limits must be finite and nonnegative")
        return compare(args)
    if not 1 <= args.threads <= 16:
        raise ValueError("threads must be between 1 and 16")
    capture(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
