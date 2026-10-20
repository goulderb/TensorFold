"""CPU greedy-trajectory discriminator for the same ModelOpt checkpoint.

Reuses the unchanged qwen35_reference lazy ModelOpt arithmetic (make_model /
Store / source_module / unmodified HF forward). Prefills one token at a time,
then autoregresses greedy tokens with use_cache, BF16 activations, and no EOS
stop. Scope is a same-checkpoint behavioral discriminator, not an independent
GPU oracle or production promotion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from contextlib import ExitStack
from pathlib import Path

# Windows treats an empty CUDA_VISIBLE_DEVICES as "all GPUs"; hide them with -1.
# qwen35_reference also assigns "" on import, so restore -1 immediately after.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["USE_HUB_KERNELS"] = "0"

import torch

import qwen35_reference as ref

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

VOCAB = 248320
SCOPE = (
    "CPU arithmetic behavioral discriminator; not an independent GPU oracle "
    "or production promotion"
)


def token_sha256(tokens: list[int]) -> str:
    """Match zig/src/server/tokens.zig tokenSha256: decimal ids joined by commas."""
    payload = ",".join(str(int(t)) for t in tokens).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def require_cpu() -> None:
    if torch.cuda.is_initialized():
        raise ValueError("greedy reference requires CPU; CUDA is already initialized")


def load_prompt(path: Path) -> list[int]:
    spec = json.loads(path.read_text(encoding="utf-8"))
    prompt = spec["prompt"]
    if (
        not isinstance(prompt, list)
        or not prompt
        or any(type(t) is not int or not 0 <= t < VOCAB for t in prompt)
    ):
        raise ValueError(f"{path}: prompt must be a nonempty list of vocab ids")
    return prompt


def top2(logits: torch.Tensor) -> tuple[list[int], list[float], float]:
    """Lowest-id wins ties, matching NumPy/native argmax."""
    scores = logits.float()
    first = int(torch.argmax(scores))
    masked = scores.clone()
    masked[first] = torch.finfo(scores.dtype).min
    second = int(torch.argmax(masked))
    top_scores = [float(scores[first]), float(scores[second])]
    return [first, second], top_scores, top_scores[0] - top_scores[1]


def ensure_empty_output(path: Path) -> None:
    if path.exists():
        if not path.is_dir():
            raise ValueError(f"output path exists and is not a directory: {path}")
        if any(path.iterdir()):
            raise ValueError("greedy reference output directory must be empty")
    else:
        path.mkdir(parents=True, exist_ok=True)


def write_case_receipt(path: Path, receipt: dict) -> None:
    ref.write_json(path, receipt)


def run_case(
    model,
    prompt: list[int],
    max_tokens: int,
    input_path: Path,
    case_dir: Path,
    meta: dict,
) -> dict:
    case_dir.mkdir(parents=True, exist_ok=True)
    state = None
    tokens: list[int] = []
    steps: list[dict] = []
    receipt = {
        **meta,
        "input": str(input_path).replace("\\", "/"),
        "input_sha256": ref.sha(input_path),
        "prompt": prompt,
        "tokens": tokens,
        "steps": steps,
        "token_sha256": token_sha256([]),
        "max_tokens": max_tokens,
        "ignore_eos": True,
        "sampling": "greedy",
        "use_cache": True,
    }
    write_case_receipt(case_dir / "result.json", receipt)

    with torch.inference_mode():
        for position, token in enumerate(prompt):
            result = model(
                input_ids=torch.tensor([[token]], dtype=torch.long),
                past_key_values=state,
                use_cache=True,
                logits_to_keep=1,
            )
            state = result.past_key_values
            if position < len(prompt) - 1:
                continue
            logits = result.logits[0, -1]
            if logits.dtype != torch.bfloat16:
                raise ValueError("reference logits must remain BF16 until top-k")
            if not torch.isfinite(logits.float()).all():
                raise ValueError("nonfinite reference logits during prefill")

        for index in range(max_tokens):
            ids, scores, margin = top2(logits)
            chosen = ids[0]
            tokens.append(chosen)
            step = {
                "index": index,
                "id": chosen,
                "top2_ids": ids,
                "top2_logits": scores,
                "margin": margin,
                "token_sha256": token_sha256(tokens),
            }
            steps.append(step)
            receipt["tokens"] = list(tokens)
            receipt["steps"] = list(steps)
            receipt["token_sha256"] = step["token_sha256"]
            write_case_receipt(case_dir / "result.json", receipt)
            print(
                f"{case_dir.name} token {index}: id {chosen} "
                f"top2 {ids} margin {margin:.6g} sha {step['token_sha256'][:12]}",
                flush=True,
            )
            if index == max_tokens - 1:
                break
            result = model(
                input_ids=torch.tensor([[chosen]], dtype=torch.long),
                past_key_values=state,
                use_cache=True,
                logits_to_keep=1,
            )
            state = result.past_key_values
            logits = result.logits[0, -1]
            if logits.dtype != torch.bfloat16:
                raise ValueError("reference logits must remain BF16 until top-k")
            if not torch.isfinite(logits.float()).all():
                raise ValueError("nonfinite reference logits during decode")

    return receipt


def validate_args(args) -> list[tuple[Path, list[int]]]:
    require_cpu()
    if not 1 <= args.threads <= 16:
        raise ValueError("threads must be between 1 and 16")
    if not 1 <= args.max_tokens <= 32:
        raise ValueError("max_tokens must be between 1 and 32")
    if args.device != "cpu":
        raise ValueError("greedy reference accepts only device=cpu")
    inputs = list(args.input)
    if not inputs:
        raise ValueError("at least one --input is required")
    prompts = [(path, load_prompt(path)) for path in inputs]
    stems = [path.stem for path in inputs]
    if len(set(stems)) != len(stems):
        raise ValueError("input filenames must have unique stems for output cases")
    ensure_empty_output(args.output)
    return prompts


def capture(args) -> int:
    import transformers

    prompts = validate_args(args)
    if transformers.__version__ != "5.16.1":
        raise ValueError("use the verified Transformers 5.16.1 integration environment")

    torch.set_num_threads(args.threads)
    reference = ref.source_module(args.source)
    require_cpu()

    meta = {
        "vocab": VOCAB,
        "config_sha256": ref.sha(args.model / "config.json"),
        "index_sha256": ref.sha(args.model / "model.safetensors.index.json"),
        "reference_revision": ref.REVISION,
        "reference_source_sha256": ref.SOURCE_SHA,
        "reference_script_sha256": ref.sha(Path(__file__)),
        "original_reference_script_sha256": ref.sha(Path(ref.__file__)),
        "arithmetic_adapter": ref.ARITHMETIC_ADAPTER,
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "environment": ref.environment_receipt(),
        "device": "cpu",
        "dtype": "bfloat16",
        "activation_quantization": "static FP8 only; NVFP4 W4A16",
        "implementation": (
            "unmodified pinned HF model forwards; lazy ModelOpt F.linear operands; "
            "greedy autoregression"
        ),
        "scope": SCOPE,
        "token_sha256_encoding": "decimal-ids-joined-by-commas",
        "token_sha256_source": "zig/src/server/tokens.zig:tokenSha256",
    }

    cases = []
    with ExitStack() as stack, torch.inference_mode():
        store = ref.Store(args.model, stack)
        model = ref.make_model(args.model, store, reference)
        require_cpu()
        for path, prompt in prompts:
            case_dir = args.output / path.stem
            receipt = run_case(
                model, prompt, args.max_tokens, path, case_dir, meta
            )
            cases.append(
                {
                    "name": path.stem,
                    "input": str(path).replace("\\", "/"),
                    "input_sha256": receipt["input_sha256"],
                    "tokens": receipt["tokens"],
                    "token_sha256": receipt["token_sha256"],
                    "result": str(case_dir / "result.json").replace("\\", "/"),
                }
            )

    summary = {
        **meta,
        "model": str(args.model).replace("\\", "/"),
        "source": str(args.source).replace("\\", "/"),
        "threads": args.threads,
        "max_tokens": args.max_tokens,
        "ignore_eos": True,
        "cases": cases,
    }
    ref.write_json(args.output / "result.json", summary)
    print(json.dumps({"cases": len(cases), "output": str(args.output)}, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--input",
        type=Path,
        action="append",
        required=True,
        help="oracle-preflight prompt JSON (repeatable; one model load)",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(".artifacts/modeling_qwen3_5_moe-e0eb2810.py"),
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument(
        "--device",
        default="cpu",
        help="must be cpu; non-cpu values are refused",
    )
    args = parser.parse_args()
    return capture(args)


if __name__ == "__main__":
    raise SystemExit(main())
