"""Validate complete Qwen tuning receipts before summarizing their timings."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import median


def require(condition, message):
    if not condition:
        raise ValueError(message)


def records(path):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    require(rows and rows[-1].get("kind") == "complete", f"incomplete experiment: {path}")
    require(not any(row.get("kind") == "failure" for row in rows), f"failed experiment: {path}")
    return rows


def spread(values):
    return {"median": median(values), "min": min(values), "max": max(values), "count": len(values)}


def sampler_summary(rows, source):
    kinds = Counter(row["kind"] for row in rows)
    require(kinds == Counter(start=1, parity_outputs=2, parity=2, turn=24,
                             pair_verified=12, case_summary=2, complete=1), "sampler record counts differ")
    start = rows[0]
    require(start["kind"] == "start" and start["source_commit"] == source, "sampler source differs")
    require(rows[-1]["correctness_pass"] is True, "sampler correctness failed")
    result = {}
    for name, prompt, outputs in (("decode-heavy", 64, 128), ("prefill-heavy", 275, 27)):
        case = [row for row in rows if row.get("case_name") == name]
        parity = next(row for row in case if row["kind"] == "parity")
        require(parity["row_count"] == outputs and parity["consumed_raw_bf16_rows_equal"] is True
                and parity["all_output_ids_equal"] is True, "raw-row parity failed")
        baseline = next(row for row in case if row["kind"] == "parity_outputs")
        reference = baseline["old_tokens"]
        require(len(reference) == outputs and reference == baseline["current_tokens"], "untimed output mismatch")
        turns = [row for row in case if row["kind"] == "turn"]
        require(len(turns) == 12, "case turn count differs")
        require(Counter((r["pair"], r["arm"]) for r in turns) ==
                Counter((pair, arm) for pair in range(6) for arm in ("old", "current")), "missing/duplicate arm")
        for row in turns:
            expected_order = ("old", "current") if row["pair"] % 2 == 0 else ("current", "old")
            require(row["arm"] == expected_order[row["order_index"]], "pair order differs")
            require(row["warmup"] == (row["pair"] == 0), "warmup label differs")
            require(row["tokens"] == reference, "timed output differs from reference")
            t = row["times"]
            require(t["draws"] == outputs and t["appends"] == prompt + outputs - 1
                    and t["decode_tokens"] == outputs - 1, "turn work count differs")
            require(t["total_ns"] == t["ttft_ns"] + t["decode_ns"]
                    and t["ttft_ns"] >= t["prefill_ns"] > 0 and t["decode_ns"] > 0
                    and t["reset_ns"] >= 0, "invalid timing boundaries")
        verified = [row for row in case if row["kind"] == "pair_verified"]
        require(len(verified) == 6 and {r["pair"] for r in verified} == set(range(6))
                and all(r["complete_output_identity"] and r["repeat_determinism"] for r in verified),
                "pair verification differs")
        arms = {}
        paired = []
        for arm in ("old", "current"):
            measured = [r for r in turns if r["arm"] == arm and not r["warmup"]]
            arms[arm] = {field.removesuffix("_ns") + "_seconds": spread([r["times"][field] / 1e9 for r in measured])
                         for field in ("total_ns", "prefill_ns", "ttft_ns", "decode_ns", "reset_ns")}
            arms[arm]["decode_tokens_per_second"] = spread([(outputs - 1) * 1e9 / r["times"]["decode_ns"] for r in measured])
            arms[arm]["first_draw_ms"] = spread([(r["times"]["ttft_ns"] - r["times"]["prefill_ns"]) / 1e6 for r in measured])
        for pair in range(1, 6):
            match = {r["arm"]: r["times"]["total_ns"] for r in turns if r["pair"] == pair}
            paired.append(1 - match["current"] / match["old"])
        reduction = 1 - arms["current"]["total_seconds"]["median"] / arms["old"]["total_seconds"]["median"]
        summary = next(row for row in case if row["kind"] == "case_summary")
        require(abs(summary["median_total_reduction"] - reduction) < 1e-12, "driver median differs")
        material = reduction >= 0.05 and sum(g > 0 for g in paired) >= 4
        require(summary["material"] == material, "driver materiality differs")
        result[name] = {"prompt_tokens": prompt, "output_tokens": outputs, "arms": arms,
                        "total_latency_reduction": reduction, "paired_reductions": paired,
                        "positive_pairs": sum(g > 0 for g in paired), "material": material,
                        "raw_rows_equal": outputs, "complete_outputs_identical": True}
    require(rows[-1]["primary_materiality_pass"] == result["decode-heavy"]["material"], "primary verdict differs")
    return result


def head_summary(rows, source):
    require(rows[0]["source_commit"] == source and rows[-1]["status"] == "pass", "head identity or completion differs")
    samples = [r for r in rows if r["kind"] == "sample"]
    require(len(samples) == 48, "head sample count differs")
    references = [r for r in rows if r["kind"] == "reference"]
    require(len(references) == 4 and all(r["norm"]["mismatches"] == 0 and r["head"]["mismatches"] == 0
            and r["head"]["top_actual"] == r["head"]["top_expected"] for r in references), "head CPU reference failed")
    deterministic = [r for r in rows if r["kind"] == "determinism"]
    require(len(deterministic) == 4 and all(r["repeat_exact"] for r in deterministic), "head repeat failed")
    results = {}
    for arm in ("full_boundary", "projection_only"):
        measured = [r for r in samples if r["arm"] == arm and not r["warmup"]]
        require(len(measured) == 20 and all(r["calls"] == 32 and r["wall_ns"] > 0 for r in measured), "head work differs")
        require(Counter((r["row"], r["pair"]) for r in measured) ==
                Counter((f"row-{row}", pair) for row in range(4) for pair in range(1, 6)), "head row/pair differs")
        results[arm] = {"wall_ms_per_call": spread([r["wall_ns"] / r["calls"] / 1e6 for r in measured])}
        if arm == "projection_only":
            results[arm]["cuda_ms_per_call"] = spread([r["cuda_ms"] / r["calls"] for r in measured])
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    window = json.loads((args.window / "window.json").read_text())
    require(window["passed"] is True and window["restored"] is True and window["clef_unchanged"] is True,
            "window did not pass and restore")
    source = window["plan"]["source_commit"]
    sampler = sampler_summary(records(args.window / "sampler-results/results.jsonl"), source)
    head = head_summary(records(args.window / "head-results/results.jsonl"), source)
    samples = window["memory"]
    require(samples and not window["sampler_failures"] and not window["cleanup_errors"], "window monitor failed")
    uuids = tuple(samples[0]["free_mib"])
    minimum = {uuid: min(r["free_mib"][uuid] for r in samples) for uuid in uuids}
    require(all(value >= 1536 for value in minimum.values()), "memory floor breached")
    prefill = sampler["prefill-heavy"]
    saving = (prefill["prompt_tokens"] - 1) * head["full_boundary"]["wall_ms_per_call"]["median"] / 1000
    scenario = saving / prefill["arms"]["current"]["total_seconds"]["median"]
    summary = {"source_commit": source, "sampler": sampler, "head": head,
               "physical_gpu_free_min_mib": minimum, "physical_samples": len(samples),
               "restoration": window["restoration"], "head_elision_scenario_fraction": scenario,
               "head_elision_scenario_is_integrated_measurement": False,
               "scope": "synthetic direct single session; no HTTP, concurrency, cache or quality qualification"}
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"cases": {name: {"reduction": row["total_latency_reduction"], "positive_pairs": row["positive_pairs"]}
                                for name, row in sampler.items()}, "head_scenario_fraction": scenario}))


if __name__ == "__main__":
    main()
