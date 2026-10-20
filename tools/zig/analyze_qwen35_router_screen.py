"""Validate the router stage journal; report component timing without model claims."""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


def analyze(path):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert rows[0]["kind"] == "start" and rows[-1]["kind"] == "complete" and rows[-1]["correctness_pass"]
    assert not any(row["kind"] == "failure" for row in rows)
    assert [i["uuid"] for i in rows[0]["identities"]] == ["GPU-f2c0ea1e-a2be-f0b4-6c23-ffe33b537a06", "GPU-dd2a988c-fe38-24ab-233a-9e9bb3b2645a"]
    if any(row["kind"] == "router_candidate_sample" for row in rows):
        return analyze_candidate(rows)
    report = {"source_commit": rows[0]["source_commit"], "scope": "fixed-input L2-warm router arithmetic diagnostic; no model speed measurement", "stages": [], "optimistic_removal_bound_seconds_per_turn": {}}
    timing = {}
    stages = ["combined", "products", "select"]
    for ordinal in (0, 1):
        group = [row for row in rows if row.get("ordinal") == ordinal]
        controls = [r for r in group if r["kind"] == "router_negative_control"]
        assert len(controls) == 1 and controls[0]["poisoned_scores_rejected"]
        parity = [r for r in group if r["kind"] == "router_row_parity"]
        assert len(parity) == 6 and sorted(r["position"] for r in parity) == list(range(6))
        assert all(r["finite_scores"] == 257 and r["scores_repeat_bit_exact"] and r["saved_native_picks_and_weights_bit_exact"] for r in parity)
        reference = next(r["scores_sha256"] for r in parity if r["position"] == 0)
        done = [r for r in group if r["kind"] == "router_screen_complete"]
        assert len(done) == 1 and done[0]["correctness_pass"] and done[0]["saved_rows_exact"] == 6
        for index, stage in enumerate(stages):
            samples = sorted((r for r in group if r["kind"] == "router_sample" and r["stage"] == stage), key=lambda r: r["round"])
            assert len(samples) == 6 and [r["round"] for r in samples] == list(range(6))
            for row in samples:
                assert row["warmup"] == (row["round"] == 0) and row["order"] == (index if row["round"] % 2 == 0 else 2 - index)
                assert row["position"] == 0 and row["route_repetitions"] == 128 and row["graph_launches"] == 1
                assert row["node_count"] == row["kernel_launches"] == (256 if stage == "combined" else 128)
                assert row["finite_scores"] == 257 and row["scores_bit_exact"] and row["saved_native_picks_and_weights_bit_exact"]
                assert row["scores_sha256"] == reference and math.isfinite(row["elapsed_ms"]) and row["elapsed_ms"] > 0
            values = [r["elapsed_ms"] * 1000 / 128 for r in samples[1:]]
            timing[ordinal, stage] = statistics.median(values)
            report["stages"].append({"ordinal": ordinal, "stage": stage, "median_us_per_route": statistics.median(values), "range_us": [min(values), max(values)], "samples_us": values})
    for stage in stages:
        report["optimistic_removal_bound_seconds_per_turn"][stage] = 191 * 20 * sum(timing[o, stage] for o in (0, 1)) / 1e6
    report["bound_assumption"] = "Remove the entire isolated stage and transfer its cost unchanged to 20 layers per GPU and 191 appends. Not a causal full-model bound."
    report["correctness_pass"] = True
    return report


def analyze_candidate(rows):
    report = {"source_commit": rows[0]["source_commit"], "scope": "exact parallel selector seam; extrapolation is not model performance", "measurements": []}
    controls = {"all_zero", "cutoff_ties", "signed_zero", "all_negative", "pos_inf", "neg_inf", "nan_first", "nan_middle", "nan_last", "nan_shared_gate"}
    combined_savings = []
    for ordinal in (0, 1):
        group = [r for r in rows if r.get("ordinal") == ordinal]
        planted = [r for r in group if r["kind"] == "router_candidate_control"]
        assert len(planted) == 10 and {r["case_id"] for r in planted} == controls
        for row in planted:
            assert row["scores_bit_exact"] and row["old_new_picks_bit_exact"] and row["old_new_weights_bit_exact"] and row["control_pass"]
            assert row["finite_outputs"] is (True if row["finite_case"] else None)
        parity = [r for r in group if r["kind"] == "router_candidate_row_parity"]
        assert len(parity) == 12
        for position in range(6):
            pair = [r for r in parity if r["position"] == position]
            assert len(pair) == 2 and {r["arm"] for r in pair} == {"old", "new"}
            assert pair[0]["scores_sha256"] == pair[1]["scores_sha256"]
            assert all(r["finite_scores"] == 257 and r["scores_bit_exact"] and r["saved_native_picks_and_weights_bit_exact"] for r in pair)
        reference = next(r["scores_sha256"] for r in parity if r["position"] == 0)
        done = [r for r in group if r["kind"] == "router_candidate_complete"]
        assert len(done) == 1 and done[0]["correctness_pass"] and done[0]["saved_rows_exact"] == 6 and done[0]["planted_controls_exact"] == 10
        for stage in ("select", "combined"):
            samples = [r for r in group if r["kind"] == "router_candidate_sample" and r["stage"] == stage]
            assert len(samples) == 12
            values = {"old": [], "new": []}
            savings = []
            for pair in range(6):
                both = sorted((r for r in samples if r["pair"] == pair), key=lambda r: r["order"])
                assert len(both) == 2 and [r["arm"] for r in both] == (["old", "new"] if pair % 2 == 0 else ["new", "old"])
                for order, r in enumerate(both):
                    assert r["order"] == order and r["warmup"] == (pair == 0)
                    assert r["position"] == 0 and r["route_repetitions"] == 128 and r["graph_launches"] == 1
                    assert r["node_count"] == r["kernel_launches"] == (256 if stage == "combined" else 128)
                    assert r["finite_scores"] == 257 and r["scores_bit_exact"] and r["saved_native_picks_and_weights_bit_exact"]
                    assert r["scores_sha256"] == reference and math.isfinite(r["elapsed_ms"]) and r["elapsed_ms"] > 0
                if pair:
                    times = {r["arm"]: r["elapsed_ms"] * 1000 / 128 for r in both}
                    for arm, data in values.items():
                        data.append(times[arm])
                    savings.append(times["old"] - times["new"])
            median = statistics.median(savings)
            if stage == "combined":
                combined_savings.append(median)
            report["measurements"].append({"ordinal": ordinal, "stage": stage, "arms_us": {arm: {"median": statistics.median(data), "range": [min(data), max(data)], "samples": data} for arm, data in values.items()}, "paired_savings_us": savings, "median_saving_us": median, "all_pairs_faster": min(savings) > 0})
    projected = 191 * 20 * sum(combined_savings) / 1e6
    report.update(correctness_pass=True, estimated_seconds_saved_per_turn=projected, estimated_turn_fraction=projected / 2.2640918, material_5pct_screen=projected > 0.05 * 2.2640918, projection_assumption="191 appends, 20 layers on each GPU, transfer isolated saving unchanged; full-turn validation required")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("journal", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = analyze(args.journal)
    args.output.write_bytes((json.dumps(result, indent=2) + "\n").encode())
    print(json.dumps(result))


if __name__ == "__main__":
    main()
