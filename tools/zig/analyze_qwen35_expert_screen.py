"""Validate the expert seam journal and estimate whether a full trial is earned."""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from pathlib import Path


def analyze(path):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert rows[0]["kind"] == "start" and rows[-1]["kind"] == "complete"
    assert rows[-1]["correctness_pass"] is True
    assert not any(r["kind"] == "failure" for r in rows)
    assert [i["uuid"] for i in rows[0]["identities"]] == ["GPU-f2c0ea1e-a2be-f0b4-6c23-ffe33b537a06", "GPU-dd2a988c-fe38-24ab-233a-9e9bb3b2645a"]
    assert Counter(r["kind"] for r in rows)["expert_kernel"] == 16
    fixture = [r for r in rows if r["kind"] == "expert_fixture"]
    assert len(fixture) == 2 and fixture[0]["manifest_sha256"] == fixture[1]["manifest_sha256"]
    for r in fixture:
        assert sorted(r["expert_order"]) == list(range(256))
        assert r["bank0_original_ids"] == [112, 106, 107, 238, 181, 57, 43, 200]
    output = {"source_commit": rows[0]["source_commit"], "fixture_sha256": fixture[0]["manifest_sha256"], "scope": "layer-zero expert operator seam; extrapolation is not whole-model performance", "measurements": [], "projection": {}}
    saving = {}
    for ordinal in (0, 1):
        for epi in ("up", "down_f32"):
            group = [r for r in rows if r.get("ordinal") == ordinal and r.get("epi") == epi]
            plan = [r for r in group if r["kind"] == "expert_plan"]
            assert len(plan) == 1 and plan[0]["actual_items"] == plan[0]["bound"] == 8
            assert plan[0]["limit"] == 0 and plan[0]["relu2"] is False
            done = [r for r in group if r["kind"] == "expert_projection_complete"]
            assert len(done) == 1 and done[0]["all_bank_prechecks"] and done[0]["saved_bank0_exact"] and done[0]["graph_finals_exact"]
            for variant in ("pipe4", "n16", "n16_pipe4"):
                parity = [r for r in group if r["kind"] == "expert_bank_parity" and r["variant"] == variant]
                assert len(parity) == 32 and sorted(r["bank"] for r in parity) == list(range(32))
                assert all(r["finite"] and r["bit_exact"] and r["saved_real_output"] == (r["bank"] == 0) for r in parity)
                samples = [r for r in group if r["kind"] == "expert_sample" and r["comparison"] == variant]
                assert len(samples) == 12
                arms = {"baseline": [], variant: []}
                paired = []
                for pair in range(6):
                    both = sorted((r for r in samples if r["pair"] == pair), key=lambda r: r["order"])
                    assert len(both) == 2 and [r["order"] for r in both] == [0, 1]
                    assert [r["arm"] for r in both] == (["baseline", variant] if pair % 2 == 0 else [variant, "baseline"])
                    for r in both:
                        assert r["warmup"] == (pair == 0) and r["kernel_launches"] == 128 and r["banks"] == 32 and r["final_bank"] == 31
                        assert r["finite"] and r["bit_exact_final"] and math.isfinite(r["elapsed_ms"]) and r["elapsed_ms"] > 0
                    if pair:
                        times = {r["arm"]: r["elapsed_ms"] * 1000 / 128 for r in both}
                        for arm, values in arms.items():
                            values.append(times[arm])
                        paired.append(times["baseline"] - times[variant])
                item = {"ordinal": ordinal, "epi": epi, "candidate": variant, "arms_us": {arm: {"median": statistics.median(values), "range": [min(values), max(values)], "samples": values} for arm, values in arms.items()}, "paired_saving_us": paired, "median_saving_us": statistics.median(paired), "all_pairs_faster": min(paired) > 0}
                saving[ordinal, epi, variant] = item
                output["measurements"].append(item)
    for variant in ("pipe4", "n16", "n16_pipe4"):
        selected = [saving[ordinal, epi, variant] for ordinal in (0, 1) for epi in ("up", "down_f32")]
        us_per_append = 20 * sum(r["median_saving_us"] for r in selected)
        seconds = 191 * us_per_append / 1e6
        output["projection"][variant] = {"estimated_seconds_saved_per_64_128_turn": seconds, "estimated_turn_fraction": seconds / 2.2640918, "all_groups_all_pairs_faster": all(r["all_pairs_faster"] for r in selected), "material_5pct_screen": seconds > 0.05 * 2.2640918, "assumption": "20 layers per GPU, 191 appends, layer-zero micro saving transfers unchanged to every layer; full-turn validation required"}
    output["correctness_pass"] = True
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("journal", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = analyze(args.journal)
    args.output.write_bytes((json.dumps(result, indent=2) + "\n").encode())
    print(json.dumps(result["projection"]))


if __name__ == "__main__":
    main()
