"""Validate rotating-bank GDN journals and bound a candidate's full-turn payoff."""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path

UUIDS = ["GPU-f2c0ea1e-a2be-f0b4-6c23-ffe33b537a06", "GPU-dd2a988c-fe38-24ab-233a-9e9bb3b2645a"]
FLOOR = 1536 * 1024 * 1024
STATE_BYTES = 2097152
OUT_BYTES = 16384
BANK_BYTES = STATE_BYTES + OUT_BYTES
BASELINE_SECONDS = 1.7110656
SCOPE = "synthetic nonzero GDN operator seam; timed 64-bank working set exceeds L2; not physical DRAM bandwidth, a saved model trace, or full-model performance"
WORK = {"banks": 64, "unique_state_bytes": 134217728, "bank_bytes": BANK_BYTES, "payload_bytes": 64 * BANK_BYTES, "updates_per_bank": 2}


def digest(value):
    assert isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value)
    return value


def analyze(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
    assert rows and rows[0]["kind"] == "start" and rows[-1]["kind"] == "complete"
    assert rows[-1]["correctness_pass"] is True and rows[-1]["production_qualification"] is False
    assert rows[0]["scope"] == rows[-1]["scope"] == SCOPE
    assert re.fullmatch(r"[0-9a-f]{40}", rows[0]["source_commit"])
    assert [(r["ordinal"], r["uuid"]) for r in rows[0]["identities"]] == list(enumerate(UUIDS))
    allowed = {"start", "complete", "gdn_fixture", "gdn_negative_control", "gdn_unwritten_state_control", "gdn_step_parity", "gdn_graph", "gdn_sample", "gdn_screen_complete"}
    assert all(r["kind"] in allowed for r in rows)
    assert sum(r["kind"] == "start" for r in rows) == sum(r["kind"] == "complete" for r in rows) == 1
    assert all(r.get("ordinal") in (0, 1) for r in rows[1:-1])
    measurements = []
    for ordinal in (0, 1):
        group = [r for r in rows if r.get("ordinal") == ordinal]
        by_kind = {kind: [r for r in group if r["kind"] == kind] for kind in allowed}
        for row in group:
            for key in ("free_bytes", "free_bytes_before"):
                if key in row:
                    assert row[key] >= FLOOR
            if "floor_bytes" in row:
                assert row["floor_bytes"] == FLOOR
            if "scope" in row:
                assert row["scope"] == SCOPE
        assert len(by_kind["gdn_fixture"]) == 1
        fixture = by_kind["gdn_fixture"][0]
        assert fixture["scope"] == SCOPE
        expected_fixture = {"seed": 0x35D019, "pattern_version": 2, "state_layout": "head,value,key", "value_heads": 32, "key_heads": 16, "dimension": 128, "state_bytes": STATE_BYTES, "out_bytes": OUT_BYTES, "q_dtype": "FP32", "k_dtype": "FP32", "v_dtype": "BF16", "state_dtype": "FP32", "out_dtype": "FP32", "q_elements": 2048, "k_elements": 2048, "v_elements": 4096, "decay_elements": 32, "beta_elements": 32}
        assert all(fixture[k] == value for k, value in expected_fixture.items())
        initial = digest(fixture["initial_state_sha256"])
        assert fixture["free_bytes"] >= FLOOR and fixture["floor_bytes"] == FLOOR
        assert len(by_kind["gdn_negative_control"]) == len(by_kind["gdn_unwritten_state_control"]) == 1
        control = by_kind["gdn_negative_control"][0]
        assert all(control[k] is True for k in ("poisoned_output_rejected", "changed_state_rejected", "changed_output_rejected", "late_bank_state_mismatch_rejected", "late_bank_output_mismatch_rejected"))
        assert by_kind["gdn_unwritten_state_control"][0]["unchanged_initial_state_with_correct_output_rejected"] is True
        parity = sorted(by_kind["gdn_step_parity"], key=lambda r: r["step"])
        assert len(parity) == 16 and [r["step"] for r in parity] == list(range(16))
        assert len({digest(r["inputs_sha256"]) for r in parity}) == 16
        for row in parity:
            assert all(row[k] is True for k in ("state_bit_exact", "output_bit_exact", "finite_state", "finite_output", "unwritten_initial_state_rejected"))
            assert digest(row["state_sha256"]) != initial
            digest(row["output_sha256"])
            assert row["free_bytes"] >= FLOOR
        graphs = by_kind["gdn_graph"]
        assert len(graphs) == 2 and {r["arm"] for r in graphs} == {"old", "new"}
        reference = graphs[0]
        graph_hashes = ("inputs_sha256", "expected_state_sha256", "expected_output_sha256", "expected_payload_sha256", "initial_payload_sha256")
        for graph in graphs:
            assert graph["scope"] == SCOPE
            assert graph["node_count"] == graph["kernel_launches"] == 128
            assert graph["grid_x"] == 32 and graph["block_x"] == 128 and graph["evolving_state"] is True
            assert all(graph[k] == value for k, value in WORK.items())
            assert all(digest(graph[k]) == digest(reference[k]) for k in graph_hashes)
            assert graph["inputs_sha256"] == parity[0]["inputs_sha256"]
            assert graph["free_bytes"] >= FLOOR
        samples = by_kind["gdn_sample"]
        assert len(samples) == 12
        times = {"old": [], "new": []}
        savings = []
        initial_states = set()
        for pair in range(6):
            both = sorted((r for r in samples if r["pair"] == pair), key=lambda r: r["order"])
            assert len(both) == 2 and [r["arm"] for r in both] == (["old", "new"] if pair % 2 == 0 else ["new", "old"])
            pair_times = {}
            for order, row in enumerate(both):
                assert row["scope"] == SCOPE
                assert row["order"] == order and row["warmup"] is (pair == 0)
                assert row["node_count"] == row["kernel_launches"] == 128 and row["graph_launches"] == 1
                assert all(row[k] == value for k, value in WORK.items())
                assert all(row[k] is True for k in ("state_bit_exact", "output_bit_exact", "finite_state", "finite_output", "reset_before_event"))
                for key, expected in (("state_sha256", "expected_state_sha256"), ("output_sha256", "expected_output_sha256"), ("payload_sha256", "expected_payload_sha256"), ("initial_payload_sha256", "initial_payload_sha256")):
                    assert digest(row[key]) == reference[expected]
                initial_states.add(digest(row["initial_state_sha256"]))
                assert row["free_bytes_before"] >= FLOOR and row["free_bytes"] >= FLOOR and row["floor_bytes"] == FLOOR
                assert isinstance(row["elapsed_ms"], (int, float)) and not isinstance(row["elapsed_ms"], bool)
                assert math.isfinite(row["elapsed_ms"]) and row["elapsed_ms"] > 0
                pair_times[row["arm"]] = row["elapsed_ms"] * 1000 / 128
            if pair:
                for arm, values in times.items():
                    values.append(pair_times[arm])
                savings.append(pair_times["old"] - pair_times["new"])
        assert len(initial_states) == 1
        done = by_kind["gdn_screen_complete"]
        assert len(done) == 1 and done[0]["correctness_pass"] is True and done[0]["production_qualification"] is False
        assert done[0]["scope"] == SCOPE
        assert all(done[0][key] == WORK[key] for key in ("banks", "unique_state_bytes", "updates_per_bank"))
        assert done[0]["correctness_steps"] == 16 and done[0]["measured_pairs"] == 5 and done[0]["excluded_warmup_pairs"] == 1
        measurements.append({"ordinal": ordinal, "arms_us_per_call": {arm: {"median": statistics.median(values), "range": [min(values), max(values)], "samples": values} for arm, values in times.items()}, "paired_savings_us": savings, "median_saving_us": statistics.median(savings), "all_pairs_faster": all(value > 0 for value in savings)})
    projected = 191 * 15 * sum(row["median_saving_us"] for row in measurements) / 1e6
    return {"source_commit": rows[0]["source_commit"], "scope": SCOPE, "correctness_pass": True, "measurements": measurements, "work": WORK, "baseline_turn_seconds": BASELINE_SECONDS, "estimated_seconds_saved_per_turn": projected, "estimated_turn_fraction": projected / BASELINE_SECONDS, "material_5pct_screen": projected > 0.05 * BASELINE_SECONDS, "all_pairs_faster": all(row["all_pairs_faster"] for row in measurements), "projection_assumption": "191 appends, 15 GDN layers per GPU; sum per-GPU median paired savings and transfer isolated cost unchanged. Full-turn validation required.", "production_qualification": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("journal", type=Path)
    parser.add_argument("output_path", type=Path, nargs="?")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.output_path
    if output is None or (args.output is not None and args.output_path is not None):
        parser.error("provide exactly one output path")
    result = analyze(args.journal)
    output.write_bytes((json.dumps(result, indent=2) + "\n").encode())
    print(json.dumps(result))


if __name__ == "__main__":
    main()
