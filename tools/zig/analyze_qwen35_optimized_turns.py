"""Verify matched full-turn work and exactness before reporting throughput."""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


def spread(values):
    return {"median": statistics.median(values), "range": [min(values), max(values)], "samples": values}


def analyze(path):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert rows[0]["kind"] == "start" and rows[-1]["kind"] == "complete" and rows[-1]["correctness_pass"]
    assert rows[-1]["production_qualification"] is False and not any(r["kind"] == "failure" for r in rows)
    assert [i["uuid"] for i in rows[0]["identities"]] == ["GPU-f2c0ea1e-a2be-f0b4-6c23-ffe33b537a06", "GPU-dd2a988c-fe38-24ab-233a-9e9bb3b2645a"]
    setup = [r for r in rows if r["kind"] == "optimized_setup"]
    assert len(setup) == 1 and setup[0]["shared_engine_and_session"] and setup[0]["context"] == 4096 and setup[0]["prompt_tokens"] == 64 and setup[0]["output_tokens"] == 128
    controls = [r for r in rows if r["kind"] == "optimized_controls"]
    assert len(controls) == 1 and controls[0]["admission_before_mutation"] and controls[0]["stale_position_changes_state"]
    correctness = [r for r in rows if r["kind"] == "optimized_correctness"]
    assert len(correctness) == 3 and {r["arm"] for r in correctness} == {"baseline", "parallel", "parallel_up"}
    tokens, state = correctness[0]["tokens"], correctness[0]["statehash"]
    assert len(tokens) == 128 and all(0 <= v < 248320 for v in tokens) and len(state) == 64
    for r in correctness:
        assert r["logit_rows"] == 128 and r["exact_rows_tokens_and_final_state"] and r["tokens"] == tokens and r["statehash"] == state
        assert (r["appends"], r["launches"], r["updates"], r["position"], r["nodes"], r["dynamic_nodes"]) == (191, 382, 4011, 191, 993, 21)
    for arm in ("baseline", "parallel", "parallel_up"):
        graphs = [r for r in rows if r["kind"] == "optimized_graph_partition" and r["arm"] == arm]
        assert len(graphs) == 2 and {r["rank"] for r in graphs} == {0, 1}
        assert sum(r["nodes"] for r in graphs) == 993 and sum(r["dynamic_nodes"] for r in graphs) == 21
        for r in graphs:
            assert r["select_nodes"] == r["up_nodes"] == r["down_nodes"] == 20 and r["select_grid"] == 1 and r["select_block"] == 256
            assert r["select_symbol"] == ("tf_qwen35_route_select" if arm == "baseline" else "tf_qwen35_route_select_parallel")
            assert r["up_grid"] == (64 if arm == "parallel_up" else 32) and r["up_block"] == 128 and r["down_variant"] == "baseline"
    report = {"source_commit": rows[0]["source_commit"], "scope": "synthetic64/128 greedy c1 ctx4096, serial two-GPU layer split; diagnostic graph, not HTTP or production qualification", "correctness_pass": True, "statehash": state, "comparisons": {}}
    for comparison, arms in [("router", ["baseline", "parallel"]), ("up", ["parallel", "parallel_up"])]:
        samples = [r for r in rows if r["kind"] == "optimized_turn" and r["comparison"] == comparison]
        assert len(samples) == 12
        collected = {arm: {"decode_tok_s": [], "total_s": [], "prefill_s": [], "ttft_s": []} for arm in arms}
        reductions = []
        for pair in range(6):
            both = sorted((r for r in samples if r["pair"] == pair), key=lambda r: r["order"])
            assert len(both) == 2 and [r["order"] for r in both] == [0, 1] and [r["arm"] for r in both] == (arms if pair % 2 == 0 else list(reversed(arms)))
            for r in both:
                assert r["warmup"] == (pair == 0) and r["exact_tokens_and_final_state"] and r["tokens"] == tokens and r["statehash"] == state
                assert (r["appends"], r["launches"], r["updates"], r["position"]) == (191, 382, 4011, 191)
                times = r["times"]
                assert times["decode_tokens"] == 127 and times["appends"] == 191 and times["draws"] == 128
                assert all(math.isfinite(times[k]) and times[k] > 0 for k in ("total_ns", "decode_ns", "prefill_ns", "ttft_ns"))
                if pair:
                    values = collected[r["arm"]]
                    values["decode_tok_s"].append(127e9 / times["decode_ns"])
                    for key, field in [("total_s", "total_ns"), ("prefill_s", "prefill_ns"), ("ttft_s", "ttft_ns")]:
                        values[key].append(times[field] / 1e9)
            if pair:
                elapsed = {r["arm"]: r["times"]["total_ns"] for r in both}
                reductions.append(1 - elapsed[arms[1]] / elapsed[arms[0]])
        report["comparisons"][comparison] = {"arms": {arm: {metric: spread(values) for metric, values in metrics.items()} for arm, metrics in collected.items()}, "paired_turn_latency_reductions": reductions, "median_turn_latency_reduction": statistics.median(reductions), "all_pairs_faster": min(reductions) > 0}
    memory = [r for r in rows if r["kind"] == "optimized_memory"]
    assert len(memory) >= 62
    assert all(r["floor_bytes"] == 1536 * 1024**2 and r["free_bytes"] >= r["floor_bytes"] for r in memory)
    report["in_process_min_free_mib"] = {str(rank): min(r["free_bytes"] for r in memory if r["rank"] == rank) / 1024**2 for rank in (0, 1)}
    report["production_qualification"] = False
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
