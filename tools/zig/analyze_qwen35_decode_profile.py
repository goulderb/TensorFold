"""Validate and summarize a restored Qwen fixed-step diagnostic window."""
import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("window", type=Path)
parser.add_argument("--output", type=Path)
parser.add_argument("--require-generation-graph", action="store_true", help="Reject missing changing-position generation evidence")
args = parser.parse_args()
receipt = json.loads((args.window / "window.json").read_text())
assert receipt["passed"] and receipt["restored"] and receipt["clef_unchanged"], "window failed or is not restored"
assert not receipt["cleanup_errors"] and not receipt["sampler_failures"]
rows = [json.loads(line) for line in (args.window / "profile-results/results.jsonl").read_text().splitlines()]
assert rows[0]["kind"] == "start" and rows[0]["source_commit"] == receipt["plan"]["source_commit"]
assert rows[-1]["kind"] == "complete" and rows[-1]["correctness_pass"]
assert not any(row["kind"] == "failure" for row in rows)
nodes = [row for row in rows if row["kind"] == "graph_nodes"]
assert len(nodes) == 1 and nodes[0]["total"] == 953 and nodes[0]["ordinary_append_state_and_logits_equal"]
parity = [row for row in rows if row["kind"] == "step_parity"]
assert [row["step"] for row in parity] == list(range(16))
assert len({row["state_and_logits_sha256"] for row in parity}) > 1, "state did not evolve"
turns = [row for row in rows if row["kind"] == "turn"]
verified = [row for row in rows if row["kind"] == "pair_verified"]
assert len(turns) == 12 and [row["pair"] for row in verified] == list(range(6))
assert all(row["exact_state_and_logits"] for row in verified)
latencies = defaultdict(list)
gains = []
for pair in range(6):
    both = [row for row in turns if row["pair"] == pair]
    assert len(both) == 2
    assert [row["arm"] for row in both] == (["eager", "graph"] if pair % 2 == 0 else ["graph", "eager"])
    assert both[0]["persistent_state_and_logits_sha256"] == both[1]["persistent_state_and_logits_sha256"]
    for row in both:
        assert row["warmup"] == (pair == 0) and row["steps"] == 16 and row["elapsed_ns"] > 0
        if pair:
            latencies[row["arm"]].append(row["elapsed_ns"] / 16 / 1e6)
    if pair:
        times = {row["arm"]: row["elapsed_ns"] for row in both}
        gains.append(1 - times["graph"] / times["eager"])
assert len({row["persistent_state_and_logits_sha256"] for row in turns}) == 1
summaries = {arm: {"median_ms_per_step": statistics.median(values), "range_ms_per_step": [min(values), max(values)]}
             for arm, values in latencies.items()}
phase_totals = defaultdict(float)
phase_keys = set()
for row in rows:
    if row["kind"] != "phase":
        continue
    assert row["sample"] in range(5) and row["rank"] in (0, 1)
    assert row["layer"] in range(40) and math.isfinite(row["ms"]) and row["ms"] >= 0
    key = (row["sample"], row["rank"], row["layer"], row["phase"])
    assert key not in phase_keys
    phase_keys.add(key)
    phase_totals[row["sample"], row["phase"]] += row["ms"]
categories = sorted({key[1] for key in phase_totals})
assert len(phase_keys) > 2000 and len(categories) >= 12
phase_summary = {name: {"median_ms": statistics.median([phase_totals[sample, name] for sample in range(5)]),
                        "range_ms": [min(phase_totals[sample, name] for sample in range(5)),
                                     max(phase_totals[sample, name] for sample in range(5))]}
                 for name in categories}
profile_turns = [row for row in rows if row["kind"] == "profile_turn"]
assert [row["sample"] for row in profile_turns] == list(range(5))
overheads = [row["phased_elapsed_ns"] / row["control_elapsed_ns"] - 1 for row in profile_turns]
free = defaultdict(list)
for sample in receipt["memory"]:
    assert len(sample["free_mib"]) == 2
    for uuid, mib in sample["free_mib"].items():
        assert mib >= 1536
        free[uuid].append(mib)
assert len(free) == 2
result = {
    "scope": "fixed-token fixed-position evolving-state forward; not generation throughput or quality qualification",
    "source_commit": rows[0]["source_commit"],
    "latency": summaries,
    "paired_latency_reductions": gains,
    "median_latency_reduction": 1 - summaries["graph"]["median_ms_per_step"] / summaries["eager"]["median_ms_per_step"],
    "positive_pairs": sum(value > 0 for value in gains),
    "graph_kernel_nodes": 953,
    "intermediate_state_checks": len(parity),
    "phase_rows": len(phase_keys),
    "phases": phase_summary,
    "event_instrumentation_overhead_fraction": {"median": statistics.median(overheads), "range": [min(overheads), max(overheads)]},
    "minimum_free_mib": {uuid: min(values) for uuid, values in free.items()},
    "restoration": receipt["restoration"],
}
if any(row["kind"] == "router_graph_nodes" for row in rows):
    router_nodes = [row for row in rows if row["kind"] == "router_graph_nodes"]
    assert len(router_nodes) == 1 and router_nodes[0]["total"] == 993
    router_parity = [row for row in rows if row["kind"] == "router_parity"]
    assert len(router_parity) == 80
    assert {(row["rank"], row["layer"], row["fixture"]) for row in router_parity} == {
        (layer // 20, layer, fixture) for layer in range(40) for fixture in ("saved-real-row", "zero-ties")
    }
    assert all(row["exact_picks_and_fp32_weights"] for row in router_parity)
    cooperative_parity = [row for row in rows if row["kind"] == "cooperative_step_parity"]
    assert [row["step"] for row in cooperative_parity] == list(range(16))
    assert [row["state_and_logits_sha256"] for row in cooperative_parity] == [row["state_and_logits_sha256"] for row in parity]
    router_turns = [row for row in rows if row["kind"] == "router_turn"]
    assert len(router_turns) == 24
    router_summary = {}
    for graph in (False, True):
        values = defaultdict(list)
        pair_gains = []
        for pair in range(6):
            both = [row for row in router_turns if row["graph"] == graph and row["pair"] == pair]
            assert len(both) == 2 and [row["order"] for row in both] == [0, 1]
            assert [row["cooperative"] for row in both] == ([False, True] if pair % 2 == 0 else [True, False])
            assert all(row["warmup"] == (pair == 0) and row["steps"] == 16 and row["exact_state_and_logits"] for row in both)
            assert all(row["elapsed_ns"] > 0 for row in both)
            if pair:
                times = {row["cooperative"]: row["elapsed_ns"] for row in both}
                pair_gains.append(1 - times[True] / times[False])
                for arm, elapsed in times.items():
                    values["cooperative" if arm else "baseline"].append(elapsed / 16 / 1e6)
        router_summary["graph" if graph else "eager"] = {
            arm: {"median_ms_per_step": statistics.median(v), "range_ms_per_step": [min(v), max(v)]}
            for arm, v in values.items()
        } | {"paired_latency_reductions": pair_gains}
    generation_parity = [row for row in rows if row["kind"] == "generation_parity"]
    assert len(generation_parity) == 1 and generation_parity[0]["rows"] == 128
    assert generation_parity[0]["exact_consumed_bf16_rows_and_tokens"]
    tokens = generation_parity[0]["reference_tokens"]
    assert len(tokens) == 128
    generation_turns = [row for row in rows if row["kind"] == "generation_turn"]
    assert len(generation_turns) == 12
    generation_values = defaultdict(lambda: defaultdict(list))
    for pair in range(6):
        both = [row for row in generation_turns if row["pair"] == pair]
        assert len(both) == 2 and [row["order"] for row in both] == [0, 1]
        assert [row["cooperative"] for row in both] == ([False, True] if pair % 2 == 0 else [True, False])
        for row in both:
            t = row["times"]
            assert row["tokens"] == tokens and row["warmup"] == (pair == 0)
            assert t["decode_tokens"] == 127 and t["appends"] == 191 and t["draws"] == 128
            assert t["total_ns"] == t["ttft_ns"] + t["decode_ns"]
            assert t["decode_ns"] > 0 and t["ttft_ns"] >= t["prefill_ns"] > 0 and t["reset_ns"] >= 0
            if pair:
                values = generation_values["cooperative" if row["cooperative"] else "baseline"]
                values["decode_tokens_per_second"].append(127e9 / t["decode_ns"])
                for name in ("total_ns", "prefill_ns", "ttft_ns", "reset_ns"):
                    values[name.removesuffix("_ns") + "_seconds"].append(t[name] / 1e9)
    result["router_candidate"] = {
        "route_fixture_checks": len(router_parity), "intermediate_checks": len(cooperative_parity),
        "fixed_step": router_summary,
        "synthetic_64_input_128_output_greedy": {
            arm: {name: {"median": statistics.median(v), "range": [min(v), max(v)]} for name, v in values.items()}
            for arm, values in generation_values.items()
        },
        "generation_exact_bf16_rows": 128,
        "generation_measured_pairs": 5,
        "production_qualified": False,
    }
    candidate_phases = [row for row in rows if row["kind"] == "router_phase"]
    if candidate_phases:
        assert len(candidate_phases) == len(phase_keys)
        assert {(r["sample"], r["rank"], r["layer"], r["phase"]) for r in candidate_phases} == phase_keys
        assert all(math.isfinite(r["ms"]) and r["ms"] >= 0 for r in candidate_phases)
        candidate_totals = defaultdict(float)
        for row in candidate_phases:
            candidate_totals[row["sample"], row["phase"]] += row["ms"]
        result["router_candidate"]["phases"] = {
            name: {"median_ms": statistics.median([candidate_totals[s, name] for s in range(5)]),
                   "range_ms": [min(candidate_totals[s, name] for s in range(5)), max(candidate_totals[s, name] for s in range(5))]}
            for name in categories
        }
        candidate_profile = [row for row in rows if row["kind"] == "router_profile_turn"]
        assert [r["sample"] for r in candidate_profile] == list(range(5))
        extra = [r["phased_elapsed_ns"] / r["control_elapsed_ns"] - 1 for r in candidate_profile]
        result["router_candidate"]["event_instrumentation_overhead_fraction"] = {
            "median": statistics.median(extra), "range": [min(extra), max(extra)]}
graph_setup = [r for r in rows if r["kind"] == "generation_graph_setup"]
if args.require_generation_graph:
    assert graph_setup, "missing changing-position generation evidence"
if graph_setup:
    assert len(graph_setup) == 1 and graph_setup[0]["nodes"] == 993 and graph_setup[0]["dynamic_nodes"] == 21
    assert graph_setup[0]["elapsed_ns"] > 0
    controls = [r for r in rows if r["kind"] == "generation_graph_controls"]
    assert len(controls) == 1 and controls[0]["admission_before_mutation"] and controls[0]["stale_position_changes_state"]
    graph_parity = [r for r in rows if r["kind"] == "generation_graph_parity"]
    assert len(graph_parity) == 1 and graph_parity[0]["rows"] == 128 and graph_parity[0]["exact_rows_tokens_and_final_state"]
    graph_tokens = graph_parity[0]["reference_tokens"]
    assert len(graph_tokens) == 128
    graph_turns = [r for r in rows if r["kind"] == "generation_graph_turn"]
    assert len(graph_turns) == 12
    values = defaultdict(lambda: defaultdict(list))
    pair_gains = []
    for pair in range(6):
        both = [r for r in graph_turns if r["pair"] == pair]
        assert len(both) == 2 and [r["order"] for r in both] == [0, 1]
        assert [r["graph"] for r in both] == ([False, True] if pair % 2 == 0 else [True, False])
        for row in both:
            assert row["warmup"] == (pair == 0) and row["tokens"] == graph_tokens and row["position"] == 191
            assert row.get("exact_tokens_and_final_state", row.get("exact_rows_tokens_and_final_state", False))
            assert [row[k] for k in ("observed_graph_appends", "observed_graph_launches", "observed_parameter_updates")] == ([191, 382, 4011] if row["graph"] else [0, 0, 0])
            t = row["times"]
            assert t["decode_tokens"] == 127 and t["appends"] == 191 and t["draws"] == 128
            assert t["total_ns"] == t["ttft_ns"] + t["decode_ns"]
            assert t["decode_ns"] > 0 and t["ttft_ns"] >= t["prefill_ns"] > 0 and t["reset_ns"] >= 0
            if pair:
                arm = values["graph" if row["graph"] else "eager"]
                arm["decode_tokens_per_second"].append(127e9 / t["decode_ns"])
                for name in ("total_ns", "prefill_ns", "ttft_ns", "reset_ns"):
                    arm[name.removesuffix("_ns") + "_seconds"].append(t[name] / 1e9)
        if pair:
            durations = {r["graph"]: r["times"]["total_ns"] for r in both}
            pair_gains.append(1 - durations[True] / durations[False])
    result["changing_position_generation"] = {
        "scope": "synthetic 64-input/128-output greedy, cooperative router, two serial GPU partitions; warmed compute graphs",
        "arms": {arm: {name: {"median": statistics.median(v), "range": [min(v), max(v)]} for name, v in data.items()} for arm, data in values.items()},
        "measured_pairs": 5,
        "paired_complete_turn_latency_reductions": pair_gains,
        "graph_construction_seconds": graph_setup[0]["elapsed_ns"] / 1e9,
        "exact_consumed_bf16_rows": 128,
        "row_parity_scope": "untimed captured turn; timed turns verify output tokens and final persistent state",
        "exact_final_persistent_state": True,
        "stale_position_negative_control_passed": True,
        "production_qualified": False,
    }
serialized = json.dumps(result, indent=2) + "\n"
if args.output:
    args.output.write_text(serialized, encoding="utf-8", newline="\n")
print(serialized, end="")
