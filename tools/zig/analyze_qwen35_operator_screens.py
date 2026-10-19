"""Validate paired offline Qwen expert and GDN operator screens."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
from pathlib import Path

UUIDS = ["GPU-f2c0ea1e-a2be-f0b4-6c23-ffe33b537a06", "GPU-dd2a988c-fe38-24ab-233a-9e9bb3b2645a"]
FLOOR = 1536 * 1024 * 1024
BASELINE_SECONDS = 1.8600909
EXPERT_SCOPE = "layer-zero expert operator seam; extrapolation is not whole-model performance"
GDN_SCOPE = "synthetic nonzero GDN operator seam; timed 64-bank working set exceeds L2; not physical DRAM bandwidth, a saved model trace, or full-model performance"
EXPERT_ONE_BYTES = {"up": 1179648, "down_f32": 589824}
GDN_WORK = {"banks": 64, "unique_state_bytes": 134217728, "bank_bytes": 2113536, "payload_bytes": 135266304, "updates_per_bank": 2}


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def digest(value):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None, "invalid SHA256")
    return value


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_window(window: Path, rows):
    receipt = read_json(window / "window.json")
    plan = read_json(window / "plan.json")
    require(receipt.get("passed") is True and receipt.get("restored") is True, "operator window did not pass and restore")
    require(receipt.get("clef_unchanged") is True and not receipt.get("cleanup_errors") and not receipt.get("sampler_failures"), "window reports Clef, cleanup, or sampler failure")
    require(plan.get("source_commit") == receipt.get("plan", {}).get("source_commit"), "receipt and plan source commit differ")
    marker = (window / "profile-SOURCE_COMMIT").read_text(encoding="utf-8").strip()
    require(marker == plan["source_commit"] == rows[0]["source_commit"], "journal, marker, and controller source differ")
    require(re.fullmatch(r"[0-9a-f]{40}", marker) is not None, "invalid screen source commit")
    require(rows[0].get("identities") and [x.get("uuid") for x in rows[0]["identities"]] == UUIDS, "journal physical GPU order differs")
    memory = receipt.get("memory", [])
    require(memory, "missing controller GPU samples")
    for sample in memory:
        free = sample.get("free_mib", {})
        require(set(free) == set(UUIDS) and all(type(n) is int and n >= 1536 for n in free.values()), "controller UUID or memory floor failed")
    restoration = receipt.get("restoration") or {}
    require(restoration.get("passed") is True and restoration.get("clef_unchanged") is True, "restoration failed")
    match = next((x for x in plan.get("experiments", []) if x.get("name") == "profile"), None)
    require(match is not None, "missing profile experiment")
    receipt_match = next((x for x in receipt.get("plan", {}).get("experiments", []) if x.get("name") == "profile"), None)
    require(receipt_match is not None and all(receipt_match.get(key) == match.get(key) for key in ("executable", "sha256", "input", "input_sha256")), "controller receipt profile pin differs from plan")
    exe = Path(match["executable"])
    require(exe.is_file() and hashlib.sha256(exe.read_bytes()).hexdigest() == match["sha256"], "screen executable hash differs from controller plan")
    input_path = window / "profile-input.json"
    require(input_path.is_file() and hashlib.sha256(input_path.read_bytes()).hexdigest() == match.get("input_sha256"), "copied screen input SHA256 differs from controller plan")
    input_data = read_json(input_path)
    require(len(input_data.get("prompt", [])) == 64 and input_data.get("output_tokens") == 128 and input_data.get("forced") == [] and input_data.get("context") == 4096 and input_data.get("cut") == 20 and input_data.get("gdn_layout_screen") is True, "screen input is not the pinned 64/128 two-operator run")
    fixture_dir = Path(input_data["expert_fixture"])
    manifest_sha = hashlib.sha256((fixture_dir / "manifest.json").read_bytes()).hexdigest()
    require(manifest_sha == input_data.get("expert_fixture_sha256"), "expert fixture manifest differs from pinned input SHA256")
    experiments = receipt.get("experiments", [])
    require(experiments and all(e.get("exit_code") == 0 and e.get("completion", {}).get("kind") == "complete" for e in experiments), "controller records experiment error or incomplete completion")
    run = next((x for x in experiments if x.get("name") == "profile"), None)
    require(run is not None and run.get("passed") is True and run.get("command", [None, None, None])[2] == match["input"] and run.get("outputs") == ["profile-results\\results.jsonl"], "controller experiment does not match the pinned input and journal")
    require(rows[-1].get("kind") == "complete" and rows[-1].get("correctness_pass") is True, "journal lacks passing final completion")
    require(not any(row.get("kind") == "failure" for row in rows), "journal contains failure row")
    return {"source_commit": marker, "executable_sha256": match["sha256"], "input_sha256": match["input_sha256"], "expert_fixture_manifest_sha256": manifest_sha, "controller_minimum_free_mib": {u: min(sample["free_mib"][u] for sample in memory) for u in UUIDS}, "restored": True, "clef_unchanged": True}


def finite_time(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def expert(rows):
    results = []
    saving = {}
    allowed = {"start", "complete", "expert_fixture", "expert_plan", "expert_bank_parity", "expert_kernel", "expert_sample", "expert_projection_complete"}
    require(all(r.get("kind") in allowed for r in rows), "unknown expert journal kind")
    require(sum(r["kind"] == "expert_fixture" for r in rows) == 2, "expected two expert fixture records")
    require(sum(r["kind"] == "expert_kernel" for r in rows) == 12, "expected 12 expert kernel records")
    fixture = [r for r in rows if r["kind"] == "expert_fixture"]
    require(all(digest(r["manifest_sha256"]) == digest(fixture[0]["manifest_sha256"]) for r in fixture), "fixture manifest differs by GPU")
    for ordinal in (0, 1):
        dev = [r for r in rows if r.get("ordinal") == ordinal]
        f = [r for r in dev if r["kind"] == "expert_fixture"]
        require(len(f) == 1 and sorted(f[0]["expert_order"]) == list(range(256)) and f[0]["bank0_original_ids"] == f[0]["expert_order"][:8] and f[0]["bank0_remapped_ids"] == list(range(8)), "invalid expert fixture order or bank-zero control")
        for epi in ("up", "down_f32"):
            group = [r for r in dev if r.get("epi") == epi]
            plan = [r for r in group if r["kind"] == "expert_plan"]
            require(len(plan) == 1 and plan[0]["actual_items"] == plan[0]["bound"] == 8 and plan[0]["picks"] == list(range(8)) and plan[0]["banks"] == 32 and plan[0]["limit"] == 0 and plan[0]["relu2"] is False, "expert plan mismatch")
            complete = [r for r in group if r["kind"] == "expert_projection_complete"]
            require(len(complete) == 1 and all(complete[0][k] is True for k in ("all_bank_prechecks", "saved_bank0_exact", "graph_finals_exact")) and complete[0]["timed_pairs_per_candidate"] == 5, "expert projection incomplete")
            kernels = [r for r in group if r["kind"] == "expert_kernel"]
            require(len(kernels) == 3 and {r["variant"] for r in kernels} == {"wn4_baseline", "wn4_n16_pipe4", "wn1_baseline"}, "expert kernel roster mismatch")
            expected_warps = {"wn4_baseline": 4, "wn4_n16_pipe4": 4, "wn1_baseline": 1}
            expected_grid = {"wn4_baseline": (32 if epi == "up" else 128), "wn4_n16_pipe4": (64 if epi == "up" else 256), "wn1_baseline": (128 if epi == "up" else 512)}
            baseline_symbol = "_ZN16tf_nvfp4_experts19nvfp4_expert_kernelILi2ELi2ELi4EEEvPK13__nv_bfloat16iiPK5uint4PKfiiPKiSA_SA_Pvifi" if epi == "up" else "_ZN16tf_nvfp4_experts19nvfp4_expert_kernelILi1ELi0ELi4EEEvPK13__nv_bfloat16iiPK5uint4PKfiiPKiSA_SA_Pvifi"
            expected_symbols = {"wn4_baseline": baseline_symbol, "wn1_baseline": baseline_symbol.replace("ELi4EE", "ELi1EE"), "wn4_n16_pipe4": "tf_nvfp4_n16_pipe4_swiglu" if epi == "up" else "tf_nvfp4_n16_pipe4_down_f32"}
            for k in kernels:
                v = k["variant"]
                require(k["warps"] == expected_warps[v] and k["block"] == 32 * expected_warps[v] and k["grid"] == expected_grid[v], "expert function geometry mismatch")
                parts = 2 if epi == "up" else 1
                expected_per_launch = 8 * (EXPERT_ONE_BYTES[epi] + parts * 4)
                expected_unique = 256 * EXPERT_ONE_BYTES[epi] + 256 * parts * 4
                require(k["kernel_launches"] == k["verified_kernel_nodes"] == 128 and k["banks"] == 32 and k["logical_weight_and_scale_bytes_per_launch"] == expected_per_launch and k["logical_unique_weight_and_scale_bytes"] == expected_unique, "expert launch or byte accounting mismatch")
                require(k["symbol"] == expected_symbols[v], "expert function symbol mismatch")
            for variant in ("wn4_n16_pipe4", "wn1_baseline"):
                parity = [r for r in group if r["kind"] == "expert_bank_parity" and r["variant"] == variant]
                require(len(parity) == 32 and sorted(r["bank"] for r in parity) == list(range(32)), "expert bank parity incomplete")
                require(all(r["warps"] == expected_warps[variant] and r["finite"] is True and r["bit_exact"] is True and r["saved_real_output"] is (r["bank"] == 0) for r in parity), "expert bank parity failed")
                samples = [r for r in group if r["kind"] == "expert_sample" and r["comparison"] == variant]
                require(len(samples) == 12, "expert samples incomplete")
                arms = {"wn4_baseline": [], variant: []}
                paired = []
                for pair in range(6):
                    both = sorted((r for r in samples if r["pair"] == pair), key=lambda r: r["order"])
                    order = ["wn4_baseline", variant] if pair % 2 == 0 else [variant, "wn4_baseline"]
                    require(len(both) == 2 and [r["arm"] for r in both] == order and [r["order"] for r in both] == [0, 1], "expert pair order mismatch")
                    times = {}
                    for r in both:
                        require(r["warmup"] is (pair == 0) and r["kernel_launches"] == 128 and r["banks"] == 32 and r["final_bank"] == 31 and r["finite"] is True and r["bit_exact_final"] is True and finite_time(r["elapsed_ms"]), "invalid expert sample")
                        times[r["arm"]] = r["elapsed_ms"] * 1000 / 128
                    if pair:
                        for arm, value in times.items():
                            arms[arm].append(value)
                        paired.append(times["wn4_baseline"] - times[variant])
                paired_spread = max(paired) - min(paired)
                advance = min(paired) > 0 and statistics.median(paired) > paired_spread
                item = {"ordinal": ordinal, "epi": epi, "candidate": variant, "arms_us_per_call": {a: {"median": statistics.median(v), "range": [min(v), max(v)], "samples": v} for a, v in arms.items()}, "paired_savings_us": paired, "paired_savings_range_us": [min(paired), max(paired)], "paired_savings_spread_us": paired_spread, "median_saving_us": statistics.median(paired), "all_pairs_faster": all(x > 0 for x in paired), "advance_screen_pass": advance, "noise_assessment": "median paired saving exceeds the full five-pair saving spread" if advance else "paired saving spread is at least the median saving or includes a nonpositive pair"}
                saving[ordinal, epi, variant] = item
                results.append(item)
    projections = {}
    for candidate in ("wn4_n16_pipe4", "wn1_baseline"):
        by_epi = {}
        for epi in ("up", "down_f32"):
            us = sum(saving[gpu, epi, candidate]["median_saving_us"] for gpu in (0, 1))
            sec = 191 * 20 * us / 1e6
            by_epi[epi] = {"estimated_seconds_saved_per_turn": sec, "estimated_turn_fraction": sec / BASELINE_SECONDS, "material_5pct_screen": sec > .05 * BASELINE_SECONDS, "all_pairs_faster": all(saving[gpu, epi, candidate]["all_pairs_faster"] for gpu in (0, 1))}
        projections[candidate] = by_epi
    return {"scope": EXPERT_SCOPE, "measurements": results, "projection": projections, "projection_assumption": "191 appends and 20 layer-zero micro calls per GPU; each per-call saving transfers uniformly to every layer, with transfers unchanged. This is a synthetic extrapolation, not model speed or DRAM utilization."}


def gdn(rows):
    results = []
    allowed = {"start", "complete", "gdn_layout_fixture", "gdn_layout_negative_control", "gdn_layout_unwritten_state_control", "gdn_layout_step_parity", "gdn_layout_graph", "gdn_layout_sample", "gdn_layout_screen_complete"}
    require(all(r.get("kind") in allowed for r in rows), "unknown GDN journal kind")
    for ordinal in (0, 1):
        dev = [r for r in rows if r.get("ordinal") == ordinal]
        fixture = [r for r in dev if r["kind"] == "gdn_layout_fixture"]
        require(len(fixture) == 1, "GDN fixture missing")
        f = fixture[0]
        expected_fixture = {"seed": 0x35D019, "pattern_version": 3, "baseline_state_layout": "head,value,key", "candidate_state_layout": "head,key,value", "beta_dtype": "BF16", "value_heads": 32, "key_heads": 16, "dimension": 128, "state_bytes": 2097152, "out_bytes": 16384, "q_dtype": "FP32", "k_dtype": "FP32", "v_dtype": "BF16", "state_dtype": "FP32", "out_dtype": "FP32", "q_elements": 2048, "k_elements": 2048, "v_elements": 4096, "decay_elements": 32, "beta_elements": 32, "scope": GDN_SCOPE}
        require(all(f[k] == v for k, v in expected_fixture.items()), "GDN fixture shape or dtype mismatch")
        initial = digest(f["initial_state_sha256"])
        require(f["free_bytes"] >= FLOOR and f["floor_bytes"] == FLOOR, "GDN fixture floor failed")
        negatives = [r for r in dev if r["kind"] == "gdn_layout_negative_control"]
        require(len(negatives) == 1 and all(negatives[0][k] is True for k in ("poisoned_output_rejected", "changed_state_rejected", "changed_output_rejected", "late_bank_state_mismatch_rejected", "late_bank_output_mismatch_rejected")), "GDN negative controls failed")
        unwritten = [r for r in dev if r["kind"] == "gdn_layout_unwritten_state_control"]
        require(len(unwritten) == 1 and unwritten[0]["unchanged_initial_state_with_correct_output_rejected"] is True, "GDN unwritten-state control failed")
        parity = sorted((r for r in dev if r["kind"] == "gdn_layout_step_parity"), key=lambda x: x["step"])
        require(len(parity) == 16 and [r["step"] for r in parity] == list(range(16)), "GDN changing-step rows missing")
        require(len({digest(r["inputs_sha256"]) for r in parity}) == 16, "GDN step inputs did not vary")
        require(len({digest(r["state_sha256"]) for r in parity}) == 16 and len({digest(r["output_sha256"]) for r in parity}) == 16, "GDN step state or output hashes did not vary")
        for r in parity:
            require(all(r[k] is True for k in ("state_bit_exact", "output_bit_exact", "finite_state", "finite_output", "unwritten_initial_state_rejected")) and digest(r["state_sha256"]) != initial and r["free_bytes"] >= FLOOR, "GDN correctness or floor failed")
            digest(r["output_sha256"])
        graphs = [r for r in dev if r["kind"] == "gdn_layout_graph"]
        require(len(graphs) == 2 and {r["arm"] for r in graphs} == {"old", "new"}, "GDN graphs missing")
        old, new = next(r for r in graphs if r["arm"] == "old"), next(r for r in graphs if r["arm"] == "new")
        expected_kernel = {"old": "tf_q35_hf_gdn_chain", "new": "tf_q35_hf_gdn_chain_transposed"}
        for graph in graphs:
            require(graph["kernel"] == expected_kernel[graph["arm"]] and graph["node_count"] == graph["kernel_launches"] == graph["verified_kernel_nodes"] == 128 and graph["verified_updates_per_bank"] == 2 and graph["local_delta_layer"] == 0, "GDN graph function or nodes mismatch")
            require((graph["grid_x"], graph["grid_z"], graph["block_x"]) == (1, 32, 128), "GDN graph geometry mismatch")
            require(all(graph[k] == v for k, v in GDN_WORK.items()) and graph["scope"] == GDN_SCOPE and graph["evolving_state"] is True and graph["free_bytes"] >= FLOOR, "GDN graph working set mismatch")
        hashes = ("inputs_sha256", "expected_state_sha256", "expected_output_sha256", "expected_payload_sha256", "initial_payload_sha256")
        require(all(digest(old[k]) == digest(new[k]) for k in hashes), "GDN graph expected payload differs across layouts")
        require(old["inputs_sha256"] == parity[0]["inputs_sha256"] and old["initial_payload_sha256"] != old["expected_payload_sha256"], "GDN graph input or changing-state hash mismatch")
        samples = [r for r in dev if r["kind"] == "gdn_layout_sample"]
        require(len(samples) == 12, "GDN sample roster incomplete")
        arm_times = {"old": [], "new": []}
        paired = []
        init_hashes = set()
        for pair in range(6):
            both = sorted((r for r in samples if r["pair"] == pair), key=lambda r: r["order"])
            order = ["old", "new"] if pair % 2 == 0 else ["new", "old"]
            require(len(both) == 2 and [r["arm"] for r in both] == order and [r["order"] for r in both] == [0, 1], "GDN pair order mismatch")
            times = {}
            for r in both:
                require(r["warmup"] is (pair == 0) and r["reset_before_event"] is True and r["node_count"] == r["kernel_launches"] == 128 and r["graph_launches"] == 1 and all(r[k] is True for k in ("state_bit_exact", "output_bit_exact", "finite_state", "finite_output")) and finite_time(r["elapsed_ms"]), "invalid GDN sample")
                require(all(r[k] == v for k, v in GDN_WORK.items()), "GDN sample working set mismatch")
                require(digest(r["state_sha256"]) == old["expected_state_sha256"] and digest(r["output_sha256"]) == old["expected_output_sha256"] and digest(r["payload_sha256"]) == old["expected_payload_sha256"] and digest(r["initial_payload_sha256"]) == old["initial_payload_sha256"], "timed GDN payload differs from graph oracle")
                require(digest(r["initial_state_sha256"]) != digest(r["state_sha256"]), "GDN initial state hash did not evolve")
                init_hashes.add(digest(r["initial_state_sha256"]))
                require(r["free_bytes_before"] >= FLOOR and r["free_bytes"] >= FLOOR and r["floor_bytes"] == FLOOR and r["scope"] == GDN_SCOPE, "GDN sample memory floor or scope failed")
                times[r["arm"]] = r["elapsed_ms"] * 1000 / 128
            if pair:
                for arm, value in times.items():
                    arm_times[arm].append(value)
                paired.append(times["old"] - times["new"])
        require(len(init_hashes) == 1, "GDN reset inputs differ across timed samples")
        done = [r for r in dev if r["kind"] == "gdn_layout_screen_complete"]
        require(len(done) == 1 and done[0]["correctness_pass"] is True and done[0]["production_qualification"] is False and done[0]["correctness_steps"] == 16 and done[0]["measured_pairs"] == 5 and done[0]["excluded_warmup_pairs"] == 1 and all(done[0][k] == GDN_WORK[k] for k in ("banks", "unique_state_bytes", "updates_per_bank")) and done[0]["scope"] == GDN_SCOPE, "GDN completion mismatch")
        results.append({"ordinal": ordinal, "arms_us_per_call": {a: {"median": statistics.median(v), "range": [min(v), max(v)], "samples": v} for a, v in arm_times.items()}, "paired_savings_us": paired, "median_saving_us": statistics.median(paired), "all_pairs_faster": all(x > 0 for x in paired)})
    projected = 191 * 15 * sum(r["median_saving_us"] for r in results) / 1e6
    for result in results:
        spread = max(result["paired_savings_us"]) - min(result["paired_savings_us"])
        result["paired_savings_range_us"] = [min(result["paired_savings_us"]), max(result["paired_savings_us"])]
        result["paired_savings_spread_us"] = spread
        result["advance_screen_pass"] = min(result["paired_savings_us"]) > 0 and result["median_saving_us"] > spread
        result["noise_assessment"] = "median paired saving exceeds the full five-pair saving spread" if result["advance_screen_pass"] else "paired saving spread is at least the median saving or includes a nonpositive pair"
    return {"scope": GDN_SCOPE, "measurements": results, "work": GDN_WORK, "estimated_seconds_saved_per_turn": projected, "estimated_turn_fraction": projected / BASELINE_SECONDS, "material_5pct_screen": projected > .05 * BASELINE_SECONDS, "all_pairs_faster": all(r["all_pairs_faster"] for r in results), "advance_screen_pass": all(r["advance_screen_pass"] for r in results), "projection_assumption": "191 appends and 15 GDN layers per GPU; sum per-GPU paired savings and transfer the layer-zero synthetic operator saving uniformly with transfer cost unchanged. This is not model speed or DRAM utilization."}


def analyze(window: Path, baseline_analysis: Path):
    base = read_json(baseline_analysis)
    require(base.get("source_commit") == "8a39ba55764aafad58d8dae402b287b21d0e760c" and base.get("input") == {"prompt_tokens": 64, "output_tokens": 128, "greedy": True}, "baseline is not the pinned newshared64/128 case")
    baseline = base["arms"]["graph"]["summary"]["inclusive_turn_seconds"]["median"]
    require(baseline == BASELINE_SECONDS, "baseline turn median differs from pinned materiality denominator")
    journal_path = window / "profile-results" / "results.jsonl"
    rows = [json.loads(line) for line in journal_path.read_text(encoding="utf-8").splitlines() if line]
    require(rows and rows[0]["kind"] == "start" and rows[-1]["kind"] == "complete", "screen journal boundaries invalid")
    allowed = {"start", "complete", "expert_fixture", "expert_plan", "expert_bank_parity", "expert_kernel", "expert_sample", "expert_projection_complete", "gdn_layout_fixture", "gdn_layout_negative_control", "gdn_layout_unwritten_state_control", "gdn_layout_step_parity", "gdn_layout_graph", "gdn_layout_sample", "gdn_layout_screen_complete"}
    require(all(r.get("kind") in allowed for r in rows) and all(r.get("ordinal") in (None, 0, 1) for r in rows), "unknown journal kind or GPU ordinal")
    controller = validate_window(window, rows)
    start = rows[0]
    require(start["implementation"] == "shared-qwen-engine" and start["scope"] == "isolated expert and GDN operator screens; no model load or model throughput claim" and (start["prompt_tokens"], start["output_tokens"], start["context"], start["cut"]) == (64, 128, 4096, 20), "screen input identity mismatch")
    require(rows[-1].get("scope") == "isolated operator parity and timing screens" and rows[-1].get("production_qualification") is False, "screen completion scope invalid")
    expert_rows = [r for r in rows if r["kind"].startswith("expert_") or r["kind"] in ("start", "complete")]
    gdn_rows = [r for r in rows if r["kind"].startswith("gdn_layout_") or r["kind"] in ("start", "complete")]
    expert_result = expert(expert_rows)
    gdn_result = gdn(gdn_rows)
    require(all(digest(r["manifest_sha256"]) == controller["expert_fixture_manifest_sha256"] for r in rows if r["kind"] == "expert_fixture"), "journal expert fixtures differ from copied input manifest")
    for result, label in ((expert_result, "expert"), (gdn_result, "GDN")):
        result["materiality_baseline_seconds"] = baseline
        result["materiality_baseline_source"] = base["source_commit"]
        if label == "GDN":
            result["candidate_for_full_trial"] = result["advance_screen_pass"] and result["estimated_seconds_saved_per_turn"] > 0.05 * baseline
        else:
            for candidate in ("wn4_n16_pipe4", "wn1_baseline"):
                by_epi = result["projection"][candidate]
                combined = sum(by_epi[epi]["estimated_seconds_saved_per_turn"] for epi in ("up", "down_f32"))
                all_faster = all(x["advance_screen_pass"] for x in result["measurements"] if x["candidate"] == candidate)
                result["projection"][candidate]["combined_estimated_seconds_saved_per_turn"] = combined
                result["projection"][candidate]["combined_estimated_turn_fraction"] = combined / baseline
                result["projection"][candidate]["candidate_for_full_trial"] = all_faster and combined > 0.05 * baseline
    return {"source_commit": controller["source_commit"], "controller": controller, "correctness_pass": True, "expert": expert_result, "gdn_layout": gdn_result, "production_qualification": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("window", type=Path)
    parser.add_argument("--baseline-analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.window, args.baseline_analysis)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
