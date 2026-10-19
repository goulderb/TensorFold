"""Validate the offline Qwen WN2 expert up-projection screen."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
from pathlib import Path

UUIDS = ["GPU-f2c0ea1e-a2be-f0b4-6c23-ffe33b537a06", "GPU-dd2a988c-fe38-24ab-233a-9e9bb3b2645a"]
FLOOR_MIB = 1536
BASELINE_SECONDS = 1.7232142
SOURCE_COMMIT = "4a13c64a68ed47e1a9b9ac606f14c334312f5dc1"
BASELINE_SOURCE_COMMIT = "86eeaa78eaad449cb6b2a17b46f9df851f8e817d"
ONE_EXPERT_WEIGHT_BYTES = 1179648
LOGICAL_BYTES_PER_LAUNCH = 8 * (ONE_EXPERT_WEIGHT_BYTES + 2 * 4)
UNIQUE_WEIGHT_AND_SCALE_BYTES = 256 * ONE_EXPERT_WEIGHT_BYTES + 256 * 2 * 4
SCOPE = "layer-zero Qwen expert up-projection operator screen; projection is an extrapolation, not whole-model performance"
FIXTURE_SCOPE = "real layer-zero expert fixture"
FIXTURE_MANIFEST_SHA256 = "c41ab1d6363d21f785f0e8d17f8a8030f041dd614aacf823169a3f9787fc366e"
BANK0_EXPERTS = [112, 106, 107, 238, 181, 57, 43, 200]
COMPARISONS = {"wn4_n16_pipe4", "wn2_n16_pipe4"}


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value) -> str:
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None, "invalid SHA256")
    return value


def finite_time(value) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def validate_window(window: Path, rows):
    receipt = read_json(window / "window.json")
    plan = read_json(window / "plan.json")
    require(receipt.get("passed") is True and receipt.get("restored") is True, "WN2 window did not pass and restore")
    require(receipt.get("clef_unchanged") is True and not receipt.get("cleanup_errors") and not receipt.get("sampler_failures"), "window reports Clef, cleanup, or sampler failure")
    require(plan.get("source_commit") == receipt.get("plan", {}).get("source_commit") == SOURCE_COMMIT, "receipt and plan source commit differ from pinned screen source")
    marker = (window / "profile-SOURCE_COMMIT").read_text(encoding="utf-8").strip()
    require(marker == plan["source_commit"] == rows[0].get("source_commit"), "journal, marker, and controller source differ")
    require(re.fullmatch(r"[0-9a-f]{40}", marker) is not None, "invalid screen source commit")
    require(rows[0].get("identities") and [x.get("uuid") for x in rows[0]["identities"]] == UUIDS, "journal physical GPU order differs")
    memory = receipt.get("memory", [])
    require(memory, "missing controller GPU samples")
    for sample in memory:
        free = sample.get("free_mib", {})
        require(set(free) == set(UUIDS) and all(type(n) is int and n >= FLOOR_MIB for n in free.values()), "controller UUID or memory floor failed")
    restoration = receipt.get("restoration") or {}
    require(restoration.get("passed") is True and restoration.get("model") == "nemotron-3.5-lightning" and restoration.get("content") == "READY" and receipt.get("clef_unchanged") is True, "restoration failed, model readiness failed, or Clef changed")
    require(set(restoration.get("free_mib", {})) == set(UUIDS) and all(type(n) is int and n >= FLOOR_MIB for n in restoration["free_mib"].values()), "restored model GPU memory floor failed")
    match = next((x for x in plan.get("experiments", []) if x.get("name") == "profile"), None)
    require(match is not None, "missing profile experiment")
    receipt_match = next((x for x in receipt.get("plan", {}).get("experiments", []) if x.get("name") == "profile"), None)
    pinned_fields = ("executable", "sha256", "input", "input_sha256")
    require(receipt_match is not None and all(receipt_match.get(key) == match.get(key) for key in pinned_fields), "controller receipt profile pin differs from plan")
    executable = Path(match["executable"])
    require(executable.is_file() and sha256_file(executable) == digest(match.get("sha256")), "screen executable hash differs from controller plan")
    input_path = window / "profile-input.json"
    require(input_path.is_file() and sha256_file(input_path) == digest(match.get("input_sha256")), "copied screen input SHA256 differs from controller plan")
    input_data = read_json(input_path)
    require(input_data.get("expert_wn2_screen") is True and input_data.get("expert_fixture") and input_data.get("expert_fixture_sha256"), "screen input is not pinned to the WN2 fixture")
    require(input_data.get("gdn_layout_screen") is False and input_data.get("gdn_layout_ab") in (None, False), "WN2 input unexpectedly enables GDN screen")
    require(input_data.get("prompt") and len(input_data["prompt"]) == 64 and input_data.get("output_tokens") == 128 and input_data.get("context") == 4096 and input_data.get("cut") == 20 and input_data.get("forced") == [], "screen input is not the pinned 64/128 profile")
    fixture_dir = Path(input_data["expert_fixture"])
    manifest_path = fixture_dir / "manifest.json"
    manifest_sha = sha256_file(manifest_path)
    require(manifest_sha == digest(input_data["expert_fixture_sha256"]) == FIXTURE_MANIFEST_SHA256, "expert fixture manifest differs from pinned fixture SHA256")
    manifest = read_json(manifest_path)
    require(len(manifest.get("expert_order", [])) == 256 and sorted(manifest["expert_order"]) == list(range(256)), "fixture manifest does not contain 256 unique experts")
    experiments = receipt.get("experiments", [])
    require(experiments and all(e.get("exit_code") == 0 and e.get("completion", {}).get("kind") == "complete" for e in experiments), "controller records experiment error or incomplete completion")
    run = next((x for x in experiments if x.get("name") == "profile"), None)
    command = run.get("command", []) if run else []
    expected_command = [Path(match["executable"]), Path(plan.get("model_dir", "")), Path(match["input"]), window.resolve() / "profile-results"]
    require(run is not None and run.get("passed") is True and len(command) == len(expected_command) and all(Path(actual) == expected for actual, expected in zip(command, expected_command)) and run.get("outputs") == ["profile-results\\results.jsonl"], "controller experiment does not match the pinned command and journal")
    require(rows[-1].get("kind") == "complete" and rows[-1].get("correctness_pass") is True, "journal lacks passing final completion")
    require(not any(row.get("kind") == "failure" for row in rows), "journal contains failure row")
    return {"source_commit": marker, "executable_sha256": match["sha256"], "input_sha256": match["input_sha256"], "expert_fixture_manifest_sha256": manifest_sha, "controller_minimum_free_mib": {u: min(s["free_mib"][u] for s in memory) for u in UUIDS}, "restored": True, "clef_unchanged": True}


def validate_rows(rows):
    require(len(rows) == 190, "expected exactly 190 WN2 screen rows")
    allowed = {"start", "complete", "expert_wn2_fixture", "expert_wn2_plan", "expert_wn2_bank_parity", "expert_wn2_kernel", "expert_wn2_sample", "expert_wn2_projection_complete"}
    require(all(row.get("kind") in allowed for row in rows), "unknown WN2 journal kind")
    require(rows[0].get("kind") == "start" and rows[-1].get("kind") == "complete", "journal boundary rows are missing")
    require(sum(row["kind"] == "start" for row in rows) == sum(row["kind"] == "complete" for row in rows) == 1, "journal boundary row count mismatch")
    require(all(row["kind"] in {"start", "complete"} or row.get("ordinal") in (0, 1) for row in rows), "journal contains an unknown GPU ordinal")
    require(rows[0].get("source_commit") == SOURCE_COMMIT and rows[-1].get("correctness_pass") is True, "journal source or completion failed")
    require(rows[0].get("implementation") == "shared-qwen-engine" and rows[0].get("scope") == "isolated expert and GDN operator screens; no model load or model throughput claim", "journal screen identity or scope mismatch")
    require(rows[-1].get("scope") == "isolated operator parity and timing screens" and rows[-1].get("production_qualification") is False, "journal completion claims production qualification")
    require(rows[0].get("prompt_tokens") == 64 and rows[0].get("output_tokens") == 128 and rows[0].get("context") == 4096 and rows[0].get("cut") == 20, "journal profile differs from pinned input")
    fixtures = [r for r in rows if r["kind"] == "expert_wn2_fixture"]
    require(len(fixtures) == 2 and {r.get("ordinal") for r in fixtures} == {0, 1}, "expected one fixture record per GPU")
    fixture_sha = digest(fixtures[0].get("manifest_sha256"))
    for row in fixtures:
        require(digest(row.get("manifest_sha256")) == fixture_sha, "fixture manifest differs by GPU")
        order = row.get("expert_order")
        require(isinstance(order, list) and len(order) == 256 and sorted(order) == list(range(256)), "invalid expert fixture order")
        require(row.get("bank0_original_ids") == BANK0_EXPERTS and order[:8] == BANK0_EXPERTS and row.get("bank0_remapped_ids") == list(range(8)), "invalid bank-zero fixture control")
    expected_bytes_per_launch = LOGICAL_BYTES_PER_LAUNCH
    expected_unique_bytes = UNIQUE_WEIGHT_AND_SCALE_BYTES
    measurements = []
    savings = {}
    for ordinal in (0, 1):
        dev = [r for r in rows[1:-1] if r.get("ordinal") == ordinal]
        require(sum(r["kind"] == "expert_wn2_fixture" for r in dev) == 1, "GPU fixture roster mismatch")
        plans = [r for r in dev if r["kind"] == "expert_wn2_plan"]
        require(len(plans) == 1 and plans[0].get("epi") == "up" and plans[0].get("actual_items") == plans[0].get("bound") == 8 and plans[0].get("picks") == list(range(8)) and plans[0].get("banks") == 32 and plans[0].get("limit") == 0 and plans[0].get("relu2") is False, "WN2 up plan mismatch")
        done = [r for r in dev if r["kind"] == "expert_wn2_projection_complete"]
        require(len(done) == 1 and done[0].get("epi") == "up" and all(done[0].get(k) is True for k in ("all_bank_prechecks", "saved_bank0_exact", "graph_finals_exact")) and done[0].get("timed_pairs_per_candidate") == 5, "WN2 projection incomplete")
        kernels = [r for r in dev if r["kind"] == "expert_wn2_kernel"]
        require(len(kernels) == 3 and {r.get("variant") for r in kernels} == {"wn4_baseline", "wn4_n16_pipe4", "wn2_n16_pipe4"}, "WN2 kernel roster mismatch")
        for kernel in kernels:
            variant = kernel["variant"]
            expected = (4, 128, 32) if variant == "wn4_baseline" else (4, 128, 64) if variant == "wn4_n16_pipe4" else (2, 64, 128)
            require(kernel.get("epi") == "up" and (kernel.get("warps"), kernel.get("block"), kernel.get("grid")) == expected, "WN2 kernel geometry mismatch")
            require(kernel.get("kernel_launches") == kernel.get("verified_kernel_nodes") == 128 and kernel.get("banks") == 32, "WN2 graph launch or bank count mismatch")
            require(kernel.get("logical_weight_and_scale_bytes_per_launch") == expected_bytes_per_launch and kernel.get("logical_unique_weight_and_scale_bytes") == expected_unique_bytes, "WN2 expert byte accounting mismatch")
            require(type(kernel.get("regs")) is int and kernel["regs"] > 0 and type(kernel.get("local_bytes")) is int and kernel["local_bytes"] == 0, "WN2 register or local-memory report invalid")
            if variant == "wn2_n16_pipe4":
                require(kernel["regs"] == 162, "WN2 candidate register count differs from pinned kernel")
            require(type(kernel.get("resident_blocks")) is int and kernel["resident_blocks"] > 0 and type(kernel.get("resident_blocks_per_sm")) is int and kernel["resident_blocks_per_sm"] > 0, "WN2 occupancy inputs invalid")
            expected_symbol = "tf_nvfp4_wn2_n16_pipe4_swiglu" if variant == "wn2_n16_pipe4" else "tf_nvfp4_n16_pipe4_swiglu" if variant == "wn4_n16_pipe4" else "_ZN16tf_nvfp4_experts19nvfp4_expert_kernelILi2ELi2ELi4EEEvPK13__nv_bfloat16iiPK5uint4PKfiiPKiSA_SA_Pvifi"
            require(kernel.get("symbol") == expected_symbol, "WN2 function symbol mismatch")
        parity = [r for r in dev if r["kind"] == "expert_wn2_bank_parity"]
        require(len(parity) == 64 and {r.get("variant") for r in parity} == {"wn4_n16_pipe4", "wn2_n16_pipe4"}, "WN2 bank parity roster mismatch")
        for variant in ("wn4_n16_pipe4", "wn2_n16_pipe4"):
            subset = [r for r in parity if r.get("variant") == variant]
            require(sorted(r.get("bank") for r in subset) == list(range(32)), "WN2 bank parity indices incomplete")
            require(all(r.get("epi") == "up" and r.get("warps") == (4 if variant == "wn4_n16_pipe4" else 2) and r.get("finite") is True and r.get("bit_exact") is True and r.get("saved_real_output") is (r.get("bank") == 0) for r in subset), "WN2 exact parity failed")
        samples = [r for r in dev if r["kind"] == "expert_wn2_sample"]
        require(len(samples) == 24 and {r.get("comparison") for r in samples} == COMPARISONS, "WN2 sample roster mismatch")
        for candidate in ("wn4_n16_pipe4", "wn2_n16_pipe4"):
            candidate_rows = [r for r in samples if r.get("comparison") == candidate]
            require(len(candidate_rows) == 12, "WN2 comparison sample count mismatch")
            arm_times = {"wn4_baseline": [], candidate: []}
            paired = []
            for pair in range(6):
                both = sorted((r for r in candidate_rows if r.get("pair") == pair), key=lambda r: r.get("order", -1))
                expected_order = ["wn4_baseline", candidate] if pair % 2 == 0 else [candidate, "wn4_baseline"]
                require(len(both) == 2 and [r.get("arm") for r in both] == expected_order and [r.get("order") for r in both] == [0, 1], "WN2 pair order mismatch")
                times = {}
                for row in both:
                    require(row.get("epi") == "up" and row.get("warmup") is (pair == 0) and row.get("kernel_launches") == 128 and row.get("banks") == 32 and row.get("final_bank") == 31 and row.get("finite") is True and row.get("bit_exact_final") is True and finite_time(row.get("elapsed_ms")), "invalid WN2 timing sample")
                    times[row["arm"]] = row["elapsed_ms"] * 1000 / 128
                if pair:
                    for arm, value in times.items():
                        arm_times[arm].append(value)
                    paired.append(times["wn4_baseline"] - times[candidate])
            spread = max(paired) - min(paired)
            advance = min(paired) > 0 and statistics.median(paired) > spread
            arm_summary = {a: {"median": statistics.median(v), "range": [min(v), max(v)], "samples": v, "logical_bytes_per_kernel_launch": expected_bytes_per_launch, "logical_effective_bandwidth_gb_s": expected_bytes_per_launch / (statistics.median(v) * 1000)} for a, v in arm_times.items()}
            result = {"ordinal": ordinal, "epi": "up", "candidate": candidate, "comparison": "paired against WN4 baseline within this screen", "arms_us_per_call": arm_summary, "paired_savings_us": paired, "paired_savings_range_us": [min(paired), max(paired)], "paired_savings_spread_us": spread, "median_saving_us": statistics.median(paired), "all_pairs_faster": all(x > 0 for x in paired), "noise_screen_pass": advance}
            savings[ordinal, candidate] = result
            measurements.append(result)
    return fixture_sha, measurements, savings


def analyze(window: Path, baseline_analysis: Path):
    rows = [json.loads(line) for line in (window / "profile-results" / "results.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    controller = validate_window(window, rows)
    fixture_sha, measurements, savings = validate_rows(rows)
    require(fixture_sha == controller["expert_fixture_manifest_sha256"], "journal fixture hash differs from controller-pinned fixture")
    baseline = read_json(baseline_analysis)
    require(baseline.get("source_commit") == BASELINE_SOURCE_COMMIT and baseline.get("production_qualification") is False, "baseline analysis source or qualification gate differs")
    require(baseline.get("candidate_accepted") is True and baseline.get("arms", {}).get("candidate", {}).get("turn_seconds", {}).get("median") == BASELINE_SECONDS, "baseline candidate or exact median does not match accepted GDN baseline")
    require(baseline.get("arms", {}).get("baseline", {}).get("turn_seconds", {}).get("median") == 1.8607677, "baseline reference median differs")
    projection = {}
    n16_range = {gpu: savings[gpu, "wn4_n16_pipe4"]["arms_us_per_call"]["wn4_n16_pipe4"]["range"] for gpu in (0, 1)}
    wn2_range = {gpu: savings[gpu, "wn2_n16_pipe4"]["arms_us_per_call"]["wn2_n16_pipe4"]["range"] for gpu in (0, 1)}
    beats_control = {gpu: wn2_range[gpu][1] < n16_range[gpu][0] for gpu in (0, 1)}
    for candidate in ("wn4_n16_pipe4", "wn2_n16_pipe4"):
        per_gpu = [savings[gpu, candidate] for gpu in (0, 1)]
        saving_us = sum(item["median_saving_us"] for item in per_gpu)
        seconds = 191 * 20 * saving_us / 1e6
        fraction = seconds / BASELINE_SECONDS
        projection[candidate] = {"estimated_seconds_saved_per_turn": seconds, "estimated_turn_fraction": fraction, "material_5pct_screen": fraction >= 0.05, "all_five_pairs_positive_on_both_gpus": all(item["all_pairs_faster"] for item in per_gpu), "noise_screen_pass_on_both_gpus": all(item["noise_screen_pass"] for item in per_gpu), "beats_wn4_n16_control_beyond_range_variation": all(beats_control.values()) if candidate == "wn2_n16_pipe4" else None, "advance_screen_pass": candidate == "wn2_n16_pipe4" and fraction >= 0.05 and all(item["all_pairs_faster"] and item["noise_screen_pass"] for item in per_gpu) and all(beats_control.values()), "formula": "191 appends x 20 layers x sum of per-GPU median paired microsecond savings / 1,000,000; compared with the accepted 1.7232142 s WN4 GDN turn median"}
    n16 = [savings[gpu, "wn4_n16_pipe4"]["median_saving_us"] for gpu in (0, 1)]
    wn2 = [savings[gpu, "wn2_n16_pipe4"]["median_saving_us"] for gpu in (0, 1)]
    independent_delta = [a - b for a, b in zip(n16, wn2)]
    comparison = {"kind": "independent comparison of paired-median savings", "pairs_are_matched_across_candidates": False, "meaning": "WN2 paired-median saving versus WN4 baseline is compared with the separately measured WN4 N16 paired-median saving; per-GPU deltas are not pairwise matched", "per_gpu_wn4_n16_pipe4_median_saving_us": n16, "per_gpu_wn2_n16_pipe4_median_saving_us": wn2, "per_gpu_delta_n16_minus_wn2_us": independent_delta, "delta_range_us": [min(independent_delta), max(independent_delta)], "median_delta_us": statistics.median(independent_delta), "per_gpu_candidate_call_ranges_us": {str(gpu): {"wn4_n16_pipe4": n16_range[gpu], "wn2_n16_pipe4": wn2_range[gpu], "wn2_beats_control_beyond_range_variation": beats_control[gpu]} for gpu in (0, 1)}, "conservative_range_gate_pass": all(beats_control.values())}
    resources = [{"ordinal": gpu, "variant": name, "registers_per_thread": next(r["regs"] for r in rows if r.get("kind") == "expert_wn2_kernel" and r.get("ordinal") == gpu and r.get("variant") == name), "local_bytes": 0, "resident_blocks": next(r["resident_blocks"] for r in rows if r.get("kind") == "expert_wn2_kernel" and r.get("ordinal") == gpu and r.get("variant") == name), "driver_calculated_max_active_blocks_per_sm": next(r["resident_blocks_per_sm"] for r in rows if r.get("kind") == "expert_wn2_kernel" and r.get("ordinal") == gpu and r.get("variant") == name), "achieved_occupancy_measured": False} for gpu in (0, 1) for name in ("wn4_baseline", "wn4_n16_pipe4", "wn2_n16_pipe4")]
    return {"scope": SCOPE, "source_commit": SOURCE_COMMIT, "controller": controller, "fixture_manifest_sha256": fixture_sha, "correctness_pass": True, "measurements": measurements, "projection": projection, "wn2_vs_known_wn4_n16_control": comparison, "payload_accounting": {"weight_bytes_per_expert": ONE_EXPERT_WEIGHT_BYTES, "global_scale_bytes_per_expert": 8, "experts_per_kernel_launch": 8, "logical_bytes_per_kernel_launch": LOGICAL_BYTES_PER_LAUNCH, "unique_experts_per_screen": 256, "unique_weight_and_scale_bytes": UNIQUE_WEIGHT_AND_SCALE_BYTES, "bandwidth_label": "logical effective bandwidth from declared bytes; not measured physical DRAM bandwidth"}, "kernel_resources": resources, "occupancy_note": "The CUDA driver's resident-block calculation is reported as a resource bound. Achieved runtime occupancy was not measured.", "production_qualification": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("window", type=Path)
    parser.add_argument("--baseline-analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.window, args.baseline_analysis)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(result["projection"], indent=2))


if __name__ == "__main__":
    main()
