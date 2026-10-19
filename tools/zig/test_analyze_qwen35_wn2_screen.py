import hashlib
import json

import analyze_qwen35_wn2_screen as analyzer
import pytest


def rows_for(manifest_sha):
    rows = [{"kind": "start", "source_commit": analyzer.SOURCE_COMMIT, "implementation": "shared-qwen-engine", "scope": "isolated expert and GDN operator screens; no model load or model throughput claim", "identities": [{"uuid": u} for u in analyzer.UUIDS], "prompt_tokens": 64, "output_tokens": 128, "context": 4096, "cut": 20}]
    expert_order = analyzer.BANK0_EXPERTS + [i for i in range(256) if i not in analyzer.BANK0_EXPERTS]
    for ordinal in (0, 1):
        rows.append({"kind": "expert_wn2_fixture", "ordinal": ordinal, "manifest_sha256": manifest_sha, "expert_order": expert_order, "bank0_original_ids": expert_order[:8], "bank0_remapped_ids": list(range(8)), "scope": analyzer.FIXTURE_SCOPE})
        rows.append({"kind": "expert_wn2_plan", "ordinal": ordinal, "epi": "up", "actual_items": 8, "bound": 8, "picks": list(range(8)), "banks": 32, "limit": 0, "relu2": False})
        kernels = [("wn4_baseline", 4, 32, "_ZN16tf_nvfp4_experts19nvfp4_expert_kernelILi2ELi2ELi4EEEvPK13__nv_bfloat16iiPK5uint4PKfiiPKiSA_SA_Pvifi"), ("wn4_n16_pipe4", 4, 64, "tf_nvfp4_n16_pipe4_swiglu"), ("wn2_n16_pipe4", 2, 128, "tf_nvfp4_wn2_n16_pipe4_swiglu")]
        for name, warps, grid, symbol in kernels:
            rows.append({"kind": "expert_wn2_kernel", "ordinal": ordinal, "epi": "up", "variant": name, "warps": warps, "grid": grid, "block": warps * 32, "symbol": symbol, "regs": 162 if name == "wn2_n16_pipe4" else 64, "local_bytes": 0, "resident_blocks": 2, "resident_blocks_per_sm": 2, "kernel_launches": 128, "verified_kernel_nodes": 128, "banks": 32, "logical_weight_and_scale_bytes_per_launch": 8 * (analyzer.ONE_EXPERT_WEIGHT_BYTES + 8), "logical_unique_weight_and_scale_bytes": 256 * analyzer.ONE_EXPERT_WEIGHT_BYTES + 256 * 8})
        for variant, warps in (("wn4_n16_pipe4", 4), ("wn2_n16_pipe4", 2)):
            for bank in range(32):
                rows.append({"kind": "expert_wn2_bank_parity", "ordinal": ordinal, "epi": "up", "variant": variant, "bank": bank, "warps": warps, "finite": True, "bit_exact": True, "saved_real_output": bank == 0})
        for comparison, candidate in (("wn4_n16_pipe4", "wn4_n16_pipe4"), ("wn2_n16_pipe4", "wn2_n16_pipe4")):
            for pair in range(6):
                arms = ["wn4_baseline", candidate] if pair % 2 == 0 else [candidate, "wn4_baseline"]
                for order, arm in enumerate(arms):
                    base = 50.0 + ordinal * 0.1 + pair * 0.01
                    saving = 0 if arm == "wn4_baseline" else 15 if candidate == "wn4_n16_pipe4" else 35
                    elapsed = (base - saving) * 128 / 1000
                    rows.append({"kind": "expert_wn2_sample", "ordinal": ordinal, "epi": "up", "comparison": comparison, "pair": pair, "warmup": pair == 0, "order": order, "arm": arm, "elapsed_ms": elapsed, "kernel_launches": 128, "banks": 32, "final_bank": 31, "finite": True, "bit_exact_final": True})
        rows.append({"kind": "expert_wn2_projection_complete", "ordinal": ordinal, "epi": "up", "all_bank_prechecks": True, "saved_bank0_exact": True, "graph_finals_exact": True, "timed_pairs_per_candidate": 5})
    rows.append({"kind": "complete", "correctness_pass": True, "scope": "isolated operator parity and timing screens", "production_qualification": False})
    return rows


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    window = tmp_path / "wn2-window"
    result_dir = window / "profile-results"
    result_dir.mkdir(parents=True)
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    manifest = json.dumps({"scope": analyzer.FIXTURE_SCOPE, "expert_order": analyzer.BANK0_EXPERTS + [i for i in range(256) if i not in analyzer.BANK0_EXPERTS]}, separators=(",", ":")).encode() + b"\n"
    (fixture / "manifest.json").write_bytes(manifest)
    manifest_sha = hashlib.sha256(manifest).hexdigest()
    monkeypatch.setattr(analyzer, "FIXTURE_MANIFEST_SHA256", manifest_sha)
    input_obj = {"prompt": list(range(64)), "output_tokens": 128, "context": 4096, "cut": 20, "forced": [], "gdn_layout_screen": False, "expert_wn2_screen": True, "expert_fixture": str(fixture), "expert_fixture_sha256": manifest_sha}
    input_bytes = (json.dumps(input_obj) + "\n").encode()
    input_path = window / "profile-input.json"
    input_path.write_bytes(input_bytes)
    rows = rows_for(manifest_sha)
    (result_dir / "results.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    (window / "profile-SOURCE_COMMIT").write_text(analyzer.SOURCE_COMMIT + "\n", encoding="utf-8")
    exe = tmp_path / "screen.exe"
    exe.write_bytes(b"screen executable")
    experiment = {"name": "profile", "executable": str(exe), "sha256": hashlib.sha256(exe.read_bytes()).hexdigest(), "input": "pinned-input.json", "input_sha256": hashlib.sha256(input_bytes).hexdigest()}
    plan = {"source_commit": analyzer.SOURCE_COMMIT, "model_dir": str(tmp_path / "model"), "experiments": [experiment]}
    (window / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    receipt = {"passed": True, "restored": True, "clef_unchanged": True, "cleanup_errors": [], "sampler_failures": [], "memory": [{"free_mib": {u: 4000 for u in analyzer.UUIDS}}], "plan": plan, "restoration": {"passed": True, "model": "nemotron-3.5-lightning", "content": "READY", "free_mib": {u: 4000 for u in analyzer.UUIDS}}, "experiments": [{"name": "profile", "passed": True, "exit_code": 0, "completion": {"kind": "complete"}, "command": [str(exe), plan["model_dir"], experiment["input"], str(result_dir)], "outputs": ["profile-results\\results.jsonl"]}]}
    (window / "window.json").write_text(json.dumps(receipt), encoding="utf-8")
    baseline = tmp_path / "baseline-analysis.json"
    baseline.write_text(json.dumps({"source_commit": analyzer.BASELINE_SOURCE_COMMIT, "candidate_accepted": True, "production_qualification": False, "arms": {"baseline": {"turn_seconds": {"median": 1.8607677}}, "candidate": {"turn_seconds": {"median": analyzer.BASELINE_SECONDS}}}}), encoding="utf-8")
    return window, baseline


def test_validates_window_and_projects_only_an_up_seam(evidence):
    window, baseline = evidence
    report = analyzer.analyze(window, baseline)
    assert len(report["measurements"]) == 4
    assert report["projection"]["wn2_n16_pipe4"]["advance_screen_pass"]
    assert report["wn2_vs_known_wn4_n16_control"]["pairs_are_matched_across_candidates"] is False
    assert report["payload_accounting"]["logical_bytes_per_kernel_launch"] == 8 * (analyzer.ONE_EXPERT_WEIGHT_BYTES + 8)
    assert report["kernel_resources"][-1]["achieved_occupancy_measured"] is False
    assert report["production_qualification"] is False


def test_candidate_must_beat_known_control_beyond_sample_range(evidence):
    window, baseline = evidence
    path = window / "profile-results" / "results.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    for row in rows:
        if row.get("kind") == "expert_wn2_sample" and row.get("comparison") == "wn2_n16_pipe4" and row.get("arm") == "wn2_n16_pipe4":
            row["elapsed_ms"] = 38 * 128 / 1000
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    report = analyzer.analyze(window, baseline)
    assert report["projection"]["wn2_n16_pipe4"]["material_5pct_screen"]
    assert not report["projection"]["wn2_n16_pipe4"]["beats_wn4_n16_control_beyond_range_variation"]
    assert not report["projection"]["wn2_n16_pipe4"]["advance_screen_pass"]


@pytest.mark.parametrize("kind,key,value", [
    ("expert_wn2_kernel", "grid", 127),
    ("expert_wn2_kernel", "block", 32),
    ("expert_wn2_kernel", "logical_weight_and_scale_bytes_per_launch", 1),
    ("expert_wn2_kernel", "logical_unique_weight_and_scale_bytes", 1),
    ("expert_wn2_kernel", "local_bytes", 4),
    ("expert_wn2_kernel", "symbol", "wrong"),
    ("expert_wn2_kernel", "verified_kernel_nodes", 127),
    ("expert_wn2_sample", "elapsed_ms", float("nan")),
    ("expert_wn2_sample", "order", 2),
    ("expert_wn2_bank_parity", "bit_exact", False),
    ("expert_wn2_plan", "actual_items", 7),
    ("expert_wn2_fixture", "manifest_sha256", "0" * 64),
])
def test_rejects_journal_mutants(evidence, kind, key, value):
    window, baseline = evidence
    path = window / "profile-results" / "results.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    next(row for row in rows if row["kind"] == kind)[key] = value
    path.write_text("\n".join(json.dumps(row, allow_nan=True) for row in rows) + "\n", encoding="utf-8")
    with pytest.raises((ValueError, TypeError)):
        analyzer.analyze(window, baseline)


@pytest.mark.parametrize("mutation", ["restoration", "source", "memory", "input_hash", "input_flag", "gdn_flag", "baseline_median", "baseline_acceptance", "unknown", "remove_pair", "extra_ordinal"])
def test_rejects_window_and_baseline_mutants(evidence, mutation):
    window, baseline = evidence
    if mutation in {"restoration", "memory"}:
        path = window / "window.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        if mutation == "restoration":
            data["restored"] = False
        else:
            data["memory"][0]["free_mib"][analyzer.UUIDS[0]] = 1000
        path.write_text(json.dumps(data), encoding="utf-8")
    elif mutation in {"input_hash", "input_flag", "gdn_flag"}:
        path = window / "profile-input.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        if mutation == "input_hash":
            data["expert_fixture_sha256"] = "0" * 64
        elif mutation == "input_flag":
            data["expert_wn2_screen"] = False
        else:
            data["gdn_layout_screen"] = True
        path.write_text(json.dumps(data) + "\n", encoding="utf-8")
        plan = json.loads((window / "plan.json").read_text(encoding="utf-8"))
        plan["experiments"][0]["input_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        (window / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
        receipt = json.loads((window / "window.json").read_text(encoding="utf-8"))
        receipt["plan"] = plan
        receipt["experiments"][0]["input_sha256"] = plan["experiments"][0]["input_sha256"]
        (window / "window.json").write_text(json.dumps(receipt), encoding="utf-8")
    elif mutation in {"baseline_median", "baseline_acceptance"}:
        data = json.loads(baseline.read_text(encoding="utf-8"))
        if mutation == "baseline_median":
            data["arms"]["candidate"]["turn_seconds"]["median"] = 1.7
        else:
            data["candidate_accepted"] = False
        baseline.write_text(json.dumps(data), encoding="utf-8")
    else:
        path = window / "profile-results" / "results.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        if mutation == "source":
            (window / "profile-SOURCE_COMMIT").write_text("0" * 40)
        elif mutation == "unknown":
            rows.insert(-1, {"kind": "surprise"})
        elif mutation == "remove_pair":
            rows.remove(next(row for row in rows if row["kind"] == "expert_wn2_sample" and row["pair"] == 5))
        else:
            extra = dict(next(row for row in rows if row["kind"] == "expert_wn2_sample"))
            extra["ordinal"] = 2
            rows.insert(-1, extra)
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    with pytest.raises((ValueError, TypeError, KeyError)):
        analyzer.analyze(window, baseline)
