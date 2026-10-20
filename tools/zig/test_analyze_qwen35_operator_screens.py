import hashlib
import json

import analyze_qwen35_operator_screens as analyzer
import pytest


def screen_rows(manifest_sha):
    rows = [{"kind": "start", "source_commit": "a" * 40, "implementation": "shared-qwen-engine", "scope": "isolated expert and GDN operator screens; no model load or model throughput claim", "identities": [{"uuid": u} for u in analyzer.UUIDS], "prompt_tokens": 64, "output_tokens": 128, "context": 4096, "cut": 20}]
    for ordinal in (0, 1):
        order = [112, 106, 107, 238, 181, 57, 43, 200] + [i for i in range(256) if i not in {112, 106, 107, 238, 181, 57, 43, 200}]
        rows.append({"kind": "expert_fixture", "ordinal": ordinal, "manifest_sha256": manifest_sha, "expert_order": order, "bank0_original_ids": order[:8], "bank0_remapped_ids": list(range(8)), "scope": "fixture"})
        for epi in ("up", "down_f32"):
            rows.append({"kind": "expert_plan", "ordinal": ordinal, "epi": epi, "actual_items": 8, "bound": 8, "picks": list(range(8)), "banks": 32, "limit": 0, "relu2": False})
            variants = [("wn4_baseline", 4, 32 if epi == "up" else 128), ("wn4_n16_pipe4", 4, 64 if epi == "up" else 256), ("wn1_baseline", 1, 128 if epi == "up" else 512)]
            for name, warps, grid in variants:
                baseline_symbol = "_ZN16tf_nvfp4_experts19nvfp4_expert_kernelILi2ELi2ELi4EEEvPK13__nv_bfloat16iiPK5uint4PKfiiPKiSA_SA_Pvifi" if epi == "up" else "_ZN16tf_nvfp4_experts19nvfp4_expert_kernelILi1ELi0ELi4EEEvPK13__nv_bfloat16iiPK5uint4PKfiiPKiSA_SA_Pvifi"
                symbol = baseline_symbol if name == "wn4_baseline" else baseline_symbol.replace("ELi4EE", "ELi1EE") if name == "wn1_baseline" else "tf_nvfp4_n16_pipe4_swiglu" if epi == "up" else "tf_nvfp4_n16_pipe4_down_f32"
                parts = 2 if epi == "up" else 1
                rows.append({"kind": "expert_kernel", "ordinal": ordinal, "epi": epi, "variant": name, "warps": warps, "symbol": symbol, "grid": grid, "block": 32 * warps, "regs": 32, "local_bytes": 0, "resident_blocks": 2, "resident_blocks_per_sm": 2, "kernel_launches": 128, "verified_kernel_nodes": 128, "banks": 32, "logical_weight_and_scale_bytes_per_launch": 8 * (analyzer.EXPERT_ONE_BYTES[epi] + parts * 4), "logical_unique_weight_and_scale_bytes": 256 * analyzer.EXPERT_ONE_BYTES[epi] + 256 * parts * 4})
            for candidate in ("wn4_n16_pipe4", "wn1_baseline"):
                for bank in range(32):
                    rows.append({"kind": "expert_bank_parity", "ordinal": ordinal, "epi": epi, "bank": bank, "variant": candidate, "warps": 4 if candidate == "wn4_n16_pipe4" else 1, "finite": True, "bit_exact": True, "saved_real_output": bank == 0})
                for pair in range(6):
                    order = ["wn4_baseline", candidate] if pair % 2 == 0 else [candidate, "wn4_baseline"]
                    for i, arm in enumerate(order):
                        rows.append({"kind": "expert_sample", "ordinal": ordinal, "epi": epi, "comparison": candidate, "pair": pair, "warmup": pair == 0, "order": i, "arm": arm, "elapsed_ms": 1.0 if arm == candidate else 1.2, "kernel_launches": 128, "banks": 32, "final_bank": 31, "finite": True, "bit_exact_final": True})
            rows.append({"kind": "expert_projection_complete", "ordinal": ordinal, "epi": epi, "all_bank_prechecks": True, "saved_bank0_exact": True, "graph_finals_exact": True, "timed_pairs_per_candidate": 5})
        rows.extend([
            {"kind": "gdn_layout_fixture", "ordinal": ordinal, "seed": 0x35D019, "pattern_version": 3, "baseline_state_layout": "head,value,key", "candidate_state_layout": "head,key,value", "beta_dtype": "BF16", "value_heads": 32, "key_heads": 16, "dimension": 128, "state_bytes": 2097152, "out_bytes": 16384, "q_dtype": "FP32", "k_dtype": "FP32", "v_dtype": "BF16", "state_dtype": "FP32", "out_dtype": "FP32", "q_elements": 2048, "k_elements": 2048, "v_elements": 4096, "decay_elements": 32, "beta_elements": 32, "initial_state_sha256": "1" * 64, "free_bytes": analyzer.FLOOR, "floor_bytes": analyzer.FLOOR, "scope": analyzer.GDN_SCOPE},
            {"kind": "gdn_layout_negative_control", "ordinal": ordinal, "poisoned_output_rejected": True, "changed_state_rejected": True, "changed_output_rejected": True, "late_bank_state_mismatch_rejected": True, "late_bank_output_mismatch_rejected": True},
            {"kind": "gdn_layout_unwritten_state_control", "ordinal": ordinal, "unchanged_initial_state_with_correct_output_rejected": True},
        ])
        for step in range(16):
            rows.append({"kind": "gdn_layout_step_parity", "ordinal": ordinal, "step": step, "inputs_sha256": f"{step:064x}", "state_sha256": f"{step+32:064x}", "output_sha256": f"{step+64:064x}", "state_bit_exact": True, "output_bit_exact": True, "finite_state": True, "finite_output": True, "unwritten_initial_state_rejected": True, "free_bytes": analyzer.FLOOR})
        expected = {"inputs_sha256": "0" * 64, "expected_state_sha256": "5" * 64, "expected_output_sha256": "6" * 64, "expected_payload_sha256": "7" * 64, "initial_payload_sha256": "8" * 64}
        for arm, kernel in (("old", "tf_q35_hf_gdn_chain"), ("new", "tf_q35_hf_gdn_chain_transposed")):
            rows.append({"kind": "gdn_layout_graph", "ordinal": ordinal, "arm": arm, "kernel": kernel, "node_count": 128, "kernel_launches": 128, "verified_kernel_nodes": 128, "verified_updates_per_bank": 2, "local_delta_layer": 0, "grid_x": 1, "grid_z": 32, "block_x": 128, "num_regs": 32, "shared_bytes": 0, "local_bytes": 0, **expected, "banks": 64, "unique_state_bytes": 134217728, "bank_bytes": 2113536, "payload_bytes": 135266304, "updates_per_bank": 2, "scope": analyzer.GDN_SCOPE, "evolving_state": True, "free_bytes": analyzer.FLOOR})
        for pair in range(6):
            order = ["old", "new"] if pair % 2 == 0 else ["new", "old"]
            for i, arm in enumerate(order):
                rows.append({"kind": "gdn_layout_sample", "ordinal": ordinal, "pair": pair, "order": i, "arm": arm, "warmup": pair == 0, "elapsed_ms": 1.0 if arm == "new" else 1.2, "kernel_launches": 128, "node_count": 128, "graph_launches": 1, "state_bit_exact": True, "output_bit_exact": True, "finite_state": True, "finite_output": True, "state_sha256": expected["expected_state_sha256"], "output_sha256": expected["expected_output_sha256"], "payload_sha256": expected["expected_payload_sha256"], "initial_payload_sha256": expected["initial_payload_sha256"], "initial_state_sha256": "1" * 64, "banks": 64, "unique_state_bytes": 134217728, "bank_bytes": 2113536, "payload_bytes": 135266304, "updates_per_bank": 2, "reset_before_event": True, "free_bytes_before": analyzer.FLOOR, "free_bytes": analyzer.FLOOR, "floor_bytes": analyzer.FLOOR, "scope": analyzer.GDN_SCOPE})
        rows.append({"kind": "gdn_layout_screen_complete", "ordinal": ordinal, "correctness_steps": 16, "banks": 64, "unique_state_bytes": 134217728, "updates_per_bank": 2, "measured_pairs": 5, "excluded_warmup_pairs": 1, "correctness_pass": True, "scope": analyzer.GDN_SCOPE, "production_qualification": False})
    rows.append({"kind": "complete", "correctness_pass": True, "scope": "isolated operator parity and timing screens", "production_qualification": False})
    return rows


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    window = tmp_path / "window"
    (window / "profile-results").mkdir(parents=True)
    fixture_dir = tmp_path / "fixture"
    fixture_dir.mkdir()
    manifest_bytes = b'{"scope":"test"}\n'
    (fixture_dir / "manifest.json").write_bytes(manifest_bytes)
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    screen_input = {"prompt": list(range(64)), "token": 17, "repetitions": 16, "output_tokens": 128, "forced": [], "context": 4096, "cut": 20, "expert_fixture": str(fixture_dir), "expert_fixture_sha256": manifest_sha, "gdn_layout_screen": True}
    input_bytes = (json.dumps(screen_input) + "\n").encode()
    (window / "profile-input.json").write_bytes(input_bytes)
    input_sha = hashlib.sha256(input_bytes).hexdigest()
    rows = screen_rows(manifest_sha)
    (window / "profile-results/results.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (window / "profile-SOURCE_COMMIT").write_text("a" * 40)
    exe = tmp_path / "screen.exe"
    exe.write_bytes(b"offline synthetic executable")
    experiment_plan = {"name": "profile", "executable": str(exe), "sha256": hashlib.sha256(exe.read_bytes()).hexdigest(), "input": str(tmp_path / "screens-input.json"), "input_sha256": input_sha}
    plan = {"source_commit": "a" * 40, "experiments": [experiment_plan]}
    (window / "plan.json").write_text(json.dumps(plan))
    receipt = {"passed": True, "restored": True, "clef_unchanged": True, "cleanup_errors": [], "sampler_failures": [], "memory": [{"free_mib": {u: 4000 for u in analyzer.UUIDS}}], "plan": plan, "restoration": {"passed": True, "clef_unchanged": True}, "experiments": [{"name": "profile", "passed": True, "exit_code": 0, "completion": {"kind": "complete"}, "command": [str(exe), "model", experiment_plan["input"], str(window / "profile-results")], "outputs": ["profile-results\\results.jsonl"]}]}
    (window / "window.json").write_text(json.dumps(receipt))
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"source_commit": "8a39ba55764aafad58d8dae402b287b21d0e760c", "input": {"prompt_tokens": 64, "output_tokens": 128, "greedy": True}, "arms": {"graph": {"summary": {"inclusive_turn_seconds": {"median": analyzer.BASELINE_SECONDS}}}}}))
    return window, baseline


def test_full_shape_and_separate_materiality_projection(evidence):
    result = analyzer.analyze(*evidence)
    assert len(result["expert"]["measurements"]) == 8
    assert result["gdn_layout"]["all_pairs_faster"]
    assert result["gdn_layout"]["estimated_seconds_saved_per_turn"] > 0
    assert result["production_qualification"] is False


@pytest.mark.parametrize("kind,key,value", [
    ("expert_bank_parity", "bank", 99), ("expert_sample", "warmup", False),
    ("expert_kernel", "symbol", "wrong"), ("gdn_layout_sample", "payload_sha256", "9" * 64),
    ("gdn_layout_sample", "reset_before_event", False), ("gdn_layout_sample", "free_bytes", analyzer.FLOOR - 1),
    ("gdn_layout_graph", "kernel", "wrong"),
])
def test_rejects_screen_mutants(evidence, kind, key, value):
    window, baseline = evidence
    path = window / "profile-results/results.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    next(r for r in rows if r["kind"] == kind)[key] = value
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    with pytest.raises(ValueError):
        analyzer.analyze(window, baseline)


def test_rejects_failed_restoration_and_wrong_source(evidence):
    window, baseline = evidence
    receipt_path = window / "window.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["restored"] = False
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="pass and restore"):
        analyzer.analyze(window, baseline)


@pytest.mark.parametrize("kind, selector", [
    ("expert_bank_parity", lambda row: row["bank"] == 31),
    ("expert_sample", lambda row: row["pair"] == 5 and row["order"] == 1),
])
def test_rejects_missing_bank_or_measured_pair(evidence, kind, selector):
    window, baseline = evidence
    path = window / "profile-results/results.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows.remove(next(r for r in rows if r["kind"] == kind and selector(r)))
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    with pytest.raises(ValueError):
        analyzer.analyze(window, baseline)


def test_rejects_extra_gpu_or_unknown_kind(evidence):
    window, baseline = evidence
    path = window / "profile-results/results.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    extra = dict(next(r for r in rows if r["kind"] == "gdn_layout_sample"))
    extra["ordinal"] = 2
    rows.insert(-1, extra)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    with pytest.raises(ValueError, match="unknown journal kind or GPU ordinal"):
        analyzer.analyze(window, baseline)
    rows = [json.loads(line) for line in path.read_text().splitlines()[:-1]]
    rows.append({"kind": "invented"})
    rows.append({"kind": "complete", "correctness_pass": True, "scope": "isolated operator parity and timing screens", "production_qualification": False})
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    with pytest.raises(ValueError, match="unknown journal kind"):
        analyzer.analyze(window, baseline)
