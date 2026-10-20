import hashlib
import json

import analyze_qwen35_quant_transport as a
import pytest


def h(text):
    return hashlib.sha256(text.encode()).hexdigest()


def summary(case, values, operations):
    ordered = sorted(values)
    return {"kind": "transport_summary", "case": case, "samples_ns": values, "median_ns": ordered[2], "min_ns": ordered[0], "max_ns": ordered[4], "operations": operations, "median_ns_per_operation": ordered[2] // operations}


def make_rows():
    src = "a" * 40
    rows = [{"kind": "start", "source_commit": src, "implementation": "shared-qwen-engine", "scope": a.SOURCE_SCOPE, "identities": [{"uuid": x} for x in a.UUIDS], "prompt_tokens": 64, "output_tokens": 128, "context": 4096, "cut": 20}]
    for ordinal in (0, 1):
        for case, (outputs, _) in a.QUANT_CASES.items():
            weight_bytes = sum(outputs) * 2048
            rows.append({"kind": "quant_reuse_fixture", "ordinal": ordinal, "case": case, "outputs": outputs, "width": 2048, "rows": 1, "input_scale": .125, "weight_scale_rule": "(projection+1)/64", "seed": 0x351019ad, "pattern_version": 1, "banks": 8, "weight_bytes_per_bank": weight_bytes, "unique_weight_bytes": 8 * weight_bytes, "l2_bytes": 1024, "groups": 128, "distinct_input_rows": 128, "payloads": 2, "output_bytes": 128 * sum(outputs) * 2, "pack_path": "modelopt.Packer.fp8Lane + modelopt.fp8Weight", "baseline_path": "qlinear.Linear.decode(.fp8)", "candidate_path": "Ck.quant8 once + same ordered Ck.lane8", "free_bytes": a.FLOOR, "scope": a.QUANT_SCOPE})
            for scope in a.QUANT_SCOPES:
                banks = 1 if scope == "hot_one_bank" else 8
                for arm in a.QUANT_ARMS:
                    projections = len(outputs)
                    qn = 128 * projections if arm == "baseline" else 128
                    gn = 128 * projections
                    rows.append({"kind": "quant_reuse_kernelwork", "ordinal": ordinal, "case": case, "cache_scope": scope, "arm": arm, "quant_symbol": "tf_ck_quant8", "quant_function_handle": ordinal + 1, "quant_nodes": qn, "gemm_nodes": gn, "total_nodes": qn + gn, "function_geometry_arguments_and_serial_dependencies_checked": True, "quantized_rows_per_node": 1, "quantized_columns_per_node": 2048, "groups": 128, "banks": banks})
                for payload in (0, 1):
                    digest = h(f"quant-{ordinal}-{case}-{scope}-{payload}")
                    rows.append({"kind": "quant_reuse_output_parity", "ordinal": ordinal, "case": case, "cache_scope": scope, "payload": payload, "all_output_bytes_compared": 128 * sum(outputs) * 2, "all_banks_checked": banks, "finite": True, "bit_exact": True, "output_sha256": digest, "input_sha256": h(f"input-{payload}-{ordinal}-{case}")})
                    rows.append({"kind": "quant_reuse_encoded_parity", "ordinal": ordinal, "case": case, "payload": payload, "banks": banks, "groups": 128, "bytes_compared_per_projection": 2048, "baseline_quantizations_checked": 128 * len(outputs), "shared_quantizations_checked": 128, "all_encoded_bytes_exact": True, "changing_rows_change_encoded_bytes": True, "unequal_scale_encoded_bytes_differ": True, "unequal_scale_eligibility_rejected": True, "independent_weight_scales": True})
                for pair in range(6):
                    arms = ["baseline", "shared_quant"] if pair % 2 == 0 else ["shared_quant", "baseline"]
                    for order, arm in enumerate(arms):
                        elapsed = (1.2 if arm == "baseline" else 1.0) + pair * .01
                        payload = pair % 2
                        rows.append({"kind": "quant_reuse_sample", "ordinal": ordinal, "case": case, "cache_scope": scope, "pair": pair, "order": order, "payload": payload, "arm": arm, "warmup": pair == 0, "elapsed_ms": elapsed, "wall_ns": 2000000 + pair * 20000, "us_per_group": elapsed * 1000 / 128, "wall_us_per_group": (2000000 + pair * 20000) / 1000 / 128, "graph_launches": 1, "groups": 128, "all_output_bytes_compared": 128 * sum(outputs) * 2, "bit_exact": True, "finite": True, "output_sha256": h(f"quant-{ordinal}-{case}-{scope}-{payload}"), "reset_outside_events": True, "free_bytes_before": a.FLOOR, "free_bytes": a.FLOOR, "scope": a.QUANT_SCOPE})
                savings = [((1.2 + pair * .01) - (1.0 + pair * .01)) * 1000 / 128 for pair in range(1, 6)]
                rows.append({"kind": "quant_reuse_case_complete", "ordinal": ordinal, "case": case, "cache_scope": scope, "paired_saving_us_per_group": savings, "wall_paired_saving_us_per_group": [0.0] * 5, "median_saving_us_per_group": sorted(savings)[2], "min_saving_us_per_group": min(savings), "max_saving_us_per_group": max(savings), "groups_per_append_on_this_partition": a.QUANT_CASES[case][1], "estimated_turn_saving_us_this_case_this_partition": 191 * a.QUANT_CASES[case][1] * sorted(savings)[2], "projection_formula": "191*(15*pair_gpu0+5*triple_gpu0+15*pair_gpu1+5*triple_gpu1) us", "accepted_turn_seconds": 1.7232142, "required_total_us_per_append": 451.1031937172775, "required_us_per_removed_quant": 9.02206387434555, "measured_pairs": 5, "excluded_warmup_pairs": 1, "correctness_pass": True, "production_qualification": False})
        rows.append({"kind": "quant_reuse_screen_complete", "ordinal": ordinal, "cases": 2, "cache_scopes": 2, "measured_pairs_per_case_scope": 5, "correctness_pass": True, "full_model_gate_pass": False, "production_qualification": False, "scope": a.QUANT_SCOPE})
    rows.append({"kind": "transport_manifest", "ordinals": [0, 1], "private_contexts": True, "peer_symbols_available": False, "peer_capability": [0, 0], "peer_enable_results": [-1, -1], "peer_available": False, "rank_sum_function": "tf_rank_sum", "rank_sum_bindings": [1, 2], "negative_control": "last output byte mutation rejected", "boundary_parts": 2, "boundary_part_bytes": 4096, "collectives_per_sample": 80, "sum_row_bytes_per_rank": 8192, "warmups": 1, "measured_samples": 5, "memory_floor_bytes": a.FLOOR, "free_bytes": [a.FLOOR, a.FLOOR], "scope": a.TRANSPORT_SCOPE})
    for direction in ("forward", "reverse", "roundtrip"):
        for sample in range(6):
            rows.append({"kind": "transport_boundary_sample", "method": "pinned_host", "direction": direction, "sample": sample, "warmup": sample == 0, "wall_ns": 32000 + sample * 32, "wall_ns_per_operation": 1000 + sample, "operations": 32, "payload_bytes_per_leg": 8192, "logical_copy_bytes_per_leg": 16384, "legs_per_operation": 2 if direction == "roundtrip" else 1, "all_bytes_exact": True, "input_sha256": h(f"boundary-{direction}-{sample}"), "output_sha256": h(f"boundary-{direction}-{sample}"), "free_before": [a.FLOOR] * 2, "free_after": [a.FLOOR] * 2})
        values = [32000 + s * 32 for s in range(1, 6)]
        rows.append(summary(f"pinned_boundary_{direction}", values, 32))
    for sample in range(6):
        order = a.METHODS[:2] if sample % 2 == 0 else tuple(reversed(a.METHODS[:2]))
        for method in order:
            rows.append({"kind": "transport_sum_sample", "method": method, "sample": sample, "warmup": sample == 0, "wall_ns": 80000 + sample * 80, "wall_ns_per_collective": 1000 + sample, "collectives": 80, "row_bytes_per_rank": 8192, "rank_sum_launches": 0 if method == "cpu_sum" else 160, "cpu_fp32_additions": 163840 if method == "cpu_sum" else 0, "dtoh_calls": 0 if method == "peer_gpu_sum" else 160, "htod_calls": 0 if method == "peer_gpu_sum" else 160, "peer_copy_calls": 160 if method == "peer_gpu_sum" else 0, "all_output_banks_exact": True, "checked_bytes": 2 * 80 * 8192, "input_sha256": [h(f"in0-{sample}"), h(f"in1-{sample}")], "output_sha256": [h(f"out-{sample}"), h(f"out-{sample}")], "free_before": [a.FLOOR] * 2, "free_after": [a.FLOOR] * 2})
    for method in a.METHODS[:2]:
        rows.append(summary(method, [80000 + s * 80 for s in range(1, 6)], 80))
    for sample in range(6):
        for rank in (0, 1):
            rows.append({"kind": "transport_host_overhead_sample", "sample": sample, "warmup": sample == 0, "rank": rank, "logit_copy_bytes": 496640, "logit_copy_wall_ns": 100000 + sample * 10, "two_meminfo_wall_ns": 1000 + sample, "all_bytes_exact": True, "free_before": [a.FLOOR] * 2, "free_after": [a.FLOOR] * 2})
    for rank in (0, 1):
        rows.append(summary(f"rank{rank}_final_bf16_logit_download", [100000 + s * 10 for s in range(1, 6)], 1))
    rows.append(summary("two_context_meminfo_queries", [1000 + s for s in range(1, 6)], 1))
    rows.append({"kind": "transport_complete", "all_bytes_exact": True, "model_loaded": False, "free_bytes": [a.FLOOR] * 2})
    rows.append({"kind": "complete", "correctness_pass": True, "production_qualification": False, "scope": "isolated operator parity and timing screens"})
    return rows


def add_peer_evidence(rows):
    manifest = next(r for r in rows if r.get("kind") == "transport_manifest")
    manifest.update(peer_symbols_available=True, peer_capability=[1, 1], peer_enable_results=[0, 0], peer_available=True)
    boundary_source = [r for r in rows if r.get("kind") == "transport_boundary_sample"]
    rows = [r for r in rows if r.get("kind") not in ("transport_boundary_sample", "transport_sum_sample", "transport_summary")]
    insert_at = next(i for i, r in enumerate(rows) if r.get("kind") == "transport_manifest") + 1
    boundary_rows = []
    for direction in ("forward", "reverse", "roundtrip"):
        pinned = [r for r in boundary_source if r.get("direction") == direction]
        peer = []
        for row in pinned:
            duplicate = row.copy()
            duplicate["method"] = "peer"
            duplicate["logical_copy_bytes_per_leg"] = 8192
            peer.append(duplicate)
        for sample in range(6):
            order = ("pinned_host", "peer") if sample % 2 == 0 else ("peer", "pinned_host")
            by_method = {"pinned_host": pinned[sample], "peer": peer[sample]}
            boundary_rows.extend(by_method[m] for m in order)
        values = [r["wall_ns"] for r in sorted(pinned, key=lambda r: r["sample"]) if r["sample"] > 0]
        boundary_rows.extend((summary(f"pinned_boundary_{direction}", values, 32), summary(f"peer_boundary_{direction}", values, 32)))
    rows[insert_at:insert_at] = boundary_rows
    sum_rows = []
    for sample in range(6):
        order = a.METHODS if sample % 2 == 0 else tuple(reversed(a.METHODS))
        for method in order:
            fields = {"rank_sum_launches": 0 if method == "cpu_sum" else 160, "cpu_fp32_additions": 163840 if method == "cpu_sum" else 0, "dtoh_calls": 0 if method == "peer_gpu_sum" else 160, "htod_calls": 0 if method == "peer_gpu_sum" else 160, "peer_copy_calls": 160 if method == "peer_gpu_sum" else 0}
            sum_rows.append({"kind": "transport_sum_sample", "method": method, "sample": sample, "warmup": sample == 0, "wall_ns": 80000 + sample * 80, "wall_ns_per_collective": 1000 + sample, "collectives": 80, "row_bytes_per_rank": 8192, **fields, "all_output_banks_exact": True, "checked_bytes": 2 * 80 * 8192, "input_sha256": [h(f"in0-{sample}"), h(f"in1-{sample}")], "output_sha256": [h(f"out-{sample}"), h(f"out-{sample}")], "free_before": [a.FLOOR] * 2, "free_after": [a.FLOOR] * 2})
    sum_rows.extend(summary(method, [80000 + s * 80 for s in range(1, 6)], 80) for method in a.METHODS)
    overhead_at = next(i for i, r in enumerate(rows) if r.get("kind") == "transport_host_overhead_sample")
    rows[overhead_at:overhead_at] = sum_rows
    end_at = next(i for i, r in enumerate(rows) if r.get("kind") == "transport_complete")
    overhead_summaries = [summary(f"rank{rank}_final_bf16_logit_download", [100000 + s * 10 for s in range(1, 6)], 1) for rank in (0, 1)]
    overhead_summaries.append(summary("two_context_meminfo_queries", [1000 + s for s in range(1, 6)], 1))
    rows[end_at:end_at] = overhead_summaries
    return rows


@pytest.fixture
def window(tmp_path):
    root = tmp_path / "window"
    (root / "profile-results").mkdir(parents=True)
    src = "a" * 40
    exe = tmp_path / "profile.exe"
    exe.write_bytes(b"synthetic executable")
    inp = {"prompt": list(range(64)), "output_tokens": 128, "context": 4096, "cut": 20, "quant_reuse_screen": True, "transport_screen": True}
    ib = (json.dumps(inp) + "\n").encode()
    (root / "profile-input.json").write_bytes(ib)
    pin = {"name": "profile", "executable": str(exe), "sha256": hashlib.sha256(exe.read_bytes()).hexdigest(), "input": str(tmp_path / "profile-input.json"), "input_sha256": hashlib.sha256(ib).hexdigest()}
    plan = {"source_commit": src, "experiments": [pin]}
    (root / "plan.json").write_text(json.dumps(plan))
    (root / "profile-SOURCE_COMMIT").write_text(src)
    (root / "window.json").write_text(json.dumps({"passed": True, "restored": True, "clef_unchanged": True, "cleanup_errors": [], "sampler_failures": [], "memory": [{"free_mib": {u: 4000 for u in a.UUIDS}}], "plan": plan, "restoration": {"passed": True, "clef_unchanged": True}, "experiments": [{"name": "profile", "passed": True, "exit_code": 0, "completion": {"kind": "complete"}, "command": [str(exe), "unused", pin["input"]], "outputs": ["profile-results\\results.jsonl"]}]}))
    baseline = tmp_path / "accepted-baseline.json"
    baseline.write_text(json.dumps({"source_commit": "b" * 40, "input": {"prompt_tokens": 64, "output_tokens": 128, "context": 4096, "cut": 20, "greedy": True}, "candidate_accepted": True, "production_qualification": False, "arms": {"candidate": {"turn_seconds": {"median": 1.6088229}}}}))
    rows = make_rows()
    (root / "profile-results/results.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return root, baseline


def test_valid_both_operator_screens(window):
    result = a.analyze(*window)
    assert result["production_qualification"] is False
    assert len(result["quant_reuse"]["measurements"]) == 8
    assert len(result["transport"]["sum_summaries"]) == 2
    assert result["transport"]["collectives_per_sample"] == 80


def test_peer_enabled_optional_method_path(window):
    root, baseline = window
    path = root / "profile-results/results.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows = add_peer_evidence(rows)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    result = a.analyze(root, baseline)
    assert result["transport"]["peer_available"] is True
    assert "peer_gpu_sum" in result["transport"]["sum_summaries"]


@pytest.mark.parametrize("mutate", [
    lambda rs: rs.remove(next(r for r in rs if r.get("kind") == "quant_reuse_sample")),
    lambda rs: next(r for r in rs if r.get("kind") == "quant_reuse_kernelwork").update(total_nodes=1),
    lambda rs: next(r for r in rs if r.get("kind") == "quant_reuse_encoded_parity").update(unequal_scale_eligibility_rejected=False),
    lambda rs: next(r for r in rs if r.get("kind") == "quant_reuse_sample").update(elapsed_ms=float("nan")),
    lambda rs: next(r for r in rs if r.get("kind") == "transport_sum_sample").update(rank_sum_launches=0),
    lambda rs: next(r for r in rs if r.get("kind") == "transport_summary").update(median_ns=1),
    lambda rs: next(r for r in rs if r.get("kind") == "transport_manifest").update(peer_capability=[1, 1], peer_available=True),
    lambda rs: next(r for r in rs if r.get("kind") == "quant_reuse_sample").update(output_sha256="f" * 64),
    lambda rs: rs.append(next(r for r in rs if r.get("kind") == "quant_reuse_sample" and r.get("ordinal") == 0 and r.get("case") == "qkv_z_pair" and r.get("cache_scope") == "hot_one_bank" and r.get("pair") == 0 and r.get("order") == 0).copy()),
    lambda rs: [rs.remove(r) for r in list(rs) if r.get("kind", "").startswith("quant_reuse_") and r.get("case") == "q_k_v_triple"],
    lambda rs: rs.append(dict(next(r for r in rs if r.get("kind") == "transport_complete"))),
])
def test_rejects_missing_corrupt_or_duplicate_evidence(window, mutate):
    root, baseline = window
    path = root / "profile-results/results.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    mutate(rows)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    with pytest.raises((ValueError, KeyError, TypeError)):
        a.analyze(root, baseline)


def test_rejects_plan_hash_and_sampler_floor_mutations(window):
    root, baseline = window
    path = root / "window.json"
    receipt = json.loads(path.read_text())
    receipt["memory"][0]["free_mib"][a.UUIDS[0]] = 1535
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        a.analyze(root, baseline)


def test_rejects_wrong_source_and_unrelated_screen_flag(window):
    root, baseline = window
    marker = root / "profile-SOURCE_COMMIT"
    marker.write_text("c" * 40)
    with pytest.raises(ValueError, match="source marker"):
        a.analyze(root, baseline)
    marker.write_text("a" * 40)
    input_path = root / "profile-input.json"
    inp = json.loads(input_path.read_text())
    inp["expert_n8_ab"] = True
    encoded = (json.dumps(inp) + "\n").encode()
    input_path.write_bytes(encoded)
    plan_path = root / "plan.json"
    plan = json.loads(plan_path.read_text())
    plan["experiments"][0]["input_sha256"] = hashlib.sha256(encoded).hexdigest()
    plan_path.write_text(json.dumps(plan))
    receipt_path = root / "window.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["plan"] = plan
    receipt["experiments"][0]["command"][2] = plan["experiments"][0]["input"]
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="unrelated expert"):
        a.analyze(root, baseline)


def test_rejects_unaccepted_or_unshaped_baseline(window):
    root, baseline = window
    data = json.loads(baseline.read_text())
    data["candidate_accepted"] = False
    baseline.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="accepted 64/128"):
        a.analyze(root, baseline)


@pytest.mark.parametrize("mutation", ("rank_mismatch", "method_mismatch"))
def test_rejects_different_reduced_outputs(window, mutation):
    root, baseline = window
    path = root / "profile-results/results.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    row = next(r for r in rows if r.get("kind") == "transport_sum_sample" and r["method"] == "cpu_sum" and r["sample"] == 1)
    row["output_sha256"][0] = h("wrong reduced output")
    if mutation == "method_mismatch":
        row["output_sha256"][1] = row["output_sha256"][0]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    with pytest.raises(ValueError, match="reduced"):
        a.analyze(root, baseline)
