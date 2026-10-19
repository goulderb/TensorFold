import hashlib
import json

import analyze_qwen35_layout_ab as analyzer
import pytest


@pytest.fixture
def window(tmp_path):
    root = tmp_path / "window"
    result_dir = root / "profile-results"
    result_dir.mkdir(parents=True)
    executable = tmp_path / "qwen35_decode_profile.exe"
    executable.write_bytes(b"pinned test executable")
    source = "a" * 40
    profile_input = {"prompt": list(range(64)), "output_tokens": 128, "context": 4096, "cut": 20, "forced": [], "gdn_layout_ab": True}
    input_bytes = json.dumps(profile_input, separators=(",", ":")).encode()
    (root / "profile-input.json").write_bytes(input_bytes)
    pin = {"name": "profile", "executable": str(executable), "sha256": hashlib.sha256(executable.read_bytes()).hexdigest(), "input": str(root / "profile-input.json"), "input_sha256": hashlib.sha256(input_bytes).hexdigest()}
    plan = {"source_commit": source, "experiments": [pin]}
    memory_mib = {uuid: 4000 for uuid in analyzer.UUIDS}
    receipt = {"passed": True, "restored": True, "clef_unchanged": True, "cleanup_errors": [], "sampler_failures": [],
               "memory": [{"free_mib": memory_mib}], "plan": plan,
               "restoration": {"passed": True, "clef_unchanged": True},
               "experiments": [{"name": "profile", "passed": True, "exit_code": 0, "completion": {"kind": "complete"},
                                "command": [str(executable), "model", str(root / "profile-input.json"), str(result_dir)],
                                "outputs": ["profile-results\\consumed-logits.bf16", "profile-results\\results.jsonl"]}]}
    (root / "plan.json").write_text(json.dumps(plan))
    (root / "window.json").write_text(json.dumps(receipt))
    (root / "profile-SOURCE_COMMIT").write_text(source)
    tokens = [0] * analyzer.OUTPUT
    payload = bytes(analyzer.OUTPUT * analyzer.VOCAB * 2)
    (result_dir / "consumed-logits.bf16").write_bytes(payload)
    payload_sha = hashlib.sha256(payload).hexdigest()
    state_sha = "b" * 64
    rows = [{"kind": "start", "source_commit": source, "implementation": "shared-qwen-engine",
             "identities": [{"ordinal": i, "uuid": uuid} for i, uuid in enumerate(analyzer.UUIDS)],
             "prompt_tokens": 64, "output_tokens": 128, "context": 4096, "cut": 20}]
    rows.append({"kind": "layout_reference", "arm": "baseline", "graph": False, "tokens": tokens,
                 "consumed_logits_sha256": payload_sha, "canonical_state_sha256": state_sha,
                 "canonical_state_bytes": 8192, "canonical_state_version": 2})
    for arm in analyzer.ARMS:
        for graph in (False, True):
            rows.append({"kind": "layout_qualification", "arm": arm, "graph": graph,
                         "all_rows_tokens_and_canonical_state_exact": True})
    for rank in range(2):
        rows.append({"kind": "layout_graph_bindings", "rank": rank, "expected_gdn_nodes": 15,
                     "baseline_verified_nodes": 15, "candidate_verified_nodes": 15})
    for pair in range(6):
        order_arms = ["baseline", "candidate"] if pair % 2 == 0 else ["candidate", "baseline"]
        for order, arm in enumerate(order_arms):
            for rank, uuid in enumerate(analyzer.UUIDS):
                rows.append({"kind": "layout_memory", "pair": pair, "order": order, "rank": rank,
                             "free_bytes": 4_000_000_000, "floor_bytes": analyzer.FLOOR})
            ns = 1_000_000_000 if arm == "baseline" else 900_000_000
            rows.append({"kind": "layout_turn", "pair": pair, "order": order, "warmup": pair == 0, "arm": arm, "graph": True,
                         "times": {"prefill_ns": 100_000_000, "decode_ns": ns - 100_000_000, "elapsed_ns": ns, "appends": 191},
                         "graph_launches": 382, "graph_nodes": 2865, "position": 191,
                         "all_rows_tokens_and_canonical_state_exact": True, "consumed_logits_sha256": payload_sha,
                         "canonical_state_sha256": state_sha, "canonical_state_version": 2, "tokens": tokens})
    for arm in analyzer.ARMS:
        rows.append({"kind": "layout_controls", "arm": arm, "distinct_prompts_and_positions": True,
                     "inactive_complete_state_unchanged": True, "reset_exact": True,
                     "partial_stage_failure_poison": True, "uncommitted_lengths": True, "canonical_state_version": 2})
    rows.append({"kind": "layout_recreated_session", "candidate_exact": True, "baseline_after_candidate_exact": True})
    rows.append({"kind": "complete", "correctness_pass": True, "scope": analyzer.SCOPE,
                 "production_qualification": False, "independent_reference_parity": "not checked by this executable"})
    journal = result_dir / "results.jsonl"
    journal.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    return root, rows, journal


def test_reports_five_pair_candidate_result_and_scope(window):
    root, _, _ = window
    result = analyzer.analyze(root)
    assert result["candidate_accepted"]
    assert result["material_5pct_turn_screen"]
    assert result["measured_pairs"] == 5 and result["warmup_pairs_excluded"] == 1
    assert result["independent_reference_gate"].startswith("FAILS")
    assert "not HTTP production performance or physical DRAM bandwidth" in result["scope"]
    assert result["arms"]["baseline"]["turn_seconds"]["median"] == 1.0


@pytest.mark.parametrize("kind,key,value", [
    ("layout_graph_bindings", "candidate_verified_nodes", 14),
    ("layout_turn", "graph_launches", 381),
    ("layout_turn", "position", 190),
    ("layout_turn", "canonical_state_sha256", "c" * 64),
    ("layout_controls", "reset_exact", False),
    ("layout_recreated_session", "candidate_exact", False),
])
def test_rejects_corrupt_protocol_rows(window, kind, key, value):
    root, rows, journal = window
    next(row for row in rows if row["kind"] == kind)[key] = value
    journal.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    with pytest.raises(ValueError):
        analyzer.analyze(root)


def test_rejects_missing_graph_binding(window):
    root, rows, journal = window
    rows.remove(next(row for row in rows if row["kind"] == "layout_graph_bindings"))
    journal.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    with pytest.raises(ValueError):
        analyzer.analyze(root)


@pytest.mark.parametrize("change", ["source", "restoration", "memory", "missing_turn", "input_shape", "executable_hash"])
def test_rejects_protected_window_mutants(window, change):
    root, rows, journal = window
    if change == "source":
        (root / "profile-SOURCE_COMMIT").write_text("c" * 40)
    elif change in ("restoration", "memory"):
        path = root / "window.json"
        receipt = json.loads(path.read_text())
        if change == "restoration":
            receipt["restoration"]["passed"] = False
        else:
            receipt["memory"][0]["free_mib"][analyzer.UUIDS[1]] = 1535
        path.write_text(json.dumps(receipt))
    elif change == "missing_turn":
        rows.remove(next(row for row in rows if row["kind"] == "layout_turn"))
        journal.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    elif change == "input_shape":
        path = root / "profile-input.json"
        data = json.loads(path.read_text())
        data["output_tokens"] = 127
        path.write_text(json.dumps(data))
    elif change == "executable_hash":
        path = root / "plan.json"
        plan = json.loads(path.read_text())
        plan["experiments"][0]["sha256"] = "0" * 64
        path.write_text(json.dumps(plan))
    with pytest.raises((ValueError, KeyError, IndexError)):
        analyzer.analyze(root)


def test_rejects_controller_memory_floor(window):
    root, _, _ = window
    receipt_path = root / "window.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["memory"][0]["free_mib"][analyzer.UUIDS[0]] = 1535
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        analyzer.analyze(root)


def test_rejects_missing_consumed_payload_in_controller_outputs(window):
    root, _, _ = window
    path = root / "window.json"
    receipt = json.loads(path.read_text())
    receipt["experiments"][0]["outputs"] = ["profile-results\\results.jsonl"]
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="journal output"):
        analyzer.analyze(root)


def test_rejects_greedy_tie_break_and_payload_hash(window):
    root, rows, journal = window
    payload_path = root / "profile-results" / "consumed-logits.bf16"
    data = bytearray(payload_path.read_bytes())
    data[0:2] = b"\x80\x3f"
    data[2:4] = b"\x80\x3f"
    payload_path.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    for row in rows:
        if row["kind"] in ("layout_reference", "layout_turn"):
            row["consumed_logits_sha256"] = digest
    # A tie between ids zero and one is resolved in favor of id zero.
    journal.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    assert analyzer.analyze(root)["candidate_accepted"]
    rows[1]["tokens"][0] = 1
    journal.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    with pytest.raises(ValueError, match="lowest-id"):
        analyzer.analyze(root)

