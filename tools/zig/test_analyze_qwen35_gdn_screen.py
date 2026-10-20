import copy
import json

import pytest
from analyze_qwen35_gdn_screen import (
    BANK_BYTES,
    FLOOR,
    OUT_BYTES,
    SCOPE,
    STATE_BYTES,
    UUIDS,
    WORK,
    analyze,
)


def journal_rows():
    rows = [{"kind": "start", "source_commit": "a" * 40, "identities": [{"ordinal": i, "uuid": uuid} for i, uuid in enumerate(UUIDS)], "scope": SCOPE}]
    for ordinal in (0, 1):
        def add(kind, device=ordinal, **fields):
            rows.append({"kind": kind, "ordinal": device, **fields})
        add("gdn_fixture", seed=0x35D019, pattern_version=2, state_layout="head,value,key", value_heads=32, key_heads=16, dimension=128, state_bytes=STATE_BYTES, out_bytes=OUT_BYTES, q_dtype="FP32", k_dtype="FP32", v_dtype="BF16", state_dtype="FP32", out_dtype="FP32", q_elements=2048, k_elements=2048, v_elements=4096, decay_elements=32, beta_elements=32, initial_state_sha256="1" * 64, free_bytes=FLOOR, floor_bytes=FLOOR, scope=SCOPE)
        add("gdn_negative_control", poisoned_output_rejected=True, changed_state_rejected=True, changed_output_rejected=True, late_bank_state_mismatch_rejected=True, late_bank_output_mismatch_rejected=True)
        add("gdn_unwritten_state_control", unchanged_initial_state_with_correct_output_rejected=True)
        for step in range(16):
            add("gdn_step_parity", step=step, inputs_sha256=f"{step:064x}", state_sha256="2" * 64, output_sha256="3" * 64, state_bit_exact=True, output_bit_exact=True, finite_state=True, finite_output=True, unwritten_initial_state_rejected=True, free_bytes=FLOOR)
        for arm in ("old", "new"):
            add("gdn_graph", arm=arm, node_count=128, kernel_launches=128, grid_x=32, block_x=128, inputs_sha256="0" * 64, expected_state_sha256="4" * 64, expected_output_sha256="5" * 64, expected_payload_sha256="6" * 64, initial_payload_sha256="7" * 64, evolving_state=True, free_bytes=FLOOR, scope=SCOPE, **WORK)
        for pair in range(6):
            for order, arm in enumerate(("old", "new") if pair % 2 == 0 else ("new", "old")):
                add("gdn_sample", pair=pair, order=order, arm=arm, warmup=pair == 0, elapsed_ms=(10 if arm == "old" else 5) + ordinal + pair / 10, kernel_launches=128, node_count=128, graph_launches=1, state_bit_exact=True, output_bit_exact=True, finite_state=True, finite_output=True, state_sha256="4" * 64, output_sha256="5" * 64, payload_sha256="6" * 64, initial_payload_sha256="7" * 64, initial_state_sha256="8" * 64, reset_before_event=True, free_bytes_before=FLOOR, free_bytes=FLOOR, floor_bytes=FLOOR, scope=SCOPE, **WORK)
        add("gdn_screen_complete", correctness_steps=16, banks=64, unique_state_bytes=134217728, updates_per_bank=2, measured_pairs=5, excluded_warmup_pairs=1, correctness_pass=True, scope=SCOPE, production_qualification=False)
    rows.append({"kind": "complete", "correctness_pass": True, "scope": SCOPE, "production_qualification": False})
    return rows


def write(tmp_path, rows):
    path = tmp_path / "results.jsonl"
    path.write_bytes(("\n".join(map(json.dumps, rows)) + "\n").encode())
    return path


def test_complete_rotating_bank_evidence_and_sum_of_gpu_savings(tmp_path):
    result = analyze(write(tmp_path, journal_rows()))
    assert result["correctness_pass"] and result["all_pairs_faster"]
    assert len(result["measurements"]) == 2
    assert result["work"]["bank_bytes"] == BANK_BYTES
    assert result["estimated_seconds_saved_per_turn"] == pytest.approx(191 * 15 * (39.0625 + 39.0625) / 1e6)
    assert result["material_5pct_screen"]
    assert not result["production_qualification"]
    assert result["measurements"][0]["arms_us_per_call"]["old"]["median"] == pytest.approx(10.3 * 1000 / 128)


@pytest.mark.parametrize("kind,key,value", [
    ("gdn_sample", "state_bit_exact", False),
    ("gdn_sample", "finite_output", False),
    ("gdn_sample", "state_sha256", "9" * 64),
    ("gdn_sample", "output_sha256", "9" * 64),
    ("gdn_sample", "payload_sha256", "9" * 64),
    ("gdn_sample", "initial_payload_sha256", "9" * 64),
    ("gdn_sample", "initial_state_sha256", "9" * 64),
    ("gdn_sample", "elapsed_ms", float("nan")),
    ("gdn_sample", "elapsed_ms", 0),
    ("gdn_sample", "elapsed_ms", True),
    ("gdn_sample", "arm", "new"),
    ("gdn_sample", "order", 1),
    ("gdn_sample", "pair", 5),
    ("gdn_sample", "warmup", False),
    ("gdn_sample", "free_bytes_before", FLOOR - 1),
    ("gdn_sample", "free_bytes", FLOOR - 1),
    ("gdn_sample", "floor_bytes", FLOOR - 1),
    ("gdn_sample", "reset_before_event", False),
    ("gdn_sample", "banks", 1),
    ("gdn_sample", "unique_state_bytes", STATE_BYTES),
    ("gdn_sample", "updates_per_bank", 128),
    ("gdn_sample", "kernel_launches", 127),
    ("gdn_sample", "graph_launches", 2),
    ("gdn_sample", "scope", "warm cache"),
    ("gdn_sample", "bank_bytes", BANK_BYTES - 1),
    ("gdn_sample", "payload_bytes", BANK_BYTES),
    ("gdn_graph", "inputs_sha256", "9" * 64),
    ("gdn_graph", "unique_state_bytes", STATE_BYTES),
    ("gdn_graph", "grid_x", 16),
    ("gdn_graph", "block_x", 64),
    ("gdn_graph", "node_count", 127),
    ("gdn_graph", "expected_payload_sha256", "9" * 64),
    ("gdn_graph", "arm", "new"),
    ("gdn_graph", "evolving_state", False),
    ("gdn_negative_control", "late_bank_output_mismatch_rejected", False),
    ("gdn_step_parity", "state_bit_exact", False),
    ("gdn_step_parity", "step", 15),
    ("gdn_step_parity", "inputs_sha256", "f" * 63),
    ("gdn_fixture", "state_layout", "head,key,value"),
    ("gdn_screen_complete", "correctness_pass", False),
    ("complete", "production_qualification", True),
])
def test_corrupt_contract_is_rejected(tmp_path, kind, key, value):
    rows = journal_rows()
    row = next(r for r in rows if r["kind"] == kind)
    row[key] = value
    with pytest.raises(AssertionError):
        analyze(write(tmp_path, rows))


@pytest.mark.parametrize("kind", ["gdn_fixture", "gdn_negative_control", "gdn_unwritten_state_control", "gdn_step_parity", "gdn_graph", "gdn_sample", "gdn_screen_complete", "complete"])
def test_missing_evidence_is_rejected(tmp_path, kind):
    rows = journal_rows()
    rows.remove(next(r for r in rows if r["kind"] == kind))
    with pytest.raises(AssertionError):
        analyze(write(tmp_path, rows))


def test_duplicated_or_third_gpu_evidence_is_rejected(tmp_path):
    rows = journal_rows()
    extra = copy.deepcopy(next(r for r in rows if r["kind"] == "gdn_sample"))
    rows.insert(-1, extra)
    with pytest.raises(AssertionError):
        analyze(write(tmp_path, rows))
    extra["ordinal"] = 2
    with pytest.raises(AssertionError):
        analyze(write(tmp_path, rows))


def test_equal_changing_inputs_are_rejected(tmp_path):
    rows = journal_rows()
    parity = [r for r in rows if r["kind"] == "gdn_step_parity"]
    parity[1]["inputs_sha256"] = parity[0]["inputs_sha256"]
    with pytest.raises(AssertionError):
        analyze(write(tmp_path, rows))


def test_regression_does_not_pass_materiality_gate(tmp_path):
    rows = journal_rows()
    for row in rows:
        if row["kind"] == "gdn_sample" and row["arm"] == "new":
            row["elapsed_ms"] += 6
    result = analyze(write(tmp_path, rows))
    assert result["estimated_seconds_saved_per_turn"] < 0
    assert not result["material_5pct_screen"] and not result["all_pairs_faster"]


@pytest.mark.parametrize("flag", [False, True])
def test_cli_writes_report_with_both_output_forms(tmp_path, monkeypatch, capsys, flag):
    import sys

    from analyze_qwen35_gdn_screen import main

    journal = write(tmp_path, journal_rows())
    output = tmp_path / "analysis.json"
    monkeypatch.setattr(sys, "argv", ["analyzer", str(journal), *(["--output"] if flag else []), str(output)])
    main()
    result = json.loads(output.read_text())
    assert result["correctness_pass"]
    assert json.loads(capsys.readouterr().out) == result
    assert b"\r" not in output.read_bytes()
