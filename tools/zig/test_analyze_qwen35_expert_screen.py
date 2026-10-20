import json

import pytest
from analyze_qwen35_expert_screen import analyze


def journal():
    ids = ["GPU-f2c0ea1e-a2be-f0b4-6c23-ffe33b537a06", "GPU-dd2a988c-fe38-24ab-233a-9e9bb3b2645a"]
    selected = [112, 106, 107, 238, 181, 57, 43, 200]
    rows = [{"kind": "start", "source_commit": "a" * 40, "identities": [{"uuid": i} for i in ids]}]
    for ordinal in (0, 1):
        rows.append({"kind": "expert_fixture", "ordinal": ordinal, "manifest_sha256": "b" * 64, "expert_order": selected + [i for i in range(256) if i not in selected], "bank0_original_ids": selected})
        for epi in ("up", "down_f32"):
            def add(kind, ordinal=ordinal, epi=epi, **fields):
                rows.append({"kind": kind, "ordinal": ordinal, "epi": epi, **fields})
            add("expert_plan", actual_items=8, bound=8, limit=0, relu2=False)
            for variant in ("baseline", "pipe4", "n16", "n16_pipe4"):
                add("expert_kernel", variant=variant)
            for variant in ("pipe4", "n16", "n16_pipe4"):
                for bank in range(32):
                    add("expert_bank_parity", variant=variant, bank=bank, finite=True, bit_exact=True, saved_real_output=bank == 0)
                for pair in range(6):
                    for order in (0, 1):
                        candidate = (pair + order) % 2 == 1
                        add("expert_sample", comparison=variant, pair=pair, order=order, arm=variant if candidate else "baseline", warmup=pair == 0, kernel_launches=128, banks=32, final_bank=31, finite=True, bit_exact_final=True, elapsed_ms=1.28 if candidate else 2.56)
            add("expert_projection_complete", all_bank_prechecks=True, saved_bank0_exact=True, graph_finals_exact=True)
    rows.append({"kind": "complete", "correctness_pass": True})
    return rows


def test_paired_operator_projection_is_explicitly_not_model_measurement(tmp_path):
    path = tmp_path / "results.jsonl"
    path.write_text("\n".join(map(json.dumps, journal())))
    result = analyze(path)
    assert result["correctness_pass"]
    assert result["measurements"][0]["median_saving_us"] == 10
    assert result["projection"]["n16"]["estimated_seconds_saved_per_64_128_turn"] == pytest.approx(0.1528)


@pytest.mark.parametrize("defect", ["missing_pair", "failed_exactness", "missing_bank", "wrong_work", "no_completion", "changed_fixture"])
def test_rejects_incomplete_or_incorrect_measurement(tmp_path, defect):
    rows = journal()
    sample = next(row for row in rows if row["kind"] == "expert_sample" and not row["warmup"])
    if defect == "missing_pair":
        rows.remove(sample)
    elif defect == "failed_exactness":
        sample["bit_exact_final"] = False
    elif defect == "missing_bank":
        rows.remove(next(row for row in rows if row["kind"] == "expert_bank_parity"))
    elif defect == "wrong_work":
        sample["kernel_launches"] = 127
    elif defect == "no_completion":
        rows.pop()
    elif defect == "changed_fixture":
        next(row for row in rows if row["kind"] == "expert_fixture")["manifest_sha256"] = "c" * 64
    path = tmp_path / "results.jsonl"
    path.write_text("\n".join(map(json.dumps, rows)))
    with pytest.raises(AssertionError):
        analyze(path)
