"""Validate the paired offline Qwen quant-reuse and transport operator screens."""
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
FLOOR_MIB = 1536
SOURCE_SCOPE = "isolated Qwen operator screens; no model load or model throughput claim"
QUANT_SCOPE = "synthetic captured FP8 input-quantization seam; current-fork equivalence only; independent HF full-logit gate remains failed"
TRANSPORT_SCOPE = "host-coordinated primitives only; 80 sums assumes two reductions per layer across 40 layers; no proof of actual TP design"
QUANT_JOURNAL_BASELINE_SECONDS = 1.7232142
QUANT_CASES = {"qkv_z_pair": ((8192, 4096), 15), "q_k_v_triple": ((8192, 512, 512), 5)}
QUANT_SCOPES = ("hot_one_bank", "rotating_eight_banks")
QUANT_ARMS = ("baseline", "shared_quant")
METHODS = ("pinned_gpu_sum", "cpu_sum", "peer_gpu_sum")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None, "invalid SHA256")
    return value


def finite_positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def same_float(actual, expected):
    return type(actual) in (int, float) and math.isfinite(actual) and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12)


def exact_rows(rows, kind, count, key_fields=()):
    chosen = [r for r in rows if r.get("kind") == kind]
    require(len(chosen) == count, f"{kind} count mismatch")
    if key_fields:
        keys = [tuple(r.get(k) for k in key_fields) for r in chosen]
        require(len(set(keys)) == count, f"duplicate {kind} tuple")
    return chosen


def check_controller(window, rows):
    receipt, plan = read_json(window / "window.json"), read_json(window / "plan.json")
    require(receipt.get("passed") is True and receipt.get("restored") is True, "controller did not pass and restore")
    require(receipt.get("clef_unchanged") is True and receipt.get("cleanup_errors") == [] and receipt.get("sampler_failures") == [], "controller reports Clef change, cleanup error, or sampler failure")
    require(plan.get("source_commit") == receipt.get("plan", {}).get("source_commit"), "receipt and plan source differ")
    source = plan.get("source_commit")
    require(isinstance(source, str) and re.fullmatch(r"[0-9a-f]{40}", source), "invalid source commit")
    require((window / "profile-SOURCE_COMMIT").read_text(encoding="utf-8").strip() == source == rows[0].get("source_commit"), "source marker, plan, and journal differ")
    require(rows[0].get("identities") and [x.get("uuid") for x in rows[0]["identities"]] == UUIDS, "journal physical GPU order differs")
    memory = receipt.get("memory")
    require(isinstance(memory, list) and memory, "missing controller memory samples")
    for sample in memory:
        free = sample.get("free_mib", {})
        require(set(free) == set(UUIDS) and all(type(n) is int and n >= FLOOR_MIB for n in free.values()), "controller GPU memory floor failed")
    restore = receipt.get("restoration") or {}
    require(restore.get("passed") is True and restore.get("clef_unchanged") is True, "restoration or Clef check failed")
    experiments = plan.get("experiments", [])
    matches = [e for e in experiments if e.get("name") == "profile"]
    require(len(matches) == 1, "expected one profile experiment")
    exp = matches[0]
    receipt_exp = [e for e in receipt.get("plan", {}).get("experiments", []) if e.get("name") == "profile"]
    require(len(receipt_exp) == 1 and all(receipt_exp[0].get(k) == exp.get(k) for k in ("executable", "sha256", "input", "input_sha256")), "receipt experiment pins differ")
    require(Path(exp["executable"]).is_file() and sha_file(exp["executable"]) == digest(exp.get("sha256")), "current executable SHA differs")
    copied = window / "profile-input.json"
    require(copied.is_file() and sha_file(copied) == digest(exp.get("input_sha256")), "copied input SHA differs")
    inp = read_json(copied)
    require(len(inp.get("prompt", [])) == 64 and inp.get("output_tokens") == 128 and inp.get("context") == 4096 and inp.get("cut") == 20, "input shape mismatch")
    require(inp.get("quant_reuse_screen") is True and inp.get("transport_screen") is True, "both requested operator screens must be enabled")
    require(all(inp.get(k, False) is False for k in ("expert_wn2_screen", "expert_n8_screen", "expert_n8_ab", "expert_up_n8", "gdn_layout_screen", "gdn_layout_ab")), "unrelated expert or GDN screen enabled")
    runs = receipt.get("experiments", [])
    require(len(runs) == 1 and runs[0].get("name") == "profile" and runs[0].get("exit_code") == 0 and runs[0].get("passed") is True and runs[0].get("completion", {}).get("kind") == "complete", "controller experiment did not complete successfully")
    require(runs[0].get("outputs") == ["profile-results\\results.jsonl"], "unexpected controller output path")
    require(runs[0].get("command", [None, None, None])[2] == exp.get("input"), "controller command input differs from plan")
    return {"source_commit": source, "executable_sha256": exp["sha256"], "input_sha256": exp["input_sha256"], "minimum_free_mib": {u: min(s["free_mib"][u] for s in memory) for u in UUIDS}}


def quant(rows, accepted_turn_seconds):
    allowed = {"quant_reuse_fixture", "quant_reuse_encoded_parity", "quant_reuse_output_parity", "quant_reuse_kernelwork", "quant_reuse_sample", "quant_reuse_case_complete", "quant_reuse_screen_complete"}
    qr = [r for r in rows if r.get("kind", "").startswith("quant_reuse_")]
    require(qr and all(r.get("kind") in allowed for r in qr), "unknown quant journal kind")
    fixtures = exact_rows(qr, "quant_reuse_fixture", 4, ("ordinal", "case"))
    for f in fixtures:
        expected_outputs, _ = QUANT_CASES[f["case"]]
        require(f.get("ordinal") in (0, 1) and tuple(f.get("outputs", [])) == expected_outputs and f.get("width") == 2048 and f.get("rows") == 1 and f.get("input_scale") == 0.125, "quant fixture shape mismatch")
        banks = 8
        weight_bytes = sum(expected_outputs) * 2048
        require(f.get("banks") == banks and f.get("weight_bytes_per_bank") == weight_bytes and f.get("unique_weight_bytes") == banks * weight_bytes and f.get("groups") == 128 and f.get("distinct_input_rows") == 128 and f.get("payloads") == 2 and f.get("output_bytes") == 128 * sum(expected_outputs) * 2, "quant fixture work count mismatch")
        require(f.get("pack_path") == "modelopt.Packer.fp8Lane + modelopt.fp8Weight" and f.get("baseline_path") == "qlinear.Linear.decode(.fp8)" and f.get("candidate_path") == "Ck.quant8 once + same ordered Ck.lane8" and f.get("scope") == QUANT_SCOPE, "quant fixture provenance or scope mismatch")
        require(type(f.get("l2_bytes")) is int and f["unique_weight_bytes"] > f["l2_bytes"] and type(f.get("free_bytes")) is int and f["free_bytes"] >= FLOOR, "quant rotating footprint or memory floor invalid")
    tuple2 = ("ordinal", "case", "cache_scope")
    kernel = exact_rows(qr, "quant_reuse_kernelwork", 16, tuple2 + ("arm",))
    parity = exact_rows(qr, "quant_reuse_output_parity", 16, tuple2 + ("payload",))
    encoded = exact_rows(qr, "quant_reuse_encoded_parity", 16, ("ordinal", "case", "payload", "banks"))
    samples = exact_rows(qr, "quant_reuse_sample", 96, ("ordinal", "case", "cache_scope", "pair", "order"))
    completions = exact_rows(qr, "quant_reuse_case_complete", 8, tuple2)
    screens = exact_rows(qr, "quant_reuse_screen_complete", 2, ("ordinal",))
    require(all(r.get("cases") == 2 and r.get("cache_scopes") == 2 and r.get("measured_pairs_per_case_scope") == 5 and r.get("correctness_pass") is True and r.get("full_model_gate_pass") is False and r.get("production_qualification") is False and r.get("scope") == QUANT_SCOPE for r in screens), "quant screen completion metadata invalid")
    measurements, savings, payload_inputs = [], {}, {}
    for ordinal in (0, 1):
        for case, (outputs, partition) in QUANT_CASES.items():
            projections = len(outputs)
            expected_counts = {"baseline": (128 * projections, 128 * projections, 256 * projections), "shared_quant": (128, 128 * projections, 128 + 128 * projections)}
            for scope in QUANT_SCOPES:
                bank_count = 1 if scope == "hot_one_bank" else 8
                group = [r for r in kernel if (r["ordinal"], r["case"], r["cache_scope"]) == (ordinal, case, scope)]
                require({r["arm"] for r in group} == set(QUANT_ARMS), "quant arm roster mismatch")
                for r in group:
                    qn, gn, total = expected_counts[r["arm"]]
                    require(r.get("quant_symbol") == "tf_ck_quant8" and type(r.get("quant_function_handle")) is int and r["quant_function_handle"] > 0 and (r.get("quant_nodes"), r.get("gemm_nodes"), r.get("total_nodes")) == (qn, gn, total), "quant graph work count mismatch")
                    require(r.get("function_geometry_arguments_and_serial_dependencies_checked") is True and r.get("quantized_rows_per_node") == 1 and r.get("quantized_columns_per_node") == 2048 and r.get("groups") == 128 and r.get("banks") == bank_count, "quant function geometry, arguments, dependencies, or bank count not proven")
                pars = [r for r in parity if (r["ordinal"], r["case"], r["cache_scope"]) == (ordinal, case, scope)]
                require({r["payload"] for r in pars} == {0, 1} and all(r.get("all_output_bytes_compared") == 128 * sum(outputs) * 2 and r.get("all_banks_checked") == bank_count and r.get("finite") is True and r.get("bit_exact") is True for r in pars), "quant payload output parity incomplete")
                require(len({digest(r.get("output_sha256")) for r in pars}) == 2 and len({digest(r.get("input_sha256")) for r in pars}) == 2, "quant changing payload hashes did not change")
                for r in pars:
                    key = ordinal, case, r["payload"]
                    current = digest(r["input_sha256"])
                    require(key not in payload_inputs or payload_inputs[key] == current, "quant cache scopes used different input payloads")
                    payload_inputs[key] = current
                e = [r for r in encoded if (r["ordinal"], r["case"], r["banks"]) == (ordinal, case, bank_count)]
                require({r.get("payload") for r in e} == {0, 1} and all(r.get("banks") == bank_count and r.get("groups") == 128 and r.get("bytes_compared_per_projection") == 2048 and r.get("baseline_quantizations_checked") == 128 * projections and r.get("shared_quantizations_checked") == 128 and all(r.get(k) is True for k in ("all_encoded_bytes_exact", "changing_rows_change_encoded_bytes", "unequal_scale_encoded_bytes_differ", "unequal_scale_eligibility_rejected", "independent_weight_scales")) for r in e), "encoded activation parity or eligibility controls failed")
                g = [r for r in samples if (r["ordinal"], r["case"], r["cache_scope"]) == (ordinal, case, scope)]
                arm_samples = {a: [] for a in QUANT_ARMS}
                paired, wall_paired = [], []
                for pair in range(6):
                    both = sorted((r for r in g if r.get("pair") == pair), key=lambda r: r.get("order", -1))
                    arms = ["baseline", "shared_quant"] if pair % 2 == 0 else ["shared_quant", "baseline"]
                    require(len(both) == 2 and [r.get("arm") for r in both] == arms and [r.get("order") for r in both] == [0, 1], "quant alternating pair malformed")
                    vals = {}
                    wall_vals = {}
                    for r in both:
                        require(r.get("warmup") is (pair == 0) and r.get("payload") == pair % 2 and r.get("graph_launches") == 1 and r.get("groups") == 128 and r.get("all_output_bytes_compared") == 128 * sum(outputs) * 2 and r.get("bit_exact") is True and r.get("finite") is True and r.get("reset_outside_events") is True and r.get("scope") == QUANT_SCOPE, "quant timed sample invalid")
                        require(finite_positive(r.get("elapsed_ms")) and type(r.get("wall_ns")) is int and r["wall_ns"] > 0 and finite_positive(r.get("us_per_group")) and finite_positive(r.get("wall_us_per_group")), "quant sample timing invalid")
                        expected = r["elapsed_ms"] * 1000 / 128
                        wall = r["wall_ns"] / 1000 / 128
                        require(math.isclose(r["us_per_group"], expected, rel_tol=1e-9, abs_tol=1e-9) and math.isclose(r["wall_us_per_group"], wall, rel_tol=1e-9, abs_tol=1e-9), "quant derived time differs from raw time")
                        require(digest(r.get("output_sha256")) == next(p["output_sha256"] for p in pars if p["payload"] == r["payload"]), "sample output hash differs from untimed parity reference")
                        require(type(r.get("free_bytes_before")) is int and type(r.get("free_bytes")) is int and min(r["free_bytes_before"], r["free_bytes"]) >= FLOOR, "quant sampler floor failed")
                        vals[r["arm"]] = r["us_per_group"]
                        wall_vals[r["arm"]] = r["wall_us_per_group"]
                    if pair:
                        for arm, value in vals.items(): arm_samples[arm].append(value)
                        paired.append(vals["baseline"] - vals["shared_quant"])
                        wall_paired.append(wall_vals["baseline"] - wall_vals["shared_quant"])
                c = next(r for r in completions if (r["ordinal"], r["case"], r["cache_scope"]) == (ordinal, case, scope))
                require(len(c.get("paired_saving_us_per_group", [])) == 5 and all(same_float(x, y) for x, y in zip(c["paired_saving_us_per_group"], paired)) and same_float(c.get("median_saving_us_per_group"), statistics.median(paired)) and same_float(c.get("min_saving_us_per_group"), min(paired)) and same_float(c.get("max_saving_us_per_group"), max(paired)), "quant summary disagrees with raw paired samples")
                required_total_us = QUANT_JOURNAL_BASELINE_SECONDS * .05 * 1e6 / 191
                required_quant_us = required_total_us / 50
                case_projection = 191 * partition * statistics.median(paired)
                reported_wall = c.get("wall_paired_saving_us_per_group", [])
                require(len(reported_wall) == 5 and all(same_float(x, y) for x, y in zip(reported_wall, wall_paired)) and same_float(c.get("estimated_turn_saving_us_this_case_this_partition"), case_projection), "quant wall savings or case projection disagrees with raw samples")
                require(c.get("groups_per_append_on_this_partition") == partition and c.get("projection_formula") == "191*(15*pair_gpu0+5*triple_gpu0+15*pair_gpu1+5*triple_gpu1) us" and c.get("accepted_turn_seconds") == QUANT_JOURNAL_BASELINE_SECONDS and same_float(c.get("required_total_us_per_append"), required_total_us) and same_float(c.get("required_us_per_removed_quant"), required_quant_us) and c.get("measured_pairs") == 5 and c.get("excluded_warmup_pairs") == 1 and c.get("correctness_pass") is True and c.get("production_qualification") is False, "quant case completion metadata invalid")
                savings[ordinal, case, scope] = statistics.median(paired)
                spread = max(paired) - min(paired)
                measurements.append({"ordinal": ordinal, "case": case, "cache_scope": scope, "arms_us_per_group": {a: {"samples": v, "median": statistics.median(v), "range": [min(v), max(v)]} for a, v in arm_samples.items()}, "paired_saving_us_per_group": paired, "median_saving_us_per_group": statistics.median(paired), "paired_saving_spread_us": spread, "advance_screen_pass": min(paired) > 0 and statistics.median(paired) > spread})
    projection = 191 * sum(part * savings[o, case, scope] for o in (0, 1) for case, (_, part) in QUANT_CASES.items() for scope in ("rotating_eight_banks",))
    per_scope = {scope: 191 * sum(part * savings[o, case, scope] for o in (0, 1) for case, (_, part) in QUANT_CASES.items()) for scope in QUANT_SCOPES}
    rotating_clean = all(m["advance_screen_pass"] for m in measurements if m["cache_scope"] == "rotating_eight_banks")
    return {"scope": QUANT_SCOPE, "measurements": measurements, "estimated_turn_saving_us": projection, "estimated_turn_saving_us_by_cache_scope": per_scope, "materiality_baseline_seconds": accepted_turn_seconds, "materiality_baseline_source": None, "journal_historical_baseline_seconds": QUANT_JOURNAL_BASELINE_SECONDS, "material_5pct_screen": projection / 1e6 >= .05 * accepted_turn_seconds, "rotating_candidate_for_integration": rotating_clean and per_scope["rotating_eight_banks"] / 1e6 >= .05 * accepted_turn_seconds, "production_qualification": False, "interpretation": "Synthetic operator projection on the current fork. It is not whole-model speed, bandwidth, or an independent HF gate pass."}


def transport(rows):
    tr = [r for r in rows if r.get("kind", "").startswith("transport_")]
    allowed = {"transport_manifest", "transport_boundary_sample", "transport_sum_sample", "transport_summary", "transport_host_overhead_sample", "transport_complete"}
    require(tr and all(r.get("kind") in allowed for r in tr), "unknown transport journal kind")
    manifest = exact_rows(tr, "transport_manifest", 1)[0]
    require(manifest.get("private_contexts") is True and manifest.get("ordinals") == [0, 1] and manifest.get("rank_sum_function") == "tf_rank_sum" and len(manifest.get("rank_sum_bindings", [])) == 2 and all(type(x) is int and x > 0 for x in manifest["rank_sum_bindings"]), "transport contexts, ranks, or actual sum binding invalid")
    require(manifest.get("boundary_parts") == 2 and manifest.get("boundary_part_bytes") == 4096 and manifest.get("collectives_per_sample") == 80 and manifest.get("sum_row_bytes_per_rank") == 8192 and manifest.get("warmups") == 1 and manifest.get("measured_samples") == 5 and manifest.get("memory_floor_bytes") == FLOOR and manifest.get("negative_control") == "last output byte mutation rejected" and manifest.get("scope") == TRANSPORT_SCOPE, "transport manifest work shape or scope mismatch")
    caps, enabled = manifest.get("peer_capability"), manifest.get("peer_enable_results")
    require(isinstance(caps, list) and len(caps) == 2 and all(type(x) is int and x in (0, 1) for x in caps) and isinstance(enabled, list) and len(enabled) == 2 and all(type(x) is int for x in enabled), "peer capability or enable roster invalid")
    peer = manifest.get("peer_available")
    require(peer is (manifest.get("peer_symbols_available") is True and caps == [1, 1] and enabled == [0, 0]), "peer availability contradicts symbols, directions, or enable results")
    require(manifest.get("free_bytes") and len(manifest["free_bytes"]) == 2 and all(type(x) is int and x >= FLOOR for x in manifest["free_bytes"]), "transport manifest floor failed")
    boundaries = exact_rows(tr, "transport_boundary_sample", 18 * (2 if peer else 1), ("method", "direction", "sample"))
    expected_dirs = ("forward", "reverse", "roundtrip")
    for direction in expected_dirs:
        observed = [r["method"] for r in tr if r["kind"] == "transport_boundary_sample" and r.get("direction") == direction]
        roster = ("pinned_host", "peer") if peer else ("pinned_host",)
        expected_order = [m for sample in range(6) for m in (roster if sample % 2 == 0 else tuple(reversed(roster)))]
        require(observed == expected_order, "boundary methods do not follow the helper's alternating order")
    expected_summary_count = 3 * (2 if peer else 1) + (3 if peer else 2) + 3
    all_summaries = [r for r in tr if r.get("kind") == "transport_summary"]
    require(len(all_summaries) == expected_summary_count and len({r.get("case") for r in all_summaries}) == expected_summary_count, "extra, duplicate, or missing transport summary")
    boundary_summary = {}
    for direction in expected_dirs:
        for method in (("pinned_host", "peer") if peer else ("pinned_host",)):
            group = [r for r in boundaries if r.get("direction") == direction and r.get("method") == method]
            require(len(group) == 6, "boundary sample roster mismatch")
            values = []
            input_hashes = []
            for sample, r in enumerate(sorted(group, key=lambda x: x.get("sample", -1))):
                require(r.get("sample") == sample and r.get("warmup") is (sample == 0) and r.get("operations") == 32 and r.get("payload_bytes_per_leg") == 8192 and r.get("logical_copy_bytes_per_leg") == (16384 if method == "pinned_host" else 8192) and r.get("legs_per_operation") == (2 if direction == "roundtrip" else 1) and r.get("all_bytes_exact") is True, "boundary operation count or bytes invalid")
                require(type(r.get("wall_ns")) is int and r["wall_ns"] > 0 and r.get("wall_ns_per_operation") == r["wall_ns"] // 32, "boundary timing invalid")
                digest(r.get("input_sha256")); digest(r.get("output_sha256"))
                require(r["input_sha256"] == r["output_sha256"], "boundary output differs from input")
                input_hashes.append(r["input_sha256"])
                for field in ("free_before", "free_after"):
                    require(len(r.get(field, [])) == 2 and all(type(x) is int and x >= FLOOR for x in r[field]), "boundary sample floor failed")
                if sample: values.append(r["wall_ns"])
            require(len(set(input_hashes)) == 6, "boundary payload did not change across samples")
            key = f"{('pinned' if method == 'pinned_host' else 'peer')}_boundary_{direction}"
            boundary_summary[key] = summary_check(tr, key, values, 32)
    method_count = 3 if peer else 2
    sums = exact_rows(tr, "transport_sum_sample", method_count * 6, ("method", "sample"))
    observed = [(r["sample"], r["method"]) for r in tr if r["kind"] == "transport_sum_sample"]
    expected_methods = METHODS[:method_count]
    expected_order = [(sample, method) for sample in range(6) for method in (expected_methods if sample % 2 == 0 else tuple(reversed(expected_methods)))]
    require(observed == expected_order, "sum methods do not follow the helper's alternating order")
    sums_summary = {}
    output_pairs = []
    inputs_by_method = {}
    for method in METHODS[:method_count]:
        group = sorted((r for r in sums if r.get("method") == method), key=lambda x: x.get("sample", -1))
        require(len(group) == 6, "sum sample roster mismatch")
        values = []
        inputs = []
        for sample, r in enumerate(group):
            require(r.get("sample") == sample and r.get("warmup") is (sample == 0) and r.get("collectives") == 80 and r.get("row_bytes_per_rank") == 8192 and r.get("rank_sum_launches") == (0 if method == "cpu_sum" else 160) and r.get("cpu_fp32_additions") == (163840 if method == "cpu_sum" else 0) and r.get("dtoh_calls") == (0 if method == "peer_gpu_sum" else 160) and r.get("htod_calls") == (0 if method == "peer_gpu_sum" else 160) and r.get("peer_copy_calls") == (160 if method == "peer_gpu_sum" else 0), "sum method work counts invalid")
            require(type(r.get("wall_ns")) is int and r["wall_ns"] > 0 and r.get("wall_ns_per_collective") == r["wall_ns"] // 80 and r.get("checked_bytes") == 2 * 80 * 8192 and r.get("all_output_banks_exact") is True, "sum timing or output bank coverage invalid")
            ih, oh = r.get("input_sha256"), r.get("output_sha256")
            require(len(ih) == 2 and len(oh) == 2, "sum hash roster invalid")
            for h in ih + oh: digest(h)
            require(ih[0] != ih[1] and oh[0] == oh[1] and oh[0] != ih[0] and oh[1] != ih[1], "sum inputs must differ and reduced rank outputs must agree and evolve")
            if sample: inputs.append(ih)
            if sample: output_pairs.append((method, tuple(oh)))
            for field in ("free_before", "free_after"):
                require(len(r.get(field, [])) == 2 and all(type(x) is int and x >= FLOOR for x in r[field]), "sum sample floor failed")
            if sample: values.append(r["wall_ns"])
        require(len({tuple(x) for x in inputs}) == 5, "sum inputs did not change across measured samples")
        inputs_by_method[method] = inputs
        sums_summary[method] = summary_check(tr, method, values, 80)
    reference_inputs = inputs_by_method[METHODS[0]]
    require(all(inputs_by_method[method] == reference_inputs for method in METHODS[1:method_count]), "transport methods did not use the same ordered rank inputs")
    for method in METHODS[:method_count]:
        require(len({pair for name, pair in output_pairs if name == method}) == 5, "sum output hashes did not change across measured samples")
        require([pair for name, pair in output_pairs if name == method] == [pair for name, pair in output_pairs if name == METHODS[0]], "sum methods produced different reduced outputs")
    overheads = exact_rows(tr, "transport_host_overhead_sample", 12, ("sample", "rank"))
    overhead = {"rank0_final_bf16_logit_download": [], "rank1_final_bf16_logit_download": [], "two_context_meminfo_queries": []}
    for sample in range(6):
        group = [r for r in overheads if r.get("sample") == sample]
        require(len(group) == 2, "host overhead rank roster mismatch")
        query = group[0].get("two_meminfo_wall_ns")
        require(type(query) is int and query > 0 and all(r.get("two_meminfo_wall_ns") == query for r in group), "meminfo query timings mismatch")
        if sample: overhead["two_context_meminfo_queries"].append(query)
        for r in group:
            rank = r.get("rank")
            require(rank in (0, 1) and r.get("warmup") is (sample == 0) and r.get("logit_copy_bytes") == 496640 and r.get("all_bytes_exact") is True and type(r.get("logit_copy_wall_ns")) is int and r["logit_copy_wall_ns"] > 0, "logit copy measurement invalid")
            for field in ("free_before", "free_after"):
                require(len(r.get(field, [])) == 2 and all(type(x) is int and x >= FLOOR for x in r[field]), "overhead sample floor failed")
            if sample: overhead[f"rank{rank}_final_bf16_logit_download"].append(r["logit_copy_wall_ns"])
    overhead_summaries = {name: summary_check(tr, name, vals, 1) for name, vals in overhead.items()}
    exact_rows(tr, "transport_complete", 1)
    complete = next(r for r in tr if r["kind"] == "transport_complete")
    require(complete.get("all_bytes_exact") is True and complete.get("model_loaded") is False and len(complete.get("free_bytes", [])) == 2 and all(type(x) is int and x >= FLOOR for x in complete["free_bytes"]), "transport completion invalid")
    return {"scope": TRANSPORT_SCOPE, "peer_available": peer, "boundary_summaries": boundary_summary, "sum_summaries": sums_summary, "host_overhead_summaries": overhead_summaries, "collectives_per_sample": 80, "architectural_assumption": "80 dependent collectives are assumed for two reductions per layer over 40 layers. Shard compute is unmeasured; this host-coordination result cannot reject every TP design.", "production_qualification": False}


def summary_check(rows, name, values, operations):
    require(len(values) == 5 and all(type(x) is int and x > 0 for x in values), "expected five positive measured times")
    summary = [r for r in rows if r.get("kind") == "transport_summary" and r.get("case") == name]
    require(len(summary) == 1, f"missing or duplicate summary for {name}")
    s = summary[0]
    require(s.get("samples_ns") == values and s.get("median_ns") == statistics.median(values) and s.get("min_ns") == min(values) and s.get("max_ns") == max(values) and s.get("operations") == operations and s.get("median_ns_per_operation") == statistics.median(values) // operations, f"summary disagrees with raw samples for {name}")
    return {"samples_ns": values, "median_ns": statistics.median(values), "range_ns": [min(values), max(values)], "operations": operations, "median_ns_per_operation": statistics.median(values) // operations}


def analyze(window: Path, baseline_analysis: Path):
    path = window / "profile-results" / "results.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    require(rows and rows[0].get("kind") == "start" and rows[-1].get("kind") == "complete", "journal boundaries invalid")
    kinds = {"start", "complete", "quant_reuse_fixture", "quant_reuse_encoded_parity", "quant_reuse_output_parity", "quant_reuse_kernelwork", "quant_reuse_sample", "quant_reuse_case_complete", "quant_reuse_screen_complete", "transport_manifest", "transport_boundary_sample", "transport_sum_sample", "transport_summary", "transport_host_overhead_sample", "transport_complete"}
    require(all(r.get("kind") in kinds and r.get("ordinal") in (None, 0, 1) for r in rows), "unknown row kind or ordinal")
    require(sum(r.get("kind") == "start" for r in rows) == 1 and sum(r.get("kind") == "complete" for r in rows) == 1, "duplicate or missing journal boundary")
    source = check_controller(window, rows)
    baseline = read_json(baseline_analysis)
    baseline_source = baseline.get("source_commit")
    baseline_input = baseline.get("input") or {}
    require(isinstance(baseline_source, str) and re.fullmatch(r"[0-9a-f]{40}", baseline_source) is not None, "accepted baseline source is not a full commit")
    require(baseline.get("candidate_accepted") is True and baseline.get("production_qualification") is False and baseline_input == {"prompt_tokens": 64, "output_tokens": 128, "context": 4096, "cut": 20, "greedy": True}, "baseline is not an accepted 64/128 full-trial case")
    accepted_turn_seconds = baseline.get("arms", {}).get("candidate", {}).get("turn_seconds", {}).get("median")
    require(finite_positive(accepted_turn_seconds), "accepted baseline candidate turn median is invalid")
    start, done = rows[0], rows[-1]
    require(start.get("scope") == SOURCE_SCOPE and start.get("implementation") == "shared-qwen-engine" and (start.get("prompt_tokens"), start.get("output_tokens"), start.get("context"), start.get("cut")) == (64, 128, 4096, 20), "operator screen identity mismatch")
    require(done.get("correctness_pass") is True and done.get("production_qualification") is False and done.get("scope") == "isolated operator parity and timing screens", "operator completion scope or status invalid")
    q = quant(rows, accepted_turn_seconds)
    t = transport(rows)
    q["materiality_baseline_source"] = t["materiality_baseline_source"] = baseline_source
    t["materiality_baseline_source"] = baseline_source
    q["controller_source_commit"] = t["controller_source_commit"] = source["source_commit"]
    return {"source_commit": source["source_commit"], "controller": source, "quant_reuse": q, "transport": t, "production_qualification": False}


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
