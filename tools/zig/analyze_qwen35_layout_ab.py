"""Validate and summarize a restored single-load Qwen full-model A/B window."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import struct
from pathlib import Path

UUIDS = ["GPU-f2c0ea1e-a2be-f0b4-6c23-ffe33b537a06", "GPU-dd2a988c-fe38-24ab-233a-9e9bb3b2645a"]
FLOOR = 1536 * 1024 * 1024
VOCAB = 248320
PROMPT = 64
OUTPUT = 128
APPENDS = PROMPT + OUTPUT - 1
DECODE_APPENDS = OUTPUT - 1
ARMS = ("baseline", "candidate")
SCOPE = "single-load captured GDN layout A/B with complete canonical state and logit parity"
EXPERT_SCOPE = "single-load captured expert N8 A/B with complete canonical state and logit parity"
MECHANISMS = {
    "gdn_layout": {"prefix": "layout", "nodes_key": "expected_gdn_nodes", "nodes": 15, "scope": SCOPE},
    "expert_n8": {"prefix": "expert", "nodes_key": "expected_expert_nodes", "nodes": 20, "scope": EXPERT_SCOPE},
}


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def valid_digest(value) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def analyze(window: Path, mechanism: str = "gdn_layout"):
    require(mechanism in MECHANISMS, f"unknown A/B mechanism: {mechanism}")
    config = MECHANISMS[mechanism]
    prefix = config["prefix"]
    window = Path(window)
    receipt = read_json(window / "window.json")
    plan = read_json(window / "plan.json")
    require(receipt.get("passed") is True and receipt.get("restored") is True, "window did not pass and restore")
    require(receipt.get("clef_unchanged") is True and not receipt.get("cleanup_errors") and not receipt.get("sampler_failures"), "receipt reports Clef, cleanup, or sampler failure")
    restoration = receipt.get("restoration") or {}
    require(restoration.get("passed") is True and restoration.get("clef_unchanged") is True, "restoration evidence failed")
    source = plan.get("source_commit")
    require(isinstance(source, str) and re.fullmatch(r"[0-9a-f]{40}", source) is not None, "invalid plan source commit")
    require(receipt.get("plan", {}).get("source_commit") == source, "receipt and plan source differ")
    marker_path = window / "profile-SOURCE_COMMIT"
    require(marker_path.read_text(encoding="utf-8").strip() == source, "profile source marker differs from plan")

    matches = [x for x in plan.get("experiments", []) if x.get("name") == "profile"]
    receipt_matches = [x for x in receipt.get("plan", {}).get("experiments", []) if x.get("name") == "profile"]
    require(len(matches) == len(receipt_matches) == 1, "missing or duplicate profile pin")
    pin, receipt_pin = matches[0], receipt_matches[0]
    require(all(receipt_pin.get(k) == pin.get(k) for k in ("executable", "sha256", "input", "input_sha256")), "controller receipt profile pin differs from plan")
    executable = Path(pin["executable"])
    require(executable.is_file() and sha256_bytes(executable.read_bytes()) == pin.get("sha256"), "profile executable is missing or its current SHA256 differs from plan")
    input_path = window / "profile-input.json"
    require(input_path.is_file(), "missing copied profile input")
    input_bytes = input_path.read_bytes()
    require(sha256_bytes(input_bytes) == pin.get("input_sha256"), "copied input SHA256 differs from plan")
    profile_input = json.loads(input_bytes)
    common_shape = (len(profile_input.get("prompt", [])) == PROMPT and profile_input.get("output_tokens") == OUTPUT
                    and profile_input.get("context") == 4096 and profile_input.get("cut") == 20 and profile_input.get("forced") == [])
    if mechanism == "gdn_layout":
        mechanism_flags_valid = profile_input.get("gdn_layout_ab") is True and profile_input.get("expert_n8_ab") in (None, False)
        shape_message = "input is not the pinned greedy 64/128 GDN layout run"
    else:
        mechanism_flags_valid = profile_input.get("expert_n8_ab") is True and profile_input.get("gdn_layout_ab") in (None, False)
        mechanism_flags_valid = mechanism_flags_valid and all(profile_input.get(k) not in (True,) for k in (
            "expert_wn2_screen", "expert_n8_screen", "gdn_layout_screen", "quant_reuse_screen", "transport_screen"))
        shape_message = "input is not the pinned greedy 64/128 expert N8 run"
    require(common_shape and mechanism_flags_valid, shape_message)
    experiments = receipt.get("experiments", [])
    run = [x for x in experiments if x.get("name") == "profile"]
    require(len(run) == 1 and run[0].get("passed") is True and run[0].get("exit_code") == 0 and run[0].get("completion", {}).get("kind") == "complete", "controller records failed or incomplete profile")
    require(len(run[0].get("command", [])) >= 3 and run[0]["command"][2] == pin["input"] and run[0].get("outputs") == ["profile-results\\consumed-logits.bf16", "profile-results\\results.jsonl"], "controller command or journal output differs from plan")

    memory = receipt.get("memory", [])
    require(bool(memory), "missing controller GPU memory samples")
    for sample in memory:
        free = sample.get("free_mib", {})
        require(set(free) == set(UUIDS) and all(type(n) is int and n >= 1536 for n in free.values()), "controller UUID or memory floor failed")

    result_path = window / "profile-results" / "results.jsonl"
    rows = [json.loads(line) for line in result_path.read_text(encoding="utf-8").splitlines() if line]
    require(rows and rows[0].get("kind") == "start" and rows[-1].get("kind") == "complete", "journal boundaries invalid")
    kinds = {"start", "complete", *(f"{prefix}_{suffix}" for suffix in ("reference", "qualification", "graph_bindings", "turn", "memory", "controls", "recreated_session"))}
    require(all(r.get("kind") in kinds for r in rows), "unknown journal row kind")
    require(sum(r["kind"] == "start" for r in rows) == sum(r["kind"] == "complete" for r in rows) == 1, "duplicate start or completion")
    start, complete = rows[0], rows[-1]
    require(start.get("source_commit") == source and start.get("implementation") == "shared-qwen-engine", "journal source or implementation differs")
    require([(x.get("ordinal"), x.get("uuid")) for x in start.get("identities", [])] == list(enumerate(UUIDS)), "physical GPU UUID order differs")
    require((start.get("prompt_tokens"), start.get("output_tokens"), start.get("context"), start.get("cut")) == (PROMPT, OUTPUT, 4096, 20), "journal input shape differs")
    require(complete.get("correctness_pass") is True and complete.get("production_qualification") is False and complete.get("scope") == config["scope"] and complete.get("independent_reference_parity") == "not checked by this executable", "completion scope or independent reference disclosure invalid")
    require(not any(r.get("kind") == "failure" for r in rows), "journal contains failure row")

    ref_rows = [r for r in rows if r["kind"] == f"{prefix}_reference"]
    require(len(ref_rows) == 1, "expected exactly one layout reference")
    reference = ref_rows[0]
    require(reference.get("arm") == "baseline" and reference.get("graph") is False and reference.get("canonical_state_version") == 2 and reference.get("canonical_state_bytes", 0) > 0, "invalid eager baseline reference")
    ref_tokens = reference.get("tokens")
    require(type(ref_tokens) is list and len(ref_tokens) == OUTPUT and all(type(t) is int and 0 <= t < VOCAB for t in ref_tokens), "invalid reference tokens")
    logits_path = window / "profile-results" / "consumed-logits.bf16"
    expected_size = OUTPUT * VOCAB * 2
    require(logits_path.is_file() and logits_path.stat().st_size == expected_size, "consumed BF16 payload has wrong byte count")
    sha = hashlib.sha256()
    with logits_path.open("rb") as f:
        for index in range(OUTPUT):
            row = f.read(VOCAB * 2)
            require(len(row) == VOCAB * 2, "truncated consumed BF16 row")
            sha.update(row)
            best_id, best_value = 0, -math.inf
            for token_id, (bits,) in enumerate(struct.iter_unpack("<H", row)):
                require((bits & 0x7F80) != 0x7F80, "consumed logits contain nonfinite BF16")
                value = struct.unpack("<f", struct.pack("<I", bits << 16))[0]
                if value > best_value:
                    best_id, best_value = token_id, value
            require(best_id == ref_tokens[index], "baseline token differs from lowest-id consumed BF16 argmax")
    logits_sha = sha.hexdigest()
    require(valid_digest(reference.get("consumed_logits_sha256")) and reference["consumed_logits_sha256"] == logits_sha and valid_digest(reference.get("canonical_state_sha256")), "reference payload or canonical-state hash invalid")

    qualifications = [r for r in rows if r["kind"] == f"{prefix}_qualification"]
    require(len(qualifications) == 4 and {(r.get("arm"), r.get("graph")) for r in qualifications} == {(a, g) for a in ARMS for g in (False, True)}, "qualification roster incomplete or duplicated")
    for r in qualifications:
        require(r.get("all_rows_tokens_and_canonical_state_exact") is True, "eager/captured qualification parity failed")
    bindings = [r for r in rows if r["kind"] == f"{prefix}_graph_bindings"]
    require(len(bindings) == 2 and {r.get("rank") for r in bindings} == {0, 1}, "graph-binding roster incomplete")
    for r in bindings:
        nodes_key = config["nodes_key"]
        require(r.get(nodes_key) == config["nodes"] and r.get("baseline_verified_nodes") == config["nodes"] and r.get("candidate_verified_nodes") == config["nodes"], "graph binding node count mismatch")

    memories = [r for r in rows if r["kind"] == f"{prefix}_memory"]
    expected_slots = {(pair, order, rank) for pair in range(6) for order in range(2) for rank in range(2)}
    require(len(memories) == 24 and {(r.get("pair"), r.get("order"), r.get("rank")) for r in memories} == expected_slots, "per-turn GPU memory samples incomplete or duplicated")
    for r in memories:
        require(type(r.get("free_bytes")) is int and r["free_bytes"] >= FLOOR and r.get("floor_bytes") == FLOOR, "per-turn GPU memory floor failed")

    turns = [r for r in rows if r["kind"] == f"{prefix}_turn"]
    require(len(turns) == 12, "expected one warmup and five paired turns")
    turns.sort(key=lambda r: (r["pair"], r["order"]))
    arm_times = {a: {"turn_seconds": [], "decode_seconds": []} for a in ARMS}
    paired = []
    for pair in range(6):
        both = turns[pair * 2:pair * 2 + 2]
        expected_order = ["baseline", "candidate"] if pair % 2 == 0 else ["candidate", "baseline"]
        require([r.get("pair") for r in both] == [pair, pair] and [r.get("order") for r in both] == [0, 1] and [r.get("arm") for r in both] == expected_order, "turn pair order mismatch")
        require([r.get("warmup") for r in both] == [pair == 0] * 2, "warmup markers invalid")
        times = {}
        for r in both:
            t = r.get("times", {})
            require(r.get("graph") is True and type(t.get("prefill_ns")) is int and t["prefill_ns"] > 0 and type(t.get("decode_ns")) is int and t["decode_ns"] > 0 and type(t.get("elapsed_ns")) is int and t["elapsed_ns"] == t["prefill_ns"] + t["decode_ns"] and t.get("appends") == APPENDS, "invalid timing or append count")
            require(r.get("position") == APPENDS and r.get("graph_launches") == 2 * APPENDS and type(r.get("graph_nodes")) is int and r["graph_nodes"] > 0, "turn position, launch, or graph node count invalid")
            require(r.get("canonical_state_version") == 2 and r.get("tokens") == ref_tokens and r.get("all_rows_tokens_and_canonical_state_exact") is True, "turn token or row/state parity failed")
            require(r.get("consumed_logits_sha256") == logits_sha and r.get("canonical_state_sha256") == reference["canonical_state_sha256"], "turn consumed-logit or canonical-state hash differs from reference")
            turn = t["elapsed_ns"] / 1e9
            decode = t["decode_ns"] / 1e9
            times[r["arm"]] = turn
            if pair:
                arm_times[r["arm"]]["turn_seconds"].append(turn)
                arm_times[r["arm"]]["decode_seconds"].append(decode)
        if pair:
            paired.append((times["baseline"] - times["candidate"]) / times["baseline"])
    controls = [r for r in rows if r["kind"] == f"{prefix}_controls"]
    require(len(controls) == 2 and {r.get("arm") for r in controls} == set(ARMS), "layout control roster incomplete")
    for r in controls:
        require(all(r.get(k) is True for k in ("distinct_prompts_and_positions", "inactive_complete_state_unchanged", "reset_exact", "partial_stage_failure_poison", "uncommitted_lengths")) and r.get("canonical_state_version") == 2, "layout controls failed")
    recreated = [r for r in rows if r["kind"] == f"{prefix}_recreated_session"]
    require(len(recreated) == 1 and recreated[0].get("candidate_exact") is True and recreated[0].get("baseline_after_candidate_exact") is True, "recreated-session parity failed")

    summaries = {}
    for arm, metrics in arm_times.items():
        summaries[arm] = {}
        for name, values in metrics.items():
            summary = {"median": statistics.median(values), "range": [min(values), max(values)]}
            if name == "decode_seconds":
                rates = [DECODE_APPENDS / value for value in values]
                summary["appends_per_second"] = {"median": statistics.median(rates), "range": [min(rates), max(rates)]}
            summaries[arm][name] = summary
    median_saving = statistics.median(paired)
    spread = max(paired) - min(paired)
    return {
        "source_commit": source,
        "mechanism": mechanism,
        "executable_sha256": pin["sha256"],
        "input_sha256": pin["input_sha256"],
        "input": {"prompt_tokens": PROMPT, "output_tokens": OUTPUT, "context": 4096, "cut": 20, "greedy": True},
        "scope": f"synthetic {config['scope']}; not HTTP production performance or physical DRAM bandwidth",
        "independent_reference_gate": "FAILS; this executable does not check independent reference parity",
        "controller_minimum_free_mib": {u: min(sample["free_mib"][u] for sample in memory) for u in UUIDS},
        "arms": summaries,
        "paired_turn_savings_fraction": paired,
        "paired_turn_savings_range_fraction": [min(paired), max(paired)],
        "median_paired_turn_saving_fraction": median_saving,
        "paired_noise_spread_fraction": spread,
        "noise_assessment": "median paired saving exceeds paired spread" if median_saving > spread else "paired spread is at least the median saving",
        "all_five_pairs_faster": all(value > 0 for value in paired),
        "material_5pct_turn_screen": median_saving >= 0.05,
        "candidate_accepted": all(value > 0 for value in paired) and median_saving > spread and median_saving >= 0.05,
        "warmup_pairs_excluded": 1,
        "measured_pairs": 5,
        "production_qualification": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("window", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--mechanism", choices=tuple(MECHANISMS), default="gdn_layout")
    args = parser.parse_args()
    output = json.dumps(analyze(args.window, args.mechanism), indent=2) + "\n"
    if args.output:
        args.output.write_text(output, encoding="utf-8", newline="\n")
    print(output, end="")


if __name__ == "__main__":
    main()
