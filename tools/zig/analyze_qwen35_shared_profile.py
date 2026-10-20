"""Validate a restored shared-Qwen eager/captured profile window and summarize timings."""
import argparse
import hashlib
import json
import math
import statistics
import struct
from pathlib import Path

VOCAB = 248320
UUIDS = ["GPU-f2c0ea1e-a2be-f0b4-6c23-ffe33b537a06", "GPU-dd2a988c-fe38-24ab-233a-9e9bb3b2645a"]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify_default_policies(profile_input, start, turns, complete):
    if profile_input.get("verify_default_policy", False) is not True:
        return
    initial = start.get("default_policies")
    require(isinstance(initial, dict) and all(initial.get(k) is True for k in ("enabled", "expert_up_n8", "verify_expert_graphs", "verify_gdn_graphs")) and initial.get("gdn_layout") == "key_value", "default HF policy was not initialized before session profiling")
    final_counts = None
    for turn in turns:
        policy = turn.get("default_policies")
        require(isinstance(policy, dict) and all(policy.get(k) is True for k in ("expert_up_n8", "verify_expert_graphs", "verify_gdn_graphs")) and policy.get("gdn_layout") == "key_value", "default policy binding missing from profile turn")
        ranks = policy.get("graph_verifiers")
        require(isinstance(ranks, list) and [entry.get("rank") for entry in ranks if isinstance(entry, dict)] == [0, 1], "default graph verifier rank bindings missing")
        for entry in ranks:
            counts = [entry.get(k) for k in ("expert_baseline_nodes", "expert_n8_nodes", "gdn_baseline_nodes", "gdn_candidate_nodes")]
            require(all(type(v) is int and v >= 0 for v in counts), "default graph verifier counts missing or invalid")
            expert_base, expert_n8, gdn_base, gdn_candidate = counts
            require(expert_base == 0 and expert_n8 % 20 == 0, "captured expert graph did not bind N8 for every 20-layer graph")
            require(gdn_base == 0 and gdn_candidate % 15 == 0, "captured GDN graph did not bind key/value for every 15-node graph")
            require(expert_n8 // 20 == gdn_candidate // 15, "expert and GDN verifier graph multiplicities differ")
            if turn["arm"] == "graph":
                require(expert_n8 > 0 and gdn_candidate > 0, "captured graph turn verified no default-policy nodes")
        if turn["arm"] == "graph":
            final_counts = ranks
    completion = complete.get("default_policies")
    require(isinstance(completion, dict) and completion.get("graph_verifiers") == final_counts, "completion default graph counters missing or stale")


def analyze(window, require_greedy=False):
    receipt = read_json(window / "window.json")
    plan = read_json(window / "plan.json")
    require(receipt.get("passed") is True and receipt.get("restored") is True, "window did not pass and restore")
    require(receipt.get("clef_unchanged") is True and not receipt.get("cleanup_errors") and not receipt.get("sampler_failures"), "receipt reports Clef, cleanup, or sampler failure")
    samples = receipt.get("memory", [])
    require(samples, "missing controller GPU memory samples")
    for sample in samples:
        free = sample.get("free_mib", {})
        require(set(free) == set(UUIDS) and all(type(value) is int and value >= 1536 for value in free.values()), "controller GPU memory floor or identities failed")
    restoration = receipt.get("restoration") or {}
    require(restoration.get("passed") is True and restoration.get("clef_unchanged") is True, "restoration evidence failed")
    require(plan.get("source_commit") == receipt.get("plan", {}).get("source_commit"), "plan source differs from receipt")
    require((window / "profile-SOURCE_COMMIT").read_text(encoding="utf-8").strip() == plan["source_commit"], "profile source marker differs from plan")
    rows = [json.loads(line) for line in (window / "profile-results" / "results.jsonl").read_text(encoding="utf-8").splitlines() if line]
    require(rows and rows[0].get("kind") == "start" and rows[-1].get("kind") == "complete", "missing start or final complete row")
    start, complete = rows[0], rows[-1]
    require(start.get("source_commit") == plan["source_commit"] and start.get("implementation") == "shared-qwen-engine", "wrong source or implementation")
    require(start.get("identities") and [v.get("uuid") for v in start["identities"]] == UUIDS, "physical GPU UUID order differs")
    require((start.get("context"), start.get("cut")) == (4096, 20), "unexpected context or cut")
    require(complete.get("correctness_pass") is True and complete.get("production_qualification") is False, "completion did not pass bounded correctness")
    require(not any(r.get("kind") == "failure" for r in rows), "profile contains failure row")
    require(start.get("prompt_tokens", 0) > 0 and start.get("output_tokens", 0) >= 2, "invalid input shape")
    p, n = start["prompt_tokens"], start["output_tokens"]
    require(p + n <= start["context"], "prompt plus output exceeds context")
    forced_path = window / "profile-input.json"
    require(forced_path.exists(), "missing pinned profile input")
    input_bytes = forced_path.read_bytes()
    profile_input = json.loads(input_bytes)
    experiments = plan.get("experiments", [])
    profile_experiment = next((x for x in experiments if x.get("name") == "profile"), None)
    require(profile_experiment is not None and hashlib.sha256(input_bytes).hexdigest() == profile_experiment.get("input_sha256"), "profile input SHA256 differs from plan")
    forced = profile_input.get("forced", [])
    require((len(profile_input.get("prompt", [])), profile_input.get("output_tokens"), profile_input.get("context"), profile_input.get("cut")) == (start["prompt_tokens"], n, 4096, 20), "profile input shape differs from start row")
    greedy = not forced
    if require_greedy:
        require(greedy and (p, n) == (64, 128), "--require-greedy requires 64 prompt tokens and 128 greedy outputs")
    else:
        require(len(forced) == n, "non-greedy window must explicitly force all output tokens")
    memory = [r for r in rows if r.get("kind") == "memory"]
    require(len(memory) == 24, "expected two GPU memory checks for each of 12 turns")
    require({(r.get("pair"), r.get("order"), r.get("rank")) for r in memory} == {(pair, order, rank) for pair in range(6) for order in range(2) for rank in range(2)}, "memory checks do not cover each turn/rank exactly once")
    for r in memory:
        require(r.get("rank") in (0, 1) and r.get("free_bytes", 0) >= 1536 * 1024 * 1024 and r.get("floor_bytes", 0) >= 1536 * 1024 * 1024, "GPU memory floor failed")
    turns = [r for r in rows if r.get("kind") == "turn"]
    require(len(turns) == 12, "expected 6 eager/captured pairs")
    turns.sort(key=lambda r: (r["pair"], r["order"]))
    verify_default_policies(profile_input, start, turns, complete)
    expected_state = None
    logits_path = window / "profile-results" / "consumed-logits.bf16"
    raw_size = logits_path.stat().st_size
    require(raw_size == n * VOCAB * 2, "consumed BF16 logits shape mismatch")
    sha = hashlib.sha256()
    with logits_path.open("rb") as f:
        for output_index in range(n):
            row = f.read(VOCAB * 2)
            require(len(row) == VOCAB * 2, "truncated consumed BF16 row")
            sha.update(row)
            best_id, best_value = 0, -math.inf
            for token_id, (bits,) in enumerate(struct.iter_unpack("<H", row)):
                require((bits & 0x7F80) != 0x7F80, "consumed logits contain nonfinite BF16")
                value = struct.unpack("<f", struct.pack("<I", bits << 16))[0]
                if value > best_value:
                    best_id, best_value = token_id, value
            if greedy:
                require(best_id == turns[0]["tokens"][output_index], "reported greedy token differs from consumed BF16 row argmax")
    logit_sha = sha.hexdigest()
    expected_tokens = None
    for pair in range(6):
        both = turns[pair * 2:pair * 2 + 2]
        order = ["eager", "graph"] if pair % 2 == 0 else ["graph", "eager"]
        require([r.get("pair") for r in both] == [pair, pair] and [r.get("order") for r in both] == [0, 1], "pair ordering malformed")
        require([r.get("arm") for r in both] == order, "arms do not alternate")
        require([r.get("warmup") for r in both] == [pair == 0] * 2, "warmup marker mismatch")
        for r in both:
            t = r.get("times", {})
            require(all(isinstance(t.get(k), int) and t[k] > 0 for k in ("prefill_ns", "decode_ns", "elapsed_ns")), "invalid timing")
            require(t["elapsed_ns"] == t["prefill_ns"] + t["decode_ns"] and t.get("appends") == p + n - 1 and r.get("steps") == p + n - 1, "timing/appends incoherent")
            require(r.get("position") == p + n - 1 and r.get("tokens") and len(r["tokens"]) == n, "position or output token count mismatch")
            require(all(type(token) is int and 0 <= token < VOCAB for token in r["tokens"]), "invalid output token")
            if not greedy:
                require(r["tokens"] == forced, "forced output tokens differ from pinned input")
            require(expected_tokens is None or r["tokens"] == expected_tokens, "output tokens differ across turns")
            expected_tokens = r["tokens"]
            require(r.get("exact_rows_tokens_and_final_state") is True and r.get("consumed_logits_sha256") == logit_sha, "row parity or consumed logits hash mismatch")
            state = r.get("persistent_state_sha256", "")
            require(len(state) == 64 and all(c in "0123456789abcdef" for c in state), "invalid persistent state SHA256")
            require(expected_state is None or state == expected_state, "persistent state differs across turns")
            expected_state = state
            require(r.get("graph_launches") == (2 * (p + n - 1) if r["arm"] == "graph" else 0), "graph launch count mismatch")
            if r["arm"] == "graph":
                require(r.get("graph_nodes", 0) > 0, "missing graph node count")
            else:
                require(type(r.get("graph_nodes")) is int and r["graph_nodes"] >= 0, "invalid cached graph node count")
        require(both[0]["persistent_state_sha256"] == both[1]["persistent_state_sha256"], "paired final state differs")
    require(len(set(expected_tokens)) > 1, "output tokens do not evolve")
    controls = [r for r in rows if r.get("kind") == "controls"]
    require(len(controls) == 1 and all(controls[0].get(k) is True for k in ("all_consumed_rows_equal", "changing_tokens_and_positions", "interleaved_session_isolation", "reset", "partial_stage_failure_poison", "uncommitted_lengths")), "control checks incomplete")
    phases = [r for r in rows if r.get("kind") == "phase"]
    require(phases and all(r.get("captured") is False and r.get("instrumented") is True and math.isfinite(r.get("elapsed_ms", -1)) and r["elapsed_ms"] >= 0 for r in phases), "invalid eager phase instrumentation")
    categories = {}
    for r in phases:
        categories.setdefault(r["phase"], []).append(r["elapsed_ms"])
    result = {"source_commit": plan["source_commit"], "implementation": "shared-qwen-engine", "input": {"prompt_tokens": p, "output_tokens": n, "greedy": greedy}, "measured_pairs": 5,
              "scope": "synthetic C=1 shared-engine eager/captured profile; not HTTP throughput or promotion evidence",
              "minimum_sampled_free_mib": {uuid: min(sample["free_mib"][uuid] for sample in samples) for uuid in UUIDS},
              "arms": {}, "phase_instrumentation": {"scope": "eager instrumented append categories; not a matched kernel profile and not critical-path utilization", "categories": {}}}
    for arm in ("eager", "graph"):
        samples = []
        for pair in range(1, 6):
            r = next(x for x in turns if x["pair"] == pair and x["arm"] == arm)
            t = r["times"]
            samples.append({"prefill_seconds": t["prefill_ns"] / 1e9, "decode_ns_per_append": t["decode_ns"] / (n - 1),
                            "between_first_logit_and_last_output_seconds": t["decode_ns"] / 1e9,
                            "inclusive_turn_seconds": t["elapsed_ns"] / 1e9,
                            "output_tokens_per_inclusive_second": n * 1e9 / t["elapsed_ns"]})
        result["arms"][arm] = {"samples": samples, "summary": {k: {"median": statistics.median(x[k] for x in samples), "range": [min(x[k] for x in samples), max(x[k] for x in samples)]} for k in samples[0]}}
    for name, values in categories.items():
        result["phase_instrumentation"]["categories"][name] = {"sample_count": len(values), "elapsed_ms": {"median": statistics.median(values), "range": [min(values), max(values)]}}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("window", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-greedy", action="store_true")
    args = parser.parse_args()
    output = json.dumps(analyze(args.window, args.require_greedy), indent=2) + "\n"
    if args.output:
        args.output.write_text(output, encoding="utf-8", newline="\n")
    print(output, end="")


if __name__ == "__main__":
    main()
