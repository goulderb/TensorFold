import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import analyze_qwen35_shared_profile as analyzer


class AnalyzerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.window = self.root / "window"
        result_dir = self.window / "profile-results"
        result_dir.mkdir(parents=True)
        source = "8a39ba55764aafad58d8dae402b287b21d0e760c"
        (self.window / "profile-SOURCE_COMMIT").write_text(source)
        identity = [{"uuid": x} for x in analyzer.UUIDS]
        receipt = {"passed": True, "restored": True, "clef_unchanged": True, "cleanup_errors": [], "sampler_failures": [],
                   "memory": [{"free_mib": {uuid: 4000 for uuid in analyzer.UUIDS}}],
                   "plan": {"source_commit": source}, "restoration": {"passed": True, "clef_unchanged": True}}
        (self.window / "window.json").write_text(json.dumps(receipt))
        p, n = 64, 128
        input_bytes = json.dumps({"prompt": list(range(p)), "output_tokens": n, "context": 4096, "cut": 20, "forced": [], "verify_default_policy": True}).encode()
        (self.window / "profile-input.json").write_bytes(input_bytes)
        (self.window / "plan.json").write_text(json.dumps({"source_commit": source, "experiments": [{"name": "profile", "input_sha256": hashlib.sha256(input_bytes).hexdigest()}]}))
        start = {"kind": "start", "source_commit": source, "implementation": "shared-qwen-engine", "identities": identity,
                 "context": 4096, "cut": 20, "prompt_tokens": p, "output_tokens": n,
                 "default_policies": {"enabled": True, "expert_up_n8": True, "gdn_layout": "key_value", "verify_expert_graphs": True, "verify_gdn_graphs": True}}
        data = bytearray()
        for i in range(n):
            row = bytearray(analyzer.VOCAB * 2)
            row[i * 2:i * 2 + 2] = b"\x80\x3f"
            data.extend(row)
        (result_dir / "consumed-logits.bf16").write_bytes(data)
        sha = hashlib.sha256(data).hexdigest()
        rows = [start]
        state = "a" * 64
        for pair in range(6):
            for order, arm in enumerate(["eager", "graph"] if pair % 2 == 0 else ["graph", "eager"]):
                graph = arm == "graph"
                appends = p + n - 1
                rows.append({"kind": "memory", "pair": pair, "order": order, "rank": 0, "free_bytes": 4_000_000_000, "floor_bytes": 1_610_612_736})
                rows.append({"kind": "memory", "pair": pair, "order": order, "rank": 1, "free_bytes": 4_000_000_000, "floor_bytes": 1_610_612_736})
                times = {"prefill_ns": 100, "decode_ns": 12700, "elapsed_ns": 12800, "appends": appends}
                rows.append({"kind": "turn", "pair": pair, "order": order, "warmup": pair == 0, "arm": arm, "graph": graph,
                             "times": times, "elapsed_ns": 12800, "steps": appends, "graph_launches": 2 * appends if graph else 0,
                             "graph_nodes": 983, "position": appends, "exact_rows_tokens_and_final_state": True,
                             "consumed_logits_sha256": sha, "persistent_state_sha256": state, "tokens": list(range(n)),
                             "default_policies": {"expert_up_n8": True, "gdn_layout": "key_value", "verify_expert_graphs": True, "verify_gdn_graphs": True,
                                                  "graph_verifiers": [{"rank": rank, "expert_baseline_nodes": 0, "expert_n8_nodes": 40 if graph else 0,
                                                                       "gdn_baseline_nodes": 0, "gdn_candidate_nodes": 30 if graph else 0} for rank in range(2)]}})
        rows.extend([{"kind": "controls", "all_consumed_rows_equal": True, "changing_tokens_and_positions": True,
                      "interleaved_session_isolation": True, "reset": True, "partial_stage_failure_poison": True, "uncommitted_lengths": True},
                     {"kind": "phase", "rank": 0, "layer": 0, "phase": "gemm", "elapsed_ms": 1.0, "captured": False, "instrumented": True},
                     {"kind": "complete", "correctness_pass": True, "production_qualification": False,
                      "default_policies": {"graph_verifiers": [{"rank": rank, "expert_baseline_nodes": 0, "expert_n8_nodes": 40, "gdn_baseline_nodes": 0, "gdn_candidate_nodes": 30} for rank in range(2)]}}])
        self.rows = rows
        self.save()

    def tearDown(self):
        self.temp.cleanup()

    def save(self):
        (self.window / "profile-results" / "results.jsonl").write_text("\n".join(json.dumps(r) for r in self.rows) + "\n")

    def reject(self, change):
        change()
        self.save()
        with self.assertRaises((ValueError, KeyError, IndexError)):
            analyzer.analyze(self.window, require_greedy=True)

    def test_valid_greedy_window_and_denominator(self):
        result = analyzer.analyze(self.window, require_greedy=True)
        self.assertEqual(result["arms"]["eager"]["samples"][0]["decode_ns_per_append"], 100)
        self.assertEqual(result["arms"]["eager"]["samples"][0]["output_tokens_per_inclusive_second"], 10_000_000)

    def test_rejects_source_mismatch(self):
        self.reject(lambda: self.rows[0].update(source_commit="wrong"))

    def test_rejects_controller_memory_floor(self):
        path = self.window / "window.json"
        receipt = json.loads(path.read_text())
        receipt["memory"][0]["free_mib"][analyzer.UUIDS[1]] = 1535
        path.write_text(json.dumps(receipt))
        with self.assertRaisesRegex(ValueError, "controller GPU memory"):
            analyzer.analyze(self.window, require_greedy=True)

    def test_rejects_missing_controller_gpu(self):
        path = self.window / "window.json"
        receipt = json.loads(path.read_text())
        del receipt["memory"][0]["free_mib"][analyzer.UUIDS[1]]
        path.write_text(json.dumps(receipt))
        with self.assertRaisesRegex(ValueError, "controller GPU memory"):
            analyzer.analyze(self.window, require_greedy=True)

    def test_rejects_missing_pair(self):
        self.reject(lambda: self.rows.remove(next(r for r in self.rows if r.get("kind") == "turn" and r.get("pair") == 5)))

    def test_rejects_stale_hash(self):
        self.reject(lambda: next(r for r in self.rows if r.get("kind") == "turn").update(consumed_logits_sha256="0" * 64))

    def test_rejects_shape(self):
        (self.window / "profile-results" / "consumed-logits.bf16").write_bytes(b"short")
        with self.assertRaises(ValueError):
            analyzer.analyze(self.window, require_greedy=True)

    def test_rejects_nonfinite(self):
        path = self.window / "profile-results" / "consumed-logits.bf16"
        data = bytearray(path.read_bytes()); data[0:2] = b"\x80\x7f"; path.write_bytes(data)
        with self.assertRaises(ValueError):
            analyzer.analyze(self.window, require_greedy=True)

    def test_rejects_bad_work_count(self):
        self.reject(lambda: next(r for r in self.rows if r.get("kind") == "turn").update(graph_launches=1))

    def test_rejects_warmup_claim(self):
        self.reject(lambda: next(r for r in self.rows if r.get("kind") == "turn" and r.get("pair") == 1).update(warmup=True))

    def test_rejects_failed_restoration(self):
        receipt = json.loads((self.window / "window.json").read_text()); receipt["restoration"]["passed"] = False
        (self.window / "window.json").write_text(json.dumps(receipt))
        with self.assertRaises(ValueError):
            analyzer.analyze(self.window, require_greedy=True)

    def test_rejects_forced_baseline(self):
        (self.window / "profile-input.json").write_text(json.dumps({"forced": [9]}))
        with self.assertRaises(ValueError):
            analyzer.analyze(self.window, require_greedy=True)

    def test_rejects_failed_control(self):
        self.reject(lambda: next(r for r in self.rows if r.get("kind") == "controls").update(reset=False))

    def test_rejects_duplicate_memory_rank(self):
        memories = [r for r in self.rows if r.get("kind") == "memory"]
        self.reject(lambda: memories[-1].update(rank=memories[-2]["rank"]))

    def test_rejects_greedy_token_not_argmax(self):
        self.reject(lambda: next(r for r in self.rows if r.get("kind") == "turn").update(tokens=[127] + list(range(1, 128))))

    def test_rejects_default_policy_not_initialized(self):
        self.reject(lambda: next(r for r in self.rows if r.get("kind") == "start")["default_policies"].update(expert_up_n8=False))

    def test_rejects_missing_default_capture_counter(self):
        entry = next(r for r in self.rows if r.get("kind") == "turn" and r.get("arm") == "graph")["default_policies"]["graph_verifiers"][0]
        self.reject(lambda: entry.pop("expert_n8_nodes"))

    def test_rejects_wrong_default_function_arm(self):
        entry = next(r for r in self.rows if r.get("kind") == "turn" and r.get("arm") == "graph")["default_policies"]["graph_verifiers"][0]
        self.reject(lambda: entry.update(expert_baseline_nodes=20))


if __name__ == "__main__":
    unittest.main()
