import json
import tempfile
import unittest
from pathlib import Path

import torch

import qwen35_greedy_reference as greedy


def argparse_namespace(**kwargs):
    return type("Args", (), kwargs)()


class GreedyReferenceTests(unittest.TestCase):
    def test_token_sha256_matches_server_tokens_zig(self):
        ids = [1, 23, 456, 0xFFFFFFFF]
        self.assertEqual(
            greedy.token_sha256(ids),
            "59ce3a8d28d46b5fe396be42f190dd3dd26a64b88007ad56d7cd3609fd42bf8c",
        )
        self.assertEqual(
            greedy.token_sha256([]),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        )
        self.assertNotEqual(greedy.token_sha256([1, 23]), greedy.token_sha256([12, 3]))

    def test_top2_margin_from_bf16_logits(self):
        logits = torch.tensor([1.0, 5.0, 3.0, -2.0], dtype=torch.bfloat16)
        ids, scores, margin = greedy.top2(logits)
        self.assertEqual(ids, [1, 2])
        self.assertGreater(scores[0], scores[1])
        self.assertAlmostEqual(margin, scores[0] - scores[1])

    def test_top2_lowest_id_wins_ties(self):
        logits = torch.tensor([7.0, 7.0, 3.0], dtype=torch.bfloat16)
        ids, scores, margin = greedy.top2(logits)
        self.assertEqual(ids, [0, 1])
        self.assertEqual(scores[0], scores[1])
        self.assertEqual(margin, 0.0)
        self.assertEqual(int(torch.argmax(logits.float())), 0)

    def test_load_prompt_accepts_oracle_shape_and_ignores_forced(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "short.json"
            path.write_text(
                json.dumps({"prompt": [760, 6511], "forced": [11751, 13]}) + "\n",
                encoding="utf-8",
                newline="\n",
            )
            self.assertEqual(greedy.load_prompt(path), [760, 6511])

    def test_ensure_empty_output_refuses_nonempty_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "out"
            root.mkdir()
            (root / "stale.json").write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must be empty"):
                greedy.ensure_empty_output(root)

    def test_cli_validation_bounds_without_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prompt = root / "chat.json"
            prompt.write_text(
                json.dumps({"prompt": [1, 2, 3], "forced": [4]}) + "\n",
                encoding="utf-8",
                newline="\n",
            )
            output = root / "out"
            with self.assertRaisesRegex(ValueError, "max_tokens"):
                greedy.validate_args(
                    argparse_namespace(
                        model=root,
                        output=output,
                        input=[prompt],
                        source=root / "missing.py",
                        threads=4,
                        max_tokens=33,
                        device="cpu",
                    )
                )
            with self.assertRaisesRegex(ValueError, "device=cpu"):
                greedy.validate_args(
                    argparse_namespace(
                        model=root,
                        output=output,
                        input=[prompt],
                        source=root / "missing.py",
                        threads=4,
                        max_tokens=16,
                        device="cuda",
                    )
                )


if __name__ == "__main__":
    unittest.main()
