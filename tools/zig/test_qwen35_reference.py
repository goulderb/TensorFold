import argparse
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import qwen35_reference as ref
import torch


class ReferenceTests(unittest.TestCase):
    def test_nvfp4_nibble_order_sign_and_both_scale_levels(self):
        codes = torch.tensor(
            [[0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE]], dtype=torch.uint8
        )
        blocks = torch.tensor([[2]], dtype=torch.float8_e4m3fn)
        actual = ref.unpack_fp4(codes, blocks, torch.tensor(0.25))
        expected = torch.tensor([ref.FP4], dtype=torch.bfloat16) * 0.5
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        with self.assertRaises(ValueError):
            ref.unpack_fp4(
                codes, torch.ones((1, 2), dtype=torch.float8_e4m3fn), torch.tensor(1.0)
            )

    def test_distribution_metrics_reject_nonfinite_and_detect_perturbation(self):
        expected = np.array([1.0, 3.0, -2.0], dtype=np.float32)
        equal = ref.row_metrics(expected, expected)
        self.assertEqual(equal["rmse"], 0)
        self.assertEqual(equal["kl_reference_to_actual"], 0)
        perturbed = ref.row_metrics(np.array([4.0, 3.0, -2.0]), expected)
        self.assertFalse(perturbed["argmax_equal"])
        self.assertGreater(perturbed["kl_reference_to_actual"], 0.1)
        with self.assertRaises(ValueError):
            ref.row_metrics(np.array([1, np.nan, 3]), expected)

    def test_cli_comparison_refuses_stream_mismatch_and_missing_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            actual, reference = root / "actual", root / "reference"
            actual.mkdir()
            reference.mkdir()
            receipt = {
                "prompt": [1],
                "tokens": [2],
                "vocab": 3,
                "config_sha256": "a",
                "index_sha256": "b",
            }
            for directory in (actual, reference):
                ref.write_json(directory / "result.json", receipt)
                np.array([1, 2, 3], dtype="<f4").tofile(directory / "row-0000.f32")
            args = argparse.Namespace(
                actual=actual,
                reference=reference,
                output=root / "report.json",
                max_rmse=0,
                max_abs=0,
                max_kl=0,
                allow_argmax_difference=False,
            )
            self.assertEqual(ref.compare(args), 0)
            receipt["tokens"] = [3]
            ref.write_json(actual / "result.json", receipt)
            with self.assertRaises(ValueError):
                ref.compare(args)
            receipt["tokens"] = [2]
            ref.write_json(actual / "result.json", receipt)
            (actual / "row-0000.f32").unlink()
            with self.assertRaises(ValueError):
                ref.compare(args)
            self.assertTrue(json.loads(args.output.read_text())["pass"])


if __name__ == "__main__":
    unittest.main()
