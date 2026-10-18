import argparse
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import qwen35_reference as ref
import torch
import torch.nn.functional as F


def tensor_store(tensors):
    store = ref.Store.__new__(ref.Store)
    store.get = tensors.__getitem__
    return store


class ReferenceTests(unittest.TestCase):
    def test_fp8_projection_scales_after_fp32_reduction(self):
        x = torch.tensor([[1.0, -0.66796875]], dtype=torch.bfloat16)
        weight = torch.tensor([[1.0, 1.5]], dtype=torch.float8_e4m3fn)
        input_scale, weight_scale = torch.tensor(0.03), torch.tensor(0.07)
        store = tensor_store(
            {
                "p.weight": weight,
                "p.input_scale": input_scale,
                "p.weight_scale": weight_scale,
            }
        )
        quantized = (x.float() * input_scale.reciprocal()).clamp(-448, 448)
        quantized = quantized.to(torch.float8_e4m3fn).float()
        dot = sum(float(a) * float(b) for a, b in zip(quantized[0], weight.float()[0]))
        expected = torch.tensor([[dot * float(input_scale * weight_scale)]]).bfloat16()
        early = F.linear(
            (quantized * input_scale).bfloat16(),
            (weight.float() * weight_scale).bfloat16(),
        )
        self.assertFalse(torch.equal(early, expected))
        torch.testing.assert_close(store.linear("p", x), expected, rtol=0, atol=0)

    def test_fp8_reciprocal_midpoint_is_frozen(self):
        x = torch.tensor([[0.84375]], dtype=torch.bfloat16)
        scale = torch.tensor(0.7941176295280457)
        reciprocal = (x.float() * (1 / scale)).to(torch.float8_e4m3fn).float()
        division = (x.float() / scale).to(torch.float8_e4m3fn).float()
        self.assertEqual(reciprocal.item(), 1.125)
        self.assertEqual(division.item(), 1.0)
        store = tensor_store(
            {
                "p.weight": torch.ones((1, 1), dtype=torch.float8_e4m3fn),
                "p.input_scale": scale,
                "p.weight_scale": torch.tensor(1.0),
            }
        )
        expected = (reciprocal * scale).bfloat16()
        torch.testing.assert_close(store.linear("p", x), expected, rtol=0, atol=0)

    def test_nvfp4_projection_global_scale_after_cancellation(self):
        x = torch.tensor([[1.0, -0.66796875] + [0.0] * 14], dtype=torch.bfloat16)
        packed = torch.tensor([[0x32] + [0] * 7], dtype=torch.uint8)
        scale = torch.tensor(1.003)
        store = tensor_store(
            {
                "p.weight": packed,
                "p.weight_scale": torch.ones((1, 1), dtype=torch.float8_e4m3fn),
                "p.weight_scale_2": scale,
            }
        )
        expected = torch.tensor([[(1.0 - 0.66796875 * 1.5) * float(scale)]]).bfloat16()
        early_weight = torch.tensor([[1.0, 1.5] + [0.0] * 14]) * scale
        self.assertFalse(torch.equal(F.linear(x, early_weight.bfloat16()), expected))
        torch.testing.assert_close(store.linear("p", x), expected, rtol=0, atol=0)

    def test_expert_operand_keeps_linear_outputs_plain_and_selects_expert(self):
        tensors = {
            f"experts.7.{projection}.weight": torch.full(
                (2, 3), value, dtype=torch.bfloat16
            )
            for projection, value in (
                ("gate_proj", 1.0),
                ("up_proj", 2.0),
                ("down_proj", 3.0),
            )
        }
        store = tensor_store(tensors)
        weights = ref.ExpertWeights(store, "experts", ("gate_proj", "up_proj"))
        operand = weights[torch.tensor(7)]
        self.assertNotIsInstance(operand, torch.Tensor)
        x = torch.ones((2, 1, 3), dtype=torch.bfloat16)
        expected = torch.tensor([3.0, 3.0, 6.0, 6.0], dtype=torch.bfloat16).expand(
            2, 1, 4
        )
        for actual in (
            F.linear(x, operand),
            F.linear(input=x, weight=operand, bias=None),
        ):
            self.assertIs(type(actual), torch.Tensor)
            self.assertEqual(actual.dtype, torch.bfloat16)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            gate, up = actual.chunk(2, dim=-1)
            self.assertEqual(gate.shape, (2, 1, 2))
            self.assertEqual(up.shape, (2, 1, 2))
        down = ref.ExpertWeights(store, "experts", ("down_proj",))[7]
        self.assertEqual(F.linear(x, down).shape, (2, 1, 2))
        with self.assertRaises(ValueError):
            F.linear(x, operand, torch.zeros(4, dtype=torch.bfloat16))
        with self.assertRaises(TypeError):
            torch.add(x, operand)
        for expert in (-1, 256):
            with self.assertRaises(ValueError):
                weights[expert]

    def test_bf16_projection_retains_dtype_and_output_chunking(self):
        x = torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16)
        weight = torch.ones((4097, 2), dtype=torch.bfloat16)
        actual = tensor_store({"p.weight": weight}).linear("p", x)
        self.assertEqual(actual.shape, (1, 4097))
        self.assertEqual(actual.dtype, torch.bfloat16)
        torch.testing.assert_close(actual, F.linear(x, weight), rtol=0, atol=0)

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
