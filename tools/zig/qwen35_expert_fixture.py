"""Pack one checkpoint layer and a saved native row for the expert seam, CPU only.

The packing indices adapt tf_pack_fp4_experts in zig/kernels/cuda/modelopt_pack.cu.
The saved outputs are a native regression control, not an independent math oracle.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np


def digest(data):
    return hashlib.sha256(data).hexdigest()


class Store:
    def __init__(self, root):
        self.root = root
        self.index = json.loads((root / "model.safetensors.index.json").read_text())
        self.headers = {}
        self.reads = {}

    def read(self, name, shape, dtype):
        filename = self.index["weight_map"][name]
        if filename not in self.headers:
            with (self.root / filename).open("rb") as stream:
                length = struct.unpack("<Q", stream.read(8))[0]
                if length > 32 << 20:
                    raise ValueError("oversized safetensors header")
                self.headers[filename] = (8 + length, json.loads(stream.read(length)))
        offset, header = self.headers[filename]
        meta = header[name]
        if meta["shape"] != shape or meta["dtype"] not in dtype:
            raise ValueError(f"unexpected tensor metadata: {name}")
        begin, end = meta["data_offsets"]
        with (self.root / filename).open("rb") as stream:
            stream.seek(offset + begin)
            data = stream.read(end - begin)
        if len(data) != end - begin:
            raise ValueError(f"truncated tensor: {name}")
        self.reads[name] = {"file": filename, "bytes": len(data), "sha256": digest(data)}
        return data

    def projection(self, name, n, k):
        codes = np.frombuffer(self.read(name + ".weight", [n, k // 2], {"U8"}), dtype=np.uint8).reshape(n, k // 2)
        scales = np.frombuffer(self.read(name + ".weight_scale", [n, k // 16], {"F8_E4M3", "F8_E4M3FN"}), dtype=np.uint8).reshape(n, k // 16)
        scalar = self.read(name + ".weight_scale_2", [], {"F32"})
        if not np.isfinite(struct.unpack("<f", scalar)[0]):
            raise ValueError("nonfinite expert scale")
        return pack(codes, scales), scalar


def indices(n, k):
    cb = np.arange(n // 32)[:, None, None]
    g = np.arange(k // 32)[None, :, None]
    w = np.arange(128)[None, None, :]
    rows = cb * 32 + (w % 4) * 8 + w // 16
    cols = g * 32 + ((w // 4) % 4) * 4
    sw = np.arange(64)[None, None, :]
    sr = cb * 32 + ((sw // 2) % 4) * 8 + (sw // 16) * 2 + sw % 2
    sc = g * 2 + (sw // 8) % 2
    return rows, cols, sr, sc


def pack(codes, scales):
    n, half_k = codes.shape
    k = 2 * half_k
    rows, cols, sr, sc = indices(n, k)
    result = np.zeros((n // 32, k // 32, 144), dtype="<u4")
    for h in range(2):
        for q in range(4):
            col = cols + 16 * h + q
            byte = codes[rows, col // 2]
            nibble = (byte >> (4 * (col % 2))) & 15
            result[..., :128] |= nibble.astype("<u4") << (4 * (2 * h + q // 2 + 4 * (q % 2)))
    scale_bytes = scales[sr, sc].reshape(n // 32, k // 32, 16, 4)
    for x in range(4):
        result[..., 128:] |= scale_bytes[..., x].astype("<u4") << (8 * x)
    return result


def roundtrip():
    rng = np.random.default_rng(0x3515)
    codes = rng.integers(0, 256, (64, 64), dtype=np.uint8)
    scales = rng.integers(0, 256, (64, 8), dtype=np.uint8)
    packed = pack(codes, scales)
    restored = np.zeros_like(codes)
    rows, cols, sr, sc = indices(64, 128)
    for h in range(2):
        for q in range(4):
            col = cols + h * 16 + q
            nibble = ((packed[..., :128] >> (4 * (2 * h + q // 2 + 4 * (q % 2)))) & 15).astype(np.uint8)
            rr, cc = np.broadcast_arrays(rows, col)
            np.bitwise_or.at(restored, (rr, cc // 2), nibble << (4 * (col % 2)).astype(np.uint8))
    recovered_scales = np.empty_like(scales)
    raw_scales = np.stack([((packed[..., 128:] >> (8 * x)) & 255).astype(np.uint8) for x in range(4)], axis=-1).reshape(2, 4, 64)
    recovered_scales[sr, sc] = raw_scales
    np.testing.assert_array_equal(codes, restored)
    np.testing.assert_array_equal(scales, recovered_scales)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("capture", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--router", action="store_true")
    args = parser.parse_args()
    roundtrip()
    old = json.loads((args.capture.parent / "component-comparison.json").read_text())
    names = {"input.bin": ("pos-000-input.bf16", 4096), "hidden.bin": ("pos-000-expert_hidden.bf16", 8192), "pairs.bin": ("pos-000-pairs.f32", 65536)}
    if args.router:
        for position in range(6):
            for label, suffix, size in [("input", "input.bf16", 4096), ("picks", "picks.i32", 32), ("weights", "weights.f32", 36)]:
                names[f"route-{position:03d}-{label}.bin"] = (f"pos-{position:03d}-{suffix}", size)
    saved = {}
    for filename, size in [*names.values(), ("pos-000-picks.i32", 32)]:
        path = args.capture / filename
        data = path.read_bytes()
        if len(data) != size or digest(data) != old["capture_sha256"][path.name]:
            raise ValueError(f"saved trace pin differs: {path.name}")
        saved[filename] = data
    picks = np.frombuffer(saved["pos-000-picks.i32"], dtype="<i4").tolist()
    if len(set(picks)) != 8 or not all(0 <= v < 256 for v in picks):
        raise ValueError("invalid captured expert IDs")
    order = picks + [e for e in range(256) if e not in picks]
    args.output.mkdir(parents=True, exist_ok=False)
    for out, (source, _) in names.items():
        (args.output / out).write_bytes(saved[source])
    store = Store(args.model)
    with (args.output / "up.bin").open("wb") as up, (args.output / "down.bin").open("wb") as down, (args.output / "up-scales.bin").open("wb") as us, (args.output / "down-scales.bin").open("wb") as ds:
        for expert in order:
            pre = f"model.language_model.layers.0.mlp.experts.{expert}"
            gate, gs = store.projection(pre + ".gate_proj", 512, 2048)
            act, acts = store.projection(pre + ".up_proj", 512, 2048)
            combined = np.stack((gate, act), axis=2)
            up.write(combined.tobytes())
            us.write(gs + acts)
            weights, scale = store.projection(pre + ".down_proj", 2048, 512)
            down.write(weights.tobytes())
            ds.write(scale)
    if args.router:
        prefix = "model.language_model.layers.0.mlp."
        (args.output / "router.bin").write_bytes(store.read(prefix + "gate.weight", [256, 2048], {"BF16"}))
        (args.output / "shared-gate.bin").write_bytes(store.read(prefix + "shared_expert_gate.weight", [1, 2048], {"BF16"}))
    manifest = {"scope": "Ornith layer0 real weights and captured position0 input; bank0 maps captured route; other banks are controlled route substitutions", "expert_order": order, "capture_pins": old["capture_sha256"], "source_tensors": store.reads, "generator_sha256": digest(Path(__file__).read_bytes()), "packing_source_sha256": digest((Path(__file__).resolve().parents[2] / "zig/kernels/cuda/modelopt_pack.cu").read_bytes()), "files": {p.name: {"bytes": p.stat().st_size, "sha256": digest(p.read_bytes())} for p in sorted(args.output.iterdir())}}
    (args.output / "manifest.json").write_bytes((json.dumps(manifest, indent=2) + "\n").encode())
    print(json.dumps({"files": manifest["files"], "captured_experts": picks, "cpu_roundtrip": "pass", "gpu_initialized": False}))


if __name__ == "__main__":
    main()
