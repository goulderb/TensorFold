import argparse
import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['USE_HUB_KERNELS'] = '0'
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[3]
parser = argparse.ArgumentParser(description='Prepare the frozen Qwen numerical seam operands on CPU.')
parser.add_argument('--output', type=Path, required=True)
OUT = parser.parse_args().output.resolve()
OUT.mkdir(parents=True, exist_ok=False)
MODEL = Path('C:/Development/local-ai-server/models/ornith15-35b-nvfp4-94e431d9')
TRACE = ROOT / '.artifacts/numerical-hour-trace2-20261019'
HF = ROOT / '.artifacts/numerical-hour-full-reference-20261019/reference-stages314-moe-attention-reciprocal-trace-short'
torch.set_num_threads(2)
manifest = {'scope': 'CPU fixture only; no model creation or forward, no CUDA initialization', 'head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(), 'torch': torch.__version__, 'inputs': {}, 'tensors': {}, 'files': {}, 'sources': {}, 'cases': {}}

def digest(data):
    return hashlib.sha256(data).hexdigest()

def source(path):
    data = path.read_bytes()
    manifest['inputs'][str(path)] = {'bytes': len(data), 'sha256': digest(data)}
    return data

def save(name, data, shape, dtype):
    path = OUT / name
    path.write_bytes(data)
    manifest['files'][name] = {'bytes': len(data), 'sha256': digest(data), 'shape': shape, 'dtype': dtype}
    assert path.read_bytes() == data

index = json.loads(source(MODEL / 'model.safetensors.index.json'))['weight_map']

def tensor(name):
    path = MODEL / index[name]
    with safe_open(path, framework='pt', device='cpu') as shard:
        t = shard.get_tensor(name).contiguous()
    assert t.device.type == 'cpu' and bool(torch.isfinite(t.float()).all())
    raw = t.reshape(-1).view(torch.uint8).numpy().tobytes()
    with path.open('rb') as stream:
        header_size = int.from_bytes(stream.read(8), 'little')
        header_raw = stream.read(header_size)
    header = json.loads(header_raw)[name]
    manifest['tensors'][name] = {'shard': str(path), 'shape': list(t.shape), 'dtype': str(t.dtype), 'sha256': digest(raw), 'bytes': len(raw), 'safetensors_header_sha256': digest(header_raw), 'absolute_byte_range': [8 + header_size + p for p in header['data_offsets']]}
    return t, raw

def captured(name, size):
    data = source(TRACE / 'capture' / name)
    assert len(data) == size * 2
    t = torch.from_numpy(np.frombuffer(data, dtype='<u2').copy()).view(torch.bfloat16)
    assert bool(torch.isfinite(t.float()).all())
    save(name, data, [size], 'bf16')
    return t

x = captured('layer-03-pos-001-input_norm.bf16', 2048)
hf_x = np.frombuffer(source(HF / 'pos-001-layer-03-stage-input_layernorm.f32'), dtype='<f4')
assert np.array_equal(hf_x, x.float().numpy())
order = [2*q + j % 2 + 8*(j//2) for q in range(4) for j in range(4)]
assert sorted(order) == list(range(16))
manifest['activation_fragment_order'] = order

with torch.inference_mode():
    for short, label, n in [('q_proj', 'q_gate', 8192), ('v_proj', 'value', 512)]:
        prefix = 'model.language_model.layers.3.self_attn.' + short
        w, raw = tensor(prefix + '.weight')
        assert w.dtype == torch.float8_e4m3fn and tuple(w.shape) == (n, 2048)
        act, act_raw = tensor(prefix + '.input_scale')
        scale, scale_raw = tensor(prefix + '.weight_scale')
        act, scale = act.float().reshape(()), scale.float().reshape(())
        assert float(act) > 0 and float(scale) > 0
        save(short + '.weight.fp8', raw, [n, 2048], 'float8_e4m3fn-row-major')
        inv = 1 / act
        codes = (x.float() * inv).clamp(-448, 448).to(torch.float8_e4m3fn).view(torch.uint8).numpy()
        assert not np.any((codes & 127) == 127)
        packed = codes.reshape(-1, 16)[:, order].copy()
        save(short + '.expected-activation.row.fp8', codes.tobytes(), [2048], 'float8_e4m3fn-row-major')
        save(short + '.expected-activation.fragment.fp8', packed.tobytes(), [2048], 'float8_e4m3fn-fragment-order')
        save(short + '.scalars.f32', torch.stack([act, scale, inv, act * scale]).numpy().astype('<f4').tobytes(), [4], 'f32-act-weight-inverse-alpha')
        captured(f'layer-03-pos-001-{label}.bf16', n)
        hf_output = source(HF / f'pos-001-layer-03-stage-self_attn-{short}.f32')
        assert len(hf_output) == n * 4 and np.isfinite(np.frombuffer(hf_output, dtype='<f4')).all()
        save(short + '.hf.expected.f32', hf_output, [n], 'f32-of-bf16')
        manifest['cases'][short] = {'act': float(act), 'weight_scale': float(scale), 'inverse': float(inv), 'alpha': float(act * scale), 'rows': 1, 'n': n, 'k': 2048, 'activation_equal_to_saved_hf': True}

before_raw = source(ROOT / '.artifacts/qualification-20261018-0151/native-trace-short/pos-000-layer-13.bf16')
assert len(before_raw) == 4096
before = torch.from_numpy(np.frombuffer(before_raw, dtype='<u2').copy()).view(torch.bfloat16)
save('layer14.before.bf16', before_raw, [2048], 'bf16')
mixer = captured('layer-14-pos-000-mixer_out.bf16', 2048)
residual = (before.float() + mixer.float()).bfloat16()
assert bool(torch.isfinite(residual.float()).all())
save('layer14.residual.expected.bf16', residual.view(torch.uint8).numpy().tobytes(), [2048], 'bf16')
norm, raw = tensor('model.language_model.layers.14.post_attention_layernorm.weight')
assert norm.dtype == torch.bfloat16 and tuple(norm.shape) == (2048,)
save('layer14.norm.raw.bf16', raw, [2048], 'bf16')
save('layer14.norm.plus-one.f32', (norm.float() + 1).numpy().astype('<f4').tobytes(), [2048], 'f32')
captured('layer-14-pos-000-input.bf16', 2048)
hf_norm = source(HF / 'pos-000-layer-14-stage-post_attention_layernorm.f32')
assert len(hf_norm) == 8192 and np.isfinite(np.frombuffer(hf_norm, dtype='<f4')).all()
save('layer14.hf.expected.f32', hf_norm, [2048], 'f32-of-bf16')

for rel in ['zig/src/cuda/ck.zig', 'zig/src/cuda/qlinear.zig', 'zig/src/cuda/norms.zig', 'zig/src/cuda/modelopt.zig', 'zig/src/cuda/fp8.zig', 'zig/kernels/cuda/ck_lane.cu', 'zig/kernels/cuda/norms.cu', 'zig/kernels/cuda/qwen35_hf.cu', 'zig/kernels/cuda/qwen35_ops.cu', 'zig/src/families/qwen3_5/cuda_hf.zig', 'zig/src/families/qwen3_5/cuda_modelopt_weights.zig', 'tools/zig/qwen35_reference.py', '.artifacts/numerical-hour-trace2-20261019/instrumentation.json', '.artifacts/numerical-hour-trace2-20261019/original-cuda-engine.zig', '.artifacts/modeling_qwen3_5_moe-e0eb2810.py']:
    data = (ROOT / rel).read_bytes()
    manifest['sources'][rel] = {'sha256': digest(data), 'bytes': len(data)}
for name in ['ck_lane', 'ck_experts', 'norms', 'qwen35_hf', 'qwen35_ops', 'modelopt_pack']:
    path = ROOT / '.artifacts/qwen35-expert-n8-20261019/fatbins-sm120' / (name + '.fatbin')
    source(path)
manifest['builder_sha256'] = digest(Path(__file__).read_bytes())
manifest['cuda_initialized'] = torch.cuda.is_initialized()
assert not manifest['cuda_initialized']
(OUT / 'fixture.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8', newline='\n')
print(json.dumps({'status': 'PASS', 'files': len(manifest['files']), 'tensors': len(manifest['tensors']), 'bytes': sum(x['bytes'] for x in manifest['files'].values()), 'cuda_initialized': False}))
