import collections
import json
import pathlib
import struct
import sys


def census(folder):
    folder = pathlib.Path(folder)
    index = json.loads((folder / 'model.safetensors.index.json').read_text())['weight_map']
    recipes = json.loads((folder / 'config.json').read_text())['quantization_config']['quantized_layers']
    tensors = {}
    sizes = {'U8': 1, 'F8_E4M3': 1, 'U16': 2, 'U32': 4, 'BF16': 2, 'F32': 4}
    files = sorted(set(index.values()))
    for filename in files:
        path = folder / filename
        with path.open('rb') as f:
            length, = struct.unpack('<Q', f.read(8))
            header = json.loads(f.read(length))
        end = 0
        for name, t in header.items():
            if name == '__metadata__':
                continue
            assert name not in tensors, name
            assert index[name] == filename, name
            count = 1
            for dim in t['shape']:
                count *= dim
            begin, stop = t['data_offsets']
            assert stop - begin == count * sizes[t['dtype']], name
            assert begin == end, (name, begin, end)
            end = stop
            tensors[name] = t
        assert path.stat().st_size == 8 + length + end, filename
    assert set(tensors) == set(index)
    formats = collections.Counter()
    used = set()
    max_scales = 0
    def check(name, dtype, shape=None):
        t = tensors[name]
        assert t['dtype'] == dtype, (name, t)
        if shape is not None:
            assert t['shape'] == shape, (name, t, shape)
        used.add(name)
        return t
    for name, t in tensors.items():
        if not name.endswith('.weight'):
            continue
        module = name[:-7]
        key = module.split('.experts.')[0] + '.experts' if '.experts.' in module else module
        rec = recipes.get(key)
        if not rec:
            assert t['dtype'] in ('BF16', 'F32'), (name, t)
            continue
        algo = rec['quant_algo']
        formats[algo] += 1
        n, k = t['shape']
        if algo == 'MXFP8':
            check(name, 'F8_E4M3')
            check(module + '.weight_scale', 'U8', [n, k // 32])
            assert k % 32 == 0
        elif algo == 'FP8_PB_WO':
            check(name, 'F8_E4M3')
            check(module + '.weight_scale', 'F32', [(n + 127) // 128, 1, (k + 127) // 128, 1])
        else:
            assert algo in ('NVFP4', 'W4A16_NVFP4')
            check(name, 'U8')
            g = check(module + '.weight_scale_2', 'F32')
            assert g['shape'] in ([], [1]), (module, g)
            assert k * 2 % 16 == 0
            c = k * 2 // 16
            if rec.get('weight_scale_encoding') == 'csf_32x2_v1':
                assert n % 32 == 0
                max_scales = max(max_scales, n * c)
                check(module + '.weight_scale.csf32_bases', 'U8', [n])
                check(module + '.weight_scale.csf32_codes', 'U8', [n * c // 2])
                tiles = ((n + 127) // 128) * ((c + 3) // 4)
                pages = (tiles + 31) // 32
                check(module + '.weight_scale.csf32_page_bases', 'U32', [pages])
                check(module + '.weight_scale.csf32_offsets', 'U16', [n // 32 * ((c + 1) // 2) + pages])
                check(module + '.weight_scale.csf32_exceptions', 'U16')
            else:
                check(module + '.weight_scale', 'F8_E4M3', [n, c])
            if module + '.input_scale' in tensors:
                a = check(module + '.input_scale', 'F32')
                assert a['shape'] in ([], [1])
    assert not [n for n in tensors if ('weight_scale' in n or 'input_scale' in n) and n not in used]
    special = collections.Counter()
    for name, t in tensors.items():
        if 'post_attention_layernorm' in name:
            assert t['dtype'] in ('F32', 'BF16')
            special['post_attention_layernorm ' + t['dtype']] += 1
        if '.hc_' in name and name.endswith(('_base', '_scale')):
            assert t['dtype'] in ('F32', 'BF16')
            special['hc base/scale ' + t['dtype']] += 1
        if '_conv1d.weight' in name:
            assert t['dtype'] == 'BF16'
            special['conv1d BF16'] += 1
    print(json.dumps({'shards': len(files), 'tensors': len(tensors), 'recipes': len(recipes), 'linears': dict(formats), 'nonquantized_tensors': len(tensors) - len(used), 'special': dict(special), 'max_csf_scratch_bytes': max_scales, 'input_scales': sum(n.endswith('.input_scale') for n in tensors)}, indent=2))


if __name__ == '__main__':
    census(sys.argv[1])
