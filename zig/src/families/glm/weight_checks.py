import argparse
import pathlib
import subprocess


ZIG = 'C:/Users/developer/tools/zig-0.17.0/zig-x86_64-windows-0.17.0/zig.exe'
BASE = pathlib.Path(__file__).resolve().parents[4]


def command(root, mode, linux=False, filter_text=None):
    args = [ZIG, mode, '--cache-dir', '.zig-cache-loader-sol', '--global-cache-dir', '.zig-cache-loader-sol-global', '-lc']
    if linux:
        args += ['-target', 'aarch64-linux-gnu']
        if mode == 'test':
            args += ['--test-no-exec']
    if filter_text:
        args += ['--test-filter', filter_text]
    if mode == 'build-exe':
        args += ['-O', 'ReleaseFast', '-femit-bin=.zig-cache-loader-sol/' + ('weight-compile' if linux else 'weight-walk.exe')]
    imports = ['core', 'cuda', 'canon', 'lanes', 'heat', 'cudnn', 'vision_sdpa', 'vision_ops', 'image_codec_headers']
    for name in imports:
        args += ['--dep', name]
    args += ['-Mroot=' + root]
    modules = [
        ('core', 'zig/src/core/root.zig', ['lanes', 'tokenizer']),
        ('cuda', 'zig/src/cuda/root.zig', ['stagger', 'kernel_options']),
        ('canon', 'zig/src/cluster/canon.zig', []),
        ('lanes', 'zig/src/core/lanes/lanes.zig', []),
        ('heat', 'zig/src/core/heat.zig', []),
        ('tokenizer', 'zig/src/core/tokenizer/tokenizer.zig', []),
        ('stagger', 'zig/src/core/stagger.zig', []),
        ('kernel_options', 'zig/src/families/glm/fixtures/weight-kernel-options.zig', []),
        ('cudnn', 'zig/src/cuda/glm_vision_cudnn.zig', ['cuda', 'cudnn_headers']),
        ('vision_sdpa', 'zig/src/cuda/glm_vision_sdpa.zig', ['cuda', 'cudnn_headers']),
        ('vision_ops', 'zig/src/cuda/glm_vision_ops.zig', ['cuda']),
        ('cudnn_headers', 'zig/src/cuda/vision_headers/cudnn_headers.zig', []),
        ('image_codec_headers', 'zig/src/cuda/vision_headers/image_codec_headers.zig', []),
    ]
    for name, path, deps in modules:
        if linux:
            args += ['-target', 'aarch64-linux-gnu']
        for dep in deps:
            args += ['--dep', dep]
        args += ['-M' + name + '=' + path]
    return args


def main():
    p = argparse.ArgumentParser()
    p.add_argument('step', choices=['test', 'regression', 'family', 'linux', 'linux-loader', 'linux-entry', 'walk'])
    p.add_argument('checkpoint', nargs='?')
    ns = p.parse_args()
    root = 'zig/src/families/glm/'
    if ns.step == 'regression':
        for file in ['cuda_checkpoint.zig', 'cuda_weight_precision.zig', 'cuda_weights_projs.zig', 'cuda_vision_weights.zig', 'cuda_vision_engine_memory.zig']:
            args = command(root + file, 'test')
            print(subprocess.list2cmdline(args), flush=True)
            subprocess.run(args, cwd=BASE, check=True)
        return
    if ns.step == 'test':
        args = command(root + 'cuda_weight_source_test.zig', 'test', filter_text='mixed loader')
    elif ns.step == 'family':
        args = command(root + 'cuda_weights.zig', 'test')
    elif ns.step == 'linux-entry':
        args = command(root + 'weight_compile.zig', 'build-exe', linux=True)
    elif ns.step == 'linux-loader':
        args = command(root + 'cuda_weight_source_test.zig', 'test', linux=True, filter_text='mixed loader')
    elif ns.step == 'linux':
        args = command(root + 'cuda.zig', 'test', linux=True)
    else:
        args = command(root + 'weight_walk.zig', 'build-exe')
    print(subprocess.list2cmdline(args), flush=True)
    subprocess.run(args, cwd=BASE, check=True)
    if ns.step == 'walk':
        subprocess.run([str(BASE / '.zig-cache-loader-sol/weight-walk.exe'), ns.checkpoint], cwd=BASE, check=True)


if __name__ == '__main__':
    main()
