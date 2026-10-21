"""Install repository-local privacy hooks without changing history or replacing unrelated hooks."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import runpy
import shutil
import subprocess
import sys

TOOLS = Path(__file__).resolve().parent
MARKER = '# TensorFold managed privacy hook v1'
PRE_COMMIT = '''#!/bin/sh
# TensorFold managed privacy hook v1
common=$(git rev-parse --path-format=absolute --git-common-dir) || exit 2
python=$(git config --get privacy.python) || exit 2
exec "$python" -I "$common/privacy-tools/privacy_check.py" --staged --identity
'''
PRE_PUSH = '''#!/bin/sh
# TensorFold managed privacy hook v1
private=$(git config --bool --get privacy.privateHistory) || exit 2
case "$private" in
    true)
        printf '%s\\n' 'Push refused: canonical private history must only leave through a verified publication export.' >&2
        exit 1 ;;
    false) exit 0 ;;
    *) exit 2 ;;
esac
'''


def git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args], stderr=subprocess.PIPE)


def configuration(repo, key):
    result = subprocess.run(['git', '-C', str(repo), 'config', '--local', '--get', key], capture_output=True)
    if result.returncode not in (0, 1):
        raise ValueError('Cannot read local Git configuration')
    return result.stdout.decode().strip() if result.returncode == 0 else None


def install(repo, email=None, canonical=False):
    common = Path(git(repo, 'rev-parse', '--path-format=absolute', '--git-common-dir').decode().strip())
    hooks = common / 'hooks'
    configured = subprocess.run(['git', '-C', str(repo), 'config', '--get', 'core.hooksPath'], capture_output=True)
    if configured.returncode == 0:
        raise ValueError('Existing core.hooksPath must be integrated explicitly; nothing was replaced')
    if configured.returncode not in (0, 1):
        raise ValueError('Cannot inspect configured hooks')
    owner = git(repo, 'config', 'user.name').decode().strip()
    policy = json.loads((TOOLS / 'privacy_policy.json').read_text(encoding='utf-8'))
    runpy.run_path(str(TOOLS / 'privacy_check.py'))['validate_policy'](policy)
    identities = {key.casefold(): value for key, value in policy['protected_identities'].items()}
    expected = identities.get(owner.casefold())
    candidate = email or expected or configuration(repo, 'user.email')
    if not candidate or not re.fullmatch(r'[A-Za-z0-9+._\[\]-]+@users\.noreply\.github\.com', candidate):
        raise ValueError('Pass --email with your approved public GitHub no-reply identity')
    if expected and candidate.casefold() != expected.casefold():
        raise ValueError('Requested identity differs from the approved public identity')
    managed = {'pre-commit': PRE_COMMIT, 'pre-push': PRE_PUSH}
    for name in managed:
        path = hooks / name
        if path.is_symlink() or (path.exists() and MARKER not in path.read_text(encoding='utf-8')):
            raise ValueError('Existing hook must be integrated explicitly; nothing was replaced')
    installed = common / 'privacy-tools'
    state_file = installed / 'installation.json'
    if hooks.is_symlink() or installed.is_symlink() or state_file.is_symlink():
        raise ValueError('Managed installation paths must not be symlinks')
    if state_file.exists():
        previous = json.loads(state_file.read_text(encoding='utf-8'))
    else:
        previous = {'original_config': {key: configuration(repo, key) for key in
                    ('user.email', 'privacy.python', 'privacy.privateHistory')}}
    for name in ('privacy_check.py', 'privacy_policy.json'):
        destination = installed / name
        if destination.is_symlink():
            raise ValueError('Managed tool destination is a symlink')
        if destination.exists():
            expected_hash = previous.get('tool_sha256', {}).get(name)
            if hashlib.sha256(destination.read_bytes()).hexdigest() != expected_hash:
                raise ValueError('Unrecognized or modified installed tool; nothing was replaced')
    for name in managed:
        path = hooks / name
        if path.exists():
            if hashlib.sha256(path.read_bytes()).hexdigest() != previous.get('hook_sha256', {}).get(name):
                raise ValueError('Modified managed hook; nothing was replaced')
    hooks.mkdir(exist_ok=True)
    installed.mkdir(exist_ok=True)
    for name in ('privacy_check.py', 'privacy_policy.json'):
        shutil.copyfile(TOOLS / name, installed / name)
    git(repo, 'config', '--local', 'user.email', candidate)
    git(repo, 'config', '--local', 'privacy.python', Path(sys.executable).as_posix())
    if canonical:
        git(repo, 'config', '--local', 'privacy.privateHistory', 'true')
    elif configuration(repo, 'privacy.privateHistory') is None:
        git(repo, 'config', '--local', 'privacy.privateHistory', 'false')
    for name, text in managed.items():
        path = hooks / name
        path.write_bytes(text.encode())
        path.chmod(path.stat().st_mode | 0o111)
    previous.update({
        'version': 1, 'python': sys.executable,
        'private_history': configuration(repo, 'privacy.privateHistory') == 'true',
        'tool_sha256': {name: hashlib.sha256((installed / name).read_bytes()).hexdigest()
                        for name in ('privacy_check.py', 'privacy_policy.json')},
        'hook_sha256': {name: hashlib.sha256((hooks / name).read_bytes()).hexdigest() for name in managed},
    })
    state_file.write_text(json.dumps(previous, indent=2, sort_keys=True), encoding='utf-8')
    print('Installed managed pre-commit and pre-push hooks in the shared Git directory.')
    print('No-reply default set locally; globals, authorship and historical commits were not changed.')
    print('Canonical push guard:', 'enabled' if previous['private_history'] else 'not enabled')
    print('Rerun this installer after reviewing checker/policy updates; unrelated hooks are never overwritten.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path.cwd())
    parser.add_argument('--email')
    parser.add_argument('--canonical', action='store_true', help='refuse every push from this private-history store')
    args = parser.parse_args(argv)
    try:
        install(args.repo, args.email, args.canonical)
        return 0
    except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError):
        print('Privacy hook installation stopped; existing hooks/configuration need review (details redacted).', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
