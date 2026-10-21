"""Check staged or outgoing Git objects without printing sensitive matches."""
import argparse
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import re
import subprocess
import sys

TOOLS = Path(__file__).resolve().parent
HOME = re.compile(r"(?:[A-Za-z]:[\\/]+Users[\\/]+|/(?:Users|home)/)([^\s\"'<>\\/]+)", re.I)
NEUTRAL_USERS = {'developer', 'user', 'username', 'public', 'default', 'runner', 'example'}
IPV4 = re.compile(r'(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])')
PRIVATE_NETS = [ipaddress.ip_network((number, bits)) for number, bits in
                ((0x0A000000, 8), (0xAC100000, 12), (0xC0A80000, 16), (0x64400000, 10))]
TOKENS = {
    'private-key': re.compile(r'-----BEGIN (?:[A-Z]+ )?PRIVATE KEY-----'),
    'github-token': re.compile(r'\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})'),
    'provider-key': re.compile(r'\b(?:sk-(?:proj-|ant-|svcacct-)?[A-Za-z0-9_-]{20,}|hf_[A-Za-z0-9]{20,})'),
    'aws-key': re.compile(r'\bAKIA[0-9A-Z]{16}\b'),
    'slack-token': re.compile(r'\bxox[baprs]-[A-Za-z0-9-]{10,}'),
    'google-key': re.compile(r'\bAIza[0-9A-Za-z_-]{35}\b'),
}
ARCHIVES = {'.zip', '.7z', '.rar', '.tar', '.gz', '.tgz', '.bz2', '.xz', '.zst'}


def git(repo, *args):
    env = dict(os.environ, GIT_NO_REPLACE_OBJECTS='1')
    return subprocess.check_output(['git', '-C', str(repo), *args], env=env, stderr=subprocess.PIPE)


def text_views(data):
    result = [data.decode('utf-8', 'replace')]
    for encoding in ('utf-16-le', 'utf-16-be'):
        for offset in (0, 1):
            result.append(data[offset:].decode(encoding, 'replace'))
    return result


def text_findings(data, literals):
    found = set()
    for text in text_views(data):
        for number, line in enumerate(text.splitlines(), 1):
            if any(match.group(1).casefold() not in NEUTRAL_USERS for match in HOME.finditer(line)):
                found.add(('personal-home', number, hashlib.sha256(line.encode()).hexdigest()))
            for match in IPV4.finditer(line):
                try:
                    address = ipaddress.ip_address(match.group())
                except ValueError:
                    continue
                if any(address in network for network in PRIVATE_NETS):
                    found.add(('private-address', number, hashlib.sha256(line.encode()).hexdigest()))
            for rule, pattern in TOKENS.items():
                if pattern.search(line):
                    found.add((rule, number, hashlib.sha256(line.encode()).hexdigest()))
            if any(value.casefold() in line.casefold() for value in literals):
                found.add(('private-literal', number, hashlib.sha256(line.encode()).hexdigest()))
    return found


def safe_path(path, literals):
    encoded = path.encode('utf-8', 'surrogateescape')
    if text_findings(encoded, literals):
        return '<redacted-path:' + hashlib.sha256(encoded).hexdigest()[:12] + '>'
    return json.dumps(path, ensure_ascii=True)[1:-1]


def private_literals(repo):
    result = subprocess.run(['git', '-C', str(repo), 'config', '--get', 'privacy.privatePolicy'],
                            capture_output=True)
    if result.returncode == 1:
        return []
    if result.returncode:
        raise ValueError('Cannot read private policy configuration')
    path = Path(result.stdout.decode().strip()).expanduser().resolve()
    bare = git(repo, 'rev-parse', '--is-bare-repository').strip() == b'true'
    if not bare:
        root = Path(git(repo, 'rev-parse', '--show-toplevel').decode().strip()).resolve()
        if path.is_relative_to(root):
            raise ValueError('Private policy must be outside the tracked worktree')
    policy = json.loads(path.read_text(encoding='utf-8'))
    values = policy['forbidden_literals']
    if not isinstance(values, list) or any(not isinstance(value, str) or not value for value in values):
        raise ValueError('Invalid private policy')
    expanded = set(values)
    for value in values:
        forward = value.replace(chr(92), '/')
        expanded.update((forward, forward.replace('/', chr(92)), forward.replace('/', chr(92) * 2)))
    return sorted(expanded)


def validate_policy(policy):
    if not isinstance(policy, dict) or policy['version'] != 1:
        raise ValueError('Unsupported policy')
    identities = policy['protected_identities']
    if not isinstance(identities, dict) or not identities:
        raise ValueError('Missing identity policy')
    if any(not isinstance(key, str) or not key or not isinstance(value, str)
           or not re.fullmatch(r'[A-Za-z0-9+._\[\]-]+@users\.noreply\.github\.com', value, re.I)
           for key, value in identities.items()):
        raise ValueError('Invalid identity policy')
    for key in ('blocked_paths', 'blocked_suffixes'):
        if not isinstance(policy[key], list) or not policy[key]:
            raise ValueError('Missing artifact boundary')
        if any(not isinstance(value, str) or not value for value in policy[key]):
            raise ValueError('Invalid artifact boundary')
    if type(policy['max_blob_bytes']) is not int or not 0 < policy['max_blob_bytes'] <= 64 * 1024**2:
        raise ValueError('Invalid object size limit')
    if not isinstance(policy['approved_binaries'], dict) or not isinstance(policy['exceptions'], list):
        raise ValueError('Invalid approval collections')
    for path, approval in policy['approved_binaries'].items():
        if not isinstance(path, str) or not path or not isinstance(approval, dict):
            raise ValueError('Invalid binary approval')
        if not isinstance(approval['sha256'], str) or not re.fullmatch('[0-9a-f]{64}', approval['sha256']):
            raise ValueError('Invalid binary digest')
        if type(approval['size']) is not int or not 0 <= approval['size'] <= 64 * 1024**2:
            raise ValueError('Invalid approved binary size')
    for approval in policy['exceptions']:
        if not isinstance(approval, dict):
            raise ValueError('Invalid fixture exception')
        if any(not isinstance(approval[key], str) or not approval[key]
               for key in ('path', 'rule', 'reason', 'line_sha256')):
            raise ValueError('Exception must have an exact path, rule and reason')
        if approval['rule'] not in TOKENS or not re.fullmatch('[0-9a-f]{64}', approval['line_sha256']):
            raise ValueError('Exception must identify one exact credential-fixture line')


def identity_problem(identity, policy, local=False):
    match = re.fullmatch(r'(.*) <([^<>]+)> -?\d+ [+-]\d{4}', identity)
    if not match:
        return True
    name, email = match.groups()
    identities = {key.casefold(): value for key, value in policy['protected_identities'].items()}
    if name.casefold() in identities:
        return email.casefold() != identities[name.casefold()].casefold()
    return local and re.fullmatch(r'[A-Za-z0-9+._\[\]-]+@users\.noreply\.github\.com', email, re.I) is None


def changed_blobs(repo, revision=None):
    if revision is None:
        rows = git(repo, 'diff', '--cached', '--raw', '-z', '--no-renames', '--no-ext-diff', '--abbrev=40')
    else:
        rows = git(repo, 'diff-tree', '--root', '-m', '-r', '--raw', '-z', '--no-renames',
                   '--no-ext-diff', '--no-commit-id', '--abbrev=40', revision)
    entries = rows.split(b'\0')
    found = set()
    for index in range(0, len(entries)-1, 2):
        fields = entries[index].split()
        if len(fields) != 5 or not fields[0].startswith(b':'):
            raise ValueError('Unexpected Git diff format')
        mode, oid = fields[1].decode(), fields[3].decode()
        if mode != '000000':
            found.add((entries[index+1].decode('utf-8', 'surrogateescape'), mode, oid))
    return sorted(found)


def scan_blob(repo, entry, policy, literals, cache):
    path, mode, oid = entry
    display = safe_path(path, literals)
    errors = []
    normalized = path.replace(chr(92), '/').casefold()
    private_directory = any(part in {'.artifacts', '.private-evidence'} for part in normalized.split('/'))
    blocked = [value.replace(chr(92), '/').casefold() for value in policy['blocked_paths']]
    if private_directory or any(normalized.startswith(value) if value.endswith('/') else normalized == value
                                for value in blocked):
        return [f'{display}: private-artifact-path']
    if Path(normalized).suffix.casefold() in policy['blocked_suffixes']:
        return [f'{display}: private-debug-artifact']
    if mode not in {'100644', '100755', '120000'}:
        return [f'{display}: unsupported Git object mode']
    if text_findings(path.encode('utf-8', 'surrogateescape'), literals):
        errors.append(f'{display}: sensitive-path-name')
    approval = policy['approved_binaries'].get(path)
    size = int(git(repo, 'cat-file', '-s', oid))
    if size > policy['max_blob_bytes'] and not (approval and size == approval['size']):
        return errors + [f'{display}: oversized-object; needs a reviewed content-hash approval']
    if oid not in cache:
        cache[oid] = git(repo, 'cat-file', 'blob', oid)
    data = cache[oid]
    digest = hashlib.sha256(data).hexdigest()
    approved = bool(approval and approval['size'] == len(data) and approval['sha256'] == digest)
    if size > policy['max_blob_bytes'] and not approved:
        errors.append(f'{display}: oversized-object; content-hash approval does not match')
    archive = Path(path).suffix.casefold() in ARCHIVES or data.startswith(
        (b'PK\x03\x04', b'7z\xbc\xaf\x27\x1c', b'\x1f\x8b'))
    try:
        data.decode('utf-8-sig')
        binary = b'\0' in data
    except UnicodeDecodeError:
        binary = True
    if data.startswith((b'\xff\xfe', b'\xfe\xff')):
        try:
            data.decode('utf-16')
            binary = False
        except UnicodeDecodeError:
            pass
    if archive and not approved:
        errors.append(f'{display}: archive-not-source; needs a reviewed content-hash approval')
    elif binary and not approved:
        errors.append(f'{display}: unexpected-binary-or-archive; needs a reviewed content-hash approval')
    allowed = {(value['path'], value['rule'], value['line_sha256']) for value in policy['exceptions']}
    for rule, number, fingerprint in sorted(text_findings(data, literals)):
        if (path, rule, fingerprint) not in allowed:
            errors.append(f'{display}:{number}: {rule} [match redacted]')
    return errors


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path.cwd())
    parser.add_argument('--policy', type=Path, default=TOOLS / 'privacy_policy.json')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--staged', action='store_true')
    modes.add_argument('--range', dest='revision_range', metavar='BASE..HEAD')
    parser.add_argument('--identity', action='store_true', help='also check the identity of the pending local commit')
    args = parser.parse_args(argv)
    try:
        policy = json.loads(args.policy.read_text(encoding='utf-8'))
        validate_policy(policy)
        literals = private_literals(args.repo)
        errors, cache, seen = [], {}, set()
        commits = []
        if args.revision_range:
            if args.identity or args.revision_range.count('..') != 1 or '...' in args.revision_range:
                raise ValueError('Use a two-dot commit range without --identity')
            base, tip = args.revision_range.split('..')
            if not base or not tip or base.startswith('-') or tip.startswith('-'):
                raise ValueError('Both commit range endpoints are required')
            base = git(args.repo, 'rev-parse', '--verify', base + '^{commit}').decode().strip()
            tip = git(args.repo, 'rev-parse', '--verify', tip + '^{commit}').decode().strip()
            commits = git(args.repo, 'rev-list', '--reverse', base + '..' + tip).decode().splitlines()
            revisions = commits
        else:
            revisions = [None]
        if args.identity:
            for variable in ('GIT_AUTHOR_IDENT', 'GIT_COMMITTER_IDENT'):
                identity = git(args.repo, 'var', variable).decode().strip()
                if (identity_problem(identity, policy, local=variable == 'GIT_COMMITTER_IDENT')
                        or text_findings(identity.encode(), literals)):
                    errors.append('pending commit: non-public identity [identity redacted]')
        for revision in revisions:
            if revision is not None:
                raw = git(args.repo, 'cat-file', 'commit', revision)
                headers, message = raw.split(b'\n\n', 1)
                for line in headers.decode('utf-8', 'replace').splitlines():
                    if line.startswith(('author ', 'committer ')):
                        if identity_problem(line.split(' ', 1)[1], policy) or text_findings(line.encode(), literals):
                            errors.append(revision[:12] + ': non-public protected identity [identity redacted]')
                if text_findings(message, literals):
                    errors.append(revision[:12] + ': sensitive commit message [match redacted]')
            for entry in changed_blobs(args.repo, revision):
                if entry not in seen:
                    seen.add(entry)
                    errors.extend(scan_blob(args.repo, entry, policy, literals, cache))
        for error in sorted(set(errors)):
            print(error)
        print(f'privacy: {len(commits)} committed changes, {len(seen)} blobs; {len(set(errors))} problems')
        return 1 if errors else 0
    except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError):
        print('privacy: cannot complete the check; inspect Git state and policy configuration (details redacted)', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
