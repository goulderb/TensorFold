"""End-to-end tests for tools/privacy_check.py: real CLI subprocesses against temporary Git repositories."""
# Forbidden strings are built from fragments at runtime so this source never contains a complete example.
import gzip
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
CHECKER = TOOLS / 'privacy_check.py'
DEFAULT_POLICY = TOOLS / 'privacy_policy.json'
PYTHON = sys.executable
OWNER = 'Brett Goulder'
OWNER_NOREPLY = '436016+goulderb@users.noreply.github.com'
FIXTURE_NAME = 'Fix Turin'
FIXTURE_EMAIL = 'fixturin@users.noreply.github.com'
DONOR = ('Dana Donor', 'dana@donor.example')
WRONG_OWNER = (OWNER, 'owner@' + 'example.com')
CLEAN = 'plain source line\nanother plain line\n'


def token_ghp():
    return 'ghp_' + 'a1B2c3D4e5' * 4


def token_sk():
    return 'sk-' + 'z9Q' * 10


def token_aws():
    return 'AKIA' + 'Y3Z9' * 4


def token_slack():
    return 'xoxb-' + 'q7Zr' * 4


def token_google():
    return 'AIza' + 'g9Q2' * 8 + 'g9Q'


def key_header():
    return '-----BEGIN ' + 'PRIVATE KEY-----'


def home_windows(user='jdoe'):
    return 'C:' + chr(92) + 'Users' + chr(92) + user


def home_mac(user='jdoe'):
    return '/Users/' + user


def home_linux(user='jdoe'):
    return '/home/' + user


def private_ip():
    return '10.' + '77.31.9'


def cgnat_ip():
    return '100.' + '64.0.7'


def private_ip_172():
    return '172.' + '16.31.9'


def private_ip_192():
    return '192.' + '168.1.50'


def line_hash(text):
    return hashlib.sha256(text.encode()).hexdigest()


def child_env(extra=None):
    env = {key: value for key, value in os.environ.items() if not key.startswith('GIT_')}
    env.update(GIT_NO_REPLACE_OBJECTS='1', GIT_CONFIG_NOSYSTEM='1', GIT_CONFIG_GLOBAL=os.devnull)
    if extra:
        env.update(extra)
    return env


def git(repo, *args, env=None, input=None):
    command = ['git']
    if repo is not None:
        command += ['-C', str(repo)]
    return subprocess.check_output(command + list(args), env=child_env(env), input=input, stderr=subprocess.PIPE)


def run_cli(repo, *args, env=None):
    return subprocess.run([PYTHON, '-I', str(CHECKER), '--repo', str(repo), *args],
                          capture_output=True, text=True, env=child_env(env))


class PrivacyCheckTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.repo = self.root / 'repo'
        git(None, 'init', '-q', '-b', 'main', str(self.repo))
        git(self.repo, 'config', 'user.name', FIXTURE_NAME)
        git(self.repo, 'config', 'user.email', FIXTURE_EMAIL)

    def stage(self, path, data, add=True):
        target = self.repo / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data if isinstance(data, bytes) else data.encode('utf-8'))
        if add:
            git(self.repo, 'add', '-A', '--', path)

    def remove(self, path):
        git(self.repo, 'rm', '-q', '--', path)

    def unstage(self, path):
        git(self.repo, 'rm', '--cached', '-q', '--', path)

    def commit(self, message='change', author=None, committer=None, allow_empty=False):
        env = {}
        if author:
            env['GIT_AUTHOR_NAME'], env['GIT_AUTHOR_EMAIL'] = author
        if committer:
            env['GIT_COMMITTER_NAME'], env['GIT_COMMITTER_EMAIL'] = committer
        args = ['commit', '-q', '-m', message] + (['--allow-empty'] if allow_empty else [])
        git(self.repo, *args, env=env or None)

    def stage_special(self, path, mode, data):
        sha = git(self.repo, 'hash-object', '-w', '--stdin', input=data).decode().strip()
        git(self.repo, 'update-index', '--add', '--cacheinfo', '%s,%s,%s' % (mode, sha, path))

    def policy(self, **overrides):
        policy = {'version': 1, 'protected_identities': {OWNER: OWNER_NOREPLY}, 'blocked_paths': ['private/'],
                  'blocked_suffixes': ['.pdb'], 'max_blob_bytes': 10485760, 'approved_binaries': {},
                  'exceptions': []}
        policy.update(overrides)
        path = self.root / ('policy%d.json' % len(list(self.root.glob('policy*.json'))))
        path.write_text(json.dumps(policy), encoding='utf-8')
        return path

    def private_policy(self, literals):
        path = self.root / 'private_literals.json'
        path.write_text(json.dumps({'forbidden_literals': literals}), encoding='utf-8')
        git(self.repo, 'config', 'privacy.privatePolicy', str(path))

    def approval_for(self, data):
        return {'sha256': hashlib.sha256(data).hexdigest(), 'size': len(data)}

    def check(self, *args, env=None):
        return run_cli(self.repo, *args, env=env)

    def assertClean(self, proc):
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)

    def assertFlagged(self, proc, count, *rules):
        self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
        for rule in rules:
            self.assertIn(rule, proc.stdout)
        self.assertRegex(proc.stdout.strip().splitlines()[-1],
                         r'privacy: \d+ committed changes, \d+ blobs; %d problems' % count)

    def assertClosed(self, proc):
        self.assertEqual(proc.returncode, 2, proc.stdout + proc.stderr)
        self.assertIn('cannot complete the check', proc.stderr)
        self.assertEqual(proc.stdout, '')

    # Staged scans read the index, never the worktree.

    def test_staged_clean_worktree_dirty_passes(self):
        self.stage('src/app.py', CLEAN)
        self.stage('src/app.py', 'token ' + token_ghp() + '\n', add=False)
        self.assertClean(self.check('--staged'))

    def test_staged_secret_worktree_clean_flagged(self):
        self.stage('notes.txt', 'key ' + token_ghp() + '\n')
        self.stage('notes.txt', CLEAN, add=False)
        self.assertEqual((self.repo / 'notes.txt').read_text(), CLEAN)
        self.assertFlagged(self.check('--staged'), 1, 'github-token')

    def test_default_mode_matches_staged_flag(self):
        self.stage('notes.txt', 'key ' + token_ghp() + '\n')
        default = self.check()
        forced = self.check('--staged')
        self.assertEqual(default.returncode, forced.returncode)
        self.assertIn('github-token', default.stdout)
        self.assertIn('github-token', forced.stdout)

    def test_untracked_secret_not_scanned(self):
        self.stage('notes.txt', 'key ' + token_ghp() + '\n', add=False)
        self.assertClean(self.check('--staged'))

    # Pending identity (--identity) reads the real Git identity configuration.

    def test_identity_owner_wrong_email_flagged(self):
        git(self.repo, 'config', 'user.name', OWNER)
        git(self.repo, 'config', 'user.email', 'owner@' + 'example.com')
        self.commit('set identity', allow_empty=True)
        proc = self.check('--identity')
        self.assertFlagged(proc, 1, 'pending commit: non-public identity [identity redacted]')
        self.assertNotIn(OWNER, proc.stdout)

    def test_identity_owner_noreply_passes(self):
        git(self.repo, 'config', 'user.name', OWNER)
        git(self.repo, 'config', 'user.email', OWNER_NOREPLY)
        self.commit('set identity', allow_empty=True)
        self.assertClean(self.check('--identity'))

    def test_identity_donor_author_preserved(self):
        self.commit('donor work', author=DONOR, allow_empty=True)
        env = {'GIT_AUTHOR_NAME': DONOR[0], 'GIT_AUTHOR_EMAIL': DONOR[1]}
        self.assertClean(self.check('--identity', env=env))

    def test_identity_local_committer_needs_noreply(self):
        self.commit('local work', allow_empty=True)
        env = {'GIT_COMMITTER_NAME': FIXTURE_NAME, 'GIT_COMMITTER_EMAIL': 'fix@' + 'example.com'}
        self.assertFlagged(self.check('--identity', env=env), 1,
                           'pending commit: non-public identity [identity redacted]')

    def test_identity_with_range_fails_closed(self):
        self.commit('init', allow_empty=True)
        self.assertClosed(self.check('--identity', '--range', 'HEAD~1..HEAD'))

    # Two-dot ranges scan every intermediate commit, identities and messages included.

    def test_range_scans_intermediate_commits(self):
        self.commit('init', allow_empty=True)
        base = git(self.repo, 'rev-parse', 'HEAD').decode().strip()
        self.stage('leak.txt', 'key ' + token_ghp() + '\n')
        self.commit('add leak')
        self.remove('leak.txt')
        self.commit('drop leak')
        proc = self.check('--range', base + '..HEAD')
        self.assertFlagged(proc, 1, 'github-token')
        self.assertIn('2 committed changes', proc.stdout)

    def test_range_flags_owner_identity_in_history(self):
        self.commit('init', allow_empty=True)
        base = git(self.repo, 'rev-parse', 'HEAD').decode().strip()
        self.stage('data.txt', CLEAN)
        self.commit('owner commit', author=WRONG_OWNER)
        self.remove('data.txt')
        self.commit('drop data')
        proc = self.check('--range', base + '..HEAD')
        self.assertFlagged(proc, 1, 'non-public protected identity [identity redacted]')

    def test_range_flags_private_literal_identity(self):
        self.private_policy(['jdoe@' + 'corp.example'])
        self.commit('init', allow_empty=True)
        base = git(self.repo, 'rev-parse', 'HEAD').decode().strip()
        self.stage('src/app.py', CLEAN)
        self.commit('hidden owner', author=('Anon', 'jdoe@' + 'corp.example'))
        proc = self.check('--range', base + '..HEAD')
        self.assertFlagged(proc, 1, 'non-public protected identity [identity redacted]')
        self.assertNotIn('jdoe@' + 'corp.example', proc.stdout)

    def test_range_flags_sensitive_commit_message(self):
        self.commit('init', allow_empty=True)
        base = git(self.repo, 'rev-parse', 'HEAD').decode().strip()
        self.commit('update ' + token_ghp(), allow_empty=True)
        proc = self.check('--range', base + '..HEAD')
        self.assertFlagged(proc, 1, 'sensitive commit message [match redacted]')
        self.assertNotIn(token_ghp(), proc.stdout)

    def test_range_clean_donor_history_passes(self):
        self.commit('init', allow_empty=True)
        self.stage('src/app.py', CLEAN)
        self.commit('donor commit', author=DONOR)
        base = git(self.repo, 'rev-parse', 'HEAD').decode().strip()
        self.assertClean(self.check('--range', base + '..HEAD'))

    def test_merge_only_range_checks_every_parent_even_when_endpoint_trees_match(self):
        self.commit('root', allow_empty=True)
        root = git(self.repo, 'rev-parse', 'HEAD').decode().strip()
        git(self.repo, 'checkout', '-q', '-b', 'left')
        self.stage('leak.txt', token_ghp() + '\n')
        self.commit('left')
        left = git(self.repo, 'rev-parse', 'HEAD').decode().strip()
        git(self.repo, 'checkout', '-q', '-b', 'right', root)
        self.stage('safe.txt', CLEAN)
        self.commit('right')
        right = git(self.repo, 'rev-parse', 'HEAD').decode().strip()
        git(self.repo, 'checkout', '-q', 'left')
        git(self.repo, 'merge', '--no-ff', 'right', '-m', 'Baseline merge')
        base = git(self.repo, 'rev-parse', 'HEAD').decode().strip()
        tree = git(self.repo, 'rev-parse', 'HEAD^{tree}').decode().strip()
        tip = git(self.repo, 'commit-tree', tree, '-p', left, '-p', right, '-m', 'Candidate merge').decode().strip()
        self.assertEqual(git(self.repo, 'diff', '--name-only', base, tip), b'')
        proc = self.check('--range', base + '..' + tip)
        self.assertFlagged(proc, 1, 'github-token')
        self.assertIn('1 committed changes', proc.stdout)

    def test_range_invalid_input_fails_closed(self):
        self.commit('init', allow_empty=True)
        self.assertClosed(self.check('--range', '...'))
        self.assertClosed(self.check('--range', 'HEAD..'))
        self.assertClosed(self.check('--range', 'nosuchrevision..HEAD'))

    # Generic content rules: home paths, private addresses, credential shapes.

    def test_personal_home_paths_flagged(self):
        content = '\n'.join([home_windows(), home_mac(), home_linux()]) + '\n'
        self.stage('notes.txt', content)
        self.assertFlagged(self.check('--staged'), 3, 'personal-home')
        self.assertNotIn('jdoe', self.check('--staged').stdout)

    def test_neutral_user_home_passes(self):
        self.stage('notes.txt', 'C:' + chr(92) + 'Users' + chr(92) + 'runner\n/home/developer/bin\n')
        self.assertClean(self.check('--staged'))

    def test_private_addresses_flagged(self):
        content = '\n'.join([private_ip(), private_ip_172(), private_ip_192(), cgnat_ip()]) + '\n'
        self.stage('notes.txt', content)
        self.assertFlagged(self.check('--staged'), 4, 'private-address')

    def test_public_and_loopback_addresses_pass(self):
        self.stage('notes.txt', '127.0.0.1\n192.0.2.1\n198.51.100.7\n203.0.113.9\n8.8.8.8\n')
        self.assertClean(self.check('--staged'))

    def test_credential_rules_flagged(self):
        content = '\n'.join([token_ghp(), token_sk(), token_aws(), token_slack(), token_google(), key_header()]) + '\n'
        self.stage('notes.txt', content)
        self.assertFlagged(self.check('--staged'), 6, 'github-token', 'provider-key', 'aws-key',
                           'slack-token', 'google-key', 'private-key')

    def test_utf16_token_detected(self):
        self.stage('notes.txt', ('start\n' + token_ghp() + '\nend').encode('utf-16'))
        self.assertFlagged(self.check('--staged'), 1, 'github-token')

    def test_nul_padded_token_detected(self):
        self.stage('notes.txt', b'clean line\n\x00\n' + token_ghp().encode())
        self.assertFlagged(self.check('--staged'), 2, 'github-token', 'unexpected-binary-or-archive')

    def test_output_never_contains_match_values(self):
        self.stage('notes.txt', '\n'.join([token_ghp(), home_windows(), private_ip(), key_header()]) + '\n')
        proc = self.check('--staged')
        self.assertEqual(proc.returncode, 1)
        for secret in [token_ghp(), home_windows(), private_ip(), key_header()]:
            self.assertNotIn(secret, proc.stdout)
            self.assertNotIn(secret, proc.stderr)

    # Private artifact boundaries, debug artifacts, and unusual index entries.

    def test_blocked_paths_and_receipts_flagged(self):
        self.stage('docs/plans/qwen35-ornith/evidence/notes.txt', CLEAN)
        self.stage('zig/src/families/glm/fixtures/weight-loader-verification.json', '{}\n')
        self.stage('.private-evidence/deep/y.txt', CLEAN)
        self.assertFlagged(self.check('--staged'), 3, 'private-artifact-path')

    def test_case_insensitive_artifact_dirs_blocked(self):
        self.stage('src/.Artifacts/x.txt', CLEAN)
        self.stage('docs/.PRIVATE-EVIDENCE/y.txt', CLEAN)
        self.assertFlagged(self.check('--staged'), 2, 'private-artifact-path')

    def test_blocked_suffixes_flagged(self):
        self.stage('build/trace.pdb', b'x')
        self.stage('prof.nsys-rep', b'y')
        self.assertFlagged(self.check('--staged'), 2, 'private-debug-artifact')

    def test_gitlink_mode_rejected(self):
        self.stage_special('submodule', '160000', b'1' * 40)
        self.assertFlagged(self.check('--staged'), 1, 'unsupported Git object mode')

    def test_symlink_content_scanned(self):
        self.stage_special('lnk-secret', '120000', token_ghp().encode())
        self.stage_special('lnk-ok', '120000', b'../assets/headers.json')
        self.assertFlagged(self.check('--staged'), 1, 'github-token')

    def test_sensitive_path_name_redacted(self):
        self.stage('docs/home/' + 'jdoe' + '/notes.md', CLEAN)
        proc = self.check('--staged')
        self.assertFlagged(proc, 1, 'sensitive-path-name', '<redacted-path:')
        self.assertNotIn('jdoe', proc.stdout)

    # Archives are rejected unless exactly approved; private-path bans always win.

    def test_unapproved_archive_flagged(self):
        self.stage('capture.gz', gzip.compress(CLEAN.encode()))
        self.stage('bundle.bin', b'PK\x03\x04' + b'x' * 8)
        self.assertFlagged(self.check('--staged'), 2, 'archive-not-source')

    def test_approved_archive_passes(self):
        data = gzip.compress(b'header table bytes')
        policy = self.policy(approved_binaries={'assets/headers.json.gz': self.approval_for(data)})
        self.stage('assets/headers.json.gz', data)
        self.assertClean(self.check('--policy', str(policy)))

    def test_approved_archive_wrong_hash_flagged(self):
        data = gzip.compress(b'header table bytes')
        approval = self.approval_for(data)
        approval['sha256'] = 'f' * 64
        policy = self.policy(approved_binaries={'assets/headers.json.gz': approval})
        self.stage('assets/headers.json.gz', data)
        self.assertFlagged(self.check('--policy', str(policy)), 1, 'archive-not-source')

    def test_approved_archive_at_private_path_still_blocked(self):
        data = gzip.compress(b'header table bytes')
        policy = self.policy(approved_binaries={'private/capture.gz': self.approval_for(data)})
        self.stage('private/capture.gz', data)
        self.assertFlagged(self.check('--policy', str(policy)), 1, 'private-artifact-path')

    # Binary approvals are exact-content, and approved contents are still scanned.

    def test_approved_binary_passes_and_contents_scanned(self):
        clean_binary = b'\x00\x01\x02runtime\x00payload'
        policy = self.policy(approved_binaries={'assets/kernel.bin': self.approval_for(clean_binary)})
        self.stage('assets/kernel.bin', clean_binary)
        self.assertClean(self.check('--policy', str(policy)))
        self.unstage('assets/kernel.bin')
        secret_binary = b'\x00' + token_ghp().encode()
        policy = self.policy(approved_binaries={'assets/payload.bin': self.approval_for(secret_binary)})
        self.stage('assets/payload.bin', secret_binary)
        self.assertFlagged(self.check('--policy', str(policy)), 1, 'github-token')

    def test_approved_binary_utf16_literal_detected(self):
        literal = 'jdo' + chr(233) + '@' + 'corp.example'
        self.private_policy([literal])
        data = b'\x01\xd8\x00' + literal.encode('utf-16-le')
        policy = self.policy(approved_binaries={'assets/payload.bin': self.approval_for(data)})
        self.stage('assets/payload.bin', data)
        proc = self.check('--policy', str(policy))
        self.assertFlagged(proc, 1, 'private-literal')
        self.assertNotIn(literal, proc.stdout)

    def test_binary_wrong_hash_flagged(self):
        data = b'\x00\x01\x02runtime'
        approval = self.approval_for(data)
        approval['sha256'] = 'e' * 64
        policy = self.policy(approved_binaries={'assets/kernel.bin': approval})
        self.stage('assets/kernel.bin', data)
        self.assertFlagged(self.check('--policy', str(policy)), 1, 'unexpected-binary-or-archive')

    def test_oversized_object_handling(self):
        big = b'a' * 100
        policy = self.policy(max_blob_bytes=64)
        self.stage('big.txt', big)
        self.assertFlagged(self.check('--policy', str(policy)), 1, 'oversized-object')
        policy = self.policy(max_blob_bytes=64, approved_binaries={'big.txt': self.approval_for(big)})
        self.assertClean(self.check('--policy', str(policy)))

    def test_oversized_wrong_hash_approval_rejected(self):
        big = b'a' * 100
        approval = self.approval_for(big)
        approval['sha256'] = 'd' * 64
        policy = self.policy(max_blob_bytes=64, approved_binaries={'big.txt': approval})
        self.stage('big.txt', big)
        self.assertFlagged(self.check('--policy', str(policy)), 1, 'oversized-object')

    # Renames and suffix changes do not bypass content and artifact rules.

    def test_renamed_secret_still_flagged(self):
        self.stage('secret.txt', 'key ' + token_ghp() + '\n')
        self.commit('add secret')
        git(self.repo, 'mv', 'secret.txt', 'renamed.txt')
        self.assertFlagged(self.check('--staged'), 1, 'github-token')

    def test_renamed_artifact_still_flagged(self):
        self.stage('artifact.pdb', b'\x00binary')
        self.commit('add artifact')
        git(self.repo, 'mv', 'artifact.pdb', 'artifact.txt')
        proc = self.check('--staged')
        self.assertFlagged(proc, 1, 'unexpected-binary-or-archive')
        self.assertNotIn('private-debug-artifact', proc.stdout)

    # Exceptions allow one exact path, rule and line fingerprint, nothing wider.

    def test_exception_allows_exact_line(self):
        policy = self.policy(exceptions=[{'path': 'notes.txt', 'rule': 'github-token',
                                          'line_sha256': line_hash(token_ghp()), 'reason': 'reviewed fixture'}])
        self.stage('notes.txt', token_ghp() + '\n')
        self.assertClean(self.check('--policy', str(policy)))

    def test_exception_scoped_to_path_rule_and_hash(self):
        exception = {'path': 'notes.txt', 'rule': 'github-token',
                     'line_sha256': line_hash(token_ghp()), 'reason': 'reviewed fixture'}
        policy = self.policy(exceptions=[exception])
        self.stage('other.txt', token_ghp() + '\n')
        self.assertFlagged(self.check('--policy', str(policy)), 1, 'github-token')
        self.unstage('other.txt')
        self.stage('notes.txt', token_ghp() + 'x\n')
        self.assertFlagged(self.check('--policy', str(policy)), 1, 'github-token')
        policy = self.policy(exceptions=[dict(exception, rule='private-key')])
        self.stage('notes.txt', token_ghp() + '\n')
        self.assertFlagged(self.check('--policy', str(policy)), 1, 'github-token')

    # External privacy.privatePolicy literals, including separator and encoding variants.

    def test_private_literal_detected(self):
        literal = 'jdoe@' + 'corp.example'
        self.private_policy([literal])
        self.stage('notes.txt', 'contact ' + literal + '\n')
        proc = self.check('--staged')
        self.assertFlagged(proc, 1, 'private-literal')
        self.assertNotIn(literal, proc.stdout)

    def test_slash_literal_expands_to_backslashes(self):
        self.private_policy(['team/secrets'])
        self.stage('notes.txt', 'see team' + chr(92) + 'secrets and team' + chr(92) * 2 + 'secrets\n')
        self.assertFlagged(self.check('--staged'), 1, 'private-literal')

    def test_backslash_literal_expands_to_doubled(self):
        self.private_policy(['team' + chr(92) + 'secrets'])
        self.stage('notes.txt', 'see team' + chr(92) * 2 + 'secrets\n')
        self.assertFlagged(self.check('--staged'), 1, 'private-literal')

    def test_utf16_nonascii_literal_detected(self):
        literal = 'jdo' + chr(233) + '@' + 'corp.example'
        self.private_policy([literal])
        self.stage('notes.txt', ('mail ' + literal + '\n').encode('utf-16'))
        proc = self.check('--staged')
        self.assertFlagged(proc, 1, 'private-literal')
        self.assertNotIn(literal, proc.stdout)

    def test_private_policy_inside_worktree_fails_closed(self):
        inside = self.repo / 'leaked_policy.json'
        inside.write_text(json.dumps({'forbidden_literals': ['x' + 'y']}), encoding='utf-8')
        git(self.repo, 'config', 'privacy.privatePolicy', str(inside))
        self.assertClosed(self.check('--staged'))

    # Fail closed on unusable policy, non-repos, and keep the tool-adjacent default.

    def test_invalid_policy_inputs_fail_closed(self):
        broken = self.root / 'broken.json'
        broken.write_text('{not json', encoding='utf-8')
        self.assertClosed(self.check('--policy', str(broken)))
        self.assertClosed(self.check('--policy', str(self.root / 'missing.json')))
        incomplete = self.root / 'incomplete.json'
        incomplete.write_text(json.dumps({'version': 1}), encoding='utf-8')
        self.assertClosed(self.check('--policy', str(incomplete)))

    def test_not_a_git_repo_fails_closed(self):
        self.assertClosed(run_cli(self.root, '--staged'))

    def test_default_policy_is_tool_adjacent(self):
        self.assertTrue(DEFAULT_POLICY.exists())
        self.stage('src/app.py', CLEAN)
        self.assertClean(self.check())


if __name__ == '__main__':
    unittest.main()
