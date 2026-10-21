"""Exercise installation and actual Git hook behavior in disposable repositories."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

TOOLS = Path(__file__).resolve().parent
INSTALL = TOOLS / 'install_privacy_hooks.py'
PUBLIC_EMAIL = '436016+goulderb@users.noreply.github.com'


class PrivacyHooks(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.repo = self.root / 'repo'
        self.repo.mkdir()
        self.env = {key: value for key, value in os.environ.items() if not key.startswith('GIT_')}
        self.env.update(GIT_CONFIG_NOSYSTEM='1', GIT_CONFIG_GLOBAL=os.devnull, GIT_TERMINAL_PROMPT='0')
        self.git('init', '-b', 'main')
        self.git('config', 'user.name', 'Brett Goulder')
        self.git('config', 'user.email', PUBLIC_EMAIL)
        (self.repo / 'safe.txt').write_text('safe\n')
        self.git('add', 'safe.txt')
        self.git('commit', '-m', 'Initial safe snapshot')

    def git(self, *args, repo=None, check=True):
        return subprocess.run(['git', '-C', str(repo or self.repo), *args], env=self.env,
                              capture_output=True, check=check)

    def install(self, *args, check=True):
        return subprocess.run([sys.executable, '-I', str(INSTALL), '--repo', str(self.repo), *args],
                              env=self.env, capture_output=True, check=check)

    def test_safe_commit_succeeds_and_personal_home_commit_is_rejected(self):
        self.install('--canonical')
        (self.repo / 'safe.txt').write_text('public change\n')
        self.git('add', 'safe.txt')
        self.git('commit', '-m', 'Safe change')
        before = self.git('rev-parse', 'HEAD').stdout
        forbidden = 'C:' + '/Users/' + 'private-person'
        (self.repo / 'unsafe.txt').write_text(forbidden + '/compiler\n')
        self.git('add', 'unsafe.txt')
        result = self.git('commit', '-m', 'Must not be committed', check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(b'personal-home', result.stdout + result.stderr)
        self.assertNotIn(forbidden.encode(), result.stdout + result.stderr)
        self.assertEqual(self.git('rev-parse', 'HEAD').stdout, before)

    def test_empty_commit_checks_actual_author_override_and_preserves_donor_authorship(self):
        self.install('--canonical')
        before = self.git('rev-parse', 'HEAD').stdout
        private_email = 'owner@' + 'example.net'
        result = self.git('commit', '--allow-empty', '--author', f'Brett Goulder <{private_email}>',
                          '-m', 'Must reject this author override', check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.git('rev-parse', 'HEAD').stdout, before)
        self.assertNotIn(private_email.encode(), result.stdout + result.stderr)
        self.git('commit', '--allow-empty', '--author', 'Dana Donor <dana@donor.example>',
                 '-m', 'Preserve the original donor author')
        self.assertEqual(self.git('show', '-s', '--format=%an <%ae>').stdout.strip(),
                         b'Dana Donor <dana@donor.example>')

    def test_shared_hook_protects_an_older_linked_worktree_without_checker_sources(self):
        linked = self.root / 'older-worktree'
        self.git('worktree', 'add', '-b', 'older', str(linked))
        self.install('--canonical')
        self.assertFalse((linked / 'tools/privacy_check.py').exists())
        (linked / 'unsafe.txt').write_text('C:' + '/Users/' + 'private-person' + '/compiler')
        self.git('add', 'unsafe.txt', repo=linked)
        result = self.git('commit', '-m', 'Must be blocked in every worktree', repo=linked, check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(b'personal-home', result.stdout + result.stderr)

    def test_canonical_push_is_rejected_before_updating_even_a_local_remote(self):
        self.install('--canonical')
        remote = self.root / 'remote.git'
        subprocess.run(['git', 'init', '--bare', str(remote)], env=self.env, capture_output=True, check=True)
        result = self.git('push', str(remote), 'HEAD:refs/heads/main', check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(b'verified publication export', result.stderr)
        refs = self.git('for-each-ref', '--format=%(refname)', repo=remote).stdout
        self.assertEqual(refs.strip(), b'')

    def test_push_fails_closed_when_private_history_setting_is_invalid_or_missing(self):
        self.install('--canonical')
        remote = self.root / 'remote.git'
        subprocess.run(['git', 'init', '--bare', str(remote)], env=self.env, capture_output=True, check=True)
        for setting in ('invalid', None):
            with self.subTest(setting=setting):
                if setting is None:
                    self.git('config', '--unset', 'privacy.privateHistory')
                else:
                    self.git('config', 'privacy.privateHistory', setting)
                result = self.git('push', str(remote), 'HEAD:refs/heads/main', check=False)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(self.git('for-each-ref', '--format=%(refname)', repo=remote).stdout.strip(), b'')

    def test_public_history_install_explicitly_allows_pushes(self):
        self.install()
        remote = self.root / 'remote.git'
        subprocess.run(['git', 'init', '--bare', str(remote)], env=self.env, capture_output=True, check=True)
        self.git('push', str(remote), 'HEAD:refs/heads/main')
        self.assertEqual(self.git('rev-parse', 'refs/heads/main', repo=remote).stdout,
                         self.git('rev-parse', 'HEAD').stdout)

    def test_existing_unrelated_hook_and_configuration_are_not_overwritten(self):
        hook = self.repo / '.git/hooks/pre-commit'
        hook.write_bytes(b'#!/bin/sh\nexit 0\n')
        original = (self.repo / '.git/config').read_bytes()
        result = self.install('--canonical', check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(hook.read_bytes(), b'#!/bin/sh\nexit 0\n')
        self.assertEqual((self.repo / '.git/config').read_bytes(), original)
        self.assertFalse((self.repo / '.git/privacy-tools').exists())

    def test_existing_hook_path_is_not_bypassed(self):
        for value in ('.custom-hooks', ''):
            with self.subTest(value=value):
                self.git('config', 'core.hooksPath', value)
                original = (self.repo / '.git/config').read_bytes()
                self.assertNotEqual(self.install(check=False).returncode, 0)
                self.assertEqual((self.repo / '.git/config').read_bytes(), original)

    def test_reinstallation_is_safe_and_preserves_the_initial_config_backup(self):
        self.git('config', 'user.email', 'developer@example.net')
        self.install('--canonical')
        state = self.repo / '.git/privacy-tools/installation.json'
        first = json.loads(state.read_text())
        self.install('--canonical')
        second = json.loads(state.read_text())
        self.assertEqual(first['original_config'], second['original_config'])
        self.assertEqual(self.git('config', '--local', 'user.email').stdout.strip(), PUBLIC_EMAIL.encode())
        self.assertEqual(self.git('config', '--local', 'privacy.privateHistory').stdout.strip(), b'true')

    def test_modified_managed_tool_is_not_silently_replaced(self):
        self.install('--canonical')
        installed = self.repo / '.git/privacy-tools/privacy_check.py'
        installed.write_bytes(installed.read_bytes() + b'\n# User modification\n')
        original = installed.read_bytes()
        self.assertNotEqual(self.install('--canonical', check=False).returncode, 0)
        self.assertEqual(installed.read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
