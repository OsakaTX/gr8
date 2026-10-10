#!/usr/bin/env python3
"""docker/mounts.py: which host paths docker/gr8 mounts, from the launcher's option parsing.
    python3 -m unittest discover -s docker"""
import contextlib
import io
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mounts as M  # noqa: E402

PEERS = '--peers', '10.0.0.1,10.0.0.2,10.0.0.3,10.0.0.4'


def run(argv, cwd):
    """mounts.main in cwd with no environment: {'home': .., 'tp': .., 'rw': [..], 'ro': [..]}, or the exit code."""
    out, old, env = io.StringIO(), os.getcwd(), dict(os.environ)
    os.environ.clear()
    try:
        os.chdir(cwd)
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            rc = M.main(list(argv))
    finally:
        os.chdir(old)
        os.environ.update(env)
    if rc:
        return rc
    got = {'rw': [], 'ro': []}
    for line in out.getvalue().splitlines():
        kind, value = line.split(' ', 1)
        if kind in ('rw', 'ro'):
            got[kind].append(value)
        else:
            got[kind] = value
    return got


class MountsTests(unittest.TestCase):
    def setUp(self):
        self.t = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.t)

    def test_local_release(self):
        got = run(['--tp', '4', *PEERS, '--model', '/data/release', '--model-cache', '/data/cache',
                   '--dflash-dir', '/data/dflash', '--g6-home', '/work/g6'], self.t)
        self.assertEqual(got['home'], '/work/g6')
        self.assertEqual(got['tp'], '4')
        self.assertEqual(got['rw'], ['/work/g6', '/data/cache'])
        self.assertIn('/data/release', got['ro'])
        self.assertIn('/data/dflash', got['ro'])

    def test_remote_model_is_not_mounted(self):
        got = run([*PEERS, '--model', 'https://huggingface.co/org/repo', '--model-cache', '/data/cache',
                   '--g6-home', '/work/g6'], self.t)
        self.assertFalse([p for p in got['ro'] if p.startswith('https:')])
        self.assertEqual(got['rw'], ['/work/g6', '/data/cache'])

    def test_tp_from_the_peer_list(self):
        self.assertEqual(run(['--peers', '10.0.0.1,10.0.0.2,10.0.0.3', '--g6-home', '/w'], self.t)['tp'], '3')
        self.assertEqual(run(['--tp', '6', '--peers', ','.join(f'10.0.0.{i}' for i in range(1, 7)),
                              '--g6-home', '/w'], self.t)['tp'], '6')

    def test_relative_paths_and_the_config_file_resolve_in_the_caller_directory(self):
        with open(os.path.join(self.t, 'cluster.env'), 'w') as f:
            f.write('TP=4\nPEERS=10.0.0.1,10.0.0.2,10.0.0.3,10.0.0.4\nMODEL=release\nG6_HOME=g6\n')
        got = run([], self.t)
        self.assertEqual(got['home'], os.path.join(self.t, 'g6'))
        self.assertIn(os.path.join(self.t, 'release'), got['ro'])
        self.assertIn(os.path.join(self.t, 'cluster.env'), got['ro'])

    def test_a_variant_name_is_not_a_path(self):
        got = run([*PEERS, '--model', '/data/release', '--variant', 'down1', '--g6-home', '/w'], self.t)
        self.assertNotIn('down1', ' '.join(got['ro']))
        got = run([*PEERS, '--model', '/data/release', '--variant', '/data/widths.json', '--g6-home', '/w'], self.t)
        self.assertIn('/data/widths.json', got['ro'])

    def test_a_bad_option_exits_2(self):
        self.assertEqual(run(['--no-such-option', 'x'], self.t), 2)

    def test_executables_must_be_in_the_image_or_under_g6_home(self):
        base = [*PEERS, '--g6-home', '/work/g6']
        self.assertEqual(run([*base, '--py', '/home/someone/venv/bin/python'], self.t), 2)     # a host venv
        self.assertEqual(run([*base, '--front-bin', '/home/someone/gr8/bin/g6front-rs'], self.t), 2)
        self.assertIsInstance(run([*base, '--py', sys.executable], self.t), dict)               # in the image
        self.assertIsInstance(run([*base, '--front-bin', '/work/g6/bin/g6front-rs'], self.t), dict)   # G6_HOME


if __name__ == '__main__':
    unittest.main()
