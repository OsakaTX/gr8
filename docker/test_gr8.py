#!/usr/bin/env python3
"""docker/gr8 against a fake docker: what it starts, refuses and creates, with no daemon and no GPU.
    python3 -m unittest discover -s docker"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
GR8 = os.path.join(HERE, 'gr8')

# The fake docker. Every call is appended to $FAKE/log as JSON. Files in $FAKE script the answers:
#   keys.rc / mounts.out / mounts.rc / down.rc   exit codes and output of those container runs
#   ps.left                                      `docker ps` lists the container this many more times (default 0)
#   ps.after_start                               ... counting only from the first `docker run -d`
# `docker exec gr8 python3 -c CODE` runs CODE here, with G6H from the last `docker run -e G6H=...`.
FAKE = r'''#!/usr/bin/env python3
import json, os, subprocess, sys
d = os.environ['FAKE']
a = sys.argv[1:]
with open(os.path.join(d, 'log'), 'a') as f:
    f.write(json.dumps(a) + '\n')
def get(name, default=''):
    p = os.path.join(d, name)
    return open(p).read() if os.path.exists(p) else default
if a[0] == 'image':
    sys.exit(0)
if a[0] == 'ps':
    if os.path.exists(os.path.join(d, 'ps.after_start')) and not os.path.exists(os.path.join(d, 'started')):
        sys.exit(0)
    left = int(get('ps.left', '0') or 0)
    if left > 0:
        open(os.path.join(d, 'ps.left'), 'w').write(str(left - 1))
        print('c0ffee')
    sys.exit(0)
if a[0] == 'logs':
    sys.exit(0)
if a[0] == 'exec':
    code = a[a.index('-c') + 1]
    r = subprocess.run([sys.executable, '-c', code], env={**os.environ, 'G6H': get('g6h')})
    sys.exit(r.returncode)
if a[0] == 'run':
    for i, x in enumerate(a):
        if x.startswith('G6H='):
            open(os.path.join(d, 'g6h'), 'w').write(x[4:])
    if any('options.KEYS' in x for x in a):
        print('TP PEERS G6_HOME MODEL MODEL_CACHE PY FRONT_BIN')
        sys.exit(int(get('keys.rc', '0') or 0))
    if '/opt/gr8/docker/mounts.py' in a:
        sys.stdout.write(get('mounts.out'))
        sys.exit(int(get('mounts.rc', '0') or 0))
    if '-d' in a:
        open(os.path.join(d, 'started'), 'w').close()
        print('c0ffee')
        sys.exit(0)
    if 'down' in a:
        sys.exit(int(get('down.rc', '0') or 0))
    sys.exit(0)
sys.exit(0)
'''


@unittest.skipUnless(shutil.which('bash') and sys.platform.startswith('linux'), 'needs bash on Linux')
class WrapperTests(unittest.TestCase):
    def setUp(self):
        self.t = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.t)
        self.fake = os.path.join(self.t, 'fake')
        self.bin = os.path.join(self.t, 'bin')
        self.home = os.path.join(self.t, 'home')        # $HOME, without .ssh unless a test makes one
        self.g6 = os.path.join(self.t, 'g6')            # G6_HOME
        self.cache = os.path.join(self.t, 'cache')      # MODEL_CACHE
        for p in (self.fake, self.bin, self.home):
            os.makedirs(p)
        with open(os.path.join(self.bin, 'docker'), 'w') as f:
            f.write(FAKE)
        os.chmod(os.path.join(self.bin, 'docker'), 0o755)
        self.helper(f'home {self.g6}\ntp 4\nrw {self.g6}\nrw {self.cache}\n')

    def helper(self, out, rc=0):
        self.script('mounts.out', out)
        self.script('mounts.rc', str(rc))

    def script(self, name, text):
        with open(os.path.join(self.fake, name), 'w') as f:
            f.write(text)

    def gr8(self, *argv, **env):
        e = {'PATH': f'{self.bin}:/usr/bin:/bin', 'HOME': self.home, 'FAKE': self.fake, 'USER': os.environ.get('USER', ''),
             **env}
        return subprocess.run(['bash', GR8, *argv], env=e, cwd=self.t, capture_output=True, text=True, timeout=60)

    def calls(self):
        p = os.path.join(self.fake, 'log')
        if not os.path.exists(p):
            return []
        with open(p) as f:
            return [json.loads(x) for x in f]

    def started(self):
        """The calls that start the launcher: a detached up, or a one-shot ./gr8 glm53 command."""
        return [c for c in self.calls() if c[0] == 'run' and ('-d' in c or any('glm53' in x for x in c))]

    def test_a_helper_that_fails_part_way_starts_nothing(self):
        self.helper(f'home {self.g6}\ntp 4\n', rc=1)           # printed home and tp, then failed
        r = self.gr8('up')
        self.assertEqual(r.returncode, 2)
        self.assertIn('the options do not work here', r.stderr)
        self.assertEqual(self.started(), [])
        self.assertFalse(os.path.exists(self.g6))

    def test_unreadable_option_names_start_nothing(self):
        self.script('keys.rc', '1')
        r = self.gr8('status')
        self.assertEqual(r.returncode, 2)
        self.assertFalse([c for c in self.calls() if '/opt/gr8/docker/mounts.py' in c])
        self.assertEqual(self.started(), [])

    def test_restart_does_not_start_when_down_fails(self):
        os.makedirs(self.g6)
        self.script('down.rc', '3')
        r = self.gr8('restart')
        self.assertEqual(r.returncode, 2)
        self.assertIn('down failed', r.stderr)
        self.assertFalse([c for c in self.calls() if c[0] == 'run' and '-d' in c])

    def test_restart_waits_for_the_old_container(self):
        os.makedirs(self.g6)
        self.script('ps.left', '2')                            # still listed for two looks after down
        self.gr8('restart')                                    # the new container then stops at once: rc 1
        calls = self.calls()
        down = next(i for i, c in enumerate(calls) if c[0] == 'run' and 'down' in c)
        start = next(i for i, c in enumerate(calls) if c[0] == 'run' and '-d' in c)
        looks = [c for c in calls[down:start] if c[0] == 'ps']
        self.assertGreaterEqual(len(looks), 3)                 # two that still list it, one that does not

    def test_read_only_commands_create_nothing(self):
        os.makedirs(self.g6)
        r = self.gr8('status')
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertFalse(os.path.exists(self.cache))
        run = self.started()[0]
        self.assertFalse([x for x in run if self.cache in x])  # not mounted: the launcher names it

    def test_a_read_only_command_with_no_g6_home_says_so(self):
        r = self.gr8('status')
        self.assertEqual(r.returncode, 2)
        self.assertIn('nothing has run on this box', r.stderr)
        self.assertFalse(os.path.exists(self.g6))

    def test_up_creates_g6_home_and_the_cache(self):
        self.gr8('up')                                         # the fake container stops at once
        self.assertTrue(os.path.isdir(self.g6))
        self.assertTrue(os.path.isdir(self.cache))
        self.assertTrue([c for c in self.calls() if c[0] == 'run' and '-d' in c])

    def test_ssh_is_mounted_only_when_there_is_one_or_it_is_named(self):
        os.makedirs(self.g6)
        self.gr8('status')
        self.assertFalse([x for x in self.started()[0] if '.ssh' in x])
        os.makedirs(os.path.join(self.home, '.ssh'))
        os.remove(os.path.join(self.fake, 'log'))
        self.gr8('status')
        self.assertTrue([x for x in self.started()[0] if f'src={self.home}/.ssh' in x])
        os.remove(os.path.join(self.fake, 'log'))
        self.gr8('status', GR8_SSH='/no/such/dir')               # named: mounted as given, docker refuses it
        self.assertTrue([x for x in self.started()[0] if 'src=/no/such/dir' in x])

    def boot(self, state_pid, my_pid, looks):
        os.makedirs(os.path.join(self.g6, 'out'))
        with open(os.path.join(self.g6, 'out', 'gr8.state'), 'w') as f:
            json.dump({'state': 'serving', 'rank': 0, 'pid': state_pid, 'time': 0}, f)
        with open(os.path.join(self.g6, 'out', 'docker-up.pid'), 'w') as f:
            f.write(str(my_pid))
        self.script('ps.left', str(looks))
        self.script('ps.after_start', '')
        return self.gr8('up')

    def test_up_returns_once_this_boots_supervisor_serves(self):
        r = self.boot(state_pid=os.getpid(), my_pid=os.getpid(), looks=50)    # a live pid stands in for it
        self.assertEqual(r.returncode, 0, r.stderr)

    def test_a_killed_boots_leftovers_are_not_this_boot(self):
        p = subprocess.Popen(['true'])
        p.wait()                                               # a pid that no longer runs
        r = self.boot(state_pid=p.pid, my_pid=p.pid, looks=3)  # both files name the same dead supervisor
        self.assertEqual(r.returncode, 1)
        self.assertIn('the container stopped', r.stderr)

    def test_an_older_boots_serving_state_is_not_this_boot(self):
        r = self.boot(state_pid=1111, my_pid=4242, looks=3)    # serving, but another supervisor's pid
        self.assertEqual(r.returncode, 1)
        self.assertIn('the container stopped', r.stderr)


if __name__ == '__main__':
    unittest.main()
