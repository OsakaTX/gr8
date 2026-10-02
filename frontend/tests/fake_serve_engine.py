#!/usr/bin/env python3
"""Fake engine for g6shim tests: the `--serve DIR` file side of g6m4/m4sim, no model.

Polls DIR/req.<n> for n = 0, 1, ...; answers in DIR/resp.<n> with MTP-like rounds (1 + accepted of k=2 drafts)
every --round-ms, cumulative `m` lines, and `end`. Emits the first stop id at token --stop-at (if >= min).
Exits on DIR/quit. With --rank R > 0 it only consumes requests (no resp), like a non-zero rank.

usage: fake_serve_engine.py DIR [--rank 0] [--round-ms 20] [--stop-at 0] [--prefill-ms 50]
"""
import argparse
import os
import time

ap = argparse.ArgumentParser()
ap.add_argument('dir')
ap.add_argument('--rank', type=int, default=0)
ap.add_argument('--round-ms', type=float, default=20.0)
ap.add_argument('--prefill-ms', type=float, default=50.0)
ap.add_argument('--stop-at', type=int, default=0, help='emit the first stop id as token #N (0 = never)')
a = ap.parse_args()
ACC = [2, 1, 2, 0, 2, 1]  # accepted drafts per round
rounds = drafts = accepted = 0
pos = [0, 0]
n = 0
log = open(os.path.join(a.dir, f'fake.rank{a.rank}.log'), 'a', buffering=1)
while True:
    if os.path.exists(os.path.join(a.dir, 'quit')):
        log.write('quit\n')
        break
    req = os.path.join(a.dir, f'req.{n}')
    if not os.path.exists(req):
        time.sleep(0.001)
        continue
    head, ids = open(req).read().splitlines()[:2]
    w = head.split()
    kv = dict(zip(w[0::2], w[1::2]))
    n_gen, mn = int(kv['n_gen']), int(kv['min'])
    stops = [] if kv['stop'] == '-' else [int(x) for x in kv['stop'].split(',')]
    prompt = [int(x) for x in ids.split(',')]
    log.write(f'req.{n} {head} prompt {len(prompt)}\n')
    if a.rank == 0:
        out = open(os.path.join(a.dir, f'resp.{n}'), 'w', buffering=1)
        time.sleep(a.prefill_ms / 1e3)
        made, reason, sid, r = 0, 'length', -1, 0
        while made < n_gen:
            k = 1 + ACC[r % len(ACC)]
            chunk = []
            for _ in range(k):
                if made >= n_gen:
                    break
                made += 1
                if stops and a.stop_at and made == max(a.stop_at, mn):
                    chunk.append(stops[0])
                    reason, sid = 'stop', stops[0]
                    break
                chunk.append(1000 + (prompt[-1] + made) % 5000)
            rounds += 1
            drafts += 2
            accepted += len(chunk) - 1
            for i in range(len(chunk) - 1):
                pos[i] += 1
            r += 1
            if r > 1:
                time.sleep(a.round_ms / 1e3)
            out.write('t ' + ' '.join(map(str, chunk)) + '\n')
            out.write(f'm {rounds} {drafts} {accepted} {pos[0]} {pos[1]}\n')
            if reason == 'stop':
                break
        out.write(f'end {reason} {sid} {made}\n')
        out.close()
    n += 1
