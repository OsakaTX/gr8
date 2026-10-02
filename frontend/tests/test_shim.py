#!/usr/bin/env python3
"""Raw PROTOCOL.md checks of g6shim.py against tests/fake_serve_engine.py (stdlib only).

usage: test_shim.py SOCK   (the shim and the fake engine(s) already running; fake rank 0 with --stop-at 10)
"""
import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from g6front import wire  # noqa: E402

EOS = [154820, 154827, 154829]
fails = []


def check(name, ok, info=''):
    print(('PASS ' if ok else 'FAIL ') + name + (f'  [{info}]' if info else ''))
    if not ok:
        fails.append(name)


async def main(sock):
    r, w = await asyncio.open_unix_connection(sock, limit=wire.MAX_FRAME)
    hello = await wire.read_frame(r)
    check('hello', hello['type'] == 'hello' and hello['version'] == 1 and hello['eos_ids'] == EOS
          and hello['vocab_size'] == 154880, str(hello))
    frames: dict[str, list] = {}
    done: dict[str, asyncio.Event] = {}
    metrics = asyncio.Queue()

    async def reader():
        while True:
            f = await wire.read_frame(r)
            if f is None:
                return
            if f['type'] == 'metrics':
                await metrics.put(f)
                continue
            frames.setdefault(f['id'], []).append((time.monotonic(), f))
            if f['type'] == 'finish':
                done.setdefault(f['id'], asyncio.Event()).set()

    rt = asyncio.create_task(reader())

    def gen(rid, **kw):
        f = {'type': 'generate', 'id': rid, 'prompt': [1, 2, 3, 4, 5], 'max_tokens': 32, 'min_tokens': 0,
             'temperature': 0.0, 'top_p': 1.0, 'top_k': 0, 'min_p': 0.0, 'seed': None, 'stop_ids': [],
             'logprobs': None, 'prompt_logprobs': None}
        f.update(kw)
        done.setdefault(rid, asyncio.Event())
        wire.write_frame(w, f)

    async def wait(rid, t=20):
        await asyncio.wait_for(done[rid].wait(), t)
        fs = [f for _, f in frames[rid]]
        ids = [i for f in fs if f['type'] == 'tokens' for i in f['ids']]
        return ids, fs[-1], fs

    # 1. length: 32 tokens, several multi-token frames, finish length
    gen('a', max_tokens=32)
    ids, fin, fs = await wait('a')
    nt = sum(f['type'] == 'tokens' for f in fs)
    check('length finish', fin['reason'] == 'length' and fin['completion_tokens'] == 32 == len(ids)
          and fin['prompt_tokens'] == 5 and fin['stop_id'] is None, str(fin))
    check('MTP frames carry >1 id', nt < 32 and max(len(f['ids']) for f in fs if f['type'] == 'tokens') > 1,
          f'{nt} token frames for 32 ids')
    # 2. stop id (fake emits stop_ids[0] as token #10)
    gen('b', stop_ids=EOS)
    ids, fin, _ = await wait('b')
    check('stop finish', fin['reason'] == 'stop' and fin['stop_id'] == EOS[0] and ids[-1] == EOS[0]
          and fin['completion_tokens'] == len(ids) == 10, str(fin))
    # 3. min_tokens pushes the stop out
    gen('c', stop_ids=EOS, min_tokens=20)
    ids, fin, _ = await wait('c')
    check('min_tokens', fin['reason'] == 'stop' and len(ids) == 20, f'{len(ids)} {fin}')
    # 4. queue: two at once run in order; cancel a running one -> abort at once; next request still clean
    gen('d', max_tokens=200)
    gen('e', max_tokens=16)
    await asyncio.sleep(0.3)
    t0 = time.monotonic()
    wire.write_frame(w, {'type': 'cancel', 'id': 'd'})
    _, fin, _ = await wait('d')
    check('cancel running -> abort now', fin['reason'] == 'abort' and time.monotonic() - t0 < 0.2, str(fin))
    ids, fin, _ = await wait('e')
    check('queued request after cancel', fin['reason'] == 'length' and len(ids) == 16, str(fin))
    # 5. cancel a queued request: it never reaches the engine
    gen('f', max_tokens=100)
    gen('g', max_tokens=8)
    await asyncio.sleep(0.05)
    wire.write_frame(w, {'type': 'cancel', 'id': 'g'})
    _, fin_g, _ = await wait('g')
    ids, fin_f, _ = await wait('f')
    check('cancel queued', fin_g['reason'] == 'abort' and fin_f['reason'] == 'length' and len(ids) == 100)
    # 6. unsupported
    gen('h', logprobs=5)
    _, fin, _ = await wait('h')
    check('logprobs -> error', fin['reason'] == 'error' and 'not supported' in fin.get('message', ''), str(fin))
    gen('i', max_tokens=5000)
    _, fin, _ = await wait('i')
    check('too long -> error', fin['reason'] == 'error', str(fin))
    # 7. seeded sampling request goes through
    gen('j', temperature=0.7, seed=42, top_p=0.9, max_tokens=4)
    ids, fin, _ = await wait('j')
    check('temp/seed request', fin['reason'] == 'length' and len(ids) == 4, str(fin))
    # 8. metrics
    wire.write_frame(w, {'type': 'metrics', 'seq': 7})
    m = await asyncio.wait_for(metrics.get(), 5)
    s = m['spec']
    check('metrics', m['seq'] == 7 and s['drafts'] > 0 and s['draft_tokens'] == 2 * s['drafts']
          and 0 < s['accepted'] <= s['draft_tokens'] and len(s['accepted_per_pos']) == 2
          and m['requests']['finished'].get('abort') == 2, str(m))
    w.close()
    rt.cancel()
    print('ALL PASS' if not fails else f'{len(fails)} FAILED: {fails}')
    return 1 if fails else 0


sys.exit(asyncio.run(main(sys.argv[1])))
