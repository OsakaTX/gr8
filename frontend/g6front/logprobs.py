"""Engine logprob entries ({lp, rank, top:[[id, lp], ...]}) -> vLLM's response shapes."""

FLOOR = -9999.0


def _lp(x):
    return max(float(x), FLOOR)


def _tokstr(t, tid, as_ids):
    return f'token_id:{tid}' if as_ids else t.token_str(tid)


def _step_dict(tid, e):
    """vLLM's per-position dict: the actual/sampled token first (with its rank), then the top-k (ranks 1..k)."""
    d = {tid: (_lp(e['lp']), e.get('rank'))}
    for r, (i, v) in enumerate(e.get('top') or [], 1):
        if i not in d:
            d[i] = (_lp(v), r)
    return d


def prompt_logprobs(lps, prompt, t):
    """choices[0].prompt_logprobs: [None, {"<id>": {"logprob", "rank", "decoded_token"}, ...}, ...]"""
    if lps is None:
        return None
    out = []
    for j, e in enumerate(lps):
        if e is None:
            out.append(None)
            continue
        out.append({str(i): {'logprob': lp, 'rank': rk, 'decoded_token': t.token_str(i)}
                    for i, (lp, rk) in _step_dict(prompt[j], e).items()})
    return out


def completion_logprobs(ids, entries, k, t, as_ids=False, offset0=0):
    """/v1/completions `logprobs`: tokens, token_logprobs, top_logprobs (k+1 entries: sampled + top-k), text_offset."""
    toks, tlps, tops, offs = [], [], [], []
    off = offset0
    for tid, e in zip(ids, entries):
        s = _tokstr(t, tid, as_ids)
        toks.append(s)
        tlps.append(_lp(e['lp']))
        d = _step_dict(tid, e)
        top = {}
        for n, (i, (lp, _)) in enumerate(d.items()):
            if n > k:
                break
            top[_tokstr(t, i, as_ids)] = lp
        tops.append(top)
        offs.append(off)
        off += len(s)
    return {'text_offset': offs, 'token_logprobs': tlps, 'tokens': toks, 'top_logprobs': tops}


def chat_logprobs(ids, entries, k, t, as_ids=False):
    """/v1/chat/completions `logprobs.content`: [{token, logprob, bytes, top_logprobs:[...k]}]."""
    content = []
    for tid, e in zip(ids, entries):
        s = _tokstr(t, tid, as_ids)
        top = []
        for i, v in (e.get('top') or [])[:k]:
            ts = _tokstr(t, i, as_ids)
            top.append({'token': ts, 'logprob': _lp(v), 'bytes': list(ts.encode('utf-8', errors='replace'))})
        content.append({'token': s, 'logprob': _lp(e['lp']), 'bytes': list(s.encode('utf-8', errors='replace')),
                        'top_logprobs': top})
    return {'content': content}
