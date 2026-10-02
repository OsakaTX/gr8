"""Stream-check a pack against its manifest: every tensor's sha256, zero padding, exact size. Stdlib only.
Reads sequentially and drops the file's page cache as it goes (unified memory on GB10).

python3 checkpack.py DIR/rank{r}.manifest.json [...]   -> one JSON line per pack; exit 1 on any mismatch
"""
import hashlib
import json
import os
import sys
import time

CHUNK = 64 << 20


def check(man_path):
    t0 = time.time()
    m = json.load(open(man_path))
    pack = man_path.rsplit('.manifest.json', 1)[0] + '.pack'
    fd = os.open(pack, os.O_RDONLY)
    size = os.fstat(fd).st_size
    ents = sorted(m['tensors'], key=lambda e: e['offset'])
    bad, pos, i, h, nonzero_pad = [], 0, 0, None, 0
    # walk the file once; each byte belongs to a tensor or to padding
    bounds = [(e['offset'], e['offset'] + e['nbytes'], e) for e in ents]
    while pos < size:
        buf = os.pread(fd, min(CHUNK, size - pos), pos)
        if not buf:
            break
        mv, p = memoryview(buf), 0
        while p < len(buf):
            at = pos + p
            if i < len(bounds) and at >= bounds[i][0]:
                lo, hi, e = bounds[i]
                if h is None:
                    h = hashlib.sha256()
                take = min(hi - at, len(buf) - p)
                h.update(mv[p:p + take])
                p += take
                if at + take == hi:
                    if h.hexdigest() != e['sha256']:
                        bad.append(e['name'])
                    h, i = None, i + 1
            else:
                nxt = bounds[i][0] if i < len(bounds) else size
                take = min(nxt - at, len(buf) - p)
                if any(mv[p:p + take]):
                    nonzero_pad += 1
                p += take
        pos += len(buf)
        os.posix_fadvise(fd, 0, pos, os.POSIX_FADV_DONTNEED)
    os.close(fd)
    ok = not bad and not nonzero_pad and i == len(bounds) and size == m['pack_bytes']
    return dict(pack=pack, rank=m['rank'], ok=ok, bytes=size, expected=m['pack_bytes'], tensors=len(ents),
                sha256_mismatch=bad[:8], n_mismatch=len(bad), nonzero_padding=nonzero_pad,
                seconds=round(time.time() - t0, 1))


if __name__ == '__main__':
    rc = 0
    for p in sys.argv[1:]:
        r = check(p)
        print(json.dumps(r), flush=True)
        rc |= not r['ok']
    sys.exit(rc)
