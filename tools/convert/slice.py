"""Cut a mini pack (chosen layers + globals) out of a full rank pack. Stdlib only (the Sparks have no numpy).

Same format, names, dtypes, shapes, sha256 and per-tensor manifest fields as the source; only offsets change, so the
loader and checkpack.py work on the result unchanged. Every tensor's sha256 is re-checked while its bytes are copied.
Reads with pread and drops the page cache of both files as it goes (unified memory on GB10).

python3 slice.py SRC/rank{r}.manifest.json OUTDIR 0,3,40,77 [--no-embed]
  -> OUTDIR/rank{r}.pack + OUTDIR/rank{r}.manifest.json, one JSON line on stdout; exit 1 on a checksum mismatch
"""
import hashlib
import json
import os
import sys
import time

CHUNK = 64 << 20
TOP = ('format', 'rank', 'align', 'pack_bytes', 'tensors', 'layers')


def cut(man_path, outdir, layers, embed=True):
    t0 = time.time()
    m = json.load(open(man_path))
    src = man_path.rsplit('.manifest.json', 1)[0] + '.pack'
    align = m['align']
    want = tuple('L%03d.' % l for l in layers) + ('G.',)
    ents = [e for e in sorted(m['tensors'], key=lambda e: e['offset'])
            if e['name'].startswith(want) and (embed or e['name'] != 'G.embed')]
    missing = [l for l in layers if not any(e['name'].startswith('L%03d.' % l) for e in ents)]
    if missing:
        raise SystemExit(f'{man_path}: no tensors for layers {missing}')
    os.makedirs(outdir, exist_ok=True)
    dst = os.path.join(outdir, os.path.basename(src))
    fi = os.open(src, os.O_RDONLY)
    fo = os.open(dst, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    off, out, bad = 0, [], []
    for e in ents:
        pad = (-off) % align
        if pad:
            os.write(fo, bytes(pad))
            off += pad
        h, pos, end = hashlib.sha256(), e['offset'], e['offset'] + e['nbytes']
        while pos < end:
            buf = os.pread(fi, min(CHUNK, end - pos), pos)
            if not buf:
                raise SystemExit(f'{src}: short read in {e["name"]}')
            h.update(buf)
            mv, n = memoryview(buf), 0
            while n < len(mv):
                n += os.write(fo, mv[n:])
            pos += len(buf)
        os.posix_fadvise(fi, e['offset'], e['nbytes'], os.POSIX_FADV_DONTNEED)
        if h.hexdigest() != e['sha256']:
            bad.append(e['name'])
        out.append(dict(e, offset=off))
        off += e['nbytes']
    pad = (-off) % align
    if pad:
        os.write(fo, bytes(pad))
        off += pad
    os.fsync(fo)
    os.posix_fadvise(fo, 0, 0, os.POSIX_FADV_DONTNEED)
    os.close(fo)
    os.close(fi)
    man = dict({k: v for k, v in m.items() if k not in TOP}, format=m['format'], rank=m['rank'], align=align,
               pack_bytes=off, tensors=out, layers=list(layers), slice_of_pack_bytes=m['pack_bytes'],
               slice_has_embed=bool(embed))
    mp = dst.rsplit('.pack', 1)[0] + '.manifest.json'
    with open(mp + '.tmp', 'w') as f:
        json.dump(man, f, indent=1, sort_keys=True)
        f.write('\n')
    os.replace(mp + '.tmp', mp)
    return dict(pack=dst, rank=m['rank'], ok=not bad, tensors=len(out), bytes=off, layers=list(layers),
                sha256_mismatch=bad[:8], seconds=round(time.time() - t0, 1))


if __name__ == '__main__':
    a = [x for x in sys.argv[1:] if not x.startswith('--')]
    r = cut(a[0], a[1], [int(x) for x in a[2].split(',')], embed='--no-embed' not in sys.argv)
    print(json.dumps(r), flush=True)
    sys.exit(0 if r['ok'] else 1)
