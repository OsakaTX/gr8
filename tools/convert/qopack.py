"""QO pack (QO-INTEGRATION, G6_PF_OWN=3): every layer's full-head attention weights, assembled from the six rank
packs' head slices. Stdlib only (the Sparks have no numpy).

Rank r holds head group g = (r - l % 6) mod 6 of layer l, heads QO_EDGES[g] .. QO_EDGES[g + 1]. Output tensors,
named L%03d.qo.<t> (engine/runtime/qo_weights.cuh qo_from_pack), same byte assembly as qo_from_ranks:
  q_b.mx.codes [16384][2048] u8, q_b.mx.scales [16384][64] u8      rows = heads, concatenated in head order
  o_proj.mx.codes [6144][16384] u8, o_proj.mx.scales [6144][512] u8  cols = heads, each row's slices concatenated
  kv_b.uk [64][192][512] bf16, kv_b.uv [64][256][512] bf16          dim 0 = heads, concatenated

python3 qopack.py OUT.pack SRC0 .. SRC5 [--layers 0-77]
  SRCr = a local rank manifest path (rankR.manifest.json next to rankR.pack), or HOST:PATH to stream the slices from
  another node over ssh (this script runs there with --extract). Every source tensor's sha256 is checked against its
  manifest. Writes OUT.pack + OUT.manifest.json (format g6pack-v1, 4096-byte alignment, deterministic).
"""
import hashlib
import json
import os
import subprocess
import sys

ALIGN = 4096
QO_EDGES = (0, 11, 22, 33, 44, 54, 64)
SRC = ('q_b.mx.codes', 'q_b.mx.scales', 'o_proj.mx.codes', 'o_proj.mx.scales', 'kv_b.uk', 'kv_b.uv')
OUT = {'q_b.mx.codes': ([16384, 2048], 'u8'), 'q_b.mx.scales': ([16384, 64], 'u8'),
       'o_proj.mx.codes': ([6144, 16384], 'u8'), 'o_proj.mx.scales': ([6144, 512], 'u8'),
       'kv_b.uk': ([64, 192, 512], 'bf16'), 'kv_b.uv': ([64, 256, 512], 'bf16')}


def entries(man, layers):
    """the source tensors this rank contributes, in stream order (layer, SRC)"""
    by = {e['name']: e for e in man['tensors']}
    return [by[f'L{l:03d}.{t}'] for l in layers for t in SRC]


def pread_all(fd, off, n):
    out, got = bytearray(n), 0
    mv = memoryview(out)
    while got < n:
        k = os.preadv(fd, [mv[got:]], off + got)
        if k <= 0:
            raise IOError(f'short read at {off + got}')
        got += k
    return bytes(out)


class Local:
    def __init__(self, man_path, layers):
        self.man = json.load(open(man_path))
        self.fd = os.open(man_path.rsplit('.manifest.json', 1)[0] + '.pack', os.O_RDONLY)
        self.ents = iter(entries(self.man, layers))

    def next(self):
        e = next(self.ents)
        b = pread_all(self.fd, e['offset'], e['nbytes'])
        os.posix_fadvise(self.fd, e['offset'], e['nbytes'], os.POSIX_FADV_DONTNEED)
        return e, b


class Remote:
    """this script with --extract on HOST: the same tensors as raw bytes on stdout, back to back"""
    def __init__(self, spec, layers, lspec):
        host, path = spec.split(':', 1)
        self.man = json.loads(subprocess.run(['ssh', '-o', 'BatchMode=yes', host, 'cat', path], check=True,
                                             capture_output=True).stdout)
        src = open(__file__).read()
        self.p = subprocess.Popen(['ssh', '-o', 'BatchMode=yes', host, 'python3', '-', '--extract', path, lspec],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        self.p.stdin.write(src.encode())
        self.p.stdin.close()
        self.ents = iter(entries(self.man, layers))

    def next(self):
        e = next(self.ents)
        b = self.p.stdout.read(e['nbytes'])
        if len(b) != e['nbytes']:
            raise IOError(f'{e["name"]}: stream ended')
        return e, b


def parse_layers(s):
    out = []
    for part in s.split(','):
        a, _, b = part.partition('-')
        out += list(range(int(a), int(b or a) + 1))
    return out


def extract(man_path, lspec):
    src, w = Local(man_path, parse_layers(lspec)), sys.stdout.buffer
    for _ in range(len(parse_layers(lspec)) * len(SRC)):
        w.write(src.next()[1])
    w.flush()


def main():
    if sys.argv[1] == '--extract':
        return extract(sys.argv[2], sys.argv[3])
    args = sys.argv[1:]
    lspec = '0-77'
    if '--layers' in args:
        i = args.index('--layers'); lspec = args[i + 1]; del args[i:i + 2]
    out, srcs = args[0], args[1:]
    if len(srcs) != 6:
        sys.exit('need six sources, rank 0 .. 5')
    layers = parse_layers(lspec)
    R = [Remote(s, layers, lspec) if ':' in s else Local(s, layers) for s in srcs]
    for r, s in enumerate(R):
        if s.man['rank'] != r:
            sys.exit(f'source {r} is rank {s.man["rank"]}')
    fd = os.open(out, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    off, ents = 0, []

    def raw(b):
        nonlocal off
        mv = memoryview(b)
        n = 0
        while n < len(mv):
            n += os.write(fd, mv[n:])
        off += len(mv)

    for l in layers:
        got = [dict() for _ in range(6)]   # by group
        for r in range(6):
            g = (r - l % 6) % 6
            h = QO_EDGES[g + 1] - QO_EDGES[g]
            for t in SRC:
                e, b = R[r].next()
                if e['name'] != f'L{l:03d}.{t}':
                    sys.exit(f'stream order: got {e["name"]}')
                if hashlib.sha256(b).hexdigest() != e['sha256']:
                    sys.exit(f'rank {r} {e["name"]}: sha256 mismatch')
                if e.get('heads') != [QO_EDGES[g], QO_EDGES[g + 1]]:
                    sys.exit(f'rank {r} {e["name"]}: heads {e.get("heads")} are not group {g}')
                got[g][t] = (e, b, h)
        for t in SRC:
            shape, dt = OUT[t]
            pad = (-off) % ALIGN
            raw(bytes(pad))
            hs, o0 = hashlib.sha256(), off
            if t.startswith('o_proj'):
                cols = [got[g][t][1] for g in range(6)]
                wd = [got[g][t][0]['shape'][1] for g in range(6)]
                if sum(wd) != shape[1]:
                    sys.exit(f'L{l:03d}.{t}: widths {wd}')
                for row in range(shape[0]):
                    b = b''.join(cols[g][row * wd[g]:(row + 1) * wd[g]] for g in range(6))
                    hs.update(b); raw(b)
            else:
                for g in range(6):
                    b = got[g][t][1]
                    hs.update(b); raw(b)
            nb = off - o0
            esz = 2 if dt == 'bf16' else 1
            n = 1
            for s in shape:
                n *= s
            if nb != n * esz:
                sys.exit(f'L{l:03d}.{t}: wrote {nb} bytes, expected {n * esz}')
            ents.append(dict(name=f'L{l:03d}.qo.{t}', dtype=dt, shape=shape, offset=o0, nbytes=nb,
                             sha256=hs.hexdigest(), layout=got[0][t][0]['layout'], heads=[0, 64]))
        os.fdatasync(fd)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        print(f'L{l:03d} done, {off / 1e9:.2f} GB', flush=True)
    raw(bytes((-off) % ALIGN))
    os.fsync(fd)
    os.close(fd)
    for s in R:
        if isinstance(s, Remote) and s.p.wait() != 0:
            sys.exit('remote extract failed')
    man = dict(format='g6pack-v1', rank=0, align=ALIGN, pack_bytes=off, tensors=ents, qo=True, layers=layers,
               converter='qopack-v1', sources=[s.man.get('converter') for s in R])
    mp = out.rsplit('.pack', 1)[0] + '.manifest.json'
    with open(mp + '.tmp', 'w') as f:
        json.dump(man, f, indent=1, sort_keys=True)
        f.write('\n')
    os.replace(mp + '.tmp', mp)
    print(f'wrote {out} ({off} bytes, {len(ents)} tensors) + {mp}')


if __name__ == '__main__':
    main()
