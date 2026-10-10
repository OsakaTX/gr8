#!/usr/bin/env python3
"""What docker/gr8 mounts into the container, from the launcher's own option parsing.

    python3 docker/mounts.py [OPTIONS] [@FILE]      in the image, in the caller's working directory

Prints `home PATH`, `tp N`, then one `rw PATH` or `ro PATH` per host path the command reads or writes: G6_HOME and the
model cache read-write, the release, its parts, the drafter directory and the config file read-only. Each path is
absolute and is mounted at the same path inside the container: the head's shim names a worker's rank directory by the
path that worker reported, and writes it over ssh on the worker's host.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'tools', 'launch'))
import options as opt  # noqa: E402

RW = ('G6_HOME', 'MODEL_CACHE')
RO = ('MODEL', 'MODEL_DIR', 'V2', 'V2_SLICES', 'V2_WIDTHS', 'VARIANT', 'BASE_PACKS', 'DFLASH_DIR', 'V2_DENSE')


def mounts(o):
    """[(mode, path)] for every path option that is set and local, plus the config file."""
    out = [('rw', o.get(k)) for k in RW if o.get(k)]
    for k in RO:
        v = o.get(k)
        if not v or (k == 'MODEL' and opt.looks_remote(v)) or not os.path.isabs(v):   # VARIANT may be a name
            continue
        out.append(('ro', v))
    if o.file:
        out.append(('ro', os.path.abspath(o.file)))
    return out


def tp(o):
    if o.get('TP'):
        return o.int('TP')
    peers = opt.split_list(o.get('PEERS') or '')
    return len(peers) if peers else 4


def main(argv):
    try:
        o = opt.parse(argv)
        n = tp(o)
    except opt.OptionError as e:
        print(f'gr8: {e}', file=sys.stderr)
        return 2
    print(f'home {os.path.abspath(o.get("G6_HOME"))}')
    print(f'tp {n}')
    for mode, path in mounts(o):
        print(f'{mode} {path}')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
