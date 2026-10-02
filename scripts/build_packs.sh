#!/bin/bash
# Build the six rank packs + the shared QO pack from the two public inputs (CPU + disk only, no GPU):
#   TR3 = davidsyoung/GLM-5.3-EXL3-TR3-3.25bpw   (EXL3 3.25 bpw experts + BF16 non-expert weights)
#   D8  = adapt-ai-systems/glm-5.3-full-exl3-gptq-dense, folder tp6/r0..r5  (GPTQ int4 g32 dense decode codes)
# usage: scripts/build_packs.sh <TR3_DIR> <D8_TP6_DIR (has r0..r5)> <OUT_DIR>
# Output: OUT/rank{0..5}.pack + .manifest.json (~62 GB each), OUT/qo.pack + qo.manifest.json (~13 GB).
# Needs ~390 GB free in OUT and python3 + numpy (no safetensors package: the reader is tools/convert/st.py).
# Our conversion ran ~15-25 min on one GB10 node; one pass writes all six ranks.
set -eu
TR3=$1; D8=$2; OUT=$3
REPO=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$OUT"
# the D8 dirs must be the published TP6 codes (rank-specific); these are the layer-000 sha256 prefixes we built from
declare -A SHA=([0]=b393ef73d3fa [1]=cbccab5f65b4 [2]=b883e0e7e6d9 [3]=ff516768d5dc [4]=d019939fad71 [5]=1f33849c3985)
for r in 0 1 2 3 4 5; do
  h=$(sha256sum "$D8/r$r/layer-000.safetensors" | cut -c1-12)
  [ "$h" = "${SHA[$r]}" ] || { echo "D8 r$r layer-000 sha256 $h != ${SHA[$r]} (wrong rank order or other codes)"; exit 1; }
  n=$(ls "$D8/r$r" | grep -c '^layer-0[0-7][0-9].safetensors$')
  [ "$n" = 78 ] || { echo "D8 r$r has $n layer files, want 78"; exit 1; }
done
cd "$REPO"
python3 -m tools.convert.convert --tr3 "$TR3" --d8 "$D8"/r0 "$D8"/r1 "$D8"/r2 "$D8"/r3 "$D8"/r4 "$D8"/r5 \
  --out "$OUT" --gv-records --drop-cache
for r in 0 1 2 3 4 5; do python3 tools/convert/checkpack.py "$OUT/rank$r.manifest.json" > "$OUT/check_r$r.json" & done
wait
ok=$(cat "$OUT"/check_r*.json | grep -c '"ok": true' || true)
echo "checkpack: $ok/6 packs ok"
[ "$ok" = 6 ] || exit 1
python3 tools/convert/qopack.py "$OUT/qo.pack" "$OUT"/rank{0,1,2,3,4,5}.manifest.json
ls -l "$OUT"/*.pack
