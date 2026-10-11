#!/bin/bash
# Build the rank packs + the shared QO pack from the two public inputs (CPU + disk only, no GPU):
#   TR3 = davidsyoung/GLM-5.3-EXL3-TR3-3.25bpw   (EXL3 3.25 bpw experts + BF16 non-expert weights)
#   D8  = adapt-ai-systems/glm-5.3-full-exl3-gptq-dense, folder tp6/r0..r5  (GPTQ int4 g32 dense decode codes)
#         or, for TP4, the tp4/r0..r3 codes (rank-specific: TP6 codes do not work at TP4)
# usage: scripts/build_packs.sh <TR3_DIR> <D8_DIR (has r0..r5, or r0..r3 for TP4)> <OUT_DIR>
# The rank count follows the D8 dir: r0..r5 = TP6, r0..r3 (no r4) = TP4 (run the engine built with G6_NR=4).
# Output: OUT/rank{0..5}.pack + .manifest.json (~62 GB each; TP4: rank{0..3}, ~90 GB each),
#         OUT/qo.pack + qo.manifest.json (~13 GB, the same bytes at TP4 and TP6).
# Needs ~390 GB free in OUT (TP4: ~375 GB) and python3 + numpy (no safetensors package: the reader is tools/convert/st.py).
# Our TP6 conversion ran ~15-25 min on one GB10 node; one pass writes all ranks.
set -eu
TR3=$1; D8=$2; OUT=$3
REPO=$(cd "$(dirname "$0")/.." && pwd)
[ -d "$D8/r0" ] || { echo "no $D8/r0: pass the GPTQ repo's tp4/ or tp6/ folder, not the repo itself" >&2; exit 2; }
mkdir -p "$OUT"
if [ -d "$D8/r4" ]; then NR=6; else NR=4; fi   # no r4: TP4, or an incomplete TP6 download (the hashes below catch that)
# the D8 dirs must be the rank-specific codes we built from; these are their layer-000 sha256 prefixes
if [ $NR = 6 ]; then
  declare -A SHA=([0]=b393ef73d3fa [1]=cbccab5f65b4 [2]=b883e0e7e6d9 [3]=ff516768d5dc [4]=d019939fad71 [5]=1f33849c3985)
else
  declare -A SHA=([0]=05db601b67fd [1]=d535fd5ebf3e [2]=4eedc593ec7b [3]=02e7e618ba7f)
fi
RANKS=$(seq 0 $((NR - 1)))
D8S=""
for r in $RANKS; do
  h=$(sha256sum "$D8/r$r/layer-000.safetensors" | cut -c1-12)
  [ "$h" = "${SHA[$r]}" ] || { echo "D8 r$r layer-000 sha256 $h != ${SHA[$r]} (TP$NR expected: wrong rank order, other codes,"\
    "or, with no r4/r5, an incomplete TP6 dir rather than TP4)"; exit 1; }
  n=$(ls "$D8/r$r" | grep -c '^layer-0[0-7][0-9].safetensors$')
  [ "$n" = 78 ] || { echo "D8 r$r has $n layer files, want 78"; exit 1; }
  D8S="$D8S $D8/r$r"
done
[ $NR = 6 ] || echo "no $D8/r4: building TP4 packs (if this is an incomplete TP6 download, stop and fetch r4/r5)"
echo "building TP$NR packs"
cd "$REPO"
# shellcheck disable=SC2086
python3 -m tools.convert.convert --tr3 "$TR3" --d8 $D8S --out "$OUT" --gv-records --drop-cache
for r in $RANKS; do python3 tools/convert/checkpack.py "$OUT/rank$r.manifest.json" > "$OUT/check_r$r.json" & done
wait
ok=$(cat "$OUT"/check_r*.json | grep -c '"ok": true' || true)
echo "checkpack: $ok/$NR packs ok"
[ "$ok" = $NR ] || exit 1
# shellcheck disable=SC2046
python3 tools/convert/qopack.py "$OUT/qo.pack" $(for r in $RANKS; do echo "$OUT/rank$r.manifest.json"; done)
ls -l "$OUT"/*.pack
