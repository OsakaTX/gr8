#!/bin/bash
# Copy the engine binary and the packs to every node (run on the head).
#   scripts/stage.sh <BIN_DIR> <PACK_DIR>    BIN_DIR has g6m4; PACK_DIR has rank{0..5}.* and qo.*
# Rank r gets ONLY its own rank pack (~62 GB) plus the shared qo pack (~13 GB). Packs must sit on each node's
# own disk: the launcher loads them locally (streaming a pack over ssh at boot was 2-3x slower for us).
set -eu
. "$(dirname "$0")/common.sh"
BIN=$1; P=$2
for ((r = 0; r < NRANK; r++)); do
  h=${HOST[$r]}
  on $r "mkdir -p $G6_HOME/bin $G6_HOME/packs $G6_HOME/serve $G6_HOME/out"
  rsync -a "$BIN/g6m4" "$h:$G6_HOME/bin/" &
  rsync -a --partial "$P/rank$r.pack" "$P/rank$r.manifest.json" "$P/qo.pack" "$P/qo.manifest.json" "$h:$G6_HOME/packs/" &
done
wait
for ((r = 0; r < NRANK; r++)); do
  echo "rank $r ${HOST[$r]}: $(on $r "md5sum $G6_HOME/bin/g6m4 | cut -c1-12 | tr -d '\n'; echo -n ' '; ls $G6_HOME/packs | tr '\n' ' '; df -h $G6_HOME | tail -1 | awk '{print \"free \" \$4}'")"
done
