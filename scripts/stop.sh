#!/bin/bash
# Stop the front end, the shim and all six engine ranks started by launch_tp6.sh.
set -u
. "$(dirname "$0")/common.sh"
for p in front shim; do [ -f "$G6_HOME/out/$p.pid" ] && kill "$(cat "$G6_HOME/out/$p.pid")" 2>/dev/null; done
for ((r = 0; r < NRANK; r++)); do on $r "touch $G6_HOME/serve/quit" & done; wait   # each rank leaves its serve loop
sleep 10
for ((r = 0; r < NRANK; r++)); do
  on $r "p=\$(cat $G6_HOME/out/g6m4.pid 2>/dev/null); [ -n \"\$p\" ] && kill -0 \$p 2>/dev/null && kill \$p; echo rank $r stopped" &
done
wait
