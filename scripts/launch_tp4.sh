#!/bin/bash
# ONE command: boot the K1H2 engine on FOUR Sparks (TP4) and serve an OpenAI-compatible API on the head (rank 0).
#   scripts/launch_tp4.sh            (after G6_NR=4 scripts/build.sh, build_packs.sh with the TP4 codes, stage.sh;
#                                     reads config/cluster.env with four HOSTS)
# The same engine settings as launch_tp6.sh; only the rank count, the KV pool and the library path differ.
# Starts, in order: g6m4 on every node (ranks 3..0, detached) -> waits for all four serve/ready files ->
# g6shim.py (engine socket, drives the four ranks through serve/ files) -> g6front on :$PORT.
# Stop with scripts/stop.sh. Logs: $G6_HOME/out/ on each node (engine), $G6_HOME/out/{shim,front}.log on the head.
# If $G6_HOME/bin/lib exists (build.sh BUNDLE_LIBS=1), the engine finds cuBLAS there.
#
# Knobs (environment):
#   CAP=393216        KV pool / max context in tokens. A TP4 rank holds ~90 GB of weights against ~62 GB at TP6, so
#                     the pool is smaller than TP6's 614400. At 393216 (PF_OWN=1) the lowest MemAvailable we saw while
#                     serving, a 256K prompt included, was 6.6 GiB; lower it if a node runs short of memory
#   MTP=2             speculative draft length k (2 = our prose/served default; 4-6 helped code/structured)
#   PF_OWN=1          G6_PF_OWN (see below; 3 = QO attention, needs the qo pack and more memory than a TP4 rank has)
#   NOHASH=1          skip the sha256 check of every pack tensor at load (faster boot; run it once without first)
#   G6ENV_X="A=1 B=2" extra engine env for every rank (see docs/SETTINGS.md)
#   READY_TMO=900     seconds to wait for all four ranks to load
set -eu
. "$(dirname "$0")/common.sh"
[ "$NRANK" = 4 ] || { echo "launch_tp4.sh needs 4 HOSTS, config has $NRANK"; exit 2; }
CAP=${CAP:-393216}; MTP=${MTP:-2}; READY_TMO=${READY_TMO:-900}

# K1H2 = the served prefill/decode env + context parallel KV (CP) past R=32768 tokens with the HASH2 staged-page
# check. Every variable is explained in docs/SETTINGS.md. One difference from TP6: G6_PF_OWN defaults to 1, not 3.
# OWN=3 needs the 13 GB full-head QO pack on every rank, and with a 90 GB TP4 rank pack that no longer fits a GB10
# (the ranks ran out of memory in graph capture even at a 64K pool). PF_OWN=3 brings it back if memory is freed.
PF_OWN=${PF_OWN:-1}
ENGINE_ENV="G6_PF_TILE=64 G6_PF_DENSE=8192 G6_PF_RS=1 G6_PF_STREAMS=2 G6_PF_ATTN_ROWS=1024 G6_PF_OWN=$PF_OWN \
G6_PF_BLAS_ROWS=1024 G6_PF_TIMES=1 G6_PF_HASH_GPU=1 G6_PF_RSDIRECT=1 G6_DEC_SPLIT=3 G6_DEC_SHARED_SCRATCH=1 \
G6_CP=1 G6_CP_R=32768 G6_CP_S3=1 G6_CP_PF_VERIFY=1 G6_CP_PF_HASH2=1 ${G6ENV_X:-}"
SERVE_ARGS="--top-p-cands 256 --chunk 8 --temp 0.7 --cap $CAP --graph --wide --mtp $MTP --prefill 4096 --pf-lanes 2 --pfv2"
[ "$PF_OWN" = 3 ] && SERVE_ARGS="$SERVE_ARGS --qo-manifest packs/qo.manifest.json --qo-pack packs/qo.pack"
[ "${NOHASH:-0}" = 1 ] && SERVE_ARGS="$SERVE_ARGS --no-hash"

for ((r = NRANK - 1; r >= 0; r--)); do
  # each process writes its own pid (setsid may fork, so $! is not always the engine's pid). The braces background only
  # the engine: backgrounding the whole && list left a shell holding the ssh session open until the engine exited
  on $r "cd $G6_HOME && rm -rf serve && mkdir -p serve out && { \
    nohup setsid bash -c 'echo \$\$ > out/g6m4.pid; exec env LD_LIBRARY_PATH=$G6_HOME/bin/lib $ENGINE_ENV ./bin/g6m4 --rank $r --ips $IPS --ips2 $IPS2 \
      --proxy-cpu $PROXY_CPU --manifest packs/rank$r.manifest.json --pack packs/rank$r.pack --image --layers all \
      --serve $G6_HOME/serve $SERVE_ARGS' > out/g6m4.r$r.txt 2> out/g6m4.r$r.err < /dev/null & }" &
done
wait
echo "engines started; waiting for four ready files (up to ${READY_TMO}s)"
t0=$(date +%s)
while :; do
  c=0; for ((r = 0; r < NRANK; r++)); do on $r "test -e $G6_HOME/serve/ready" && c=$((c + 1)); done
  [ $c = $NRANK ] && break
  for ((r = 0; r < NRANK; r++)); do
    on $r "p=\$(cat $G6_HOME/out/g6m4.pid 2>/dev/null); [ -z \"\$p\" ] || kill -0 \$p 2>/dev/null" || { echo "rank $r exited; tail of its log:"; on $r "tail -20 $G6_HOME/out/g6m4.r$r.err"; exit 1; }
  done
  [ $(( $(date +%s) - t0 )) -lt "$READY_TMO" ] || { echo "timeout: $c/$NRANK ranks ready"; exit 1; }
  sleep 5
done
echo "all ranks ready after $(( $(date +%s) - t0 ))s"

# shim + front end on the head (rank 0). The head's own serve dir is local; the others are host:path over ssh.
DIRS="$G6_HOME/serve"; for ((r = 1; r < NRANK; r++)); do DIRS="$DIRS ${HOST[$r]}:$G6_HOME/serve"; done
PY=${PY:-$G6_HOME/venv/bin/python}
mkdir -p "$G6_HOME/run" "$G6_HOME/out"; SOCK=$G6_HOME/run/engine.sock; rm -f "$SOCK"
PYTHONPATH="$REPO/frontend" nohup setsid bash -c 'echo $$ > "$0"; exec "$@"' "$G6_HOME/out/shim.pid" \
  "$PY" "$REPO/frontend/g6shim.py" --rank-dirs $DIRS --sock "$SOCK" --max-model-len "$CAP" --stall-timeout 2400 \
  > "$G6_HOME/out/shim.log" 2>&1 < /dev/null &
for i in $(seq 1 30); do [ -S "$SOCK" ] && break; sleep 1; done
[ -S "$SOCK" ] || { echo "shim did not open $SOCK"; tail -20 "$G6_HOME/out/shim.log"; exit 1; }
(cd "$REPO/frontend" && { nohup setsid bash -c 'echo $$ > "$0"; exec "$@"' "$G6_HOME/out/front.pid" \
  "$PY" -m g6front --model-dir "$MODEL_DIR" --engine-sock "$SOCK" --port "$PORT" > "$G6_HOME/out/front.log" 2>&1 < /dev/null & })
for i in $(seq 1 60); do curl -sf "http://127.0.0.1:$PORT/health" > /dev/null && break; sleep 1; done
echo "front end /health: $(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health")  ->  http://<head>:$PORT/v1"
