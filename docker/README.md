# The gr8 container

One image for GB10 boxes (linux/arm64, CUDA 13, sm_121a). The image holds the TP=4 engine (`bin/`, with the multi-session
path), the TP=3 engine (`bin3/`), the TP=6 engine (`bin6/`), the Rust front end (`bin/g6front-rs`), the shims and the
whole repository, on the CUDA 13 runtime image (which brings cuBLAS) with libibverbs and the mlx5 provider. Each box
runs one container: one rank (`g6m4`) per box, and the head (rank 0) also runs the shim and the front end. The model
cache stays on the host and is mounted in.

CI builds this image but does not push it anywhere. Build the image yourself on one Spark and copy it to the others.
On a GB10 the build takes about 10 minutes (all three engines, the front end and the loader), plus the first download
of the CUDA base images (several GB, on the building box only):

    docker build -t gr8 .                                   # from the repo root, or from an unzipped tester zip
    for h in node2 node3 node4; do docker save gr8 | ssh $h docker load; done

## docker/gr8: the launcher in the container

`docker/gr8` runs `./gr8 glm53` inside the image, one container per box. It takes the same options, the same config file
and the same environment as a bare-metal box, and the launcher does the same checks and the same handshake. Run the same
command on every box, from any directory that holds your `cluster.env` (or pass every option on the command line):

    docker/gr8 up --tp 4 --peers 10.0.0.1,10.0.0.2,10.0.0.3,10.0.0.4 \
        --model https://huggingface.co/kindlingai/glm-5.3-exlr8-k2-k3-k4 --model-cache /opt/g6/cache \
        --drafter https://huggingface.co/incoai/GLM-5.3-DFlash2

Then, from any box: `docker/gr8 status`, `docker/gr8 smoke`, `docker/gr8 logs supervisor`, and `docker/gr8 down` to stop
every box. `up` follows the boot until the cluster serves, as `./gr8 glm53 up` does; Ctrl-C leaves the boot running.
Stop the cluster with `docker/gr8 down`, not `docker stop` or `docker kill` on one box: those stop that box's supervisor
with its engine, so nothing tells the others, which keep their engines (requests fail with HTTP 500) until a
`docker/gr8 down` from any box. An engine that dies on its own (the OOM killer, a crash) stops every box, as on bare
metal. `restart` runs `down`, waits for this box's container to leave, then `up`.

The same goes for a watchdog of your own (a memory floor, a health check): have it end the engine, not the container,
for example `sudo kill -KILL $(pgrep -x g6m4)`. The box's supervisor then reports `engine exited (-9)` to the head and
the head stops every box; killing the container would leave the other boxes up and unable to serve.
`check`, `show-config` and `prepare` work as on bare metal. `build`, `stage` and `packs` do not apply: the image is the
build. `docker/gr8 bench` appends its results to `G6_HOME/out/pbench-results.jsonl` (bare metal: `bench/results.jsonl`).

Before the first `up`:

- at TP=6 only, passwordless ssh from the head to every other box, by the address in `--peers`: the TP=6 head's shim
  writes each request into the other ranks' `s2/` directories over ssh (TP=4 and TP=3 need none: the head engine passes
  requests over its control links). Check it from the head, with no prompt:
  `for p in 10.0.0.2 10.0.0.3 10.0.0.4 10.0.0.5 10.0.0.6; do ssh -o BatchMode=yes $p true && echo "$p ok"; done`
- the NVIDIA container toolkit (on DGX OS by default) and access to docker for whoever runs `docker/gr8`.

### Settings

`docker/gr8` adds no options of its own to the launcher's. Four environment variables say how it runs the container:

| variable | default | |
|---|---|---|
| `GR8_IMAGE` | `gr8` | the image |
| `GR8_USER` | you | the account the container runs as. Starting a container takes docker access, which is root-equivalent; the engine needs none of it, so an account with no sudo and no docker can run it while you start it |
| `GR8_SSH` | `~GR8_USER/.ssh` | TP=6: the ssh directory the head's shim uses, mounted read-only as that account's `~/.ssh`. At TP=4 and TP=3 no ssh directory is mounted |
| `G6_HOME` | `~GR8_USER/gr8-docker` | this box's working directory (`out/`, `s2/`, `run/`); `bin/` there becomes a link to the image's engine for the TP, so it must not be a bare-metal build's `bin/` |

The shim runs on the image's `python3` (`PY=/usr/bin/python3` unless you set `PY`).

With `GR8_USER` set to another account, that account must be able to read the directory you run `docker/gr8` from (it
is the container's working directory), and its `G6_HOME` and model cache must already exist, writable by it:
`docker/gr8` creates them only for itself.

### Networks

`docker/gr8` adds no network settings. The container uses the host network and sees the host's `/sys/class/infiniband`,
so the launcher finds this box's place in `--peers` and its two fabric rails exactly as on bare metal, and `--ips`,
`--ips2`, `--node-ip`, `--port` and `--control-port` mean what they mean there. Whatever your boxes need on bare metal
(a switch, direct cables, one subnet per rail, addresses the launcher cannot pick on its own), pass the same options here.

### What the container gets

| | why |
|---|---|
| `--network host`, `--device /dev/infiniband`, `--cap-add IPC_LOCK`, `--ulimit memlock=-1` | the RoCE fabric |
| `--gpus all` | the GPU |
| `--pid host` | the launcher refuses to start when another `g6m4` runs on the box; it can see one started outside the container only with the host's process list |
| `--user` (GR8_USER), `--cap-drop ALL` (but `IPC_LOCK`), `--security-opt no-new-privileges`, `--read-only` with a private `/tmp` | the engine runs with no more rights than it needs |
| `--oom-score-adj 1000` (`up`) | under memory pressure the kernel ends the engine first, before sshd or the rest of the box |
| `G6_HOME` and the model cache read-write; the release, the drafter and the config file read-only, each at the same path as on the host | the launcher hands these paths to the engine and the shim as they are (at TP=6 the head's shim also writes each worker's `s2/` over ssh, by the path that worker reported) |
| `/etc/passwd` and `/etc/group` read-only; at TP=6, `GR8_SSH` read-only | GR8_USER's name and home; the TP=6 shim's ssh |

A missing path makes `docker run` stop with "bind source path does not exist" (`docker/gr8` mounts with `--mount`, which
refuses a missing path where `-v` would create one owned by root).

## By hand

The steps below start the same containers without `docker/gr8`.

### Container requirements

| need | flag |
|---|---|
| the GPU | `--gpus all` (the NVIDIA container toolkit, on DGX OS by default) |
| the RoCE fabric | `--network host --device /dev/infiniband --cap-add IPC_LOCK --ulimit memlock=-1` |
| the working directory (`s2/`, `out/`) and the model cache | `-v $G6_HOME:$G6_HOME -v $MODEL_CACHE:$MODEL_CACHE -w $G6_HOME` (the same paths on every box) |
| files the host user can read and write | `--user $(id -u):$(id -g)` |

### One rank

The engine's environment and arguments are the `engine:` line that `./gr8 glm53 up` writes to `out/supervisor.log` on
each box (fill the cache first with `./gr8 glm53 prepare`). Its `--s2-epoch` must be the same on every rank: pick one
(for example `$(date +%s)`). On box `r`, after `rm -rf $G6_HOME/s2 && mkdir -p $G6_HOME/s2 $G6_HOME/out`:

    docker run -d --name gr8-rank --gpus all --network host --device /dev/infiniband --cap-add IPC_LOCK \
      --ulimit memlock=-1 --user $(id -u):$(id -g) -v $G6_HOME:$G6_HOME -v $MODEL_CACHE:$MODEL_CACHE -w $G6_HOME gr8 \
      env $ENGINE_ENV g6m4 --rank $r --ips $IPS --ips2 $IPS2 --proxy-cpu $PROXY_CPU $ENGINE_ARGS

`docker logs gr8-rank` shows the engine's output. Rank 0 writes `$G6_HOME/s2/ready.s2` when all ranks have loaded.
TP=3 runs `/opt/gr8/bin3/g6m4`. TP=6 runs `/opt/gr8/bin6/g6m4`, and its head runs `frontend/g6shim.py` in place of
`g6s2shim.py`.

### The head: shim and front end

The shim writes requests into the head's own `s2/` directory, and the head engine passes them to the other ranks:

    docker run -d --name gr8-head --network host --user $(id -u):$(id -g) \
      -v $G6_HOME:$G6_HOME -v $MODEL_CACHE:$MODEL_CACHE:ro gr8 bash -c "
        mkdir -p $G6_HOME/run
        g6front-rs --model-dir $TOKENIZER_DIR --engine-sock $G6_HOME/run/engine.sock --port 8100 &
        exec python3 frontend/g6s2shim.py --sock $G6_HOME/run/engine.sock --dir $G6_HOME/s2"

`TOKENIZER_DIR` is the release's `tokenizer/` directory in the cache (the `--model-dir` of the front end's line in
`out/supervisor.log`). Start it after `ready.s2` exists. Then `curl http://<head>:8100/health`.

### Stopping

`docker stop gr8-head gr8-rank` on each box. If one rank stops, the whole engine stops. Restart every container.

## Not covered

A prebuilt image: CI builds it and pushes it nowhere, so every cluster builds its own.
