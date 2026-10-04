# Known issues

Status of the crashes and bugs we are still working out in the multi-session (S2) build (`EXTRA=-DDEC_S2`).
All runs: six DGX Sparks, GPU clock capped at 2200 MHz. "S" = concurrent sessions.

## 1. Multi-session decode fault ("FABRIC" error word) - Open

- **Symptom:** the engine stops (all ranks exit rc=3). Ranks print the error word `fabric 1627389953`
  (`1627389957` in the latest run), `attn 0 cp 0`. Attention and CP report no error.
- **Reproduced 7 times.** It is **not** specific to S=8. Two decoding chats alone (8K prompts, gen 4096, no
  long prefill) trigger it after about 4,300-8,300 main steps (ctx 942-11,527).
- Every crash happens on a **main step with rows 6**.
- With the new control-path logging, ranks 1-5 print the same line:
  `control recv failed: ret=-1 errno=11 (Resource temporarily unavailable) remaining=4 of 4 bytes, frames sent=50 recv=97`.
  That is a receive timeout (EAGAIN) while waiting for the next frame header: 0 of 4 bytes arrived and the link did not break.
  Rank 0 (the sender) logs no control failure.
- In the step-timer trace, the last completed main step took about 9.3 s on the ranks that completed it. Normal steps take milliseconds.
- **Working theory (unconfirmed):** rank 0 stalls first, in the step or the fabric wait, and the other ranks time
  out on the control channel. We are checking what rank 0 is doing during the stall.
- Short S2 screens pass in the same boots: exact-output checks at S=2/3/4 (gen 256/1024).

## 2. S=3/4 descriptor race - Fixed

- A batched KV-append descriptor was read while it was still zero, before its upload finished (the memset and the upload ran on different streams).
- After the fix: no fault, and exact outputs at S=2/3/4 in 4 of 4 passes, all six ranks rc=0.

## 3. Test harness: exact-check result looked up in the wrong place - Open (harness only)

- Our test harness reported CRITERIA FAIL (`s2exact:missing`) on runs where every exact-check pass succeeded
  (all sessions identical, S=2/3/4). The harness read the result from the wrong place. This is not an engine bug.

## 4. Prefix-reuse miss near 786K tokens - Open

- A request that should reuse a ~786K-token resident prefix misses and does a full re-prefill.
- Next step: log a reason code for each miss and return the longest valid resident prefix, not all-or-nothing.

## 5. Aggregate decode stays flat past S=4 - Open

- Aggregate decode, tok/s: S1 39.7, S2 53.1, S4 52.1, S6 52.8 (2200 MHz cap).
- We do not know yet if the cause is the scheduler or weight bandwidth.

## 6. First step after prefill - Not a bug

- We measured it. The first decode step after the prompt runs at 0.989x the steady step time (median over 35 requests, min 0.66, max 1.13).
- The first step stays on the CUDA-graph path. No non-graph path appears in the trace.
