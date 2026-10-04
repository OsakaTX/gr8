# Changelog

## 2026-10-04: S2 multi-session snapshot (since b85a7a6, the K1H2 snapshot)

The default build (`scripts/build.sh`) is still K1H2. The new multi-session code is compiled in only with
`EXTRA=-DDEC_S2 scripts/build.sh`, and it is turned on at run time with `g6m4 --s2`. See KNOWN-ISSUES.md: S2 still has an open crash on long multi-chat decode.

- **S2 batched multi-session decode (opt-in, `-DDEC_S2`).** New files: `engine/runtime/s2.cuh`, `s2_batch.h`,
  `draft_shape.h`, `engine/runtime/s2/*` (scheduler, sessions, run pool, control channel, native dispatch).
  - Up to 8 sessions decode in one step. Sessions are grouped into MTP row classes (S = 3..8 at mtp 2 use ceil(S/2) groups per round).
  - Each session's KV is one contiguous run in a shared pool. A slot can move to a new base, and the 2->1 / S=1 transitions are handled.
  - Prefill can start at a base page. Below R the loop runs on the CP binding (`g6m4 --s2`).
  - The rank-0 shim talks to the native adapter over persistent control sockets.
- S2 slot views are sized for MX decode rows, not for the prefill chunk. This saves about 8.9 GB per rank at the 614K boot.
- A batched S2 chain draft step now has one row per session.
- If a step fails, S2 prints the fabric / attention / CP error words before the fail-closed check.
- New per-step timer `G6_STEP_TMR=1`, off by default. It prints a per-session request/event label for each step (TAIL / DECODE1 / ROUND / MIXED).
- Control-path disconnect logging: on a recv/send failure the engine prints ret, errno, remaining bytes and frame counts before it throws the epoch-fatal error.
- SELX (`engine/attn/selx.cuh`): our own exact top-2048 select for prefill index tiles. It runs on tiles with n >= 32 rows and context >= `G6_PF_SELX_CTX`. Its output is identical to SEL32's. Unset = SEL32 (unchanged behaviour).
- CP decode `-DCP_F1A`: the phase-2 per-partial wait + get is fused into one launch (off by default).
- The front end metrics have a hook for S2.
