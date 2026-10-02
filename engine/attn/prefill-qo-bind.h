#pragma once
#include "bind.h"
// Row-owner (query-owned) prefill attention for the runtime's G6_PF_OWN=3 mode (engine/runtime/QO-INTEGRATION.md).
// One state per layer: private FP16 copies of the FULL 64-head UK / UV, plus a workspace shared by every state that
// names the same storage owner (FP16 key table, absorbed Q, FP32 context, FP16 context, one cuBLAS handle).
// The binding passed to run supplies the layer's latent cache (replicated on every rank, already appended for the
// rows) and the error word; its head split is not used.
struct G6AttnQo;
struct G6AttnQoStep {
    int rows, context;          // owner rows; context = last owner position + 1 (keys 0..context-1 are cached)
    const void* q_b;            // FP16 [rows][ld_qb]: 64 heads x 256 (nope 192 | rope 64) at column 0
    int ld_qb;                  // elements
    const int* positions;       // device [rows]
    const int* ids;             // device [rows][2048] selected positions, -1 padding (indexer order)
    void* out;                  // FP16 [rows][64*256]: per head UV x context, the o_proj input
};
// uk: [64,192,512] BF16, uv: [64,256,512] BF16 (logical head order). max_rows: owner rows per call.
// Kernel read once at create: G6_PF_QO=b (default: FP16 key-view prepass + core_b) | v1 (prototype, records direct) |
// r1 (the R1 variant, compiled in when engine/attn/prefill-qo-r1.cuh exists). No persisting-L2 window is set here.
extern "C" int g6_attn_qo_create(G6AttnQo** out, const G6AttnBinding* b, const G6AttnTensor* uk, const G6AttnTensor* uv,
                                 int max_rows, void* stream, G6AttnQo* storage_owner);
extern "C" int g6_attn_qo_run(G6AttnQo*, G6AttnBinding*, const G6AttnQoStep*, void* stream);
extern "C" size_t g6_attn_qo_bytes(const G6AttnQo*);
// measurement (gate b): while on, run synchronizes the stream between its stages and adds their wall ms to
// ms[0] absorb (+ pack_q), ms[1] prepass (B only), ms[2] core, ms[3] cast + UV. Off by default; no effect on results.
extern "C" void g6_attn_qo_prof(int on, double* ms);
// load-time warm-up: the absorb and UV strided-batched GEMMs once at `rows` (<= max_rows) on the state's workspace
// (contents are scratch), then a stream sync. cuBLAS's first call per (handle, shape) costs ~0.1-0.4 s on the host.
extern "C" int g6_attn_qo_warm(G6AttnQo*, int rows, void* stream);     // private weights + workspace (root only)
extern "C" int g6_attn_qo_release(G6AttnQo*);
