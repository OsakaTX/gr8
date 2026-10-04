// SELX: our own exact per-row top-2048 select for prefill tiles with > 1 row, final sort fused in. Written from the idea
// (one CTA per row, radix histogram select); no code from vLLM. Self-contained (CUDA + cub only, no ga:: / G6Attn), so TP6
// (select_index for count > 1) and TP4 (instead of J5 for > 1-row tiles) can include it as is. n = 1 stays on SEL32.
// Output = SEL32's exactly: key = ordered_score << 32 | (0xffffffff - id) (score desc, then id asc; -0 == +0; inf / nan
// never selected), indices[r * TOP + j] = id of the j-th largest key, -1 fill when the row has < TOP finite scores.
// Per row (one CTA of BT threads; keys are unique, so every step below is exact):
//   1 bound: radix histogram (11-bit digits) of a sample = every SAMP-th block of SB scores, 2 levels -> a key b with
//     >= TOP sample keys >= b, so >= TOP row keys >= b: every top-TOP key is >= b. Rows with len < 2 * TOP * SAMP (the sample
//     would hold < 2 TOP keys) and len <= capg skip it (b = 1: every finite key to cand).
//   2 filter: one pass over the row, keys >= b -> cand (global scratch, capg keys per row). ~TOP * SAMP keys expected.
//     If more than capg pass (b too low for this row) the row falls back to radix levels straight on the scores
//     (collecting into cand as soon as a level's bin fits), counted in stats[0].
//   3 select: radix levels (11-bit digits, 64-bit key) over cand: keys above the chosen bin go to smem out[], the bin is
//     refined, until a bin holds exactly the remaining need.
//   4 sort: bitonic sort of the TOP keys in smem, descending -> indices.
// Scratch: cand_bytes(rows, capg). stats (nullable): [0] += fallback rows, [1] += internal overflow (a bug; must stay 0).
#pragma once
#include <cub/cub.cuh>
#include <cstdint>
namespace selx {
constexpr int TOP = 2048, W = 11, NB = 1 << W, SB = 1024, SAMP = 8, CAPG = 65536;
static_assert((TOP & (TOP - 1)) == 0 && SB % 4 == 0, "selx");
__device__ __forceinline__ uint32_t ord(float s)
{
    uint32_t b = __float_as_uint(s);
    if ((b & 0x7fffffffu) >= 0x7f800000u) return 0;   // inf / nan: key 0 = never selected
    if ((b & 0x7fffffffu) == 0) b = 0;                 // -0 == +0
    return (b & 0x80000000u) ? ~b : (b ^ 0x80000000u);
}
__device__ __forceinline__ uint64_t key(float s, int id) { const uint32_t o = ord(s); return o ? ((uint64_t)o << 32) | (0xffffffffu - (uint32_t)id) : 0; }
// pass state: range = keys whose top lp bits == pp (lp 0: all); current prefix p of l bits (l >= lp); fin: emit and stop
struct St { uint64_t pp, p; int lp, l, need, fin, nout, ncand, B, above, hb, total; };
template <int BT>
struct Smem { int hist[NB]; uint64_t out[TOP]; St s; typename cub::BlockScan<int, BT>::TempStorage scan; };
// U float4 / u64x4 loads per thread in flight, then f on every key; every thread runs the same rounds (warp collectives in f)
constexpr int U = 4;
template <int BT, class Load, class F>
__device__ __forceinline__ void each4(int n4, Load load, F f)
{
    for (int b = 0; b < n4; b += BT * U)
    {
        uint64_t k[U][4];
        #pragma unroll
        for (int u = 0; u < U; ++u) load(b + u * BT + (int)threadIdx.x, k[u]);
        #pragma unroll
        for (int u = 0; u < U; ++u)
        #pragma unroll
            for (int e = 0; e < 4; ++e) f(k[u][e]);
    }
}
// 4 consecutive scores from x[i0 .. i0+3] (bounded by len; ids base + i)
__device__ __forceinline__ void load_row(const float* x, int len, bool v4, int i0, int idbase, uint64_t* k)
{
    if (v4 && i0 + 3 < len)
    {
        const float4 q = __ldg((const float4*)(x + i0));
        k[0] = key(q.x, idbase + i0); k[1] = key(q.y, idbase + i0 + 1); k[2] = key(q.z, idbase + i0 + 2); k[3] = key(q.w, idbase + i0 + 3);
    }
    else
        for (int e = 0; e < 4; ++e) k[e] = i0 + e < len ? key(__ldg(x + i0 + e), idbase + i0 + e) : 0;
}
// warp-aggregated append of k (where take) to buf[*cnt...]; returns nothing, drops slots >= cap (caller checks *cnt)
__device__ __forceinline__ void append(bool take, uint64_t k, int* cnt, uint64_t* buf, int cap)
{
    const unsigned m = __ballot_sync(0xffffffffu, take);
    if (!m) return;
    const int lane = threadIdx.x & 31, lead = __ffs(m) - 1;
    int base = 0;
    if (lane == lead) base = atomicAdd(cnt, __popc(m));
    base = __shfl_sync(0xffffffffu, base, lead);
    if (take) { const int slot = base + __popc(m & ((1u << lane) - 1)); if (slot < cap) buf[slot] = k; }
}
template <int BT>
struct Row {
    Smem<BT>& m; const float* x; int len; bool v4; uint64_t* cand; int capg;
    // one radix pass over src: emit (emit_ok) keys above the prefix into out, histogram the next digit of keys on the
    // prefix, collect them into cand (collect). src: 0 row, 1 sample, 2 cand.
    __device__ void pass(int src, bool emit_ok, bool collect)
    {
        const uint64_t pp = m.s.pp, p = m.s.p; const int lp = m.s.lp, l = m.s.l, fin = m.s.fin;
        const int w = min(W, 64 - l), sh = 64 - l - w;
        auto f = [&](uint64_t k) {
            const bool in = k != 0 && (lp == 0 || (k >> (64 - lp)) == pp);
            bool em = false, hs = false;
            if (in)
            {
                const uint64_t u = l ? k >> (64 - l) : 0;
                if (fin) em = l == lp || u >= p;
                else { em = l && u > p; hs = !l || u == p; }
                em &= emit_ok;
            }
            append(em, k, &m.s.nout, m.out, TOP);
            if (__any_sync(0xffffffffu, hs))
            {
                const unsigned d = hs ? unsigned(k >> sh) & ((1u << w) - 1) : NB;
                const unsigned peers = __match_any_sync(0xffffffffu, d);
                if (hs && (threadIdx.x & 31) == __ffs(peers) - 1) atomicAdd(m.hist + d, __popc(peers));
                if (collect) append(hs, k, &m.s.ncand, cand, capg);
            }
        };
        if (src == 0) each4<BT>((len + 3) / 4, [&](int i4, uint64_t* k) { load_row(x, len, v4, 4 * i4, 0, k); }, f);
        else if (src == 1)
        {   // sample: blocks 0, SAMP, 2 SAMP, ... of SB scores
            const int nblk = (len + SB - 1) / SB, ns = (nblk + SAMP - 1) / SAMP;
            each4<BT>(ns * (SB / 4), [&](int i4, uint64_t* k) {
                const int i0 = (i4 / (SB / 4)) * SB * SAMP + (i4 % (SB / 4)) * 4;
                load_row(x, len, v4, i0, 0, k); }, f);
        }
        else
        {
            const int n = m.s.ncand;
            each4<BT>((n + 3) / 4, [&](int i4, uint64_t* k) { for (int e = 0; e < 4; ++e) k[e] = 4 * i4 + e < n ? cand[4 * i4 + e] : 0; }, f);
        }
    }
    // after a histogram pass: bin B with above < need <= above + hist[B] (bins scanned from the top), total
    __device__ void choose()
    {
        constexpr int G = NB / BT;
        const int t = threadIdx.x;
        int c = 0;
        #pragma unroll
        for (int j = 0; j < G; ++j) c += m.hist[NB - 1 - t * G - j];
        int before, total;
        cub::BlockScan<int, BT>(m.scan).ExclusiveSum(c, before, total);
        const int need = m.s.need;
        if (before < need && before + c >= need)
        {
            int acc = before;
            for (int j = 0; j < G; ++j)
            {
                const int b = NB - 1 - t * G - j, h = m.hist[b];
                if (acc + h >= need) { m.s.B = b; m.s.above = acc; m.s.hb = h; break; }
                acc += h;
            }
        }
        if (t == 0) m.s.total = total;
        __syncthreads();
    }
    __device__ void clear(bool ncand)
    {
        for (int j = threadIdx.x; j < NB; j += BT) m.hist[j] = 0;
        if (threadIdx.x == 0 && ncand) m.s.ncand = 0;
        __syncthreads();
    }
    // thread 0, after choose(): descend into bin B (or take the whole range if it holds < need keys)
    __device__ void descend()
    {
        St& s = m.s;
        s.pp = s.p; s.lp = s.l;
        if (s.total < s.need) { s.fin = 1; return; }
        const int w = min(W, 64 - s.l);
        s.p = s.l ? (s.p << w) | (uint64_t)s.B : (uint64_t)s.B; s.l += w; s.need -= s.above; s.fin = s.hb == s.need;
    }
    // radix levels until the fin pass; src 0 (row, collecting into cand once a bin fits) or 2 (cand)
    __device__ void levels(int src)
    {
        for (;;)
        {
            const bool collect = src == 0 && m.s.l > 0;
            clear(collect);
            pass(src, true, collect);
            __syncthreads();
            if (m.s.fin) return;
            choose();
            const bool to_cand = collect && m.s.total <= capg;
            __syncthreads();
            if (threadIdx.x == 0) descend();
            __syncthreads();
            if (to_cand) src = 2;
        }
    }
};
template <int BT>
__global__ void __launch_bounds__(BT) select_rows(const float* __restrict__ scores, int64_t stride, int len, const int* ends,
                                                   int* indices, uint64_t* cand, int capg, int* stats)
{
    __shared__ Smem<BT> m;
    const int r = blockIdx.x, t = threadIdx.x;
    const int n = ends ? max(0, min(len, ends[r])) : len;
    const float* x = scores + (int64_t)r * stride;
    Row<BT> rw{m, x, n, ((uintptr_t)x & 15) == 0, cand + (int64_t)r * capg, capg};
    if (t == 0) { m.s = St{}; m.s.need = TOP; }
    __syncthreads();
    // 1 bound from the sample (2 levels)
    uint64_t bound = 1;
    if (n > capg || n >= 2 * TOP * SAMP)
    {
        rw.clear(false); rw.pass(1, false, false); __syncthreads(); rw.choose();
        if (m.s.total >= TOP)
        {
            const uint64_t B0 = m.s.B; const int need1 = TOP - m.s.above;
            bound = B0 << (64 - W);
            if (m.s.hb != need1)
            {
                __syncthreads();
                if (t == 0) { m.s.p = B0; m.s.l = W; m.s.need = need1; }   // range stays all (lp 0): nothing is emitted
                rw.clear(false); rw.pass(1, false, false); __syncthreads(); rw.choose();
                bound = ((B0 << W) | (uint64_t)m.s.B) << (64 - 2 * W);
            }
        }
        __syncthreads();
        if (t == 0) { m.s = St{}; m.s.need = TOP; }
        __syncthreads();
    }
    // 2 filter
    each4<BT>((n + 3) / 4, [&](int i4, uint64_t* k) { load_row(x, n, rw.v4, 4 * i4, 0, k); },
              [&](uint64_t k) { append(k != 0 && k >= bound, k, &m.s.ncand, rw.cand, capg); });
    __syncthreads();
    // 3 select
    if (m.s.ncand <= capg) rw.levels(2);
    else
    {
        if (t == 0 && stats) atomicAdd(stats, 1);
        __syncthreads();
        if (t == 0) { m.s.ncand = 0; }
        rw.levels(0);
    }
    __syncthreads();
    if (t == 0 && m.s.nout > TOP && stats) atomicAdd(stats + 1, 1);
    // 4 sort (bitonic, descending; keys unique, -1 fill keys 0 sort last)
    for (int j = min(m.s.nout, TOP) + t; j < TOP; j += BT) m.out[j] = 0;
    __syncthreads();
    for (int k = 2; k <= TOP; k <<= 1)
        for (int j = k >> 1; j > 0; j >>= 1)
        {
            for (int i = t; i < TOP / 2; i += BT)
            {
                const int a = 2 * i - (i & (j - 1)), b = a + j;    // a has bit j clear
                const uint64_t x0 = m.out[a], x1 = m.out[b];
                const bool desc = (a & k) == 0;
                if (desc ? x0 < x1 : x0 > x1) { m.out[a] = x1; m.out[b] = x0; }
            }
            __syncthreads();
        }
    int* o = indices + (int64_t)r * TOP;
    for (int j = t; j < TOP; j += BT) { const uint64_t k = m.out[j]; o[j] = k ? int(0xffffffffu - uint32_t(k)) : -1; }
}
// rows x [0, len) scores at scores + r * stride (row r's length min(len, ends[r]) if ends) -> indices + r * TOP
inline size_t cand_bytes(int rows, int capg = CAPG) { return (size_t)rows * capg * sizeof(uint64_t); }
inline cudaError_t select(const float* scores, int64_t stride, int len, const int* ends, int rows, int* indices,
                          uint64_t* cand, int capg, int* stats, cudaStream_t st)
{
    if (rows <= 0) return cudaSuccess;
    select_rows<512><<<rows, 512, 0, st>>>(scores, stride, len, ends, indices, cand, capg, stats);
    return cudaGetLastError();
}
}  // namespace selx
