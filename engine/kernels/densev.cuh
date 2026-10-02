// Dense small-M GEMV for 8-bit and 16-bit weights, y[M][O] = x[M][K] @ W^T, M <= 8. Our own code.
// Same shape as K2 (rowtile.cuh): weights are the mma A operand, a tile = 16 output rows x 16 inputs,
// activations are the B operand (global fragment table [K/16][32][2]), fp32 accumulate.
//   Mxfp8  E4M3FN codes, one E8M0 scale per (row, 32 inputs). Group = 2 tiles of 256 B + 16 scale bytes
//          (8 pairs {s[i], s[i+8]}) = 528 B. Lane l (r = l/4, c = l%4) reads 2 u32 per tile:
//          {row r k 2c,2c+1 | row r+8 k 2c,2c+1}, {row r k 2c+8,2c+9 | row r+8 k 2c+8,2c+9}.
//          Activations fp16. The scale multiplies the group's fp32 partial sum (exact, power of two).
//   Int4v  the int4 g32 K2 records of rowtile.cuh (group = 2 tiles of 128 B + 16 fp16 scales = 288 B), fp16 activations.
//   Bf16w  BF16 weights, no scale. Group = 1 tile = 512 B; lane l reads 4 u32 = a0..a3 of the mma:
//          (row r, k 2c..), (row r+8, k 2c..), (row r, k 2c+8..), (row r+8, k 2c+8..).
//          Activations must be BF16 fragments (x rounded to bf16: the bf16 mma takes both operands as bf16).
// Output dtype per launch: fp16, bf16 (fabric partials) or fp32 (router / head logits).
#pragma once
#include "rowtile.cuh"
#ifdef DEC_SLOT
#include "decslot.cuh"
#endif

namespace gv
{
enum { O_F16 = 0, O_BF16 = 1, O_F32 = 2 };

__device__ __forceinline__ void ld_nc_v2(const void* p, u32* v)
{
    asm volatile("ld.global.nc.L1::no_allocate.v2.u32 {%0,%1}, [%2];" : "=r"(v[0]), "=r"(v[1]) : "l"(p));
}
__device__ __forceinline__ void ld_nc_v4(const void* p, u32* v)
{
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3]) : "l"(p));
}

__device__ __forceinline__ void mma_bf16(float (&c)[4], u32 a0, u32 a1, u32 a2, u32 a3, u32 b0, u32 b1)
{
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

using ::f2bf;

__device__ __forceinline__ void put(void* y, size_t i, float v, int odt)
{
    if (odt == O_F32) static_cast<float*>(y)[i] = v;
    else if (odt == O_BF16) static_cast<uint16_t*>(y)[i] = f2bf(v);
    else static_cast<__half*>(y)[i] = __float2half_rn(v);
}
}   // namespace gv

struct Mxfp8
{
    static constexpr int G = 2, TW = 2, GR = 528, PFB = 2;
    static constexpr bool SCALED = true;
    static __device__ __forceinline__ void ld_tile(const uint8_t* tile, int lane, u32* w) { gv::ld_nc_v2(tile + lane * 8, w); }
    static __device__ __forceinline__ void mma(float (&g)[4], const u32* w, u32 b0, u32 b1)
    {
        mma16816(g, e4m3x2(w[0]), e4m3x2(w[0] >> 16), e4m3x2(w[1]), e4m3x2(w[1] >> 16), b0, b1);
    }
    static __device__ __forceinline__ u32 ld_scale(const uint8_t* grp, int lane) { return ld_nc16(grp + 512 + (lane >> 2) * 2); }
    static __device__ __forceinline__ float e8m0(u32 code) { return __uint_as_float(code ? code << 23 : 0x00400000u); }   // 2^(code-127)
    static __device__ __forceinline__ float2 scale(u32 raw) { return make_float2(e8m0(raw & 0xffu), e8m0((raw >> 8) & 0xffu)); }
};

struct Bf16w
{
    static constexpr int G = 1, TW = 4, GR = 512, PFB = 4;
    static constexpr bool SCALED = false;
    static __device__ __forceinline__ void ld_tile(const uint8_t* tile, int lane, u32* w) { gv::ld_nc_v4(tile + lane * 16, w); }
    static __device__ __forceinline__ void mma(float (&g)[4], const u32* w, u32 b0, u32 b1) { gv::mma_bf16(g, w[0], w[1], w[2], w[3], b0, b1); }
    static __device__ __forceinline__ u32 ld_scale(const uint8_t*, int) { return 0; }
    static __device__ __forceinline__ float2 scale(u32) { return make_float2(1.f, 1.f); }
};

// int4 g32 W4A16 (the K2 / rt_record layout of rowtile.cuh) behind the same interface, so o_proj and the other
// D8 producers get the output dtype switch: gv_gemv<Int4v> with odt = O_BF16 writes a fabric partial directly.
struct Int4v
{
    static constexpr int G = 2, TW = 1, GR = 288, PFB = 2;
    static constexpr bool SCALED = true;
    static __device__ __forceinline__ void ld_tile(const uint8_t* tile, int lane, u32* w) { w[0] = ld_nc(tile + lane * 4); }
    static __device__ __forceinline__ void mma(float (&g)[4], const u32* w, u32 b0, u32 b1)
    {
        mma16816(g, Int4::dec(w[0] & 0xffu), Int4::dec((w[0] >> 8) & 0xffu), Int4::dec((w[0] >> 16) & 0xffu), Int4::dec(w[0] >> 24), b0, b1);
    }
    static __device__ __forceinline__ u32 ld_scale(const uint8_t* grp, int lane) { return Int4::ld_scale(grp, lane); }
    static __device__ __forceinline__ float2 scale(u32 raw) { return Int4::scale(raw); }
};

template <class C>
struct GvGroup
{
    u32 c[C::G * C::TW];
    u32 s;
    __device__ __forceinline__ void load(const uint8_t* grp, int lane)
    {
        #pragma unroll
        for (int i = 0; i < C::G; ++i) C::ld_tile(grp + i * C::TW * 128, lane, c + i * C::TW);
        s = C::ld_scale(grp, lane);
    }
    // acc += scale * (weights @ B fragments); bf = fragment table already offset to this lane, ks = first k16 step
    __device__ __forceinline__ void run(const u32* bf, int ks, float (&acc)[4]) const
    {
        if (C::SCALED)
        {
            float g[4] = {0.f, 0.f, 0.f, 0.f};
            #pragma unroll
            for (int i = 0; i < C::G; ++i) C::mma(g, c + i * C::TW, bf[(ks + i) * 64], bf[(ks + i) * 64 + 1]);
            const float2 sc = C::scale(s);
            acc[0] = fmaf(sc.x, g[0], acc[0]); acc[1] = fmaf(sc.x, g[1], acc[1]);
            acc[2] = fmaf(sc.y, g[2], acc[2]); acc[3] = fmaf(sc.y, g[3], acc[3]);
        }
        else
        {
            #pragma unroll
            for (int i = 0; i < C::G; ++i) C::mma(acc, c + i * C::TW, bf[(ks + i) * 64], bf[(ks + i) * 64 + 1]);
        }
    }
};

// one warp: groups g0..g1-1 of one output tile, C::PFB groups read ahead
template <class C>
__device__ __forceinline__ void gv_tile_part(const uint8_t* tile, const u32* bf, int g0, int g1, float (&acc)[4])
{
    constexpr int PFB = C::PFB;
    const int lane = threadIdx.x & 31;
    const int Q = g1 - g0;
    GvGroup<C> nx[PFB];
    #pragma unroll
    for (int p = 0; p < PFB; ++p)
        if (p < Q) nx[p].load(tile + static_cast<size_t>(g0 + p) * C::GR, lane);
    for (int q = 0; q < Q; q += PFB)
    {
        GvGroup<C> cur[PFB];
        #pragma unroll
        for (int p = 0; p < PFB; ++p) cur[p] = nx[p];
        #pragma unroll
        for (int p = 0; p < PFB; ++p)
            if (q + PFB + p < Q) nx[p].load(tile + static_cast<size_t>(g0 + q + PFB + p) * C::GR, lane);
        #pragma unroll
        for (int p = 0; p < PFB; ++p)
            if (q + p < Q) cur[p].run(bf, (g0 + q + p) * C::G, acc);
    }
}

#if DEC_I4PAIR
// DEC_I4PAIR (results/peak/decode/PROPOSAL-gemv.md): the decode Int4 records (q_a, o_proj) are repacked once at load
// (k_i4pair) so that inside every warp's k-range [g0, g1) groups 2j, 2j+1 form one 576-B pair record:
//   [512 B: lane l's 16 B = {g2j tile0 word l, g2j tile1 word l, g2j+1 tile0 word l, g2j+1 tile1 word l}]
//   [64 B:  r = 0..7: {g2j scale word r, g2j+1 scale word r}]
// An odd last group of a range keeps the 288-B layout. One v4 weight load + one v2 scale load per lane per pair instead
// of 2 x (2 x 4 B + 4 B); the mma / scale / accumulate sequence is the stock one (same operands, same order) -> same bits.
__device__ __forceinline__ void gv_i4pair_part(const uint8_t* tile, const u32* bf, int g0, int g1, float (&acc)[4])
{
    const int lane = threadIdx.x & 31;
    const int NP = (g1 - g0) / 2;
    const uint8_t* base = tile + static_cast<size_t>(g0) * Int4v::GR;
    u32 nw[2][4], ns[2][2];
    #pragma unroll
    for (int p = 0; p < 2; ++p)
        if (p < NP) { gv::ld_nc_v4(base + p * 576 + lane * 16, nw[p]); gv::ld_nc_v2(base + p * 576 + 512 + (lane >> 2) * 8, ns[p]); }
    for (int q = 0; q < NP; q += 2)
    {
        u32 cw[2][4], cs[2][2];
        #pragma unroll
        for (int p = 0; p < 2; ++p)
        {
            #pragma unroll
            for (int j = 0; j < 4; ++j) cw[p][j] = nw[p][j];
            cs[p][0] = ns[p][0]; cs[p][1] = ns[p][1];
        }
        #pragma unroll
        for (int p = 0; p < 2; ++p)
            if (q + 2 + p < NP)
            {
                const uint8_t* b = base + static_cast<size_t>(q + 2 + p) * 576;
                gv::ld_nc_v4(b + lane * 16, nw[p]); gv::ld_nc_v2(b + 512 + (lane >> 2) * 8, ns[p]);
            }
        #pragma unroll
        for (int p = 0; p < 2; ++p)
            if (q + p < NP)
            {
                #pragma unroll
                for (int h = 0; h < 2; ++h)   // = GvGroup<Int4v>::run of group g0 + 2 (q + p) + h
                {
                    const int ks = (g0 + 2 * (q + p) + h) * Int4v::G;
                    float g[4] = {0.f, 0.f, 0.f, 0.f};
                    #pragma unroll
                    for (int i = 0; i < Int4v::G; ++i) Int4v::mma(g, &cw[p][2 * h + i], bf[(ks + i) * 64], bf[(ks + i) * 64 + 1]);
                    const float2 sc = Int4v::scale(cs[p][h]);
                    acc[0] = fmaf(sc.x, g[0], acc[0]); acc[1] = fmaf(sc.x, g[1], acc[1]);
                    acc[2] = fmaf(sc.y, g[2], acc[2]); acc[3] = fmaf(sc.y, g[3], acc[3]);
                }
            }
    }
    if ((g1 - g0) & 1) gv_tile_part<Int4v>(tile, bf, g1 - 1, g1, acc);
}

// in-place repack of an Int4 record tensor [OT][NG2][288] for k-part count ks (must equal the launch's ks).
// grid (OT, ks), dynamic smem = NG2 / ks * 288 bytes; one CTA permutes one warp range.
__global__ void k_i4pair(uint8_t* w, int NG2, int ks)
{
    extern __shared__ u32 sm[];
    const int NGP = NG2 / ks, NP = NGP / 2, nw = NGP * Int4v::GR / 4;
    u32* r = reinterpret_cast<u32*>(w + (static_cast<size_t>(blockIdx.x) * NG2 + static_cast<size_t>(blockIdx.y) * NGP) * Int4v::GR);
    for (int i = threadIdx.x; i < nw; i += blockDim.x) sm[i] = r[i];
    __syncthreads();
    for (int i = threadIdx.x; i < NP * 144; i += blockDim.x)   // 144 words per pair
    {
        const int p = i / 144, o = i % 144;
        int src;
        if (o < 128) { const int l = o >> 2, j = o & 3, h = j >> 1, t = j & 1; src = (2 * p + h) * 72 + t * 32 + l; }
        else { const int k = o - 128, rr = k >> 1, h = k & 1; src = (2 * p + h) * 72 + 64 + rr; }
        r[p * 144 + o] = sm[src];
    }
}
inline void launch_i4pair(void* w, int OT, int NG2, int ks, cudaStream_t st)
{
    k_i4pair<<<dim3(OT, ks), 256, NG2 / ks * Int4v::GR, st>>>(static_cast<uint8_t*>(w), NG2, ks);
}
#endif

// stream: [out-tile][NG2 = K / (16 G)][GR].  CTA = tiles_per_cta output tiles x ks k-parts, one warp each
// (warp = tile * ks + part); parts are summed through smem in fixed order.  y[tok][ystride] of dtype odt.
template <class C>
__global__ void gv_gemv(const uint8_t* stream, const u32* xfrag, int K, int out_tiles, int tiles_per_cta, int ks,
                        int M, void* y, int ystride, int odt)
{
    __shared__ float4 red[32][32];   // [warp][lane]
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int tl = warp / ks, part = warp % ks;
    const int ot = blockIdx.x * tiles_per_cta + tl;
    const int NG2 = K / (16 * C::G), NGP = NG2 / ks;
    const bool on = tl < tiles_per_cta && ot < out_tiles;
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
#if DEC_I4PAIR
    if constexpr (C::GR == Int4v::GR)
    {
        if (on) gv_i4pair_part(stream + static_cast<size_t>(ot) * NG2 * C::GR, xfrag + lane * 2, part * NGP, (part + 1) * NGP, acc);
    }
    else
#endif
    if (on) gv_tile_part<C>(stream + static_cast<size_t>(ot) * NG2 * C::GR, xfrag + lane * 2, part * NGP, (part + 1) * NGP, acc);
    if (ks > 1)
    {
        if (on) red[warp][lane] = make_float4(acc[0], acc[1], acc[2], acc[3]);
        __syncthreads();
        if (on && part == 0)
            for (int q = 1; q < ks; ++q)
            {
                const float4 v = red[warp + q][lane];
                acc[0] += v.x; acc[1] += v.y; acc[2] += v.z; acc[3] += v.w;
            }
    }
    if (!on || part) return;
    // C regs: (row r, tok 2c), (row r, tok 2c+1), (row r+8, tok 2c), (row r+8, tok 2c+1)
    const int r = lane >> 2, c = lane & 3;
    const size_t o = static_cast<size_t>(ot) * 16 + r;
    if (2 * c < M)
    {
        gv::put(y, static_cast<size_t>(2 * c) * ystride + o, acc[0], odt);
        gv::put(y, static_cast<size_t>(2 * c) * ystride + o + 8, acc[2], odt);
    }
    if (2 * c + 1 < M)
    {
        gv::put(y, static_cast<size_t>(2 * c + 1) * ystride + o, acc[1], odt);
        gv::put(y, static_cast<size_t>(2 * c + 1) * ystride + o + 8, acc[3], odt);
    }
}

#ifdef DEC_SLOT
// gv_gemv with the BF16 output in this rank's send slot of collective s.c (row stride ystride), published by the
// last CTA when DEC_SLOT >= 2 (decslot.cuh). Same arithmetic and store order as gv_gemv with odt = O_BF16.
template <class C>
__global__ void gv_gemv_slot(const uint8_t* stream, const u32* xfrag, int K, int out_tiles, int tiles_per_cta, int ks,
                             int M, int ystride, dslot::Slot s)
{
    __shared__ float4 red[32][32];   // [warp][lane]
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int tl = warp / ks, part = warp % ks;
    const int ot = blockIdx.x * tiles_per_cta + tl;
    const int NG2 = K / (16 * C::G), NGP = NG2 / ks;
    const bool on = tl < tiles_per_cta && ot < out_tiles;
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
#if DEC_I4PAIR
    if constexpr (C::GR == Int4v::GR)
    {
        if (on) gv_i4pair_part(stream + static_cast<size_t>(ot) * NG2 * C::GR, xfrag + lane * 2, part * NGP, (part + 1) * NGP, acc);
    }
    else
#endif
    if (on) gv_tile_part<C>(stream + static_cast<size_t>(ot) * NG2 * C::GR, xfrag + lane * 2, part * NGP, (part + 1) * NGP, acc);
    if (ks > 1)
    {
        if (on) red[warp][lane] = make_float4(acc[0], acc[1], acc[2], acc[3]);
        __syncthreads();
        if (on && part == 0)
            for (int q = 1; q < ks; ++q)
            {
                const float4 v = red[warp + q][lane];
                acc[0] += v.x; acc[1] += v.y; acc[2] += v.z; acc[3] += v.w;
            }
    }
    const bool st = on && !part;
    if (st)
    {
        void* y = dslot::ptr(s);
        const int r = lane >> 2, c = lane & 3;
        const size_t o = static_cast<size_t>(ot) * 16 + r;
        if (2 * c < M)
        {
            gv::put(y, static_cast<size_t>(2 * c) * ystride + o, acc[0], gv::O_BF16);
            gv::put(y, static_cast<size_t>(2 * c) * ystride + o + 8, acc[2], gv::O_BF16);
        }
        if (2 * c + 1 < M)
        {
            gv::put(y, static_cast<size_t>(2 * c + 1) * ystride + o, acc[1], gv::O_BF16);
            gv::put(y, static_cast<size_t>(2 * c + 1) * ystride + o + 8, acc[3], gv::O_BF16);
        }
    }
#if DEC_SLOT >= 2
    dslot::publish(s, st);
#endif
}
#endif

#ifdef HOST_CMDS
inline uint16_t h_f2bf(float f) { uint32_t u; memcpy(&u, &f, 4); u += 0x7fffu + ((u >> 16) & 1u); return (uint16_t)(u >> 16); }
inline float h_h2f(uint16_t h)
{
    const int e = (h >> 10) & 31, m = h & 1023;
    const float v = e == 0 ? ldexpf((float)m, -24) : ldexpf((float)(m + 1024), e - 25);
    return (h & 0x8000) ? -v : v;
}

// m1 gv codec=mxfp8|bf16|int4 K= O= M= tpc= ks= odt=f16|bf16|f32 [pool= steps= reps= dump=DIR]
template <class C>
static int run_gv(const Args& a, const char* codec)
{
    const int K = a.i("K", 6144), O = a.i("O", 576), M = a.i("M", 1);
    const int ks = a.i("ks", 1), tpc = a.i("tpc", 4), warps = tpc * ks;
    const int steps = a.i("steps", 64), reps = a.i("reps", 9);
    const std::string dump = a.s("dump", ""), os = a.s("odt", "f16");
    const int odt = os == "f32" ? gv::O_F32 : os == "bf16" ? gv::O_BF16 : gv::O_F16;
    const bool bf = !C::SCALED, i4 = C::GR == 288;
    const int gk = 16 * C::G, NG = K / gk, OT = O / 16;
    if (K % gk || O % 16 || M < 1 || M > 8 || NG % ks || warps > 32) { fprintf(stderr, "bad shape\n"); return 1; }
    const size_t mat = (size_t)OT * NG * C::GR;
    const int P = a.i("pool", (long)std::min<size_t>(64, std::max<size_t>(2, (size_t)1e9 / mat)));
    Rng rng(a.i("seed", 1));
    std::vector<uint8_t> w(mat * P);
    if (bf)
    {
        uint16_t tab[4096];
        for (auto& t : tab) { float f = rng.uni() / 32; if (rng.next() & 1) f = -f; t = h_f2bf(f); }
        uint16_t* p = (uint16_t*)w.data();
        for (size_t i = 0; i < w.size() / 2; ++i) p[i] = tab[(i * 2654435761ull >> 7) & 4095];
    }
    else
    {
        rng.fill(w.data(), w.size());
        for (size_t g = 0; g < w.size() / C::GR; ++g)
        {
            uint8_t* q = &w[g * C::GR];
            for (int i = 0; i < 16 && i4; ++i) { const uint16_t hv = f2h(0.002f + 0.004f * rng.uni()); memcpy(q + 256 + i * 2, &hv, 2); }
            for (int i = 0; i < 512 && !i4; ++i) if ((q[i] & 0x7f) == 0x7f) q[i] ^= 1;      // E4M3FN: 0x7f / 0xff are NaN
            for (int i = 0; i < 16 && !i4; ++i) q[512 + i] = (uint8_t)(127 - 9 + rng.next() % 5);
        }
    }
    std::vector<uint16_t> x = rand_half(rng, (size_t)8 * K, 0.f, 1.f, true);
    for (int m = M; m < 8; ++m) for (int k = 0; k < K; ++k) x[(size_t)m * K + k] = 0;
    std::vector<uint16_t> xa = x;                       // operand actually fed to the mma
    if (bf) for (auto& v : xa) v = h_f2bf(h_h2f(v));
    std::vector<uint32_t> xf = make_xfrag(xa, K);
    const uint8_t* dw = dev_upload(w);
    const u32* dxf = dev_upload(xf);
#if DEC_I4PAIR
    if (i4) for (int i = 0; i < P; ++i) launch_i4pair(const_cast<uint8_t*>(dw) + (size_t)i * mat, OT, NG, ks, 0);   // pair records
    CK(cudaDeviceSynchronize());
#endif
    const size_t ob = odt == gv::O_F32 ? 4 : 2;
    uint8_t* y = dev_zero<uint8_t>((size_t)8 * O * ob);
    const int grid = (OT + tpc - 1) / tpc;
    cudaStream_t st;
    CK(cudaStreamCreate(&st));
    // warm=1: launch_warm (common.cuh) on the matrix, then the GEMV; warm=2: the warm-up alone. GEMV time out of the L2 =
    // us(warm=1) - us(warm=2), against us(warm=0) from DRAM.
    const int warm = a.i("warm", 0);
    auto step = [&](int i) {
        if (warm) launch_warm(dw + (size_t)(i % P) * mat, mat, st);
        if (warm != 2) gv_gemv<C><<<grid, warps * 32, 0, st>>>(dw + (size_t)(i % P) * mat, dxf, K, OT, tpc, ks, M, y, O, odt);
    };
    if (!dump.empty())
    {
        step(0);
        CK(cudaStreamSynchronize(st));
        CK(cudaGetLastError());
        FILE* f = fopen((dump + "/meta.txt").c_str(), "w");
        fprintf(f, "codec=%s\nK=%d\nO=%d\nM=%d\nodt=%s\n", codec, K, O, M, os.c_str());
        fclose(f);
        dump_host(dump, "x.bin", x.data(), x.size());
        dump_host(dump, "w.bin", w.data(), mat);
        dump_dev(dump, "y.bin", y, (size_t)8 * O * ob);
        printf("{\"cmd\":\"gv\",\"dumped\":\"%s\"}\n", dump.c_str());
        return 0;
    }
    for (int i = 0; i < P; ++i) step(i);
    CK(cudaStreamSynchronize(st));
    std::vector<double> t;
    for (int r = 0; r < reps; ++r)
    {
        const double t0 = now_us();
        for (int i = 0; i < steps; ++i) step(i);
        CK(cudaStreamSynchronize(st));
        t.push_back((now_us() - t0) / steps);
    }
    CK(cudaGetLastError());
    const Stats s = stats(t);
    printf("{\"cmd\":\"gv\",\"codec\":\"%s\",\"K\":%d,\"O\":%d,\"M\":%d,\"odt\":\"%s\",\"warps\":%d,\"tpc\":%d,\"ks\":%d,\"grid\":%d,\"pool\":%d,\"warm\":%d,\"mb_useful\":%.3f,"
           "\"us_med\":%.2f,\"us_min\":%.2f,\"gbps_med\":%.1f,\"gbps_best\":%.1f}\n",
           codec, K, O, M, os.c_str(), warps, tpc, ks, grid, P, warm, mat / 1e6, s.med, s.mn, mat / 1e6 / s.med * 1e3, mat / 1e6 / s.mn * 1e3);
    return 0;
}

static int cmd_gv(const Args& a)
{
    const std::string c = a.s("codec", "mxfp8");
    return c == "bf16" ? run_gv<Bf16w>(a, "bf16") : c == "int4" ? run_gv<Int4v>(a, "int4") : run_gv<Mxfp8>(a, "mxfp8");
}
#endif
