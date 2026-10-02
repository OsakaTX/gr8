// Row-tile W4A16 kernels (NVFP4 experts, int4 g32 dense/shared expert). Our own code.
//   Weights are the mma A operand: a tile = 16 output rows x 16 inputs = 128 B, stored so that lane l
//   reads ONE u32 = {row r byte c, row r+8 byte c, row r byte c+4, row r+8 byte c+4} (r = l/4, c = l%4),
//   byte c of a row = inputs 2c (low nibble), 2c+1.  Activations (<= 8 tokens) are the B operand.
//   A scale group = G tiles + 16 row scales stored as 8 pairs {s[i], s[i+8]}; the scale is applied to the
//   group's fp32 partial sum (exact codes in the mma, no per-weight multiply).
#pragma once
#include "k1args.cuh"

struct Nvfp4
{
    static constexpr int G = 1, GR = 144;
    static __device__ __forceinline__ u32 dec(u32 byte) { return e2m1x2(byte); }
    static __device__ __forceinline__ u32 ld_scale(const uint8_t* grp, int lane) { return ld_nc16(grp + 128 + (lane >> 2) * 2); }
    static __device__ __forceinline__ float2 scale(u32 raw) { return __half22float2(as_h2(e4m3x2(raw))); }
};

struct Int4
{
    static constexpr int G = 2, GR = 288;
    // two offset-binary nibbles -> half2 (q - 8): q = 4*qh + ql, each 2-bit part through the E2M1
    // converter (codes 0..3 = 0, 0.5, 1, 1.5), exact in fp16.
    static __device__ __forceinline__ u32 dec(u32 byte)
    {
        const __half2 hl = as_h2(e2m1x2(byte & 0x33u));
        const __half2 hh = as_h2(e2m1x2((byte >> 2) & 0x33u));
        const __half2 k8 = __float2half2_rn(8.f), k2 = __float2half2_rn(2.f), km8 = __float2half2_rn(-8.f);
        return as_u32(__hfma2(hh, k8, __hfma2(hl, k2, km8)));
    }
    static __device__ __forceinline__ u32 ld_scale(const uint8_t* grp, int lane) { return ld_nc(grp + 256 + (lane >> 2) * 4); }
    static __device__ __forceinline__ float2 scale(u32 raw) { return __half22float2(as_h2(raw)); }
};

template <class C>
struct RtGroup
{
    u32 c[C::G];
    u32 s;
    __device__ __forceinline__ void load(const uint8_t* grp, int lane)
    {
        #pragma unroll
        for (int i = 0; i < C::G; ++i) c[i] = ld_nc(grp + i * 128 + lane * 4);
        s = C::ld_scale(grp, lane);
    }
    // acc += scale * (codes @ B fragments);  bf = fragment table [ks][lane][2] already offset to this lane
    __device__ __forceinline__ void run(const u32* bf, int ks, float (&acc)[4]) const
    {
        float g[4] = {0.f, 0.f, 0.f, 0.f};
        #pragma unroll
        for (int i = 0; i < C::G; ++i)
        {
            const u32 w = c[i];
#ifndef RT_PROBE
            mma16816(g, C::dec(w & 0xffu), C::dec((w >> 8) & 0xffu), C::dec((w >> 16) & 0xffu), C::dec(w >> 24),
                     bf[(ks + i) * 64], bf[(ks + i) * 64 + 1]);
#else       // cost probe: loads only, no dequant, no mma
            g[0] += __int_as_float((w ^ bf[(ks + i) * 64] ^ bf[(ks + i) * 64 + 1]) & 0x3fffffff);
#endif
        }
        const float2 sc = C::scale(s);
        acc[0] = fmaf(sc.x, g[0], acc[0]); acc[1] = fmaf(sc.x, g[1], acc[1]);
        acc[2] = fmaf(sc.y, g[2], acc[2]); acc[3] = fmaf(sc.y, g[3], acc[3]);
    }
};


// ---------------------------------------------------------------------------------------------
// FC1: one warp = one 16-column chunk of the intermediate dim (16 gate rows + 16 up rows) and one of
// `nparts` equal k ranges (nparts > 1: the CTA sums the parts through smem, see rt_fc1_cta).
// gate / up: [chunk][NG = 384/G groups], group records `stride` bytes apart. One interleaved stream
// [chunk][NG][gate GR | up GR] is (stream, stream + GR, 2 * GR); two separate [chunk][NG][GR] streams are (gate, up, GR).
// Grp = the group loader (RtGroup: 4-bit codes; GvGroup<Mxfp8> of densev.cuh: 8-bit codes).
template <class C, class Grp = RtGroup<C>>
__device__ __forceinline__ void rt_fc1_part(const uint8_t* gate, const uint8_t* up, int stride, const u32* xfrag, int chunk, int part,
                                            int nparts, float (&ag)[4], float (&au)[4])
{
    constexpr int PFB = 4 / C::G;
    constexpr int NG = KSH / C::G;
    const int lane = threadIdx.x & 31;
    const int NGP = NG / nparts, g0 = part * NGP;
    const size_t o0 = (static_cast<size_t>(chunk) * NG + g0) * stride;
    const uint8_t *bg = gate + o0, *bu = up + o0;
    const u32* bf = xfrag + lane * 2;

    Grp ng[PFB], nu[PFB];
    #pragma unroll
    for (int p = 0; p < PFB; ++p)
    {
        ng[p].load(bg + static_cast<size_t>(p) * stride, lane);
        nu[p].load(bu + static_cast<size_t>(p) * stride, lane);
    }
    for (int g = 0; g < NGP; g += PFB)
    {
        Grp cg[PFB], cu[PFB];
        #pragma unroll
        for (int p = 0; p < PFB; ++p) { cg[p] = ng[p]; cu[p] = nu[p]; }
        if (g + PFB < NGP)
        {
            #pragma unroll
            for (int p = 0; p < PFB; ++p)
            {
                ng[p].load(bg + static_cast<size_t>(g + PFB + p) * stride, lane);
                nu[p].load(bu + static_cast<size_t>(g + PFB + p) * stride, lane);
            }
        }
        #pragma unroll
        for (int p = 0; p < PFB; ++p)
        {
            cg[p].run(bf, (g0 + g + p) * C::G, ag);
            cu[p].run(bf, (g0 + g + p) * C::G, au);
        }
    }
}

// Whole-CTA fc1: warp = (chunk cl = warp / ks, k part = warp % ks), chunks chunk0 .. chunk0+cw-1 (< nch).
// Writes act[tok][chunk*16 + row] fp16, rows `aw` apart.  Every thread of the CTA must call this (it has a barrier when ks > 1).
template <class C, class Grp = RtGroup<C>>
__device__ __forceinline__ void rt_fc1_cta(const uint8_t* gate, const uint8_t* up, int stride, const u32* xfrag, int chunk0, int cw, int nch,
                                           int ks, float alpha_g, float alpha_u, __half* act /* [8][aw] */, int aw, float* smem)
{
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int cl = warp / ks, part = warp % ks, chunk = chunk0 + cl;
    const bool on = cl < cw && chunk < nch;
    float ag[4] = {0.f, 0.f, 0.f, 0.f}, au[4] = {0.f, 0.f, 0.f, 0.f};
    if (on) rt_fc1_part<C, Grp>(gate, up, stride, xfrag, chunk, part, ks, ag, au);
    if (ks > 1)
    {
        float4* ps = reinterpret_cast<float4*>(smem) + (warp * 32 + lane) * 2;
        if (on) { ps[0] = make_float4(ag[0], ag[1], ag[2], ag[3]); ps[1] = make_float4(au[0], au[1], au[2], au[3]); }
        __syncthreads();
        if (on && part == 0)
            for (int q = 1; q < ks; ++q)     // fixed order
            {
                const float4 g = ps[q * 64], u = ps[q * 64 + 1];
                ag[0] += g.x; ag[1] += g.y; ag[2] += g.z; ag[3] += g.w;
                au[0] += u.x; au[1] += u.y; au[2] += u.z; au[3] += u.w;
            }
    }
    if (!on || part) return;
    // C regs: (row r, tok 2c), (row r, tok 2c+1), (row r+8, tok 2c), (row r+8, tok 2c+1)
    const int r = lane >> 2, c = lane & 3;
    __half* a0 = act + (2 * c) * aw + chunk * 16 + r;
    a0[0] = __float2half_rn(silu_f(alpha_g * ag[0]) * (alpha_u * au[0]));
    a0[aw] = __float2half_rn(silu_f(alpha_g * ag[1]) * (alpha_u * au[1]));
    a0[8] = __float2half_rn(silu_f(alpha_g * ag[2]) * (alpha_u * au[2]));
    a0[aw + 8] = __float2half_rn(silu_f(alpha_g * ag[3]) * (alpha_u * au[3]));
}

// build a B fragment table [nks][32][2] in smem from row-major fp16 rows src[8][stride]
__device__ __forceinline__ void rt_build_bfrag(u32* smem, const __half* src, int stride, int nks)
{
    for (int i = threadIdx.x; i < nks * 64; i += blockDim.x)
    {
        const int ks = i >> 6, l = (i >> 1) & 31, reg = i & 1;
        smem[i] = __ldcg(reinterpret_cast<const u32*>(src + (l >> 2) * stride + ks * 16 + reg * 8 + 2 * (l & 3)));
    }
}

// ---------------------------------------------------------------------------------------------
// GEMV body: warps take output tiles ot0+warp, +nwarps, ... < ot_end; per tile NG2 groups (all k).
// stream: [out-tile][NG2][GR].  y[tok][out] = f16(scale_tok[tok] * acc) for tokens in mask (BF16 bits if bf16).
template <class C, class Grp = RtGroup<C>>
__device__ __forceinline__ void rt_gemv_body(const uint8_t* stream, const u32* bfrag, int NG2, int ot0, int ot_end,
                                             const float* tok_scale /* [8] */, float alpha, int mask,
                                             __half* y, int ystride, int bf16 = 0)
{
    constexpr int PFB = 4 / C::G;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int nwarps = blockDim.x >> 5;
    const int first = ot0 + warp;
    const int ntiles = (ot_end - first + nwarps - 1) / nwarps;
    if (first >= ot_end) return;
    const size_t tile_bytes = static_cast<size_t>(NG2) * C::GR;
    const u32* bf = bfrag + lane * 2;
    const int r = lane >> 2, c = lane & 3;
    const bool w0 = (mask >> (2 * c)) & 1, w1 = (mask >> (2 * c + 1)) & 1;
    const float s0 = alpha * (tok_scale ? tok_scale[2 * c] : 1.f), s1 = alpha * (tok_scale ? tok_scale[2 * c + 1] : 1.f);

    auto grp_ptr = [&](int q) { return stream + static_cast<size_t>(first + (q / NG2) * nwarps) * tile_bytes + static_cast<size_t>(q % NG2) * C::GR; };
    const int Q = ntiles * NG2;
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    Grp nx[PFB];
    #pragma unroll
    for (int p = 0; p < PFB; ++p)
        if (p < Q) nx[p].load(grp_ptr(p), lane);
    for (int q = 0; q < Q; q += PFB)
    {
        Grp cur[PFB];
        #pragma unroll
        for (int p = 0; p < PFB; ++p) cur[p] = nx[p];
        #pragma unroll
        for (int p = 0; p < PFB; ++p)
            if (q + PFB + p < Q) nx[p].load(grp_ptr(q + PFB + p), lane);
        #pragma unroll
        for (int p = 0; p < PFB; ++p)
        {
            if (q + p < Q)
            {
                const int g = (q + p) % NG2;
                cur[p].run(bf, g * C::G, acc);
                if (g == NG2 - 1)
                {
                    const int o = (first + ((q + p) / NG2) * nwarps) * 16 + r;
                    if (w0)
                    {
                        put16(y + (2 * c) * ystride + o, s0 * acc[0], bf16);
                        put16(y + (2 * c) * ystride + o + 8, s0 * acc[2], bf16);
                    }
                    if (w1)
                    {
                        put16(y + (2 * c + 1) * ystride + o, s1 * acc[1], bf16);
                        put16(y + (2 * c + 1) * ystride + o + 8, s1 * acc[3], bf16);
                    }
                    acc[0] = acc[1] = acc[2] = acc[3] = 0.f;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// K2: dense W4A16 GEMV / skinny GEMM, y[M][O] = x[M][K] @ W^T, x given as global fragment table.
// CTA = tiles_per_cta output tiles x ks k-parts, one warp each (warp = tile * ks + part); parts are summed
// through smem in fixed order.  Small grids (q_a: 128 tiles) need ks > 1 to fill the machine.
template <class C>
__device__ __forceinline__ void rt_tile_part(const uint8_t* tile, const u32* bf, int g0, int g1, float (&acc)[4])
{
    constexpr int PFB = 4 / C::G;
    const int lane = threadIdx.x & 31;
    const int Q = g1 - g0;
    RtGroup<C> nx[PFB];
    #pragma unroll
    for (int p = 0; p < PFB; ++p)
        if (p < Q) nx[p].load(tile + static_cast<size_t>(g0 + p) * C::GR, lane);
    for (int q = 0; q < Q; q += PFB)
    {
        RtGroup<C> cur[PFB];
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

template <class C>
__global__ void k2_gemv(const uint8_t* stream, const u32* xfrag, int K, int out_tiles, int tiles_per_cta, int ks,
                        int M, __half* y, int ystride)
{
    __shared__ float4 red[32][32];   // [warp][lane]
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int tl = warp / ks, part = warp % ks;
    const int ot = blockIdx.x * tiles_per_cta + tl;
    const int NG2 = K / (16 * C::G), NGP = NG2 / ks;
    const bool on = tl < tiles_per_cta && ot < out_tiles;
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    if (on) rt_tile_part<C>(stream + static_cast<size_t>(ot) * NG2 * C::GR, xfrag + lane * 2, part * NGP, (part + 1) * NGP, acc);
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
    const int r = lane >> 2, c = lane & 3, o = ot * 16 + r;
    const int mask = (1 << M) - 1;
    if ((mask >> (2 * c)) & 1)
    {
        y[(2 * c) * ystride + o] = __float2half_rn(acc[0]);
        y[(2 * c) * ystride + o + 8] = __float2half_rn(acc[2]);
    }
    if ((mask >> (2 * c + 1)) & 1)
    {
        y[(2 * c + 1) * ystride + o] = __float2half_rn(acc[1]);
        y[(2 * c + 1) * ystride + o + 8] = __float2half_rn(acc[3]);
    }
}
