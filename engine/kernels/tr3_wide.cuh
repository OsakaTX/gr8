// TR3 K3 fc1 in the "wide" order (RESULTS-M1 item 1c): the same bytes regrouped so that a warp reads its 8 k-steps
// of tiles with three 64-bit loads (256 B per instruction, contiguous per warp) instead of eight 32-bit tile loads
// (96 B each; every load touches two 64 B lines and half of them are shared with the neighbour tile).
//
// Pack order of one fc1 matrix (gate or up of one block): [KSH k16 steps][8 tiles j][96 B].
// Wide order:                                            [KSH/8 groups][3 loads n][8 tiles j][32 lanes][8 B].
// The 6144 B of a group (8 k-steps x 8 tiles) stay where they are, so the reorder runs in place, one CTA per group
// (k_tr3_widen, once after the pack is on the device; K4 experts are left alone). Inside a group, lane l of warp j
// owns 192 bits: its chunk of tile (k-step 8*group + p, j) at bits 24p .. 24p+23, little endian over the three u64.
// A lane's chunk of a tile = stream bits [24l, 24l+24), MSB first = the new bits of its 8 weights. The 13 stream
// bits before them (the rest of its first 16-bit state) are the low 13 bits of lane l-1's chunk of the same tile
// (lane 31 for lane 0: the stream is tail-biting), fetched with one shuffle:
//   m = (chunk[l-1] << 24) | chunk[l];   state j = (m >> (21 - 3j)) & 0xffff,  j = 0..7.
#pragma once
#include <cstring>
#include <cuda_runtime.h>
#include "common.cuh"

constexpr int TR3W_GROUP = 8 * 8 * 96;          // bytes of one group of a K3 fc1 matrix
constexpr int TR3W_MAT = KSH * 8 * 96;          // bytes of one K3 fc1 matrix

// chunk p (0..7) of a lane out of its three u64 of a group
__host__ __device__ __forceinline__ u32 tr3w_chunk(const u64 (&w)[3], int p)
{
    const int bit = 24 * p, n = bit >> 6, sh = bit & 63;
    u64 v = w[n] >> sh;
    if (sh > 40) v |= w[n + 1] << (64 - sh);
    return static_cast<u32>(v) & 0xffffffu;
}

// the lane's chunk of one tile in pack order (words are LE u32, stream bits MSB first)
__host__ __device__ __forceinline__ u32 tr3w_tile_chunk(const uint8_t* tile, int lane)
{
    const int wi = lane * 3 / 4, off = (lane * 24) & 31;          // first word, first bit inside it (0, 24, 16, 8)
    const u32* t = reinterpret_cast<const u32*>(tile);
    const u64 m = (static_cast<u64>(t[wi]) << 32) | t[wi < 23 ? wi + 1 : 23];
    return static_cast<u32>(m >> (40 - off)) & 0xffffffu;
}

__host__ __device__ __forceinline__ void tr3w_pack(u64 (&v)[3], int p, u32 chunk)
{
    const int bit = 24 * p, n = bit >> 6, sh = bit & 63;
    v[n] |= static_cast<u64>(chunk) << sh;
    if (sh > 40) v[n + 1] |= static_cast<u64>(chunk) >> (64 - sh);
}

// In-place pack order -> wide order for every K3 expert of the rank's tr3.fc1. tab = tr3.table [E][6] (fc1 byte
// offset, -, -, n blocks, bits, -). grid = E * nb * 2 * (KSH / 8), nb = max blocks per expert (g6nr::EB: 3 at TP6,
// 4 at TP4), block = 256.
// Run it exactly once per tensor: a second run would scramble the bits.
__global__ void __launch_bounds__(256) k_tr3_widen(uint8_t* fc1, const long long* tab, int nb)
{
    const int lane = threadIdx.x & 31, j = threadIdx.x >> 5;
    const int g = blockIdx.x % (KSH / 8);
    const int bm = (blockIdx.x / (KSH / 8)) % (2 * nb);           // block * 2 + matrix
    const long long* tr = tab + 6 * (blockIdx.x / (KSH / 8 * 2 * nb));
    if (tr[4] != 3 || bm >= 2 * static_cast<int>(tr[3])) return;
    uint8_t* grp = fc1 + tr[0] + static_cast<size_t>(bm) * TR3W_MAT + static_cast<size_t>(g) * TR3W_GROUP;
    u64 v[3] = {0, 0, 0};
    #pragma unroll
    for (int p = 0; p < 8; ++p) tr3w_pack(v, p, tr3w_tile_chunk(grp + (p * 8 + j) * 96, lane));
    __syncthreads();                                              // every read of the group before its first write
    #pragma unroll
    for (int n = 0; n < 3; ++n) *reinterpret_cast<u64*>(grp + (n * 8 + j) * 256 + lane * 8) = v[n];
}

inline void launch_tr3_widen(void* fc1, const long long* tab, int n_experts, int nb, cudaStream_t st)
{
    k_tr3_widen<<<n_experts * nb * 2 * (KSH / 8), 256, 0, st>>>(static_cast<uint8_t*>(fc1), tab, nb);
}

// fc2 (down) of a K3 expert, same idea. Pack order of one block: [KSH n-tiles][8 k-tiles p][96 B]; a warp of the
// decode fc2 reads the 8 k-tiles of one n-tile (768 B, contiguous) with eight 32-bit tile loads.
// Wide order: [KSH n-tiles][3 loads n][32 lanes][8 B]: the 768 B of an n-tile stay where they are, the lane's chunk of
// k-tile p sits at bits 24p .. 24p+23 of its three u64. grid = E * nb * (KSH / 8), block = 256 (warp j = n-tile
// 8 * group + j). tab [E][6]: fc2 byte offset at [1]. Run it exactly once per tensor.
constexpr int TR3W_NT = 8 * 96;                 // bytes of one n-tile of a K3 fc2 block
__global__ void __launch_bounds__(256) k_tr3_widen2(uint8_t* fc2, const long long* tab, int nb)
{
    const int lane = threadIdx.x & 31, j = threadIdx.x >> 5;
    const int g = blockIdx.x % (KSH / 8);
    const int b = (blockIdx.x / (KSH / 8)) % nb;
    const long long* tr = tab + 6 * (blockIdx.x / (KSH / 8 * nb));
    if (tr[4] != 3 || b >= static_cast<int>(tr[3])) return;
    uint8_t* grp = fc2 + tr[1] + static_cast<size_t>(b) * TR3W_MAT + static_cast<size_t>(g * 8 + j) * TR3W_NT;
    u64 v[3] = {0, 0, 0};
    #pragma unroll
    for (int p = 0; p < 8; ++p) tr3w_pack(v, p, tr3w_tile_chunk(grp + p * 96, lane));
    __syncthreads();                                              // every read of the n-tile before its first write
    #pragma unroll
    for (int n = 0; n < 3; ++n) *reinterpret_cast<u64*>(grp + n * 256 + lane * 8) = v[n];
}

inline void launch_tr3_widen2(void* fc2, const long long* tab, int n_experts, int nb, cudaStream_t st)
{
    k_tr3_widen2<<<n_experts * nb * (KSH / 8), 256, 0, st>>>(static_cast<uint8_t*>(fc2), tab, nb);
}

// host reference of the fc2 reorder, one K3 block (TR3W_MAT bytes), in place
inline void tr3w_widen2_host(uint8_t* mat)
{
    uint8_t tmp[TR3W_NT];
    for (int nt = 0; nt < KSH; ++nt)
    {
        uint8_t* grp = mat + static_cast<size_t>(nt) * TR3W_NT;
        for (int lane = 0; lane < 32; ++lane)
        {
            u64 v[3] = {0, 0, 0};
            for (int p = 0; p < 8; ++p) tr3w_pack(v, p, tr3w_tile_chunk(grp + p * 96, lane));
            for (int n = 0; n < 3; ++n) memcpy(tmp + n * 256 + lane * 8, &v[n], 8);
        }
        memcpy(grp, tmp, TR3W_NT);
    }
}

// host reference of the same reorder, one K3 matrix (TR3W_MAT bytes), in place
inline void tr3w_widen_host(uint8_t* mat)
{
    uint8_t tmp[TR3W_GROUP];
    for (int g = 0; g < KSH / 8; ++g)
    {
        uint8_t* grp = mat + static_cast<size_t>(g) * TR3W_GROUP;
        for (int j = 0; j < 8; ++j)
            for (int lane = 0; lane < 32; ++lane)
            {
                u64 v[3] = {0, 0, 0};
                for (int p = 0; p < 8; ++p) tr3w_pack(v, p, tr3w_tile_chunk(grp + (p * 8 + j) * 96, lane));
                for (int n = 0; n < 3; ++n) memcpy(tmp + (n * 8 + j) * 256 + lane * 8, &v[n], 8);
            }
        memcpy(grp, tmp, TR3W_GROUP);
    }
}
