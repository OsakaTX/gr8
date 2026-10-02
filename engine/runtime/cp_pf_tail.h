// J4 4d CP prefill staging (prefill.cuh): which bytes of a staged page are sent and hashed. Shared by the kernels, the
// proxy ranges and the host test (tools/cp/tailtest.cpp).
// Only positions < start leave a rank: the tail page lp = start / ps (start not a multiple of ps) is sent and hashed for its
// first start % ps rows only. Rows >= start are the chunk's own rows, which every rank's attention appends into its pool
// while its proxy may still be sending use g (4D audit): a full-page send would read them
// mid-write (torn bytes) and a full-page hash would compare bytes the sender changed after hashing (false VERIFY mismatch).
#pragma once
#include <cstddef>
#if defined(__CUDACC__)
#define CP_PF_HD __host__ __device__ __forceinline__
#else
#define CP_PF_HD inline
#endif

// rows of logical page lp that hold positions < start (ps rows per page); 0 for a page at or past start (never negative:
// rows * CP_KVB is an RDMA byte count)
CP_PF_HD int cp_pf_rows(int lp, int ps, int start)
{
    const int r = start - lp * ps;
    return r <= 0 ? 0 : r < ps ? r : ps;
}
CP_PF_HD unsigned long long cp_mix(unsigned long long x)
{
    x += 0x9e3779b97f4a7c15ull; x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull; x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}
// one 8-B word of the pool at word index idx: the summand of the order-independent page hash (k_cp_pf_hash)
CP_PF_HD unsigned long long cp_pf_hw(unsigned long long word, size_t idx) { return cp_mix(word ^ cp_mix(idx)); }
