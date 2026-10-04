// G6_PF_OWN=3 full-head prefill weights (QO-INTEGRATION.md section 4): q_b and o_proj MXFP8 for all 64 heads, UK / UV
// BF16 for all 64 heads, logical head order. Identical on every rank.
//   qo_from_pack:  the names L%03d.qo.{q_b.mx.codes, q_b.mx.scales, o_proj.mx.codes, o_proj.mx.scales, kv_b.uk, kv_b.uv}
//   qo_from_ranks: assembled on the device from the NR ranks' head slices (m4sim holds all the packs). Rank r holds
//                  head group g = (r - l % NR) mod NR of layer l, heads start(64, g) .. + width(64, g) (g6nr.h,
//                  attn/bind.cu; TP6 edges 0, 11, 22, 33, 44, 54, 64).
#pragma once
#include "rank.cuh"
#include "../attn/bind.h"

namespace g6rt
{

struct QoW
{
    bind::Tensor qb_c, qb_s, o_c, o_s;   // [16384][2048], [16384][64], [6144][16384], [6144][512]
    G6AttnTensor uk{}, uv{};             // [64][192][512], [64][256][512] BF16
};
inline G6AttnTensor qo_attn_tensor(const TensorRef& r)
{
    G6AttnTensor t{};
    t.data = r.p; t.ndim = r.nd;
    for (int i = 0; i < 3 && i < r.nd; ++i) t.shape[i] = r.shape[i];
    return t;
}
inline bool qo_check(const QoW& w)
{
    auto is = [](const bind::Tensor& t, long long a, long long b) { return t.p && t.nd == 2 && t.shape[0] == a && t.shape[1] == b; };
    auto is3 = [](const G6AttnTensor& t, long long a, long long b, long long c) { return t.data && t.ndim == 3 && t.shape[0] == a && t.shape[1] == b && t.shape[2] == c; };
    return is(w.qb_c, 16384, 2048) && is(w.qb_s, 16384, 64) && is(w.o_c, HID, 16384) && is(w.o_s, HID, 512) &&
           is3(w.uk, 64, 192, 512) && is3(w.uv, 64, 256, 512);
}
inline bool qo_from_pack(const Find& find, int l, QoW& w)
{
    char n[80];
    auto get = [&](const char* s) { snprintf(n, sizeof n, "L%03d.qo.%s", l, s); return find(n); };
    w.qb_c = as_bind(get("q_b.mx.codes")); w.qb_s = as_bind(get("q_b.mx.scales"));
    w.o_c = as_bind(get("o_proj.mx.codes")); w.o_s = as_bind(get("o_proj.mx.scales"));
    w.uk = qo_attn_tensor(get("kv_b.uk")); w.uv = qo_attn_tensor(get("kv_b.uv"));
    return qo_check(w);
}
// finds[r] = rank r's tensors. Device allocations are never freed (load-time, like the packs).
inline QoW qo_from_ranks(const std::vector<Find>& finds, int l)
{
    if (finds.size() != (size_t)g6nr::NR) die("qo_from_ranks needs every rank's pack");
    QoW w;
    auto alloc2 = [](bind::Tensor& t, long long a, long long b, size_t esz) {
        t.p = dmalloc<char>((size_t)a * b * esz); t.nd = 2; t.shape[0] = a; t.shape[1] = b;
    };
    auto alloc3 = [](G6AttnTensor& t, long long a, long long b, long long c) {
        t.data = dmalloc<char>((size_t)a * b * c * 2); t.ndim = 3; t.shape[0] = a; t.shape[1] = b; t.shape[2] = c;
    };
    alloc2(w.qb_c, 16384, 2048, 1); alloc2(w.qb_s, 16384, 64, 1);
    alloc2(w.o_c, HID, 16384, 1); alloc2(w.o_s, HID, 512, 1);
    alloc3(w.uk, 64, 192, 512); alloc3(w.uv, 64, 256, 512);
    char n[80];
    for (int g = 0; g < g6nr::NR; ++g)
    {
        const int r = (g + l % g6nr::NR) % g6nr::NR, h0 = g6nr::start(g6nr::HEADS, g), h = g6nr::width(g6nr::HEADS, g);
        auto get = [&](const char* s) {
            snprintf(n, sizeof n, "L%03d.%s", l, s);
            TensorRef t = finds[r](n);
            if (!t.p) die(std::string("qo_from_ranks: rank ") + std::to_string(r) + " has no " + n);
            return t;
        };
        const TensorRef qc = get("q_b.mx.codes"), qs = get("q_b.mx.scales"), oc = get("o_proj.mx.codes"), os = get("o_proj.mx.scales");
        const TensorRef uk = get("kv_b.uk"), uv = get("kv_b.uv");
        if (qc.shape[0] != h * 256 || qc.shape[1] != 2048 || qs.shape[1] != 64 || oc.shape[0] != HID || oc.shape[1] != h * 256 ||
            os.shape[1] != h * 8 || uk.shape[0] != h || uv.shape[0] != h)
            die("qo_from_ranks: layer " + std::to_string(l) + " rank " + std::to_string(r) + " slice shapes are not head group " + std::to_string(g));
        char* qbc = (char*)w.qb_c.p; char* qbs = (char*)w.qb_s.p; char* oc_ = (char*)w.o_c.p; char* os_ = (char*)w.o_s.p;
        G6RT_CK(cudaMemcpy(qbc + (size_t)h0 * 256 * 2048, qc.p, (size_t)h * 256 * 2048, cudaMemcpyDeviceToDevice));
        G6RT_CK(cudaMemcpy(qbs + (size_t)h0 * 256 * 64, qs.p, (size_t)h * 256 * 64, cudaMemcpyDeviceToDevice));
        G6RT_CK(cudaMemcpy2D(oc_ + (size_t)h0 * 256, 16384, oc.p, (size_t)h * 256, (size_t)h * 256, HID, cudaMemcpyDeviceToDevice));
        G6RT_CK(cudaMemcpy2D(os_ + (size_t)h0 * 8, 512, os.p, (size_t)h * 8, (size_t)h * 8, HID, cudaMemcpyDeviceToDevice));
        G6RT_CK(cudaMemcpy((char*)w.uk.data + (size_t)h0 * 192 * 512 * 2, uk.p, (size_t)h * 192 * 512 * 2, cudaMemcpyDeviceToDevice));
        G6RT_CK(cudaMemcpy((char*)w.uv.data + (size_t)h0 * 256 * 512 * 2, uv.p, (size_t)h * 256 * 512 * 2, cudaMemcpyDeviceToDevice));
    }
    if (!qo_check(w)) die("qo_from_ranks: assembled shapes wrong");
    return w;
}

}   // namespace g6rt
