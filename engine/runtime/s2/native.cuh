// Native dispatch half of Service. Requires fc2's DEC_S2 rank.cuh APIs (cdddf65+).
// Opt-in integration only: this header is NOT included by legacy g6m4 today.
// Real variable-run bind/prefill view selection remains TP4/fc2-owned.
#pragma once
#ifdef DEC_S2
#include "service.h"
#include "../prefill.cuh"
#include "../../comms/fabric.h"
#include <map>

namespace g6s2 {
// Use g6m4's EXISTING capture arrays/main_enqueue. In particular solo_main must
// replay its production split3 A/B graphs, not capture an S2 replacement graph.
struct NativeContext {
    g6rt::Model& main;
    g6rt::Draft& draft;
    g6rt::Prefill& prefill;
    g6rt::Prefill& prefill2;
    g6rt::DraftPrefill& draft_prefill;
    g6::Fabric& main_fabric;
    g6::Fabric& draft_fabric;
    g6::Fabric& prefill_fabric;
    g6::Fabric& prefill_fabric2;
    cudaStream_t stream;
    g6rt::PfLanes* lane_streams = nullptr;
    g6rt::StepIn *input, *draft_input;
    g6rt::StepOut *output, *draft_output;
    int prefill_rows = 0, split = 3;
    bool two_lanes = true;
    std::function<void(uint64_t, int, int)> enqueue_main; // (step, rows, context): main fabric + CP collectives past R
    std::function<void(int bucket, int rows)> solo_main;
    std::function<void(int bucket, bool chain, int rows)> solo_draft;
    // Must select the canonical run in BOTH main and draft prefill bindings,
    // including QO offsets. begin/end cannot reset KV or other sessions.
    std::function<void(const Slot&)> begin_prefill;
    std::function<void()> end_prefill;
    // optional: around the main prefill step only, as legacy g6m4 (G6_CP staged view, cp_pf_enter / cp_pf_exit)
    std::function<void()> pf_enter, pf_exit;
    // optional: runtime error words the adapter does not own (CP fabric / CP staging proxy); nonzero fails the epoch
    std::function<unsigned()> extra_error;
};
class NativeDispatch {
    NativeContext c_;
    // first-step timer: off-by-default per-step timer (design r3/r4); set from G6_STEP_TMR=1 at construction
    const bool tmr_ = ::getenv("G6_STEP_TMR") && *::getenv("G6_STEP_TMR") == '1';
    static const char* step_kind(const Launch& l) {
        return l.step.kind == s2::Step::MAIN ? "main" : (l.step.kind == s2::Step::CHAIN ? "chain" : "draft");
    }
    // first-step timer: E1/E2 event label (design r3/r4).
    // first-step timer r3: the header is derived from the events of the
    // sessions ACTUALLY in the step, not the Step::event default: a CHAIN step
    // never set s.event (printed ROUND_MAIN while its sessions said TAIL/DECODE1),
    // and a mixed DRAFT step inherited v.event before filtering. Uniform -> that
    // class; mixed -> MIXED_MAIN; no per-session events (hand-built test steps)
    // -> the step-level field. The per-session `reqs slot=req:event` stays
    // authoritative either way.
    static const char* step_event(const Launch& l) {
        int cls = -1;
        for (size_t j = 0; j < l.events.size(); ++j) {
            const int e = l.events[j];
            if (cls == -1) cls = e;
            else if (cls != e) return "MIXED_MAIN";
        }
        if (cls == -1) cls = l.event;
        return cls == s2::Step::TAIL ? "TAIL_MAIN" : cls == s2::Step::DECODE1 ? "DECODE_MAIN_1" : "ROUND_MAIN";
    }
    g6rt::S2In *layout_ = nullptr, *draft_layout_ = nullptr;
    const g6rt::S2In *previous_layout_ = nullptr, *previous_draft_layout_ = nullptr;
    uint64_t prefill_sequence_ = 1, prefill_sequence2_ = 1;
    struct Graphs { cudaGraphExec_t a = nullptr, b = nullptr; std::vector<int> slots; };
    std::map<std::string, Graphs> graphs_;
    static void destroy(Graphs& g) {
        if (g.a) G6RT_CK(cudaGraphExecDestroy(g.a));
        if (g.b) G6RT_CK(cudaGraphExecDestroy(g.b));
        g.a = g.b = nullptr;
    }
    cudaGraphExec_t capture(const std::function<void()>& body) {
        cudaGraph_t graph; cudaGraphExec_t exec;
        G6RT_CK(cudaStreamBeginCapture(c_.stream,cudaStreamCaptureModeThreadLocal));
        body(); G6RT_CK(cudaStreamEndCapture(c_.stream,&graph));
        G6RT_CK(cudaGraphInstantiate(&exec,graph,0)); G6RT_CK(cudaGraphDestroy(graph)); return exec;
    }
    void fence() { G6RT_CK(cudaStreamSynchronize(c_.stream)); }
    // first-step timer: "slot=req:event" per session, in step order.
    // event code: 0 = TAIL (E1), 1 = DECODE1 (E2), 2 = ROUND.
    static std::string req_list(const Launch& l) {
        std::string s;
        for (size_t j = 0; j < l.requests.size(); ++j) {
            if (j) s += ",";
            const int e = l.events.empty() ? int(s2::Step::ROUND) : l.events[j];
            s += std::to_string(l.req_slots[j]) + "=" + std::to_string(l.requests[j]) + ":" + std::to_string(e);
        }
        return s;
    }
    void mode(int old, int next) {
        const auto t0 = std::chrono::steady_clock::now(); fence();
        if (old >= 0) g6rt::s2_hprev_from_solo(c_.draft,old,c_.stream);
        if (next >= 0) {
            const int base = c_.main.s2->base[next];
            require(base == c_.draft.d.s2->base[next], "main/draft survivor bases differ");
            // the survivor's own run (pages[next], set by bind), not the last solo run's pages
            const int pages = c_.main.s2->pages[next];
            require(pages > 0 && pages == c_.draft.d.s2->pages[next], "main/draft survivor runs differ");
            g6rt::s2_solo_run(c_.main,base,pages); g6rt::s2_solo_run(c_.draft.d,base,pages);
            g6rt::s2_hprev_to_solo(c_.draft,next,c_.stream);
        }
        c_.main.s2on = c_.draft.d.s2on = next < 0; fence();
        fprintf(stderr,"# s2 mode rank %d solo %d -> %d: %.3f ms (tables + hprev; no KV relocation)\n",c_.main.rank,old,next,
            std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count());
    }
    std::vector<int> launch(const Launch& l) {
        const bool draft = l.step.kind != s2::Step::MAIN, chain = l.step.kind == s2::Step::CHAIN;
        auto& model = draft ? c_.draft.d : c_.main;
        auto& fabric = draft ? c_.draft_fabric : c_.main_fabric;
        const int rows = l.step.M();
        g6rt::StepIn input{}; g6rt::S2In layout{}; native_inputs(l,input,layout);
        require(model.s2on == (l.solo < 0), "native solo mode differs from host");
        model.s2->h = layout;
        for (int j = 0; j < layout.ns; ++j) {
            model.s2->bk[layout.sess[j]] = l.buckets[j]; model.s2->src0[j] = l.source_rows[j];
        }
        const bool captured = l.solo < 0 && !graphs_.count(l.graph_key); // first-step timer: pre-insertion path decision
        if (captured) {
            const auto tc0 = tmr_ ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
            g6rt::s2_prepare(model,rows,c_.stream); fence(); // always outside capture
            Graphs g; g.slots = l.step.sess;
            if (draft) g.a = capture([&]{g6rt::launch_draft_step(c_.draft,rows,l.context,chain,c_.stream);});
            else {
                // Conservative production split3, including S>=2. Never silently
                // opt into unsplit without the separate IDs AND accepts gate.
                g.a = capture([&]{g6rt::launch_step(c_.main,rows,l.context,c_.stream,0,c_.split);});
                g.b = capture([&]{g6rt::launch_step(c_.main,rows,l.context,c_.stream,1,c_.split);});
            }
            graphs_.emplace(l.graph_key,std::move(g));
            if (tmr_) // first-step timer: capture time separate from launch+fence (design r4); actual key, not a literal
                fprintf(stderr,"# step_tmr rank %d step %llu %s %s cap %lld us (graph_key %s reqs %s)\n",c_.main.rank,
                    (unsigned long long)input.step,step_kind(l),step_event(l),
                    (long long)std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now()-tc0).count(),
                    l.graph_key.c_str(),req_list(l).c_str());
        }
        *(draft ? c_.draft_input : c_.input) = input;
        *(draft ? draft_layout_ : layout_) = layout;
        if (draft) {
            std::vector<uint32_t> sizes(3,uint32_t(rows*::HID*2));
            sizes[2] = uint32_t(glue::head_bytes(rows,model.nc)); c_.draft_fabric.enqueue_step(input.step,sizes);
        } else c_.enqueue_main(input.step,rows,input.context); // preserves existing RS order/fabric layout
        const auto tl0 = tmr_ ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{}; // after fabric enqueue, BEFORE the solo/S2 launch branches (design r4 fix 2)
        if (l.solo >= 0) {
            if (draft) c_.solo_draft(l.buckets[0],chain,rows); else c_.solo_main(l.buckets[0],rows);
        } else {
            auto& g = graphs_.at(l.graph_key); G6RT_CK(cudaGraphLaunch(g.a,c_.stream));
            if (g.b) G6RT_CK(cudaGraphLaunch(g.b,c_.stream));
        }
        fence(); const auto out = *(draft ? c_.draft_output : c_.output);
        if (tmr_) // first-step timer: launch+fence span (design r4); labels per r3, from the PRE-insertion decision, never graphs_.count
            fprintf(stderr,"# step_tmr rank %d step %llu %s %s %s %lld us (graph_key %s rows %d ctx %d reqs %s)\n",c_.main.rank,
                (unsigned long long)input.step,step_kind(l),step_event(l),
                l.solo >= 0 ? "SOLO_PRECAPTURED" : (captured ? "S2_CAPTURE" : "S2_REUSE"),
                (long long)std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now()-tl0).count(),
                l.graph_key.c_str(),rows,int(l.context),req_list(l).c_str());
        const unsigned fe = *fabric.dev().error, ae = out.attn_err, ce = c_.extra_error ? c_.extra_error() : 0u;
        if (fe || ae || ce) // S2 error words: name the error word before the fail-closed require
            fprintf(stderr,"# S2 step error words: fabric %u attn %u cp %u (%s step %llu rows %d solo %d ctx %d)\n",fe,ae,ce,
                draft ? (chain ? "chain" : "draft") : "main",(unsigned long long)input.step,rows,l.solo,int(l.context));
        require(!fe && !ae && !ce, "S2 native step fabric/attention error");
        return {out.tok,out.tok + rows};
    }
    void prefill(const Slot& slot,const Payload& payload,uint32_t first,uint32_t rows) {
        require(rows <= uint64_t(c_.prefill_rows)*(c_.two_lanes?2:1), "prefill quantum exceeds native lanes");
        c_.begin_prefill(slot); // real run/view selector supplied by TP4/fc2
        const int a = int(first), ma = std::min(int(rows),c_.prefill_rows), mb = int(rows)-ma;
        std::vector<int> pos(rows); for (unsigned r=0;r<rows;++r) pos[r]=a+int(r);
        auto& p = c_.prefill; auto& p2 = c_.prefill2; auto& dq = c_.draft_prefill;
        g6rt::pf_inputs(p,ma,payload.prompt.data()+a,pos.data());
        g6rt::pf_enqueue(c_.prefill_fabric,p,prefill_sequence_,c_.main.nl,ma);
        if (mb) {
            g6rt::pf_inputs(p2,mb,payload.prompt.data()+a+ma,pos.data()+ma);
            g6rt::pf_enqueue(c_.prefill_fabric2,p2,prefill_sequence2_,c_.main.nl,mb);
            if (c_.pf_enter) c_.pf_enter();
            if (c_.lane_streams) g6rt::pf_step2s(c_.main,p,p2,ma,mb,a+ma,a+ma+mb,prefill_sequence_,prefill_sequence2_,c_.stream,*c_.lane_streams);
            else g6rt::pf_step2(c_.main,p,p2,ma,mb,a+ma,a+ma+mb,prefill_sequence_,prefill_sequence2_,c_.stream);
            prefill_sequence2_ += c_.main.nl;
        } else {
            if (c_.pf_enter) c_.pf_enter();
            g6rt::pf_step(c_.main,p,ma,a+ma,prefill_sequence_,c_.stream);
        }
        if (c_.pf_exit) c_.pf_exit();
        fence(); prefill_sequence_ += c_.main.nl;
        g6rt::pf_draft_inputs(dq,ma,payload.prompt.data()+a+1,pos.data());
        g6rt::pf_draft(c_.draft,dq,p.x,ma,a+ma,c_.stream); fence();
        if (mb) {
            g6rt::pf_draft_inputs(dq,mb,payload.prompt.data()+a+ma+1,pos.data()+ma);
            g6rt::pf_draft(c_.draft,dq,p2.x,mb,a+ma+mb,c_.stream); fence();
        }
        require(!*c_.prefill_fabric.dev().error && (!mb || !*c_.prefill_fabric2.dev().error) &&
                !g6rt::attn_errors(c_.main) && !g6rt::attn_errors(c_.draft.d) && !(c_.extra_error && c_.extra_error()), "S2 prefill fabric/attention error");
        c_.end_prefill(); // restore decode views before yielding; NEVER late_draft
    }
public:
    explicit NativeDispatch(NativeContext c, const Config& scheduler) : c_(std::move(c)) {
        scheduler.validate();
        require(c_.main.s2 && c_.draft.d.s2 && c_.main.graph && c_.draft.d.graph && c_.split == 3 && c_.split < c_.main.nl,
                "native S2 needs initialized pools and production split3 graphs");
        require(c_.prefill_rows > 0 && c_.input && c_.draft_input && c_.output && c_.draft_output &&
                bool(c_.enqueue_main) && bool(c_.solo_main) && bool(c_.solo_draft) && bool(c_.begin_prefill) && bool(c_.end_prefill), "incomplete native context");
        require(scheduler.max_rows <= g6rt::MX && scheduler.limit() <= unsigned(c_.main.s2->nslots) &&
                scheduler.limit() <= unsigned(c_.draft.d.s2->nslots) && scheduler.pool_rows / PAGE >= unsigned(c_.main.s2->pool_pages) &&
                scheduler.pool_rows / PAGE >= unsigned(c_.draft.d.s2->pool_pages) && // logical capacity >= physical (pool.h)
                scheduler.prefill_rows == unsigned(c_.prefill_rows) &&
                scheduler.two_lanes == c_.two_lanes, "native storage/row/lane limits differ from advertised config");
        require(c_.main.in == c_.input && c_.draft.d.in == c_.draft_input && c_.main.out == c_.output && c_.draft.d.out == c_.draft_output,
                "native staging pointers differ from captured model inputs");
        G6RT_CK(cudaHostAlloc(reinterpret_cast<void**>(&layout_),sizeof(g6rt::S2In),cudaHostAllocMapped));
        G6RT_CK(cudaHostAlloc(reinterpret_cast<void**>(&draft_layout_),sizeof(g6rt::S2In),cudaHostAllocMapped));
        *layout_ = {}; *draft_layout_ = {};
        previous_layout_ = c_.main.s2->in; previous_draft_layout_ = c_.draft.d.s2->in;
        c_.main.s2->in = layout_; c_.draft.d.s2->in = draft_layout_;
    }
    ~NativeDispatch() {
        for (auto& item:graphs_) destroy(item.second);
        c_.main.s2->in = previous_layout_; c_.draft.d.s2->in = previous_draft_layout_;
        cudaFreeHost(layout_); cudaFreeHost(draft_layout_);
    }
    NativeDispatch(const NativeDispatch&) = delete; NativeDispatch& operator=(const NativeDispatch&) = delete;
    void retire(const Slot& old) {
        for (auto it=graphs_.begin();it!=graphs_.end();) {
            if (std::find(it->second.slots.begin(),it->second.slots.end(),int(old.key.slot)) == it->second.slots.end()) ++it;
            else { destroy(it->second); it=graphs_.erase(it); }
        }
    }
    DispatchHooks hooks() {
        return {
            [&](bool draft,int slot){return (draft?c_.draft.d:c_.main).s2->base[slot];},
            [&](bool draft,int ctx){return g6rt::bucket_of(draft?c_.draft.d:c_.main,ctx);},
            [&](int old,int next){mode(old,next);},
            [&](const Launch& l){return launch(l);},
            [&](const Slot& s,const Payload& p,uint32_t first,uint32_t rows){prefill(s,p,first,rows);}
        };
    }
    void synchronize() { fence(); }
};
} // namespace g6s2
#endif // DEC_S2
