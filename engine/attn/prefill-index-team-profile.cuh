#pragma once
// Host-only measurement helpers; no kernel arithmetic or dispatch here.
#include <new>
namespace gp {
struct PrefillProfile {
    cudaEvent_t* events=nullptr;
    int capacity=0,created=0,used=0;
    G6AttnPrefillProfile report{};
};
static cudaError_t profile_release(PrefillProfile& p){
    cudaError_t first=cudaSuccess;
    for(int i=0;i<p.created;i++){auto e=cudaEventDestroy(p.events[i]);if(first==cudaSuccess)first=e;}
    delete[] p.events;p.events=nullptr;p.capacity=p.created=p.used=0;return first;
}
static cudaError_t profile_create(PrefillProfile& p,bool enabled,int rows,int tile){
    if(!enabled)return cudaSuccess;
    if(rows<=0||tile<=0)return cudaErrorInvalidValue;
    const int64_t count=1+5*((int64_t(rows)+tile-1)/tile);
    if(count>INT_MAX)return cudaErrorInvalidValue;
    p.capacity=int(count);p.events=new(std::nothrow) cudaEvent_t[p.capacity]{};
    if(!p.events){p.capacity=0;return cudaErrorMemoryAllocation;}
    for(;p.created<p.capacity;p.created++){
        auto e=cudaEventCreate(p.events+p.created);
        if(e!=cudaSuccess){profile_release(p);return e;}
    }
    p.report.status=G6_PF_PROFILE_EMPTY;return cudaSuccess;
}
static void profile_reset(PrefillProfile& p){
    uint64_t next=p.report.sequence+1;p.report={};p.report.sequence=next;
    p.report.status=p.events?G6_PF_PROFILE_FAILED:G6_PF_PROFILE_OFF;p.used=0;
}
static cudaError_t profile_mark(PrefillProfile& p,cudaStream_t st){
    if(p.used>=p.capacity)return cudaErrorInvalidValue;
    auto e=cudaEventRecord(p.events[p.used],st);if(e==cudaSuccess)p.used++;return e;
}
static cudaError_t profile_begin(PrefillProfile& p,cudaStream_t st,int rows,int context,int heads,int tile,bool full,bool traced,bool& active){
    active=false;auto& r=p.report;r.rows=rows;r.context=context;r.heads=heads;
    r.query_tile=tile;r.tiles=(rows-1)/tile+1;r.full_index=full;r.traced=traced;
    if(!p.events)return cudaSuccess;
    cudaStreamCaptureStatus capture;auto e=cudaStreamIsCapturing(st,&capture);if(e!=cudaSuccess)return e;
    if(capture!=cudaStreamCaptureStatusNone){r.status=G6_PF_PROFILE_CAPTURE_SKIPPED;return cudaSuccess;}
    e=profile_mark(p,st);if(e==cudaSuccess)active=true;return e;
}
static cudaError_t profile_read(PrefillProfile& p,G6AttnPrefillProfile& out){
    out=p.report;if(out.status!=G6_PF_PROFILE_READY)return cudaSuccess;
    // Mark failed first: any CUDA error leaves no old/retryable timing record.
    p.report.status=G6_PF_PROFILE_FAILED;out.status=G6_PF_PROFILE_FAILED;
    if(p.used!=1+5*out.tiles)return cudaErrorInvalidValue;
    auto e=cudaEventSynchronize(p.events[p.used-1]);if(e!=cudaSuccess)return e;
    G6AttnPrefillProfile result=out;
    auto add=[&](int a,int b,double& sum){float ms=0;auto rc=cudaEventElapsedTime(&ms,p.events[a],p.events[b]);if(rc==cudaSuccess)sum+=ms;return rc;};
#define GP_PROFILE_ADD(a,b,target) do{e=add(a,b,target);if(e!=cudaSuccess)return e;}while(0)
    GP_PROFILE_ADD(0,1,result.glue_ms);
    for(int tile=0;tile<out.tiles;tile++){
        int i=1+5*tile; // index, UK, MLA, UV, end
        if(tile)GP_PROFILE_ADD(i-1,i,result.other_ms);
        GP_PROFILE_ADD(i,i+1,(out.full_index?result.index_ms:result.glue_ms));
        GP_PROFILE_ADD(i+1,i+2,result.glue_ms);
        GP_PROFILE_ADD(i+2,i+3,result.mla_ms);
        GP_PROFILE_ADD(i+3,i+4,result.glue_ms);
    }
    GP_PROFILE_ADD(0,p.used-1,result.total_ms);
#undef GP_PROFILE_ADD
    result.status=G6_PF_PROFILE_READY;out=result;p.report.status=G6_PF_PROFILE_CONSUMED;return cudaSuccess;
}
}
