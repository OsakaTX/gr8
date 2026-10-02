// Mapping-only experiment. Keep cuBLAS QK/PV and the original ordered-list
// normalization/duplicate-merge arithmetic. Not wired into production.
#pragma once
namespace gpmap {
// Staging the dense row makes selected-list accesses shared-memory gathers.
// After gathering to registers the same arena becomes the normalized TOP list.
__global__ void normalize_expand(G6Attn a,int start,int n,int R,int D,
 const float* dense,const int* head,const int* next,__half* out,
 float* trace_scores=nullptr,float* trace_probs=nullptr){
 int t=threadIdx.x,row=blockIdx.x/a.heads,h=blockIdx.x%a.heads;
 __half* dst=out+((int64_t)h*R+row)*D;
 if(row>=n){for(int d=t;d<D;d+=256)dst[d]=__float2half_rn(0);return;}
 extern __shared__ float arena[];
 __shared__ float tmp[8],mx,den;
 const float* src=dense+((int64_t)h*R+row)*D;
 for(int d=t;d<D;d+=256)arena[d]=src[d];
 __syncthreads();
 float value[8],m=-INFINITY;
 #pragma unroll
 for(int k=0;k<8;k++){
  int j=t+256*k,id=a.indices[(int64_t)(start+row)*2048+j];
  bool valid=id>=0&&id<=a.positions[start+row]&&id<a.context;
  value[k]=valid?arena[id]:-INFINITY;m=fmaxf(m,value[k]);
  if(trace_scores)trace_scores[((int64_t)(start+row)*a.heads+h)*2048+j]=value[k];
 }
 // This CTA uses exactly the old ga::normalize lane/warp reduction order.
 m=ga::warp_max(m);if((t&31)==0)tmp[t/32]=m;__syncthreads();
 if(t==0){mx=-INFINITY;for(int j=0;j<8;j++)mx=fmaxf(mx,tmp[j]);}__syncthreads();
 float s=0;
 #pragma unroll
 for(int k=0;k<8;k++){float v=isfinite(mx)?expf(value[k]-mx):0;arena[t+k*256]=v;s+=v;}
 s=ga::warp_sum(s);if((t&31)==0)tmp[t/32]=s;__syncthreads();
 if(t==0){den=0;for(int j=0;j<8;j++)den+=tmp[j];if(den==0)ga::error(a,16);}__syncthreads();
 #pragma unroll
 for(int k=0;k<8;k++){int j=t+k*256;float v=den>0?arena[j]/den:0;arena[j]=v;
  if(trace_probs)trace_probs[((int64_t)(start+row)*a.heads+h)*2048+j]=v;}
 __syncthreads();
 // Same linked-entry sum order and sole RN-half conversion as v5. Invalid and
 // future IDs never entered the map; duplicate entries are not collapsed early.
 for(int id=t;id<D;id+=256){float sum=0;
  if(id<a.context)for(int j=head[(int64_t)row*a.context+id];j>=0;j=next[(int64_t)row*2048+j])sum+=arena[j];
  dst[id]=__float2half_rn(sum);
 }
}
} // namespace gpmap
