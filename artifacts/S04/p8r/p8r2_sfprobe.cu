#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
__device__ __forceinline__ void mma(const uint32_t a[4],const uint32_t b[2],uint32_t sa,uint32_t sb,float o[4]){
  const float c[4]={0,0,0,0};
  asm volatile("mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X."
    "f32.e2m1.e2m1.f32.ue4m3 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13},"
    "%14,{%15,%16},%17,{%18,%19};\n":"=f"(o[0]),"=f"(o[1]),"=f"(o[2]),"=f"(o[3])
    :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]),"f"(c[0]),"f"(c[1]),"f"(c[2]),"f"(c[3]),
     "r"(sa),"h"((uint16_t)0),"h"((uint16_t)0),"r"(sb),"h"((uint16_t)0),"h"((uint16_t)0));
}
// A: only slot `slot` on lane0 = code 3 (1.5). B all code3. sfa lane L byte0 = 0x38|(L%8); sb=0x38383838.
__global__ void probe(const int* slotp, float* out, int mode){
  int lane=threadIdx.x&31; int slot=slotp[0];
  uint32_t a[4]={0,0,0,0}; a[slot]=0x3u;  // lane0 only; others 0
  uint32_t b[2]={0x33333333u,0x33333333u};
  uint32_t sa=0x38383838u, sb=0x38383838u;
  if(lane==0) sa = 0x38383838u | (uint32_t)((lane&7));
  // vary sfa across lanes: byte0 = 0x38|(L%8)
  sa = 0x38383800u | 0x38u;    // bytes1..3 = 0x38
  sa |= (uint32_t)(0x38u | (uint32_t)(lane & 7u)); // byte0
  float o[4]; mma(a,b,sa,sb,o);
  if(mode==0) out[lane]=o[0];
}
int main(){
  cudaSetDevice(0); int* ds; float* dd; cudaMalloc(&ds,4); cudaMalloc(&dd,128);
  for(int slot=0;slot<4;++slot){
    int s=slot; cudaMemcpy(ds,&s,4,cudaMemcpyHostToDevice);
    probe<<<1,32>>>(ds,dd,0); cudaDeviceSynchronize();
    float h[32]; cudaMemcpy(h,dd,128,cudaMemcpyDeviceToHost);
    std::printf("A slot %d (nonzero on lane0): D[lane] for lanes0..8:", slot);
    for(int l=0;l<9;l++) std::printf(" %.4g", h[l]);
    std::printf("\n");
  }
  cudaFree(ds);cudaFree(dd); return 0;
}
