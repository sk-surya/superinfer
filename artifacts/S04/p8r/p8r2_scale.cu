#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
__device__ __forceinline__ void mma(const uint32_t a[4], const uint32_t b[2], uint32_t s, float out[4]) {
  const float c[4] = {0,0,0,0};
  asm volatile("mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X."
    "f32.e2m1.e2m1.f32.ue4m3 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13},"
    "%14,{%15,%16},%17,{%18,%19};\n"
    :"=f"(out[0]),"=f"(out[1]),"=f"(out[2]),"=f"(out[3])
    :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]),
     "f"(c[0]),"f"(c[1]),"f"(c[2]),"f"(c[3]),"r"(s),"h"((uint16_t)0),"h"((uint16_t)0),
     "r"(s),"h"((uint16_t)0),"h"((uint16_t)0));
}
__global__ void cal(const uint32_t* sv, float* out) {
  int i=blockIdx.x; uint32_t s=sv[i];
  uint32_t a[4]={0x33333333u,0,0,0}, b[2]={0x33333333u,0x33333333u};
  float d[4]; mma(a,b,s,d);
  if(threadIdx.x%32==0) out[i]=d[0];
}
int main(){
  cudaSetDevice(0);
  uint32_t sv[16]; for(int i=0;i<16;i++) sv[i]=0x30u|(uint32_t)i;
  uint32_t* ds; float* dd; cudaMalloc(&ds,64); cudaMalloc(&dd,64);
  cudaMemcpy(ds,sv,64,cudaMemcpyHostToDevice); cal<<<16,32>>>(ds,dd);
  cudaDeviceSynchronize(); float h[16]; cudaMemcpy(h,dd,64,cudaMemcpyDeviceToHost);
  std::printf("scale byte -> D (A=B=code3=1.5, so D = 2.25*sA*sB):\n");
  for(int i=0;i<16;i++) std::printf("  0x%02X -> %.6g\n", 0x30+i, h[i]);
  cudaFree(ds);cudaFree(dd); return 0;
}
