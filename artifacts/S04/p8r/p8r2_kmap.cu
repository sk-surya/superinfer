// Extract the K mapping: place a single A element at position pa (lane,reg,nib) and a
// single B element at qb; with scales 1.0, D is nonzero iff k(pa) == k(qb).
// We print, for several pa, the set of qb sharing the same k.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <vector>

__device__ __forceinline__ void mma(const uint32_t a[4], const uint32_t b[2], float o[4]) {
  const float c[4] = {0, 0, 0, 0};
  asm volatile(
      "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X."
      "f32.e2m1.e2m1.f32.ue4m3 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13},"
      "%14,{%15,%16},%17,{%18,%19};\n"
      : "=f"(o[0]), "=f"(o[1]), "=f"(o[2]), "=f"(o[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "f"(c[0]),
        "f"(c[1]), "f"(c[2]), "f"(c[3]), "r"(0x38383838u), "h"((uint16_t)0),
        "h"((uint16_t)0), "r"(0x38383838u), "h"((uint16_t)0), "h"((uint16_t)0));
}

// pa[3] = {aLane, aReg, aNib};  qb[3] = {bLane, bReg, bNib}
__global__ void probe(const int* pa, const int* qb, float* d) {
  const int lane = threadIdx.x & 31;
  uint32_t a[4] = {0, 0, 0, 0}, b[2] = {0, 0};
  if (lane == pa[0]) a[pa[1]] = 0x2u << (4 * pa[2]);
  if (lane == qb[0]) b[qb[1]] = 0x2u << (4 * qb[2]);
  float o[4];
  mma(a, b, o);
  for (int i = 0; i < 4; ++i) d[lane * 4 + i] = o[i];
}

int main() {
  if (cudaSetDevice(0) != cudaSuccess) return 77;
  int *dp, *dq; float* dd;
  cudaMalloc(&dp, 3 * sizeof(int)); cudaMalloc(&dq, 3 * sizeof(int));
  cudaMalloc(&dd, 128 * sizeof(float));
  float h[128];
  auto nonzero = [&](const int pa[3], const int qb[3]) {
    cudaMemcpy(dp, pa, 3 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dq, qb, 3 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(dd, 0, 128 * sizeof(float));
    probe<<<1, 32>>>(dp, dq, dd);
    cudaDeviceSynchronize();
    cudaMemcpy(h, dd, sizeof(h), cudaMemcpyDeviceToHost);
    int cnt = 0;
    for (int i = 0; i < 128; ++i) if (std::fabs(h[i]) > 1e-6f) ++cnt;
    return cnt;
  };
  const int aps[4][3] = {{0, 0, 0}, {0, 1, 0}, {0, 2, 0}, {0, 3, 0}};
  for (int pi = 0; pi < 4; ++pi) {
    std::printf("A pos (lane0,reg%d,nib0) matches B positions:\n", aps[pi][1]);
    for (int bLane = 0; bLane < 32; ++bLane)
      for (int bReg = 0; bReg < 2; ++bReg)
        for (int bNib = 0; bNib < 8; ++bNib) {
          int qb[3] = {bLane, bReg, bNib};
          if (nonzero(aps[pi], qb)) std::printf("   B(lane%2d,reg%d,nib%d)\n", bLane, bReg, bNib);
        }
  }
  cudaFree(dp); cudaFree(dq); cudaFree(dd);
  return 0;
}
