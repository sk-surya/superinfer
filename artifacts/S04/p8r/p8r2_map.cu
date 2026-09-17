// Empirical extraction of the m16n8k64 mxf4nvf4 fragment maps.
//  A stimulus: only (lane,reg) nonzero (all 8 nibbles = code2 = 1.0), B all 1.0
//              -> D nonzero exactly in the row(s) that A element feeds.
//  B stimulus: A all 1.0, only (lane,reg) of B nonzero -> D nonzero exactly in the col.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>

__device__ __forceinline__ void mma(const uint32_t a[4], const uint32_t b[2], uint32_t sa,
                                    uint32_t sb, float o[4]) {
  const float c[4] = {0, 0, 0, 0};
  asm volatile(
      "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X."
      "f32.e2m1.e2m1.f32.ue4m3 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13},"
      "%14,{%15,%16},%17,{%18,%19};\n"
      : "=f"(o[0]), "=f"(o[1]), "=f"(o[2]), "=f"(o[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "f"(c[0]),
        "f"(c[1]), "f"(c[2]), "f"(c[3]), "r"(sa), "h"((uint16_t)0), "h"((uint16_t)0),
        "r"(sb), "h"((uint16_t)0), "h"((uint16_t)0));
}

__global__ void probe(const int* p, float* d) {
  const int lane = threadIdx.x & 31;
  const int tLane = p[0], tm = p[1], tReg = p[2];  // stimulus lane / mode(0=A,1=B) / reg
  uint32_t a[4] = {0x22222222u, 0x22222222u, 0x22222222u, 0x22222222u};
  uint32_t b[2] = {0x22222222u, 0x22222222u};
  if (tm == 0) { a[0] = a[1] = a[2] = a[3] = 0; if (lane == tLane) a[tReg] = 0x22222222u; }
  else         { b[0] = b[1] = 0;               if (lane == tLane) b[tReg] = 0x22222222u; }
  float o[4];
  mma(a, b, 0x38383838u, 0x38383838u, o);
  for (int i = 0; i < 4; ++i) d[lane * 4 + i] = o[i];
}

int main() {
  if (cudaSetDevice(0) != cudaSuccess) return 77;
  int* dp; float* dd;
  cudaMalloc(&dp, 3 * sizeof(int)); cudaMalloc(&dd, 128 * sizeof(float));
  float h[128];
  auto run = [&](int lane, int mode, int reg) {
    int p[3] = {lane, mode, reg};
    cudaMemcpy(dp, p, sizeof(p), cudaMemcpyHostToDevice);
    cudaMemset(dd, 0, 128 * sizeof(float));
    probe<<<1, 32>>>(dp, dd);
    cudaDeviceSynchronize();
    cudaMemcpy(h, dd, sizeof(h), cudaMemcpyDeviceToHost);
  };
  auto mat = [&](int m, int n) {
    const int lane = 4 * m + (n / 2);
    return h[lane * 4 + (2 * (m % 8) == 0 ? 0 : 0) + 0];  // placeholder, unused
  };
  (void)mat;

  std::printf("A map: (lane,reg) -> nonzero rows (m) [value ~1.0*64 if all k]\n");
  for (int lane = 0; lane < 32; ++lane) {
    for (int reg = 0; reg < 4; ++reg) {
      run(lane, 0, reg);
      std::printf("  lane%2d reg%d -> rows:", lane, reg);
      for (int l = 0; l < 32; ++l)
        for (int i = 0; i < 4; ++i)
          if (std::fabs(h[l * 4 + i]) > 1e-6f) {
            const int m = (l / 4) + 8 * (i / 2);
            std::printf(" %d(%.0f)", m, h[l * 4 + i]);
          }
      std::printf("\n");
    }
  }
  std::printf("\nB map: (lane,reg) -> nonzero cols (n)\n");
  for (int lane = 0; lane < 32; ++lane) {
    for (int reg = 0; reg < 2; ++reg) {
      run(lane, 1, reg);
      std::printf("  lane%2d reg%d -> cols:", lane, reg);
      for (int l = 0; l < 32; ++l)
        for (int i = 0; i < 4; ++i)
          if (std::fabs(h[l * 4 + i]) > 1e-6f) {
            const int n = 2 * (l % 4) + (i % 2);
            std::printf(" %d", n);
          }
      std::printf("\n");
    }
  }
  cudaFree(dp); cudaFree(dd);
  return 0;
}
