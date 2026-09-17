// P8-R2 probe: identify A/B fragment mapping empirically.
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>

__device__ __forceinline__ void mma_mxf4(const uint32_t a[4], const uint32_t b[2], float out[4]) {
  const float c[4] = {0.f, 0.f, 0.f, 0.f};
  asm volatile(
      "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X."
      "f32.e2m1.e2m1.f32.ue4m3 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13}, "
      "%14, {%15, %16}, %17, {%18, %19};\n"
      : "=f"(out[0]), "=f"(out[1]), "=f"(out[2]), "=f"(out[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
        "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]), "r"(0x38383838U),
        "h"(static_cast<uint16_t>(0)), "h"(static_cast<uint16_t>(0)), "r"(0x38383838U),
        "h"(static_cast<uint16_t>(0)), "h"(static_cast<uint16_t>(0)));
}

// mode 0: A stimulus (a[stim] bit), B all 1.0 ; mode 1: A all 1.0, B stimulus
__global__ void probe(const uint32_t* stim, int mode, float* d) {
  const int lane = threadIdx.x % 32;
  uint32_t a[4] = {0x22222222U, 0x22222222U, 0x22222222U, 0x22222222U};
  uint32_t b[2] = {0x22222222U, 0x22222222U};
  if (mode == 0) {
    for (int i = 0; i < 4; ++i) a[i] = 0;
    a[stim[0]] = stim[1];
  } else {
    b[0] = 0; b[1] = 0;
    b[stim[0]] = stim[1];
  }
  float out[4];
  mma_mxf4(a, b, out);
  d[lane * 4 + 0] = out[0];
  d[lane * 4 + 1] = out[1];
  d[lane * 4 + 2] = out[2];
  d[lane * 4 + 3] = out[3];
}

int main() {
  if (cudaSetDevice(0) != cudaSuccess) return 77;
  uint32_t* ds; float* dd;
  cudaMalloc(&ds, 2 * sizeof(uint32_t)); cudaMalloc(&dd, 128 * sizeof(float));
  // A stimulus: register 1, only lowest nibble = 1.0 code (0x2)
  uint32_t s0[2] = {1, 0x00000002U};
  cudaMemcpy(ds, s0, sizeof(s0), cudaMemcpyHostToDevice);
  probe<<<1, 32>>>(ds, 0, dd);
  cudaDeviceSynchronize();
  float h[128]; cudaMemcpy(h, dd, sizeof(h), cudaMemcpyDeviceToHost);
  std::printf("A a1=+1: nonzero D (m,n):");
  for (int lane = 0; lane < 32; ++lane)
    for (int i = 0; i < 4; ++i) {
      const int m = (lane / 4) + 8 * (i / 2);
      const int n = 2 * (lane % 4) + (i % 2);
      if (h[lane * 4 + i] != 0.f) std::printf(" (%d,%d)=%g", m, n, h[lane * 4 + i]);
    }
  std::printf("\n");
  // B stimulus: register 0, lowest nibble = 1.0
  uint32_t s1[2] = {0, 0x00000002U};
  cudaMemcpy(ds, s1, sizeof(s1), cudaMemcpyHostToDevice);
  probe<<<1, 32>>>(ds, 1, dd);
  cudaDeviceSynchronize();
  cudaMemcpy(h, dd, sizeof(h), cudaMemcpyDeviceToHost);
  std::printf("B b0 lane0=+1: nonzero D (m,n):");
  int shown = 0;
  for (int lane = 0; lane < 32 && shown < 24; ++lane)
    for (int i = 0; i < 4; ++i) {
      const int m = (lane / 4) + 8 * (i / 2);
      const int n = 2 * (lane % 4) + (i % 2);
      if (h[lane * 4 + i] != 0.f) { std::printf(" (%d,%d)=%g", m, n, h[lane * 4 + i]); ++shown; }
    }
  std::printf("\n");
  cudaFree(ds); cudaFree(dd);
  return 0;
}
