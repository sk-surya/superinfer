// P8-R2 calibration: extract hardware E2M1 code values via single-element A.
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

// a0 of lane 0 = code c (lowest nibble); B all 0x3 (uniform). Report D at lane0,(0,0).
__global__ void calib(const uint32_t* codes, float* out, int n) {
  const int lane = threadIdx.x % 32;
  const int idx = blockIdx.x;
  const uint32_t c = codes[idx];
  uint32_t a[4] = {c, 0u, 0u, 0u};
  uint32_t b[2] = {0x33333333U, 0x33333333U};
  float d[4];
  mma_mxf4(a, b, d);
  if (lane == 0 && n > 0) out[idx] = d[0];
}

int main() {
  if (cudaSetDevice(0) != cudaSuccess) return 77;
  uint32_t codes[16];
  for (int c = 0; c < 16; ++c) codes[c] = static_cast<uint32_t>(c);  // nibble 0 = c, rest 0
  uint32_t* dc; float* do_;
  cudaMalloc(&dc, sizeof(codes)); cudaMalloc(&do_, 16 * sizeof(float));
  cudaMemcpy(dc, codes, sizeof(codes), cudaMemcpyHostToDevice);
  calib<<<16, 32>>>(dc, do_, 16);
  cudaDeviceSynchronize();
  float h[16]; cudaMemcpy(h, do_, sizeof(h), cudaMemcpyDeviceToHost);
  std::printf("hardware E2M1 code -> value (D = code * val(3), B uniform code 3):\n");
  for (int c = 0; c < 16; ++c) std::printf("  code %2d -> D=%.6g\n", c, h[c]);
  cudaFree(dc); cudaFree(do_);
  return 0;
}
