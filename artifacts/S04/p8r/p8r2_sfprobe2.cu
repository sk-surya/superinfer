// P8-R2 scale-supplier probe (v2, corrected).
//
// For a target (row m, k-block kb) we place a SINGLE nonzero A element at (m, k=16*kb)
// on the one lane that owns it (per PTX 9.7.16.5.11), set B uniformly to code 3 (1.5),
// and give every lane a distinct byte value in scale byte `kb` (0x38|(L%8), i.e. UE4M3
// values 2.0..3.75).  D[m][n] then reveals which lane's byte `kb` the hardware used for
// (row m, block kb):
//     D = 1.5 (A) * sA * 1.5 (B) * 2.0 (sB) = 4.5 * sA
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <cmath>

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

// args: m, kb  (target row and k-block; k = 16*kb)
__global__ void probe(const int* args, float* out) {
  const int lane = threadIdx.x & 31;
  const int m = args[0], kb = args[1];
  const int g = m % 8;
  const int q = ((16 * kb) % 32) / 8;
  const int reg = (16 * kb < 32 ? 0 : 2) + (m >= 8 ? 1 : 0);
  const int holder = 4 * g + q;

  uint32_t a[4] = {0, 0, 0, 0};
  if (lane == holder) a[reg] = 0x3u;  // nibble 0 => k = 16*kb (since (16*kb)%8 == 0)

  uint32_t b[2] = {0x33333333u, 0x33333333u};

  // byte kb distinct per lane; other bytes fixed at 0x38.
  uint32_t sa = 0x38383838u;
  sa &= ~(0xFFu << (8 * kb));
  sa |= (uint32_t)(0x38u | (uint32_t)(lane & 7)) << (8 * kb);
  const uint32_t sb = 0x38383838u;

  float o[4];
  mma(a, b, sa, sb, o);
  if (lane == 4 * m) out[0] = o[0];  // c0 = (row groupID=m, col 2*(lane%4)=0)
}

int main() {
  if (cudaSetDevice(0) != cudaSuccess) return 77;
  int* da; float* dd;
  cudaMalloc(&da, 2 * sizeof(int)); cudaMalloc(&dd, sizeof(float));
  std::printf("measuring SFA supplier lane for each (row m, k-block kb)\n");
  std::printf("holder = lane that owns A[m][16*kb];  supplier byte value -> lane&7\n");
  for (int m = 0; m < 16; ++m) {
    std::printf("row %2d:", m);
    for (int kb = 0; kb < 4; ++kb) {
      int args[2] = {m, kb};
      cudaMemcpy(da, args, sizeof(args), cudaMemcpyHostToDevice);
      cudaMemset(dd, 0, sizeof(float));
      probe<<<1, 32>>>(da, dd);
      if (cudaDeviceSynchronize() != cudaSuccess) { std::printf(" execfail"); continue; }
      float h = 0.f; cudaMemcpy(&h, dd, sizeof(float), cudaMemcpyDeviceToHost);
      // sA = h/4.5 ; byte = 0x38 | (lane&7) ; value(0x38|k) = 2^(7-6)*(1+k/8) = 2+k/4
      const float sA = h / 4.5f;
      const int k = (int)lroundf((sA - 2.0f) * 4.0f);
      std::printf("  kb%d: D=%8.4f sA=%6.3f -> lane&7=%2d", kb, h, sA, k);
    }
    std::printf("\n");
  }
  cudaFree(da); cudaFree(dd);
  return 0;
}
