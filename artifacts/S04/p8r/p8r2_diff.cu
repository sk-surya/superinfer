// P8-R2: synthetic m16n8k64 mxf4nvf4 block-scaled MMA differential vs FP32 reference.
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

__device__ __forceinline__ void mma_mxf4(const float d[4], const uint32_t a[4],
                                         const uint32_t b[2], const float c[4],
                                         uint32_t sa, uint32_t sb, float out[4]) {
  asm volatile(
      "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X."
      "f32.e2m1.e2m1.f32.ue4m3 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13}, "
      "%14, {%15, %16}, %17, {%18, %19};\n"
      : "=f"(out[0]), "=f"(out[1]), "=f"(out[2]), "=f"(out[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
        "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]), "r"(sa),
        "h"(static_cast<uint16_t>(0)), "h"(static_cast<uint16_t>(0)), "r"(sb),
        "h"(static_cast<uint16_t>(0)), "h"(static_cast<uint16_t>(0)));
}

__device__ __forceinline__ float e2m1_value(uint32_t code) {
  const float mag[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
  return (code & 8U) ? -mag[code & 7U] : mag[code & 7U];
}

__device__ __forceinline__ float ue4m3_value(uint32_t byte) {
  const uint32_t e = (byte >> 3) & 0xFU, m = byte & 7U;
  if (e == 0) return static_cast<float>(m) / 8.0f * exp2f(-6.0f);
  return (1.0f + static_cast<float>(m) / 8.0f) * exp2f(static_cast<float>(e) - 7.0f);
}

// A is 16x64 e2m1 row-major (nibble codes); B is 64x8; SFA 16x4; SFB 4x8 (UE4M3 bytes).
__global__ void mma_kernel(const uint8_t* a, const uint8_t* b, const uint8_t* sfa,
                           const uint8_t* sfb, float* d) {
  const int lane = threadIdx.x % 32;
  const int g = lane >> 2, q = lane & 3;
  uint32_t regs[4] = {0, 0, 0, 0};
  const int row_lo = g, row_hi = g + 8;
  auto pack8 = [](const uint8_t* src) {
    uint32_t v = 0;
    for (int i = 0; i < 8; ++i) v |= static_cast<uint32_t>(src[i] & 0xFU) << (4 * i);
    return v;
  };
  regs[0] = pack8(a + (row_lo * 64 + q * 8));
  regs[1] = pack8(a + (row_lo * 64 + 32 + q * 8));
  regs[2] = pack8(a + (row_hi * 64 + q * 8));
  regs[3] = pack8(a + (row_hi * 64 + 32 + q * 8));
  uint32_t bregs[2];
  bregs[0] = pack8(b + (q * 8) * 8 + g);  // B[k][n], k-major rows of 8
  bregs[1] = pack8(b + (32 + q * 8) * 8 + g);
  // SFA: threads 0,1 of each quad supply scales; give every lane a plausible value.
  uint32_t sa = 0, sb = 0;
  for (int i = 0; i < 4; ++i) {
    sa |= static_cast<uint32_t>(sfa[row_lo * 4 + i]) << (8 * i);
  }
  for (int i = 0; i < 4; ++i) sb |= static_cast<uint32_t>(sfb[i * 8 + g]) << (8 * i);
  const float c[4] = {0.f, 0.f, 0.f, 0.f};
  float out[4];
  mma_mxf4(c, regs, bregs, c, sa, sb, out);
  (void)e2m1_value;
  (void)ue4m3_value;
  d[lane * 4 + 0] = out[0];
  d[lane * 4 + 1] = out[1];
  d[lane * 4 + 2] = out[2];
  d[lane * 4 + 3] = out[3];
}

static float e2m1_host(int code) {
  const float mag[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
  return (code & 8) ? -mag[code & 7] : mag[code & 7];
}
static float ue4m3_host(int byte) {
  const int e = (byte >> 3) & 0xF, m = byte & 7;
  if (e == 0) return static_cast<float>(m) / 8.0f * std::exp2(-6.0f);
  return (1.0f + static_cast<float>(m) / 8.0f) * std::exp2(static_cast<float>(e) - 7.0f);
}

int main() {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) return 77;
  if (cudaSetDevice(0) != cudaSuccess) return 77;
  std::vector<uint8_t> a(16 * 64), b(64 * 8), sfa(16 * 4), sfb(4 * 8);
  uint64_t state = 0x9E3779B97F4A7C15ULL;
  auto nxt = [&]() { state = state * 6364136223846793005ULL + 1442695040888963407ULL; return static_cast<uint32_t>(state >> 33); };
  for (auto& v : a) v = static_cast<uint8_t>(nxt() & 0xF);
  for (auto& v : b) v = static_cast<uint8_t>(nxt() & 0xF);
  // UE4M3 scales: values 0x20..0x2F are small positive; keep them exact in the reference.
  for (auto& v : sfa) v = static_cast<uint8_t>(0x28U | (nxt() & 0x3U));
  for (auto& v : sfb) v = static_cast<uint8_t>(0x28U | (nxt() & 0x3U));
  uint8_t *da, *db, *dsfa, *dsfb; float* dd;
  cudaMalloc(&da, a.size()); cudaMalloc(&db, b.size()); cudaMalloc(&dsfa, sfa.size());
  cudaMalloc(&dsfb, sfb.size()); cudaMalloc(&dd, 32 * 4 * sizeof(float));
  cudaMemcpy(da, a.data(), a.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(db, b.data(), b.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(dsfa, sfa.data(), sfa.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(dsfb, sfb.data(), sfb.size(), cudaMemcpyHostToDevice);
  mma_kernel<<<1, 32>>>(da, db, dsfa, dsfb, dd);
  if (cudaDeviceSynchronize() != cudaSuccess) { std::printf("exec failed\n"); return 1; }
  float host[128];
  cudaMemcpy(host, dd, sizeof(host), cudaMemcpyDeviceToHost);
  // Reference D[m][n] = sum_k e2m1(a[m][k])*ue4m3(sfa[m][k/16]) * e2m1(b[k][n])*ue4m3(sfb[k/16][n])
  double max_err = 0.0; double max_mag = 0.0;
  for (int m = 0; m < 16; ++m) {
    for (int n = 0; n < 8; ++n) {
      double sum = 0.0;
      for (int k = 0; k < 64; ++k) {
        const double av = e2m1_host(a[m * 64 + k]) * ue4m3_host(sfa[m * 4 + k / 16]);
        const double bv = e2m1_host(b[k * 8 + n]) * ue4m3_host(sfb[(k / 16) * 8 + n]);
        sum += av * bv;
      }
      // D fragment: lane = (m%8)*4 + (n/2)? m16n8 C layout: lane = (m%8)*4 + (n%2)*2? Use PTX C layout:
      // C/D m16n8: row = groupID + 8*(i/2), col = 2*(lane%4) + (i%2), i=0..3; groupID=lane/4.
      (void)sum;
    }
  }
  // Compare using the documented C layout: for i in 0..3, m = (lane/4) + 8*(i/2), n = 2*(lane%4)+(i%2).
  for (int lane = 0; lane < 32; ++lane) {
    const int g = lane / 4, q = lane % 4;
    for (int i = 0; i < 4; ++i) {
      const int m = g + 8 * (i / 2);
      const int n = 2 * q + (i % 2);
      double sum = 0.0;
      for (int k = 0; k < 64; ++k) {
        sum += static_cast<double>(e2m1_host(a[m * 64 + k]) * ue4m3_host(sfa[m * 4 + k / 16])) *
               static_cast<double>(e2m1_host(b[k * 8 + n]) * ue4m3_host(sfb[(k / 16) * 8 + n]));
      }
      max_mag = std::max(max_mag, std::fabs(sum));
      max_err = std::max(max_err, std::fabs(sum - host[lane * 4 + i]));
    }
  }
  std::printf("P8R2 differential: max_abs=%.6g max_mag=%.6g rel=%.6g\n", max_err, max_mag,
              max_mag > 0 ? max_err / max_mag : 0.0);
  return max_err <= 1e-3 * (max_mag > 1 ? max_mag : 1.0) ? 0 : 2;
}
