// P8-R2 (v3): m16n8k64 mxf4nvf4 block-scaled MMA differential vs independent FP32 reference.
//
// Fragment layouts are taken verbatim from CUTLASS include/cute/atom/mma_traits_sm120.hpp
// (SM120_16x8x64_TN_VS) and PTX ISA 9.7.16.5:
//   A  (T32,V32)->(M16,K64): reg r=(v/8): row g+8*(r&1), k = 8*q + (v%8) + 32*(r/2)
//   B  (T32,V16)->(N8 ,K64): reg r=(v/8): row(k) 8*q + (v%8) + 32*r, col(n) g
//   SFA(T32,V64)->(M16,K64): lane 4g -> row g ; lane 4g+1 -> row g+8 ; 4 bytes = K-blocks 0..3
//   SFB(T32,V64)->(N8 ,K64): lane 4n -> col n ; 4 bytes = K-blocks 0..3
//   C/D (SM80_16x8_Row): c0=(g,2q) c1=(g,2q+1) c2=(g+8,2q) c3=(g+8,2q+1)
// Selectors {byte-id,thread-id} = {0,0} are the only legal values for scale_vec::4X.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <vector>

__device__ __forceinline__ void mma_mxf4(const uint32_t a[4], const uint32_t b[2], uint32_t sa,
                                         uint32_t sb, float out[4]) {
  const float c[4] = {0.f, 0.f, 0.f, 0.f};
  asm volatile(
      "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X."
      "f32.e2m1.e2m1.f32.ue4m3 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13}, "
      "%14, {%15, %16}, %17, {%18, %19};\n"
      : "=f"(out[0]), "=f"(out[1]), "=f"(out[2]), "=f"(out[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "f"(c[0]),
        "f"(c[1]), "f"(c[2]), "f"(c[3]), "r"(sa), "h"(static_cast<uint16_t>(0)),
        "h"(static_cast<uint16_t>(0)), "r"(sb), "h"(static_cast<uint16_t>(0)),
        "h"(static_cast<uint16_t>(0)));
}

__global__ void mma_kernel(const uint8_t* a, const uint8_t* b, const uint8_t* sfa,
                           const uint8_t* sfb, float* d) {
  const int lane = threadIdx.x & 31;
  const int g = lane >> 2, q = lane & 3;
  auto pack8 = [](const uint8_t* src) {
    uint32_t v = 0;
    for (int i = 0; i < 8; ++i) v |= static_cast<uint32_t>(src[i] & 0xFU) << (4 * i);
    return v;
  };
  uint32_t aregs[4];
  aregs[0] = pack8(a + (g + 0) * 64 + q * 8);
  aregs[1] = pack8(a + (g + 8) * 64 + q * 8);
  aregs[2] = pack8(a + (g + 0) * 64 + 32 + q * 8);
  aregs[3] = pack8(a + (g + 8) * 64 + 32 + q * 8);
  // B fragments span K (stride 8 in the [k*8+n] array), not N.
  uint32_t bregs[2] = {0u, 0u};
  for (int i = 0; i < 8; ++i) {
    bregs[0] |= static_cast<uint32_t>(b[(q * 8 + i) * 8 + g] & 0xFU) << (4 * i);
    bregs[1] |= static_cast<uint32_t>(b[(32 + q * 8 + i) * 8 + g] & 0xFU) << (4 * i);
  }
  // SFA: lane 4g -> row g, lane 4g+1 -> row g+8 (others unused duplicates).
  const int m_owner = (lane & 1) ? (g + 8) : g;
  uint32_t sa = 0;
  for (int i = 0; i < 4; ++i) sa |= static_cast<uint32_t>(sfa[m_owner * 4 + i]) << (8 * i);
  // SFB: lane 4n -> col n.
  const int n_owner = g;
  uint32_t sb = 0;
  for (int i = 0; i < 4; ++i) sb |= static_cast<uint32_t>(sfb[i * 8 + n_owner]) << (8 * i);
  float out[4];
  mma_mxf4(aregs, bregs, sa, sb, out);
  for (int i = 0; i < 4; ++i) d[lane * 4 + i] = out[i];
}

static float e2m1_host(int code) {
  const float mag[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
  return (code & 8) ? -mag[code & 7] : mag[code & 7];
}
// IEEE E4M3, bias 7 (confirmed by the on-GPU scale-supplier probe).
static float ue4m3_host(int byte) {
  const int s = (byte >> 7) & 1, e = (byte >> 3) & 0xF, m = byte & 7;
  if (e == 0) return (s ? -1.f : 1.f) * (static_cast<float>(m) / 8.0f) * std::exp2(-6.0f);
  return (s ? -1.f : 1.f) * (1.0f + static_cast<float>(m) / 8.0f) * std::exp2(static_cast<float>(e) - 7.0f);
}

static int run_case(bool random_scales) {
  std::vector<uint8_t> a(16 * 64), b(64 * 8), sfa(16 * 4), sfb(4 * 8);
  uint64_t state = 0x9E3779B97F4A7C15ULL;
  auto nxt = [&]() {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<uint32_t>(state >> 33);
  };
  for (auto& v : a) v = static_cast<uint8_t>(nxt() & 0xF);
  for (auto& v : b) v = static_cast<uint8_t>(nxt() & 0xF);
  for (auto& v : sfa) v = random_scales ? static_cast<uint8_t>(0x30U | (nxt() & 0xFU)) : 0x38U;
  for (auto& v : sfb) v = random_scales ? static_cast<uint8_t>(0x30U | (nxt() & 0xFU)) : 0x38U;

  uint8_t *da, *db, *dsfa, *dsfb;
  float* dd;
  cudaMalloc(&da, a.size()); cudaMalloc(&db, b.size());
  cudaMalloc(&dsfa, sfa.size()); cudaMalloc(&dsfb, sfb.size());
  cudaMalloc(&dd, 128 * sizeof(float));
  cudaMemcpy(da, a.data(), a.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(db, b.data(), b.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(dsfa, sfa.data(), sfa.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(dsfb, sfb.data(), sfb.size(), cudaMemcpyHostToDevice);
  mma_kernel<<<1, 32>>>(da, db, dsfa, dsfb, dd);
  if (cudaDeviceSynchronize() != cudaSuccess) { std::printf("exec failed\n"); return 1; }
  float host[128];
  cudaMemcpy(host, dd, sizeof(host), cudaMemcpyDeviceToHost);

  double max_err = 0.0, max_mag = 0.0;
  for (int lane = 0; lane < 32; ++lane) {
    const int g = lane / 4, q = lane % 4;
    for (int i = 0; i < 4; ++i) {
      const int m = g + 8 * (i / 2);
      const int n = 2 * q + (i % 2);
      double sum = 0.0;
      for (int k = 0; k < 64; ++k)
        sum += static_cast<double>(e2m1_host(a[m * 64 + k]) * ue4m3_host(sfa[m * 4 + k / 16])) *
               static_cast<double>(e2m1_host(b[k * 8 + n]) * ue4m3_host(sfb[(k / 16) * 8 + n]));
      max_mag = std::max(max_mag, std::fabs(sum));
      max_err = std::max(max_err, std::fabs(sum - host[lane * 4 + i]));
    }
  }
  const double denom = max_mag > 1.0 ? max_mag : 1.0;
  std::printf("P8R2 v3 (%-6s scales): max_abs=%.6g max_mag=%.6g rel=%.6g  %s\n",
              random_scales ? "random" : "uniform", max_err, max_mag, max_err / denom,
              max_err <= 1e-3 * denom ? "PASS" : "FAIL");
  cudaFree(da); cudaFree(db); cudaFree(dsfa); cudaFree(dsfb); cudaFree(dd);
  return max_err <= 1e-3 * denom ? 0 : 2;
}

int main() {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) return 77;
  if (cudaSetDevice(0) != cudaSuccess) return 77;
  const int r0 = run_case(false);
  const int r1 = run_case(true);
  return (r0 == 0 && r1 == 0) ? 0 : 2;
}
