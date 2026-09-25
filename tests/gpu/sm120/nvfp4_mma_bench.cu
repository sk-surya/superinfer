// S04-P8R Arm A/B/C: native SM120 block-scaled NVFP4 mma.sync GEMV on the real Qwen3.8 shapes.
//
// Arm A (N=1) : one logical decode token, 7 of 8 MMA columns are dead.
// Arm B (N=8) : eight genuine tokens through the same tile -> separates tile under-utilization
//               from fundamental native-MMA cost.
// Arm C       : the bare warp-level MMA loop (activation already quantized) -> the hardware floor.
//
// Weight layout is SuperInfer's existing NVFP4 layout:
//   packed[rows][inputs/2]  (one byte = two E2M1 codes, low nibble = even k)
//   scales[rows][inputs/16] (E4M3FN block scales, bias 7 -- same encoding the .ue4m3 operand uses)
//   tensor_scale            (FP32 scalar)
// Fragment layouts / scale ownership are the P8-R2-verified contract.
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <functional>
#include <string_view>
#include <vector>

// ---------------------------------------------------------------- device helpers
__device__ __forceinline__ float decode_e2m1(std::uint8_t code) {
  const float mag[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
  return (code & 8U) ? -mag[code & 7U] : mag[code & 7U];
}
__device__ __forceinline__ float decode_e4m3(std::uint8_t code) {
  const std::uint8_t e = (code >> 3U) & 0x0FU, m = code & 0x07U;
  if (e == 0) return static_cast<float>(m) / 8.0f * exp2f(-6.0f);
  return (1.0f + static_cast<float>(m) / 8.0f) * exp2f(static_cast<float>(e) - 7.0f);
}
__device__ __forceinline__ std::uint8_t encode_e2m1(float x) {
  const float mag[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
  const float ax = fabsf(x);
  int best = 0;
  float bd = 1e30f;
  for (int i = 0; i < 8; ++i) {
    const float d = fabsf(ax - mag[i]);
    if (d < bd) { bd = d; best = i; }
  }
  return static_cast<std::uint8_t>((x < 0.f) ? (best | 8) : best);
}
__device__ __forceinline__ std::uint8_t encode_e4m3(float x) {
  if (!(x > 0.f)) return 0;
  int e = static_cast<int>(floorf(log2f(x))) + 7;
  if (e < 1) e = 1;
  if (e > 15) return 0x7FU;
  const float base = exp2f(static_cast<float>(e) - 7.0f);
  int m = static_cast<int>(lroundf((x / base - 1.0f) * 8.0f));
  if (m >= 8) { m = 0; if (++e > 15) return 0x7FU; }
  if (m < 0) m = 0;
  return static_cast<std::uint8_t>((e << 3) | m);
}
__device__ __forceinline__ std::uint32_t ld_b32(const std::uint8_t* p) {
  return *reinterpret_cast<const std::uint32_t*>(p);
}
__device__ __forceinline__ void mma_mxf4(const std::uint32_t a[4], const std::uint32_t b[2],
                                         std::uint32_t sa, std::uint32_t sb, float d[4]) {
  const float c0 = d[0], c1 = d[1], c2 = d[2], c3 = d[3];  // C accumulates into D
  asm volatile(
      "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X."
      "f32.e2m1.e2m1.f32.ue4m3 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13}, "
      "%14, {%15, %16}, %17, {%18, %19};\n"
      : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "f"(c0),
        "f"(c1), "f"(c2), "f"(c3), "r"(sa), "h"(static_cast<std::uint16_t>(0)),
        "h"(static_cast<std::uint16_t>(0)), "r"(sb), "h"(static_cast<std::uint16_t>(0)),
        "h"(static_cast<std::uint16_t>(0)));
}

// ---------------------------------------------------------------- activation quantization
// input[inputs] -> packed[inputs/2], scales[inputs/16]; also writes dequantised activations.
__global__ void quantize_activation(const float* in, std::size_t inputs, std::uint8_t* packed,
                                    std::uint8_t* scales, float* dequant) {
  const std::size_t blk = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (blk * 16U >= inputs) return;
  const float* p = in + blk * 16U;
  float amax = 0.f;
  for (int i = 0; i < 16; ++i) amax = fmaxf(amax, fabsf(p[i]));
  const std::uint8_t sb = encode_e4m3(amax / 6.0f);
  const float sd = decode_e4m3(sb);
  const float inv = (sd > 0.f) ? (1.0f / sd) : 0.f;
  scales[blk] = sb;
  for (int i = 0; i < 8; ++i) {
    const float lo = fminf(fmaxf(p[2 * i] * inv, -6.f), 6.f);
    const float hi = fminf(fmaxf(p[2 * i + 1] * inv, -6.f), 6.f);
    packed[blk * 8 + i] = static_cast<std::uint8_t>((encode_e2m1(hi) << 4) | encode_e2m1(lo));
  }
  if (dequant != nullptr)
    for (int i = 0; i < 16; ++i)
      dequant[blk * 16 + i] =
          decode_e2m1((packed[blk * 8 + i / 2] >> (4 * (i % 2))) & 0xFU) * sd;
}

// ---------------------------------------------------------------- Arm A/B: native MMA GEMV
// N = number of activation columns fed through the tile (1 = Arm A, 8 = Arm B).
template <int N>
__global__ void mma_gemv(const std::uint8_t* packed, const std::uint8_t* scales,
                         const std::uint8_t* bpacked, const std::uint8_t* bscales,
                         const float* tensor_scale, float* out, std::size_t rows,
                         std::size_t inputs) {
  const int lane = threadIdx.x & 31;
  const int warp = static_cast<int>((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
  const int g = lane >> 2, q = lane & 3;
  const std::size_t row0 = static_cast<std::size_t>(warp) * 16U;
  if (row0 >= rows) return;
  const std::size_t wrow = inputs / 2U;   // packed weight row bytes
  const std::size_t srow = inputs / 16U;  // weight scale row bytes
  const std::size_t brow = inputs / 2U;   // packed activation row bytes
  const std::size_t bsrow = inputs / 16U;
  const int m_owner = (lane & 1) ? (g + 8) : g;  // SFA: lane 4g -> row g, lane 4g+1 -> row g+8
  const float tensor = (tensor_scale != nullptr) ? *tensor_scale : 1.0f;

  // One M16N8K64 MMA covers all 8 columns natively: N=1 and N=8 issue the identical instruction.
  float d[4] = {0.f, 0.f, 0.f, 0.f};

  for (std::size_t k0 = 0; k0 < inputs; k0 += 64U) {
    const std::size_t byteoff = k0 / 2U;
    const std::size_t blk = k0 / 16U;
    std::uint32_t a[4];
    a[0] = ld_b32(packed + (row0 + g) * wrow + byteoff + 4U * q);
    a[1] = ld_b32(packed + (row0 + g + 8) * wrow + byteoff + 4U * q);
    a[2] = ld_b32(packed + (row0 + g) * wrow + byteoff + 16U + 4U * q);
    a[3] = ld_b32(packed + (row0 + g + 8) * wrow + byteoff + 16U + 4U * q);
    const std::uint32_t sa = ld_b32(scales + (row0 + m_owner) * srow + blk);
    const std::uint32_t sb = ld_b32(bscales + (std::size_t)((N == 1) ? 0 : g) * bsrow + blk);
    // B fragments: lane (g,q) owns column g, K values 8q..8q+7 (b0) and 32+8q.. (b1).
    const std::size_t bcol = (N == 1) ? 0U : static_cast<std::size_t>(g);
    const std::uint32_t b0 = ld_b32(bpacked + bcol * brow + byteoff + 4U * q);
    const std::uint32_t b1 = ld_b32(bpacked + bcol * brow + byteoff + 16U + 4U * q);
    const std::uint32_t bb[2] = {b0, b1};
    mma_mxf4(a, bb, sa, sb, d);
  }

  // Epilogue: C/D SM80_16x8_Row.  c0=(g,2q) c1=(g,2q+1) c2=(g+8,2q) c3=(g+8,2q+1).
  if constexpr (N == 1) {
    if (q == 0) {
      out[row0 + g] = d[0] * tensor;
      if (row0 + g + 8 < rows) out[row0 + g + 8] = d[2] * tensor;
    }
  } else {
    const int n0 = 2 * q;
    out[(row0 + g) * N + n0] = d[0] * tensor;
    out[(row0 + g) * N + n0 + 1] = d[1] * tensor;
    if (row0 + g + 8 < rows) {
      out[(row0 + g + 8) * N + n0] = d[2] * tensor;
      out[(row0 + g + 8) * N + n0 + 1] = d[3] * tensor;
    }
  }
}

// ---------------------------------------------------------------- Arm A2: MMA-native weight layout
// One-time repack so each lane's four A fragments are 16 contiguous bytes -> a single uint4 load.
__global__ void repack_weights(const std::uint8_t* packed, std::size_t rows, std::size_t inputs,
                               std::uint8_t* out) {
  const std::size_t kbars = inputs / 64U, tiles = rows / 16U;
  const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= tiles * kbars * 32U) return;
  const int lane = static_cast<int>(idx % 32U);
  const std::size_t kbar = (idx / 32U) % kbars;
  const std::size_t tile = idx / (32U * kbars);
  const int g = lane >> 2, q = lane & 3;
  const std::size_t row0 = tile * 16U, k0 = kbar * 64U, wrow = inputs / 2U;
  const std::uint32_t w[4] = {
      ld_b32(packed + (row0 + g) * wrow + k0 / 2U + 4U * q),
      ld_b32(packed + (row0 + g + 8) * wrow + k0 / 2U + 4U * q),
      ld_b32(packed + (row0 + g) * wrow + k0 / 2U + 16U + 4U * q),
      ld_b32(packed + (row0 + g + 8) * wrow + k0 / 2U + 16U + 4U * q)};
  *reinterpret_cast<uint4*>(out + idx * 16U) = make_uint4(w[0], w[1], w[2], w[3]);
}

template <int N>
__global__ void mma_gemv_repacked(const std::uint8_t* wmma, const std::uint8_t* scales,
                                  const std::uint8_t* bpacked, const std::uint8_t* bscales,
                                  const float* tensor_scale, float* out, std::size_t rows,
                                  std::size_t inputs) {
  const int lane = threadIdx.x & 31;
  const int warp = static_cast<int>((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
  const int g = lane >> 2, q = lane & 3;
  const std::size_t row0 = static_cast<std::size_t>(warp) * 16U;
  if (row0 >= rows) return;
  const std::size_t kbars = inputs / 64U;
  const std::size_t srow = inputs / 16U;
  const std::size_t brow = inputs / 2U, bsrow = inputs / 16U;
  const std::size_t tile = row0 / 16U;
  const int m_owner = (lane & 1) ? (g + 8) : g;
  const float tensor = (tensor_scale != nullptr) ? *tensor_scale : 1.0f;
  const std::size_t bcol = (N == 1) ? 0U : static_cast<std::size_t>(g);

  float d[4] = {0.f, 0.f, 0.f, 0.f};
  const std::size_t wbase = (tile * kbars) * 32U * 16U + static_cast<std::size_t>(lane) * 16U;
  for (std::size_t kbar = 0; kbar < kbars; ++kbar) {
    const uint4 av =
        *reinterpret_cast<const uint4*>(wmma + wbase + kbar * 32U * 16U);
    const std::uint32_t a[4] = {av.x, av.y, av.z, av.w};
    const std::size_t byteoff = kbar * 32U, blk = kbar * 4U;
    const std::uint32_t sa = ld_b32(scales + (row0 + m_owner) * srow + blk);
    const std::uint32_t sb = ld_b32(bscales + bcol * bsrow + blk);
    const std::uint32_t b0 = ld_b32(bpacked + bcol * brow + byteoff + 4U * q);
    const std::uint32_t b1 = ld_b32(bpacked + bcol * brow + byteoff + 16U + 4U * q);
    const std::uint32_t bb[2] = {b0, b1};
    mma_mxf4(a, bb, sa, sb, d);
  }
  if constexpr (N == 1) {
    if (q == 0) {
      out[row0 + g] = d[0] * tensor;
      if (row0 + g + 8 < rows) out[row0 + g + 8] = d[2] * tensor;
    }
  } else {
    const int n0 = 2 * q;
    out[(row0 + g) * N + n0] = d[0] * tensor;
    out[(row0 + g) * N + n0 + 1] = d[1] * tensor;
    if (row0 + g + 8 < rows) {
      out[(row0 + g + 8) * N + n0] = d[2] * tensor;
      out[(row0 + g + 8) * N + n0 + 1] = d[3] * tensor;
    }
  }
}

// ---------------------------------------------------------------- independent FP32 reference
// out[m*N+n] = tensor * sum_k decode(weight) * activation_dequant[k*N+n]
__global__ void ref_gemv(const std::uint8_t* packed, const std::uint8_t* scales,
                         const float* act, const float* tensor_scale, float* out, std::size_t rows,
                         std::size_t inputs, int N) {
  const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= rows * static_cast<std::size_t>(N)) return;
  const std::size_t m = idx / N, n = idx % N;
  const std::uint8_t* wr = packed + m * (inputs / 2U);
  const std::uint8_t* sr = scales + m * (inputs / 16U);
  double sum = 0.0;
  for (std::size_t k = 0; k < inputs; ++k) {
    const std::uint8_t byte = wr[k / 2U];
    const std::uint8_t code = (k % 2U == 0) ? (byte & 0xFU) : (byte >> 4U);
    const double w = static_cast<double>(decode_e2m1(code)) *
                     static_cast<double>(decode_e4m3(sr[k / 16U]));
    sum += w * static_cast<double>(act[k * N + n]);
  }
  out[idx] = static_cast<float>(sum * ((tensor_scale != nullptr) ? *tensor_scale : 1.0f));
}

// ---------------------------------------------------------------- pure weight streaming
__global__ void stream_weights(const std::uint8_t* packed, float* sink, std::size_t bytes) {
  const std::size_t i = (static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x) * 4U;
  if (i + 4 <= bytes) {
    const std::uint32_t v = ld_b32(packed + i);
    if (v == 0xDEADBEEFU) sink[0] = 1.f;
  }
}

// ---------------------------------------------------------------- harness
// The projection census is DERIVED from the produced Physical Plan by
// `tools/qwen38_nvfp4_census.py --emit-header`; it is never hand-maintained. The static_assert makes a
// stale or partial census a hard failure instead of a silent under-measurement.
#include "qwen38_nvfp4_census_generated.h"

namespace census = superinfer::sm120::qwen38_census;
static_assert(census::kTotalNvfp4Launches == 401,
              "regenerate qwen38_nvfp4_census_generated.h from the accepted artifact plan dump");

// P7 residual (non-NVFP4) GPU time per decoded token, from the P7 e2e/profile evidence:
//   120.8 ms/token total GPU (7.25 s / 60 tokens)  -  76.5 ms/token weighted software NVFP4.
// This is a PREDICTION input only; it is superseded by integrated measurement.
inline constexpr double kP7NonNvfp4ResidualMsPerToken = 44.3;

static float time_kernel(int iters, std::function<void()> run) {
  cudaEvent_t a, b;
  cudaEventCreate(&a);
  cudaEventCreate(&b);
  cudaEventRecord(a);
  for (int i = 0; i < iters; ++i) run();
  cudaEventRecord(b);
  cudaEventSynchronize(b);
  float ms = 0.f;
  cudaEventElapsedTime(&ms, a, b);
  cudaEventDestroy(a);
  cudaEventDestroy(b);
  return ms / iters;
}

int main() {
  if (cudaSetDevice(0) != cudaSuccess) { std::printf("no GPU\n"); return 77; }
  cudaDeviceProp prop{};
  cudaGetDeviceProperties(&prop, 0);
  std::printf("device: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);

  double weighted_n1 = 0.0, weighted_n8 = 0.0, weighted_quant = 0.0, weighted_weights = 0.0;
  double weighted_a2n1 = 0.0, weighted_a2n8 = 0.0, repack_once_ms = 0.0;
  std::size_t census_total = 0;
  for (const auto& cls : census::kProjectionClasses) census_total += static_cast<std::size_t>(cls.count);
  assert(census_total == census::kTotalNvfp4Launches);
  std::printf("\ncensus: %zu classes, %zu nvfp4_linear launches/token (generated header)\n",
              census::kProjectionClasses.size(), census_total);
  std::printf("%-14s %6s %8s %7s %7s %9s %9s %9s %9s %9s\n", "shape", "count", "wt(MB)", "quant",
              "stream", "ArmA(N1)", "ArmB/8", "A2(N1)", "A2(N8)/8", "relA/B");
  for (const auto& shape : census::kProjectionClasses) {
    const std::size_t bytes = shape.rows * (shape.inputs / 2U);
    const std::size_t sbytes = shape.rows * (shape.inputs / 16U);

    std::uint8_t *d_packed, *d_scales, *d_bpacked, *d_bscales;
    float *d_in, *d_act, *d_out, *d_ref, *d_ref2, *d_sink, *d_tensor;
    assert(cudaMalloc(&d_packed, bytes) == cudaSuccess);
    assert(cudaMalloc(&d_scales, sbytes) == cudaSuccess);
    assert(cudaMalloc(&d_bpacked, 8 * (shape.inputs / 2U)) == cudaSuccess);
    assert(cudaMalloc(&d_bscales, 8 * (shape.inputs / 16U)) == cudaSuccess);
    assert(cudaMalloc(&d_in, 8 * shape.inputs * sizeof(float)) == cudaSuccess);
    assert(cudaMalloc(&d_act, 8 * shape.inputs * sizeof(float)) == cudaSuccess);
    assert(cudaMalloc(&d_out, shape.rows * 8 * sizeof(float)) == cudaSuccess);
    assert(cudaMalloc(&d_ref, shape.rows * 8 * sizeof(float)) == cudaSuccess);
    assert(cudaMalloc(&d_ref2, shape.rows * 8 * sizeof(float)) == cudaSuccess);
    assert(cudaMalloc(&d_sink, sizeof(float)) == cudaSuccess);
    assert(cudaMalloc(&d_tensor, sizeof(float)) == cudaSuccess);

    std::vector<std::uint8_t> h_packed(bytes), h_scales(sbytes);
    std::uint64_t st = 0x1234ULL + shape.rows * 131U + shape.inputs;
    auto nxt = [&]() {
      st = st * 6364136223846793005ULL + 1442695040888963407ULL;
      return static_cast<std::uint32_t>(st >> 33);
    };
    for (auto& v : h_packed) v = static_cast<std::uint8_t>(nxt() & 0xFFU);
    for (auto& v : h_scales) v = static_cast<std::uint8_t>(0x30U | (nxt() & 0x0FU));
    std::vector<float> h_in(8 * shape.inputs);
    for (auto& v : h_in) v = (static_cast<float>(static_cast<int>(nxt() % 2001) - 1000) / 250.0f);
    const float tensor = 1.0f;

    cudaMemcpy(d_packed, h_packed.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_scales, h_scales.data(), sbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_in, h_in.data(), h_in.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tensor, &tensor, sizeof(float), cudaMemcpyHostToDevice);

    // activation quantisation for all 8 columns
    const std::size_t threads = 256;
    quantize_activation<<<(1U * shape.inputs / 16U + threads - 1) / threads, threads>>>(
        d_in, 1U * shape.inputs, d_bpacked, d_bscales, d_act);
    for (int n = 1; n < 8; ++n)
      quantize_activation<<<(shape.inputs / 16U + threads - 1) / threads, threads>>>(
          d_in + n * shape.inputs, shape.inputs, d_bpacked + n * (shape.inputs / 2U),
          d_bscales + n * (shape.inputs / 16U), d_act + n * shape.inputs);
    cudaDeviceSynchronize();

    const int warps = static_cast<int>((shape.rows + 15U) / 16U);
    const int block = 256;
    const int grid = (warps * 32 + block - 1) / block;

    const float ms_stream =
        time_kernel(10, [&] { stream_weights<<<grid, block>>>(d_packed, d_sink, bytes); });
    const float ms_quant = time_kernel(10, [&] {
      quantize_activation<<<(shape.inputs / 16U + threads - 1) / threads, threads>>>(
          d_in, shape.inputs, d_bpacked, d_bscales, d_act);
    });
    const float ms_n1 = time_kernel(10, [&] {
      mma_gemv<1><<<grid, block>>>(d_packed, d_scales, d_bpacked, d_bscales, d_tensor, d_out,
                                   shape.rows, shape.inputs);
    });
    const float ms_n8 = time_kernel(10, [&] {
      mma_gemv<8><<<grid, block>>>(d_packed, d_scales, d_bpacked, d_bscales, d_tensor, d_out,
                                   shape.rows, shape.inputs);
    });
    // Arm A2: repacked MMA-native layout (one-time repack, amortised)
    std::uint8_t* d_wmma;
    assert(cudaMalloc(&d_wmma, bytes) == cudaSuccess);
    const std::size_t rp_threads = (shape.rows / 16U) * (shape.inputs / 64U) * 32U;
    const int rp_grid = static_cast<int>((rp_threads + block - 1) / block);
    repack_weights<<<rp_grid, block>>>(d_packed, shape.rows, shape.inputs, d_wmma);
    cudaDeviceSynchronize();
    const float ms_repack = time_kernel(
        3, [&] { repack_weights<<<rp_grid, block>>>(d_packed, shape.rows, shape.inputs, d_wmma); });
    const float ms_a2_n1 = time_kernel(10, [&] {
      mma_gemv_repacked<1><<<grid, block>>>(d_wmma, d_scales, d_bpacked, d_bscales, d_tensor, d_out,
                                            shape.rows, shape.inputs);
    });
    const float ms_a2_n8 = time_kernel(10, [&] {
      mma_gemv_repacked<8><<<grid, block>>>(d_wmma, d_scales, d_bpacked, d_bscales, d_tensor, d_out,
                                            shape.rows, shape.inputs);
    });

    // correctness: Arm A (N=1) vs independent FP32 reference using the SAME quantised activation
    mma_gemv<1><<<grid, block>>>(d_packed, d_scales, d_bpacked, d_bscales, d_tensor, d_out,
                                 shape.rows, shape.inputs);
    ref_gemv<<<(shape.rows + 255U) / 256U, 256>>>(d_packed, d_scales, d_act, d_tensor, d_ref,
                                                  shape.rows, shape.inputs, 1);
    cudaDeviceSynchronize();
    std::vector<float> h_out(shape.rows), h_ref(shape.rows);
    cudaMemcpy(h_out.data(), d_out, shape.rows * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ref.data(), d_ref, shape.rows * sizeof(float), cudaMemcpyDeviceToHost);
    double max_err = 0, max_mag = 0;
    for (std::size_t m = 0; m < shape.rows; ++m) {
      max_err = std::max(max_err, std::fabs((double)h_out[m] - h_ref[m]));
      max_mag = std::max(max_mag, std::fabs((double)h_ref[m]));
    }
    const double rel = max_mag > 0 ? max_err / max_mag : 0.0;

    // Projection-level A-vs-B signal: native (quantized activation) vs P7-equivalent (FP32 activation).
    ref_gemv<<<(shape.rows + 255U) / 256U, 256>>>(d_packed, d_scales, d_in, d_tensor, d_ref2,
                                                  shape.rows, shape.inputs, 1);
    cudaDeviceSynchronize();
    std::vector<float> h_ref2(shape.rows);
    cudaMemcpy(h_ref2.data(), d_ref2, shape.rows * sizeof(float), cudaMemcpyDeviceToHost);
    double se = 0, sref = 0, mrel = 0;
    for (std::size_t m = 0; m < shape.rows; ++m) {
      const double nv = h_out[m], rv = h_ref2[m];
      se += (nv - rv) * (nv - rv);
      sref += rv * rv;
      const double denom = std::fabs(rv) > 1e-6 ? std::fabs(rv) : 1e-6;
      mrel = std::max(mrel, std::fabs(nv - rv) / denom);
    }
    if (true)
      std::printf("  actquant effect %s: rel-L2=%.4f  max-rel=%.4f\n", shape.name,
                  sref > 0 ? std::sqrt(se / sref) : 0.0, mrel);
    mma_gemv_repacked<1><<<grid, block>>>(d_wmma, d_scales, d_bpacked, d_bscales, d_tensor, d_out,
                                          shape.rows, shape.inputs);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out.data(), d_out, shape.rows * sizeof(float), cudaMemcpyDeviceToHost);
    double a2_err = 0;
    for (std::size_t m = 0; m < shape.rows; ++m)
      a2_err = std::max(a2_err, std::fabs((double)h_out[m] - h_ref[m]));
    if (a2_err != 0.0) std::printf("  WARNING A2 mismatch %.3e\n", a2_err);
    mma_gemv<1><<<grid, block>>>(d_packed, d_scales, d_bpacked, d_bscales, d_tensor, d_out,
                                 shape.rows, shape.inputs);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out.data(), d_out, shape.rows * sizeof(float), cudaMemcpyDeviceToHost);

    if (std::string_view(shape.name) == "1024x5120") {
      std::printf("  DEBUG shape=%s out[0..3]=", shape.name);
      for (int i = 0; i < 4; ++i) std::printf("%.4f ", h_out[i]);
      std::printf(" ref[0..3]=");
      for (int i = 0; i < 4; ++i) std::printf("%.4f ", h_ref[i]);
      std::printf("\n");
      std::printf("  DEBUG act q[0..7]=");
      std::vector<float> ha(16);
      cudaMemcpy(ha.data(), d_act, 16 * sizeof(float), cudaMemcpyDeviceToHost);
      for (int i = 0; i < 8; ++i) std::printf("%.4f ", ha[i]);
      std::printf("  in[0..7]=");
      for (int i = 0; i < 8; ++i) std::printf("%.4f ", h_in[i]);
      std::printf("\n");
    }

    std::printf("%-14s %6llu %8.1f %7.3f %7.3f %9.3f %9.3f %9.3f %9.3f %9.2e\n", shape.name,
                (unsigned long long)shape.count, bytes / 1e6, ms_quant, ms_stream, ms_n1,
                ms_n8 / 8.0, ms_a2_n1, ms_a2_n8 / 8.0, rel);

    const double mult = static_cast<double>(shape.count);
    weighted_n1 += mult * ms_n1;
    weighted_n8 += mult * (ms_n8 / 8.0);
    weighted_a2n1 += mult * ms_a2_n1;
    weighted_a2n8 += mult * (ms_a2_n8 / 8.0);
    weighted_quant += mult * ms_quant;
    weighted_weights += mult * ms_stream;
    repack_once_ms += ms_repack;

    cudaFree(d_packed); cudaFree(d_scales); cudaFree(d_bpacked); cudaFree(d_bscales);
    cudaFree(d_in); cudaFree(d_act); cudaFree(d_out); cudaFree(d_ref); cudaFree(d_sink);
    cudaFree(d_tensor); cudaFree(d_wmma); cudaFree(d_ref2);
  }
  // Terminally distinct accounting. ArmA/A2 are MMA-path timings only; they do NOT include the
  // separately measured activation-quantisation launches.
  const double mma_ms_per_token = weighted_n1;
  const double activation_quant_ms_per_token = weighted_quant;
  const double native_unfused_total = mma_ms_per_token + activation_quant_ms_per_token;
  const double native_repacked_unfused = weighted_a2n1 + activation_quant_ms_per_token;

  std::printf("\n--- NVFP4 projection subsystem, per decoded token (ms) ---\n");
  std::printf("  mma_ms_per_token                       = %.3f   (Arm A, natural .sinf layout)\n",
              mma_ms_per_token);
  std::printf("  activation_quant_ms_per_token          = %.3f   (321-401 explicit quantise launches)\n",
              activation_quant_ms_per_token);
  std::printf("  native_unfused_total_ms_per_token      = %.3f   (mma + activation_quant)\n",
              native_unfused_total);
  std::printf("  repack_cost_once_ms                    = %.3f   (one-time, not per token)\n",
              repack_once_ms);
  std::printf("  native_repacked_unfused_ms_per_token   = %.3f   (A2 mma + activation_quant)\n\n",
              native_repacked_unfused);

  std::printf("PROJECTION SUBSYSTEM throughput (NOT model tok/s): native_unfused = %.1f proj-tok/s\n",
              native_unfused_total > 0 ? 1000.0 / native_unfused_total : 0.0);
  std::printf(
      "PREDICTION ONLY whole-model bound = projection subsystem + P7 non-NVFP4 residual %.1f ms:\n"
      "  native_unfused  => %.1f ms/token => <= %.1f model tok/s\n"
      "  repacked        => %.1f ms/token => <= %.1f model tok/s\n"
      "  (software P7 reference: %.1f ms NVFP4 + %.1f residual = %.1f ms/token => %.1f model tok/s)\n",
      kP7NonNvfp4ResidualMsPerToken, native_unfused_total + kP7NonNvfp4ResidualMsPerToken,
      1000.0 / (native_unfused_total + kP7NonNvfp4ResidualMsPerToken),
      native_repacked_unfused + kP7NonNvfp4ResidualMsPerToken,
      1000.0 / (native_repacked_unfused + kP7NonNvfp4ResidualMsPerToken), 76.5,
      kP7NonNvfp4ResidualMsPerToken, 76.5 + kP7NonNvfp4ResidualMsPerToken,
      1000.0 / (76.5 + kP7NonNvfp4ResidualMsPerToken));
  std::printf(
      "\nArm B (N=8, batched/speculative) projection throughput: %.1f proj-tok/s (%.1f repacked); "
      "PREDICTION ONLY, assumes all 8 columns accepted.\n",
      weighted_n8 > 0 ? 1000.0 / weighted_n8 : 0.0,
      weighted_a2n8 > 0 ? 1000.0 / weighted_a2n8 : 0.0);
  return 0;
}
