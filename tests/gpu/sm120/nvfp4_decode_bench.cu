// P7 decode-cost decomposition and hardware-E2M1 prototype benchmark.
//
// Stages measured per real Qwen3.8 NVFP4 shape:
//   A  raw packed+scale streaming (memory floor)
//   B  software E2M1 decode only (no scale, no multiply)
//   C  E2M1 decode + scale + multiply, no cross-lane reduction
//   D  full software warp-per-row GEMV (P6 incumbent)
//   E  hardware E2M1 decode (cvt.rn.f16x2.e2m1x2) + predecoded FP16 scales
//   E2/E4  E with 2/4 warps per output row (P7-4 mapping sweep)
//
// Also proves exhaustively that the hardware E2M1 conversion equals the
// software decode for all 256 packed bytes.

#include <sm120/runtime/cuda_plan_executor.cuh>

#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include <cuda_fp16.h>
#include <cstdint>

namespace {

using superinfer::sm120::cuda_runtime::detail::decode_e2m1_device;
using superinfer::sm120::cuda_runtime::detail::decode_e4m3fn_device;

__device__ __forceinline__ float2 e2m1x2_hw(std::uint8_t packed) {
  unsigned int storage;
  unsigned short tmp = static_cast<unsigned short>(packed);
  asm("{ .reg .b8 __$t, __$z;                 \n"
      " mov.b16 {__$t, __$z}, %1;             \n"
      " cvt.rn.f16x2.e2m1x2 %0, __$t;         }\n"
      : "=r"(storage)
      : "h"(tmp));
  __half2 h = *reinterpret_cast<__half2*>(&storage);
  return make_float2(__half2float(__low2half(h)), __half2float(__high2half(h)));
}

struct Shape {
  const char* name;
  std::size_t rows;
  std::size_t inputs;
};

constexpr std::array<Shape, 6> kShapes{{
    {"lm_head_248320x5120", 248320, 5120},
    {"mlp_17408x5120", 17408, 5120},
    {"gdn_10240x5120", 10240, 5120},
    {"attn_6144x5120", 6144, 5120},
    {"down_5120x17408", 5120, 17408},
    {"small_1024x5120", 1024, 5120},
}};

// Real per-token multiplicities for weighting.
constexpr std::array<int, 6> kMultiplicity{{1, 128, 48, 48, 64, 32}};

std::uint64_t g_state = 0x243F6A8885A308D3ULL;
std::uint32_t next_word() {
  g_state = g_state * 6364136223846793005ULL + 1442695040888963407ULL;
  return static_cast<std::uint32_t>(g_state >> 33U);
}

__global__ void e2m1_check_kernel(std::uint8_t* bad, float* maxerr) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= 256) return;
  const auto packed = static_cast<std::uint8_t>(i);
  const float2 hw = e2m1x2_hw(packed);
  if (hw.x != decode_e2m1_device(packed & 0x0FU) ||
      hw.y != decode_e2m1_device(packed >> 4U)) {
    bad[0] = packed;
  }
  maxerr[0] = fmaxf(maxerr[0], fabsf(hw.x) + fabsf(hw.y));
}

// Host-side E4M3 decode matching detail::decode_e4m3fn_device for positive scales.
float decode_e4m3fn_host(std::uint8_t code) {
  const bool negative = (code & 0x80U) != 0;
  const std::uint8_t exponent = (code >> 3U) & 0x0FU;
  const std::uint8_t mantissa = code & 0x07U;
  if (negative || (exponent == 0x0FU && mantissa == 0x07U)) return std::nanf("");
  if (exponent == 0) return std::ldexp(static_cast<float>(mantissa) / 8.0F, -6);
  return std::ldexp(1.0F + static_cast<float>(mantissa) / 8.0F,
                    static_cast<int>(exponent) - 7);
}

__global__ void stream_kernel(const std::uint8_t* packed, const std::uint8_t* scales,
                              float* out, std::size_t packed_bytes, std::size_t scale_bytes) {
  std::uint32_t acc = 0;
  for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < packed_bytes / 16U;
       i += blockDim.x * gridDim.x) {
    const uint4 w = *reinterpret_cast<const uint4*>(packed + i * 16U);
    acc ^= w.x ^ w.y ^ w.z ^ w.w;
  }
  for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < scale_bytes;
       i += blockDim.x * gridDim.x) {
    acc += scales[i];
  }
  if (acc == 0xFFFFFFFFU) out[0] = 1.0F;
}

__global__ void decode_sw_kernel(const std::uint8_t* packed, float* out,
                                 std::size_t rows, std::size_t packed_row_bytes) {
  const std::size_t warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32U;
  const std::size_t lane = threadIdx.x % 32U;
  if (warp >= rows) return;
  const std::uint8_t* row = packed + warp * packed_row_bytes;
  float sum = 0.0F;
  const std::size_t groups = packed_row_bytes / 16U;
  for (std::size_t g = lane; g < groups; g += 32U) {
    const uint4 w = *reinterpret_cast<const uint4*>(row + g * 16U);
    const std::uint8_t bytes[16] = {
        static_cast<std::uint8_t>(w.x), static_cast<std::uint8_t>(w.x >> 8U),
        static_cast<std::uint8_t>(w.x >> 16U), static_cast<std::uint8_t>(w.x >> 24U),
        static_cast<std::uint8_t>(w.y), static_cast<std::uint8_t>(w.y >> 8U),
        static_cast<std::uint8_t>(w.y >> 16U), static_cast<std::uint8_t>(w.y >> 24U),
        static_cast<std::uint8_t>(w.z), static_cast<std::uint8_t>(w.z >> 8U),
        static_cast<std::uint8_t>(w.z >> 16U), static_cast<std::uint8_t>(w.z >> 24U),
        static_cast<std::uint8_t>(w.w), static_cast<std::uint8_t>(w.w >> 8U),
        static_cast<std::uint8_t>(w.w >> 16U), static_cast<std::uint8_t>(w.w >> 24U)};
    for (std::size_t b = 0; b < 16U; ++b) {
      sum += decode_e2m1_device(bytes[b] & 0x0FU);
      sum += decode_e2m1_device(bytes[b] >> 4U);
    }
  }
  if ((warp & 0U) == 0U && lane == 0U) out[warp] = sum;
}

// E2M1 decode + scale + multiply, no cross-lane reduction.
__global__ void decode_mul_kernel(const float* input, const std::uint8_t* packed,
                                  const std::uint8_t* scales, const float* tensor_scale,
                                  float* out, std::size_t rows, std::size_t inputs) {
  const std::size_t warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32U;
  const std::size_t lane = threadIdx.x % 32U;
  if (warp >= rows) return;
  const float tensor = *tensor_scale;
  const std::uint8_t* row = packed + warp * (inputs / 2U);
  const std::uint8_t* scale_row = scales + warp * (inputs / 16U);
  const std::size_t chunk = ((inputs + 31U) / 32U + 31U) / 32U * 32U;
  const std::size_t begin = lane * chunk;
  const std::size_t end = begin < inputs ? (begin + chunk < inputs ? begin + chunk : inputs) : inputs;
  float sum = 0.0F;
  for (std::size_t column = begin; column + 32U <= end; column += 32U) {
    const std::size_t group = column / 32U;
    const uint4 w = *reinterpret_cast<const uint4*>(row + group * 16U);
    const std::uint8_t bytes[16] = {
        static_cast<std::uint8_t>(w.x), static_cast<std::uint8_t>(w.x >> 8U),
        static_cast<std::uint8_t>(w.x >> 16U), static_cast<std::uint8_t>(w.x >> 24U),
        static_cast<std::uint8_t>(w.y), static_cast<std::uint8_t>(w.y >> 8U),
        static_cast<std::uint8_t>(w.y >> 16U), static_cast<std::uint8_t>(w.y >> 24U),
        static_cast<std::uint8_t>(w.z), static_cast<std::uint8_t>(w.z >> 8U),
        static_cast<std::uint8_t>(w.z >> 16U), static_cast<std::uint8_t>(w.z >> 24U),
        static_cast<std::uint8_t>(w.w), static_cast<std::uint8_t>(w.w >> 8U),
        static_cast<std::uint8_t>(w.w >> 16U), static_cast<std::uint8_t>(w.w >> 24U)};
    const float s0 = decode_e4m3fn_device(scale_row[group * 2U]);
    const float s1 = decode_e4m3fn_device(scale_row[group * 2U + 1U]);
    for (std::size_t b = 0; b < 8U; ++b) {
      sum += (decode_e2m1_device(bytes[b] & 0x0FU) * s0 * tensor) * input[column + b * 2U];
      sum += (decode_e2m1_device(bytes[b] >> 4U) * s0 * tensor) * input[column + b * 2U + 1U];
    }
    for (std::size_t b = 8U; b < 16U; ++b) {
      sum += (decode_e2m1_device(bytes[b] & 0x0FU) * s1 * tensor) * input[column + b * 2U];
      sum += (decode_e2m1_device(bytes[b] >> 4U) * s1 * tensor) * input[column + b * 2U + 1U];
    }
  }
  if (lane == 0U) out[warp] = sum;
}

// Hardware E2M1 decode + predecoded FP16 scales, `warps_per_row` warps per row.
// Each warp owns a contiguous column slice; partial sums are combined in a fixed
// order by warp 0 (deterministic across runs).
template <int kWarpsPerRow>
__global__ void full_warp_hw_kernel(const float* input, const std::uint8_t* packed,
                                    const std::uint16_t* scales_fp16, const float* tensor_scale,
                                    float* out, std::size_t rows, std::size_t inputs) {
  const std::size_t warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32U;
  const std::size_t lane = threadIdx.x % 32U;
  const std::size_t row = warp / kWarpsPerRow;
  const std::size_t sub = warp % kWarpsPerRow;
  if (row >= rows) return;
  const float tensor = *tensor_scale;
  const std::uint8_t* row_packed = packed + row * (inputs / 2U);
  const std::uint16_t* row_scale = scales_fp16 + row * (inputs / 16U);
  const std::size_t columns_per_sub = ((inputs + kWarpsPerRow - 1) / kWarpsPerRow + 31U) / 32U * 32U;
  const std::size_t sub_begin = sub * columns_per_sub;
  const std::size_t sub_end = sub_begin < inputs
                                  ? (sub_begin + columns_per_sub < inputs ? sub_begin + columns_per_sub
                                                                          : inputs)
                                  : inputs;
  const std::size_t chunk = ((sub_end - sub_begin + 31U) / 32U + 31U) / 32U * 32U;
  const std::size_t begin = sub_begin + lane * chunk;
  const std::size_t end = begin < sub_end ? (begin + chunk < sub_end ? begin + chunk : sub_end)
                                          : sub_end;
  float sum = 0.0F;
  for (std::size_t column = begin; column + 32U <= end; column += 32U) {
    const std::size_t group = column / 32U;
    const uint4 w = *reinterpret_cast<const uint4*>(row_packed + group * 16U);
    const std::uint8_t bytes[16] = {
        static_cast<std::uint8_t>(w.x), static_cast<std::uint8_t>(w.x >> 8U),
        static_cast<std::uint8_t>(w.x >> 16U), static_cast<std::uint8_t>(w.x >> 24U),
        static_cast<std::uint8_t>(w.y), static_cast<std::uint8_t>(w.y >> 8U),
        static_cast<std::uint8_t>(w.y >> 16U), static_cast<std::uint8_t>(w.y >> 24U),
        static_cast<std::uint8_t>(w.z), static_cast<std::uint8_t>(w.z >> 8U),
        static_cast<std::uint8_t>(w.z >> 16U), static_cast<std::uint8_t>(w.z >> 24U),
        static_cast<std::uint8_t>(w.w), static_cast<std::uint8_t>(w.w >> 8U),
        static_cast<std::uint8_t>(w.w >> 16U), static_cast<std::uint8_t>(w.w >> 24U)};
    const float s0 = __half2float(*reinterpret_cast<const __half*>(&row_scale[group * 2U]));
    const float s1 = __half2float(*reinterpret_cast<const __half*>(&row_scale[group * 2U + 1U]));
    for (std::size_t b = 0; b < 8U; ++b) {
      const float2 v0 = e2m1x2_hw(static_cast<std::uint8_t>(bytes[b] & 0x0FU) |
                                  static_cast<std::uint8_t>((bytes[b] >> 4U) << 4U));
      sum += (v0.x * s0 * tensor) * input[column + b * 2U];
      sum += (v0.y * s0 * tensor) * input[column + b * 2U + 1U];
    }
    for (std::size_t b = 8U; b < 16U; ++b) {
      const float2 v1 = e2m1x2_hw(static_cast<std::uint8_t>(bytes[b] & 0x0FU) |
                                  static_cast<std::uint8_t>((bytes[b] >> 4U) << 4U));
      sum += (v1.x * s1 * tensor) * input[column + b * 2U];
      sum += (v1.y * s1 * tensor) * input[column + b * 2U + 1U];
    }
  }
  for (std::size_t offset = 16U; offset > 0U; offset >>= 1U) {
    sum += __shfl_down_sync(0xFFFFFFFFU, sum, offset);
  }
  if (lane == 0U) out[row * kWarpsPerRow + sub] = sum;
}

// Lane-interleaved warp-per-row GEMV: lane l processes column groups
// l, l+32, l+64, ... so packed-weight and input accesses are coalesced across
// the warp. Hardware E2M1 decode + predecoded FP16 scales.
__global__ void full_warp_interleaved_hw_kernel(const float* input, const std::uint8_t* packed,
                                                const std::uint16_t* scales_fp16,
                                                const float* tensor_scale, float* out,
                                                std::size_t rows, std::size_t inputs) {
  const std::size_t warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32U;
  const std::size_t lane = threadIdx.x % 32U;
  if (warp >= rows) return;
  const float tensor = *tensor_scale;
  const std::uint8_t* row_packed = packed + warp * (inputs / 2U);
  const std::uint16_t* row_scale = scales_fp16 + warp * (inputs / 16U);
  const std::size_t groups = inputs / 32U;
  float sum = 0.0F;
  for (std::size_t group = lane; group < groups; group += 32U) {
    const uint4 w = *reinterpret_cast<const uint4*>(row_packed + group * 16U);
    const std::uint8_t bytes[16] = {
        static_cast<std::uint8_t>(w.x), static_cast<std::uint8_t>(w.x >> 8U),
        static_cast<std::uint8_t>(w.x >> 16U), static_cast<std::uint8_t>(w.x >> 24U),
        static_cast<std::uint8_t>(w.y), static_cast<std::uint8_t>(w.y >> 8U),
        static_cast<std::uint8_t>(w.y >> 16U), static_cast<std::uint8_t>(w.y >> 24U),
        static_cast<std::uint8_t>(w.z), static_cast<std::uint8_t>(w.z >> 8U),
        static_cast<std::uint8_t>(w.z >> 16U), static_cast<std::uint8_t>(w.z >> 24U),
        static_cast<std::uint8_t>(w.w), static_cast<std::uint8_t>(w.w >> 8U),
        static_cast<std::uint8_t>(w.w >> 16U), static_cast<std::uint8_t>(w.w >> 24U)};
    const float s0 = __half2float(*reinterpret_cast<const __half*>(&row_scale[group * 2U]));
    const float s1 = __half2float(*reinterpret_cast<const __half*>(&row_scale[group * 2U + 1U]));
    for (std::size_t b = 0; b < 8U; ++b) {
      const float2 v = e2m1x2_hw(bytes[b]);
      sum += (v.x * s0 * tensor) * input[group * 32U + b * 2U];
      sum += (v.y * s0 * tensor) * input[group * 32U + b * 2U + 1U];
    }
    for (std::size_t b = 8U; b < 16U; ++b) {
      const float2 v = e2m1x2_hw(bytes[b]);
      sum += (v.x * s1 * tensor) * input[group * 32U + b * 2U];
      sum += (v.y * s1 * tensor) * input[group * 32U + b * 2U + 1U];
    }
  }
  for (std::size_t offset = 16U; offset > 0U; offset >>= 1U) {
    sum += __shfl_down_sync(0xFFFFFFFFU, sum, offset);
  }
  if (lane == 0U) out[warp] = sum;
}

// C_noload: E2M1 decode + scale + tensor multiply, NO input load.
__global__ void decode_noload_kernel(const std::uint8_t* packed, const std::uint8_t* scales,
                                     const float* tensor_scale, float* out, std::size_t rows,
                                     std::size_t inputs) {
  const std::size_t warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32U;
  const std::size_t lane = threadIdx.x % 32U;
  if (warp >= rows) return;
  const float tensor = *tensor_scale;
  const std::uint8_t* row = packed + warp * (inputs / 2U);
  const std::uint8_t* scale_row = scales + warp * (inputs / 16U);
  const std::size_t groups = inputs / 32U;
  float sum = 0.0F;
  for (std::size_t group = lane; group < groups; group += 32U) {
    const uint4 w = *reinterpret_cast<const uint4*>(row + group * 16U);
    const std::uint8_t bytes[16] = {
        static_cast<std::uint8_t>(w.x), static_cast<std::uint8_t>(w.x >> 8U),
        static_cast<std::uint8_t>(w.x >> 16U), static_cast<std::uint8_t>(w.x >> 24U),
        static_cast<std::uint8_t>(w.y), static_cast<std::uint8_t>(w.y >> 8U),
        static_cast<std::uint8_t>(w.y >> 16U), static_cast<std::uint8_t>(w.y >> 24U),
        static_cast<std::uint8_t>(w.z), static_cast<std::uint8_t>(w.z >> 8U),
        static_cast<std::uint8_t>(w.z >> 16U), static_cast<std::uint8_t>(w.z >> 24U),
        static_cast<std::uint8_t>(w.w), static_cast<std::uint8_t>(w.w >> 8U),
        static_cast<std::uint8_t>(w.w >> 16U), static_cast<std::uint8_t>(w.w >> 24U)};
    const float s0 = decode_e4m3fn_device(scale_row[group * 2U]);
    const float s1 = decode_e4m3fn_device(scale_row[group * 2U + 1U]);
    for (std::size_t b = 0; b < 16U; ++b) {
      const float s = b < 8U ? s0 : s1;
      sum += (decode_e2m1_device(bytes[b] & 0x0FU) * s * tensor);
      sum += (decode_e2m1_device(bytes[b] >> 4U) * s * tensor);
    }
  }
  for (std::size_t offset = 16U; offset > 0U; offset >>= 1U) {
    sum += __shfl_down_sync(0xFFFFFFFFU, sum, offset);
  }
  if (lane == 0U) out[warp] = sum;
}

// C_mul1: E2M1 decode + input load + one multiply (no scale/tensor).
__global__ void decode_mul1_kernel(const float* input, const std::uint8_t* packed, float* out,
                                   std::size_t rows, std::size_t inputs) {
  const std::size_t warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32U;
  const std::size_t lane = threadIdx.x % 32U;
  if (warp >= rows) return;
  const std::uint8_t* row = packed + warp * (inputs / 2U);
  const std::size_t groups = inputs / 32U;
  float sum = 0.0F;
  for (std::size_t group = lane; group < groups; group += 32U) {
    const uint4 w = *reinterpret_cast<const uint4*>(row + group * 16U);
    const std::uint8_t bytes[16] = {
        static_cast<std::uint8_t>(w.x), static_cast<std::uint8_t>(w.x >> 8U),
        static_cast<std::uint8_t>(w.x >> 16U), static_cast<std::uint8_t>(w.x >> 24U),
        static_cast<std::uint8_t>(w.y), static_cast<std::uint8_t>(w.y >> 8U),
        static_cast<std::uint8_t>(w.y >> 16U), static_cast<std::uint8_t>(w.y >> 24U),
        static_cast<std::uint8_t>(w.z), static_cast<std::uint8_t>(w.z >> 8U),
        static_cast<std::uint8_t>(w.z >> 16U), static_cast<std::uint8_t>(w.z >> 24U),
        static_cast<std::uint8_t>(w.w), static_cast<std::uint8_t>(w.w >> 8U),
        static_cast<std::uint8_t>(w.w >> 16U), static_cast<std::uint8_t>(w.w >> 24U)};
    for (std::size_t b = 0; b < 16U; ++b) {
      sum += decode_e2m1_device(bytes[b] & 0x0FU) * input[group * 32U + b * 2U];
      sum += decode_e2m1_device(bytes[b] >> 4U) * input[group * 32U + b * 2U + 1U];
    }
  }
  for (std::size_t offset = 16U; offset > 0U; offset >>= 1U) {
    sum += __shfl_down_sync(0xFFFFFFFFU, sum, offset);
  }
  if (lane == 0U) out[warp] = sum;
}

__global__ void combine_sums(float* partials, float* out, std::size_t rows, int warps_per_row) {
  const std::size_t row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows) return;
  float sum = 0.0F;
  for (int i = 0; i < warps_per_row; ++i) sum += partials[row * warps_per_row + i];
  out[row] = sum;
}

template <typename F>
double benchmark(F&& launch, int iterations) {
  launch();
  cudaDeviceSynchronize();
  cudaEvent_t start{}, stop{};
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  for (int i = 0; i < iterations; ++i) launch();
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float ms = 0.0F;
  cudaEventElapsedTime(&ms, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return static_cast<double>(ms) / iterations;
}

std::uint32_t blocks_for(std::size_t rows) {
  std::uint32_t b = static_cast<std::uint32_t>((rows + 255U) / 256U);
  return b == 0U ? 1U : b;
}
std::uint32_t warp_blocks(std::size_t rows, int warps_per_row) {
  std::uint32_t b = static_cast<std::uint32_t>((rows * 32U * warps_per_row + 255U) / 256U);
  return b == 0U ? 1U : b;
}

}  // namespace

int main() {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) return 77;
  if (cudaSetDevice(0) != cudaSuccess) return 77;
  cudaDeviceProp properties{};
  if (cudaGetDeviceProperties(&properties, 0) != cudaSuccess) return 77;
  if (properties.major != 12 || properties.minor != 0) return 77;

  // Exhaustive hardware E2M1 proof over all 256 packed bytes.
  std::uint8_t* d_bad = nullptr;
  float* d_err = nullptr;
  assert(cudaMalloc(&d_bad, 2) == cudaSuccess);
  assert(cudaMalloc(&d_err, sizeof(float)) == cudaSuccess);
  assert(cudaMemset(d_bad, 0, 2) == cudaSuccess);
  assert(cudaMemset(d_err, 0, sizeof(float)) == cudaSuccess);
  e2m1_check_kernel<<<1, 256>>>(d_bad, d_err);
  assert(cudaGetLastError() == cudaSuccess);
  assert(cudaDeviceSynchronize() == cudaSuccess);
  std::uint8_t bad = 0;
  assert(cudaMemcpy(&bad, d_bad, 1, cudaMemcpyDeviceToHost) == cudaSuccess);
  std::printf("hardware E2M1 exhaustive: %s (first mismatch byte=%u)\n",
              bad == 0 ? "EXACT over all 256 packed bytes" : "MISMATCH", bad);
  cudaFree(d_bad);
  cudaFree(d_err);

  std::printf("\n%-24s %10s %10s %10s %10s %10s %10s\n", "shape", "A stream", "B swdec",
              "C dec+mul", "D fullP6", "E hw+fp16", "E2/E4");
  double weighted_d = 0.0, weighted_e = 0.0, weighted_f = 0.0;
  for (std::size_t s = 0; s < kShapes.size(); ++s) {
    const auto& shape = kShapes[s];
    const std::size_t packed_bytes = shape.rows * (shape.inputs / 2U);
    const std::size_t scale_bytes = shape.rows * (shape.inputs / 16U);
    float *d_input = nullptr, *d_out = nullptr, *d_out_b = nullptr, *d_partial = nullptr,
          *d_scale2 = nullptr;
    std::uint8_t *d_packed = nullptr, *d_scales = nullptr;
    std::uint16_t* d_scales_fp16 = nullptr;
    assert(cudaMalloc(&d_input, shape.inputs * sizeof(float)) == cudaSuccess);
    assert(cudaMalloc(&d_packed, packed_bytes) == cudaSuccess);
    assert(cudaMalloc(&d_scales, scale_bytes) == cudaSuccess);
    assert(cudaMalloc(&d_scales_fp16, scale_bytes * 2U) == cudaSuccess);
    assert(cudaMalloc(&d_scale2, sizeof(float)) == cudaSuccess);
    assert(cudaMalloc(&d_out, shape.rows * sizeof(float)) == cudaSuccess);
    assert(cudaMalloc(&d_out_b, shape.rows * sizeof(float)) == cudaSuccess);
    assert(cudaMalloc(&d_partial, shape.rows * 4U * sizeof(float)) == cudaSuccess);
    std::vector<float> h_input(shape.inputs);
    std::vector<std::uint8_t> h_packed(packed_bytes), h_scales(scale_bytes);
    std::vector<std::uint16_t> h_scales16(scale_bytes);
    for (auto& v : h_input) v = (static_cast<float>(next_word() % 2000U) - 1000.0F) / 500.0F;
    for (auto& v : h_packed) v = static_cast<std::uint8_t>(next_word());
    for (std::size_t i = 0; i < scale_bytes; ++i) {
      const std::uint8_t code = static_cast<std::uint8_t>(0x20U | (next_word() & 0x0FU));
      h_scales[i] = code;
      const float decoded = decode_e4m3fn_host(code);
      const __half h = __float2half(decoded);
      assert(__half2float(h) == decoded);  // exact round-trip
      h_scales16[i] = *reinterpret_cast<const std::uint16_t*>(&h);
    }
    const float host_scale2 = 1.0e-4F;
    assert(cudaMemcpy(d_input, h_input.data(), shape.inputs * sizeof(float),
                      cudaMemcpyHostToDevice) == cudaSuccess);
    assert(cudaMemcpy(d_packed, h_packed.data(), packed_bytes, cudaMemcpyHostToDevice) ==
           cudaSuccess);
    assert(cudaMemcpy(d_scales, h_scales.data(), scale_bytes, cudaMemcpyHostToDevice) ==
           cudaSuccess);
    assert(cudaMemcpy(d_scales_fp16, h_scales16.data(), scale_bytes * 2U,
                      cudaMemcpyHostToDevice) == cudaSuccess);
    assert(cudaMemcpy(d_scale2, &host_scale2, sizeof(float), cudaMemcpyHostToDevice) ==
           cudaSuccess);

    const std::uint32_t stream_blocks =
        static_cast<std::uint32_t>((packed_bytes / 16U + 255U) / 256U);
    const double ms_a = benchmark(
        [&] {
          stream_kernel<<<stream_blocks, 256>>>(d_packed, d_scales, d_out, packed_bytes,
                                                scale_bytes);
        },
        20);
    const std::uint32_t wb = warp_blocks(shape.rows, 1);
    const double ms_b = benchmark(
        [&] {
          decode_sw_kernel<<<wb, 256>>>(d_packed, d_out_b, shape.rows, shape.inputs / 2U);
        },
        20);
    const double ms_noload = benchmark(
        [&] {
          decode_noload_kernel<<<wb, 256>>>(d_packed, d_scales, d_scale2, d_out_b, shape.rows,
                                            shape.inputs);
        },
        20);
    const double ms_mul1 = benchmark(
        [&] {
          decode_mul1_kernel<<<wb, 256>>>(d_input, d_packed, d_out_b, shape.rows, shape.inputs);
        },
        20);
    const double ms_c = benchmark(
        [&] {
          decode_mul_kernel<<<wb, 256>>>(d_input, d_packed, d_scales, d_scale2, d_out_b,
                                         shape.rows, shape.inputs);
        },
        20);
    const double ms_d = benchmark(
        [&] {
          superinfer::sm120::cuda_runtime::detail::nvfp4_linear_warp_f32<<<wb, 256>>>(
              d_input, d_packed, d_scales, d_scale2, d_out, shape.inputs, shape.rows);
        },
        20);
    const double ms_e = benchmark(
        [&] {
          full_warp_hw_kernel<1><<<wb, 256>>>(d_input, d_packed, d_scales_fp16, d_scale2, d_out_b,
                                              shape.rows, shape.inputs);
        },
        20);
    const double ms_e2 = benchmark(
        [&] {
          full_warp_hw_kernel<2><<<warp_blocks(shape.rows, 2), 256>>>(
              d_input, d_packed, d_scales_fp16, d_scale2, d_partial, shape.rows, shape.inputs);
          combine_sums<<<blocks_for(shape.rows), 256>>>(d_partial, d_out_b, shape.rows, 2);
        },
        20);
    const double ms_e4 = benchmark(
        [&] {
          full_warp_hw_kernel<4><<<warp_blocks(shape.rows, 4), 256>>>(
              d_input, d_packed, d_scales_fp16, d_scale2, d_partial, shape.rows, shape.inputs);
          combine_sums<<<blocks_for(shape.rows), 256>>>(d_partial, d_out_b, shape.rows, 4);
        },
        20);
    std::printf("%-22s A=%.3f B=%.3f noload=%.3f mul1=%.3f C=%.3f D=%.3f E=%.3f\n", shape.name,
                ms_a, ms_b, ms_noload, ms_mul1, ms_c, ms_d, ms_e);
    // Per-projection differential D vs E.
    std::vector<float> ref(shape.rows), hw(shape.rows);
    superinfer::sm120::cuda_runtime::detail::nvfp4_linear_warp_f32<<<wb, 256>>>(
        d_input, d_packed, d_scales, d_scale2, d_out, shape.inputs, shape.rows);
    cudaDeviceSynchronize();
    assert(cudaMemcpy(ref.data(), d_out, shape.rows * sizeof(float),
                      cudaMemcpyDeviceToHost) == cudaSuccess);
    const std::uint32_t hwb = warp_blocks(shape.rows, 1);
    full_warp_hw_kernel<1><<<hwb, 256>>>(d_input, d_packed, d_scales_fp16, d_scale2, d_out_b,
                                         shape.rows, shape.inputs);
    cudaDeviceSynchronize();
    assert(cudaMemcpy(hw.data(), d_out_b, shape.rows * sizeof(float),
                      cudaMemcpyDeviceToHost) == cudaSuccess);
    double max_diff = 0.0;
    for (std::size_t r = 0; r < shape.rows; ++r) {
      max_diff = std::fmax(max_diff, std::fabs(static_cast<double>(ref[r]) - hw[r]));
    }
    std::printf("    D-vs-E max_abs=%.6g\n", max_diff);
    const double ms_f = benchmark(
        [&] {
          full_warp_interleaved_hw_kernel<<<hwb, 256>>>(d_input, d_packed, d_scales_fp16,
                                                        d_scale2, d_out_b, shape.rows,
                                                        shape.inputs);
        },
        20);
    // Interleaved differential vs D.
    full_warp_interleaved_hw_kernel<<<hwb, 256>>>(d_input, d_packed, d_scales_fp16, d_scale2,
                                                  d_out_b, shape.rows, shape.inputs);
    cudaDeviceSynchronize();
    assert(cudaMemcpy(hw.data(), d_out_b, shape.rows * sizeof(float),
                      cudaMemcpyDeviceToHost) == cudaSuccess);
    double max_diff_f = 0.0;
    for (std::size_t r = 0; r < shape.rows; ++r) {
      max_diff_f = std::fmax(max_diff_f, std::fabs(static_cast<double>(ref[r]) - hw[r]));
    }
    std::printf("    lane-interleaved hw: %.3f ms   D-vs-F max_abs=%.6g\n", ms_f, max_diff_f);
    weighted_d += ms_d * kMultiplicity[s];
    weighted_e += ms_e * kMultiplicity[s];
    weighted_f += ms_f * kMultiplicity[s];
    cudaFree(d_input);
    cudaFree(d_packed);
    cudaFree(d_scales);
    cudaFree(d_scales_fp16);
    cudaFree(d_scale2);
    cudaFree(d_out);
    cudaFree(d_out_b);
    cudaFree(d_partial);
  }
  std::printf("\nweighted per-token: D(P6)=%.1f ms  E(hw+fp16)=%.1f ms  F(lane-interleaved hw)=%.1f ms\n",
              weighted_d, weighted_e, weighted_f);
  std::printf("speedups: E/D=%.2fx  F/D=%.2fx\n", weighted_d / weighted_e, weighted_d / weighted_f);
  return 0;
}
