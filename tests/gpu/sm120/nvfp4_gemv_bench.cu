// P6 microbenchmark: NVFP4 GEMV layout/candidate comparison.
//
// Compares the current row-per-thread kernel against two candidates on the real
// Qwen3.8 NVFP4 projection shapes:
//   A) row-per-thread over a 32-row-interleaved packed+scale layout (bit-exact)
//   B) warp-per-output-row over the existing row-major layout (order changes)
// Reports time, effective bytes, achieved GB/s and roofline fraction per shape.
// Nsight Compute counters are unavailable on this host (RmProfilingAdminOnly=1),
// so achieved bandwidth is measured directly with CUDA events.

#include <sm120/runtime/cuda_plan_executor.cuh>

#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

using superinfer::sm120::cuda_runtime::detail::nvfp4_linear_rows_vec_f32;

struct Shape {
  const char* name;
  std::size_t rows;   // output features
  std::size_t inputs; // input features (logical)
};

// Real Qwen3.8 NVFP4 projection shapes (out, in) with multiplicities.
constexpr std::array<Shape, 8> kShapes{{
    {"lm_head_248320x5120", 248320, 5120},
    {"mlp_17408x5120", 17408, 5120},
    {"qkv_12288x5120", 12288, 5120},
    {"gdn_10240x5120", 10240, 5120},
    {"attn_6144x5120", 6144, 5120},
    {"down_5120x17408", 5120, 17408},
    {"down_5120x6144", 5120, 6144},
    {"small_1024x5120", 1024, 5120},
}};

struct Buffers {
  std::size_t rows{0};
  std::size_t inputs{0};
  float* input{nullptr};
  std::uint8_t* packed{nullptr};
  std::uint8_t* scales{nullptr};
  float* scale_2{nullptr};
  float* out_ref{nullptr};
  float* out_a{nullptr};
  float* out_b{nullptr};
};

std::uint64_t g_state = 0x9E3779B97F4A7C15ULL;
std::uint32_t next_word() {
  g_state = g_state * 6364136223846793005ULL + 1442695040888963407ULL;
  return static_cast<std::uint32_t>(g_state >> 33U);
}

// Interleaved layout: rows grouped in 32s; within a group the packed column
// groups are lane-major so a warp's 32 lanes read 512 contiguous bytes.
__global__ void repack_packed_interleaved(const std::uint8_t* src, std::uint8_t* dst,
                                          std::size_t rows, std::size_t packed_row_bytes) {
  const std::size_t group = blockIdx.x;
  const std::size_t row_base = group * 32U;
  const std::size_t groups = packed_row_bytes / 16U;
  for (std::size_t index = threadIdx.x; index < groups * 32U; index += blockDim.x) {
    const std::size_t col_group = index / 32U;
    const std::size_t lane = index % 32U;
    if (row_base + lane >= rows) continue;
    const std::uint8_t* source = src + (row_base + lane) * packed_row_bytes + col_group * 16U;
    std::uint8_t* target = dst + group * (32U * packed_row_bytes) + col_group * (32U * 16U) +
                           lane * 16U;
    for (std::size_t byte = 0; byte < 16U; ++byte) target[byte] = source[byte];
  }
}

__global__ void repack_scales_interleaved(const std::uint8_t* src, std::uint8_t* dst,
                                          std::size_t rows, std::size_t scale_columns) {
  const std::size_t group = blockIdx.x;
  const std::size_t row_base = group * 32U;
  for (std::size_t index = threadIdx.x; index < scale_columns * 32U; index += blockDim.x) {
    const std::size_t column = index / 32U;
    const std::size_t lane = index % 32U;
    if (row_base + lane >= rows) continue;
    dst[group * (32U * scale_columns) + column * 32U + lane] =
        src[(row_base + lane) * scale_columns + column];
  }
}

// Candidate A: identical per-row serial accumulation, interleaved addressing.
__global__ void nvfp4_linear_interleaved_f32(const float* input, const std::uint8_t* packed,
                                             const std::uint8_t* scales,
                                             const float* tensor_scale, float* output,
                                             std::size_t input_elements,
                                             std::size_t output_elements) {
  const float tensor = *tensor_scale;
  const std::size_t packed_row_bytes = input_elements / 2U;
  const std::size_t scale_columns = input_elements / 16U;
  for (std::size_t row = blockIdx.x * blockDim.x + threadIdx.x; row < output_elements;
       row += blockDim.x * gridDim.x) {
    const std::size_t group = row / 32U;
    const std::size_t lane = row % 32U;
    const std::uint8_t* packed_base = packed + group * (32U * packed_row_bytes) + lane * 16U;
    const std::uint8_t* scale_base = scales + group * (32U * scale_columns) + lane;
    float sum = 0.0F;
    std::size_t column = 0;
    std::size_t col_group = 0;
    for (; column + 32U <= input_elements; column += 32U, ++col_group) {
      const uint4 word =
          *reinterpret_cast<const uint4*>(packed_base + col_group * (32U * 16U));
      const std::uint8_t bytes[16] = {
          static_cast<std::uint8_t>(word.x), static_cast<std::uint8_t>(word.x >> 8U),
          static_cast<std::uint8_t>(word.x >> 16U), static_cast<std::uint8_t>(word.x >> 24U),
          static_cast<std::uint8_t>(word.y), static_cast<std::uint8_t>(word.y >> 8U),
          static_cast<std::uint8_t>(word.y >> 16U), static_cast<std::uint8_t>(word.y >> 24U),
          static_cast<std::uint8_t>(word.z), static_cast<std::uint8_t>(word.z >> 8U),
          static_cast<std::uint8_t>(word.z >> 16U), static_cast<std::uint8_t>(word.z >> 24U),
          static_cast<std::uint8_t>(word.w), static_cast<std::uint8_t>(word.w >> 8U),
          static_cast<std::uint8_t>(word.w >> 16U), static_cast<std::uint8_t>(word.w >> 24U)};
      const float scale0 = superinfer::sm120::cuda_runtime::detail::decode_e4m3fn_device(
          scale_base[(column / 16U) * 32U]);
      const float scale1 = superinfer::sm120::cuda_runtime::detail::decode_e4m3fn_device(
          scale_base[(column / 16U + 1U) * 32U]);
      for (std::size_t pair = 0; pair < 8U; ++pair) {
        sum += (superinfer::sm120::cuda_runtime::detail::decode_e2m1_device(bytes[pair] & 0x0FU) *
                scale0 * tensor) *
               input[column + pair * 2U];
        sum += (superinfer::sm120::cuda_runtime::detail::decode_e2m1_device(bytes[pair] >> 4U) *
                scale0 * tensor) *
               input[column + pair * 2U + 1U];
      }
      for (std::size_t pair = 8U; pair < 16U; ++pair) {
        sum += (superinfer::sm120::cuda_runtime::detail::decode_e2m1_device(bytes[pair] & 0x0FU) *
                scale1 * tensor) *
               input[column + pair * 2U];
        sum += (superinfer::sm120::cuda_runtime::detail::decode_e2m1_device(bytes[pair] >> 4U) *
                scale1 * tensor) *
               input[column + pair * 2U + 1U];
      }
    }
    output[row] = sum;
  }
}

// Candidate B: warp per output row, contiguous per-lane column ranges, tree
// reduction. Accumulation order differs from the incumbent.
__global__ void nvfp4_linear_warp_f32(const float* input, const std::uint8_t* packed,
                                      const std::uint8_t* scales, const float* tensor_scale,
                                      float* output, std::size_t input_elements,
                                      std::size_t output_elements) {
  const float tensor = *tensor_scale;
  const std::size_t warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32U;
  const std::size_t lane = threadIdx.x % 32U;
  if (warp >= output_elements) return;
  const std::uint8_t* packed_row = packed + warp * (input_elements / 2U);
  const std::uint8_t* scale_row = scales + warp * (input_elements / 16U);
  // Contiguous chunk per lane, rounded to a multiple of 32 columns.
  const std::size_t chunk = ((input_elements + 31U) / 32U + 31U) / 32U * 32U;
  const std::size_t begin = lane * chunk;
  const std::size_t end = (begin < input_elements) ? (begin + chunk < input_elements
                                                          ? begin + chunk
                                                          : input_elements)
                                                   : input_elements;
  float sum = 0.0F;
  std::size_t column = begin;
  while (column < end) {
    const std::size_t group = column / 32U;
    const uint4 word = *reinterpret_cast<const uint4*>(packed_row + group * 16U);
    const std::uint8_t bytes[16] = {
        static_cast<std::uint8_t>(word.x), static_cast<std::uint8_t>(word.x >> 8U),
        static_cast<std::uint8_t>(word.x >> 16U), static_cast<std::uint8_t>(word.x >> 24U),
        static_cast<std::uint8_t>(word.y), static_cast<std::uint8_t>(word.y >> 8U),
        static_cast<std::uint8_t>(word.y >> 16U), static_cast<std::uint8_t>(word.y >> 24U),
        static_cast<std::uint8_t>(word.z), static_cast<std::uint8_t>(word.z >> 8U),
        static_cast<std::uint8_t>(word.z >> 16U), static_cast<std::uint8_t>(word.z >> 24U),
        static_cast<std::uint8_t>(word.w), static_cast<std::uint8_t>(word.w >> 8U),
        static_cast<std::uint8_t>(word.w >> 16U), static_cast<std::uint8_t>(word.w >> 24U)};
    const float scale0 =
        superinfer::sm120::cuda_runtime::detail::decode_e4m3fn_device(scale_row[group * 2U]);
    const float scale1 =
        superinfer::sm120::cuda_runtime::detail::decode_e4m3fn_device(scale_row[group * 2U + 1U]);
    for (std::size_t pair = 0; pair < 8U; ++pair) {
      sum += (superinfer::sm120::cuda_runtime::detail::decode_e2m1_device(bytes[pair] & 0x0FU) *
              scale0 * tensor) *
             input[column + pair * 2U];
      sum += (superinfer::sm120::cuda_runtime::detail::decode_e2m1_device(bytes[pair] >> 4U) *
              scale0 * tensor) *
             input[column + pair * 2U + 1U];
    }
    for (std::size_t pair = 8U; pair < 16U; ++pair) {
      sum += (superinfer::sm120::cuda_runtime::detail::decode_e2m1_device(bytes[pair] & 0x0FU) *
              scale1 * tensor) *
             input[column + pair * 2U];
      sum += (superinfer::sm120::cuda_runtime::detail::decode_e2m1_device(bytes[pair] >> 4U) *
              scale1 * tensor) *
             input[column + pair * 2U + 1U];
    }
    column += 32U;
  }
  for (std::size_t offset = 16U; offset > 0U; offset >>= 1U) {
    sum += __shfl_down_sync(0xFFFFFFFFU, sum, offset);
  }
  if (lane == 0U) output[warp] = sum;
}

constexpr double kHbmBytesPerSecond = 1.79e12;  // RTX 5090 peak DRAM bandwidth

void report(const Shape& shape, const char* label, double ms, std::size_t packed_bytes,
            std::size_t scale_bytes) {
  const double bytes = static_cast<double>(packed_bytes + scale_bytes);
  const double gbps = bytes / (ms * 1.0e-3) / 1.0e9;
  std::printf("  %-22s %8.3f ms  %8.1f GB/s  %5.1f%% roofline\n", label, ms, gbps,
              100.0 * (gbps * 1.0e9) / kHbmBytesPerSecond);
}

}  // namespace

int main() {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) return 77;
  if (cudaSetDevice(0) != cudaSuccess) return 77;
  cudaDeviceProp properties{};
  if (cudaGetDeviceProperties(&properties, 0) != cudaSuccess) return 77;
  if (properties.major != 12 || properties.minor != 0) return 77;

  std::printf("NVFP4 GEMV layout/candidate benchmark (device: %s)\n", properties.name);
  for (const auto& shape : kShapes) {
    Buffers buffers;
    buffers.rows = shape.rows;
    buffers.inputs = shape.inputs;
    const std::size_t packed_bytes = shape.rows * (shape.inputs / 2U);
    const std::size_t scale_bytes = shape.rows * (shape.inputs / 16U);
    std::vector<float> host_input(shape.inputs);
    std::vector<std::uint8_t> host_packed(packed_bytes);
    std::vector<std::uint8_t> host_scales(scale_bytes);
    float host_scale_2 = 1.0e-4F;
    for (auto& value : host_input) value = (static_cast<float>(next_word() % 2000U) - 1000.0F) / 500.0F;
    for (auto& value : host_packed) value = static_cast<std::uint8_t>(next_word());
    for (auto& value : host_scales) value = static_cast<std::uint8_t>(0x20U | (next_word() & 0x0FU));
    assert(cudaMalloc(&buffers.input, shape.inputs * sizeof(float)) == cudaSuccess);
    assert(cudaMalloc(&buffers.packed, packed_bytes) == cudaSuccess);
    assert(cudaMalloc(&buffers.scales, scale_bytes) == cudaSuccess);
    assert(cudaMalloc(&buffers.scale_2, sizeof(float)) == cudaSuccess);
    assert(cudaMalloc(&buffers.out_ref, shape.rows * sizeof(float)) == cudaSuccess);
    assert(cudaMalloc(&buffers.out_a, shape.rows * sizeof(float)) == cudaSuccess);
    assert(cudaMalloc(&buffers.out_b, shape.rows * sizeof(float)) == cudaSuccess);
    assert(cudaMemcpy(buffers.input, host_input.data(), shape.inputs * sizeof(float),
                      cudaMemcpyHostToDevice) == cudaSuccess);
    assert(cudaMemcpy(buffers.packed, host_packed.data(), packed_bytes,
                      cudaMemcpyHostToDevice) == cudaSuccess);
    assert(cudaMemcpy(buffers.scales, host_scales.data(), scale_bytes,
                      cudaMemcpyHostToDevice) == cudaSuccess);
    assert(cudaMemcpy(buffers.scale_2, &host_scale_2, sizeof(float), cudaMemcpyHostToDevice) ==
           cudaSuccess);

    std::uint8_t* packed_interleaved = nullptr;
    std::uint8_t* scales_interleaved = nullptr;
    assert(cudaMalloc(&packed_interleaved, packed_bytes) == cudaSuccess);
    assert(cudaMalloc(&scales_interleaved, scale_bytes) == cudaSuccess);
    Buffers interleaved = buffers;
    interleaved.packed = packed_interleaved;
    interleaved.scales = scales_interleaved;

    const std::size_t groups = shape.rows / 32U;
    repack_packed_interleaved<<<static_cast<std::uint32_t>(groups), 256>>>(
        buffers.packed, packed_interleaved, shape.rows, shape.inputs / 2U);
    repack_scales_interleaved<<<static_cast<std::uint32_t>(groups), 256>>>(
        buffers.scales, scales_interleaved, shape.rows, shape.inputs / 16U);
    assert(cudaGetLastError() == cudaSuccess);
    assert(cudaDeviceSynchronize() == cudaSuccess);

    std::uint32_t blocks = static_cast<std::uint32_t>((shape.rows + 255U) / 256U);
    if (blocks > 4096U) blocks = 4096U;
    // Candidate B uses one warp (32 threads) per output row.
    std::uint32_t warp_blocks = static_cast<std::uint32_t>((shape.rows * 32U + 255U) / 256U);
    if (warp_blocks > 2000000U) warp_blocks = 2000000U;

    // Validate A bit-exact and B tolerance.
    nvfp4_linear_rows_vec_f32<<<blocks, 256>>>(buffers.input, buffers.packed, buffers.scales,
                                               buffers.scale_2, buffers.out_ref, buffers.inputs,
                                               buffers.rows);
    nvfp4_linear_interleaved_f32<<<blocks, 256>>>(buffers.input, packed_interleaved,
                                                  scales_interleaved, buffers.scale_2,
                                                  buffers.out_a, buffers.inputs, buffers.rows);
    nvfp4_linear_warp_f32<<<warp_blocks, 256>>>(buffers.input, buffers.packed, buffers.scales,
                                                buffers.scale_2, buffers.out_b, buffers.inputs,
                                                buffers.rows);
    assert(cudaGetLastError() == cudaSuccess);
    assert(cudaDeviceSynchronize() == cudaSuccess);
    std::vector<float> out_ref(shape.rows), out_a(shape.rows), out_b(shape.rows);
    assert(cudaMemcpy(out_ref.data(), buffers.out_ref, shape.rows * sizeof(float),
                      cudaMemcpyDeviceToHost) == cudaSuccess);
    assert(cudaMemcpy(out_a.data(), buffers.out_a, shape.rows * sizeof(float),
                      cudaMemcpyDeviceToHost) == cudaSuccess);
    assert(cudaMemcpy(out_b.data(), buffers.out_b, shape.rows * sizeof(float),
                      cudaMemcpyDeviceToHost) == cudaSuccess);
    bool a_exact = true;
    double b_max = 0.0, b_rmse = 0.0;
    for (std::size_t row = 0; row < shape.rows; ++row) {
      if (out_a[row] != out_ref[row]) a_exact = false;
      const double diff = std::fabs(static_cast<double>(out_b[row]) - out_ref[row]);
      b_max = diff > b_max ? diff : b_max;
      b_rmse += diff * diff;
    }
    b_rmse = std::sqrt(b_rmse / static_cast<double>(shape.rows));

    // Timed with events via explicit loops.
    const auto bench = [&](auto launch, float* out) {
      launch(buffers);
      cudaDeviceSynchronize();
      cudaEvent_t start{}, stop{};
      cudaEventCreate(&start);
      cudaEventCreate(&stop);
      cudaEventRecord(start);
      const int iterations = shape.rows > 100000 ? 10 : 30;
      for (int iteration = 0; iteration < iterations; ++iteration) launch(buffers);
      cudaEventRecord(stop);
      cudaEventSynchronize(stop);
      float ms = 0.0F;
      cudaEventElapsedTime(&ms, start, stop);
      cudaEventDestroy(start);
      cudaEventDestroy(stop);
      (void)out;
      return static_cast<double>(ms) / iterations;
    };

    std::printf("%s  (packed %.1f MB, scales %.2f MB)\n", shape.name,
                packed_bytes / 1.0e6, scale_bytes / 1.0e6);
    const double ms_ref = bench(
        [&](const Buffers& b) {
          nvfp4_linear_rows_vec_f32<<<blocks, 256>>>(b.input, b.packed, b.scales, b.scale_2,
                                                     b.out_ref, b.inputs, b.rows);
        },
        buffers.out_ref);
    const double ms_a = bench(
        [&](const Buffers& b) {
          nvfp4_linear_interleaved_f32<<<blocks, 256>>>(b.input, b.packed, b.scales, b.scale_2,
                                                        b.out_a, b.inputs, b.rows);
        },
        buffers.out_a);
    const double ms_b = bench(
        [&](const Buffers& b) {
          nvfp4_linear_warp_f32<<<warp_blocks, 256>>>(b.input, b.packed, b.scales, b.scale_2,
                                                      b.out_b, b.inputs, b.rows);
        },
        buffers.out_b);
    report(shape, "incumbent", ms_ref, packed_bytes, scale_bytes);
    report(shape, "A interleaved", ms_a, packed_bytes, scale_bytes);
    report(shape, "B warp-per-row", ms_b, packed_bytes, scale_bytes);
    std::printf("    A bit-exact: %s   B max_abs=%.6g rmse=%.6g\n", a_exact ? "yes" : "NO",
                b_max, b_rmse);

    cudaFree(packed_interleaved);
    cudaFree(scales_interleaved);
    cudaFree(buffers.input);
    cudaFree(buffers.packed);
    cudaFree(buffers.scales);
    cudaFree(buffers.scale_2);
    cudaFree(buffers.out_ref);
    cudaFree(buffers.out_a);
    cudaFree(buffers.out_b);
  }
  return 0;
}
