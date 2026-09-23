#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <sm120/runtime/cuda_ownership.cuh>
#include <superinfer/base/views.hpp>
#include <superinfer/ir/physical_plan.hpp>

#include <cuda_fp4.h>
#include <cuda_fp8.h>

namespace superinfer::sm120::cuda_runtime {

struct CudaExecutionTrace final {
  std::uint64_t commands_executed{0};
  std::uint64_t launches{0};
};

namespace detail {

__global__ inline void residual_f32(const float* left, const float* right, float* output,
                                    std::size_t elements) {
  for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < elements;
       index += blockDim.x * gridDim.x) {
    output[index] = left[index] + right[index];
  }
}

__global__ inline void silu_mul_f32(const float* gate, const float* value, float* output,
                                    std::size_t elements) {
  for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < elements;
       index += blockDim.x * gridDim.x) {
    const float gate_value = gate[index];
    output[index] = gate_value / (1.0F + expf(-gate_value)) * value[index];
  }
}

__global__ inline void sigmoid_mul_f32(const float* gate, const float* value, float* output,
                                       std::size_t elements) {
  for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < elements;
       index += blockDim.x * gridDim.x) {
    output[index] = (1.0F / (1.0F + expf(-gate[index]))) * value[index];
  }
}

__global__ inline void split_f32(const float* input, float* first, float* second,
                                std::size_t first_elements, std::size_t total_elements) {
  for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < total_elements;
       index += blockDim.x * gridDim.x) {
    if (index < first_elements) {
      first[index] = input[index];
    } else {
      second[index - first_elements] = input[index];
    }
  }
}

__global__ inline void split_last_f32(const float* input, float* first, float* second,
                                      std::size_t outer, std::size_t first_elements,
                                      std::size_t second_elements) {
  const std::size_t total_elements = outer * (first_elements + second_elements);
  for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < total_elements;
       index += blockDim.x * gridDim.x) {
    const std::size_t group = index / (first_elements + second_elements);
    const std::size_t inner = index % (first_elements + second_elements);
    if (inner < first_elements) {
      first[group * first_elements + inner] = input[index];
    } else {
      second[group * second_elements + inner - first_elements] = input[index];
    }
  }
}

__device__ inline float bf16_to_float_device(std::uint16_t value) {
  return __uint_as_float(static_cast<std::uint32_t>(value) << 16U);
}

__device__ inline std::uint16_t float_to_bf16_device(float value) {
  const std::uint32_t bits = __float_as_uint(value);
  return static_cast<std::uint16_t>((bits + (((bits >> 16U) & 1U) + 0x7FFFU)) >> 16U);
}

__global__ inline void rope_f32(const float* input, float* output, std::size_t heads,
                                std::size_t head_dimension, std::size_t rotary_dimension,
                                std::size_t position, float theta) {
  const std::size_t elements = heads * head_dimension;
  for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < elements;
       index += blockDim.x * gridDim.x) {
    const std::size_t dimension = index % head_dimension;
    if (dimension >= rotary_dimension) {
      output[index] = input[index];
      continue;
    }
    const std::size_t half = rotary_dimension / 2U;
    const std::size_t pair = dimension < half ? dimension : dimension - half;
    const float inverse_frequency = powf(theta, -2.0F * static_cast<float>(pair) /
                                                  static_cast<float>(rotary_dimension));
    const float angle = static_cast<float>(position) * inverse_frequency;
    const float cosine = cosf(angle);
    const float sine = sinf(angle);
    const std::size_t base = index - dimension;
    const float first = input[base + pair];
    const float second = input[base + pair + half];
    output[index] = dimension < half ? first * cosine - second * sine
                                     : second * cosine + first * sine;
  }
}

__global__ inline void cache_append_f32_bf16(const float* keys, const float* values,
                                             std::uint16_t* key_cache,
                                             std::uint16_t* value_cache, std::size_t heads,
                                             std::size_t head_dimension, std::size_t position,
                                             std::size_t capacity) {
  const std::size_t elements = heads * head_dimension;
  for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < elements;
       index += blockDim.x * gridDim.x) {
    const std::size_t destination = position * elements + index;
    if (position >= capacity) continue;
    key_cache[destination] = float_to_bf16_device(keys[index]);
    value_cache[destination] = float_to_bf16_device(values[index]);
  }
}

__global__ inline void gated_delta_parameters_f32(
    const float* a, const float* b, const float* a_log, const float* dt_bias,
    float* log_decay, float* beta, std::size_t elements) {
  for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < elements;
       index += blockDim.x * gridDim.x) {
    const float shifted = a[index] + dt_bias[index];
    const float softplus = fmaxf(shifted, 0.0F) + log1pf(expf(-fabsf(shifted)));
    log_decay[index] = -expf(a_log[index]) * softplus;
    beta[index] = 1.0F / (1.0F + expf(-b[index]));
  }
}

__global__ inline void causal_conv_silu_f32(const float* input, const float* weights,
                                            std::uint16_t* state, float* output,
                                            std::size_t channels, std::size_t kernel_size) {
  for (std::size_t channel = blockIdx.x * blockDim.x + threadIdx.x; channel < channels;
       channel += blockDim.x * gridDim.x) {
    for (std::size_t tap = 0; tap + 1 < kernel_size; ++tap) {
      state[tap * channels + channel] = state[(tap + 1) * channels + channel];
    }
    state[(kernel_size - 1U) * channels + channel] = float_to_bf16_device(input[channel]);
    float sum = 0.0F;
    for (std::size_t tap = 0; tap < kernel_size; ++tap) {
      sum += weights[channel * kernel_size + tap] *
             bf16_to_float_device(state[tap * channels + channel]);
    }
    output[channel] = sum / (1.0F + expf(-sum));
  }
}

__global__ inline void grouped_attention_bf16_cache(
    const float* query, const std::uint16_t* keys, const std::uint16_t* values, float* output,
    std::size_t query_heads, std::size_t kv_heads, std::size_t head_dimension,
    std::size_t positions) {
  const std::size_t group = query_heads / kv_heads;
  const float scale = rsqrtf(static_cast<float>(head_dimension));
  for (std::size_t query_head = blockIdx.x * blockDim.x + threadIdx.x;
       query_head < query_heads; query_head += blockDim.x * gridDim.x) {
    const std::size_t kv_head = query_head / group;
    float maximum = -3.402823466e+38F;
    for (std::size_t position = 0; position < positions; ++position) {
      float score = 0.0F;
      for (std::size_t dimension = 0; dimension < head_dimension; ++dimension) {
        const std::size_t cache_index = (position * kv_heads + kv_head) * head_dimension + dimension;
        score += query[query_head * head_dimension + dimension] *
                 __uint_as_float(static_cast<std::uint32_t>(keys[cache_index]) << 16U);
      }
      maximum = fmaxf(maximum, score * scale);
    }
    float denominator = 0.0F;
    for (std::size_t position = 0; position < positions; ++position) {
      float score = 0.0F;
      for (std::size_t dimension = 0; dimension < head_dimension; ++dimension) {
        const std::size_t cache_index = (position * kv_heads + kv_head) * head_dimension + dimension;
        score += query[query_head * head_dimension + dimension] *
                 __uint_as_float(static_cast<std::uint32_t>(keys[cache_index]) << 16U);
      }
      denominator += expf(score * scale - maximum);
    }
    for (std::size_t dimension = 0; dimension < head_dimension; ++dimension) {
      float result = 0.0F;
      for (std::size_t position = 0; position < positions; ++position) {
        float score = 0.0F;
        for (std::size_t key_dimension = 0; key_dimension < head_dimension; ++key_dimension) {
          const std::size_t cache_index =
              (position * kv_heads + kv_head) * head_dimension + key_dimension;
          score += query[query_head * head_dimension + key_dimension] *
                   __uint_as_float(static_cast<std::uint32_t>(keys[cache_index]) << 16U);
        }
        const float probability = expf(score * scale - maximum) / denominator;
        const std::size_t value_index = (position * kv_heads + kv_head) * head_dimension + dimension;
        result += probability *
                  __uint_as_float(static_cast<std::uint32_t>(values[value_index]) << 16U);
      }
      output[query_head * head_dimension + dimension] = result;
    }
  }
}

__device__ inline float bf16_bits_to_float_device(std::uint16_t value) {
  return __uint_as_float(static_cast<std::uint32_t>(value) << 16U);
}

/** KV-window attention with scores computed once (S04-P1).
 *
 * Bit-identical to `grouped_attention_bf16_cache`: the per-position Q·K score,
 * the max, the per-position softmax numerator, and the per-dimension value
 * accumulation all use the same operands in the same order. The only change is
 * that Q·K is computed once instead of three times, and the value pass no
 * longer recomputes it inside a per-dimension loop (removing an O(head_dim^2)
 * factor). Grid: one block per query head; shared memory holds `positions`
 * scores followed by `positions` probabilities.
 */
__global__ inline void grouped_attention_bf16_cache_cached(
    const float* query, const std::uint16_t* keys, const std::uint16_t* values, float* output,
    std::size_t query_heads, std::size_t kv_heads, std::size_t head_dimension,
    std::size_t positions) {
  extern __shared__ float shared[];
  float* scores = shared;
  float* probabilities = shared + positions;
  const std::size_t group = query_heads / kv_heads;
  const float scale = rsqrtf(static_cast<float>(head_dimension));
  const std::size_t query_head = blockIdx.x;
  if (query_head >= query_heads) return;
  const std::size_t kv_head = query_head / group;
  // Phase 1: Q.K score per position, once. Each thread keeps the incumbent's
  // serial dimension accumulation for the positions it owns.
  for (std::size_t position = threadIdx.x; position < positions; position += blockDim.x) {
    float score = 0.0F;
    for (std::size_t dimension = 0; dimension < head_dimension; ++dimension) {
      const std::size_t cache_index =
          (position * kv_heads + kv_head) * head_dimension + dimension;
      score += query[query_head * head_dimension + dimension] *
               bf16_bits_to_float_device(keys[cache_index]);
    }
    scores[position] = score;
  }
  __syncthreads();
  // Max over positions: fmaxf is exact and order-independent, so any reduction
  // order reproduces the incumbent's sequential maximum bit-for-bit.
  __shared__ float reduction[256];
  float local_max = -3.402823466e+38F;
  for (std::size_t position = threadIdx.x; position < positions; position += blockDim.x) {
    local_max = fmaxf(local_max, scores[position] * scale);
  }
  reduction[threadIdx.x] = local_max;
  __syncthreads();
  for (std::size_t stride = blockDim.x / 2U; stride > 0U; stride >>= 1U) {
    if (threadIdx.x < stride) {
      reduction[threadIdx.x] = fmaxf(reduction[threadIdx.x], reduction[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  __shared__ float maximum_shared;
  if (threadIdx.x == 0) maximum_shared = reduction[0];
  __syncthreads();
  const float maximum = maximum_shared;
  // Phase 2: softmax numerator per position (parallel) then denominator summed
  // sequentially by thread 0 to preserve the incumbent's summation order.
  for (std::size_t position = threadIdx.x; position < positions; position += blockDim.x) {
    probabilities[position] = expf(scores[position] * scale - maximum);
  }
  __syncthreads();
  __shared__ float denominator_shared;
  if (threadIdx.x == 0) {
    float denominator = 0.0F;
    for (std::size_t position = 0; position < positions; ++position) {
      denominator += probabilities[position];
    }
    denominator_shared = denominator;
  }
  __syncthreads();
  const float denominator = denominator_shared;
  for (std::size_t position = threadIdx.x; position < positions; position += blockDim.x) {
    probabilities[position] = probabilities[position] / denominator;
  }
  __syncthreads();
  // Phase 4: value accumulation, dimension-outer/position-inner exactly as the
  // incumbent, reusing the cached probabilities instead of recomputing Q.K.
  for (std::size_t dimension = threadIdx.x; dimension < head_dimension;
       dimension += blockDim.x) {
    float result = 0.0F;
    for (std::size_t position = 0; position < positions; ++position) {
      const std::size_t value_index =
          (position * kv_heads + kv_head) * head_dimension + dimension;
      result += probabilities[position] * bf16_bits_to_float_device(values[value_index]);
    }
    output[query_head * head_dimension + dimension] = result;
  }
}

__global__ inline void embedding_f32(const std::uint32_t* token, const float* table, float* output,
                                     std::size_t vocabulary, std::size_t hidden) {
  const std::uint32_t row = *token;
  for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < hidden;
       index += blockDim.x * gridDim.x) {
    output[index] = row < vocabulary ? table[static_cast<std::size_t>(row) * hidden + index] : 0.0F;
  }
}

__global__ inline void embedding_bf16(const std::uint32_t* token, const std::uint16_t* table,
                                      float* output, std::size_t vocabulary, std::size_t hidden) {
  const std::uint32_t row = *token;
  for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < hidden;
       index += blockDim.x * gridDim.x) {
    if (row >= vocabulary) {
      output[index] = 0.0F;
      continue;
    }
    const std::uint32_t bits = static_cast<std::uint32_t>(
        table[static_cast<std::size_t>(row) * hidden + index])
                               << 16U;
    output[index] = __uint_as_float(bits);
  }
}

__global__ inline void cast_bf16_to_f32(const std::uint16_t* input, float* output,
                                        std::size_t elements) {
  for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < elements;
       index += blockDim.x * gridDim.x) {
    output[index] = __uint_as_float(static_cast<std::uint32_t>(input[index]) << 16U);
  }
}

__global__ inline void cast_f32_to_bf16(const float* input, std::uint16_t* output,
                                        std::size_t elements) {
  for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < elements;
       index += blockDim.x * gridDim.x) {
    const std::uint32_t bits = __float_as_uint(input[index]);
    const std::uint32_t rounding = ((bits >> 16U) & 1U) + 0x7FFFU;
    output[index] = static_cast<std::uint16_t>((bits + rounding) >> 16U);
  }
}

__device__ inline float decode_e2m1_device(std::uint8_t code) {
  constexpr float magnitudes[8] = {0.0F, 0.5F, 1.0F, 1.5F, 2.0F, 3.0F, 4.0F, 6.0F};
  const float magnitude = magnitudes[code & 0x07U];
  return (code & 0x08U) == 0 ? magnitude : -magnitude;
}

__device__ inline float decode_e4m3fn_device(std::uint8_t code) {
  const bool negative = (code & 0x80U) != 0;
  const std::uint8_t exponent = (code >> 3U) & 0x0FU;
  const std::uint8_t mantissa = code & 0x07U;
  if (negative || (exponent == 0x0FU && mantissa == 0x07U)) return __int_as_float(0x7FC00000U);
  if (exponent == 0) return ldexpf(static_cast<float>(mantissa) / 8.0F, -6);
  return ldexpf(1.0F + static_cast<float>(mantissa) / 8.0F,
                static_cast<int>(exponent) - 7);
}

__global__ inline void nvfp4_dequantize(const std::uint8_t* packed, const std::uint8_t* scales,
                                        float* output, std::size_t elements, float scalar) {
  for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < elements;
       index += blockDim.x * gridDim.x) {
    const std::uint8_t packed_value = packed[index / 2U];
    const std::uint8_t code = (index % 2U == 0) ? (packed_value & 0x0FU) : (packed_value >> 4U);
    output[index] = decode_e2m1_device(code) * decode_e4m3fn_device(scales[index / 16U]) * scalar;
  }
}

__global__ inline void linear_f32(const float* input, const float* weights, float* output,
                                  std::size_t input_elements, std::size_t output_elements) {
  for (std::size_t row = blockIdx.x * blockDim.x + threadIdx.x; row < output_elements;
       row += blockDim.x * gridDim.x) {
    float sum = 0.0F;
    for (std::size_t column = 0; column < input_elements; ++column) {
      sum += weights[row * input_elements + column] * input[column];
    }
    output[row] = sum;
  }
}

__global__ inline void gated_dense_ffn_f32(const float* input, const float* gate,
                                           const float* up, const float* down, float* output,
                                           std::size_t hidden, std::size_t intermediate) {
  for (std::size_t row = blockIdx.x * blockDim.x + threadIdx.x; row < hidden;
       row += blockDim.x * gridDim.x) {
    float sum = 0.0F;
    for (std::size_t intermediate_index = 0; intermediate_index < intermediate;
         ++intermediate_index) {
      float gate_value = 0.0F;
      float up_value = 0.0F;
      for (std::size_t column = 0; column < hidden; ++column) {
        gate_value += gate[intermediate_index * hidden + column] * input[column];
        up_value += up[intermediate_index * hidden + column] * input[column];
      }
      const float gated = gate_value / (1.0F + expf(-gate_value)) * up_value;
      sum += down[row * intermediate + intermediate_index] * gated;
    }
    output[row] = sum;
  }
}

__global__ inline void nvfp4_linear_f32(const float* input, const std::uint8_t* packed,
                                        const std::uint8_t* scales, const float* tensor_scale,
                                        float* output,
                                        std::size_t input_elements, std::size_t output_elements) {
  for (std::size_t row = blockIdx.x * blockDim.x + threadIdx.x; row < output_elements;
       row += blockDim.x * gridDim.x) {
    float sum = 0.0F;
    for (std::size_t column = 0; column < input_elements; ++column) {
      const std::uint8_t packed_value = packed[(row * input_elements + column) / 2U];
      const std::uint8_t code = (column % 2U == 0) ? (packed_value & 0x0FU) : (packed_value >> 4U);
      const float weight = decode_e2m1_device(code) *
                           decode_e4m3fn_device(scales[(row * input_elements + column) / 16U]) *
                           *tensor_scale;
      sum += weight * input[column];
    }
    output[row] = sum;
  }
}

/** Row-parallel NVFP4 projection (R02): identical per-row operation order to
 * `nvfp4_linear_f32`, spread over up to 2048 blocks. Each output row is still
 * reduced serially by exactly one thread in column order, so results are
 * bit-identical to the baseline for every shape; only SM occupancy changes.
 * The incumbent stays available as the promotion fallback.
 */
__global__ inline void nvfp4_linear_rows_f32(const float* input, const std::uint8_t* packed,
                                            const std::uint8_t* scales,
                                            const float* tensor_scale, float* output,
                                            std::size_t input_elements,
                                            std::size_t output_elements) {
  for (std::size_t row = blockIdx.x * blockDim.x + threadIdx.x; row < output_elements;
       row += blockDim.x * gridDim.x) {
    float sum = 0.0F;
    for (std::size_t column = 0; column < input_elements; ++column) {
      const std::uint8_t packed_value = packed[(row * input_elements + column) / 2U];
      const std::uint8_t code = (column % 2U == 0) ? (packed_value & 0x0FU) : (packed_value >> 4U);
      const float weight = decode_e2m1_device(code) *
                           decode_e4m3fn_device(scales[(row * input_elements + column) / 16U]) *
                           *tensor_scale;
      sum += weight * input[column];
    }
    output[row] = sum;
  }
}

/** Vectorized NVFP4 projection (S04-P2).
 *
 * Bit-identical to `nvfp4_linear_rows_f32`: the same per-row, per-column
 * accumulation order is preserved. Only the memory access pattern and the
 * redundant block-scale decode change:
 *  - the FP8 block scale is decoded once per 16 columns instead of per column;
 *  - packed weights are read 16 bytes at a time (32 codes) instead of one byte
 *    per two columns.
 * Falls back to the scalar incumbent when the row is not 32-element aligned or
 * the packed pointer is not 16-byte aligned.
 */
__device__ inline void accumulate_nvfp4_code(std::uint8_t code, float scale, float tensor,
                                             const float* input, std::size_t column,
                                             float& sum) {
  sum += (decode_e2m1_device(code) * scale * tensor) * input[column];
}

__global__ inline void nvfp4_linear_rows_vec_f32(const float* input,
                                                 const std::uint8_t* packed,
                                                 const std::uint8_t* scales,
                                                 const float* tensor_scale, float* output,
                                                 std::size_t input_elements,
                                                 std::size_t output_elements) {
  const float tensor = *tensor_scale;
  for (std::size_t row = blockIdx.x * blockDim.x + threadIdx.x; row < output_elements;
       row += blockDim.x * gridDim.x) {
    const std::uint8_t* packed_row = packed + row * (input_elements / 2U);
    const std::uint8_t* scale_row = scales + row * (input_elements / 16U);
    float sum = 0.0F;
    std::size_t column = 0;
    // The 16-byte packed load requires each row to start 16-byte aligned, i.e.
    // input_elements % 32 == 0. Otherwise the whole row takes the scalar path.
    const bool vector_aligned = (input_elements % 32U == 0U) &&
                                (reinterpret_cast<std::uintptr_t>(packed_row) % 16U == 0U);
    for (; vector_aligned && column + 32U <= input_elements; column += 32U) {
      const uint4 word = *reinterpret_cast<const uint4*>(packed_row + column / 2U);
      const std::uint8_t bytes[16] = {
          static_cast<std::uint8_t>(word.x), static_cast<std::uint8_t>(word.x >> 8U),
          static_cast<std::uint8_t>(word.x >> 16U), static_cast<std::uint8_t>(word.x >> 24U),
          static_cast<std::uint8_t>(word.y), static_cast<std::uint8_t>(word.y >> 8U),
          static_cast<std::uint8_t>(word.y >> 16U), static_cast<std::uint8_t>(word.y >> 24U),
          static_cast<std::uint8_t>(word.z), static_cast<std::uint8_t>(word.z >> 8U),
          static_cast<std::uint8_t>(word.z >> 16U), static_cast<std::uint8_t>(word.z >> 24U),
          static_cast<std::uint8_t>(word.w), static_cast<std::uint8_t>(word.w >> 8U),
          static_cast<std::uint8_t>(word.w >> 16U), static_cast<std::uint8_t>(word.w >> 24U)};
      const float scale0 = decode_e4m3fn_device(scale_row[column / 16U]);
      const float scale1 = decode_e4m3fn_device(scale_row[column / 16U + 1U]);
      for (std::size_t pair = 0; pair < 8U; ++pair) {
        accumulate_nvfp4_code(bytes[pair] & 0x0FU, scale0, tensor, input, column + pair * 2U,
                              sum);
        accumulate_nvfp4_code(bytes[pair] >> 4U, scale0, tensor, input, column + pair * 2U + 1U,
                              sum);
      }
      for (std::size_t pair = 8U; pair < 16U; ++pair) {
        accumulate_nvfp4_code(bytes[pair] & 0x0FU, scale1, tensor, input, column + pair * 2U,
                              sum);
        accumulate_nvfp4_code(bytes[pair] >> 4U, scale1, tensor, input, column + pair * 2U + 1U,
                              sum);
      }
    }
    for (; column < input_elements; ++column) {
      const std::uint8_t packed_value = packed_row[column / 2U];
      const std::uint8_t code = (column % 2U == 0) ? (packed_value & 0x0FU) : (packed_value >> 4U);
      accumulate_nvfp4_code(code, decode_e4m3fn_device(scale_row[column / 16U]), tensor, input,
                            column, sum);
    }
    output[row] = sum;
  }
}

/** Reference-correct grouped-query attention over a pre-materialized contiguous KV window. */

/** Warp-per-output-row NVFP4 GEMV (S04-P6 candidate B).
 *
 * Maps one warp to each output row; each lane reduces a contiguous column
 * range and the warp tree-reduces. Global weight loads are naturally coalesced
 * (lanes read contiguous column groups of the same row). The accumulation order
 * differs from the serial incumbent, so this is tolerance-qualified, not
 * bit-exact.
 */
__global__ inline void nvfp4_linear_warp_f32(const float* input, const std::uint8_t* packed,
                                             const std::uint8_t* scales,
                                             const float* tensor_scale, float* output,
                                             std::size_t input_elements,
                                             std::size_t output_elements) {
  const float tensor = *tensor_scale;
  const std::size_t warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32U;
  const std::size_t lane = threadIdx.x % 32U;
  if (warp >= output_elements) return;
  const std::uint8_t* packed_row = packed + warp * (input_elements / 2U);
  const std::uint8_t* scale_row = scales + warp * (input_elements / 16U);
  const std::size_t chunk = ((input_elements + 31U) / 32U + 31U) / 32U * 32U;
  const std::size_t begin = lane * chunk;
  const std::size_t end = begin < input_elements
                              ? (begin + chunk < input_elements ? begin + chunk : input_elements)
                              : input_elements;
  float sum = 0.0F;
  std::size_t column = begin;
  for (; column + 32U <= end; column += 32U) {
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
    const float scale0 = decode_e4m3fn_device(scale_row[group * 2U]);
    const float scale1 = decode_e4m3fn_device(scale_row[group * 2U + 1U]);
    for (std::size_t pair = 0; pair < 8U; ++pair) {
      sum += (decode_e2m1_device(bytes[pair] & 0x0FU) * scale0 * tensor) *
             input[column + pair * 2U];
      sum += (decode_e2m1_device(bytes[pair] >> 4U) * scale0 * tensor) *
             input[column + pair * 2U + 1U];
    }
    for (std::size_t pair = 8U; pair < 16U; ++pair) {
      sum += (decode_e2m1_device(bytes[pair] & 0x0FU) * scale1 * tensor) *
             input[column + pair * 2U];
      sum += (decode_e2m1_device(bytes[pair] >> 4U) * scale1 * tensor) *
             input[column + pair * 2U + 1U];
    }
  }
  for (; column < end; ++column) {
    const std::uint8_t packed_value = packed_row[column / 2U];
    const std::uint8_t code =
        (column % 2U == 0) ? (packed_value & 0x0FU) : (packed_value >> 4U);
    sum += (decode_e2m1_device(code) * decode_e4m3fn_device(scale_row[column / 16U]) * tensor) *
           input[column];
  }
  for (std::size_t offset = 16U; offset > 0U; offset >>= 1U) {
    sum += __shfl_down_sync(0xFFFFFFFFU, sum, offset);
  }
  if (lane == 0U) output[warp] = sum;
}

// ============================================================================================
// S04 performance reset (D-022) E0a — donor-scheduled NVFP4 streaming GEMV.
//
// Schedule adapted from NInfer (Neroued/ninfer, Apache-2.0),
// src/ops/linear/nvfp4/nvfp4_gemv.cuh + shapes/n5120_k6144.cu
// (`Nvfp4GemvSchedule<8, 2, 16, 4, StagedRaw, Default, 2>`): multiple output rows per warp,
// vector packed-code loads, independent accumulator chains, and hardware E2M1/E4M3 decode.
//
// The critical property P7 lacked: the activation value loaded by a lane is reused across
// `ROWS_PER_WARP` output rows, so activation traffic drops by that factor. The donor's prepared
// 128-row swizzled scale plane is not required here; SuperInfer keeps its existing row-major
// `[rows][K/16]` E4M3 scale layout and its FP32 activation recipe (weight-only quantization).
// ============================================================================================

/** Hardware E2M1x2 decode: one packed byte -> two float values. */
__device__ inline float2 decode_e2m1x2_device(std::uint8_t storage) {
  __nv_fp4x2_e2m1 value;
  value.__x = storage;
  return static_cast<float2>(value);
}

/** Hardware E4M3 decode of a single (positive) block scale byte. */
__device__ inline float decode_e4m3_scalar_device(std::uint8_t storage) {
  __nv_fp8x2_e4m3 value;
  value.__x = static_cast<std::uint16_t>(storage) | (static_cast<std::uint16_t>(storage) << 8);
  return static_cast<float2>(value).x;
}

/**
 * Donor-scheduled NVFP4 streaming GEMV: `output[row] = sum_k w[row,k] * input[k]`.
 *
 * Each warp owns ROWS_PER_WARP output rows. Every lane covers 16 K values per phase (exactly one
 * 16-wide scale group per lane per phase) and reuses those 16 activations across all its rows.
 * K must be a multiple of 512 and `rows` a multiple of WARPS_PER_CTA*ROWS_PER_WARP.
 */
template <int WARPS_PER_CTA, int ROWS_PER_WARP, int CHAINS, int MIN_BLOCKS>
__global__ __launch_bounds__(WARPS_PER_CTA * 32, MIN_BLOCKS) void nvfp4_gemv_rows_f32(
    const float* __restrict__ input, const std::uint8_t* __restrict__ packed,
    const std::uint8_t* __restrict__ scales, const float* __restrict__ tensor_scale,
    float* __restrict__ output, std::size_t rows, std::size_t inputs) {
  constexpr int kValuesPerLane = 16;
  constexpr int kValuesPerPhase = 32 * kValuesPerLane;  // 512
  constexpr int kPairsPerLane = kValuesPerLane / 2;     // 8 packed bytes
  constexpr int kRowsPerCta = WARPS_PER_CTA * ROWS_PER_WARP;

  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const std::size_t row0 = static_cast<std::size_t>(blockIdx.x) * kRowsPerCta +
                           static_cast<std::size_t>(warp) * ROWS_PER_WARP;
  if (row0 >= rows) return;

  const float tensor = *tensor_scale;
  const std::size_t code_row_bytes = inputs / 2U;
  const std::size_t scale_row_bytes = inputs / 16U;
  const std::size_t phases = inputs / kValuesPerPhase;

  float accumulators[ROWS_PER_WARP][CHAINS] = {};

  for (std::size_t phase = 0; phase < phases; ++phase) {
    const std::size_t value_base = phase * kValuesPerPhase + static_cast<std::size_t>(lane) * kValuesPerLane;
    // One 16-value group per lane per phase: 4 x float4 activation loads.
    const float4 a0 = *reinterpret_cast<const float4*>(input + value_base);
    const float4 a1 = *reinterpret_cast<const float4*>(input + value_base + 4U);
    const float4 a2 = *reinterpret_cast<const float4*>(input + value_base + 8U);
    const float4 a3 = *reinterpret_cast<const float4*>(input + value_base + 12U);
    const float activations[kValuesPerLane] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w,
                                               a2.x, a2.y, a2.z, a2.w, a3.x, a3.y, a3.z, a3.w};
    const std::size_t group = phase * 32U + static_cast<std::size_t>(lane);
    const std::size_t code_offset = phase * (kValuesPerPhase / 2U) + static_cast<std::size_t>(lane) * kPairsPerLane;

#pragma unroll
    for (int local_row = 0; local_row < ROWS_PER_WARP; ++local_row) {
      const std::size_t row = row0 + static_cast<std::size_t>(local_row);
      if (row >= rows) break;
      const uint2 codes = *reinterpret_cast<const uint2*>(
          packed + row * code_row_bytes + code_offset);
      const float coefficient =
          decode_e4m3_scalar_device(scales[row * scale_row_bytes + group]) * tensor;
      const std::uint32_t words[2] = {codes.x, codes.y};
#pragma unroll
      for (int word = 0; word < 2; ++word) {
#pragma unroll
        for (int byte_in_word = 0; byte_in_word < 4; ++byte_in_word) {
          const int pair = word * 4 + byte_in_word;
          const std::uint8_t packed_byte =
              static_cast<std::uint8_t>(words[word] >> (8 * byte_in_word));
          const float2 decoded = decode_e2m1x2_device(packed_byte);
          constexpr int kChainMask = CHAINS - 1;
          const int chain_a = (2 * pair) & kChainMask;
          const int chain_b = (2 * pair + 1) & kChainMask;
          accumulators[local_row][chain_a] =
              fmaf(decoded.x * coefficient, activations[2 * pair], accumulators[local_row][chain_a]);
          accumulators[local_row][chain_b] = fmaf(decoded.y * coefficient,
                                                  activations[2 * pair + 1],
                                                  accumulators[local_row][chain_b]);
        }
      }
    }
  }

#pragma unroll
  for (int local_row = 0; local_row < ROWS_PER_WARP; ++local_row) {
    const std::size_t row = row0 + static_cast<std::size_t>(local_row);
    if (row >= rows) break;
    float total = 0.0F;
#pragma unroll
    for (int chain = 0; chain < CHAINS; ++chain) total += accumulators[local_row][chain];
    for (int offset = 16; offset > 0; offset >>= 1) {
      total += __shfl_down_sync(0xFFFFFFFFU, total, offset);
    }
    if (lane == 0) output[row] = total;
  }
}

/**
 * E0a specialized small-output FP32 control projection (S04 performance reset, D-022).
 *
 * The incumbent `linear_f32` assigns one thread per output row and iterates the whole K serially at
 * one FMA per step, so the 96 GDN control projections (48x5120 each) cost ~18.5 ms/token. This
 * kernel assigns multiple rows per warp, vectorizes both the weight and the (shared) activation
 * loads as float4, and reduces with warp shuffles.
 */
template <int WARPS_PER_CTA, int ROWS_PER_WARP>
__global__ __launch_bounds__(WARPS_PER_CTA * 32, 4) void linear_f32_rows_kernel(
    const float* __restrict__ input, const float* __restrict__ weights, float* __restrict__ output,
    std::size_t inputs, std::size_t rows) {
  constexpr int kRowsPerCta = WARPS_PER_CTA * ROWS_PER_WARP;
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const std::size_t row0 = static_cast<std::size_t>(blockIdx.x) * kRowsPerCta +
                           static_cast<std::size_t>(warp) * ROWS_PER_WARP;
  if (row0 >= rows) return;
  const std::size_t quads = inputs / 4U;

  float accumulators[ROWS_PER_WARP] = {};
  for (std::size_t quad = static_cast<std::size_t>(lane); quad < quads; quad += 32U) {
    const float4 activation = *reinterpret_cast<const float4*>(input + quad * 4U);
#pragma unroll
    for (int local_row = 0; local_row < ROWS_PER_WARP; ++local_row) {
      const std::size_t row = row0 + static_cast<std::size_t>(local_row);
      if (row >= rows) break;
      const float4 weight = *reinterpret_cast<const float4*>(weights + row * inputs + quad * 4U);
      accumulators[local_row] =
          fmaf(weight.x, activation.x,
               fmaf(weight.y, activation.y,
                    fmaf(weight.z, activation.z,
                         fmaf(weight.w, activation.w, accumulators[local_row]))));
    }
  }
#pragma unroll
  for (int local_row = 0; local_row < ROWS_PER_WARP; ++local_row) {
    const std::size_t row = row0 + static_cast<std::size_t>(local_row);
    if (row >= rows) break;
    float total = accumulators[local_row];
    for (int offset = 16; offset > 0; offset >>= 1) {
      total += __shfl_down_sync(0xFFFFFFFFU, total, offset);
    }
    if (lane == 0) output[row] = total;
  }
}

// ============================================================================================
// S04-P8R-Q experimental native path: SM120 block-scaled NVFP4 tensor-core GEMV
// (`mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3`).
//
// The weight operand keeps the existing row-major `.sinf` layout. The activation is dynamically
// quantised to block-16 E2M1 codes plus UE4M3 scales using the pinned `amax/6` rule. The P7 software
// kernels remain the oracle and the fallback; this path is selected at specialization time and never
// branches on model identity or environment inside the hot path.
// ============================================================================================

__device__ inline std::uint8_t encode_e2m1_device(float value) {
  constexpr float magnitudes[8] = {0.0F, 0.5F, 1.0F, 1.5F, 2.0F, 3.0F, 4.0F, 6.0F};
  const float magnitude = fabsf(value);
  int best = 0;
  float best_delta = 1.0e30F;
  for (int index = 0; index < 8; ++index) {
    const float delta = fabsf(magnitude - magnitudes[index]);
    if (delta < best_delta) {
      best_delta = delta;
      best = index;
    }
  }
  return static_cast<std::uint8_t>(value < 0.0F ? (best | 0x08) : best);
}

__device__ inline std::uint8_t encode_e4m3fn_device(float value) {
  if (!(value > 0.0F)) return 0;
  int exponent = static_cast<int>(floorf(log2f(value))) + 7;
  if (exponent < 1) exponent = 1;
  if (exponent > 15) return 0x7FU;
  const float base = exp2f(static_cast<float>(exponent) - 7.0F);
  int mantissa = static_cast<int>(lroundf((value / base - 1.0F) * 8.0F));
  if (mantissa >= 8) {
    mantissa = 0;
    if (++exponent > 15) return 0x7FU;
  }
  if (mantissa < 0) mantissa = 0;
  return static_cast<std::uint8_t>((exponent << 3) | mantissa);
}

/** Accumulating M16N8K64 block-scaled NVFP4 MMA. `d` is both C and D (no separate zero C). */
__device__ inline void mma_mxf4_device(const std::uint32_t a[4], const std::uint32_t b[2],
                                       std::uint32_t scale_a, std::uint32_t scale_b, float d[4]) {
  const float c0 = d[0], c1 = d[1], c2 = d[2], c3 = d[3];
  asm volatile(
      "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X."
      "f32.e2m1.e2m1.f32.ue4m3 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13}, "
      "%14, {%15, %16}, %17, {%18, %19};\n"
      : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "f"(c0), "f"(c1),
        "f"(c2), "f"(c3), "r"(scale_a), "h"(static_cast<std::uint16_t>(0)),
        "h"(static_cast<std::uint16_t>(0)), "r"(scale_b), "h"(static_cast<std::uint16_t>(0)),
        "h"(static_cast<std::uint16_t>(0)));
}

/**
 * Dynamic block-16 activation quantisation: `input[input_elements]` (f32) -> packed E2M1 codes
 * `packed[input_elements/2]` (low nibble = even column) plus UE4M3 scales `scales[input_elements/16]`.
 * The scale is `amax/6` encoded as UE4M3, the pinned P8-R reference rule.
 */
__global__ inline void nvfp4_activation_quantize_f32(const float* input, std::uint8_t* packed,
                                                     std::uint8_t* scales,
                                                     std::size_t input_elements) {
  const std::size_t block = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (block * 16U >= input_elements) return;
  const float* values = input + block * 16U;
  float amax = 0.0F;
  for (int index = 0; index < 16; ++index) amax = fmaxf(amax, fabsf(values[index]));
  const std::uint8_t scale_code = encode_e4m3fn_device(amax / 6.0F);
  const float scale = decode_e4m3fn_device(scale_code);
  const float inverse = scale > 0.0F ? 1.0F / scale : 0.0F;
  scales[block] = scale_code;
  for (int index = 0; index < 8; ++index) {
    const float low = fminf(fmaxf(values[index * 2] * inverse, -6.0F), 6.0F);
    const float high = fminf(fmaxf(values[index * 2 + 1] * inverse, -6.0F), 6.0F);
    packed[block * 8U + index] =
        static_cast<std::uint8_t>((encode_e2m1_device(high) << 4) | encode_e2m1_device(low));
  }
}

/**
 * Reduce the global amax of one activation vector (single-token decode: one FP32 global scale).
 * One block; shared-memory tree reduction. Deterministic.
 */
__global__ inline void nvfp4_activation_global_amax_f32(const float* input,
                                                        std::size_t input_elements,
                                                        float* global_amax) {
  __shared__ float scratch[256];
  float local = 0.0F;
  for (std::size_t index = threadIdx.x; index < input_elements; index += blockDim.x) {
    local = fmaxf(local, fabsf(input[index]));
  }
  scratch[threadIdx.x] = local;
  __syncthreads();
  for (int stride = static_cast<int>(blockDim.x) / 2; stride > 0; stride >>= 1) {
    if (static_cast<int>(threadIdx.x) < stride) {
      scratch[threadIdx.x] = fmaxf(scratch[threadIdx.x], scratch[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) *global_amax = scratch[0];
}

/**
 * Canonical two-level (hierarchical) NVFP4 activation quantization:
 *   x ~= q_e2m1 * s_block_e4m3 * s_global_f32
 *   s_global      = global_amax / (448 * 6)          [FP32, one value per activation vector]
 *   s_block_real  = (block_amax / 6) / s_global
 *   s_block       = round_E4M3(s_block_real)         [clamped to the E4M3 normal range]
 *   q             = round_E2M1(x / (s_global * s_block))
 * Zero global amax, zero/denormal block amax are handled explicitly and deterministically.
 */
__global__ inline void nvfp4_activation_quantize_two_level_f32(const float* input,
                                                               std::size_t input_elements,
                                                               const float* global_scale,
                                                               std::uint8_t* packed,
                                                               std::uint8_t* scales) {
  const std::size_t block = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (block * 16U >= input_elements) return;
  const float s_global = *global_scale;
  const float* values = input + block * 16U;
  float amax = 0.0F;
  for (int index = 0; index < 16; ++index) amax = fmaxf(amax, fabsf(values[index]));
  std::uint8_t scale_code = 0;
  float s_block = 0.0F;
  if (s_global > 0.0F && amax > 0.0F) {
    const float block_precision_floor = 1.0e-6F;  // smallest E4M3 normal
    float block_scale = (amax / 6.0F) / s_global;
    if (block_scale < block_precision_floor) block_scale = block_precision_floor;
    if (block_scale > 448.0F) block_scale = 448.0F;
    scale_code = encode_e4m3fn_device(block_scale);
    s_block = decode_e4m3fn_device(scale_code);
  }
  scales[block] = scale_code;
  const float denominator = s_global * s_block;
  const float inverse = denominator > 0.0F ? 1.0F / denominator : 0.0F;
  for (int index = 0; index < 8; ++index) {
    const float low =
        inverse > 0.0F ? fminf(fmaxf(values[index * 2] * inverse, -6.0F), 6.0F) : 0.0F;
    const float high =
        inverse > 0.0F ? fminf(fmaxf(values[index * 2 + 1] * inverse, -6.0F), 6.0F) : 0.0F;
    packed[block * 8U + index] =
        static_cast<std::uint8_t>((encode_e2m1_device(high) << 4) | encode_e2m1_device(low));
  }
}

/**
 * Native NVFP4 projection: one warp per 16 output rows, one M16N8K64 MMA per 64 K.
 *
 * Fragment and scale-ownership contract is the P8-R2-verified one: A reg r -> row g+8*(r&1),
 * K 8*quad+(v%8)+32*(r/2); SFA lane 4g -> row g, lane 4g+1 -> row g+8 (four bytes = four K-blocks);
 * B lane (g,quad) -> column g, K 8*quad.. and 32+8*quad..; SFB lane 4n -> column n. Only output
 * column 0 is consumed (single-token decode); the other seven are unused output.
 * `activation_global` is the FP32 activation global scale (1.0 for the one-level representation).
 */
__global__ inline void nvfp4_linear_mma_f32(const std::uint8_t* packed, const std::uint8_t* scales,
                                            const std::uint8_t* activation_packed,
                                            const std::uint8_t* activation_scales,
                                            const float* tensor_scale,
                                            const float* activation_global, float* output,
                                            std::size_t rows, std::size_t inputs) {
  const int lane = static_cast<int>(threadIdx.x & 31U);
  const int warp = static_cast<int>((blockIdx.x * blockDim.x + threadIdx.x) / 32U);
  const int group = lane >> 2;
  const int quad = lane & 3;
  const std::size_t row0 = static_cast<std::size_t>(warp) * 16U;
  if (row0 >= rows) return;
  // One device-scalar read per warp (broadcast/cached); no host round trip.
  const float tensor = (*tensor_scale) * (activation_global != nullptr ? *activation_global : 1.0F);
  const std::size_t weight_row = inputs / 2U;
  const std::size_t scale_row = inputs / 16U;
  const int scale_owner = (lane & 1) ? (group + 8) : group;
  float d[4] = {0.0F, 0.0F, 0.0F, 0.0F};
  for (std::size_t k = 0; k < inputs; k += 64U) {
    const std::size_t byte_offset = k / 2U;
    const std::size_t block = k / 16U;
    const std::uint32_t a[4] = {
        *reinterpret_cast<const std::uint32_t*>(packed + (row0 + group) * weight_row +
                                                byte_offset + 4U * static_cast<std::size_t>(quad)),
        *reinterpret_cast<const std::uint32_t*>(packed + (row0 + group + 8) * weight_row +
                                                byte_offset + 4U * static_cast<std::size_t>(quad)),
        *reinterpret_cast<const std::uint32_t*>(packed + (row0 + group) * weight_row + byte_offset +
                                                16U + 4U * static_cast<std::size_t>(quad)),
        *reinterpret_cast<const std::uint32_t*>(packed + (row0 + group + 8) * weight_row +
                                                byte_offset + 16U +
                                                4U * static_cast<std::size_t>(quad))};
    const std::uint32_t scale_a =
        *reinterpret_cast<const std::uint32_t*>(scales + (row0 + scale_owner) * scale_row + block);
    std::uint32_t b0 = 0;
    std::uint32_t b1 = 0;
    std::uint32_t scale_b = 0;
    if (group == 0) {
      b0 = *reinterpret_cast<const std::uint32_t*>(activation_packed + byte_offset +
                                                   4U * static_cast<std::size_t>(quad));
      b1 = *reinterpret_cast<const std::uint32_t*>(activation_packed + byte_offset + 16U +
                                                   4U * static_cast<std::size_t>(quad));
    }
    if (lane == 0) {
      scale_b = *reinterpret_cast<const std::uint32_t*>(activation_scales + block);
    }
    const std::uint32_t b[2] = {b0, b1};
    mma_mxf4_device(a, b, scale_a, scale_b, d);
  }
  if (quad == 0) {
    if (row0 + static_cast<std::size_t>(group) < rows) {
      output[row0 + static_cast<std::size_t>(group)] = d[0] * tensor;
    }
    if (row0 + static_cast<std::size_t>(group) + 8U < rows) {
      output[row0 + static_cast<std::size_t>(group) + 8U] = d[2] * tensor;
    }
  }
}

__global__ inline void grouped_attention_f32(const float* query, const float* keys,
                                             const float* values, float* output,
                                             std::size_t query_heads, std::size_t kv_heads,
                                             std::size_t head_dimension, std::size_t positions) {
  const std::size_t group = query_heads / kv_heads;
  const float scale = rsqrtf(static_cast<float>(head_dimension));
  for (std::size_t query_head = blockIdx.x * blockDim.x + threadIdx.x;
       query_head < query_heads; query_head += blockDim.x * gridDim.x) {
    const std::size_t kv_head = query_head / group;
    float maximum = -3.402823466e+38F;
    for (std::size_t position = 0; position < positions; ++position) {
      float score = 0.0F;
      for (std::size_t dimension = 0; dimension < head_dimension; ++dimension) {
        score += query[query_head * head_dimension + dimension] *
                 keys[(position * kv_heads + kv_head) * head_dimension + dimension];
      }
      maximum = fmaxf(maximum, score * scale);
    }
    float denominator = 0.0F;
    for (std::size_t position = 0; position < positions; ++position) {
      float score = 0.0F;
      for (std::size_t dimension = 0; dimension < head_dimension; ++dimension) {
        score += query[query_head * head_dimension + dimension] *
                 keys[(position * kv_heads + kv_head) * head_dimension + dimension];
      }
      denominator += expf(score * scale - maximum);
    }
    for (std::size_t dimension = 0; dimension < head_dimension; ++dimension) {
      float attended = 0.0F;
      for (std::size_t position = 0; position < positions; ++position) {
        float score = 0.0F;
        for (std::size_t score_dimension = 0; score_dimension < head_dimension;
             ++score_dimension) {
          score += query[query_head * head_dimension + score_dimension] *
                   keys[(position * kv_heads + kv_head) * head_dimension + score_dimension];
        }
        const float probability = expf(score * scale - maximum) / denominator;
        attended += probability *
                    values[(position * kv_heads + kv_head) * head_dimension + dimension];
      }
      output[query_head * head_dimension + dimension] = attended;
    }
  }
}

/** Reference-correct recurrent gated-delta attention with an in-place FP32 state matrix. */
__global__ inline void gated_delta_attention_f32(
    const float* query, const float* keys, const float* values, const float* log_decay,
    const float* beta, float* state, float* output, std::size_t key_heads,
    std::size_t value_heads, std::size_t key_dimension, std::size_t value_dimension,
    std::size_t positions) {
  const std::size_t head = blockIdx.x * blockDim.x + threadIdx.x;
  if (head >= value_heads) return;
  const std::size_t heads_per_value = value_heads / key_heads;
  const std::size_t key_head = head / heads_per_value;
  const std::size_t state_base = head * key_dimension * value_dimension;
  const float scale = rsqrtf(static_cast<float>(key_dimension));
  for (std::size_t position = 0; position < positions; ++position) {
    const float decay = expf(log_decay[position * value_heads + head]);
    for (std::size_t key_index = 0; key_index < key_dimension; ++key_index) {
      for (std::size_t value_index = 0; value_index < value_dimension; ++value_index) {
        state[state_base + key_index * value_dimension + value_index] *= decay;
      }
    }
    const std::size_t query_base = (position * key_heads + key_head) * key_dimension;
    float query_norm = 0.0F;
    float key_norm = 0.0F;
    for (std::size_t key_index = 0; key_index < key_dimension; ++key_index) {
      query_norm += query[query_base + key_index] * query[query_base + key_index];
      key_norm += keys[query_base + key_index] * keys[query_base + key_index];
    }
    const float query_scale = rsqrtf(query_norm + 1.0e-6F);
    const float key_scale = rsqrtf(key_norm + 1.0e-6F);
    const float beta_value = beta[position * value_heads + head];
    const std::size_t value_base = (position * value_heads + head) * value_dimension;
    for (std::size_t value_index = 0; value_index < value_dimension; ++value_index) {
      float key_value = 0.0F;
      for (std::size_t key_index = 0; key_index < key_dimension; ++key_index) {
        key_value += state[state_base + key_index * value_dimension + value_index] *
                     (keys[query_base + key_index] * key_scale);
      }
      const float delta = (values[value_base + value_index] - key_value) * beta_value;
      for (std::size_t key_index = 0; key_index < key_dimension; ++key_index) {
        state[state_base + key_index * value_dimension + value_index] +=
            (keys[query_base + key_index] * key_scale) * delta;
      }
    }
    const std::size_t output_base = value_base;
    for (std::size_t value_index = 0; value_index < value_dimension; ++value_index) {
      float result = 0.0F;
      for (std::size_t key_index = 0; key_index < key_dimension; ++key_index) {
        result += state[state_base + key_index * value_dimension + value_index] *
                  (query[query_base + key_index] * query_scale);
      }
      output[output_base + value_index] = result * scale;
    }
  }
}

/** Gated DeltaNet with value-dimension parallelism (S04-P3).
 *
 * Bit-identical to `gated_delta_attention_f32`: each `value_index` is an
 * independent recurrence column, so assigning one thread per (value head,
 * value index) preserves every summation and update order. Only the head-level
 * scalars (norms, scales, decay, beta) are shared to avoid redundant work; they
 * are computed with the identical expressions.
 */
__global__ inline void gated_delta_attention_parallel_f32(
    const float* query, const float* keys, const float* values, const float* log_decay,
    const float* beta, float* state, float* output, std::size_t key_heads,
    std::size_t value_heads, std::size_t key_dimension, std::size_t value_dimension,
    std::size_t positions) {
  const std::size_t head = blockIdx.x;
  if (head >= value_heads) return;
  const std::size_t heads_per_value = value_heads / key_heads;
  const std::size_t key_head = head / heads_per_value;
  const std::size_t state_base = head * key_dimension * value_dimension;
  const float scale = rsqrtf(static_cast<float>(key_dimension));
  __shared__ float shared_query_scale;
  __shared__ float shared_key_scale;
  __shared__ float shared_beta;
  __shared__ float shared_decay;
  for (std::size_t position = 0; position < positions; ++position) {
    const std::size_t query_base = (position * key_heads + key_head) * key_dimension;
    const std::size_t value_base = (position * value_heads + head) * value_dimension;
    if (threadIdx.x == 0) {
      float query_norm = 0.0F;
      float key_norm = 0.0F;
      for (std::size_t key_index = 0; key_index < key_dimension; ++key_index) {
        query_norm += query[query_base + key_index] * query[query_base + key_index];
        key_norm += keys[query_base + key_index] * keys[query_base + key_index];
      }
      shared_query_scale = rsqrtf(query_norm + 1.0e-6F);
      shared_key_scale = rsqrtf(key_norm + 1.0e-6F);
      shared_beta = beta[position * value_heads + head];
      shared_decay = expf(log_decay[position * value_heads + head]);
    }
    __syncthreads();
    const float query_scale = shared_query_scale;
    const float key_scale = shared_key_scale;
    const float beta_value = shared_beta;
    const float decay = shared_decay;
    for (std::size_t value_index = threadIdx.x; value_index < value_dimension;
         value_index += blockDim.x) {
      for (std::size_t key_index = 0; key_index < key_dimension; ++key_index) {
        state[state_base + key_index * value_dimension + value_index] *= decay;
      }
      float key_value = 0.0F;
      for (std::size_t key_index = 0; key_index < key_dimension; ++key_index) {
        key_value += state[state_base + key_index * value_dimension + value_index] *
                     (keys[query_base + key_index] * key_scale);
      }
      const float delta = (values[value_base + value_index] - key_value) * beta_value;
      for (std::size_t key_index = 0; key_index < key_dimension; ++key_index) {
        state[state_base + key_index * value_dimension + value_index] +=
            (keys[query_base + key_index] * key_scale) * delta;
      }
      float result = 0.0F;
      for (std::size_t key_index = 0; key_index < key_dimension; ++key_index) {
        result += state[state_base + key_index * value_dimension + value_index] *
                  (query[query_base + key_index] * query_scale);
      }
      output[value_base + value_index] = result * scale;
    }
    __syncthreads();
  }
}

/** RMSNorm with BF16 scale, row-parallel (S04-P4).
 *
 * Bit-identical to `rms_norm_f32_bf16_scale`: the sum of squares is still
 * accumulated sequentially in the original element order (thread 0 over a
 * shared-memory staging of the row), so the denominator is bit-for-bit the
 * same. The staging load and the output write are parallelised. One block per
 * row.
 */
__global__ inline void rms_norm_f32_bf16_scale_parallel(
    const float* input, const std::uint16_t* scale, float* output, std::size_t elements,
    std::size_t scale_elements, float epsilon, bool add_one_to_scale) {
  __shared__ float row_values[8192];
  __shared__ float denominator_slot;
  __shared__ float warp_partials[32];
  const std::size_t rows = elements / scale_elements;
  const std::size_t row = blockIdx.x;
  if (row >= rows) return;
  const float* input_row = input + row * scale_elements;
  float* output_row = output + row * scale_elements;
  // S04 performance reset E0b: the sum of squares used to be computed by a single thread
  // (threadIdx.x == 0) serially over the whole row, which dominated this kernel (~25 us/launch).
  // Each thread now accumulates a strided partial and the block reduces it. Reduction order changes,
  // so this is tolerance-qualified rather than bit-identical.
  float sum_squares = 0.0F;
  for (std::size_t index = threadIdx.x; index < scale_elements; index += blockDim.x) {
    const float value = input_row[index];
    row_values[index] = value;
    sum_squares = fmaf(value, value, sum_squares);
  }
  __syncthreads();
  for (int offset = 16; offset > 0; offset >>= 1) {
    sum_squares += __shfl_down_sync(0xFFFFFFFFU, sum_squares, offset);
  }
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  if ((threadIdx.x & 31) == 0) warp_partials[warp] = sum_squares;
  __syncthreads();
  if (threadIdx.x == 0) {
    float total = 0.0F;
    for (int index = 0; index < static_cast<int>(blockDim.x >> 5); ++index) {
      total += warp_partials[index];
    }
    denominator_slot = sqrtf(total / static_cast<float>(scale_elements) + epsilon);
  }
  __syncthreads();
  const float denominator = denominator_slot;
  const float offset = add_one_to_scale ? 1.0F : 0.0F;
  for (std::size_t index = threadIdx.x; index < scale_elements; index += blockDim.x) {
    output_row[index] = row_values[index] / denominator * (bf16_to_float_device(scale[index]) + offset);
  }
}

__global__ inline void rms_norm_f32(const float* input, const float* scale, float* output,
                                    std::size_t elements, std::size_t scale_elements,
                                    float epsilon, bool add_one_to_scale) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  for (std::size_t row = 0; row < elements / scale_elements; ++row) {
    float sum_squares = 0.0F;
    for (std::size_t index = 0; index < scale_elements; ++index) {
      const float value = input[row * scale_elements + index];
      sum_squares += value * value;
    }
    const float denominator = sqrtf(sum_squares / static_cast<float>(scale_elements) + epsilon);
    for (std::size_t index = 0; index < scale_elements; ++index) {
      output[row * scale_elements + index] =
          input[row * scale_elements + index] / denominator *
          (scale[index] + (add_one_to_scale ? 1.0F : 0.0F));
    }
  }
}

__global__ inline void rms_norm_f32_bf16_scale(const float* input, const std::uint16_t* scale,
                                               float* output, std::size_t elements,
                                               std::size_t scale_elements, float epsilon,
                                               bool add_one_to_scale) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  for (std::size_t row = 0; row < elements / scale_elements; ++row) {
    float sum_squares = 0.0F;
    for (std::size_t index = 0; index < scale_elements; ++index) {
      const float value = input[row * scale_elements + index];
      sum_squares += value * value;
    }
    const float denominator = sqrtf(sum_squares / static_cast<float>(scale_elements) + epsilon);
    for (std::size_t index = 0; index < scale_elements; ++index) {
      output[row * scale_elements + index] =
          input[row * scale_elements + index] / denominator *
          (bf16_to_float_device(scale[index]) + (add_one_to_scale ? 1.0F : 0.0F));
    }
  }
}

__global__ inline void layer_norm_f32(const float* input, const float* scale, const float* bias,
                                      float* output, std::size_t elements, float epsilon) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  float mean = 0.0F;
  for (std::size_t index = 0; index < elements; ++index) mean += input[index];
  mean /= static_cast<float>(elements);
  float variance = 0.0F;
  for (std::size_t index = 0; index < elements; ++index) {
    const float centered = input[index] - mean;
    variance += centered * centered;
  }
  const float denominator = sqrtf(variance / static_cast<float>(elements) + epsilon);
  for (std::size_t index = 0; index < elements; ++index) {
    output[index] = (input[index] - mean) / denominator * scale[index] + bias[index];
  }
}

using LaunchFunction = cudaError_t (*)(const ir::physical::CommandDescriptor&,
                                       const ir::physical::Plan&, void*, void*, cudaStream_t);

inline void* buffer_pointer(const ir::physical::Plan& plan, void* arena,
                            ir::physical::BufferId id) {
  return static_cast<std::byte*>(arena) + static_cast<std::size_t>(plan.buffers()[id.value()].offset);
}

inline cudaError_t launch_copy(const ir::physical::CommandDescriptor& command,
                               const ir::physical::Plan& plan, void* arena, void*,
                               cudaStream_t stream) {
  if (command.buffers.size() != 2) return cudaErrorInvalidValue;
  const auto& source = plan.buffers()[command.buffers[0].value()];
  const auto& destination = plan.buffers()[command.buffers[1].value()];
  const std::uint64_t bytes = source.size;
  return cudaMemcpyAsync(buffer_pointer(plan, arena, destination.id),
                         buffer_pointer(plan, arena, source.id), static_cast<std::size_t>(bytes),
                         cudaMemcpyDeviceToDevice, stream);
}

inline cudaError_t launch_embedding(const ir::physical::CommandDescriptor& command,
                                    const ir::physical::Plan& plan, void* arena, void*,
                                    cudaStream_t stream) {
  if (command.buffers.size() != 3) return cudaErrorInvalidValue;
  const auto& token = plan.buffers()[command.buffers[0].value()];
  const auto& table = plan.buffers()[command.buffers[1].value()];
  const auto& output = plan.buffers()[command.buffers[2].value()];
  if (token.size != sizeof(std::uint32_t) || output.size == 0 || output.size % sizeof(float) != 0 ||
      table.size == 0 || table.size % output.size != 0) {
    return cudaErrorInvalidValue;
  }
  const std::size_t hidden = static_cast<std::size_t>(output.size / sizeof(float));
  const std::size_t vocabulary = static_cast<std::size_t>(table.size / output.size);
  embedding_f32<<<1, 256, 0, stream>>>(
      static_cast<const std::uint32_t*>(buffer_pointer(plan, arena, token.id)),
      static_cast<const float*>(buffer_pointer(plan, arena, table.id)),
      static_cast<float*>(buffer_pointer(plan, arena, output.id)), vocabulary, hidden);
  return cudaGetLastError();
}

inline cudaError_t launch_embedding_bf16(const ir::physical::CommandDescriptor& command,
                                         const ir::physical::Plan& plan, void* arena, void*,
                                         cudaStream_t stream) {
  if (command.buffers.size() != 3) return cudaErrorInvalidValue;
  const auto& token = plan.buffers()[command.buffers[0].value()];
  const auto& table = plan.buffers()[command.buffers[1].value()];
  const auto& output = plan.buffers()[command.buffers[2].value()];
  if (token.size != sizeof(std::uint32_t) || output.size == 0 || output.size % sizeof(float) != 0 ||
      table.size == 0 || table.size % sizeof(std::uint16_t) != 0 ||
      (table.size / sizeof(std::uint16_t)) % (output.size / sizeof(float)) != 0) {
    return cudaErrorInvalidValue;
  }
  const std::size_t hidden = static_cast<std::size_t>(output.size / sizeof(float));
  const std::size_t vocabulary = static_cast<std::size_t>(table.size / (hidden * sizeof(std::uint16_t)));
  embedding_bf16<<<1, 256, 0, stream>>>(
      static_cast<const std::uint32_t*>(buffer_pointer(plan, arena, token.id)),
      static_cast<const std::uint16_t*>(buffer_pointer(plan, arena, table.id)),
      static_cast<float*>(buffer_pointer(plan, arena, output.id)), vocabulary, hidden);
  return cudaGetLastError();
}

inline cudaError_t launch_cast_bf16_to_f32(const ir::physical::CommandDescriptor& command,
                                           const ir::physical::Plan& plan, void* arena, void*,
                                           cudaStream_t stream) {
  if (command.buffers.size() != 2) return cudaErrorInvalidValue;
  const auto& input = plan.buffers()[command.buffers[0].value()];
  const auto& output = plan.buffers()[command.buffers[1].value()];
  if (input.size == 0 || input.size % sizeof(std::uint16_t) != 0 ||
      output.size != input.size * 2U) return cudaErrorInvalidValue;
  const std::size_t elements = static_cast<std::size_t>(input.size / sizeof(std::uint16_t));
  // S04-P5: the kernel is elementwise and its loop is already grid-stride, so
  // occupying the device is bit-identical; only the block count changes.
  std::uint32_t blocks = static_cast<std::uint32_t>((elements + 255U) / 256U);
  if (blocks == 0U) blocks = 1U;
  if (blocks > 4096U) blocks = 4096U;
  cast_bf16_to_f32<<<blocks, 256, 0, stream>>>(
      static_cast<const std::uint16_t*>(buffer_pointer(plan, arena, input.id)),
      static_cast<float*>(buffer_pointer(plan, arena, output.id)), elements);
  return cudaGetLastError();
}

inline cudaError_t launch_cast_f32_to_bf16(const ir::physical::CommandDescriptor& command,
                                           const ir::physical::Plan& plan, void* arena, void*,
                                           cudaStream_t stream) {
  if (command.buffers.size() != 2) return cudaErrorInvalidValue;
  const auto& input = plan.buffers()[command.buffers[0].value()];
  const auto& output = plan.buffers()[command.buffers[1].value()];
  if (input.size == 0 || input.size % sizeof(float) != 0 ||
      output.size != input.size / 2U) return cudaErrorInvalidValue;
  const std::size_t elements = static_cast<std::size_t>(input.size / sizeof(float));
  std::uint32_t blocks = static_cast<std::uint32_t>((elements + 255U) / 256U);
  if (blocks == 0U) blocks = 1U;
  if (blocks > 4096U) blocks = 4096U;
  cast_f32_to_bf16<<<blocks, 256, 0, stream>>>(
      static_cast<const float*>(buffer_pointer(plan, arena, input.id)),
      static_cast<std::uint16_t*>(buffer_pointer(plan, arena, output.id)), elements);
  return cudaGetLastError();
}

inline cudaError_t launch_nvfp4_dequantize(const ir::physical::CommandDescriptor& command,
                                           const ir::physical::Plan& plan, void* arena, void*,
                                           cudaStream_t stream) {
  if (command.buffers.size() != 3) return cudaErrorInvalidValue;
  const auto& packed = plan.buffers()[command.buffers[0].value()];
  const auto& scales = plan.buffers()[command.buffers[1].value()];
  const auto& output = plan.buffers()[command.buffers[2].value()];
  if (output.size == 0 || output.size % sizeof(float) != 0 || output.size / sizeof(float) % 16 != 0 ||
      packed.size != output.size / sizeof(float) / 2U ||
      scales.size != output.size / sizeof(float) / 16U) {
    return cudaErrorInvalidValue;
  }
  nvfp4_dequantize<<<1, 256, 0, stream>>>(
      static_cast<const std::uint8_t*>(buffer_pointer(plan, arena, packed.id)),
      static_cast<const std::uint8_t*>(buffer_pointer(plan, arena, scales.id)),
      static_cast<float*>(buffer_pointer(plan, arena, output.id)),
      static_cast<std::size_t>(output.size / sizeof(float)), command.scalar);
  return cudaGetLastError();
}

inline cudaError_t launch_lm_head(const ir::physical::CommandDescriptor& command,
                                  const ir::physical::Plan& plan, void* arena, void*,
                                  cudaStream_t stream) {
  if (command.buffers.size() != 3) return cudaErrorInvalidValue;
  const auto& input = plan.buffers()[command.buffers[0].value()];
  const auto& weights = plan.buffers()[command.buffers[1].value()];
  const auto& output = plan.buffers()[command.buffers[2].value()];
  const std::size_t input_elements = static_cast<std::size_t>(input.size / sizeof(float));
  const std::size_t output_elements = static_cast<std::size_t>(output.size / sizeof(float));
  // S04-P5: row-parallel launch (lm_head); each row reduction order is unchanged.
  std::uint32_t blocks = static_cast<std::uint32_t>((output_elements + 255U) / 256U);
  if (blocks == 0U) blocks = 1U;
  if (blocks > 4096U) blocks = 4096U;
  linear_f32<<<blocks, 256, 0, stream>>>(
      static_cast<const float*>(buffer_pointer(plan, arena, input.id)),
      static_cast<const float*>(buffer_pointer(plan, arena, weights.id)),
      static_cast<float*>(buffer_pointer(plan, arena, output.id)), input_elements,
      output_elements);
  return cudaGetLastError();
}

inline cudaError_t launch_gated_dense_ffn(const ir::physical::CommandDescriptor& command,
                                          const ir::physical::Plan& plan, void* arena, void*,
                                          cudaStream_t stream) {
  if (command.buffers.size() != 5) return cudaErrorInvalidValue;
  const auto& input = plan.buffers()[command.buffers[0].value()];
  const auto& gate = plan.buffers()[command.buffers[1].value()];
  const auto& up = plan.buffers()[command.buffers[2].value()];
  const auto& down = plan.buffers()[command.buffers[3].value()];
  const auto& output = plan.buffers()[command.buffers[4].value()];
  const std::size_t hidden = static_cast<std::size_t>(input.size / sizeof(float));
  const std::size_t intermediate = static_cast<std::size_t>(gate.size / sizeof(float) / hidden);
  gated_dense_ffn_f32<<<1, 256, 0, stream>>>(
      static_cast<const float*>(buffer_pointer(plan, arena, input.id)),
      static_cast<const float*>(buffer_pointer(plan, arena, gate.id)),
      static_cast<const float*>(buffer_pointer(plan, arena, up.id)),
      static_cast<const float*>(buffer_pointer(plan, arena, down.id)),
      static_cast<float*>(buffer_pointer(plan, arena, output.id)), hidden, intermediate);
  return cudaGetLastError();
}

inline cudaError_t launch_nvfp4_linear(const ir::physical::CommandDescriptor& command,
                                       const ir::physical::Plan& plan, void* arena, void*,
                                       cudaStream_t stream) {
  if (command.buffers.size() != 5) return cudaErrorInvalidValue;
  const auto& input = plan.buffers()[command.buffers[0].value()];
  const auto& packed = plan.buffers()[command.buffers[1].value()];
  const auto& scales = plan.buffers()[command.buffers[2].value()];
  const auto& tensor_scale = plan.buffers()[command.buffers[3].value()];
  const auto& output = plan.buffers()[command.buffers[4].value()];
  const std::size_t input_elements = static_cast<std::size_t>(input.size / sizeof(float));
  const std::size_t output_elements = static_cast<std::size_t>(output.size / sizeof(float));
  // Row-parallel launch (R02): one 256-thread block per 256 output rows keeps the
  // per-row operation order bit-identical to the single-block baseline while
  // occupying the device. Small shapes collapse to a single block, i.e. the
  // exact baseline mapping.
  std::uint32_t blocks =
      static_cast<std::uint32_t>((output_elements + 255U) / 256U);
  if (blocks == 0U) blocks = 1U;
  if (blocks > 2048U) blocks = 2048U;
  // S04-P2: vectorized/scale-hoisted path when alignment permits; the scalar
  // row-parallel incumbent is the fallback.
  const auto* packed_pointer =
      static_cast<const std::uint8_t*>(buffer_pointer(plan, arena, packed.id));
  const bool vectorizable =
      (input_elements % 32U == 0U) && (reinterpret_cast<std::uintptr_t>(packed_pointer) % 16U == 0U);
  // P7-0 shape-adaptive dispatch (verified in the P6/decode harness):
  // warp-per-row wins on the many small/medium projections, while the
  // row-per-thread vector kernel wins on the two very large ones (LM head and
  // the 17408x5120 MLP). The choice is a deterministic function of the
  // compile-time output size, not a model-name or environment decision.
  const char* warp_selector = std::getenv("SUPERINFER_QWEN38_NVFP4_WARP");
  constexpr std::size_t kWarpMaxRows = 16384U;
  const bool use_warp = vectorizable && output_elements < kWarpMaxRows &&
                        !(warp_selector != nullptr && warp_selector[0] == '0');
  if (use_warp) {
    std::uint32_t warp_blocks =
        static_cast<std::uint32_t>((output_elements * 32U + 255U) / 256U);
    if (warp_blocks == 0U) warp_blocks = 1U;
    nvfp4_linear_warp_f32<<<warp_blocks, 256, 0, stream>>>(
        static_cast<const float*>(buffer_pointer(plan, arena, input.id)), packed_pointer,
        static_cast<const std::uint8_t*>(buffer_pointer(plan, arena, scales.id)),
        static_cast<const float*>(buffer_pointer(plan, arena, tensor_scale.id)),
        static_cast<float*>(buffer_pointer(plan, arena, output.id)), input_elements,
        output_elements);
    return cudaGetLastError();
  }
  if (vectorizable) {
    nvfp4_linear_rows_vec_f32<<<blocks, 256, 0, stream>>>(
        static_cast<const float*>(buffer_pointer(plan, arena, input.id)), packed_pointer,
        static_cast<const std::uint8_t*>(buffer_pointer(plan, arena, scales.id)),
        static_cast<const float*>(buffer_pointer(plan, arena, tensor_scale.id)),
        static_cast<float*>(buffer_pointer(plan, arena, output.id)), input_elements,
        output_elements);
    return cudaGetLastError();
  }
  nvfp4_linear_rows_f32<<<blocks, 256, 0, stream>>>(
      static_cast<const float*>(buffer_pointer(plan, arena, input.id)),
      static_cast<const std::uint8_t*>(buffer_pointer(plan, arena, packed.id)),
      static_cast<const std::uint8_t*>(buffer_pointer(plan, arena, scales.id)),
      static_cast<const float*>(buffer_pointer(plan, arena, tensor_scale.id)),
      static_cast<float*>(buffer_pointer(plan, arena, output.id)), input_elements,
      output_elements);
  return cudaGetLastError();
}

/**
 * E0a donor-scheduled streaming GEMV launch (S04 performance reset, D-022).
 *
 * Consumes the existing row-major packed weights (`[rows][K/2]`), the existing row-major E4M3
 * block-scale plane (`[rows][K/16]`), and the existing FP32 activation buffer. No per-token
 * conversion kernel and no repack: the schedule is what changed.
 */
inline cudaError_t launch_nvfp4_linear_gemv_rows(const ir::physical::CommandDescriptor& command,
                                                 const ir::physical::Plan& plan, void* arena, void*,
                                                 cudaStream_t stream) {
  if (command.buffers.size() != 5) return cudaErrorInvalidValue;
  const auto& input = plan.buffers()[command.buffers[0].value()];
  const auto& packed = plan.buffers()[command.buffers[1].value()];
  const auto& scales = plan.buffers()[command.buffers[2].value()];
  const auto& tensor_scale = plan.buffers()[command.buffers[3].value()];
  const auto& output = plan.buffers()[command.buffers[4].value()];
  const std::size_t input_elements = static_cast<std::size_t>(input.size / sizeof(float));
  const std::size_t output_elements = static_cast<std::size_t>(output.size / sizeof(float));
  constexpr std::uint32_t kWarpsPerCta = 8;
  constexpr std::uint32_t kRowsPerWarp = 4;
  constexpr std::uint32_t kChains = 4;
  constexpr std::uint32_t kRowsPerCta = kWarpsPerCta * kRowsPerWarp;
  if (input_elements == 0U || output_elements == 0U || input_elements % 512U != 0U ||
      output_elements % kRowsPerCta != 0U) {
    return cudaErrorInvalidValue;
  }
  const std::uint32_t blocks = static_cast<std::uint32_t>(output_elements / kRowsPerCta);
  nvfp4_gemv_rows_f32<kWarpsPerCta, kRowsPerWarp, kChains, 2>
      <<<(blocks == 0U ? 1U : blocks), kWarpsPerCta * 32U, 0, stream>>>(
          static_cast<const float*>(buffer_pointer(plan, arena, input.id)),
          static_cast<const std::uint8_t*>(buffer_pointer(plan, arena, packed.id)),
          static_cast<const std::uint8_t*>(buffer_pointer(plan, arena, scales.id)),
          static_cast<const float*>(buffer_pointer(plan, arena, tensor_scale.id)),
          static_cast<float*>(buffer_pointer(plan, arena, output.id)), output_elements,
          input_elements);
  return cudaGetLastError();
}

/**
 * E0a launch for the specialized FP32 control projection (kernel 30).
 */
inline cudaError_t launch_linear_f32_rows(const ir::physical::CommandDescriptor& command,
                                          const ir::physical::Plan& plan, void* arena, void*,
                                          cudaStream_t stream) {
  if (command.buffers.size() != 3) return cudaErrorInvalidValue;
  const auto& input = plan.buffers()[command.buffers[0].value()];
  const auto& weights = plan.buffers()[command.buffers[1].value()];
  const auto& output = plan.buffers()[command.buffers[2].value()];
  const std::size_t input_elements = static_cast<std::size_t>(input.size / sizeof(float));
  const std::size_t output_elements = static_cast<std::size_t>(output.size / sizeof(float));
  constexpr std::uint32_t kWarpsPerCta = 8;
  constexpr std::uint32_t kRowsPerWarp = 2;
  constexpr std::uint32_t kRowsPerCta = kWarpsPerCta * kRowsPerWarp;
  if (input_elements == 0U || output_elements == 0U || input_elements % 4U != 0U) {
    return cudaErrorInvalidValue;
  }
  const std::uint32_t blocks =
      static_cast<std::uint32_t>((output_elements + kRowsPerCta - 1U) / kRowsPerCta);
  linear_f32_rows_kernel<kWarpsPerCta, kRowsPerWarp>
      <<<(blocks == 0U ? 1U : blocks), kWarpsPerCta * 32U, 0, stream>>>(
          static_cast<const float*>(buffer_pointer(plan, arena, input.id)),
          static_cast<const float*>(buffer_pointer(plan, arena, weights.id)),
          static_cast<float*>(buffer_pointer(plan, arena, output.id)), input_elements,
          output_elements);
  return cudaGetLastError();
}

/**
 * S04-P8R-Q experimental native NVFP4 projection launch.
 *
 * Quantises the activation into the session workspace (allocated once; no allocation here) and runs
 * the warp-per-16-rows MMA GEMV. Requires the contract the provider advertises: sm_120a,
 * `inputs % 64 == 0`, `rows % 16 == 0` (guarded below; the provider only selects it when satisfied).
 * The plan is single-stream ordered, so the shared scratch cannot be aliased by a concurrent command.
 */
inline cudaError_t launch_nvfp4_linear_mma(const ir::physical::CommandDescriptor& command,
                                           const ir::physical::Plan& plan, void* arena,
                                           void* workspace, cudaStream_t stream) {
  if (command.buffers.size() != 5) return cudaErrorInvalidValue;
  const auto& input = plan.buffers()[command.buffers[0].value()];
  const auto& packed = plan.buffers()[command.buffers[1].value()];
  const auto& scales = plan.buffers()[command.buffers[2].value()];
  const auto& tensor_scale = plan.buffers()[command.buffers[3].value()];
  const auto& output = plan.buffers()[command.buffers[4].value()];
  const std::size_t input_elements = static_cast<std::size_t>(input.size / sizeof(float));
  const std::size_t output_elements = static_cast<std::size_t>(output.size / sizeof(float));
  if (input_elements == 0U || output_elements == 0U || input_elements % 64U != 0U) {
    return cudaErrorInvalidValue;
  }
  if (workspace == nullptr) return cudaErrorInvalidValue;
  auto* activation_packed = static_cast<std::uint8_t*>(workspace);
  const std::size_t activation_packed_bytes = ((input_elements / 2U) + 15U) / 16U * 16U;
  auto* activation_scales = activation_packed + activation_packed_bytes;
  const float* input_pointer = static_cast<const float*>(buffer_pointer(plan, arena, input.id));
  const std::uint32_t quantize_blocks =
      static_cast<std::uint32_t>((input_elements / 16U + 255U) / 256U);
  nvfp4_activation_quantize_f32<<<(quantize_blocks == 0U ? 1U : quantize_blocks), 256, 0, stream>>>(
      input_pointer, activation_packed, activation_scales, input_elements);
  const std::uint32_t warps = static_cast<std::uint32_t>((output_elements + 15U) / 16U);
  const std::uint32_t blocks = (warps * 32U + 255U) / 256U;
  nvfp4_linear_mma_f32<<<(blocks == 0U ? 1U : blocks), 256, 0, stream>>>(
      static_cast<const std::uint8_t*>(buffer_pointer(plan, arena, packed.id)),
      static_cast<const std::uint8_t*>(buffer_pointer(plan, arena, scales.id)), activation_packed,
      activation_scales, static_cast<const float*>(buffer_pointer(plan, arena, tensor_scale.id)),
      nullptr, static_cast<float*>(buffer_pointer(plan, arena, output.id)), output_elements,
      input_elements);
  return cudaGetLastError();
}

/**
 * S04-P9-1 canonical two-level activation scaling launch (kernel 28).
 *
 * Two-phase: reduce the activation global amax, form `s_global = global_amax/(448*6)`, then quantize
 * each block with `s_block = round_E4M3((block_amax/6)/s_global)`. The MMA consumes E2M1 + UE4M3 as
 * before; the FP32 `s_global` is applied in the epilogue together with the weight tensor scale.
 * Workspace layout: [packed | scales | global_scale(f32)].
 */
inline cudaError_t launch_nvfp4_linear_mma_two_level(const ir::physical::CommandDescriptor& command,
                                                     const ir::physical::Plan& plan, void* arena,
                                                     void* workspace, cudaStream_t stream) {
  if (command.buffers.size() != 5) return cudaErrorInvalidValue;
  const auto& input = plan.buffers()[command.buffers[0].value()];
  const auto& packed = plan.buffers()[command.buffers[1].value()];
  const auto& scales = plan.buffers()[command.buffers[2].value()];
  const auto& tensor_scale = plan.buffers()[command.buffers[3].value()];
  const auto& output = plan.buffers()[command.buffers[4].value()];
  const std::size_t input_elements = static_cast<std::size_t>(input.size / sizeof(float));
  const std::size_t output_elements = static_cast<std::size_t>(output.size / sizeof(float));
  if (input_elements == 0U || output_elements == 0U || input_elements % 64U != 0U) {
    return cudaErrorInvalidValue;
  }
  if (workspace == nullptr) return cudaErrorInvalidValue;
  auto* activation_packed = static_cast<std::uint8_t*>(workspace);
  const std::size_t activation_packed_bytes = ((input_elements / 2U) + 15U) / 16U * 16U;
  auto* activation_scales = activation_packed + activation_packed_bytes;
  const std::size_t activation_scales_bytes = ((input_elements / 16U) + 15U) / 16U * 16U;
  auto* activation_global = reinterpret_cast<float*>(activation_scales + activation_scales_bytes);
  const float* input_pointer = static_cast<const float*>(buffer_pointer(plan, arena, input.id));
  nvfp4_activation_global_amax_f32<<<1, 256, 0, stream>>>(input_pointer, input_elements,
                                                          activation_global);
  const std::uint32_t quantize_blocks =
      static_cast<std::uint32_t>((input_elements / 16U + 255U) / 256U);
  nvfp4_activation_quantize_two_level_f32<<<(quantize_blocks == 0U ? 1U : quantize_blocks), 256, 0,
                                            stream>>>(input_pointer, input_elements,
                                                      activation_global, activation_packed,
                                                      activation_scales);
  const std::uint32_t warps = static_cast<std::uint32_t>((output_elements + 15U) / 16U);
  const std::uint32_t blocks = (warps * 32U + 255U) / 256U;
  nvfp4_linear_mma_f32<<<(blocks == 0U ? 1U : blocks), 256, 0, stream>>>(
      static_cast<const std::uint8_t*>(buffer_pointer(plan, arena, packed.id)),
      static_cast<const std::uint8_t*>(buffer_pointer(plan, arena, scales.id)), activation_packed,
      activation_scales, static_cast<const float*>(buffer_pointer(plan, arena, tensor_scale.id)),
      activation_global, static_cast<float*>(buffer_pointer(plan, arena, output.id)), output_elements,
      input_elements);
  return cudaGetLastError();
}

inline cudaError_t launch_attention(const ir::physical::CommandDescriptor& command,
                                    const ir::physical::Plan& plan, void* arena, void*,
                                    cudaStream_t stream) {
  if (command.buffers.size() != 4) return cudaErrorInvalidValue;
  const auto& query = plan.buffers()[command.buffers[0].value()];
  const auto& keys = plan.buffers()[command.buffers[1].value()];
  const auto& values = plan.buffers()[command.buffers[2].value()];
  const auto& output = plan.buffers()[command.buffers[3].value()];
  const auto dimensions = command.attention;
  grouped_attention_f32<<<1, 256, 0, stream>>>(
      static_cast<const float*>(buffer_pointer(plan, arena, query.id)),
      static_cast<const float*>(buffer_pointer(plan, arena, keys.id)),
      static_cast<const float*>(buffer_pointer(plan, arena, values.id)),
      static_cast<float*>(buffer_pointer(plan, arena, output.id)), dimensions.query_heads,
      dimensions.key_value_heads, dimensions.head_dimension, dimensions.positions);
  return cudaGetLastError();
}

inline cudaError_t launch_gated_delta_attention(
    const ir::physical::CommandDescriptor& command, const ir::physical::Plan& plan, void* arena,
    void*, cudaStream_t stream) {
  if (command.buffers.size() != 7) return cudaErrorInvalidValue;
  const auto dimensions = command.attention;
  // S04-P3: parallelize across value heads/dimensions when the shape permits;
  // the single-block incumbent is the fallback.
  const std::size_t block = dimensions.value_dimension == 0
                                ? 0
                                : (dimensions.value_dimension < 128
                                       ? dimensions.value_dimension
                                       : (dimensions.value_dimension < 256 ? 128U : 256U));
  if (dimensions.value_heads != 0 && dimensions.key_value_heads != 0 && block != 0 &&
      dimensions.value_heads <= 65535U) {
    gated_delta_attention_parallel_f32<<<static_cast<std::uint32_t>(dimensions.value_heads),
                                        static_cast<std::uint32_t>(block), 0, stream>>>(
        static_cast<const float*>(buffer_pointer(plan, arena, plan.buffers()[command.buffers[0].value()].id)),
        static_cast<const float*>(buffer_pointer(plan, arena, plan.buffers()[command.buffers[1].value()].id)),
        static_cast<const float*>(buffer_pointer(plan, arena, plan.buffers()[command.buffers[2].value()].id)),
        static_cast<const float*>(buffer_pointer(plan, arena, plan.buffers()[command.buffers[3].value()].id)),
        static_cast<const float*>(buffer_pointer(plan, arena, plan.buffers()[command.buffers[4].value()].id)),
        static_cast<float*>(buffer_pointer(plan, arena, plan.buffers()[command.buffers[5].value()].id)),
        static_cast<float*>(buffer_pointer(plan, arena, plan.buffers()[command.buffers[6].value()].id)),
        dimensions.key_value_heads, dimensions.value_heads, dimensions.head_dimension,
        dimensions.value_dimension, dimensions.positions);
    return cudaGetLastError();
  }
  gated_delta_attention_f32<<<1, 256, 0, stream>>>(
      static_cast<const float*>(buffer_pointer(plan, arena, plan.buffers()[command.buffers[0].value()].id)),
      static_cast<const float*>(buffer_pointer(plan, arena, plan.buffers()[command.buffers[1].value()].id)),
      static_cast<const float*>(buffer_pointer(plan, arena, plan.buffers()[command.buffers[2].value()].id)),
      static_cast<const float*>(buffer_pointer(plan, arena, plan.buffers()[command.buffers[3].value()].id)),
      static_cast<const float*>(buffer_pointer(plan, arena, plan.buffers()[command.buffers[4].value()].id)),
      static_cast<float*>(buffer_pointer(plan, arena, plan.buffers()[command.buffers[5].value()].id)),
      static_cast<float*>(buffer_pointer(plan, arena, plan.buffers()[command.buffers[6].value()].id)),
      dimensions.key_value_heads, dimensions.value_heads, dimensions.head_dimension,
      dimensions.value_dimension, dimensions.positions);
  return cudaGetLastError();
}

inline cudaError_t launch_residual(const ir::physical::CommandDescriptor& command,
                                   const ir::physical::Plan& plan, void* arena, void*,
                                   cudaStream_t stream) {
  if (command.buffers.size() < 3) return cudaErrorInvalidValue;
  const auto& left = plan.buffers()[command.buffers[0].value()];
  const auto& right = plan.buffers()[command.buffers[1].value()];
  const auto& output = plan.buffers()[command.buffers[2].value()];
  const std::uint64_t bytes = left.size;
  if (bytes % sizeof(float) != 0) return cudaErrorInvalidValue;
  const std::uint32_t residual_blocks = static_cast<std::uint32_t>(
      (static_cast<std::size_t>(bytes / sizeof(float)) + 255U) / 256U);
  residual_f32<<<(residual_blocks == 0U ? 1U : residual_blocks), 256, 0, stream>>>(
      static_cast<const float*>(buffer_pointer(plan, arena, left.id)),
      static_cast<const float*>(buffer_pointer(plan, arena, right.id)),
      static_cast<float*>(buffer_pointer(plan, arena, output.id)),
      static_cast<std::size_t>(bytes / sizeof(float)));
  return cudaGetLastError();
}

inline cudaError_t launch_silu_mul(const ir::physical::CommandDescriptor& command,
                                   const ir::physical::Plan& plan, void* arena, void*,
                                   cudaStream_t stream) {
  if (command.buffers.size() < 3) return cudaErrorInvalidValue;
  const auto& gate = plan.buffers()[command.buffers[0].value()];
  const auto& value = plan.buffers()[command.buffers[1].value()];
  const auto& output = plan.buffers()[command.buffers[2].value()];
  if (gate.size % sizeof(float) != 0) return cudaErrorInvalidValue;
  const std::uint32_t silu_blocks = static_cast<std::uint32_t>(
      (static_cast<std::size_t>(gate.size / sizeof(float)) + 255U) / 256U);
  silu_mul_f32<<<(silu_blocks == 0U ? 1U : silu_blocks), 256, 0, stream>>>(
      static_cast<const float*>(buffer_pointer(plan, arena, gate.id)),
      static_cast<const float*>(buffer_pointer(plan, arena, value.id)),
      static_cast<float*>(buffer_pointer(plan, arena, output.id)),
      static_cast<std::size_t>(gate.size / sizeof(float)));
  return cudaGetLastError();
}

inline cudaError_t launch_sigmoid_mul(const ir::physical::CommandDescriptor& command,
                                      const ir::physical::Plan& plan, void* arena, void*,
                                      cudaStream_t stream) {
  if (command.buffers.size() < 3) return cudaErrorInvalidValue;
  const auto& gate = plan.buffers()[command.buffers[0].value()];
  const auto& value = plan.buffers()[command.buffers[1].value()];
  const auto& output = plan.buffers()[command.buffers[2].value()];
  if (gate.size % sizeof(float) != 0) return cudaErrorInvalidValue;
  const std::uint32_t sigmoid_blocks = static_cast<std::uint32_t>(
      (static_cast<std::size_t>(gate.size / sizeof(float)) + 255U) / 256U);
  sigmoid_mul_f32<<<(sigmoid_blocks == 0U ? 1U : sigmoid_blocks), 256, 0, stream>>>(
      static_cast<const float*>(buffer_pointer(plan, arena, gate.id)),
      static_cast<const float*>(buffer_pointer(plan, arena, value.id)),
      static_cast<float*>(buffer_pointer(plan, arena, output.id)),
      static_cast<std::size_t>(gate.size / sizeof(float)));
  return cudaGetLastError();
}

inline cudaError_t launch_split(const ir::physical::CommandDescriptor& command,
                                const ir::physical::Plan& plan, void* arena, void*,
                                cudaStream_t stream) {
  if (command.buffers.size() != 3) return cudaErrorInvalidValue;
  const auto& input = plan.buffers()[command.buffers[0].value()];
  const auto& first = plan.buffers()[command.buffers[1].value()];
  const auto& second = plan.buffers()[command.buffers[2].value()];
  const std::size_t first_elements = static_cast<std::size_t>(first.size / sizeof(float));
  const std::size_t total_elements = static_cast<std::size_t>(input.size / sizeof(float));
  const std::uint32_t split_blocks = static_cast<std::uint32_t>((total_elements + 255U) / 256U);
  split_f32<<<(split_blocks == 0U ? 1U : split_blocks), 256, 0, stream>>>(
      static_cast<const float*>(buffer_pointer(plan, arena, input.id)),
      static_cast<float*>(buffer_pointer(plan, arena, first.id)),
      static_cast<float*>(buffer_pointer(plan, arena, second.id)), first_elements,
      total_elements);
  return cudaGetLastError();
}

inline cudaError_t launch_split_last(const ir::physical::CommandDescriptor& command,
                                     const ir::physical::Plan& plan, void* arena, void*,
                                     cudaStream_t stream) {
  if (command.buffers.size() != 3) return cudaErrorInvalidValue;
  const auto& input = plan.buffers()[command.buffers[0].value()];
  const auto& first = plan.buffers()[command.buffers[1].value()];
  const auto& second = plan.buffers()[command.buffers[2].value()];
  const auto dimensions = command.split;
  const std::uint32_t split_last_blocks = static_cast<std::uint32_t>(
      (static_cast<std::size_t>(dimensions.outer) *
           (static_cast<std::size_t>(dimensions.first) + dimensions.second) +
       255U) / 256U);
  split_last_f32<<<(split_last_blocks == 0U ? 1U : split_last_blocks), 256, 0, stream>>>(
      static_cast<const float*>(buffer_pointer(plan, arena, input.id)),
      static_cast<float*>(buffer_pointer(plan, arena, first.id)),
      static_cast<float*>(buffer_pointer(plan, arena, second.id)), dimensions.outer,
      dimensions.first, dimensions.second);
  return cudaGetLastError();
}

inline cudaError_t launch_rope(const ir::physical::CommandDescriptor& command,
                               const ir::physical::Plan& plan, void* arena, void*,
                               cudaStream_t stream) {
  if (command.buffers.size() != 2) return cudaErrorInvalidValue;
  const auto& input = plan.buffers()[command.buffers[0].value()];
  const auto& output = plan.buffers()[command.buffers[1].value()];
  const auto dimensions = command.rope;
  const std::uint32_t rope_blocks = static_cast<std::uint32_t>(
      (static_cast<std::size_t>(dimensions.heads) * dimensions.head_dimension + 255U) / 256U);
  rope_f32<<<(rope_blocks == 0U ? 1U : rope_blocks), 256, 0, stream>>>(
      static_cast<const float*>(buffer_pointer(plan, arena, input.id)),
      static_cast<float*>(buffer_pointer(plan, arena, output.id)), dimensions.heads,
      dimensions.head_dimension, dimensions.rotary_dimension, dimensions.position,
      command.scalar);
  return cudaGetLastError();
}

inline cudaError_t launch_cache_append(const ir::physical::CommandDescriptor& command,
                                       const ir::physical::Plan& plan, void* arena, void*,
                                       cudaStream_t stream) {
  if (command.buffers.size() != 4) return cudaErrorInvalidValue;
  const auto dimensions = command.cache_append;
  const auto& keys = plan.buffers()[command.buffers[0].value()];
  const auto& values = plan.buffers()[command.buffers[1].value()];
  const auto& key_cache = plan.buffers()[command.buffers[2].value()];
  const auto& value_cache = plan.buffers()[command.buffers[3].value()];
  cache_append_f32_bf16<<<1, 256, 0, stream>>>(
      static_cast<const float*>(buffer_pointer(plan, arena, keys.id)),
      static_cast<const float*>(buffer_pointer(plan, arena, values.id)),
      static_cast<std::uint16_t*>(buffer_pointer(plan, arena, key_cache.id)),
      static_cast<std::uint16_t*>(buffer_pointer(plan, arena, value_cache.id)), dimensions.heads,
      dimensions.head_dimension, dimensions.position, dimensions.capacity);
  return cudaGetLastError();
}

inline cudaError_t launch_gated_delta_parameters(
    const ir::physical::CommandDescriptor& command, const ir::physical::Plan& plan, void* arena,
    void*, cudaStream_t stream) {
  if (command.buffers.size() != 6) return cudaErrorInvalidValue;
  const auto& a = plan.buffers()[command.buffers[0].value()];
  const auto& b = plan.buffers()[command.buffers[1].value()];
  const auto& a_log = plan.buffers()[command.buffers[2].value()];
  const auto& dt_bias = plan.buffers()[command.buffers[3].value()];
  const auto& log_decay = plan.buffers()[command.buffers[4].value()];
  const auto& beta = plan.buffers()[command.buffers[5].value()];
  gated_delta_parameters_f32<<<1, 256, 0, stream>>>(
      static_cast<const float*>(buffer_pointer(plan, arena, a.id)),
      static_cast<const float*>(buffer_pointer(plan, arena, b.id)),
      static_cast<const float*>(buffer_pointer(plan, arena, a_log.id)),
      static_cast<const float*>(buffer_pointer(plan, arena, dt_bias.id)),
      static_cast<float*>(buffer_pointer(plan, arena, log_decay.id)),
      static_cast<float*>(buffer_pointer(plan, arena, beta.id)),
      static_cast<std::size_t>(a.size / sizeof(float)));
  return cudaGetLastError();
}

inline cudaError_t launch_causal_conv_silu(const ir::physical::CommandDescriptor& command,
                                           const ir::physical::Plan& plan, void* arena, void*,
                                           cudaStream_t stream) {
  if (command.buffers.size() != 4) return cudaErrorInvalidValue;
  const auto dimensions = command.convolution;
  const auto& input = plan.buffers()[command.buffers[0].value()];
  const auto& weights = plan.buffers()[command.buffers[1].value()];
  const auto& state = plan.buffers()[command.buffers[2].value()];
  const auto& output = plan.buffers()[command.buffers[3].value()];
  // S04 performance reset E0b: was <<<1,256>>>; the kernel is a grid-stride loop over channels.
  const std::uint32_t conv_blocks = static_cast<std::uint32_t>(
      (static_cast<std::size_t>(dimensions.channels) + 255U) / 256U);
  causal_conv_silu_f32<<<(conv_blocks == 0U ? 1U : conv_blocks), 256, 0, stream>>>(
      static_cast<const float*>(buffer_pointer(plan, arena, input.id)),
      static_cast<const float*>(buffer_pointer(plan, arena, weights.id)),
      static_cast<std::uint16_t*>(buffer_pointer(plan, arena, state.id)),
      static_cast<float*>(buffer_pointer(plan, arena, output.id)), dimensions.channels,
      dimensions.kernel_size);
  return cudaGetLastError();
}

inline cudaError_t launch_attention_bf16_cache(
    const ir::physical::CommandDescriptor& command, const ir::physical::Plan& plan, void* arena,
    void*, cudaStream_t stream) {
  if (command.buffers.size() != 4) return cudaErrorInvalidValue;
  const auto& query = plan.buffers()[command.buffers[0].value()];
  const auto& keys = plan.buffers()[command.buffers[1].value()];
  const auto& values = plan.buffers()[command.buffers[2].value()];
  const auto& output = plan.buffers()[command.buffers[3].value()];
  const auto dimensions = command.attention;
  // S04-P1: compute Q.K once per position (bit-identical, removes the
  // O(head_dim^2) value pass). Guard on the shared-memory budget; fall back to
  // the incumbent single-block kernel when the KV window cannot be cached.
  const std::size_t cached_bytes = 2U * dimensions.positions * sizeof(float);
  const std::size_t query_heads = dimensions.query_heads;
  int maximum_shared_bytes = 0;
  cudaDeviceGetAttribute(&maximum_shared_bytes, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
  if (query_heads != 0 && cached_bytes <= static_cast<std::size_t>(maximum_shared_bytes)) {
    if (cudaFuncSetAttribute(grouped_attention_bf16_cache_cached,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(cached_bytes)) != cudaSuccess) {
      return cudaGetLastError();
    }
    grouped_attention_bf16_cache_cached<<<static_cast<std::uint32_t>(query_heads), 256,
                                          cached_bytes, stream>>>(
        static_cast<const float*>(buffer_pointer(plan, arena, query.id)),
        static_cast<const std::uint16_t*>(buffer_pointer(plan, arena, keys.id)),
        static_cast<const std::uint16_t*>(buffer_pointer(plan, arena, values.id)),
        static_cast<float*>(buffer_pointer(plan, arena, output.id)), dimensions.query_heads,
        dimensions.key_value_heads, dimensions.head_dimension, dimensions.positions);
    return cudaGetLastError();
  }
  grouped_attention_bf16_cache<<<1, 256, 0, stream>>>(
      static_cast<const float*>(buffer_pointer(plan, arena, query.id)),
      static_cast<const std::uint16_t*>(buffer_pointer(plan, arena, keys.id)),
      static_cast<const std::uint16_t*>(buffer_pointer(plan, arena, values.id)),
      static_cast<float*>(buffer_pointer(plan, arena, output.id)), dimensions.query_heads,
      dimensions.key_value_heads, dimensions.head_dimension, dimensions.positions);
  return cudaGetLastError();
}

inline cudaError_t launch_rms_norm(const ir::physical::CommandDescriptor& command,
                                   const ir::physical::Plan& plan, void* arena, void*,
                                   cudaStream_t stream) {
  if (command.buffers.size() < 3) return cudaErrorInvalidValue;
  const auto& input = plan.buffers()[command.buffers[0].value()];
  const auto& output = plan.buffers()[command.buffers[1].value()];
  const auto& scale = plan.buffers()[command.buffers[2].value()];
  const std::uint64_t bytes = input.size;
  const std::size_t elements = static_cast<std::size_t>(bytes / sizeof(float));
  const std::size_t scale_elements = static_cast<std::size_t>(scale.size / sizeof(float));
  if (bytes == 0 || bytes % sizeof(float) != 0 || output.size != bytes || scale_elements == 0 ||
      elements % scale_elements != 0) return cudaErrorInvalidValue;
  rms_norm_f32<<<1, 1, 0, stream>>>(
      static_cast<const float*>(buffer_pointer(plan, arena, input.id)),
      static_cast<const float*>(buffer_pointer(plan, arena, scale.id)),
      static_cast<float*>(buffer_pointer(plan, arena, output.id)), elements, scale_elements,
      command.epsilon, command.add_one_to_scale);
  return cudaGetLastError();
}

inline cudaError_t launch_rms_norm_bf16(const ir::physical::CommandDescriptor& command,
                                        const ir::physical::Plan& plan, void* arena, void*,
                                        cudaStream_t stream) {
  if (command.buffers.size() < 3) return cudaErrorInvalidValue;
  const auto& input = plan.buffers()[command.buffers[0].value()];
  const auto& output = plan.buffers()[command.buffers[1].value()];
  const auto& scale = plan.buffers()[command.buffers[2].value()];
  const std::size_t elements = static_cast<std::size_t>(input.size / sizeof(float));
  const std::size_t scale_elements = static_cast<std::size_t>(scale.size / sizeof(std::uint16_t));
  if (input.size == 0 || input.size % sizeof(float) != 0 || output.size != input.size ||
      scale.size == 0 || scale.size % sizeof(std::uint16_t) != 0 ||
      elements % scale_elements != 0) {
    return cudaErrorInvalidValue;
  }
  const std::size_t rows = elements / scale_elements;
  if (scale_elements <= 8192U && rows != 0 && rows <= 65535U) {
    rms_norm_f32_bf16_scale_parallel<<<static_cast<std::uint32_t>(rows), 256, 0, stream>>>(
        static_cast<const float*>(buffer_pointer(plan, arena, input.id)),
        static_cast<const std::uint16_t*>(buffer_pointer(plan, arena, scale.id)),
        static_cast<float*>(buffer_pointer(plan, arena, output.id)), elements, scale_elements,
        command.epsilon, command.add_one_to_scale);
    return cudaGetLastError();
  }
  rms_norm_f32_bf16_scale<<<1, 1, 0, stream>>>(
      static_cast<const float*>(buffer_pointer(plan, arena, input.id)),
      static_cast<const std::uint16_t*>(buffer_pointer(plan, arena, scale.id)),
      static_cast<float*>(buffer_pointer(plan, arena, output.id)), elements, scale_elements,
      command.epsilon, command.add_one_to_scale);
  return cudaGetLastError();
}

inline cudaError_t launch_layer_norm(const ir::physical::CommandDescriptor& command,
                                     const ir::physical::Plan& plan, void* arena, void*,
                                     cudaStream_t stream) {
  if (command.buffers.size() < 4) return cudaErrorInvalidValue;
  const auto& input = plan.buffers()[command.buffers[0].value()];
  const auto& output = plan.buffers()[command.buffers[1].value()];
  const auto& scale = plan.buffers()[command.buffers[2].value()];
  const auto& bias = plan.buffers()[command.buffers[3].value()];
  const std::uint64_t bytes = input.size;
  if (bytes == 0 || bytes % sizeof(float) != 0) return cudaErrorInvalidValue;
  layer_norm_f32<<<1, 1, 0, stream>>>(
      static_cast<const float*>(buffer_pointer(plan, arena, input.id)),
      static_cast<const float*>(buffer_pointer(plan, arena, scale.id)),
      static_cast<const float*>(buffer_pointer(plan, arena, bias.id)),
      static_cast<float*>(buffer_pointer(plan, arena, output.id)),
      static_cast<std::size_t>(bytes / sizeof(float)), command.epsilon);
  return cudaGetLastError();
}

inline base::Status validate_command(const ir::physical::CommandDescriptor& command,
                                     const ir::physical::Plan& plan) {
  const auto exact_buffers = [&](std::size_t count) {
    return command.buffers.size() == count;
  };
  const auto same_sizes = [&](std::size_t count) {
    if (!exact_buffers(count)) return false;
    const std::uint64_t expected = plan.buffers()[command.buffers.front().value()].size;
    for (std::size_t index = 1; index < count; ++index) {
      if (plan.buffers()[command.buffers[index].value()].size != expected) return false;
    }
    return expected != 0 && expected % sizeof(float) == 0;
  };
  const auto has_dtype = [&](std::size_t index, ir::physical::PhysicalDType dtype) {
    return index < command.buffers.size() &&
           plan.buffers()[command.buffers[index].value()].tensor.dtype == dtype;
  };
  const auto all_dtype = [&](ir::physical::PhysicalDType dtype) {
    for (std::size_t index = 0; index < command.buffers.size(); ++index) {
      if (!has_dtype(index, dtype)) return false;
    }
    return true;
  };
  switch (command.kernel.value()) {
    case 1:
      if (!exact_buffers(2) ||
          plan.buffers()[command.buffers[0].value()].size !=
              plan.buffers()[command.buffers[1].value()].size ||
          plan.buffers()[command.buffers[0].value()].tensor.dtype !=
              plan.buffers()[command.buffers[1].value()].tensor.dtype ||
          plan.buffers()[command.buffers[0].value()].tensor.encoding !=
              plan.buffers()[command.buffers[1].value()].tensor.encoding) {
        return base::Status::invalid_argument("CUDA copy requires equal-sized, identically typed buffers");
      }
      return {};
    case 7:
      if (!exact_buffers(3) || plan.buffers()[command.buffers[0].value()].size != sizeof(std::uint32_t) ||
          !has_dtype(0, ir::physical::PhysicalDType::int32) ||
          !has_dtype(1, ir::physical::PhysicalDType::f32) ||
          !has_dtype(2, ir::physical::PhysicalDType::f32) ||
          plan.buffers()[command.buffers[2].value()].size == 0 ||
          plan.buffers()[command.buffers[2].value()].size % sizeof(float) != 0 ||
          plan.buffers()[command.buffers[1].value()].size == 0 ||
          plan.buffers()[command.buffers[1].value()].size %
              plan.buffers()[command.buffers[2].value()].size != 0) {
        return base::Status::invalid_argument("CUDA embedding requires token, table, and f32 output buffers");
      }
      return {};
    case 8:
      if (!exact_buffers(3) || plan.buffers()[command.buffers[0].value()].size != sizeof(std::uint32_t) ||
          !has_dtype(0, ir::physical::PhysicalDType::int32) ||
          !has_dtype(1, ir::physical::PhysicalDType::bf16) ||
          !has_dtype(2, ir::physical::PhysicalDType::f32) ||
          plan.buffers()[command.buffers[2].value()].size == 0 ||
          plan.buffers()[command.buffers[2].value()].size % sizeof(float) != 0 ||
          plan.buffers()[command.buffers[1].value()].size == 0 ||
          plan.buffers()[command.buffers[1].value()].size % sizeof(std::uint16_t) != 0 ||
          (plan.buffers()[command.buffers[1].value()].size / sizeof(std::uint16_t)) %
              (plan.buffers()[command.buffers[2].value()].size / sizeof(float)) != 0) {
        return base::Status::invalid_argument(
            "CUDA BF16 embedding requires token, BF16 table, and f32 output buffers");
      }
      return {};
    case 9:
      if (!exact_buffers(3) || plan.buffers()[command.buffers[2].value()].size == 0 ||
          !has_dtype(0, ir::physical::PhysicalDType::u8) ||
          !has_dtype(1, ir::physical::PhysicalDType::u8) ||
          !has_dtype(2, ir::physical::PhysicalDType::f32) ||
          plan.buffers()[command.buffers[0].value()].tensor.encoding !=
              ir::physical::StorageEncoding::nvfp4_packed ||
          plan.buffers()[command.buffers[1].value()].tensor.encoding !=
              ir::physical::StorageEncoding::fp8_e4m3_group_scale ||
          plan.buffers()[command.buffers[2].value()].tensor.encoding !=
              ir::physical::StorageEncoding::none ||
          plan.buffers()[command.buffers[2].value()].size % sizeof(float) != 0 ||
          plan.buffers()[command.buffers[2].value()].size / sizeof(float) % 16 != 0 ||
          plan.buffers()[command.buffers[0].value()].size !=
              plan.buffers()[command.buffers[2].value()].size / sizeof(float) / 2U ||
          plan.buffers()[command.buffers[1].value()].size !=
              plan.buffers()[command.buffers[2].value()].size / sizeof(float) / 16U) {
        return base::Status::invalid_argument(
            "CUDA NVFP4 dequantization requires packed, scale, and aligned f32 buffers");
      }
      return {};
    case 16:
      if (!exact_buffers(2) || !has_dtype(0, ir::physical::PhysicalDType::bf16) ||
          !has_dtype(1, ir::physical::PhysicalDType::f32) ||
          plan.buffers()[command.buffers[0].value()].size == 0 ||
          plan.buffers()[command.buffers[0].value()].size > UINT64_MAX / 2U ||
          plan.buffers()[command.buffers[1].value()].size !=
              plan.buffers()[command.buffers[0].value()].size * 2U) {
        return base::Status::invalid_argument("CUDA BF16-to-F32 cast has invalid buffers");
      }
      return {};
    case 17:
      if (!exact_buffers(2) || !has_dtype(0, ir::physical::PhysicalDType::f32) ||
          !has_dtype(1, ir::physical::PhysicalDType::bf16) ||
          plan.buffers()[command.buffers[0].value()].size == 0 ||
          plan.buffers()[command.buffers[0].value()].size % sizeof(float) != 0 ||
          plan.buffers()[command.buffers[1].value()].size !=
              plan.buffers()[command.buffers[0].value()].size / 2U) {
        return base::Status::invalid_argument("CUDA F32-to-BF16 cast has invalid buffers");
      }
      return {};
    case 10:
    case 30: {
      if (!exact_buffers(3) || plan.buffers()[command.buffers[0].value()].size == 0 ||
          !all_dtype(ir::physical::PhysicalDType::f32) ||
          plan.buffers()[command.buffers[1].value()].size == 0 ||
          plan.buffers()[command.buffers[2].value()].size == 0 ||
          plan.buffers()[command.buffers[0].value()].size % sizeof(float) != 0 ||
          plan.buffers()[command.buffers[1].value()].size % sizeof(float) != 0 ||
          plan.buffers()[command.buffers[2].value()].size % sizeof(float) != 0) {
        return base::Status::invalid_argument(
            "CUDA LM head requires non-empty f32 input, weight, and output buffers");
      }
      const std::uint64_t input_elements =
          plan.buffers()[command.buffers[0].value()].size / sizeof(float);
      const std::uint64_t output_elements =
          plan.buffers()[command.buffers[2].value()].size / sizeof(float);
      const std::uint64_t weight_elements =
          plan.buffers()[command.buffers[1].value()].size / sizeof(float);
      if (input_elements == 0 || weight_elements % input_elements != 0 ||
          weight_elements / input_elements != output_elements) {
        return base::Status::invalid_argument(
            "CUDA LM head weight shape does not match input and output");
      }
      if (command.kernel.value() == 30 && input_elements % 4U != 0U) {
        return base::Status::invalid_argument(
            "CUDA specialized control projection requires K % 4 == 0");
      }
      return {};
    }
    case 11: {
      if (!exact_buffers(5)) {
        return base::Status::invalid_argument(
            "CUDA gated FFN requires input, gate, up, down, and output buffers");
      }
      const auto& input = plan.buffers()[command.buffers[0].value()];
      const auto& gate = plan.buffers()[command.buffers[1].value()];
      const auto& up = plan.buffers()[command.buffers[2].value()];
      const auto& down = plan.buffers()[command.buffers[3].value()];
      const auto& output = plan.buffers()[command.buffers[4].value()];
      if (!all_dtype(ir::physical::PhysicalDType::f32) || input.size == 0 || output.size != input.size ||
          input.size % sizeof(float) != 0 ||
          gate.size == 0 || up.size != gate.size || down.size == 0 ||
          gate.size % input.size != 0 || down.size != output.size / sizeof(float) *
              (gate.size / input.size) * sizeof(float) ||
          gate.size % sizeof(float) != 0 || up.size % sizeof(float) != 0 ||
          down.size % sizeof(float) != 0) {
        return base::Status::invalid_argument(
            "CUDA gated FFN weight shapes do not match hidden and intermediate dimensions");
      }
      return {};
    }
    case 13:
    case 27:
    case 28:
    case 29: {
      if (!exact_buffers(5)) {
        return base::Status::invalid_argument(
            "CUDA NVFP4 linear requires f32 input, packed weights, scales, tensor scale, and output buffers");
      }
      const auto& input = plan.buffers()[command.buffers[0].value()];
      const auto& packed = plan.buffers()[command.buffers[1].value()];
      const auto& scales = plan.buffers()[command.buffers[2].value()];
      const auto& tensor_scale = plan.buffers()[command.buffers[3].value()];
      const auto& output = plan.buffers()[command.buffers[4].value()];
      if (input.tensor.dtype != ir::physical::PhysicalDType::f32 ||
          packed.tensor.dtype != ir::physical::PhysicalDType::u8 ||
          scales.tensor.dtype != ir::physical::PhysicalDType::u8 ||
          tensor_scale.tensor.dtype != ir::physical::PhysicalDType::f32 ||
          output.tensor.dtype != ir::physical::PhysicalDType::f32 ||
          packed.tensor.encoding != ir::physical::StorageEncoding::nvfp4_packed ||
          scales.tensor.encoding != ir::physical::StorageEncoding::fp8_e4m3_group_scale ||
          input.tensor.encoding != ir::physical::StorageEncoding::none ||
          tensor_scale.tensor.encoding != ir::physical::StorageEncoding::none ||
          output.tensor.encoding != ir::physical::StorageEncoding::none ||
          input.size == 0 || tensor_scale.size != sizeof(float) || output.size == 0 ||
          input.size % sizeof(float) != 0 ||
          output.size % sizeof(float) != 0 || input.size / sizeof(float) % 16 != 0) {
        return base::Status::invalid_argument(
            "CUDA NVFP4 linear requires non-empty aligned f32 input/output buffers");
      }
      const std::uint64_t input_elements = input.size / sizeof(float);
      const std::uint64_t output_elements = output.size / sizeof(float);
      const std::uint64_t packed_row_bytes = input_elements / 2U;
      const std::uint64_t scale_row_bytes = input_elements / 16U;
      if (packed_row_bytes == 0 || scale_row_bytes == 0 ||
          packed.size % packed_row_bytes != 0 || packed.size / packed_row_bytes != output_elements ||
          scales.size % scale_row_bytes != 0 || scales.size / scale_row_bytes != output_elements) {
        return base::Status::invalid_argument(
            "CUDA NVFP4 linear packed weights or scales have an invalid shape");
      }
      if (command.kernel.value() == 27 || command.kernel.value() == 28) {
        // Native MMA contract: M16N8K64 tile, four 16-wide scale blocks, and workspace scratch.
        if (input_elements % 64U != 0U || output_elements % 16U != 0U ||
            command.workspace_size == 0U) {
          return base::Status::invalid_argument(
              "CUDA native NVFP4 MMA requires K % 64 == 0, rows % 16 == 0, and workspace");
        }
      }
      if (command.kernel.value() == 29) {
        // Donor-schedule streaming GEMV contract: 512-wide K phases, 32-row CTAs.
        if (input_elements % 512U != 0U || output_elements % 32U != 0U) {
          return base::Status::invalid_argument(
              "CUDA NVFP4 streaming GEMV requires K % 512 == 0 and rows % 32 == 0");
        }
      }
      return {};
    }
    case 14: {
      if (!exact_buffers(4)) {
        return base::Status::invalid_argument(
            "CUDA attention requires query, key, value, and output buffers");
      }
      const auto& query = plan.buffers()[command.buffers[0].value()];
      const auto& keys = plan.buffers()[command.buffers[1].value()];
      const auto& values = plan.buffers()[command.buffers[2].value()];
      const auto& output = plan.buffers()[command.buffers[3].value()];
      const auto dimensions = command.attention;
      if (!all_dtype(ir::physical::PhysicalDType::f32) ||
          dimensions.query_heads == 0 || dimensions.key_value_heads == 0 ||
          dimensions.head_dimension == 0 || dimensions.positions == 0 ||
          dimensions.query_heads % dimensions.key_value_heads != 0) {
        return base::Status::invalid_argument("CUDA attention dimensions are invalid");
      }
      const auto product = [](std::uint64_t first, std::uint64_t second,
                              std::uint64_t third) -> std::uint64_t {
        if (first != 0 && second > std::numeric_limits<std::uint64_t>::max() / first) return 0;
        const std::uint64_t first_two = first * second;
        if (third != 0 && first_two > std::numeric_limits<std::uint64_t>::max() / third) return 0;
        return first_two * third;
      };
      const std::uint64_t query_elements = product(
          dimensions.query_heads, dimensions.head_dimension, 1);
      const std::uint64_t cache_elements = product(
          dimensions.positions, dimensions.key_value_heads, dimensions.head_dimension);
      if (query_elements == 0 || cache_elements == 0 ||
          query.size != query_elements * sizeof(float) ||
          output.size != query_elements * sizeof(float) ||
          keys.size != cache_elements * sizeof(float) ||
          values.size != cache_elements * sizeof(float)) {
        return base::Status::invalid_argument(
            "CUDA attention buffer sizes do not match its authored dimensions");
      }
      return {};
    }
    case 15: {
      if (!exact_buffers(7)) {
        return base::Status::invalid_argument(
            "CUDA gated delta attention requires query, key, value, gates, state, and output buffers");
      }
      const auto dimensions = command.attention;
      if (dimensions.query_heads == 0 || dimensions.key_value_heads == 0 ||
          dimensions.value_heads == 0 || dimensions.head_dimension == 0 ||
          dimensions.value_dimension == 0 || dimensions.positions == 0 ||
          dimensions.query_heads != dimensions.key_value_heads ||
          dimensions.value_heads % dimensions.key_value_heads != 0) {
        return base::Status::invalid_argument("CUDA gated delta attention dimensions are invalid");
      }
      const auto product = [](std::uint64_t first, std::uint64_t second,
                              std::uint64_t third) -> std::uint64_t {
        if (first != 0 && second > std::numeric_limits<std::uint64_t>::max() / first) return 0;
        const std::uint64_t first_two = first * second;
        if (third != 0 && first_two > std::numeric_limits<std::uint64_t>::max() / third) return 0;
        return first_two * third;
      };
      const std::uint64_t qk_elements = product(
          dimensions.positions, dimensions.key_value_heads, dimensions.head_dimension);
      const std::uint64_t value_elements = product(
          dimensions.positions, dimensions.value_heads, dimensions.value_dimension);
      const std::uint64_t state_elements = product(
          dimensions.value_heads, dimensions.head_dimension, dimensions.value_dimension);
      const auto bytes = [](std::uint64_t elements) -> std::uint64_t {
        return elements > std::numeric_limits<std::uint64_t>::max() / sizeof(float)
                   ? 0
                   : elements * sizeof(float);
      };
      const auto& query = plan.buffers()[command.buffers[0].value()];
      const auto& keys = plan.buffers()[command.buffers[1].value()];
      const auto& values = plan.buffers()[command.buffers[2].value()];
      const auto& log_decay = plan.buffers()[command.buffers[3].value()];
      const auto& beta = plan.buffers()[command.buffers[4].value()];
      const auto& state = plan.buffers()[command.buffers[5].value()];
      const auto& output = plan.buffers()[command.buffers[6].value()];
      if (!all_dtype(ir::physical::PhysicalDType::f32) ||
          qk_elements == 0 || value_elements == 0 || state_elements == 0 ||
          query.size != bytes(qk_elements) || keys.size != bytes(qk_elements) ||
          values.size != bytes(value_elements) ||
          log_decay.size != bytes(static_cast<std::uint64_t>(dimensions.positions) *
                                  dimensions.value_heads) ||
          beta.size != bytes(static_cast<std::uint64_t>(dimensions.positions) *
                             dimensions.value_heads) ||
          state.size != bytes(state_elements) || output.size != bytes(value_elements)) {
        return base::Status::invalid_argument(
            "CUDA gated delta attention buffer sizes do not match authored dimensions");
      }
      return {};
    }
    case 4:
      if (!same_sizes(3) || !all_dtype(ir::physical::PhysicalDType::f32)) {
        return base::Status::invalid_argument("CUDA residual requires three equal-sized f32 buffers");
      }
      return {};
    case 5:
      if (!exact_buffers(3) || !has_dtype(0, ir::physical::PhysicalDType::f32) ||
          !has_dtype(1, ir::physical::PhysicalDType::f32) ||
          !has_dtype(2, ir::physical::PhysicalDType::f32)) {
        return base::Status::invalid_argument("CUDA RMSNorm requires input, output, and scale buffers");
      }
      if (plan.buffers()[command.buffers[0].value()].size == 0 ||
          plan.buffers()[command.buffers[0].value()].size !=
              plan.buffers()[command.buffers[1].value()].size ||
          plan.buffers()[command.buffers[0].value()].size % sizeof(float) != 0 ||
          plan.buffers()[command.buffers[2].value()].size == 0 ||
          plan.buffers()[command.buffers[2].value()].size % sizeof(float) != 0 ||
          (plan.buffers()[command.buffers[0].value()].size / sizeof(float)) %
                  (plan.buffers()[command.buffers[2].value()].size / sizeof(float)) !=
              0) {
        return base::Status::invalid_argument("CUDA RMSNorm scale must tile complete f32 rows");
      }
      return {};
    case 12:
      if (!exact_buffers(3) || plan.buffers()[command.buffers[0].value()].size == 0 ||
          !has_dtype(0, ir::physical::PhysicalDType::f32) ||
          !has_dtype(1, ir::physical::PhysicalDType::f32) ||
          !has_dtype(2, ir::physical::PhysicalDType::bf16) ||
          plan.buffers()[command.buffers[0].value()].size % sizeof(float) != 0 ||
          plan.buffers()[command.buffers[1].value()].size !=
              plan.buffers()[command.buffers[0].value()].size ||
          plan.buffers()[command.buffers[2].value()].size == 0 ||
          plan.buffers()[command.buffers[2].value()].size % sizeof(std::uint16_t) != 0 ||
          (plan.buffers()[command.buffers[0].value()].size / sizeof(float)) %
                  (plan.buffers()[command.buffers[2].value()].size /
                   sizeof(std::uint16_t)) != 0) {
        return base::Status::invalid_argument(
            "CUDA BF16 RMSNorm requires f32 input/output and a BF16 scale buffer");
      }
      return {};
    case 6:
      if (!same_sizes(4) || !all_dtype(ir::physical::PhysicalDType::f32)) {
        return base::Status::invalid_argument("CUDA LayerNorm requires input, output, scale, and bias buffers");
      }
      return {};
    case 18:
      if (!same_sizes(3) || !all_dtype(ir::physical::PhysicalDType::f32)) {
        return base::Status::invalid_argument("CUDA SiLU multiply requires three equal-sized f32 buffers");
      }
      return {};
    case 19:
      if (!same_sizes(3) || !all_dtype(ir::physical::PhysicalDType::f32)) {
        return base::Status::invalid_argument("CUDA sigmoid multiply requires three equal-sized f32 buffers");
      }
      return {};
    case 20: {
      if (!exact_buffers(2) || !all_dtype(ir::physical::PhysicalDType::f32)) {
        return base::Status::invalid_argument("CUDA RoPE requires two equal-sized f32 buffers");
      }
      const auto dimensions = command.rope;
      if (dimensions.heads == 0 || dimensions.head_dimension == 0 ||
          dimensions.rotary_dimension == 0 || dimensions.rotary_dimension > dimensions.head_dimension ||
          dimensions.rotary_dimension % 2 != 0 || !std::isfinite(command.scalar) ||
          command.scalar <= 1.0F || plan.buffers()[command.buffers[0].value()].size !=
              static_cast<std::uint64_t>(dimensions.heads) * dimensions.head_dimension * sizeof(float)) {
        return base::Status::invalid_argument("CUDA RoPE dimensions or theta are invalid");
      }
      return {};
    }
    case 21: {
      if (!exact_buffers(3) || !all_dtype(ir::physical::PhysicalDType::f32)) {
        return base::Status::invalid_argument("CUDA split requires three equal-type f32 buffers");
      }
      const auto& input = plan.buffers()[command.buffers[0].value()];
      const auto& first = plan.buffers()[command.buffers[1].value()];
      const auto& second = plan.buffers()[command.buffers[2].value()];
      if (input.size == 0 || first.size == 0 || second.size == 0 ||
          input.size != first.size + second.size || input.size % sizeof(float) != 0) {
        return base::Status::invalid_argument("CUDA split buffer sizes do not form one f32 input");
      }
      return {};
    }
    case 22: {
      if (!exact_buffers(4) || !has_dtype(0, ir::physical::PhysicalDType::f32) ||
          !has_dtype(1, ir::physical::PhysicalDType::f32) ||
          !has_dtype(2, ir::physical::PhysicalDType::bf16) ||
          !has_dtype(3, ir::physical::PhysicalDType::bf16)) {
        return base::Status::invalid_argument(
            "CUDA cache append requires f32 K/V rows and BF16 cache buffers");
      }
      const auto dimensions = command.cache_append;
      const std::uint64_t row_bytes = static_cast<std::uint64_t>(dimensions.heads) *
                                      dimensions.head_dimension * sizeof(float);
      const std::uint64_t cache_bytes = static_cast<std::uint64_t>(dimensions.capacity) *
                                        dimensions.heads * dimensions.head_dimension *
                                        sizeof(std::uint16_t);
      if (dimensions.heads == 0 || dimensions.head_dimension == 0 || dimensions.capacity == 0 ||
          dimensions.position >= dimensions.capacity || row_bytes == 0 ||
          plan.buffers()[command.buffers[0].value()].size != row_bytes ||
          plan.buffers()[command.buffers[1].value()].size != row_bytes ||
          plan.buffers()[command.buffers[2].value()].size != cache_bytes ||
          plan.buffers()[command.buffers[3].value()].size != cache_bytes) {
        return base::Status::invalid_argument("CUDA cache append dimensions do not match buffers");
      }
      return {};
    }
    case 23: {
      if (!exact_buffers(4) || !has_dtype(0, ir::physical::PhysicalDType::f32) ||
          !has_dtype(1, ir::physical::PhysicalDType::bf16) ||
          !has_dtype(2, ir::physical::PhysicalDType::bf16) ||
          !has_dtype(3, ir::physical::PhysicalDType::f32)) {
        return base::Status::invalid_argument(
            "CUDA BF16-cache attention requires f32 query/output and BF16 caches");
      }
      const auto dimensions = command.attention;
      const std::uint64_t query_bytes = static_cast<std::uint64_t>(dimensions.query_heads) *
                                        dimensions.head_dimension * sizeof(float);
      const std::uint64_t output_bytes = query_bytes;
      const std::uint64_t cache_bytes = static_cast<std::uint64_t>(dimensions.positions) *
                                        dimensions.key_value_heads * dimensions.head_dimension *
                                        sizeof(std::uint16_t);
      if (dimensions.query_heads == 0 || dimensions.key_value_heads == 0 ||
          dimensions.head_dimension == 0 || dimensions.positions == 0 ||
          dimensions.query_heads % dimensions.key_value_heads != 0 || query_bytes == 0 ||
          plan.buffers()[command.buffers[0].value()].size != query_bytes ||
          plan.buffers()[command.buffers[1].value()].size < cache_bytes ||
          plan.buffers()[command.buffers[2].value()].size < cache_bytes ||
          plan.buffers()[command.buffers[3].value()].size != output_bytes) {
        return base::Status::invalid_argument("CUDA BF16-cache attention dimensions do not match buffers");
      }
      return {};
    }
    case 24:
      if (!same_sizes(6) || !all_dtype(ir::physical::PhysicalDType::f32)) {
        return base::Status::invalid_argument(
            "CUDA gated-delta parameters require six equal-sized f32 buffers");
      }
      return {};
    case 25: {
      if (!exact_buffers(4) || !has_dtype(0, ir::physical::PhysicalDType::f32) ||
          !has_dtype(1, ir::physical::PhysicalDType::f32) ||
          !has_dtype(2, ir::physical::PhysicalDType::bf16) ||
          !has_dtype(3, ir::physical::PhysicalDType::f32)) {
        return base::Status::invalid_argument(
            "CUDA causal convolution requires f32 input/weights/output and BF16 state");
      }
      const auto dimensions = command.convolution;
      const auto& input = plan.buffers()[command.buffers[0].value()];
      const auto& weights = plan.buffers()[command.buffers[1].value()];
      const auto& state = plan.buffers()[command.buffers[2].value()];
      const auto& output = plan.buffers()[command.buffers[3].value()];
      if (dimensions.channels == 0 || dimensions.kernel_size == 0 || input.size !=
              static_cast<std::uint64_t>(dimensions.channels) * sizeof(float) ||
          output.size != input.size || weights.size !=
              static_cast<std::uint64_t>(dimensions.channels) * dimensions.kernel_size *
                  sizeof(float) || state.size !=
              static_cast<std::uint64_t>(dimensions.channels) * dimensions.kernel_size *
                  sizeof(std::uint16_t)) {
        return base::Status::invalid_argument("CUDA causal convolution dimensions do not match buffers");
      }
      return {};
    }
    case 26: {
      if (!exact_buffers(3) || !all_dtype(ir::physical::PhysicalDType::f32)) {
        return base::Status::invalid_argument("CUDA last-dimension split requires f32 operands");
      }
      const auto dimensions = command.split;
      const auto& input = plan.buffers()[command.buffers[0].value()];
      const auto& first = plan.buffers()[command.buffers[1].value()];
      const auto& second = plan.buffers()[command.buffers[2].value()];
      if (dimensions.outer == 0 || dimensions.first == 0 || dimensions.second == 0 ||
          input.size != static_cast<std::uint64_t>(dimensions.outer) *
                            (dimensions.first + dimensions.second) * sizeof(float) ||
          first.size != static_cast<std::uint64_t>(dimensions.outer) * dimensions.first * sizeof(float) ||
          second.size != static_cast<std::uint64_t>(dimensions.outer) * dimensions.second * sizeof(float)) {
        return base::Status::invalid_argument("CUDA last-dimension split sizes do not match dimensions");
      }
      return {};
    }
    default:
      return base::Status::unsupported("CUDA kernel ID is not registered");
  }
}

inline LaunchFunction resolve(std::uint64_t kernel_id) {
  switch (kernel_id) {
    case 1: return &launch_copy;
    case 7: return &launch_embedding;
    case 8: return &launch_embedding_bf16;
    case 16: return &launch_cast_bf16_to_f32;
    case 17: return &launch_cast_f32_to_bf16;
    case 9: return &launch_nvfp4_dequantize;
    case 10: return &launch_lm_head;
    case 11: return &launch_gated_dense_ffn;
    case 4: return &launch_residual;
    case 5: return &launch_rms_norm;
    case 12: return &launch_rms_norm_bf16;
    case 13: return &launch_nvfp4_linear;
    case 14: return &launch_attention;
    case 15: return &launch_gated_delta_attention;
    case 6: return &launch_layer_norm;
    case 18: return &launch_silu_mul;
    case 19: return &launch_sigmoid_mul;
    case 20: return &launch_rope;
    case 21: return &launch_split;
    case 22: return &launch_cache_append;
    case 23: return &launch_attention_bf16_cache;
    case 24: return &launch_gated_delta_parameters;
    case 25: return &launch_causal_conv_silu;
    case 26: return &launch_split_last;
    // S04-P8R-Q experimental native SM120 block-scaled NVFP4 MMA projection.
    case 27: return &launch_nvfp4_linear_mma;
    case 29: return &launch_nvfp4_linear_gemv_rows;
    case 30: return &launch_linear_f32_rows;
    case 28: return &launch_nvfp4_linear_mma_two_level;
    default: return nullptr;
  }
}

inline base::Status contextual(const base::Status& source, std::string_view context) {
  base::Status copy = source;
  copy.with_context(context);
  return copy;
}

}  // namespace detail

/**
 * CUDA-backed Physical Plan shell for lifecycle and scheduling qualification.
 *
 * Construction validates and binds every resource. execute() launches only prebound baseline
 * functions in Physical Plan dependency order; it performs no allocation or device-wide sync.
 */
class CudaPlanSession final {
 public:
  static base::Result<CudaPlanSession> create(const ir::physical::Plan& plan,
                                              std::uint32_t target_capability,
                                              std::string_view kernel_catalog) {
    base::Status plan_status = plan.verify();
    if (!plan_status.ok()) return detail::contextual(plan_status, "CUDA physical plan");
    if (target_capability != 120 || plan.capability().target_capability != target_capability ||
        plan.capability().kernel_catalog != kernel_catalog) {
      return base::Status::unsupported("CUDA physical plan capability does not match target");
    }
    if (kernel_catalog != "baseline-v1") {
      return base::Status::unsupported("CUDA kernel catalog is not registered");
    }

    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) {
      return base::Status::unavailable("CUDA active device is unavailable");
    }
    cudaDeviceProp properties{};
    const cudaError_t property_error = cudaGetDeviceProperties(&properties, device);
    if (property_error != cudaSuccess) {
      return detail::contextual(cuda_status(property_error, "cudaGetDeviceProperties"),
                                "CUDA target probe");
    }
    if (properties.major != 12 || properties.minor != 0) {
      return base::Status::unsupported("active CUDA device is not sm_120a");
    }
    if (plan.resources().arena_bytes > properties.totalGlobalMem -
                                           std::min<std::size_t>(properties.totalGlobalMem,
                                                                 plan.resources().workspace_bytes)) {
      return base::Status::resource_exhausted("physical plan exceeds active CUDA device memory");
    }

    CudaPlanSession session{plan};
    session.commands_plan_ = plan.commands();
    session.launchers_.reserve(plan.commands().size());
    for (const ir::physical::CommandDescriptor& command : plan.commands()) {
      if (command.kernel.value() == 0) {
        return base::Status::failed_precondition("CUDA command has no stable kernel ID");
      }
      const detail::LaunchFunction launcher = detail::resolve(command.kernel.value());
      if (launcher == nullptr) return base::Status::unsupported("CUDA kernel ID is not registered");
      ++session.lifecycle_trace_->kernel_bindings;
      // The baseline catalog has no per-command workspace contract. The experimental native NVFP4
      // MMA kernel (id 27) is the one exception: it uses the session workspace for the dynamically
      // quantised activation scratch, sized by the provider and allocated once at session creation.
      if (command.workspace_size != 0 && command.kernel.value() != 27 &&
          command.kernel.value() != 28) {
        return base::Status::unsupported("CUDA baseline has no command workspace contract");
      }
      const base::Status command_status = detail::validate_command(command, plan);
      if (!command_status.ok()) {
        base::Status error = command_status;
        error.with_context("physical command " + std::to_string(command.id.value()) +
                           " kernel " + std::to_string(command.kernel.value()));
        return error;
      }
      session.launchers_.push_back(launcher);
    }
    auto device_arena = DeviceBuffer::allocate(plan.resources().arena_bytes,
                                                session.lifecycle_trace_.get());
    if (!device_arena.has_value()) return detail::contextual(device_arena.error(), "CUDA device arena");
    auto workspace = DeviceBuffer::allocate(plan.resources().workspace_bytes,
                                             session.lifecycle_trace_.get());
    if (!workspace.has_value()) return detail::contextual(workspace.error(), "CUDA workspace arena");

    std::uint32_t stream_count = 0;
    for (const ir::physical::CommandDescriptor& command : plan.commands()) {
      if (command.kernel.value() == 0) {
        return base::Status::failed_precondition("CUDA command has no stable kernel ID");
      }
      if (command.stream == std::numeric_limits<std::uint32_t>::max()) {
        return base::Status::resource_exhausted("CUDA stream ordinal cannot be incremented");
      }
      stream_count = std::max(stream_count, command.stream + 1);
    }

    session.streams_.reserve(stream_count);
    for (std::uint32_t index = 0; index < stream_count; ++index) {
      auto stream = StreamOwner::create(session.lifecycle_trace_.get());
      if (!stream.has_value()) return detail::contextual(stream.error(), "CUDA stream creation");
      session.streams_.push_back(std::move(stream).value());
    }
    session.single_stream_ordered_ = stream_count <= 1;
    if (!session.single_stream_ordered_) {
      session.events_.reserve(plan.commands().size());
      for (std::size_t index = 0; index < plan.commands().size(); ++index) {
        auto event = EventOwner::create(session.lifecycle_trace_.get());
        if (!event.has_value()) return detail::contextual(event.error(), "CUDA event creation");
        session.events_.push_back(std::move(event).value());
      }
    }

    std::vector<bool> emitted(plan.commands().size(), false);
    session.command_order_.reserve(plan.commands().size());
    for (std::size_t rank = 0; rank < plan.commands().size(); ++rank) {
      bool found = false;
      for (std::size_t index = 0; index < plan.commands().size(); ++index) {
        if (emitted[index]) continue;
        bool dependencies_emitted = true;
        for (const ir::physical::CommandId dependency : plan.commands()[index].dependencies) {
          if (!emitted[dependency.value()]) {
            dependencies_emitted = false;
            break;
          }
        }
        if (!dependencies_emitted) continue;
        emitted[index] = true;
        session.command_order_.push_back(index);
        found = true;
        break;
      }
      if (!found) return base::Status::failed_precondition("CUDA command schedule is not executable");
    }
    session.device_arena_ = std::move(device_arena).value();
    session.workspace_ = std::move(workspace).value();
    return session;
  }

  CudaPlanSession(CudaPlanSession&&) noexcept = default;
  CudaPlanSession& operator=(CudaPlanSession&&) noexcept = default;
  CudaPlanSession(const CudaPlanSession&) = delete;
  CudaPlanSession& operator=(const CudaPlanSession&) = delete;

  base::Status execute() noexcept {
    if (poisoned_) return base::Status::failed_precondition("CUDA session is poisoned");
    for (const std::size_t command_index : command_order_) {
      const ir::physical::CommandDescriptor& command = commands_plan_[command_index];
      cudaStream_t stream = streams_[command.stream].get();
      if (!single_stream_ordered_) {
        for (const ir::physical::CommandId dependency : command.dependencies) {
          const cudaError_t wait_error = cudaStreamWaitEvent(stream, events_[dependency.value()].get(), 0);
          if (wait_error != cudaSuccess) return poison(wait_error, "cudaStreamWaitEvent");
        }
      }
      const cudaError_t launch_error = launchers_[command_index](
          command, plan_, device_arena_.data(), workspace_.data(), stream);
      if (launch_error != cudaSuccess) return poison(launch_error, "baseline command launch");
      if (!single_stream_ordered_) {
        const cudaError_t record_error = cudaEventRecord(events_[command_index].get(), stream);
        if (record_error != cudaSuccess) return poison(record_error, "cudaEventRecord");
      }
      ++trace_.commands_executed;
      ++trace_.launches;
    }
    return {};
  }

  /**
   * Executes one continuation segment with a supplied physical decode position.
   *
   * This test/profiling entry point reuses the validated command schedule and only updates the
   * position fields of already-bound cache, RoPE, and full-attention commands. It performs no
   * allocation or model dispatch; production execution uses execute() with compile-time position
   * specialization.
   */
  base::Status execute_at_position_for_test(std::uint32_t position) noexcept {
    const std::vector<ir::physical::CommandId> no_trace;
    std::vector<std::vector<std::byte>> ignored;
    return execute_at_position_for_test(position, no_trace, ignored);
  }

  /**
   * Executes one continuation segment and copies selected command output buffers after launch.
   *
   * This is a diagnostic-only entry point. It synchronizes at each selected command and may
   * allocate host vectors, so it must never be used by production execution. This compatibility
   * overload captures the final declared buffer; callers that need a semantic result must use
   * the explicit-output overload below.
   */
  base::Status execute_at_position_for_test(
      std::uint32_t position, const std::vector<ir::physical::CommandId>& trace_commands,
      std::vector<std::vector<std::byte>>& trace_captures) noexcept {
    std::vector<std::pair<ir::physical::CommandId, ir::physical::BufferId>> trace_requests;
    trace_requests.reserve(trace_commands.size());
    for (const auto command_id : trace_commands) {
      if (command_id.value() >= commands_plan_.size()) {
        return base::Status::invalid_argument("CUDA trace command is undefined");
      }
      const auto& command = commands_plan_[command_id.value()];
      if (command.buffers.empty()) {
        return base::Status::failed_precondition("diagnostic command has no output buffer");
      }
      trace_requests.emplace_back(command_id, command.buffers.back());
    }
    return execute_at_position_for_test(position, trace_requests, trace_captures);
  }

  /**
   * Executes one continuation segment and copies explicitly selected command result buffers.
   *
   * This diagnostic-only entry point makes the captured physical buffer explicit because a
   * command's last operand is not necessarily its result (for example RMSNorm weights and KV
   * cache operands are declared after the result). Requests must be ordered by command ID and
   * name a buffer belonging to the requested command. It is never used by production execution.
   */
  base::Status execute_at_position_for_test(
      std::uint32_t position,
      const std::vector<std::pair<ir::physical::CommandId, ir::physical::BufferId>>& trace_requests,
      std::vector<std::vector<std::byte>>& trace_captures) noexcept {
    if (poisoned_) return base::Status::failed_precondition("CUDA session is poisoned");
    trace_captures.clear();
    trace_captures.reserve(trace_requests.size());
    for (std::size_t index = 0; index < trace_requests.size(); ++index) {
      const auto command_id = trace_requests[index].first;
      const auto output_id = trace_requests[index].second;
      if (command_id.value() >= commands_plan_.size() ||
          (index != 0 && trace_requests[index - 1].first.value() >= command_id.value())) {
        return base::Status::invalid_argument("CUDA trace commands must be unique and ordered");
      }
      const auto& command = commands_plan_[command_id.value()];
      if (std::find(command.buffers.begin(), command.buffers.end(), output_id) ==
          command.buffers.end()) {
        return base::Status::invalid_argument("CUDA trace output buffer is not a command operand");
      }
    }
    std::uint32_t cache_capacity = 0;
    for (const auto& command : commands_plan_) {
      if (command.cache_append.capacity != 0) {
        cache_capacity = command.cache_append.capacity;
        break;
      }
    }
    if (cache_capacity == 0 || position >= cache_capacity) {
      return base::Status::out_of_range("CUDA decode position exceeds physical cache capacity");
    }
    std::size_t next_trace = 0;
    for (const std::size_t command_index : command_order_) {
      ir::physical::CommandDescriptor command = commands_plan_[command_index];
      if (command.cache_append.capacity != 0) command.cache_append.position = position;
      if (command.rope.heads != 0) command.rope.position = position;
      if (command.attention.positions != 0 && command.cache_append.capacity == 0 &&
          command.attention.value_heads == 0) {
        command.attention.positions = position + 1U;
      }
      cudaStream_t stream = streams_[command.stream].get();
      if (!single_stream_ordered_) {
        for (const ir::physical::CommandId dependency : command.dependencies) {
          const cudaError_t wait_error = cudaStreamWaitEvent(stream, events_[dependency.value()].get(), 0);
          if (wait_error != cudaSuccess) return poison(wait_error, "cudaStreamWaitEvent");
        }
      }
      const cudaError_t launch_error = launchers_[command_index](
          command, plan_, device_arena_.data(), workspace_.data(), stream);
      if (launch_error != cudaSuccess) return poison(launch_error, "baseline continuation launch");
      if (!single_stream_ordered_) {
        const cudaError_t record_error = cudaEventRecord(events_[command_index].get(), stream);
        if (record_error != cudaSuccess) return poison(record_error, "cudaEventRecord");
      }
      if (next_trace < trace_requests.size() &&
          trace_requests[next_trace].first.value() == command.id.value()) {
        ++lifecycle_trace_->device_synchronizations;
        const cudaError_t sync_error = cudaDeviceSynchronize();
        if (sync_error != cudaSuccess) return poison(sync_error, "diagnostic command trace synchronization");
        const ir::physical::BufferId output_id = trace_requests[next_trace].second;
        if (output_id.value() >= plan_.buffers().size()) {
          return base::Status::out_of_range("diagnostic command output buffer is undefined");
        }
        const auto& output = plan_.buffers()[output_id.value()];
        trace_captures.emplace_back(output.size);
        const cudaError_t copy_error = cudaMemcpy(
            trace_captures.back().data(),
            detail::buffer_pointer(plan_, device_arena_.data(), output.id), output.size,
            cudaMemcpyDeviceToHost);
        if (copy_error != cudaSuccess) return poison(copy_error, "diagnostic command trace copy");
        ++next_trace;
      }
      ++trace_.commands_executed;
      ++trace_.launches;
    }
    if (next_trace != trace_requests.size()) {
      return base::Status::failed_precondition("diagnostic command was not scheduled");
    }
    return {};
  }

  /** Explicit test/profiling synchronization; never called by execute(). */
  base::Status synchronize_for_test() noexcept {
    if (poisoned_) return base::Status::failed_precondition("CUDA session is poisoned");
    ++lifecycle_trace_->device_synchronizations;
    const cudaError_t error = cudaDeviceSynchronize();
    if (error != cudaSuccess) return poison(error, "explicit test synchronization");
    return {};
  }

  /**
   * Fills the device arena with a byte pattern for memory-initialization diagnostics.
   *
   * This is test/profiling-only API. It is intentionally not called by execute() and does not
   * change production initialization policy. Callers must invoke it before uploading the
   * artifact/state or after synchronizing prior work on the session streams.
   */
  base::Status fill_device_for_test(std::uint8_t value) noexcept {
    if (poisoned_) return base::Status::failed_precondition("CUDA session is poisoned");
    const cudaError_t error = cudaMemset(device_arena_.data(), static_cast<int>(value),
                                         static_cast<std::size_t>(device_arena_.bytes()));
    if (error != cudaSuccess) return poison(error, "test device arena fill");
    return {};
  }

  base::Status copy_to_device(ir::physical::BufferId id, base::ConstByteView source) noexcept {
    if (poisoned_) return base::Status::failed_precondition("CUDA session is poisoned");
    const auto validation = validate_copy(id, source.size());
    if (!validation.ok()) return validation;
    const auto& buffer = plan_.buffers()[id.value()];
    const cudaError_t copy_error = cudaMemcpy(detail::buffer_pointer(plan_, device_arena_.data(), buffer.id),
                                              source.data(), source.size(), cudaMemcpyHostToDevice);
    if (copy_error != cudaSuccess) return poison(copy_error, "host-to-device copy");
    return {};
  }

  base::Status copy_from_device(ir::physical::BufferId id, base::ByteView destination) noexcept {
    if (poisoned_) return base::Status::failed_precondition("CUDA session is poisoned");
    const auto validation = validate_copy(id, destination.size());
    if (!validation.ok()) return validation;
    ++lifecycle_trace_->device_synchronizations;
    const cudaError_t sync_error = cudaDeviceSynchronize();
    if (sync_error != cudaSuccess) return poison(sync_error, "device-to-host copy boundary");
    const auto& buffer = plan_.buffers()[id.value()];
    const cudaError_t copy_error = cudaMemcpy(destination.data(),
                                              detail::buffer_pointer(plan_, device_arena_.data(), buffer.id),
                                              destination.size(), cudaMemcpyDeviceToHost);
    if (copy_error != cudaSuccess) return poison(copy_error, "device-to-host copy");
    return {};
  }

  [[nodiscard]] const CudaExecutionTrace& trace() const noexcept { return trace_; }
  [[nodiscard]] const CudaLifecycleTrace& lifecycle_trace() const noexcept {
    return *lifecycle_trace_;
  }
  [[nodiscard]] std::uint64_t device_arena_bytes() const noexcept { return device_arena_.bytes(); }
  [[nodiscard]] std::uint64_t workspace_bytes() const noexcept { return workspace_.bytes(); }
  [[nodiscard]] bool poisoned() const noexcept { return poisoned_; }

 private:
  explicit CudaPlanSession(const ir::physical::Plan& plan)
      : plan_(plan), lifecycle_trace_(std::make_shared<CudaLifecycleTrace>()) {}

  base::Status validate_copy(ir::physical::BufferId id, std::size_t bytes) const noexcept {
    if (id.value() >= plan_.buffers().size()) return base::Status::out_of_range("CUDA buffer is undefined");
    if (bytes > plan_.buffers()[id.value()].size) {
      return base::Status::out_of_range("host copy exceeds physical buffer");
    }
    return {};
  }

  base::Status poison(cudaError_t error, std::string_view context) noexcept {
    poisoned_ = true;
    return cuda_status(error, context);
  }

  ir::physical::Plan plan_;
  std::shared_ptr<CudaLifecycleTrace> lifecycle_trace_;
  DeviceBuffer device_arena_;
  DeviceBuffer workspace_;
  std::vector<StreamOwner> streams_;
  std::vector<EventOwner> events_;
  std::vector<std::size_t> command_order_;
  std::vector<ir::physical::CommandDescriptor> commands_plan_;
  std::vector<detail::LaunchFunction> launchers_;
  CudaExecutionTrace trace_{};
  bool poisoned_{false};
  bool single_stream_ordered_{false};
};

}  // namespace superinfer::sm120::cuda_runtime
