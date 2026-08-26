#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string_view>
#include <vector>

#include <sm120/runtime/cuda_ownership.cuh>
#include <superinfer/base/views.hpp>
#include <superinfer/ir/physical_plan.hpp>

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

/** Reference-correct grouped-query attention over a pre-materialized contiguous KV window. */
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

__global__ inline void rms_norm_f32(const float* input, const float* scale, float* output,
                                    std::size_t elements, float epsilon, bool add_one_to_scale) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  float sum_squares = 0.0F;
  for (std::size_t index = 0; index < elements; ++index) sum_squares += input[index] * input[index];
  const float denominator = sqrtf(sum_squares / static_cast<float>(elements) + epsilon);
  for (std::size_t index = 0; index < elements; ++index) {
    output[index] = input[index] / denominator * (scale[index] + (add_one_to_scale ? 1.0F : 0.0F));
  }
}

__device__ inline float bf16_to_float_device(std::uint16_t value) {
  return __uint_as_float(static_cast<std::uint32_t>(value) << 16U);
}

__global__ inline void rms_norm_f32_bf16_scale(const float* input, const std::uint16_t* scale,
                                               float* output, std::size_t elements, float epsilon,
                                               bool add_one_to_scale) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  float sum_squares = 0.0F;
  for (std::size_t index = 0; index < elements; ++index) sum_squares += input[index] * input[index];
  const float denominator = sqrtf(sum_squares / static_cast<float>(elements) + epsilon);
  for (std::size_t index = 0; index < elements; ++index) {
    output[index] = input[index] / denominator *
                    (bf16_to_float_device(scale[index]) + (add_one_to_scale ? 1.0F : 0.0F));
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
  cast_bf16_to_f32<<<1, 256, 0, stream>>>(
      static_cast<const std::uint16_t*>(buffer_pointer(plan, arena, input.id)),
      static_cast<float*>(buffer_pointer(plan, arena, output.id)),
      static_cast<std::size_t>(input.size / sizeof(std::uint16_t)));
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
  cast_f32_to_bf16<<<1, 256, 0, stream>>>(
      static_cast<const float*>(buffer_pointer(plan, arena, input.id)),
      static_cast<std::uint16_t*>(buffer_pointer(plan, arena, output.id)),
      static_cast<std::size_t>(input.size / sizeof(float)));
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
  linear_f32<<<1, 256, 0, stream>>>(
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
  nvfp4_linear_f32<<<1, 256, 0, stream>>>(
      static_cast<const float*>(buffer_pointer(plan, arena, input.id)),
      static_cast<const std::uint8_t*>(buffer_pointer(plan, arena, packed.id)),
      static_cast<const std::uint8_t*>(buffer_pointer(plan, arena, scales.id)),
      static_cast<const float*>(buffer_pointer(plan, arena, tensor_scale.id)),
      static_cast<float*>(buffer_pointer(plan, arena, output.id)), input_elements,
      output_elements);
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
  residual_f32<<<1, 256, 0, stream>>>(
      static_cast<const float*>(buffer_pointer(plan, arena, left.id)),
      static_cast<const float*>(buffer_pointer(plan, arena, right.id)),
      static_cast<float*>(buffer_pointer(plan, arena, output.id)),
      static_cast<std::size_t>(bytes / sizeof(float)));
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
  if (bytes == 0 || bytes % sizeof(float) != 0) return cudaErrorInvalidValue;
  rms_norm_f32<<<1, 1, 0, stream>>>(
      static_cast<const float*>(buffer_pointer(plan, arena, input.id)),
      static_cast<const float*>(buffer_pointer(plan, arena, scale.id)),
      static_cast<float*>(buffer_pointer(plan, arena, output.id)),
      static_cast<std::size_t>(bytes / sizeof(float)), command.epsilon, command.add_one_to_scale);
  return cudaGetLastError();
}

inline cudaError_t launch_rms_norm_bf16(const ir::physical::CommandDescriptor& command,
                                        const ir::physical::Plan& plan, void* arena, void*,
                                        cudaStream_t stream) {
  if (command.buffers.size() < 3) return cudaErrorInvalidValue;
  const auto& input = plan.buffers()[command.buffers[0].value()];
  const auto& output = plan.buffers()[command.buffers[1].value()];
  const auto& scale = plan.buffers()[command.buffers[2].value()];
  if (input.size == 0 || input.size % sizeof(float) != 0 || output.size != input.size ||
      scale.size != input.size / sizeof(float) * sizeof(std::uint16_t)) {
    return cudaErrorInvalidValue;
  }
  rms_norm_f32_bf16_scale<<<1, 1, 0, stream>>>(
      static_cast<const float*>(buffer_pointer(plan, arena, input.id)),
      static_cast<const std::uint16_t*>(buffer_pointer(plan, arena, scale.id)),
      static_cast<float*>(buffer_pointer(plan, arena, output.id)),
      static_cast<std::size_t>(input.size / sizeof(float)), command.epsilon,
      command.add_one_to_scale);
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
    case 10: {
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
    case 13: {
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
      if (!same_sizes(3) || !all_dtype(ir::physical::PhysicalDType::f32)) {
        return base::Status::invalid_argument("CUDA RMSNorm requires input, output, and scale buffers");
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
          plan.buffers()[command.buffers[2].value()].size !=
              plan.buffers()[command.buffers[0].value()].size / sizeof(float) *
                  sizeof(std::uint16_t)) {
        return base::Status::invalid_argument(
            "CUDA BF16 RMSNorm requires f32 input/output and a BF16 scale buffer");
      }
      return {};
    case 6:
      if (!same_sizes(4) || !all_dtype(ir::physical::PhysicalDType::f32)) {
        return base::Status::invalid_argument("CUDA LayerNorm requires input, output, scale, and bias buffers");
      }
      return {};
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
      if (command.workspace_size != 0) {
        return base::Status::unsupported("CUDA baseline has no command workspace contract");
      }
      const base::Status command_status = detail::validate_command(command, plan);
      if (!command_status.ok()) return command_status;
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
    session.events_.reserve(plan.commands().size());
    for (std::size_t index = 0; index < plan.commands().size(); ++index) {
      auto event = EventOwner::create(session.lifecycle_trace_.get());
      if (!event.has_value()) return detail::contextual(event.error(), "CUDA event creation");
      session.events_.push_back(std::move(event).value());
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
      for (const ir::physical::CommandId dependency : command.dependencies) {
        const cudaError_t wait_error = cudaStreamWaitEvent(stream, events_[dependency.value()].get(), 0);
        if (wait_error != cudaSuccess) return poison(wait_error, "cudaStreamWaitEvent");
      }
      const cudaError_t launch_error = launchers_[command_index](
          command, plan_, device_arena_.data(), workspace_.data(), stream);
      if (launch_error != cudaSuccess) return poison(launch_error, "baseline command launch");
      const cudaError_t record_error = cudaEventRecord(events_[command_index].get(), stream);
      if (record_error != cudaSuccess) return poison(record_error, "cudaEventRecord");
      ++trace_.commands_executed;
      ++trace_.launches;
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

  base::Status copy_to_device(ir::physical::BufferId id, base::ConstByteView source) noexcept {
    if (poisoned_) return base::Status::failed_precondition("CUDA session is poisoned");
    const auto validation = validate_copy(id, source.size());
    if (!validation.ok()) return validation;
    ++lifecycle_trace_->device_synchronizations;
    const cudaError_t sync_error = cudaDeviceSynchronize();
    if (sync_error != cudaSuccess) return poison(sync_error, "host-to-device copy boundary");
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
};

}  // namespace superinfer::sm120::cuda_runtime
