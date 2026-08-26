#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

#include <superinfer/base/checked_math.hpp>
#include <superinfer/base/result.hpp>

namespace superinfer::sm120::reference {

/**
 * Small, independent CPU contracts for operations that future target providers must reproduce.
 *
 * The functions own returned host vectors, allocate only outside the runtime hot path, and do not
 * inspect model identity or storage encodings. Inputs are row-major and caller-owned. Invalid
 * shapes, non-finite inputs, and out-of-range token IDs return a typed status.
 */
class ReferencePrimitives final {
 public:
  static base::Result<std::vector<float>> embedding(std::span<const float> table,
                                                     std::size_t vocabulary,
                                                     std::size_t hidden,
                                                     std::uint32_t token) {
    const auto elements = checked_elements(vocabulary, hidden);
    if (!elements.has_value()) return elements.error();
    if (vocabulary == 0 || hidden == 0 || table.size() != elements.value()) {
      return base::Status::invalid_argument("embedding table shape is invalid");
    }
    if (token >= vocabulary) return base::Status::out_of_range("embedding token is out of range");
    const std::size_t begin = static_cast<std::size_t>(token) * hidden;
    std::vector<float> result(table.begin() + begin, table.begin() + begin + hidden);
    if (!finite(result)) return base::Status::invalid_argument("embedding table contains non-finite values");
    return result;
  }

  static base::Result<std::vector<float>> linear(std::span<const float> weights,
                                                  std::size_t output,
                                                  std::size_t input,
                                                  std::span<const float> values,
                                                  const std::vector<float>* bias = nullptr) {
    const auto weight_elements = checked_elements(output, input);
    if (!weight_elements.has_value()) return weight_elements.error();
    if (output == 0 || input == 0 || weights.size() != weight_elements.value() ||
        values.size() != input || (bias != nullptr && bias->size() != output)) {
      return base::Status::invalid_argument("linear shape is invalid");
    }
    if (!finite(weights) || !finite(values) || (bias != nullptr && !finite(*bias))) {
      return base::Status::invalid_argument("linear input contains non-finite values");
    }
    std::vector<float> result(output, 0.0F);
    for (std::size_t row = 0; row < output; ++row) {
      float sum = bias == nullptr ? 0.0F : (*bias)[row];
      for (std::size_t column = 0; column < input; ++column) {
        sum += weights[row * input + column] * values[column];
      }
      result[row] = sum;
    }
    if (!finite(result)) return base::Status::invalid_argument("linear result is non-finite");
    return result;
  }

  /**
   * Dequantizes a row-major ModelOpt NVFP4 matrix using the checkpoint layout.
   *
   * The packed weights have shape [output, input / 2], with the even input in the low nibble
   * and the odd input in the high nibble. Block scales are raw FP8 E4M3FN bytes with shape
   * [output, input / 16]. Each scale covers 16 consecutive input values and is multiplied by
   * the positive per-tensor weight scale. The returned vector is host-owned and this reference
   * path is not used by the runtime hot path.
   */
  static base::Result<std::vector<float>> dequantize_nvfp4(
      std::span<const std::uint8_t> packed_weights,
      std::size_t output,
      std::size_t input,
      std::span<const std::uint8_t> block_scales,
      float weight_scale_2) {
    if (output == 0 || input == 0 || input % 16 != 0 || !std::isfinite(weight_scale_2) ||
        weight_scale_2 <= 0.0F) {
      return base::Status::invalid_argument("NVFP4 dimensions or tensor scale are invalid");
    }
    const auto packed_row = checked_elements(output, input / 2);
    if (!packed_row.has_value()) return packed_row.error();
    const auto output_elements = checked_elements(output, input);
    if (!output_elements.has_value()) return output_elements.error();
    const auto scale_elements = checked_elements(output, input / 16);
    if (!scale_elements.has_value()) return scale_elements.error();
    if (packed_weights.size() != packed_row.value() ||
        block_scales.size() != scale_elements.value()) {
      return base::Status::invalid_argument("NVFP4 packed or scale shape is invalid");
    }

    std::vector<float> result(output_elements.value(), 0.0F);
    for (std::size_t row = 0; row < output; ++row) {
      for (std::size_t column = 0; column < input; ++column) {
        const std::uint8_t packed = packed_weights[row * (input / 2) + column / 2];
        const std::uint8_t code = (column % 2 == 0) ? (packed & 0x0FU) : (packed >> 4U);
        const float fp4_value = decode_e2m1(code);
        const auto scale = decode_e4m3fn(
            block_scales[row * (input / 16) + column / 16]);
        if (!scale.has_value()) {
          base::Status error = scale.error();
          return error.with_context("NVFP4 block scale");
        }
        result[row * input + column] = fp4_value * scale.value() * weight_scale_2;
      }
    }
    if (!finite(result)) return base::Status::invalid_argument("NVFP4 result is non-finite");
    return result;
  }

  static base::Result<std::vector<float>> gated_ffn(std::span<const float> gate_weights,
                                                     std::span<const float> up_weights,
                                                     std::span<const float> down_weights,
                                                     std::size_t hidden,
                                                     std::size_t intermediate,
                                                     std::span<const float> input) {
    const auto gate = linear(gate_weights, intermediate, hidden, input);
    if (!gate.has_value()) {
      base::Status error = gate.error();
      return error.with_context("gated FFN gate projection");
    }
    const auto up = linear(up_weights, intermediate, hidden, input);
    if (!up.has_value()) {
      base::Status error = up.error();
      return error.with_context("gated FFN up projection");
    }
    std::vector<float> gated(intermediate, 0.0F);
    for (std::size_t index = 0; index < intermediate; ++index) {
      gated[index] = gate.value()[index] / (1.0F + std::exp(-gate.value()[index])) * up.value()[index];
    }
    const auto down = linear(down_weights, hidden, intermediate, gated);
    if (!down.has_value()) {
      base::Status error = down.error();
      return error.with_context("gated FFN down projection");
    }
    return down;
  }

  static base::Result<std::vector<float>> grouped_attention(std::span<const float> query,
                                                              std::span<const float> keys,
                                                              std::span<const float> values,
                                                              std::size_t query_heads,
                                                              std::size_t kv_heads,
                                                              std::size_t head_dimension,
                                                              std::size_t positions) {
    if (query_heads == 0 || kv_heads == 0 || head_dimension == 0 || positions == 0 ||
        query_heads % kv_heads != 0) {
      return base::Status::invalid_argument("grouped attention dimensions are invalid");
    }
    const auto query_elements = checked_elements(query_heads, head_dimension);
    const auto cache_heads = checked_elements(positions, kv_heads);
    if (!query_elements.has_value()) return query_elements.error();
    if (!cache_heads.has_value()) return cache_heads.error();
    const auto cache_elements = checked_elements(cache_heads.value(), head_dimension);
    if (!cache_elements.has_value()) return cache_elements.error();
    if (query.size() != query_elements.value() || keys.size() != cache_elements.value() ||
        values.size() != cache_elements.value()) {
      return base::Status::invalid_argument("grouped attention tensor shapes are invalid");
    }
    if (!finite(query) || !finite(keys) || !finite(values)) {
      return base::Status::invalid_argument("grouped attention input contains non-finite values");
    }
    std::vector<float> output(query.size(), 0.0F);
    const float scale = 1.0F / std::sqrt(static_cast<float>(head_dimension));
    const std::size_t group = query_heads / kv_heads;
    std::vector<float> scores(positions);
    for (std::size_t query_head = 0; query_head < query_heads; ++query_head) {
      const std::size_t kv_head = query_head / group;
      float maximum = -std::numeric_limits<float>::infinity();
      for (std::size_t position = 0; position < positions; ++position) {
        float score = 0.0F;
        for (std::size_t dimension = 0; dimension < head_dimension; ++dimension) {
          score += query[query_head * head_dimension + dimension] *
                   keys[(position * kv_heads + kv_head) * head_dimension + dimension];
        }
        scores[position] = score * scale;
        maximum = std::max(maximum, scores[position]);
      }
      float denominator = 0.0F;
      for (float& score : scores) {
        score = std::exp(score - maximum);
        denominator += score;
      }
      for (std::size_t position = 0; position < positions; ++position) {
        const float probability = scores[position] / denominator;
        for (std::size_t dimension = 0; dimension < head_dimension; ++dimension) {
          output[query_head * head_dimension + dimension] +=
              probability * values[(position * kv_heads + kv_head) * head_dimension + dimension];
        }
      }
    }
    return output;
  }

 private:
  static float decode_e2m1(std::uint8_t code) {
    constexpr std::array<float, 8> magnitudes{
        0.0F, 0.5F, 1.0F, 1.5F, 2.0F, 3.0F, 4.0F, 6.0F};
    const float magnitude = magnitudes[code & 0x07U];
    return (code & 0x08U) == 0 ? magnitude : -magnitude;
  }

  static base::Result<float> decode_e4m3fn(std::uint8_t code) {
    const bool negative = (code & 0x80U) != 0;
    const std::uint8_t exponent = (code >> 3U) & 0x0FU;
    const std::uint8_t mantissa = code & 0x07U;
    if (negative || (exponent == 0x0FU && mantissa == 0x07U)) {
      return base::Status::invalid_argument("NVFP4 block scale is not a finite positive E4M3 value");
    }
    if (exponent == 0) {
      return std::ldexp(static_cast<float>(mantissa) / 8.0F, -6);
    }
    return std::ldexp(1.0F + static_cast<float>(mantissa) / 8.0F,
                      static_cast<int>(exponent) - 7);
  }

  static base::Result<std::size_t> checked_elements(std::size_t first, std::size_t second) {
    const auto result = base::checked_mul(static_cast<std::uint64_t>(first),
                                          static_cast<std::uint64_t>(second));
    if (!result.has_value()) return result.error();
    if (result.value() > std::numeric_limits<std::size_t>::max()) {
      return base::Status::overflow("reference primitive element count exceeds host size");
    }
    return static_cast<std::size_t>(result.value());
  }

  static bool finite(std::span<const float> values) {
    return std::all_of(values.begin(), values.end(), [](float value) { return std::isfinite(value); });
  }

  static bool finite(const std::vector<float>& values) { return finite(std::span<const float>{values}); }
};

}  // namespace superinfer::sm120::reference
