#include <sm120/kernels/baseline/reference_primitives.h>

#include <cassert>
#include <cmath>
#include <cstdint>
#include <vector>

int main() {
  using superinfer::sm120::reference::ReferencePrimitives;

  const std::vector<float> table{0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F};
  const auto embedded = ReferencePrimitives::embedding(table, 3, 2, 2);
  assert(embedded.has_value());
  assert((embedded.value() == std::vector<float>{4.0F, 5.0F}));

  const std::vector<float> weights{1.0F, 2.0F, 3.0F, 4.0F};
  const std::vector<float> input{2.0F, -1.0F};
  const std::vector<float> bias{0.5F, -0.5F};
  const auto projected = ReferencePrimitives::linear(weights, 2, 2, input, &bias);
  assert(projected.has_value());
  assert(std::fabs(projected.value()[0] - 0.5F) < 1.0e-6F);
  assert(std::fabs(projected.value()[1] - 1.5F) < 1.0e-6F);

  const std::vector<float> identity{1.0F, 0.0F, 0.0F, 1.0F};
  const auto ffn = ReferencePrimitives::gated_ffn(identity, identity, identity, 2, 2, input);
  assert(ffn.has_value());
  assert(ffn.value().size() == 2);
  assert(std::isfinite(ffn.value()[0]) && std::isfinite(ffn.value()[1]));

  const std::vector<float> query{1.0F, 0.0F, 0.0F, 1.0F};
  const std::vector<float> keys{1.0F, 0.0F, 0.0F, 1.0F};
  const std::vector<float> values{2.0F, 0.0F, 0.0F, 2.0F};
  const auto attended = ReferencePrimitives::grouped_attention(
      query, keys, values, 2, 1, 2, 2);
  assert(attended.has_value());
  assert(attended.value().size() == 4);
  assert(std::isfinite(attended.value()[0]));

  const auto bad_token = ReferencePrimitives::embedding(table, 3, 2, 3);
  assert(!bad_token.has_value());
  const std::vector<float> short_input{1.0F};
  const auto bad_projection = ReferencePrimitives::linear(weights, 2, 2, short_input, nullptr);
  assert(!bad_projection.has_value());

  const std::vector<std::uint8_t> packed_nvfp4 = {
      0x10, 0x32, 0x54, 0x76, 0x98, 0xba, 0xdc, 0xfe,
  };
  const std::vector<std::uint8_t> block_scales = {0x38};  // E4M3 1.0.
  const auto dequantized = ReferencePrimitives::dequantize_nvfp4(
      packed_nvfp4, 1, 16, block_scales, 2.0F);
  assert(dequantized.has_value());
  const std::vector<float> expected_nvfp4 = {
      0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 6.0F, 8.0F, 12.0F,
      -0.0F, -1.0F, -2.0F, -3.0F, -4.0F, -6.0F, -8.0F, -12.0F,
  };
  assert(dequantized.value().size() == expected_nvfp4.size());
  for (std::size_t index = 0; index < expected_nvfp4.size(); ++index) {
    assert(std::fabs(dequantized.value()[index] - expected_nvfp4[index]) < 1.0e-6F);
  }

  const std::vector<std::uint8_t> two_block_packed(16, 0x11);
  const std::vector<std::uint8_t> two_block_scales = {0x38, 0x40};  // 1.0, 2.0.
  const auto two_blocks = ReferencePrimitives::dequantize_nvfp4(
      two_block_packed, 1, 32, two_block_scales, 1.0F);
  assert(two_blocks.has_value());
  assert(two_blocks.value()[0] == 0.5F);
  assert(two_blocks.value()[15] == 0.5F);
  assert(two_blocks.value()[16] == 1.0F);
  assert(two_blocks.value()[31] == 1.0F);

  assert(!ReferencePrimitives::dequantize_nvfp4(
                   packed_nvfp4, 1, 8, block_scales, 1.0F)
              .has_value());
  assert(!ReferencePrimitives::dequantize_nvfp4(
                   packed_nvfp4, 1, 16, {}, 1.0F)
              .has_value());
  const std::vector<std::uint8_t> invalid_block_scale = {0x7f};
  assert(!ReferencePrimitives::dequantize_nvfp4(
                   packed_nvfp4, 1, 16, invalid_block_scale, 1.0F)
              .has_value());
  assert(!ReferencePrimitives::dequantize_nvfp4(
                   packed_nvfp4, 1, 16, block_scales, 0.0F)
              .has_value());
  return 0;
}
