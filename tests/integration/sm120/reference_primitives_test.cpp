#include <sm120/kernels/baseline/reference_primitives.h>

#include <cassert>
#include <cmath>
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
  return 0;
}
