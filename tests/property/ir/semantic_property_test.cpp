#include <superinfer/ir/semantic/builder.hpp>
#include <superinfer/ir/semantic/verifier.hpp>

#include <cassert>
#include <cstdint>
#include <string>
#include <utility>

int main() {
  using namespace superinfer::ir::semantic;

  for (std::uint64_t dimension = 0; dimension < 16; ++dimension) {
    Builder builder;
    const auto input = builder.add_tensor(
        "input", TensorSpec{{Dimension::static_value(dimension)}, DType::f32,
                             QuantizationIntent::none, TensorRole::activation});
    assert(input.has_value());
    assert(builder.add_entry_point("decode", {input.value()}, {input.value()}).ok());
    const auto result = std::move(builder).build();
    assert((dimension == 0) == !result.has_value());
  }

  Builder undefined_reference;
  const auto input = undefined_reference.add_tensor(
      "input", TensorSpec{{Dimension::static_value(4)}, DType::f32,
                           QuantizationIntent::none, TensorRole::activation});
  assert(input.has_value());
  assert(undefined_reference.add_operation("bad", OperationKind::residual,
                                           {TensorId{99}}, {input.value()})
             .has_value());
  assert(undefined_reference.add_entry_point("decode", {input.value()}, {input.value()}).ok());
  const auto bad_reference = std::move(undefined_reference).build();
  assert(!bad_reference.has_value());
  assert(bad_reference.error().code() == superinfer::base::StatusCode::out_of_range);

  Builder odd_rope;
  const auto rope_input = odd_rope.add_tensor(
      "input", TensorSpec{{Dimension::static_value(8)}, DType::f16,
                           QuantizationIntent::none, TensorRole::activation});
  assert(rope_input.has_value());
  const auto rope_output = odd_rope.add_tensor(
      "output", TensorSpec{{Dimension::static_value(8)}, DType::f16,
                            QuantizationIntent::none, TensorRole::activation});
  assert(rope_output.has_value());
  OperationAttributes attributes{4, 2, 8, 3, 0, 0, 0};
  assert(odd_rope.add_operation("attention", OperationKind::grouped_query_attention,
                                {rope_input.value()}, {rope_output.value()}, attributes)
             .has_value());
  assert(odd_rope.add_entry_point("decode", {rope_input.value()}, {rope_output.value()}).ok());
  const auto bad_rope = std::move(odd_rope).build();
  assert(!bad_rope.has_value());
  assert(bad_rope.error().message().find("even") != std::string::npos);
  return 0;
}
