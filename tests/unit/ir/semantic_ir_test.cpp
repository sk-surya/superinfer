#include <superinfer/ir/semantic/builder.hpp>

#include <cassert>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

using namespace superinfer::ir::semantic;

Module make_fixture(bool reverse_tensor_order) {
  Builder builder;
  const TensorSpec hidden_spec{{Dimension::static_value(2), Dimension::static_value(4)}, DType::f32,
                               QuantizationIntent::none, TensorRole::activation};
  const TensorSpec normalized_spec{{Dimension::static_value(2), Dimension::static_value(4)}, DType::f32,
                                   QuantizationIntent::none, TensorRole::activation};
  const TensorSpec weight_spec{{Dimension::static_value(4)}, DType::f32, QuantizationIntent::none,
                               TensorRole::weight};
  TensorId hidden;
  TensorId normalized;
  TensorId weight;
  if (reverse_tensor_order) {
    const auto weight_result = builder.add_tensor("norm_weight", weight_spec);
    const auto normalized_result = builder.add_tensor("normalized", normalized_spec);
    const auto hidden_result = builder.add_tensor("hidden", hidden_spec);
    assert(weight_result.has_value() && normalized_result.has_value() && hidden_result.has_value());
    weight = weight_result.value();
    normalized = normalized_result.value();
    hidden = hidden_result.value();
  } else {
    const auto hidden_result = builder.add_tensor("hidden", hidden_spec);
    const auto normalized_result = builder.add_tensor("normalized", normalized_spec);
    const auto weight_result = builder.add_tensor("norm_weight", weight_spec);
    assert(hidden_result.has_value() && normalized_result.has_value() && weight_result.has_value());
    hidden = hidden_result.value();
    normalized = normalized_result.value();
    weight = weight_result.value();
  }

  const std::vector<TensorId> inputs{hidden, weight};
  const auto norm = builder.add_operation("norm", OperationKind::rms_norm, inputs,
                                         {normalized}, {});
  assert(norm.has_value());
  assert(builder.add_entry_point("decode", {hidden}, {normalized}).ok());
  const auto result = std::move(builder).build();
  assert(result.has_value());
  return std::move(result).value();
}

}  // namespace

int main() {
  using namespace superinfer::ir::semantic;

  {
    Builder forward_reference_builder;
    const auto input = forward_reference_builder.add_tensor(
        "input", TensorSpec{{Dimension::static_value(1)}, DType::f32,
                             QuantizationIntent::none, TensorRole::activation});
    const auto future = forward_reference_builder.add_tensor(
        "future", TensorSpec{{Dimension::static_value(1)}, DType::f32,
                              QuantizationIntent::none, TensorRole::activation});
    const auto first_output = forward_reference_builder.add_tensor(
        "first_output", TensorSpec{{Dimension::static_value(1)}, DType::f32,
                                    QuantizationIntent::none, TensorRole::activation});
    assert(input.has_value() && future.has_value() && first_output.has_value());
    assert(forward_reference_builder
               .add_operation("consumer", OperationKind::residual,
                              {input.value(), future.value()}, {first_output.value()})
               .has_value());
    assert(forward_reference_builder
               .add_operation("producer", OperationKind::residual,
                              {input.value(), input.value()}, {future.value()})
               .has_value());
    assert(forward_reference_builder.add_entry_point("decode", {input.value()},
                                                     {first_output.value()})
               .ok());
    const auto forward_reference = std::move(forward_reference_builder).build();
    assert(!forward_reference.has_value());
    assert(forward_reference.error().message().find("later-produced") != std::string::npos);
  }

  const auto first = make_fixture(false);
  const auto second = make_fixture(true);
  assert(first.dump() == second.dump());
  assert(first.verify().ok());
  assert(first.dump().find("cuda") == std::string::npos);
  assert(first.dump().find("offset") == std::string::npos);
  assert(first.dump().find("KernelId") == std::string::npos);

  Builder attention_builder;
  const auto q = attention_builder.add_tensor(
      "q", TensorSpec{{Dimension::static_value(1), Dimension::static_value(7 * 8)}, DType::f16,
                       QuantizationIntent::none, TensorRole::activation});
  const auto output = attention_builder.add_tensor(
      "output", TensorSpec{{Dimension::static_value(1), Dimension::static_value(7 * 8)}, DType::f16,
                            QuantizationIntent::none, TensorRole::activation});
  assert(q.has_value() && output.has_value());
  OperationAttributes invalid_gqa;
  invalid_gqa.num_heads = 7;
  invalid_gqa.num_kv_heads = 2;
  invalid_gqa.head_dimension = 8;
  const auto attention = attention_builder.add_operation(
      "attention", OperationKind::grouped_query_attention, {q.value()}, {output.value()},
      invalid_gqa);
  assert(attention.has_value());
  assert(attention_builder.add_entry_point("decode", {q.value()}, {output.value()}).ok());
  const auto invalid_result = std::move(attention_builder).build();
  assert(!invalid_result.has_value());
  assert(invalid_result.error().message().find("kv heads") != std::string::npos);

  const std::filesystem::path golden =
      std::filesystem::path{SUPERINFER_SOURCE_DIR} / "tests/golden/ir/semantic-basic.txt";
  std::ifstream golden_file(golden);
  assert(golden_file.good());
  const std::string expected{std::istreambuf_iterator<char>{golden_file}, {}};
  assert(first.dump() == expected);

  Builder family_builder;
  const auto family_input = family_builder.add_tensor(
      "input", TensorSpec{{Dimension::static_value(1), Dimension::static_value(8)}, DType::f16,
                           QuantizationIntent::none, TensorRole::activation});
  assert(family_input.has_value());
  OperationAttributes attention_attributes{4, 2, 2, 2, 0, 0, 0};
  OperationAttributes gated_delta_attributes{16, 16, 128, 0, 0, 0, 0, 128, 128, 4, 48};
  OperationAttributes moe_attributes{0, 0, 0, 0, 8, 2, 0};
  const std::vector<OperationKind> kinds{
      OperationKind::embedding, OperationKind::rms_norm, OperationKind::layer_norm,
      OperationKind::rope, OperationKind::qkv_projection, OperationKind::gated_delta_attention,
      OperationKind::multi_head_attention,
      OperationKind::grouped_query_attention, OperationKind::local_attention, OperationKind::residual,
      OperationKind::gated_dense_ffn, OperationKind::moe_route, OperationKind::moe_top_k,
      OperationKind::moe_expert, OperationKind::moe_combine, OperationKind::lm_head,
      OperationKind::decode_logits, OperationKind::sampling_inputs};
  for (std::size_t index = 0; index < kinds.size(); ++index) {
    const bool is_gated_delta = kinds[index] == OperationKind::gated_delta_attention;
    const bool is_attention = kinds[index] == OperationKind::multi_head_attention ||
                              kinds[index] == OperationKind::grouped_query_attention ||
                              kinds[index] == OperationKind::local_attention;
    const bool is_moe = kinds[index] == OperationKind::moe_route ||
                        kinds[index] == OperationKind::moe_top_k ||
                        kinds[index] == OperationKind::moe_expert ||
                        kinds[index] == OperationKind::moe_combine;
    const auto operation = family_builder.add_operation(
        "family-" + std::to_string(index), kinds[index], {family_input.value()}, {},
        is_gated_delta ? gated_delta_attributes
                       : (is_attention ? attention_attributes : (is_moe ? moe_attributes : OperationAttributes{})));
    assert(operation.has_value());
  }
  assert(family_builder.add_entry_point("decode", {family_input.value()}, {family_input.value()}).ok());
  assert(std::move(family_builder).build().has_value());
  return 0;
}
