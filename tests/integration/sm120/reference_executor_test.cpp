#include <sm120/kernels/baseline/reference_executor.h>
#include <superinfer/ir/semantic/builder.hpp>

#include <cassert>
#include <cmath>
#include <vector>

namespace {

superinfer::ir::semantic::Module make_fixture() {
  using namespace superinfer;
  ir::semantic::Builder builder;
  const auto x = builder.add_tensor("x", {{ir::semantic::Dimension::static_value(2)},
                                           ir::semantic::DType::f32,
                                           ir::semantic::QuantizationIntent::none,
                                           ir::semantic::TensorRole::activation});
  const auto y = builder.add_tensor("y", {{ir::semantic::Dimension::static_value(2)},
                                           ir::semantic::DType::f32,
                                           ir::semantic::QuantizationIntent::none,
                                           ir::semantic::TensorRole::activation});
  const auto sum = builder.add_tensor("sum", {{ir::semantic::Dimension::static_value(2)},
                                               ir::semantic::DType::f32,
                                               ir::semantic::QuantizationIntent::none,
                                               ir::semantic::TensorRole::activation});
  const auto norm = builder.add_tensor("norm", {{ir::semantic::Dimension::static_value(2)},
                                                 ir::semantic::DType::f32,
                                                 ir::semantic::QuantizationIntent::none,
                                                 ir::semantic::TensorRole::activation});
  assert(x.has_value() && y.has_value() && sum.has_value() && norm.has_value());
  assert(builder.add_operation("residual", ir::semantic::OperationKind::residual,
                               {x.value(), y.value()}, {sum.value()})
             .has_value());
  assert(builder.add_operation("norm", ir::semantic::OperationKind::rms_norm,
                               {sum.value()}, {norm.value()})
             .has_value());
  const auto module = std::move(builder).build();
  assert(module.has_value());
  return std::move(module).value();
}

struct WeightedFixture final {
  superinfer::ir::semantic::Module module;
  superinfer::ir::semantic::TensorId token;
  superinfer::ir::semantic::TensorId table;
  superinfer::ir::semantic::TensorId embedding;
  superinfer::ir::semantic::TensorId gate_weight;
  superinfer::ir::semantic::TensorId up_weight;
  superinfer::ir::semantic::TensorId down_weight;
  superinfer::ir::semantic::TensorId ffn;
  superinfer::ir::semantic::TensorId projection_weight;
};

WeightedFixture make_weighted_fixture() {
  using namespace superinfer;
  using namespace ir::semantic;
  Builder builder;
  const auto token = builder.add_tensor("token", {{Dimension::static_value(1)}, DType::int32,
                                                    QuantizationIntent::none, TensorRole::activation});
  const auto table = builder.add_tensor(
      "embedding_weight", {{Dimension::static_value(3), Dimension::static_value(2)}, DType::f32,
      QuantizationIntent::none, TensorRole::weight});
  const auto embedding = builder.add_tensor(
      "embedding", {{Dimension::static_value(1), Dimension::static_value(2)}, DType::f32,
      QuantizationIntent::none, TensorRole::activation});
  const auto gate_weight = builder.add_tensor(
      "gate_weight", {{Dimension::static_value(2), Dimension::static_value(2)}, DType::f32,
      QuantizationIntent::none, TensorRole::weight});
  const auto up_weight = builder.add_tensor(
      "up_weight", {{Dimension::static_value(2), Dimension::static_value(2)}, DType::f32,
      QuantizationIntent::none, TensorRole::weight});
  const auto down_weight = builder.add_tensor(
      "down_weight", {{Dimension::static_value(2), Dimension::static_value(2)}, DType::f32,
      QuantizationIntent::none, TensorRole::weight});
  const auto ffn = builder.add_tensor(
      "ffn", {{Dimension::static_value(1), Dimension::static_value(2)}, DType::f32,
      QuantizationIntent::none, TensorRole::activation});
  const auto projection_weight = builder.add_tensor(
      "projection_weight", {{Dimension::static_value(2), Dimension::static_value(2)}, DType::f32,
      QuantizationIntent::none, TensorRole::weight});
  const auto logits = builder.add_tensor(
      "logits", {{Dimension::static_value(1), Dimension::static_value(2)}, DType::f32,
      QuantizationIntent::none, TensorRole::logits});
  assert(token.has_value() && table.has_value() && embedding.has_value() && gate_weight.has_value() &&
         up_weight.has_value() && down_weight.has_value() && ffn.has_value() &&
         projection_weight.has_value() && logits.has_value());
  assert(builder.add_operation("embedding", OperationKind::embedding,
                               {token.value(), table.value()}, {embedding.value()})
             .has_value());
  assert(builder.add_operation(
             "ffn", OperationKind::gated_dense_ffn,
             {embedding.value(), gate_weight.value(), up_weight.value(), down_weight.value()}, {ffn.value()})
             .has_value());
  assert(builder.add_operation("lm_head", OperationKind::lm_head,
                               {ffn.value(), projection_weight.value()}, {logits.value()})
             .has_value());
  const auto module = std::move(builder).build();
  assert(module.has_value());
  return {std::move(module).value(), token.value(), table.value(), embedding.value(), gate_weight.value(),
          up_weight.value(), down_weight.value(), ffn.value(), projection_weight.value()};
}

}  // namespace

int main() {
  using namespace superinfer;
  const auto module = make_fixture();
  const auto result = sm120::ReferenceExecutor::run(
      module, {{ir::semantic::TensorId{0}, {2}, {1.0F, 2.0F}},
               {ir::semantic::TensorId{1}, {2}, {3.0F, 4.0F}}});
  assert(result.has_value());
  assert(result.value().tensors.at(2).name == "sum");
  assert(result.value().find("norm") != nullptr);
  assert(result.value().tensors.at(2).values.at(0) == 4.0F);
  assert(result.value().tensors.at(2).values.at(1) == 6.0F);
  const float scale = std::sqrt((16.0F + 36.0F) / 2.0F + 1.0e-5F);
  assert(std::abs(result.value().tensors.at(3).values.at(0) - 4.0F / scale) < 1.0e-5F);
  assert(std::abs(result.value().tensors.at(3).values.at(1) - 6.0F / scale) < 1.0e-5F);

  const auto replay = sm120::ReferenceExecutor::run(
      module, {{ir::semantic::TensorId{0}, {2}, {1.0F, 2.0F}},
               {ir::semantic::TensorId{1}, {2}, {3.0F, 4.0F}}});
  assert(replay.has_value());
  assert(replay.value().tensors.at(3).values == result.value().tensors.at(3).values);

  const WeightedFixture weighted = make_weighted_fixture();
  const auto weighted_result = sm120::ReferenceExecutor::run(
      weighted.module,
      {{weighted.token, {1}, {2.0F}},
       {weighted.table, {3, 2}, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F}},
       {weighted.gate_weight, {2, 2}, {1.0F, 0.0F, 0.0F, 1.0F}},
       {weighted.up_weight, {2, 2}, {1.0F, 0.0F, 0.0F, 1.0F}},
       {weighted.down_weight, {2, 2}, {1.0F, 0.0F, 0.0F, 1.0F}},
       {weighted.projection_weight, {2, 2}, {1.0F, 0.0F, 0.0F, 1.0F}}});
  assert(weighted_result.has_value());
  assert(weighted_result.value().find("embedding") != nullptr);
  assert(weighted_result.value().find("logits") != nullptr);
  const float expected_first = 5.0F / (1.0F + std::exp(-5.0F)) * 5.0F;
  const float expected_second = 6.0F / (1.0F + std::exp(-6.0F)) * 6.0F;
  assert(std::abs(weighted_result.value().find("logits")->values[0] - expected_first) < 1.0e-5F);
  assert(std::abs(weighted_result.value().find("logits")->values[1] - expected_second) < 1.0e-5F);
  return 0;
}
