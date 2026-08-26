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
  return 0;
}
