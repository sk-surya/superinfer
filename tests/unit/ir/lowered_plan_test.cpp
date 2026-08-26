#include <superinfer/ir/lowered/module.hpp>
#include <superinfer/runtime/physical_plan.h>

#include <cassert>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

int main() {
  using namespace superinfer;
  using namespace ir::lowered;

  ModuleBuilder lowered_builder;
  const auto lowered_tensor = lowered_builder.add_tensor(
      ir::semantic::TensorId{0}, {2, 4}, LayoutKind::row_major,
      base::MemorySpace::device, 16, ir::semantic::DType::f16, ir::semantic::DType::f32);
  assert(lowered_tensor.has_value());
  assert(lowered_builder
             .add_fusion("norm_residual", std::vector<LoweredTensorId>{lowered_tensor.value()})
             .ok());
  assert(lowered_builder.add_kernel_requirement("rms_norm", 120, {lowered_tensor.value()}).ok());
  const auto lowered = std::move(lowered_builder).build();
  assert(lowered.has_value());
  assert(lowered.value().verify().ok());
  assert(lowered.value().dump().find("row_major") != std::string::npos);
  std::ifstream lowered_golden(std::filesystem::path{SUPERINFER_SOURCE_DIR} /
                               "tests/golden/ir/lowered-basic.txt");
  assert(lowered_golden.good());
  const std::string lowered_expected{std::istreambuf_iterator<char>{lowered_golden}, {}};
  assert(lowered.value().dump() == lowered_expected);

  ir::physical::PlanBuilder plan_builder;
  plan_builder.set_resource_bounds({128, 64, 4});
  const auto first_buffer = plan_builder.add_buffer(0, 32, 16);
  const auto second_buffer = plan_builder.add_buffer(32, 32, 16);
  assert(first_buffer.has_value() && second_buffer.has_value());
  const auto first_command = plan_builder.add_command(
      base::KernelId{1}, {first_buffer.value()}, {}, 0, 0, 0);
  assert(first_command.has_value());
  const auto second_command = plan_builder.add_command(
      base::KernelId{2}, {second_buffer.value()}, {first_command.value()}, 0, 16, 0);
  assert(second_command.has_value());
  const auto plan = std::move(plan_builder).finalize({120, "fixture-catalog"});
  assert(plan.has_value());
  assert(plan.value().verify().ok());
  assert(plan.value().commands().size() == 2);
  std::ifstream physical_golden(std::filesystem::path{SUPERINFER_SOURCE_DIR} /
                                "tests/golden/ir/physical-basic.txt");
  assert(physical_golden.good());
  const std::string physical_expected{std::istreambuf_iterator<char>{physical_golden}, {}};
  assert(plan.value().dump() == physical_expected);

  ir::physical::PlanBuilder overlap_builder;
  overlap_builder.set_resource_bounds({64, 0, 2});
  assert(overlap_builder.add_buffer(0, 32, 16).has_value());
  assert(overlap_builder.add_buffer(16, 16, 16).has_value());
  const auto overlap = std::move(overlap_builder).finalize({120, "fixture-catalog"});
  assert(!overlap.has_value());
  assert(overlap.error().message().find("overlap") != std::string::npos);

  ir::physical::PlanBuilder bad_dependency_builder;
  bad_dependency_builder.set_resource_bounds({64, 0, 2});
  const auto buffer = bad_dependency_builder.add_buffer(0, 16, 16);
  assert(buffer.has_value());
  const auto command_a = bad_dependency_builder.add_command(
      base::KernelId{1}, {buffer.value()}, {ir::physical::CommandId{1}}, 0, 0, 0);
  assert(command_a.has_value());
  const auto bad_dependency = std::move(bad_dependency_builder).finalize({120, "fixture-catalog"});
  assert(!bad_dependency.has_value());
  assert(bad_dependency.error().message().find("dependency") != std::string::npos);

  ir::physical::PlanBuilder cycle_builder;
  cycle_builder.set_resource_bounds({64, 0, 2});
  const auto cycle_buffer = cycle_builder.add_buffer(0, 16, 16);
  assert(cycle_buffer.has_value());
  assert(cycle_builder
             .add_command(base::KernelId{1}, {cycle_buffer.value()}, {ir::physical::CommandId{1}}, 0,
                          0, 0)
             .has_value());
  assert(cycle_builder
             .add_command(base::KernelId{2}, {cycle_buffer.value()}, {ir::physical::CommandId{0}}, 0,
                          0, 0)
             .has_value());
  const auto cycle = std::move(cycle_builder).finalize({120, "fixture-catalog"});
  assert(!cycle.has_value());
  assert(cycle.error().message().find("cycle") != std::string::npos);

  ir::physical::PlanBuilder typed_builder;
  typed_builder.set_resource_bounds({64, 0, 1});
  ir::physical::PhysicalTensorDescriptor typed_tensor;
  typed_tensor.dtype = ir::physical::PhysicalDType::bf16;
  typed_tensor.shape = {2, 4};
  typed_tensor.layout = ir::physical::PhysicalLayout::row_major;
  const auto typed_buffer = typed_builder.add_buffer(0, 16, 16, typed_tensor);
  assert(typed_buffer.has_value());
  assert(typed_builder
             .add_command(base::KernelId{1}, {typed_buffer.value()}, {}, 0, 0, 0)
             .has_value());
  const auto typed_plan = std::move(typed_builder).finalize({120, "fixture-catalog"});
  assert(typed_plan.has_value());
  assert(typed_plan.value().commands().front().operands.front().dtype ==
         ir::physical::PhysicalDType::bf16);
  assert(typed_plan.value().commands().front().operands.front().shape == std::vector<std::uint64_t>({2, 4}));
  return 0;
}
