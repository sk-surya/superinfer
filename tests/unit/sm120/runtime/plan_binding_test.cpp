#include <superinfer/runtime/plan_binding.hpp>

#include <cassert>

int main() {
  using namespace superinfer;
  ir::physical::PlanBuilder builder;
  builder.set_resource_bounds({32, 0, 1});
  const auto buffer = builder.add_buffer(0, 16, 16);
  assert(buffer.has_value());
  assert(builder.add_command(base::KernelId{5}, {buffer.value()}, {}, 0, 0, 0).has_value());
  const auto plan = std::move(builder).finalize({120, "baseline-v1"});
  assert(plan.has_value());

  auto binding = runtime::PlanBinding::create(plan.value(), 120, "baseline-v1");
  assert(binding.has_value());
  assert(binding.value().execute().ok());
  assert(binding.value().trace().commands_executed == 1);
  assert(binding.value().trace().entries_executed == 1);
  assert(binding.value().execute().ok());
  assert(binding.value().trace().commands_executed == 2);

  const auto rejected = runtime::PlanBinding::create(plan.value(), 120, "other-v1");
  assert(!rejected.has_value());
  assert(rejected.error().code() == base::StatusCode::unsupported);

  ir::physical::PlanBuilder reordered_builder;
  reordered_builder.set_resource_bounds({32, 0, 2});
  assert(reordered_builder.add_buffer(0, 16, 16).has_value());
  assert(reordered_builder
             .add_command(base::KernelId{5}, {buffer.value()}, {ir::physical::CommandId{1}}, 0, 0,
                          0)
             .has_value());
  assert(reordered_builder
             .add_command(base::KernelId{6}, {buffer.value()}, {}, 0, 0, 0)
             .has_value());
  const auto reordered_plan = std::move(reordered_builder).finalize({120, "baseline-v1"});
  assert(reordered_plan.has_value());
  auto reordered = runtime::PlanBinding::create(reordered_plan.value(), 120, "baseline-v1");
  assert(reordered.has_value());
  assert(reordered.value().execute().ok());
  assert(reordered.value().trace().last_kernel_value == 5);

  ir::physical::PlanBuilder zero_kernel_builder;
  zero_kernel_builder.set_resource_bounds({0, 0, 1});
  assert(zero_kernel_builder.add_command(base::KernelId{}, {}, {}, 0, 0, 0).has_value());
  const auto zero_kernel_plan = std::move(zero_kernel_builder).finalize({120, "baseline-v1"});
  assert(zero_kernel_plan.has_value());
  const auto zero_kernel = runtime::PlanBinding::create(zero_kernel_plan.value(), 120, "baseline-v1");
  assert(!zero_kernel.has_value());
  assert(zero_kernel.error().code() == base::StatusCode::failed_precondition);
  return 0;
}
