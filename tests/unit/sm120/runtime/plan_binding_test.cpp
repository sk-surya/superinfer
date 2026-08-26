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
  return 0;
}
