#include <sm120/runtime/cuda_plan_executor.cuh>

#include <cassert>

namespace {

superinfer::ir::physical::Plan make_plan() {
  using namespace superinfer;
  ir::physical::PlanBuilder builder;
  builder.set_resource_bounds({64, 0, 2});
  assert(builder.add_buffer(0, 16, 16).has_value());
  assert(builder
             .add_command(base::KernelId{5}, {ir::physical::BufferId{0}},
                          {ir::physical::CommandId{1}}, 0, 0, 0)
             .has_value());
  assert(builder.add_command(base::KernelId{6}, {ir::physical::BufferId{0}}, {}, 0, 0, 0)
             .has_value());
  const auto plan = std::move(builder).finalize({120, "baseline-v1"});
  assert(plan.has_value());
  return std::move(plan).value();
}

}  // namespace

int main() {
  int device_count = 0;
  assert(cudaGetDeviceCount(&device_count) == cudaSuccess);
  if (device_count == 0) return 77;
  assert(cudaSetDevice(0) == cudaSuccess);

  using namespace superinfer;
  const auto plan = make_plan();
  auto session = sm120::cuda_runtime::CudaPlanSession::create(plan, 120, "baseline-v1");
  assert(session.has_value());
  assert(session.value().device_arena_bytes() == 64);
  assert(session.value().execute().ok());
  assert(session.value().execute().ok());
  assert(session.value().execute().ok());
  assert(session.value().synchronize_for_test().ok());
  assert(session.value().trace().commands_executed == 6);
  assert(session.value().trace().launches == 6);

  const auto rejected_target = sm120::cuda_runtime::CudaPlanSession::create(plan, 89, "baseline-v1");
  assert(!rejected_target.has_value());
  assert(rejected_target.error().code() == base::StatusCode::unsupported);

  ir::physical::PlanBuilder zero_builder;
  zero_builder.set_resource_bounds({0, 0, 1});
  assert(zero_builder.add_command(base::KernelId{}, {}, {}, 0, 0, 0).has_value());
  const auto zero_plan = std::move(zero_builder).finalize({120, "baseline-v1"});
  assert(zero_plan.has_value());
  const auto rejected_kernel = sm120::cuda_runtime::CudaPlanSession::create(
      zero_plan.value(), 120, "baseline-v1");
  assert(!rejected_kernel.has_value());
  assert(rejected_kernel.error().code() == base::StatusCode::failed_precondition);
  return 0;
}
