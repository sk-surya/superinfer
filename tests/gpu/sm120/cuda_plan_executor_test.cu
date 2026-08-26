#include <sm120/runtime/cuda_plan_executor.cuh>

#include <cassert>
#include <array>
#include <cmath>
#include <cstddef>

namespace {

superinfer::ir::physical::Plan make_plan() {
  using namespace superinfer;
  ir::physical::PlanBuilder builder;
  builder.set_resource_bounds({64, 0, 2});
  assert(builder.add_buffer(0, 16, 16).has_value());
  assert(builder
             .add_command(base::KernelId{2}, {ir::physical::BufferId{0}},
                          {ir::physical::CommandId{1}}, 0, 0, 0)
             .has_value());
  assert(builder.add_command(base::KernelId{6}, {ir::physical::BufferId{0}}, {}, 0, 0, 0)
             .has_value());
  const auto plan = std::move(builder).finalize({120, "baseline-v1"});
  assert(plan.has_value());
  return std::move(plan).value();
}

superinfer::ir::physical::Plan make_numeric_plan() {
  using namespace superinfer;
  ir::physical::PlanBuilder builder;
  builder.set_resource_bounds({64, 0, 2});
  for (std::uint64_t offset = 0; offset < 64; offset += 16) {
    assert(builder.add_buffer(offset, 16, 16).has_value());
  }
  assert(builder
             .add_command(base::KernelId{1}, {ir::physical::BufferId{2}, ir::physical::BufferId{3}},
                          {ir::physical::CommandId{1}}, 0, 0, 0)
             .has_value());
  assert(builder
             .add_command(base::KernelId{4},
                          {ir::physical::BufferId{0}, ir::physical::BufferId{1},
                           ir::physical::BufferId{2}},
                          {}, 0, 0, 0)
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

  const auto numeric_plan = make_numeric_plan();
  auto numeric = sm120::cuda_runtime::CudaPlanSession::create(numeric_plan, 120, "baseline-v1");
  assert(numeric.has_value());
  const std::array<float, 4> left{1.0F, 2.0F, 3.0F, 4.0F};
  const std::array<float, 4> right{5.0F, 6.0F, 7.0F, 8.0F};
  assert(numeric.value()
             .copy_to_device(ir::physical::BufferId{0},
                              base::ConstByteView(reinterpret_cast<const std::byte*>(left.data()),
                                                  sizeof(left)))
             .ok());
  assert(numeric.value()
             .copy_to_device(ir::physical::BufferId{1},
                              base::ConstByteView(reinterpret_cast<const std::byte*>(right.data()),
                                                  sizeof(right)))
             .ok());
  assert(numeric.value().execute().ok());
  assert(numeric.value().synchronize_for_test().ok());
  std::array<float, 4> sum{};
  std::array<float, 4> copied{};
  assert(numeric.value()
             .copy_from_device(ir::physical::BufferId{2},
                               base::ByteView(reinterpret_cast<std::byte*>(sum.data()), sizeof(sum)))
             .ok());
  assert(numeric.value()
             .copy_from_device(ir::physical::BufferId{3},
                               base::ByteView(reinterpret_cast<std::byte*>(copied.data()),
                                              sizeof(copied)))
             .ok());
  assert((sum == std::array<float, 4>{6.0F, 8.0F, 10.0F, 12.0F}));
  assert(copied == sum);

  ir::physical::PlanBuilder norm_builder;
  norm_builder.set_resource_bounds({32, 0, 1});
  assert(norm_builder.add_buffer(0, 16, 16).has_value());
  assert(norm_builder.add_buffer(16, 16, 16).has_value());
  assert(norm_builder
             .add_command(base::KernelId{5}, {ir::physical::BufferId{0}, ir::physical::BufferId{1}},
                          {}, 0, 0, 0)
             .has_value());
  const auto norm_plan = std::move(norm_builder).finalize({120, "baseline-v1"});
  assert(norm_plan.has_value());
  auto norm = sm120::cuda_runtime::CudaPlanSession::create(norm_plan.value(), 120, "baseline-v1");
  assert(norm.has_value());
  assert(norm.value()
             .copy_to_device(ir::physical::BufferId{0},
                              base::ConstByteView(reinterpret_cast<const std::byte*>(left.data()),
                                                  sizeof(left)))
             .ok());
  assert(norm.value().execute().ok());
  assert(norm.value().synchronize_for_test().ok());
  std::array<float, 4> normalized{};
  assert(norm.value()
             .copy_from_device(ir::physical::BufferId{1},
                               base::ByteView(reinterpret_cast<std::byte*>(normalized.data()),
                                              sizeof(normalized)))
             .ok());
  const float denominator = std::sqrt((1.0F + 4.0F + 9.0F + 16.0F) / 4.0F + 1.0e-5F);
  for (std::size_t index = 0; index < normalized.size(); ++index) {
    assert(std::abs(normalized[index] - left[index] / denominator) < 1.0e-5F);
  }
  return 0;
}
