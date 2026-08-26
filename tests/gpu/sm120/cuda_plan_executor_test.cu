#include <sm120/runtime/cuda_plan_executor.cuh>
#include <sm120/compiler/specializer.h>
#include <sm120/kernels/baseline/provider.h>

#include <cassert>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <utility>

namespace {

__global__ void injected_async_fault() {
  if (threadIdx.x == 0) *static_cast<volatile std::uint32_t*>(nullptr) = 1U;
}

superinfer::ir::physical::Plan make_plan() {
  using namespace superinfer;
  ir::physical::PlanBuilder builder;
  builder.set_resource_bounds({48, 0, 2});
  assert(builder.add_buffer(0, 16, 16).has_value());
  assert(builder.add_buffer(16, 16, 16).has_value());
  assert(builder.add_buffer(32, 16, 16).has_value());
  assert(builder
             .add_command(base::KernelId{4},
                          {ir::physical::BufferId{0}, ir::physical::BufferId{1},
                           ir::physical::BufferId{2}},
                          {ir::physical::CommandId{1}}, 0, 0, 0)
             .has_value());
  assert(builder
             .add_command(base::KernelId{4},
                          {ir::physical::BufferId{0}, ir::physical::BufferId{1},
                           ir::physical::BufferId{2}}, {}, 0, 0, 0)
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

superinfer::ir::physical::Plan make_embedding_plan() {
  using namespace superinfer;
  ir::physical::PlanBuilder builder;
  builder.set_resource_bounds({48, 0, 1});
  assert(builder.add_buffer(0, 4, 4).has_value());
  assert(builder.add_buffer(8, 24, 8).has_value());
  assert(builder.add_buffer(40, 8, 8).has_value());
  assert(builder
             .add_command(base::KernelId{7},
                          {ir::physical::BufferId{0}, ir::physical::BufferId{1},
                           ir::physical::BufferId{2}},
                          {}, 0, 0, 0)
             .has_value());
  const auto plan = std::move(builder).finalize({120, "baseline-v1"});
  assert(plan.has_value());
  return std::move(plan).value();
}

superinfer::ir::physical::Plan make_bf16_embedding_plan() {
  using namespace superinfer;
  ir::physical::PlanBuilder builder;
  builder.set_resource_bounds({32, 0, 1});
  assert(builder.add_buffer(0, 4, 4).has_value());
  assert(builder.add_buffer(8, 12, 8).has_value());
  assert(builder.add_buffer(24, 8, 8).has_value());
  assert(builder
             .add_command(base::KernelId{8},
                          {ir::physical::BufferId{0}, ir::physical::BufferId{1},
                           ir::physical::BufferId{2}},
                          {}, 0, 0, 0)
             .has_value());
  const auto plan = std::move(builder).finalize({120, "baseline-v1"});
  assert(plan.has_value());
  return std::move(plan).value();
}

superinfer::ir::physical::Plan make_nvfp4_dequantize_plan() {
  using namespace superinfer;
  ir::physical::PlanBuilder builder;
  builder.set_resource_bounds({80, 0, 1});
  assert(builder.add_buffer(0, 8, 8).has_value());
  assert(builder.add_buffer(8, 1, 1).has_value());
  assert(builder.add_buffer(16, 64, 8).has_value());
  assert(builder
             .add_command(base::KernelId{9},
                          {ir::physical::BufferId{0}, ir::physical::BufferId{1},
                           ir::physical::BufferId{2}},
                          {}, 0, 0, 0, 1.0e-5F, 2.0F)
             .has_value());
  const auto plan = std::move(builder).finalize({120, "baseline-v1"});
  assert(plan.has_value());
  return std::move(plan).value();
}

}  // namespace

int main() {
  int device_count = 0;
  const cudaError_t count_error = cudaGetDeviceCount(&device_count);
  if (count_error == cudaErrorNoDevice || count_error == cudaErrorInsufficientDriver) return 77;
  assert(count_error == cudaSuccess);
  if (device_count == 0) return 77;
  const cudaError_t set_error = cudaSetDevice(0);
  if (set_error == cudaErrorNoDevice || set_error == cudaErrorInsufficientDriver) return 77;
  assert(set_error == cudaSuccess);
  cudaDeviceProp properties{};
  const cudaError_t property_error = cudaGetDeviceProperties(&properties, 0);
  if (property_error == cudaErrorNoDevice || property_error == cudaErrorInsufficientDriver) return 77;
  assert(property_error == cudaSuccess);
  if (properties.major != 12 || properties.minor != 0) return 77;

  using namespace superinfer;
  const auto plan = make_plan();
  auto session = sm120::cuda_runtime::CudaPlanSession::create(plan, 120, "baseline-v1");
  assert(session.has_value());
  assert(session.value().device_arena_bytes() == 48);
  assert(session.value().lifecycle_trace().device_allocations == 1);
  assert(session.value().lifecycle_trace().stream_creations == 1);
  assert(session.value().lifecycle_trace().event_creations == 2);
  assert(session.value().lifecycle_trace().kernel_bindings == 2);
  assert(session.value().lifecycle_trace().device_synchronizations == 0);
  assert(session.value().execute().ok());
  assert(session.value().execute().ok());
  assert(session.value().execute().ok());
  assert(session.value().lifecycle_trace().device_synchronizations == 0);
  assert(session.value().synchronize_for_test().ok());
  assert(session.value().lifecycle_trace().device_synchronizations == 1);
  assert(session.value().trace().commands_executed == 6);
  assert(session.value().trace().launches == 6);

  const auto rejected_target = sm120::cuda_runtime::CudaPlanSession::create(plan, 89, "baseline-v1");
  assert(!rejected_target.has_value());
  assert(rejected_target.error().code() == base::StatusCode::unsupported);

  ir::physical::PlanBuilder future_catalog_builder;
  future_catalog_builder.set_resource_bounds({0, 0, 1});
  const auto future_catalog_plan = std::move(future_catalog_builder).finalize({120, "future-v2"});
  assert(future_catalog_plan.has_value());
  const auto rejected_catalog = sm120::cuda_runtime::CudaPlanSession::create(
      future_catalog_plan.value(), 120, "future-v2");
  assert(!rejected_catalog.has_value());
  assert(rejected_catalog.error().code() == base::StatusCode::unsupported);

  ir::physical::PlanBuilder zero_builder;
  zero_builder.set_resource_bounds({0, 0, 1});
  assert(zero_builder.add_command(base::KernelId{}, {}, {}, 0, 0, 0).has_value());
  const auto zero_plan = std::move(zero_builder).finalize({120, "baseline-v1"});
  assert(zero_plan.has_value());
  const auto rejected_kernel = sm120::cuda_runtime::CudaPlanSession::create(
      zero_plan.value(), 120, "baseline-v1");
  assert(!rejected_kernel.has_value());
  assert(rejected_kernel.error().code() == base::StatusCode::failed_precondition);

  ir::physical::PlanBuilder unknown_builder;
  unknown_builder.set_resource_bounds({0, 0, 1});
  assert(unknown_builder.add_command(base::KernelId{99}, {}, {}, 0, 0, 0).has_value());
  const auto unknown_plan = std::move(unknown_builder).finalize({120, "baseline-v1"});
  assert(unknown_plan.has_value());
  const auto rejected_unknown = sm120::cuda_runtime::CudaPlanSession::create(
      unknown_plan.value(), 120, "baseline-v1");
  assert(!rejected_unknown.has_value());
  assert(rejected_unknown.error().code() == base::StatusCode::unsupported);

  ir::physical::PlanBuilder mismatch_builder;
  mismatch_builder.set_resource_bounds({32, 0, 1});
  assert(mismatch_builder.add_buffer(0, 16, 16).has_value());
  assert(mismatch_builder.add_buffer(16, 8, 8).has_value());
  assert(mismatch_builder
             .add_command(base::KernelId{1},
                          {ir::physical::BufferId{0}, ir::physical::BufferId{1}}, {}, 0, 0, 0)
             .has_value());
  const auto mismatch_plan = std::move(mismatch_builder).finalize({120, "baseline-v1"});
  assert(mismatch_plan.has_value());
  const auto rejected_mismatch = sm120::cuda_runtime::CudaPlanSession::create(
      mismatch_plan.value(), 120, "baseline-v1");
  assert(!rejected_mismatch.has_value());
  assert(rejected_mismatch.error().code() == base::StatusCode::invalid_argument);

  ir::physical::PlanBuilder workspace_builder;
  workspace_builder.set_resource_bounds({48, 16, 1});
  for (std::uint64_t offset = 0; offset < 48; offset += 16) {
    assert(workspace_builder.add_buffer(offset, 16, 16).has_value());
  }
  assert(workspace_builder
             .add_command(base::KernelId{4},
                          {ir::physical::BufferId{0}, ir::physical::BufferId{1},
                           ir::physical::BufferId{2}}, {}, 0, 0, 8)
             .has_value());
  const auto workspace_plan = std::move(workspace_builder).finalize({120, "baseline-v1"});
  assert(workspace_plan.has_value());
  const auto rejected_workspace = sm120::cuda_runtime::CudaPlanSession::create(
      workspace_plan.value(), 120, "baseline-v1");
  assert(!rejected_workspace.has_value());
  assert(rejected_workspace.error().code() == base::StatusCode::unsupported);

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

  const auto embedding_plan = make_embedding_plan();
  auto embedding = sm120::cuda_runtime::CudaPlanSession::create(embedding_plan, 120, "baseline-v1");
  assert(embedding.has_value());
  const std::uint32_t token = 2;
  const std::array<float, 6> table{1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
  assert(embedding.value().copy_to_device(
      ir::physical::BufferId{0},
      base::ConstByteView(reinterpret_cast<const std::byte*>(&token), sizeof(token))).ok());
  assert(embedding.value().copy_to_device(
      ir::physical::BufferId{1},
      base::ConstByteView(reinterpret_cast<const std::byte*>(table.data()), sizeof(table))).ok());
  assert(embedding.value().execute().ok());
  assert(embedding.value().synchronize_for_test().ok());
  std::array<float, 2> embedded{};
  assert(embedding.value().copy_from_device(
      ir::physical::BufferId{2},
      base::ByteView(reinterpret_cast<std::byte*>(embedded.data()), sizeof(embedded))).ok());
  assert((embedded == std::array<float, 2>{5.0F, 6.0F}));

  const auto bf16_embedding_plan = make_bf16_embedding_plan();
  auto bf16_embedding = sm120::cuda_runtime::CudaPlanSession::create(
      bf16_embedding_plan, 120, "baseline-v1");
  assert(bf16_embedding.has_value());
  const std::array<std::uint16_t, 6> bf16_table{0x3f80, 0x4000, 0x4040,
                                                 0x4080, 0x40a0, 0x40c0};
  assert(bf16_embedding.value().copy_to_device(
      ir::physical::BufferId{0},
      base::ConstByteView(reinterpret_cast<const std::byte*>(&token), sizeof(token))).ok());
  assert(bf16_embedding.value().copy_to_device(
      ir::physical::BufferId{1},
      base::ConstByteView(reinterpret_cast<const std::byte*>(bf16_table.data()),
                          sizeof(bf16_table))).ok());
  assert(bf16_embedding.value().execute().ok());
  assert(bf16_embedding.value().synchronize_for_test().ok());
  std::array<float, 2> bf16_embedded{};
  assert(bf16_embedding.value().copy_from_device(
      ir::physical::BufferId{2},
      base::ByteView(reinterpret_cast<std::byte*>(bf16_embedded.data()),
                     sizeof(bf16_embedded))).ok());
  assert((bf16_embedded == std::array<float, 2>{5.0F, 6.0F}));

  const auto nvfp4_plan = make_nvfp4_dequantize_plan();
  auto nvfp4 = sm120::cuda_runtime::CudaPlanSession::create(nvfp4_plan, 120, "baseline-v1");
  assert(nvfp4.has_value());
  const std::array<std::uint8_t, 8> packed_nvfp4{
      0x10, 0x32, 0x54, 0x76, 0x98, 0xba, 0xdc, 0xfe};
  const std::uint8_t scale_nvfp4 = 0x38;  // E4M3 1.0.
  assert(nvfp4.value().copy_to_device(
      ir::physical::BufferId{0},
      base::ConstByteView(reinterpret_cast<const std::byte*>(packed_nvfp4.data()),
                          sizeof(packed_nvfp4))).ok());
  assert(nvfp4.value().copy_to_device(
      ir::physical::BufferId{1},
      base::ConstByteView(reinterpret_cast<const std::byte*>(&scale_nvfp4),
                          sizeof(scale_nvfp4))).ok());
  assert(nvfp4.value().execute().ok());
  assert(nvfp4.value().synchronize_for_test().ok());
  std::array<float, 16> dequantized_nvfp4{};
  assert(nvfp4.value().copy_from_device(
      ir::physical::BufferId{2},
      base::ByteView(reinterpret_cast<std::byte*>(dequantized_nvfp4.data()),
                     sizeof(dequantized_nvfp4))).ok());
  const std::array<float, 16> expected_nvfp4{
      0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 6.0F, 8.0F, 12.0F,
      -0.0F, -1.0F, -2.0F, -3.0F, -4.0F, -6.0F, -8.0F, -12.0F};
  for (std::size_t index = 0; index < expected_nvfp4.size(); ++index) {
    assert(std::abs(dequantized_nvfp4[index] - expected_nvfp4[index]) < 1.0e-5F);
  }

  ir::physical::PlanBuilder norm_builder;
  norm_builder.set_resource_bounds({48, 0, 1});
  assert(norm_builder.add_buffer(0, 16, 16).has_value());
  assert(norm_builder.add_buffer(16, 16, 16).has_value());
  assert(norm_builder.add_buffer(32, 16, 16).has_value());
  assert(norm_builder
             .add_command(base::KernelId{5}, {ir::physical::BufferId{0}, ir::physical::BufferId{1},
                                               ir::physical::BufferId{2}},
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
  const std::array<float, 4> scale{1.0F, 1.5F, 2.0F, 2.5F};
  assert(norm.value()
             .copy_to_device(ir::physical::BufferId{2},
                             base::ConstByteView(reinterpret_cast<const std::byte*>(scale.data()),
                                                 sizeof(scale)))
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
    assert(std::abs(normalized[index] - left[index] / denominator * scale[index]) < 1.0e-5F);
  }

  ir::physical::PlanBuilder layer_builder;
  layer_builder.set_resource_bounds({64, 0, 1});
  for (std::uint64_t offset = 0; offset < 64; offset += 16) {
    assert(layer_builder.add_buffer(offset, 16, 16).has_value());
  }
  assert(layer_builder
             .add_command(base::KernelId{6},
                          {ir::physical::BufferId{0}, ir::physical::BufferId{1},
                           ir::physical::BufferId{2}, ir::physical::BufferId{3}},
                          {}, 0, 0, 0)
             .has_value());
  const auto layer_plan = std::move(layer_builder).finalize({120, "baseline-v1"});
  assert(layer_plan.has_value());
  auto layer = sm120::cuda_runtime::CudaPlanSession::create(layer_plan.value(), 120, "baseline-v1");
  assert(layer.has_value());
  const std::array<float, 4> bias{0.1F, 0.2F, 0.3F, 0.4F};
  for (const auto& upload : std::array<std::pair<ir::physical::BufferId, const std::array<float, 4>*>, 3>{
           {{ir::physical::BufferId{0}, &left}, {ir::physical::BufferId{2}, &scale},
            {ir::physical::BufferId{3}, &bias}}}) {
    assert(layer.value()
               .copy_to_device(upload.first,
                               base::ConstByteView(reinterpret_cast<const std::byte*>(upload.second->data()),
                                                   sizeof(float) * upload.second->size()))
               .ok());
  }
  assert(layer.value().execute().ok());
  assert(layer.value().synchronize_for_test().ok());
  std::array<float, 4> layered{};
  assert(layer.value()
             .copy_from_device(ir::physical::BufferId{1},
                               base::ByteView(reinterpret_cast<std::byte*>(layered.data()),
                                              sizeof(layered)))
             .ok());
  const float layer_mean = 2.5F;
  const float layer_denominator = std::sqrt((2.25F + 0.25F + 0.25F + 2.25F) / 4.0F + 1.0e-5F);
  for (std::size_t index = 0; index < layered.size(); ++index) {
    const float expected = (left[index] - layer_mean) / layer_denominator * scale[index] + bias[index];
    assert(std::abs(layered[index] - expected) < 1.0e-5F);
  }

  // Exercise the compiler-produced operand binding, not only the hand-authored fixture above.
  ir::lowered::ModuleBuilder lowered_builder;
  const auto lowered_input = lowered_builder.add_tensor(
      ir::semantic::TensorId{0}, {4}, ir::lowered::LayoutKind::row_major,
      base::MemorySpace::device, 16, ir::semantic::DType::f32, ir::semantic::DType::f32);
  const auto lowered_output = lowered_builder.add_tensor(
      ir::semantic::TensorId{1}, {4}, ir::lowered::LayoutKind::row_major,
      base::MemorySpace::device, 16, ir::semantic::DType::f32, ir::semantic::DType::f32);
  const auto lowered_scale = lowered_builder.add_tensor(
      ir::semantic::TensorId{2}, {4}, ir::lowered::LayoutKind::row_major,
      base::MemorySpace::device, 16, ir::semantic::DType::f32, ir::semantic::DType::f32);
  assert(lowered_input.has_value() && lowered_output.has_value() && lowered_scale.has_value());
  assert(lowered_builder
             .add_kernel_requirement("rms_norm", 120,
                                     {lowered_input.value(), lowered_output.value(),
                                      lowered_scale.value()})
             .ok());
  const auto lowered = std::move(lowered_builder).build();
  assert(lowered.has_value());
  sm120::BaselineProvider provider;
  const auto specialized = sm120::Specializer{}.compile(
      lowered.value(), {compiler::TargetProfile::offline_sm120a(1ULL << 30U, "baseline-v1"),
                        256, 8}, provider);
  assert(specialized.has_value());
  auto compiled_session = sm120::cuda_runtime::CudaPlanSession::create(
      specialized.value().plan, 120, "baseline-v1");
  assert(compiled_session.has_value());
  assert(specialized.value().plan.commands().front().buffers.size() == 3);
  assert(compiled_session.value()
             .copy_to_device(ir::physical::BufferId{0},
                             base::ConstByteView(reinterpret_cast<const std::byte*>(left.data()),
                                                 sizeof(left)))
             .ok());
  assert(compiled_session.value()
             .copy_to_device(ir::physical::BufferId{2},
                             base::ConstByteView(reinterpret_cast<const std::byte*>(scale.data()),
                                                 sizeof(scale)))
             .ok());
  assert(compiled_session.value().execute().ok());
  assert(compiled_session.value().synchronize_for_test().ok());
  std::array<float, 4> compiled_output{};
  assert(compiled_session.value()
             .copy_from_device(ir::physical::BufferId{1},
                               base::ByteView(reinterpret_cast<std::byte*>(compiled_output.data()),
                                              sizeof(compiled_output)))
             .ok());
  for (std::size_t index = 0; index < compiled_output.size(); ++index) {
    assert(std::abs(compiled_output[index] - left[index] / denominator * scale[index]) < 1.0e-5F);
  }

  if (std::getenv("SUPERINFER_INJECT_ASYNC_FAULT") != nullptr) {
    auto poisoned = sm120::cuda_runtime::CudaPlanSession::create(plan, 120, "baseline-v1");
    assert(poisoned.has_value());
    injected_async_fault<<<1, 1>>>();
    assert(cudaGetLastError() == cudaSuccess);
    assert(!poisoned.value().synchronize_for_test().ok());
    assert(poisoned.value().poisoned());
    assert(!poisoned.value().execute().ok());
  }
  return 0;
}
