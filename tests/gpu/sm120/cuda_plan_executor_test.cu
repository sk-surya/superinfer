#include <sm120/runtime/cuda_plan_executor.cuh>
#include <sm120/compiler/specializer.h>
#include <sm120/kernels/baseline/provider.h>

#include <cassert>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <utility>
#include <vector>

namespace {

superinfer::ir::physical::PhysicalTensorDescriptor typed_tensor(
    superinfer::ir::physical::PhysicalDType dtype, std::vector<std::uint64_t> shape,
    superinfer::ir::physical::StorageEncoding encoding =
        superinfer::ir::physical::StorageEncoding::none) {
  return {dtype, std::move(shape), superinfer::ir::physical::PhysicalLayout::row_major, 0,
          encoding};
}

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

superinfer::ir::physical::Plan make_silu_mul_plan() {
  using namespace superinfer;
  ir::physical::PlanBuilder builder;
  builder.set_resource_bounds({48, 0, 1});
  assert(builder.add_buffer(0, 16, 16,
                            typed_tensor(ir::physical::PhysicalDType::f32, {4})).has_value());
  assert(builder.add_buffer(16, 16, 16,
                            typed_tensor(ir::physical::PhysicalDType::f32, {4})).has_value());
  assert(builder.add_buffer(32, 16, 16,
                            typed_tensor(ir::physical::PhysicalDType::f32, {4})).has_value());
  assert(builder.add_command(base::KernelId{18},
                             {ir::physical::BufferId{0}, ir::physical::BufferId{1},
                              ir::physical::BufferId{2}}, {}, 0, 0, 0).has_value());
  const auto plan = std::move(builder).finalize({120, "baseline-v1"});
  assert(plan.has_value());
  return std::move(plan).value();
}

superinfer::ir::physical::Plan make_sigmoid_mul_plan() {
  using namespace superinfer;
  ir::physical::PlanBuilder builder;
  builder.set_resource_bounds({48, 0, 1});
  assert(builder.add_buffer(0, 16, 16,
                            typed_tensor(ir::physical::PhysicalDType::f32, {4})).has_value());
  assert(builder.add_buffer(16, 16, 16,
                            typed_tensor(ir::physical::PhysicalDType::f32, {4})).has_value());
  assert(builder.add_buffer(32, 16, 16,
                            typed_tensor(ir::physical::PhysicalDType::f32, {4})).has_value());
  assert(builder.add_command(base::KernelId{19},
                             {ir::physical::BufferId{0}, ir::physical::BufferId{1},
                              ir::physical::BufferId{2}}, {}, 0, 0, 0).has_value());
  const auto plan = std::move(builder).finalize({120, "baseline-v1"});
  assert(plan.has_value());
  return std::move(plan).value();
}

superinfer::ir::physical::Plan make_rope_plan() {
  using namespace superinfer;
  ir::physical::PlanBuilder builder;
  builder.set_resource_bounds({32, 0, 1});
  assert(builder.add_buffer(0, 16, 16,
                            typed_tensor(ir::physical::PhysicalDType::f32, {1, 4})).has_value());
  assert(builder.add_buffer(16, 16, 16,
                            typed_tensor(ir::physical::PhysicalDType::f32, {1, 4})).has_value());
  assert(builder.add_command(base::KernelId{20},
                             {ir::physical::BufferId{0}, ir::physical::BufferId{1}}, {}, 0, 0, 0,
                             1.0e-5F, 10000.0F, {}, false, {}, {1, 4, 4, 1}).has_value());
  const auto plan = std::move(builder).finalize({120, "baseline-v1"});
  assert(plan.has_value());
  return std::move(plan).value();
}

superinfer::ir::physical::Plan make_cache_append_plan() {
  using namespace superinfer;
  ir::physical::PlanBuilder builder;
  builder.set_resource_bounds({48, 0, 1});
  assert(builder.add_buffer(0, 8, 8,
                            typed_tensor(ir::physical::PhysicalDType::f32, {1, 2})).has_value());
  assert(builder.add_buffer(8, 8, 8,
                            typed_tensor(ir::physical::PhysicalDType::f32, {1, 2})).has_value());
  assert(builder.add_buffer(16, 8, 8,
                            typed_tensor(ir::physical::PhysicalDType::bf16, {2, 1, 2})).has_value());
  assert(builder.add_buffer(32, 8, 8,
                            typed_tensor(ir::physical::PhysicalDType::bf16, {2, 1, 2})).has_value());
  assert(builder.add_command(base::KernelId{22},
                             {ir::physical::BufferId{0}, ir::physical::BufferId{1},
                              ir::physical::BufferId{2}, ir::physical::BufferId{3}}, {}, 0, 0, 0,
                             1.0e-5F, 1.0F, {}, false, {}, {}, {1, 2, 1, 2}).has_value());
  const auto plan = std::move(builder).finalize({120, "baseline-v1"});
  assert(plan.has_value());
  return std::move(plan).value();
}

superinfer::ir::physical::Plan make_bf16_cache_attention_plan() {
  using namespace superinfer;
  ir::physical::PlanBuilder builder;
  builder.set_resource_bounds({56, 0, 1});
  assert(builder.add_buffer(0, 16, 16,
                            typed_tensor(ir::physical::PhysicalDType::f32, {2, 2})).has_value());
  assert(builder.add_buffer(16, 8, 8,
                            typed_tensor(ir::physical::PhysicalDType::bf16, {2, 1, 2})).has_value());
  assert(builder.add_buffer(24, 8, 8,
                            typed_tensor(ir::physical::PhysicalDType::bf16, {2, 1, 2})).has_value());
  assert(builder.add_buffer(32, 16, 16,
                            typed_tensor(ir::physical::PhysicalDType::f32, {2, 2})).has_value());
  assert(builder.add_command(base::KernelId{23},
                             {ir::physical::BufferId{0}, ir::physical::BufferId{1},
                              ir::physical::BufferId{2}, ir::physical::BufferId{3}}, {}, 0, 0, 0,
                             1.0e-5F, 1.0F, {2, 1, 2, 2, 0, 0}).has_value());
  const auto plan = std::move(builder).finalize({120, "baseline-v1"});
  assert(plan.has_value());
  return std::move(plan).value();
}

superinfer::ir::physical::Plan make_split_plan() {
  using namespace superinfer;
  ir::physical::PlanBuilder builder;
  builder.set_resource_bounds({32, 0, 1});
  assert(builder.add_buffer(0, 16, 16,
                            typed_tensor(ir::physical::PhysicalDType::f32, {4})).has_value());
  assert(builder.add_buffer(16, 8, 8,
                            typed_tensor(ir::physical::PhysicalDType::f32, {2})).has_value());
  assert(builder.add_buffer(24, 8, 8,
                            typed_tensor(ir::physical::PhysicalDType::f32, {2})).has_value());
  assert(builder.add_command(base::KernelId{21},
                             {ir::physical::BufferId{0}, ir::physical::BufferId{1},
                              ir::physical::BufferId{2}}, {}, 0, 0, 0).has_value());
  const auto plan = std::move(builder).finalize({120, "baseline-v1"});
  assert(plan.has_value());
  return std::move(plan).value();
}

superinfer::ir::physical::Plan make_gated_delta_parameters_plan() {
  using namespace superinfer;
  ir::physical::PlanBuilder builder;
  builder.set_resource_bounds({48, 0, 1});
  for (std::uint64_t offset = 0; offset < 48; offset += 8) {
    assert(builder.add_buffer(offset, 8, 8,
                              typed_tensor(ir::physical::PhysicalDType::f32, {2})).has_value());
  }
  assert(builder.add_command(base::KernelId{24},
                             {ir::physical::BufferId{0}, ir::physical::BufferId{1},
                              ir::physical::BufferId{2}, ir::physical::BufferId{3},
                              ir::physical::BufferId{4}, ir::physical::BufferId{5}}, {}, 0, 0, 0)
             .has_value());
  const auto plan = std::move(builder).finalize({120, "baseline-v1"});
  assert(plan.has_value());
  return std::move(plan).value();
}

superinfer::ir::physical::Plan make_causal_conv_plan() {
  using namespace superinfer;
  ir::physical::PlanBuilder builder;
  builder.set_resource_bounds({40, 0, 1});
  assert(builder.add_buffer(0, 8, 8,
                            typed_tensor(ir::physical::PhysicalDType::f32, {2})).has_value());
  assert(builder.add_buffer(8, 16, 8,
                            typed_tensor(ir::physical::PhysicalDType::f32, {2, 2})).has_value());
  assert(builder.add_buffer(24, 8, 8,
                            typed_tensor(ir::physical::PhysicalDType::bf16, {2, 2})).has_value());
  assert(builder.add_buffer(32, 8, 8,
                            typed_tensor(ir::physical::PhysicalDType::f32, {2})).has_value());
  assert(builder.add_command(base::KernelId{25},
                             {ir::physical::BufferId{0}, ir::physical::BufferId{1},
                              ir::physical::BufferId{2}, ir::physical::BufferId{3}}, {}, 0, 0, 0,
                             1.0e-5F, 1.0F, {}, false, {}, {}, {}, {2, 2}).has_value());
  const auto plan = std::move(builder).finalize({120, "baseline-v1"});
  assert(plan.has_value());
  return std::move(plan).value();
}

superinfer::ir::physical::Plan make_embedding_plan() {
  using namespace superinfer;
  ir::physical::PlanBuilder builder;
  builder.set_resource_bounds({48, 0, 1});
  assert(builder.add_buffer(0, 4, 4, typed_tensor(ir::physical::PhysicalDType::int32, {1})).has_value());
  assert(builder.add_buffer(8, 24, 8, typed_tensor(ir::physical::PhysicalDType::f32, {3, 2})).has_value());
  assert(builder.add_buffer(40, 8, 8, typed_tensor(ir::physical::PhysicalDType::f32, {1, 2})).has_value());
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
  assert(builder.add_buffer(0, 4, 4, typed_tensor(ir::physical::PhysicalDType::int32, {1})).has_value());
  assert(builder.add_buffer(8, 12, 8, typed_tensor(ir::physical::PhysicalDType::bf16, {3, 2})).has_value());
  assert(builder.add_buffer(24, 8, 8, typed_tensor(ir::physical::PhysicalDType::f32, {1, 2})).has_value());
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

superinfer::ir::physical::Plan make_bf16_to_f32_cast_plan() {
  using namespace superinfer;
  ir::physical::PlanBuilder builder;
  builder.set_resource_bounds({32, 0, 1});
  assert(builder.add_buffer(0, 8, 8, typed_tensor(ir::physical::PhysicalDType::bf16, {4})).has_value());
  assert(builder.add_buffer(16, 16, 8, typed_tensor(ir::physical::PhysicalDType::f32, {4})).has_value());
  assert(builder.add_command(base::KernelId{16},
                             {ir::physical::BufferId{0}, ir::physical::BufferId{1}}, {}, 0, 0, 0)
             .has_value());
  const auto plan = std::move(builder).finalize({120, "baseline-v1"});
  assert(plan.has_value());
  return std::move(plan).value();
}

superinfer::ir::physical::Plan make_f32_to_bf16_cast_plan() {
  using namespace superinfer;
  ir::physical::PlanBuilder builder;
  builder.set_resource_bounds({32, 0, 1});
  assert(builder.add_buffer(0, 16, 8, typed_tensor(ir::physical::PhysicalDType::f32, {4})).has_value());
  assert(builder.add_buffer(16, 8, 8, typed_tensor(ir::physical::PhysicalDType::bf16, {4})).has_value());
  assert(builder.add_command(base::KernelId{17},
                             {ir::physical::BufferId{0}, ir::physical::BufferId{1}}, {}, 0, 0, 0)
             .has_value());
  const auto plan = std::move(builder).finalize({120, "baseline-v1"});
  assert(plan.has_value());
  return std::move(plan).value();
}

superinfer::ir::physical::Plan make_nvfp4_dequantize_plan() {
  using namespace superinfer;
  ir::physical::PlanBuilder builder;
  builder.set_resource_bounds({80, 0, 1});
  assert(builder.add_buffer(0, 8, 8,
                            typed_tensor(ir::physical::PhysicalDType::u8, {16},
                                         ir::physical::StorageEncoding::nvfp4_packed))
             .has_value());
  assert(builder.add_buffer(8, 1, 1,
                            typed_tensor(ir::physical::PhysicalDType::u8, {1},
                                         ir::physical::StorageEncoding::fp8_e4m3_group_scale))
             .has_value());
  assert(builder.add_buffer(16, 64, 8, typed_tensor(ir::physical::PhysicalDType::f32, {16})).has_value());
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

superinfer::ir::physical::Plan make_lm_head_plan() {
  using namespace superinfer;
  ir::physical::PlanBuilder builder;
  builder.set_resource_bounds({48, 0, 1});
  assert(builder.add_buffer(0, 8, 8).has_value());
  assert(builder.add_buffer(8, 24, 8).has_value());
  assert(builder.add_buffer(32, 12, 4).has_value());
  assert(builder
             .add_command(base::KernelId{10},
                          {ir::physical::BufferId{0}, ir::physical::BufferId{1},
                           ir::physical::BufferId{2}},
                          {}, 0, 0, 0)
             .has_value());
  const auto plan = std::move(builder).finalize({120, "baseline-v1"});
  assert(plan.has_value());
  return std::move(plan).value();
}

superinfer::ir::physical::Plan make_ffn_plan() {
  using namespace superinfer;
  ir::physical::PlanBuilder builder;
  builder.set_resource_bounds({80, 0, 1});
  assert(builder.add_buffer(0, 8, 8).has_value());
  assert(builder.add_buffer(8, 16, 8).has_value());
  assert(builder.add_buffer(24, 16, 8).has_value());
  assert(builder.add_buffer(40, 16, 8).has_value());
  assert(builder.add_buffer(56, 8, 8).has_value());
  assert(builder
             .add_command(base::KernelId{11},
                          {ir::physical::BufferId{0}, ir::physical::BufferId{1},
                           ir::physical::BufferId{2}, ir::physical::BufferId{3},
                           ir::physical::BufferId{4}},
                          {}, 0, 0, 0)
             .has_value());
  const auto plan = std::move(builder).finalize({120, "baseline-v1"});
  assert(plan.has_value());
  return std::move(plan).value();
}

superinfer::ir::physical::Plan make_bf16_rms_norm_plan() {
  using namespace superinfer;
  ir::physical::PlanBuilder builder;
  builder.set_resource_bounds({40, 0, 1});
  assert(builder.add_buffer(0, 16, 8, typed_tensor(ir::physical::PhysicalDType::f32, {4})).has_value());
  assert(builder.add_buffer(16, 16, 8, typed_tensor(ir::physical::PhysicalDType::f32, {4})).has_value());
  assert(builder.add_buffer(32, 8, 8, typed_tensor(ir::physical::PhysicalDType::bf16, {4})).has_value());
  assert(builder
             .add_command(base::KernelId{12},
                          {ir::physical::BufferId{0}, ir::physical::BufferId{1},
                           ir::physical::BufferId{2}},
                          {}, 0, 0, 0)
             .has_value());
  const auto plan = std::move(builder).finalize({120, "baseline-v1"});
  assert(plan.has_value());
  return std::move(plan).value();
}

superinfer::ir::physical::Plan make_grouped_bf16_rms_norm_plan() {
  using namespace superinfer;
  ir::physical::PlanBuilder builder;
  builder.set_resource_bounds({40, 0, 1});
  assert(builder.add_buffer(0, 16, 16,
                            typed_tensor(ir::physical::PhysicalDType::f32, {2, 2})).has_value());
  assert(builder.add_buffer(16, 16, 16,
                            typed_tensor(ir::physical::PhysicalDType::f32, {2, 2})).has_value());
  assert(builder.add_buffer(32, 4, 4,
                            typed_tensor(ir::physical::PhysicalDType::bf16, {2})).has_value());
  assert(builder.add_command(base::KernelId{12},
                             {ir::physical::BufferId{0}, ir::physical::BufferId{1},
                              ir::physical::BufferId{2}}, {}, 0, 0, 0).has_value());
  const auto plan = std::move(builder).finalize({120, "baseline-v1"});
  assert(plan.has_value());
  return std::move(plan).value();
}

superinfer::ir::physical::Plan make_nvfp4_linear_plan() {
  using namespace superinfer;
  ir::physical::PlanBuilder builder;
  builder.set_resource_bounds({128, 0, 1});
  assert(builder.add_buffer(0, 64, 8, typed_tensor(ir::physical::PhysicalDType::f32, {16})).has_value());
  assert(builder.add_buffer(64, 32, 8,
                            typed_tensor(ir::physical::PhysicalDType::u8, {4, 8},
                                         ir::physical::StorageEncoding::nvfp4_packed))
             .has_value());
  assert(builder.add_buffer(96, 4, 1,
                            typed_tensor(ir::physical::PhysicalDType::u8, {4, 1},
                                         ir::physical::StorageEncoding::fp8_e4m3_group_scale))
             .has_value());
  assert(builder.add_buffer(104, 4, 4, typed_tensor(ir::physical::PhysicalDType::f32, {1})).has_value());
  assert(builder.add_buffer(112, 16, 8, typed_tensor(ir::physical::PhysicalDType::f32, {4})).has_value());
  assert(builder
             .add_command(base::KernelId{13},
                          {ir::physical::BufferId{0}, ir::physical::BufferId{1},
                           ir::physical::BufferId{2}, ir::physical::BufferId{3},
                           ir::physical::BufferId{4}},
                          {}, 0, 0, 0, 1.0e-5F, 2.0F)
             .has_value());
  const auto plan = std::move(builder).finalize({120, "baseline-v1"});
  assert(plan.has_value());
  return std::move(plan).value();
}

superinfer::ir::physical::Plan make_attention_plan() {
  using namespace superinfer;
  ir::physical::PlanBuilder builder;
  builder.set_resource_bounds({64, 0, 1});
  assert(builder.add_buffer(0, 16, 16).has_value());
  assert(builder.add_buffer(16, 16, 16).has_value());
  assert(builder.add_buffer(32, 16, 16).has_value());
  assert(builder.add_buffer(48, 16, 16).has_value());
  assert(builder
             .add_command(base::KernelId{14},
                          {ir::physical::BufferId{0}, ir::physical::BufferId{1},
                           ir::physical::BufferId{2}, ir::physical::BufferId{3}},
                          {}, 0, 0, 0, 1.0e-5F, 1.0F, {2, 1, 2, 2, 0, 0})
             .has_value());
  const auto plan = std::move(builder).finalize({120, "baseline-v1"});
  assert(plan.has_value());
  return std::move(plan).value();
}

superinfer::ir::physical::Plan make_gated_delta_plan() {
  using namespace superinfer;
  ir::physical::PlanBuilder builder;
  builder.set_resource_bounds({224, 0, 1});
  assert(builder.add_buffer(0, 16, 16).has_value());
  assert(builder.add_buffer(16, 16, 16).has_value());
  assert(builder.add_buffer(32, 32, 16).has_value());
  assert(builder.add_buffer(64, 16, 16).has_value());
  assert(builder.add_buffer(80, 16, 16).has_value());
  assert(builder.add_buffer(96, 32, 16).has_value());
  assert(builder.add_buffer(128, 32, 16).has_value());
  assert(builder
             .add_command(base::KernelId{15},
                          {ir::physical::BufferId{0}, ir::physical::BufferId{1},
                           ir::physical::BufferId{2}, ir::physical::BufferId{3},
                           ir::physical::BufferId{4}, ir::physical::BufferId{5},
                           ir::physical::BufferId{6}},
                          {}, 0, 0, 0, 1.0e-5F, 1.0F, {1, 1, 2, 2, 2, 2})
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

  const auto silu_mul_plan = make_silu_mul_plan();
  auto silu_mul = sm120::cuda_runtime::CudaPlanSession::create(
      silu_mul_plan, 120, "baseline-v1");
  assert(silu_mul.has_value());
  const std::array<float, 4> silu_gate{0.0F, 1.0F, -1.0F, 2.0F};
  const std::array<float, 4> silu_value{2.0F, 2.0F, 2.0F, 2.0F};
  assert(silu_mul.value().copy_to_device(
      ir::physical::BufferId{0}, base::ConstByteView(
          reinterpret_cast<const std::byte*>(silu_gate.data()), sizeof(silu_gate))).ok());
  assert(silu_mul.value().copy_to_device(
      ir::physical::BufferId{1}, base::ConstByteView(
          reinterpret_cast<const std::byte*>(silu_value.data()), sizeof(silu_value))).ok());
  assert(silu_mul.value().execute().ok());
  assert(silu_mul.value().synchronize_for_test().ok());
  std::array<float, 4> silu_output{};
  assert(silu_mul.value().copy_from_device(
      ir::physical::BufferId{2}, base::ByteView(
          reinterpret_cast<std::byte*>(silu_output.data()), sizeof(silu_output))).ok());
  for (std::size_t index = 0; index < silu_output.size(); ++index) {
    const float expected = silu_gate[index] / (1.0F + std::exp(-silu_gate[index])) * 2.0F;
    assert(std::abs(silu_output[index] - expected) < 1.0e-5F);
  }

  const auto sigmoid_mul_plan = make_sigmoid_mul_plan();
  auto sigmoid_mul = sm120::cuda_runtime::CudaPlanSession::create(
      sigmoid_mul_plan, 120, "baseline-v1");
  assert(sigmoid_mul.has_value());
  assert(sigmoid_mul.value().copy_to_device(
      ir::physical::BufferId{0}, base::ConstByteView(
          reinterpret_cast<const std::byte*>(silu_gate.data()), sizeof(silu_gate))).ok());
  assert(sigmoid_mul.value().copy_to_device(
      ir::physical::BufferId{1}, base::ConstByteView(
          reinterpret_cast<const std::byte*>(silu_value.data()), sizeof(silu_value))).ok());
  assert(sigmoid_mul.value().execute().ok());
  assert(sigmoid_mul.value().synchronize_for_test().ok());
  std::array<float, 4> sigmoid_output{};
  assert(sigmoid_mul.value().copy_from_device(
      ir::physical::BufferId{2}, base::ByteView(
          reinterpret_cast<std::byte*>(sigmoid_output.data()), sizeof(sigmoid_output))).ok());
  for (std::size_t index = 0; index < sigmoid_output.size(); ++index) {
    const float expected = 1.0F / (1.0F + std::exp(-silu_gate[index])) * 2.0F;
    assert(std::abs(sigmoid_output[index] - expected) < 1.0e-5F);
  }

  const auto rope_plan = make_rope_plan();
  auto rope = sm120::cuda_runtime::CudaPlanSession::create(rope_plan, 120, "baseline-v1");
  assert(rope.has_value());
  const std::array<float, 4> rope_input{1.0F, 0.0F, 0.0F, 1.0F};
  assert(rope.value().copy_to_device(
      ir::physical::BufferId{0}, base::ConstByteView(
          reinterpret_cast<const std::byte*>(rope_input.data()), sizeof(rope_input))).ok());
  assert(rope.value().execute().ok());
  assert(rope.value().synchronize_for_test().ok());
  std::array<float, 4> rope_output{};
  assert(rope.value().copy_from_device(
      ir::physical::BufferId{1}, base::ByteView(
          reinterpret_cast<std::byte*>(rope_output.data()), sizeof(rope_output))).ok());
  const float cosine_one = std::cos(1.0F);
  const float sine_one_hundredth = std::sin(0.01F);
  const float cosine_one_hundredth = std::cos(0.01F);
  const std::array<float, 4> expected_rope{cosine_one, -sine_one_hundredth,
                                           std::sin(1.0F), cosine_one_hundredth};
  for (std::size_t index = 0; index < rope_output.size(); ++index) {
    assert(std::abs(rope_output[index] - expected_rope[index]) < 1.0e-5F);
  }

  const auto cache_append_plan = make_cache_append_plan();
  auto cache_append = sm120::cuda_runtime::CudaPlanSession::create(
      cache_append_plan, 120, "baseline-v1");
  assert(cache_append.has_value());
  const std::array<float, 2> cache_keys{1.0F, 2.0F};
  const std::array<float, 2> cache_values{3.0F, 4.0F};
  for (const auto& upload : std::array<std::pair<ir::physical::BufferId, base::ConstByteView>, 2>{
           {{ir::physical::BufferId{0}, base::ConstByteView(
                reinterpret_cast<const std::byte*>(cache_keys.data()), sizeof(cache_keys))},
            {ir::physical::BufferId{1}, base::ConstByteView(
                reinterpret_cast<const std::byte*>(cache_values.data()), sizeof(cache_values))}}}) {
    assert(cache_append.value().copy_to_device(upload.first, upload.second).ok());
  }
  assert(cache_append.value().execute().ok());
  assert(cache_append.value().synchronize_for_test().ok());
  std::array<std::uint16_t, 4> cache_key_bits{};
  std::array<std::uint16_t, 4> cache_value_bits{};
  assert(cache_append.value().copy_from_device(
      ir::physical::BufferId{2}, base::ByteView(
          reinterpret_cast<std::byte*>(cache_key_bits.data()), sizeof(cache_key_bits))).ok());
  assert(cache_append.value().copy_from_device(
      ir::physical::BufferId{3}, base::ByteView(
          reinterpret_cast<std::byte*>(cache_value_bits.data()), sizeof(cache_value_bits))).ok());
  assert(cache_key_bits[2] == 0x3f80 && cache_key_bits[3] == 0x4000);
  assert(cache_value_bits[2] == 0x4040 && cache_value_bits[3] == 0x4080);

  const auto split_plan = make_split_plan();
  auto split = sm120::cuda_runtime::CudaPlanSession::create(split_plan, 120, "baseline-v1");
  assert(split.has_value());
  const std::array<float, 4> split_input{1.0F, 2.0F, 3.0F, 4.0F};
  assert(split.value().copy_to_device(
      ir::physical::BufferId{0}, base::ConstByteView(
          reinterpret_cast<const std::byte*>(split_input.data()), sizeof(split_input))).ok());
  assert(split.value().execute().ok());
  assert(split.value().synchronize_for_test().ok());
  std::array<float, 2> split_first{};
  std::array<float, 2> split_second{};
  assert(split.value().copy_from_device(
      ir::physical::BufferId{1}, base::ByteView(
          reinterpret_cast<std::byte*>(split_first.data()), sizeof(split_first))).ok());
  assert(split.value().copy_from_device(
      ir::physical::BufferId{2}, base::ByteView(
          reinterpret_cast<std::byte*>(split_second.data()), sizeof(split_second))).ok());
  assert((split_first == std::array<float, 2>{1.0F, 2.0F}));
  assert((split_second == std::array<float, 2>{3.0F, 4.0F}));

  const auto parameters_plan = make_gated_delta_parameters_plan();
  auto parameters = sm120::cuda_runtime::CudaPlanSession::create(
      parameters_plan, 120, "baseline-v1");
  assert(parameters.has_value());
  const float log_two = std::log(2.0F);
  const std::array<float, 2> parameter_input_a{0.0F, 1.0F};
  const std::array<float, 2> parameter_input_b{0.0F, 1.0F};
  const std::array<float, 2> parameter_a_log{0.0F, log_two};
  const std::array<float, 2> parameter_dt_bias{0.0F, 0.0F};
  for (const auto& upload : std::array<std::pair<ir::physical::BufferId, base::ConstByteView>, 4>{
           {{ir::physical::BufferId{0}, base::ConstByteView(
                reinterpret_cast<const std::byte*>(parameter_input_a.data()), sizeof(parameter_input_a))},
            {ir::physical::BufferId{1}, base::ConstByteView(
                reinterpret_cast<const std::byte*>(parameter_input_b.data()), sizeof(parameter_input_b))},
            {ir::physical::BufferId{2}, base::ConstByteView(
                reinterpret_cast<const std::byte*>(parameter_a_log.data()), sizeof(parameter_a_log))},
            {ir::physical::BufferId{3}, base::ConstByteView(
                reinterpret_cast<const std::byte*>(parameter_dt_bias.data()), sizeof(parameter_dt_bias))}}}) {
    assert(parameters.value().copy_to_device(upload.first, upload.second).ok());
  }
  assert(parameters.value().execute().ok());
  assert(parameters.value().synchronize_for_test().ok());
  std::array<float, 2> parameter_log_decay{};
  std::array<float, 2> parameter_beta{};
  assert(parameters.value().copy_from_device(
      ir::physical::BufferId{4}, base::ByteView(
          reinterpret_cast<std::byte*>(parameter_log_decay.data()), sizeof(parameter_log_decay))).ok());
  assert(parameters.value().copy_from_device(
      ir::physical::BufferId{5}, base::ByteView(
          reinterpret_cast<std::byte*>(parameter_beta.data()), sizeof(parameter_beta))).ok());
  assert(std::abs(parameter_log_decay[0] + std::log(2.0F)) < 1.0e-5F);
  assert(std::abs(parameter_log_decay[1] + 2.0F *
                 (1.0F + std::log1p(std::exp(-1.0F)))) < 1.0e-5F);
  assert(std::abs(parameter_beta[0] - 0.5F) < 1.0e-5F);
  assert(std::abs(parameter_beta[1] - 1.0F / (1.0F + std::exp(-1.0F))) < 1.0e-5F);

  const auto convolution_plan = make_causal_conv_plan();
  auto convolution = sm120::cuda_runtime::CudaPlanSession::create(
      convolution_plan, 120, "baseline-v1");
  assert(convolution.has_value());
  const std::array<float, 2> convolution_input{5.0F, 6.0F};
  const std::array<float, 4> convolution_weights{1.0F, 2.0F, 3.0F, 4.0F};
  const std::array<std::uint16_t, 4> convolution_state{0x3f80, 0x4000, 0x4040, 0x4080};
  assert(convolution.value().copy_to_device(
      ir::physical::BufferId{0}, base::ConstByteView(
          reinterpret_cast<const std::byte*>(convolution_input.data()), sizeof(convolution_input))).ok());
  assert(convolution.value().copy_to_device(
      ir::physical::BufferId{1}, base::ConstByteView(
          reinterpret_cast<const std::byte*>(convolution_weights.data()), sizeof(convolution_weights))).ok());
  assert(convolution.value().copy_to_device(
      ir::physical::BufferId{2}, base::ConstByteView(
          reinterpret_cast<const std::byte*>(convolution_state.data()), sizeof(convolution_state))).ok());
  assert(convolution.value().execute().ok());
  assert(convolution.value().synchronize_for_test().ok());
  std::array<float, 2> convolution_output{};
  std::array<std::uint16_t, 4> convolution_state_output{};
  assert(convolution.value().copy_from_device(
      ir::physical::BufferId{3}, base::ByteView(
          reinterpret_cast<std::byte*>(convolution_output.data()), sizeof(convolution_output))).ok());
  assert(convolution.value().copy_from_device(
      ir::physical::BufferId{2}, base::ByteView(
          reinterpret_cast<std::byte*>(convolution_state_output.data()), sizeof(convolution_state_output))).ok());
  assert(std::abs(convolution_output[0] - 13.0F / (1.0F + std::exp(-13.0F))) < 1.0e-5F);
  assert(std::abs(convolution_output[1] - 36.0F / (1.0F + std::exp(-36.0F))) < 1.0e-5F);
  assert((convolution_state_output == std::array<std::uint16_t, 4>{0x4040, 0x4080, 0x40a0, 0x40c0}));

  const auto bf16_cache_attention_plan = make_bf16_cache_attention_plan();
  auto bf16_cache_attention = sm120::cuda_runtime::CudaPlanSession::create(
      bf16_cache_attention_plan, 120, "baseline-v1");
  assert(bf16_cache_attention.has_value());
  const std::array<float, 4> bf16_cache_query{1.0F, 0.0F, 0.0F, 1.0F};
  const std::array<std::uint16_t, 4> bf16_cache_keys{0x3f80, 0x0000, 0x0000, 0x3f80};
  const std::array<std::uint16_t, 4> bf16_cache_values{0x4000, 0x0000, 0x0000, 0x4080};
  for (const auto& upload : std::array<std::pair<ir::physical::BufferId, base::ConstByteView>, 3>{
           {{ir::physical::BufferId{0}, base::ConstByteView(
                reinterpret_cast<const std::byte*>(bf16_cache_query.data()), sizeof(bf16_cache_query))},
            {ir::physical::BufferId{1}, base::ConstByteView(
                reinterpret_cast<const std::byte*>(bf16_cache_keys.data()), sizeof(bf16_cache_keys))},
            {ir::physical::BufferId{2}, base::ConstByteView(
                reinterpret_cast<const std::byte*>(bf16_cache_values.data()), sizeof(bf16_cache_values))}}}) {
    assert(bf16_cache_attention.value().copy_to_device(upload.first, upload.second).ok());
  }
  assert(bf16_cache_attention.value().execute().ok());
  assert(bf16_cache_attention.value().synchronize_for_test().ok());
  std::array<float, 4> bf16_cache_output{};
  assert(bf16_cache_attention.value().copy_from_device(
      ir::physical::BufferId{3}, base::ByteView(
          reinterpret_cast<std::byte*>(bf16_cache_output.data()), sizeof(bf16_cache_output))).ok());
  const float bf16_attention_probability = std::exp(1.0F / std::sqrt(2.0F)) /
                                           (std::exp(1.0F / std::sqrt(2.0F)) + 1.0F);
  assert(std::abs(bf16_cache_output[0] - 2.0F * bf16_attention_probability) < 1.0e-5F);
  assert(std::abs(bf16_cache_output[1] - 4.0F * (1.0F - bf16_attention_probability)) < 1.0e-5F);

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

  const auto bf16_to_f32_plan = make_bf16_to_f32_cast_plan();
  auto bf16_to_f32 = sm120::cuda_runtime::CudaPlanSession::create(
      bf16_to_f32_plan, 120, "baseline-v1");
  assert(bf16_to_f32.has_value());
  const std::array<std::uint16_t, 4> bf16_values{0x3f80, 0x4000, 0x4040, 0x4080};
  assert(bf16_to_f32.value().copy_to_device(
      ir::physical::BufferId{0},
      base::ConstByteView(reinterpret_cast<const std::byte*>(bf16_values.data()),
                          sizeof(bf16_values))).ok());
  assert(bf16_to_f32.value().execute().ok());
  assert(bf16_to_f32.value().synchronize_for_test().ok());
  std::array<float, 4> f32_values{};
  assert(bf16_to_f32.value().copy_from_device(
      ir::physical::BufferId{1},
      base::ByteView(reinterpret_cast<std::byte*>(f32_values.data()), sizeof(f32_values))).ok());
  assert((f32_values == std::array<float, 4>{1.0F, 2.0F, 3.0F, 4.0F}));

  const auto f32_to_bf16_plan = make_f32_to_bf16_cast_plan();
  auto f32_to_bf16 = sm120::cuda_runtime::CudaPlanSession::create(
      f32_to_bf16_plan, 120, "baseline-v1");
  assert(f32_to_bf16.has_value());
  const std::array<float, 4> f32_input{1.0F, 2.0F, 3.0F, 4.0F};
  assert(f32_to_bf16.value().copy_to_device(
      ir::physical::BufferId{0},
      base::ConstByteView(reinterpret_cast<const std::byte*>(f32_input.data()),
                          sizeof(f32_input))).ok());
  assert(f32_to_bf16.value().execute().ok());
  assert(f32_to_bf16.value().synchronize_for_test().ok());
  std::array<std::uint16_t, 4> bf16_output{};
  assert(f32_to_bf16.value().copy_from_device(
      ir::physical::BufferId{1},
      base::ByteView(reinterpret_cast<std::byte*>(bf16_output.data()), sizeof(bf16_output))).ok());
  assert(bf16_output == bf16_values);

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

  const auto attention_plan = make_attention_plan();
  auto attention = sm120::cuda_runtime::CudaPlanSession::create(
      attention_plan, 120, "baseline-v1");
  assert(attention.has_value());
  const std::array<float, 4> attention_query{1.0F, 0.0F, 0.0F, 1.0F};
  const std::array<float, 4> attention_keys{1.0F, 0.0F, 0.0F, 1.0F};
  const std::array<float, 4> attention_values{2.0F, 0.0F, 0.0F, 4.0F};
  assert(attention.value().copy_to_device(
      ir::physical::BufferId{0}, base::ConstByteView(
          reinterpret_cast<const std::byte*>(attention_query.data()), sizeof(attention_query))).ok());
  assert(attention.value().copy_to_device(
      ir::physical::BufferId{1}, base::ConstByteView(
          reinterpret_cast<const std::byte*>(attention_keys.data()), sizeof(attention_keys))).ok());
  assert(attention.value().copy_to_device(
      ir::physical::BufferId{2}, base::ConstByteView(
          reinterpret_cast<const std::byte*>(attention_values.data()), sizeof(attention_values))).ok());
  assert(attention.value().execute().ok());
  assert(attention.value().synchronize_for_test().ok());
  std::array<float, 4> attention_output{};
  assert(attention.value().copy_from_device(
      ir::physical::BufferId{3}, base::ByteView(
          reinterpret_cast<std::byte*>(attention_output.data()), sizeof(attention_output))).ok());
  const float attention_probability = std::exp(1.0F / std::sqrt(2.0F)) /
                                      (std::exp(1.0F / std::sqrt(2.0F)) + 1.0F);
  assert(std::abs(attention_output[0] - 2.0F * attention_probability) < 1.0e-5F);
  assert(std::abs(attention_output[1] - 4.0F * (1.0F - attention_probability)) < 1.0e-5F);

  const auto gated_delta_plan = make_gated_delta_plan();
  auto gated_delta = sm120::cuda_runtime::CudaPlanSession::create(
      gated_delta_plan, 120, "baseline-v1");
  assert(gated_delta.has_value());
  const std::array<float, 4> delta_query{1.0F, 0.0F, 0.0F, 1.0F};
  const std::array<float, 4> delta_key{1.0F, 0.0F, 0.0F, 1.0F};
  const std::array<float, 8> delta_value{2.0F, 3.0F, 4.0F, 5.0F,
                                         6.0F, 7.0F, 8.0F, 9.0F};
  const std::array<float, 4> delta_log_decay{0.0F, 0.0F, 0.0F, 0.0F};
  const std::array<float, 4> delta_beta{1.0F, 1.0F, 1.0F, 1.0F};
  const std::array<float, 8> zero_delta_state{};
  for (const auto& upload : std::array<std::pair<ir::physical::BufferId, base::ConstByteView>, 6>{
           {{ir::physical::BufferId{0}, base::ConstByteView(
                reinterpret_cast<const std::byte*>(delta_query.data()), sizeof(delta_query))},
            {ir::physical::BufferId{1}, base::ConstByteView(
                reinterpret_cast<const std::byte*>(delta_key.data()), sizeof(delta_key))},
            {ir::physical::BufferId{2}, base::ConstByteView(
                reinterpret_cast<const std::byte*>(delta_value.data()), sizeof(delta_value))},
            {ir::physical::BufferId{3}, base::ConstByteView(
                reinterpret_cast<const std::byte*>(delta_log_decay.data()), sizeof(delta_log_decay))},
            {ir::physical::BufferId{4}, base::ConstByteView(
                reinterpret_cast<const std::byte*>(delta_beta.data()), sizeof(delta_beta))},
            {ir::physical::BufferId{5}, base::ConstByteView(
                reinterpret_cast<const std::byte*>(zero_delta_state.data()), sizeof(zero_delta_state))}}}) {
    assert(gated_delta.value().copy_to_device(upload.first, upload.second).ok());
  }
  assert(gated_delta.value().execute().ok());
  assert(gated_delta.value().synchronize_for_test().ok());
  std::array<float, 8> delta_output{};
  std::array<float, 8> delta_state{};
  assert(gated_delta.value().copy_from_device(
      ir::physical::BufferId{6}, base::ByteView(
          reinterpret_cast<std::byte*>(delta_output.data()), sizeof(delta_output))).ok());
  assert(gated_delta.value().copy_from_device(
      ir::physical::BufferId{5}, base::ByteView(
          reinterpret_cast<std::byte*>(delta_state.data()), sizeof(delta_state))).ok());
  for (std::size_t index = 0; index < delta_output.size(); ++index) {
    assert(std::abs(delta_output[index] - delta_value[index] / std::sqrt(2.0F)) < 1.0e-5F);
  }
  const std::array<float, 8> expected_delta_state{2.0F, 3.0F, 6.0F, 7.0F,
                                                   4.0F, 5.0F, 8.0F, 9.0F};
  for (std::size_t index = 0; index < delta_state.size(); ++index) {
    assert(std::abs(delta_state[index] - expected_delta_state[index]) < 1.0e-5F);
  }

  const auto lm_head_plan = make_lm_head_plan();
  auto lm_head = sm120::cuda_runtime::CudaPlanSession::create(lm_head_plan, 120, "baseline-v1");
  assert(lm_head.has_value());
  const std::array<float, 2> lm_input{2.0F, 3.0F};
  const std::array<float, 6> lm_weights{1.0F, 0.0F, 0.0F, 1.0F, 1.0F, 1.0F};
  assert(lm_head.value().copy_to_device(
      ir::physical::BufferId{0},
      base::ConstByteView(reinterpret_cast<const std::byte*>(lm_input.data()),
                          sizeof(lm_input))).ok());
  assert(lm_head.value().copy_to_device(
      ir::physical::BufferId{1},
      base::ConstByteView(reinterpret_cast<const std::byte*>(lm_weights.data()),
                          sizeof(lm_weights))).ok());
  assert(lm_head.value().execute().ok());
  assert(lm_head.value().synchronize_for_test().ok());
  std::array<float, 3> lm_output{};
  assert(lm_head.value().copy_from_device(
      ir::physical::BufferId{2},
      base::ByteView(reinterpret_cast<std::byte*>(lm_output.data()), sizeof(lm_output))).ok());
  assert((lm_output == std::array<float, 3>{2.0F, 3.0F, 5.0F}));

  const auto ffn_plan = make_ffn_plan();
  auto ffn = sm120::cuda_runtime::CudaPlanSession::create(ffn_plan, 120, "baseline-v1");
  assert(ffn.has_value());
  const std::array<float, 2> ffn_input{2.0F, 3.0F};
  const std::array<float, 4> identity_weights{1.0F, 0.0F, 0.0F, 1.0F};
  for (const auto& upload : std::array<std::pair<ir::physical::BufferId, const float*>, 4>{
           {{ir::physical::BufferId{0}, ffn_input.data()},
            {ir::physical::BufferId{1}, identity_weights.data()},
            {ir::physical::BufferId{2}, identity_weights.data()},
            {ir::physical::BufferId{3}, identity_weights.data()}}}) {
    const std::size_t bytes = upload.first.value() == 0 ? sizeof(ffn_input) : sizeof(identity_weights);
    assert(ffn.value().copy_to_device(
        upload.first, base::ConstByteView(reinterpret_cast<const std::byte*>(upload.second), bytes))
               .ok());
  }
  assert(ffn.value().execute().ok());
  assert(ffn.value().synchronize_for_test().ok());
  std::array<float, 2> ffn_output{};
  assert(ffn.value().copy_from_device(
      ir::physical::BufferId{4},
      base::ByteView(reinterpret_cast<std::byte*>(ffn_output.data()), sizeof(ffn_output))).ok());
  assert(std::abs(ffn_output[0] - 2.0F / (1.0F + std::exp(-2.0F)) * 2.0F) < 1.0e-5F);
  assert(std::abs(ffn_output[1] - 3.0F / (1.0F + std::exp(-3.0F)) * 3.0F) < 1.0e-5F);

  const auto bf16_norm_plan = make_bf16_rms_norm_plan();
  auto bf16_norm = sm120::cuda_runtime::CudaPlanSession::create(
      bf16_norm_plan, 120, "baseline-v1");
  assert(bf16_norm.has_value());
  const std::array<float, 4> norm_input{1.0F, 2.0F, 3.0F, 4.0F};
  const std::array<std::uint16_t, 4> norm_scale{0x3f80, 0x3fc0, 0x4000, 0x4020};
  assert(bf16_norm.value().copy_to_device(
      ir::physical::BufferId{0},
      base::ConstByteView(reinterpret_cast<const std::byte*>(norm_input.data()),
                          sizeof(norm_input))).ok());
  assert(bf16_norm.value().copy_to_device(
      ir::physical::BufferId{2},
      base::ConstByteView(reinterpret_cast<const std::byte*>(norm_scale.data()),
                          sizeof(norm_scale))).ok());
  assert(bf16_norm.value().execute().ok());
  assert(bf16_norm.value().synchronize_for_test().ok());
  std::array<float, 4> norm_output{};
  assert(bf16_norm.value().copy_from_device(
      ir::physical::BufferId{1},
      base::ByteView(reinterpret_cast<std::byte*>(norm_output.data()), sizeof(norm_output))).ok());
  const float norm_denominator = std::sqrt((1.0F + 4.0F + 9.0F + 16.0F) / 4.0F + 1.0e-5F);
  for (std::size_t index = 0; index < norm_output.size(); ++index) {
    const float scale = 1.0F + static_cast<float>(index) * 0.5F;
    assert(std::abs(norm_output[index] - norm_input[index] / norm_denominator * scale) < 1.0e-5F);
  }

  const auto grouped_norm_plan = make_grouped_bf16_rms_norm_plan();
  auto grouped_norm = sm120::cuda_runtime::CudaPlanSession::create(
      grouped_norm_plan, 120, "baseline-v1");
  assert(grouped_norm.has_value());
  assert(grouped_norm.value().copy_to_device(
      ir::physical::BufferId{0}, base::ConstByteView(
          reinterpret_cast<const std::byte*>(norm_input.data()), sizeof(norm_input))).ok());
  assert(grouped_norm.value().copy_to_device(
      ir::physical::BufferId{2}, base::ConstByteView(
          reinterpret_cast<const std::byte*>(norm_scale.data()), sizeof(std::uint16_t) * 2)).ok());
  assert(grouped_norm.value().execute().ok());
  assert(grouped_norm.value().synchronize_for_test().ok());
  std::array<float, 4> grouped_norm_output{};
  assert(grouped_norm.value().copy_from_device(
      ir::physical::BufferId{1}, base::ByteView(
          reinterpret_cast<std::byte*>(grouped_norm_output.data()), sizeof(grouped_norm_output))).ok());
  const float first_row_denominator = std::sqrt((1.0F + 4.0F) / 2.0F + 1.0e-5F);
  const float second_row_denominator = std::sqrt((9.0F + 16.0F) / 2.0F + 1.0e-5F);
  assert(std::abs(grouped_norm_output[0] - 1.0F / first_row_denominator) < 1.0e-5F);
  assert(std::abs(grouped_norm_output[1] - 2.0F / first_row_denominator * 1.5F) < 1.0e-5F);
  assert(std::abs(grouped_norm_output[2] - 3.0F / second_row_denominator) < 1.0e-5F);
  assert(std::abs(grouped_norm_output[3] - 4.0F / second_row_denominator * 1.5F) < 1.0e-5F);

  const auto nvfp4_linear_plan = make_nvfp4_linear_plan();
  auto nvfp4_linear = sm120::cuda_runtime::CudaPlanSession::create(
      nvfp4_linear_plan, 120, "baseline-v1");
  assert(nvfp4_linear.has_value());
  std::array<std::uint8_t, 8> nvfp4_linear_packed{};
  nvfp4_linear_packed.fill(0x11);
  const std::uint8_t nvfp4_linear_scale = 0x38;
  const float nvfp4_linear_tensor_scale = 2.0F;
  std::array<float, 16> ones{};
  ones.fill(1.0F);
  assert(nvfp4_linear.value().copy_to_device(
      ir::physical::BufferId{0},
      base::ConstByteView(reinterpret_cast<const std::byte*>(ones.data()), sizeof(ones))).ok());
  assert(nvfp4_linear.value().copy_to_device(
      ir::physical::BufferId{1},
      base::ConstByteView(reinterpret_cast<const std::byte*>(nvfp4_linear_packed.data()),
                          sizeof(nvfp4_linear_packed))).ok());
  assert(nvfp4_linear.value().copy_to_device(
      ir::physical::BufferId{2},
      base::ConstByteView(reinterpret_cast<const std::byte*>(&nvfp4_linear_scale),
                          sizeof(nvfp4_linear_scale))).ok());
  assert(nvfp4_linear.value().copy_to_device(
      ir::physical::BufferId{3},
      base::ConstByteView(reinterpret_cast<const std::byte*>(&nvfp4_linear_tensor_scale),
                          sizeof(nvfp4_linear_tensor_scale))).ok());
  assert(nvfp4_linear.value().execute().ok());
  assert(nvfp4_linear.value().synchronize_for_test().ok());
  float nvfp4_linear_output = 0.0F;
  assert(nvfp4_linear.value().copy_from_device(
      ir::physical::BufferId{4},
      base::ByteView(reinterpret_cast<std::byte*>(&nvfp4_linear_output),
                     sizeof(nvfp4_linear_output))).ok());
  assert(std::abs(nvfp4_linear_output - 16.0F) < 1.0e-5F);

  ir::physical::PlanBuilder norm_builder;
  norm_builder.set_resource_bounds({48, 0, 1});
  assert(norm_builder.add_buffer(0, 16, 16).has_value());
  assert(norm_builder.add_buffer(16, 16, 16).has_value());
  assert(norm_builder.add_buffer(32, 16, 16).has_value());
  assert(norm_builder
             .add_command(base::KernelId{5}, {ir::physical::BufferId{0}, ir::physical::BufferId{1},
                                               ir::physical::BufferId{2}},
                          {}, 0, 0, 0, 1.0e-6F, 1.0F, {}, true)
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
    assert(std::abs(normalized[index] - left[index] / denominator * (scale[index] + 1.0F)) <
           1.0e-5F);
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
                                     {lowered_input.value(), lowered_scale.value(),
                                      lowered_output.value()})
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
