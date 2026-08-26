#include <superinfer/compiler/target.h>
#include <superinfer/ir/lowered/module.hpp>
#include <sm120/compiler/specializer.h>
#include <sm120/kernels/baseline/provider.h>

#include <cassert>
#include <string>

namespace {

class RejectingProvider final : public superinfer::kernels::KernelProvider {
 public:
  superinfer::base::Result<std::vector<superinfer::kernels::KernelCandidate>> enumerate(
      const superinfer::kernels::KernelQuery&) const override {
    return superinfer::base::Status::unsupported("provider fixture rejection");
  }
};

superinfer::ir::lowered::Module make_fixture() {
  using namespace superinfer;
  ir::lowered::ModuleBuilder builder;
  const auto hidden = builder.add_tensor(
      ir::semantic::TensorId{0}, {2, 4}, ir::lowered::LayoutKind::row_major,
      base::MemorySpace::device, 16, ir::semantic::DType::f16, ir::semantic::DType::f32);
  const auto output = builder.add_tensor(
      ir::semantic::TensorId{1}, {2, 4}, ir::lowered::LayoutKind::row_major,
      base::MemorySpace::device, 16, ir::semantic::DType::f16, ir::semantic::DType::f32);
  const auto scale = builder.add_tensor(
      ir::semantic::TensorId{2}, {2, 4}, ir::lowered::LayoutKind::row_major,
      base::MemorySpace::device, 16, ir::semantic::DType::f32, ir::semantic::DType::f32);
  assert(hidden.has_value() && output.has_value() && scale.has_value());
  assert(builder
             .add_kernel_requirement("rms_norm", 120,
                                     {hidden.value(), scale.value(), output.value()})
             .ok());
  const auto module = std::move(builder).build();
  assert(module.has_value());
  return std::move(module).value();
}

superinfer::ir::lowered::Module make_bf16_embedding_fixture() {
  using namespace superinfer;
  ir::lowered::ModuleBuilder builder;
  const auto token = builder.add_tensor(
      ir::semantic::TensorId{0}, {1}, ir::lowered::LayoutKind::row_major,
      base::MemorySpace::device, 16, ir::semantic::DType::int32, ir::semantic::DType::f32);
  const auto table = builder.add_tensor(
      ir::semantic::TensorId{1}, {4, 2}, ir::lowered::LayoutKind::row_major,
      base::MemorySpace::device, 16, ir::semantic::DType::bf16, ir::semantic::DType::f32,
      ir::semantic::TensorRole::weight);
  const auto output = builder.add_tensor(
      ir::semantic::TensorId{2}, {1, 2}, ir::lowered::LayoutKind::row_major,
      base::MemorySpace::device, 16, ir::semantic::DType::f32, ir::semantic::DType::f32);
  assert(token.has_value() && table.has_value() && output.has_value());
  assert(builder.add_kernel_requirement(
                         "embedding", 120, {token.value(), table.value(), output.value()})
             .ok());
  const auto module = std::move(builder).build();
  assert(module.has_value());
  return std::move(module).value();
}

superinfer::ir::lowered::Module make_f32_lm_head_fixture() {
  using namespace superinfer;
  ir::lowered::ModuleBuilder builder;
  const auto input = builder.add_tensor(
      ir::semantic::TensorId{0}, {1, 2}, ir::lowered::LayoutKind::row_major,
      base::MemorySpace::device, 16, ir::semantic::DType::f32, ir::semantic::DType::f32);
  const auto weights = builder.add_tensor(
      ir::semantic::TensorId{1}, {3, 2}, ir::lowered::LayoutKind::row_major,
      base::MemorySpace::device, 16, ir::semantic::DType::f32, ir::semantic::DType::f32,
      ir::semantic::TensorRole::weight);
  const auto output = builder.add_tensor(
      ir::semantic::TensorId{2}, {1, 3}, ir::lowered::LayoutKind::row_major,
      base::MemorySpace::device, 16, ir::semantic::DType::f32, ir::semantic::DType::f32,
      ir::semantic::TensorRole::logits);
  assert(input.has_value() && weights.has_value() && output.has_value());
  assert(builder.add_kernel_requirement(
                         "lm_head", 120, {input.value(), weights.value(), output.value()})
             .ok());
  const auto module = std::move(builder).build();
  assert(module.has_value());
  return std::move(module).value();
}

superinfer::ir::lowered::Module make_f32_ffn_fixture() {
  using namespace superinfer;
  ir::lowered::ModuleBuilder builder;
  const auto input = builder.add_tensor(
      ir::semantic::TensorId{0}, {1, 2}, ir::lowered::LayoutKind::row_major,
      base::MemorySpace::device, 16, ir::semantic::DType::f32, ir::semantic::DType::f32);
  const auto gate = builder.add_tensor(
      ir::semantic::TensorId{1}, {2, 2}, ir::lowered::LayoutKind::row_major,
      base::MemorySpace::device, 16, ir::semantic::DType::f32, ir::semantic::DType::f32,
      ir::semantic::TensorRole::weight);
  const auto up = builder.add_tensor(
      ir::semantic::TensorId{2}, {2, 2}, ir::lowered::LayoutKind::row_major,
      base::MemorySpace::device, 16, ir::semantic::DType::f32, ir::semantic::DType::f32,
      ir::semantic::TensorRole::weight);
  const auto down = builder.add_tensor(
      ir::semantic::TensorId{3}, {2, 2}, ir::lowered::LayoutKind::row_major,
      base::MemorySpace::device, 16, ir::semantic::DType::f32, ir::semantic::DType::f32,
      ir::semantic::TensorRole::weight);
  const auto output = builder.add_tensor(
      ir::semantic::TensorId{4}, {1, 2}, ir::lowered::LayoutKind::row_major,
      base::MemorySpace::device, 16, ir::semantic::DType::f32, ir::semantic::DType::f32);
  assert(input.has_value() && gate.has_value() && up.has_value() && down.has_value() &&
         output.has_value());
  assert(builder.add_kernel_requirement(
                         "gated_dense_ffn", 120,
                         {input.value(), gate.value(), up.value(), down.value(), output.value()})
             .ok());
  const auto module = std::move(builder).build();
  assert(module.has_value());
  return std::move(module).value();
}

superinfer::ir::lowered::Module make_bf16_rms_norm_fixture() {
  using namespace superinfer;
  ir::lowered::ModuleBuilder builder;
  const auto input = builder.add_tensor(
      ir::semantic::TensorId{0}, {4}, ir::lowered::LayoutKind::row_major,
      base::MemorySpace::device, 16, ir::semantic::DType::f32, ir::semantic::DType::f32);
  const auto output = builder.add_tensor(
      ir::semantic::TensorId{1}, {4}, ir::lowered::LayoutKind::row_major,
      base::MemorySpace::device, 16, ir::semantic::DType::f32, ir::semantic::DType::f32);
  const auto scale = builder.add_tensor(
      ir::semantic::TensorId{2}, {4}, ir::lowered::LayoutKind::row_major,
      base::MemorySpace::device, 16, ir::semantic::DType::bf16, ir::semantic::DType::f32,
      ir::semantic::TensorRole::weight);
  assert(input.has_value() && output.has_value() && scale.has_value());
  assert(builder.add_kernel_requirement(
                         "rms_norm", 120, {input.value(), scale.value(), output.value()})
             .ok());
  const auto module = std::move(builder).build();
  assert(module.has_value());
  return std::move(module).value();
}

}  // namespace

int main() {
  using namespace superinfer;
  const auto target = compiler::TargetProfile::offline_sm120a(1ULL << 30U, "baseline-v1");
  sm120::Specializer specializer;
  sm120::BaselineProvider provider;
  const auto result = specializer.compile(make_fixture(), {target, 256, 64}, provider);
  assert(result.has_value());
  assert(result.value().plan.verify().ok());
  assert(result.value().plan.capability().target_capability == 120);
  assert(result.value().plan.capability().kernel_catalog == "baseline-v1");
  assert(result.value().memory.device_arena_bytes == 64);
  assert(result.value().memory.allocations.size() == 3);
  assert(result.value().memory.allocations.front().offset == 0);
  assert(result.value().plan.commands().size() == 1);
  assert(result.value().plan.commands().front().kernel.value() == 5);
  assert(result.value().plan.commands().front().buffers.size() == 3);

  const auto bf16_result = specializer.compile(
      make_bf16_embedding_fixture(), {target, 256, 64}, provider);
  assert(bf16_result.has_value());
  assert(bf16_result.value().plan.commands().front().kernel.value() == 8);

  const auto lm_result = specializer.compile(
      make_f32_lm_head_fixture(), {target, 256, 64}, provider);
  assert(lm_result.has_value());
  assert(lm_result.value().plan.commands().front().kernel.value() == 10);

  const auto ffn_result = specializer.compile(
      make_f32_ffn_fixture(), {target, 256, 64}, provider);
  assert(ffn_result.has_value());
  assert(ffn_result.value().plan.commands().front().kernel.value() == 11);

  const auto bf16_norm_result = specializer.compile(
      make_bf16_rms_norm_fixture(), {target, 256, 64}, provider);
  assert(bf16_norm_result.has_value());
  assert(bf16_norm_result.value().plan.commands().front().kernel.value() == 12);

  const auto second = specializer.compile(make_fixture(), {target, 256, 64}, provider);
  assert(second.has_value());
  assert(second.value().plan.dump() == result.value().plan.dump());
  assert(second.value().memory.dump() == result.value().memory.dump());

  auto incompatible = target;
  incompatible.compute_capability = 89;
  const auto rejected = specializer.compile(make_fixture(), {incompatible, 256, 64}, provider);
  assert(!rejected.has_value());
  assert(rejected.error().code() == base::StatusCode::unsupported);

  const auto low_budget = specializer.compile(
      make_fixture(), {compiler::TargetProfile::offline_sm120a(8, "baseline-v1"), 256, 64}, provider);
  assert(!low_budget.has_value());
  assert(low_budget.error().code() == base::StatusCode::resource_exhausted);

  ir::lowered::ModuleBuilder role_builder;
  const auto kv_tensor = role_builder.add_tensor(
      ir::semantic::TensorId{0}, {2, 4}, ir::lowered::LayoutKind::row_major,
      base::MemorySpace::device, 16, ir::semantic::DType::f16, ir::semantic::DType::f32,
      ir::semantic::TensorRole::kv_cache);
  assert(kv_tensor.has_value());
  assert(role_builder.add_kernel_requirement("rms_norm", 120, {kv_tensor.value()}).ok());
  const auto role_module = std::move(role_builder).build();
  assert(role_module.has_value());
  const auto role_result = specializer.compile(role_module.value(), {target, 256, 64}, provider);
  assert(role_result.has_value());
  assert(role_result.value().memory.allocations.front().allocation_class ==
         compiler::AllocationClass::kv_state);

  RejectingProvider rejecting_provider;
  const auto provider_rejection = specializer.compile(make_fixture(), {target, 256, 64}, rejecting_provider);
  assert(!provider_rejection.has_value());
  assert(provider_rejection.error().context().back() == "rms_norm");
  assert(provider_rejection.error().message().find("provider fixture rejection") != std::string::npos);
  return 0;
}
