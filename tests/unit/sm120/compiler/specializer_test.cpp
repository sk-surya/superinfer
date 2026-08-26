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
      base::MemorySpace::device, 16, ir::semantic::DType::f16, ir::semantic::DType::f32);
  assert(hidden.has_value() && output.has_value() && scale.has_value());
  assert(builder
             .add_kernel_requirement("rms_norm", 120,
                                     {hidden.value(), output.value(), scale.value()})
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
  assert(result.value().memory.device_arena_bytes == 48);
  assert(result.value().memory.allocations.size() == 3);
  assert(result.value().memory.allocations.front().offset == 0);
  assert(result.value().plan.commands().size() == 1);
  assert(result.value().plan.commands().front().kernel.value() == 5);
  assert(result.value().plan.commands().front().buffers.size() == 3);

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
