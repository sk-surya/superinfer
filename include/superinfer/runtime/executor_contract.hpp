#pragma once

#include <cstdint>

#include <superinfer/ir/physical_plan.hpp>

namespace superinfer::runtime {

/**
 * Minimal runtime-facing plan binding.
 *
 * The caller owns the validated plan and must keep it alive for this non-owning contract. This
 * type has no model/frontend dependency and performs no allocation or dispatch by itself.
 */
class ExecutorContract final {
 public:
  explicit ExecutorContract(const ir::physical::Plan& plan) noexcept : plan_(&plan) {}
  [[nodiscard]] std::uint32_t plan_version() const noexcept { return plan_->version(); }

 private:
  const ir::physical::Plan* plan_;
};

}  // namespace superinfer::runtime

