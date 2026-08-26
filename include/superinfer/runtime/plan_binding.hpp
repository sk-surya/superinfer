#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include <superinfer/base/result.hpp>
#include <superinfer/ir/physical_plan.hpp>

namespace superinfer::runtime {

/** Bounded execution counters retained outside the token hot path for diagnostics. */
struct ExecutionTrace final {
  std::uint64_t commands_executed{0};
  std::uint64_t entries_executed{0};
};

/**
 * Owns all host-side bindings derived from one validated Physical Plan.
 *
 * Construction copies the immutable plan and resolves its stable command IDs. The object is
 * move-only, has explicit poisoned-session behavior, and performs no allocation during execute().
 * This CPU-safe shell deliberately does not claim CUDA execution when the toolkit is unavailable.
 */
class PlanBinding final {
 public:
  static base::Result<PlanBinding> create(const ir::physical::Plan& plan,
                                          std::uint32_t compute_capability,
                                          std::string_view kernel_catalog) {
    base::Status plan_status = plan.verify();
    if (!plan_status.ok()) return plan_status.with_context("runtime physical plan");
    if (plan.capability().target_capability != compute_capability ||
        plan.capability().kernel_catalog != kernel_catalog) {
      return base::Status::unsupported("physical plan capability does not match runtime target");
    }
    PlanBinding binding{plan};
    binding.resolved_kernels_.reserve(plan.commands().size());
    for (const ir::physical::CommandDescriptor& command : plan.commands()) {
      binding.resolved_kernels_.push_back(command.kernel);
    }
    return binding;
  }

  PlanBinding(PlanBinding&&) noexcept = default;
  PlanBinding& operator=(PlanBinding&&) noexcept = default;
  PlanBinding(const PlanBinding&) = delete;
  PlanBinding& operator=(const PlanBinding&) = delete;

  /** Executes the prebound command sequence; no graph/config/provider inspection occurs here. */
  base::Status execute() noexcept {
    if (poisoned_) return base::Status::failed_precondition("runtime session is poisoned");
    for (const base::KernelId kernel : resolved_kernels_) {
      if (kernel.value() == 0) {
        poisoned_ = true;
        return base::Status::failed_precondition("runtime command has no resolved kernel");
      }
      ++trace_.commands_executed;
    }
    ++trace_.entries_executed;
    return {};
  }

  [[nodiscard]] const ir::physical::Plan& plan() const noexcept { return plan_; }
  [[nodiscard]] const ExecutionTrace& trace() const noexcept { return trace_; }
  [[nodiscard]] bool poisoned() const noexcept { return poisoned_; }

 private:
  explicit PlanBinding(const ir::physical::Plan& plan) : plan_(plan) {}

  ir::physical::Plan plan_;
  std::vector<base::KernelId> resolved_kernels_;
  ExecutionTrace trace_{};
  bool poisoned_{false};
};

}  // namespace superinfer::runtime
