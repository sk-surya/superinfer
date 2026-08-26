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
  std::uint64_t last_kernel_value{0};
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
      if (command.kernel.value() == 0) {
        return base::Status::failed_precondition("physical command has no resolved kernel");
      }
      binding.resolved_kernels_.push_back(command.kernel);
    }
    std::vector<bool> emitted(plan.commands().size(), false);
    binding.command_order_.reserve(plan.commands().size());
    for (std::size_t rank = 0; rank < plan.commands().size(); ++rank) {
      bool found = false;
      for (std::size_t index = 0; index < plan.commands().size(); ++index) {
        if (emitted[index]) continue;
        bool dependencies_emitted = true;
        for (const base::StrongId<ir::physical::CommandIdTag>& dependency :
             plan.commands()[index].dependencies) {
          if (!emitted[dependency.value()]) {
            dependencies_emitted = false;
            break;
          }
        }
        if (!dependencies_emitted) continue;
        emitted[index] = true;
        binding.command_order_.push_back(index);
        found = true;
        break;
      }
      if (!found) return base::Status::failed_precondition("physical command schedule is not executable");
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
    for (const std::size_t command_index : command_order_) {
      const base::KernelId kernel = resolved_kernels_[command_index];
      if (kernel.value() == 0) {
        poisoned_ = true;
        return base::Status::failed_precondition("runtime command has no resolved kernel");
      }
      trace_.last_kernel_value = kernel.value();
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
  std::vector<std::size_t> command_order_;
  ExecutionTrace trace_{};
  bool poisoned_{false};
};

}  // namespace superinfer::runtime
