#pragma once

#include <array>
#include <cstdint>
#include <string_view>
#include <vector>

#include <superinfer/kernels/kernel_provider.hpp>

namespace superinfer::sm120 {

/**
 * Correctness-first capability registry for the synthetic S02 operation set.
 *
 * The provider owns no device state and only returns immutable candidate metadata. Queries are
 * operation/target based; model identity is neither accepted nor inspected.
 */
class BaselineProvider final : public kernels::KernelProvider {
 public:
  base::Result<std::vector<kernels::KernelCandidate>> enumerate(
      const kernels::KernelQuery& query) const override {
    if (query.target_capability != 120) {
      return base::Status::unsupported("baseline provider requires sm_120a");
    }
    constexpr std::array<std::string_view, 13> operations{
        "copy", "cast", "elementwise", "residual", "rms_norm", "layer_norm", "rope",
        "matmul", "embedding", "attention", "moe_route", "activation", "sampling"};
    for (std::size_t index = 0; index < operations.size(); ++index) {
      if (query.operation == operations[index]) {
        return std::vector<kernels::KernelCandidate>{{base::KernelId{index + 1},
                                                       "sm120.baseline", true, 0}};
      }
    }
    return base::Status::unsupported("no baseline candidate for operation");
  }
};

}  // namespace superinfer::sm120
