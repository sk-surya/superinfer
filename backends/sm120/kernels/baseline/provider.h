#pragma once

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
    // Keep this registry synchronized with the CUDA resolver. An advertised candidate must be
    // executable; operations without a physical baseline remain explicit selector misses until
    // their command contract and differential fixture exist.
    if (query.operation == "copy") {
      return std::vector<kernels::KernelCandidate>{{base::KernelId{1}, "sm120.baseline", true, 0}};
    }
    if (query.operation == "residual") {
      return std::vector<kernels::KernelCandidate>{{base::KernelId{4}, "sm120.baseline", true, 0}};
    }
    if (query.operation == "rms_norm") {
      return std::vector<kernels::KernelCandidate>{{base::KernelId{5}, "sm120.baseline", true, 0}};
    }
    if (query.operation == "layer_norm") {
      return std::vector<kernels::KernelCandidate>{{base::KernelId{6}, "sm120.baseline", true, 0}};
    }
    return base::Status::unsupported("no executable baseline candidate for operation");
  }
};

}  // namespace superinfer::sm120
